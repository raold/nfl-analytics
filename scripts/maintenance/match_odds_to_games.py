#!/usr/bin/env python3
"""
Match odds events to games using flexible matching logic.

This script handles timezone issues and date mismatches between
the games table and odds_history table.
"""

import os
import sys
from datetime import datetime, timedelta
import psycopg
from typing import Dict, List, Tuple

# Database connection parameters
DB_PARAMS = {
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "port": os.environ.get("POSTGRES_PORT", "5544"),
    "dbname": os.environ.get("POSTGRES_DB", "devdb01"),
    "user": os.environ.get("POSTGRES_USER", "dro"),
    "password": os.environ.get("POSTGRES_PASSWORD", "sicillionbillions"),
}


def get_connection():
    """Create database connection."""
    return psycopg.connect(**DB_PARAMS)


def load_team_mappings(conn) -> Dict[str, str]:
    """Load team abbreviation to full name mappings."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT abbreviation, full_name, odds_api_name
            FROM team_mappings
        """)
        mappings = {}
        for abbr, full_name, odds_name in cur:
            mappings[abbr] = odds_name or full_name
        return mappings


def load_games_without_odds(conn) -> List[Dict]:
    """Load all games that don't have odds_api_event_id."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT game_id, season, week, home_team, away_team, kickoff
            FROM games
            WHERE odds_api_event_id IS NULL
            AND season >= 2023
            ORDER BY kickoff
        """)
        games = []
        for row in cur:
            games.append({
                "game_id": row[0],
                "season": row[1],
                "week": row[2],
                "home_team": row[3],
                "away_team": row[4],
                "kickoff": row[5],
            })
        return games


def load_odds_events(conn) -> List[Dict]:
    """Load unique odds events."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT
                event_id,
                home_team,
                away_team,
                MIN(commence_time) as commence_time,
                COUNT(DISTINCT snapshot_at) as snapshot_count
            FROM odds_history
            GROUP BY event_id, home_team, away_team
        """)
        events = []
        for row in cur:
            events.append({
                "event_id": row[0],
                "home_team": row[1],
                "away_team": row[2],
                "commence_time": row[3],
                "snapshot_count": row[4],
            })
        return events


def match_game_to_event(
    game: Dict,
    events: List[Dict],
    team_mappings: Dict[str, str],
    date_tolerance_days: int = 2
) -> Tuple[str, float, str]:
    """
    Match a game to an odds event.

    Returns: (event_id, confidence, match_type)
    """
    # Get full team names
    home_full = team_mappings.get(game["home_team"], game["home_team"])
    away_full = team_mappings.get(game["away_team"], game["away_team"])
    game_date = game["kickoff"]

    best_match = None
    best_confidence = 0
    best_type = None

    for event in events:
        confidence = 0
        match_type = []

        # Check team names
        if event["home_team"] == home_full and event["away_team"] == away_full:
            confidence += 0.5
            match_type.append("exact_teams")
        elif (home_full.lower() in event["home_team"].lower() and
              away_full.lower() in event["away_team"].lower()):
            confidence += 0.3
            match_type.append("partial_teams")
        elif (event["home_team"].lower() in home_full.lower() and
              event["away_team"].lower() in away_full.lower()):
            confidence += 0.3
            match_type.append("partial_teams_reverse")
        else:
            continue  # No team match

        # Check dates (with tolerance)
        if event["commence_time"]:
            date_diff = abs((event["commence_time"] - game_date).days)
            if date_diff == 0:
                confidence += 0.5
                match_type.append("same_date")
            elif date_diff <= 1:
                confidence += 0.3
                match_type.append("date_1day")
            elif date_diff <= date_tolerance_days:
                confidence += 0.1
                match_type.append(f"date_{date_diff}days")

        if confidence > best_confidence:
            best_confidence = confidence
            best_match = event["event_id"]
            best_type = "+".join(match_type)

    return best_match, best_confidence, best_type


def update_games_with_event_ids(conn, matches: List[Tuple[str, str, float, str]]):
    """Update games table with matched event IDs."""
    with conn.cursor() as cur:
        for game_id, event_id, confidence, match_type in matches:
            if confidence >= 0.8:  # Only high-confidence matches
                cur.execute("""
                    UPDATE games
                    SET
                        odds_api_event_id = %s,
                        updated_at = NOW()
                    WHERE game_id = %s
                """, (event_id, game_id))

        conn.commit()
        print(f"Updated {cur.rowcount} games with event IDs")


def main():
    """Main matching process."""
    print("=" * 60)
    print("Matching Odds Events to Games")
    print("=" * 60)

    conn = get_connection()

    try:
        # Load data
        print("\nLoading data...")
        team_mappings = load_team_mappings(conn)
        games = load_games_without_odds(conn)
        events = load_odds_events(conn)

        print(f"  Found {len(games)} games without odds")
        print(f"  Found {len(events)} unique odds events")
        print(f"  Found {len(team_mappings)} team mappings")

        # Match games to events
        print("\nMatching games to events...")
        matches = []
        confidence_buckets = {
            "perfect": [],
            "high": [],
            "medium": [],
            "low": [],
            "unmatched": []
        }

        for game in games:
            event_id, confidence, match_type = match_game_to_event(
                game, events, team_mappings
            )

            if event_id:
                matches.append((game["game_id"], event_id, confidence, match_type))

                if confidence >= 1.0:
                    confidence_buckets["perfect"].append(game["game_id"])
                elif confidence >= 0.8:
                    confidence_buckets["high"].append(game["game_id"])
                elif confidence >= 0.6:
                    confidence_buckets["medium"].append(game["game_id"])
                else:
                    confidence_buckets["low"].append(game["game_id"])
            else:
                confidence_buckets["unmatched"].append(game["game_id"])

        # Print statistics
        print("\nMatch Statistics:")
        print(f"  Perfect matches (1.0): {len(confidence_buckets['perfect'])}")
        print(f"  High confidence (0.8+): {len(confidence_buckets['high'])}")
        print(f"  Medium confidence (0.6+): {len(confidence_buckets['medium'])}")
        print(f"  Low confidence (<0.6): {len(confidence_buckets['low'])}")
        print(f"  Unmatched: {len(confidence_buckets['unmatched'])}")

        # Show sample matches
        if matches:
            print("\nSample matches (first 10):")
            for game_id, event_id, conf, match_type in matches[:10]:
                print(f"  {game_id} -> {event_id[:8]}... (conf: {conf:.2f}, type: {match_type})")

        # Ask for confirmation (auto-approve if --auto flag or non-interactive)
        if matches:
            high_conf_matches = [m for m in matches if m[2] >= 0.8]
            print(f"\nReady to update {len(high_conf_matches)} high-confidence matches.")

            # Check if we're in interactive mode
            import sys
            if sys.stdin.isatty() and '--auto' not in sys.argv:
                response = input("Proceed? (y/n): ")
                proceed = response.lower() == 'y'
            else:
                print("Auto-approving updates (non-interactive mode)")
                proceed = True

            if proceed:
                update_games_with_event_ids(conn, high_conf_matches)

                # Update snapshot counts
                print("\nUpdating snapshot counts...")
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE games g
                        SET odds_snapshots_count = (
                            SELECT COUNT(DISTINCT snapshot_at)
                            FROM odds_history o
                            WHERE o.event_id = g.odds_api_event_id
                        )
                        WHERE g.odds_api_event_id IS NOT NULL
                    """)
                    conn.commit()
                    print(f"Updated snapshot counts for {cur.rowcount} games")
            else:
                print("Update cancelled.")

        # Show coverage statistics
        print("\nFinal Coverage:")
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE odds_api_event_id IS NOT NULL) as matched,
                    COUNT(*) as total,
                    ROUND(100.0 * COUNT(*) FILTER (WHERE odds_api_event_id IS NOT NULL) / COUNT(*), 2) as pct
                FROM games
                WHERE season >= 2023
            """)
            matched, total, pct = cur.fetchone()
            print(f"  {matched} of {total} games have odds ({pct}%)")

            # Show by season
            cur.execute("""
                SELECT
                    season,
                    COUNT(*) FILTER (WHERE odds_api_event_id IS NOT NULL) as matched,
                    COUNT(*) as total
                FROM games
                WHERE season >= 2023
                GROUP BY season
                ORDER BY season
            """)
            print("\n  By season:")
            for season, matched, total in cur:
                print(f"    {season}: {matched}/{total} ({100*matched/total:.1f}%)")

    finally:
        conn.close()

    print("\n" + "=" * 60)
    print("Matching complete!")


if __name__ == "__main__":
    main()