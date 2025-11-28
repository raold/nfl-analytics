#!/usr/bin/env python3
"""
Debug script to test live score fetching and matching.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import requests
import pandas as pd
import psycopg
from datetime import datetime

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5544")),
    "dbname": os.getenv("DB_NAME", "devdb01"),
    "user": os.getenv("DB_USER", "dro"),
    "password": os.getenv("DB_PASSWORD", "sicillionbillions"),
}


def fetch_live_scores_espn():
    """Fetch live scores from ESPN API."""
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        params = {"limit": 100}

        espn_to_db_teams = {
            "LAR": "LAR",
            "LA": "LAR",
            "LAC": "LAC",
            "WSH": "WAS",
            "WAS": "WAS",
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        live_scores = {}

        if "events" in data:
            for event in data["events"]:
                try:
                    competitions = event.get("competitions", [])
                    if not competitions:
                        continue

                    competition = competitions[0]
                    competitors = competition.get("competitors", [])

                    home_team = None
                    away_team = None
                    home_score = 0
                    away_score = 0

                    for comp in competitors:
                        team_abbr = comp.get("team", {}).get("abbreviation", "")
                        team_abbr = espn_to_db_teams.get(team_abbr, team_abbr)
                        score = int(comp.get("score", 0))

                        if comp.get("homeAway") == "home":
                            home_team = team_abbr
                            home_score = score
                        elif comp.get("homeAway") == "away":
                            away_team = team_abbr
                            away_score = score

                    status = event.get("status", {})
                    status_type = status.get("type", {}).get("state", "pre")
                    clock = status.get("displayClock", "")
                    period = status.get("period", 0)
                    detail = status.get("type", {}).get("detail", "")

                    if home_team and away_team:
                        game_key = f"{away_team}_{home_team}"

                        live_scores[game_key] = {
                            "away_team": away_team,
                            "home_team": home_team,
                            "away_score": away_score,
                            "home_score": home_score,
                            "status": status_type,
                            "clock": clock,
                            "period": period,
                            "detail": detail,
                        }
                except Exception as e:
                    print(f"Error parsing event: {e}")
                    continue

        return live_scores
    except Exception as e:
        print(f"Failed to fetch ESPN scores: {e}")
        return {}


def fetch_current_week_games_from_db():
    """Fetch current week's games from database."""
    current_season = datetime.now().year

    query = """
    WITH current_week AS (
        SELECT season, week
        FROM games
        WHERE season = %s AND game_type = 'REG'
        ORDER BY ABS(EXTRACT(EPOCH FROM (kickoff - NOW()))) ASC
        LIMIT 1
    )
    SELECT
        g.game_id,
        g.season,
        g.week,
        g.home_team,
        g.away_team,
        g.home_score,
        g.away_score,
        g.kickoff,
        CASE
            WHEN g.home_score IS NOT NULL THEN 'completed'
            WHEN g.kickoff < NOW() - INTERVAL '4 hours' AND g.home_score IS NULL THEN 'completed'
            WHEN g.kickoff < NOW() THEN 'in_progress'
            ELSE 'scheduled'
        END as game_status
    FROM games g
    WHERE g.season = (SELECT season FROM current_week)
      AND g.week = (SELECT week FROM current_week)
      AND g.game_type = 'REG'
    ORDER BY g.kickoff DESC
    LIMIT 20
    """

    with psycopg.connect(**DB_CONFIG) as conn:
        df = pd.read_sql_query(query, conn, params=(current_season,))

    return df


def normalize_team_display(team_abbr):
    """Normalize team abbreviation for display."""
    if team_abbr == "LA":
        return "LAR"
    return team_abbr


def main():
    """Main diagnostic function."""
    print("=" * 80)
    print("LIVE SCORES DIAGNOSTIC")
    print("=" * 80)

    # 1. Fetch ESPN scores
    print("\n1. Fetching ESPN live scores...")
    espn_scores = fetch_live_scores_espn()
    print(f"   Found {len(espn_scores)} games from ESPN")

    if espn_scores:
        print("\n   ESPN Game Keys:")
        for game_key, data in espn_scores.items():
            print(
                f"   - {game_key}: {data['away_team']} {data['away_score']} @ "
                f"{data['home_team']} {data['home_score']} ({data['status']})"
            )

    # 2. Fetch database games
    print("\n2. Fetching current week games from database...")
    df = fetch_current_week_games_from_db()
    print(f"   Found {len(df)} games in database")

    if not df.empty:
        print(f"\n   Current week: Season {df['season'].iloc[0]}, Week {df['week'].iloc[0]}")
        print("\n   Database Games:")
        for idx, row in df.iterrows():
            print(
                f"   - {row['game_id']}: {row['away_team']} @ {row['home_team']} "
                f"(status: {row['game_status']})"
            )

    # 3. Test matching logic
    print("\n3. Testing game key matching...")
    if not df.empty:
        df["away_team_display"] = df["away_team"].apply(normalize_team_display)
        df["home_team_display"] = df["home_team"].apply(normalize_team_display)

        matches = 0
        mismatches = []

        for idx, row in df.iterrows():
            game_key = f"{row['away_team_display']}_{row['home_team_display']}"
            db_status = row["game_status"]

            if game_key in espn_scores:
                espn_data = espn_scores[game_key]
                matches += 1
                print(
                    f"\n   ✅ MATCH: {game_key}"
                    f"\n      DB Status: {db_status}"
                    f"\n      ESPN Status: {espn_data['status']}"
                    f"\n      ESPN Scores: {espn_data['away_team']} {espn_data['away_score']} @ "
                    f"{espn_data['home_team']} {espn_data['home_score']}"
                )
            else:
                mismatches.append((game_key, row["game_id"], db_status))

        if mismatches:
            print(f"\n   ❌ {len(mismatches)} games NOT matched:")
            for game_key, game_id, db_status in mismatches:
                print(f"      - {game_key} (game_id: {game_id}, status: {db_status})")

        print(f"\n   Summary: {matches}/{len(df)} games matched")

    # 4. Check for in_progress games
    print("\n4. Checking for in-progress games...")
    if not df.empty:
        in_progress = df[df["game_status"] == "in_progress"]
        print(f"   Found {len(in_progress)} in-progress games in DB")

        live_games_count = sum(1 for v in espn_scores.values() if v["status"] == "in")
        print(f"   Found {live_games_count} live games on ESPN")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
