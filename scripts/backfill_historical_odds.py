#!/usr/bin/env python3
# PROVENANCE ONLY; DO NOT RUN. This is the script that filled public.odds_history on
# 2025-12-07/08, kept to document how those rows were made. Its "closing" and "opening"
# requests are mistimed game-day snapshots, not closing and opening lines, and its matcher
# drops games: it maps the Rams to LAR where games uses LA, and compares the API's UTC date
# with the stored kickoff date (docs/POSTMORTEMS.md section 7). A run also spends paid Odds
# API quota.
"""
Backfill historical odds from The Odds API.

OPTIMIZED Strategy:
- Default: Closing lines only (~1,900 calls for 7 seasons)
- Optional: Both opening + closing (~3,800 calls)
- Resume capability: Skips dates already fetched
- Smart timing: Uses actual kickoff times for accurate closing lines

API Efficiency:
- Each call returns ALL games for that snapshot (not per-game)
- Closing lines only = 50% API savings
- Resume = skip already-fetched dates
"""

import argparse
import datetime as dt
import os
import sys
import time
from typing import Any

import psycopg
import requests
from dotenv import load_dotenv

API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "americanfootball_nfl"

# Track API usage
api_calls_made = 0
total_events_fetched = 0


def get_connection() -> psycopg.Connection:
    """Create database connection."""
    return psycopg.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5544"),
        dbname=os.environ.get("POSTGRES_DB", "devdb01"),
        user=os.environ.get("POSTGRES_USER", "dro"),
        password=os.environ.get("POSTGRES_PASSWORD", "sicillionbillions"),
    )


def fetch_historical_odds(api_key: str, date: dt.datetime) -> tuple[list[dict], dict]:
    """Fetch historical odds for a specific datetime."""
    global api_calls_made

    url = f"{API_BASE}/historical/sports/{SPORT_KEY}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "spreads,totals,h2h",
        "date": date.isoformat().replace("+00:00", "Z"),
    }

    response = requests.get(url, params=params, timeout=30)
    api_calls_made += 1

    if response.status_code == 429:
        raise RuntimeError("Rate limited!")

    response.raise_for_status()

    # Get quota info from headers
    quota_info = {
        "remaining": response.headers.get("x-requests-remaining", "?"),
        "used": response.headers.get("x-requests-used", "?"),
    }

    data = response.json()
    if isinstance(data, dict) and "data" in data:
        return data["data"], quota_info
    return data if isinstance(data, list) else [], quota_info


def get_nfl_game_dates(conn: psycopg.Connection, season: int) -> list[dt.date]:
    """Get all unique game dates for a season."""
    query = """
        SELECT DISTINCT DATE(kickoff) as game_date
        FROM games
        WHERE season = %s AND kickoff IS NOT NULL
        ORDER BY game_date
    """
    with conn.cursor() as cur:
        cur.execute(query, (season,))
        return [row[0] for row in cur.fetchall()]


def get_earliest_kickoff(conn: psycopg.Connection, game_date: dt.date) -> dt.datetime | None:
    """Get earliest kickoff time for a game date (for smarter closing line timing)."""
    query = """
        SELECT MIN(kickoff) FROM games
        WHERE DATE(kickoff) = %s AND kickoff IS NOT NULL
    """
    with conn.cursor() as cur:
        cur.execute(query, (game_date,))
        row = cur.fetchone()
        if row and row[0]:
            kickoff = row[0]
            if not kickoff.tzinfo:
                kickoff = kickoff.replace(tzinfo=dt.UTC)
            return kickoff
        return None


def get_fetched_dates(conn: psycopg.Connection, market_type: str) -> set[dt.date]:
    """Get dates that have already been fetched for a given market type."""
    query = """
        SELECT DISTINCT DATE(last_update) as fetch_date
        FROM odds_history
        WHERE market_type = %s AND last_update IS NOT NULL
    """
    with conn.cursor() as cur:
        cur.execute(query, (market_type,))
        return {row[0] for row in cur.fetchall()}


def insert_odds(conn: psycopg.Connection, events: list[dict], snapshot_type: str):
    """Insert odds into database."""
    global total_events_fetched

    insert_sql = """
        INSERT INTO odds_history (
            captured_at, game_id, bookmaker, market_type,
            home_spread, away_spread, home_spread_price, away_spread_price,
            over_under, over_price, under_price,
            home_moneyline, away_moneyline, last_update
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) ON CONFLICT DO NOTHING
    """

    inserted = 0
    with conn.cursor() as cur:
        for event in events:
            event_id = event.get("id")
            home_team = event.get("home_team", "")
            away_team = event.get("away_team", "")
            commence = event.get("commence_time", "")

            # Try to match to game_id
            game_id = find_game_id(conn, home_team, away_team, commence)
            if not game_id:
                continue

            for bookmaker in event.get("bookmakers", []):
                book_key = bookmaker.get("key", "")
                last_update = bookmaker.get("last_update")

                spreads = {}
                totals = {}
                h2h = {}

                for market in bookmaker.get("markets", []):
                    if market["key"] == "spreads":
                        for outcome in market.get("outcomes", []):
                            if outcome["name"] == home_team:
                                spreads["home_spread"] = outcome.get("point")
                                spreads["home_spread_price"] = int(american_odds(outcome.get("price", 0)))
                            else:
                                spreads["away_spread"] = outcome.get("point")
                                spreads["away_spread_price"] = int(american_odds(outcome.get("price", 0)))

                    elif market["key"] == "totals":
                        for outcome in market.get("outcomes", []):
                            if outcome["name"] == "Over":
                                totals["over_under"] = outcome.get("point")
                                totals["over_price"] = int(american_odds(outcome.get("price", 0)))
                            else:
                                totals["under_price"] = int(american_odds(outcome.get("price", 0)))

                    elif market["key"] == "h2h":
                        for outcome in market.get("outcomes", []):
                            if outcome["name"] == home_team:
                                h2h["home_moneyline"] = int(american_odds(outcome.get("price", 0)))
                            else:
                                h2h["away_moneyline"] = int(american_odds(outcome.get("price", 0)))

                cur.execute(insert_sql, (
                    dt.datetime.now(dt.UTC),
                    game_id,
                    book_key,
                    snapshot_type,
                    spreads.get("home_spread"),
                    spreads.get("away_spread"),
                    spreads.get("home_spread_price"),
                    spreads.get("away_spread_price"),
                    totals.get("over_under"),
                    totals.get("over_price"),
                    totals.get("under_price"),
                    h2h.get("home_moneyline"),
                    h2h.get("away_moneyline"),
                    last_update,
                ))
                inserted += 1

    conn.commit()
    total_events_fetched += inserted
    return inserted


def find_game_id(conn: psycopg.Connection, home_team: str, away_team: str, commence: str) -> str | None:
    """Find game_id from teams and date."""
    # Normalize team names
    team_map = {
        "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
        "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
        "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
        "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
        "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
        "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Oakland Raiders": "OAK",
        "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA",
        "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO",
        "New York Giants": "NYG", "New York Jets": "NYJ", "Philadelphia Eagles": "PHI",
        "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA",
        "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
        "Washington Redskins": "WAS", "Washington Football Team": "WAS",
        "St. Louis Rams": "LAR", "San Diego Chargers": "LAC",
    }

    home_abbr = team_map.get(home_team)
    away_abbr = team_map.get(away_team)

    if not home_abbr or not away_abbr:
        return None

    # Parse date
    try:
        game_date = dt.datetime.fromisoformat(commence.replace("Z", "+00:00")).date()
    except (ValueError, AttributeError):
        return None

    query = """
        SELECT game_id FROM games
        WHERE home_team = %s AND away_team = %s
        AND DATE(kickoff) = %s
    """
    with conn.cursor() as cur:
        cur.execute(query, (home_abbr, away_abbr, game_date))
        row = cur.fetchone()
        return row[0] if row else None


def american_odds(decimal_odds: float) -> float:
    """Convert decimal odds to American."""
    if decimal_odds >= 2.0:
        return (decimal_odds - 1) * 100
    elif decimal_odds > 1.0:
        return -100 / (decimal_odds - 1)
    return 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Backfill historical NFL odds from The Odds API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: Closing lines only for 2018-2024 (~1,900 calls)
  python backfill_historical_odds.py

  # Include opening lines too (~3,800 calls)
  python backfill_historical_odds.py --include-opening

  # Specific season only
  python backfill_historical_odds.py --season 2024

  # Dry run to see what would be fetched
  python backfill_historical_odds.py --dry-run

  # Limit API calls
  python backfill_historical_odds.py --max-calls 100
        """
    )
    parser.add_argument(
        "--include-opening",
        action="store_true",
        help="Include opening lines (doubles API calls)"
    )
    parser.add_argument(
        "--season",
        type=int,
        choices=range(2020, 2025),
        help="Specific season to backfill (default: all 2020-2024, API data starts 2020)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fetched without making API calls"
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="Maximum API calls to make (for testing/budgeting)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't skip already-fetched dates"
    )
    return parser.parse_args()


def main():
    load_dotenv(override=True)  # Override shell env vars with .env values
    args = parse_args()

    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("Error: ODDS_API_KEY not set")
        sys.exit(1)

    conn = get_connection()

    # Determine seasons to process
    # Note: The Odds API historical data only goes back to 2020
    seasons = [args.season] if args.season else [2020, 2021, 2022, 2023, 2024]

    # Calculate total game dates
    total_dates = sum(len(get_nfl_game_dates(conn, s)) for s in seasons)
    calls_per_date = 2 if args.include_opening else 1
    estimated_calls = total_dates * calls_per_date

    print("=" * 70)
    print("Historical Odds Backfill - OPTIMIZED")
    print("=" * 70)
    print(f"Seasons: {seasons[0]}-{seasons[-1]} ({len(seasons)} season{'s' if len(seasons) > 1 else ''})")
    print(f"Strategy: {'Opening + Closing' if args.include_opening else 'Closing only (50% API savings)'}")
    print(f"Game dates: {total_dates}")
    print(f"Estimated calls: ~{estimated_calls}")
    if args.max_calls:
        print(f"Max calls limit: {args.max_calls}")
    if args.dry_run:
        print("MODE: DRY RUN (no API calls)")
    print("=" * 70)

    # Check already-fetched dates for resume capability
    already_fetched_closing = set()
    already_fetched_opening = set()
    if not args.no_resume:
        already_fetched_closing = get_fetched_dates(conn, "closing")
        if args.include_opening:
            already_fetched_opening = get_fetched_dates(conn, "opening")
        if already_fetched_closing or already_fetched_opening:
            print(f"\nResume mode: Skipping {len(already_fetched_closing)} closing / {len(already_fetched_opening)} opening already-fetched dates")

    global api_calls_made, total_events_fetched

    for season in seasons:
        print(f"\n{'='*70}")
        print(f"SEASON {season}")
        print(f"{'='*70}")

        game_dates = get_nfl_game_dates(conn, season)
        print(f"Found {len(game_dates)} game dates")

        for i, game_date in enumerate(game_dates):
            # Check max calls limit
            if args.max_calls and api_calls_made >= args.max_calls:
                print(f"\n⚠️  Reached max calls limit ({args.max_calls}). Stopping.")
                break

            # Get earliest kickoff for smart timing
            earliest_kickoff = get_earliest_kickoff(conn, game_date)

            # Closing line: 30 min before earliest kickoff (or noon ET fallback)
            if earliest_kickoff:
                closing_dt = earliest_kickoff - dt.timedelta(minutes=30)
            else:
                closing_dt = dt.datetime.combine(game_date, dt.time(16, 0), tzinfo=dt.UTC)

            # Opening line: 10am ET (14:00 UTC)
            opening_dt = dt.datetime.combine(game_date, dt.time(14, 0), tzinfo=dt.UTC)

            print(f"\n  {game_date} ({i+1}/{len(game_dates)})")

            # Fetch opening (if requested and not already fetched)
            if args.include_opening:
                if game_date in already_fetched_opening and not args.no_resume:
                    print(f"    Opening: SKIPPED (already fetched)")
                elif args.dry_run:
                    print(f"    Opening: WOULD FETCH {opening_dt.isoformat()}")
                else:
                    try:
                        events, quota = fetch_historical_odds(api_key, opening_dt)
                        inserted = insert_odds(conn, events, "opening")
                        print(f"    Opening: {len(events)} events, {inserted} rows | API: {api_calls_made} calls, {quota['remaining']} remaining")
                    except Exception as e:
                        print(f"    Opening ERROR: {e}")
                    time.sleep(0.3)

            # Fetch closing (always, unless already fetched)
            if game_date in already_fetched_closing and not args.no_resume:
                print(f"    Closing: SKIPPED (already fetched)")
            elif args.dry_run:
                print(f"    Closing: WOULD FETCH {closing_dt.isoformat()}")
            else:
                try:
                    events, quota = fetch_historical_odds(api_key, closing_dt)
                    inserted = insert_odds(conn, events, "closing")
                    print(f"    Closing: {len(events)} events, {inserted} rows | API: {api_calls_made} calls, {quota['remaining']} remaining")
                except Exception as e:
                    print(f"    Closing ERROR: {e}")
                time.sleep(0.3)

        # Check max calls after each season
        if args.max_calls and api_calls_made >= args.max_calls:
            break

    print("\n" + "=" * 70)
    print("BACKFILL COMPLETE" if not args.dry_run else "DRY RUN COMPLETE")
    print("=" * 70)
    print(f"Total API calls: {api_calls_made}")
    print(f"Total rows inserted: {total_events_fetched}")

    # Final count
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM odds_history")
        total = cur.fetchone()[0]
        print(f"Total odds_history rows: {total}")

        cur.execute("SELECT market_type, COUNT(*) FROM odds_history GROUP BY market_type")
        for row in cur.fetchall():
            print(f"  - {row[0]}: {row[1]} rows")

    conn.close()


if __name__ == "__main__":
    main()
