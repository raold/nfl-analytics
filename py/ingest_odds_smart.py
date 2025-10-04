#!/usr/bin/env python3
"""
Smart odds fetching script that minimizes API usage.

Features:
- Checks what's already fetched before making API calls
- Prioritizes games by proximity to kickoff
- Tracks API usage against monthly quota
- Only fetches what's needed based on game timing
"""

import argparse
import datetime as dt
import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import psycopg
import requests
from dotenv import load_dotenv

# API configuration
API_BASE = "https://api.the-odds-api.com/v4"
DEFAULT_SPORT_KEY = "americanfootball_nfl"
DEFAULT_REGIONS = "us"
DEFAULT_MARKETS = "h2h,spreads,totals"


class FetchPriority(Enum):
    """Priority levels for fetching odds."""

    URGENT = 10  # Game within 24 hours
    HIGH = 8  # Game within 3 days
    MEDIUM = 5  # Game within a week
    LOW = 3  # Game more than a week out
    SKIP = 0  # Already played or sufficient data


@dataclass
class GameOddsStatus:
    """Status of odds coverage for a game."""

    game_id: str
    event_id: str | None
    kickoff: dt.datetime
    last_fetched: dt.datetime | None
    snapshot_count: int
    hours_until_game: float
    hours_since_fetch: float
    priority: FetchPriority
    should_fetch: bool


def get_connection() -> psycopg.Connection:
    """Create database connection."""
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5544")
    dbname = os.environ.get("POSTGRES_DB", "devdb01")
    user = os.environ.get("POSTGRES_USER", "dro")
    password = os.environ.get("POSTGRES_PASSWORD", "sicillionbillions")
    return psycopg.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        autocommit=False,
    )


def check_api_quota(conn: psycopg.Connection) -> tuple[int, int]:
    """Check current API usage against monthly quota."""
    current_month = dt.date.today().replace(day=1)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT calls_made, quota_limit
            FROM api_usage_tracker
            WHERE api_name = 'the-odds-api'
            AND month = %s
        """,
            (current_month,),
        )

        result = cur.fetchone()
        if result:
            return result[0] or 0, result[1] or 500
        else:
            # Initialize tracker for this month
            cur.execute(
                """
                INSERT INTO api_usage_tracker (month, api_name, quota_limit, calls_made)
                VALUES (%s, 'the-odds-api', 500, 0)
                ON CONFLICT (month, api_name) DO NOTHING
            """,
                (current_month,),
            )
            conn.commit()
            return 0, 500


def update_api_usage(conn: psycopg.Connection, calls: int = 1):
    """Update API usage tracker."""
    current_month = dt.date.today().replace(day=1)

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE api_usage_tracker
            SET
                calls_made = calls_made + %s,
                last_call_at = NOW(),
                updated_at = NOW()
            WHERE api_name = 'the-odds-api'
            AND month = %s
        """,
            (calls, current_month),
        )
        conn.commit()


def calculate_priority(
    kickoff: dt.datetime, last_fetched: dt.datetime | None, snapshot_count: int
) -> tuple[FetchPriority, bool]:
    """
    Calculate fetch priority for a game.

    Returns: (priority, should_fetch)
    """
    now = dt.datetime.now(dt.UTC)
    hours_until = (kickoff - now).total_seconds() / 3600

    # Game already played
    if hours_until < -3:  # 3 hours after kickoff
        return FetchPriority.SKIP, False

    hours_since_fetch = float("inf")
    if last_fetched:
        hours_since_fetch = (now - last_fetched).total_seconds() / 3600

    # Determine priority and fetch frequency
    if hours_until < 24:  # Within 24 hours
        priority = FetchPriority.URGENT
        # Fetch every hour if we have less than 24 snapshots
        should_fetch = hours_since_fetch > 1 or snapshot_count < 24
    elif hours_until < 72:  # Within 3 days
        priority = FetchPriority.HIGH
        # Fetch every 6 hours
        should_fetch = hours_since_fetch > 6 or snapshot_count < 12
    elif hours_until < 168:  # Within a week
        priority = FetchPriority.MEDIUM
        # Fetch daily
        should_fetch = hours_since_fetch > 24 or snapshot_count < 7
    else:  # More than a week out
        priority = FetchPriority.LOW
        # Fetch weekly
        should_fetch = hours_since_fetch > 168 or snapshot_count < 2

    return priority, should_fetch


def get_games_needing_odds(conn: psycopg.Connection) -> list[GameOddsStatus]:
    """Get list of games and their odds coverage status."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                g.game_id,
                g.odds_api_event_id,
                g.kickoff,
                g.odds_last_fetched,
                COALESCE(g.odds_snapshots_count, 0) as snapshot_count
            FROM games g
            WHERE g.season >= 2024  -- Focus on current/recent seasons
            AND g.kickoff > NOW() - INTERVAL '7 days'  -- Recent or upcoming games
            ORDER BY g.kickoff
        """
        )

        games = []
        now = dt.datetime.now(dt.UTC)

        for row in cur:
            game_id, event_id, kickoff, last_fetched, snapshot_count = row

            # Make timezone aware if needed
            if kickoff and not kickoff.tzinfo:
                kickoff = kickoff.replace(tzinfo=dt.UTC)
            if last_fetched and not last_fetched.tzinfo:
                last_fetched = last_fetched.replace(tzinfo=dt.UTC)

            hours_until = (kickoff - now).total_seconds() / 3600 if kickoff else float("inf")
            hours_since = (
                (now - last_fetched).total_seconds() / 3600 if last_fetched else float("inf")
            )

            priority, should_fetch = calculate_priority(kickoff, last_fetched, snapshot_count)

            games.append(
                GameOddsStatus(
                    game_id=game_id,
                    event_id=event_id,
                    kickoff=kickoff,
                    last_fetched=last_fetched,
                    snapshot_count=snapshot_count,
                    hours_until_game=hours_until,
                    hours_since_fetch=hours_since,
                    priority=priority,
                    should_fetch=should_fetch,
                )
            )

        return games


def fetch_odds_for_date(
    api_key: str,
    date: dt.date,
    sport_key: str = DEFAULT_SPORT_KEY,
    regions: str = DEFAULT_REGIONS,
    markets: str = DEFAULT_MARKETS,
) -> list[dict[str, Any]]:
    """Fetch odds for a specific date."""
    url = f"{API_BASE}/historical/sports/{sport_key}/odds"
    snapshot_at = dt.datetime.combine(date, dt.time(0, 0, tzinfo=dt.UTC))

    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "date": snapshot_at.isoformat().replace("+00:00", "Z"),
    }

    response = requests.get(url, params=params, timeout=30)

    if response.status_code == 429:
        reset = response.headers.get("x-requests-reset")
        raise RuntimeError(f"Hit API rate limit. Reset at UTC epoch {reset or 'unknown'}.")

    response.raise_for_status()

    # Handle different response formats
    json_data = response.json()
    if isinstance(json_data, dict) and "data" in json_data:
        return json_data["data"]
    elif isinstance(json_data, list):
        return json_data
    else:
        return []


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Smart odds fetching with minimal API usage")
    parser.add_argument(
        "--max-calls",
        type=int,
        default=10,
        help="Maximum API calls to make in this run (default: 10)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be fetched without making API calls"
    )
    parser.add_argument("--force", action="store_true", help="Force fetch even if recently updated")
    parser.add_argument("--date", help="Specific date to fetch (YYYY-MM-DD)")
    parser.add_argument(
        "--priority",
        choices=["urgent", "high", "medium", "low", "all"],
        default="high",
        help="Minimum priority level to fetch (default: high)",
    )
    return parser.parse_args()


def main():
    """Main smart fetching process."""
    load_dotenv()
    args = parse_args()

    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("Error: Missing ODDS_API_KEY environment variable", file=sys.stderr)
        sys.exit(1)

    conn = get_connection()

    try:
        # Check API quota
        calls_made, quota_limit = check_api_quota(conn)
        remaining = quota_limit - calls_made

        print("=" * 60)
        print("Smart Odds Fetching")
        print("=" * 60)
        print(f"\nAPI Quota: {calls_made}/{quota_limit} used ({remaining} remaining)")

        if remaining <= 0:
            print("ERROR: Monthly API quota exhausted!")
            sys.exit(1)

        if remaining < args.max_calls:
            print(f"Warning: Only {remaining} API calls remaining this month")
            args.max_calls = min(args.max_calls, remaining)

        # Get games needing odds
        games = get_games_needing_odds(conn)

        # Filter by priority
        priority_map = {
            "urgent": FetchPriority.URGENT,
            "high": FetchPriority.HIGH,
            "medium": FetchPriority.MEDIUM,
            "low": FetchPriority.LOW,
        }

        if args.priority != "all":
            min_priority = priority_map[args.priority]
            games = [g for g in games if g.priority.value >= min_priority.value]

        # Filter to games that need fetching
        if not args.force:
            games_to_fetch = [g for g in games if g.should_fetch]
        else:
            games_to_fetch = games

        print("\nGames Analysis:")
        print(f"  Total games checked: {len(games)}")
        print(f"  Games needing update: {len(games_to_fetch)}")

        if not games_to_fetch:
            print("\nNo games need odds updates at this time.")
            return

        # Sort by priority (highest first)
        games_to_fetch.sort(key=lambda g: (-g.priority.value, g.hours_until_game))

        print(f"\nTop {min(10, len(games_to_fetch))} games to fetch:")
        for game in games_to_fetch[:10]:
            status = "NO EVENT ID" if not game.event_id else f"{game.snapshot_count} snapshots"
            print(
                f"  {game.game_id}: {game.priority.name} "
                f"(kickoff in {game.hours_until_game:.1f}h, {status})"
            )

        if args.dry_run:
            print("\nDRY RUN - No API calls made")
            return

        # Determine dates to fetch
        dates_to_fetch = set()
        for game in games_to_fetch[: args.max_calls]:
            if game.kickoff:
                # Fetch for game date
                dates_to_fetch.add(game.kickoff.date())

        print(f"\nFetching odds for {len(dates_to_fetch)} dates...")

        # Fetch odds
        api_calls = 0
        total_events = 0

        for date in sorted(dates_to_fetch):
            if api_calls >= args.max_calls:
                print(f"Reached max calls limit ({args.max_calls})")
                break

            print(f"\n  Fetching {date}...")
            try:
                events = fetch_odds_for_date(api_key, date)
                api_calls += 1
                total_events += len(events)
                print(f"    Found {len(events)} events")

                # Process and store events (implement storage logic here)
                # This would involve calling the flatten_events and upsert_rows
                # functions from the original ingest_odds_history.py

                # Update API usage
                update_api_usage(conn, 1)

                # Update games last fetched timestamp
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE games
                        SET odds_last_fetched = NOW()
                        WHERE DATE(kickoff) = %s
                    """,
                        (date,),
                    )
                    conn.commit()

                time.sleep(1)  # Rate limiting

            except Exception as e:
                print(f"    Error: {e}")
                continue

        print("\n" + "=" * 60)
        print("Summary:")
        print(f"  API calls made: {api_calls}")
        print(f"  Total events fetched: {total_events}")
        print(f"  Quota remaining: {remaining - api_calls}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
