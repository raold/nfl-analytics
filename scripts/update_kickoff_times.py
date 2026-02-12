#!/usr/bin/env python3
"""
Update kickoff times from ESPN API for future games.

This script fetches real kickoff times from the ESPN API and updates
the database for games that have placeholder midnight kickoff times.
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import psycopg
import requests

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5544")),
    "dbname": os.getenv("DB_NAME", "devdb01"),
    "user": os.getenv("DB_USER", "dro"),
    "password": os.getenv("DB_PASSWORD", "sicillionbillions"),
}

# Team abbreviation mapping (ESPN to our database)
ESPN_TO_DB_TEAMS = {"LAR": "LAR", "LA": "LAR", "LAC": "LAC", "WSH": "WAS", "WAS": "WAS"}


def fetch_espn_kickoffs(season: int, week: int):
    """Fetch kickoff times from ESPN API for a specific week."""
    logger.info(f"Fetching kickoff times from ESPN for {season} Week {week}")

    # Calculate date range for the week
    # NFL weeks typically start on Thursday
    base_date = datetime(season, 1, 1)
    # Week 1 typically starts in early September
    week_start = base_date + timedelta(days=(week - 1) * 7 + 245)  # Approximate

    kickoff_times = {}

    # Fetch multiple days to cover Thursday-Monday
    for day_offset in range(-3, 4):
        check_date = week_start + timedelta(days=day_offset)
        date_str = check_date.strftime("%Y%m%d")

        try:
            url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
            params = {"dates": date_str, "limit": 100}

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            for event in data.get("events", []):
                comp = event["competitions"][0]

                # Get teams
                home_team = None
                away_team = None
                for competitor in comp["competitors"]:
                    team_abbr = competitor["team"]["abbreviation"]
                    team_abbr = ESPN_TO_DB_TEAMS.get(team_abbr, team_abbr)

                    if competitor["homeAway"] == "home":
                        home_team = team_abbr
                    else:
                        away_team = team_abbr

                # Get kickoff time
                kickoff_str = event.get("date")
                if kickoff_str and home_team and away_team:
                    # Parse ISO format timestamp
                    kickoff_dt = datetime.fromisoformat(kickoff_str.replace("Z", "+00:00"))

                    # Create game_id in our format
                    game_id = f"{season}_{week:02d}_{away_team}_{home_team}"
                    kickoff_times[game_id] = kickoff_dt

        except Exception as e:
            logger.warning(f"Error fetching data for {date_str}: {e}")
            continue

    logger.info(f"Found {len(kickoff_times)} games with kickoff times")
    return kickoff_times


def update_database_kickoffs(kickoff_times: dict):
    """Update kickoff times in the database."""
    logger.info(f"Updating {len(kickoff_times)} kickoff times in database")

    updated = 0
    not_found = 0

    with psycopg.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cursor:
            for game_id, kickoff_dt in kickoff_times.items():
                # Update the kickoff time
                cursor.execute(
                    """
                    UPDATE games
                    SET kickoff = %s
                    WHERE game_id = %s
                    """,
                    (kickoff_dt, game_id),
                )

                if cursor.rowcount > 0:
                    updated += 1
                    logger.info(f"Updated {game_id}: {kickoff_dt}")
                else:
                    not_found += 1
                    logger.warning(f"Game not found in database: {game_id}")

        conn.commit()

    logger.info(f"Update complete: {updated} updated, {not_found} not found")
    return updated, not_found


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Update kickoff times from ESPN API")
    parser.add_argument("--season", type=int, default=2025, help="NFL season year")
    parser.add_argument("--week", type=int, default=12, help="NFL week number")
    parser.add_argument("--all-future", action="store_true", help="Update all future weeks")

    args = parser.parse_args()

    if args.all_future:
        # Update weeks 12-18 (rest of 2025 season)
        for week in range(12, 19):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {args.season} Week {week}")
            logger.info(f"{'='*60}\n")

            kickoff_times = fetch_espn_kickoffs(args.season, week)
            if kickoff_times:
                update_database_kickoffs(kickoff_times)
    else:
        # Update single week
        kickoff_times = fetch_espn_kickoffs(args.season, args.week)
        if kickoff_times:
            update_database_kickoffs(kickoff_times)


if __name__ == "__main__":
    main()
