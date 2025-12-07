#!/usr/bin/env python3
"""
Sync kickoff times from ESPN API current scoreboard.

This script fetches the current week's schedule from ESPN and updates
our database with the actual kickoff times.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import psycopg
import requests

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5544")),
    "dbname": os.getenv("DB_NAME", "devdb01"),
    "user": os.getenv("DB_USER", "dro"),
    "password": os.getenv("DB_PASSWORD", "sicillionbillions")
}

# Team abbreviation mapping (ESPN to our database)
ESPN_TO_DB_TEAMS = {
    'LAR': 'LAR',
    'LA': 'LAR',
    'LAC': 'LAC',
    'WSH': 'WAS',
    'WAS': 'WAS'
}


def fetch_current_week_kickoffs():
    """Fetch kickoff times from ESPN API current scoreboard."""
    logger.info("Fetching current week schedule from ESPN")

    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        games = []
        for event in data.get('events', []):
            comp = event['competitions'][0]

            # Get teams
            home_team = None
            away_team = None
            for competitor in comp['competitors']:
                team_abbr = competitor['team']['abbreviation']
                team_abbr = ESPN_TO_DB_TEAMS.get(team_abbr, team_abbr)

                if competitor['homeAway'] == 'home':
                    home_team = team_abbr
                else:
                    away_team = team_abbr

            # Get kickoff time and season/week info
            kickoff_str = event.get('date')
            season = event.get('season', {}).get('year')
            week_info = event.get('week', {})
            week = week_info.get('number') if isinstance(week_info, dict) else None

            if kickoff_str and home_team and away_team and season and week:
                # Parse ISO format timestamp
                kickoff_dt = datetime.fromisoformat(kickoff_str.replace('Z', '+00:00'))

                games.append({
                    'home_team': home_team,
                    'away_team': away_team,
                    'kickoff': kickoff_dt,
                    'season': season,
                    'week': week
                })

        logger.info(f"Found {len(games)} games on current scoreboard")
        return games

    except Exception as e:
        logger.error(f"Error fetching ESPN data: {e}")
        return []


def update_database_kickoffs(games: list):
    """Update kickoff times in the database."""
    logger.info(f"Updating kickoff times for {len(games)} games")

    updated = 0
    not_found = []

    with psycopg.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cursor:
            for game in games:
                # Update by matching season, week, home_team, away_team
                cursor.execute(
                    """
                    UPDATE games
                    SET kickoff = %s
                    WHERE season = %s
                      AND week = %s
                      AND home_team = %s
                      AND away_team = %s
                    RETURNING game_id
                    """,
                    (game['kickoff'], game['season'], game['week'], game['home_team'], game['away_team'])
                )

                result = cursor.fetchone()
                if result:
                    updated += 1
                    game_id = result[0]
                    # Convert to ET for display
                    from datetime import timedelta
                    kickoff_et = game['kickoff'] - timedelta(hours=5)  # EST offset
                    logger.info(f"✅ Updated {game_id}: {kickoff_et.strftime('%a %b %d, %I:%M %p')} ET")
                else:
                    not_found.append(f"{game['away_team']}@{game['home_team']}")

        conn.commit()

    logger.info(f"\n{'='*60}")
    logger.info(f"Update complete: {updated} updated, {len(not_found)} not found")
    if not_found:
        logger.warning(f"Games not found in database: {', '.join(not_found)}")
    logger.info(f"{'='*60}\n")

    return updated, len(not_found)


def main():
    """Main execution function."""
    games = fetch_current_week_kickoffs()
    if games:
        update_database_kickoffs(games)
    else:
        logger.error("No games found from ESPN API")


if __name__ == "__main__":
    main()
