#!/usr/bin/env python3
"""
Fix Kickoff Times for All 2025 NFL Games

Fetches correct kickoff times from ESPN API and updates the database.
This fixes the issue where nflverse data has incorrect kickoff times.

Usage:
    python py/data/fix_kickoff_times.py --season 2025 --start-week 6 --end-week 18
    python py/data/fix_kickoff_times.py --season 2025 --week 6  # Single week
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import psycopg2
import requests
from bs4 import BeautifulSoup

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Database config
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5544)),
    'dbname': os.getenv('DB_NAME', 'devdb01'),
    'user': os.getenv('DB_USER', 'dro'),
    'password': os.getenv('DB_PASSWORD', 'sicillionbillions')
}

# Team abbreviation mapping (ESPN to database)
TEAM_MAP = {
    'PHI': 'PHI', 'NYG': 'NYG', 'DEN': 'DEN', 'NYJ': 'NYJ',
    'ARI': 'ARI', 'IND': 'IND', 'LAC': 'LAC', 'MIA': 'MIA',
    'NE': 'NE', 'NO': 'NO', 'CLE': 'CLE', 'PIT': 'PIT',
    'DAL': 'DAL', 'CAR': 'CAR', 'SEA': 'SEA', 'JAX': 'JAX',
    'LAR': 'LA', 'LA': 'LA', 'BAL': 'BAL', 'TEN': 'TEN',
    'LV': 'LV', 'CIN': 'CIN', 'GB': 'GB', 'SF': 'SF',
    'TB': 'TB', 'DET': 'DET', 'KC': 'KC', 'BUF': 'BUF',
    'ATL': 'ATL', 'CHI': 'CHI', 'WSH': 'WAS', 'WAS': 'WAS',
    'HOU': 'HOU', 'MIN': 'MIN',
}


class KickoffTimeFixer:
    """Fetch and fix kickoff times from ESPN."""

    def __init__(self, db_config: Dict = None):
        self.db_config = db_config or DB_CONFIG
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

    def fetch_espn_schedule(self, season: int, week: int) -> List[Dict]:
        """Fetch schedule from ESPN API for a specific week."""
        # Use ESPN's API directly
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        params = {
            'week': week,
            'seasontype': 2,  # Regular season
            'dates': season
        }

        logger.info(f"Fetching schedule from ESPN API: Week {week}, {season}")

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            games = []

            if 'events' not in data:
                logger.warning("No events found in ESPN API response")
                return games

            for event in data['events']:
                try:
                    # Get teams
                    competitions = event.get('competitions', [])
                    if not competitions:
                        continue

                    competition = competitions[0]
                    competitors = competition.get('competitors', [])

                    if len(competitors) < 2:
                        continue

                    # Home team is usually first, away team second in ESPN API
                    home_team = None
                    away_team = None

                    for comp in competitors:
                        abbr = comp.get('team', {}).get('abbreviation', '')
                        mapped_abbr = TEAM_MAP.get(abbr, abbr)

                        if comp.get('homeAway') == 'home':
                            home_team = mapped_abbr
                        elif comp.get('homeAway') == 'away':
                            away_team = mapped_abbr

                    if not home_team or not away_team:
                        logger.warning(f"Could not determine home/away teams for event {event.get('id')}")
                        continue

                    # Get kickoff time
                    date_str = event.get('date')  # ISO format: 2025-10-12T17:00Z
                    if not date_str:
                        logger.warning(f"No date for game {away_team} @ {home_team}")
                        continue

                    # Check status - skip completed games
                    status = event.get('status', {}).get('type', {}).get('state', '')
                    if status == 'post':
                        logger.info(f"Skipping completed game: {away_team} @ {home_team}")
                        continue

                    logger.info(f"Found game: {away_team} @ {home_team} at {date_str}")

                    games.append({
                        'away_team': away_team,
                        'home_team': home_team,
                        'kickoff_utc': date_str,
                        'week': week,
                        'season': season
                    })

                except Exception as e:
                    logger.warning(f"Error parsing event: {e}")
                    continue

            logger.info(f"Found {len(games)} games for Week {week}")
            return games

        except Exception as e:
            logger.error(f"Error fetching ESPN API: {e}")
            return []

    def parse_kickoff_time(self, game: Dict) -> Optional[str]:
        """
        Parse kickoff time from ESPN API data.

        ESPN API returns ISO format UTC timestamps: "2025-10-12T17:00Z"
        We just need to convert to PostgreSQL timestamptz format.
        """
        try:
            kickoff_utc = game.get('kickoff_utc')
            if not kickoff_utc:
                logger.warning(f"No kickoff_utc for {game['away_team']} @ {game['home_team']}")
                return None

            # Parse ISO format (e.g., "2025-10-12T17:00Z")
            # Remove 'Z' and add '+00' for PostgreSQL
            if kickoff_utc.endswith('Z'):
                kickoff_utc = kickoff_utc[:-1] + '+00'
            elif not kickoff_utc.endswith('+00'):
                kickoff_utc = kickoff_utc + '+00'

            # Convert to PostgreSQL timestamptz format: YYYY-MM-DD HH:MM:SS+00
            dt = datetime.fromisoformat(kickoff_utc.replace('Z', '+00:00'))
            utc_timestamp = dt.strftime("%Y-%m-%d %H:%M:%S+00")

            logger.info(f"  {game['away_team']} @ {game['home_team']}: {kickoff_utc} -> {utc_timestamp}")

            return utc_timestamp

        except Exception as e:
            logger.error(f"Error parsing kickoff time: {e}")
            return None

    def update_database(self, games: List[Dict], dry_run: bool = False) -> int:
        """Update kickoff times in database."""
        if not games:
            logger.warning("No games to update")
            return 0

        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()

        updated_count = 0

        try:
            for game in games:
                kickoff_utc = self.parse_kickoff_time(game)
                if not kickoff_utc:
                    continue

                # Construct game_id
                season = game['season']
                week = game['week']
                away = game['away_team']
                home = game['home_team']
                game_id = f"{season}_{week:02d}_{away}_{home}"

                if dry_run:
                    logger.info(f"[DRY RUN] Would update {game_id} to {kickoff_utc}")
                    updated_count += 1
                else:
                    # Update database
                    query = """
                    UPDATE games
                    SET kickoff = %s::timestamptz
                    WHERE game_id = %s
                    """

                    cur.execute(query, (kickoff_utc, game_id))

                    if cur.rowcount > 0:
                        logger.info(f"✓ Updated {game_id} to {kickoff_utc}")
                        updated_count += 1
                    else:
                        logger.warning(f"Game not found in database: {game_id}")

            if not dry_run:
                conn.commit()
                logger.info(f"Committed {updated_count} updates to database")

            return updated_count

        except Exception as e:
            logger.error(f"Error updating database: {e}")
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()

    def fix_week(self, season: int, week: int, dry_run: bool = False) -> int:
        """Fix kickoff times for a specific week."""
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}Fixing Week {week}, {season}")

        games = self.fetch_espn_schedule(season, week)
        if not games:
            logger.warning(f"No games found for Week {week}")
            return 0

        updated = self.update_database(games, dry_run=dry_run)
        return updated


def main():
    parser = argparse.ArgumentParser(description='Fix NFL game kickoff times from ESPN')
    parser.add_argument('--season', type=int, required=True, help='Season (e.g., 2025)')
    parser.add_argument('--week', type=int, help='Single week to fix')
    parser.add_argument('--start-week', type=int, help='Start week (for range)')
    parser.add_argument('--end-week', type=int, help='End week (for range)')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without updating database')

    args = parser.parse_args()

    # Validate arguments
    if args.week:
        weeks = [args.week]
    elif args.start_week and args.end_week:
        weeks = list(range(args.start_week, args.end_week + 1))
    else:
        parser.error("Either specify --week OR --start-week and --end-week")

    print("=" * 80)
    print("NFL KICKOFF TIME FIXER")
    print("=" * 80)
    print(f"Season: {args.season}")
    print(f"Weeks: {weeks}")
    print(f"Mode: {'DRY RUN (no changes)' if args.dry_run else 'LIVE (will update database)'}")
    print("=" * 80)

    fixer = KickoffTimeFixer()

    total_updated = 0
    for week in weeks:
        updated = fixer.fix_week(args.season, week, dry_run=args.dry_run)
        total_updated += updated
        print(f"\nWeek {week}: {updated} games updated")

    print("\n" + "=" * 80)
    print(f"{'[DRY RUN] ' if args.dry_run else ''}Total: {total_updated} games updated")
    print("=" * 80)

    if args.dry_run:
        print("\nThis was a dry run. Run without --dry-run to apply changes.")
    else:
        print("\n✅ Kickoff times have been updated in the database!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
