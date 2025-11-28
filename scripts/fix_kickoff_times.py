#!/usr/bin/env python3
"""
Fix NFL game kickoff times to standard NFL schedule times.

Standard NFL kickoff times (ET):
- Thursday Night: 8:15 PM (some 8:20 PM)
- Sunday Early: 1:00 PM
- Sunday Late: 4:05 PM or 4:25 PM
- Sunday Night: 8:20 PM
- Monday Night: 8:15 PM (some 8:20 PM)
- International: 9:30 AM
- Thanksgiving: 12:30 PM, 4:30 PM, 8:20 PM
- Black Friday: 3:00 PM
"""

import psycopg
from datetime import datetime, timezone
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Database config
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "dbname": "devdb01",
    "user": "dro",
    "password": "sicillionbillions"
}

def fix_kickoff_times(season, week):
    """Fix kickoff times for a specific week to standard NFL times."""

    conn = psycopg.connect(**DB_CONFIG)
    cursor = conn.cursor()

    try:
        # Get current kickoff times
        cursor.execute("""
            SELECT game_id, home_team, away_team, kickoff,
                   EXTRACT(DOW FROM kickoff) as day_of_week,
                   EXTRACT(HOUR FROM kickoff AT TIME ZONE 'America/New_York') as hour_et,
                   EXTRACT(MINUTE FROM kickoff AT TIME ZONE 'America/New_York') as minute_et
            FROM games
            WHERE season = %s AND week = %s
            ORDER BY kickoff
        """, (season, week))

        games = cursor.fetchall()
        logger.info(f"Found {len(games)} games in Week {week}, {season}")

        updates = []
        for game_id, home, away, kickoff, dow, hour_et, minute_et in games:
            new_kickoff = None

            # Check if time needs fixing (non-standard minutes)
            if minute_et not in [0, 5, 15, 20, 25, 30]:
                logger.info(f"Fixing {game_id}: {home} vs {away} - current time {hour_et:02.0f}:{minute_et:02.0f} ET")

                # Determine proper kickoff time based on day and approximate time
                if dow == 4:  # Thursday
                    new_kickoff = kickoff.replace(hour=1, minute=15, second=0)  # 8:15 PM ET next day in UTC
                elif dow == 5:  # Friday
                    new_kickoff = kickoff.replace(hour=20, minute=0, second=0)  # 3:00 PM ET
                elif dow == 0:  # Sunday
                    if 11 <= hour_et <= 13:
                        # 1:00 PM ET game
                        new_kickoff = kickoff.replace(hour=18, minute=0, second=0)
                    elif 15 <= hour_et <= 17:
                        # 4:05 or 4:25 PM ET game
                        new_kickoff = kickoff.replace(hour=21, minute=5, second=0)
                    elif hour_et >= 19:
                        # 8:20 PM ET Sunday Night Football
                        new_kickoff = kickoff.replace(hour=1, minute=20, second=0)
                        new_kickoff = new_kickoff.replace(day=new_kickoff.day + 1)
                elif dow == 1:  # Monday
                    # 8:15 PM ET Monday Night Football
                    new_kickoff = kickoff.replace(hour=1, minute=15, second=0)
                    new_kickoff = new_kickoff.replace(day=new_kickoff.day + 1)

                if new_kickoff:
                    updates.append((new_kickoff, game_id))

        # Apply updates
        if updates:
            cursor.executemany("""
                UPDATE games
                SET kickoff = %s
                WHERE game_id = %s
            """, updates)
            conn.commit()
            logger.info(f"Updated {len(updates)} kickoff times")
        else:
            logger.info("All kickoff times appear correct")

        # Show final times
        cursor.execute("""
            SELECT game_id,
                   home_team || ' vs ' || away_team as matchup,
                   TO_CHAR(kickoff AT TIME ZONE 'America/New_York', 'Dy MM/DD HH12:MI AM TZ') as kickoff_display
            FROM games
            WHERE season = %s AND week = %s
            ORDER BY kickoff
        """, (season, week))

        logger.info("\nFinal kickoff times:")
        for game_id, matchup, kickoff_display in cursor.fetchall():
            logger.info(f"  {game_id}: {matchup} - {kickoff_display}")

    except Exception as e:
        logger.error(f"Error fixing kickoff times: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    import sys

    # Fix specific weeks or current season
    if len(sys.argv) > 2:
        season = int(sys.argv[1])
        week = int(sys.argv[2])
        fix_kickoff_times(season, week)
    else:
        # Fix all weeks in current season
        current_season = 2024
        for week in range(1, 19):  # Weeks 1-18
            logger.info(f"\n=== Fixing Week {week}, {current_season} ===")
            fix_kickoff_times(current_season, week)