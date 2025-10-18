#!/usr/bin/env python3
"""
Fix kickoff times in the games table.

The kickoff times are currently all set to midnight UTC.
This script updates them with proper kickoff times based on typical NFL game times.
"""

from datetime import datetime

import psycopg2
import pytz

# Database connection parameters
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "dbname": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",
}


def get_typical_kickoff_time(day_of_week, is_primetime=False, is_morning=False):
    """
    Get typical kickoff time based on day of week and game type.

    Returns hour in ET (Eastern Time).
    """
    if day_of_week == 3:  # Thursday
        return 20.25  # 8:15 PM ET
    elif day_of_week == 5:  # Saturday (late season)
        return 16.5 if not is_primetime else 20.25  # 4:30 PM or 8:15 PM ET
    elif day_of_week == 6:  # Sunday
        if is_morning:
            return 9.5  # 9:30 AM ET (London games)
        elif is_primetime:
            return 20.25  # 8:15 PM ET (SNF)
        else:
            return 13  # 1:00 PM ET (default)
    elif day_of_week == 0:  # Monday
        return 20.25  # 8:15 PM ET (MNF)
    else:
        return 13  # Default to 1 PM ET


def fix_kickoff_times():
    """Fix kickoff times in the games table."""

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    try:
        # Get all games with their current kickoff times
        cur.execute(
            """
            SELECT game_id, season, week, kickoff, home_team, away_team
            FROM games
            ORDER BY season, week, game_id
        """
        )

        games = cur.fetchall()
        print(f"Found {len(games)} games to process")

        # Eastern Time zone
        et_tz = pytz.timezone("America/New_York")

        updates = []
        for game_id, season, week, kickoff_ts, home_team, away_team in games:
            # Get the date from the current kickoff (which has correct date, wrong time)
            game_date = kickoff_ts.date()
            day_of_week = game_date.weekday()

            # Determine if this is a primetime game based on team matchup and week
            # This is a simplified heuristic - in reality would need actual schedule data
            is_primetime = False
            is_morning = False

            # Thursday games are always primetime
            if day_of_week == 3:
                is_primetime = True
            # Monday games are always primetime
            elif day_of_week == 0:
                is_primetime = True
            # Sunday night games (one per week typically)
            elif day_of_week == 6:
                # Simple heuristic: first game alphabetically on Sunday might be London game
                # Last game might be SNF
                games_on_date = [g for g in games if g[3].date() == game_date]
                if games_on_date:
                    if game_id == games_on_date[0][0]:
                        # Could be London game
                        if "JAX" in (home_team, away_team) or week <= 8:
                            is_morning = True
                    elif game_id == games_on_date[-1][0]:
                        is_primetime = True

            # Get typical kickoff hour
            kickoff_hour = get_typical_kickoff_time(day_of_week, is_primetime, is_morning)
            hour = int(kickoff_hour)
            minute = int((kickoff_hour - hour) * 60)

            # Create the proper kickoff time in ET
            kickoff_et = datetime(
                game_date.year, game_date.month, game_date.day, hour, minute, 0, tzinfo=et_tz
            )

            # Convert to UTC for storage
            kickoff_utc = kickoff_et.astimezone(pytz.UTC)

            updates.append((kickoff_utc, game_id))

            if len(updates) % 100 == 0:
                print(f"Processed {len(updates)} games...")

        # Update all games
        print(f"Updating {len(updates)} games with proper kickoff times...")
        cur.executemany(
            """
            UPDATE games
            SET kickoff = %s
            WHERE game_id = %s
        """,
            updates,
        )

        conn.commit()
        print(f"Successfully updated {len(updates)} games")

        # Verify a sample
        cur.execute(
            """
            SELECT game_id, season, week, kickoff,
                   kickoff AT TIME ZONE 'America/New_York' as kickoff_et
            FROM games
            WHERE season = 2024 AND week = 1
            ORDER BY kickoff
            LIMIT 10
        """
        )

        print("\nSample of updated games (2024 Week 1):")
        print("Game ID | Kickoff UTC | Kickoff ET")
        print("-" * 60)
        for row in cur.fetchall():
            game_id, season, week, kickoff_utc, kickoff_et = row
            print(f"{game_id} | {kickoff_utc} | {kickoff_et}")

    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    fix_kickoff_times()
