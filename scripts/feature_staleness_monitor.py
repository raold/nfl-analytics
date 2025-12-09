#!/usr/bin/env python3
"""
NFL Analytics Feature Staleness Monitor

Tracks freshness of critical features/tables to ensure data is current for predictions.
Detects when materialized views need refreshing, when data ingestion has stalled,
or when seasonal data is out of date.

Usage:
    python scripts/feature_staleness_monitor.py                # Run all checks, update metadata
    python scripts/feature_staleness_monitor.py --report       # Show current status (read-only)
    python scripts/feature_staleness_monitor.py --init         # Initialize metadata table
    python scripts/feature_staleness_monitor.py --verbose      # Detailed output
    python scripts/feature_staleness_monitor.py --alert-only   # Only show stale features

Exit Codes:
    0 - All features fresh
    1 - Warning: Some features stale (not critical)
    2 - Critical: Mission-critical features stale

Author: Claude Code (Sonnet 4.5)
Date: December 7, 2025
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import psycopg2
from psycopg2.extras import DictCursor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Database connection
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "database": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",
}


class StalenessLevel(Enum):
    """Staleness severity levels."""

    OK = "OK"
    STALE = "STALE"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


@dataclass
class FeatureStatus:
    """Feature staleness status."""

    feature_name: str
    last_updated: datetime | None
    row_count: int
    max_season: int | None
    max_week: int | None
    is_stale: bool
    staleness_level: StalenessLevel
    staleness_reason: str
    checked_at: datetime


def get_db_connection():
    """Get PostgreSQL database connection."""
    return psycopg2.connect(**DB_CONFIG, cursor_factory=DictCursor)


def get_current_nfl_context() -> dict:
    """
    Determine current NFL season context.

    Returns:
        dict with 'season', 'week', 'in_season' (bool)
    """
    now = datetime.now()
    year = now.year
    month = now.month

    # NFL season runs Sep-Feb (mostly year, playoffs in year+1)
    if month >= 9:  # September or later = current year season
        season = year
    elif month <= 2:  # Jan-Feb = previous year's season (playoffs)
        season = year - 1
    else:  # Mar-Aug = offseason
        season = year - 1

    # Rough week estimation (Sep 1 = Week 1)
    # This is approximate - actual weeks vary by year
    if month >= 9:
        week = min((now.day + (month - 9) * 30) // 7 + 1, 18)
    elif month <= 2:
        week = 18 + month * 2  # Playoffs (weeks 19-22)
    else:
        week = None  # Offseason

    in_season = month >= 9 or month <= 2

    return {"season": season, "week": week, "in_season": in_season}


def init_metadata_table(conn):
    """Create metadata.feature_freshness table if it doesn't exist."""
    logger.info("Initializing metadata.feature_freshness table...")

    with conn.cursor() as cur:
        # Create metadata schema if needed
        cur.execute("CREATE SCHEMA IF NOT EXISTS metadata")

        # Create table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata.feature_freshness (
                feature_name TEXT PRIMARY KEY,
                last_updated TIMESTAMP WITH TIME ZONE,
                row_count BIGINT,
                max_season INT,
                max_week INT,
                is_stale BOOLEAN,
                staleness_reason TEXT,
                checked_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """
        )

        # Create index on checked_at for history queries
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_feature_freshness_checked_at
            ON metadata.feature_freshness(checked_at DESC)
        """
        )

        conn.commit()
        logger.info("Metadata table initialized successfully")


def check_mart_table_freshness(
    conn, table_name: str, context: dict, criticality: str = "important"
) -> FeatureStatus:
    """
    Check freshness of mart table (season/week based).

    Args:
        conn: Database connection
        table_name: Full table name (schema.table)
        context: Current NFL context (season, week, in_season)
        criticality: 'critical' or 'important'

    Returns:
        FeatureStatus
    """
    try:
        with conn.cursor() as cur:
            # Check if table exists
            schema, table = table_name.split(".")
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = %s AND table_name = %s
                )
            """,
                (schema, table),
            )

            if not cur.fetchone()[0]:
                return FeatureStatus(
                    feature_name=table_name,
                    last_updated=None,
                    row_count=0,
                    max_season=None,
                    max_week=None,
                    is_stale=True,
                    staleness_level=StalenessLevel.CRITICAL,
                    staleness_reason="Table does not exist",
                    checked_at=datetime.now(),
                )

            # Get row count and max season/week
            cur.execute(
                f"""
                SELECT
                    COUNT(*) as row_count,
                    MAX(season) as max_season,
                    MAX(week) as max_week
                FROM {table_name}
                WHERE season IS NOT NULL
            """
            )
            result = cur.fetchone()
            row_count = result[0] or 0
            max_season = result[1]
            max_week = result[2]

            # Determine staleness
            is_stale = False
            staleness_reason = "Data is current"
            staleness_level = StalenessLevel.OK

            if row_count == 0:
                is_stale = True
                staleness_reason = "Table is empty"
                staleness_level = (
                    StalenessLevel.CRITICAL if criticality == "critical" else StalenessLevel.STALE
                )
            elif max_season is None:
                is_stale = True
                staleness_reason = "No season data found"
                staleness_level = StalenessLevel.STALE
            elif context["in_season"]:
                # During season, should have current week data
                if max_season < context["season"]:
                    is_stale = True
                    staleness_reason = (
                        f"Season behind: has {max_season}, current is {context['season']}"
                    )
                    staleness_level = StalenessLevel.CRITICAL
                elif context["week"] and max_week and max_week < context["week"] - 1:
                    # Allow 1 week lag (games may not be ingested yet)
                    is_stale = True
                    staleness_reason = f"Week behind: has {max_week}, current is {context['week']}"
                    staleness_level = StalenessLevel.STALE
            else:
                # Offseason - check if we have last season's data
                if max_season < context["season"]:
                    is_stale = True
                    staleness_reason = (
                        f"Missing last season: has {max_season}, expected {context['season']}"
                    )
                    staleness_level = StalenessLevel.STALE

            # Get last updated timestamp if available (from pg_stat_user_tables)
            cur.execute(
                """
                SELECT last_vacuum, last_autovacuum, last_analyze, last_autoanalyze
                FROM pg_stat_user_tables
                WHERE schemaname = %s AND relname = %s
            """,
                (schema, table),
            )
            stats = cur.fetchone()
            last_updated = None
            if stats:
                # Use most recent of available timestamps
                timestamps = [t for t in stats if t is not None]
                if timestamps:
                    last_updated = max(timestamps)

            return FeatureStatus(
                feature_name=table_name,
                last_updated=last_updated,
                row_count=row_count,
                max_season=max_season,
                max_week=max_week,
                is_stale=is_stale,
                staleness_level=staleness_level,
                staleness_reason=staleness_reason,
                checked_at=datetime.now(),
            )

    except Exception as e:
        logger.error(f"Error checking {table_name}: {e}")
        return FeatureStatus(
            feature_name=table_name,
            last_updated=None,
            row_count=0,
            max_season=None,
            max_week=None,
            is_stale=True,
            staleness_level=StalenessLevel.UNKNOWN,
            staleness_reason=f"Error: {str(e)[:100]}",
            checked_at=datetime.now(),
        )


def check_weather_freshness(conn, context: dict) -> FeatureStatus:
    """Check if weather data exists for recent games."""
    try:
        with conn.cursor() as cur:
            # Check if weather table exists
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = 'weather'
                )
            """
            )

            if not cur.fetchone()[0]:
                return FeatureStatus(
                    feature_name="public.weather",
                    last_updated=None,
                    row_count=0,
                    max_season=None,
                    max_week=None,
                    is_stale=True,
                    staleness_level=StalenessLevel.STALE,
                    staleness_reason="Weather table does not exist",
                    checked_at=datetime.now(),
                )

            # Get total row count
            cur.execute("SELECT COUNT(*) FROM public.weather")
            row_count = cur.fetchone()[0]

            # Check for recent games with weather data
            cur.execute(
                """
                SELECT
                    COUNT(*) as games_with_weather,
                    COUNT(CASE WHEN w.game_id IS NOT NULL THEN 1 END) as weather_coverage
                FROM public.games g
                LEFT JOIN public.weather w ON g.game_id = w.game_id
                WHERE g.kickoff >= CURRENT_DATE - INTERVAL '7 days'
            """
            )
            result = cur.fetchone()
            recent_games = result[0]
            weather_coverage = result[1]

            is_stale = False
            staleness_reason = (
                f"Weather data coverage: {weather_coverage}/{recent_games} recent games"
            )
            staleness_level = StalenessLevel.OK

            if recent_games > 0 and weather_coverage == 0:
                is_stale = True
                staleness_reason = f"No weather data for {recent_games} recent games"
                staleness_level = StalenessLevel.STALE
            elif recent_games > 0 and weather_coverage < recent_games * 0.5:
                is_stale = True
                staleness_reason = f"Low weather coverage: {weather_coverage}/{recent_games} games"
                staleness_level = StalenessLevel.STALE

            return FeatureStatus(
                feature_name="public.weather",
                last_updated=None,
                row_count=row_count,
                max_season=None,
                max_week=None,
                is_stale=is_stale,
                staleness_level=staleness_level,
                staleness_reason=staleness_reason,
                checked_at=datetime.now(),
            )

    except Exception as e:
        logger.error(f"Error checking weather: {e}")
        return FeatureStatus(
            feature_name="public.weather",
            last_updated=None,
            row_count=0,
            max_season=None,
            max_week=None,
            is_stale=True,
            staleness_level=StalenessLevel.UNKNOWN,
            staleness_reason=f"Error: {str(e)[:100]}",
            checked_at=datetime.now(),
        )


def check_odds_freshness(conn) -> FeatureStatus:
    """Check if odds_history has recent captures."""
    try:
        with conn.cursor() as cur:
            # Check if table exists
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = 'odds_history'
                )
            """
            )

            if not cur.fetchone()[0]:
                return FeatureStatus(
                    feature_name="public.odds_history",
                    last_updated=None,
                    row_count=0,
                    max_season=None,
                    max_week=None,
                    is_stale=True,
                    staleness_level=StalenessLevel.STALE,
                    staleness_reason="Odds table does not exist",
                    checked_at=datetime.now(),
                )

            # Get row count and last captured_at
            cur.execute(
                """
                SELECT
                    COUNT(*) as row_count,
                    MAX(captured_at) as last_capture
                FROM public.odds_history
            """
            )
            result = cur.fetchone()
            row_count = result[0] or 0
            last_capture = result[1]

            is_stale = False
            staleness_reason = "Odds data is current"
            staleness_level = StalenessLevel.OK

            if row_count == 0:
                is_stale = True
                staleness_reason = "No odds data captured"
                staleness_level = StalenessLevel.STALE
            elif last_capture is None:
                is_stale = True
                staleness_reason = "No capture timestamp found"
                staleness_level = StalenessLevel.STALE
            else:
                hours_since_capture = (
                    datetime.now() - last_capture.replace(tzinfo=None)
                ).total_seconds() / 3600

                if hours_since_capture > 24:
                    is_stale = True
                    staleness_reason = f"Last capture {hours_since_capture:.1f} hours ago"
                    staleness_level = StalenessLevel.STALE
                else:
                    staleness_reason = f"Last capture {hours_since_capture:.1f} hours ago"

            return FeatureStatus(
                feature_name="public.odds_history",
                last_updated=last_capture,
                row_count=row_count,
                max_season=None,
                max_week=None,
                is_stale=is_stale,
                staleness_level=staleness_level,
                staleness_reason=staleness_reason,
                checked_at=datetime.now(),
            )

    except Exception as e:
        logger.error(f"Error checking odds_history: {e}")
        return FeatureStatus(
            feature_name="public.odds_history",
            last_updated=None,
            row_count=0,
            max_season=None,
            max_week=None,
            is_stale=True,
            staleness_level=StalenessLevel.UNKNOWN,
            staleness_reason=f"Error: {str(e)[:100]}",
            checked_at=datetime.now(),
        )


def check_materialized_view_freshness(conn, view_name: str, context: dict) -> FeatureStatus:
    """Check freshness of materialized view."""
    try:
        with conn.cursor() as cur:
            # Check if MV exists
            schema, mv = view_name.split(".")
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM pg_matviews
                    WHERE schemaname = %s AND matviewname = %s
                )
            """,
                (schema, mv),
            )

            if not cur.fetchone()[0]:
                return FeatureStatus(
                    feature_name=view_name,
                    last_updated=None,
                    row_count=0,
                    max_season=None,
                    max_week=None,
                    is_stale=True,
                    staleness_level=StalenessLevel.CRITICAL,
                    staleness_reason="Materialized view does not exist",
                    checked_at=datetime.now(),
                )

            # Get row count
            cur.execute(f"SELECT COUNT(*) FROM {view_name}")
            row_count = cur.fetchone()[0]

            # Get last refresh time (pg_stat_user_tables has stats for MVs too)
            cur.execute(
                """
                SELECT last_vacuum, last_autovacuum, last_analyze, last_autoanalyze
                FROM pg_stat_user_tables
                WHERE schemaname = %s AND relname = %s
            """,
                (schema, mv),
            )
            stats = cur.fetchone()
            last_updated = None
            if stats:
                timestamps = [t for t in stats if t is not None]
                if timestamps:
                    last_updated = max(timestamps)

            # MVs should be refreshed daily
            is_stale = False
            staleness_reason = "MV is current"
            staleness_level = StalenessLevel.OK

            if row_count == 0:
                is_stale = True
                staleness_reason = "MV is empty"
                staleness_level = StalenessLevel.CRITICAL
            elif last_updated is None:
                is_stale = True
                staleness_reason = "No refresh timestamp (never refreshed?)"
                staleness_level = StalenessLevel.STALE
            else:
                hours_since_refresh = (
                    datetime.now() - last_updated.replace(tzinfo=None)
                ).total_seconds() / 3600

                if hours_since_refresh > 48:
                    is_stale = True
                    staleness_reason = f"Last refresh {hours_since_refresh:.1f} hours ago (>48h)"
                    staleness_level = StalenessLevel.CRITICAL
                elif hours_since_refresh > 24:
                    is_stale = True
                    staleness_reason = f"Last refresh {hours_since_refresh:.1f} hours ago (>24h)"
                    staleness_level = StalenessLevel.STALE
                else:
                    staleness_reason = f"Last refresh {hours_since_refresh:.1f} hours ago"

            return FeatureStatus(
                feature_name=view_name,
                last_updated=last_updated,
                row_count=row_count,
                max_season=None,
                max_week=None,
                is_stale=is_stale,
                staleness_level=staleness_level,
                staleness_reason=staleness_reason,
                checked_at=datetime.now(),
            )

    except Exception as e:
        logger.error(f"Error checking {view_name}: {e}")
        return FeatureStatus(
            feature_name=view_name,
            last_updated=None,
            row_count=0,
            max_season=None,
            max_week=None,
            is_stale=True,
            staleness_level=StalenessLevel.UNKNOWN,
            staleness_reason=f"Error: {str(e)[:100]}",
            checked_at=datetime.now(),
        )


def save_feature_status(conn, status: FeatureStatus):
    """Save feature status to metadata.feature_freshness table."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO metadata.feature_freshness
                (feature_name, last_updated, row_count, max_season, max_week,
                 is_stale, staleness_reason, checked_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (feature_name)
            DO UPDATE SET
                last_updated = EXCLUDED.last_updated,
                row_count = EXCLUDED.row_count,
                max_season = EXCLUDED.max_season,
                max_week = EXCLUDED.max_week,
                is_stale = EXCLUDED.is_stale,
                staleness_reason = EXCLUDED.staleness_reason,
                checked_at = EXCLUDED.checked_at
        """,
            (
                status.feature_name,
                status.last_updated,
                status.row_count,
                status.max_season,
                status.max_week,
                status.is_stale,
                status.staleness_reason,
                status.checked_at,
            ),
        )
    conn.commit()


def run_all_checks(conn, context: dict, save_to_db: bool = True) -> list[FeatureStatus]:
    """Run all feature freshness checks."""
    statuses = []

    # Check mart tables (season/week based)
    logger.info("Checking mart tables...")
    statuses.append(
        check_mart_table_freshness(conn, "mart.team_playoff_context", context, "important")
    )
    statuses.append(check_mart_table_freshness(conn, "public.rosters_weekly", context, "critical"))

    # Check materialized views
    logger.info("Checking materialized views...")
    statuses.append(check_materialized_view_freshness(conn, "mart.player_game_stats", context))
    statuses.append(check_materialized_view_freshness(conn, "mart.game_summary", context))

    # Check weather and odds
    logger.info("Checking weather and odds...")
    statuses.append(check_weather_freshness(conn, context))
    statuses.append(check_odds_freshness(conn))

    # Save to database
    if save_to_db:
        logger.info("Saving results to metadata.feature_freshness...")
        for status in statuses:
            save_feature_status(conn, status)

    return statuses


def print_report(statuses: list[FeatureStatus], context: dict, alert_only: bool = False):
    """Print feature freshness report."""
    print("=" * 90)
    print("FEATURE STALENESS REPORT")
    print("=" * 90)
    print(f"Current NFL Context: Season {context['season']}, Week {context['week']}")
    print(f"In Season: {context['in_season']}")
    print(f"Checked At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)

    # Group by staleness level
    critical = [s for s in statuses if s.staleness_level == StalenessLevel.CRITICAL]
    stale = [s for s in statuses if s.staleness_level == StalenessLevel.STALE]
    ok = [s for s in statuses if s.staleness_level == StalenessLevel.OK]
    unknown = [s for s in statuses if s.staleness_level == StalenessLevel.UNKNOWN]

    # Print summary
    print(
        f"\nSummary: {len(ok)} OK, {len(stale)} STALE, {len(critical)} CRITICAL, {len(unknown)} UNKNOWN"
    )
    print()

    # Print details
    if critical:
        print("CRITICAL ISSUES:")
        print("-" * 90)
        for s in critical:
            print(f"  {s.feature_name:35s} | {s.staleness_level.value:8s} | {s.staleness_reason}")
            print(f"    Rows: {s.row_count:,} | Season: {s.max_season} | Week: {s.max_week}")
            print("    Action: IMMEDIATE REFRESH REQUIRED")
            print()

    if stale:
        print("STALE FEATURES:")
        print("-" * 90)
        for s in stale:
            print(f"  {s.feature_name:35s} | {s.staleness_level.value:8s} | {s.staleness_reason}")
            print(f"    Rows: {s.row_count:,} | Season: {s.max_season} | Week: {s.max_week}")
            if "MV" in s.staleness_reason or "refresh" in s.staleness_reason.lower():
                print("    Action: Run 'python scripts/refresh_marts.py'")
            elif "weather" in s.feature_name.lower():
                print("    Action: Check weather data ingestion pipeline")
            elif "odds" in s.feature_name.lower():
                print("    Action: Run 'python scripts/fetch_odds.py'")
            else:
                print(f"    Action: Check data ingestion for {s.feature_name}")
            print()

    if not alert_only and ok:
        print("OK FEATURES:")
        print("-" * 90)
        for s in ok:
            print(f"  {s.feature_name:35s} | {s.staleness_level.value:8s} | {s.staleness_reason}")
            print(f"    Rows: {s.row_count:,} | Season: {s.max_season} | Week: {s.max_week}")
            print()

    if unknown:
        print("UNKNOWN STATUS:")
        print("-" * 90)
        for s in unknown:
            print(f"  {s.feature_name:35s} | {s.staleness_level.value:8s} | {s.staleness_reason}")
            print()

    print("=" * 90)


def get_status_from_db(conn) -> list[FeatureStatus]:
    """Retrieve current feature status from database."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                feature_name,
                last_updated,
                row_count,
                max_season,
                max_week,
                is_stale,
                staleness_reason,
                checked_at
            FROM metadata.feature_freshness
            ORDER BY
                CASE
                    WHEN staleness_reason LIKE '%CRITICAL%' THEN 1
                    WHEN is_stale THEN 2
                    ELSE 3
                END,
                feature_name
        """
        )

        statuses = []
        for row in cur.fetchall():
            # Determine staleness level from reason
            if "CRITICAL" in row[6] or "does not exist" in row[6] or "empty" in row[6].lower():
                level = StalenessLevel.CRITICAL
            elif row[5]:  # is_stale
                level = StalenessLevel.STALE
            elif "Error" in row[6]:
                level = StalenessLevel.UNKNOWN
            else:
                level = StalenessLevel.OK

            statuses.append(
                FeatureStatus(
                    feature_name=row[0],
                    last_updated=row[1],
                    row_count=row[2] or 0,
                    max_season=row[3],
                    max_week=row[4],
                    is_stale=row[5] or False,
                    staleness_level=level,
                    staleness_reason=row[6] or "Unknown",
                    checked_at=row[7] or datetime.now(),
                )
            )

        return statuses


def main():
    parser = argparse.ArgumentParser(description="Monitor Feature Staleness")
    parser.add_argument("--init", action="store_true", help="Initialize metadata table")
    parser.add_argument(
        "--report", action="store_true", help="Show current status from DB (read-only)"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--alert-only", action="store_true", help="Only show stale features")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    conn = get_db_connection()

    try:
        # Initialize table if requested
        if args.init:
            init_metadata_table(conn)
            print("Metadata table initialized successfully!")
            return 0

        # Get current NFL context
        context = get_current_nfl_context()

        if args.report:
            # Read-only report from existing data
            logger.info("Fetching current status from database...")
            statuses = get_status_from_db(conn)
        else:
            # Run fresh checks
            logger.info("Running feature freshness checks...")
            statuses = run_all_checks(conn, context, save_to_db=True)

        # Print report
        print_report(statuses, context, alert_only=args.alert_only)

        # Determine exit code
        critical_count = sum(1 for s in statuses if s.staleness_level == StalenessLevel.CRITICAL)
        stale_count = sum(1 for s in statuses if s.staleness_level == StalenessLevel.STALE)

        if critical_count > 0:
            logger.error(f"CRITICAL: {critical_count} features require immediate attention")
            return 2
        elif stale_count > 0:
            logger.warning(f"WARNING: {stale_count} features are stale")
            return 1
        else:
            logger.info("All features are fresh")
            return 0

    except Exception as e:
        logger.error(f"Error running staleness monitor: {e}")
        import traceback

        traceback.print_exc()
        return 2

    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
