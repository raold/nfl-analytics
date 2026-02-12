#!/usr/bin/env python3
"""
Refresh Materialized Views and Mart Tables

Refreshes all materialized views in the mart schema and updates
timestamps. Should be run daily (preferably after data ingestion).

Usage:
  python scripts/refresh_marts.py                    # Refresh all MVs
  python scripts/refresh_marts.py --mv game_summary  # Refresh specific MV
  python scripts/refresh_marts.py --status           # Check MV freshness
  python scripts/refresh_marts.py --concurrent       # Use CONCURRENTLY (no lock)

Author: Claude Code (Opus 4.5)
Date: December 7, 2025
"""

import argparse
import logging
from datetime import datetime

import psycopg2
from psycopg2 import sql
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

# Materialized views to refresh (in dependency order)
MATERIALIZED_VIEWS = [
    ("mart", "player_hierarchy"),
    ("mart", "player_game_stats"),
    ("mart", "game_summary"),
]

# Regular tables to verify have data
MART_TABLES = [
    ("mart", "team_epa"),
    ("mart", "team_playoff_context"),
    ("mart", "team_injury_load"),
    ("mart", "team_4th_down_features"),
    ("mart", "position_groups"),
    ("mart", "bayesian_player_ratings"),
]


def get_db_connection():
    """Get PostgreSQL database connection."""
    return psycopg2.connect(**DB_CONFIG, cursor_factory=DictCursor)


def get_mv_status(conn) -> list[dict]:
    """Get status of all materialized views."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                schemaname,
                matviewname,
                pg_size_pretty(pg_relation_size(schemaname || '.' || matviewname)) as size,
                (SELECT COUNT(*) FROM pg_stat_user_tables
                 WHERE schemaname = mv.schemaname AND relname = mv.matviewname) as stats_available
            FROM pg_matviews mv
            WHERE schemaname NOT LIKE 'pg_%'
            ORDER BY schemaname, matviewname
        """)
        return [dict(row) for row in cur.fetchall()]


def get_table_counts(conn, tables: list[tuple[str, str]]) -> dict[str, int]:
    """Get row counts for specified tables."""
    counts = {}
    for schema, table in tables:
        try:
            with conn.cursor() as cur:
                query = sql.SQL("SELECT COUNT(*) FROM {}.{}").format(
                    sql.Identifier(schema), sql.Identifier(table)
                )
                cur.execute(query)
                counts[f"{schema}.{table}"] = cur.fetchone()[0]
        except psycopg2.Error as e:
            logger.warning(f"Could not count {schema}.{table}: {e}")
            counts[f"{schema}.{table}"] = -1
    return counts


def refresh_materialized_view(conn, schema: str, name: str, concurrent: bool = False):
    """Refresh a single materialized view."""
    full_name = f"{schema}.{name}"
    qualified = sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(name))

    logger.info(f"Refreshing {full_name}...")
    start = datetime.now()

    try:
        with conn.cursor() as cur:
            # CONCURRENTLY requires a unique index; fall back to regular if fails
            try:
                if concurrent:
                    cur.execute(sql.SQL("REFRESH MATERIALIZED VIEW CONCURRENTLY {}").format(qualified))
                else:
                    cur.execute(sql.SQL("REFRESH MATERIALIZED VIEW {}").format(qualified))
            except psycopg2.errors.ObjectNotInPrerequisiteState:
                if concurrent:
                    logger.warning(f"CONCURRENTLY failed for {full_name}, using regular refresh")
                    conn.rollback()
                    cur.execute(sql.SQL("REFRESH MATERIALIZED VIEW {}").format(qualified))
                else:
                    raise

        conn.commit()
        elapsed = (datetime.now() - start).total_seconds()
        logger.info(f"  Refreshed {full_name} in {elapsed:.2f}s")
        return True

    except psycopg2.Error as e:
        conn.rollback()
        logger.error(f"  Failed to refresh {full_name}: {e}")
        return False


def refresh_all_mvs(concurrent: bool = False) -> dict:
    """Refresh all materialized views."""
    conn = get_db_connection()

    results = {"success": 0, "failed": 0, "views": []}

    for schema, name in MATERIALIZED_VIEWS:
        success = refresh_materialized_view(conn, schema, name, concurrent)
        if success:
            results["success"] += 1
        else:
            results["failed"] += 1
        results["views"].append({
            "name": f"{schema}.{name}",
            "success": success,
        })

    conn.close()
    return results


def check_status():
    """Check and display status of all marts."""
    conn = get_db_connection()

    print("=" * 70)
    print("MART STATUS REPORT")
    print("=" * 70)

    # Materialized views
    print("\nMaterialized Views:")
    mvs = get_mv_status(conn)
    for mv in mvs:
        print(f"  {mv['schemaname']}.{mv['matviewname']:20s} - {mv['size']}")

    # Table row counts
    print("\nMart Tables:")
    counts = get_table_counts(conn, MART_TABLES)
    for table, count in sorted(counts.items()):
        status = "OK" if count > 0 else "EMPTY"
        print(f"  {table:35s} - {count:>8,} rows [{status}]")

    # MV row counts
    print("\nMV Row Counts:")
    mv_counts = get_table_counts(conn, MATERIALIZED_VIEWS)
    for mv, count in sorted(mv_counts.items()):
        print(f"  {mv:35s} - {count:>8,} rows")

    conn.close()

    # Summary
    empty_tables = [t for t, c in counts.items() if c == 0]
    if empty_tables:
        print(f"\n WARNING: {len(empty_tables)} empty tables: {', '.join(empty_tables)}")
    else:
        print("\n All mart tables have data.")


def main():
    parser = argparse.ArgumentParser(description="Refresh Mart Materialized Views")
    parser.add_argument("--status", action="store_true", help="Check MV status")
    parser.add_argument("--mv", type=str, help="Refresh specific MV (e.g., game_summary)")
    parser.add_argument("--concurrent", action="store_true",
                        help="Use CONCURRENTLY (no table lock)")
    parser.add_argument("--all", action="store_true", help="Refresh all MVs")

    args = parser.parse_args()

    if args.status:
        check_status()
        return

    if args.mv:
        conn = get_db_connection()
        # Find the schema for this MV
        schema = "mart"
        for s, n in MATERIALIZED_VIEWS:
            if n == args.mv:
                schema = s
                break
        refresh_materialized_view(conn, schema, args.mv, args.concurrent)
        conn.close()
        return

    # Default: refresh all
    print("=" * 70)
    print("REFRESHING ALL MATERIALIZED VIEWS")
    print("=" * 70)

    results = refresh_all_mvs(args.concurrent)

    print(f"\nResults: {results['success']} succeeded, {results['failed']} failed")

    if results["failed"] > 0:
        print("Failed views:")
        for v in results["views"]:
            if not v["success"]:
                print(f"  - {v['name']}")


if __name__ == "__main__":
    main()
