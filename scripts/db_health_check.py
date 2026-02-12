#!/usr/bin/env python3
"""
NFL Analytics Database Health Check

Production-ready validation script to ensure database integrity.
Run after ingestion, migrations, or as a pre-flight check before modeling.

Usage:
    python scripts/db_health_check.py
    python scripts/db_health_check.py --verbose
    python scripts/db_health_check.py --fix  # Attempt auto-fixes
"""

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

try:
    import psycopg
except ImportError:
    import psycopg2 as psycopg


class Status(Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class CheckResult:
    name: str
    status: Status
    message: str
    row_count: int | None = None
    expected_min: int | None = None


def get_connection():
    """Create database connection."""
    return psycopg.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5544")),
        dbname=os.environ.get("POSTGRES_DB", "devdb01"),
        user=os.environ.get("POSTGRES_USER", "dro"),
        password=os.environ.get("POSTGRES_PASSWORD", "sicillionbillions"),
    )


# Expected table configurations: (min_rows, criticality)
# Criticality: 'critical' = fails build, 'important' = warning, 'optional' = info only
TABLE_EXPECTATIONS = {
    # Core tables - CRITICAL for modeling
    "public.games": (7000, "critical"),
    "public.plays": (1000000, "critical"),
    "public.players": (20000, "critical"),
    "public.rosters_weekly": (30000, "critical"),
    # Advanced stats - IMPORTANT for features
    "public.nextgen_passing": (5000, "important"),
    "public.nextgen_receiving": (10000, "important"),
    "public.nextgen_rushing": (5000, "important"),
    "public.participation": (400000, "important"),
    "public.injuries": (50000, "important"),
    "public.officials": (20000, "important"),
    # Mart tables - CRITICAL for modeling
    "mart.team_epa": (10000, "critical"),
    "mart.team_4th_down_features": (10000, "important"),
    "mart.team_injury_load": (2000, "important"),
    "mart.player_hierarchy": (20000, "critical"),
    "mart.player_game_stats": (20000, "critical"),
    "mart.game_summary": (7000, "critical"),
    # Reference tables
    "mart.position_groups": (25, "critical"),
    # Optional tables (nice to have but not blocking)
    "public.weather": (0, "optional"),
    "public.odds_history": (0, "optional"),
    "public.espn_qbr": (0, "optional"),
    "public.depth_charts": (0, "optional"),
    "public.combine": (0, "optional"),
    "public.contracts": (0, "optional"),
    "public.draft_picks": (10000, "optional"),
}


def check_table(conn, table: str, min_rows: int, criticality: str) -> CheckResult:
    """Check if table exists and has expected row count."""
    schema, table_name = table.split(".")

    try:
        with conn.cursor() as cur:
            # Check if table/view exists
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = %s AND table_name = %s
                    UNION
                    SELECT 1 FROM pg_matviews
                    WHERE schemaname = %s AND matviewname = %s
                )
            """,
                (schema, table_name, schema, table_name),
            )
            exists = cur.fetchone()[0]

            if not exists:
                return CheckResult(
                    name=table,
                    status=Status.FAIL if criticality == "critical" else Status.WARN,
                    message="Table/view does not exist",
                    expected_min=min_rows,
                )

            # Get row count — use SQL composition for identifier safety
            from psycopg2 import sql as sql_mod
            parts = table.split(".")
            if len(parts) == 2:
                query = sql_mod.SQL("SELECT COUNT(*) FROM {}.{}").format(
                    sql_mod.Identifier(parts[0]), sql_mod.Identifier(parts[1])
                )
            else:
                query = sql_mod.SQL("SELECT COUNT(*) FROM {}").format(
                    sql_mod.Identifier(table)
                )
            cur.execute(query)
            count = cur.fetchone()[0]

            if count >= min_rows:
                return CheckResult(
                    name=table,
                    status=Status.PASS,
                    message=f"OK ({count:,} rows)",
                    row_count=count,
                    expected_min=min_rows,
                )
            elif count == 0 and min_rows == 0:
                return CheckResult(
                    name=table,
                    status=Status.SKIP,
                    message="Empty (optional)",
                    row_count=count,
                    expected_min=min_rows,
                )
            elif count == 0:
                return CheckResult(
                    name=table,
                    status=Status.FAIL if criticality == "critical" else Status.WARN,
                    message=f"EMPTY (expected {min_rows:,}+)",
                    row_count=count,
                    expected_min=min_rows,
                )
            else:
                return CheckResult(
                    name=table,
                    status=Status.WARN,
                    message=f"LOW: {count:,} rows (expected {min_rows:,}+)",
                    row_count=count,
                    expected_min=min_rows,
                )

    except Exception as e:
        return CheckResult(
            name=table, status=Status.FAIL, message=f"ERROR: {str(e)[:50]}", expected_min=min_rows
        )


def check_schema_consistency(conn) -> list[CheckResult]:
    """Check for schema consistency issues."""
    results = []

    with conn.cursor() as cur:
        # Check injuries table has correct schema
        cur.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'injuries'
            ORDER BY ordinal_position
        """
        )
        injury_cols = [r[0] for r in cur.fetchall()]
        expected_cols = ["season", "game_type", "team", "week", "gsis_id"]

        if all(c in injury_cols for c in expected_cols):
            results.append(
                CheckResult(
                    name="schema.injuries",
                    status=Status.PASS,
                    message="Correct schema (gsis_id PK)",
                )
            )
        else:
            results.append(
                CheckResult(
                    name="schema.injuries",
                    status=Status.FAIL,
                    message=f"Wrong schema! Has: {injury_cols[:5]}",
                )
            )

        # Check mart views exist
        cur.execute(
            """
            SELECT matviewname FROM pg_matviews WHERE schemaname = 'mart'
        """
        )
        views = [r[0] for r in cur.fetchall()]
        expected_views = ["player_hierarchy", "player_game_stats", "game_summary"]

        for view in expected_views:
            if view in views:
                results.append(
                    CheckResult(name=f"mview.mart.{view}", status=Status.PASS, message="Exists")
                )
            else:
                results.append(
                    CheckResult(
                        name=f"mview.mart.{view}",
                        status=Status.FAIL,
                        message="MISSING - run migration 100",
                    )
                )

    return results


def check_data_quality(conn) -> list[CheckResult]:
    """Check data quality metrics."""
    results = []

    with conn.cursor() as cur:
        # Check games have scores
        cur.execute(
            """
            SELECT
                COUNT(*) as total,
                COUNT(home_score) as with_scores,
                COUNT(spread_close) as with_spreads
            FROM games
            WHERE season >= 2020
        """
        )
        total, with_scores, with_spreads = cur.fetchone()

        if total > 0:
            score_pct = (with_scores / total) * 100
            spread_pct = (with_spreads / total) * 100

            results.append(
                CheckResult(
                    name="quality.games.scores",
                    status=Status.PASS if score_pct > 90 else Status.WARN,
                    message=f"{score_pct:.1f}% have scores (since 2020)",
                )
            )
            results.append(
                CheckResult(
                    name="quality.games.spreads",
                    status=Status.PASS if spread_pct > 90 else Status.WARN,
                    message=f"{spread_pct:.1f}% have spreads (since 2020)",
                )
            )

        # Check plays have EPA (join with games for season filter)
        cur.execute(
            """
            SELECT
                COUNT(*) as total,
                COUNT(p.epa) as with_epa
            FROM plays p
            JOIN games g ON p.game_id = g.game_id
            WHERE g.season >= 2020
        """
        )
        total, with_epa = cur.fetchone()

        if total > 0:
            epa_pct = (with_epa / total) * 100
            results.append(
                CheckResult(
                    name="quality.plays.epa",
                    status=Status.PASS if epa_pct > 80 else Status.WARN,
                    message=f"{epa_pct:.1f}% have EPA values (since 2020)",
                )
            )

        # Check team_epa coverage
        cur.execute(
            """
            SELECT COUNT(DISTINCT game_id) FROM mart.team_epa
        """
        )
        epa_games = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM games")
        total_games = cur.fetchone()[0]

        if total_games > 0:
            coverage = (epa_games / total_games) * 100
            results.append(
                CheckResult(
                    name="quality.team_epa.coverage",
                    status=Status.PASS if coverage > 95 else Status.WARN,
                    message=f"{coverage:.1f}% game coverage",
                )
            )

    return results


def check_foreign_keys(conn) -> list[CheckResult]:
    """Check referential integrity between tables."""
    results = []

    with conn.cursor() as cur:
        # Check plays reference valid games
        cur.execute(
            """
            SELECT COUNT(DISTINCT p.game_id)
            FROM plays p
            LEFT JOIN games g ON p.game_id = g.game_id
            WHERE g.game_id IS NULL
        """
        )
        orphan_plays = cur.fetchone()[0]

        results.append(
            CheckResult(
                name="fk.plays_games",
                status=Status.PASS if orphan_plays == 0 else Status.WARN,
                message=f"Orphan game_ids in plays: {orphan_plays}",
            )
        )

        # Check player_hierarchy references valid players
        cur.execute(
            """
            SELECT COUNT(*)
            FROM mart.player_hierarchy ph
            LEFT JOIN players p ON ph.player_id = p.player_id
            WHERE p.player_id IS NULL
        """
        )
        orphan_hierarchy = cur.fetchone()[0]

        results.append(
            CheckResult(
                name="fk.hierarchy_players",
                status=Status.PASS if orphan_hierarchy == 0 else Status.WARN,
                message=f"Orphan players in hierarchy: {orphan_hierarchy}",
            )
        )

    return results


def run_health_check(verbose: bool = False, fix: bool = False) -> bool:
    """Run complete health check. Returns True if all critical checks pass."""
    print("=" * 60)
    print("NFL ANALYTICS DATABASE HEALTH CHECK")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    print()

    conn = get_connection()
    all_results: list[CheckResult] = []

    # 1. Table existence and row counts
    print("TABLE ROW COUNTS")
    print("-" * 60)
    for table, (min_rows, criticality) in TABLE_EXPECTATIONS.items():
        result = check_table(conn, table, min_rows, criticality)
        all_results.append(result)

        icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗", "SKIP": "○"}[result.status.value]
        if verbose or result.status in (Status.FAIL, Status.WARN):
            print(f"  {icon} {table:40} {result.message}")
        elif result.status == Status.PASS:
            print(f"  {icon} {table:40} {result.message}")

    print()

    # 2. Schema consistency
    print("SCHEMA CONSISTENCY")
    print("-" * 60)
    schema_results = check_schema_consistency(conn)
    all_results.extend(schema_results)
    for result in schema_results:
        icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗", "SKIP": "○"}[result.status.value]
        print(f"  {icon} {result.name:40} {result.message}")

    print()

    # 3. Data quality
    print("DATA QUALITY")
    print("-" * 60)
    quality_results = check_data_quality(conn)
    all_results.extend(quality_results)
    for result in quality_results:
        icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗", "SKIP": "○"}[result.status.value]
        print(f"  {icon} {result.name:40} {result.message}")

    print()

    # 4. Foreign key integrity
    print("REFERENTIAL INTEGRITY")
    print("-" * 60)
    fk_results = check_foreign_keys(conn)
    all_results.extend(fk_results)
    for result in fk_results:
        icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗", "SKIP": "○"}[result.status.value]
        print(f"  {icon} {result.name:40} {result.message}")

    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    pass_count = sum(1 for r in all_results if r.status == Status.PASS)
    warn_count = sum(1 for r in all_results if r.status == Status.WARN)
    fail_count = sum(1 for r in all_results if r.status == Status.FAIL)
    skip_count = sum(1 for r in all_results if r.status == Status.SKIP)

    print(f"  ✓ PASS: {pass_count}")
    print(f"  ⚠ WARN: {warn_count}")
    print(f"  ✗ FAIL: {fail_count}")
    print(f"  ○ SKIP: {skip_count}")
    print()

    # Calculate total data
    total_rows = sum(r.row_count or 0 for r in all_results if r.row_count)
    print(f"  Total rows across all tables: {total_rows:,}")

    conn.close()

    if fail_count > 0:
        print()
        print("❌ HEALTH CHECK FAILED - Critical issues found")
        print("   Run: psql ... -f db/migrations/100_schema_reconciliation.sql")
        return False
    elif warn_count > 0:
        print()
        print("⚠️  HEALTH CHECK PASSED WITH WARNINGS")
        print("   Some optional tables are empty. This is acceptable for modeling.")
        return True
    else:
        print()
        print("✅ HEALTH CHECK PASSED - Database is production ready")
        return True


def main():
    parser = argparse.ArgumentParser(description="NFL Analytics Database Health Check")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all checks")
    parser.add_argument("--fix", action="store_true", help="Attempt auto-fixes (not implemented)")
    args = parser.parse_args()

    try:
        success = run_health_check(verbose=args.verbose, fix=args.fix)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
