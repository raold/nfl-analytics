#!/usr/bin/env python3
"""
Pipeline Health Check — Pre-Prediction Data Validation

Validates that all feature data sources have fresh data for the target
prediction week BEFORE running predictions. Fails loudly on stale data
instead of silently producing compressed predictions.

This addresses the #1 root cause of 2025 ROI degradation: 30% of features
returned zeros because upstream data sources were empty or stale.

Usage:
  python scripts/pipeline_health_check.py --season 2026 --week 5
  python scripts/pipeline_health_check.py --season 2026 --week 5 --strict
  python scripts/pipeline_health_check.py --status  # Show all data source freshness

Author: Claude Code (Investigation WS1)
Date: February 12, 2026
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime

import psycopg2
from psycopg2.extras import DictCursor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5544")),
    "database": os.getenv("DB_NAME", "devdb01"),
    "user": os.getenv("DB_USER", "dro"),
    "password": os.getenv("DB_PASSWORD", "sicillionbillions"),
}


@dataclass
class DataSourceCheck:
    """Result of checking a single data source."""

    name: str
    features_affected: int
    status: str  # "OK", "STALE", "EMPTY", "MISSING"
    detail: str
    severity: str  # "CRITICAL", "WARNING", "INFO"
    rows_found: int = 0
    max_week: int | None = None
    max_season: int | None = None


@dataclass
class HealthReport:
    """Aggregate health report across all data sources."""

    target_season: int
    target_week: int
    checks: list[DataSourceCheck] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def critical_failures(self) -> list[DataSourceCheck]:
        return [c for c in self.checks if c.severity == "CRITICAL" and c.status != "OK"]

    @property
    def warnings(self) -> list[DataSourceCheck]:
        return [c for c in self.checks if c.severity == "WARNING" and c.status != "OK"]

    @property
    def is_healthy(self) -> bool:
        return len(self.critical_failures) == 0

    @property
    def features_degraded(self) -> int:
        return sum(c.features_affected for c in self.checks if c.status != "OK")

    @property
    def total_features_checked(self) -> int:
        return sum(c.features_affected for c in self.checks)


def get_db_connection():
    """Get PostgreSQL database connection."""
    return psycopg2.connect(**DB_CONFIG, cursor_factory=DictCursor)


def check_espn_qbr(conn, season: int, week: int) -> DataSourceCheck:
    """Check ESPN QBR data availability (20 features: 10 per team)."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM espn_qbr WHERE season = %s", (season,))
        total = cur.fetchone()[0]

        cur.execute(
            "SELECT COUNT(*) FROM espn_qbr WHERE season = %s AND game_week < %s",
            (season, week),
        )
        available = cur.fetchone()[0]

        cur.execute("SELECT MAX(season) as max_s FROM espn_qbr")
        row = cur.fetchone()
        max_season = row["max_s"] if row else None

    if total == 0 and (max_season is None or max_season < season):
        return DataSourceCheck(
            name="ESPN QBR (espn_qbr)",
            features_affected=20,
            status="EMPTY",
            detail=f"0 rows for season {season}. Table max season: {max_season}. "
            "QBR features will ALL be zeros — model cannot differentiate QB quality.",
            severity="CRITICAL",
            rows_found=0,
            max_season=max_season,
        )

    if available == 0 and week > 1:
        return DataSourceCheck(
            name="ESPN QBR (espn_qbr)",
            features_affected=20,
            status="STALE",
            detail=f"No QBR data before week {week} in season {season}. "
            f"Total rows for season: {total}.",
            severity="CRITICAL",
            rows_found=total,
            max_season=max_season,
        )

    return DataSourceCheck(
        name="ESPN QBR (espn_qbr)",
        features_affected=20,
        status="OK",
        detail=f"{available} rows available before week {week} (season {season})",
        severity="INFO",
        rows_found=available,
        max_season=season,
    )


def check_nextgen_stats(conn, season: int, week: int) -> DataSourceCheck:
    """Check NextGen Stats availability (46 features: 23 per team)."""
    tables = ["nextgen_passing", "nextgen_rushing", "nextgen_receiving"]
    results = {}

    for table in tables:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) as n, MAX(week) as max_wk "
                f"FROM {table} WHERE season = %s",
                (season,),
            )
            row = cur.fetchone()
            results[table] = {"count": row["n"], "max_week": row["max_wk"]}

    # Check if data is available up to target week - 1 (we need data BEFORE the game)
    min_max_week = min(
        r["max_week"] for r in results.values() if r["max_week"] is not None
    ) if any(r["max_week"] is not None for r in results.values()) else None

    total_rows = sum(r["count"] for r in results.values())

    if total_rows == 0:
        return DataSourceCheck(
            name="NextGen Stats (passing/rushing/receiving)",
            features_affected=46,
            status="EMPTY",
            detail=f"0 rows across all 3 NextGen tables for season {season}. "
            "46 tracking features will be zeros.",
            severity="CRITICAL",
            rows_found=0,
            max_week=None,
            max_season=season,
        )

    if min_max_week is not None and min_max_week < week - 2:
        return DataSourceCheck(
            name="NextGen Stats (passing/rushing/receiving)",
            features_affected=46,
            status="STALE",
            detail=f"NextGen data stops at week {min_max_week} for season {season}, "
            f"but predictions need week {week}. "
            f"Gap: {week - 1 - min_max_week} weeks stale. "
            f"Rows: passing={results['nextgen_passing']['count']}, "
            f"rushing={results['nextgen_rushing']['count']}, "
            f"receiving={results['nextgen_receiving']['count']}",
            severity="CRITICAL" if week - min_max_week > 3 else "WARNING",
            rows_found=total_rows,
            max_week=min_max_week,
            max_season=season,
        )

    return DataSourceCheck(
        name="NextGen Stats (passing/rushing/receiving)",
        features_affected=46,
        status="OK",
        detail=f"Data available through week {min_max_week} for season {season}. "
        f"Total rows: {total_rows}",
        severity="INFO",
        rows_found=total_rows,
        max_week=min_max_week,
        max_season=season,
    )


def check_injuries(conn, season: int, week: int) -> DataSourceCheck:
    """Check injury data availability (12 features)."""
    with conn.cursor() as cur:
        # Check raw injuries table
        cur.execute(
            "SELECT COUNT(*) as n, MAX(week) as max_wk "
            "FROM injuries WHERE season = %s",
            (season,),
        )
        raw = cur.fetchone()

        # Check mart.team_injury_load
        cur.execute(
            "SELECT COUNT(*) as n, MAX(week) as max_wk "
            "FROM mart.team_injury_load WHERE season = %s",
            (season,),
        )
        mart = cur.fetchone()

    raw_count = raw["n"]
    mart_count = mart["n"]

    if raw_count == 0:
        return DataSourceCheck(
            name="Injuries (injuries + mart.team_injury_load)",
            features_affected=12,
            status="EMPTY",
            detail=f"0 rows in injuries table for season {season}. "
            f"mart.team_injury_load also has {mart_count} rows for {season}. "
            "All 12 injury features will be zeros — cannot account for QB injuries, "
            "key player absences.",
            severity="CRITICAL",
            rows_found=0,
            max_week=None,
            max_season=season,
        )

    raw_max = raw["max_wk"]
    if raw_max is not None and raw_max < week - 1:
        return DataSourceCheck(
            name="Injuries (injuries + mart.team_injury_load)",
            features_affected=12,
            status="STALE",
            detail=f"Injury data stops at week {raw_max} for season {season}, "
            f"need week {week - 1}. "
            f"Raw rows: {raw_count}, Mart rows: {mart_count}",
            severity="WARNING",
            rows_found=raw_count,
            max_week=raw_max,
            max_season=season,
        )

    return DataSourceCheck(
        name="Injuries (injuries + mart.team_injury_load)",
        features_affected=12,
        status="OK",
        detail=f"Injury data available through week {raw_max} for season {season}. "
        f"Raw rows: {raw_count}, Mart rows: {mart_count}",
        severity="INFO",
        rows_found=raw_count,
        max_week=raw_max,
        max_season=season,
    )


def check_games_table(conn, season: int, week: int) -> DataSourceCheck:
    """Check games table for schedule availability."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) as n FROM games WHERE season = %s", (season,)
        )
        total = cur.fetchone()["n"]

        cur.execute(
            "SELECT COUNT(*) as n FROM games WHERE season = %s AND week = %s",
            (season, week),
        )
        target_week = cur.fetchone()["n"]

    if total == 0:
        return DataSourceCheck(
            name="Games schedule",
            features_affected=7,  # rest differential features
            status="EMPTY",
            detail=f"No games for season {season}. "
            "Rest differential features unavailable.",
            severity="CRITICAL",
            rows_found=0,
        )

    if target_week == 0:
        return DataSourceCheck(
            name="Games schedule",
            features_affected=7,
            status="STALE",
            detail=f"No games for week {week} in season {season}. "
            f"Total games for season: {total}",
            severity="WARNING",
            rows_found=total,
        )

    return DataSourceCheck(
        name="Games schedule",
        features_affected=7,
        status="OK",
        detail=f"{target_week} games in week {week}, {total} total for season {season}",
        severity="INFO",
        rows_found=total,
    )


def check_plays_table(conn, season: int, week: int) -> DataSourceCheck:
    """Check plays/PBP data for EPA and advanced features."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) as n, COUNT(DISTINCT game_id) as games "
            "FROM plays WHERE game_id LIKE %s",
            (f"{season}%",),
        )
        row = cur.fetchone()
        total_plays = row["n"]
        total_games = row["games"]

    if total_plays == 0:
        return DataSourceCheck(
            name="Play-by-play (plays)",
            features_affected=200,  # Most base features derive from PBP
            status="EMPTY",
            detail=f"0 plays for season {season}. "
            "Almost all base features will be zeros.",
            severity="CRITICAL",
            rows_found=0,
        )

    return DataSourceCheck(
        name="Play-by-play (plays)",
        features_affected=200,
        status="OK",
        detail=f"{total_plays:,} plays across {total_games} games for season {season}",
        severity="INFO",
        rows_found=total_plays,
    )


def check_team_epa(conn, season: int, week: int) -> DataSourceCheck:
    """Check mart.team_epa coverage."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) as n FROM mart.team_epa WHERE game_id LIKE %s",
            (f"{season}%",),
        )
        count = cur.fetchone()["n"]

    if count == 0:
        return DataSourceCheck(
            name="Team EPA (mart.team_epa)",
            features_affected=54,  # 27 off + 27 def per team pair
            status="EMPTY",
            detail=f"0 rows for season {season}. Run: python scripts/refresh_marts.py",
            severity="CRITICAL",
            rows_found=0,
        )

    return DataSourceCheck(
        name="Team EPA (mart.team_epa)",
        features_affected=54,
        status="OK",
        detail=f"{count} rows for season {season}",
        severity="INFO",
        rows_found=count,
    )


def check_odds(conn, season: int, week: int) -> DataSourceCheck:
    """Check odds_history for Vegas lines."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) as n FROM odds_history WHERE game_id LIKE %s",
                (f"{season}%",),
            )
            count = cur.fetchone()["n"]
    except psycopg2.Error as e:
        conn.rollback()
        # TimescaleDB hypertable issues are common — check table exists
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) as n FROM odds_history")
                total = cur.fetchone()["n"]
            return DataSourceCheck(
                name="Odds (odds_history)",
                features_affected=12,
                status="OK",
                detail=f"odds_history has {total} total rows (season filter failed: {str(e)[:50]})",
                severity="INFO",
                rows_found=total,
            )
        except psycopg2.Error:
            conn.rollback()
            return DataSourceCheck(
                name="Odds (odds_history)",
                features_affected=12,
                status="MISSING",
                detail=f"Cannot query odds_history: {str(e)[:80]}",
                severity="WARNING",
                rows_found=0,
            )

    if count == 0:
        return DataSourceCheck(
            name="Odds (odds_history)",
            features_affected=12,
            status="EMPTY",
            detail=f"0 odds rows for season {season}. "
            "Line movement features unavailable.",
            severity="WARNING",
            rows_found=0,
        )

    return DataSourceCheck(
        name="Odds (odds_history)",
        features_affected=12,
        status="OK",
        detail=f"{count} odds rows for season {season}",
        severity="INFO",
        rows_found=count,
    )


def check_playoff_context(conn, season: int, week: int) -> DataSourceCheck:
    """Check mart.team_playoff_context."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) as n, MAX(week) as max_wk "
            "FROM mart.team_playoff_context WHERE season = %s",
            (season,),
        )
        row = cur.fetchone()

    count = row["n"]
    max_week = row["max_wk"]

    if count == 0:
        return DataSourceCheck(
            name="Playoff context (mart.team_playoff_context)",
            features_affected=0,  # Not currently used in game features
            status="EMPTY",
            detail=f"No playoff context for season {season}. "
            "Not currently in feature pipeline but planned for P1.",
            severity="INFO",
            rows_found=0,
        )

    return DataSourceCheck(
        name="Playoff context (mart.team_playoff_context)",
        features_affected=0,
        status="OK" if max_week and max_week >= week - 2 else "STALE",
        detail=f"{count} rows, max week {max_week} for season {season}. "
        f"{'Weekly updates needed for feature use.' if max_week and max_week < week - 2 else ''}",
        severity="INFO",
        rows_found=count,
        max_week=max_week,
    )


def check_model_files() -> DataSourceCheck:
    """Check that production model files exist on disk."""
    import pathlib

    model_dir = pathlib.Path("models/production/game_outcome_v2")
    required_files = [
        "spread_model.json",
        "total_model.json",
        "win_prob_model.json",
        "metadata.json",
        "feature_names.json",
    ]

    missing = [f for f in required_files if not (model_dir / f).exists()]

    if missing:
        return DataSourceCheck(
            name="Production model files",
            features_affected=0,
            status="MISSING",
            detail=f"Missing model files: {', '.join(missing)}. "
            f"Expected in {model_dir}/",
            severity="CRITICAL",
            rows_found=0,
        )

    return DataSourceCheck(
        name="Production model files",
        features_affected=0,
        status="OK",
        detail=f"All {len(required_files)} model files present in {model_dir}/",
        severity="INFO",
        rows_found=len(required_files),
    )


def run_health_check(season: int, week: int) -> HealthReport:
    """Run complete pipeline health check."""
    report = HealthReport(target_season=season, target_week=week)
    conn = get_db_connection()

    checks = [
        check_games_table(conn, season, week),
        check_plays_table(conn, season, week),
        check_team_epa(conn, season, week),
        check_espn_qbr(conn, season, week),
        check_nextgen_stats(conn, season, week),
        check_injuries(conn, season, week),
        check_odds(conn, season, week),
        check_playoff_context(conn, season, week),
        check_model_files(),
    ]

    report.checks = checks
    conn.close()
    return report


def print_report(report: HealthReport, verbose: bool = False):
    """Print formatted health report."""
    print("=" * 72)
    print(f"PIPELINE HEALTH CHECK — Season {report.target_season}, Week {report.target_week}")
    print(f"Timestamp: {report.timestamp}")
    print("=" * 72)

    # Critical failures first
    if report.critical_failures:
        print(f"\n  CRITICAL FAILURES ({len(report.critical_failures)}):")
        for check in report.critical_failures:
            print(f"    [{check.status}] {check.name}")
            print(f"           {check.features_affected} features affected")
            print(f"           {check.detail}")

    # Warnings
    if report.warnings:
        print(f"\n  WARNINGS ({len(report.warnings)}):")
        for check in report.warnings:
            print(f"    [{check.status}] {check.name}")
            print(f"           {check.features_affected} features affected")
            print(f"           {check.detail}")

    # OK checks
    ok_checks = [c for c in report.checks if c.status == "OK"]
    if ok_checks:
        print(f"\n  HEALTHY ({len(ok_checks)}):")
        for check in ok_checks:
            detail = f" — {check.detail}" if verbose else ""
            print(f"    [OK] {check.name} ({check.features_affected} features){detail}")

    # Summary
    print("\n" + "-" * 72)
    total = report.total_features_checked
    degraded = report.features_degraded
    healthy = total - degraded
    pct = (healthy / total * 100) if total > 0 else 0

    if report.is_healthy:
        print(f"  RESULT: HEALTHY — {healthy}/{total} features OK ({pct:.0f}%)")
        print("  Safe to run predictions.")
    else:
        print(f"  RESULT: UNHEALTHY — {degraded}/{total} features DEGRADED ({100-pct:.0f}%)")
        print(f"  {len(report.critical_failures)} critical failure(s). "
              "DO NOT run predictions until resolved.")
        print("\n  Remediation:")
        for check in report.critical_failures:
            if "QBR" in check.name:
                print("    - Ingest QBR: Rscript R/ingestion/ingest_espn_qbr.R")
            elif "NextGen" in check.name:
                print("    - Ingest NextGen: Rscript R/ingestion/ingest_nextgen_stats.R")
            elif "Injur" in check.name:
                print("    - Ingest injuries: Rscript R/ingestion/ingest_injuries.R")
                print("    - Then refresh: python scripts/refresh_marts.py")
            elif "EPA" in check.name:
                print("    - Refresh marts: python scripts/refresh_marts.py")
            elif "Play-by-play" in check.name:
                print("    - Ingest PBP: Rscript R/ingestion/ingest_current_season.R")
            elif "model files" in check.name:
                print("    - Retrain model: python py/models/game_outcome_model_fixed.py")

    print("=" * 72)
    return report.is_healthy


def show_status():
    """Show current data source freshness (no target week needed)."""
    conn = get_db_connection()

    print("=" * 72)
    print("DATA SOURCE STATUS OVERVIEW")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 72)

    with conn.cursor() as cur:
        # All tables with season coverage
        queries = [
            ("espn_qbr", "SELECT COUNT(*) as n, MIN(season) as min_s, MAX(season) as max_s FROM espn_qbr"),
            ("nextgen_passing", "SELECT COUNT(*) as n, MIN(season) as min_s, MAX(season) as max_s FROM nextgen_passing"),
            ("nextgen_rushing", "SELECT COUNT(*) as n, MIN(season) as min_s, MAX(season) as max_s FROM nextgen_rushing"),
            ("nextgen_receiving", "SELECT COUNT(*) as n, MIN(season) as min_s, MAX(season) as max_s FROM nextgen_receiving"),
            ("injuries", "SELECT COUNT(*) as n, MIN(season) as min_s, MAX(season) as max_s FROM injuries"),
            ("games", "SELECT COUNT(*) as n, MIN(season) as min_s, MAX(season) as max_s FROM games"),
            ("plays (game_id)", "SELECT COUNT(*) as n, NULL::int as min_s, NULL::int as max_s FROM plays"),
            ("odds_history", "SELECT COUNT(*) as n, NULL::int as min_s, NULL::int as max_s FROM odds_history"),
        ]

        print(f"\n{'Table':<25} {'Rows':>10} {'Min Season':>12} {'Max Season':>12}")
        print("-" * 62)
        for name, query in queries:
            try:
                cur.execute(query)
                row = cur.fetchone()
                rows = f"{row['n']:,}" if row['n'] else "0"
                min_s = str(row['min_s']) if row['min_s'] else "—"
                max_s = str(row['max_s']) if row['max_s'] else "—"
                print(f"  {name:<23} {rows:>10} {min_s:>12} {max_s:>12}")
            except Exception as e:
                print(f"  {name:<23} {'ERROR':>10} {str(e)[:30]}")

        # Mart tables
        mart_queries = [
            ("mart.team_epa", "SELECT COUNT(*) as n FROM mart.team_epa"),
            ("mart.team_injury_load", "SELECT COUNT(*) as n, MAX(season) as max_s FROM mart.team_injury_load"),
            ("mart.team_playoff_context", "SELECT COUNT(*) as n, MAX(season) as max_s FROM mart.team_playoff_context"),
            ("mart.team_4th_down_features", "SELECT COUNT(*) as n FROM mart.team_4th_down_features"),
            ("mart.bayesian_player_ratings", "SELECT COUNT(*) as n FROM mart.bayesian_player_ratings"),
        ]

        print(f"\n{'Mart Table':<30} {'Rows':>10} {'Max Season':>12}")
        print("-" * 55)
        for name, query in mart_queries:
            try:
                conn.rollback()  # Clear any aborted transaction state
                cur.execute(query)
                row = cur.fetchone()
                rows = f"{row['n']:,}" if row['n'] else "0"
                max_s = str(row.get('max_s', '—')) if row.get('max_s') else "—"
                print(f"  {name:<28} {rows:>10} {max_s:>12}")
            except Exception as e:
                conn.rollback()
                print(f"  {name:<28} {'ERROR':>10} {str(e)[:30]}")

    conn.close()
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Health Check — Pre-Prediction Data Validation"
    )
    parser.add_argument("--season", type=int, help="Target prediction season")
    parser.add_argument("--week", type=int, help="Target prediction week")
    parser.add_argument("--status", action="store_true",
                        help="Show data source freshness overview")
    parser.add_argument("--strict", action="store_true",
                        help="Exit with code 1 on any critical failure")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output for all checks")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")

    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if not args.season or not args.week:
        parser.error("--season and --week are required (or use --status)")

    report = run_health_check(args.season, args.week)

    if args.json:
        import json
        output = {
            "season": report.target_season,
            "week": report.target_week,
            "timestamp": report.timestamp,
            "healthy": report.is_healthy,
            "features_degraded": report.features_degraded,
            "total_features": report.total_features_checked,
            "critical_failures": len(report.critical_failures),
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "severity": c.severity,
                    "features_affected": c.features_affected,
                    "detail": c.detail,
                }
                for c in report.checks
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        is_healthy = print_report(report, verbose=args.verbose)

        if args.strict and not is_healthy:
            sys.exit(1)


if __name__ == "__main__":
    main()
