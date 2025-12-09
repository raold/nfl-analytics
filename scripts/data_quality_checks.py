#!/usr/bin/env python3
"""
Data Quality Checks for NFL Analytics Database

Runs comprehensive data quality validation including:
- Orphan record detection (plays without games, rosters without players)
- Future data leakage detection (scores on unplayed games)
- Duplicate detection (primary key violations)
- Referential integrity checks
- Data freshness checks

Usage:
  python scripts/data_quality_checks.py              # Run all checks
  python scripts/data_quality_checks.py --fix        # Run checks and auto-fix where possible
  python scripts/data_quality_checks.py --check orphans  # Run specific check

Author: Claude Code (Opus 4.5)
Date: December 7, 2025
"""

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

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


@dataclass
class CheckResult:
    """Result of a data quality check."""
    name: str
    passed: bool
    count: int
    message: str
    fixable: bool = False


def get_db_connection():
    """Get PostgreSQL database connection."""
    return psycopg2.connect(**DB_CONFIG, cursor_factory=DictCursor)


def check_orphan_plays(conn) -> CheckResult:
    """Check for plays that reference non-existent games."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM plays p
            LEFT JOIN games g ON p.game_id = g.game_id
            WHERE g.game_id IS NULL
        """)
        count = cur.fetchone()[0]

    return CheckResult(
        name="Orphan Plays",
        passed=count == 0,
        count=count,
        message=f"{count} plays reference non-existent games" if count > 0 else "All plays have valid game references",
        fixable=True,
    )


def check_orphan_participation(conn) -> CheckResult:
    """Check for participation records without matching plays."""
    with conn.cursor() as cur:
        # participation uses nflverse_game_id, plays uses game_id
        cur.execute("""
            SELECT COUNT(*) FROM participation par
            LEFT JOIN plays p ON par.nflverse_game_id = p.game_id AND par.play_id = p.play_id
            WHERE p.play_id IS NULL
        """)
        count = cur.fetchone()[0]

    return CheckResult(
        name="Orphan Participation",
        passed=count == 0,
        count=count,
        message=f"{count} participation records without matching plays" if count > 0 else "All participation records valid",
        fixable=True,
    )


def check_future_scores(conn) -> CheckResult:
    """Check for games with scores that haven't happened yet (data leakage)."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM games
            WHERE kickoff > NOW()
              AND (home_score IS NOT NULL OR away_score IS NOT NULL)
        """)
        count = cur.fetchone()[0]

    return CheckResult(
        name="Future Scores (Data Leakage)",
        passed=count == 0,
        count=count,
        message=f"CRITICAL: {count} future games have scores!" if count > 0 else "No data leakage detected",
        fixable=True,
    )


def check_duplicate_games(conn) -> CheckResult:
    """Check for duplicate game_id entries."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT game_id, COUNT(*) as cnt
            FROM games
            GROUP BY game_id
            HAVING COUNT(*) > 1
        """)
        duplicates = cur.fetchall()

    count = len(duplicates)
    return CheckResult(
        name="Duplicate Games",
        passed=count == 0,
        count=count,
        message=f"{count} duplicate game_ids found" if count > 0 else "No duplicate games",
        fixable=False,
    )


def check_duplicate_plays(conn) -> CheckResult:
    """Check for duplicate play entries."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT game_id, play_id, COUNT(*) as cnt
            FROM plays
            GROUP BY game_id, play_id
            HAVING COUNT(*) > 1
            LIMIT 10
        """)
        duplicates = cur.fetchall()

    count = len(duplicates)
    return CheckResult(
        name="Duplicate Plays",
        passed=count == 0,
        count=count,
        message=f"{count}+ duplicate play entries found" if count > 0 else "No duplicate plays",
        fixable=False,
    )


def check_roster_player_integrity(conn) -> CheckResult:
    """Check for roster entries referencing non-existent players (informational only)."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM rosters_weekly r
            LEFT JOIN players p ON r.gsis_id = p.player_id
            WHERE p.player_id IS NULL AND r.gsis_id IS NOT NULL
        """)
        count = cur.fetchone()[0]

    # This is informational - rosters_weekly doesn't have FK constraint
    return CheckResult(
        name="Roster-Player Links",
        passed=True,  # Not a failure, just informational
        count=count,
        message=f"{count} roster entries without player records (informational)" if count > 0 else "All roster players have player records",
        fixable=False,
    )


def check_games_with_missing_kickoff(conn) -> CheckResult:
    """Check for games missing kickoff time."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM games
            WHERE kickoff IS NULL
        """)
        count = cur.fetchone()[0]

    return CheckResult(
        name="Missing Kickoff Times",
        passed=count == 0,
        count=count,
        message=f"{count} games missing kickoff time" if count > 0 else "All games have kickoff times",
        fixable=False,
    )


def check_data_freshness(conn) -> CheckResult:
    """Check if data is reasonably fresh."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT MAX(kickoff) as latest_game FROM games
            WHERE home_score IS NOT NULL
        """)
        latest = cur.fetchone()[0]

    if latest is None:
        return CheckResult(
            name="Data Freshness",
            passed=False,
            count=0,
            message="No completed games found!",
            fixable=False,
        )

    days_old = (datetime.now(latest.tzinfo) - latest).days
    passed = days_old <= 14  # Within 2 weeks is considered fresh

    return CheckResult(
        name="Data Freshness",
        passed=passed,
        count=days_old,
        message=f"Latest game: {days_old} days ago ({latest.strftime('%Y-%m-%d')})",
        fixable=False,
    )


def check_negative_scores(conn) -> CheckResult:
    """Check for negative scores (data corruption)."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM games
            WHERE home_score < 0 OR away_score < 0
        """)
        count = cur.fetchone()[0]

    return CheckResult(
        name="Negative Scores",
        passed=count == 0,
        count=count,
        message=f"CRITICAL: {count} games have negative scores!" if count > 0 else "All scores valid",
        fixable=False,
    )


def fix_orphan_plays(conn) -> int:
    """Delete orphan plays."""
    with conn.cursor() as cur:
        cur.execute("""
            DELETE FROM plays p
            WHERE NOT EXISTS (
                SELECT 1 FROM games g WHERE g.game_id = p.game_id
            )
        """)
        count = cur.rowcount
    conn.commit()
    return count


def fix_orphan_participation(conn) -> int:
    """Delete orphan participation records."""
    with conn.cursor() as cur:
        cur.execute("""
            DELETE FROM participation par
            WHERE NOT EXISTS (
                SELECT 1 FROM plays p
                WHERE p.game_id = par.nflverse_game_id AND p.play_id = par.play_id
            )
        """)
        count = cur.rowcount
    conn.commit()
    return count


def fix_future_scores(conn) -> int:
    """Clear scores from future games."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE games
            SET home_score = NULL, away_score = NULL
            WHERE kickoff > NOW()
              AND (home_score IS NOT NULL OR away_score IS NOT NULL)
        """)
        count = cur.rowcount
    conn.commit()
    return count


# Map of check names to functions
CHECKS = {
    "orphan_plays": check_orphan_plays,
    "orphan_participation": check_orphan_participation,
    "future_scores": check_future_scores,
    "duplicate_games": check_duplicate_games,
    "duplicate_plays": check_duplicate_plays,
    "roster_integrity": check_roster_player_integrity,
    "missing_kickoff": check_games_with_missing_kickoff,
    "data_freshness": check_data_freshness,
    "negative_scores": check_negative_scores,
}

FIXES = {
    "orphan_plays": fix_orphan_plays,
    "orphan_participation": fix_orphan_participation,
    "future_scores": fix_future_scores,
}


def run_all_checks(fix: bool = False) -> list[CheckResult]:
    """Run all data quality checks."""
    conn = get_db_connection()
    results = []

    print("=" * 70)
    print("DATA QUALITY CHECKS")
    print("=" * 70)

    for name, check_fn in CHECKS.items():
        result = check_fn(conn)
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        icon = "" if result.passed else ""
        print(f"\n{icon} [{status}] {result.name}")
        print(f"   {result.message}")

        # Auto-fix if requested and fixable
        if fix and not result.passed and result.fixable and name in FIXES:
            fixed_count = FIXES[name](conn)
            print(f"    Fixed: {fixed_count} records")

    conn.close()

    # Summary
    print("\n" + "=" * 70)
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    print(f"SUMMARY: {passed} passed, {failed} failed")

    if failed > 0:
        print("\nFailed checks:")
        for r in results:
            if not r.passed:
                fix_note = " (fixable with --fix)" if r.fixable else ""
                print(f"  - {r.name}: {r.count} issues{fix_note}")

    return results


def run_single_check(check_name: str, fix: bool = False):
    """Run a single check by name."""
    if check_name not in CHECKS:
        print(f"Unknown check: {check_name}")
        print(f"Available checks: {', '.join(CHECKS.keys())}")
        return

    conn = get_db_connection()
    result = CHECKS[check_name](conn)

    status = "PASS" if result.passed else "FAIL"
    print(f"[{status}] {result.name}")
    print(f"   {result.message}")

    if fix and not result.passed and result.fixable and check_name in FIXES:
        fixed_count = FIXES[check_name](conn)
        print(f"   Fixed: {fixed_count} records")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="NFL Analytics Data Quality Checks")
    parser.add_argument("--fix", action="store_true", help="Auto-fix issues where possible")
    parser.add_argument("--check", type=str, help="Run specific check only")

    args = parser.parse_args()

    if args.check:
        run_single_check(args.check, args.fix)
    else:
        run_all_checks(args.fix)


if __name__ == "__main__":
    main()
