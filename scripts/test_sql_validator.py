#!/usr/bin/env python3
"""
Manual test script for SQL validator

Usage:
    python scripts/test_sql_validator.py
"""

import sys
from pathlib import Path

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_PROJECT_ROOT))

from py.utils.sql_validator import SQLValidator


def test_cases():
    """Run test cases for SQL validator."""
    validator = SQLValidator()

    tests = [
        {
            "name": "Wrong: game_id in player_game_stats",
            "query": "SELECT pgs.game_id FROM mart.player_game_stats pgs",
            "should_fail": True,
        },
        {
            "name": "Correct: (season, week) join",
            "query": """
                SELECT pgs.player_id, g.game_id
                FROM mart.player_game_stats pgs
                LEFT JOIN games g ON g.season = pgs.season AND g.week = pgs.week
            """,
            "should_fail": False,
        },
        {
            "name": "Wrong: game_date instead of kickoff",
            "query": "SELECT g.game_date FROM games g",
            "should_fail": True,
        },
        {
            "name": "Wrong: spread_line instead of spread_close",
            "query": "SELECT g.spread_line FROM games g",
            "should_fail": True,
        },
        {
            "name": "Correct: canonical pattern from production",
            "query": """
                SELECT
                    pgs.player_id,
                    pgs.season,
                    pgs.week,
                    pgs.stat_yards,
                    g.spread_close,
                    ph.current_team
                FROM mart.player_game_stats pgs
                JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
                LEFT JOIN games g ON
                    g.season = pgs.season
                    AND g.week = pgs.week
                    AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
                WHERE pgs.stat_category = 'passing'
            """,
            "should_fail": False,
        },
    ]

    print("=" * 80)
    print("SQL VALIDATOR TEST SUITE")
    print("=" * 80)
    print()

    passed = 0
    failed = 0

    for test in tests:
        print(f"Test: {test['name']}")
        print("-" * 80)

        errors = validator.validate_query(test["query"])
        has_errors = len(errors) > 0

        if has_errors == test["should_fail"]:
            print("✅ PASSED")
            passed += 1
        else:
            print("❌ FAILED")
            failed += 1

        if errors:
            print("\nErrors found:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("\nNo errors found.")

        print()

    print("=" * 80)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("=" * 80)

    validator.close()

    return failed == 0


if __name__ == "__main__":
    success = test_cases()
    sys.exit(0 if success else 1)
