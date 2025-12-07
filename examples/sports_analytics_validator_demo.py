"""
Sports Analytics Validator - Demonstration

This script demonstrates the Sports Analytics Validator capabilities
with real-world examples of common NFL data validation scenarios.

Run:
    python examples/sports_analytics_validator_demo.py
"""

import pandas as pd

from py.utils.sports_analytics_validator import (
    SportsAnalyticsValidator,
)


def demo_position_eligibility():
    """Demo: Position eligibility validation"""
    print("\n" + "=" * 80)
    print("DEMO 1: Position Eligibility Validation")
    print("=" * 80)

    # Create test data with position violations
    data = [
        # VALID: QB with passing stats
        {
            "player_position": "QB",
            "stat_category": "passing",
            "stat_yards": 350,
            "player_display_name": "P. Mahomes",
        },
        # VALID: QB with rushing stats (scrambles)
        {
            "player_position": "QB",
            "stat_category": "rushing",
            "stat_yards": 50,
            "player_display_name": "L. Jackson",
        },
        # ERROR: QB with excessive rushing yards
        {
            "player_position": "QB",
            "stat_category": "rushing",
            "stat_yards": 250,
            "player_display_name": "Invalid QB",
        },
        # ERROR: RB with passing stats (impossible)
        {
            "player_position": "RB",
            "stat_category": "passing",
            "stat_yards": 100,
            "player_display_name": "Invalid RB",
        },
        # VALID: RB with rushing stats
        {
            "player_position": "RB",
            "stat_category": "rushing",
            "stat_yards": 150,
            "player_display_name": "D. Henry",
        },
    ]

    df = pd.DataFrame(data)
    validator = SportsAnalyticsValidator()
    report = validator.validate_dataframe(df, dataset_name="position_eligibility_test")

    print(f"\nRows checked: {len(df)}")
    print(f"Issues found: {len(report.issues)}\n")

    for issue in report.issues:
        print(f"[{issue.severity.value}] Row {issue.row_index}: {issue.message}")
        print(f"  Player: {df.iloc[issue.row_index]['player_display_name']}")
        print(f"  Details: {issue.details}\n")


def demo_stat_ranges():
    """Demo: Stat range validation"""
    print("\n" + "=" * 80)
    print("DEMO 2: Stat Range Validation")
    print("=" * 80)

    data = [
        # VALID: Normal passing game
        {
            "stat_category": "passing",
            "stat_yards": 350,
            "stat_attempts": 45,
            "label": "Normal passing",
        },
        # ERROR: Impossible passing yards
        {
            "stat_category": "passing",
            "stat_yards": 700,
            "stat_attempts": 50,
            "label": "Impossible passing",
        },
        # ERROR: Excessive attempts
        {
            "stat_category": "passing",
            "stat_yards": 400,
            "stat_attempts": 90,
            "label": "Too many attempts",
        },
        # VALID: Big rushing game
        {
            "stat_category": "rushing",
            "stat_yards": 200,
            "stat_attempts": 30,
            "label": "Big rushing game",
        },
        # ERROR: Impossible rushing yards
        {
            "stat_category": "rushing",
            "stat_yards": 350,
            "stat_attempts": 35,
            "label": "Impossible rushing",
        },
    ]

    df = pd.DataFrame(data)
    validator = SportsAnalyticsValidator()
    report = validator.validate_dataframe(df, dataset_name="stat_range_test")

    print(f"\nRows checked: {len(df)}")
    print(f"Issues found: {len(report.issues)}\n")

    for issue in report.issues:
        print(f"[{issue.severity.value}] {df.iloc[issue.row_index]['label']}")
        print(f"  {issue.message}")
        print(f"  Details: {issue.details}\n")


def demo_schedule_constraints():
    """Demo: Schedule constraint validation"""
    print("\n" + "=" * 80)
    print("DEMO 3: Schedule Constraint Validation")
    print("=" * 80)

    data = [
        # VALID: Regular season weeks
        {"week": 1, "season": 2024, "label": "Week 1"},
        {"week": 10, "season": 2024, "label": "Week 10"},
        {"week": 18, "season": 2024, "label": "Week 18"},
        # VALID: Playoff weeks
        {"week": 19, "season": 2024, "label": "Wild Card"},
        {"week": 22, "season": 2024, "label": "Super Bowl"},
        # ERROR: Invalid week numbers
        {"week": 0, "season": 2024, "label": "Invalid Week 0"},
        {"week": 25, "season": 2024, "label": "Invalid Week 25"},
    ]

    df = pd.DataFrame(data)
    validator = SportsAnalyticsValidator()
    report = validator.validate_dataframe(df, dataset_name="schedule_test")

    print(f"\nRows checked: {len(df)}")
    print(f"Issues found: {len(report.issues)}\n")

    for issue in report.issues:
        print(f"[{issue.severity.value}] {df.iloc[issue.row_index]['label']}")
        print(f"  {issue.message}")
        print(f"  Details: {issue.details}\n")


def demo_team_abbreviations():
    """Demo: Team abbreviation validation"""
    print("\n" + "=" * 80)
    print("DEMO 4: Team Abbreviation Validation")
    print("=" * 80)

    data = [
        # VALID: Current teams
        {"home_team": "KC", "away_team": "LAC", "season": 2024, "label": "KC vs LAC (2024)"},
        {"home_team": "LAR", "away_team": "SF", "season": 2024, "label": "LAR vs SF (2024)"},
        # ERROR: Ambiguous LA
        {"home_team": "LA", "away_team": "DEN", "season": 2024, "label": "LA vs DEN (ambiguous)"},
        # ERROR: Outdated abbreviation
        {
            "home_team": "SD",
            "away_team": "KC",
            "season": 2024,
            "label": "SD vs KC (2024 - relocated)",
        },
        # VALID: Historical abbreviation (before relocation)
        {"home_team": "SD", "away_team": "KC", "season": 2016, "label": "SD vs KC (2016 - valid)"},
        # ERROR: Invalid team
        {"home_team": "XXX", "away_team": "KC", "season": 2024, "label": "Invalid team XXX"},
    ]

    df = pd.DataFrame(data)
    validator = SportsAnalyticsValidator()
    report = validator.validate_dataframe(df, dataset_name="team_abbreviation_test")

    print(f"\nRows checked: {len(df)}")
    print(f"Issues found: {len(report.issues)}\n")

    for issue in report.issues:
        print(f"[{issue.severity.value}] {df.iloc[issue.row_index]['label']}")
        print(f"  {issue.message}")
        print(f"  Details: {issue.details}\n")


def demo_sql_query_validation():
    """Demo: SQL query validation"""
    print("\n" + "=" * 80)
    print("DEMO 5: SQL Query Validation")
    print("=" * 80)

    queries = [
        # ERROR: Invalid week
        ("Invalid week", "SELECT * FROM games WHERE week = 25"),
        # ERROR: Ambiguous LA
        ("Ambiguous LA", "SELECT * FROM games WHERE home_team = 'LA'"),
        # ERROR: game_id in player_game_stats
        ("game_id error", "SELECT pgs.game_id FROM mart.player_game_stats pgs"),
        # ERROR: Incorrect column name
        ("Wrong column", "SELECT game_date FROM games WHERE week = 1"),
        # VALID: Correct query
        (
            "Valid query",
            """
            SELECT g.kickoff, g.spread_close
            FROM games g
            WHERE g.season = 2024
            AND g.week BETWEEN 1 AND 18
            AND g.home_team = 'KC'
            """,
        ),
    ]

    validator = SportsAnalyticsValidator()

    for label, query in queries:
        print(f"\n{label}:")
        print(f"Query: {query[:80]}...")

        issues = validator.check_query_logic(query)

        if issues:
            print(f"Issues found: {len(issues)}")
            for issue in issues:
                print(f"  [{issue.severity.value}] {issue.message}")
                if issue.details:
                    print(f"    {issue.details}")
        else:
            print("  ✅ No issues found")


def demo_full_validation_report():
    """Demo: Complete validation report"""
    print("\n" + "=" * 80)
    print("DEMO 6: Full Validation Report")
    print("=" * 80)

    # Create dataset with multiple types of violations
    data = [
        # VALID rows
        {
            "player_position": "QB",
            "stat_category": "passing",
            "stat_yards": 350,
            "week": 10,
            "season": 2024,
            "home_team": "KC",
            "away_team": "LAC",
            "home_score": 28,
            "away_score": 24,
        },
        # Multiple errors in one row
        {
            "player_position": "RB",
            "stat_category": "passing",
            "stat_yards": 700,
            "week": 25,
            "season": 2024,
            "home_team": "LA",
            "away_team": "DEN",
            "home_score": -7,
            "away_score": 14,
        },
        # Warning: High score
        {
            "player_position": "WR",
            "stat_category": "receiving",
            "stat_yards": 150,
            "week": 12,
            "season": 2024,
            "home_team": "BUF",
            "away_team": "MIA",
            "home_score": 75,
            "away_score": 14,
        },
    ]

    df = pd.DataFrame(data)
    validator = SportsAnalyticsValidator()
    report = validator.validate_dataframe(df, dataset_name="comprehensive_test.csv")

    print(report.generate_report())


if __name__ == "__main__":
    print("=" * 80)
    print("SPORTS ANALYTICS VALIDATOR - DEMONSTRATION")
    print("=" * 80)
    print("\nThis demo shows the validator catching common NFL data violations")

    # Run all demos
    demo_position_eligibility()
    demo_stat_ranges()
    demo_schedule_constraints()
    demo_team_abbreviations()
    demo_sql_query_validation()
    demo_full_validation_report()

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nFor more information, see:")
    print("  - docs/SPORTS_ANALYTICS_VALIDATOR_GUIDE.md")
    print("  - tests/unit/test_sports_analytics_validator.py")
    print("=" * 80)
