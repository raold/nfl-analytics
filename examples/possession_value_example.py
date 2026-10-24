#!/usr/bin/env python3
"""
Example Usage of Possession Value Module
=========================================

Demonstrates how to use the possession_value module for:
1. Computing expected points from field position/situation
2. Calculating time-adjusted possession values
3. Extracting team-level possession metrics from database
4. Batch feature extraction for game predictions

Author: Claude Code (Sonnet 4.5)
Date: November 27, 2025
"""

import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import psycopg2

from nfl_analytics.features.possession_value import (
    compute_possession_value,
    compute_possession_value_features,
    get_expected_points,
    get_team_possession_value_features,
    time_value_factor,
    urgency_factor,
)


def example_1_expected_points():
    """Example 1: Basic expected points calculation."""
    print("\n" + "=" * 80)
    print("Example 1: Expected Points by Field Position and Situation")
    print("=" * 80)

    situations = [
        (50, 1, 10, "1st & 10 at midfield"),
        (20, 1, 10, "1st & 10 at opponent 20 (red zone)"),
        (30, 3, 8, "3rd & 8 at opponent 30"),
        (99, 4, 15, "4th & 15 at own 1-yard line"),
        (1, 2, 1, "2nd & goal from 1-yard line"),
    ]

    for field_pos, down, distance, desc in situations:
        ep = get_expected_points(field_pos, down, distance)
        print(f"{desc:40s} → EP: {ep:+.2f}")


def example_2_time_and_urgency():
    """Example 2: Time decay and urgency adjustments."""
    print("\n" + "=" * 80)
    print("Example 2: Time Decay and Urgency Multipliers")
    print("=" * 80)

    print("\nTime Value Decay:")
    times = [3600, 1800, 900, 300, 120]
    for seconds in times:
        factor = time_value_factor(seconds)
        mins = seconds // 60
        print(f"  {mins:2d} min remaining → {factor:.2%} of full value")

    print("\nUrgency Multipliers (4th Quarter):")
    scenarios = [
        (-14, 600, "Down 14, 10 min left"),
        (-7, 300, "Down 7, 5 min left"),
        (0, 180, "Tied, 3 min left"),
        (10, 240, "Up 10, 4 min left"),
    ]
    for score_diff, time, desc in scenarios:
        urgency = urgency_factor(score_diff, time)
        print(f"  {desc:25s} → {urgency:.2f}x multiplier")


def example_3_composite_possession_value():
    """Example 3: Full possession value computation."""
    print("\n" + "=" * 80)
    print("Example 3: Composite Possession Value")
    print("=" * 80)

    situations = [
        (50, 3600, 0, 1, 10, "Normal: 1st & 10 at midfield, full game, tied"),
        (20, 300, -7, 3, 5, "Urgent: 3rd & 5 at opp 20, down 7, 5 min left"),
        (99, 120, -10, 4, 15, "Desperate: 4th & 15 at own 1, down 10, 2 min left"),
        (10, 600, 3, 2, 3, "Conservative: 2nd & 3 at opp 10, up 3, 10 min left"),
    ]

    for fp, time, score, down, dist, desc in situations:
        pv = compute_possession_value(fp, time, score, down, dist)
        base_ep = get_expected_points(fp, down, dist)
        time_factor = time_value_factor(time)
        urg = urgency_factor(score, time)

        print(f"\n{desc}")
        print(f"  Base EP: {base_ep:+.2f} | Time Factor: {time_factor:.2f} | Urgency: {urg:.2f}x")
        print(f"  → Final Possession Value: {pv:+.2f} EP")


def example_4_team_features():
    """Example 4: Team-level possession metrics from database."""
    print("\n" + "=" * 80)
    print("Example 4: Team Possession Metrics (Database)")
    print("=" * 80)

    # Connect to database
    conn = psycopg2.connect(
        host="localhost",
        port=5544,
        database="devdb01",
        user="dro",
        password="sicillionbillions",  # pragma: allowlist secret
    )

    # Get features for multiple teams
    teams = ["KC", "BUF", "SF", "DAL", "BAL"]
    season = datetime.now().year if datetime.now().month >= 9 else datetime.now().year - 1
    week = 10

    print(f"\nTeam Possession Metrics (Season {season}, Before Week {week}):\n")
    print(
        f"{'Team':<6} {'Avg Field Pos':>14} {'Avg EP/Poss':>12} {'Pace Factor':>12} {'Poss/Game':>10}"
    )
    print("-" * 60)

    for team in teams:
        features = get_team_possession_value_features(conn, team, season, week, n_games=4)
        print(
            f"{team:<6} {features['avg_starting_field_position']:>14.1f} "
            f"{features['avg_possession_value']:>12.2f} "
            f"{features['pace_factor']:>12.2f} "
            f"{features['possessions_per_game']:>10.1f}"
        )

    conn.close()


def example_5_batch_computation():
    """Example 5: Batch feature extraction for game predictions."""
    print("\n" + "=" * 80)
    print("Example 5: Batch Feature Extraction")
    print("=" * 80)

    # Connect to database
    conn = psycopg2.connect(
        host="localhost",
        port=5544,
        database="devdb01",
        user="dro",
        password="sicillionbillions",  # pragma: allowlist secret
    )

    # Get sample games
    query = """
    SELECT game_id, season, week, home_team, away_team
    FROM games
    WHERE season = datetime.now().year if datetime.now().month >= 9 else datetime.now().year - 1 AND week = 9
    ORDER BY game_id
    LIMIT 5
    """
    games_df = pd.read_sql(query, conn)

    # Compute possession value features
    result = compute_possession_value_features(conn, games_df)

    # Display results
    print(f"\nProcessed {len(result)} games:\n")
    display_cols = [
        "game_id",
        "home_team",
        "away_team",
        "home_avg_possession_value",
        "away_avg_possession_value",
        "home_pace_factor",
        "away_pace_factor",
    ]

    for _, row in result[display_cols].iterrows():
        print(f"{row['game_id']:<20s} {row['home_team']:>3s} vs {row['away_team']:<3s}")
        print(
            f"  Home: EP/Poss={row['home_avg_possession_value']:>5.2f}, Pace={row['home_pace_factor']:>4.2f}x"
        )
        print(
            f"  Away: EP/Poss={row['away_avg_possession_value']:>5.2f}, Pace={row['away_pace_factor']:>4.2f}x"
        )
        print()

    conn.close()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("POSSESSION VALUE MODULE - COMPREHENSIVE EXAMPLES")
    print("=" * 80)

    # Run all examples
    example_1_expected_points()
    example_2_time_and_urgency()
    example_3_composite_possession_value()
    example_4_team_features()
    example_5_batch_computation()

    print("\n" + "=" * 80)
    print("All examples complete!")
    print("=" * 80)
