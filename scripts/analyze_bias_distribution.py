#!/usr/bin/env python3
"""
Systematic Bias Investigation: Distribution Shift Analysis

Compare training (2018-2021) vs validation (2022-2024) distributions
to identify potential causes of systematic under-prediction.
"""


import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from scipy import stats


# Database connection
def get_db_connection():
    return psycopg2.connect(
        host="localhost", port=5544, dbname="devdb01", user="dro", password="sicillionbillions"
    )


def analyze_passing_yards_distribution():
    """Compare passing yards distribution across time periods."""

    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # Get passing yards by season
    query = """
    SELECT
        season,
        week,
        stat_yards as passing_yards,
        stat_attempts
    FROM mart.player_game_stats
    WHERE stat_category = 'passing'
        AND stat_yards IS NOT NULL
        AND stat_attempts >= 10
        AND season BETWEEN 2018 AND 2024
    ORDER BY season, week
    """

    cursor.execute(query)
    data = pd.DataFrame(cursor.fetchall())

    # Split into training and validation
    train = data[(data["season"] >= 2018) & (data["season"] <= 2021)]
    val = data[(data["season"] >= 2022) & (data["season"] <= 2024)]

    print("=" * 80)
    print("PASSING YARDS DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print()

    # Overall statistics
    print("1. OVERALL STATISTICS")
    print("-" * 80)
    print("Training (2018-2021):")
    print(f"  Games: {len(train):,}")
    print(f"  Mean: {train['passing_yards'].mean():.2f} yards")
    print(f"  Median: {train['passing_yards'].median():.2f} yards")
    print(f"  Std Dev: {train['passing_yards'].std():.2f} yards")
    print()
    print("Validation (2022-2024):")
    print(f"  Games: {len(val):,}")
    print(f"  Mean: {val['passing_yards'].mean():.2f} yards")
    print(f"  Median: {val['passing_yards'].median():.2f} yards")
    print(f"  Std Dev: {val['passing_yards'].std():.2f} yards")
    print()
    print("DIFFERENCE:")
    print(
        f"  Mean difference: {val['passing_yards'].mean() - train['passing_yards'].mean():.2f} yards"
    )
    print(
        f"  ({(val['passing_yards'].mean() / train['passing_yards'].mean() - 1) * 100:.2f}% increase)"
    )
    print()

    # Statistical test
    t_stat, p_value = stats.ttest_ind(val["passing_yards"], train["passing_yards"])
    print(f"T-test: t={t_stat:.4f}, p={p_value:.4e}")
    if p_value < 0.05:
        print("  ✓ SIGNIFICANT difference between periods")
    else:
        print("  ✗ No significant difference")
    print()

    # By season breakdown
    print("2. BY SEASON BREAKDOWN")
    print("-" * 80)
    for season in range(2018, 2025):
        season_data = data[data["season"] == season]
        print(
            f"Season {season}: mean={season_data['passing_yards'].mean():.2f}, "
            f"n={len(season_data):,}"
        )
    print()

    # Percentiles
    print("3. PERCENTILE COMPARISON")
    print("-" * 80)
    percentiles = [10, 25, 50, 75, 90]
    print(f"{'Percentile':<12} {'Training':>12} {'Validation':>12} {'Difference':>12}")
    print("-" * 60)
    for p in percentiles:
        train_p = np.percentile(train["passing_yards"], p)
        val_p = np.percentile(val["passing_yards"], p)
        diff = val_p - train_p
        print(f"{p}th{'':<9} {train_p:>12.1f} {val_p:>12.1f} {diff:>12.1f}")
    print()

    # Attempts analysis (check if game pace changed)
    print("4. PASSING ATTEMPTS ANALYSIS")
    print("-" * 80)
    print("Training (2018-2021):")
    print(f"  Mean attempts: {train['stat_attempts'].mean():.2f}")
    print("Validation (2022-2024):")
    print(f"  Mean attempts: {val['stat_attempts'].mean():.2f}")
    print(
        f"Difference: {val['stat_attempts'].mean() - train['stat_attempts'].mean():.2f} attempts/game"
    )
    print(
        f"  ({(val['stat_attempts'].mean() / train['stat_attempts'].mean() - 1) * 100:.2f}% change)"
    )
    print()

    # Yards per attempt
    train["ypa"] = train["passing_yards"] / train["stat_attempts"]
    val["ypa"] = val["passing_yards"] / val["stat_attempts"]
    print("5. YARDS PER ATTEMPT ANALYSIS")
    print("-" * 80)
    print(f"Training (2018-2021): {train['ypa'].mean():.3f} yards/attempt")
    print(f"Validation (2022-2024): {val['ypa'].mean():.3f} yards/attempt")
    print(f"Difference: {val['ypa'].mean() - train['ypa'].mean():.3f} yards/attempt")
    print()

    cursor.close()
    conn.close()


def analyze_betting_lines():
    """Check if betting lines are systematically biased."""

    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    query = """
    SELECT
        season,
        home_score + away_score as actual_total,
        total_close as betting_total,
        home_score + away_score - total_close as total_deviation
    FROM games
    WHERE season BETWEEN 2018 AND 2024
        AND total_close IS NOT NULL
    ORDER BY season
    """

    cursor.execute(query)
    data = pd.DataFrame(cursor.fetchall())

    train = data[(data["season"] >= 2018) & (data["season"] <= 2021)]
    val = data[(data["season"] >= 2022) & (data["season"] <= 2024)]

    print("=" * 80)
    print("BETTING LINES ANALYSIS")
    print("=" * 80)
    print()

    print("1. TOTAL SCORE PREDICTIONS")
    print("-" * 80)
    print("Training (2018-2021):")
    print(f"  Avg actual total: {train['actual_total'].mean():.2f}")
    print(f"  Avg betting total: {train['betting_total'].mean():.2f}")
    print(f"  Avg deviation: {train['total_deviation'].mean():.2f}")
    print()
    print("Validation (2022-2024):")
    print(f"  Avg actual total: {val['actual_total'].mean():.2f}")
    print(f"  Avg betting total: {val['betting_total'].mean():.2f}")
    print(f"  Avg deviation: {val['total_deviation'].mean():.2f}")
    print()
    print(
        f"Change in deviation: {val['total_deviation'].mean() - train['total_deviation'].mean():.2f}"
    )
    print()

    # By season
    print("2. BY SEASON")
    print("-" * 80)
    for season in range(2018, 2025):
        season_data = data[data["season"] == season]
        print(
            f"Season {season}: actual={season_data['actual_total'].mean():.2f}, "
            f"betting={season_data['betting_total'].mean():.2f}, "
            f"deviation={season_data['total_deviation'].mean():.2f}"
        )
    print()

    cursor.close()
    conn.close()


def analyze_weather_data():
    """Check weather data completeness."""

    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    query = """
    SELECT
        season,
        COUNT(*) as total_games,
        SUM(CASE WHEN temp IS NULL THEN 1 ELSE 0 END) as missing_temp,
        SUM(CASE WHEN wind IS NULL THEN 1 ELSE 0 END) as missing_wind,
        AVG(CASE WHEN temp IS NOT NULL THEN CAST(temp AS NUMERIC) ELSE NULL END) as avg_temp
    FROM games
    WHERE season BETWEEN 2018 AND 2024
    GROUP BY season
    ORDER BY season
    """

    cursor.execute(query)
    data = pd.DataFrame(cursor.fetchall())

    print("=" * 80)
    print("WEATHER DATA COMPLETENESS")
    print("=" * 80)
    print()

    print(f"{'Season':<8} {'Total':>8} {'Missing Temp':>14} {'Missing Wind':>14} {'Avg Temp':>10}")
    print("-" * 80)
    for _, row in data.iterrows():
        print(
            f"{row['season']:<8} {row['total_games']:>8} "
            f"{row['missing_temp']:>14} ({row['missing_temp']/row['total_games']*100:>5.1f}%) "
            f"{row['missing_wind']:>14} ({row['missing_wind']/row['total_games']*100:>5.1f}%) "
            f"{row['avg_temp']:>10.1f}°F"
            if row["avg_temp"]
            else "       N/A"
        )
    print()

    cursor.close()
    conn.close()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("SYSTEMATIC BIAS INVESTIGATION")
    print("Analyzing potential causes of 12-17 yard under-prediction")
    print("=" * 80 + "\n")

    analyze_passing_yards_distribution()
    analyze_betting_lines()
    analyze_weather_data()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80 + "\n")
