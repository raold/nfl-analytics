#!/usr/bin/env python3
"""
Calculate Travel Features for NFL Games

Combines stadium distance matrix with game schedule to compute travel distance,
timezone shifts, and rest days for all away teams.

Author: Claude Code (Sonnet 4.5)
Date: November 16, 2025
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Database connection
import psycopg2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Database credentials
DB_CONFIG = {
    'host': 'localhost',
    'port': 5544,
    'database': 'devdb01',
    'user': 'dro',
    'password': os.getenv('DB_PASSWORD', 'sicillionbillions')  # pragma: allowlist secret
}


def load_distance_matrices() -> tuple:
    """
    Load stadium distance and timezone matrices.

    Returns:
        Tuple of (distance_matrix, timezone_matrix) DataFrames
    """
    data_dir = project_root / 'data'

    distance_matrix = pd.read_csv(
        data_dir / 'stadium_distance_matrix.csv',
        index_col=0
    )

    timezone_matrix = pd.read_csv(
        data_dir / 'stadium_timezone_matrix.csv',
        index_col=0
    )

    return distance_matrix, timezone_matrix


def extract_games_with_qb_stats() -> pd.DataFrame:
    """
    Extract games with QB passing stats from database.

    Returns:
        DataFrame with game and QB data
    """
    query = """
    WITH game_qb_stats AS (
        SELECT
            g.game_id,
            g.season,
            g.week,
            g.home_team,
            g.away_team,
            g.kickoff,
            g.stadium,
            g.away_rest,
            g.home_rest,
            g.spread_close,
            g.total_close,
            -- Away QB stats
            pgs_away.player_id AS away_qb_id,
            pgs_away.player_display_name AS away_qb_name,
            pgs_away.stat_yards AS away_passing_yards,
            pgs_away.stat_attempts AS away_passing_attempts,
            pgs_away.stat_completions AS away_passing_completions,
            pgs_away.stat_touchdowns AS away_passing_tds,
            pgs_away.stat_negative AS away_interceptions,
            -- Home QB stats
            pgs_home.player_id AS home_qb_id,
            pgs_home.player_display_name AS home_qb_name,
            pgs_home.stat_yards AS home_passing_yards,
            pgs_home.stat_attempts AS home_passing_attempts,
            pgs_home.stat_completions AS home_passing_completions,
            pgs_home.stat_touchdowns AS home_passing_tds,
            pgs_home.stat_negative AS home_interceptions
        FROM public.games g
        -- Join away QB stats (primary QB by most attempts)
        LEFT JOIN LATERAL (
            SELECT *
            FROM mart.player_game_stats
            WHERE season = g.season
              AND week = g.week
              AND current_team = g.away_team
              AND stat_category = 'passing'
            ORDER BY stat_attempts DESC NULLS LAST
            LIMIT 1
        ) pgs_away ON true
        -- Join home QB stats (primary QB by most attempts)
        LEFT JOIN LATERAL (
            SELECT *
            FROM mart.player_game_stats
            WHERE season = g.season
              AND week = g.week
              AND current_team = g.home_team
              AND stat_category = 'passing'
            ORDER BY stat_attempts DESC NULLS LAST
            LIMIT 1
        ) pgs_home ON true
        WHERE g.season >= 2018
          AND g.season <= 2024
          AND g.week >= 1  -- Regular season
          AND g.game_type = 'REG'
    )
    SELECT * FROM game_qb_stats
    ORDER BY season, week, kickoff
    """

    print(f"Connecting to database: {DB_CONFIG['database']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}")

    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"✓ Extracted {len(df)} games with QB stats")

    return df


def calculate_travel_features(
    df: pd.DataFrame,
    distance_matrix: pd.DataFrame,
    timezone_matrix: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate travel features for all games.

    Args:
        df: Games DataFrame
        distance_matrix: Stadium distance matrix
        timezone_matrix: Timezone shift matrix

    Returns:
        DataFrame with travel features added
    """
    # Calculate away team travel distance
    df['away_travel_distance'] = df.apply(
        lambda row: distance_matrix.loc[row['away_team'], row['home_team']]
        if row['away_team'] in distance_matrix.index and row['home_team'] in distance_matrix.columns
        else np.nan,
        axis=1
    )

    # Calculate away team timezone shift
    df['away_timezone_shift'] = df.apply(
        lambda row: timezone_matrix.loc[row['away_team'], row['home_team']]
        if row['away_team'] in timezone_matrix.index and row['home_team'] in timezone_matrix.columns
        else 0,
        axis=1
    )

    # Calculate days since last game (using away_rest from database)
    df['away_days_rest'] = df['away_rest']

    # Calculate travel load score
    # Formula: distance * timezone_shift / rest_days
    # Higher score = more fatigued team
    df['away_travel_load'] = np.where(
        df['away_days_rest'] > 0,
        (df['away_travel_distance'] * (1 + df['away_timezone_shift'])) / df['away_days_rest'],
        np.nan
    )

    # Create treatment indicators
    df['is_long_distance_travel'] = df['away_travel_distance'] >= 2000  # Coast-to-coast
    df['is_medium_distance_travel'] = (df['away_travel_distance'] >= 1000) & (df['away_travel_distance'] < 2000)
    df['is_short_week'] = df['away_days_rest'] < 7  # Thursday night games
    df['is_cross_country'] = (df['away_travel_distance'] >= 2000) & (df['away_timezone_shift'] >= 3)

    return df


def calculate_season_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate season-to-date averages for QBs (for PSM covariates).

    Args:
        df: Games DataFrame with QB stats

    Returns:
        DataFrame with season averages added
    """
    # Initialize columns
    df['away_qb_ytd_avg_yards'] = np.nan
    df['home_qb_ytd_avg_yards'] = np.nan
    df['away_qb_ytd_games'] = 0
    df['home_qb_ytd_games'] = 0

    # Sort by season, week for cumulative calculation
    df = df.sort_values(['season', 'week', 'kickoff'])

    # Calculate away QB season average (excluding current game)
    for (qb_id, season), group_idx in df.groupby(['away_qb_id', 'season']).groups.items():
        if pd.isna(qb_id):
            continue
        group = df.loc[group_idx].sort_values('week')
        expanding_mean = group['away_passing_yards'].expanding().mean().shift(1)
        expanding_count = group['away_passing_yards'].expanding().count().shift(1)
        df.loc[group_idx, 'away_qb_ytd_avg_yards'] = expanding_mean.values
        df.loc[group_idx, 'away_qb_ytd_games'] = expanding_count.values

    # Calculate home QB season average (excluding current game)
    for (qb_id, season), group_idx in df.groupby(['home_qb_id', 'season']).groups.items():
        if pd.isna(qb_id):
            continue
        group = df.loc[group_idx].sort_values('week')
        expanding_mean = group['home_passing_yards'].expanding().mean().shift(1)
        expanding_count = group['home_passing_yards'].expanding().count().shift(1)
        df.loc[group_idx, 'home_qb_ytd_avg_yards'] = expanding_mean.values
        df.loc[group_idx, 'home_qb_ytd_games'] = expanding_count.values

    # Fill missing season averages with overall average
    overall_avg = df['away_passing_yards'].mean()
    df['away_qb_ytd_avg_yards'] = df['away_qb_ytd_avg_yards'].fillna(overall_avg)
    df['home_qb_ytd_avg_yards'] = df['home_qb_ytd_avg_yards'].fillna(overall_avg)

    return df


def compute_travel_statistics(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics for travel features.

    Args:
        df: Travel features DataFrame

    Returns:
        Dictionary with statistics
    """
    # Filter to games with non-null QB stats
    df_qb = df[df['away_passing_yards'].notna() & df['home_passing_yards'].notna()]

    stats = {
        'total_games': len(df),
        'games_with_qb_stats': len(df_qb),
        'seasons': sorted(df['season'].unique().tolist()),
        'travel_distance_stats': {
            'mean': float(df['away_travel_distance'].mean()),
            'std': float(df['away_travel_distance'].std()),
            'min': float(df['away_travel_distance'].min()),
            'max': float(df['away_travel_distance'].max()),
            'median': float(df['away_travel_distance'].median())
        },
        'travel_treatment_counts': {
            'long_distance': int(df['is_long_distance_travel'].sum()),
            'medium_distance': int(df['is_medium_distance_travel'].sum()),
            'short_week': int(df['is_short_week'].sum()),
            'cross_country': int(df['is_cross_country'].sum())
        },
        'rest_days_distribution': df['away_days_rest'].value_counts().sort_index().to_dict(),
        'timezone_shift_distribution': df['away_timezone_shift'].value_counts().sort_index().to_dict(),
        'qb_stats': {
            'away_avg_yards': float(df_qb['away_passing_yards'].mean()),
            'home_avg_yards': float(df_qb['home_passing_yards'].mean()),
            'away_std_yards': float(df_qb['away_passing_yards'].std()),
            'home_std_yards': float(df_qb['home_passing_yards'].std())
        }
    }

    return stats


def identify_extreme_travel_games(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Identify games with most extreme travel conditions.

    Args:
        df: Travel DataFrame
        top_n: Number of games to return

    Returns:
        DataFrame with extreme travel games
    """
    # Get top travel load games
    extreme = df.nlargest(top_n, 'away_travel_load')[
        ['game_id', 'season', 'week', 'away_team', 'home_team',
         'away_travel_distance', 'away_timezone_shift', 'away_days_rest',
         'away_travel_load', 'away_passing_yards']
    ]

    return extreme


def main():
    """Main execution function."""
    print("=" * 80)
    print("NFL TRAVEL FEATURE CALCULATION")
    print("=" * 80)
    print()

    # Create data directory
    data_dir = project_root / 'data'
    data_dir.mkdir(exist_ok=True)

    # Load distance matrices
    print("Loading distance matrices...")
    distance_matrix, timezone_matrix = load_distance_matrices()
    print(f"✓ Loaded {distance_matrix.shape[0]} x {distance_matrix.shape[1]} distance matrix")
    print(f"✓ Loaded {timezone_matrix.shape[0]} x {timezone_matrix.shape[1]} timezone matrix")

    # Extract games with QB stats
    print("\nExtracting games with QB stats...")
    df = extract_games_with_qb_stats()

    # Calculate travel features
    print("\nCalculating travel features...")
    df = calculate_travel_features(df, distance_matrix, timezone_matrix)

    # Calculate season averages
    print("Calculating season-to-date QB averages...")
    df = calculate_season_averages(df)

    # Compute statistics
    print("\nComputing travel statistics...")
    stats = compute_travel_statistics(df)

    # Print statistics
    print("\n" + "=" * 80)
    print("TRAVEL DATA SUMMARY")
    print("=" * 80)
    print(f"Total games: {stats['total_games']}")
    print(f"Games with QB stats: {stats['games_with_qb_stats']}")
    print(f"Seasons: {stats['seasons'][0]}-{stats['seasons'][-1]}")
    print()

    print("Travel distance statistics:")
    dist_stats = stats['travel_distance_stats']
    print(f"  Mean:   {dist_stats['mean']:.1f} miles")
    print(f"  Median: {dist_stats['median']:.1f} miles")
    print(f"  Std:    {dist_stats['std']:.1f} miles")
    print(f"  Min:    {dist_stats['min']:.1f} miles")
    print(f"  Max:    {dist_stats['max']:.1f} miles")
    print()

    print("Travel treatment groups:")
    treatment = stats['travel_treatment_counts']
    print(f"  Long distance (≥2000 mi):  {treatment['long_distance']:4d} games "
          f"({100 * treatment['long_distance'] / stats['total_games']:.1f}%)")
    print(f"  Medium distance (1000-2000): {treatment['medium_distance']:4d} games "
          f"({100 * treatment['medium_distance'] / stats['total_games']:.1f}%)")
    print(f"  Short week (<7 days):      {treatment['short_week']:4d} games "
          f"({100 * treatment['short_week'] / stats['total_games']:.1f}%)")
    print(f"  Cross-country (≥2000 mi, ≥3 TZ): {treatment['cross_country']:4d} games "
          f"({100 * treatment['cross_country'] / stats['total_games']:.1f}%)")
    print()

    print("QB passing yards (games with stats):")
    qb_stats = stats['qb_stats']
    print(f"  Away QB avg: {qb_stats['away_avg_yards']:.1f} ± {qb_stats['away_std_yards']:.1f} yards")
    print(f"  Home QB avg: {qb_stats['home_avg_yards']:.1f} ± {qb_stats['home_std_yards']:.1f} yards")
    print(f"  Home advantage: +{qb_stats['home_avg_yards'] - qb_stats['away_avg_yards']:.1f} yards")
    print()

    # Identify extreme travel games
    print("\nTop 20 extreme travel load games:")
    extreme_games = identify_extreme_travel_games(df, top_n=20)
    for idx, row in extreme_games.iterrows():
        yards_str = f"{row['away_passing_yards']:.0f} yds" if pd.notna(row['away_passing_yards']) else "N/A"
        print(f"  {row['season']} Week {row['week']:2d}: "
              f"{row['away_team']:3s} @ {row['home_team']:3s} - "
              f"{row['away_travel_distance']:>6.0f} mi, {row['away_timezone_shift']:.0f} TZ, "
              f"{row['away_days_rest']:.0f} days rest (load: {row['away_travel_load']:.1f}, {yards_str})")
    print()

    # Save outputs
    output_path = data_dir / 'travel_features_2018_2024.csv'
    df.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path} ({len(df)} games)")

    # Save statistics
    import json

    def convert_numpy_types(obj):
        """Convert numpy types to native Python types."""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj

    stats_serializable = convert_numpy_types(stats)
    stats_path = data_dir / 'travel_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats_serializable, f, indent=2)
    print(f"✓ Saved: {stats_path}")

    # Save extreme games
    extreme_path = data_dir / 'extreme_travel_games.csv'
    extreme_games.to_csv(extreme_path, index=False)
    print(f"✓ Saved: {extreme_path}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total games with travel features: {stats['total_games']}")
    print(f"Long-distance travel games: {treatment['long_distance']} "
          f"({100 * treatment['long_distance'] / stats['total_games']:.1f}%)")
    print(f"Average travel distance: {dist_stats['mean']:.1f} miles")
    print(f"Max travel distance: {dist_stats['max']:.1f} miles")
    print()
    print("Treatment definition:")
    print("  - Long distance: ≥2000 miles (coast-to-coast)")
    print("  - Cross-country: ≥2000 miles AND ≥3 timezone shift")
    print()
    print("✓ Travel feature calculation complete!")


if __name__ == '__main__':
    main()
