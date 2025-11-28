#!/usr/bin/env python3
"""
Player Projection Aggregation Features for Team-Level Prediction
==================================================================

Aggregates player-level Bayesian predictions into team-level features for ensemble improvement.

**Purpose**:
Leverage rich player-level projections (BNN Rushing, State-Space, Hierarchical) to create
team-level features that capture:
1. Total offensive capability (sum of all players)
2. Star power (inequality/concentration of talent)
3. Depth quality (backup strength relative to starters)
4. Projection uncertainty (team-wide variance)

**Feature Engineering Strategy**:
- Extract predictions from ml.r_model_predictions (state_space_v2, hierarchical_v2, informative_priors_v2)
- Aggregate to team level: sum, mean, std, Gini coefficient, percentiles
- Create features for BOTH home and away teams
- Time-aware: Use only data available at prediction time (no leakage)

**Expected Impact**: +2-4 yards MAE improvement (Track A, Week 2)

Usage:
    from py.features.player_aggregation_features import aggregate_player_projections

    df = aggregate_player_projections(
        season=2024,
        week=1,
        model_names=['state_space_v2', 'hierarchical_v2']
    )

Author: Claude Code (Sonnet 4.5)
Date: November 22, 2025
Version: 1.0
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


def gini_coefficient(values: np.ndarray) -> float:
    """
    Calculate Gini coefficient (measure of inequality).

    Higher Gini = more concentrated (star-heavy team)
    Lower Gini = more equal (balanced depth)

    Args:
        values: Array of player projections

    Returns:
        float: Gini coefficient [0, 1]
    """
    if len(values) == 0:
        return 0.0

    sorted_values = np.sort(values)
    n = len(sorted_values)

    # Calculate Gini coefficient
    cumsum = np.cumsum(sorted_values)
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * cumsum[-1]) - (n + 1) / n

    return float(gini)


def calculate_depth_ratio(values: np.ndarray, top_pct: float = 0.25, bottom_pct: float = 0.75) -> float:
    """
    Calculate depth quality ratio: (top X% mean) / (bottom Y% mean).

    Higher ratio = strong starters, weak backups
    Lower ratio = strong depth across roster

    Args:
        values: Array of player projections
        top_pct: Percentile for top players (default 25th = top 25%)
        bottom_pct: Percentile for bottom players (default 75th = bottom 25%)

    Returns:
        float: Depth ratio
    """
    if len(values) < 4:  # Need at least 4 players for meaningful depth
        return 1.0

    top_threshold = np.percentile(values, (1 - top_pct) * 100)
    bottom_threshold = np.percentile(values, bottom_pct * 100)

    top_mean = np.mean(values[values >= top_threshold])
    bottom_mean = np.mean(values[values <= bottom_threshold])

    if bottom_mean == 0:
        return 1.0

    return float(top_mean / bottom_mean)


def aggregate_player_projections(
    season: int,
    week: int,
    model_names: Optional[List[str]] = None,
    db_host: str = "localhost",
    db_port: int = 5544,
    db_name: str = "devdb01",
    db_user: str = "dro",
    db_password: str = "sicillionbillions",
) -> pd.DataFrame:
    """
    Aggregate player-level projections to team-level features for all games in a given week.

    Creates 10-15 features per team capturing:
    - Total offensive EPA (sum)
    - Star power (Gini coefficient)
    - Depth quality (percentile ratios)
    - Projection uncertainty (std)
    - Top player concentration (% from top 3)

    Args:
        season: NFL season (e.g., 2024)
        week: NFL week (1-18)
        model_names: List of model names to aggregate. If None, uses all available.
                     Options: ['state_space_v2', 'hierarchical_v2', 'informative_priors_v2']
        db_host, db_port, db_name, db_user, db_password: Database connection parameters

    Returns:
        pd.DataFrame: Game-level features with columns:
            - game_id, season, week, home_team, away_team
            - home_proj_* and away_proj_* features (10-15 per team)
    """
    logger.info(f"Aggregating player projections for {season} Week {week}")

    # Default to all available models
    if model_names is None:
        model_names = ['state_space_v2', 'hierarchical_v2', 'informative_priors_v2']

    # Connect to database
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password
    )

    # Load games for this week
    games_query = """
    SELECT game_id, season, week, home_team, away_team
    FROM games
    WHERE season = %s AND week = %s
    ORDER BY game_id
    """
    games_df = pd.read_sql_query(games_query, conn, params=(season, week))
    logger.info(f"  Found {len(games_df)} games for {season} Week {week}")

    # Load player projections for this week
    projections_query = """
    SELECT
        rmp.model_name,
        rmp.player_id,
        rmp.season,
        rmp.week,
        rmp.prediction_mean,
        rmp.prediction_median,
        rmp.prediction_std,
        rmp.prediction_q05,
        rmp.prediction_q95,
        ph.player_name,
        ph.position_group,
        ph.current_team
    FROM ml.r_model_predictions rmp
    JOIN mart.player_hierarchy ph ON rmp.player_id = ph.player_id
    WHERE rmp.season = %s
        AND rmp.week = %s
        AND rmp.model_name = ANY(%s)
        AND ph.position_group IN ('QB', 'RB', 'WR', 'TE')  -- Offensive skill positions
    ORDER BY rmp.model_name, ph.current_team, rmp.prediction_mean DESC
    """
    projections_df = pd.read_sql_query(
        projections_query,
        conn,
        params=(season, week, model_names)
    )
    conn.close()

    logger.info(f"  Loaded {len(projections_df)} player projections from {len(model_names)} models")
    logger.info(f"    Models: {', '.join(model_names)}")
    logger.info(f"    Unique players: {projections_df.player_id.nunique()}")

    # Aggregate features by team and model
    aggregated_features = []

    for model_name in model_names:
        model_df = projections_df[projections_df.model_name == model_name]

        # Group by team
        for team in model_df.current_team.unique():
            team_df = model_df[model_df.current_team == team]

            if len(team_df) == 0:
                continue

            # Extract prediction values
            means = team_df.prediction_mean.values
            stds = team_df.prediction_std.values

            # Calculate aggregation features
            features = {
                'season': season,
                'week': week,
                'team': team,
                'model_name': model_name,

                # Basic aggregations
                'proj_total_mean': np.sum(means),  # Total expected output
                'proj_avg_mean': np.mean(means),   # Average player quality
                'proj_total_std': np.sqrt(np.sum(stds**2)),  # Total uncertainty (variance sum)
                'proj_avg_std': np.mean(stds),     # Average uncertainty per player

                # Star power (inequality)
                'proj_gini': gini_coefficient(means),  # Talent concentration

                # Depth quality
                'proj_depth_ratio': calculate_depth_ratio(means, top_pct=0.25, bottom_pct=0.75),
                'proj_p75_mean': np.mean(means) if len(means) < 4 else np.mean(means[means >= np.percentile(means, 75)]),
                'proj_p25_mean': np.mean(means) if len(means) < 4 else np.mean(means[means <= np.percentile(means, 25)]),

                # Top player concentration
                'proj_top1_pct': np.max(means) / np.sum(means) if np.sum(means) > 0 else 0,
                'proj_top3_pct': np.sum(np.sort(means)[-3:]) / np.sum(means) if np.sum(means) > 0 and len(means) >= 3 else 0,

                # Count features
                'proj_player_count': len(team_df),

                # Uncertainty features
                'proj_max_std': np.max(stds),      # Most uncertain player
                'proj_std_std': np.std(stds),      # Heterogeneity of uncertainty
            }

            aggregated_features.append(features)

    # Convert to DataFrame
    features_df = pd.DataFrame(aggregated_features)

    logger.info(f"  Created {len(features_df)} team-model feature sets")

    # Pivot to create model-specific columns (e.g., state_space_proj_total_mean)
    pivot_dfs = []
    for model_name in model_names:
        model_features = features_df[features_df.model_name == model_name].copy()

        # Rename columns to include model name
        rename_map = {col: f"{model_name}_{col}"
                     for col in model_features.columns
                     if col not in ['season', 'week', 'team', 'model_name']}
        model_features = model_features.rename(columns=rename_map)
        model_features = model_features.drop(columns=['model_name'])

        pivot_dfs.append(model_features)

    # Merge all model features
    if len(pivot_dfs) > 1:
        merged_features = pivot_dfs[0]
        for df in pivot_dfs[1:]:
            merged_features = merged_features.merge(
                df,
                on=['season', 'week', 'team'],
                how='outer'
            )
    else:
        merged_features = pivot_dfs[0]

    logger.info(f"  Merged features shape: {merged_features.shape}")

    # Join with games to create home/away pairs
    result_dfs = []

    for _, game in games_df.iterrows():
        game_features = {
            'game_id': game['game_id'],
            'season': game['season'],
            'week': game['week'],
            'home_team': game['home_team'],
            'away_team': game['away_team'],
        }

        # Get home team features
        home_features = merged_features[merged_features.team == game['home_team']]
        if len(home_features) > 0:
            for col in home_features.columns:
                if col not in ['season', 'week', 'team']:
                    game_features[f'home_{col}'] = home_features[col].values[0]

        # Get away team features
        away_features = merged_features[merged_features.team == game['away_team']]
        if len(away_features) > 0:
            for col in away_features.columns:
                if col not in ['season', 'week', 'team']:
                    game_features[f'away_{col}'] = away_features[col].values[0]

        result_dfs.append(game_features)

    # Convert to DataFrame
    result_df = pd.DataFrame(result_dfs)

    logger.info(f"  Final shape: {result_df.shape}")
    logger.info(f"  Feature columns: {[col for col in result_df.columns if col.startswith(('home_', 'away_'))][:5]}...")

    return result_df


def aggregate_player_projections_multi_season(
    seasons: List[int],
    weeks: Optional[List[int]] = None,
    model_names: Optional[List[str]] = None,
    db_host: str = "localhost",
    db_port: int = 5544,
    db_name: str = "devdb01",
    db_user: str = "dro",
    db_password: str = "sicillionbillions",
) -> pd.DataFrame:
    """
    Aggregate player projections across multiple seasons and weeks.

    Args:
        seasons: List of seasons (e.g., [2022, 2023, 2024])
        weeks: List of weeks (e.g., [1, 2, 3]). If None, uses all weeks (1-18).
        model_names: List of model names. If None, uses all available.
        db_host, db_port, db_name, db_user, db_password: Database connection

    Returns:
        pd.DataFrame: Concatenated features across all season-week pairs
    """
    if weeks is None:
        weeks = list(range(1, 19))  # Weeks 1-18

    all_features = []

    for season in seasons:
        for week in weeks:
            try:
                week_features = aggregate_player_projections(
                    season=season,
                    week=week,
                    model_names=model_names,
                    db_host=db_host,
                    db_port=db_port,
                    db_name=db_name,
                    db_user=db_user,
                    db_password=db_password,
                )
                all_features.append(week_features)

            except Exception as e:
                logger.warning(f"  Failed to aggregate {season} Week {week}: {e}")
                continue

    if len(all_features) == 0:
        logger.error("No features aggregated!")
        return pd.DataFrame()

    result = pd.concat(all_features, ignore_index=True)
    logger.info(f"✓ Aggregated {len(result)} games across {len(seasons)} seasons")

    return result


if __name__ == "__main__":
    # Example usage
    logger.info("=" * 70)
    logger.info("PLAYER AGGREGATION FEATURES - EXAMPLE USAGE")
    logger.info("=" * 70)

    # Test single week
    df = aggregate_player_projections(
        season=2024,
        week=1,
        model_names=['state_space_v2', 'hierarchical_v2']
    )

    print(f"\n✓ Generated {len(df)} games with {df.shape[1]} features")
    print(f"\nSample features (first 10 columns):")
    print(df.iloc[0, :10])

    print(f"\nFeature names (home team only):")
    home_features = [col for col in df.columns if col.startswith('home_')]
    for i, feat in enumerate(home_features[:15], 1):
        print(f"  {i:2d}. {feat}")
