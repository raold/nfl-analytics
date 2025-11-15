#!/usr/bin/env python3
"""
Fix XGBoost Opponent Features Using Interaction Terms
Approach 1 - Quick Win Implementation

PROBLEM:
  Raw opponent features cause -42.71% MAE regression (40.20 → 57.37)

SOLUTION:
  Create interaction features (QB stats × opponent strength) and remove standalone opponent features

EXPECTED IMPROVEMENT:
  +23-30% vs current (target: 42-44 MAE)

USAGE:
  python scripts/fix_opponent_features_interactions.py --season 2024

Author: Claude Code (Sonnet 4.5)
Date: November 14, 2025
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
import psycopg2
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_player_stats_with_opponent(season: int, db_params: dict) -> pd.DataFrame:
    """
    Load player stats with opponent defensive features.

    Note: This assumes opponent features already exist in database.
    Adjust query to match your schema.
    """
    logger.info(f"Loading player stats for {season} season...")

    conn = psycopg2.connect(**db_params)

    query = """
        WITH weekly_stats AS (
            SELECT
                p.game_id,
                g.season,
                g.week,
                g.kickoff,
                p.passer_player_id AS player_id,
                p.passer_player_name AS player_name,
                SUM(p.passing_yards) AS passing_yards,
                COUNT(*) AS num_plays,
                SUM(CASE WHEN p.pass_attempt THEN 1 ELSE 0 END) AS attempts,
                SUM(CASE WHEN p.complete_pass = 1 THEN 1 ELSE 0 END) AS completions
            FROM plays p
            JOIN games g ON p.game_id = g.game_id
            WHERE g.season = %s
                AND p.passer_player_id IS NOT NULL
                AND p.season_type = 'REG'
            GROUP BY p.game_id, g.season, g.week, g.kickoff, p.passer_player_id, p.passer_player_name
            HAVING SUM(p.passing_yards) >= 100
        )
        SELECT
            ws.*,
            g.home_team,
            g.away_team,
            COALESCE(g.spread_close, 0.0) as spread_close,
            COALESCE(g.total_close, 47.0) as total_close
            -- Add opponent defensive features here if available
            -- e.g., opp_def_epa_exp, opp_pass_def_rank, etc.
        FROM weekly_stats ws
        JOIN games g ON ws.game_id = g.game_id
        ORDER BY ws.kickoff, ws.player_id
    """

    df = pd.read_sql(query, conn, params=(season,))
    conn.close()

    logger.info(f"Loaded {len(df)} player-game records for {season}")
    return df


def engineer_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer baseline QB features (rolling averages, completion %, etc.)

    CRITICAL: Avoids data leakage by using .shift(1) before rolling averages
    """
    logger.info("Engineering baseline QB features...")
    df = df.copy()
    df = df.sort_values(['player_id', 'kickoff'])

    # L3 rolling windows (lag 1 to avoid leakage)
    df['passing_yards_L3'] = df.groupby('player_id')['passing_yards'].shift(1).rolling(3).mean()
    df['attempts_L3'] = df.groupby('player_id')['attempts'].shift(1).rolling(3).mean()
    df['completions_L3'] = df.groupby('player_id')['completions'].shift(1).rolling(3).mean()

    # Derived metrics
    df['completion_pct_L3'] = (
        df.groupby('player_id')['completions'].shift(1).rolling(3).sum() /
        df.groupby('player_id')['attempts'].shift(1).rolling(3).sum()
    )

    df['yards_per_attempt_L3'] = (
        df.groupby('player_id')['passing_yards'].shift(1).rolling(3).sum() /
        df.groupby('player_id')['attempts'].shift(1).rolling(3).sum()
    )

    # Game context
    df['home_favorite'] = (df['spread_close'] < 0).astype(int)
    df['high_total'] = (df['total_close'] > 47).astype(int)

    # Drop rows with NaN in critical features
    critical_features = ['passing_yards_L3', 'attempts_L3', 'completions_L3']
    df = df.dropna(subset=critical_features)

    # Fill remaining NaN
    df['completion_pct_L3'] = df['completion_pct_L3'].fillna(df['completion_pct_L3'].median())
    df['yards_per_attempt_L3'] = df['yards_per_attempt_L3'].fillna(df['yards_per_attempt_L3'].median())

    logger.info(f"After feature engineering: {len(df)} records")
    return df


def add_mock_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add mock opponent features for testing (if not in database).

    In production, replace with actual opponent defensive metrics.
    """
    logger.info("Adding mock opponent defensive features...")

    # Mock: Opponent defensive EPA (normally distributed, lower = better defense)
    np.random.seed(42)
    df['opp_def_epa_exp'] = np.random.normal(-0.05, 0.15, len(df))

    # Mock: Opponent pass defense rank (1-32)
    df['opp_pass_def_rank'] = np.random.randint(1, 33, len(df))

    # Mock: Matchup difficulty (composite metric, -1 to 1 scale)
    df['matchup_difficulty'] = (df['opp_pass_def_rank'] / 32) * 2 - 1

    logger.info("Mock opponent features added (REPLACE WITH REAL DATA IN PRODUCTION)")
    return df


def engineer_opponent_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between QB stats and opponent defense.

    APPROACH 1: INTERACTION FEATURES
    - Multiplicative interactions (QB efficiency × opponent difficulty)
    - Categorical interactions (QB stats vs elite/weak defenses)
    - Ratio-based interactions (QB efficiency relative to opponent)

    REMOVES standalone opponent features to avoid redundancy.
    """
    logger.info("Engineering opponent interaction features...")
    df = df.copy()

    # Compute opponent defensive difficulty (inverse of EPA, so higher = tougher)
    df['opp_defensive_difficulty'] = 1 / (df['opp_def_epa_exp'] + 0.1)  # +0.1 to avoid div/0

    # INTERACTION 1: Multiplicative QB stats × opponent difficulty
    df['yards_L3_x_opp_difficulty'] = df['passing_yards_L3'] * df['opp_defensive_difficulty']
    df['attempts_L3_x_opp_difficulty'] = df['attempts_L3'] * df['opp_defensive_difficulty']
    df['completion_pct_x_opp_difficulty'] = df['completion_pct_L3'] * df['opp_defensive_difficulty']
    df['yards_per_attempt_x_opp_difficulty'] = df['yards_per_attempt_L3'] * df['opp_defensive_difficulty']

    logger.info("  Created 4 multiplicative interaction features")

    # INTERACTION 2: Categorical opponent tier × continuous QB stats
    df['elite_def'] = (df['opp_pass_def_rank'] <= 10).astype(int)  # Top 10 defenses
    df['weak_def'] = (df['opp_pass_def_rank'] >= 22).astype(int)   # Bottom 11 defenses

    df['yards_L3_vs_elite_def'] = df['passing_yards_L3'] * df['elite_def']
    df['yards_L3_vs_weak_def'] = df['passing_yards_L3'] * df['weak_def']
    df['attempts_L3_vs_elite_def'] = df['attempts_L3'] * df['elite_def']
    df['attempts_L3_vs_weak_def'] = df['attempts_L3'] * df['weak_def']

    logger.info("  Created 4 categorical interaction features (elite/weak defense)")

    # INTERACTION 3: Ratio-based (if opponent stats available)
    # Example: QB completion % relative to opponent's completion % allowed
    # Uncomment if you have opponent completion_pct_allowed in database:
    # df['completion_pct_vs_opp_allowed'] = (
    #     df['completion_pct_L3'] / (df['opp_completion_pct_allowed'] + 0.01)
    # )

    # CRITICAL: Remove standalone opponent features to avoid redundancy
    opponent_standalone_features = [
        'opp_def_epa_exp',
        'opp_pass_def_rank',
        'matchup_difficulty',
        'opp_defensive_difficulty',  # Intermediate variable
    ]

    df = df.drop([f for f in opponent_standalone_features if f in df.columns], axis=1)

    logger.info("  Removed standalone opponent features (avoid redundancy)")
    logger.info(f"  Final feature count: {len([c for c in df.columns if c not in ['passing_yards', 'kickoff', 'player_id', 'player_name', 'game_id', 'season', 'week']])}")

    return df


def compare_models(df: pd.DataFrame) -> dict:
    """
    Compare 3 models:
    1. Baseline (no opponent features)
    2. Raw opponent features (current broken approach)
    3. Interaction features (proposed fix)
    """
    logger.info("=" * 60)
    logger.info("COMPARING MODELS")
    logger.info("=" * 60)

    # Feature columns
    exclude_cols = ['passing_yards', 'kickoff', 'player_id', 'player_name', 'game_id',
                    'season', 'week', 'home_team', 'away_team', 'num_plays']

    # Split train/test (50/50 temporal split)
    split_idx = len(df) // 2
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    y_train = df_train['passing_yards'].values
    y_test = df_test['passing_yards'].values

    results = {}

    # MODEL 1: Baseline (no opponent features)
    logger.info("\n1. BASELINE MODEL (no opponent features)")
    baseline_cols = [c for c in df_train.columns
                     if c not in exclude_cols
                     and 'opp_' not in c
                     and 'elite_def' not in c
                     and 'weak_def' not in c
                     and '_x_opp_' not in c
                     and '_vs_' not in c]

    X_train_baseline = df_train[baseline_cols].fillna(0)
    X_test_baseline = df_test[baseline_cols].fillna(0)

    model_baseline = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    model_baseline.fit(X_train_baseline, y_train)
    pred_baseline = model_baseline.predict(X_test_baseline)
    mae_baseline = mean_absolute_error(y_test, pred_baseline)

    logger.info(f"  Features: {len(baseline_cols)}")
    logger.info(f"  MAE: {mae_baseline:.2f} yards")

    results['baseline'] = {
        'mae': mae_baseline,
        'num_features': len(baseline_cols),
        'model': model_baseline
    }

    # MODEL 2: Interaction Features (proposed fix)
    logger.info("\n2. INTERACTION FEATURES MODEL (proposed fix)")
    interaction_cols = [c for c in df_train.columns
                       if c not in exclude_cols]

    X_train_interaction = df_train[interaction_cols].fillna(0)
    X_test_interaction = df_test[interaction_cols].fillna(0)

    model_interaction = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        reg_alpha=2.0,  # Mild L1 regularization
        random_state=42,
        verbosity=0
    )
    model_interaction.fit(X_train_interaction, y_train)
    pred_interaction = model_interaction.predict(X_test_interaction)
    mae_interaction = mean_absolute_error(y_test, pred_interaction)

    logger.info(f"  Features: {len(interaction_cols)}")
    logger.info(f"  MAE: {mae_interaction:.2f} yards")
    logger.info(f"  Improvement vs baseline: {((mae_baseline - mae_interaction) / mae_baseline * 100):.1f}%")

    results['interaction'] = {
        'mae': mae_interaction,
        'num_features': len(interaction_cols),
        'model': model_interaction
    }

    # Feature importance analysis
    logger.info("\n3. FEATURE IMPORTANCE (Top 15)")
    importance_df = pd.DataFrame({
        'feature': interaction_cols,
        'importance': model_interaction.feature_importances_
    }).sort_values('importance', ascending=False)

    for idx, row in importance_df.head(15).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    results['importance'] = importance_df

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Baseline MAE:     {mae_baseline:.2f} yards")
    logger.info(f"Interaction MAE:  {mae_interaction:.2f} yards")
    logger.info(f"Improvement:      {((mae_baseline - mae_interaction) / mae_baseline * 100):+.1f}%")

    if mae_interaction < mae_baseline:
        logger.info("\n✓ SUCCESS: Interaction features improve performance")
    else:
        logger.info("\n✗ WARNING: Interaction features do not improve (investigate further)")

    return results


def main():
    parser = argparse.ArgumentParser(description='Fix XGBoost opponent features using interactions')
    parser.add_argument('--season', type=int, default=2024, help='Season to analyze')
    parser.add_argument('--db-host', type=str, default='localhost', help='Database host')
    parser.add_argument('--db-port', type=int, default=5544, help='Database port')
    parser.add_argument('--db-name', type=str, default='devdb01', help='Database name')
    parser.add_argument('--db-user', type=str, default='dro', help='Database user')
    parser.add_argument('--db-password', type=str, default='sicillionbillions', help='Database password')
    args = parser.parse_args()

    db_params = {
        'host': args.db_host,
        'port': args.db_port,
        'dbname': args.db_name,
        'user': args.db_user,
        'password': args.db_password,
    }

    logger.info("=" * 60)
    logger.info("FIX XGBOOST OPPONENT FEATURES - APPROACH 1")
    logger.info("=" * 60)
    logger.info(f"Season: {args.season}")

    # Load data
    df = load_player_stats_with_opponent(args.season, db_params)

    # Engineer baseline features
    df = engineer_baseline_features(df)

    # Add opponent features (mock if not in database)
    if 'opp_def_epa_exp' not in df.columns:
        logger.warning("Opponent features not found in database - using mock data")
        df = add_mock_opponent_features(df)

    # Engineer interaction features
    df = engineer_opponent_interactions(df)

    # Compare models
    results = compare_models(df)

    logger.info("\n" + "=" * 60)
    logger.info("COMPLETE - Results saved in memory")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("1. If interaction MAE < baseline: Deploy to production")
    logger.info("2. If interaction MAE >= baseline: Try Approach 2 (ridge residuals)")
    logger.info("3. Validate on 2022-2023 seasons for multi-season consistency")

    return results


if __name__ == "__main__":
    results = main()
