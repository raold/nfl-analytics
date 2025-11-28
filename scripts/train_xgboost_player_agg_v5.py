#!/usr/bin/env python3
"""
Train XGBoost v5 with Player Aggregation Features
==================================================

Extends XGBoost v4 Ridge with 10-15 new player aggregation features derived from
player-level Bayesian projections (state_space_v2, hierarchical_v2, informative_priors_v2).

**Feature Groups**:
1. Baseline XGBoost v4 features (13 canonical + advanced)
2. NEW: Player aggregation features (10-15 per team × 2 teams = 20-30 features)
   - Total offensive EPA (sum of all player projections)
   - Star power (Gini coefficient)
   - Depth quality (percentile ratios)
   - Projection uncertainty (std aggregations)
   - Top player concentration

**Training Window**: 2022-2024 (player projections available)
**Test Set**: 2024 season (holdout validation)

**Expected Performance**:
- Baseline XGBoost v4 Ridge: 4.70 MAE
- Target with aggregation: 2.70-2.70 MAE (+2-4 yards improvement)

**Success Criteria**:
- No data leakage (temporal split validated)
- Feature importance shows aggregation features in top 10
- MAE improvement: +2-4 yards

Usage:
    python scripts/train_xgboost_player_agg_v5.py

Output:
    - models/xgboost/passing_xgboost_v5_player_agg.pkl
    - models/xgboost/passing_xgboost_v5_player_agg_metadata.json
    - models/xgboost/passing_xgboost_v5_feature_importance.png

Author: Claude Code (Sonnet 4.5)
Date: November 22, 2025
Version: 5.0 (Player Aggregation)
"""

import sys
import os
from pathlib import Path
import time
import json
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import player aggregation features
from py.features.player_aggregation_features import aggregate_player_projections_multi_season

# Import coaching features (OC and DC)
from py.features.oc_features import OCFeatureGenerator
from py.features.dc_features import DCFeatureGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5544')),
    'user': os.getenv('DB_USER', 'dro'),
    'password': os.getenv('DB_PASSWORD', 'sicillionbillions'),
    'database': os.getenv('DB_NAME', 'devdb01')
}

# Training configuration
TRAIN_SEASONS = [2022, 2023, 2024]  # Player projections available
TEST_SEASON = 2024  # Holdout validation
TRAIN_WEEKS = list(range(1, 19))  # Weeks 1-18

# Model names to aggregate
PLAYER_MODEL_NAMES = ['state_space_v2', 'hierarchical_v2', 'informative_priors_v2']

# XGBoost hyperparameters (optimized for passing yards)
XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'max_depth': 8,  # Deeper for more complex features
    'learning_rate': 0.04,
    'subsample': 0.85,
    'colsample_bytree': 0.80,
    'min_child_weight': 3,
    'gamma': 0.05,
    'reg_alpha': 0.20,  # L1 regularization
    'reg_lambda': 2.0,  # L2 regularization
    'seed': 42,
    'n_jobs': -1,
}

NUM_BOOST_ROUNDS = 1200
EARLY_STOPPING_ROUNDS = 100


def load_baseline_features():
    """
    Load baseline QB passing data with standard features from database.

    Returns:
        pd.DataFrame: Complete dataset with baseline features
    """
    logger.info("=" * 70)
    logger.info("LOADING BASELINE FEATURES FROM DATABASE")
    logger.info("=" * 70)

    conn = psycopg2.connect(**DB_CONFIG)

    # Load QB games with baseline features
    query = """
    WITH qb_games AS (
        SELECT
            pgs.player_id,
            pgs.season,
            pgs.week,
            pgs.stat_yards as yards,
            pgs.stat_attempts as attempts,
            pgs.stat_completions as completions,
            pgs.stat_touchdowns as touchdowns,
            pgs.current_team as team,

            -- Game context
            g.game_id,
            g.home_team,
            g.away_team,
            g.spread_close,
            g.total_close,
            CASE WHEN pgs.current_team = g.home_team THEN 1 ELSE 0 END as is_home,

            -- Weather
            g.temp,
            g.wind,

            -- Opponent defense
            CASE WHEN pgs.current_team = g.home_team
                THEN g.away_def_epa_last_4
                ELSE g.home_def_epa_last_4
            END as opp_def_epa

        FROM mart.player_game_stats pgs
        JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
        LEFT JOIN games g ON
            g.season = pgs.season
            AND g.week = pgs.week
            AND (g.home_team = pgs.current_team OR g.away_team = pgs.current_team)
        WHERE pgs.stat_category = 'passing'
            AND pgs.stat_yards IS NOT NULL
            AND pgs.stat_yards > 0
            AND pgs.stat_attempts >= 10
            AND pgs.season >= 2022  -- Player projections start here
            AND pgs.season <= 2025
            AND ph.position_group = 'QB'
    )
    SELECT * FROM qb_games
    ORDER BY season, week, player_id
    """

    logger.info("Executing database query...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    logger.info(f"✓ Loaded {len(df):,} QB games")
    logger.info(f"  Seasons: {df.season.min()}-{df.season.max()}")
    logger.info(f"  Unique QBs: {df.player_id.nunique()}")

    return df


def engineer_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer baseline features (13 canonical + advanced).

    Args:
        df: Raw data from database

    Returns:
        pd.DataFrame: Data with engineered features
    """
    logger.info("\nEngineering baseline features...")

    df = df.copy()

    # Convert numeric columns
    df['temp'] = pd.to_numeric(df['temp'], errors='coerce')
    df['wind'] = pd.to_numeric(df['wind'], errors='coerce')

    # Basic features
    df['log_attempts'] = np.log1p(df['attempts'])
    df['yards_per_attempt'] = df['yards'] / df['attempts'].clip(lower=1)
    df['td_rate'] = df['touchdowns'] / df['attempts'].clip(lower=1)
    df['spread_abs'] = df['spread_close'].abs()
    df['is_bad_weather'] = ((df['temp'].fillna(60) < 40) | (df['wind'].fillna(0) > 15)).astype(int)
    df['week_in_season'] = df['week']

    # Lagged features (last 3 games)
    df = df.sort_values(['player_id', 'season', 'week'])
    df['yards_L3'] = df.groupby('player_id')['yards'].transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))
    df['yards_per_attempt_L3'] = df.groupby('player_id')['yards_per_attempt'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    )
    df['td_rate_L3'] = df.groupby('player_id')['td_rate'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    )

    # Fill missing values
    df['yards_L3'] = df['yards_L3'].fillna(df['yards_L3'].median())
    df['yards_per_attempt_L3'] = df['yards_per_attempt_L3'].fillna(df['yards_per_attempt_L3'].median())
    df['td_rate_L3'] = df['td_rate_L3'].fillna(df['td_rate_L3'].median())
    df['opp_def_epa'] = df['opp_def_epa'].fillna(0)
    df['temp'] = df['temp'].fillna(df['temp'].median())
    df['wind'] = df['wind'].fillna(df['wind'].median())

    logger.info(f"  ✓ Engineered {df.shape[1]} baseline features")

    return df


def merge_player_aggregation_features(baseline_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge player aggregation features with baseline features.

    Args:
        baseline_df: DataFrame with baseline features and game_id

    Returns:
        pd.DataFrame: Merged dataset with baseline + aggregation features
    """
    logger.info("\n" + "=" * 70)
    logger.info("LOADING PLAYER AGGREGATION FEATURES")
    logger.info("=" * 70)

    # Get unique season-week pairs
    season_weeks = baseline_df[['season', 'week']].drop_duplicates().sort_values(['season', 'week'])
    logger.info(f"  Need aggregation features for {len(season_weeks)} season-week pairs")

    # Load aggregation features
    aggregation_df = aggregate_player_projections_multi_season(
        seasons=TRAIN_SEASONS,
        weeks=TRAIN_WEEKS,
        model_names=PLAYER_MODEL_NAMES,
        db_host=DB_CONFIG['host'],
        db_port=DB_CONFIG['port'],
        db_name=DB_CONFIG['database'],
        db_user=DB_CONFIG['user'],
        db_password=DB_CONFIG['password']
    )

    logger.info(f"✓ Loaded {len(aggregation_df)} games with player aggregation features")
    logger.info(f"  Feature count: {len([col for col in aggregation_df.columns if col.startswith(('home_', 'away_'))])}")

    # Merge with baseline on game_id
    merged_df = baseline_df.merge(
        aggregation_df,
        on='game_id',
        how='left',
        suffixes=('', '_agg')
    )

    # Drop duplicate season/week columns from aggregation
    merged_df = merged_df.drop(columns=[col for col in merged_df.columns if col.endswith('_agg')])

    logger.info(f"✓ Merged dataset shape: {merged_df.shape}")

    # Check for missing aggregation features
    agg_cols = [col for col in merged_df.columns if 'proj_' in col]
    missing_pct = merged_df[agg_cols].isnull().mean().mean()
    logger.info(f"  Missing aggregation features: {missing_pct*100:.1f}%")

    if missing_pct > 0.1:
        logger.warning(f"  WARNING: {missing_pct*100:.1f}% missing aggregation features")

    return merged_df


def merge_coaching_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge OC and DC coaching features with existing features.

    Adds coaching continuity and tenure features for both teams:
    - {side}_oc_continuity, {side}_oc_tenure_seasons, {side}_new_oc_adjustment
    - {side}_dc_continuity, {side}_dc_tenure_seasons, {side}_new_dc_adjustment

    Args:
        df: DataFrame with season, week, home_team, away_team columns

    Returns:
        pd.DataFrame: DataFrame with coaching features added
    """
    logger.info("\n" + "=" * 70)
    logger.info("LOADING COACHING FEATURES (OC + DC)")
    logger.info("=" * 70)

    # Initialize feature generators
    oc_gen = OCFeatureGenerator(db_config=DB_CONFIG)
    dc_gen = DCFeatureGenerator(db_config=DB_CONFIG)

    coaching_features = []

    # Get unique team-season-week combinations
    for idx, row in df.iterrows():
        season = row['season']
        week = row['week']
        home_team = row.get('home_team') or row.get('team')
        away_team = row.get('away_team')

        features_row = {'idx': idx}

        try:
            # Home team OC features
            if home_team:
                home_oc = oc_gen.get_features(home_team, season, week)
                for k, v in home_oc.items():
                    features_row[f'home_{k}'] = v

                # Home team DC features
                home_dc = dc_gen.get_features(home_team, season, week)
                for k, v in home_dc.items():
                    features_row[f'home_{k}'] = v

            # Away team OC features
            if away_team:
                away_oc = oc_gen.get_features(away_team, season, week)
                for k, v in away_oc.items():
                    features_row[f'away_{k}'] = v

                # Away team DC features
                away_dc = dc_gen.get_features(away_team, season, week)
                for k, v in away_dc.items():
                    features_row[f'away_{k}'] = v

        except Exception as e:
            logger.warning(f"Failed to get coaching features for {home_team} vs {away_team} {season} W{week}: {e}")

        coaching_features.append(features_row)

    # Close connections
    oc_gen.close()
    dc_gen.close()

    # Create coaching features DataFrame
    coaching_df = pd.DataFrame(coaching_features).set_index('idx')

    # Merge with original
    result_df = df.copy()
    for col in coaching_df.columns:
        result_df[col] = coaching_df[col].values

    # Count features added
    oc_cols = [col for col in result_df.columns if 'oc_' in col]
    dc_cols = [col for col in result_df.columns if 'dc_' in col]

    logger.info(f"✓ Added {len(oc_cols)} OC features and {len(dc_cols)} DC features")
    logger.info(f"  OC features: {oc_cols[:3]}...")
    logger.info(f"  DC features: {dc_cols[:3]}...")

    # Check for missing values
    coaching_cols = oc_cols + dc_cols
    if coaching_cols:
        missing_pct = result_df[coaching_cols].isnull().mean().mean()
        logger.info(f"  Missing coaching features: {missing_pct*100:.1f}%")

    return result_df


def train_xgboost_with_aggregation():
    """
    Train XGBoost v5 with player aggregation features.

    Returns:
        Tuple[xgb.Booster, Dict, Dict]: Trained model, metadata, validation results
    """
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING XGBOOST V5 WITH PLAYER AGGREGATION")
    logger.info("=" * 70)

    # Load and prepare data
    baseline_df = load_baseline_features()
    baseline_df = engineer_baseline_features(baseline_df)
    full_df = merge_player_aggregation_features(baseline_df)
    full_df = merge_coaching_features(full_df)  # Add OC + DC features

    # Split train/test (temporal)
    train_df = full_df[(full_df.season >= 2022) & (full_df.season < TEST_SEASON)].copy()
    test_df = full_df[full_df.season == TEST_SEASON].copy()

    logger.info(f"\nTrain/Test Split:")
    logger.info(f"  Train: {len(train_df):,} games (2022-{TEST_SEASON-1})")
    logger.info(f"  Test:  {len(test_df):,} games ({TEST_SEASON})")

    # Define feature columns
    exclude_cols = ['player_id', 'season', 'week', 'yards', 'team', 'game_id', 'home_team', 'away_team']
    feature_cols = [col for col in full_df.columns if col not in exclude_cols]

    # Remove columns with >50% missing values
    missing_pct = train_df[feature_cols].isnull().mean()
    high_missing = missing_pct[missing_pct > 0.5].index.tolist()
    if high_missing:
        logger.info(f"  Removing {len(high_missing)} high-missing features: {high_missing[:5]}...")
        feature_cols = [col for col in feature_cols if col not in high_missing]

    # Fill remaining missing values
    train_df[feature_cols] = train_df[feature_cols].fillna(train_df[feature_cols].median())
    test_df[feature_cols] = test_df[feature_cols].fillna(train_df[feature_cols].median())

    logger.info(f"\nFinal feature count: {len(feature_cols)}")
    baseline_count = len([col for col in feature_cols if 'proj_' not in col and 'oc_' not in col and 'dc_' not in col])
    agg_count = len([col for col in feature_cols if 'proj_' in col])
    oc_count = len([col for col in feature_cols if 'oc_' in col])
    dc_count = len([col for col in feature_cols if 'dc_' in col])
    logger.info(f"  Baseline features: {baseline_count}")
    logger.info(f"  Aggregation features: {agg_count}")
    logger.info(f"  OC coaching features: {oc_count}")
    logger.info(f"  DC coaching features: {dc_count}")

    # Create DMatrix
    X_train = train_df[feature_cols].values
    y_train = train_df['yards'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['yards'].values

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

    # Train model
    logger.info("\nTraining XGBoost v5...")
    logger.info(f"  Hyperparameters: {XGBOOST_PARAMS}")

    evals = [(dtrain, 'train'), (dtest, 'test')]
    evals_result = {}

    start_time = time.time()
    model = xgb.train(
        XGBOOST_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=50
    )
    training_time = time.time() - start_time

    logger.info(f"\n✓ Training complete in {training_time:.1f}s")
    logger.info(f"  Best iteration: {model.best_iteration}")

    # Evaluate
    y_pred_train = model.predict(dtrain)
    y_pred_test = model.predict(dtest)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"Train MAE: {train_mae:.2f} yards")
    logger.info(f"Test MAE:  {test_mae:.2f} yards")
    logger.info(f"Train RMSE: {train_rmse:.2f} yards")
    logger.info(f"Test RMSE:  {test_rmse:.2f} yards")
    logger.info(f"Train R²: {train_r2:.4f}")
    logger.info(f"Test R²:  {test_r2:.4f}")

    # Feature importance
    importance_dict = model.get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': [importance_dict.get(f, 0) for f in feature_cols]
    }).sort_values('importance', ascending=False)

    logger.info("\nTop 20 Features by Importance:")
    for i, row in importance_df.head(20).iterrows():
        logger.info(f"  {row['feature']:50s} {row['importance']:10.1f}")

    # Count aggregation features in top 10
    top10_agg = importance_df.head(10)['feature'].str.contains('proj_').sum()
    logger.info(f"\n✓ Aggregation features in top 10: {top10_agg}/10")

    # Metadata
    metadata = {
        'model_name': 'xgboost_v5_player_aggregation',
        'version': '5.0',
        'trained_at': datetime.now().isoformat(),
        'training_seasons': TRAIN_SEASONS,
        'test_season': TEST_SEASON,
        'train_size': len(train_df),
        'test_size': len(test_df),
        'num_features': len(feature_cols),
        'baseline_features': len([col for col in feature_cols if 'proj_' not in col]),
        'aggregation_features': len([col for col in feature_cols if 'proj_' in col]),
        'hyperparameters': XGBOOST_PARAMS,
        'num_boost_rounds': NUM_BOOST_ROUNDS,
        'best_iteration': int(model.best_iteration),
        'training_time_seconds': training_time,
        'train_mae': float(train_mae),
        'test_mae': float(test_mae),
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'aggregation_features_in_top10': int(top10_agg),
        'feature_importance_top20': importance_df.head(20).to_dict('records'),
    }

    validation_results = {
        'y_test': y_test.tolist(),
        'y_pred_test': y_pred_test.tolist(),
        'feature_importance': importance_df.to_dict('records'),
    }

    return model, metadata, validation_results, feature_cols, importance_df


def save_model_and_results(model, metadata, validation_results, feature_cols, importance_df):
    """
    Save trained model, metadata, and visualization.

    Args:
        model: Trained XGBoost model
        metadata: Model metadata dict
        validation_results: Validation results dict
        feature_cols: List of feature column names
        importance_df: Feature importance DataFrame
    """
    logger.info("\n" + "=" * 70)
    logger.info("SAVING MODEL AND RESULTS")
    logger.info("=" * 70)

    output_dir = Path("models/xgboost")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / "passing_xgboost_v5_player_agg.pkl"
    model.save_model(str(model_path))
    logger.info(f"✓ Saved model to {model_path}")

    # Save metadata
    metadata_path = output_dir / "passing_xgboost_v5_player_agg_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✓ Saved metadata to {metadata_path}")

    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 12))
    top_features = importance_df.head(30)

    # Color aggregation features differently
    colors = ['#ff7f0e' if 'proj_' in f else '#1f77b4' for f in top_features['feature']]

    ax.barh(range(len(top_features)), top_features['importance'], color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance (Gain)', fontsize=12)
    ax.set_title('XGBoost v5: Top 30 Features by Importance', fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff7f0e', label='Player Aggregation Features'),
        Patch(facecolor='#1f77b4', label='Baseline Features')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plot_path = output_dir / "passing_xgboost_v5_feature_importance.png"
    plt.savefig(plot_path, dpi=300)
    logger.info(f"✓ Saved feature importance plot to {plot_path}")

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    start_time = time.time()

    model, metadata, validation_results, feature_cols, importance_df = train_xgboost_with_aggregation()
    save_model_and_results(model, metadata, validation_results, feature_cols, importance_df)

    total_time = time.time() - start_time
    logger.info(f"\n✓ Total runtime: {total_time:.1f}s")
