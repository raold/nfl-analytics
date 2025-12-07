#!/usr/bin/env python3
"""
Train XGBoost v3 Model with Opponent Strength Features (16 Features)
Author: NFL Analytics Research Team
Date: November 14, 2025

Training Window: 2018-2024 (full weight) + 2015-2017 (exponential decay H=4)
Hold-out Test: 2025 season (148 games)
Features: 16 features (13 baseline + 3 opponent strength)

New Features (v3):
- opp_def_epa_exp: Exponentially weighted opponent defensive EPA
- opp_pass_def_rank: Opponent pass defense percentile rank (0-100)
- matchup_difficulty: Composite difficulty index (0-100)

Expected Performance: MAE < 53.5 yards (baseline v2: ~54 yards)
Ensemble Weight: 30% (4-way ensemble)

Usage:
    python py/models/train_xgboost_v3_with_opponent_strength.py

Output:
    - models/xgboost/passing_xgboost_v3_with_opponent_strength.pkl
    - models/xgboost/passing_xgboost_v3_with_opponent_strength_metadata.json
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# Canonical training window configuration
CANONICAL_WINDOW = {
    'full_weight_years': list(range(2018, 2025)),  # 2018-2024
    'decay_years': [2015, 2016, 2017],
    'decay_parameter': 4.0,  # H=4
}

# XGBoost hyperparameters (optimized for passing yards)
XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'max_depth': 7,  # Slightly deeper than baseline
    'learning_rate': 0.05,  # Lower for better generalization
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'min_child_weight': 3,
    'gamma': 0.05,
    'reg_alpha': 0.15,  # L1 regularization
    'reg_lambda': 1.5,  # L2 regularization
    'seed': 42,
    'n_jobs': -1,
}

NUM_BOOST_ROUNDS = 1000
EARLY_STOPPING_ROUNDS = 75


def load_data_from_db():
    """
    Load QB passing data with advanced features from database.

    Returns:
        pd.DataFrame: Complete dataset with all features
    """
    logger.info("=" * 70)
    logger.info("LOADING DATA FROM DATABASE")
    logger.info("=" * 70)

    conn = psycopg2.connect(**DB_CONFIG)

    # Comprehensive query with Phase 5 features
    query = """
    WITH qb_games AS (
        -- Base QB performance data
        SELECT
            pgs.player_id,
            pgs.season,
            pgs.week,
            pgs.stat_yards as yards,
            pgs.stat_touchdowns as touchdowns,
            pgs.stat_attempts as attempts,
            pgs.stat_completions as completions,
            pgs.avg_time_to_throw,
            pgs.avg_air_yards_differential,
            pgs.cpoe,
            pgs.current_team as team,
            pgs.player_display_name as qb_name,

            -- Game context
            g.game_id,
            g.home_team,
            g.away_team,
            g.home_score,
            g.away_score,
            g.spread_close,
            g.total_close,
            CASE WHEN pgs.current_team = g.home_team THEN 1 ELSE 0 END as is_home,

            -- Weather
            g.temp,
            g.wind,
            g.stadium,
            g.roof,
            g.surface,

            -- Game dynamics
            g.is_division_game,
            g.is_thursday_night,
            g.is_short_week,
            g.away_rest,
            g.home_rest,
            CASE WHEN pgs.current_team = g.home_team
                THEN g.home_rest
                ELSE g.away_rest
            END as rest_days,

            -- Opponent defense strength (last 4 games)
            CASE WHEN pgs.current_team = g.home_team
                THEN g.away_def_epa_last_4
                ELSE g.home_def_epa_last_4
            END as opp_def_epa,

            -- Turnovers & Penalties
            g.home_turnovers,
            g.away_turnovers,
            g.home_penalties,
            g.away_penalties

        FROM mart.player_game_stats pgs
        JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
        LEFT JOIN games g ON
            g.season = pgs.season
            AND g.week = pgs.week
            AND (g.home_team = pgs.current_team OR g.away_team = pgs.current_team)
        WHERE pgs.stat_category = 'passing'
            AND pgs.stat_yards IS NOT NULL
            AND pgs.stat_yards > 0
            AND pgs.stat_attempts >= 10  -- Minimum attempts filter
            AND pgs.season >= 2015
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
    logger.info(f"  Total games: {len(df):,}")

    return df


def load_opponent_strength_features():
    """
    Load opponent strength features from CSV file.

    Returns:
        pd.DataFrame: Opponent strength features
    """
    logger.info("\n" + "=" * 70)
    logger.info("LOADING OPPONENT STRENGTH FEATURES")
    logger.info("=" * 70)

    features_path = Path('data/processed/features/opponent_strength_features.csv')

    if not features_path.exists():
        raise FileNotFoundError(
            f"Opponent strength features not found at {features_path}. "
            f"Run py/features/opponent_strength.py first."
        )

    opp_df = pd.read_csv(features_path)

    logger.info(f"✓ Loaded {len(opp_df):,} opponent strength records")
    logger.info(f"  Columns: {list(opp_df.columns)}")
    logger.info(f"  Seasons: {opp_df.season.min()}-{opp_df.season.max()}")

    # Select only the features we need
    opp_features = opp_df[[
        'game_id', 'team',
        'opp_def_epa_exp',
        'opp_pass_def_rank',
        'matchup_difficulty'
    ]].copy()

    # Count non-null values
    non_null_counts = opp_features[['opp_def_epa_exp', 'opp_pass_def_rank', 'matchup_difficulty']].notna().sum()
    logger.info("\nNon-null feature counts:")
    for col, count in non_null_counts.items():
        logger.info(f"  {col}: {count:,} ({count/len(opp_features)*100:.1f}%)")

    return opp_features


def engineer_features(df: pd.DataFrame, opp_features: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer 16 features (13 baseline + 3 opponent strength).

    Baseline 13 features:
    1. log_attempts
    2. yards_per_attempt
    3. td_rate
    4. is_home (from query)
    5. spread_abs
    6. total_line (renamed from total_close)
    7. is_bad_weather
    8. yards_L3
    9. yards_per_attempt_L3
    10. td_rate_L3
    11. week_in_season
    12. temp (from query)
    13. wind (from query)

    New opponent strength features (v3):
    14. opp_def_epa_exp
    15. opp_pass_def_rank
    16. matchup_difficulty

    Args:
        df: Raw data from database
        opp_features: Opponent strength features

    Returns:
        pd.DataFrame: Engineered features (16 total)
    """
    logger.info("\n" + "=" * 70)
    logger.info("ENGINEERING 16 FEATURES (13 BASELINE + 3 OPPONENT STRENGTH)")
    logger.info("=" * 70)

    # Sort for rolling windows
    df = df.sort_values(['player_id', 'season', 'week']).copy()

    # ===== MERGE OPPONENT STRENGTH FEATURES =====
    logger.info("\nMerging opponent strength features...")
    before_merge = len(df)
    df = df.merge(
        opp_features,
        on=['game_id', 'team'],
        how='left'
    )
    after_merge = len(df)

    if after_merge != before_merge:
        logger.warning(f"Row count changed after merge: {before_merge} -> {after_merge}")
    else:
        logger.info(f"✓ Merge successful, {after_merge:,} rows preserved")

    # Check merge quality
    merge_stats = {
        'opp_def_epa_exp': df['opp_def_epa_exp'].notna().sum(),
        'opp_pass_def_rank': df['opp_pass_def_rank'].notna().sum(),
        'matchup_difficulty': df['matchup_difficulty'].notna().sum()
    }
    logger.info("\nOpponent strength feature coverage:")
    for feat, count in merge_stats.items():
        logger.info(f"  {feat}: {count:,} / {len(df):,} ({count/len(df)*100:.1f}%)")

    # ===== BASIC EFFICIENCY METRICS (3 features) =====
    logger.info("\nCreating basic efficiency metrics...")
    df['yards_per_attempt'] = df['yards'] / df['attempts'].replace(0, np.nan)
    df['td_rate'] = df['touchdowns'] / df['attempts'].replace(0, np.nan)
    df['log_attempts'] = np.log1p(df['attempts'])

    # ===== WEATHER FEATURES (3 features) =====
    logger.info("Creating weather features...")
    df['temp'] = pd.to_numeric(df['temp'], errors='coerce')
    df['wind'] = pd.to_numeric(df['wind'], errors='coerce')

    # is_bad_weather: cold (<32°F) OR windy (>15 mph), accounting for domes
    df['is_dome'] = (df['roof'].isin(['dome', 'closed'])).astype(int)
    df['is_cold'] = ((df['temp'] < 32) & (~df['is_dome'].astype(bool))).astype(int)
    df['is_windy'] = ((df['wind'] > 15) & (~df['is_dome'].astype(bool))).astype(int)
    df['is_bad_weather'] = (df['is_cold'] | df['is_windy']).astype(int)

    # ===== GAME SCRIPT FEATURES (2 features) =====
    logger.info("Creating game script features...")
    df['spread_abs'] = df['spread_close'].abs()
    df['total_line'] = df['total_close']  # Rename for ensemble compatibility

    # ===== SCHEDULE FACTORS (1 feature) =====
    logger.info("Creating schedule factors...")
    df['week_in_season'] = df['week'].astype(float)

    # ===== ROLLING AVERAGES (3 features - ANTI-LEAKAGE) =====
    logger.info("Creating rolling averages (L3 only)...")

    # Last 3 games (shifted to avoid lookahead bias)
    for col in ['yards', 'yards_per_attempt', 'td_rate']:
        df[f'{col}_L3'] = (
            df.groupby('player_id')[col]
            .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
            .shift(1)  # CRITICAL: Shift to avoid lookahead bias
        )

    # Fill NaNs with median (avoid lookahead)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            if col in ['opp_def_epa_exp', 'opp_pass_def_rank', 'matchup_difficulty']:
                logger.info(f"  Filled {col} NaNs with median: {median_val:.4f}")

    logger.info("\n✓ Feature engineering complete")
    logger.info("  Total features: 16 (13 baseline + 3 opponent strength)")
    logger.info(f"  Samples: {len(df):,}")

    return df


def apply_exponential_decay_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Apply exponential decay weights to training data.

    Canonical window:
    - 2018-2024: weight = 1.0
    - 2017: weight = exp(-1/4) = 0.779
    - 2016: weight = exp(-2/4) = 0.607
    - 2015: weight = exp(-3/4) = 0.472

    Args:
        df: Training dataframe with 'season' column

    Returns:
        np.ndarray: Sample weights
    """
    logger.info("\n" + "=" * 70)
    logger.info("APPLYING EXPONENTIAL DECAY WEIGHTS")
    logger.info("=" * 70)

    H = CANONICAL_WINDOW['decay_parameter']
    full_weight_years = CANONICAL_WINDOW['full_weight_years']

    weights = np.ones(len(df))

    for year in CANONICAL_WINDOW['decay_years']:
        mask = df['season'] == year
        decay_weight = np.exp(-(2018 - year) / H)
        weights[mask] = decay_weight

        logger.info(f"  {year}: {mask.sum():,} games, weight={decay_weight:.3f}")

    # Full weight years
    full_mask = df['season'].isin(full_weight_years)
    logger.info(f"  2018-2024: {full_mask.sum():,} games, weight=1.000")

    logger.info(f"\n✓ Weights applied to {len(weights):,} samples")
    logger.info(f"  Mean weight: {weights.mean():.3f}")
    logger.info(f"  Min weight: {weights.min():.3f}")

    return weights


def prepare_train_test_split(df: pd.DataFrame):
    """
    Split data into train/test sets with 16 features (13 baseline + 3 opponent strength).

    Train: 2015-2024 (with exponential decay weights)
    Test: 2025 season (hold-out set)

    Args:
        df: Engineered features dataframe

    Returns:
        tuple: (X_train, y_train, X_test, y_test, feature_cols, sample_weights)
    """
    logger.info("\n" + "=" * 70)
    logger.info("PREPARING TRAIN/TEST SPLIT (16 FEATURES)")
    logger.info("=" * 70)

    # 16 features: 13 baseline + 3 opponent strength
    feature_cols = [
        # Baseline 13 features
        'log_attempts',
        'yards_per_attempt',
        'td_rate',
        'is_home',
        'spread_abs',
        'total_line',
        'is_bad_weather',
        'yards_L3',
        'yards_per_attempt_L3',
        'td_rate_L3',
        'week_in_season',
        'temp',
        'wind',
        # New opponent strength features (v3)
        'opp_def_epa_exp',
        'opp_pass_def_rank',
        'matchup_difficulty'
    ]

    # Verify all features exist
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    # Train/test split
    train_df = df[df['season'] <= 2024].copy()
    test_df = df[df['season'] == 2025].copy()

    X_train = train_df[feature_cols].values
    y_train = train_df['yards'].values

    X_test = test_df[feature_cols].values
    y_test = test_df['yards'].values

    # Apply exponential decay weights to training set
    sample_weights = apply_exponential_decay_weights(train_df)

    logger.info("\n✓ Train/test split complete")
    logger.info(f"  Train: {len(X_train):,} samples (2015-2024)")
    logger.info(f"  Test: {len(X_test):,} samples (2025)")
    logger.info(f"  Features: {len(feature_cols)}")
    logger.info("\n16 Features (13 Baseline + 3 Opponent Strength):")
    for i, col in enumerate(feature_cols, 1):
        marker = " [NEW]" if col in ['opp_def_epa_exp', 'opp_pass_def_rank', 'matchup_difficulty'] else ""
        logger.info(f"  {i:2d}. {col}{marker}")

    return X_train, y_train, X_test, y_test, feature_cols, sample_weights


def train_xgboost_model(X_train, y_train, X_test, y_test, feature_cols, sample_weights):
    """
    Train XGBoost model with canonical window and evaluate.

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        feature_cols: Feature names
        sample_weights: Exponential decay weights

    Returns:
        tuple: (trained_model, test_predictions, metrics)
    """
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING XGBOOST MODEL")
    logger.info("=" * 70)

    start_time = time.time()

    # Create DMatrix with sample weights
    dtrain = xgb.DMatrix(
        X_train,
        label=y_train,
        feature_names=feature_cols,
        weight=sample_weights
    )
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

    logger.info(f"Training with {NUM_BOOST_ROUNDS} boost rounds...")
    logger.info(f"Early stopping: {EARLY_STOPPING_ROUNDS} rounds")
    logger.info("Hyperparameters:")
    for key, val in XGBOOST_PARAMS.items():
        logger.info(f"  {key}: {val}")

    # Training
    evals = [(dtrain, 'train'), (dtest, 'test')]
    evals_result = {}

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

    logger.info(f"\n✓ Training complete in {training_time/60:.2f} minutes")
    logger.info(f"  Best iteration: {model.best_iteration}")
    logger.info(f"  Best train MAE: {evals_result['train']['mae'][model.best_iteration]:.2f}")
    logger.info(f"  Best test MAE: {evals_result['test']['mae'][model.best_iteration]:.2f}")

    # Generate predictions
    y_pred = model.predict(dtest)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Baseline (mean predictor)
    baseline_mae = mean_absolute_error(y_test, np.full_like(y_test, y_train.mean()))

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'baseline_mae': baseline_mae,
        'improvement_pct': (1 - mae / baseline_mae) * 100,
        'training_time_minutes': training_time / 60,
        'best_iteration': model.best_iteration,
        'num_features': len(feature_cols),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
    }

    return model, y_pred, metrics


def evaluate_and_save(model, y_test, y_pred, metrics, feature_cols):
    """
    Evaluate model performance and save artifacts.

    Args:
        model: Trained XGBoost model
        y_test: True test values
        y_pred: Predicted test values
        metrics: Performance metrics dict
        feature_cols: Feature names
    """
    logger.info("\n" + "=" * 70)
    logger.info("MODEL EVALUATION - 2025 HOLDOUT SET")
    logger.info("=" * 70)

    logger.info("\nPerformance Metrics:")
    logger.info(f"  MAE: {metrics['mae']:.2f} yards")
    logger.info(f"  RMSE: {metrics['rmse']:.2f} yards")
    logger.info(f"  R²: {metrics['r2']:.4f}")
    logger.info(f"  Mean Prediction: {y_pred.mean():.1f} yards")
    logger.info(f"  Mean Actual: {y_test.mean():.1f} yards")
    logger.info("\nBaseline Comparison:")
    logger.info(f"  Baseline MAE: {metrics['baseline_mae']:.2f} yards")
    logger.info(f"  Improvement: {metrics['improvement_pct']:.1f}%")
    logger.info("\nTraining Info:")
    logger.info(f"  Training time: {metrics['training_time_minutes']:.2f} minutes")
    logger.info(f"  Best iteration: {metrics['best_iteration']}")
    logger.info(f"  Features: {metrics['num_features']}")
    logger.info(f"  Train samples: {metrics['train_samples']:,}")
    logger.info(f"  Test samples: {metrics['test_samples']:,}")

    # Feature importance
    importance = model.get_score(importance_type='gain')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    logger.info("\n" + "=" * 70)
    logger.info("TOP 16 FEATURE IMPORTANCE (by gain)")
    logger.info("=" * 70)
    for i, (feat, score) in enumerate(sorted_importance, 1):
        marker = " [NEW]" if feat in ['opp_def_epa_exp', 'opp_pass_def_rank', 'matchup_difficulty'] else ""
        logger.info(f"  {i:2d}. {feat:30s} {score:10.1f}{marker}")

    # Save model
    model_dir = Path('models/xgboost')
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / 'passing_xgboost_v3_with_opponent_strength.pkl'
    model.save_model(str(model_path))

    # Save metadata
    metadata = {
        'model_name': 'XGBoost Passing v3 (Canonical Window, 16 Features with Opponent Strength)',
        'training_date': pd.Timestamp.now().isoformat(),
        'canonical_window': CANONICAL_WINDOW,
        'hyperparameters': XGBOOST_PARAMS,
        'metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v
                   for k, v in metrics.items()},
        'feature_cols': feature_cols,
        'num_features': len(feature_cols),
        'ensemble_compatible': True,
        'feature_importance_all': {k: float(v) for k, v in sorted_importance},
        'ensemble_weight': 0.30,
        'test_season': 2025,
        'train_seasons': '2015-2024 (with H=4 decay on 2015-2017)',
        'opponent_strength_features': [
            'opp_def_epa_exp',
            'opp_pass_def_rank',
            'matchup_difficulty'
        ],
        'note': 'v3 adds 3 opponent strength features for improved matchup prediction',
    }

    metadata_path = model_dir / 'passing_xgboost_v3_with_opponent_strength_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("MODEL SAVED")
    logger.info("=" * 70)
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Metadata: {metadata_path}")
    logger.info(f"  Size: {model_path.stat().st_size / 1024:.1f} KB")

    return model_path, metadata_path


def main():
    """Main training pipeline"""

    logger.info("\n" + "=" * 70)
    logger.info("XGBOOST v3 TRAINING (WITH OPPONENT STRENGTH)")
    logger.info("=" * 70)
    logger.info("Training Window: 2018-2024 + H=4 decay (2015-2017)")
    logger.info("Test Set: 2025 season")
    logger.info("Features: 16 (13 baseline + 3 opponent strength)")
    logger.info("Target: MAE < 53.5 yards (v2 baseline: ~54 yards)")
    logger.info("=" * 70 + "\n")

    pipeline_start = time.time()

    # 1. Load data
    df = load_data_from_db()

    # 2. Load opponent strength features
    opp_features = load_opponent_strength_features()

    # 3. Engineer features (including opponent strength)
    df = engineer_features(df, opp_features)

    # 4. Train/test split
    X_train, y_train, X_test, y_test, feature_cols, sample_weights = prepare_train_test_split(df)

    # 5. Train model
    model, y_pred, metrics = train_xgboost_model(
        X_train, y_train, X_test, y_test, feature_cols, sample_weights
    )

    # 6. Evaluate and save
    model_path, metadata_path = evaluate_and_save(
        model, y_test, y_pred, metrics, feature_cols
    )

    pipeline_time = time.time() - pipeline_start

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {pipeline_time/60:.2f} minutes")
    logger.info(f"MAE: {metrics['mae']:.2f} yards (Target: <53.5, v2 baseline: ~54)")
    logger.info("Features: 16 (13 baseline + 3 opponent strength)")

    # Compare to v2 baseline
    v2_baseline = 54.0
    improvement_vs_v2 = ((v2_baseline - metrics['mae']) / v2_baseline) * 100
    logger.info("\nv2 → v3 Improvement:")
    logger.info(f"  v2 MAE: {v2_baseline:.2f} yards")
    logger.info(f"  v3 MAE: {metrics['mae']:.2f} yards")
    logger.info(f"  Improvement: {improvement_vs_v2:+.2f}%")

    status = "✓ PASS" if metrics['mae'] < 53.5 else "⚠ BORDERLINE" if metrics['mae'] < 54.0 else "✗ NEEDS IMPROVEMENT"
    logger.info(f"Status: {status}")

    logger.info("\nModel ready for 4-way ensemble integration (30% weight)")
    logger.info("New features: opp_def_epa_exp, opp_pass_def_rank, matchup_difficulty")
    logger.info("=" * 70 + "\n")


if __name__ == '__main__':
    main()
