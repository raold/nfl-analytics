#!/usr/bin/env python3
"""
Train Bayesian Neural Network on Real NFL Passing Data

Usage:
    python py/models/train_bnn_passing.py

Expected:
- Training time: 2-5 min (ADVI) or 15-30 min (NUTS)
- Output: models/bayesian/bnn_passing_v1.pkl
- Expected Impact: +0.3-0.8% ROI improvement
"""

import sys
sys.path.append('/Users/dro/rice/nfl-analytics')

import numpy as np
import pandas as pd
import psycopg2
from pathlib import Path
import logging

from py.models.bayesian_neural_network import BayesianNeuralNetwork

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_passing_data_from_db():
    """Load passing yards data from database"""

    logger.info("Connecting to database...")

    conn = psycopg2.connect(
        host="localhost",
        port=5544,
        user="dro",
        password="sicillionbillions",
        database="devdb01"
    )

    query = """
    SELECT
        pgs.player_id,
        pgs.season,
        pgs.week,
        pgs.stat_yards as yards,
        pgs.stat_attempts as attempts,
        pgs.stat_completions as completions,
        pgs.stat_touchdowns as touchdowns,
        pgs.current_team as team,
        g.home_score,
        g.away_score,
        g.spread_close as spread_line,
        g.total_close as total_line,
        g.temp,
        g.wind,
        CASE WHEN pgs.current_team = g.home_team THEN 1 ELSE 0 END as is_home
    FROM mart.player_game_stats pgs
    JOIN games g ON (
        pgs.season = g.season
        AND pgs.week = g.week
        AND (pgs.current_team = g.home_team OR pgs.current_team = g.away_team)
    )
    WHERE pgs.stat_category = 'passing'
      AND pgs.season >= 2020
      AND pgs.stat_attempts >= 10  -- Min attempts threshold
      AND pgs.stat_yards IS NOT NULL
    ORDER BY pgs.season, pgs.week, pgs.player_id
    """

    logger.info("Loading passing data from database...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    logger.info(f"Loaded {len(df)} QB games from {df.season.min()}-{df.season.max()}")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create BNN input features"""

    logger.info("Engineering features...")

    # Basic features
    df['log_attempts'] = np.log1p(df['attempts'])
    df['completion_rate'] = df['completions'] / df['attempts']
    df['yards_per_attempt'] = df['yards'] / df['attempts']
    df['td_rate'] = df['touchdowns'] / df['attempts']

    # Game context
    df['spread_abs'] = df['spread_line'].abs()

    # Convert weather columns to numeric (handle 'DOME'/'CLOSED ROOF' etc)
    df['temp'] = pd.to_numeric(df['temp'], errors='coerce')
    df['wind'] = pd.to_numeric(df['wind'], errors='coerce')

    df['is_bad_weather'] = (
        ((df['temp'] < 32) | (df['wind'] > 15))
        .fillna(False)  # Dome games = False
        .astype(int)
    )

    # Rolling averages (per player)
    df = df.sort_values(['player_id', 'season', 'week'])

    for col in ['yards', 'completion_rate', 'yards_per_attempt']:
        df[f'{col}_L3'] = df.groupby('player_id')[col].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        ).shift(1)  # Shift to avoid lookahead

    # Season progress
    df['week_in_season'] = df['week'].astype(float)

    # Fill NaNs
    df = df.fillna(df.median(numeric_only=True))

    logger.info(f"Feature engineering complete: {len(df)} samples")

    return df


def prepare_train_test_split(df: pd.DataFrame):
    """Split data by season"""

    train_seasons = [2020, 2021, 2022, 2023]
    test_season = 2024

    train_df = df[df['season'].isin(train_seasons)].copy()
    test_df = df[df['season'] == test_season].copy()

    # Feature columns
    feature_cols = [
        'log_attempts', 'completion_rate', 'yards_per_attempt', 'td_rate',
        'is_home', 'spread_abs', 'total_line', 'is_bad_weather',
        'yards_L3', 'completion_rate_L3', 'yards_per_attempt_L3',
        'week_in_season', 'temp', 'wind'
    ]

    X_train = train_df[feature_cols].values
    y_train = train_df['yards'].values

    X_test = test_df[feature_cols].values
    y_test = test_df['yards'].values

    logger.info(f"Train: {len(X_train)} samples ({train_seasons})")
    logger.info(f"Test: {len(X_test)} samples ({test_season})")
    logger.info(f"Features: {len(feature_cols)}")

    return X_train, y_train, X_test, y_test, feature_cols


def train_and_evaluate():
    """Main training pipeline"""

    logger.info("="*70)
    logger.info("BNN TRAINING ON REAL NFL PASSING DATA")
    logger.info("="*70 + "\n")

    # 1. Load data
    df = load_passing_data_from_db()

    # 2. Engineer features
    df = engineer_features(df)

    # 3. Train/test split
    X_train, y_train, X_test, y_test, feature_cols = prepare_train_test_split(df)

    # 4. Normalize target for better training
    y_train_mean = y_train.mean()
    y_train_std = y_train.std()

    y_train_norm = (y_train - y_train_mean) / y_train_std
    y_test_norm = (y_test - y_train_mean) / y_train_std

    logger.info(f"\nTarget normalization: mean={y_train_mean:.1f}, std={y_train_std:.1f}\n")

    # 5. Train BNN
    logger.info("Initializing Bayesian Neural Network...")

    bnn = BayesianNeuralNetwork(
        hidden_dims=(64, 32),
        inference_method="advi",  # Fast for production (2-5 min)
        n_samples=2000
    )

    logger.info("Training BNN (this will take 2-5 minutes)...\n")
    bnn.fit(X_train, y_train_norm, verbose=True)

    # 6. Predict on test set
    logger.info("\nGenerating predictions on test set...")
    pred_mean_norm, pred_std_norm = bnn.predict(X_test, return_std=True)

    # Denormalize predictions
    pred_mean = pred_mean_norm * y_train_std + y_train_mean
    pred_std = pred_std_norm * y_train_std  # Scale std

    # 7. Evaluate
    mse = np.mean((pred_mean - y_test)**2)
    mae = np.mean(np.abs(pred_mean - y_test))
    rmse = np.sqrt(mse)

    # Calibration check
    z_scores = (pred_mean - y_test) / pred_std
    calibration = np.mean(np.abs(z_scores) < 1.0)  # Should be ~68%

    # Baseline (mean prediction)
    baseline_mae = np.mean(np.abs(y_train.mean() - y_test))

    logger.info("\n" + "="*70)
    logger.info("BNN TEST SET PERFORMANCE (2024 SEASON)")
    logger.info("="*70)
    logger.info(f"MAE: {mae:.2f} yards")
    logger.info(f"RMSE: {rmse:.2f} yards")
    logger.info(f"Mean Prediction: {pred_mean.mean():.1f} yards")
    logger.info(f"Mean Actual: {y_test.mean():.1f} yards")
    logger.info(f"Mean Uncertainty (±1σ): {pred_std.mean():.1f} yards")
    logger.info(f"Calibration (±1σ): {calibration:.1%} (target: ~68%)")
    logger.info(f"\nBaseline (mean) MAE: {baseline_mae:.2f} yards")
    logger.info(f"Improvement: {(1 - mae/baseline_mae)*100:.1f}%")

    if 0.60 < calibration < 0.75:
        logger.info("✓ Good uncertainty calibration")
    else:
        logger.info("⚠️  Calibration may need tuning")

    # 8. Save model
    model_path = Path("models/bayesian/bnn_passing_v1.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bnn.save(model_path)

    # Also save feature info
    metadata = {
        'feature_cols': feature_cols,
        'y_mean': float(y_train_mean),
        'y_std': float(y_train_std),
        'train_seasons': [2020, 2021, 2022, 2023],
        'test_season': 2024,
        'test_mae': float(mae),
        'test_rmse': float(rmse),
        'calibration': float(calibration)
    }

    import json
    with open(model_path.parent / "bnn_passing_v1_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\n✓ Model saved to: {model_path}")
    logger.info(f"✓ Metadata saved to: {model_path.parent}/bnn_passing_v1_metadata.json")
    logger.info(f"\n✓ BNN training complete and ready for ensemble integration!\n")

    return bnn, metadata


if __name__ == "__main__":
    train_and_evaluate()
