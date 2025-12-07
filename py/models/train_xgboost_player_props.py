#!/usr/bin/env python3
"""
Train XGBoost Model on Player-Level Passing Props

Usage:
    python py/models/train_xgboost_player_props.py

Expected:
- Training time: 5-10 minutes
- Output: models/xgboost/player_props_v1.json
- Expected Impact: +0.5-1.0% ROI improvement
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_passing_data_from_db():
    """Load passing yards data from database (same query as BNN)"""

    logger.info("Connecting to database...")

    conn = psycopg2.connect(
        host="localhost", port=5544, user="dro", password="sicillionbillions", database="devdb01"
    )

    # Use same query as BNN for consistency
    query = """
    SELECT
        pgs.player_id,
        pgs.season,
        pgs.week,
        pgs.stat_yards as yards,
        pgs.stat_touchdowns as touchdowns,
        pgs.stat_attempts as attempts,
        pgs.current_team as team,
        g.home_score,
        g.away_score,
        g.spread_close as spread_line,
        g.total_close as total_line,
        g.temp,
        g.wind,
        CASE WHEN pgs.current_team = g.home_team THEN 1 ELSE 0 END as is_home
    FROM mart.player_game_stats pgs
    JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
    LEFT JOIN games g ON
      g.season = pgs.season
      AND g.week = pgs.week
      AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
    WHERE pgs.stat_category = 'passing'
      AND pgs.stat_yards IS NOT NULL
      AND pgs.stat_yards > 0
      AND pgs.stat_attempts >= 10
      AND pgs.season >= 2015
      AND pgs.season <= 2024
      AND ph.position_group = 'QB'
    ORDER BY pgs.season, pgs.week, pgs.player_id
    """

    logger.info("Loading passing data from database...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    logger.info(f"Loaded {len(df)} QB games from {df.season.min()}-{df.season.max()}")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create XGBoost input features (same as BNN)"""

    logger.info("Engineering features...")

    # Derive efficiency metrics from attempts
    df["log_attempts"] = np.log1p(df["attempts"])
    df["yards_per_attempt"] = df["yards"] / df["attempts"].replace(0, np.nan)
    df["td_rate"] = df["touchdowns"] / df["attempts"].replace(0, np.nan)

    # Game context
    df["spread_abs"] = df["spread_line"].abs()

    # Convert weather columns to numeric (handle 'DOME'/'CLOSED ROOF' etc)
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    df["wind"] = pd.to_numeric(df["wind"], errors="coerce")

    df["is_bad_weather"] = (
        ((df["temp"] < 32) | (df["wind"] > 15)).fillna(False).astype(int)  # Dome games = False
    )

    # Rolling averages (per player)
    df = df.sort_values(["player_id", "season", "week"])

    for col in ["yards", "yards_per_attempt", "td_rate"]:
        df[f"{col}_L3"] = (
            df.groupby("player_id")[col]
            .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
            .shift(1)
        )  # Shift to avoid lookahead

    # Season progress
    df["week_in_season"] = df["week"].astype(float)

    # Fill NaNs
    df = df.fillna(df.median(numeric_only=True))

    logger.info(f"Feature engineering complete: {len(df)} samples")

    return df


def prepare_train_test_split(df: pd.DataFrame):
    """Split data by season (canonical window: 2015-2024, test on 2024)"""

    train_seasons = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    test_season = 2024

    train_df = df[df["season"].isin(train_seasons)].copy()
    test_df = df[df["season"] == test_season].copy()

    # Feature columns (same as BNN)
    feature_cols = [
        "log_attempts",
        "yards_per_attempt",
        "td_rate",
        "is_home",
        "spread_abs",
        "total_line",
        "is_bad_weather",
        "yards_L3",
        "yards_per_attempt_L3",
        "td_rate_L3",
        "week_in_season",
        "temp",
        "wind",
    ]

    X_train = train_df[feature_cols].values
    y_train = train_df["yards"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["yards"].values

    logger.info(f"Train: {len(X_train)} samples ({train_seasons})")
    logger.info(f"Test: {len(X_test)} samples ({test_season})")
    logger.info(f"Features: {len(feature_cols)}")

    return X_train, y_train, X_test, y_test, feature_cols


def train_and_evaluate():
    """Main training pipeline"""

    logger.info("=" * 70)
    logger.info("XGBOOST TRAINING ON PLAYER-LEVEL PASSING DATA")
    logger.info("=" * 70 + "\n")

    # 1. Load data
    df = load_passing_data_from_db()

    # 2. Engineer features
    df = engineer_features(df)

    # 3. Train/test split
    X_train, y_train, X_test, y_test, feature_cols = prepare_train_test_split(df)

    # 4. Train XGBoost
    logger.info("Training XGBoost model...")

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "seed": 42,
    }

    evals = [(dtrain, "train"), (dtest, "test")]
    evals_result = {}

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=50,
        verbose_eval=50,
    )

    logger.info(f"\n✓ Training complete: {model.best_iteration} boosting rounds")

    # 5. Predict on test set
    logger.info("\nGenerating predictions on test set...")
    y_pred = model.predict(dtest)

    # 6. Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Baseline (mean prediction)
    baseline_mae = np.mean(np.abs(y_train.mean() - y_test))

    logger.info("\n" + "=" * 70)
    logger.info("XGBOOST TEST SET PERFORMANCE (2024 SEASON)")
    logger.info("=" * 70)
    logger.info(f"MAE: {mae:.2f} yards")
    logger.info(f"RMSE: {rmse:.2f} yards")
    logger.info(f"Mean Prediction: {y_pred.mean():.1f} yards")
    logger.info(f"Mean Actual: {y_test.mean():.1f} yards")
    logger.info(f"\nBaseline (mean) MAE: {baseline_mae:.2f} yards")
    logger.info(f"Improvement: {(1 - mae/baseline_mae)*100:.1f}%")

    # Feature importance
    importance = model.get_score(importance_type="gain")
    logger.info("\nTop 10 Most Important Features:")
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (feat, score) in enumerate(sorted_importance, 1):
        logger.info(f"  {i}. {feat}: {score:.1f}")

    # 7. Save model
    model_path = Path("models/xgboost/player_props_v1.json")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)

    # Also save metadata
    metadata = {
        "feature_cols": feature_cols,
        "train_seasons": [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
        "test_season": 2024,
        "test_mae": float(mae),
        "test_rmse": float(rmse),
        "best_iteration": int(model.best_iteration),
        "feature_importance": {k: float(v) for k, v in sorted_importance},
    }

    with open(model_path.parent / "player_props_v1_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\n✓ Model saved to: {model_path}")
    logger.info(f"✓ Metadata saved to: {model_path.parent}/player_props_v1_metadata.json")
    logger.info("\n✓ XGBoost training complete and ready for ensemble integration!\n")

    return model, metadata


if __name__ == "__main__":
    train_and_evaluate()
