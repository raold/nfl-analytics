#!/usr/bin/env python3
"""
BNN NUTS Comparison Study
Compares homoscedastic vs heteroscedastic with MCMC (NUTS) inference

Tests 4 configurations:
1. Homoscedastic + σ=0.3 + (24,12)
2. Heteroscedastic + σ=0.3 + (24,12)
3. Homoscedastic + σ=0.5 + (24,12)
4. Heteroscedastic + σ=0.5 + (24,12)

Expected runtime: 3-6 hours per config (12-24 hours total)
Can be parallelized on 4090 or run sequentially on M4.

Usage:
    python nfl_analytics/models/bnn_nuts_comparison.py
    # OR run single config:
    python nfl_analytics/models/bnn_nuts_comparison.py --config 1
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

from nfl_analytics.models.bayesian_neural_network import BayesianNeuralNetwork

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_passing_data_from_db():
    """Load passing yards data from database"""
    conn = psycopg2.connect(
        host="localhost", port=5544, user="dro", password="sicillionbillions", database="devdb01"
    )

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
    """Create BNN input features"""
    logger.info("Engineering features...")

    # Derive efficiency metrics from attempts
    df["log_attempts"] = np.log1p(df["attempts"])
    df["yards_per_attempt"] = df["yards"] / df["attempts"].replace(0, np.nan)
    df["td_rate"] = df["touchdowns"] / df["attempts"].replace(0, np.nan)

    # Game context
    df["spread_abs"] = df["spread_line"].abs()
    df["log_yards"] = np.log1p(df["yards"])

    # Convert weather columns to numeric
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    df["wind"] = pd.to_numeric(df["wind"], errors="coerce")

    df["is_bad_weather"] = ((df["temp"] < 32) | (df["wind"] > 15)).fillna(False).astype(int)

    # Rolling averages (per player)
    df = df.sort_values(["player_id", "season", "week"])

    for col in ["yards", "yards_per_attempt", "td_rate"]:
        df[f"{col}_L3"] = (
            df.groupby("player_id")[col]
            .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
            .shift(1)
        )

    # Season progress
    df["week_in_season"] = df["week"].astype(float)

    # Fill NaNs
    df = df.fillna(df.median(numeric_only=True))

    logger.info(f"Feature engineering complete: {len(df)} samples")
    return df


def prepare_train_test_split(df: pd.DataFrame):
    """Split data by season"""
    train_seasons = [2020, 2021, 2022, 2023]
    test_season = 2024

    train_df = df[df["season"].isin(train_seasons)].copy()
    test_df = df[df["season"] == test_season].copy()

    # Feature columns
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


def train_and_evaluate_config(
    config_num: int,
    prior_scale: float,
    heteroscedastic: bool,
    X_train: np.ndarray,
    y_train_norm: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_train_mean: float,
    y_train_std: float,
):
    """Train single NUTS configuration and evaluate"""

    arch_type = "heteroscedastic" if heteroscedastic else "homoscedastic"
    logger.info(f"\n{'='*80}")
    logger.info(f"CONFIG {config_num}: {arch_type.upper()} + σ={prior_scale} + NUTS")
    logger.info(f"{'='*80}\n")

    start_time = time.time()

    try:
        # Create model with proper heteroscedastic parameter
        bnn = BayesianNeuralNetwork(
            hidden_dims=(24, 12),  # Best from grid search
            inference_method="nuts",
            n_samples=2000,  # 4 chains × 500 draws
            prior_scale=prior_scale,
            heteroscedastic=heteroscedastic,  # CRITICAL: Pass architecture parameter
        )

        # Train with NUTS
        logger.info("Training with NUTS (MCMC sampling)...")
        logger.info("Expected runtime: 3-6 hours")

        bnn.fit(X_train, y_train_norm, verbose=True)

        # Predict
        pred_mean_norm, pred_std_norm = bnn.predict(X_test, return_std=True)

        # Denormalize
        pred_mean = pred_mean_norm * y_train_std + y_train_mean
        pred_std = pred_std_norm * y_train_std

        # Evaluate
        mse = np.mean((pred_mean - y_test) ** 2)
        mae = np.mean(np.abs(pred_mean - y_test))
        rmse = np.sqrt(mse)

        # Calibration
        z_scores = (pred_mean - y_test) / pred_std
        calibration = np.mean(np.abs(z_scores) < 1.0)

        # Sharpness (average prediction interval width)
        sharpness = np.mean(2 * 1.96 * pred_std)  # 95% PI width

        elapsed_time = time.time() - start_time

        # Store results
        result = {
            "config_num": config_num,
            "architecture": arch_type,
            "prior_scale": prior_scale,
            "inference": "nuts",
            "mae": float(mae),
            "rmse": float(rmse),
            "mse": float(mse),
            "mean_pred_std": float(pred_std.mean()),
            "calibration": float(calibration),
            "sharpness": float(sharpness),
            "training_time_hours": elapsed_time / 3600,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        logger.info(f"\n✓ CONFIG {config_num} RESULTS:")
        logger.info(f"  MAE: {mae:.2f} yards")
        logger.info(f"  RMSE: {rmse:.2f} yards")
        logger.info(f"  Mean Uncertainty: {pred_std.mean():.1f} yards")
        logger.info(f"  Calibration (±1σ): {calibration:.1%} (target: ~68%)")
        logger.info(f"  Sharpness (95% PI width): {sharpness:.1f} yards")
        logger.info(f"  Training time: {elapsed_time/3600:.2f} hours")

        # Save model
        model_path = Path(f"models/bayesian/bnn_nuts_{arch_type}_sigma{prior_scale}_v1.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        bnn.save(model_path)
        logger.info(f"✓ Model saved to {model_path}")

        return result, True

    except Exception as e:
        logger.error(f"✗ CONFIG {config_num} FAILED: {e}")

        elapsed_time = time.time() - start_time

        result = {
            "config_num": config_num,
            "architecture": arch_type,
            "prior_scale": prior_scale,
            "inference": "nuts",
            "error": str(e),
            "training_time_hours": elapsed_time / 3600,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        return result, False


def main(config_to_run: int | None = None):
    """Main NUTS comparison pipeline"""
    logger.info("=" * 80)
    logger.info("BNN NUTS COMPARISON - Homoscedastic vs Heteroscedastic")
    logger.info("=" * 80 + "\n")

    # Load and prepare data
    df = load_passing_data_from_db()
    df = engineer_features(df)
    X_train, y_train, X_test, y_test, feature_cols = prepare_train_test_split(df)

    # Normalize target
    y_train_mean = y_train.mean()
    y_train_std = y_train.std()
    y_train_norm = (y_train - y_train_mean) / y_train_std

    logger.info(f"\nTarget normalization: mean={y_train_mean:.1f}, std={y_train_std:.1f}\n")

    # Test configurations
    configs = [
        {"config_num": 1, "prior_scale": 0.3, "heteroscedastic": False},
        {"config_num": 2, "prior_scale": 0.3, "heteroscedastic": True},
        {"config_num": 3, "prior_scale": 0.5, "heteroscedastic": False},
        {"config_num": 4, "prior_scale": 0.5, "heteroscedastic": True},
    ]

    # Filter if specific config requested
    if config_to_run is not None:
        configs = [c for c in configs if c["config_num"] == config_to_run]
        if not configs:
            logger.error(f"Invalid config number: {config_to_run}")
            return

    logger.info("NUTS COMPARISON CONFIGURATIONS:")
    logger.info("1. Homoscedastic + σ=0.3 + (24,12)")
    logger.info("2. Heteroscedastic + σ=0.3 + (24,12)")
    logger.info("3. Homoscedastic + σ=0.5 + (24,12)")
    logger.info("4. Heteroscedastic + σ=0.5 + (24,12)")
    logger.info(f"Total: {len(configs)} configurations\n")
    logger.info("=" * 80 + "\n")

    results = []

    for config in configs:
        result, success = train_and_evaluate_config(
            config_num=config["config_num"],
            prior_scale=config["prior_scale"],
            heteroscedastic=config["heteroscedastic"],
            X_train=X_train,
            y_train_norm=y_train_norm,
            X_test=X_test,
            y_test=y_test,
            y_train_mean=y_train_mean,
            y_train_std=y_train_std,
        )

        results.append(result)

        # Save incremental results
        results_df = pd.DataFrame(results)
        results_path = Path("analysis/results/bnn_nuts_comparison_results.csv")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path, index=False)

        logger.info(f"\n✓ Incremental results saved to {results_path}\n")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("NUTS COMPARISON COMPLETE - SUMMARY")
    logger.info("=" * 80 + "\n")

    results_df = pd.DataFrame(results)
    valid_results = results_df[~results_df["mae"].isna()].copy()

    if len(valid_results) > 0:
        # Compare calibration (closer to 68% is better)
        valid_results["calib_error"] = np.abs(valid_results["calibration"] - 0.68)

        # Composite score: 60% MAE + 30% calibration + 10% sharpness
        valid_results["mae_score"] = valid_results["mae"] / valid_results["mae"].min()
        valid_results["calib_score"] = (
            valid_results["calib_error"] / valid_results["calib_error"].min()
        )
        valid_results["sharp_score"] = valid_results["sharpness"] / valid_results["sharpness"].min()

        valid_results["composite_score"] = (
            0.6 * valid_results["mae_score"]
            + 0.3 * valid_results["calib_score"]
            + 0.1 * valid_results["sharp_score"]
        )

        best_idx = valid_results["composite_score"].idxmin()
        best = valid_results.loc[best_idx]

        logger.info(f"BEST CONFIGURATION (Config #{int(best['config_num'])}):")
        logger.info(f"  Architecture: {best['architecture']}")
        logger.info(f"  Prior Scale (σ): {best['prior_scale']}")
        logger.info(f"  MAE: {best['mae']:.2f} yards")
        logger.info(f"  Mean Uncertainty: {best['mean_pred_std']:.1f} yards")
        logger.info(f"  Calibration: {best['calibration']:.1%} (target: 68%)")
        logger.info(f"  Sharpness: {best['sharpness']:.1f} yards")
        logger.info(f"  Training time: {best['training_time_hours']:.2f} hours")

        # Compare to ADVI baseline
        logger.info("\nCOMPARISON TO ADVI BASELINE:")
        logger.info("  ADVI Calibration: 93.8% (over-confident)")
        logger.info(f"  NUTS Calibration: {best['calibration']:.1%}")
        logger.info(f"  Improvement: {93.8 - best['calibration']*100:.1f} pp closer to 68% target")

        logger.info(f"\n✓ Full results saved to: {results_path}")

    else:
        logger.error("✗ No valid results from NUTS comparison!")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BNN NUTS Comparison Study")
    parser.add_argument(
        "--config", type=int, choices=[1, 2, 3, 4], help="Run specific config only (1-4)"
    )

    args = parser.parse_args()

    main(config_to_run=args.config)
