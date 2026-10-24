#!/usr/bin/env python3
"""
BNN Temperature Scaling - Post-hoc Calibration Fix

Temperature scaling is a simple yet effective post-hoc calibration method
that rescales prediction uncertainties by a learned temperature parameter T.

Key Insight:
- ADVI BNN already works (MAE 335, calibration 93.8%)
- Problem: Over-confident (93.8% coverage vs 68% target)
- Solution: Scale uncertainties by T > 1 to widen prediction intervals

Method (Guo et al. 2017, "On Calibration of Modern Neural Networks"):
1. Load pre-trained ADVI BNN model
2. Tune temperature T on validation set to achieve 68% calibration
3. Apply temperature scaling: σ_calibrated = σ_raw * T

Expected Result:
- MAE: 335 yards (unchanged - only uncertainty rescaled)
- Calibration: 68% (fixed from 93.8%)
- Training time: 1-5 minutes (no model retraining required)

Usage:
    python nfl_analytics/models/bnn_temperature_scaling.py
"""

#!/usr/bin/env python3
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2
from scipy.optimize import minimize_scalar

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_analytics.models.bayesian_neural_network import BayesianNeuralNetwork

logging.basicConfig(level=logging.INFO)
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


def prepare_train_val_test_split(df: pd.DataFrame):
    """Split data by season for temperature tuning"""
    train_seasons = [2020, 2021, 2022]  # Training (for ADVI model)
    val_season = 2023  # Validation (for temperature tuning)
    test_season = 2024  # Test (for final evaluation)

    train_df = df[df["season"].isin(train_seasons)].copy()
    val_df = df[df["season"] == val_season].copy()
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

    X_val = val_df[feature_cols].values
    y_val = val_df["yards"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["yards"].values

    logger.info(f"Train: {len(X_train)} samples ({train_seasons})")
    logger.info(f"Val: {len(X_val)} samples ({val_season})")
    logger.info(f"Test: {len(X_test)} samples ({test_season})")
    logger.info(f"Features: {len(feature_cols)}")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def compute_calibration(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray) -> float:
    """
    Compute calibration metric: fraction of samples within ±1σ

    Target: ~68% (1-sigma coverage for normal distribution)
    """
    z_scores = np.abs((y_true - y_pred) / y_std)
    calibration = np.mean(z_scores < 1.0)
    return calibration


def tune_temperature(
    bnn: BayesianNeuralNetwork,
    X_val: np.ndarray,
    y_val: np.ndarray,
    target_calibration: float = 0.68,
) -> float:
    """
    Tune temperature parameter T to achieve target calibration

    Method:
    - Grid search over T ∈ [0.1, 3.0]
    - Find T that minimizes |calibration - 0.68|

    Returns:
        Optimal temperature T
    """
    logger.info("\nTuning temperature parameter...")
    logger.info(f"Target calibration: {target_calibration:.1%}")

    # Get predictions from pre-trained model
    pred_mean, pred_std = bnn.predict(X_val, return_std=True)

    def calibration_loss(T):
        """Loss function: squared error from target calibration"""
        scaled_std = pred_std * T
        calib = compute_calibration(y_val, pred_mean, scaled_std)
        loss = (calib - target_calibration) ** 2
        return loss

    # Optimize temperature using golden section search
    result = minimize_scalar(calibration_loss, bounds=(0.1, 3.0), method="bounded")

    T_opt = result.x

    # Compute final calibration with optimal T
    scaled_std = pred_std * T_opt
    final_calib = compute_calibration(y_val, pred_mean, scaled_std)

    logger.info(f"\n✓ Optimal Temperature: T = {T_opt:.3f}")
    logger.info(f"  Validation Calibration: {final_calib:.1%} (target: {target_calibration:.1%})")
    logger.info(f"  Calibration Error: {abs(final_calib - target_calibration):.1%}")

    return T_opt


def main():
    """Main temperature scaling pipeline"""
    logger.info("=" * 80)
    logger.info("BNN TEMPERATURE SCALING - Post-hoc Calibration Fix")
    logger.info("=" * 80 + "\n")

    # Load and prepare data
    df = load_passing_data_from_db()
    df = engineer_features(df)
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = prepare_train_val_test_split(df)

    # Load pre-trained ADVI BNN or train new one
    model_path = Path("models/bayesian/bnn_advi_sigma0.3_arch32x16_v1.pkl")

    if not model_path.exists():
        logger.warning(f"Pre-trained model not found: {model_path}")
        logger.info("Training new ADVI model (Config 2: σ=0.3, 32×16)...")
        logger.info("Expected training time: 3-5 minutes\n")

        # Train ADVI model (best from grid search)
        bnn = BayesianNeuralNetwork(
            hidden_dims=(32, 16),
            inference_method="advi",
            n_samples=2000,
            prior_scale=0.3,
            heteroscedastic=True,
        )

        # Normalize target for training
        y_train_mean = y_train.mean()
        y_train_std = y_train.std()
        y_train_norm = (y_train - y_train_mean) / y_train_std

        logger.info(f"Training on {len(X_train)} samples...")
        bnn.fit(X_train, y_train_norm, verbose=True)

        # Save for future use
        model_path.parent.mkdir(parents=True, exist_ok=True)
        bnn.save(model_path)
        logger.info(f"✓ Model trained and saved to {model_path}\n")
    else:
        logger.info(f"Loading pre-trained ADVI model: {model_path}")
        bnn = BayesianNeuralNetwork.load(model_path)
        logger.info("✓ Model loaded successfully\n")

    # Evaluate baseline (before temperature scaling)
    logger.info("=" * 80)
    logger.info("BASELINE EVALUATION (Before Temperature Scaling)")
    logger.info("=" * 80)

    pred_mean_val, pred_std_val = bnn.predict(X_val, return_std=True)
    pred_mean_test, pred_std_test = bnn.predict(X_test, return_std=True)

    mae_val = np.mean(np.abs(pred_mean_val - y_val))
    mae_test = np.mean(np.abs(pred_mean_test - y_test))

    calib_val = compute_calibration(y_val, pred_mean_val, pred_std_val)
    calib_test = compute_calibration(y_test, pred_mean_test, pred_std_test)

    logger.info("\nValidation Set (2023):")
    logger.info(f"  MAE: {mae_val:.2f} yards")
    logger.info(f"  Calibration: {calib_val:.1%} (target: 68%)")
    logger.info(f"  Mean Uncertainty: {pred_std_val.mean():.1f} yards")

    logger.info("\nTest Set (2024):")
    logger.info(f"  MAE: {mae_test:.2f} yards")
    logger.info(f"  Calibration: {calib_test:.1%} (target: 68%)")
    logger.info(f"  Mean Uncertainty: {pred_std_test.mean():.1f} yards")

    # Tune temperature on validation set
    logger.info("\n" + "=" * 80)
    logger.info("TEMPERATURE TUNING")
    logger.info("=" * 80)

    T_opt = tune_temperature(bnn, X_val, y_val, target_calibration=0.68)

    # Evaluate with temperature scaling
    logger.info("\n" + "=" * 80)
    logger.info("CALIBRATED EVALUATION (After Temperature Scaling)")
    logger.info("=" * 80)

    # Apply temperature scaling
    pred_std_val_scaled = pred_std_val * T_opt
    pred_std_test_scaled = pred_std_test * T_opt

    calib_val_scaled = compute_calibration(y_val, pred_mean_val, pred_std_val_scaled)
    calib_test_scaled = compute_calibration(y_test, pred_mean_test, pred_std_test_scaled)

    logger.info("\nValidation Set (2023) - CALIBRATED:")
    logger.info(f"  MAE: {mae_val:.2f} yards (unchanged)")
    logger.info(f"  Calibration: {calib_val_scaled:.1%} (target: 68%)")
    logger.info(f"  Mean Uncertainty: {pred_std_val_scaled.mean():.1f} yards")
    logger.info(f"  Temperature: T = {T_opt:.3f}")

    logger.info("\nTest Set (2024) - CALIBRATED:")
    logger.info(f"  MAE: {mae_test:.2f} yards (unchanged)")
    logger.info(f"  Calibration: {calib_test_scaled:.1%} (target: 68%)")
    logger.info(f"  Mean Uncertainty: {pred_std_test_scaled.mean():.1f} yards")
    logger.info(f"  Temperature: T = {T_opt:.3f}")

    # Save results
    results = {
        "method": "temperature_scaling",
        "base_model": "bnn_advi_sigma0.3_arch32x16_v1",
        "temperature": float(T_opt),
        "mae_val": float(mae_val),
        "mae_test": float(mae_test),
        "calibration_val_before": float(calib_val),
        "calibration_val_after": float(calib_val_scaled),
        "calibration_test_before": float(calib_test),
        "calibration_test_after": float(calib_test_scaled),
        "mean_uncertainty_val_before": float(pred_std_val.mean()),
        "mean_uncertainty_val_after": float(pred_std_val_scaled.mean()),
        "mean_uncertainty_test_before": float(pred_std_test.mean()),
        "mean_uncertainty_test_after": float(pred_std_test_scaled.mean()),
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    results_path = Path("analysis/results/bnn_temperature_scaling_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    import json

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Results saved to {results_path}")

    # Save calibrated model wrapper
    calibrated_model = {
        "base_model_path": str(model_path),
        "temperature": T_opt,
        "method": "temperature_scaling",
        "calibration_target": 0.68,
        "achieved_calibration_test": float(calib_test_scaled),
    }

    calibrated_path = Path("models/bayesian/bnn_temperature_scaled_v1.pkl")
    joblib.dump(calibrated_model, calibrated_path)
    logger.info(f"✓ Calibrated model saved to {calibrated_path}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEMPERATURE SCALING SUMMARY")
    logger.info("=" * 80)
    logger.info("\n✓ Temperature Scaling Complete")
    logger.info(f"  Optimal Temperature: T = {T_opt:.3f}")
    logger.info(f"  Test MAE: {mae_test:.2f} yards (unchanged)")
    logger.info(f"  Test Calibration: {calib_test:.1%} → {calib_test_scaled:.1%}")
    logger.info(
        f"  Calibration Improvement: {abs(calib_test_scaled - 0.68):.1%} from target (vs {abs(calib_test - 0.68):.1%} before)"
    )

    if abs(calib_test_scaled - 0.68) < 0.05:
        logger.info("\n✓✓ EXCELLENT CALIBRATION - Ready for production")
    elif abs(calib_test_scaled - 0.68) < 0.10:
        logger.info("\n✓ GOOD CALIBRATION - Acceptable for production")
    else:
        logger.info("\n⚠️  POOR CALIBRATION - Consider MC Dropout or Laplace instead")

    logger.info("\n" + "=" * 80)
    logger.info("NEXT STEPS")
    logger.info("=" * 80)
    logger.info("\n1. Compare with MC Dropout and Laplace implementations")
    logger.info("2. Select best approach based on calibration + MAE")
    logger.info("3. Integrate into 4-way ensemble")
    logger.info("4. Run full backtest (2022-2024)")

    return results


if __name__ == "__main__":
    main()
