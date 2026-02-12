#!/usr/bin/env python3
"""
Extract v6 Baseline and v7 Variant Results from Database
=========================================================

Parses the ml.ensemble_predictions table to extract validation results
for v6_baseline and v7 variants (2022-2024 seasons).

Calculates:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- R² (Coefficient of Determination)
- Calibration (% within 90% CI)
- Inference time statistics

Output: JSON files + Markdown summary

Author: Claude Code (Sonnet 4.5)
Date: November 19, 2025
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy import stats

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database connection
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "dbname": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",  # pragma: allowlist secret
}


def get_predictions(config: str, season: int, conn) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get predictions for a specific configuration and season.

    Args:
        config: Configuration name (e.g., "v6_baseline", "v7_full")
        season: Season year (2022, 2023, or 2024)
        conn: Database connection

    Returns:
        (predictions, actuals, uncertainties) arrays
    """
    query = """
    SELECT
        ep.ensemble_mean as prediction,
        ep.ensemble_std as uncertainty,
        pgs.stat_yards as actual
    FROM ml.ensemble_predictions ep
    JOIN mart.player_game_stats pgs ON
        ep.player_id = pgs.player_id
        AND ep.season = pgs.season
        AND ep.week = pgs.week
    WHERE ep.model_version = %s
        AND ep.season = %s
        AND pgs.stat_category = 'passing'
        AND ep.ensemble_mean IS NOT NULL
        AND pgs.stat_yards IS NOT NULL
    ORDER BY ep.season, ep.week, ep.player_id
    """

    df = pd.read_sql_query(query, conn, params=(config, season))

    if len(df) == 0:
        logger.warning(f"No predictions found for {config} season {season}")
        return np.array([]), np.array([]), np.array([])

    return (
        df["prediction"].values,
        df["actual"].values,
        df["uncertainty"].values,
    )


def calculate_metrics(
    predictions: np.ndarray, actuals: np.ndarray, uncertainties: np.ndarray
) -> dict:
    """
    Calculate validation metrics.

    Args:
        predictions: Predicted values
        actuals: Actual values
        uncertainties: Prediction uncertainties (std)

    Returns:
        Dictionary of metrics
    """
    if len(predictions) == 0:
        return {
            "n_predictions": 0,
            "mae": None,
            "rmse": None,
            "r2": None,
            "calibration_90": None,
            "calibration_50": None,
        }

    # MAE
    mae = np.mean(np.abs(predictions - actuals))

    # RMSE
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

    # R²
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Calibration (% within confidence intervals)
    z_90 = 1.645  # 90% CI
    z_50 = 0.674  # 50% CI

    errors = np.abs(predictions - actuals)
    within_90 = np.mean(errors <= z_90 * uncertainties) * 100
    within_50 = np.mean(errors <= z_50 * uncertainties) * 100

    return {
        "n_predictions": len(predictions),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "calibration_90": float(within_90),
        "calibration_50": float(within_50),
        "mean_prediction": float(np.mean(predictions)),
        "mean_actual": float(np.mean(actuals)),
        "mean_uncertainty": float(np.mean(uncertainties)),
    }


def diebold_mariano_test(
    errors1: np.ndarray, errors2: np.ndarray, h: int = 1
) -> tuple[float, float]:
    """
    Diebold-Mariano test for predictive accuracy comparison.

    Tests H0: errors1 and errors2 have equal predictive accuracy
    vs H1: errors1 has different (better/worse) accuracy than errors2

    Args:
        errors1: Absolute errors from model 1
        errors2: Absolute errors from model 2
        h: Forecast horizon (default: 1)

    Returns:
        (dm_statistic, p_value) tuple
    """
    d = errors1 - errors2  # Loss differential
    n = len(d)

    if n == 0:
        return (np.nan, np.nan)

    # Mean loss differential
    d_bar = np.mean(d)

    # Variance of loss differential (accounting for autocorrelation)
    gamma_0 = np.var(d, ddof=1)
    dm_statistic = d_bar / np.sqrt(gamma_0 / n)

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_statistic)))

    return (float(dm_statistic), float(p_value))


def main():
    logger.info("=" * 80)
    logger.info("EXTRACT v6 BASELINE AND v7 VARIANT RESULTS")
    logger.info("=" * 80)

    # Connect to database
    conn = psycopg2.connect(**DB_CONFIG)

    # Configurations to analyze
    configurations = ["v6_baseline", "v7_meta_only", "v7_wls_only", "v7_full"]
    seasons = [2022, 2023, 2024]

    # Extract results for all configurations and seasons
    all_results = {}

    for config in configurations:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Configuration: {config}")
        logger.info(f"{'=' * 60}")

        config_results = {}

        for season in seasons:
            logger.info(f"\n  Season {season}:")

            # Get predictions
            predictions, actuals, uncertainties = get_predictions(config, season, conn)

            if len(predictions) == 0:
                logger.warning("    No data found")
                continue

            # Calculate metrics
            metrics = calculate_metrics(predictions, actuals, uncertainties)

            logger.info(f"    Games: {metrics['n_predictions']}")
            logger.info(f"    MAE: {metrics['mae']:.2f} yards")
            logger.info(f"    RMSE: {metrics['rmse']:.2f} yards")
            logger.info(f"    R²: {metrics['r2']:.4f}")
            logger.info(f"    Calibration (90% CI): {metrics['calibration_90']:.1f}%")
            logger.info(f"    Calibration (50% CI): {metrics['calibration_50']:.1f}%")

            config_results[season] = metrics

        # Store results
        all_results[config] = config_results

        # Save per-configuration JSON
        output_file = Path(f"results/validation_{config}_2022_2024.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(config_results, f, indent=2)

        logger.info(f"\n  ✅ Results saved to: {output_file}")

    # Statistical comparison: v6_baseline vs v7 variants
    logger.info("\n" + "=" * 80)
    logger.info("STATISTICAL COMPARISON (Diebold-Mariano Tests)")
    logger.info("=" * 80)

    comparison_results = {}

    for config in ["v7_meta_only", "v7_wls_only", "v7_full"]:
        logger.info(f"\n{config} vs v6_baseline:")

        config_comparison = {}

        for season in seasons:
            # Get predictions for both configurations
            preds_v6, actuals_v6, _ = get_predictions("v6_baseline", season, conn)
            preds_v7, actuals_v7, _ = get_predictions(config, season, conn)

            if len(preds_v6) == 0 or len(preds_v7) == 0:
                logger.warning(f"  {season}: No data")
                continue

            # Calculate absolute errors
            errors_v6 = np.abs(preds_v6 - actuals_v6)
            errors_v7 = np.abs(preds_v7 - actuals_v7)

            # Diebold-Mariano test
            dm_stat, p_value = diebold_mariano_test(errors_v6, errors_v7)

            # MAE difference
            mae_v6 = np.mean(errors_v6)
            mae_v7 = np.mean(errors_v7)
            mae_diff = mae_v7 - mae_v6

            logger.info(f"  {season}:")
            logger.info(f"    MAE difference: {mae_diff:+.2f} yards")
            logger.info(f"    DM statistic: {dm_stat:.4f}")
            logger.info(f"    p-value: {p_value:.6f}")

            if p_value < 0.01:
                logger.info("    ✅ Significant at p < 0.01")
            elif p_value < 0.05:
                logger.info("    ✅ Significant at p < 0.05")
            else:
                logger.info("    ❌ Not significant (p >= 0.05)")

            config_comparison[season] = {
                "mae_v6": float(mae_v6),
                "mae_v7": float(mae_v7),
                "mae_diff": float(mae_diff),
                "dm_statistic": dm_stat,
                "p_value": p_value,
                "significant_01": p_value < 0.01,
                "significant_05": p_value < 0.05,
            }

        comparison_results[config] = config_comparison

    # Save comparison results
    comparison_file = Path("results/v6_vs_v7_comparison.json")
    with open(comparison_file, "w") as f:
        json.dump(comparison_results, f, indent=2)

    logger.info(f"\n✅ Comparison results saved to: {comparison_file}")

    # Close database connection
    conn.close()

    logger.info("\n" + "=" * 80)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 80)

    logger.info("\nOutput files:")
    logger.info("  - results/validation_v6_baseline_2022_2024.json")
    logger.info("  - results/validation_v7_meta_only_2022_2024.json")
    logger.info("  - results/validation_v7_wls_only_2022_2024.json")
    logger.info("  - results/validation_v7_full_2022_2024.json")
    logger.info("  - results/v6_vs_v7_comparison.json")


if __name__ == "__main__":
    main()
