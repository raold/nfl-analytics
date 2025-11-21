#!/usr/bin/env python3
"""
Quick v6 vs v7 Comparison - 2024 Season Sample
===============================================

Runs v6_baseline and v7_full on 2024 Week 10-12 games (recent sample)
to generate comparison metrics quickly (15-20 min instead of 16 hours).

Purpose: Provide immediate insights for go/no-go decision on v7.0.

Author: Claude Code (Sonnet 4.5)
Date: November 19, 2025
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy import stats

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from py.ensemble.enhanced_ensemble_v7_production import EnhancedEnsembleV7

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database config
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "dbname": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",  # pragma: allowlist secret
}


def get_test_games(season: int, week_start: int, week_end: int):
    """Get test games for validation."""
    conn = psycopg2.connect(**DB_CONFIG)

    query = """
    SELECT DISTINCT
        pgs.player_id,
        pgs.season,
        pgs.week,
        pgs.stat_yards as actual
    FROM mart.player_game_stats pgs
    WHERE pgs.season = %s
        AND pgs.week >= %s
        AND pgs.week <= %s
        AND pgs.stat_category = 'passing'
        AND pgs.stat_yards IS NOT NULL
        AND pgs.stat_attempts >= 10  -- Minimum 10 attempts
    ORDER BY pgs.season, pgs.week, pgs.player_id
    """

    df = pd.read_sql_query(query, conn, params=(season, week_start, week_end))
    conn.close()

    logger.info(f"Loaded {len(df)} QB games from {season} W{week_start}-W{week_end}")
    return df


def run_validation(config_name: str, test_df: pd.DataFrame, enable_meta: bool, enable_wls: bool):
    """
    Run validation for a specific configuration.

    Args:
        config_name: Configuration name (e.g., "v6_baseline", "v7_full")
        test_df: Test games DataFrame
        enable_meta: Enable meta-learning
        enable_wls: Enable WLS recency adjustment

    Returns:
        DataFrame with predictions
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Configuration: {config_name}")
    logger.info(f"  Meta-learning: {'ENABLED' if enable_meta else 'DISABLED'}")
    logger.info(f"  WLS recency: {'ENABLED' if enable_wls else 'DISABLED'}")
    logger.info(f"{'=' * 60}")

    # Initialize ensemble
    ensemble = EnhancedEnsembleV7(
        db_config=DB_CONFIG,
        enable_meta_learning=enable_meta,
        enable_wls_recency=enable_wls,
    )

    predictions = []
    failures = 0

    for idx, row in test_df.iterrows():
        player_id = row["player_id"]
        season = int(row["season"])
        week = int(row["week"])
        actual = float(row["actual"])

        try:
            # Generate prediction
            result = ensemble.predict(
                player_id=player_id,
                season=season,
                week=week,
                stat_type="passing_yards",
            )

            if result and "ensemble_mean" in result:
                predictions.append(
                    {
                        "player_id": player_id,
                        "season": season,
                        "week": week,
                        "prediction": result["ensemble_mean"],
                        "uncertainty": result["ensemble_std"],
                        "actual": actual,
                        "n_components": result.get("n_components", 0),
                    }
                )
            else:
                failures += 1

        except Exception as e:
            logger.warning(f"  Failed for {player_id} W{week}: {str(e)[:100]}")
            failures += 1

        # Progress logging
        if (idx + 1) % 10 == 0:
            logger.info(f"  Progress: {idx + 1}/{len(test_df)} games ({failures} failures)")

    logger.info(f"\n  Complete: {len(predictions)} successful, {failures} failed")

    return pd.DataFrame(predictions)


def calculate_metrics(df: pd.DataFrame):
    """Calculate validation metrics from predictions DataFrame."""
    predictions = df["prediction"].values
    actuals = df["actual"].values
    uncertainties = df["uncertainty"].values

    # MAE
    mae = np.mean(np.abs(predictions - actuals))

    # RMSE
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

    # R²
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Calibration
    z_90 = 1.645
    errors = np.abs(predictions - actuals)
    calibration_90 = np.mean(errors <= z_90 * uncertainties) * 100

    return {
        "n_predictions": len(predictions),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "calibration_90": float(calibration_90),
        "mean_prediction": float(np.mean(predictions)),
        "mean_actual": float(np.mean(actuals)),
    }


def diebold_mariano_test(errors1: np.ndarray, errors2: np.ndarray):
    """Diebold-Mariano test for predictive accuracy."""
    d = errors1 - errors2
    n = len(d)

    if n == 0:
        return (np.nan, np.nan)

    d_bar = np.mean(d)
    gamma_0 = np.var(d, ddof=1)
    dm_statistic = d_bar / np.sqrt(gamma_0 / n)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_statistic)))

    return (float(dm_statistic), float(p_value))


def main():
    logger.info("=" * 80)
    logger.info("QUICK v6 vs v7 COMPARISON (2024 Sample)")
    logger.info("=" * 80)

    # Get test data (2024 weeks 10-12, ~50 games)
    test_df = get_test_games(season=2024, week_start=10, week_end=12)

    if len(test_df) == 0:
        logger.error("No test games found!")
        return

    # Run v6_baseline
    v6_df = run_validation(
        config_name="v6_baseline",
        test_df=test_df,
        enable_meta=False,
        enable_wls=False,
    )

    # Run v7_full
    v7_df = run_validation(
        config_name="v7_full",
        test_df=test_df,
        enable_meta=True,
        enable_wls=True,
    )

    # Calculate metrics
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)

    v6_metrics = calculate_metrics(v6_df)
    v7_metrics = calculate_metrics(v7_df)

    logger.info("\nv6_baseline:")
    logger.info(f"  Games: {v6_metrics['n_predictions']}")
    logger.info(f"  MAE: {v6_metrics['mae']:.2f} yards")
    logger.info(f"  RMSE: {v6_metrics['rmse']:.2f} yards")
    logger.info(f"  R²: {v6_metrics['r2']:.4f}")
    logger.info(f"  Calibration (90% CI): {v6_metrics['calibration_90']:.1f}%")

    logger.info("\nv7_full:")
    logger.info(f"  Games: {v7_metrics['n_predictions']}")
    logger.info(f"  MAE: {v7_metrics['mae']:.2f} yards")
    logger.info(f"  RMSE: {v7_metrics['rmse']:.2f} yards")
    logger.info(f"  R²: {v7_metrics['r2']:.4f}")
    logger.info(f"  Calibration (90% CI): {v7_metrics['calibration_90']:.1f}%")

    # Statistical test
    logger.info("\n" + "=" * 80)
    logger.info("STATISTICAL COMPARISON")
    logger.info("=" * 80)

    errors_v6 = np.abs(v6_df["prediction"].values - v6_df["actual"].values)
    errors_v7 = np.abs(v7_df["prediction"].values - v7_df["actual"].values)

    dm_stat, p_value = diebold_mariano_test(errors_v6, errors_v7)
    mae_diff = v7_metrics["mae"] - v6_metrics["mae"]

    logger.info(f"\nMAE Difference: {mae_diff:+.2f} yards")
    logger.info(f"DM Statistic: {dm_stat:.4f}")
    logger.info(f"p-value: {p_value:.6f}")

    if p_value < 0.05:
        logger.info("✅ Statistically significant (p < 0.05)")
    else:
        logger.info("❌ Not statistically significant (p >= 0.05)")

    # Save results
    results = {
        "test_period": "2024 W10-W12",
        "test_games": len(test_df),
        "timestamp": datetime.now().isoformat(),
        "v6_baseline": v6_metrics,
        "v7_full": v7_metrics,
        "comparison": {
            "mae_difference": mae_diff,
            "dm_statistic": dm_stat,
            "p_value": p_value,
            "significant_05": p_value < 0.05,
        },
    }

    output_file = Path("results/quick_v6_v7_comparison_2024.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
