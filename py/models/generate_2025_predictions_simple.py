#!/usr/bin/env python3
"""
Simplified 2025 Hold-Out Prediction Aggregator

Since we already have state-space v2 predictions from the ablation study,
this script reformats them into the ensemble format and documents model status.

This is a stopgap solution until full R/Python model loading is implemented.

Author: Claude Code (Sonnet 4.5)
Date: November 9, 2025
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results"
OUTPUT_FILE = RESULTS_DIR / "ensemble_predictions_2025_holdout.csv"
METRICS_FILE = RESULTS_DIR / "ensemble_model_performance_2025.csv"


def load_state_space_predictions() -> pd.DataFrame:
    """Load existing state-space v2 predictions from ablation study."""
    logger.info("Loading state-space v2 predictions from ablation study...")

    ablation_file = RESULTS_DIR / "ablation_predictions_scenario_B.csv"
    df = pd.read_csv(ablation_file)

    # Reformat to ensemble format
    predictions = pd.DataFrame(
        {
            "game_id": df["game_id"],
            "player_id": df["player_id"],
            "player_name": df["player_name"],
            "season": df["season"],
            "week": df["week"],
            "model_name": "state_space_v2",
            "pred_mean": df["pred_yards"],
            "pred_std": (df["pred_upper_90"] - df["pred_lower_90"]) / (2 * 1.645),  # 90% CI
            "actual_yards": df["actual_yards"],
        }
    )

    logger.info(
        f"Loaded {len(predictions)} predictions from {predictions['player_id'].nunique()} QBs"
    )
    return predictions


def document_model_status():
    """Document which models are available and their status."""
    logger.info("\n" + "=" * 80)
    logger.info("MODEL AVAILABILITY STATUS")
    logger.info("=" * 80)

    models = [
        {
            "model_name": "state_space_v2",
            "status": "AVAILABLE",
            "source": "ablation_predictions_scenario_B.csv",
            "path": "models/bayesian/simplified_random_walk_v2.rds",
            "mae_2025": 47.77,
            "coverage_90": 0.865,
            "notes": "Canonical window (2018-2024 + H=4 decay), APPROVED for production",
        },
        {
            "model_name": "hierarchical_v2",
            "status": "FILE EXISTS, PREDICTIONS PENDING",
            "source": "models/bayesian/passing_yards_hierarchical_v2.rds",
            "path": "models/bayesian/passing_yards_hierarchical_v2.rds",
            "mae_2025": None,
            "coverage_90": None,
            "notes": "Trained with canonical window, requires R prediction script",
        },
        {
            "model_name": "informative_priors_v2",
            "status": "FILE EXISTS, PREDICTIONS PENDING",
            "source": "models/bayesian/passing_informative_priors_v2.rds",
            "path": "models/bayesian/passing_informative_priors_v2.rds",
            "mae_2025": None,
            "coverage_90": None,
            "notes": "Trained with canonical window, requires R prediction script",
        },
        {
            "model_name": "bnn_v3.1",
            "status": "FILE EXISTS, ARCHITECTURE NEEDED",
            "source": "models/bayesian/bnn_multioutput_passing_v1_canonical.pt",
            "path": "models/bayesian/bnn_multioutput_passing_v1_canonical.pt",
            "mae_2025": None,
            "coverage_90": 0.895,  # From calibration
            "notes": "PyTorch model, requires architecture class for loading",
        },
        {
            "model_name": "xgboost_v3",
            "status": "NOT APPLICABLE",
            "source": "models/xgboost/v3_production/model.json",
            "path": "models/xgboost/v3_production/model.json",
            "mae_2025": None,
            "coverage_90": None,
            "notes": "Game outcome model (binary), not for passing yards prediction",
        },
    ]

    for model in models:
        logger.info(f"\n{model['model_name']:25s}")
        logger.info(f"  Status:     {model['status']}")
        logger.info(f"  Source:     {model['source']}")
        if model["mae_2025"]:
            logger.info(f"  MAE (2025): {model['mae_2025']:.2f} yards")
        if model["coverage_90"]:
            logger.info(f"  Coverage:   {model['coverage_90']:.1%}")
        logger.info(f"  Notes:      {model['notes']}")

    logger.info("\n" + "=" * 80)

    return pd.DataFrame(models)


def calculate_performance(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate performance metrics for available models."""
    logger.info("\nCalculating performance metrics...")

    metrics = []

    for model_name in predictions_df["model_name"].unique():
        model_preds = predictions_df[predictions_df["model_name"] == model_name].copy()

        # Calculate errors
        model_preds["error"] = model_preds["actual_yards"] - model_preds["pred_mean"]
        model_preds["abs_error"] = model_preds["error"].abs()
        model_preds["sq_error"] = model_preds["error"] ** 2

        # Calculate 90% CI coverage
        if model_preds["pred_std"].notna().all():
            model_preds["pred_lower_90"] = (
                model_preds["pred_mean"] - 1.645 * model_preds["pred_std"]
            )
            model_preds["pred_upper_90"] = (
                model_preds["pred_mean"] + 1.645 * model_preds["pred_std"]
            )
            model_preds["covered_90"] = (
                model_preds["actual_yards"] >= model_preds["pred_lower_90"]
            ) & (model_preds["actual_yards"] <= model_preds["pred_upper_90"])
            coverage_90 = model_preds["covered_90"].mean()
        else:
            coverage_90 = np.nan

        # Aggregate metrics
        metrics.append(
            {
                "model_name": model_name,
                "n_predictions": len(model_preds),
                "n_games": model_preds["game_id"].nunique(),
                "n_qbs": model_preds["player_id"].nunique(),
                "mae": model_preds["abs_error"].mean(),
                "rmse": np.sqrt(model_preds["sq_error"].mean()),
                "coverage_90": coverage_90,
                "mean_interval_width": (
                    (model_preds["pred_std"] * 2 * 1.645).mean()
                    if model_preds["pred_std"].notna().all()
                    else np.nan
                ),
                "mean_pred": model_preds["pred_mean"].mean(),
                "mean_actual": model_preds["actual_yards"].mean(),
                "status": "COMPLETE",
            }
        )

    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.sort_values("mae")

    logger.info("\nPerformance Metrics (2025 Hold-Out):")
    logger.info("=" * 80)
    for _, row in metrics_df.iterrows():
        logger.info(
            f"{row['model_name']:25s} | "
            f"MAE: {row['mae']:6.2f} | "
            f"RMSE: {row['rmse']:6.2f} | "
            f"Coverage: {row['coverage_90']:5.1%} | "
            f"N: {row['n_predictions']:3.0f}"
        )
    logger.info("=" * 80)

    return metrics_df


def main():
    """Main execution."""
    logger.info("=" * 80)
    logger.info("SIMPLIFIED 2025 HOLD-OUT PREDICTION AGGREGATOR")
    logger.info("=" * 80)

    # Document model status
    model_status = document_model_status()

    # Load available predictions (state-space only for now)
    predictions_df = load_state_space_predictions()

    # Calculate performance
    metrics_df = calculate_performance(predictions_df)

    # Save outputs
    logger.info(f"\nSaving predictions to {OUTPUT_FILE}...")
    predictions_df.to_csv(OUTPUT_FILE, index=False)

    logger.info(f"Saving metrics to {METRICS_FILE}...")
    metrics_df.to_csv(METRICS_FILE, index=False)

    # Save model status
    status_file = RESULTS_DIR / "ensemble_model_status_2025.csv"
    logger.info(f"Saving model status to {status_file}...")
    model_status.to_csv(status_file, index=False)

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Models with predictions: {predictions_df['model_name'].nunique()}/5")
    logger.info(f"Total predictions: {len(predictions_df)}")
    logger.info(f"Games: {predictions_df['game_id'].nunique()}")
    logger.info(f"QBs: {predictions_df['player_id'].nunique()}")
    logger.info(f"Weeks: {predictions_df['week'].min()}-{predictions_df['week'].max()}")
    logger.info("\nOutput files:")
    logger.info(f"  - {OUTPUT_FILE}")
    logger.info(f"  - {METRICS_FILE}")
    logger.info(f"  - {status_file}")
    logger.info("=" * 80)

    logger.info("\nNEXT STEPS:")
    logger.info(
        "1. Implement R model prediction scripts for hierarchical_v2 and informative_priors_v2"
    )
    logger.info("2. Create BNN architecture loader for bnn_v3.1 predictions")
    logger.info("3. Retrain XGBoost for passing yards (if needed for ensemble)")
    logger.info("4. Once all predictions available, optimize ensemble weights")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
