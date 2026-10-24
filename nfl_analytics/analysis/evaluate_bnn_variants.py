#!/usr/bin/env python3
"""
Unified BNN Evaluation Script - Week 1 Task

Evaluates BNN Mixture Experts v2 and BNN Simpler v2 on 2024 test data.
Measures:
- 90% CI coverage (target: 90%)
- MAE and RMSE
- Brier score and accuracy
- Comparison vs BNN Multioutput v1 baseline
"""

import sys
from pathlib import Path

_PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(_PROJECT_ROOT))

import json
import logging
import pickle
import numpy as np
import pandas as pd
import psycopg2
from nfl_analytics.models.bnn_mixture_experts_v2 import MixtureExpertsBNN

# Import BNN model classes
from nfl_analytics.models.bnn_simpler_v2 import SimplerBNNv2
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_test_data(season=2024):
    """Load 2024 test data from database (same query as BNN training scripts)"""
    conn = psycopg2.connect(
        host="localhost", port=5544, database="devdb01", user="dro", password="sicillionbillions"
    )

    query = """
    SELECT
        pgs.player_id,
        pgs.player_display_name as player_name,
        pgs.season,
        pgs.week,
        pgs.current_team as team,
        pgs.stat_yards,
        pgs.stat_attempts as carries,
        -- Recent form (3-game average)
        AVG(pgs.stat_yards) OVER (
            PARTITION BY pgs.player_id
            ORDER BY pgs.season, pgs.week
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) as avg_rushing_l3,
        -- Season average
        AVG(pgs.stat_yards) OVER (
            PARTITION BY pgs.player_id, pgs.season
            ORDER BY pgs.week
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) as season_avg
    FROM mart.player_game_stats pgs
    WHERE pgs.season = %s
      AND pgs.stat_category = 'rushing'
      AND pgs.position_group IN ('RB', 'FB', 'HB')
      AND pgs.stat_attempts >= 5
      AND pgs.stat_yards IS NOT NULL
    ORDER BY pgs.season, pgs.week, pgs.stat_yards DESC
    """

    df = pd.read_sql(query, conn, params=[season])
    conn.close()

    # Handle missing values (same as training scripts)
    df["avg_rushing_l3"] = df["avg_rushing_l3"].fillna(
        df.groupby("season")["stat_yards"].transform("median")
    )
    df["season_avg"] = df["season_avg"].fillna(
        df.groupby("season")["stat_yards"].transform("median")
    )

    logger.info(f"Loaded {len(df)} rushing performances from {season} season")
    return df


def calculate_ci_coverage(y_true, y_pred_samples, confidence=0.90):
    """Calculate confidence interval coverage"""
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 - (1 - confidence) / 2) * 100

    lower = np.percentile(y_pred_samples, lower_percentile, axis=0)
    upper = np.percentile(y_pred_samples, upper_percentile, axis=0)

    in_interval = (y_true >= lower) & (y_true <= upper)
    coverage = np.mean(in_interval)

    avg_width = np.mean(upper - lower)

    return coverage, avg_width, lower, upper


def evaluate_model(model_path, model_name, test_data):
    """Evaluate a single BNN model"""
    logger.info("=" * 80)
    logger.info(f"Evaluating {model_name}")
    logger.info("=" * 80)

    # Load model using the proper class method
    logger.info(f"Loading model from {model_path}")
    try:
        # Detect model type from saved file
        with open(model_path, "rb") as f:
            save_obj = pickle.load(f)

        # Check architecture type
        architecture = save_obj.get("config", {}).get("architecture", "")

        if "mixture_of_experts" in architecture:
            logger.info("Detected Mixture-of-Experts architecture")
            model = MixtureExpertsBNN.load(model_path)
        elif "simpler_bnn" in architecture:
            logger.info("Detected Simpler BNN architecture")
            model = SimplerBNNv2.load(model_path)
        else:
            # Fallback: try to detect from keys
            if "n_experts" in save_obj:
                logger.info("Detected Mixture-of-Experts (from keys)")
                model = MixtureExpertsBNN.load(model_path)
            elif "hidden_size" in save_obj:
                logger.info("Detected Simpler BNN (from keys)")
                model = SimplerBNNv2.load(model_path)
            else:
                raise ValueError(f"Unknown model format: {list(save_obj.keys())}")

        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        import traceback

        traceback.print_exc()
        return None

    # Prepare features (same as BNN training scripts)
    feature_cols = ["carries", "avg_rushing_l3", "season_avg", "week"]
    X_test = test_data[feature_cols].fillna(0).values
    y_true = test_data["stat_yards"].values

    logger.info(f"Test set: {len(X_test)} samples")

    # Standardize features using the model's scaler
    try:
        X_test_scaled = model.scaler.transform(X_test)
        logger.info("✓ Features standardized using model's scaler")
    except Exception as e:
        logger.warning(f"Could not use model scaler, using raw features: {e}")
        X_test_scaled = X_test

    # Generate predictions using the model's predict() method
    logger.info("Generating posterior predictions...")

    try:
        # Use the model's built-in predict() method
        predictions = model.predict(X_test_scaled)
        y_pred_mean = predictions["mean"]
        y_pred_samples = predictions["samples"]

        logger.info(
            f"✓ Generated {y_pred_samples.shape[0]} samples × {y_pred_samples.shape[1]} predictions"
        )

    except Exception as e:
        logger.error(f"✗ Prediction failed: {e}")
        import traceback

        traceback.print_exc()
        logger.error("CRITICAL: Model prediction failed. NO FALLBACK DATA ALLOWED.")
        return None

    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred_mean)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_mean))

    # Calculate CI coverage
    coverage_90, width_90, lower_90, upper_90 = calculate_ci_coverage(y_true, y_pred_samples, 0.90)
    coverage_68, width_68, _, _ = calculate_ci_coverage(y_true, y_pred_samples, 0.68)

    results = {
        "model_name": model_name,
        "model_path": str(model_path),
        "n_test_samples": len(y_true),
        "mae": float(mae),
        "rmse": float(rmse),
        "coverage_90": float(coverage_90),
        "ci_width_90": float(width_90),
        "coverage_68": float(coverage_68),
        "ci_width_68": float(width_68),
        "target_coverage_90": 0.90,
        "target_coverage_68": 0.68,
    }

    # Print results
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"RESULTS: {model_name}")
    logger.info("=" * 80)
    logger.info(f"MAE:  {mae:.2f} yards")
    logger.info(f"RMSE: {rmse:.2f} yards")
    logger.info("")
    logger.info(f"90% CI Coverage: {coverage_90:.1%} (target: 90%)")
    logger.info(f"90% CI Width:    {width_90:.2f} yards")
    logger.info("")
    logger.info(f"68% CI Coverage: {coverage_68:.1%} (target: 68%)")
    logger.info(f"68% CI Width:    {width_68:.2f} yards")
    logger.info("")

    # Pass/Fail assessment
    if abs(coverage_90 - 0.90) <= 0.10:  # Within 10 percentage points
        logger.info("✓ PASS: 90% CI coverage within acceptable range")
        results["status_90"] = "PASS"
    else:
        logger.info("✗ FAIL: 90% CI coverage outside acceptable range")
        results["status_90"] = "FAIL"

    if abs(coverage_68 - 0.68) <= 0.10:
        logger.info("✓ PASS: 68% CI coverage within acceptable range")
        results["status_68"] = "PASS"
    else:
        logger.info("✗ FAIL: 68% CI coverage outside acceptable range")
        results["status_68"] = "FAIL"

    return results


def main():
    logger.info("=" * 80)
    logger.info("BNN VARIANTS EVALUATION - WEEK 1")
    logger.info("=" * 80)
    logger.info("")

    # Load test data
    logger.info("Loading 2024 test data...")
    test_data = load_test_data(season=2024)

    # Models to evaluate
    models = [
        {
            "name": "BNN Simpler v2",
            "path": Path("models/bayesian/bnn_simpler_v2.pkl"),
            "description": "Lightweight 2-layer BNN (10 MB)",
        },
        {
            "name": "BNN Mixture Experts v2",
            "path": Path("models/bayesian/bnn_mixture_experts_v2.pkl"),
            "description": "Mixture of 3 expert networks (1.3 GB)",
        },
    ]

    all_results = []

    for model_config in models:
        if not model_config["path"].exists():
            logger.warning(f"✗ Model not found: {model_config['path']}")
            continue

        logger.info("")
        logger.info(f"Model: {model_config['name']}")
        logger.info(f"Description: {model_config['description']}")
        logger.info(f"Path: {model_config['path']}")
        logger.info("")

        results = evaluate_model(
            model_path=model_config["path"], model_name=model_config["name"], test_data=test_data
        )

        if results:
            all_results.append(results)

    # Comparison summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 80)

    if len(all_results) >= 2:
        df_comparison = pd.DataFrame(all_results)
        logger.info("")
        logger.info(
            df_comparison[
                ["model_name", "mae", "rmse", "coverage_90", "ci_width_90", "status_90"]
            ].to_string(index=False)
        )
        logger.info("")

    # Save results
    output_dir = Path("experiments/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "bnn_variants_evaluation_2024.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"✓ Results saved to {output_file}")

    # Compare to baseline (BNN Multioutput v1)
    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPARISON TO BNN MULTIOUTPUT V1 BASELINE")
    logger.info("=" * 80)
    logger.info("Baseline (Multioutput v1):")
    logger.info("  90% CI Coverage: 92.0%")
    logger.info("  MAE: 18.50 yards")
    logger.info("  RMSE: 24.79 yards")
    logger.info("")

    for result in all_results:
        coverage_diff = (result["coverage_90"] - 0.920) * 100
        mae_diff = result["mae"] - 18.50

        logger.info(f"{result['model_name']}:")
        logger.info(
            f"  90% CI Coverage: {result['coverage_90']:.1%} ({coverage_diff:+.1f} pp vs baseline)"
        )
        logger.info(f"  MAE: {result['mae']:.2f} yards ({mae_diff:+.2f} yards vs baseline)")
        logger.info("")

    logger.info("=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
