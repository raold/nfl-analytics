#!/usr/bin/env python3
"""
Optimize 4-Way Ensemble Weights - Production Implementation

Implements 3 weighting strategies for the 4-way ensemble:
1. Inverse Variance Weighting: w_i = (1/σ²_i) / Σ(1/σ²_j)
2. Equal Weighting: w_i = 1/4 (baseline)
3. Performance-Based Weighting: w_i = (1/MAE_i) / Σ(1/MAE_j)

Current weights (arbitrary):
- State-Space v2: 30%
- Hierarchical v2: 30%
- XGBoost v2: 25%
- Informative Priors v2: 15%

Usage:
    python scripts/optimize_4way_weights.py \\
        --predictions analysis/results/ensemble_4way_backtest_2022_2024_FIXED.csv \\
        --output models/ensemble/weights_optimized_2022_2024.json

Author: Claude Code (Sonnet 4.5)
Date: November 15, 2025
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


MODEL_NAMES = ["state_space", "hierarchical", "xgboost", "informative_priors"]

MODEL_DISPLAY_NAMES = {
    "state_space": "State-Space v2",
    "hierarchical": "Hierarchical v2",
    "xgboost": "XGBoost v2",
    "informative_priors": "Informative Priors v2",
}


def load_predictions(csv_path: Path) -> pd.DataFrame:
    """Load predictions from 4-way backtest CSV."""
    if csv_path.suffix != ".csv":
        raise ValueError(f"Expected a .csv file, got: {csv_path.suffix}")
    logger.info(f"Loading predictions from: {csv_path}")
    df = pd.read_csv(csv_path)

    logger.info(f"  Loaded {len(df)} predictions")
    logger.info(f"  Seasons: {df['season'].min()} - {df['season'].max()}")
    logger.info(f"  Players: {df['player_id'].nunique()}")

    return df


def calculate_model_performance(df: pd.DataFrame, model_name: str) -> dict:
    """Calculate performance metrics for a single model."""
    mean_col = f"{model_name}_mean"
    std_col = f"{model_name}_std"

    # Filter to predictions where this model contributed
    model_df = df[df[mean_col].notna()].copy()

    if len(model_df) == 0:
        logger.warning(f"No predictions found for {model_name}")
        return None

    # Calculate errors
    errors = model_df["actual_yards"] - model_df[mean_col]
    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors**2).mean())
    bias = errors.mean()

    # Calculate average uncertainty
    avg_std = model_df[std_col].mean()
    avg_variance = (model_df[std_col] ** 2).mean()

    # Calculate calibration (90% CI)
    quantile_10 = model_df[mean_col] - 1.645 * model_df[std_col]
    quantile_90 = model_df[mean_col] + 1.645 * model_df[std_col]
    within_ci = (
        (model_df["actual_yards"] >= quantile_10) & (model_df["actual_yards"] <= quantile_90)
    ).sum()
    calibration_90 = within_ci / len(model_df) * 100

    return {
        "name": model_name,
        "display_name": MODEL_DISPLAY_NAMES[model_name],
        "n_predictions": len(model_df),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "avg_std": avg_std,
        "avg_variance": avg_variance,
        "calibration_90": calibration_90,
    }


def inverse_variance_weighting(performances: list[dict]) -> dict[str, float]:
    """Calculate weights via inverse variance: w_i = (1/σ²_i) / Σ(1/σ²_j)."""
    inverse_variances = np.array([1.0 / p["avg_variance"] for p in performances])
    weights = inverse_variances / inverse_variances.sum()

    return {p["name"]: w for p, w in zip(performances, weights)}


def equal_weighting(performances: list[dict]) -> dict[str, float]:
    """Equal weights: w_i = 1/N."""
    n = len(performances)
    return {p["name"]: 1.0 / n for p in performances}


def performance_weighting(performances: list[dict]) -> dict[str, float]:
    """Performance-based weights: w_i = (1/MAE_i) / Σ(1/MAE_j)."""
    inverse_maes = np.array([1.0 / p["mae"] for p in performances])
    weights = inverse_maes / inverse_maes.sum()

    return {p["name"]: w for p, w in zip(performances, weights)}


def evaluate_ensemble(df: pd.DataFrame, weights: dict[str, float]) -> dict:
    """Evaluate ensemble performance with given weights."""
    # Filter to predictions where all models contributed
    valid_df = df.copy()
    for model in MODEL_NAMES:
        valid_df = valid_df[valid_df[f"{model}_mean"].notna()]

    if len(valid_df) == 0:
        logger.error("No predictions with all 4 models!")
        return None

    # Calculate weighted ensemble mean
    ensemble_mean = np.zeros(len(valid_df))
    for model, weight in weights.items():
        ensemble_mean += weight * valid_df[f"{model}_mean"].values

    # Calculate ensemble variance (within + between)
    within_var = np.zeros(len(valid_df))
    for model, weight in weights.items():
        within_var += weight * (valid_df[f"{model}_std"].values ** 2)

    between_var = np.zeros(len(valid_df))
    for model, weight in weights.items():
        between_var += weight * ((valid_df[f"{model}_mean"].values - ensemble_mean) ** 2)

    ensemble_var = within_var + between_var
    ensemble_std = np.sqrt(ensemble_var)

    # Calculate errors
    errors = valid_df["actual_yards"].values - ensemble_mean
    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors**2).mean())

    # Calculate calibration
    quantile_10 = ensemble_mean - 1.645 * ensemble_std
    quantile_90 = ensemble_mean + 1.645 * ensemble_std
    within_ci = (
        (valid_df["actual_yards"].values >= quantile_10)
        & (valid_df["actual_yards"].values <= quantile_90)
    ).sum()
    calibration_90 = within_ci / len(valid_df) * 100

    return {
        "n_predictions": len(valid_df),
        "mae": mae,
        "rmse": rmse,
        "avg_std": ensemble_std.mean(),
        "calibration_90": calibration_90,
    }


def optimize_weights(df: pd.DataFrame) -> dict:
    """Run all 3 weighting methods and compare results."""
    logger.info("\n" + "=" * 80)
    logger.info("ENSEMBLE WEIGHT OPTIMIZATION")
    logger.info("=" * 80)

    # Step 1: Calculate individual model performance
    logger.info("\n📊 Individual Model Performance:")
    performances = []
    for model in MODEL_NAMES:
        perf = calculate_model_performance(df, model)
        if perf:
            performances.append(perf)
            logger.info(
                f"  {perf['display_name']:25s} | "
                f"{perf['n_predictions']:4d} preds | "
                f"MAE: {perf['mae']:5.2f} | "
                f"RMSE: {perf['rmse']:5.2f} | "
                f"σ: {perf['avg_std']:5.2f} | "
                f"Cal: {perf['calibration_90']:5.1f}%"
            )

    if len(performances) < 4:
        logger.error("Need all 4 models to optimize!")
        return None

    # Step 2: Calculate weights using 3 methods
    methods = {
        "inverse_variance": inverse_variance_weighting(performances),
        "equal": equal_weighting(performances),
        "performance": performance_weighting(performances),
    }

    # Step 3: Evaluate each method
    logger.info("\n🔍 Evaluating Weight Methods:")
    results = {}
    for method_name, weights in methods.items():
        logger.info(f"\n  Method: {method_name}")
        logger.info("  Weights:")
        for model, weight in weights.items():
            logger.info(f"    {MODEL_DISPLAY_NAMES[model]:25s}: {weight:.3f} ({weight*100:.1f}%)")

        ensemble_perf = evaluate_ensemble(df, weights)
        logger.info(f"  Ensemble MAE: {ensemble_perf['mae']:.2f} yards")
        logger.info(f"  Ensemble RMSE: {ensemble_perf['rmse']:.2f} yards")
        logger.info(f"  Calibration: {ensemble_perf['calibration_90']:.1f}%")

        results[method_name] = {"weights": weights, "performance": ensemble_perf}

    # Step 4: Find best method (lowest MAE)
    best_method = min(results.keys(), key=lambda m: results[m]["performance"]["mae"])
    logger.info(f"\n✅ Best Method: {best_method}")
    logger.info(f"   MAE: {results[best_method]['performance']['mae']:.2f} yards")

    # Step 5: Compare to current arbitrary weights
    current_weights = {
        "state_space": 0.30,
        "hierarchical": 0.30,
        "xgboost": 0.25,
        "informative_priors": 0.15,
    }
    current_perf = evaluate_ensemble(df, current_weights)
    logger.info("\n📊 Current (Arbitrary) Weights Performance:")
    logger.info(f"   MAE: {current_perf['mae']:.2f} yards")
    logger.info(f"   Calibration: {current_perf['calibration_90']:.1f}%")

    improvement = current_perf["mae"] - results[best_method]["performance"]["mae"]
    improvement_pct = improvement / current_perf["mae"] * 100
    logger.info(f"\n💡 Improvement: {improvement:.2f} yards ({improvement_pct:.1f}%)")

    return {
        "best_method": best_method,
        "best_weights": results[best_method]["weights"],
        "best_performance": results[best_method]["performance"],
        "all_methods": results,
        "current_weights": current_weights,
        "current_performance": current_perf,
        "improvement_yards": improvement,
        "improvement_pct": improvement_pct,
        "individual_models": performances,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Optimize 4-way ensemble weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--predictions", type=Path, required=True, help="Path to 4-way backtest CSV file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/ensemble/weights_optimized.json"),
        help="Output JSON file for optimized weights",
    )

    args = parser.parse_args()

    # Load data
    df = load_predictions(args.predictions)

    # Optimize weights
    result = optimize_weights(df)

    if result is None:
        logger.error("Optimization failed!")
        return 1

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    output_data = {
        "best_method": result["best_method"],
        "recommended_weights": {
            MODEL_DISPLAY_NAMES[k]: float(v) for k, v in result["best_weights"].items()
        },
        "performance": {
            "mae": float(result["best_performance"]["mae"]),
            "rmse": float(result["best_performance"]["rmse"]),
            "calibration_90": float(result["best_performance"]["calibration_90"]),
            "n_predictions": int(result["best_performance"]["n_predictions"]),
        },
        "improvement_vs_current": {
            "mae_yards": float(result["improvement_yards"]),
            "mae_pct": float(result["improvement_pct"]),
        },
        "all_methods": {
            method: {
                "weights": {MODEL_DISPLAY_NAMES[k]: float(v) for k, v in data["weights"].items()},
                "mae": float(data["performance"]["mae"]),
                "rmse": float(data["performance"]["rmse"]),
                "calibration_90": float(data["performance"]["calibration_90"]),
            }
            for method, data in result["all_methods"].items()
        },
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\n✅ Results saved to: {args.output}")
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
