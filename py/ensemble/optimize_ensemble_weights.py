#!/usr/bin/env python3
"""
Optimize 5-Way Ensemble Weights via Inverse Variance Weighting

Implements 3 weighting strategies for 5-way ensemble:
1. Inverse Variance: w_i = (1/σ_i^2) / Σ(1/σ_j^2)
2. Equal Weighting: w_i = 1/N (baseline)
3. Performance-Based: w_i = (1/MAE_i) / Σ(1/MAE_j)

Target ensemble configuration (from MODEL_REGISTRY.yaml):
- State-Space v2: 30% (best MAE 47.77)
- BNN v3.1: 25% (calibrated)
- Hierarchical v2: 25%
- XGBoost v2: 20% (baseline)
- Informative Priors v2: 0% (redundant with Hierarchical)

Usage:
    python py/ensemble/optimize_ensemble_weights.py \
        --predictions analysis/results/ensemble_predictions_2025_holdout.csv \
        --output models/ensemble_weights_2025_optimized.json

Expected Performance:
- Ensemble MAE < 45.2 yards (target from MODEL_REGISTRY.yaml)
- 90% calibration coverage ≥ 89%
- Sharpe ratio estimation for betting

Author: NFL Analytics Research Team
Date: November 9, 2025
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class ModelPerformance:
    """Performance metrics for a single model."""

    name: str
    mae: float
    rmse: float
    variance: float  # σ^2
    std: float  # σ
    calibration_90: float  # 90% prediction interval coverage
    bias: float  # Mean prediction error
    n_predictions: int


@dataclass
class WeightingResult:
    """Results from a weighting method."""

    method: str
    weights: dict[str, float]
    ensemble_mae: float
    ensemble_rmse: float
    ensemble_std: float
    calibration_90: float
    improvement_vs_best_single: float
    sharpe_ratio: float | None = None


@dataclass
class OptimizationResult:
    """Final optimization result."""

    optimal_method: str
    optimal_weights: dict[str, float]
    all_methods: list[WeightingResult]
    comparison_table: pd.DataFrame
    metadata: dict


# ============================================================================
# Core Functions
# ============================================================================


def load_predictions(predictions_path: Path) -> pd.DataFrame:
    """
    Load ensemble predictions from CSV.

    Expected columns:
    - game_id, player_id, actual_yards
    - pred_state_space, pred_bnn, pred_hierarchical, pred_xgboost, pred_informative
    - std_state_space, std_bnn, std_hierarchical, std_xgboost, std_informative

    Args:
        predictions_path: Path to predictions CSV

    Returns:
        DataFrame with predictions and actuals
    """
    if not predictions_path.exists():
        logger.error(f"Predictions file not found: {predictions_path}")
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    df = pd.read_csv(predictions_path)

    # Validate required columns
    required_cols = ["actual_yards"]
    pred_cols = [col for col in df.columns if col.startswith("pred_")]
    std_cols = [col for col in df.columns if col.startswith("std_")]

    if "actual_yards" not in df.columns:
        logger.error("Missing 'actual_yards' column")
        raise ValueError("Predictions must contain 'actual_yards' column")

    if len(pred_cols) == 0:
        logger.error("No prediction columns found (expected pred_*)")
        raise ValueError("Predictions must contain pred_* columns")

    logger.info(f"✓ Loaded {len(df)} predictions")
    logger.info(f"  Models: {len(pred_cols)}")
    logger.info(f"  Columns: {', '.join(pred_cols)}")

    return df


def compute_model_performance(
    df: pd.DataFrame, model_name: str, pred_col: str, std_col: str | None = None
) -> ModelPerformance:
    """
    Compute performance metrics for a single model.

    Args:
        df: Predictions dataframe
        model_name: Model name (e.g., "state_space_v2")
        pred_col: Column name for predictions
        std_col: Column name for uncertainties (optional)

    Returns:
        ModelPerformance object
    """
    actual = df["actual_yards"].values
    pred = df[pred_col].values

    errors = pred - actual
    abs_errors = np.abs(errors)
    squared_errors = errors**2

    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(squared_errors))
    variance = np.var(errors)
    std = np.std(errors)
    bias = np.mean(errors)

    # Calibration (90% prediction interval coverage)
    if std_col and std_col in df.columns:
        pred_std = df[std_col].values
        z_score = 1.645  # 90% confidence interval
        lower = pred - z_score * pred_std
        upper = pred + z_score * pred_std
        coverage = np.mean((actual >= lower) & (actual <= upper)) * 100
    else:
        # Fallback: use empirical standard deviation
        z_score = 1.645
        lower = pred - z_score * std
        upper = pred + z_score * std
        coverage = np.mean((actual >= lower) & (actual <= upper)) * 100

    return ModelPerformance(
        name=model_name,
        mae=mae,
        rmse=rmse,
        variance=variance,
        std=std,
        calibration_90=coverage,
        bias=bias,
        n_predictions=len(actual),
    )


def inverse_variance_weighting(performances: list[ModelPerformance]) -> dict[str, float]:
    """
    Compute weights using inverse variance weighting.

    w_i = (1/σ_i^2) / Σ(1/σ_j^2)

    Args:
        performances: List of ModelPerformance objects

    Returns:
        Dictionary of model_name -> weight
    """
    variances = np.array([p.variance for p in performances])

    # Avoid division by zero
    variances = np.maximum(variances, 1e-6)

    inv_variances = 1.0 / variances
    weights = inv_variances / inv_variances.sum()

    return {p.name: float(w) for p, w in zip(performances, weights)}


def equal_weighting(performances: list[ModelPerformance]) -> dict[str, float]:
    """
    Compute equal weights (baseline).

    w_i = 1/N

    Args:
        performances: List of ModelPerformance objects

    Returns:
        Dictionary of model_name -> weight
    """
    n = len(performances)
    weight = 1.0 / n
    return {p.name: weight for p in performances}


def performance_based_weighting(performances: list[ModelPerformance]) -> dict[str, float]:
    """
    Compute weights based on inverse MAE.

    w_i = (1/MAE_i) / Σ(1/MAE_j)

    Args:
        performances: List of ModelPerformance objects

    Returns:
        Dictionary of model_name -> weight
    """
    maes = np.array([p.mae for p in performances])

    # Avoid division by zero
    maes = np.maximum(maes, 1e-6)

    inv_maes = 1.0 / maes
    weights = inv_maes / inv_maes.sum()

    return {p.name: float(w) for p, w in zip(performances, weights)}


def compute_ensemble_predictions(
    df: pd.DataFrame, weights: dict[str, float], model_columns: dict[str, str]
) -> np.ndarray:
    """
    Compute weighted ensemble predictions.

    Args:
        df: Predictions dataframe
        weights: Dictionary of model_name -> weight
        model_columns: Dictionary of model_name -> column_name

    Returns:
        Array of ensemble predictions
    """
    ensemble_pred = np.zeros(len(df))

    for model_name, weight in weights.items():
        col = model_columns[model_name]
        ensemble_pred += weight * df[col].values

    return ensemble_pred


def evaluate_weighting_method(
    df: pd.DataFrame,
    weights: dict[str, float],
    model_columns: dict[str, str],
    std_columns: dict[str, str],
    method_name: str,
    best_single_mae: float,
) -> WeightingResult:
    """
    Evaluate a weighting method on hold-out data.

    Args:
        df: Predictions dataframe
        weights: Dictionary of model_name -> weight
        model_columns: Dictionary of model_name -> pred_column_name
        std_columns: Dictionary of model_name -> std_column_name
        method_name: Name of weighting method
        best_single_mae: MAE of best individual model

    Returns:
        WeightingResult object
    """
    # Compute ensemble predictions
    ensemble_pred = compute_ensemble_predictions(df, weights, model_columns)
    actual = df["actual_yards"].values

    # Compute ensemble uncertainty (variance pooling)
    ensemble_var = 0.0
    for model_name, weight in weights.items():
        std_col = std_columns.get(model_name)
        if std_col and std_col in df.columns:
            model_var = df[std_col].values ** 2
            ensemble_var += (weight**2) * model_var

    ensemble_std = np.sqrt(ensemble_var.mean()) if ensemble_var.mean() > 0 else 0.0

    # Performance metrics
    errors = ensemble_pred - actual
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))

    # Calibration
    if ensemble_std > 0:
        z_score = 1.645  # 90% CI
        lower = ensemble_pred - z_score * np.sqrt(ensemble_var)
        upper = ensemble_pred + z_score * np.sqrt(ensemble_var)
        calibration = np.mean((actual >= lower) & (actual <= upper)) * 100
    else:
        calibration = 0.0

    # Sharpe ratio estimation (simplified)
    # Sharpe ≈ E[returns] / σ[returns]
    # Approximate as: mean(correct_direction) / std(errors)
    correct_direction = np.mean(np.sign(ensemble_pred - 250) == np.sign(actual - 250))
    sharpe = correct_direction / (np.std(errors) / 100.0) if np.std(errors) > 0 else 0.0

    improvement = best_single_mae - mae

    return WeightingResult(
        method=method_name,
        weights=weights,
        ensemble_mae=mae,
        ensemble_rmse=rmse,
        ensemble_std=ensemble_std,
        calibration_90=calibration,
        improvement_vs_best_single=improvement,
        sharpe_ratio=sharpe,
    )


def select_optimal_method(results: list[WeightingResult]) -> WeightingResult:
    """
    Select optimal weighting method based on multiple criteria.

    Selection criteria (in order of priority):
    1. Calibration ≥ 89%
    2. Lowest MAE
    3. Highest Sharpe ratio

    Args:
        results: List of WeightingResult objects

    Returns:
        Best WeightingResult
    """
    # Filter by calibration threshold
    calibrated = [r for r in results if r.calibration_90 >= 89.0]

    if calibrated:
        # Among well-calibrated models, select lowest MAE
        best = min(calibrated, key=lambda r: r.ensemble_mae)
        logger.info(f"✓ Selected {best.method} (calibrated, lowest MAE)")
    else:
        # Fallback: select lowest MAE (even if not well-calibrated)
        best = min(results, key=lambda r: r.ensemble_mae)
        logger.warning("⚠ No method meets calibration threshold (89%)")
        logger.warning(
            f"  Selected {best.method} (lowest MAE, calibration: {best.calibration_90:.1f}%)"
        )

    return best


def create_comparison_table(
    performances: list[ModelPerformance], results: list[WeightingResult]
) -> pd.DataFrame:
    """
    Create comparison table of all methods.

    Args:
        performances: Individual model performances
        results: Weighting method results

    Returns:
        Comparison DataFrame
    """
    # Individual models
    individual_rows = []
    for p in performances:
        individual_rows.append(
            {
                "model": p.name,
                "type": "individual",
                "mae_yards": p.mae,
                "rmse_yards": p.rmse,
                "calibration_90": p.calibration_90,
                "sharpe_ratio": None,
                "improvement_vs_best": 0.0,
            }
        )

    # Ensemble methods
    ensemble_rows = []
    for r in results:
        ensemble_rows.append(
            {
                "model": f"ensemble_{r.method}",
                "type": "ensemble",
                "mae_yards": r.ensemble_mae,
                "rmse_yards": r.ensemble_rmse,
                "calibration_90": r.calibration_90,
                "sharpe_ratio": r.sharpe_ratio,
                "improvement_vs_best": r.improvement_vs_best_single,
            }
        )

    df = pd.DataFrame(individual_rows + ensemble_rows)
    df = df.sort_values("mae_yards")

    return df


# ============================================================================
# Main Optimization Pipeline
# ============================================================================


def optimize_ensemble_weights(
    predictions_path: Path,
    output_path: Path,
    target_mae: float = 45.2,
    target_calibration: float = 89.0,
) -> OptimizationResult:
    """
    Main optimization pipeline.

    Args:
        predictions_path: Path to predictions CSV
        output_path: Path to save optimized weights JSON
        target_mae: Target ensemble MAE (from MODEL_REGISTRY.yaml)
        target_calibration: Target 90% calibration coverage

    Returns:
        OptimizationResult with optimal weights and comparison
    """
    logger.info("=" * 80)
    logger.info("5-WAY ENSEMBLE WEIGHT OPTIMIZATION")
    logger.info("=" * 80)

    # Load predictions
    logger.info("\n[1/5] Loading predictions...")
    df = load_predictions(predictions_path)

    # Identify model columns
    pred_cols = [col for col in df.columns if col.startswith("pred_")]
    std_cols = [col for col in df.columns if col.startswith("std_")]

    # Map model names to columns
    model_columns = {}
    std_columns = {}
    for col in pred_cols:
        model_name = col.replace("pred_", "")
        model_columns[model_name] = col

        # Find corresponding std column
        std_col = f"std_{model_name}"
        if std_col in std_cols:
            std_columns[model_name] = std_col

    logger.info(f"  Models detected: {list(model_columns.keys())}")

    # Compute individual model performances
    logger.info("\n[2/5] Computing individual model performances...")
    performances = []
    for model_name, pred_col in model_columns.items():
        std_col = std_columns.get(model_name)
        perf = compute_model_performance(df, model_name, pred_col, std_col)
        performances.append(perf)
        logger.info(
            f"  {model_name:25s} | MAE: {perf.mae:6.2f} | RMSE: {perf.rmse:6.2f} | Cal: {perf.calibration_90:5.1f}%"
        )

    best_single_mae = min(p.mae for p in performances)
    logger.info(f"\n  Best single model MAE: {best_single_mae:.2f} yards")

    # Compute weights using all 3 methods
    logger.info("\n[3/5] Computing ensemble weights...")

    weights_inverse_var = inverse_variance_weighting(performances)
    logger.info("  Inverse Variance:")
    for model, weight in weights_inverse_var.items():
        logger.info(f"    {model:25s}: {weight:5.3f}")

    weights_equal = equal_weighting(performances)
    logger.info("  Equal:")
    for model, weight in weights_equal.items():
        logger.info(f"    {model:25s}: {weight:5.3f}")

    weights_performance = performance_based_weighting(performances)
    logger.info("  Performance-Based (Inverse MAE):")
    for model, weight in weights_performance.items():
        logger.info(f"    {model:25s}: {weight:5.3f}")

    # Evaluate all methods
    logger.info("\n[4/5] Evaluating weighting methods...")
    results = [
        evaluate_weighting_method(
            df, weights_inverse_var, model_columns, std_columns, "inverse_variance", best_single_mae
        ),
        evaluate_weighting_method(
            df, weights_equal, model_columns, std_columns, "equal", best_single_mae
        ),
        evaluate_weighting_method(
            df,
            weights_performance,
            model_columns,
            std_columns,
            "performance_based",
            best_single_mae,
        ),
    ]

    for r in results:
        logger.info(
            f"  {r.method:20s} | MAE: {r.ensemble_mae:6.2f} | RMSE: {r.ensemble_rmse:6.2f} | "
            f"Cal: {r.calibration_90:5.1f}% | Sharpe: {r.sharpe_ratio:5.2f} | "
            f"Δ: {r.improvement_vs_best_single:+6.2f}"
        )

    # Select optimal method
    logger.info("\n[5/5] Selecting optimal method...")
    optimal = select_optimal_method(results)

    logger.info(f"\n✓ OPTIMAL METHOD: {optimal.method.upper()}")
    logger.info(f"  MAE: {optimal.ensemble_mae:.2f} yards (target: {target_mae:.2f})")
    logger.info(f"  Calibration: {optimal.calibration_90:.1f}% (target: {target_calibration:.1f}%)")
    logger.info(f"  Improvement vs best single: {optimal.improvement_vs_best_single:+.2f} yards")
    logger.info(f"  Sharpe ratio: {optimal.sharpe_ratio:.2f}")

    # Create comparison table
    comparison = create_comparison_table(performances, results)

    # Save results
    logger.info(f"\n[SAVE] Writing results to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_json = {
        "method": optimal.method,
        "weights": optimal.weights,
        "ensemble_mae": float(optimal.ensemble_mae),
        "ensemble_rmse": float(optimal.ensemble_rmse),
        "ensemble_std": float(optimal.ensemble_std),
        "calibration_90": float(optimal.calibration_90),
        "sharpe_ratio": float(optimal.sharpe_ratio) if optimal.sharpe_ratio else None,
        "improvement_vs_best_single": float(optimal.improvement_vs_best_single),
        "target_mae": target_mae,
        "target_calibration": target_calibration,
        "meets_target_mae": bool(optimal.ensemble_mae < target_mae),
        "meets_target_calibration": bool(optimal.calibration_90 >= target_calibration),
        "all_methods": [
            {
                "method": r.method,
                "mae": float(r.ensemble_mae),
                "rmse": float(r.ensemble_rmse),
                "calibration_90": float(r.calibration_90),
                "sharpe_ratio": float(r.sharpe_ratio) if r.sharpe_ratio else None,
                "improvement": float(r.improvement_vs_best_single),
            }
            for r in results
        ],
        "individual_models": [
            {
                "name": p.name,
                "mae": float(p.mae),
                "rmse": float(p.rmse),
                "calibration_90": float(p.calibration_90),
                "variance": float(p.variance),
                "std": float(p.std),
                "bias": float(p.bias),
            }
            for p in performances
        ],
        "metadata": {
            "n_predictions": len(df),
            "n_models": len(performances),
            "best_single_model": min(performances, key=lambda p: p.mae).name,
            "best_single_mae": float(best_single_mae),
            "optimization_date": "2025-11-09",
            "canonical_window": "2018-2024 + H=4 decay",
        },
    }

    with open(output_path, "w") as f:
        json.dump(result_json, f, indent=2)

    logger.info(f"✓ Saved to {output_path}")

    # Save comparison table
    comparison_path = output_path.parent / f"{output_path.stem}_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    logger.info(f"✓ Saved comparison to {comparison_path}")

    return OptimizationResult(
        optimal_method=optimal.method,
        optimal_weights=optimal.weights,
        all_methods=results,
        comparison_table=comparison,
        metadata=result_json["metadata"],
    )


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Optimize 5-way ensemble weights via inverse variance weighting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize weights on 2025 hold-out
  python py/ensemble/optimize_ensemble_weights.py \\
      --predictions analysis/results/ensemble_predictions_2025_holdout.csv \\
      --output models/ensemble_weights_2025_optimized.json

  # With custom targets
  python py/ensemble/optimize_ensemble_weights.py \\
      --predictions analysis/results/ensemble_predictions_2025_holdout.csv \\
      --output models/ensemble_weights_2025_optimized.json \\
      --target-mae 45.0 \\
      --target-calibration 90.0
        """,
    )

    parser.add_argument(
        "--predictions", type=Path, required=True, help="Path to ensemble predictions CSV"
    )

    parser.add_argument(
        "--output", type=Path, required=True, help="Path to save optimized weights JSON"
    )

    parser.add_argument(
        "--target-mae",
        type=float,
        default=45.2,
        help="Target ensemble MAE (default: 45.2 from MODEL_REGISTRY.yaml)",
    )

    parser.add_argument(
        "--target-calibration",
        type=float,
        default=89.0,
        help="Target 90%% calibration coverage (default: 89.0)",
    )

    args = parser.parse_args()

    # Run optimization
    result = optimize_ensemble_weights(
        predictions_path=args.predictions,
        output_path=args.output,
        target_mae=args.target_mae,
        target_calibration=args.target_calibration,
    )

    # Print summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"\nOptimal Method: {result.optimal_method.upper()}")
    print("\nOptimal Weights:")
    for model, weight in result.optimal_weights.items():
        print(f"  {model:25s}: {weight:5.1%}")

    print("\nPerformance Comparison:")
    print(result.comparison_table.to_string(index=False))

    print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()
