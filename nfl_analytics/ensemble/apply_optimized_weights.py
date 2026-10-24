#!/usr/bin/env python3
"""
Apply Optimized Ensemble Weights

Loads optimized weights from JSON and applies them to generate
ensemble predictions for new data.

Usage:
    # Apply weights to new predictions
    python nfl_analytics/ensemble/apply_optimized_weights.py \\
        --weights models/ensemble_weights_2025_optimized.json \\
        --predictions analysis/results/week11_predictions.csv \\
        --output analysis/results/week11_ensemble.csv

    # Quick check of weight configuration
    python nfl_analytics/ensemble/apply_optimized_weights.py \\
        --weights models/ensemble_weights_2025_optimized.json \\
        --info

Author: NFL Analytics Research Team
Date: November 9, 2025
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_weights(weights_path: Path) -> dict:
    """Load optimized weights from JSON."""
    with open(weights_path) as f:
        weights_data = json.load(f)

    logger.info(f"✓ Loaded weights from {weights_path}")
    logger.info(f"  Method: {weights_data['method']}")
    logger.info(f"  Optimization Date: {weights_data['metadata']['optimization_date']}")

    return weights_data


def apply_weights(predictions_df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """
    Apply optimized weights to predictions.

    Args:
        predictions_df: DataFrame with pred_* and std_* columns
        weights: Dictionary of model_name -> weight

    Returns:
        DataFrame with ensemble_pred and ensemble_std columns added
    """
    # Compute weighted ensemble prediction
    ensemble_pred = np.zeros(len(predictions_df))

    for model_name, weight in weights.items():
        pred_col = f"pred_{model_name}"
        if pred_col not in predictions_df.columns:
            logger.warning(f"Missing prediction column: {pred_col}")
            continue

        ensemble_pred += weight * predictions_df[pred_col].values

    predictions_df["ensemble_pred"] = ensemble_pred

    # Compute ensemble uncertainty (variance pooling)
    ensemble_var = np.zeros(len(predictions_df))

    for model_name, weight in weights.items():
        std_col = f"std_{model_name}"
        if std_col not in predictions_df.columns:
            logger.warning(f"Missing uncertainty column: {std_col}")
            continue

        model_var = predictions_df[std_col].values ** 2
        ensemble_var += (weight**2) * model_var

    predictions_df["ensemble_std"] = np.sqrt(ensemble_var)

    # Compute 90% prediction interval
    z_score = 1.645  # 90% CI
    predictions_df["ensemble_lower_90"] = ensemble_pred - z_score * predictions_df["ensemble_std"]
    predictions_df["ensemble_upper_90"] = ensemble_pred + z_score * predictions_df["ensemble_std"]

    logger.info(f"✓ Applied weights to {len(predictions_df)} predictions")

    return predictions_df


def display_weights_info(weights_data: dict):
    """Display weights configuration information."""
    print("\n" + "=" * 80)
    print("OPTIMIZED ENSEMBLE WEIGHTS")
    print("=" * 80)

    print(f"\nMethod: {weights_data['method'].upper()}")
    print(f"Optimization Date: {weights_data['metadata']['optimization_date']}")
    print(f"Training Window: {weights_data['metadata']['canonical_window']}")

    print("\nWeights:")
    for model, weight in sorted(weights_data["weights"].items(), key=lambda x: -x[1]):
        print(f"  {model:25s}: {weight:5.1%}")

    print("\nValidation Performance:")
    print(f"  MAE: {weights_data['ensemble_mae']:.2f} yards")
    print(f"  RMSE: {weights_data['ensemble_rmse']:.2f} yards")
    print(f"  Calibration (90%): {weights_data['calibration_90']:.1f}%")
    print(f"  Sharpe Ratio: {weights_data['sharpe_ratio']:.2f}")
    print(f"  Improvement vs Best Single: {weights_data['improvement_vs_best_single']:+.2f} yards")

    print("\nTargets:")
    print(f"  MAE Target: {weights_data['target_mae']:.2f} yards")
    print(f"  Calibration Target: {weights_data['target_calibration']:.1f}%")
    print(f"  Meets MAE Target: {'✓' if weights_data['meets_target_mae'] else '✗'}")
    print(f"  Meets Calibration Target: {'✓' if weights_data['meets_target_calibration'] else '✗'}")

    print("\nAll Methods Tested:")
    for method_data in weights_data["all_methods"]:
        print(
            f"  {method_data['method']:20s} | MAE: {method_data['mae']:6.2f} | "
            f"Cal: {method_data['calibration_90']:5.1f}% | "
            f"Sharpe: {method_data['sharpe_ratio']:5.2f}"
        )

    print("\nIndividual Model Performance:")
    for model_data in weights_data["individual_models"]:
        print(
            f"  {model_data['name']:25s} | MAE: {model_data['mae']:6.2f} | "
            f"RMSE: {model_data['rmse']:6.2f} | "
            f"Cal: {model_data['calibration_90']:5.1f}%"
        )

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Apply optimized ensemble weights to predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Apply weights to new predictions
  python nfl_analytics/ensemble/apply_optimized_weights.py \\
      --weights models/ensemble_weights_2025_optimized.json \\
      --predictions analysis/results/week11_predictions.csv \\
      --output analysis/results/week11_ensemble.csv

  # Display weights info
  python nfl_analytics/ensemble/apply_optimized_weights.py \\
      --weights models/ensemble_weights_2025_optimized.json \\
      --info
        """,
    )

    parser.add_argument(
        "--weights", type=Path, required=True, help="Path to optimized weights JSON"
    )

    parser.add_argument(
        "--predictions", type=Path, help="Path to predictions CSV (required unless --info)"
    )

    parser.add_argument(
        "--output", type=Path, help="Path to save ensemble predictions (required unless --info)"
    )

    parser.add_argument(
        "--info", action="store_true", help="Display weights configuration info and exit"
    )

    args = parser.parse_args()

    # Load weights
    weights_data = load_weights(args.weights)

    # Display info and exit if requested
    if args.info:
        display_weights_info(weights_data)
        return

    # Validate required arguments
    if not args.predictions:
        parser.error("--predictions is required (unless --info is specified)")
    if not args.output:
        parser.error("--output is required (unless --info is specified)")

    # Load predictions
    logger.info(f"\nLoading predictions from {args.predictions}")
    predictions_df = pd.read_csv(args.predictions)
    logger.info(f"✓ Loaded {len(predictions_df)} predictions")

    # Apply weights
    logger.info("\nApplying optimized weights...")
    ensemble_df = apply_weights(predictions_df, weights_data["weights"])

    # Save results
    logger.info(f"\nSaving ensemble predictions to {args.output}")
    ensemble_df.to_csv(args.output, index=False)
    logger.info(f"✓ Saved {len(ensemble_df)} ensemble predictions")

    # Display summary
    print("\n" + "=" * 80)
    print("ENSEMBLE SUMMARY")
    print("=" * 80)
    print(f"\nMethod: {weights_data['method'].upper()}")
    print(f"N Predictions: {len(ensemble_df)}")
    print("\nEnsemble Predictions:")
    print(f"  Mean: {ensemble_df['ensemble_pred'].mean():.2f} yards")
    print(f"  Std: {ensemble_df['ensemble_pred'].std():.2f} yards")
    print(f"  Min: {ensemble_df['ensemble_pred'].min():.2f} yards")
    print(f"  Max: {ensemble_df['ensemble_pred'].max():.2f} yards")
    print("\nEnsemble Uncertainty:")
    print(f"  Mean: {ensemble_df['ensemble_std'].mean():.2f} yards")
    print(f"  Median: {ensemble_df['ensemble_std'].median():.2f} yards")
    print("\n✓ COMPLETE")


if __name__ == "__main__":
    main()
