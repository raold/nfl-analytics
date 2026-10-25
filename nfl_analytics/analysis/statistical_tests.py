#!/usr/bin/env python3
"""Statistical significance tests for model comparison.

This module implements the Diebold-Mariano test (Diebold & Mariano, 1995)
for comparing predictive accuracy of forecasting models. The DM test accounts
for forecast error correlation and provides asymptotically valid inference
under weak regularity conditions.

Key features:
- Diebold-Mariano test for comparing two models
- Support for multiple loss functions (squared error, absolute error)
- Bootstrap confidence intervals for effect sizes
- Multiple comparison corrections (Bonferroni, Holm)
- Comprehensive output suitable for dissertation tables

References:
    Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy.
    Journal of Business & Economic Statistics, 13(3), 253-263.

Usage:
    python nfl_analytics/analysis/statistical_tests.py \
        --predictions analysis/results/ensemble_predictions_2025_holdout.csv \
        --output analysis/results/statistical_tests_results.csv

Author: NFL Analytics Research Team
Date: November 9, 2025
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


class DieboldMarianoTest:
    """Diebold-Mariano test for comparing predictive accuracy of two models.

    The test statistic is:
        DM = mean(d_t) / sqrt(var(d_t) / T)

    where d_t is the loss differential at time t and T is the number of
    observations. Under the null hypothesis (equal predictive accuracy),
    DM ~ N(0, 1) asymptotically.

    Attributes:
        loss_fn: Loss function to use ('squared', 'absolute')
        h: Forecast horizon (default: 1)
        alternative: Test direction ('two-sided', 'less', 'greater')
    """

    def __init__(
        self,
        loss_fn: Literal["squared", "absolute"] = "absolute",
        h: int = 1,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    ):
        """Initialize Diebold-Mariano test parameters.

        Args:
            loss_fn: Loss function ('squared' for MSE, 'absolute' for MAE)
            h: Forecast horizon for Newey-West HAC variance estimation
            alternative: Alternative hypothesis direction
        """
        self.loss_fn = loss_fn
        self.h = h
        self.alternative = alternative

    def compute_loss(self, errors: np.ndarray) -> np.ndarray:
        """Compute loss from forecast errors.

        Args:
            errors: Forecast errors (actual - predicted)

        Returns:
            Loss values
        """
        if self.loss_fn == "squared":
            return errors**2
        elif self.loss_fn == "absolute":
            return np.abs(errors)
        else:
            raise ValueError(f"Unknown loss function: {self.loss_fn}")

    def newey_west_variance(self, d: np.ndarray) -> float:
        """Compute Newey-West HAC variance estimator.

        This accounts for autocorrelation in the loss differential series
        up to lag h-1.

        Args:
            d: Loss differential series

        Returns:
            HAC variance estimate
        """
        T = len(d)
        d_centered = d - np.mean(d)

        # Variance (lag 0)
        var = np.mean(d_centered**2)

        # Add autocovariances with Bartlett kernel weights
        for lag in range(1, self.h):
            if lag >= T:
                break

            # Bartlett kernel weight
            weight = 1 - lag / self.h

            # Autocovariance at lag
            acov = np.mean(d_centered[:-lag] * d_centered[lag:])

            # Add both positive and negative lags
            var += 2 * weight * acov

        return var

    def test(
        self,
        actual: np.ndarray,
        pred1: np.ndarray,
        pred2: np.ndarray,
    ) -> dict:
        """Perform Diebold-Mariano test comparing two models.

        Args:
            actual: Actual values
            pred1: Predictions from model 1
            pred2: Predictions from model 2

        Returns:
            Dictionary containing:
                - statistic: DM test statistic
                - p_value_one_sided: One-sided p-value (model 1 < model 2)
                - p_value_two_sided: Two-sided p-value
                - mean_loss1: Mean loss for model 1
                - mean_loss2: Mean loss for model 2
                - mean_diff: Mean loss difference (loss1 - loss2)
                - std_diff: Standard deviation of loss difference
                - n_obs: Number of observations
        """
        # Validate inputs
        actual = np.asarray(actual)
        pred1 = np.asarray(pred1)
        pred2 = np.asarray(pred2)

        if not (len(actual) == len(pred1) == len(pred2)):
            raise ValueError("All arrays must have same length")

        # Remove any missing values
        mask = ~(np.isnan(actual) | np.isnan(pred1) | np.isnan(pred2))
        actual = actual[mask]
        pred1 = pred1[mask]
        pred2 = pred2[mask]

        T = len(actual)

        if T < 10:
            warnings.warn(f"Small sample size (n={T}). Results may be unreliable.")

        # Compute forecast errors
        e1 = actual - pred1
        e2 = actual - pred2

        # Compute losses
        loss1 = self.compute_loss(e1)
        loss2 = self.compute_loss(e2)

        # Loss differential
        d = loss1 - loss2

        # Mean loss difference
        mean_d = np.mean(d)

        # Newey-West variance
        var_d = self.newey_west_variance(d)

        # DM statistic
        if var_d <= 0:
            warnings.warn("Non-positive variance estimate. Using standard variance.")
            var_d = np.var(d, ddof=1)

        dm_stat = mean_d / np.sqrt(var_d / T)

        # P-values
        # One-sided: H1: model 1 is more accurate (loss1 < loss2, so d < 0)
        p_value_one_sided = stats.norm.cdf(dm_stat)

        # Two-sided: H1: models differ in accuracy
        p_value_two_sided = 2 * min(p_value_one_sided, 1 - p_value_one_sided)

        return {
            "statistic": dm_stat,
            "p_value_one_sided": p_value_one_sided,
            "p_value_two_sided": p_value_two_sided,
            "mean_loss1": np.mean(loss1),
            "mean_loss2": np.mean(loss2),
            "mean_diff": mean_d,
            "std_diff": np.std(d, ddof=1),
            "n_obs": T,
        }


def bootstrap_confidence_interval(
    actual: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    loss_fn: str = "absolute",
    n_boot: int = 5000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for mean loss difference.

    Args:
        actual: Actual values
        pred1: Predictions from model 1
        pred2: Predictions from model 2
        loss_fn: Loss function ('squared' or 'absolute')
        n_boot: Number of bootstrap samples
        confidence: Confidence level (default: 0.95)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    rng = np.random.RandomState(random_state)

    T = len(actual)
    boot_diffs = np.zeros(n_boot)

    for i in range(n_boot):
        # Bootstrap sample with replacement
        idx = rng.choice(T, size=T, replace=True)

        actual_boot = actual[idx]
        pred1_boot = pred1[idx]
        pred2_boot = pred2[idx]

        # Compute errors
        e1 = actual_boot - pred1_boot
        e2 = actual_boot - pred2_boot

        # Compute losses
        if loss_fn == "squared":
            loss1 = e1**2
            loss2 = e2**2
        else:  # absolute
            loss1 = np.abs(e1)
            loss2 = np.abs(e2)

        # Mean difference
        boot_diffs[i] = np.mean(loss1) - np.mean(loss2)

    # Percentile confidence interval
    alpha = 1 - confidence
    lower = np.percentile(boot_diffs, 100 * alpha / 2)
    upper = np.percentile(boot_diffs, 100 * (1 - alpha / 2))

    return lower, upper


def compare_all_models(
    predictions_df: pd.DataFrame,
    ensemble_col: str = "pred_mean",
    actual_col: str = "actual_yards",
    model_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compare ensemble against all individual models using DM test.

    Args:
        predictions_df: DataFrame with predictions and actuals
        ensemble_col: Column name for ensemble predictions
        actual_col: Column name for actual values
        model_cols: List of column names for individual model predictions.
                    If None, will auto-detect columns matching pattern.

    Returns:
        DataFrame with test results for each comparison
    """
    results = []

    # Auto-detect model columns if not provided
    if model_cols is None:
        # Look for columns that might be predictions
        candidate_cols = [
            col
            for col in predictions_df.columns
            if col
            not in [
                ensemble_col,
                actual_col,
                "game_id",
                "player_id",
                "player_name",
                "season",
                "week",
                "model_name",
            ]
        ]
        model_cols = candidate_cols

    # Initialize DM test
    dm = DieboldMarianoTest(loss_fn="absolute", h=1)

    # Get ensemble predictions
    ensemble_pred = predictions_df[ensemble_col].values
    actual = predictions_df[actual_col].values

    # Test ensemble vs each model
    for model_col in model_cols:
        if model_col not in predictions_df.columns:
            warnings.warn(f"Column '{model_col}' not found. Skipping.")
            continue

        model_pred = predictions_df[model_col].values

        # Perform DM test
        result = dm.test(actual, ensemble_pred, model_pred)

        # Bootstrap CI
        ci_lower, ci_upper = bootstrap_confidence_interval(
            actual, ensemble_pred, model_pred, loss_fn="absolute"
        )

        # Compute MAE for each model
        mae_ensemble = np.mean(np.abs(actual - ensemble_pred))
        mae_model = np.mean(np.abs(actual - model_pred))

        results.append(
            {
                "comparison": f"Ensemble vs {model_col}",
                "n_obs": result["n_obs"],
                "mae_ensemble": mae_ensemble,
                "mae_model": mae_model,
                "mean_diff_mae": result["mean_diff"],
                "dm_statistic": result["statistic"],
                "p_value_one_sided": result["p_value_one_sided"],
                "p_value_two_sided": result["p_value_two_sided"],
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper,
                "significant_5pct": result["p_value_one_sided"] < 0.05,
            }
        )

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Apply multiple testing correction (Holm-Bonferroni)
    if len(results_df) > 0:
        results_df = apply_holm_bonferroni(results_df)

    return results_df


def apply_holm_bonferroni(
    results_df: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Apply the Holm-Bonferroni step-down correction.

    Sorted p-values p_(1) <= ... <= p_(m) are compared with alpha / (m - i + 1) in order,
    and testing stops at the first one that fails: every hypothesis after it is retained,
    even if its own p-value is below its threshold.

    Args:
        results_df: DataFrame with a p_value_one_sided column
        alpha: Family-wise error rate

    Returns:
        DataFrame sorted by p-value with holm_threshold, p_holm (adjusted p-value)
        and significant_holm columns
    """
    n_tests = len(results_df)

    results_df = results_df.sort_values("p_value_one_sided").reset_index(drop=True)
    p = results_df["p_value_one_sided"].to_numpy(dtype=float)

    results_df["holm_threshold"] = alpha / (n_tests - np.arange(n_tests))
    # Step-down: reject while p_(i) < threshold_(i); the first failure ends the procedure
    passes = p < results_df["holm_threshold"].to_numpy()
    results_df["significant_holm"] = np.cumprod(passes).astype(bool)
    results_df["p_holm"] = np.minimum(
        np.maximum.accumulate((n_tests - np.arange(n_tests)) * p), 1.0
    )

    return results_df


def compare_to_baseline(
    predictions_df: pd.DataFrame,
    ensemble_col: str = "pred_mean",
    actual_col: str = "actual_yards",
    baseline_type: str = "season_avg",
) -> dict:
    """Compare ensemble to naive baseline predictions.

    Args:
        predictions_df: DataFrame with predictions and actuals
        ensemble_col: Column name for ensemble predictions
        actual_col: Column name for actual values
        baseline_type: Type of baseline ('season_avg', 'last_game', 'league_avg')

    Returns:
        Dictionary with test results
    """
    actual = predictions_df[actual_col].values
    ensemble_pred = predictions_df[ensemble_col].values

    # Create baseline predictions
    if baseline_type == "season_avg":
        # Use within-season average as baseline (by player)
        baseline_pred = (
            predictions_df.groupby(["player_id", "season"])[actual_col].transform("mean").values
        )
    elif baseline_type == "league_avg":
        # Use league-wide average
        baseline_pred = np.full_like(actual, np.mean(actual), dtype=float)
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")

    # Perform DM test
    dm = DieboldMarianoTest(loss_fn="absolute", h=1)
    result = dm.test(actual, ensemble_pred, baseline_pred)

    # Bootstrap CI
    ci_lower, ci_upper = bootstrap_confidence_interval(
        actual, ensemble_pred, baseline_pred, loss_fn="absolute"
    )

    # Compute MAE
    mae_ensemble = np.mean(np.abs(actual - ensemble_pred))
    mae_baseline = np.mean(np.abs(actual - baseline_pred))

    return {
        "comparison": f"Ensemble vs {baseline_type}",
        "n_obs": result["n_obs"],
        "mae_ensemble": mae_ensemble,
        "mae_baseline": mae_baseline,
        "mae_improvement": mae_baseline - mae_ensemble,
        "pct_improvement": 100 * (mae_baseline - mae_ensemble) / mae_baseline,
        "dm_statistic": result["statistic"],
        "p_value_one_sided": result["p_value_one_sided"],
        "p_value_two_sided": result["p_value_two_sided"],
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "significant_5pct": result["p_value_one_sided"] < 0.05,
    }


def main():
    """Run statistical tests on ensemble predictions."""
    parser = argparse.ArgumentParser(
        description="Run Diebold-Mariano tests comparing ensemble vs individual models"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="analysis/results/ensemble_predictions_2025_holdout.csv",
        help="Path to predictions CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis/results/statistical_tests_results.csv",
        help="Path to output CSV file",
    )
    parser.add_argument(
        "--ensemble-col",
        type=str,
        default="pred_mean",
        help="Column name for ensemble predictions",
    )
    parser.add_argument(
        "--actual-col",
        type=str,
        default="actual_yards",
        help="Column name for actual values",
    )

    args = parser.parse_args()

    # Load predictions
    print(f"Loading predictions from {args.predictions}...")
    predictions_df = pd.read_csv(args.predictions)
    print(f"Loaded {len(predictions_df)} predictions")

    # Check if this is state-space only data (from existing file)
    # or if we need to create synthetic ensemble data
    if "model_name" in predictions_df.columns:
        # This appears to be state-space only data
        print("\nWarning: Data appears to be single-model predictions.")
        print("Creating synthetic ensemble for demonstration purposes.")
        print("Replace with actual 5-way ensemble predictions for production use.\n")

        # For now, compare different scenarios if available
        # Or just document the methodology
        ensemble_pred = predictions_df["pred_mean"].values
        actual = predictions_df["actual_yards"].values

        # Compute basic statistics
        mae = np.mean(np.abs(actual - ensemble_pred))
        rmse = np.sqrt(np.mean((actual - ensemble_pred) ** 2))

        results = [
            {
                "comparison": "State-Space v2 (2025 Holdout)",
                "n_obs": len(predictions_df),
                "mae": mae,
                "rmse": rmse,
                "note": "Full 5-way ensemble comparison pending model retraining",
            }
        ]

        results_df = pd.DataFrame(results)

    else:
        # Compare ensemble to individual models
        print("\nComparing ensemble to individual models...")
        results_df = compare_all_models(
            predictions_df,
            ensemble_col=args.ensemble_col,
            actual_col=args.actual_col,
        )

        # Compare to baseline
        print("\nComparing ensemble to naive baseline...")
        baseline_result = compare_to_baseline(
            predictions_df,
            ensemble_col=args.ensemble_col,
            actual_col=args.actual_col,
        )

        # Append baseline result
        results_df = pd.concat([results_df, pd.DataFrame([baseline_result])], ignore_index=True)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("STATISTICAL TESTS SUMMARY")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
