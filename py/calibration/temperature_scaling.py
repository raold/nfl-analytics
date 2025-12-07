#!/usr/bin/env python3
"""
Temperature Scaling for Regression Uncertainty Calibration

This module implements temperature scaling to calibrate prediction intervals
for regression tasks. Temperature scaling adjusts the variance of predictions
to achieve target coverage (typically 90% for 90% prediction intervals).

References:
    - Kuleshov et al. (2018): "Accurate Uncertainties for Deep Learning Using
      Calibrated Regression"
    - Guo et al. (2017): "On Calibration of Modern Neural Networks"

Author: Claude Code
Date: November 13, 2025
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

logger = logging.getLogger(__name__)


class TemperatureScaler:
    """
    Temperature scaling for regression prediction intervals.

    Temperature scaling adjusts prediction variance to achieve target coverage:
        σ_calibrated = σ_raw × √T

    Where T is the temperature parameter optimized to minimize calibration error.

    Attributes:
        temperature (float): Optimal temperature parameter (T)
        target_coverage (float): Target coverage rate (default: 0.90)
        method (str): Optimization method ('brentq', 'minimize')
    """

    def __init__(self, target_coverage: float = 0.90, method: str = "brentq"):
        """
        Initialize temperature scaler.

        Args:
            target_coverage: Target coverage rate (e.g., 0.90 for 90% PI)
            method: Optimization method ('brentq' or 'minimize')
        """
        self.target_coverage = target_coverage
        self.method = method
        self.temperature: float | None = None
        self.calibration_data: pd.DataFrame | None = None

    def fit(self, predictions: pd.DataFrame) -> float:
        """
        Estimate optimal temperature on calibration data.

        Args:
            predictions: DataFrame with columns:
                - 'pred_mean': Mean prediction
                - 'pred_std': Standard deviation
                - 'actual': Actual value

        Returns:
            Optimal temperature T
        """
        logger.info("=" * 80)
        logger.info("TEMPERATURE SCALING CALIBRATION")
        logger.info("=" * 80)

        # Validate inputs
        required_cols = ["pred_mean", "pred_std", "actual"]
        missing = [col for col in required_cols if col not in predictions.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        n_samples = len(predictions)
        logger.info(f"Calibration samples: {n_samples}")
        logger.info(f"Target coverage: {self.target_coverage:.1%}")

        # Store calibration data
        self.calibration_data = predictions.copy()

        # Define coverage function
        def coverage_at_temperature(T: float) -> float:
            """Calculate coverage with temperature T."""
            # Apply temperature scaling: σ_calibrated = σ_raw × √T
            std_scaled = predictions["pred_std"] * np.sqrt(T)

            # Calculate z-scores
            z_scores = (predictions["actual"] - predictions["pred_mean"]) / std_scaled

            # Calculate coverage (proportion within target quantile)
            alpha = 1 - self.target_coverage
            lower_quantile = stats.norm.ppf(alpha / 2)
            upper_quantile = stats.norm.ppf(1 - alpha / 2)

            within_interval = (z_scores >= lower_quantile) & (z_scores <= upper_quantile)
            coverage = within_interval.mean()

            return coverage

        # Define objective function (coverage error)
        def objective(T: float) -> float:
            """Minimize squared coverage error."""
            coverage = coverage_at_temperature(T)
            error = (coverage - self.target_coverage) ** 2
            return error

        # Optimize temperature
        logger.info("\nOptimizing temperature parameter...")

        if self.method == "brentq":
            # Use Brent's method to find root of (coverage - target) = 0
            def coverage_error(T: float) -> float:
                return coverage_at_temperature(T) - self.target_coverage

            try:
                # Search in range [0.5, 2.0] (typical range)
                self.temperature = optimize.brentq(
                    coverage_error, a=0.5, b=2.0, xtol=0.001, maxiter=100
                )
                logger.info("✓ Brent optimization converged")
            except ValueError:
                # If no root in range, fall back to minimize
                logger.warning("Brent method failed, falling back to minimize")
                result = optimize.minimize_scalar(objective, bounds=(0.5, 2.0), method="bounded")
                self.temperature = result.x
        else:
            # Use bounded scalar minimization
            result = optimize.minimize_scalar(objective, bounds=(0.5, 2.0), method="bounded")
            self.temperature = result.x
            logger.info(f"✓ Optimization converged in {result.nit} iterations")

        # Validate result
        final_coverage = coverage_at_temperature(self.temperature)
        coverage_error = abs(final_coverage - self.target_coverage)

        logger.info(f"\nOptimal Temperature: T = {self.temperature:.4f}")
        logger.info(f"Final Coverage: {final_coverage:.1%}")
        logger.info(f"Coverage Error: {coverage_error:.1%}")

        if coverage_error > 0.02:  # 2% tolerance
            logger.warning(f"⚠️  Large coverage error: {coverage_error:.1%}")
        else:
            logger.info("✅ Temperature scaling calibrated successfully")

        return self.temperature

    def transform(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Apply temperature scaling to new predictions.

        Args:
            predictions: DataFrame with 'pred_mean' and 'pred_std'

        Returns:
            DataFrame with temperature-scaled standard deviations
        """
        if self.temperature is None:
            raise ValueError("Temperature not fitted. Call fit() first.")

        # Apply scaling: σ_calibrated = σ_raw × √T
        predictions = predictions.copy()
        predictions["pred_std_raw"] = predictions["pred_std"]
        predictions["pred_std"] = predictions["pred_std_raw"] * np.sqrt(self.temperature)
        predictions["temperature"] = self.temperature

        logger.info(f"Applied temperature scaling (T = {self.temperature:.4f})")

        return predictions

    def fit_transform(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Fit temperature scaler and transform predictions.

        Args:
            predictions: DataFrame with 'pred_mean', 'pred_std', 'actual'

        Returns:
            DataFrame with temperature-scaled predictions
        """
        self.fit(predictions)
        return self.transform(predictions)

    def get_calibration_report(self) -> dict:
        """
        Generate calibration report with before/after metrics.

        Returns:
            Dictionary with calibration metrics
        """
        if self.calibration_data is None or self.temperature is None:
            raise ValueError("No calibration data available. Call fit() first.")

        df = self.calibration_data

        # Calculate metrics before temperature scaling
        alpha = 1 - self.target_coverage
        z_raw = (df["actual"] - df["pred_mean"]) / df["pred_std"]
        lower_q = stats.norm.ppf(alpha / 2)
        upper_q = stats.norm.ppf(1 - alpha / 2)

        coverage_raw = ((z_raw >= lower_q) & (z_raw <= upper_q)).mean()
        width_raw = df["pred_std"].mean() * 2 * stats.norm.ppf(1 - alpha / 2)

        # Calculate metrics after temperature scaling
        std_scaled = df["pred_std"] * np.sqrt(self.temperature)
        z_scaled = (df["actual"] - df["pred_mean"]) / std_scaled
        coverage_scaled = ((z_scaled >= lower_q) & (z_scaled <= upper_q)).mean()
        width_scaled = std_scaled.mean() * 2 * stats.norm.ppf(1 - alpha / 2)

        # Calculate MAE (unchanged by temperature scaling)
        mae = np.abs(df["actual"] - df["pred_mean"]).mean()

        report = {
            "temperature": self.temperature,
            "target_coverage": self.target_coverage,
            "coverage_before": coverage_raw,
            "coverage_after": coverage_scaled,
            "width_before": width_raw,
            "width_after": width_scaled,
            "mae": mae,
            "n_samples": len(df),
        }

        return report

    def save(self, path: Path):
        """Save temperature scaler to disk."""
        import pickle

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "temperature": self.temperature,
                    "target_coverage": self.target_coverage,
                    "method": self.method,
                },
                f,
            )
        logger.info(f"Saved temperature scaler to {path}")

    @classmethod
    def load(cls, path: Path) -> "TemperatureScaler":
        """Load temperature scaler from disk."""
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)

        scaler = cls(target_coverage=data["target_coverage"], method=data["method"])
        scaler.temperature = data["temperature"]
        logger.info(f"Loaded temperature scaler from {path} (T = {scaler.temperature:.4f})")
        return scaler


def calculate_calibration_metrics(predictions: pd.DataFrame, target_coverage: float = 0.90) -> dict:
    """
    Calculate calibration metrics for regression predictions.

    Args:
        predictions: DataFrame with 'pred_mean', 'pred_std', 'actual'
        target_coverage: Target coverage rate

    Returns:
        Dictionary with calibration metrics
    """
    alpha = 1 - target_coverage

    # Calculate z-scores
    z_scores = (predictions["actual"] - predictions["pred_mean"]) / predictions["pred_std"]

    # Calculate coverage
    lower_q = stats.norm.ppf(alpha / 2)
    upper_q = stats.norm.ppf(1 - alpha / 2)
    within_interval = (z_scores >= lower_q) & (z_scores <= upper_q)
    coverage = within_interval.mean()

    # Calculate interval width
    interval_width = predictions["pred_std"].mean() * 2 * stats.norm.ppf(1 - alpha / 2)

    # Calculate errors
    mae = np.abs(predictions["actual"] - predictions["pred_mean"]).mean()
    rmse = np.sqrt(((predictions["actual"] - predictions["pred_mean"]) ** 2).mean())

    # Binomial test for coverage
    n = len(predictions)
    n_within = within_interval.sum()
    p_value = 2 * min(
        stats.binom.cdf(n_within, n, target_coverage),
        1 - stats.binom.cdf(n_within - 1, n, target_coverage),
    )

    return {
        "coverage": coverage,
        "target_coverage": target_coverage,
        "interval_width": interval_width,
        "mae": mae,
        "rmse": rmse,
        "n_samples": n,
        "n_within": n_within,
        "p_value": p_value,
        "calibrated": abs(coverage - target_coverage) <= 0.05,  # 5% tolerance
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Load 2025 holdout predictions
    results_path = Path("analysis/results/ensemble_4way_calibration_2025_holdout.csv")
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        exit(1)

    df = pd.read_csv(results_path)

    # Rename columns to match expected format
    df = df.rename(columns={"actual_yards": "actual"})

    # Fit temperature scaler
    scaler = TemperatureScaler(target_coverage=0.90)
    scaler.fit(df)

    # Get calibration report
    report = scaler.get_calibration_report()

    logger.info("\n" + "=" * 80)
    logger.info("CALIBRATION REPORT")
    logger.info("=" * 80)
    logger.info(f"Temperature: T = {report['temperature']:.4f}")
    logger.info(f"Target Coverage: {report['target_coverage']:.1%}")
    logger.info("\nBefore Temperature Scaling:")
    logger.info(f"  Coverage: {report['coverage_before']:.1%}")
    logger.info(f"  Mean Interval Width: {report['width_before']:.1f} yards")
    logger.info("\nAfter Temperature Scaling:")
    logger.info(f"  Coverage: {report['coverage_after']:.1%}")
    logger.info(f"  Mean Interval Width: {report['width_after']:.1f} yards")
    logger.info(f"\nMAE: {report['mae']:.2f} yards (unchanged)")

    # Save temperature scaler
    save_path = Path("models/calibration/temperature_scaler_4way.pkl")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    scaler.save(save_path)
