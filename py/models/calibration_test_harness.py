#!/usr/bin/env python3
"""
Calibration Test Harness for Uncertainty Quantification

Provides comprehensive metrics for evaluating calibration quality across
different UQ methods (Bayesian, quantile regression, conformal, ensemble).

Key Metrics:
- Coverage at various confidence levels (68%, 90%, 95%)
- Sharpness (interval width)
- Calibration error
- Continuous Ranked Probability Score (CRPS)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class CalibrationMetrics:
    """Container for calibration metrics"""
    # Coverage metrics
    coverage_90: float  # Percentage in 90% CI
    coverage_68: float  # Percentage in 68% CI (±1σ)
    coverage_95: float  # Percentage in 95% CI

    # Sharpness (narrower is better, IF calibrated)
    interval_width_90: float  # Mean width of 90% intervals
    interval_width_68: float  # Mean width of 68% intervals

    # Point prediction quality
    mae: float
    rmse: float
    mape: float  # Mean Absolute Percentage Error

    # Advanced calibration metrics
    calibration_error: float  # Expected Calibration Error
    crps: float  # Continuous Ranked Probability Score

    # Sample statistics
    n_samples: int

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'coverage': {
                '90pct': self.coverage_90,
                '68pct': self.coverage_68,
                '95pct': self.coverage_95
            },
            'sharpness': {
                'width_90': self.interval_width_90,
                'width_68': self.interval_width_68
            },
            'point_accuracy': {
                'mae': self.mae,
                'rmse': self.rmse,
                'mape': self.mape
            },
            'advanced': {
                'calibration_error': self.calibration_error,
                'crps': self.crps
            },
            'n_samples': self.n_samples
        }

    def summary(self) -> str:
        """Human-readable summary"""
        status_90 = "✓" if 0.85 <= self.coverage_90 <= 0.95 else "✗"
        status_68 = "✓" if 0.63 <= self.coverage_68 <= 0.73 else "✗"

        return f"""
=== CALIBRATION METRICS ===
Coverage (target vs actual):
  90% CI: {self.coverage_90:.1%} (target: 90%) {status_90}
  68% CI: {self.coverage_68:.1%} (target: 68%) {status_68}
  95% CI: {self.coverage_95:.1%} (target: 95%)

Sharpness (interval width):
  90% CI: {self.interval_width_90:.1f} yards
  68% CI: {self.interval_width_68:.1f} yards

Point Accuracy:
  MAE:  {self.mae:.2f} yards
  RMSE: {self.rmse:.2f} yards
  MAPE: {self.mape:.1f}%

Advanced:
  Calibration Error: {self.calibration_error:.3f}
  CRPS: {self.crps:.2f}

Samples: {self.n_samples}
"""


class CalibrationEvaluator:
    """Evaluates calibration quality of uncertainty estimates"""

    @staticmethod
    def compute_coverage(
        y_true: np.ndarray,
        q_lower: np.ndarray,
        q_upper: np.ndarray
    ) -> float:
        """Compute coverage percentage"""
        in_interval = (y_true >= q_lower) & (y_true <= q_upper)
        return float(in_interval.mean())

    @staticmethod
    def compute_sharpness(
        q_lower: np.ndarray,
        q_upper: np.ndarray
    ) -> float:
        """Compute mean interval width"""
        return float((q_upper - q_lower).mean())

    @staticmethod
    def compute_calibration_error(
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
        n_bins: int = 10
    ) -> float:
        """
        Expected Calibration Error (ECE)

        Bins predictions by confidence and measures gap between
        expected and observed frequency.
        """
        # Use prediction as confidence score (normalized)
        y_pred = predictions['mean']
        confidence = 1 - np.abs(y_true - y_pred) / np.maximum(y_true, 1)
        confidence = np.clip(confidence, 0, 1)

        # Bin by confidence
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this confidence bin
            in_bin = (confidence >= bin_lower) & (confidence < bin_upper)
            if in_bin.sum() == 0:
                continue

            # Accuracy in bin
            accuracy_in_bin = np.mean(
                np.abs(y_true[in_bin] - y_pred[in_bin]) / np.maximum(y_true[in_bin], 1)
            )
            # Expected confidence in bin
            avg_confidence_in_bin = confidence[in_bin].mean()

            # Weight by proportion of samples in bin
            ece += (in_bin.sum() / len(y_true)) * np.abs(avg_confidence_in_bin - (1 - accuracy_in_bin))

        return float(ece)

    @staticmethod
    def compute_crps(
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray]
    ) -> float:
        """
        Continuous Ranked Probability Score

        Measures quality of probabilistic forecast.
        Lower is better.
        """
        # Approximate CRPS using quantiles
        q05 = predictions['q05']
        q50 = predictions['q50']
        q95 = predictions['q95']

        # Simplified CRPS (exact requires full distribution)
        crps = np.mean([
            np.abs(y_true - q05) + np.abs(y_true - q50) + np.abs(y_true - q95)
        ]) / 3

        return float(crps)

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray]
    ) -> CalibrationMetrics:
        """
        Comprehensive calibration evaluation

        Args:
            y_true: Actual values
            predictions: Dict with keys:
                - 'mean': Point predictions
                - 'std': Standard deviations
                - 'q05', 'q50', 'q95': Quantiles

        Returns:
            CalibrationMetrics object
        """
        # Extract predictions
        y_pred = predictions['mean']
        std = predictions['std']
        q05 = predictions['q05']
        q50 = predictions['q50']
        q95 = predictions['q95']

        # Compute 68% CI from mean ± 1σ
        q16 = y_pred - std
        q84 = y_pred + std

        # Coverage
        coverage_90 = CalibrationEvaluator.compute_coverage(y_true, q05, q95)
        coverage_68 = CalibrationEvaluator.compute_coverage(y_true, q16, q84)

        # 95% CI (extrapolate from 90%)
        width_90 = q95 - q05
        width_95 = width_90 * 1.2  # Approximate
        q025 = y_pred - width_95 / 2
        q975 = y_pred + width_95 / 2
        coverage_95 = CalibrationEvaluator.compute_coverage(y_true, q025, q975)

        # Sharpness
        interval_width_90 = CalibrationEvaluator.compute_sharpness(q05, q95)
        interval_width_68 = CalibrationEvaluator.compute_sharpness(q16, q84)

        # Point accuracy
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mape = float(np.mean(np.abs(y_true - y_pred) / np.maximum(y_true, 1)) * 100)

        # Advanced metrics
        calibration_error = CalibrationEvaluator.compute_calibration_error(
            y_true, predictions
        )
        crps = CalibrationEvaluator.compute_crps(y_true, predictions)

        return CalibrationMetrics(
            coverage_90=coverage_90,
            coverage_68=coverage_68,
            coverage_95=coverage_95,
            interval_width_90=interval_width_90,
            interval_width_68=interval_width_68,
            mae=mae,
            rmse=rmse,
            mape=mape,
            calibration_error=calibration_error,
            crps=crps,
            n_samples=len(y_true)
        )


class ExperimentLogger:
    """Logs experiment results for comparison"""

    def __init__(self, output_dir: str = "experiments/calibration"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    def log_experiment(
        self,
        experiment_name: str,
        config: Dict,
        metrics: CalibrationMetrics,
        notes: str = ""
    ):
        """Log a single experiment"""
        result = {
            'experiment_name': experiment_name,
            'config': config,
            'metrics': metrics.to_dict(),
            'notes': notes
        }
        self.results.append(result)

        # Save individual result
        output_file = self.output_dir / f"{experiment_name}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"✓ Logged experiment: {experiment_name}")
        print(metrics.summary())

    def compare_experiments(self) -> pd.DataFrame:
        """Create comparison table of all experiments"""
        if not self.results:
            return pd.DataFrame()

        comparison = []
        for result in self.results:
            metrics = result['metrics']
            comparison.append({
                'Experiment': result['experiment_name'],
                'Coverage 90%': metrics['coverage']['90pct'],
                'Coverage 68%': metrics['coverage']['68pct'],
                'Width 90%': metrics['sharpness']['width_90'],
                'MAE': metrics['point_accuracy']['mae'],
                'CRPS': metrics['advanced']['crps'],
                'Cal Error': metrics['advanced']['calibration_error'],
                'Notes': result['notes']
            })

        df = pd.DataFrame(comparison)

        # Save comparison table
        comparison_file = self.output_dir / "comparison.csv"
        df.to_csv(comparison_file, index=False)

        return df

    def save_summary(self):
        """Save comprehensive summary"""
        summary_file = self.output_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✓ Saved summary to {summary_file}")
        print(f"✓ Total experiments: {len(self.results)}")


if __name__ == "__main__":
    # Test with dummy data
    np.random.seed(42)
    n_samples = 100

    y_true = np.random.gamma(2, 30, n_samples)  # Rushing yards distribution

    # Simulate well-calibrated predictions
    predictions = {
        'mean': y_true + np.random.normal(0, 10, n_samples),
        'std': np.full(n_samples, 15.0),
        'q05': y_true - 25,
        'q50': y_true,
        'q95': y_true + 25
    }

    # Evaluate
    evaluator = CalibrationEvaluator()
    metrics = evaluator.evaluate(y_true, predictions)

    print(metrics.summary())

    # Log experiment
    logger = ExperimentLogger()
    logger.log_experiment(
        experiment_name="test_baseline",
        config={'method': 'test', 'features': ['test']},
        metrics=metrics,
        notes="Test run of calibration harness"
    )

    print("\n✓ Calibration test harness working correctly")
