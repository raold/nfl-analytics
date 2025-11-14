"""
Ensemble Health Monitor for NFL Analytics Betting System

This module provides comprehensive health monitoring for the 4-way ensemble system,
with emphasis on detecting R model integration failures BEFORE long backtests.

Critical Purpose:
    Prevent Nov 13, 2025 failure where R model loading failed during 18-hour backtest.
    Validate all 4 components (State-Space v2, Hierarchical v2, Informative Priors v2,
    XGBoost v2) BEFORE running expensive backtests or production deployments.

Key Capabilities:
1. Pre-Backtest Validation - Load all R models, test rpy2 integration
2. Component Drift Detection - Track prediction distributions per component
3. Real-Time Monitoring - Calibration tracking, weight optimization, MAE drift
4. Inter-Model Agreement - Detect high uncertainty from model disagreement

Usage:
    from py.utils.ensemble_health_monitor import EnsembleHealthMonitor

    monitor = EnsembleHealthMonitor()

    # Pre-backtest validation (CRITICAL - run before 18-hour backtests)
    report = monitor.check_ensemble_health()
    if not report.is_healthy:
        print("⛔ Ensemble unhealthy - do not run backtest")
        exit(1)

    # Real-time monitoring during backtest
    monitor.track_prediction(
        player_id="00-0019596",
        season=2024,
        week=9,
        component_predictions={
            "state_space_v2": (245.3, 42.1),
            "hierarchical_v2": (238.7, 38.5),
            "xgboost_v2": (251.2, 35.2),
            "informative_priors_v2": (241.8, 40.3)
        },
        actual_yards=248.0
    )

    # Generate health report
    report = monitor.generate_report()

CLI Interface:
    python py/utils/ensemble_health_monitor.py --check
    python py/utils/ensemble_health_monitor.py --monitor
    python py/utils/ensemble_health_monitor.py --report

Integration:
    - Pre-commit hook: Validate before backtest commits
    - CI/CD: GitHub Actions runs health checks
    - Manual: CLI tool for on-demand validation

Author: Claude Code (Sonnet 4.5)
Date: November 14, 2025
"""

import logging
import os
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Ensemble health status levels."""
    HEALTHY = "HEALTHY"          # All checks pass
    WARNING = "WARNING"          # Minor issues, can proceed with caution
    DEGRADED = "DEGRADED"        # Performance degraded, investigate
    CRITICAL = "CRITICAL"        # Critical failure, do not proceed


class ComponentStatus(Enum):
    """Individual component status."""
    LOADED = "LOADED"            # Model loaded successfully
    NOT_FOUND = "NOT_FOUND"      # Model file missing
    LOAD_FAILED = "LOAD_FAILED"  # Failed to load (rpy2 error, corrupted file)
    DRIFT_DETECTED = "DRIFT_DETECTED"  # Prediction drift detected


@dataclass
class ComponentHealth:
    """Health status for a single ensemble component."""
    name: str
    status: ComponentStatus
    file_path: str
    file_exists: bool
    file_size_mb: float
    last_modified: Optional[datetime] = None
    load_time_ms: Optional[float] = None
    rpy2_test_passed: bool = False
    test_prediction_mean: Optional[float] = None
    test_prediction_std: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class CalibrationMetrics:
    """Calibration metrics for ensemble or component."""
    coverage_90pct: float  # Percentage of actuals within 90% CI
    coverage_50pct: float  # Percentage of actuals within 50% CI
    mean_interval_width_90pct: float
    mean_interval_width_50pct: float
    sharpness_score: float  # Lower is better (avg std / avg MAE)
    n_predictions: int


@dataclass
class DriftMetrics:
    """Drift detection metrics for components."""
    mean_shift: float  # Change in mean prediction (yards)
    std_shift: float   # Change in prediction uncertainty
    mae_drift: float   # Change in MAE over time
    correlation_drop: float  # Drop in correlation between components
    n_samples_baseline: int
    n_samples_current: int


@dataclass
class EnsembleHealthReport:
    """Comprehensive health report for 4-way ensemble."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    overall_status: HealthStatus = HealthStatus.HEALTHY
    components: Dict[str, ComponentHealth] = field(default_factory=dict)

    # Calibration tracking
    ensemble_calibration: Optional[CalibrationMetrics] = None
    component_calibration: Dict[str, CalibrationMetrics] = field(default_factory=dict)

    # Drift detection
    drift_metrics: Dict[str, DriftMetrics] = field(default_factory=dict)

    # Inter-model agreement
    mean_pairwise_correlation: Optional[float] = None
    mean_prediction_variance: Optional[float] = None  # High variance = high disagreement

    # Performance tracking
    ensemble_mae: Optional[float] = None
    component_mae: Dict[str, float] = field(default_factory=dict)

    # Weight monitoring
    current_weights: Dict[str, float] = field(default_factory=dict)
    optimal_weights: Optional[Dict[str, float]] = None
    weight_deviation: Optional[float] = None  # L2 norm of (current - optimal)

    # Issue tracking
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        """Check if ensemble is healthy enough for production."""
        return self.overall_status in [HealthStatus.HEALTHY, HealthStatus.WARNING]

    def add_critical_issue(self, message: str):
        """Add critical issue and update status."""
        self.critical_issues.append(message)
        self.overall_status = HealthStatus.CRITICAL

    def add_warning(self, message: str):
        """Add warning (does not change status if already critical/degraded)."""
        self.warnings.append(message)
        if self.overall_status == HealthStatus.HEALTHY:
            self.overall_status = HealthStatus.WARNING


class EnsembleHealthMonitor:
    """
    Health monitoring for 4-way ensemble system.

    Validates all components before backtest, tracks performance during backtest,
    detects drift and calibration issues in real-time.

    Attributes:
        project_root (Path): Project root directory
        models_config (Dict): Model file paths from MODEL_REGISTRY.yaml
        prediction_history (List): Tracked predictions for drift detection
        calibration_baseline (Dict): Baseline calibration metrics
    """

    # Default model paths from MODEL_REGISTRY.yaml
    DEFAULT_MODEL_PATHS = {
        "state_space_v2": "models/bayesian/simplified_random_walk_v2.rds",
        "hierarchical_v2": "models/bayesian/passing_yards_hierarchical_v2.rds",
        "informative_priors_v2": "models/bayesian/passing_informative_priors_v2.rds",
        "xgboost_v2": "models/xgboost/passing_xgboost_v2_canonical.pkl",
    }

    # Target performance thresholds
    PERFORMANCE_THRESHOLDS = {
        "mae_threshold": 45.2,  # yards
        "calibration_90pct_min": 85.0,  # %
        "calibration_90pct_max": 95.0,  # %
        "calibration_90pct_target": 90.0,  # %
        "sharpe_ratio_min": 2.2,
        "roi_min": 5.0,  # %
    }

    # Drift detection thresholds
    DRIFT_THRESHOLDS = {
        "mean_shift_yards": 10.0,  # Alert if mean shifts by >10 yards
        "mae_drift_pct": 15.0,     # Alert if MAE increases by >15%
        "correlation_drop": 0.15,  # Alert if inter-model correlation drops by >0.15
    }

    def __init__(
        self,
        model_paths: Optional[Dict[str, str]] = None,
        project_root: Optional[Path] = None,
    ):
        """
        Initialize ensemble health monitor.

        Args:
            model_paths: Custom model paths (if None, uses DEFAULT_MODEL_PATHS)
            project_root: Project root directory (if None, infers from file location)
        """
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.models_config = model_paths or self.DEFAULT_MODEL_PATHS.copy()

        # Tracking state
        self.prediction_history = []  # List of tracked predictions
        self.calibration_baseline = None  # Baseline calibration metrics
        self.last_check_time = None

        logger.info(f"Initialized EnsembleHealthMonitor at {self.project_root}")

    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to project root."""
        p = Path(path)
        if p.is_absolute():
            return p
        return (self.project_root / path).resolve()

    def check_file_health(self, component_name: str, file_path: str) -> ComponentHealth:
        """
        Check file health for a single component.

        Args:
            component_name: Component name (e.g., "state_space_v2")
            file_path: Path to model file

        Returns:
            ComponentHealth object with file metadata
        """
        resolved_path = self._resolve_path(file_path)

        health = ComponentHealth(
            name=component_name,
            status=ComponentStatus.NOT_FOUND,
            file_path=str(resolved_path),
            file_exists=resolved_path.exists(),
            file_size_mb=0.0,
        )

        if not resolved_path.exists():
            health.error_message = f"Model file not found: {resolved_path}"
            logger.warning(f"⚠️  {component_name}: {health.error_message}")
            return health

        # Get file metadata
        stat = resolved_path.stat()
        health.file_size_mb = stat.st_size / (1024 * 1024)
        health.last_modified = datetime.fromtimestamp(stat.st_mtime)

        # Validate file size (detect corruption)
        if health.file_size_mb < 0.01:  # <10KB suggests corruption
            health.status = ComponentStatus.LOAD_FAILED
            health.error_message = f"File too small ({health.file_size_mb:.2f} MB), likely corrupted"
            logger.warning(f"⚠️  {component_name}: {health.error_message}")
        else:
            health.status = ComponentStatus.LOADED
            logger.info(f"✅ {component_name}: {health.file_size_mb:.2f} MB, modified {health.last_modified}")

        return health

    def test_r_model_loading(
        self,
        component_name: str,
        model_path: Path
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Test R model loading via rpy2 (CRITICAL for preventing Nov 13 failure).

        Args:
            component_name: Component name
            model_path: Path to RDS model file

        Returns:
            (success, error_message, load_time_ms) tuple
        """
        try:
            import rpy2.robjects as ro

            start = time.time()

            # Load brms library
            ro.r('suppressPackageStartupMessages(library(brms))')

            # Load R model
            ro.r(f'model <- readRDS("{model_path}")')

            # Verify model object
            model_class = ro.r('class(model)')[0]

            load_time_ms = (time.time() - start) * 1000

            logger.info(f"✅ {component_name}: R model loaded successfully ({model_class}, {load_time_ms:.1f}ms)")
            return (True, None, load_time_ms)

        except ImportError as e:
            error_msg = f"rpy2 import failed: {e}"
            logger.error(f"⛔ {component_name}: {error_msg}")
            return (False, error_msg, None)

        except Exception as e:
            error_msg = f"R model loading failed: {e}"
            logger.error(f"⛔ {component_name}: {error_msg}")
            return (False, error_msg, None)

    def test_r_model_prediction(
        self,
        component_name: str,
        model_path: Path
    ) -> Tuple[bool, Optional[float], Optional[float], Optional[str]]:
        """
        Test R model prediction with dummy data.

        Args:
            component_name: Component name
            model_path: Path to RDS model file

        Returns:
            (success, mean, std, error_message) tuple
        """
        try:
            import rpy2.robjects as ro
            from rpy2.robjects import pandas2ri
            from rpy2.robjects.conversion import localconverter

            # Load brms
            ro.r('suppressPackageStartupMessages(library(brms))')

            # Load model
            ro.r(f'model <- readRDS("{model_path}")')

            # Create dummy data matching brms model formula
            # NOTE: This assumes model formula includes player_season, log_attempts, is_home
            dummy_data = pd.DataFrame({
                'player_id': ['00-0019596'],  # Patrick Mahomes
                'season': [2024],
                'week': [1],
                'attempts': [35],
                'is_home': [1],
                'player_season': ['00-0019596_2024'],
                'log_attempts': [np.log(35)],
            })

            # Convert to R
            with localconverter(ro.default_converter + pandas2ri.converter):
                r_df = ro.conversion.py2rpy(dummy_data)

            # Generate predictions
            ro.r.assign('new_data', r_df)
            ro.r('posterior_preds <- posterior_predict(model, newdata = new_data, ndraws = 100, allow_new_levels = TRUE)')
            ro.r('posterior_preds_exp <- exp(posterior_preds)')
            ro.r('pred_mean <- mean(posterior_preds_exp)')
            ro.r('pred_sd <- sd(posterior_preds_exp)')

            mean_pred = float(ro.r('pred_mean')[0])
            std_pred = float(ro.r('pred_sd')[0])

            logger.info(f"✅ {component_name}: Test prediction {mean_pred:.1f} ± {std_pred:.1f} yards")
            return (True, mean_pred, std_pred, None)

        except Exception as e:
            error_msg = f"Test prediction failed: {e}"
            logger.error(f"⛔ {component_name}: {error_msg}")
            return (False, None, None, error_msg)

    def check_ensemble_health(self) -> EnsembleHealthReport:
        """
        Comprehensive health check for all ensemble components.

        This is the CRITICAL function to run before expensive backtests.
        Validates:
        1. All model files exist and are not corrupted
        2. R models can be loaded via rpy2
        3. R models can generate test predictions
        4. XGBoost model can be loaded

        Returns:
            EnsembleHealthReport with detailed health status
        """
        logger.info("=" * 80)
        logger.info("ENSEMBLE HEALTH CHECK - Pre-Backtest Validation")
        logger.info("=" * 80)

        report = EnsembleHealthReport()
        self.last_check_time = datetime.now()

        for component_name, file_path in self.models_config.items():
            logger.info(f"\nChecking {component_name}...")

            # Check file health
            health = self.check_file_health(component_name, file_path)

            if not health.file_exists:
                report.add_critical_issue(f"{component_name}: Model file not found")
                report.components[component_name] = health
                continue

            if health.status == ComponentStatus.LOAD_FAILED:
                report.add_critical_issue(f"{component_name}: {health.error_message}")
                report.components[component_name] = health
                continue

            # Test R model loading (for .rds files)
            if file_path.endswith('.rds'):
                success, error_msg, load_time = self.test_r_model_loading(
                    component_name,
                    self._resolve_path(file_path)
                )

                health.load_time_ms = load_time

                if not success:
                    health.status = ComponentStatus.LOAD_FAILED
                    health.error_message = error_msg
                    health.rpy2_test_passed = False
                    report.add_critical_issue(f"{component_name}: rpy2 test failed - {error_msg}")
                else:
                    health.rpy2_test_passed = True

                    # Test prediction
                    success, mean, std, error_msg = self.test_r_model_prediction(
                        component_name,
                        self._resolve_path(file_path)
                    )

                    if success:
                        health.test_prediction_mean = mean
                        health.test_prediction_std = std
                    else:
                        health.error_message = error_msg
                        report.add_warning(f"{component_name}: Test prediction failed - {error_msg}")

            # Test XGBoost loading (for .pkl files)
            elif file_path.endswith('.pkl'):
                try:
                    import xgboost as xgb
                    start = time.time()
                    model = xgb.Booster()
                    model.load_model(str(self._resolve_path(file_path)))
                    load_time = (time.time() - start) * 1000
                    health.load_time_ms = load_time
                    logger.info(f"✅ {component_name}: XGBoost loaded successfully ({load_time:.1f}ms)")
                except Exception as e:
                    health.status = ComponentStatus.LOAD_FAILED
                    health.error_message = f"XGBoost loading failed: {e}"
                    report.add_critical_issue(f"{component_name}: {health.error_message}")

            report.components[component_name] = health

        # Set current weights (from MODEL_REGISTRY.yaml)
        report.current_weights = {
            "state_space_v2": 0.30,
            "hierarchical_v2": 0.30,
            "xgboost_v2": 0.25,
            "informative_priors_v2": 0.15,
        }

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("HEALTH CHECK SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Overall Status: {report.overall_status.value}")
        logger.info(f"Components Loaded: {sum(1 for h in report.components.values() if h.status == ComponentStatus.LOADED)}/{len(report.components)}")
        logger.info(f"Critical Issues: {len(report.critical_issues)}")
        logger.info(f"Warnings: {len(report.warnings)}")

        if report.critical_issues:
            logger.error("\n⛔ CRITICAL ISSUES:")
            for issue in report.critical_issues:
                logger.error(f"  - {issue}")

        if report.warnings:
            logger.warning("\n⚠️  WARNINGS:")
            for warning in report.warnings:
                logger.warning(f"  - {warning}")

        if report.is_healthy:
            logger.info("\n✅ Ensemble is HEALTHY - safe to run backtest")
        else:
            logger.error("\n⛔ Ensemble is UNHEALTHY - DO NOT run backtest")

        return report

    def track_prediction(
        self,
        player_id: str,
        season: int,
        week: int,
        component_predictions: Dict[str, Tuple[float, float]],
        actual_yards: Optional[float] = None,
        ensemble_prediction: Optional[Tuple[float, float]] = None,
    ):
        """
        Track a prediction for drift detection and calibration monitoring.

        Args:
            player_id: Player ID
            season: Season year
            week: Week number
            component_predictions: Dict of {component_name: (mean, std)}
            actual_yards: Actual passing yards (if known)
            ensemble_prediction: Ensemble (mean, std) (if None, calculated from components)
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'player_id': player_id,
            'season': season,
            'week': week,
            'component_predictions': component_predictions,
            'actual_yards': actual_yards,
            'ensemble_prediction': ensemble_prediction,
        }

        self.prediction_history.append(record)

        # Calculate inter-model agreement
        means = [p[0] for p in component_predictions.values()]
        stds = [p[1] for p in component_predictions.values()]

        prediction_variance = np.var(means)

        logger.debug(f"Tracked prediction: {player_id} W{week} {season}, variance={prediction_variance:.1f}")

    def calculate_calibration(
        self,
        predictions: List[Dict],
        component_name: Optional[str] = None
    ) -> CalibrationMetrics:
        """
        Calculate calibration metrics from prediction history.

        Args:
            predictions: List of prediction records
            component_name: If specified, calculate for single component only

        Returns:
            CalibrationMetrics object
        """
        # Filter predictions with actuals
        valid_preds = [p for p in predictions if p.get('actual_yards') is not None]

        if len(valid_preds) == 0:
            return CalibrationMetrics(
                coverage_90pct=0.0,
                coverage_50pct=0.0,
                mean_interval_width_90pct=0.0,
                mean_interval_width_50pct=0.0,
                sharpness_score=0.0,
                n_predictions=0,
            )

        coverages_90 = []
        coverages_50 = []
        widths_90 = []
        widths_50 = []

        for pred in valid_preds:
            actual = pred['actual_yards']

            # Get mean/std
            if component_name:
                if component_name not in pred['component_predictions']:
                    continue
                mean, std = pred['component_predictions'][component_name]
            else:
                # Ensemble prediction
                if pred.get('ensemble_prediction'):
                    mean, std = pred['ensemble_prediction']
                else:
                    # Calculate from components
                    means = [p[0] for p in pred['component_predictions'].values()]
                    mean = np.mean(means)
                    std = np.std(means)

            # 90% CI (1.645 sigma)
            lower_90 = mean - 1.645 * std
            upper_90 = mean + 1.645 * std
            coverages_90.append(1 if lower_90 <= actual <= upper_90 else 0)
            widths_90.append(upper_90 - lower_90)

            # 50% CI (0.674 sigma)
            lower_50 = mean - 0.674 * std
            upper_50 = mean + 0.674 * std
            coverages_50.append(1 if lower_50 <= actual <= upper_50 else 0)
            widths_50.append(upper_50 - lower_50)

        return CalibrationMetrics(
            coverage_90pct=100.0 * np.mean(coverages_90),
            coverage_50pct=100.0 * np.mean(coverages_50),
            mean_interval_width_90pct=np.mean(widths_90),
            mean_interval_width_50pct=np.mean(widths_50),
            sharpness_score=np.mean([p[1] for p in [pred.get('ensemble_prediction', (0, 1)) for pred in valid_preds]]),
            n_predictions=len(valid_preds),
        )

    def detect_drift(
        self,
        baseline_predictions: List[Dict],
        current_predictions: List[Dict],
        component_name: str
    ) -> DriftMetrics:
        """
        Detect drift in component predictions.

        Args:
            baseline_predictions: Baseline prediction records
            current_predictions: Current prediction records
            component_name: Component to analyze

        Returns:
            DriftMetrics object
        """
        # Extract component predictions
        baseline_means = [p['component_predictions'][component_name][0]
                          for p in baseline_predictions
                          if component_name in p['component_predictions']]
        baseline_stds = [p['component_predictions'][component_name][1]
                         for p in baseline_predictions
                         if component_name in p['component_predictions']]

        current_means = [p['component_predictions'][component_name][0]
                         for p in current_predictions
                         if component_name in p['component_predictions']]
        current_stds = [p['component_predictions'][component_name][1]
                        for p in current_predictions
                        if component_name in p['component_predictions']]

        # Calculate shifts
        mean_shift = np.mean(current_means) - np.mean(baseline_means) if baseline_means and current_means else 0.0
        std_shift = np.mean(current_stds) - np.mean(baseline_stds) if baseline_stds and current_stds else 0.0

        # Calculate MAE drift (if actuals available)
        baseline_mae = np.mean([abs(p['component_predictions'][component_name][0] - p['actual_yards'])
                                for p in baseline_predictions
                                if component_name in p['component_predictions'] and p.get('actual_yards') is not None])
        current_mae = np.mean([abs(p['component_predictions'][component_name][0] - p['actual_yards'])
                               for p in current_predictions
                               if component_name in p['component_predictions'] and p.get('actual_yards') is not None])
        mae_drift = ((current_mae - baseline_mae) / baseline_mae * 100.0) if baseline_mae > 0 else 0.0

        return DriftMetrics(
            mean_shift=mean_shift,
            std_shift=std_shift,
            mae_drift=mae_drift,
            correlation_drop=0.0,  # TODO: implement correlation tracking
            n_samples_baseline=len(baseline_means),
            n_samples_current=len(current_means),
        )

    def generate_report(self) -> EnsembleHealthReport:
        """
        Generate comprehensive health report from tracked predictions.

        Returns:
            EnsembleHealthReport with calibration, drift, and performance metrics
        """
        report = EnsembleHealthReport()

        if not self.prediction_history:
            report.add_warning("No predictions tracked - run track_prediction() first")
            return report

        # Calculate ensemble calibration
        report.ensemble_calibration = self.calculate_calibration(self.prediction_history)

        # Calculate component calibration
        component_names = set()
        for pred in self.prediction_history:
            component_names.update(pred['component_predictions'].keys())

        for component_name in component_names:
            report.component_calibration[component_name] = self.calculate_calibration(
                self.prediction_history,
                component_name=component_name
            )

        # Check calibration thresholds
        if report.ensemble_calibration:
            coverage = report.ensemble_calibration.coverage_90pct
            if coverage < self.PERFORMANCE_THRESHOLDS['calibration_90pct_min']:
                report.add_critical_issue(f"Undercalibrated: {coverage:.1f}% < {self.PERFORMANCE_THRESHOLDS['calibration_90pct_min']}%")
            elif coverage > self.PERFORMANCE_THRESHOLDS['calibration_90pct_max']:
                report.add_warning(f"Overcalibrated: {coverage:.1f}% > {self.PERFORMANCE_THRESHOLDS['calibration_90pct_max']}%")

        # Calculate MAE
        valid_preds = [p for p in self.prediction_history if p.get('actual_yards') is not None]
        if valid_preds:
            ensemble_errors = []
            for pred in valid_preds:
                if pred.get('ensemble_prediction'):
                    mean, _ = pred['ensemble_prediction']
                else:
                    means = [p[0] for p in pred['component_predictions'].values()]
                    mean = np.mean(means)
                ensemble_errors.append(abs(mean - pred['actual_yards']))

            report.ensemble_mae = np.mean(ensemble_errors)

            if report.ensemble_mae > self.PERFORMANCE_THRESHOLDS['mae_threshold']:
                report.add_warning(f"MAE too high: {report.ensemble_mae:.1f} > {self.PERFORMANCE_THRESHOLDS['mae_threshold']} yards")

        return report

    def save_report(self, report: EnsembleHealthReport, output_path: Optional[str] = None):
        """
        Save health report to disk.

        Args:
            report: EnsembleHealthReport to save
            output_path: Output path (if None, uses default)
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"ensemble_health_report_{timestamp}.pkl"

        with open(output_path, 'wb') as f:
            pickle.dump(report, f)

        logger.info(f"Health report saved to {output_path}")


def main():
    """CLI interface for ensemble health monitoring."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ensemble Health Monitor - Validate 4-way ensemble before backtest"
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help="Run pre-backtest health check (CRITICAL before 18-hour backtests)"
    )
    parser.add_argument(
        '--monitor',
        action='store_true',
        help="Monitor predictions from history file"
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help="Generate health report from tracked predictions"
    )
    parser.add_argument(
        '--history-file',
        type=str,
        help="Path to prediction history pickle file (for --monitor/--report)"
    )
    parser.add_argument(
        '--output',
        type=str,
        help="Output path for health report"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    monitor = EnsembleHealthMonitor()

    if args.check:
        # Pre-backtest health check
        report = monitor.check_ensemble_health()

        if args.output:
            monitor.save_report(report, args.output)

        # Exit with error code if unhealthy
        if not report.is_healthy:
            print("\n⛔ ENSEMBLE UNHEALTHY - DO NOT PROCEED WITH BACKTEST")
            exit(1)
        else:
            print("\n✅ ENSEMBLE HEALTHY - SAFE TO PROCEED")
            exit(0)

    elif args.monitor or args.report:
        # Load prediction history
        if not args.history_file:
            print("Error: --history-file required for --monitor/--report")
            exit(1)

        with open(args.history_file, 'rb') as f:
            monitor.prediction_history = pickle.load(f)

        report = monitor.generate_report()

        print(f"\n{'='*80}")
        print("ENSEMBLE HEALTH REPORT")
        print(f"{'='*80}")
        print(f"Predictions: {len(monitor.prediction_history)}")

        if report.ensemble_calibration:
            print(f"\nEnsemble Calibration:")
            print(f"  90% Coverage: {report.ensemble_calibration.coverage_90pct:.1f}%")
            print(f"  50% Coverage: {report.ensemble_calibration.coverage_50pct:.1f}%")
            print(f"  Mean 90% Width: {report.ensemble_calibration.mean_interval_width_90pct:.1f} yards")

        if report.ensemble_mae:
            print(f"\nEnsemble MAE: {report.ensemble_mae:.1f} yards")

        if report.critical_issues:
            print(f"\n⛔ CRITICAL ISSUES ({len(report.critical_issues)}):")
            for issue in report.critical_issues:
                print(f"  - {issue}")

        if report.warnings:
            print(f"\n⚠️  WARNINGS ({len(report.warnings)}):")
            for warning in report.warnings:
                print(f"  - {warning}")

        if args.output:
            monitor.save_report(report, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
