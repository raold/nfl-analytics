#!/usr/bin/env python3
"""
Example: Ensemble Health Monitor Integration

Demonstrates how to integrate the ensemble health monitor into existing
backtest workflows to prevent R model loading failures.

This example shows the EXACT pattern that would have prevented the November 13, 2025
failure where R models failed to load during 18-hour backtest.

Author: Claude Code (Sonnet 4.5)
Date: November 14, 2025
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def example_1_pre_backtest_validation():
    """
    Example 1: Pre-Backtest Validation (CRITICAL)

    Run this BEFORE expensive backtests to catch R model loading failures.
    """
    from nfl_analytics.utils.ensemble_health_monitor import EnsembleHealthMonitor

    logger.info("=" * 80)
    logger.info("EXAMPLE 1: Pre-Backtest Validation")
    logger.info("=" * 80)

    # Initialize monitor
    monitor = EnsembleHealthMonitor()

    # Run health check
    report = monitor.check_ensemble_health()

    # Check if healthy
    if report.is_healthy:
        logger.info("\n✅ ENSEMBLE HEALTHY - Safe to run backtest")
        logger.info(
            f"   Components loaded: {sum(1 for h in report.components.values() if h.rpy2_test_passed)}/4"
        )
        logger.info(f"   Warnings: {len(report.warnings)}")
        return True
    else:
        logger.error("\n⛔ ENSEMBLE UNHEALTHY - DO NOT RUN BACKTEST")
        logger.error(f"   Critical issues: {len(report.critical_issues)}")
        for issue in report.critical_issues:
            logger.error(f"   - {issue}")
        return False


def example_2_integrated_backtest():
    """
    Example 2: Integrated Backtest with Health Monitoring

    Shows how to integrate health monitoring into backtest workflow.
    """
    import pandas as pd

    from nfl_analytics.ensemble.enhanced_ensemble_v5 import EnhancedEnsemble4Way
    from nfl_analytics.utils.ensemble_health_monitor import EnsembleHealthMonitor

    logger.info("=" * 80)
    logger.info("EXAMPLE 2: Integrated Backtest with Health Monitoring")
    logger.info("=" * 80)

    # Pre-flight check
    monitor = EnsembleHealthMonitor()
    report = monitor.check_ensemble_health()

    if not report.is_healthy:
        logger.error("⛔ Pre-flight check failed - aborting backtest")
        return

    logger.info("\n✅ Pre-flight check passed - starting backtest")

    # Initialize ensemble
    ensemble = EnhancedEnsemble4Way()

    # Simulate backtest (replace with real data)
    test_data = pd.DataFrame(
        {
            "player_id": ["00-0019596", "00-0033077", "00-0036212"],  # Mahomes, Allen, Herbert
            "season": [2024, 2024, 2024],
            "week": [1, 1, 1],
            "actual_yards": [275, 301, 254],  # Simulated actuals
        }
    )

    logger.info(f"\nRunning backtest on {len(test_data)} games...")

    for idx, row in test_data.iterrows():
        # Get predictions (this would fail if R models not loaded)
        try:
            result = ensemble.predict(
                player_id=row["player_id"], season=row["season"], week=row["week"]
            )

            # Track prediction
            monitor.track_prediction(
                player_id=row["player_id"],
                season=row["season"],
                week=row["week"],
                component_predictions=result.get("component_predictions", {}),
                actual_yards=row["actual_yards"],
                ensemble_prediction=(
                    result.get("ensemble_mean", 0.0),
                    result.get("ensemble_std", 0.0),
                ),
            )

            logger.info(
                f"  {row['player_id']} W{row['week']}: {result.get('ensemble_mean', 0):.1f} yards (actual: {row['actual_yards']})"
            )

        except Exception as e:
            logger.error(f"⚠️  Prediction failed for {row['player_id']}: {e}")
            continue

    # Generate final report
    final_report = monitor.generate_report()

    logger.info("\n" + "=" * 80)
    logger.info("BACKTEST COMPLETE - Health Report")
    logger.info("=" * 80)

    if final_report.ensemble_calibration:
        logger.info(
            f"Calibration (90% CI): {final_report.ensemble_calibration.coverage_90pct:.1f}%"
        )
        logger.info(
            f"Mean interval width: {final_report.ensemble_calibration.mean_interval_width_90pct:.1f} yards"
        )

    if final_report.ensemble_mae:
        logger.info(f"Ensemble MAE: {final_report.ensemble_mae:.1f} yards")

    # Save report
    monitor.save_report(final_report, "ensemble_health_report_example.pkl")


def example_3_continuous_monitoring():
    """
    Example 3: Continuous Monitoring During Production

    Shows how to track predictions in production for drift detection.
    """
    import numpy as np

    from nfl_analytics.utils.ensemble_health_monitor import EnsembleHealthMonitor

    logger.info("=" * 80)
    logger.info("EXAMPLE 3: Continuous Monitoring")
    logger.info("=" * 80)

    monitor = EnsembleHealthMonitor()

    # Simulate baseline predictions (first 50 games)
    logger.info("\nSimulating baseline predictions...")
    for i in range(50):
        monitor.track_prediction(
            player_id=f"player_{i % 10}",
            season=2024,
            week=i // 10 + 1,
            component_predictions={
                "state_space_v2": (np.random.normal(250, 40), 35),
                "hierarchical_v2": (np.random.normal(248, 38), 33),
                "xgboost_v2": (np.random.normal(252, 42), 37),
                "informative_priors_v2": (np.random.normal(249, 39), 34),
            },
            actual_yards=np.random.normal(250, 50),
        )

    baseline_preds = monitor.prediction_history[:50]

    # Simulate current predictions with drift (next 50 games)
    logger.info("Simulating current predictions with drift...")
    for i in range(50, 100):
        # Introduce drift: hierarchical_v2 mean shifts by +15 yards
        monitor.track_prediction(
            player_id=f"player_{i % 10}",
            season=2024,
            week=i // 10 + 1,
            component_predictions={
                "state_space_v2": (np.random.normal(250, 40), 35),
                "hierarchical_v2": (np.random.normal(265, 38), 33),  # +15 yards drift
                "xgboost_v2": (np.random.normal(252, 42), 37),
                "informative_priors_v2": (np.random.normal(249, 39), 34),
            },
            actual_yards=np.random.normal(250, 50),
        )

    current_preds = monitor.prediction_history[50:]

    # Detect drift
    drift = monitor.detect_drift(baseline_preds, current_preds, "hierarchical_v2")

    logger.info("\n" + "=" * 80)
    logger.info("DRIFT DETECTION RESULTS")
    logger.info("=" * 80)
    logger.info("Component: hierarchical_v2")
    logger.info(f"Mean shift: {drift.mean_shift:+.1f} yards")
    logger.info(f"Std shift: {drift.std_shift:+.1f}")
    logger.info(f"MAE drift: {drift.mae_drift:+.1f}%")

    if abs(drift.mean_shift) > monitor.DRIFT_THRESHOLDS["mean_shift_yards"]:
        logger.warning(f"⚠️  DRIFT DETECTED: Mean shifted by {drift.mean_shift:+.1f} yards")
        logger.warning("   Recommended action: Retrain hierarchical_v2 model")


def example_4_ci_cd_integration():
    """
    Example 4: CI/CD Integration Pattern

    Shows how to use health monitor in CI/CD pipelines.
    """
    from nfl_analytics.utils.ensemble_health_monitor import EnsembleHealthMonitor

    logger.info("=" * 80)
    logger.info("EXAMPLE 4: CI/CD Integration")
    logger.info("=" * 80)

    # This would be run in GitHub Actions workflow
    monitor = EnsembleHealthMonitor()
    report = monitor.check_ensemble_health()

    # Exit codes for CI/CD
    if len(report.critical_issues) > 0:
        logger.error(f"\n⛔ CI/CD FAILED - {len(report.critical_issues)} critical issues")
        for issue in report.critical_issues:
            logger.error(f"  - {issue}")
        exit(1)  # Fail CI/CD pipeline

    elif len(report.warnings) > 0:
        logger.warning(f"\n⚠️  CI/CD PASSED WITH WARNINGS - {len(report.warnings)} warnings")
        for warning in report.warnings:
            logger.warning(f"  - {warning}")
        exit(0)  # Pass but flag warnings

    else:
        logger.info("\n✅ CI/CD PASSED - No issues detected")
        exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ensemble Health Monitor Integration Examples")
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help="Example to run (1: pre-backtest, 2: integrated, 3: monitoring, 4: CI/CD)",
    )

    args = parser.parse_args()

    examples = {
        1: example_1_pre_backtest_validation,
        2: example_2_integrated_backtest,
        3: example_3_continuous_monitoring,
        4: example_4_ci_cd_integration,
    }

    try:
        examples[args.example]()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        exit(1)
