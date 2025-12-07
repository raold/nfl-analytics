"""
A/B Test Monitor Integration Examples

Demonstrates real-world integration patterns for the A/B Test Monitor Agent.

Examples:
1. Pre-backtest guardrail validation
2. Real-time monitoring loop
3. Statistical analysis workflow
4. Dashboard integration
5. Production deployment gate

Author: Claude Code (Sonnet 4.5)
Created: November 22, 2025
"""

import logging
import time
from datetime import datetime, timedelta

import numpy as np
from py.utils.ab_test_monitor import ABTestMonitor, Decision

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Example 1: Pre-Backtest Guardrail Validation
# ============================================================================


def example_1_pre_backtest_validation():
    """
    Prevent expensive backtest if A/B test is already violated.

    ROI: Saves 18 hours if backtest would fail anyway.
    """
    logger.info("Example 1: Pre-backtest guardrail validation")

    monitor = ABTestMonitor()

    # Check guardrails before running expensive backtest
    logger.info("Checking guardrails before backtest...")
    status = monitor.check_guardrails()

    if status.violated:
        logger.error(f"⛔ Guardrails violated: {status.violations}")
        logger.error("Skipping 18-hour backtest to save resources")
        return False
    else:
        logger.info("✅ Guardrails passed. Safe to run backtest.")
        # Would run: python py/backtest/historical_validation_4way.py
        return True


# ============================================================================
# Example 2: Real-Time Monitoring Loop
# ============================================================================


def example_2_realtime_monitoring_loop(duration_days: int = 7):
    """
    Continuous monitoring with auto-stop on violations.

    Args:
        duration_days: How long to monitor (default 7 days)

    ROI: Auto-stop within 1 hour of violation vs manual detection (hours/days).
    """
    logger.info(f"Example 2: Real-time monitoring for {duration_days} days")

    monitor = ABTestMonitor()

    start_time = datetime.now()
    end_time = start_time + timedelta(days=duration_days)

    while datetime.now() < end_time:
        logger.info(f"--- Monitoring check at {datetime.now():%H:%M:%S} ---")

        # 1. Check guardrails
        status = monitor.check_guardrails()
        if status.violated:
            logger.error(f"⛔ Auto-stopping due to violations: {status.violations}")
            stopped = monitor.auto_stop_if_violated()
            if stopped:
                logger.error("Test stopped. Alerting stakeholders...")
                # send_slack_alert("A/B test auto-stopped!")
                break

        # 2. Get current metrics
        metrics = monitor.get_current_metrics()
        logger.info(f"Control MAE: {metrics['control'].get('mae', 'N/A'):.2f}")
        logger.info(f"Treatment MAE: {metrics['treatment'].get('mae', 'N/A'):.2f}")

        # 3. Calculate significance (if enough data)
        if metrics["treatment"].get("sample_size", 0) >= 50:
            mae_sig = monitor.calculate_significance("mae", test_type="t-test")
            logger.info(f"MAE significance: p={mae_sig.p_value:.4f}")

            # 4. Get recommendation
            decision = monitor.recommend_decision()
            logger.info(f"Recommendation: {decision.value}")

            # 5. Early stopping
            if decision == Decision.PROMOTE_TREATMENT:
                logger.info("✅ Early stopping: Promoting treatment")
                break
            elif decision == Decision.STOP_NO_EFFECT:
                logger.info("⏹️  Early stopping: No effect detected (futility)")
                break

        # Wait 1 hour (in production, would be cron job)
        logger.info("Sleeping 1 hour...")
        time.sleep(3600)

    logger.info("Monitoring complete")


# ============================================================================
# Example 3: Statistical Analysis Workflow
# ============================================================================


def example_3_statistical_analysis():
    """
    Comprehensive statistical analysis across multiple metrics and methods.

    ROI: Rigorous statistical validation prevents premature deployments.
    """
    logger.info("Example 3: Statistical analysis workflow")

    monitor = ABTestMonitor()

    # 1. Check sample size
    metrics = monitor.get_current_metrics()
    control_n = metrics["control"].get("sample_size", 0)
    treatment_n = metrics["treatment"].get("sample_size", 0)

    logger.info(f"Sample sizes: Control={control_n}, Treatment={treatment_n}")

    if treatment_n < 50:
        logger.warning("Insufficient sample size for analysis")
        return

    # 2. Multi-method significance testing
    logger.info("\n=== Statistical Significance Tests ===")

    for metric in ["mae", "roi", "win_rate"]:
        logger.info(f"\n{metric.upper()}:")

        # T-test
        ttest = monitor.calculate_significance(metric, test_type="t-test")
        logger.info(
            f"  T-test: p={ttest.p_value:.4f}, diff={ttest.difference:.2f} "
            f"({ttest.difference_pct:+.1f}%)"
        )

        # Bootstrap
        bootstrap = monitor.calculate_significance(metric, test_type="bootstrap")
        logger.info(
            f"  Bootstrap: p={bootstrap.p_value:.4f}, "
            f"95% CI=[{bootstrap.confidence_interval_95[0]:.2f}, "
            f"{bootstrap.confidence_interval_95[1]:.2f}]"
        )

        # Bayesian
        bayesian = monitor.calculate_significance(metric, test_type="bayesian")
        logger.info(
            f"  Bayesian: p={bayesian.p_value:.4f}, "
            f"95% credible interval=[{bayesian.confidence_interval_95[0]:.2f}, "
            f"{bayesian.confidence_interval_95[1]:.2f}]"
        )

        # Effect size
        logger.info(f"  Cohen's d: {ttest.effect_size:.3f}")

        # Consensus
        all_significant = ttest.significant and bootstrap.significant and bayesian.significant
        logger.info(f"  Consensus: {'✅ SIGNIFICANT' if all_significant else '❌ NOT SIGNIFICANT'}")

    # 3. Final recommendation
    logger.info("\n=== Final Recommendation ===")
    decision = monitor.recommend_decision()
    logger.info(f"Decision: {decision.value}")

    # 4. Generate report
    report = monitor.generate_report()
    logger.info(f"\n{report}")


# ============================================================================
# Example 4: Dashboard Integration
# ============================================================================


def example_4_dashboard_integration() -> dict:
    """
    Prepare data for Streamlit/Grafana dashboard.

    Returns:
        Dict with dashboard data (metrics, status, recommendation)

    ROI: Real-time visibility into experiment health for stakeholders.
    """
    logger.info("Example 4: Dashboard integration")

    monitor = ABTestMonitor()

    # 1. Current metrics
    metrics = monitor.get_current_metrics()

    # 2. Guardrail status
    guardrail_status = monitor.check_guardrails()

    # 3. Significance tests
    mae_sig = monitor.calculate_significance("mae")
    roi_sig = monitor.calculate_significance("roi")

    # 4. Recommendation
    decision = monitor.recommend_decision()

    # 5. Export results
    results_df = monitor.export_results()

    # Prepare dashboard data
    dashboard_data = {
        "metrics": {
            "control": metrics.get("control", {}),
            "treatment": metrics.get("treatment", {}),
        },
        "guardrails": {
            "violated": guardrail_status.violated,
            "violations": guardrail_status.violations,
            "warnings": guardrail_status.warnings,
        },
        "significance": {"mae": mae_sig.to_dict(), "roi": roi_sig.to_dict()},
        "recommendation": decision.value,
        "results_df": results_df,
        "timestamp": datetime.now().isoformat(),
    }

    logger.info("Dashboard data prepared")
    logger.info(f"  Control MAE: {metrics['control'].get('mae', 'N/A')}")
    logger.info(f"  Treatment MAE: {metrics['treatment'].get('mae', 'N/A')}")
    logger.info(f"  Guardrails: {'⛔ VIOLATED' if guardrail_status.violated else '✅ PASSED'}")
    logger.info(f"  Recommendation: {decision.value}")

    return dashboard_data


# ============================================================================
# Example 5: Production Deployment Gate
# ============================================================================


def example_5_production_deployment_gate() -> bool:
    """
    Validate A/B test before deploying new model to production.

    Returns:
        True if safe to deploy, False otherwise

    ROI: Prevents premature deployments that could lose $10k+ bankroll.
    """
    logger.info("Example 5: Production deployment gate")

    monitor = ABTestMonitor()

    try:
        # 1. Check sample size
        metrics = monitor.get_current_metrics()
        if metrics["treatment"].get("sample_size", 0) < 100:
            raise RuntimeError("Insufficient sample size (need ≥100 games)")

        # 2. Check guardrails
        status = monitor.check_guardrails()
        if status.violated:
            raise RuntimeError(f"Guardrails violated: {status.violations}")

        # 3. Check recommendation
        decision = monitor.recommend_decision()
        if decision != Decision.PROMOTE_TREATMENT:
            raise RuntimeError(f"Not recommended for promotion: {decision.value}")

        # 4. Verify statistical significance
        mae_sig = monitor.calculate_significance("mae")
        roi_sig = monitor.calculate_significance("roi")

        if not mae_sig.significant:
            raise RuntimeError(f"MAE not significant (p={mae_sig.p_value:.4f})")

        if not roi_sig.significant:
            raise RuntimeError(f"ROI not significant (p={roi_sig.p_value:.4f})")

        # 5. Check effect size (practical significance)
        if abs(mae_sig.effect_size) < 0.2:
            logger.warning(f"Small effect size (d={mae_sig.effect_size:.3f})")

        # 6. Final approval
        logger.info("✅ All deployment gates passed:")
        logger.info(f"  ✅ Sample size: {metrics['treatment']['sample_size']} ≥ 100")
        logger.info("  ✅ Guardrails: PASSED")
        logger.info("  ✅ Recommendation: PROMOTE")
        logger.info(f"  ✅ MAE significant: p={mae_sig.p_value:.4f} < 0.05")
        logger.info(f"  ✅ ROI significant: p={roi_sig.p_value:.4f} < 0.05")
        logger.info(f"  ✅ Effect size: d={mae_sig.effect_size:.3f}")

        logger.info("\n🚀 Safe to deploy ensemble_v6 to production")
        return True

    except RuntimeError as e:
        logger.error(f"⛔ Deployment blocked: {e}")
        logger.error("❌ NOT safe to deploy")
        return False


# ============================================================================
# Example 6: Simulated A/B Test
# ============================================================================


def example_6_simulate_ab_test(n_days: int = 28):
    """
    Simulate a complete A/B test workflow with synthetic data.

    Args:
        n_days: Number of days to simulate (default 28 = 4 weeks)

    Demonstrates:
    - Daily metric tracking
    - Guardrail enforcement
    - Statistical testing
    - Decision evolution over time
    """
    logger.info(f"Example 6: Simulating {n_days}-day A/B test")

    # Create synthetic daily results
    np.random.seed(42)

    # Control: MAE ~40 yards, ROI ~5%
    control_mae = np.random.normal(40, 5, n_days)
    control_roi = np.random.normal(5, 2, n_days)

    # Treatment: MAE ~33 yards (7 yard improvement), ROI ~7% (2pp improvement)
    treatment_mae = np.random.normal(33, 5, n_days)
    treatment_roi = np.random.normal(7, 2, n_days)

    logger.info("\n=== Daily Monitoring ===")

    for day in range(n_days):
        logger.info(f"\n--- Day {day + 1} ---")

        # Cumulative metrics
        control_mae_cumulative = control_mae[: day + 1].mean()
        treatment_mae_cumulative = treatment_mae[: day + 1].mean()
        mae_diff = treatment_mae_cumulative - control_mae_cumulative

        control_roi_cumulative = control_roi[: day + 1].mean()
        treatment_roi_cumulative = treatment_roi[: day + 1].mean()
        roi_diff = treatment_roi_cumulative - control_roi_cumulative

        logger.info(f"Control MAE: {control_mae_cumulative:.2f}")
        logger.info(f"Treatment MAE: {treatment_mae_cumulative:.2f}")
        logger.info(f"Difference: {mae_diff:+.2f} yards")

        logger.info(f"Control ROI: {control_roi_cumulative:.2f}%")
        logger.info(f"Treatment ROI: {treatment_roi_cumulative:.2f}%")
        logger.info(f"Difference: {roi_diff:+.2f}pp")

        # Check if we would stop early
        if day >= 13:  # After 2 weeks (min 50 samples)
            # Simulate t-test p-value (would use actual data in production)
            mae_p_value = np.abs(mae_diff) / (5 / np.sqrt(day + 1))  # Rough estimate
            mae_p_value = 1 - np.clip(mae_p_value, 0, 1)

            logger.info(f"Estimated MAE p-value: {mae_p_value:.4f}")

            if mae_p_value < 0.01 and mae_diff < -5:  # Significant improvement
                logger.info("🎉 Early stopping: Treatment significantly better!")
                logger.info(f"Can promote after {day+1} days (vs planned 28)")
                break

    logger.info("\n=== Final Analysis ===")
    logger.info(f"Total duration: {day+1} days")
    logger.info(
        f"Final MAE improvement: {mae_diff:+.2f} yards ({(mae_diff/control_mae_cumulative)*100:+.1f}%)"
    )
    logger.info(f"Final ROI improvement: {roi_diff:+.2f}pp")


# ============================================================================
# Main
# ============================================================================


def main():
    """Run all examples."""
    print("=" * 80)
    print("A/B Test Monitor Integration Examples")
    print("=" * 80)

    # Example 1: Pre-backtest validation
    print("\n" + "=" * 80)
    example_1_pre_backtest_validation()

    # Example 3: Statistical analysis (skip example 2 as it sleeps)
    print("\n" + "=" * 80)
    example_3_statistical_analysis()

    # Example 4: Dashboard integration
    print("\n" + "=" * 80)
    example_4_dashboard_integration()

    # Example 5: Production deployment gate
    print("\n" + "=" * 80)
    example_5_production_deployment_gate()

    # Example 6: Simulated A/B test
    print("\n" + "=" * 80)
    example_6_simulate_ab_test(n_days=28)

    print("\n" + "=" * 80)
    print("All examples complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
