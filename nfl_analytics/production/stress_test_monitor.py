"""
stress_test_monitor.py

Weekly Bootstrap Stress Testing for Production Models

Runs bootstrap simulations on live production bets to verify model performance:
- Compares actual ROI vs bootstrapped distribution
- Detects statistical underperformance (below 5th percentile)
- Tests model resilience under adverse scenarios
- Generates weekly stress test reports

Usage:
    # Run weekly stress test
    python py/production/stress_test_monitor.py run --weeks 4

    # Generate stress test report
    python py/production/stress_test_monitor.py report

    # Check if model failing stress tests
    python py/production/stress_test_monitor.py check
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class StressTestResult:
    """Stress test results."""

    test_date: datetime
    weeks_tested: int
    n_bets: int
    actual_roi: float
    bootstrap_mean_roi: float
    bootstrap_median_roi: float
    bootstrap_std_roi: float
    bootstrap_5th_percentile: float
    bootstrap_95th_percentile: float
    percentile_rank: float  # Where actual ROI ranks in bootstrap distribution
    worst_case_roi: float  # Worst 1% of bootstrap scenarios
    best_case_roi: float  # Best 1% of bootstrap scenarios
    passed: bool  # True if actual ROI > 5th percentile


# ============================================================================
# Stress Test Monitor
# ============================================================================


class StressTestMonitor:
    """
    Monitor production model performance via bootstrap stress tests.
    """

    def __init__(
        self,
        db_url: str = "postgresql://dro:sicillionbillions@localhost:5544/devdb01",
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95,
    ):
        """
        Initialize stress test monitor.

        Args:
            db_url: Database connection URL
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
        """
        self.engine = create_engine(db_url)
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level

        logger.info(f"Stress test monitor initialized (bootstrap samples: {n_bootstrap})")

    def get_recent_bets(self, weeks: int = 4) -> pd.DataFrame:
        """
        Get recent production bets.

        Args:
            weeks: Number of weeks to look back

        Returns:
            Bets dataframe
        """
        start_date = datetime.now() - timedelta(weeks=weeks)

        query = """
            SELECT *
            FROM bets
            WHERE timestamp >= %s
                AND result IS NOT NULL
            ORDER BY timestamp
        """

        with self.engine.connect() as conn:
            bets = pd.read_sql(query, conn, params=(start_date,))

        logger.info(f"Loaded {len(bets)} bets from last {weeks} weeks")

        return bets

    def bootstrap_resample(self, bets: pd.DataFrame) -> np.ndarray:
        """
        Bootstrap resample bets and calculate ROI distribution.

        Args:
            bets: Bets dataframe

        Returns:
            Array of bootstrap ROI values (n_bootstrap,)
        """
        n_bets = len(bets)
        bootstrap_rois = np.zeros(self.n_bootstrap)

        payouts = bets["payout"].values
        stakes = bets["stake"].values

        for i in range(self.n_bootstrap):
            # Resample with replacement
            idx = np.random.choice(n_bets, size=n_bets, replace=True)
            bootstrap_payouts = payouts[idx]
            bootstrap_stakes = stakes[idx]

            # Calculate ROI
            roi = bootstrap_payouts.sum() / bootstrap_stakes.sum()
            bootstrap_rois[i] = roi

        return bootstrap_rois

    def run_stress_test(self, weeks: int = 4) -> StressTestResult:
        """
        Run bootstrap stress test on recent bets.

        Args:
            weeks: Number of weeks to test

        Returns:
            StressTestResult
        """
        logger.info(f"Running stress test on last {weeks} weeks of bets...")

        # Load recent bets
        bets = self.get_recent_bets(weeks=weeks)

        if len(bets) < 10:
            logger.warning(f"Only {len(bets)} bets available, need at least 10 for stress test")
            return StressTestResult(
                test_date=datetime.now(),
                weeks_tested=weeks,
                n_bets=len(bets),
                actual_roi=0.0,
                bootstrap_mean_roi=0.0,
                bootstrap_median_roi=0.0,
                bootstrap_std_roi=0.0,
                bootstrap_5th_percentile=0.0,
                bootstrap_95th_percentile=0.0,
                percentile_rank=0.0,
                worst_case_roi=0.0,
                best_case_roi=0.0,
                passed=True,  # Pass by default if not enough data
            )

        # Calculate actual ROI
        actual_roi = bets["payout"].sum() / bets["stake"].sum()

        # Bootstrap resample
        logger.info(f"Running {self.n_bootstrap:,} bootstrap samples...")
        bootstrap_rois = self.bootstrap_resample(bets)

        # Calculate statistics
        bootstrap_mean = np.mean(bootstrap_rois)
        bootstrap_median = np.median(bootstrap_rois)
        bootstrap_std = np.std(bootstrap_rois)

        # Confidence intervals
        alpha = 1 - self.confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100

        bootstrap_5th = np.percentile(bootstrap_rois, lower_percentile)
        bootstrap_95th = np.percentile(bootstrap_rois, upper_percentile)

        # Worst/best case (1st and 99th percentiles)
        worst_case = np.percentile(bootstrap_rois, 1)
        best_case = np.percentile(bootstrap_rois, 99)

        # Where does actual ROI rank?
        percentile_rank = stats.percentileofscore(bootstrap_rois, actual_roi)

        # Pass if actual ROI > 5th percentile
        passed = actual_roi >= bootstrap_5th

        logger.info(f"Actual ROI: {actual_roi:+.2%} (Percentile: {percentile_rank:.1f}%)")
        logger.info(f"Bootstrap Mean: {bootstrap_mean:+.2%} ± {bootstrap_std:.2%}")
        logger.info(f"95% CI: [{bootstrap_5th:+.2%}, {bootstrap_95th:+.2%}]")
        logger.info(f"Stress Test: {'PASSED' if passed else 'FAILED'}")

        return StressTestResult(
            test_date=datetime.now(),
            weeks_tested=weeks,
            n_bets=len(bets),
            actual_roi=actual_roi,
            bootstrap_mean_roi=bootstrap_mean,
            bootstrap_median_roi=bootstrap_median,
            bootstrap_std_roi=bootstrap_std,
            bootstrap_5th_percentile=bootstrap_5th,
            bootstrap_95th_percentile=bootstrap_95th,
            percentile_rank=percentile_rank,
            worst_case_roi=worst_case,
            best_case_roi=best_case,
            passed=passed,
        )

    def save_stress_test(self, result: StressTestResult):
        """
        Save stress test result to database.

        Args:
            result: StressTestResult
        """
        query = """
            INSERT INTO stress_tests (
                test_date, weeks_tested, n_bets, actual_roi,
                bootstrap_mean_roi, bootstrap_median_roi, bootstrap_std_roi,
                bootstrap_5th_percentile, bootstrap_95th_percentile,
                percentile_rank, worst_case_roi, best_case_roi, passed
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
        """

        with self.engine.connect() as conn:
            conn.execute(
                query,
                (
                    result.test_date,
                    result.weeks_tested,
                    result.n_bets,
                    result.actual_roi,
                    result.bootstrap_mean_roi,
                    result.bootstrap_median_roi,
                    result.bootstrap_std_roi,
                    result.bootstrap_5th_percentile,
                    result.bootstrap_95th_percentile,
                    result.percentile_rank,
                    result.worst_case_roi,
                    result.best_case_roi,
                    result.passed,
                ),
            )
            conn.commit()

        logger.info("Stress test result saved to database")

    def get_stress_test_history(self, limit: int = 10) -> pd.DataFrame:
        """
        Get historical stress test results.

        Args:
            limit: Number of recent tests to retrieve

        Returns:
            Stress tests dataframe
        """
        query = f"""
            SELECT *
            FROM stress_tests
            ORDER BY test_date DESC
            LIMIT {limit}
        """

        with self.engine.connect() as conn:
            tests = pd.read_sql(query, conn)

        return tests

    def generate_report(self) -> dict:
        """
        Generate stress test report.

        Returns:
            Report dictionary
        """
        # Get latest stress test
        latest = self.get_stress_test_history(limit=1)

        if len(latest) == 0:
            logger.warning("No stress tests found")
            return {}

        result = latest.iloc[0]

        # Get historical tests
        history = self.get_stress_test_history(limit=10)

        # Calculate failure rate
        failure_rate = (not history["passed"]).mean()

        report = {
            "latest_test": {
                "test_date": result["test_date"].isoformat(),
                "weeks_tested": int(result["weeks_tested"]),
                "n_bets": int(result["n_bets"]),
                "actual_roi": float(result["actual_roi"]),
                "bootstrap_mean_roi": float(result["bootstrap_mean_roi"]),
                "bootstrap_std_roi": float(result["bootstrap_std_roi"]),
                "confidence_interval": [
                    float(result["bootstrap_5th_percentile"]),
                    float(result["bootstrap_95th_percentile"]),
                ],
                "percentile_rank": float(result["percentile_rank"]),
                "worst_case_roi": float(result["worst_case_roi"]),
                "best_case_roi": float(result["best_case_roi"]),
                "passed": bool(result["passed"]),
            },
            "historical_summary": {
                "total_tests": len(history),
                "passed": int((history["passed"]).sum()),
                "failed": int((not history["passed"]).sum()),
                "failure_rate": float(failure_rate),
                "avg_actual_roi": float(history["actual_roi"].mean()),
                "avg_percentile_rank": float(history["percentile_rank"].mean()),
            },
        }

        return report

    def check_health(self) -> list[str]:
        """
        Check if model is healthy based on stress tests.

        Returns:
            List of alerts/warnings
        """
        alerts = []

        # Get recent stress tests
        tests = self.get_stress_test_history(limit=5)

        if len(tests) == 0:
            alerts.append("⚠️ WARNING: No stress tests found. Run stress test first.")
            return alerts

        latest = tests.iloc[0]

        # Alert 1: Latest stress test failed
        if not latest["passed"]:
            alerts.append(
                f"🚨 ALERT: Latest stress test FAILED. Actual ROI ({latest['actual_roi']:.2%}) "
                f"below 5th percentile ({latest['bootstrap_5th_percentile']:.2%})."
            )

        # Alert 2: Multiple recent failures
        recent_failures = (not tests.head(3)["passed"]).sum()
        if recent_failures >= 2:
            alerts.append(
                f"🚨 ALERT: {recent_failures}/3 recent stress tests failed. "
                "Model may be underperforming."
            )

        # Alert 3: Actual ROI significantly below expected
        if latest["percentile_rank"] < 10:
            alerts.append(
                f"⚠️ WARNING: Actual ROI in bottom 10% of bootstrap distribution "
                f"(Percentile: {latest['percentile_rank']:.1f}%). Review model."
            )

        # Alert 4: High variance
        if latest["bootstrap_std_roi"] > 0.15:
            alerts.append(
                f"⚠️ WARNING: High variance detected (std = {latest['bootstrap_std_roi']:.2%}). "
                "Consider reducing bet sizes."
            )

        # Alert 5: Negative worst-case scenario
        if latest["worst_case_roi"] < -0.20:
            alerts.append(
                f"⚠️ WARNING: Worst-case scenario shows -{abs(latest['worst_case_roi']):.0%} ROI. "
                "Be prepared for drawdowns."
            )

        if len(alerts) == 0:
            alerts.append("✅ All stress tests passed. Model performing as expected.")

        return alerts


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Production Stress Test Monitor")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run stress test
    run_parser = subparsers.add_parser("run", help="Run stress test")
    run_parser.add_argument(
        "--weeks", type=int, default=4, help="Number of weeks to test (default: 4)"
    )
    run_parser.add_argument(
        "--bootstrap", type=int, default=10000, help="Bootstrap samples (default: 10000)"
    )

    # Generate report
    subparsers.add_parser("report", help="Generate stress test report")

    # Check health
    subparsers.add_parser("check", help="Check model health")

    args = parser.parse_args()

    # Initialize monitor
    monitor = StressTestMonitor(n_bootstrap=getattr(args, "bootstrap", 10000))

    # Execute command
    if args.command == "run":
        result = monitor.run_stress_test(weeks=args.weeks)
        monitor.save_stress_test(result)

        print("\n" + "=" * 70)
        print("STRESS TEST RESULT")
        print("=" * 70)
        print(f"Test Date: {result.test_date}")
        print(f"Weeks Tested: {result.weeks_tested}")
        print(f"Bets: {result.n_bets}")
        print()
        print("Actual Performance:")
        print(f"  ROI: {result.actual_roi:+.2%}")
        print(f"  Percentile Rank: {result.percentile_rank:.1f}%")
        print()
        print("Bootstrap Distribution:")
        print(f"  Mean: {result.bootstrap_mean_roi:+.2%}")
        print(f"  Median: {result.bootstrap_median_roi:+.2%}")
        print(f"  Std Dev: {result.bootstrap_std_roi:.2%}")
        print(
            f"  95% CI: [{result.bootstrap_5th_percentile:+.2%}, {result.bootstrap_95th_percentile:+.2%}]"
        )
        print()
        print("Extreme Scenarios:")
        print(f"  Worst Case (1%): {result.worst_case_roi:+.2%}")
        print(f"  Best Case (99%): {result.best_case_roi:+.2%}")
        print()
        print(f"Result: {'✅ PASSED' if result.passed else '🚨 FAILED'}")
        print("=" * 70)

    elif args.command == "report":
        report = monitor.generate_report()

        if not report:
            print("No stress test data available. Run 'stress_test_monitor.py run' first.")
            return

        print("\n" + "=" * 70)
        print("STRESS TEST REPORT")
        print("=" * 70)

        latest = report["latest_test"]
        print("Latest Test:")
        print(f"  Date: {latest['test_date']}")
        print(f"  Bets: {latest['n_bets']} ({latest['weeks_tested']} weeks)")
        print(
            f"  Actual ROI: {latest['actual_roi']:+.2%} (Percentile: {latest['percentile_rank']:.1f}%)"
        )
        print(
            f"  Bootstrap: {latest['bootstrap_mean_roi']:+.2%} ± {latest['bootstrap_std_roi']:.2%}"
        )
        print(
            f"  95% CI: [{latest['confidence_interval'][0]:+.2%}, {latest['confidence_interval'][1]:+.2%}]"
        )
        print(f"  Status: {'✅ PASSED' if latest['passed'] else '🚨 FAILED'}")
        print()

        history = report["historical_summary"]
        print("Historical Summary:")
        print(f"  Total Tests: {history['total_tests']}")
        print(f"  Passed: {history['passed']} | Failed: {history['failed']}")
        print(f"  Failure Rate: {history['failure_rate']:.1%}")
        print(f"  Avg ROI: {history['avg_actual_roi']:+.2%}")
        print(f"  Avg Percentile: {history['avg_percentile_rank']:.1f}%")
        print("=" * 70)

    elif args.command == "check":
        alerts = monitor.check_health()

        print("\n" + "=" * 70)
        print("MODEL HEALTH CHECK")
        print("=" * 70)
        for alert in alerts:
            print(f"  {alert}")
        print("=" * 70)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
