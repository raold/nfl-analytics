#!/usr/bin/env python3
"""
A/B Test Analysis Script - Ensemble v5.0 vs v5.1
=================================================

Statistical analysis of A/B test results to determine:
1. Performance improvement (MAE, RMSE, calibration)
2. Statistical significance (paired t-test, bootstrap CI)
3. Deployment decision (deploy v5.1 to 100% or rollback)

Decision Criteria:
- MAE improvement > 2% (target: 4.2% from validation)
- p-value < 0.05 (95% confidence)
- Calibration ≥ 89% (both groups)
- Minimum 100 games per group
- No significant degradation in any metric

Usage:
    # Analyze current A/B test results
    python scripts/analyze_ab_test.py

    # Generate detailed report
    python scripts/analyze_ab_test.py --detailed --output reports/ab_test_analysis.html

    # Check if ready for deployment
    python scripts/analyze_ab_test.py --check-deployment

Author: Claude Code (Sonnet 4.5)
Date: November 14, 2025
Version: 1.0
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
from scipy import stats

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ABTestAnalyzer:
    """
    Analyzer for A/B test results between ensemble v5.0 and v5.1.

    Performs statistical tests and generates deployment recommendations.
    """

    # Decision criteria
    MIN_SAMPLES = 100  # Minimum games per group
    MIN_MAE_IMPROVEMENT = 0.02  # 2% improvement threshold
    TARGET_MAE_IMPROVEMENT = 0.042  # 4.2% target from validation
    MIN_CALIBRATION = 0.89  # 89% calibration threshold
    ALPHA = 0.05  # Significance level

    def __init__(
        self,
        db_host: str = "localhost",
        db_port: int = 5544,
        db_name: str = "devdb01",
        db_user: str = "dro",
        db_password: str = "sicillionbillions",
    ):
        """Initialize analyzer with database connection."""
        self.db_config = {
            "host": db_host,
            "port": db_port,
            "dbname": db_name,
            "user": db_user,
            "password": db_password,
        }

        logger.info("A/B Test Analyzer initialized")

    def _get_db_connection(self):
        """Create database connection."""
        return psycopg2.connect(**self.db_config)

    def load_test_data(self) -> pd.DataFrame:
        """
        Load A/B test predictions from ml.predictions table.

        Returns:
            DataFrame with columns: ab_group, model_version, prediction_mean,
                                    actual_value, prediction_error, abs_prediction_error,
                                    calibrated, etc.
        """
        conn = self._get_db_connection()

        query = """
        SELECT
            m.model_name,
            CASE
                WHEN m.model_name = 'ensemble_v5.0' THEN 'A'
                WHEN m.model_name = 'ensemble_v5.1' THEN 'B'
                ELSE NULL
            END as ab_group,
            REPLACE(m.model_name, 'ensemble_', '') as model_version,
            p.player_id,
            p.player_name,
            p.season,
            p.week,
            p.team,
            p.opponent,
            p.prediction_mean,
            p.prediction_std,
            p.prediction_q10,
            p.prediction_q90,
            p.actual_value,
            p.prediction_error,
            p.abs_prediction_error,
            p.calibrated,
            p.status,
            p.predicted_at
        FROM ml.predictions p
        JOIN ml.model_registry m ON p.model_id = m.model_id
        WHERE m.model_name IN ('ensemble_v5.0', 'ensemble_v5.1')
          AND p.stat_type = 'passing_yards'
        ORDER BY p.predicted_at ASC
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        logger.info(f"Loaded {len(df)} predictions from database")
        logger.info(f"  Group A (v5.0): {(df['ab_group'] == 'A').sum()} predictions")
        logger.info(f"  Group B (v5.1): {(df['ab_group'] == 'B').sum()} predictions")
        logger.info(f"  Completed: {df['actual_value'].notna().sum()} games")

        return df

    def compute_metrics(self, df: pd.DataFrame) -> dict:
        """
        Compute performance metrics for each A/B group.

        Args:
            df: DataFrame with predictions

        Returns:
            Dict with metrics for each group
        """
        # Filter to completed predictions only
        completed = df[df["actual_value"].notna()].copy()

        if len(completed) == 0:
            logger.warning("No completed predictions found")
            return {"group_a": {}, "group_b": {}}

        metrics = {}

        for group in ["A", "B"]:
            group_df = completed[completed["ab_group"] == group]

            if len(group_df) == 0:
                metrics[f"group_{group.lower()}"] = {}
                continue

            # Point prediction metrics
            mae = group_df["abs_prediction_error"].mean()
            rmse = np.sqrt((group_df["prediction_error"] ** 2).mean())
            mape = (group_df["abs_prediction_error"] / group_df["actual_value"]).mean()

            # Calibration metrics
            calibration_90 = group_df["calibrated"].mean()
            avg_uncertainty = group_df["prediction_std"].mean()

            # Distribution metrics
            median_error = group_df["prediction_error"].median()
            q25_error = group_df["prediction_error"].quantile(0.25)
            q75_error = group_df["prediction_error"].quantile(0.75)

            metrics[f"group_{group.lower()}"] = {
                "n_predictions": len(group_df),
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "calibration_90": calibration_90,
                "avg_uncertainty": avg_uncertainty,
                "median_error": median_error,
                "q25_error": q25_error,
                "q75_error": q75_error,
            }

        return metrics

    def statistical_tests(self, df: pd.DataFrame) -> dict:
        """
        Perform statistical significance tests.

        Tests:
        1. Paired t-test on absolute errors (MAE comparison)
        2. Bootstrap confidence interval for MAE difference
        3. Welch's t-test on prediction errors

        Args:
            df: DataFrame with predictions

        Returns:
            Dict with test results
        """
        completed = df[df["actual_value"].notna()].copy()

        group_a = completed[completed["ab_group"] == "A"]
        group_b = completed[completed["ab_group"] == "B"]

        if len(group_a) < 2 or len(group_b) < 2:
            logger.warning("Insufficient data for statistical tests")
            return {}

        results = {}

        # 1. Paired t-test on absolute errors (if same games)
        # For A/B test with different assignments, use independent t-test
        errors_a = group_a["abs_prediction_error"].values
        errors_b = group_b["abs_prediction_error"].values

        # Welch's t-test (unequal variances, unequal sample sizes)
        t_stat, p_value = stats.ttest_ind(errors_a, errors_b, equal_var=False)

        results["ttest"] = {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < self.ALPHA,
            "direction": "B better" if t_stat > 0 else "A better",
        }

        # 2. Bootstrap confidence interval for MAE difference
        mae_diff_bootstrap = self._bootstrap_mae_difference(errors_a, errors_b)
        mae_diff_ci = np.percentile(mae_diff_bootstrap, [2.5, 97.5])

        results["bootstrap"] = {
            "mae_diff_mean": mae_diff_bootstrap.mean(),
            "mae_diff_ci_lower": mae_diff_ci[0],
            "mae_diff_ci_upper": mae_diff_ci[1],
            "ci_excludes_zero": mae_diff_ci[0] > 0 or mae_diff_ci[1] < 0,
        }

        # 3. Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(errors_a) - 1) * errors_a.std() ** 2 + (len(errors_b) - 1) * errors_b.std() ** 2)
            / (len(errors_a) + len(errors_b) - 2)
        )
        cohens_d = (errors_a.mean() - errors_b.mean()) / pooled_std

        results["effect_size"] = {
            "cohens_d": cohens_d,
            "interpretation": self._interpret_cohens_d(cohens_d),
        }

        # 4. Calibration comparison (binomial test)
        n_a = len(group_a)
        n_b = len(group_b)
        calibrated_a = group_a["calibrated"].sum()
        calibrated_b = group_b["calibrated"].sum()

        # Test if calibration rates are different
        # Using chi-squared test for proportions
        contingency_table = np.array(
            [[calibrated_a, n_a - calibrated_a], [calibrated_b, n_b - calibrated_b]]
        )
        chi2, p_cal, _, _ = stats.chi2_contingency(contingency_table)

        results["calibration_test"] = {
            "chi2_statistic": chi2,
            "p_value": p_cal,
            "significant_difference": p_cal < self.ALPHA,
        }

        return results

    def _bootstrap_mae_difference(
        self, errors_a: np.ndarray, errors_b: np.ndarray, n_iterations: int = 10000
    ) -> np.ndarray:
        """
        Bootstrap confidence interval for MAE difference.

        Args:
            errors_a: Absolute errors for group A
            errors_b: Absolute errors for group B
            n_iterations: Number of bootstrap iterations

        Returns:
            Array of bootstrapped MAE differences (A - B)
        """
        np.random.seed(42)
        mae_diffs = []

        for _ in range(n_iterations):
            # Resample with replacement
            sample_a = np.random.choice(errors_a, size=len(errors_a), replace=True)
            sample_b = np.random.choice(errors_b, size=len(errors_b), replace=True)

            # Compute MAE difference
            mae_diff = sample_a.mean() - sample_b.mean()
            mae_diffs.append(mae_diff)

        return np.array(mae_diffs)

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def make_deployment_decision(self, metrics: dict, statistical_tests: dict) -> dict:
        """
        Make deployment decision based on metrics and tests.

        Decision logic:
        1. Check minimum sample size (100 games per group)
        2. Check MAE improvement > 2% (target: 4.2%)
        3. Check statistical significance (p < 0.05)
        4. Check calibration ≥ 89% (both groups)
        5. Check no significant degradation in other metrics

        Args:
            metrics: Performance metrics for each group
            statistical_tests: Statistical test results

        Returns:
            Dict with deployment recommendation
        """
        decision = {
            "recommendation": "WAIT",
            "confidence": "LOW",
            "reasons": [],
            "blockers": [],
        }

        group_a = metrics["group_a"]
        group_b = metrics["group_b"]

        # Check 1: Minimum sample size
        if group_a.get("n_predictions", 0) < self.MIN_SAMPLES:
            decision["blockers"].append(
                f"Group A has {group_a.get('n_predictions', 0)} predictions (need {self.MIN_SAMPLES})"
            )

        if group_b.get("n_predictions", 0) < self.MIN_SAMPLES:
            decision["blockers"].append(
                f"Group B has {group_b.get('n_predictions', 0)} predictions (need {self.MIN_SAMPLES})"
            )

        if decision["blockers"]:
            decision["recommendation"] = "WAIT"
            decision["confidence"] = "LOW"
            return decision

        # Check 2: MAE improvement
        mae_a = group_a["mae"]
        mae_b = group_b["mae"]
        mae_improvement = (mae_a - mae_b) / mae_a

        if mae_improvement < 0:
            decision["blockers"].append(
                f"Group B (v5.1) has WORSE MAE: {mae_b:.2f} vs {mae_a:.2f} ({mae_improvement:.1%} change)"
            )
        elif mae_improvement < self.MIN_MAE_IMPROVEMENT:
            decision["blockers"].append(
                f"MAE improvement too small: {mae_improvement:.1%} < {self.MIN_MAE_IMPROVEMENT:.1%} threshold"
            )
        else:
            decision["reasons"].append(
                f"MAE improved by {mae_improvement:.1%} (target: {self.TARGET_MAE_IMPROVEMENT:.1%})"
            )

        # Check 3: Statistical significance. The t-test is two-sided, so a significant
        # result only supports deployment when the treatment is the better model.
        significant = bool(statistical_tests.get("ttest", {}).get("significant"))
        if significant and mae_improvement > 0:
            decision["reasons"].append(
                f"Statistically significant improvement (p={statistical_tests['ttest']['p_value']:.4f})"
            )
        elif significant:
            decision["blockers"].append(
                f"Group B is significantly WORSE (p={statistical_tests['ttest']['p_value']:.4f})"
            )
        else:
            decision["blockers"].append(
                f"Not statistically significant (p={statistical_tests['ttest']['p_value']:.4f} > 0.05)"
            )

        # Check 4: Calibration
        cal_a = group_a["calibration_90"]
        cal_b = group_b["calibration_90"]

        if cal_a < self.MIN_CALIBRATION:
            decision["blockers"].append(
                f"Group A calibration too low: {cal_a:.1%} < {self.MIN_CALIBRATION:.1%}"
            )

        if cal_b < self.MIN_CALIBRATION:
            decision["blockers"].append(
                f"Group B calibration too low: {cal_b:.1%} < {self.MIN_CALIBRATION:.1%}"
            )

        if cal_a >= self.MIN_CALIBRATION and cal_b >= self.MIN_CALIBRATION:
            decision["reasons"].append(
                f"Both groups well-calibrated (A: {cal_a:.1%}, B: {cal_b:.1%})"
            )

        # Check 5: RMSE degradation
        rmse_a = group_a["rmse"]
        rmse_b = group_b["rmse"]
        rmse_change = (rmse_b - rmse_a) / rmse_a

        if rmse_change > 0.05:  # 5% degradation threshold
            decision["blockers"].append(
                f"RMSE degraded by {rmse_change:.1%} (B: {rmse_b:.2f} vs A: {rmse_a:.2f})"
            )

        # Make final decision. A worse treatment is never deployed, whatever the other votes.
        if mae_improvement < 0:
            decision["recommendation"] = "ROLLBACK" if significant else "WAIT"
            decision["confidence"] = "HIGH" if significant else "LOW"
        elif len(decision["blockers"]) == 0:
            decision["recommendation"] = "DEPLOY"
            decision["confidence"] = "HIGH"
            decision["reasons"].append("All criteria met - safe to deploy v5.1 to 100%")
        elif len(decision["reasons"]) > len(decision["blockers"]):
            decision["recommendation"] = "DEPLOY_WITH_CAUTION"
            decision["confidence"] = "MEDIUM"
        else:
            decision["recommendation"] = "WAIT" if len(decision["blockers"]) < 3 else "ROLLBACK"
            decision["confidence"] = "LOW" if decision["recommendation"] == "WAIT" else "HIGH"

        return decision

    def generate_report(
        self,
        df: pd.DataFrame,
        metrics: dict,
        statistical_tests: dict,
        decision: dict,
        output_format: str = "text",
    ) -> str:
        """
        Generate analysis report.

        Args:
            df: Predictions DataFrame
            metrics: Performance metrics
            statistical_tests: Statistical test results
            decision: Deployment decision
            output_format: "text" or "html"

        Returns:
            Report string
        """
        if output_format == "html":
            return self._generate_html_report(df, metrics, statistical_tests, decision)
        else:
            return self._generate_text_report(df, metrics, statistical_tests, decision)

    def _generate_text_report(
        self, df: pd.DataFrame, metrics: dict, statistical_tests: dict, decision: dict
    ) -> str:
        """Generate plain text report."""
        lines = []
        lines.append("=" * 70)
        lines.append("A/B TEST ANALYSIS REPORT")
        lines.append("Ensemble v5.0 (Control) vs v5.1 (Treatment)")
        lines.append("=" * 70)
        lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Sample sizes
        completed = df[df["actual_value"].notna()]
        lines.append("SAMPLE SIZES")
        lines.append("-" * 70)
        lines.append(f"Total predictions:   {len(df)}")
        lines.append(f"Completed games:     {len(completed)}")
        lines.append(f"Group A (v5.0):      {metrics['group_a'].get('n_predictions', 0)} games")
        lines.append(f"Group B (v5.1):      {metrics['group_b'].get('n_predictions', 0)} games")
        lines.append("")

        # Performance metrics
        lines.append("PERFORMANCE METRICS")
        lines.append("-" * 70)
        lines.append(
            f"{'Metric':<20} {'Group A (v5.0)':>15} {'Group B (v5.1)':>15} {'Improvement':>15}"
        )
        lines.append("-" * 70)

        for metric_name, display_name in [
            ("mae", "MAE (yards)"),
            ("rmse", "RMSE (yards)"),
            ("calibration_90", "Calibration 90%"),
            ("avg_uncertainty", "Avg Uncertainty"),
        ]:
            val_a = metrics["group_a"].get(metric_name)
            val_b = metrics["group_b"].get(metric_name)

            if val_a is not None and val_b is not None:
                if metric_name == "calibration_90":
                    improvement = val_b - val_a
                    lines.append(
                        f"{display_name:<20} {val_a:>14.1%} {val_b:>14.1%} {improvement:>+14.1%}"
                    )
                else:
                    improvement = (val_a - val_b) / val_a
                    lines.append(
                        f"{display_name:<20} {val_a:>15.2f} {val_b:>15.2f} {improvement:>+14.1%}"
                    )

        lines.append("")

        # Statistical tests
        lines.append("STATISTICAL TESTS")
        lines.append("-" * 70)

        if "ttest" in statistical_tests:
            ttest = statistical_tests["ttest"]
            lines.append(
                f"Welch's t-test:      t={ttest['t_statistic']:>6.3f}, p={ttest['p_value']:.4f}"
            )
            lines.append(f"Significant:         {ttest['significant']} ({ttest['direction']})")

        if "bootstrap" in statistical_tests:
            boot = statistical_tests["bootstrap"]
            lines.append(
                f"Bootstrap 95% CI:    [{boot['mae_diff_ci_lower']:.2f}, {boot['mae_diff_ci_upper']:.2f}]"
            )
            lines.append(f"CI excludes zero:    {boot['ci_excludes_zero']}")

        if "effect_size" in statistical_tests:
            effect = statistical_tests["effect_size"]
            lines.append(
                f"Cohen's d:           {effect['cohens_d']:.3f} ({effect['interpretation']})"
            )

        lines.append("")

        # Decision
        lines.append("DEPLOYMENT DECISION")
        lines.append("=" * 70)
        lines.append(f"Recommendation:      {decision['recommendation']}")
        lines.append(f"Confidence:          {decision['confidence']}")
        lines.append("")

        if decision["reasons"]:
            lines.append("Reasons:")
            for reason in decision["reasons"]:
                lines.append(f"  ✓ {reason}")
            lines.append("")

        if decision["blockers"]:
            lines.append("Blockers:")
            for blocker in decision["blockers"]:
                lines.append(f"  ✗ {blocker}")
            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

    def _generate_html_report(
        self, df: pd.DataFrame, metrics: dict, statistical_tests: dict, decision: dict
    ) -> str:
        """Generate HTML report (simplified version)."""
        # For brevity, using same text format wrapped in <pre>
        text_report = self._generate_text_report(df, metrics, statistical_tests, decision)
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>A/B Test Analysis Report</title>
    <style>
        body {{ font-family: monospace; padding: 20px; }}
        pre {{ background: #f4f4f4; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <pre>{text_report}</pre>
</body>
</html>
"""
        return html

    def run_analysis(self, output_file: str | None = None, output_format: str = "text") -> dict:
        """
        Run complete A/B test analysis.

        Args:
            output_file: Optional file to save report
            output_format: "text" or "html"

        Returns:
            Dict with all analysis results
        """
        logger.info("\n" + "=" * 70)
        logger.info("Running A/B Test Analysis")
        logger.info("=" * 70 + "\n")

        # Load data
        df = self.load_test_data()

        # Compute metrics
        metrics = self.compute_metrics(df)

        # Statistical tests
        completed = df[df["actual_value"].notna()]
        if len(completed) >= 2:
            statistical_tests = self.statistical_tests(df)
        else:
            statistical_tests = {}
            logger.warning("Not enough completed predictions for statistical tests")

        # Deployment decision
        decision = self.make_deployment_decision(metrics, statistical_tests)

        # Generate report
        report = self.generate_report(df, metrics, statistical_tests, decision, output_format)

        # Print report
        print("\n" + report)

        # Save to file
        if output_file:
            with open(output_file, "w") as f:
                f.write(report)
            logger.info(f"\n✅ Report saved to: {output_file}")

        return {
            "data": df,
            "metrics": metrics,
            "statistical_tests": statistical_tests,
            "decision": decision,
            "report": report,
        }


def main():
    """Command-line interface for A/B test analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze A/B test results for ensemble v5.0 vs v5.1"
    )
    parser.add_argument(
        "--output", help="Output file for report (e.g., reports/ab_test_analysis.txt)"
    )
    parser.add_argument(
        "--format", choices=["text", "html"], default="text", help="Report format (default: text)"
    )
    parser.add_argument(
        "--check-deployment",
        action="store_true",
        help="Check if ready for deployment (exit code 0 if ready, 1 otherwise)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run analysis
    analyzer = ABTestAnalyzer()
    results = analyzer.run_analysis(output_file=args.output, output_format=args.format)

    # Check deployment readiness
    if args.check_deployment:
        decision = results["decision"]
        if decision["recommendation"] == "DEPLOY":
            logger.info("\n✅ READY FOR DEPLOYMENT")
            sys.exit(0)
        else:
            logger.warning(f"\n⚠️  NOT READY: {decision['recommendation']}")
            sys.exit(1)


if __name__ == "__main__":
    main()
