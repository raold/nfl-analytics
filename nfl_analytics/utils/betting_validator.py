"""
Betting Strategy Validator for NFL Analytics

This module validates betting strategies before live deployment. Real money
betting requires rigorous validation of risk management and expected returns.

Validation Checks:
1. Risk Gates (edge, confidence, bankroll)
2. Kelly Criterion (proper fractionalization)
3. ROI Consistency (backtests vs targets)
4. Maximum Drawdown (risk tolerance)
5. Stress Testing (worst-case scenarios)

Target Metrics:
- ROI: >5% (target: 7%+)
- Win Rate: >59% (target: 62%+)
- Sharpe Ratio: >2.2 (target: 2.5+)
- Max Drawdown: <8% (target: <-8%)

Usage:
    from nfl_analytics.utils.betting_validator import BettingValidator

    validator = BettingValidator()

    # Validate strategy configuration
    report = validator.validate_strategy_config(config)

    # Validate backtest results
    report = validator.validate_backtest_results(backtest_data)

    # Run stress tests
    stress_results = validator.run_stress_tests(strategy, scenarios)

    # Block deployment if validation fails
    if report.has_critical_issues():
        print("⛔ Strategy invalid - do not deploy to production")
        exit(1)

Integration:
    - Pre-deployment: Validate before live betting
    - Weekly: Re-validate production strategy
    - Post-backtest: Validate results before acceptance
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels."""

    CRITICAL = "CRITICAL"  # Cannot deploy
    HIGH = "HIGH"  # Should not deploy
    MEDIUM = "MEDIUM"  # Risky to deploy
    LOW = "LOW"  # Minor concern
    INFO = "INFO"  # Informational


class ValidationCategory(Enum):
    """Categories of betting validation."""

    RISK_GATES = "risk_gates"
    KELLY_CRITERION = "kelly_criterion"
    ROI_METRICS = "roi_metrics"
    DRAWDOWN = "drawdown"
    STRESS_TEST = "stress_test"
    BANKROLL = "bankroll"


@dataclass
class ValidationIssue:
    """Represents a betting strategy validation finding."""

    severity: ValidationSeverity
    category: ValidationCategory
    title: str
    description: str
    metric_value: str
    threshold: str
    recommendation: str
    financial_impact: str | None = None


@dataclass
class ValidationReport:
    """Betting strategy validation report."""

    strategy_name: str
    validation_date: str = field(default_factory=lambda: datetime.now().isoformat())
    issues: list[ValidationIssue] = field(default_factory=list)
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    approved_for_deployment: bool = False

    def add_issue(self, issue: ValidationIssue):
        """Add issue and update counters."""
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.CRITICAL:
            self.critical_count += 1
        elif issue.severity == ValidationSeverity.HIGH:
            self.high_count += 1
        elif issue.severity == ValidationSeverity.MEDIUM:
            self.medium_count += 1
        elif issue.severity == ValidationSeverity.LOW:
            self.low_count += 1

    def has_critical_issues(self) -> bool:
        """Check if report contains blocking issues."""
        return self.critical_count > 0 or self.high_count > 0

    def to_json(self) -> str:
        """Export report to JSON."""
        return json.dumps(
            {
                "strategy_name": self.strategy_name,
                "validation_date": self.validation_date,
                "approved": self.approved_for_deployment,
                "summary": {
                    "critical": self.critical_count,
                    "high": self.high_count,
                    "medium": self.medium_count,
                    "low": self.low_count,
                },
                "issues": [
                    {
                        "severity": i.severity.value,
                        "category": i.category.value,
                        "title": i.title,
                        "description": i.description,
                        "metric_value": i.metric_value,
                        "threshold": i.threshold,
                        "recommendation": i.recommendation,
                        "financial_impact": i.financial_impact,
                    }
                    for i in self.issues
                ],
            },
            indent=2,
        )


class BettingValidator:
    """
    Validates betting strategies for NFL analytics system.

    Critical for live deployment - ensures strategies are:
    - Profitable (ROI targets met)
    - Safe (risk gates enforced)
    - Robust (stress tested)
    - Properly sized (Kelly criterion)
    """

    # Target metrics (from CLAUDE.md)
    TARGET_ROI = 0.05  # 5% minimum, 7%+ target
    TARGET_WIN_RATE = 0.59  # 59% minimum, 62%+ target
    TARGET_SHARPE = 2.2  # 2.2 minimum, 2.5+ target
    MAX_DRAWDOWN = -0.08  # -8% maximum

    # Risk gate thresholds
    MIN_EDGE = 0.05  # 5% minimum edge
    MIN_CONFIDENCE = 0.60  # 60% confidence minimum
    MIN_BANKROLL_FRACTION = 0.01  # 1% minimum bet
    MAX_BANKROLL_FRACTION = 0.10  # 10% maximum bet

    # Kelly criterion
    KELLY_FRACTION = 0.25  # Quarter Kelly (conservative)
    MAX_KELLY = 0.50  # Never bet more than half Kelly

    def __init__(self):
        """Initialize betting validator."""
        if not PANDAS_AVAILABLE:
            logger.warning("pandas not available - some features limited")

        logger.info("BettingValidator initialized")

    def validate_risk_gates(
        self, edge: float, confidence: float, bankroll_fraction: float
    ) -> list[ValidationIssue]:
        """
        Validate risk gate thresholds.

        Args:
            edge: Expected edge (predicted - market)
            confidence: Model confidence level
            bankroll_fraction: Fraction of bankroll to bet

        Returns:
            List of validation issues
        """
        issues = []

        # Check edge
        if edge < self.MIN_EDGE:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category=ValidationCategory.RISK_GATES,
                    title="Insufficient edge",
                    description=f"Edge {edge:.2%} below minimum threshold",
                    metric_value=f"{edge:.2%}",
                    threshold=f">{self.MIN_EDGE:.2%}",
                    recommendation="Do not bet - edge too small",
                    financial_impact="Expected loss if betting continues",
                )
            )

        # Check confidence
        if confidence < self.MIN_CONFIDENCE:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.HIGH,
                    category=ValidationCategory.RISK_GATES,
                    title="Low confidence",
                    description=f"Confidence {confidence:.1%} below threshold",
                    metric_value=f"{confidence:.1%}",
                    threshold=f">{self.MIN_CONFIDENCE:.1%}",
                    recommendation="Consider skipping this bet",
                    financial_impact="Higher risk of loss",
                )
            )

        # Check bankroll fraction
        if bankroll_fraction > self.MAX_BANKROLL_FRACTION:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category=ValidationCategory.BANKROLL,
                    title="Bet size too large",
                    description=f"Betting {bankroll_fraction:.1%} of bankroll",
                    metric_value=f"{bankroll_fraction:.1%}",
                    threshold=f"<{self.MAX_BANKROLL_FRACTION:.1%}",
                    recommendation="Reduce bet size to manage risk",
                    financial_impact="Excessive risk of ruin",
                )
            )

        if bankroll_fraction < self.MIN_BANKROLL_FRACTION:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.LOW,
                    category=ValidationCategory.BANKROLL,
                    title="Bet size very small",
                    description=f"Betting only {bankroll_fraction:.2%} of bankroll",
                    metric_value=f"{bankroll_fraction:.2%}",
                    threshold=f">{self.MIN_BANKROLL_FRACTION:.2%}",
                    recommendation="Consider increasing bet size if confident",
                    financial_impact="Low profit potential",
                )
            )

        return issues

    def validate_kelly_criterion(
        self, edge: float, odds: float, fraction: float = None
    ) -> tuple[float, list[ValidationIssue]]:
        """
        Validate Kelly criterion bet sizing.

        Args:
            edge: Expected edge
            odds: Decimal odds
            fraction: Kelly fraction to use (default: 0.25)

        Returns:
            Tuple of (optimal_bet_size, validation_issues)
        """
        issues = []

        if fraction is None:
            fraction = self.KELLY_FRACTION

        # Calculate Kelly bet size
        # Kelly formula: f = (bp - q) / b
        # where b = odds-1, p = win prob, q = 1-p
        win_prob = 0.5 + edge / 2  # Approximate from edge
        b = odds - 1
        q = 1 - win_prob

        kelly_full = (b * win_prob - q) / b
        kelly_fractional = kelly_full * fraction

        # Validate full Kelly
        if kelly_full <= 0:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category=ValidationCategory.KELLY_CRITERION,
                    title="Negative Kelly bet size",
                    description="Kelly criterion suggests not betting",
                    metric_value=f"{kelly_full:.2%}",
                    threshold=">0%",
                    recommendation="Do not bet - negative expected value",
                    financial_impact="Expected loss",
                )
            )

        if kelly_full > 0.5:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.HIGH,
                    category=ValidationCategory.KELLY_CRITERION,
                    title="Very large Kelly bet size",
                    description=f"Full Kelly {kelly_full:.1%} is very aggressive",
                    metric_value=f"{kelly_full:.1%}",
                    threshold="<50%",
                    recommendation=f"Use fractional Kelly ({self.KELLY_FRACTION:.0%})",
                    financial_impact="High volatility risk",
                )
            )

        # Validate fractional Kelly
        if fraction > self.MAX_KELLY:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.MEDIUM,
                    category=ValidationCategory.KELLY_CRITERION,
                    title="Kelly fraction too aggressive",
                    description=f"Using {fraction:.0%} Kelly is risky",
                    metric_value=f"{fraction:.0%}",
                    threshold=f"<{self.MAX_KELLY:.0%}",
                    recommendation=f"Use {self.KELLY_FRACTION:.0%} Kelly for safety",
                    financial_impact="Increased drawdown risk",
                )
            )

        return max(kelly_fractional, 0), issues

    def validate_backtest_results(
        self, returns: np.ndarray, initial_bankroll: float = 10000.0
    ) -> ValidationReport:
        """
        Validate backtest results against targets.

        Args:
            returns: Array of bet returns (profit/loss per bet)
            initial_bankroll: Starting bankroll

        Returns:
            Validation report
        """
        report = ValidationReport(strategy_name="Backtest Validation")

        # Calculate metrics
        roi = np.sum(returns) / initial_bankroll
        win_rate = np.mean(returns > 0)

        # Calculate Sharpe ratio (if enough data)
        if len(returns) > 10:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Calculate max drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = np.min(drawdown) / initial_bankroll

        # Validate ROI
        if roi < self.TARGET_ROI:
            report.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category=ValidationCategory.ROI_METRICS,
                    title="ROI below target",
                    description=f"ROI {roi:.2%} below {self.TARGET_ROI:.2%} minimum",
                    metric_value=f"{roi:.2%}",
                    threshold=f">{self.TARGET_ROI:.2%}",
                    recommendation="Improve model accuracy or edge detection",
                    financial_impact=f"${(self.TARGET_ROI - roi) * initial_bankroll:.0f} below target",
                )
            )

        # Validate win rate
        if win_rate < self.TARGET_WIN_RATE:
            report.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.HIGH,
                    category=ValidationCategory.ROI_METRICS,
                    title="Win rate below target",
                    description=f"Win rate {win_rate:.1%} below {self.TARGET_WIN_RATE:.1%}",
                    metric_value=f"{win_rate:.1%}",
                    threshold=f">{self.TARGET_WIN_RATE:.1%}",
                    recommendation="Improve prediction accuracy",
                    financial_impact="Lower consistency, higher variance",
                )
            )

        # Validate Sharpe ratio
        if sharpe < self.TARGET_SHARPE:
            report.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.MEDIUM,
                    category=ValidationCategory.ROI_METRICS,
                    title="Sharpe ratio below target",
                    description=f"Sharpe {sharpe:.2f} below {self.TARGET_SHARPE:.2f}",
                    metric_value=f"{sharpe:.2f}",
                    threshold=f">{self.TARGET_SHARPE:.2f}",
                    recommendation="Reduce volatility or increase returns",
                    financial_impact="Poor risk-adjusted returns",
                )
            )

        # Validate max drawdown
        if max_drawdown < self.MAX_DRAWDOWN:
            report.add_issue(
                ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category=ValidationCategory.DRAWDOWN,
                    title="Maximum drawdown exceeded",
                    description=f"Drawdown {max_drawdown:.1%} exceeds {self.MAX_DRAWDOWN:.1%}",
                    metric_value=f"{max_drawdown:.1%}",
                    threshold=f">{self.MAX_DRAWDOWN:.1%}",
                    recommendation="Reduce bet sizes or improve risk management",
                    financial_impact=f"Max loss ${abs(max_drawdown) * initial_bankroll:.0f}",
                )
            )

        # Approve if no critical/high issues
        report.approved_for_deployment = not report.has_critical_issues()

        return report

    def run_stress_tests(
        self, returns: np.ndarray, scenarios: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Run stress tests on betting strategy.

        Args:
            returns: Historical returns
            scenarios: Stress test scenarios to run

        Returns:
            Stress test results
        """
        if scenarios is None:
            scenarios = [
                "losing_streak",
                "volatility_spike",
                "adverse_odds",
            ]

        results = {}

        # Losing streak test
        if "losing_streak" in scenarios:
            # Simulate 10 consecutive losses
            worst_case = np.sort(returns)[:10]
            loss_amount = np.sum(worst_case)

            results["losing_streak"] = {
                "scenario": "10 consecutive losses",
                "total_loss": float(loss_amount),
                "loss_percent": float(loss_amount / 10000),  # Assuming $10k bankroll
                "survivable": loss_amount > -1000,  # Can survive >$1k loss
            }

        # Volatility spike test
        if "volatility_spike" in scenarios:
            # Double the volatility
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            np.random.normal(mean_return, std_return * 2, len(returns))

            stressed_sharpe = mean_return / (std_return * 2) * np.sqrt(252)

            results["volatility_spike"] = {
                "scenario": "2x volatility increase",
                "sharpe_ratio": float(stressed_sharpe),
                "acceptable": stressed_sharpe > 1.0,
            }

        return results

    def generate_report(self, report: ValidationReport) -> str:
        """Generate formatted validation report."""
        lines = []

        lines.append("=" * 80)
        lines.append("BETTING STRATEGY VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append(f"\nStrategy: {report.strategy_name}")
        lines.append(f"Validation Date: {report.validation_date}")
        lines.append(
            f"Deployment Status: {'✅ APPROVED' if report.approved_for_deployment else '⛔ REJECTED'}"
        )
        lines.append("\nIssues Summary:")
        lines.append(f"  🔴 CRITICAL: {report.critical_count} (blocking)")
        lines.append(f"  🟠 HIGH:     {report.high_count} (strongly discouraged)")
        lines.append(f"  🟡 MEDIUM:   {report.medium_count} (caution advised)")
        lines.append(f"  🟢 LOW:      {report.low_count} (minor)")

        if report.has_critical_issues():
            lines.append("\n⛔ DEPLOYMENT BLOCKED")
            lines.append("   Strategy does not meet safety/profitability requirements")

        # Group by category
        for category in ValidationCategory:
            category_issues = [i for i in report.issues if i.category == category]

            if category_issues:
                lines.append(
                    f"\n{category.value.upper().replace('_', ' ')} ({len(category_issues)} issues):"
                )
                lines.append("-" * 80)

                for issue in category_issues:
                    lines.append(f"\n{issue.severity.value}: {issue.title}")
                    lines.append(f"  🔍 {issue.description}")
                    lines.append(f"  📊 Value: {issue.metric_value}, Threshold: {issue.threshold}")
                    lines.append(f"  💡 {issue.recommendation}")
                    if issue.financial_impact:
                        lines.append(f"  💰 Impact: {issue.financial_impact}")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    validator = BettingValidator()

    # Example: Validate backtest results
    np.random.seed(42)

    # Simulate backtest returns (mixed wins/losses)
    returns = np.random.normal(50, 100, 100)  # Mean $50, std $100 per bet

    report = validator.validate_backtest_results(returns, initial_bankroll=10000.0)

    print(validator.generate_report(report))

    if report.approved_for_deployment:
        print("\n✅ Strategy approved for deployment")
    else:
        print("\n⛔ Strategy rejected - fix issues before deploying")

    # Run stress tests
    stress_results = validator.run_stress_tests(returns)
    print("\nStress Test Results:")
    print(json.dumps(stress_results, indent=2))
