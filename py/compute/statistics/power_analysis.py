#!/usr/bin/env python3
"""
Statistical Power Analysis for NFL Analytics.

Implements power calculations, sample size determination, and optimal
stopping rules for sports betting analytics experiments.
"""

import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import scipy.stats as stats
from scipy import optimize

logger = logging.getLogger(__name__)


class TestType(Enum):
    """Types of statistical tests for power analysis."""

    ONE_SAMPLE_T = "one_sample_t"
    TWO_SAMPLE_T = "two_sample_t"
    PAIRED_T = "paired_t"
    PROPORTION_ONE = "proportion_one"
    PROPORTION_TWO = "proportion_two"
    CORRELATION = "correlation"
    ANOVA = "anova"
    MANN_WHITNEY = "mann_whitney"


class EffectSizeConvention(Enum):
    """Cohen's conventions for effect sizes."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass
class PowerResult:
    """Results of power analysis calculations."""

    test_type: str
    effect_size: float
    alpha: float
    power: float
    sample_size: int | None = None
    critical_value: float | None = None
    interpretation: str = ""

    def __str__(self) -> str:
        """String representation of power analysis."""
        result = f"Power Analysis: {self.test_type}\n"
        result += f"Effect size: {self.effect_size:.3f}\n"
        result += f"Alpha: {self.alpha:.3f}\n"
        result += f"Power: {self.power:.3f}\n"

        if self.sample_size:
            result += f"Sample size: {self.sample_size}\n"

        if self.critical_value:
            result += f"Critical value: {self.critical_value:.3f}\n"

        result += f"Interpretation: {self.interpretation}"
        return result


@dataclass
class SequentialDesign:
    """Sequential testing design parameters."""

    max_n: int
    alpha: float
    beta: float
    effect_size: float
    alpha_spending_function: str = "obrien_fleming"
    interim_analyses: list[int] | None = None


class PowerAnalyzer:
    """
    Comprehensive statistical power analysis tool.

    Provides power calculations, sample size determination, and
    sequential testing designs for various statistical tests.
    """

    def __init__(self):
        """Initialize power analyzer."""
        # Cohen's effect size conventions
        self.effect_size_conventions = {
            TestType.ONE_SAMPLE_T: {"small": 0.2, "medium": 0.5, "large": 0.8},
            TestType.TWO_SAMPLE_T: {"small": 0.2, "medium": 0.5, "large": 0.8},
            TestType.PAIRED_T: {"small": 0.2, "medium": 0.5, "large": 0.8},
            TestType.CORRELATION: {"small": 0.1, "medium": 0.3, "large": 0.5},
            TestType.ANOVA: {"small": 0.1, "medium": 0.25, "large": 0.4},
        }

    def power_t_test(
        self,
        effect_size: float,
        n: int,
        alpha: float = 0.05,
        test_type: TestType = TestType.TWO_SAMPLE_T,
        alternative: str = "two-sided",
    ) -> PowerResult:
        """
        Calculate statistical power for t-tests.

        Args:
            effect_size: Cohen's d
            n: Sample size (per group for two-sample)
            alpha: Type I error rate
            test_type: Type of t-test
            alternative: Alternative hypothesis ("two-sided", "greater", "less")

        Returns:
            PowerResult with power calculation
        """
        if alternative == "two-sided":
            alpha_tail = alpha / 2
        else:
            alpha_tail = alpha

        if test_type == TestType.ONE_SAMPLE_T:
            # One-sample t-test
            df = n - 1
            critical_t = stats.t.ppf(1 - alpha_tail, df)
            ncp = effect_size * np.sqrt(n)  # Non-centrality parameter

            if alternative == "two-sided":
                power = 1 - stats.nct.cdf(critical_t, df, ncp) + stats.nct.cdf(-critical_t, df, ncp)
            elif alternative == "greater":
                power = 1 - stats.nct.cdf(critical_t, df, ncp)
            else:  # less
                power = stats.nct.cdf(-critical_t, df, ncp)

        elif test_type == TestType.TWO_SAMPLE_T:
            # Two-sample t-test (equal sample sizes)
            df = 2 * n - 2
            critical_t = stats.t.ppf(1 - alpha_tail, df)
            ncp = effect_size * np.sqrt(n / 2)

            if alternative == "two-sided":
                power = 1 - stats.nct.cdf(critical_t, df, ncp) + stats.nct.cdf(-critical_t, df, ncp)
            elif alternative == "greater":
                power = 1 - stats.nct.cdf(critical_t, df, ncp)
            else:  # less
                power = stats.nct.cdf(-critical_t, df, ncp)

        elif test_type == TestType.PAIRED_T:
            # Paired t-test
            df = n - 1
            critical_t = stats.t.ppf(1 - alpha_tail, df)
            ncp = effect_size * np.sqrt(n)

            if alternative == "two-sided":
                power = 1 - stats.nct.cdf(critical_t, df, ncp) + stats.nct.cdf(-critical_t, df, ncp)
            elif alternative == "greater":
                power = 1 - stats.nct.cdf(critical_t, df, ncp)
            else:  # less
                power = stats.nct.cdf(-critical_t, df, ncp)

        else:
            raise ValueError(f"Unsupported test type: {test_type}")

        interpretation = self._interpret_power(power, effect_size, test_type)

        return PowerResult(
            test_type=test_type.value,
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            sample_size=n,
            critical_value=critical_t,
            interpretation=interpretation,
        )

    def sample_size_t_test(
        self,
        effect_size: float,
        power: float = 0.8,
        alpha: float = 0.05,
        test_type: TestType = TestType.TWO_SAMPLE_T,
        alternative: str = "two-sided",
    ) -> PowerResult:
        """
        Calculate required sample size for desired power in t-tests.

        Args:
            effect_size: Cohen's d
            power: Desired statistical power
            alpha: Type I error rate
            test_type: Type of t-test
            alternative: Alternative hypothesis

        Returns:
            PowerResult with required sample size
        """

        def power_func(n):
            result = self.power_t_test(effect_size, int(n), alpha, test_type, alternative)
            return result.power - power

        # Find sample size using root finding
        try:
            # Start with reasonable bounds
            n_min = 2
            n_max = 10000

            # Check if solution exists in range
            if power_func(n_max) < 0:
                warnings.warn("Required sample size may be very large (>10000)")
                n_max = 100000

            n_optimal = optimize.brentq(power_func, n_min, n_max)
            n_required = int(np.ceil(n_optimal))

        except ValueError:
            # If no solution found, use approximation
            n_required = self._approximate_sample_size_t_test(
                effect_size, power, alpha, test_type, alternative
            )

        # Verify the power with calculated sample size
        result = self.power_t_test(effect_size, n_required, alpha, test_type, alternative)

        interpretation = (
            f"For {test_type.value} with effect size {effect_size:.3f}, "
            f"need n={n_required} per group to achieve {power:.1%} power "
            f"at α={alpha:.3f}"
        )

        result.interpretation = interpretation
        return result

    def power_proportion_test(
        self,
        p1: float,
        p2: float | None = None,
        n: int = None,
        alpha: float = 0.05,
        alternative: str = "two-sided",
    ) -> PowerResult:
        """
        Calculate power for proportion tests.

        Args:
            p1: Proportion in group 1 (or observed proportion for one-sample)
            p2: Proportion in group 2 (None for one-sample test)
            n: Sample size (per group for two-sample)
            alpha: Type I error rate
            alternative: Alternative hypothesis

        Returns:
            PowerResult with power calculation
        """
        if p2 is None:
            # One-sample proportion test
            test_type = "one_sample_proportion"
            p0 = 0.5  # Default null hypothesis proportion

            # Effect size (Cohen's h)
            effect_size = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p0)))

            # Standard error under null
            se_null = np.sqrt(p0 * (1 - p0) / n)

            # Critical value
            if alternative == "two-sided":
                z_critical = stats.norm.ppf(1 - alpha / 2)
            else:
                z_critical = stats.norm.ppf(1 - alpha)

            # Power calculation
            se_alt = np.sqrt(p1 * (1 - p1) / n)
            z_power = (z_critical * se_null - abs(p1 - p0)) / se_alt

            if alternative == "two-sided":
                power = 1 - stats.norm.cdf(z_power) + stats.norm.cdf(-z_power)
            else:
                power = 1 - stats.norm.cdf(z_power)

        else:
            # Two-sample proportion test
            test_type = "two_sample_proportion"

            # Effect size (Cohen's h)
            effect_size = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

            # Pooled proportion
            p_pooled = (p1 + p2) / 2

            # Standard errors
            se_null = np.sqrt(2 * p_pooled * (1 - p_pooled) / n)
            se_alt = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)

            # Critical value
            if alternative == "two-sided":
                z_critical = stats.norm.ppf(1 - alpha / 2)
            else:
                z_critical = stats.norm.ppf(1 - alpha)

            # Power calculation
            z_power = (z_critical * se_null - abs(p1 - p2)) / se_alt

            if alternative == "two-sided":
                power = 1 - stats.norm.cdf(z_power) + stats.norm.cdf(-z_power)
            else:
                power = 1 - stats.norm.cdf(z_power)

        power = max(0, min(1, power))  # Bound between 0 and 1

        interpretation = (
            f"For {test_type} comparing {p1:.3f} vs {p2 if p2 else 0.5:.3f}, "
            f"power is {power:.3f} with n={n}"
        )

        return PowerResult(
            test_type=test_type,
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            sample_size=n,
            interpretation=interpretation,
        )

    def power_correlation(
        self, r: float, n: int, alpha: float = 0.05, alternative: str = "two-sided"
    ) -> PowerResult:
        """
        Calculate power for correlation test.

        Args:
            r: True correlation coefficient
            n: Sample size
            alpha: Type I error rate
            alternative: Alternative hypothesis

        Returns:
            PowerResult with power calculation
        """
        # Fisher z-transformation
        z_r = 0.5 * np.log((1 + r) / (1 - r))
        se = 1 / np.sqrt(n - 3)

        # Critical value
        if alternative == "two-sided":
            z_critical = stats.norm.ppf(1 - alpha / 2)
        else:
            z_critical = stats.norm.ppf(1 - alpha)

        # Power calculation
        z_power = (abs(z_r) - z_critical * se) / se

        if alternative == "two-sided":
            power = 1 - stats.norm.cdf(z_power) + stats.norm.cdf(-z_power)
        else:
            power = 1 - stats.norm.cdf(z_power)

        power = max(0, min(1, power))

        interpretation = (
            f"For correlation r={r:.3f} with n={n}, " f"power is {power:.3f} at α={alpha:.3f}"
        )

        return PowerResult(
            test_type="correlation",
            effect_size=r,
            alpha=alpha,
            power=power,
            sample_size=n,
            interpretation=interpretation,
        )

    def sequential_design(
        self,
        effect_size: float,
        alpha: float = 0.05,
        beta: float = 0.2,
        max_n: int = 1000,
        n_interim: int = 4,
        spending_function: str = "obrien_fleming",
    ) -> SequentialDesign:
        """
        Design sequential testing procedure with interim analyses.

        Args:
            effect_size: Expected effect size
            alpha: Type I error rate
            beta: Type II error rate (1 - power)
            max_n: Maximum sample size
            n_interim: Number of interim analyses
            spending_function: Alpha spending function

        Returns:
            SequentialDesign with boundaries and stopping rules
        """
        # Information fractions (equally spaced interim analyses)
        info_fractions = np.linspace(1 / n_interim, 1, n_interim)
        interim_n = [int(frac * max_n) for frac in info_fractions]

        design = SequentialDesign(
            max_n=max_n,
            alpha=alpha,
            beta=beta,
            effect_size=effect_size,
            alpha_spending_function=spending_function,
            interim_analyses=interim_n,
        )

        return design

    def optimal_stopping_power(
        self, effect_sizes: list[float], n_max: int, alpha: float = 0.05, check_frequency: int = 50
    ) -> dict[str, Any]:
        """
        Calculate optimal stopping rules for power optimization.

        Args:
            effect_sizes: Range of possible effect sizes
            n_max: Maximum sample size
            alpha: Type I error rate
            check_frequency: How often to check for stopping

        Returns:
            Dictionary with stopping recommendations
        """
        results = {}
        check_points = list(range(check_frequency, n_max + 1, check_frequency))

        for effect_size in effect_sizes:
            powers = []
            for n in check_points:
                power_result = self.power_t_test(effect_size, n, alpha)
                powers.append(power_result.power)

            # Find first n where power > 0.8
            adequate_power_idx = next((i for i, p in enumerate(powers) if p >= 0.8), None)

            if adequate_power_idx is not None:
                optimal_n = check_points[adequate_power_idx]
                optimal_power = powers[adequate_power_idx]
            else:
                optimal_n = n_max
                optimal_power = powers[-1]

            results[f"effect_{effect_size:.2f}"] = {
                "optimal_n": optimal_n,
                "power": optimal_power,
                "all_n": check_points,
                "all_powers": powers,
            }

        return results

    def _approximate_sample_size_t_test(
        self, effect_size: float, power: float, alpha: float, test_type: TestType, alternative: str
    ) -> int:
        """
        Approximate sample size calculation for t-tests.

        Uses Cohen's formulas as starting point.
        """
        if alternative == "two-sided":
            z_alpha = stats.norm.ppf(1 - alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)

        z_beta = stats.norm.ppf(power)

        if test_type == TestType.ONE_SAMPLE_T:
            n = ((z_alpha + z_beta) / effect_size) ** 2
        elif test_type == TestType.TWO_SAMPLE_T:
            n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        elif test_type == TestType.PAIRED_T:
            n = ((z_alpha + z_beta) / effect_size) ** 2
        else:
            n = 100  # Default fallback

        return int(np.ceil(n))

    def _interpret_power(self, power: float, effect_size: float, test_type: TestType) -> str:
        """Generate interpretation of power analysis results."""
        if power < 0.5:
            power_desc = "inadequate"
        elif power < 0.8:
            power_desc = "moderate"
        elif power < 0.95:
            power_desc = "good"
        else:
            power_desc = "excellent"

        # Effect size interpretation
        conventions = self.effect_size_conventions.get(test_type, {})
        if effect_size <= conventions.get("small", 0.2):
            effect_desc = "small"
        elif effect_size <= conventions.get("medium", 0.5):
            effect_desc = "medium"
        elif effect_size <= conventions.get("large", 0.8):
            effect_desc = "large"
        else:
            effect_desc = "very large"

        return (
            f"Power of {power:.1%} is {power_desc} for detecting "
            f"a {effect_desc} effect (d={effect_size:.3f})"
        )

    def get_effect_size_convention(
        self, test_type: TestType, magnitude: EffectSizeConvention
    ) -> float:
        """
        Get conventional effect size for test type.

        Args:
            test_type: Type of statistical test
            magnitude: Effect size magnitude (small, medium, large)

        Returns:
            Conventional effect size value
        """
        conventions = self.effect_size_conventions.get(test_type, {})
        return conventions.get(magnitude.value, 0.5)


# Convenience functions
def sample_size_calculation(
    effect_size: float, power: float = 0.8, alpha: float = 0.05, test_type: str = "two_sample_t"
) -> int:
    """
    Calculate required sample size for given parameters.

    Args:
        effect_size: Expected effect size
        power: Desired statistical power
        alpha: Type I error rate
        test_type: Type of test

    Returns:
        Required sample size
    """
    analyzer = PowerAnalyzer()
    test_enum = TestType(test_type)
    result = analyzer.sample_size_t_test(effect_size, power, alpha, test_enum)
    return result.sample_size


def power_calculation(
    effect_size: float, n: int, alpha: float = 0.05, test_type: str = "two_sample_t"
) -> float:
    """
    Calculate statistical power for given parameters.

    Args:
        effect_size: Effect size
        n: Sample size
        alpha: Type I error rate
        test_type: Type of test

    Returns:
        Statistical power
    """
    analyzer = PowerAnalyzer()
    test_enum = TestType(test_type)
    result = analyzer.power_t_test(effect_size, n, alpha, test_enum)
    return result.power


def test():
    """Test the power analysis module."""
    print("=== Statistical Power Analysis Demo ===\n")

    analyzer = PowerAnalyzer()

    # Test 1: Power calculation for t-test
    print("1. Power Calculation:")
    result = analyzer.power_t_test(
        effect_size=0.5, n=30, alpha=0.05, test_type=TestType.TWO_SAMPLE_T  # Medium effect
    )
    print(result)
    print()

    # Test 2: Sample size calculation
    print("2. Sample Size Calculation:")
    result = analyzer.sample_size_t_test(
        effect_size=0.5, power=0.8, alpha=0.05, test_type=TestType.TWO_SAMPLE_T
    )
    print(result)
    print()

    # Test 3: Power for proportion test
    print("3. Proportion Test Power:")
    result = analyzer.power_proportion_test(
        p1=0.6, p2=0.4, n=50, alpha=0.05  # 60% success rate  # 40% success rate
    )
    print(result)
    print()

    # Test 4: Correlation power
    print("4. Correlation Test Power:")
    result = analyzer.power_correlation(r=0.3, n=100, alpha=0.05)  # Moderate correlation
    print(result)
    print()

    # Test 5: Optimal stopping
    print("5. Optimal Stopping Analysis:")
    stopping_results = analyzer.optimal_stopping_power(
        effect_sizes=[0.2, 0.5, 0.8], n_max=200, alpha=0.05, check_frequency=20
    )

    for effect, results in stopping_results.items():
        print(f"{effect}: Optimal n = {results['optimal_n']}, " f"Power = {results['power']:.3f}")
    print()

    # Test 6: Effect size conventions
    print("6. Effect Size Conventions:")
    for test_type in [TestType.TWO_SAMPLE_T, TestType.CORRELATION]:
        for magnitude in [
            EffectSizeConvention.SMALL,
            EffectSizeConvention.MEDIUM,
            EffectSizeConvention.LARGE,
        ]:
            effect = analyzer.get_effect_size_convention(test_type, magnitude)
            print(f"{test_type.value} - {magnitude.value}: {effect}")


if __name__ == "__main__":
    test()
