#!/usr/bin/env python3
"""
Effect Size Calculations for NFL Analytics.

Implements state-of-the-art effect size measures for quantifying practical
significance in sports betting analytics and computational experiments.
"""

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import scipy.stats as stats

logger = logging.getLogger(__name__)


class EffectSizeMagnitude(Enum):
    """Standard interpretation of effect size magnitudes."""

    NEGLIGIBLE = "negligible"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    VERY_LARGE = "very_large"


@dataclass
class EffectSizeResult:
    """Container for effect size calculation results."""

    name: str
    value: float
    magnitude: EffectSizeMagnitude
    confidence_interval: tuple[float, float] | None = None
    interpretation: str = ""

    def __str__(self) -> str:
        """String representation of effect size."""
        result = f"{self.name}: {self.value:.4f} ({self.magnitude.value})"
        if self.confidence_interval:
            ci_low, ci_high = self.confidence_interval
            result += f", 95% CI: ({ci_low:.4f}, {ci_high:.4f})"
        return result


class EffectSizeCalculator:
    """
    Comprehensive effect size calculator for various statistical contexts.

    Implements modern effect size measures with confidence intervals and
    interpretations based on current research standards.
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize effect size calculator.

        Args:
            confidence_level: Confidence level for bootstrap CIs
        """
        self.confidence_level = confidence_level

    def cohens_d(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        pooled: bool = True,
        bias_correction: bool = True,
    ) -> EffectSizeResult:
        """
        Calculate Cohen's d for standardized mean difference.

        Cohen's d is the most widely used measure of effect size for
        comparing means between two groups.

        Args:
            group1: First group data
            group2: Second group data
            pooled: Use pooled standard deviation (default: True)
            bias_correction: Apply Hedges' g correction for small samples

        Returns:
            EffectSizeResult with Cohen's d value and interpretation
        """
        n1, n2 = len(group1), len(group2)
        m1, m2 = np.mean(group1), np.mean(group2)
        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

        # Calculate pooled or separate standard deviation
        if pooled:
            # Pooled standard deviation
            pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
            denominator = pooled_std
        else:
            # Control group standard deviation (Glass's delta)
            denominator = s2

        # Cohen's d
        d = (m1 - m2) / denominator if denominator > 0 else 0

        # Bias correction (Hedges' g)
        if bias_correction:
            correction_factor = 1 - (3 / (4 * (n1 + n2) - 9))
            d *= correction_factor
            name = "Hedges' g"
        else:
            name = "Cohen's d"

        # Magnitude interpretation
        magnitude = self._interpret_cohens_d(abs(d))

        # Bootstrap confidence interval
        ci = self._bootstrap_cohens_d_ci(group1, group2, pooled, bias_correction)

        interpretation = (
            f"Standardized mean difference of {d:.3f} indicates {magnitude.value} "
            f"practical significance. Group 1 mean is "
            f"{'higher' if d > 0 else 'lower'} than Group 2."
        )

        return EffectSizeResult(
            name=name,
            value=d,
            magnitude=magnitude,
            confidence_interval=ci,
            interpretation=interpretation,
        )

    def cliffs_delta(self, group1: np.ndarray, group2: np.ndarray) -> EffectSizeResult:
        """
        Calculate Cliff's delta for non-parametric effect size.

        Cliff's delta measures the degree of overlap between two distributions
        and is robust to outliers and non-normal distributions.

        Args:
            group1: First group data
            group2: Second group data

        Returns:
            EffectSizeResult with Cliff's delta value and interpretation
        """
        n1, n2 = len(group1), len(group2)

        # Count dominances
        dominance_matrix = group1[:, np.newaxis] > group2
        more = np.sum(dominance_matrix)

        # Count reverse dominances
        less = np.sum(group1[:, np.newaxis] < group2)

        # Cliff's delta
        delta = (more - less) / (n1 * n2)

        # Magnitude interpretation
        magnitude = self._interpret_cliffs_delta(abs(delta))

        # Bootstrap confidence interval
        ci = self._bootstrap_cliffs_delta_ci(group1, group2)

        interpretation = (
            f"Non-parametric effect size of {delta:.3f} indicates {magnitude.value} "
            f"effect. Values from Group 1 are {'more likely to be higher' if delta > 0 else 'more likely to be lower'} "
            f"than values from Group 2 in {abs(delta)*100:.1f}% of comparisons."
        )

        return EffectSizeResult(
            name="Cliff's delta",
            value=delta,
            magnitude=magnitude,
            confidence_interval=ci,
            interpretation=interpretation,
        )

    def eta_squared(self, groups: list[np.ndarray], partial: bool = False) -> EffectSizeResult:
        """
        Calculate eta-squared (η²) for ANOVA effect size.

        Eta-squared represents the proportion of variance explained
        by the factor in ANOVA designs.

        Args:
            groups: List of group arrays
            partial: Calculate partial eta-squared

        Returns:
            EffectSizeResult with eta-squared value and interpretation
        """
        # Combine all data
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        len(all_data)

        # Group information
        group_means = [np.mean(group) for group in groups]
        group_sizes = [len(group) for group in groups]

        # Sum of squares between groups
        ss_between = sum(n * (mean - grand_mean) ** 2 for n, mean in zip(group_sizes, group_means))

        # Sum of squares within groups
        ss_within = sum(np.sum((group - np.mean(group)) ** 2) for group in groups)

        # Total sum of squares
        ss_total = ss_between + ss_within

        if partial:
            # Partial eta-squared
            eta_sq = ss_between / (ss_between + ss_within)
            name = "Partial η²"
        else:
            # Classical eta-squared
            eta_sq = ss_between / ss_total
            name = "η²"

        # Magnitude interpretation
        magnitude = self._interpret_eta_squared(eta_sq)

        interpretation = (
            f"{name} of {eta_sq:.3f} indicates that {eta_sq*100:.1f}% of variance "
            f"is explained by the grouping factor, representing {magnitude.value} effect."
        )

        return EffectSizeResult(
            name=name, value=eta_sq, magnitude=magnitude, interpretation=interpretation
        )

    def odds_ratio(
        self,
        group1_success: int,
        group1_total: int,
        group2_success: int,
        group2_total: int,
        confidence_interval: bool = True,
    ) -> EffectSizeResult:
        """
        Calculate odds ratio for binary outcomes.

        Particularly relevant for sports betting where outcomes are
        often binary (win/loss, over/under, etc.).

        Args:
            group1_success: Number of successes in group 1
            group1_total: Total observations in group 1
            group2_success: Number of successes in group 2
            group2_total: Total observations in group 2
            confidence_interval: Calculate CI using log transformation

        Returns:
            EffectSizeResult with odds ratio and interpretation
        """
        # Calculate proportions
        p1 = group1_success / group1_total
        p2 = group2_success / group2_total

        # Calculate odds
        odds1 = p1 / (1 - p1) if p1 < 1 else np.inf
        odds2 = p2 / (1 - p2) if p2 < 1 else np.inf

        # Odds ratio
        if odds2 == 0:
            or_value = np.inf
        elif np.isinf(odds1) or np.isinf(odds2):
            or_value = np.inf
        else:
            or_value = odds1 / odds2

        # Magnitude interpretation
        magnitude = self._interpret_odds_ratio(or_value)

        # Confidence interval using log transformation
        ci = None
        if confidence_interval and not np.isinf(or_value) and or_value > 0:
            # Standard error of log(OR)
            se_log_or = (
                np.sqrt(
                    1 / group1_success
                    + 1 / (group1_total - group1_success)
                    + 1 / group2_success
                    + 1 / (group2_total - group2_success)
                )
                if all(
                    x > 0
                    for x in [
                        group1_success,
                        group1_total - group1_success,
                        group2_success,
                        group2_total - group2_success,
                    ]
                )
                else None
            )

            if se_log_or is not None:
                z_score = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
                log_or = np.log(or_value)

                log_ci_lower = log_or - z_score * se_log_or
                log_ci_upper = log_or + z_score * se_log_or

                ci = (np.exp(log_ci_lower), np.exp(log_ci_upper))

        direction = "more likely" if or_value > 1 else "less likely"
        fold_change = max(or_value, 1 / or_value) if or_value > 0 else np.inf

        interpretation = (
            f"Odds ratio of {or_value:.3f} indicates {magnitude.value} effect. "
            f"Group 1 is {direction} to have success than Group 2 "
            f"by a factor of {fold_change:.2f}."
        )

        return EffectSizeResult(
            name="Odds Ratio",
            value=or_value,
            magnitude=magnitude,
            confidence_interval=ci,
            interpretation=interpretation,
        )

    def correlation_effect_size(
        self, x: np.ndarray, y: np.ndarray, method: str = "pearson"
    ) -> EffectSizeResult:
        """
        Calculate correlation coefficient as effect size.

        Args:
            x: First variable
            y: Second variable
            method: Correlation method ("pearson", "spearman", "kendall")

        Returns:
            EffectSizeResult with correlation coefficient
        """
        if method == "pearson":
            corr, p_value = stats.pearsonr(x, y)
            name = "Pearson r"
        elif method == "spearman":
            corr, p_value = stats.spearmanr(x, y)
            name = "Spearman ρ"
        elif method == "kendall":
            corr, p_value = stats.kendalltau(x, y)
            name = "Kendall τ"
        else:
            raise ValueError("Method must be 'pearson', 'spearman', or 'kendall'")

        # Magnitude interpretation
        magnitude = self._interpret_correlation(abs(corr))

        # Fisher z-transformation confidence interval for Pearson
        ci = None
        if method == "pearson" and len(x) > 3:
            n = len(x)
            z_score = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)

            # Fisher z-transformation
            fisher_z = 0.5 * np.log((1 + corr) / (1 - corr))
            se_fisher = 1 / np.sqrt(n - 3)

            fisher_ci_lower = fisher_z - z_score * se_fisher
            fisher_ci_upper = fisher_z + z_score * se_fisher

            # Transform back to correlation scale
            ci_lower = (np.exp(2 * fisher_ci_lower) - 1) / (np.exp(2 * fisher_ci_lower) + 1)
            ci_upper = (np.exp(2 * fisher_ci_upper) - 1) / (np.exp(2 * fisher_ci_upper) + 1)
            ci = (ci_lower, ci_upper)

        direction = "positive" if corr > 0 else "negative"
        interpretation = (
            f"{name} of {corr:.3f} indicates {magnitude.value} {direction} "
            f"association between variables."
        )

        return EffectSizeResult(
            name=name,
            value=corr,
            magnitude=magnitude,
            confidence_interval=ci,
            interpretation=interpretation,
        )

    def rank_biserial_correlation(self, group1: np.ndarray, group2: np.ndarray) -> EffectSizeResult:
        """
        Calculate rank-biserial correlation for Mann-Whitney U effect size.

        Args:
            group1: First group data
            group2: Second group data

        Returns:
            EffectSizeResult with rank-biserial correlation
        """
        n1, n2 = len(group1), len(group2)

        # Mann-Whitney U statistic
        u_statistic, _ = stats.mannwhitneyu(group1, group2, alternative="two-sided")

        # Rank-biserial correlation
        r = 1 - (2 * u_statistic) / (n1 * n2)

        magnitude = self._interpret_correlation(abs(r))

        interpretation = (
            f"Rank-biserial correlation of {r:.3f} indicates {magnitude.value} "
            f"effect size for non-parametric comparison."
        )

        return EffectSizeResult(
            name="Rank-biserial r", value=r, magnitude=magnitude, interpretation=interpretation
        )

    def _interpret_cohens_d(self, d: float) -> EffectSizeMagnitude:
        """Interpret Cohen's d magnitude using standard conventions."""
        if d < 0.2:
            return EffectSizeMagnitude.NEGLIGIBLE
        elif d < 0.5:
            return EffectSizeMagnitude.SMALL
        elif d < 0.8:
            return EffectSizeMagnitude.MEDIUM
        elif d < 1.3:
            return EffectSizeMagnitude.LARGE
        else:
            return EffectSizeMagnitude.VERY_LARGE

    def _interpret_cliffs_delta(self, delta: float) -> EffectSizeMagnitude:
        """Interpret Cliff's delta magnitude."""
        if delta < 0.11:
            return EffectSizeMagnitude.NEGLIGIBLE
        elif delta < 0.28:
            return EffectSizeMagnitude.SMALL
        elif delta < 0.43:
            return EffectSizeMagnitude.MEDIUM
        elif delta < 0.71:
            return EffectSizeMagnitude.LARGE
        else:
            return EffectSizeMagnitude.VERY_LARGE

    def _interpret_eta_squared(self, eta_sq: float) -> EffectSizeMagnitude:
        """Interpret eta-squared magnitude."""
        if eta_sq < 0.01:
            return EffectSizeMagnitude.NEGLIGIBLE
        elif eta_sq < 0.06:
            return EffectSizeMagnitude.SMALL
        elif eta_sq < 0.14:
            return EffectSizeMagnitude.MEDIUM
        elif eta_sq < 0.25:
            return EffectSizeMagnitude.LARGE
        else:
            return EffectSizeMagnitude.VERY_LARGE

    def _interpret_odds_ratio(self, or_value: float) -> EffectSizeMagnitude:
        """Interpret odds ratio magnitude."""
        if np.isinf(or_value):
            return EffectSizeMagnitude.VERY_LARGE

        # Use log odds ratio for interpretation
        log_or = abs(np.log(or_value)) if or_value > 0 else np.inf

        if log_or < 0.2:
            return EffectSizeMagnitude.NEGLIGIBLE
        elif log_or < 0.5:
            return EffectSizeMagnitude.SMALL
        elif log_or < 0.8:
            return EffectSizeMagnitude.MEDIUM
        elif log_or < 1.4:
            return EffectSizeMagnitude.LARGE
        else:
            return EffectSizeMagnitude.VERY_LARGE

    def _interpret_correlation(self, r: float) -> EffectSizeMagnitude:
        """Interpret correlation coefficient magnitude."""
        if r < 0.1:
            return EffectSizeMagnitude.NEGLIGIBLE
        elif r < 0.3:
            return EffectSizeMagnitude.SMALL
        elif r < 0.5:
            return EffectSizeMagnitude.MEDIUM
        elif r < 0.7:
            return EffectSizeMagnitude.LARGE
        else:
            return EffectSizeMagnitude.VERY_LARGE

    def _bootstrap_cohens_d_ci(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        pooled: bool,
        bias_correction: bool,
        n_bootstrap: int = 1000,
    ) -> tuple[float, float]:
        """Bootstrap confidence interval for Cohen's d."""
        bootstrap_ds = []

        for _ in range(n_bootstrap):
            # Bootstrap samples
            boot_group1 = np.random.choice(group1, size=len(group1), replace=True)
            boot_group2 = np.random.choice(group2, size=len(group2), replace=True)

            # Calculate Cohen's d for bootstrap sample
            result = self.cohens_d(boot_group1, boot_group2, pooled, bias_correction)
            bootstrap_ds.append(result.value)

        # Percentile confidence interval
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_ds, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_ds, 100 * (1 - alpha / 2))

        return ci_lower, ci_upper

    def _bootstrap_cliffs_delta_ci(
        self, group1: np.ndarray, group2: np.ndarray, n_bootstrap: int = 1000
    ) -> tuple[float, float]:
        """Bootstrap confidence interval for Cliff's delta."""
        bootstrap_deltas = []

        for _ in range(n_bootstrap):
            # Bootstrap samples
            boot_group1 = np.random.choice(group1, size=len(group1), replace=True)
            boot_group2 = np.random.choice(group2, size=len(group2), replace=True)

            # Calculate Cliff's delta for bootstrap sample
            result = self.cliffs_delta(boot_group1, boot_group2)
            bootstrap_deltas.append(result.value)

        # Percentile confidence interval
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_deltas, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_deltas, 100 * (1 - alpha / 2))

        return ci_lower, ci_upper


# Convenience functions for direct calculation
def cohens_d(group1: np.ndarray, group2: np.ndarray, **kwargs) -> float:
    """Calculate Cohen's d directly."""
    calculator = EffectSizeCalculator()
    return calculator.cohens_d(group1, group2, **kwargs).value


def cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cliff's delta directly."""
    calculator = EffectSizeCalculator()
    return calculator.cliffs_delta(group1, group2).value


def eta_squared(groups: list[np.ndarray], partial: bool = False) -> float:
    """Calculate eta-squared directly."""
    calculator = EffectSizeCalculator()
    return calculator.eta_squared(groups, partial).value


def odds_ratio(
    group1_success: int, group1_total: int, group2_success: int, group2_total: int
) -> float:
    """Calculate odds ratio directly."""
    calculator = EffectSizeCalculator()
    return calculator.odds_ratio(group1_success, group1_total, group2_success, group2_total).value


def test():
    """Test the effect size module."""
    np.random.seed(42)

    print("=== Effect Size Calculations Demo ===\n")

    # Test data
    group1 = np.random.normal(100, 15, 50)  # IQ-like data
    group2 = np.random.normal(110, 15, 50)  # Higher mean
    group3 = np.random.normal(95, 15, 50)  # Lower mean

    calculator = EffectSizeCalculator()

    # Cohen's d
    print("1. Cohen's d:")
    result = calculator.cohens_d(group1, group2)
    print(result)
    print(f"   {result.interpretation}\n")

    # Cliff's delta
    print("2. Cliff's delta:")
    result = calculator.cliffs_delta(group1, group2)
    print(result)
    print(f"   {result.interpretation}\n")

    # Eta-squared
    print("3. Eta-squared:")
    result = calculator.eta_squared([group1, group2, group3])
    print(result)
    print(f"   {result.interpretation}\n")

    # Odds ratio
    print("4. Odds ratio:")
    result = calculator.odds_ratio(40, 100, 25, 100)  # 40% vs 25% success
    print(result)
    print(f"   {result.interpretation}\n")

    # Correlation
    print("5. Correlation effect size:")
    x = np.random.normal(0, 1, 100)
    y = 0.5 * x + np.random.normal(0, 1, 100)  # Moderate correlation
    result = calculator.correlation_effect_size(x, y)
    print(result)
    print(f"   {result.interpretation}\n")


if __name__ == "__main__":
    test()
