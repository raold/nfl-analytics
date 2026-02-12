#!/usr/bin/env python3
"""
Multiple Comparison Correction Methods.

Implements state-of-the-art methods for controlling Type I error rates
when performing multiple statistical tests, including FDR and FWER
controlling procedures based on 2024 research.
"""

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import scipy.stats as stats

logger = logging.getLogger(__name__)


class CorrectionMethod(Enum):
    """Available multiple comparison correction methods."""

    # FDR Methods (Less Conservative)
    BENJAMINI_HOCHBERG = "benjamini_hochberg"
    BENJAMINI_YEKUTIELI = "benjamini_yekutieli"
    ADAPTIVE_FDR = "adaptive_fdr"

    # FWER Methods (More Conservative)
    BONFERRONI = "bonferroni"
    HOLM_BONFERRONI = "holm_bonferroni"
    SIDAK = "sidak"
    HOCHBERG = "hochberg"
    HOMMEL = "hommel"


@dataclass
class CorrectionResult:
    """Results of multiple comparison correction."""

    method: str
    original_alpha: float
    corrected_alpha: float | list[float]
    raw_p_values: np.ndarray
    corrected_p_values: np.ndarray
    rejected: np.ndarray  # Boolean array of rejected hypotheses
    num_rejected: int
    num_discoveries: int  # For FDR methods
    estimated_fdr: float | None = None
    estimated_power: float | None = None

    def summary(self) -> str:
        """Generate summary of correction results."""
        rejection_rate = self.num_rejected / len(self.raw_p_values) * 100

        summary = f"Multiple Comparison Correction: {self.method}\n"
        summary += f"Total tests: {len(self.raw_p_values)}\n"
        summary += f"Rejected hypotheses: {self.num_rejected} ({rejection_rate:.1f}%)\n"
        summary += f"Original α: {self.original_alpha:.3f}\n"

        if isinstance(self.corrected_alpha, list):
            summary += f"Adjusted α range: {min(self.corrected_alpha):.6f} - {max(self.corrected_alpha):.6f}\n"
        else:
            summary += f"Adjusted α: {self.corrected_alpha:.6f}\n"

        if self.estimated_fdr is not None:
            summary += f"Estimated FDR: {self.estimated_fdr:.3f}\n"

        return summary


class MultipleComparisonCorrection:
    """
    Comprehensive multiple comparison correction framework.

    Implements both FDR-controlling and FWER-controlling procedures
    with modern adaptive methods based on 2024 research standards.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize multiple comparison correction.

        Args:
            alpha: Family-wise Type I error rate
        """
        self.alpha = alpha

    def correct(
        self,
        p_values: list[float] | np.ndarray,
        method: CorrectionMethod = CorrectionMethod.BENJAMINI_HOCHBERG,
        **kwargs,
    ) -> CorrectionResult:
        """
        Apply multiple comparison correction to p-values.

        Args:
            p_values: Raw p-values from multiple tests
            method: Correction method to use
            **kwargs: Method-specific parameters

        Returns:
            CorrectionResult with corrected p-values and decisions
        """
        p_values = np.asarray(p_values)

        if method == CorrectionMethod.BENJAMINI_HOCHBERG:
            return self._benjamini_hochberg(p_values, **kwargs)
        elif method == CorrectionMethod.BENJAMINI_YEKUTIELI:
            return self._benjamini_yekutieli(p_values, **kwargs)
        elif method == CorrectionMethod.ADAPTIVE_FDR:
            return self._adaptive_fdr(p_values, **kwargs)
        elif method == CorrectionMethod.BONFERRONI:
            return self._bonferroni(p_values)
        elif method == CorrectionMethod.HOLM_BONFERRONI:
            return self._holm_bonferroni(p_values)
        elif method == CorrectionMethod.SIDAK:
            return self._sidak(p_values)
        elif method == CorrectionMethod.HOCHBERG:
            return self._hochberg(p_values)
        elif method == CorrectionMethod.HOMMEL:
            return self._hommel(p_values)
        else:
            raise ValueError(f"Unknown correction method: {method}")

    def _benjamini_hochberg(
        self, p_values: np.ndarray, fdr_level: float = None
    ) -> CorrectionResult:
        """
        Benjamini-Hochberg FDR control procedure.

        The standard FDR method, less conservative than FWER methods.
        Particularly effective for exploratory research.
        """
        if fdr_level is None:
            fdr_level = self.alpha

        m = len(p_values)

        # Sort p-values and keep track of original indices
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]

        # Calculate adjusted p-values
        adjusted_p = np.zeros_like(sorted_p)

        # Work backwards through sorted p-values
        for i in range(m - 1, -1, -1):
            if i == m - 1:
                adjusted_p[i] = sorted_p[i]
            else:
                adjusted_p[i] = min(adjusted_p[i + 1], sorted_p[i] * m / (i + 1))

        # Restore original order
        corrected_p = np.zeros_like(p_values)
        corrected_p[sorted_indices] = adjusted_p

        # Determine rejections
        rejected = corrected_p <= fdr_level
        num_rejected = np.sum(rejected)

        # Estimate actual FDR
        if num_rejected > 0:
            estimated_fdr = fdr_level * m / num_rejected
        else:
            estimated_fdr = 0.0

        return CorrectionResult(
            method="Benjamini-Hochberg",
            original_alpha=self.alpha,
            corrected_alpha=fdr_level,
            raw_p_values=p_values,
            corrected_p_values=corrected_p,
            rejected=rejected,
            num_rejected=num_rejected,
            num_discoveries=num_rejected,
            estimated_fdr=min(estimated_fdr, 1.0),
        )

    def _benjamini_yekutieli(
        self, p_values: np.ndarray, fdr_level: float = None
    ) -> CorrectionResult:
        """
        Benjamini-Yekutieli FDR control procedure.

        More conservative than B-H but valid under arbitrary dependence
        between test statistics. Recommended for dependent tests.
        """
        if fdr_level is None:
            fdr_level = self.alpha

        m = len(p_values)

        # Calculate c(m) constant for arbitrary dependence
        c_m = np.sum(1.0 / np.arange(1, m + 1))

        # Adjust FDR level
        adjusted_fdr = fdr_level / c_m

        # Apply B-H procedure with adjusted level
        result = self._benjamini_hochberg(p_values, adjusted_fdr)
        result.method = "Benjamini-Yekutieli"
        result.corrected_alpha = adjusted_fdr

        return result

    def _adaptive_fdr(
        self, p_values: np.ndarray, fdr_level: float = None, lambda_seq: np.ndarray | None = None
    ) -> CorrectionResult:
        """
        Adaptive FDR procedure (Benjamini-Hochberg-Yekutieli).

        Estimates the proportion of true null hypotheses and adapts
        the procedure accordingly. More powerful than standard B-H.
        """
        if fdr_level is None:
            fdr_level = self.alpha

        m = len(p_values)

        # Default lambda sequence
        if lambda_seq is None:
            lambda_seq = np.arange(0.05, 0.95, 0.05)

        # Estimate proportion of true nulls
        pi0_estimates = []

        for lam in lambda_seq:
            w_lam = np.sum(p_values > lam)
            pi0_hat = w_lam / (m * (1 - lam))
            pi0_estimates.append(min(pi0_hat, 1.0))

        # Use most conservative estimate
        pi0 = np.max(pi0_estimates) if pi0_estimates else 1.0

        # Adaptive B-H procedure
        adaptive_alpha = fdr_level / pi0

        result = self._benjamini_hochberg(p_values, adaptive_alpha)
        result.method = "Adaptive FDR"
        result.corrected_alpha = adaptive_alpha

        return result

    def _bonferroni(self, p_values: np.ndarray) -> CorrectionResult:
        """
        Bonferroni correction for FWER control.

        Most conservative method. Appropriate when any false positive
        would be highly problematic.
        """
        m = len(p_values)
        corrected_alpha = self.alpha / m
        corrected_p = p_values * m
        corrected_p = np.minimum(corrected_p, 1.0)  # Cap at 1

        rejected = corrected_p <= self.alpha

        return CorrectionResult(
            method="Bonferroni",
            original_alpha=self.alpha,
            corrected_alpha=corrected_alpha,
            raw_p_values=p_values,
            corrected_p_values=corrected_p,
            rejected=rejected,
            num_rejected=np.sum(rejected),
            num_discoveries=np.sum(rejected),
        )

    def _holm_bonferroni(self, p_values: np.ndarray) -> CorrectionResult:
        """
        Holm-Bonferroni step-down procedure.

        More powerful than Bonferroni while maintaining FWER control.
        Tests are evaluated sequentially from smallest to largest p-value.
        """
        m = len(p_values)

        # Sort p-values and track indices
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]

        # Step-down procedure
        rejected_sorted = np.zeros(m, dtype=bool)

        for i in range(m):
            # Adjusted alpha for step i
            alpha_i = self.alpha / (m - i)

            if sorted_p[i] <= alpha_i:
                rejected_sorted[i] = True
            else:
                # Stop at first non-rejection
                break

        # Calculate adjusted p-values
        adjusted_p_sorted = np.zeros(m)
        for i in range(m):
            adjusted_p_sorted[i] = max(
                sorted_p[i] * (m - i), adjusted_p_sorted[i - 1] if i > 0 else 0
            )

        # Restore original order
        corrected_p = np.zeros_like(p_values)
        rejected = np.zeros(m, dtype=bool)

        corrected_p[sorted_indices] = adjusted_p_sorted
        rejected[sorted_indices] = rejected_sorted

        # Cap corrected p-values at 1
        corrected_p = np.minimum(corrected_p, 1.0)

        # Create alpha list for step-down
        alpha_list = [self.alpha / (m - i) for i in range(m)]

        return CorrectionResult(
            method="Holm-Bonferroni",
            original_alpha=self.alpha,
            corrected_alpha=alpha_list,
            raw_p_values=p_values,
            corrected_p_values=corrected_p,
            rejected=rejected,
            num_rejected=np.sum(rejected),
            num_discoveries=np.sum(rejected),
        )

    def _sidak(self, p_values: np.ndarray) -> CorrectionResult:
        """
        Šidák correction for FWER control.

        Slightly less conservative than Bonferroni when tests are independent.
        """
        m = len(p_values)
        corrected_alpha = 1 - (1 - self.alpha) ** (1 / m)

        # Šidák-corrected p-values
        corrected_p = 1 - (1 - p_values) ** m
        corrected_p = np.minimum(corrected_p, 1.0)

        rejected = corrected_p <= self.alpha

        return CorrectionResult(
            method="Šidák",
            original_alpha=self.alpha,
            corrected_alpha=corrected_alpha,
            raw_p_values=p_values,
            corrected_p_values=corrected_p,
            rejected=rejected,
            num_rejected=np.sum(rejected),
            num_discoveries=np.sum(rejected),
        )

    def _hochberg(self, p_values: np.ndarray) -> CorrectionResult:
        """
        Hochberg step-up procedure.

        More powerful than Holm procedure, valid under certain
        dependence conditions.
        """
        m = len(p_values)

        # Sort p-values in descending order
        sorted_indices = np.argsort(p_values)[::-1]  # Largest to smallest
        sorted_p = p_values[sorted_indices]

        # Step-up procedure
        rejected_sorted = np.zeros(m, dtype=bool)

        for i in range(m):
            # Adjusted alpha for step i (from largest p-value)
            alpha_i = self.alpha / (i + 1)

            if sorted_p[i] <= alpha_i:
                # Reject this and all smaller p-values
                rejected_sorted[i:] = True
                break

        # Calculate adjusted p-values
        adjusted_p_sorted = np.zeros(m)
        for i in range(m - 1, -1, -1):  # Work backwards
            adjusted_p_sorted[i] = min(
                sorted_p[i] * (i + 1), adjusted_p_sorted[i + 1] if i < m - 1 else np.inf
            )

        # Restore original order
        corrected_p = np.zeros_like(p_values)
        rejected = np.zeros(m, dtype=bool)

        # Reverse the sorted indices to get back to original order
        reverse_indices = np.argsort(sorted_indices)
        corrected_p = adjusted_p_sorted[reverse_indices]
        rejected = rejected_sorted[reverse_indices]

        corrected_p = np.minimum(corrected_p, 1.0)

        return CorrectionResult(
            method="Hochberg",
            original_alpha=self.alpha,
            corrected_alpha=self.alpha,  # No single adjusted alpha
            raw_p_values=p_values,
            corrected_p_values=corrected_p,
            rejected=rejected,
            num_rejected=np.sum(rejected),
            num_discoveries=np.sum(rejected),
        )

    def _hommel(self, p_values: np.ndarray) -> CorrectionResult:
        """
        Hommel procedure.

        Most powerful among FWER-controlling procedures,
        but computationally intensive for large m.
        """
        m = len(p_values)
        sorted_p = np.sort(p_values)

        # Find largest j such that P(j) > alpha * j / m
        j_max = 0
        for j in range(1, m + 1):
            if sorted_p[j - 1] > self.alpha * j / m:
                j_max = j
                break

        if j_max == 0:
            # All null hypotheses rejected
            rejected = np.ones(m, dtype=bool)
        else:
            # Apply Bonferroni to first j_max - 1 hypotheses
            rejected = (
                p_values <= self.alpha / (j_max - 1) if j_max > 1 else np.zeros(m, dtype=bool)
            )

        # Calculate adjusted p-values (simplified)
        if j_max > 1:
            corrected_p = p_values * (j_max - 1)
        else:
            corrected_p = p_values * m

        corrected_p = np.minimum(corrected_p, 1.0)

        return CorrectionResult(
            method="Hommel",
            original_alpha=self.alpha,
            corrected_alpha=self.alpha / (j_max - 1) if j_max > 1 else self.alpha / m,
            raw_p_values=p_values,
            corrected_p_values=corrected_p,
            rejected=rejected,
            num_rejected=np.sum(rejected),
            num_discoveries=np.sum(rejected),
        )

    def compare_methods(
        self, p_values: list[float] | np.ndarray, methods: list[CorrectionMethod] | None = None
    ) -> dict[str, CorrectionResult]:
        """
        Compare multiple correction methods on the same data.

        Args:
            p_values: Raw p-values from multiple tests
            methods: List of methods to compare (default: all major methods)

        Returns:
            Dictionary mapping method names to CorrectionResults
        """
        if methods is None:
            methods = [
                CorrectionMethod.BENJAMINI_HOCHBERG,
                CorrectionMethod.BENJAMINI_YEKUTIELI,
                CorrectionMethod.BONFERRONI,
                CorrectionMethod.HOLM_BONFERRONI,
                CorrectionMethod.SIDAK,
            ]

        results = {}
        for method in methods:
            try:
                result = self.correct(p_values, method)
                results[result.method] = result
            except Exception as e:
                logger.warning(f"Failed to apply {method.value}: {e}")
                continue

        return results

    def power_analysis(
        self,
        effect_sizes: np.ndarray,
        sample_sizes: np.ndarray,
        method: CorrectionMethod = CorrectionMethod.BENJAMINI_HOCHBERG,
        n_simulations: int = 1000,
    ) -> dict[str, float]:
        """
        Estimate statistical power under multiple testing correction.

        Args:
            effect_sizes: True effect sizes for each test
            sample_sizes: Sample sizes for each test
            method: Correction method
            n_simulations: Number of simulation runs

        Returns:
            Dictionary with power estimates
        """
        len(effect_sizes)
        power_estimates = []

        for _ in range(n_simulations):
            # Simulate p-values
            p_values = []

            for i, (effect_size, n) in enumerate(zip(effect_sizes, sample_sizes)):
                if effect_size == 0:
                    # Null hypothesis true
                    p = np.random.uniform(0, 1)
                else:
                    # Alternative hypothesis true - simulate based on effect size
                    # This is a simplified simulation
                    z_score = effect_size * np.sqrt(n / 2)
                    p = 2 * (1 - stats.norm.cdf(abs(z_score)))

                p_values.append(p)

            # Apply correction
            result = self.correct(p_values, method)

            # Calculate power (proportion of true positives detected)
            true_positives = np.sum(result.rejected & (effect_sizes != 0))
            total_positives = np.sum(effect_sizes != 0)

            if total_positives > 0:
                power = true_positives / total_positives
            else:
                power = 0.0

            power_estimates.append(power)

        return {
            "mean_power": np.mean(power_estimates),
            "std_power": np.std(power_estimates),
            "ci_lower": np.percentile(power_estimates, 2.5),
            "ci_upper": np.percentile(power_estimates, 97.5),
        }


# Convenience functions for direct use
def fdr_benjamini_hochberg(
    p_values: list[float] | np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Benjamini-Hochberg FDR correction.

    Args:
        p_values: Raw p-values
        alpha: FDR level

    Returns:
        Tuple of (rejected, corrected_p_values)
    """
    corrector = MultipleComparisonCorrection(alpha)
    result = corrector.correct(p_values, CorrectionMethod.BENJAMINI_HOCHBERG)
    return result.rejected, result.corrected_p_values


def fdr_benjamini_yekutieli(
    p_values: list[float] | np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Benjamini-Yekutieli FDR correction.

    Args:
        p_values: Raw p-values
        alpha: FDR level

    Returns:
        Tuple of (rejected, corrected_p_values)
    """
    corrector = MultipleComparisonCorrection(alpha)
    result = corrector.correct(p_values, CorrectionMethod.BENJAMINI_YEKUTIELI)
    return result.rejected, result.corrected_p_values


def bonferroni_correction(
    p_values: list[float] | np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Bonferroni correction.

    Args:
        p_values: Raw p-values
        alpha: FWER level

    Returns:
        Tuple of (rejected, corrected_p_values)
    """
    corrector = MultipleComparisonCorrection(alpha)
    result = corrector.correct(p_values, CorrectionMethod.BONFERRONI)
    return result.rejected, result.corrected_p_values


def holm_bonferroni(
    p_values: list[float] | np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Holm-Bonferroni correction.

    Args:
        p_values: Raw p-values
        alpha: FWER level

    Returns:
        Tuple of (rejected, corrected_p_values)
    """
    corrector = MultipleComparisonCorrection(alpha)
    result = corrector.correct(p_values, CorrectionMethod.HOLM_BONFERRONI)
    return result.rejected, result.corrected_p_values


def test():
    """Test the multiple comparisons module."""
    np.random.seed(42)

    print("=== Multiple Comparison Correction Demo ===\n")

    # Simulate p-values from multiple tests
    # Some true positives, some true negatives
    true_effects = np.array([0, 0, 0, 0, 0, 2, 0, 3, 0, 1.5, 0, 0, 2.5, 0, 0])
    p_values = []

    for effect in true_effects:
        if effect == 0:
            # Null hypothesis true
            p = np.random.uniform(0, 1)
        else:
            # Alternative true - simulate significant p-value
            z = effect + np.random.normal(0, 0.5)
            p = 2 * (1 - stats.norm.cdf(abs(z)))
        p_values.append(p)

    p_values = np.array(p_values)

    print(f"Raw p-values: {p_values}")
    print(f"True effects: {true_effects}")
    print(f"True positives at α=0.05: {np.sum((p_values < 0.05) & (true_effects > 0))}")
    print(f"False positives at α=0.05: {np.sum((p_values < 0.05) & (true_effects == 0))}")
    print()

    # Test different correction methods
    corrector = MultipleComparisonCorrection(alpha=0.05)

    methods_to_test = [
        CorrectionMethod.BENJAMINI_HOCHBERG,
        CorrectionMethod.BENJAMINI_YEKUTIELI,
        CorrectionMethod.BONFERRONI,
        CorrectionMethod.HOLM_BONFERRONI,
    ]

    for method in methods_to_test:
        print(f"=== {method.value.upper().replace('_', ' ')} ===")
        result = corrector.correct(p_values, method)
        print(result.summary())

        # Calculate true/false positives
        tp = np.sum(result.rejected & (true_effects > 0))
        fp = np.sum(result.rejected & (true_effects == 0))
        fn = np.sum(~result.rejected & (true_effects > 0))
        tn = np.sum(~result.rejected & (true_effects == 0))

        print(f"True Positives: {tp}, False Positives: {fp}")
        print(f"True Negatives: {tn}, False Negatives: {fn}")

        if tp + fp > 0:
            precision = tp / (tp + fp)
            print(f"Precision: {precision:.3f}")

        if tp + fn > 0:
            sensitivity = tp / (tp + fn)
            print(f"Sensitivity: {sensitivity:.3f}")

        print()

    # Method comparison
    print("=== METHOD COMPARISON ===")
    comparison = corrector.compare_methods(p_values)

    for method_name, result in comparison.items():
        print(
            f"{method_name}: {result.num_rejected} rejections, "
            f"estimated FDR: {result.estimated_fdr:.3f}"
            if result.estimated_fdr
            else "N/A"
        )


if __name__ == "__main__":
    test()
