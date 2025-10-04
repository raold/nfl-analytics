#!/usr/bin/env python3
"""
A/B Testing Framework for NFL Analytics.

Implements state-of-the-art A/B testing methods including sequential testing,
Bayesian A/B tests, and multi-armed bandits for sports betting experiments.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import scipy.stats as stats

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Status of A/B test."""

    PLANNING = "planning"
    RUNNING = "running"
    STOPPED_SUCCESS = "stopped_success"
    STOPPED_FUTILITY = "stopped_futility"
    STOPPED_EARLY = "stopped_early"
    COMPLETED = "completed"


class AllocationMethod(Enum):
    """Methods for allocating subjects to treatment arms."""

    FIXED = "fixed"  # Fixed randomization
    ADAPTIVE = "adaptive"  # Thompson sampling
    EPSILON_GREEDY = "epsilon_greedy"  # ε-greedy
    UCB = "ucb"  # Upper Confidence Bound


@dataclass
class TestArm:
    """Single arm of an A/B test."""

    name: str
    is_control: bool = False
    n: int = 0  # Number of observations
    sum_outcome: float = 0.0  # Sum of outcomes
    sum_squared: float = 0.0  # Sum of squared outcomes
    successes: int = 0  # For binary outcomes
    allocation_probability: float = 0.5
    prior_alpha: float = 1.0  # Beta prior parameter
    prior_beta: float = 1.0  # Beta prior parameter

    @property
    def mean(self) -> float:
        """Sample mean."""
        return self.sum_outcome / self.n if self.n > 0 else 0.0

    @property
    def variance(self) -> float:
        """Sample variance."""
        if self.n <= 1:
            return 0.0
        return (self.sum_squared - self.sum_outcome**2 / self.n) / (self.n - 1)

    @property
    def success_rate(self) -> float:
        """Success rate for binary outcomes."""
        return self.successes / self.n if self.n > 0 else 0.0

    @property
    def posterior_alpha(self) -> float:
        """Posterior alpha for Beta distribution."""
        return self.prior_alpha + self.successes

    @property
    def posterior_beta(self) -> float:
        """Posterior beta for Beta distribution."""
        return self.prior_beta + self.n - self.successes

    def add_observation(self, outcome: float, is_success: bool = None):
        """Add a new observation to this arm."""
        self.n += 1
        self.sum_outcome += outcome
        self.sum_squared += outcome**2

        if is_success is not None:
            if is_success:
                self.successes += 1
        elif outcome > 0:  # Assume binary if not specified
            self.successes += 1


@dataclass
class TestResult:
    """Results of an A/B test analysis."""

    test_name: str
    status: TestStatus
    arms: dict[str, TestArm]
    winner: str | None = None
    confidence: float = 0.0
    p_value: float | None = None
    effect_size: float | None = None
    confidence_interval: tuple[float, float] | None = None
    bayesian_probability: float | None = None
    expected_loss: dict[str, float] | None = None
    recommendation: str = ""
    analysis_time: datetime = field(default_factory=datetime.now)

    def summary(self) -> str:
        """Generate test summary."""
        summary = f"A/B Test: {self.test_name}\n"
        summary += f"Status: {self.status.value}\n"
        summary += f"Total observations: {sum(arm.n for arm in self.arms.values())}\n"

        for name, arm in self.arms.items():
            summary += f"\n{name} ({'Control' if arm.is_control else 'Treatment'}):\n"
            summary += f"  N: {arm.n}\n"
            summary += f"  Mean: {arm.mean:.4f}\n"
            summary += f"  Success rate: {arm.success_rate:.3f}\n"

        if self.winner:
            summary += f"\nWinner: {self.winner} (confidence: {self.confidence:.3f})\n"

        if self.p_value is not None:
            summary += f"P-value: {self.p_value:.4f}\n"

        if self.effect_size is not None:
            summary += f"Effect size: {self.effect_size:.4f}\n"

        summary += f"Recommendation: {self.recommendation}"
        return summary


class ABTest:
    """
    Comprehensive A/B testing framework.

    Supports fixed and sequential testing, Bayesian analysis,
    and adaptive allocation methods.
    """

    def __init__(
        self,
        name: str,
        alpha: float = 0.05,
        power: float = 0.8,
        minimum_effect_size: float = 0.1,
        allocation_method: AllocationMethod = AllocationMethod.FIXED,
    ):
        """
        Initialize A/B test.

        Args:
            name: Test name
            alpha: Type I error rate
            power: Desired statistical power
            minimum_effect_size: Minimum detectable effect size
            allocation_method: Method for allocating subjects
        """
        self.name = name
        self.alpha = alpha
        self.power = power
        self.minimum_effect_size = minimum_effect_size
        self.allocation_method = allocation_method

        self.arms: dict[str, TestArm] = {}
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        self.status = TestStatus.PLANNING

        # Sequential testing parameters
        self.max_n: int | None = None
        self.interim_analyses: list[int] = []
        self.alpha_spending: list[float] = []

        # Bayesian parameters
        self.credible_level = 0.95
        self.loss_threshold = 0.01

    def add_arm(
        self, name: str, is_control: bool = False, prior_alpha: float = 1.0, prior_beta: float = 1.0
    ) -> TestArm:
        """
        Add a treatment arm to the test.

        Args:
            name: Arm name
            is_control: Whether this is the control arm
            prior_alpha: Beta prior alpha parameter
            prior_beta: Beta prior beta parameter

        Returns:
            Created TestArm
        """
        if name in self.arms:
            raise ValueError(f"Arm '{name}' already exists")

        arm = TestArm(
            name=name, is_control=is_control, prior_alpha=prior_alpha, prior_beta=prior_beta
        )

        self.arms[name] = arm

        # Set initial allocation probabilities
        self._update_allocation_probabilities()

        return arm

    def calculate_sample_size(
        self,
        baseline_rate: float,
        treatment_effect: float | None = None,
        test_type: str = "proportion",
    ) -> int:
        """
        Calculate required sample size for the test.

        Args:
            baseline_rate: Baseline conversion/success rate
            treatment_effect: Expected treatment effect (None to use minimum_effect_size)
            test_type: Type of test ("proportion" or "continuous")

        Returns:
            Required sample size per arm
        """
        if treatment_effect is None:
            treatment_effect = self.minimum_effect_size

        if test_type == "proportion":
            # Two-proportion test
            p1 = baseline_rate
            p2 = baseline_rate + treatment_effect

            # Pooled proportion
            p_pooled = (p1 + p2) / 2

            # Z-scores
            z_alpha = stats.norm.ppf(1 - self.alpha / 2)
            z_beta = stats.norm.ppf(self.power)

            # Sample size calculation
            numerator = (
                z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled))
                + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
            ) ** 2
            denominator = (p2 - p1) ** 2

            n_per_arm = numerator / denominator

        elif test_type == "continuous":
            # Two-sample t-test
            # Assuming equal variances and effect size in standard deviations
            z_alpha = stats.norm.ppf(1 - self.alpha / 2)
            z_beta = stats.norm.ppf(self.power)

            n_per_arm = 2 * ((z_alpha + z_beta) / treatment_effect) ** 2

        else:
            raise ValueError("test_type must be 'proportion' or 'continuous'")

        return int(np.ceil(n_per_arm))

    def setup_sequential_design(
        self, max_n: int, n_interim: int = 4, spending_function: str = "obrien_fleming"
    ):
        """
        Setup sequential testing design with interim analyses.

        Args:
            max_n: Maximum sample size per arm
            n_interim: Number of interim analyses
            spending_function: Alpha spending function
        """
        self.max_n = max_n

        # Information fractions
        info_fractions = np.linspace(1 / n_interim, 1, n_interim)
        self.interim_analyses = [int(frac * max_n) for frac in info_fractions]

        # Alpha spending function
        if spending_function == "obrien_fleming":
            self.alpha_spending = self._obrien_fleming_spending(info_fractions)
        elif spending_function == "pocock":
            self.alpha_spending = self._pocock_spending(info_fractions)
        else:
            # Equal spending
            self.alpha_spending = [self.alpha * frac for frac in info_fractions]

    def start_test(self):
        """Start the A/B test."""
        if len(self.arms) < 2:
            raise ValueError("Must have at least 2 arms to start test")

        self.status = TestStatus.RUNNING
        self.start_time = datetime.now()
        logger.info(f"Started A/B test: {self.name}")

    def add_observation(
        self, arm_name: str, outcome: float, is_success: bool = None
    ) -> TestResult | None:
        """
        Add an observation to the test.

        Args:
            arm_name: Name of the arm
            outcome: Outcome value
            is_success: Whether outcome is a success (for binary outcomes)

        Returns:
            TestResult if test should stop, None otherwise
        """
        if self.status != TestStatus.RUNNING:
            raise ValueError("Test is not running")

        if arm_name not in self.arms:
            raise ValueError(f"Unknown arm: {arm_name}")

        # Add observation
        self.arms[arm_name].add_observation(outcome, is_success)

        # Update allocation probabilities if using adaptive method
        if self.allocation_method != AllocationMethod.FIXED:
            self._update_allocation_probabilities()

        # Check for interim analysis
        total_n = sum(arm.n for arm in self.arms.values())
        min_arm_n = min(arm.n for arm in self.arms.values())

        # Check if we should perform interim analysis
        if self.interim_analyses and min_arm_n in self.interim_analyses:
            result = self.analyze()
            if result.status in [
                TestStatus.STOPPED_SUCCESS,
                TestStatus.STOPPED_FUTILITY,
                TestStatus.STOPPED_EARLY,
            ]:
                self._stop_test(result.status)
                return result

        # Check if test is complete
        if self.max_n and min_arm_n >= self.max_n:
            result = self.analyze()
            self._stop_test(TestStatus.COMPLETED)
            return result

        return None

    def get_allocation(self) -> str:
        """
        Get the next arm allocation based on allocation method.

        Returns:
            Name of arm to allocate next subject to
        """
        if self.allocation_method == AllocationMethod.FIXED:
            # Simple randomization
            arm_names = list(self.arms.keys())
            return np.random.choice(arm_names)

        elif self.allocation_method == AllocationMethod.ADAPTIVE:
            # Thompson sampling
            arm_names = list(self.arms.keys())
            probabilities = [self.arms[name].allocation_probability for name in arm_names]
            return np.random.choice(arm_names, p=probabilities)

        elif self.allocation_method == AllocationMethod.EPSILON_GREEDY:
            # ε-greedy allocation
            epsilon = 0.1
            if np.random.random() < epsilon:
                # Explore: random allocation
                return np.random.choice(list(self.arms.keys()))
            else:
                # Exploit: best performing arm
                best_arm = max(self.arms.keys(), key=lambda x: self.arms[x].mean)
                return best_arm

        elif self.allocation_method == AllocationMethod.UCB:
            # Upper Confidence Bound
            total_n = sum(arm.n for arm in self.arms.values())
            if total_n == 0:
                return np.random.choice(list(self.arms.keys()))

            ucb_values = {}
            for name, arm in self.arms.items():
                if arm.n == 0:
                    ucb_values[name] = float("inf")
                else:
                    confidence_radius = np.sqrt(2 * np.log(total_n) / arm.n)
                    ucb_values[name] = arm.mean + confidence_radius

            return max(ucb_values.keys(), key=lambda x: ucb_values[x])

    def analyze(self) -> TestResult:
        """
        Analyze current test results.

        Returns:
            TestResult with current analysis
        """
        if len(self.arms) != 2:
            # Multi-arm analysis not implemented
            return self._multi_arm_analysis()

        # Two-arm analysis
        arms = list(self.arms.values())
        control_arm = next((arm for arm in arms if arm.is_control), arms[0])
        treatment_arm = next((arm for arm in arms if not arm.is_control), arms[1])

        # Frequentist analysis
        freq_result = self._frequentist_analysis(control_arm, treatment_arm)

        # Bayesian analysis
        bayes_result = self._bayesian_analysis(control_arm, treatment_arm)

        # Combine results
        result = TestResult(
            test_name=self.name,
            status=self.status,
            arms={arm.name: arm for arm in arms},
            winner=freq_result.get("winner"),
            confidence=freq_result.get("confidence", 0.0),
            p_value=freq_result.get("p_value"),
            effect_size=freq_result.get("effect_size"),
            confidence_interval=freq_result.get("confidence_interval"),
            bayesian_probability=bayes_result.get("probability"),
            expected_loss=bayes_result.get("expected_loss"),
            recommendation=self._generate_recommendation(freq_result, bayes_result),
        )

        return result

    def _frequentist_analysis(self, control: TestArm, treatment: TestArm) -> dict[str, Any]:
        """Perform frequentist statistical analysis."""
        if control.n == 0 or treatment.n == 0:
            return {"winner": None, "confidence": 0.0}

        # Two-sample t-test for continuous outcomes
        if control.variance > 0 and treatment.variance > 0:
            # Welch's t-test
            statistic, p_value = stats.ttest_ind_from_stats(
                control.mean,
                control.variance,
                control.n,
                treatment.mean,
                treatment.variance,
                treatment.n,
                equal_var=False,
            )

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((control.variance + treatment.variance) / 2)
            effect_size = (treatment.mean - control.mean) / pooled_std if pooled_std > 0 else 0

            # Confidence interval for difference
            se_diff = np.sqrt(control.variance / control.n + treatment.variance / treatment.n)
            df = (control.variance / control.n + treatment.variance / treatment.n) ** 2 / (
                (control.variance / control.n) ** 2 / (control.n - 1)
                + (treatment.variance / treatment.n) ** 2 / (treatment.n - 1)
            )

            t_critical = stats.t.ppf(1 - self.alpha / 2, df)
            diff = treatment.mean - control.mean
            margin_error = t_critical * se_diff
            ci = (diff - margin_error, diff + margin_error)

        else:
            # Proportion test
            p1 = control.success_rate
            p2 = treatment.success_rate
            n1, n2 = control.n, treatment.n

            # Two-proportion z-test
            p_pooled = (control.successes + treatment.successes) / (n1 + n2)
            se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))

            if se_pooled > 0:
                z_stat = (p2 - p1) / se_pooled
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            else:
                z_stat = 0
                p_value = 1.0

            # Effect size (Cohen's h)
            effect_size = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))

            # Confidence interval for difference in proportions
            se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
            z_critical = stats.norm.ppf(1 - self.alpha / 2)
            diff = p2 - p1
            margin_error = z_critical * se_diff
            ci = (diff - margin_error, diff + margin_error)

        # Determine winner
        winner = None
        confidence = 0.0

        if p_value < self.alpha:
            if treatment.mean > control.mean:
                winner = treatment.name
            else:
                winner = control.name
            confidence = 1 - p_value

        return {
            "winner": winner,
            "confidence": confidence,
            "p_value": p_value,
            "effect_size": effect_size,
            "confidence_interval": ci,
        }

    def _bayesian_analysis(self, control: TestArm, treatment: TestArm) -> dict[str, Any]:
        """Perform Bayesian analysis."""
        # For binary outcomes, use Beta-Binomial model
        n_samples = 10000

        # Sample from posterior distributions
        control_samples = np.random.beta(control.posterior_alpha, control.posterior_beta, n_samples)

        treatment_samples = np.random.beta(
            treatment.posterior_alpha, treatment.posterior_beta, n_samples
        )

        # Probability that treatment is better
        prob_treatment_better = np.mean(treatment_samples > control_samples)

        # Expected loss
        diff_samples = treatment_samples - control_samples
        expected_loss_treatment = np.mean(np.maximum(0, -diff_samples))
        expected_loss_control = np.mean(np.maximum(0, diff_samples))

        return {
            "probability": prob_treatment_better,
            "expected_loss": {
                treatment.name: expected_loss_treatment,
                control.name: expected_loss_control,
            },
        }

    def _multi_arm_analysis(self) -> TestResult:
        """Analyze multi-arm test (placeholder)."""
        # Simplified multi-arm analysis
        best_arm = max(self.arms.keys(), key=lambda x: self.arms[x].mean)

        return TestResult(
            test_name=self.name,
            status=self.status,
            arms=self.arms.copy(),
            winner=best_arm,
            recommendation="Multi-arm analysis requires more sophisticated methods",
        )

    def _update_allocation_probabilities(self):
        """Update allocation probabilities for adaptive methods."""
        if self.allocation_method == AllocationMethod.ADAPTIVE:
            # Thompson sampling: allocation proportional to probability of being best
            n_samples = 1000
            arm_samples = {}

            for name, arm in self.arms.items():
                if arm.n > 0:
                    arm_samples[name] = np.random.beta(
                        arm.posterior_alpha, arm.posterior_beta, n_samples
                    )
                else:
                    arm_samples[name] = np.random.beta(1, 1, n_samples)

            # Calculate probability each arm is best
            for name in self.arms:
                other_samples = [arm_samples[other] for other in self.arms if other != name]
                if other_samples:
                    prob_best = np.mean(arm_samples[name] > np.max(other_samples, axis=0))
                else:
                    prob_best = 1.0

                self.arms[name].allocation_probability = prob_best

            # Normalize probabilities
            total_prob = sum(arm.allocation_probability for arm in self.arms.values())
            if total_prob > 0:
                for arm in self.arms.values():
                    arm.allocation_probability /= total_prob

    def _obrien_fleming_spending(self, info_fractions: np.ndarray) -> list[float]:
        """O'Brien-Fleming alpha spending function."""
        spending = []
        cumulative = 0.0

        for i, frac in enumerate(info_fractions):
            if i == 0:
                alpha_i = 2 * (
                    1 - stats.norm.cdf(stats.norm.ppf(1 - self.alpha / 2) / np.sqrt(frac))
                )
            else:
                alpha_i = (
                    2 * (1 - stats.norm.cdf(stats.norm.ppf(1 - self.alpha / 2) / np.sqrt(frac)))
                    - cumulative
                )

            cumulative += alpha_i
            spending.append(alpha_i)

        return spending

    def _pocock_spending(self, info_fractions: np.ndarray) -> list[float]:
        """Pocock alpha spending function."""
        # Equal spending at each analysis
        alpha_per_analysis = self.alpha / len(info_fractions)
        return [alpha_per_analysis] * len(info_fractions)

    def _generate_recommendation(
        self, freq_result: dict[str, Any], bayes_result: dict[str, Any]
    ) -> str:
        """Generate recommendation based on analysis results."""
        recommendations = []

        # Frequentist recommendation
        if freq_result.get("winner"):
            p_val = freq_result.get("p_value", 1.0)
            effect = freq_result.get("effect_size", 0.0)
            recommendations.append(
                f"Frequentist: {freq_result['winner']} wins "
                f"(p={p_val:.3f}, effect={effect:.3f})"
            )
        else:
            recommendations.append("Frequentist: No significant difference")

        # Bayesian recommendation
        prob = bayes_result.get("probability", 0.5)
        if prob > 0.95:
            recommendations.append(f"Bayesian: Strong evidence for treatment (p={prob:.3f})")
        elif prob < 0.05:
            recommendations.append(f"Bayesian: Strong evidence for control (p={1-prob:.3f})")
        else:
            recommendations.append(f"Bayesian: Inconclusive (p={prob:.3f})")

        # Overall recommendation
        total_n = sum(arm.n for arm in self.arms.values())
        if total_n < 100:
            recommendations.append("Recommendation: Collect more data")
        elif freq_result.get("winner") and prob > 0.9:
            recommendations.append(f"Recommendation: Implement {freq_result['winner']}")
        else:
            recommendations.append(
                "Recommendation: No clear winner, consider practical significance"
            )

        return " | ".join(recommendations)

    def _stop_test(self, status: TestStatus):
        """Stop the test with given status."""
        self.status = status
        self.end_time = datetime.now()
        logger.info(f"Stopped A/B test: {self.name} with status: {status.value}")


def test():
    """Test the A/B testing framework."""
    np.random.seed(42)

    print("=== A/B Testing Framework Demo ===\n")

    # Create test
    test = ABTest(
        name="Betting Strategy Test",
        alpha=0.05,
        power=0.8,
        minimum_effect_size=0.05,
        allocation_method=AllocationMethod.ADAPTIVE,
    )

    # Add arms
    test.add_arm("control", is_control=True)
    test.add_arm("treatment")

    # Calculate sample size
    n_required = test.calculate_sample_size(
        baseline_rate=0.15,  # 15% win rate
        treatment_effect=0.05,  # 5% improvement
        test_type="proportion",
    )
    print(f"Required sample size per arm: {n_required}")

    # Setup sequential design
    test.setup_sequential_design(max_n=n_required, n_interim=3, spending_function="obrien_fleming")

    # Start test
    test.start_test()

    # Simulate data collection
    print("\nSimulating test execution...")

    control_rate = 0.15
    treatment_rate = 0.20  # 5% improvement

    for i in range(n_required * 2):  # Simulate collecting data
        # Get allocation
        arm_name = test.get_allocation()

        # Simulate outcome
        if arm_name == "control":
            success = np.random.random() < control_rate
        else:
            success = np.random.random() < treatment_rate

        outcome = 1.0 if success else 0.0

        # Add observation
        result = test.add_observation(arm_name, outcome, success)

        # Check if test stopped early
        if result:
            print(f"Test stopped early after {i+1} observations")
            break

        # Print interim results every 100 observations
        if (i + 1) % 100 == 0:
            interim_result = test.analyze()
            print(f"Interim analysis at n={i+1}:")
            print(
                f"  Control: {test.arms['control'].success_rate:.3f} "
                f"({test.arms['control'].n} obs)"
            )
            print(
                f"  Treatment: {test.arms['treatment'].success_rate:.3f} "
                f"({test.arms['treatment'].n} obs)"
            )

    # Final analysis
    final_result = test.analyze()
    print("\n=== FINAL RESULTS ===")
    print(final_result.summary())


if __name__ == "__main__":
    test()
