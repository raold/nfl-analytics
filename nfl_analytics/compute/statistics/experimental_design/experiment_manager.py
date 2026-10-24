#!/usr/bin/env python3
"""
Experiment Management System for NFL Analytics.

Manages experimental designs, randomization, control groups, and
treatment effect estimation for sports betting experiments.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RandomizationMethod(Enum):
    """Methods for randomizing subjects to treatment groups."""

    SIMPLE = "simple"  # Simple randomization
    BLOCKED = "blocked"  # Block randomization
    STRATIFIED = "stratified"  # Stratified randomization
    CLUSTER = "cluster"  # Cluster randomization
    MATCHED_PAIRS = "matched_pairs"  # Matched pairs design


class ExperimentType(Enum):
    """Types of experiments."""

    AB_TEST = "ab_test"
    FACTORIAL = "factorial"
    CROSSOVER = "crossover"
    DOSE_RESPONSE = "dose_response"
    ADAPTIVE = "adaptive"


@dataclass
class StratificationFactor:
    """Factor for stratified randomization."""

    name: str
    levels: list[str]
    allocation_ratios: dict[str, float] | None = None


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""

    name: str
    experiment_type: ExperimentType
    randomization_method: RandomizationMethod
    treatment_arms: list[str]
    control_arm: str
    target_sample_size: int
    stratification_factors: list[StratificationFactor] = field(default_factory=list)
    block_size: int | None = None
    allocation_ratio: dict[str, float] = field(default_factory=dict)
    minimum_effect_size: float = 0.1
    alpha: float = 0.05
    power: float = 0.8
    primary_outcome: str = "success_rate"
    secondary_outcomes: list[str] = field(default_factory=list)
    eligibility_criteria: dict[str, Any] = field(default_factory=dict)
    experiment_duration: timedelta | None = None

    def __post_init__(self):
        """Initialize default allocation ratios."""
        if not self.allocation_ratio:
            n_arms = len(self.treatment_arms) + 1  # +1 for control
            equal_ratio = 1.0 / n_arms
            self.allocation_ratio = {arm: equal_ratio for arm in self.treatment_arms}
            self.allocation_ratio[self.control_arm] = equal_ratio


@dataclass
class Subject:
    """Individual subject in an experiment."""

    id: str
    assigned_arm: str
    assignment_time: datetime
    stratification_values: dict[str, str] = field(default_factory=dict)
    baseline_data: dict[str, Any] = field(default_factory=dict)
    outcomes: dict[str, Any] = field(default_factory=dict)
    compliance: bool = True
    dropout_time: datetime | None = None


class ExperimentManager:
    """
    Comprehensive experiment management system.

    Handles randomization, subject assignment, data collection,
    and analysis for various experimental designs.
    """

    def __init__(self, config: ExperimentConfig, random_seed: int = 42):
        """
        Initialize experiment manager.

        Args:
            config: Experiment configuration
            random_seed: Random seed for reproducibility
        """
        self.config = config
        self.random_seed = random_seed
        np.random.seed(random_seed)

        self.subjects: dict[str, Subject] = {}
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None

        # Randomization state
        self._randomization_list: list[str] = []
        self._current_block: list[str] = []
        self._stratum_assignments: dict[str, list[str]] = {}

        # Setup randomization
        self._initialize_randomization()

    def _initialize_randomization(self):
        """Initialize randomization scheme based on method."""
        if self.config.randomization_method == RandomizationMethod.SIMPLE:
            self._initialize_simple_randomization()
        elif self.config.randomization_method == RandomizationMethod.BLOCKED:
            self._initialize_blocked_randomization()
        elif self.config.randomization_method == RandomizationMethod.STRATIFIED:
            self._initialize_stratified_randomization()
        else:
            logger.warning(
                f"Randomization method {self.config.randomization_method} not fully implemented"
            )

    def _initialize_simple_randomization(self):
        """Initialize simple randomization."""
        # Create weighted list based on allocation ratios
        arms = [self.config.control_arm] + self.config.treatment_arms
        weights = [self.config.allocation_ratio.get(arm, 1.0) for arm in arms]

        # Normalize weights
        total_weight = sum(weights)
        self._allocation_probabilities = {
            arm: weight / total_weight for arm, weight in zip(arms, weights)
        }

    def _initialize_blocked_randomization(self):
        """Initialize block randomization."""
        if self.config.block_size is None:
            # Default block size: multiple of number of arms
            n_arms = len(self.config.treatment_arms) + 1
            self.config.block_size = n_arms * 4

        self._generate_new_block()

    def _initialize_stratified_randomization(self):
        """Initialize stratified randomization."""
        # Generate all possible strata combinations
        if not self.config.stratification_factors:
            logger.warning("No stratification factors provided for stratified randomization")
            return

        # Create stratum identifiers
        strata = self._generate_strata_combinations()

        # Initialize randomization for each stratum
        for stratum in strata:
            self._stratum_assignments[stratum] = []

    def _generate_strata_combinations(self) -> list[str]:
        """Generate all combinations of stratification factors."""
        if not self.config.stratification_factors:
            return ["default"]

        # Cartesian product of all factor levels
        import itertools

        factor_levels = [factor.levels for factor in self.config.stratification_factors]
        combinations = list(itertools.product(*factor_levels))

        # Convert to string identifiers
        strata = []
        for combo in combinations:
            stratum_id = "_".join(
                f"{factor.name}:{level}"
                for factor, level in zip(self.config.stratification_factors, combo)
            )
            strata.append(stratum_id)

        return strata

    def _generate_new_block(self):
        """Generate a new randomization block."""
        arms = [self.config.control_arm] + self.config.treatment_arms
        n_arms = len(arms)

        # Calculate number of subjects per arm in block
        subjects_per_arm = {}
        remaining_size = self.config.block_size

        for arm in arms:
            ratio = self.config.allocation_ratio.get(arm, 1.0 / n_arms)
            n_subjects = int(ratio * self.config.block_size)
            subjects_per_arm[arm] = n_subjects
            remaining_size -= n_subjects

        # Distribute remaining subjects randomly
        arms_list = list(arms)
        for _ in range(remaining_size):
            arm = np.random.choice(arms_list)
            subjects_per_arm[arm] += 1

        # Create block
        block = []
        for arm, count in subjects_per_arm.items():
            block.extend([arm] * count)

        # Shuffle the block
        np.random.shuffle(block)
        self._current_block = block

    def assign_subject(
        self,
        subject_id: str,
        baseline_data: dict[str, Any] | None = None,
        stratification_values: dict[str, str] | None = None,
    ) -> str:
        """
        Assign a subject to a treatment arm.

        Args:
            subject_id: Unique subject identifier
            baseline_data: Baseline characteristics
            stratification_values: Values for stratification factors

        Returns:
            Assigned treatment arm
        """
        if subject_id in self.subjects:
            raise ValueError(f"Subject {subject_id} already assigned")

        # Check eligibility
        if not self._check_eligibility(baseline_data or {}):
            raise ValueError(f"Subject {subject_id} does not meet eligibility criteria")

        # Get assignment based on randomization method
        if self.config.randomization_method == RandomizationMethod.SIMPLE:
            assigned_arm = self._simple_assignment()
        elif self.config.randomization_method == RandomizationMethod.BLOCKED:
            assigned_arm = self._blocked_assignment()
        elif self.config.randomization_method == RandomizationMethod.STRATIFIED:
            assigned_arm = self._stratified_assignment(stratification_values or {})
        else:
            # Fallback to simple
            assigned_arm = self._simple_assignment()

        # Create subject record
        subject = Subject(
            id=subject_id,
            assigned_arm=assigned_arm,
            assignment_time=datetime.now(),
            stratification_values=stratification_values or {},
            baseline_data=baseline_data or {},
        )

        self.subjects[subject_id] = subject

        logger.info(f"Assigned subject {subject_id} to arm {assigned_arm}")
        return assigned_arm

    def _simple_assignment(self) -> str:
        """Perform simple randomization assignment."""
        arms = list(self._allocation_probabilities.keys())
        probabilities = list(self._allocation_probabilities.values())
        return np.random.choice(arms, p=probabilities)

    def _blocked_assignment(self) -> str:
        """Perform blocked randomization assignment."""
        if not self._current_block:
            self._generate_new_block()

        assigned_arm = self._current_block.pop(0)
        return assigned_arm

    def _stratified_assignment(self, stratification_values: dict[str, str]) -> str:
        """Perform stratified randomization assignment."""
        # Create stratum identifier
        stratum_parts = []
        for factor in self.config.stratification_factors:
            value = stratification_values.get(factor.name, "unknown")
            stratum_parts.append(f"{factor.name}:{value}")

        stratum_id = "_".join(stratum_parts)

        # Get or create stratum assignment list
        if stratum_id not in self._stratum_assignments:
            self._stratum_assignments[stratum_id] = []

        stratum_list = self._stratum_assignments[stratum_id]

        # If stratum list is empty, generate new assignments
        if not stratum_list:
            # Simple randomization within stratum
            arms = [self.config.control_arm] + self.config.treatment_arms
            # Generate enough assignments for a reasonable block
            for _ in range(20):  # Generate 20 assignments
                arm = np.random.choice(
                    arms, p=[self.config.allocation_ratio.get(arm, 1.0 / len(arms)) for arm in arms]
                )
                stratum_list.append(arm)

            np.random.shuffle(stratum_list)

        return stratum_list.pop(0)

    def _check_eligibility(self, baseline_data: dict[str, Any]) -> bool:
        """Check if subject meets eligibility criteria."""
        for criterion, required_value in self.config.eligibility_criteria.items():
            subject_value = baseline_data.get(criterion)

            if isinstance(required_value, dict):
                # Range criterion
                if "min" in required_value and subject_value < required_value["min"]:
                    return False
                if "max" in required_value and subject_value > required_value["max"]:
                    return False
            elif isinstance(required_value, list):
                # Categorical criterion
                if subject_value not in required_value:
                    return False
            else:
                # Exact match
                if subject_value != required_value:
                    return False

        return True

    def record_outcome(
        self,
        subject_id: str,
        outcome_name: str,
        value: Any,
        measurement_time: datetime | None = None,
    ):
        """
        Record an outcome for a subject.

        Args:
            subject_id: Subject identifier
            outcome_name: Name of the outcome
            value: Outcome value
            measurement_time: When outcome was measured
        """
        if subject_id not in self.subjects:
            raise ValueError(f"Subject {subject_id} not found")

        subject = self.subjects[subject_id]
        subject.outcomes[outcome_name] = {
            "value": value,
            "measurement_time": measurement_time or datetime.now(),
        }

    def record_dropout(self, subject_id: str, dropout_time: datetime | None = None):
        """Record subject dropout."""
        if subject_id not in self.subjects:
            raise ValueError(f"Subject {subject_id} not found")

        self.subjects[subject_id].dropout_time = dropout_time or datetime.now()

    def get_assignment_summary(self) -> dict[str, Any]:
        """Get summary of treatment assignments."""
        summary = {
            "total_subjects": len(self.subjects),
            "assignments_by_arm": {},
            "assignments_by_stratum": {},
        }

        # Count by arm
        for subject in self.subjects.values():
            arm = subject.assigned_arm
            summary["assignments_by_arm"][arm] = summary["assignments_by_arm"].get(arm, 0) + 1

        # Count by stratum (if stratified)
        if self.config.randomization_method == RandomizationMethod.STRATIFIED:
            for subject in self.subjects.values():
                # Create stratum identifier
                stratum_parts = []
                for factor in self.config.stratification_factors:
                    value = subject.stratification_values.get(factor.name, "unknown")
                    stratum_parts.append(f"{factor.name}:{value}")

                stratum_id = "_".join(stratum_parts)

                if stratum_id not in summary["assignments_by_stratum"]:
                    summary["assignments_by_stratum"][stratum_id] = {}

                arm = subject.assigned_arm
                stratum_counts = summary["assignments_by_stratum"][stratum_id]
                stratum_counts[arm] = stratum_counts.get(arm, 0) + 1

        return summary

    def analyze_balance(self) -> dict[str, Any]:
        """Analyze balance of baseline characteristics across arms."""
        if not self.subjects:
            return {"error": "No subjects assigned"}

        # Extract baseline data
        baseline_df = pd.DataFrame(
            [
                {"subject_id": sid, "assigned_arm": subject.assigned_arm, **subject.baseline_data}
                for sid, subject in self.subjects.items()
            ]
        )

        if baseline_df.empty:
            return {"error": "No baseline data available"}

        balance_results = {}

        # Analyze each baseline variable
        for column in baseline_df.columns:
            if column in ["subject_id", "assigned_arm"]:
                continue

            # Check if variable is numeric or categorical
            if pd.api.types.is_numeric_dtype(baseline_df[column]):
                # Numeric variable - compare means
                arm_stats = (
                    baseline_df.groupby("assigned_arm")[column]
                    .agg(["count", "mean", "std"])
                    .round(3)
                )

                # Simple ANOVA F-test
                try:
                    from scipy.stats import f_oneway

                    groups = [
                        group[column].dropna()
                        for name, group in baseline_df.groupby("assigned_arm")
                    ]
                    if len(groups) >= 2 and all(len(g) > 1 for g in groups):
                        f_stat, p_value = f_oneway(*groups)
                        balance_results[column] = {
                            "type": "numeric",
                            "arm_stats": arm_stats.to_dict(),
                            "f_statistic": f_stat,
                            "p_value": p_value,
                            "balanced": p_value > 0.05,
                        }
                except ImportError:
                    balance_results[column] = {
                        "type": "numeric",
                        "arm_stats": arm_stats.to_dict(),
                        "balanced": None,
                    }
            else:
                # Categorical variable - compare proportions
                crosstab = pd.crosstab(baseline_df["assigned_arm"], baseline_df[column])

                # Chi-square test
                try:
                    from scipy.stats import chi2_contingency

                    chi2, p_value, dof, expected = chi2_contingency(crosstab)
                    balance_results[column] = {
                        "type": "categorical",
                        "crosstab": crosstab.to_dict(),
                        "chi2_statistic": chi2,
                        "p_value": p_value,
                        "balanced": p_value > 0.05,
                    }
                except ImportError:
                    balance_results[column] = {
                        "type": "categorical",
                        "crosstab": crosstab.to_dict(),
                        "balanced": None,
                    }

        return balance_results

    def estimate_treatment_effects(self) -> dict[str, Any]:
        """Estimate treatment effects for all outcomes."""
        if not self.subjects:
            return {"error": "No subjects available"}

        # Extract outcome data
        outcome_data = []
        for subject in self.subjects.values():
            subject_data = {
                "subject_id": subject.id,
                "assigned_arm": subject.assigned_arm,
                "is_control": subject.assigned_arm == self.config.control_arm,
            }

            # Add outcomes
            for outcome_name, outcome_info in subject.outcomes.items():
                subject_data[outcome_name] = outcome_info["value"]

            outcome_data.append(subject_data)

        outcome_df = pd.DataFrame(outcome_data)

        if outcome_df.empty:
            return {"error": "No outcome data available"}

        results = {}

        # Analyze each outcome
        outcomes_to_analyze = [self.config.primary_outcome] + self.config.secondary_outcomes
        for outcome in outcomes_to_analyze:
            if outcome not in outcome_df.columns:
                continue

            # Treatment effect estimation
            control_data = outcome_df[outcome_df["assigned_arm"] == self.config.control_arm][
                outcome
            ].dropna()

            arm_results = {}
            for arm in self.config.treatment_arms:
                treatment_data = outcome_df[outcome_df["assigned_arm"] == arm][outcome].dropna()

                if len(control_data) > 0 and len(treatment_data) > 0:
                    # Calculate treatment effect
                    control_mean = control_data.mean()
                    treatment_mean = treatment_data.mean()
                    effect = treatment_mean - control_mean

                    # Calculate standard error and confidence interval
                    pooled_var = (
                        (len(control_data) - 1) * control_data.var()
                        + (len(treatment_data) - 1) * treatment_data.var()
                    ) / (len(control_data) + len(treatment_data) - 2)

                    se = np.sqrt(pooled_var * (1 / len(control_data) + 1 / len(treatment_data)))

                    # 95% confidence interval
                    from scipy.stats import t

                    df = len(control_data) + len(treatment_data) - 2
                    t_critical = t.ppf(0.975, df)
                    ci_lower = effect - t_critical * se
                    ci_upper = effect + t_critical * se

                    # T-test
                    try:
                        from scipy.stats import ttest_ind

                        t_stat, p_value = ttest_ind(treatment_data, control_data)
                    except ImportError:
                        t_stat, p_value = None, None

                    arm_results[arm] = {
                        "treatment_mean": treatment_mean,
                        "control_mean": control_mean,
                        "effect": effect,
                        "standard_error": se,
                        "confidence_interval": (ci_lower, ci_upper),
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "sample_sizes": {
                            "treatment": len(treatment_data),
                            "control": len(control_data),
                        },
                    }

            results[outcome] = arm_results

        return results

    def export_data(self, format: str = "dataframe") -> pd.DataFrame | dict[str, Any]:
        """
        Export experiment data.

        Args:
            format: Export format ("dataframe", "json", "dict")

        Returns:
            Exported data in specified format
        """
        # Prepare data
        export_data = []
        for subject in self.subjects.values():
            subject_record = {
                "subject_id": subject.id,
                "assigned_arm": subject.assigned_arm,
                "assignment_time": subject.assignment_time.isoformat(),
                "compliance": subject.compliance,
                **subject.baseline_data,
                **subject.stratification_values,
            }

            # Add outcomes
            for outcome_name, outcome_info in subject.outcomes.items():
                subject_record[f"{outcome_name}_value"] = outcome_info["value"]
                subject_record[f"{outcome_name}_time"] = outcome_info[
                    "measurement_time"
                ].isoformat()

            if subject.dropout_time:
                subject_record["dropout_time"] = subject.dropout_time.isoformat()

            export_data.append(subject_record)

        if format == "dataframe":
            return pd.DataFrame(export_data)
        elif format == "json":
            return json.dumps(export_data, indent=2)
        else:
            return export_data


def test():
    """Test the experiment manager."""
    print("=== Experiment Manager Demo ===\n")

    # Create experiment configuration
    config = ExperimentConfig(
        name="NFL Betting Strategy Experiment",
        experiment_type=ExperimentType.AB_TEST,
        randomization_method=RandomizationMethod.STRATIFIED,
        treatment_arms=["conservative_kelly", "aggressive_kelly"],
        control_arm="fixed_bet",
        target_sample_size=300,
        stratification_factors=[
            StratificationFactor(
                name="experience_level", levels=["novice", "intermediate", "expert"]
            ),
            StratificationFactor(name="bankroll_size", levels=["small", "medium", "large"]),
        ],
        minimum_effect_size=0.05,
        primary_outcome="profit_rate",
        secondary_outcomes=["num_bets", "max_drawdown"],
        eligibility_criteria={"min_age": 21, "min_bankroll": 1000, "location": ["US", "CA", "UK"]},
    )

    # Initialize experiment manager
    manager = ExperimentManager(config, random_seed=42)

    # Simulate subject enrollment
    print("Simulating subject enrollment...")

    experience_levels = ["novice", "intermediate", "expert"]
    bankroll_sizes = ["small", "medium", "large"]
    locations = ["US", "CA", "UK"]

    for i in range(150):  # Enroll 150 subjects
        subject_id = f"subject_{i:03d}"

        # Generate baseline data
        baseline_data = {
            "age": np.random.randint(21, 65),
            "min_bankroll": np.random.randint(1000, 10000),
            "location": np.random.choice(locations),
            "prior_betting_experience": np.random.randint(0, 10),
        }

        # Generate stratification values
        stratification_values = {
            "experience_level": np.random.choice(experience_levels),
            "bankroll_size": np.random.choice(bankroll_sizes),
        }

        try:
            assigned_arm = manager.assign_subject(subject_id, baseline_data, stratification_values)

            # Simulate outcomes based on treatment
            if assigned_arm == "fixed_bet":
                profit_rate = np.random.normal(0.02, 0.15)  # 2% average return
            elif assigned_arm == "conservative_kelly":
                profit_rate = np.random.normal(0.05, 0.18)  # 5% average return
            else:  # aggressive_kelly
                profit_rate = np.random.normal(0.08, 0.25)  # 8% average return

            num_bets = np.random.poisson(20)
            max_drawdown = np.random.uniform(0.05, 0.30)

            # Record outcomes
            manager.record_outcome(subject_id, "profit_rate", profit_rate)
            manager.record_outcome(subject_id, "num_bets", num_bets)
            manager.record_outcome(subject_id, "max_drawdown", max_drawdown)

        except ValueError as e:
            print(f"Skipped {subject_id}: {e}")

    # Analyze results
    print(f"\nEnrolled {len(manager.subjects)} subjects")

    # Assignment summary
    summary = manager.get_assignment_summary()
    print("\nAssignment Summary:")
    print(f"Total subjects: {summary['total_subjects']}")
    print("Assignments by arm:")
    for arm, count in summary["assignments_by_arm"].items():
        print(f"  {arm}: {count}")

    # Balance analysis
    print("\nBalance Analysis:")
    balance = manager.analyze_balance()
    for variable, results in balance.items():
        if "error" not in results:
            balanced = results.get("balanced", "unknown")
            p_val = results.get("p_value", "N/A")
            print(f"  {variable}: {'Balanced' if balanced else 'Imbalanced'} (p={p_val})")

    # Treatment effects
    print("\nTreatment Effect Analysis:")
    effects = manager.estimate_treatment_effects()
    if "error" not in effects:
        for outcome, arm_results in effects.items():
            print(f"\n{outcome}:")
            for arm, results in arm_results.items():
                effect = results["effect"]
                p_val = results.get("p_value", "N/A")
                ci = results["confidence_interval"]
                print(
                    f"  {arm} vs control: {effect:.4f} "
                    f"(95% CI: {ci[0]:.4f} to {ci[1]:.4f}, p={p_val})"
                )


if __name__ == "__main__":
    test()
