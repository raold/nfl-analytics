"""
Synthetic Control Method

Constructs synthetic controls for causal inference when only one unit is treated.
"""

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class SyntheticControl:
    """Synthetic Control Method estimator"""

    def __init__(self):
        self.weights = None
        self.treated_unit = None
        self.control_units = None

    def fit(
        self,
        df: pd.DataFrame,
        treated_unit: str,
        outcome: str,
        time_var: str = "week",
        unit_var: str = "team",
        treatment_time: int = None,
        predictors: list[str] | None = None,
    ) -> dict:
        """
        Fit synthetic control by finding optimal weights for control units.

        Args:
            df: Panel dataset
            treated_unit: Name of treated unit
            outcome: Outcome variable
            time_var: Time variable
            unit_var: Unit variable
            treatment_time: Time when treatment occurred
            predictors: Pre-treatment predictor variables

        Returns:
            Dictionary with weights and fit metrics
        """
        self.treated_unit = treated_unit

        # Split into pre and post treatment
        pre_treatment = df[df[time_var] < treatment_time]
        post_treatment = df[df[time_var] >= treatment_time]

        # Get treated and control outcomes
        treated_pre = pre_treatment[pre_treatment[unit_var] == treated_unit][outcome].values
        control_units_list = pre_treatment[pre_treatment[unit_var] != treated_unit][
            unit_var
        ].unique()
        self.control_units = control_units_list

        # Build control matrix (time x units)
        control_matrix = []
        for unit in control_units_list:
            unit_data = pre_treatment[pre_treatment[unit_var] == unit][outcome].values
            if len(unit_data) == len(treated_pre):
                control_matrix.append(unit_data)

        control_matrix = np.array(control_matrix).T  # time x units

        # Optimize weights to minimize pre-treatment fit
        def objective(weights):
            synthetic = control_matrix @ weights
            return np.sum((treated_pre - synthetic) ** 2)

        # Constraints: weights sum to 1 and are non-negative
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(control_units_list))]
        x0 = np.ones(len(control_units_list)) / len(control_units_list)

        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)

        self.weights = dict(zip(control_units_list, result.x))

        # Calculate pre-treatment RMSPE
        synthetic_pre = control_matrix @ result.x
        rmspe_pre = np.sqrt(np.mean((treated_pre - synthetic_pre) ** 2))

        logger.info(f"Synthetic control fit - RMSPE: {rmspe_pre:.3f}")
        logger.info(f"Non-zero weights: {sum(result.x > 0.01)}/{len(control_units_list)}")

        return {
            "weights": self.weights,
            "rmspe_pre": rmspe_pre,
            "optimization_success": result.success,
        }

    def estimate_treatment_effect(
        self, df: pd.DataFrame, outcome: str, time_var: str, unit_var: str, treatment_time: int
    ) -> pd.DataFrame:
        """
        Estimate treatment effects post-treatment.

        Returns:
            DataFrame with actual, synthetic, and treatment effect by time
        """
        if self.weights is None:
            raise ValueError("Must call fit() first")

        post_treatment = df[df[time_var] >= treatment_time]

        # Get treated outcomes
        treated_post = post_treatment[post_treatment[unit_var] == self.treated_unit][
            [time_var, outcome]
        ]
        treated_post = treated_post.rename(columns={outcome: "actual"})

        # Build synthetic control
        synthetic_vals = []
        for t in treated_post[time_var]:
            synthetic_t = 0
            for unit, weight in self.weights.items():
                unit_val = post_treatment[
                    (post_treatment[unit_var] == unit) & (post_treatment[time_var] == t)
                ][outcome]
                if len(unit_val) > 0:
                    synthetic_t += weight * unit_val.values[0]
            synthetic_vals.append(synthetic_t)

        treated_post["synthetic"] = synthetic_vals
        treated_post["treatment_effect"] = treated_post["actual"] - treated_post["synthetic"]

        logger.info(f"Average treatment effect: {treated_post['treatment_effect'].mean():.3f}")

        return treated_post
