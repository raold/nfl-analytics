"""
Validation and Sensitivity Analysis for Causal Inference
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CausalValidator:
    """Validation methods for causal estimates"""

    def placebo_test(
        self, df: pd.DataFrame, estimator, outcome: str, treatment: str, n_placebo: int = 100
    ) -> dict:
        """
        Placebo test: randomly assign treatment and check if effect is zero.

        Args:
            df: Panel data
            estimator: Estimator object with estimate() method
            outcome: Outcome variable
            treatment: Treatment variable
            n_placebo: Number of placebo iterations

        Returns:
            Dict with placebo distribution and p-value
        """
        # Get actual treatment effect
        actual_ate = estimator.estimate(df, outcome=outcome, treatment=treatment)[0]

        # Run placebo tests
        placebo_effects = []

        for i in range(n_placebo):
            df_placebo = df.copy()
            df_placebo[f"{treatment}_placebo"] = np.random.permutation(df[treatment].values)

            placebo_ate = estimator.estimate(
                df_placebo, outcome=outcome, treatment=f"{treatment}_placebo"
            )[0]

            placebo_effects.append(placebo_ate)

        placebo_effects = np.array(placebo_effects)

        # Calculate p-value
        p_value = np.mean(np.abs(placebo_effects) >= np.abs(actual_ate))

        logger.info(f"Placebo test: actual ATE={actual_ate:.3f}, p={p_value:.4f}")

        return {
            "actual_ate": actual_ate,
            "placebo_mean": placebo_effects.mean(),
            "placebo_std": placebo_effects.std(),
            "p_value": p_value,
            "placebo_effects": placebo_effects,
        }

    def sensitivity_to_unmeasured_confounding(
        self,
        ate: float,
        se: float,
        r2_confounder_outcome: float = 0.1,
        r2_confounder_treatment: float = 0.1,
    ) -> dict:
        """
        Rosenbaum bounds sensitivity analysis.

        How strong would an unmeasured confounder need to be to change conclusions?

        Args:
            ate: Estimated treatment effect
            se: Standard error
            r2_confounder_outcome: R^2 of confounder on outcome
            r2_confounder_treatment: R^2 of confounder on treatment

        Returns:
            Sensitivity bounds
        """
        # Simplified sensitivity bound
        # Full implementation would use Rosenbaum's Gamma
        bias = np.sqrt(r2_confounder_outcome * r2_confounder_treatment) * ate

        lower_bound = ate - bias - 1.96 * se
        upper_bound = ate + bias + 1.96 * se

        robust = not (lower_bound < 0 < upper_bound)

        logger.info(f"Sensitivity: ATE={ate:.3f}, bias={bias:.3f}, robust={robust}")

        return {
            "ate": ate,
            "potential_bias": bias,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "robust_to_confounding": robust,
        }

    def heterogeneous_effects_by_subgroup(
        self, df: pd.DataFrame, estimator, outcome: str, treatment: str, subgroup_var: str
    ) -> pd.DataFrame:
        """
        Estimate treatment effects by subgroup.

        Args:
            df: Panel data
            estimator: Estimator with estimate() method
            outcome: Outcome variable
            treatment: Treatment variable
            subgroup_var: Variable to stratify by

        Returns:
            DataFrame with subgroup-specific estimates
        """
        results = []

        for subgroup in df[subgroup_var].unique():
            df_sub = df[df[subgroup_var] == subgroup]

            if df_sub[treatment].sum() > 5:  # Need sufficient treated units
                ate, se, pval, summary = estimator.estimate(
                    df_sub, outcome=outcome, treatment=treatment
                )

                results.append(
                    {
                        "subgroup": subgroup,
                        "ate": ate,
                        "se": se,
                        "pvalue": pval,
                        "n_obs": summary["n_obs"],
                    }
                )

        results_df = pd.DataFrame(results)

        logger.info(f"Heterogeneous effects across {len(results)} subgroups")

        return results_df
