"""
Difference-in-Differences (DiD) Estimator

Implements two-way fixed effects regression for causal inference.
"""

import logging

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

logger = logging.getLogger(__name__)


class DiffInDiffEstimator:
    """Difference-in-Differences estimator with two-way fixed effects"""

    def __init__(self):
        self.model = None
        self.results = None

    def estimate(
        self,
        df: pd.DataFrame,
        outcome: str,
        treatment: str,
        unit_fe: str = 'player_id',
        time_fe: str = 'week',
        controls: list | None = None,
        cluster_var: str | None = None
    ) -> tuple[float, float, float, dict]:
        """
        Estimate DiD with two-way fixed effects.

        Model: Y_it = α + β*Treatment_it + γ_i + δ_t + X_it'θ + ε_it

        Args:
            df: Panel dataset
            outcome: Outcome variable name
            treatment: Treatment variable name (0/1)
            unit_fe: Unit fixed effects variable (e.g., player_id)
            time_fe: Time fixed effects variable (e.g., week)
            controls: List of control variables
            cluster_var: Variable to cluster standard errors

        Returns:
            (ate, se, pvalue, summary_dict)
        """
        # Build formula
        formula_parts = [outcome, '~', treatment]

        if controls:
            formula_parts.append('+')
            formula_parts.append(' + '.join(controls))

        # Add fixed effects
        formula_parts.append(f'+ C({unit_fe}) + C({time_fe})')

        formula = ' '.join(formula_parts)

        logger.info(f"DiD formula: {formula}")

        # Estimate model
        self.model = smf.ols(formula, data=df).fit()

        if cluster_var:
            # Cluster-robust standard errors
            self.results = self.model.get_robustcov_results(
                cov_type='cluster',
                groups=df[cluster_var]
            )
        else:
            self.results = self.model

        # Extract treatment effect
        ate = self.results.params[treatment]
        se = self.results.bse[treatment]
        pvalue = self.results.pvalues[treatment]

        summary = {
            'ate': ate,
            'se': se,
            'pvalue': pvalue,
            'ci_lower': ate - 1.96 * se,
            'ci_upper': ate + 1.96 * se,
            'n_obs': int(self.results.nobs),
            'r_squared': self.results.rsquared,
            'adj_r_squared': self.results.rsquared_adj
        }

        logger.info(f"DiD ATE: {ate:.3f} (SE: {se:.3f}, p={pvalue:.4f})")

        return ate, se, pvalue, summary

    def parallel_trends_test(
        self,
        df: pd.DataFrame,
        outcome: str,
        treatment: str,
        time_var: str = 'week',
        pre_periods: int = 4
    ) -> dict:
        """
        Test parallel trends assumption using event study.

        Args:
            df: Panel dataset
            outcome: Outcome variable
            treatment: Treatment variable
            time_var: Time variable
            pre_periods: Number of pre-treatment periods to test

        Returns:
            Dictionary with test results
        """
        # Create relative time indicators
        df = df.copy()
        df['rel_time'] = df[time_var] - df.groupby('player_id')[time_var].transform('min')

        # Test pre-treatment trends
        pre_df = df[df['rel_time'] < 0].tail(pre_periods)

        # Regress outcome on time for treatment and control groups
        treated = pre_df[pre_df[treatment] == 1]
        control = pre_df[pre_df[treatment] == 0]

        if len(treated) > 0 and len(control) > 0:
            trend_treated = np.polyfit(treated['rel_time'], treated[outcome], 1)[0]
            trend_control = np.polyfit(control['rel_time'], control[outcome], 1)[0]

            trend_diff = trend_treated - trend_control

            logger.info(f"Pre-treatment trend difference: {trend_diff:.4f}")

            return {
                'trend_treated': trend_treated,
                'trend_control': trend_control,
                'trend_diff': trend_diff,
                'parallel_trends': abs(trend_diff) < 0.5  # Rule of thumb
            }

        return {'error': 'Insufficient data for parallel trends test'}
