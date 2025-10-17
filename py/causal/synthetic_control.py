#!/usr/bin/env python3
"""
Synthetic Control Method for Causal Inference

Implements the synthetic control method (Abadie et al. 2010) for estimating
causal effects in comparative case studies.

Key idea: Create a "synthetic" version of the treated unit as a weighted
combination of control units. The synthetic control mimics the treated unit's
pre-treatment trajectory and provides a counterfactual for post-treatment.

Applications in NFL:
- Player injury impact: What would team performance be without the injury?
- Coaching changes: What would have happened with the old coach?
- Trade effects: What would player performance be on old team?
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize, nnls
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticControl:
    """
    Implements synthetic control method for causal inference.

    Creates a synthetic control unit as a weighted average of untreated units
    that best matches the treated unit's pre-treatment characteristics.
    """

    def __init__(self, method: str = 'optimization'):
        """
        Args:
            method: Method for constructing synthetic control
                - 'optimization': Constrained optimization (Abadie et al.)
                - 'ridge': Ridge regression with regularization
                - 'elastic_net': Elastic net for sparse weights
        """
        self.method = method
        self.weights_ = None
        self.treated_unit_ = None
        self.control_units_ = None
        self.pre_treatment_fit_ = None
        self.treatment_effect_ = None

    def fit(
        self,
        df: pd.DataFrame,
        treated_unit_id: str,
        unit_col: str,
        time_col: str,
        outcome_col: str,
        treatment_time: int,
        predictor_cols: Optional[List[str]] = None,
        match_on_outcomes: bool = True
    ) -> 'SyntheticControl':
        """
        Fit synthetic control model.

        Args:
            df: Panel dataframe
            treated_unit_id: ID of treated unit
            unit_col: Column identifying units (e.g., 'player_id', 'team')
            time_col: Column identifying time (e.g., 'week', 'game_number')
            outcome_col: Outcome variable to match
            treatment_time: Time when treatment occurs
            predictor_cols: Additional covariates to match on
            match_on_outcomes: Whether to match on pre-treatment outcomes

        Returns:
            self (fitted model)
        """
        # Separate treated and control units
        self.treated_unit_ = treated_unit_id
        treated_df = df[df[unit_col] == treated_unit_id].sort_values(time_col)
        control_df = df[df[unit_col] != treated_unit_id].copy()

        # Get list of control units
        self.control_units_ = control_df[unit_col].unique()

        # Pre-treatment period
        pre_treated = treated_df[treated_df[time_col] < treatment_time]
        pre_control = control_df[control_df[time_col] < treatment_time]

        if len(pre_treated) == 0:
            raise ValueError(f"No pre-treatment observations for {treated_unit_id}")

        # Build outcome matrix for pre-treatment period
        # Rows = time periods, Cols = control units
        outcome_matrix = pre_control.pivot(
            index=time_col,
            columns=unit_col,
            values=outcome_col
        )

        # Get treated outcome trajectory
        treated_trajectory = pre_treated.set_index(time_col)[outcome_col]

        # Align indices
        common_times = outcome_matrix.index.intersection(treated_trajectory.index)
        if len(common_times) == 0:
            raise ValueError("No common time periods between treated and control units")

        outcome_matrix = outcome_matrix.loc[common_times]
        treated_trajectory = treated_trajectory.loc[common_times]

        # Handle missing values
        outcome_matrix = outcome_matrix.fillna(outcome_matrix.mean())
        treated_trajectory = treated_trajectory.fillna(treated_trajectory.mean())

        # Add covariates if specified
        if predictor_cols:
            # Average pre-treatment covariate values
            treated_covariates = pre_treated[predictor_cols].mean()
            control_covariates = pre_control.groupby(unit_col)[predictor_cols].mean()

            # Align control covariates with outcome matrix
            control_covariates = control_covariates.loc[outcome_matrix.columns]
        else:
            treated_covariates = None
            control_covariates = None

        # Compute weights
        self.weights_ = self._compute_weights(
            treated_trajectory.values,
            outcome_matrix.values,
            treated_covariates,
            control_covariates
        )

        # Store weights as Series
        self.weights_ = pd.Series(self.weights_, index=outcome_matrix.columns)

        # Compute pre-treatment fit
        synthetic_trajectory = outcome_matrix @ self.weights_
        self.pre_treatment_fit_ = {
            'treated': treated_trajectory,
            'synthetic': synthetic_trajectory,
            'rmse': np.sqrt(np.mean((treated_trajectory - synthetic_trajectory)**2)),
            'mae': np.mean(np.abs(treated_trajectory - synthetic_trajectory))
        }

        # Compute treatment effects
        post_treated = treated_df[treated_df[time_col] >= treatment_time]
        post_control = control_df[control_df[time_col] >= treatment_time]

        post_outcome_matrix = post_control.pivot(
            index=time_col,
            columns=unit_col,
            values=outcome_col
        )

        post_treated_trajectory = post_treated.set_index(time_col)[outcome_col]

        # Align indices
        common_post_times = post_outcome_matrix.index.intersection(post_treated_trajectory.index)
        if len(common_post_times) > 0:
            post_outcome_matrix = post_outcome_matrix.loc[common_post_times]
            post_treated_trajectory = post_treated_trajectory.loc[common_post_times]

            # Compute synthetic control for post-treatment
            post_outcome_matrix = post_outcome_matrix[self.weights_.index]
            post_outcome_matrix = post_outcome_matrix.fillna(post_outcome_matrix.mean())

            post_synthetic_trajectory = post_outcome_matrix @ self.weights_

            # Treatment effects
            treatment_effects = post_treated_trajectory - post_synthetic_trajectory

            self.treatment_effect_ = {
                'effects': treatment_effects,
                'average_effect': treatment_effects.mean(),
                'cumulative_effect': treatment_effects.sum(),
                'treated': post_treated_trajectory,
                'synthetic': post_synthetic_trajectory
            }
        else:
            logger.warning("No post-treatment observations found")
            self.treatment_effect_ = None

        # Log fit quality
        logger.info(f"Synthetic control fitted for {treated_unit_id}")
        logger.info(f"Pre-treatment RMSE: {self.pre_treatment_fit_['rmse']:.2f}")
        logger.info(f"Non-zero weights: {(self.weights_ > 0.01).sum()}/{len(self.weights_)}")

        if self.treatment_effect_:
            logger.info(f"Average treatment effect: {self.treatment_effect_['average_effect']:.2f}")

        return self

    def _compute_weights(
        self,
        treated_trajectory: np.ndarray,
        control_matrix: np.ndarray,
        treated_covariates: Optional[pd.Series] = None,
        control_covariates: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Compute synthetic control weights.

        Args:
            treated_trajectory: Pre-treatment outcomes for treated unit (T,)
            control_matrix: Pre-treatment outcomes for controls (T, N)
            treated_covariates: Pre-treatment covariates for treated unit
            control_covariates: Pre-treatment covariates for controls

        Returns:
            Weights for control units (N,)
        """
        n_controls = control_matrix.shape[1]

        if self.method == 'optimization':
            # Constrained optimization: min ||Y_treated - Y_control @ W||^2
            # s.t. W >= 0, sum(W) = 1

            def objective(w):
                synthetic = control_matrix @ w
                return np.sum((treated_trajectory - synthetic)**2)

            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Sum to 1
            ]
            bounds = [(0, 1) for _ in range(n_controls)]  # Non-negative

            # Initial guess: equal weights
            w0 = np.ones(n_controls) / n_controls

            result = minimize(
                objective,
                w0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )

            if not result.success:
                logger.warning(f"Optimization did not converge: {result.message}")

            weights = result.x

        elif self.method == 'nnls':
            # Non-negative least squares (doesn't enforce sum=1)
            weights, _ = nnls(control_matrix, treated_trajectory)
            weights = weights / weights.sum()  # Normalize

        elif self.method == 'ridge':
            # Ridge regression (allows negative weights)
            scaler = StandardScaler()
            control_scaled = scaler.fit_transform(control_matrix)
            treated_scaled = scaler.transform(treated_trajectory.reshape(-1, 1)).ravel()

            ridge = Ridge(alpha=1.0, fit_intercept=False)
            ridge.fit(control_scaled, treated_scaled)
            weights = ridge.coef_

            # Normalize to sum to 1
            weights = np.maximum(weights, 0)  # Make non-negative
            weights = weights / weights.sum()

        elif self.method == 'elastic_net':
            # Elastic net for sparse weights
            scaler = StandardScaler()
            control_scaled = scaler.fit_transform(control_matrix)
            treated_scaled = scaler.transform(treated_trajectory.reshape(-1, 1)).ravel()

            elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=False, positive=True)
            elastic.fit(control_scaled, treated_scaled)
            weights = elastic.coef_

            # Normalize
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = np.ones(n_controls) / n_controls

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return weights

    def placebo_test(
        self,
        df: pd.DataFrame,
        unit_col: str,
        time_col: str,
        outcome_col: str,
        treatment_time: int,
        n_placebos: Optional[int] = None
    ) -> Dict:
        """
        Run placebo tests on control units.

        Applies synthetic control to each control unit (as if it were treated)
        and compares its effect to the actual treated unit's effect.

        Args:
            df: Panel dataframe
            unit_col: Column identifying units
            time_col: Column identifying time
            outcome_col: Outcome variable
            treatment_time: Treatment time
            n_placebos: Number of placebo tests to run (default: all controls)

        Returns:
            Dictionary with placebo test results and p-value
        """
        if self.treatment_effect_ is None:
            raise ValueError("Must fit model with post-treatment data first")

        actual_effect = self.treatment_effect_['average_effect']

        # Run synthetic control on each control unit
        placebo_effects = []
        control_units = self.control_units_

        if n_placebos is not None:
            control_units = np.random.choice(control_units, size=min(n_placebos, len(control_units)), replace=False)

        for control_id in control_units:
            try:
                # Fit synthetic control for this placebo unit
                placebo_sc = SyntheticControl(method=self.method)
                placebo_sc.fit(
                    df=df,
                    treated_unit_id=control_id,
                    unit_col=unit_col,
                    time_col=time_col,
                    outcome_col=outcome_col,
                    treatment_time=treatment_time
                )

                if placebo_sc.treatment_effect_ is not None:
                    placebo_effect = placebo_sc.treatment_effect_['average_effect']
                    placebo_effects.append({
                        'unit': control_id,
                        'effect': placebo_effect,
                        'pre_rmse': placebo_sc.pre_treatment_fit_['rmse']
                    })

            except Exception as e:
                logger.debug(f"Placebo test failed for {control_id}: {e}")
                continue

        if len(placebo_effects) == 0:
            logger.warning("No successful placebo tests")
            return {'p_value': np.nan, 'rank': np.nan, 'placebo_effects': []}

        placebo_effects_df = pd.DataFrame(placebo_effects)

        # Compute p-value (proportion of placebos with effect >= actual effect)
        p_value = (np.abs(placebo_effects_df['effect']) >= np.abs(actual_effect)).mean()

        # Rank of actual effect
        rank = (np.abs(placebo_effects_df['effect']) >= np.abs(actual_effect)).sum() + 1
        total = len(placebo_effects_df) + 1

        logger.info(f"Placebo test: p-value = {p_value:.3f}, rank = {rank}/{total}")

        return {
            'p_value': p_value,
            'rank': rank,
            'total': total,
            'placebo_effects': placebo_effects_df,
            'actual_effect': actual_effect
        }

    def get_contribution_weights(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get top contributing control units with their weights.

        Args:
            top_n: Number of top contributors to return

        Returns:
            DataFrame with units and weights
        """
        if self.weights_ is None:
            raise ValueError("Must fit model first")

        weights_df = pd.DataFrame({
            'unit': self.weights_.index,
            'weight': self.weights_.values
        }).sort_values('weight', ascending=False)

        return weights_df.head(top_n)

    def plot_trajectories(self) -> Dict:
        """
        Generate data for plotting treated vs synthetic trajectories.

        Returns:
            Dictionary with pre and post treatment data for plotting
        """
        if self.pre_treatment_fit_ is None:
            raise ValueError("Must fit model first")

        plot_data = {
            'pre_treatment': {
                'time': self.pre_treatment_fit_['treated'].index.tolist(),
                'treated': self.pre_treatment_fit_['treated'].values.tolist(),
                'synthetic': self.pre_treatment_fit_['synthetic'].values.tolist()
            }
        }

        if self.treatment_effect_ is not None:
            plot_data['post_treatment'] = {
                'time': self.treatment_effect_['treated'].index.tolist(),
                'treated': self.treatment_effect_['treated'].values.tolist(),
                'synthetic': self.treatment_effect_['synthetic'].values.tolist(),
                'effects': self.treatment_effect_['effects'].values.tolist()
            }

        return plot_data

    def estimate_ate_from_multiple_units(
        self,
        df: pd.DataFrame,
        treated_unit_ids: List[str],
        unit_col: str,
        time_col: str,
        outcome_col: str,
        treatment_times: Dict[str, int]
    ) -> Dict:
        """
        Estimate average treatment effect from multiple treated units.

        Args:
            df: Panel dataframe
            treated_unit_ids: List of treated unit IDs
            unit_col: Column identifying units
            time_col: Column identifying time
            outcome_col: Outcome variable
            treatment_times: Dict mapping unit IDs to treatment times

        Returns:
            Dictionary with pooled treatment effects
        """
        effects = []

        for unit_id in treated_unit_ids:
            try:
                sc = SyntheticControl(method=self.method)
                sc.fit(
                    df=df,
                    treated_unit_id=unit_id,
                    unit_col=unit_col,
                    time_col=time_col,
                    outcome_col=outcome_col,
                    treatment_time=treatment_times[unit_id]
                )

                if sc.treatment_effect_ is not None:
                    effects.append({
                        'unit': unit_id,
                        'effect': sc.treatment_effect_['average_effect'],
                        'pre_rmse': sc.pre_treatment_fit_['rmse']
                    })

            except Exception as e:
                logger.warning(f"Synthetic control failed for {unit_id}: {e}")
                continue

        if len(effects) == 0:
            logger.warning("No successful synthetic controls")
            return {'ate': np.nan, 'std': np.nan, 'n_units': 0}

        effects_df = pd.DataFrame(effects)

        ate = effects_df['effect'].mean()
        std = effects_df['effect'].std()

        logger.info(f"Pooled ATE from {len(effects)} units: {ate:.2f} (±{std:.2f})")

        return {
            'ate': ate,
            'std': std,
            'n_units': len(effects),
            'effects': effects_df
        }


def main():
    """Example usage with NFL-like data"""

    # Simulate panel data: team performance over weeks
    np.random.seed(42)

    # 10 teams, 20 weeks
    teams = [f'Team_{i}' for i in range(10)]
    weeks = list(range(1, 21))

    data = []
    for team in teams:
        # Team-specific baseline
        baseline = np.random.normal(25, 5)

        for week in weeks:
            # Natural variation
            points = baseline + np.random.normal(0, 3)

            # Team 0 loses star player at week 10 (treatment)
            if team == 'Team_0' and week >= 10:
                points -= 8  # True effect: -8 points

            data.append({
                'team': team,
                'week': week,
                'points': points
            })

    df = pd.DataFrame(data)

    print("\n" + "="*80)
    print("SYNTHETIC CONTROL METHOD - NFL INJURY EXAMPLE")
    print("="*80)
    print("\nScenario: Team_0 loses star player at week 10")
    print("Question: What would Team_0's performance be without the injury?\n")

    # Fit synthetic control
    sc = SyntheticControl(method='optimization')
    sc.fit(
        df=df,
        treated_unit_id='Team_0',
        unit_col='team',
        time_col='week',
        outcome_col='points',
        treatment_time=10
    )

    # Show top contributors
    print("\n" + "="*80)
    print("TOP CONTROL UNITS (Synthetic Team_0 composition)")
    print("="*80)
    weights = sc.get_contribution_weights(top_n=5)
    print(weights.to_string(index=False))

    # Show treatment effects
    print("\n" + "="*80)
    print("TREATMENT EFFECTS")
    print("="*80)
    print(f"Average effect: {sc.treatment_effect_['average_effect']:.2f} points")
    print(f"Cumulative effect: {sc.treatment_effect_['cumulative_effect']:.2f} points")
    print(f"Pre-treatment fit RMSE: {sc.pre_treatment_fit_['rmse']:.2f}")

    # Placebo tests
    print("\n" + "="*80)
    print("PLACEBO TESTS (Statistical Significance)")
    print("="*80)
    placebo_results = sc.placebo_test(
        df=df,
        unit_col='team',
        time_col='week',
        outcome_col='points',
        treatment_time=10
    )
    print(f"P-value: {placebo_results['p_value']:.3f}")
    print(f"Rank: {placebo_results['rank']}/{placebo_results['total']}")
    print(f"Significant at 0.05 level: {placebo_results['p_value'] < 0.05}")

    # Plot data
    print("\n" + "="*80)
    print("POST-TREATMENT TRAJECTORY")
    print("="*80)
    if sc.treatment_effect_:
        plot_data = sc.plot_trajectories()
        post = plot_data['post_treatment']

        print(f"\n{'Week':<6} {'Actual':<10} {'Synthetic':<10} {'Effect':<10}")
        print("-" * 36)
        for i, week in enumerate(post['time']):
            print(f"{week:<6} {post['treated'][i]:<10.1f} {post['synthetic'][i]:<10.1f} {post['effects'][i]:<10.1f}")


if __name__ == "__main__":
    main()
