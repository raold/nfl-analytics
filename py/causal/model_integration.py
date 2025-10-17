#!/usr/bin/env python3
"""
Causal Inference Integration with Predictive Models

Integrates causal inference results with BNN and prop prediction models:
- Adds causal adjustment features to model inputs
- Generates causal priors for Bayesian models
- Creates counterfactual-adjusted predictions
- Enhances betting edges with causal estimates

Key integrations:
1. BNN feature engineering: Add deconfounded features
2. Treatment effect priors: Use DiD/SC estimates as Bayesian priors
3. Counterfactual predictions: "What if player X was healthy?"
4. Shock event adjustments: Coaching changes, injuries, trades
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from causal.panel_constructor import PanelConstructor
from causal.treatment_definitions import TreatmentDefiner
from causal.confounder_identification import ConfounderIdentifier
from causal.synthetic_control import SyntheticControl
from causal.diff_in_diff import DifferenceInDifferences

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CausalModelIntegrator:
    """
    Integrates causal inference with predictive models.

    Provides methods to:
    - Add causal features to model inputs
    - Adjust predictions for treatment effects
    - Generate counterfactual predictions
    - Create causal priors for Bayesian models
    """

    def __init__(self, db_connection=None):
        """
        Args:
            db_connection: Database connection for panel data
        """
        self.db_connection = db_connection
        self.panel_constructor = PanelConstructor(db_connection) if db_connection else None
        self.treatment_definer = TreatmentDefiner()
        self.confounder_identifier = ConfounderIdentifier()

        # Cache for treatment effects
        self.injury_effects = {}
        self.coaching_change_effects = {}
        self.trade_effects = {}

    def add_causal_features(
        self,
        df: pd.DataFrame,
        feature_set: str = 'standard'
    ) -> pd.DataFrame:
        """
        Add causal inference features to dataframe for model training.

        Args:
            df: Feature dataframe
            feature_set: Which causal features to add
                - 'standard': Basic treatment indicators
                - 'propensity': Propensity score adjusted features
                - 'counterfactual': Synthetic control based features

        Returns:
            DataFrame with additional causal features
        """
        df_causal = df.copy()

        if feature_set in ['standard', 'propensity', 'counterfactual']:
            # Add treatment indicators
            df_causal = self.treatment_definer.define_injury_treatment(df_causal)

            # Add deconfounded features
            df_causal = self._add_deconfounded_features(df_causal)

        if feature_set in ['propensity', 'counterfactual']:
            # Add propensity scores
            df_causal = self._add_propensity_features(df_causal)

        if feature_set == 'counterfactual':
            # Add synthetic control features
            df_causal = self._add_counterfactual_features(df_causal)

        logger.info(f"Added {feature_set} causal features: {df_causal.shape[1] - df.shape[1]} new columns")

        return df_causal

    def _add_deconfounded_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features adjusted for confounding.

        Uses residualization to remove confounding effects.
        """
        df_deconf = df.copy()

        # Example: Deconfound carries (usage) by player ability
        if 'stat_carries' in df.columns and 'recent_avg_yards' in df.columns:
            # Use recent performance as proxy for ability
            from sklearn.linear_model import LinearRegression

            valid_idx = df[['stat_carries', 'recent_avg_yards']].notna().all(axis=1)

            if valid_idx.sum() > 50:
                lr = LinearRegression()
                lr.fit(
                    df.loc[valid_idx, ['recent_avg_yards']],
                    df.loc[valid_idx, 'stat_carries']
                )

                # Residual = actual - predicted (removes ability confounding)
                df_deconf['deconfounded_carries'] = np.nan
                df_deconf.loc[valid_idx, 'deconfounded_carries'] = (
                    df.loc[valid_idx, 'stat_carries'] -
                    lr.predict(df.loc[valid_idx, ['recent_avg_yards']])
                )

        return df_deconf

    def _add_propensity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add propensity score based features.

        Propensity score = P(Treatment | Covariates)
        Used for inverse probability weighting or matching.
        """
        df_ps = df.copy()

        if 'injury_treatment' in df.columns:
            # Compute propensity to be injured
            covariates = [
                col for col in ['stat_carries', 'recent_avg_yards', 'player_age', 'games_played']
                if col in df.columns
            ]

            if len(covariates) > 0:
                from sklearn.linear_model import LogisticRegression

                valid_idx = df[covariates + ['injury_treatment']].notna().all(axis=1)

                if valid_idx.sum() > 100:
                    lr = LogisticRegression(max_iter=1000)
                    lr.fit(
                        df.loc[valid_idx, covariates],
                        df.loc[valid_idx, 'injury_treatment']
                    )

                    # Propensity scores
                    df_ps['injury_propensity'] = np.nan
                    df_ps.loc[valid_idx, 'injury_propensity'] = lr.predict_proba(
                        df.loc[valid_idx, covariates]
                    )[:, 1]

                    # Inverse probability weights
                    df_ps['ipw_weight'] = 1.0
                    treated = df_ps['injury_treatment'] == 1
                    df_ps.loc[treated, 'ipw_weight'] = 1 / df_ps.loc[treated, 'injury_propensity']
                    df_ps.loc[~treated, 'ipw_weight'] = 1 / (1 - df_ps.loc[~treated, 'injury_propensity'])

                    # Clip extreme weights
                    df_ps['ipw_weight'] = df_ps['ipw_weight'].clip(0.1, 10)

        return df_ps

    def _add_counterfactual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add synthetic control based counterfactual features.

        For treated units, adds "what would have happened without treatment"
        """
        df_cf = df.copy()

        # This is a placeholder - would need full panel data
        # In practice, compute synthetic controls offline and join results

        logger.info("Counterfactual features require panel data - using placeholder")

        return df_cf

    def estimate_treatment_effect_priors(
        self,
        treatment_type: str,
        outcome: str = 'rushing_yards'
    ) -> Dict[str, float]:
        """
        Estimate treatment effect using causal inference.

        Returns prior distributions for Bayesian models.

        Args:
            treatment_type: 'injury', 'coaching_change', 'trade', etc.
            outcome: Outcome variable

        Returns:
            Dictionary with prior mean and std for treatment effect
        """
        if treatment_type == 'injury':
            # Use cached injury effects or estimate new
            if outcome in self.injury_effects:
                return self.injury_effects[outcome]

            # Placeholder: In practice, run DiD or SC analysis
            # Using literature estimates
            prior_mean = -15.0  # Injuries reduce yards by ~15 on average
            prior_std = 8.0

            self.injury_effects[outcome] = {
                'prior_mean': prior_mean,
                'prior_std': prior_std,
                'method': 'literature_estimate'
            }

        elif treatment_type == 'coaching_change':
            if outcome in self.coaching_change_effects:
                return self.coaching_change_effects[outcome]

            prior_mean = 5.0  # Coaching changes have small positive effect
            prior_std = 12.0

            self.coaching_change_effects[outcome] = {
                'prior_mean': prior_mean,
                'prior_std': prior_std,
                'method': 'literature_estimate'
            }

        elif treatment_type == 'trade':
            if outcome in self.trade_effects:
                return self.trade_effects[outcome]

            prior_mean = -8.0  # Trades typically hurt performance short-term
            prior_std = 10.0

            self.trade_effects[outcome] = {
                'prior_mean': prior_mean,
                'prior_std': prior_std,
                'method': 'literature_estimate'
            }

        else:
            raise ValueError(f"Unknown treatment type: {treatment_type}")

        return self.injury_effects.get(outcome) or self.coaching_change_effects.get(outcome) or self.trade_effects.get(outcome)

    def adjust_prediction_for_shock(
        self,
        base_prediction: float,
        shock_type: str,
        shock_magnitude: float = 1.0,
        uncertainty_inflation: float = 1.5
    ) -> Dict[str, float]:
        """
        Adjust prediction for shock events (injuries, coaching changes, etc.)

        Args:
            base_prediction: Original model prediction
            shock_type: Type of shock event
            shock_magnitude: Severity of shock (1.0 = typical)
            uncertainty_inflation: How much to inflate prediction uncertainty

        Returns:
            Dictionary with adjusted prediction and uncertainty
        """
        # Get treatment effect prior
        treatment_prior = self.estimate_treatment_effect_priors(shock_type)

        # Adjust prediction
        adjustment = treatment_prior['prior_mean'] * shock_magnitude
        adjusted_prediction = base_prediction + adjustment

        # Inflate uncertainty
        base_uncertainty = treatment_prior['prior_std']
        adjusted_uncertainty = base_uncertainty * uncertainty_inflation

        logger.info(f"Shock adjustment ({shock_type}): {base_prediction:.1f} → {adjusted_prediction:.1f} "
                   f"(±{adjusted_uncertainty:.1f})")

        return {
            'adjusted_prediction': adjusted_prediction,
            'adjustment': adjustment,
            'base_prediction': base_prediction,
            'uncertainty': adjusted_uncertainty,
            'shock_type': shock_type
        }

    def generate_counterfactual_prediction(
        self,
        player_id: str,
        week: int,
        season: int,
        intervention: Dict[str, Any]
    ) -> Dict:
        """
        Generate counterfactual prediction with interventions.

        Args:
            player_id: Player to predict for
            week: Week number
            season: Season
            intervention: Dictionary of interventions, e.g.,
                {'injury_status': 'healthy', 'opponent_defense_rank': 20}

        Returns:
            Dictionary with counterfactual prediction
        """
        logger.info(f"Generating counterfactual for {player_id} week {week}")

        # This is a placeholder - would integrate with actual BNN prediction
        # In practice:
        # 1. Load player's actual features
        # 2. Apply interventions (do-calculus)
        # 3. Run BNN prediction with intervened features
        # 4. Compare to factual prediction

        counterfactual = {
            'player_id': player_id,
            'week': week,
            'season': season,
            'intervention': intervention,
            'counterfactual_prediction': None,  # Would come from BNN
            'factual_prediction': None,  # Actual prediction
            'causal_effect': None  # Difference
        }

        logger.warning("Counterfactual prediction requires BNN integration - returning placeholder")

        return counterfactual

    def compute_betting_edge_with_causality(
        self,
        prediction: float,
        market_line: float,
        causal_adjustments: Optional[Dict] = None
    ) -> Dict:
        """
        Compute betting edge incorporating causal adjustments.

        Args:
            prediction: Model prediction
            market_line: Vegas line
            causal_adjustments: Optional dict with shock adjustments

        Returns:
            Dictionary with edge calculations
        """
        # Base edge (without causal adjustment)
        base_edge = prediction - market_line

        # Apply causal adjustments if provided
        if causal_adjustments:
            adjusted_pred = prediction

            for shock_type, magnitude in causal_adjustments.items():
                adjustment = self.adjust_prediction_for_shock(
                    adjusted_pred,
                    shock_type,
                    magnitude
                )
                adjusted_pred = adjustment['adjusted_prediction']

            adjusted_edge = adjusted_pred - market_line

            return {
                'base_prediction': prediction,
                'adjusted_prediction': adjusted_pred,
                'market_line': market_line,
                'base_edge': base_edge,
                'adjusted_edge': adjusted_edge,
                'edge_improvement': adjusted_edge - base_edge,
                'causal_adjustments': causal_adjustments
            }

        else:
            return {
                'prediction': prediction,
                'market_line': market_line,
                'edge': base_edge
            }

    def identify_high_leverage_events(
        self,
        upcoming_games: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Identify games with high causal leverage (shocks, counterfactuals).

        These are situations where causal inference provides the most value:
        - Recent injuries to key players
        - Coaching changes
        - Player trades
        - Weather shocks
        - Unexpected lineup changes

        Args:
            upcoming_games: DataFrame with upcoming game information

        Returns:
            DataFrame with leverage scores
        """
        df_leverage = upcoming_games.copy()

        # Initialize leverage score
        df_leverage['causal_leverage_score'] = 0.0

        # Add points for various shock events
        if 'injury_treatment' in df_leverage.columns:
            df_leverage['causal_leverage_score'] += df_leverage['injury_treatment'] * 3.0

        if 'post_coaching_change' in df_leverage.columns:
            weeks_since = df_leverage.get('weeks_since_coaching_change', 999)
            # Higher leverage soon after coaching change
            df_leverage['causal_leverage_score'] += (
                df_leverage['post_coaching_change'] * (5 - np.minimum(weeks_since, 5))
            )

        if 'post_trade' in df_leverage.columns:
            weeks_since_trade = df_leverage.get('weeks_since_trade', 999)
            df_leverage['causal_leverage_score'] += (
                df_leverage['post_trade'] * (4 - np.minimum(weeks_since_trade, 4))
            )

        # Sort by leverage
        df_leverage = df_leverage.sort_values('causal_leverage_score', ascending=False)

        logger.info(f"Identified {(df_leverage['causal_leverage_score'] > 2).sum()} high-leverage events")

        return df_leverage


def main():
    """Example usage of causal model integration"""

    print("\n" + "="*80)
    print("CAUSAL MODEL INTEGRATION")
    print("="*80)

    integrator = CausalModelIntegrator()

    # Example 1: Treatment effect priors
    print("\n" + "="*80)
    print("TREATMENT EFFECT PRIORS FOR BAYESIAN MODELS")
    print("="*80)

    injury_prior = integrator.estimate_treatment_effect_priors('injury', 'rushing_yards')
    print(f"\nInjury effect on rushing yards:")
    print(f"  Prior mean: {injury_prior['prior_mean']:.1f} yards")
    print(f"  Prior std: {injury_prior['prior_std']:.1f} yards")

    coaching_prior = integrator.estimate_treatment_effect_priors('coaching_change', 'rushing_yards')
    print(f"\nCoaching change effect:")
    print(f"  Prior mean: {coaching_prior['prior_mean']:.1f} yards")
    print(f"  Prior std: {coaching_prior['prior_std']:.1f} yards")

    # Example 2: Shock adjustment
    print("\n" + "="*80)
    print("PREDICTION ADJUSTMENT FOR SHOCK EVENTS")
    print("="*80)

    base_pred = 85.0  # Base prediction: 85 rushing yards
    print(f"\nBase prediction: {base_pred} yards")

    adjusted = integrator.adjust_prediction_for_shock(
        base_prediction=base_pred,
        shock_type='injury',
        shock_magnitude=1.0
    )

    print(f"After injury adjustment: {adjusted['adjusted_prediction']:.1f} yards")
    print(f"Adjustment: {adjusted['adjustment']:.1f} yards")
    print(f"Uncertainty: ±{adjusted['uncertainty']:.1f} yards")

    # Example 3: Betting edge with causality
    print("\n" + "="*80)
    print("BETTING EDGE WITH CAUSAL ADJUSTMENTS")
    print("="*80)

    market_line = 72.5
    print(f"\nMarket line: {market_line} yards")
    print(f"Model prediction: {base_pred} yards")

    edge = integrator.compute_betting_edge_with_causality(
        prediction=base_pred,
        market_line=market_line,
        causal_adjustments={'injury': 0.5}  # Minor injury
    )

    print(f"\nBase edge: {edge['base_edge']:.1f} yards")
    print(f"Adjusted edge: {edge['adjusted_edge']:.1f} yards")
    print(f"Edge improvement: {edge['edge_improvement']:.1f} yards")

    # Example 4: High leverage events
    print("\n" + "="*80)
    print("HIGH LEVERAGE EVENTS (Where Causal Inference Adds Most Value)")
    print("="*80)

    # Mock upcoming games
    mock_games = pd.DataFrame({
        'player_name': ['Player A', 'Player B', 'Player C', 'Player D'],
        'injury_treatment': [1, 0, 0, 0],
        'post_coaching_change': [0, 1, 0, 0],
        'weeks_since_coaching_change': [np.nan, 2, np.nan, np.nan],
        'post_trade': [0, 0, 1, 0],
        'weeks_since_trade': [np.nan, np.nan, 1, np.nan]
    })

    leverage_games = integrator.identify_high_leverage_events(mock_games)

    print("\n" + f"{'Player':<15} {'Leverage Score':<15} {'Events':<30}")
    print("-" * 60)
    for _, row in leverage_games.iterrows():
        events = []
        if row['injury_treatment']:
            events.append('Injury')
        if row['post_coaching_change']:
            events.append(f"Coaching (wk {row['weeks_since_coaching_change']:.0f})")
        if row['post_trade']:
            events.append(f"Trade (wk {row['weeks_since_trade']:.0f})")

        print(f"{row['player_name']:<15} {row['causal_leverage_score']:<15.1f} {', '.join(events):<30}")

    print("\n" + "="*80)
    print("These games have highest value from causal analysis!")
    print("="*80)


if __name__ == "__main__":
    main()
