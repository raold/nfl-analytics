"""
Model Integration for Causal Inference

Integrate causal estimates with BNN/props models.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class CausalModelIntegrator:
    """Integrate causal effects into predictive models"""

    def adjust_bnn_prior(
        self,
        baseline_mean: float,
        baseline_std: float,
        treatment_effect: float,
        treatment_effect_se: float,
        is_treated: bool
    ) -> dict[str, float]:
        """
        Adjust BNN prior based on causal treatment effect.

        Args:
            baseline_mean: Baseline prior mean
            baseline_std: Baseline prior std
            treatment_effect: Estimated ATE
            treatment_effect_se: Standard error of ATE
            is_treated: Whether unit received treatment

        Returns:
            Adjusted prior parameters
        """
        if is_treated:
            adjusted_mean = baseline_mean + treatment_effect

            # Increase uncertainty due to treatment effect uncertainty
            adjusted_std = np.sqrt(baseline_std**2 + treatment_effect_se**2)
        else:
            adjusted_mean = baseline_mean
            adjusted_std = baseline_std

        return {
            'mean': adjusted_mean,
            'std': adjusted_std,
            'treatment_adjusted': is_treated
        }

    def incorporate_live_treatment(
        self,
        player_id: str,
        game_id: str,
        baseline_prediction: float,
        treatment_effects: dict[str, float]
    ) -> dict:
        """
        Adjust live game prediction for observed treatments.

        Example:
            Player just suffered in-game injury (treatment)
            → Adjust remaining game prediction downward by ATE

        Args:
            player_id: Player ID
            game_id: Game ID
            baseline_prediction: Model baseline prediction
            treatment_effects: Dict of treatment -> ATE

        Returns:
            Adjusted prediction with explanation
        """
        adjusted_prediction = baseline_prediction
        adjustments = []

        for treatment, ate in treatment_effects.items():
            adjusted_prediction += ate
            adjustments.append(f"{treatment}: {ate:+.2f}")

        result = {
            'player_id': player_id,
            'game_id': game_id,
            'baseline_prediction': baseline_prediction,
            'adjusted_prediction': adjusted_prediction,
            'total_adjustment': adjusted_prediction - baseline_prediction,
            'adjustments': adjustments
        }

        logger.info(f"Adjusted prediction for {player_id}: {baseline_prediction:.1f} → {adjusted_prediction:.1f}")

        return result
