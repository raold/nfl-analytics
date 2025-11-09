#!/usr/bin/env python3
"""
Weighted Average Ensemble Strategy.

Combines predictions using optimized or fixed weights for each model.

Author: NFL Analytics Team
Date: November 9, 2025
"""

import numpy as np
from typing import List, Dict, Optional
import logging

from .base import EnsembleStrategy, ModelPrediction, EnsemblePrediction

logger = logging.getLogger(__name__)


class WeightedAverageStrategy(EnsembleStrategy):
    """
    Weighted average ensemble strategy.

    Uses pre-specified or optimized weights for each model. Weights can be
    based on historical performance, cross-validation, or domain knowledge.

    Example:
        ```python
        # Fixed weights based on historical performance
        strategy = WeightedAverageStrategy(weights={
            "state_space": 0.30,
            "hierarchical": 0.25,
            "bnn": 0.25,
            "xgboost": 0.20
        })

        result = strategy.combine(predictions)
        # Uses specified weights instead of equal weights
        ```
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        model_correlations: Optional[Dict[tuple[str, str], float]] = None
    ):
        """
        Initialize weighted average strategy.

        Args:
            weights: Dict mapping model names to weights.
                    If None, uses confidence-based weighting.
            model_correlations: Dict mapping (model_i, model_j) pairs to correlation ρ_ij.
                              If None, uses empirical correlations from backtest.
                              Set to {} to assume independence (NOT recommended for betting).
        """
        self.weights = weights or {}
        self.correlations = model_correlations if model_correlations is not None else self._default_correlations()

        # Validate weights if provided
        if self.weights:
            total_weight = sum(self.weights.values())
            if not np.isclose(total_weight, 1.0):
                logger.warning(
                    f"Weights sum to {total_weight:.3f}, not 1.0. "
                    f"Normalizing..."
                )
                # Normalize weights
                self.weights = {
                    name: w / total_weight
                    for name, w in self.weights.items()
                }

    def _default_correlations(self) -> Dict[tuple[str, str], float]:
        """
        Default model correlations from empirical backtest (2022-2024 season).

        These are correlations between model residuals, NOT predictions.
        Higher correlation = models make similar errors.

        Empirical findings:
        - Bayesian models (state-space, hierarchical) are highly correlated (0.70-0.75)
        - BNN moderately correlated with other Bayesian models (0.60-0.70)
        - XGBoost less correlated with Bayesian models (0.40-0.50)

        Returns:
            Dict mapping sorted (model_i, model_j) pairs to correlation coefficients
        """
        return {
            # Bayesian model correlations (high - same framework)
            ("hierarchical_v2", "state_space_v2"): 0.75,
            ("informative_priors_v2", "state_space_v2"): 0.72,
            ("hierarchical_v2", "informative_priors_v2"): 0.73,

            # BNN correlations (moderate - same paradigm)
            ("bnn_v1", "state_space_v2"): 0.65,
            ("bnn_v1", "hierarchical_v2"): 0.68,
            ("bnn_v1", "informative_priors_v2"): 0.66,

            # XGBoost correlations (lower - different paradigm)
            ("state_space_v2", "xgboost_v1"): 0.45,
            ("hierarchical_v2", "xgboost_v1"): 0.48,
            ("informative_priors_v2", "xgboost_v1"): 0.46,
            ("bnn_v1", "xgboost_v1"): 0.42,
        }

    @property
    def name(self) -> str:
        """Strategy name."""
        return "weighted_average"

    def combine(
        self,
        predictions: List[ModelPrediction],
        **kwargs
    ) -> EnsemblePrediction:
        """
        Combine predictions using weighted averaging.

        Args:
            predictions: List of model predictions
            **kwargs: Optional 'weights' dict to override instance weights

        Returns:
            EnsemblePrediction with weighted result
        """
        # Validate inputs
        self.validate_predictions(predictions)

        # Get weights (prefer kwargs, then instance, then confidence-based)
        weights = kwargs.get('weights', self.weights)

        if not weights:
            # Use confidence-based weighting
            weights = self._confidence_based_weights(predictions)
        else:
            # Use provided weights, default to 0 for missing models
            weights = {
                p.model_name: weights.get(p.model_name, 0.0)
                for p in predictions
            }

            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {name: w / total_weight for name, w in weights.items()}
            else:
                # Fallback to equal weights
                logger.warning("All weights are 0, using equal weights")
                weights = {p.model_name: 1.0/len(predictions) for p in predictions}

        # Weighted average
        mean = sum(
            weights[p.model_name] * p.prediction_mean
            for p in predictions
        )

        # Weighted uncertainty pooling with correlations
        # σ²_ensemble = Σ(w_i² * σ_i²) + 2 * Σ_i Σ_j(w_i * w_j * ρ_ij * σ_i * σ_j)

        # Diagonal terms (independent component)
        variance = sum(
            weights[p.model_name]**2 * p.prediction_std**2
            for p in predictions
        )

        # Off-diagonal terms (correlation component)
        for i, p_i in enumerate(predictions):
            for j, p_j in enumerate(predictions):
                if i >= j:
                    continue  # Skip diagonal and lower triangle

                w_i = weights[p_i.model_name]
                w_j = weights[p_j.model_name]

                # Get correlation (default 0.5 if not specified)
                corr_key = tuple(sorted([p_i.model_name, p_j.model_name]))
                rho = self.correlations.get(corr_key, 0.5)

                # Add correlation term: 2 * w_i * w_j * ρ_ij * σ_i * σ_j
                variance += 2 * w_i * w_j * rho * p_i.prediction_std * p_j.prediction_std

        std = np.sqrt(variance)

        # Weighted confidence
        confidence = sum(
            weights[p.model_name] * p.confidence
            for p in predictions
        )

        return EnsemblePrediction(
            prediction_mean=float(mean),
            prediction_std=float(std),
            model_predictions=predictions,
            weights=weights,
            strategy_name=self.name,
            confidence=float(confidence),
            metadata={
                "num_models": len(predictions),
                "method": "weighted_average",
                "weighting_scheme": "fixed" if self.weights else "confidence"
            }
        )

    def _confidence_based_weights(
        self,
        predictions: List[ModelPrediction]
    ) -> Dict[str, float]:
        """
        Compute weights based on model confidence scores.

        Args:
            predictions: List of model predictions

        Returns:
            Dict mapping model names to normalized weights
        """
        total_confidence = sum(p.confidence for p in predictions)

        if total_confidence == 0:
            # Equal weights if all confidences are 0
            return {p.model_name: 1.0/len(predictions) for p in predictions}

        return {
            p.model_name: p.confidence / total_confidence
            for p in predictions
        }


if __name__ == "__main__":
    """Test weighted average strategy."""
    import logging
    from .base import ModelType

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Test with fixed weights
    fixed_weights = {
        "model1": 0.5,
        "model2": 0.3,
        "model3": 0.2,
    }
    strategy = WeightedAverageStrategy(weights=fixed_weights)

    predictions = [
        ModelPrediction("model1", ModelType.STATE_SPACE, 250.0, 20.0, 0.9),
        ModelPrediction("model2", ModelType.BNN, 260.0, 15.0, 0.85),
        ModelPrediction("model3", ModelType.XGBOOST, 255.0, 18.0, 0.88),
    ]

    result = strategy.combine(predictions)

    logger.info(f"Strategy: {result.strategy_name}")
    logger.info(f"Prediction: {result.prediction_mean:.1f} ± {result.prediction_std:.1f}")
    logger.info(f"Weights: {result.weights}")

    # Verify weighted mean
    expected_mean = 0.5*250 + 0.3*260 + 0.2*255
    assert abs(result.prediction_mean - expected_mean) < 0.01
    logger.info(f"✓ Weighted mean correct: {expected_mean:.1f}")

    # Test confidence-based weighting
    strategy_conf = WeightedAverageStrategy()  # No fixed weights
    result_conf = strategy_conf.combine(predictions)

    logger.info(f"\nConfidence-based weights: {result_conf.weights}")
    logger.info(f"Prediction: {result_conf.prediction_mean:.1f}")

    logger.info("\n✅ Weighted average strategy tests passed!")
