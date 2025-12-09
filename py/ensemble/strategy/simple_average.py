#!/usr/bin/env python3
"""
Simple Average Ensemble Strategy.

Combines predictions using equal weights for all models.

Author: NFL Analytics Team
Date: November 9, 2025
"""


import numpy as np

from .base import EnsemblePrediction, EnsembleStrategy, ModelPrediction


class SimpleAverageStrategy(EnsembleStrategy):
    """
    Simple average ensemble strategy.

    Combines model predictions using equal weights. This is the baseline
    ensemble approach and often performs surprisingly well.

    Example:
        ```python
        strategy = SimpleAverageStrategy()

        predictions = [
            ModelPrediction("model1", ModelType.BNN, 250, 20, 0.9),
            ModelPrediction("model2", ModelType.XGBOOST, 260, 15, 0.85),
        ]

        result = strategy.combine(predictions)
        # result.prediction_mean = 255.0  (average of 250 and 260)
        # result.weights = {"model1": 0.5, "model2": 0.5}
        ```
    """

    @property
    def name(self) -> str:
        """Strategy name."""
        return "simple_average"

    def combine(self, predictions: list[ModelPrediction], **kwargs) -> EnsemblePrediction:
        """
        Combine predictions using simple averaging.

        Args:
            predictions: List of model predictions
            **kwargs: Unused for this strategy

        Returns:
            EnsemblePrediction with averaged result
        """
        # Validate inputs
        self.validate_predictions(predictions)

        # Equal weights for all models
        weight = 1.0 / len(predictions)
        weights = {p.model_name: weight for p in predictions}

        # Average predictions
        mean = np.mean([p.prediction_mean for p in predictions])

        # Combine uncertainties (variance pooling)
        # σ²_ensemble = Σ(w_i² * σ_i²)
        std = np.sqrt(np.sum([weight**2 * p.prediction_std**2 for p in predictions]))

        # Average confidence
        confidence = np.mean([p.confidence for p in predictions])

        return EnsemblePrediction(
            prediction_mean=float(mean),
            prediction_std=float(std),
            model_predictions=predictions,
            weights=weights,
            strategy_name=self.name,
            confidence=float(confidence),
            metadata={"num_models": len(predictions), "method": "simple_average"},
        )


if __name__ == "__main__":
    """Test simple average strategy."""
    import logging

    from .base import ModelType

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    strategy = SimpleAverageStrategy()

    # Create test predictions
    predictions = [
        ModelPrediction("model1", ModelType.BNN, 250.0, 20.0, 0.9),
        ModelPrediction("model2", ModelType.XGBOOST, 260.0, 15.0, 0.85),
        ModelPrediction("model3", ModelType.STATE_SPACE, 255.0, 18.0, 0.88),
    ]

    result = strategy.combine(predictions)

    logger.info(f"Strategy: {result.strategy_name}")
    logger.info(f"Prediction: {result.prediction_mean:.1f} ± {result.prediction_std:.1f}")
    logger.info(f"Weights: {result.weights}")
    logger.info(f"Confidence: {result.confidence:.3f}")

    # Verify equal weights
    assert all(abs(w - 1 / 3) < 0.001 for w in result.weights.values())
    logger.info("✓ All weights equal (1/3)")

    # Verify mean is average
    expected_mean = (250 + 260 + 255) / 3
    assert abs(result.prediction_mean - expected_mean) < 0.01
    logger.info(f"✓ Mean is average: {expected_mean:.1f}")

    logger.info("\n✅ Simple average strategy tests passed!")
