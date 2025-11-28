#!/usr/bin/env python3
"""
Base Ensemble Strategy Pattern.

Defines the interface and data structures for ensemble prediction strategies.

Author: NFL Analytics Team
Date: November 9, 2025
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import numpy as np


# ============================================================================
# Canonical Ensemble Weight Presets
# ============================================================================

# v6.0 Weights - Optimized for game-level predictions (spreads, totals)
# Validated: 2022-2024 backtest, 54.5% cover rate, +4.0% ROI on >2pt edge games
# Optimized via CVX convex optimization on 2023 validation set
DEFAULT_WEIGHTS_V6 = {
    "state_space_v2": 0.27,
    "hierarchical_v2": 0.27,
    "xgboost_v4_ridge": 0.23,
    "informative_priors_v2": 0.13,
    "hybrid_static_incremental": 0.10,
}

# v8.0 Dirichlet Weights - Optimized for player-level predictions (props: yards, TDs)
# Learned via Bayesian stacking on 2022-2024 passing yards data (Nov 26, 2025)
# Grid search: 108 configs, best: prior_strength=1.0, likelihood_scale=20.0, scoring=squared
# Results: 31.5% MAE improvement (89.42 -> 61.26 yards) vs static v6.0 weights
DIRICHLET_WEIGHTS_V8 = {
    "state_space_v2": 0.082,
    "hierarchical_v2": 0.396,
    "xgboost_v4_ridge": 0.026,
    "informative_priors_v2": 0.395,
    "hybrid_static_incremental": 0.101,
}


class ModelType(Enum):
    """Supported model types."""
    STATE_SPACE = "state_space"
    HIERARCHICAL = "hierarchical"
    INFORMATIVE_PRIORS = "informative_priors"
    BNN = "bnn"
    XGBOOST = "xgboost"
    CUSTOM = "custom"


@dataclass
class ModelPrediction:
    """
    Prediction from a single model.

    Attributes:
        model_name: Name of the model (e.g., "state_space_v2")
        model_type: Type of model
        prediction_mean: Point prediction
        prediction_std: Prediction uncertainty (standard deviation)
        confidence: Model confidence score (0-1)
        metadata: Additional model-specific metadata
    """
    model_name: str
    model_type: ModelType
    prediction_mean: float
    prediction_std: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate prediction values."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        if self.prediction_std < 0:
            raise ValueError(f"Std must be >= 0, got {self.prediction_std}")


@dataclass
class EnsemblePrediction:
    """
    Combined ensemble prediction.

    Attributes:
        prediction_mean: Ensemble point prediction
        prediction_std: Ensemble uncertainty
        model_predictions: Individual model predictions
        weights: Model weights used in ensemble
        strategy_name: Name of ensemble strategy
        confidence: Overall confidence score
        metadata: Additional ensemble metadata
    """
    prediction_mean: float
    prediction_std: float
    model_predictions: List[ModelPrediction]
    weights: Dict[str, float]
    strategy_name: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_models(self) -> int:
        """Number of models in ensemble."""
        return len(self.model_predictions)

    def get_prediction_range(self, num_std: float = 1.0) -> tuple[float, float]:
        """
        Get prediction interval.

        Args:
            num_std: Number of standard deviations for interval

        Returns:
            (lower_bound, upper_bound) tuple
        """
        margin = num_std * self.prediction_std
        return (self.prediction_mean - margin, self.prediction_mean + margin)


@dataclass
class EnsembleConfig:
    """
    Configuration for ensemble predictor.

    Attributes:
        strategy_name: Name of ensemble strategy to use
        model_paths: Dict mapping model names to file paths
        weights: Dict mapping model names to weights (for weighted strategies)
        confidence_threshold: Minimum model confidence to include in ensemble
        require_minimum_models: Minimum number of models required for valid prediction
        cache_predictions: Whether to cache model predictions
        db_config: Database configuration dict
    """
    strategy_name: str = "weighted_average"
    model_paths: Dict[str, str] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    confidence_threshold: float = 0.5
    require_minimum_models: int = 2
    cache_predictions: bool = True
    db_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        Set default database config if not provided.

        SECURITY: Credentials come from environment variables.
        Never hardcode passwords!
        """
        if not self.db_config:
            import os
            self.db_config = {
                "host": os.getenv("DB_HOST", "localhost"),
                "port": int(os.getenv("DB_PORT", "5544")),
                "dbname": os.getenv("DB_NAME", "devdb01"),
                "user": os.getenv("DB_USER", "dro"),
                "password": os.getenv("DB_PASSWORD", ""),
            }

            if not self.db_config["password"]:
                raise ValueError(
                    "DB_PASSWORD environment variable is required. "
                    "See .env.example for configuration."
                )


class EnsembleStrategy(ABC):
    """
    Abstract base class for ensemble strategies.

    All ensemble strategies must implement the combine() method to specify
    how individual model predictions are combined into a final prediction.

    Example implementation:

        ```python
        class SimpleAverageStrategy(EnsembleStrategy):
            def combine(self, predictions: List[ModelPrediction]) -> EnsemblePrediction:
                # Simple average of predictions
                mean = np.mean([p.prediction_mean for p in predictions])
                std = np.mean([p.prediction_std for p in predictions])

                weights = {p.model_name: 1.0/len(predictions) for p in predictions}

                return EnsemblePrediction(
                    prediction_mean=mean,
                    prediction_std=std,
                    model_predictions=predictions,
                    weights=weights,
                    strategy_name=self.name
                )
        ```
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name (e.g., 'simple_average', 'weighted', 'thompson')."""
        pass

    @abstractmethod
    def combine(
        self,
        predictions: List[ModelPrediction],
        **kwargs
    ) -> EnsemblePrediction:
        """
        Combine individual model predictions into ensemble prediction.

        Args:
            predictions: List of predictions from individual models
            **kwargs: Strategy-specific parameters

        Returns:
            EnsemblePrediction with combined result

        Raises:
            ValueError: If predictions list is empty or invalid
        """
        pass

    def validate_predictions(self, predictions: List[ModelPrediction]) -> None:
        """
        Validate predictions before combining.

        Args:
            predictions: List of model predictions

        Raises:
            ValueError: If predictions are invalid
        """
        if not predictions:
            raise ValueError("Cannot combine empty predictions list")

        for pred in predictions:
            if np.isnan(pred.prediction_mean) or np.isinf(pred.prediction_mean):
                raise ValueError(
                    f"Invalid prediction from {pred.model_name}: "
                    f"mean={pred.prediction_mean}"
                )
            if np.isnan(pred.prediction_std) or pred.prediction_std < 0:
                raise ValueError(
                    f"Invalid prediction std from {pred.model_name}: "
                    f"std={pred.prediction_std}"
                )

    def filter_predictions(
        self,
        predictions: List[ModelPrediction],
        confidence_threshold: float = 0.5
    ) -> List[ModelPrediction]:
        """
        Filter predictions by confidence threshold.

        Args:
            predictions: List of model predictions
            confidence_threshold: Minimum confidence to include

        Returns:
            Filtered list of predictions
        """
        return [
            p for p in predictions
            if p.confidence >= confidence_threshold
        ]


if __name__ == "__main__":
    """Test data structures."""
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Test ModelPrediction
    pred = ModelPrediction(
        model_name="test_model",
        model_type=ModelType.BNN,
        prediction_mean=250.5,
        prediction_std=25.0,
        confidence=0.85,
        metadata={"training_samples": 1000}
    )
    logger.info(f"✓ ModelPrediction created: {pred.model_name}")

    # Test EnsemblePrediction
    ensemble = EnsemblePrediction(
        prediction_mean=255.0,
        prediction_std=20.0,
        model_predictions=[pred],
        weights={"test_model": 1.0},
        strategy_name="test_strategy"
    )
    logger.info(f"✓ EnsemblePrediction created: {ensemble.num_models} models")

    lower, upper = ensemble.get_prediction_range(num_std=2.0)
    logger.info(f"✓ 95% Prediction interval: [{lower:.1f}, {upper:.1f}]")

    # Test EnsembleConfig
    config = EnsembleConfig(
        strategy_name="weighted",
        weights={"model1": 0.6, "model2": 0.4}
    )
    logger.info(f"✓ EnsembleConfig created: strategy={config.strategy_name}")

    logger.info("\n✅ All data structure tests passed!")
