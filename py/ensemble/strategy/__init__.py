"""
Ensemble Strategy Pattern for NFL Analytics.

Consolidates 8 ensemble implementations into a unified, testable architecture.

Author: NFL Analytics Team
Date: November 9, 2025
"""

from .base import (
    EnsembleStrategy,
    ModelPrediction,
    EnsemblePrediction,
    EnsembleConfig,
)
from .simple_average import SimpleAverageStrategy
from .weighted_average import WeightedAverageStrategy
from .thompson_sampling import ThompsonSamplingStrategy
from .stacking import StackingStrategy
from .ensemble_predictor import UnifiedEnsemblePredictor

__all__ = [
    # Base classes
    "EnsembleStrategy",
    "ModelPrediction",
    "EnsemblePrediction",
    "EnsembleConfig",
    # Strategies
    "SimpleAverageStrategy",
    "WeightedAverageStrategy",
    "ThompsonSamplingStrategy",
    "StackingStrategy",
    # Main predictor
    "UnifiedEnsemblePredictor",
]
