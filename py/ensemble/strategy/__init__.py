"""
Ensemble Strategy Pattern for NFL Analytics.

Consolidates 8 ensemble implementations into a unified, testable architecture.

Author: NFL Analytics Team
Date: November 9, 2025
"""

from .base import (
    EnsembleConfig,
    EnsemblePrediction,
    EnsembleStrategy,
    ModelPrediction,
)
from .ensemble_predictor import UnifiedEnsemblePredictor
from .simple_average import SimpleAverageStrategy
from .stacking import StackingStrategy
from .thompson_sampling import ThompsonSamplingStrategy
from .weighted_average import WeightedAverageStrategy

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
