"""
NFL Analytics Feature Engineering Module
=========================================

Unified feature extraction for NFL game and player predictions.

Core Components:
- GameFeatureExtractor: Main entry point for game-level features (305 features)
- SyntheticInteractionEngine: QB-OL synergy, defensive balance, explosive mismatch (Group A)
- SyntheticMomentumEngine: Form intensity, pressure consistency, RZ efficiency (Group B)
- ResearchDrivenEngine: nfelo EPA weighting, coaching rationality, injury curves

Usage:
    from nfl_analytics.features import GameFeatureExtractor
    from nfl_analytics.features.synthetic_interaction_features import SyntheticInteractionEngine
    from nfl_analytics.features.synthetic_momentum_features import SyntheticMomentumEngine
    from nfl_analytics.features.research_driven_features import ResearchDrivenEngine

Feature Count:
- Base features: 305 (via GameFeatureExtractor)
- Synthetic interaction: +11 features
- Synthetic momentum: +11 features
- Research-driven: +15 features
- Total available: ~342 features
"""

# Core extractors
from nfl_analytics.features.game_feature_extractor import GameFeatureExtractor

# Synthetic feature engines
from nfl_analytics.features.synthetic_interaction_features import (
    SyntheticInteractionEngine,
    InteractionFeatures,
)
from nfl_analytics.features.synthetic_momentum_features import (
    SyntheticMomentumEngine,
    MomentumFeatures,
)
from nfl_analytics.features.research_driven_features import (
    ResearchDrivenEngine,
    ResearchFeatures,
)

__all__ = [
    # Core
    "GameFeatureExtractor",
    # Synthetic engines
    "SyntheticInteractionEngine",
    "SyntheticMomentumEngine",
    "ResearchDrivenEngine",
    # Feature containers
    "InteractionFeatures",
    "MomentumFeatures",
    "ResearchFeatures",
]
