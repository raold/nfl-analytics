"""Feature Extractors Package.

This package provides the Strategy pattern implementation for feature extraction.
Each extractor implements the FeatureExtractorStrategy interface, allowing the
GameFeatureExtractor to orchestrate multiple extractors uniformly.

Architecture:
- base.py: Abstract base class defining the extractor interface
- registry.py: Registry for managing and orchestrating extractors
- Individual wrappers: Adapt existing extractors to the strategy pattern

Usage:
    from py.features.extractors import FeatureExtractorStrategy, FeatureRegistry
    from py.features.extractors.offensive import OffensiveFeatureExtractor

    # Register extractors
    registry = FeatureRegistry()
    registry.register(OffensiveFeatureExtractor())

    # Extract all features for a game
    features = registry.extract_all(home_team, away_team, season, week)

Created: December 13, 2025
Phase 2 of code quality remediation plan.
"""

from py.features.extractors.base import (
    FeatureExtractorStrategy,
    FeatureExtractionError,
    FeatureExtractionResult,
)
from py.features.extractors.registry import FeatureRegistry

__all__ = [
    'FeatureExtractorStrategy',
    'FeatureExtractionError',
    'FeatureExtractionResult',
    'FeatureRegistry',
]
