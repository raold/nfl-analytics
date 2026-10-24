"""Feature Extractors Package.

This package provides the Strategy pattern implementation for feature extraction.
Each extractor implements the FeatureExtractorStrategy interface, allowing the
GameFeatureExtractor to orchestrate multiple extractors uniformly.

Architecture:
- base.py: Abstract base class defining the extractor interface
- registry.py: Registry for managing and orchestrating extractors
- Individual wrappers: Adapt existing extractors to the strategy pattern

Usage:
    from nfl_analytics.features.extractors import (
        FeatureRegistry,
        OffensiveFeatureExtractor,
        DefensiveFeatureExtractor,
    )

    # Register extractors
    registry = FeatureRegistry()
    registry.register(OffensiveFeatureExtractor())
    registry.register(DefensiveFeatureExtractor())

    # Extract all features for a game
    summary = registry.extract_all('KC', 'BUF', 2024, 10)
    print(summary.features)

Created: December 13, 2025
Phase 2 of code quality remediation plan.
"""

from nfl_analytics.features.extractors.base import (
    ExtractionLevel,
    FeatureExtractionError,
    FeatureExtractionResult,
    FeatureExtractorStrategy,
    FeaturePriority,
    GameLevelExtractorStrategy,
)
from nfl_analytics.features.extractors.registry import (
    ExtractionSummary,
    FeatureRegistry,
    create_default_registry,
)

# Strategy pattern wrappers for existing extractors
from nfl_analytics.features.extractors.offensive import OffensiveFeatureExtractor
from nfl_analytics.features.extractors.defensive import DefensiveFeatureExtractor
from nfl_analytics.features.extractors.nextgen import NextGenFeatureExtractor
from nfl_analytics.features.extractors.qbr import QBRFeatureExtractor
from nfl_analytics.features.extractors.opponent_adjusted import OpponentAdjustedFeatureExtractor

__all__ = [
    # Base classes
    "FeatureExtractorStrategy",
    "GameLevelExtractorStrategy",
    "FeatureExtractionError",
    "FeatureExtractionResult",
    "ExtractionLevel",
    "FeaturePriority",
    # Registry
    "FeatureRegistry",
    "ExtractionSummary",
    "create_default_registry",
    # Concrete extractors
    "OffensiveFeatureExtractor",
    "DefensiveFeatureExtractor",
    "NextGenFeatureExtractor",
    "QBRFeatureExtractor",
    "OpponentAdjustedFeatureExtractor",
]
