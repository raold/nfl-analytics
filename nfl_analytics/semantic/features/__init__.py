"""
Semantic feature engineering.

Transforms LLM outputs into ML-ready features:
    - InjuryFeatures: is_key_player_injured, injury_severity_index
    - SentimentFeatures: player_sentiment_24h, team_sentiment_trend
    - NarrativeFeatures: coaching_change_recent, public_overreaction
"""

from .semantic_features import (
    SemanticFeatureExtractor,
    SemanticFeatureConfig,
    TeamSemanticState,
)

__all__ = [
    "SemanticFeatureExtractor",
    "SemanticFeatureConfig",
    "TeamSemanticState",
]
