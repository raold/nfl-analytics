"""
Repository pattern for NFL Analytics data layer.

Provides production-ready database access with:
- Connection pooling (20-50ms latency reduction)
- Query result caching (50% cache hit rate target)
- Type-safe query building
- Transaction support

Author: NFL Analytics Team
Date: November 9, 2025
"""

from .base import (
    BaseRepository,
    ConnectionPoolError,
    DatabaseConfig,
    QueryResult,
    RepositoryError,
)
from .feature_repository import FeatureRepository
from .game_repository import GameRepository
from .player_repository import PlayerRepository
from .prediction_repository import PredictionRepository

__all__ = [
    # Base classes
    "BaseRepository",
    "DatabaseConfig",
    "QueryResult",
    "RepositoryError",
    "ConnectionPoolError",
    # Domain repositories
    "GameRepository",
    "PlayerRepository",
    "PredictionRepository",
    "FeatureRepository",
]
