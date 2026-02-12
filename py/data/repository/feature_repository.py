#!/usr/bin/env python3
"""
Feature Repository - Domain-specific queries for ML features.

Author: NFL Analytics Team
Date: November 9, 2025
"""

from .base import BaseRepository


class FeatureRepository(BaseRepository):
    """Repository for machine learning features."""

    def get_game_features(
        self,
        game_id: str,
        feature_set: str = "baseline",
        cache_ttl: int = 3600,
    ) -> dict | None:
        """
        Get ML features for a game.

        Args:
            game_id: Game identifier
            feature_set: Feature set name (baseline, enhanced, advanced)
            cache_ttl: Cache TTL in seconds

        Returns:
            Feature dict or None if not found
        """
        # Map feature set to materialized view (whitelist validation)
        view_map = {
            "baseline": "mart.game_features_baseline",
            "enhanced": "mart.game_features_enhanced",
            "advanced": "mart.game_features_advanced",
        }

        if feature_set not in view_map:
            raise ValueError(f"Unknown feature_set '{feature_set}'. Must be one of: {list(view_map.keys())}")

        view = view_map[feature_set]

        sql = f"SELECT * FROM {view} WHERE game_id = %s"
        result = self.query(
            sql,
            (game_id,),
            cache_key=f"features_{feature_set}_{game_id}",
            cache_ttl=cache_ttl,
        )
        return result.first()

    def get_player_features(
        self,
        player_id: str,
        game_id: str,
        cache_ttl: int = 3600,
    ) -> dict | None:
        """
        Get ML features for a player in a specific game.

        Args:
            player_id: Player identifier
            game_id: Game identifier
            cache_ttl: Cache TTL in seconds

        Returns:
            Feature dict or None if not found
        """
        sql = """
            SELECT *
            FROM mart.player_game_features
            WHERE player_id = %s
              AND game_id = %s
        """
        result = self.query(
            sql,
            (player_id, game_id),
            cache_key=f"player_features_{player_id}_{game_id}",
            cache_ttl=cache_ttl,
        )
        return result.first()

    def refresh_materialized_views(self) -> None:
        """
        Refresh all feature materialized views.

        This should be called after ingesting new game data.
        """
        views = [
            "mart.game_features_baseline",
            "mart.game_features_enhanced",
            "mart.game_features_advanced",
        ]

        for view in views:
            self.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view}")

        # Clear all feature caches
        self.clear_cache()


if __name__ == "__main__":
    """Test feature repository."""
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    repo = FeatureRepository()

    logger.info("✅ Feature repository loaded successfully!")
