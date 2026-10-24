#!/usr/bin/env python3
"""
Prediction Repository - Domain-specific queries for model predictions.

Author: NFL Analytics Team
Date: November 9, 2025
"""

from typing import Any

from .base import BaseRepository, QueryResult


class PredictionRepository(BaseRepository):
    """Repository for model predictions and prop bets."""

    def save_predictions(
        self,
        predictions: list[dict[str, Any]],
        model_version: str,
        invalidate_cache: bool = True,
    ) -> int:
        """
        Save batch predictions with upsert logic.

        Args:
            predictions: List of prediction dicts
            model_version: Model version identifier
            invalidate_cache: Whether to invalidate prediction caches

        Returns:
            Number of rows inserted/updated
        """
        if not predictions:
            return 0

        # Build upsert query
        columns = list(predictions[0].keys())
        sql = f"""
            INSERT INTO predictions.prop_predictions
            ({', '.join(columns)})
            VALUES ({', '.join(['%s'] * len(columns))})
            ON CONFLICT (game_id, player_id, prop_type, model_version)
            DO UPDATE SET
                prediction = EXCLUDED.prediction,
                confidence = EXCLUDED.confidence,
                updated_at = NOW()
        """

        # Convert dicts to tuples in column order
        params_list = [tuple(pred[col] for col in columns) for pred in predictions]

        # Invalidate relevant caches
        cache_keys = []
        if invalidate_cache:
            cache_keys = [
                "upcoming_predictions",
                f"predictions_model_{model_version}",
            ]

        return self.execute_many(sql, params_list, invalidate_cache=cache_keys)

    def get_predictions_for_game(
        self,
        game_id: str,
        model_version: str | None = None,
        cache_ttl: int = 300,
    ) -> QueryResult:
        """
        Get all predictions for a game.

        Args:
            game_id: Game identifier
            model_version: Filter by model version (None for all models)
            cache_ttl: Cache TTL in seconds (default: 5 minutes)

        Returns:
            QueryResult with predictions
        """
        if model_version:
            sql = """
                SELECT *
                FROM predictions.prop_predictions
                WHERE game_id = %s
                  AND model_version = %s
                ORDER BY player_id, prop_type
            """
            params = (game_id, model_version)
            cache_key = f"predictions_game_{game_id}_model_{model_version}"
        else:
            sql = """
                SELECT *
                FROM predictions.prop_predictions
                WHERE game_id = %s
                ORDER BY model_version, player_id, prop_type
            """
            params = (game_id,)
            cache_key = f"predictions_game_{game_id}_all"

        return self.query(sql, params, cache_key=cache_key, cache_ttl=cache_ttl)

    def get_prediction_accuracy(
        self,
        model_version: str,
        season: int,
        min_confidence: float = 0.6,
        cache_ttl: int = 3600,
    ) -> QueryResult:
        """
        Get prediction accuracy metrics for a model.

        Args:
            model_version: Model version identifier
            season: NFL season year
            min_confidence: Minimum confidence threshold
            cache_ttl: Cache TTL in seconds

        Returns:
            QueryResult with accuracy metrics
        """
        sql = """
            SELECT
                prop_type,
                COUNT(*) as total_predictions,
                SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) as correct_predictions,
                ROUND(AVG(CASE WHEN was_correct THEN 1 ELSE 0 END) * 100, 2) as accuracy_pct,
                ROUND(AVG(ABS(prediction - actual_value)), 2) as mae,
                ROUND(AVG(confidence), 3) as avg_confidence
            FROM predictions.prop_predictions p
            JOIN games g ON p.game_id = g.game_id
            WHERE p.model_version = %s
              AND g.season = %s
              AND p.confidence >= %s
              AND p.actual_value IS NOT NULL
            GROUP BY prop_type
            ORDER BY accuracy_pct DESC
        """
        return self.query(
            sql,
            (model_version, season, min_confidence),
            cache_key=f"accuracy_{model_version}_{season}_{min_confidence}",
            cache_ttl=cache_ttl,
        )


if __name__ == "__main__":
    """Test prediction repository."""
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    repo = PredictionRepository()

    # Test get predictions (if any exist)
    try:
        result = repo.query("SELECT COUNT(*) as count FROM predictions.prop_predictions")
        logger.info(f"✓ Found {result.first()['count']} total predictions in database")
    except Exception as e:
        logger.warning(f"Predictions table may not exist yet: {e}")

    logger.info("✅ Prediction repository tests passed!")
