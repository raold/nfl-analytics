#!/usr/bin/env python3
"""
Store Props Predictions in ml.predictions Table

Stores rushing yards, receiving yards, and touchdown predictions
in the database for tracking and validation.
"""

import sys

sys.path.append("/Users/dro/rice/nfl-analytics")

import logging
from datetime import datetime
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PropsPredictionStore:
    """Stores props predictions in ml.predictions table."""

    def __init__(self):
        self.db_config = {
            "host": "localhost",
            "port": 5544,
            "database": "devdb01",
            "user": "dro",
            "password": "sicillionbillions",
        }

    def get_model_id(self, model_name: str) -> Optional[int]:
        """Get model_id from ml.model_registry by model name."""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()

        query = """
        SELECT model_id
        FROM ml.model_registry
        WHERE model_name = %s
        LIMIT 1
        """

        cursor.execute(query, (model_name,))
        result = cursor.fetchone()

        cursor.close()
        conn.close()

        if result:
            return result[0]
        else:
            logger.warning(f"Model '{model_name}' not found in registry")
            return None

    def store_prediction(
        self,
        model_name: str,
        game_id: str,
        season: int,
        week: int,
        player_id: str,
        player_name: str,
        team: str,
        opponent: str,
        is_home: bool,
        stat_type: str,
        prediction_mean: float,
        prediction_std: float,
        prediction_q05: float = None,
        prediction_q50: float = None,
        prediction_q95: float = None,
        line: float = None,
        side: str = None,
        odds: int = None,
    ) -> Optional[int]:
        """
        Store a single prediction in ml.predictions table.

        Args:
            model_name: Model name from ml.model_registry
            game_id: Game ID
            season: Season year
            week: Week number
            player_id: Player ID
            player_name: Player name
            team: Team abbreviation
            opponent: Opponent abbreviation
            is_home: Is home game
            stat_type: Stat type (passing_yards, rushing_yards, receiving_yards, touchdowns)
            prediction_mean: Mean prediction
            prediction_std: Standard deviation
            prediction_q05: 5th percentile (optional)
            prediction_q50: Median (optional)
            prediction_q95: 95th percentile (optional)
            line: Betting line (optional)
            side: Over/Under side (optional)
            odds: American odds (optional)

        Returns:
            prediction_id if successful, None otherwise
        """
        # Get model_id
        model_id = self.get_model_id(model_name)
        if model_id is None:
            logger.error(f"Cannot store prediction: model '{model_name}' not in registry")
            return None

        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()

        # Insert prediction
        query = """
        INSERT INTO ml.predictions (
            model_id, game_id, season, week,
            player_id, player_name, team, opponent, is_home,
            stat_type, line, side, odds,
            prediction_mean, prediction_median, prediction_std,
            prediction_q05, prediction_q50, prediction_q95,
            status
        ) VALUES (
            %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s,
            'pending'
        )
        ON CONFLICT (model_id, game_id, player_id, stat_type)
        DO UPDATE SET
            prediction_mean = EXCLUDED.prediction_mean,
            prediction_median = EXCLUDED.prediction_median,
            prediction_std = EXCLUDED.prediction_std,
            prediction_q05 = EXCLUDED.prediction_q05,
            prediction_q50 = EXCLUDED.prediction_q50,
            prediction_q95 = EXCLUDED.prediction_q95,
            updated_at = NOW()
        RETURNING prediction_id
        """

        try:
            cursor.execute(
                query,
                (
                    model_id,
                    game_id,
                    season,
                    week,
                    player_id,
                    player_name,
                    team,
                    opponent,
                    is_home,
                    stat_type,
                    line,
                    side,
                    odds,
                    prediction_mean,
                    prediction_q50 or prediction_mean,  # Use median if available, else mean
                    prediction_std,
                    prediction_q05,
                    prediction_q50,
                    prediction_q95,
                ),
            )

            result = cursor.fetchone()
            prediction_id = result[0] if result else None

            conn.commit()

            logger.info(
                f"Stored prediction: {player_name} ({stat_type}) W{week} {season} "
                f"-> {prediction_mean:.1f}±{prediction_std:.1f}"
            )

            cursor.close()
            conn.close()

            return prediction_id

        except Exception as e:
            logger.error(f"Failed to store prediction: {e}")
            conn.rollback()
            cursor.close()
            conn.close()
            return None

    def bulk_store_predictions(self, predictions: list[dict]) -> int:
        """
        Store multiple predictions at once.

        Args:
            predictions: List of prediction dictionaries (same schema as store_prediction)

        Returns:
            Number of predictions stored successfully
        """
        stored_count = 0

        for pred in predictions:
            result = self.store_prediction(**pred)
            if result is not None:
                stored_count += 1

        logger.info(f"Stored {stored_count}/{len(predictions)} predictions")
        return stored_count


def example_usage():
    """Example: Store sample predictions for rushing, receiving, and TD"""

    store = PropsPredictionStore()

    # Example 1: Rushing yards prediction (Derrick Henry)
    rushing_pred = {
        "model_name": "bnn_rushing_v3",
        "game_id": "2025_12_TEN_HOU",
        "season": 2025,
        "week": 12,
        "player_id": "00-0034844",
        "player_name": "Derrick Henry",
        "team": "TEN",
        "opponent": "HOU",
        "is_home": False,
        "stat_type": "rushing_yards",
        "prediction_mean": 95.3,
        "prediction_std": 18.5,
        "prediction_q05": 62.8,
        "prediction_q50": 94.1,
        "prediction_q95": 127.8,
        "line": 92.5,
        "side": "over",
        "odds": -110,
    }

    # Example 2: Receiving yards prediction (Tyreek Hill)
    receiving_pred = {
        "model_name": "bnn_receiving_v1",
        "game_id": "2025_12_MIA_NYJ",
        "season": 2025,
        "week": 12,
        "player_id": "00-0033077",
        "player_name": "Tyreek Hill",
        "team": "MIA",
        "opponent": "NYJ",
        "is_home": True,
        "stat_type": "receiving_yards",
        "prediction_mean": 88.4,
        "prediction_std": 22.1,
        "prediction_q05": 52.0,
        "prediction_q50": 86.8,
        "prediction_q95": 124.8,
        "line": 85.5,
        "side": "over",
        "odds": -115,
    }

    # Example 3: Touchdown probability (Travis Kelce)
    td_pred = {
        "model_name": "xgboost_touchdowns_v1",
        "game_id": "2025_12_KC_LV",
        "season": 2025,
        "week": 12,
        "player_id": "00-0029701",
        "player_name": "Travis Kelce",
        "team": "KC",
        "opponent": "LV",
        "is_home": False,
        "stat_type": "touchdowns",
        "prediction_mean": 0.42,  # 42% probability
        "prediction_std": 0.0,  # Binary classifier, no std
        "prediction_q05": 0.32,
        "prediction_q50": 0.42,
        "prediction_q95": 0.52,
        "line": 0.5,  # Anytime TD scorer
        "side": "over",
        "odds": +120,
    }

    # Store all predictions
    predictions = [rushing_pred, receiving_pred, td_pred]
    stored_count = store.bulk_store_predictions(predictions)

    logger.info(f"\n✓ Example complete: {stored_count}/3 predictions stored")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PROPS PREDICTION STORAGE EXAMPLE")
    logger.info("=" * 60)

    example_usage()

    logger.info("\n" + "=" * 60)
    logger.info("✓ COMPLETE")
    logger.info("=" * 60)
