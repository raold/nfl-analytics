#!/usr/bin/env python3
"""
Populate test predictions for Ensemble v7.0 WLS adjustment testing.

Creates synthetic prediction history with controlled bias pattern:
- 50 predictions for Mahomes 2024 weeks 1-10
- Systematic +5 yard overestimation (to test bias correction)

Author: Ensemble v7.0
Date: November 18, 2025
"""

import logging
import sys
from pathlib import Path

import numpy as np
import psycopg2

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database config
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "database": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",
}


def populate_test_predictions():
    """
    Populate ml.predictions with synthetic v7.0 predictions.

    Creates 50 predictions with systematic +5 yard bias.
    """
    logger.info("=" * 80)
    logger.info("Populating test predictions for v7.0 WLS adjustment")
    logger.info("=" * 80)

    # Generate synthetic predictions
    predictions = []

    # Player: Patrick Mahomes
    player_id = "00-0033873"
    player_name = "P.Mahomes"
    team = "KC"

    # Create 50 predictions across weeks 1-10 (5 predictions per week)
    np.random.seed(42)

    for week in range(1, 11):  # Weeks 1-10
        for rep in range(5):  # 5 predictions per week
            # True value: ~250 yards with noise
            actual_value = np.random.normal(250, 30)

            # Prediction: true + 5 yard bias + noise
            prediction_mean = actual_value + 5.0 + np.random.normal(0, 5)

            # Error
            prediction_error = prediction_mean - actual_value

            predictions.append(
                {
                    "player_id": player_id,
                    "player_name": player_name,
                    "team": team,
                    "season": 2024,
                    "week": week,
                    "opponent": "OPP",
                    "is_home": True,
                    "stat_type": "passing_yards",
                    "prediction_mean": round(prediction_mean, 2),
                    "actual_value": round(actual_value, 2),
                    "prediction_error": round(prediction_error, 2),
                    "ensemble_version": "v7.0",
                }
            )

    logger.info(f"Generated {len(predictions)} synthetic predictions")
    logger.info(
        f"  Average bias: +{np.mean([p['prediction_error'] for p in predictions]):.2f} yards"
    )

    # Insert into database
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Get or create model_id for v7.0 ensemble
    cursor.execute(
        """
        SELECT model_id FROM ml.model_registry
        WHERE model_name = 'enhanced_ensemble_v7.0'
        LIMIT 1
    """
    )
    result = cursor.fetchone()

    if result:
        model_id = result[0]
        logger.info(f"Using existing model_id: {model_id}")
    else:
        # Create model entry
        cursor.execute(
            """
            INSERT INTO ml.model_registry (
                model_name,
                model_type,
                model_family,
                version,
                model_path,
                status,
                description
            ) VALUES (
                'enhanced_ensemble_v7.0',
                'ensemble',
                'meta_learning_wls',
                'v7.0',
                'nfl_analytics/ensemble/enhanced_ensemble_v7_production.py',
                'testing',
                'v7.0 ensemble with WLS adjustment testing data'
            )
            RETURNING model_id
        """
        )
        model_id = cursor.fetchone()[0]
        logger.info(f"Created new model_id: {model_id}")

    # Insert predictions
    insert_query = """
        INSERT INTO ml.predictions (
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
            prediction_mean,
            actual_value,
            prediction_error,
            ensemble_version
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (model_id, game_id, player_id, stat_type) DO NOTHING
    """

    inserted = 0
    for pred in predictions:
        # Generate game_id
        game_id = f"2024_{pred['week']:02d}_{pred['team']}"

        cursor.execute(
            insert_query,
            (
                model_id,
                game_id,
                pred["season"],
                pred["week"],
                pred["player_id"],
                pred["player_name"],
                pred["team"],
                pred["opponent"],
                pred["is_home"],
                pred["stat_type"],
                pred["prediction_mean"],
                pred["actual_value"],
                pred["prediction_error"],
                pred["ensemble_version"],
            ),
        )

        if cursor.rowcount > 0:
            inserted += 1

    conn.commit()
    cursor.close()
    conn.close()

    logger.info(f"✅ Inserted {inserted}/{len(predictions)} predictions")
    logger.info("=" * 80)


if __name__ == "__main__":
    populate_test_predictions()
