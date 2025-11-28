#!/usr/bin/env python3
"""
Generate predictions for all Week 12 2025 games using GameOutcomePredictor.

This script creates baseline predictions for display in the dashboard.
"""

import os
import sys
from pathlib import Path
import psycopg
from datetime import datetime
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from py.ensemble.game_outcome_predictor import GameOutcomePredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5544")),
    "dbname": os.getenv("DB_NAME", "devdb01"),
    "user": os.getenv("DB_USER", "dro"),
    "password": os.getenv("DB_PASSWORD", "sicillionbillions")
}


def get_week12_games():
    """Fetch all Week 12 2025 games from database."""
    query = """
    SELECT game_id, home_team, away_team, spread_close, total_close
    FROM games
    WHERE season = 2025 AND week = 12
    ORDER BY kickoff
    """

    with psycopg.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            games = cursor.fetchall()

    return [
        {
            'game_id': row[0],
            'home_team': row[1],
            'away_team': row[2],
            'spread_close': row[3],
            'total_close': row[4]
        }
        for row in games
    ]


def store_prediction(game_id, home_team, away_team, prediction_data, sportsbook_spread, sportsbook_total):
    """Store game prediction in database."""
    query = """
    INSERT INTO ml.game_predictions (
        game_id,
        home_team,
        away_team,
        season,
        week,
        ensemble_version,
        predicted_home_score,
        predicted_away_score,
        predicted_spread,
        predicted_total,
        predicted_home_win_prob,
        spread_std,
        total_std,
        sportsbook_spread,
        sportsbook_total,
        spread_edge,
        total_edge,
        created_at
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
    ON CONFLICT ON CONSTRAINT unique_game_model_prediction
    DO UPDATE SET
        predicted_home_score = EXCLUDED.predicted_home_score,
        predicted_away_score = EXCLUDED.predicted_away_score,
        predicted_spread = EXCLUDED.predicted_spread,
        predicted_total = EXCLUDED.predicted_total,
        predicted_home_win_prob = EXCLUDED.predicted_home_win_prob,
        spread_std = EXCLUDED.spread_std,
        total_std = EXCLUDED.total_std,
        sportsbook_spread = EXCLUDED.sportsbook_spread,
        sportsbook_total = EXCLUDED.sportsbook_total,
        spread_edge = EXCLUDED.spread_edge,
        total_edge = EXCLUDED.total_edge,
        created_at = EXCLUDED.created_at
    RETURNING prediction_id
    """

    # Calculate edges
    spread_edge = prediction_data['predicted_spread'] - sportsbook_spread if sportsbook_spread is not None else None
    total_edge = prediction_data['predicted_total'] - sportsbook_total if sportsbook_total is not None else None

    data = (
        game_id,
        home_team,
        away_team,
        2025,
        12,
        'baseline_v1.0',  # Using baseline since we don't have full ensemble running
        prediction_data['predicted_home_score'],
        prediction_data['predicted_away_score'],
        prediction_data['predicted_spread'],
        prediction_data['predicted_total'],
        prediction_data['predicted_home_win_prob'],
        prediction_data.get('spread_std', 0),
        prediction_data.get('total_std', 0),
        sportsbook_spread,
        sportsbook_total,
        spread_edge,
        total_edge,
        datetime.utcnow()
    )

    with psycopg.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, data)
            result = cursor.fetchone()
            conn.commit()
            return result[0] if result else None


def main():
    """Generate predictions for all Week 12 games."""
    logger.info("="*60)
    logger.info("Generating predictions for Week 12 2025 games")
    logger.info("="*60)

    # Get all games
    games = get_week12_games()
    logger.info(f"Found {len(games)} games to process\n")

    # Initialize predictor
    predictor = GameOutcomePredictor()

    # Generate predictions
    success_count = 0
    error_count = 0

    for game in games:
        try:
            logger.info(f"Processing {game['game_id']}: {game['away_team']} @ {game['home_team']}")

            # Generate prediction
            prediction = predictor.predict_game(
                home_team=game['home_team'],
                away_team=game['away_team'],
                season=2025,
                week=12
            )

            # Store in database
            prediction_id = store_prediction(
                game_id=game['game_id'],
                home_team=game['home_team'],
                away_team=game['away_team'],
                prediction_data=prediction,
                sportsbook_spread=game['spread_close'],
                sportsbook_total=game['total_close']
            )

            # Calculate and log edge
            spread_edge = prediction['predicted_spread'] - game['spread_close'] if game['spread_close'] is not None else None
            total_edge = prediction['predicted_total'] - game['total_close'] if game['total_close'] is not None else None

            logger.info(f"  ✅ Predicted: {game['home_team']} {prediction['predicted_spread']:+.1f} | Total {prediction['predicted_total']:.1f}")
            logger.info(f"  📊 Sportsbook: {game['home_team']} {game['spread_close']:+.1f} | Total {game['total_close']:.1f}")
            if spread_edge is not None:
                logger.info(f"  📈 Edge: Spread {spread_edge:+.1f} | Total {total_edge:+.1f}")
            logger.info(f"  💾 Stored as prediction_id={prediction_id}\n")

            success_count += 1

        except Exception as e:
            logger.error(f"  ❌ Error processing {game['game_id']}: {e}\n")
            import traceback
            logger.error(traceback.format_exc())
            error_count += 1
            continue

    logger.info("="*60)
    logger.info(f"Generation complete: {success_count} success, {error_count} errors")
    logger.info("="*60)


if __name__ == "__main__":
    main()
