#!/usr/bin/env python3
"""
Generate Game Predictions Using Ensemble v6.1

Replaces baseline_v1.0 predictions with production Ensemble v6.1 game predictions.
Uses GameOutcomePredictor which aggregates QB predictions into team scores.

Usage:
    python scripts/generate_game_predictions_v6.py --season 2025 --week 12

Author: Claude Code (Sonnet 4.5)
Date: November 21, 2025
"""
import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from py.ensemble.game_outcome_predictor import GameOutcomePredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5544')),
    'dbname': os.getenv('DB_NAME', 'devdb01'),
    'user': os.getenv('DB_USER', 'dro'),
    'password': os.getenv('DB_PASSWORD', 'sicillionbillions'),
}


def get_games(season: int, week: int):
    """Get games for a given season/week."""
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT game_id, home_team, away_team, season, week,
                       kickoff, spread_close, total_close
                FROM games
                WHERE season = %s AND week = %s AND game_type = 'REG'
                ORDER BY kickoff
            """, (season, week))
            return cur.fetchall()


def store_prediction(game_prediction: dict):
    """Store game prediction in ml.game_predictions."""
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            # Delete existing prediction for this game_id + ensemble_version
            cur.execute("""
                DELETE FROM ml.game_predictions
                WHERE game_id = %(game_id)s AND ensemble_version = 'v6.1'
            """, game_prediction)

            # Insert new prediction
            cur.execute("""
                INSERT INTO ml.game_predictions (
                    game_id, season, week, home_team, away_team,
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
                    %(game_id)s, %(season)s, %(week)s, %(home_team)s, %(away_team)s,
                    'v6.1',
                    %(predicted_home_score)s,
                    %(predicted_away_score)s,
                    %(predicted_spread)s,
                    %(predicted_total)s,
                    %(predicted_home_win_prob)s,
                    %(spread_std)s,
                    %(total_std)s,
                    %(sportsbook_spread)s,
                    %(sportsbook_total)s,
                    %(spread_edge)s,
                    %(total_edge)s,
                    NOW()
                )
            """, game_prediction)
        conn.commit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--season', type=int, required=True)
    parser.add_argument('--week', type=int, required=True)
    args = parser.parse_args()

    logger.info(f"\n{'='*80}")
    logger.info(f"GENERATING GAME PREDICTIONS - Ensemble v6.1")
    logger.info(f"{'='*80}\n")
    logger.info(f"Season: {args.season}, Week: {args.week}")

    # Initialize predictor
    predictor = GameOutcomePredictor()

    # Get games
    games = get_games(args.season, args.week)
    logger.info(f"Found {len(games)} games\n")

    # Generate predictions
    predictions_stored = 0
    for game in games:
        logger.info(f"Predicting: {game['away_team']} @ {game['home_team']}")

        try:
            # Predict game outcome
            result = predictor.predict_game(
                home_team=game['home_team'],
                away_team=game['away_team'],
                season=game['season'],
                week=game['week']
            )

            # Calculate edges
            spread_edge = result['predicted_spread'] - (game['spread_close'] or 0)
            total_edge = result['predicted_total'] - (game['total_close'] or 0)

            # Prepare for storage (convert NumPy types to Python types)
            prediction = {
                'game_id': game['game_id'],
                'season': int(game['season']),
                'week': int(game['week']),
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'predicted_home_score': float(result['predicted_home_score']),
                'predicted_away_score': float(result['predicted_away_score']),
                'predicted_spread': float(result['predicted_spread']),
                'predicted_total': float(result['predicted_total']),
                'predicted_home_win_prob': float(result['predicted_home_win_prob']),
                'spread_std': float(result['spread_std']),
                'total_std': float(result['total_std']),
                'sportsbook_spread': float(game['spread_close']) if game['spread_close'] is not None else None,
                'sportsbook_total': float(game['total_close']) if game['total_close'] is not None else None,
                'spread_edge': float(spread_edge),
                'total_edge': float(total_edge)
            }

            # Store in database
            store_prediction(prediction)
            predictions_stored += 1

            logger.info(f"  ✅ {result['predicted_away_score']:.1f} @ {result['predicted_home_score']:.1f} "
                       f"(spread: {result['predicted_spread']:+.1f}, total: {result['predicted_total']:.1f})")

        except Exception as e:
            logger.error(f"  ❌ Failed: {e}")

    logger.info(f"\n{'='*80}")
    logger.info(f"✅ Generated and stored {predictions_stored}/{len(games)} predictions")
    logger.info(f"{'='*80}\n")


if __name__ == '__main__':
    main()
