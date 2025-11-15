#!/usr/bin/env python3
"""
Generate Week 12 2024 Predictions using Ensemble v6.0
======================================================

Populates ml.predictions table with ensemble v6.0 predictions for all Week 12 2024 games.

Usage:
    python scripts/generate_week12_predictions.py

Requirements:
    - Ensemble v6.0 models loaded
    - Database connection to devdb01
    - Week 12 2024 schedule in games table

Author: Claude Code (Sonnet 4.5)
Date: November 15, 2025
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

from py.ensemble.enhanced_ensemble_v6_production import EnhancedEnsemble5Way

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_week12_starters(conn) -> List[Dict]:
    """
    Get starting QBs for Week 12 2024 games from existing predictions.

    Returns list of dicts with player_id, player_name, team, opponent, is_home, game_id
    """
    query = """
    SELECT DISTINCT
        player_id,
        player_name,
        team,
        opponent,
        is_home,
        game_id,
        season,
        week
    FROM ml.predictions
    WHERE season = 2024
      AND week = 12
      AND stat_type = 'passing_yards'
    ORDER BY player_name;
    """

    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    cur.close()

    starters = []
    for row in rows:
        player_id, player_name, team, opponent, is_home, game_id, season, week = row
        starters.append({
            'player_id': player_id,
            'player_name': player_name,
            'team': team,
            'opponent': opponent,
            'is_home': is_home,
            'game_id': game_id,
            'season': season,
            'week': week
        })

    logger.info(f"Found {len(starters)} starting QBs for Week 12 2024")
    return starters


def insert_prediction(
    conn,
    model_id: int,
    game_info: Dict,
    prediction: Dict,
    ensemble_version: str,
    model_components: Dict
) -> int:
    """
    Insert prediction into ml.predictions table.

    Returns prediction_id of inserted row.
    """
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
        prediction_std,
        prediction_q05,
        prediction_q95,
        ensemble_version,
        model_components,
        status,
        created_at
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s, %s
    )
    ON CONFLICT (model_id, game_id, player_id, stat_type)
    DO UPDATE SET
        prediction_mean = EXCLUDED.prediction_mean,
        prediction_std = EXCLUDED.prediction_std,
        prediction_q05 = EXCLUDED.prediction_q05,
        prediction_q95 = EXCLUDED.prediction_q95,
        ensemble_version = EXCLUDED.ensemble_version,
        model_components = EXCLUDED.model_components,
        updated_at = NOW()
    RETURNING prediction_id;
    """

    # Calculate 90% prediction intervals
    lower_90 = prediction['lower_90']
    upper_90 = prediction['upper_90']

    cur = conn.cursor()
    cur.execute(
        insert_query,
        (
            model_id,
            game_info['game_id'],
            game_info['season'],
            game_info['week'],
            game_info['player_id'],
            game_info['player_name'],
            game_info['team'],
            game_info['opponent'],
            game_info['is_home'],
            'passing_yards',
            prediction['ensemble_mean'],
            prediction['calibrated_std'],
            lower_90,
            upper_90,
            ensemble_version,
            json.dumps(model_components),
            'pending',
            datetime.now()
        )
    )

    prediction_id = cur.fetchone()[0]
    cur.close()
    conn.commit()

    return prediction_id


def main():
    """Generate Week 12 2024 predictions using Ensemble v6.0"""

    logger.info("="*80)
    logger.info("Week 12 2024 Prediction Generation - Ensemble v6.0")
    logger.info("="*80)

    # Database connection
    db_password = os.getenv("DB_PASSWORD")
    if not db_password:
        logger.error("DB_PASSWORD environment variable not set")
        return 1

    db_config = {
        "host": "localhost",
        "port": 5544,
        "dbname": "devdb01",
        "user": "dro",
        "password": db_password
    }

    try:
        conn = psycopg2.connect(**db_config)
        logger.info("✅ Database connection established")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return 1

    # Load Ensemble v6.0
    logger.info("\nLoading Ensemble v6.0...")
    try:
        ensemble = EnhancedEnsemble5Way(db_password=db_password)
        logger.info(f"✅ Ensemble loaded with weights: {ensemble.weights}")
        logger.info(f"   Temperature: {ensemble.temperature:.4f}")
        logger.info(f"   Version: {ensemble.ensemble_version}")
    except Exception as e:
        logger.error(f"❌ Ensemble loading failed: {e}")
        import traceback
        traceback.print_exc()
        conn.close()
        return 1

    # Get Week 12 starters
    logger.info("\nFetching Week 12 2024 starting QBs...")
    try:
        starters = get_week12_starters(conn)
        logger.info(f"✅ Found {len(starters)} QBs for Week 12")
    except Exception as e:
        logger.error(f"❌ Failed to fetch starters: {e}")
        import traceback
        traceback.print_exc()
        conn.close()
        return 1

    # Get model_id for ensemble v6.0
    cur = conn.cursor()
    cur.execute("SELECT model_id FROM ml.model_registry WHERE model_name = 'ensemble_v6.0';")
    model_id_row = cur.fetchone()
    cur.close()

    if not model_id_row:
        logger.error("❌ ensemble_v6.0 not found in model_registry")
        conn.close()
        return 1

    model_id = model_id_row[0]
    logger.info(f"✅ Using model_id={model_id} (ensemble_v6.0)")

    # Ensemble metadata
    ensemble_version = "v6.0"
    model_components = {
        "state_space_v2": 0.27,
        "hierarchical_v2": 0.27,
        "xgboost_v4_ridge": 0.23,
        "informative_priors_v2": 0.13,
        "hybrid_static_incremental": 0.10
    }

    # Generate predictions
    logger.info("\n" + "="*80)
    logger.info("Generating predictions...")
    logger.info("="*80)

    predictions_inserted = 0
    predictions_failed = 0
    results = []

    for i, starter in enumerate(starters, 1):
        player_name = starter['player_name']
        team = starter['team']
        opponent = starter['opponent']

        logger.info(f"\n[{i}/{len(starters)}] {player_name} ({team} vs {opponent})")

        try:
            # Generate prediction
            pred = ensemble.predict(
                player_id=starter['player_id'],
                season=starter['season'],
                week=starter['week'],
                return_components=False,
                return_uncertainty=True
            )

            logger.info(f"  Prediction: {pred['ensemble_mean']:.1f} ± {pred['calibrated_std']:.1f} yards")
            logger.info(f"  90% PI: [{pred['lower_90']:.1f}, {pred['upper_90']:.1f}]")
            logger.info(f"  Inference time: {pred['prediction_time_ms']:.1f}ms")

            # Insert into database
            prediction_id = insert_prediction(
                conn,
                model_id,
                starter,
                pred,
                ensemble_version,
                model_components
            )

            logger.info(f"  ✅ Inserted prediction_id={prediction_id}")
            predictions_inserted += 1

            results.append({
                'player_name': player_name,
                'team': team,
                'opponent': opponent,
                'prediction_mean': pred['ensemble_mean'],
                'prediction_std': pred['calibrated_std'],
                'lower_90': pred['lower_90'],
                'upper_90': pred['upper_90']
            })

        except Exception as e:
            logger.error(f"  ❌ Prediction failed: {e}")
            predictions_failed += 1

    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Total QBs: {len(starters)}")
    logger.info(f"Predictions inserted: {predictions_inserted}")
    logger.info(f"Predictions failed: {predictions_failed}")

    # Top 5 predictions
    if results:
        logger.info("\n📊 Top 5 Predictions (by predicted yards):")
        results_sorted = sorted(results, key=lambda x: x['prediction_mean'], reverse=True)
        for i, r in enumerate(results_sorted[:5], 1):
            logger.info(
                f"  {i}. {r['player_name']:20s} ({r['team']} vs {r['opponent']:3s}): "
                f"{r['prediction_mean']:.1f} yards "
                f"[{r['lower_90']:.1f}, {r['upper_90']:.1f}]"
            )

    # Verify database
    logger.info("\n📋 Database Verification:")
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*)
        FROM ml.predictions
        WHERE season = 2024 AND week = 12 AND ensemble_version = 'v6.0'
    """)
    count = cur.fetchone()[0]
    cur.close()
    logger.info(f"  Week 12 predictions with v6.0: {count}")

    conn.close()
    logger.info("\n✅ Week 12 prediction generation complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
