#!/usr/bin/env python3
"""
Standalone script to create a StandardScaler for cVAE conditions by extracting
the same data used during training (2020-2024 seasons).

This bypasses import issues by inlining all necessary code.
"""

import pickle
import logging
from pathlib import Path
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_conditions_from_database():
    """
    Extract pre-game condition vectors from database (same as cVAE training).
    """
    db_config = {
        'host': 'localhost',
        'port': 5544,
        'database': 'devdb01',
        'user': 'dro',
        'password': os.getenv('DB_PASSWORD', 'sicillionbillions')
    }

    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    query = """
    WITH game_quarter_agg AS (
        SELECT
            g.game_id,
            g.season,
            g.week,
            g.home_team,
            g.away_team,
            COALESCE(g.spread_close, 0) as spread_line,
            COALESCE(g.total_close, 45) as total_line,
            COALESCE(g.home_moneyline, -110) as home_moneyline,
            COALESCE(g.away_moneyline, -110) as away_moneyline
        FROM games g
        WHERE g.season >= %s AND g.season <= %s
            AND g.game_type = 'REG'
            AND g.home_score IS NOT NULL
    ),
    team_recent_stats AS (
        SELECT
            gq.*,
            COALESCE(
                (SELECT AVG(home_score) FROM games g2
                 WHERE g2.home_team = gq.home_team
                 AND g2.season = gq.season
                 AND g2.week < gq.week
                 AND g2.week >= gq.week - 3), 21) as home_recent_ppg,
            COALESCE(
                (SELECT AVG(away_score) FROM games g2
                 WHERE g2.home_team = gq.home_team
                 AND g2.season = gq.season
                 AND g2.week < gq.week
                 AND g2.week >= gq.week - 3), 21) as home_recent_papg,
            COALESCE(
                (SELECT AVG(home_score) FROM games g2
                 WHERE g2.away_team = gq.away_team
                 AND g2.season = gq.season
                 AND g2.week < gq.week
                 AND g2.week >= gq.week - 3), 21) as away_recent_ppg,
            COALESCE(
                (SELECT AVG(home_score) FROM games g2
                 WHERE g2.away_team = gq.away_team
                 AND g2.season = gq.season
                 AND g2.week < gq.week
                 AND g2.week >= gq.week - 3), 21) as away_recent_papg,
            350.0 as home_recent_ypg,
            350.0 as away_recent_ypg
        FROM game_quarter_agg gq
    )
    SELECT * FROM team_recent_stats
    ORDER BY season, week
    """

    try:
        cursor.execute(query, (2020, 2024))
        games = cursor.fetchall()
        logger.info(f"Extracted {len(games)} games from 2020-2024")

        conditions_list = []
        for game in games:
            condition = create_condition_vector(game)
            conditions_list.append(condition)

    except Exception as e:
        logger.error(f"Error extracting data: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

    return np.array(conditions_list)


def create_condition_vector(game):
    """
    Create 20-dim pre-game condition vector (same as cvae_data_pipeline.py).
    """
    condition = np.zeros(20)

    # Betting lines
    condition[0] = game.get('spread_line', 0) if game.get('spread_line') else 0
    condition[1] = game.get('total_line', 45) if game.get('total_line') else 45
    condition[2] = game.get('home_moneyline', -110) if game.get('home_moneyline') else -110
    condition[3] = game.get('away_moneyline', -110) if game.get('away_moneyline') else -110

    # Recent performance
    condition[4] = game.get('home_recent_ppg', 21) if game.get('home_recent_ppg') else 21
    condition[5] = game.get('home_recent_papg', 21) if game.get('home_recent_papg') else 21
    condition[6] = game.get('away_recent_ppg', 21) if game.get('away_recent_ppg') else 21
    condition[7] = game.get('away_recent_papg', 21) if game.get('away_recent_papg') else 21
    condition[8] = game.get('home_recent_ypg', 350) if game.get('home_recent_ypg') else 350
    condition[9] = game.get('away_recent_ypg', 350) if game.get('away_recent_ypg') else 350

    # Rest (simplified)
    condition[10] = 7  # Home rest days
    condition[11] = 7  # Away rest days

    # Weather (defaults)
    condition[12] = 60  # Temperature (F)
    condition[13] = 5   # Wind speed (mph)
    condition[14] = 0   # Precipitation (mm)

    # Game context
    condition[15] = 0  # Division game
    condition[16] = 0  # Primetime
    condition[17] = game.get('week', 9)  # Week of season

    # Win percentages (estimate from spread)
    spread = condition[0]
    condition[18] = 0.5 - spread / 50  # Home win %
    condition[19] = 0.5 + spread / 50  # Away win %

    return condition


def create_scaler():
    """
    Create StandardScaler from the same training data used for cVAE training.
    """
    logger.info("=" * 80)
    logger.info("CREATING CVAE CONDITION SCALER (STANDALONE)")
    logger.info("=" * 80)
    logger.info("")

    # Extract all conditions from 2020-2024
    logger.info("Extracting game conditions from database...")
    conditions = extract_conditions_from_database()

    logger.info(f"Extracted {len(conditions)} games")
    logger.info(f"Condition shape: {conditions.shape}")
    logger.info("")

    # Use only training split (70% of data) to fit scaler
    # This matches the split used in train_cvae.py
    n_train = int(len(conditions) * 0.70)
    train_conditions = conditions[:n_train]

    logger.info(f"Fitting scaler on {len(train_conditions)} training samples")

    # Create and fit StandardScaler
    scaler = StandardScaler()
    scaler.fit(train_conditions)

    logger.info("")
    logger.info("Scaler Statistics:")
    logger.info(f"  Mean: {scaler.mean_[:5]}... (first 5 features)")
    logger.info(f"  Std:  {scaler.scale_[:5]}... (first 5 features)")
    logger.info("")

    # Save scaler
    output_path = Path("models/cvae/scaler_conditions.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(scaler, f)

    logger.info(f"✓ Scaler saved to {output_path}")
    logger.info("")

    # Validation: transform a sample and check it's normalized
    sample_transformed = scaler.transform(train_conditions[:100])
    logger.info("Validation:")
    logger.info(f"  Original mean:     {np.mean(train_conditions[:100], axis=0)[:3]}")
    logger.info(f"  Original std:      {np.std(train_conditions[:100], axis=0)[:3]}")
    logger.info(f"  Transformed mean:  {np.mean(sample_transformed, axis=0)[:3]} (should be ~0)")
    logger.info(f"  Transformed std:   {np.std(sample_transformed, axis=0)[:3]} (should be ~1)")
    logger.info("")
    logger.info("✓ Scaler created successfully!")

    return scaler


if __name__ == "__main__":
    create_scaler()
