#!/usr/bin/env python3
"""
Create a StandardScaler for cVAE conditions by extracting the same data
used during training (2020-2024 seasons).

This scaler will be used to normalize pre-game features before feeding to cVAE.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import logging
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler

from py.data.cvae_data_pipeline import CVAEDataPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_scaler():
    """
    Create StandardScaler from the same training data used for cVAE training.
    """
    logger.info("=" * 80)
    logger.info("CREATING CVAE CONDITION SCALER")
    logger.info("=" * 80)
    logger.info("")

    # Connect to database and extract training data
    db_config = {
        'host': 'localhost',
        'port': 5544,
        'database': 'devdb01',
        'user': 'dro',
        'password': os.getenv('DB_PASSWORD', 'sicillionbillions')
    }

    pipeline = CVAEDataPipeline(db_config)

    try:
        pipeline.connect()
        logger.info("Extracting game trajectories from 2020-2024 (same as training)...")

        trajectories, conditions = pipeline.extract_game_trajectories(
            start_season=2020,
            end_season=2024
        )

        logger.info(f"Extracted {len(trajectories)} games")
        logger.info(f"Condition shape: {conditions.shape}")
        logger.info("")

    except Exception as e:
        logger.error(f"Error extracting data: {e}")
        raise
    finally:
        pipeline.disconnect()

    # Use only training split (70% of data) to fit scaler
    # This matches the split used in train_cvae.py (line 141)
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
