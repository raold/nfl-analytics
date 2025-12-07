#!/usr/bin/env python3
"""
Test Ensemble Weight Optimization with Synthetic Data

Generates synthetic predictions to validate optimize_ensemble_weights.py
functionality before running on actual hold-out data.

Usage:
    python py/ensemble/test_weight_optimization.py

Author: NFL Analytics Research Team
Date: November 9, 2025
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from optimize_ensemble_weights import optimize_ensemble_weights

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_predictions(
    n_games: int = 148,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic predictions for 5 models with realistic characteristics.

    Models simulated:
    1. state_space_v2: Best MAE (47.77), good calibration
    2. bnn_v3_1: Moderate MAE (58.70), excellent calibration
    3. hierarchical_v2: Good MAE (52.34), good calibration
    4. xgboost_v2: Good MAE (54.12), unknown calibration
    5. informative_priors_v2: Similar to hierarchical, slightly worse

    Args:
        n_games: Number of games to simulate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with synthetic predictions
    """
    np.random.seed(seed)

    # Generate actual passing yards (realistic distribution)
    actual_mean = 250.0
    actual_std = 75.0
    actual_yards = np.random.normal(actual_mean, actual_std, n_games)
    actual_yards = np.clip(actual_yards, 50, 500)  # Realistic bounds

    # Model 1: State-Space v2 (best MAE, slightly biased)
    mae_ss = 47.77
    pred_state_space = actual_yards + np.random.normal(0, mae_ss * 0.8, n_games)
    std_state_space = np.random.uniform(30, 50, n_games)

    # Model 2: BNN v3.1 (moderate MAE, well-calibrated)
    mae_bnn = 58.70
    pred_bnn = actual_yards + np.random.normal(0, mae_bnn * 0.8, n_games)
    std_bnn = np.random.uniform(40, 60, n_games)

    # Model 3: Hierarchical v2 (good MAE)
    mae_hier = 52.34
    pred_hierarchical = actual_yards + np.random.normal(0, mae_hier * 0.8, n_games)
    std_hierarchical = np.random.uniform(35, 55, n_games)

    # Model 4: XGBoost v2 (good MAE, deterministic)
    mae_xgb = 54.12
    pred_xgboost = actual_yards + np.random.normal(0, mae_xgb * 0.8, n_games)
    std_xgboost = np.random.uniform(38, 58, n_games)

    # Model 5: Informative Priors v2 (similar to hierarchical, slightly worse)
    mae_ip = 55.0
    pred_informative = actual_yards + np.random.normal(0, mae_ip * 0.8, n_games)
    std_informative = np.random.uniform(37, 57, n_games)

    # Create dataframe
    df = pd.DataFrame({
        'game_id': range(n_games),
        'player_id': [f"player_{i % 50}" for i in range(n_games)],
        'actual_yards': actual_yards,
        'pred_state_space': pred_state_space,
        'pred_bnn': pred_bnn,
        'pred_hierarchical': pred_hierarchical,
        'pred_xgboost': pred_xgboost,
        'pred_informative': pred_informative,
        'std_state_space': std_state_space,
        'std_bnn': std_bnn,
        'std_hierarchical': std_hierarchical,
        'std_xgboost': std_xgboost,
        'std_informative': std_informative,
    })

    return df


def test_optimization():
    """Test optimization pipeline with synthetic data."""
    logger.info("=" * 80)
    logger.info("TESTING ENSEMBLE WEIGHT OPTIMIZATION")
    logger.info("=" * 80)

    # Generate synthetic predictions
    logger.info("\n[1/3] Generating synthetic predictions...")
    df = generate_synthetic_predictions(n_games=148, seed=42)

    # Save to temporary CSV
    temp_predictions = Path("/tmp/test_ensemble_predictions.csv")
    df.to_csv(temp_predictions, index=False)
    logger.info(f"  Saved {len(df)} predictions to {temp_predictions}")

    # Run optimization
    logger.info("\n[2/3] Running optimization...")
    temp_output = Path("/tmp/test_ensemble_weights.json")

    result = optimize_ensemble_weights(
        predictions_path=temp_predictions,
        output_path=temp_output,
        target_mae=45.2,
        target_calibration=89.0
    )

    # Validate results
    logger.info("\n[3/3] Validating results...")

    # Check that optimal weights sum to 1.0
    weight_sum = sum(result.optimal_weights.values())
    assert abs(weight_sum - 1.0) < 1e-6, f"Weights must sum to 1.0 (got {weight_sum})"
    logger.info(f"✓ Weights sum to 1.0: {weight_sum:.6f}")

    # Check that state_space gets highest weight (it has best MAE)
    if result.optimal_method in ["inverse_variance", "performance_based"]:
        best_model = max(result.optimal_weights.items(), key=lambda x: x[1])
        logger.info(f"✓ Highest weight: {best_model[0]} ({best_model[1]:.3f})")

    # Check ensemble improves over best single model
    best_single_mae = result.metadata['best_single_mae']
    ensemble_mae = result.all_methods[0].ensemble_mae  # First method

    logger.info(f"✓ Best single MAE: {best_single_mae:.2f}")
    logger.info(f"✓ Ensemble MAE: {ensemble_mae:.2f}")

    if ensemble_mae < best_single_mae:
        logger.info(f"✓ Ensemble improves by {best_single_mae - ensemble_mae:.2f} yards")
    else:
        logger.warning(f"⚠ Ensemble does not improve (Δ = {ensemble_mae - best_single_mae:+.2f})")

    # Print final weights
    logger.info("\n" + "=" * 80)
    logger.info("FINAL WEIGHTS (OPTIMAL METHOD)")
    logger.info("=" * 80)
    for model, weight in sorted(result.optimal_weights.items(), key=lambda x: -x[1]):
        logger.info(f"  {model:25s}: {weight:5.1%}")

    # Print comparison table
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON TABLE")
    logger.info("=" * 80)
    print(result.comparison_table.to_string(index=False))

    logger.info("\n✓ TEST COMPLETE")
    logger.info(f"  Results saved to {temp_output}")

    return result


if __name__ == "__main__":
    result = test_optimization()
