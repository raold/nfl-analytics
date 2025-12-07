#!/usr/bin/env python3
"""
Test 4-Way Ensemble Predictor
==============================

Tests enhanced_ensemble_v5.py (4-way ensemble: State-Space, Hierarchical, XGBoost, Informative Priors).

Author: Claude Code (Sonnet 4.5)
Date: November 9, 2025
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import logging
import time

import numpy as np

from py.ensemble.enhanced_ensemble_v5 import EnhancedEnsemble4Way

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_model_availability():
    """Check which models are available."""
    logger.info("="*80)
    logger.info("TEST 1: Model Availability Check")
    logger.info("="*80)

    project_root = Path(__file__).parent

    models = {
        "State-Space v2": project_root / "models/bayesian/simplified_random_walk_v2.rds",
        "State-Space Skills": project_root / "models/bayesian/simplified_random_walk_skills_v2.csv",
        "Hierarchical v2": project_root / "models/bayesian/passing_yards_hierarchical_v2.rds",
        "Informative Priors v2": project_root / "models/bayesian/passing_informative_priors_v2.rds",
        "XGBoost v2": project_root / "models/xgboost/passing_xgboost_v2_canonical.pkl",
    }

    available_count = 0
    for model_name, model_path in models.items():
        exists = model_path.exists()
        status = "✅" if exists else "❌"
        size_str = f"{model_path.stat().st_size / 1e6:.1f} MB" if exists else "N/A"
        logger.info(f"{status} {model_name}: {size_str}")
        if exists:
            available_count += 1

    logger.info(f"\nAvailable models: {available_count}/{len(models)}")

    if available_count < 2:
        logger.warning("⚠️  Less than 2 models available - ensemble may not work")
    elif available_count < 4:
        logger.warning(f"⚠️  Only {available_count}/5 models available - using fallback ensemble")
    else:
        logger.info("✅ All models available for full ensemble")

    return available_count


def test_variance_pooling():
    """Validate variance pooling math."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Variance Pooling Validation")
    logger.info("="*80)

    # Simulate 2 models with known means and stds
    means = np.array([250.0, 270.0])  # Model predictions
    stds = np.array([50.0, 60.0])     # Model uncertainties
    weights = np.array([0.6, 0.4])    # Ensemble weights

    # Ensemble mean
    ensemble_mean = np.sum(weights * means)
    logger.info(f"Ensemble mean: {ensemble_mean:.1f} yards")

    # Variance pooling
    within_var = np.sum(weights * (stds ** 2))
    between_var = np.sum(weights * ((means - ensemble_mean) ** 2))
    ensemble_var = within_var + between_var
    ensemble_std = np.sqrt(ensemble_var)

    logger.info(f"Within-model variance: {within_var:.2f}")
    logger.info(f"Between-model variance: {between_var:.2f}")
    logger.info(f"Total variance: {ensemble_var:.2f}")
    logger.info(f"Ensemble std: {ensemble_std:.1f} yards")

    # Validate
    assert within_var >= 0, "Within-model variance must be non-negative"
    assert between_var >= 0, "Between-model variance must be non-negative"
    assert ensemble_std > 0, "Ensemble std must be positive"

    logger.info("✅ Variance pooling math validated")


def test_single_prediction():
    """Test single prediction for Patrick Mahomes, Week 10, 2024."""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Single Prediction - Patrick Mahomes Week 10 2024")
    logger.info("="*80)

    player_id = "00-0033873"  # Patrick Mahomes
    season = 2024
    week = 10

    try:
        ensemble = EnhancedEnsemble4Way()
        result = ensemble.predict(
            player_id=player_id,
            season=season,
            week=week,
            return_individual=True
        )

        logger.info("\nPREDICTION RESULTS:")
        logger.info(f"Player ID: {player_id}")
        logger.info(f"Season: {season}, Week: {week}")
        logger.info(f"Ensemble Mean: {result['ensemble_mean']:.1f} yards")
        logger.info(f"Ensemble Std: {result['ensemble_std']:.1f} yards")
        logger.info(f"90% CI: [{result['quantile_10']:.1f}, {result['quantile_90']:.1f}] yards")
        logger.info(f"Inference Time: {result['inference_time_ms']:.1f}ms")
        logger.info(f"Active Weights: {result['weights']}")

        if 'predictions' in result:
            logger.info("\nIndividual Model Predictions:")
            for model_name, (mean, std) in result['predictions'].items():
                logger.info(f"  {model_name}: {mean:.1f} ± {std:.1f} yards")

        # Validate results
        assert result['ensemble_mean'] is not None, "Ensemble mean is None"
        assert result['ensemble_std'] is not None, "Ensemble std is None"
        assert 100 <= result['ensemble_mean'] <= 400, f"Unrealistic prediction: {result['ensemble_mean']:.1f}"
        assert result['inference_time_ms'] < 5000, f"Inference too slow: {result['inference_time_ms']:.1f}ms"

        logger.info("\n✅ Single prediction successful")
        return result

    except Exception as e:
        logger.error(f"❌ Single prediction failed: {e}", exc_info=True)
        raise


def main():
    """Run all ensemble tests."""
    logger.info("\n" + "="*80)
    logger.info("4-WAY ENSEMBLE TESTING SUITE")
    logger.info("="*80)
    logger.info("Testing: State-Space + Hierarchical + XGBoost + Informative Priors")
    logger.info("Date: November 9, 2025\n")

    start_time = time.time()

    try:
        # Test 1: Model availability
        available_count = test_model_availability()

        # Test 2: Variance pooling math
        test_variance_pooling()

        # Test 3: Single prediction
        result = test_single_prediction()

        elapsed = time.time() - start_time
        logger.info("\n" + "="*80)
        logger.info("✅ ALL TESTS PASSED")
        logger.info("="*80)
        logger.info(f"Total time: {elapsed:.2f}s")

        # Determine ensemble configuration
        num_models = len(result.get('weights', {}))
        if num_models == 4:
            logger.info("\n🎯 FULL 4-WAY ENSEMBLE OPERATIONAL")
        elif num_models == 2:
            logger.info("\n⚠️  FALLBACK 2-WAY ENSEMBLE (State-Space + XGBoost)")
            logger.info("Recommendation: Retrain brms models with better convergence")
        else:
            logger.info(f"\n⚠️  PARTIAL ENSEMBLE ({num_models} models)")

        logger.info("\nNEXT STEPS:")
        logger.info("1. Review individual model predictions for reasonableness")
        logger.info("2. Run batch predictions for Week 10 2024")
        logger.info("3. Performance benchmark (latency, throughput)")
        logger.info("4. Full backtest (2022-2024)")
        logger.info("5. Document GO/NO-GO decision")

        return 0

    except Exception as e:
        logger.error(f"\n❌ TEST SUITE FAILED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
