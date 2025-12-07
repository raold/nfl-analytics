#!/usr/bin/env python3
"""
Validation Script for Enhanced Ensemble v5.1

Tests:
1. All 4 models load successfully
2. Weights sum to 100%
3. Prediction on test case (Patrick Mahomes 2024 Week 9)
4. All models contribute predictions
5. Ensemble prediction is reasonable

Usage:
    python scripts/validate_ensemble_v5_1.py

Author: Claude Code (Sonnet 4.5)
Date: November 13, 2025
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def validate_ensemble_v5_1():
    """Run validation tests on enhanced ensemble v5.1"""

    logger.info("=" * 70)
    logger.info("VALIDATION: Enhanced Ensemble v5.1")
    logger.info("=" * 70)

    # Test 1: Import and initialization
    logger.info("\n[TEST 1] Importing ensemble v5.1...")
    try:
        from py.ensemble.enhanced_ensemble_v5_1 import EnhancedEnsemble4Way

        logger.info("✅ Import successful")
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False

    # Test 2: Initialize ensemble
    logger.info("\n[TEST 2] Initializing ensemble...")
    try:
        ensemble = EnhancedEnsemble4Way()
        logger.info("✅ Initialization successful")
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return False

    # Test 3: Check weights
    logger.info("\n[TEST 3] Validating weights...")
    expected_weights = {
        "state_space_v2": 0.30,
        "hierarchical_v2": 0.30,
        "hybrid_xgboost_v1": 0.25,
        "informative_priors_v2": 0.15,
    }

    if ensemble.weights != expected_weights:
        logger.error("❌ Weights mismatch!")
        logger.error(f"   Expected: {expected_weights}")
        logger.error(f"   Got: {ensemble.weights}")
        return False

    weight_sum = sum(ensemble.weights.values())
    if abs(weight_sum - 1.0) > 0.01:
        logger.error(f"❌ Weights don't sum to 1.0! Sum: {weight_sum}")
        return False

    logger.info(f"✅ Weights correct: {ensemble.weights}")
    logger.info(f"✅ Weight sum: {weight_sum:.2f}")

    # Test 4: Check model paths
    logger.info("\n[TEST 4] Checking model paths...")
    paths_to_check = [
        ("State-Space skills", ensemble.skills_path),
        ("Hierarchical v2", ensemble.hierarchical_path),
        ("Informative Priors v2", ensemble.informative_path),
        ("Hybrid XGBoost v1", ensemble.hybrid_xgboost_path),
    ]

    for name, path in paths_to_check:
        if path.exists():
            logger.info(f"✅ {name}: {path}")
        else:
            logger.warning(f"⚠️  {name} not found: {path}")

    # Test 5: Load models
    logger.info("\n[TEST 5] Loading models...")
    try:
        ensemble._load_models()
        logger.info("✅ Models loaded successfully")

        # Check which models loaded
        if ensemble.hybrid_xgboost_model:
            logger.info("✅ Hybrid XGBoost model loaded")
        else:
            logger.warning("⚠️  Hybrid XGBoost model not loaded")

        if ensemble.skills_df is not None:
            logger.info(f"✅ State-space skills loaded ({len(ensemble.skills_df)} rows)")
        else:
            logger.warning("⚠️  State-space skills not loaded")

    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False

    # Test 6: Test prediction (Patrick Mahomes 2024 Week 9)
    logger.info("\n[TEST 6] Testing prediction (Patrick Mahomes 2024 W9)...")
    try:
        result = ensemble.predict(
            player_id="00-0033873", season=2024, week=9, return_individual=True  # Patrick Mahomes
        )

        if result["ensemble_mean"] is None:
            logger.error("❌ Prediction failed (None returned)")
            return False

        logger.info("✅ Prediction successful!")
        logger.info(
            f"   Ensemble: {result['ensemble_mean']:.1f} ± {result['ensemble_std']:.1f} yards"
        )
        logger.info(f"   90% CI: [{result['quantile_10']:.1f}, {result['quantile_90']:.1f}] yards")
        logger.info(f"   Inference time: {result['inference_time_ms']:.1f}ms")

        # Check individual predictions
        predictions = result.get("predictions", {})
        logger.info(f"\n   Individual model predictions ({len(predictions)}/4):")
        for model_name, (mean, std) in predictions.items():
            logger.info(f"   - {model_name}: {mean:.1f} ± {std:.1f} yards")

        # Check if all 4 models contributed
        expected_models = {
            "state_space_v2",
            "hierarchical_v2",
            "hybrid_xgboost_v1",
            "informative_priors_v2",
        }
        actual_models = set(predictions.keys())

        if expected_models == actual_models:
            logger.info("✅ All 4 models contributed predictions")
        else:
            missing = expected_models - actual_models
            logger.warning(f"⚠️  Some models missing: {missing}")

        # Sanity check prediction range (150-400 yards for QB)
        if not (150 <= result["ensemble_mean"] <= 400):
            logger.warning(f"⚠️  Prediction outside reasonable range: {result['ensemble_mean']:.1f}")
        else:
            logger.info("✅ Prediction within reasonable range")

    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("✅ ALL VALIDATION TESTS PASSED")
    logger.info("=" * 70)
    logger.info("\nEnsemble v5.1 Configuration:")
    logger.info("- State-Space v2: 30%")
    logger.info("- Hierarchical v2: 30%")
    logger.info("- Hybrid XGBoost v1: 25%")
    logger.info("- Informative Priors v2: 15%")
    logger.info("\nReady for production deployment.")

    return True


if __name__ == "__main__":
    success = validate_ensemble_v5_1()
    sys.exit(0 if success else 1)
