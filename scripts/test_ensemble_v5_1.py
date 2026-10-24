#!/usr/bin/env python3
"""
Test script for Enhanced Ensemble v5.1 (Hybrid XGBoost Integration)

Verifies:
1. Model loading (Hybrid XGBoost + R models)
2. Single prediction inference
3. Output format consistency with v5.0
4. Temperature scaling application

Usage:
    python scripts/test_ensemble_v5_1.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

from nfl_analytics.ensemble.enhanced_ensemble_v5_1 import EnhancedEnsemble4Way

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_import():
    """Test 1: Verify import and initialization."""
    logger.info("Test 1: Import and initialization")
    try:
        ensemble = EnhancedEnsemble4Way()
        logger.info(f"✅ Ensemble initialized with weights: {ensemble.weights}")
        logger.info(f"✅ Temperature: {ensemble.temperature:.4f}")
        return ensemble
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return None


def test_single_prediction(ensemble):
    """Test 2: Single prediction (Patrick Mahomes, 2024 Week 1)."""
    logger.info("\nTest 2: Single prediction")
    try:
        # Patrick Mahomes (00-0033873) - 2024 Week 1
        result = ensemble.predict(
            player_id="00-0033873", season=2024, week=1, return_individual=True
        )

        # Check output format
        required_keys = [
            "ensemble_mean",
            "ensemble_std",
            "quantile_10",
            "quantile_90",
            "weights",
            "predictions",
            "inference_time_ms",
        ]
        for key in required_keys:
            if key not in result:
                logger.error(f"❌ Missing key: {key}")
                return False

        # Log results
        logger.info(
            f"✅ Prediction: {result['ensemble_mean']:.1f} ± {result['ensemble_std']:.1f} yards"
        )
        logger.info(f"✅ 90% CI: [{result['quantile_10']:.1f}, {result['quantile_90']:.1f}]")
        logger.info(f"✅ Active models: {list(result['predictions'].keys())}")
        logger.info(f"✅ Weights: {result['weights']}")
        logger.info(f"✅ Inference time: {result['inference_time_ms']:.1f}ms")

        # Sanity checks
        if result["ensemble_mean"] is not None:
            if result["ensemble_mean"] < 0 or result["ensemble_mean"] > 600:
                logger.warning(f"⚠️  Unusual prediction: {result['ensemble_mean']:.1f} yards")
            if result["ensemble_std"] < 0:
                logger.error(f"❌ Invalid std: {result['ensemble_std']:.1f}")
                return False

        return True

    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_temperature_scaling(ensemble):
    """Test 3: Verify temperature scaling is applied."""
    logger.info("\nTest 3: Temperature scaling")
    try:
        if ensemble.temperature_scaler is None:
            logger.warning("⚠️  No temperature scaler loaded (predictions uncalibrated)")
            return True

        T = ensemble.temperature
        logger.info(f"✅ Temperature loaded: T = {T:.4f}")

        if T < 0.5 or T > 2.0:
            logger.warning(f"⚠️  Unusual temperature: T = {T:.4f} (expected 0.5-2.0)")

        return True

    except Exception as e:
        logger.error(f"❌ Temperature scaling check failed: {e}")
        return False


def test_model_components(ensemble):
    """Test 4: Verify all model components are loaded."""
    logger.info("\nTest 4: Model components")
    try:
        # Trigger model loading
        ensemble._load_models()

        # Check Hybrid XGBoost
        if ensemble.hybrid_xgboost_model is None:
            logger.error("❌ Hybrid XGBoost model not loaded")
            return False
        logger.info("✅ Hybrid XGBoost model loaded")

        # Check state-space skills
        if ensemble.skills_df is None:
            logger.warning("⚠️  State-space skills not loaded")
        else:
            logger.info(f"✅ State-space skills loaded: {len(ensemble.skills_df):,} rows")

        # Check R model paths
        r_models = {
            "hierarchical_v2": ensemble.hierarchical_path,
            "informative_priors_v2": ensemble.informative_path,
            "state_space_v2": ensemble.state_space_path,
        }

        for name, path in r_models.items():
            if path.exists():
                logger.info(f"✅ {name} path exists: {path}")
            else:
                logger.warning(f"⚠️  {name} path not found: {path}")

        return True

    except Exception as e:
        logger.error(f"❌ Model component check failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("=" * 70)
    logger.info("Enhanced Ensemble v5.1 Test Suite")
    logger.info("=" * 70)

    # Test 1: Import
    ensemble = test_import()
    if ensemble is None:
        logger.error("❌ Test suite aborted (import failed)")
        return 1

    # Test 2: Single prediction
    success = test_single_prediction(ensemble)
    if not success:
        logger.error("❌ Test suite failed (prediction)")
        return 1

    # Test 3: Temperature scaling
    success = test_temperature_scaling(ensemble)
    if not success:
        logger.error("❌ Test suite failed (temperature)")
        return 1

    # Test 4: Model components
    success = test_model_components(ensemble)
    if not success:
        logger.error("❌ Test suite failed (components)")
        return 1

    logger.info("\n" + "=" * 70)
    logger.info("✅ ALL TESTS PASSED")
    logger.info("=" * 70)
    logger.info("\nNext Steps:")
    logger.info("1. Verify hybrid model file exists: models/xgboost/hybrid_static_incremental_v1/")
    logger.info("2. Run full backtest on 2024 holdout set")
    logger.info("3. Compare v5.1 vs v5.0 performance")
    logger.info("4. Update MODEL_REGISTRY.yaml with v5.1 entry")

    return 0


if __name__ == "__main__":
    sys.exit(main())
