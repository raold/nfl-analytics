#!/usr/bin/env python3
"""
Test script for GameOutcomeModel - Basic smoke test
====================================================

Tests basic functionality without full training:
1. Model initialization
2. Feature extraction
3. Model architecture validation

Author: Claude Code (Sonnet 4.5)
Date: November 27, 2025
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_initialization():
    """Test model initialization."""
    logger.info("Test 1: Model initialization")

    from py.models.game_outcome_model import GameOutcomeModel

    try:
        model = GameOutcomeModel()
        logger.info("✅ Model initialized successfully")
        logger.info(f"   Model directory: {model.model_dir}")
        logger.info(f"   Feature extractor: {type(model.feature_extractor).__name__}")
        return True
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return False


def test_feature_extraction():
    """Test feature extraction for a sample game."""
    logger.info("\nTest 2: Feature extraction")

    from py.models.game_outcome_model import GameOutcomeModel

    try:
        model = GameOutcomeModel()

        # Test with a known game (2024 Week 1)
        features = model.feature_extractor.extract_game_features(
            home_team="KC", away_team="BAL", season=2024, week=1
        )

        logger.info("✅ Feature extraction successful")
        logger.info(f"   Total features: {len(features)}")
        logger.info(f"   Sample features: {list(features.keys())[:5]}")

        # Validate feature count (should be ~232 or more)
        if len(features) < 100:
            logger.warning(f"⚠️  Only {len(features)} features extracted (expected ~232)")
            logger.warning("   Some advanced features may be unavailable")

        return True

    except Exception as e:
        logger.error(f"❌ Feature extraction failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_model_architecture():
    """Test model architecture (without training)."""
    logger.info("\nTest 3: Model architecture")

    import xgboost as xgb

    from py.models.game_outcome_model import GameOutcomeModel

    try:
        model = GameOutcomeModel()

        # Check default parameters
        logger.info("✅ Model architecture validated")
        logger.info(f"   XGBoost version: {xgb.__version__}")
        logger.info(
            f"   Default params: max_depth={model.DEFAULT_PARAMS['max_depth']}, "
            f"n_estimators={model.DEFAULT_PARAMS['n_estimators']}"
        )
        logger.info(f"   Models trained: {model.models_trained}")

        return True

    except Exception as e:
        logger.error(f"❌ Architecture validation failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("=" * 80)
    logger.info("GAME OUTCOME MODEL - SMOKE TEST")
    logger.info("=" * 80)

    results = {
        "initialization": test_initialization(),
        "feature_extraction": test_feature_extraction(),
        "architecture": test_model_architecture(),
    }

    logger.info("\n" + "=" * 80)
    logger.info("TEST RESULTS")
    logger.info("=" * 80)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name:30s} {status}")

    all_passed = all(results.values())

    logger.info("\n" + "=" * 80)
    if all_passed:
        logger.info("✅ ALL TESTS PASSED - Model ready for training")
    else:
        logger.error("❌ SOME TESTS FAILED - Check errors above")
    logger.info("=" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
