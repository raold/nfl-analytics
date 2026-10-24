#!/usr/bin/env python3
"""
Quick test for Ensemble v7.0 single prediction with all 5 base models.

Tests that we're generating REAL predictions (not placeholders) from:
- XGBoost v4 (Ridge + residual)
- Hierarchical v2 (R Bayesian)
- Informative Priors v2 (R Bayesian)
- State-Space v2 (R Bayesian with skills)
- ELO ratings (simple conversion)
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nfl_analytics.ensemble.enhanced_ensemble_v7_production import load_ensemble_v7

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_single_prediction():
    """Test single prediction with all components."""
    logger.info("=" * 80)
    logger.info("TESTING ENSEMBLE V7.0 - SINGLE PREDICTION")
    logger.info("=" * 80)

    # Load ensemble with full v7.0 configuration
    logger.info("\n1. Loading ensemble v7.0 (meta + WLS)...")
    ensemble = load_ensemble_v7(use_meta_learning=True, use_wls_recency=True, meta_wls_blend=0.6)

    # Test case: Patrick Mahomes, 2024 Week 10
    # (Recent game, known good data)
    test_player_id = "00-0033873"  # Patrick Mahomes
    test_season = 2024
    test_week = 10

    logger.info("\n2. Generating prediction for:")
    logger.info(f"   Player ID: {test_player_id} (Patrick Mahomes)")
    logger.info(f"   Season:    {test_season}")
    logger.info(f"   Week:      {test_week}")

    # Get prediction with components visible
    try:
        result = ensemble.predict(
            player_id=test_player_id, season=test_season, week=test_week, return_components=True
        )

        if "error" in result:
            logger.error(f"\n❌ PREDICTION FAILED: {result['error']}")
            logger.error(f"   Available components: {result.get('available_components', 0)}/5")
            logger.error(f"   Required minimum: {result.get('required_minimum', 3)}")
            return False

        logger.info("\n3. Prediction Results:")
        logger.info(f"   Ensemble Mean:  {result['ensemble_mean']:.1f} yards")
        logger.info(f"   Ensemble Std:   {result['ensemble_std']:.1f} yards")
        logger.info(
            f"   90% CI:         [{result['ensemble_ci_lower']:.1f}, {result['ensemble_ci_upper']:.1f}]"
        )

        logger.info("\n4. Component Predictions:")
        if "components" in result:
            for component_name, pred in result["components"].items():
                logger.info(
                    f"   {component_name:25s}: {pred['mean']:6.1f} ± {pred['std']:5.1f} yards"
                )

        logger.info("\n5. Adaptive Weights:")
        for component_name, weight in result["weights"].items():
            logger.info(f"   {component_name:25s}: {weight:.3f} ({weight*100:.1f}%)")

        logger.info("\n6. Metadata:")
        for key, value in result["metadata"].items():
            logger.info(f"   {key:25s}: {value}")

        # Validate: Check that predictions are NOT placeholders
        logger.info("\n7. Validation Checks:")

        checks_passed = 0
        checks_total = 3

        # Check 1: Ensemble mean not exactly 250.0 (placeholder value)
        if abs(result["ensemble_mean"] - 250.0) > 1.0:
            logger.info(
                f"   ✅ Ensemble mean ({result['ensemble_mean']:.1f}) is NOT placeholder (250.0)"
            )
            checks_passed += 1
        else:
            logger.warning(
                f"   ⚠️  Ensemble mean ({result['ensemble_mean']:.1f}) suspiciously close to placeholder"
            )

        # Check 2: At least 3 components available
        num_components = len(result.get("components", {}))
        if num_components >= 3:
            logger.info(f"   ✅ {num_components}/5 components available (minimum 3 required)")
            checks_passed += 1
        else:
            logger.warning(f"   ⚠️  Only {num_components}/5 components available")

        # Check 3: Component predictions vary (not all same)
        if "components" in result and len(result["components"]) >= 2:
            means = [p["mean"] for p in result["components"].values()]
            variance = max(means) - min(means)
            if variance > 5.0:
                logger.info(f"   ✅ Component predictions vary ({variance:.1f} yard range)")
                checks_passed += 1
            else:
                logger.warning(
                    f"   ⚠️  Component predictions suspiciously similar ({variance:.1f} yard range)"
                )

        logger.info("\n" + "=" * 80)
        if checks_passed == checks_total:
            logger.info(f"✅ ALL VALIDATION CHECKS PASSED ({checks_passed}/{checks_total})")
            logger.info("✅ Ensemble v7.0 is generating REAL predictions (not placeholders)")
            logger.info("=" * 80)
            return True
        else:
            logger.warning(f"⚠️  SOME VALIDATION CHECKS FAILED ({checks_passed}/{checks_total})")
            logger.warning("⚠️  Review output above for details")
            logger.info("=" * 80)
            return False

    except Exception as e:
        logger.error(f"\n❌ EXCEPTION OCCURRED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_single_prediction()
    sys.exit(0 if success else 1)
