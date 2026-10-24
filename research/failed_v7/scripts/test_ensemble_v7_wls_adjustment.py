#!/usr/bin/env python3
"""
Test Ensemble v7.0 WLS Ensemble-Level Adjustment

Tests Phase B implementation:
1. _get_recent_ensemble_errors()
2. _apply_ensemble_wls()
3. Integration in predict() method

Author: Ensemble v7.0
Date: November 18, 2025
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nfl_analytics.ensemble.enhanced_ensemble_v7_production import EnhancedEnsemble7Way

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_wls_adjustment():
    """
    Test WLS ensemble-level adjustment.

    Test case: Patrick Mahomes 2024 Week 10
    """
    logger.info("=" * 80)
    logger.info("TEST: Ensemble v7.0 WLS Adjustment")
    logger.info("=" * 80)

    # Initialize ensemble
    logger.info("\n1. Initializing ensemble...")
    ensemble = EnhancedEnsemble7Way(
        use_meta_learning=False,  # Disable for isolated test
        use_wls_recency=False,  # Disable component-level WLS
    )

    # Load models
    ensemble._load_models()

    # Test case: Patrick Mahomes 2024 Week 10
    player_id = "00-0033873"
    season = 2024
    week = 10

    logger.info(f"\n2. Test case: Patrick Mahomes {season} Week {week}")

    # Get prediction WITHOUT WLS adjustment
    logger.info("\n3. Prediction WITHOUT WLS adjustment...")
    prediction_raw = ensemble.predict(
        player_id=player_id, season=season, week=week, return_components=True
    )

    if "error" in prediction_raw:
        logger.error(f"❌ Prediction failed: {prediction_raw['error']}")
        return

    # Display raw prediction
    logger.info(f"\n   Raw Ensemble Mean: {prediction_raw['ensemble_mean_raw']:.2f} yards")
    logger.info(f"   Components available: {list(prediction_raw['components'].keys())}")

    # Check if WLS adjustment was applied
    wls_metadata = prediction_raw["metadata"].get("wls_ensemble_adjustment", {})

    if wls_metadata.get("wls_applied", False):
        logger.info("\n4. WLS Adjustment Results:")
        logger.info("   ✅ WLS applied: YES")
        logger.info(f"   Adjustment: {prediction_raw['wls_adjustment']:+.2f} yards")
        logger.info(
            f"   Raw → Adjusted: {prediction_raw['ensemble_mean_raw']:.2f} → {prediction_raw['ensemble_mean']:.2f}"
        )
        logger.info(f"   Bias estimate: {wls_metadata.get('wls_bias_estimate', 0):.2f} yards")
        logger.info(f"   Recent predictions used: {wls_metadata.get('n_recent', 0)}")
        logger.info(f"   Halflife: {wls_metadata.get('halflife', 4)} weeks")

        # Validate adjustment is reasonable
        adj = abs(prediction_raw["wls_adjustment"])
        if adj > 20:
            logger.warning(f"   ⚠️  Large adjustment: {adj:.2f} yards (> 20 yards)")
        elif adj > 0:
            logger.info(f"   ✅ Adjustment magnitude reasonable: {adj:.2f} yards")
        else:
            logger.info("   ℹ️  Zero adjustment (no bias detected)")
    else:
        logger.info("\n4. WLS Adjustment Results:")
        logger.info("   ⚠️  WLS not applied (insufficient data)")
        logger.info(f"   Reason: {wls_metadata}")

    # Display final prediction
    logger.info("\n5. Final Prediction:")
    logger.info(f"   Ensemble Mean: {prediction_raw['ensemble_mean']:.2f} yards")
    logger.info(f"   Ensemble Std: {prediction_raw['ensemble_std']:.2f} yards")
    logger.info(
        f"   90% CI: [{prediction_raw['ensemble_ci_lower']:.2f}, {prediction_raw['ensemble_ci_upper']:.2f}]"
    )

    # Test _get_recent_ensemble_errors() directly
    logger.info("\n6. Direct test: _get_recent_ensemble_errors()...")
    df_errors = ensemble._get_recent_ensemble_errors(
        ensemble_version="v7.0", n_recent=50, halflife_weeks=4.0
    )

    if df_errors is not None:
        logger.info(f"   ✅ Retrieved {len(df_errors)} recent predictions")
        logger.info(f"   Columns: {list(df_errors.columns)}")
        logger.info("   Sample errors (5 most recent):")
        logger.info(df_errors.head())

        # Show statistics
        logger.info("\n   Error statistics:")
        logger.info(f"   Mean error: {df_errors['error'].mean():.2f} yards")
        logger.info(f"   Std error: {df_errors['error'].std():.2f} yards")
        logger.info(
            f"   WLS-weighted error: {(df_errors['error'] * df_errors['wls_weight']).sum():.2f} yards"
        )
    else:
        logger.warning("   ⚠️  No recent errors available")

    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    test_wls_adjustment()
