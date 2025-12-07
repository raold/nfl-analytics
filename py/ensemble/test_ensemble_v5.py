#!/usr/bin/env python3
"""
Quick Test of 5-Way Ensemble with Canonical Models

Tests the NEW ensemble predictor on a small sample to validate:
1. All 5 models load correctly
2. Predictions are generated successfully
3. Output format is correct

Usage:
    python py/ensemble/test_ensemble_v5.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import logging

from enhanced_ensemble_v5_canonical import EnhancedEnsemble5Way

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ensemble():
    """Test ensemble on known players"""
    logger.info("="*80)
    logger.info("TESTING 5-WAY ENSEMBLE WITH CANONICAL MODELS")
    logger.info("="*80)

    # Initialize ensemble
    logger.info("\n1. Initializing ensemble...")
    ensemble = EnhancedEnsemble5Way()

    # Test players (known QBs from 2024)
    test_cases = [
        {"player_id": "00-0019596", "player_name": "Tom Brady", "season": 2022, "week": 1},  # Brady's last season
        {"player_id": "00-0033873", "player_name": "Patrick Mahomes", "season": 2024, "week": 1},
        {"player_id": "00-0036389", "player_name": "Jalen Hurts", "season": 2024, "week": 1},
    ]

    logger.info(f"\n2. Testing {len(test_cases)} predictions...\n")

    results = []
    for i, test in enumerate(test_cases, 1):
        logger.info(f"Test {i}/{len(test_cases)}: {test['player_name']} (Week {test['week']}, {test['season']})")
        logger.info("-" * 60)

        try:
            prediction = ensemble.predict(
                player_id=test['player_id'],
                season=test['season'],
                week=test['week'],
                combination_method="inverse_variance"
            )

            if prediction:
                mean = prediction.get('mean', None)
                std = prediction.get('std', None)
                predictions = prediction.get('predictions', {})

                logger.info(f"  ✓ Ensemble Mean: {mean:.1f} yards")
                logger.info(f"  ✓ Ensemble Std: {std:.1f} yards")
                logger.info(f"  ✓ 90% CI: [{mean - 1.645*std:.1f}, {mean + 1.645*std:.1f}]")
                logger.info("  ✓ Individual Models:")

                for model_name, (model_mean, model_std) in predictions.items():
                    logger.info(f"      - {model_name}: {model_mean:.1f} ± {model_std:.1f} yards")

                results.append({
                    "player": test['player_name'],
                    "mean": mean,
                    "std": std,
                    "success": True,
                    "num_models": len(predictions)
                })
                logger.info("  ✓ PASS\n")
            else:
                logger.warning("  ✗ FAIL: No prediction returned\n")
                results.append({
                    "player": test['player_name'],
                    "success": False,
                    "error": "No prediction"
                })

        except Exception as e:
            logger.error(f"  ✗ FAIL: {str(e)}\n")
            results.append({
                "player": test['player_name'],
                "success": False,
                "error": str(e)
            })

    # Summary
    logger.info("="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)

    successes = sum(1 for r in results if r.get('success', False))
    logger.info(f"Successful predictions: {successes}/{len(test_cases)}")

    if successes == len(test_cases):
        logger.info("\n✓ ALL TESTS PASSED - Ensemble ready for full backtest!")
        return True
    else:
        logger.warning(f"\n⚠️  {len(test_cases) - successes} tests failed - review errors above")
        return False


if __name__ == "__main__":
    success = test_ensemble()
    sys.exit(0 if success else 1)
