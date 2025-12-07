#!/usr/bin/env python3
"""
Test Simplified ELO Predictor - Validate Predictions in Range

Tests the simplified ELO prediction formula (Nov 16, 2025) without player skill factors.
Verifies predictions are realistic (200-400 yards typical range).

Usage:
    python scripts/test_simplified_elo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

from py.ensemble.enhanced_ensemble_v6_elo import EnhancedEnsemble6Way

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_elo_predictions():
    """Test ELO predictions on sample 2024 games."""

    logger.info("="*80)
    logger.info("Testing Simplified ELO Predictor (No Player Skill Factors)")
    logger.info("="*80)

    # Initialize ensemble
    ensemble = EnhancedEnsemble6Way()

    # Test cases: (player_id, season, week, player_name)
    # Using 2024 Week 9 (verified player IDs from mart.player_hierarchy)
    test_cases = [
        ("00-0033873", 2024, 9, "Patrick Mahomes"),  # KC QB - 291 yards actual
        ("00-0039163", 2024, 9, "C.J. Stroud"),      # HOU QB
        ("00-0034855", 2024, 9, "Baker Mayfield"),   # TB QB
        ("00-0023459", 2024, 9, "Aaron Rodgers"),    # NYJ QB
        ("00-0039732", 2024, 9, "Bo Nix"),           # DEN QB (rookie)
    ]

    results = []

    for player_id, season, week, name in test_cases:
        try:
            # Get ELO prediction only
            elo_pred = ensemble._get_elo_prediction(player_id, season, week)

            if elo_pred is not None:
                mean, std = elo_pred
                results.append({
                    'player': name,
                    'mean': mean,
                    'std': std,
                    'status': 'PASS' if 150 <= mean <= 450 else 'FAIL'
                })
                logger.info(f"  {name:20s}: {mean:6.1f} ± {std:5.1f} yards [{results[-1]['status']}]")
            else:
                logger.warning(f"  {name:20s}: FAILED (no prediction)")
                results.append({
                    'player': name,
                    'mean': None,
                    'std': None,
                    'status': 'ERROR'
                })
        except Exception as e:
            logger.error(f"  {name:20s}: ERROR - {e}")
            results.append({
                'player': name,
                'mean': None,
                'std': None,
                'status': 'ERROR'
            })

    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)

    passed = sum(1 for r in results if r['status'] == 'PASS')
    failed = sum(1 for r in results if r['status'] == 'FAIL')
    errors = sum(1 for r in results if r['status'] == 'ERROR')

    valid_means = [r['mean'] for r in results if r['mean'] is not None]

    if valid_means:
        logger.info(f"Tests Run: {len(results)}")
        logger.info(f"  PASS:    {passed}")
        logger.info(f"  FAIL:    {failed} (predictions outside 150-450 yard range)")
        logger.info(f"  ERROR:   {errors}")
        logger.info(f"\nPrediction Range: {min(valid_means):.1f} - {max(valid_means):.1f} yards")
        logger.info(f"Average: {sum(valid_means)/len(valid_means):.1f} yards")

        if passed == len(results):
            logger.info("\n✅ ALL TESTS PASSED - ELO predictor working correctly")
            return 0
        else:
            logger.warning("\n⚠️  SOME TESTS FAILED - Review predictions above")
            return 1
    else:
        logger.error("\n❌ ALL PREDICTIONS FAILED")
        return 1

if __name__ == "__main__":
    exit(test_elo_predictions())
