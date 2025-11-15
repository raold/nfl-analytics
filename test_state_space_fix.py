#!/usr/bin/env python3
"""
Quick test to verify state-space prediction fix.

Expected: Predictions should be ~200-300 yards (not 400-900 yards)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from py.ensemble.enhanced_ensemble_v5 import EnhancedEnsemble4Way

def test_predictions():
    """Test state-space predictions on known games."""

    ensemble = EnhancedEnsemble4Way()

    # Test cases: (player_id, season, week, expected_yards_approx)
    test_cases = [
        ("00-0034857", 2022, 1, 292),  # J.Allen vs LA (actual: 292)
        ("00-0026498", 2022, 1, 191),  # M.Stafford vs BUF (actual: 191)
        ("00-0033873", 2022, 21, 315), # P.Mahomes vs CIN (actual: 315)
    ]

    print("="*80)
    print("STATE-SPACE PREDICTION FIX TEST")
    print("="*80)
    print()

    all_pass = True

    for player_id, season, week, actual_yards in test_cases:
        result = ensemble.predict(player_id, season, week, return_individual=True)

        state_space_pred = result['predictions'].get('state_space_v2')

        if state_space_pred is None:
            print(f"❌ FAIL: No state-space prediction for {player_id} W{week} {season}")
            all_pass = False
            continue

        pred_mean, pred_std = state_space_pred

        # Check prediction is reasonable (50-500 yards)
        if pred_mean < 50 or pred_mean > 500:
            print(f"❌ FAIL: {player_id} W{week} {season}")
            print(f"   Prediction: {pred_mean:.1f} ± {pred_std:.1f} yards")
            print(f"   Actual: {actual_yards} yards")
            print(f"   ERROR: Prediction out of range (50-500 yards)")
            all_pass = False
        else:
            error = abs(pred_mean - actual_yards)
            print(f"✅ PASS: {player_id} W{week} {season}")
            print(f"   Prediction: {pred_mean:.1f} ± {pred_std:.1f} yards")
            print(f"   Actual: {actual_yards} yards")
            print(f"   Error: {error:.1f} yards")
        print()

    print("="*80)
    if all_pass:
        print("✅ ALL TESTS PASSED - State-space predictions are reasonable")
    else:
        print("❌ SOME TESTS FAILED - Check predictions above")
    print("="*80)

    return 0 if all_pass else 1

if __name__ == "__main__":
    exit(test_predictions())
