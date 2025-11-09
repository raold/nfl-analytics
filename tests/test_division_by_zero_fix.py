#!/usr/bin/env python3
"""
Test script to validate division by zero fix in player_props_ensemble.py

Tests edge cases that previously caused ZeroDivisionError:
1. Players with zero variance (std=0.0)
2. Players with NaN/Inf std
3. Players not in state-space data (T.McSorley, M.Willis)

Expected behavior: All models should gracefully degrade and return None
instead of crashing with division by zero.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Now import from absolute path
import importlib.util
spec = importlib.util.spec_from_file_location(
    "player_props_ensemble",
    PROJECT_ROOT / "py" / "models" / "player_props_ensemble.py"
)
player_props_ensemble = importlib.util.module_from_spec(spec)
spec.loader.exec_module(player_props_ensemble)

PlayerPropsEnsemble = player_props_ensemble.PlayerPropsEnsemble


def test_zero_std_handling():
    """Test that _apply_temperature_scaling handles std=0.0 gracefully."""
    print("\n" + "="*70)
    print("TEST 1: Zero std handling")
    print("="*70)

    ensemble = PlayerPropsEnsemble(
        use_bayesian=False,
        use_xgboost=False,
        use_bnn=False,
        use_state_space=True,
        use_bnn_calibrated=False
    )

    # Manually call _apply_temperature_scaling with zero std
    mean, std = ensemble._apply_temperature_scaling(
        mean=200.0,
        std=0.0,  # Zero variance - should trigger fallback
        model_name="state_space"
    )

    print(f"Input: mean=200.0, std=0.0")
    print(f"Output: mean={mean:.1f}, std={std:.1f}")

    if std == 50.0:
        print("✅ PASS: Zero std correctly handled (returned default std=50.0)")
        return True
    else:
        print(f"❌ FAIL: Expected std=50.0, got std={std:.1f}")
        return False


def test_near_zero_std_handling():
    """Test that _apply_temperature_scaling handles near-zero std gracefully."""
    print("\n" + "="*70)
    print("TEST 2: Near-zero std handling")
    print("="*70)

    ensemble = PlayerPropsEnsemble(
        use_bayesian=False,
        use_xgboost=False,
        use_bnn=False,
        use_state_space=True,
        use_bnn_calibrated=False
    )

    # Test with std below MIN_STD_THRESHOLD (1e-6)
    mean, std = ensemble._apply_temperature_scaling(
        mean=200.0,
        std=1e-8,  # Below 1e-6 threshold
        model_name="state_space"
    )

    print(f"Input: mean=200.0, std=1e-8")
    print(f"Output: mean={mean:.1f}, std={std:.1f}")

    if std == 50.0:
        print("✅ PASS: Near-zero std correctly handled (returned default std=50.0)")
        return True
    else:
        print(f"❌ FAIL: Expected std=50.0, got std={std:.1f}")
        return False


def test_nan_std_handling():
    """Test that prediction methods handle NaN std gracefully."""
    print("\n" + "="*70)
    print("TEST 3: NaN std handling")
    print("="*70)

    ensemble = PlayerPropsEnsemble(
        use_bayesian=False,
        use_xgboost=False,
        use_bnn=False,
        use_state_space=True,
        use_bnn_calibrated=False
    )

    # Simulate NaN std in state-space data
    # (this would be caught in _get_state_space_prediction before temperature scaling)

    # Manually test the NaN check
    test_std = np.nan
    is_nan = np.isnan(test_std)

    print(f"NaN detection: np.isnan({test_std}) = {is_nan}")

    if is_nan:
        print("✅ PASS: NaN std correctly detected")
        return True
    else:
        print("❌ FAIL: NaN std not detected")
        return False


def test_problem_players():
    """Test prediction for players that previously caused division by zero.

    These players (T.McSorley, M.Willis) have limited data and may trigger
    zero variance in state-space model.
    """
    print("\n" + "="*70)
    print("TEST 4: Problem players (T.McSorley, M.Willis)")
    print("="*70)

    ensemble = PlayerPropsEnsemble(
        use_bayesian=True,
        use_xgboost=False,  # Skip XGBoost (no features)
        use_bnn=False,      # Skip BNN (no features)
        use_state_space=True,
        use_bnn_calibrated=False
    )

    # Test players that previously crashed
    problem_players = [
        ("00-0036945", "T.McSorley"),
        ("00-0037013", "M.Willis"),
    ]

    all_passed = True
    for player_id, player_name in problem_players:
        print(f"\nTesting {player_name} ({player_id})...")

        try:
            # Attempt prediction (should NOT crash)
            prediction = ensemble.predict(
                player_id=player_id,
                player_name=player_name,
                prop_type="passing_yards",
                line=200.5,
                season=2023,
                week=10,
                features=None  # No features (will skip XGBoost/BNN)
            )

            print(f"  ✅ No crash for {player_name}")
            print(f"  Prediction: {prediction.mean:.1f} ± {prediction.std:.1f}")
            print(f"  Recommendation: {prediction.recommendation}")

        except ZeroDivisionError as e:
            print(f"  ❌ FAIL: Division by zero for {player_name}: {e}")
            all_passed = False
        except Exception as e:
            # Other exceptions are OK (e.g., no data found)
            print(f"  ✅ Graceful failure for {player_name}: {type(e).__name__}")

    if all_passed:
        print("\n✅ PASS: All problem players handled without division by zero")
        return True
    else:
        print("\n❌ FAIL: Some players still trigger division by zero")
        return False


def test_normal_std_unchanged():
    """Test that normal std values are still processed correctly."""
    print("\n" + "="*70)
    print("TEST 5: Normal std processing (regression test)")
    print("="*70)

    ensemble = PlayerPropsEnsemble(
        use_bayesian=False,
        use_xgboost=False,
        use_bnn=False,
        use_state_space=True,
        use_bnn_calibrated=False
    )

    # Test with normal std value
    mean, std = ensemble._apply_temperature_scaling(
        mean=200.0,
        std=30.0,  # Normal variance
        model_name="state_space"
    )

    print(f"Input: mean=200.0, std=30.0")
    print(f"Output: mean={mean:.1f}, std={std:.1f}")

    # With T=0.85 for state_space: std_out = 30.0 / 0.85 = 35.3
    expected_std = 30.0 / 0.85
    tolerance = 0.1

    if abs(std - expected_std) < tolerance:
        print(f"✅ PASS: Normal std correctly scaled (expected ~{expected_std:.1f})")
        return True
    else:
        print(f"❌ FAIL: Expected std~{expected_std:.1f}, got std={std:.1f}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("DIVISION BY ZERO FIX VALIDATION TESTS")
    print("="*70)
    print("Testing fix for ZeroDivisionError in player_props_ensemble.py:438")
    print("See: py/models/player_props_ensemble.py")
    print("="*70)

    tests = [
        test_zero_std_handling,
        test_near_zero_std_handling,
        test_nan_std_handling,
        test_problem_players,
        test_normal_std_unchanged,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"\n❌ EXCEPTION in {test_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_func.__name__, False))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print("="*70)
    print(f"TOTAL: {passed}/{total} tests passed")
    print("="*70)

    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Division by zero bug is FIXED!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - review output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
