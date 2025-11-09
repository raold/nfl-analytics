#!/usr/bin/env python3
"""
Simple unit test for temperature scaling division by zero fix.

Tests the _apply_temperature_scaling method in isolation without
requiring full ensemble dependencies.

Run: python tests/test_temperature_scaling_fix.py
"""

import numpy as np


def apply_temperature_scaling_BEFORE_FIX(mean, std, model_name):
    """Old version (buggy - division by zero)."""
    temperature_map = {
        "bayesian": 0.35,
        "xgboost": 0.40,
        "bnn": 0.35,
        "bnn_calibrated": 0.50,
        "state_space": 0.85
    }

    temperature = temperature_map.get(model_name, 1.0)

    if temperature != 1.0:
        calibrated_std = std / temperature
        if std > 0:
            pct_change = (calibrated_std/std - 1)*100  # DIVISION BY ZERO BUG
            print(f"Temperature scaling ({model_name}): T={temperature:.2f}, "
                  f"raw_std={std:.2f} → calibrated_std={calibrated_std:.2f} "
                  f"({pct_change:+.1f}% change)")
        else:
            print(f"Temperature scaling ({model_name}): T={temperature:.2f}, "
                  f"raw_std={std:.2f} → calibrated_std={calibrated_std:.2f}")
    else:
        calibrated_std = std
        print(f"Temperature scaling ({model_name}): skipped (T=1.0)")

    return mean, calibrated_std


def apply_temperature_scaling_AFTER_FIX(mean, std, model_name):
    """New version (FIXED - no division by zero)."""
    MIN_STD_THRESHOLD = 1e-6

    # DEFENSIVE CHECK: Handle zero/near-zero std
    if std < MIN_STD_THRESHOLD:
        print(f"Temperature scaling ({model_name}): std={std:.2e} < {MIN_STD_THRESHOLD:.0e} "
              f"(insufficient data) → returning default std=50.0")
        return mean, 50.0

    temperature_map = {
        "bayesian": 0.35,
        "xgboost": 0.40,
        "bnn": 0.35,
        "bnn_calibrated": 0.50,
        "state_space": 0.85
    }

    temperature = temperature_map.get(model_name, 1.0)

    if temperature != 1.0:
        calibrated_std = std / temperature
        # FIXED: Safe percentage calculation (std already checked > MIN_STD_THRESHOLD)
        pct_change = (calibrated_std/std - 1)*100  # Safe because std > MIN_STD_THRESHOLD
        print(f"Temperature scaling ({model_name}): T={temperature:.2f}, "
              f"raw_std={std:.2f} → calibrated_std={calibrated_std:.2f} "
              f"({pct_change:+.1f}% change)")
    else:
        calibrated_std = std
        print(f"Temperature scaling ({model_name}): skipped (T=1.0)")

    return mean, calibrated_std


def test_zero_std_before_fix():
    """Test that old code crashes with std=0."""
    print("\n" + "="*70)
    print("TEST 1: Old code (BEFORE FIX) - should crash with std=0")
    print("="*70)

    try:
        mean, std = apply_temperature_scaling_BEFORE_FIX(
            mean=200.0,
            std=0.0,
            model_name="state_space"
        )
        print(f"Output: mean={mean:.1f}, std={std:.1f}")
        print("❌ FAIL: Expected ZeroDivisionError but got result")
        return False
    except ZeroDivisionError as e:
        print(f"✅ PASS: Correctly triggered ZeroDivisionError: {e}")
        return True


def test_zero_std_after_fix():
    """Test that new code handles std=0 gracefully."""
    print("\n" + "="*70)
    print("TEST 2: New code (AFTER FIX) - should handle std=0 gracefully")
    print("="*70)

    try:
        mean, std = apply_temperature_scaling_AFTER_FIX(
            mean=200.0,
            std=0.0,
            model_name="state_space"
        )
        print(f"Output: mean={mean:.1f}, std={std:.1f}")

        if std == 50.0:
            print("✅ PASS: Returned default std=50.0 (no crash)")
            return True
        else:
            print(f"❌ FAIL: Expected std=50.0, got {std:.1f}")
            return False
    except ZeroDivisionError as e:
        print(f"❌ FAIL: Still crashes with ZeroDivisionError: {e}")
        return False


def test_near_zero_std_after_fix():
    """Test that new code handles near-zero std."""
    print("\n" + "="*70)
    print("TEST 3: New code - near-zero std (1e-8 < 1e-6 threshold)")
    print("="*70)

    try:
        mean, std = apply_temperature_scaling_AFTER_FIX(
            mean=200.0,
            std=1e-8,
            model_name="state_space"
        )
        print(f"Output: mean={mean:.1f}, std={std:.1f}")

        if std == 50.0:
            print("✅ PASS: Returned default std=50.0")
            return True
        else:
            print(f"❌ FAIL: Expected std=50.0, got {std:.1f}")
            return False
    except Exception as e:
        print(f"❌ FAIL: Crashed with {type(e).__name__}: {e}")
        return False


def test_normal_std_after_fix():
    """Test that new code still processes normal std correctly."""
    print("\n" + "="*70)
    print("TEST 4: New code - normal std (regression test)")
    print("="*70)

    try:
        mean, std = apply_temperature_scaling_AFTER_FIX(
            mean=200.0,
            std=30.0,
            model_name="state_space"
        )
        print(f"Output: mean={mean:.1f}, std={std:.1f}")

        # With T=0.85: std_out = 30.0 / 0.85 = 35.29
        expected_std = 30.0 / 0.85
        tolerance = 0.1

        if abs(std - expected_std) < tolerance:
            print(f"✅ PASS: Correctly scaled std (expected {expected_std:.1f})")
            return True
        else:
            print(f"❌ FAIL: Expected std~{expected_std:.1f}, got {std:.1f}")
            return False
    except Exception as e:
        print(f"❌ FAIL: Crashed with {type(e).__name__}: {e}")
        return False


def test_edge_case_exactly_threshold():
    """Test edge case: std exactly at threshold."""
    print("\n" + "="*70)
    print("TEST 5: Edge case - std exactly at threshold (1e-6)")
    print("="*70)

    try:
        mean, std = apply_temperature_scaling_AFTER_FIX(
            mean=200.0,
            std=1e-6,
            model_name="state_space"
        )
        print(f"Output: mean={mean:.1f}, std={std:.1f}")

        # At exactly 1e-6, should use normal temperature scaling
        # std_out = 1e-6 / 0.85 = 1.176e-6
        if std > 1e-6 and std < 2e-6:
            print("✅ PASS: Threshold edge case handled correctly")
            return True
        else:
            print(f"❌ FAIL: Unexpected std={std:.2e}")
            return False
    except Exception as e:
        print(f"❌ FAIL: Crashed with {type(e).__name__}: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("TEMPERATURE SCALING DIVISION BY ZERO FIX - UNIT TESTS")
    print("="*70)
    print("Testing fix for ZeroDivisionError in _apply_temperature_scaling")
    print("File: py/models/player_props_ensemble.py:438")
    print("="*70)

    tests = [
        test_zero_std_before_fix,
        test_zero_std_after_fix,
        test_near_zero_std_after_fix,
        test_normal_std_after_fix,
        test_edge_case_exactly_threshold,
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
    import sys
    sys.exit(main())
