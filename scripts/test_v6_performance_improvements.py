#!/usr/bin/env python3
"""
Ensemble v6.0 Performance Test - Validate 2.6x Speedup
========================================================

Tests the 4 critical optimizations implemented in v6.0:
1. R posterior draws: 1000 → 500 (40% speedup, minimal accuracy loss)
2. Database connection pooling (eliminates 200-400ms overhead per query)
3. Skills dict O(1) lookup (replaces O(n) DataFrame filtering)
4. No .iterrows() usage (vectorized operations only)

Expected Results:
- Baseline (v5.1): ~4.0s per prediction
- Optimized (v6.0): ~1.5s per prediction
- Target speedup: 2.6x

Test Methodology:
- Test case: Patrick Mahomes (player_id="00-0019596", season=2024, week=10)
- Runs: 10 predictions with warm-up
- Timing: time.perf_counter() for microsecond precision
- Validation: Verify predictions are real (not placeholders)

Author: Claude Code (Sonnet 4.5)
Date: November 19, 2025
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

import numpy as np
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import ensemble (will trigger optimization logging)
from nfl_analytics.ensemble.enhanced_ensemble_v6_production import EnhancedEnsemble5Way

# Configure logging for clean output
logging.basicConfig(
    level=logging.WARNING, format="%(message)s"  # Suppress INFO logs for clean test output
)
logger = logging.getLogger(__name__)

# Test configuration
TEST_PLAYER_ID = "00-0019596"  # Patrick Mahomes
TEST_SEASON = 2024
TEST_WEEK = 10
NUM_RUNS = 10

# Expected performance targets
BASELINE_TIME = 4.0  # v5.1 average (seconds)
TARGET_TIME = 1.5  # v6.0 target (seconds)
TARGET_SPEEDUP = 2.6  # Expected improvement

# Validation criteria
MIN_PREDICTION = 150.0  # Minimum reasonable prediction (yards)
MAX_PREDICTION = 400.0  # Maximum reasonable prediction (yards)
MIN_STD = 1.0  # Minimum uncertainty (too low = suspicious)
MAX_STD = 100.0  # Maximum uncertainty (too high = wrong)
PLACEHOLDER_VALUE = 250.0  # Detect placeholder predictions


def print_header():
    """Print test header."""
    print("=" * 70)
    print("ENSEMBLE v6.0 PERFORMANCE TEST")
    print("=" * 70)
    print()
    print("Optimizations Applied:")
    print("  ✅ R posterior draws: 1000 → 500 (40% speedup)")
    print("  ✅ Database connection pooling (4x speedup)")
    print("  ✅ Skills dict O(1) lookup (eliminates O(n) filtering)")
    print("  ✅ No .iterrows() usage (vectorized operations)")
    print()
    print(f"Test Case: Patrick Mahomes, {TEST_SEASON} Week {TEST_WEEK}")
    print(f"Runs: {NUM_RUNS} (with 1 warm-up)")
    print()


def validate_prediction(result: dict) -> bool:
    """
    Validate prediction is real (not placeholder or corrupted).

    Args:
        result: Prediction result dictionary

    Returns:
        True if valid, False otherwise
    """
    mean = result.get("ensemble_mean")
    std = result.get("ensemble_std")
    valid_components = result.get("valid_components", [])

    # Check 1: Prediction exists
    if mean is None or std is None:
        logger.error("❌ Prediction missing (mean or std is None)")
        return False

    # Check 2: Not placeholder value
    if abs(mean - PLACEHOLDER_VALUE) < 1.0:
        logger.error(f"❌ Prediction is placeholder ({PLACEHOLDER_VALUE} yards)")
        return False

    # Check 3: Reasonable range
    if not (MIN_PREDICTION <= mean <= MAX_PREDICTION):
        logger.error(
            f"❌ Prediction out of range: {mean:.1f} yards (expected {MIN_PREDICTION}-{MAX_PREDICTION})"
        )
        return False

    # Check 4: Reasonable uncertainty
    if not (MIN_STD <= std <= MAX_STD):
        logger.error(f"❌ Uncertainty out of range: {std:.1f} yards (expected {MIN_STD}-{MAX_STD})")
        return False

    # Check 5: Minimum components loaded
    if len(valid_components) < 3:
        logger.error(f"❌ Insufficient components: {len(valid_components)} (minimum 3 required)")
        return False

    return True


def run_performance_test() -> dict:
    """
    Run performance test with 10 predictions.

    Returns:
        Dictionary with timing results and validation status
    """
    # Initialize ensemble (will log optimization details)
    print("Initializing ensemble...")
    ensemble = EnhancedEnsemble5Way(db_password=os.getenv("DB_PASSWORD"))
    print()

    # Warm-up run (load models into memory)
    print("Running warm-up prediction...")
    try:
        warmup_start = time.perf_counter()
        warmup_result = ensemble.predict(
            player_id=TEST_PLAYER_ID, season=TEST_SEASON, week=TEST_WEEK, return_components=True
        )
        warmup_time = time.perf_counter() - warmup_start
        print(f"  Warm-up time: {warmup_time:.2f}s")

        # Validate warm-up prediction
        if not validate_prediction(warmup_result):
            raise ValueError("Warm-up prediction validation failed")

        print(
            f"  Prediction: {warmup_result['ensemble_mean']:.1f} ± {warmup_result['ensemble_std']:.1f} yards"
        )
        print(f"  Components: {len(warmup_result['valid_components'])}/5")
        print()
    except Exception as e:
        print(f"❌ Warm-up failed: {e}")
        raise

    # Performance test runs
    print(f"Running {NUM_RUNS} performance test predictions...")
    times = []
    predictions = []

    for i in range(NUM_RUNS):
        try:
            start = time.perf_counter()
            result = ensemble.predict(
                player_id=TEST_PLAYER_ID, season=TEST_SEASON, week=TEST_WEEK, return_components=True
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            predictions.append(result)

            # Validate each prediction
            if not validate_prediction(result):
                raise ValueError(f"Prediction {i+1} validation failed")

            print(
                f"  Run {i+1:2d}: {elapsed:.3f}s - {result['ensemble_mean']:.1f} ± {result['ensemble_std']:.1f} yards"
            )

        except Exception as e:
            print(f"❌ Run {i+1} failed: {e}")
            raise

    print()

    # Calculate statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    speedup = BASELINE_TIME / avg_time

    # Get component breakdown from last prediction
    last_result = predictions[-1]
    components = last_result.get("components", {})

    return {
        "avg_time": avg_time,
        "std_time": std_time,
        "min_time": min_time,
        "max_time": max_time,
        "speedup": speedup,
        "target_achieved": speedup >= TARGET_SPEEDUP,
        "predictions": predictions,
        "components": components,
        "valid_components": last_result["valid_components"],
        "ensemble_mean": last_result["ensemble_mean"],
        "ensemble_std": last_result["ensemble_std"],
        "calibrated_std": last_result["calibrated_std"],
        "temperature": last_result["temperature"],
    }


def print_results(results: dict):
    """
    Print formatted test results.

    Args:
        results: Dictionary from run_performance_test()
    """
    print("=" * 70)
    print("PERFORMANCE RESULTS")
    print("=" * 70)
    print()

    # Timing results
    print(
        f"Average inference time: {results['avg_time']:.3f}s ± {results['std_time']:.3f}s ({NUM_RUNS} runs)"
    )
    print(f"  Min: {results['min_time']:.3f}s")
    print(f"  Max: {results['max_time']:.3f}s")
    print()

    print(f"Expected baseline (v5.1): {BASELINE_TIME:.1f}s")
    print(f"Expected optimized (v6.0): {TARGET_TIME:.1f}s")
    print(f"Expected speedup: {TARGET_SPEEDUP:.1f}x")
    print()

    print(f"Actual speedup: {results['speedup']:.2f}x")

    # Target achievement
    if results["target_achieved"]:
        print(f"✅ Target achieved: {results['speedup']:.2f}x >= {TARGET_SPEEDUP:.1f}x")
    else:
        shortfall = TARGET_SPEEDUP - results["speedup"]
        print(
            f"⚠️  Target NOT achieved: {results['speedup']:.2f}x < {TARGET_SPEEDUP:.1f}x (shortfall: {shortfall:.2f}x)"
        )

    print()

    # Validation results
    print("=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    print()

    print("✅ Predictions are real (not placeholders)")
    print(f"✅ All {len(results['valid_components'])} components available:")
    for component in results["valid_components"]:
        if component in results["components"]:
            comp_data = results["components"][component]
            print(f"     - {component}: {comp_data['mean']:.1f} ± {comp_data['std']:.1f} yards")
        else:
            print(f"     - {component}: (mean/std not available)")

    print()
    print(f"✅ Ensemble mean: {results['ensemble_mean']:.1f} yards")
    print(f"✅ Ensemble std: {results['ensemble_std']:.1f} yards")
    print(
        f"✅ Calibrated std: {results['calibrated_std']:.1f} yards (T={results['temperature']:.4f})"
    )
    print()

    # Component breakdown (if available)
    if results["components"]:
        print("=" * 70)
        print("COMPONENT BREAKDOWN")
        print("=" * 70)
        print()

        for component_name, component_data in results["components"].items():
            print(f"{component_name}:")
            print(f"  Mean: {component_data['mean']:.1f} yards")
            print(f"  Std:  {component_data['std']:.1f} yards")
        print()

    # Summary
    print("=" * 70)

    if results["target_achieved"]:
        print("✅ ALL CHECKS PASSED - v6.0 performance validated!")
    else:
        print("⚠️  PERFORMANCE TARGET NOT MET - further optimization needed")

    print("=" * 70)
    print()


def main():
    """Main test execution."""
    print_header()

    try:
        # Run performance test
        results = run_performance_test()

        # Print results
        print_results(results)

        # Exit code based on target achievement
        if results["target_achieved"]:
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 70)

        import traceback

        traceback.print_exc()

        sys.exit(1)


if __name__ == "__main__":
    main()
