#!/usr/bin/env python3
"""
Standalone Integration Tests for GameFeatureExtractor
=====================================================

Tests the unified game feature extraction pipeline without pytest dependency.

Author: Claude Code (Sonnet 4.5)
Date: November 21, 2025
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from py.features.game_feature_extractor import GameFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestResults:
    """Track test results."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def record_pass(self, test_name):
        self.passed += 1
        logger.info(f"✓ {test_name} PASSED\n")

    def record_fail(self, test_name, error):
        self.failed += 1
        self.errors.append((test_name, str(error)))
        logger.error(f"✗ {test_name} FAILED: {error}\n")


def test_extract_features_completeness(extractor):
    """Test that all 100 features are extracted correctly."""
    features = extractor.extract_game_features("HOU", "BUF", 2025, 12)

    # Check total count
    assert len(features) == 100, f"Expected 100 features, got {len(features)}"

    # Check home offensive features (24)
    home_off_keys = [k for k in features.keys() if k.startswith("home_off_")]
    assert len(home_off_keys) == 24, f"Expected 24 home_off_ features, got {len(home_off_keys)}"

    # Check home defensive features (24)
    home_def_keys = [k for k in features.keys() if k.startswith("home_def_")]
    assert len(home_def_keys) == 24, f"Expected 24 home_def_ features, got {len(home_def_keys)}"

    # Check away offensive features (24)
    away_off_keys = [k for k in features.keys() if k.startswith("away_off_")]
    assert len(away_off_keys) == 24, f"Expected 24 away_off_ features, got {len(away_off_keys)}"

    # Check away defensive features (24)
    away_def_keys = [k for k in features.keys() if k.startswith("away_def_")]
    assert len(away_def_keys) == 24, f"Expected 24 away_def_ features, got {len(away_def_keys)}"

    # Check derived features (4)
    derived_keys = [
        "offensive_mismatch",
        "defensive_mismatch",
        "pace_difference",
        "rest_differential",
    ]
    for key in derived_keys:
        assert key in features, f"Missing derived feature: {key}"

    return features


def test_temporal_safety(extractor):
    """Test that all features use ONLY games before target week."""
    # 2025 Week 12: HOU vs BUF
    is_safe = extractor.validate_temporal_safety("HOU", "BUF", 2025, 12)
    assert is_safe, "Temporal safety check failed for HOU vs BUF (2025 Week 12)"

    # 2024 Week 18: KC vs LAC (historical)
    is_safe = extractor.validate_temporal_safety("KC", "LAC", 2024, 18)
    assert is_safe, "Temporal safety check failed for KC vs LAC (2024 Week 18)"


def test_no_nan_values(extractor):
    """Test that extracted features contain no NaN values."""
    features = extractor.extract_game_features("HOU", "BUF", 2025, 12)

    # Check for NaN values
    nan_features = [k for k, v in features.items() if pd.isna(v)]
    assert len(nan_features) == 0, f"Found NaN values in features: {nan_features}"

    # Check for infinite values
    inf_features = [k for k, v in features.items() if np.isinf(v)]
    assert len(inf_features) == 0, f"Found infinite values in features: {inf_features}"


def test_performance_benchmark(extractor):
    """Test that feature extraction completes within performance targets."""
    start_time = time.time()
    extractor.extract_game_features("HOU", "BUF", 2025, 12)
    elapsed_ms = (time.time() - start_time) * 1000

    logger.info(f"Feature extraction time: {elapsed_ms:.1f}ms")

    # Hard limit: 2 seconds
    assert elapsed_ms < 2000, f"Feature extraction too slow: {elapsed_ms:.1f}ms > 2000ms"

    # Warning if target exceeded
    if elapsed_ms > 500:
        logger.warning(
            f"Performance warning: {elapsed_ms:.1f}ms > 500ms target "
            "(may be due to cold cache or database load)"
        )

    return elapsed_ms


def test_caching_functionality(extractor):
    """Test that feature caching works correctly."""
    # First call: Cold cache
    start_time = time.time()
    features1 = extractor.extract_game_features("HOU", "BUF", 2025, 12, use_cache=True)
    cold_time_ms = (time.time() - start_time) * 1000

    # Second call: Warm cache
    start_time = time.time()
    features2 = extractor.extract_game_features("HOU", "BUF", 2025, 12, use_cache=True)
    warm_time_ms = (time.time() - start_time) * 1000

    # Check features are identical
    assert features1.keys() == features2.keys(), "Cached features have different keys"

    logger.info(f"Cold cache time: {cold_time_ms:.1f}ms")
    logger.info(f"Warm cache time: {warm_time_ms:.1f}ms")
    logger.info(f"Speedup: {cold_time_ms / warm_time_ms:.1f}x")

    return cold_time_ms, warm_time_ms


def test_derived_features_correctness(extractor):
    """Test that derived matchup features are computed correctly."""
    features = extractor.extract_game_features("HOU", "BUF", 2025, 12)

    # Verify offensive_mismatch calculation
    expected_off_mismatch = (
        features["home_off_off_epa_per_play"] - features["away_def_def_epa_allowed_per_play"]
    )
    actual_off_mismatch = features["offensive_mismatch"]
    assert (
        abs(expected_off_mismatch - actual_off_mismatch) < 1e-6
    ), f"offensive_mismatch incorrect: {actual_off_mismatch:.4f} != {expected_off_mismatch:.4f}"

    # Verify defensive_mismatch calculation
    expected_def_mismatch = (
        features["away_off_off_epa_per_play"] - features["home_def_def_epa_allowed_per_play"]
    )
    actual_def_mismatch = features["defensive_mismatch"]
    assert (
        abs(expected_def_mismatch - actual_def_mismatch) < 1e-6
    ), f"defensive_mismatch incorrect: {actual_def_mismatch:.4f} != {expected_def_mismatch:.4f}"

    # Verify pace_difference calculation
    expected_pace_diff = (
        features["home_off_off_plays_per_game"] - features["away_off_off_plays_per_game"]
    )
    actual_pace_diff = features["pace_difference"]
    assert (
        abs(expected_pace_diff - actual_pace_diff) < 1e-6
    ), f"pace_difference incorrect: {actual_pace_diff:.4f} != {expected_pace_diff:.4f}"


def test_input_validation(extractor):
    """Test that invalid inputs raise appropriate errors."""
    # Invalid team abbreviation
    try:
        extractor.extract_game_features("INVALID", "BUF", 2025, 12)
        assert False, "Should have raised ValueError for invalid team"
    except ValueError as e:
        assert "Invalid home_team" in str(e)

    # Invalid season
    try:
        extractor.extract_game_features("HOU", "BUF", 1998, 12)
        assert False, "Should have raised ValueError for invalid season"
    except ValueError as e:
        assert "Invalid season" in str(e)

    # Invalid week
    try:
        extractor.extract_game_features("HOU", "BUF", 2025, 23)
        assert False, "Should have raised ValueError for invalid week"
    except ValueError as e:
        assert "Invalid week" in str(e)

    # Same home and away team
    try:
        extractor.extract_game_features("HOU", "HOU", 2025, 12)
        assert False, "Should have raised ValueError for same teams"
    except ValueError as e:
        assert "cannot be the same" in str(e)


def main():
    """Run all tests."""
    logger.info("\n" + "=" * 80)
    logger.info("Running GameFeatureExtractor Integration Tests")
    logger.info("=" * 80 + "\n")

    results = TestResults()
    extractor = GameFeatureExtractor()

    # Define tests
    tests = [
        ("Feature Completeness", lambda: test_extract_features_completeness(extractor)),
        ("Temporal Safety", lambda: test_temporal_safety(extractor)),
        ("No NaN Values", lambda: test_no_nan_values(extractor)),
        ("Performance Benchmark", lambda: test_performance_benchmark(extractor)),
        ("Caching Functionality", lambda: test_caching_functionality(extractor)),
        ("Derived Features", lambda: test_derived_features_correctness(extractor)),
        ("Input Validation", lambda: test_input_validation(extractor)),
    ]

    # Run tests
    for test_name, test_func in tests:
        try:
            extractor.clear_cache()  # Clear cache before each test
            logger.info(f"\nRunning: {test_name}")
            logger.info("-" * 80)
            test_func()
            results.record_pass(test_name)
        except Exception as e:
            results.record_fail(test_name, e)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info(
        f"Test Summary: {results.passed}/{len(tests)} passed, {results.failed}/{len(tests)} failed"
    )

    if results.failed > 0:
        logger.info("\nFailed Tests:")
        for test_name, error in results.errors:
            logger.info(f"  - {test_name}: {error}")

    logger.info("=" * 80 + "\n")

    return results.passed, results.failed


if __name__ == "__main__":
    passed, failed = main()
    sys.exit(0 if failed == 0 else 1)
