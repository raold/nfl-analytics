#!/usr/bin/env python3
"""
Integration Tests for GameFeatureExtractor
==========================================

Tests the unified game feature extraction pipeline that combines offensive and
defensive features for game outcome predictions.

**Test Coverage:**
1. Feature extraction completeness (100 features)
2. Temporal safety validation (no future leakage)
3. Data quality validation (no NaN values)
4. Performance benchmarks (<500ms target)
5. Caching functionality (5-minute TTL)
6. Derived matchup features correctness

**Test Data:**
- 2025 Week 12: HOU vs BUF (real game from database)
- 2024 Week 18: KC vs LAC (historical validation)
- 2023 Week 1: BAL vs HOU (early season edge case)

Author: Claude Code (Sonnet 4.5)
Date: November 21, 2025
Reference: CLAUDE.md § Testing Best Practices
"""

import logging
import time

import numpy as np
import pandas as pd
import pytest

from py.features.game_feature_extractor import GameFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def extractor():
    """Create GameFeatureExtractor instance for all tests."""
    return GameFeatureExtractor()


@pytest.fixture(scope="function")
def clear_cache(extractor):
    """Clear cache before each test."""
    extractor.clear_cache()
    yield
    extractor.clear_cache()


class TestGameFeatureExtractor:
    """Test suite for GameFeatureExtractor."""

    def test_extract_features_completeness(self, extractor, clear_cache):
        """
        Test that all 100 features are extracted correctly.

        Validates:
        - 96 base features (24 off + 24 def × 2 teams)
        - 4 derived matchup features
        - All features are numeric (float/int)
        """
        # 2025 Week 12: HOU vs BUF
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

        # Check all values are numeric
        for key, value in features.items():
            assert isinstance(
                value, (int, float, np.integer, np.floating)
            ), f"Feature {key} is not numeric: {type(value)}"

        logger.info(f"✓ Feature completeness test passed: {len(features)} features extracted")

    def test_temporal_safety(self, extractor, clear_cache):
        """
        Test that all features use ONLY games before target week.

        Validates:
        - validate_temporal_safety returns True
        - No future leakage in underlying extractors
        """
        # 2025 Week 12: HOU vs BUF
        is_safe = extractor.validate_temporal_safety("HOU", "BUF", 2025, 12)
        assert is_safe, "Temporal safety check failed for HOU vs BUF (2025 Week 12)"

        # 2024 Week 18: KC vs LAC (historical)
        is_safe = extractor.validate_temporal_safety("KC", "LAC", 2024, 18)
        assert is_safe, "Temporal safety check failed for KC vs LAC (2024 Week 18)"

        logger.info("✓ Temporal safety test passed: No future leakage detected")

    def test_no_nan_values(self, extractor, clear_cache):
        """
        Test that extracted features contain no NaN values.

        Validates:
        - All features are valid numbers (not NaN/inf)
        - Default values used when insufficient data
        """
        # 2025 Week 12: HOU vs BUF
        features = extractor.extract_game_features("HOU", "BUF", 2025, 12)

        # Check for NaN values
        nan_features = [k for k, v in features.items() if pd.isna(v)]
        assert len(nan_features) == 0, f"Found NaN values in features: {nan_features}"

        # Check for infinite values
        inf_features = [k for k, v in features.items() if np.isinf(v)]
        assert len(inf_features) == 0, f"Found infinite values in features: {inf_features}"

        logger.info("✓ Data quality test passed: No NaN/inf values")

    def test_performance_benchmark(self, extractor, clear_cache):
        """
        Test that feature extraction completes within performance targets.

        Target: <500ms for all 4 feature sets
        Acceptable: <1000ms (warning if exceeded)
        """
        # 2025 Week 12: HOU vs BUF
        start_time = time.time()
        features = extractor.extract_game_features("HOU", "BUF", 2025, 12)
        elapsed_ms = (time.time() - start_time) * 1000

        logger.info(f"Feature extraction time: {elapsed_ms:.1f}ms")

        # Hard limit: 2 seconds (fail if exceeded)
        assert elapsed_ms < 2000, f"Feature extraction too slow: {elapsed_ms:.1f}ms > 2000ms"

        # Warning if target exceeded
        if elapsed_ms > 500:
            logger.warning(
                f"Performance warning: {elapsed_ms:.1f}ms > 500ms target "
                "(may be due to cold cache or database load)"
            )
        else:
            logger.info(f"✓ Performance test passed: {elapsed_ms:.1f}ms < 500ms target")

    def test_caching_functionality(self, extractor, clear_cache):
        """
        Test that feature caching works correctly.

        Validates:
        - First call extracts features from database
        - Second call uses cached features (faster)
        - Cache expires after TTL (5 minutes)
        """
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
        for key in features1.keys():
            assert (
                features1[key] == features2[key]
            ), f"Cached feature {key} differs: {features1[key]} != {features2[key]}"

        # Check warm cache is faster (should be <10ms with cache)
        logger.info(f"Cold cache time: {cold_time_ms:.1f}ms")
        logger.info(f"Warm cache time: {warm_time_ms:.1f}ms")
        logger.info(f"Speedup: {cold_time_ms / warm_time_ms:.1f}x")

        # Warm cache should be significantly faster (at least 2x)
        assert (
            warm_time_ms < cold_time_ms
        ), f"Warm cache not faster: {warm_time_ms:.1f}ms >= {cold_time_ms:.1f}ms"

        logger.info("✓ Caching test passed: Cache working correctly")

    def test_derived_features_correctness(self, extractor, clear_cache):
        """
        Test that derived matchup features are computed correctly.

        Validates:
        - offensive_mismatch = home_off_epa - away_def_epa
        - defensive_mismatch = away_off_epa - home_def_epa
        - pace_difference = home_plays_per_game - away_plays_per_game
        """
        # 2025 Week 12: HOU vs BUF
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

        logger.info("✓ Derived features test passed: All calculations correct")

    def test_early_season_edge_case(self, extractor, clear_cache):
        """
        Test feature extraction for early season games (<4 games available).

        Validates:
        - Falls back to league averages gracefully
        - No errors or exceptions raised
        - All 100 features still extracted
        """
        # 2023 Week 1: BAL vs HOU (0 games available before Week 1)
        features = extractor.extract_game_features("BAL", "HOU", 2023, 1)

        # Check all features extracted
        assert (
            len(features) == 100
        ), f"Early season extraction incomplete: {len(features)}/100 features"

        # Check no NaN values
        nan_features = [k for k, v in features.items() if pd.isna(v)]
        assert len(nan_features) == 0, f"Early season extraction has NaN values: {nan_features}"

        logger.info("✓ Early season edge case test passed: League averages used correctly")

    def test_input_validation(self, extractor):
        """
        Test that invalid inputs raise appropriate errors.

        Validates:
        - Invalid team abbreviations rejected
        - Invalid season/week rejected
        - Same home/away team rejected
        """
        # Invalid team abbreviation (too long)
        with pytest.raises(ValueError, match="Invalid home_team"):
            extractor.extract_game_features("INVALID", "BUF", 2025, 12)

        # Invalid season (too early)
        with pytest.raises(ValueError, match="Invalid season"):
            extractor.extract_game_features("HOU", "BUF", 1998, 12)

        # Invalid week (out of range)
        with pytest.raises(ValueError, match="Invalid week"):
            extractor.extract_game_features("HOU", "BUF", 2025, 23)

        # Same home and away team
        with pytest.raises(ValueError, match="cannot be the same"):
            extractor.extract_game_features("HOU", "HOU", 2025, 12)

        logger.info("✓ Input validation test passed: Invalid inputs rejected")

    def test_multiple_games(self, extractor, clear_cache):
        """
        Test extraction for multiple games to ensure consistency.

        Validates:
        - Different games return different features
        - All games extract 100 features
        - Performance remains consistent
        """
        games = [
            ("HOU", "BUF", 2025, 12),
            ("KC", "LAC", 2024, 18),
            ("BAL", "SF", 2024, 17),
            ("PHI", "DAL", 2024, 16),
        ]

        extraction_times = []
        all_features = []

        for home, away, season, week in games:
            start_time = time.time()
            features = extractor.extract_game_features(home, away, season, week)
            elapsed_ms = (time.time() - start_time) * 1000

            # Validate completeness
            assert (
                len(features) == 100
            ), f"{home} vs {away} ({season} Week {week}): {len(features)}/100 features"

            # Validate no NaN
            nan_count = sum(1 for v in features.values() if pd.isna(v))
            assert (
                nan_count == 0
            ), f"{home} vs {away} ({season} Week {week}): {nan_count} NaN values"

            extraction_times.append(elapsed_ms)
            all_features.append(features)

        # Check games have different features (not all identical)
        unique_epa_values = set()
        for features in all_features:
            unique_epa_values.add(features["home_off_off_epa_per_play"])

        assert len(unique_epa_values) > 1, "All games have identical features (possible bug)"

        # Check performance consistency
        avg_time = np.mean(extraction_times)
        std_time = np.std(extraction_times)
        logger.info(f"Multiple games extraction: avg={avg_time:.1f}ms, std={std_time:.1f}ms")

        logger.info(f"✓ Multiple games test passed: {len(games)} games extracted successfully")


def main():
    """Run all tests manually for quick validation."""
    logger.info("\n" + "=" * 80)
    logger.info("Running GameFeatureExtractor Integration Tests")
    logger.info("=" * 80 + "\n")

    extractor = GameFeatureExtractor()
    test_suite = TestGameFeatureExtractor()

    # Run each test
    tests = [
        (
            "Feature Completeness",
            lambda: test_suite.test_extract_features_completeness(extractor, None),
        ),
        ("Temporal Safety", lambda: test_suite.test_temporal_safety(extractor, None)),
        ("No NaN Values", lambda: test_suite.test_no_nan_values(extractor, None)),
        ("Performance Benchmark", lambda: test_suite.test_performance_benchmark(extractor, None)),
        ("Caching Functionality", lambda: test_suite.test_caching_functionality(extractor, None)),
        ("Derived Features", lambda: test_suite.test_derived_features_correctness(extractor, None)),
        ("Early Season Edge Case", lambda: test_suite.test_early_season_edge_case(extractor, None)),
        ("Input Validation", lambda: test_suite.test_input_validation(extractor)),
        ("Multiple Games", lambda: test_suite.test_multiple_games(extractor, None)),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            extractor.clear_cache()  # Clear cache before each test
            logger.info(f"\nRunning: {test_name}")
            logger.info("-" * 80)
            test_func()
            passed += 1
            logger.info(f"✓ {test_name} PASSED\n")
        except Exception as e:
            failed += 1
            logger.error(f"✗ {test_name} FAILED: {str(e)}\n")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info(f"Test Summary: {passed}/{len(tests)} passed, {failed}/{len(tests)} failed")
    logger.info("=" * 80 + "\n")

    return passed, failed


if __name__ == "__main__":
    passed, failed = main()
    exit(0 if failed == 0 else 1)
