#!/usr/bin/env python3
"""
Test Optimization Modules
==========================

Quick test suite for database pool, feature cache, and optimized backtest.

Usage:
    python scripts/test_optimization_modules.py

Expected: All tests pass in <30 seconds

Author: Claude Code (Sonnet 4.5)
Date: November 13, 2025
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_database_pool():
    """Test database connection pooling."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 1: Database Connection Pool")
    logger.info("=" * 70)

    try:
        from py.utils.database_pool import DatabasePoolManager

        # Create pool
        pool = DatabasePoolManager(minconn=2, maxconn=5)

        # Test query
        result = pool.execute_query("SELECT COUNT(*) FROM games", fetchone=True)
        game_count = result[0]
        logger.info(f"  ✅ Query executed: {game_count:,} games in database")

        # Test connection reuse (should be fast)
        start = time.time()
        for i in range(10):
            pool.execute_query("SELECT 1", fetchone=True)
        elapsed = time.time() - start
        logger.info(f"  ✅ 10 queries executed in {elapsed*1000:.1f}ms ({elapsed*100:.1f}ms avg)")

        # Verify speed improvement
        if elapsed < 0.1:  # Should be <100ms for 10 queries
            logger.info("  ✅ Connection pooling working (fast queries)")
        else:
            logger.warning(f"  ⚠️  Queries slower than expected ({elapsed*1000:.1f}ms)")

        # Cleanup
        pool.close_all_connections()

        logger.info("  ✅ Database pool test PASSED")
        return True

    except Exception as e:
        logger.error(f"  ❌ Database pool test FAILED: {e}")
        return False


def test_feature_cache():
    """Test feature caching."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Feature Cache")
    logger.info("=" * 70)

    try:
        from py.utils.feature_cache import FeatureCache

        # Create cache
        cache = FeatureCache(max_size=100, ttl_seconds=10)

        # Test basic get/set
        player_id = "00-0019596"
        season = 2024
        week = 9
        features = {"yards": 275, "tds": 2, "attempts": 35}

        cache.set(player_id, season, week, features)
        cached_features = cache.get(player_id, season, week)

        if cached_features == features:
            logger.info("  ✅ Basic get/set working")
        else:
            logger.error("  ❌ Cache get/set failed")
            return False

        # Test cache miss
        missing = cache.get("00-0000000", 2024, 1)
        if missing is None:
            logger.info("  ✅ Cache miss handled correctly")
        else:
            logger.error("  ❌ Cache miss should return None")
            return False

        # Test LRU eviction
        for i in range(150):
            cache.set(f"player_{i}", 2024, i % 18, {"value": i})

        if len(cache.cache) <= 100:
            logger.info(f"  ✅ LRU eviction working (size={len(cache.cache)} ≤ 100)")
        else:
            logger.error(f"  ❌ LRU eviction failed (size={len(cache.cache)} > 100)")
            return False

        # Test hit rate calculation
        cache.clear()
        cache.stats = {"hits": 50, "misses": 50, "evictions": 0, "ttl_expirations": 0}
        hit_rate = cache.get_hit_rate()

        if 0.49 < hit_rate < 0.51:
            logger.info(f"  ✅ Hit rate calculation working ({hit_rate*100:.1f}%)")
        else:
            logger.error(f"  ❌ Hit rate calculation failed ({hit_rate*100:.1f}% ≠ 50%)")
            return False

        logger.info("  ✅ Feature cache test PASSED")
        return True

    except Exception as e:
        logger.error(f"  ❌ Feature cache test FAILED: {e}")
        return False


def test_optimized_backtest_imports():
    """Test that optimized backtest can be imported."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 3: Optimized Backtest Imports")
    logger.info("=" * 70)

    try:
        from py.backtest.historical_validation_4way_optimized import OptimizedHistoricalValidator

        logger.info("  ✅ OptimizedHistoricalValidator imported successfully")

        # Test initialization (no backtest run)
        validator = OptimizedHistoricalValidator(
            start_season=2024,
            end_season=2024,
            end_week=1,
            initial_bankroll=10000.0,
        )

        logger.info("  ✅ Validator initialized successfully")

        # Test pool and cache exist
        if validator.db_pool is not None:
            logger.info("  ✅ Database pool initialized")
        else:
            logger.error("  ❌ Database pool not initialized")
            return False

        if validator.feature_cache is not None:
            logger.info("  ✅ Feature cache initialized")
        else:
            logger.error("  ❌ Feature cache not initialized")
            return False

        # Cleanup
        validator.close()

        logger.info("  ✅ Optimized backtest import test PASSED")
        return True

    except Exception as e:
        logger.error(f"  ❌ Optimized backtest import test FAILED: {e}")
        return False


def test_benchmark_imports():
    """Test that benchmark script can be imported."""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: Benchmark Script Imports")
    logger.info("=" * 70)

    try:
        # Test imports (don't run actual benchmark)
        import scripts.benchmark_ensemble_optimization as benchmark

        logger.info("  ✅ Benchmark script imported successfully")

        # Check required functions exist
        required_functions = [
            "run_optimized_backtest",
            "validate_results",
            "plot_timing_comparison",
            "generate_report",
        ]

        for func_name in required_functions:
            if hasattr(benchmark, func_name):
                logger.info(f"  ✅ Function '{func_name}' exists")
            else:
                logger.error(f"  ❌ Function '{func_name}' missing")
                return False

        logger.info("  ✅ Benchmark import test PASSED")
        return True

    except Exception as e:
        logger.error(f"  ❌ Benchmark import test FAILED: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "=" * 70)
    logger.info("OPTIMIZATION MODULES TEST SUITE")
    logger.info("=" * 70)

    start_time = time.time()

    tests = [
        ("Database Pool", test_database_pool),
        ("Feature Cache", test_feature_cache),
        ("Optimized Backtest Imports", test_optimized_backtest_imports),
        ("Benchmark Script Imports", test_benchmark_imports),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    elapsed = time.time() - start_time

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {status}: {test_name}")

    logger.info(
        f"\nTotal: {passed_count}/{total_count} tests passed ({passed_count/total_count*100:.0f}%)"
    )
    logger.info(f"Time: {elapsed:.1f}s")

    if passed_count == total_count:
        logger.info("\n🎉 All tests passed! Optimization modules are working correctly.")
        return 0
    else:
        logger.error(
            f"\n❌ {total_count - passed_count} test(s) failed. Please review errors above."
        )
        return 1


if __name__ == "__main__":
    exit(main())
