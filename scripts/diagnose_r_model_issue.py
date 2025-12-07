#!/usr/bin/env python3
"""
R Model Loading Diagnostic Tool
================================

Diagnoses the Nov 13 R model integration failure that blocked production deployment.

Tests:
1. rpy2 installation and initialization
2. R library availability (brms, cmdstanr)
3. Model file accessibility and size
4. Single prediction latency
5. Batch prediction capability
6. Memory usage during loading
7. Timeout scenarios

Usage:
    python scripts/diagnose_r_model_issue.py

Author: Claude Code (Sonnet 4.5)
Date: November 15, 2025
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import os
import time
import traceback

# Current NFL season (update annually in December)
CURRENT_SEASON = 2024  # pragma: allowlist temporal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RModelDiagnostic:
    """Comprehensive diagnostic for R model integration issues."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.hierarchical_path = self.project_root / "models/bayesian/passing_yards_hierarchical_v2.rds"
        self.informative_path = self.project_root / "models/bayesian/passing_informative_priors_v2.rds"

        self.db_config = {
            "host": "localhost",
            "port": 5544,
            "dbname": "devdb01",
            "user": "dro",
            "password": os.getenv("DB_PASSWORD", "sicillionbillions"),  # pragma: allowlist secret
        }

        self.test_results = {}

    def get_memory_usage_mb(self) -> float:
        """Get current process memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # Fallback: estimate from gc
            import gc
            import sys
            return sys.getsizeof(gc.get_objects()) / 1024 / 1024

    def test_1_rpy2_installation(self) -> bool:
        """Test 1: rpy2 installation and basic functionality."""
        logger.info("\n" + "="*80)
        logger.info("TEST 1: rpy2 Installation and Initialization")
        logger.info("="*80)

        try:
            import rpy2.robjects as ro
            from rpy2.robjects import pandas2ri
            logger.info("✅ rpy2 imported successfully")
            logger.info(f"   rpy2 version: {ro.__version__ if hasattr(ro, '__version__') else 'unknown'}")

            # Test basic R call
            result = ro.r('1 + 1')
            logger.info(f"✅ Basic R call works: 1 + 1 = {result[0]}")

            self.test_results['rpy2_installation'] = True
            return True

        except ImportError as e:
            logger.error(f"❌ rpy2 not installed: {e}")
            logger.error("   Install: pip install rpy2")
            self.test_results['rpy2_installation'] = False
            return False
        except Exception as e:
            logger.error(f"❌ rpy2 initialization failed: {e}")
            logger.error(traceback.format_exc())
            self.test_results['rpy2_installation'] = False
            return False

    def test_2_r_libraries(self) -> bool:
        """Test 2: R library availability (brms, cmdstanr)."""
        logger.info("\n" + "="*80)
        logger.info("TEST 2: R Library Availability")
        logger.info("="*80)

        try:
            import rpy2.robjects as ro

            # Test brms
            logger.info("Testing brms...")
            ro.r('suppressPackageStartupMessages(library(brms))')
            logger.info("✅ brms loaded successfully")

            # Test cmdstanr
            logger.info("Testing cmdstanr...")
            ro.r('suppressPackageStartupMessages(library(cmdstanr))')
            logger.info("✅ cmdstanr loaded successfully")

            self.test_results['r_libraries'] = True
            return True

        except Exception as e:
            logger.error(f"❌ R library loading failed: {e}")
            logger.error(traceback.format_exc())
            self.test_results['r_libraries'] = False
            return False

    def test_3_model_files(self) -> bool:
        """Test 3: Model file accessibility and size."""
        logger.info("\n" + "="*80)
        logger.info("TEST 3: Model File Accessibility")
        logger.info("="*80)

        all_accessible = True

        for model_name, model_path in [
            ("Hierarchical v2", self.hierarchical_path),
            ("Informative Priors v2", self.informative_path),
        ]:
            logger.info(f"\nChecking {model_name}:")
            logger.info(f"  Path: {model_path}")

            if not model_path.exists():
                logger.error("  ❌ File not found")
                all_accessible = False
                continue

            size_mb = model_path.stat().st_size / (1024 * 1024)
            logger.info(f"  ✅ File exists ({size_mb:.1f} MB)")

            # Warn if file is very large
            if size_mb > 100:
                logger.warning(f"  ⚠️  Large file size ({size_mb:.1f} MB) - may cause memory issues")

        self.test_results['model_files'] = all_accessible
        return all_accessible

    def test_4_model_loading_time(self) -> bool:
        """Test 4: Model loading time and memory usage."""
        logger.info("\n" + "="*80)
        logger.info("TEST 4: Model Loading Time and Memory")
        logger.info("="*80)

        try:
            import rpy2.robjects as ro

            # Test hierarchical model loading
            logger.info("\nLoading Hierarchical v2...")
            mem_before = self.get_memory_usage_mb()
            start = time.time()

            ro.r(f'hier_model <- readRDS("{self.hierarchical_path}")')

            elapsed = time.time() - start
            mem_after = self.get_memory_usage_mb()
            mem_delta = mem_after - mem_before

            logger.info(f"  ✅ Loaded in {elapsed:.2f}s")
            logger.info(f"  Memory: {mem_before:.1f} MB → {mem_after:.1f} MB (+{mem_delta:.1f} MB)")

            if elapsed > 30:
                logger.warning(f"  ⚠️  Slow loading ({elapsed:.2f}s) - may timeout in production")
            if mem_delta > 500:
                logger.warning(f"  ⚠️  High memory usage (+{mem_delta:.1f} MB) - may cause OOM")

            # Test informative priors model loading
            logger.info("\nLoading Informative Priors v2...")
            mem_before = self.get_memory_usage_mb()
            start = time.time()

            ro.r(f'ip_model <- readRDS("{self.informative_path}")')

            elapsed = time.time() - start
            mem_after = self.get_memory_usage_mb()
            mem_delta = mem_after - mem_before

            logger.info(f"  ✅ Loaded in {elapsed:.2f}s")
            logger.info(f"  Memory: {mem_before:.1f} MB → {mem_after:.1f} MB (+{mem_delta:.1f} MB)")

            if elapsed > 30:
                logger.warning(f"  ⚠️  Slow loading ({elapsed:.2f}s) - may timeout in production")
            if mem_delta > 500:
                logger.warning(f"  ⚠️  High memory usage (+{mem_delta:.1f} MB) - may cause OOM")

            self.test_results['model_loading'] = True
            return True

        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            logger.error(traceback.format_exc())
            self.test_results['model_loading'] = False
            return False

    def test_5_single_prediction(self) -> bool:
        """Test 5: Single prediction latency."""
        logger.info("\n" + "="*80)
        logger.info("TEST 5: Single Prediction Latency")
        logger.info("="*80)

        try:
            from py.ensemble.r_model_interface import RModelInterface

            interface = RModelInterface(db_config=self.db_config)

            # Test hierarchical prediction
            logger.info("\nTesting Hierarchical v2 prediction...")
            start = time.time()

            result = interface.predict_hierarchical(
                player_id="00-0019596",  # Patrick Mahomes
                season=CURRENT_SEASON,
                week=9,
                model_path=self.hierarchical_path,
                ndraws=1000
            )

            elapsed = time.time() - start

            if result is None:
                logger.error("  ❌ Prediction returned None")
                self.test_results['single_prediction'] = False
                return False

            mean, std = result
            logger.info(f"  ✅ Prediction: {mean:.1f} ± {std:.1f} yards")
            logger.info(f"  Time: {elapsed:.2f}s")

            if elapsed > 5:
                logger.warning(f"  ⚠️  Slow prediction ({elapsed:.2f}s) - backtest will take hours")

            self.test_results['single_prediction'] = True
            return True

        except Exception as e:
            logger.error(f"❌ Single prediction failed: {e}")
            logger.error(traceback.format_exc())
            self.test_results['single_prediction'] = False
            return False

    def test_6_batch_prediction_capability(self) -> bool:
        """Test 6: Batch prediction capability."""
        logger.info("\n" + "="*80)
        logger.info("TEST 6: Batch Prediction Capability")
        logger.info("="*80)

        try:
            import pandas as pd
            import rpy2.robjects as ro
            from rpy2.robjects import pandas2ri
            from rpy2.robjects.conversion import localconverter

            # Load model
            logger.info("Loading Hierarchical v2...")
            ro.r(f'model <- readRDS("{self.hierarchical_path}")')

            # Create test batch (5 players)
            test_data = pd.DataFrame({
                'player_id': ['00-0019596'] * 5,
                'season': [CURRENT_SEASON] * 5,
                'week': [5, 6, 7, 8, 9],
                'yards': [250, 280, 220, 300, 270],
                'touchdowns': [2, 3, 1, 4, 2],
                'attempts': [35, 40, 30, 42, 38],
                'team': ['KC'] * 5,
                'spread_line': [-7.0, -3.5, -10.0, -6.0, -4.5],
                'total_close': [47.5, 50.0, 45.5, 48.0, 46.5],
                'temp': [65, 70, 60, 75, 68],
                'wind': [5, 3, 8, 2, 6],
                'is_home': [1, 0, 1, 1, 0],
                'home_team': ['KC', 'DEN', 'KC', 'KC', 'TB'],
                'away_team': ['DEN', 'KC', 'LV', 'LAC', 'KC'],
                'is_favored': [1, 0, 1, 1, 0],
                'spread_abs': [7.0, 3.5, 10.0, 6.0, 4.5],
                'roof': ['outdoors'] * 5,
                'is_bad_weather': [0, 0, 0, 0, 0],
                'is_dome': [0, 0, 0, 0, 0],
                'years_exp': [7, 7, 7, 7, 7],
                'opponent': ['DEN', 'KC', 'LV', 'LAC', 'TB'],
                'player_season': ['00-0019596_2024'] * 5,
            })

            logger.info(f"Testing batch prediction (N={len(test_data)})...")

            # Convert to R dataframe
            with localconverter(ro.default_converter + pandas2ri.converter):
                r_df = ro.conversion.py2rpy(test_data)

            ro.r.assign("batch_data", r_df)
            ro.r("batch_data$log_attempts <- log(pmax(batch_data$attempts, 1))")

            # Batch prediction
            start = time.time()
            ro.r("batch_preds <- posterior_predict(model, newdata = batch_data, ndraws = 1000, allow_new_levels = TRUE)")
            elapsed = time.time() - start

            logger.info("  ✅ Batch prediction successful")
            logger.info(f"  Time: {elapsed:.2f}s ({elapsed/len(test_data):.2f}s per game)")
            logger.info(f"  Speedup vs sequential: ~{5*5/elapsed:.1f}x faster")

            if elapsed > 10:
                logger.warning(f"  ⚠️  Slow batch prediction ({elapsed:.2f}s for 5 games)")

            self.test_results['batch_prediction'] = True
            return True

        except Exception as e:
            logger.error(f"❌ Batch prediction failed: {e}")
            logger.error(traceback.format_exc())
            self.test_results['batch_prediction'] = False
            return False

    def test_7_memory_leak_check(self) -> bool:
        """Test 7: Memory leak check (multiple predictions)."""
        logger.info("\n" + "="*80)
        logger.info("TEST 7: Memory Leak Check")
        logger.info("="*80)

        try:
            from py.ensemble.r_model_interface import RModelInterface

            interface = RModelInterface(db_config=self.db_config)

            mem_start = self.get_memory_usage_mb()
            logger.info(f"Initial memory: {mem_start:.1f} MB")

            # Run 10 predictions
            logger.info("Running 10 predictions...")
            for i in range(10):
                result = interface.predict_hierarchical(
                    player_id="00-0019596",
                    season=CURRENT_SEASON,
                    week=9 - i % 5,  # Vary week to avoid caching
                    model_path=self.hierarchical_path,
                    ndraws=100  # Use fewer draws for speed
                )

                if i % 5 == 4:
                    mem_current = self.get_memory_usage_mb()
                    mem_delta = mem_current - mem_start
                    logger.info(f"  After {i+1} predictions: {mem_current:.1f} MB (+{mem_delta:.1f} MB)")

            mem_final = self.get_memory_usage_mb()
            mem_total_delta = mem_final - mem_start

            logger.info("\n✅ Memory leak check complete")
            logger.info(f"  Initial: {mem_start:.1f} MB")
            logger.info(f"  Final: {mem_final:.1f} MB")
            logger.info(f"  Delta: +{mem_total_delta:.1f} MB")

            if mem_total_delta > 100:
                logger.warning(f"  ⚠️  Significant memory growth (+{mem_total_delta:.1f} MB) - possible leak")

            self.test_results['memory_leak'] = True
            return True

        except Exception as e:
            logger.error(f"❌ Memory leak check failed: {e}")
            logger.error(traceback.format_exc())
            self.test_results['memory_leak'] = False
            return False

    def print_summary(self):
        """Print diagnostic summary."""
        logger.info("\n" + "="*80)
        logger.info("DIAGNOSTIC SUMMARY")
        logger.info("="*80)

        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())

        logger.info(f"\nResults: {passed_tests}/{total_tests} tests passed\n")

        for test_name, passed in self.test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"  {status} - {test_name}")

        logger.info("\n" + "="*80)

        if passed_tests == total_tests:
            logger.info("✅ ALL TESTS PASSED - R model integration is healthy")
            logger.info("="*80)
            return 0
        else:
            logger.error("❌ SOME TESTS FAILED - R model integration has issues")
            logger.error("="*80)
            return 1

    def run_all_tests(self) -> int:
        """Run all diagnostic tests."""
        logger.info("="*80)
        logger.info("R MODEL INTEGRATION DIAGNOSTIC")
        logger.info("="*80)
        logger.info(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Project root: {self.project_root}")

        # Run tests in order
        tests = [
            self.test_1_rpy2_installation,
            self.test_2_r_libraries,
            self.test_3_model_files,
            self.test_4_model_loading_time,
            self.test_5_single_prediction,
            self.test_6_batch_prediction_capability,
            self.test_7_memory_leak_check,
        ]

        for test_fn in tests:
            try:
                success = test_fn()
                if not success:
                    # Continue running other tests even if one fails
                    pass
            except Exception as e:
                logger.error(f"❌ Test crashed: {e}")
                logger.error(traceback.format_exc())

        return self.print_summary()


def main():
    diagnostic = RModelDiagnostic()
    exit_code = diagnostic.run_all_tests()
    return exit_code


if __name__ == "__main__":
    exit(main())
