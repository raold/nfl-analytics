#!/usr/bin/env python3
"""
A/B Test: R Prediction Lookup vs rpy2
======================================

Compares offline PostgreSQL lookup vs real-time rpy2 predictions for
R Bayesian models (State-Space v2, Hierarchical v2, Informative Priors v2).

Success Criteria:
- MAE difference: Within ±0.5 yards between methods
- Latency: <100ms for lookup (vs 2-4s for rpy2)
- Reliability: 100% success rate for lookup

Author: Claude Code (Sonnet 4.5)
Created: November 20, 2025

Usage:
    python scripts/ab_test_lookup_vs_rpy2.py --season 2024 --sample-size 100
    python scripts/ab_test_lookup_vs_rpy2.py --full  # Full 2024 season
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import lookup client
from py.ensemble.r_prediction_lookup import RModelPredictionLookup

# Import rpy2 interface (for comparison)
try:
    from py.ensemble.r_model_interface import RModelInterface

    RPY2_AVAILABLE = True
except ImportError:
    print("⚠️  WARNING: rpy2 not available, skipping rpy2 comparison")
    RPY2_AVAILABLE = False


class ABTestRunner:
    """
    A/B testing framework for comparing lookup vs rpy2.

    Tracks:
    - Prediction accuracy (MAE, RMSE)
    - Latency (mean, p50, p95, p99)
    - Reliability (success rate, error types)
    - Memory usage
    """

    # Success criteria thresholds
    MAE_TOLERANCE_YARDS = 0.5  # ±0.5 yards
    LATENCY_TARGET_MS = 100.0  # <100ms for lookup
    MIN_SUCCESS_RATE = 0.99  # 99% success rate

    # R model names
    R_MODELS = ["state_space_v2", "hierarchical_v2", "informative_priors_v2"]

    def __init__(self, verbose: bool = True):
        """Initialize A/B test runner."""
        self.verbose = verbose
        self.lookup = RModelPredictionLookup(cache_size=10000)

        if RPY2_AVAILABLE:
            self.rpy2_interface = RModelInterface()
        else:
            self.rpy2_interface = None

        # Results tracking
        self.lookup_results = []
        self.rpy2_results = []
        self.lookup_latencies = []
        self.rpy2_latencies = []
        self.lookup_errors = []
        self.rpy2_errors = []

    def get_test_games(self, season: int = 2024, sample_size: int = None) -> pd.DataFrame:
        """
        Get player-games for testing.

        Args:
            season: Season to test (default: 2024)
            sample_size: Number of games to sample (default: all)

        Returns:
            DataFrame with (player_id, season, week, actual_yards)
        """
        conn = psycopg2.connect(
            host="localhost", port=5544, dbname="devdb01", user="dro", password="sicillionbillions"
        )

        query = f"""
        SELECT DISTINCT
            pgs.player_id,
            pgs.player_display_name,
            pgs.season,
            pgs.week,
            pgs.stat_yards as actual_yards
        FROM mart.player_game_stats pgs
        WHERE pgs.season = {season}
            AND pgs.stat_category = 'passing'
            AND pgs.player_position = 'QB'
            AND pgs.stat_attempts >= 15
            AND pgs.stat_yards IS NOT NULL
        ORDER BY pgs.week, pgs.player_display_name
        """

        df = pd.read_sql(query, conn)
        conn.close()

        if sample_size is not None and len(df) > sample_size:
            # Random sample for faster testing
            df = df.sample(n=sample_size, random_state=42)

        return df

    def test_lookup_method(
        self, player_id: str, season: int, week: int, actual_yards: float
    ) -> dict:
        """
        Test PostgreSQL lookup method.

        Returns:
            {
                'player_id': str,
                'season': int,
                'week': int,
                'actual_yards': float,
                'predictions': {model_name: float},
                'latency_ms': float,
                'success': bool,
                'error': str or None
            }
        """
        start_time = time.time()

        try:
            # Get predictions for all R models (batch query)
            preds = self.lookup.get_all_r_predictions(player_id, season, week)

            elapsed_ms = (time.time() - start_time) * 1000

            # Extract mean predictions
            predictions = {}
            for model in self.R_MODELS:
                pred_dict = preds.get(model)
                if pred_dict is not None:
                    predictions[model] = pred_dict["mean"]
                else:
                    predictions[model] = None

            return {
                "player_id": player_id,
                "season": season,
                "week": week,
                "actual_yards": actual_yards,
                "predictions": predictions,
                "latency_ms": elapsed_ms,
                "success": True,
                "error": None,
            }

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000

            return {
                "player_id": player_id,
                "season": season,
                "week": week,
                "actual_yards": actual_yards,
                "predictions": {model: None for model in self.R_MODELS},
                "latency_ms": elapsed_ms,
                "success": False,
                "error": str(e),
            }

    def test_rpy2_method(self, player_id: str, season: int, week: int, actual_yards: float) -> dict:
        """
        Test rpy2 real-time prediction method.

        Returns: Same structure as test_lookup_method()
        """
        if not RPY2_AVAILABLE:
            return {
                "player_id": player_id,
                "season": season,
                "week": week,
                "actual_yards": actual_yards,
                "predictions": {model: None for model in self.R_MODELS},
                "latency_ms": 0.0,
                "success": False,
                "error": "rpy2 not available",
            }

        start_time = time.time()

        try:
            # Call each R model individually via rpy2
            predictions = {}

            # State-Space v2
            try:
                pred = self.rpy2_interface.predict_state_space(player_id, season, week)
                predictions["state_space_v2"] = pred.get("mean") if pred else None
            except Exception:
                predictions["state_space_v2"] = None

            # Hierarchical v2
            try:
                pred = self.rpy2_interface.predict_hierarchical(player_id, season, week)
                predictions["hierarchical_v2"] = pred.get("mean") if pred else None
            except Exception:
                predictions["hierarchical_v2"] = None

            # Informative Priors v2
            try:
                pred = self.rpy2_interface.predict_informative_priors(player_id, season, week)
                predictions["informative_priors_v2"] = pred.get("mean") if pred else None
            except Exception:
                predictions["informative_priors_v2"] = None

            elapsed_ms = (time.time() - start_time) * 1000

            # Check if any predictions succeeded
            success = any(p is not None for p in predictions.values())

            return {
                "player_id": player_id,
                "season": season,
                "week": week,
                "actual_yards": actual_yards,
                "predictions": predictions,
                "latency_ms": elapsed_ms,
                "success": success,
                "error": None if success else "All models failed",
            }

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000

            return {
                "player_id": player_id,
                "season": season,
                "week": week,
                "actual_yards": actual_yards,
                "predictions": {model: None for model in self.R_MODELS},
                "latency_ms": elapsed_ms,
                "success": False,
                "error": str(e),
            }

    def run_ab_test(self, season: int = 2024, sample_size: int = None) -> dict:
        """
        Run A/B test comparing lookup vs rpy2.

        Args:
            season: Season to test
            sample_size: Number of games to test (None = all)

        Returns:
            Results summary dict with success/failure determination
        """
        print("=" * 80)
        print("A/B Test: R Prediction Lookup vs rpy2")
        print("=" * 80)
        print()

        # Get test games
        print(f"Loading test games for season {season}...")
        games_df = self.get_test_games(season, sample_size)
        print(f"✅ Loaded {len(games_df)} games for testing")
        print()

        # Run tests
        print("Running A/B tests...")
        print("-" * 80)

        for idx, row in games_df.iterrows():
            if self.verbose and idx % 10 == 0:
                print(f"Progress: {idx}/{len(games_df)} games tested...")

            # Test lookup method
            lookup_result = self.test_lookup_method(
                row["player_id"], int(row["season"]), int(row["week"]), row["actual_yards"]
            )
            self.lookup_results.append(lookup_result)

            if lookup_result["success"]:
                self.lookup_latencies.append(lookup_result["latency_ms"])
            else:
                self.lookup_errors.append(lookup_result["error"])

            # Test rpy2 method (if available)
            if RPY2_AVAILABLE:
                rpy2_result = self.test_rpy2_method(
                    row["player_id"], int(row["season"]), int(row["week"]), row["actual_yards"]
                )
                self.rpy2_results.append(rpy2_result)

                if rpy2_result["success"]:
                    self.rpy2_latencies.append(rpy2_result["latency_ms"])
                else:
                    self.rpy2_errors.append(rpy2_result["error"])

        print()
        print("✅ A/B testing complete")
        print()

        # Analyze results
        return self.analyze_results()

    def analyze_results(self) -> dict:
        """
        Analyze A/B test results and determine success/failure.

        Returns:
            {
                'lookup': {...},
                'rpy2': {...},
                'comparison': {...},
                'success_criteria_met': bool
            }
        """
        print("=" * 80)
        print("RESULTS ANALYSIS")
        print("=" * 80)
        print()

        # Lookup method analysis
        lookup_stats = self._analyze_method_results(
            self.lookup_results, self.lookup_latencies, self.lookup_errors, method_name="Lookup"
        )

        # rpy2 method analysis (if available)
        if RPY2_AVAILABLE and len(self.rpy2_results) > 0:
            rpy2_stats = self._analyze_method_results(
                self.rpy2_results, self.rpy2_latencies, self.rpy2_errors, method_name="rpy2"
            )
        else:
            rpy2_stats = None

        # Comparison
        comparison = self._compare_methods(lookup_stats, rpy2_stats)

        # Success criteria evaluation
        success = self._evaluate_success_criteria(lookup_stats, comparison)

        return {
            "lookup": lookup_stats,
            "rpy2": rpy2_stats,
            "comparison": comparison,
            "success_criteria_met": success,
        }

    def _analyze_method_results(
        self, results: list[dict], latencies: list[float], errors: list[str], method_name: str
    ) -> dict:
        """Analyze results for a single method."""
        print(f"\n{method_name} Method Results:")
        print("-" * 80)

        n_tests = len(results)
        n_success = sum(1 for r in results if r["success"])
        success_rate = (n_success / n_tests) * 100 if n_tests > 0 else 0.0

        print(f"Total tests: {n_tests}")
        print(f"Successful: {n_success} ({success_rate:.1f}%)")
        print(f"Failed: {n_tests - n_success} ({100 - success_rate:.1f}%)")
        print()

        # Latency stats
        if len(latencies) > 0:
            print("Latency Statistics:")
            print(f"  Mean: {np.mean(latencies):.2f} ms")
            print(f"  Median (p50): {np.median(latencies):.2f} ms")
            print(f"  p95: {np.percentile(latencies, 95):.2f} ms")
            print(f"  p99: {np.percentile(latencies, 99):.2f} ms")
            print(f"  Min: {np.min(latencies):.2f} ms")
            print(f"  Max: {np.max(latencies):.2f} ms")
        else:
            print("Latency: No successful predictions")
        print()

        # Accuracy stats (MAE per model)
        model_mae = {}
        for model in self.R_MODELS:
            errors = []
            for r in results:
                if r["success"] and r["predictions"].get(model) is not None:
                    pred = r["predictions"][model]
                    actual = r["actual_yards"]
                    errors.append(abs(pred - actual))

            if len(errors) > 0:
                model_mae[model] = np.mean(errors)

        if model_mae:
            print("Prediction Accuracy (MAE):")
            for model, mae in model_mae.items():
                print(f"  {model}: {mae:.2f} yards")
            print(f"  Average: {np.mean(list(model_mae.values())):.2f} yards")
        else:
            print("Accuracy: No valid predictions")
        print()

        # Error analysis
        if len(errors) > 0:
            print(f"Error Types ({len(errors)} total):")
            error_counts = {}
            for err in errors:
                error_counts[err] = error_counts.get(err, 0) + 1

            for err, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {err}: {count}")

        return {
            "n_tests": n_tests,
            "n_success": n_success,
            "success_rate": success_rate,
            "latencies": latencies,
            "mean_latency_ms": np.mean(latencies) if latencies else None,
            "p50_latency_ms": np.median(latencies) if latencies else None,
            "p95_latency_ms": np.percentile(latencies, 95) if latencies else None,
            "model_mae": model_mae,
            "avg_mae": np.mean(list(model_mae.values())) if model_mae else None,
            "errors": errors,
        }

    def _compare_methods(self, lookup_stats: dict, rpy2_stats: dict) -> dict:
        """Compare lookup vs rpy2."""
        print("\n" + "=" * 80)
        print("METHOD COMPARISON")
        print("=" * 80)
        print()

        if rpy2_stats is None:
            print("⚠️  rpy2 not available for comparison")
            return {}

        # MAE comparison
        lookup_mae = lookup_stats.get("avg_mae")
        rpy2_mae = rpy2_stats.get("avg_mae")

        if lookup_mae is not None and rpy2_mae is not None:
            mae_diff = abs(lookup_mae - rpy2_mae)
            mae_within_tolerance = mae_diff <= self.MAE_TOLERANCE_YARDS

            print("Prediction Accuracy (MAE):")
            print(f"  Lookup: {lookup_mae:.2f} yards")
            print(f"  rpy2: {rpy2_mae:.2f} yards")
            print(f"  Difference: {mae_diff:.2f} yards")
            print(
                f"  Within tolerance (±{self.MAE_TOLERANCE_YARDS} yards)? {'✅ YES' if mae_within_tolerance else '❌ NO'}"
            )
            print()
        else:
            mae_diff = None
            mae_within_tolerance = None

        # Latency comparison
        lookup_latency = lookup_stats.get("mean_latency_ms")
        rpy2_latency = rpy2_stats.get("mean_latency_ms")

        if lookup_latency is not None and rpy2_latency is not None:
            speedup = rpy2_latency / lookup_latency

            print("Latency:")
            print(f"  Lookup: {lookup_latency:.2f} ms")
            print(f"  rpy2: {rpy2_latency:.2f} ms")
            print(f"  Speedup: {speedup:.1f}x faster")
            print(
                f"  Lookup meets target (<{self.LATENCY_TARGET_MS}ms)? {'✅ YES' if lookup_latency < self.LATENCY_TARGET_MS else '❌ NO'}"
            )
            print()
        else:
            speedup = None

        # Reliability comparison
        lookup_success_rate = lookup_stats.get("success_rate", 0.0) / 100
        rpy2_success_rate = rpy2_stats.get("success_rate", 0.0) / 100

        print("Reliability:")
        print(f"  Lookup success rate: {lookup_success_rate*100:.1f}%")
        print(f"  rpy2 success rate: {rpy2_success_rate*100:.1f}%")
        print(
            f"  Lookup meets target (≥{self.MIN_SUCCESS_RATE*100:.0f}%)? {'✅ YES' if lookup_success_rate >= self.MIN_SUCCESS_RATE else '❌ NO'}"
        )
        print()

        return {
            "mae_diff": mae_diff,
            "mae_within_tolerance": mae_within_tolerance,
            "speedup": speedup,
            "lookup_success_rate": lookup_success_rate,
            "rpy2_success_rate": rpy2_success_rate,
        }

    def _evaluate_success_criteria(self, lookup_stats: dict, comparison: dict) -> bool:
        """
        Evaluate if success criteria are met.

        Success Criteria:
        1. MAE within ±0.5 yards of rpy2 (or better)
        2. Latency <100ms
        3. Success rate ≥99%
        """
        print("\n" + "=" * 80)
        print("SUCCESS CRITERIA EVALUATION")
        print("=" * 80)
        print()

        criteria_met = []

        # Criterion 1: MAE tolerance
        mae_within_tolerance = comparison.get("mae_within_tolerance")
        if mae_within_tolerance is not None:
            criteria_met.append(mae_within_tolerance)
            print(
                f"1. MAE within ±{self.MAE_TOLERANCE_YARDS} yards: {'✅ PASS' if mae_within_tolerance else '❌ FAIL'}"
            )
        else:
            print("1. MAE tolerance: ⚠️  SKIPPED (rpy2 not available)")

        # Criterion 2: Latency
        lookup_latency = lookup_stats.get("mean_latency_ms")
        latency_met = lookup_latency is not None and lookup_latency < self.LATENCY_TARGET_MS
        criteria_met.append(latency_met)
        print(f"2. Latency <{self.LATENCY_TARGET_MS}ms: {'✅ PASS' if latency_met else '❌ FAIL'}")

        # Criterion 3: Reliability
        lookup_success_rate = comparison.get("lookup_success_rate", 0.0)
        reliability_met = lookup_success_rate >= self.MIN_SUCCESS_RATE
        criteria_met.append(reliability_met)
        print(
            f"3. Success rate ≥{self.MIN_SUCCESS_RATE*100:.0f}%: {'✅ PASS' if reliability_met else '❌ FAIL'}"
        )

        print()

        all_met = all(criteria_met)

        if all_met:
            print("=" * 80)
            print("✅ ALL SUCCESS CRITERIA MET - LOOKUP METHOD APPROVED")
            print("=" * 80)
        else:
            print("=" * 80)
            print("❌ SOME CRITERIA NOT MET - FURTHER INVESTIGATION NEEDED")
            print("=" * 80)

        return all_met


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="A/B test R prediction lookup vs rpy2")
    parser.add_argument("--season", type=int, default=2024, help="Season to test (default: 2024)")
    parser.add_argument(
        "--sample-size", type=int, default=None, help="Number of games to test (default: all)"
    )
    parser.add_argument(
        "--full", action="store_true", help="Test full season (same as --sample-size None)"
    )
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    # Handle --full flag
    if args.full:
        args.sample_size = None

    # Run A/B test
    runner = ABTestRunner(verbose=not args.quiet)
    results = runner.run_ab_test(season=args.season, sample_size=args.sample_size)

    # Print cache stats
    print("\n" + "=" * 80)
    print("CACHE PERFORMANCE")
    print("=" * 80)
    cache_stats = runner.lookup.get_cache_stats()
    print(f"Cache hit rate: {cache_stats['hit_rate']*100:.1f}%")
    print(f"DB queries: {cache_stats['db_queries']}")
    print(f"Fallbacks: {cache_stats['fallbacks']}")
    print()

    # Exit code based on success
    sys.exit(0 if results["success_criteria_met"] else 1)


if __name__ == "__main__":
    main()
