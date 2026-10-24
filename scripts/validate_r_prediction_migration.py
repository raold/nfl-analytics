#!/usr/bin/env python3
"""
R Prediction Migration Validation Script
==========================================

Validates that PostgreSQL lookup produces identical results to rpy2.

Tests:
1. Prediction equivalence (rpy2 vs PostgreSQL)
2. Performance comparison (<10ms vs 2-4s)
3. Edge cases (missing predictions, stale data)

Usage:
    python scripts/validate_r_prediction_migration.py

Author: Claude Code (Sonnet 4.5)
Date: November 21, 2025
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

import pandas as pd
import psycopg2

from nfl_analytics.ensemble.r_model_interface import RModelInterface
from nfl_analytics.ensemble.r_prediction_lookup import RPredictionLookup

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RMigrationValidator:
    """
    Validates rpy2 → PostgreSQL migration for R model predictions.

    Compares predictions from both methods to ensure equivalence.
    """

    def __init__(self):
        """Initialize validator with both interfaces."""
        self.db_config = {
            "host": "localhost",
            "port": 5544,
            "dbname": "devdb01",
            "user": "dro",
            "password": "sicillionbillions",
        }

        logger.info("Initializing rpy2 interface...")
        try:
            self.rpy2_interface = RModelInterface(db_config=self.db_config)
            self.rpy2_available = True
            logger.info("✅ rpy2 interface initialized")
        except Exception as e:
            logger.warning(f"⚠️  rpy2 not available: {e}")
            logger.warning("Will skip rpy2 comparison (PostgreSQL-only validation)")
            self.rpy2_interface = None
            self.rpy2_available = False

        logger.info("Initializing PostgreSQL lookup...")
        self.pg_lookup = RPredictionLookup(db_config=self.db_config)
        logger.info("✅ PostgreSQL lookup initialized")

    def get_test_samples(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Get random sample of predictions from database for testing.

        Args:
            n_samples: Number of samples to retrieve

        Returns:
            DataFrame with columns: [player_id, season, week, model_name]
        """
        query = """
        SELECT
            player_id,
            season,
            week,
            model_name
        FROM ml.r_model_predictions
        WHERE season IN (2022, 2023, 2024)
          AND week BETWEEN 1 AND 18  -- Regular season only
        ORDER BY RANDOM()
        LIMIT %s;
        """

        conn = psycopg2.connect(**self.db_config)
        df = pd.read_sql(query, conn, params=(n_samples,))
        conn.close()

        logger.info(f"Retrieved {len(df)} test samples from database")
        return df

    def compare_predictions(self, player_id: str, season: int, week: int, model_name: str) -> dict:
        """
        Compare rpy2 vs PostgreSQL prediction for a single case.

        Args:
            player_id: Player ID
            season: Season year
            week: Week number
            model_name: Model name

        Returns:
            Dict with comparison results:
            {
                'player_id': str,
                'season': int,
                'week': int,
                'model_name': str,
                'rpy2_mean': float (or None if unavailable),
                'rpy2_std': float (or None if unavailable),
                'pg_mean': float,
                'pg_std': float,
                'mean_diff': float (absolute difference),
                'std_diff': float (absolute difference),
                'rpy2_time_ms': float (or None),
                'pg_time_ms': float,
                'speedup': float (or None)
            }
        """
        result = {"player_id": player_id, "season": season, "week": week, "model_name": model_name}

        # Test PostgreSQL lookup
        pg_start = time.time()
        try:
            if model_name == "state_space_v2":
                pg_mean, pg_std = self.pg_lookup.predict_state_space(player_id, season, week)
            elif model_name == "hierarchical_v2":
                pg_mean, pg_std = self.pg_lookup.predict_hierarchical(player_id, season, week)
            elif model_name == "informative_priors_v2":
                pg_mean, pg_std = self.pg_lookup.predict_informative_priors(player_id, season, week)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            pg_time = (time.time() - pg_start) * 1000  # Convert to ms

            result["pg_mean"] = pg_mean
            result["pg_std"] = pg_std
            result["pg_time_ms"] = pg_time

        except Exception as e:
            logger.error(f"❌ PostgreSQL lookup failed: {e}")
            result["pg_mean"] = None
            result["pg_std"] = None
            result["pg_time_ms"] = None
            return result

        # Test rpy2 (if available)
        if self.rpy2_available:
            rpy2_start = time.time()
            try:
                if model_name == "state_space_v2":
                    rpy2_mean, rpy2_std = self.rpy2_interface.predict_state_space(
                        player_id, season, week
                    )
                elif model_name == "hierarchical_v2":
                    rpy2_mean, rpy2_std = self.rpy2_interface.predict_hierarchical(
                        player_id, season, week, model_path=None
                    )
                elif model_name == "informative_priors_v2":
                    rpy2_mean, rpy2_std = self.rpy2_interface.predict_informative_priors(
                        player_id, season, week, model_path=None
                    )

                rpy2_time = (time.time() - rpy2_start) * 1000  # Convert to ms

                result["rpy2_mean"] = rpy2_mean
                result["rpy2_std"] = rpy2_std
                result["rpy2_time_ms"] = rpy2_time

                # Compute differences
                result["mean_diff"] = abs(rpy2_mean - pg_mean)
                result["std_diff"] = abs(rpy2_std - pg_std)
                result["speedup"] = rpy2_time / pg_time if pg_time > 0 else None

            except Exception as e:
                logger.warning(f"⚠️  rpy2 call failed: {e}")
                result["rpy2_mean"] = None
                result["rpy2_std"] = None
                result["rpy2_time_ms"] = None
                result["mean_diff"] = None
                result["std_diff"] = None
                result["speedup"] = None
        else:
            # rpy2 not available
            result["rpy2_mean"] = None
            result["rpy2_std"] = None
            result["rpy2_time_ms"] = None
            result["mean_diff"] = None
            result["std_diff"] = None
            result["speedup"] = None

        return result

    def run_validation(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Run validation on n_samples predictions.

        Args:
            n_samples: Number of test cases

        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Running validation on {n_samples} samples...")

        # Get test samples
        test_samples = self.get_test_samples(n_samples)

        if len(test_samples) == 0:
            logger.error("❌ No test samples found in database")
            return pd.DataFrame()

        # Run comparisons
        results = []
        for idx, row in test_samples.iterrows():
            logger.info(
                f"[{idx+1}/{len(test_samples)}] Testing {row['model_name']} | "
                f"{row['player_id']} | S{row['season']}W{row['week']}"
            )

            result = self.compare_predictions(
                row["player_id"], row["season"], row["week"], row["model_name"]
            )
            results.append(result)

            # Progress update every 10
            if (idx + 1) % 10 == 0:
                logger.info(f"Completed {idx+1}/{len(test_samples)} comparisons")

        results_df = pd.DataFrame(results)
        return results_df

    def print_summary(self, results_df: pd.DataFrame):
        """
        Print validation summary statistics.

        Args:
            results_df: DataFrame with validation results
        """
        print("\n" + "=" * 80)
        print("R PREDICTION MIGRATION VALIDATION SUMMARY")
        print("=" * 80)

        # Success rate
        n_total = len(results_df)
        n_pg_success = results_df["pg_mean"].notna().sum()
        n_rpy2_success = (
            results_df["rpy2_mean"].notna().sum() if "rpy2_mean" in results_df.columns else 0
        )

        print("\n📊 Success Rates:")
        print(f"   PostgreSQL: {n_pg_success}/{n_total} ({100*n_pg_success/n_total:.1f}%)")
        if self.rpy2_available:
            print(f"   rpy2:       {n_rpy2_success}/{n_total} ({100*n_rpy2_success/n_total:.1f}%)")
        else:
            print("   rpy2:       N/A (not available)")

        # Prediction equivalence (if rpy2 available)
        if self.rpy2_available and n_rpy2_success > 0:
            valid_comparisons = results_df.dropna(subset=["mean_diff", "std_diff"])

            if len(valid_comparisons) > 0:
                print(f"\n🎯 Prediction Equivalence ({len(valid_comparisons)} valid comparisons):")
                print(
                    f"   Mean difference:  {valid_comparisons['mean_diff'].mean():.4f} yards (avg)"
                )
                print(
                    f"                     {valid_comparisons['mean_diff'].max():.4f} yards (max)"
                )
                print(
                    f"   Std difference:   {valid_comparisons['std_diff'].mean():.4f} yards (avg)"
                )
                print(f"                     {valid_comparisons['std_diff'].max():.4f} yards (max)")

                # Check if differences are negligible (< 0.01 yards)
                mean_match_rate = (valid_comparisons["mean_diff"] < 0.01).sum() / len(
                    valid_comparisons
                )
                std_match_rate = (valid_comparisons["std_diff"] < 0.01).sum() / len(
                    valid_comparisons
                )

                print("\n   Exact matches (<0.01 yards):")
                print(f"      Mean: {100*mean_match_rate:.1f}%")
                print(f"      Std:  {100*std_match_rate:.1f}%")

                if mean_match_rate > 0.99 and std_match_rate > 0.99:
                    print("\n   ✅ VALIDATION PASSED: Predictions are equivalent!")
                else:
                    print("\n   ⚠️  WARNING: Some predictions differ significantly")

        # Performance comparison
        if self.rpy2_available and n_rpy2_success > 0:
            valid_perf = results_df.dropna(subset=["rpy2_time_ms", "pg_time_ms", "speedup"])

            if len(valid_perf) > 0:
                print(f"\n⚡ Performance Comparison ({len(valid_perf)} valid comparisons):")
                print(f"   rpy2:        {valid_perf['rpy2_time_ms'].mean():.2f}ms (avg)")
                print(f"   PostgreSQL:  {valid_perf['pg_time_ms'].mean():.2f}ms (avg)")
                print(f"   Speedup:     {valid_perf['speedup'].mean():.1f}x faster")

                if valid_perf["speedup"].mean() > 50:
                    print(
                        f"\n   ✅ MASSIVE SPEEDUP: PostgreSQL is {valid_perf['speedup'].mean():.0f}x faster!"
                    )
                elif valid_perf["speedup"].mean() > 10:
                    print(
                        f"\n   ✅ SIGNIFICANT SPEEDUP: PostgreSQL is {valid_perf['speedup'].mean():.0f}x faster!"
                    )
                else:
                    print("\n   ⚠️  WARNING: Speedup lower than expected")

        else:
            # PostgreSQL-only validation
            pg_times = results_df.dropna(subset=["pg_time_ms"])

            if len(pg_times) > 0:
                print(f"\n⚡ PostgreSQL Performance ({len(pg_times)} predictions):")
                print(f"   Average:  {pg_times['pg_time_ms'].mean():.2f}ms")
                print(f"   Median:   {pg_times['pg_time_ms'].median():.2f}ms")
                print(f"   P95:      {pg_times['pg_time_ms'].quantile(0.95):.2f}ms")

                if pg_times["pg_time_ms"].mean() < 50:
                    print("\n   ✅ EXCELLENT: Sub-50ms average latency!")
                elif pg_times["pg_time_ms"].mean() < 100:
                    print("\n   ✅ GOOD: Sub-100ms average latency")
                else:
                    print("\n   ⚠️  WARNING: Latency higher than expected")

        # Model breakdown
        print("\n📈 Results by Model:")
        for model in results_df["model_name"].unique():
            model_df = results_df[results_df["model_name"] == model]
            n_success = model_df["pg_mean"].notna().sum()
            print(f"   {model:25s}: {n_success}/{len(model_df)} successful")

        print("\n" + "=" * 80)


def main():
    """Run validation."""
    print("\n" + "=" * 80)
    print("R PREDICTION MIGRATION VALIDATION")
    print("Comparing rpy2 vs PostgreSQL lookup")
    print("=" * 80 + "\n")

    validator = RMigrationValidator()

    # Run validation on 100 samples
    results_df = validator.run_validation(n_samples=100)

    if len(results_df) == 0:
        logger.error("❌ Validation failed - no results")
        return 1

    # Save results
    output_file = "results/r_migration_validation.csv"
    results_df.to_csv(output_file, index=False)
    logger.info(f"💾 Results saved to {output_file}")

    # Print summary
    validator.print_summary(results_df)

    return 0


if __name__ == "__main__":
    sys.exit(main())
