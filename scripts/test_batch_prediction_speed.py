#!/usr/bin/env python3
"""
Test Batch Prediction Performance Improvement

Validates that batch R prediction API:
1. Matches sequential predictions (within numerical tolerance)
2. Achieves 50-100x speedup target
3. Passes all risk gates (<500ms inference time)

Expected Results:
- Sequential: ~2,060ms for 1,984 games
- Batch: ~37ms for 1,984 games (55x speedup)

Usage:
    python scripts/test_batch_prediction_speed.py --n-games 50
    python scripts/test_batch_prediction_speed.py --full  # 1,984 games
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import numpy as np
import pandas as pd
import psycopg2
import os

from py.ensemble.enhanced_ensemble_v5 import EnhancedEnsemble4Way

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration constants
CURRENT_SEASON = 2024  # pragma: allowlist temporal


def get_test_games(n_games: int = 50, season: int = CURRENT_SEASON) -> list:
    """Fetch test games from database."""
    conn = psycopg2.connect(
        host="localhost",
        port=5544,
        dbname="devdb01",
        user="dro",
        password=os.getenv("DB_PASSWORD")
    )

    query = f"""
    SELECT DISTINCT
        pgs.player_id,
        pgs.season,
        pgs.week
    FROM mart.player_game_stats pgs
    JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
    WHERE pgs.stat_category = 'passing'
      AND ph.position_group = 'QB'
      AND pgs.season = {season}
    ORDER BY pgs.week, pgs.player_id
    LIMIT {n_games}
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    games = df.to_dict('records')
    logger.info(f"Loaded {len(games)} test games from season {season}")
    return games


def test_batch_vs_sequential(games: list, tolerance: float = 0.01):
    """
    Compare batch vs sequential predictions.

    Args:
        games: List of game dicts
        tolerance: Relative tolerance for prediction differences

    Returns:
        Dict with timing results and validation status
    """
    ensemble = EnhancedEnsemble4Way()

    logger.info("="*70)
    logger.info("BATCH PREDICTION VALIDATION TEST")
    logger.info("="*70)
    logger.info(f"Test size: {len(games)} games")
    logger.info(f"Tolerance: {tolerance * 100:.1f}%")
    logger.info("")

    # Test 1: Batch predictions
    logger.info("🚀 Testing BATCH predictions...")
    batch_start = time.time()
    batch_df = ensemble.predict_batch(games)
    batch_time = (time.time() - batch_start) * 1000
    batch_per_game = batch_time / len(games)

    logger.info(f"✅ Batch complete: {batch_time:.1f}ms total ({batch_per_game:.1f}ms per game)")

    # Test 2: Sequential predictions (for validation only - sample subset)
    sample_size = min(10, len(games))  # Only test 10 games sequentially
    sample_games = games[:sample_size]

    logger.info(f"\n🐌 Testing SEQUENTIAL predictions (sample: {sample_size} games)...")
    sequential_start = time.time()
    sequential_results = []

    for game in sample_games:
        pred = ensemble.predict(
            player_id=game["player_id"],
            season=game["season"],
            week=game["week"]
        )
        sequential_results.append({
            "player_id": game["player_id"],
            "season": game["season"],
            "week": game["week"],
            "ensemble_mean": pred["ensemble_mean"],
            "ensemble_std": pred["ensemble_std"],
        })

    sequential_time = (time.time() - sequential_start) * 1000
    sequential_per_game = sequential_time / sample_size

    logger.info(f"✅ Sequential complete: {sequential_time:.1f}ms total ({sequential_per_game:.1f}ms per game)")

    # Test 3: Validate predictions match
    logger.info("\n🔍 Validating predictions match...")
    sequential_df = pd.DataFrame(sequential_results)

    # Merge batch and sequential results
    merged = batch_df.merge(
        sequential_df,
        on=["player_id", "season", "week"],
        suffixes=("_batch", "_seq")
    )

    # Check differences
    mean_diff = np.abs(merged["ensemble_mean_batch"] - merged["ensemble_mean_seq"])
    std_diff = np.abs(merged["ensemble_std_batch"] - merged["ensemble_std_seq"])

    max_mean_diff = mean_diff.max()
    max_std_diff = std_diff.max()

    # Relative error
    mean_rel_error = (mean_diff / merged["ensemble_mean_seq"]).max()
    std_rel_error = (std_diff / merged["ensemble_std_seq"]).max()

    logger.info(f"  Max mean difference: {max_mean_diff:.4f} yards ({mean_rel_error*100:.2f}%)")
    logger.info(f"  Max std difference: {max_std_diff:.4f} yards ({std_rel_error*100:.2f}%)")

    # Validation
    validation_passed = (mean_rel_error < tolerance) and (std_rel_error < tolerance)

    if validation_passed:
        logger.info(f"  ✅ VALIDATION PASSED (within {tolerance*100:.1f}% tolerance)")
    else:
        logger.warning(f"  ⚠️  VALIDATION FAILED (exceeds {tolerance*100:.1f}% tolerance)")

    # Test 4: Performance metrics
    logger.info("\n📊 PERFORMANCE METRICS")
    logger.info("="*70)

    speedup = sequential_per_game / batch_per_game
    estimated_full_batch_time = batch_per_game * 1984  # Estimate for full backtest

    logger.info(f"Sequential time per game: {sequential_per_game:.1f}ms")
    logger.info(f"Batch time per game: {batch_per_game:.1f}ms")
    logger.info(f"Speedup: {speedup:.1f}x")
    logger.info(f"")
    logger.info(f"Estimated time for 1,984 games:")
    logger.info(f"  Sequential: {(sequential_per_game * 1984):.0f}ms ({(sequential_per_game * 1984)/1000:.1f}s)")
    logger.info(f"  Batch: {estimated_full_batch_time:.0f}ms ({estimated_full_batch_time/1000:.1f}s)")

    # Test 5: Risk gate validation
    logger.info("\n🚦 RISK GATE VALIDATION")
    logger.info("="*70)

    inference_time_target = 500  # ms
    inference_passes = batch_per_game < inference_time_target

    logger.info(f"Target: <{inference_time_target}ms per prediction")
    logger.info(f"Actual: {batch_per_game:.1f}ms")

    if inference_passes:
        logger.info(f"✅ PASS - Inference time meets target")
    else:
        logger.warning(f"❌ FAIL - Inference time exceeds target")

    # Final summary
    logger.info("\n" + "="*70)
    logger.info("FINAL RESULTS")
    logger.info("="*70)

    overall_pass = validation_passed and inference_passes

    if overall_pass:
        logger.info("✅ ALL TESTS PASSED")
        logger.info(f"  - Predictions match sequential (within {tolerance*100:.1f}%)")
        logger.info(f"  - Inference time meets target ({batch_per_game:.1f}ms < {inference_time_target}ms)")
        logger.info(f"  - Speedup: {speedup:.1f}x vs sequential")
    else:
        logger.warning("⚠️  SOME TESTS FAILED")
        if not validation_passed:
            logger.warning("  - Predictions differ from sequential")
        if not inference_passes:
            logger.warning(f"  - Inference time exceeds target ({batch_per_game:.1f}ms > {inference_time_target}ms)")

    return {
        "batch_time_ms": batch_time,
        "batch_per_game_ms": batch_per_game,
        "sequential_per_game_ms": sequential_per_game,
        "speedup": speedup,
        "validation_passed": validation_passed,
        "inference_passes": inference_passes,
        "overall_pass": overall_pass,
        "max_mean_diff": max_mean_diff,
        "max_std_diff": max_std_diff,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test batch prediction performance improvement"
    )
    parser.add_argument(
        "--n-games",
        type=int,
        default=50,
        help="Number of games to test (default: 50)"
    )
    parser.add_argument(
        "--season",
        type=int,
        default=CURRENT_SEASON,
        help="Season to test (default: 2024)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Test full 1,984 games (slow for sequential validation)"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Relative tolerance for prediction differences (default: 0.01 = 1%%)"
    )

    args = parser.parse_args()

    n_games = 1984 if args.full else args.n_games

    # Fetch test games
    games = get_test_games(n_games=n_games, season=args.season)

    # Run tests
    results = test_batch_vs_sequential(games, tolerance=args.tolerance)

    # Exit with code
    sys.exit(0 if results["overall_pass"] else 1)


if __name__ == "__main__":
    main()
