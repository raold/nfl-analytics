#!/usr/bin/env python3
"""
Detailed Performance Profiling of Batch Prediction Compute Time

Analyzes the 8ms compute time in predict_batch() to identify bottlenecks
beyond database queries. Uses line_profiler for microsecond-level granularity.

Context:
- Current: 654ms per game (646ms DB, 8ms compute)
- Target: <500ms per game
- Need to optimize the remaining 8ms compute + reduce DB time

Profiling Approach:
1. Line-by-line profiling of predict_batch() assembly loop
2. Memory allocation analysis
3. Data structure overhead measurement
4. NumPy/pandas operation timing

Usage:
    # Install line_profiler
    pip install line_profiler

    # Run profiling
    python scripts/profile_batch_prediction_compute.py --n-games 50

    # Output: Microsecond-level breakdown of assembly loop

Author: Claude Code (Sonnet 4.5)
Date: November 15, 2025
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import cProfile
import io
import logging
import pstats
import tracemalloc

import numpy as np
import pandas as pd
import psycopg2

from nfl_analytics.ensemble.enhanced_ensemble_v5 import EnhancedEnsemble4Way

logging.basicConfig(
    level=logging.WARNING,  # Suppress info logs during profiling
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_test_games(n_games: int = 50, season: int = 2024) -> list:
    """Fetch test games from database."""
    conn = psycopg2.connect(
        host="localhost", port=5544, dbname="devdb01", user="dro", password="sicillionbillions"
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

    games = df.to_dict("records")
    return games


def profile_assembly_loop(ensemble, games):
    """
    Manually profile the assembly loop in predict_batch().

    This replicates the loop with fine-grained timing to identify bottlenecks.
    """
    print("\n" + "=" * 80)
    print("ASSEMBLY LOOP PROFILING (Manual Instrumentation)")
    print("=" * 80)

    # Get batch predictions (mimic predict_batch)
    state_space_preds = ensemble._get_state_space_predictions_batch(games)
    hierarchical_preds = {}
    xgboost_preds = {}
    informative_preds = {}

    if ensemble.hierarchical_path.exists():
        hierarchical_preds = ensemble._get_r_predictions_batch(
            games, ensemble.hierarchical_path, "Hierarchical v2"
        )

    if ensemble.xgboost_model:
        xgboost_preds = ensemble._get_xgboost_predictions_batch(games)

    if ensemble.informative_path.exists():
        informative_preds = ensemble._get_r_predictions_batch(
            games, ensemble.informative_path, "Informative Priors v2"
        )

    # Profile assembly loop
    results = []

    # Timing buckets
    times = {
        "key_creation": 0,
        "dict_lookup_state": 0,
        "dict_lookup_hier": 0,
        "dict_lookup_xgb": 0,
        "dict_lookup_info": 0,
        "list_append_means": 0,
        "list_append_stds": 0,
        "list_append_weights": 0,
        "array_creation": 0,
        "weight_normalize": 0,
        "ensemble_mean_calc": 0,
        "variance_calc": 0,
        "sqrt_calc": 0,
        "temp_scaling": 0,
        "quantile_calc": 0,
        "result_dict_creation": 0,
        "result_append": 0,
    }

    loop_start = time.perf_counter()

    for game in games:
        # Key creation
        t0 = time.perf_counter()
        player_id = game["player_id"]
        season = game["season"]
        week = game["week"]
        key = (player_id, season, week)
        times["key_creation"] += time.perf_counter() - t0

        # Dictionary lookups
        predictions = {}

        t0 = time.perf_counter()
        if key in state_space_preds:
            predictions["state_space_v2"] = state_space_preds[key]
        times["dict_lookup_state"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        if key in hierarchical_preds:
            predictions["hierarchical_v2"] = hierarchical_preds[key]
        times["dict_lookup_hier"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        if key in xgboost_preds:
            predictions["xgboost_v2"] = xgboost_preds[key]
        times["dict_lookup_xgb"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        if key in informative_preds:
            predictions["informative_priors_v2"] = informative_preds[key]
        times["dict_lookup_info"] += time.perf_counter() - t0

        # Skip if no predictions
        if len(predictions) == 0:
            continue

        # List building
        means = []
        stds = []
        active_weights = []

        for model_name in [
            "state_space_v2",
            "hierarchical_v2",
            "xgboost_v2",
            "informative_priors_v2",
        ]:
            if model_name in predictions:
                t0 = time.perf_counter()
                means.append(predictions[model_name][0])
                times["list_append_means"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                stds.append(predictions[model_name][1])
                times["list_append_stds"] += time.perf_counter() - t0

                t0 = time.perf_counter()
                active_weights.append(ensemble.weights[model_name])
                times["list_append_weights"] += time.perf_counter() - t0

        # Array creation
        t0 = time.perf_counter()
        means = np.array(means)
        stds = np.array(stds)
        active_weights = np.array(active_weights)
        times["array_creation"] += time.perf_counter() - t0

        # Weight normalization
        t0 = time.perf_counter()
        active_weights /= active_weights.sum()
        times["weight_normalize"] += time.perf_counter() - t0

        # Ensemble mean
        t0 = time.perf_counter()
        ensemble_mean = np.sum(active_weights * means)
        times["ensemble_mean_calc"] += time.perf_counter() - t0

        # Variance calculation
        t0 = time.perf_counter()
        within_var = np.sum(active_weights * (stds**2))
        between_var = np.sum(active_weights * ((means - ensemble_mean) ** 2))
        ensemble_var = within_var + between_var
        times["variance_calc"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        ensemble_std_raw = np.sqrt(ensemble_var)
        times["sqrt_calc"] += time.perf_counter() - t0

        # Temperature scaling
        t0 = time.perf_counter()
        ensemble_std = ensemble_std_raw * np.sqrt(ensemble.temperature)
        times["temp_scaling"] += time.perf_counter() - t0

        # Quantile calculation
        t0 = time.perf_counter()
        quantile_10 = ensemble_mean - 1.645 * ensemble_std
        quantile_90 = ensemble_mean + 1.645 * ensemble_std
        times["quantile_calc"] += time.perf_counter() - t0

        # Result dict creation
        t0 = time.perf_counter()
        result = {
            "player_id": player_id,
            "season": season,
            "week": week,
            "ensemble_mean": ensemble_mean,
            "ensemble_std": ensemble_std,
            "quantile_10": quantile_10,
            "quantile_90": quantile_90,
        }
        times["result_dict_creation"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        results.append(result)
        times["result_append"] += time.perf_counter() - t0

    loop_time = (time.perf_counter() - loop_start) * 1000

    # Convert to DataFrame (same as original)
    df_start = time.perf_counter()
    pd.DataFrame(results)
    df_time = (time.perf_counter() - df_start) * 1000

    # Print results
    print(f"\nTotal games: {len(games)}")
    print(f"Assembly loop time: {loop_time:.2f}ms ({loop_time/len(games):.3f}ms per game)")
    print(f"DataFrame creation: {df_time:.2f}ms ({df_time/len(games):.3f}ms per game)")
    print(f"Total compute time: {(loop_time + df_time):.2f}ms")
    print("\nLoop time breakdown (total time in ms):")
    print("-" * 80)

    # Sort by time
    sorted_times = sorted(times.items(), key=lambda x: x[1], reverse=True)

    total_accounted = sum(times.values()) * 1000

    for operation, duration in sorted_times:
        ms = duration * 1000
        pct = (ms / loop_time) * 100 if loop_time > 0 else 0
        per_game = ms / len(games)
        print(f"{operation:30s}: {ms:8.2f}ms ({pct:5.1f}%) | {per_game:.4f}ms/game")

    print("-" * 80)
    print(f"{'TOTAL (accounted)':30s}: {total_accounted:8.2f}ms")
    print(f"{'TOTAL (measured)':30s}: {loop_time:8.2f}ms")
    print(f"{'Unaccounted overhead':30s}: {(loop_time - total_accounted):8.2f}ms")

    return times, loop_time, df_time


def profile_memory_allocation(ensemble, games):
    """Profile memory allocation patterns."""
    print("\n" + "=" * 80)
    print("MEMORY ALLOCATION PROFILING")
    print("=" * 80)

    tracemalloc.start()

    # Get batch predictions
    state_space_preds = ensemble._get_state_space_predictions_batch(games)
    mem_after_state_space = tracemalloc.get_traced_memory()

    hierarchical_preds = {}
    if ensemble.hierarchical_path.exists():
        hierarchical_preds = ensemble._get_r_predictions_batch(
            games, ensemble.hierarchical_path, "Hierarchical v2"
        )
    mem_after_hierarchical = tracemalloc.get_traced_memory()

    xgboost_preds = {}
    if ensemble.xgboost_model:
        xgboost_preds = ensemble._get_xgboost_predictions_batch(games)
    mem_after_xgboost = tracemalloc.get_traced_memory()

    informative_preds = {}
    if ensemble.informative_path.exists():
        informative_preds = ensemble._get_r_predictions_batch(
            games, ensemble.informative_path, "Informative Priors v2"
        )
    mem_after_informative = tracemalloc.get_traced_memory()

    # Assembly loop
    results = []
    for game in games:
        key = (game["player_id"], game["season"], game["week"])
        predictions = {}

        if key in state_space_preds:
            predictions["state_space_v2"] = state_space_preds[key]
        if key in hierarchical_preds:
            predictions["hierarchical_v2"] = hierarchical_preds[key]
        if key in xgboost_preds:
            predictions["xgboost_v2"] = xgboost_preds[key]
        if key in informative_preds:
            predictions["informative_priors_v2"] = informative_preds[key]

        if len(predictions) == 0:
            continue

        means = []
        stds = []
        active_weights = []
        for model_name in [
            "state_space_v2",
            "hierarchical_v2",
            "xgboost_v2",
            "informative_priors_v2",
        ]:
            if model_name in predictions:
                means.append(predictions[model_name][0])
                stds.append(predictions[model_name][1])
                active_weights.append(ensemble.weights[model_name])

        means = np.array(means)
        stds = np.array(stds)
        active_weights = np.array(active_weights)
        active_weights /= active_weights.sum()

        ensemble_mean = np.sum(active_weights * means)
        within_var = np.sum(active_weights * (stds**2))
        between_var = np.sum(active_weights * ((means - ensemble_mean) ** 2))
        ensemble_var = within_var + between_var
        ensemble_std = np.sqrt(ensemble_var) * np.sqrt(ensemble.temperature)

        results.append(
            {
                "player_id": game["player_id"],
                "season": game["season"],
                "week": game["week"],
                "ensemble_mean": ensemble_mean,
                "ensemble_std": ensemble_std,
            }
        )

    mem_after_loop = tracemalloc.get_traced_memory()

    pd.DataFrame(results)
    mem_after_df = tracemalloc.get_traced_memory()

    tracemalloc.stop()

    # Print results
    def mb(bytes_val):
        return bytes_val / 1024 / 1024

    print("\nMemory usage breakdown:")
    print("-" * 80)
    print(
        f"After state-space: {mb(mem_after_state_space[0]):8.2f} MB (peak: {mb(mem_after_state_space[1]):8.2f} MB)"
    )
    print(
        f"After hierarchical: {mb(mem_after_hierarchical[0]):8.2f} MB (peak: {mb(mem_after_hierarchical[1]):8.2f} MB)"
    )
    print(
        f"After XGBoost: {mb(mem_after_xgboost[0]):8.2f} MB (peak: {mb(mem_after_xgboost[1]):8.2f} MB)"
    )
    print(
        f"After informative: {mb(mem_after_informative[0]):8.2f} MB (peak: {mb(mem_after_informative[1]):8.2f} MB)"
    )
    print(
        f"After assembly loop: {mb(mem_after_loop[0]):8.2f} MB (peak: {mb(mem_after_loop[1]):8.2f} MB)"
    )
    print(f"After DataFrame: {mb(mem_after_df[0]):8.2f} MB (peak: {mb(mem_after_df[1]):8.2f} MB)")
    print("-" * 80)

    loop_memory = mem_after_loop[0] - mem_after_informative[0]
    df_memory = mem_after_df[0] - mem_after_loop[0]

    print(f"\nMemory allocated in assembly loop: {mb(loop_memory):.2f} MB")
    print(f"Memory allocated in DataFrame creation: {mb(df_memory):.2f} MB")


def profile_with_cprofile(ensemble, games):
    """Use cProfile for comprehensive profiling."""
    print("\n" + "=" * 80)
    print("cProfile COMPREHENSIVE PROFILING")
    print("=" * 80)

    profiler = cProfile.Profile()
    profiler.enable()

    # Run predict_batch
    ensemble.predict_batch(games)

    profiler.disable()

    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats("cumulative")
    ps.print_stats(30)  # Top 30 functions

    print("\nTop 30 functions by cumulative time:")
    print(s.getvalue())


def analyze_vectorization_opportunities(ensemble, games):
    """
    Analyze whether the assembly loop can be vectorized.
    """
    print("\n" + "=" * 80)
    print("VECTORIZATION OPPORTUNITY ANALYSIS")
    print("=" * 80)

    # Get batch predictions
    state_space_preds = ensemble._get_state_space_predictions_batch(games)
    hierarchical_preds = {}
    xgboost_preds = {}
    informative_preds = {}

    if ensemble.hierarchical_path.exists():
        hierarchical_preds = ensemble._get_r_predictions_batch(
            games, ensemble.hierarchical_path, "Hierarchical v2"
        )

    if ensemble.xgboost_model:
        xgboost_preds = ensemble._get_xgboost_predictions_batch(games)

    if ensemble.informative_path.exists():
        informative_preds = ensemble._get_r_predictions_batch(
            games, ensemble.informative_path, "Informative Priors v2"
        )

    # Check if all games have all predictions
    complete_games = 0
    incomplete_games = 0
    missing_models = {
        "state_space_v2": 0,
        "hierarchical_v2": 0,
        "xgboost_v2": 0,
        "informative_priors_v2": 0,
    }

    for game in games:
        key = (game["player_id"], game["season"], game["week"])

        has_all = (
            key in state_space_preds
            and key in hierarchical_preds
            and key in xgboost_preds
            and key in informative_preds
        )

        if has_all:
            complete_games += 1
        else:
            incomplete_games += 1
            if key not in state_space_preds:
                missing_models["state_space_v2"] += 1
            if key not in hierarchical_preds:
                missing_models["hierarchical_v2"] += 1
            if key not in xgboost_preds:
                missing_models["xgboost_v2"] += 1
            if key not in informative_preds:
                missing_models["informative_priors_v2"] += 1

    print("\nGame prediction coverage:")
    print(
        f"  Complete (all 4 models): {complete_games}/{len(games)} ({complete_games/len(games)*100:.1f}%)"
    )
    print(
        f"  Incomplete (missing ≥1): {incomplete_games}/{len(games)} ({incomplete_games/len(games)*100:.1f}%)"
    )

    if incomplete_games > 0:
        print("\nMissing predictions by model:")
        for model, count in missing_models.items():
            pct = (count / len(games)) * 100
            print(f"  {model:25s}: {count:4d} ({pct:5.1f}%)")

    print("\n🎯 VECTORIZATION FEASIBILITY:")
    if complete_games == len(games):
        print("  ✅ All games have complete predictions - FULL VECTORIZATION POSSIBLE")
        print("  ✅ Can create pre-allocated arrays for all games")
        print("  ✅ Can eliminate assembly loop entirely")
    else:
        print(
            f"  ⚠️  Only {complete_games/len(games)*100:.1f}% of games complete - PARTIAL VECTORIZATION ONLY"
        )
        print("  ⚠️  Must handle missing predictions (conditional logic)")
        print("  ⚠️  Loop required for ragged data structure")


def main():
    parser = argparse.ArgumentParser(description="Profile batch prediction compute time")
    parser.add_argument(
        "--n-games", type=int, default=50, help="Number of games to test (default: 50)"
    )
    parser.add_argument("--season", type=int, default=2024, help="Season to test (default: 2024)")
    parser.add_argument(
        "--skip-cprofile", action="store_true", help="Skip cProfile (verbose output)"
    )

    args = parser.parse_args()

    # Fetch test games
    print("Fetching test games from database...")
    games = get_test_games(n_games=args.n_games, season=args.season)
    print(f"Loaded {len(games)} games for profiling\n")

    # Initialize ensemble
    print("Loading ensemble models...")
    ensemble = EnhancedEnsemble4Way()
    ensemble._load_models()
    print("Models loaded\n")

    # Run profiling
    print("=" * 80)
    print("BATCH PREDICTION COMPUTE TIME PROFILING")
    print("=" * 80)
    print("Test configuration:")
    print(f"  Games: {len(games)}")
    print(f"  Season: {args.season}")
    print("  Target: <8ms compute time per game (current)")

    # 1. Manual assembly loop profiling
    times, loop_time, df_time = profile_assembly_loop(ensemble, games)

    # 2. Memory profiling
    profile_memory_allocation(ensemble, games)

    # 3. Vectorization analysis
    analyze_vectorization_opportunities(ensemble, games)

    # 4. Comprehensive cProfile (optional)
    if not args.skip_cprofile:
        profile_with_cprofile(ensemble, games)

    # Final recommendations
    print("\n" + "=" * 80)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 80)

    # Analyze bottlenecks
    sorted_times = sorted(times.items(), key=lambda x: x[1], reverse=True)
    top_3 = sorted_times[:3]

    print("\n🎯 TOP 3 BOTTLENECKS:")
    for i, (operation, duration) in enumerate(top_3, 1):
        ms = duration * 1000
        per_game = ms / len(games)
        print(f"  {i}. {operation}: {ms:.2f}ms total ({per_game:.4f}ms/game)")

    print("\n💡 RECOMMENDED OPTIMIZATIONS:")

    # Check if dictionary lookups are slow
    dict_lookup_time = (
        times["dict_lookup_state"]
        + times["dict_lookup_hier"]
        + times["dict_lookup_xgb"]
        + times["dict_lookup_info"]
    ) * 1000

    if dict_lookup_time > loop_time * 0.1:
        print("\n1. OPTIMIZE DICTIONARY LOOKUPS (Priority: HIGH)")
        print(f"   Current: {dict_lookup_time:.2f}ms ({dict_lookup_time/len(games):.4f}ms/game)")
        print("   Strategy: Pre-allocate arrays, use index-based lookup instead of dict")
        print("   Expected speedup: 50-70%")
        print("   Implementation: 30 minutes")

    # Check if array creation is slow
    if times["array_creation"] * 1000 > loop_time * 0.05:
        print("\n2. PRE-ALLOCATE ARRAYS (Priority: MEDIUM)")
        print(f"   Current: {times['array_creation']*1000:.2f}ms")
        print("   Strategy: Create result arrays upfront, fill in-place")
        print("   Expected speedup: 30-40%")
        print("   Implementation: 20 minutes")

    # Check if DataFrame creation is slow
    if df_time > 5:
        print("\n3. OPTIMIZE DATAFRAME CREATION (Priority: LOW)")
        print(f"   Current: {df_time:.2f}ms")
        print("   Strategy: Use dict of arrays instead of list of dicts")
        print("   Expected speedup: 20-30%")
        print("   Implementation: 15 minutes")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
