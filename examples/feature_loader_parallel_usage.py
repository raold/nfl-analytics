#!/usr/bin/env python3
"""
Ultra-Fast Parallel Feature Loader - Usage Examples
====================================================

This demonstrates how to use ParallelFeatureLoader for maximum performance
when extracting features for XGBoost training, backtesting, or API endpoints.

**Performance Targets:**
- Sequential: ~1 game/sec
- Batch mode (no cache): ~10-20 games/sec
- Batch mode (with cache): ~50-200 games/sec

**Author:** Claude Code (Sonnet 4.5)
**Date:** November 27, 2025
"""

import time

import pandas as pd

from nfl_analytics.features.feature_loader_parallel import ParallelFeatureLoader


def example_1_single_season():
    """
    Example 1: Extract features for an entire season (batch mode).

    Use case: Training XGBoost on historical data (2020-2024).
    """
    print("\n" + "=" * 80)
    print("Example 1: Extract Full Season (Batch Mode)")
    print("=" * 80)

    loader = ParallelFeatureLoader(use_redis=True, n_workers=8)

    # Extract all games from 2023 season (weeks 1-18)
    start = time.time()
    features_df = loader.extract_season_features(
        season=2023, weeks=list(range(1, 19))  # Weeks 1-18
    )
    elapsed = time.time() - start

    print(f"\n✓ Extracted {len(features_df)} games in {elapsed:.2f}s")
    print(f"  Throughput: {len(features_df)/elapsed:.1f} games/sec")
    print(f"  Features: {len(features_df.columns)} columns")
    print(f"  Memory: {features_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Display sample features
    print("\n  Sample game:")
    sample = features_df.iloc[0]
    print(
        f"    Game: {sample['home_team']} vs {sample['away_team']} ({sample['season']} Week {sample['week']})"
    )
    print(f"    Spread: {sample['spread_close']:.1f}")
    print(f"    Home EPA: {sample.get('home_off_epa_per_play', 0):.3f}")
    print(f"    Away EPA: {sample.get('away_off_epa_per_play', 0):.3f}")

    return features_df


def example_2_batch_extraction():
    """
    Example 2: Extract features for specific games (API use case).

    Use case: Dashboard fetching predictions for Week 12 games.
    """
    print("\n" + "=" * 80)
    print("Example 2: Batch Game Extraction (API Endpoint)")
    print("=" * 80)

    loader = ParallelFeatureLoader(use_redis=True, n_workers=8)

    # Create batch request for Week 12 games
    games_df = pd.DataFrame(
        {
            "season": [2024] * 5,
            "week": [12] * 5,
            "home_team": ["KC", "BUF", "SF", "PHI", "BAL"],
            "away_team": ["LAC", "MIA", "GB", "LAR", "PIT"],
        }
    )

    # Extract features with caching
    start = time.time()
    features_df = loader.extract_game_features_batch(games_df, use_cache=True)
    elapsed = time.time() - start

    print(f"\n✓ Extracted {len(features_df)} games in {elapsed:.2f}s")
    print(f"  Throughput: {len(features_df)/elapsed:.1f} games/sec")

    # Run again to test cache performance
    print("\n  Re-running same batch (should hit cache)...")
    start = time.time()
    cached_df = loader.extract_game_features_batch(games_df, use_cache=True)
    elapsed_cached = time.time() - start

    print(f"✓ Extracted {len(cached_df)} games in {elapsed_cached:.2f}s")
    print(f"  Throughput: {len(cached_df)/elapsed_cached:.1f} games/sec")
    print(f"  Speedup: {elapsed/elapsed_cached:.1f}x faster with cache")

    return features_df


def example_3_backtest_workflow():
    """
    Example 3: Extract features for multi-season backtest.

    Use case: Validating ensemble model on 2022-2024 seasons.
    """
    print("\n" + "=" * 80)
    print("Example 3: Multi-Season Backtest Workflow")
    print("=" * 80)

    loader = ParallelFeatureLoader(use_redis=True, n_workers=8)

    all_seasons_features = []
    total_start = time.time()

    # Extract 3 seasons (2022-2024)
    for season in [2022, 2023, 2024]:
        start = time.time()
        season_features = loader.extract_season_features(
            season=season, weeks=list(range(1, 13))  # First 12 weeks only
        )
        elapsed = time.time() - start

        all_seasons_features.append(season_features)
        print(
            f"  {season}: {len(season_features)} games in {elapsed:.2f}s ({len(season_features)/elapsed:.1f} games/sec)"
        )

    # Combine all seasons
    combined_df = pd.concat(all_seasons_features, ignore_index=True)
    total_elapsed = time.time() - total_start

    print(f"\n✓ Total: {len(combined_df)} games in {total_elapsed:.2f}s")
    print(f"  Average throughput: {len(combined_df)/total_elapsed:.1f} games/sec")
    print(f"  Sequential estimate: {len(combined_df) * 1.0:.0f}s (1 game/sec)")
    print(f"  Actual speedup: {len(combined_df) / total_elapsed:.1f}x faster")

    return combined_df


def example_4_cache_management():
    """
    Example 4: Cache management and monitoring.

    Use case: Production monitoring and cache invalidation.
    """
    print("\n" + "=" * 80)
    print("Example 4: Cache Management")
    print("=" * 80)

    loader = ParallelFeatureLoader(use_redis=True, n_workers=8)

    # Get cache statistics
    stats = loader.get_cache_stats()
    print("\n  Cache Statistics:")
    print(f"    Total keys: {stats['total_keys']}")
    print(f"    Memory used: {stats['memory_used_mb']:.2f} MB")
    print(f"    Keyspace hits: {stats['keyspace_hits']}")
    print(f"    Keyspace misses: {stats['keyspace_misses']}")
    print(f"    Hit rate: {stats['hit_rate']*100:.1f}%")

    # Clear specific pattern
    print("\n  Clearing cache for 2024 season...")
    loader.clear_cache(pattern="features:2024:*")

    # Check stats after clear
    stats_after = loader.get_cache_stats()
    print(f"    Keys after clear: {stats_after['total_keys']}")
    print(f"    Memory freed: {stats['memory_used_mb'] - stats_after['memory_used_mb']:.2f} MB")


def example_5_feature_selection():
    """
    Example 5: Selective feature extraction (specific feature types only).

    Use case: When you only need certain feature categories.
    """
    print("\n" + "=" * 80)
    print("Example 5: Selective Feature Extraction")
    print("=" * 80)

    loader = ParallelFeatureLoader(use_redis=True, n_workers=8)

    # Only extract vegas and environmental features (skip team stats and injuries)
    start = time.time()
    features_df = loader.extract_season_features(
        season=2024,
        weeks=[1, 2, 3, 4, 5],
        feature_types=["vegas", "environmental"],  # Subset of features
    )
    elapsed = time.time() - start

    print(f"\n✓ Extracted {len(features_df)} games in {elapsed:.2f}s")
    print(f"  Features: {len(features_df.columns)} columns (reduced set)")
    print(f"  Available columns: {list(features_df.columns[:15])}...")

    return features_df


def example_6_integration_with_xgboost():
    """
    Example 6: Integration with XGBoost training pipeline.

    Use case: Training XGBoost v5 model on historical data.
    """
    print("\n" + "=" * 80)
    print("Example 6: XGBoost Training Integration")
    print("=" * 80)

    loader = ParallelFeatureLoader(use_redis=True, n_workers=8)

    # Extract training data (2020-2023)
    print("\n  Extracting training data (2020-2023)...")
    train_features = []
    for season in [2020, 2021, 2022, 2023]:
        season_data = loader.extract_season_features(season=season, weeks=list(range(1, 19)))
        train_features.append(season_data)

    train_df = pd.concat(train_features, ignore_index=True)
    print(f"✓ Training set: {len(train_df)} games, {len(train_df.columns)} features")

    # Extract validation data (2024)
    print("\n  Extracting validation data (2024 weeks 1-10)...")
    val_df = loader.extract_season_features(season=2024, weeks=list(range(1, 11)))
    print(f"✓ Validation set: {len(val_df)} games, {len(val_df.columns)} features")

    # Prepare feature matrix (X) and target (y)
    # NOTE: This assumes you have actual outcomes in the database
    # For demo purposes, we'll just show the feature preparation step

    feature_cols = [
        col
        for col in train_df.columns
        if col not in ["game_id", "season", "week", "home_team", "away_team"]
    ]
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]

    print("\n  Feature matrix shapes:")
    print(f"    X_train: {X_train.shape}")
    print(f"    X_val: {X_val.shape}")

    # At this point, you would:
    # 1. Load actual outcomes (home_score, away_score) from database
    # 2. Create target variable (e.g., spread_cover = (home_score - away_score) > spread_close)
    # 3. Train XGBoost model: xgb.train(params, dtrain, ...)
    # 4. Validate on X_val

    print("\n  Next steps:")
    print("    1. Load actual game outcomes from database")
    print("    2. Create target variable (spread_cover, total_over, etc.)")
    print("    3. Train XGBoost model on X_train")
    print("    4. Validate on X_val")

    return X_train, X_val


def benchmark_comparison():
    """
    Benchmark: Compare sequential vs parallel vs cached extraction.
    """
    print("\n" + "=" * 80)
    print("BENCHMARK: Sequential vs Parallel vs Cached")
    print("=" * 80)

    loader = ParallelFeatureLoader(use_redis=True, n_workers=8)

    # Test dataset: 2024 weeks 1-5 (~75 games)
    season = 2024
    weeks = [1, 2, 3, 4, 5]

    print(f"\n  Dataset: {season} weeks {weeks}")

    # Run 1: Cold start (no cache)
    loader.clear_cache(pattern=f"features:{season}:*")
    start = time.time()
    df1 = loader.extract_season_features(season=season, weeks=weeks)
    time_cold = time.time() - start

    # Run 2: Warm start (cache populated)
    start = time.time()
    games_sample = df1[["season", "week", "home_team", "away_team"]]
    df2 = loader.extract_game_features_batch(games_sample, use_cache=True)
    time_warm = time.time() - start

    # Run 3: No cache (for comparison)
    loader.clear_cache(pattern=f"features:{season}:*")
    start = time.time()
    df3 = loader.extract_game_features_batch(games_sample, use_cache=False)
    time_no_cache = time.time() - start

    print("\n  Results:")
    print(f"    Cold start (batch SQL):     {time_cold:.2f}s ({len(df1)/time_cold:.1f} games/sec)")
    print(f"    Warm start (Redis cache):   {time_warm:.2f}s ({len(df2)/time_warm:.1f} games/sec)")
    print(
        f"    No cache (batch SQL):       {time_no_cache:.2f}s ({len(df3)/time_no_cache:.1f} games/sec)"
    )
    print("\n  Speedup:")
    print(f"    Batch vs Sequential (est): {len(df1) / time_cold:.1f}x")
    print(f"    Cache vs Cold start:        {time_cold / time_warm:.1f}x")

    # Estimate sequential performance (1 game/sec)
    sequential_estimate = len(df1) * 1.0
    print(f"\n  Sequential estimate: {sequential_estimate:.0f}s (1 game/sec)")
    print(f"  Actual performance:  {time_cold:.2f}s")
    print(f"  Total speedup:       {sequential_estimate / time_cold:.1f}x")


def main():
    """
    Run all examples and benchmark.
    """
    print("\n" + "=" * 80)
    print("ULTRA-FAST PARALLEL FEATURE LOADER - USAGE EXAMPLES")
    print("=" * 80)

    try:
        # Example 1: Single season batch extraction
        example_1_single_season()

        # Example 2: Batch game extraction (API use case)
        example_2_batch_extraction()

        # Example 3: Multi-season backtest
        example_3_backtest_workflow()

        # Example 4: Cache management
        example_4_cache_management()

        # Example 5: Selective feature extraction
        example_5_feature_selection()

        # Example 6: XGBoost integration
        example_6_integration_with_xgboost()

        # Benchmark
        benchmark_comparison()

        print("\n" + "=" * 80)
        print("ALL EXAMPLES COMPLETE!")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
