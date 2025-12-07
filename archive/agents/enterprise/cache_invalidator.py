#!/usr/bin/env python3
"""
Cache Invalidation Service for Redis Phase 2.

Centralized cache invalidation API with pattern matching, bulk operations,
and integration with model registry updates.

Author: NFL Analytics Team
Date: November 10, 2025
"""

import logging
from dataclasses import dataclass
from typing import Any

from py.utils.claude_cache import ClaudeRedisCache, get_default_cache

logger = logging.getLogger(__name__)


@dataclass
class InvalidationResult:
    """Result of cache invalidation operation."""
    keys_matched: int
    keys_deleted: int
    patterns_used: list[str]
    execution_time_ms: float
    errors: list[str]


class CacheInvalidator:
    """
    Centralized cache invalidation service.

    Provides pattern-based invalidation for different cache types:
    - Model cache: model:{type}:{name}:*
    - Query cache: query:{key}
    - R results: r_result:{model}:{player}:*
    - Streamlit: streamlit:{function}:*
    - Player features: player_features:{player_id}:*

    Example usage:
        ```python
        invalidator = CacheInvalidator()

        # Invalidate all BNN models
        result = invalidator.invalidate_pattern("model:bnn:*")

        # Invalidate specific player's features
        result = invalidator.invalidate_player("00-0019596")

        # Invalidate after model retraining
        result = invalidator.invalidate_model("bnn_passing_v1")
        ```
    """

    def __init__(self, redis_cache: ClaudeRedisCache | None = None):
        """
        Initialize cache invalidator.

        Args:
            redis_cache: Redis cache instance (uses default if None)
        """
        self._redis_cache = redis_cache or get_default_cache()

        if not self._redis_cache or not self._redis_cache.is_connected():
            logger.warning("Redis cache unavailable - invalidation disabled")

    def invalidate_pattern(
        self,
        pattern: str,
        dry_run: bool = False
    ) -> InvalidationResult:
        """
        Invalidate all keys matching a pattern.

        Args:
            pattern: Redis key pattern (supports * and ? wildcards)
            dry_run: If True, only count matching keys without deleting

        Returns:
            InvalidationResult with deletion statistics

        Example:
            # Invalidate all XGBoost models
            result = invalidator.invalidate_pattern("model:xgboost:*")
        """
        if not self._redis_cache:
            return InvalidationResult(
                keys_matched=0,
                keys_deleted=0,
                patterns_used=[pattern],
                execution_time_ms=0.0,
                errors=["Redis cache unavailable"]
            )

        import time
        start_time = time.time()

        errors = []
        keys_matched = 0
        keys_deleted = 0

        try:
            # Use Redis SCAN for safe pattern matching
            # Note: ClaudeRedisCache wraps redis-py, may need to access underlying client
            redis_client = self._redis_cache._client

            # Scan for matching keys (cursor-based iteration)
            cursor = 0
            matched_keys = []

            while True:
                cursor, keys = redis_client.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )
                matched_keys.extend(keys)

                if cursor == 0:
                    break

            keys_matched = len(matched_keys)

            if not dry_run and matched_keys:
                # Delete in batches (pipeline for efficiency)
                pipeline = redis_client.pipeline()
                for key in matched_keys:
                    pipeline.delete(key)
                results = pipeline.execute()
                keys_deleted = sum(results)

                logger.info(
                    f"Invalidated {keys_deleted}/{keys_matched} keys "
                    f"matching pattern: {pattern}"
                )
            else:
                logger.info(
                    f"[DRY RUN] Found {keys_matched} keys matching pattern: {pattern}"
                )

        except Exception as e:
            error_msg = f"Pattern invalidation failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

        execution_time_ms = (time.time() - start_time) * 1000

        return InvalidationResult(
            keys_matched=keys_matched,
            keys_deleted=keys_deleted,
            patterns_used=[pattern],
            execution_time_ms=execution_time_ms,
            errors=errors
        )

    def invalidate_model(
        self,
        model_name: str,
        model_type: str | None = None
    ) -> InvalidationResult:
        """
        Invalidate cached model(s).

        Args:
            model_name: Model name (e.g., "bnn_passing_v1")
            model_type: Model type filter ("bnn", "xgboost", "r_model", or None for all)

        Returns:
            InvalidationResult

        Example:
            # Invalidate specific BNN model
            result = invalidator.invalidate_model("bnn_passing_v1", "bnn")

            # Invalidate all versions of a model
            result = invalidator.invalidate_model("bnn_passing_*")
        """
        if model_type:
            pattern = f"model:{model_type}:{model_name}*"
        else:
            pattern = f"model:*:{model_name}*"

        logger.info(f"Invalidating model cache: {model_name} (pattern: {pattern})")
        return self.invalidate_pattern(pattern)

    def invalidate_player(
        self,
        player_id: str,
        season: int | None = None,
        week: int | None = None
    ) -> InvalidationResult:
        """
        Invalidate player-specific caches.

        Clears:
        - Player features: player_features:{player_id}:*
        - R results: r_result:*:{player_id}:*

        Args:
            player_id: Player ID (e.g., "00-0019596")
            season: Optional season filter
            week: Optional week filter

        Returns:
            InvalidationResult

        Example:
            # Invalidate all Patrick Mahomes caches
            result = invalidator.invalidate_player("00-0019596")

            # Invalidate specific week
            result = invalidator.invalidate_player("00-0019596", season=2024, week=10)
        """
        patterns = []

        # Player features pattern
        if season and week:
            patterns.append(f"player_features:{player_id}:s{season}w{week}")
        elif season:
            patterns.append(f"player_features:{player_id}:s{season}*")
        else:
            patterns.append(f"player_features:{player_id}:*")

        # R results pattern
        if season and week:
            patterns.append(f"r_result:*:{player_id}:s{season}w{week}")
        elif season:
            patterns.append(f"r_result:*:{player_id}:s{season}*")
        else:
            patterns.append(f"r_result:*:{player_id}:*")

        logger.info(f"Invalidating player caches: {player_id}")

        # Invalidate all patterns
        results = []
        for pattern in patterns:
            result = self.invalidate_pattern(pattern)
            results.append(result)

        # Combine results
        total_matched = sum(r.keys_matched for r in results)
        total_deleted = sum(r.keys_deleted for r in results)
        all_errors = [e for r in results for e in r.errors]
        total_time = sum(r.execution_time_ms for r in results)

        return InvalidationResult(
            keys_matched=total_matched,
            keys_deleted=total_deleted,
            patterns_used=patterns,
            execution_time_ms=total_time,
            errors=all_errors
        )

    def invalidate_queries(self, query_keys: list[str]) -> InvalidationResult:
        """
        Invalidate specific query caches.

        Args:
            query_keys: List of query cache keys (without "query:" prefix)

        Returns:
            InvalidationResult

        Example:
            result = invalidator.invalidate_queries([
                "games_season_2024",
                "player_stats_season_2024_week_10"
            ])
        """
        if not self._redis_cache:
            return InvalidationResult(
                keys_matched=0,
                keys_deleted=0,
                patterns_used=[],
                execution_time_ms=0.0,
                errors=["Redis cache unavailable"]
            )

        import time
        start_time = time.time()

        errors = []
        keys_deleted = 0

        try:
            redis_client = self._redis_cache._client

            # Add "query:" prefix
            full_keys = [f"query:{key}" for key in query_keys]

            # Delete in batch
            if full_keys:
                pipeline = redis_client.pipeline()
                for key in full_keys:
                    pipeline.delete(key)
                results = pipeline.execute()
                keys_deleted = sum(results)

                logger.info(f"Invalidated {keys_deleted}/{len(full_keys)} query caches")

        except Exception as e:
            error_msg = f"Query invalidation failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

        execution_time_ms = (time.time() - start_time) * 1000

        return InvalidationResult(
            keys_matched=len(query_keys),
            keys_deleted=keys_deleted,
            patterns_used=[f"query:{k}" for k in query_keys],
            execution_time_ms=execution_time_ms,
            errors=errors
        )

    def invalidate_streamlit(self) -> InvalidationResult:
        """
        Invalidate all Streamlit dashboard caches.

        Returns:
            InvalidationResult

        Example:
            # Clear dashboard cache after data refresh
            result = invalidator.invalidate_streamlit()
        """
        logger.info("Invalidating Streamlit dashboard caches")
        return self.invalidate_pattern("streamlit:*")

    def invalidate_all(self, confirm: bool = False) -> InvalidationResult:
        """
        Invalidate ALL caches (nuclear option).

        Args:
            confirm: Must be True to proceed (safety check)

        Returns:
            InvalidationResult

        ⚠️  WARNING: This clears the entire cache! Use with extreme caution.

        Example:
            # Clear everything (requires explicit confirmation)
            result = invalidator.invalidate_all(confirm=True)
        """
        if not confirm:
            raise ValueError(
                "invalidate_all() requires confirm=True. "
                "This will clear the ENTIRE cache!"
            )

        logger.warning("⚠️  CLEARING ENTIRE CACHE (all patterns)")

        patterns = [
            "model:*",
            "query:*",
            "r_result:*",
            "streamlit:*",
            "player_features:*"
        ]

        results = []
        for pattern in patterns:
            result = self.invalidate_pattern(pattern)
            results.append(result)

        # Combine results
        total_matched = sum(r.keys_matched for r in results)
        total_deleted = sum(r.keys_deleted for r in results)
        all_errors = [e for r in results for e in r.errors]
        total_time = sum(r.execution_time_ms for r in results)

        logger.warning(
            f"✅ Cleared entire cache: {total_deleted} keys deleted "
            f"in {total_time:.2f}ms"
        )

        return InvalidationResult(
            keys_matched=total_matched,
            keys_deleted=total_deleted,
            patterns_used=patterns,
            execution_time_ms=total_time,
            errors=all_errors
        )

    def stats(self) -> dict[str, Any]:
        """
        Get cache statistics for monitoring.

        Returns:
            Dictionary with Redis statistics
        """
        if not self._redis_cache:
            return {"error": "Redis cache unavailable"}

        try:
            return self._redis_cache.stats()
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}


# Convenience functions for common invalidation patterns

def invalidate_after_model_training(
    model_name: str,
    model_type: str,
    player_ids: list[str] | None = None
) -> InvalidationResult:
    """
    Invalidate caches after model retraining.

    This should be called after successfully training a new model version
    to ensure predictions use the updated model.

    Args:
        model_name: Name of retrained model
        model_type: Type ("bnn", "xgboost", "r_model")
        player_ids: Optional list of player IDs to invalidate (if model is player-specific)

    Returns:
        InvalidationResult

    Example:
        # After retraining BNN model
        result = invalidate_after_model_training(
            model_name="bnn_passing_v2",
            model_type="bnn"
        )
    """
    invalidator = CacheInvalidator()

    # Invalidate model cache
    result = invalidator.invalidate_model(model_name, model_type)

    # Invalidate player-specific caches if provided
    if player_ids:
        for player_id in player_ids:
            player_result = invalidator.invalidate_player(player_id)
            result.keys_deleted += player_result.keys_deleted
            result.keys_matched += player_result.keys_matched
            result.execution_time_ms += player_result.execution_time_ms
            result.errors.extend(player_result.errors)

    logger.info(
        f"Post-training invalidation complete: {result.keys_deleted} keys cleared"
    )
    return result


def invalidate_after_data_ingestion(
    season: int | None = None,
    week: int | None = None
) -> InvalidationResult:
    """
    Invalidate caches after new data ingestion.

    Args:
        season: Season year (None = all seasons)
        week: Week number (None = all weeks)

    Returns:
        InvalidationResult

    Example:
        # After ingesting Week 10 data
        result = invalidate_after_data_ingestion(season=2024, week=10)
    """
    invalidator = CacheInvalidator()

    patterns = []

    # Invalidate all queries (they may reference new data)
    patterns.append("query:*")

    # Invalidate player features for affected weeks
    if season and week:
        patterns.append(f"player_features:*:s{season}w{week}")
    elif season:
        patterns.append(f"player_features:*:s{season}*")
    else:
        patterns.append("player_features:*")

    # Invalidate Streamlit dashboard
    patterns.append("streamlit:*")

    results = []
    for pattern in patterns:
        result = invalidator.invalidate_pattern(pattern)
        results.append(result)

    # Combine results
    total_matched = sum(r.keys_matched for r in results)
    total_deleted = sum(r.keys_deleted for r in results)
    all_errors = [e for r in results for e in r.errors]
    total_time = sum(r.execution_time_ms for r in results)

    logger.info(
        f"Post-ingestion invalidation complete: {total_deleted} keys cleared"
    )

    return InvalidationResult(
        keys_matched=total_matched,
        keys_deleted=total_deleted,
        patterns_used=patterns,
        execution_time_ms=total_time,
        errors=all_errors
    )


if __name__ == "__main__":
    """Test cache invalidator."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*60)
    print("CACHE INVALIDATOR TEST")
    print("="*60 + "\n")

    invalidator = CacheInvalidator()

    # Test 1: Pattern matching (dry run)
    print("TEST 1: Pattern Matching (dry run)")
    result = invalidator.invalidate_pattern("model:*", dry_run=True)
    print(f"  Matched: {result.keys_matched} keys")
    print(f"  Errors: {len(result.errors)}")
    print()

    # Test 2: Model invalidation
    print("TEST 2: Model Invalidation")
    result = invalidator.invalidate_model("bnn_passing_v1", "bnn")
    print(f"  Deleted: {result.keys_deleted} keys")
    print(f"  Time: {result.execution_time_ms:.2f}ms")
    print()

    # Test 3: Player invalidation
    print("TEST 3: Player Invalidation")
    result = invalidator.invalidate_player("00-0019596", season=2024, week=10)
    print(f"  Patterns: {len(result.patterns_used)}")
    print(f"  Deleted: {result.keys_deleted} keys")
    print()

    # Test 4: Query invalidation
    print("TEST 4: Query Invalidation")
    result = invalidator.invalidate_queries([
        "games_season_2024",
        "player_stats_2024_week_10"
    ])
    print(f"  Deleted: {result.keys_deleted} keys")
    print()

    # Test 5: Streamlit invalidation
    print("TEST 5: Streamlit Dashboard Invalidation")
    result = invalidator.invalidate_streamlit()
    print(f"  Deleted: {result.keys_deleted} keys")
    print()

    # Test 6: Cache stats
    print("TEST 6: Cache Statistics")
    stats = invalidator.stats()
    print(f"  Stats: {stats}")
    print()

    print("="*60)
    print("✅ ALL TESTS COMPLETE")
    print("="*60)
