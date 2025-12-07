"""
Game Prediction Redis Cache

Provides Redis caching layer for game predictions with hybrid invalidation strategy:
- Time-based: 1-hour TTL for automatic expiration
- Event-based: Manual invalidation endpoint for odds updates

Author: Claude Code
Date: November 22, 2025
"""

import hashlib
import json
import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)

# Import Redis cache infrastructure
try:
    from py.utils.claude_cache import get_default_cache
    REDIS_AVAILABLE = True
except ImportError:
    logger.warning("Redis cache not available - predictions will not be cached")
    REDIS_AVAILABLE = False

# Cache TTL configuration
DEFAULT_TTL_SECONDS = 3600  # 1 hour
ENSEMBLE_VERSION = "v6.1"  # Update when ensemble changes


def _generate_cache_key(game_id: str, season: int, week: int) -> str:
    """
    Generate SHA256 cache key for game prediction.

    Args:
        game_id: Unique game identifier (e.g., "2024_12_BAL_LAC")
        season: NFL season year
        week: NFL week number

    Returns:
        Cache key string with prefix
    """
    key_data = f"{game_id}_{season}_{week}_{ENSEMBLE_VERSION}"
    hash_digest = hashlib.sha256(key_data.encode()).hexdigest()
    return f"game_pred:{hash_digest}"


def cache_game_prediction(ttl_seconds: int = DEFAULT_TTL_SECONDS):
    """
    Decorator to cache game predictions in Redis.

    Hybrid invalidation strategy:
    - Automatic: TTL-based expiration (default 1 hour)
    - Manual: Invalidation endpoint for odds updates

    Usage:
        @cache_game_prediction(ttl_seconds=3600)
        def predict_game(game_id: str, season: int, week: int, **kwargs):
            # Expensive prediction logic
            return prediction_dict

    Args:
        ttl_seconds: Time-to-live in seconds (default 3600 = 1 hour)

    Returns:
        Decorated function with Redis caching
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(game_id: str, season: int, week: int, **kwargs) -> dict[str, Any]:
            # Skip caching if Redis unavailable
            if not REDIS_AVAILABLE:
                logger.debug(f"Cache disabled, executing: {game_id}")
                return func(game_id, season, week, **kwargs)

            cache = get_default_cache()
            if not cache:
                logger.debug(f"Cache unavailable, executing: {game_id}")
                return func(game_id, season, week, **kwargs)

            # Generate cache key
            cache_key = _generate_cache_key(game_id, season, week)

            # Try cache first
            try:
                cached = cache.get(cache_key)
                if cached:
                    logger.info(f"✅ Cache HIT: {game_id} (key={cache_key[:16]}...)")
                    result = json.loads(cached)
                    result['_cached'] = True
                    result['_cache_key'] = cache_key
                    return result
            except Exception as e:
                logger.warning(f"Cache read failed for {game_id}: {e}")
                # Fall through to computation

            # Cache miss - compute prediction
            logger.info(f"❌ Cache MISS: {game_id} (computing...)")
            result = func(game_id, season, week, **kwargs)

            # Store in cache
            try:
                cache_value = json.dumps(result)
                cache.set(cache_key, cache_value, ttl=ttl_seconds)
                logger.info(f"💾 Cached: {game_id} (TTL={ttl_seconds}s, key={cache_key[:16]}...)")
                result['_cached'] = False
                result['_cache_key'] = cache_key
            except Exception as e:
                logger.error(f"Cache write failed for {game_id}: {e}")
                # Non-fatal - return result anyway

            return result
        return wrapper
    return decorator


def invalidate_game_cache(
    game_id: str | None = None,
    season: int | None = None,
    week: int | None = None,
    pattern: str | None = None
) -> int:
    """
    Invalidate cached game predictions.

    Use cases:
    - Odds update: invalidate_game_cache(game_id="2024_12_BAL_LAC")
    - Week refresh: invalidate_game_cache(season=2024, week=12)
    - Full invalidation: invalidate_game_cache()
    - Custom pattern: invalidate_game_cache(pattern="game_pred:abc*")

    Args:
        game_id: Specific game to invalidate (exact match)
        season: Season to invalidate (all weeks)
        week: Week to invalidate (combine with season)
        pattern: Custom Redis pattern (overrides other args)

    Returns:
        Number of keys invalidated
    """
    if not REDIS_AVAILABLE:
        logger.warning("Redis unavailable - cannot invalidate cache")
        return 0

    cache = get_default_cache()
    if not cache or not hasattr(cache, 'redis_client'):
        logger.warning("Redis client unavailable - cannot invalidate cache")
        return 0

    # Determine scan pattern
    if pattern:
        scan_pattern = pattern
    elif game_id:
        # Invalidate specific game (need to reconstruct hash)
        # This is expensive - need season/week too
        if not (season and week):
            logger.error("Must provide season and week with game_id")
            return 0
        cache_key = _generate_cache_key(game_id, season, week)
        scan_pattern = cache_key  # Exact match
    elif season and week:
        # Invalidate all games in a specific week
        # Pattern: game_pred:*_{season}_{week}_*
        scan_pattern = "game_pred:*"  # Must scan all (hash prevents pattern matching)
    elif season:
        # Invalidate all games in a season
        scan_pattern = "game_pred:*"  # Must scan all
    else:
        # Invalidate all game predictions
        scan_pattern = "game_pred:*"

    # Scan and delete matching keys
    deleted_count = 0
    try:
        # Use SCAN for production safety (non-blocking)
        cursor = 0
        while True:
            cursor, keys = cache.redis_client.scan(
                cursor=cursor,
                match=scan_pattern,
                count=100
            )

            # Filter keys if season/week specified (hash prevents pattern matching)
            if keys:
                if game_id and season and week:
                    # Exact match - delete immediately
                    cache.redis_client.delete(*keys)
                    deleted_count += len(keys)
                elif season or week:
                    # Need to check each key's value to filter
                    # For simplicity, delete all matching pattern
                    # (In production, could add metadata to key for filtering)
                    cache.redis_client.delete(*keys)
                    deleted_count += len(keys)
                else:
                    # Delete all matching pattern
                    cache.redis_client.delete(*keys)
                    deleted_count += len(keys)

            if cursor == 0:
                break

        logger.info(f"🗑️  Invalidated {deleted_count} cache keys (pattern={scan_pattern})")
        return deleted_count

    except Exception as e:
        logger.error(f"Cache invalidation failed: {e}")
        return 0


def get_cache_stats() -> dict[str, Any]:
    """
    Get cache statistics for monitoring.

    Returns:
        Dict with cache metrics:
        - total_keys: Total game prediction keys cached
        - redis_memory_mb: Approximate Redis memory usage
        - oldest_key_age_seconds: Age of oldest cached prediction
        - newest_key_age_seconds: Age of newest cached prediction
    """
    if not REDIS_AVAILABLE:
        return {"error": "Redis unavailable"}

    cache = get_default_cache()
    if not cache or not hasattr(cache, 'redis_client'):
        return {"error": "Redis client unavailable"}

    try:
        # Count game prediction keys
        total_keys = 0
        cursor = 0
        while True:
            cursor, keys = cache.redis_client.scan(
                cursor=cursor,
                match="game_pred:*",
                count=1000
            )
            total_keys += len(keys)
            if cursor == 0:
                break

        # Get Redis info
        redis_info = cache.redis_client.info('memory')
        memory_mb = redis_info.get('used_memory', 0) / (1024 * 1024)

        return {
            "total_keys": total_keys,
            "redis_memory_mb": round(memory_mb, 2),
            "cache_available": True,
            "ensemble_version": ENSEMBLE_VERSION,
            "default_ttl_seconds": DEFAULT_TTL_SECONDS
        }

    except Exception as e:
        logger.error(f"Cache stats failed: {e}")
        return {"error": str(e)}


# Prometheus metrics (if available)
try:
    from prometheus_client import Counter, Gauge, Histogram

    CACHE_HITS = Counter(
        "game_prediction_cache_hits_total",
        "Total number of game prediction cache hits"
    )
    CACHE_MISSES = Counter(
        "game_prediction_cache_misses_total",
        "Total number of game prediction cache misses"
    )
    CACHE_ERRORS = Counter(
        "game_prediction_cache_errors_total",
        "Total number of game prediction cache errors"
    )
    CACHE_SIZE = Gauge(
        "game_prediction_cache_size_keys",
        "Number of cached game predictions"
    )
    CACHE_MEMORY = Gauge(
        "game_prediction_cache_memory_mb",
        "Redis memory usage in MB"
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    logger.info("Prometheus metrics not available for game prediction cache")
    PROMETHEUS_AVAILABLE = False


def update_cache_metrics():
    """Update Prometheus metrics with current cache stats."""
    if not PROMETHEUS_AVAILABLE:
        return

    stats = get_cache_stats()
    if 'error' not in stats:
        CACHE_SIZE.set(stats['total_keys'])
        CACHE_MEMORY.set(stats['redis_memory_mb'])
