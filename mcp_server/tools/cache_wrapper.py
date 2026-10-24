"""
Redis cache wrapper for MCP server tool responses.

Provides transparent caching layer for MCP tools with:
- Redis-backed persistent caching (7-day default TTL)
- SHA256 key hashing for consistency
- Cache hit/miss metrics tracking
- Graceful fallback to uncached execution
- Tool-specific TTL configuration

Author: NFL Analytics Team
Date: November 13, 2025
"""

import hashlib
import json
import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

try:
    from nfl_analytics.utils.claude_cache import ClaudeRedisCache, get_default_cache

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class MCPCacheWrapper:
    """
    Cache wrapper for MCP server tool responses.

    Wraps MCP tool functions to provide transparent caching with Redis.
    Falls back to uncached execution if Redis unavailable.

    Example usage:
        ```python
        cache_wrapper = MCPCacheWrapper()

        @cache_wrapper.cached(ttl=3600, key_prefix="team_news")
        def get_team_news(team: str, limit: int = 10) -> list:
            # Expensive operation (API call, web scraping, etc.)
            return fetch_espn_news(team, limit)

        # First call: cache miss, executes function
        news = get_team_news("PHI", limit=5)

        # Second call: cache hit, returns cached result
        news = get_team_news("PHI", limit=5)  # <200ms
        ```
    """

    # Tool-specific TTL configuration (in seconds)
    DEFAULT_TTLS = {
        "team_news": 3600,  # 1 hour (news updates frequently)
        "database_query": 300,  # 5 minutes (data changes frequently)
        "player_stats": 1800,  # 30 minutes
        "github_operations": 600,  # 10 minutes
        "file_operations": 60,  # 1 minute (files change often)
        "model_metadata": 3600,  # 1 hour
        "default": 7 * 24 * 3600,  # 7 days (ClaudeRedisCache default)
    }

    def __init__(self, redis_cache: ClaudeRedisCache | None = None):
        """
        Initialize MCP cache wrapper.

        Args:
            redis_cache: Redis cache instance (uses default if None)
        """
        self._redis_cache = redis_cache or get_default_cache()
        self._enabled = REDIS_AVAILABLE and self._redis_cache is not None

        if not self._enabled:
            logger.warning(
                "Redis cache unavailable for MCP tools - "
                "all tool calls will execute without caching"
            )
        else:
            logger.info("MCP Redis cache wrapper initialized")

    def _generate_cache_key(
        self, key_prefix: str, func_name: str, args: tuple, kwargs: dict
    ) -> str:
        """
        Generate cache key from function name and arguments.

        Args:
            key_prefix: Tool category prefix (e.g., "team_news", "database_query")
            func_name: Function name
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Cache key string (format: "mcp:{prefix}:{func}:{args_hash}")
        """
        # Create stable representation of arguments
        args_str = json.dumps(
            {"args": args, "kwargs": sorted(kwargs.items())},  # Sort for consistency
            sort_keys=True,
            default=str,
        )

        # Hash arguments to keep key length reasonable
        args_hash = hashlib.sha256(args_str.encode()).hexdigest()[:16]

        return f"mcp:{key_prefix}:{func_name}:{args_hash}"

    def cached(self, ttl: int | None = None, key_prefix: str = "default") -> Callable:
        """
        Decorator to add Redis caching to MCP tool functions.

        Args:
            ttl: Cache TTL in seconds (uses DEFAULT_TTLS if None)
            key_prefix: Tool category for cache key (e.g., "team_news")

        Returns:
            Decorated function with caching

        Example:
            ```python
            @cache_wrapper.cached(ttl=3600, key_prefix="team_news")
            def get_team_news(team: str, limit: int) -> list:
                return fetch_espn_news(team, limit)
            ```
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # If caching disabled, execute function directly
                if not self._enabled:
                    return func(*args, **kwargs)

                # Generate cache key
                cache_key = self._generate_cache_key(key_prefix, func.__name__, args, kwargs)

                # Try to get from cache
                cached_result = self._redis_cache.get(cache_key)
                if cached_result is not None:
                    logger.debug(
                        f"MCP cache HIT: {key_prefix}.{func.__name__} "
                        f"(key: {cache_key[:40]}...)"
                    )
                    return cached_result

                # Cache miss - execute function
                logger.debug(
                    f"MCP cache MISS: {key_prefix}.{func.__name__} " f"(key: {cache_key[:40]}...)"
                )
                result = func(*args, **kwargs)

                # Cache the result
                cache_ttl = ttl or self.DEFAULT_TTLS.get(key_prefix, self.DEFAULT_TTLS["default"])
                self._redis_cache.set(cache_key, result, ttl=cache_ttl)

                logger.debug(
                    f"MCP cached result: {key_prefix}.{func.__name__} " f"(TTL: {cache_ttl}s)"
                )

                return result

            # Preserve function metadata
            wrapper.__wrapped__ = func
            return wrapper

        return decorator

    def invalidate_prefix(self, key_prefix: str) -> int:
        """
        Invalidate all cached results for a tool category.

        Args:
            key_prefix: Tool category (e.g., "team_news", "database_query")

        Returns:
            Number of keys deleted

        Example:
            # Invalidate all team news caches
            cache_wrapper.invalidate_prefix("team_news")
        """
        if not self._enabled:
            logger.warning("Redis unavailable - cannot invalidate cache")
            return 0

        try:
            pattern = f"mcp:{key_prefix}:*"

            # Use CacheInvalidator if available
            try:
                from nfl_analytics.utils.cache_invalidator import CacheInvalidator

                invalidator = CacheInvalidator(self._redis_cache)
                result = invalidator.invalidate_pattern(pattern)
                return result.keys_deleted
            except ImportError:
                # Fallback: manual pattern deletion
                redis_client = self._redis_cache.redis_client
                cursor = 0
                deleted = 0

                while True:
                    cursor, keys = redis_client.scan(cursor=cursor, match=pattern, count=100)
                    if keys:
                        deleted += redis_client.delete(*keys)
                    if cursor == 0:
                        break

                logger.info(f"Invalidated {deleted} keys for prefix: {key_prefix}")
                return deleted

        except Exception as e:
            logger.error(f"Error invalidating cache prefix {key_prefix}: {e}")
            return 0

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats (hits, misses, hit_rate, etc.)
        """
        if not self._enabled:
            return {"error": "Redis cache unavailable"}

        try:
            return self._redis_cache.get_stats()
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}


# Global singleton instance
_cache_wrapper: MCPCacheWrapper | None = None


def get_mcp_cache_wrapper() -> MCPCacheWrapper:
    """
    Get global MCP cache wrapper instance (singleton).

    Returns:
        MCPCacheWrapper instance
    """
    global _cache_wrapper

    if _cache_wrapper is None:
        _cache_wrapper = MCPCacheWrapper()

    return _cache_wrapper


# Convenience decorators for common tool types
def cache_news(ttl: int = 3600):
    """Cache news fetcher results (default: 1 hour)."""
    wrapper = get_mcp_cache_wrapper()
    return wrapper.cached(ttl=ttl, key_prefix="team_news")


def cache_database_query(ttl: int = 300):
    """Cache database query results (default: 5 minutes)."""
    wrapper = get_mcp_cache_wrapper()
    return wrapper.cached(ttl=ttl, key_prefix="database_query")


def cache_github_operation(ttl: int = 600):
    """Cache GitHub operation results (default: 10 minutes)."""
    wrapper = get_mcp_cache_wrapper()
    return wrapper.cached(ttl=ttl, key_prefix="github_operations")


def cache_file_operation(ttl: int = 60):
    """Cache file operation results (default: 1 minute)."""
    wrapper = get_mcp_cache_wrapper()
    return wrapper.cached(ttl=ttl, key_prefix="file_operations")


def cache_model_metadata(ttl: int = 3600):
    """Cache model metadata results (default: 1 hour)."""
    wrapper = get_mcp_cache_wrapper()
    return wrapper.cached(ttl=ttl, key_prefix="model_metadata")


if __name__ == "__main__":
    """Test MCP cache wrapper."""
    import time

    logging.basicConfig(level=logging.DEBUG)

    print("\n" + "=" * 60)
    print("MCP CACHE WRAPPER TEST")
    print("=" * 60 + "\n")

    cache_wrapper = get_mcp_cache_wrapper()

    # Test 1: Basic caching
    print("TEST 1: Basic Caching")

    @cache_wrapper.cached(ttl=60, key_prefix="test")
    def expensive_operation(x: int, y: int) -> int:
        """Simulate expensive operation."""
        time.sleep(0.1)  # Simulate delay
        return x + y

    # First call (cache miss)
    start = time.time()
    result1 = expensive_operation(5, 3)
    elapsed1 = time.time() - start
    print(f"  First call: {result1} ({elapsed1*1000:.2f}ms)")

    # Second call (cache hit)
    start = time.time()
    result2 = expensive_operation(5, 3)
    elapsed2 = time.time() - start
    print(f"  Second call: {result2} ({elapsed2*1000:.2f}ms)")

    assert result1 == result2, "Results should match"
    assert elapsed2 < elapsed1 / 2, "Cached call should be much faster"
    print("  ✅ Basic caching works")
    print()

    # Test 2: Different arguments
    print("TEST 2: Different Arguments")
    result3 = expensive_operation(10, 20)
    print(f"  Different args: {result3} (should be cache miss)")
    print()

    # Test 3: Cache statistics
    print("TEST 3: Cache Statistics")
    stats = cache_wrapper.get_stats()
    print(f"  Stats: {json.dumps(stats, indent=2)}")
    print()

    # Test 4: Convenience decorators
    print("TEST 4: Convenience Decorators")

    @cache_news(ttl=60)
    def get_news(team: str) -> list:
        return [f"News for {team}"]

    news1 = get_news("PHI")
    news2 = get_news("PHI")
    print(f"  News cache working: {news1 == news2}")
    print()

    # Test 5: Cache invalidation
    print("TEST 5: Cache Invalidation")
    deleted = cache_wrapper.invalidate_prefix("test")
    print(f"  Deleted {deleted} keys from 'test' prefix")
    print()

    print("=" * 60)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 60)
