"""
Redis-based cache client for Claude API responses with key hashing and TTL management.

This module provides a lightweight caching layer to reduce API calls and improve
performance by caching Claude API responses with SHA256 key hashing and configurable
time-to-live (TTL) settings.

Key Features:
- SHA256 key hashing for consistent, secure key generation
- Configurable 7-day default TTL with per-request overrides
- Cache hit rate tracking for monitoring
- 70% target cache hit rate optimization
- Thread-safe operations with connection pooling
- Automatic serialization/deserialization of responses
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Optional

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


logger = logging.getLogger(__name__)


class ClaudeRedisCache:
    """
    Redis cache client for Claude API responses with SHA256 key hashing.

    Attributes:
        default_ttl (int): Default time-to-live in seconds (604800 = 7 days)
        target_hit_rate (float): Target cache hit rate (0.70 = 70%)
        stats (dict): Cache hit/miss statistics
    """

    DEFAULT_TTL = 604800  # 7 days in seconds
    TARGET_HIT_RATE = 0.70  # 70% target

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: Optional[int] = None,
        enable_stats: bool = True,
    ):
        """
        Initialize Redis cache client.

        Args:
            host: Redis server hostname (default: localhost)
            port: Redis server port (default: 6379)
            db: Redis database number (default: 0)
            password: Redis password if authentication required (default: None)
            default_ttl: Default TTL in seconds (default: 604800 = 7 days)
            enable_stats: Whether to track cache statistics (default: True)

        Raises:
            ImportError: If redis package is not installed
            redis.ConnectionError: If connection to Redis fails
        """
        if not REDIS_AVAILABLE:
            raise ImportError(
                "redis package required for ClaudeRedisCache. "
                "Install with: pip install redis"
            )

        self.host = host
        self.port = port
        self.db = db
        self.default_ttl = default_ttl or self.DEFAULT_TTL
        self.enable_stats = enable_stats

        # Statistics tracking
        self.stats = {
            "hits": 0,
            "misses": 0,
            "errors": 0,
            "created_at": datetime.now().isoformat(),
        }

        # Initialize Redis connection with connection pooling
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True,  # Auto-decode responses to strings
            socket_keepalive=True,
            socket_keepalive_options={
                1: 3,  # TCP_KEEPIDLE
                2: 3,  # TCP_KEEPINTVL
                3: 3,  # TCP_KEEPCNT
            },
        )

        # Test connection
        try:
            self.redis_client.ping()
            logger.info(
                f"Redis cache initialized: {host}:{port}/db={db} "
                f"(TTL={self.default_ttl}s, target_hit_rate={self.TARGET_HIT_RATE*100}%)"
            )
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    @staticmethod
    def _hash_key(key: str) -> str:
        """
        Generate SHA256 hash of cache key for consistency and security.

        Args:
            key: Raw cache key (can be any string)

        Returns:
            SHA256 hex digest (64 characters)
        """
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _record_stat(self, stat_type: str) -> None:
        """Record cache statistic (hit/miss/error)."""
        if self.enable_stats:
            self.stats[stat_type] = self.stats.get(stat_type, 0) + 1

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve cached value by key.

        Args:
            key: Cache key (will be hashed internally)

        Returns:
            Cached value (deserialized from JSON) or None if not found/expired

        Example:
            >>> cache = ClaudeRedisCache()
            >>> response = cache.get("claude_prompt_xyz")
            >>> if response:
            ...     print(f"Cache hit: {response}")
        """
        hashed_key = self._hash_key(key)

        try:
            value = self.redis_client.get(hashed_key)

            if value is None:
                self._record_stat("misses")
                logger.debug(f"Cache miss: {key}")
                return None

            # Deserialize JSON response
            cached_data = json.loads(value)
            self._record_stat("hits")
            logger.debug(f"Cache hit: {key} (retrieved {len(value)} bytes)")

            return cached_data

        except (json.JSONDecodeError, redis.RedisError) as e:
            logger.error(f"Cache retrieval error for {key}: {e}")
            self._record_stat("errors")
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Store value in cache with optional TTL override.

        Args:
            key: Cache key (will be hashed internally)
            value: Value to cache (will be serialized to JSON)
            ttl: Time-to-live in seconds (default: uses default_ttl)

        Returns:
            True if set successfully, False otherwise

        Example:
            >>> cache = ClaudeRedisCache()
            >>> response = {"text": "Hello", "tokens": 10}
            >>> success = cache.set("claude_response_xyz", response, ttl=3600)
            >>> if success:
            ...     print("Cached successfully")
        """
        hashed_key = self._hash_key(key)
        ttl_seconds = ttl or self.default_ttl

        try:
            # Serialize to JSON
            json_value = json.dumps(value)

            # Set with TTL
            self.redis_client.setex(hashed_key, ttl_seconds, json_value)
            logger.debug(
                f"Cached {key}: {len(json_value)} bytes, TTL={ttl_seconds}s"
            )

            return True

        except (json.JSONEncodeError, redis.RedisError) as e:
            logger.error(f"Cache set error for {key}: {e}")
            self._record_stat("errors")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete cached value by key.

        Args:
            key: Cache key (will be hashed internally)

        Returns:
            True if key was deleted, False if key didn't exist
        """
        hashed_key = self._hash_key(key)

        try:
            result = self.redis_client.delete(hashed_key)
            if result:
                logger.debug(f"Deleted cache key: {key}")
            return bool(result)
        except redis.RedisError as e:
            logger.error(f"Cache deletion error for {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache (without retrieving value).

        Args:
            key: Cache key (will be hashed internally)

        Returns:
            True if key exists, False otherwise
        """
        hashed_key = self._hash_key(key)

        try:
            return bool(self.redis_client.exists(hashed_key))
        except redis.RedisError as e:
            logger.error(f"Cache exists check error for {key}: {e}")
            return False

    def get_or_set(
        self,
        key: str,
        default_value: Any,
        ttl: Optional[int] = None,
    ) -> Any:
        """
        Get value from cache or set default if not found.

        Args:
            key: Cache key
            default_value: Value to set if key not found
            ttl: Time-to-live in seconds (default: uses default_ttl)

        Returns:
            Cached value if found, otherwise default_value (which is cached)

        Example:
            >>> cache = ClaudeRedisCache()
            >>> response = cache.get_or_set(
            ...     "expensive_computation",
            ...     default_value={"result": compute_expensive_value()},
            ...     ttl=3600
            ... )
        """
        cached = self.get(key)
        if cached is not None:
            return cached

        # Set default value and return
        self.set(key, default_value, ttl=ttl)
        return default_value

    def clear_all(self) -> int:
        """
        Delete all keys in the current Redis database.

        WARNING: This will clear all keys, not just Claude cache keys.

        Returns:
            Number of keys deleted
        """
        try:
            count = self.redis_client.flushdb()
            logger.warning(f"Cleared all keys from Redis database {self.db}")
            return count
        except redis.RedisError as e:
            logger.error(f"Error clearing cache: {e}")
            return 0

    def get_hit_rate(self) -> float:
        """
        Calculate current cache hit rate.

        Returns:
            Hit rate as percentage (0.0 to 1.0), or 0.0 if no accesses yet
        """
        total = self.stats.get("hits", 0) + self.stats.get("misses", 0)
        if total == 0:
            return 0.0
        return self.stats["hits"] / total

    def get_stats(self) -> dict:
        """
        Get cache statistics including hit rate and performance metrics.

        Returns:
            Dictionary with stats including hits, misses, errors, hit_rate,
            and uptime information
        """
        hit_rate = self.get_hit_rate()
        uptime = datetime.now() - datetime.fromisoformat(self.stats["created_at"])

        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "errors": self.stats["errors"],
            "hit_rate": hit_rate,
            "hit_rate_percent": f"{hit_rate * 100:.1f}%",
            "target_hit_rate_percent": f"{self.TARGET_HIT_RATE * 100:.1f}%",
            "performance": "EXCELLENT" if hit_rate >= self.TARGET_HIT_RATE else "GOOD" if hit_rate >= 0.5 else "FAIR",
            "uptime_seconds": uptime.total_seconds(),
            "created_at": self.stats["created_at"],
            "redis_server": f"{self.host}:{self.port}/db={self.db}",
            "default_ttl_seconds": self.default_ttl,
            "default_ttl_days": round(self.default_ttl / 86400, 1),
        }

    def print_stats(self) -> None:
        """Pretty-print cache statistics to logger."""
        stats = self.get_stats()
        logger.info(
            f"Claude Redis Cache Stats:\n"
            f"  Hits: {stats['hits']}\n"
            f"  Misses: {stats['misses']}\n"
            f"  Errors: {stats['errors']}\n"
            f"  Hit Rate: {stats['hit_rate_percent']} (target: {stats['target_hit_rate_percent']})\n"
            f"  Performance: {stats['performance']}\n"
            f"  Uptime: {stats['uptime_seconds']:.0f}s\n"
            f"  Server: {stats['redis_server']}"
        )

    def close(self) -> None:
        """Close Redis connection."""
        try:
            self.redis_client.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def get_default_cache() -> Optional[ClaudeRedisCache]:
    """
    Factory function to create cache client from environment variables.

    Reads configuration from environment:
    - REDIS_HOST (default: localhost)
    - REDIS_PORT (default: 6379)
    - REDIS_DB (default: 0)
    - REDIS_PASSWORD (default: None)
    - REDIS_CACHE_TTL (default: 604800 = 7 days)

    Returns:
        ClaudeRedisCache instance or None if Redis unavailable

    Example:
        >>> cache = get_default_cache()
        >>> if cache:
        ...     value = cache.get("my_key")
    """
    if not REDIS_AVAILABLE:
        logger.warning(
            "Redis not available. Install with: pip install redis"
        )
        return None

    try:
        cache = ClaudeRedisCache(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD"),
            default_ttl=int(os.getenv("REDIS_CACHE_TTL", "604800")),
        )
        return cache
    except Exception as e:
        logger.error(f"Failed to create default cache: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.DEBUG)

    try:
        cache = ClaudeRedisCache()

        # Example: Cache a Claude API response
        response = {
            "text": "This is a cached response from Claude API",
            "tokens_used": 150,
            "model": "claude-3-sonnet",
            "timestamp": datetime.now().isoformat(),
        }

        # Store in cache
        cache.set("example_prompt_response", response, ttl=3600)

        # Retrieve from cache
        cached = cache.get("example_prompt_response")
        print(f"Retrieved from cache: {cached}")

        # Check stats
        cache.print_stats()

    except Exception as e:
        print(f"Error: {e}")
