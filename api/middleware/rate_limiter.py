"""
Redis-Based Rate Limiting Middleware

Implements sliding window rate limiting using Redis for distributed tracking.

Features:
- Sliding window algorithm (more accurate than fixed window)
- Configurable limits per endpoint
- Rate limit headers (X-RateLimit-*)
- 429 Too Many Requests response
- Redis failover to in-memory (degraded mode)

Author: Claude Code
Date: 2025-11-15
"""

import logging
import os
import time
from collections.abc import Callable
from datetime import datetime

from fastapi import Request, status
from fastapi.responses import JSONResponse
from redis import Redis
from redis.exceptions import RedisError
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Redis-based rate limiting middleware using sliding window algorithm

    Algorithm:
    1. Use sorted set with timestamps as scores
    2. Remove entries older than window
    3. Count entries in window
    4. Add new entry if under limit
    5. Return 429 if over limit

    Redis key format: ratelimit:{identifier}:{endpoint}
    """

    def __init__(
        self,
        app,
        redis_client: Redis | None = None,
        default_limit: int = 100,
        window_seconds: int = 3600,
        identifier_func: Callable | None = None,
        endpoint_limits: dict[str, int] | None = None
    ):
        """
        Initialize rate limiter

        Args:
            app: FastAPI application
            redis_client: Redis client instance (creates new if None)
            default_limit: Default requests per window
            window_seconds: Time window in seconds (default: 1 hour)
            identifier_func: Function to extract identifier from request
            endpoint_limits: Per-endpoint rate limits (overrides default)
        """
        super().__init__(app)
        self.redis_client = redis_client or self._create_redis_client()
        self.default_limit = default_limit
        self.window_seconds = window_seconds
        self.identifier_func = identifier_func or self._default_identifier
        self.endpoint_limits = endpoint_limits or {}

        # Fallback to in-memory if Redis unavailable
        self._redis_available = self._check_redis()
        self._memory_store: dict[str, list] = {}  # Fallback storage

        if not self._redis_available:
            logger.warning(
                "Redis unavailable - rate limiting in DEGRADED MODE "
                "(in-memory, not distributed)"
            )

    def _create_redis_client(self) -> Redis:
        """Create Redis client from environment variables"""
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        return Redis.from_url(redis_url, decode_responses=False)

    def _check_redis(self) -> bool:
        """Check if Redis is available"""
        try:
            self.redis_client.ping()
            return True
        except (RedisError, ConnectionError) as e:
            logger.error(f"Redis connection failed: {e}")
            return False

    def _default_identifier(self, request: Request) -> str:
        """
        Default identifier extraction (uses API key if present, else IP)

        Args:
            request: FastAPI request

        Returns:
            Identifier string
        """
        # Try API key first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{api_key[:16]}"  # Use prefix to avoid key exposure

        # Fallback to IP address
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Use first IP in X-Forwarded-For chain
            ip = forwarded_for.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"

        return f"ip:{ip}"

    def _get_endpoint_limit(self, endpoint: str) -> int:
        """
        Get rate limit for endpoint

        Args:
            endpoint: Request endpoint path

        Returns:
            Rate limit for endpoint
        """
        return self.endpoint_limits.get(endpoint, self.default_limit)

    async def dispatch(self, request: Request, call_next):
        """
        Process request with rate limiting

        Args:
            request: FastAPI request
            call_next: Next middleware in chain

        Returns:
            Response or 429 if rate limited
        """
        # Skip rate limiting for health check and metrics
        if request.url.path in ["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)

        # Get identifier and endpoint
        identifier = self.identifier_func(request)
        endpoint = request.url.path
        limit = self._get_endpoint_limit(endpoint)

        # Check rate limit
        is_allowed, remaining, reset_time = await self._check_rate_limit(
            identifier, endpoint, limit
        )

        # Add rate limit headers
        headers = {
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(reset_time),
        }

        if not is_allowed:
            retry_after = max(1, reset_time - int(time.time()))
            headers["Retry-After"] = str(retry_after)

            logger.warning(
                f"Rate limit exceeded: {identifier} on {endpoint} "
                f"({limit} requests per {self.window_seconds}s)"
            )

            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "RateLimitExceeded",
                    "message": (
                        f"Rate limit exceeded. Limit: {limit} requests per "
                        f"{self.window_seconds // 3600} hour(s). "
                        f"Try again in {retry_after} seconds."
                    ),
                    "details": {
                        "limit": limit,
                        "remaining": 0,
                        "reset_at": datetime.fromtimestamp(reset_time).isoformat()
                    },
                    "timestamp": datetime.utcnow().isoformat()
                },
                headers=headers
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers to response
        for header, value in headers.items():
            response.headers[header] = value

        return response

    async def _check_rate_limit(
        self,
        identifier: str,
        endpoint: str,
        limit: int
    ) -> tuple[bool, int, int]:
        """
        Check rate limit using sliding window algorithm

        Args:
            identifier: Request identifier
            endpoint: Request endpoint
            limit: Rate limit

        Returns:
            Tuple of (is_allowed, remaining_requests, reset_timestamp)
        """
        if self._redis_available:
            return await self._check_rate_limit_redis(identifier, endpoint, limit)
        else:
            return await self._check_rate_limit_memory(identifier, endpoint, limit)

    async def _check_rate_limit_redis(
        self,
        identifier: str,
        endpoint: str,
        limit: int
    ) -> tuple[bool, int, int]:
        """
        Redis-based rate limiting with sliding window

        Uses sorted set with timestamps as scores for accurate sliding window.

        Args:
            identifier: Request identifier
            endpoint: Request endpoint
            limit: Rate limit

        Returns:
            Tuple of (is_allowed, remaining_requests, reset_timestamp)
        """
        key = f"ratelimit:{identifier}:{endpoint}"
        now = time.time()
        window_start = now - self.window_seconds

        try:
            pipe = self.redis_client.pipeline()

            # Remove old entries outside window
            pipe.zremrangebyscore(key, 0, window_start)

            # Count entries in window
            pipe.zcard(key)

            # Add current request timestamp
            pipe.zadd(key, {str(now): now})

            # Set expiry on key
            pipe.expire(key, self.window_seconds)

            results = pipe.execute()
            count = results[1]  # ZCARD result

            # Calculate remaining and reset time
            remaining = max(0, limit - count - 1)  # -1 for current request
            reset_time = int(now + self.window_seconds)

            is_allowed = count < limit

            return is_allowed, remaining, reset_time

        except RedisError as e:
            logger.error(f"Redis error in rate limiting: {e}")
            # Fail open (allow request) on Redis errors
            return True, limit - 1, int(now + self.window_seconds)

    async def _check_rate_limit_memory(
        self,
        identifier: str,
        endpoint: str,
        limit: int
    ) -> tuple[bool, int, int]:
        """
        In-memory fallback rate limiting (not distributed)

        WARNING: Only works for single instance. Use Redis for production.

        Args:
            identifier: Request identifier
            endpoint: Request endpoint
            limit: Rate limit

        Returns:
            Tuple of (is_allowed, remaining_requests, reset_timestamp)
        """
        key = f"{identifier}:{endpoint}"
        now = time.time()
        window_start = now - self.window_seconds

        # Get or create entry
        if key not in self._memory_store:
            self._memory_store[key] = []

        # Remove old entries
        self._memory_store[key] = [
            ts for ts in self._memory_store[key]
            if ts > window_start
        ]

        # Count requests in window
        count = len(self._memory_store[key])

        # Calculate remaining and reset time
        remaining = max(0, limit - count - 1)
        reset_time = int(now + self.window_seconds)

        is_allowed = count < limit

        if is_allowed:
            # Add current request
            self._memory_store[key].append(now)

        return is_allowed, remaining, reset_time

    def get_usage_stats(self, identifier: str, endpoint: str) -> dict:
        """
        Get current usage statistics for identifier/endpoint

        Args:
            identifier: Request identifier
            endpoint: Request endpoint

        Returns:
            Dictionary with usage statistics
        """
        key = f"ratelimit:{identifier}:{endpoint}"
        now = time.time()
        window_start = now - self.window_seconds

        try:
            if self._redis_available:
                # Count requests in current window
                count = self.redis_client.zcount(key, window_start, now)
                limit = self._get_endpoint_limit(endpoint)

                return {
                    "identifier": identifier,
                    "endpoint": endpoint,
                    "requests_in_window": count,
                    "limit": limit,
                    "remaining": max(0, limit - count),
                    "window_seconds": self.window_seconds,
                    "window_start": datetime.fromtimestamp(window_start).isoformat(),
                    "window_end": datetime.fromtimestamp(now).isoformat()
                }
            else:
                # In-memory fallback
                store_key = f"{identifier}:{endpoint}"
                count = len(self._memory_store.get(store_key, []))
                limit = self._get_endpoint_limit(endpoint)

                return {
                    "identifier": identifier,
                    "endpoint": endpoint,
                    "requests_in_window": count,
                    "limit": limit,
                    "remaining": max(0, limit - count),
                    "window_seconds": self.window_seconds,
                    "mode": "memory_fallback"
                }

        except Exception as e:
            logger.error(f"Error getting usage stats: {e}")
            return {
                "error": str(e),
                "identifier": identifier,
                "endpoint": endpoint
            }


def create_rate_limiter(
    redis_client: Redis | None = None,
    default_limit: int = 100,
    window_seconds: int = 3600
) -> RateLimitMiddleware:
    """
    Factory function to create rate limiter middleware

    Args:
        redis_client: Redis client instance
        default_limit: Default requests per window
        window_seconds: Time window in seconds

    Returns:
        Configured RateLimitMiddleware instance
    """
    # Define per-endpoint limits
    endpoint_limits = {
        "/predict": 100,  # 100 predictions per hour
        "/models": 1000,  # Higher limit for read-only
        "/health": 10000,  # Very high limit for health checks
    }

    return RateLimitMiddleware(
        app=None,  # Will be set by FastAPI
        redis_client=redis_client,
        default_limit=default_limit,
        window_seconds=window_seconds,
        endpoint_limits=endpoint_limits
    )
