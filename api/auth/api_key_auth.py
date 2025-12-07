"""
API Key Authentication Module

Provides FastAPI dependency for validating API keys against database.

Features:
- UUID4-based key generation
- Database-backed validation
- Key rotation support
- Usage tracking
- Endpoint-level permissions

Author: Claude Code
Date: 2025-11-15
"""

import hashlib
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Any

import psycopg
from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

# API key header configuration
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class APIKey:
    """API Key model for authenticated requests"""

    def __init__(
        self,
        key_id: int,
        api_key: str,
        key_name: str,
        owner_email: str,
        rate_limit_per_hour: int,
        rate_limit_per_day: int,
        allowed_endpoints: list[str] | None = None,
        allowed_ip_ranges: list[str] | None = None,
        expires_at: datetime | None = None,
    ):
        self.key_id = key_id
        self.api_key = api_key
        self.key_name = key_name
        self.owner_email = owner_email
        self.rate_limit_per_hour = rate_limit_per_hour
        self.rate_limit_per_day = rate_limit_per_day
        self.allowed_endpoints = allowed_endpoints or []
        self.allowed_ip_ranges = allowed_ip_ranges or []
        self.expires_at = expires_at

    def is_expired(self) -> bool:
        """Check if API key is expired"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def can_access_endpoint(self, endpoint: str) -> bool:
        """Check if API key has permission for endpoint"""
        # NULL allowed_endpoints means all endpoints allowed
        if not self.allowed_endpoints:
            return True

        # Check if endpoint matches allowed list
        return endpoint in self.allowed_endpoints

    def __repr__(self) -> str:
        return f"APIKey(key_name={self.key_name}, owner={self.owner_email})"


class APIKeyAuthenticator:
    """Handles API key authentication and validation"""

    def __init__(self, db_url: str | None = None):
        """
        Initialize authenticator

        Args:
            db_url: Database connection string (uses env var if not provided)
        """
        self.db_url = db_url or os.getenv(
            "DATABASE_URL",
            "postgresql://dro:sicillionbillions@localhost:5544/devdb01"  # pragma: allowlist secret
        )
        self._conn: psycopg.Connection | None = None

    def _get_connection(self) -> psycopg.Connection:
        """Get or create database connection"""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.db_url, row_factory=dict_row)
        return self._conn

    def close(self):
        """Close database connection"""
        if self._conn and not self._conn.closed:
            self._conn.close()

    async def validate_api_key(
        self,
        api_key: str,
        endpoint: str,
        request: Request | None = None
    ) -> APIKey:
        """
        Validate API key against database

        Args:
            api_key: API key to validate
            endpoint: Endpoint being accessed
            request: FastAPI request object (optional, for IP validation)

        Returns:
            APIKey object if valid

        Raises:
            HTTPException: If validation fails (401, 403, 429)
        """
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required. Include 'X-API-Key' header in request.",
                headers={"WWW-Authenticate": "ApiKey"}
            )

        conn = self._get_connection()

        try:
            with conn.cursor() as cur:
                # Fetch API key record
                cur.execute(
                    """
                    SELECT
                        id, api_key, key_name, owner_email, owner_name,
                        is_active, is_revoked, rate_limit_per_hour,
                        rate_limit_per_day, allowed_endpoints, allowed_ip_ranges,
                        expires_at, last_used_at, total_requests
                    FROM auth.api_keys
                    WHERE api_key = %s
                    """,
                    (api_key,)
                )
                row = cur.fetchone()

                if not row:
                    logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid API key",
                        headers={"WWW-Authenticate": "ApiKey"}
                    )

                # Check if key is active
                if not row["is_active"] or row["is_revoked"]:
                    logger.warning(
                        f"Inactive/revoked API key attempted: "
                        f"{row['key_name']} ({row['owner_email']})"
                    )
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="API key is inactive or revoked. Contact administrator."
                    )

                # Create APIKey object
                api_key_obj = APIKey(
                    key_id=row["id"],
                    api_key=row["api_key"],
                    key_name=row["key_name"],
                    owner_email=row["owner_email"],
                    rate_limit_per_hour=row["rate_limit_per_hour"],
                    rate_limit_per_day=row["rate_limit_per_day"],
                    allowed_endpoints=row["allowed_endpoints"],
                    allowed_ip_ranges=row["allowed_ip_ranges"],
                    expires_at=row["expires_at"]
                )

                # Check expiry
                if api_key_obj.is_expired():
                    logger.warning(
                        f"Expired API key attempted: "
                        f"{row['key_name']} (expired {row['expires_at']})"
                    )
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"API key expired on {row['expires_at'].strftime('%Y-%m-%d')}. "
                               f"Please rotate your key."
                    )

                # Check endpoint permissions
                if not api_key_obj.can_access_endpoint(endpoint):
                    logger.warning(
                        f"Endpoint permission denied: {row['key_name']} "
                        f"attempted {endpoint}"
                    )
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"API key does not have permission for endpoint: {endpoint}"
                    )

                # Check rate limits
                await self._check_rate_limit(api_key_obj.key_id, conn)

                logger.info(
                    f"API key validated: {row['key_name']} ({row['owner_email']}) "
                    f"accessing {endpoint}"
                )

                return api_key_obj

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"API key validation error: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication service error"
            )

    async def _check_rate_limit(
        self,
        api_key_id: int,
        conn: psycopg.Connection
    ):
        """
        Check rate limit for API key

        Args:
            api_key_id: API key ID
            conn: Database connection

        Raises:
            HTTPException: If rate limit exceeded (429)
        """
        with conn.cursor() as cur:
            # Check hourly limit
            cur.execute(
                "SELECT * FROM auth.check_rate_limit(%s, 1)",
                (api_key_id,)
            )
            hourly = cur.fetchone()

            if hourly["is_limited"]:
                retry_after = 3600  # 1 hour in seconds
                logger.warning(
                    f"Hourly rate limit exceeded for API key ID {api_key_id}: "
                    f"{hourly['requests_in_window']}/{hourly['limit_per_window']}"
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=(
                        f"Hourly rate limit exceeded. "
                        f"Used {hourly['requests_in_window']}/{hourly['limit_per_window']} requests. "
                        f"Limit resets in {retry_after} seconds."
                    ),
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(hourly["limit_per_window"]),
                        "X-RateLimit-Remaining": str(
                            max(0, hourly["limit_per_window"] - hourly["requests_in_window"])
                        ),
                        "X-RateLimit-Reset": str(int(datetime.utcnow().timestamp()) + retry_after)
                    }
                )

            # Check daily limit
            cur.execute(
                "SELECT * FROM auth.check_rate_limit(%s, 24)",
                (api_key_id,)
            )
            daily = cur.fetchone()

            if daily["is_limited"]:
                retry_after = 86400  # 24 hours in seconds
                logger.warning(
                    f"Daily rate limit exceeded for API key ID {api_key_id}: "
                    f"{daily['requests_in_window']}/{daily['limit_per_window']}"
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=(
                        f"Daily rate limit exceeded. "
                        f"Used {daily['requests_in_window']}/{daily['limit_per_window']} requests. "
                        f"Limit resets in {retry_after} seconds."
                    ),
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(daily["limit_per_window"]),
                        "X-RateLimit-Remaining": str(
                            max(0, daily["limit_per_window"] - daily["requests_in_window"])
                        ),
                        "X-RateLimit-Reset": str(int(datetime.utcnow().timestamp()) + retry_after)
                    }
                )

    async def log_usage(
        self,
        api_key_id: int,
        request_id: uuid.UUID,
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: float,
        ip_address: str | None = None,
        user_agent: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None
    ):
        """
        Log API key usage

        Args:
            api_key_id: API key ID
            request_id: Request UUID for correlation
            endpoint: Endpoint accessed
            method: HTTP method
            status_code: HTTP status code
            response_time_ms: Response time in milliseconds
            ip_address: Client IP address
            user_agent: Client user agent
            error_type: Error type if applicable
            error_message: Error message if applicable
        """
        conn = self._get_connection()

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT auth.log_api_key_usage(
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """,
                    (
                        api_key_id, request_id, endpoint, method, status_code,
                        response_time_ms, ip_address, user_agent,
                        error_type, error_message
                    )
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to log API usage: {e}", exc_info=True)
            # Don't fail the request if logging fails
            pass

    @staticmethod
    def generate_api_key() -> str:
        """
        Generate a new API key

        Returns:
            64-character API key (UUID4 + timestamp hash)
        """
        # Generate UUID4
        key_uuid = uuid.uuid4().hex

        # Add timestamp for uniqueness
        timestamp = datetime.utcnow().isoformat()
        hash_suffix = hashlib.sha256(timestamp.encode()).hexdigest()[:32]

        # Combine (64 characters total)
        return f"{key_uuid}{hash_suffix}"[:64]

    def create_api_key(
        self,
        key_name: str,
        owner_email: str,
        owner_name: str | None = None,
        rate_limit_per_hour: int = 100,
        rate_limit_per_day: int = 1000,
        allowed_endpoints: list[str] | None = None,
        expires_in_days: int | None = None
    ) -> dict[str, Any]:
        """
        Create a new API key

        Args:
            key_name: Descriptive name for key
            owner_email: Owner email address
            owner_name: Owner full name
            rate_limit_per_hour: Hourly rate limit
            rate_limit_per_day: Daily rate limit
            allowed_endpoints: List of allowed endpoints (None = all)
            expires_in_days: Days until expiration (None = never)

        Returns:
            Dictionary with api_key, key_id, and metadata
        """
        api_key = self.generate_api_key()
        key_prefix = api_key[:8]

        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        conn = self._get_connection()

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO auth.api_keys (
                        api_key, key_name, key_prefix, owner_email, owner_name,
                        rate_limit_per_hour, rate_limit_per_day,
                        allowed_endpoints, expires_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, created_at
                    """,
                    (
                        api_key, key_name, key_prefix, owner_email, owner_name,
                        rate_limit_per_hour, rate_limit_per_day,
                        allowed_endpoints, expires_at
                    )
                )
                result = cur.fetchone()
                conn.commit()

                logger.info(
                    f"Created API key: {key_name} for {owner_email} "
                    f"(ID: {result['id']})"
                )

                return {
                    "api_key": api_key,
                    "key_id": result["id"],
                    "key_name": key_name,
                    "key_prefix": key_prefix,
                    "owner_email": owner_email,
                    "rate_limit_per_hour": rate_limit_per_hour,
                    "rate_limit_per_day": rate_limit_per_day,
                    "created_at": result["created_at"],
                    "expires_at": expires_at
                }

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to create API key: {e}", exc_info=True)
            raise


# Global authenticator instance
_authenticator: APIKeyAuthenticator | None = None


def get_authenticator() -> APIKeyAuthenticator:
    """Get global authenticator instance (singleton)"""
    global _authenticator
    if _authenticator is None:
        _authenticator = APIKeyAuthenticator()
    return _authenticator


async def require_api_key(
    request: Request,
    api_key: str | None = Security(API_KEY_HEADER)
) -> APIKey:
    """
    FastAPI dependency for API key authentication

    Usage:
        @app.get("/protected")
        async def protected_endpoint(
            api_key: APIKey = Depends(require_api_key)
        ):
            # Endpoint code here
            pass

    Args:
        request: FastAPI request object
        api_key: API key from X-API-Key header

    Returns:
        Validated APIKey object

    Raises:
        HTTPException: If authentication fails
    """
    authenticator = get_authenticator()
    endpoint = request.url.path

    # Validate API key
    validated_key = await authenticator.validate_api_key(api_key, endpoint, request)

    return validated_key
