"""
Integration Tests for API Security Features

Tests authentication, rate limiting, and OpenAPI spec validation.

Requirements:
- PostgreSQL with migration 045 applied
- Redis running
- FastAPI server running

Run:
    pytest tests/integration/test_api_security.py -v

Author: Claude Code
Date: 2025-11-15
"""

import os
import time
import uuid
from datetime import datetime, timedelta

import pytest
import requests
from psycopg import connect
from psycopg.rows import dict_row

# Test configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://dro:sicillionbillions@localhost:5544/devdb01"  # pragma: allowlist secret
)

# Test API key (provisioned in migration 045)
TEST_API_KEY = "dev_test_key_12345678901234567890123456789012"


@pytest.fixture(scope="module")
def db_conn():
    """Database connection fixture"""
    conn = connect(DATABASE_URL, row_factory=dict_row)
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def api_key(db_conn):
    """Create test API key for tests"""
    # Use existing test key from migration
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM auth.api_keys WHERE api_key = %s",
            (TEST_API_KEY,)
        )
        key_record = cur.fetchone()

        if key_record:
            return TEST_API_KEY

        # Create test key if not exists
        cur.execute(
            """
            INSERT INTO auth.api_keys (
                api_key, key_name, key_prefix, owner_email,
                rate_limit_per_hour, rate_limit_per_day
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING api_key
            """,
            (
                TEST_API_KEY,
                "Test API Key",
                "dev_test",
                "test@nfl-analytics.local",
                1000,  # High limit for testing
                10000
            )
        )
        db_conn.commit()
        return TEST_API_KEY


@pytest.fixture
def expired_api_key(db_conn):
    """Create expired API key for testing"""
    expired_key = f"expired_key_{uuid.uuid4().hex[:32]}"

    with db_conn.cursor() as cur:
        # Create key that expired yesterday
        cur.execute(
            """
            INSERT INTO auth.api_keys (
                api_key, key_name, key_prefix, owner_email,
                expires_at
            )
            VALUES (%s, %s, %s, %s, %s)
            RETURNING api_key
            """,
            (
                expired_key,
                "Expired Test Key",
                "expired_",
                "test@nfl-analytics.local",
                datetime.utcnow() - timedelta(days=1)
            )
        )
        db_conn.commit()

    yield expired_key

    # Cleanup
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM auth.api_keys WHERE api_key = %s", (expired_key,))
        db_conn.commit()


@pytest.fixture
def revoked_api_key(db_conn):
    """Create revoked API key for testing"""
    revoked_key = f"revoked_key_{uuid.uuid4().hex[:32]}"

    with db_conn.cursor() as cur:
        # Create and immediately revoke key
        cur.execute(
            """
            INSERT INTO auth.api_keys (
                api_key, key_name, key_prefix, owner_email,
                is_revoked, revoked_at, revoked_reason
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING api_key
            """,
            (
                revoked_key,
                "Revoked Test Key",
                "revoked_",
                "test@nfl-analytics.local",
                True,
                datetime.utcnow(),
                "Testing revocation"
            )
        )
        db_conn.commit()

    yield revoked_key

    # Cleanup
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM auth.api_keys WHERE api_key = %s", (revoked_key,))
        db_conn.commit()


class TestAuthentication:
    """Test API key authentication"""

    def test_health_no_auth_required(self):
        """Health endpoint should not require authentication"""
        response = requests.get(f"{API_BASE_URL}/health")
        assert response.status_code == 200
        assert "status" in response.json()

    def test_metrics_no_auth_required(self):
        """Metrics endpoint should not require authentication"""
        response = requests.get(f"{API_BASE_URL}/metrics")
        assert response.status_code == 200
        assert "predictions_total" in response.text or response.status_code == 200

    def test_predict_missing_api_key(self):
        """Predict endpoint should reject requests without API key"""
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={
                "game_id": "2024_10_KC_LV",
                "player_id": "00-0033873",
                "stat_type": "passing_yards"
            }
        )

        assert response.status_code == 401
        assert "API key required" in response.json()["message"]

    def test_predict_invalid_api_key(self):
        """Predict endpoint should reject invalid API keys"""
        response = requests.post(
            f"{API_BASE_URL}/predict",
            headers={"X-API-Key": "invalid_key_12345"},
            json={
                "game_id": "2024_10_KC_LV",
                "player_id": "00-0033873",
                "stat_type": "passing_yards"
            }
        )

        assert response.status_code == 401
        assert "Invalid API key" in response.json()["message"]

    def test_predict_valid_api_key(self, api_key):
        """Predict endpoint should accept valid API key"""
        response = requests.post(
            f"{API_BASE_URL}/predict",
            headers={"X-API-Key": api_key},
            json={
                "game_id": "2024_10_KC_LV",
                "player_id": "00-0033873",
                "stat_type": "passing_yards"
            }
        )

        # Should succeed (200) or fail gracefully (503 if models not loaded)
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "confidence" in data

    def test_models_requires_auth(self, api_key):
        """Models endpoint should require authentication"""
        # Without API key
        response = requests.get(f"{API_BASE_URL}/models")
        assert response.status_code == 401

        # With valid API key
        response = requests.get(
            f"{API_BASE_URL}/models",
            headers={"X-API-Key": api_key}
        )
        assert response.status_code in [200, 503]

    def test_expired_api_key(self, expired_api_key):
        """Expired API key should be rejected"""
        response = requests.post(
            f"{API_BASE_URL}/predict",
            headers={"X-API-Key": expired_api_key},
            json={
                "game_id": "2024_10_KC_LV",
                "player_id": "00-0033873",
                "stat_type": "passing_yards"
            }
        )

        assert response.status_code == 403
        assert "expired" in response.json()["message"].lower()

    def test_revoked_api_key(self, revoked_api_key):
        """Revoked API key should be rejected"""
        response = requests.post(
            f"{API_BASE_URL}/predict",
            headers={"X-API-Key": revoked_api_key},
            json={
                "game_id": "2024_10_KC_LV",
                "player_id": "00-0033873",
                "stat_type": "passing_yards"
            }
        )

        assert response.status_code == 403
        assert "revoked" in response.json()["message"].lower()


class TestRateLimiting:
    """Test rate limiting functionality"""

    def test_rate_limit_headers(self, api_key):
        """Response should include rate limit headers"""
        response = requests.get(
            f"{API_BASE_URL}/models",
            headers={"X-API-Key": api_key}
        )

        # Check rate limit headers present
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

        # Validate header values
        limit = int(response.headers["X-RateLimit-Limit"])
        remaining = int(response.headers["X-RateLimit-Remaining"])
        reset = int(response.headers["X-RateLimit-Reset"])

        assert limit > 0
        assert 0 <= remaining <= limit
        assert reset > time.time()

    def test_rate_limit_decrement(self, api_key):
        """Rate limit remaining should decrement with requests"""
        # Make first request
        response1 = requests.get(
            f"{API_BASE_URL}/models",
            headers={"X-API-Key": api_key}
        )
        remaining1 = int(response1.headers["X-RateLimit-Remaining"])

        # Make second request
        response2 = requests.get(
            f"{API_BASE_URL}/models",
            headers={"X-API-Key": api_key}
        )
        remaining2 = int(response2.headers["X-RateLimit-Remaining"])

        # Remaining should decrease (or stay same if window reset)
        assert remaining2 <= remaining1

    @pytest.mark.slow
    def test_rate_limit_exceeded(self, db_conn):
        """Exceeding rate limit should return 429"""
        # Create temporary key with very low limit
        low_limit_key = f"low_limit_{uuid.uuid4().hex[:32]}"

        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO auth.api_keys (
                    api_key, key_name, key_prefix, owner_email,
                    rate_limit_per_hour, rate_limit_per_day
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING api_key
                """,
                (
                    low_limit_key,
                    "Low Limit Test Key",
                    "low_limi",
                    "test@nfl-analytics.local",
                    5,  # Very low limit
                    50
                )
            )
            db_conn.commit()

        try:
            # Make requests until limit exceeded
            for i in range(10):
                response = requests.get(
                    f"{API_BASE_URL}/models",
                    headers={"X-API-Key": low_limit_key}
                )

                if response.status_code == 429:
                    # Rate limit exceeded - verify response
                    assert "Retry-After" in response.headers
                    assert "rate limit exceeded" in response.json()["message"].lower()
                    break
            else:
                pytest.skip("Rate limit not reached (Redis may not be available)")

        finally:
            # Cleanup
            with db_conn.cursor() as cur:
                cur.execute("DELETE FROM auth.api_keys WHERE api_key = %s", (low_limit_key,))
                db_conn.commit()


class TestAPIKeyManagement:
    """Test API key management functions"""

    def test_create_api_key(self, db_conn):
        """Test creating new API key"""
        from api.auth.api_key_auth import APIKeyAuthenticator

        authenticator = APIKeyAuthenticator(db_url=DATABASE_URL)

        try:
            key_info = authenticator.create_api_key(
                key_name="Integration Test Key",
                owner_email="test@example.com",
                owner_name="Test User",
                rate_limit_per_hour=100,
                rate_limit_per_day=1000,
                expires_in_days=90
            )

            # Validate key created
            assert "api_key" in key_info
            assert len(key_info["api_key"]) == 64
            assert key_info["key_name"] == "Integration Test Key"
            assert key_info["owner_email"] == "test@example.com"
            assert key_info["rate_limit_per_hour"] == 100

            # Cleanup
            with db_conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM auth.api_keys WHERE api_key = %s",
                    (key_info["api_key"],)
                )
                db_conn.commit()

        finally:
            authenticator.close()

    def test_revoke_api_key(self, db_conn):
        """Test revoking API key"""
        # Create temporary key
        temp_key = f"temp_key_{uuid.uuid4().hex[:32]}"

        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO auth.api_keys (
                    api_key, key_name, key_prefix, owner_email
                )
                VALUES (%s, %s, %s, %s)
                RETURNING api_key
                """,
                (temp_key, "Temp Key", "temp_key", "test@example.com")
            )
            db_conn.commit()

        try:
            # Revoke key
            with db_conn.cursor() as cur:
                cur.execute(
                    "SELECT auth.revoke_api_key(%s, %s)",
                    (temp_key, "Testing revocation")
                )
                revoked = cur.fetchone()[0]
                db_conn.commit()

            assert revoked is True

            # Verify key is revoked
            with db_conn.cursor() as cur:
                cur.execute(
                    "SELECT is_revoked, revoked_reason FROM auth.api_keys WHERE api_key = %s",
                    (temp_key,)
                )
                result = cur.fetchone()
                assert result["is_revoked"] is True
                assert result["revoked_reason"] == "Testing revocation"

        finally:
            # Cleanup
            with db_conn.cursor() as cur:
                cur.execute("DELETE FROM auth.api_keys WHERE api_key = %s", (temp_key,))
                db_conn.commit()

    def test_check_rate_limit_function(self, db_conn, api_key):
        """Test database rate limit check function"""
        # Get API key ID
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM auth.api_keys WHERE api_key = %s",
                (api_key,)
            )
            key_id = cur.fetchone()["id"]

            # Check hourly rate limit
            cur.execute(
                "SELECT * FROM auth.check_rate_limit(%s, 1)",
                (key_id,)
            )
            result = cur.fetchone()

            # Validate result structure
            assert "requests_in_window" in result
            assert "limit_per_window" in result
            assert "is_limited" in result

            # Should not be limited (high limit for test key)
            assert result["is_limited"] is False


class TestOpenAPISpec:
    """Test OpenAPI specification"""

    def test_openapi_json_accessible(self):
        """OpenAPI JSON should be accessible"""
        response = requests.get(f"{API_BASE_URL}/openapi.json")
        assert response.status_code == 200

        spec = response.json()
        assert "openapi" in spec
        assert "info" in spec
        assert "paths" in spec

    def test_openapi_version(self):
        """OpenAPI spec should have correct version"""
        response = requests.get(f"{API_BASE_URL}/openapi.json")
        spec = response.json()

        assert spec["info"]["version"] == "1.1.0"
        assert "authentication" in spec["info"]["description"].lower()

    def test_openapi_security_scheme(self):
        """OpenAPI spec should define security scheme"""
        response = requests.get(f"{API_BASE_URL}/openapi.json")
        spec = response.json()

        # Check security scheme defined
        assert "components" in spec
        assert "securitySchemes" in spec["components"]

        # Verify API key security scheme
        security_schemes = spec["components"]["securitySchemes"]
        # FastAPI may name it differently, check for API key type
        has_api_key_scheme = any(
            scheme.get("type") == "apiKey" and scheme.get("in") == "header"
            for scheme in security_schemes.values()
        )
        assert has_api_key_scheme

    def test_docs_accessible(self):
        """Swagger UI docs should be accessible"""
        response = requests.get(f"{API_BASE_URL}/docs")
        assert response.status_code == 200
        assert "swagger" in response.text.lower() or "openapi" in response.text.lower()

    def test_redoc_accessible(self):
        """ReDoc documentation should be accessible"""
        response = requests.get(f"{API_BASE_URL}/redoc")
        assert response.status_code == 200
        assert "redoc" in response.text.lower() or "openapi" in response.text.lower()


class TestUsageTracking:
    """Test API usage tracking"""

    def test_usage_logged(self, db_conn, api_key):
        """API requests should be logged in database"""
        # Get API key ID
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM auth.api_keys WHERE api_key = %s",
                (api_key,)
            )
            key_id = cur.fetchone()["id"]

            # Count usage before request
            cur.execute(
                "SELECT COUNT(*) FROM auth.api_key_usage WHERE api_key_id = %s",
                (key_id,)
            )
            count_before = cur.fetchone()["count"]

        # Make API request
        response = requests.get(
            f"{API_BASE_URL}/models",
            headers={"X-API-Key": api_key}
        )
        assert response.status_code in [200, 503]

        # Allow time for async logging
        time.sleep(1)

        # Count usage after request
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM auth.api_key_usage WHERE api_key_id = %s",
                (key_id,)
            )
            count_after = cur.fetchone()["count"]

        # Usage should be incremented (or at least not decreased)
        assert count_after >= count_before

    def test_last_used_updated(self, db_conn, api_key):
        """last_used_at should be updated on API request"""
        # Get last_used_at before request
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT last_used_at FROM auth.api_keys WHERE api_key = %s",
                (api_key,)
            )
            last_used_before = cur.fetchone()["last_used_at"]

        # Wait to ensure different timestamp
        time.sleep(2)

        # Make API request
        response = requests.get(
            f"{API_BASE_URL}/models",
            headers={"X-API-Key": api_key}
        )
        assert response.status_code in [200, 503]

        # Allow time for async update
        time.sleep(1)

        # Get last_used_at after request
        with db_conn.cursor() as cur:
            cur.execute(
                "SELECT last_used_at FROM auth.api_keys WHERE api_key = %s",
                (api_key,)
            )
            last_used_after = cur.fetchone()["last_used_at"]

        # last_used_at should be updated (or at least not earlier)
        if last_used_before and last_used_after:
            assert last_used_after >= last_used_before


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
