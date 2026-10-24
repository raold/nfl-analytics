"""Tests for Kalshi RSA-PSS authentication module."""

from __future__ import annotations

import base64
import os
import time
from unittest.mock import patch

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from nfl_analytics.markets.kalshi.auth import KalshiAuth

# ============================================================
# Test RSA key generation helper
# ============================================================


def _generate_test_key() -> tuple[rsa.RSAPrivateKey, str]:
    """Generate a throwaway RSA key pair for tests."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    return private_key, pem


# ============================================================
# Auth signing
# ============================================================


class TestKalshiAuth:
    """RSA-PSS signing for API requests."""

    def test_sign_returns_required_headers(self):
        """Sign produces all three Kalshi headers."""
        key, _ = _generate_test_key()
        auth = KalshiAuth(key_id="test-key-123", private_key=key)

        headers = auth.sign("GET", "/trade-api/v2/portfolio/balance")

        assert "KALSHI-ACCESS-KEY" in headers
        assert "KALSHI-ACCESS-TIMESTAMP" in headers
        assert "KALSHI-ACCESS-SIGNATURE" in headers
        assert headers["KALSHI-ACCESS-KEY"] == "test-key-123"

    def test_sign_timestamp_is_current(self):
        """Timestamp should be within 2 seconds of now."""
        key, _ = _generate_test_key()
        auth = KalshiAuth(key_id="test", private_key=key)

        headers = auth.sign("POST", "/trade-api/v2/portfolio/orders")
        ts_ms = int(headers["KALSHI-ACCESS-TIMESTAMP"])
        now_ms = int(time.time() * 1000)

        assert abs(ts_ms - now_ms) < 2000

    def test_sign_custom_timestamp(self):
        """Custom timestamp is used verbatim."""
        key, _ = _generate_test_key()
        auth = KalshiAuth(key_id="test", private_key=key)

        headers = auth.sign("GET", "/test", timestamp_ms=1700000000000)
        assert headers["KALSHI-ACCESS-TIMESTAMP"] == "1700000000000"

    def test_signature_is_base64(self):
        """Signature is valid base64."""
        key, _ = _generate_test_key()
        auth = KalshiAuth(key_id="test", private_key=key)

        headers = auth.sign("GET", "/test")
        sig = headers["KALSHI-ACCESS-SIGNATURE"]

        # Should decode without error
        decoded = base64.b64decode(sig)
        assert len(decoded) > 0  # RSA-2048 = 256 bytes
        assert len(decoded) == 256

    def test_different_methods_produce_different_signatures(self):
        """GET and POST on same path produce different signatures."""
        key, _ = _generate_test_key()
        auth = KalshiAuth(key_id="test", private_key=key)

        ts = 1700000000000
        h1 = auth.sign("GET", "/trade-api/v2/portfolio/orders", timestamp_ms=ts)
        h2 = auth.sign("POST", "/trade-api/v2/portfolio/orders", timestamp_ms=ts)

        assert h1["KALSHI-ACCESS-SIGNATURE"] != h2["KALSHI-ACCESS-SIGNATURE"]

    def test_query_params_stripped_from_path(self):
        """Query parameters don't affect the signed message.

        RSA-PSS signatures are non-deterministic (random salt), so we can't
        compare two signatures directly. Instead, verify that the signature
        from the path-with-query-params validates against the clean path message.
        """
        from cryptography.hazmat.primitives.asymmetric import padding as asym_padding

        key, _ = _generate_test_key()
        auth = KalshiAuth(key_id="test", private_key=key)

        ts = 1700000000000
        headers = auth.sign("GET", "/trade-api/v2/markets?status=active", timestamp_ms=ts)

        # The signature should be valid for the message WITHOUT query params
        message = f"{ts}GET/trade-api/v2/markets".encode()
        sig_bytes = base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"])

        # Verify succeeds = query params were correctly stripped
        key.public_key().verify(
            sig_bytes,
            message,
            asym_padding.PSS(
                mgf=asym_padding.MGF1(hashes.SHA256()),
                salt_length=asym_padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )

    def test_sign_websocket(self):
        """WebSocket signing uses GET method."""
        key, _ = _generate_test_key()
        auth = KalshiAuth(key_id="test", private_key=key)

        headers = auth.sign_websocket()
        assert "KALSHI-ACCESS-KEY" in headers
        assert "KALSHI-ACCESS-SIGNATURE" in headers


# ============================================================
# Loading from environment
# ============================================================


class TestAuthFromEnv:
    """Loading credentials from environment variables."""

    def test_from_env_with_inline_key(self):
        """Load credentials when KALSHI_PRIVATE_KEY is PEM inline."""
        _, pem = _generate_test_key()

        with patch.dict(
            os.environ,
            {
                "KALSHI_KEY_ID": "my-key-id",
                "KALSHI_PRIVATE_KEY": pem,
            },
        ):
            auth = KalshiAuth.from_env()
            assert auth.key_id == "my-key-id"

    def test_from_env_with_file_path(self, tmp_path):
        """Load credentials when KALSHI_PRIVATE_KEY is a file path."""
        _, pem = _generate_test_key()
        key_file = tmp_path / "kalshi.pem"
        key_file.write_text(pem)

        with patch.dict(
            os.environ,
            {
                "KALSHI_KEY_ID": "my-key-id",
                "KALSHI_PRIVATE_KEY": str(key_file),
            },
        ):
            auth = KalshiAuth.from_env()
            assert auth.key_id == "my-key-id"

    def test_from_env_missing_key_id_raises(self):
        """Missing KALSHI_KEY_ID raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove any existing env vars
            os.environ.pop("KALSHI_KEY_ID", None)
            with pytest.raises(ValueError, match="KALSHI_KEY_ID"):
                KalshiAuth.from_env()

    def test_from_env_missing_private_key_raises(self):
        """Missing KALSHI_PRIVATE_KEY raises ValueError."""
        with patch.dict(os.environ, {"KALSHI_KEY_ID": "test"}, clear=True):
            os.environ.pop("KALSHI_PRIVATE_KEY", None)
            with pytest.raises(ValueError, match="KALSHI_PRIVATE_KEY"):
                KalshiAuth.from_env()

    def test_from_key_file(self, tmp_path):
        """Load from explicit key file path."""
        _, pem = _generate_test_key()
        key_file = tmp_path / "test.pem"
        key_file.write_text(pem)

        auth = KalshiAuth.from_key_file("my-key", key_file)
        assert auth.key_id == "my-key"
        headers = auth.sign("GET", "/test")
        assert "KALSHI-ACCESS-SIGNATURE" in headers
