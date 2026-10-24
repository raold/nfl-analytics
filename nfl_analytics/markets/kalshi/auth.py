"""
RSA-PSS authentication for the Kalshi API.

Kalshi requires three headers on every authenticated request:
  KALSHI-ACCESS-KEY      — API key ID (from Kalshi dashboard)
  KALSHI-ACCESS-TIMESTAMP — Unix timestamp in milliseconds (string)
  KALSHI-ACCESS-SIGNATURE — Base64-encoded RSA-PSS signature

Signature construction:
  message = f"{timestamp_ms}{METHOD}{path}"   (no query params, no body)
  Sign with RSA-PSS, SHA-256, salt_length=DIGEST_LENGTH (32 bytes)
  Base64-encode the result

Usage:
    from py.markets.kalshi.auth import KalshiAuth

    auth = KalshiAuth.from_env()  # Reads KALSHI_KEY_ID + KALSHI_PRIVATE_KEY
    headers = auth.sign("POST", "/trade-api/v2/portfolio/orders")
"""

from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KalshiAuth:
    """Generates Kalshi API authentication headers via RSA-PSS signing.

    Attributes:
        key_id: Kalshi API key ID (displayed in dashboard).
        private_key: RSA private key (loaded from PEM).
    """

    key_id: str
    private_key: rsa.RSAPrivateKey

    @classmethod
    def from_env(cls) -> KalshiAuth:
        """Load credentials from environment variables.

        Expects:
            KALSHI_KEY_ID       — API key ID
            KALSHI_PRIVATE_KEY  — PEM-encoded private key (inline or file path)
        """
        key_id = os.environ.get("KALSHI_KEY_ID", "")
        if not key_id:
            raise ValueError("KALSHI_KEY_ID environment variable not set")

        key_data = os.environ.get("KALSHI_PRIVATE_KEY", "")
        if not key_data:
            raise ValueError("KALSHI_PRIVATE_KEY environment variable not set")

        # If it looks like a file path, read from file
        if not key_data.startswith("-----") and Path(key_data).is_file():
            key_data = Path(key_data).read_text()

        private_key = serialization.load_pem_private_key(
            key_data.encode(), password=None
        )
        if not isinstance(private_key, rsa.RSAPrivateKey):
            raise TypeError(f"Expected RSA private key, got {type(private_key).__name__}")

        logger.info("Loaded Kalshi API key: %s...", key_id[:8])
        return cls(key_id=key_id, private_key=private_key)

    @classmethod
    def from_key_file(cls, key_id: str, key_path: str | Path) -> KalshiAuth:
        """Load credentials from a PEM key file."""
        key_data = Path(key_path).read_text()
        private_key = serialization.load_pem_private_key(
            key_data.encode(), password=None
        )
        if not isinstance(private_key, rsa.RSAPrivateKey):
            raise TypeError(f"Expected RSA private key, got {type(private_key).__name__}")
        return cls(key_id=key_id, private_key=private_key)

    def sign(self, method: str, path: str, timestamp_ms: int | None = None) -> dict[str, str]:
        """Generate authentication headers for a Kalshi API request.

        Args:
            method: HTTP method (GET, POST, DELETE, etc.).
            path: Request path without query params (e.g., "/trade-api/v2/portfolio/orders").
            timestamp_ms: Unix timestamp in milliseconds. Auto-generated if None.

        Returns:
            Dict of headers to include in the request.
        """
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        ts_str = str(timestamp_ms)

        # Strip query parameters from path before signing
        clean_path = path.split("?")[0]

        # message = timestamp + METHOD + path (no body, no query params)
        message = f"{ts_str}{method.upper()}{clean_path}".encode()

        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,  # 32 bytes for SHA-256
            ),
            hashes.SHA256(),
        )

        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts_str,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
        }

    def sign_websocket(self, ws_path: str = "/trade-api/ws/v2") -> dict[str, str]:
        """Generate authentication headers for WebSocket upgrade.

        WebSocket auth uses GET method against the WebSocket path.
        """
        return self.sign("GET", ws_path)
