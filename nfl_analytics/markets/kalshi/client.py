"""
Kalshi prediction market REST client.

Implements the MarketClient ABC for the Kalshi exchange. Handles RSA-PSS
authentication, rate limiting, retries, and response parsing.

All monetary values in cents. Contract IDs are Kalshi tickers (e.g., "KXNFL-26-SPREAD-KC-BUF-3.5").

Usage:
    from nfl_analytics.markets.kalshi.client import KalshiClient
    from nfl_analytics.markets.kalshi.auth import KalshiAuth

    auth = KalshiAuth.from_env()
    client = KalshiClient(auth=auth)

    # Or for demo environment:
    client = KalshiClient(auth=auth, demo=True)

    balance = await client.get_balance_cents()
    order = await client.place_order(order)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import httpx

from nfl_analytics.markets.base import (
    ConnectionError as MarketConnectionError,
    Contract,
    ContractNotFoundError,
    Fill,
    InsufficientBalanceError,
    MarketClient,
    MarketData,
    MarketError,
    Order,
    OrderBook,
    OrderRejectedError,
    OrderStatus,
    Position,
    RateLimitError,
)
from nfl_analytics.markets.kalshi.auth import KalshiAuth
from nfl_analytics.markets.kalshi.schemas import (
    build_batch_cancel_body,
    build_batch_create_body,
    build_create_order_body,
    parse_contract,
    parse_fill,
    parse_market_data,
    parse_order,
    parse_orderbook,
    parse_position,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROD_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
DEMO_BASE_URL = "https://demo-api.kalshi.co/trade-api/v2"

# Rate limits (Basic tier)
DEFAULT_WRITE_RATE = 10  # writes per second
DEFAULT_READ_RATE = 20  # reads per second

MAX_BATCH_SIZE = 20  # Kalshi batch order limit
MAX_RETRIES = 3
INITIAL_BACKOFF_MS = 100


# ---------------------------------------------------------------------------
# Token bucket rate limiter
# ---------------------------------------------------------------------------


class _TokenBucket:
    """Token bucket rate limiter.

    Allows bursts up to `capacity` then enforces `rate` tokens/second.
    """

    def __init__(self, rate: float, capacity: int | None = None):
        self.rate = rate
        self.capacity = capacity or int(rate)
        self._tokens = float(self.capacity)
        self._last_refill = time.monotonic()

    async def acquire(self, tokens: int = 1) -> None:
        """Wait until `tokens` are available, then consume them."""
        while True:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return
            # Sleep until enough tokens accumulate
            needed = tokens - self._tokens
            wait = needed / self.rate
            await asyncio.sleep(wait)

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_refill = now


# ---------------------------------------------------------------------------
# Client configuration
# ---------------------------------------------------------------------------


@dataclass
class KalshiClientConfig:
    """Configuration for the Kalshi REST client."""

    write_rate_per_sec: int = DEFAULT_WRITE_RATE
    read_rate_per_sec: int = DEFAULT_READ_RATE
    max_retries: int = MAX_RETRIES
    initial_backoff_ms: int = INITIAL_BACKOFF_MS
    timeout_seconds: float = 30.0
    max_positions_per_page: int = 200
    max_fills_per_page: int = 200


# ---------------------------------------------------------------------------
# KalshiClient
# ---------------------------------------------------------------------------


class KalshiClient(MarketClient):
    """Kalshi REST API client implementing the MarketClient interface.

    Features:
        - RSA-PSS authenticated requests
        - Token bucket rate limiting (separate read/write)
        - Exponential backoff retry on 429/5xx
        - Batch order placement (up to 20 orders)
        - Demo mode for testing without real money
    """

    def __init__(
        self,
        auth: KalshiAuth,
        demo: bool = False,
        config: KalshiClientConfig | None = None,
    ):
        self._auth = auth
        self._demo = demo
        self._config = config or KalshiClientConfig()
        self._base_url = DEMO_BASE_URL if demo else PROD_BASE_URL

        # Rate limiters
        self._read_limiter = _TokenBucket(self._config.read_rate_per_sec)
        self._write_limiter = _TokenBucket(self._config.write_rate_per_sec)

        # httpx async client with connection pooling
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._config.timeout_seconds),
            http2=True,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

        # Track order strategy names (not in Kalshi response)
        self._order_strategies: dict[str, str] = {}

        logger.info(
            "KalshiClient initialized (demo=%s, base=%s)",
            demo,
            self._base_url,
        )

    async def close(self) -> None:
        """Close the HTTP client connection pool."""
        await self._http.aclose()

    # -------------------------------------------------------------------
    # Internal HTTP helpers
    # -------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        params: dict | None = None,
        is_write: bool = False,
    ) -> dict:
        """Make an authenticated request with rate limiting and retry."""
        limiter = self._write_limiter if is_write else self._read_limiter
        await limiter.acquire()

        url = path  # httpx uses base_url prefix

        for attempt in range(self._config.max_retries + 1):
            # Generate fresh auth headers each attempt (timestamp changes)
            full_path = f"/trade-api/v2{path}"
            headers = self._auth.sign(method, full_path)
            if json_body is not None:
                headers["Content-Type"] = "application/json"

            try:
                response = await self._http.request(
                    method,
                    url,
                    json=json_body,
                    params=params,
                    headers=headers,
                )
            except httpx.ConnectError as e:
                if attempt < self._config.max_retries:
                    backoff = self._backoff_ms(attempt)
                    logger.warning("Connection error (attempt %d), retrying in %dms: %s", attempt + 1, backoff, e)
                    await asyncio.sleep(backoff / 1000)
                    continue
                raise MarketConnectionError(f"Failed to connect after {self._config.max_retries + 1} attempts: {e}") from e
            except httpx.TimeoutException as e:
                if attempt < self._config.max_retries:
                    backoff = self._backoff_ms(attempt)
                    logger.warning("Timeout (attempt %d), retrying in %dms", attempt + 1, backoff)
                    await asyncio.sleep(backoff / 1000)
                    continue
                raise MarketConnectionError(f"Request timed out: {e}") from e

            # Handle response
            if response.status_code == 429:
                if attempt < self._config.max_retries:
                    backoff = self._backoff_ms(attempt, base_ms=500)
                    logger.warning("Rate limited (attempt %d), backing off %dms", attempt + 1, backoff)
                    await asyncio.sleep(backoff / 1000)
                    continue
                raise RateLimitError(retry_after_ms=1000)

            if response.status_code >= 500:
                if attempt < self._config.max_retries:
                    backoff = self._backoff_ms(attempt)
                    logger.warning("Server error %d (attempt %d), retrying in %dms", response.status_code, attempt + 1, backoff)
                    await asyncio.sleep(backoff / 1000)
                    continue
                raise MarketError(f"Server error {response.status_code}: {response.text}")

            if response.status_code == 404:
                raise ContractNotFoundError(f"Not found: {path}")

            if response.status_code in (400, 403):
                error_data = response.json() if response.content else {}
                error_msg = error_data.get("message", response.text)
                code = error_data.get("code", "")

                if "insufficient" in error_msg.lower() or "balance" in error_msg.lower():
                    raise InsufficientBalanceError(error_msg)
                raise OrderRejectedError(f"[{code}] {error_msg}")

            if response.status_code in (200, 201):
                return response.json() if response.content else {}

            raise MarketError(f"Unexpected status {response.status_code}: {response.text}")

        raise MarketError("Exhausted retries")

    def _backoff_ms(self, attempt: int, base_ms: int | None = None) -> int:
        """Exponential backoff with jitter."""
        import random
        base = base_ms or self._config.initial_backoff_ms
        delay = base * (2 ** attempt)
        jitter = random.randint(0, delay // 2)
        return delay + jitter

    # -------------------------------------------------------------------
    # MarketClient ABC — Orders
    # -------------------------------------------------------------------

    async def place_order(self, order: Order) -> Order:
        """Submit a limit order to Kalshi.

        Returns the order with exchange_order_id and updated status.
        """
        self._order_strategies[order.order_id] = order.strategy

        body = build_create_order_body(order)
        data = await self._request("POST", "/portfolio/orders", json_body=body, is_write=True)

        result = parse_order(data.get("order", data))
        result.strategy = order.strategy
        return result

    async def cancel_order(self, order_id: str) -> Order:
        """Cancel a resting order on Kalshi."""
        data = await self._request("DELETE", f"/portfolio/orders/{order_id}", is_write=True)

        result = parse_order(data.get("order", data))
        result.strategy = self._order_strategies.get(order_id, "")
        return result

    async def get_order(self, order_id: str) -> Order:
        """Get current state of an order."""
        data = await self._request("GET", f"/portfolio/orders/{order_id}")

        result = parse_order(data.get("order", data))
        result.strategy = self._order_strategies.get(order_id, "")
        return result

    # -------------------------------------------------------------------
    # MarketClient ABC — Portfolio
    # -------------------------------------------------------------------

    async def get_positions(self) -> list[Position]:
        """Get all open positions with pagination."""
        positions: list[Position] = []
        cursor: str | None = None

        while True:
            params: dict = {"limit": self._config.max_positions_per_page}
            if cursor:
                params["cursor"] = cursor

            data = await self._request("GET", "/portfolio/positions", params=params)

            for mp in data.get("market_positions", []):
                pos = parse_position(mp)
                if pos.quantity > 0:
                    positions.append(pos)

            cursor = data.get("cursor")
            if not cursor or not data.get("market_positions"):
                break

        return positions

    async def get_balance_cents(self) -> int:
        """Get account cash balance in cents."""
        data = await self._request("GET", "/portfolio/balance")
        return data.get("balance", 0)

    async def get_fills(self, order_id: str | None = None) -> list[Fill]:
        """Get fills, optionally filtered by order_id, with pagination."""
        fills: list[Fill] = []
        cursor: str | None = None

        while True:
            params: dict = {"limit": self._config.max_fills_per_page}
            if order_id:
                params["order_id"] = order_id
            if cursor:
                params["cursor"] = cursor

            data = await self._request("GET", "/portfolio/fills", params=params)

            for f in data.get("fills", []):
                fills.append(parse_fill(f))

            cursor = data.get("cursor")
            if not cursor or not data.get("fills"):
                break

        return fills

    # -------------------------------------------------------------------
    # MarketClient ABC — Market data
    # -------------------------------------------------------------------

    async def get_orderbook(self, contract_id: str) -> OrderBook:
        """Get the current order book for a contract."""
        data = await self._request("GET", f"/markets/{contract_id}/orderbook")
        return parse_orderbook(data, contract_id)

    async def get_market_data(self, contract_id: str) -> MarketData:
        """Get latest market data for a contract."""
        data = await self._request("GET", f"/markets/{contract_id}")
        market = data.get("market", data)
        return parse_market_data(market)

    async def get_contract(self, contract_id: str) -> Contract:
        """Get contract metadata."""
        data = await self._request("GET", f"/markets/{contract_id}")
        market = data.get("market", data)
        return parse_contract(market)

    # -------------------------------------------------------------------
    # Batch operations (Kalshi-specific overrides)
    # -------------------------------------------------------------------

    async def batch_place_orders(self, orders: list[Order]) -> list[Order]:
        """Place up to 20 orders in a single API call.

        Kalshi's batch endpoint allows each order to independently succeed
        or fail. Returns all results.
        """
        if len(orders) > MAX_BATCH_SIZE:
            # Split into chunks
            results: list[Order] = []
            for i in range(0, len(orders), MAX_BATCH_SIZE):
                chunk = orders[i : i + MAX_BATCH_SIZE]
                results.extend(await self.batch_place_orders(chunk))
            return results

        # Track strategies
        for o in orders:
            self._order_strategies[o.order_id] = o.strategy

        body = build_batch_create_body(orders)
        data = await self._request("POST", "/portfolio/orders/batched", json_body=body, is_write=True)

        results = []
        for i, item in enumerate(data.get("orders", [])):
            if item.get("error"):
                logger.warning("Batch order %d failed: %s", i, item["error"])
                order = orders[i]
                order.status = OrderStatus.REJECTED
                results.append(order)
            else:
                order_data = item.get("order", item)
                result = parse_order(order_data)
                result.strategy = orders[i].strategy if i < len(orders) else ""
                results.append(result)

        return results

    async def cancel_all_orders(self, contract_id: str | None = None) -> int:
        """Cancel all open orders, optionally filtered by contract.

        Uses Kalshi's batch cancel for efficiency (0.2 tx per cancel).
        """
        # First get all open orders
        params: dict = {"status": "resting"}
        if contract_id:
            params["ticker"] = contract_id

        data = await self._request("GET", "/portfolio/orders", params=params)
        order_ids = [o["order_id"] for o in data.get("orders", [])]

        if not order_ids:
            return 0

        # Batch cancel in chunks of 20
        cancelled = 0
        for i in range(0, len(order_ids), MAX_BATCH_SIZE):
            chunk = order_ids[i : i + MAX_BATCH_SIZE]
            body = build_batch_cancel_body(chunk)
            await self._request("DELETE", "/portfolio/orders/batched", json_body=body, is_write=True)
            cancelled += len(chunk)

        logger.info("Cancelled %d orders%s", cancelled, f" for {contract_id}" if contract_id else "")
        return cancelled

    # -------------------------------------------------------------------
    # Exchange status
    # -------------------------------------------------------------------

    async def get_exchange_status(self) -> dict:
        """Check if the Kalshi exchange is active and trading."""
        return await self._request("GET", "/exchange/status")

    # -------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------

    @property
    def is_demo(self) -> bool:
        return self._demo

    @property
    def base_url(self) -> str:
        return self._base_url
