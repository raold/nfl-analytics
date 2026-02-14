"""Tests for Kalshi REST client with mocked HTTP responses."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from py.markets.base import (
    ContractNotFoundError,
    InsufficientBalanceError,
    MarketError,
    Order,
    OrderRejectedError,
    OrderStatus,
    RateLimitError,
    Side,
)
from py.markets.kalshi.auth import KalshiAuth
from py.markets.kalshi.client import KalshiClient, KalshiClientConfig

pytestmark = pytest.mark.anyio


# ============================================================
# Fixtures
# ============================================================


def _mock_auth() -> KalshiAuth:
    """Create a mock KalshiAuth that returns fake headers."""
    auth = MagicMock(spec=KalshiAuth)
    auth.sign.return_value = {
        "KALSHI-ACCESS-KEY": "test-key",
        "KALSHI-ACCESS-TIMESTAMP": "1700000000000",
        "KALSHI-ACCESS-SIGNATURE": "fake-signature",
    }
    return auth


def _mock_response(status_code: int = 200, json_data: dict | None = None) -> httpx.Response:
    """Create a mock httpx response."""
    return httpx.Response(
        status_code=status_code,
        json=json_data or {},
        request=httpx.Request("GET", "https://test.com"),
    )


class MockTransport(httpx.AsyncBaseTransport):
    """Mock httpx transport that returns pre-configured responses."""

    def __init__(self, responses: dict[str, httpx.Response] | None = None, default_response: httpx.Response | None = None):
        self._responses = responses or {}
        self._default = default_response or _mock_response()
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = f"{request.method} {request.url.path}"
        return self._responses.get(key, self._default)


@pytest.fixture
def client_and_transport():
    """Create a KalshiClient with mock transport."""
    auth = _mock_auth()
    transport = MockTransport()
    config = KalshiClientConfig(max_retries=1, initial_backoff_ms=10)

    client = KalshiClient(auth=auth, demo=True, config=config)
    # Replace the internal http client with one using mock transport
    client._http = httpx.AsyncClient(
        base_url="https://demo-api.kalshi.co/trade-api/v2",
        transport=transport,
        timeout=httpx.Timeout(5.0),
    )
    return client, transport


# ============================================================
# Place order
# ============================================================


class TestPlaceOrder:
    """Order placement via REST API."""

    async def test_place_order_success(self, client_and_transport):
        """Successful order placement."""
        client, transport = client_and_transport
        transport._default = _mock_response(201, {
            "order": {
                "order_id": "exchange-123",
                "client_order_id": "my-order",
                "ticker": "KXNFL-26-WIN-KC",
                "side": "yes",
                "status": "resting",
                "yes_price": 55,
                "initial_count": 10,
                "fill_count": 0,
                "taker_fill_cost": 0,
                "maker_fill_cost": 0,
            }
        })

        order = Order(
            order_id="my-order",
            contract_id="KXNFL-26-WIN-KC",
            side=Side.YES,
            price_cents=55,
            quantity=10,
            strategy="directional",
        )
        result = await client.place_order(order)

        assert result.status == OrderStatus.OPEN
        assert result.exchange_order_id == "exchange-123"
        assert result.strategy == "directional"

    async def test_place_order_immediately_filled(self, client_and_transport):
        """Order fills immediately (crosses spread)."""
        client, transport = client_and_transport
        transport._default = _mock_response(201, {
            "order": {
                "order_id": "exchange-456",
                "client_order_id": "my-order",
                "ticker": "TEST-01",
                "side": "yes",
                "status": "executed",
                "yes_price": 55,
                "initial_count": 5,
                "fill_count": 5,
                "taker_fill_cost": 275,
                "maker_fill_cost": 0,
            }
        })

        order = Order(
            order_id="my-order",
            contract_id="TEST-01",
            side=Side.YES,
            price_cents=55,
            quantity=5,
        )
        result = await client.place_order(order)

        assert result.status == OrderStatus.FILLED
        assert result.filled_quantity == 5


# ============================================================
# Cancel order
# ============================================================


class TestCancelOrder:
    """Order cancellation."""

    async def test_cancel_order(self, client_and_transport):
        """Cancel a resting order."""
        client, transport = client_and_transport
        transport._responses[f"DELETE /trade-api/v2/portfolio/orders/order-123"] = _mock_response(200, {
            "order": {
                "order_id": "order-123",
                "ticker": "TEST-01",
                "side": "yes",
                "status": "canceled",
                "yes_price": 50,
                "initial_count": 10,
                "fill_count": 0,
            }
        })

        result = await client.cancel_order("order-123")
        assert result.status == OrderStatus.CANCELLED


# ============================================================
# Get balance
# ============================================================


class TestGetBalance:
    """Account balance queries."""

    async def test_get_balance(self, client_and_transport):
        """Get balance in cents."""
        client, transport = client_and_transport
        transport._responses["GET /trade-api/v2/portfolio/balance"] = _mock_response(200, {
            "balance": 50000,
        })

        balance = await client.get_balance_cents()
        assert balance == 50000


# ============================================================
# Get positions
# ============================================================


class TestGetPositions:
    """Position queries with pagination."""

    async def test_get_positions(self, client_and_transport):
        """Get positions from single page."""
        client, transport = client_and_transport
        transport._default = _mock_response(200, {
            "market_positions": [
                {"ticker": "KXNFL-26-WIN-KC", "position": 10, "market_exposure": 550},
                {"ticker": "KXNFL-26-WIN-BUF", "position": -5, "market_exposure": 225},
            ],
            "cursor": "",
        })

        positions = await client.get_positions()

        assert len(positions) == 2
        assert positions[0].contract_id == "KXNFL-26-WIN-KC"
        assert positions[0].side == Side.YES
        assert positions[0].quantity == 10
        assert positions[1].side == Side.NO
        assert positions[1].quantity == 5


# ============================================================
# Get orderbook
# ============================================================


class TestGetOrderbook:
    """Orderbook queries."""

    async def test_get_orderbook(self, client_and_transport):
        """Get orderbook with levels."""
        client, transport = client_and_transport
        transport._default = _mock_response(200, {
            "orderbook": {
                "yes": [[50, 100], [55, 200]],
                "no": [[45, 150], [50, 300]],
            }
        })

        ob = await client.get_orderbook("TEST-01")

        assert ob.contract_id == "TEST-01"
        assert len(ob.yes_bids) == 2
        assert ob.yes_bids[0].price_cents == 55  # Best bid
        assert len(ob.yes_asks) == 2


# ============================================================
# Get market data
# ============================================================


class TestGetMarketData:
    """Market data queries."""

    async def test_get_market_data(self, client_and_transport):
        """Get market data for a contract."""
        client, transport = client_and_transport
        transport._default = _mock_response(200, {
            "market": {
                "ticker": "KXNFL-26-WIN-KC",
                "last_price": 55,
                "yes_bid": 54,
                "yes_ask": 56,
                "volume": 1000,
                "open_interest": 500,
            }
        })

        md = await client.get_market_data("KXNFL-26-WIN-KC")

        assert md.contract_id == "KXNFL-26-WIN-KC"
        assert md.last_price_cents == 55
        assert md.spread_cents == 2


# ============================================================
# Error handling
# ============================================================


class TestErrorHandling:
    """HTTP error → MarketError mapping."""

    async def test_404_raises_not_found(self, client_and_transport):
        """404 maps to ContractNotFoundError."""
        client, transport = client_and_transport
        transport._default = _mock_response(404, {"message": "Market not found"})

        with pytest.raises(ContractNotFoundError):
            await client.get_market_data("INVALID-TICKER")

    async def test_400_insufficient_balance(self, client_and_transport):
        """400 with balance message maps to InsufficientBalanceError."""
        client, transport = client_and_transport
        transport._default = _mock_response(400, {
            "message": "Insufficient balance to place order",
            "code": "insufficient_balance",
        })

        order = Order(
            order_id="test", contract_id="TEST-01",
            side=Side.YES, price_cents=50, quantity=1,
        )
        with pytest.raises(InsufficientBalanceError):
            await client.place_order(order)

    async def test_400_generic_rejection(self, client_and_transport):
        """400 without balance message maps to OrderRejectedError."""
        client, transport = client_and_transport
        transport._default = _mock_response(400, {
            "message": "Invalid order parameters",
            "code": "bad_request",
        })

        order = Order(
            order_id="test", contract_id="TEST-01",
            side=Side.YES, price_cents=50, quantity=1,
        )
        with pytest.raises(OrderRejectedError):
            await client.place_order(order)

    async def test_429_rate_limit_retries_then_raises(self, client_and_transport):
        """429 retries then raises RateLimitError."""
        client, transport = client_and_transport
        # All responses are 429
        transport._default = _mock_response(429, {"message": "Rate limited"})

        with pytest.raises(RateLimitError):
            await client.get_balance_cents()

        # Should have tried max_retries + 1 times
        assert len(transport.requests) == 2  # max_retries=1 → 2 attempts

    async def test_500_retries_then_raises(self, client_and_transport):
        """500 retries then raises MarketError."""
        client, transport = client_and_transport
        transport._default = _mock_response(500, {"message": "Internal error"})

        with pytest.raises(MarketError, match="Server error"):
            await client.get_balance_cents()


# ============================================================
# Batch operations
# ============================================================


class TestBatchOperations:
    """Batch order placement and cancellation."""

    async def test_batch_place_orders(self, client_and_transport):
        """Batch placement of multiple orders."""
        client, transport = client_and_transport
        transport._default = _mock_response(200, {
            "orders": [
                {
                    "order": {
                        "order_id": f"exchange-{i}",
                        "client_order_id": f"my-{i}",
                        "ticker": f"C-{i}",
                        "side": "yes",
                        "status": "resting",
                        "yes_price": 50,
                        "initial_count": 1,
                        "fill_count": 0,
                        "taker_fill_cost": 0,
                        "maker_fill_cost": 0,
                    }
                }
                for i in range(3)
            ]
        })

        orders = [
            Order(order_id=f"my-{i}", contract_id=f"C-{i}", side=Side.YES, price_cents=50, quantity=1, strategy="test")
            for i in range(3)
        ]
        results = await client.batch_place_orders(orders)

        assert len(results) == 3
        assert all(r.status == OrderStatus.OPEN for r in results)

    async def test_batch_with_partial_failure(self, client_and_transport):
        """Batch with one order failing."""
        client, transport = client_and_transport
        transport._default = _mock_response(200, {
            "orders": [
                {
                    "order": {
                        "order_id": "exchange-0",
                        "client_order_id": "my-0",
                        "ticker": "C-0",
                        "side": "yes",
                        "status": "resting",
                        "yes_price": 50,
                        "initial_count": 1,
                        "fill_count": 0,
                        "taker_fill_cost": 0,
                        "maker_fill_cost": 0,
                    }
                },
                {
                    "error": "insufficient_balance",
                },
            ]
        })

        orders = [
            Order(order_id="my-0", contract_id="C-0", side=Side.YES, price_cents=50, quantity=1),
            Order(order_id="my-1", contract_id="C-1", side=Side.YES, price_cents=50, quantity=1),
        ]
        results = await client.batch_place_orders(orders)

        assert len(results) == 2
        assert results[0].status == OrderStatus.OPEN
        assert results[1].status == OrderStatus.REJECTED


# ============================================================
# Client properties
# ============================================================


class TestClientProperties:
    """Client configuration and properties."""

    def test_demo_mode(self):
        """Demo mode uses demo base URL."""
        auth = _mock_auth()
        client = KalshiClient(auth=auth, demo=True)
        assert client.is_demo is True
        assert "demo" in client.base_url

    def test_prod_mode(self):
        """Production mode uses prod base URL."""
        auth = _mock_auth()
        client = KalshiClient(auth=auth, demo=False)
        assert client.is_demo is False
        assert "elections.kalshi.com" in client.base_url
