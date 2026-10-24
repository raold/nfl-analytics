"""Tests for Kalshi API response → internal dataclass conversion."""

from __future__ import annotations

import pytest

from nfl_analytics.markets.base import OrderStatus, Side
from nfl_analytics.markets.kalshi.schemas import (
    build_create_order_body,
    build_batch_create_body,
    build_batch_cancel_body,
    parse_contract,
    parse_fill,
    parse_market_data,
    parse_order,
    parse_orderbook,
    parse_position,
    parse_ticker_message,
)


# ============================================================
# Order parsing
# ============================================================


class TestParseOrder:
    """Convert Kalshi order JSON to our Order dataclass."""

    def test_resting_order(self):
        """Resting order maps to OPEN status."""
        data = {
            "order_id": "exchange-123",
            "client_order_id": "my-order-1",
            "ticker": "KXNFL-26-SPREAD-KC-BUF-3.5",
            "side": "yes",
            "action": "buy",
            "type": "limit",
            "status": "resting",
            "yes_price": 55,
            "no_price": 45,
            "initial_count": 10,
            "fill_count": 0,
            "remaining_count": 10,
            "taker_fill_cost": 0,
            "maker_fill_cost": 0,
            "created_time": "2024-01-01T00:00:00Z",
            "last_update_time": "2024-01-01T00:00:00Z",
        }
        order = parse_order(data)

        assert order.order_id == "my-order-1"  # Uses client_order_id
        assert order.exchange_order_id == "exchange-123"
        assert order.contract_id == "KXNFL-26-SPREAD-KC-BUF-3.5"
        assert order.side == Side.YES
        assert order.price_cents == 55
        assert order.quantity == 10
        assert order.filled_quantity == 0
        assert order.status == OrderStatus.OPEN

    def test_executed_order(self):
        """Executed order maps to FILLED status."""
        data = {
            "order_id": "exchange-456",
            "ticker": "KXNFL-26-WIN-KC",
            "side": "yes",
            "status": "executed",
            "yes_price": 60,
            "initial_count": 5,
            "fill_count": 5,
            "taker_fill_cost": 275,
            "maker_fill_cost": 25,
            "created_time": "2024-06-15T12:00:00Z",
        }
        order = parse_order(data)

        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 5
        assert order.avg_fill_price_cents == 60  # (275+25) / 5

    def test_canceled_order(self):
        """Canceled order maps to CANCELLED status."""
        data = {
            "order_id": "exchange-789",
            "ticker": "KXNFL-26-TOTAL-KC-BUF-47.5",
            "side": "no",
            "status": "canceled",
            "no_price": 42,
            "initial_count": 20,
            "fill_count": 0,
        }
        order = parse_order(data)

        assert order.status == OrderStatus.CANCELLED
        assert order.side == Side.NO
        assert order.price_cents == 42

    def test_partially_filled_resting(self):
        """Resting order with fills maps to PARTIAL."""
        data = {
            "order_id": "exchange-111",
            "ticker": "TEST-01",
            "side": "yes",
            "status": "resting",
            "yes_price": 50,
            "initial_count": 10,
            "fill_count": 3,
            "taker_fill_cost": 150,
            "maker_fill_cost": 0,
        }
        order = parse_order(data)

        assert order.status == OrderStatus.PARTIAL
        assert order.filled_quantity == 3
        assert order.quantity == 10

    def test_no_client_order_id_uses_exchange_id(self):
        """When no client_order_id, use exchange order_id."""
        data = {
            "order_id": "exchange-only",
            "ticker": "TEST-01",
            "side": "yes",
            "status": "resting",
            "yes_price": 50,
            "initial_count": 1,
            "fill_count": 0,
        }
        order = parse_order(data)
        assert order.order_id == "exchange-only"


# ============================================================
# Fill parsing
# ============================================================


class TestParseFill:
    """Convert Kalshi fill JSON to our Fill dataclass."""

    def test_basic_fill(self):
        """Standard fill with all fields."""
        data = {
            "fill_id": "fill-abc",
            "trade_id": "trade-123",
            "order_id": "order-456",
            "ticker": "KXNFL-26-WIN-KC",
            "side": "yes",
            "action": "buy",
            "count": 5,
            "yes_price": 55,
            "no_price": 45,
            "is_taker": True,
            "fee_cost": 10,
            "created_time": "2024-01-01T10:00:00Z",
        }
        fill = parse_fill(data)

        assert fill.fill_id == "fill-abc"
        assert fill.order_id == "order-456"
        assert fill.contract_id == "KXNFL-26-WIN-KC"
        assert fill.side == Side.YES
        assert fill.price_cents == 55
        assert fill.quantity == 5
        assert fill.fee_cents == 10

    def test_no_side_fill(self):
        """NO fill uses no_price."""
        data = {
            "fill_id": "fill-def",
            "order_id": "order-789",
            "ticker": "TEST-01",
            "side": "no",
            "count": 3,
            "yes_price": 60,
            "no_price": 40,
            "fee_cost": 5,
        }
        fill = parse_fill(data)

        assert fill.side == Side.NO
        assert fill.price_cents == 40
        assert fill.quantity == 3

    def test_fee_as_dollar_string(self):
        """Fee given as dollar string is converted to cents."""
        data = {
            "fill_id": "fill-ghi",
            "order_id": "order-abc",
            "ticker": "TEST-01",
            "side": "yes",
            "count": 1,
            "yes_price": 50,
            "fee_cost": "0.15",
        }
        fill = parse_fill(data)
        assert fill.fee_cents == 15

    def test_missing_fill_id_uses_trade_id(self):
        """Falls back to trade_id if fill_id not present."""
        data = {
            "trade_id": "trade-only",
            "order_id": "order-x",
            "ticker": "TEST-01",
            "side": "yes",
            "count": 1,
            "yes_price": 50,
        }
        fill = parse_fill(data)
        assert fill.fill_id == "trade-only"


# ============================================================
# Position parsing
# ============================================================


class TestParsePosition:
    """Convert Kalshi position JSON to our Position dataclass."""

    def test_positive_position_is_yes(self):
        """Positive position count = YES side."""
        data = {
            "ticker": "KXNFL-26-WIN-KC",
            "position": 10,
            "market_exposure": 550,
        }
        pos = parse_position(data)

        assert pos.contract_id == "KXNFL-26-WIN-KC"
        assert pos.side == Side.YES
        assert pos.quantity == 10
        assert pos.avg_entry_cents == 55  # 550 / 10

    def test_negative_position_is_no(self):
        """Negative position count = NO side."""
        data = {
            "ticker": "KXNFL-26-SPREAD-KC-BUF-3.5",
            "position": -5,
            "market_exposure": 225,
        }
        pos = parse_position(data)

        assert pos.side == Side.NO
        assert pos.quantity == 5
        assert pos.avg_entry_cents == 45  # 225 / 5

    def test_zero_position(self):
        """Zero position = no holdings."""
        data = {
            "ticker": "KXNFL-26-WIN-BUF",
            "position": 0,
            "market_exposure": 0,
        }
        pos = parse_position(data)
        assert pos.quantity == 0


# ============================================================
# OrderBook parsing
# ============================================================


class TestParseOrderbook:
    """Convert Kalshi orderbook JSON to our OrderBook."""

    def test_basic_orderbook(self):
        """Standard 2-level orderbook."""
        data = {
            "orderbook": {
                "yes": [[50, 100], [55, 200]],
                "no": [[45, 150], [50, 300]],
            }
        }
        ob = parse_orderbook(data, "TEST-01")

        assert ob.contract_id == "TEST-01"
        # YES bids sorted descending (55 first, then 50)
        assert len(ob.yes_bids) == 2
        assert ob.yes_bids[0].price_cents == 55
        assert ob.yes_bids[0].quantity == 200
        assert ob.yes_bids[1].price_cents == 50
        assert ob.yes_bids[1].quantity == 100

        # YES asks = 100 - NO bids (55 from no@45, 50 from no@50)
        assert len(ob.yes_asks) == 2
        # Sorted ascending: 50 first, then 55
        assert ob.yes_asks[0].price_cents == 50
        assert ob.yes_asks[1].price_cents == 55

    def test_empty_orderbook(self):
        """Empty orderbook has no levels."""
        data = {"orderbook": {"yes": [], "no": []}}
        ob = parse_orderbook(data, "TEST-01")

        assert len(ob.yes_bids) == 0
        assert len(ob.yes_asks) == 0
        assert ob.mid_price_cents is None

    def test_one_sided_book(self):
        """Only yes bids, no asks."""
        data = {"orderbook": {"yes": [[50, 100]], "no": []}}
        ob = parse_orderbook(data, "TEST-01")

        assert len(ob.yes_bids) == 1
        assert len(ob.yes_asks) == 0


# ============================================================
# Market data parsing
# ============================================================


class TestParseMarketData:
    """Convert Kalshi market JSON to MarketData."""

    def test_basic_market(self):
        """Standard market with all fields."""
        data = {
            "ticker": "KXNFL-26-WIN-KC",
            "last_price": 55,
            "yes_bid": 54,
            "yes_ask": 56,
            "volume": 1000,
            "open_interest": 500,
        }
        md = parse_market_data(data)

        assert md.contract_id == "KXNFL-26-WIN-KC"
        assert md.last_price_cents == 55
        assert md.yes_bid_cents == 54
        assert md.yes_ask_cents == 56
        assert md.volume_24h == 1000
        assert md.open_interest == 500

    def test_dollar_string_fallback(self):
        """Falls back to dollar string fields when cents are 0."""
        data = {
            "ticker": "TEST-01",
            "last_price": 0,
            "last_price_dollars": "0.5500",
            "yes_bid": 0,
            "yes_bid_dollars": "0.5400",
            "yes_ask": 0,
            "yes_ask_dollars": "0.5600",
            "volume": 100,
            "open_interest": 50,
        }
        md = parse_market_data(data)

        assert md.last_price_cents == 55
        assert md.yes_bid_cents == 54
        assert md.yes_ask_cents == 56

    def test_ticker_message_parsing(self):
        """Parse WebSocket ticker message."""
        msg = {
            "market_ticker": "KXNFL-26-SPREAD-KC-BUF-3.5",
            "yes_bid": 50,
            "yes_ask": 53,
            "price": 51,
            "volume": 500,
            "open_interest": 200,
        }
        md = parse_ticker_message(msg)

        assert md.contract_id == "KXNFL-26-SPREAD-KC-BUF-3.5"
        assert md.last_price_cents == 51
        assert md.yes_bid_cents == 50
        assert md.yes_ask_cents == 53


# ============================================================
# Contract parsing
# ============================================================


class TestParseContract:
    """Convert Kalshi market JSON to Contract."""

    def test_active_nfl_contract(self):
        """Active NFL contract."""
        data = {
            "ticker": "KXNFL-26-WIN-KC",
            "event_ticker": "KXNFL-26",
            "title": "Kansas City Chiefs to win",
            "market_type": "binary",
            "status": "active",
            "last_price": 65,
            "volume": 5000,
            "open_interest": 2000,
        }
        contract = parse_contract(data)

        assert contract.contract_id == "KXNFL-26-WIN-KC"
        assert contract.event_id == "KXNFL-26"
        assert contract.title == "Kansas City Chiefs to win"
        assert contract.is_active is True
        assert contract.last_price_cents == 65

    def test_closed_contract(self):
        """Closed/determined contract."""
        data = {
            "ticker": "KXNFL-25-WIN-SEA",
            "event_ticker": "KXNFL-25",
            "title": "Seattle Seahawks to win",
            "status": "determined",
            "last_price": 100,
        }
        contract = parse_contract(data)
        assert contract.is_active is False


# ============================================================
# Request body builders
# ============================================================


class TestRequestBodyBuilders:
    """Build Kalshi API request bodies from our dataclasses."""

    def test_create_order_yes(self):
        """YES buy order body."""
        from nfl_analytics.markets.base import Order, Side

        order = Order(
            order_id="my-order-1",
            contract_id="KXNFL-26-WIN-KC",
            side=Side.YES,
            price_cents=55,
            quantity=10,
        )
        body = build_create_order_body(order)

        assert body["ticker"] == "KXNFL-26-WIN-KC"
        assert body["side"] == "yes"
        assert body["action"] == "buy"
        assert body["count"] == 10
        assert body["yes_price"] == 55
        assert body["client_order_id"] == "my-order-1"
        assert "no_price" not in body

    def test_create_order_no(self):
        """NO buy order body."""
        from nfl_analytics.markets.base import Order, Side

        order = Order(
            order_id="my-order-2",
            contract_id="KXNFL-26-SPREAD-KC-BUF-3.5",
            side=Side.NO,
            price_cents=42,
            quantity=5,
        )
        body = build_create_order_body(order)

        assert body["side"] == "no"
        assert body["no_price"] == 42
        assert "yes_price" not in body

    def test_batch_create_body(self):
        """Batch create includes all orders."""
        from nfl_analytics.markets.base import Order, Side

        orders = [
            Order(order_id=f"o-{i}", contract_id=f"C-{i}", side=Side.YES, price_cents=50, quantity=1)
            for i in range(3)
        ]
        body = build_batch_create_body(orders)

        assert len(body["orders"]) == 3
        assert body["orders"][0]["ticker"] == "C-0"
        assert body["orders"][2]["client_order_id"] == "o-2"

    def test_batch_cancel_body(self):
        """Batch cancel includes order IDs."""
        body = build_batch_cancel_body(["order-1", "order-2", "order-3"])

        assert len(body["orders"]) == 3
        assert body["orders"][0]["order_id"] == "order-1"
