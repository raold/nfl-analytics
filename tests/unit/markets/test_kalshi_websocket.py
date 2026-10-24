"""Tests for Kalshi WebSocket client — message parsing and local orderbook."""

from __future__ import annotations

import json

import pytest

from nfl_analytics.markets.base import OrderBookLevel, Side
from nfl_analytics.markets.kalshi.websocket import KalshiWebSocket, LocalOrderBook


# ============================================================
# Local orderbook state
# ============================================================


class TestLocalOrderBook:
    """Local orderbook maintained from snapshots + deltas."""

    def test_apply_snapshot(self):
        """Snapshot replaces all state."""
        from nfl_analytics.markets.base import OrderBook

        local = LocalOrderBook(contract_id="TEST-01")

        ob = OrderBook(
            contract_id="TEST-01",
            yes_bids=[
                OrderBookLevel(price_cents=55, quantity=200),
                OrderBookLevel(price_cents=50, quantity=100),
            ],
            yes_asks=[
                OrderBookLevel(price_cents=56, quantity=150),
                OrderBookLevel(price_cents=60, quantity=300),
            ],
        )
        local.apply_snapshot(ob, seq=1)

        assert local.last_seq == 1
        assert local.yes_levels[55] == 200
        assert local.yes_levels[50] == 100
        # YES asks → NO bids: ask@56 → no_bid@44, ask@60 → no_bid@40
        assert local.no_levels[44] == 150
        assert local.no_levels[40] == 300

    def test_apply_delta_add_quantity(self):
        """Delta adds quantity at a price level."""
        local = LocalOrderBook(contract_id="TEST-01")
        local.yes_levels[50] = 100
        local.last_seq = 1

        ok = local.apply_delta({"side": "yes", "price": 50, "delta": 50}, seq=2)

        assert ok is True
        assert local.yes_levels[50] == 150
        assert local.last_seq == 2

    def test_apply_delta_remove_quantity(self):
        """Negative delta reduces quantity; zero removes level."""
        local = LocalOrderBook(contract_id="TEST-01")
        local.yes_levels[50] = 100
        local.last_seq = 1

        ok = local.apply_delta({"side": "yes", "price": 50, "delta": -100}, seq=2)

        assert ok is True
        assert 50 not in local.yes_levels  # Level removed

    def test_apply_delta_new_level(self):
        """Delta creates a new price level."""
        local = LocalOrderBook(contract_id="TEST-01")
        local.last_seq = 1

        ok = local.apply_delta({"side": "no", "price": 45, "delta": 200}, seq=2)

        assert ok is True
        assert local.no_levels[45] == 200

    def test_seq_gap_detected(self):
        """Sequence gap returns False."""
        local = LocalOrderBook(contract_id="TEST-01")
        local.last_seq = 5

        ok = local.apply_delta({"side": "yes", "price": 50, "delta": 10}, seq=8)

        assert ok is False  # Expected seq=6, got 8

    def test_to_orderbook(self):
        """Convert local state to OrderBook dataclass."""
        local = LocalOrderBook(contract_id="TEST-01")
        local.yes_levels = {55: 200, 50: 100}
        local.no_levels = {45: 150, 40: 300}

        ob = local.to_orderbook()

        assert ob.contract_id == "TEST-01"
        # YES bids sorted descending
        assert ob.yes_bids[0].price_cents == 55
        assert ob.yes_bids[1].price_cents == 50
        # YES asks = 100 - NO price, sorted ascending
        assert ob.yes_asks[0].price_cents == 55  # 100 - 45
        assert ob.yes_asks[1].price_cents == 60  # 100 - 40

    def test_snapshot_clears_previous_state(self):
        """New snapshot replaces old data completely."""
        from nfl_analytics.markets.base import OrderBook

        local = LocalOrderBook(contract_id="TEST-01")
        local.yes_levels = {50: 100, 55: 200}
        local.no_levels = {45: 150}

        # New snapshot with different data
        ob = OrderBook(
            contract_id="TEST-01",
            yes_bids=[OrderBookLevel(price_cents=60, quantity=300)],
            yes_asks=[OrderBookLevel(price_cents=62, quantity=100)],
        )
        local.apply_snapshot(ob, seq=10)

        assert 50 not in local.yes_levels
        assert 55 not in local.yes_levels
        assert local.yes_levels[60] == 300
        assert local.last_seq == 10


# ============================================================
# WebSocket message handling
# ============================================================


class TestMessageHandling:
    """Parse and route WebSocket messages via _handle_message."""

    def _make_ws(self) -> KalshiWebSocket:
        """Create a WebSocket client with mock auth."""
        from unittest.mock import MagicMock
        auth = MagicMock()
        auth.sign.return_value = {}
        auth.sign_websocket.return_value = {}
        return KalshiWebSocket(auth=auth, demo=True)

    def test_ticker_message_triggers_callback(self):
        """Ticker message invokes on_ticker callback."""
        ws = self._make_ws()
        received = []
        ws.on_ticker = lambda md: received.append(md)

        ws._handle_message(json.dumps({
            "type": "ticker",
            "sid": 1,
            "msg": {
                "market_ticker": "KXNFL-26-WIN-KC",
                "yes_bid": 54,
                "yes_ask": 56,
                "price": 55,
                "volume": 1000,
                "open_interest": 500,
            }
        }))

        assert len(received) == 1
        assert received[0].contract_id == "KXNFL-26-WIN-KC"
        assert received[0].last_price_cents == 55

    def test_fill_message_triggers_callback(self):
        """Fill message invokes on_fill callback."""
        ws = self._make_ws()
        received = []
        ws.on_fill = lambda fill: received.append(fill)

        ws._handle_message(json.dumps({
            "type": "fill",
            "sid": 13,
            "msg": {
                "fill_id": "fill-123",
                "trade_id": "trade-456",
                "order_id": "order-789",
                "market_ticker": "KXNFL-26-WIN-KC",
                "side": "yes",
                "action": "buy",
                "count": 5,
                "yes_price": 55,
                "no_price": 45,
                "is_taker": True,
            }
        }))

        assert len(received) == 1
        assert received[0].fill_id == "fill-123"
        assert received[0].side == Side.YES
        assert received[0].quantity == 5

    def test_orderbook_snapshot_updates_local_state(self):
        """Orderbook snapshot initializes local book and triggers callback."""
        ws = self._make_ws()
        received = []
        ws.on_orderbook = lambda ob: received.append(ob)

        ws._handle_message(json.dumps({
            "type": "orderbook_snapshot",
            "sid": 2,
            "seq": 1,
            "msg": {
                "market_ticker": "TEST-01",
                "yes": [[50, 100], [55, 200]],
                "no": [[45, 150]],
            }
        }))

        assert len(received) == 1
        assert received[0].contract_id == "TEST-01"
        assert "TEST-01" in ws._orderbooks

    def test_orderbook_delta_updates_local_state(self):
        """Orderbook delta modifies existing local book."""
        ws = self._make_ws()
        # First set up with snapshot
        ws._handle_message(json.dumps({
            "type": "orderbook_snapshot",
            "sid": 2,
            "seq": 1,
            "msg": {
                "market_ticker": "TEST-01",
                "yes": [[50, 100]],
                "no": [],
            }
        }))

        received = []
        ws.on_orderbook = lambda ob: received.append(ob)

        # Apply delta
        ws._handle_message(json.dumps({
            "type": "orderbook_delta",
            "sid": 2,
            "seq": 2,
            "msg": {
                "market_ticker": "TEST-01",
                "price": 50,
                "delta": 50,
                "side": "yes",
            }
        }))

        assert len(received) == 1
        local = ws._orderbooks["TEST-01"]
        assert local.yes_levels[50] == 150  # 100 + 50

    def test_subscribed_message_tracked(self):
        """Subscription confirmation is recorded."""
        ws = self._make_ws()

        ws._handle_message(json.dumps({
            "type": "subscribed",
            "id": 1,
            "msg": {"sid": 2, "channel": "orderbook_delta"}
        }))

        assert 2 in ws._subscriptions

    def test_invalid_json_handled(self):
        """Invalid JSON doesn't crash."""
        ws = self._make_ws()
        ws._handle_message("this is not json")  # Should not raise

    def test_unknown_message_type_ignored(self):
        """Unknown message types are silently ignored."""
        ws = self._make_ws()
        ws._handle_message(json.dumps({
            "type": "unknown_future_type",
            "sid": 99,
            "msg": {},
        }))
        # No error

    def test_get_orderbook_returns_none_for_unknown(self):
        """get_orderbook returns None for unsubscribed contract."""
        ws = self._make_ws()
        assert ws.get_orderbook("UNKNOWN") is None

    def test_get_orderbook_returns_current_state(self):
        """get_orderbook returns the locally-maintained book."""
        ws = self._make_ws()

        ws._handle_message(json.dumps({
            "type": "orderbook_snapshot",
            "sid": 2,
            "seq": 1,
            "msg": {
                "market_ticker": "TEST-01",
                "yes": [[50, 100]],
                "no": [[48, 200]],
            }
        }))

        ob = ws.get_orderbook("TEST-01")
        assert ob is not None
        assert ob.yes_bids[0].price_cents == 50
