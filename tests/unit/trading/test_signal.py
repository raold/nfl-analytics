"""
Unit tests for py/trading/signal.py — Trade signal dataclass.
"""

import pytest

from py.markets.base import OrderType, Side
from py.trading.signal import Signal


class TestSignal:
    def test_create_signal(self):
        sig = Signal(
            contract_id="KXNFL-26-KC",
            side=Side.YES,
            quantity=10,
            price_cents=55,
            edge_cents=7,
            model_prob=0.62,
            confidence=0.85,
            strategy="directional",
        )
        assert sig.model_prob_cents == 62
        assert sig.edge_pct == 0.07
        assert sig.expected_value_cents == 7.0

    def test_passes_threshold_default(self):
        sig = Signal(
            contract_id="test",
            side=Side.YES,
            quantity=1,
            price_cents=50,
            edge_cents=5,
            model_prob=0.55,
            confidence=0.7,
            strategy="test",
        )
        assert sig.passes_threshold() is True  # 5 >= 3 and 0.7 >= 0.5

    def test_fails_threshold_low_edge(self):
        sig = Signal(
            contract_id="test",
            side=Side.YES,
            quantity=1,
            price_cents=50,
            edge_cents=2,  # Below default 3
            model_prob=0.52,
            confidence=0.8,
            strategy="test",
        )
        assert sig.passes_threshold() is False

    def test_fails_threshold_low_confidence(self):
        sig = Signal(
            contract_id="test",
            side=Side.YES,
            quantity=1,
            price_cents=50,
            edge_cents=10,
            model_prob=0.60,
            confidence=0.3,  # Below default 0.5
            strategy="test",
        )
        assert sig.passes_threshold() is False

    def test_custom_threshold(self):
        sig = Signal(
            contract_id="test",
            side=Side.YES,
            quantity=1,
            price_cents=50,
            edge_cents=5,
            model_prob=0.55,
            confidence=0.6,
            strategy="test",
        )
        assert sig.passes_threshold(min_edge_cents=5, min_confidence=0.6) is True
        assert sig.passes_threshold(min_edge_cents=6, min_confidence=0.6) is False

    def test_to_order(self):
        sig = Signal(
            contract_id="KXNFL-26-KC",
            side=Side.YES,
            quantity=5,
            price_cents=60,
            edge_cents=8,
            model_prob=0.68,
            confidence=0.9,
            strategy="market_maker",
        )
        order = sig.to_order()

        assert order.contract_id == "KXNFL-26-KC"
        assert order.side == Side.YES
        assert order.quantity == 5
        assert order.price_cents == 60
        assert order.order_type == OrderType.LIMIT
        assert order.strategy == "market_maker"
        assert order.order_id  # Should have a UUID

    def test_to_order_custom_id(self):
        sig = Signal(
            contract_id="test",
            side=Side.NO,
            quantity=1,
            price_cents=40,
            edge_cents=5,
            model_prob=0.45,
            confidence=0.7,
            strategy="directional",
        )
        order = sig.to_order(order_id="custom-123")
        assert order.order_id == "custom-123"

    def test_repr(self):
        sig = Signal(
            contract_id="KC-WIN",
            side=Side.YES,
            quantity=10,
            price_cents=55,
            edge_cents=7,
            model_prob=0.62,
            confidence=0.85,
            strategy="directional",
        )
        r = repr(sig)
        assert "yes" in r
        assert "KC-WIN" in r
        assert "55¢" in r
        assert "directional" in r

    def test_no_side_signal(self):
        """NO signals represent betting against an event."""
        sig = Signal(
            contract_id="BUF-WIN",
            side=Side.NO,
            quantity=20,
            price_cents=35,
            edge_cents=10,
            model_prob=0.25,  # Model thinks 25% chance → NO at 35¢ is cheap
            confidence=0.75,
            strategy="directional",
        )
        assert sig.model_prob_cents == 25
        order = sig.to_order()
        assert order.side == Side.NO
