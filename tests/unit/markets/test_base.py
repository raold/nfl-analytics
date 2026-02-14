"""
Unit tests for py/markets/base.py — core dataclasses and enums.

Tests cover:
- Order construction and validation
- Fill cost calculations
- Position P&L and update logic
- OrderBook properties (spread, midpoint, crossed detection)
- MarketData implied probability
- Error hierarchy
"""

import pytest

from py.markets.base import (
    Contract,
    ContractNotFoundError,
    ContractType,
    Fill,
    InsufficientBalanceError,
    MarketData,
    MarketError,
    Order,
    OrderBook,
    OrderBookLevel,
    OrderRejectedError,
    OrderStatus,
    Position,
    RateLimitError,
    Side,
    Sport,
)


# ---------------------------------------------------------------------------
# Order tests
# ---------------------------------------------------------------------------


class TestOrder:
    def test_create_valid_order(self):
        order = Order(
            order_id="test-1",
            contract_id="KXNFL-26-KC",
            side=Side.YES,
            price_cents=55,
            quantity=10,
        )
        assert order.price_cents == 55
        assert order.quantity == 10
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == 0
        assert order.remaining_quantity == 10
        assert order.is_active is True

    def test_cost_calculations(self):
        order = Order(
            order_id="test-2",
            contract_id="KXNFL-26-KC",
            side=Side.YES,
            price_cents=60,
            quantity=5,
        )
        assert order.cost_cents == 300
        assert order.cost_dollars == 3.00

    def test_remaining_after_partial_fill(self):
        order = Order(
            order_id="test-3",
            contract_id="KXNFL-26-KC",
            side=Side.YES,
            price_cents=50,
            quantity=10,
            filled_quantity=3,
            status=OrderStatus.PARTIAL,
        )
        assert order.remaining_quantity == 7
        assert order.is_active is True

    def test_filled_order_not_active(self):
        order = Order(
            order_id="test-4",
            contract_id="KXNFL-26-KC",
            side=Side.YES,
            price_cents=50,
            quantity=10,
            filled_quantity=10,
            status=OrderStatus.FILLED,
        )
        assert order.remaining_quantity == 0
        assert order.is_active is False

    def test_cancelled_order_not_active(self):
        order = Order(
            order_id="test-5",
            contract_id="KXNFL-26-KC",
            side=Side.NO,
            price_cents=40,
            quantity=5,
            status=OrderStatus.CANCELLED,
        )
        assert order.is_active is False

    def test_price_cents_below_range_raises(self):
        with pytest.raises(ValueError, match="price_cents must be 1-99"):
            Order(
                order_id="test-6",
                contract_id="KXNFL-26-KC",
                side=Side.YES,
                price_cents=0,
                quantity=1,
            )

    def test_price_cents_above_range_raises(self):
        with pytest.raises(ValueError, match="price_cents must be 1-99"):
            Order(
                order_id="test-7",
                contract_id="KXNFL-26-KC",
                side=Side.YES,
                price_cents=100,
                quantity=1,
            )

    def test_zero_quantity_raises(self):
        with pytest.raises(ValueError, match="quantity must be >= 1"):
            Order(
                order_id="test-8",
                contract_id="KXNFL-26-KC",
                side=Side.YES,
                price_cents=50,
                quantity=0,
            )

    def test_boundary_prices(self):
        """Test edge prices: 1¢ (99% NO) and 99¢ (99% YES)."""
        order_low = Order(
            order_id="low",
            contract_id="KXNFL-26-KC",
            side=Side.YES,
            price_cents=1,
            quantity=100,
        )
        assert order_low.cost_cents == 100

        order_high = Order(
            order_id="high",
            contract_id="KXNFL-26-KC",
            side=Side.YES,
            price_cents=99,
            quantity=1,
        )
        assert order_high.cost_cents == 99


# ---------------------------------------------------------------------------
# Fill tests
# ---------------------------------------------------------------------------


class TestFill:
    def test_net_cost_no_fees(self):
        fill = Fill(
            fill_id="f1",
            order_id="o1",
            contract_id="KXNFL-26-KC",
            side=Side.YES,
            price_cents=55,
            quantity=10,
        )
        assert fill.net_cost_cents == 550

    def test_net_cost_with_fees(self):
        fill = Fill(
            fill_id="f2",
            order_id="o2",
            contract_id="KXNFL-26-KC",
            side=Side.NO,
            price_cents=45,
            quantity=5,
            fee_cents=10,
        )
        assert fill.net_cost_cents == 235  # 45*5 + 10


# ---------------------------------------------------------------------------
# Position tests
# ---------------------------------------------------------------------------


class TestPosition:
    def test_yes_position_profit(self):
        """YES position profits when price rises."""
        pos = Position(
            contract_id="KXNFL-26-KC",
            side=Side.YES,
            quantity=10,
            avg_entry_cents=50,
        )
        assert pos.cost_basis_cents == 500
        # Price rose to 60
        assert pos.unrealized_pnl_cents(60) == 100  # (60-50)*10
        # Price dropped to 40
        assert pos.unrealized_pnl_cents(40) == -100  # (40-50)*10

    def test_no_position_profit(self):
        """NO position profits when YES price drops."""
        pos = Position(
            contract_id="KXNFL-26-KC",
            side=Side.NO,
            quantity=10,
            avg_entry_cents=50,
        )
        # YES price dropped to 40 → NO position profits
        assert pos.unrealized_pnl_cents(40) == 100  # (50-40)*10
        # YES price rose to 60 → NO position loses
        assert pos.unrealized_pnl_cents(60) == -100  # (50-60)*10

    def test_update_from_fill_same_side(self):
        """Adding to position updates average entry."""
        pos = Position(
            contract_id="KXNFL-26-KC",
            side=Side.YES,
            quantity=10,
            avg_entry_cents=50,
        )
        fill = Fill(
            fill_id="f1",
            order_id="o1",
            contract_id="KXNFL-26-KC",
            side=Side.YES,
            price_cents=60,
            quantity=10,
        )
        pos.update_from_fill(fill)
        assert pos.quantity == 20
        assert pos.avg_entry_cents == 55  # (50*10 + 60*10) / 20

    def test_update_from_fill_opposite_side_reduces(self):
        """Opposite side fill reduces position."""
        pos = Position(
            contract_id="KXNFL-26-KC",
            side=Side.YES,
            quantity=10,
            avg_entry_cents=50,
        )
        fill = Fill(
            fill_id="f1",
            order_id="o1",
            contract_id="KXNFL-26-KC",
            side=Side.NO,
            price_cents=55,
            quantity=4,
        )
        pos.update_from_fill(fill)
        assert pos.quantity == 6
        assert pos.side == Side.YES  # Still YES, just smaller

    def test_update_from_fill_flip_side(self):
        """Closing more than position flips the side."""
        pos = Position(
            contract_id="KXNFL-26-KC",
            side=Side.YES,
            quantity=5,
            avg_entry_cents=50,
        )
        fill = Fill(
            fill_id="f1",
            order_id="o1",
            contract_id="KXNFL-26-KC",
            side=Side.NO,
            price_cents=55,
            quantity=8,
        )
        pos.update_from_fill(fill)
        assert pos.quantity == 3
        assert pos.side == Side.NO
        assert pos.avg_entry_cents == 55


# ---------------------------------------------------------------------------
# OrderBook tests
# ---------------------------------------------------------------------------


class TestOrderBook:
    def test_basic_properties(self):
        book = OrderBook(
            contract_id="test",
            yes_bids=[
                OrderBookLevel(price_cents=48, quantity=100),
                OrderBookLevel(price_cents=47, quantity=200),
            ],
            yes_asks=[
                OrderBookLevel(price_cents=52, quantity=100),
                OrderBookLevel(price_cents=53, quantity=200),
            ],
        )
        assert book.best_bid_cents == 48
        assert book.best_ask_cents == 52
        assert book.mid_price_cents == 50.0
        assert book.spread_cents == 4
        assert book.is_crossed is False

    def test_empty_book(self):
        book = OrderBook(contract_id="empty")
        assert book.best_bid_cents is None
        assert book.best_ask_cents is None
        assert book.mid_price_cents is None
        assert book.spread_cents is None
        assert book.is_crossed is False

    def test_crossed_book(self):
        """Crossed book means bid >= ask (should trigger fill)."""
        book = OrderBook(
            contract_id="crossed",
            yes_bids=[OrderBookLevel(price_cents=55, quantity=10)],
            yes_asks=[OrderBookLevel(price_cents=55, quantity=10)],
        )
        assert book.is_crossed is True

    def test_tight_spread(self):
        """1¢ spread is the tightest possible."""
        book = OrderBook(
            contract_id="tight",
            yes_bids=[OrderBookLevel(price_cents=50, quantity=10)],
            yes_asks=[OrderBookLevel(price_cents=51, quantity=10)],
        )
        assert book.spread_cents == 1
        assert book.mid_price_cents == 50.5


# ---------------------------------------------------------------------------
# MarketData tests
# ---------------------------------------------------------------------------


class TestMarketData:
    def test_implied_probability(self):
        data = MarketData(
            contract_id="test",
            last_price_cents=65,
            yes_bid_cents=63,
            yes_ask_cents=67,
            volume_24h=1000,
            open_interest=500,
        )
        assert data.mid_cents == 65.0
        assert data.spread_cents == 4
        assert data.implied_prob == 0.65

    def test_extreme_probabilities(self):
        """Near-certain events at 95¢."""
        data = MarketData(
            contract_id="certain",
            last_price_cents=95,
            yes_bid_cents=94,
            yes_ask_cents=96,
            volume_24h=10000,
            open_interest=5000,
        )
        assert data.implied_prob == 0.95


# ---------------------------------------------------------------------------
# Error hierarchy tests
# ---------------------------------------------------------------------------


class TestErrors:
    def test_error_hierarchy(self):
        """All specific errors inherit from MarketError."""
        assert issubclass(OrderRejectedError, MarketError)
        assert issubclass(InsufficientBalanceError, MarketError)
        assert issubclass(RateLimitError, MarketError)
        assert issubclass(ContractNotFoundError, MarketError)

    def test_rate_limit_retry_after(self):
        err = RateLimitError(retry_after_ms=2000)
        assert err.retry_after_ms == 2000
        assert "2000ms" in str(err)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_side_values(self):
        assert Side.YES.value == "yes"
        assert Side.NO.value == "no"

    def test_order_status_values(self):
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.FILLED.value == "filled"

    def test_sport_values(self):
        assert Sport.NFL.value == "nfl"
        assert Sport.NBA.value == "nba"
