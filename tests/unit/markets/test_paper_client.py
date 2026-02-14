"""
Unit tests for py/markets/paper_client.py — Paper trading client.

Tests cover:
- Contract registration
- Order placement (immediate fills and resting orders)
- YES/NO fill logic
- Balance management (reservation, refunds, settlement payouts)
- Order cancellation
- Price movement triggering resting order fills
- Contract settlement
- Edge cases (insufficient balance, unregistered contracts)
"""

import pytest

from py.markets.base import (
    ContractNotFoundError,
    InsufficientBalanceError,
    Order,
    OrderRejectedError,
    OrderStatus,
    Side,
)
from py.markets.paper_client import PaperMarketClient


@pytest.fixture
def client():
    """Fresh paper client with $1,000 balance and 4¢ spread."""
    c = PaperMarketClient(initial_balance_cents=100_000, default_spread_cents=4)
    c.register_contract("KC-WIN", mid_price_cents=60, title="Chiefs Win")
    c.register_contract("BUF-WIN", mid_price_cents=45, title="Bills Win")
    return c


def _make_order(
    order_id: str = "o1",
    contract_id: str = "KC-WIN",
    side: Side = Side.YES,
    price_cents: int = 55,
    quantity: int = 10,
) -> Order:
    return Order(
        order_id=order_id,
        contract_id=contract_id,
        side=side,
        price_cents=price_cents,
        quantity=quantity,
    )


# ---------------------------------------------------------------------------
# Contract registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_contract(self, client):
        """Registered contract is accessible."""
        import asyncio

        contract = asyncio.run(client.get_contract("KC-WIN"))
        assert contract.title == "Chiefs Win"
        assert contract.is_active is True

    def test_unregistered_contract_raises(self, client):
        import asyncio

        with pytest.raises(ContractNotFoundError):
            asyncio.run(client.get_contract("NONEXISTENT"))


# ---------------------------------------------------------------------------
# Order placement — immediate fills
# ---------------------------------------------------------------------------


class TestImmediateFills:
    def test_yes_buy_above_ask_fills_immediately(self, client):
        """YES buy at price >= ask fills at ask price.

        KC-WIN mid=60, spread=4 → ask=62, bid=58.
        Buy YES at 65¢ → fills at 62¢ (ask).
        """
        import asyncio

        order = _make_order(price_cents=65, quantity=5)
        result = asyncio.run(client.place_order(order))

        assert result.status == OrderStatus.FILLED
        assert result.filled_quantity == 5
        assert result.avg_fill_price_cents == 62  # Filled at ask

    def test_yes_buy_at_ask_fills(self, client):
        """Limit exactly at ask should fill."""
        import asyncio

        order = _make_order(price_cents=62, quantity=5)  # Ask is 62
        result = asyncio.run(client.place_order(order))

        assert result.status == OrderStatus.FILLED
        assert result.avg_fill_price_cents == 62

    def test_yes_buy_below_ask_rests(self, client):
        """YES buy below ask rests on book.

        KC-WIN ask=62. Buy at 55¢ → should rest.
        """
        import asyncio

        order = _make_order(price_cents=55, quantity=5)
        result = asyncio.run(client.place_order(order))

        assert result.status == OrderStatus.OPEN
        assert result.filled_quantity == 0
        assert client.open_order_count == 1

    def test_no_buy_fills_correctly(self, client):
        """NO buy fills at (100 - best_bid).

        KC-WIN bid=58 → NO cost = 100 - 58 = 42¢.
        Buy NO at 42¢ → fills.
        """
        import asyncio

        order = _make_order(side=Side.NO, price_cents=42, quantity=5)
        result = asyncio.run(client.place_order(order))

        assert result.status == OrderStatus.FILLED
        assert result.avg_fill_price_cents == 42  # 100 - 58

    def test_no_buy_below_cost_rests(self, client):
        """NO buy below (100 - bid) should rest.

        KC-WIN bid=58 → NO cost = 42¢. Bidding 35¢ for NO rests.
        """
        import asyncio

        order = _make_order(side=Side.NO, price_cents=35, quantity=5)
        result = asyncio.run(client.place_order(order))

        assert result.status == OrderStatus.OPEN


# ---------------------------------------------------------------------------
# Balance management
# ---------------------------------------------------------------------------


class TestBalance:
    def test_balance_decrements_on_fill(self, client):
        """Balance should decrease by actual fill cost, not limit price."""
        import asyncio

        initial = asyncio.run(client.get_balance_cents())
        assert initial == 100_000

        # Buy 10 YES at limit 65¢, fills at ask 62¢
        order = _make_order(price_cents=65, quantity=10)
        asyncio.run(client.place_order(order))

        balance = asyncio.run(client.get_balance_cents())
        # Paid: 62¢ * 10 = 620¢. So balance = 100000 - 620 = 99380
        assert balance == 99_380

    def test_balance_reserved_on_resting_order(self, client):
        """Resting orders reserve funds at limit price."""
        import asyncio

        order = _make_order(price_cents=55, quantity=10)  # Below ask, will rest
        asyncio.run(client.place_order(order))

        balance = asyncio.run(client.get_balance_cents())
        # Reserved: 55¢ * 10 = 550¢
        assert balance == 100_000 - 550

    def test_insufficient_balance_rejects(self, client):
        """Order exceeding balance is rejected."""
        import asyncio

        # Try to buy 2000 contracts at 55¢ = 110,000¢ > 100,000¢ balance
        order = _make_order(price_cents=55, quantity=2000)
        with pytest.raises(InsufficientBalanceError):
            asyncio.run(client.place_order(order))

    def test_balance_refund_on_cancel(self, client):
        """Cancelling a resting order refunds reserved funds."""
        import asyncio

        order = _make_order(price_cents=55, quantity=10)
        asyncio.run(client.place_order(order))

        balance_after_order = asyncio.run(client.get_balance_cents())
        assert balance_after_order == 100_000 - 550

        asyncio.run(client.cancel_order("o1"))
        balance_after_cancel = asyncio.run(client.get_balance_cents())
        assert balance_after_cancel == 100_000


# ---------------------------------------------------------------------------
# Order cancellation
# ---------------------------------------------------------------------------


class TestCancellation:
    def test_cancel_resting_order(self, client):
        import asyncio

        order = _make_order(price_cents=55, quantity=5)
        asyncio.run(client.place_order(order))

        cancelled = asyncio.run(client.cancel_order("o1"))
        assert cancelled.status == OrderStatus.CANCELLED
        assert client.open_order_count == 0

    def test_cancel_filled_order_raises(self, client):
        import asyncio

        order = _make_order(price_cents=65, quantity=5)  # Will fill immediately
        asyncio.run(client.place_order(order))

        with pytest.raises(OrderRejectedError, match="cannot cancel"):
            asyncio.run(client.cancel_order("o1"))

    def test_cancel_nonexistent_raises(self, client):
        import asyncio

        with pytest.raises(OrderRejectedError, match="not found"):
            asyncio.run(client.cancel_order("nonexistent"))

    def test_cancel_all_orders(self, client):
        import asyncio

        # Place 3 resting orders
        for i in range(3):
            order = _make_order(order_id=f"o{i}", price_cents=55, quantity=1)
            asyncio.run(client.place_order(order))

        count = asyncio.run(client.cancel_all_orders())
        assert count == 3
        assert client.open_order_count == 0


# ---------------------------------------------------------------------------
# Price movement → resting order fills
# ---------------------------------------------------------------------------


class TestPriceMovement:
    def test_price_drop_fills_resting_yes_buy(self, client):
        """Resting YES buy fills when ask drops to limit price.

        Place YES buy at 55¢ (rests because ask=62).
        Move price so ask=55 → fill.
        """
        import asyncio

        order = _make_order(price_cents=55, quantity=5)
        asyncio.run(client.place_order(order))
        assert order.status == OrderStatus.OPEN

        # Move mid price down: mid=53 → ask=55 → matches our limit
        fills = client.set_price("KC-WIN", 53)
        assert len(fills) == 1
        assert fills[0].price_cents == 55  # Filled at new ask
        assert order.status == OrderStatus.FILLED

    def test_price_rise_doesnt_fill_yes_buy(self, client):
        """Resting YES buy doesn't fill on price rise."""
        import asyncio

        order = _make_order(price_cents=55, quantity=5)
        asyncio.run(client.place_order(order))

        fills = client.set_price("KC-WIN", 70)  # Ask goes to 72
        assert len(fills) == 0
        assert order.status == OrderStatus.OPEN

    def test_multiple_resting_orders_partial_trigger(self, client):
        """Only orders matching the new price fill."""
        import asyncio

        # Order at 55¢ and 50¢
        o1 = _make_order(order_id="o1", price_cents=55, quantity=5)
        o2 = _make_order(order_id="o2", price_cents=50, quantity=5)
        asyncio.run(client.place_order(o1))
        asyncio.run(client.place_order(o2))

        # Mid=53 → ask=55 → o1 fills, o2 doesn't
        fills = client.set_price("KC-WIN", 53)
        assert len(fills) == 1
        assert fills[0].order_id == "o1"
        assert o1.status == OrderStatus.FILLED
        assert o2.status == OrderStatus.OPEN


# ---------------------------------------------------------------------------
# Position tracking
# ---------------------------------------------------------------------------


class TestPositions:
    def test_position_created_on_fill(self, client):
        import asyncio

        order = _make_order(price_cents=65, quantity=10)  # Fills at ask=62
        asyncio.run(client.place_order(order))

        positions = asyncio.run(client.get_positions())
        assert len(positions) == 1
        assert positions[0].contract_id == "KC-WIN"
        assert positions[0].side == Side.YES
        assert positions[0].quantity == 10
        assert positions[0].avg_entry_cents == 62

    def test_multiple_fills_aggregate(self, client):
        """Multiple fills on same contract update the position."""
        import asyncio

        # First buy: 10 @ 62¢
        o1 = _make_order(order_id="o1", price_cents=65, quantity=10)
        asyncio.run(client.place_order(o1))

        # Second buy: 10 @ 62¢ (same ask)
        o2 = _make_order(order_id="o2", price_cents=65, quantity=10)
        asyncio.run(client.place_order(o2))

        positions = asyncio.run(client.get_positions())
        assert len(positions) == 1
        assert positions[0].quantity == 20

    def test_separate_contracts_separate_positions(self, client):
        import asyncio

        o1 = _make_order(order_id="o1", contract_id="KC-WIN", price_cents=65, quantity=5)
        o2 = _make_order(order_id="o2", contract_id="BUF-WIN", price_cents=50, quantity=5)
        asyncio.run(client.place_order(o1))
        asyncio.run(client.place_order(o2))

        positions = asyncio.run(client.get_positions())
        assert len(positions) == 2


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------


class TestSettlement:
    def test_yes_position_wins(self, client):
        """YES position on winning contract receives 100¢ per contract."""
        import asyncio

        order = _make_order(price_cents=65, quantity=10)  # Fills at 62¢
        asyncio.run(client.place_order(order))

        balance_before = asyncio.run(client.get_balance_cents())
        payouts = client.settle_contract("KC-WIN", outcome_yes=True)

        assert len(payouts) == 1
        assert payouts[0][1] == 1000  # 100¢ × 10 contracts

        balance_after = asyncio.run(client.get_balance_cents())
        assert balance_after == balance_before + 1000

    def test_yes_position_loses(self, client):
        """YES position on losing contract gets nothing."""
        import asyncio

        order = _make_order(price_cents=65, quantity=10)
        asyncio.run(client.place_order(order))

        balance_before = asyncio.run(client.get_balance_cents())
        payouts = client.settle_contract("KC-WIN", outcome_yes=False)

        assert payouts[0][1] == 0
        balance_after = asyncio.run(client.get_balance_cents())
        assert balance_after == balance_before  # No payout

    def test_no_position_wins(self, client):
        """NO position profits when event doesn't happen."""
        import asyncio

        order = _make_order(side=Side.NO, price_cents=42, quantity=10)
        asyncio.run(client.place_order(order))

        payouts = client.settle_contract("KC-WIN", outcome_yes=False)
        assert payouts[0][1] == 1000  # NO wins → 100¢ each

    def test_settlement_cancels_resting_orders(self, client):
        """Settlement cancels resting orders and refunds."""
        import asyncio

        resting = _make_order(order_id="rest", price_cents=55, quantity=10)
        asyncio.run(client.place_order(resting))
        assert resting.status == OrderStatus.OPEN

        client.settle_contract("KC-WIN", outcome_yes=True)
        assert resting.status == OrderStatus.CANCELLED
        # Funds refunded
        assert asyncio.run(client.get_balance_cents()) >= 100_000 - 550 + 550

    def test_settlement_deactivates_contract(self, client):
        import asyncio

        client.settle_contract("KC-WIN", outcome_yes=True)
        contract = asyncio.run(client.get_contract("KC-WIN"))
        assert contract.is_active is False


# ---------------------------------------------------------------------------
# Fills history
# ---------------------------------------------------------------------------


class TestFillsHistory:
    def test_fills_recorded(self, client):
        import asyncio

        order = _make_order(price_cents=65, quantity=10)
        asyncio.run(client.place_order(order))

        all_fills = asyncio.run(client.get_fills())
        assert len(all_fills) == 1
        assert all_fills[0].order_id == "o1"
        assert all_fills[0].quantity == 10

    def test_fills_filtered_by_order(self, client):
        import asyncio

        o1 = _make_order(order_id="o1", price_cents=65, quantity=5)
        o2 = _make_order(order_id="o2", price_cents=65, quantity=3)
        asyncio.run(client.place_order(o1))
        asyncio.run(client.place_order(o2))

        fills_o1 = asyncio.run(client.get_fills(order_id="o1"))
        assert len(fills_o1) == 1
        assert fills_o1[0].quantity == 5


# ---------------------------------------------------------------------------
# Summary / diagnostics
# ---------------------------------------------------------------------------


class TestDiagnostics:
    def test_summary_initial_state(self, client):
        summary = client.summary()
        assert summary["balance_dollars"] == 1000.0
        assert summary["total_pnl_dollars"] == 0.0
        assert summary["open_positions"] == 0
        assert summary["total_fills"] == 0

    def test_pnl_after_fill(self, client):
        """PNL = (cash + position_market_value) - initial_balance.

        Buy 10 YES @ 62¢ (ask). Mid = 60¢.
        Cash: 100000 - 620 = 99380¢
        Position value: 60 * 10 = 600¢
        Equity: 99380 + 600 = 99980¢
        PNL: 99980 - 100000 = -20¢ (lost 2¢/contract on entry above mid)
        """
        import asyncio

        order = _make_order(price_cents=65, quantity=10)  # Fills at ask=62
        asyncio.run(client.place_order(order))

        assert client.total_pnl_cents == -20
        assert client.total_pnl_dollars == -0.20

        summary = client.summary()
        assert summary["open_positions"] == 1
        assert summary["total_fills"] == 1

    def test_pnl_after_price_move(self, client):
        """PNL changes with market price."""
        import asyncio

        order = _make_order(price_cents=65, quantity=10)  # Fills at ask=62
        asyncio.run(client.place_order(order))

        # Move price up → profit
        client.set_price("KC-WIN", 70)
        # Cash: 99380, Position value: 70*10=700, PNL: (99380+700)-100000 = +80
        assert client.total_pnl_cents == 80

        # Move price down → loss
        client.set_price("KC-WIN", 50)
        # Cash: 99380, Position value: 50*10=500, PNL: (99380+500)-100000 = -120
        assert client.total_pnl_cents == -120
