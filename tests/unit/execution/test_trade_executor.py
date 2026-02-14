"""Tests for trade executor — signal submission with retry logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from py.markets.base import (
    ConnectionError as MarketConnectionError,
    MarketClient,
    MarketError,
    Order,
    OrderRejectedError,
    OrderStatus,
    RateLimitError,
    Side,
)
from py.execution.trade_executor import ExecutorConfig, TradeExecutor
from py.trading.signal import Signal

pytestmark = pytest.mark.anyio


# ============================================================
# Helpers
# ============================================================


def _signal(
    contract_id: str = "TEST-01",
    side: Side = Side.YES,
    price_cents: int = 55,
    quantity: int = 10,
    edge_cents: int = 5,
) -> Signal:
    return Signal(
        contract_id=contract_id,
        side=side,
        quantity=quantity,
        price_cents=price_cents,
        edge_cents=edge_cents,
        model_prob=0.60,
        confidence=0.8,
        strategy="test",
    )


def _filled_order(order: Order) -> Order:
    """Make a copy of the order that's filled."""
    order.status = OrderStatus.FILLED
    order.filled_quantity = order.quantity
    order.avg_fill_price_cents = order.price_cents
    return order


def _mock_client() -> MagicMock:
    """Create a mock MarketClient."""
    client = AsyncMock(spec=MarketClient)
    # Default: place_order returns a FILLED order
    async def _place_order(order):
        return _filled_order(order)
    client.place_order.side_effect = _place_order
    return client


# ============================================================
# Single signal submission
# ============================================================


class TestSingleSignal:
    """Submit a single signal."""

    async def test_submit_signal_success(self):
        """Single signal gets submitted and filled."""
        client = _mock_client()
        executor = TradeExecutor(client, ExecutorConfig(max_retries=0))

        signal = _signal()
        order = await executor.submit_signal(signal)

        assert order.status == OrderStatus.FILLED
        assert executor.stats["submitted"] == 1
        assert executor.stats["filled"] == 1

    async def test_submit_signal_rejected(self):
        """Signal rejected by exchange."""
        client = _mock_client()
        client.place_order.side_effect = OrderRejectedError("Bad order")
        executor = TradeExecutor(client, ExecutorConfig(max_retries=0))

        signal = _signal()
        order = await executor.submit_signal(signal)

        assert order.status == OrderStatus.REJECTED
        assert executor.stats["rejected"] == 1

    async def test_submit_signal_resting(self):
        """Signal becomes a resting order."""
        client = _mock_client()
        async def _place_resting(order):
            order.status = OrderStatus.OPEN
            return order
        client.place_order.side_effect = _place_resting
        executor = TradeExecutor(client, ExecutorConfig(max_retries=0))

        signal = _signal()
        order = await executor.submit_signal(signal)

        assert order.status == OrderStatus.OPEN
        assert executor.stats["submitted"] == 1
        assert executor.stats["filled"] == 0


# ============================================================
# Retry logic
# ============================================================


class TestRetryLogic:
    """Retry on transient failures."""

    async def test_retry_on_rate_limit(self):
        """Rate limit error triggers retry."""
        client = _mock_client()
        call_count = 0
        async def _rate_limit_then_success(order):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitError()
            return _filled_order(order)
        client.place_order.side_effect = _rate_limit_then_success

        executor = TradeExecutor(client, ExecutorConfig(max_retries=2, initial_backoff_ms=10))
        signal = _signal()
        order = await executor.submit_signal(signal)

        assert order.status == OrderStatus.FILLED
        assert call_count == 2

    async def test_retry_on_connection_error(self):
        """Connection error triggers retry."""
        client = _mock_client()
        call_count = 0
        async def _connect_fail_then_success(order):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise MarketConnectionError("Network down")
            return _filled_order(order)
        client.place_order.side_effect = _connect_fail_then_success

        executor = TradeExecutor(client, ExecutorConfig(max_retries=2, initial_backoff_ms=10))
        signal = _signal()
        order = await executor.submit_signal(signal)

        assert order.status == OrderStatus.FILLED
        assert call_count == 2

    async def test_no_retry_on_rejection(self):
        """OrderRejectedError does not retry (client error)."""
        client = _mock_client()
        client.place_order.side_effect = OrderRejectedError("Invalid")

        executor = TradeExecutor(client, ExecutorConfig(max_retries=3, initial_backoff_ms=10))
        signal = _signal()
        order = await executor.submit_signal(signal)

        # Should only call once (no retry)
        assert client.place_order.call_count == 1
        assert order.status == OrderStatus.REJECTED

    async def test_exhausted_retries(self):
        """All retries exhausted → rejected."""
        client = _mock_client()
        client.place_order.side_effect = RateLimitError()

        executor = TradeExecutor(client, ExecutorConfig(max_retries=2, initial_backoff_ms=10))
        signal = _signal()
        order = await executor.submit_signal(signal)

        assert order.status == OrderStatus.REJECTED
        assert client.place_order.call_count == 3  # initial + 2 retries


# ============================================================
# Multiple signals
# ============================================================


class TestMultipleSignals:
    """Submit multiple signals at once."""

    async def test_submit_multiple_individual(self):
        """Multiple signals submitted individually."""
        client = _mock_client()
        executor = TradeExecutor(
            client,
            ExecutorConfig(use_batch_api=False, max_retries=0),
        )

        signals = [_signal(f"C-{i}") for i in range(3)]
        orders = await executor.submit_signals(signals)

        assert len(orders) == 3
        assert all(o.status == OrderStatus.FILLED for o in orders)

    async def test_submit_batch(self):
        """Multiple signals submitted via batch API."""
        client = _mock_client()
        async def _batch_place(orders):
            for o in orders:
                _filled_order(o)
            return orders
        client.batch_place_orders.side_effect = _batch_place

        executor = TradeExecutor(
            client,
            ExecutorConfig(use_batch_api=True, max_retries=0),
        )

        signals = [_signal(f"C-{i}") for i in range(3)]
        orders = await executor.submit_signals(signals)

        assert len(orders) == 3
        client.batch_place_orders.assert_called_once()

    async def test_empty_signals_returns_empty(self):
        """No signals → no orders."""
        client = _mock_client()
        executor = TradeExecutor(client)

        orders = await executor.submit_signals([])
        assert orders == []


# ============================================================
# Callbacks
# ============================================================


class TestCallbacks:
    """Fill and reject callbacks."""

    async def test_on_fill_callback(self):
        """on_fill callback invoked on successful fill."""
        client = _mock_client()
        executor = TradeExecutor(client, ExecutorConfig(max_retries=0))

        fills = []
        executor.on_fill = lambda order, signal: fills.append((order, signal))

        signal = _signal()
        await executor.submit_signal(signal)

        assert len(fills) == 1
        assert fills[0][0].status == OrderStatus.FILLED
        assert fills[0][1].contract_id == "TEST-01"

    async def test_on_reject_callback(self):
        """on_reject callback invoked on rejection."""
        client = _mock_client()
        client.place_order.side_effect = OrderRejectedError("Bad")
        executor = TradeExecutor(client, ExecutorConfig(max_retries=0))

        rejects = []
        executor.on_reject = lambda order, signal, reason: rejects.append((order, signal, reason))

        signal = _signal()
        await executor.submit_signal(signal)

        assert len(rejects) == 1
        assert "Bad" in rejects[0][2]


# ============================================================
# Order management
# ============================================================


class TestOrderManagement:
    """Cancel and refresh tracked orders."""

    async def test_cancel_tracked_order(self):
        """Cancel a tracked order."""
        client = _mock_client()
        async def _cancel(oid):
            o = Order(order_id=oid, contract_id="C-1", side=Side.YES, price_cents=50, quantity=1)
            o.status = OrderStatus.CANCELLED
            return o
        client.cancel_order.side_effect = _cancel

        executor = TradeExecutor(client, ExecutorConfig(max_retries=0))

        signal = _signal()
        order = await executor.submit_signal(signal)
        result = await executor.cancel_order(order.order_id)

        assert result is not None
        assert result.status == OrderStatus.CANCELLED

    async def test_cancel_untracked_order(self):
        """Cancelling an untracked order returns None."""
        client = _mock_client()
        executor = TradeExecutor(client)

        result = await executor.cancel_order("nonexistent")
        assert result is None

    async def test_get_active_orders(self):
        """List only active (resting) orders."""
        client = _mock_client()
        async def _place_resting(order):
            order.status = OrderStatus.OPEN
            return order
        client.place_order.side_effect = _place_resting
        executor = TradeExecutor(client, ExecutorConfig(max_retries=0, use_batch_api=False))

        signals = [_signal(f"C-{i}") for i in range(3)]
        await executor.submit_signals(signals)

        active = executor.get_active_orders()
        assert len(active) == 3


# ============================================================
# Stats
# ============================================================


class TestStats:
    """Executor statistics tracking."""

    async def test_stats_tracking(self):
        """Stats correctly count submitted, filled, rejected."""
        client = _mock_client()
        call_count = 0
        async def _alternate(order):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise OrderRejectedError("Nope")
            return _filled_order(order)
        client.place_order.side_effect = _alternate

        executor = TradeExecutor(client, ExecutorConfig(max_retries=0, use_batch_api=False))

        signals = [_signal(f"C-{i}") for i in range(4)]
        await executor.submit_signals(signals)

        stats = executor.stats
        assert stats["submitted"] == 4
        assert stats["filled"] == 2
        assert stats["rejected"] == 2
