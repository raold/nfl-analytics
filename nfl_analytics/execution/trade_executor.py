"""
Trade executor — bridge between strategy signals and market orders.

Manages order submission with retry logic, rate-limit awareness, batch
grouping, and order lifecycle tracking. Strategies submit signals; the
executor converts them to orders and manages their lifecycle.

Features:
    - Retry on transient failures (rate limit, connection errors)
    - Batch grouping: signals arriving within a window are batched
    - Order state tracking: pending → submitted → open/filled/rejected
    - Callbacks for fill/reject events (for risk manager, P&L tracker)
    - Cancel-and-replace for amending orders

Pipeline: Signal → RiskCheck → TradeExecutor → MarketClient → Exchange

Usage:
    from nfl_analytics.execution.trade_executor import TradeExecutor

    executor = TradeExecutor(market_client)
    results = await executor.submit_signals(approved_signals)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Callable

from nfl_analytics.markets.base import (
    ConnectionError as MarketConnectionError,
    Fill,
    MarketClient,
    MarketError,
    Order,
    OrderRejectedError,
    OrderStatus,
    RateLimitError,
)
from nfl_analytics.trading.signal import Signal

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 20  # Kalshi batch limit


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ExecutorConfig:
    """Configuration for the trade executor."""

    max_retries: int = 3
    initial_backoff_ms: int = 200
    batch_window_ms: int = 200  # Group signals within this window
    max_concurrent_orders: int = 5  # Max parallel order submissions
    use_batch_api: bool = True  # Use batch placement when available


# ---------------------------------------------------------------------------
# Order tracking
# ---------------------------------------------------------------------------


@dataclass
class TrackedOrder:
    """An order being managed by the executor."""

    signal: Signal
    order: Order
    submitted_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    attempts: int = 0
    last_error: str = ""


# ---------------------------------------------------------------------------
# Trade executor
# ---------------------------------------------------------------------------


class TradeExecutor:
    """Manages order submission from signals to exchange orders.

    The executor is the final step before orders hit the exchange. It
    handles retry logic, batching, and order lifecycle tracking.
    """

    def __init__(
        self,
        client: MarketClient,
        config: ExecutorConfig | None = None,
    ):
        self._client = client
        self._config = config or ExecutorConfig()

        # Active order tracking
        self._tracked: dict[str, TrackedOrder] = {}  # order_id → TrackedOrder
        self._submitted_count = 0
        self._filled_count = 0
        self._rejected_count = 0

        # Callbacks
        self.on_fill: Callable[[Order, Signal], None] | None = None
        self.on_reject: Callable[[Order, Signal, str], None] | None = None

        # Concurrency control
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent_orders)

    # -------------------------------------------------------------------
    # Signal submission
    # -------------------------------------------------------------------

    async def submit_signals(self, signals: list[Signal]) -> list[Order]:
        """Convert signals to orders and submit them.

        If batch API is enabled and multiple signals arrive together,
        they're grouped into a single batch request.

        Returns list of submitted orders (may include rejected ones).
        """
        if not signals:
            return []

        # Convert signals to orders
        orders = [sig.to_order() for sig in signals]

        # Track each order
        for sig, order in zip(signals, orders):
            self._tracked[order.order_id] = TrackedOrder(signal=sig, order=order)

        # Submit
        if self._config.use_batch_api and len(orders) > 1:
            results = await self._submit_batch(orders, signals)
        else:
            results = await self._submit_individual(orders, signals)

        return results

    async def submit_signal(self, signal: Signal) -> Order:
        """Submit a single signal. Convenience wrapper."""
        results = await self.submit_signals([signal])
        return results[0]

    # -------------------------------------------------------------------
    # Order management
    # -------------------------------------------------------------------

    async def cancel_order(self, order_id: str) -> Order | None:
        """Cancel a tracked order."""
        tracked = self._tracked.get(order_id)
        if not tracked:
            logger.warning("Order %s not tracked by executor", order_id)
            return None

        try:
            result = await self._client.cancel_order(order_id)
            tracked.order = result
            logger.info("Cancelled order %s", order_id[:8])
            return result
        except MarketError as e:
            logger.warning("Failed to cancel order %s: %s", order_id[:8], e)
            return None

    async def cancel_all(self, contract_id: str | None = None) -> int:
        """Cancel all active orders, optionally filtered by contract."""
        return await self._client.cancel_all_orders(contract_id)

    async def refresh_order(self, order_id: str) -> Order | None:
        """Refresh order status from exchange."""
        tracked = self._tracked.get(order_id)
        if not tracked:
            return None

        try:
            result = await self._client.get_order(order_id)
            tracked.order = result
            return result
        except MarketError as e:
            logger.warning("Failed to refresh order %s: %s", order_id[:8], e)
            return None

    # -------------------------------------------------------------------
    # Internal submission
    # -------------------------------------------------------------------

    async def _submit_batch(self, orders: list[Order], signals: list[Signal]) -> list[Order]:
        """Submit orders using the batch API."""
        results: list[Order] = []

        for i in range(0, len(orders), MAX_BATCH_SIZE):
            chunk_orders = orders[i : i + MAX_BATCH_SIZE]
            chunk_signals = signals[i : i + MAX_BATCH_SIZE]

            for attempt in range(self._config.max_retries + 1):
                try:
                    batch_results = await self._client.batch_place_orders(chunk_orders)
                    for result, sig in zip(batch_results, chunk_signals):
                        self._handle_result(result, sig)
                    results.extend(batch_results)
                    break
                except RateLimitError:
                    if attempt < self._config.max_retries:
                        backoff = self._backoff_ms(attempt, base_ms=500)
                        logger.warning("Rate limited on batch (attempt %d), backing off %dms", attempt + 1, backoff)
                        await asyncio.sleep(backoff / 1000)
                    else:
                        # Mark all as rejected
                        for order, sig in zip(chunk_orders, chunk_signals):
                            order.status = OrderStatus.REJECTED
                            self._handle_result(order, sig)
                        results.extend(chunk_orders)
                except MarketConnectionError:
                    if attempt < self._config.max_retries:
                        backoff = self._backoff_ms(attempt)
                        logger.warning("Connection error on batch (attempt %d), retrying in %dms", attempt + 1, backoff)
                        await asyncio.sleep(backoff / 1000)
                    else:
                        for order, sig in zip(chunk_orders, chunk_signals):
                            order.status = OrderStatus.REJECTED
                            self._handle_result(order, sig)
                        results.extend(chunk_orders)

        return results

    async def _submit_individual(self, orders: list[Order], signals: list[Signal]) -> list[Order]:
        """Submit orders individually with concurrency control."""
        async def _submit_one(order: Order, signal: Signal) -> Order:
            async with self._semaphore:
                return await self._submit_with_retry(order, signal)

        tasks = [_submit_one(o, s) for o, s in zip(orders, signals)]
        return list(await asyncio.gather(*tasks))

    async def _submit_with_retry(self, order: Order, signal: Signal) -> Order:
        """Submit a single order with retry on transient failures."""
        tracked = self._tracked.get(order.order_id)

        for attempt in range(self._config.max_retries + 1):
            if tracked:
                tracked.attempts = attempt + 1

            try:
                result = await self._client.place_order(order)
                self._handle_result(result, signal)
                return result

            except RateLimitError:
                if attempt < self._config.max_retries:
                    backoff = self._backoff_ms(attempt, base_ms=500)
                    logger.warning("Rate limited (attempt %d), backing off %dms", attempt + 1, backoff)
                    await asyncio.sleep(backoff / 1000)
                else:
                    order.status = OrderStatus.REJECTED
                    if tracked:
                        tracked.last_error = "Rate limit exhausted"
                    self._handle_result(order, signal)

            except MarketConnectionError as e:
                if attempt < self._config.max_retries:
                    backoff = self._backoff_ms(attempt)
                    await asyncio.sleep(backoff / 1000)
                else:
                    order.status = OrderStatus.REJECTED
                    if tracked:
                        tracked.last_error = str(e)
                    self._handle_result(order, signal)

            except OrderRejectedError as e:
                # Don't retry client errors
                order.status = OrderStatus.REJECTED
                if tracked:
                    tracked.last_error = str(e)
                self._handle_result(order, signal)
                return order

            except MarketError as e:
                if attempt < self._config.max_retries:
                    backoff = self._backoff_ms(attempt)
                    await asyncio.sleep(backoff / 1000)
                else:
                    order.status = OrderStatus.REJECTED
                    if tracked:
                        tracked.last_error = str(e)
                    self._handle_result(order, signal)

        return order

    def _handle_result(self, order: Order, signal: Signal) -> None:
        """Process order result — update stats and invoke callbacks."""
        self._submitted_count += 1

        if order.status == OrderStatus.FILLED:
            self._filled_count += 1
            if self.on_fill:
                self.on_fill(order, signal)
            logger.info(
                "Filled: %s %dx %s @%d¢ (edge=%d¢)",
                signal.side.value,
                order.filled_quantity,
                signal.contract_id[:20],
                order.avg_fill_price_cents,
                signal.edge_cents,
            )

        elif order.status == OrderStatus.REJECTED:
            self._rejected_count += 1
            tracked = self._tracked.get(order.order_id)
            reason = tracked.last_error if tracked else "unknown"
            if self.on_reject:
                self.on_reject(order, signal, reason)
            logger.warning(
                "Rejected: %s %dx %s @%d¢ — %s",
                signal.side.value,
                signal.quantity,
                signal.contract_id[:20],
                signal.price_cents,
                reason,
            )

        elif order.status in (OrderStatus.OPEN, OrderStatus.PARTIAL):
            logger.info(
                "Resting: %s %dx %s @%d¢",
                signal.side.value,
                signal.quantity,
                signal.contract_id[:20],
                signal.price_cents,
            )

    def _backoff_ms(self, attempt: int, base_ms: int | None = None) -> int:
        """Exponential backoff with jitter."""
        import random
        base = base_ms or self._config.initial_backoff_ms
        delay = base * (2 ** attempt)
        jitter = random.randint(0, delay // 2)
        return delay + jitter

    # -------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        """Executor statistics."""
        active = sum(
            1 for t in self._tracked.values()
            if t.order.is_active
        )
        return {
            "submitted": self._submitted_count,
            "filled": self._filled_count,
            "rejected": self._rejected_count,
            "active_orders": active,
            "tracked_total": len(self._tracked),
        }

    def get_tracked_order(self, order_id: str) -> TrackedOrder | None:
        return self._tracked.get(order_id)

    def get_active_orders(self) -> list[TrackedOrder]:
        return [t for t in self._tracked.values() if t.order.is_active]
