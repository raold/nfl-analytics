"""
Real-time P&L tracking for prediction market trading.

Aggregates realized and unrealized P&L from fills, positions, and
settlements. Uses FIFO matching for realized P&L on closing fills.

All monetary values are in cents to match the rest of the system.

Usage:
    from nfl_analytics.monitoring.pnl_tracker import PnLTracker, PnLConfig

    tracker = PnLTracker(initial_equity_cents=100_000)
    tracker.record_fill("KXNFL-KC", Side.YES, 62, 10)   # Buy 10 @ 62¢
    tracker.record_fill("KXNFL-KC", Side.NO, 55, 10)     # Close (sell YES)
    print(tracker.realized_pnl_cents)  # Profit from round trip
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

from nfl_analytics.markets.base import Position, Side

logger = logging.getLogger(__name__)


@dataclass
class PnLConfig:
    """Configuration for P&L tracking."""

    snapshot_interval_seconds: int = 60


@dataclass
class _PositionState:
    """Internal FIFO position state for a single contract.

    Tracks the side we're holding, total quantity, and a FIFO queue
    of (price_cents, quantity) tuples for realized P&L matching.
    """

    side: Side
    quantity: int
    fills_fifo: deque[tuple[int, int]] = field(default_factory=deque)


class PnLTracker:
    """Real-time P&L aggregation from fills, positions, and settlements.

    Tracks:
    - Cash balance (reduced by buys, increased by sells)
    - Per-contract FIFO position state for realized P&L
    - Unrealized P&L from open positions at current prices
    - Daily P&L with new_trading_day() resets
    - Trade count and win rate
    """

    def __init__(
        self,
        initial_equity_cents: int,
        config: PnLConfig | None = None,
    ):
        self.config = config or PnLConfig()

        # Cash tracking
        self._initial_equity_cents = initial_equity_cents
        self._cash_cents = initial_equity_cents

        # Per-contract FIFO position state
        self._positions: dict[str, _PositionState] = {}

        # Realized P&L
        self._realized_pnl_cents = 0

        # Unrealized P&L (updated externally via update_positions)
        self._unrealized_pnl_cents = 0

        # Daily tracking
        self._day_start_equity_cents = initial_equity_cents

        # Trade stats (a "trade" = a closing fill that realizes P&L)
        self._trade_count = 0
        self._winning_trades = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def record_fill(
        self,
        contract_id: str,
        side: Side,
        price_cents: int,
        quantity: int,
    ) -> None:
        """Record a fill and update cash and realized P&L.

        Buy YES: cash decreases by price × qty.
        Close YES (sell via buying NO): FIFO match against open YES fills,
        compute realized P&L as (100 - entry - (100 - exit)) × qty per lot.

        In Kalshi's model, buying YES at 62 and selling at 70 means:
        - Entry cost: 62 × qty
        - Exit proceeds: 70 × qty (you sell your YES contracts)
        - P&L = (70 - 62) × qty = 8 × qty
        """
        pos = self._positions.get(contract_id)

        if pos is None or pos.quantity == 0:
            # New position — opening fill
            self._cash_cents -= price_cents * quantity
            self._positions[contract_id] = _PositionState(
                side=side,
                quantity=quantity,
                fills_fifo=deque([(price_cents, quantity)]),
            )
            return

        if pos.side == side:
            # Adding to existing position (same side)
            self._cash_cents -= price_cents * quantity
            pos.quantity += quantity
            pos.fills_fifo.append((price_cents, quantity))
            return

        # Closing fill (opposite side) — FIFO matching
        remaining = quantity

        # First, match against existing FIFO queue
        while remaining > 0 and pos.fills_fifo:
            entry_price, entry_qty = pos.fills_fifo[0]

            matched = min(remaining, entry_qty)

            # Realized P&L: sell_price - buy_price per contract
            # If we held YES and are closing: pnl = exit - entry
            # If we held NO and are closing: pnl = exit - entry
            # (the "exit price" is what the other side pays us)
            pnl = (price_cents - entry_price) * matched
            self._realized_pnl_cents += pnl
            self._cash_cents += (entry_price + pnl) * matched  # Return cost basis + pnl

            self._trade_count += 1
            if pnl > 0:
                self._winning_trades += 1

            remaining -= matched
            if matched == entry_qty:
                pos.fills_fifo.popleft()
            else:
                pos.fills_fifo[0] = (entry_price, entry_qty - matched)

        pos.quantity -= quantity - remaining  # Reduce by what we matched

        # If we still have remaining, we flipped sides
        if remaining > 0:
            self._cash_cents -= price_cents * remaining
            pos.side = side
            pos.quantity = remaining
            pos.fills_fifo = deque([(price_cents, remaining)])
        elif pos.quantity <= 0:
            pos.quantity = 0
            pos.fills_fifo.clear()

    def update_positions(
        self,
        positions: list[Position],
        market_prices: dict[str, int],
    ) -> None:
        """Recompute unrealized P&L from current positions and market prices."""
        total_unrealized = 0
        for position in positions:
            price = market_prices.get(position.contract_id)
            if price is not None:
                total_unrealized += position.unrealized_pnl_cents(price)
        self._unrealized_pnl_cents = total_unrealized

    def record_settlement(
        self,
        contract_id: str,
        outcome_yes: bool,
        quantity: int,
        avg_entry_cents: int,
    ) -> None:
        """Record contract settlement.

        YES outcome on YES position: pnl = (100 - entry) × qty
        NO outcome on YES position: pnl = -entry × qty
        """
        if outcome_yes:
            pnl = (100 - avg_entry_cents) * quantity
        else:
            pnl = -avg_entry_cents * quantity

        self._realized_pnl_cents += pnl
        self._cash_cents += avg_entry_cents * quantity + pnl  # Return basis + pnl

        self._trade_count += 1
        if pnl > 0:
            self._winning_trades += 1

        # Remove position state
        if contract_id in self._positions:
            del self._positions[contract_id]

    def new_trading_day(self) -> None:
        """Reset daily tracking. Call at start of each trading session."""
        self._day_start_equity_cents = self.total_equity_cents

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_equity_cents(self) -> int:
        """Total equity = cash + unrealized P&L."""
        return self._cash_cents + self._unrealized_pnl_cents

    @property
    def realized_pnl_cents(self) -> int:
        """Total realized P&L from closed trades."""
        return self._realized_pnl_cents

    @property
    def unrealized_pnl_cents(self) -> int:
        """Unrealized P&L from open positions at current prices."""
        return self._unrealized_pnl_cents

    @property
    def daily_pnl_cents(self) -> int:
        """P&L since the start of the current trading day."""
        return self.total_equity_cents - self._day_start_equity_cents

    @property
    def trade_count(self) -> int:
        """Number of completed (closed) trades."""
        return self._trade_count

    @property
    def win_rate(self) -> float:
        """Win rate of completed trades (0.0 to 1.0)."""
        if self._trade_count == 0:
            return 0.0
        return self._winning_trades / self._trade_count

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Get P&L summary."""
        return {
            "total_equity_cents": self.total_equity_cents,
            "cash_cents": self._cash_cents,
            "realized_pnl_cents": self._realized_pnl_cents,
            "unrealized_pnl_cents": self._unrealized_pnl_cents,
            "daily_pnl_cents": self.daily_pnl_cents,
            "trade_count": self._trade_count,
            "winning_trades": self._winning_trades,
            "win_rate": round(self.win_rate, 4),
            "open_positions": sum(
                1 for p in self._positions.values() if p.quantity > 0
            ),
        }
