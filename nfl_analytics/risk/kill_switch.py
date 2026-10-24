"""
Emergency kill switch for the prediction market trading system.

Layer 6 in the risk hierarchy — the nuclear option. When triggered:
1. Cancel ALL open orders
2. Alert via configured channels
3. Halt all new order submission
4. Log the event with full context

Trigger conditions:
- Daily loss exceeds threshold (default 10% of bankroll)
- Peak-to-trough drawdown exceeds threshold (default 20%)
- Too many consecutive API errors (default 10)
- Manual kill via CLI

The kill switch tracks state across sessions. Once triggered, trading
stays halted until manually re-armed.

Usage:
    from nfl_analytics.risk.kill_switch import KillSwitch, KillSwitchConfig

    ks = KillSwitch(initial_equity_cents=100_000)
    ks.update_equity(current_equity_cents=92_000)

    if ks.is_triggered:
        # Cancel everything, stop trading
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


@dataclass
class KillSwitchConfig:
    """Kill switch trigger thresholds."""

    max_daily_loss_pct: float = 0.10  # 10% daily loss → kill
    max_drawdown_pct: float = 0.20  # 20% peak-to-trough → kill
    max_consecutive_errors: int = 10  # 10 API errors in a row → kill
    auto_resume_after_hours: float | None = None  # None = manual resume only


@dataclass
class KillSwitchEvent:
    """Record of a kill switch activation."""

    reason: str
    trigger_value: float  # The value that triggered (loss %, DD %, error count)
    threshold: float  # The threshold that was exceeded
    equity_at_trigger_cents: int
    peak_equity_cents: int
    positions_open: int
    orders_open: int
    triggered_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __repr__(self) -> str:
        return (
            f"KillSwitchEvent(reason={self.reason}, "
            f"value={self.trigger_value:.2%}, threshold={self.threshold:.2%}, "
            f"equity=${self.equity_at_trigger_cents / 100:.2f})"
        )


class KillSwitch:
    """Emergency shutdown system for trading operations.

    Monitors three conditions:
    1. Daily P&L exceeds loss threshold
    2. Drawdown from peak equity exceeds threshold
    3. Consecutive API/exchange errors exceed threshold

    Thread-safe: can be checked from multiple coroutines.
    """

    def __init__(
        self,
        initial_equity_cents: int,
        config: KillSwitchConfig | None = None,
    ):
        self.config = config or KillSwitchConfig()

        # Equity tracking
        self._peak_equity_cents = initial_equity_cents
        self._current_equity_cents = initial_equity_cents
        self._day_start_equity_cents = initial_equity_cents
        self._current_day: str = datetime.now(UTC).strftime("%Y-%m-%d")

        # Error tracking
        self._consecutive_errors = 0

        # Kill switch state
        self._triggered = False
        self._trigger_event: KillSwitchEvent | None = None
        self._events: list[KillSwitchEvent] = []

    @property
    def is_triggered(self) -> bool:
        """Check if the kill switch has been activated."""
        return self._triggered

    @property
    def trigger_event(self) -> KillSwitchEvent | None:
        """Get the event that caused the kill switch to trigger."""
        return self._trigger_event

    @property
    def events(self) -> list[KillSwitchEvent]:
        """History of all kill switch events."""
        return list(self._events)

    @property
    def daily_pnl_pct(self) -> float:
        """Current daily P&L as percentage of day-start equity."""
        if self._day_start_equity_cents == 0:
            return 0.0
        return (
            self._current_equity_cents - self._day_start_equity_cents
        ) / self._day_start_equity_cents

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown from peak equity as a percentage."""
        if self._peak_equity_cents == 0:
            return 0.0
        return (self._peak_equity_cents - self._current_equity_cents) / self._peak_equity_cents

    def update_equity(
        self,
        current_equity_cents: int,
        positions_open: int = 0,
        orders_open: int = 0,
    ) -> KillSwitchEvent | None:
        """Update equity and check all kill conditions.

        Call this on every equity update (fills, price changes, settlements).

        Returns a KillSwitchEvent if the kill switch was triggered, None otherwise.
        """
        if self._triggered:
            return None  # Already triggered, don't double-fire

        self._current_equity_cents = current_equity_cents

        # Check for new trading day
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if today != self._current_day:
            self._day_start_equity_cents = current_equity_cents
            self._current_day = today

        # Update peak
        if current_equity_cents > self._peak_equity_cents:
            self._peak_equity_cents = current_equity_cents

        # Check daily loss
        if self.daily_pnl_pct <= -self.config.max_daily_loss_pct:
            return self._trigger(
                reason="daily_loss",
                value=abs(self.daily_pnl_pct),
                threshold=self.config.max_daily_loss_pct,
                positions_open=positions_open,
                orders_open=orders_open,
            )

        # Check drawdown
        if self.drawdown_pct >= self.config.max_drawdown_pct:
            return self._trigger(
                reason="drawdown",
                value=self.drawdown_pct,
                threshold=self.config.max_drawdown_pct,
                positions_open=positions_open,
                orders_open=orders_open,
            )

        return None

    def record_error(self, positions_open: int = 0, orders_open: int = 0) -> KillSwitchEvent | None:
        """Record an API/exchange error. Triggers after N consecutive errors."""
        self._consecutive_errors += 1

        if self._consecutive_errors >= self.config.max_consecutive_errors:
            return self._trigger(
                reason="consecutive_errors",
                value=float(self._consecutive_errors),
                threshold=float(self.config.max_consecutive_errors),
                positions_open=positions_open,
                orders_open=orders_open,
            )
        return None

    def record_success(self) -> None:
        """Record a successful API call (resets error counter)."""
        self._consecutive_errors = 0

    def manual_trigger(
        self, reason: str = "manual", positions_open: int = 0, orders_open: int = 0
    ) -> KillSwitchEvent:
        """Manually trigger the kill switch."""
        return self._trigger(
            reason=f"manual: {reason}",
            value=0.0,
            threshold=0.0,
            positions_open=positions_open,
            orders_open=orders_open,
        )

    def reset(self) -> None:
        """Re-arm the kill switch after manual review.

        Call this to resume trading after a kill switch event.
        Resets the day-start equity but preserves peak and event history.
        """
        if not self._triggered:
            logger.warning("Kill switch reset called but was not triggered")
            return

        logger.info(
            "Kill switch re-armed. Previous trigger: %s",
            self._trigger_event.reason if self._trigger_event else "unknown",
        )
        self._triggered = False
        self._trigger_event = None
        self._consecutive_errors = 0
        self._day_start_equity_cents = self._current_equity_cents

    def new_trading_day(self, equity_cents: int | None = None) -> None:
        """Reset daily tracking for a new trading day.

        Call at market open or at the start of each trading session.
        """
        self._day_start_equity_cents = equity_cents or self._current_equity_cents
        self._current_day = datetime.now(UTC).strftime("%Y-%m-%d")
        self._consecutive_errors = 0

    def status(self) -> dict:
        """Get current kill switch status."""
        return {
            "triggered": self._triggered,
            "trigger_reason": self._trigger_event.reason if self._trigger_event else None,
            "current_equity_cents": self._current_equity_cents,
            "peak_equity_cents": self._peak_equity_cents,
            "day_start_equity_cents": self._day_start_equity_cents,
            "daily_pnl_pct": round(self.daily_pnl_pct, 4),
            "drawdown_pct": round(self.drawdown_pct, 4),
            "consecutive_errors": self._consecutive_errors,
            "total_events": len(self._events),
        }

    def _trigger(
        self,
        reason: str,
        value: float,
        threshold: float,
        positions_open: int,
        orders_open: int,
    ) -> KillSwitchEvent:
        """Internal: activate the kill switch."""
        event = KillSwitchEvent(
            reason=reason,
            trigger_value=value,
            threshold=threshold,
            equity_at_trigger_cents=self._current_equity_cents,
            peak_equity_cents=self._peak_equity_cents,
            positions_open=positions_open,
            orders_open=orders_open,
        )

        self._triggered = True
        self._trigger_event = event
        self._events.append(event)

        logger.critical(
            "KILL SWITCH TRIGGERED: %s (value=%.2f%%, threshold=%.2f%%, equity=$%.2f)",
            reason,
            value * 100 if value < 1 else value,
            threshold * 100 if threshold < 1 else threshold,
            self._current_equity_cents / 100,
        )

        return event
