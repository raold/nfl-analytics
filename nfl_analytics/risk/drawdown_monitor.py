"""
Rolling drawdown monitor for prediction market trading.

Tracks equity over time and computes drawdown metrics that are richer
than the kill switch's simple peak-to-trough check. The kill switch
says "stop trading NOW." The drawdown monitor says "here's your risk
profile over time."

Metrics provided:
- Current drawdown (peak-to-trough)
- Maximum drawdown (worst ever)
- Rolling N-day max drawdown
- Time in drawdown (days since last peak)
- Recovery factor (gain from trough / max drawdown)
- Daily P&L series for Sharpe/Sortino computation

Usage:
    from py.risk.drawdown_monitor import DrawdownMonitor, DrawdownConfig

    monitor = DrawdownMonitor(initial_equity_cents=100_000)
    monitor.record_equity(98_000)   # Lost $20
    monitor.record_equity(102_000)  # Recovered to new peak

    print(monitor.max_drawdown_pct)  # 0.02 (2%)
    print(monitor.current_drawdown_pct)  # 0.0 (recovered)
    print(monitor.sharpe_ratio)  # Rolling Sharpe
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


@dataclass
class DrawdownConfig:
    """Configuration for drawdown monitoring."""

    rolling_window_days: int = 30  # Rolling window for max drawdown
    sharpe_window_days: int = 30  # Rolling window for Sharpe ratio
    risk_free_annual_pct: float = 0.05  # Risk-free rate for Sharpe (5% T-bill)


@dataclass
class DrawdownEvent:
    """Record of a drawdown period."""

    peak_equity_cents: int
    trough_equity_cents: int
    drawdown_pct: float
    peak_date: datetime
    trough_date: datetime
    recovered: bool = False
    recovery_date: datetime | None = None

    @property
    def duration_days(self) -> float:
        """Days from peak to trough (or to now if not recovered)."""
        end = self.recovery_date or self.trough_date
        return (end - self.peak_date).total_seconds() / 86400.0

    @property
    def depth_cents(self) -> int:
        return self.peak_equity_cents - self.trough_equity_cents


@dataclass
class EquitySnapshot:
    """Single equity observation."""

    equity_cents: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class DrawdownMonitor:
    """Tracks equity series and computes drawdown analytics.

    Records equity snapshots over time and maintains:
    - Peak/trough tracking with event history
    - Rolling window max drawdown
    - Daily return series for Sharpe/Sortino
    - Time-in-drawdown counter
    """

    def __init__(
        self,
        initial_equity_cents: int,
        config: DrawdownConfig | None = None,
    ):
        self.config = config or DrawdownConfig()

        # Equity series
        self._snapshots: deque[EquitySnapshot] = deque()
        self._snapshots.append(EquitySnapshot(equity_cents=initial_equity_cents))

        # Peak tracking
        self._peak_equity_cents = initial_equity_cents
        self._peak_date = datetime.now(UTC)

        # Trough tracking (within current drawdown period)
        self._trough_equity_cents = initial_equity_cents
        self._trough_date = datetime.now(UTC)

        # Max drawdown ever
        self._max_drawdown_pct: float = 0.0
        self._max_drawdown_event: DrawdownEvent | None = None

        # Current drawdown event (None if at peak)
        self._current_event: DrawdownEvent | None = None

        # History of completed drawdown events
        self._events: list[DrawdownEvent] = []

        # Daily P&L for Sharpe (deque capped at window size)
        self._daily_returns: deque[float] = deque(maxlen=self.config.sharpe_window_days)
        self._last_day_equity: int = initial_equity_cents
        self._last_day: str = datetime.now(UTC).strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def record_equity(
        self, equity_cents: int, timestamp: datetime | None = None
    ) -> None:
        """Record an equity snapshot.

        Call this on every fill, settlement, or periodic equity update.
        Updates drawdown tracking and daily return series.
        """
        ts = timestamp or datetime.now(UTC)
        self._snapshots.append(EquitySnapshot(equity_cents=equity_cents, timestamp=ts))

        # Trim old snapshots beyond rolling window
        cutoff_days = max(self.config.rolling_window_days, self.config.sharpe_window_days) + 1
        while len(self._snapshots) > cutoff_days * 100:  # ~100 updates/day max
            self._snapshots.popleft()

        # Daily return tracking
        today = ts.strftime("%Y-%m-%d")
        if today != self._last_day:
            # New day — compute yesterday's return
            if self._last_day_equity > 0:
                daily_return = (equity_cents - self._last_day_equity) / self._last_day_equity
                self._daily_returns.append(daily_return)
            self._last_day_equity = equity_cents
            self._last_day = today

        # Update peak/trough
        if equity_cents >= self._peak_equity_cents:
            # New peak — close any open drawdown event
            if self._current_event is not None:
                self._current_event.recovered = True
                self._current_event.recovery_date = ts
                self._events.append(self._current_event)
                self._current_event = None

            self._peak_equity_cents = equity_cents
            self._peak_date = ts
            self._trough_equity_cents = equity_cents
            self._trough_date = ts
        else:
            # Below peak — update trough if new low
            if equity_cents < self._trough_equity_cents:
                self._trough_equity_cents = equity_cents
                self._trough_date = ts

            # Update current drawdown event
            dd_pct = (self._peak_equity_cents - equity_cents) / self._peak_equity_cents
            if self._current_event is None:
                self._current_event = DrawdownEvent(
                    peak_equity_cents=self._peak_equity_cents,
                    trough_equity_cents=equity_cents,
                    drawdown_pct=dd_pct,
                    peak_date=self._peak_date,
                    trough_date=ts,
                )
            else:
                if equity_cents <= self._current_event.trough_equity_cents:
                    self._current_event.trough_equity_cents = equity_cents
                    self._current_event.drawdown_pct = dd_pct
                    self._current_event.trough_date = ts

            # Update max drawdown
            if dd_pct > self._max_drawdown_pct:
                self._max_drawdown_pct = dd_pct
                self._max_drawdown_event = DrawdownEvent(
                    peak_equity_cents=self._peak_equity_cents,
                    trough_equity_cents=equity_cents,
                    drawdown_pct=dd_pct,
                    peak_date=self._peak_date,
                    trough_date=ts,
                )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_equity_cents(self) -> int:
        """Latest recorded equity."""
        return self._snapshots[-1].equity_cents if self._snapshots else 0

    @property
    def peak_equity_cents(self) -> int:
        return self._peak_equity_cents

    @property
    def current_drawdown_pct(self) -> float:
        """Current drawdown from peak (0.0 = at peak, 0.1 = 10% below)."""
        if self._peak_equity_cents == 0:
            return 0.0
        current = self.current_equity_cents
        return max(0.0, (self._peak_equity_cents - current) / self._peak_equity_cents)

    @property
    def max_drawdown_pct(self) -> float:
        """Maximum drawdown ever observed."""
        return self._max_drawdown_pct

    @property
    def max_drawdown_event(self) -> DrawdownEvent | None:
        """The worst drawdown event."""
        return self._max_drawdown_event

    @property
    def in_drawdown(self) -> bool:
        """Whether we're currently below peak equity."""
        return self.current_equity_cents < self._peak_equity_cents

    @property
    def time_in_drawdown_days(self) -> float:
        """Days since the last peak (0 if at peak)."""
        if not self.in_drawdown:
            return 0.0
        now = datetime.now(UTC)
        return (now - self._peak_date).total_seconds() / 86400.0

    @property
    def events(self) -> list[DrawdownEvent]:
        """History of completed (recovered) drawdown events."""
        return list(self._events)

    @property
    def current_event(self) -> DrawdownEvent | None:
        """Current active drawdown event (None if at peak)."""
        return self._current_event

    # ------------------------------------------------------------------
    # Rolling metrics
    # ------------------------------------------------------------------

    @property
    def rolling_max_drawdown_pct(self) -> float:
        """Maximum drawdown within the rolling window.

        Scans recent snapshots for worst peak-to-trough in the window.
        """
        if len(self._snapshots) < 2:
            return 0.0

        now = self._snapshots[-1].timestamp
        cutoff = now.timestamp() - (self.config.rolling_window_days * 86400)

        # Filter to rolling window
        window = [s for s in self._snapshots if s.timestamp.timestamp() >= cutoff]
        if len(window) < 2:
            return self.current_drawdown_pct

        # Compute max drawdown in window
        peak = window[0].equity_cents
        max_dd = 0.0
        for snap in window:
            if snap.equity_cents > peak:
                peak = snap.equity_cents
            elif peak > 0:
                dd = (peak - snap.equity_cents) / peak
                max_dd = max(max_dd, dd)

        return max_dd

    @property
    def sharpe_ratio(self) -> float | None:
        """Rolling annualized Sharpe ratio.

        Returns None if insufficient data (< 5 daily returns).
        """
        if len(self._daily_returns) < 5:
            return None

        returns = list(self._daily_returns)
        mean_daily = sum(returns) / len(returns)
        variance = sum((r - mean_daily) ** 2 for r in returns) / len(returns)
        std_daily = math.sqrt(variance) if variance > 0 else 0.0

        if std_daily == 0:
            return None

        # Annualize (252 trading days)
        daily_rf = self.config.risk_free_annual_pct / 252.0
        excess = mean_daily - daily_rf
        return (excess / std_daily) * math.sqrt(252)

    @property
    def sortino_ratio(self) -> float | None:
        """Rolling annualized Sortino ratio (downside deviation only).

        Returns None if insufficient data.
        """
        if len(self._daily_returns) < 5:
            return None

        returns = list(self._daily_returns)
        mean_daily = sum(returns) / len(returns)

        # Downside deviation: std of negative returns only
        downside = [min(0, r) ** 2 for r in returns]
        downside_var = sum(downside) / len(downside)
        downside_std = math.sqrt(downside_var) if downside_var > 0 else 0.0

        if downside_std == 0:
            return None

        daily_rf = self.config.risk_free_annual_pct / 252.0
        excess = mean_daily - daily_rf
        return (excess / downside_std) * math.sqrt(252)

    @property
    def recovery_factor(self) -> float | None:
        """Total return / max drawdown. Higher = better recovery ability.

        Returns None if no drawdown has occurred.
        """
        if self._max_drawdown_pct == 0 or len(self._snapshots) < 2:
            return None

        first = self._snapshots[0].equity_cents
        current = self.current_equity_cents
        total_return_pct = (current - first) / first if first > 0 else 0.0

        return total_return_pct / self._max_drawdown_pct

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Get comprehensive drawdown summary."""
        return {
            "current_equity_cents": self.current_equity_cents,
            "peak_equity_cents": self._peak_equity_cents,
            "current_drawdown_pct": round(self.current_drawdown_pct, 4),
            "max_drawdown_pct": round(self._max_drawdown_pct, 4),
            "rolling_max_drawdown_pct": round(self.rolling_max_drawdown_pct, 4),
            "in_drawdown": self.in_drawdown,
            "time_in_drawdown_days": round(self.time_in_drawdown_days, 1),
            "sharpe_ratio": round(self.sharpe_ratio, 2) if self.sharpe_ratio is not None else None,
            "sortino_ratio": round(self.sortino_ratio, 2) if self.sortino_ratio is not None else None,
            "recovery_factor": round(self.recovery_factor, 2) if self.recovery_factor is not None else None,
            "completed_drawdowns": len(self._events),
            "snapshots_stored": len(self._snapshots),
            "daily_returns_stored": len(self._daily_returns),
        }
