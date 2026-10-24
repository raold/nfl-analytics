"""Tests for rolling drawdown monitor — risk analytics beyond the kill switch."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from nfl_analytics.risk.drawdown_monitor import DrawdownConfig, DrawdownMonitor


def _ts(days_offset: float = 0) -> datetime:
    """Create a timestamp with a days offset from a fixed base."""
    base = datetime(2025, 1, 1, tzinfo=UTC)
    return base + timedelta(days=days_offset)


# ============================================================
# Initialization
# ============================================================


class TestDrawdownInit:
    def test_initial_state(self):
        mon = DrawdownMonitor(initial_equity_cents=100_000)
        assert mon.current_equity_cents == 100_000
        assert mon.peak_equity_cents == 100_000
        assert mon.current_drawdown_pct == 0.0
        assert mon.max_drawdown_pct == 0.0
        assert not mon.in_drawdown

    def test_custom_config(self):
        config = DrawdownConfig(rolling_window_days=60, sharpe_window_days=60)
        mon = DrawdownMonitor(initial_equity_cents=50_000, config=config)
        assert mon.config.rolling_window_days == 60
        assert mon.config.sharpe_window_days == 60

    def test_initial_snapshot(self):
        mon = DrawdownMonitor(initial_equity_cents=100_000)
        assert len(mon._snapshots) == 1
        assert mon._snapshots[0].equity_cents == 100_000


# ============================================================
# Equity recording
# ============================================================


class TestEquityRecording:
    def test_rising_equity(self):
        mon = DrawdownMonitor(initial_equity_cents=100_000)
        mon.record_equity(110_000, timestamp=_ts(1))
        assert mon.peak_equity_cents == 110_000
        assert mon.current_drawdown_pct == 0.0
        assert not mon.in_drawdown

    def test_falling_creates_drawdown(self):
        mon = DrawdownMonitor(initial_equity_cents=100_000)
        mon.record_equity(90_000, timestamp=_ts(1))
        assert mon.current_drawdown_pct == pytest.approx(0.10)
        assert mon.in_drawdown
        assert mon.current_event is not None
        assert mon.current_event.trough_equity_cents == 90_000

    def test_recovery_closes_event(self):
        mon = DrawdownMonitor(initial_equity_cents=100_000)
        mon.record_equity(90_000, timestamp=_ts(1))
        mon.record_equity(100_000, timestamp=_ts(2))
        assert not mon.in_drawdown
        assert mon.current_event is None
        assert len(mon.events) == 1
        assert mon.events[0].recovered is True

    def test_multiple_dips(self):
        mon = DrawdownMonitor(initial_equity_cents=100_000)
        # First dip
        mon.record_equity(95_000, timestamp=_ts(1))
        mon.record_equity(100_000, timestamp=_ts(2))
        # Second dip (deeper)
        mon.record_equity(85_000, timestamp=_ts(3))
        mon.record_equity(100_000, timestamp=_ts(4))
        assert len(mon.events) == 2
        assert mon.max_drawdown_pct == pytest.approx(0.15)

    def test_snapshot_trimming(self):
        """Old snapshots beyond the rolling window should be trimmed."""
        config = DrawdownConfig(rolling_window_days=2, sharpe_window_days=2)
        mon = DrawdownMonitor(initial_equity_cents=100_000, config=config)
        # Add more snapshots than the trim threshold (3 days × 100 = 300)
        for i in range(400):
            mon.record_equity(100_000 + i, timestamp=_ts(i / 100.0))
        # Should have been trimmed
        assert len(mon._snapshots) <= 301  # 3 days × 100 + 1

    def test_explicit_timestamp(self):
        mon = DrawdownMonitor(initial_equity_cents=100_000)
        ts = datetime(2025, 6, 15, 12, 0, tzinfo=UTC)
        mon.record_equity(95_000, timestamp=ts)
        assert mon._snapshots[-1].timestamp == ts


# ============================================================
# Drawdown metrics
# ============================================================


class TestDrawdownMetrics:
    def test_current_drawdown_pct(self):
        mon = DrawdownMonitor(initial_equity_cents=100_000)
        mon.record_equity(80_000, timestamp=_ts(1))
        assert mon.current_drawdown_pct == pytest.approx(0.20)

    def test_at_peak_is_zero(self):
        mon = DrawdownMonitor(initial_equity_cents=100_000)
        mon.record_equity(110_000, timestamp=_ts(1))
        assert mon.current_drawdown_pct == 0.0

    def test_max_drawdown(self):
        mon = DrawdownMonitor(initial_equity_cents=100_000)
        mon.record_equity(85_000, timestamp=_ts(1))
        mon.record_equity(105_000, timestamp=_ts(2))
        mon.record_equity(95_000, timestamp=_ts(3))
        # Max DD was 15% (100k → 85k), not 9.5% (105k → 95k)
        assert mon.max_drawdown_pct == pytest.approx(0.15)

    def test_in_drawdown(self):
        mon = DrawdownMonitor(initial_equity_cents=100_000)
        assert not mon.in_drawdown
        mon.record_equity(99_999, timestamp=_ts(1))
        assert mon.in_drawdown

    def test_time_in_drawdown(self):
        """time_in_drawdown_days should be zero when at peak."""
        mon = DrawdownMonitor(initial_equity_cents=100_000)
        assert mon.time_in_drawdown_days == 0.0

    def test_recovery_factor(self):
        """Recovery factor = total_return / max_drawdown."""
        mon = DrawdownMonitor(initial_equity_cents=100_000)
        mon.record_equity(80_000, timestamp=_ts(1))  # 20% DD
        mon.record_equity(120_000, timestamp=_ts(2))  # 20% total return
        # recovery_factor = 0.20 / 0.20 = 1.0
        assert mon.recovery_factor == pytest.approx(1.0)


# ============================================================
# Daily returns and Sharpe/Sortino
# ============================================================


class TestDailyReturns:
    def test_day_change_computes_return(self):
        mon = DrawdownMonitor(initial_equity_cents=100_000)
        # Day 1: equity at 105k
        mon.record_equity(105_000, timestamp=_ts(1))
        assert len(mon._daily_returns) == 1
        assert mon._daily_returns[0] == pytest.approx(0.05)

    def test_sharpe_insufficient_data(self):
        mon = DrawdownMonitor(initial_equity_cents=100_000)
        mon.record_equity(105_000, timestamp=_ts(1))
        assert mon.sharpe_ratio is None  # Need >= 5 returns

    def test_sharpe_computation(self):
        """Sharpe should return a float when given enough data."""
        mon = DrawdownMonitor(initial_equity_cents=100_000)
        # Generate 6 daily returns (alternating up/down)
        equity = 100_000
        for day in range(1, 7):
            change = 1000 if day % 2 == 1 else -500
            equity += change
            mon.record_equity(equity, timestamp=_ts(day))
        assert mon.sharpe_ratio is not None
        assert isinstance(mon.sharpe_ratio, float)

    def test_sortino_computation(self):
        """Sortino should return a float when given enough data."""
        mon = DrawdownMonitor(initial_equity_cents=100_000)
        equity = 100_000
        for day in range(1, 7):
            change = 2000 if day % 2 == 1 else -1000
            equity += change
            mon.record_equity(equity, timestamp=_ts(day))
        assert mon.sortino_ratio is not None
        assert isinstance(mon.sortino_ratio, float)


# ============================================================
# Rolling metrics
# ============================================================


class TestRollingMetrics:
    def test_within_window(self):
        """Rolling max DD should capture drawdowns within the window."""
        mon = DrawdownMonitor(initial_equity_cents=100_000)
        mon.record_equity(90_000, timestamp=_ts(1))
        mon.record_equity(100_000, timestamp=_ts(2))
        assert mon.rolling_max_drawdown_pct == pytest.approx(0.10)

    def test_single_snapshot_zero(self):
        """With only the initial snapshot, rolling max DD should be 0."""
        mon = DrawdownMonitor(initial_equity_cents=100_000)
        assert mon.rolling_max_drawdown_pct == 0.0


# ============================================================
# Summary
# ============================================================


class TestSummary:
    def test_all_keys(self):
        mon = DrawdownMonitor(initial_equity_cents=100_000)
        s = mon.summary()
        expected_keys = {
            "current_equity_cents",
            "peak_equity_cents",
            "current_drawdown_pct",
            "max_drawdown_pct",
            "rolling_max_drawdown_pct",
            "in_drawdown",
            "time_in_drawdown_days",
            "sharpe_ratio",
            "sortino_ratio",
            "recovery_factor",
            "completed_drawdowns",
            "snapshots_stored",
            "daily_returns_stored",
        }
        assert set(s.keys()) == expected_keys

    def test_values_consistent(self):
        mon = DrawdownMonitor(initial_equity_cents=100_000)
        mon.record_equity(90_000, timestamp=_ts(1))
        s = mon.summary()
        assert s["current_equity_cents"] == 90_000
        assert s["peak_equity_cents"] == 100_000
        assert s["current_drawdown_pct"] == pytest.approx(0.10, abs=0.001)
        assert s["in_drawdown"] is True
