"""Tests for kill switch — Layer 6 (nuclear option) of risk hierarchy."""

from __future__ import annotations

import pytest

from py.risk.kill_switch import KillSwitch, KillSwitchConfig, KillSwitchEvent


class TestKillSwitchInit:
    """Kill switch initialization and defaults."""

    def test_initial_state(self):
        ks = KillSwitch(initial_equity_cents=100_000)
        assert not ks.is_triggered
        assert ks.trigger_event is None
        assert ks.events == []
        assert ks.daily_pnl_pct == 0.0
        assert ks.drawdown_pct == 0.0

    def test_custom_config(self):
        config = KillSwitchConfig(max_daily_loss_pct=0.05, max_drawdown_pct=0.15)
        ks = KillSwitch(initial_equity_cents=100_000, config=config)
        assert ks.config.max_daily_loss_pct == 0.05
        assert ks.config.max_drawdown_pct == 0.15

    def test_status_dict(self):
        ks = KillSwitch(initial_equity_cents=100_000)
        status = ks.status()
        assert status["triggered"] is False
        assert status["current_equity_cents"] == 100_000
        assert status["peak_equity_cents"] == 100_000
        assert status["consecutive_errors"] == 0


# ============================================================
# Daily loss trigger
# ============================================================


class TestDailyLoss:
    """Kill switch triggers on daily loss exceeding threshold."""

    def test_small_loss_no_trigger(self):
        """5% loss should not trigger at 10% threshold."""
        ks = KillSwitch(initial_equity_cents=100_000)
        event = ks.update_equity(95_000)
        assert event is None
        assert not ks.is_triggered
        assert ks.daily_pnl_pct == pytest.approx(-0.05)

    def test_exact_threshold_triggers(self):
        """Exactly 10% loss triggers the kill switch."""
        ks = KillSwitch(initial_equity_cents=100_000)
        event = ks.update_equity(90_000)
        assert event is not None
        assert ks.is_triggered
        assert event.reason == "daily_loss"
        assert event.equity_at_trigger_cents == 90_000

    def test_beyond_threshold_triggers(self):
        """15% loss triggers kill switch."""
        ks = KillSwitch(initial_equity_cents=100_000)
        event = ks.update_equity(85_000)
        assert event is not None
        assert ks.is_triggered

    def test_no_double_fire(self):
        """Once triggered, subsequent updates don't fire again."""
        ks = KillSwitch(initial_equity_cents=100_000)
        event1 = ks.update_equity(90_000)
        event2 = ks.update_equity(80_000)
        assert event1 is not None
        assert event2 is None  # Already triggered
        assert len(ks.events) == 1

    def test_profit_no_trigger(self):
        """Positive equity movement doesn't trigger."""
        ks = KillSwitch(initial_equity_cents=100_000)
        event = ks.update_equity(110_000)
        assert event is None
        assert not ks.is_triggered
        assert ks.daily_pnl_pct == pytest.approx(0.10)


# ============================================================
# Drawdown trigger
# ============================================================


class TestDrawdown:
    """Kill switch triggers on peak-to-trough drawdown."""

    def test_no_drawdown_from_start(self):
        """No drawdown when equity stays flat."""
        ks = KillSwitch(initial_equity_cents=100_000)
        ks.update_equity(100_000)
        assert ks.drawdown_pct == 0.0

    def test_drawdown_below_threshold(self):
        """15% drawdown from peak doesn't trigger at 20% threshold."""
        ks = KillSwitch(initial_equity_cents=100_000)
        # Run up to peak
        ks.update_equity(120_000)  # New peak
        # Drop 15%: 120,000 → 102,000
        event = ks.update_equity(102_000)
        assert event is None
        assert not ks.is_triggered
        assert ks.drawdown_pct == pytest.approx(0.15)

    def test_drawdown_triggers(self):
        """20% drawdown from peak triggers kill switch."""
        ks = KillSwitch(initial_equity_cents=100_000)
        ks.update_equity(120_000)  # Peak
        # Drop 20%: 120,000 → 96,000
        event = ks.update_equity(96_000)
        assert event is not None
        assert ks.is_triggered
        assert event.reason == "drawdown"
        assert event.peak_equity_cents == 120_000

    def test_peak_updates_correctly(self):
        """Peak equity tracks the highest observed value."""
        ks = KillSwitch(initial_equity_cents=100_000)
        ks.update_equity(110_000)
        ks.update_equity(105_000)  # Dip
        ks.update_equity(115_000)  # New peak
        status = ks.status()
        assert status["peak_equity_cents"] == 115_000


# ============================================================
# Consecutive error trigger
# ============================================================


class TestConsecutiveErrors:
    """Kill switch triggers after N consecutive API errors."""

    def test_single_error_no_trigger(self):
        """One error is not a problem."""
        ks = KillSwitch(initial_equity_cents=100_000)
        event = ks.record_error()
        assert event is None
        assert not ks.is_triggered

    def test_errors_reset_on_success(self):
        """A successful call resets the error counter."""
        ks = KillSwitch(initial_equity_cents=100_000)
        for _ in range(9):
            ks.record_error()
        ks.record_success()  # Resets counter
        event = ks.record_error()  # Back to 1
        assert event is None

    def test_consecutive_errors_trigger(self):
        """10 consecutive errors triggers kill switch."""
        ks = KillSwitch(initial_equity_cents=100_000)
        for i in range(9):
            event = ks.record_error()
            assert event is None
        event = ks.record_error()  # 10th error
        assert event is not None
        assert ks.is_triggered
        assert event.reason == "consecutive_errors"

    def test_custom_error_threshold(self):
        """Custom error threshold is respected."""
        config = KillSwitchConfig(max_consecutive_errors=3)
        ks = KillSwitch(initial_equity_cents=100_000, config=config)
        ks.record_error()
        ks.record_error()
        event = ks.record_error()
        assert event is not None
        assert ks.is_triggered


# ============================================================
# Manual trigger and reset
# ============================================================


class TestManualTriggerReset:
    """Manual kill switch operations."""

    def test_manual_trigger(self):
        """Manual trigger activates kill switch."""
        ks = KillSwitch(initial_equity_cents=100_000)
        event = ks.manual_trigger("suspicious activity")
        assert ks.is_triggered
        assert "manual" in event.reason
        assert "suspicious activity" in event.reason

    def test_reset_after_trigger(self):
        """Reset re-arms the kill switch."""
        ks = KillSwitch(initial_equity_cents=100_000)
        ks.update_equity(90_000)  # Trigger on daily loss
        assert ks.is_triggered

        ks.reset()
        assert not ks.is_triggered
        assert ks.trigger_event is None

    def test_reset_preserves_history(self):
        """Reset keeps event history for audit trail."""
        ks = KillSwitch(initial_equity_cents=100_000)
        ks.update_equity(90_000)
        ks.reset()
        assert len(ks.events) == 1  # History preserved

    def test_reset_when_not_triggered(self):
        """Reset when not triggered is a no-op (warning logged)."""
        ks = KillSwitch(initial_equity_cents=100_000)
        ks.reset()  # Should not raise
        assert not ks.is_triggered

    def test_new_trading_day_resets_daily(self):
        """New trading day resets daily P&L tracking."""
        ks = KillSwitch(initial_equity_cents=100_000)
        ks.update_equity(95_000)  # 5% loss
        assert ks.daily_pnl_pct == pytest.approx(-0.05)

        ks.new_trading_day()
        assert ks.daily_pnl_pct == 0.0  # Reset


# ============================================================
# Event history
# ============================================================


class TestEventHistory:
    """Kill switch event tracking."""

    def test_event_fields(self):
        """KillSwitchEvent captures full context."""
        ks = KillSwitch(initial_equity_cents=100_000)
        event = ks.update_equity(90_000, positions_open=3, orders_open=5)
        assert event.positions_open == 3
        assert event.orders_open == 5
        assert event.triggered_at is not None

    def test_event_repr(self):
        """Events have a useful repr."""
        ks = KillSwitch(initial_equity_cents=100_000)
        event = ks.update_equity(90_000)
        repr_str = repr(event)
        assert "daily_loss" in repr_str
        assert "equity=$900.00" in repr_str
