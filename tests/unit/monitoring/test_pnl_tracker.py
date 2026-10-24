"""Tests for real-time P&L tracker — FIFO matching, settlements, daily tracking."""

from __future__ import annotations

import pytest

from nfl_analytics.markets.base import Position, Side
from nfl_analytics.monitoring.pnl_tracker import PnLConfig, PnLTracker

# ============================================================
# Initialization
# ============================================================


class TestPnLInit:
    def test_initial_state(self):
        tracker = PnLTracker(initial_equity_cents=100_000)
        assert tracker.total_equity_cents == 100_000
        assert tracker.realized_pnl_cents == 0
        assert tracker.unrealized_pnl_cents == 0
        assert tracker.daily_pnl_cents == 0
        assert tracker.trade_count == 0
        assert tracker.win_rate == 0.0

    def test_custom_config(self):
        config = PnLConfig(snapshot_interval_seconds=30)
        tracker = PnLTracker(initial_equity_cents=50_000, config=config)
        assert tracker.config.snapshot_interval_seconds == 30


# ============================================================
# Fill processing (FIFO matching)
# ============================================================


class TestFillProcessing:
    def test_buy_reduces_cash(self):
        """Buying YES reduces cash by price × quantity."""
        tracker = PnLTracker(initial_equity_cents=100_000)
        tracker.record_fill("KXNFL-KC", Side.YES, 60, 10)
        # Cash: 100000 - 60*10 = 99400
        assert tracker._cash_cents == 99_400

    def test_round_trip_profit(self):
        """Buy YES at 60, close (opposite side) at 70 → +10¢/contract profit."""
        tracker = PnLTracker(initial_equity_cents=100_000)
        tracker.record_fill("KXNFL-KC", Side.YES, 60, 10)  # Buy
        tracker.record_fill("KXNFL-KC", Side.NO, 70, 10)  # Close
        assert tracker.realized_pnl_cents == 100  # (70-60)*10
        assert tracker.trade_count == 1
        assert tracker.win_rate == 1.0

    def test_round_trip_loss(self):
        """Buy YES at 60, close at 50 → -10¢/contract loss."""
        tracker = PnLTracker(initial_equity_cents=100_000)
        tracker.record_fill("KXNFL-KC", Side.YES, 60, 10)
        tracker.record_fill("KXNFL-KC", Side.NO, 50, 10)
        assert tracker.realized_pnl_cents == -100  # (50-60)*10
        assert tracker.trade_count == 1
        assert tracker.win_rate == 0.0

    def test_multiple_fills(self):
        """FIFO: buy 5@60 then 5@65, close 10 — matches oldest first."""
        tracker = PnLTracker(initial_equity_cents=100_000)
        tracker.record_fill("KXNFL-KC", Side.YES, 60, 5)
        tracker.record_fill("KXNFL-KC", Side.YES, 65, 5)
        tracker.record_fill("KXNFL-KC", Side.NO, 70, 10)
        # FIFO: 5@(70-60) + 5@(70-65) = 50 + 25 = 75
        assert tracker.realized_pnl_cents == 75
        assert tracker.trade_count == 2  # Two FIFO lots matched

    def test_independent_contracts(self):
        """Fills on different contracts don't interact."""
        tracker = PnLTracker(initial_equity_cents=100_000)
        tracker.record_fill("KXNFL-KC", Side.YES, 60, 10)
        tracker.record_fill("KXNFL-BUF", Side.YES, 40, 5)
        # KC and BUF are tracked independently
        assert tracker._positions["KXNFL-KC"].quantity == 10
        assert tracker._positions["KXNFL-BUF"].quantity == 5


# ============================================================
# Position updates (unrealized P&L)
# ============================================================


class TestPositionUpdate:
    def test_unrealized_pnl(self):
        """Unrealized P&L reflects current market prices."""
        tracker = PnLTracker(initial_equity_cents=100_000)
        positions = [
            Position(contract_id="KXNFL-KC", side=Side.YES, quantity=10, avg_entry_cents=60)
        ]
        tracker.update_positions(positions, {"KXNFL-KC": 70})
        assert tracker.unrealized_pnl_cents == 100  # (70-60)*10

    def test_total_equity_includes_unrealized(self):
        """total_equity = cash + unrealized."""
        tracker = PnLTracker(initial_equity_cents=100_000)
        tracker.record_fill("KXNFL-KC", Side.YES, 60, 10)  # Cash: 99400
        positions = [
            Position(contract_id="KXNFL-KC", side=Side.YES, quantity=10, avg_entry_cents=60)
        ]
        tracker.update_positions(positions, {"KXNFL-KC": 70})
        # Cash 99400 + unrealized 100 = 99500
        assert tracker.total_equity_cents == 99_500


# ============================================================
# Settlement
# ============================================================


class TestSettlement:
    def test_yes_outcome_yes_position(self):
        """YES outcome: pnl = (100 - entry) × qty."""
        tracker = PnLTracker(initial_equity_cents=100_000)
        tracker.record_settlement("KXNFL-KC", outcome_yes=True, quantity=10, avg_entry_cents=60)
        assert tracker.realized_pnl_cents == 400  # (100-60)*10

    def test_no_outcome_yes_position(self):
        """NO outcome: pnl = -entry × qty (total loss)."""
        tracker = PnLTracker(initial_equity_cents=100_000)
        tracker.record_settlement("KXNFL-KC", outcome_yes=False, quantity=10, avg_entry_cents=60)
        assert tracker.realized_pnl_cents == -600  # -60*10


# ============================================================
# Daily tracking
# ============================================================


class TestDailyTracking:
    def test_daily_pnl(self):
        """daily_pnl_cents tracks change since day start."""
        tracker = PnLTracker(initial_equity_cents=100_000)
        # Buy first (reduces cash), then settle
        tracker.record_fill("KXNFL-KC", Side.YES, 60, 10)  # Cash: 99400
        tracker.record_settlement("KXNFL-KC", outcome_yes=True, quantity=10, avg_entry_cents=60)
        # Settlement adds back 60*10 + (100-60)*10 = 600 + 400 = 1000
        # Cash: 99400 + 1000 = 100400, daily_pnl = 100400 - 100000 = 400
        assert tracker.daily_pnl_cents == 400

    def test_new_trading_day_resets(self):
        """new_trading_day() resets daily_pnl_cents to 0."""
        tracker = PnLTracker(initial_equity_cents=100_000)
        tracker.record_fill("KXNFL-KC", Side.YES, 60, 10)
        tracker.record_settlement("KXNFL-KC", outcome_yes=True, quantity=10, avg_entry_cents=60)
        assert tracker.daily_pnl_cents == 400
        tracker.new_trading_day()
        assert tracker.daily_pnl_cents == 0


# ============================================================
# Win rate
# ============================================================


class TestWinRate:
    def test_no_trades(self):
        tracker = PnLTracker(initial_equity_cents=100_000)
        assert tracker.win_rate == 0.0

    def test_all_winners(self):
        tracker = PnLTracker(initial_equity_cents=100_000)
        tracker.record_fill("A", Side.YES, 50, 5)
        tracker.record_fill("A", Side.NO, 60, 5)  # +50¢
        tracker.record_fill("B", Side.YES, 40, 5)
        tracker.record_fill("B", Side.NO, 55, 5)  # +75¢
        assert tracker.win_rate == 1.0

    def test_mixed(self):
        tracker = PnLTracker(initial_equity_cents=100_000)
        tracker.record_fill("A", Side.YES, 50, 5)
        tracker.record_fill("A", Side.NO, 60, 5)  # Win
        tracker.record_fill("B", Side.YES, 40, 5)
        tracker.record_fill("B", Side.NO, 30, 5)  # Loss
        assert tracker.win_rate == pytest.approx(0.5)


# ============================================================
# Summary
# ============================================================


class TestSummary:
    def test_all_keys(self):
        tracker = PnLTracker(initial_equity_cents=100_000)
        s = tracker.summary()
        expected_keys = {
            "total_equity_cents",
            "cash_cents",
            "realized_pnl_cents",
            "unrealized_pnl_cents",
            "daily_pnl_cents",
            "trade_count",
            "winning_trades",
            "win_rate",
            "open_positions",
        }
        assert set(s.keys()) == expected_keys
