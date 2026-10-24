"""Tests for portfolio health monitoring agent."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nfl_analytics.agents.portfolio_monitor import (
    HealthAlert,
    HealthReport,
    MonitorConfig,
    PortfolioMonitor,
)
from nfl_analytics.markets.base import Position, Side
from nfl_analytics.monitoring.alert_manager import AlertLevel, AlertManager
from nfl_analytics.monitoring.pnl_tracker import PnLTracker
from nfl_analytics.risk.drawdown_monitor import DrawdownMonitor


# ============================================================
# Helpers
# ============================================================


def _tracker(equity: int = 100_000) -> PnLTracker:
    return PnLTracker(initial_equity_cents=equity)


def _drawdown(initial: int = 100_000) -> DrawdownMonitor:
    return DrawdownMonitor(initial_equity_cents=initial)


def _position(
    contract_id: str = "C-1",
    quantity: int = 10,
    avg_entry_cents: int = 50,
) -> Position:
    return Position(
        contract_id=contract_id,
        side=Side.YES,
        quantity=quantity,
        avg_entry_cents=avg_entry_cents,
    )


# ============================================================
# Healthy portfolio
# ============================================================


class TestHealthyPortfolio:
    """No alerts when everything is normal."""

    def test_no_alerts(self):
        """Fresh portfolio with no drawdown produces no alerts."""
        pnl = _tracker()
        dd = _drawdown()
        monitor = PortfolioMonitor(pnl_tracker=pnl, drawdown_monitor=dd)
        report = monitor.check_health()
        assert not report.has_warnings
        assert not report.has_critical
        assert len(report.alerts) == 0

    def test_report_summary_healthy(self):
        """Healthy report summary says HEALTHY."""
        pnl = _tracker()
        monitor = PortfolioMonitor(pnl_tracker=pnl)
        report = monitor.check_health()
        summary = report.summary()
        assert "HEALTHY" in summary


# ============================================================
# Drawdown thresholds
# ============================================================


class TestDrawdownThresholds:
    """Drawdown crossing thresholds triggers alerts."""

    def test_drawdown_warning(self):
        """5% drawdown triggers warning."""
        dd = _drawdown(100_000)
        dd.record_equity(95_000)  # 5% drawdown
        pnl = _tracker()
        monitor = PortfolioMonitor(pnl_tracker=pnl, drawdown_monitor=dd)
        report = monitor.check_health()
        assert report.has_warnings
        assert any(a.level == "warning" and a.category == "drawdown" for a in report.alerts)

    def test_drawdown_elevated(self):
        """10% drawdown triggers elevated alert."""
        dd = _drawdown(100_000)
        dd.record_equity(90_000)  # 10% drawdown
        pnl = _tracker()
        monitor = PortfolioMonitor(pnl_tracker=pnl, drawdown_monitor=dd)
        report = monitor.check_health()
        assert any(a.level == "elevated" and a.category == "drawdown" for a in report.alerts)

    def test_drawdown_critical(self):
        """15% drawdown triggers critical alert."""
        dd = _drawdown(100_000)
        dd.record_equity(85_000)  # 15% drawdown
        pnl = _tracker()
        monitor = PortfolioMonitor(pnl_tracker=pnl, drawdown_monitor=dd)
        report = monitor.check_health()
        assert report.has_critical
        assert any(a.level == "critical" and a.category == "drawdown" for a in report.alerts)


# ============================================================
# Daily loss threshold
# ============================================================


class TestDailyLoss:
    """Excessive daily loss triggers alert."""

    def test_daily_loss_exceeds_threshold(self):
        """Large daily loss triggers critical alert."""
        pnl = _tracker(100_000)
        # Simulate loss: buy high, close low
        pnl.record_fill("KXNFL-KC", Side.YES, 80, 50)  # Buy 50@80
        pnl.record_fill("KXNFL-KC", Side.NO, 60, 50)    # Close at 60 → -1000¢

        config = MonitorConfig(max_daily_loss_pct=0.005)  # Very low threshold
        monitor = PortfolioMonitor(pnl_tracker=pnl, config=config)
        report = monitor.check_health()
        assert any(a.category == "daily_loss" for a in report.alerts)


# ============================================================
# Win rate
# ============================================================


class TestWinRate:
    """Low win rate triggers warning."""

    def test_low_win_rate(self):
        """Win rate below threshold triggers warning."""
        pnl = _tracker(100_000)
        # Create enough losing trades to trigger
        for i in range(12):
            cid = f"C-{i}"
            pnl.record_fill(cid, Side.YES, 50, 5)
            pnl.record_fill(cid, Side.NO, 40, 5)  # Loss

        config = MonitorConfig(min_win_rate=0.40, min_trades_for_win_rate=10)
        monitor = PortfolioMonitor(pnl_tracker=pnl, config=config)
        report = monitor.check_health()
        assert any(a.category == "win_rate" for a in report.alerts)

    def test_win_rate_not_checked_with_few_trades(self):
        """Win rate not checked when below min_trades_for_win_rate."""
        pnl = _tracker(100_000)
        pnl.record_fill("C-1", Side.YES, 50, 5)
        pnl.record_fill("C-1", Side.NO, 40, 5)  # 1 losing trade

        config = MonitorConfig(min_trades_for_win_rate=10)
        monitor = PortfolioMonitor(pnl_tracker=pnl, config=config)
        report = monitor.check_health()
        assert not any(a.category == "win_rate" for a in report.alerts)


# ============================================================
# Position concentration
# ============================================================


class TestPositionConcentration:
    """Single position too large relative to portfolio."""

    def test_concentrated_position(self):
        """Position >30% of equity triggers warning."""
        pnl = _tracker(100_000)
        positions = [_position("KXNFL-KC", quantity=700, avg_entry_cents=50)]
        # 700 * 50 = 35,000 = 35% of 100k
        monitor = PortfolioMonitor(pnl_tracker=pnl)
        report = monitor.check_health(positions=positions)
        assert any(a.category == "concentration" for a in report.alerts)

    def test_small_position_ok(self):
        """Small position doesn't trigger."""
        pnl = _tracker(100_000)
        positions = [_position("KXNFL-KC", quantity=10, avg_entry_cents=50)]
        # 10 * 50 = 500 = 0.5% of 100k
        monitor = PortfolioMonitor(pnl_tracker=pnl)
        report = monitor.check_health(positions=positions)
        assert not any(a.category == "concentration" for a in report.alerts)


# ============================================================
# AlertManager integration
# ============================================================


class TestAlertManagerIntegration:
    """Alerts dispatched via AlertManager when configured."""

    def test_alerts_dispatched(self):
        """Critical drawdown dispatches alert to AlertManager."""
        dd = _drawdown(100_000)
        dd.record_equity(84_000)  # 16% drawdown → critical

        pnl = _tracker()
        alert_mgr = AlertManager()
        monitor = PortfolioMonitor(
            pnl_tracker=pnl,
            drawdown_monitor=dd,
            alert_manager=alert_mgr,
        )
        report = monitor.check_health()
        assert report.has_critical
        assert alert_mgr._total_alerts >= 1


# ============================================================
# No drawdown monitor
# ============================================================


class TestNoDrawdownMonitor:
    """Monitor works without DrawdownMonitor (PnL checks only)."""

    def test_pnl_only(self):
        """Only PnL-based checks run when no drawdown monitor."""
        pnl = _tracker(100_000)
        monitor = PortfolioMonitor(pnl_tracker=pnl)
        report = monitor.check_health()
        assert report.drawdown_summary is None
        assert not report.has_warnings


# ============================================================
# Summary stats
# ============================================================


class TestMonitorSummary:
    """Monitor tracks statistics."""

    def test_summary_counts(self):
        """Summary reflects check and alert counts."""
        pnl = _tracker()
        dd = _drawdown(100_000)
        dd.record_equity(95_000)  # 5% → warning

        monitor = PortfolioMonitor(pnl_tracker=pnl, drawdown_monitor=dd)
        monitor.check_health()
        monitor.check_health()
        s = monitor.summary()
        assert s["total_checks"] == 2
        assert s["alerts_triggered"] >= 2  # At least 1 alert per check


# ============================================================
# Report properties
# ============================================================


class TestReportProperties:
    """HealthReport properties compute correctly."""

    def test_has_critical_true(self):
        """has_critical is True when critical alert exists."""
        report = HealthReport(alerts=[
            HealthAlert(level="critical", category="test", message="msg", value=0.2, threshold=0.15),
        ])
        assert report.has_critical

    def test_has_critical_false(self):
        """has_critical is False when only warnings."""
        report = HealthReport(alerts=[
            HealthAlert(level="warning", category="test", message="msg", value=0.06, threshold=0.05),
        ])
        assert not report.has_critical

    def test_summary_string(self):
        """Report summary includes alert details."""
        report = HealthReport(alerts=[
            HealthAlert(level="warning", category="drawdown", message="5% DD", value=0.05, threshold=0.05),
        ])
        summary = report.summary()
        assert "WARNING" in summary
        assert "drawdown" in summary
