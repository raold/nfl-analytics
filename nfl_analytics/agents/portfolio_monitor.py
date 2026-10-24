"""Continuous portfolio health monitoring agent.

Polls PnLTracker and DrawdownMonitor to detect concerning portfolio
states and trigger AlertManager warnings at configurable thresholds.

Usage:
    from nfl_analytics.agents.portfolio_monitor import PortfolioMonitor, MonitorConfig

    monitor = PortfolioMonitor(pnl_tracker=pnl, drawdown_monitor=dd)
    report = monitor.check_health()
    if report.alerts:
        for alert in report.alerts:
            print(f"[{alert.level}] {alert.message}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from nfl_analytics.markets.base import Position
from nfl_analytics.monitoring.alert_manager import AlertLevel, AlertManager
from nfl_analytics.monitoring.pnl_tracker import PnLTracker
from nfl_analytics.risk.drawdown_monitor import DrawdownMonitor

logger = logging.getLogger(__name__)


@dataclass
class MonitorConfig:
    """Configuration for portfolio health monitoring thresholds."""

    drawdown_warning_pct: float = 0.05
    drawdown_elevated_pct: float = 0.10
    drawdown_critical_pct: float = 0.15
    max_position_concentration_pct: float = 0.30
    max_daily_loss_pct: float = 0.08
    min_win_rate: float = 0.40
    min_trades_for_win_rate: int = 10


@dataclass
class HealthAlert:
    """A single health alert raised by the portfolio monitor."""

    level: str  # "warning", "elevated", "critical"
    category: str
    message: str
    value: float
    threshold: float


@dataclass
class HealthReport:
    """Result of a portfolio health check."""

    alerts: list[HealthAlert] = field(default_factory=list)
    pnl_summary: dict = field(default_factory=dict)
    drawdown_summary: dict | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def has_critical(self) -> bool:
        """True if any critical-level alerts are present."""
        return any(a.level == "critical" for a in self.alerts)

    @property
    def has_warnings(self) -> bool:
        """True if any alerts of any level are present."""
        return len(self.alerts) > 0

    def summary(self) -> str:
        """Human-readable summary of the health check."""
        lines = [
            f"Portfolio Health Check — {self.timestamp:%Y-%m-%d %H:%M UTC}",
            f"{'=' * 50}",
        ]

        if not self.alerts:
            lines.append("Status: HEALTHY — no alerts")
        else:
            critical = sum(1 for a in self.alerts if a.level == "critical")
            elevated = sum(1 for a in self.alerts if a.level == "elevated")
            warning = sum(1 for a in self.alerts if a.level == "warning")
            lines.append(
                f"Status: {'CRITICAL' if critical else 'ELEVATED' if elevated else 'WARNING'}"
            )
            lines.append(
                f"Alerts: {critical} critical, {elevated} elevated, {warning} warning"
            )
            lines.append("")
            for alert in self.alerts:
                lines.append(
                    f"  [{alert.level.upper()}] {alert.category}: {alert.message}"
                )

        if self.pnl_summary:
            equity = self.pnl_summary.get("total_equity_cents", 0)
            daily = self.pnl_summary.get("daily_pnl_cents", 0)
            lines.append(f"\nEquity: ${equity / 100:.2f} | Daily P&L: ${daily / 100:.2f}")

        if self.drawdown_summary:
            dd = self.drawdown_summary.get("current_drawdown_pct", 0)
            max_dd = self.drawdown_summary.get("max_drawdown_pct", 0)
            lines.append(f"Drawdown: {dd:.1%} current, {max_dd:.1%} max")

        return "\n".join(lines)


class PortfolioMonitor:
    """Continuous portfolio health monitoring.

    Checks drawdown levels, daily P&L losses, win rate, and position
    concentration against configurable thresholds. Optionally dispatches
    alerts via AlertManager.
    """

    def __init__(
        self,
        pnl_tracker: PnLTracker,
        drawdown_monitor: DrawdownMonitor | None = None,
        alert_manager: AlertManager | None = None,
        config: MonitorConfig | None = None,
    ):
        self._pnl = pnl_tracker
        self._drawdown = drawdown_monitor
        self._alert_manager = alert_manager
        self.config = config or MonitorConfig()

        # Stats
        self._total_checks = 0
        self._total_alerts = 0

    def check_health(
        self,
        positions: list[Position] | None = None,
    ) -> HealthReport:
        """Run a comprehensive portfolio health check.

        Args:
            positions: Current open positions (for concentration check).

        Returns:
            HealthReport with any alerts and summary data.
        """
        self._total_checks += 1
        alerts: list[HealthAlert] = []

        pnl_summary = self._pnl.summary()

        # Check drawdown levels
        drawdown_summary = None
        if self._drawdown is not None:
            drawdown_summary = self._drawdown.summary()
            dd_pct = self._drawdown.current_drawdown_pct

            if dd_pct >= self.config.drawdown_critical_pct:
                alerts.append(HealthAlert(
                    level="critical",
                    category="drawdown",
                    message=(
                        f"Drawdown at {dd_pct:.1%} — exceeds critical threshold "
                        f"{self.config.drawdown_critical_pct:.1%}"
                    ),
                    value=dd_pct,
                    threshold=self.config.drawdown_critical_pct,
                ))
            elif dd_pct >= self.config.drawdown_elevated_pct:
                alerts.append(HealthAlert(
                    level="elevated",
                    category="drawdown",
                    message=(
                        f"Drawdown at {dd_pct:.1%} — exceeds elevated threshold "
                        f"{self.config.drawdown_elevated_pct:.1%}"
                    ),
                    value=dd_pct,
                    threshold=self.config.drawdown_elevated_pct,
                ))
            elif dd_pct >= self.config.drawdown_warning_pct:
                alerts.append(HealthAlert(
                    level="warning",
                    category="drawdown",
                    message=(
                        f"Drawdown at {dd_pct:.1%} — exceeds warning threshold "
                        f"{self.config.drawdown_warning_pct:.1%}"
                    ),
                    value=dd_pct,
                    threshold=self.config.drawdown_warning_pct,
                ))

        # Check daily P&L loss
        total_equity = pnl_summary.get("total_equity_cents", 0)
        daily_pnl = pnl_summary.get("daily_pnl_cents", 0)
        if total_equity > 0 and daily_pnl < 0:
            daily_loss_pct = abs(daily_pnl) / total_equity
            if daily_loss_pct >= self.config.max_daily_loss_pct:
                alerts.append(HealthAlert(
                    level="critical",
                    category="daily_loss",
                    message=(
                        f"Daily loss {daily_loss_pct:.1%} exceeds threshold "
                        f"{self.config.max_daily_loss_pct:.1%}"
                    ),
                    value=daily_loss_pct,
                    threshold=self.config.max_daily_loss_pct,
                ))

        # Check win rate (only if enough trades)
        trade_count = pnl_summary.get("trade_count", 0)
        win_rate = pnl_summary.get("win_rate", 0.0)
        if trade_count >= self.config.min_trades_for_win_rate:
            if win_rate < self.config.min_win_rate:
                alerts.append(HealthAlert(
                    level="warning",
                    category="win_rate",
                    message=(
                        f"Win rate {win_rate:.1%} below minimum {self.config.min_win_rate:.1%} "
                        f"({trade_count} trades)"
                    ),
                    value=win_rate,
                    threshold=self.config.min_win_rate,
                ))

        # Check position concentration
        if positions and total_equity > 0:
            for pos in positions:
                position_value = pos.avg_entry_cents * pos.quantity
                concentration = position_value / total_equity
                if concentration >= self.config.max_position_concentration_pct:
                    alerts.append(HealthAlert(
                        level="warning",
                        category="concentration",
                        message=(
                            f"Position {pos.contract_id} is {concentration:.1%} of portfolio "
                            f"(max {self.config.max_position_concentration_pct:.1%})"
                        ),
                        value=concentration,
                        threshold=self.config.max_position_concentration_pct,
                    ))

        # Dispatch alerts via AlertManager if configured
        if self._alert_manager is not None and alerts:
            for ha in alerts:
                alert_level = {
                    "warning": AlertLevel.WARNING,
                    "elevated": AlertLevel.WARNING,
                    "critical": AlertLevel.CRITICAL,
                }.get(ha.level, AlertLevel.INFO)
                self._alert_manager.alert(
                    alert_level,
                    f"Portfolio {ha.category}",
                    ha.message,
                    {"value": ha.value, "threshold": ha.threshold},
                )

        self._total_alerts += len(alerts)

        report = HealthReport(
            alerts=alerts,
            pnl_summary=pnl_summary,
            drawdown_summary=drawdown_summary,
        )

        if report.has_critical:
            logger.warning("Portfolio health: CRITICAL — %d alerts", len(alerts))
        elif report.has_warnings:
            logger.info("Portfolio health: %d alerts", len(alerts))
        else:
            logger.debug("Portfolio health: OK")

        return report

    def summary(self) -> dict:
        """Get monitor summary statistics."""
        return {
            "total_checks": self._total_checks,
            "alerts_triggered": self._total_alerts,
        }
