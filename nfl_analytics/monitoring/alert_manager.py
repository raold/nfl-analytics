"""
Alert dispatch with cooldown for prediction market trading.

Always logs alerts locally; optionally POSTs to a webhook URL (Slack,
Discord, PagerDuty, etc.). Per-title cooldown prevents alert fatigue
during high-frequency events.

Usage:
    from nfl_analytics.monitoring.alert_manager import AlertManager, AlertConfig

    mgr = AlertManager(AlertConfig(webhook_url="https://hooks.slack.com/..."))
    mgr.alert(AlertLevel.WARNING, "Drawdown Warning", "5% drawdown reached")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AlertConfig:
    """Configuration for alert dispatch."""

    webhook_url: str | None = None
    alert_cooldown_seconds: int = 300
    enabled: bool = True


@dataclass
class Alert:
    """A dispatched alert."""

    level: AlertLevel
    title: str
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict | None = None


class AlertManager:
    """Dispatches alerts with per-title cooldown.

    Always logs alerts via Python logging. If a webhook_url is configured,
    also POSTs the alert payload. Webhook failures are swallowed to avoid
    blocking the trading loop.
    """

    def __init__(self, config: AlertConfig | None = None):
        self.config = config or AlertConfig()
        self._last_alert_time: dict[str, datetime] = {}
        self._total_alerts = 0
        self._suppressed_alerts = 0

    def alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        metadata: dict | None = None,
    ) -> Alert | None:
        """Dispatch an alert if enabled and not in cooldown.

        Returns the Alert if dispatched, None if suppressed or disabled.
        """
        if not self.config.enabled:
            return None

        if not self._check_cooldown(title):
            self._suppressed_alerts += 1
            return None

        alert = Alert(
            level=level,
            title=title,
            message=message,
            metadata=metadata,
        )

        self._last_alert_time[title] = alert.timestamp
        self._total_alerts += 1

        # Always log
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.CRITICAL: logging.CRITICAL,
        }.get(level, logging.INFO)
        logger.log(log_level, "[%s] %s: %s", level.value.upper(), title, message)

        # Optionally send webhook
        if self.config.webhook_url:
            self._send_webhook(alert)

        return alert

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def alert_fill(
        self,
        contract_id: str,
        side: str,
        price_cents: int,
        quantity: int,
        pnl_cents: int = 0,
    ) -> Alert | None:
        """Alert on a fill."""
        msg = f"{side} {quantity}x {contract_id} @{price_cents}¢"
        if pnl_cents:
            msg += f" (P&L: {pnl_cents}¢)"
        return self.alert(
            AlertLevel.INFO,
            "Fill",
            msg,
            {
                "contract_id": contract_id,
                "side": side,
                "price_cents": price_cents,
                "quantity": quantity,
                "pnl_cents": pnl_cents,
            },
        )

    def alert_risk_breach(
        self,
        layer: str,
        reason: str,
        contract_id: str,
    ) -> Alert | None:
        """Alert on a risk layer blocking a trade."""
        return self.alert(
            AlertLevel.WARNING,
            f"Risk Breach: {layer}",
            f"{reason} (contract: {contract_id})",
            {"layer": layer, "contract_id": contract_id},
        )

    def alert_kill_switch(self, reason: str, equity_cents: int) -> Alert | None:
        """Alert on kill switch activation."""
        return self.alert(
            AlertLevel.CRITICAL,
            "Kill Switch Activated",
            f"{reason} (equity: ${equity_cents / 100:.2f})",
            {"reason": reason, "equity_cents": equity_cents},
        )

    def alert_drawdown_warning(
        self,
        drawdown_pct: float,
        equity_cents: int,
    ) -> Alert | None:
        """Alert on drawdown reaching a warning threshold."""
        return self.alert(
            AlertLevel.WARNING,
            "Drawdown Warning",
            f"Drawdown at {drawdown_pct:.1%} (equity: ${equity_cents / 100:.2f})",
            {"drawdown_pct": drawdown_pct, "equity_cents": equity_cents},
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_cooldown(self, title: str) -> bool:
        """Check if enough time has passed since the last alert with this title."""
        last = self._last_alert_time.get(title)
        if last is None:
            return True
        elapsed = (datetime.now(UTC) - last).total_seconds()
        return elapsed >= self.config.alert_cooldown_seconds

    def _send_webhook(self, alert: Alert) -> None:
        """POST alert to webhook URL. Fire-and-forget, swallows exceptions."""
        try:
            import httpx

            payload = {
                "level": alert.level.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
            }
            if alert.metadata:
                payload["metadata"] = alert.metadata

            with httpx.Client(timeout=5.0) as client:
                client.post(self.config.webhook_url, json=payload)  # type: ignore[arg-type]
        except Exception:
            logger.debug("Webhook dispatch failed for alert: %s", alert.title, exc_info=True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Get alert manager summary."""
        return {
            "enabled": self.config.enabled,
            "webhook_configured": self.config.webhook_url is not None,
            "total_alerts": self._total_alerts,
            "suppressed_alerts": self._suppressed_alerts,
            "cooldown_seconds": self.config.alert_cooldown_seconds,
        }
