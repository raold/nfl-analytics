"""Tests for alert manager — dispatch, cooldown, webhooks."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from nfl_analytics.monitoring.alert_manager import Alert, AlertConfig, AlertLevel, AlertManager

# ============================================================
# Initialization
# ============================================================


class TestInit:
    def test_default_config(self):
        mgr = AlertManager()
        assert mgr.config.enabled is True
        assert mgr.config.webhook_url is None
        assert mgr.config.alert_cooldown_seconds == 300

    def test_disabled_no_alerts(self):
        mgr = AlertManager(AlertConfig(enabled=False))
        result = mgr.alert(AlertLevel.INFO, "Test", "hello")
        assert result is None


# ============================================================
# Alert dispatch
# ============================================================


class TestDispatch:
    def test_info(self):
        mgr = AlertManager()
        result = mgr.alert(AlertLevel.INFO, "Test", "hello")
        assert result is not None
        assert result.level == AlertLevel.INFO
        assert result.title == "Test"

    def test_warning(self):
        mgr = AlertManager()
        result = mgr.alert(AlertLevel.WARNING, "Warn", "something")
        assert result is not None
        assert result.level == AlertLevel.WARNING

    def test_critical(self):
        mgr = AlertManager()
        result = mgr.alert(AlertLevel.CRITICAL, "Crit", "bad")
        assert result is not None
        assert result.level == AlertLevel.CRITICAL

    def test_returns_alert(self):
        mgr = AlertManager()
        result = mgr.alert(AlertLevel.INFO, "Test", "msg", {"key": "val"})
        assert isinstance(result, Alert)
        assert result.metadata == {"key": "val"}


# ============================================================
# Cooldown
# ============================================================


class TestCooldown:
    def test_suppresses_duplicate(self):
        mgr = AlertManager(AlertConfig(alert_cooldown_seconds=300))
        first = mgr.alert(AlertLevel.INFO, "Same Title", "first")
        second = mgr.alert(AlertLevel.INFO, "Same Title", "second")
        assert first is not None
        assert second is None  # Suppressed by cooldown

    def test_different_titles_ok(self):
        mgr = AlertManager(AlertConfig(alert_cooldown_seconds=300))
        a = mgr.alert(AlertLevel.INFO, "Title A", "a")
        b = mgr.alert(AlertLevel.INFO, "Title B", "b")
        assert a is not None
        assert b is not None

    def test_expires_allows_resend(self):
        mgr = AlertManager(AlertConfig(alert_cooldown_seconds=300))
        mgr.alert(AlertLevel.INFO, "Test", "first")
        # Manipulate last alert time to simulate cooldown expiry
        mgr._last_alert_time["Test"] = datetime.now(UTC) - timedelta(seconds=301)
        result = mgr.alert(AlertLevel.INFO, "Test", "after cooldown")
        assert result is not None


# ============================================================
# Convenience methods
# ============================================================


class TestConvenience:
    def test_fill(self):
        mgr = AlertManager()
        result = mgr.alert_fill("KXNFL-KC", "yes", 62, 10, pnl_cents=80)
        assert result is not None
        assert "62¢" in result.message

    def test_risk_breach(self):
        mgr = AlertManager()
        result = mgr.alert_risk_breach("position_limits", "Too many contracts", "KXNFL-KC")
        assert result is not None
        assert result.level == AlertLevel.WARNING

    def test_kill_switch(self):
        mgr = AlertManager()
        result = mgr.alert_kill_switch("daily_loss", 88_000)
        assert result is not None
        assert result.level == AlertLevel.CRITICAL

    def test_drawdown_warning(self):
        mgr = AlertManager()
        result = mgr.alert_drawdown_warning(0.15, 85_000)
        assert result is not None
        assert "15.0%" in result.message


# ============================================================
# Webhook
# ============================================================


class TestWebhook:
    def test_no_url_no_call(self):
        """No webhook URL → no HTTP call."""
        mgr = AlertManager(AlertConfig(webhook_url=None))
        with patch("nfl_analytics.monitoring.alert_manager.httpx", create=True) as mock_httpx:
            mgr.alert(AlertLevel.INFO, "Test", "hello")
            mock_httpx.Client.assert_not_called()

    def test_url_set_posts(self):
        """With webhook URL, should POST the alert."""
        mgr = AlertManager(AlertConfig(webhook_url="https://hooks.example.com/test"))
        with patch("httpx.Client") as MockClient:
            mock_instance = MagicMock()
            MockClient.return_value.__enter__ = MagicMock(return_value=mock_instance)
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            mgr.alert(AlertLevel.INFO, "Test", "hello")
            mock_instance.post.assert_called_once()

    def test_error_swallowed(self):
        """Webhook errors should not propagate to caller."""
        mgr = AlertManager(AlertConfig(webhook_url="https://hooks.example.com/test"))
        with patch("httpx.Client", side_effect=Exception("network error")):
            # Should not raise
            result = mgr.alert(AlertLevel.INFO, "Test", "hello")
            assert result is not None


# ============================================================
# Summary
# ============================================================


class TestSummary:
    def test_summary_keys(self):
        mgr = AlertManager()
        s = mgr.summary()
        assert "enabled" in s
        assert "total_alerts" in s
        assert "suppressed_alerts" in s
