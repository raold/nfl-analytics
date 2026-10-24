"""Tests for structured JSON trade logger."""

from __future__ import annotations

import json
import logging

import pytest

from nfl_analytics.markets.base import Side
from nfl_analytics.monitoring.trade_logger import TradeLogger, TradeLoggerConfig, _JsonFormatter
from nfl_analytics.risk.kill_switch import KillSwitchEvent
from nfl_analytics.risk.risk_manager import RiskCheckResult
from nfl_analytics.trading.signal import Signal


def _make_signal(**kwargs) -> Signal:
    defaults = dict(
        contract_id="KXNFL-KC",
        side=Side.YES,
        quantity=10,
        price_cents=62,
        edge_cents=7,
        model_prob=0.69,
        confidence=0.82,
        strategy="directional",
    )
    defaults.update(kwargs)
    return Signal(**defaults)


class _CaptureHandler(logging.Handler):
    """Handler that captures log records for testing."""

    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord):
        self.records.append(record)


def _make_logger_with_capture() -> tuple[TradeLogger, _CaptureHandler]:
    """Create a TradeLogger with no filesystem and a capture handler."""
    tl = TradeLogger(TradeLoggerConfig(log_dir=""))
    handler = _CaptureHandler()
    handler.setFormatter(_JsonFormatter())
    tl._logger.addHandler(handler)
    return tl, handler


# ============================================================
# Initialization
# ============================================================


class TestInit:
    def test_default_config(self):
        tl = TradeLogger(TradeLoggerConfig(log_dir=""))
        assert tl.config.max_file_size_mb == 50

    def test_no_filesystem_when_empty_dir(self):
        """log_dir='' skips filesystem setup (no handlers from RotatingFileHandler)."""
        tl = TradeLogger(TradeLoggerConfig(log_dir=""))
        # Only the NullHandler from logging, not a RotatingFileHandler
        file_handlers = [
            h for h in tl._logger.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(file_handlers) == 0


# ============================================================
# Signal logging
# ============================================================


class TestSignalLogging:
    def test_basic(self):
        tl, handler = _make_logger_with_capture()
        signal = _make_signal()
        tl.log_signal(signal)
        assert len(handler.records) == 1
        assert handler.records[0].event == "signal"  # type: ignore[attr-defined]

    def test_with_risk_result(self):
        tl, handler = _make_logger_with_capture()
        signal = _make_signal()
        result = RiskCheckResult(approved=False, signal=signal, reason="kill switch active", layer="kill_switch")
        tl.log_signal(signal, result)
        data = handler.records[0].event_data  # type: ignore[attr-defined]
        assert data["risk_approved"] is False
        assert data["risk_reason"] == "kill switch active"


# ============================================================
# Fill logging
# ============================================================


class TestFillLogging:
    def test_basic(self):
        tl, handler = _make_logger_with_capture()
        tl.log_fill("KXNFL-KC", "yes", 62, 10)
        assert handler.records[0].event == "fill"  # type: ignore[attr-defined]

    def test_with_pnl(self):
        tl, handler = _make_logger_with_capture()
        tl.log_fill("KXNFL-KC", "yes", 62, 10, pnl_cents=80)
        data = handler.records[0].event_data  # type: ignore[attr-defined]
        assert data["pnl_cents"] == 80


# ============================================================
# Risk logging
# ============================================================


class TestRiskLogging:
    def test_breach(self):
        tl, handler = _make_logger_with_capture()
        signal = _make_signal()
        result = RiskCheckResult(
            approved=False,
            signal=signal,
            reason="position limit exceeded",
            layer="position_limits",
            original_quantity=10,
        )
        tl.log_risk_breach(result)
        assert handler.records[0].event == "risk_breach"  # type: ignore[attr-defined]
        assert handler.records[0].levelno == logging.WARNING

    def test_kill_switch_critical(self):
        tl, handler = _make_logger_with_capture()
        event = KillSwitchEvent(
            reason="daily_loss",
            trigger_value=0.12,
            threshold=0.10,
            equity_at_trigger_cents=88_000,
            peak_equity_cents=100_000,
            positions_open=3,
            orders_open=1,
        )
        tl.log_kill_switch(event)
        assert handler.records[0].event == "kill_switch"  # type: ignore[attr-defined]
        assert handler.records[0].levelno == logging.CRITICAL


# ============================================================
# Equity snapshot
# ============================================================


class TestEquitySnapshot:
    def test_snapshot(self):
        tl, handler = _make_logger_with_capture()
        tl.log_equity_snapshot(100_000, 500, -200, 12)
        data = handler.records[0].event_data  # type: ignore[attr-defined]
        assert data["equity_cents"] == 100_000
        assert data["trade_count"] == 12


# ============================================================
# JSON format
# ============================================================


class TestJsonFormat:
    def test_valid_json(self):
        tl, handler = _make_logger_with_capture()
        tl.log_fill("KXNFL-KC", "yes", 62, 10)
        formatted = handler.formatter.format(handler.records[0])
        parsed = json.loads(formatted)
        assert isinstance(parsed, dict)

    def test_required_fields(self):
        tl, handler = _make_logger_with_capture()
        tl.log_fill("KXNFL-KC", "yes", 62, 10)
        formatted = handler.formatter.format(handler.records[0])
        parsed = json.loads(formatted)
        assert "timestamp" in parsed
        assert "event" in parsed
        assert "data" in parsed
