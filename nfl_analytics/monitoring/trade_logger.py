"""
Structured JSON trade logging with automatic rotation.

Produces one JSON object per line (JSONL format) for each trading event:
signals, orders, fills, risk breaches, kill switch activations, and
equity snapshots.

Uses Python's RotatingFileHandler for automatic log rotation when files
exceed max_file_size_mb.

Usage:
    from nfl_analytics.monitoring.trade_logger import TradeLogger, TradeLoggerConfig

    logger = TradeLogger()
    logger.log_signal(signal, risk_result)
    logger.log_fill("KXNFL-KC", Side.YES, 62, 10, pnl_cents=80)
"""

from __future__ import annotations

import json
import logging
import logging.handlers
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from nfl_analytics.risk.kill_switch import KillSwitchEvent
from nfl_analytics.risk.risk_manager import RiskCheckResult
from nfl_analytics.trading.signal import Signal

_logger = logging.getLogger(__name__)


@dataclass
class TradeLoggerConfig:
    """Configuration for trade logging."""

    log_dir: str = "logs/trading"
    max_file_size_mb: int = 50
    rotate_count: int = 10


class _JsonFormatter(logging.Formatter):
    """Format log records as JSON objects (one per line)."""

    def format(self, record: logging.LogRecord) -> str:
        data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": getattr(record, "event", record.getMessage()),
            "data": getattr(record, "event_data", {}),
        }
        level = record.levelname
        if level != "INFO":
            data["level"] = level
        return json.dumps(data, default=str)


class TradeLogger:
    """Structured JSON logger for all trading events.

    Writes JSONL to a rotating log file. Each method produces a
    structured log entry with event type and relevant data fields.

    Pass log_dir="" to skip filesystem setup (for testing).
    """

    def __init__(self, config: TradeLoggerConfig | None = None):
        self.config = config or TradeLoggerConfig()

        # Create a dedicated logger (not root)
        self._logger = logging.getLogger(f"trade_logger.{id(self)}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        if self.config.log_dir:
            log_path = Path(self.config.log_dir)
            log_path.mkdir(parents=True, exist_ok=True)

            handler = logging.handlers.RotatingFileHandler(
                log_path / "trades.jsonl",
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.rotate_count,
            )
            handler.setFormatter(_JsonFormatter())
            self._logger.addHandler(handler)

    def _log(self, level: int, event: str, data: dict) -> None:
        """Emit a structured log entry."""
        record = self._logger.makeRecord(
            name=self._logger.name,
            level=level,
            fn="",
            lno=0,
            msg=event,
            args=(),
            exc_info=None,
        )
        record.event = event  # type: ignore[attr-defined]
        record.event_data = data  # type: ignore[attr-defined]
        self._logger.handle(record)

    # ------------------------------------------------------------------
    # Event methods
    # ------------------------------------------------------------------

    def log_signal(
        self,
        signal: Signal,
        risk_result: RiskCheckResult | None = None,
    ) -> None:
        """Log a strategy signal and optional risk check result."""
        data: dict = {
            "contract_id": signal.contract_id,
            "side": signal.side.value,
            "quantity": signal.quantity,
            "price_cents": signal.price_cents,
            "edge_cents": signal.edge_cents,
            "model_prob": signal.model_prob,
            "confidence": signal.confidence,
            "strategy": signal.strategy,
        }
        if risk_result is not None:
            data["risk_approved"] = risk_result.approved
            data["risk_layer"] = risk_result.layer
            if not risk_result.approved:
                data["risk_reason"] = risk_result.reason
        self._log(logging.INFO, "signal", data)

    def log_order(
        self,
        order_id: str,
        contract_id: str,
        side: str,
        price_cents: int,
        quantity: int,
        strategy: str,
    ) -> None:
        """Log an order submission."""
        self._log(logging.INFO, "order", {
            "order_id": order_id,
            "contract_id": contract_id,
            "side": side,
            "price_cents": price_cents,
            "quantity": quantity,
            "strategy": strategy,
        })

    def log_fill(
        self,
        contract_id: str,
        side: str,
        price_cents: int,
        quantity: int,
        pnl_cents: int = 0,
    ) -> None:
        """Log a fill (executed trade)."""
        data: dict = {
            "contract_id": contract_id,
            "side": side,
            "price_cents": price_cents,
            "quantity": quantity,
        }
        if pnl_cents:
            data["pnl_cents"] = pnl_cents
        self._log(logging.INFO, "fill", data)

    def log_risk_breach(self, result: RiskCheckResult) -> None:
        """Log a risk layer blocking a signal."""
        self._log(logging.WARNING, "risk_breach", {
            "layer": result.layer,
            "reason": result.reason,
            "contract_id": result.signal.contract_id,
            "side": result.signal.side.value,
            "quantity": result.signal.quantity,
            "original_quantity": result.original_quantity,
        })

    def log_kill_switch(self, event: KillSwitchEvent) -> None:
        """Log a kill switch activation at CRITICAL level."""
        self._log(logging.CRITICAL, "kill_switch", {
            "reason": event.reason,
            "trigger_value": event.trigger_value,
            "threshold": event.threshold,
            "equity_at_trigger_cents": event.equity_at_trigger_cents,
            "peak_equity_cents": event.peak_equity_cents,
            "positions_open": event.positions_open,
            "orders_open": event.orders_open,
        })

    def log_equity_snapshot(
        self,
        equity_cents: int,
        realized_pnl_cents: int,
        unrealized_pnl_cents: int,
        trade_count: int,
    ) -> None:
        """Log a periodic equity snapshot."""
        self._log(logging.INFO, "equity_snapshot", {
            "equity_cents": equity_cents,
            "realized_pnl_cents": realized_pnl_cents,
            "unrealized_pnl_cents": unrealized_pnl_cents,
            "trade_count": trade_count,
        })
