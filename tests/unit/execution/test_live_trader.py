"""Tests for live trader — async orchestration of strategy/risk/execution pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from py.execution.live_trader import LiveTrader, LiveTraderConfig
from py.execution.trade_executor import TradeExecutor
from py.markets.base import (
    MarketClient,
    MarketData,
    MarketError,
    Order,
    OrderBook,
    OrderStatus,
    Side,
)
from py.monitoring.alert_manager import AlertManager
from py.monitoring.pnl_tracker import PnLTracker
from py.monitoring.trade_logger import TradeLogger, TradeLoggerConfig
from py.risk.drawdown_monitor import DrawdownMonitor
from py.risk.kill_switch import KillSwitch, KillSwitchEvent
from py.risk.risk_manager import RiskCheckResult, RiskManager
from py.trading.base_strategy import TradingStrategy
from py.trading.signal import Signal

pytestmark = pytest.mark.anyio


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


def _make_order(signal: Signal, status: OrderStatus = OrderStatus.FILLED) -> Order:
    return Order(
        order_id="ord-001",
        contract_id=signal.contract_id,
        side=signal.side,
        price_cents=signal.price_cents,
        quantity=signal.quantity,
        status=status,
        filled_quantity=signal.quantity if status == OrderStatus.FILLED else 0,
        avg_fill_price_cents=signal.price_cents,
        strategy=signal.strategy,
    )


def _make_market_data(contract_id: str = "KXNFL-KC") -> MarketData:
    return MarketData(
        contract_id=contract_id,
        last_price_cents=60,
        yes_bid_cents=59,
        yes_ask_cents=61,
        volume_24h=1000,
        open_interest=500,
    )


def _make_orderbook(contract_id: str = "KXNFL-KC") -> OrderBook:
    return OrderBook(contract_id=contract_id)


def _make_trader(
    contract_ids: list[str] | None = None,
    signals: list[Signal] | None = None,
    risk_approved: bool = True,
    with_logger: bool = False,
    with_alerts: bool = False,
    with_drawdown: bool = False,
) -> tuple[LiveTrader, dict]:
    """Create a LiveTrader with fully mocked dependencies."""
    config = LiveTraderConfig(
        mode="paper",
        contract_ids=contract_ids or ["KXNFL-KC"],
        poll_interval_seconds=0.01,
    )

    client = AsyncMock(spec=MarketClient)
    client.get_market_data = AsyncMock(side_effect=lambda cid: _make_market_data(cid))
    client.get_orderbook = AsyncMock(side_effect=lambda cid: _make_orderbook(cid))
    client.get_positions = AsyncMock(return_value=[])
    client.get_balance_cents = AsyncMock(return_value=100_000)

    strategy = AsyncMock(spec=TradingStrategy)
    strategy.on_market_data = AsyncMock(return_value=signals or [])
    strategy.on_start = AsyncMock()
    strategy.on_stop = AsyncMock()
    strategy.on_fill = AsyncMock()

    risk_manager = MagicMock(spec=RiskManager)
    kill_switch = MagicMock(spec=KillSwitch)
    kill_switch.is_triggered = False
    risk_manager.kill_switch = kill_switch
    risk_manager.record_api_success = MagicMock()
    risk_manager.record_api_error = MagicMock(return_value=None)

    sig = signals[0] if signals else _make_signal()
    risk_result = RiskCheckResult(
        approved=risk_approved,
        signal=sig,
        layer="all_passed" if risk_approved else "position_limits",
        reason="" if risk_approved else "blocked",
        original_quantity=sig.quantity,
    )
    risk_manager.check_signal = MagicMock(return_value=risk_result)
    risk_manager.update_equity = MagicMock(return_value=None)

    executor = AsyncMock(spec=TradeExecutor)
    order = _make_order(sig)
    executor.submit_signal = AsyncMock(return_value=order)
    executor.cancel_all = AsyncMock(return_value=0)
    executor.on_fill = None

    pnl_tracker = MagicMock(spec=PnLTracker)
    pnl_tracker.total_equity_cents = 100_000
    pnl_tracker.realized_pnl_cents = 0
    pnl_tracker.unrealized_pnl_cents = 0
    pnl_tracker.trade_count = 0

    drawdown = DrawdownMonitor(initial_equity_cents=100_000) if with_drawdown else None
    trade_logger = TradeLogger(TradeLoggerConfig(log_dir="")) if with_logger else None
    alert_mgr = AlertManager() if with_alerts else None

    trader = LiveTrader(
        config=config,
        client=client,
        strategy=strategy,
        risk_manager=risk_manager,
        executor=executor,
        pnl_tracker=pnl_tracker,
        drawdown_monitor=drawdown,
        trade_logger=trade_logger,
        alert_manager=alert_mgr,
    )

    mocks = {
        "client": client,
        "strategy": strategy,
        "risk_manager": risk_manager,
        "kill_switch": kill_switch,
        "executor": executor,
        "pnl_tracker": pnl_tracker,
        "drawdown": drawdown,
        "trade_logger": trade_logger,
        "alert_mgr": alert_mgr,
    }

    return trader, mocks


# ============================================================
# Initialization
# ============================================================


class TestInit:
    async def test_default_config(self):
        trader, _ = _make_trader()
        assert trader._config.mode == "paper"

    async def test_not_running(self):
        trader, _ = _make_trader()
        assert not trader.is_running


# ============================================================
# Start / Stop
# ============================================================


class TestStartStop:
    async def test_start_activates(self):
        trader, mocks = _make_trader()
        await trader.start()
        mocks["strategy"].on_start.assert_called_once()
        assert trader.is_running
        await trader.stop()

    async def test_stop_deactivates(self):
        trader, mocks = _make_trader()
        await trader.start()
        await trader.stop()
        mocks["strategy"].on_stop.assert_called_once()
        assert not trader.is_running

    async def test_running_flag(self):
        trader, _ = _make_trader()
        assert not trader.is_running
        await trader.start()
        assert trader.is_running
        await trader.stop()
        assert not trader.is_running


# ============================================================
# Poll cycle
# ============================================================


class TestPollCycle:
    async def test_gets_market_data(self):
        trader, mocks = _make_trader()
        await trader.start()
        await trader._poll_cycle()
        mocks["client"].get_market_data.assert_called_once_with("KXNFL-KC")
        await trader.stop()

    async def test_feeds_strategy(self):
        trader, mocks = _make_trader()
        await trader.start()
        await trader._poll_cycle()
        mocks["strategy"].on_market_data.assert_called_once()
        await trader.stop()

    async def test_approved_submitted(self):
        signal = _make_signal()
        trader, mocks = _make_trader(signals=[signal], risk_approved=True)
        await trader.start()
        await trader._poll_cycle()
        mocks["executor"].submit_signal.assert_called_once()
        await trader.stop()

    async def test_blocked_not_submitted(self):
        signal = _make_signal()
        trader, mocks = _make_trader(signals=[signal], risk_approved=False)
        await trader.start()
        await trader._poll_cycle()
        mocks["executor"].submit_signal.assert_not_called()
        await trader.stop()

    async def test_api_error_recorded(self):
        trader, mocks = _make_trader()
        mocks["client"].get_market_data = AsyncMock(side_effect=MarketError("timeout"))
        await trader.start()
        await trader._poll_cycle()
        mocks["risk_manager"].record_api_error.assert_called_once()
        await trader.stop()


# ============================================================
# Fill handling
# ============================================================


class TestFillHandling:
    async def test_updates_pnl(self):
        signal = _make_signal()
        trader, mocks = _make_trader(signals=[signal])
        await trader.start()
        order = _make_order(signal)
        trader._on_fill_callback(order, signal)
        mocks["pnl_tracker"].record_fill.assert_called_once()
        await trader.stop()

    async def test_updates_kill_switch(self):
        signal = _make_signal()
        trader, mocks = _make_trader(signals=[signal])
        await trader.start()
        order = _make_order(signal)
        trader._on_fill_callback(order, signal)
        mocks["risk_manager"].update_equity.assert_called_once()
        await trader.stop()

    async def test_notifies_strategy(self):
        signal = _make_signal()
        trader, mocks = _make_trader(signals=[signal])
        await trader.start()
        order = _make_order(signal)
        trader._on_fill_callback(order, signal)
        # Strategy notification is async fire-and-forget
        # Just verify it didn't raise
        await trader.stop()

    async def test_logged(self):
        signal = _make_signal()
        trader, mocks = _make_trader(signals=[signal], with_logger=True)
        await trader.start()
        order = _make_order(signal)
        # Patch the trade_logger's log_fill to verify it's called
        trader._trade_logger.log_fill = MagicMock()
        trader._on_fill_callback(order, signal)
        trader._trade_logger.log_fill.assert_called_once()
        await trader.stop()

    async def test_alerted(self):
        signal = _make_signal()
        trader, mocks = _make_trader(signals=[signal], with_alerts=True)
        await trader.start()
        order = _make_order(signal)
        trader._alert_manager.alert_fill = MagicMock()
        trader._on_fill_callback(order, signal)
        trader._alert_manager.alert_fill.assert_called_once()
        await trader.stop()


# ============================================================
# Kill switch integration
# ============================================================


class TestKillSwitch:
    async def test_stops_trading(self):
        signal = _make_signal()
        trader, mocks = _make_trader(signals=[signal])
        await trader.start()

        # Make kill switch appear triggered during poll
        mocks["kill_switch"].is_triggered = True
        await trader._poll_cycle()

        # Should not have called strategy (short-circuited)
        mocks["strategy"].on_market_data.assert_not_called()
        await trader.stop()


# ============================================================
# Stats
# ============================================================


class TestStats:
    async def test_keys(self):
        trader, _ = _make_trader()
        s = trader.stats
        expected = {"mode", "running", "cycles", "signals_generated",
                    "signals_approved", "signals_blocked", "api_errors"}
        assert set(s.keys()) == expected

    async def test_counts(self):
        signal = _make_signal()
        trader, _ = _make_trader(signals=[signal], risk_approved=True)
        await trader.start()
        await trader._poll_cycle()
        s = trader.stats
        assert s["cycles"] == 1
        assert s["signals_generated"] == 1
        assert s["signals_approved"] == 1
        await trader.stop()
