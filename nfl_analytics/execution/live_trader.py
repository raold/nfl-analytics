"""
Live trading orchestrator — main loop wiring strategy, risk, and execution.

Connects the full pipeline:
    MarketClient → Strategy → RiskManager → TradeExecutor → Exchange
             ↑                                       |
             └──── PnLTracker ← DrawdownMonitor ←────┘
                        ↓
                   TradeLogger + AlertManager

Supports both paper and live modes (determined by which MarketClient
is injected). The poll loop iterates contracts, feeds market data to
the strategy, and submits approved signals.

Usage:
    from py.execution.live_trader import LiveTrader, LiveTraderConfig

    trader = LiveTrader(
        config=LiveTraderConfig(mode="paper"),
        client=paper_client,
        strategy=nfl_strategy,
        risk_manager=risk_mgr,
        executor=executor,
        pnl_tracker=pnl,
    )
    await trader.start()
    await trader.run()
    await trader.stop()
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from py.execution.trade_executor import TradeExecutor
from py.markets.base import MarketClient, MarketData, MarketError, Order, OrderBook
from py.monitoring.alert_manager import AlertLevel, AlertManager
from py.monitoring.pnl_tracker import PnLTracker
from py.monitoring.trade_logger import TradeLogger
from py.risk.drawdown_monitor import DrawdownMonitor
from py.risk.risk_manager import RiskManager
from py.trading.base_strategy import TradingStrategy
from py.trading.signal import Signal

logger = logging.getLogger(__name__)


@dataclass
class LiveTraderConfig:
    """Configuration for the live trader."""

    mode: str = "paper"  # "paper" or "live"
    poll_interval_seconds: float = 5.0
    equity_snapshot_interval_seconds: float = 60.0
    max_contracts_per_cycle: int = 50
    reconciliation_interval_seconds: float = 3600.0
    contract_ids: list[str] = field(default_factory=list)


class LiveTrader:
    """Main trading orchestration loop.

    Wires strategy → risk → execution → monitoring into a single
    async service. Call start() to activate, run() to begin polling,
    and stop() for graceful shutdown.
    """

    def __init__(
        self,
        config: LiveTraderConfig,
        client: MarketClient,
        strategy: TradingStrategy,
        risk_manager: RiskManager,
        executor: TradeExecutor,
        pnl_tracker: PnLTracker,
        drawdown_monitor: DrawdownMonitor | None = None,
        trade_logger: TradeLogger | None = None,
        alert_manager: AlertManager | None = None,
    ):
        self._config = config
        self._client = client
        self._strategy = strategy
        self._risk_manager = risk_manager
        self._executor = executor
        self._pnl_tracker = pnl_tracker
        self._drawdown_monitor = drawdown_monitor
        self._trade_logger = trade_logger
        self._alert_manager = alert_manager

        # State
        self._running = False
        self._tasks: list[asyncio.Task] = []

        # Stats
        self._cycles = 0
        self._signals_generated = 0
        self._signals_approved = 0
        self._signals_blocked = 0
        self._api_errors = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Activate the strategy and wire executor callbacks."""
        await self._strategy.on_start()

        # Wire fill callback
        self._executor.on_fill = self._on_fill_callback

        self._running = True
        logger.info("LiveTrader started (mode=%s)", self._config.mode)

    async def stop(self) -> None:
        """Graceful shutdown: cancel background tasks, stop strategy."""
        self._running = False

        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        await self._strategy.on_stop()
        logger.info("LiveTrader stopped")

    async def run(self) -> None:
        """Main loop: poll market data, generate signals, execute.

        Launches background tasks for equity snapshots and reconciliation,
        then enters the poll loop until stopped.
        """
        # Launch background periodic tasks
        self._tasks.append(asyncio.create_task(self._equity_snapshot_loop()))
        self._tasks.append(asyncio.create_task(self._reconciliation_loop()))

        while self._running:
            try:
                await self._poll_cycle()
            except asyncio.CancelledError:
                break
            except Exception:
                self._api_errors += 1
                logger.exception("Error in poll cycle")
                ks_event = self._risk_manager.record_api_error()
                if ks_event:
                    await self._handle_kill_switch(ks_event)

            if self._running:
                await asyncio.sleep(self._config.poll_interval_seconds)

    # ------------------------------------------------------------------
    # Poll cycle
    # ------------------------------------------------------------------

    async def _poll_cycle(self) -> None:
        """Iterate over contracts, get market data, feed strategy."""
        self._cycles += 1
        contracts = self._config.contract_ids[: self._config.max_contracts_per_cycle]

        for contract_id in contracts:
            if not self._running:
                break

            if self._risk_manager.kill_switch.is_triggered:
                break

            try:
                md = await self._client.get_market_data(contract_id)
                ob = await self._client.get_orderbook(contract_id)
                self._risk_manager.record_api_success()
                await self._handle_market_data(contract_id, md, ob)
            except MarketError:
                self._api_errors += 1
                logger.warning("API error for %s", contract_id, exc_info=True)
                ks_event = self._risk_manager.record_api_error()
                if ks_event:
                    await self._handle_kill_switch(ks_event)

    async def _handle_market_data(
        self,
        contract_id: str,
        md: MarketData,
        ob: OrderBook,
    ) -> None:
        """Signal pipeline for a single contract: strategy → risk → execute."""
        signals = await self._strategy.on_market_data(contract_id, md, ob)

        for signal in signals:
            self._signals_generated += 1

            # Risk check
            positions = await self._client.get_positions()
            result = self._risk_manager.check_signal(signal, positions=positions)

            if self._trade_logger:
                self._trade_logger.log_signal(signal, result)

            if result.approved:
                self._signals_approved += 1
                order = await self._executor.submit_signal(result.signal)
                if self._trade_logger:
                    self._trade_logger.log_order(
                        order.order_id,
                        signal.contract_id,
                        signal.side.value,
                        signal.price_cents,
                        signal.quantity,
                        signal.strategy,
                    )
            else:
                self._signals_blocked += 1
                if self._trade_logger:
                    self._trade_logger.log_risk_breach(result)
                if self._alert_manager:
                    self._alert_manager.alert_risk_breach(
                        result.layer, result.reason, signal.contract_id
                    )

    # ------------------------------------------------------------------
    # Fill handling
    # ------------------------------------------------------------------

    def _on_fill_callback(self, order: Order, signal: Signal) -> None:
        """Sync callback from executor on fill events.

        Updates PnL, drawdown monitor, kill switch, and notifies strategy.
        """
        # Update PnL
        self._pnl_tracker.record_fill(
            order.contract_id,
            order.side,
            order.avg_fill_price_cents,
            order.filled_quantity,
        )

        # Update kill switch equity
        equity = self._pnl_tracker.total_equity_cents
        ks_event = self._risk_manager.update_equity(equity)

        # Update drawdown monitor
        if self._drawdown_monitor:
            self._drawdown_monitor.record_equity(equity)

        # Log
        if self._trade_logger:
            self._trade_logger.log_fill(
                order.contract_id,
                order.side.value,
                order.avg_fill_price_cents,
                order.filled_quantity,
            )

        # Alert
        if self._alert_manager:
            self._alert_manager.alert_fill(
                order.contract_id,
                order.side.value,
                order.avg_fill_price_cents,
                order.filled_quantity,
            )

        # Notify strategy (fire-and-forget async)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._strategy.on_fill(
                    order.contract_id,
                    order.side.value,
                    order.avg_fill_price_cents,
                    order.filled_quantity,
                )
            )
        except RuntimeError:
            pass  # No running loop (in tests)

        # Handle kill switch if triggered
        if ks_event:
            if self._trade_logger:
                self._trade_logger.log_kill_switch(ks_event)
            if self._alert_manager:
                self._alert_manager.alert_kill_switch(
                    ks_event.reason, ks_event.equity_at_trigger_cents
                )
            self._running = False

    # ------------------------------------------------------------------
    # Background tasks
    # ------------------------------------------------------------------

    async def _equity_snapshot_loop(self) -> None:
        """Periodic equity recording for monitoring."""
        while self._running:
            try:
                await asyncio.sleep(self._config.equity_snapshot_interval_seconds)
                if not self._running:
                    break

                equity = self._pnl_tracker.total_equity_cents
                if self._drawdown_monitor:
                    self._drawdown_monitor.record_equity(equity)
                if self._trade_logger:
                    self._trade_logger.log_equity_snapshot(
                        equity,
                        self._pnl_tracker.realized_pnl_cents,
                        self._pnl_tracker.unrealized_pnl_cents,
                        self._pnl_tracker.trade_count,
                    )
            except asyncio.CancelledError:
                break
            except Exception:
                logger.debug("Equity snapshot error", exc_info=True)

    async def _reconciliation_loop(self) -> None:
        """Periodic position verification against exchange."""
        while self._running:
            try:
                await asyncio.sleep(self._config.reconciliation_interval_seconds)
                if not self._running:
                    break

                positions = await self._client.get_positions()
                balance = await self._client.get_balance_cents()
                logger.info(
                    "Reconciliation: %d positions, balance=%d¢",
                    len(positions),
                    balance,
                )
            except asyncio.CancelledError:
                break
            except Exception:
                logger.debug("Reconciliation error", exc_info=True)

    async def _handle_kill_switch(self, event) -> None:
        """Handle kill switch activation: log, alert, cancel orders."""
        if self._trade_logger:
            self._trade_logger.log_kill_switch(event)
        if self._alert_manager:
            self._alert_manager.alert_kill_switch(
                event.reason, event.equity_at_trigger_cents
            )
        try:
            await self._executor.cancel_all()
        except Exception:
            logger.warning("Failed to cancel all orders on kill switch", exc_info=True)
        self._running = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> dict:
        """Get trader statistics."""
        return {
            "mode": self._config.mode,
            "running": self._running,
            "cycles": self._cycles,
            "signals_generated": self._signals_generated,
            "signals_approved": self._signals_approved,
            "signals_blocked": self._signals_blocked,
            "api_errors": self._api_errors,
        }
