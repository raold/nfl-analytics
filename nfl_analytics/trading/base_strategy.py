"""
Abstract base class for prediction market trading strategies.

All strategies (directional, market maker, hybrid) implement this interface.
The main loop feeds MarketData events to on_market_data(), which returns
Signals for the risk layer to validate and the executor to submit.

Pipeline:
    MarketData → Strategy.on_market_data() → list[Signal]
    → RiskManager.check() → TradeExecutor.submit()

Usage:
    class MyStrategy(TradingStrategy):
        async def on_market_data(self, data, book):
            if self._has_edge(data):
                return [Signal(...)]
            return []
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime

from nfl_analytics.markets.base import MarketClient, MarketData, OrderBook
from nfl_analytics.trading.signal import Signal

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration shared by all strategies."""

    min_edge_cents: int = 3  # Minimum edge to generate signal (3¢ = 3%)
    min_confidence: float = 0.50  # Minimum model confidence
    min_meta_prob: float = 0.40  # Minimum meta-labeling probability
    max_contracts_per_signal: int = 100  # Max contracts per order
    cooldown_seconds: int = 60  # Min time between signals on same contract
    enabled_sports: list[str] = field(default_factory=lambda: ["nfl"])


class TradingStrategy(ABC):
    """Abstract base class for prediction market strategies.

    Subclasses must implement:
    - on_market_data(): Generate signals from market updates
    - name: Strategy identifier string

    Optional overrides:
    - on_fill(): React to order fills (e.g., update inventory)
    - on_start()/on_stop(): Lifecycle hooks
    """

    def __init__(self, config: StrategyConfig | None = None):
        self.config = config or StrategyConfig()
        self._last_signal_time: dict[str, datetime] = {}  # contract_id → last signal time
        self._active = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier (e.g., 'directional', 'market_maker')."""

    @abstractmethod
    async def on_market_data(
        self,
        contract_id: str,
        market_data: MarketData,
        orderbook: OrderBook | None = None,
    ) -> list[Signal]:
        """Generate trade signals from a market data update.

        Called whenever a contract's market data changes. Returns zero or
        more Signals. Signals are NOT orders — they go through risk checks
        first.

        Args:
            contract_id: The contract that updated.
            market_data: Latest price/volume snapshot.
            orderbook: Order book snapshot (None if unavailable).

        Returns:
            List of Signal objects (empty = no action).
        """

    async def on_fill(self, contract_id: str, side: str, price_cents: int, quantity: int) -> None:
        """Called when an order from this strategy fills.

        Override to update inventory tracking, hedge ratios, etc.
        Default: no-op.
        """

    async def on_start(self) -> None:
        """Called when the strategy is started. Override for initialization."""
        self._active = True
        logger.info("Strategy %s started", self.name)

    async def on_stop(self) -> None:
        """Called when the strategy is stopped. Override for cleanup."""
        self._active = False
        logger.info("Strategy %s stopped", self.name)

    def _check_cooldown(self, contract_id: str) -> bool:
        """Check if enough time has passed since last signal on this contract."""
        last = self._last_signal_time.get(contract_id)
        if last is None:
            return True
        elapsed = (datetime.now(UTC) - last).total_seconds()
        return elapsed >= self.config.cooldown_seconds

    def _record_signal(self, contract_id: str) -> None:
        """Record that a signal was generated for cooldown tracking."""
        self._last_signal_time[contract_id] = datetime.now(UTC)
