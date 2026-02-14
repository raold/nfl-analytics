"""Tests for TradingStrategy ABC and StrategyConfig."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from py.markets.base import MarketData, OrderBook, Side
from py.trading.base_strategy import StrategyConfig, TradingStrategy
from py.trading.signal import Signal

pytestmark = pytest.mark.anyio


class ConcreteStrategy(TradingStrategy):
    """Minimal concrete implementation for testing the ABC."""

    @property
    def name(self) -> str:
        return "test_strategy"

    async def on_market_data(
        self,
        contract_id: str,
        market_data: MarketData,
        orderbook: OrderBook | None = None,
    ) -> list[Signal]:
        return []


class TestStrategyConfig:
    """Configuration defaults and overrides."""

    def test_defaults(self):
        config = StrategyConfig()
        assert config.min_edge_cents == 3
        assert config.min_confidence == 0.50
        assert config.min_meta_prob == 0.40
        assert config.max_contracts_per_signal == 100
        assert config.cooldown_seconds == 60
        assert config.enabled_sports == ["nfl"]

    def test_custom_config(self):
        config = StrategyConfig(min_edge_cents=5, cooldown_seconds=120)
        assert config.min_edge_cents == 5
        assert config.cooldown_seconds == 120


class TestTradingStrategy:
    """TradingStrategy ABC lifecycle and cooldown."""

    async def test_start_stop(self):
        strategy = ConcreteStrategy()
        assert not strategy._active
        await strategy.on_start()
        assert strategy._active
        await strategy.on_stop()
        assert not strategy._active

    def test_name_property(self):
        strategy = ConcreteStrategy()
        assert strategy.name == "test_strategy"

    def test_cooldown_initial(self):
        """No cooldown on first signal."""
        strategy = ConcreteStrategy()
        assert strategy._check_cooldown("CONTRACT-1") is True

    def test_cooldown_blocks_rapid_fire(self):
        """Cooldown blocks signals within the window."""
        strategy = ConcreteStrategy(StrategyConfig(cooldown_seconds=60))
        strategy._record_signal("CONTRACT-1")
        assert strategy._check_cooldown("CONTRACT-1") is False

    def test_cooldown_per_contract(self):
        """Cooldown is per-contract — different contracts unaffected."""
        strategy = ConcreteStrategy(StrategyConfig(cooldown_seconds=60))
        strategy._record_signal("CONTRACT-1")
        assert strategy._check_cooldown("CONTRACT-1") is False
        assert strategy._check_cooldown("CONTRACT-2") is True  # Different contract

    def test_cooldown_expires(self):
        """Cooldown allows signals after the window expires."""
        strategy = ConcreteStrategy(StrategyConfig(cooldown_seconds=60))
        # Backdate the last signal time
        strategy._last_signal_time["CONTRACT-1"] = datetime.now(UTC) - timedelta(seconds=61)
        assert strategy._check_cooldown("CONTRACT-1") is True

    async def test_on_fill_default_noop(self):
        """Default on_fill is a no-op."""
        strategy = ConcreteStrategy()
        await strategy.on_fill("CONTRACT-1", "yes", 50, 10)  # Should not raise

    async def test_on_market_data_returns_empty(self):
        """Concrete implementation returns empty list."""
        strategy = ConcreteStrategy()
        md = MarketData(
            contract_id="TEST-01",
            last_price_cents=50,
            yes_bid_cents=49,
            yes_ask_cents=51,
            volume_24h=100,
            open_interest=0,
        )
        signals = await strategy.on_market_data("TEST-01", md)
        assert signals == []
