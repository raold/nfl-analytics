"""Tests for multi-sport market router — prefix matching and strategy caching."""

from __future__ import annotations

import pytest

from py.markets.market_router import MarketRouter, SportConfig
from py.trading.base_strategy import TradingStrategy, StrategyConfig
from py.markets.base import MarketData, OrderBook
from py.trading.signal import Signal


class _DummyStrategy(TradingStrategy):
    """Minimal strategy for testing."""

    def __init__(self, name: str = "dummy"):
        super().__init__()
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def on_market_data(self, contract_id, market_data, orderbook=None):
        return []


def _nfl_config() -> SportConfig:
    return SportConfig(
        sport_name="nfl",
        contract_prefix="KXNFL",
        strategy_factory=lambda: _DummyStrategy("nfl_strat"),
    )


def _nba_config() -> SportConfig:
    return SportConfig(
        sport_name="nba",
        contract_prefix="KXNBA",
        strategy_factory=lambda: _DummyStrategy("nba_strat"),
    )


# ============================================================
# Initialization
# ============================================================


class TestInit:
    def test_empty_router(self):
        router = MarketRouter()
        assert router.supported_sports == []

    def test_empty_sports(self):
        router = MarketRouter()
        assert router.get_all_strategies() == {}


# ============================================================
# Registration
# ============================================================


class TestRegistration:
    def test_single(self):
        router = MarketRouter()
        router.register_sport(_nfl_config())
        assert "nfl" in router.supported_sports

    def test_multiple(self):
        router = MarketRouter()
        router.register_sport(_nfl_config())
        router.register_sport(_nba_config())
        assert len(router.supported_sports) == 2

    def test_supported_list(self):
        router = MarketRouter()
        router.register_sport(_nfl_config())
        router.register_sport(_nba_config())
        assert set(router.supported_sports) == {"nfl", "nba"}


# ============================================================
# Routing
# ============================================================


class TestRouting:
    def test_nfl(self):
        router = MarketRouter()
        router.register_sport(_nfl_config())
        assert router.route("KXNFL-26-AFC-KC") == "nfl"

    def test_nba(self):
        router = MarketRouter()
        router.register_sport(_nba_config())
        assert router.route("KXNBA-26-LAL") == "nba"

    def test_unknown_none(self):
        router = MarketRouter()
        router.register_sport(_nfl_config())
        assert router.route("KXMLB-26-NYY") is None

    def test_case_insensitive(self):
        router = MarketRouter()
        router.register_sport(_nfl_config())
        assert router.route("kxnfl-26-kc") == "nfl"

    def test_empty_string(self):
        router = MarketRouter()
        router.register_sport(_nfl_config())
        assert router.route("") is None


# ============================================================
# Strategy management
# ============================================================


class TestStrategy:
    def test_creates_instance(self):
        router = MarketRouter()
        router.register_sport(_nfl_config())
        strategy = router.get_strategy("KXNFL-26-KC")
        assert strategy is not None
        assert strategy.name == "nfl_strat"

    def test_caches(self):
        """get_strategy() should return the same instance on repeated calls."""
        router = MarketRouter()
        router.register_sport(_nfl_config())
        s1 = router.get_strategy("KXNFL-26-KC")
        s2 = router.get_strategy("KXNFL-26-BUF")
        assert s1 is s2  # Same sport, same cached strategy

    def test_unknown_none(self):
        router = MarketRouter()
        assert router.get_strategy("KXMLB-26-NYY") is None

    def test_get_all(self):
        router = MarketRouter()
        router.register_sport(_nfl_config())
        router.register_sport(_nba_config())
        # Trigger lazy creation
        router.get_strategy("KXNFL-26-KC")
        router.get_strategy("KXNBA-26-LAL")
        all_strats = router.get_all_strategies()
        assert len(all_strats) == 2
        assert "nfl" in all_strats
        assert "nba" in all_strats


# ============================================================
# Edge cases
# ============================================================


class TestEdgeCases:
    def test_overlapping_prefixes(self):
        """Longest prefix should win."""
        router = MarketRouter()
        # Register a short and long prefix
        router.register_sport(SportConfig(
            sport_name="generic",
            contract_prefix="KX",
            strategy_factory=lambda: _DummyStrategy("generic"),
        ))
        router.register_sport(SportConfig(
            sport_name="nfl",
            contract_prefix="KXNFL",
            strategy_factory=lambda: _DummyStrategy("nfl_specific"),
        ))
        # "KXNFL-26-KC" should match the longer "KXNFL" prefix
        assert router.route("KXNFL-26-KC") == "nfl"
        # "KXMLB-26-NYY" should match the shorter "KX" prefix
        assert router.route("KXMLB-26-NYY") == "generic"
