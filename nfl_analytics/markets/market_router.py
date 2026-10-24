"""
Multi-sport market router for prediction market trading.

Routes contract IDs to sport-specific strategy stacks via prefix matching.
Each sport registers a contract_prefix (e.g., "KXNFL") and a factory
function that creates the strategy instance on demand.

Longest-prefix-first matching ensures the most specific route wins.

Usage:
    from nfl_analytics.markets.market_router import MarketRouter, SportConfig

    router = MarketRouter()
    router.register_sport(SportConfig(
        sport_name="nfl",
        contract_prefix="KXNFL",
        strategy_factory=lambda: NFLDirectional(),
    ))
    strategy = router.get_strategy("KXNFL-26-AFC-KC")  # Returns NFLDirectional
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from nfl_analytics.trading.base_strategy import TradingStrategy

logger = logging.getLogger(__name__)


@dataclass
class SportConfig:
    """Configuration for a sport-specific strategy stack."""

    sport_name: str
    contract_prefix: str  # e.g., "KXNFL", "KXNBA"
    strategy_factory: Callable[[], TradingStrategy]
    model_dir: str = ""


class MarketRouter:
    """Routes contract IDs to sport-specific strategies via prefix matching.

    Prefixes are matched longest-first so more specific prefixes win.
    Strategy instances are lazy-created from factories and cached.
    """

    def __init__(self):
        self._configs: dict[str, SportConfig] = {}  # sport_name → config
        self._prefix_map: list[tuple[str, str]] = []  # (prefix, sport_name), sorted longest first
        self._strategies: dict[str, TradingStrategy] = {}  # sport_name → cached strategy

    def register_sport(self, config: SportConfig) -> None:
        """Register a sport configuration.

        Replaces any existing registration for the same sport_name.
        """
        self._configs[config.sport_name] = config

        # Rebuild prefix map sorted by length descending (longest first)
        self._prefix_map = sorted(
            [(c.contract_prefix.upper(), name) for name, c in self._configs.items()],
            key=lambda x: len(x[0]),
            reverse=True,
        )

        logger.info(
            "Registered sport %s with prefix %s",
            config.sport_name,
            config.contract_prefix,
        )

    def route(self, contract_id: str) -> str | None:
        """Route a contract ID to a sport name.

        Returns the sport_name of the longest matching prefix, or None
        if no prefix matches.
        """
        upper_id = contract_id.upper()
        for prefix, sport_name in self._prefix_map:
            if upper_id.startswith(prefix):
                return sport_name
        return None

    def get_strategy(self, contract_id: str) -> TradingStrategy | None:
        """Get (or create) the strategy for a contract ID.

        Returns None if the contract doesn't match any registered sport.
        Strategy instances are cached per sport.
        """
        sport_name = self.route(contract_id)
        if sport_name is None:
            return None

        if sport_name not in self._strategies:
            config = self._configs[sport_name]
            self._strategies[sport_name] = config.strategy_factory()
            logger.info("Created strategy instance for sport %s", sport_name)

        return self._strategies[sport_name]

    def get_config(self, sport_name: str) -> SportConfig | None:
        """Get configuration for a sport by name."""
        return self._configs.get(sport_name)

    def get_all_strategies(self) -> dict[str, TradingStrategy]:
        """Get all instantiated strategies."""
        return dict(self._strategies)

    @property
    def supported_sports(self) -> list[str]:
        """List of registered sport names."""
        return list(self._configs.keys())
