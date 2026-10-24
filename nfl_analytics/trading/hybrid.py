"""
Hybrid trading strategy: market making + directional in a single engine.

Routes market updates to the appropriate mode based on edge magnitude:

    |edge| >= directional_threshold  ->  One-sided directional signal
    |edge| <  directional_threshold  ->  Two-sided MM quotes

This captures the best of both worlds:
- Collect spread when the model sees no strong edge (most of the time)
- Take directional positions when the model sees a real mispricing

Both modes share inventory state — a directional fill that builds YES
inventory will cause the MM to skew quotes toward selling that inventory.

The hybrid owns the model interaction and injects fair values into the
internal MarketMaker. This keeps the MM focused on quoting mechanics.

Usage:
    from py.trading.hybrid import HybridStrategy, HybridConfig

    strategy = HybridStrategy(
        config=HybridConfig(directional_edge_cents=5),
    )
    strategy.set_week(2026, 1)
    signals = await strategy.on_market_data(contract_id, market_data, orderbook)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from py.markets.base import MarketData, OrderBook, Side
from py.trading.base_strategy import StrategyConfig, TradingStrategy
from py.trading.directional import (
    DirectionalConfig,
    GamePrediction,
    ParsedContract,
    parse_contract_id,
)
from py.trading.market_maker import MarketMaker, MarketMakerConfig
from py.trading.signal import Signal

logger = logging.getLogger(__name__)


@dataclass
class HybridConfig(StrategyConfig):
    """Configuration for hybrid MM + directional strategy.

    The hybrid switches between two modes based on edge magnitude:
    - MM mode: |edge| < directional_edge_cents -> two-sided quotes
    - Directional mode: |edge| >= directional_edge_cents -> one-sided bet

    Both modes share inventory and model predictions.
    """

    # Mode switching
    directional_edge_cents: int = 5  # Edge threshold for directional mode

    # Market maker sub-config
    mm_config: MarketMakerConfig = field(default_factory=MarketMakerConfig)

    # Directional sizing
    kelly_fraction_cap: float = 0.25  # Max Kelly fraction for directional
    max_directional_contracts: int = 50  # Max contracts per directional signal
    use_meta_labeling: bool = True
    min_meta_prob: float = 0.40

    # Model
    model_dir: str = "models/production/game_outcome_v2"
    prediction_cache_seconds: int = 300
    bankroll_cents: int = 100_000


class HybridStrategy(TradingStrategy):
    """Combined market making and directional trading strategy.

    The strategy computes fair value from the model, then decides:
    1. If edge is small (< threshold): pass fair value to internal MM,
       which generates two-sided quotes to capture spread.
    2. If edge is large (>= threshold): generate a one-sided directional
       signal to capture the mispricing.

    The internal MM tracks inventory. Directional fills also update the
    MM's inventory so that subsequent MM quotes properly skew.
    """

    def __init__(
        self,
        config: HybridConfig | None = None,
        model: object | None = None,
        meta_classifier: object | None = None,
    ):
        cfg = config or HybridConfig()
        super().__init__(cfg)
        self._config: HybridConfig = cfg

        # Internal market maker (handles two-sided quoting)
        self._mm = MarketMaker(cfg.mm_config)

        # Model components (lazy-loaded, same pattern as directional)
        self._model = model
        self._meta_classifier = meta_classifier

        # Prediction cache: game_key -> GamePrediction
        self._prediction_cache: dict[str, GamePrediction] = {}

        # Current week for feature extraction
        self._current_week: int | None = None
        self._current_season: int | None = None

        # Stats
        self._stats = {
            "directional_signals": 0,
            "mm_signals": 0,
            "directional_fills": 0,
            "mm_fills": 0,
        }

    @property
    def name(self) -> str:
        return "hybrid"

    @property
    def stats(self) -> dict:
        """Combined stats from both modes."""
        return {
            **self._stats,
            "mm_stats": self._mm.stats,
        }

    @property
    def market_maker(self) -> MarketMaker:
        """Access the internal market maker (for inventory inspection, etc.)."""
        return self._mm

    def set_week(self, season: int, week: int) -> None:
        """Set the current season/week for model predictions.

        Clears the prediction cache when the week changes.
        """
        if self._current_week != week:
            self._prediction_cache.clear()
            logger.info("Week changed to %d/%d, cleared prediction cache", season, week)
        self._current_week = week
        self._current_season = season

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    async def on_start(self) -> None:
        await super().on_start()
        await self._mm.on_start()
        self._ensure_model_loaded()

    async def on_stop(self) -> None:
        await self._mm.on_stop()
        await super().on_stop()

    async def on_market_data(
        self,
        contract_id: str,
        market_data: MarketData,
        orderbook: OrderBook | None = None,
    ) -> list[Signal]:
        """Route to MM or directional based on edge magnitude.

        Flow:
        1. Parse contract ID to identify the matchup
        2. Get model prediction (cached per game)
        3. Compute edge = model_prob - market_price
        4. |edge| >= threshold -> directional signal
        5. |edge| < threshold -> MM two-sided quotes
        """
        # Parse contract
        parsed = parse_contract_id(contract_id)
        if parsed is None:
            return []

        if self._current_week is None:
            return []

        # Get model fair value
        fair_value_result = self._get_fair_value(parsed)
        if fair_value_result is None:
            return []

        fair_value_cents, confidence, model_prob, prediction = fair_value_result

        # Get market price for edge computation
        market_price_cents = self._get_market_price(market_data, orderbook)
        if market_price_cents is None:
            return []

        # Compute edge
        edge_cents = round(fair_value_cents) - market_price_cents

        # Route based on edge magnitude
        if abs(edge_cents) >= self._config.directional_edge_cents:
            # Large edge -> directional
            signals = self._directional_signal(
                contract_id, fair_value_cents, edge_cents,
                market_price_cents, confidence, model_prob, prediction,
            )
            if signals:
                self._stats["directional_signals"] += len(signals)
                logger.info(
                    "HYBRID directional: %s edge=%d fv=%.1f mkt=%d",
                    contract_id, edge_cents, fair_value_cents, market_price_cents,
                )
            return signals
        else:
            # Small edge -> market make
            self._mm.set_fair_value(contract_id, fair_value_cents, confidence)
            signals = await self._mm.on_market_data(contract_id, market_data, orderbook)
            if signals:
                self._stats["mm_signals"] += len(signals)
            return signals

    async def on_fill(
        self, contract_id: str, side: str, price_cents: int, quantity: int
    ) -> None:
        """Forward fills to MM for inventory tracking.

        Both directional and MM fills update the same inventory, so the
        MM will properly skew after directional fills.
        """
        await self._mm.on_fill(contract_id, side, price_cents, quantity)

        if side == "yes":
            self._stats["directional_fills" if quantity > self._config.mm_config.quote_size else "mm_fills"] += 1
        else:
            self._stats["mm_fills"] += 1

    # ------------------------------------------------------------------
    # Directional signal generation
    # ------------------------------------------------------------------

    def _directional_signal(
        self,
        contract_id: str,
        fair_value_cents: float,
        edge_cents: int,
        market_price_cents: int,
        confidence: float,
        model_prob: float,
        prediction: GamePrediction | None,
    ) -> list[Signal]:
        """Generate a one-sided directional signal.

        Same logic as DirectionalStrategy but inline, sharing the
        hybrid's model and prediction cache.
        """
        if not self._check_cooldown(contract_id):
            return []

        if edge_cents >= self._config.directional_edge_cents:
            # Model says YES is underpriced -> buy YES
            side = Side.YES
            price_cents = market_price_cents
            effective_edge = edge_cents
            effective_prob = model_prob
        elif edge_cents <= -self._config.directional_edge_cents:
            # Model says NO is underpriced -> buy NO
            side = Side.NO
            price_cents = 100 - market_price_cents
            effective_edge = abs(edge_cents)
            effective_prob = 1.0 - model_prob
        else:
            return []

        # Meta-labeling quality filter
        meta_prob = 1.0
        if self._config.use_meta_labeling and self._meta_classifier is not None:
            meta_prob = self._compute_meta_prob(prediction, market_price_cents)
            if meta_prob < self._config.min_meta_prob:
                return []

        # Kelly sizing
        quantity = self._compute_quantity(effective_prob, price_cents, meta_prob)
        if quantity <= 0:
            return []

        signal = Signal(
            contract_id=contract_id,
            side=side,
            quantity=quantity,
            price_cents=price_cents,
            edge_cents=effective_edge,
            model_prob=effective_prob,
            confidence=confidence,
            strategy=self.name,
            meta_prob=meta_prob,
        )

        self._record_signal(contract_id)
        return [signal]

    # ------------------------------------------------------------------
    # Model interaction
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        """Lazy-load the GameOutcomeModel if not injected."""
        if self._model is not None:
            return

        try:
            from py.models.game_outcome_model_fixed import GameOutcomeModel

            self._model = GameOutcomeModel(model_dir=self._config.model_dir)
            self._model.load_models()
            logger.info("Loaded GameOutcomeModel from %s", self._config.model_dir)
        except Exception:
            logger.exception("Failed to load GameOutcomeModel")

    def _get_fair_value(
        self, parsed: ParsedContract
    ) -> tuple[float, float, float, GamePrediction | None] | None:
        """Get fair value for a contract from the model.

        Returns (fair_value_cents, confidence, model_prob, prediction) or None.
        """
        if not parsed.away_team:
            return None

        prediction = self._get_prediction(parsed)
        if prediction is None:
            return None

        model_prob = self._contract_probability(parsed, prediction)
        if model_prob is None:
            return None

        fair_value_cents = model_prob * 100.0
        confidence = max(0.1, 1.0 - (prediction.spread_std / 20.0))

        return fair_value_cents, confidence, model_prob, prediction

    def _get_prediction(self, parsed: ParsedContract) -> GamePrediction | None:
        """Get or compute a cached prediction for this game."""
        game_key = parsed.game_key
        cached = self._prediction_cache.get(game_key)
        if cached is not None:
            age = (datetime.now(UTC) - cached.predicted_at).total_seconds()
            if age < self._config.prediction_cache_seconds:
                return cached

        if self._model is None:
            self._ensure_model_loaded()
        if self._model is None:
            return None

        try:
            result = self._model.predict(
                home_team=parsed.home_team,
                away_team=parsed.away_team,
                season=parsed.season,
                week=self._current_week,
            )

            features = {}
            if hasattr(self._model, "feature_extractor"):
                try:
                    features = self._model.feature_extractor.extract_game_features(
                        parsed.home_team, parsed.away_team,
                        parsed.season, self._current_week,
                    )
                except Exception:
                    pass

            pred = GamePrediction(
                home_team=parsed.home_team,
                away_team=parsed.away_team,
                season=parsed.season,
                week=self._current_week,
                home_spread=result["home_spread"],
                total_points=result["total_points"],
                home_win_probability=result["home_win_probability"],
                spread_std=result.get("spread_std", 3.5),
                total_std=result.get("total_std", 5.0),
                features=features,
            )

            self._prediction_cache[game_key] = pred
            return pred

        except Exception:
            logger.exception("Prediction failed for %s@%s", parsed.away_team, parsed.home_team)
            return None

    def _contract_probability(
        self, parsed: ParsedContract, pred: GamePrediction
    ) -> float | None:
        """Convert model prediction to probability for this contract type."""
        if parsed.contract_type == "win":
            return pred.home_win_probability

        if parsed.contract_type == "spread":
            if parsed.line is None:
                return None
            from scipy.stats import norm

            prob = float(norm.cdf(
                pred.home_spread + parsed.line,
                loc=0,
                scale=pred.spread_std,
            ))
            return max(0.01, min(0.99, prob))

        if parsed.contract_type == "total":
            if parsed.line is None:
                return None
            from scipy.stats import norm

            prob = 1.0 - float(norm.cdf(
                parsed.line,
                loc=pred.total_points,
                scale=pred.total_std,
            ))
            return max(0.01, min(0.99, prob))

        return None

    def _get_market_price(
        self, market_data: MarketData, orderbook: OrderBook | None
    ) -> int | None:
        """Extract market price from order book mid or last trade."""
        if orderbook is not None and orderbook.mid_price_cents is not None:
            return int(orderbook.mid_price_cents)
        if market_data.last_price_cents and 1 <= market_data.last_price_cents <= 99:
            return market_data.last_price_cents
        if market_data.yes_ask_cents and 1 <= market_data.yes_ask_cents <= 99:
            return market_data.yes_ask_cents
        return None

    def _compute_meta_prob(
        self, pred: GamePrediction | None, market_price_cents: int
    ) -> float:
        """Run meta-labeling classifier."""
        if self._meta_classifier is None or pred is None:
            return 1.0

        try:
            prediction_dict = {
                "predicted_spread": pred.home_spread,
                "probability": pred.home_win_probability,
                "vegas_spread": 0.0,
                "uncertainty": pred.spread_std,
            }
            meta_features = self._meta_classifier.compute_meta_features_from_prediction(
                prediction=prediction_dict,
                game_features=pred.features,
            )
            return self._meta_classifier.predict_probability(meta_features)
        except Exception:
            return 1.0

    def _compute_quantity(
        self, model_prob: float, price_cents: int, meta_prob: float
    ) -> int:
        """Kelly-optimal position size for directional signals."""
        if price_cents <= 0 or price_cents >= 100:
            return 0

        payout_ratio = (100 - price_cents) / price_cents
        if payout_ratio <= 0:
            return 0

        kelly_raw = (model_prob * payout_ratio - (1 - model_prob)) / payout_ratio
        if kelly_raw <= 0:
            return 0

        kelly_capped = min(kelly_raw, self._config.kelly_fraction_cap)

        # Meta-labeling scaling
        if self._config.use_meta_labeling and meta_prob < 1.0:
            if meta_prob < 0.4:
                return 0
            elif meta_prob < 0.6:
                kelly_capped *= (meta_prob - 0.4) / 0.2
            # else: full Kelly

        bankroll_dollars = self._config.bankroll_cents / 100.0
        cost = price_cents / 100.0
        quantity = int(kelly_capped * bankroll_dollars / cost)

        return min(max(0, quantity), self._config.max_directional_contracts)
