"""
Edge-based directional trading strategy for prediction markets.

Wires the full NFL analytics pipeline into a trading strategy:
    contract → parse(home, away, season, week, type)
             → GameFeatureExtractor (382 features)
             → GameOutcomeModel (spread, total, win_prob)
             → edge = model_prob - contract_price
             → MetaLabeling filter (trade quality gate)
             → Kelly sizing (position size)
             → Signal

This is the primary strategy for Phase 2. It generates buy/sell signals
when the model's probability estimate diverges from the market price by
more than a configurable threshold (default: 5 cents = 5%).

Signal flow:
    MarketData update
    → Extract (home_team, away_team, season, week) from contract_id
    → Check cooldown (no spam)
    → Load model predictions (cached per game_key)
    → Map contract type to model output (spread/total/win)
    → Calculate edge in cents
    → If |edge| >= threshold:
        → Run meta-labeling quality filter
        → Calculate Kelly-optimal position size
        → Emit Signal for risk layer

Usage:
    from py.trading.directional import DirectionalStrategy, DirectionalConfig

    strategy = DirectionalStrategy(
        model_dir="models/production/game_outcome_v2",
    )
    await strategy.on_start()

    signals = await strategy.on_market_data(
        contract_id="KXNFL-26-WIN-KC",
        market_data=market_data,
    )
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

from py.markets.base import MarketData, OrderBook, Side
from py.trading.base_strategy import StrategyConfig, TradingStrategy
from py.trading.signal import Signal

logger = logging.getLogger(__name__)

# Default production model directory
DEFAULT_MODEL_DIR = "models/production/game_outcome_v2"


@dataclass
class DirectionalConfig(StrategyConfig):
    """Configuration specific to directional edge trading.

    Extends base StrategyConfig with model paths and edge thresholds.
    """

    model_dir: str = DEFAULT_MODEL_DIR
    min_edge_cents: int = 5  # Higher than base (3) — directional needs more edge
    max_contracts_per_signal: int = 50  # Conservative for directional
    prediction_cache_seconds: int = 300  # Cache model predictions for 5 min
    use_meta_labeling: bool = True  # Filter trades through meta-label quality gate
    use_kelly_sizing: bool = True  # Size positions with Kelly criterion
    kelly_fraction_cap: float = 0.25  # Max Kelly fraction (quarter-Kelly default)
    bankroll_cents: int = 100_000  # Default bankroll for sizing ($1,000)


@dataclass
class GamePrediction:
    """Cached prediction from GameOutcomeModel.

    Stores all model outputs for a single game, reusable across
    multiple contract types (spread, total, win_prob) for the same matchup.
    """

    home_team: str
    away_team: str
    season: int
    week: int
    home_spread: float
    total_points: float
    home_win_probability: float
    spread_std: float
    total_std: float
    features: dict[str, float]  # Raw features for meta-labeling
    predicted_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def away_win_probability(self) -> float:
        return 1.0 - self.home_win_probability


# ---------------------------------------------------------------------------
# Contract ID parsing
# ---------------------------------------------------------------------------

# Expected patterns for Kalshi NFL contracts:
#   Win:    KXNFL-{YY}-WIN-{TEAM}
#   Spread: KXNFL-{YY}-SPREAD-{HOME}-{AWAY}-{LINE}
#   Total:  KXNFL-{YY}-TOTAL-{HOME}-{AWAY}-{LINE}
#   Moneyline: KXNFL-{YY}-ML-{TEAM}
#
# These patterns are configurable — real Kalshi IDs may differ.
# The parser is lenient and extracts what it can.

_WIN_PATTERN = re.compile(
    r"KXNFL-(\d{2})-WIN-([A-Z]{2,3})$", re.IGNORECASE
)
_SPREAD_PATTERN = re.compile(
    r"KXNFL-(\d{2})-SPREAD-([A-Z]{2,3})-([A-Z]{2,3})-([\d.+-]+)$",
    re.IGNORECASE,
)
_TOTAL_PATTERN = re.compile(
    r"KXNFL-(\d{2})-TOTAL-([A-Z]{2,3})-([A-Z]{2,3})-([\d.]+)$",
    re.IGNORECASE,
)


@dataclass
class ParsedContract:
    """Parsed information from a Kalshi NFL contract ID."""

    contract_type: str  # "win", "spread", "total"
    season: int
    home_team: str
    away_team: str
    line: float | None = None  # Spread or total line

    @property
    def game_key(self) -> str:
        """Unique key for caching predictions per game."""
        return f"{self.season}-{self.home_team}-{self.away_team}"


def parse_contract_id(contract_id: str) -> ParsedContract | None:
    """Parse a Kalshi NFL contract ID into structured data.

    Returns None if the contract ID doesn't match any known pattern.
    This allows the strategy to silently skip non-NFL or unrecognized contracts.
    """
    # Win market: KXNFL-26-WIN-KC → YES = KC wins
    m = _WIN_PATTERN.match(contract_id)
    if m:
        year = 2000 + int(m.group(1))
        team = m.group(2).upper()
        return ParsedContract(
            contract_type="win",
            season=year,
            home_team=team,
            away_team="",  # Win markets are team-level, opponent unknown from ID
        )

    # Spread market: KXNFL-26-SPREAD-KC-BUF-3.5 → YES = KC covers -3.5
    m = _SPREAD_PATTERN.match(contract_id)
    if m:
        year = 2000 + int(m.group(1))
        home = m.group(2).upper()
        away = m.group(3).upper()
        line = float(m.group(4))
        return ParsedContract(
            contract_type="spread",
            season=year,
            home_team=home,
            away_team=away,
            line=line,
        )

    # Total market: KXNFL-26-TOTAL-KC-BUF-47.5 → YES = over 47.5
    m = _TOTAL_PATTERN.match(contract_id)
    if m:
        year = 2000 + int(m.group(1))
        home = m.group(2).upper()
        away = m.group(3).upper()
        line = float(m.group(4))
        return ParsedContract(
            contract_type="total",
            season=year,
            home_team=home,
            away_team=away,
            line=line,
        )

    return None


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class DirectionalStrategy(TradingStrategy):
    """Edge-based directional trading strategy.

    Generates buy signals when the model's probability estimate exceeds
    the market price by more than the configured edge threshold, and sell
    signals when the market overprices an outcome.

    The strategy:
    1. Parses contract_id to identify the matchup and contract type
    2. Runs the GameOutcomeModel to get spread/total/win_prob predictions
    3. Converts predictions to probabilities (cents) for each contract type
    4. Calculates edge = model_prob_cents - contract_price_cents
    5. Filters through meta-labeling for trade quality
    6. Sizes positions using Kelly criterion

    Thread-safe: predictions are cached per game key with a TTL.
    """

    def __init__(
        self,
        config: DirectionalConfig | None = None,
        # Allow dependency injection for testing
        model: object | None = None,
        meta_classifier: object | None = None,
        kelly_sizer: object | None = None,
    ):
        cfg = config or DirectionalConfig()
        super().__init__(cfg)
        self._config: DirectionalConfig = cfg

        # Lazy-loaded model components (heavy — defer until first use)
        self._model = model
        self._meta_classifier = meta_classifier
        self._kelly_sizer = kelly_sizer

        # Prediction cache: game_key → GamePrediction
        self._prediction_cache: dict[str, GamePrediction] = {}

        # Track current week for cache invalidation
        self._current_week: int | None = None

    @property
    def name(self) -> str:
        return "directional"

    def set_week(self, season: int, week: int) -> None:
        """Set the current season/week for feature extraction.

        Must be called before processing market data. Clears the
        prediction cache when the week changes.
        """
        if self._current_week != week:
            self._prediction_cache.clear()
            logger.info("Week changed to %d/%d, cleared prediction cache", season, week)
        self._current_week = week
        self._current_season = season

    async def on_start(self) -> None:
        """Initialize model components on strategy start."""
        await super().on_start()
        self._ensure_model_loaded()

    async def on_market_data(
        self,
        contract_id: str,
        market_data: MarketData,
        orderbook: OrderBook | None = None,
    ) -> list[Signal]:
        """Generate directional trade signals from market data.

        Returns 0 or 1 signals per market data update. A signal is
        generated only when:
        - Contract ID parses to a known NFL market type
        - Cooldown has elapsed since last signal on this contract
        - Model edge exceeds threshold
        - Meta-labeling approves the trade quality (if enabled)
        """
        # Parse the contract to identify the matchup
        parsed = parse_contract_id(contract_id)
        if parsed is None:
            return []  # Not an NFL contract we can price

        # Cooldown check
        if not self._check_cooldown(contract_id):
            return []

        # Need season/week set
        if self._current_week is None:
            logger.debug("Week not set, skipping %s", contract_id)
            return []

        # Get or compute prediction for this game
        prediction = self._get_prediction(parsed)
        if prediction is None:
            return []  # Model couldn't produce a prediction

        # Convert model output to probability for this contract type
        model_prob = self._contract_probability(parsed, prediction)
        if model_prob is None:
            return []

        # Get market price (mid-point or last trade)
        market_price_cents = self._get_market_price(market_data, orderbook)
        if market_price_cents is None:
            return []

        # Calculate edge
        model_prob_cents = round(model_prob * 100)
        edge_cents = model_prob_cents - market_price_cents

        # Determine side and edge magnitude
        if edge_cents >= self._config.min_edge_cents:
            # Model says YES is underpriced → buy YES
            side = Side.YES
            price_cents = market_price_cents  # Bid at current market
            effective_edge = edge_cents
        elif edge_cents <= -self._config.min_edge_cents:
            # Model says NO is underpriced → buy NO
            side = Side.NO
            price_cents = 100 - market_price_cents  # NO price is complement
            effective_edge = abs(edge_cents)
            # Flip model_prob for NO side
            model_prob = 1.0 - model_prob
        else:
            return []  # Edge below threshold

        # Meta-labeling quality filter
        meta_prob = 1.0
        if self._config.use_meta_labeling and self._meta_classifier is not None:
            meta_prob = self._compute_meta_prob(prediction, market_price_cents)
            if meta_prob < self._config.min_meta_prob:
                logger.debug(
                    "Meta-label rejected %s: prob=%.2f < %.2f",
                    contract_id,
                    meta_prob,
                    self._config.min_meta_prob,
                )
                return []

        # Kelly sizing
        quantity = self._compute_quantity(
            model_prob=model_prob,
            price_cents=price_cents,
            meta_prob=meta_prob,
        )
        if quantity <= 0:
            return []

        # Build signal
        signal = Signal(
            contract_id=contract_id,
            side=side,
            quantity=quantity,
            price_cents=price_cents,
            edge_cents=effective_edge,
            model_prob=model_prob,
            confidence=1.0 - (prediction.spread_std / 20.0),  # Normalize std to confidence
            strategy=self.name,
            meta_prob=meta_prob,
        )

        self._record_signal(contract_id)

        logger.info(
            "Signal: %s %s %dx @%d¢ edge=%d¢ model=%.1f%% meta=%.2f",
            side.value.upper(),
            contract_id,
            quantity,
            price_cents,
            effective_edge,
            model_prob * 100,
            meta_prob,
        )

        return [signal]

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        """Lazy-load the GameOutcomeModel if not injected."""
        if self._model is not None:
            return

        try:
            from py.models.game_outcome_model_fixed import GameOutcomeModel

            self._model = GameOutcomeModel(model_dir=self._config.model_dir)
            self._model.load_models()
            logger.info(
                "Loaded GameOutcomeModel from %s (%d features)",
                self._config.model_dir,
                len(self._model.feature_names),
            )
        except Exception:
            logger.exception("Failed to load GameOutcomeModel")

    def _get_prediction(self, parsed: ParsedContract) -> GamePrediction | None:
        """Get or compute a cached prediction for this game.

        Predictions are cached per game_key because multiple contracts
        (spread, total, win) can reference the same matchup. The cache
        has a TTL to pick up model updates during a session.
        """
        if not parsed.away_team:
            # Win markets don't have opponent in the ID — can't predict
            logger.debug("Win contract without opponent, skipping")
            return None

        game_key = parsed.game_key
        cached = self._prediction_cache.get(game_key)
        if cached is not None:
            age = (datetime.now(UTC) - cached.predicted_at).total_seconds()
            if age < self._config.prediction_cache_seconds:
                return cached

        # Run model prediction
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

            # Also get raw features for meta-labeling
            features = {}
            if hasattr(self._model, "feature_extractor"):
                try:
                    features = self._model.feature_extractor.extract_game_features(
                        parsed.home_team, parsed.away_team,
                        parsed.season, self._current_week,
                    )
                except Exception:
                    logger.debug("Feature extraction for meta-labeling failed")

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
            logger.debug(
                "Predicted %s@%s: spread=%.1f, total=%.1f, win_prob=%.3f",
                parsed.away_team,
                parsed.home_team,
                pred.home_spread,
                pred.total_points,
                pred.home_win_probability,
            )
            return pred

        except Exception:
            logger.exception(
                "Prediction failed for %s@%s",
                parsed.away_team,
                parsed.home_team,
            )
            return None

    def _contract_probability(
        self, parsed: ParsedContract, pred: GamePrediction
    ) -> float | None:
        """Convert model prediction to a probability for the specific contract type.

        Each contract type maps to a different model output:
        - win: home_win_probability (direct)
        - spread: P(home covers spread) from normal CDF
        - total: P(total > line) from normal CDF
        """
        if parsed.contract_type == "win":
            return pred.home_win_probability

        if parsed.contract_type == "spread":
            if parsed.line is None:
                return None
            # P(home margin > -line) = P(home covers)
            # Model predicts home_spread (positive = home favored)
            # Contract: YES = home covers the line
            # If line=-3.5 and model says home_spread=-2.0, then
            # home margin expected = -2.0, need margin > 3.5 → unlikely
            #
            # Use normal CDF: P(margin > -line) where margin ~ N(spread, std)
            from scipy.stats import norm

            # spread is home perspective: negative = home underdog
            # line from contract: positive means home must win by that much
            prob = float(norm.cdf(
                pred.home_spread + parsed.line,
                loc=0,
                scale=pred.spread_std,
            ))
            return max(0.01, min(0.99, prob))

        if parsed.contract_type == "total":
            if parsed.line is None:
                return None
            # P(total > line) using normal CDF
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
        """Extract the market price in cents for edge calculation.

        Prefers mid-price from the order book (most accurate), falls
        back to last trade price, then to yes_ask.
        """
        if orderbook is not None and orderbook.mid_price_cents is not None:
            return int(orderbook.mid_price_cents)
        if market_data.last_price_cents and 1 <= market_data.last_price_cents <= 99:
            return market_data.last_price_cents
        if market_data.yes_ask_cents and 1 <= market_data.yes_ask_cents <= 99:
            return market_data.yes_ask_cents
        return None

    def _compute_meta_prob(
        self, pred: GamePrediction, market_price_cents: int
    ) -> float:
        """Run meta-labeling classifier on this trade opportunity.

        Builds meta-features from the model prediction and game features,
        then asks the meta-classifier for a quality probability.
        """
        if self._meta_classifier is None:
            return 1.0

        try:
            prediction_dict = {
                "predicted_spread": pred.home_spread,
                "probability": pred.home_win_probability,
                "vegas_spread": 0.0,  # We don't have vegas in prediction market context
                "uncertainty": pred.spread_std,
            }
            meta_features = self._meta_classifier.compute_meta_features_from_prediction(
                prediction=prediction_dict,
                game_features=pred.features,
            )
            return self._meta_classifier.predict_probability(meta_features)
        except Exception:
            logger.debug("Meta-labeling failed, using default 1.0")
            return 1.0

    def _compute_quantity(
        self,
        model_prob: float,
        price_cents: int,
        meta_prob: float,
    ) -> int:
        """Compute position size using Kelly criterion.

        Kelly fraction = (p * b - q) / b
        where p = model_prob, q = 1-p, b = net payout per dollar risked.

        For binary contracts at price P cents:
            Win: receive 100, pay P → net profit = 100-P, so b = (100-P)/P
            Lose: pay P → net loss = P

        Kelly = (p * (100-P)/P - (1-p)) / ((100-P)/P)
             = p - P/100  (after simplification, edge in probability units)

        Then scale by meta-labeling multiplier and Kelly fraction cap.
        """
        if price_cents <= 0 or price_cents >= 100:
            return 0

        # Binary contract Kelly
        p = model_prob
        cost = price_cents / 100.0  # Fraction of $1
        payout_ratio = (100 - price_cents) / price_cents  # b = (100-P)/P

        if payout_ratio <= 0:
            return 0

        kelly_raw = (p * payout_ratio - (1 - p)) / payout_ratio

        if kelly_raw <= 0:
            return 0

        # Cap Kelly fraction
        kelly_capped = min(kelly_raw, self._config.kelly_fraction_cap)

        # Scale by meta-labeling quality
        if self._config.use_meta_labeling:
            # meta_prob < 0.4 → 0 (already filtered above)
            # meta_prob 0.4-0.6 → linear 0-1
            # meta_prob > 0.6 → 1.0
            if meta_prob < 0.4:
                meta_mult = 0.0
            elif meta_prob > 0.6:
                meta_mult = 1.0
            else:
                meta_mult = (meta_prob - 0.4) / 0.2
            kelly_capped *= meta_mult

        # Convert Kelly fraction to contract quantity
        # Kelly fraction × bankroll = total dollars to risk
        # At price P cents, each contract costs P cents
        bankroll_dollars = self._config.bankroll_cents / 100.0
        dollars_to_risk = kelly_capped * bankroll_dollars
        quantity = int(dollars_to_risk / cost)

        # Cap
        quantity = min(quantity, self._config.max_contracts_per_signal)

        return max(0, quantity)
