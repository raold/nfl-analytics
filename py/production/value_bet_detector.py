"""
Value Bet Detector with LLM Explanations.

Enhances the EV bet selector with semantic analysis and LLM-generated explanations.
Provides human-readable justifications for value bets to support decision-making.

Key Features:
- Identifies value bets using model probabilities vs market odds
- Generates LLM explanations for each bet recommendation
- Incorporates semantic signals (sentiment, injury news) into explanations
- Supports human-in-the-loop decision workflow

Usage:
    detector = ValueBetDetector()
    bets = detector.detect_value_bets(predictions, min_edge=0.03)
    for bet in bets:
        print(f"{bet.game}: {bet.recommendation}")
        print(f"Justification: {bet.explanation}")
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class BetType(Enum):
    """Types of bets."""

    SPREAD = "spread"
    MONEYLINE = "moneyline"
    TOTAL = "total"
    PROP = "prop"


class Confidence(Enum):
    """Confidence levels for recommendations."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ValueBet:
    """A detected value bet with explanation."""

    # Game info
    game_id: str
    home_team: str
    away_team: str
    game_time: datetime | None

    # Bet details
    bet_type: BetType
    bet_side: str  # e.g., "KC -3.5", "Over 47.5"
    odds: int  # American odds

    # Value metrics
    model_prob: float
    market_implied: float
    edge: float  # model_prob - market_implied
    expected_value: float  # EV per $100

    # Confidence and sizing
    confidence: Confidence
    kelly_fraction: float
    recommended_size: float  # Fractional Kelly bet size

    # Explanations
    explanation: str = ""  # LLM-generated justification
    key_factors: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)

    # Semantic signals (from news/social)
    sentiment_score: float | None = None
    injury_impact: float | None = None
    media_attention: float | None = None

    @property
    def edge_pct(self) -> str:
        """Edge as percentage string."""
        return f"{self.edge * 100:.1f}%"

    @property
    def is_actionable(self) -> bool:
        """Check if bet meets minimum thresholds."""
        return self.edge >= 0.02 and self.confidence != Confidence.LOW

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "game_id": self.game_id,
            "matchup": f"{self.away_team} @ {self.home_team}",
            "bet": self.bet_side,
            "odds": self.odds,
            "model_prob": round(self.model_prob, 3),
            "market_implied": round(self.market_implied, 3),
            "edge": round(self.edge, 3),
            "edge_pct": self.edge_pct,
            "ev_per_100": round(self.expected_value, 2),
            "confidence": self.confidence.value,
            "recommended_size": round(self.recommended_size, 2),
            "explanation": self.explanation,
            "key_factors": self.key_factors,
            "risks": self.risks,
        }


class ValueBetDetector:
    """
    Detects value bets and generates LLM explanations.

    Combines quantitative edge detection with semantic analysis
    to provide justified bet recommendations.
    """

    def __init__(
        self,
        min_edge: float = 0.03,
        max_bets_per_week: int = 10,
        use_llm_explanations: bool = True,
        inference_client: Any | None = None,
    ):
        """
        Initialize value bet detector.

        Args:
            min_edge: Minimum edge to consider (default 3%)
            max_bets_per_week: Maximum bets to recommend per week
            use_llm_explanations: Whether to generate LLM explanations
            inference_client: SemanticInferenceClient for explanations
        """
        self.min_edge = min_edge
        self.max_bets_per_week = max_bets_per_week
        self.use_llm_explanations = use_llm_explanations
        self._inference = inference_client

    def _get_inference_client(self):
        """Lazy-load inference client."""
        if self._inference is None and self.use_llm_explanations:
            try:
                from py.semantic.models.inference import SemanticInferenceClient

                self._inference = SemanticInferenceClient()
                logger.info("Initialized inference client for LLM explanations")
            except Exception as e:
                logger.warning(f"Could not initialize inference client: {e}")
                self.use_llm_explanations = False
        return self._inference

    def detect_value_bets(
        self,
        predictions: list[dict],
        market_odds: dict[str, dict] | None = None,
        semantic_features: dict[str, dict] | None = None,
    ) -> list[ValueBet]:
        """
        Detect value bets from model predictions.

        Args:
            predictions: List of prediction dicts with:
                - game_id, home_team, away_team
                - model_prob (home win probability)
                - spread_line, spread_odds (optional)
            market_odds: Optional dict of game_id -> odds data
            semantic_features: Optional dict of game_id -> semantic signals

        Returns:
            List of ValueBet objects sorted by edge
        """
        value_bets = []

        for pred in predictions:
            try:
                bet = self._evaluate_prediction(pred, market_odds, semantic_features)
                if bet and bet.edge >= self.min_edge:
                    value_bets.append(bet)
            except Exception as e:
                logger.warning(f"Failed to evaluate prediction: {e}")
                continue

        # Sort by edge (highest first)
        value_bets.sort(key=lambda b: b.edge, reverse=True)

        # Limit to max bets per week
        value_bets = value_bets[: self.max_bets_per_week]

        # Generate LLM explanations for top bets
        if self.use_llm_explanations:
            for bet in value_bets:
                try:
                    bet.explanation = self._generate_explanation(bet, semantic_features)
                except Exception as e:
                    logger.warning(f"Failed to generate explanation: {e}")
                    bet.explanation = self._generate_fallback_explanation(bet)

        return value_bets

    def _evaluate_prediction(
        self,
        pred: dict,
        market_odds: dict[str, dict] | None,
        semantic_features: dict[str, dict] | None,
    ) -> ValueBet | None:
        """Evaluate a single prediction for value."""
        game_id = pred.get("game_id", "")
        home_team = pred.get("home_team", "")
        away_team = pred.get("away_team", "")
        model_prob = pred.get("model_prob", 0.5)

        # Get market odds (use prediction or external source)
        if market_odds and game_id in market_odds:
            odds_data = market_odds[game_id]
            odds = odds_data.get("spread_odds", -110)
        else:
            odds = pred.get("spread_odds", -110)

        # Calculate implied probability
        market_implied = self._american_odds_to_prob(odds)

        # Calculate edge
        edge = model_prob - market_implied

        # Skip if below minimum
        if edge < self.min_edge * 0.5:  # Pre-filter before LLM
            return None

        # Calculate EV per $100
        ev = self._calculate_ev(model_prob, odds)

        # Calculate Kelly fraction
        kelly = self._kelly_criterion(model_prob, odds)

        # Determine confidence
        confidence = self._assess_confidence(edge, model_prob)

        # Get semantic signals if available
        sentiment = None
        injury_impact = None
        if semantic_features and game_id in semantic_features:
            sem = semantic_features[game_id]
            sentiment = sem.get("sentiment_differential", 0)
            injury_impact = sem.get("injury_advantage", 0)

        # Identify key factors
        key_factors = self._identify_key_factors(pred, edge, semantic_features)
        risks = self._identify_risks(pred, edge, semantic_features)

        return ValueBet(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            game_time=pred.get("game_time"),
            bet_type=BetType.SPREAD,
            bet_side=f"{home_team} {pred.get('spread_line', '-3')}",
            odds=odds,
            model_prob=model_prob,
            market_implied=market_implied,
            edge=edge,
            expected_value=ev,
            confidence=confidence,
            kelly_fraction=kelly,
            recommended_size=kelly * 0.25,  # Quarter Kelly
            key_factors=key_factors,
            risks=risks,
            sentiment_score=sentiment,
            injury_impact=injury_impact,
        )

    def _generate_explanation(
        self,
        bet: ValueBet,
        semantic_features: dict[str, dict] | None,
    ) -> str:
        """Generate LLM explanation for a value bet."""
        inference = self._get_inference_client()
        if not inference:
            return self._generate_fallback_explanation(bet)

        # Build context for LLM
        context = f"""
        Game: {bet.away_team} @ {bet.home_team}
        Bet: {bet.bet_side} at {bet.odds}
        Model probability: {bet.model_prob:.1%}
        Market implied: {bet.market_implied:.1%}
        Edge: {bet.edge:.1%}
        Expected Value: ${bet.expected_value:.2f} per $100

        Key factors supporting this bet:
        {chr(10).join(f'- {f}' for f in bet.key_factors)}

        Risks to consider:
        {chr(10).join(f'- {r}' for r in bet.risks)}
        """

        if bet.sentiment_score is not None:
            context += f"\n        Sentiment: {'Positive' if bet.sentiment_score > 0 else 'Negative'} ({bet.sentiment_score:.2f})"

        if bet.injury_impact is not None:
            context += f"\n        Injury advantage: {bet.injury_impact:.2f}"

        prompt = f"""Based on the following betting analysis, provide a 2-3 sentence justification
for why this bet offers value. Focus on the key factors and why the market may be mispricing this game.

{context}

Provide a concise, professional explanation:"""

        try:
            result = inference.summarize(prompt, max_length=100)
            return result.strip()
        except Exception as e:
            logger.warning(f"LLM explanation failed: {e}")
            return self._generate_fallback_explanation(bet)

    def _generate_fallback_explanation(self, bet: ValueBet) -> str:
        """Generate rule-based explanation when LLM unavailable."""
        parts = [
            f"Model identifies {bet.edge_pct} edge on {bet.bet_side}.",
        ]

        if bet.key_factors:
            parts.append(f"Key factor: {bet.key_factors[0]}.")

        if bet.confidence == Confidence.HIGH:
            parts.append("High confidence based on model consistency.")
        elif bet.confidence == Confidence.MEDIUM:
            parts.append("Moderate confidence; consider reduced sizing.")

        return " ".join(parts)

    def _identify_key_factors(
        self,
        pred: dict,
        edge: float,
        semantic_features: dict[str, dict] | None,
    ) -> list[str]:
        """Identify key factors supporting the bet."""
        factors = []

        # Edge strength
        if edge >= 0.05:
            factors.append("Strong model edge (5%+)")
        elif edge >= 0.03:
            factors.append("Solid model edge (3%+)")

        # Matchup advantages
        if pred.get("offensive_mismatch", 0) > 0.1:
            factors.append("Favorable offensive matchup")
        if pred.get("defensive_mismatch", 0) < -0.1:
            factors.append("Favorable defensive matchup")

        # Semantic signals
        game_id = pred.get("game_id", "")
        if semantic_features and game_id in semantic_features:
            sem = semantic_features[game_id]
            if sem.get("sentiment_differential", 0) > 0.2:
                factors.append("Positive sentiment momentum")
            if sem.get("injury_advantage", 0) > 0.2:
                factors.append("Injury situation favors this side")

        return factors[:4]  # Top 4 factors

    def _identify_risks(
        self,
        pred: dict,
        edge: float,
        semantic_features: dict[str, dict] | None,
    ) -> list[str]:
        """Identify risks for the bet."""
        risks = []

        # Edge uncertainty
        if edge < 0.04:
            risks.append("Edge near threshold - sizing uncertainty")

        # Variance
        if pred.get("home_off_epa_std_4wk", 0) > 0.2:
            risks.append("High offensive variance in recent games")

        # Semantic risks
        game_id = pred.get("game_id", "")
        if semantic_features and game_id in semantic_features:
            sem = semantic_features[game_id]
            if sem.get("home_key_player_risk", 0) > 0.5:
                risks.append("Key player injury risk detected")
            if sem.get("home_public_overreaction", 0) > 0.5:
                risks.append("Possible public overreaction in market")

        return risks[:3]  # Top 3 risks

    def _american_odds_to_prob(self, odds: int) -> float:
        """Convert American odds to implied probability."""
        if odds < 0:
            return abs(odds) / (abs(odds) + 100)
        else:
            return 100 / (odds + 100)

    def _calculate_ev(self, model_prob: float, odds: int) -> float:
        """Calculate expected value per $100 bet."""
        if odds < 0:
            win_amount = 100 / (abs(odds) / 100)
        else:
            win_amount = odds
        return (model_prob * win_amount) - ((1 - model_prob) * 100)

    def _kelly_criterion(self, model_prob: float, odds: int) -> float:
        """Calculate Kelly fraction."""
        if odds < 0:
            b = 100 / abs(odds)
        else:
            b = odds / 100

        q = 1 - model_prob
        kelly = (model_prob * b - q) / b

        return max(0, min(kelly, 0.25))  # Cap at 25%

    def _assess_confidence(self, edge: float, model_prob: float) -> Confidence:
        """Assess confidence level based on edge and probability."""
        if edge >= 0.05 and 0.4 <= model_prob <= 0.7:
            return Confidence.HIGH
        elif edge >= 0.03 and 0.35 <= model_prob <= 0.75:
            return Confidence.MEDIUM
        else:
            return Confidence.LOW

    def generate_weekly_report(self, value_bets: list[ValueBet]) -> str:
        """Generate a weekly value bet report."""
        if not value_bets:
            return "No value bets identified this week."

        lines = [
            "=" * 60,
            "WEEKLY VALUE BET REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Bets Identified: {len(value_bets)}",
            "=" * 60,
            "",
        ]

        for i, bet in enumerate(value_bets, 1):
            lines.extend(
                [
                    f"#{i}: {bet.away_team} @ {bet.home_team}",
                    f"    Bet: {bet.bet_side} ({bet.odds})",
                    f"    Edge: {bet.edge_pct} | EV: ${bet.expected_value:.2f}/100",
                    f"    Confidence: {bet.confidence.value.upper()}",
                    f"    Recommended: {bet.recommended_size:.1%} bankroll",
                    f"    Justification: {bet.explanation}",
                    "",
                ]
            )

        return "\n".join(lines)


def test_value_detector():
    """Test the value bet detector."""
    detector = ValueBetDetector(min_edge=0.02, use_llm_explanations=False)

    # Mock predictions
    predictions = [
        {
            "game_id": "2025_16_KC_BAL",
            "home_team": "BAL",
            "away_team": "KC",
            "model_prob": 0.58,
            "spread_line": "-3.5",
            "spread_odds": -110,
            "offensive_mismatch": 0.15,
        },
        {
            "game_id": "2025_16_SF_DAL",
            "home_team": "DAL",
            "away_team": "SF",
            "model_prob": 0.45,
            "spread_line": "+2.5",
            "spread_odds": -110,
        },
    ]

    print("Testing ValueBetDetector...")
    bets = detector.detect_value_bets(predictions)
    print(f"Found {len(bets)} value bets")

    for bet in bets:
        print(f"  {bet.bet_side}: {bet.edge_pct} edge")
        print(f"    {bet.explanation}")

    print()
    print(detector.generate_weekly_report(bets))


if __name__ == "__main__":
    test_value_detector()
