"""
Residual Arbitrage Model

Identifies and exploits pricing inefficiencies when sportsbooks disagree.
Instead of pure arbitrage (guaranteed profit), this finds "residual" arbitrage
where model confidence + market disagreement creates +EV opportunities.

Key concepts:
1. Market consensus as truth signal (wisdom of crowds)
2. Outlier detection for mispriced lines
3. Correlation arbitrage across related markets
4. Synthetic positions from multiple bets

Expected impact: +0.5-1% ROI from systematic mispricing exploitation
"""

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sklearn.covariance import EllipticEnvelope

logger = logging.getLogger(__name__)


@dataclass
class MarketLine:
    """Single market line from a book"""

    book: str
    market: str  # 'spread', 'total', 'moneyline'
    team: str
    line: float
    price: int  # American odds
    timestamp: datetime
    volume: float | None = None  # Betting volume if available

    @property
    def implied_prob(self) -> float:
        """No-vig implied probability"""
        if self.price > 0:
            return 100 / (self.price + 100)
        else:
            return abs(self.price) / (abs(self.price) + 100)

    @property
    def fair_line(self) -> float:
        """Convert to fair odds (remove vig)"""
        # Approximate vig removal
        if self.price < 0:
            return self.price + 5  # Rough approximation
        else:
            return self.price - 5


@dataclass
class ArbitrageOpportunity:
    """Identified arbitrage or quasi-arbitrage opportunity"""

    type: str  # 'pure', 'residual', 'correlation', 'synthetic'
    markets: list[MarketLine]
    model_prob: float | None
    consensus_prob: float
    outlier_book: str | None
    edge: float
    confidence: float
    required_stakes: dict[str, float]  # book -> stake amount
    expected_profit: float
    risk_level: str  # 'low', 'medium', 'high'


class MarketConsensus:
    """Calculate market consensus from multiple books"""

    @staticmethod
    def calculate_consensus(
        lines: list[MarketLine], weights: dict[str, float] | None = None
    ) -> tuple[float, float]:
        """
        Calculate consensus line and probability.

        Sharp books get higher weight in consensus.
        """
        if not lines:
            return 0.0, 0.5

        # Default weights (sharp books weighted higher)
        if weights is None:
            weights = {
                "Pinnacle": 2.0,
                "Bookmaker": 2.0,
                "Circa": 1.8,
                "BetOnline": 1.3,
                "Heritage": 1.2,
                "DraftKings": 1.0,
                "FanDuel": 1.0,
                "BetMGM": 0.9,
                "Caesars": 0.9,
            }

        # Calculate weighted consensus
        total_weight = 0
        weighted_prob_sum = 0
        weighted_line_sum = 0

        for line in lines:
            w = weights.get(line.book, 1.0)

            # Volume weighting if available
            if line.volume:
                w *= np.log1p(line.volume) / 10  # Log scale for volume

            total_weight += w
            weighted_prob_sum += w * line.implied_prob
            weighted_line_sum += w * line.line

        consensus_prob = weighted_prob_sum / total_weight if total_weight > 0 else 0.5
        consensus_line = weighted_line_sum / total_weight if total_weight > 0 else 0.0

        return consensus_line, consensus_prob

    @staticmethod
    def calculate_dispersion(lines: list[MarketLine]) -> float:
        """Calculate market dispersion (disagreement level)"""
        if len(lines) < 2:
            return 0.0

        probs = [line.implied_prob for line in lines]
        return np.std(probs)


class OutlierDetector:
    """Detect outlier lines that may be mispriced"""

    def __init__(self, contamination: float = 0.1):
        """
        Args:
            contamination: Expected proportion of outliers
        """
        self.contamination = contamination
        self.detector = EllipticEnvelope(contamination=contamination, support_fraction=0.9)

    def find_outliers(
        self, lines: list[MarketLine], features: list[str] | None = None
    ) -> list[tuple[MarketLine, float]]:
        """
        Find outlier lines using robust covariance.

        Returns:
            List of (line, outlier_score) tuples
        """
        if len(lines) < 5:
            return []

        # Extract features
        if features is None:
            # Default: use line value and implied probability
            X = np.array([[line.line, line.implied_prob] for line in lines])
        else:
            # Custom features
            X = self._extract_features(lines, features)

        # Fit detector
        self.detector.fit(X)

        # Get outlier scores (negative = outlier)
        scores = self.detector.score_samples(X)

        # Identify outliers
        outliers = []
        for i, score in enumerate(scores):
            if score < 0:  # Outlier
                outliers.append((lines[i], abs(score)))

        return sorted(outliers, key=lambda x: x[1], reverse=True)

    def _extract_features(self, lines: list[MarketLine], features: list[str]) -> np.ndarray:
        """Extract specified features from lines"""
        # Simplified for demo
        return np.array([[line.line, line.implied_prob] for line in lines])


class CorrelationArbitrage:
    """Find arbitrage across correlated markets"""

    # Historical correlations between markets
    MARKET_CORRELATIONS = {
        ("spread", "total"): {
            "favorite_cover_over": 0.15,  # Favorites covering correlates with overs
            "dog_cover_under": 0.10,  # Dogs covering correlates with unders
        },
        ("team_total", "spread"): {
            "correlation": 0.60  # Team totals highly correlated with spreads
        },
        ("first_half", "game"): {"correlation": 0.75},  # First half outcomes predict game
    }

    @staticmethod
    def find_correlation_arbs(
        spreads: list[MarketLine], totals: list[MarketLine], model_correlation: float = 0.15
    ) -> list[ArbitrageOpportunity]:
        """
        Find arbitrage opportunities using correlation.

        Example: If spread and total are mispriced relative to correlation,
        bet both to exploit the inefficiency.
        """
        opportunities = []

        # Get consensus for each market
        spread_consensus = MarketConsensus.calculate_consensus(spreads)[1]
        total_consensus = MarketConsensus.calculate_consensus(totals)[1]

        # Check each spread/total combination
        for spread_line in spreads:
            for total_line in totals:
                # Calculate joint probability using correlation
                joint_prob = CorrelationArbitrage._calculate_joint_prob(
                    spread_line.implied_prob, total_line.implied_prob, model_correlation
                )

                # Expected from consensus
                expected_joint = CorrelationArbitrage._calculate_joint_prob(
                    spread_consensus, total_consensus, model_correlation
                )

                # Edge from correlation mispricing
                edge = joint_prob - expected_joint

                if abs(edge) > 0.03:  # 3% edge threshold
                    # Calculate stakes for correlation bet
                    stakes = CorrelationArbitrage._calculate_correlation_stakes(
                        spread_line, total_line, edge
                    )

                    opp = ArbitrageOpportunity(
                        type="correlation",
                        markets=[spread_line, total_line],
                        model_prob=joint_prob,
                        consensus_prob=expected_joint,
                        outlier_book=None,
                        edge=edge,
                        confidence=min(0.8, abs(edge) * 10),  # Confidence from edge size
                        required_stakes=stakes,
                        expected_profit=abs(edge) * 100,  # Rough estimate
                        risk_level="medium",
                    )
                    opportunities.append(opp)

        return opportunities

    @staticmethod
    def _calculate_joint_prob(prob1: float, prob2: float, correlation: float) -> float:
        """Calculate joint probability with correlation"""
        # Using Gaussian copula approximation
        from scipy.stats import norm

        # Convert to z-scores
        z1 = norm.ppf(prob1)
        z2 = norm.ppf(prob2)

        # Joint CDF with correlation
        mean = [0, 0]
        cov = [[1, correlation], [correlation, 1]]

        # Approximate joint probability
        from scipy.stats import multivariate_normal

        joint = multivariate_normal.cdf([z1, z2], mean, cov)

        return joint

    @staticmethod
    def _calculate_correlation_stakes(
        line1: MarketLine, line2: MarketLine, edge: float
    ) -> dict[str, float]:
        """Calculate optimal stakes for correlation arbitrage"""
        # Simplified stake calculation
        base_stake = 100  # $100 base

        # Adjust stakes based on edge
        stake1 = base_stake * (1 + edge)
        stake2 = base_stake * (1 - edge)

        return {line1.book: stake1, line2.book: stake2}


class SyntheticArbitrage:
    """Create synthetic positions from multiple bets"""

    @staticmethod
    def find_synthetic_arbs(all_markets: dict[str, list[MarketLine]]) -> list[ArbitrageOpportunity]:
        """
        Find synthetic arbitrage by combining multiple markets.

        Example: Create synthetic over by betting:
        - Home team over their total
        - Away team over their total
        Compare to game total for arbitrage.
        """
        opportunities = []

        # Example: Team totals vs game total
        if (
            "home_total" in all_markets
            and "away_total" in all_markets
            and "game_total" in all_markets
        ):
            home_totals = all_markets["home_total"]
            away_totals = all_markets["away_total"]
            game_totals = all_markets["game_total"]

            for home_line in home_totals:
                for away_line in away_totals:
                    # Synthetic total from team totals
                    synthetic_total = home_line.line + away_line.line

                    # Compare to game totals
                    for game_line in game_totals:
                        diff = synthetic_total - game_line.line

                        # Significant difference = opportunity
                        if abs(diff) > 2:  # 2+ point difference
                            # Calculate synthetic position profit
                            synthetic_prob = (home_line.implied_prob + away_line.implied_prob) / 2
                            edge = synthetic_prob - game_line.implied_prob

                            if abs(edge) > 0.04:  # 4% edge
                                opp = ArbitrageOpportunity(
                                    type="synthetic",
                                    markets=[home_line, away_line, game_line],
                                    model_prob=synthetic_prob,
                                    consensus_prob=game_line.implied_prob,
                                    outlier_book=game_line.book,
                                    edge=edge,
                                    confidence=0.7,
                                    required_stakes={
                                        home_line.book: 100,
                                        away_line.book: 100,
                                        game_line.book: -200,  # Opposite side
                                    },
                                    expected_profit=edge * 200,
                                    risk_level="high",  # Synthetic = higher risk
                                )
                                opportunities.append(opp)

        return opportunities


class ResidualArbitrageEngine:
    """Main engine for finding and exploiting residual arbitrage"""

    def __init__(self):
        self.outlier_detector = OutlierDetector()
        self.recent_opportunities: list[ArbitrageOpportunity] = []

    def find_all_opportunities(
        self, markets: dict[str, list[MarketLine]], model_probs: dict[str, float] | None = None
    ) -> list[ArbitrageOpportunity]:
        """
        Find all types of arbitrage opportunities.

        Args:
            markets: Dict of market_type -> list of lines
            model_probs: Optional model probabilities for validation

        Returns:
            List of arbitrage opportunities sorted by edge
        """
        all_opportunities = []

        # 1. Pure arbitrage (guaranteed profit)
        pure_arbs = self._find_pure_arbitrage(markets)
        all_opportunities.extend(pure_arbs)

        # 2. Residual arbitrage (outlier + model edge)
        residual_arbs = self._find_residual_arbitrage(markets, model_probs)
        all_opportunities.extend(residual_arbs)

        # 3. Correlation arbitrage
        if "spread" in markets and "total" in markets:
            corr_arbs = CorrelationArbitrage.find_correlation_arbs(
                markets["spread"], markets["total"]
            )
            all_opportunities.extend(corr_arbs)

        # 4. Synthetic arbitrage
        synthetic_arbs = SyntheticArbitrage.find_synthetic_arbs(markets)
        all_opportunities.extend(synthetic_arbs)

        # Sort by edge
        all_opportunities.sort(key=lambda x: abs(x.edge), reverse=True)

        # Store for tracking
        self.recent_opportunities = all_opportunities[:20]  # Keep top 20

        return all_opportunities

    def _find_pure_arbitrage(
        self, markets: dict[str, list[MarketLine]]
    ) -> list[ArbitrageOpportunity]:
        """Find guaranteed profit opportunities"""
        opportunities = []

        for market_type, lines in markets.items():
            if len(lines) < 2:
                continue

            # Check each pair of lines
            for i, line1 in enumerate(lines):
                for line2 in lines[i + 1 :]:
                    # Check if opposite sides create arbitrage
                    total_prob = line1.implied_prob + line2.implied_prob

                    if total_prob < 0.99:  # Less than 99% = arbitrage!
                        (1 / total_prob - 1) * 100

                        # Calculate required stakes
                        stake1 = 100 / line1.implied_prob
                        stake2 = 100 / line2.implied_prob
                        total_stake = stake1 + stake2
                        guaranteed_profit = 100 - total_stake

                        opp = ArbitrageOpportunity(
                            type="pure",
                            markets=[line1, line2],
                            model_prob=None,
                            consensus_prob=0.5,  # Not relevant for pure arb
                            outlier_book=None,
                            edge=1 - total_prob,
                            confidence=1.0,  # Guaranteed
                            required_stakes={line1.book: stake1, line2.book: stake2},
                            expected_profit=guaranteed_profit,
                            risk_level="low",  # No risk!
                        )
                        opportunities.append(opp)

        return opportunities

    def _find_residual_arbitrage(
        self, markets: dict[str, list[MarketLine]], model_probs: dict[str, float] | None
    ) -> list[ArbitrageOpportunity]:
        """Find arbitrage from outliers + model edge"""
        opportunities = []

        for market_type, lines in markets.items():
            if len(lines) < 5:  # Need enough lines for outlier detection
                continue

            # Find outliers
            outliers = self.outlier_detector.find_outliers(lines)

            # Calculate consensus
            consensus_line, consensus_prob = MarketConsensus.calculate_consensus(lines)
            dispersion = MarketConsensus.calculate_dispersion(lines)

            # Check each outlier
            for outlier_line, outlier_score in outliers:
                # Get model probability if available
                model_prob = model_probs.get(market_type) if model_probs else None

                # Calculate edge
                if model_prob:
                    # Model + outlier edge
                    edge = model_prob - outlier_line.implied_prob
                else:
                    # Pure outlier edge (vs consensus)
                    edge = consensus_prob - outlier_line.implied_prob

                # High dispersion + outlier + edge = opportunity
                if dispersion > 0.05 and abs(edge) > 0.03:
                    confidence = min(0.9, outlier_score / 10 + abs(edge) * 5)

                    opp = ArbitrageOpportunity(
                        type="residual",
                        markets=[outlier_line],
                        model_prob=model_prob,
                        consensus_prob=consensus_prob,
                        outlier_book=outlier_line.book,
                        edge=edge,
                        confidence=confidence,
                        required_stakes={outlier_line.book: 100},
                        expected_profit=edge * 100,
                        risk_level="medium",
                    )
                    opportunities.append(opp)

        return opportunities

    def generate_report(self) -> str:
        """Generate arbitrage report"""
        report = []
        report.append("=" * 80)
        report.append("RESIDUAL ARBITRAGE REPORT")
        report.append("=" * 80)
        report.append("")

        if not self.recent_opportunities:
            report.append("No arbitrage opportunities found")
            return "\n".join(report)

        # Group by type
        by_type = {}
        for opp in self.recent_opportunities:
            if opp.type not in by_type:
                by_type[opp.type] = []
            by_type[opp.type].append(opp)

        # Summary
        report.append("OPPORTUNITY SUMMARY:")
        for arb_type, opps in by_type.items():
            total_edge = sum(abs(o.edge) for o in opps)
            avg_confidence = np.mean([o.confidence for o in opps])
            report.append(f"  {arb_type.upper()}:")
            report.append(f"    Count: {len(opps)}")
            report.append(f"    Total edge: {total_edge:.1%}")
            report.append(f"    Avg confidence: {avg_confidence:.1%}")
        report.append("")

        # Top opportunities
        report.append("TOP 5 OPPORTUNITIES:")
        for opp in self.recent_opportunities[:5]:
            report.append(f"\n{opp.type.upper()} ARBITRAGE:")
            report.append(f"  Edge: {opp.edge:.2%}")
            report.append(f"  Confidence: {opp.confidence:.1%}")
            report.append(f"  Risk: {opp.risk_level}")
            report.append(f"  Expected profit: ${opp.expected_profit:.2f}")

            if opp.outlier_book:
                report.append(f"  Outlier: {opp.outlier_book}")

            report.append("  Markets:")
            for market in opp.markets:
                report.append(f"    {market.book}: {market.team} {market.line} @ {market.price:+d}")

        return "\n".join(report)


def demo_residual_arbitrage():
    """Demonstrate residual arbitrage detection"""

    # Create sample market lines with more variation for demo
    markets = {
        "spread": [
            MarketLine("Pinnacle", "spread", "KC -3", -3, -105, datetime.now()),
            MarketLine("DraftKings", "spread", "KC -3", -3, -110, datetime.now()),
            MarketLine("FanDuel", "spread", "KC -3", -3, -108, datetime.now()),
            MarketLine("BetMGM", "spread", "KC -3", -2.5, -115, datetime.now()),  # Different line
            MarketLine("Caesars", "spread", "KC -3", -3.5, +120, datetime.now()),  # Big outlier!
            MarketLine("BetOnline", "spread", "KC -3", -3, -107, datetime.now()),
            MarketLine(
                "LocalBook", "spread", "BUF +3", 3, +105, datetime.now()
            ),  # Arbitrage opportunity
        ],
        "total": [
            MarketLine("Pinnacle", "total", "Over 51", 51, -108, datetime.now()),
            MarketLine("DraftKings", "total", "Over 51.5", 51.5, -110, datetime.now()),
            MarketLine("FanDuel", "total", "Over 51", 51, -105, datetime.now()),
            MarketLine("BetMGM", "total", "Over 50.5", 50.5, -105, datetime.now()),
            MarketLine("Caesars", "total", "Over 52", 52, -120, datetime.now()),
        ],
        "home_total": [
            MarketLine("DraftKings", "home_total", "KC Over 27", 27, -115, datetime.now()),
            MarketLine("FanDuel", "home_total", "KC Over 26.5", 26.5, -110, datetime.now()),
        ],
        "away_total": [
            MarketLine("DraftKings", "away_total", "BUF Over 24", 24, -110, datetime.now()),
            MarketLine("FanDuel", "away_total", "BUF Over 24.5", 24.5, -115, datetime.now()),
        ],
    }

    # Model probabilities
    model_probs = {
        "spread": 0.58,  # 58% KC covers
        "total": 0.52,  # 52% over
    }

    # Create engine and find opportunities
    engine = ResidualArbitrageEngine()
    opportunities = engine.find_all_opportunities(markets, model_probs)

    # Generate and print report
    report = engine.generate_report()
    print(report)

    # Show detailed opportunity analysis
    if opportunities:
        print("\n" + "=" * 80)
        print("DETAILED OPPORTUNITY ANALYSIS")
        print("=" * 80)

        for i, opp in enumerate(opportunities[:3], 1):
            print(f"\nOPPORTUNITY #{i}: {opp.type.upper()}")
            print(f"Edge: {opp.edge:.2%}")
            print(f"Confidence: {opp.confidence:.0%}")
            print(f"Risk Level: {opp.risk_level}")

            print("\nRequired Stakes:")
            for book, stake in opp.required_stakes.items():
                print(f"  {book}: ${stake:,.2f}")

            print(f"\nExpected Profit: ${opp.expected_profit:,.2f}")

            if opp.type == "residual":
                print(f"Consensus Probability: {opp.consensus_prob:.1%}")
                if opp.model_prob:
                    print(f"Model Probability: {opp.model_prob:.1%}")
                print(f"Outlier Book: {opp.outlier_book}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    demo_residual_arbitrage()
