"""
Matchup Segmentation by Uncertainty

Segments games into confidence tiers based on model uncertainty,
allowing for differentiated betting strategies:
- High confidence: Larger bets, accept lower edges
- Medium confidence: Standard Kelly sizing
- Low confidence: Require higher edges, smaller bets

Uncertainty sources:
1. Epistemic: Model uncertainty from limited training data
2. Aleatoric: Inherent randomness in outcomes
3. Market: Disagreement between books
4. Situational: Injuries, weather, rest days

Expected impact: 20-30% reduction in losses on uncertain games
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


@dataclass
class GameContext:
    """Contextual factors affecting game uncertainty"""
    game_id: str
    home_team: str
    away_team: str

    # Injury impact (0-1 scale)
    home_injury_impact: float
    away_injury_impact: float

    # Rest differential (days)
    home_rest_days: int
    away_rest_days: int

    # Weather conditions
    weather_impact: float  # 0=dome/perfect, 1=extreme conditions
    temperature: Optional[float] = None
    wind_speed: Optional[float] = None
    precipitation: Optional[float] = None

    # Situational factors
    divisional_game: bool = False
    primetime: bool = False
    playoff_implications: bool = False
    revenge_game: bool = False
    lookahead_spot: bool = False
    letdown_spot: bool = False

    # Travel
    travel_distance: float = 0.0  # Miles for away team
    timezone_change: int = 0  # Hours difference


@dataclass
class ModelUncertainty:
    """Model's prediction uncertainty"""
    game_id: str

    # Prediction intervals
    spread_mean: float
    spread_std: float
    spread_confidence_interval: Tuple[float, float]  # 95% CI

    total_mean: float
    total_std: float
    total_confidence_interval: Tuple[float, float]

    # Ensemble disagreement
    model_disagreement: float  # Std dev across models

    # Feature importance uncertainty
    feature_stability: float  # How consistent are important features

    # Historical accuracy for similar games
    similar_games_accuracy: float
    similar_games_count: int


@dataclass
class MarketUncertainty:
    """Market-based uncertainty metrics"""
    game_id: str

    # Line movement
    spread_movement: float  # Total movement in points
    total_movement: float

    # Book disagreement
    spread_dispersion: float  # Std dev across books
    total_dispersion: float

    # Sharp vs public
    sharp_public_divergence: float  # How much they disagree

    # Volume metrics
    handle_ratio: float  # This game vs average
    sharp_action_unclear: bool  # Mixed sharp signals


@dataclass
class SegmentProfile:
    """Profile for an uncertainty segment"""
    name: str
    confidence_level: str  # 'high', 'medium', 'low'

    # Thresholds
    min_edge_required: float
    max_kelly_fraction: float
    max_bet_count: int  # Max bets in this segment

    # Historical performance
    historical_roi: float
    historical_win_rate: float
    avg_uncertainty_score: float

    # Strategy adjustments
    prefer_totals: bool = False
    prefer_dogs: bool = False
    avoid_favorites: bool = False
    require_clv: bool = False


class UncertaintyCalculator:
    """Calculate composite uncertainty scores"""

    @staticmethod
    def calculate_epistemic_uncertainty(
        model: ModelUncertainty,
        training_sample_size: int
    ) -> float:
        """
        Calculate model uncertainty from limited data.

        Higher with:
        - Fewer similar historical games
        - High model disagreement
        - Wide confidence intervals
        """
        # Base uncertainty from sample size
        size_factor = 1 / np.log1p(model.similar_games_count)

        # Model disagreement factor
        disagreement_factor = model.model_disagreement / 3.0  # Normalize by typical std

        # Confidence interval width
        spread_ci_width = model.spread_confidence_interval[1] - model.spread_confidence_interval[0]
        ci_factor = spread_ci_width / 14.0  # Normalize by typical game range

        # Feature stability (inverse)
        stability_factor = 1 - model.feature_stability

        # Weighted combination
        epistemic = (
            0.3 * size_factor +
            0.3 * disagreement_factor +
            0.2 * ci_factor +
            0.2 * stability_factor
        )

        return np.clip(epistemic, 0, 1)

    @staticmethod
    def calculate_aleatoric_uncertainty(
        context: GameContext
    ) -> float:
        """
        Calculate inherent randomness/unpredictability.

        Higher with:
        - Weather impact
        - Injuries to key players
        - Emotional games (rivalry, revenge)
        """
        # Weather uncertainty
        weather_factor = context.weather_impact

        # Injury uncertainty
        injury_factor = max(context.home_injury_impact, context.away_injury_impact)

        # Situational volatility
        situational_factors = [
            context.divisional_game,
            context.revenge_game,
            context.playoff_implications,
            context.primetime
        ]
        situational_factor = sum(situational_factors) / 8.0  # Normalize

        # Rest/travel fatigue uncertainty
        rest_differential = abs(context.home_rest_days - context.away_rest_days)
        rest_factor = 1 / (1 + np.exp(-0.5 * (rest_differential - 3)))  # Sigmoid

        travel_factor = context.travel_distance / 3000  # Normalize by cross-country

        # Weighted combination
        aleatoric = (
            0.3 * weather_factor +
            0.3 * injury_factor +
            0.2 * situational_factor +
            0.1 * rest_factor +
            0.1 * travel_factor
        )

        return np.clip(aleatoric, 0, 1)

    @staticmethod
    def calculate_market_uncertainty(
        market: MarketUncertainty
    ) -> float:
        """
        Calculate uncertainty from market signals.

        Higher with:
        - High line movement
        - Book disagreement
        - Sharp/public divergence
        - Unclear sharp action
        """
        # Line movement factor
        movement_factor = (market.spread_movement + market.total_movement / 2) / 7.0

        # Book disagreement
        dispersion_factor = (market.spread_dispersion + market.total_dispersion / 2) / 2.0

        # Sharp/public confusion
        divergence_factor = market.sharp_public_divergence
        unclear_factor = 0.5 if market.sharp_action_unclear else 0.0

        # Volume (inverse - low volume = high uncertainty)
        volume_factor = 1 / (1 + market.handle_ratio)

        # Weighted combination
        market_uncertainty = (
            0.25 * movement_factor +
            0.25 * dispersion_factor +
            0.20 * divergence_factor +
            0.15 * unclear_factor +
            0.15 * volume_factor
        )

        return np.clip(market_uncertainty, 0, 1)

    @staticmethod
    def calculate_composite_uncertainty(
        epistemic: float,
        aleatoric: float,
        market: float,
        weights: Optional[Tuple[float, float, float]] = None
    ) -> float:
        """
        Combine all uncertainty sources into composite score.

        Args:
            epistemic: Model uncertainty
            aleatoric: Inherent randomness
            market: Market disagreement
            weights: Optional weight tuple (default: equal weights)

        Returns:
            Composite uncertainty score (0-1)
        """
        if weights is None:
            weights = (0.4, 0.3, 0.3)  # Slightly favor model uncertainty

        composite = (
            weights[0] * epistemic +
            weights[1] * aleatoric +
            weights[2] * market
        )

        return np.clip(composite, 0, 1)


class MatchupSegmenter:
    """Segment games by uncertainty for differentiated strategies"""

    # Default segment profiles
    DEFAULT_SEGMENTS = {
        'high_confidence': SegmentProfile(
            name='High Confidence',
            confidence_level='high',
            min_edge_required=0.025,  # 2.5% edge
            max_kelly_fraction=0.30,  # 30% Kelly
            max_bet_count=10,
            historical_roi=0.045,
            historical_win_rate=0.56,
            avg_uncertainty_score=0.25
        ),
        'medium_confidence': SegmentProfile(
            name='Medium Confidence',
            confidence_level='medium',
            min_edge_required=0.035,  # 3.5% edge
            max_kelly_fraction=0.20,  # 20% Kelly
            max_bet_count=5,
            historical_roi=0.015,
            historical_win_rate=0.53,
            avg_uncertainty_score=0.50,
            prefer_totals=True  # Totals more stable
        ),
        'low_confidence': SegmentProfile(
            name='Low Confidence',
            confidence_level='low',
            min_edge_required=0.050,  # 5% edge required
            max_kelly_fraction=0.10,  # 10% Kelly max
            max_bet_count=2,
            historical_roi=-0.025,
            historical_win_rate=0.49,
            avg_uncertainty_score=0.75,
            prefer_dogs=True,  # Dogs have more value in chaos
            avoid_favorites=True,
            require_clv=True  # Must beat closing line
        )
    }

    def __init__(self, segments: Optional[Dict[str, SegmentProfile]] = None):
        self.segments = segments or self.DEFAULT_SEGMENTS
        self.game_segments: Dict[str, str] = {}  # game_id -> segment_name

    def segment_game(
        self,
        game_id: str,
        model_uncertainty: ModelUncertainty,
        game_context: GameContext,
        market_uncertainty: MarketUncertainty
    ) -> Tuple[str, float, Dict]:
        """
        Assign game to uncertainty segment.

        Returns:
            Tuple of (segment_name, uncertainty_score, details)
        """
        # Calculate uncertainty components
        epistemic = UncertaintyCalculator.calculate_epistemic_uncertainty(
            model_uncertainty,
            training_sample_size=1000  # Placeholder
        )

        aleatoric = UncertaintyCalculator.calculate_aleatoric_uncertainty(
            game_context
        )

        market = UncertaintyCalculator.calculate_market_uncertainty(
            market_uncertainty
        )

        # Composite score
        composite = UncertaintyCalculator.calculate_composite_uncertainty(
            epistemic, aleatoric, market
        )

        # Assign to segment
        if composite < 0.33:
            segment_name = 'high_confidence'
        elif composite < 0.67:
            segment_name = 'medium_confidence'
        else:
            segment_name = 'low_confidence'

        # Store assignment
        self.game_segments[game_id] = segment_name

        # Return details
        details = {
            'epistemic_uncertainty': epistemic,
            'aleatoric_uncertainty': aleatoric,
            'market_uncertainty': market,
            'composite_score': composite,
            'segment': self.segments[segment_name]
        }

        return segment_name, composite, details

    def adjust_bet_strategy(
        self,
        base_edge: float,
        base_kelly: float,
        segment_name: str
    ) -> Tuple[bool, float, str]:
        """
        Adjust betting strategy based on segment.

        Args:
            base_edge: Calculated edge
            base_kelly: Base Kelly fraction
            segment_name: Assigned segment

        Returns:
            Tuple of (should_bet, adjusted_kelly, reason)
        """
        segment = self.segments[segment_name]

        # Check minimum edge
        if base_edge < segment.min_edge_required:
            return False, 0.0, f"Edge {base_edge:.3f} below minimum {segment.min_edge_required:.3f}"

        # Adjust Kelly sizing
        adjusted_kelly = min(base_kelly, segment.max_kelly_fraction)

        # Further adjustments for low confidence
        if segment_name == 'low_confidence':
            # Extra conservative in chaos
            adjusted_kelly *= 0.7
            reason = "Ultra-conservative sizing for high uncertainty"
        elif segment_name == 'medium_confidence':
            # Moderate adjustment
            adjusted_kelly *= 0.85
            reason = "Moderate sizing for medium uncertainty"
        else:
            # High confidence - can be aggressive
            reason = "Standard sizing for high confidence game"

        return True, adjusted_kelly, reason

    def generate_segmentation_report(self) -> str:
        """Generate report on game segmentation"""
        report = []
        report.append("=" * 80)
        report.append("UNCERTAINTY SEGMENTATION REPORT")
        report.append("=" * 80)
        report.append("")

        # Count games per segment
        segment_counts = {}
        for segment_name in self.game_segments.values():
            if segment_name not in segment_counts:
                segment_counts[segment_name] = 0
            segment_counts[segment_name] += 1

        # Summary
        report.append("SEGMENT DISTRIBUTION:")
        for name, profile in self.segments.items():
            count = segment_counts.get(name, 0)
            pct = (count / len(self.game_segments) * 100) if self.game_segments else 0
            report.append(f"  {profile.name}:")
            report.append(f"    Games: {count} ({pct:.1f}%)")
            report.append(f"    Min edge: {profile.min_edge_required:.1%}")
            report.append(f"    Max Kelly: {profile.max_kelly_fraction:.1%}")
            report.append(f"    Historical ROI: {profile.historical_roi:+.1%}")
        report.append("")

        # Strategy adjustments
        report.append("STRATEGY ADJUSTMENTS BY SEGMENT:")
        report.append("")

        report.append("High Confidence Games:")
        report.append("  - Accept edges as low as 2.5%")
        report.append("  - Use up to 30% Kelly sizing")
        report.append("  - Maximum 10 bets in this segment")
        report.append("")

        report.append("Medium Confidence Games:")
        report.append("  - Require 3.5% minimum edge")
        report.append("  - Limit to 20% Kelly sizing")
        report.append("  - Prefer totals over spreads")
        report.append("  - Maximum 5 bets in this segment")
        report.append("")

        report.append("Low Confidence Games:")
        report.append("  - Require 5% minimum edge")
        report.append("  - Ultra-conservative 10% Kelly max")
        report.append("  - Prefer underdogs")
        report.append("  - Must achieve CLV to validate")
        report.append("  - Maximum 2 bets in this segment")

        return "\n".join(report)


def demo_uncertainty_segmentation():
    """Demonstrate uncertainty-based game segmentation"""

    # Create sample games with varying uncertainty
    games = [
        # High confidence game
        {
            'game_id': 'KC_vs_HOU',
            'model': ModelUncertainty(
                game_id='KC_vs_HOU',
                spread_mean=-10.5,
                spread_std=2.1,
                spread_confidence_interval=(-14.5, -6.5),
                total_mean=47.0,
                total_std=3.5,
                total_confidence_interval=(40, 54),
                model_disagreement=1.8,
                feature_stability=0.85,
                similar_games_accuracy=0.58,
                similar_games_count=150
            ),
            'context': GameContext(
                game_id='KC_vs_HOU',
                home_team='KC',
                away_team='HOU',
                home_injury_impact=0.1,
                away_injury_impact=0.05,
                home_rest_days=7,
                away_rest_days=7,
                weather_impact=0.0,  # Dome
                divisional_game=False,
                travel_distance=500
            ),
            'market': MarketUncertainty(
                game_id='KC_vs_HOU',
                spread_movement=0.5,
                total_movement=1.0,
                spread_dispersion=0.3,
                total_dispersion=0.5,
                sharp_public_divergence=0.1,
                handle_ratio=1.2,
                sharp_action_unclear=False
            )
        },

        # Medium confidence game
        {
            'game_id': 'BUF_at_MIA',
            'model': ModelUncertainty(
                game_id='BUF_at_MIA',
                spread_mean=-3.0,
                spread_std=3.5,
                spread_confidence_interval=(-9, 3),
                total_mean=50.0,
                total_std=4.5,
                total_confidence_interval=(41, 59),
                model_disagreement=3.2,
                feature_stability=0.65,
                similar_games_accuracy=0.52,
                similar_games_count=45
            ),
            'context': GameContext(
                game_id='BUF_at_MIA',
                home_team='MIA',
                away_team='BUF',
                home_injury_impact=0.25,
                away_injury_impact=0.15,
                home_rest_days=7,
                away_rest_days=10,
                weather_impact=0.3,  # Hot weather
                temperature=88,
                divisional_game=True,
                travel_distance=1200
            ),
            'market': MarketUncertainty(
                game_id='BUF_at_MIA',
                spread_movement=2.5,
                total_movement=3.0,
                spread_dispersion=1.1,
                total_dispersion=1.5,
                sharp_public_divergence=0.35,
                handle_ratio=0.9,
                sharp_action_unclear=True
            )
        },

        # Low confidence game
        {
            'game_id': 'CHI_at_GB',
            'model': ModelUncertainty(
                game_id='CHI_at_GB',
                spread_mean=-1.5,
                spread_std=5.2,
                spread_confidence_interval=(-11, 8),
                total_mean=41.0,
                total_std=6.0,
                total_confidence_interval=(29, 53),
                model_disagreement=4.5,
                feature_stability=0.45,
                similar_games_accuracy=0.48,
                similar_games_count=12
            ),
            'context': GameContext(
                game_id='CHI_at_GB',
                home_team='GB',
                away_team='CHI',
                home_injury_impact=0.4,  # Key injuries
                away_injury_impact=0.35,
                home_rest_days=4,  # Short week
                away_rest_days=10,
                weather_impact=0.8,  # Snow/wind
                temperature=25,
                wind_speed=25,
                divisional_game=True,
                revenge_game=True,
                playoff_implications=True
            ),
            'market': MarketUncertainty(
                game_id='CHI_at_GB',
                spread_movement=4.0,
                total_movement=5.5,
                spread_dispersion=2.0,
                total_dispersion=2.5,
                sharp_public_divergence=0.6,
                handle_ratio=0.6,
                sharp_action_unclear=True
            )
        }
    ]

    # Create segmenter
    segmenter = MatchupSegmenter()

    # Segment each game
    print("=" * 80)
    print("GAME UNCERTAINTY ANALYSIS")
    print("=" * 80)

    for game_data in games:
        segment_name, uncertainty_score, details = segmenter.segment_game(
            game_data['game_id'],
            game_data['model'],
            game_data['context'],
            game_data['market']
        )

        print(f"\n{game_data['game_id']}:")
        print(f"  Segment: {details['segment'].name}")
        print(f"  Uncertainty Score: {uncertainty_score:.3f}")
        print(f"    - Epistemic: {details['epistemic_uncertainty']:.3f}")
        print(f"    - Aleatoric: {details['aleatoric_uncertainty']:.3f}")
        print(f"    - Market: {details['market_uncertainty']:.3f}")

        # Show betting adjustments
        base_edge = 0.04  # 4% edge
        base_kelly = 0.25  # 25% Kelly

        should_bet, adjusted_kelly, reason = segmenter.adjust_bet_strategy(
            base_edge, base_kelly, segment_name
        )

        print(f"\n  Betting Strategy (4% edge, 25% Kelly base):")
        print(f"    Should bet: {should_bet}")
        print(f"    Adjusted Kelly: {adjusted_kelly:.1%}")
        print(f"    Reason: {reason}")

    # Generate report
    print("\n")
    report = segmenter.generate_segmentation_report()
    print(report)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    demo_uncertainty_segmentation()