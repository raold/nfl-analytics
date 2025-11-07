"""
Pinnacle Calibration Module

Adjusts model predictions and bet sizing for Pinnacle's lower vig environment.
Key differences from standard sportsbooks:
- Lower hold: 4.5-5.5% (vs 5.7-6.5% square books)
- Sharper lines: Pinnacle lines are closer to "true" probability
- Lower edge threshold: Can profitably bet smaller edges
- More aggressive Kelly: Lower vig allows higher Kelly fractions

Author: Claude Code
Date: November 6, 2025
Version: 1.0

Usage:
    from py.models.pinnacle_calibration import PinnacleCalibrator

    calibrator = PinnacleCalibrator(
        standard_vig=0.055,  # DraftKings/FanDuel typical
        pinnacle_vig=0.045   # Pinnacle typical
    )

    # Adjust edge threshold for Pinnacle
    min_edge_standard = 0.03  # 3% for square books
    min_edge_pinnacle = calibrator.adjust_edge_threshold(min_edge_standard)
    # Returns: 0.015 (1.5% for Pinnacle)

    # Adjust Kelly fraction for Pinnacle
    kelly_standard = 0.25  # 25% Kelly for square books
    kelly_pinnacle = calibrator.adjust_kelly_fraction(kelly_standard)
    # Returns: 0.35 (35% Kelly for Pinnacle)
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PinnacleCalibrator:
    """
    Calibrate betting parameters for Pinnacle's lower vig environment.

    The key insight: With lower vig, the breakeven point drops, allowing
    profitable betting on smaller edges with more aggressive sizing.

    Mathematical Basis:
    - Breakeven win rate = vig / (1 + vig)
    - Standard books (5.5% vig): 52.4% breakeven
    - Pinnacle (4.5% vig): 51.8% breakeven
    - Difference: 0.6 percentage points = more opportunities
    """

    def __init__(
        self,
        standard_vig: float = 0.055,
        pinnacle_vig: float = 0.045,
        risk_tolerance: float = 1.0
    ):
        """
        Initialize Pinnacle calibrator.

        Args:
            standard_vig: Typical vig for square books (e.g., 0.055 = 5.5%)
            pinnacle_vig: Typical vig for Pinnacle (e.g., 0.045 = 4.5%)
            risk_tolerance: Risk multiplier (1.0 = normal, <1.0 = conservative)
        """
        self.standard_vig = standard_vig
        self.pinnacle_vig = pinnacle_vig
        self.risk_tolerance = risk_tolerance

        # Calculate vig savings
        self.vig_savings = standard_vig - pinnacle_vig

        logger.info(
            f"Pinnacle Calibrator initialized: "
            f"{standard_vig:.1%} → {pinnacle_vig:.1%} "
            f"(savings: {self.vig_savings:.1%})"
        )

    def adjust_edge_threshold(
        self,
        standard_min_edge: float,
        safety_margin: float = 0.005
    ) -> float:
        """
        Adjust minimum edge threshold for Pinnacle.

        With lower vig, we can profitably bet smaller edges. The adjustment
        is approximately equal to the vig savings, minus a safety margin.

        Args:
            standard_min_edge: Minimum edge for standard books (e.g., 0.03)
            safety_margin: Conservative buffer (e.g., 0.005 = 0.5%)

        Returns:
            Adjusted minimum edge for Pinnacle

        Example:
            Standard: Need 3% edge to overcome 5.5% vig
            Pinnacle: Need 1.5% edge to overcome 4.5% vig
            Adjustment: 3% - (5.5% - 4.5%) - 0.5% = 1.5%
        """
        adjusted_edge = standard_min_edge - self.vig_savings + safety_margin

        # Floor at 1% edge (never bet <1% edge even at Pinnacle)
        adjusted_edge = max(adjusted_edge, 0.01)

        logger.info(
            f"Edge threshold adjusted: "
            f"{standard_min_edge:.1%} → {adjusted_edge:.1%}"
        )

        return adjusted_edge

    def adjust_kelly_fraction(
        self,
        standard_kelly: float,
        aggression_factor: float = 1.4
    ) -> float:
        """
        Adjust Kelly fraction for Pinnacle's lower variance environment.

        With lower vig, the same edge has higher expected value, justifying
        more aggressive position sizing. We scale up Kelly fraction while
        respecting risk tolerance.

        Args:
            standard_kelly: Kelly fraction for standard books (e.g., 0.25)
            aggression_factor: How much to increase Kelly (e.g., 1.4 = 40% more)

        Returns:
            Adjusted Kelly fraction for Pinnacle

        Example:
            Standard: 25% Kelly (conservative)
            Pinnacle: 35% Kelly (40% more aggressive)
        """
        adjusted_kelly = standard_kelly * aggression_factor * self.risk_tolerance

        # Cap at 50% Kelly (never bet >50% of bankroll on single bet)
        adjusted_kelly = min(adjusted_kelly, 0.50)

        logger.info(
            f"Kelly fraction adjusted: "
            f"{standard_kelly:.1%} → {adjusted_kelly:.1%}"
        )

        return adjusted_kelly

    def estimate_roi_improvement(
        self,
        current_roi: float,
        current_vig: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Estimate ROI improvement from switching to Pinnacle.

        Simple model: ROI improvement ≈ vig savings
        More sophisticated: Account for increased bet volume at lower edge threshold

        Args:
            current_roi: Current ROI at standard books (e.g., -0.075 = -7.5%)
            current_vig: Current vig (defaults to standard_vig)

        Returns:
            Dict with estimated metrics

        Example:
            Current: -7.5% ROI at 5.5% vig
            Pinnacle: +1.5% ROI at 4.5% vig (9% improvement)
        """
        if current_vig is None:
            current_vig = self.standard_vig

        # Conservative estimate: ROI improves by vig savings
        conservative_roi = current_roi + self.vig_savings

        # Optimistic estimate: Also benefit from more betting opportunities
        # Lower edge threshold → ~30% more bets → ~30% more profit
        opportunity_multiplier = 1.3
        optimistic_roi = current_roi + (self.vig_savings * opportunity_multiplier)

        # Realistic estimate: Split the difference
        realistic_roi = (conservative_roi + optimistic_roi) / 2

        return {
            'current_roi': current_roi,
            'conservative_roi': conservative_roi,
            'realistic_roi': realistic_roi,
            'optimistic_roi': optimistic_roi,
            'vig_savings': self.vig_savings,
            'expected_improvement': realistic_roi - current_roi
        }

    def calibrate_bet_opportunities(
        self,
        predictions_df: pd.DataFrame,
        prop_lines_df: pd.DataFrame,
        standard_min_edge: float = 0.03,
        standard_kelly: float = 0.25
    ) -> pd.DataFrame:
        """
        Identify betting opportunities calibrated for Pinnacle.

        Args:
            predictions_df: Model predictions with columns: player_id, prop_type, prediction, std
            prop_lines_df: Prop lines with columns: player_id, prop_type, line_value, over_odds, under_odds, bookmaker_key
            standard_min_edge: Minimum edge for standard books
            standard_kelly: Kelly fraction for standard books

        Returns:
            DataFrame of betting opportunities with Pinnacle-calibrated sizing
        """
        # Adjust thresholds for Pinnacle
        pinnacle_min_edge = self.adjust_edge_threshold(standard_min_edge)
        pinnacle_kelly = self.adjust_kelly_fraction(standard_kelly)

        # Filter to Pinnacle lines only
        pinnacle_lines = prop_lines_df[
            prop_lines_df['bookmaker_key'] == 'pinnacle'
        ].copy()

        if pinnacle_lines.empty:
            logger.warning("No Pinnacle lines found in prop_lines_df")
            return pd.DataFrame()

        # Merge predictions with lines
        opportunities = pinnacle_lines.merge(
            predictions_df,
            on=['player_id', 'prop_type'],
            how='inner'
        )

        if opportunities.empty:
            logger.warning("No matching predictions for Pinnacle lines")
            return pd.DataFrame()

        # Calculate edge for OVER bets
        opportunities['over_implied_prob'] = self._odds_to_prob(opportunities['over_odds'])
        opportunities['model_prob_over'] = opportunities.apply(
            lambda row: 1 - self._norm_cdf(row['line_value'], row['prediction'], row['std']),
            axis=1
        )
        opportunities['edge_over'] = opportunities['model_prob_over'] - opportunities['over_implied_prob']

        # Calculate edge for UNDER bets
        opportunities['under_implied_prob'] = self._odds_to_prob(opportunities['under_odds'])
        opportunities['model_prob_under'] = opportunities.apply(
            lambda row: self._norm_cdf(row['line_value'], row['prediction'], row['std']),
            axis=1
        )
        opportunities['edge_under'] = opportunities['model_prob_under'] - opportunities['under_implied_prob']

        # Filter to opportunities exceeding Pinnacle threshold
        over_bets = opportunities[opportunities['edge_over'] >= pinnacle_min_edge].copy()
        over_bets['bet_side'] = 'over'
        over_bets['edge'] = over_bets['edge_over']
        over_bets['odds'] = over_bets['over_odds']

        under_bets = opportunities[opportunities['edge_under'] >= pinnacle_min_edge].copy()
        under_bets['bet_side'] = 'under'
        under_bets['edge'] = under_bets['edge_under']
        under_bets['odds'] = under_bets['under_odds']

        # Combine
        all_bets = pd.concat([over_bets, under_bets], ignore_index=True)

        # Calculate Kelly bet size
        all_bets['kelly_fraction'] = pinnacle_kelly
        all_bets['bet_amount_pct'] = all_bets['edge'] * pinnacle_kelly

        logger.info(
            f"Found {len(all_bets)} Pinnacle opportunities "
            f"(min edge: {pinnacle_min_edge:.1%})"
        )

        return all_bets

    @staticmethod
    def _odds_to_prob(odds: pd.Series) -> pd.Series:
        """Convert American odds to implied probability."""
        prob = np.where(
            odds > 0,
            100.0 / (odds + 100.0),
            -odds / (-odds + 100.0)
        )
        return prob

    @staticmethod
    def _norm_cdf(x: float, mean: float, std: float) -> float:
        """Normal CDF (cumulative distribution function)."""
        from scipy.stats import norm
        return norm.cdf(x, loc=mean, scale=std)

    def get_calibration_summary(self) -> Dict:
        """Get summary of calibration adjustments."""
        return {
            'standard_vig': self.standard_vig,
            'pinnacle_vig': self.pinnacle_vig,
            'vig_savings': self.vig_savings,
            'risk_tolerance': self.risk_tolerance,
            'edge_threshold_example': {
                'standard_3pct': self.adjust_edge_threshold(0.03),
                'standard_2pct': self.adjust_edge_threshold(0.02)
            },
            'kelly_fraction_example': {
                'standard_25pct': self.adjust_kelly_fraction(0.25),
                'standard_15pct': self.adjust_kelly_fraction(0.15)
            }
        }


# ============================================================================
# CLI for testing
# ============================================================================

def main():
    """Test the Pinnacle calibrator."""
    import argparse

    parser = argparse.ArgumentParser(description='Pinnacle Calibration Tool')
    parser.add_argument('--standard-vig', type=float, default=0.055, help='Standard book vig (default: 5.5%)')
    parser.add_argument('--pinnacle-vig', type=float, default=0.045, help='Pinnacle vig (default: 4.5%)')
    parser.add_argument('--current-roi', type=float, default=-0.075, help='Current ROI (default: -7.5%)')

    args = parser.parse_args()

    # Initialize calibrator
    calibrator = PinnacleCalibrator(
        standard_vig=args.standard_vig,
        pinnacle_vig=args.pinnacle_vig
    )

    # Show calibration summary
    print("\n" + "="*80)
    print("PINNACLE CALIBRATION SUMMARY")
    print("="*80)

    summary = calibrator.get_calibration_summary()
    print(f"\nVig Comparison:")
    print(f"  Standard books: {summary['standard_vig']:.1%}")
    print(f"  Pinnacle:       {summary['pinnacle_vig']:.1%}")
    print(f"  Savings:        {summary['vig_savings']:.1%}")

    print(f"\nEdge Threshold Adjustments:")
    for standard, pinnacle in summary['edge_threshold_example'].items():
        print(f"  {standard}: {pinnacle:.1%}")

    print(f"\nKelly Fraction Adjustments:")
    for standard, pinnacle in summary['kelly_fraction_example'].items():
        print(f"  {standard}: {pinnacle:.1%}")

    # Estimate ROI improvement
    roi_est = calibrator.estimate_roi_improvement(args.current_roi)
    print(f"\nROI Improvement Estimate:")
    print(f"  Current ROI:      {roi_est['current_roi']:+.1%}")
    print(f"  Conservative:     {roi_est['conservative_roi']:+.1%}")
    print(f"  Realistic:        {roi_est['realistic_roi']:+.1%} ⭐")
    print(f"  Optimistic:       {roi_est['optimistic_roi']:+.1%}")
    print(f"  Expected gain:    {roi_est['expected_improvement']:+.1%}")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
