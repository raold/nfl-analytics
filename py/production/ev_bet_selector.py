#!/usr/bin/env python3
"""
EV-Optimal Bet Selector for NFL Betting.

Implements edge-based bet selection to maximize ROI rather than accuracy.
Filters bets based on expected value, CLV history, and market conditions.

Key improvements over probability-threshold approach:
- Only bets when edge exceeds minimum threshold
- Implements "no-bet band" around breakeven
- Filters by historical CLV conversion rates
- Considers market microstructure (hold, line movement)

Usage:
    python py/production/ev_bet_selector.py \
        --predictions predictions.csv \
        --min-edge 0.02 \
        --deadzone 0.002 \
        --output selected_bets.csv
"""

import argparse
import sys
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from production.kelly_sizing import (
    american_to_decimal_odds,
    american_odds_to_implied_prob,
    kelly_criterion
)


# ============================================================================
# Edge Calculation
# ============================================================================

def calculate_edge(
    model_prob: float,
    market_odds: int,
    hold_adjustment: float = 0.0
) -> float:
    """
    Calculate betting edge with hold adjustment.

    Args:
        model_prob: Model's win probability
        market_odds: American odds from sportsbook
        hold_adjustment: Adjustment for book hold (e.g., 0.01 for 1% hold reduction)

    Returns:
        Edge (positive = +EV, negative = -EV)
    """
    implied_prob = american_odds_to_implied_prob(market_odds)
    # Adjust for hold (reduce implied probability)
    implied_prob_adj = implied_prob * (1 - hold_adjustment)
    return model_prob - implied_prob_adj


def calculate_ev(
    model_prob: float,
    market_odds: int,
    bet_amount: float = 100.0
) -> float:
    """
    Calculate expected value of a bet.

    Args:
        model_prob: Model's win probability
        market_odds: American odds
        bet_amount: Bet size ($)

    Returns:
        Expected value ($)
    """
    decimal_odds = american_to_decimal_odds(market_odds)
    win_amount = bet_amount * decimal_odds
    ev = (model_prob * win_amount) - ((1 - model_prob) * bet_amount)
    return ev


# ============================================================================
# CLV Analysis
# ============================================================================

def load_clv_history(clv_file: str) -> pd.DataFrame:
    """Load historical CLV data."""
    try:
        df = pd.read_csv(clv_file)
        # Expected columns: game_id, model_prob, closing_prob, clv, roi
        return df
    except Exception as e:
        print(f"Warning: Could not load CLV history: {e}")
        return pd.DataFrame()


def get_clv_percentile(model_prob: float, market_prob: float, clv_history: pd.DataFrame) -> float:
    """
    Get CLV percentile based on historical data.

    Args:
        model_prob: Model's probability
        market_prob: Market implied probability
        clv_history: Historical CLV data

    Returns:
        CLV percentile (0-100)
    """
    if clv_history.empty:
        return 50.0  # Default to median if no history

    clv = model_prob - market_prob
    percentile = (clv_history['clv'] < clv).mean() * 100
    return percentile


def get_clv_roi_by_decile(clv_history: pd.DataFrame) -> Dict[int, float]:
    """
    Calculate ROI by CLV decile.

    Returns:
        Dict mapping decile (1-10) to average ROI
    """
    if clv_history.empty:
        return {i: 0.0 for i in range(1, 11)}

    clv_history['clv_decile'] = pd.qcut(clv_history['clv'], 10, labels=False) + 1
    roi_by_decile = clv_history.groupby('clv_decile')['roi'].mean().to_dict()

    # Fill missing deciles
    for i in range(1, 11):
        if i not in roi_by_decile:
            roi_by_decile[i] = 0.0

    return roi_by_decile


# ============================================================================
# Market Microstructure
# ============================================================================

def calculate_line_velocity(
    line_history: List[Tuple[datetime, float]]
) -> float:
    """
    Calculate line movement velocity (points per hour).

    Args:
        line_history: List of (timestamp, line) tuples

    Returns:
        Line velocity (positive = moving toward home)
    """
    if len(line_history) < 2:
        return 0.0

    # Sort by timestamp
    line_history = sorted(line_history, key=lambda x: x[0])

    # Calculate weighted velocity (recent moves weighted more)
    velocities = []
    weights = []

    for i in range(1, len(line_history)):
        dt = (line_history[i][0] - line_history[i-1][0]).total_seconds() / 3600  # hours
        if dt > 0:
            dline = line_history[i][1] - line_history[i-1][1]
            velocity = dline / dt
            velocities.append(velocity)
            # Exponential weighting - more recent = higher weight
            weight = np.exp(-0.1 * (len(line_history) - i))
            weights.append(weight)

    if velocities:
        weighted_velocity = np.average(velocities, weights=weights)
        return weighted_velocity
    return 0.0


def calculate_cross_book_consensus(
    book_odds: Dict[str, int]
) -> Tuple[float, float]:
    """
    Calculate cross-book consensus and disagreement.

    Args:
        book_odds: Dict mapping book name to American odds

    Returns:
        (consensus_prob, std_dev)
    """
    if not book_odds:
        return 0.5, 0.0

    probs = [american_odds_to_implied_prob(odds) for odds in book_odds.values()]
    consensus = np.mean(probs)
    disagreement = np.std(probs)

    return consensus, disagreement


def find_reduced_juice_windows(
    book: str,
    current_time: datetime
) -> bool:
    """
    Check if current time is in reduced juice window.

    Common reduced juice windows:
    - DraftKings: Wed 3-4pm ET
    - FanDuel: Thu 2-3pm ET
    - BetMGM: Fri 12-1pm ET
    """
    reduced_juice_windows = {
        'DraftKings': [(2, 15, 16)],  # Wed 3-4pm
        'FanDuel': [(3, 14, 15)],      # Thu 2-3pm
        'BetMGM': [(4, 12, 13)],       # Fri 12-1pm
    }

    if book not in reduced_juice_windows:
        return False

    windows = reduced_juice_windows[book]
    for day, start_hour, end_hour in windows:
        if (current_time.weekday() == day and
            start_hour <= current_time.hour < end_hour):
            return True

    return False


# ============================================================================
# Bet Selection Logic
# ============================================================================

class EVBetSelector:
    """EV-optimal bet selector with multiple filters."""

    def __init__(
        self,
        min_edge: float = 0.02,          # 2% minimum edge
        deadzone: float = 0.002,          # 0.2% no-bet band
        min_clv_percentile: int = 70,    # Top 30% CLV only
        max_book_hold: float = 0.05,     # Max 5% hold
        min_ev_dollars: float = 2.0,     # Min $2 EV per $100 bet
        kelly_fraction: float = 0.25,    # Quarter Kelly
        max_bet_pct: float = 0.05,       # Max 5% of bankroll
    ):
        self.min_edge = min_edge
        self.deadzone = deadzone
        self.min_clv_percentile = min_clv_percentile
        self.max_book_hold = max_book_hold
        self.min_ev_dollars = min_ev_dollars
        self.kelly_fraction = kelly_fraction
        self.max_bet_pct = max_bet_pct

        self.clv_history = pd.DataFrame()
        self.clv_roi_by_decile = {}

    def load_clv_history(self, clv_file: str):
        """Load historical CLV data."""
        self.clv_history = load_clv_history(clv_file)
        self.clv_roi_by_decile = get_clv_roi_by_decile(self.clv_history)

    def should_bet(
        self,
        model_prob: float,
        market_odds: int,
        opposite_odds: int = None,
        book_name: str = None,
        line_velocity: float = 0.0,
        time_to_kickoff: timedelta = None,
        bankroll: float = 10000.0,
    ) -> Dict:
        """
        Determine if a bet should be placed.

        Args:
            model_prob: Model's win probability
            market_odds: American odds for the bet
            opposite_odds: American odds for opposite side (to calculate hold)
            book_name: Sportsbook name
            line_velocity: Line movement velocity
            time_to_kickoff: Time until game starts
            bankroll: Current bankroll

        Returns:
            Dict with bet decision and details
        """
        result = {
            'should_bet': False,
            'reason': '',
            'edge': 0.0,
            'ev': 0.0,
            'kelly_size': 0.0,
            'bet_amount': 0.0,
            'confidence': 0.0,
        }

        # Calculate hold if opposite odds provided
        hold = 0.0
        if opposite_odds:
            prob1 = american_odds_to_implied_prob(market_odds)
            prob2 = american_odds_to_implied_prob(opposite_odds)
            hold = (prob1 + prob2) - 1.0

            if hold > self.max_book_hold:
                result['reason'] = f'Hold too high: {hold:.3f}'
                return result

        # Calculate edge with hold adjustment
        hold_adjustment = hold / 2 if hold > 0 else 0
        edge = calculate_edge(model_prob, market_odds, hold_adjustment)
        result['edge'] = edge

        # Check minimum edge (with deadzone)
        if edge < self.min_edge:
            result['reason'] = f'Insufficient edge: {edge:.3f}'
            return result

        # Check deadzone (no-bet band near breakeven)
        if abs(edge) < self.deadzone:
            result['reason'] = f'In deadzone: {edge:.3f}'
            return result

        # Calculate EV
        ev_per_100 = calculate_ev(model_prob, market_odds, 100.0)
        result['ev'] = ev_per_100

        if ev_per_100 < self.min_ev_dollars:
            result['reason'] = f'Insufficient EV: ${ev_per_100:.2f}'
            return result

        # Check CLV percentile if history available
        if not self.clv_history.empty:
            market_prob = american_odds_to_implied_prob(market_odds)
            clv_percentile = get_clv_percentile(model_prob, market_prob, self.clv_history)

            if clv_percentile < self.min_clv_percentile:
                result['reason'] = f'Low CLV percentile: {clv_percentile:.0f}'
                return result

            # Get expected ROI from CLV decile
            clv_decile = int(clv_percentile / 10) + 1
            expected_roi = self.clv_roi_by_decile.get(clv_decile, 0.0)

            if expected_roi < 0:
                result['reason'] = f'Negative expected ROI from CLV: {expected_roi:.2f}%'
                return result

        # Line velocity check (bet against steam)
        if line_velocity != 0:
            # Positive velocity = line moving toward home
            # If we're betting home and line moving away, that's bad
            if (model_prob > 0.5 and line_velocity < -0.5) or \
               (model_prob < 0.5 and line_velocity > 0.5):
                result['reason'] = f'Adverse line movement: {line_velocity:.2f}'
                return result

        # Timing considerations
        if time_to_kickoff:
            hours_to_kickoff = time_to_kickoff.total_seconds() / 3600

            # Early week: only bet if expecting favorable line movement
            if hours_to_kickoff > 72:  # More than 3 days
                if abs(line_velocity) < 0.1:  # No strong movement expected
                    result['reason'] = 'Too early, no line velocity'
                    return result

            # Very close to kickoff: require higher edge
            if hours_to_kickoff < 2:
                if edge < self.min_edge * 1.5:
                    result['reason'] = f'Close to kickoff, need higher edge'
                    return result

        # Check for reduced juice windows
        if book_name and find_reduced_juice_windows(book_name, datetime.now()):
            # Boost edge for reduced juice
            edge *= 1.1
            result['edge'] = edge

        # Calculate Kelly size
        decimal_odds = american_to_decimal_odds(market_odds)
        kelly_size = kelly_criterion(
            win_prob=model_prob,
            decimal_odds=decimal_odds,
            kelly_fraction=self.kelly_fraction,
            max_bet_fraction=self.max_bet_pct
        )

        result['kelly_size'] = kelly_size
        result['bet_amount'] = kelly_size * bankroll

        # Calculate confidence score (0-100)
        confidence = 50.0
        confidence += edge * 500  # +5 points per 1% edge
        confidence += min(20, ev_per_100 * 2)  # Up to +20 for EV
        if not self.clv_history.empty:
            confidence += (clv_percentile - 50) / 2  # +0-25 for CLV
        confidence = min(100, max(0, confidence))

        result['confidence'] = confidence
        result['should_bet'] = True
        result['reason'] = 'All checks passed'

        return result

    def select_bets(
        self,
        predictions_df: pd.DataFrame,
        odds_df: pd.DataFrame = None,
        bankroll: float = 10000.0,
        max_bets: int = None,
    ) -> pd.DataFrame:
        """
        Select bets from a DataFrame of predictions.

        Args:
            predictions_df: DataFrame with columns: game_id, model_prob, team
            odds_df: Optional DataFrame with odds data
            bankroll: Current bankroll
            max_bets: Maximum number of bets to select

        Returns:
            DataFrame of selected bets
        """
        selected = []

        for _, row in predictions_df.iterrows():
            # Get odds (from odds_df or predictions_df)
            if odds_df is not None:
                odds_row = odds_df[odds_df['game_id'] == row['game_id']]
                if odds_row.empty:
                    continue
                market_odds = odds_row.iloc[0]['odds']
                opposite_odds = odds_row.iloc[0].get('opposite_odds', None)
                book_name = odds_row.iloc[0].get('book', None)
            else:
                market_odds = row.get('odds', -110)
                opposite_odds = row.get('opposite_odds', None)
                book_name = row.get('book', None)

            # Calculate line velocity if history available
            line_velocity = 0.0
            if 'line_history' in row:
                line_velocity = calculate_line_velocity(row['line_history'])

            # Time to kickoff
            time_to_kickoff = None
            if 'kickoff' in row:
                time_to_kickoff = row['kickoff'] - datetime.now()

            # Check if should bet
            decision = self.should_bet(
                model_prob=row['model_prob'],
                market_odds=market_odds,
                opposite_odds=opposite_odds,
                book_name=book_name,
                line_velocity=line_velocity,
                time_to_kickoff=time_to_kickoff,
                bankroll=bankroll,
            )

            if decision['should_bet']:
                bet_row = {
                    'game_id': row['game_id'],
                    'team': row.get('team', ''),
                    'model_prob': row['model_prob'],
                    'market_odds': market_odds,
                    'edge': decision['edge'],
                    'ev': decision['ev'],
                    'kelly_size': decision['kelly_size'],
                    'bet_amount': decision['bet_amount'],
                    'confidence': decision['confidence'],
                    'book': book_name or 'unknown',
                }
                selected.append(bet_row)

        if not selected:
            return pd.DataFrame()

        selected_df = pd.DataFrame(selected)

        # Sort by EV and limit number of bets
        selected_df = selected_df.sort_values('ev', ascending=False)

        if max_bets:
            selected_df = selected_df.head(max_bets)

        return selected_df


# ============================================================================
# Backtesting
# ============================================================================

def backtest_selector(
    predictions_file: str,
    results_file: str,
    min_edge: float = 0.02,
    deadzone: float = 0.002,
    output_file: str = None,
) -> Dict:
    """
    Backtest the EV bet selector on historical data.

    Args:
        predictions_file: CSV with predictions
        results_file: CSV with actual results
        min_edge: Minimum edge threshold
        deadzone: No-bet band width
        output_file: Optional output CSV

    Returns:
        Dict with backtest metrics
    """
    # Load data
    predictions = pd.read_csv(predictions_file)
    results = pd.read_csv(results_file)

    # Initialize selector
    selector = EVBetSelector(min_edge=min_edge, deadzone=deadzone)

    # Select bets
    selected = selector.select_bets(predictions)

    if selected.empty:
        return {
            'n_bets': 0,
            'roi': 0.0,
            'win_rate': 0.0,
            'avg_edge': 0.0,
            'avg_ev': 0.0,
        }

    # Merge with results
    selected = selected.merge(results[['game_id', 'won']], on='game_id')

    # Calculate metrics
    total_wagered = selected['bet_amount'].sum()
    total_return = (selected['bet_amount'] * selected['won'] *
                   selected['market_odds'].apply(american_to_decimal_odds)).sum()
    total_profit = total_return - total_wagered

    metrics = {
        'n_bets': len(selected),
        'roi': (total_profit / total_wagered * 100) if total_wagered > 0 else 0,
        'win_rate': selected['won'].mean() * 100,
        'avg_edge': selected['edge'].mean() * 100,
        'avg_ev': selected['ev'].mean(),
        'total_wagered': total_wagered,
        'total_profit': total_profit,
    }

    if output_file:
        selected.to_csv(output_file, index=False)
        print(f"Selected bets saved to {output_file}")

    return metrics


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='EV-Optimal Bet Selector for NFL Betting'
    )

    # Input files
    parser.add_argument('--predictions', required=True, help='Predictions CSV file')
    parser.add_argument('--odds', help='Optional odds CSV file')
    parser.add_argument('--clv-history', help='Historical CLV data')

    # Selection parameters
    parser.add_argument('--min-edge', type=float, default=0.02,
                       help='Minimum edge to bet (default: 0.02)')
    parser.add_argument('--deadzone', type=float, default=0.002,
                       help='No-bet band around breakeven (default: 0.002)')
    parser.add_argument('--min-clv-percentile', type=int, default=70,
                       help='Minimum CLV percentile (default: 70)')
    parser.add_argument('--max-hold', type=float, default=0.05,
                       help='Maximum book hold (default: 0.05)')
    parser.add_argument('--min-ev', type=float, default=2.0,
                       help='Minimum EV per $100 (default: 2.0)')

    # Kelly sizing
    parser.add_argument('--kelly-fraction', type=float, default=0.25,
                       help='Kelly fraction (default: 0.25)')
    parser.add_argument('--max-bet-pct', type=float, default=0.05,
                       help='Max bet as pct of bankroll (default: 0.05)')
    parser.add_argument('--bankroll', type=float, default=10000.0,
                       help='Current bankroll (default: 10000)')

    # Output
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--max-bets', type=int, help='Maximum number of bets')

    # Backtesting
    parser.add_argument('--backtest', action='store_true',
                       help='Run backtest mode')
    parser.add_argument('--results', help='Results file for backtesting')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("EV-Optimal Bet Selector")
    print("=" * 80)

    if args.backtest:
        if not args.results:
            print("ERROR: --results required for backtesting")
            return 1

        print("\nBacktesting Parameters:")
        print(f"  Predictions: {args.predictions}")
        print(f"  Results: {args.results}")
        print(f"  Min edge: {args.min_edge:.3f}")
        print(f"  Deadzone: {args.deadzone:.3f}")

        metrics = backtest_selector(
            predictions_file=args.predictions,
            results_file=args.results,
            min_edge=args.min_edge,
            deadzone=args.deadzone,
            output_file=args.output,
        )

        print("\nBacktest Results:")
        print(f"  Bets placed: {metrics['n_bets']}")
        print(f"  ROI: {metrics['roi']:+.2f}%")
        print(f"  Win rate: {metrics['win_rate']:.1f}%")
        print(f"  Avg edge: {metrics['avg_edge']:.2f}%")
        print(f"  Avg EV: ${metrics['avg_ev']:.2f}")
        print(f"  Total wagered: ${metrics['total_wagered']:,.0f}")
        print(f"  Total profit: ${metrics['total_profit']:+,.0f}")

    else:
        # Load predictions
        predictions_df = pd.read_csv(args.predictions)
        print(f"\nLoaded {len(predictions_df)} predictions")

        # Load odds if provided
        odds_df = None
        if args.odds:
            odds_df = pd.read_csv(args.odds)
            print(f"Loaded odds for {len(odds_df)} games")

        # Initialize selector
        selector = EVBetSelector(
            min_edge=args.min_edge,
            deadzone=args.deadzone,
            min_clv_percentile=args.min_clv_percentile,
            max_book_hold=args.max_hold,
            min_ev_dollars=args.min_ev,
            kelly_fraction=args.kelly_fraction,
            max_bet_pct=args.max_bet_pct,
        )

        # Load CLV history if provided
        if args.clv_history:
            selector.load_clv_history(args.clv_history)
            print(f"Loaded CLV history with {len(selector.clv_history)} games")

        # Select bets
        selected = selector.select_bets(
            predictions_df=predictions_df,
            odds_df=odds_df,
            bankroll=args.bankroll,
            max_bets=args.max_bets,
        )

        if selected.empty:
            print("\nNo bets meet selection criteria")
            return 0

        # Save results
        selected.to_csv(args.output, index=False)

        print(f"\nSelected {len(selected)} bets:")
        print(f"  Total to wager: ${selected['bet_amount'].sum():,.0f}")
        print(f"  Average edge: {selected['edge'].mean()*100:.2f}%")
        print(f"  Average EV: ${selected['ev'].mean():.2f}")
        print(f"  Average confidence: {selected['confidence'].mean():.1f}")

        print(f"\nBets saved to {args.output}")

        # Show top bets
        print("\nTop 5 bets by EV:")
        print(selected.nlargest(5, 'ev')[['game_id', 'team', 'edge', 'ev', 'bet_amount', 'confidence']])

    return 0


if __name__ == '__main__':
    sys.exit(main())