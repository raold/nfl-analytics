#!/usr/bin/env python3
"""
EV-Optimal Bet Selector for NFL Player Props.

Extends the game-level EV bet selector specifically for player props betting.
Key differences from game-level betting:
- Higher book holds (7-10% vs 4-5%)
- Lower betting limits
- More line movement volatility
- Need to check injury status
- Correlation between props (don't overbet correlated props)

Usage:
    python py/production/props_ev_selector.py \
        --predictions props_predictions.csv \
        --prop-lines prop_lines.csv \
        --min-edge 0.03 \
        --output selected_props.csv
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
# Props-Specific Edge Calculation
# ============================================================================

def calculate_prop_edge(
    model_prediction: float,
    model_std: float,
    line_value: float,
    over_odds: int = None,
    under_odds: int = None,
    hold_adjustment: float = 0.0
) -> Dict:
    """
    Calculate edge for a prop bet (over or under).

    Args:
        model_prediction: Model's predicted value (e.g., 255.3 passing yards)
        model_std: Model's uncertainty (standard deviation)
        line_value: Prop line (e.g., 250.5 yards)
        over_odds: American odds for over
        under_odds: American odds for under
        hold_adjustment: Adjustment for book hold

    Returns:
        Dict with edge for both over and under
    """
    from scipy.stats import norm

    # Calculate probability of over/under using normal distribution
    if model_std > 0:
        # P(X > line_value) = 1 - CDF(line_value)
        prob_over = 1 - norm.cdf(line_value, loc=model_prediction, scale=model_std)
        prob_under = norm.cdf(line_value, loc=model_prediction, scale=model_std)
    else:
        # No uncertainty - deterministic
        prob_over = 1.0 if model_prediction > line_value else 0.0
        prob_under = 1.0 if model_prediction < line_value else 0.0

    result = {
        'prob_over': prob_over,
        'prob_under': prob_under,
        'edge_over': 0.0,
        'edge_under': 0.0,
        'best_bet': None
    }

    # Calculate edge for over
    if over_odds:
        implied_prob_over = american_odds_to_implied_prob(over_odds)
        implied_prob_over_adj = implied_prob_over * (1 - hold_adjustment)
        result['edge_over'] = prob_over - implied_prob_over_adj

    # Calculate edge for under
    if under_odds:
        implied_prob_under = american_odds_to_implied_prob(under_odds)
        implied_prob_under_adj = implied_prob_under * (1 - hold_adjustment)
        result['edge_under'] = prob_under - implied_prob_under_adj

    # Determine best bet
    if result['edge_over'] > result['edge_under']:
        result['best_bet'] = 'over'
        result['best_edge'] = result['edge_over']
        result['best_odds'] = over_odds
        result['best_prob'] = prob_over
    else:
        result['best_bet'] = 'under'
        result['best_edge'] = result['edge_under']
        result['best_odds'] = under_odds
        result['best_prob'] = prob_under

    return result


def calculate_prop_ev(
    model_prob: float,
    odds: int,
    bet_amount: float = 100.0
) -> float:
    """
    Calculate expected value of a prop bet.

    Args:
        model_prob: Model's win probability
        odds: American odds
        bet_amount: Bet size ($)

    Returns:
        Expected value ($)
    """
    decimal_odds = american_to_decimal_odds(odds)
    win_amount = bet_amount * decimal_odds
    ev = (model_prob * win_amount) - ((1 - model_prob) * bet_amount)
    return ev


# ============================================================================
# Injury Checking
# ============================================================================

def check_injury_status(player_id: str, game_date: datetime, db_conn) -> Tuple[bool, str]:
    """
    Check if player is injured/questionable.

    Args:
        player_id: GSIS player ID
        game_date: Game date
        db_conn: Database connection

    Returns:
        (is_healthy, injury_status)
    """
    try:
        query = """
            SELECT report_status
            FROM injuries
            WHERE gsis_id = %s
                AND season = EXTRACT(YEAR FROM %s::DATE)
                AND week = (
                    SELECT week FROM games
                    WHERE game_id = (
                        SELECT game_id FROM games
                        WHERE kickoff::DATE = %s::DATE
                        LIMIT 1
                    )
                )
            ORDER BY week DESC
            LIMIT 1
        """

        cur = db_conn.cursor()
        cur.execute(query, (player_id, game_date, game_date))
        result = cur.fetchone()
        cur.close()

        if result:
            status = result[0]
            # Out, Doubtful, Questionable = red flags
            if status in ['Out', 'Doubtful']:
                return False, status
            elif status == 'Questionable':
                return True, status  # Can bet but with caution
            else:
                return True, status
        else:
            # No injury report = healthy
            return True, 'Healthy'

    except Exception as e:
        print(f"Error checking injury status: {e}")
        return True, 'Unknown'


# ============================================================================
# Correlation Matrix
# ============================================================================

def calculate_prop_correlation(
    player_id: str,
    prop_type1: str,
    prop_type2: str,
    historical_data: pd.DataFrame
) -> float:
    """
    Calculate correlation between two prop types for a player.

    Example: passing_yards and passing_tds are highly correlated (0.7+)

    Args:
        player_id: GSIS player ID
        prop_type1: First prop type
        prop_type2: Second prop type
        historical_data: Historical player stats

    Returns:
        Correlation coefficient (-1 to 1)
    """
    try:
        player_data = historical_data[historical_data['player_id'] == player_id]

        if len(player_data) < 10:  # Need at least 10 games
            return 0.0

        if prop_type1 not in player_data.columns or prop_type2 not in player_data.columns:
            return 0.0

        corr = player_data[prop_type1].corr(player_data[prop_type2])
        return corr if not np.isnan(corr) else 0.0

    except Exception as e:
        print(f"Error calculating correlation: {e}")
        return 0.0


# Default correlation matrix (based on typical NFL stats)
DEFAULT_CORRELATIONS = {
    ('passing_yards', 'passing_tds'): 0.72,
    ('passing_yards', 'completions'): 0.85,
    ('passing_tds', 'completions'): 0.58,
    ('rushing_yards', 'rushing_attempts'): 0.91,
    ('rushing_yards', 'rushing_tds'): 0.45,
    ('receiving_yards', 'receptions'): 0.88,
    ('receiving_yards', 'receiving_tds'): 0.42,
}


def get_correlation(prop_type1: str, prop_type2: str) -> float:
    """Get correlation between two prop types."""
    if prop_type1 == prop_type2:
        return 1.0

    # Try both orderings
    corr = DEFAULT_CORRELATIONS.get((prop_type1, prop_type2))
    if corr is None:
        corr = DEFAULT_CORRELATIONS.get((prop_type2, prop_type1))

    return corr if corr is not None else 0.0


# ============================================================================
# Props EV Bet Selector
# ============================================================================

class PropsEVSelector:
    """EV-optimal bet selector for player props."""

    def __init__(
        self,
        min_edge: float = 0.03,             # 3% minimum edge (higher than games)
        deadzone: float = 0.005,             # 0.5% no-bet band
        max_book_hold: float = 0.10,        # Max 10% hold (props have higher hold)
        min_ev_dollars: float = 3.0,        # Min $3 EV per $100 bet
        kelly_fraction: float = 0.15,       # More conservative for props (15% vs 25%)
        max_bet_pct: float = 0.03,          # Max 3% of bankroll per bet
        check_injuries: bool = True,
        max_correlated_exposure: float = 0.08,  # Max 8% exposure to correlated props
    ):
        self.min_edge = min_edge
        self.deadzone = deadzone
        self.max_book_hold = max_book_hold
        self.min_ev_dollars = min_ev_dollars
        self.kelly_fraction = kelly_fraction
        self.max_bet_pct = max_bet_pct
        self.check_injuries = check_injuries
        self.max_correlated_exposure = max_correlated_exposure

        self.historical_data = pd.DataFrame()
        self.db_conn = None

    def load_historical_data(self, data: pd.DataFrame):
        """Load historical player stats for correlation analysis."""
        self.historical_data = data

    def set_db_connection(self, conn):
        """Set database connection for injury checks."""
        self.db_conn = conn

    def should_bet_prop(
        self,
        player_id: str,
        player_name: str,
        prop_type: str,
        model_prediction: float,
        model_std: float,
        line_value: float,
        over_odds: int,
        under_odds: int,
        bookmaker: str,
        game_date: datetime = None,
        bankroll: float = 10000.0,
        current_exposure: Dict = None
    ) -> Dict:
        """
        Determine if a prop bet should be placed.

        Args:
            player_id: GSIS player ID
            player_name: Player name
            prop_type: Type of prop (e.g., 'passing_yards')
            model_prediction: Model's predicted value
            model_std: Model's uncertainty (std dev)
            line_value: Prop line
            over_odds: Odds for over
            under_odds: Odds for under
            bookmaker: Sportsbook name
            game_date: Game date
            bankroll: Current bankroll
            current_exposure: Dict of current prop exposure by player/type

        Returns:
            Dict with bet decision and details
        """
        result = {
            'should_bet': False,
            'reason': '',
            'bet_side': None,
            'edge': 0.0,
            'ev': 0.0,
            'kelly_size': 0.0,
            'bet_amount': 0.0,
            'confidence': 0.0,
        }

        current_exposure = current_exposure or {}

        # Calculate hold
        prob_over_implied = american_odds_to_implied_prob(over_odds)
        prob_under_implied = american_odds_to_implied_prob(under_odds)
        hold = (prob_over_implied + prob_under_implied) - 1.0

        if hold > self.max_book_hold:
            result['reason'] = f'Hold too high: {hold:.3f}'
            return result

        # Check injury status
        if self.check_injuries and self.db_conn and game_date:
            is_healthy, injury_status = check_injury_status(player_id, game_date, self.db_conn)
            if not is_healthy:
                result['reason'] = f'Player injury: {injury_status}'
                return result
            elif injury_status == 'Questionable':
                # Require higher edge for questionable players
                required_edge = self.min_edge * 1.5
            else:
                required_edge = self.min_edge
        else:
            required_edge = self.min_edge

        # Calculate edge with hold adjustment
        hold_adjustment = hold / 2 if hold > 0 else 0
        edge_calc = calculate_prop_edge(
            model_prediction=model_prediction,
            model_std=model_std,
            line_value=line_value,
            over_odds=over_odds,
            under_odds=under_odds,
            hold_adjustment=hold_adjustment
        )

        best_edge = edge_calc['best_edge']
        best_bet = edge_calc['best_bet']
        best_odds = edge_calc['best_odds']
        best_prob = edge_calc['best_prob']

        result['bet_side'] = best_bet
        result['edge'] = best_edge

        # Check minimum edge
        if best_edge < required_edge:
            result['reason'] = f'Insufficient edge: {best_edge:.3f} (need {required_edge:.3f})'
            return result

        # Check deadzone
        if abs(best_edge) < self.deadzone:
            result['reason'] = f'In deadzone: {best_edge:.3f}'
            return result

        # Calculate EV
        ev_per_100 = calculate_prop_ev(best_prob, best_odds, 100.0)
        result['ev'] = ev_per_100

        if ev_per_100 < self.min_ev_dollars:
            result['reason'] = f'Insufficient EV: ${ev_per_100:.2f}'
            return result

        # Check correlation / exposure limits
        # If we already have bets on correlated props for this player, reduce sizing
        exposure_key = f"{player_id}_{prop_type}"
        total_correlated_exposure = 0.0

        for existing_bet_key, existing_amount in current_exposure.items():
            existing_player_id, existing_prop_type = existing_bet_key.split('_', 1)
            if existing_player_id == player_id:
                corr = get_correlation(prop_type, existing_prop_type)
                if abs(corr) > 0.5:  # Significant correlation
                    total_correlated_exposure += (existing_amount / bankroll) * abs(corr)

        if total_correlated_exposure > self.max_correlated_exposure:
            result['reason'] = f'Too much correlated exposure: {total_correlated_exposure:.2%}'
            return result

        # Calculate Kelly size
        decimal_odds = american_to_decimal_odds(best_odds)
        kelly_size = kelly_criterion(
            win_prob=best_prob,
            decimal_odds=decimal_odds,
            kelly_fraction=self.kelly_fraction,
            max_bet_fraction=self.max_bet_pct
        )

        # Further reduce sizing if there's correlated exposure
        if total_correlated_exposure > 0.02:  # > 2% correlated exposure
            correlation_adj = 1 - (total_correlated_exposure / self.max_correlated_exposure)
            kelly_size *= max(0.5, correlation_adj)  # At least 50% sizing

        result['kelly_size'] = kelly_size
        result['bet_amount'] = kelly_size * bankroll

        # Calculate confidence score (0-100)
        confidence = 50.0
        confidence += best_edge * 500  # +5 points per 1% edge
        confidence += min(20, ev_per_100 * 2)  # Up to +20 for EV

        # Reduce confidence for high uncertainty
        if model_std > 0:
            uncertainty_factor = model_std / abs(model_prediction - line_value + 0.1)
            confidence *= (1 - min(0.3, uncertainty_factor))  # Up to 30% reduction

        # Reduce confidence for correlated exposure
        confidence *= (1 - total_correlated_exposure)

        confidence = min(100, max(0, confidence))
        result['confidence'] = confidence

        result['should_bet'] = True
        result['reason'] = 'All checks passed'

        return result

    def select_props(
        self,
        predictions_df: pd.DataFrame,
        prop_lines_df: pd.DataFrame,
        bankroll: float = 10000.0,
        max_bets: int = None,
    ) -> pd.DataFrame:
        """
        Select prop bets from predictions and available lines.

        Args:
            predictions_df: DataFrame with columns: player_id, prop_type, prediction, std
            prop_lines_df: DataFrame with prop lines and odds
            bankroll: Current bankroll
            max_bets: Maximum number of bets to select

        Returns:
            DataFrame of selected prop bets
        """
        selected = []
        current_exposure = {}

        # Merge predictions with prop lines
        merged = predictions_df.merge(
            prop_lines_df,
            on=['player_id', 'prop_type'],
            how='inner'
        )

        # Sort by expected EV (approximate)
        if 'expected_ev' not in merged.columns:
            merged['expected_ev'] = abs(merged['prediction'] - merged['line_value'])

        merged = merged.sort_values('expected_ev', ascending=False)

        for _, row in merged.iterrows():
            # Check if should bet
            decision = self.should_bet_prop(
                player_id=row.get('player_id'),
                player_name=row.get('player_name'),
                prop_type=row.get('prop_type'),
                model_prediction=row.get('prediction'),
                model_std=row.get('std', row.get('prediction') * 0.1),  # Default 10% uncertainty
                line_value=row.get('line_value'),
                over_odds=row.get('over_odds'),
                under_odds=row.get('under_odds'),
                bookmaker=row.get('bookmaker', 'unknown'),
                game_date=row.get('game_date'),
                bankroll=bankroll,
                current_exposure=current_exposure
            )

            if decision['should_bet']:
                bet_row = {
                    'player_id': row['player_id'],
                    'player_name': row.get('player_name'),
                    'prop_type': row['prop_type'],
                    'line_value': row['line_value'],
                    'bet_side': decision['bet_side'],
                    'odds': decision.get('best_odds'),
                    'prediction': row['prediction'],
                    'edge': decision['edge'],
                    'ev': decision['ev'],
                    'kelly_size': decision['kelly_size'],
                    'bet_amount': decision['bet_amount'],
                    'confidence': decision['confidence'],
                    'bookmaker': row.get('bookmaker', 'unknown'),
                    'game_date': row.get('game_date'),
                }
                selected.append(bet_row)

                # Track exposure
                exposure_key = f"{row['player_id']}_{row['prop_type']}"
                current_exposure[exposure_key] = decision['bet_amount']

        if not selected:
            return pd.DataFrame()

        selected_df = pd.DataFrame(selected)

        # Sort by EV and limit
        selected_df = selected_df.sort_values('ev', ascending=False)

        if max_bets:
            selected_df = selected_df.head(max_bets)

        return selected_df


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='EV-Optimal Prop Bet Selector'
    )

    # Input files
    parser.add_argument('--predictions', required=True, help='Predictions CSV file')
    parser.add_argument('--prop-lines', required=True, help='Prop lines CSV file')

    # Selection parameters
    parser.add_argument('--min-edge', type=float, default=0.03,
                       help='Minimum edge (default: 0.03)')
    parser.add_argument('--deadzone', type=float, default=0.005,
                       help='No-bet band (default: 0.005)')
    parser.add_argument('--max-hold', type=float, default=0.10,
                       help='Maximum book hold (default: 0.10)')
    parser.add_argument('--min-ev', type=float, default=3.0,
                       help='Minimum EV per $100 (default: 3.0)')

    # Kelly sizing
    parser.add_argument('--kelly-fraction', type=float, default=0.15,
                       help='Kelly fraction (default: 0.15)')
    parser.add_argument('--max-bet-pct', type=float, default=0.03,
                       help='Max bet pct of bankroll (default: 0.03)')
    parser.add_argument('--bankroll', type=float, default=10000.0,
                       help='Current bankroll (default: 10000)')

    # Output
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--max-bets', type=int, help='Maximum number of bets')

    # Flags
    parser.add_argument('--no-injury-check', action='store_true',
                       help='Skip injury status checking')

    args = parser.parse_args()

    print("=" * 80)
    print("Props EV-Optimal Bet Selector")
    print("=" * 80)

    # Load data
    predictions_df = pd.read_csv(args.predictions)
    prop_lines_df = pd.read_csv(args.prop_lines)

    print(f"Loaded {len(predictions_df)} predictions")
    print(f"Loaded {len(prop_lines_df)} prop lines")

    # Initialize selector
    selector = PropsEVSelector(
        min_edge=args.min_edge,
        deadzone=args.deadzone,
        max_book_hold=args.max_hold,
        min_ev_dollars=args.min_ev,
        kelly_fraction=args.kelly_fraction,
        max_bet_pct=args.max_bet_pct,
        check_injuries=not args.no_injury_check,
    )

    # Select props
    selected = selector.select_props(
        predictions_df=predictions_df,
        prop_lines_df=prop_lines_df,
        bankroll=args.bankroll,
        max_bets=args.max_bets,
    )

    if selected.empty:
        print("\nNo props meet selection criteria")
        return 0

    # Save results
    selected.to_csv(args.output, index=False)

    print(f"\nSelected {len(selected)} prop bets:")
    print(f"  Total to wager: ${selected['bet_amount'].sum():,.0f}")
    print(f"  Average edge: {selected['edge'].mean()*100:.2f}%")
    print(f"  Average EV: ${selected['ev'].mean():.2f}")
    print(f"  Average confidence: {selected['confidence'].mean():.1f}")

    print(f"\nBets saved to {args.output}")

    # Show top bets
    print("\nTop 10 bets by EV:")
    print(selected.nlargest(10, 'ev')[['player_name', 'prop_type', 'bet_side', 'line_value', 'edge', 'ev', 'bet_amount']])

    return 0


if __name__ == '__main__':
    sys.exit(main())
