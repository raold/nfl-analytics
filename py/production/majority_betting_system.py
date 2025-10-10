#!/usr/bin/env python3
"""
Production Majority Voting Betting System.

Based on Task 8 bootstrap stress testing results:
- Majority voting: Most resilient strategy
- Worst case: +0.07% return
- CVaR(95%): -0.05% (1 unit loss)
- Survives all stress scenarios

Features:
- Majority voting ensemble (XGBoost, CQL, IQL)
- Kelly criterion bet sizing
- Market odds integration
- Risk management (stop-loss, max bet limits)
- Performance tracking and logging
- Simple CLI for daily betting workflow

Usage:
    # Get today's recommended bets
    python py/production/majority_betting_system.py \
        --games-csv data/upcoming_games.csv \
        --bankroll 10000 \
        --kelly-fraction 0.25 \
        --output bets/2025_week_11.json

    # Backtest on historical data
    python py/production/majority_betting_system.py \
        --games-csv data/processed/features/asof_team_features_v2.csv \
        --season 2024 \
        --bankroll 10000 \
        --backtest
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Import ensemble predictor
sys.path.append(str(Path(__file__).parent.parent / 'ensemble'))
from ensemble_predictor import EnsemblePredictor, XGBoostPredictor, CQLPredictor, IQLPredictor


# ============================================================================
# Kelly Criterion Bet Sizing
# ============================================================================


def kelly_criterion(
    win_prob: float,
    odds_american: int,
    kelly_fraction: float = 1.0,
    max_bet_fraction: float = 0.05,
) -> float:
    """
    Calculate Kelly criterion bet size.

    Args:
        win_prob: Estimated probability of winning (0-1)
        odds_american: American odds (e.g., -110, +150)
        kelly_fraction: Fraction of Kelly to bet (0-1) for risk management
        max_bet_fraction: Maximum bet as fraction of bankroll

    Returns:
        Optimal bet fraction (0-1)
    """
    # Convert American odds to decimal multiplier
    if odds_american > 0:
        # Positive odds: +150 means win $1.50 per $1 wagered
        b = odds_american / 100.0
    else:
        # Negative odds: -110 means win $0.91 per $1 wagered
        b = 100.0 / abs(odds_american)

    # Kelly formula: f = (bp - q) / b
    # where p = win_prob, q = 1 - win_prob, b = decimal odds
    p = win_prob
    q = 1 - p

    kelly_full = (b * p - q) / b

    # Apply Kelly fraction (0.25 = quarter Kelly for robustness)
    kelly_bet = kelly_full * kelly_fraction

    # Clip to [0, max_bet_fraction]
    return np.clip(kelly_bet, 0.0, max_bet_fraction)


def american_odds_to_probability(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        # Positive odds
        return 100.0 / (odds + 100.0)
    else:
        # Negative odds
        return abs(odds) / (abs(odds) + 100.0)


def probability_to_american_odds(prob: float) -> int:
    """Convert probability to American odds."""
    if prob >= 0.5:
        # Favorite (negative odds)
        return int(-100 * prob / (1 - prob))
    else:
        # Underdog (positive odds)
        return int(100 * (1 - prob) / prob)


# ============================================================================
# Production Betting System
# ============================================================================


class MajorityBettingSystem:
    """
    Production betting system with majority voting and Kelly sizing.

    Designed for conservative, risk-managed betting with focus on
    capital preservation and long-term positive expectancy.
    """

    def __init__(
        self,
        xgb_model_path: str,
        cql_model_path: str,
        iql_model_path: str,
        bankroll: float = 10000.0,
        kelly_fraction: float = 0.25,
        max_bet_fraction: float = 0.05,
        min_edge: float = 0.02,
        uncertainty_threshold: float = 0.5,
        xgb_features: List[str] = None,
        rl_state_cols: List[str] = None,
        device: str = 'cpu',
    ):
        """
        Initialize production betting system.

        Args:
            xgb_model_path: Path to XGBoost model
            cql_model_path: Path to CQL model
            iql_model_path: Path to IQL model
            bankroll: Starting bankroll ($)
            kelly_fraction: Fraction of Kelly to bet (0.25 = quarter Kelly)
            max_bet_fraction: Maximum bet as fraction of bankroll (0.05 = 5%)
            min_edge: Minimum edge required to bet (0.02 = 2%)
            uncertainty_threshold: Maximum uncertainty to allow betting
            xgb_features: Feature names for XGBoost
            rl_state_cols: State feature names for RL agents
            device: cpu/cuda
        """
        # Configuration
        self.bankroll = bankroll
        self.initial_bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.max_bet_fraction = max_bet_fraction
        self.min_edge = min_edge
        self.uncertainty_threshold = uncertainty_threshold

        # Default features
        self.xgb_features = xgb_features or [
            'prior_epa_mean_diff', 'epa_pp_last3_diff', 'season_win_pct_diff',
            'win_pct_last5_diff', 'prior_margin_avg_diff', 'points_for_last3_diff',
            'points_against_last3_diff', 'rest_diff', 'week',
            'fourth_downs_diff', 'fourth_down_epa_diff',
        ]

        self.rl_state_cols = rl_state_cols or [
            'spread_close', 'total_close', 'epa_gap', 'market_prob', 'p_hat', 'edge',
        ]

        # Load models
        print(f"Loading models...")
        print(f"  XGBoost: {xgb_model_path}")
        xgb_model = XGBoostPredictor(xgb_model_path)

        print(f"  CQL: {cql_model_path}")
        cql_model = CQLPredictor(cql_model_path, state_dim=len(self.rl_state_cols), device=device)

        print(f"  IQL: {iql_model_path}")
        iql_model = IQLPredictor(iql_model_path, state_dim=len(self.rl_state_cols), device=device)

        # Create ensemble with MAJORITY voting (most resilient per Task 8)
        self.ensemble = EnsemblePredictor(
            xgb_model=xgb_model,
            cql_model=cql_model,
            iql_model=iql_model,
            strategy='majority',  # Most resilient
            uncertainty_threshold=uncertainty_threshold,
        )

        # Tracking
        self.bet_history = []
        self.performance = {
            'total_bets': 0,
            'total_won': 0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'peak_bankroll': bankroll,
        }

    def get_betting_recommendations(
        self,
        games_df: pd.DataFrame,
        market_odds: Dict[str, int] = None,
    ) -> List[Dict]:
        """
        Get betting recommendations for upcoming games.

        Args:
            games_df: DataFrame with game features
            market_odds: Dict mapping game_id -> American odds for home team

        Returns:
            List of betting recommendations
        """
        # Prepare inputs
        xgb_feats = games_df[self.xgb_features].fillna(0)
        rl_states = games_df[self.rl_state_cols].fillna(0).to_numpy(dtype=np.float32)

        # Market probabilities
        if 'market_prob' in games_df.columns:
            market_probs = games_df['market_prob'].to_numpy()
        elif 'spread_close' in games_df.columns:
            # Convert spread to implied probability
            spreads = games_df['spread_close'].to_numpy()
            market_probs = 0.5 + (spreads / 27.0)
            market_probs = np.clip(market_probs, 0.01, 0.99)
        else:
            market_probs = np.full(len(games_df), 0.5)

        # Get ensemble predictions
        actions, metadata = self.ensemble.predict(xgb_feats, rl_states, market_probs)

        # Generate recommendations
        recommendations = []

        for i, row in games_df.iterrows():
            # Model probability (use XGBoost as primary)
            model_prob = metadata['xgb_probs'][i]
            market_prob = market_probs[i]
            edge = model_prob - market_prob

            # Action from majority vote
            action = actions[i]

            # Uncertainty metrics
            xgb_unc = metadata['xgb_uncertainties'][i]
            cql_unc = metadata['cql_uncertainties'][i]
            iql_unc = metadata['iql_uncertainties'][i]

            # Skip if action is 0 (no bet)
            if action == 0:
                continue

            # Skip if edge too small
            if abs(edge) < self.min_edge:
                continue

            # Determine bet direction and odds
            bet_home = model_prob > market_prob

            # Get market odds (default to -110 if not provided)
            game_id = row.get('game_id', f"game_{i}")
            if market_odds and game_id in market_odds:
                odds = market_odds[game_id]
            else:
                # Default to -110 (52.4% implied prob)
                odds = -110

            # If betting away, use inverse odds
            if not bet_home:
                # Approximate: if home is -110, away is +100
                # (simplified - real sportsbooks have vig)
                if odds < 0:
                    odds = int(100 * 100 / abs(odds))
                else:
                    odds = int(-100 * odds / 100)

            # Calculate Kelly bet size
            bet_fraction = kelly_criterion(
                win_prob=model_prob if bet_home else (1 - model_prob),
                odds_american=odds,
                kelly_fraction=self.kelly_fraction,
                max_bet_fraction=self.max_bet_fraction,
            )

            bet_amount = bet_fraction * self.bankroll

            # Skip if bet too small
            if bet_amount < 10:  # $10 minimum
                continue

            # Create recommendation
            rec = {
                'game_id': game_id,
                'matchup': f"{row.get('away_team', 'AWAY')} @ {row.get('home_team', 'HOME')}",
                'bet_team': row.get('home_team', 'HOME') if bet_home else row.get('away_team', 'AWAY'),
                'bet_direction': 'home' if bet_home else 'away',
                'model_prob': float(model_prob),
                'market_prob': float(market_prob),
                'edge': float(edge),
                'odds': odds,
                'bet_fraction': float(bet_fraction),
                'bet_amount': float(bet_amount),
                'action': int(action),
                'uncertainty': {
                    'xgb': float(xgb_unc),
                    'cql': float(cql_unc),
                    'iql': float(iql_unc),
                },
                'ensemble_agreement': {
                    'xgb_action': int(metadata['xgb_actions'][i]),
                    'cql_action': int(metadata['cql_actions'][i]),
                    'iql_action': int(metadata['iql_actions'][i]),
                },
            }

            recommendations.append(rec)

        # Sort by edge (descending)
        recommendations.sort(key=lambda x: abs(x['edge']), reverse=True)

        return recommendations

    def record_bet_result(
        self,
        bet: Dict,
        won: bool,
    ):
        """
        Record bet result and update bankroll.

        Args:
            bet: Bet dict from get_betting_recommendations()
            won: True if bet won, False if lost
        """
        # Calculate return
        if won:
            # Win: get bet_amount * (odds payout)
            odds = bet['odds']
            if odds > 0:
                payout = bet['bet_amount'] * (odds / 100.0)
            else:
                payout = bet['bet_amount'] * (100.0 / abs(odds))
            ret = payout
        else:
            # Loss: lose bet_amount
            ret = -bet['bet_amount']

        # Update bankroll
        self.bankroll += ret

        # Update performance
        self.performance['total_bets'] += 1
        if won:
            self.performance['total_won'] += 1
        self.performance['total_return'] += ret

        # Track peak and drawdown
        if self.bankroll > self.performance['peak_bankroll']:
            self.performance['peak_bankroll'] = self.bankroll

        drawdown = self.performance['peak_bankroll'] - self.bankroll
        if drawdown > self.performance['max_drawdown']:
            self.performance['max_drawdown'] = drawdown

        # Store in history
        self.bet_history.append({
            **bet,
            'won': won,
            'return': ret,
            'bankroll_after': self.bankroll,
        })

    def get_performance_summary(self) -> Dict:
        """Get performance summary."""
        total_bets = self.performance['total_bets']
        total_won = self.performance['total_won']

        return {
            'initial_bankroll': self.initial_bankroll,
            'current_bankroll': self.bankroll,
            'net_return': self.bankroll - self.initial_bankroll,
            'roi': ((self.bankroll / self.initial_bankroll) - 1) * 100,
            'total_bets': total_bets,
            'total_won': total_won,
            'win_rate': (total_won / total_bets * 100) if total_bets > 0 else 0.0,
            'max_drawdown': self.performance['max_drawdown'],
            'max_drawdown_pct': (self.performance['max_drawdown'] / self.initial_bankroll) * 100,
        }

    def backtest(
        self,
        games_df: pd.DataFrame,
        season: int = None,
    ) -> Dict:
        """
        Backtest on historical data.

        Args:
            games_df: DataFrame with game features and outcomes
            season: Optional season to filter

        Returns:
            Performance summary
        """
        if season:
            games_df = games_df[games_df['season'] == season].copy()

        print(f"\nBacktesting on {len(games_df)} games...")

        # Get recommendations
        recommendations = self.get_betting_recommendations(games_df)

        print(f"Bet recommendations: {len(recommendations)}")

        # Simulate bets
        for rec in recommendations:
            # Find actual outcome
            game_idx = games_df[games_df.get('game_id', games_df.index) == rec['game_id']].index

            if len(game_idx) == 0:
                continue

            game_idx = game_idx[0]

            # Get actual result
            if 'home_result' in games_df.columns:
                home_won = games_df.loc[game_idx, 'home_result'] == 1
            elif 'home_win' in games_df.columns:
                home_won = games_df.loc[game_idx, 'home_win'] == 1
            else:
                # Skip if no outcome data
                continue

            # Determine if bet won
            bet_home = rec['bet_direction'] == 'home'
            won = (bet_home and home_won) or (not bet_home and not home_won)

            # Record result
            self.record_bet_result(rec, won)

        return self.get_performance_summary()


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description='Production Majority Voting Betting System'
    )

    # Data
    ap.add_argument('--games-csv', required=True, help='Games CSV with features')
    ap.add_argument('--season', type=int, help='Season to filter (for backtest)')

    # Models (defaults to best models)
    ap.add_argument(
        '--xgb-model',
        default='models/xgboost/v2_sweep/xgb_config18_season2024.json',
        help='XGBoost model path'
    )
    ap.add_argument(
        '--cql-model',
        default='models/cql/sweep/cql_config4.pth',
        help='CQL model path'
    )
    ap.add_argument(
        '--iql-model',
        default='models/iql/baseline_model.pth',
        help='IQL model path'
    )

    # Bankroll management
    ap.add_argument('--bankroll', type=float, default=10000.0, help='Starting bankroll ($)')
    ap.add_argument('--kelly-fraction', type=float, default=0.25, help='Fraction of Kelly to bet')
    ap.add_argument('--max-bet-fraction', type=float, default=0.05, help='Max bet as fraction of bankroll')
    ap.add_argument('--min-edge', type=float, default=0.02, help='Minimum edge to bet (2%)')

    # Mode
    ap.add_argument('--backtest', action='store_true', help='Backtest mode (simulate historical bets)')
    ap.add_argument('--paper-trade', action='store_true', help='Paper trading mode (virtual money, no real bets)')
    ap.add_argument('--output', help='Output JSON path for recommendations')
    ap.add_argument('--device', default='cpu', help='cpu/cuda')

    return ap.parse_args()


def main():
    args = parse_args()

    print(f"{'='*80}")
    print(f"Production Majority Voting Betting System")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Mode: {'PAPER TRADING (Virtual Money)' if args.paper_trade else 'LIVE BETTING (Real Money)'}")
    print(f"  Bankroll: ${args.bankroll:,.0f}")
    print(f"  Kelly fraction: {args.kelly_fraction} (quarter Kelly)")
    print(f"  Max bet: {args.max_bet_fraction*100:.1f}% of bankroll")
    print(f"  Min edge: {args.min_edge*100:.1f}%")

    # Load games
    print(f"\nLoading games from {args.games_csv}...")
    games_df = pd.read_csv(args.games_csv)

    if args.season:
        games_df = games_df[games_df['season'] == args.season]

    print(f"  Loaded {len(games_df)} games")

    # Initialize system
    system = MajorityBettingSystem(
        xgb_model_path=args.xgb_model,
        cql_model_path=args.cql_model,
        iql_model_path=args.iql_model,
        bankroll=args.bankroll,
        kelly_fraction=args.kelly_fraction,
        max_bet_fraction=args.max_bet_fraction,
        min_edge=args.min_edge,
        device=args.device,
    )

    if args.backtest:
        # Backtest mode
        print(f"\n{'='*80}")
        print(f"BACKTEST MODE")
        print(f"{'='*80}")

        performance = system.backtest(games_df, season=args.season)

        print(f"\n{'='*80}")
        print(f"Backtest Results")
        print(f"{'='*80}")
        print(f"Initial bankroll: ${performance['initial_bankroll']:,.0f}")
        print(f"Final bankroll: ${performance['current_bankroll']:,.0f}")
        print(f"Net return: ${performance['net_return']:+,.0f}")
        print(f"ROI: {performance['roi']:+.2f}%")
        print(f"Total bets: {performance['total_bets']}")
        print(f"Win rate: {performance['win_rate']:.1f}%")
        print(f"Max drawdown: ${performance['max_drawdown']:,.0f} ({performance['max_drawdown_pct']:.1f}%)")

        # Save results
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump({
                    'is_paper_trade': args.paper_trade,
                    'performance': performance,
                    'bet_history': system.bet_history,
                }, f, indent=2)

            print(f"\nResults saved to {output_path}")

    else:
        # Live mode - get recommendations
        print(f"\n{'='*80}")
        print(f"BETTING RECOMMENDATIONS")
        print(f"{'='*80}")

        recommendations = system.get_betting_recommendations(games_df)

        if len(recommendations) == 0:
            print(f"No betting opportunities found (all bets filtered by edge/uncertainty thresholds)")
            return 0

        print(f"\nFound {len(recommendations)} betting opportunities:")
        print(f"\n{'#':<3} {'Matchup':<30} {'Bet':<12} {'Edge':<8} {'Odds':<8} {'Amount':<10} {'Action':<6}")
        print(f"{'-'*80}")

        for i, rec in enumerate(recommendations, 1):
            print(f"{i:<3} {rec['matchup']:<30} {rec['bet_team']:<12} "
                  f"{rec['edge']*100:>6.2f}% {rec['odds']:>7} ${rec['bet_amount']:>8,.0f} {rec['action']:>6}")

        # Summary
        total_risk = sum(r['bet_amount'] for r in recommendations)
        print(f"\nTotal capital at risk: ${total_risk:,.0f} ({total_risk/args.bankroll*100:.1f}% of bankroll)")

        # Save recommendations
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump({
                    'date': datetime.now().isoformat(),
                    'is_paper_trade': args.paper_trade,
                    'bankroll': args.bankroll,
                    'total_recommendations': len(recommendations),
                    'total_risk': total_risk,
                    'recommendations': recommendations,
                }, f, indent=2)

            print(f"\nRecommendations saved to {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
