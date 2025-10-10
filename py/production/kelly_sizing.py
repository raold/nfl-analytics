#!/usr/bin/env python3
"""
Kelly Criterion Bet Sizing for NFL Betting.

Implements optimal bet sizing using the Kelly Criterion with fractional Kelly
for risk management and bankroll preservation.

Theory:
    Full Kelly: f* = (bp - q) / b
    where:
        f* = optimal bet fraction
        b = decimal odds (payout per $1 wagered)
        p = win probability
        q = 1 - p (loss probability)

    Fractional Kelly: f = fraction * f*
        - Quarter Kelly (0.25): Most common for risk management
        - Half Kelly (0.5): Moderate growth, lower variance
        - Full Kelly (1.0): Maximum growth, high variance

Usage:
    # Calculate optimal bet size
    python py/production/kelly_sizing.py \
        --win-prob 0.55 \
        --odds -110 \
        --bankroll 10000 \
        --kelly-fraction 0.25

    # Simulate bankroll growth
    python py/production/kelly_sizing.py \
        --simulate \
        --win-prob 0.55 \
        --odds -110 \
        --bankroll 10000 \
        --n-bets 1000
"""

import argparse
import sys
from typing import Tuple, Dict, List
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# Kelly Criterion Calculations
# ============================================================================


def american_to_decimal_odds(american_odds: int) -> float:
    """
    Convert American odds to decimal odds (payout per $1 wagered).

    Args:
        american_odds: American odds (e.g., -110, +150)

    Returns:
        Decimal odds (e.g., 0.909, 1.5)
    """
    if american_odds > 0:
        # Positive odds: +150 means win $1.50 per $1 wagered
        return american_odds / 100.0
    else:
        # Negative odds: -110 means win $0.909 per $1 wagered
        return 100.0 / abs(american_odds)


def decimal_to_american_odds(decimal_odds: float) -> int:
    """Convert decimal odds to American odds."""
    if decimal_odds >= 1.0:
        # Positive American odds
        return int(decimal_odds * 100)
    else:
        # Negative American odds
        return int(-100 / decimal_odds)


def american_odds_to_implied_prob(american_odds: int) -> float:
    """
    Convert American odds to implied probability (with vig).

    Args:
        american_odds: American odds

    Returns:
        Implied probability (0-1)
    """
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    else:
        return abs(american_odds) / (abs(american_odds) + 100.0)


def kelly_criterion(
    win_prob: float,
    decimal_odds: float,
    kelly_fraction: float = 1.0,
    max_bet_fraction: float = 0.10,
) -> float:
    """
    Calculate Kelly criterion bet size.

    Args:
        win_prob: Probability of winning (0-1)
        decimal_odds: Decimal odds (payout per $1)
        kelly_fraction: Fraction of Kelly to bet (0-1)
        max_bet_fraction: Maximum bet as fraction of bankroll

    Returns:
        Optimal bet fraction (0-1)
    """
    # Kelly formula: f* = (bp - q) / b
    b = decimal_odds
    p = win_prob
    q = 1 - p

    # Full Kelly
    kelly_full = (b * p - q) / b

    # Apply Kelly fraction
    kelly_bet = kelly_full * kelly_fraction

    # Ensure non-negative (only bet if +EV)
    kelly_bet = max(0.0, kelly_bet)

    # Cap at max bet fraction
    kelly_bet = min(kelly_bet, max_bet_fraction)

    return kelly_bet


def estimate_edge(
    model_prob: float,
    market_odds: int,
) -> float:
    """
    Estimate betting edge (model prob - implied prob).

    Args:
        model_prob: Model's win probability
        market_odds: American odds from sportsbook

    Returns:
        Edge (positive = +EV, negative = -EV)
    """
    implied_prob = american_odds_to_implied_prob(market_odds)
    return model_prob - implied_prob


def optimal_bet_size(
    model_prob: float,
    market_odds: int,
    bankroll: float,
    kelly_fraction: float = 0.25,
    max_bet_fraction: float = 0.05,
    min_edge: float = 0.01,
) -> Dict:
    """
    Calculate optimal bet size with Kelly criterion.

    Args:
        model_prob: Model's win probability
        market_odds: American odds from sportsbook
        bankroll: Current bankroll ($)
        kelly_fraction: Fraction of Kelly to bet
        max_bet_fraction: Maximum bet as fraction of bankroll
        min_edge: Minimum edge required to bet

    Returns:
        Dict with bet recommendation
    """
    # Calculate edge
    edge = estimate_edge(model_prob, market_odds)

    # Check if +EV
    if edge < min_edge:
        return {
            'should_bet': False,
            'edge': edge,
            'reason': f'Insufficient edge ({edge:.3f} < {min_edge:.3f})',
        }

    # Calculate Kelly bet
    decimal_odds = american_to_decimal_odds(market_odds)
    bet_fraction = kelly_criterion(
        win_prob=model_prob,
        decimal_odds=decimal_odds,
        kelly_fraction=kelly_fraction,
        max_bet_fraction=max_bet_fraction,
    )

    bet_amount = bet_fraction * bankroll

    # Expected value
    ev = bet_amount * (model_prob * decimal_odds - (1 - model_prob))

    return {
        'should_bet': True,
        'edge': edge,
        'model_prob': model_prob,
        'implied_prob': american_odds_to_implied_prob(market_odds),
        'decimal_odds': decimal_odds,
        'bet_fraction': bet_fraction,
        'bet_amount': bet_amount,
        'expected_value': ev,
        'roi': (ev / bet_amount) * 100 if bet_amount > 0 else 0.0,
    }


# ============================================================================
# Bankroll Growth Simulation
# ============================================================================


def simulate_kelly_growth(
    win_prob: float,
    decimal_odds: float,
    initial_bankroll: float,
    n_bets: int,
    kelly_fraction: float = 1.0,
    max_bet_fraction: float = 0.10,
    seed: int = None,
) -> Dict:
    """
    Simulate bankroll growth using Kelly criterion.

    Args:
        win_prob: Win probability
        decimal_odds: Decimal odds
        initial_bankroll: Starting bankroll
        n_bets: Number of bets to simulate
        kelly_fraction: Fraction of Kelly to bet
        max_bet_fraction: Maximum bet fraction
        seed: Random seed for reproducibility

    Returns:
        Dict with simulation results
    """
    if seed is not None:
        np.random.seed(seed)

    # Calculate Kelly bet fraction
    bet_fraction = kelly_criterion(win_prob, decimal_odds, kelly_fraction, max_bet_fraction)

    # Initialize
    bankroll = initial_bankroll
    bankroll_history = [bankroll]

    wins = 0
    losses = 0

    # Simulate bets
    for i in range(n_bets):
        # Bet size (fixed fraction strategy)
        bet_size = bet_fraction * bankroll

        # Outcome
        win = np.random.random() < win_prob

        if win:
            # Win: gain bet_size * decimal_odds
            bankroll += bet_size * decimal_odds
            wins += 1
        else:
            # Loss: lose bet_size
            bankroll -= bet_size
            losses += 1

        bankroll_history.append(bankroll)

    # Calculate metrics
    bankroll_history = np.array(bankroll_history)
    returns = np.diff(bankroll_history)

    peak = np.maximum.accumulate(bankroll_history)
    drawdown = (peak - bankroll_history) / peak * 100  # percentage

    return {
        'final_bankroll': bankroll,
        'total_return': bankroll - initial_bankroll,
        'roi': ((bankroll / initial_bankroll) - 1) * 100,
        'n_bets': n_bets,
        'wins': wins,
        'losses': losses,
        'win_rate': (wins / n_bets) * 100,
        'bet_fraction': bet_fraction,
        'max_drawdown_pct': drawdown.max(),
        'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
        'bankroll_history': bankroll_history,
    }


def compare_kelly_fractions(
    win_prob: float,
    decimal_odds: float,
    initial_bankroll: float,
    n_bets: int,
    fractions: List[float] = None,
    n_simulations: int = 100,
) -> Dict:
    """
    Compare different Kelly fractions via Monte Carlo simulation.

    Args:
        win_prob: Win probability
        decimal_odds: Decimal odds
        initial_bankroll: Starting bankroll
        n_bets: Number of bets per simulation
        fractions: List of Kelly fractions to compare
        n_simulations: Number of Monte Carlo trials

    Returns:
        Dict with comparison results
    """
    if fractions is None:
        fractions = [0.125, 0.25, 0.5, 0.75, 1.0]  # 1/8, 1/4, 1/2, 3/4, Full Kelly

    results = {}

    for frac in fractions:
        final_bankrolls = []
        max_drawdowns = []

        for sim in range(n_simulations):
            sim_result = simulate_kelly_growth(
                win_prob=win_prob,
                decimal_odds=decimal_odds,
                initial_bankroll=initial_bankroll,
                n_bets=n_bets,
                kelly_fraction=frac,
                max_bet_fraction=0.50,  # Allow large bets for comparison
                seed=sim,
            )
            final_bankrolls.append(sim_result['final_bankroll'])
            max_drawdowns.append(sim_result['max_drawdown_pct'])

        final_bankrolls = np.array(final_bankrolls)
        max_drawdowns = np.array(max_drawdowns)

        results[f'{frac:.3f}'] = {
            'kelly_fraction': frac,
            'median_final': np.median(final_bankrolls),
            'mean_final': np.mean(final_bankrolls),
            'std_final': np.std(final_bankrolls),
            'p5': np.percentile(final_bankrolls, 5),
            'p95': np.percentile(final_bankrolls, 95),
            'median_drawdown': np.median(max_drawdowns),
            'max_drawdown': np.max(max_drawdowns),
            'ruin_rate': np.mean(final_bankrolls < initial_bankroll * 0.5) * 100,  # >50% loss
        }

    return results


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description='Kelly Criterion Bet Sizing for NFL Betting'
    )

    # Single bet calculation
    ap.add_argument('--win-prob', type=float, help='Win probability (0-1)')
    ap.add_argument('--odds', type=int, help='American odds (e.g., -110)')
    ap.add_argument('--bankroll', type=float, default=10000.0, help='Bankroll ($)')
    ap.add_argument('--kelly-fraction', type=float, default=0.25, help='Fraction of Kelly to bet')
    ap.add_argument('--max-bet-fraction', type=float, default=0.05, help='Max bet fraction')
    ap.add_argument('--min-edge', type=float, default=0.01, help='Minimum edge to bet')

    # Simulation
    ap.add_argument('--simulate', action='store_true', help='Run simulation')
    ap.add_argument('--n-bets', type=int, default=1000, help='Number of bets to simulate')
    ap.add_argument('--n-simulations', type=int, default=100, help='Number of Monte Carlo trials')
    ap.add_argument('--compare-fractions', action='store_true', help='Compare Kelly fractions')

    # Output
    ap.add_argument('--plot', action='store_true', help='Generate plots')
    ap.add_argument('--output', help='Output file for plot (e.g., kelly_growth.png)')

    return ap.parse_args()


def main():
    args = parse_args()

    print(f"{'='*80}")
    print(f"Kelly Criterion Bet Sizing")
    print(f"{'='*80}")

    # Validate inputs
    if not args.simulate and not args.win_prob:
        print("ERROR: --win-prob required for single bet calculation")
        return 1

    if not args.simulate and not args.odds:
        print("ERROR: --odds required for single bet calculation")
        return 1

    if args.simulate:
        # Simulation mode
        print(f"\nSimulation Parameters:")
        print(f"  Win probability: {args.win_prob:.3f}")
        print(f"  Odds: {args.odds} ({american_to_decimal_odds(args.odds):.3f} decimal)")
        print(f"  Initial bankroll: ${args.bankroll:,.0f}")
        print(f"  Number of bets: {args.n_bets}")

        if args.compare_fractions:
            # Compare Kelly fractions
            print(f"\n{'='*80}")
            print(f"Comparing Kelly Fractions ({args.n_simulations} simulations each)")
            print(f"{'='*80}")

            decimal_odds = american_to_decimal_odds(args.odds)
            results = compare_kelly_fractions(
                win_prob=args.win_prob,
                decimal_odds=decimal_odds,
                initial_bankroll=args.bankroll,
                n_bets=args.n_bets,
                n_simulations=args.n_simulations,
            )

            # Print table
            print(f"\n{'Fraction':<10} {'Median Final':<15} {'Mean Final':<15} "
                  f"{'5th %ile':<12} {'95th %ile':<12} {'Med DD %':<10} {'Ruin %':<8}")
            print(f"{'-'*90}")

            for frac, res in sorted(results.items(), key=lambda x: float(x[0])):
                print(f"{res['kelly_fraction']:<10.3f} "
                      f"${res['median_final']:<14,.0f} "
                      f"${res['mean_final']:<14,.0f} "
                      f"${res['p5']:<11,.0f} "
                      f"${res['p95']:<11,.0f} "
                      f"{res['median_drawdown']:<9.1f} "
                      f"{res['ruin_rate']:<7.1f}")

            print(f"\nRecommendation:")
            print(f"  Quarter Kelly (0.25): Best risk-adjusted returns")
            print(f"  Half Kelly (0.5): Moderate growth, acceptable drawdowns")
            print(f"  Full Kelly (1.0): Maximum growth, high variance (not recommended)")

        else:
            # Single simulation
            decimal_odds = american_to_decimal_odds(args.odds)
            result = simulate_kelly_growth(
                win_prob=args.win_prob,
                decimal_odds=decimal_odds,
                initial_bankroll=args.bankroll,
                n_bets=args.n_bets,
                kelly_fraction=args.kelly_fraction,
                max_bet_fraction=args.max_bet_fraction,
            )

            print(f"\nSimulation Results:")
            print(f"  Kelly fraction: {result['bet_fraction']:.4f} ({args.kelly_fraction*100:.0f}% Kelly)")
            print(f"  Final bankroll: ${result['final_bankroll']:,.0f}")
            print(f"  Total return: ${result['total_return']:+,.0f}")
            print(f"  ROI: {result['roi']:+.2f}%")
            print(f"  Win rate: {result['win_rate']:.1f}% ({result['wins']}/{result['n_bets']})")
            print(f"  Max drawdown: {result['max_drawdown_pct']:.1f}%")
            print(f"  Sharpe ratio: {result['sharpe_ratio']:.3f}")

            # Plot if requested
            if args.plot:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(12, 6))
                plt.plot(result['bankroll_history'], linewidth=2)
                plt.axhline(args.bankroll, color='k', linestyle='--', label='Initial bankroll')
                plt.xlabel('Bet Number')
                plt.ylabel('Bankroll ($)')
                plt.title(f'Kelly Growth Simulation (p={args.win_prob:.2f}, {args.kelly_fraction*100:.0f}% Kelly)')
                plt.legend()
                plt.grid(True, alpha=0.3)

                if args.output:
                    plt.savefig(args.output, dpi=300, bbox_inches='tight')
                    print(f"\nPlot saved to {args.output}")
                else:
                    plt.show()

    else:
        # Single bet calculation
        print(f"\nBet Analysis:")
        print(f"  Model win probability: {args.win_prob:.3f}")
        print(f"  Market odds: {args.odds}")
        print(f"  Bankroll: ${args.bankroll:,.0f}")

        result = optimal_bet_size(
            model_prob=args.win_prob,
            market_odds=args.odds,
            bankroll=args.bankroll,
            kelly_fraction=args.kelly_fraction,
            max_bet_fraction=args.max_bet_fraction,
            min_edge=args.min_edge,
        )

        if result['should_bet']:
            print(f"\n✓ BET RECOMMENDED")
            print(f"  Edge: {result['edge']*100:+.2f}%")
            print(f"  Model prob: {result['model_prob']:.3f}")
            print(f"  Implied prob: {result['implied_prob']:.3f}")
            print(f"  Decimal odds: {result['decimal_odds']:.3f}")
            print(f"  Kelly fraction: {result['bet_fraction']:.4f}")
            print(f"  Bet amount: ${result['bet_amount']:,.0f}")
            print(f"  Expected value: ${result['expected_value']:+,.2f}")
            print(f"  ROI: {result['roi']:+.2f}%")
        else:
            print(f"\n✗ NO BET")
            print(f"  Reason: {result['reason']}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
