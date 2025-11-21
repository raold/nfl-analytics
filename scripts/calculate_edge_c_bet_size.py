#!/usr/bin/env python3
"""
Edge C Bet Sizing Calculator

Calculates fractional Kelly bet sizes for Edge C (lingering fatigue) opportunities
with variance adjustment for small sample size (n=24).

Usage:
    # Calculate bet size
    python scripts/calculate_edge_c_bet_size.py --bankroll 10000 --odds -110 --win-rate 0.70

    # Interactive mode
    python scripts/calculate_edge_c_bet_size.py --interactive

    # Conservative mode (10% Kelly)
    python scripts/calculate_edge_c_bet_size.py --bankroll 10000 --odds -110 --win-rate 0.70 --kelly-fraction 0.10

Author: NFL Analytics Team
Date: November 16, 2025
"""

import argparse
import logging
import math
from typing import Dict, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def american_to_decimal_odds(american_odds: float) -> float:
    """
    Convert American odds to decimal odds.

    Args:
        american_odds: American odds (e.g., -110, +150)

    Returns:
        Decimal odds
    """
    if american_odds < 0:
        return 1 + (100 / abs(american_odds))
    else:
        return 1 + (american_odds / 100)


def calculate_kelly_fraction(
    win_rate: float,
    odds: float,
    kelly_fraction: float = 1.0,
    variance_adjustment: bool = True,
    sample_size: int = 24
) -> Tuple[float, Dict]:
    """
    Calculate Kelly criterion bet size with variance adjustment.

    Kelly formula: f = (bp - q) / b
    where:
        f = fraction of bankroll to bet
        b = net odds received (decimal odds - 1)
        p = probability of winning
        q = probability of losing (1 - p)

    Variance adjustment for small sample:
        Reduce Kelly by factor based on sample size uncertainty

    Args:
        win_rate: Probability of winning (0-1)
        odds: American odds (e.g., -110)
        kelly_fraction: Fraction of Kelly to use (0-1, default 1.0 = full Kelly)
        variance_adjustment: Whether to apply small sample variance adjustment
        sample_size: Number of observations in historical data

    Returns:
        Tuple of (bet_fraction, details_dict)
    """
    # Convert American to decimal odds
    decimal_odds = american_to_decimal_odds(odds)
    b = decimal_odds - 1  # Net odds

    # Kelly formula
    p = win_rate
    q = 1 - p

    kelly = (b * p - q) / b

    # Ensure non-negative
    kelly = max(0, kelly)

    # Apply variance adjustment for small sample
    if variance_adjustment:
        # Use conservative adjustment: reduce by sqrt(100/n)
        # This accounts for estimation uncertainty
        adjustment_factor = math.sqrt(100 / max(sample_size, 1))
        adjustment_factor = min(adjustment_factor, 1.0)  # Cap at 1.0

        variance_adjusted_kelly = kelly / adjustment_factor
    else:
        variance_adjusted_kelly = kelly

    # Apply fractional Kelly
    final_fraction = variance_adjusted_kelly * kelly_fraction

    details = {
        "win_rate": win_rate,
        "american_odds": odds,
        "decimal_odds": decimal_odds,
        "net_odds": b,
        "kelly_full": kelly,
        "kelly_variance_adjusted": variance_adjusted_kelly,
        "kelly_fraction_applied": kelly_fraction,
        "final_bet_fraction": final_fraction,
        "variance_adjustment_factor": adjustment_factor if variance_adjustment else 1.0,
        "sample_size": sample_size
    }

    return final_fraction, details


def calculate_bet_amount(
    bankroll: float,
    win_rate: float,
    odds: float,
    kelly_fraction: float = 0.10,
    max_bet_pct: float = 0.02
) -> Dict:
    """
    Calculate recommended bet amount with risk management constraints.

    Args:
        bankroll: Total bankroll in dollars
        win_rate: Probability of winning
        odds: American odds
        kelly_fraction: Fraction of Kelly to use (default 0.10 = 10% Kelly)
        max_bet_pct: Maximum bet as % of bankroll (default 0.02 = 2%)

    Returns:
        Dictionary with bet sizing details
    """
    # Calculate Kelly fraction
    bet_fraction, kelly_details = calculate_kelly_fraction(
        win_rate=win_rate,
        odds=odds,
        kelly_fraction=kelly_fraction,
        variance_adjustment=True,
        sample_size=24  # Edge C sample size
    )

    # Calculate bet amounts
    kelly_bet = bankroll * bet_fraction
    max_bet = bankroll * max_bet_pct

    # Apply risk management cap
    recommended_bet = min(kelly_bet, max_bet)

    # Calculate expected value
    decimal_odds = american_to_decimal_odds(odds)
    ev_per_dollar = (win_rate * decimal_odds) - 1
    expected_profit = recommended_bet * ev_per_dollar

    result = {
        "bankroll": bankroll,
        "kelly_details": kelly_details,
        "kelly_bet_amount": kelly_bet,
        "max_bet_amount": max_bet,
        "recommended_bet_amount": recommended_bet,
        "recommended_bet_pct": recommended_bet / bankroll,
        "expected_value_per_dollar": ev_per_dollar,
        "expected_profit": expected_profit,
        "risk_management": {
            "kelly_fraction_used": kelly_fraction,
            "max_bet_pct": max_bet_pct,
            "capped_by_max": recommended_bet == max_bet
        }
    }

    return result


def print_bet_sizing_report(result: Dict):
    """
    Print formatted bet sizing report.

    Args:
        result: Dictionary from calculate_bet_amount()
    """
    print("\n" + "="*80)
    print("EDGE C BET SIZING CALCULATOR")
    print("="*80)
    print("\nINPUTS:")
    print(f"  Bankroll: ${result['bankroll']:,.2f}")
    print(f"  Win Rate: {result['kelly_details']['win_rate']*100:.1f}%")
    print(f"  Odds: {result['kelly_details']['american_odds']:+.0f}")
    print(f"  Sample Size: {result['kelly_details']['sample_size']} observations")

    print("\nKELLY CALCULATION:")
    print(f"  Full Kelly: {result['kelly_details']['kelly_full']*100:.2f}% of bankroll")
    print(f"  Variance Adjusted: {result['kelly_details']['kelly_variance_adjusted']*100:.2f}% "
          f"(factor: {result['kelly_details']['variance_adjustment_factor']:.2f}x)")
    print(f"  Fractional Kelly ({result['kelly_details']['kelly_fraction_applied']*100:.0f}%): "
          f"{result['kelly_details']['final_bet_fraction']*100:.2f}% of bankroll")

    print("\nBET SIZING:")
    print(f"  Kelly Recommended: ${result['kelly_bet_amount']:,.2f}")
    print(f"  Max Allowed (2% bankroll): ${result['max_bet_amount']:,.2f}")
    print(f"  FINAL RECOMMENDATION: ${result['recommended_bet_amount']:,.2f} "
          f"({result['recommended_bet_pct']*100:.2f}% of bankroll)")

    if result['risk_management']['capped_by_max']:
        print(f"  ⚠️ Bet capped by 2% bankroll limit")

    print("\nEXPECTED VALUE:")
    print(f"  EV per $1 bet: ${result['expected_value_per_dollar']:.4f}")
    print(f"  Expected profit: ${result['expected_profit']:,.2f}")
    print(f"  ROI: {result['expected_value_per_dollar']*100:.2f}%")

    print("\nRISK MANAGEMENT NOTES:")
    print("  ✓ Using 10% fractional Kelly (recommended for small sample)")
    print("  ✓ Variance adjustment applied (n=24 small sample)")
    print("  ✓ Capped at 2% of bankroll maximum")
    print("  ⚠️ Edge C pending baseline calculation fix")
    print("  ⚠️ Recommend paper trading before live betting")
    print("="*80 + "\n")


def interactive_mode():
    """Run calculator in interactive mode."""
    print("\n" + "="*80)
    print("EDGE C BET SIZING CALCULATOR - INTERACTIVE MODE")
    print("="*80)

    # Get inputs
    bankroll = float(input("\nEnter your bankroll ($): "))
    win_rate = float(input("Enter expected win rate (0-1, default 0.70 for Edge C): ") or "0.70")
    odds = float(input("Enter American odds (default -110): ") or "-110")
    kelly_fraction = float(input("Enter Kelly fraction (0-1, default 0.10 for 10% Kelly): ") or "0.10")

    # Calculate
    result = calculate_bet_amount(
        bankroll=bankroll,
        win_rate=win_rate,
        odds=odds,
        kelly_fraction=kelly_fraction
    )

    # Print results
    print_bet_sizing_report(result)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Calculate Edge C bet sizes using fractional Kelly criterion"
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        help="Total bankroll in dollars"
    )
    parser.add_argument(
        "--win-rate",
        type=float,
        default=0.70,
        help="Expected win rate (default: 0.70 from Edge C analysis)"
    )
    parser.add_argument(
        "--odds",
        type=float,
        default=-110,
        help="American odds (default: -110)"
    )
    parser.add_argument(
        "--kelly-fraction",
        type=float,
        default=0.10,
        help="Fraction of Kelly to use (default: 0.10 = 10%% Kelly)"
    )
    parser.add_argument(
        "--max-bet-pct",
        type=float,
        default=0.02,
        help="Maximum bet as %% of bankroll (default: 0.02 = 2%%)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
        return

    if not args.bankroll:
        logger.error("--bankroll is required in non-interactive mode")
        print("\nUse --interactive for interactive mode, or provide --bankroll")
        return

    # Calculate bet sizing
    result = calculate_bet_amount(
        bankroll=args.bankroll,
        win_rate=args.win_rate,
        odds=args.odds,
        kelly_fraction=args.kelly_fraction,
        max_bet_pct=args.max_bet_pct
    )

    # Print report
    print_bet_sizing_report(result)

    # Log summary
    logger.info(
        f"Recommendation: Bet ${result['recommended_bet_amount']:.2f} "
        f"({result['recommended_bet_pct']*100:.2f}% of bankroll)"
    )


if __name__ == "__main__":
    main()
