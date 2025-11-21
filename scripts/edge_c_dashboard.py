#!/usr/bin/env python3
"""
Edge C Validation Dashboard

Displays current Edge C statistics, upcoming opportunities, and paper trade performance.

Usage:
    python scripts/edge_c_dashboard.py
    python scripts/edge_c_dashboard.py --update  # Refresh from database
    python scripts/edge_c_dashboard.py --paper-trades results/edge_c_paper_trades.csv

Author: NFL Analytics Team
Date: November 16, 2025
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Edge C validated statistics (from Phase 0 analysis)
EDGE_C_STATS = {
    "edge_name": "Lingering Fatigue (Edge C)",
    "sample_size": 24,
    "short_rest_sample": 14,
    "avg_penalty_yards": -209.8,
    "short_rest_penalty": -195.6,
    "std_penalty_yards": 74.67,
    "p_value": 0.0,
    "t_statistic": -13.766,
    "estimated_win_rate": 0.70,
    "estimated_roi_pct": 33.6,
    "status": "PENDING BASELINE FIX",
    "last_validated": "2025-11-16"
}


def load_paper_trades(csv_path: str) -> pd.DataFrame:
    """
    Load paper trade results from CSV.

    Args:
        csv_path: Path to paper trades CSV

    Returns:
        DataFrame with paper trades
    """
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} paper trades from {csv_path}")
        return df
    except FileNotFoundError:
        logger.warning(f"Paper trades file not found: {csv_path}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading paper trades: {e}")
        return pd.DataFrame()


def calculate_paper_trade_performance(df: pd.DataFrame) -> Dict:
    """
    Calculate paper trade performance metrics.

    Args:
        df: DataFrame with paper trades

    Returns:
        Dictionary with performance metrics
    """
    if df.empty or len(df) == 0:
        return {
            "total_trades": 0,
            "completed_trades": 0,
            "win_rate": None,
            "total_profit": 0,
            "avg_roi": None,
            "best_trade": None,
            "worst_trade": None
        }

    # Filter to completed trades
    completed = df[df['outcome'].notna()].copy()

    if len(completed) == 0:
        return {
            "total_trades": len(df),
            "completed_trades": 0,
            "win_rate": None,
            "total_profit": 0,
            "avg_roi": None,
            "best_trade": None,
            "worst_trade": None
        }

    # Calculate metrics
    wins = len(completed[completed['outcome'] == 'WIN'])
    win_rate = wins / len(completed) if len(completed) > 0 else 0

    total_profit = completed['profit_loss'].sum()
    avg_roi = completed['roi'].mean() if len(completed) > 0 else 0

    best_trade = completed.loc[completed['profit_loss'].idxmax()].to_dict() if len(completed) > 0 else None
    worst_trade = completed.loc[completed['profit_loss'].idxmin()].to_dict() if len(completed) > 0 else None

    return {
        "total_trades": len(df),
        "completed_trades": len(completed),
        "win_rate": win_rate,
        "total_profit": total_profit,
        "avg_roi": avg_roi,
        "best_trade": best_trade,
        "worst_trade": worst_trade
    }


def load_upcoming_opportunities(alerts_path: str = "results/edge_c_alerts.json") -> List[Dict]:
    """
    Load upcoming Edge C opportunities.

    Args:
        alerts_path: Path to alerts JSON file

    Returns:
        List of opportunities
    """
    try:
        with open(alerts_path, 'r') as f:
            data = json.load(f)
            return data.get('opportunities', [])
    except FileNotFoundError:
        logger.warning(f"Alerts file not found: {alerts_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading alerts: {e}")
        return []


def print_dashboard(
    edge_stats: Dict,
    upcoming_opps: List[Dict],
    paper_perf: Dict
):
    """
    Print formatted dashboard.

    Args:
        edge_stats: Edge C statistics
        upcoming_opps: List of upcoming opportunities
        paper_perf: Paper trade performance metrics
    """
    print("\n" + "="*100)
    print(" "*35 + "EDGE C VALIDATION DASHBOARD")
    print("="*100)

    # Section 1: Edge Statistics
    print("\n" + "─"*100)
    print("EDGE C STATISTICS (Lingering Fatigue)")
    print("─"*100)
    print(f"Status: {edge_stats['status']}")
    print(f"Last Validated: {edge_stats['last_validated']}")
    print(f"\nSample Size: {edge_stats['sample_size']} QB performances "
          f"({edge_stats['short_rest_sample']} with <7 day rest)")
    print(f"\nPenalty Effect:")
    print(f"  All Rest: {edge_stats['avg_penalty_yards']:.1f} ± {edge_stats['std_penalty_yards']:.1f} yards")
    print(f"  Short Rest (<7 days): {edge_stats['short_rest_penalty']:.1f} yards")
    print(f"\nStatistical Significance:")
    print(f"  p-value: {edge_stats['p_value']:.6f} (p < 0.0001)")
    print(f"  t-statistic: {edge_stats['t_statistic']:.3f}")
    print(f"\nExpected Performance:")
    print(f"  Win Rate: {edge_stats['estimated_win_rate']*100:.1f}%")
    print(f"  ROI: {edge_stats['estimated_roi_pct']:.1f}%")

    # Section 2: Upcoming Opportunities
    print("\n" + "─"*100)
    print(f"UPCOMING OPPORTUNITIES (Next 4 Weeks)")
    print("─"*100)

    if not upcoming_opps:
        print("No opportunities currently scheduled.")
        print("Run: python scripts/monitor_international_schedule.py")
    else:
        print(f"Found {len(upcoming_opps)} opportunities:\n")
        for i, opp in enumerate(upcoming_opps, 1):
            print(f"{i}. {opp['qb_name']} ({opp['team']}) - Week {opp['next_week']}")
            print(f"   International: Week {opp['intl_week']} in {opp['intl_location']}")
            print(f"   Next Game: {opp['next_game_date']} vs {opp['opponent']} ({opp['days_since_intl']} days)")
            print(f"   Edge Strength: {opp['edge_strength']}")
            print(f"   Expected Penalty: {opp['expected_penalty_yards']:.1f} yards")
            print()

    # Section 3: Paper Trade Performance
    print("─"*100)
    print("PAPER TRADE PERFORMANCE")
    print("─"*100)

    if paper_perf['total_trades'] == 0:
        print("No paper trades recorded yet.")
        print("Start tracking in: results/edge_c_paper_trades.csv")
    elif paper_perf['completed_trades'] == 0:
        print(f"Total Trades: {paper_perf['total_trades']}")
        print("Completed Trades: 0 (all pending)")
    else:
        print(f"Total Trades: {paper_perf['total_trades']}")
        print(f"Completed Trades: {paper_perf['completed_trades']}")
        print(f"\nPerformance:")
        print(f"  Win Rate: {paper_perf['win_rate']*100:.1f}%")
        print(f"  Total Profit/Loss: ${paper_perf['total_profit']:,.2f}")
        print(f"  Average ROI: {paper_perf['avg_roi']*100:.1f}%")

        if paper_perf['best_trade']:
            print(f"\nBest Trade:")
            print(f"  {paper_perf['best_trade']['qb']} on {paper_perf['best_trade']['date']}")
            print(f"  Profit: ${paper_perf['best_trade']['profit_loss']:,.2f} "
                  f"({paper_perf['best_trade']['roi']*100:.1f}% ROI)")

        if paper_perf['worst_trade']:
            print(f"\nWorst Trade:")
            print(f"  {paper_perf['worst_trade']['qb']} on {paper_perf['worst_trade']['date']}")
            print(f"  Loss: ${paper_perf['worst_trade']['profit_loss']:,.2f} "
                  f"({paper_perf['worst_trade']['roi']*100:.1f}% ROI)")

        # Validation against expected
        if paper_perf['completed_trades'] >= 5:
            expected_wr = EDGE_C_STATS['estimated_win_rate']
            expected_roi = EDGE_C_STATS['estimated_roi_pct']

            wr_diff = paper_perf['win_rate'] - expected_wr
            roi_diff = paper_perf['avg_roi'] - (expected_roi/100)

            print(f"\nValidation vs Expected:")
            print(f"  Win Rate: {paper_perf['win_rate']*100:.1f}% vs {expected_wr*100:.1f}% expected "
                  f"({wr_diff*100:+.1f} pp)")
            print(f"  ROI: {paper_perf['avg_roi']*100:.1f}% vs {expected_roi:.1f}% expected "
                  f"({roi_diff*100:+.1f} pp)")

            if paper_perf['win_rate'] < 0.52:
                print(f"\n  ⚠️ WARNING: Win rate below breakeven (52.4%). Edge may not be viable.")
            elif paper_perf['avg_roi'] < 0:
                print(f"\n  ⚠️ WARNING: Negative ROI. Edge may not be viable.")
            elif paper_perf['completed_trades'] >= 10:
                print(f"\n  ✓ Sample size sufficient ({paper_perf['completed_trades']} trades). "
                      f"Ready for live betting evaluation.")

    # Section 4: Risk Management Status
    print("\n" + "─"*100)
    print("RISK MANAGEMENT STATUS")
    print("─"*100)
    print("Baseline Calculation Fix: PENDING")
    print("  ⚠️ Current baseline uses incorrect current_team filter")
    print("  ⚠️ Must fix before live betting")
    print("  ✓ Paper trading can proceed with awareness of baseline uncertainty")
    print("\nLive Betting Criteria (ALL must be met):")
    print(f"  [ ] Baseline calculation fixed and validated")
    print(f"  [ ] Paper trade sample ≥10 trades")
    print(f"  [ ] Paper trade win rate ≥60%")
    print(f"  [ ] Paper trade ROI ≥+10%")
    print(f"  [ ] Statistical significance maintained (p < 0.05)")
    print("\nBet Sizing (when live):")
    print(f"  • Use fractional Kelly: 10% recommended (small sample)")
    print(f"  • Max 2% bankroll per bet")
    print(f"  • Max 3 bets per week")
    print("\nExit Criteria:")
    print(f"  • If ROI drops below 0% after 20 live bets → abandon edge")
    print(f"  • If win rate drops below 55% after 20 bets → reduce Kelly to 5%")

    print("\n" + "="*100 + "\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Edge C validation dashboard"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update upcoming opportunities from database"
    )
    parser.add_argument(
        "--paper-trades",
        type=str,
        default="results/edge_c_paper_trades.csv",
        help="Path to paper trades CSV"
    )
    parser.add_argument(
        "--alerts",
        type=str,
        default="results/edge_c_alerts.json",
        help="Path to alerts JSON"
    )

    args = parser.parse_args()

    # Update opportunities if requested
    if args.update:
        logger.info("Updating upcoming opportunities...")
        import subprocess
        subprocess.run([
            "python",
            "scripts/monitor_international_schedule.py",
            "--output",
            args.alerts
        ])

    # Load data
    upcoming_opps = load_upcoming_opportunities(args.alerts)
    paper_trades_df = load_paper_trades(args.paper_trades)
    paper_perf = calculate_paper_trade_performance(paper_trades_df)

    # Display dashboard
    print_dashboard(
        edge_stats=EDGE_C_STATS,
        upcoming_opps=upcoming_opps,
        paper_perf=paper_perf
    )


if __name__ == "__main__":
    main()
