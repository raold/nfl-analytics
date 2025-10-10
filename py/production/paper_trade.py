#!/usr/bin/env python3
"""
paper_trade.py

NFL Week 6 (2025) Paper Trading Orchestrator

Simplified wrapper for running the production betting system in paper trading mode
during Week 6 as a trial run before going live in Week 7.

This script:
1. Generates betting recommendations with --paper-trade flag
2. Saves recommendations to paper_trades/ directory
3. Provides easy commands for logging and updating virtual bets
4. Generates paper trading performance reports

Usage:
    # Generate Week 6 recommendations (run Monday after lineups announced)
    python py/production/paper_trade.py recommend --week 6

    # Log a paper trade bet
    python py/production/paper_trade.py log --game-id "2025_06_KC_BUF" ...

    # Update paper trade result after game
    python py/production/paper_trade.py update --bet-id 1 --result win ...

    # Generate paper trading report
    python py/production/paper_trade.py report

    # View all paper trades
    python py/production/paper_trade.py list
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import pandas as pd
from sqlalchemy import create_engine


def run_command(cmd: List[str], description: str = None):
    """
    Run a shell command and handle errors.

    Args:
        cmd: Command and arguments as list
        description: Human-readable description of what command does
    """
    if description:
        print(f"\n{description}...")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        sys.exit(1)

    print(result.stdout)
    return result.stdout


def recommend_bets(week: int, season: int = 2025):
    """
    Generate paper trading bet recommendations for a week.

    Args:
        week: NFL week number
        season: NFL season year
    """
    print("="*80)
    print(f"WEEK {week} PAPER TRADING RECOMMENDATIONS")
    print("="*80)

    # Path to upcoming games CSV (user should prepare this)
    games_csv = f"data/week{week}_games.csv"

    if not Path(games_csv).exists():
        print(f"\nERROR: {games_csv} not found!")
        print("\nYou need to create this file with Week {week} game data.")
        print("Required columns: game_id, home_team, away_team, spread_close, total_close, ...")
        print("\nSee data/upcoming_games_template.csv for format.")
        sys.exit(1)

    # Output path
    output_dir = Path("paper_trades")
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / f"week{week}_recommendations.json"

    # Run betting system with --paper-trade flag
    cmd = [
        sys.executable,
        "py/production/majority_betting_system.py",
        "--games-csv", games_csv,
        "--bankroll", "10000",
        "--kelly-fraction", "0.25",
        "--max-bet-fraction", "0.05",
        "--min-edge", "0.02",
        "--paper-trade",  # PAPER TRADING MODE
        "--output", str(output_path),
    ]

    run_command(cmd, f"Generating Week {week} betting recommendations")

    print(f"\n✅ Recommendations saved to {output_path}")
    print(f"\nNext steps:")
    print(f"1. Review recommendations in {output_path}")
    print(f"2. For each bet you want to place, run:")
    print(f"   python py/production/paper_trade.py log --game-id <game> --bet-type <type> ...")
    print(f"3. After games finish, update results with:")
    print(f"   python py/production/paper_trade.py update --bet-id <id> --result <win/loss> ...")


def log_bet(args: argparse.Namespace):
    """
    Log a paper trade bet to the database.

    Args:
        args: Command line arguments
    """
    cmd = [
        sys.executable,
        "py/production/monitor_performance.py",
        "log",
        "--game-id", args.game_id,
        "--week", str(args.week),
        "--season", str(args.season),
        "--bet-type", args.bet_type,
        "--side", args.side,
        "--line", str(args.line),
        "--odds", str(args.odds),
        "--stake", str(args.stake),
        "--prediction", str(args.prediction),
        "--paper-trade",  # PAPER TRADING MODE
    ]

    run_command(cmd, "Logging paper trade bet")


def update_bet(args: argparse.Namespace):
    """
    Update a paper trade bet result.

    Args:
        args: Command line arguments
    """
    cmd = [
        sys.executable,
        "py/production/monitor_performance.py",
        "update",
        "--bet-id", str(args.bet_id),
        "--result", args.result,
        "--home-score", str(args.home_score),
        "--away-score", str(args.away_score),
    ]

    if args.closing_line:
        cmd.extend(["--closing-line", str(args.closing_line)])

    run_command(cmd, "Updating paper trade bet result")


def list_bets():
    """
    List all paper trade bets.
    """
    engine = create_engine("postgresql://dro:sicillionbillions@localhost:5544/devdb01")

    query = """
        SELECT bet_id, timestamp, game_id, bet_type, side, line, odds, stake,
               prediction, result, payout
        FROM bets
        WHERE is_paper_trade = TRUE
        ORDER BY timestamp DESC
    """

    with engine.connect() as conn:
        bets = pd.read_sql(query, conn)

    if len(bets) == 0:
        print("No paper trade bets found.")
        return

    print("\n" + "="*80)
    print("PAPER TRADE BETS")
    print("="*80)
    print(bets.to_string(index=False))
    print("="*80)


def generate_report():
    """
    Generate paper trading performance report.
    """
    print("="*80)
    print("PAPER TRADING PERFORMANCE REPORT")
    print("="*80)

    engine = create_engine("postgresql://dro:sicillionbillions@localhost:5544/devdb01")

    # Get all paper trade bets
    query = """
        SELECT *
        FROM bets
        WHERE is_paper_trade = TRUE
        ORDER BY timestamp
    """

    with engine.connect() as conn:
        bets = pd.read_sql(query, conn)

    if len(bets) == 0:
        print("\nNo paper trade bets found. Run 'recommend' and 'log' commands first.")
        return

    # Calculate metrics
    settled = bets[bets["result"].notna()]

    print(f"\nBets Placed: {len(bets)}")
    print(f"Bets Settled: {len(settled)}")
    print(f"Bets Pending: {len(bets) - len(settled)}")

    if len(settled) > 0:
        wins = (settled["result"] == "win").sum()
        losses = (settled["result"] == "loss").sum()
        pushes = (settled["result"] == "push").sum()

        total_staked = settled["stake"].sum()
        total_payout = settled["payout"].sum()
        roi = (total_payout / total_staked) if total_staked > 0 else 0

        print(f"\nResults:")
        print(f"  Wins: {wins}")
        print(f"  Losses: {losses}")
        print(f"  Pushes: {pushes}")
        print(f"  Win Rate: {wins/(wins+losses)*100:.1f}%" if (wins+losses) > 0 else "  Win Rate: N/A")

        print(f"\nFinancial:")
        print(f"  Total Staked: ${total_staked:,.2f}")
        print(f"  Total Payout: ${total_payout:+,.2f}")
        print(f"  ROI: {roi:+.2%}")
        print(f"  Final Bankroll: ${10000 + total_payout:,.2f}")

    print("\n" + "="*80)

    # Assessment
    print("\nPaper Trading Assessment:")
    if len(settled) >= 10:
        if total_payout > 0:
            print("✅ Positive ROI - System showing promise")
        else:
            print("⚠️  Negative ROI - Review strategy before going live")

        if wins/(wins+losses) >= 0.53:
            print("✅ Win rate above breakeven (52.4%)")
        else:
            print("⚠️  Win rate below breakeven - Monitor closely")

        print("\nRecommendation for Week 7:")
        if roi > 0.02 and wins/(wins+losses) >= 0.53:
            print("🟢 PROCEED TO LIVE BETTING (start small)")
        elif roi > -0.05:
            print("🟡 REVIEW AND ITERATE (paper trade another week)")
        else:
            print("🔴 DO NOT GO LIVE (significant issues detected)")
    else:
        print("⏳ Need at least 10 settled bets for meaningful assessment")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="NFL Week 6 Paper Trading Orchestrator")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Recommend bets
    rec_parser = subparsers.add_parser("recommend", help="Generate betting recommendations")
    rec_parser.add_argument("--week", type=int, required=True, help="NFL week number")
    rec_parser.add_argument("--season", type=int, default=2025, help="NFL season")

    # Log bet
    log_parser = subparsers.add_parser("log", help="Log a paper trade bet")
    log_parser.add_argument("--game-id", required=True, help="Game ID")
    log_parser.add_argument("--week", type=int, required=True, help="Week number")
    log_parser.add_argument("--season", type=int, default=2025, help="Season")
    log_parser.add_argument("--bet-type", required=True, choices=["spread", "total", "moneyline"])
    log_parser.add_argument("--side", required=True, help="home, away, over, under")
    log_parser.add_argument("--line", type=float, required=True, help="Betting line")
    log_parser.add_argument("--odds", type=int, required=True, help="American odds")
    log_parser.add_argument("--stake", type=float, required=True, help="Bet amount")
    log_parser.add_argument("--prediction", type=float, required=True, help="Model probability")

    # Update bet
    update_parser = subparsers.add_parser("update", help="Update bet result")
    update_parser.add_argument("--bet-id", type=int, required=True)
    update_parser.add_argument("--result", required=True, choices=["win", "loss", "push"])
    update_parser.add_argument("--home-score", type=int, required=True)
    update_parser.add_argument("--away-score", type=int, required=True)
    update_parser.add_argument("--closing-line", type=float, help="Closing line")

    # List bets
    subparsers.add_parser("list", help="List all paper trade bets")

    # Report
    subparsers.add_parser("report", help="Generate paper trading report")

    args = parser.parse_args()

    if args.command == "recommend":
        recommend_bets(args.week, args.season)
    elif args.command == "log":
        log_bet(args)
    elif args.command == "update":
        update_bet(args)
    elif args.command == "list":
        list_bets()
    elif args.command == "report":
        generate_report()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
