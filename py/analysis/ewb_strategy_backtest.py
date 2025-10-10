#!/usr/bin/env python3
"""
Early Week Betting (EWB) Strategy Backtest vs Closing Line Value (CLV).

Compares two betting strategies:
1. Early Week Betting (EWB): Bet Tuesday-Wednesday when lines open
2. Closing Line (CL): Bet Sunday morning before kickoff

Key Questions:
- Does EWB provide edge vs CL?
- How much CLV do we gain/lose by betting early?
- Should we wait for sharp money or bet immediately?

Expected Results:
- Hypothesis: EWB has +0.5-1.5% edge (avoid line movement against us)
- Counter: If model is slow (Tuesday data incomplete), CL may be better
- Answer: Depends on model accuracy at different timepoints

Usage:
    # Backtest EWB vs CL
    python py/analysis/ewb_strategy_backtest.py \
        --lines-csv data/odds/historical_lines_2024.csv \
        --results-csv data/results/games_2024.csv \
        --output results/ewb_backtest_2024.json

    # Compare strategies
    python py/analysis/ewb_strategy_backtest.py \
        --lines-csv data/odds/historical_lines_2024.csv \
        --results-csv data/results/games_2024.csv \
        --strategies ewb cl adaptive

Author: NFL Analytics System
Date: 2025-10-10
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class BettingResult:
    """Result of a single bet."""
    game_id: str
    strategy: str  # 'ewb', 'cl', 'adaptive'
    line: float  # Spread bet at
    result: int  # 1 = win, 0 = loss, 0.5 = push
    payout: float  # Net payout ($)
    clv: float  # Closing line value (CL - bet line)


@dataclass
class StrategyPerformance:
    """Performance metrics for a betting strategy."""
    strategy: str
    total_bets: int
    wins: int
    losses: int
    pushes: int
    win_rate: float
    total_payout: float
    roi: float
    avg_clv: float
    sharpe_ratio: float
    max_drawdown: float


# ============================================================================
# EWB Strategy Backtest
# ============================================================================


class EWBBacktest:
    """
    Backtest Early Week Betting (EWB) vs Closing Line strategies.
    """

    def __init__(self, bet_amount: float = 100.0, odds: int = -110):
        """
        Initialize backtest.

        Args:
            bet_amount: Amount to bet per game ($)
            odds: American odds (default: -110)
        """
        self.bet_amount = bet_amount
        self.odds = odds

        # Calculate payout multiplier
        if odds > 0:
            self.payout_mult = odds / 100.0
        else:
            self.payout_mult = 100.0 / abs(odds)

        self.results: Dict[str, List[BettingResult]] = {
            'ewb': [],
            'cl': [],
            'adaptive': [],
        }

    def load_data(
        self,
        lines_csv: str,
        results_csv: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load historical lines and game results.

        lines_csv format:
            game_id,timestamp,spread,source
            KC_vs_BUF,2024-10-08T09:00:00,-3,opening
            KC_vs_BUF,2024-10-13T12:00:00,-3.5,closing

        results_csv format:
            game_id,home_score,away_score,spread_close
            KC_vs_BUF,27,24,-3.5
        """
        lines_df = pd.read_csv(lines_csv)
        results_df = pd.read_csv(results_csv)

        return lines_df, results_df

    def calculate_bet_result(
        self,
        spread_bet: float,
        spread_close: float,
        home_score: int,
        away_score: int,
        bet_home: bool = True,
    ) -> Tuple[int, float]:
        """
        Calculate bet result.

        Args:
            spread_bet: Spread we bet at
            spread_close: Closing spread (for CLV)
            home_score: Home team score
            away_score: Away team score
            bet_home: True if bet home, False if bet away

        Returns:
            (result, payout)
            result: 1 = win, 0 = loss, 0.5 = push
            payout: Net payout ($)
        """
        # Determine if bet won
        if bet_home:
            # Bet home team
            cover_margin = (home_score - away_score) + spread_bet
        else:
            # Bet away team
            cover_margin = (away_score - home_score) - spread_bet

        # Result
        if cover_margin > 0:
            # Win
            result = 1
            payout = self.bet_amount * self.payout_mult
        elif cover_margin < 0:
            # Loss
            result = 0
            payout = -self.bet_amount
        else:
            # Push
            result = 0.5
            payout = 0

        return result, payout

    def backtest_ewb(
        self,
        lines_df: pd.DataFrame,
        results_df: pd.DataFrame,
    ):
        """
        Backtest Early Week Betting (EWB) strategy.

        Strategy: Bet when opening line is posted (Tuesday-Wednesday)
        """
        # Get opening lines
        opening_lines = lines_df[lines_df['source'] == 'opening'].copy()
        closing_lines = lines_df[lines_df['source'] == 'closing'].copy()

        # Merge with results
        data = opening_lines.merge(
            closing_lines[['game_id', 'spread']],
            on='game_id',
            suffixes=('_open', '_close')
        ).merge(
            results_df,
            on='game_id'
        )

        # Simulate bets
        for _, row in data.iterrows():
            game_id = row['game_id']
            spread_open = row['spread_open']
            spread_close = row['spread_close']
            home_score = row['home_score']
            away_score = row['away_score']

            # Bet at opening line
            result, payout = self.calculate_bet_result(
                spread_bet=spread_open,
                spread_close=spread_close,
                home_score=home_score,
                away_score=away_score,
                bet_home=True,
            )

            # Calculate CLV
            clv = spread_close - spread_open

            self.results['ewb'].append(BettingResult(
                game_id=game_id,
                strategy='ewb',
                line=spread_open,
                result=result,
                payout=payout,
                clv=clv,
            ))

    def backtest_closing_line(
        self,
        lines_df: pd.DataFrame,
        results_df: pd.DataFrame,
    ):
        """
        Backtest Closing Line strategy.

        Strategy: Bet at closing line (Sunday morning)
        """
        # Get closing lines
        closing_lines = lines_df[lines_df['source'] == 'closing'].copy()

        # Merge with results
        data = closing_lines.merge(results_df, on='game_id')

        # Simulate bets
        for _, row in data.iterrows():
            game_id = row['game_id']
            spread_close = row['spread']
            home_score = row['home_score']
            away_score = row['away_score']

            # Bet at closing line
            result, payout = self.calculate_bet_result(
                spread_bet=spread_close,
                spread_close=spread_close,
                home_score=home_score,
                away_score=away_score,
                bet_home=True,
            )

            # CLV = 0 (bet at closing)
            clv = 0

            self.results['cl'].append(BettingResult(
                game_id=game_id,
                strategy='cl',
                line=spread_close,
                result=result,
                payout=payout,
                clv=clv,
            ))

    def backtest_adaptive(
        self,
        lines_df: pd.DataFrame,
        results_df: pd.DataFrame,
        clv_threshold: float = 0.5,
    ):
        """
        Backtest Adaptive strategy.

        Strategy:
        - If opening line moves >0.5 points in our favor → bet early (EWB)
        - Otherwise → wait for closing line
        """
        opening_lines = lines_df[lines_df['source'] == 'opening'].copy()
        closing_lines = lines_df[lines_df['source'] == 'closing'].copy()

        data = opening_lines.merge(
            closing_lines[['game_id', 'spread']],
            on='game_id',
            suffixes=('_open', '_close')
        ).merge(
            results_df,
            on='game_id'
        )

        for _, row in data.iterrows():
            game_id = row['game_id']
            spread_open = row['spread_open']
            spread_close = row['spread_close']
            home_score = row['home_score']
            away_score = row['away_score']

            # Calculate expected CLV
            expected_clv = spread_close - spread_open

            # Decision: bet early if line moves in our favor
            if abs(expected_clv) >= clv_threshold:
                # Bet at opening (EWB)
                bet_line = spread_open
            else:
                # Wait for closing
                bet_line = spread_close

            result, payout = self.calculate_bet_result(
                spread_bet=bet_line,
                spread_close=spread_close,
                home_score=home_score,
                away_score=away_score,
                bet_home=True,
            )

            clv = spread_close - bet_line

            self.results['adaptive'].append(BettingResult(
                game_id=game_id,
                strategy='adaptive',
                line=bet_line,
                result=result,
                payout=payout,
                clv=clv,
            ))

    def calculate_performance(
        self,
        strategy: str,
    ) -> StrategyPerformance:
        """Calculate performance metrics for a strategy."""
        results = self.results[strategy]

        if len(results) == 0:
            return StrategyPerformance(
                strategy=strategy,
                total_bets=0,
                wins=0,
                losses=0,
                pushes=0,
                win_rate=0,
                total_payout=0,
                roi=0,
                avg_clv=0,
                sharpe_ratio=0,
                max_drawdown=0,
            )

        # Count results
        wins = sum(1 for r in results if r.result == 1)
        losses = sum(1 for r in results if r.result == 0)
        pushes = sum(1 for r in results if r.result == 0.5)

        # Payouts
        payouts = [r.payout for r in results]
        total_payout = sum(payouts)
        total_risked = self.bet_amount * len(results)
        roi = (total_payout / total_risked) * 100

        # CLV
        clv_values = [r.clv for r in results]
        avg_clv = np.mean(clv_values)

        # Sharpe ratio
        if len(payouts) > 1 and np.std(payouts) > 0:
            sharpe_ratio = np.mean(payouts) / np.std(payouts)
        else:
            sharpe_ratio = 0

        # Max drawdown
        cumulative = np.cumsum(payouts)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = drawdown.max()

        return StrategyPerformance(
            strategy=strategy,
            total_bets=len(results),
            wins=wins,
            losses=losses,
            pushes=pushes,
            win_rate=(wins / len(results)) * 100,
            total_payout=total_payout,
            roi=roi,
            avg_clv=avg_clv,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
        )

    def compare_strategies(self) -> pd.DataFrame:
        """Compare all strategies."""
        performances = []

        for strategy in ['ewb', 'cl', 'adaptive']:
            perf = self.calculate_performance(strategy)
            performances.append(asdict(perf))

        return pd.DataFrame(performances)

    def statistical_test(
        self,
        strategy1: str,
        strategy2: str,
    ) -> Dict:
        """
        Perform statistical test between two strategies.

        Returns:
            Dict with test results (t-test, p-value)
        """
        payouts1 = [r.payout for r in self.results[strategy1]]
        payouts2 = [r.payout for r in self.results[strategy2]]

        # Paired t-test (same games)
        if len(payouts1) == len(payouts2):
            t_stat, p_value = stats.ttest_rel(payouts1, payouts2)
            test_type = 'paired_ttest'
        else:
            # Independent t-test
            t_stat, p_value = stats.ttest_ind(payouts1, payouts2)
            test_type = 'independent_ttest'

        # Effect size (Cohen's d)
        mean_diff = np.mean(payouts1) - np.mean(payouts2)
        pooled_std = np.sqrt((np.var(payouts1) + np.var(payouts2)) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

        return {
            'strategy1': strategy1,
            'strategy2': strategy2,
            'test_type': test_type,
            't_statistic': t_stat,
            'p_value': p_value,
            'mean_diff': mean_diff,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05,
        }


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description='EWB Strategy Backtest vs Closing Line'
    )

    ap.add_argument('--lines-csv', required=True,
                    help='CSV with historical lines (opening and closing)')
    ap.add_argument('--results-csv', required=True,
                    help='CSV with game results')
    ap.add_argument('--strategies', nargs='+', default=['ewb', 'cl', 'adaptive'],
                    help='Strategies to test')
    ap.add_argument('--bet-amount', type=float, default=100.0,
                    help='Bet amount per game ($)')
    ap.add_argument('--odds', type=int, default=-110,
                    help='American odds')
    ap.add_argument('--output', help='Output JSON path')

    return ap.parse_args()


def main():
    args = parse_args()

    print(f"{'='*80}")
    print(f"Early Week Betting (EWB) Strategy Backtest")
    print(f"{'='*80}")

    # Initialize backtest
    backtest = EWBBacktest(bet_amount=args.bet_amount, odds=args.odds)

    # Load data
    print(f"\nLoading data...")
    print(f"  Lines: {args.lines_csv}")
    print(f"  Results: {args.results_csv}")

    lines_df, results_df = backtest.load_data(args.lines_csv, args.results_csv)

    print(f"  Loaded {len(lines_df)} line snapshots, {len(results_df)} game results")

    # Run backtests
    print(f"\nRunning backtests...")

    if 'ewb' in args.strategies:
        print(f"  Running EWB strategy...")
        backtest.backtest_ewb(lines_df, results_df)

    if 'cl' in args.strategies:
        print(f"  Running Closing Line strategy...")
        backtest.backtest_closing_line(lines_df, results_df)

    if 'adaptive' in args.strategies:
        print(f"  Running Adaptive strategy...")
        backtest.backtest_adaptive(lines_df, results_df)

    # Compare strategies
    print(f"\n{'='*80}")
    print(f"Strategy Comparison")
    print(f"{'='*80}")

    comparison = backtest.compare_strategies()
    print(f"\n{comparison.to_string(index=False)}")

    # Statistical tests
    if 'ewb' in args.strategies and 'cl' in args.strategies:
        print(f"\n{'='*80}")
        print(f"Statistical Test: EWB vs Closing Line")
        print(f"{'='*80}")

        test = backtest.statistical_test('ewb', 'cl')

        print(f"\nTest: {test['test_type']}")
        print(f"  t-statistic: {test['t_statistic']:.3f}")
        print(f"  p-value: {test['p_value']:.4f}")
        print(f"  Mean difference: ${test['mean_diff']:+.2f} per bet")
        print(f"  Cohen's d: {test['cohens_d']:.3f}")
        print(f"  Significant: {'YES' if test['significant'] else 'NO'} (α=0.05)")

        if test['significant']:
            winner = 'EWB' if test['mean_diff'] > 0 else 'Closing Line'
            print(f"\n✅ {winner} strategy is statistically better (p<0.05)")
        else:
            print(f"\n⚠️ No significant difference between strategies")

    # Recommendation
    print(f"\n{'='*80}")
    print(f"Recommendation")
    print(f"{'='*80}")

    best_strategy = comparison.loc[comparison['roi'].idxmax()]

    print(f"\nBest strategy: {best_strategy['strategy'].upper()}")
    print(f"  ROI: {best_strategy['roi']:+.2f}%")
    print(f"  Win rate: {best_strategy['win_rate']:.1f}%")
    print(f"  Avg CLV: {best_strategy['avg_clv']:+.2f} points")
    print(f"  Sharpe ratio: {best_strategy['sharpe_ratio']:.3f}")

    if best_strategy['strategy'] == 'ewb':
        print(f"\n✅ Bet early (Tuesday-Wednesday) when lines open")
        print(f"   Expected edge: {best_strategy['avg_clv']:.2f} points CLV")
    elif best_strategy['strategy'] == 'cl':
        print(f"\n✅ Wait for closing line (Sunday morning)")
        print(f"   Sharp money moves lines against us")
    else:
        print(f"\n✅ Use adaptive strategy (bet early if line moves >0.5 pts)")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            'timestamp': datetime.now().isoformat(),
            'bet_amount': args.bet_amount,
            'odds': args.odds,
            'comparison': comparison.to_dict('records'),
            'statistical_tests': [
                backtest.statistical_test('ewb', 'cl') if 'ewb' in args.strategies and 'cl' in args.strategies else None,
            ],
            'results': {
                strategy: [asdict(r) for r in results]
                for strategy, results in backtest.results.items()
            },
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
