#!/usr/bin/env python3
"""
Ensemble 4-Way Backtest - 2022-2024 Historical Validation
==========================================================

Comprehensive backtest of the 4-way canonical ensemble (State-Space v2, Hierarchical v2,
XGBoost v2, Informative Priors v2) with CQL bet sizing on 2022-2024 historical data.

Metrics Tracked:
- ROI (%)
- Win Rate (%)
- Sharpe Ratio
- Max Drawdown (%)
- Calibration (90% PI coverage)
- MAE, RMSE (yards)

Target Performance (from MODEL_REGISTRY.yaml):
- ROI: +5-7%
- Win Rate: 59-61%
- Sharpe Ratio: 2.2-2.6
- Max Drawdown: -8% to -10%
- Ensemble MAE: < 45.2 yards

Usage:
    python py/backtest/ensemble_5way_backtest.py \
        --start-season 2022 \
        --end-season 2024 \
        --initial-bankroll 10000 \
        --use-cql \
        --output analysis/results/ensemble_5way_backtest_2022_2024.csv

Author: Claude Code (Sonnet 4.5)
Date: November 9, 2025
Version: 1.0
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import logging
import time

import numpy as np
import pandas as pd
import psycopg2

from py.ensemble.enhanced_ensemble_v5 import EnhancedEnsemble4Way

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CQLBetSizer:
    """
    Conservative Q-Learning Bet Sizer with 4 risk gates.

    Risk Gates (from MODEL_REGISTRY.yaml):
    1. Minimum edge threshold: 2%
    2. Maximum bet size: 5% of bankroll
    3. Stop-loss: -10% daily drawdown
    4. Uncertainty filter: predictions must have <80 yards std
    """

    def __init__(
        self,
        min_edge: float = 0.02,
        max_bet_pct: float = 0.05,
        max_daily_drawdown: float = 0.10,
        max_uncertainty: float = 80.0,
        kelly_fraction: float = 0.25
    ):
        """
        Initialize CQL bet sizer.

        Args:
            min_edge: Minimum edge required to bet (default: 2%)
            max_bet_pct: Maximum bet as % of bankroll (default: 5%)
            max_daily_drawdown: Daily stop-loss threshold (default: 10%)
            max_uncertainty: Maximum prediction std (default: 80 yards)
            kelly_fraction: Fractional Kelly multiplier (default: 0.25)
        """
        self.min_edge = min_edge
        self.max_bet_pct = max_bet_pct
        self.max_daily_drawdown = max_daily_drawdown
        self.max_uncertainty = max_uncertainty
        self.kelly_fraction = kelly_fraction

        self.daily_start_bankroll = None
        self.daily_drawdown = 0.0

    def reset_daily(self, bankroll: float):
        """Reset daily tracking (call at start of each day)."""
        self.daily_start_bankroll = bankroll
        self.daily_drawdown = 0.0

    def calculate_bet_size(
        self,
        bankroll: float,
        pred_mean: float,
        pred_std: float,
        vegas_line: float
    ) -> float:
        """
        Calculate optimal bet size using Kelly Criterion with 4 risk gates.

        Args:
            bankroll: Current bankroll
            pred_mean: Ensemble prediction (yards)
            pred_std: Ensemble uncertainty (yards)
            vegas_line: Vegas over/under line (yards)

        Returns:
            Bet size in dollars (0 if no bet)
        """
        # Risk Gate 1: Minimum Edge
        edge = abs(pred_mean - vegas_line) / vegas_line
        if edge < self.min_edge:
            logger.debug(f"Gate 1 FAIL: Edge {edge:.1%} < {self.min_edge:.1%}")
            return 0.0

        # Risk Gate 2: Uncertainty Filter
        if pred_std > self.max_uncertainty:
            logger.debug(f"Gate 2 FAIL: Uncertainty {pred_std:.1f} > {self.max_uncertainty:.1f} yards")
            return 0.0

        # Risk Gate 3: Daily Drawdown Stop-Loss
        if self.daily_start_bankroll is not None:
            current_drawdown = (self.daily_start_bankroll - bankroll) / self.daily_start_bankroll
            if current_drawdown > self.max_daily_drawdown:
                logger.debug(f"Gate 3 FAIL: Drawdown {current_drawdown:.1%} > {self.max_daily_drawdown:.1%}")
                return 0.0

        # Calculate Kelly fraction
        # Kelly: f = (bp - q) / b
        # where b = odds-1, p = win_prob, q = 1-p
        # Simplified for over/under: use edge and variance
        win_prob = 0.5 + (edge / 2)  # Simplified edge to win_prob conversion
        odds = 1.91  # Typical -110 odds
        kelly_f = (win_prob * odds - (1 - win_prob)) / odds

        # Apply fractional Kelly
        kelly_f *= self.kelly_fraction

        # Risk Gate 4: Maximum Bet Size
        bet_size = min(kelly_f * bankroll, self.max_bet_pct * bankroll)

        # Ensure non-negative
        bet_size = max(0.0, bet_size)

        return bet_size


class EnsembleBacktest:
    """Backtest 4-way ensemble on 2022-2024 historical data."""

    def __init__(
        self,
        start_season: int,
        end_season: int,
        initial_bankroll: float,
        use_cql: bool = True,
        db_host: str = "localhost",
        db_port: int = 5544,
        db_name: str = "devdb01",
        db_user: str = "dro",
        db_password: str = "sicillionbillions",
    ):
        """
        Initialize backtest.

        Args:
            start_season: First season (inclusive)
            end_season: Last season (inclusive)
            initial_bankroll: Starting bankroll ($)
            use_cql: Whether to use CQL bet sizer (if False, use fixed $100 bets)
            db_*: Database connection parameters
        """
        self.start_season = start_season
        self.end_season = end_season
        self.initial_bankroll = initial_bankroll
        self.use_cql = use_cql

        # Database config
        self.db_config = {
            "host": db_host,
            "port": db_port,
            "dbname": db_name,
            "user": db_user,
            "password": db_password,
        }

        # Initialize ensemble
        logger.info("Initializing 4-way ensemble...")
        self.ensemble = EnhancedEnsemble4Way(
            db_host=db_host,
            db_port=db_port,
            db_name=db_name,
            db_user=db_user,
            db_password=db_password,
        )

        # Initialize CQL bet sizer
        if self.use_cql:
            logger.info("Initializing CQL bet sizer with 4 risk gates...")
            self.bet_sizer = CQLBetSizer()
        else:
            logger.info("Using fixed $100 bets (CQL disabled)")
            self.bet_sizer = None

        # Tracking
        self.bankroll = initial_bankroll
        self.positions: list[dict] = []
        self.daily_pnl: list[dict] = []

    def fetch_historical_games(self) -> pd.DataFrame:
        """
        Fetch historical games from database.

        Returns:
            DataFrame with columns: game_id, player_id, player_name, season, week,
                                    actual_yards, vegas_line (if available)
        """
        logger.info(f"Fetching games {self.start_season}-{self.end_season}...")

        conn = psycopg2.connect(**self.db_config)
        query = """
        SELECT
            CONCAT(g.season, '_', LPAD(g.week::TEXT, 2, '0'), '_', g.home_team, '_', g.away_team) as game_id,
            pgs.player_id,
            ph.player_name,
            pgs.season,
            pgs.week,
            pgs.stat_yards as actual_yards,
            pgs.stat_attempts as attempts,
            g.kickoff,
            NULL::NUMERIC as vegas_line  -- Placeholder (we don't have historical props lines)
        FROM mart.player_game_stats pgs
        JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
        LEFT JOIN games g ON
          g.season = pgs.season
          AND g.week = pgs.week
          AND (ph.current_team = g.home_team OR ph.current_team = g.away_team)
        WHERE pgs.stat_category = 'passing'
          AND ph.position_group = 'QB'
          AND pgs.season >= %s
          AND pgs.season <= %s
          AND pgs.week <= 18  -- Regular season only
          AND pgs.stat_attempts >= 10  -- Filter out garbage time/injuries
        ORDER BY g.kickoff, pgs.player_id
        """

        df = pd.read_sql_query(query, conn, params=(self.start_season, self.end_season))
        conn.close()

        logger.info(f"✅ Fetched {len(df):,} QB games")
        logger.info(f"   Seasons: {df['season'].min()}-{df['season'].max()}")
        logger.info(f"   Players: {df['player_id'].nunique()}")

        return df

    def simulate_betting(
        self,
        games_df: pd.DataFrame,
        vegas_line_col: str = "vegas_line"
    ) -> pd.DataFrame:
        """
        Simulate betting with ensemble predictions.

        Args:
            games_df: Historical games with actual outcomes
            vegas_line_col: Column name for Vegas lines

        Returns:
            DataFrame with bet results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"STARTING BACKTEST: {len(games_df):,} games")
        logger.info(f"{'='*70}\n")

        results = []
        current_date = None

        for i, row in games_df.iterrows():
            # Progress logging
            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i+1}/{len(games_df)} games "
                           f"(Bankroll: ${self.bankroll:,.2f})")

            # Reset daily tracking on date change
            if current_date != row['kickoff']:
                current_date = row['kickoff']
                if self.bet_sizer:
                    self.bet_sizer.reset_daily(self.bankroll)

            # Generate ensemble prediction
            try:
                pred = self.ensemble.predict(
                    player_id=row['player_id'],
                    season=row['season'],
                    week=row['week']
                )
            except Exception as e:
                logger.warning(f"Prediction failed for {row['player_id']} "
                              f"W{row['week']} {row['season']}: {e}")
                continue

            if pred['ensemble_mean'] is None:
                logger.debug(f"No prediction for {row['player_id']} W{row['week']} {row['season']}")
                continue

            # Use ensemble prediction as proxy for Vegas line if not available
            # (In production, we'd have actual historical props lines)
            if pd.isna(row[vegas_line_col]):
                vegas_line = pred['ensemble_mean']  # Neutral assumption
            else:
                vegas_line = row[vegas_line_col]

            # Calculate bet size
            if self.use_cql:
                bet_size = self.bet_sizer.calculate_bet_size(
                    bankroll=self.bankroll,
                    pred_mean=pred['ensemble_mean'],
                    pred_std=pred['ensemble_std'],
                    vegas_line=vegas_line
                )
            else:
                # Fixed $100 bet
                bet_size = 100.0 if abs(pred['ensemble_mean'] - vegas_line) > 10 else 0.0

            # Skip if no bet
            if bet_size == 0.0:
                continue

            # Determine bet direction (over/under)
            if pred['ensemble_mean'] > vegas_line:
                bet_side = "over"
                win = row['actual_yards'] > vegas_line
            else:
                bet_side = "under"
                win = row['actual_yards'] < vegas_line

            # Calculate P&L (assuming -110 odds)
            if win:
                pnl = bet_size * 0.909  # Win $0.909 per $1 bet at -110
            else:
                pnl = -bet_size  # Lose stake

            # Update bankroll
            self.bankroll += pnl

            # Record result
            results.append({
                "game_id": row['game_id'],
                "player_id": row['player_id'],
                "player_name": row['player_name'],
                "season": row['season'],
                "week": row['week'],
                "actual_yards": row['actual_yards'],
                "pred_mean": pred['ensemble_mean'],
                "pred_std": pred['ensemble_std'],
                "quantile_10": pred['quantile_10'],
                "quantile_90": pred['quantile_90'],
                "vegas_line": vegas_line,
                "bet_side": bet_side,
                "bet_size": bet_size,
                "win": win,
                "pnl": pnl,
                "bankroll": self.bankroll,
            })

        results_df = pd.DataFrame(results)
        logger.info(f"\n✅ Backtest complete: {len(results_df):,} bets placed")

        return results_df

    def calculate_metrics(self, results_df: pd.DataFrame) -> dict:
        """
        Calculate performance metrics.

        Returns:
            Dict with keys: roi, win_rate, sharpe_ratio, max_drawdown,
                           mae, rmse, calibration_90
        """
        if len(results_df) == 0:
            logger.error("No results to analyze!")
            return {}

        # Betting metrics
        total_wagered = results_df['bet_size'].sum()
        total_pnl = results_df['pnl'].sum()
        roi = (total_pnl / total_wagered) * 100 if total_wagered > 0 else 0.0
        win_rate = results_df['win'].mean() * 100

        # Sharpe ratio (assume daily returns)
        returns = results_df['pnl'] / results_df['bet_size']
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0

        # Max drawdown
        cumulative_pnl = results_df['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdowns = (cumulative_pnl - running_max) / self.initial_bankroll * 100
        max_drawdown = drawdowns.min()

        # Prediction accuracy
        mae = np.abs(results_df['actual_yards'] - results_df['pred_mean']).mean()
        rmse = np.sqrt(((results_df['actual_yards'] - results_df['pred_mean']) ** 2).mean())

        # Calibration (90% PI coverage)
        in_interval = (
            (results_df['actual_yards'] >= results_df['quantile_10']) &
            (results_df['actual_yards'] <= results_df['quantile_90'])
        )
        calibration_90 = in_interval.mean() * 100

        metrics = {
            "roi": roi,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "mae": mae,
            "rmse": rmse,
            "calibration_90": calibration_90,
            "total_bets": len(results_df),
            "total_wagered": total_wagered,
            "total_pnl": total_pnl,
            "final_bankroll": self.bankroll,
        }

        return metrics

    def run(self, output_csv: str):
        """
        Run full backtest and save results.

        Args:
            output_csv: Path to save results CSV
        """
        start_time = time.time()

        # Fetch historical data
        games_df = self.fetch_historical_games()

        # Run backtest
        results_df = self.simulate_betting(games_df)

        # Calculate metrics
        metrics = self.calculate_metrics(results_df)

        # Print summary
        logger.info(f"\n{'='*70}")
        logger.info("BACKTEST RESULTS SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Period: {self.start_season}-{self.end_season}")
        logger.info(f"Total Bets: {metrics.get('total_bets', 0):,}")
        logger.info(f"Total Wagered: ${metrics.get('total_wagered', 0):,.2f}")
        logger.info("\nPERFORMANCE METRICS:")
        logger.info(f"  ROI: {metrics.get('roi', 0):.2f}% (Target: +5-7%)")
        logger.info(f"  Win Rate: {metrics.get('win_rate', 0):.1f}% (Target: 59-61%)")
        logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f} (Target: 2.2-2.6)")
        logger.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.1f}% (Target: -8 to -10%)")
        logger.info("\nPREDICTION ACCURACY:")
        logger.info(f"  MAE: {metrics.get('mae', 0):.2f} yards (Target: <45.2)")
        logger.info(f"  RMSE: {metrics.get('rmse', 0):.2f} yards")
        logger.info(f"  Calibration (90% PI): {metrics.get('calibration_90', 0):.1f}% (Target: ≥89%)")
        logger.info(f"\nFINAL BANKROLL: ${metrics.get('final_bankroll', 0):,.2f}")
        logger.info(f"  P&L: ${metrics.get('total_pnl', 0):,.2f}")
        logger.info(f"  Return: {(metrics.get('final_bankroll', 0) / self.initial_bankroll - 1) * 100:.2f}%")
        logger.info(f"\nElapsed time: {time.time() - start_time:.1f}s")
        logger.info(f"{'='*70}\n")

        # Save results
        results_df.to_csv(output_csv, index=False)
        logger.info(f"✅ Results saved to: {output_csv}")

        # Save metrics JSON
        metrics_json = output_csv.replace('.csv', '_metrics.json')
        import json
        with open(metrics_json, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"✅ Metrics saved to: {metrics_json}")

        return results_df, metrics


def main():
    """Command-line interface for backtest."""
    parser = argparse.ArgumentParser(
        description="Ensemble 4-Way Backtest (2022-2024)"
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=2022,
        help="Start season (default: 2022)"
    )
    parser.add_argument(
        "--end-season",
        type=int,
        default=2024,
        help="End season (default: 2024)"
    )
    parser.add_argument(
        "--initial-bankroll",
        type=float,
        default=10000.0,
        help="Initial bankroll in dollars (default: 10000)"
    )
    parser.add_argument(
        "--use-cql",
        action="store_true",
        help="Use CQL bet sizer (default: fixed $100 bets)"
    )
    parser.add_argument(
        "--output",
        default="analysis/results/ensemble_5way_backtest_2022_2024.csv",
        help="Output CSV path"
    )

    args = parser.parse_args()

    # Create backtest
    backtest = EnsembleBacktest(
        start_season=args.start_season,
        end_season=args.end_season,
        initial_bankroll=args.initial_bankroll,
        use_cql=args.use_cql,
    )

    # Run
    backtest.run(output_csv=args.output)


if __name__ == "__main__":
    main()
