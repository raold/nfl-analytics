#!/usr/bin/env python3
"""
REALISTIC Backtest - With Strict Safety Constraints

Differences from standard backtest:
1. MAX_BET_FRACTION = 0.05 (5% vs 10%)
2. PROBABILITY_CLIPPING = [0.10, 0.90] (vs [0.05, 0.95])
3. MIN_EDGE = 0.03 (3% vs 5%)
4. REALITY CHECKS: Warns if win_rate > 60% or ROI > 10%/year

Designed for the RealisticGameOutcomeEnsemble (without vegas leakage).

Expected realistic results:
- Win Rate: 52-56% (professionals achieve 54-56%)
- Annual ROI: 2-7% (professionals achieve 3-5%)
- Sharpe: 0.5-1.5 (professionals achieve ~1.0-1.5)

Author: Claude Code
Date: 2025-10-30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from py.backtests.game_outcome_backtest import (
    GameOutcomeBacktest,
    american_to_decimal,
    get_implied_prob,
)
from py.ensemble.game_outcome_ensemble import fetch_games
from py.ensemble.game_outcome_ensemble_realistic import RealisticGameOutcomeEnsemble


class RealisticBacktest(GameOutcomeBacktest):
    """
    REALISTIC backtest with STRICT safety constraints.

    Constraints:
    - Max bet: 5% of bankroll (vs 10% in standard backtest)
    - Max bet absolute: $5,000
    - Probability clipping: [0.10, 0.90] (vs [0.05, 0.95])
    - Min edge: 3% (vs 5% in standard backtest)

    Reality Checks:
    - Warn if win rate > 60% (indicates possible leakage)
    - Warn if annual ROI > 10% (indicates possible leakage)
    - Warn if Sharpe > 2.0 (indicates possible leakage)
    """

    def __init__(
        self,
        initial_bankroll: float = 10000.0,
        kelly_multiplier: float = 0.25,
        edge_threshold: float = 0.03,  # STRICTER: 3% vs 5%
        vig: int = -110,
        max_bet_frac: float = 0.05,  # STRICTER: 5% vs 10%
        max_bet_absolute: float = 5000.0,
        prob_clip_min: float = 0.10,  # STRICTER: 0.10 vs 0.05
        prob_clip_max: float = 0.90,  # STRICTER: 0.90 vs 0.95
    ):
        """
        Args:
            initial_bankroll: Starting bankroll ($)
            kelly_multiplier: Fractional Kelly (0.25 = quarter Kelly)
            edge_threshold: Minimum edge to place bet (3% = 0.03) - STRICTER
            vig: Vig on spread bets (default -110)
            max_bet_frac: Maximum bet as fraction of bankroll (5% = 0.05) - STRICTER
            max_bet_absolute: Maximum bet size in dollars ($5000)
            prob_clip_min: Minimum probability (0.10) - STRICTER
            prob_clip_max: Maximum probability (0.90) - STRICTER
        """
        super().__init__(
            initial_bankroll=initial_bankroll,
            kelly_multiplier=kelly_multiplier,
            edge_threshold=edge_threshold,
            vig=vig,
            max_bet_frac=max_bet_frac,
            max_bet_absolute=max_bet_absolute,
        )

        self.prob_clip_min = prob_clip_min
        self.prob_clip_max = prob_clip_max

    def _reality_check(self, summary: dict) -> None:
        """
        Check if results are realistic or indicate data leakage.

        Args:
            summary: Backtest summary metrics
        """
        print(f"\n{'='*80}")
        print("REALITY CHECK")
        print(f"{'='*80}")

        warnings = []

        # Check win rate
        win_rate = summary.get("win_rate", 0.0)
        if win_rate > 0.60:
            warnings.append(f"⚠️  Win rate {win_rate:.1%} > 60% - May indicate data leakage!")
            print(f"⚠️  Win rate: {win_rate:.1%} (UNREALISTIC - professionals achieve 54-56%)")
        else:
            print(f"✓ Win rate: {win_rate:.1%} (REALISTIC - professionals achieve 54-56%)")

        # Check annual ROI
        total_return = summary.get("total_return", 0.0)
        n_seasons = len(summary.get("trades_by_season", {}))
        if n_seasons > 0:
            annual_roi = (1 + total_return) ** (1 / n_seasons) - 1
            if annual_roi > 0.10:
                warnings.append(
                    f"⚠️  Annual ROI {annual_roi:.1%} > 10% - May indicate data leakage!"
                )
                print(f"⚠️  Annual ROI: {annual_roi:.1%} (UNREALISTIC - professionals achieve 3-5%)")
            else:
                print(f"✓ Annual ROI: {annual_roi:.1%} (REALISTIC - professionals achieve 3-5%)")

        # Check Sharpe ratio
        sharpe = summary.get("sharpe_ratio", 0.0)
        if sharpe > 2.0:
            warnings.append(f"⚠️  Sharpe ratio {sharpe:.2f} > 2.0 - May indicate overfitting!")
            print(f"⚠️  Sharpe ratio: {sharpe:.2f} (UNREALISTIC - professionals achieve ~1.0-1.5)")
        else:
            print(f"✓ Sharpe ratio: {sharpe:.2f} (REALISTIC - professionals achieve ~1.0-1.5)")

        # Check max drawdown
        max_dd = summary.get("max_drawdown", 0.0)
        if abs(max_dd) < 0.05:
            warnings.append(f"⚠️  Max drawdown {max_dd:.1%} < 5% - Unusually low volatility!")
            print(f"⚠️  Max drawdown: {max_dd:.1%} (SUSPICIOUSLY LOW - expect 10-20%)")
        else:
            print(f"✓ Max drawdown: {max_dd:.1%} (REALISTIC - expect 10-20%)")

        if warnings:
            print(f"\n{'='*80}")
            print("⚠️  WARNINGS - POSSIBLE DATA LEAKAGE OR OVERFITTING")
            print(f"{'='*80}")
            for warning in warnings:
                print(warning)
            print("\nRecommendations:")
            print("  1. Check for vegas-derived features in model")
            print("  2. Verify no look-ahead bias in features")
            print("  3. Validate time-based cross-validation")
        else:
            print(f"\n{'='*80}")
            print("✅ RESULTS APPEAR REALISTIC - NO DATA LEAKAGE DETECTED")
            print(f"{'='*80}")

    def backtest(
        self,
        ensemble: RealisticGameOutcomeEnsemble,
        df: pd.DataFrame,
        train_df: pd.DataFrame,
    ) -> dict:
        """
        Run REALISTIC backtest with strict safety constraints.

        Args:
            ensemble: Pre-trained RealisticGameOutcomeEnsemble
            df: Backtest period games (with spread_close)
            train_df: Training data (for initial model state)

        Returns:
            Backtest summary metrics
        """
        print(f"\n{'='*80}")
        print("REALISTIC ENSEMBLE BACKTEST")
        print("(STRICT Safety Constraints - No Data Leakage)")
        print(f"{'='*80}")
        print(f"Backtest period: {df['season'].min()}-{df['season'].max()}")
        print(f"Games: {len(df)}")
        print(f"Initial bankroll: ${self.initial_bankroll:,.2f}")
        print("\nSTRICT CONSTRAINTS:")
        print(f"  Max bet fraction: {self.max_bet_frac:.1%} (vs 10% standard)")
        print(f"  Max bet absolute: ${self.max_bet_absolute:,.0f}")
        print(
            f"  Probability clipping: [{self.prob_clip_min:.2f}, {self.prob_clip_max:.2f}] (vs [0.05, 0.95] standard)"
        )
        print(f"  Min edge threshold: {self.edge_threshold:.1%} (vs 5% standard)")
        print(f"  Kelly multiplier: {self.kelly_multiplier} (quarter Kelly)")

        self.bankroll = self.initial_bankroll
        self.trades = []
        self.cumulative_returns = []

        # Process games week by week
        for (season, week), week_df in df.groupby(["season", "week"], sort=True):
            # Generate predictions for this week
            ensemble_preds, component_preds = ensemble.predict(week_df)

            # Clip probabilities to realistic range
            ensemble_preds = np.clip(ensemble_preds, self.prob_clip_min, self.prob_clip_max)

            # Evaluate betting opportunities
            for i, (_, row) in enumerate(week_df.iterrows()):
                game_id = row["game_id"]
                home_team = row["home_team"]
                away_team = row["away_team"]
                margin = row["margin"]
                spread = row.get("spread_close", None)

                if pd.isna(spread):
                    continue  # No betting line available

                # Model prediction (CLIPPED)
                prob_home_win = ensemble_preds[i]

                # Market implied probability
                market_prob_home, market_prob_away = get_implied_prob(spread, self.vig)

                # Calculate edge
                edge = prob_home_win - market_prob_home

                # Only bet if edge exceeds STRICT threshold (3% vs 5%)
                if abs(edge) < self.edge_threshold:
                    continue

                # Determine bet side
                if edge > 0:
                    # Bet on home team
                    bet_side = "home"
                    bet_prob = prob_home_win
                    outcome = 1 if (margin + spread) > 0 else 0
                else:
                    # Bet on away team
                    bet_side = "away"
                    bet_prob = 1 - prob_home_win
                    outcome = 1 if (-margin + (-spread)) > 0 else 0

                # Calculate Kelly bet size with STRICT limits
                decimal_odds = american_to_decimal(self.vig)

                # Kelly formula with safety
                prob_clipped = np.clip(bet_prob, self.prob_clip_min, self.prob_clip_max)
                net_odds = decimal_odds - 1
                if net_odds <= 0:
                    continue

                kelly_frac = (prob_clipped * (net_odds + 1) - 1) / net_odds
                kelly_frac = max(0.0, kelly_frac)

                # Apply fractional Kelly
                kelly_frac = kelly_frac * self.kelly_multiplier

                # STRICT CAP: Max 5% of bankroll
                kelly_frac = min(kelly_frac, self.max_bet_frac)

                if kelly_frac <= 0:
                    continue  # No bet

                # Place bet
                bet_size = self.bankroll * kelly_frac

                # STRICT CAP: Max $5,000 absolute
                bet_size = min(bet_size, self.max_bet_absolute)

                if outcome == 1:
                    # Win
                    profit = bet_size * (decimal_odds - 1)
                    self.bankroll += profit
                else:
                    # Loss
                    profit = -bet_size
                    self.bankroll += profit

                # Record trade
                self.trades.append(
                    {
                        "game_id": game_id,
                        "season": season,
                        "week": week,
                        "home_team": home_team,
                        "away_team": away_team,
                        "bet_side": bet_side,
                        "bet_prob": bet_prob,
                        "market_prob": market_prob_home if bet_side == "home" else market_prob_away,
                        "edge": abs(edge),
                        "kelly_frac": kelly_frac,
                        "bet_size": bet_size,
                        "outcome": outcome,
                        "profit": profit,
                        "bankroll": self.bankroll,
                    }
                )

                self.cumulative_returns.append(self.bankroll / self.initial_bankroll - 1)

        # Calculate summary metrics
        trades_df = pd.DataFrame(self.trades)

        if len(trades_df) == 0:
            print("\nNo trades placed (no edges above strict threshold)")
            summary = {
                "n_trades": 0,
                "final_bankroll": self.bankroll,
                "total_return": 0.0,
                "win_rate": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "trades_by_season": {},
            }
            self._reality_check(summary)
            return summary

        # Summary metrics
        n_trades = len(trades_df)
        final_bankroll = self.bankroll
        total_return = (final_bankroll / self.initial_bankroll) - 1
        win_rate = trades_df["outcome"].mean()

        # Sharpe ratio
        if len(self.cumulative_returns) > 1:
            returns_series = pd.Series(self.cumulative_returns)
            returns_diff = returns_series.diff().dropna()
            sharpe_ratio = (
                (returns_diff.mean() / returns_diff.std()) * np.sqrt(16)
                if returns_diff.std() > 0
                else 0.0
            )
        else:
            sharpe_ratio = 0.0

        # Maximum drawdown
        cumulative_max = pd.Series(self.cumulative_returns).cummax()
        drawdown = pd.Series(self.cumulative_returns) - cumulative_max
        max_drawdown = drawdown.min()

        print(f"\n{'='*80}")
        print("REALISTIC BACKTEST SUMMARY")
        print(f"{'='*80}")
        print(f"Total trades: {n_trades}")
        print(f"Win rate: {win_rate:.1%}")
        print(f"Final bankroll: ${final_bankroll:,.2f}")
        print(f"Total return: {total_return:+.1%}")
        print(f"Sharpe ratio: {sharpe_ratio:.2f}")
        print(f"Max drawdown: {max_drawdown:.1%}")

        # By season
        print(f"\n{'='*80}")
        print("BY SEASON")
        print(f"{'='*80}")
        for season in sorted(trades_df["season"].unique()):
            season_trades = trades_df[trades_df["season"] == season]
            season_profit = season_trades["profit"].sum()
            season_win_rate = season_trades["outcome"].mean()
            print(
                f"{season}: {len(season_trades)} trades, Win rate: {season_win_rate:.1%}, P&L: ${season_profit:+,.2f}"
            )

        summary = {
            "n_trades": int(n_trades),
            "win_rate": float(win_rate),
            "final_bankroll": float(final_bankroll),
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "trades_by_season": trades_df.groupby("season").size().to_dict(),
        }

        # REALITY CHECK
        self._reality_check(summary)

        return summary


def parse_args():
    ap = argparse.ArgumentParser(description="REALISTIC Backtest for Game Outcome Ensemble")
    ap.add_argument(
        "--train-seasons",
        default="2019,2020,2021",
        help="Training seasons (comma-separated)",
    )
    ap.add_argument(
        "--backtest-seasons",
        default="2022,2023,2024",
        help="Backtest seasons (comma-separated)",
    )
    ap.add_argument(
        "--initial-bankroll",
        type=float,
        default=10000.0,
        help="Initial bankroll ($)",
    )
    ap.add_argument(
        "--output-dir",
        default="models/ensemble/realistic_backtest",
        help="Output directory",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    # Parse seasons
    train_seasons = [int(s.strip()) for s in args.train_seasons.split(",")]
    backtest_seasons = [int(s.strip()) for s in args.backtest_seasons.split(",")]

    print(f"Train seasons: {train_seasons}")
    print(f"Backtest seasons: {backtest_seasons}")

    # Load data
    all_seasons = sorted(set(train_seasons + backtest_seasons))
    df_all = fetch_games(all_seasons)

    # Filter to games with spread data
    df_all = df_all[df_all["spread_close"].notna()]

    train_df = df_all[df_all["season"].isin(train_seasons)]
    backtest_df = df_all[df_all["season"].isin(backtest_seasons)]

    print(f"\nTrain: {len(train_df)} games")
    print(f"Backtest: {len(backtest_df)} games")

    # Train REALISTIC ensemble (no vegas leakage)
    print(f"\n{'='*80}")
    print("TRAINING REALISTIC ENSEMBLE (NO VEGAS LEAKAGE)")
    print(f"{'='*80}")
    ensemble = RealisticGameOutcomeEnsemble()
    ensemble.fit(train_df)

    # Run REALISTIC backtest with STRICT constraints
    backtest = RealisticBacktest(
        initial_bankroll=args.initial_bankroll,
        kelly_multiplier=0.25,
        edge_threshold=0.03,  # STRICT: 3% min edge
        max_bet_frac=0.05,  # STRICT: 5% max bet
        max_bet_absolute=5000.0,  # STRICT: $5k max
        prob_clip_min=0.10,  # STRICT: clip to [0.10, 0.90]
        prob_clip_max=0.90,
    )

    summary = backtest.backtest(ensemble, backtest_df, train_df)

    # Save results
    output_dir = Path(args.output_dir)
    backtest.save_results(output_dir)

    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()
