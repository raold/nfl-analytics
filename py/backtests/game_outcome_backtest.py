"""
Backtest for 5-Component Game Outcome Ensemble

Simulates real-time deployment from 2022-2025:
- Train on data up to game week N-1
- Predict week N
- Update model with week N results
- Calculate cumulative returns using Kelly criterion

Betting Strategy:
- Bet when ensemble prediction differs from market line by >= threshold
- Use Kelly criterion for position sizing
- Track P&L, Sharpe ratio, maximum drawdown
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from py.ensemble.game_outcome_ensemble import GameOutcomeEnsemble, fetch_games


def kelly_fraction(prob: float, odds: float, kelly_multiplier: float = 0.25, max_bet_frac: float = 0.10) -> float:
    """
    Calculate Kelly criterion bet size with SAFETY LIMITS.

    Args:
        prob: Probability of winning (0-1)
        odds: Decimal odds (e.g., 1.91 for -110)
        kelly_multiplier: Fractional Kelly (default 0.25 = quarter Kelly)
        max_bet_frac: Maximum bet as fraction of bankroll (default 0.10 = 10%)

    Returns:
        Fraction of bankroll to bet (0-1), capped at max_bet_frac

    Safety Mechanisms:
    1. Clip probability to [0.05, 0.95] to prevent extreme predictions
    2. Cap Kelly fraction at max_bet_frac (default 10% of bankroll)
    3. Apply fractional Kelly multiplier for additional safety
    """
    # SAFETY: Clip probability to reasonable range
    # Prevents overconfident predictions from causing massive bets
    prob = np.clip(prob, 0.05, 0.95)

    if prob <= 0 or prob >= 1:
        return 0.0

    # Kelly formula: f = (p * (b + 1) - 1) / b
    # where b = odds - 1 (net odds)
    net_odds = odds - 1
    if net_odds <= 0:
        return 0.0

    kelly_frac = (prob * (net_odds + 1) - 1) / net_odds
    kelly_frac = max(0.0, kelly_frac)  # No betting on negative edge

    # Apply fractional Kelly
    kelly_frac = kelly_frac * kelly_multiplier

    # SAFETY: Cap at maximum bet fraction
    kelly_frac = min(kelly_frac, max_bet_frac)

    return kelly_frac


def american_to_decimal(american_odds: float) -> float:
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return 1 + (american_odds / 100)
    else:
        return 1 + (100 / abs(american_odds))


def get_implied_prob(spread: float, vig: float = -110) -> tuple[float, float]:
    """
    Get implied probabilities from spread line.

    Args:
        spread: Point spread (positive = home underdog)
        vig: Vig on spread bet (default -110)

    Returns:
        (prob_home_covers, prob_away_covers)
    """
    # For spread bets, both sides usually have same vig (e.g., -110)
    # Implied prob = 1 / decimal_odds
    decimal_odds = american_to_decimal(vig)
    implied_prob = 1 / decimal_odds

    # Adjust for vig (no-vig probability)
    total_prob = implied_prob * 2  # Both sides have same vig
    fair_prob = implied_prob / total_prob

    return fair_prob, fair_prob


class GameOutcomeBacktest:
    """Backtest for game outcome ensemble with Kelly criterion betting."""

    def __init__(
        self,
        initial_bankroll: float = 10000.0,
        kelly_multiplier: float = 0.25,
        edge_threshold: float = 0.05,
        vig: int = -110,
        max_bet_frac: float = 0.10,
        max_bet_absolute: float = 5000.0,
    ):
        """
        Args:
            initial_bankroll: Starting bankroll ($)
            kelly_multiplier: Fractional Kelly (0.25 = quarter Kelly for safety)
            edge_threshold: Minimum edge to place bet (5% = 0.05)
            vig: Vig on spread bets (default -110)
            max_bet_frac: Maximum bet as fraction of bankroll (default 0.10 = 10%)
            max_bet_absolute: Maximum bet size in dollars (default $5000)

        Safety Features:
        - Probabilities clipped to [0.05, 0.95] range
        - Bets capped at max_bet_frac of bankroll (default 10%)
        - Bets capped at max_bet_absolute dollars (default $5000)
        """
        self.initial_bankroll = initial_bankroll
        self.kelly_multiplier = kelly_multiplier
        self.edge_threshold = edge_threshold
        self.vig = vig
        self.max_bet_frac = max_bet_frac
        self.max_bet_absolute = max_bet_absolute

        self.bankroll = initial_bankroll
        self.trades: list[dict] = []
        self.cumulative_returns: list[float] = []

    def backtest(
        self,
        ensemble: GameOutcomeEnsemble,
        df: pd.DataFrame,
        train_df: pd.DataFrame,
    ) -> dict:
        """
        Run sequential backtest on df, starting with ensemble trained on train_df.

        Args:
            ensemble: Pre-trained ensemble
            df: Backtest period games (with spread_close)
            train_df: Training data (for initial model state)

        Returns:
            Backtest summary metrics
        """
        print(f"\n{'='*80}")
        print("GAME OUTCOME ENSEMBLE BACKTEST")
        print(f"{'='*80}")
        print(f"Backtest period: {df['season'].min()}-{df['season'].max()}")
        print(f"Games: {len(df)}")
        print(f"Initial bankroll: ${self.initial_bankroll:,.2f}")
        print(f"Kelly multiplier: {self.kelly_multiplier}")
        print(f"Edge threshold: {self.edge_threshold:.1%}")
        print(f"Vig: {self.vig}")

        self.bankroll = self.initial_bankroll
        self.trades = []
        self.cumulative_returns = []

        # Process games week by week
        for (season, week), week_df in df.groupby(["season", "week"], sort=True):
            print(f"\nProcessing {season} Week {week}: {len(week_df)} games")

            # Generate predictions for this week
            ensemble_preds, component_preds = ensemble.predict(week_df)

            # Evaluate betting opportunities
            for i, (_, row) in enumerate(week_df.iterrows()):
                game_id = row["game_id"]
                home_team = row["home_team"]
                away_team = row["away_team"]
                margin = row["margin"]
                spread = row.get("spread_close", None)

                if pd.isna(spread):
                    continue  # No betting line available

                # Model prediction
                prob_home_win = ensemble_preds[i]

                # Market implied probability
                market_prob_home, market_prob_away = get_implied_prob(spread, self.vig)

                # Calculate edge
                edge = prob_home_win - market_prob_home

                # Only bet if edge exceeds threshold
                if abs(edge) < self.edge_threshold:
                    continue

                # Determine bet side
                if edge > 0:
                    # Bet on home team
                    bet_side = "home"
                    bet_prob = prob_home_win
                    bet_spread = spread  # Home team spread
                    outcome = 1 if (margin + spread) > 0 else 0
                else:
                    # Bet on away team
                    bet_side = "away"
                    bet_prob = 1 - prob_home_win
                    bet_spread = -spread  # Away team gets opposite spread
                    outcome = 1 if (-margin + (-spread)) > 0 else 0

                # Calculate Kelly bet size with safety limits
                decimal_odds = american_to_decimal(self.vig)
                kelly_frac = kelly_fraction(bet_prob, decimal_odds, self.kelly_multiplier, self.max_bet_frac)

                if kelly_frac <= 0:
                    continue  # No bet

                # Place bet
                bet_size = self.bankroll * kelly_frac

                # SAFETY: Cap at absolute maximum bet size
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
                self.trades.append({
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
                })

                self.cumulative_returns.append(self.bankroll / self.initial_bankroll - 1)

            print(f"  Trades this week: {len([t for t in self.trades if t['season'] == season and t['week'] == week])}")
            print(f"  Bankroll: ${self.bankroll:,.2f}")

        # Calculate summary metrics
        trades_df = pd.DataFrame(self.trades)

        if len(trades_df) == 0:
            print("\nNo trades placed (no edges above threshold)")
            return {
                "n_trades": 0,
                "final_bankroll": self.bankroll,
                "total_return": 0.0,
                "win_rate": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
            }

        # Summary metrics
        n_trades = len(trades_df)
        final_bankroll = self.bankroll
        total_return = (final_bankroll / self.initial_bankroll) - 1
        win_rate = trades_df["outcome"].mean()

        # Sharpe ratio (assume 252 trading days/year, ~16 NFL weeks/year)
        if len(self.cumulative_returns) > 1:
            returns_series = pd.Series(self.cumulative_returns)
            returns_diff = returns_series.diff().dropna()
            sharpe_ratio = (returns_diff.mean() / returns_diff.std()) * np.sqrt(16) if returns_diff.std() > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        # Maximum drawdown
        cumulative_max = pd.Series(self.cumulative_returns).cummax()
        drawdown = pd.Series(self.cumulative_returns) - cumulative_max
        max_drawdown = drawdown.min()

        print(f"\n{'='*80}")
        print("BACKTEST SUMMARY")
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
            print(f"{season}: {len(season_trades)} trades, Win rate: {season_win_rate:.1%}, P&L: ${season_profit:+,.2f}")

        return {
            "n_trades": int(n_trades),
            "win_rate": float(win_rate),
            "final_bankroll": float(final_bankroll),
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "trades_by_season": trades_df.groupby("season").size().to_dict(),
        }

    def save_results(self, output_dir: Path):
        """Save backtest results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save trades
        trades_df = pd.DataFrame(self.trades)
        trades_file = output_dir / "trades.csv"
        trades_df.to_csv(trades_file, index=False)
        print(f"\nTrades saved to {trades_file}")

        # Save cumulative returns
        returns_df = pd.DataFrame({
            "trade_num": range(len(self.cumulative_returns)),
            "cumulative_return": self.cumulative_returns,
        })
        returns_file = output_dir / "cumulative_returns.csv"
        returns_df.to_csv(returns_file, index=False)
        print(f"Cumulative returns saved to {returns_file}")


def parse_args():
    ap = argparse.ArgumentParser(description="Backtest 5-Component Game Outcome Ensemble")
    ap.add_argument(
        "--train-seasons",
        default="2019,2020,2021",
        help="Training seasons (comma-separated)",
    )
    ap.add_argument(
        "--backtest-seasons",
        default="2022,2023,2024,2025",
        help="Backtest seasons (comma-separated)",
    )
    ap.add_argument(
        "--initial-bankroll",
        type=float,
        default=10000.0,
        help="Initial bankroll ($)",
    )
    ap.add_argument(
        "--kelly-multiplier",
        type=float,
        default=0.25,
        help="Fractional Kelly (0.25 = quarter Kelly)",
    )
    ap.add_argument(
        "--edge-threshold",
        type=float,
        default=0.05,
        help="Minimum edge to bet (5%% = 0.05)",
    )
    ap.add_argument(
        "--output-dir",
        default="models/ensemble/backtest",
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

    # Train ensemble
    print(f"\n{'='*80}")
    print("TRAINING ENSEMBLE")
    print(f"{'='*80}")
    ensemble = GameOutcomeEnsemble()
    ensemble.fit(train_df)

    # Run backtest
    backtest = GameOutcomeBacktest(
        initial_bankroll=args.initial_bankroll,
        kelly_multiplier=args.kelly_multiplier,
        edge_threshold=args.edge_threshold,
    )

    summary = backtest.backtest(ensemble, backtest_df, train_df)

    # Save results
    output_dir = Path(args.output_dir)
    backtest.save_results(output_dir)

    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
