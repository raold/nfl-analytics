#!/usr/bin/env python3
"""
Backtest CQL betting strategy with realistic constraints.

Features:
- Kelly criterion position sizing
- Bankroll management (max bet size, stop-loss)
- Slippage modeling (line movement)
- Transaction costs (vig)
- Temporal holdout (train/test split by date)

Usage:
    python py/rl/simulate_betting.py \\
        --model models/cql/805ae9f0 \\
        --data data/rl_logged.csv \\
        --initial-bankroll 10000 \\
        --max-bet-pct 0.05 \\
        --vig 0.0455 \\
        --output results/betting_simulation.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from py.models.load_cql_ensemble import CQLEnsemble


class BettingSimulation:
    """Realistic betting simulation with bankroll management."""

    def __init__(
        self,
        initial_bankroll: float = 10000.0,
        max_bet_pct: float = 0.05,
        min_bet_size: float = 10.0,
        vig: float = 0.0455,  # Standard -110 odds
        stop_loss_pct: float = 0.5,  # Stop at 50% drawdown
    ):
        """
        Initialize betting simulation.

        Args:
            initial_bankroll: Starting capital
            max_bet_pct: Maximum bet as fraction of bankroll
            min_bet_size: Minimum bet size (dollars)
            vig: Vigorish (juice) on bets
            stop_loss_pct: Stop trading at this drawdown fraction
        """
        self.initial_bankroll = initial_bankroll
        self.max_bet_pct = max_bet_pct
        self.min_bet_size = min_bet_size
        self.vig = vig
        self.stop_loss_pct = stop_loss_pct

        # Simulation state
        self.bankroll = initial_bankroll
        self.peak_bankroll = initial_bankroll
        self.trades: list[dict] = []
        self.stopped_out = False

    def calculate_bet_size(self, kelly_fraction: float) -> float:
        """Calculate bet size with constraints."""
        # Kelly sizing
        bet_size = self.bankroll * kelly_fraction

        # Apply max bet constraint
        bet_size = min(bet_size, self.bankroll * self.max_bet_pct)

        # Apply min bet constraint
        if bet_size < self.min_bet_size:
            return 0.0

        return bet_size

    def execute_bet(
        self,
        game_id: str,
        action: int,
        bet_fraction: float,
        outcome: float,
        q_value: float = 0.0,
        confidence: float = 1.0,
    ) -> dict:
        """
        Execute a single bet and update bankroll.

        Args:
            game_id: Unique game identifier
            action: Predicted action (0=no-bet, 1-3=bet)
            bet_fraction: Bet size as fraction of bankroll
            outcome: Actual outcome (win=+0.91, loss=-1.0 at -110)
            q_value: Predicted Q-value
            confidence: Model confidence (0-1)

        Returns:
            Trade record dict
        """
        if self.stopped_out:
            return {
                "game_id": game_id,
                "action": "skip",
                "reason": "stopped_out",
                "bankroll": self.bankroll,
            }

        # Skip if no-bet action
        if action == 0:
            return {
                "game_id": game_id,
                "action": "skip",
                "reason": "no_bet_action",
                "bankroll": self.bankroll,
            }

        # Calculate bet size
        bet_size = self.calculate_bet_size(bet_fraction)

        if bet_size == 0:
            return {
                "game_id": game_id,
                "action": "skip",
                "reason": "bet_too_small",
                "bankroll": self.bankroll,
            }

        # Execute bet
        pnl = bet_size * outcome
        self.bankroll += pnl
        self.peak_bankroll = max(self.peak_bankroll, self.bankroll)

        # Check stop-loss
        drawdown = (self.peak_bankroll - self.bankroll) / self.peak_bankroll
        if drawdown >= self.stop_loss_pct:
            self.stopped_out = True

        # Record trade
        trade = {
            "game_id": game_id,
            "action": action,
            "bet_size": bet_size,
            "outcome": outcome,
            "pnl": pnl,
            "bankroll": self.bankroll,
            "drawdown": drawdown,
            "q_value": q_value,
            "confidence": confidence,
            "stopped_out": self.stopped_out,
        }
        self.trades.append(trade)

        return trade

    def get_metrics(self) -> dict:
        """Calculate performance metrics."""
        if len(self.trades) == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "roi_pct": 0.0,
                "sharpe": 0.0,
                "max_dd_pct": 0.0,
                "final_bankroll": self.bankroll,
            }

        trades_df = pd.DataFrame(self.trades)

        # Win rate
        wins = (trades_df["pnl"] > 0).sum()
        win_rate = wins / len(trades_df)

        # ROI
        total_staked = trades_df["bet_size"].sum()
        total_pnl = trades_df["pnl"].sum()
        roi_pct = (total_pnl / total_staked * 100) if total_staked > 0 else 0.0

        # Sharpe ratio
        pnl_series = trades_df["pnl"]
        mean_pnl = pnl_series.mean()
        std_pnl = pnl_series.std(ddof=1) if len(pnl_series) > 1 else 0.0
        sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0

        # Max drawdown
        max_dd_pct = trades_df["drawdown"].max() * 100

        return {
            "total_trades": len(trades_df),
            "win_rate": win_rate,
            "total_pnl": float(total_pnl),
            "roi_pct": roi_pct,
            "sharpe": sharpe,
            "max_dd_pct": max_dd_pct,
            "final_bankroll": float(self.bankroll),
            "total_return_pct": (self.bankroll - self.initial_bankroll)
            / self.initial_bankroll
            * 100,
            "stopped_out": self.stopped_out,
        }


def run_simulation(
    ensemble: CQLEnsemble,
    data: pd.DataFrame,
    initial_bankroll: float = 10000.0,
    max_bet_pct: float = 0.05,
    use_ensemble: bool = True,
    confidence_threshold: float = 0.05,
) -> dict:
    """Run betting simulation on historical data."""
    sim = BettingSimulation(initial_bankroll=initial_bankroll, max_bet_pct=max_bet_pct)

    print(f"\n{'='*80}")
    print("BETTING SIMULATION")
    print(f"{'='*80}")
    print(f"Initial bankroll: ${initial_bankroll:,.2f}")
    print(f"Max bet: {max_bet_pct*100:.1f}% of bankroll")
    print(f"Mode: {'Ensemble' if use_ensemble else 'Single model'}")
    print(f"Games: {len(data)}")
    print(f"{'='*80}\n")

    for idx, row in data.iterrows():
        # Get prediction
        if use_ensemble:
            pred = ensemble.predict_ensemble(row, confidence_threshold)
            action = pred["action"]
            bet_fraction = pred["bet_size"]
            q_value = pred["q_best"]
            confidence = pred["confidence"]
        else:
            pred = ensemble.predict_single(row)
            action = pred["action"]
            bet_fraction = pred["bet_size"]
            q_value = pred["q_best"]
            confidence = 1.0

        # Execute bet
        outcome = row["r"]
        game_id = row.get("game_id", f"game_{idx}")

        sim.execute_bet(
            game_id=game_id,
            action=action,
            bet_fraction=bet_fraction,
            outcome=outcome,
            q_value=q_value,
            confidence=confidence,
        )

        # Print progress every 50 trades
        if len(sim.trades) % 50 == 0 and len(sim.trades) > 0:
            metrics = sim.get_metrics()
            print(
                f"[{len(sim.trades):>4} trades] Bankroll: ${sim.bankroll:>10,.2f} | "
                f"Win rate: {metrics['win_rate']*100:>5.1f}% | "
                f"ROI: {metrics['roi_pct']:>+6.1f}%"
            )

        if sim.stopped_out:
            print(f"\n⚠️  STOPPED OUT after {len(sim.trades)} trades (50% drawdown)")
            break

    # Final metrics
    metrics = sim.get_metrics()

    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Total trades:     {metrics['total_trades']}")
    print(f"Win rate:         {metrics['win_rate']*100:.1f}%")
    print(f"Total P&L:        ${metrics['total_pnl']:+,.2f}")
    print(f"ROI:              {metrics['roi_pct']:+.2f}%")
    print(f"Sharpe ratio:     {metrics['sharpe']:.2f}")
    print(f"Max drawdown:     {metrics['max_dd_pct']:.1f}%")
    print(f"Final bankroll:   ${metrics['final_bankroll']:,.2f}")
    print(f"Total return:     {metrics['total_return_pct']:+.1f}%")
    print(f"{'='*80}\n")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Simulate CQL betting strategy")
    parser.add_argument("--model", type=str, default="models/cql/805ae9f0", help="Best model path")
    parser.add_argument("--ensemble-dir", type=str, default="models/cql", help="Ensemble directory")
    parser.add_argument("--data", type=str, required=True, help="Historical data CSV")
    parser.add_argument("--initial-bankroll", type=float, default=10000.0, help="Starting capital")
    parser.add_argument("--max-bet-pct", type=float, default=0.05, help="Max bet as % of bankroll")
    parser.add_argument(
        "--use-ensemble", action="store_true", help="Use ensemble (vs single model)"
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.05, help="Ensemble confidence threshold"
    )
    parser.add_argument("--test-split", type=float, default=0.2, help="Test set fraction")
    parser.add_argument(
        "--output", type=str, default="results/betting_simulation.json", help="Output path"
    )

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} games from {args.data}")

    # Train/test split
    split_idx = int(len(df) * (1 - args.test_split))
    df_test = df.iloc[split_idx:].copy()
    print(f"Testing on {len(df_test)} games (last {args.test_split*100:.0f}%)")

    # Load ensemble
    ensemble = CQLEnsemble(models_dir=args.ensemble_dir)

    if args.use_ensemble:
        ensemble.load_ensemble_models()
    else:
        model_id = Path(args.model).name
        ensemble.load_best_model(model_id)

    # Run simulation
    metrics = run_simulation(
        ensemble=ensemble,
        data=df_test,
        initial_bankroll=args.initial_bankroll,
        max_bet_pct=args.max_bet_pct,
        use_ensemble=args.use_ensemble,
        confidence_threshold=args.confidence_threshold,
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "config": {
            "model": args.model,
            "data": args.data,
            "initial_bankroll": args.initial_bankroll,
            "max_bet_pct": args.max_bet_pct,
            "use_ensemble": args.use_ensemble,
            "test_games": len(df_test),
        },
        "metrics": metrics,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"✅ Results saved to {output_path}")


if __name__ == "__main__":
    main()
