"""
Logged fills backtesting for RL agents.

Simulates realistic order fills with adverse selection, line movement,
and book limits to evaluate RL policies under production conditions.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class FillSimulationParams:
    """Parameters for fill simulation."""

    base_fill_rate: float = 0.70  # Base probability of fill
    adverse_selection_factor: float = 0.15  # Penalty for +EV bets
    line_movement_std: float = 0.5  # Std of line movement
    min_edge_for_action: float = 0.02  # Minimum edge to place bet
    max_bet_size: float = 5.0  # Maximum bet size in units
    slippage_per_unit: float = 0.01  # Price slippage per unit bet


@dataclass
class BettingOutcome:
    """Single betting outcome with fill details."""

    game_id: str
    week: int
    season: int
    action: str  # 'bet' or 'pass'
    attempted_edge: float
    fill_probability: float
    was_filled: bool
    bet_size: float
    actual_outcome: int  # 1 for win, 0 for loss
    payout: float
    clv: float  # Closing line value


class LoggedFillsBacktest:
    """Backtest RL agents with realistic fill simulation."""

    def __init__(
        self, dataset_path: str, params: FillSimulationParams | None = None, random_seed: int = 42
    ):
        """
        Initialize backtest framework.

        Args:
            dataset_path: Path to RL dataset CSV
            params: Fill simulation parameters
            random_seed: Random seed for reproducibility
        """
        self.dataset = pd.read_csv(dataset_path)
        self.params = params or FillSimulationParams()
        self.rng = np.random.RandomState(random_seed)

        # Validate dataset
        required_cols = ["game_id", "season", "week", "model_prob", "market_prob", "outcome", "clv"]
        for col in required_cols:
            if col not in self.dataset.columns:
                raise ValueError(f"Dataset missing required column: {col}")

    def calculate_fill_probability(self, edge: float, bet_size: float, clv: float) -> float:
        """
        Calculate probability of order being filled.

        Args:
            edge: Perceived edge (model_prob - market_prob)
            bet_size: Bet size in units
            clv: Closing line value

        Returns:
            Probability order is filled [0, 1]
        """
        # Base fill rate
        fill_prob = self.params.base_fill_rate

        # Adverse selection: harder to fill +EV bets
        if edge > 0:
            adverse_penalty = self.params.adverse_selection_factor * edge
            fill_prob *= 1 - adverse_penalty

        # Bet size: larger bets less likely to fill
        size_penalty = self.params.slippage_per_unit * (bet_size - 1)
        fill_prob *= 1 - min(size_penalty, 0.3)

        # CLV: negative CLV means easier fill (we got bad line)
        if clv < 0:
            fill_prob *= 1.1  # Slightly easier to fill bad bets

        return np.clip(fill_prob, 0.05, 0.95)

    def simulate_fills(
        self,
        actions: np.ndarray,
        bet_sizes: np.ndarray,
        model_probs: np.ndarray,
        market_probs: np.ndarray,
        outcomes: np.ndarray,
        clvs: np.ndarray,
    ) -> list[BettingOutcome]:
        """
        Simulate fills for a series of betting decisions.

        Args:
            actions: Binary array (1=bet, 0=pass)
            bet_sizes: Bet sizes in units
            model_probs: Model win probabilities
            market_probs: Market implied probabilities
            outcomes: Actual outcomes (1=win, 0=loss)
            clvs: Closing line values

        Returns:
            List of betting outcomes with fill details
        """
        results = []

        for i in range(len(actions)):
            if actions[i] == 0:
                # No action
                outcome = BettingOutcome(
                    game_id=self.dataset.iloc[i]["game_id"],
                    week=self.dataset.iloc[i]["week"],
                    season=self.dataset.iloc[i]["season"],
                    action="pass",
                    attempted_edge=0.0,
                    fill_probability=0.0,
                    was_filled=False,
                    bet_size=0.0,
                    actual_outcome=outcomes[i],
                    payout=0.0,
                    clv=0.0,
                )
            else:
                # Betting action
                edge = model_probs[i] - market_probs[i]
                bet_size = bet_sizes[i]

                # Calculate fill probability
                fill_prob = self.calculate_fill_probability(edge, bet_size, clvs[i])

                # Simulate fill
                was_filled = self.rng.random() < fill_prob

                # Calculate payout if filled
                if was_filled:
                    # Assuming -110 odds (American style)
                    if outcomes[i] == 1:
                        payout = bet_size * 0.909  # Win
                    else:
                        payout = -bet_size  # Loss
                else:
                    payout = 0.0

                outcome = BettingOutcome(
                    game_id=self.dataset.iloc[i]["game_id"],
                    week=self.dataset.iloc[i]["week"],
                    season=self.dataset.iloc[i]["season"],
                    action="bet",
                    attempted_edge=edge,
                    fill_probability=fill_prob,
                    was_filled=was_filled,
                    bet_size=bet_size if was_filled else 0.0,
                    actual_outcome=outcomes[i],
                    payout=payout,
                    clv=clvs[i] if was_filled else 0.0,
                )

            results.append(outcome)

        return results

    def evaluate_policy(
        self,
        policy_name: str,
        actions: np.ndarray,
        bet_sizes: np.ndarray | None = None,
        n_simulations: int = 1000,
    ) -> dict[str, float]:
        """
        Evaluate a betting policy with Monte Carlo fill simulation.

        Args:
            policy_name: Name of the policy
            actions: Binary array of betting decisions
            bet_sizes: Bet sizes (if None, uses uniform 1.0)
            n_simulations: Number of Monte Carlo runs

        Returns:
            Dictionary of evaluation metrics
        """
        if bet_sizes is None:
            bet_sizes = np.ones(len(actions))

        # Extract data
        model_probs = self.dataset["model_prob"].values
        market_probs = self.dataset["market_prob"].values
        outcomes = self.dataset["outcome"].values
        clvs = self.dataset["clv"].values

        # Run simulations
        all_profits = []
        all_fill_rates = []
        all_active_weeks = []
        all_clv_filled = []

        for sim in range(n_simulations):
            results = self.simulate_fills(
                actions, bet_sizes, model_probs, market_probs, outcomes, clvs
            )

            # Calculate metrics for this simulation
            filled_bets = [r for r in results if r.was_filled]

            if len(filled_bets) > 0:
                total_profit = sum(r.payout for r in filled_bets)
                sum(r.bet_size for r in filled_bets)
                fill_rate = len(filled_bets) / sum(actions)

                # Active weeks (weeks with at least one fill)
                weeks_with_fills = len(set(r.week for r in filled_bets))

                # Average CLV on filled orders
                avg_clv = np.mean([r.clv for r in filled_bets])

                all_profits.append(total_profit)
                all_fill_rates.append(fill_rate)
                all_active_weeks.append(weeks_with_fills)
                all_clv_filled.append(avg_clv)

        # Aggregate statistics
        mean_profit = np.mean(all_profits)
        std_profit = np.std(all_profits)
        mean_fill_rate = np.mean(all_fill_rates)
        mean_active_weeks = np.mean(all_active_weeks)
        mean_clv = np.mean(all_clv_filled)

        # Calculate Sharpe ratio (adjusted for active weeks)
        total_weeks = self.dataset["week"].nunique()
        utilization = mean_active_weeks / total_weeks

        if std_profit > 0:
            sharpe = mean_profit / std_profit
            utilization_adj_sharpe = sharpe * np.sqrt(utilization)
        else:
            sharpe = 0.0
            utilization_adj_sharpe = 0.0

        return {
            "policy_name": policy_name,
            "mean_profit": mean_profit,
            "std_profit": std_profit,
            "sharpe_ratio": sharpe,
            "fill_rate": mean_fill_rate,
            "active_weeks": mean_active_weeks,
            "total_weeks": total_weeks,
            "utilization": utilization,
            "utilization_adj_sharpe": utilization_adj_sharpe,
            "mean_clv_filled": mean_clv,
            "n_simulations": n_simulations,
        }

    def compare_policies(
        self, policies: dict[str, np.ndarray], n_simulations: int = 1000
    ) -> pd.DataFrame:
        """
        Compare multiple betting policies.

        Args:
            policies: Dictionary mapping policy name to actions array
            n_simulations: Number of Monte Carlo runs per policy

        Returns:
            DataFrame with comparison metrics
        """
        results = []

        for name, actions in policies.items():
            metrics = self.evaluate_policy(name, actions, n_simulations=n_simulations)
            results.append(metrics)

        return pd.DataFrame(results)

    def generate_comparison_table(self, comparison_df: pd.DataFrame, output_path: Path):
        """Generate LaTeX comparison table."""

        lines = [
            r"\begin{table}[t]",
            r"  \centering",
            r"  \small",
            r"  \caption{RL vs Baseline: Performance with Realistic Fills}",
            r"  \begin{tabular}{lcccccc}",
            r"    \toprule",
            r"    Policy & Profit & Sharpe & Fill Rate & Active Wks & Util. & CLV \\",
            r"    \midrule",
        ]

        for _, row in comparison_df.iterrows():
            lines.append(
                f"    {row['policy_name']} & "
                f"\\${row['mean_profit']:.0f} & "
                f"{row['sharpe_ratio']:.2f} & "
                f"{row['fill_rate']*100:.1f}\\% & "
                f"{row['active_weeks']:.1f} & "
                f"{row['utilization']*100:.1f}\\% & "
                f"{row['mean_clv_filled']:.3f} \\\\"
            )

        lines.extend(
            [
                r"    \bottomrule",
                r"  \end{tabular}",
                r"  \label{tab:rl_vs_baseline}",
                r"\end{table}",
            ]
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines))
        print(f"✓ Generated: {output_path}")

    def generate_utilization_table(self, comparison_df: pd.DataFrame, output_path: Path):
        """Generate utilization-adjusted Sharpe table."""

        lines = [
            r"\begin{table}[t]",
            r"  \centering",
            r"  \small",
            r"  \caption{Utilization-Adjusted Performance Metrics}",
            r"  \begin{tabular}{lcccc}",
            r"    \toprule",
            r"    Policy & Sharpe & Utilization & Adj. Sharpe & Interpretation \\",
            r"    \midrule",
        ]

        for _, row in comparison_df.iterrows():
            # Determine interpretation
            if row["utilization_adj_sharpe"] > 1.0:
                interp = "Excellent"
            elif row["utilization_adj_sharpe"] > 0.5:
                interp = "Good"
            elif row["utilization_adj_sharpe"] > 0.0:
                interp = "Marginal"
            else:
                interp = "Poor"

            lines.append(
                f"    {row['policy_name']} & "
                f"{row['sharpe_ratio']:.2f} & "
                f"{row['utilization']*100:.1f}\\% & "
                f"{row['utilization_adj_sharpe']:.2f} & "
                f"{interp} \\\\"
            )

        lines.extend(
            [
                r"    \bottomrule",
                r"  \end{tabular}",
                r"  \label{tab:utilization_adjusted_sharpe}",
                r"\end{table}",
            ]
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines))
        print(f"✓ Generated: {output_path}")


def create_synthetic_policies(dataset: pd.DataFrame) -> dict[str, np.ndarray]:
    """Create synthetic policies for demonstration."""
    len(dataset)

    # Extract model and market probabilities
    model_probs = dataset["model_prob"].values
    market_probs = dataset["market_prob"].values
    edges = model_probs - market_probs

    policies = {}

    # GLM Baseline: Bet when edge > 2%
    policies["GLM"] = (edges > 0.02).astype(int)

    # XGBoost: Bet when edge > 1.5%
    policies["XGBoost"] = (edges > 0.015).astype(int)

    # DQN: More selective (edge > 3%)
    policies["DQN"] = (edges > 0.03).astype(int)

    # PPO: Balanced (edge > 2.5%)
    policies["PPO"] = (edges > 0.025).astype(int)

    # Kelly: Proportional to edge
    kelly_fractions = np.clip(edges / 0.1, 0, 1)
    policies["Kelly"] = (kelly_fractions > 0.1).astype(int)

    return policies


def main():
    """Main execution for demonstration."""
    import argparse

    parser = argparse.ArgumentParser(description="Backtest RL policies with realistic fills")
    parser.add_argument(
        "--dataset", type=str, default="data/rl_logged.csv", help="Path to RL dataset"
    )
    parser.add_argument(
        "--n-simulations", type=int, default=1000, help="Number of Monte Carlo simulations"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/dissertation/figures/out"),
        help="Output directory for tables",
    )

    args = parser.parse_args()

    print("Loading dataset...")
    backtest = LoggedFillsBacktest(args.dataset)

    print("Creating synthetic policies...")
    policies = create_synthetic_policies(backtest.dataset)

    print(f"Running {args.n_simulations} simulations per policy...")
    comparison = backtest.compare_policies(policies, n_simulations=args.n_simulations)

    print("\nResults:")
    print(comparison.to_string(index=False))

    # Generate tables
    print("\nGenerating LaTeX tables...")
    backtest.generate_comparison_table(comparison, args.output_dir / "rl_vs_baseline_table.tex")

    backtest.generate_utilization_table(
        comparison, args.output_dir / "utilization_adjusted_sharpe_table.tex"
    )

    # Save results to JSON
    results_path = args.output_dir / "rl_backtest_results.json"
    comparison.to_json(results_path, orient="records", indent=2)
    print(f"✓ Saved results: {results_path}")

    print("\n✅ RL backtest with logged fills complete!")


if __name__ == "__main__":
    main()
