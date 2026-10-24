"""
thompson_switch_logic.py

Adaptive Ensemble Switching via Thompson Sampling

Uses Thompson Sampling (Bayesian bandit algorithm) to adaptively select
the best-performing model (XGBoost, CQL, IQL) based on recent performance.

Thompson Sampling balances:
- **Exploration**: Try models that haven't been tested much
- **Exploitation**: Use models that have performed well

Algorithm:
1. Maintain Beta distribution for each model: Beta(α, β)
   - α = wins + 1 (successes)
   - β = losses + 1 (failures)
2. Sample from each model's distribution
3. Select model with highest sample
4. After bet settles, update distribution:
   - Win → α += 1
   - Loss → β += 1

Usage:
    # Initialize Thompson sampler
    python py/production/thompson_switch_logic.py init

    # Select model for next bet
    python py/production/thompson_switch_logic.py select

    # Update after bet result
    python py/production/thompson_switch_logic.py update \
        --model xgboost \
        --result win

    # Get current status
    python py/production/thompson_switch_logic.py status
"""

import argparse
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import beta
from sqlalchemy import create_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class ModelStats:
    """Statistics for a single model."""

    model_name: str
    alpha: float  # Wins + 1
    beta: float  # Losses + 1
    n_bets: int
    n_wins: int
    n_losses: int
    win_rate: float
    expected_win_rate: float  # Alpha / (Alpha + Beta)
    variance: float  # Variance of Beta distribution


# ============================================================================
# Thompson Sampler
# ============================================================================


class ThompsonSampler:
    """
    Thompson Sampling for adaptive model selection.
    """

    def __init__(
        self,
        db_url: str = "postgresql://dro:sicillionbillions@localhost:5544/devdb01",
        models: list = None,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ):
        """
        Initialize Thompson sampler.

        Args:
            db_url: Database connection URL
            models: List of model names (default: ['xgboost', 'cql', 'iql'])
            prior_alpha: Prior successes (default: 1.0 = uniform prior)
            prior_beta: Prior failures (default: 1.0 = uniform prior)
        """
        self.engine = create_engine(db_url)
        self.models = models or ["xgboost", "cql", "iql"]
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

        logger.info(f"Thompson sampler initialized for models: {self.models}")

    def initialize_models(self):
        """
        Initialize model statistics in database.
        """
        for model_name in self.models:
            query = """
                INSERT INTO model_stats (
                    model_name, alpha, beta, n_bets, n_wins, n_losses, last_updated
                )
                VALUES (%s, %s, %s, 0, 0, 0, %s)
                ON CONFLICT (model_name) DO NOTHING
            """

            with self.engine.connect() as conn:
                conn.execute(
                    query,
                    (model_name, self.prior_alpha, self.prior_beta, datetime.now()),
                )
                conn.commit()

        logger.info(f"Initialized {len(self.models)} models with uniform prior")

    def get_model_stats(self, model_name: str) -> ModelStats:
        """
        Get current statistics for a model.

        Args:
            model_name: Model name

        Returns:
            ModelStats
        """
        query = "SELECT * FROM model_stats WHERE model_name = %s"

        with self.engine.connect() as conn:
            result = pd.read_sql(query, conn, params=(model_name,))

        if len(result) == 0:
            raise ValueError(f"Model '{model_name}' not found in database")

        row = result.iloc[0]

        alpha = row["alpha"]
        beta_param = row["beta"]

        expected_win_rate = alpha / (alpha + beta_param)
        variance = (alpha * beta_param) / ((alpha + beta_param) ** 2 * (alpha + beta_param + 1))

        return ModelStats(
            model_name=row["model_name"],
            alpha=alpha,
            beta=beta_param,
            n_bets=row["n_bets"],
            n_wins=row["n_wins"],
            n_losses=row["n_losses"],
            win_rate=row["n_wins"] / row["n_bets"] if row["n_bets"] > 0 else 0.0,
            expected_win_rate=expected_win_rate,
            variance=variance,
        )

    def select_model(self) -> str:
        """
        Select model using Thompson Sampling.

        Returns:
            Selected model name
        """
        # Get all model stats
        stats = [self.get_model_stats(name) for name in self.models]

        # Sample from each model's Beta distribution
        samples = {model.model_name: beta.rvs(model.alpha, model.beta) for model in stats}

        # Select model with highest sample
        selected_model = max(samples, key=samples.get)

        logger.info(f"Thompson Sampling selected '{selected_model}' (samples: {samples})")

        return selected_model

    def update_model(self, model_name: str, result: str):
        """
        Update model statistics after bet result.

        Args:
            model_name: Model name
            result: 'win' or 'loss'
        """
        if result not in ["win", "loss"]:
            raise ValueError(f"Invalid result: {result} (must be 'win' or 'loss')")

        # Update alpha/beta based on result
        if result == "win":
            query = """
                UPDATE model_stats
                SET alpha = alpha + 1,
                    n_bets = n_bets + 1,
                    n_wins = n_wins + 1,
                    last_updated = %s
                WHERE model_name = %s
            """
        else:  # loss
            query = """
                UPDATE model_stats
                SET beta = beta + 1,
                    n_bets = n_bets + 1,
                    n_losses = n_losses + 1,
                    last_updated = %s
                WHERE model_name = %s
            """

        with self.engine.connect() as conn:
            conn.execute(query, (datetime.now(), model_name))
            conn.commit()

        logger.info(f"Updated '{model_name}' with result: {result}")

    def get_all_stats(self) -> pd.DataFrame:
        """
        Get statistics for all models.

        Returns:
            Dataframe with all model stats
        """
        stats = [self.get_model_stats(name) for name in self.models]
        df = pd.DataFrame([asdict(s) for s in stats])

        return df

    def reset(self, model_name: str | None = None):
        """
        Reset statistics for a model (or all models).

        Args:
            model_name: Model to reset (None = reset all)
        """
        if model_name:
            query = """
                UPDATE model_stats
                SET alpha = %s,
                    beta = %s,
                    n_bets = 0,
                    n_wins = 0,
                    n_losses = 0,
                    last_updated = %s
                WHERE model_name = %s
            """

            with self.engine.connect() as conn:
                conn.execute(
                    query,
                    (self.prior_alpha, self.prior_beta, datetime.now(), model_name),
                )
                conn.commit()

            logger.info(f"Reset statistics for '{model_name}'")

        else:
            # Reset all models
            for name in self.models:
                self.reset(name)

    def plot_distributions(self):
        """
        Plot Beta distributions for all models.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed, cannot plot distributions")
            return

        stats = self.get_all_stats()

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.linspace(0, 1, 1000)

        for _, row in stats.iterrows():
            y = beta.pdf(x, row["alpha"], row["beta"])
            label = (
                f"{row['model_name']} (α={row['alpha']:.1f}, β={row['beta']:.1f}, "
                f"E[p]={row['expected_win_rate']:.3f})"
            )
            ax.plot(x, y, label=label, linewidth=2)

        ax.set_xlabel("Win Probability", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title("Model Win Probability Distributions (Thompson Sampling)", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("results/thompson_distributions.png", dpi=150)
        logger.info("Saved distributions plot to results/thompson_distributions.png")
        plt.close()

    def simulate_regret(self, n_bets: int = 1000, true_win_rates: dict[str, float] = None):
        """
        Simulate Thompson Sampling and calculate regret.

        Regret = (Optimal win rate - Thompson win rate) × n_bets

        Args:
            n_bets: Number of bets to simulate
            true_win_rates: True win rates for each model (for simulation)

        Returns:
            Cumulative regret over time
        """
        if true_win_rates is None:
            # Default: XGBoost slightly better
            true_win_rates = {"xgboost": 0.55, "cql": 0.53, "iql": 0.52}

        optimal_win_rate = max(true_win_rates.values())

        # Reset statistics
        self.reset()

        cumulative_regret = []
        cumulative_wins = 0

        for i in range(n_bets):
            # Select model
            selected = self.select_model()

            # Simulate bet outcome (binomial draw)
            win = np.random.random() < true_win_rates[selected]

            # Update statistics
            self.update_model(selected, "win" if win else "loss")

            # Calculate regret
            if win:
                cumulative_wins += 1

            actual_win_rate = cumulative_wins / (i + 1)
            regret = optimal_win_rate - actual_win_rate
            cumulative_regret.append(regret * (i + 1))

        logger.info(
            f"Simulation complete: {n_bets} bets, final regret = {cumulative_regret[-1]:.2f}"
        )

        return np.array(cumulative_regret)


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Thompson Sampling Adaptive Switching")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Initialize
    subparsers.add_parser("init", help="Initialize model statistics")

    # Select model
    subparsers.add_parser("select", help="Select model using Thompson Sampling")

    # Update model
    update_parser = subparsers.add_parser("update", help="Update model after bet result")
    update_parser.add_argument(
        "--model",
        required=True,
        choices=["xgboost", "cql", "iql"],
        help="Model name",
    )
    update_parser.add_argument(
        "--result", required=True, choices=["win", "loss"], help="Bet result"
    )

    # Status
    subparsers.add_parser("status", help="Show current model statistics")

    # Reset
    reset_parser = subparsers.add_parser("reset", help="Reset model statistics")
    reset_parser.add_argument("--model", help="Model to reset (default: all)")

    # Plot
    subparsers.add_parser("plot", help="Plot Beta distributions")

    # Simulate
    simulate_parser = subparsers.add_parser("simulate", help="Simulate Thompson Sampling")
    simulate_parser.add_argument(
        "--n-bets", type=int, default=1000, help="Number of bets to simulate"
    )

    args = parser.parse_args()

    # Initialize sampler
    sampler = ThompsonSampler()

    # Execute command
    if args.command == "init":
        sampler.initialize_models()
        print("Models initialized with uniform prior (α=1, β=1)")

    elif args.command == "select":
        selected = sampler.select_model()
        print(f"\n{'=' * 50}")
        print(f"SELECTED MODEL: {selected.upper()}")
        print("=" * 50)

        # Show current stats
        stats = sampler.get_all_stats()
        print("\nCurrent Model Statistics:")
        print(stats.to_string(index=False))
        print()

    elif args.command == "update":
        sampler.update_model(args.model, args.result)
        print(f"Updated '{args.model}' with result: {args.result}")

        # Show updated stats
        stats = sampler.get_model_stats(args.model)
        print("\nUpdated Statistics:")
        print(f"  Model: {stats.model_name}")
        print(f"  Bets: {stats.n_bets} (W: {stats.n_wins}, L: {stats.n_losses})")
        print(f"  Win Rate: {stats.win_rate:.1%}")
        print(f"  Expected Win Rate: {stats.expected_win_rate:.1%}")
        print(f"  Uncertainty (std): {np.sqrt(stats.variance):.2%}")

    elif args.command == "status":
        stats = sampler.get_all_stats()

        print("\n" + "=" * 70)
        print("THOMPSON SAMPLING STATUS")
        print("=" * 70)
        print()

        for _, row in stats.iterrows():
            print(f"Model: {row['model_name'].upper()}")
            print(f"  Bets: {row['n_bets']} (W: {row['n_wins']}, L: {row['n_losses']})")
            print(f"  Win Rate: {row['win_rate']:.1%}")
            print(
                f"  Expected Win Rate: {row['expected_win_rate']:.1%} ± {np.sqrt(row['variance']):.2%}"
            )
            print(f"  Beta Parameters: α={row['alpha']:.1f}, β={row['beta']:.1f}")
            print()

        print("=" * 70)

    elif args.command == "reset":
        sampler.reset(model_name=args.model)
        print(f"Reset {'all models' if args.model is None else args.model} to uniform prior")

    elif args.command == "plot":
        sampler.plot_distributions()
        print("Distributions plotted to results/thompson_distributions.png")

    elif args.command == "simulate":
        print(f"Simulating Thompson Sampling with {args.n_bets} bets...")
        regret = sampler.simulate_regret(n_bets=args.n_bets)

        print()
        print(f"Final cumulative regret: {regret[-1]:.2f} bets")
        print(f"Regret per bet: {regret[-1] / args.n_bets:.4f}")
        print(f"Regret per bet (%): {regret[-1] / args.n_bets * 100:.2f}%")

        # Show final model selection counts
        stats = sampler.get_all_stats()
        print("\nFinal Model Usage:")
        for _, row in stats.iterrows():
            print(f"  {row['model_name']}: {row['n_bets']} bets ({row['n_bets']/args.n_bets:.1%})")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
