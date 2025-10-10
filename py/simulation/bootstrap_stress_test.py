"""
Task 8: Bootstrap Stress Testing for Risk Validation

Uses bootstrap resampling and parameter perturbations to validate risk metrics
without needing to train a complex neural simulator.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict


@dataclass
class RiskMetrics:
    """Risk metrics for a strategy."""
    strategy_name: str
    n_trials: int
    mean_return: float
    median_return: float
    std_return: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    mean_drawdown: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    calmar_ratio: float  # Return / max_drawdown


class BootstrapStressTest:
    """
    Bootstrap-based stress testing for betting strategies.

    Instead of training a neural network, we:
    1. Bootstrap resample actual bet outcomes
    2. Apply perturbations (reduce win rate, increase variance, etc.)
    3. Compute risk metrics across trials
    """

    def __init__(self, random_seed: int = 42):
        self.rng = np.random.RandomState(random_seed)

    def compute_equity_curve(self,
                             outcomes: np.ndarray,
                             bet_sizes: np.ndarray,
                             initial_bankroll: float = 100.0,
                             vig: float = 0.0476) -> Tuple[np.ndarray, float]:
        """
        Compute equity curve from bet outcomes.

        Args:
            outcomes: Binary array (1 = win, 0 = loss)
            bet_sizes: Bet sizes in units
            initial_bankroll: Starting bankroll
            vig: Vigorish (0.0476 = -110 odds)

        Returns:
            equity_curve: Bankroll over time
            final_return: Final return percentage
        """
        n_bets = len(outcomes)
        equity = np.zeros(n_bets + 1)
        equity[0] = initial_bankroll

        for i in range(n_bets):
            if outcomes[i] == 1:
                # Win: profit = bet_size * (1 - vig)
                profit = bet_sizes[i] * (1 - vig)
                equity[i + 1] = equity[i] + profit
            else:
                # Loss: lose entire bet_size
                equity[i + 1] = equity[i] - bet_sizes[i]

        final_return = (equity[-1] - initial_bankroll) / initial_bankroll
        return equity, final_return

    def compute_drawdown(self, equity_curve: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute drawdown curve and max drawdown.

        Args:
            equity_curve: Equity over time

        Returns:
            drawdown_curve: Drawdown at each point
            max_drawdown: Maximum drawdown
        """
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_dd = abs(drawdown.min())
        return drawdown, max_dd

    def bootstrap_simulation(self,
                             outcomes: np.ndarray,
                             bet_sizes: np.ndarray,
                             n_trials: int = 1000,
                             perturbation: Dict[str, float] = None,
                             initial_bankroll: float = 100.0,
                             vig: float = 0.0476) -> RiskMetrics:
        """
        Run bootstrap simulation with optional perturbations.

        Args:
            outcomes: Historical bet outcomes (1 = win, 0 = loss)
            bet_sizes: Historical bet sizes
            n_trials: Number of bootstrap trials
            perturbation: Optional perturbations to apply:
                - 'win_rate_delta': Add/subtract from win rate (e.g., -0.05)
                - 'variance_multiplier': Multiply outcome variance (e.g., 1.5)
                - 'correlation': Induce correlated outcomes (e.g., 0.3)
            initial_bankroll: Starting bankroll
            vig: Vigorish

        Returns:
            RiskMetrics object
        """
        n_bets = len(outcomes)
        base_win_rate = outcomes.mean()

        # Apply win rate perturbation if specified
        target_win_rate = base_win_rate
        if perturbation and 'win_rate_delta' in perturbation:
            target_win_rate += perturbation['win_rate_delta']
            target_win_rate = np.clip(target_win_rate, 0.0, 1.0)

        # Storage for results
        returns = []
        max_drawdowns = []
        win_rates_trial = []

        for trial in range(n_trials):
            # Bootstrap resample indices
            indices = self.rng.choice(n_bets, size=n_bets, replace=True)

            # Get resampled outcomes and bet sizes
            trial_outcomes = outcomes[indices].copy()
            trial_bet_sizes = bet_sizes[indices]

            # Apply win rate perturbation by flipping some outcomes
            if perturbation and 'win_rate_delta' in perturbation:
                current_wins = trial_outcomes.sum()
                target_wins = int(n_bets * target_win_rate)
                delta_wins = target_wins - current_wins

                if delta_wins > 0:
                    # Need more wins: flip some losses to wins
                    loss_indices = np.where(trial_outcomes == 0)[0]
                    if len(loss_indices) >= delta_wins:
                        flip_indices = self.rng.choice(loss_indices, size=delta_wins, replace=False)
                        trial_outcomes[flip_indices] = 1
                elif delta_wins < 0:
                    # Need fewer wins: flip some wins to losses
                    win_indices = np.where(trial_outcomes == 1)[0]
                    if len(win_indices) >= abs(delta_wins):
                        flip_indices = self.rng.choice(win_indices, size=abs(delta_wins), replace=False)
                        trial_outcomes[flip_indices] = 0

            # Apply correlation perturbation (induce streaks)
            if perturbation and 'correlation' in perturbation:
                corr = perturbation['correlation']
                # Simple approach: group outcomes into streaks
                streak_length = int(1 / (1 - corr)) if corr < 1 else 1
                for i in range(0, n_bets, streak_length):
                    # Randomly choose outcome for this streak
                    streak_outcome = self.rng.binomial(1, target_win_rate)
                    trial_outcomes[i:i+streak_length] = streak_outcome

            # Compute equity curve
            equity, final_return = self.compute_equity_curve(
                trial_outcomes, trial_bet_sizes, initial_bankroll, vig
            )

            # Compute drawdown
            _, max_dd = self.compute_drawdown(equity)

            # Store results
            returns.append(final_return)
            max_drawdowns.append(max_dd)
            win_rates_trial.append(trial_outcomes.mean())

        # Convert to arrays
        returns = np.array(returns)
        max_drawdowns = np.array(max_drawdowns)
        win_rates_trial = np.array(win_rates_trial)

        # Compute risk metrics
        mean_return = returns.mean()
        median_return = np.median(returns)
        std_return = returns.std()
        sharpe = mean_return / std_return if std_return > 0 else 0.0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else std_return
        sortino = mean_return / downside_std if downside_std > 0 else 0.0

        # Value at Risk (VaR) and Conditional VaR (CVaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()

        # Calmar ratio
        mean_max_dd = max_drawdowns.mean()
        calmar = mean_return / mean_max_dd if mean_max_dd > 0 else 0.0

        return RiskMetrics(
            strategy_name="strategy",
            n_trials=n_trials,
            mean_return=mean_return,
            median_return=median_return,
            std_return=std_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            win_rate=win_rates_trial.mean(),
            mean_drawdown=mean_max_dd,
            max_drawdown=max_drawdowns.max(),
            var_95=var_95,
            cvar_95=cvar_95,
            var_99=var_99,
            cvar_99=cvar_99,
            calmar_ratio=calmar,
        )

    def stress_test_scenarios(self,
                              outcomes: np.ndarray,
                              bet_sizes: np.ndarray,
                              n_trials: int = 1000) -> Dict[str, RiskMetrics]:
        """
        Run multiple stress test scenarios.

        Scenarios:
        1. Baseline: Bootstrap with no perturbations
        2. Model degradation: Win rate drops by 5%
        3. Severe degradation: Win rate drops by 10%
        4. Variance shock: Increase outcome volatility
        5. Correlated losses: Induce losing streaks
        6. Combined worst case: Multiple perturbations

        Returns:
            Dict of scenario_name -> RiskMetrics
        """
        scenarios = {
            'baseline': {},
            'model_degradation_5pct': {'win_rate_delta': -0.05},
            'model_degradation_10pct': {'win_rate_delta': -0.10},
            'variance_shock': {'variance_multiplier': 1.5},
            'correlated_losses': {'correlation': 0.3},
            'worst_case': {'win_rate_delta': -0.10, 'correlation': 0.3},
        }

        results = {}
        for scenario_name, perturbation in scenarios.items():
            print(f"\nRunning scenario: {scenario_name}")
            if perturbation:
                print(f"  Perturbations: {perturbation}")

            metrics = self.bootstrap_simulation(
                outcomes=outcomes,
                bet_sizes=bet_sizes,
                n_trials=n_trials,
                perturbation=perturbation if perturbation else None,
            )

            metrics.strategy_name = scenario_name
            results[scenario_name] = metrics

            print(f"  Mean return: {metrics.mean_return:.4f}")
            print(f"  Sharpe: {metrics.sharpe_ratio:.4f}")
            print(f"  Sortino: {metrics.sortino_ratio:.4f}")
            print(f"  Max DD: {metrics.max_drawdown:.4f}")
            print(f"  VaR(95%): {metrics.var_95:.4f}")
            print(f"  CVaR(95%): {metrics.cvar_95:.4f}")
            print(f"  Win rate: {metrics.win_rate:.4f}")

        return results


def load_strategy_results(results_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load strategy results from JSON file.

    Returns:
        outcomes: Binary array of bet outcomes
        bet_sizes: Array of bet sizes
    """
    with open(results_path, 'r') as f:
        data = json.load(f)

    # Extract bet outcomes and sizes
    bets = data.get('bets_detail', data.get('bets', []))

    outcomes = []
    bet_sizes = []

    for bet in bets:
        # Determine outcome
        if 'bet_won' in bet:
            won = bet['bet_won']
            # Handle both boolean and numeric formats
            if isinstance(won, bool):
                outcome = 1 if won else 0
            else:
                outcome = 1 if won == 1.0 or won == True else 0
        elif 'won' in bet:
            outcome = 1 if bet['won'] else 0
        elif 'result' in bet:
            outcome = 1 if bet['result'] > 0 else 0
        else:
            # Try to infer from return
            ret = bet.get('return', 0)
            outcome = 1 if ret > 0 else 0

        # Determine bet size
        if 'bet_size' in bet:
            size = bet['bet_size']
        elif 'action' in bet:
            action = bet['action']
            if abs(action) == 1:
                size = 1.0
            elif abs(action) == 2:
                size = 2.0
            else:
                size = 5.0
        else:
            size = 1.0

        outcomes.append(outcome)
        bet_sizes.append(size)

    return np.array(outcomes), np.array(bet_sizes)


def main():
    """
    Run bootstrap stress tests on ensemble strategies.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Bootstrap stress testing')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to strategy results JSON')
    parser.add_argument('--output', type=str, default='results/simulation/stress_test.json',
                        help='Output path for stress test results')
    parser.add_argument('--n-trials', type=int, default=1000,
                        help='Number of bootstrap trials')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    print("=" * 80)
    print("Task 8: Bootstrap Stress Testing")
    print("=" * 80)

    # Load strategy results
    print(f"\nLoading strategy results from {args.results}...")
    outcomes, bet_sizes = load_strategy_results(args.results)

    print(f"  Loaded {len(outcomes)} bets")
    print(f"  Win rate: {outcomes.mean():.3f}")
    print(f"  Total bets: {bet_sizes.sum():.1f} units")

    # Run stress tests
    tester = BootstrapStressTest(random_seed=args.seed)

    results = tester.stress_test_scenarios(
        outcomes=outcomes,
        bet_sizes=bet_sizes,
        n_trials=args.n_trials,
    )

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        scenario_name: asdict(metrics)
        for scenario_name, metrics in results.items()
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Stress test complete! Results saved to {args.output}")
    print(f"{'=' * 80}")

    # Summary table
    print("\n" + "=" * 120)
    print(f"{'Scenario':<30} {'Mean Ret':<10} {'Sharpe':<10} {'Sortino':<10} {'Max DD':<10} {'VaR(95%)':<12} {'CVaR(95%)':<12}")
    print("=" * 120)

    for scenario_name, metrics in results.items():
        print(f"{scenario_name:<30} "
              f"{metrics.mean_return:>9.4f} "
              f"{metrics.sharpe_ratio:>9.4f} "
              f"{metrics.sortino_ratio:>9.4f} "
              f"{metrics.max_drawdown:>9.4f} "
              f"{metrics.var_95:>11.4f} "
              f"{metrics.cvar_95:>11.4f}")

    print("=" * 120)


if __name__ == '__main__':
    main()
