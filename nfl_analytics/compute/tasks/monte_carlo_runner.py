"""
Monte Carlo Simulation Task Executor.

Runs large-scale simulations for risk metrics, CVaR optimization,
and confidence intervals.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import stats


class MonteCarloRunner:
    """Executor for Monte Carlo simulation tasks."""

    def __init__(self):
        self.checkpoint_iter = 0

    def run(
        self, config: dict[str, Any], progress_callback: Callable[[float, str | None], None]
    ) -> dict[str, Any]:
        """Run Monte Carlo simulations."""
        n_scenarios = config.get("n_scenarios", 100000)
        use_qmc = config.get("use_qmc", False)
        metrics = config.get("metrics", ["var", "cvar"])
        alpha = config.get("alpha", 0.95)

        # Generate scenarios
        scenarios = self._generate_scenarios(n_scenarios, use_qmc, progress_callback)

        # Calculate metrics
        results = {}
        for metric in metrics:
            if metric == "var":
                results["var"] = self._calculate_var(scenarios, alpha)
            elif metric == "cvar":
                results["cvar"] = self._calculate_cvar(scenarios, alpha)
            elif metric == "max_drawdown":
                results["max_drawdown"] = self._calculate_max_drawdown(scenarios)
            elif metric == "sharpe":
                results["sharpe"] = self._calculate_sharpe(scenarios)

        # Bootstrap confidence intervals
        if config.get("bootstrap", True):
            ci_results = self._bootstrap_confidence_intervals(
                scenarios, metrics, alpha, n_bootstrap=1000
            )
            results["confidence_intervals"] = ci_results

        return results

    def _generate_scenarios(
        self, n_scenarios: int, use_qmc: bool, progress_callback: Callable
    ) -> np.ndarray:
        """Generate Monte Carlo or Quasi-Monte Carlo scenarios."""
        # Simulate portfolio returns over time
        n_periods = 100  # 100 weeks
        n_assets = 10  # 10 different bet types

        if use_qmc:
            # Quasi-Monte Carlo using Sobol sequence
            from scipy.stats import qmc

            sampler = qmc.Sobol(d=n_assets * n_periods, scramble=True)
            samples = sampler.random(n_scenarios)
            # Transform uniform to normal
            scenarios = stats.norm.ppf(samples).reshape(n_scenarios, n_periods, n_assets)
        else:
            # Standard Monte Carlo
            scenarios = np.zeros((n_scenarios, n_periods, n_assets))

            batch_size = 10000
            for i in range(0, n_scenarios, batch_size):
                end_idx = min(i + batch_size, n_scenarios)
                batch_size_actual = end_idx - i

                # Generate returns with correlation
                mean_returns = np.random.uniform(-0.02, 0.05, n_assets)
                cov_matrix = self._generate_correlation_matrix(n_assets)

                for period in range(n_periods):
                    scenarios[i:end_idx, period, :] = np.random.multivariate_normal(
                        mean_returns, cov_matrix * 0.01, batch_size_actual
                    )

                # Heavy computation for heat generation
                dummy = np.random.randn(1000, 1000)
                for _ in range(5):
                    dummy = dummy @ dummy.T
                    dummy = np.tanh(dummy)

                progress = (i + batch_size_actual) / n_scenarios * 0.8  # 80% for generation
                progress_callback(progress, None)

        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + scenarios, axis=1)
        portfolio_values = cumulative_returns.sum(axis=2)  # Equal weight portfolio

        return portfolio_values

    def _generate_correlation_matrix(self, n_assets: int) -> np.ndarray:
        """Generate positive semi-definite correlation matrix."""
        # Random correlation with structure
        A = np.random.randn(n_assets, n_assets)
        corr = A @ A.T
        D = np.diag(1.0 / np.sqrt(np.diag(corr)))
        corr = D @ corr @ D
        return corr

    def _calculate_var(self, scenarios: np.ndarray, alpha: float) -> float:
        """Calculate Value at Risk."""
        return np.percentile(scenarios[:, -1], (1 - alpha) * 100)

    def _calculate_cvar(self, scenarios: np.ndarray, alpha: float) -> float:
        """Calculate Conditional Value at Risk."""
        var = self._calculate_var(scenarios, alpha)
        return scenarios[scenarios[:, -1] <= var, -1].mean()

    def _calculate_max_drawdown(self, scenarios: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        drawdowns = []
        for scenario in scenarios:
            peaks = np.maximum.accumulate(scenario)
            drawdown = (scenario - peaks) / peaks
            drawdowns.append(drawdown.min())
        return np.mean(drawdowns)

    def _calculate_sharpe(self, scenarios: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        returns = np.diff(scenarios, axis=1) / scenarios[:, :-1]
        mean_returns = returns.mean(axis=1).mean()
        std_returns = returns.std(axis=1).mean()
        return mean_returns / std_returns if std_returns > 0 else 0

    def _bootstrap_confidence_intervals(
        self, scenarios: np.ndarray, metrics: list, alpha: float, n_bootstrap: int = 1000
    ) -> dict[str, tuple]:
        """Calculate bootstrap confidence intervals."""
        results = {}
        n_scenarios = len(scenarios)

        for metric in metrics:
            bootstrap_values = []

            for _ in range(n_bootstrap):
                # Resample scenarios
                indices = np.random.choice(n_scenarios, n_scenarios, replace=True)
                resampled = scenarios[indices]

                if metric == "var":
                    value = self._calculate_var(resampled, alpha)
                elif metric == "cvar":
                    value = self._calculate_cvar(resampled, alpha)
                elif metric == "max_drawdown":
                    value = self._calculate_max_drawdown(resampled)
                elif metric == "sharpe":
                    value = self._calculate_sharpe(resampled)
                else:
                    continue

                bootstrap_values.append(value)

            # Calculate confidence intervals
            lower = np.percentile(bootstrap_values, 2.5)
            upper = np.percentile(bootstrap_values, 97.5)
            results[metric] = (lower, upper)

        return results

    def load_checkpoint(self, checkpoint_path: str):
        """Load from checkpoint."""
        # Monte Carlo can resume from iteration number
        import json

        try:
            with open(checkpoint_path) as f:
                data = json.load(f)
                self.checkpoint_iter = data.get("iteration", 0)
        except Exception:
            self.checkpoint_iter = 0
