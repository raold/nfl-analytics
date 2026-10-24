"""
State-space model parameter sweep executor.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import linalg
from scipy.stats import norm


class StateSpaceTrainer:
    """Executor for state-space model training."""

    def run(
        self, config: dict[str, Any], progress_callback: Callable[[float, str | None], None]
    ) -> dict[str, Any]:
        """Run state-space parameter sweep."""
        q = config.get("process_noise", 0.01)
        r = config.get("obs_noise", 15)
        seasons = config.get("seasons", [2020, 2021, 2022, 2023])
        use_smoother = config.get("smoother", False)
        time_varying_hfa = config.get("time_varying_hfa", False)

        results = {"q": q, "r": r, "log_likelihood": 0, "brier_score": 0, "accuracy": 0}

        # Simulate Kalman filter training
        n_teams = 32
        n_weeks = 18
        n_seasons = len(seasons)

        for season_idx, season in enumerate(seasons):
            # Initialize state (team strengths)
            theta = np.zeros(n_teams)
            P = np.eye(n_teams) * 10  # Initial uncertainty

            # Process noise
            Q = np.eye(n_teams) * q

            # Observation noise
            R_obs = r**2

            log_lik = 0
            predictions = []

            for week in range(n_weeks):
                # Simulate games for this week
                n_games = 16
                for game in range(n_games):
                    # Random matchup
                    home_team = np.random.randint(0, n_teams)
                    away_team = np.random.randint(0, n_teams)
                    if home_team == away_team:
                        continue

                    # Design matrix
                    H = np.zeros(n_teams)
                    H[home_team] = 1
                    H[away_team] = -1

                    # Add home field advantage
                    hfa = 3.0
                    if time_varying_hfa:
                        hfa = 3.0 + np.sin(week / n_weeks * 2 * np.pi) * 0.5

                    # Predict
                    y_pred = H @ theta + hfa
                    S = H @ P @ H.T + R_obs

                    # Simulate actual margin
                    y_actual = y_pred + np.random.normal(0, np.sqrt(R_obs))

                    # Calculate likelihood
                    log_lik += norm.logpdf(y_actual, y_pred, np.sqrt(S))

                    # Kalman update
                    K = P @ H.T / S
                    innovation = y_actual - y_pred
                    theta = theta + K * innovation
                    P = P - np.outer(K, K) * S

                    # Store prediction
                    win_prob = norm.cdf(y_pred / np.sqrt(S))
                    predictions.append(win_prob)

                # State evolution (add process noise)
                P = P + Q

                # Heavy computation for heat
                dummy = np.random.randn(500, 500)
                dummy = linalg.expm(dummy / 100)
                del dummy

                # Progress
                progress = (season_idx * n_weeks + week + 1) / (n_seasons * n_weeks)
                progress_callback(progress, None)

            # Calculate metrics
            predictions = np.array(predictions)
            outcomes = np.random.binomial(1, predictions)  # Simulated outcomes

            brier = np.mean((predictions - outcomes) ** 2)
            accuracy = np.mean((predictions > 0.5) == outcomes)

            results["log_likelihood"] += log_lik
            results["brier_score"] += brier
            results["accuracy"] += accuracy

        # Average over seasons
        results["log_likelihood"] /= n_seasons
        results["brier_score"] /= n_seasons
        results["accuracy"] /= n_seasons

        # Apply smoother if requested
        if use_smoother:
            # Simplified smoother (would be RTS in practice)
            results["smoothed"] = True

        return results

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        pass
