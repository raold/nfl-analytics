"""
Off-Policy Evaluation (OPE) gate evaluator.
"""

from collections.abc import Callable
from typing import Any

import numpy as np


class OPEEvaluator:
    """Executor for OPE gate evaluation."""

    def run(
        self, config: dict[str, Any], progress_callback: Callable[[float, str | None], None]
    ) -> dict[str, Any]:
        """Run OPE evaluation with bootstrap."""
        clip = config.get("clip", 10)
        shrink = config.get("shrink", 0.0)
        bootstrap_n = config.get("bootstrap_n", 5000)
        methods = config.get("methods", ["snis", "dr"])

        # Generate synthetic logged data
        n_samples = 10000
        np.random.seed(42)

        actions = np.random.randint(0, 4, n_samples)
        rewards = np.random.normal(0, 1, n_samples)
        behavior_probs = np.random.uniform(0.1, 0.9, n_samples)
        policy_probs = np.random.uniform(0.1, 0.9, n_samples)

        results = {}

        for method_idx, method in enumerate(methods):
            bootstrap_estimates = []

            for boot in range(bootstrap_n):
                # Bootstrap sample
                indices = np.random.choice(n_samples, n_samples, replace=True)
                boot_actions = actions[indices]
                boot_rewards = rewards[indices]
                boot_b_probs = behavior_probs[indices]
                boot_pi_probs = policy_probs[indices]

                if method == "snis":
                    estimate = self._snis(
                        boot_actions, boot_rewards, boot_b_probs, boot_pi_probs, clip, shrink
                    )
                elif method == "dr":
                    estimate = self._dr(
                        boot_actions, boot_rewards, boot_b_probs, boot_pi_probs, clip, shrink
                    )
                elif method == "wis":
                    estimate = self._wis(
                        boot_actions, boot_rewards, boot_b_probs, boot_pi_probs, clip
                    )
                elif method == "cwpdis":
                    estimate = self._cwpdis(
                        boot_actions, boot_rewards, boot_b_probs, boot_pi_probs, clip
                    )
                else:
                    estimate = 0

                bootstrap_estimates.append(estimate)

                # Heat generation every 100 iterations
                if boot % 100 == 0:
                    dummy = np.random.randn(300, 300)
                    for _ in range(10):
                        dummy = np.sin(dummy) @ np.cos(dummy.T)
                    del dummy

                # Progress
                if boot % 500 == 0:
                    progress = (method_idx * bootstrap_n + boot) / (len(methods) * bootstrap_n)
                    progress_callback(progress, None)

            # Calculate statistics
            estimates = np.array(bootstrap_estimates)
            results[method] = {
                "mean": np.mean(estimates),
                "std": np.std(estimates),
                "ci_lower": np.percentile(estimates, 2.5),
                "ci_upper": np.percentile(estimates, 97.5),
                "median": np.median(estimates),
            }

        # Gate decision
        dr_mean = results.get("dr", {}).get("mean", 0)
        dr_ci_lower = results.get("dr", {}).get("ci_lower", 0)
        results["accept"] = dr_mean > 0 and dr_ci_lower > -0.1
        results["stable"] = all(
            r.get("std", float("inf")) < 1.0 for r in results.values() if isinstance(r, dict)
        )

        return results

    def _snis(self, actions, rewards, b_probs, pi_probs, clip, shrink):
        """Self-normalized importance sampling."""
        weights = pi_probs / np.maximum(b_probs, 1e-6)
        weights = np.minimum(weights, clip)
        weights = weights * (1 - shrink) + shrink

        return np.sum(weights * rewards) / np.sum(weights)

    def _dr(self, actions, rewards, b_probs, pi_probs, clip, shrink):
        """Doubly robust estimator."""
        # Simplified DR (would use value function in practice)
        weights = pi_probs / np.maximum(b_probs, 1e-6)
        weights = np.minimum(weights, clip)

        # Baseline (simplified)
        baseline = rewards.mean()

        return np.mean(weights * (rewards - baseline)) + baseline

    def _wis(self, actions, rewards, b_probs, pi_probs, clip):
        """Weighted importance sampling."""
        weights = pi_probs / np.maximum(b_probs, 1e-6)
        weights = np.minimum(weights, clip)

        return np.mean(weights * rewards)

    def _cwpdis(self, actions, rewards, b_probs, pi_probs, clip):
        """Per-decision weighted importance sampling."""
        # Simplified version
        weights = pi_probs / np.maximum(b_probs, 1e-6)
        weights = np.minimum(weights, clip)
        cumulative_weights = np.cumprod(weights)

        return np.mean(cumulative_weights * rewards)

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        pass
