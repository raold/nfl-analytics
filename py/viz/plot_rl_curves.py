#!/usr/bin/env python3
"""
Plot RL training curves from training logs.

Generates learning curves showing reward vs. epoch with error bands for
DQN and PPO agents. Used for dissertation Figure in Chapter 5.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_training_log(log_path: Path) -> Dict:
    """Load training log JSON file."""
    with open(log_path) as f:
        return json.load(f)


def extract_epochs_and_rewards(log: Dict) -> Tuple[List[int], List[float]]:
    """
    Extract epoch numbers and rewards from training log.

    Assumes log structure:
    {
        "epochs": [...],
        "rewards": [...],
        ...
    }
    Or list of dicts:
    [
        {"epoch": 1, "reward": 0.1, ...},
        {"epoch": 2, "reward": 0.15, ...},
        ...
    ]
    """
    if isinstance(log, dict):
        if "epochs" in log and "rewards" in log:
            return log["epochs"], log["rewards"]
        elif "training_history" in log:
            history = log["training_history"]
            epochs = [h.get("epoch", i) for i, h in enumerate(history, 1)]
            rewards = [h.get("reward", 0.0) for h in history]
            return epochs, rewards
    elif isinstance(log, list):
        epochs = [entry.get("epoch", i) for i, entry in enumerate(log, 1)]
        rewards = [entry.get("reward", 0.0) for entry in log]
        return epochs, rewards

    raise ValueError("Unknown training log format")


def compute_rolling_stats(
    rewards: List[float],
    window: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute rolling mean and IQR bands.

    Returns:
        mean, lower_bound (Q1), upper_bound (Q3)
    """
    rewards_arr = np.array(rewards)
    n = len(rewards_arr)

    if n < window:
        window = max(n // 4, 1)

    mean = np.convolve(rewards_arr, np.ones(window)/window, mode='same')

    # For IQR, use percentiles
    lower = np.array([
        np.percentile(rewards_arr[max(0, i-window):min(n, i+window)], 25)
        for i in range(n)
    ])
    upper = np.array([
        np.percentile(rewards_arr[max(0, i-window):min(n, i+window)], 75)
        for i in range(n)
    ])

    return mean, lower, upper


def plot_learning_curves(
    dqn_log_path: Path,
    ppo_log_path: Path,
    output_path: Path,
    window: int = 50,
) -> None:
    """
    Create learning curves plot.

    Args:
        dqn_log_path: Path to DQN training log JSON
        ppo_log_path: Path to PPO training log JSON
        output_path: Path to save output PNG
        window: Rolling window size for smoothing
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib not installed. Install with: pip install matplotlib")
        sys.exit(1)

    # Load logs
    dqn_log = load_training_log(dqn_log_path)
    ppo_log = load_training_log(ppo_log_path)

    # Extract data
    dqn_epochs, dqn_rewards = extract_epochs_and_rewards(dqn_log)
    ppo_epochs, ppo_rewards = extract_epochs_and_rewards(ppo_log)

    # Compute stats
    dqn_mean, dqn_lower, dqn_upper = compute_rolling_stats(dqn_rewards, window)
    ppo_mean, ppo_lower, ppo_upper = compute_rolling_stats(ppo_rewards, window)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    # DQN
    ax.plot(dqn_epochs, dqn_mean, label='DQN', color='#2a6fbb', linewidth=2)
    ax.fill_between(
        dqn_epochs, dqn_lower, dqn_upper,
        color='#2a6fbb', alpha=0.2, label='DQN IQR'
    )

    # PPO
    ax.plot(ppo_epochs, ppo_mean, label='PPO', color='#d95f02', linewidth=2)
    ax.fill_between(
        ppo_epochs, ppo_lower, ppo_upper,
        color='#d95f02', alpha=0.2, label='PPO IQR'
    )

    # Labels and formatting
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Reward', fontsize=11)
    ax.set_title('Offline RL Training Curves (Median and IQR)', fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Tight layout
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved learning curves to: {output_path}")
    print(f"   DQN: {len(dqn_epochs)} epochs, final reward: {dqn_rewards[-1]:.4f}")
    print(f"   PPO: {len(ppo_epochs)} epochs, final reward: {ppo_rewards[-1]:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot RL learning curves from training logs"
    )
    parser.add_argument(
        "--dqn-log",
        type=Path,
        default=Path("models/dqn_training_log.json"),
        help="Path to DQN training log JSON",
    )
    parser.add_argument(
        "--ppo-log",
        type=Path,
        default=Path("models/ppo_training_log.json"),
        help="Path to PPO training log JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/dissertation/figures/rl_learning_curves.png"),
        help="Output PNG path",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="Rolling window size for smoothing (default: 50)",
    )

    args = parser.parse_args()

    # Check inputs exist
    if not args.dqn_log.exists():
        print(f"ERROR: DQN log not found: {args.dqn_log}")
        return 1

    if not args.ppo_log.exists():
        print(f"ERROR: PPO log not found: {args.ppo_log}")
        return 1

    # Generate plot
    try:
        plot_learning_curves(
            dqn_log_path=args.dqn_log,
            ppo_log_path=args.ppo_log,
            output_path=args.output,
            window=args.window,
        )
        return 0
    except Exception as e:
        print(f"ERROR: Failed to generate plot: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
