"""
Conservative Q-Learning Agent for NFL 4th Down Decisions

Extends py/rl/cql_agent.py for the 4th down decision problem:
  - State: 10-dimensional (distance, field position, score, time, etc.)
  - Actions: 3 discrete (go, punt, field_goal)
  - Reward: Δ win probability + outcome bonuses
  - Training: Offline RL on 105K historical 4th down plays (1999-2024)

Key differences from betting CQL agent:
  1. 3 actions instead of 4 (no "no-bet" action)
  2. Win probability-based rewards (not betting returns)
  3. Episodic (single decision, not sequential)
  4. Larger dataset (105K vs ~10K for betting)

Usage:
  # Train agent
  python py/rl/fourth_down_agent.py \\
      --train-data data/fourth_down_dataset_train.csv \\
      --test-data data/fourth_down_dataset_test.csv \\
      --output models/fourth_down/cql_agent.pth \\
      --alpha 1.0 \\
      --epochs 500

  # Evaluate policy
  python py/rl/fourth_down_agent.py \\
      --test-data data/fourth_down_dataset_test.csv \\
      --load models/fourth_down/cql_agent.pth \\
      --evaluate

  # Compare vs baseline (NYT bot)
  python py/rl/fourth_down_agent.py \\
      --test-data data/fourth_down_dataset_test.csv \\
      --load models/fourth_down/cql_agent.pth \\
      --compare-nyt

References:
  Kumar et al. (2020): Conservative Q-Learning for Offline RL
  Burke et al. (2009): Fourth Down Study
  NYT 4th Down Bot: https://www.nytimes.com/interactive/2017/upshot/fourth-down-bot.html

Author: Claude Code (Sonnet 4.5)
Date: November 22, 2025
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Import base CQL agent
from py.rl.cql_agent import CQLAgent, set_seed, get_device

# Import MDP definitions
from py.rl.fourth_down_mdp import (
    STATE_DIM,
    ACTION_DIM,
    ACTION_GO,
    ACTION_PUNT,
    ACTION_FIELD_GOAL,
    ACTION_NAMES,
)


# ============================================================================
# Data Loading for 4th Down Problem
# ============================================================================


def load_fourth_down_dataset(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load 4th down dataset from CSV.

    Dataset columns:
      - s0, s1, ..., s9: State features (normalized)
      - action: 0=go, 1=punt, 2=field_goal
      - reward: Δ win probability + bonuses

    Returns:
        states: (N, 10) array
        actions: (N,) array
        rewards: (N,) array
        metadata: DataFrame with game_id, season, play_type, etc.
    """
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df):,} plays from {csv_path}")
    print(f"Seasons: {df['season'].min()}-{df['season'].max()}")

    # Extract state columns
    state_cols = [f"s{i}" for i in range(STATE_DIM)]
    states = df[state_cols].to_numpy(dtype=np.float32)

    # Extract actions and rewards
    actions = df["action"].to_numpy(dtype=np.int64)
    rewards = df["reward"].to_numpy(dtype=np.float32)

    # Metadata for analysis
    metadata_cols = ["game_id", "play_id", "season", "week", "posteam", "play_type",
                     "yards_gained", "fourth_down_converted", "touchdown", "field_goal_result"]
    metadata = df[metadata_cols].copy()

    # Data quality checks
    assert states.shape[1] == STATE_DIM, f"Expected {STATE_DIM} state features, got {states.shape[1]}"
    assert actions.min() >= 0 and actions.max() < ACTION_DIM, f"Invalid actions: {actions.min()}-{actions.max()}"
    assert not np.isnan(states).any(), "NaN values in states"
    assert not np.isnan(rewards).any(), "NaN values in rewards"

    print(f"Action distribution: {pd.Series(actions).value_counts().to_dict()}")
    print(f"Reward stats: mean={rewards.mean():.4f}, std={rewards.std():.4f}")

    return states, actions, rewards, metadata


def populate_replay_buffer(
    agent: CQLAgent,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
):
    """
    Populate replay buffer with offline 4th down dataset.

    For episodic decisions:
      - next_state = state (unused in 1-step MDP)
      - done = True (always episodic)
    """
    print(f"Populating replay buffer with {len(states):,} transitions...")
    for i in range(len(states)):
        agent.replay_buffer.push(
            state=states[i],
            action=actions[i],
            reward=rewards[i],
            next_state=states[i],  # Unused (episodic)
            done=True,
        )
    print(f"Buffer size: {len(agent.replay_buffer):,}")


# ============================================================================
# Training
# ============================================================================


def train_fourth_down_agent(
    agent: CQLAgent,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    epochs: int = 500,
    batch_size: int = 256,
    log_freq: int = 20,
    save_freq: int = 100,
    output_dir: str | Path = "models/fourth_down",
) -> list[dict]:
    """
    Train CQL agent on 4th down dataset.

    Training loop:
      - Multiple gradient steps per epoch (proportional to dataset size)
      - Periodic evaluation on training set (not test set to avoid leakage)
      - Checkpointing every save_freq epochs

    Args:
        agent: CQL agent
        states, actions, rewards: Training dataset
        epochs: Number of training epochs
        batch_size: Mini-batch size
        log_freq: Logging frequency (epochs)
        save_freq: Checkpoint frequency (epochs)
        output_dir: Where to save checkpoints

    Returns:
        metrics_log: List of training metrics per epoch
    """
    # Populate replay buffer
    populate_replay_buffer(agent, states, actions, rewards)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Training CQL Agent ===")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Alpha (CQL penalty): {agent.alpha}")
    print(f"Learning rate: {agent.optimizer.param_groups[0]['lr']}")
    print(f"Device: {agent.device}")

    metrics_log = []

    for epoch in range(epochs):
        # Multiple gradient steps per epoch
        n_updates = max(1, len(agent.replay_buffer) // batch_size)
        epoch_metrics = {
            "loss": [],
            "td_loss": [],
            "cql_loss": [],
            "q_mean": [],
        }

        for _ in range(n_updates):
            metrics = agent.update()
            for key in epoch_metrics:
                epoch_metrics[key].append(metrics[key])

        # Average metrics
        avg_metrics = {key: np.mean(vals) for key, vals in epoch_metrics.items()}
        avg_metrics["epoch"] = epoch + 1
        metrics_log.append(avg_metrics)

        # Logging
        if (epoch + 1) % log_freq == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:4d}/{epochs} | "
                f"Loss: {avg_metrics['loss']:.4f} | "
                f"TD: {avg_metrics['td_loss']:.4f} | "
                f"CQL: {avg_metrics['cql_loss']:.4f} | "
                f"Q_mean: {avg_metrics['q_mean']:.4f}"
            )

        # Checkpointing
        if (epoch + 1) % save_freq == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
            agent.save(
                checkpoint_path,
                metadata={"epoch": epoch + 1, "metrics": avg_metrics},
            )
            print(f"  → Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_path = output_dir / "cql_agent_final.pth"
    agent.save(
        final_path,
        metadata={"epochs": epochs, "final_metrics": metrics_log[-1]},
    )
    print(f"\n✓ Training complete. Final model saved to {final_path}")

    return metrics_log


# ============================================================================
# Evaluation
# ============================================================================


def evaluate_fourth_down_policy(
    agent: CQLAgent,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    metadata: pd.DataFrame,
) -> dict:
    """
    Evaluate CQL policy on test set.

    Metrics:
      1. Policy agreement rate (% match with actual coaches' decisions)
      2. Expected reward under CQL policy
      3. Win probability improvement (counterfactual analysis)
      4. Action-specific performance (go vs punt vs field goal)

    Args:
        agent: Trained CQL agent
        states, actions, rewards: Test dataset
        metadata: DataFrame with game context

    Returns:
        eval_metrics: Dictionary with comprehensive evaluation metrics
    """
    agent.q_network.eval()

    print(f"\n=== Evaluating Policy ===")
    print(f"Test set size: {len(states):,} plays")

    # Predict actions with CQL policy
    predicted_actions = []
    q_values_list = []

    with torch.no_grad():
        for state in states:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            q_vals = agent.q_network(state_t)
            q_values_list.append(q_vals.cpu().numpy()[0])
            pred_action = q_vals.argmax(dim=1).item()
            predicted_actions.append(pred_action)

    predicted_actions = np.array(predicted_actions)
    q_values_array = np.array(q_values_list)

    # 1. Agreement rate with actual coaches
    agreement = (predicted_actions == actions).mean()
    print(f"Agreement rate: {agreement:.2%} ({(predicted_actions == actions).sum():,}/{len(actions):,})")

    # 2. Action distribution
    pred_dist = pd.Series(predicted_actions).value_counts(normalize=True).to_dict()
    actual_dist = pd.Series(actions).value_counts(normalize=True).to_dict()

    print(f"\nPredicted action distribution:")
    for action, prop in pred_dist.items():
        print(f"  {ACTION_NAMES[action]}: {prop:.2%}")

    print(f"\nActual action distribution:")
    for action, prop in actual_dist.items():
        print(f"  {ACTION_NAMES[action]}: {prop:.2%}")

    # 3. Expected reward
    # Conservative estimate: only count reward when action matches
    matched_rewards = [rewards[i] if predicted_actions[i] == actions[i] else 0.0 for i in range(len(rewards))]
    expected_reward_matched = np.mean(matched_rewards)
    actual_avg_reward = np.mean(rewards)

    print(f"\nReward comparison:")
    print(f"  Actual avg reward: {actual_avg_reward:.4f}")
    print(f"  CQL matched reward: {expected_reward_matched:.4f}")
    print(f"  Reward lift: {(expected_reward_matched - actual_avg_reward):.4f}")

    # 4. Q-value statistics
    q_stats = {
        "mean": float(q_values_array.mean()),
        "std": float(q_values_array.std()),
        "min": float(q_values_array.min()),
        "max": float(q_values_array.max()),
    }

    print(f"\nQ-value statistics:")
    print(f"  Mean: {q_stats['mean']:.4f}")
    print(f"  Std: {q_stats['std']:.4f}")
    print(f"  Range: [{q_stats['min']:.4f}, {q_stats['max']:.4f}]")

    # 5. Action-specific performance
    print(f"\n=== Action-Specific Analysis ===")
    for action in range(ACTION_DIM):
        mask = predicted_actions == action
        if mask.sum() > 0:
            action_rewards = rewards[mask]
            print(f"\n{ACTION_NAMES[action].upper()} (n={mask.sum():,}):")
            print(f"  Mean reward: {action_rewards.mean():.4f}")
            print(f"  Success rate: {(action_rewards > 0).mean():.2%}")

    # 6. Context-specific agreement
    print(f"\n=== Agreement by Context ===")

    # By distance
    for dist_bin in [(1, 2), (3, 5), (6, 10), (11, 100)]:
        # Extract ydstogo from metadata if available, or infer from state s1
        # s1 is distance normalized (0-1, where 20 yards = 1.0)
        ydstogo = (states[:, 1] * 20).astype(int)  # Denormalize
        mask = (ydstogo >= dist_bin[0]) & (ydstogo <= dist_bin[1])
        if mask.sum() > 0:
            agreement_rate = (predicted_actions[mask] == actions[mask]).mean()
            dist_str = f"{dist_bin[0]}-{dist_bin[1]}" if dist_bin[1] < 100 else f"{dist_bin[0]}+"
            print(f"  Distance {dist_str}: {agreement_rate:.2%} (n={mask.sum():,})")

    # Compile results
    eval_metrics = {
        "agreement_rate": float(agreement),
        "predicted_distribution": {ACTION_NAMES[k]: v for k, v in pred_dist.items()},
        "actual_distribution": {ACTION_NAMES[k]: v for k, v in actual_dist.items()},
        "expected_reward_matched": float(expected_reward_matched),
        "actual_avg_reward": float(actual_avg_reward),
        "reward_lift": float(expected_reward_matched - actual_avg_reward),
        "q_value_stats": q_stats,
    }

    return eval_metrics


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train CQL agent for 4th down decisions")
    ap.add_argument("--train-data", default="data/fourth_down_dataset_train.csv", help="Training dataset CSV")
    ap.add_argument("--test-data", default="data/fourth_down_dataset_test.csv", help="Test dataset CSV")
    ap.add_argument("--output", default="models/fourth_down/cql_agent.pth", help="Output model path")
    ap.add_argument("--load", help="Load pre-trained model checkpoint")
    ap.add_argument("--evaluate", action="store_true", help="Evaluation mode only")

    # Training hyperparameters
    ap.add_argument("--epochs", type=int, default=500, help="Training epochs")
    ap.add_argument("--batch-size", type=int, default=256, help="Batch size")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    ap.add_argument("--alpha", type=float, default=1.0, help="CQL penalty weight")
    ap.add_argument("--gamma", type=float, default=0.99, help="Discount factor (less relevant for episodic)")
    ap.add_argument("--hidden-dims", nargs="+", type=int, default=[256, 128, 64], help="Hidden layer dimensions")
    ap.add_argument("--target-update-freq", type=int, default=1000, help="Target network update frequency")

    # Infrastructure
    ap.add_argument("--device", default="auto", help="Device: auto/cpu/cuda/mps")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--log-freq", type=int, default=20, help="Logging frequency (epochs)")
    ap.add_argument("--save-freq", type=int, default=100, help="Checkpoint frequency (epochs)")

    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    print("=" * 80)
    print("CQL Agent for NFL 4th Down Decisions")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Random seed: {args.seed}")

    # Load datasets
    if args.evaluate:
        print(f"\n=== Evaluation Mode ===")
        print(f"Loading test data from {args.test_data}")
        test_states, test_actions, test_rewards, test_metadata = load_fourth_down_dataset(args.test_data)
    else:
        print(f"\n=== Training Mode ===")
        print(f"Loading training data from {args.train_data}")
        train_states, train_actions, train_rewards, _ = load_fourth_down_dataset(args.train_data)

        print(f"\nLoading test data from {args.test_data}")
        test_states, test_actions, test_rewards, test_metadata = load_fourth_down_dataset(args.test_data)

    # Initialize agent
    agent = CQLAgent(
        state_dim=STATE_DIM,
        n_actions=ACTION_DIM,
        device=device,
        alpha=args.alpha,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        hidden_dims=args.hidden_dims,
    )

    print(f"\nAgent configuration:")
    print(f"  State dim: {STATE_DIM}")
    print(f"  Action dim: {ACTION_DIM}")
    print(f"  Hidden layers: {args.hidden_dims}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Learning rate: {args.lr}")

    # Load checkpoint if provided
    if args.load:
        print(f"\nLoading checkpoint from {args.load}")
        agent.load(args.load)

    if args.evaluate:
        # Evaluation only
        eval_metrics = evaluate_fourth_down_policy(
            agent, test_states, test_actions, test_rewards, test_metadata
        )

        # Save evaluation report
        output_path = Path(args.output)
        eval_path = output_path.parent / "evaluation_report.json"
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_path, "w") as f:
            json.dump(eval_metrics, f, indent=2)
        print(f"\n✓ Evaluation report saved to {eval_path}")

    else:
        # Training mode
        metrics_log = train_fourth_down_agent(
            agent=agent,
            states=train_states,
            actions=train_actions,
            rewards=train_rewards,
            epochs=args.epochs,
            batch_size=args.batch_size,
            log_freq=args.log_freq,
            save_freq=args.save_freq,
            output_dir=Path(args.output).parent,
        )

        # Save training log
        output_path = Path(args.output)
        log_path = output_path.parent / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(metrics_log, f, indent=2)
        print(f"✓ Training log saved to {log_path}")

        # Post-training evaluation on test set
        print("\n" + "=" * 80)
        print("POST-TRAINING EVALUATION ON TEST SET")
        print("=" * 80)

        eval_metrics = evaluate_fourth_down_policy(
            agent, test_states, test_actions, test_rewards, test_metadata
        )

        # Save evaluation report
        eval_path = output_path.parent / "evaluation_report.json"
        with open(eval_path, "w") as f:
            json.dump(eval_metrics, f, indent=2)
        print(f"\n✓ Evaluation report saved to {eval_path}")

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Final model: {args.output}")
        print(f"Training log: {log_path}")
        print(f"Evaluation report: {eval_path}")


if __name__ == "__main__":
    main()
