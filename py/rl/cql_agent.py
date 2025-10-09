"""
Conservative Q-Learning (CQL) agent for NFL betting with offline RL.

Implements:
- CQL with conservative penalty (Kumar et al., 2020)
- Prevents Q-value overestimation on unseen actions
- MPS (Apple Silicon) and CUDA device support
- Offline RL training from logged betting dataset
- Discrete action spaces: {no-bet, small-bet, medium-bet, large-bet}

Key difference from DQN:
  CQL adds a penalty to minimize Q-values on all actions, then maximize
  Q-values on dataset actions. This prevents overestimation on OOD actions.

  Loss = TD_loss + alpha * CQL_penalty
  CQL_penalty = E[log sum exp Q(s, a)] - E[Q(s, a_dataset)]

Hyperparameters:
- alpha: CQL penalty weight (0.1-10.0, higher = more conservative)
- lr: learning rate (1e-5 to 1e-3)
- layers: hidden layer sizes [4, 5, 6] → [[128, 64], [128, 64, 32], [256, 128, 64]]

Usage:
  # Train CQL agent
  python py/rl/cql_agent.py \\
      --dataset data/rl_logged.csv \\
      --output models/cql_model.pth \\
      --alpha 1.0 \\
      --lr 1e-4 \\
      --epochs 200 \\
      --device mps

  # Evaluate policy
  python py/rl/cql_agent.py \\
      --dataset data/rl_logged_test.csv \\
      --load models/cql_model.pth \\
      --evaluate

References:
  Kumar et al. (2020): "Conservative Q-Learning for Offline RL"
  https://arxiv.org/abs/2006.04779
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ============================================================================
# Utilities
# ============================================================================


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(device_arg: str = "auto") -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_arg)


# ============================================================================
# Q-Network Architecture
# ============================================================================


class QNetwork(nn.Module):
    """
    MLP for Q(s, a) estimation with LayerNorm for stability.

    Supports flexible architectures:
    - 4 layers: [128, 64]
    - 5 layers: [128, 64, 32]
    - 6 layers: [256, 128, 64]
    """

    def __init__(self, state_dim: int, n_actions: int, hidden_dims: list[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, n_actions))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return Q-values for all actions given state batch."""
        return self.network(state)


# ============================================================================
# Experience Replay Buffer
# ============================================================================


class ReplayBuffer:
    """
    Fixed-size buffer for offline RL. Stores (s, a, r, s', done) tuples.
    For one-step betting, s' and done are mostly unused (episodic = 1 step).
    """

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ):
        """Add experience tuple to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        """Sample random batch of experiences."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ============================================================================
# CQL Agent
# ============================================================================


class CQLAgent:
    """
    Conservative Q-Learning (CQL) agent for offline RL.

    Key innovation:
      Loss = TD_loss + alpha * CQL_penalty

    Where:
      TD_loss = standard DQN loss (Bellman error)
      CQL_penalty = log-sum-exp Q(s, a) - Q(s, a_dataset)

    This penalizes high Q-values on unseen actions, preventing overestimation.

    Hyperparameters:
    - alpha: CQL penalty weight (default 1.0)
    - gamma: discount factor (default 0.99, less relevant for 1-step)
    - lr: learning rate (default 1e-4)
    - batch_size: mini-batch size (default 128)
    - target_update_freq: sync target network every N steps (default 1000)
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        device: torch.device,
        alpha: float = 1.0,  # CQL penalty weight
        gamma: float = 0.99,
        lr: float = 1e-4,
        batch_size: int = 128,
        target_update_freq: int = 1000,
        buffer_capacity: int = 100000,
        hidden_dims: list[int] = None,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.device = device
        self.alpha = alpha  # CQL-specific
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Q-networks
        self.q_network = QNetwork(state_dim, n_actions, hidden_dims).to(device)
        self.target_network = QNetwork(state_dim, n_actions, hidden_dims).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # Training step counter
        self.train_step = 0

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Select action using epsilon-greedy policy.

        For offline RL, epsilon=0 (greedy).
        """
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_t)
            return q_values.argmax(dim=1).item()

    def compute_cql_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        q_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CQL penalty term.

        CQL_penalty = log-sum-exp(Q(s, a)) - Q(s, a_dataset)

        This pushes down Q-values on all actions, then pushes up Q-values
        on dataset actions, resulting in conservative estimates.
        """
        # log-sum-exp over actions: shape (batch_size,)
        logsumexp_q = torch.logsumexp(q_values, dim=1)

        # Q-values for dataset actions: shape (batch_size,)
        dataset_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # CQL penalty: E[logsumexp - dataset_q]
        cql_penalty = (logsumexp_q - dataset_q).mean()

        return cql_penalty

    def update(self) -> dict[str, float]:
        """
        Perform one gradient step on a mini-batch.

        Returns dict with loss components and diagnostics.
        """
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0, "td_loss": 0.0, "cql_loss": 0.0, "q_mean": 0.0}

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q-values: Q(s, a)
        q_values = self.q_network(states)
        q_current = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            q_next = self.target_network(next_states).max(dim=1)[0]
            q_target = rewards + self.gamma * q_next * (1 - dones)

        # TD loss (Bellman error)
        td_loss = F.smooth_l1_loss(q_current, q_target)

        # CQL penalty
        cql_loss = self.compute_cql_loss(states, actions, q_values)

        # Total loss
        loss = td_loss + self.alpha * cql_loss

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update target network periodically
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {
            "loss": loss.item(),
            "td_loss": td_loss.item(),
            "cql_loss": cql_loss.item(),
            "q_mean": q_current.mean().item(),
            "q_target_mean": q_target.mean().item(),
        }

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_step": self.train_step,
                "state_dim": self.state_dim,
                "n_actions": self.n_actions,
                "alpha": self.alpha,
            },
            path,
        )

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_step = checkpoint["train_step"]
        if "alpha" in checkpoint:
            self.alpha = checkpoint["alpha"]


# ============================================================================
# Data Loading
# ============================================================================


def load_dataset(csv_path: str, state_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load offline RL dataset from CSV.

    Returns:
        states: (N, state_dim) array
        actions: (N,) array (4-action space: 0=no-bet, 1=small, 2=medium, 3=large)
        rewards: (N,) array
    """
    df = pd.read_csv(csv_path)

    # Filter out rows with NaN in critical columns
    df = df.dropna(subset=state_cols + ["action", "r"])

    # Extract state features
    states = df[state_cols].to_numpy(dtype=np.float32)

    # Map binary actions to 4-action space based on edge magnitude
    # Simple heuristic: no-bet=0, small=1 (edge 0-0.03), medium=2 (0.03-0.06), large=3 (>0.06)
    actions_binary = df["action"].to_numpy(dtype=int)
    edges = df["edge"].to_numpy(dtype=float) if "edge" in df.columns else np.zeros(len(df))
    actions = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        if actions_binary[i] == 0:
            actions[i] = 0  # no-bet
        else:
            edge_abs = abs(edges[i])
            if edge_abs < 0.03:
                actions[i] = 1  # small bet
            elif edge_abs < 0.06:
                actions[i] = 2  # medium bet
            else:
                actions[i] = 3  # large bet

    rewards = df["r"].to_numpy(dtype=np.float32)

    return states, actions, rewards


def populate_replay_buffer(
    agent: CQLAgent, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray
):
    """
    Populate replay buffer with offline dataset.

    For one-step betting, next_state = state (unused), done = True.
    """
    for i in range(len(states)):
        agent.replay_buffer.push(
            state=states[i],
            action=actions[i],
            reward=rewards[i],
            next_state=states[i],  # one-step, so next_state unused
            done=True,  # episodic
        )


# ============================================================================
# Training Loop
# ============================================================================


def train_cql(
    agent: CQLAgent,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    epochs: int = 200,
    batch_size: int = 128,
    log_freq: int = 10,
) -> list[dict[str, float]]:
    """
    Train CQL agent on offline dataset.

    Returns list of training metrics per epoch.
    """
    # Populate replay buffer once
    populate_replay_buffer(agent, states, actions, rewards)

    print(f"Training CQL (alpha={agent.alpha}) for {epochs} epochs on {len(states)} samples...")
    print(
        f"Device: {agent.device}, Batch size: {batch_size}, Buffer size: {len(agent.replay_buffer)}"
    )

    metrics_log = []

    for epoch in range(epochs):
        # Multiple gradient steps per epoch
        n_updates = max(1, len(agent.replay_buffer) // batch_size)
        epoch_losses = []
        epoch_td_losses = []
        epoch_cql_losses = []
        epoch_q_means = []

        for _ in range(n_updates):
            metrics = agent.update()
            epoch_losses.append(metrics["loss"])
            epoch_td_losses.append(metrics["td_loss"])
            epoch_cql_losses.append(metrics["cql_loss"])
            epoch_q_means.append(metrics["q_mean"])

        avg_loss = np.mean(epoch_losses)
        avg_td_loss = np.mean(epoch_td_losses)
        avg_cql_loss = np.mean(epoch_cql_losses)
        avg_q = np.mean(epoch_q_means)

        metrics_log.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "td_loss": avg_td_loss,
            "cql_loss": avg_cql_loss,
            "q_mean": avg_q,
        })

        if (epoch + 1) % log_freq == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"TD: {avg_td_loss:.4f} | "
                f"CQL: {avg_cql_loss:.4f} | "
                f"Q_mean: {avg_q:.4f}"
            )

    return metrics_log


# ============================================================================
# Evaluation
# ============================================================================


def evaluate_policy(
    agent: CQLAgent, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray
) -> dict[str, float]:
    """
    Evaluate learned policy on dataset.

    Metrics:
    - Average Q-value
    - Policy match rate (% agreement with logged actions)
    - Estimated reward under greedy policy
    - Action distribution
    - Q-value statistics (conservative estimates should be lower)
    """
    agent.q_network.eval()

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

    # Match rate
    match_rate = (predicted_actions == actions).mean()

    # Action distribution
    action_dist = pd.Series(predicted_actions).value_counts(normalize=True).to_dict()

    # Estimate reward under greedy policy
    # Conservative: only count reward when action matches dataset
    policy_reward = np.mean(
        [rewards[i] if predicted_actions[i] == actions[i] else 0.0 for i in range(len(rewards))]
    )

    # Q-value statistics
    q_stats = {
        "mean": float(q_values_array.mean()),
        "std": float(q_values_array.std()),
        "min": float(q_values_array.min()),
        "max": float(q_values_array.max()),
        "q25": float(np.percentile(q_values_array, 25)),
        "q50": float(np.percentile(q_values_array, 50)),
        "q75": float(np.percentile(q_values_array, 75)),
    }

    return {
        "match_rate": float(match_rate),
        "avg_q_value": float(q_values_array.mean()),
        "q_value_stats": q_stats,
        "action_distribution": action_dist,
        "estimated_policy_reward": float(policy_reward),
        "logged_avg_reward": float(rewards.mean()),
    }


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CQL agent for NFL betting (offline RL)")
    ap.add_argument("--dataset", required=True, help="Path to offline RL dataset CSV")
    ap.add_argument("--output", default="models/cql_model.pth", help="Output model path")
    ap.add_argument("--load", help="Load pre-trained model checkpoint")
    ap.add_argument("--evaluate", action="store_true", help="Evaluation mode")
    ap.add_argument("--epochs", type=int, default=200, help="Training epochs")
    ap.add_argument("--batch-size", type=int, default=128, help="Batch size")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    ap.add_argument("--alpha", type=float, default=1.0, help="CQL penalty weight (0.1-10.0)")
    ap.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    ap.add_argument("--device", default="auto", help="Device: auto/cpu/cuda/mps")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument(
        "--hidden-dims", nargs="+", type=int, default=[128, 64, 32], help="Hidden layer dimensions"
    )
    ap.add_argument(
        "--state-cols",
        nargs="+",
        default=["spread_close", "total_close", "epa_gap", "market_prob", "p_hat", "edge"],
        help="State feature columns",
    )
    ap.add_argument("--log-freq", type=int, default=10, help="Logging frequency (epochs)")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    print(f"Device: {device}")
    print(f"Loading dataset from {args.dataset}...")

    # Load data
    states, actions, rewards = load_dataset(args.dataset, args.state_cols)
    state_dim = states.shape[1]
    n_actions = 4  # {no-bet, small, medium, large}

    print(f"Dataset: {len(states)} samples, state_dim={state_dim}, n_actions={n_actions}")
    print(f"Action distribution: {pd.Series(actions).value_counts(normalize=True).to_dict()}")
    print(f"Mean reward: {rewards.mean():.4f}, Std reward: {rewards.std():.4f}")

    # Initialize agent
    agent = CQLAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        device=device,
        alpha=args.alpha,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_dims=args.hidden_dims,
    )

    print(f"CQL hyperparameters: alpha={args.alpha}, lr={args.lr}, layers={args.hidden_dims}")

    # Load checkpoint if provided
    if args.load:
        print(f"Loading checkpoint from {args.load}...")
        agent.load(args.load)

    if args.evaluate:
        # Evaluation mode
        print("\n=== Evaluation ===")
        eval_metrics = evaluate_policy(agent, states, actions, rewards)
        print(json.dumps(eval_metrics, indent=2))

        # Save evaluation report
        eval_path = Path(args.output).parent / "cql_eval.json"
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_path, "w") as f:
            json.dump(eval_metrics, f, indent=2)
        print(f"Evaluation report saved to {eval_path}")
    else:
        # Training mode
        print("\n=== Training ===")
        metrics_log = train_cql(
            agent=agent,
            states=states,
            actions=actions,
            rewards=rewards,
            epochs=args.epochs,
            batch_size=args.batch_size,
            log_freq=args.log_freq,
        )

        # Save model
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        agent.save(args.output)
        print(f"\nModel saved to {args.output}")

        # Save training log
        log_path = Path(args.output).parent / "cql_training_log.json"
        with open(log_path, "w") as f:
            json.dump(metrics_log, f, indent=2)
        print(f"Training log saved to {log_path}")

        # Run quick evaluation
        print("\n=== Post-Training Evaluation ===")
        eval_metrics = evaluate_policy(agent, states, actions, rewards)
        print(json.dumps(eval_metrics, indent=2))


if __name__ == "__main__":
    main()
