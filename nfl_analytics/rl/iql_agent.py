#!/usr/bin/env python3
"""
Implicit Q-Learning (IQL) agent for NFL betting with offline RL.

IQL improves on CQL by:
1. Learning V(s) separately from Q(s, a) → more stable value estimation
2. Implicit policy extraction via advantage-weighted regression → no explicit actor
3. Expectile regression for V(s) → learns upper-tail of value distribution (optimism)
4. Avoids distributional shift via in-sample learning (like CQL)

Key differences from CQL:
- CQL: penalizes Q-values on OOD actions
- IQL: learns Q and V separately, uses expectile for V, extracts policy via AWR

Architecture:
- Q-network: Q(s, a) for all actions
- V-network: V(s) scalar value
- Policy extraction: π(a|s) ∝ exp(β * [Q(s,a) - V(s)])

Loss components:
1. V-loss: expectile regression on TD target
2. Q-loss: standard MSE on Q(s, a) vs [r + γ V(s')]
3. Policy loss (optional): advantage-weighted behavioral cloning

Hyperparameters:
- expectile (τ): 0.7-0.95 (higher = more optimistic V estimates)
- temperature (β): 0.1-10.0 (for policy extraction from advantage)
- lr_v, lr_q: learning rates for V and Q networks
- hidden_dims: network architecture

Usage:
    # Train IQL agent
    python py/rl/iql_agent.py \\
        --dataset data/rl_logged.csv \\
        --output models/iql_model.pth \\
        --expectile 0.9 \\
        --temperature 3.0 \\
        --lr 1e-4 \\
        --epochs 500 \\
        --device cuda

References:
    Kostrikov et al. (2021): "Offline Reinforcement Learning with Implicit Q-Learning"
    https://arxiv.org/abs/2110.06169
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path

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
# Network Architectures
# ============================================================================


class QNetwork(nn.Module):
    """Q-network: Q(s, a) for all actions."""

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
        return self.network(state)


class VNetwork(nn.Module):
    """V-network: V(s) scalar value function."""

    def __init__(self, state_dim: int, hidden_dims: list[int] = None):
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
        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).squeeze(-1)


# ============================================================================
# Replay Buffer
# ============================================================================


class ReplayBuffer:
    """Fixed-size buffer for offline RL."""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
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
# IQL Agent
# ============================================================================


class IQLAgent:
    """
    Implicit Q-Learning (IQL) agent.

    Key components:
    1. Q-network: Q(s, a)
    2. V-network: V(s) learned via expectile regression
    3. Implicit policy: π(a|s) ∝ exp(β * [Q(s,a) - V(s)])

    Hyperparameters:
    - expectile (τ): 0.7-0.95, controls V(s) optimism (0.5 = mean, 1.0 = max)
    - temperature (β): for policy extraction from advantage
    - gamma: discount factor
    - lr_v, lr_q: separate learning rates for V and Q
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        device: torch.device,
        expectile: float = 0.9,
        temperature: float = 3.0,
        gamma: float = 0.99,
        lr_v: float = 3e-4,
        lr_q: float = 3e-4,
        batch_size: int = 128,
        buffer_capacity: int = 100000,
        hidden_dims: list[int] = None,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.device = device
        self.expectile = expectile
        self.temperature = temperature
        self.gamma = gamma
        self.batch_size = batch_size

        # Networks
        self.q_network = QNetwork(state_dim, n_actions, hidden_dims).to(device)
        self.v_network = VNetwork(state_dim, hidden_dims).to(device)

        # Target networks (for stability)
        self.q_target = QNetwork(state_dim, n_actions, hidden_dims).to(device)
        self.q_target.load_state_dict(self.q_network.state_dict())
        self.q_target.eval()

        # Optimizers
        self.optimizer_v = optim.Adam(self.v_network.parameters(), lr=lr_v)
        self.optimizer_q = optim.Adam(self.q_network.parameters(), lr=lr_q)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # Training step counter
        self.train_step = 0

    def expectile_loss(self, diff: torch.Tensor, expectile: float) -> torch.Tensor:
        """
        Expectile regression loss.

        For expectile τ:
        - If diff > 0: weight = τ
        - If diff < 0: weight = (1 - τ)

        This asymmetric loss makes V(s) learn the τ-expectile of the return distribution.
        For τ > 0.5, V(s) is optimistic (upper tail).
        """
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        return (weight * diff**2).mean()

    def update_v(
        self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor
    ) -> tuple[float, float]:
        """
        Update V-network using expectile regression.

        Target: Q(s, a) (from current Q network)
        Loss: expectile_loss(Q(s, a) - V(s))

        This makes V(s) track the τ-expectile of Q-values on dataset actions.
        """
        # Get V(s)
        v_pred = self.v_network(states)

        # Get Q(s, a) for dataset actions
        with torch.no_grad():
            q_values = self.q_network(states)
            q_dataset = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Expectile loss: V(s) should match τ-expectile of Q(s, a)
        diff = q_dataset - v_pred
        v_loss = self.expectile_loss(diff, self.expectile)

        # Backprop
        self.optimizer_v.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.v_network.parameters(), max_norm=10.0)
        self.optimizer_v.step()

        return v_loss.item(), v_pred.mean().item()

    def update_q(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[float, float]:
        """
        Update Q-network using standard TD learning.

        Target: r + γ * V(s')
        Loss: MSE(Q(s, a), target)

        Note: uses V(s') as target (not max_a Q(s', a)), which is more stable.
        """
        # Get Q(s, a)
        q_values = self.q_network(states)
        q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get V(s') for TD target
        with torch.no_grad():
            v_next = self.v_network(next_states)
            q_target = rewards + self.gamma * v_next * (1 - dones)

        # Q loss
        q_loss = F.mse_loss(q_pred, q_target)

        # Backprop
        self.optimizer_q.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer_q.step()

        return q_loss.item(), q_pred.mean().item()

    def update(self) -> dict[str, float]:
        """Perform one gradient step (update both V and Q)."""
        if len(self.replay_buffer) < self.batch_size:
            return {"v_loss": 0.0, "q_loss": 0.0, "v_mean": 0.0, "q_mean": 0.0}

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Update V
        v_loss, v_mean = self.update_v(states, actions, next_states)

        # Update Q
        q_loss, q_mean = self.update_q(states, actions, rewards, next_states, dones)

        self.train_step += 1

        # Periodic target network update
        if self.train_step % 1000 == 0:
            self.q_target.load_state_dict(self.q_network.state_dict())

        return {
            "v_loss": v_loss,
            "q_loss": q_loss,
            "v_mean": v_mean,
            "q_mean": q_mean,
        }

    def select_action(self, state: np.ndarray) -> int:
        """
        Select action using implicit policy: π(a|s) ∝ exp(β * A(s, a))
        where A(s, a) = Q(s, a) - V(s)
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_t)
            v_value = self.v_network(state_t)

            # Advantage: A(s, a) = Q(s, a) - V(s)
            advantages = q_values - v_value

            # Policy: π(a|s) ∝ exp(β * A(s, a))
            logits = self.temperature * advantages
            probs = F.softmax(logits, dim=1)

            # Sample from policy (or take argmax for greedy)
            action = probs.argmax(dim=1).item()

            return action

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "v_network": self.v_network.state_dict(),
                "q_target": self.q_target.state_dict(),
                "optimizer_q": self.optimizer_q.state_dict(),
                "optimizer_v": self.optimizer_v.state_dict(),
                "train_step": self.train_step,
                "state_dim": self.state_dim,
                "n_actions": self.n_actions,
                "expectile": self.expectile,
                "temperature": self.temperature,
            },
            path,
        )

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.v_network.load_state_dict(checkpoint["v_network"])
        self.q_target.load_state_dict(checkpoint["q_target"])
        self.optimizer_q.load_state_dict(checkpoint["optimizer_q"])
        self.optimizer_v.load_state_dict(checkpoint["optimizer_v"])
        self.train_step = checkpoint["train_step"]
        if "expectile" in checkpoint:
            self.expectile = checkpoint["expectile"]
        if "temperature" in checkpoint:
            self.temperature = checkpoint["temperature"]


# ============================================================================
# Data Loading & Training
# ============================================================================


def load_dataset(csv_path: str, state_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load offline RL dataset from CSV."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=state_cols + ["action", "r"])

    states = df[state_cols].to_numpy(dtype=np.float32)

    # Map binary actions to 4-action space
    actions_binary = df["action"].to_numpy(dtype=int)
    edges = df["edge"].to_numpy(dtype=float) if "edge" in df.columns else np.zeros(len(df))
    actions = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        if actions_binary[i] == 0:
            actions[i] = 0
        else:
            edge_abs = abs(edges[i])
            if edge_abs < 0.03:
                actions[i] = 1
            elif edge_abs < 0.06:
                actions[i] = 2
            else:
                actions[i] = 3

    rewards = df["r"].to_numpy(dtype=np.float32)

    return states, actions, rewards


def populate_replay_buffer(
    agent: IQLAgent, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray
):
    """Populate replay buffer with offline dataset."""
    for i in range(len(states)):
        agent.replay_buffer.push(
            state=states[i],
            action=actions[i],
            reward=rewards[i],
            next_state=states[i],
            done=True,
        )


def train_iql(
    agent: IQLAgent,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    epochs: int = 500,
    batch_size: int = 128,
    log_freq: int = 10,
) -> list[dict[str, float]]:
    """Train IQL agent on offline dataset."""
    populate_replay_buffer(agent, states, actions, rewards)

    print(
        f"Training IQL (expectile={agent.expectile}, temperature={agent.temperature}) for {epochs} epochs..."
    )
    print(f"Device: {agent.device}, Batch size: {batch_size}, Buffer: {len(agent.replay_buffer)}")

    metrics_log = []

    for epoch in range(epochs):
        n_updates = max(1, len(agent.replay_buffer) // batch_size)
        epoch_v_losses = []
        epoch_q_losses = []
        epoch_v_means = []
        epoch_q_means = []

        for _ in range(n_updates):
            metrics = agent.update()
            epoch_v_losses.append(metrics["v_loss"])
            epoch_q_losses.append(metrics["q_loss"])
            epoch_v_means.append(metrics["v_mean"])
            epoch_q_means.append(metrics["q_mean"])

        avg_v_loss = np.mean(epoch_v_losses)
        avg_q_loss = np.mean(epoch_q_losses)
        avg_v = np.mean(epoch_v_means)
        avg_q = np.mean(epoch_q_means)

        metrics_log.append(
            {
                "epoch": epoch + 1,
                "v_loss": avg_v_loss,
                "q_loss": avg_q_loss,
                "v_mean": avg_v,
                "q_mean": avg_q,
            }
        )

        if (epoch + 1) % log_freq == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"V_loss: {avg_v_loss:.4f} | "
                f"Q_loss: {avg_q_loss:.4f} | "
                f"V_mean: {avg_v:.4f} | "
                f"Q_mean: {avg_q:.4f}"
            )

    return metrics_log


def evaluate_policy(
    agent: IQLAgent, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray
) -> dict[str, float]:
    """Evaluate learned IQL policy."""
    agent.q_network.eval()
    agent.v_network.eval()

    predicted_actions = []

    with torch.no_grad():
        for state in states:
            action = agent.select_action(state)
            predicted_actions.append(action)

    predicted_actions = np.array(predicted_actions)

    match_rate = (predicted_actions == actions).mean()
    action_dist = pd.Series(predicted_actions).value_counts(normalize=True).to_dict()
    policy_reward = np.mean(
        [rewards[i] if predicted_actions[i] == actions[i] else 0.0 for i in range(len(rewards))]
    )

    return {
        "match_rate": float(match_rate),
        "action_distribution": action_dist,
        "estimated_policy_reward": float(policy_reward),
        "logged_avg_reward": float(rewards.mean()),
    }


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="IQL agent for NFL betting (offline RL)")
    ap.add_argument("--dataset", required=True, help="Path to offline RL dataset CSV")
    ap.add_argument("--output", default="models/iql_model.pth", help="Output model path")
    ap.add_argument("--load", help="Load pre-trained model checkpoint")
    ap.add_argument("--evaluate", action="store_true", help="Evaluation mode")
    ap.add_argument("--epochs", type=int, default=500, help="Training epochs")
    ap.add_argument("--batch-size", type=int, default=128, help="Batch size")
    ap.add_argument("--lr-v", type=float, default=3e-4, help="Learning rate for V-network")
    ap.add_argument("--lr-q", type=float, default=3e-4, help="Learning rate for Q-network")
    ap.add_argument(
        "--expectile", type=float, default=0.9, help="Expectile for V-learning (0.7-0.95)"
    )
    ap.add_argument(
        "--temperature", type=float, default=3.0, help="Temperature for policy extraction"
    )
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

    states, actions, rewards = load_dataset(args.dataset, args.state_cols)
    state_dim = states.shape[1]
    n_actions = 4

    print(f"Dataset: {len(states)} samples, state_dim={state_dim}, n_actions={n_actions}")
    print(f"Action distribution: {pd.Series(actions).value_counts(normalize=True).to_dict()}")
    print(f"Mean reward: {rewards.mean():.4f}, Std reward: {rewards.std():.4f}")

    agent = IQLAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        device=device,
        expectile=args.expectile,
        temperature=args.temperature,
        gamma=args.gamma,
        lr_v=args.lr_v,
        lr_q=args.lr_q,
        batch_size=args.batch_size,
        hidden_dims=args.hidden_dims,
    )

    print(
        f"IQL hyperparameters: expectile={args.expectile}, temperature={args.temperature}, lr_v={args.lr_v}, lr_q={args.lr_q}"
    )

    if args.load:
        print(f"Loading checkpoint from {args.load}...")
        agent.load(args.load)

    if args.evaluate:
        print("\n=== Evaluation ===")
        eval_metrics = evaluate_policy(agent, states, actions, rewards)
        print(json.dumps(eval_metrics, indent=2))

        eval_path = Path(args.output).parent / "iql_eval.json"
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_path, "w") as f:
            json.dump(eval_metrics, f, indent=2)
        print(f"Evaluation report saved to {eval_path}")
    else:
        print("\n=== Training ===")
        metrics_log = train_iql(
            agent=agent,
            states=states,
            actions=actions,
            rewards=rewards,
            epochs=args.epochs,
            batch_size=args.batch_size,
            log_freq=args.log_freq,
        )

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        agent.save(args.output)
        print(f"\nModel saved to {args.output}")

        log_path = Path(args.output).parent / "iql_training_log.json"
        with open(log_path, "w") as f:
            json.dump(metrics_log, f, indent=2)
        print(f"Training log saved to {log_path}")

        print("\n=== Post-Training Evaluation ===")
        eval_metrics = evaluate_policy(agent, states, actions, rewards)
        print(json.dumps(eval_metrics, indent=2))


if __name__ == "__main__":
    main()
