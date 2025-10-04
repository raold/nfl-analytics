"""
Deep Q-Network (DQN) agent for NFL betting with offline RL support.

Implements:
- DQN with experience replay and target networks (Mnih et al., 2015)
- MPS (Apple Silicon) and CUDA device support
- Offline RL training from logged dataset
- CLV-shaped reward options
- Discrete action spaces: {no-bet, small-bet, medium-bet, large-bet}

State representation:
  - Model probability (p_hat)
  - Market probability (market_prob)
  - Spread (spread_close)
  - Total (total_close)
  - EPA gap (epa_gap)
  - Edge (p_hat - market_prob)

Actions:
  0: No bet
  1: Small bet (e.g., 1% bankroll)
  2: Medium bet (e.g., 2.5% bankroll)
  3: Large bet (e.g., 5% bankroll)

Reward:
  - PnL: +b*stake on win, -stake on loss (b = net odds, e.g., 0.91 for -110)
  - CLV-shaped: scale by edge magnitude to emphasize +EV decisions

Usage:
  # Train from offline dataset
  python py/rl/dqn_agent.py --dataset data/rl_logged.csv --output models/dqn_model.pth --epochs 200 --device mps

  # Evaluate policy
  python py/rl/dqn_agent.py --dataset data/rl_logged_test.csv --load models/dqn_model.pth --evaluate --device mps
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


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
    Simple MLP for Q(s, a) estimation.
    
    Architecture: state_dim -> 128 -> 64 -> 32 -> n_actions
    Uses LayerNorm for stability (important for MPS).
    """
    def __init__(self, state_dim: int, n_actions: int, hidden_dims: List[int] = None):
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
    For one-step betting, s' and done are unused (episodic = 1 step).
    """
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add experience tuple to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample random batch of experiences."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


# ============================================================================
# DQN Agent
# ============================================================================

class DQNAgent:
    """
    DQN agent with target network and experience replay.
    
    Hyperparameters:
    - gamma: discount factor (0.99 default, but one-step betting → less relevant)
    - lr: learning rate (1e-4 default)
    - batch_size: mini-batch size (128 default)
    - target_update_freq: sync target network every N steps (1000 default)
    - epsilon: exploration rate (for online RL; offline uses logged actions)
    """
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        device: torch.device,
        gamma: float = 0.99,
        lr: float = 1e-4,
        batch_size: int = 128,
        target_update_freq: int = 1000,
        buffer_capacity: int = 100000,
        hidden_dims: List[int] = None
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.device = device
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
        
        For offline RL, epsilon=0 (greedy). For online RL, decay epsilon over time.
        """
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_t)
            return q_values.argmax(dim=1).item()
    
    def update(self) -> Dict[str, float]:
        """
        Perform one gradient step on a mini-batch.
        
        Returns dict with loss and other diagnostics.
        """
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0, "q_mean": 0.0}
        
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
        
        # Huber loss (more stable than MSE for outliers)
        loss = F.smooth_l1_loss(q_current, q_target)
        
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
            "q_mean": q_current.mean().item(),
            "q_target_mean": q_target.mean().item()
        }
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_step": self.train_step,
            "state_dim": self.state_dim,
            "n_actions": self.n_actions
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_step = checkpoint["train_step"]


# ============================================================================
# Data Loading
# ============================================================================

def load_dataset(csv_path: str, state_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load offline RL dataset from CSV.
    
    Returns:
        states: (N, state_dim) array
        actions: (N,) array (binary: 0=no-bet, 1=bet; will map to 4-action space)
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
    agent: DQNAgent, 
    states: np.ndarray, 
    actions: np.ndarray, 
    rewards: np.ndarray
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
            done=True  # episodic
        )


# ============================================================================
# Training Loop
# ============================================================================

def train_dqn(
    agent: DQNAgent,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    epochs: int = 200,
    batch_size: int = 128,
    log_freq: int = 10
) -> List[Dict[str, float]]:
    """
    Train DQN agent on offline dataset.
    
    Returns list of training metrics per epoch.
    """
    # Populate replay buffer once
    populate_replay_buffer(agent, states, actions, rewards)
    
    print(f"Training DQN for {epochs} epochs on {len(states)} samples...")
    print(f"Device: {agent.device}, Batch size: {batch_size}, Buffer size: {len(agent.replay_buffer)}")
    
    metrics_log = []
    
    for epoch in range(epochs):
        # Multiple gradient steps per epoch (simulate batch training)
        n_updates = max(1, len(agent.replay_buffer) // batch_size)
        epoch_losses = []
        epoch_q_means = []
        
        for _ in range(n_updates):
            metrics = agent.update()
            epoch_losses.append(metrics["loss"])
            epoch_q_means.append(metrics["q_mean"])
        
        avg_loss = np.mean(epoch_losses)
        avg_q = np.mean(epoch_q_means)
        
        metrics_log.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "q_mean": avg_q
        })
        
        if (epoch + 1) % log_freq == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Q_mean: {avg_q:.4f}")
    
    return metrics_log


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_policy(
    agent: DQNAgent,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate learned policy on dataset.
    
    Metrics:
    - Average Q-value
    - Policy match rate (% agreement with logged actions)
    - Average reward under greedy policy
    - Action distribution
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
    
    # Estimate reward under greedy policy (conservative: assume same reward for same state)
    # This is approximate since true counterfactual is unknown
    policy_reward = np.mean([rewards[i] if predicted_actions[i] == actions[i] else 0.0 
                             for i in range(len(rewards))])
    
    return {
        "match_rate": float(match_rate),
        "avg_q_value": float(q_values_array.mean()),
        "action_distribution": action_dist,
        "estimated_policy_reward": float(policy_reward),
        "logged_avg_reward": float(rewards.mean())
    }


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="DQN agent for NFL betting")
    ap.add_argument("--dataset", required=True, help="Path to offline RL dataset CSV")
    ap.add_argument("--output", default="models/dqn_model.pth", help="Output model path")
    ap.add_argument("--load", help="Load pre-trained model checkpoint")
    ap.add_argument("--evaluate", action="store_true", help="Evaluation mode")
    ap.add_argument("--epochs", type=int, default=200, help="Training epochs")
    ap.add_argument("--batch-size", type=int, default=128, help="Batch size")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    ap.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    ap.add_argument("--device", default="auto", help="Device: auto/cpu/cuda/mps")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 64, 32], 
                    help="Hidden layer dimensions")
    ap.add_argument("--state-cols", nargs="+", 
                    default=["spread_close", "total_close", "epa_gap", "market_prob", "p_hat", "edge"],
                    help="State feature columns")
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
    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        device=device,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_dims=args.hidden_dims
    )
    
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
        eval_path = Path(args.output).parent / "dqn_eval.json"
        with open(eval_path, "w") as f:
            json.dump(eval_metrics, f, indent=2)
        print(f"Evaluation report saved to {eval_path}")
    else:
        # Training mode
        print("\n=== Training ===")
        metrics_log = train_dqn(
            agent=agent,
            states=states,
            actions=actions,
            rewards=rewards,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Save model
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        agent.save(args.output)
        print(f"\nModel saved to {args.output}")
        
        # Save training log
        log_path = Path(args.output).parent / "dqn_training_log.json"
        with open(log_path, "w") as f:
            json.dump(metrics_log, f, indent=2)
        print(f"Training log saved to {log_path}")
        
        # Run quick evaluation
        print("\n=== Post-Training Evaluation ===")
        eval_metrics = evaluate_policy(agent, states, actions, rewards)
        print(json.dumps(eval_metrics, indent=2))


if __name__ == "__main__":
    main()
