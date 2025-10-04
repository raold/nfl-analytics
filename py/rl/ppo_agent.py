"""
Proximal Policy Optimization (PPO) agent for NFL betting.

Implements:
- PPO with clipped surrogate objective (Schulman et al., 2017)
- Actor-Critic architecture with shared feature extractor
- Continuous action space: bet fraction ∈ [0, kelly_max]
- Entropy regularization for exploration
- GAE (Generalized Advantage Estimation)
- MPS/CUDA/CPU device support

State representation:
  - Model probability (p_hat)
  - Market probability (market_prob)
  - Spread (spread_close)
  - Total (total_close)
  - EPA gap (epa_gap)
  - Edge (p_hat - market_prob)

Action:
  Continuous bet fraction ∈ [0, kelly_max], where kelly_max ≈ edge / (b - 1)
  - 0.0: No bet
  - >0.0: Bet fraction of bankroll

Reward:
  - PnL: +b*stake on win, -stake on loss
  - Optional CLV-shaped reward scaling

Usage:
  # Train from offline dataset
  python py/rl/ppo_agent.py --dataset data/rl_logged.csv --output models/ppo_model.pth --epochs 200 --device mps

  # Evaluate policy
  python py/rl/ppo_agent.py --dataset data/rl_logged_test.csv --load models/ppo_model.pth --evaluate --device mps
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


class ActorCritic(nn.Module):
    """
    Actor-Critic network with shared feature extractor.
    
    Actor: Outputs Beta distribution parameters (alpha, beta) for continuous action ∈ [0, 1]
    Critic: Outputs state value V(s)
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Actor head: outputs (alpha, beta) for Beta distribution
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # Ensure alpha > 0
        )
        self.actor_var = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # Ensure beta > 0
        )
        
        # Critic head: outputs V(s)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[Beta, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: (batch_size, state_dim)
        
        Returns:
            dist: Beta distribution over actions ∈ [0, 1]
            value: State value V(s), shape (batch_size, 1)
        """
        features = self.shared(state)
        
        # Actor: Beta(alpha, beta) distribution
        alpha = self.actor_mean(features) + 1.0  # Add 1 to avoid degenerate Beta(1,1)
        beta = self.actor_var(features) + 1.0
        dist = Beta(alpha, beta)
        
        # Critic: V(s)
        value = self.critic(features)
        
        return dist, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            state: (batch_size, state_dim)
            deterministic: If True, use mean of Beta distribution
        
        Returns:
            action: (batch_size, 1), values ∈ [0, 1]
            log_prob: (batch_size, 1)
            value: (batch_size, 1)
        """
        dist, value = self.forward(state)
        
        if deterministic:
            # Use mode of Beta distribution: (alpha - 1) / (alpha + beta - 2)
            alpha = dist.concentration1
            beta_param = dist.concentration0
            action = (alpha - 1) / (alpha + beta_param - 2)
            action = torch.clamp(action, 0.0, 1.0)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action, log_prob, value
    
    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log_prob and entropy for given state-action pairs.
        
        Args:
            state: (batch_size, state_dim)
            action: (batch_size, 1)
        
        Returns:
            log_prob: (batch_size, 1)
            value: (batch_size, 1)
            entropy: (batch_size, 1)
        """
        dist, value = self.forward(state)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_prob, value, entropy


class RolloutBuffer:
    """Storage for on-policy rollout data."""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def add(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def get(self):
        return (
            torch.stack(self.states),
            torch.stack(self.actions),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.stack(self.log_probs),
            torch.stack(self.values),
            torch.tensor(self.dones, dtype=torch.float32),
        )
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()


class PPOAgent:
    """
    PPO agent with clipped surrogate objective.
    
    Args:
        state_dim: Dimension of state space
        hidden_dim: Hidden layer size
        lr: Learning rate
        gamma: Discount factor
        epsilon: PPO clipping parameter
        value_coef: Coefficient for value loss
        entropy_coef: Coefficient for entropy bonus
        gae_lambda: GAE lambda parameter
        device: torch device
    """
    
    def __init__(
        self,
        state_dim: int = 6,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        gae_lambda: float = 0.95,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.device = torch.device(device)
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda
        
        self.policy = ActorCritic(state_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.buffer = RolloutBuffer()
        self.train_step = 0
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[float, float, float]:
        """
        Select action from policy.
        
        Args:
            state: (state_dim,)
            deterministic: If True, use mode of distribution
        
        Returns:
            action: bet fraction ∈ [0, 1]
            log_prob: log probability
            value: state value
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, value = self.policy.get_action(state_t, deterministic)
        
        return action.item(), log_prob.item(), value.item()
    
    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: (T,)
            values: (T,)
            dones: (T,)
            next_value: scalar
        
        Returns:
            advantages: (T,)
            returns: (T,)
        """
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        # Append next_value for bootstrap
        values_ext = torch.cat([values, torch.tensor([next_value])])
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value_t = 0
                last_gae = 0
            else:
                next_value_t = values_ext[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(self, n_epochs: int = 10, batch_size: int = 64):
        """
        Update policy using PPO.
        
        Args:
            n_epochs: Number of epochs to train on buffer
            batch_size: Mini-batch size
        """
        states, actions, rewards, old_log_probs, old_values, dones = self.buffer.get()
        
        # Compute GAE
        with torch.no_grad():
            _, next_value = self.policy.forward(states[-1].unsqueeze(0))
            next_value = next_value.item()
        
        advantages, returns = self.compute_gae(rewards, old_values.squeeze(), dones, next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        
        # PPO update
        dataset_size = states.shape[0]
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}
        n_updates = 0
        
        for epoch in range(n_epochs):
            indices = torch.randperm(dataset_size)
            
            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                batch_idx = indices[start:end]
                
                # Evaluate actions
                log_probs, values, entropy = self.policy.evaluate_actions(
                    states[batch_idx], actions[batch_idx]
                )
                
                # PPO clipped surrogate objective
                ratio = torch.exp(log_probs - old_log_probs[batch_idx])
                surr1 = ratio * advantages[batch_idx].unsqueeze(1)
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages[batch_idx].unsqueeze(1)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, returns[batch_idx].unsqueeze(1))
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                # Track metrics
                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.mean().item()
                metrics["total_loss"] += loss.item()
                n_updates += 1
        
        # Average metrics
        for key in metrics:
            metrics[key] /= n_updates
        
        self.train_step += 1
        self.buffer.clear()
        
        return metrics
    
    def save(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_step": self.train_step,
            "config": {
                "state_dim": self.state_dim,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "value_coef": self.value_coef,
                "entropy_coef": self.entropy_coef,
                "gae_lambda": self.gae_lambda,
            },
        }
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_step = checkpoint["train_step"]


def train_ppo_offline(
    agent: PPOAgent,
    dataset: pd.DataFrame,
    n_epochs: int = 200,
    rollout_length: int = 512,
    ppo_epochs: int = 10,
    batch_size: int = 64,
) -> List[Dict]:
    """
    Train PPO on offline dataset using episodic rollouts.
    
    Args:
        agent: PPOAgent instance
        dataset: DataFrame with columns [p_hat, market_prob, spread_close, total_close, epa_gap, edge, r, b_prob]
        n_epochs: Number of training epochs
        rollout_length: Length of rollout before PPO update
        ppo_epochs: Number of PPO update epochs per rollout
        batch_size: Mini-batch size for PPO update
    
    Returns:
        training_log: List of metrics per epoch
    """
    required_cols = ["p_hat", "market_prob", "spread_close", "total_close", "epa_gap", "r", "b_prob"]
    for col in required_cols:
        if col not in dataset.columns:
            raise ValueError(f"Dataset missing required column: {col}")
    
    # Extract features
    state_cols = ["p_hat", "market_prob", "spread_close", "total_close", "epa_gap"]
    if "edge" not in dataset.columns:
        dataset["edge"] = dataset["p_hat"] - dataset["market_prob"]
    state_cols.append("edge")
    
    states = dataset[state_cols].values
    rewards = dataset["r"].values
    net_odds = dataset["b_prob"].values  # Net odds for computing PnL
    
    n_samples = len(dataset)
    training_log = []
    
    for epoch in range(n_epochs):
        # Shuffle dataset
        indices = np.random.permutation(n_samples)
        
        epoch_rewards = []
        epoch_actions = []
        
        for i in range(0, n_samples, rollout_length):
            rollout_idx = indices[i:i + rollout_length]
            
            # Collect rollout
            for idx in rollout_idx:
                state = states[idx]
                logged_reward = rewards[idx]  # Use logged reward from dataset
                
                # Select action
                action, log_prob, value = agent.select_action(state, deterministic=False)
                
                # Use logged reward (offline RL - we don't recompute outcomes)
                reward = logged_reward * action  # Scale by action size
                
                # Store transition
                state_t = torch.FloatTensor(state)
                action_t = torch.tensor([action], dtype=torch.float32)
                log_prob_t = torch.tensor([log_prob], dtype=torch.float32)
                value_t = torch.tensor([value], dtype=torch.float32)
                done = (idx == rollout_idx[-1])
                
                agent.buffer.add(state_t, action_t, reward, log_prob_t, value_t, done)
                
                epoch_rewards.append(reward)
                epoch_actions.append(action)
            
            # PPO update
            if len(agent.buffer.states) >= batch_size:
                metrics = agent.update(n_epochs=ppo_epochs, batch_size=batch_size)
        
        # Log epoch metrics
        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
        avg_action = np.mean(epoch_actions) if epoch_actions else 0.0
        
        log_entry = {
            "epoch": epoch + 1,
            "avg_reward": float(avg_reward),
            "avg_action": float(avg_action),
            "train_step": agent.train_step,
        }
        if "metrics" in locals():
            log_entry.update(metrics)
        
        training_log.append(log_entry)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | Reward: {avg_reward:.4f} | Action: {avg_action:.4f}")
    
    return training_log


def evaluate_ppo(agent: PPOAgent, dataset: pd.DataFrame) -> Dict:
    """
    Evaluate PPO agent on dataset.
    
    Args:
        agent: PPOAgent instance
        dataset: DataFrame with columns [p_hat, market_prob, spread_close, total_close, epa_gap, edge, r, b_prob]
    
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    required_cols = ["p_hat", "market_prob", "spread_close", "total_close", "epa_gap", "r", "b_prob"]
    for col in required_cols:
        if col not in dataset.columns:
            raise ValueError(f"Dataset missing required column: {col}")
    
    # Extract features
    state_cols = ["p_hat", "market_prob", "spread_close", "total_close", "epa_gap"]
    if "edge" not in dataset.columns:
        dataset["edge"] = dataset["p_hat"] - dataset["market_prob"]
    state_cols.append("edge")
    
    states = dataset[state_cols].values
    rewards = dataset["r"].values
    
    total_pnl = 0.0
    total_bets = 0
    actions = []
    
    for i in range(len(dataset)):
        state = states[i]
        logged_reward = rewards[i]
        
        # Select action (deterministic)
        action, _, _ = agent.select_action(state, deterministic=True)
        actions.append(action)
        
        # Compute PnL using logged reward
        if action > 0.01:
            pnl = logged_reward * action
            total_pnl += pnl
            total_bets += 1
    
    avg_pnl = total_pnl / len(dataset) if len(dataset) > 0 else 0.0
    avg_action = np.mean(actions) if actions else 0.0
    bet_rate = total_bets / len(dataset) if len(dataset) > 0 else 0.0
    
    return {
        "total_pnl": float(total_pnl),
        "avg_pnl": float(avg_pnl),
        "total_bets": total_bets,
        "bet_rate": float(bet_rate),
        "avg_action": float(avg_action),
    }


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate PPO agent for NFL betting")
    parser.add_argument("--dataset", type=str, required=True, help="Path to logged dataset CSV")
    parser.add_argument("--output", type=str, default="models/ppo_model.pth", help="Output path for model")
    parser.add_argument("--load", type=str, help="Load model from checkpoint")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--rollout-length", type=int, default=512, help="Rollout length before PPO update")
    parser.add_argument("--ppo-epochs", type=int, default=10, help="PPO update epochs per rollout")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clipping parameter")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="Device")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate mode (no training)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    df = pd.read_csv(args.dataset)
    print(f"Loaded {len(df)} samples")
    
    # Initialize agent
    agent = PPOAgent(
        state_dim=6,
        lr=args.lr,
        gamma=args.gamma,
        epsilon=args.epsilon,
        device=args.device,
    )
    
    if args.load:
        print(f"Loading model from {args.load}...")
        agent.load(args.load)
    
    if args.evaluate:
        # Evaluate mode
        print("Evaluating agent...")
        metrics = evaluate_ppo(agent, df)
        print(f"\nEvaluation Results:")
        print(f"  Total PnL: {metrics['total_pnl']:.2f}")
        print(f"  Avg PnL per game: {metrics['avg_pnl']:.4f}")
        print(f"  Total bets: {metrics['total_bets']}")
        print(f"  Bet rate: {metrics['bet_rate']:.2%}")
        print(f"  Avg action: {metrics['avg_action']:.4f}")
        
        # Save metrics
        output_dir = Path(args.output).parent
        metrics_path = output_dir / "ppo_eval_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {metrics_path}")
    else:
        # Training mode
        print(f"Training PPO agent for {args.epochs} epochs on {args.device}...")
        training_log = train_ppo_offline(
            agent,
            df,
            n_epochs=args.epochs,
            rollout_length=args.rollout_length,
            ppo_epochs=args.ppo_epochs,
            batch_size=args.batch_size,
        )
        
        # Save model
        print(f"\nSaving model to {args.output}...")
        os.makedirs(Path(args.output).parent, exist_ok=True)
        agent.save(args.output)
        
        # Save training log
        log_path = Path(args.output).parent / "ppo_training_log.json"
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)
        print(f"Training log saved to {log_path}")
        
        print(f"\nTraining complete!")
        print(f"  Final avg reward: {training_log[-1]['avg_reward']:.4f}")
        print(f"  Final avg action: {training_log[-1]['avg_action']:.4f}")


if __name__ == "__main__":
    main()
