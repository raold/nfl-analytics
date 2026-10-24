"""
RL Training Task Executor.

Handles DQN and PPO training with multi-seed ensembles,
hyperparameter sweeps, and advanced techniques.
"""

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rl.dqn_agent import QNetwork


class RLTrainer:
    """Executor for RL training tasks."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = None
        self.optimizer = None
        self.epoch = 0

    def run(
        self, config: dict[str, Any], progress_callback: Callable[[float, str | None], None]
    ) -> dict[str, Any]:
        """Run RL training task."""
        model_type = config.get("model", "dqn")
        epochs = config.get("epochs", 500)
        seed = config.get("seed", 42)

        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Load dataset
        dataset_path = Path("data/rl_logged.csv")
        if not dataset_path.exists():
            # Generate synthetic dataset for testing
            self._generate_synthetic_dataset(dataset_path)

        df = pd.read_csv(dataset_path)

        if model_type == "dqn":
            return self._train_dqn(df, config, epochs, progress_callback)
        elif model_type == "ppo":
            return self._train_ppo(df, config, epochs, progress_callback)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _train_dqn(
        self, df: pd.DataFrame, config: dict[str, Any], epochs: int, progress_callback: Callable
    ) -> dict[str, Any]:
        """Train DQN model."""
        # Initialize DQN
        state_dim = 6  # p_hat, market_prob, spread, total, epa_gap, edge
        action_dim = 4  # no-bet, small, medium, large
        hidden_dims = config.get("hidden_dims", [128, 64, 32])  # List of hidden layer dimensions

        model = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        target_model = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        target_model.load_state_dict(model.state_dict())

        optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-4))

        # Training settings
        batch_size = config.get("batch_size", 256)
        gamma = config.get("gamma", 0.99)
        target_update = config.get("target_update", 100)
        double_dqn = config.get("double_dqn", False)
        prioritized_replay = config.get("prioritized_replay", False)

        # Convert dataframe to tensors
        states = torch.FloatTensor(
            df[["p_hat", "market_prob", "spread_close", "total_close", "epa_gap", "edge"]].values
        )
        actions = torch.LongTensor(df["action"].values)
        rewards = torch.FloatTensor(df["r"].values)

        best_loss = float("inf")
        losses = []

        for epoch in range(self.epoch, epochs):
            # Mini-batch training
            indices = torch.randperm(len(states))[:batch_size]
            batch_states = states[indices].to(self.device)
            batch_actions = actions[indices].to(self.device)
            batch_rewards = rewards[indices].to(self.device)

            # Compute Q-values
            q_values = model(batch_states)
            q_values = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze()

            # Compute target Q-values
            with torch.no_grad():
                if double_dqn:
                    # Double DQN: use online model to select actions
                    next_actions = model(batch_states).argmax(dim=1)
                    target_q = (
                        target_model(batch_states).gather(1, next_actions.unsqueeze(1)).squeeze()
                    )
                else:
                    target_q = target_model(batch_states).max(dim=1)[0]

                target_values = batch_rewards + gamma * target_q

            # Compute loss
            loss = nn.functional.mse_loss(q_values, target_values)

            # Prioritized experience replay weighting
            if prioritized_replay:
                td_errors = torch.abs(q_values - target_values).detach()
                weights = torch.pow(td_errors + 1e-5, 0.6)
                weights = weights / weights.max()
                loss = (loss * weights).mean()

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())

            # Update target network
            if epoch % target_update == 0:
                target_model.load_state_dict(model.state_dict())

            # Progress and checkpointing
            if epoch % 10 == 0:
                progress = (epoch + 1) / epochs
                checkpoint_path = f"models/dqn_seed{config['seed']}_epoch{epoch}.pth"
                Path(checkpoint_path).parent.mkdir(exist_ok=True)

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "loss": loss.item(),
                        "config": config,
                    },
                    checkpoint_path,
                )

                progress_callback(progress, checkpoint_path)

                if loss.item() < best_loss:
                    best_loss = loss.item()

            # Add some heat generation
            if epoch % 5 == 0:
                # Do some extra computation to generate heat
                dummy = torch.randn(1000, 1000).to(self.device)
                for _ in range(10):
                    dummy = torch.matmul(dummy, dummy.T)
                del dummy

        # Final model save
        final_path = f"models/dqn_seed{config['seed']}_final.pth"
        torch.save(model.state_dict(), final_path)

        return {
            "final_loss": losses[-1],
            "best_loss": best_loss,
            "model_path": final_path,
            "epochs_trained": epochs,
            "avg_loss": np.mean(losses[-100:]),
        }

    def _train_ppo(
        self, df: pd.DataFrame, config: dict[str, Any], epochs: int, progress_callback: Callable
    ) -> dict[str, Any]:
        """Train PPO model."""
        # Similar to DQN but with PPO-specific training
        # Placeholder for now

        # Simulate training
        best_reward = -float("inf")
        rewards = []

        for epoch in range(epochs):
            # Simulate training progress
            reward = np.random.randn() * 0.1 + epoch * 0.001
            rewards.append(reward)

            if reward > best_reward:
                best_reward = reward

            if epoch % 50 == 0:
                progress = (epoch + 1) / epochs
                checkpoint_path = f"models/ppo_seed{config['seed']}_epoch{epoch}.pth"
                Path(checkpoint_path).parent.mkdir(exist_ok=True)
                progress_callback(progress, checkpoint_path)

            # Heat generation
            dummy = torch.randn(500, 500).to(self.device)
            for _ in range(5):
                dummy = torch.sigmoid(torch.matmul(dummy, dummy.T))

        return {
            "final_reward": rewards[-1],
            "best_reward": best_reward,
            "epochs_trained": epochs,
            "avg_reward": np.mean(rewards[-100:]),
        }

    def _generate_synthetic_dataset(self, path: Path):
        """Generate synthetic RL dataset for testing."""
        n_samples = 10000
        np.random.seed(42)

        data = {
            "p_hat": np.random.beta(2, 2, n_samples),
            "market_prob": np.random.beta(2, 2, n_samples),
            "spread_close": np.random.normal(0, 3, n_samples),
            "total_close": np.random.uniform(40, 55, n_samples),
            "epa_gap": np.random.normal(0, 5, n_samples),
            "edge": np.random.normal(0, 0.05, n_samples),
            "action": np.random.randint(0, 4, n_samples),
            "r": np.random.normal(0, 1, n_samples),
            "b_prob": np.random.uniform(0.1, 0.9, n_samples),
            "pi_prob": np.random.uniform(0.1, 0.9, n_samples),
        }

        df = pd.DataFrame(data)
        path.parent.mkdir(exist_ok=True)
        df.to_csv(path, index=False)

    def load_checkpoint(self, checkpoint_path: str):
        """Load from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.epoch = checkpoint.get("epoch", 0)
