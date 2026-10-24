"""
Conservative Q-Learning (CQL) Feature Selector for NFL Analytics.

Learns optimal feature selection policies for NFL betting models using offline RL.

Key Innovation:
  CQL penalizes Q-values for out-of-distribution actions (untested feature combinations),
  preventing overestimation on novel feature sets. Critical for limited historical data.

Architecture:
  - State: Binary feature flags + performance metrics (MAE, calibration, etc.)
  - Actions: Toggle individual features or feature groups
  - Reward: Performance improvement (negative MAE change)
  - CQL Loss: L = TD_loss + α * (log Σ exp(Q) - Q(s, a_data))

Why CQL over DQN?
  - Only ~1000 historical games for validation
  - Novel feature combinations have high uncertainty
  - Conservative approach prefers proven feature sets
  - Prevents "fantasy" Q-values for untested features

Usage:
  # Training
  python nfl_analytics/rl/cql_feature_selector.py \\
      --dataset data/feature_performance.csv \\
      --output models/cql_feature_selector.pth \\
      --alpha 2.0 \\
      --epochs 500

  # Inference
  from nfl_analytics.rl.cql_feature_selector import CQLFeatureSelector
  agent = CQLFeatureSelector.load("models/cql_feature_selector.pth")
  feature_mask = agent.select_features(base_features, budget=100)

References:
  Kumar et al. (2020): "Conservative Q-Learning for Offline RL"
  https://arxiv.org/abs/2006.04779
"""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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


class FeatureQNetwork(nn.Module):
    """
    Q-Network for feature selection with attention mechanism.

    Architecture:
      Input: [feature_flags (N), performance_metrics (M), budget_state (1)]
      Hidden: Multi-layer MLP with LayerNorm
      Output: Q-values for each action (toggle feature i, toggle group j, etc.)

    Args:
        state_dim: Total state dimension (N + M + 1)
        n_actions: Number of discrete actions
        hidden_dims: List of hidden layer sizes (default: [256, 256])
        use_attention: Whether to use self-attention on feature flags
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: list[int] = None,
        use_attention: bool = False,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.use_attention = use_attention
        self.state_dim = state_dim
        self.n_actions = n_actions

        # Optional attention layer for feature flags
        if use_attention:
            self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
            self.feature_projection = nn.Linear(state_dim, 64)
            in_dim = 64
        else:
            in_dim = state_dim

        # MLP layers with LayerNorm for stability
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # Regularization
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, n_actions))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: state -> Q-values.

        Args:
            state: (batch_size, state_dim) tensor

        Returns:
            Q-values: (batch_size, n_actions) tensor
        """
        if self.use_attention:
            # Apply self-attention to feature flags
            x = self.feature_projection(state)
            x = x.unsqueeze(1)  # (batch, 1, embed_dim)
            attn_out, _ = self.attention(x, x, x)
            x = attn_out.squeeze(1)
        else:
            x = state

        return self.network(x)


# ============================================================================
# Prioritized Replay Buffer
# ============================================================================


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay for CQL feature selection.

    Prioritizes transitions with:
      1. Large reward (significant performance improvement)
      2. High TD error (large Bellman residual)
      3. Novel feature combinations (low visitation count)

    Args:
        capacity: Maximum buffer size
        alpha: Priority exponent (0 = uniform, 1 = full prioritization)
        beta: Importance sampling exponent (annealed from beta_start to 1.0)
    """

    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: float = None,
    ):
        """Add experience with priority."""
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        priority = priority if priority is not None else max_priority

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray
    ]:
        """Sample batch with importance sampling weights."""
        if self.size < batch_size:
            batch_size = self.size

        # Compute sampling probabilities
        priorities = self.priorities[: self.size]
        probs = priorities**self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)

        # Compute importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize for stability

        # Extract batch
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
            torch.FloatTensor(weights),
            indices,
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self) -> int:
        return self.size


# ============================================================================
# CQL Feature Selector Agent
# ============================================================================


class CQLFeatureSelector:
    """
    Conservative Q-Learning agent for feature selection.

    Key Innovation:
      CQL Loss = TD_loss + α * (log Σ exp(Q(s,a)) - Q(s, a_data))

      This penalizes high Q-values on all actions (conservative),
      then rewards Q-values on dataset actions (correct for bias).
      Result: Conservative estimates that prevent overoptimism on
      untested feature combinations.

    Args:
        state_dim: Dimension of state space (feature flags + metrics)
        action_dim: Number of discrete actions (toggle features)
        device: torch.device for computation
        alpha: CQL regularization strength (higher = more conservative)
        gamma: Discount factor (0.99 for multi-step episodes)
        lr: Learning rate
        batch_size: Mini-batch size for updates
        target_update_freq: Steps between target network updates
        buffer_capacity: Replay buffer size
        hidden_dims: Hidden layer dimensions
        use_attention: Whether to use attention mechanism
        prioritized_replay: Whether to use prioritized experience replay
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: torch.device = None,
        alpha: float = 1.0,
        gamma: float = 0.99,
        lr: float = 1e-4,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        buffer_capacity: int = 100000,
        hidden_dims: list[int] = None,
        use_attention: bool = False,
        prioritized_replay: bool = True,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device if device is not None else get_device("auto")
        self.alpha = alpha
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.prioritized_replay = prioritized_replay

        if hidden_dims is None:
            hidden_dims = [256, 256]

        # Q-networks
        self.q_network = FeatureQNetwork(state_dim, action_dim, hidden_dims, use_attention).to(
            self.device
        )
        self.target_network = FeatureQNetwork(state_dim, action_dim, hidden_dims, use_attention).to(
            self.device
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer with weight decay for regularization
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-5)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=20
        )

        # Replay buffer
        if prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)
        else:
            self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # Training metrics
        self.train_step = 0
        self.episode_count = 0
        self.metrics_history = []

    def select_action(
        self, state: np.ndarray, epsilon: float = 0.0, temperature: float = 1.0
    ) -> int:
        """
        Select action using epsilon-greedy or softmax policy.

        Args:
            state: Current state (feature flags + metrics)
            epsilon: Exploration rate (0.0 = greedy)
            temperature: Softmax temperature (1.0 = standard softmax, 0.0 = greedy)

        Returns:
            action: Index of selected action
        """
        # Epsilon-greedy exploration
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_t).cpu().numpy()[0]

            # Temperature-based action selection
            if temperature > 0.01:
                # Softmax with temperature
                logits = q_values / temperature
                exp_logits = np.exp(logits - logits.max())  # Numerical stability
                probs = exp_logits / exp_logits.sum()
                return np.random.choice(self.action_dim, p=probs)
            else:
                # Greedy
                return int(q_values.argmax())

    def compute_cql_loss(
        self, states: torch.Tensor, actions: torch.Tensor, q_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CQL penalty term.

        CQL Penalty = E[log Σ exp(Q(s,a))] - E[Q(s, a_data)]

        This pushes down Q-values on all actions (conservative estimate),
        then pushes up Q-values on dataset actions (correct for bias).

        Args:
            states: Batch of states
            actions: Batch of actions taken in dataset
            q_values: Q(s, a) for all actions

        Returns:
            CQL penalty (scalar)
        """
        # log-sum-exp over all actions: (batch_size,)
        logsumexp_q = torch.logsumexp(q_values, dim=1)

        # Q-values for dataset actions: (batch_size,)
        dataset_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # CQL penalty: E[logsumexp - dataset_q]
        cql_penalty = (logsumexp_q - dataset_q).mean()

        return cql_penalty

    def update(self) -> dict[str, float]:
        """
        Perform one gradient update step.

        Returns:
            Dictionary of training metrics (loss, TD loss, CQL loss, Q-values)
        """
        if len(self.replay_buffer) < self.batch_size:
            return {
                "loss": 0.0,
                "td_loss": 0.0,
                "cql_loss": 0.0,
                "q_mean": 0.0,
                "q_std": 0.0,
            }

        # Sample batch
        if self.prioritized_replay:
            (
                states,
                actions,
                rewards,
                next_states,
                dones,
                weights,
                indices,
            ) = self.replay_buffer.sample(self.batch_size)
            weights = weights.to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(
                self.batch_size
            )
            weights = torch.ones(self.batch_size).to(self.device)
            indices = None

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q-values: Q(s, a)
        q_values = self.q_network(states)
        q_current = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values: r + γ * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            q_next = self.target_network(next_states).max(dim=1)[0]
            q_target = rewards + self.gamma * q_next * (1 - dones)

        # TD loss (weighted by importance sampling)
        td_errors = q_current - q_target
        td_loss = (weights * td_errors.pow(2)).mean()

        # CQL penalty
        cql_loss = self.compute_cql_loss(states, actions, q_values)

        # Total loss
        loss = td_loss + self.alpha * cql_loss

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update priorities if using prioritized replay
        if self.prioritized_replay and indices is not None:
            priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
            self.replay_buffer.update_priorities(indices, priorities)

        # Update target network periodically
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Metrics
        metrics = {
            "loss": loss.item(),
            "td_loss": td_loss.item(),
            "cql_loss": cql_loss.item(),
            "q_mean": q_current.mean().item(),
            "q_std": q_current.std().item(),
            "q_target_mean": q_target.mean().item(),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

        self.metrics_history.append(metrics)
        return metrics

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def save(self, path: str, metadata: dict = None):
        """
        Save model checkpoint with metadata.

        Args:
            path: File path for checkpoint
            metadata: Optional metadata dict (training history, hyperparams, etc.)
        """
        # Extract network architecture
        hidden_dims = []
        for module in self.q_network.network:
            if isinstance(module, nn.Linear):
                hidden_dims.append(module.out_features)
        # Remove output layer dimension
        hidden_dims = hidden_dims[:-1]

        checkpoint = {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "train_step": self.train_step,
            "episode_count": self.episode_count,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "metrics_history": self.metrics_history,
            "hidden_dims": hidden_dims,
            "use_attention": self.q_network.use_attention,
        }

        if metadata is not None:
            checkpoint["metadata"] = metadata

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, device: torch.device = None) -> CQLFeatureSelector:
        """
        Load model checkpoint.

        Args:
            path: File path to checkpoint
            device: Target device (auto-detected if None)

        Returns:
            Loaded CQLFeatureSelector agent
        """
        if device is None:
            device = get_device("auto")

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Create agent with saved hyperparameters
        agent = cls(
            state_dim=checkpoint["state_dim"],
            action_dim=checkpoint["action_dim"],
            device=device,
            alpha=checkpoint.get("alpha", 1.0),
            gamma=checkpoint.get("gamma", 0.99),
            batch_size=checkpoint.get("batch_size", 64),
            hidden_dims=checkpoint.get("hidden_dims", [256, 256]),
            use_attention=checkpoint.get("use_attention", False),
            prioritized_replay=checkpoint.get("prioritized_replay", True),
        )

        # Load network weights
        agent.q_network.load_state_dict(checkpoint["q_network"])
        agent.target_network.load_state_dict(checkpoint["target_network"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        agent.scheduler.load_state_dict(checkpoint["scheduler"])
        agent.train_step = checkpoint["train_step"]
        agent.episode_count = checkpoint.get("episode_count", 0)
        agent.metrics_history = checkpoint.get("metrics_history", [])

        return agent

    def select_features(
        self,
        base_features: list[str],
        feature_budget: int = 100,
        epsilon: float = 0.0,
    ) -> np.ndarray:
        """
        Select optimal feature subset given budget constraint.

        Performs greedy rollout: start with empty set, iteratively add
        features that maximize Q-value until budget exhausted.

        Args:
            base_features: List of all available features
            feature_budget: Maximum number of features to select
            epsilon: Exploration rate

        Returns:
            Binary mask (1 = selected, 0 = not selected)
        """
        n_features = len(base_features)
        feature_mask = np.zeros(n_features, dtype=int)
        feature_count = 0

        # Iteratively select features
        for _ in range(feature_budget):
            # Construct state: [feature_mask, performance_metrics, budget_state]
            # For now, use placeholder metrics (should integrate with env)
            state = np.concatenate(
                [feature_mask, np.zeros(10), np.array([feature_count / feature_budget])]
            )

            # Select action (toggle feature i)
            action = self.select_action(state, epsilon=epsilon)

            # Toggle feature (if action is within bounds)
            if action < n_features and feature_mask[action] == 0:
                feature_mask[action] = 1
                feature_count += 1
            elif feature_count >= feature_budget:
                break

        return feature_mask


# ============================================================================
# Standard Replay Buffer (Non-Prioritized)
# ============================================================================


class ReplayBuffer:
    """Standard uniform replay buffer for CQL."""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add experience tuple."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample random batch."""
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
# Training Loop
# ============================================================================


def train_cql_feature_selector(
    agent: CQLFeatureSelector,
    train_data: pd.DataFrame,
    epochs: int = 500,
    log_freq: int = 20,
) -> list[dict[str, float]]:
    """
    Train CQL feature selector on offline dataset.

    Args:
        agent: CQLFeatureSelector agent
        train_data: DataFrame with columns: state, action, reward, next_state, done
        epochs: Number of training epochs
        log_freq: Logging frequency (epochs)

    Returns:
        List of training metrics per epoch
    """
    # Populate replay buffer
    print(f"Loading {len(train_data)} transitions into replay buffer...")
    for _, row in train_data.iterrows():
        agent.store_transition(
            state=row["state"],
            action=row["action"],
            reward=row["reward"],
            next_state=row["next_state"],
            done=row["done"],
        )

    print(f"\nTraining CQL Feature Selector (α={agent.alpha}) for {epochs} epochs...")
    print(f"Device: {agent.device}, Batch size: {agent.batch_size}")
    print(f"Buffer size: {len(agent.replay_buffer)}, State dim: {agent.state_dim}")

    epoch_metrics = []

    for epoch in range(epochs):
        # Multiple gradient steps per epoch
        n_updates = max(10, len(agent.replay_buffer) // agent.batch_size)
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

        # Aggregate metrics
        avg_loss = np.mean(epoch_losses)
        avg_td_loss = np.mean(epoch_td_losses)
        avg_cql_loss = np.mean(epoch_cql_losses)
        avg_q = np.mean(epoch_q_means)

        epoch_metrics.append(
            {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "td_loss": avg_td_loss,
                "cql_loss": avg_cql_loss,
                "q_mean": avg_q,
            }
        )

        # Learning rate scheduling
        agent.scheduler.step(avg_loss)

        # Logging
        if (epoch + 1) % log_freq == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"TD: {avg_td_loss:.4f} | "
                f"CQL: {avg_cql_loss:.4f} | "
                f"Q_mean: {avg_q:.4f}"
            )

    return epoch_metrics


# ============================================================================
# Test Block
# ============================================================================


if __name__ == "__main__":
    """
    Test CQL Feature Selector on synthetic environment.

    Synthetic Setup:
      - State: 100 binary feature flags + 10 performance metrics + 1 budget indicator
      - Actions: 100 (toggle each feature)
      - Reward: -MAE improvement (simulated)
      - Episode: Select features until budget exhausted
    """
    print("=" * 80)
    print("CQL Feature Selector - Test Block")
    print("=" * 80)

    # Setup
    set_seed(42)
    device = get_device("auto")
    print(f"\nDevice: {device}")

    # Synthetic environment parameters
    n_features = 100
    n_metrics = 10
    state_dim = n_features + n_metrics + 1  # feature_flags + metrics + budget
    action_dim = n_features  # toggle each feature

    print(f"State dim: {state_dim}")
    print(f"Action dim: {action_dim}")

    # Create agent
    agent = CQLFeatureSelector(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        alpha=1.5,  # More conservative for feature selection
        gamma=0.95,  # Shorter horizon
        lr=1e-3,
        batch_size=32,
        hidden_dims=[128, 128],
        use_attention=False,
        prioritized_replay=True,
    )

    print(f"\nAgent created with CQL α={agent.alpha}")

    # Generate synthetic offline data
    print("\nGenerating synthetic offline dataset...")
    n_samples = 1000
    train_data = []

    for _ in range(n_samples):
        # Random feature mask
        feature_mask = np.random.randint(0, 2, size=n_features)

        # Synthetic performance metrics (simulate MAE, calibration, etc.)
        # Better performance when ~30-50 features selected
        n_selected = feature_mask.sum()
        ideal_count = 40
        mae = abs(n_selected - ideal_count) * 0.5 + np.random.randn() * 2.0
        calibration = max(0.5, 1.0 - abs(n_selected - ideal_count) / 100.0)
        metrics = np.array([mae, calibration] + [np.random.randn() for _ in range(n_metrics - 2)])

        budget_indicator = n_selected / n_features
        state = np.concatenate([feature_mask, metrics, [budget_indicator]])

        # Random action (toggle a feature)
        action = np.random.randint(0, action_dim)

        # Reward: improvement in MAE (negative change)
        # Simulate: toggling near ideal count improves MAE
        new_mask = feature_mask.copy()
        new_mask[action] = 1 - new_mask[action]
        new_count = new_mask.sum()
        new_mae = abs(new_count - ideal_count) * 0.5
        reward = mae - new_mae  # positive if MAE decreased

        # Next state
        next_metrics = metrics.copy()
        next_metrics[0] = new_mae  # Update MAE
        next_metrics[1] = max(0.5, 1.0 - abs(new_count - ideal_count) / 100.0)
        next_state = np.concatenate([new_mask, next_metrics, [new_count / n_features]])

        # Done flag (episode ends when budget exhausted)
        done = new_count >= 50

        train_data.append(
            {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done,
            }
        )

    train_df = pd.DataFrame(train_data)
    print(f"Generated {len(train_df)} transitions")
    print(f"Mean reward: {train_df['reward'].mean():.4f}")
    print(f"Reward std: {train_df['reward'].std():.4f}")

    # Train agent
    print("\n" + "=" * 80)
    print("Training...")
    print("=" * 80)

    metrics = train_cql_feature_selector(agent=agent, train_data=train_df, epochs=100, log_freq=10)

    # Test feature selection
    print("\n" + "=" * 80)
    print("Testing Feature Selection")
    print("=" * 80)

    base_features = [f"feature_{i}" for i in range(n_features)]
    selected_mask = agent.select_features(
        base_features=base_features, feature_budget=40, epsilon=0.0
    )

    selected_features = [f for f, m in zip(base_features, selected_mask) if m == 1]
    print(f"\nSelected {len(selected_features)}/{n_features} features")
    print("Optimal range: 30-50 features (synthetic ground truth)")
    print(f"Performance: {'GOOD' if 30 <= len(selected_features) <= 50 else 'SUBOPTIMAL'}")

    # Save model
    output_path = "models/cql_feature_selector_test.pth"
    agent.save(output_path, metadata={"test_run": True, "synthetic_data": True})
    print(f"\nModel saved to {output_path}")

    # Test load
    loaded_agent = CQLFeatureSelector.load(output_path, device=device)
    print(f"Model loaded successfully (train_step={loaded_agent.train_step})")

    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)
