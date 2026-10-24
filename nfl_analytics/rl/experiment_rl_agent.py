"""
Reinforcement Learning Agents for Experiment-Augmented Bet Sizing

Implements DQN, PPO, and SAC agents that leverage the 44-dimensional state space
(15 base + 29 experiment features) to learn optimal betting policies.

Agents support:
- Feature importance tracking to identify which experiments contribute most
- Ablation studies to measure individual experiment contributions
- Kelly-aware reward shaping for proper bet sizing
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for discrete action betting.

    Architecture: 44-dim input → 128 → 128 → 64 → 5-dim output (Q-values)
    """

    def __init__(
        self, state_dim: int = 44, action_dim: int = 5, hidden_dims: list[int] = [128, 128, 64]
    ):
        super().__init__()

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass: state → Q-values for each action"""
        return self.network(state)


class ReplayBuffer:
    """Experience replay buffer for DQN training"""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        """Sample random batch from buffer"""
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent for experiment-augmented betting.

    Uses double DQN with target network for stable learning.
    Tracks feature importance via gradient analysis.
    """

    def __init__(
        self,
        state_dim: int = 44,
        action_dim: int = 5,
        hidden_dims: list[int] = [128, 128, 64],
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10,
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Initialize networks
        self.policy_net = DQNNetwork(state_dim, action_dim, hidden_dims)
        self.target_net = DQNNetwork(state_dim, action_dim, hidden_dims)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer and replay buffer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Feature importance tracking
        self.feature_gradients = np.zeros(state_dim)
        self.feature_usage_counts = np.zeros(state_dim)

        # Training metrics
        self.update_count = 0
        self.losses = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: 44-dimensional state
            training: If True, use epsilon-greedy; else greedy

        Returns:
            Action index (0-4)
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> float | None:
        """
        Perform one training step using experience replay.

        Returns:
            Loss value if training occurred, else None
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Enable gradient tracking for feature importance
        states.requires_grad = True

        # Compute current Q-values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Compute target Q-values (Double DQN)
        with torch.no_grad():
            # Select best actions using policy network
            next_actions = self.policy_net(next_states).argmax(1)
            # Evaluate using target network
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Compute loss
        loss = F.mse_loss(current_q, target_q)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()

        # Track feature importance via gradients
        if states.grad is not None:
            feature_grads = states.grad.abs().mean(dim=0).numpy()
            self.feature_gradients += feature_grads
            self.feature_usage_counts += (feature_grads > 0).astype(float)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)

        self.optimizer.step()

        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        self.losses.append(loss.item())
        return loss.item()

    def get_feature_importance(self) -> np.ndarray:
        """
        Get normalized feature importance scores.

        Returns:
            44-dimensional array of feature importance (sum to 1.0)
        """
        if self.feature_usage_counts.sum() == 0:
            return np.zeros(self.state_dim)

        # Normalize by usage count
        importance = self.feature_gradients / (self.feature_usage_counts + 1e-8)

        # Normalize to sum to 1.0
        importance = importance / (importance.sum() + 1e-8)

        return importance

    def save(self, path: str):
        """Save agent state"""
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "update_count": self.update_count,
                "feature_gradients": self.feature_gradients,
                "feature_usage_counts": self.feature_usage_counts,
                "losses": self.losses,
            },
            path,
        )

    def load(self, path: str):
        """Load agent state"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.update_count = checkpoint["update_count"]
        self.feature_gradients = checkpoint["feature_gradients"]
        self.feature_usage_counts = checkpoint["feature_usage_counts"]
        self.losses = checkpoint["losses"]


class PPOActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.

    Shares base network, splits into actor (policy) and critic (value) heads.
    """

    def __init__(
        self, state_dim: int = 44, action_dim: int = 5, hidden_dims: list[int] = [128, 128]
    ):
        super().__init__()

        # Shared base network
        base_layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            base_layers.append(nn.Linear(prev_dim, hidden_dim))
            base_layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.base = nn.Sequential(*base_layers)

        # Actor head (policy)
        self.actor = nn.Linear(prev_dim, action_dim)

        # Critic head (value function)
        self.critic = nn.Linear(prev_dim, 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            (action_probs, state_value)
        """
        base_output = self.base(state)

        # Actor: action probabilities
        action_logits = self.actor(base_output)
        action_probs = F.softmax(action_logits, dim=-1)

        # Critic: state value
        state_value = self.critic(base_output)

        return action_probs, state_value


class PPOAgent:
    """
    Proximal Policy Optimization agent for experiment-augmented betting.

    More stable than DQN for continuous learning.
    Uses clipped objective to prevent destructive policy updates.
    """

    def __init__(
        self,
        state_dim: int = 44,
        action_dim: int = 5,
        hidden_dims: list[int] = [128, 128],
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        epochs_per_update: int = 4,
        batch_size: int = 64,
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size

        # Initialize network
        self.network = PPOActorCritic(state_dim, action_dim, hidden_dims)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # Episode buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

        # Feature importance tracking
        self.feature_gradients = np.zeros(state_dim)
        self.feature_usage_counts = np.zeros(state_dim)

        # Training metrics
        self.policy_losses = []
        self.value_losses = []

    def select_action(self, state: np.ndarray, training: bool = True) -> tuple[int, float, float]:
        """
        Select action using current policy.

        Args:
            state: 44-dimensional state
            training: If True, sample from policy; else use argmax

        Returns:
            (action, log_prob, state_value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action_probs, state_value = self.network(state_tensor)

        if training:
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        else:
            action = action_probs.argmax()
            log_prob = torch.log(action_probs[0, action])

        return action.item(), log_prob.item(), state_value.item()

    def store_experience(self, state, action, reward, log_prob, value, done):
        """Store experience in episode buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.

        Returns:
            (advantages, returns)
        """
        advantages = []
        returns = []

        gae = 0
        next_value = 0

        # Compute advantages in reverse
        for i in reversed(range(len(self.rewards))):
            delta = self.rewards[i] + self.gamma * next_value * (1 - self.dones[i]) - self.values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[i]) * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[i])

            next_value = self.values[i]

        advantages = np.array(advantages)
        returns = np.array(returns)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def train_step(self) -> tuple[float, float]:
        """
        Perform PPO update using collected experiences.

        Returns:
            (policy_loss, value_loss)
        """
        if len(self.states) == 0:
            return 0.0, 0.0

        # Compute advantages and returns
        advantages, returns = self.compute_gae()

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)
        advantages_tensor = torch.FloatTensor(advantages)
        returns_tensor = torch.FloatTensor(returns)

        # Enable gradient tracking for feature importance
        states.requires_grad = True

        # PPO update epochs
        policy_losses = []
        value_losses = []

        for _ in range(self.epochs_per_update):
            # Forward pass
            action_probs, state_values = self.network(states)

            # Compute new log probs
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)

            # Policy loss with clipping
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages_tensor
            surr2 = (
                torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_tensor
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(state_values.squeeze(), returns_tensor)

            # Total loss
            loss = policy_loss + 0.5 * value_loss

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()

            # Track feature importance
            if states.grad is not None:
                feature_grads = states.grad.abs().mean(dim=0).detach().numpy()
                self.feature_gradients += feature_grads
                self.feature_usage_counts += (feature_grads > 0).astype(float)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)

            self.optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

        # Clear episode buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

        avg_policy_loss = np.mean(policy_losses)
        avg_value_loss = np.mean(value_losses)

        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)

        return avg_policy_loss, avg_value_loss

    def get_feature_importance(self) -> np.ndarray:
        """Get normalized feature importance scores"""
        if self.feature_usage_counts.sum() == 0:
            return np.zeros(self.state_dim)

        importance = self.feature_gradients / (self.feature_usage_counts + 1e-8)
        importance = importance / (importance.sum() + 1e-8)

        return importance

    def save(self, path: str):
        """Save agent state"""
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "feature_gradients": self.feature_gradients,
                "feature_usage_counts": self.feature_usage_counts,
                "policy_losses": self.policy_losses,
                "value_losses": self.value_losses,
            },
            path,
        )

    def load(self, path: str):
        """Load agent state"""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.feature_gradients = checkpoint["feature_gradients"]
        self.feature_usage_counts = checkpoint["feature_usage_counts"]
        self.policy_losses = checkpoint["policy_losses"]
        self.value_losses = checkpoint["value_losses"]


class AblationStudyRunner:
    """
    Run ablation studies to measure individual experiment contributions.

    Trains agents with different feature subsets to isolate experiment impact.
    """

    def __init__(self, base_agent_class, base_agent_params: dict):
        self.base_agent_class = base_agent_class
        self.base_agent_params = base_agent_params

    def run_ablation(
        self,
        env,
        experiment_indices: list[int],
        num_episodes: int = 100,
        experiment_name: str = "ablation",
    ) -> dict:
        """
        Run ablation study by training agent with specific features removed.

        Args:
            env: Gym environment
            experiment_indices: Indices (0-28) of experiment features to REMOVE
            num_episodes: Training episodes
            experiment_name: Name for this ablation

        Returns:
            Results dict with performance metrics
        """
        # Create feature mask (1 = keep, 0 = remove)
        feature_mask = np.ones(44)
        for idx in experiment_indices:
            feature_mask[15 + idx] = 0  # Offset by 15 base features

        # Train agent with masked features
        agent = self.base_agent_class(**self.base_agent_params)

        episode_returns = []

        for episode in range(num_episodes):
            state = env.reset()
            state = state * feature_mask  # Apply mask

            episode_return = 0
            done = False

            while not done:
                action = agent.select_action(state, training=True)

                if isinstance(action, tuple):  # PPO returns (action, log_prob, value)
                    action, log_prob, value = action

                next_state, reward, done, info = env.step(action)
                next_state = next_state * feature_mask  # Apply mask

                if isinstance(agent, DQNAgent):
                    agent.store_experience(state, action, reward, next_state, done)
                    agent.train_step()
                elif isinstance(agent, PPOAgent):
                    agent.store_experience(state, action, reward, log_prob, value, done)

                state = next_state
                episode_return += reward

            # PPO update at end of episode
            if isinstance(agent, PPOAgent):
                agent.train_step()

            episode_returns.append(episode_return)

        return {
            "experiment_name": experiment_name,
            "removed_features": experiment_indices,
            "mean_return": np.mean(episode_returns),
            "std_return": np.std(episode_returns),
            "final_bankroll": env.current_bankroll,
            "roi": (env.current_bankroll - env.initial_bankroll) / env.initial_bankroll,
            "sharpe": np.mean(episode_returns) / (np.std(episode_returns) + 1e-8),
        }
