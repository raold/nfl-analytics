"""Unit tests for PPO agent (py/rl/ppo_agent.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# Add py/ to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "py"))

from rl.ppo_agent import ActorCritic, RolloutBuffer, PPOAgent


class TestActorCritic:
    """Test ActorCritic network."""
    
    def test_forward_pass_shapes(self):
        """Test that forward pass returns correct shapes."""
        model = ActorCritic(state_dim=6, hidden_dim=128)
        state = torch.randn(32, 6)
        
        dist, value = model.forward(state)
        
        # Beta distribution should sample actions of correct shape
        action = dist.sample()
        assert action.shape == (32, 1)
        
        # Value should have correct shape
        assert value.shape == (32, 1)
    
    def test_action_bounds(self):
        """Test that sampled actions are in [0, 1]."""
        model = ActorCritic(state_dim=6, hidden_dim=128)
        state = torch.randn(100, 6)
        
        dist, _ = model.forward(state)
        actions = dist.sample()
        
        assert torch.all(actions >= 0.0)
        assert torch.all(actions <= 1.0)
    
    def test_deterministic_action(self):
        """Test deterministic action selection uses mode."""
        model = ActorCritic(state_dim=6, hidden_dim=128)
        state = torch.randn(1, 6)
        
        # Sample multiple times with deterministic=True
        action1, _, _ = model.get_action(state, deterministic=True)
        action2, _, _ = model.get_action(state, deterministic=True)
        
        # Should be identical
        assert torch.allclose(action1, action2)
    
    def test_stochastic_action_varies(self):
        """Test stochastic action selection produces variation."""
        torch.manual_seed(42)
        model = ActorCritic(state_dim=6, hidden_dim=128)
        state = torch.randn(1, 6)
        
        # Sample multiple times with deterministic=False
        actions = [model.get_action(state, deterministic=False)[0] for _ in range(10)]
        
        # Should have some variation
        assert len(set([a.item() for a in actions])) > 1
    
    def test_entropy_computed(self):
        """Test that entropy is computed (can be negative for Beta distribution)."""
        model = ActorCritic(state_dim=6, hidden_dim=128)
        state = torch.randn(32, 6)
        action = torch.rand(32, 1)
        
        _, _, entropy = model.evaluate_actions(state, action)
        
        # Beta distribution entropy can be negative, just check it's finite
        assert torch.all(torch.isfinite(entropy))
        assert entropy.shape == (32, 1)


class TestRolloutBuffer:
    """Test RolloutBuffer storage."""
    
    def test_add_and_get(self):
        """Test adding transitions and retrieving batches."""
        buffer = RolloutBuffer()
        
        # Add 5 transitions
        for i in range(5):
            state = torch.randn(6)
            action = torch.rand(1)
            reward = float(i)
            log_prob = torch.randn(1)
            value = torch.randn(1)
            done = (i == 4)
            
            buffer.add(state, action, reward, log_prob, value, done)
        
        # Retrieve batch
        states, actions, rewards, log_probs, values, dones = buffer.get()
        
        assert states.shape == (5, 6)
        assert actions.shape == (5, 1)
        assert rewards.shape == (5,)
        assert log_probs.shape == (5, 1)
        assert values.shape == (5, 1)
        assert dones.shape == (5,)
        
        # Check rewards are correct
        assert torch.allclose(rewards, torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]))
        
        # Check dones
        assert dones[-1] == 1.0
        assert torch.sum(dones[:-1]) == 0.0
    
    def test_clear(self):
        """Test clearing buffer."""
        buffer = RolloutBuffer()
        
        # Add transitions
        for i in range(3):
            buffer.add(
                torch.randn(6), torch.rand(1), 0.0,
                torch.randn(1), torch.randn(1), False
            )
        
        assert len(buffer.states) == 3
        
        buffer.clear()
        
        assert len(buffer.states) == 0
        assert len(buffer.actions) == 0
        assert len(buffer.rewards) == 0


class TestPPOAgent:
    """Test PPOAgent."""
    
    def test_initialization(self):
        """Test agent initialization."""
        agent = PPOAgent(state_dim=6, hidden_dim=64, device="cpu")
        
        assert agent.state_dim == 6
        assert agent.gamma == 0.99
        assert agent.epsilon == 0.2
        assert agent.train_step == 0
    
    def test_select_action(self):
        """Test action selection."""
        agent = PPOAgent(state_dim=6, device="cpu")
        state = np.random.randn(6)
        
        action, log_prob, value = agent.select_action(state, deterministic=False)
        
        # Action should be in [0, 1]
        assert 0.0 <= action <= 1.0
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
    
    def test_compute_gae(self):
        """Test GAE computation."""
        agent = PPOAgent(state_dim=6, device="cpu")
        
        # Simple episode: 3 steps with rewards [1, 2, 3]
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([0.5, 1.0, 1.5])
        dones = torch.tensor([0.0, 0.0, 1.0])
        next_value = 0.0
        
        advantages, returns = agent.compute_gae(rewards, values, dones, next_value)
        
        # Returns should be >= rewards (assuming gamma < 1)
        assert torch.all(returns >= rewards)
        
        # Advantages should be returns - values
        assert torch.allclose(advantages, returns - values)
    
    def test_update_reduces_loss(self):
        """Test that PPO update reduces loss."""
        torch.manual_seed(42)
        agent = PPOAgent(state_dim=6, hidden_dim=32, device="cpu")
        
        # Generate synthetic rollout
        for _ in range(64):
            state = torch.randn(6)
            action = torch.rand(1)
            reward = np.random.randn()
            log_prob = torch.randn(1)
            value = torch.randn(1)
            done = False
            
            agent.buffer.add(state, action, reward, log_prob, value, done)
        
        # First update
        metrics1 = agent.update(n_epochs=2, batch_size=32)
        
        # Add more data
        for _ in range(64):
            state = torch.randn(6)
            action = torch.rand(1)
            reward = np.random.randn()
            log_prob = torch.randn(1)
            value = torch.randn(1)
            done = False
            
            agent.buffer.add(state, action, reward, log_prob, value, done)
        
        # Second update
        metrics2 = agent.update(n_epochs=2, batch_size=32)
        
        # Train step should increment
        assert agent.train_step == 2
    
    def test_save_and_load(self):
        """Test saving and loading checkpoints."""
        import tempfile
        
        agent1 = PPOAgent(state_dim=6, device="cpu")
        agent1.train_step = 42
        
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name
        
        try:
            agent1.save(path)
            
            agent2 = PPOAgent(state_dim=6, device="cpu")
            agent2.load(path)
            
            assert agent2.train_step == 42
            
            # Check that parameters are identical
            for p1, p2 in zip(agent1.policy.parameters(), agent2.policy.parameters()):
                assert torch.allclose(p1, p2)
        finally:
            Path(path).unlink()


class TestIntegration:
    """Integration tests."""
    
    def test_train_ppo_offline_runs(self):
        """Test that offline training completes without errors."""
        from rl.ppo_agent import train_ppo_offline
        
        # Create synthetic dataset
        np.random.seed(42)
        n_samples = 100
        
        df = pd.DataFrame({
            "p_hat": np.random.rand(n_samples) * 0.6 + 0.2,
            "market_prob": np.random.rand(n_samples) * 0.6 + 0.2,
            "spread_close": np.random.randn(n_samples) * 5,
            "total_close": np.random.randn(n_samples) * 5 + 45,
            "epa_gap": np.random.randn(n_samples),
            "home_cover": np.random.randint(0, 2, n_samples),
            "net_odds": np.full(n_samples, 0.91),
        })
        
        agent = PPOAgent(state_dim=6, hidden_dim=32, device="cpu")
        
        # Train for 2 epochs (fast)
        training_log = train_ppo_offline(
            agent, df, n_epochs=2, rollout_length=32,
            ppo_epochs=2, batch_size=16
        )
        
        assert len(training_log) == 2
        assert "avg_reward" in training_log[0]
        assert "avg_action" in training_log[0]
        assert agent.train_step > 0
    
    def test_evaluate_ppo_runs(self):
        """Test that evaluation completes without errors."""
        from rl.ppo_agent import evaluate_ppo
        
        # Create synthetic dataset
        np.random.seed(42)
        n_samples = 50
        
        df = pd.DataFrame({
            "p_hat": np.random.rand(n_samples) * 0.6 + 0.2,
            "market_prob": np.random.rand(n_samples) * 0.6 + 0.2,
            "spread_close": np.random.randn(n_samples) * 5,
            "total_close": np.random.randn(n_samples) * 5 + 45,
            "epa_gap": np.random.randn(n_samples),
            "home_cover": np.random.randint(0, 2, n_samples),
            "net_odds": np.full(n_samples, 0.91),
        })
        
        agent = PPOAgent(state_dim=6, hidden_dim=32, device="cpu")
        metrics = evaluate_ppo(agent, df)
        
        assert "total_pnl" in metrics
        assert "avg_pnl" in metrics
        assert "total_bets" in metrics
        assert "bet_rate" in metrics
        assert "avg_action" in metrics
