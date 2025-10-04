"""Unit tests for DQN agent."""
import numpy as np
import pytest
import torch

# Need to add py/ to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "py"))

from rl.dqn_agent import (
    DQNAgent,
    QNetwork,
    ReplayBuffer,
    set_seed,
    get_device,
    load_dataset,
)


class TestQNetwork:
    """Test Q-network architecture."""
    
    def test_network_initialization(self):
        """Test Q-network creates proper layers."""
        state_dim = 6
        n_actions = 4
        net = QNetwork(state_dim, n_actions)
        
        assert net is not None
        assert len(list(net.parameters())) > 0
    
    def test_forward_pass(self):
        """Test forward pass returns correct shape."""
        state_dim = 6
        n_actions = 4
        batch_size = 32
        
        net = QNetwork(state_dim, n_actions)
        states = torch.randn(batch_size, state_dim)
        q_values = net(states)
        
        assert q_values.shape == (batch_size, n_actions)
    
    def test_custom_hidden_dims(self):
        """Test custom hidden dimensions."""
        state_dim = 6
        n_actions = 4
        hidden_dims = [64, 32]
        
        net = QNetwork(state_dim, n_actions, hidden_dims)
        states = torch.randn(10, state_dim)
        q_values = net(states)
        
        assert q_values.shape == (10, n_actions)


class TestReplayBuffer:
    """Test experience replay buffer."""
    
    def test_buffer_initialization(self):
        """Test buffer initializes with correct capacity."""
        buffer = ReplayBuffer(capacity=1000)
        assert len(buffer) == 0
    
    def test_buffer_push(self):
        """Test adding experiences to buffer."""
        buffer = ReplayBuffer(capacity=100)
        state = np.array([1.0, 2.0, 3.0])
        next_state = np.array([1.1, 2.1, 3.1])
        
        buffer.push(state, action=0, reward=1.0, next_state=next_state, done=True)
        assert len(buffer) == 1
        
        # Add more
        for i in range(50):
            buffer.push(state, action=i % 4, reward=float(i), next_state=next_state, done=False)
        assert len(buffer) == 51
    
    def test_buffer_capacity_limit(self):
        """Test buffer respects max capacity."""
        capacity = 10
        buffer = ReplayBuffer(capacity=capacity)
        state = np.array([1.0, 2.0])
        next_state = np.array([1.0, 2.0])
        
        # Add more than capacity
        for i in range(20):
            buffer.push(state, action=0, reward=0.0, next_state=next_state, done=False)
        
        assert len(buffer) == capacity  # Should cap at capacity
    
    def test_buffer_sample(self):
        """Test sampling from buffer."""
        buffer = ReplayBuffer(capacity=100)
        state = np.array([1.0, 2.0, 3.0])
        next_state = np.array([1.1, 2.1, 3.1])
        
        for i in range(50):
            buffer.push(state, action=i % 4, reward=float(i), next_state=next_state, done=False)
        
        batch_size = 16
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        
        assert states.shape == (batch_size, 3)
        assert actions.shape == (batch_size,)
        assert rewards.shape == (batch_size,)
        assert next_states.shape == (batch_size, 3)
        assert dones.shape == (batch_size,)


class TestDQNAgent:
    """Test DQN agent."""
    
    @pytest.fixture
    def agent(self):
        """Create test agent."""
        set_seed(42)
        device = torch.device("cpu")  # Use CPU for tests
        return DQNAgent(
            state_dim=6,
            n_actions=4,
            device=device,
            gamma=0.99,
            lr=1e-3,
            batch_size=32,
            target_update_freq=100,
            hidden_dims=[64, 32]
        )
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.state_dim == 6
        assert agent.n_actions == 4
        assert agent.gamma == 0.99
        assert agent.batch_size == 32
        assert len(agent.replay_buffer) == 0
        assert agent.train_step == 0
    
    def test_select_action_greedy(self, agent):
        """Test greedy action selection (epsilon=0)."""
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        action = agent.select_action(state, epsilon=0.0)
        
        assert isinstance(action, int)
        assert 0 <= action < 4
    
    def test_select_action_exploration(self, agent):
        """Test epsilon-greedy exploration."""
        set_seed(42)
        state = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        
        # With epsilon=1.0, should always explore
        actions = [agent.select_action(state, epsilon=1.0) for _ in range(20)]
        # Should see variety in actions (not all the same)
        assert len(set(actions)) > 1
    
    def test_update_requires_buffer(self, agent):
        """Test update returns zero metrics when buffer too small."""
        metrics = agent.update()
        assert metrics["loss"] == 0.0
        assert metrics["q_mean"] == 0.0
    
    def test_update_with_data(self, agent):
        """Test update performs gradient step."""
        # Populate buffer
        for i in range(100):
            state = np.random.randn(6)
            next_state = np.random.randn(6)
            action = i % 4
            reward = np.random.randn()
            agent.replay_buffer.push(state, action, reward, next_state, done=False)
        
        # Perform update
        metrics = agent.update()
        
        assert metrics["loss"] > 0  # Should have non-zero loss
        assert "q_mean" in metrics
        assert agent.train_step == 1
    
    def test_target_network_update(self, agent):
        """Test target network updates periodically."""
        # Populate buffer
        for i in range(100):
            state = np.random.randn(6)
            next_state = np.random.randn(6)
            agent.replay_buffer.push(state, i % 4, 0.0, next_state, done=False)
        
        # Get initial target network weights
        initial_weights = agent.target_network.network[0].weight.data.clone()
        
        # Do updates up to target_update_freq
        for _ in range(agent.target_update_freq):
            agent.update()
        
        # Target network should be updated now
        updated_weights = agent.target_network.network[0].weight.data
        
        # Weights should match q_network now
        q_weights = agent.q_network.network[0].weight.data
        assert torch.allclose(updated_weights, q_weights)
    
    def test_save_and_load(self, agent, tmp_path):
        """Test model checkpoint save/load."""
        # Train a bit
        for i in range(50):
            state = np.random.randn(6)
            next_state = np.random.randn(6)
            agent.replay_buffer.push(state, i % 4, 0.0, next_state, done=False)
        
        for _ in range(10):
            agent.update()
        
        # Save
        save_path = tmp_path / "test_checkpoint.pth"
        agent.save(str(save_path))
        assert save_path.exists()
        
        # Create new agent and load
        new_agent = DQNAgent(
            state_dim=6,
            n_actions=4,
            device=torch.device("cpu"),
            hidden_dims=[64, 32]
        )
        new_agent.load(str(save_path))
        
        # Check train_step was restored
        assert new_agent.train_step == agent.train_step
        
        # Check weights match
        for p1, p2 in zip(agent.q_network.parameters(), new_agent.q_network.parameters()):
            assert torch.allclose(p1, p2)


class TestUtilities:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting for reproducibility."""
        set_seed(42)
        r1 = np.random.rand()
        
        set_seed(42)
        r2 = np.random.rand()
        
        assert r1 == r2
    
    def test_get_device_cpu(self):
        """Test device selection defaults to CPU in tests."""
        device = get_device("cpu")
        assert device.type == "cpu"
    
    def test_load_dataset_structure(self, tmp_path):
        """Test dataset loading from CSV."""
        # Create mock CSV
        csv_path = tmp_path / "test_dataset.csv"
        csv_content = """spread_close,total_close,epa_gap,market_prob,p_hat,edge,action,r
-3.0,45.5,0.5,0.55,0.60,0.05,1,0.91
3.5,48.0,-0.3,0.45,0.42,-0.03,0,0.0
-7.0,50.0,1.2,0.65,0.70,0.05,1,0.91
"""
        csv_path.write_text(csv_content)
        
        state_cols = ["spread_close", "total_close", "epa_gap", "market_prob", "p_hat", "edge"]
        states, actions, rewards = load_dataset(str(csv_path), state_cols)
        
        assert states.shape == (3, 6)
        assert actions.shape == (3,)
        assert rewards.shape == (3,)
        assert states.dtype == np.float32
        assert actions.dtype == int
        assert rewards.dtype == np.float32


class TestIntegration:
    """Integration tests for full training loop."""
    
    def test_mini_training_run(self):
        """Test a minimal training loop completes without errors."""
        set_seed(42)
        device = torch.device("cpu")
        
        # Create small dataset
        n_samples = 100
        state_dim = 6
        states = np.random.randn(n_samples, state_dim).astype(np.float32)
        actions = np.random.randint(0, 4, n_samples)
        rewards = np.random.randn(n_samples).astype(np.float32)
        
        # Create agent
        agent = DQNAgent(
            state_dim=state_dim,
            n_actions=4,
            device=device,
            batch_size=16,
            hidden_dims=[32, 16]
        )
        
        # Populate buffer
        for i in range(n_samples):
            agent.replay_buffer.push(states[i], actions[i], rewards[i], states[i], done=True)
        
        # Train for a few epochs
        initial_loss = None
        for epoch in range(5):
            metrics = agent.update()
            if epoch == 0:
                initial_loss = metrics["loss"]
        
        # Should have completed without errors
        assert agent.train_step == 5
        assert initial_loss is not None
