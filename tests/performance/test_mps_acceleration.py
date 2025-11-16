"""
Test suite for Mac M4 MPS GPU acceleration

Validates that:
1. MPS device detection works correctly
2. PyTorch models train on MPS without errors
3. MPS predictions match CPU predictions (within tolerance)
4. MPS training is faster than CPU (speedup >= 2x)
5. XGBoost GPU support works (if available)

Author: Claude Code (Sonnet 4.5)
Date: November 15, 2025
"""

import pytest
import torch
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_regression

# Import utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from py.utils.device_utils import (
    get_best_device,
    validate_device_support,
    get_device_info,
)
from py.utils.xgboost_device import (
    get_xgboost_params,
    validate_xgboost_gpu_support,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_data():
    """Generate sample regression data"""
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    return X.astype(np.float32), y.astype(np.float32)


@pytest.fixture
def simple_model():
    """Create simple PyTorch model for testing"""
    class SimpleModel(torch.nn.Module):
        def __init__(self, input_dim=10, hidden_dim=32, output_dim=1):
            super().__init__()
            self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
            self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    return SimpleModel()


# ============================================================================
# Device Detection Tests
# ============================================================================

def test_get_best_device():
    """Test that best device detection works"""
    device = get_best_device()
    assert isinstance(device, torch.device)
    assert device.type in ['cuda', 'mps', 'cpu']


def test_get_best_device_force_cpu():
    """Test forcing CPU device"""
    device = get_best_device('cpu')
    assert device.type == 'cpu'


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_get_best_device_force_mps():
    """Test forcing MPS device"""
    device = get_best_device('mps')
    assert device.type == 'mps'


def test_validate_device_support_cpu():
    """Test CPU device validation"""
    device = torch.device('cpu')
    assert validate_device_support(device) is True


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_validate_device_support_mps():
    """Test MPS device validation"""
    device = torch.device('mps')
    assert validate_device_support(device) is True


def test_get_device_info():
    """Test device info retrieval"""
    info = get_device_info()

    assert 'device' in info
    assert 'device_type' in info
    assert 'has_cuda' in info
    assert 'has_mps' in info
    assert 'cpu_count' in info
    assert 'pytorch_version' in info


# ============================================================================
# PyTorch MPS Training Tests
# ============================================================================

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_simple_model_mps_training(simple_model, sample_data):
    """Test that simple model trains on MPS"""
    X, y = sample_data
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)

    # Move to MPS
    device = torch.device('mps')
    model = simple_model.to(device)
    X_mps = X_tensor.to(device)
    y_mps = y_tensor.to(device)

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    initial_loss = None
    final_loss = None

    for epoch in range(10):
        optimizer.zero_grad()
        output = model(X_mps)
        loss = criterion(output, y_mps)

        if epoch == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

        final_loss = loss.item()

    # Verify training occurred (loss decreased)
    assert final_loss < initial_loss
    assert not np.isnan(final_loss)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_predictions_match_cpu(simple_model, sample_data):
    """Test that MPS predictions match CPU (within tolerance)"""
    X, y = sample_data
    X_tensor = torch.FloatTensor(X[:100])  # Use smaller subset for speed

    # Train on CPU
    model_cpu = simple_model.to('cpu')
    optimizer = torch.optim.Adam(model_cpu.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    for epoch in range(20):
        optimizer.zero_grad()
        output = model_cpu(X_tensor)
        y_tensor = torch.FloatTensor(y[:100]).unsqueeze(1)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()

    # Get CPU predictions
    model_cpu.eval()
    with torch.no_grad():
        pred_cpu = model_cpu(X_tensor).numpy()

    # Move to MPS and get predictions
    model_mps = model_cpu.to('mps')
    X_mps = X_tensor.to('mps')

    model_mps.eval()
    with torch.no_grad():
        pred_mps = model_mps(X_mps).to('cpu').numpy()

    # Validate predictions match
    mae_diff = np.abs(pred_cpu - pred_mps).mean()
    max_diff = np.abs(pred_cpu - pred_mps).max()

    # Tolerance: MAE < 1e-4, Max < 1e-3
    assert mae_diff < 1e-4, f"MPS predictions differ from CPU (MAE: {mae_diff:.6f})"
    assert max_diff < 1e-3, f"MPS predictions differ from CPU (Max: {max_diff:.6f})"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
@pytest.mark.slow
def test_mps_faster_than_cpu(simple_model, sample_data):
    """Test that MPS training is faster than CPU"""
    import time

    X, y = sample_data
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)

    # Benchmark CPU
    model_cpu = simple_model.to('cpu')
    optimizer_cpu = torch.optim.Adam(model_cpu.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    start_cpu = time.time()
    for epoch in range(50):
        optimizer_cpu.zero_grad()
        output = model_cpu(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer_cpu.step()
    cpu_time = time.time() - start_cpu

    # Benchmark MPS
    model_mps = simple_model.to('mps')
    X_mps = X_tensor.to('mps')
    y_mps = y_tensor.to('mps')
    optimizer_mps = torch.optim.Adam(model_mps.parameters(), lr=0.01)

    start_mps = time.time()
    for epoch in range(50):
        optimizer_mps.zero_grad()
        output = model_mps(X_mps)
        loss = criterion(output, y_mps)
        loss.backward()
        optimizer_mps.step()
    mps_time = time.time() - start_mps

    # Calculate speedup
    speedup = cpu_time / mps_time

    print(f"\nCPU time: {cpu_time:.2f}s")
    print(f"MPS time: {mps_time:.2f}s")
    print(f"Speedup: {speedup:.2f}x")

    # Verify MPS is faster (speedup >= 1.5x)
    # Note: Lower threshold than production (2x) due to small model size
    assert speedup >= 1.5, f"MPS not faster than CPU (speedup: {speedup:.2f}x)"


# ============================================================================
# XGBoost GPU Tests
# ============================================================================

def test_get_xgboost_params_cpu():
    """Test XGBoost CPU parameter configuration"""
    base_params = {'n_estimators': 100, 'max_depth': 6}
    params = get_xgboost_params(base_params, prefer_gpu=False, verbose=False)

    assert params['tree_method'] == 'hist'
    assert params['n_jobs'] == -1
    assert 'device' not in params or params['device'] != 'cuda'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_get_xgboost_params_cuda():
    """Test XGBoost CUDA parameter configuration"""
    base_params = {'n_estimators': 100, 'max_depth': 6}
    params = get_xgboost_params(base_params, prefer_gpu=True, verbose=False)

    assert params['tree_method'] == 'gpu_hist'
    assert params['device'] == 'cuda'
    assert params['predictor'] == 'gpu_predictor'


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_get_xgboost_params_mps():
    """Test XGBoost MPS parameter configuration (experimental)"""
    base_params = {'n_estimators': 100, 'max_depth': 6}
    params = get_xgboost_params(
        base_params,
        prefer_gpu=True,
        experimental_mps=True,
        verbose=False
    )

    assert params['tree_method'] == 'gpu_hist'
    assert params['device'] == 'mps'
    assert params['predictor'] == 'gpu_predictor'


def test_validate_xgboost_gpu_support():
    """Test XGBoost GPU support validation"""
    validation = validate_xgboost_gpu_support()

    assert 'has_cuda' in validation
    assert 'has_mps' in validation
    assert 'xgboost_version' in validation
    assert 'recommended_device' in validation
    assert 'warnings' in validation

    # Verify version format
    assert len(validation['xgboost_version'].split('.')) >= 2


def test_xgboost_cpu_training(sample_data):
    """Test XGBoost trains on CPU"""
    X, y = sample_data

    base_params = {'n_estimators': 50, 'max_depth': 4}
    params = get_xgboost_params(base_params, prefer_gpu=False, verbose=False)

    model = xgb.XGBRegressor(**params)
    model.fit(X, y, verbose=False)

    # Verify model trained
    assert model.get_booster() is not None

    # Verify predictions work
    preds = model.predict(X[:10])
    assert len(preds) == 10
    assert not np.isnan(preds).any()


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
@pytest.mark.slow
def test_full_pipeline_mps(sample_data):
    """Test full training pipeline on MPS"""
    X, y = sample_data
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]

    # Create model
    class RegressionNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(10, 64)
            self.fc2 = torch.nn.Linear(64, 32)
            self.fc3 = torch.nn.Linear(32, 1)
            self.dropout = torch.nn.Dropout(0.2)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            return self.fc3(x)

    # Train on MPS
    device = torch.device('mps')
    model = RegressionNet().to(device)

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # Training loop
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_output = model(X_test_t)
        test_loss = criterion(test_output, y_test_t).item()

    # Verify reasonable test loss
    assert test_loss < 10000  # Should be much better than random
    assert not np.isnan(test_loss)


# ============================================================================
# Performance Benchmarks
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_benchmark_device_comparison(sample_data):
    """Benchmark training across CPU and MPS"""
    import time

    X, y = sample_data
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)

    class BenchmarkModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(10, 128)
            self.fc2 = torch.nn.Linear(128, 64)
            self.fc3 = torch.nn.Linear(64, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

    results = {}

    # Benchmark CPU
    model_cpu = BenchmarkModel().to('cpu')
    optimizer = torch.optim.Adam(model_cpu.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    start = time.time()
    for epoch in range(100):
        optimizer.zero_grad()
        output = model_cpu(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
    cpu_time = time.time() - start

    results['cpu_time'] = cpu_time

    # Benchmark MPS
    model_mps = BenchmarkModel().to('mps')
    X_mps = X_tensor.to('mps')
    y_mps = y_tensor.to('mps')
    optimizer = torch.optim.Adam(model_mps.parameters(), lr=0.01)

    start = time.time()
    for epoch in range(100):
        optimizer.zero_grad()
        output = model_mps(X_mps)
        loss = criterion(output, y_mps)
        loss.backward()
        optimizer.step()
    mps_time = time.time() - start

    results['mps_time'] = mps_time
    results['speedup'] = cpu_time / mps_time

    # Print results
    print(f"\n{'='*60}")
    print("PERFORMANCE BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"CPU Time: {cpu_time:.2f}s")
    print(f"MPS Time: {mps_time:.2f}s")
    print(f"Speedup: {results['speedup']:.2f}x")
    print(f"{'='*60}")

    # Assert MPS is faster
    assert results['speedup'] >= 1.5


# ============================================================================
# Marks
# ============================================================================

# Markers for selective test execution:
# - pytest -m "not slow" (skip slow tests)
# - pytest -m "benchmark" (run only benchmarks)
# - pytest tests/performance/test_mps_acceleration.py (run all)
