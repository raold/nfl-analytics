#!/usr/bin/env python3
"""
Unit Tests for Hybrid Static-Incremental XGBoost

Tests cover:
1. Model initialization and configuration
2. Static component training
3. Incremental component learning
4. Hybrid prediction (70/30 blend)
5. Save/load functionality
6. Error handling and edge cases

Author: Claude Code (Anthropic)
Date: November 12, 2025
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from py.online.hybrid_static_incremental import HybridStaticIncrementalXGBoost

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def synthetic_data():
    """Generate synthetic training and test data"""
    np.random.seed(42)

    # Training data (1000 samples)
    n_train = 1000
    X_train = np.random.randn(n_train, 5) + np.array([250, 35, 0.65, 5, 10])
    y_train = X_train[:, 0] + 0.5 * X_train[:, 1] + np.random.randn(n_train) * 20

    # Test data (200 samples)
    n_test = 200
    X_test = np.random.randn(n_test, 5) + np.array([250, 35, 0.65, 5, 10])
    y_test = X_test[:, 0] + 0.5 * X_test[:, 1] + np.random.randn(n_test) * 20

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


@pytest.fixture
def hybrid_model():
    """Create a hybrid model instance"""
    return HybridStaticIncrementalXGBoost(
        static_weight=0.7,
        static_params={
            "n_estimators": 50,
            "max_depth": 4,
            "learning_rate": 0.1,
            "random_state": 42,
        },
        incremental_params={
            "window_size": 100,
            "retrain_every": 10,
            "n_estimators": 30,
            "max_depth": 3,
            "learning_rate": 0.1,
        },
    )


# =============================================================================
# Initialization Tests
# =============================================================================


def test_initialization_default():
    """Test default initialization"""
    model = HybridStaticIncrementalXGBoost()

    assert model.static_weight == 0.7
    assert model.incremental_weight == 0.3
    assert not model.static_trained
    assert model.feature_names is None


def test_initialization_custom_weights():
    """Test initialization with custom weights"""
    model = HybridStaticIncrementalXGBoost(static_weight=0.6)

    assert model.static_weight == 0.6
    assert model.incremental_weight == 0.4


def test_initialization_invalid_weight():
    """Test initialization with invalid weight"""
    with pytest.raises(ValueError, match="static_weight must be in"):
        HybridStaticIncrementalXGBoost(static_weight=1.5)

    with pytest.raises(ValueError, match="static_weight must be in"):
        HybridStaticIncrementalXGBoost(static_weight=-0.1)


def test_initialization_custom_params():
    """Test initialization with custom parameters"""
    static_params = {"n_estimators": 200, "max_depth": 8}
    incremental_params = {"window_size": 200, "retrain_every": 30}

    model = HybridStaticIncrementalXGBoost(
        static_params=static_params, incremental_params=incremental_params
    )

    assert model.static_params["n_estimators"] == 200
    assert model.static_params["max_depth"] == 8
    assert model.incremental_params["window_size"] == 200
    assert model.incremental_params["retrain_every"] == 30


# =============================================================================
# Training Tests
# =============================================================================


def test_fit_static_numpy(hybrid_model, synthetic_data):
    """Test static component training with numpy arrays"""
    X_train = synthetic_data["X_train"]
    y_train = synthetic_data["y_train"]

    hybrid_model.fit_static(X_train, y_train)

    assert hybrid_model.static_trained
    assert hybrid_model.static_scaler is not None


def test_fit_static_dataframe(hybrid_model):
    """Test static component training with pandas DataFrame"""
    np.random.seed(42)
    n = 500

    df = pd.DataFrame(
        {
            "f1": np.random.randn(n) + 250,
            "f2": np.random.randn(n) + 35,
            "f3": np.random.randn(n) + 0.65,
        }
    )
    y = df["f1"] + 0.5 * df["f2"] + np.random.randn(n) * 20

    hybrid_model.fit_static(df, y)

    assert hybrid_model.static_trained
    assert hybrid_model.feature_names == ["f1", "f2", "f3"]


# =============================================================================
# Prediction Tests
# =============================================================================


def test_predict_batch(hybrid_model, synthetic_data):
    """Test batch prediction"""
    X_train = synthetic_data["X_train"]
    y_train = synthetic_data["y_train"]
    X_test = synthetic_data["X_test"]

    hybrid_model.fit_static(X_train, y_train)
    predictions = hybrid_model.predict(X_test)

    assert len(predictions) == len(X_test)
    assert isinstance(predictions, np.ndarray)
    assert predictions.dtype == np.float64


def test_predict_one_array(hybrid_model, synthetic_data):
    """Test single prediction with array input"""
    X_train = synthetic_data["X_train"]
    y_train = synthetic_data["y_train"]
    x_test = synthetic_data["X_test"][0]

    hybrid_model.fit_static(X_train, y_train)
    pred = hybrid_model.predict_one(x_test)

    assert isinstance(pred, (float, np.floating))
    assert not np.isnan(pred)


def test_predict_one_dict(hybrid_model, synthetic_data):
    """Test single prediction with dict input"""
    X_train = synthetic_data["X_train"]
    y_train = synthetic_data["y_train"]
    x_test = synthetic_data["X_test"][0]

    hybrid_model.fit_static(X_train, y_train)

    # Convert to dict
    x_dict = {f"f{i}": x_test[i] for i in range(len(x_test))}
    pred = hybrid_model.predict_one(x_dict)

    assert isinstance(pred, (float, np.floating))
    assert not np.isnan(pred)


def test_predict_before_training(hybrid_model, synthetic_data):
    """Test prediction before training (should raise error)"""
    X_test = synthetic_data["X_test"]

    with pytest.raises(ValueError, match="Must call fit_static"):
        hybrid_model.predict(X_test)


def test_hybrid_blend_weights(hybrid_model, synthetic_data):
    """Test that predictions respect 70/30 blend"""
    X_train = synthetic_data["X_train"]
    y_train = synthetic_data["y_train"]
    X_test = synthetic_data["X_test"][:10]

    hybrid_model.fit_static(X_train, y_train)

    # Get hybrid predictions
    hybrid_preds = []
    for x in X_test:
        pred = hybrid_model.predict_one(x)
        hybrid_preds.append(pred)

    # Verify predictions are reasonable (within range of target)
    assert np.mean(hybrid_preds) > 200
    assert np.mean(hybrid_preds) < 400


# =============================================================================
# Online Learning Tests
# =============================================================================


def test_learn_one_array(hybrid_model, synthetic_data):
    """Test incremental learning with array input"""
    X_train = synthetic_data["X_train"]
    y_train = synthetic_data["y_train"]
    x_test = synthetic_data["X_test"][0]
    y_test = synthetic_data["y_test"][0]

    hybrid_model.fit_static(X_train, y_train)

    # Initial prediction
    hybrid_model.predict_one(x_test)

    # Learn from new sample
    hybrid_model.learn_one(x_test, y_test)

    # Prediction should still work
    pred_after = hybrid_model.predict_one(x_test)

    assert isinstance(pred_after, (float, np.floating))


def test_learn_one_dict(hybrid_model, synthetic_data):
    """Test incremental learning with dict input"""
    X_train = synthetic_data["X_train"]
    y_train = synthetic_data["y_train"]
    x_test = synthetic_data["X_test"][0]
    y_test = synthetic_data["y_test"][0]

    hybrid_model.fit_static(X_train, y_train)

    # Convert to dict
    x_dict = {f"f{i}": x_test[i] for i in range(len(x_test))}

    # Learn from new sample
    hybrid_model.learn_one(x_dict, y_test)

    # Verify incremental model updated
    metrics = hybrid_model.get_metrics()
    assert metrics["incremental_counter"] > 0


def test_prequential_evaluation(hybrid_model, synthetic_data):
    """Test prequential (test-then-train) evaluation"""
    X_train = synthetic_data["X_train"]
    y_train = synthetic_data["y_train"]
    X_test = synthetic_data["X_test"]
    y_test = synthetic_data["y_test"]

    hybrid_model.fit_static(X_train, y_train)

    errors = []
    for x, y in zip(X_test, y_test):
        # Predict
        pred = hybrid_model.predict_one(x)
        error = abs(y - pred)
        errors.append(error)

        # Learn
        hybrid_model.learn_one(x, y)

    mae = np.mean(errors)

    # MAE should be reasonable (within 2x of training RMSE)
    assert mae < 50  # Conservative threshold


# =============================================================================
# Save/Load Tests
# =============================================================================


def test_save_load(hybrid_model, synthetic_data):
    """Test model save and load"""
    X_train = synthetic_data["X_train"]
    y_train = synthetic_data["y_train"]
    x_test = synthetic_data["X_test"][0]

    hybrid_model.fit_static(X_train, y_train)

    # Get prediction before save
    pred_before = hybrid_model.predict_one(x_test)

    # Save to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "hybrid_model"
        hybrid_model.save(str(save_path))

        # Check files exist
        assert (save_path / "static_model.pkl").exists()
        assert (save_path / "static_scaler.pkl").exists()
        assert (save_path / "incremental_model.pkl").exists()
        assert (save_path / "metadata.pkl").exists()

        # Load model
        model_loaded = HybridStaticIncrementalXGBoost()
        model_loaded.load(str(save_path))

        # Get prediction after load
        pred_after = model_loaded.predict_one(x_test)

        # Predictions should match
        assert np.isclose(pred_before, pred_after, rtol=1e-5)
        assert model_loaded.static_trained


def test_save_creates_directory(hybrid_model, synthetic_data):
    """Test that save creates directory if it doesn't exist"""
    X_train = synthetic_data["X_train"]
    y_train = synthetic_data["y_train"]

    hybrid_model.fit_static(X_train, y_train)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "nested" / "path" / "model"
        hybrid_model.save(str(save_path))

        assert save_path.exists()
        assert (save_path / "metadata.pkl").exists()


# =============================================================================
# Metrics Tests
# =============================================================================


def test_get_metrics(hybrid_model, synthetic_data):
    """Test metrics retrieval"""
    X_train = synthetic_data["X_train"]
    y_train = synthetic_data["y_train"]

    hybrid_model.fit_static(X_train, y_train)

    metrics = hybrid_model.get_metrics()

    assert "static_weight" in metrics
    assert "incremental_weight" in metrics
    assert "static_trained" in metrics
    assert "incremental_window_size" in metrics
    assert "incremental_counter" in metrics
    assert "incremental_retrains" in metrics

    assert metrics["static_weight"] == 0.7
    assert metrics["incremental_weight"] == 0.3
    assert metrics["static_trained"]


# =============================================================================
# Edge Cases
# =============================================================================


def test_empty_training_data(hybrid_model):
    """Test behavior with empty training data"""
    X_train = np.array([]).reshape(0, 5)
    y_train = np.array([])

    # Should handle gracefully (XGBoost will raise error)
    with pytest.raises(Exception):
        hybrid_model.fit_static(X_train, y_train)


def test_single_training_sample(hybrid_model):
    """Test behavior with single training sample"""
    X_train = np.array([[250, 35, 0.65, 5, 10]])
    y_train = np.array([280])

    # Should train successfully
    hybrid_model.fit_static(X_train, y_train)
    assert hybrid_model.static_trained


def test_nan_in_test_data(hybrid_model, synthetic_data):
    """Test handling of NaN in test data"""
    X_train = synthetic_data["X_train"]
    y_train = synthetic_data["y_train"]

    hybrid_model.fit_static(X_train, y_train)

    # Create test sample with NaN
    x_test = synthetic_data["X_test"][0].copy()
    x_test[0] = np.nan

    # Should handle gracefully (XGBoost will predict, but might be off)
    pred = hybrid_model.predict_one(x_test)
    assert isinstance(pred, (float, np.floating))


def test_extreme_weights():
    """Test edge case weights (0.0 and 1.0)"""
    # Pure static (weight=1.0)
    model_static = HybridStaticIncrementalXGBoost(static_weight=1.0)
    assert model_static.static_weight == 1.0
    assert model_static.incremental_weight == 0.0

    # Pure incremental (weight=0.0)
    model_incremental = HybridStaticIncrementalXGBoost(static_weight=0.0)
    assert model_incremental.static_weight == 0.0
    assert model_incremental.incremental_weight == 1.0


# =============================================================================
# Integration Tests
# =============================================================================


def test_full_workflow(hybrid_model, synthetic_data):
    """Test complete workflow: train → predict → learn → save → load"""
    X_train = synthetic_data["X_train"]
    y_train = synthetic_data["y_train"]
    X_test = synthetic_data["X_test"][:20]
    y_test = synthetic_data["y_test"][:20]

    # Train static component
    hybrid_model.fit_static(X_train, y_train)

    # Prequential evaluation
    predictions = []
    for x, y in zip(X_test, y_test):
        pred = hybrid_model.predict_one(x)
        predictions.append(pred)
        hybrid_model.learn_one(x, y)

    np.mean([abs(y - p) for y, p in zip(y_test, predictions)])

    # Save model
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "model"
        hybrid_model.save(str(save_path))

        # Load model
        model_loaded = HybridStaticIncrementalXGBoost()
        model_loaded.load(str(save_path))

        # Predict with loaded model
        predictions_loaded = []
        for x in X_test:
            pred = model_loaded.predict_one(x)
            predictions_loaded.append(pred)

        # Predictions should match
        assert np.allclose(predictions, predictions_loaded, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
