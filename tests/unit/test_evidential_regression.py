#!/usr/bin/env python3
"""
Unit Tests for Evidential Deep Learning Module

Tests:
1. NIG loss computation
2. Uncertainty estimates (aleatoric vs epistemic)
3. Calibration on synthetic data
4. OOD detection (high ν for outliers)
5. Model save/load
6. Numerical stability

Author: Claude (Anthropic)
Date: November 9, 2025
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_regression

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from py.models.evidential_regression import (
    EvidentialNeuralNetwork,
    EvidentialLoss,
    EvidentialRegressor
)


class TestEvidentialNeuralNetwork:
    """Test EvidentialNeuralNetwork class"""

    def test_initialization(self):
        """Test model initialization"""
        model = EvidentialNeuralNetwork(
            input_dim=10,
            hidden_dims=(64, 32),
            dropout_rate=0.2
        )

        assert model.input_dim == 10
        assert model.hidden_dims == (64, 32)
        assert model.dropout_rate == 0.2

    def test_forward_pass_shape(self):
        """Test forward pass output shapes"""
        model = EvidentialNeuralNetwork(input_dim=10, hidden_dims=(64, 32))
        x = torch.randn(32, 10)  # batch_size=32, input_dim=10

        gamma, nu, alpha, beta = model(x)

        assert gamma.shape == (32,)
        assert nu.shape == (32,)
        assert alpha.shape == (32,)
        assert beta.shape == (32,)

    def test_forward_pass_constraints(self):
        """Test that output constraints are satisfied"""
        model = EvidentialNeuralNetwork(input_dim=10, hidden_dims=(64, 32))
        x = torch.randn(100, 10)

        gamma, nu, alpha, beta = model(x)

        # ν, α, β must be positive
        assert torch.all(nu > 0), "ν must be positive"
        assert torch.all(alpha > 0), "α must be positive"
        assert torch.all(beta > 0), "β must be positive"

        # α must be > 1 for valid variance
        assert torch.all(alpha > 1), "α must be > 1"

    def test_predict_with_uncertainty(self):
        """Test uncertainty prediction"""
        model = EvidentialNeuralNetwork(input_dim=10, hidden_dims=(64, 32))
        x = torch.randn(32, 10)

        predictions = model.predict_with_uncertainty(x)

        # Check all outputs present
        assert 'mean' in predictions
        assert 'aleatoric_std' in predictions
        assert 'epistemic_std' in predictions
        assert 'total_std' in predictions
        assert 'alpha' in predictions
        assert 'beta' in predictions
        assert 'nu' in predictions

        # Check shapes
        assert predictions['mean'].shape == (32,)
        assert predictions['aleatoric_std'].shape == (32,)
        assert predictions['epistemic_std'].shape == (32,)
        assert predictions['total_std'].shape == (32,)

        # Check uncertainty is positive
        assert torch.all(predictions['aleatoric_std'] > 0)
        assert torch.all(predictions['epistemic_std'] > 0)
        assert torch.all(predictions['total_std'] > 0)

        # Check total >= aleatoric (always true mathematically)
        assert torch.all(
            predictions['total_std'] >= predictions['aleatoric_std'] - 1e-5
        ), "Total uncertainty should be >= aleatoric"

    def test_uncertainty_decomposition(self):
        """Test that total uncertainty = sqrt(aleatoric^2 + epistemic^2)"""
        model = EvidentialNeuralNetwork(input_dim=5, hidden_dims=(32,))
        x = torch.randn(100, 5)

        predictions = model.predict_with_uncertainty(x)

        # Calculate expected total variance
        # Total variance = aleatoric_var * (1 + 1/nu)
        aleatoric_var = predictions['aleatoric_std'] ** 2
        epistemic_var = predictions['epistemic_std'] ** 2
        expected_total_var = aleatoric_var + epistemic_var

        actual_total_var = predictions['total_std'] ** 2

        # Allow small numerical error
        assert torch.allclose(
            actual_total_var,
            expected_total_var,
            rtol=1e-3
        ), "Total variance decomposition incorrect"


class TestEvidentialLoss:
    """Test EvidentialLoss class"""

    def test_loss_initialization(self):
        """Test loss initialization"""
        loss_fn = EvidentialLoss(lambda_reg=0.01, epsilon=1e-8)

        assert loss_fn.lambda_reg == 0.01
        assert loss_fn.epsilon == 1e-8

    def test_loss_computation(self):
        """Test loss computation on simple data"""
        loss_fn = EvidentialLoss(lambda_reg=0.01)

        y_true = torch.randn(32)
        gamma = torch.randn(32)
        nu = torch.abs(torch.randn(32)) + 1.0
        alpha = torch.abs(torch.randn(32)) + 2.0
        beta = torch.abs(torch.randn(32)) + 1.0

        loss = loss_fn(y_true, gamma, nu, alpha, beta)

        # Loss should be positive scalar
        assert loss.ndim == 0, "Loss should be scalar"
        assert loss > 0, "Loss should be positive"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_loss_gradient(self):
        """Test that loss is differentiable"""
        loss_fn = EvidentialLoss(lambda_reg=0.01)

        y_true = torch.randn(32)
        gamma = torch.randn(32, requires_grad=True)
        nu = torch.abs(torch.randn(32)) + 1.0
        nu.requires_grad = True
        alpha = torch.abs(torch.randn(32)) + 2.0
        alpha.requires_grad = True
        beta = torch.abs(torch.randn(32)) + 1.0
        beta.requires_grad = True

        loss = loss_fn(y_true, gamma, nu, alpha, beta)
        loss.backward()

        # All parameters should have gradients
        assert gamma.grad is not None
        assert nu.grad is not None
        assert alpha.grad is not None
        assert beta.grad is not None

        # Gradients should be finite
        assert torch.all(torch.isfinite(gamma.grad))
        assert torch.all(torch.isfinite(nu.grad))
        assert torch.all(torch.isfinite(alpha.grad))
        assert torch.all(torch.isfinite(beta.grad))

    def test_loss_regularization_effect(self):
        """Test that regularization increases loss for large errors"""
        loss_fn_noreg = EvidentialLoss(lambda_reg=0.0)
        loss_fn_reg = EvidentialLoss(lambda_reg=0.1)

        y_true = torch.tensor([10.0])
        gamma = torch.tensor([0.0])  # Large error
        nu = torch.tensor([5.0])
        alpha = torch.tensor([3.0])
        beta = torch.tensor([2.0])

        loss_noreg = loss_fn_noreg(y_true, gamma, nu, alpha, beta)
        loss_reg = loss_fn_reg(y_true, gamma, nu, alpha, beta)

        # Regularized loss should be larger
        assert loss_reg > loss_noreg, "Regularization should increase loss for large errors"

    def test_nig_nll_components(self):
        """Test individual NIG NLL components"""
        loss_fn = EvidentialLoss(lambda_reg=0.0)

        # Simple case
        y_true = torch.tensor([1.0])
        gamma = torch.tensor([1.0])  # Perfect prediction
        nu = torch.tensor([10.0])
        alpha = torch.tensor([5.0])
        beta = torch.tensor([2.0])

        nll = loss_fn._nig_nll(y_true, gamma, nu, alpha, beta)

        assert torch.isfinite(nll), "NLL should be finite"
        assert nll > 0, "NLL should be positive (for perfect prediction)"


class TestEvidentialRegressor:
    """Test EvidentialRegressor class"""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic regression data"""
        np.random.seed(42)
        X, y = make_regression(
            n_samples=200,
            n_features=10,
            n_informative=8,
            noise=5.0,
            random_state=42
        )
        return X, y

    def test_initialization(self):
        """Test regressor initialization"""
        model = EvidentialRegressor(
            hidden_dims=(64, 32),
            dropout_rate=0.2,
            lambda_reg=0.01,
            learning_rate=1e-3,
            batch_size=32,
            max_epochs=10,
            patience=5,
            device='cpu',
            random_state=42
        )

        assert model.hidden_dims == (64, 32)
        assert model.dropout_rate == 0.2
        assert model.lambda_reg == 0.01
        assert model.learning_rate == 1e-3
        assert model.batch_size == 32
        assert model.max_epochs == 10
        assert model.patience == 5
        assert model.device == 'cpu'
        assert model.random_state == 42

    def test_fit_predict(self, synthetic_data):
        """Test fitting and prediction"""
        X, y = synthetic_data

        # Split data
        split = 150
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Train model
        model = EvidentialRegressor(
            hidden_dims=(32, 16),
            max_epochs=20,
            patience=5,
            random_state=42
        )

        model.fit(X_train, y_train, verbose=False)

        # Make predictions
        y_pred = model.predict(X_test)

        # Check output shape
        assert y_pred.shape == y_test.shape

        # Check predictions are reasonable
        assert np.all(np.isfinite(y_pred))

    def test_predict_with_uncertainty(self, synthetic_data):
        """Test prediction with uncertainty quantification"""
        X, y = synthetic_data

        split = 150
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = EvidentialRegressor(
            hidden_dims=(32, 16),
            max_epochs=20,
            patience=5,
            random_state=42
        )

        model.fit(X_train, y_train, verbose=False)

        # Predict with uncertainty
        predictions = model.predict(X_test, return_uncertainty=True)

        # Check all keys present
        assert 'mean' in predictions
        assert 'aleatoric_std' in predictions
        assert 'epistemic_std' in predictions
        assert 'total_std' in predictions

        # Check shapes
        assert predictions['mean'].shape == y_test.shape
        assert predictions['aleatoric_std'].shape == y_test.shape
        assert predictions['epistemic_std'].shape == y_test.shape
        assert predictions['total_std'].shape == y_test.shape

        # Check uncertainty is positive
        assert np.all(predictions['aleatoric_std'] > 0)
        assert np.all(predictions['epistemic_std'] > 0)
        assert np.all(predictions['total_std'] > 0)

    def test_calibration_on_synthetic_data(self, synthetic_data):
        """Test calibration on synthetic data"""
        X, y = synthetic_data

        split = 150
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = EvidentialRegressor(
            hidden_dims=(64, 32),
            max_epochs=50,
            patience=10,
            random_state=42
        )

        model.fit(X_train, y_train, verbose=False)

        predictions = model.predict(X_test, return_uncertainty=True)

        # Calculate z-scores
        z_scores = (y_test - predictions['mean']) / predictions['total_std']

        # Check empirical coverage
        # 68% of points should be within ±1σ
        in_1sigma = np.abs(z_scores) < 1.0
        coverage_68 = in_1sigma.mean()

        # 95% of points should be within ±2σ
        in_2sigma = np.abs(z_scores) < 2.0
        coverage_95 = in_2sigma.mean()

        # Allow some tolerance for small test set
        assert 0.50 < coverage_68 < 0.85, f"68% coverage: {coverage_68:.2f}"
        assert 0.80 < coverage_95 < 1.00, f"95% coverage: {coverage_95:.2f}"

    def test_ood_detection(self, synthetic_data):
        """Test OOD detection via epistemic uncertainty"""
        X, y = synthetic_data

        split = 150
        X_train, X_test = X[:split], X[split:]
        y_train, _ = y[:split], y[split:]

        model = EvidentialRegressor(
            hidden_dims=(32, 16),
            max_epochs=20,
            patience=5,
            random_state=42
        )

        model.fit(X_train, y_train, verbose=False)

        # In-distribution predictions
        pred_in = model.predict(X_test, return_uncertainty=True)

        # Out-of-distribution predictions (10× feature values)
        X_ood = X_test * 10
        pred_ood = model.predict(X_ood, return_uncertainty=True)

        # OOD should have higher epistemic uncertainty
        mean_epistemic_in = pred_in['epistemic_std'].mean()
        mean_epistemic_ood = pred_ood['epistemic_std'].mean()

        assert mean_epistemic_ood > mean_epistemic_in, \
            "OOD samples should have higher epistemic uncertainty"

    def test_save_load(self, synthetic_data, tmp_path):
        """Test model save/load"""
        X, y = synthetic_data

        split = 150
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Train model
        model = EvidentialRegressor(
            hidden_dims=(32, 16),
            max_epochs=10,
            patience=5,
            random_state=42
        )

        model.fit(X_train, y_train, verbose=False)

        # Make predictions before save
        y_pred_before = model.predict(X_test, return_uncertainty=True)

        # Save model
        model_path = tmp_path / "test_model.pt"
        model.save(str(model_path))

        # Load model
        model_loaded = EvidentialRegressor()
        model_loaded.load(str(model_path))

        # Make predictions after load
        y_pred_after = model_loaded.predict(X_test, return_uncertainty=True)

        # Predictions should be identical
        np.testing.assert_allclose(
            y_pred_before['mean'],
            y_pred_after['mean'],
            rtol=1e-5
        )
        np.testing.assert_allclose(
            y_pred_before['total_std'],
            y_pred_after['total_std'],
            rtol=1e-5
        )

    def test_early_stopping(self, synthetic_data):
        """Test early stopping functionality"""
        X, y = synthetic_data

        split = 150
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        model = EvidentialRegressor(
            hidden_dims=(32, 16),
            max_epochs=100,
            patience=3,  # Low patience for early stopping
            random_state=42
        )

        model.fit(X_train, y_train, X_val, y_val, verbose=False)

        # Should stop early (well before 100 epochs)
        n_epochs_trained = len(model.training_history_)
        assert n_epochs_trained < 100, "Early stopping should trigger"

    def test_numerical_stability(self):
        """Test numerical stability with extreme values"""
        np.random.seed(42)

        # Create data with extreme values
        X = np.random.randn(100, 5) * 1000  # Large feature values
        y = np.random.randn(100) * 1000  # Large target values

        model = EvidentialRegressor(
            hidden_dims=(16,),
            max_epochs=10,
            random_state=42
        )

        # Should not crash or produce NaNs
        model.fit(X, y, verbose=False)

        # Predictions should be finite
        y_pred = model.predict(X, return_uncertainty=True)
        assert np.all(np.isfinite(y_pred['mean']))
        assert np.all(np.isfinite(y_pred['total_std']))


class TestIntegration:
    """Integration tests"""

    def test_end_to_end_workflow(self):
        """Test complete workflow from data to predictions"""
        np.random.seed(42)

        # Generate synthetic NFL-like data
        n_samples = 500
        n_features = 5

        # Features: carries, avg_rushing_l3, avg_carries_l3, season_avg, rushing_tds
        X = np.random.randn(n_samples, n_features)
        X[:, 0] = np.abs(X[:, 0]) * 5 + 10  # carries (10-25)
        X[:, 1] = np.abs(X[:, 1]) * 20 + 50  # avg_rushing_l3 (30-90)
        X[:, 2] = np.abs(X[:, 2]) * 3 + 10  # avg_carries_l3 (7-16)
        X[:, 3] = np.abs(X[:, 3]) * 20 + 50  # season_avg (30-90)
        X[:, 4] = np.abs(X[:, 4]) * 0.3  # rushing_tds (0-1)

        # Target: rushing yards (linear + noise)
        y = (
            5 * X[:, 0] +  # 5 yards per carry
            0.3 * X[:, 1] +  # Some regression to L3 mean
            np.random.randn(n_samples) * 15  # Noise
        )

        # Split
        split = 400
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Train
        model = EvidentialRegressor(
            hidden_dims=(64, 32),
            max_epochs=30,
            patience=10,
            random_state=42
        )

        model.fit(X_train, y_train, verbose=False)

        # Predict
        predictions = model.predict(X_test, return_uncertainty=True)

        # Validate
        mae = np.mean(np.abs(predictions['mean'] - y_test))
        assert mae < 30, f"MAE should be reasonable: {mae:.2f}"

        # Check calibration
        z_scores = (y_test - predictions['mean']) / predictions['total_std']
        coverage_68 = (np.abs(z_scores) < 1.0).mean()
        assert 0.50 < coverage_68 < 0.85, f"Calibration: {coverage_68:.2%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
