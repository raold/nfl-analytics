#!/usr/bin/env python3
"""
Unit Tests for Dirichlet Bayesian Stacking
==========================================

Tests the DirichletBayesianStacking class for correctness, numerical stability,
and expected behavior.

Coverage targets:
- Initialization (alpha values, weight validation)
- Prediction (weighted average correctness)
- Update (Bayesian update mechanics)
- Convergence (weights → optimal as data grows)
- Edge cases (single component, extreme priors, numerical stability)

Author: Claude Code (Sonnet 4.5)
Date: November 22, 2025
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from py.ensemble.dirichlet_bayesian_stacking import DirichletBayesianStacking


class TestInitialization:
    """Test DirichletBayesianStacking initialization."""

    def test_initialization_basic(self):
        """Test basic initialization with valid inputs."""
        component_names = ["model_a", "model_b", "model_c"]
        prior_weights = {"model_a": 0.5, "model_b": 0.3, "model_c": 0.2}

        stacker = DirichletBayesianStacking(
            component_names=component_names,
            prior_weights=prior_weights,
            prior_strength=10.0,
            likelihood_scale=40.0
        )

        # Check initialization
        assert stacker.K == 3
        assert stacker.num_updates == 0
        assert len(stacker.prediction_history) == 0

        # Check alpha initialization: α_i = prior_strength * prior_weights[i]
        assert np.isclose(stacker.alpha["model_a"], 10.0 * 0.5)
        assert np.isclose(stacker.alpha["model_b"], 10.0 * 0.3)
        assert np.isclose(stacker.alpha["model_c"], 10.0 * 0.2)

        # Check weights match prior
        weights = stacker.get_weights()
        assert np.isclose(weights["model_a"], 0.5)
        assert np.isclose(weights["model_b"], 0.3)
        assert np.isclose(weights["model_c"], 0.2)

    def test_initialization_invalid_prior_weights_sum(self):
        """Test initialization fails if prior weights don't sum to 1.0."""
        component_names = ["model_a", "model_b"]
        prior_weights = {"model_a": 0.5, "model_b": 0.4}  # Sum = 0.9

        with pytest.raises(ValueError, match="Prior weights must sum to 1.0"):
            DirichletBayesianStacking(
                component_names=component_names,
                prior_weights=prior_weights
            )

    def test_initialization_mismatched_components(self):
        """Test initialization fails if prior weights keys don't match components."""
        component_names = ["model_a", "model_b"]
        prior_weights = {"model_a": 0.5, "model_c": 0.5}  # model_c not in components

        with pytest.raises(ValueError, match="don't match component names"):
            DirichletBayesianStacking(
                component_names=component_names,
                prior_weights=prior_weights
            )


class TestPrediction:
    """Test ensemble prediction logic."""

    def test_predict_weighted_average(self):
        """Test that prediction is correct weighted average."""
        component_names = ["model_a", "model_b", "model_c"]
        prior_weights = {"model_a": 0.5, "model_b": 0.3, "model_c": 0.2}

        stacker = DirichletBayesianStacking(
            component_names=component_names,
            prior_weights=prior_weights
        )

        # Predict with known values
        component_preds = {"model_a": 100.0, "model_b": 200.0, "model_c": 300.0}

        ensemble_pred, weights = stacker.predict(component_preds)

        # Expected: 0.5*100 + 0.3*200 + 0.2*300 = 50 + 60 + 60 = 170
        expected = 0.5 * 100.0 + 0.3 * 200.0 + 0.2 * 300.0
        assert np.isclose(ensemble_pred, expected)
        assert np.isclose(sum(weights.values()), 1.0)

    def test_predict_invalid_components(self):
        """Test prediction fails if component keys don't match."""
        component_names = ["model_a", "model_b"]
        prior_weights = {"model_a": 0.5, "model_b": 0.5}

        stacker = DirichletBayesianStacking(
            component_names=component_names,
            prior_weights=prior_weights
        )

        # Wrong component name
        component_preds = {"model_a": 100.0, "model_c": 200.0}

        with pytest.raises(ValueError, match="don't match component names"):
            stacker.predict(component_preds)


class TestUpdate:
    """Test Bayesian update mechanics."""

    def test_update_increases_alpha_for_good_prediction(self):
        """Test that alpha increases more for components with good predictions."""
        component_names = ["model_a", "model_b"]
        prior_weights = {"model_a": 0.5, "model_b": 0.5}

        stacker = DirichletBayesianStacking(
            component_names=component_names,
            prior_weights=prior_weights,
            prior_strength=10.0,
            likelihood_scale=40.0
        )

        # Store initial alpha
        alpha_a_before = stacker.alpha["model_a"]
        alpha_b_before = stacker.alpha["model_b"]

        # model_a predicts well (error = 5), model_b predicts poorly (error = 50)
        component_preds = {"model_a": 245.0, "model_b": 200.0}
        actual = 250.0

        stacker.update(component_preds, actual)

        # Check alpha increased
        assert stacker.alpha["model_a"] > alpha_a_before
        assert stacker.alpha["model_b"] > alpha_b_before

        # Check model_a alpha increased more (smaller error)
        alpha_a_increase = stacker.alpha["model_a"] - alpha_a_before
        alpha_b_increase = stacker.alpha["model_b"] - alpha_b_before
        assert alpha_a_increase > alpha_b_increase

    def test_update_weights_adapt(self):
        """Test that weights adapt after update."""
        component_names = ["model_a", "model_b"]
        prior_weights = {"model_a": 0.5, "model_b": 0.5}

        stacker = DirichletBayesianStacking(
            component_names=component_names,
            prior_weights=prior_weights,
            prior_strength=5.0  # Weak prior for faster adaptation
        )

        # Store initial weights
        weights_before = stacker.get_weights()

        # model_a consistently predicts well
        for _ in range(10):
            component_preds = {"model_a": 245.0, "model_b": 200.0}
            actual = 250.0
            stacker.update(component_preds, actual)

        weights_after = stacker.get_weights()

        # model_a weight should increase
        assert weights_after["model_a"] > weights_before["model_a"]
        assert weights_after["model_b"] < weights_before["model_b"]

    def test_update_history_tracking(self):
        """Test that prediction history is tracked correctly."""
        component_names = ["model_a", "model_b"]
        prior_weights = {"model_a": 0.5, "model_b": 0.5}

        stacker = DirichletBayesianStacking(
            component_names=component_names,
            prior_weights=prior_weights
        )

        # Make 3 updates
        for i in range(3):
            component_preds = {"model_a": 240.0 + i * 10, "model_b": 250.0 + i * 5}
            actual = 245.0 + i * 7
            stacker.update(component_preds, actual)

        # Check history
        assert len(stacker.prediction_history) == 3
        assert stacker.num_updates == 3

        # Check first history entry
        first_entry = stacker.prediction_history[0]
        assert "predictions" in first_entry
        assert "actual" in first_entry
        assert "weights_before" in first_entry
        assert "weights_after" in first_entry
        assert "likelihoods" in first_entry


class TestScoringRules:
    """Test different scoring rule implementations."""

    def test_exponential_scoring_rule(self):
        """Test exponential scoring rule."""
        component_names = ["model_a"]
        prior_weights = {"model_a": 1.0}

        stacker = DirichletBayesianStacking(
            component_names=component_names,
            prior_weights=prior_weights,
            prior_strength=10.0,
            likelihood_scale=40.0,
            scoring_rule="exponential"
        )

        # Perfect prediction → likelihood ≈ 1.0
        component_preds = {"model_a": 250.0}
        actual = 250.0
        stacker.update(component_preds, actual)

        history = stacker.get_prediction_history()
        assert np.isclose(history[0]["likelihoods"]["model_a"], 1.0, atol=0.01)

    def test_squared_scoring_rule(self):
        """Test squared (Gaussian) scoring rule."""
        component_names = ["model_a"]
        prior_weights = {"model_a": 1.0}

        stacker = DirichletBayesianStacking(
            component_names=component_names,
            prior_weights=prior_weights,
            prior_strength=10.0,
            likelihood_scale=40.0,
            scoring_rule="squared"
        )

        # Perfect prediction → likelihood = 1.0
        component_preds = {"model_a": 250.0}
        actual = 250.0
        stacker.update(component_preds, actual)

        history = stacker.get_prediction_history()
        assert np.isclose(history[0]["likelihoods"]["model_a"], 1.0)

    def test_log_scoring_rule(self):
        """Test log scoring rule."""
        component_names = ["model_a"]
        prior_weights = {"model_a": 1.0}

        stacker = DirichletBayesianStacking(
            component_names=component_names,
            prior_weights=prior_weights,
            prior_strength=10.0,
            likelihood_scale=40.0,
            scoring_rule="log"
        )

        # Perfect prediction → likelihood = 1.0
        component_preds = {"model_a": 250.0}
        actual = 250.0
        stacker.update(component_preds, actual)

        history = stacker.get_prediction_history()
        assert np.isclose(history[0]["likelihoods"]["model_a"], 1.0)


class TestConvergence:
    """Test convergence properties."""

    def test_convergence_to_best_model(self):
        """Test that weights converge to best performing model."""
        component_names = ["model_good", "model_bad"]
        prior_weights = {"model_good": 0.5, "model_bad": 0.5}

        stacker = DirichletBayesianStacking(
            component_names=component_names,
            prior_weights=prior_weights,
            prior_strength=5.0  # Weak prior for faster convergence
        )

        # Simulate 100 predictions where model_good always better
        np.random.seed(42)
        for _ in range(100):
            actual = np.random.uniform(220, 280)
            component_preds = {
                "model_good": actual + np.random.normal(0, 5),  # Small error
                "model_bad": actual + np.random.normal(0, 50),  # Large error
            }
            stacker.update(component_preds, actual)

        # model_good weight should be much higher
        weights = stacker.get_weights()
        assert weights["model_good"] > 0.6  # Expect >60% weight (realistic for 100 samples)
        assert weights["model_bad"] < 0.4

    def test_convergence_numerical_stability(self):
        """Test that alpha values don't overflow after many updates."""
        component_names = ["model_a", "model_b"]
        prior_weights = {"model_a": 0.5, "model_b": 0.5}

        stacker = DirichletBayesianStacking(
            component_names=component_names,
            prior_weights=prior_weights
        )

        # Simulate 1000 updates
        np.random.seed(42)
        for _ in range(1000):
            actual = np.random.uniform(220, 280)
            component_preds = {
                "model_a": actual + np.random.normal(0, 20),
                "model_b": actual + np.random.normal(0, 20),
            }
            stacker.update(component_preds, actual)

        # Check alpha values are finite
        for name in component_names:
            assert np.isfinite(stacker.alpha[name])
            assert stacker.alpha[name] > 0

        # Check weights still sum to 1.0
        weights = stacker.get_weights()
        assert np.isclose(sum(weights.values()), 1.0)


class TestDiagnostics:
    """Test diagnostic methods."""

    def test_get_diagnostics(self):
        """Test get_diagnostics returns expected fields."""
        component_names = ["model_a", "model_b"]
        prior_weights = {"model_a": 0.5, "model_b": 0.5}

        stacker = DirichletBayesianStacking(
            component_names=component_names,
            prior_weights=prior_weights
        )

        diagnostics = stacker.get_diagnostics()

        # Check all expected fields present
        assert "alpha" in diagnostics
        assert "weights" in diagnostics
        assert "alpha_sum" in diagnostics
        assert "num_updates" in diagnostics
        assert "weight_variance" in diagnostics
        assert "alpha_concentration" in diagnostics

    def test_get_weight_history(self):
        """Test weight history tracking."""
        component_names = ["model_a", "model_b"]
        prior_weights = {"model_a": 0.5, "model_b": 0.5}

        stacker = DirichletBayesianStacking(
            component_names=component_names,
            prior_weights=prior_weights
        )

        # Make 3 updates
        for i in range(3):
            component_preds = {"model_a": 240.0, "model_b": 260.0}
            actual = 250.0
            stacker.update(component_preds, actual)

        weight_history = stacker.get_weight_history()

        # Check history length
        assert len(weight_history) == 3

        # Check each entry is a dict of weights
        for weights in weight_history:
            assert isinstance(weights, dict)
            assert set(weights.keys()) == set(component_names)
            assert np.isclose(sum(weights.values()), 1.0)


class TestPersistence:
    """Test save/load functionality."""

    def test_save_and_load(self):
        """Test that stacker can be saved and loaded."""
        component_names = ["model_a", "model_b"]
        prior_weights = {"model_a": 0.5, "model_b": 0.5}

        stacker = DirichletBayesianStacking(
            component_names=component_names,
            prior_weights=prior_weights,
            prior_strength=10.0,
            likelihood_scale=40.0
        )

        # Make some updates
        for _ in range(5):
            component_preds = {"model_a": 240.0, "model_b": 260.0}
            actual = 250.0
            stacker.update(component_preds, actual)

        # Save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "stacker.json"
            stacker.save(str(filepath))

            # Load from file
            loaded_stacker = DirichletBayesianStacking.load(str(filepath))

            # Check state matches
            assert loaded_stacker.component_names == stacker.component_names
            assert loaded_stacker.num_updates == stacker.num_updates
            assert loaded_stacker.alpha == stacker.alpha
            assert loaded_stacker.prior_strength == stacker.prior_strength
            assert loaded_stacker.likelihood_scale == stacker.likelihood_scale

            # Check weights match
            original_weights = stacker.get_weights()
            loaded_weights = loaded_stacker.get_weights()
            for name in component_names:
                assert np.isclose(
                    original_weights[name],
                    loaded_weights[name]
                )

    def test_load_nonexistent_file(self):
        """Test load fails gracefully for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            DirichletBayesianStacking.load("/nonexistent/path.json")


class TestReset:
    """Test reset functionality."""

    def test_reset_to_original_prior(self):
        """Test reset restores original prior."""
        component_names = ["model_a", "model_b"]
        prior_weights = {"model_a": 0.5, "model_b": 0.5}

        stacker = DirichletBayesianStacking(
            component_names=component_names,
            prior_weights=prior_weights,
            prior_strength=10.0
        )

        # Store initial weights
        initial_weights = stacker.get_weights()

        # Make updates where model_a consistently better
        for _ in range(10):
            component_preds = {"model_a": 248.0, "model_b": 260.0}  # model_a closer
            actual = 250.0
            stacker.update(component_preds, actual)

        # Weights should have changed (model_a should increase)
        updated_weights = stacker.get_weights()
        assert updated_weights["model_a"] > initial_weights["model_a"]

        # Reset
        stacker.reset()

        # Weights should match initial
        reset_weights = stacker.get_weights()
        assert np.allclose(
            list(initial_weights.values()),
            list(reset_weights.values())
        )
        assert stacker.num_updates == 0

    def test_reset_with_new_prior(self):
        """Test reset with new prior weights."""
        component_names = ["model_a", "model_b"]
        prior_weights = {"model_a": 0.5, "model_b": 0.5}

        stacker = DirichletBayesianStacking(
            component_names=component_names,
            prior_weights=prior_weights
        )

        # Reset with new prior
        new_prior = {"model_a": 0.7, "model_b": 0.3}
        stacker.reset(prior_weights=new_prior)

        # Check weights match new prior
        weights = stacker.get_weights()
        assert np.isclose(weights["model_a"], 0.7)
        assert np.isclose(weights["model_b"], 0.3)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_component(self):
        """Test with single component (weight should stay 1.0)."""
        component_names = ["model_a"]
        prior_weights = {"model_a": 1.0}

        stacker = DirichletBayesianStacking(
            component_names=component_names,
            prior_weights=prior_weights
        )

        # Make updates
        for _ in range(10):
            component_preds = {"model_a": 250.0}
            actual = 250.0
            stacker.update(component_preds, actual)

        # Weight should still be 1.0
        weights = stacker.get_weights()
        assert np.isclose(weights["model_a"], 1.0)

    def test_extreme_prior_strength(self):
        """Test with extreme prior strength (very strong prior)."""
        component_names = ["model_a", "model_b"]
        prior_weights = {"model_a": 0.5, "model_b": 0.5}

        stacker = DirichletBayesianStacking(
            component_names=component_names,
            prior_weights=prior_weights,
            prior_strength=1000.0  # Very strong prior
        )

        # Store initial weights
        initial_weights = stacker.get_weights()

        # Make 10 updates where model_a is clearly better
        for _ in range(10):
            component_preds = {"model_a": 250.0, "model_b": 200.0}
            actual = 250.0
            stacker.update(component_preds, actual)

        # Weights should barely change (strong prior resists updates)
        updated_weights = stacker.get_weights()
        assert abs(updated_weights["model_a"] - initial_weights["model_a"]) < 0.05

    def test_perfect_predictions_all_components(self):
        """Test when all components predict perfectly."""
        component_names = ["model_a", "model_b"]
        prior_weights = {"model_a": 0.5, "model_b": 0.5}

        stacker = DirichletBayesianStacking(
            component_names=component_names,
            prior_weights=prior_weights
        )

        # All components predict perfectly
        for _ in range(10):
            actual = 250.0
            component_preds = {"model_a": actual, "model_b": actual}
            stacker.update(component_preds, actual)

        # Weights should remain approximately equal
        weights = stacker.get_weights()
        assert abs(weights["model_a"] - weights["model_b"]) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
