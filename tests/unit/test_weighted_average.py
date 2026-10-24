#!/usr/bin/env python3
"""
Unit tests for WeightedAverageStrategy with correlation-aware uncertainty.

Tests the critical fix for Task 7: correlation-aware uncertainty pooling.
Validates that ensemble uncertainty correctly accounts for model correlations.

Author: NFL Analytics Team
Date: November 9, 2025
"""


import numpy as np
import pytest

from nfl_analytics.ensemble.strategy.base import ModelPrediction, ModelType
from nfl_analytics.ensemble.strategy.weighted_average import WeightedAverageStrategy


class TestWeightedAverageStrategy:
    """Test weighted average strategy with correlation-aware uncertainty."""

    def test_initialization_with_custom_weights(self):
        """Test initialization with custom weights."""
        weights = {"model1": 0.5, "model2": 0.3, "model3": 0.2}
        strategy = WeightedAverageStrategy(weights=weights)

        assert strategy.weights == weights
        assert strategy.name == "weighted_average"

    def test_initialization_normalizes_weights(self):
        """Test that weights are normalized to sum to 1.0."""
        weights = {"model1": 0.5, "model2": 0.3, "model3": 0.3}  # Sum = 1.1
        strategy = WeightedAverageStrategy(weights=weights)

        # Weights should be normalized
        total = sum(strategy.weights.values())
        assert np.isclose(total, 1.0)

        # Proportions should be preserved
        assert np.isclose(strategy.weights["model1"], 0.5 / 1.1)
        assert np.isclose(strategy.weights["model2"], 0.3 / 1.1)
        assert np.isclose(strategy.weights["model3"], 0.3 / 1.1)

    def test_default_correlations_structure(self):
        """Test that default correlations have correct structure."""
        strategy = WeightedAverageStrategy()
        correlations = strategy.correlations

        # Should have correlations for key model pairs
        assert ("hierarchical_v2", "state_space_v2") in correlations
        assert ("bnn_v1", "xgboost_v1") in correlations

        # All correlations should be between 0 and 1
        for corr in correlations.values():
            assert 0 <= corr <= 1

        # Bayesian models should be highly correlated
        assert correlations[("hierarchical_v2", "state_space_v2")] > 0.7

        # XGBoost should be less correlated with Bayesian models
        assert correlations[("bnn_v1", "xgboost_v1")] < 0.5

    def test_weighted_mean_calculation(self):
        """Test that weighted mean is calculated correctly."""
        weights = {"model1": 0.5, "model2": 0.3, "model3": 0.2}
        strategy = WeightedAverageStrategy(weights=weights)

        predictions = [
            ModelPrediction("model1", ModelType.STATE_SPACE, 250.0, 20.0, 0.9),
            ModelPrediction("model2", ModelType.BNN, 260.0, 15.0, 0.85),
            ModelPrediction("model3", ModelType.XGBOOST, 255.0, 18.0, 0.88),
        ]

        result = strategy.combine(predictions)

        # Expected mean: 0.5*250 + 0.3*260 + 0.2*255 = 125 + 78 + 51 = 254
        expected_mean = 0.5 * 250 + 0.3 * 260 + 0.2 * 255
        assert np.isclose(result.prediction_mean, expected_mean)

    def test_uncertainty_with_zero_correlation(self):
        """Test uncertainty calculation with zero correlation (independent models)."""
        weights = {"model1": 0.5, "model2": 0.5}
        correlations = {("model1", "model2"): 0.0}  # Independent
        strategy = WeightedAverageStrategy(weights=weights, model_correlations=correlations)

        predictions = [
            ModelPrediction("model1", ModelType.STATE_SPACE, 250.0, 20.0, 0.9),
            ModelPrediction("model2", ModelType.BNN, 260.0, 20.0, 0.9),
        ]

        result = strategy.combine(predictions)

        # Expected std (independent): sqrt(0.5^2 * 20^2 + 0.5^2 * 20^2)
        # = sqrt(0.25 * 400 + 0.25 * 400) = sqrt(200) ≈ 14.14
        expected_std = np.sqrt(0.5**2 * 20**2 + 0.5**2 * 20**2)
        assert np.isclose(result.prediction_std, expected_std, rtol=0.01)

    def test_uncertainty_with_perfect_correlation(self):
        """Test uncertainty calculation with perfect correlation (ρ=1.0)."""
        weights = {"model1": 0.5, "model2": 0.5}
        correlations = {("model1", "model2"): 1.0}  # Perfect correlation
        strategy = WeightedAverageStrategy(weights=weights, model_correlations=correlations)

        predictions = [
            ModelPrediction("model1", ModelType.STATE_SPACE, 250.0, 20.0, 0.9),
            ModelPrediction("model2", ModelType.BNN, 260.0, 20.0, 0.9),
        ]

        result = strategy.combine(predictions)

        # Expected std (perfect correlation):
        # σ² = Σ(w_i² * σ_i²) + 2 * w1 * w2 * ρ * σ1 * σ2
        # = 0.5^2 * 20^2 + 0.5^2 * 20^2 + 2 * 0.5 * 0.5 * 1.0 * 20 * 20
        # = 100 + 100 + 200 = 400
        # σ = 20
        expected_variance = 0.5**2 * 20**2 + 0.5**2 * 20**2 + 2 * 0.5 * 0.5 * 1.0 * 20 * 20
        expected_std = np.sqrt(expected_variance)
        assert np.isclose(result.prediction_std, expected_std, rtol=0.01)
        assert np.isclose(result.prediction_std, 20.0, rtol=0.01)

    def test_uncertainty_with_realistic_correlation(self):
        """Test uncertainty with realistic Bayesian model correlation (ρ=0.75)."""
        weights = {
            "state_space_v2": 0.30,
            "hierarchical_v2": 0.25,
            "bnn_v1": 0.25,
            "xgboost_v1": 0.20,
        }

        # Use realistic correlations from empirical backtest
        correlations = {
            ("hierarchical_v2", "state_space_v2"): 0.75,
            ("bnn_v1", "state_space_v2"): 0.65,
            ("bnn_v1", "hierarchical_v2"): 0.68,
            ("state_space_v2", "xgboost_v1"): 0.45,
            ("hierarchical_v2", "xgboost_v1"): 0.48,
            ("bnn_v1", "xgboost_v1"): 0.42,
        }

        strategy = WeightedAverageStrategy(weights=weights, model_correlations=correlations)

        predictions = [
            ModelPrediction("state_space_v2", ModelType.STATE_SPACE, 250.0, 20.0, 0.90),
            ModelPrediction("hierarchical_v2", ModelType.HIERARCHICAL, 255.0, 18.0, 0.88),
            ModelPrediction("bnn_v1", ModelType.BNN, 260.0, 22.0, 0.85),
            ModelPrediction("xgboost_v1", ModelType.XGBOOST, 245.0, 25.0, 0.82),
        ]

        result = strategy.combine(predictions)

        # Calculate expected variance manually
        w = [0.30, 0.25, 0.25, 0.20]
        sigma = [20.0, 18.0, 22.0, 25.0]

        # Diagonal terms
        variance = sum(w[i] ** 2 * sigma[i] ** 2 for i in range(4))

        # Off-diagonal terms (6 pairs for 4 models)
        pairs = [
            (0, 1, 0.75),  # state_space - hierarchical
            (0, 2, 0.65),  # state_space - bnn
            (0, 3, 0.45),  # state_space - xgboost
            (1, 2, 0.68),  # hierarchical - bnn
            (1, 3, 0.48),  # hierarchical - xgboost
            (2, 3, 0.42),  # bnn - xgboost
        ]

        for i, j, rho in pairs:
            variance += 2 * w[i] * w[j] * rho * sigma[i] * sigma[j]

        expected_std = np.sqrt(variance)

        # Verify uncertainty is higher than independence assumption
        variance_independent = sum(w[i] ** 2 * sigma[i] ** 2 for i in range(4))
        std_independent = np.sqrt(variance_independent)

        assert result.prediction_std > std_independent
        assert np.isclose(result.prediction_std, expected_std, rtol=0.01)

    def test_uncertainty_increases_with_correlation(self):
        """Test that uncertainty increases as correlation increases."""
        weights = {"model1": 0.5, "model2": 0.5}

        predictions = [
            ModelPrediction("model1", ModelType.STATE_SPACE, 250.0, 20.0, 0.9),
            ModelPrediction("model2", ModelType.BNN, 260.0, 20.0, 0.9),
        ]

        # Test with increasing correlations
        correlations = [0.0, 0.5, 0.75, 1.0]
        stds = []

        for rho in correlations:
            strategy = WeightedAverageStrategy(
                weights=weights, model_correlations={("model1", "model2"): rho}
            )
            result = strategy.combine(predictions)
            stds.append(result.prediction_std)

        # Verify uncertainty increases monotonically with correlation
        for i in range(len(stds) - 1):
            assert stds[i] < stds[i + 1], f"std[{i}]={stds[i]} should be < std[{i+1}]={stds[i+1]}"

    def test_confidence_based_weights(self):
        """Test confidence-based weighting when no weights provided."""
        strategy = WeightedAverageStrategy()  # No weights

        predictions = [
            ModelPrediction("model1", ModelType.STATE_SPACE, 250.0, 20.0, 0.9),
            ModelPrediction("model2", ModelType.BNN, 260.0, 15.0, 0.6),
            ModelPrediction("model3", ModelType.XGBOOST, 255.0, 18.0, 0.3),
        ]

        result = strategy.combine(predictions)

        # Weights should be proportional to confidence
        # Total confidence = 0.9 + 0.6 + 0.3 = 1.8
        expected_w1 = 0.9 / 1.8  # 0.5
        expected_w2 = 0.6 / 1.8  # 0.333
        expected_w3 = 0.3 / 1.8  # 0.167

        assert np.isclose(result.weights["model1"], expected_w1, rtol=0.01)
        assert np.isclose(result.weights["model2"], expected_w2, rtol=0.01)
        assert np.isclose(result.weights["model3"], expected_w3, rtol=0.01)

    def test_fallback_correlation_default(self):
        """Test that missing correlations default to 0.5."""
        weights = {"model_new": 0.5, "model_old": 0.5}
        correlations = {}  # No correlations specified

        strategy = WeightedAverageStrategy(weights=weights, model_correlations=correlations)

        predictions = [
            ModelPrediction("model_new", ModelType.STATE_SPACE, 250.0, 20.0, 0.9),
            ModelPrediction("model_old", ModelType.BNN, 260.0, 20.0, 0.9),
        ]

        result = strategy.combine(predictions)

        # Should use default correlation of 0.5
        # σ² = 0.5^2 * 20^2 + 0.5^2 * 20^2 + 2 * 0.5 * 0.5 * 0.5 * 20 * 20
        # = 100 + 100 + 100 = 300
        expected_variance = 0.5**2 * 20**2 + 0.5**2 * 20**2 + 2 * 0.5 * 0.5 * 0.5 * 20 * 20
        expected_std = np.sqrt(expected_variance)
        assert np.isclose(result.prediction_std, expected_std, rtol=0.01)

    def test_metadata_includes_num_models(self):
        """Test that metadata includes number of models."""
        strategy = WeightedAverageStrategy(weights={"m1": 0.5, "m2": 0.3, "m3": 0.2})

        predictions = [
            ModelPrediction("m1", ModelType.STATE_SPACE, 250.0, 20.0, 0.9),
            ModelPrediction("m2", ModelType.BNN, 260.0, 15.0, 0.85),
            ModelPrediction("m3", ModelType.XGBOOST, 255.0, 18.0, 0.88),
        ]

        result = strategy.combine(predictions)

        assert result.metadata["num_models"] == 3
        assert result.metadata["method"] == "weighted_average"
        assert result.metadata["weighting_scheme"] == "fixed"


class TestCorrelationAwareUncertaintyEdgeCases:
    """Test edge cases for correlation-aware uncertainty calculation."""

    def test_single_model_no_correlation_terms(self):
        """Test that single model has no correlation terms."""
        strategy = WeightedAverageStrategy(weights={"model1": 1.0})

        predictions = [
            ModelPrediction("model1", ModelType.STATE_SPACE, 250.0, 20.0, 0.9),
        ]

        result = strategy.combine(predictions)

        # With one model, uncertainty should equal model's uncertainty
        assert np.isclose(result.prediction_std, 20.0)

    def test_asymmetric_correlations_handled(self):
        """Test that correlations are looked up symmetrically."""
        weights = {"model1": 0.5, "model2": 0.5}

        # Provide correlation with sorted key (as per implementation)
        correlations = {("model1", "model2"): 0.7}  # Sorted alphabetically

        strategy = WeightedAverageStrategy(weights=weights, model_correlations=correlations)

        predictions = [
            ModelPrediction("model1", ModelType.STATE_SPACE, 250.0, 20.0, 0.9),
            ModelPrediction("model2", ModelType.BNN, 260.0, 20.0, 0.9),
        ]

        result = strategy.combine(predictions)

        # Should find correlation regardless of prediction order
        # σ² = 100 + 100 + 2 * 0.5 * 0.5 * 0.7 * 20 * 20 = 200 + 140 = 340
        expected_variance = 0.5**2 * 20**2 + 0.5**2 * 20**2 + 2 * 0.5 * 0.5 * 0.7 * 20 * 20
        expected_std = np.sqrt(expected_variance)
        assert np.isclose(result.prediction_std, expected_std, rtol=0.01)

    def test_zero_weights_handled(self):
        """Test that zero weights are handled correctly."""
        weights = {"model1": 0.5, "model2": 0.5, "model3": 0.0}  # model3 has 0 weight
        strategy = WeightedAverageStrategy(weights=weights)

        predictions = [
            ModelPrediction("model1", ModelType.STATE_SPACE, 250.0, 20.0, 0.9),
            ModelPrediction("model2", ModelType.BNN, 260.0, 20.0, 0.9),
            ModelPrediction(
                "model3", ModelType.XGBOOST, 1000.0, 100.0, 0.5
            ),  # Should not affect result
        ]

        result = strategy.combine(predictions)

        # model3 should not contribute to uncertainty
        # Only model1 and model2 matter (equal weights, so same as before)
        assert result.prediction_std < 25.0  # Should not be affected by model3's huge uncertainty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
