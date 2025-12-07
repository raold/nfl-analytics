#!/usr/bin/env python3
"""
Unit Tests for Bayesian Stacking
=================================

Tests for py/ensemble/bayesian_stacking.py:
- ModelLoader: Loading ensemble models
- BayesianStacker: LOO-CV, weight optimization, prediction
- ValidationFramework: Data splitting, statistical tests

Author: Claude Code (Sonnet 4.5)
Date: November 9, 2025
Version: 1.0
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from py.ensemble.bayesian_stacking import (
    BayesianStacker,
    LOOResult,
    ModelLoader,
    StackingResult,
    ValidationFramework,
)

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def synthetic_validation_data():
    """Generate synthetic validation data for testing."""
    np.random.seed(42)
    n_games = 100

    actual_yards = np.random.gamma(shape=5.0, scale=50.0, size=n_games)

    df = pd.DataFrame({
        "player_id": [f"player_{i%10}" for i in range(n_games)],
        "season": np.random.choice([2022, 2023], size=n_games),
        "week": np.random.randint(1, 18, size=n_games),
        "actual_yards": actual_yards,
        "state_space_v2_pred_mean": actual_yards + np.random.normal(0, 45, size=n_games),
        "state_space_v2_pred_std": np.random.uniform(35, 55, size=n_games),
        "hierarchical_v2_pred_mean": actual_yards + np.random.normal(0, 50, size=n_games),
        "hierarchical_v2_pred_std": np.random.uniform(30, 50, size=n_games),
        "informative_priors_v2_pred_mean": actual_yards + np.random.normal(0, 52, size=n_games),
        "informative_priors_v2_pred_std": np.random.uniform(32, 52, size=n_games),
        "xgboost_v2_pred_mean": actual_yards + np.random.normal(0, 56, size=n_games),
        "xgboost_v2_pred_std": np.random.uniform(60, 80, size=n_games),
    })

    return df


@pytest.fixture
def mock_loo_results():
    """Generate mock LOO results."""
    return {
        "state_space_v2": LOOResult(
            model_name="state_space_v2",
            elpd_loo=-1420.5,
            p_loo=45.3,
            se_elpd=12.8,
            pareto_k_max=0.62,
            pareto_k_good_pct=98.5,
            log_likelihoods=np.random.normal(-2.8, 0.5, size=100)
        ),
        "hierarchical_v2": LOOResult(
            model_name="hierarchical_v2",
            elpd_loo=-1435.2,
            p_loo=52.1,
            se_elpd=13.5,
            pareto_k_max=0.58,
            pareto_k_good_pct=99.1,
            log_likelihoods=np.random.normal(-2.9, 0.6, size=100)
        ),
        "informative_priors_v2": LOOResult(
            model_name="informative_priors_v2",
            elpd_loo=-1442.8,
            p_loo=48.7,
            se_elpd=14.2,
            pareto_k_max=0.65,
            pareto_k_good_pct=97.8,
            log_likelihoods=np.random.normal(-3.0, 0.7, size=100)
        ),
        "xgboost_v2": LOOResult(
            model_name="xgboost_v2",
            elpd_loo=-1458.3,
            p_loo=38.2,
            se_elpd=15.1,
            pareto_k_max=0.69,
            pareto_k_good_pct=96.2,
            log_likelihoods=np.random.normal(-3.1, 0.8, size=100)
        ),
    }


# ============================================================================
# ModelLoader Tests
# ============================================================================

class TestModelLoader:
    """Test suite for ModelLoader class."""

    def test_initialization(self):
        """Test ModelLoader initialization."""
        loader = ModelLoader()
        assert loader.models == {}
        assert loader.db_config["host"] == "localhost"
        assert loader.db_config["port"] == 5544

    def test_load_xgboost_missing_file(self):
        """Test loading XGBoost when file is missing."""
        loader = ModelLoader()
        with patch("pathlib.Path.exists", return_value=False):
            model = loader.load_xgboost()
            assert model is None

    def test_load_state_space_missing_file(self):
        """Test loading state-space when skills CSV is missing."""
        loader = ModelLoader()
        with patch("pathlib.Path.exists", return_value=False):
            skills = loader.load_state_space()
            assert skills is None


# ============================================================================
# BayesianStacker Tests
# ============================================================================

class TestBayesianStacker:
    """Test suite for BayesianStacker class."""

    def test_initialization(self):
        """Test BayesianStacker initialization."""
        stacker = BayesianStacker()
        assert stacker.models == {}
        assert len(stacker.heuristic_weights) == 4
        assert np.isclose(sum(stacker.heuristic_weights.values()), 1.0)

    def test_compute_log_likelihood(self, synthetic_validation_data):
        """Test log-likelihood computation."""
        stacker = BayesianStacker()

        y_true = synthetic_validation_data["actual_yards"].values[:10]
        y_pred_mean = synthetic_validation_data["state_space_v2_pred_mean"].values[:10]
        y_pred_std = synthetic_validation_data["state_space_v2_pred_std"].values[:10]

        log_lik = stacker._compute_log_likelihood(
            "state_space_v2", y_true, y_pred_mean, y_pred_std
        )

        assert len(log_lik) == 10
        assert np.all(np.isfinite(log_lik))
        assert np.all(log_lik < 0)  # Log probabilities are negative

    def test_compute_loo_cv(self, synthetic_validation_data):
        """Test LOO-CV computation."""
        stacker = BayesianStacker()
        stacker.models = {"state_space_v2": Mock(), "hierarchical_v2": Mock()}

        loo_results = stacker.compute_loo_cv(synthetic_validation_data, save_results=False)

        assert len(loo_results) == 2
        assert "state_space_v2" in loo_results
        assert "hierarchical_v2" in loo_results

        # Check LOOResult structure
        for result in loo_results.values():
            assert isinstance(result, LOOResult)
            assert np.isfinite(result.elpd_loo)
            assert result.pareto_k_max < 1.0
            assert 0 <= result.pareto_k_good_pct <= 100

    def test_optimize_weights(self, mock_loo_results):
        """Test weight optimization."""
        stacker = BayesianStacker()

        result = stacker.optimize_weights(mock_loo_results, method="stacking")

        assert isinstance(result, StackingResult)
        assert result.status == "SUCCESS"
        assert len(result.weights) == 4

        # Weights should sum to 1.0
        assert np.isclose(sum(result.weights.values()), 1.0, atol=0.01)

        # Weights should be non-negative
        assert all(w >= 0 for w in result.weights.values())

        # ELPD with stacking should be better than equal weights
        assert result.improvement > 0

    def test_optimize_weights_empty_loo(self):
        """Test weight optimization with empty LOO results."""
        stacker = BayesianStacker()

        # This should fail gracefully
        result = stacker.optimize_weights({}, method="stacking")
        assert result.status == "FAILED"

    def test_predict_stacked(self):
        """Test stacked prediction generation."""
        stacker = BayesianStacker()

        predictions = {
            "state_space_v2": (265.3, 45.2),
            "hierarchical_v2": (272.1, 38.6),
            "informative_priors_v2": (268.5, 41.3),
            "xgboost_v2": (258.7, 52.1)
        }

        weights = {
            "state_space_v2": 0.35,
            "hierarchical_v2": 0.28,
            "informative_priors_v2": 0.14,
            "xgboost_v2": 0.23
        }

        ensemble_mean, ensemble_std = stacker.predict_stacked(predictions, weights)

        assert np.isfinite(ensemble_mean)
        assert np.isfinite(ensemble_std)
        assert ensemble_std > 0

        # Ensemble mean should be between min and max individual predictions
        means = [p[0] for p in predictions.values()]
        assert min(means) <= ensemble_mean <= max(means)

    def test_predict_stacked_missing_models(self):
        """Test stacked prediction with missing models."""
        stacker = BayesianStacker()

        predictions = {
            "state_space_v2": (265.3, 45.2),
            # hierarchical_v2 missing
            "informative_priors_v2": (268.5, 41.3),
            "xgboost_v2": (258.7, 52.1)
        }

        weights = {
            "state_space_v2": 0.35,
            "hierarchical_v2": 0.28,
            "informative_priors_v2": 0.14,
            "xgboost_v2": 0.23
        }

        ensemble_mean, ensemble_std = stacker.predict_stacked(predictions, weights)

        # Should still work, normalizing weights for available models
        assert np.isfinite(ensemble_mean)
        assert np.isfinite(ensemble_std)

    def test_predict_stacked_no_predictions(self):
        """Test stacked prediction with no predictions."""
        stacker = BayesianStacker()

        predictions = {}
        weights = {
            "state_space_v2": 0.35,
            "hierarchical_v2": 0.28,
            "informative_priors_v2": 0.14,
            "xgboost_v2": 0.23
        }

        ensemble_mean, ensemble_std = stacker.predict_stacked(predictions, weights)

        # Should return NaN
        assert np.isnan(ensemble_mean)
        assert np.isnan(ensemble_std)

    def test_generate_report(self, mock_loo_results):
        """Test report generation."""
        stacker = BayesianStacker()

        stacking_result = StackingResult(
            weights={
                "state_space_v2": 0.35,
                "hierarchical_v2": 0.28,
                "informative_priors_v2": 0.14,
                "xgboost_v2": 0.23
            },
            elpd_loo_stacking=-2847.32,
            elpd_loo_equal=-2859.14,
            improvement=11.82,
            status="SUCCESS",
            optimization_time_sec=2.34
        )

        report = stacker.generate_report(
            mock_loo_results,
            stacking_result,
            output_path=None
        )

        assert isinstance(report, str)
        assert "Bayesian Stacking" in report
        assert "ELPD_loo" in report
        assert "Optimal Weights" in report
        assert "state_space_v2" in report


# ============================================================================
# ValidationFramework Tests
# ============================================================================

class TestValidationFramework:
    """Test suite for ValidationFramework class."""

    def test_initialization(self):
        """Test ValidationFramework initialization."""
        framework = ValidationFramework()
        assert framework.db_config["host"] == "localhost"
        assert framework.db_config["port"] == 5544

    def test_statistical_tests(self):
        """Test Diebold-Mariano statistical test."""
        framework = ValidationFramework()

        np.random.seed(42)
        n = 100

        # Method 1: Worse performance (higher errors)
        errors_heuristic = np.random.normal(0, 50, size=n)

        # Method 2: Better performance (lower errors)
        errors_stacking = np.random.normal(0, 45, size=n)

        test_results = framework.statistical_tests(errors_heuristic, errors_stacking)

        assert "mean_loss_differential" in test_results
        assert "t_statistic" in test_results
        assert "p_value" in test_results
        assert "significant_at_05" in test_results

        # Mean loss differential should be positive (stacking better)
        assert test_results["mean_loss_differential"] > 0

    def test_statistical_tests_identical_errors(self):
        """Test statistical test with identical errors."""
        framework = ValidationFramework()

        np.random.seed(42)
        errors = np.random.normal(0, 50, size=100)

        test_results = framework.statistical_tests(errors, errors)

        # p-value should be 1.0 (no difference)
        assert test_results["p_value"] > 0.5
        assert not test_results["significant_at_05"]


# ============================================================================
# Integration Tests
# ============================================================================

class TestBayesianStackingIntegration:
    """Integration tests for full stacking workflow."""

    def test_full_workflow(self, synthetic_validation_data):
        """Test complete stacking workflow end-to-end."""
        # 1. Initialize stacker
        stacker = BayesianStacker()
        stacker.models = {"state_space_v2": Mock(), "hierarchical_v2": Mock()}

        # 2. Compute LOO-CV
        loo_results = stacker.compute_loo_cv(synthetic_validation_data, save_results=False)
        assert len(loo_results) == 2

        # 3. Optimize weights
        stacking_result = stacker.optimize_weights(loo_results)
        assert stacking_result.status == "SUCCESS"

        # 4. Generate stacked prediction
        predictions = {
            "state_space_v2": (265.3, 45.2),
            "hierarchical_v2": (272.1, 38.6),
        }
        ensemble_mean, ensemble_std = stacker.predict_stacked(
            predictions, stacking_result.weights
        )
        assert np.isfinite(ensemble_mean)

        # 5. Generate report
        report = stacker.generate_report(loo_results, stacking_result)
        assert isinstance(report, str)
        assert len(report) > 100


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_negative_std_predictions(self):
        """Test handling of invalid (negative) std predictions."""
        stacker = BayesianStacker()

        # This should handle gracefully (clip to small positive value)
        predictions = {
            "state_space_v2": (265.3, -10.0),  # Invalid negative std
        }
        weights = {"state_space_v2": 1.0}

        # Should not crash
        ensemble_mean, ensemble_std = stacker.predict_stacked(predictions, weights)
        # Result may be invalid, but should not crash

    def test_zero_weight_model(self):
        """Test stacking with zero-weight model."""
        stacker = BayesianStacker()

        predictions = {
            "state_space_v2": (265.3, 45.2),
            "hierarchical_v2": (272.1, 38.6),
        }

        weights = {
            "state_space_v2": 1.0,
            "hierarchical_v2": 0.0,  # Zero weight
        }

        ensemble_mean, ensemble_std = stacker.predict_stacked(predictions, weights)

        # Should be identical to state_space_v2 prediction
        assert np.isclose(ensemble_mean, 265.3, atol=0.1)

    def test_very_small_validation_set(self):
        """Test with very small validation set (edge case)."""
        stacker = BayesianStacker()
        stacker.models = {"state_space_v2": Mock()}

        # Only 3 games
        small_df = pd.DataFrame({
            "player_id": ["p1", "p2", "p3"],
            "season": [2023, 2023, 2023],
            "week": [1, 2, 3],
            "actual_yards": [250, 280, 260],
            "state_space_v2_pred_mean": [255, 275, 265],
            "state_space_v2_pred_std": [45, 50, 48],
        })

        # Should still work (though not statistically meaningful)
        loo_results = stacker.compute_loo_cv(small_df, save_results=False)
        assert len(loo_results) == 1


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.slow
class TestPerformance:
    """Performance tests (marked as slow)."""

    def test_large_validation_set(self):
        """Test performance with large validation set."""
        np.random.seed(42)
        n_games = 5000  # Large validation set

        actual_yards = np.random.gamma(shape=5.0, scale=50.0, size=n_games)

        large_df = pd.DataFrame({
            "player_id": [f"player_{i%100}" for i in range(n_games)],
            "season": np.random.choice([2020, 2021, 2022, 2023], size=n_games),
            "week": np.random.randint(1, 18, size=n_games),
            "actual_yards": actual_yards,
            "state_space_v2_pred_mean": actual_yards + np.random.normal(0, 45, size=n_games),
            "state_space_v2_pred_std": np.random.uniform(35, 55, size=n_games),
        })

        stacker = BayesianStacker()
        stacker.models = {"state_space_v2": Mock()}

        import time
        start = time.time()
        loo_results = stacker.compute_loo_cv(large_df, save_results=False)
        elapsed = time.time() - start

        # Should complete in reasonable time (<5 seconds)
        assert elapsed < 5.0
        assert len(loo_results) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
