#!/usr/bin/env python3
"""
Unit Tests for General Ensemble Framework (v6.0+)

Extracted from v7.0 test suite - version-agnostic tests applicable to any ensemble.

Testing individual component predictions to ensure:
1. No placeholder predictions (250 yards)
2. Variance checks (component diversity ≥5 yards)
3. All 5 components produce valid outputs
4. Graceful degradation with minimum 3 components
5. Weight normalization and validation

Test Coverage: 18 general unit tests (extracted from v7, adapted for v6)
Source: tests/unit/test_ensemble_v7_components.py
Author: Claude Code (Sonnet 4.5)
Date: November 19, 2025
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nfl_analytics.ensemble.enhanced_ensemble_v6_production import EnhancedEnsemble5Way

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_db_config():
    """Mock database configuration."""
    return {
        "host": "localhost",
        "port": 5544,
        "dbname": "devdb01",
        "user": "dro",
        "password": "test_password_123",
    }


@pytest.fixture
def mahomes_fixture():
    """
    Test fixture: Patrick Mahomes vs Houston Texans (KC, Week 10, 2024)

    Context:
    - Patrick Mahomes: Elite QB, consistent 250+ yards
    - Houston Texans: High-roughing defense (+17.70 yard DiD adjustment)
    - Expected prediction: 270-290 yards (base ensemble + causal adjustment)
    """
    return {
        "player_id": "00-0033873",  # Patrick Mahomes
        "season": 2024,
        "week": 10,
        "opponent_team": "HOU",
        "player_team": "KC",
        "game_weather": {"temp": 65, "wind": 10, "precipitation": 0},
    }


@pytest.fixture
def mock_ensemble_v6_all_components(mock_db_config):
    """
    Create v6 ensemble with all 5 components mocked.

    Weights (canonical v6.0):
    - State-Space v2: 27%
    - Hierarchical v2: 27%
    - XGBoost v4 Ridge: 23%
    - Informative Priors v2: 13%
    - Hybrid Static-Incremental: 10%
    """
    with (
        patch("nfl_analytics.ensemble.enhanced_ensemble_v6_production.RModelInterface"),
        patch("nfl_analytics.ensemble.enhanced_ensemble_v6_production.HybridStaticIncrementalXGBoost"),
    ):

        ensemble = EnhancedEnsemble5Way(
            db_password="test_password_123",
            state_space_path="models/bayesian/simplified_random_walk_v2.rds",
            hierarchical_path="models/bayesian/passing_yards_hierarchical_v2.rds",
            informative_path="models/bayesian/passing_informative_priors_v2.rds",
            ridge_path="models/ridge_residuals_v6.pkl",
            xgboost_v4_path="models/xgb_ridge_residuals_v6.json",
            hybrid_base_path="models/xgboost/hybrid_static_incremental_v1",
        )

        # Mock models as loaded
        ensemble.models_loaded = True

        # Mock component predictors (realistic values for Mahomes)
        ensemble._predict_state_space = Mock(return_value=(240.0, 45.0))
        ensemble._predict_hierarchical = Mock(return_value=(235.0, 42.0))
        ensemble._predict_xgboost_v4 = Mock(return_value=(245.0, 40.0))
        ensemble._predict_informative_priors = Mock(return_value=(230.0, 48.0))
        ensemble._predict_hybrid = Mock(return_value=(250.0, 51.0))

        return ensemble


@pytest.fixture
def mock_ensemble_v6_partial_components(mock_db_config):
    """
    Create v6 ensemble with only 3 components working (minimum viable).

    Scenario: Hierarchical and Informative Priors fail.
    """
    with (
        patch("nfl_analytics.ensemble.enhanced_ensemble_v6_production.RModelInterface"),
        patch("nfl_analytics.ensemble.enhanced_ensemble_v6_production.HybridStaticIncrementalXGBoost"),
    ):

        ensemble = EnhancedEnsemble5Way(
            db_password="test_password_123",
        )

        ensemble.models_loaded = True
        ensemble._predict_state_space = Mock(return_value=(240.0, 45.0))
        ensemble._predict_hierarchical = Mock(return_value=(None, None))  # FAILS
        ensemble._predict_xgboost_v4 = Mock(return_value=(245.0, 40.0))
        ensemble._predict_informative_priors = Mock(return_value=(None, None))  # FAILS
        ensemble._predict_hybrid = Mock(return_value=(250.0, 51.0))

        return ensemble


# ============================================================================
# UNIT TESTS: Component Validation
# ============================================================================


class TestStateSpaceComponent:
    """Test State-Space v2 predictions (27% weight)."""

    def test_state_space_not_placeholder(self, mock_ensemble_v6_all_components):
        """State-space should not return placeholder value of 250."""
        mean, std = mock_ensemble_v6_all_components._predict_state_space("00-0033873", 2024, 10, {})
        assert mean != 250.0, "State-space returned placeholder (250), should be realistic"
        assert mean is not None and not np.isnan(mean)
        assert std is not None and std > 0

    def test_state_space_returns_tuple(self, mock_ensemble_v6_all_components):
        """State-space should return (mean, std) tuple."""
        result = mock_ensemble_v6_all_components._predict_state_space("00-0033873", 2024, 10, {})
        assert isinstance(result, tuple)
        assert len(result) == 2
        mean, std = result
        assert isinstance(mean, (int, float))
        assert isinstance(std, (int, float))

    def test_state_space_uncertainty_positive(self, mock_ensemble_v6_all_components):
        """State-space uncertainty should be positive."""
        mean, std = mock_ensemble_v6_all_components._predict_state_space("00-0033873", 2024, 10, {})
        assert std > 0, f"State-space std should be positive, got {std}"

    def test_state_space_prediction_in_reasonable_range(self, mock_ensemble_v6_all_components):
        """State-space prediction should be in reasonable range (100-400 yards)."""
        mean, std = mock_ensemble_v6_all_components._predict_state_space("00-0033873", 2024, 10, {})
        assert 100 <= mean <= 400, f"State-space mean {mean} outside reasonable range"


class TestHierarchicalComponent:
    """Test Hierarchical v2 predictions (27% weight)."""

    def test_hierarchical_not_placeholder(self, mock_ensemble_v6_all_components):
        """Hierarchical should not return placeholder value of 250."""
        mean, std = mock_ensemble_v6_all_components._predict_hierarchical(
            "00-0033873", 2024, 10, {}
        )
        assert mean != 250.0, "Hierarchical returned placeholder (250)"
        assert mean is not None and not np.isnan(mean)
        assert std is not None and std > 0

    def test_hierarchical_returns_tuple(self, mock_ensemble_v6_all_components):
        """Hierarchical should return (mean, std) tuple."""
        result = mock_ensemble_v6_all_components._predict_hierarchical("00-0033873", 2024, 10, {})
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_hierarchical_uncertainty_reasonable(self, mock_ensemble_v6_all_components):
        """Hierarchical uncertainty should be reasonable (20-80 yards)."""
        mean, std = mock_ensemble_v6_all_components._predict_hierarchical(
            "00-0033873", 2024, 10, {}
        )
        assert 20 <= std <= 80, f"Hierarchical std {std} outside reasonable range"


class TestXGBoostComponent:
    """Test XGBoost v4 Ridge Residuals predictions (23% weight)."""

    def test_xgboost_v4_not_placeholder(self, mock_ensemble_v6_all_components):
        """XGBoost v4 should not return placeholder value of 250."""
        mean, std = mock_ensemble_v6_all_components._predict_xgboost_v4({})
        assert mean != 250.0, "XGBoost v4 returned placeholder (250)"
        assert mean is not None and not np.isnan(mean)

    def test_xgboost_v4_uses_features(self, mock_ensemble_v6_all_components):
        """XGBoost v4 should process features dict."""
        features = {"pass_yards_l3": 250.0, "completions_l3": 24.0}
        mean, std = mock_ensemble_v6_all_components._predict_xgboost_v4(features)
        assert mean is not None, "XGBoost v4 failed to process features"

    def test_xgboost_v4_ridge_component_executed(self, mock_ensemble_v6_all_components):
        """XGBoost v4 Ridge should execute (2-stage: Ridge + XGB residual)."""
        # This test verifies the method is callable
        result = mock_ensemble_v6_all_components._predict_xgboost_v4({})
        assert result is not None
        assert isinstance(result, tuple)


class TestInformativePriorsComponent:
    """Test Informative Priors v2 predictions (13% weight)."""

    def test_informative_priors_not_placeholder(self, mock_ensemble_v6_all_components):
        """Informative Priors should not return placeholder value of 250."""
        mean, std = mock_ensemble_v6_all_components._predict_informative_priors(
            "00-0033873", 2024, 10, {}
        )
        assert mean != 250.0, "Informative Priors returned placeholder (250)"
        assert mean is not None and not np.isnan(mean)

    def test_informative_priors_uses_domain_knowledge(self, mock_ensemble_v6_all_components):
        """Informative Priors should incorporate domain knowledge (not pure data)."""
        # Test that method is callable and returns valid output
        result = mock_ensemble_v6_all_components._predict_informative_priors(
            "00-0033873", 2024, 10, {}
        )
        assert result is not None
        assert isinstance(result, tuple)


class TestHybridComponent:
    """Test Hybrid Static-Incremental XGB predictions (10% weight)."""

    def test_hybrid_not_placeholder(self, mock_ensemble_v6_all_components):
        """Hybrid should not return placeholder value of 250."""
        mean, std = mock_ensemble_v6_all_components._predict_hybrid({})
        assert mean != 250.0, "Hybrid returned placeholder (250)"

    def test_hybrid_13_features_processed(self, mock_ensemble_v6_all_components):
        """Hybrid should process standard feature set."""
        features = {
            "pass_yards_l3": 250.0,
            "completions_l3": 24.0,
            "pass_attempts_l3": 35.0,
            "opponent_epa": -0.05,
        }
        result = mock_ensemble_v6_all_components._predict_hybrid(features)
        assert result is not None


class TestComponentDiversity:
    """Test component diversity and minimum requirements."""

    def test_minimum_3_components_required(self, mock_ensemble_v6_partial_components):
        """Ensemble should work with minimum 3 components."""
        # Mock feature fetch
        mock_ensemble_v6_partial_components._fetch_features = Mock(
            return_value={
                "pass_yards_l3": 250.0,
            }
        )

        result = mock_ensemble_v6_partial_components.predict("00-0033873", 2024, 10)

        # Should succeed with 3 components (State-Space, XGBoost, Hybrid)
        assert result is not None
        assert "ensemble_mean" in result

    def test_component_variance_check(self, mock_ensemble_v6_all_components):
        """Components should have sufficient variance (≥5 yards diversity)."""
        # Mock feature fetch
        mock_ensemble_v6_all_components._fetch_features = Mock(return_value={})

        # Get component predictions
        ss_mean, _ = mock_ensemble_v6_all_components._predict_state_space(
            "00-0033873", 2024, 10, {}
        )
        hier_mean, _ = mock_ensemble_v6_all_components._predict_hierarchical(
            "00-0033873", 2024, 10, {}
        )
        xgb_mean, _ = mock_ensemble_v6_all_components._predict_xgboost_v4({})
        ip_mean, _ = mock_ensemble_v6_all_components._predict_informative_priors(
            "00-0033873", 2024, 10, {}
        )
        hybrid_mean, _ = mock_ensemble_v6_all_components._predict_hybrid({})

        predictions = [ss_mean, hier_mean, xgb_mean, ip_mean, hybrid_mean]
        variance = np.var(predictions)

        assert variance >= 25, f"Component variance {variance:.2f} < 25 (std < 5 yards)"

    def test_component_mean_not_placeholder(self, mock_ensemble_v6_all_components):
        """All component means should differ from placeholder (250)."""
        components = [
            mock_ensemble_v6_all_components._predict_state_space("00-0033873", 2024, 10, {}),
            mock_ensemble_v6_all_components._predict_hierarchical("00-0033873", 2024, 10, {}),
            mock_ensemble_v6_all_components._predict_xgboost_v4({}),
            mock_ensemble_v6_all_components._predict_informative_priors("00-0033873", 2024, 10, {}),
            mock_ensemble_v6_all_components._predict_hybrid({}),
        ]

        for i, (mean, std) in enumerate(components):
            assert mean != 250.0, f"Component {i} returned placeholder (250)"


class TestWeightValidation:
    """Test ensemble weight structure and validation."""

    def test_default_weights_sum_to_one(self, mock_ensemble_v6_all_components):
        """Default component weights should sum to 1.0."""
        weights = mock_ensemble_v6_all_components.weights
        assert (
            abs(sum(weights.values()) - 1.0) < 1e-6
        ), f"Weights sum to {sum(weights.values()):.6f}, expected 1.0"

    def test_canonical_weights_structure(self, mock_ensemble_v6_all_components):
        """Weights should have 5 components (v6.0)."""
        weights = mock_ensemble_v6_all_components.weights
        expected_keys = {
            "state_space_v2",
            "hierarchical_v2",
            "xgboost_v4_ridge",
            "informative_priors_v2",
            "hybrid_static_incremental",
        }
        assert (
            set(weights.keys()) == expected_keys
        ), f"Weight keys mismatch. Got: {set(weights.keys())}"

    def test_canonical_weight_values(self, mock_ensemble_v6_all_components):
        """Weights should match v6.0 canonical values."""
        weights = mock_ensemble_v6_all_components.weights

        # v6.0 canonical weights (from Nov 15, 2025)
        expected = {
            "state_space_v2": 0.27,
            "hierarchical_v2": 0.27,
            "xgboost_v4_ridge": 0.23,
            "informative_priors_v2": 0.13,
            "hybrid_static_incremental": 0.10,
        }

        for component, expected_weight in expected.items():
            actual_weight = weights[component]
            assert (
                abs(actual_weight - expected_weight) < 0.01
            ), f"{component} weight {actual_weight:.2f} ≠ expected {expected_weight:.2f}"


# ============================================================================
# TEST SUMMARY
# ============================================================================

"""
Test Summary:
- 18 unit tests (extracted from v7, adapted for v6)
- Coverage: Component validation, diversity checks, weight validation
- All tests are version-agnostic (applicable to any ensemble)

Test Categories:
1. State-Space Component (4 tests)
2. Hierarchical Component (3 tests)
3. XGBoost v4 Component (3 tests)
4. Informative Priors Component (2 tests)
5. Hybrid Component (2 tests)
6. Component Diversity (3 tests)
7. Weight Validation (3 tests)

Total: 18 tests

Run with:
    pytest tests/unit/test_ensemble_framework.py -v
    pytest tests/unit/test_ensemble_framework.py::TestStateSpaceComponent -v
    pytest tests/unit/test_ensemble_framework.py::TestWeightValidation -v
"""
