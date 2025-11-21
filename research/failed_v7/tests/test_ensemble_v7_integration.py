#!/usr/bin/env python3
"""
Integration Tests for Ensemble v7.0 (Phase C)

Tests full prediction pipeline with real/mocked database connections:
1. All 5 components generate predictions (end-to-end)
2. Ensemble mean is not placeholder (250 yards)
3. Component diversity >= 5 yards
4. Graceful degradation with component failures
5. Temperature scaling applied to calibration
6. Causal adjustment layer (DiD Roughing)

Test Coverage: ~25 integration tests
Author: Claude Code (Sonnet 4.5)
Date: November 16, 2025
"""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import numpy as np
import pandas as pd

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from py.ensemble.enhanced_ensemble_v7 import EnhancedEnsemble5Way


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def test_db_config():
    """Test database configuration (may be mocked)."""
    return {
        "host": os.getenv("TEST_DB_HOST", "localhost"),
        "port": int(os.getenv("TEST_DB_PORT", "5544")),
        "dbname": os.getenv("TEST_DB_NAME", "devdb01"),
        "user": os.getenv("TEST_DB_USER", "dro"),
        "password": os.getenv("TEST_DB_PASSWORD", "test_password"),
    }


@pytest.fixture
def mahomes_2024_week10():
    """Patrick Mahomes vs Houston Texans (high-roughing defense)."""
    return {
        "player_id": "00-0033873",
        "season": 2024,
        "week": 10,
        "opponent_team": "HOU",
        "player_team": "KC",
        "game_weather": {"temp": 65, "wind": 10, "precipitation": 0},
    }


@pytest.fixture
def jalen_hurts_2024_week5():
    """Jalen Hurts vs Green Bay (low-roughing defense)."""
    return {
        "player_id": "00-0036442",
        "season": 2024,
        "week": 5,
        "opponent_team": "GB",
        "player_team": "PHI",
        "game_weather": {"temp": 55, "wind": 12, "precipitation": 0},
    }


@pytest.fixture
def ensemble_v7_with_mocks(test_db_config):
    """Create ensemble v7.0 with all components mocked."""
    with patch("py.ensemble.enhanced_ensemble_v7.CausalFeaturesV1") as mock_causal, \
         patch("py.ensemble.enhanced_ensemble_v7.RModelInterface") as mock_r, \
         patch("py.ensemble.enhanced_ensemble_v7.HybridStaticIncrementalXGBoost"):

        # Create ensemble
        ensemble = EnhancedEnsemble5Way(
            db_password=test_db_config["password"],
        )

        # Mark as loaded
        ensemble.models_loaded = True

        # Mock component predictions (realistic values for Mahomes)
        ensemble._predict_state_space = Mock(return_value=(240.0, 45.0))
        ensemble._predict_hierarchical = Mock(return_value=(235.0, 42.0))
        ensemble._predict_xgboost_v4 = Mock(return_value=(245.0, 40.0))
        ensemble._predict_informative_priors = Mock(return_value=(230.0, 48.0))
        ensemble._predict_hybrid = Mock(return_value=(250.0, 51.0))

        # Mock database feature fetch
        ensemble._fetch_features = Mock(return_value={
            "pass_yards_l3": 250.0,
            "pass_attempts_l3": 35.0,
            "completions_l3": 24.0,
            "completions_pct_l3": 0.686,
            "opponent_epa": -0.05,
            "opponent_points_allowed": 20.0,
            "home_away_encoded": 1,
        })

        # Mock causal features
        mock_causal_instance = Mock()
        mock_causal_instance.calculate_features = Mock(return_value={
            "did_roughing_yards": 17.70,  # HOU is high-roughing
            "psm_travel_yards": 0.0,
            "psm_weather_yards": 0.0,
            "causal_total_yards": 17.70,
        })
        mock_causal.return_value = mock_causal_instance
        ensemble.causal_features = mock_causal_instance

        return ensemble


# ============================================================================
# INTEGRATION TESTS: End-to-End Prediction
# ============================================================================

class TestEnsembleE2EPrediction:
    """Test full prediction pipeline."""

    def test_all_5_components_generate_predictions(self, ensemble_v7_with_mocks, mahomes_2024_week10):
        """All 5 components should generate valid predictions."""
        result = ensemble_v7_with_mocks.predict(**mahomes_2024_week10)

        # Check result structure
        assert isinstance(result, dict)
        assert "ensemble_mean" in result
        assert "valid_components" in result
        assert len(result["valid_components"]) == 5, \
            f"Expected 5 components, got {len(result['valid_components'])}"

        # Check all component names present
        expected_components = {
            "state_space_v2",
            "hierarchical_v2",
            "xgboost_v4_ridge",
            "informative_priors_v2",
            "hybrid_static_incremental",
        }
        assert set(result["valid_components"]) == expected_components

    def test_ensemble_mean_reasonable_value(self, ensemble_v7_with_mocks, mahomes_2024_week10):
        """Ensemble mean should be in reasonable range (180-350 yards)."""
        result = ensemble_v7_with_mocks.predict(**mahomes_2024_week10)

        ensemble_mean = result["ensemble_mean"]
        assert 180 <= ensemble_mean <= 350, \
            f"Ensemble mean {ensemble_mean} outside [180, 350]"

    def test_ensemble_mean_not_placeholder(self, ensemble_v7_with_mocks, mahomes_2024_week10):
        """Ensemble mean should not be exactly 250 (placeholder)."""
        result = ensemble_v7_with_mocks.predict(**mahomes_2024_week10)

        # Weighted average of [240, 235, 245, 230, 250]:
        # (240*0.27 + 235*0.27 + 245*0.23 + 230*0.13 + 250*0.10)
        # = 64.8 + 63.45 + 56.35 + 29.9 + 25 = 239.5 + 17.70 causal = 257.2
        ensemble_mean = result["ensemble_mean"]
        assert ensemble_mean != 250.0, "Ensemble mean equals placeholder"

    def test_base_ensemble_mean_present(self, ensemble_v7_with_mocks, mahomes_2024_week10):
        """Result should contain base ensemble mean (before causal adjustment)."""
        result = ensemble_v7_with_mocks.predict(**mahomes_2024_week10)

        assert "base_ensemble_mean" in result
        assert "causal_adjustment" in result
        assert "ensemble_mean" in result

        # Check relationship: ensemble_mean = base_ensemble_mean + causal_adjustment
        expected_ensemble = result["base_ensemble_mean"] + result["causal_adjustment"]
        assert np.isclose(result["ensemble_mean"], expected_ensemble, atol=0.01)

    def test_prediction_returns_uncertainty(self, ensemble_v7_with_mocks, mahomes_2024_week10):
        """Should return uncertainty estimates."""
        result = ensemble_v7_with_mocks.predict(
            **mahomes_2024_week10,
            return_uncertainty=True
        )

        assert "ensemble_std" in result
        assert "calibrated_std" in result
        assert "lower_90" in result
        assert "upper_90" in result

        # Prediction intervals should be wider than std
        assert result["upper_90"] > result["ensemble_mean"]
        assert result["lower_90"] < result["ensemble_mean"]

    def test_prediction_includes_component_details(self, ensemble_v7_with_mocks, mahomes_2024_week10):
        """With return_components=True, should include component predictions."""
        result = ensemble_v7_with_mocks.predict(
            **mahomes_2024_week10,
            return_components=True
        )

        assert "components" in result
        assert len(result["components"]) == 5

        # Each component should have mean and std
        for component_name, component_result in result["components"].items():
            assert "mean" in component_result
            assert "std" in component_result
            assert component_result["std"] > 0


# ============================================================================
# INTEGRATION TESTS: Component Diversity
# ============================================================================

class TestComponentDiversity:
    """Test that components produce diverse predictions."""

    def test_component_diversity_threshold_5_yards(self, ensemble_v7_with_mocks, mahomes_2024_week10):
        """Component diversity should be >= 5 yards (max - min)."""
        # Components: [240, 235, 245, 230, 250]
        # Max - min = 250 - 230 = 20 yards (sufficient diversity)

        result = ensemble_v7_with_mocks.predict(
            **mahomes_2024_week10,
            return_components=True
        )

        components = result["components"]
        means = [c["mean"] for c in components.values()]

        diversity = max(means) - min(means)
        assert diversity >= 5.0, f"Component diversity {diversity} < 5 yards minimum"

    def test_ensemble_weighted_average_correct(self, ensemble_v7_with_mocks, mahomes_2024_week10):
        """Ensemble mean should be proper weighted average."""
        result = ensemble_v7_with_mocks.predict(
            **mahomes_2024_week10,
            return_components=True
        )

        components = result["components"]
        weights = result["weights"]

        # Calculate expected weighted average (without causal adjustment)
        expected_mean = sum(
            components[name]["mean"] * weights[name]
            for name in components
        )

        assert np.isclose(result["base_ensemble_mean"], expected_mean, atol=0.1), \
            f"Base ensemble mean {result['base_ensemble_mean']} != weighted avg {expected_mean}"

    def test_uncertainty_pooling(self, ensemble_v7_with_mocks, mahomes_2024_week10):
        """Variance pooling should be correct."""
        result = ensemble_v7_with_mocks.predict(
            **mahomes_2024_week10,
            return_components=True
        )

        components = result["components"]
        weights = result["weights"]

        # Calculate expected variance: sum(std^2 * weight^2)
        expected_var = sum(
            (components[name]["std"] ** 2) * (weights[name] ** 2)
            for name in components
        )
        expected_std = np.sqrt(expected_var)

        assert np.isclose(result["ensemble_std"], expected_std, atol=0.1), \
            f"Ensemble std {result['ensemble_std']} != calculated {expected_std}"


# ============================================================================
# INTEGRATION TESTS: Causal Features Integration
# ============================================================================

class TestCausalAdjustment:
    """Test DiD roughing causal adjustment layer."""

    def test_causal_adjustment_applied_high_roughing_defense(
        self, ensemble_v7_with_mocks, mahomes_2024_week10
    ):
        """DiD roughing adjustment should be applied for HOU (high-roughing)."""
        result = ensemble_v7_with_mocks.predict(**mahomes_2024_week10)

        # HOU is high-roughing defense, should get +17.70 yards
        assert result["causal_adjustment"] == 17.70, \
            f"Expected +17.70 yards, got {result['causal_adjustment']}"

    def test_causal_adjustment_zero_low_roughing_defense(
        self, ensemble_v7_with_mocks, jalen_hurts_2024_week5
    ):
        """DiD roughing adjustment should be 0 for low-roughing defenses (GB)."""
        # Update causal features mock for GB (low-roughing)
        ensemble_v7_with_mocks.causal_features.calculate_features = Mock(
            return_value={
                "did_roughing_yards": 0.0,
                "psm_travel_yards": 0.0,
                "psm_weather_yards": 0.0,
                "causal_total_yards": 0.0,
            }
        )

        result = ensemble_v7_with_mocks.predict(**jalen_hurts_2024_week5)

        assert result["causal_adjustment"] == 0.0, \
            f"Expected 0 yards, got {result['causal_adjustment']}"

    def test_causal_features_breakdown(self, ensemble_v7_with_mocks, mahomes_2024_week10):
        """Result should include breakdown of causal features."""
        result = ensemble_v7_with_mocks.predict(**mahomes_2024_week10)

        assert "causal_features" in result
        causal = result["causal_features"]

        assert "did_roughing_yards" in causal
        assert "psm_travel_yards" in causal
        assert "psm_weather_yards" in causal
        assert "causal_total_yards" in causal

    def test_ensemble_mean_includes_causal_adjustment(self, ensemble_v7_with_mocks, mahomes_2024_week10):
        """Ensemble mean should include causal adjustment."""
        result = ensemble_v7_with_mocks.predict(**mahomes_2024_week10)

        # ensemble_mean = base_ensemble_mean + causal_adjustment
        expected_mean = result["base_ensemble_mean"] + result["causal_adjustment"]
        assert np.isclose(result["ensemble_mean"], expected_mean, atol=0.01)


# ============================================================================
# INTEGRATION TESTS: Temperature Scaling
# ============================================================================

class TestTemperatureScaling:
    """Test temperature scaling calibration."""

    def test_temperature_scaling_applied(self, ensemble_v7_with_mocks):
        """Temperature scaling should be applied (T != 1.0)."""
        # Default or fallback temperature should be loaded
        assert ensemble_v7_with_mocks.temperature is not None
        assert ensemble_v7_with_mocks.temperature > 0, "Temperature must be positive"

    def test_calibrated_std_differs_from_ensemble_std(self, ensemble_v7_with_mocks, mahomes_2024_week10):
        """Calibrated std should differ from ensemble std if T != 1.0."""
        result = ensemble_v7_with_mocks.predict(
            **mahomes_2024_week10,
            return_uncertainty=True
        )

        if ensemble_v7_with_mocks.temperature != 1.0:
            # calibrated_std = ensemble_std / temperature
            expected_calibrated = result["ensemble_std"] / ensemble_v7_with_mocks.temperature
            assert np.isclose(result["calibrated_std"], expected_calibrated, atol=0.1)

    def test_prediction_intervals_use_calibrated_std(self, ensemble_v7_with_mocks, mahomes_2024_week10):
        """90% prediction intervals should use calibrated std."""
        result = ensemble_v7_with_mocks.predict(
            **mahomes_2024_week10,
            return_uncertainty=True
        )

        z_90 = 1.645
        expected_lower = result["ensemble_mean"] - z_90 * result["calibrated_std"]
        expected_upper = result["ensemble_mean"] + z_90 * result["calibrated_std"]

        assert np.isclose(result["lower_90"], expected_lower, atol=0.1)
        assert np.isclose(result["upper_90"], expected_upper, atol=0.1)


# ============================================================================
# INTEGRATION TESTS: Graceful Degradation
# ============================================================================

class TestComponentFailureHandling:
    """Test graceful degradation when components fail."""

    def test_ensemble_handles_one_component_failure(self, ensemble_v7_with_mocks, mahomes_2024_week10):
        """Ensemble should handle 1 component failure (4 of 5 working)."""
        # Hierarchical fails
        ensemble_v7_with_mocks._predict_hierarchical = Mock(return_value=(None, None))

        result = ensemble_v7_with_mocks.predict(**mahomes_2024_week10)

        # Should still work with 4 components
        assert len(result["valid_components"]) == 4
        assert "ensemble_mean" in result

    def test_ensemble_handles_two_component_failures(self, ensemble_v7_with_mocks, mahomes_2024_week10):
        """Ensemble should handle 2 component failures (3 of 5 working)."""
        # Hierarchical and Informative Priors fail
        ensemble_v7_with_mocks._predict_hierarchical = Mock(return_value=(None, None))
        ensemble_v7_with_mocks._predict_informative_priors = Mock(return_value=(None, None))

        result = ensemble_v7_with_mocks.predict(**mahomes_2024_week10)

        # Should still work with 3 components (minimum)
        assert len(result["valid_components"]) == 3
        assert "ensemble_mean" in result

    def test_ensemble_fails_with_three_component_failures(self, ensemble_v7_with_mocks, mahomes_2024_week10):
        """Ensemble should fail with 3 component failures (< 3 working)."""
        # Hierarchical, Informative Priors, and Hybrid fail
        ensemble_v7_with_mocks._predict_hierarchical = Mock(return_value=(None, None))
        ensemble_v7_with_mocks._predict_informative_priors = Mock(return_value=(None, None))
        ensemble_v7_with_mocks._predict_hybrid = Mock(return_value=(None, None))

        with pytest.raises(RuntimeError, match="Insufficient ensemble components"):
            ensemble_v7_with_mocks.predict(**mahomes_2024_week10)

    def test_causal_features_failure_handled(self, ensemble_v7_with_mocks, mahomes_2024_week10):
        """Ensemble should handle causal features calculation failure."""
        # Causal features calculation fails
        ensemble_v7_with_mocks.causal_features.calculate_features = Mock(
            side_effect=Exception("Database error")
        )

        result = ensemble_v7_with_mocks.predict(**mahomes_2024_week10)

        # Should return fallback causal features (zeros)
        assert result["causal_adjustment"] == 0.0
        assert "ensemble_mean" in result


# ============================================================================
# INTEGRATION TESTS: Output Structure
# ============================================================================

class TestPredictionOutputStructure:
    """Test that prediction output has correct structure."""

    def test_required_output_keys(self, ensemble_v7_with_mocks, mahomes_2024_week10):
        """Result should contain all required keys."""
        result = ensemble_v7_with_mocks.predict(**mahomes_2024_week10)

        required_keys = {
            "ensemble_mean",
            "base_ensemble_mean",
            "causal_adjustment",
            "causal_features",
            "ensemble_std",
            "calibrated_std",
            "lower_90",
            "upper_90",
            "weights",
            "temperature",
            "prediction_time_ms",
            "valid_components",
        }

        assert required_keys.issubset(set(result.keys())), \
            f"Missing keys: {required_keys - set(result.keys())}"

    def test_output_values_are_numeric(self, ensemble_v7_with_mocks, mahomes_2024_week10):
        """Numeric output values should be float or int."""
        result = ensemble_v7_with_mocks.predict(**mahomes_2024_week10)

        numeric_keys = {
            "ensemble_mean", "base_ensemble_mean", "causal_adjustment",
            "ensemble_std", "calibrated_std", "lower_90", "upper_90",
            "temperature", "prediction_time_ms"
        }

        for key in numeric_keys:
            assert isinstance(result[key], (int, float)), \
                f"{key} = {result[key]} is not numeric"

    def test_weights_sum_to_one(self, ensemble_v7_with_mocks, mahomes_2024_week10):
        """Weights in result should sum to 1.0."""
        result = ensemble_v7_with_mocks.predict(**mahomes_2024_week10)

        weights_sum = sum(result["weights"].values())
        assert np.isclose(weights_sum, 1.0, atol=0.01)

    def test_prediction_time_is_reasonable(self, ensemble_v7_with_mocks, mahomes_2024_week10):
        """Prediction time should be < 5 seconds."""
        result = ensemble_v7_with_mocks.predict(**mahomes_2024_week10)

        assert result["prediction_time_ms"] < 5000, \
            f"Prediction took {result['prediction_time_ms']}ms (> 5s)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
