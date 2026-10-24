#!/usr/bin/env python3
"""
Unit tests for EnsemblePredictor model loading.

Tests critical fixes:
- Task 3: BNN model loading with inference
- Task 4: XGBoost model loading with inference
- Task 5: R model loading via RService
- Task 6: Feature conversion helper

Author: NFL Analytics Team
Date: November 9, 2025
"""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from nfl_analytics.ensemble.strategy.base import ModelType


@pytest.fixture(autouse=True)
def _no_redis(monkeypatch):
    """Use only the in-memory model cache: constructing the predictor otherwise waits
    ~5s for a Redis connection and lets a local Redis leak state between tests."""
    monkeypatch.setattr("nfl_analytics.ensemble.strategy.ensemble_predictor.get_default_cache", None)


class TestBNNModelLoading:
    """Test BNN model loading (Task 3)."""

    @pytest.fixture(autouse=True)
    def _requires_pymc(self):
        # bayesian_neural_network imports pymc/arviz at module level (uv sync --extra bayes)
        pytest.importorskip("pymc")

    @pytest.fixture
    def mock_bnn_model(self):
        """Mock BayesianNeuralNetwork model."""
        model = Mock()
        model.hidden_dims = [128, 64, 32]
        model.inference_method = "variational"
        model.predict.return_value = (
            np.array([255.0]),  # mean
            np.array([18.0]),  # std
        )
        return model

    @pytest.fixture
    def predictor(self):
        """EnsemblePredictor instance."""
        # Import here to avoid circular dependencies
        from nfl_analytics.ensemble.strategy.base import EnsembleConfig
        from nfl_analytics.ensemble.strategy.ensemble_predictor import UnifiedEnsemblePredictor

        predictor = UnifiedEnsemblePredictor(EnsembleConfig(strategy_name="weighted_average"))
        return predictor

    def test_bnn_model_loads_and_caches(self, predictor, mock_bnn_model):
        """Test that BNN model is loaded and cached."""
        model_path = Path("models/bayesian/bnn_passing_v1.pkl")

        with patch(
            "nfl_analytics.models.bayesian_neural_network.BayesianNeuralNetwork.load",
            return_value=mock_bnn_model,
        ):
            # First prediction
            predictor._predict_bnn_model(
                model_name="bnn_v1",
                path=model_path,
                player_id="12345",
                features={"yards_per_game": 250.0, "attempts_per_game": 30.0},
            )

            # Second prediction (should use cached model)
            predictor._predict_bnn_model(
                model_name="bnn_v1",
                path=model_path,
                player_id="12345",
                features={"yards_per_game": 250.0, "attempts_per_game": 30.0},
            )

            # Model should be loaded only once
            assert "bnn_v1" in predictor._model_cache
            assert predictor._model_cache["bnn_v1"] == mock_bnn_model

    def test_bnn_prediction_returns_mean_and_std(self, predictor, mock_bnn_model):
        """Test that BNN prediction includes mean and std."""
        model_path = Path("models/bayesian/bnn_passing_v1.pkl")

        with patch(
            "nfl_analytics.models.bayesian_neural_network.BayesianNeuralNetwork.load",
            return_value=mock_bnn_model,
        ):
            result = predictor._predict_bnn_model(
                model_name="bnn_v1",
                path=model_path,
                player_id="12345",
                features={"yards_per_game": 250.0},
            )

            assert result is not None
            assert result.model_type == ModelType.BNN
            assert result.prediction_mean == 255.0
            assert result.prediction_std == 18.0
            assert 0 <= result.confidence <= 1

    def test_bnn_confidence_based_on_uncertainty(self, predictor):
        """Test that BNN confidence decreases with higher uncertainty."""
        model_path = Path("models/bayesian/bnn_passing_v1.pkl")

        # Mock model with low uncertainty
        mock_model_low_std = Mock()
        mock_model_low_std.hidden_dims = [128, 64]
        mock_model_low_std.inference_method = "variational"
        mock_model_low_std.predict.return_value = (
            np.array([250.0]),
            np.array([10.0]),  # Low std
        )

        # Mock model with high uncertainty
        mock_model_high_std = Mock()
        mock_model_high_std.hidden_dims = [128, 64]
        mock_model_high_std.inference_method = "variational"
        mock_model_high_std.predict.return_value = (
            np.array([250.0]),
            np.array([40.0]),  # High std
        )

        with patch("nfl_analytics.models.bayesian_neural_network.BayesianNeuralNetwork.load") as mock_load:
            # Low uncertainty prediction
            mock_load.return_value = mock_model_low_std
            result_low = predictor._predict_bnn_model(
                model_name="bnn_low",
                path=model_path,
                player_id="12345",
                features={"yards": 250.0},
            )

            # High uncertainty prediction
            mock_load.return_value = mock_model_high_std
            predictor._model_cache.clear()  # Clear cache
            result_high = predictor._predict_bnn_model(
                model_name="bnn_high",
                path=model_path,
                player_id="12345",
                features={"yards": 250.0},
            )

            # Lower uncertainty should have higher confidence
            assert result_low.confidence > result_high.confidence

    def test_bnn_returns_none_when_not_installed(self, predictor):
        """Test graceful degradation when BNN not installed."""
        model_path = Path("models/bayesian/bnn_passing_v1.pkl")

        # Mock BayesianNeuralNetwork as None (not installed)
        with patch("nfl_analytics.ensemble.strategy.ensemble_predictor.BayesianNeuralNetwork", None):
            result = predictor._predict_bnn_model(
                model_name="bnn_v1",
                path=model_path,
                player_id="12345",
                features={"yards": 250.0},
            )

            assert result is None


class TestXGBoostModelLoading:
    """Test XGBoost model loading (Task 4)."""

    @pytest.fixture
    def mock_xgb_model(self):
        """Mock XGBoost booster."""
        model = Mock()
        model.predict.return_value = np.array([260.0])
        return model

    @pytest.fixture
    def predictor(self):
        """EnsemblePredictor instance."""
        from nfl_analytics.ensemble.strategy.base import EnsembleConfig
        from nfl_analytics.ensemble.strategy.ensemble_predictor import UnifiedEnsemblePredictor

        predictor = UnifiedEnsemblePredictor(EnsembleConfig(strategy_name="weighted_average"))
        return predictor

    def test_xgboost_model_loads_and_caches(self, predictor, mock_xgb_model):
        """Test that XGBoost model is loaded and cached."""
        model_path = Path("models/xgboost/player_props_v1.json")

        with patch("xgboost.Booster") as mock_booster_class:
            mock_booster_instance = Mock()
            mock_booster_instance.load_model = Mock()
            mock_booster_instance.predict.return_value = np.array([260.0])
            mock_booster_class.return_value = mock_booster_instance

            # First prediction
            result = predictor._predict_xgboost_model(
                model_name="xgb_v1",
                path=model_path,
                player_id="12345",
                features={"yards_per_game": 250.0, "attempts": 30.0},
            )

            # Model should be cached
            assert "xgb_v1" in predictor._model_cache
            assert result is not None
            assert result.model_type == ModelType.XGBOOST

    def test_xgboost_uses_heuristic_uncertainty(self, predictor):
        """Test that XGBoost uses 15% CV heuristic for uncertainty."""
        model_path = Path("models/xgboost/player_props_v1.json")

        with patch("xgboost.Booster") as mock_booster_class:
            mock_booster_instance = Mock()
            mock_booster_instance.load_model = Mock()
            mock_booster_instance.predict.return_value = np.array([200.0])
            mock_booster_class.return_value = mock_booster_instance

            result = predictor._predict_xgboost_model(
                model_name="xgb_v1",
                path=model_path,
                player_id="12345",
                features={"yards": 250.0},
            )

            # Uncertainty should be ~15% of prediction
            expected_std = 200.0 * 0.15
            assert np.isclose(result.prediction_std, expected_std, rtol=0.01)

    def test_xgboost_confidence_based_on_features(self, predictor):
        """Test that XGBoost confidence based on feature completeness."""
        model_path = Path("models/xgboost/player_props_v1.json")

        with patch("xgboost.Booster") as mock_booster_class:
            mock_booster_instance = Mock()
            mock_booster_instance.load_model = Mock()
            mock_booster_instance.predict.return_value = np.array([250.0])
            mock_booster_class.return_value = mock_booster_instance

            # Many features (complete)
            features_complete = {f"feat_{i}": i for i in range(50)}
            result_complete = predictor._predict_xgboost_model(
                model_name="xgb_v1",
                path=model_path,
                player_id="12345",
                features=features_complete,
            )

            # Few features (incomplete)
            features_incomplete = {f"feat_{i}": i for i in range(10)}
            predictor._model_cache.clear()
            result_incomplete = predictor._predict_xgboost_model(
                model_name="xgb_v2",
                path=model_path,
                player_id="12345",
                features=features_incomplete,
            )

            # More features should have higher confidence
            assert result_complete.confidence > result_incomplete.confidence

    def test_xgboost_returns_none_when_not_installed(self, predictor):
        """Test graceful degradation when XGBoost not installed."""
        model_path = Path("models/xgboost/player_props_v1.json")

        with patch("nfl_analytics.ensemble.strategy.ensemble_predictor.xgb", None):
            result = predictor._predict_xgboost_model(
                model_name="xgb_v1",
                path=model_path,
                player_id="12345",
                features={"yards": 250.0},
            )

            assert result is None


class TestRModelLoading:
    """Test R model loading via RService (Task 5)."""

    @pytest.fixture
    def mock_r_service(self):
        """Mock RService."""
        service = Mock()
        health = Mock()
        health.is_healthy = True
        health.r_version = "4.3.0"
        health.missing_packages = []
        service.health_check.return_value = health

        # Mock R execution result
        result = Mock()
        result.success = True
        result.stdout = '{"mean": 255.50, "std": 19.20}'
        result.stderr = ""
        service.execute_with_cache.return_value = result

        return service

    @pytest.fixture
    def predictor(self):
        """EnsemblePredictor instance."""
        from nfl_analytics.ensemble.strategy.base import EnsembleConfig
        from nfl_analytics.ensemble.strategy.ensemble_predictor import UnifiedEnsemblePredictor

        predictor = UnifiedEnsemblePredictor(EnsembleConfig(strategy_name="weighted_average"))
        return predictor

    def test_r_model_checks_service_health(self, predictor, mock_r_service):
        """Test that R service health is checked before prediction."""
        model_path = Path("models/bayesian/passing_yards_hierarchical_v2.rds")

        with patch(
            "nfl_analytics.ensemble.strategy.ensemble_predictor.get_r_service",
            return_value=mock_r_service,
        ):
            result = predictor._predict_r_model(
                model_name="hierarchical_v2",
                path=model_path,
                player_id="12345",
                season=2024,
                week=10,
            )

            # Health check should be called
            mock_r_service.health_check.assert_called_once()
            assert result is not None

    def test_r_model_returns_none_when_unhealthy(self, predictor):
        """Test that None returned when R service unhealthy."""
        model_path = Path("models/bayesian/passing_yards_hierarchical_v2.rds")

        mock_service = Mock()
        health = Mock()
        health.is_healthy = False
        health.missing_packages = ["brms", "cmdstanr"]
        mock_service.health_check.return_value = health

        with patch(
            "nfl_analytics.ensemble.strategy.ensemble_predictor.get_r_service",
            return_value=mock_service,
        ):
            result = predictor._predict_r_model(
                model_name="hierarchical_v2",
                path=model_path,
                player_id="12345",
                season=2024,
                week=10,
            )

            assert result is None

    def test_r_model_parses_json_output(self, predictor, mock_r_service):
        """Test that JSON output from R is parsed correctly."""
        model_path = Path("models/bayesian/passing_yards_state_space_v2.rds")

        with patch(
            "nfl_analytics.ensemble.strategy.ensemble_predictor.get_r_service",
            return_value=mock_r_service,
        ):
            result = predictor._predict_r_model(
                model_name="state_space_v2",
                path=model_path,
                player_id="12345",
                season=2024,
                week=10,
            )

            assert result.prediction_mean == 255.50
            assert result.prediction_std == 19.20

    def test_r_model_executes_with_retry(self, predictor, mock_r_service):
        """Test that R execution uses retry=True."""
        model_path = Path("models/bayesian/passing_yards_hierarchical_v2.rds")

        with patch(
            "nfl_analytics.ensemble.strategy.ensemble_predictor.get_r_service",
            return_value=mock_r_service,
        ):
            predictor._predict_r_model(
                model_name="hierarchical_v2",
                path=model_path,
                player_id="12345",
                season=2024,
                week=10,
            )

            # Execute should be called with retry=True
            call_args = mock_r_service.execute_with_cache.call_args
            assert call_args[1]["retry"] is True
            assert call_args[1]["timeout"] == 60

    def test_r_model_returns_none_when_not_installed(self, predictor):
        """Test graceful degradation when R service not available."""
        model_path = Path("models/bayesian/passing_yards_hierarchical_v2.rds")

        with patch("nfl_analytics.ensemble.strategy.ensemble_predictor.get_r_service", None):
            result = predictor._predict_r_model(
                model_name="hierarchical_v2",
                path=model_path,
                player_id="12345",
                season=2024,
                week=10,
            )

            assert result is None


class TestFeatureConversion:
    """Test feature conversion helper (Task 6)."""

    @pytest.fixture
    def predictor(self):
        """EnsemblePredictor instance."""
        from nfl_analytics.ensemble.strategy.base import EnsembleConfig
        from nfl_analytics.ensemble.strategy.ensemble_predictor import UnifiedEnsemblePredictor

        predictor = UnifiedEnsemblePredictor(EnsembleConfig(strategy_name="weighted_average"))
        return predictor

    def test_converts_dict_to_sorted_array(self, predictor):
        """Test that features dict is converted to sorted numpy array."""
        features = {"z_feature": 3.0, "a_feature": 1.0, "m_feature": 2.0}

        result = predictor._features_dict_to_array(features, model_type="bnn")

        # Should be sorted by key: a_feature, m_feature, z_feature
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_converts_to_float32(self, predictor):
        """Test that array is float32 dtype."""
        features = {"feat1": 1, "feat2": 2, "feat3": 3}  # Integers

        result = predictor._features_dict_to_array(features, model_type="xgboost")

        assert result.dtype == np.float32

    def test_handles_missing_features_gracefully(self, predictor):
        """Test that missing features don't crash (return None)."""
        features = None  # Invalid

        result = predictor._features_dict_to_array(features, model_type="bnn")

        assert result is None

    def test_deterministic_ordering(self, predictor):
        """Test that feature ordering is deterministic across calls."""
        features = {"b": 2.0, "c": 3.0, "a": 1.0}

        result1 = predictor._features_dict_to_array(features, model_type="bnn")
        result2 = predictor._features_dict_to_array(features, model_type="bnn")

        np.testing.assert_array_equal(result1, result2)


class TestModelLoadingIntegration:
    """Integration tests for full prediction workflow."""

    @pytest.fixture
    def predictor(self):
        """EnsemblePredictor with weighted average strategy."""
        from nfl_analytics.ensemble.strategy.base import EnsembleConfig
        from nfl_analytics.ensemble.strategy.ensemble_predictor import UnifiedEnsemblePredictor

        weights = {
            "state_space_v2": 0.30,
            "hierarchical_v2": 0.25,
            "bnn_v1": 0.25,
            "xgboost_v1": 0.20,
        }
        predictor = UnifiedEnsemblePredictor(
            EnsembleConfig(strategy_name="weighted_average", weights=weights)
        )
        return predictor

    def test_model_cache_shared_across_predictions(self, predictor):
        """Test that model cache is shared across different predictions."""
        pytest.importorskip("pymc")  # BNN path; uv sync --extra bayes
        # Clear cache
        predictor._model_cache.clear()

        # Mock BNN model
        mock_model = Mock()
        mock_model.hidden_dims = [128, 64]
        mock_model.inference_method = "variational"
        mock_model.predict.return_value = (np.array([250.0]), np.array([15.0]))

        model_path = Path("models/bayesian/bnn_passing_v1.pkl")

        with patch(
            "nfl_analytics.models.bayesian_neural_network.BayesianNeuralNetwork.load",
            return_value=mock_model,
        ) as mock_load:
            # First prediction
            predictor._predict_bnn_model(
                model_name="bnn_v1",
                path=model_path,
                player_id="player1",
                features={"yards": 250.0},
            )

            # Second prediction (different player, same model)
            predictor._predict_bnn_model(
                model_name="bnn_v1",
                path=model_path,
                player_id="player2",
                features={"yards": 260.0},
            )

            # Model should be loaded only once
            mock_load.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
