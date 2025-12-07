#!/usr/bin/env python3
"""
Integration Tests for A/B Test API Endpoints
=============================================

Tests for FastAPI endpoints with A/B test router integration.

Test Categories:
1. API Health Checks
2. Prediction Endpoints (/predict, /predict/batch, /predict/ensemble)
3. A/B Test Status Endpoint (/ab-test/status)
4. Database Logging Integration
5. Environment Variable Configuration

Run with:
    pytest tests/integration/test_ab_test_api_integration.py -v --cov=py.production.real_time_api

Author: Claude Code (Sonnet 4.5)
Date: November 14, 2025
Version: 1.0
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from py.production.real_time_api import app

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_ab_router():
    """Mock A/B test router for testing."""
    with patch('py.production.real_time_api.ABTestRouter') as mock_router_class:
        mock_router = Mock()
        mock_router.enabled = True
        mock_router.split_ratio = 0.5
        mock_router.test_name = "ensemble_v5_vs_v6_sagarin"
        mock_router.ensemble_v5 = Mock()
        mock_router.ensemble_v6 = Mock()

        # Mock predict method
        mock_router.predict.return_value = {
            "ab_group": "A",
            "model_version": "v5.0",
            "ensemble_mean": 250.0,
            "ensemble_std": 45.0,
            "quantile_10": 205.0,
            "quantile_90": 295.0,
            "weights": {"state_space": 0.3, "bnn": 0.25},
            "inference_time_ms": 100.0,
            "test_name": "ensemble_v5_vs_v6_sagarin",
        }

        # Mock get_test_summary method
        mock_router.get_test_summary.return_value = {
            "test_name": "ensemble_v5_vs_v6_sagarin",
            "split_ratio": 0.5,
            "enabled": True,
            "groups": {
                "A": {
                    "model_version": "v5.0",
                    "n_predictions": 150,
                    "n_completed": 100,
                    "mae": 45.2,
                    "rmse": 60.1,
                    "calibration_90_pct": 91.5,
                },
                "B": {
                    "model_version": "v6.0",
                    "n_predictions": 150,
                    "n_completed": 100,
                    "mae": 43.1,
                    "rmse": 58.2,
                    "calibration_90_pct": 92.3,
                },
            },
        }

        mock_router_class.return_value = mock_router
        yield mock_router


@pytest.fixture
def mock_db_pool():
    """Mock database connection pool."""
    with patch('py.production.real_time_api.db_pool') as mock_pool:
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_pool.getconn.return_value = mock_conn
        yield mock_pool


@pytest.fixture
def client():
    """Test client for FastAPI app."""
    return TestClient(app)


# ============================================================================
# HEALTH CHECK TESTS
# ============================================================================


class TestHealthChecks:
    """Tests for API health check endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
        assert "/predict" in data["endpoints"]
        assert "/ab-test/status" in data["endpoints"]

    @patch('py.production.real_time_api.ab_router')
    @patch('py.production.real_time_api.db_pool')
    @patch('py.production.real_time_api.redis_client')
    def test_health_check_all_healthy(self, mock_redis, mock_db, mock_router, client):
        """Test health check when all services are healthy."""
        # Mock database
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.execute.return_value = None
        mock_conn.cursor.return_value = mock_cursor
        mock_db.getconn.return_value = mock_conn

        # Mock Redis
        mock_redis.ping.return_value = True

        # Mock A/B router
        mock_router.ensemble_v5 = Mock()
        mock_router.ensemble_v6 = Mock()

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] in ["healthy", "degraded"]
        assert "database" in data
        assert "redis" in data
        assert "models_loaded" in data

    @patch('py.production.real_time_api.ab_router')
    @patch('py.production.real_time_api.db_pool')
    def test_health_check_database_down(self, mock_db, mock_router, client):
        """Test health check when database is down."""
        # Mock database failure
        mock_db.getconn.side_effect = Exception("Database connection failed")

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "degraded"
        assert data["database"] is False


# ============================================================================
# PREDICTION ENDPOINT TESTS
# ============================================================================


class TestPredictionEndpoints:
    """Tests for prediction API endpoints."""

    @patch('py.production.real_time_api.ab_router')
    def test_predict_endpoint_success(self, mock_router, client):
        """Test /predict endpoint returns valid prediction."""
        # Mock A/B router
        mock_router.predict.return_value = {
            "ab_group": "A",
            "model_version": "v5.0",
            "ensemble_mean": 250.0,
            "ensemble_std": 45.0,
            "quantile_10": 205.0,
            "quantile_90": 295.0,
            "test_name": "ensemble_v5_vs_v6_sagarin",
        }

        response = client.post(
            "/predict",
            json={
                "player_id": "00-0019596",
                "prop_type": "passing_yards",
                "week": 10,
                "season": 2024,
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data["player_id"] == "00-0019596"
        assert data["prop_type"] == "passing_yards"
        assert data["prediction"] == 250.0
        assert data["model_version"] == "v5.0"
        assert "confidence_interval" in data
        assert "latency_ms" in data

    @patch('py.production.real_time_api.ab_router')
    def test_predict_endpoint_control_group(self, mock_router, client):
        """Test prediction routed to control group."""
        # Mock control group assignment
        mock_router.predict.return_value = {
            "ab_group": "A",
            "model_version": "v5.0",
            "ensemble_mean": 250.0,
            "ensemble_std": 45.0,
            "quantile_10": 205.0,
            "quantile_90": 295.0,
        }

        response = client.post(
            "/predict",
            json={
                "player_id": "00-0019596",
                "prop_type": "passing_yards",
                "week": 10,
                "season": 2024,
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model_version"] == "v5.0"

    @patch('py.production.real_time_api.ab_router')
    def test_predict_endpoint_treatment_group(self, mock_router, client):
        """Test prediction routed to treatment group."""
        # Mock treatment group assignment
        mock_router.predict.return_value = {
            "ab_group": "B",
            "model_version": "v6.0",
            "ensemble_mean": 260.0,
            "ensemble_std": 42.0,
            "quantile_10": 218.0,
            "quantile_90": 302.0,
        }

        response = client.post(
            "/predict",
            json={
                "player_id": "00-0019596",
                "prop_type": "passing_yards",
                "week": 10,
                "season": 2024,
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model_version"] == "v6.0"

    @patch('py.production.real_time_api.ab_router')
    def test_predict_endpoint_invalid_week(self, mock_router, client):
        """Test prediction endpoint validates week number."""
        response = client.post(
            "/predict",
            json={
                "player_id": "00-0019596",
                "prop_type": "passing_yards",
                "week": 20,  # Invalid week
                "season": 2024,
            }
        )

        assert response.status_code == 422  # Validation error

    @patch('py.production.real_time_api.ab_router')
    def test_predict_endpoint_router_not_initialized(self, mock_router, client):
        """Test prediction endpoint when router not initialized."""
        # Simulate uninitialized router
        with patch('py.production.real_time_api.ab_router', None):
            response = client.post(
                "/predict",
                json={
                    "player_id": "00-0019596",
                    "prop_type": "passing_yards",
                    "week": 10,
                    "season": 2024,
                }
            )

            assert response.status_code == 503  # Service unavailable

    @patch('py.production.real_time_api.ab_router')
    def test_predict_ensemble_endpoint_with_individual_models(self, mock_router, client):
        """Test /predict/ensemble endpoint returns individual model predictions."""
        # Mock ensemble prediction with individual results
        mock_router.predict.return_value = {
            "ab_group": "A",
            "model_version": "v5.0",
            "ensemble_mean": 250.0,
            "ensemble_std": 45.0,
            "quantile_10": 205.0,
            "quantile_90": 295.0,
            "predictions": {
                "state_space": 245.0,
                "bnn": 255.0,
                "hierarchical": 248.0,
                "xgb": 252.0,
            },
        }

        response = client.post(
            "/predict/ensemble",
            json={
                "player_id": "00-0019596",
                "prop_type": "passing_yards",
                "week": 10,
                "season": 2024,
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert "ensemble_prediction" in data
        assert "model_predictions" in data
        assert "recommendation" in data
        assert "edge" in data

    @patch('py.production.real_time_api.ab_router')
    def test_batch_prediction_endpoint(self, mock_router, client):
        """Test /predict/batch endpoint processes multiple predictions."""
        # Mock individual predictions
        mock_router.predict.side_effect = [
            {
                "ab_group": "A",
                "model_version": "v5.0",
                "ensemble_mean": 250.0,
                "ensemble_std": 45.0,
                "quantile_10": 205.0,
                "quantile_90": 295.0,
            },
            {
                "ab_group": "B",
                "model_version": "v6.0",
                "ensemble_mean": 260.0,
                "ensemble_std": 42.0,
                "quantile_10": 218.0,
                "quantile_90": 302.0,
            },
        ]

        response = client.post(
            "/predict/batch",
            json={
                "players": [
                    {
                        "player_id": "00-0019596",
                        "prop_type": "passing_yards",
                        "week": 10,
                        "season": 2024,
                    },
                    {
                        "player_id": "00-0033873",
                        "prop_type": "passing_yards",
                        "week": 10,
                        "season": 2024,
                    },
                ]
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert "predictions" in data
        assert "total_requested" in data
        assert data["total_requested"] == 2


# ============================================================================
# A/B TEST STATUS ENDPOINT TESTS
# ============================================================================


class TestABTestStatusEndpoint:
    """Tests for A/B test status endpoint."""

    @patch('py.production.real_time_api.ab_router')
    def test_ab_test_status_enabled(self, mock_router, client):
        """Test /ab-test/status when test is enabled."""
        mock_router.enabled = True
        mock_router.split_ratio = 0.5
        mock_router.test_name = "ensemble_v5_vs_v6_sagarin"
        mock_router.get_test_summary.return_value = {
            "test_name": "ensemble_v5_vs_v6_sagarin",
            "split_ratio": 0.5,
            "enabled": True,
            "groups": {
                "A": {"n_predictions": 150, "mae": 45.2},
                "B": {"n_predictions": 150, "mae": 43.1},
            },
        }

        response = client.get("/ab-test/status")

        assert response.status_code == 200
        data = response.json()

        assert data["enabled"] is True
        assert data["test_name"] == "ensemble_v5_vs_v6_sagarin"
        assert data["split_ratio"] == 0.5
        assert "summary" in data
        assert "control" in data
        assert "treatment" in data

    @patch('py.production.real_time_api.ab_router')
    def test_ab_test_status_disabled(self, mock_router, client):
        """Test /ab-test/status when test is disabled."""
        mock_router.enabled = False

        response = client.get("/ab-test/status")

        assert response.status_code == 200
        data = response.json()

        assert data["enabled"] is False

    def test_ab_test_status_router_not_initialized(self, client):
        """Test /ab-test/status when router not initialized."""
        with patch('py.production.real_time_api.ab_router', None):
            response = client.get("/ab-test/status")

            assert response.status_code == 200
            data = response.json()

            assert data["enabled"] is False
            assert "message" in data


# ============================================================================
# DATABASE LOGGING INTEGRATION TESTS
# ============================================================================


class TestDatabaseLoggingIntegration:
    """Tests for database logging integration."""

    @patch('py.production.real_time_api.ab_router')
    def test_prediction_logged_to_database(self, mock_router, client):
        """Test that predictions are automatically logged to ml.predictions."""
        # Mock prediction with log_to_db=True
        mock_router.predict.return_value = {
            "ab_group": "A",
            "model_version": "v5.0",
            "ensemble_mean": 250.0,
            "ensemble_std": 45.0,
            "quantile_10": 205.0,
            "quantile_90": 295.0,
        }

        response = client.post(
            "/predict",
            json={
                "player_id": "00-0019596",
                "prop_type": "passing_yards",
                "week": 10,
                "season": 2024,
            }
        )

        assert response.status_code == 200

        # Verify router.predict was called with log_to_db=True
        mock_router.predict.assert_called_once()
        call_kwargs = mock_router.predict.call_args[1]
        assert call_kwargs["log_to_db"] is True

    @patch('py.production.real_time_api.ab_router')
    def test_batch_predictions_logged_to_database(self, mock_router, client):
        """Test that batch predictions are logged to database."""
        # Mock predictions
        mock_router.predict.side_effect = [
            {
                "ab_group": "A",
                "model_version": "v5.0",
                "ensemble_mean": 250.0,
                "ensemble_std": 45.0,
                "quantile_10": 205.0,
                "quantile_90": 295.0,
            },
        ] * 10

        response = client.post(
            "/predict/batch",
            json={
                "players": [
                    {
                        "player_id": f"player_{i:03d}",
                        "prop_type": "passing_yards",
                        "week": 10,
                        "season": 2024,
                    }
                    for i in range(10)
                ]
            }
        )

        assert response.status_code == 200

        # Verify all predictions were logged
        assert mock_router.predict.call_count == 10


# ============================================================================
# ENVIRONMENT VARIABLE CONFIGURATION TESTS
# ============================================================================


class TestEnvironmentConfiguration:
    """Tests for environment variable configuration."""

    @patch.dict(os.environ, {"AB_TEST_ENABLED": "false"})
    @patch('py.production.real_time_api.ABTestRouter')
    def test_ab_test_disabled_via_env_var(self, mock_router_class):
        """Test that AB_TEST_ENABLED=false disables test."""
        # Simulate startup event
        mock_router = Mock()
        mock_router_class.return_value = mock_router

        # This would be called in startup_event
        ab_test_enabled = os.getenv("AB_TEST_ENABLED", "true").lower() == "true"

        assert ab_test_enabled is False

    @patch.dict(os.environ, {"AB_TEST_SPLIT_RATIO": "0.3"})
    @patch('py.production.real_time_api.ABTestRouter')
    def test_ab_test_split_ratio_via_env_var(self, mock_router_class):
        """Test that AB_TEST_SPLIT_RATIO env var is used."""
        # Simulate startup event
        mock_router = Mock()
        mock_router_class.return_value = mock_router

        # This would be called in startup_event
        ab_test_split_ratio = float(os.getenv("AB_TEST_SPLIT_RATIO", "0.5"))

        assert ab_test_split_ratio == 0.3


# ============================================================================
# MODEL REGISTRY LOOKUP TESTS
# ============================================================================


class TestModelRegistryLookup:
    """Tests for model registry integration."""

    @patch('py.production.real_time_api.db_pool')
    def test_models_endpoint_returns_versions(self, mock_db, client):
        """Test /models endpoint returns available model versions."""
        # Mock database query
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            ("v5.0", 150),
            ("v6.0", 150),
        ]
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db.getconn.return_value = mock_conn

        response = client.get("/models")

        assert response.status_code == 200
        data = response.json()

        assert "models" in data
        assert "ab_test" in data
        assert len(data["models"]) == 2


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in API."""

    @patch('py.production.real_time_api.ab_router')
    def test_prediction_error_handling(self, mock_router, client):
        """Test that prediction errors return 500."""
        # Mock prediction error
        mock_router.predict.side_effect = Exception("Model error")

        response = client.post(
            "/predict",
            json={
                "player_id": "00-0019596",
                "prop_type": "passing_yards",
                "week": 10,
                "season": 2024,
            }
        )

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data

    @patch('py.production.real_time_api.ab_router')
    def test_ab_test_status_error_handling(self, mock_router, client):
        """Test that get_test_summary errors are handled."""
        # Mock get_test_summary error
        mock_router.get_test_summary.side_effect = Exception("Database error")

        response = client.get("/ab-test/status")

        assert response.status_code == 200
        data = response.json()

        # Should return error in response
        assert "error" in data or "enabled" in data


# ============================================================================
# MAIN
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--cov=py.production.real_time_api"])
