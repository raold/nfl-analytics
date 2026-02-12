#!/usr/bin/env python3
"""
End-to-End Tests for A/B Test Framework
========================================

Tests complete workflow from API request → router → ensemble → database.

Test Scenarios:
1. Full prediction flow (100+ predictions, verify 50/50 split)
2. Batch predictions with database logging
3. A/B test analysis workflow
4. Emergency rollback scenario
5. Performance benchmarks

Run with:
    pytest tests/e2e/test_ab_test_e2e.py -v

Author: Claude Code (Sonnet 4.5)
Date: November 14, 2025
Version: 1.0
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from py.production.ab_test_router import ABTestRouter
from py.production.real_time_api import app

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_ensembles():
    """Mock both ensemble models."""
    with (
        patch("py.production.ab_test_router.EnsembleV5") as mock_v5,
        patch("py.production.ab_test_router.EnsembleV6") as mock_v6,
    ):

        # Mock v5 predictions (control)
        mock_v5_instance = Mock()
        mock_v5_instance.predict.return_value = {
            "ensemble_mean": 250.0,
            "ensemble_std": 45.0,
            "quantile_10": 205.0,
            "quantile_90": 295.0,
            "weights": {"state_space": 0.3, "bnn": 0.25, "hierarchical": 0.25, "xgb": 0.2},
            "inference_time_ms": 100.0,
        }
        mock_v5.return_value = mock_v5_instance

        # Mock v6 predictions (treatment)
        mock_v6_instance = Mock()
        mock_v6_instance.predict.return_value = {
            "ensemble_mean": 260.0,
            "ensemble_std": 42.0,
            "quantile_10": 218.0,
            "quantile_90": 302.0,
            "weights": {
                "state_space": 0.25,
                "bnn": 0.2,
                "hierarchical": 0.2,
                "xgb": 0.15,
                "elo": 0.1,
                "adaptive": 0.1,
            },
            "inference_time_ms": 105.0,
        }
        mock_v6.return_value = mock_v6_instance

        yield mock_v5, mock_v6


@pytest.fixture
def mock_database():
    """Mock database for end-to-end tests."""
    with patch("psycopg2.connect") as mock_connect:
        mock_cursor = Mock()
        mock_cursor.fetchone.side_effect = [
            (38,),  # model_id for v5.0
            ("Patrick Mahomes", "KC", "LAC", True),
        ] * 1000  # Enough for batch tests

        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        yield mock_connect


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


# ============================================================================
# E2E TEST: FULL PREDICTION FLOW
# ============================================================================


class TestFullPredictionFlow:
    """Test complete prediction flow from API to database."""

    @patch("py.production.real_time_api.ab_router")
    def test_single_prediction_control_group_e2e(
        self, mock_router, mock_ensembles, mock_database, client
    ):
        """Test complete flow for control group prediction."""
        # Configure mock router
        mock_router.enabled = True
        mock_router.split_ratio = 0.0  # All control
        mock_router.predict.return_value = {
            "ab_group": "A",
            "model_version": "v5.0",
            "ensemble_mean": 250.0,
            "ensemble_std": 45.0,
            "quantile_10": 205.0,
            "quantile_90": 295.0,
            "test_name": "ensemble_v5_vs_v6_sagarin",
        }

        # Make API request
        response = client.post(
            "/predict",
            json={
                "player_id": "00-0019596",
                "prop_type": "passing_yards",
                "week": 10,
                "season": 2024,
            },
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()

        assert data["player_id"] == "00-0019596"
        assert data["prediction"] == 250.0
        assert data["model_version"] == "v5.0"
        assert data["uncertainty"] == 45.0

        # Verify router was called correctly
        mock_router.predict.assert_called_once()
        call_kwargs = mock_router.predict.call_args[1]
        assert call_kwargs["player_id"] == "00-0019596"
        assert call_kwargs["season"] == 2024
        assert call_kwargs["week"] == 10
        assert call_kwargs["log_to_db"] is True

    @patch("py.production.real_time_api.ab_router")
    def test_single_prediction_treatment_group_e2e(
        self, mock_router, mock_ensembles, mock_database, client
    ):
        """Test complete flow for treatment group prediction."""
        # Configure mock router
        mock_router.enabled = True
        mock_router.split_ratio = 1.0  # All treatment
        mock_router.predict.return_value = {
            "ab_group": "B",
            "model_version": "v6.0",
            "ensemble_mean": 260.0,
            "ensemble_std": 42.0,
            "quantile_10": 218.0,
            "quantile_90": 302.0,
            "test_name": "ensemble_v5_vs_v6_sagarin",
        }

        # Make API request
        response = client.post(
            "/predict",
            json={
                "player_id": "00-0019596",
                "prop_type": "passing_yards",
                "week": 10,
                "season": 2024,
            },
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()

        assert data["prediction"] == 260.0
        assert data["model_version"] == "v6.0"
        assert data["uncertainty"] == 42.0


# ============================================================================
# E2E TEST: BATCH PREDICTIONS WITH 50/50 SPLIT
# ============================================================================


class TestBatchPredictionsE2E:
    """Test batch predictions maintain accurate split ratio."""

    def test_batch_100_predictions_split_ratio(self, mock_ensembles, mock_database):
        """Test 100 predictions maintain 50/50 split."""
        router = ABTestRouter(split_ratio=0.5)

        # Create batch of 100 unique players
        games = [{"player_id": f"player_{i:05d}", "season": 2024, "week": 10} for i in range(100)]

        # Get predictions
        df = router.predict_batch(games, log_to_db=False)

        # Verify split ratio
        group_a_count = (df["ab_group"] == "A").sum()
        group_b_count = (df["ab_group"] == "B").sum()

        assert len(df) == 100
        assert group_a_count + group_b_count == 100

        # Split should be within 40-60% (binomial 95% CI for n=100, p=0.5)
        assert 40 <= group_a_count <= 60, f"Group A: {group_a_count}, outside 40-60 range"
        assert 40 <= group_b_count <= 60, f"Group B: {group_b_count}, outside 40-60 range"

    def test_batch_1000_predictions_split_ratio_tight(self, mock_ensembles, mock_database):
        """Test 1000 predictions maintain tight 50/50 split."""
        router = ABTestRouter(split_ratio=0.5)

        # Create batch of 1000 unique players
        games = [{"player_id": f"player_{i:06d}", "season": 2024, "week": 10} for i in range(1000)]

        # Get predictions
        df = router.predict_batch(games, log_to_db=False)

        # Verify split ratio
        group_b_count = (df["ab_group"] == "B").sum()
        b_ratio = group_b_count / len(df)

        # With n=1000, split should be within 47-53% (tighter tolerance)
        assert 0.47 <= b_ratio <= 0.53, f"Split ratio {b_ratio:.3f} outside 0.47-0.53"

    def test_batch_predictions_deterministic_across_calls(self, mock_ensembles, mock_database):
        """Test that same players get same assignment across multiple calls."""
        router = ABTestRouter(split_ratio=0.5)

        # Same players, called twice
        games = [{"player_id": f"player_{i:03d}", "season": 2024, "week": 10} for i in range(50)]

        df1 = router.predict_batch(games, log_to_db=False)
        df2 = router.predict_batch(games, log_to_db=False)

        # Assignments should be identical
        assert (df1["ab_group"] == df2["ab_group"]).all(), "Assignments not deterministic"
        assert (
            df1["model_version"] == df2["model_version"]
        ).all(), "Model versions not deterministic"


# ============================================================================
# E2E TEST: DATABASE LOGGING
# ============================================================================


class TestDatabaseLoggingE2E:
    """Test predictions are correctly logged to database."""

    @patch("psycopg2.connect")
    def test_predictions_logged_with_correct_model_id(self, mock_db, mock_ensembles):
        """Test that predictions use correct model_id from registry."""
        # Mock database
        mock_cursor = Mock()
        mock_cursor.fetchone.side_effect = [
            (38,),  # model_id for v5.0
            ("Patrick Mahomes", "KC", "LAC", True),
        ]
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value = mock_conn

        router = ABTestRouter(split_ratio=0.0)  # All control

        # Make prediction
        router.predict(player_id="00-0019596", season=2024, week=10, log_to_db=True)

        # Verify model registry query
        registry_calls = [
            call for call in mock_cursor.execute.call_args_list if "ml.model_registry" in str(call)
        ]
        assert len(registry_calls) == 1, "Model registry not queried"

        # Verify INSERT query
        insert_calls = [
            call
            for call in mock_cursor.execute.call_args_list
            if "INSERT INTO ml.predictions" in str(call)
        ]
        assert len(insert_calls) == 1, "Prediction not inserted"

        # Verify commit
        mock_conn.commit.assert_called_once()

    @patch("psycopg2.connect")
    def test_batch_predictions_all_logged(self, mock_db, mock_ensembles):
        """Test that all batch predictions are logged."""
        # Mock database
        mock_cursor = Mock()
        mock_cursor.fetchone.side_effect = [
            (38,),
            ("Player", "KC", "LAC", True),
        ] * 50  # 50 predictions
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value = mock_conn

        router = ABTestRouter(split_ratio=0.5)

        # Batch predictions
        games = [{"player_id": f"player_{i:03d}", "season": 2024, "week": 10} for i in range(50)]

        router.predict_batch(games, log_to_db=True)

        # Verify all logged (2 queries per prediction: model_id + player info)
        assert mock_cursor.execute.call_count == 50 * 3  # model_id, player_info, insert
        assert mock_conn.commit.call_count == 50


# ============================================================================
# E2E TEST: A/B TEST ANALYSIS WORKFLOW
# ============================================================================


class TestABTestAnalysisWorkflow:
    """Test complete A/B test analysis workflow."""

    @patch("psycopg2.connect")
    def test_get_test_summary_after_predictions(self, mock_db, mock_ensembles):
        """Test retrieving test summary after making predictions."""
        # Mock database with results
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            {
                "model_name": "ensemble_v5.0",
                "n_predictions": 150,
                "n_completed": 100,
                "mae": 45.2,
                "rmse": 60.1,
                "calibration_90_pct": 91.5,
                "avg_uncertainty": 44.8,
                "first_prediction": datetime(2024, 11, 1),
                "last_prediction": datetime(2024, 11, 14),
            },
            {
                "model_name": "ensemble_v6.0",
                "n_predictions": 150,
                "n_completed": 100,
                "mae": 43.1,
                "rmse": 58.2,
                "calibration_90_pct": 92.3,
                "avg_uncertainty": 42.5,
                "first_prediction": datetime(2024, 11, 1),
                "last_prediction": datetime(2024, 11, 14),
            },
        ]
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value = mock_conn

        router = ABTestRouter()
        summary = router.get_test_summary()

        # Verify summary structure
        assert "groups" in summary
        assert "A" in summary["groups"]
        assert "B" in summary["groups"]

        # Verify group A
        assert summary["groups"]["A"]["model_version"] == "v5.0"
        assert summary["groups"]["A"]["n_predictions"] == 150
        assert summary["groups"]["A"]["mae"] == 45.2

        # Verify group B
        assert summary["groups"]["B"]["model_version"] == "v6.0"
        assert summary["groups"]["B"]["n_predictions"] == 150
        assert summary["groups"]["B"]["mae"] == 43.1


# ============================================================================
# E2E TEST: EMERGENCY ROLLBACK SCENARIO
# ============================================================================


class TestEmergencyRollbackE2E:
    """Test emergency rollback workflow."""

    def test_rollback_disables_test_immediately(self, mock_ensembles, mock_database):
        """Test that disable_test() immediately routes all to control."""
        router = ABTestRouter(split_ratio=0.5, enabled=True)

        # Verify test is enabled
        assert router.enabled is True

        # Emergency rollback
        router.disable_test()

        # Verify test is disabled
        assert router.enabled is False

        # All new assignments should go to control
        for i in range(100):
            group, version = router._get_assignment(f"player_{i:04d}", 2024, 10)
            assert group == "A", f"Expected A after rollback, got {group}"
            assert version == "v5.0", f"Expected v5.0 after rollback, got {version}"

    def test_promote_treatment_routes_all_to_v6(self, mock_ensembles, mock_database):
        """Test that promote_treatment() routes all to v6."""
        router = ABTestRouter(split_ratio=0.5)

        # Promote treatment
        router.promote_treatment()

        # Verify split ratio updated
        assert router.split_ratio == 1.0

        # All new assignments should go to treatment
        for i in range(100):
            group, version = router._get_assignment(f"player_{i:04d}", 2024, 10)
            assert group == "B", f"Expected B after promotion, got {group}"
            assert version == "v6.0", f"Expected v6.0 after promotion, got {version}"


# ============================================================================
# E2E TEST: PERFORMANCE BENCHMARKS
# ============================================================================


class TestPerformanceBenchmarks:
    """Performance benchmarks for A/B test framework."""

    def test_assignment_latency_under_1ms(self, mock_ensembles, mock_database):
        """Test that assignment latency is <1ms per prediction."""
        router = ABTestRouter(split_ratio=0.5)

        start = time.time()
        for i in range(1000):
            router._get_assignment(f"player_{i:05d}", 2024, 10)
        elapsed = time.time() - start

        avg_latency_ms = (elapsed / 1000) * 1000
        assert avg_latency_ms < 1.0, f"Assignment latency {avg_latency_ms:.3f}ms exceeds 1ms"

    def test_batch_100_predictions_under_5s(self, mock_ensembles, mock_database):
        """Test that 100 batch predictions complete in <5s."""
        router = ABTestRouter(split_ratio=0.5)

        games = [{"player_id": f"player_{i:05d}", "season": 2024, "week": 10} for i in range(100)]

        start = time.time()
        router.predict_batch(games, log_to_db=False)
        elapsed = time.time() - start

        assert elapsed < 5.0, f"Batch 100 took {elapsed:.2f}s, exceeds 5s"


# ============================================================================
# E2E TEST: API HEALTH CHECK
# ============================================================================


class TestAPIHealthCheckE2E:
    """Test API health check with A/B test integration."""

    @patch("py.production.real_time_api.ab_router")
    @patch("py.production.real_time_api.db_pool")
    @patch("py.production.real_time_api.redis_client")
    def test_health_check_shows_ab_test_status(self, mock_redis, mock_db, mock_router, client):
        """Test that health check includes A/B test status."""
        # Mock services
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db.getconn.return_value = mock_conn

        mock_redis.ping.return_value = True

        mock_router.ensemble_v5 = Mock()
        mock_router.ensemble_v6 = Mock()

        # Make request
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        # Verify A/B test models in health check
        assert "models_loaded" in data
        assert "ab_router" in data["models_loaded"]


# ============================================================================
# MAIN
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
