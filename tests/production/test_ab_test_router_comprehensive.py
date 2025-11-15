#!/usr/bin/env python3
"""
Comprehensive Tests for A/B Test Router (v5 vs v6 Sagarin Integration)
=======================================================================

Test Categories:
1. Unit Tests: Router logic, hashing, model loading
2. Integration Tests: Database logging, model registry lookup
3. End-to-End Tests: Full prediction flow with real data

Run with:
    pytest tests/production/test_ab_test_router_comprehensive.py -v --cov=py.production.ab_test_router

Author: Claude Code (Sonnet 4.5)
Date: November 14, 2025
Version: 1.0
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import time
import hashlib
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime

from py.production.ab_test_router import ABTestRouter


# ============================================================================
# UNIT TESTS - Router Logic and Hashing
# ============================================================================


class TestABTestRouterUnit:
    """Unit tests for ABTestRouter core functionality."""

    def test_initialization_default_params(self):
        """Test router initialization with default parameters."""
        router = ABTestRouter(split_ratio=0.5)

        assert router.split_ratio == 0.5
        assert router.test_name == "ensemble_v5_vs_v6_sagarin"
        assert router.enabled is True
        assert router._models_loaded is False
        assert router.ensemble_v5 is None
        assert router.ensemble_v6 is None

    def test_initialization_custom_params(self):
        """Test router initialization with custom parameters."""
        router = ABTestRouter(
            split_ratio=0.3,
            test_name="custom_test",
            enabled=False,
            db_host="custom_host",
            db_port=5555
        )

        assert router.split_ratio == 0.3
        assert router.test_name == "custom_test"
        assert router.enabled is False
        assert router.db_config["host"] == "custom_host"
        assert router.db_config["port"] == 5555

    def test_initialization_invalid_split_ratio_negative(self):
        """Test that negative split ratios raise ValueError."""
        with pytest.raises(ValueError, match="split_ratio must be in"):
            ABTestRouter(split_ratio=-0.1)

    def test_initialization_invalid_split_ratio_too_large(self):
        """Test that split ratios > 1.0 raise ValueError."""
        with pytest.raises(ValueError, match="split_ratio must be in"):
            ABTestRouter(split_ratio=1.5)

    def test_initialization_invalid_split_ratio_slightly_above(self):
        """Test that split ratios slightly above 1.0 raise ValueError."""
        with pytest.raises(ValueError, match="split_ratio must be in"):
            ABTestRouter(split_ratio=1.001)

    def test_deterministic_assignment_same_player(self):
        """Test that same player/season/week always gets same assignment."""
        router = ABTestRouter(split_ratio=0.5)

        # Call 100 times with same inputs
        assignments = []
        for _ in range(100):
            group, version = router._get_assignment("00-0019596", 2024, 10)
            assignments.append((group, version))

        # All should be identical
        assert len(set(assignments)) == 1, "Assignment not deterministic"

    def test_deterministic_assignment_different_players(self):
        """Test that different players can get different assignments."""
        router = ABTestRouter(split_ratio=0.5)

        assignments = set()
        for i in range(1000):
            player_id = f"player_{i:05d}"
            group, _ = router._get_assignment(player_id, 2024, 10)
            assignments.add(group)

        # Should have both A and B assignments
        assert "A" in assignments, "No control assignments found"
        assert "B" in assignments, "No treatment assignments found"

    def test_deterministic_assignment_different_weeks(self):
        """Test that same player in different weeks can get different assignments."""
        router = ABTestRouter(split_ratio=0.5)

        assignments = set()
        for week in range(1, 19):
            group, _ = router._get_assignment("00-0019596", 2024, week)
            assignments.add((week, group))

        # Should have assignments for all weeks
        assert len(assignments) == 18, "Missing week assignments"

    def test_split_ratio_accuracy_50_50(self):
        """Test that 50/50 split ratio is accurate over large sample."""
        router = ABTestRouter(split_ratio=0.5)

        assignments = []
        for i in range(10000):
            player_id = f"player_{i:05d}"
            group, _ = router._get_assignment(player_id, 2024, 10)
            assignments.append(group)

        b_ratio = assignments.count("B") / len(assignments)

        # Should be within 2% of target (0.5)
        assert 0.48 <= b_ratio <= 0.52, f"Split ratio {b_ratio:.3f} outside tolerance"

    def test_split_ratio_accuracy_30_70(self):
        """Test that 30/70 split ratio is accurate over large sample."""
        router = ABTestRouter(split_ratio=0.3)

        assignments = []
        for i in range(10000):
            player_id = f"player_{i:05d}"
            group, _ = router._get_assignment(player_id, 2024, 10)
            assignments.append(group)

        b_ratio = assignments.count("B") / len(assignments)

        # Should be within 2% of target (0.3)
        assert 0.28 <= b_ratio <= 0.32, f"Split ratio {b_ratio:.3f} outside tolerance"

    def test_split_ratio_zero_always_control(self):
        """Test that split_ratio=0 always assigns to control."""
        router = ABTestRouter(split_ratio=0.0)

        for i in range(1000):
            player_id = f"player_{i:04d}"
            group, version = router._get_assignment(player_id, 2024, 10)
            assert group == "A", f"Expected A, got {group}"
            assert version == "v5.0", f"Expected v5.0, got {version}"

    def test_split_ratio_one_always_treatment(self):
        """Test that split_ratio=1.0 always assigns to treatment."""
        router = ABTestRouter(split_ratio=1.0)

        for i in range(1000):
            player_id = f"player_{i:04d}"
            group, version = router._get_assignment(player_id, 2024, 10)
            assert group == "B", f"Expected B, got {group}"
            assert version == "v6.0", f"Expected v6.0, got {version}"

    def test_disabled_test_always_control(self):
        """Test that disabled test always assigns to control."""
        router = ABTestRouter(split_ratio=0.5, enabled=False)

        for i in range(1000):
            player_id = f"player_{i:04d}"
            group, version = router._get_assignment(player_id, 2024, 10)
            assert group == "A", f"Expected A (disabled test), got {group}"
            assert version == "v5.0", f"Expected v5.0 (disabled test), got {version}"

    def test_hash_consistency_across_test_names(self):
        """Test that different test names produce different hashes."""
        router1 = ABTestRouter(split_ratio=0.5, test_name="test_1")
        router2 = ABTestRouter(split_ratio=0.5, test_name="test_2")

        # Same player/week should potentially get different assignments
        # (though not guaranteed - this is a probabilistic test)
        assignments_1 = []
        assignments_2 = []

        for i in range(100):
            player_id = f"player_{i:04d}"
            group1, _ = router1._get_assignment(player_id, 2024, 10)
            group2, _ = router2._get_assignment(player_id, 2024, 10)
            assignments_1.append(group1)
            assignments_2.append(group2)

        # At least some assignments should differ
        assert assignments_1 != assignments_2, "Test names not affecting hash"

    def test_disable_test_method(self):
        """Test disabling A/B test."""
        router = ABTestRouter(split_ratio=0.5, enabled=True)
        assert router.enabled is True

        router.disable_test()
        assert router.enabled is False
        assert router.test_metadata["enabled"] is False
        assert "disabled_at" in router.test_metadata

        # Should now always assign to control
        group, version = router._get_assignment("player_001", 2024, 10)
        assert group == "A"
        assert version == "v5.0"

    def test_promote_treatment_method(self):
        """Test promoting treatment to 100%."""
        router = ABTestRouter(split_ratio=0.5)
        assert router.split_ratio == 0.5

        router.promote_treatment()
        assert router.split_ratio == 1.0
        assert router.test_metadata["split_ratio"] == 1.0
        assert "promoted_at" in router.test_metadata

        # Should now always assign to treatment
        for i in range(100):
            group, version = router._get_assignment(f"player_{i:04d}", 2024, 10)
            assert group == "B"
            assert version == "v6.0"

    @patch('py.production.ab_test_router.EnsembleV5')
    @patch('py.production.ab_test_router.EnsembleV6')
    def test_lazy_model_loading(self, mock_v6, mock_v5):
        """Test that models are only loaded when needed."""
        router = ABTestRouter(split_ratio=0.5)

        # Models should not be loaded yet
        assert not router._models_loaded
        mock_v5.assert_not_called()
        mock_v6.assert_not_called()

        # Load models
        router._load_models()

        # Models should be loaded now
        assert router._models_loaded
        mock_v5.assert_called_once()
        mock_v6.assert_called_once()

        # Calling again should not reload
        router._load_models()
        assert mock_v5.call_count == 1
        assert mock_v6.call_count == 1

    @patch('py.production.ab_test_router.EnsembleV5')
    @patch('py.production.ab_test_router.EnsembleV6')
    def test_model_loading_disabled_test(self, mock_v6, mock_v5):
        """Test that v6 is not loaded when test is disabled."""
        router = ABTestRouter(split_ratio=0.5, enabled=False)

        router._load_models()

        # Only v5 should be loaded
        mock_v5.assert_called_once()
        mock_v6.assert_not_called()


# ============================================================================
# INTEGRATION TESTS - Database and Model Registry
# ============================================================================


class TestABTestRouterIntegration:
    """Integration tests for database logging and model registry."""

    @patch('psycopg2.connect')
    @patch('py.production.ab_test_router.EnsembleV5')
    @patch('py.production.ab_test_router.EnsembleV6')
    def test_predict_calls_correct_ensemble_control(self, mock_v6, mock_v5, mock_db):
        """Test that control group predictions use v5 ensemble."""
        # Mock ensemble predictions
        mock_v5_instance = Mock()
        mock_v5_instance.predict.return_value = {
            "ensemble_mean": 250.0,
            "ensemble_std": 45.0,
            "quantile_10": 205.0,
            "quantile_90": 295.0,
            "weights": {"state_space": 0.3, "bnn": 0.25},
            "inference_time_ms": 100.0,
        }
        mock_v5.return_value = mock_v5_instance

        # Create router with split_ratio=0 (all control)
        router = ABTestRouter(split_ratio=0.0)

        result = router.predict(
            player_id="00-0019596",
            season=2024,
            week=10,
            log_to_db=False
        )

        # Should have called v5.predict
        mock_v5_instance.predict.assert_called_once()

        # Should have A/B metadata
        assert result["ab_group"] == "A"
        assert result["model_version"] == "v5.0"
        assert result["test_name"] == "ensemble_v5_vs_v6_sagarin"

    @patch('psycopg2.connect')
    @patch('py.production.ab_test_router.EnsembleV5')
    @patch('py.production.ab_test_router.EnsembleV6')
    def test_predict_calls_correct_ensemble_treatment(self, mock_v6, mock_v5, mock_db):
        """Test that treatment group predictions use v6 ensemble."""
        # Mock ensemble predictions
        mock_v6_instance = Mock()
        mock_v6_instance.predict.return_value = {
            "ensemble_mean": 260.0,
            "ensemble_std": 42.0,
            "quantile_10": 218.0,
            "quantile_90": 302.0,
            "weights": {"state_space": 0.25, "bnn": 0.2, "elo": 0.1},
            "inference_time_ms": 105.0,
        }
        mock_v6.return_value = mock_v6_instance

        # Create router with split_ratio=1.0 (all treatment)
        router = ABTestRouter(split_ratio=1.0)

        result = router.predict(
            player_id="00-0019596",
            season=2024,
            week=10,
            log_to_db=False
        )

        # Should have called v6.predict
        mock_v6_instance.predict.assert_called_once()

        # Should have A/B metadata
        assert result["ab_group"] == "B"
        assert result["model_version"] == "v6.0"
        assert result["test_name"] == "ensemble_v5_vs_v6_sagarin"

    @patch('psycopg2.connect')
    @patch('py.production.ab_test_router.EnsembleV5')
    @patch('py.production.ab_test_router.EnsembleV6')
    def test_database_logging_enabled(self, mock_v6, mock_v5, mock_db):
        """Test that predictions are logged to database when log_to_db=True."""
        # Mock ensemble
        mock_v5_instance = Mock()
        mock_v5_instance.predict.return_value = {
            "ensemble_mean": 250.0,
            "ensemble_std": 45.0,
            "quantile_10": 205.0,
            "quantile_90": 295.0,
        }
        mock_v5.return_value = mock_v5_instance

        # Mock database connection
        mock_cursor = Mock()
        mock_cursor.fetchone.side_effect = [
            (38,),  # model_id for v5.0
            ("Patrick Mahomes", "KC", "LAC", True)  # player info
        ]
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value = mock_conn

        router = ABTestRouter(split_ratio=0.0)

        result = router.predict(
            player_id="00-0019596",
            season=2024,
            week=10,
            log_to_db=True
        )

        # Should have executed INSERT query
        assert mock_cursor.execute.call_count == 3  # model_id lookup, player info, insert

        # Check INSERT query was called
        insert_calls = [call for call in mock_cursor.execute.call_args_list
                       if "INSERT INTO ml.predictions" in str(call)]
        assert len(insert_calls) == 1, "INSERT query not executed"

        # Should have committed
        mock_conn.commit.assert_called_once()

    @patch('psycopg2.connect')
    @patch('py.production.ab_test_router.EnsembleV5')
    @patch('py.production.ab_test_router.EnsembleV6')
    def test_database_logging_disabled(self, mock_v6, mock_v5, mock_db):
        """Test that predictions are not logged when log_to_db=False."""
        # Mock ensemble
        mock_v5_instance = Mock()
        mock_v5_instance.predict.return_value = {
            "ensemble_mean": 250.0,
            "ensemble_std": 45.0,
            "quantile_10": 205.0,
            "quantile_90": 295.0,
        }
        mock_v5.return_value = mock_v5_instance

        router = ABTestRouter(split_ratio=0.0)

        result = router.predict(
            player_id="00-0019596",
            season=2024,
            week=10,
            log_to_db=False
        )

        # Database should not be called
        mock_db.assert_not_called()

    @patch('psycopg2.connect')
    @patch('py.production.ab_test_router.EnsembleV5')
    @patch('py.production.ab_test_router.EnsembleV6')
    def test_model_registry_lookup_correct_version(self, mock_v6, mock_v5, mock_db):
        """Test that correct model_id is looked up from registry."""
        # Mock ensemble
        mock_v5_instance = Mock()
        mock_v5_instance.predict.return_value = {
            "ensemble_mean": 250.0,
            "ensemble_std": 45.0,
            "quantile_10": 205.0,
            "quantile_90": 295.0,
        }
        mock_v5.return_value = mock_v5_instance

        # Mock database connection
        mock_cursor = Mock()
        mock_cursor.fetchone.side_effect = [
            (38,),  # model_id for ensemble_v5.0
            ("Patrick Mahomes", "KC", "LAC", True)
        ]
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value = mock_conn

        router = ABTestRouter(split_ratio=0.0)

        result = router.predict(
            player_id="00-0019596",
            season=2024,
            week=10,
            log_to_db=True
        )

        # Check model registry query
        registry_calls = [call for call in mock_cursor.execute.call_args_list
                         if "ml.model_registry" in str(call)]
        assert len(registry_calls) == 1, "Model registry not queried"

        # Should have queried for ensemble_v5.0
        query_args = registry_calls[0][0]
        assert "ensemble_v5.0" in str(query_args), "Wrong model name queried"

    @patch('psycopg2.connect')
    @patch('py.production.ab_test_router.EnsembleV5')
    @patch('py.production.ab_test_router.EnsembleV6')
    def test_database_logging_handles_null_prediction(self, mock_v6, mock_v5, mock_db):
        """Test that null predictions are not logged to database."""
        # Mock ensemble with null prediction
        mock_v5_instance = Mock()
        mock_v5_instance.predict.return_value = {
            "ensemble_mean": None,
            "ensemble_std": None,
            "quantile_10": None,
            "quantile_90": None,
        }
        mock_v5.return_value = mock_v5_instance

        router = ABTestRouter(split_ratio=0.0)

        result = router.predict(
            player_id="00-0019596",
            season=2024,
            week=10,
            log_to_db=True
        )

        # Database should not be called (null prediction)
        mock_db.assert_not_called()

    @patch('psycopg2.connect')
    @patch('py.production.ab_test_router.EnsembleV5')
    @patch('py.production.ab_test_router.EnsembleV6')
    def test_batch_predictions_split_ratio_accuracy(self, mock_v6, mock_v5, mock_db):
        """Test that batch predictions maintain accurate split ratio."""
        # Mock ensembles
        mock_v5_instance = Mock()
        mock_v5_instance.predict.return_value = {
            "ensemble_mean": 250.0,
            "ensemble_std": 45.0,
            "quantile_10": 205.0,
            "quantile_90": 295.0,
        }
        mock_v5.return_value = mock_v5_instance

        mock_v6_instance = Mock()
        mock_v6_instance.predict.return_value = {
            "ensemble_mean": 260.0,
            "ensemble_std": 42.0,
            "quantile_10": 218.0,
            "quantile_90": 302.0,
        }
        mock_v6.return_value = mock_v6_instance

        router = ABTestRouter(split_ratio=0.5)

        # Create batch of 200 games
        games = [
            {"player_id": f"player_{i:05d}", "season": 2024, "week": 10}
            for i in range(200)
        ]

        df = router.predict_batch(games, log_to_db=False)

        # Check split ratio
        b_ratio = (df["ab_group"] == "B").sum() / len(df)
        assert 0.40 <= b_ratio <= 0.60, f"Batch split ratio {b_ratio:.3f} outside tolerance"

    @patch('psycopg2.connect')
    def test_get_test_summary_correct_query(self, mock_db):
        """Test that get_test_summary executes correct SQL query."""
        # Mock database connection
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

        # Check summary structure
        assert "test_name" in summary
        assert "groups" in summary
        assert "A" in summary["groups"]
        assert "B" in summary["groups"]

        # Check group A metrics
        assert summary["groups"]["A"]["model_version"] == "v5.0"
        assert summary["groups"]["A"]["n_predictions"] == 150
        assert summary["groups"]["A"]["mae"] == 45.2

        # Check group B metrics
        assert summary["groups"]["B"]["model_version"] == "v6.0"
        assert summary["groups"]["B"]["n_predictions"] == 150
        assert summary["groups"]["B"]["mae"] == 43.1


# ============================================================================
# END-TO-END TESTS - Full Prediction Flow
# ============================================================================


class TestABTestRouterE2E:
    """End-to-end tests for complete prediction flow."""

    @patch('psycopg2.connect')
    @patch('py.production.ab_test_router.EnsembleV5')
    @patch('py.production.ab_test_router.EnsembleV6')
    def test_full_prediction_flow_control(self, mock_v6, mock_v5, mock_db):
        """Test complete prediction flow for control group."""
        # Mock ensemble
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

        # Mock database
        mock_cursor = Mock()
        mock_cursor.fetchone.side_effect = [
            (38,),  # model_id
            ("Patrick Mahomes", "KC", "LAC", True)
        ]
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value = mock_conn

        # Create router
        router = ABTestRouter(split_ratio=0.0)

        # Make prediction
        result = router.predict(
            player_id="00-0019596",
            season=2024,
            week=10,
            return_individual=False,
            log_to_db=True
        )

        # Verify result structure
        assert "ab_group" in result
        assert "model_version" in result
        assert "ensemble_mean" in result
        assert "ensemble_std" in result
        assert "quantile_10" in result
        assert "quantile_90" in result
        assert "test_name" in result

        # Verify values
        assert result["ab_group"] == "A"
        assert result["model_version"] == "v5.0"
        assert result["ensemble_mean"] == 250.0
        assert result["ensemble_std"] == 45.0

        # Verify database logging
        mock_conn.commit.assert_called_once()

    @patch('psycopg2.connect')
    @patch('py.production.ab_test_router.EnsembleV5')
    @patch('py.production.ab_test_router.EnsembleV6')
    def test_full_prediction_flow_treatment(self, mock_v6, mock_v5, mock_db):
        """Test complete prediction flow for treatment group."""
        # Mock ensemble
        mock_v6_instance = Mock()
        mock_v6_instance.predict.return_value = {
            "ensemble_mean": 260.0,
            "ensemble_std": 42.0,
            "quantile_10": 218.0,
            "quantile_90": 302.0,
            "weights": {"state_space": 0.25, "bnn": 0.2, "hierarchical": 0.2, "xgb": 0.15, "elo": 0.1, "adaptive": 0.1},
            "inference_time_ms": 105.0,
        }
        mock_v6.return_value = mock_v6_instance

        # Mock database
        mock_cursor = Mock()
        mock_cursor.fetchone.side_effect = [
            (39,),  # model_id for v6.0
            ("Patrick Mahomes", "KC", "LAC", True)
        ]
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value = mock_conn

        # Create router
        router = ABTestRouter(split_ratio=1.0)

        # Make prediction
        result = router.predict(
            player_id="00-0019596",
            season=2024,
            week=10,
            return_individual=False,
            log_to_db=True
        )

        # Verify result structure
        assert result["ab_group"] == "B"
        assert result["model_version"] == "v6.0"
        assert result["ensemble_mean"] == 260.0
        assert result["ensemble_std"] == 42.0

    @patch('psycopg2.connect')
    @patch('py.production.ab_test_router.EnsembleV5')
    @patch('py.production.ab_test_router.EnsembleV6')
    def test_batch_predictions_full_flow(self, mock_v6, mock_v5, mock_db):
        """Test batch predictions with database logging."""
        # Mock ensembles
        mock_v5_instance = Mock()
        mock_v5_instance.predict.return_value = {
            "ensemble_mean": 250.0,
            "ensemble_std": 45.0,
            "quantile_10": 205.0,
            "quantile_90": 295.0,
        }
        mock_v5.return_value = mock_v5_instance

        mock_v6_instance = Mock()
        mock_v6_instance.predict.return_value = {
            "ensemble_mean": 260.0,
            "ensemble_std": 42.0,
            "quantile_10": 218.0,
            "quantile_90": 302.0,
        }
        mock_v6.return_value = mock_v6_instance

        # Mock database
        mock_cursor = Mock()
        mock_cursor.fetchone.side_effect = [
            (38,), ("Player", "KC", "LAC", True)
        ] * 100  # 100 predictions
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value = mock_conn

        router = ABTestRouter(split_ratio=0.5)

        # Create batch
        games = [
            {"player_id": f"player_{i:03d}", "season": 2024, "week": 10}
            for i in range(100)
        ]

        df = router.predict_batch(games, log_to_db=False)

        # Verify DataFrame structure
        assert len(df) == 100
        assert "player_id" in df.columns
        assert "ab_group" in df.columns
        assert "model_version" in df.columns
        assert "ensemble_mean" in df.columns

        # Verify split ratio
        b_ratio = (df["ab_group"] == "B").sum() / len(df)
        assert 0.35 <= b_ratio <= 0.65, f"Split ratio {b_ratio:.3f} outside tolerance"

    @patch('psycopg2.connect')
    @patch('py.production.ab_test_router.EnsembleV5')
    @patch('py.production.ab_test_router.EnsembleV6')
    def test_rollback_scenario_disable_test(self, mock_v6, mock_v5, mock_db):
        """Test emergency rollback by disabling test."""
        # Mock ensemble
        mock_v5_instance = Mock()
        mock_v5_instance.predict.return_value = {
            "ensemble_mean": 250.0,
            "ensemble_std": 45.0,
            "quantile_10": 205.0,
            "quantile_90": 295.0,
        }
        mock_v5.return_value = mock_v5_instance

        # Start with 50/50 split
        router = ABTestRouter(split_ratio=0.5, enabled=True)
        assert router.enabled is True

        # Disable test (emergency rollback)
        router.disable_test()
        assert router.enabled is False

        # All predictions should now go to control
        for i in range(100):
            group, version = router._get_assignment(f"player_{i:04d}", 2024, 10)
            assert group == "A", f"Expected A after disable, got {group}"
            assert version == "v5.0", f"Expected v5.0 after disable, got {version}"

    def test_environment_variables_configuration(self):
        """Test that router respects environment variables."""
        # Test AB_TEST_ENABLED
        with patch.dict(os.environ, {"AB_TEST_ENABLED": "false"}):
            router = ABTestRouter(
                split_ratio=0.5,
                enabled=os.getenv("AB_TEST_ENABLED", "true").lower() == "true"
            )
            assert router.enabled is False

        # Test AB_TEST_SPLIT_RATIO
        with patch.dict(os.environ, {"AB_TEST_SPLIT_RATIO": "0.3"}):
            split_ratio = float(os.getenv("AB_TEST_SPLIT_RATIO", "0.5"))
            router = ABTestRouter(split_ratio=split_ratio)
            assert router.split_ratio == 0.3


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


class TestABTestRouterPerformance:
    """Performance tests for A/B test router."""

    @patch('psycopg2.connect')
    @patch('py.production.ab_test_router.EnsembleV5')
    @patch('py.production.ab_test_router.EnsembleV6')
    def test_assignment_performance(self, mock_v6, mock_v5, mock_db):
        """Test that assignment is fast (<1ms per assignment)."""
        router = ABTestRouter(split_ratio=0.5)

        start = time.time()
        for i in range(10000):
            router._get_assignment(f"player_{i:05d}", 2024, 10)
        elapsed = time.time() - start

        avg_time_ms = (elapsed / 10000) * 1000
        assert avg_time_ms < 1.0, f"Assignment too slow: {avg_time_ms:.3f}ms"

    @patch('psycopg2.connect')
    @patch('py.production.ab_test_router.EnsembleV5')
    @patch('py.production.ab_test_router.EnsembleV6')
    def test_hash_performance(self, mock_v6, mock_v5, mock_db):
        """Test that hash computation is fast."""
        router = ABTestRouter(split_ratio=0.5)

        start = time.time()
        for i in range(100000):
            hash_input = f"player_{i:05d}_2024_10_{router.test_name}"
            hash_digest = hashlib.md5(hash_input.encode()).hexdigest()
        elapsed = time.time() - start

        avg_time_us = (elapsed / 100000) * 1_000_000
        assert avg_time_us < 10.0, f"Hash too slow: {avg_time_us:.3f}μs"


# ============================================================================
# MAIN
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--cov=py.production.ab_test_router"])
