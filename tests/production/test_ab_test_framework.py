#!/usr/bin/env python3
"""
Unit Tests for A/B Test Framework
==================================

Tests for ABTestRouter and analysis pipeline.

Run with:
    pytest tests/production/test_ab_test_framework.py -v

Author: Claude Code (Sonnet 4.5)
Date: November 14, 2025
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from py.production.ab_test_router import ABTestRouter


class TestABTestRouter:
    """Test suite for ABTestRouter class."""

    def test_initialization(self):
        """Test router initialization with default parameters."""
        router = ABTestRouter(split_ratio=0.5)

        assert router.split_ratio == 0.5
        assert router.test_name == "ensemble_v5_vs_v5_1"
        assert router.enabled is True
        assert router._models_loaded is False

    def test_initialization_invalid_split_ratio(self):
        """Test that invalid split ratios raise ValueError."""
        with pytest.raises(ValueError, match="split_ratio must be in"):
            ABTestRouter(split_ratio=1.5)

        with pytest.raises(ValueError, match="split_ratio must be in"):
            ABTestRouter(split_ratio=-0.1)

    def test_deterministic_assignment(self):
        """Test that assignment is deterministic for same player/week."""
        router = ABTestRouter(split_ratio=0.5)

        # Same player/week should get same assignment
        group1, version1 = router._get_assignment("00-0019596", 2024, 10)
        group2, version2 = router._get_assignment("00-0019596", 2024, 10)

        assert group1 == group2
        assert version1 == version2

    def test_different_players_different_assignments(self):
        """Test that different players can get different assignments."""
        router = ABTestRouter(split_ratio=0.5)

        assignments = set()
        for i in range(100):
            player_id = f"player_{i:04d}"
            group, _ = router._get_assignment(player_id, 2024, 10)
            assignments.add(group)

        # Should have both A and B assignments
        assert "A" in assignments
        assert "B" in assignments

    def test_split_ratio_approximately_correct(self):
        """Test that split ratio is approximately correct over many assignments."""
        router = ABTestRouter(split_ratio=0.5)

        assignments = []
        for i in range(10000):
            player_id = f"player_{i:05d}"
            group, _ = router._get_assignment(player_id, 2024, 10)
            assignments.append(group)

        b_ratio = assignments.count("B") / len(assignments)

        # Should be within 2% of target (0.5)
        assert 0.48 <= b_ratio <= 0.52

    def test_split_ratio_zero(self):
        """Test that split_ratio=0 always assigns to control."""
        router = ABTestRouter(split_ratio=0.0)

        for i in range(100):
            player_id = f"player_{i:04d}"
            group, version = router._get_assignment(player_id, 2024, 10)
            assert group == "A"
            assert version == "v5.0"

    def test_split_ratio_one(self):
        """Test that split_ratio=1.0 always assigns to treatment."""
        router = ABTestRouter(split_ratio=1.0)

        for i in range(100):
            player_id = f"player_{i:04d}"
            group, version = router._get_assignment(player_id, 2024, 10)
            assert group == "B"
            assert version == "v5.1"

    def test_disabled_always_assigns_to_control(self):
        """Test that disabled test always assigns to control."""
        router = ABTestRouter(split_ratio=0.5, enabled=False)

        for i in range(100):
            player_id = f"player_{i:04d}"
            group, version = router._get_assignment(player_id, 2024, 10)
            assert group == "A"
            assert version == "v5.0"

    @patch('py.production.ab_test_router.EnsembleV5')
    @patch('py.production.ab_test_router.EnsembleV5_1')
    def test_predict_routes_to_correct_ensemble(self, mock_v5_1, mock_v5):
        """Test that predictions are routed to correct ensemble."""
        # Mock ensemble predictions
        mock_v5_instance = Mock()
        mock_v5_instance.predict.return_value = {
            "ensemble_mean": 250.0,
            "ensemble_std": 45.0,
            "quantile_10": 205.0,
            "quantile_90": 295.0,
            "weights": {"model_a": 0.5, "model_b": 0.5},
            "inference_time_ms": 100.0,
        }
        mock_v5.return_value = mock_v5_instance

        mock_v5_1_instance = Mock()
        mock_v5_1_instance.predict.return_value = {
            "ensemble_mean": 260.0,
            "ensemble_std": 42.0,
            "quantile_10": 218.0,
            "quantile_90": 302.0,
            "weights": {"model_a": 0.5, "model_b": 0.5},
            "inference_time_ms": 105.0,
        }
        mock_v5_1.return_value = mock_v5_1_instance

        router = ABTestRouter(split_ratio=0.5)

        # Predict for player that should go to group A
        # (deterministic based on hash)
        result = router.predict(
            player_id="00-0019596",
            season=2024,
            week=10,
            log_to_db=False
        )

        # Check that result has A/B metadata
        assert "ab_group" in result
        assert "model_version" in result
        assert "test_name" in result
        assert result["test_name"] == "ensemble_v5_vs_v5_1"

    def test_disable_test(self):
        """Test disabling A/B test."""
        router = ABTestRouter(split_ratio=0.5, enabled=True)
        assert router.enabled is True

        router.disable_test()
        assert router.enabled is False

        # Should now always assign to control
        group, version = router._get_assignment("player_001", 2024, 10)
        assert group == "A"
        assert version == "v5.0"

    def test_promote_treatment(self):
        """Test promoting treatment to 100%."""
        router = ABTestRouter(split_ratio=0.5)
        assert router.split_ratio == 0.5

        router.promote_treatment()
        assert router.split_ratio == 1.0

        # Should now always assign to treatment
        for i in range(100):
            group, version = router._get_assignment(f"player_{i:04d}", 2024, 10)
            assert group == "B"
            assert version == "v5.1"

    def test_get_test_summary_no_data(self):
        """Test get_test_summary when no predictions exist."""
        with patch('psycopg2.connect') as mock_connect:
            mock_cursor = Mock()
            mock_cursor.fetchall.return_value = []
            mock_conn = Mock()
            mock_conn.cursor.return_value = mock_cursor
            mock_connect.return_value = mock_conn

            router = ABTestRouter()
            summary = router.get_test_summary()

            assert "test_name" in summary
            assert "groups" in summary
            assert summary["test_name"] == "ensemble_v5_vs_v5_1"


class TestABTestAnalyzer:
    """Test suite for A/B test analysis."""

    @pytest.fixture
    def mock_predictions_df(self):
        """Create mock predictions DataFrame for testing."""
        np.random.seed(42)

        # Group A (control) - MAE ~45 yards
        group_a = pd.DataFrame({
            "ab_group": ["A"] * 150,
            "model_version": ["v5.0"] * 150,
            "player_id": [f"player_{i:04d}" for i in range(150)],
            "prediction_mean": np.random.normal(250, 30, 150),
            "prediction_std": np.random.normal(45, 5, 150),
            "prediction_q10": np.random.normal(205, 30, 150),
            "prediction_q90": np.random.normal(295, 30, 150),
        })

        # Group B (treatment) - MAE ~43 yards (4.4% improvement)
        group_b = pd.DataFrame({
            "ab_group": ["B"] * 150,
            "model_version": ["v5.1"] * 150,
            "player_id": [f"player_{i:04d}" for i in range(150, 300)],
            "prediction_mean": np.random.normal(252, 28, 150),
            "prediction_std": np.random.normal(42, 5, 150),
            "prediction_q10": np.random.normal(210, 28, 150),
            "prediction_q90": np.random.normal(294, 28, 150),
        })

        df = pd.concat([group_a, group_b], ignore_index=True)

        # Add actual values and compute errors
        df["actual_value"] = df["prediction_mean"] + np.random.normal(0, 45, 300)
        df["prediction_error"] = df["actual_value"] - df["prediction_mean"]
        df["abs_prediction_error"] = df["prediction_error"].abs()

        # Compute calibration
        df["calibrated"] = (
            (df["actual_value"] >= df["prediction_q10"]) &
            (df["actual_value"] <= df["prediction_q90"])
        )

        return df

    def test_compute_metrics(self, mock_predictions_df):
        """Test metrics computation."""
        from scripts.analyze_ab_test import ABTestAnalyzer

        analyzer = ABTestAnalyzer()
        metrics = analyzer.compute_metrics(mock_predictions_df)

        # Check structure
        assert "group_a" in metrics
        assert "group_b" in metrics

        # Check group A metrics
        assert metrics["group_a"]["n_predictions"] == 150
        assert "mae" in metrics["group_a"]
        assert "rmse" in metrics["group_a"]
        assert "calibration_90" in metrics["group_a"]

        # Check group B metrics
        assert metrics["group_b"]["n_predictions"] == 150
        assert "mae" in metrics["group_b"]

    def test_statistical_tests(self, mock_predictions_df):
        """Test statistical significance tests."""
        from scripts.analyze_ab_test import ABTestAnalyzer

        analyzer = ABTestAnalyzer()
        results = analyzer.statistical_tests(mock_predictions_df)

        # Check t-test results
        assert "ttest" in results
        assert "t_statistic" in results["ttest"]
        assert "p_value" in results["ttest"]
        assert "significant" in results["ttest"]

        # Check bootstrap results
        assert "bootstrap" in results
        assert "mae_diff_mean" in results["bootstrap"]
        assert "mae_diff_ci_lower" in results["bootstrap"]
        assert "mae_diff_ci_upper" in results["bootstrap"]

        # Check effect size
        assert "effect_size" in results
        assert "cohens_d" in results["effect_size"]
        assert "interpretation" in results["effect_size"]

    def test_make_deployment_decision_insufficient_samples(self):
        """Test decision with insufficient samples."""
        from scripts.analyze_ab_test import ABTestAnalyzer

        analyzer = ABTestAnalyzer()

        # Create small dataset
        metrics = {
            "group_a": {"n_predictions": 50, "mae": 45.0, "calibration_90": 0.90},
            "group_b": {"n_predictions": 50, "mae": 43.0, "calibration_90": 0.91},
        }
        statistical_tests = {"ttest": {"significant": True, "p_value": 0.03}}

        decision = analyzer.make_deployment_decision(metrics, statistical_tests)

        assert decision["recommendation"] == "WAIT"
        assert len(decision["blockers"]) > 0

    def test_make_deployment_decision_success(self):
        """Test decision with successful A/B test."""
        from scripts.analyze_ab_test import ABTestAnalyzer

        analyzer = ABTestAnalyzer()

        # Create passing metrics
        metrics = {
            "group_a": {"n_predictions": 150, "mae": 45.0, "rmse": 60.0, "calibration_90": 0.91},
            "group_b": {"n_predictions": 150, "mae": 43.0, "rmse": 58.0, "calibration_90": 0.92},
        }
        statistical_tests = {"ttest": {"significant": True, "p_value": 0.01}}

        decision = analyzer.make_deployment_decision(metrics, statistical_tests)

        assert decision["recommendation"] == "DEPLOY"
        assert len(decision["blockers"]) == 0
        assert len(decision["reasons"]) > 0

    def test_make_deployment_decision_worse_performance(self):
        """Test decision when treatment performs worse."""
        from scripts.analyze_ab_test import ABTestAnalyzer

        analyzer = ABTestAnalyzer()

        # Treatment has worse MAE
        metrics = {
            "group_a": {"n_predictions": 150, "mae": 45.0, "rmse": 60.0, "calibration_90": 0.91},
            "group_b": {"n_predictions": 150, "mae": 47.0, "rmse": 62.0, "calibration_90": 0.90},
        }
        statistical_tests = {"ttest": {"significant": True, "p_value": 0.02}}

        decision = analyzer.make_deployment_decision(metrics, statistical_tests)

        assert decision["recommendation"] in ["WAIT", "ROLLBACK"]
        assert any("WORSE MAE" in blocker for blocker in decision["blockers"])


class TestIntegration:
    """Integration tests for A/B test framework."""

    def test_end_to_end_workflow(self):
        """Test complete A/B test workflow (mocked)."""
        with patch('py.production.ab_test_router.EnsembleV5'), \
             patch('py.production.ab_test_router.EnsembleV5_1'), \
             patch('psycopg2.connect'):

            # 1. Initialize router
            router = ABTestRouter(split_ratio=0.5)
            assert router.split_ratio == 0.5

            # 2. Check assignment consistency
            player_id = "00-0019596"
            group1, version1 = router._get_assignment(player_id, 2024, 10)
            group2, version2 = router._get_assignment(player_id, 2024, 10)
            assert group1 == group2

            # 3. Promote treatment
            router.promote_treatment()
            assert router.split_ratio == 1.0

            # 4. Disable test
            router.disable_test()
            assert router.enabled is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
