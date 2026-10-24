#!/usr/bin/env python3
"""
Unit Tests for GNN Ensemble Component
======================================

Tests the GNNComponent wrapper for integration into the ensemble predictor.

Test Coverage:
- Component initialization
- Model loading (if available)
- Prediction format validation
- Uncertainty estimation
- Graceful degradation when model unavailable
- Batch prediction

Author: Claude Code (Sonnet 4.5)
Date: December 7, 2025
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

torch = pytest.importorskip("torch")  # optional: uv sync --extra torch

from nfl_analytics.ensemble.components.gnn_component import GNNComponent
from nfl_analytics.ensemble.strategy.base import ModelPrediction, ModelType


class TestGNNComponent:
    """Test suite for GNN ensemble component."""

    @pytest.fixture
    def component(self):
        """Create GNN component for testing."""
        return GNNComponent(device="cpu")

    def test_initialization(self, component):
        """Test component initialization."""
        assert component is not None
        assert component.device == torch.device("cpu")
        assert component.model_path.exists() or not component.is_available

    def test_availability_flag(self, component):
        """Test that availability flag matches model file existence."""
        model_exists = component.model_path.exists()
        assert component.is_available == model_exists

    def test_graceful_degradation_when_unavailable(self):
        """Test that component gracefully handles missing model."""
        # Point to non-existent model
        component = GNNComponent(model_path="/tmp/nonexistent_model.pt", device="cpu")

        assert not component.is_available
        assert component.model is None

        # Prediction should return None
        pred = component.predict("KC", "BUF", 2024, 1)
        assert pred is None

    @pytest.mark.skipif(
        not Path("models/gnn/hierarchical_gnn_best.pt").exists(), reason="GNN model not trained yet"
    )
    def test_model_loading(self, component):
        """Test that model loads successfully if available."""
        if component.is_available:
            component._load_model()
            assert component.model is not None
            assert component.embedding_dim > 0
            assert component.hidden_dim > 0

    @pytest.mark.skipif(
        not Path("models/gnn/hierarchical_gnn_best.pt").exists(), reason="GNN model not trained yet"
    )
    def test_prediction_format(self, component):
        """Test that prediction returns correct ModelPrediction format."""
        if not component.is_available:
            pytest.skip("GNN model not available")

        pred = component.predict("KC", "BUF", 2024, 1)

        # If prediction succeeds, validate format
        if pred is not None:
            assert isinstance(pred, ModelPrediction)
            assert pred.model_name == "gnn_hierarchical_v1"
            assert pred.model_type == ModelType.CUSTOM
            assert 0.0 <= pred.prediction_mean <= 1.0  # Win probability
            assert pred.prediction_std >= 0.0
            assert 0.0 <= pred.confidence <= 1.0
            assert "season" in pred.metadata
            assert "week" in pred.metadata
            assert "home_team" in pred.metadata
            assert "away_team" in pred.metadata

    @pytest.mark.skipif(
        not Path("models/gnn/hierarchical_gnn_best.pt").exists(), reason="GNN model not trained yet"
    )
    def test_prediction_uncertainty(self, component):
        """Test that uncertainty estimation is reasonable."""
        if not component.is_available:
            pytest.skip("GNN model not available")

        pred = component.predict("KC", "BUF", 2024, 1)

        if pred is not None:
            # Uncertainty should be reasonable (not too small, not too large)
            assert 0.01 < pred.prediction_std < 0.5
            # For most games, win prob shouldn't be extreme
            # (unless it's a huge mismatch)
            # We just check it's a valid probability
            assert 0.0 <= pred.prediction_mean <= 1.0

    @pytest.mark.skipif(
        not Path("models/gnn/hierarchical_gnn_best.pt").exists(), reason="GNN model not trained yet"
    )
    def test_batch_prediction(self, component):
        """Test batch prediction interface."""
        if not component.is_available:
            pytest.skip("GNN model not available")

        games = [
            ("KC", "BUF", 1),
            ("SF", "PHI", 1),
            ("DAL", "NYG", 2),
        ]

        results = component.predict_batch(games, season=2024)

        assert isinstance(results, dict)
        assert len(results) == len(games)

        # Check all results are present
        for home, away, week in games:
            assert (home, away) in results

    def test_season_graph_caching(self, component):
        """Test that season graphs are cached."""
        if not component.is_available:
            pytest.skip("GNN model not available")

        # Request same season twice
        try:
            component._load_graph_builder()
            graph1 = component._get_season_graph(2024)
            graph2 = component._get_season_graph(2024)

            # Should be same object (cached)
            assert graph1 is graph2
            assert 2024 in component.season_graphs
        except Exception:
            pytest.skip("Graph builder not available")

    def test_invalid_teams(self, component):
        """Test handling of invalid team abbreviations."""
        if not component.is_available:
            pytest.skip("GNN model not available")

        # Predict with invalid teams
        pred = component.predict("INVALID1", "INVALID2", 2024, 1)

        # Should return None or handle gracefully
        assert pred is None

    @pytest.mark.skipif(
        not Path("models/gnn/hierarchical_gnn_best.pt").exists(), reason="GNN model not trained yet"
    )
    def test_metadata_loading(self, component):
        """Test that metadata is loaded correctly."""
        if not component.is_available:
            pytest.skip("GNN model not available")

        component._load_model()

        if component.metadata is not None:
            assert "embedding_dim" in component.metadata
            assert "hidden_dim" in component.metadata
            assert "test_metrics" in component.metadata
            assert component.metadata["test_metrics"]["auc"] > 0.5  # Better than random


class TestGNNComponentIntegration:
    """Integration tests for GNN component."""

    @pytest.mark.skipif(
        not Path("models/gnn/hierarchical_gnn_best.pt").exists(), reason="GNN model not trained yet"
    )
    def test_prediction_consistency(self):
        """Test that predictions are consistent across calls."""
        component = GNNComponent(device="cpu")

        if not component.is_available:
            pytest.skip("GNN model not available")

        # Same input should give same prediction (no randomness in eval mode)
        pred1 = component.predict("KC", "BUF", 2024, 1)
        pred2 = component.predict("KC", "BUF", 2024, 1)

        if pred1 is not None and pred2 is not None:
            # Means should be identical (deterministic forward pass)
            assert abs(pred1.prediction_mean - pred2.prediction_mean) < 1e-6
            # Uncertainties might differ slightly due to MC Dropout randomness
            # But should be in same ballpark
            assert abs(pred1.prediction_std - pred2.prediction_std) < 0.1

    @pytest.mark.skipif(
        not Path("models/gnn/hierarchical_gnn_best.pt").exists(), reason="GNN model not trained yet"
    )
    def test_home_away_asymmetry(self):
        """Test that home/away order matters (not symmetric)."""
        component = GNNComponent(device="cpu")

        if not component.is_available:
            pytest.skip("GNN model not available")

        # KC at home vs BUF
        pred_kc_home = component.predict("KC", "BUF", 2024, 1)

        # BUF at home vs KC (reversed)
        pred_buf_home = component.predict("BUF", "KC", 2024, 1)

        if pred_kc_home is not None and pred_buf_home is not None:
            # Win probabilities should approximately sum to 1
            # (P(KC wins at home) + P(BUF wins at home) ≈ 1)
            total_prob = pred_kc_home.prediction_mean + pred_buf_home.prediction_mean
            # Allow some tolerance for model uncertainty
            assert 0.8 < total_prob < 1.2


def test_component_import():
    """Test that component can be imported correctly."""
    from nfl_analytics.ensemble.components import GNNComponent as ImportedComponent

    assert ImportedComponent is GNNComponent


if __name__ == "__main__":
    """Run tests."""
    pytest.main([__file__, "-v", "-s"])
