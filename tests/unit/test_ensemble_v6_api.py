#!/usr/bin/env python3
"""
Ensemble v6 API Contract Tests
================================

Ensures API stability for production ensemble.

Author: Claude Code (Sonnet 4.5)
Date: November 15, 2025
"""

import inspect

import pytest


class TestEnsembleV6API:
    """Test suite for Ensemble v6 API contract."""

    def test_canonical_class_exists(self):
        """Test that canonical class can be imported."""
        from py.ensemble.enhanced_ensemble_v6 import EnhancedEnsemble4Way

        assert EnhancedEnsemble4Way is not None

    def test_init_signature(self):
        """Test __init__ signature has required parameters."""
        from py.ensemble.enhanced_ensemble_v6 import EnhancedEnsemble4Way

        sig = inspect.signature(EnhancedEnsemble4Way.__init__)
        params = list(sig.parameters.keys())

        # Required parameters
        assert "self" in params
        assert "state_space_path" in params
        assert "hierarchical_path" in params
        assert "informative_path" in params
        assert "ridge_path" in params  # XGBoost v4 Ridge Residuals
        assert "xgboost_path" in params

        # Optional parameters
        assert "weights" in params
        assert "temperature_scaler_path" in params
        assert "db_host" in params
        assert "db_port" in params
        assert "db_name" in params
        assert "db_user" in params
        assert "db_password" in params

    def test_no_deprecated_params(self):
        """Test that deprecated parameters have been removed."""
        from py.ensemble.enhanced_ensemble_v6 import EnhancedEnsemble4Way

        sig = inspect.signature(EnhancedEnsemble4Way.__init__)
        params = list(sig.parameters.keys())

        # Should NOT have deprecated params from old versions
        assert "bnn_path" not in params  # Removed in v6
        assert "hybrid_xgboost_path" not in params  # Replaced by ridge_path
        assert "use_adaptive_weights" not in params  # Never existed (TypeError source)
        assert "adaptive_weights_path" not in params  # Research version only

    def test_default_weights_structure(self):
        """Test DEFAULT_WEIGHTS has correct structure."""
        from py.ensemble.enhanced_ensemble_v6 import EnhancedEnsemble4Way

        weights = EnhancedEnsemble4Way.DEFAULT_WEIGHTS

        # Should be a dictionary
        assert isinstance(weights, dict)

        # Should have exactly 4 models (4-way ensemble)
        assert len(weights) == 4

        # Should have correct model names
        assert "state_space_v2" in weights
        assert "hierarchical_v2" in weights
        assert "xgboost_v4_ridge" in weights
        assert "informative_priors_v2" in weights

        # Weights should sum to 1.0 (within tolerance)
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01

        # Individual weights should be positive
        for model, weight in weights.items():
            assert weight > 0, f"{model} has non-positive weight: {weight}"

    def test_predict_method_exists(self):
        """Test that predict method exists with correct signature."""
        from py.ensemble.enhanced_ensemble_v6 import EnhancedEnsemble4Way

        assert hasattr(EnhancedEnsemble4Way, "predict")

        sig = inspect.signature(EnhancedEnsemble4Way.predict)
        params = list(sig.parameters.keys())

        # Required parameters
        assert "self" in params
        assert "player_id" in params
        assert "season" in params
        assert "week" in params

    def test_return_type_annotations(self):
        """Test that predict method has return type annotation."""
        from py.ensemble.enhanced_ensemble_v6 import EnhancedEnsemble4Way

        sig = inspect.signature(EnhancedEnsemble4Way.predict)

        # Should have return annotation
        assert sig.return_annotation != inspect.Signature.empty

        # Return type should be Dict
        return_type = sig.return_annotation
        # Handle both typing.Dict and dict (Python 3.9+)
        assert "Dict" in str(return_type) or "dict" in str(return_type)


class TestEnsembleV6ELO_API:
    """Test suite for ELO research version API."""

    def test_elo_version_exists(self):
        """Test that ELO research version can be imported."""
        from py.ensemble.enhanced_ensemble_v6_elo import EnhancedEnsemble6Way

        assert EnhancedEnsemble6Way is not None

    def test_elo_has_adaptive_weights_param(self):
        """Test that ELO version has adaptive_weights_path parameter."""
        from py.ensemble.enhanced_ensemble_v6_elo import EnhancedEnsemble6Way

        sig = inspect.signature(EnhancedEnsemble6Way.__init__)
        params = list(sig.parameters.keys())

        # ELO version should have these extra params
        assert "adaptive_weights_path" in params
        assert "elo_ratings_path" in params

    def test_elo_default_weights_has_5_models(self):
        """Test that ELO version has 5 models (including ELO)."""
        from py.ensemble.enhanced_ensemble_v6_elo import EnhancedEnsemble6Way

        weights = EnhancedEnsemble6Way.DEFAULT_WEIGHTS

        # Should have exactly 5 models (5-way ensemble)
        assert len(weights) == 5

        # Should include ELO
        assert "elo" in weights


class TestAPIDocumentation:
    """Test that API is properly documented."""

    def test_canonical_has_docstring(self):
        """Test that canonical class has docstring."""
        from py.ensemble.enhanced_ensemble_v6 import EnhancedEnsemble4Way

        assert EnhancedEnsemble4Way.__doc__ is not None
        assert len(EnhancedEnsemble4Way.__doc__.strip()) > 0

    def test_predict_has_docstring(self):
        """Test that predict method has docstring."""
        from py.ensemble.enhanced_ensemble_v6 import EnhancedEnsemble4Way

        assert EnhancedEnsemble4Way.predict.__doc__ is not None
        assert len(EnhancedEnsemble4Way.predict.__doc__.strip()) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
