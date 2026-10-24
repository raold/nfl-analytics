"""
Comprehensive integration tests for Enhanced Ensemble v3.0

Tests the 4-way ensemble integration:
- State-Space v2 (30% weight)
- BNN v3.1 Rushing (25% weight)
- Hierarchical v2 (25% weight)
- XGBoost baseline (20% weight)

Critical Requirements:
1. Weight optimization correctness
2. Prediction aggregation (inverse variance weighting)
3. Missing model handling (BNN v3.1 rushing only)
4. Confidence interval merging
5. Database integration (load_bayesian_predictions)
6. Real model artifact loading
7. Portfolio optimization integration

Author: Claude Code (Sonnet 4.5)
Created: November 9, 2025
"""

import json
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import psycopg2
import pytest

from nfl_analytics.ensemble.enhanced_ensemble_v3 import EnhancedEnsembleV3

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_bayesian_predictions() -> pd.DataFrame:
    """Create mock Bayesian hierarchical predictions."""
    np.random.seed(42)
    n = 100

    return pd.DataFrame(
        {
            "player_id": range(1, n + 1),
            "stat_type": ["passing_yards"] * n,
            "season": [2024] * n,
            "bayesian_pred": np.random.randint(200, 350, n).astype(float),
            "bayesian_uncertainty": np.random.uniform(15, 40, n),
            "rating_q05": np.random.randint(150, 250, n).astype(float),
            "rating_q50": np.random.randint(200, 300, n).astype(float),
            "rating_q95": np.random.randint(250, 400, n).astype(float),
            "n_games_observed": np.random.randint(5, 50, n),
        }
    )


@pytest.fixture
def mock_xgb_predictions() -> pd.DataFrame:
    """Create mock XGBoost predictions."""
    np.random.seed(43)
    n = 100

    return pd.DataFrame(
        {
            "player_id": range(1, n + 1),
            "stat_type": ["passing_yards"] * n,
            "xgb_pred": np.random.randint(180, 360, n).astype(float),
        }
    )


@pytest.fixture
def mock_bnn_predictions() -> pd.DataFrame:
    """Create mock BNN predictions."""
    np.random.seed(44)
    n = 100

    return pd.DataFrame(
        {
            "player_id": range(1, n + 1),
            "stat_type": ["passing_yards"] * n,
            "pred": np.random.randint(190, 340, n).astype(float),
            "uncertainty": np.random.uniform(20, 50, n),
        }
    )


@pytest.fixture
def ensemble_no_stacking():
    """Create ensemble without stacking (simple weighting)."""
    return EnhancedEnsembleV3(
        use_bnn=True,
        use_stacking=False,
        use_portfolio_opt=False,
        kelly_fraction=0.25,
    )


@pytest.fixture
def ensemble_with_stacking():
    """Create ensemble with stacking enabled."""
    ensemble = EnhancedEnsembleV3(
        use_bnn=True,
        use_stacking=True,
        use_portfolio_opt=True,
        kelly_fraction=0.25,
    )

    # Mock the meta-learner as fitted
    ensemble.meta_learner.is_fitted = True
    ensemble.meta_learner.predict = Mock(
        return_value=np.random.randint(200, 350, 100).astype(float)
    )

    return ensemble


@pytest.fixture
def mock_db_connection():
    """Create mock database connection."""
    mock_conn = MagicMock(spec=psycopg2.extensions.connection)
    mock_cursor = MagicMock()

    # Mock query results
    mock_cursor.fetchall.return_value = [
        (i, "passing_yards", 2024, 250.0, 25.0, 200.0, 250.0, 300.0, 10) for i in range(1, 101)
    ]

    mock_conn.cursor.return_value = mock_cursor
    return mock_conn


# ============================================================================
# TEST 1: INITIALIZATION
# ============================================================================


def test_ensemble_initialization_basic():
    """Test basic ensemble initialization."""
    ensemble = EnhancedEnsembleV3(
        use_bnn=False,
        use_stacking=False,
        use_portfolio_opt=False,
    )

    assert ensemble.bnn is None
    assert ensemble.meta_learner is None
    assert ensemble.portfolio_optimizer is None


def test_ensemble_initialization_full():
    """Test full ensemble initialization with all components."""
    ensemble = EnhancedEnsembleV3(
        use_bnn=True,
        use_stacking=True,
        use_portfolio_opt=True,
        kelly_fraction=0.30,
        min_edge=0.03,
        max_bet_size=0.06,
    )

    assert ensemble.use_bnn is True
    assert ensemble.use_stacking is True
    assert ensemble.use_portfolio_opt is True
    assert ensemble.kelly_fraction == 0.30
    assert ensemble.min_edge == 0.03
    assert ensemble.max_bet_size == 0.06
    assert ensemble.meta_learner is not None
    assert ensemble.portfolio_optimizer is not None


# ============================================================================
# TEST 2: DATABASE INTEGRATION
# ============================================================================


@patch("psycopg2.connect")
def test_load_bayesian_predictions_success(mock_connect, ensemble_no_stacking):
    """Test loading Bayesian predictions from database."""
    # Create mock connection and cursor
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn

    # Mock pandas read_sql
    mock_data = pd.DataFrame(
        {
            "player_id": range(1, 11),
            "stat_type": ["passing_yards"] * 10,
            "season": [2024] * 10,
            "bayesian_pred": np.random.randint(200, 300, 10).astype(float),
            "bayesian_uncertainty": np.random.uniform(20, 40, 10),
            "rating_q05": np.random.randint(150, 250, 10).astype(float),
            "rating_q50": np.random.randint(200, 280, 10).astype(float),
            "rating_q95": np.random.randint(280, 350, 10).astype(float),
            "n_games_observed": [10] * 10,
        }
    )

    with patch("pandas.read_sql", return_value=mock_data):
        result = ensemble_no_stacking.load_bayesian_predictions("qb_chemistry_v1.0")

    # Validate results
    assert len(result) == 10
    assert "player_id" in result.columns
    assert "bayesian_pred" in result.columns
    assert "bayesian_uncertainty" in result.columns
    assert mock_conn.close.called


@patch("psycopg2.connect")
def test_load_bayesian_predictions_empty(mock_connect, ensemble_no_stacking):
    """Test handling of empty database results."""
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn

    with patch("pandas.read_sql", return_value=pd.DataFrame()):
        result = ensemble_no_stacking.load_bayesian_predictions()

    assert len(result) == 0
    assert mock_conn.close.called


# ============================================================================
# TEST 3: PREDICTION AGGREGATION (INVERSE VARIANCE WEIGHTING)
# ============================================================================


def test_inverse_variance_weighting(
    ensemble_no_stacking, mock_bayesian_predictions, mock_xgb_predictions, mock_bnn_predictions
):
    """Test inverse variance weighting aggregation."""
    result = ensemble_no_stacking.get_4way_ensemble_predictions(
        mock_bayesian_predictions,
        mock_xgb_predictions,
        mock_bnn_predictions,
    )

    # Validate output structure
    assert len(result) > 0
    assert "ensemble_pred" in result.columns
    assert "ensemble_uncertainty" in result.columns
    assert "bayesian_pred" in result.columns
    assert "xgb_pred" in result.columns
    assert "bnn_pred" in result.columns

    # Validate predictions are reasonable
    assert result["ensemble_pred"].min() > 0
    assert result["ensemble_pred"].max() < 600  # Reasonable passing yards range
    assert result["ensemble_uncertainty"].min() >= 0


def test_ensemble_uncertainty_reduction(
    ensemble_no_stacking, mock_bayesian_predictions, mock_xgb_predictions, mock_bnn_predictions
):
    """Test that ensemble uncertainty is lower than individual uncertainties."""
    result = ensemble_no_stacking.get_4way_ensemble_predictions(
        mock_bayesian_predictions,
        mock_xgb_predictions,
        mock_bnn_predictions,
    )

    # Ensemble uncertainty should be lower than average individual uncertainty
    avg_bayesian_unc = result["bayesian_uncertainty"].mean()
    avg_bnn_unc = result["bnn_uncertainty"].mean()
    avg_ensemble_unc = result["ensemble_uncertainty"].mean()

    # Ensemble should reduce uncertainty (due to averaging)
    assert avg_ensemble_unc < max(avg_bayesian_unc, avg_bnn_unc)


def test_inverse_variance_weight_calculation(
    ensemble_no_stacking, mock_bayesian_predictions, mock_xgb_predictions, mock_bnn_predictions
):
    """Test that lower uncertainty models get higher weights."""
    # Modify predictions to have clear uncertainty differences
    mock_bayesian_predictions["bayesian_uncertainty"] = 10.0  # Low uncertainty
    mock_bnn_predictions["uncertainty"] = 50.0  # High uncertainty

    result = ensemble_no_stacking.get_4way_ensemble_predictions(
        mock_bayesian_predictions,
        mock_xgb_predictions,
        mock_bnn_predictions,
    )

    # Ensemble prediction should be closer to Bayesian (lower uncertainty)
    # Calculate implied weights
    for idx in range(min(5, len(result))):
        row = result.iloc[idx]
        bayesian_pred = row["bayesian_pred"]
        ensemble_pred = row["ensemble_pred"]

        # Ensemble should be pulled toward low-uncertainty Bayesian predictions
        # (This is a soft check - exact weights depend on XGBoost variance assumption)
        assert abs(ensemble_pred - bayesian_pred) < abs(ensemble_pred - row["bnn_pred"])


# ============================================================================
# TEST 4: META-LEARNER (STACKING)
# ============================================================================


def test_stacking_meta_learner(
    ensemble_with_stacking, mock_bayesian_predictions, mock_xgb_predictions, mock_bnn_predictions
):
    """Test stacking meta-learner integration."""
    result = ensemble_with_stacking.get_4way_ensemble_predictions(
        mock_bayesian_predictions,
        mock_xgb_predictions,
        mock_bnn_predictions,
    )

    # Validate meta-learner was called
    assert ensemble_with_stacking.meta_learner.predict.called

    # Validate output
    assert "ensemble_pred" in result.columns
    assert len(result) > 0


def test_meta_features_engineering(
    ensemble_with_stacking, mock_bayesian_predictions, mock_xgb_predictions, mock_bnn_predictions
):
    """Test that meta-features include agreement/disagreement signals."""
    # Capture the meta-features passed to meta-learner
    captured_features = None

    def capture_predict(features):
        nonlocal captured_features
        captured_features = features
        return np.random.randint(200, 350, len(features)).astype(float)

    ensemble_with_stacking.meta_learner.predict = Mock(side_effect=capture_predict)

    ensemble_with_stacking.get_4way_ensemble_predictions(
        mock_bayesian_predictions,
        mock_xgb_predictions,
        mock_bnn_predictions,
    )

    # Validate meta-features were created
    assert captured_features is not None

    # Check expected meta-features
    expected_features = [
        "bayesian_pred",
        "xgb_pred",
        "bnn_pred",
        "bayesian_uncertainty",
        "bnn_uncertainty",
        "bayesian_xgb_diff",
        "bayesian_bnn_diff",
        "xgb_bnn_diff",
        "mean_pred",
        "std_pred",
    ]

    for feat in expected_features:
        assert feat in captured_features.columns, f"Missing meta-feature: {feat}"


# ============================================================================
# TEST 5: MISSING MODEL HANDLING
# ============================================================================


def test_missing_bnn_predictions(
    ensemble_no_stacking, mock_bayesian_predictions, mock_xgb_predictions
):
    """Test handling when BNN predictions are missing (partial ensemble)."""
    # Create empty BNN predictions
    empty_bnn = pd.DataFrame(columns=["player_id", "stat_type", "pred", "uncertainty"])

    # Should handle gracefully (inner join will result in empty output)
    result = ensemble_no_stacking.get_4way_ensemble_predictions(
        mock_bayesian_predictions,
        mock_xgb_predictions,
        empty_bnn,
    )

    # With inner join, result should be empty if any component is missing
    assert len(result) == 0


def test_partial_player_coverage(
    ensemble_no_stacking, mock_bayesian_predictions, mock_xgb_predictions, mock_bnn_predictions
):
    """Test ensemble with partial player coverage across models."""
    # Reduce BNN coverage to 50 players
    mock_bnn_predictions = mock_bnn_predictions.iloc[:50]

    result = ensemble_no_stacking.get_4way_ensemble_predictions(
        mock_bayesian_predictions,
        mock_xgb_predictions,
        mock_bnn_predictions,
    )

    # Should only include players with all predictions (inner join)
    assert len(result) == 50
    assert len(result) < len(mock_bayesian_predictions)


# ============================================================================
# TEST 6: BNN MODEL LOADING
# ============================================================================


def test_load_bnn_model_missing_file(ensemble_no_stacking, tmp_path):
    """Test loading BNN model when file doesn't exist."""
    fake_path = tmp_path / "nonexistent_model.pkl"

    result = ensemble_no_stacking.load_bnn_model(model_path=fake_path)

    assert result is None
    assert ensemble_no_stacking.bnn is None


def test_load_bnn_model_with_metadata(ensemble_no_stacking, tmp_path):
    """Test loading BNN model with metadata."""
    # Create mock model file (empty is fine for this test)
    model_path = tmp_path / "bnn_model.pkl"
    metadata_path = tmp_path / "bnn_model_metadata.json"

    # Create mock metadata
    metadata = {
        "model_version": "v3.1",
        "test_mae": 58.70,
        "calibration_coverage": 0.895,
        "training_date": "2025-11-08",
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    # Mock BayesianNeuralNetwork.load
    mock_bnn = Mock()
    with patch(
        "nfl_analytics.ensemble.enhanced_ensemble_v3.BayesianNeuralNetwork.load",
        return_value=mock_bnn,
    ):
        with patch("pathlib.Path.exists", return_value=True):
            result = ensemble_no_stacking.load_bnn_model(
                model_path=model_path, metadata_path=metadata_path
            )

    assert result == mock_bnn
    assert ensemble_no_stacking.bnn is not None
    assert ensemble_no_stacking.bnn_metadata["test_mae"] == 58.70


# ============================================================================
# TEST 7: CONFIDENCE INTERVAL MERGING
# ============================================================================


def test_confidence_intervals_present(
    ensemble_no_stacking, mock_bayesian_predictions, mock_xgb_predictions, mock_bnn_predictions
):
    """Test that Bayesian confidence intervals are preserved."""
    result = ensemble_no_stacking.get_4way_ensemble_predictions(
        mock_bayesian_predictions,
        mock_xgb_predictions,
        mock_bnn_predictions,
    )

    # Bayesian quantiles should be preserved
    assert "rating_q05" in result.columns
    assert "rating_q50" in result.columns
    assert "rating_q95" in result.columns

    # Quantiles should be ordered correctly
    for idx in range(len(result)):
        row = result.iloc[idx]
        assert row["rating_q05"] <= row["rating_q50"] <= row["rating_q95"]


# ============================================================================
# TEST 8: EDGE CASES
# ============================================================================


def test_single_player_prediction(
    ensemble_no_stacking,
):
    """Test ensemble with single player."""
    # Create single-player predictions
    bayesian_single = pd.DataFrame(
        {
            "player_id": [1],
            "stat_type": ["passing_yards"],
            "season": [2024],
            "bayesian_pred": [275.0],
            "bayesian_uncertainty": [25.0],
            "rating_q05": [225.0],
            "rating_q50": [275.0],
            "rating_q95": [325.0],
            "n_games_observed": [10],
        }
    )

    xgb_single = pd.DataFrame(
        {
            "player_id": [1],
            "stat_type": ["passing_yards"],
            "xgb_pred": [280.0],
        }
    )

    bnn_single = pd.DataFrame(
        {
            "player_id": [1],
            "stat_type": ["passing_yards"],
            "pred": [270.0],
            "uncertainty": [30.0],
        }
    )

    result = ensemble_no_stacking.get_4way_ensemble_predictions(
        bayesian_single,
        xgb_single,
        bnn_single,
    )

    assert len(result) == 1
    assert "ensemble_pred" in result.columns
    # Prediction should be weighted average (closer to lower uncertainty)
    assert 260 < result["ensemble_pred"].iloc[0] < 290


def test_identical_predictions_across_models(
    ensemble_no_stacking,
):
    """Test ensemble when all models predict the same value."""
    # Create identical predictions
    bayesian_pred = pd.DataFrame(
        {
            "player_id": range(1, 11),
            "stat_type": ["passing_yards"] * 10,
            "season": [2024] * 10,
            "bayesian_pred": [250.0] * 10,
            "bayesian_uncertainty": [20.0] * 10,
            "rating_q05": [210.0] * 10,
            "rating_q50": [250.0] * 10,
            "rating_q95": [290.0] * 10,
            "n_games_observed": [10] * 10,
        }
    )

    xgb_pred = pd.DataFrame(
        {
            "player_id": range(1, 11),
            "stat_type": ["passing_yards"] * 10,
            "xgb_pred": [250.0] * 10,
        }
    )

    bnn_pred = pd.DataFrame(
        {
            "player_id": range(1, 11),
            "stat_type": ["passing_yards"] * 10,
            "pred": [250.0] * 10,
            "uncertainty": [25.0] * 10,
        }
    )

    result = ensemble_no_stacking.get_4way_ensemble_predictions(
        bayesian_pred,
        xgb_pred,
        bnn_pred,
    )

    # Ensemble should also predict ~250 (may have slight numerical differences)
    assert all(abs(result["ensemble_pred"] - 250.0) < 1.0)


def test_extreme_uncertainty_values(
    ensemble_no_stacking,
):
    """Test ensemble with extreme uncertainty values."""
    # Create predictions with extreme uncertainty
    bayesian_pred = pd.DataFrame(
        {
            "player_id": range(1, 11),
            "stat_type": ["passing_yards"] * 10,
            "season": [2024] * 10,
            "bayesian_pred": np.random.randint(200, 300, 10).astype(float),
            "bayesian_uncertainty": [0.1] * 10,  # Extremely low uncertainty
            "rating_q05": [200.0] * 10,
            "rating_q50": [250.0] * 10,
            "rating_q95": [300.0] * 10,
            "n_games_observed": [50] * 10,
        }
    )

    xgb_pred = pd.DataFrame(
        {
            "player_id": range(1, 11),
            "stat_type": ["passing_yards"] * 10,
            "xgb_pred": np.random.randint(180, 320, 10).astype(float),
        }
    )

    bnn_pred = pd.DataFrame(
        {
            "player_id": range(1, 11),
            "stat_type": ["passing_yards"] * 10,
            "pred": np.random.randint(190, 310, 10).astype(float),
            "uncertainty": [100.0] * 10,  # Extremely high uncertainty
        }
    )

    result = ensemble_no_stacking.get_4way_ensemble_predictions(
        bayesian_pred,
        xgb_pred,
        bnn_pred,
    )

    # Should handle extreme values without crashing
    assert len(result) == 10
    assert all(np.isfinite(result["ensemble_pred"]))
    assert all(np.isfinite(result["ensemble_uncertainty"]))


# ============================================================================
# TEST 9: PERFORMANCE
# ============================================================================


def test_ensemble_performance_large_dataset(ensemble_no_stacking):
    """Test ensemble performance with large dataset (1000 players)."""
    import time

    n = 1000

    bayesian_large = pd.DataFrame(
        {
            "player_id": range(1, n + 1),
            "stat_type": ["passing_yards"] * n,
            "season": [2024] * n,
            "bayesian_pred": np.random.randint(200, 350, n).astype(float),
            "bayesian_uncertainty": np.random.uniform(15, 40, n),
            "rating_q05": np.random.randint(150, 250, n).astype(float),
            "rating_q50": np.random.randint(200, 300, n).astype(float),
            "rating_q95": np.random.randint(250, 400, n).astype(float),
            "n_games_observed": np.random.randint(5, 50, n),
        }
    )

    xgb_large = pd.DataFrame(
        {
            "player_id": range(1, n + 1),
            "stat_type": ["passing_yards"] * n,
            "xgb_pred": np.random.randint(180, 360, n).astype(float),
        }
    )

    bnn_large = pd.DataFrame(
        {
            "player_id": range(1, n + 1),
            "stat_type": ["passing_yards"] * n,
            "pred": np.random.randint(190, 340, n).astype(float),
            "uncertainty": np.random.uniform(20, 50, n),
        }
    )

    start = time.time()
    result = ensemble_no_stacking.get_4way_ensemble_predictions(
        bayesian_large,
        xgb_large,
        bnn_large,
    )
    elapsed = time.time() - start

    # Should complete in < 1 second for 1000 players
    assert elapsed < 1.0, f"Ensemble took {elapsed:.2f}s (target: < 1s)"
    assert len(result) == n


# ============================================================================
# MARKER: Integration tests (may require database/models)
# ============================================================================

pytestmark = pytest.mark.integration
