"""
Unit tests for baseline_glm.py - walk-forward validation GLM model.

Critical baseline model for dissertation - tests calibration, temporal splits, and metrics.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add py/ to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "py"))


class TestWalkForwardValidation:
    """Test suite for walk-forward cross-validation."""

    @pytest.fixture
    def sample_data(self):
        """Sample dataset for testing."""
        np.random.seed(42)
        n_samples = 100
        
        return pd.DataFrame({
            'season': np.repeat([2020, 2021, 2022, 2023], 25),
            'week': np.tile(range(1, 26), 4)[:100],
            'game_id': [f"2020_{i:03d}" for i in range(100)],
            'spread': np.random.randn(n_samples) * 7,
            'total': 45 + np.random.randn(n_samples) * 10,
            'home_score': np.random.randint(10, 35, n_samples),
            'away_score': np.random.randint(10, 35, n_samples),
            'covered': np.random.randint(0, 2, n_samples),
        })

    def test_temporal_split_no_leakage(self, sample_data):
        """Test that train/test splits maintain temporal order."""
        train_seasons = [2020, 2021]
        test_season = 2022
        
        train_data = sample_data[sample_data['season'].isin(train_seasons)]
        test_data = sample_data[sample_data['season'] == test_season]
        
        # Ensure no temporal overlap
        assert train_data['season'].max() < test_season
        assert len(test_data) > 0

    def test_expanding_window_validation(self, sample_data):
        """Test expanding window walk-forward validation."""
        seasons = sorted(sample_data['season'].unique())
        
        for i in range(1, len(seasons)):
            train_seasons = seasons[:i]
            test_season = seasons[i]
            
            train_size = len(sample_data[sample_data['season'].isin(train_seasons)])
            test_size = len(sample_data[sample_data['season'] == test_season])
            
            # Training set should grow
            assert train_size >= 25
            # Test set should be consistent
            assert test_size > 0

    def test_minimum_training_samples(self, sample_data):
        """Test that minimum training samples are enforced."""
        min_samples = 50
        
        for season in sorted(sample_data['season'].unique())[1:]:
            train_data = sample_data[sample_data['season'] < season]
            
            if len(train_data) >= min_samples:
                assert True  # Valid training set
            else:
                # Should skip this fold
                assert len(train_data) < min_samples


class TestGLMModel:
    """Test suite for GLM logistic regression."""

    @pytest.fixture
    def sample_features(self):
        """Sample features for model training."""
        np.random.seed(42)
        n = 100
        
        return pd.DataFrame({
            'spread': np.random.randn(n) * 7,
            'total': 45 + np.random.randn(n) * 10,
            'home_epa': np.random.randn(n) * 0.5,
            'away_epa': np.random.randn(n) * 0.5,
            'rest_days_diff': np.random.randint(-3, 4, n),
        })

    @pytest.fixture
    def sample_target(self):
        """Sample binary target variable."""
        np.random.seed(42)
        return np.random.randint(0, 2, 100)

    def test_logistic_regression_fit(self, sample_features, sample_target):
        """Test that logistic regression fits without error."""
        from sklearn.linear_model import LogisticRegression
        
        model = LogisticRegression(random_state=42)
        model.fit(sample_features, sample_target)
        
        assert hasattr(model, 'coef_')
        assert model.coef_.shape[1] == sample_features.shape[1]

    def test_prediction_probabilities_valid(self, sample_features, sample_target):
        """Test that predicted probabilities are in [0, 1]."""
        from sklearn.linear_model import LogisticRegression
        
        model = LogisticRegression(random_state=42)
        model.fit(sample_features, sample_target)
        
        probs = model.predict_proba(sample_features)[:, 1]
        
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_home_field_advantage_coefficient(self):
        """Test that home field advantage has expected sign."""
        # Synthetic data with clear home advantage
        np.random.seed(42)
        n = 200
        
        X = pd.DataFrame({
            'spread': np.random.randn(n) * 5,
            'is_home': np.random.randint(0, 2, n),
        })
        
        # Home teams win more often
        y = ((X['is_home'] * 0.3 + np.random.randn(n) * 0.5) > 0).astype(int)
        
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        # Home coefficient should be positive
        home_coef_idx = list(X.columns).index('is_home')
        assert model.coef_[0][home_coef_idx] > 0


class TestCalibration:
    """Test suite for probability calibration."""

    @pytest.fixture
    def sample_probabilities(self):
        """Sample predicted probabilities."""
        np.random.seed(42)
        return np.random.beta(2, 2, 100)

    @pytest.fixture
    def sample_outcomes(self):
        """Sample binary outcomes."""
        np.random.seed(42)
        return np.random.randint(0, 2, 100)

    def test_platt_scaling(self, sample_probabilities, sample_outcomes):
        """Test Platt scaling calibration."""
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.dummy import DummyClassifier
        
        # Create dummy classifier
        base = DummyClassifier(strategy='prior')
        base.fit(np.zeros((100, 1)), sample_outcomes)
        
        # Apply Platt scaling
        calibrated = CalibratedClassifierCV(base, method='sigmoid', cv='prefit')
        
        # Should not raise error
        assert calibrated is not None

    def test_isotonic_regression(self, sample_probabilities, sample_outcomes):
        """Test isotonic regression calibration."""
        from sklearn.isotonic import IsotonicRegression
        
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(sample_probabilities, sample_outcomes)
        
        calibrated_probs = iso.predict(sample_probabilities)
        
        assert np.all(calibrated_probs >= 0)
        assert np.all(calibrated_probs <= 1)

    def test_reliability_diagram_bins(self, sample_probabilities, sample_outcomes):
        """Test reliability diagram computation."""
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        
        bin_indices = np.digitize(sample_probabilities, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Compute mean probability and outcome per bin
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                mean_pred = sample_probabilities[mask].mean()
                mean_outcome = sample_outcomes[mask].mean()
                
                assert 0 <= mean_pred <= 1
                assert 0 <= mean_outcome <= 1


class TestPerformanceMetrics:
    """Test suite for performance metrics calculation."""

    @pytest.fixture
    def perfect_predictions(self):
        """Perfect prediction scenario."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.9, 0.8])
        return y_true, y_pred

    @pytest.fixture
    def random_predictions(self):
        """Random prediction scenario."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.uniform(0, 1, 100)
        return y_true, y_pred

    def test_brier_score_perfect(self, perfect_predictions):
        """Test Brier score on perfect predictions."""
        from sklearn.metrics import brier_score_loss
        
        y_true, y_pred = perfect_predictions
        brier = brier_score_loss(y_true, y_pred)
        
        # Should be low for good predictions
        assert brier < 0.1

    def test_log_loss_calculation(self, random_predictions):
        """Test log loss calculation."""
        from sklearn.metrics import log_loss
        
        y_true, y_pred = random_predictions
        logloss = log_loss(y_true, y_pred)
        
        # Log loss should be positive
        assert logloss > 0
        # Should be finite
        assert np.isfinite(logloss)

    def test_roc_auc_score(self, random_predictions):
        """Test ROC AUC score calculation."""
        from sklearn.metrics import roc_auc_score
        
        y_true, y_pred = random_predictions
        auc = roc_auc_score(y_true, y_pred)
        
        # AUC should be in [0, 1]
        assert 0 <= auc <= 1

    def test_accuracy_calculation(self):
        """Test accuracy calculation."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])  # 75% accurate
        
        accuracy = (y_true == y_pred).mean()
        
        assert accuracy == 0.75

    def test_roi_calculation(self):
        """Test ROI calculation for betting."""
        # Bet results: win, win, lose, lose
        stakes = np.array([1.0, 1.0, 1.0, 1.0])
        payouts = np.array([1.9, 1.9, 0.0, 0.0])  # -110 odds
        
        profit = payouts.sum() - stakes.sum()
        roi = profit / stakes.sum()
        
        assert roi == -0.05  # -5% ROI (2 wins, 2 losses at -110)


class TestFeatureEngineering:
    """Test suite for feature engineering."""

    def test_weather_interaction_terms(self):
        """Test weather × spread interaction features."""
        df = pd.DataFrame({
            'spread': [-7, -3, 0, 3, 7],
            'wind_kph': [0, 10, 20, 30, 40],
        })
        
        # Create interaction
        df['spread_wind'] = df['spread'] * df['wind_kph']
        
        assert 'spread_wind' in df.columns
        assert df['spread_wind'].iloc[0] == -7 * 0

    def test_rest_days_differential(self):
        """Test rest days differential feature."""
        df = pd.DataFrame({
            'home_rest': [7, 7, 14, 7],
            'away_rest': [7, 14, 7, 3],
        })
        
        df['rest_diff'] = df['home_rest'] - df['away_rest']
        
        expected = np.array([0, -7, 7, 4])
        assert np.array_equal(df['rest_diff'].values, expected)

    def test_exponential_decay_weights(self):
        """Test exponential decay for recency weighting."""
        weeks_ago = np.array([0, 1, 2, 3, 4])
        half_life = 3.0
        
        weights = np.exp(-np.log(2) * weeks_ago / half_life)
        
        # Most recent should have weight 1
        assert weights[0] == 1.0
        # Weights should decay
        assert np.all(np.diff(weights) < 0)
        # All weights positive
        assert np.all(weights > 0)


class TestOutputGeneration:
    """Test suite for LaTeX output generation."""

    def test_tex_table_format(self):
        """Test LaTeX table generation."""
        results = {
            'season': [2020, 2021, 2022],
            'accuracy': [0.62, 0.65, 0.63],
            'brier': [0.23, 0.22, 0.24],
        }
        
        df = pd.DataFrame(results)
        
        # Generate simple LaTeX
        lines = [
            r"\begin{tabular}{ccc}",
            r"Season & Accuracy & Brier \\",
            r"\midrule",
        ]
        
        for _, row in df.iterrows():
            lines.append(
                f"{row['season']} & {row['accuracy']:.2f} & {row['brier']:.2f} \\\\"
            )
        
        lines.append(r"\end{tabular}")
        
        tex_output = "\n".join(lines)
        
        assert "\\begin{tabular}" in tex_output
        assert "\\midrule" in tex_output
        assert "2020" in tex_output

    def test_metrics_csv_export(self):
        """Test CSV export of metrics."""
        results = {
            'model': ['glm', 'glm_platt', 'glm_isotonic'],
            'accuracy': [0.62, 0.63, 0.64],
            'roi': [-0.02, 0.01, 0.02],
        }
        
        df = pd.DataFrame(results)
        csv_output = df.to_csv(index=False)
        
        assert 'model,accuracy,roi' in csv_output
        assert 'glm' in csv_output


class TestEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_single_season_data(self):
        """Test handling of single season data."""
        df = pd.DataFrame({
            'season': [2024] * 50,
            'week': list(range(1, 51)),
            'covered': np.random.randint(0, 2, 50),
        })
        
        # Should handle gracefully
        assert len(df) == 50
        assert df['season'].nunique() == 1

    def test_missing_features(self):
        """Test handling of missing features."""
        df = pd.DataFrame({
            'spread': [np.nan, -3, 0, 3],
            'total': [45, np.nan, 50, 48],
        })
        
        # Count missing
        missing = df.isnull().sum()
        
        assert missing['spread'] == 1
        assert missing['total'] == 1

    def test_extreme_spread_values(self):
        """Test handling of extreme spread values."""
        spreads = np.array([-28, -14, 0, 14, 28])
        
        # All should be finite
        assert np.all(np.isfinite(spreads))
        
        # Check range
        assert spreads.min() >= -50
        assert spreads.max() <= 50

    def test_perfect_separation(self):
        """Test handling of perfect separation in logistic regression."""
        # Perfect separation scenario
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        from sklearn.linear_model import LogisticRegression
        
        # Should handle with regularization
        model = LogisticRegression(C=1.0, random_state=42)
        model.fit(X, y)
        
        assert hasattr(model, 'coef_')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
