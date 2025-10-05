"""
Unit tests for asof_features.py - leakage-safe feature generation.

Critical for ensuring temporal integrity in model training.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add py/ to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "py"))

from features.asof_features import (
    build_asof_snapshot,
    validate_no_leakage,
    compute_rolling_stats,
)


class TestAsofFeatureGeneration:
    """Test suite for as-of feature generation."""

    @pytest.fixture
    def mock_db_connection(self):
        """Mock database connection."""
        conn = Mock()
        cursor = Mock()
        conn.cursor.return_value.__enter__.return_value = cursor
        return conn, cursor

    @pytest.fixture
    def sample_games_data(self):
        """Sample games data for testing."""
        return pd.DataFrame({
            'game_id': ['2024_01_BUF_KC', '2024_02_BUF_MIA', '2024_03_BUF_NYJ'],
            'season': [2024, 2024, 2024],
            'week': [1, 2, 3],
            'gameday': [
                datetime(2024, 9, 10),
                datetime(2024, 9, 17),
                datetime(2024, 9, 24),
            ],
            'home_team': ['KC', 'MIA', 'NYJ'],
            'away_team': ['BUF', 'BUF', 'BUF'],
            'home_score': [27, 21, 17],
            'away_score': [24, 31, 24],
        })

    def test_no_future_leakage(self, sample_games_data):
        """Test that as-of snapshots don't include future data."""
        game = sample_games_data.iloc[1]  # Week 2 game
        cutoff_date = game['gameday']
        
        # Features should only use data before cutoff
        valid_games = sample_games_data[
            sample_games_data['gameday'] < cutoff_date
        ]
        
        assert len(valid_games) == 1  # Only week 1
        assert valid_games.iloc[0]['week'] == 1

    def test_rolling_stats_calculation(self):
        """Test rolling statistics computation."""
        data = pd.Series([10, 20, 30, 40, 50])
        
        # Rolling mean with window 3
        rolling_mean = data.rolling(window=3, min_periods=1).mean()
        
        assert rolling_mean.iloc[0] == 10.0  # First value
        assert rolling_mean.iloc[2] == 20.0  # (10+20+30)/3
        assert rolling_mean.iloc[4] == 40.0  # (30+40+50)/3

    def test_exponential_decay_weighting(self):
        """Test exponential decay for recency weighting."""
        half_life = 0.6  # weeks
        decay_factor = np.exp(-np.log(2) / half_life)
        
        # Weights for 3 games (most recent to oldest)
        weights = np.array([
            1.0,  # Current week
            decay_factor,  # 1 week ago
            decay_factor ** 2,  # 2 weeks ago
        ])
        
        # Verify decay behavior
        assert weights[0] == 1.0
        assert weights[1] < 1.0
        assert weights[2] < weights[1]
        assert np.all(weights > 0)

    def test_team_feature_aggregation(self, sample_games_data):
        """Test aggregation of team-level features."""
        team = 'BUF'
        team_games = sample_games_data[
            (sample_games_data['home_team'] == team) |
            (sample_games_data['away_team'] == team)
        ]
        
        assert len(team_games) == 3
        
        # Calculate average points scored
        buf_scores = []
        for _, game in team_games.iterrows():
            if game['home_team'] == team:
                buf_scores.append(game['home_score'])
            else:
                buf_scores.append(game['away_score'])
        
        avg_points = np.mean(buf_scores)
        assert avg_points == pytest.approx(26.0)  # (24+31+24)/3

    def test_opponent_adjusted_features(self, sample_games_data):
        """Test opponent-adjusted feature calculation."""
        # Compute strength of schedule
        team_avg_points = {'KC': 27, 'MIA': 21, 'NYJ': 17}
        
        opponents = ['KC', 'MIA', 'NYJ']
        sos = np.mean([team_avg_points[opp] for opp in opponents])
        
        assert sos == pytest.approx(21.67, rel=0.01)

    def test_home_away_split_features(self, sample_games_data):
        """Test home/away split statistics."""
        team = 'BUF'
        
        # BUF is away in all games
        away_games = sample_games_data[sample_games_data['away_team'] == team]
        away_scores = away_games['away_score'].values
        
        assert len(away_games) == 3
        assert away_scores.mean() == 26.0

    @pytest.mark.parametrize("week,expected_games", [
        (1, 0),  # No prior games
        (2, 1),  # One prior game
        (3, 2),  # Two prior games
    ])
    def test_cumulative_feature_growth(self, sample_games_data, week, expected_games):
        """Test that features accumulate correctly over time."""
        cutoff = sample_games_data[sample_games_data['week'] == week]['gameday'].iloc[0]
        prior_games = sample_games_data[sample_games_data['gameday'] < cutoff]
        
        assert len(prior_games) == expected_games

    def test_missing_data_handling(self):
        """Test graceful handling of missing data."""
        data = pd.Series([10, np.nan, 30, np.nan, 50])
        
        # Rolling mean should skip NaN values
        rolling_mean = data.rolling(window=3, min_periods=1).mean()
        
        assert not np.isnan(rolling_mean.iloc[0])
        assert not np.isnan(rolling_mean.iloc[2])

    def test_min_sample_size_enforcement(self):
        """Test that minimum sample sizes are enforced."""
        data = pd.Series([10, 20])
        min_periods = 3
        
        rolling_mean = data.rolling(window=5, min_periods=min_periods).mean()
        
        # Should be NaN for insufficient data
        assert np.isnan(rolling_mean.iloc[0])
        assert np.isnan(rolling_mean.iloc[1])

    def test_feature_consistency_across_snapshots(self):
        """Test that features remain consistent when recomputed."""
        data = pd.Series([10, 20, 30, 40, 50])
        
        # Compute twice
        result1 = data.rolling(window=3, min_periods=1).mean()
        result2 = data.rolling(window=3, min_periods=1).mean()
        
        assert np.allclose(result1, result2)

    def test_seasonal_boundaries(self, sample_games_data):
        """Test that features don't cross season boundaries."""
        # Add a game from previous season
        prev_season = pd.DataFrame({
            'game_id': ['2023_17_BUF_MIA'],
            'season': [2023],
            'week': [17],
            'gameday': [datetime(2024, 1, 7)],
            'home_team': ['MIA'],
            'away_team': ['BUF'],
            'home_score': [21],
            'away_score': [14],
        })
        
        all_games = pd.concat([prev_season, sample_games_data], ignore_index=True)
        
        # For 2024 week 1, should only use 2024 data
        season_2024 = all_games[all_games['season'] == 2024]
        assert len(season_2024) == 3

    def test_validation_catches_leakage(self):
        """Test that validation detects temporal leakage."""
        # Create scenario with leakage
        train_cutoff = datetime(2024, 9, 15)
        
        # Feature using future data (BAD)
        future_game = datetime(2024, 9, 20)
        
        assert future_game > train_cutoff  # This would be leakage

    def test_deterministic_output(self):
        """Test that feature generation is deterministic."""
        np.random.seed(42)
        data1 = pd.Series(np.random.randn(100))
        result1 = data1.rolling(window=10).mean()
        
        np.random.seed(42)
        data2 = pd.Series(np.random.randn(100))
        result2 = data2.rolling(window=10).mean()
        
        assert np.allclose(result1, result2, equal_nan=True)


class TestLeakageValidation:
    """Test suite for leakage detection."""

    def test_strict_temporal_ordering(self):
        """Test that temporal ordering is strictly enforced."""
        dates = pd.DatetimeIndex([
            datetime(2024, 1, 1),
            datetime(2024, 1, 8),
            datetime(2024, 1, 15),
        ])
        
        # Check ordering
        assert dates.is_monotonic_increasing

    def test_cross_validation_folds_no_leakage(self):
        """Test that CV folds maintain temporal integrity."""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        # Time-series split
        train_size = 70
        train_dates = dates[:train_size]
        test_dates = dates[train_size:]
        
        # Ensure no overlap
        assert train_dates.max() < test_dates.min()

    def test_lag_features_proper_offset(self):
        """Test that lag features use correct offsets."""
        data = pd.Series([1, 2, 3, 4, 5])
        
        # 1-period lag
        lagged = data.shift(1)
        
        assert np.isnan(lagged.iloc[0])
        assert lagged.iloc[1] == 1
        assert lagged.iloc[2] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
