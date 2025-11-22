#!/usr/bin/env python3
"""
Unit Tests for GameOutcomePredictor
====================================

Tests the game outcome prediction logic including:
- QB lookup
- Defensive EPA calculation
- Yards to score conversion
- Uncertainty propagation
- Edge cases (missing QB data)

Author: Claude Code (Sonnet 4.5)
Date: November 21, 2025
"""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from py.ensemble.game_outcome_predictor import GameOutcomePredictor


class TestGameOutcomePredictor:
    """Test suite for GameOutcomePredictor."""

    @pytest.fixture
    def predictor(self):
        """Create predictor instance with mocked ensemble."""
        with patch('py.ensemble.game_outcome_predictor.EnhancedEnsemble5Way') as mock_ensemble:
            # Mock ensemble predict method
            mock_instance = Mock()
            mock_instance.predict.return_value = {
                'ensemble_mean': 250.0,
                'ensemble_std': 40.0
            }
            mock_ensemble.return_value = mock_instance

            predictor = GameOutcomePredictor()
            predictor.ensemble = mock_instance
            yield predictor

    def test_yards_to_score_basic(self, predictor):
        """Test basic yards to score conversion."""
        # 240 yards, neutral defense
        score = predictor._yards_to_score(qb_yards=240.0, opponent_def_epa=0.0)
        expected = 10 + (240 / 20)  # = 10 + 12 = 22
        assert score == pytest.approx(expected, rel=0.01)

    def test_yards_to_score_good_defense(self, predictor):
        """Test score adjustment for good defense (negative EPA)."""
        # 240 yards against good defense (negative EPA)
        score = predictor._yards_to_score(qb_yards=240.0, opponent_def_epa=-0.1)
        expected = 10 + (240 / 20) + (-5 * -0.1)  # = 22 + 0.5 = 22.5
        assert score == pytest.approx(expected, rel=0.01)

    def test_yards_to_score_bad_defense(self, predictor):
        """Test score adjustment for bad defense (positive EPA)."""
        # 240 yards against bad defense (positive EPA)
        score = predictor._yards_to_score(qb_yards=240.0, opponent_def_epa=0.1)
        expected = 10 + (240 / 20) + (-5 * 0.1)  # = 22 - 0.5 = 21.5
        assert score == pytest.approx(expected, rel=0.01)

    def test_propagate_uncertainty(self, predictor):
        """Test uncertainty propagation from yards to score."""
        # QB std = 40 yards
        score_std = predictor._propagate_uncertainty(qb_std=40.0)

        # Expected: sqrt((40/20)^2 + 3.5^2) = sqrt(4 + 12.25) = sqrt(16.25) = 4.03
        expected = np.sqrt((40/20)**2 + 3.5**2)
        assert score_std == pytest.approx(expected, rel=0.01)

    def test_propagate_uncertainty_zero_qb(self, predictor):
        """Test uncertainty with zero QB uncertainty (pure defensive)."""
        score_std = predictor._propagate_uncertainty(qb_std=0.0)

        # Expected: just defensive uncertainty = 3.5
        assert score_std == pytest.approx(3.5, rel=0.01)

    @patch('py.ensemble.game_outcome_predictor.psycopg2.connect')
    def test_predict_game_known_teams(self, mock_connect, predictor):
        """Test full game prediction with known teams."""
        # Mock database connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Mock QB lookups
        mock_cursor.fetchone.side_effect = [
            # Home QB
            {'player_id': '00-0039163', 'player_display_name': 'C.J. Stroud', 'stat_attempts': 33},
            # Away QB
            {'player_id': '00-0036442', 'player_display_name': 'Josh Allen', 'stat_attempts': 28},
            # Home defense EPA
            (-0.05,),
            # Away defense EPA
            (0.05,),
        ]

        # Mock ensemble predictions
        predictor.ensemble.predict.side_effect = [
            {'ensemble_mean': 260.0, 'ensemble_std': 42.0},  # Home QB
            {'ensemble_mean': 240.0, 'ensemble_std': 38.0},  # Away QB
        ]

        # Predict game
        result = predictor.predict_game(
            home_team='HOU',
            away_team='BUF',
            season=2025,
            week=12
        )

        # Assertions
        assert 'predicted_home_score' in result
        assert 'predicted_away_score' in result
        assert 'predicted_total' in result
        assert 'predicted_spread' in result
        assert 'predicted_home_win_prob' in result
        assert 'spread_std' in result
        assert 'total_std' in result

        # Check values are reasonable
        assert 10 <= result['predicted_home_score'] <= 50
        assert 10 <= result['predicted_away_score'] <= 50
        assert 20 <= result['predicted_total'] <= 100
        assert -30 <= result['predicted_spread'] <= 30
        assert 0 <= result['predicted_home_win_prob'] <= 1
        assert result['spread_std'] > 0
        assert result['total_std'] > 0

        # Check QB info
        assert result['home_qb_id'] == '00-0039163'
        assert result['home_qb_name'] == 'C.J. Stroud'
        assert result['away_qb_id'] == '00-0036442'
        assert result['away_qb_name'] == 'Josh Allen'

    @patch('py.ensemble.game_outcome_predictor.psycopg2.connect')
    def test_predict_game_missing_qb(self, mock_connect, predictor):
        """Test game prediction with missing QB (fallback to default)."""
        # Mock database connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Mock QB lookups (one QB missing)
        mock_cursor.fetchone.side_effect = [
            # Home QB (found)
            {'player_id': '00-0039163', 'player_display_name': 'C.J. Stroud', 'stat_attempts': 33},
            # Away QB (not found)
            None,
            # Home defense EPA
            (-0.05,),
            # Away defense EPA
            (0.05,),
        ]

        # Mock ensemble prediction (only for home QB)
        predictor.ensemble.predict.return_value = {
            'ensemble_mean': 260.0,
            'ensemble_std': 42.0
        }

        # Predict game
        result = predictor.predict_game(
            home_team='HOU',
            away_team='BUF',
            season=2025,
            week=12
        )

        # Assertions
        assert result['home_qb_id'] == '00-0039163'
        assert result['home_qb_name'] == 'C.J. Stroud'
        assert result['away_qb_id'] is None
        assert result['away_qb_name'] is None

        # Away score should use default 240 yards
        # score = 10 + 240/20 + (-5 * 0.05) = 10 + 12 - 0.25 = 21.75
        expected_away_score = 10 + (240 / 20) + (-5 * 0.05)
        assert result['predicted_away_score'] == pytest.approx(expected_away_score, rel=0.1)

    def test_win_probability_calculation(self, predictor):
        """Test win probability calculation from spread."""
        # Test various spreads
        test_cases = [
            (7.0, 0.5 + 0.25),   # 7-point favorite ~73%
            (0.0, 0.5),          # Even spread = 50%
            (-7.0, 0.5 - 0.25),  # 7-point underdog ~27%
        ]

        for spread, expected_prob in test_cases:
            prob = 1 / (1 + np.exp(-spread / 7))
            assert prob == pytest.approx(expected_prob, abs=0.05)

    def test_uncertainty_combination(self, predictor):
        """Test that home and away uncertainties combine correctly."""
        home_std = 4.0
        away_std = 4.0

        # Total uncertainty should be sqrt(home^2 + away^2)
        expected_total_std = np.sqrt(home_std**2 + away_std**2)
        assert expected_total_std == pytest.approx(5.66, rel=0.01)

    def test_prediction_schema(self, predictor):
        """Test that prediction returns all required fields."""
        with patch('py.ensemble.game_outcome_predictor.psycopg2.connect'):
            # Mock minimal DB response
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_cursor.fetchone.side_effect = [
                {'player_id': '1', 'player_display_name': 'QB1', 'stat_attempts': 30},
                {'player_id': '2', 'player_display_name': 'QB2', 'stat_attempts': 28},
                (0.0,), (0.0,)
            ]
            mock_conn.cursor.return_value = mock_cursor

            with patch('py.ensemble.game_outcome_predictor.psycopg2.connect', return_value=mock_conn):
                result = predictor.predict_game('KC', 'BUF', 2025, 12)

            # Check all required fields exist
            required_fields = [
                'predicted_home_score',
                'predicted_away_score',
                'predicted_total',
                'predicted_spread',
                'predicted_home_win_prob',
                'spread_std',
                'total_std',
                'home_qb_id',
                'home_qb_name',
                'away_qb_id',
                'away_qb_name',
                'processing_time_ms'
            ]

            for field in required_fields:
                assert field in result, f"Missing required field: {field}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
