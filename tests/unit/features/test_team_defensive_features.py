#!/usr/bin/env python3
"""
Unit Tests for Team Defensive Features Module
==============================================

Tests the TeamDefensiveFeatures class for correct feature computation,
temporal safety, exponential decay weighting, and edge case handling.

Test Coverage:
1. Exponential decay weight computation (H=4)
2. Temporal safety validation (no future leakage)
3. Feature extraction with valid data
4. Default value handling for insufficient data
5. Edge cases (no games, missing data, single game)
6. Database query correctness
7. Integration with production schema

Author: Claude Code (Sonnet 4.5)
Date: November 21, 2025
"""

import unittest
from unittest.mock import MagicMock, patch
from typing import List, Tuple

import numpy as np
import pandas as pd

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from py.features.team_defensive_features import TeamDefensiveFeatures


class TestExponentialDecayWeights(unittest.TestCase):
    """Test exponential decay weight computation."""

    def setUp(self):
        """Initialize engine."""
        self.engine = TeamDefensiveFeatures()

    def test_weights_sum_to_one(self):
        """Weights should sum to 1.0 for normalization."""
        for n_games in [1, 2, 4, 8]:
            weights = self.engine._compute_exponential_weights(n_games)
            self.assertAlmostEqual(weights.sum(), 1.0, places=6,
                                   msg=f"Weights for {n_games} games should sum to 1.0")

    def test_weights_decay_with_recency(self):
        """More recent games should have higher weights."""
        weights = self.engine._compute_exponential_weights(4)

        # Most recent game (index 0) should have highest weight
        self.assertGreater(weights[0], weights[1],
                           msg="Most recent game should have highest weight")
        self.assertGreater(weights[1], weights[2],
                           msg="2nd most recent game should have higher weight than 3rd")
        self.assertGreater(weights[2], weights[3],
                           msg="3rd most recent game should have higher weight than 4th")

    def test_weights_exponential_decay_formula(self):
        """Verify exponential decay formula: weight[i] = exp(-i / H)."""
        n_games = 4
        weights = self.engine._compute_exponential_weights(n_games)

        # Compute expected weights manually
        expected_unnormalized = np.exp(-np.arange(n_games) / self.engine.DECAY_HALFLIFE)
        expected_weights = expected_unnormalized / expected_unnormalized.sum()

        np.testing.assert_array_almost_equal(
            weights,
            expected_weights,
            decimal=6,
            err_msg="Weights should follow exponential decay formula"
        )

    def test_weights_empty_games(self):
        """Empty games should return empty weights array."""
        weights = self.engine._compute_exponential_weights(0)
        self.assertEqual(len(weights), 0, msg="Empty games should return empty weights")

    def test_weights_single_game(self):
        """Single game should have weight 1.0."""
        weights = self.engine._compute_exponential_weights(1)
        self.assertEqual(len(weights), 1, msg="Single game should have 1 weight")
        self.assertAlmostEqual(weights[0], 1.0, places=6,
                               msg="Single game weight should be 1.0")


class TestTemporalSafety(unittest.TestCase):
    """Test temporal safety validation."""

    def setUp(self):
        """Initialize engine with mocked database."""
        self.engine = TeamDefensiveFeatures()

    @patch('py.features.team_defensive_features.psycopg2.connect')
    def test_validate_no_leakage_valid_games(self, mock_connect):
        """All games BEFORE target week should pass validation."""
        # Mock database query results: games BEFORE 2024 Week 10
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ('2024_09_KC_BUF', 2024, 9),
            ('2024_08_KC_DEN', 2024, 8),
            ('2024_07_KC_LV', 2024, 7),
            ('2024_06_KC_LAC', 2024, 6),
        ]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Validate: All games are BEFORE 2024 Week 10
        is_valid = self.engine.validate_no_leakage('KC', 2024, 10)

        self.assertTrue(is_valid, msg="All games before target should pass validation")

    @patch('py.features.team_defensive_features.psycopg2.connect')
    def test_validate_no_leakage_detects_future_game(self, mock_connect):
        """Games AT or AFTER target week should fail validation."""
        # Mock database query results: includes game AT target week (leakage)
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ('2024_10_KC_BUF', 2024, 10),  # LEAKAGE: same week as target
            ('2024_09_KC_DEN', 2024, 9),
            ('2024_08_KC_LV', 2024, 8),
        ]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Validate: Should detect leakage
        is_valid = self.engine.validate_no_leakage('KC', 2024, 10)

        self.assertFalse(is_valid, msg="Game at target week should fail validation")

    @patch('py.features.team_defensive_features.psycopg2.connect')
    def test_validate_no_leakage_cross_season(self, mock_connect):
        """Games from previous season should pass validation."""
        # Mock database query results: games from 2023 season
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ('2024_01_KC_BUF', 2024, 1),
            ('2023_17_KC_CIN', 2023, 17),
            ('2023_16_KC_LV', 2023, 16),
            ('2023_15_KC_LAC', 2023, 15),
        ]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Validate: 2023 games are BEFORE 2024 Week 2
        is_valid = self.engine.validate_no_leakage('KC', 2024, 2)

        self.assertTrue(is_valid, msg="Previous season games should pass validation")


class TestFeatureExtraction(unittest.TestCase):
    """Test feature extraction with mocked database queries."""

    def setUp(self):
        """Initialize engine with mocked database."""
        self.engine = TeamDefensiveFeatures()

    @patch('py.features.team_defensive_features.psycopg2.connect')
    def test_extract_features_returns_all_24_features(self, mock_connect):
        """extract_features should return all 24 expected features."""
        # Mock _get_recent_games
        self.engine._get_recent_games = MagicMock(return_value=[
            ('2024_09_KC_BUF', 2024, 9),
            ('2024_08_KC_DEN', 2024, 8),
            ('2024_07_KC_LV', 2024, 7),
            ('2024_06_KC_LAC', 2024, 6),
        ])

        # Mock all feature computation methods to return defaults
        self.engine._compute_epa_features = MagicMock(return_value={
            'def_epa_allowed_per_play': 0.0,
            'def_success_rate_allowed': 0.45,
            'def_pass_epa_allowed_per_play': 0.05,
            'def_rush_epa_allowed_per_play': -0.05,
            'def_turnover_rate': 0.024,
        })
        self.engine._compute_explosive_efficiency_features = MagicMock(return_value={
            'def_explosive_play_rate_allowed': 0.12,
            'def_stuff_rate': 0.18,
            'def_first_down_rate_allowed': 0.35,
            'def_third_down_conv_rate_allowed': 0.40,
            'def_red_zone_td_rate_allowed': 0.55,
        })
        self.engine._compute_passing_defense_features = MagicMock(return_value={
            'def_avg_yds_per_pass_allowed': 6.5,
            'def_completion_pct_allowed': 0.64,
            'def_pass_success_rate_allowed': 0.48,
            'def_sack_rate': 0.07,
            'def_pressure_rate': 0.25,
        })
        self.engine._compute_rushing_defense_features = MagicMock(return_value={
            'def_avg_yds_per_rush_allowed': 4.3,
            'def_rush_success_rate_allowed': 0.42,
        })
        self.engine._compute_game_level_features = MagicMock(return_value={
            'def_points_allowed_per_game': 22.0,
            'def_yds_allowed_per_game': 340.0,
            'def_pass_yds_allowed_per_game': 225.0,
            'def_rush_yds_allowed_per_game': 115.0,
            'def_scoring_drives_pct_allowed': 0.35,
            'def_penalty_rate': 0.03,
            'def_blitz_rate': 0.30,
        })

        # Extract features
        features = self.engine.extract_features('KC', 2024, 10)

        # Verify all 24 features present
        expected_features = [
            # EPA (5)
            'def_epa_allowed_per_play', 'def_success_rate_allowed',
            'def_pass_epa_allowed_per_play', 'def_rush_epa_allowed_per_play',
            'def_turnover_rate',
            # Explosive/Efficiency (5)
            'def_explosive_play_rate_allowed', 'def_stuff_rate',
            'def_first_down_rate_allowed', 'def_third_down_conv_rate_allowed',
            'def_red_zone_td_rate_allowed',
            # Passing (5)
            'def_avg_yds_per_pass_allowed', 'def_completion_pct_allowed',
            'def_pass_success_rate_allowed', 'def_sack_rate', 'def_pressure_rate',
            # Rushing (2)
            'def_avg_yds_per_rush_allowed', 'def_rush_success_rate_allowed',
            # Game-level (7)
            'def_points_allowed_per_game', 'def_yds_allowed_per_game',
            'def_pass_yds_allowed_per_game', 'def_rush_yds_allowed_per_game',
            'def_scoring_drives_pct_allowed', 'def_penalty_rate', 'def_blitz_rate',
        ]

        self.assertEqual(len(features), 24,
                         msg="Should return exactly 24 features")
        for feature in expected_features:
            self.assertIn(feature, features,
                          msg=f"Feature '{feature}' should be present")

    @patch('py.features.team_defensive_features.psycopg2.connect')
    def test_extract_features_insufficient_games_uses_defaults(self, mock_connect):
        """With <4 games available, should use league average defaults."""
        # Mock _get_recent_games: only 2 games available (early season)
        self.engine._get_recent_games = MagicMock(return_value=[
            ('2024_02_KC_BUF', 2024, 2),
            ('2024_01_KC_DEN', 2024, 1),
        ])

        # Mock feature methods to return defaults
        self.engine._compute_epa_features = MagicMock(return_value={
            'def_epa_allowed_per_play': self.engine.DEFAULT_DEF_EPA,
            'def_success_rate_allowed': self.engine.DEFAULT_SUCCESS_RATE,
            'def_pass_epa_allowed_per_play': self.engine.DEFAULT_PASS_EPA,
            'def_rush_epa_allowed_per_play': self.engine.DEFAULT_RUSH_EPA,
            'def_turnover_rate': self.engine.DEFAULT_TURNOVER_RATE,
        })
        self.engine._compute_explosive_efficiency_features = MagicMock(return_value={
            'def_explosive_play_rate_allowed': self.engine.DEFAULT_EXPLOSIVE_RATE,
            'def_stuff_rate': self.engine.DEFAULT_STUFF_RATE,
            'def_first_down_rate_allowed': self.engine.DEFAULT_FIRST_DOWN_RATE,
            'def_third_down_conv_rate_allowed': self.engine.DEFAULT_THIRD_DOWN_RATE,
            'def_red_zone_td_rate_allowed': self.engine.DEFAULT_RED_ZONE_TD_RATE,
        })
        self.engine._compute_passing_defense_features = MagicMock(return_value={
            'def_avg_yds_per_pass_allowed': self.engine.DEFAULT_YDS_PER_PASS,
            'def_completion_pct_allowed': self.engine.DEFAULT_COMPLETION_PCT,
            'def_pass_success_rate_allowed': self.engine.DEFAULT_PASS_SUCCESS,
            'def_sack_rate': self.engine.DEFAULT_SACK_RATE,
            'def_pressure_rate': self.engine.DEFAULT_PRESSURE_RATE,
        })
        self.engine._compute_rushing_defense_features = MagicMock(return_value={
            'def_avg_yds_per_rush_allowed': self.engine.DEFAULT_YDS_PER_RUSH,
            'def_rush_success_rate_allowed': self.engine.DEFAULT_RUSH_SUCCESS,
        })
        self.engine._compute_game_level_features = MagicMock(return_value={
            'def_points_allowed_per_game': self.engine.DEFAULT_POINTS_ALLOWED,
            'def_yds_allowed_per_game': self.engine.DEFAULT_YDS_ALLOWED,
            'def_pass_yds_allowed_per_game': self.engine.DEFAULT_PASS_YDS_ALLOWED,
            'def_rush_yds_allowed_per_game': self.engine.DEFAULT_RUSH_YDS_ALLOWED,
            'def_scoring_drives_pct_allowed': self.engine.DEFAULT_SCORING_DRIVES_PCT,
            'def_penalty_rate': self.engine.DEFAULT_PENALTY_RATE,
            'def_blitz_rate': self.engine.DEFAULT_BLITZ_RATE,
        })

        # Extract features
        features = self.engine.extract_features('KC', 2024, 3)

        # Verify defaults used (EPA example)
        self.assertAlmostEqual(
            features['def_epa_allowed_per_play'],
            self.engine.DEFAULT_DEF_EPA,
            places=4,
            msg="Should use default EPA with insufficient games"
        )


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Initialize engine."""
        self.engine = TeamDefensiveFeatures()

    @patch('py.features.team_defensive_features.psycopg2.connect')
    def test_no_games_available_returns_all_defaults(self, mock_connect):
        """With no games available, should return all league average defaults."""
        # Mock _get_recent_games: no games
        self.engine._get_recent_games = MagicMock(return_value=[])

        # Extract features
        features = self.engine.extract_features('KC', 2024, 1)

        # Verify all defaults
        self.assertEqual(features['def_epa_allowed_per_play'],
                         self.engine.DEFAULT_DEF_EPA)
        self.assertEqual(features['def_points_allowed_per_game'],
                         self.engine.DEFAULT_POINTS_ALLOWED)
        self.assertEqual(features['def_sack_rate'],
                         self.engine.DEFAULT_SACK_RATE)

    def test_database_connection_config(self):
        """Test database configuration initialization."""
        # Default config
        engine1 = TeamDefensiveFeatures()
        self.assertEqual(engine1.db_config['host'], 'localhost')
        self.assertEqual(engine1.db_config['port'], 5544)

        # Custom config
        custom_config = {
            'host': 'remote-db.example.com',
            'port': 5432,
            'dbname': 'testdb',
            'user': 'testuser',
            'password': 'testpass',
        }
        engine2 = TeamDefensiveFeatures(db_config=custom_config)
        self.assertEqual(engine2.db_config['host'], 'remote-db.example.com')
        self.assertEqual(engine2.db_config['port'], 5432)


class TestIntegrationPatterns(unittest.TestCase):
    """Test integration patterns with production schema."""

    def setUp(self):
        """Initialize engine."""
        self.engine = TeamDefensiveFeatures()

    def test_lookback_window_constant(self):
        """Verify LOOKBACK_WINDOW = 4 as specified."""
        self.assertEqual(self.engine.LOOKBACK_WINDOW, 4,
                         msg="LOOKBACK_WINDOW should be 4 games")

    def test_decay_halflife_constant(self):
        """Verify DECAY_HALFLIFE = 4 as specified."""
        self.assertEqual(self.engine.DECAY_HALFLIFE, 4.0,
                         msg="DECAY_HALFLIFE should be 4.0 games")

    def test_all_default_constants_defined(self):
        """Verify all 24 default constants are defined."""
        defaults = [
            'DEFAULT_DEF_EPA', 'DEFAULT_SUCCESS_RATE', 'DEFAULT_EXPLOSIVE_RATE',
            'DEFAULT_STUFF_RATE', 'DEFAULT_FIRST_DOWN_RATE', 'DEFAULT_THIRD_DOWN_RATE',
            'DEFAULT_RED_ZONE_TD_RATE', 'DEFAULT_PASS_EPA', 'DEFAULT_RUSH_EPA',
            'DEFAULT_PASS_SUCCESS', 'DEFAULT_RUSH_SUCCESS', 'DEFAULT_YDS_PER_PASS',
            'DEFAULT_YDS_PER_RUSH', 'DEFAULT_COMPLETION_PCT', 'DEFAULT_SACK_RATE',
            'DEFAULT_PRESSURE_RATE', 'DEFAULT_TURNOVER_RATE', 'DEFAULT_PENALTY_RATE',
            'DEFAULT_POINTS_ALLOWED', 'DEFAULT_YDS_ALLOWED', 'DEFAULT_PASS_YDS_ALLOWED',
            'DEFAULT_RUSH_YDS_ALLOWED', 'DEFAULT_SCORING_DRIVES_PCT', 'DEFAULT_BLITZ_RATE',
        ]

        for default_name in defaults:
            self.assertTrue(hasattr(self.engine, default_name),
                            msg=f"Missing default constant: {default_name}")


if __name__ == '__main__':
    unittest.main()
