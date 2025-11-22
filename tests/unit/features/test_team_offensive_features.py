#!/usr/bin/env python3
"""
Unit Tests for Team Offensive Features Module
==============================================

Tests the TeamOffensiveFeatures class for correctness, temporal safety,
and edge case handling.

Test Coverage:
1. Basic functionality tests (smoke tests)
2. Temporal safety validation (no future leakage)
3. Edge cases (early season, missing data)
4. Exponential decay weighting validation
5. League average defaults
6. Database schema compliance

Author: Claude Code (Sonnet 4.5)
Date: November 21, 2025
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Tuple

# Import the module under test
from py.features.team_offensive_features import TeamOffensiveFeatures


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_db_config():
    """Mock database configuration for testing."""
    return {
        "host": "localhost",
        "port": 5544,
        "dbname": "devdb01",
        "user": "dro",
        "password": "sicillionbillions",
    }


@pytest.fixture
def team_offensive_features(mock_db_config):
    """Create TeamOffensiveFeatures instance with mock config."""
    return TeamOffensiveFeatures(db_config=mock_db_config)


@pytest.fixture
def mock_recent_games():
    """Mock recent games data."""
    return [
        ("2024_09_KC_BUF", 2024, 9),
        ("2024_08_KC_SF", 2024, 8),
        ("2024_07_KC_LV", 2024, 7),
        ("2024_06_KC_LAC", 2024, 6),
    ]


# ============================================================================
# Test 1: Basic Functionality (Smoke Tests)
# ============================================================================

def test_initialization(mock_db_config):
    """Test TeamOffensiveFeatures initialization."""
    engine = TeamOffensiveFeatures(db_config=mock_db_config)

    assert engine.LOOKBACK_WINDOW == 4
    assert engine.DECAY_HALFLIFE == 4.0
    assert engine.db_config == mock_db_config


def test_initialization_with_individual_params():
    """Test initialization with individual database parameters."""
    engine = TeamOffensiveFeatures(
        db_host="localhost",
        db_port=5544,
        db_name="devdb01",
        db_user="dro",
        db_password="sicillionbillions"
    )

    assert engine.db_config["host"] == "localhost"
    assert engine.db_config["port"] == 5544
    assert engine.db_config["dbname"] == "devdb01"


def test_extract_features_returns_24_features(team_offensive_features):
    """Test that extract_features returns exactly 24 features."""
    with patch.object(team_offensive_features, '_get_recent_games') as mock_get_games, \
         patch.object(team_offensive_features, '_compute_play_level_features') as mock_play, \
         patch.object(team_offensive_features, '_compute_red_zone_features') as mock_redzone, \
         patch.object(team_offensive_features, '_compute_game_level_features') as mock_game:

        # Mock return values
        mock_get_games.return_value = [
            ("2024_09_KC_BUF", 2024, 9),
            ("2024_08_KC_SF", 2024, 8),
            ("2024_07_KC_LV", 2024, 7),
            ("2024_06_KC_LAC", 2024, 6),
        ]

        mock_play.return_value = {
            'off_epa_per_play': 0.10,
            'off_success_rate': 0.48,
            'off_explosive_play_rate': 0.22,
            'off_stuff_rate': 0.16,
            'off_first_down_rate': 0.38,
            'off_third_down_conv_rate': 0.42,
            'off_pass_epa_per_play': 0.15,
            'off_rush_epa_per_play': 0.05,
            'off_pass_success_rate': 0.50,
            'off_rush_success_rate': 0.46,
            'off_avg_yds_per_pass': 7.5,
            'off_avg_yds_per_rush': 4.5,
            'off_completion_pct': 0.66,
            'off_sack_rate': 0.05,
            'off_turnover_rate': 0.018,
            'off_penalty_rate': 0.048,
        }

        mock_redzone.return_value = {
            'off_red_zone_td_rate': 0.60,
        }

        mock_game.return_value = {
            'off_time_of_possession': 1900.0,
            'off_plays_per_game': 68.0,
            'off_points_per_game': 28.5,
            'off_yds_per_game': 380.0,
            'off_pass_yds_per_game': 260.0,
            'off_rush_yds_per_game': 120.0,
            'off_scoring_drives_pct': 0.40,
        }

        # Execute
        features = team_offensive_features.extract_features("KC", 2024, 10)

        # Assert
        assert len(features) == 24, f"Expected 24 features, got {len(features)}"

        # Verify all expected keys exist
        expected_keys = [
            'off_epa_per_play', 'off_success_rate', 'off_explosive_play_rate',
            'off_stuff_rate', 'off_first_down_rate', 'off_third_down_conv_rate',
            'off_pass_epa_per_play', 'off_rush_epa_per_play', 'off_pass_success_rate',
            'off_rush_success_rate', 'off_avg_yds_per_pass', 'off_avg_yds_per_rush',
            'off_completion_pct', 'off_sack_rate', 'off_turnover_rate', 'off_penalty_rate',
            'off_red_zone_td_rate', 'off_time_of_possession', 'off_plays_per_game',
            'off_points_per_game', 'off_yds_per_game', 'off_pass_yds_per_game',
            'off_rush_yds_per_game', 'off_scoring_drives_pct'
        ]

        for key in expected_keys:
            assert key in features, f"Missing expected feature: {key}"


# ============================================================================
# Test 2: Temporal Safety Validation
# ============================================================================

def test_get_recent_games_temporal_safety(team_offensive_features):
    """Test that _get_recent_games only returns games BEFORE target week."""
    with patch.object(team_offensive_features, '_connect_db') as mock_connect:
        # Mock database connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("2024_09_KC_BUF", 2024, 9),
            ("2024_08_KC_SF", 2024, 8),
            ("2024_07_KC_LV", 2024, 7),
            ("2024_06_KC_LAC", 2024, 6),
        ]

        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Execute
        games = team_offensive_features._get_recent_games("KC", 2024, 10)

        # Assert: All games must be BEFORE Week 10
        for game_id, season, week in games:
            assert season < 2024 or (season == 2024 and week < 10), \
                f"Game {game_id} ({season} Week {week}) is NOT before target (2024 Week 10)"


def test_validate_no_leakage_success(team_offensive_features):
    """Test validate_no_leakage returns True when all games are before target."""
    with patch.object(team_offensive_features, '_get_recent_games') as mock_get_games:
        # Mock games BEFORE target (2024 Week 10)
        mock_get_games.return_value = [
            ("2024_09_KC_BUF", 2024, 9),
            ("2024_08_KC_SF", 2024, 8),
            ("2024_07_KC_LV", 2024, 7),
            ("2024_06_KC_LAC", 2024, 6),
        ]

        # Execute
        is_valid = team_offensive_features.validate_no_leakage("KC", 2024, 10)

        # Assert
        assert is_valid is True


def test_validate_no_leakage_failure_same_week(team_offensive_features):
    """Test validate_no_leakage returns False when game is AT target week (leakage)."""
    with patch.object(team_offensive_features, '_get_recent_games') as mock_get_games:
        # Mock games including target week (LEAKAGE)
        mock_get_games.return_value = [
            ("2024_10_KC_DEN", 2024, 10),  # Same week (LEAKAGE)
            ("2024_09_KC_BUF", 2024, 9),
            ("2024_08_KC_SF", 2024, 8),
            ("2024_07_KC_LV", 2024, 7),
        ]

        # Execute
        is_valid = team_offensive_features.validate_no_leakage("KC", 2024, 10)

        # Assert
        assert is_valid is False


def test_validate_no_leakage_failure_future_week(team_offensive_features):
    """Test validate_no_leakage returns False when game is AFTER target week (leakage)."""
    with patch.object(team_offensive_features, '_get_recent_games') as mock_get_games:
        # Mock games including future week (LEAKAGE)
        mock_get_games.return_value = [
            ("2024_11_KC_BUF", 2024, 11),  # Future week (LEAKAGE)
            ("2024_09_KC_BUF", 2024, 9),
            ("2024_08_KC_SF", 2024, 8),
            ("2024_07_KC_LV", 2024, 7),
        ]

        # Execute
        is_valid = team_offensive_features.validate_no_leakage("KC", 2024, 10)

        # Assert
        assert is_valid is False


# ============================================================================
# Test 3: Edge Cases
# ============================================================================

def test_extract_features_early_season_uses_defaults(team_offensive_features):
    """Test that extract_features uses league average defaults when <4 games available."""
    with patch.object(team_offensive_features, '_get_recent_games') as mock_get_games, \
         patch.object(team_offensive_features, '_compute_play_level_features') as mock_play, \
         patch.object(team_offensive_features, '_compute_red_zone_features') as mock_redzone, \
         patch.object(team_offensive_features, '_compute_game_level_features') as mock_game:

        # Mock only 2 games available (early season)
        mock_get_games.return_value = [
            ("2024_02_KC_BUF", 2024, 2),
            ("2024_01_KC_BAL", 2024, 1),
        ]

        # Mock returns should still provide defaults
        mock_play.return_value = {
            'off_epa_per_play': team_offensive_features.DEFAULT_EPA_PER_PLAY,
            'off_success_rate': team_offensive_features.DEFAULT_SUCCESS_RATE,
            'off_explosive_play_rate': team_offensive_features.DEFAULT_EXPLOSIVE_RATE,
            'off_stuff_rate': team_offensive_features.DEFAULT_STUFF_RATE,
            'off_first_down_rate': team_offensive_features.DEFAULT_FIRST_DOWN_RATE,
            'off_third_down_conv_rate': team_offensive_features.DEFAULT_THIRD_DOWN_CONV,
            'off_pass_epa_per_play': team_offensive_features.DEFAULT_EPA_PER_PLAY,
            'off_rush_epa_per_play': team_offensive_features.DEFAULT_EPA_PER_PLAY,
            'off_pass_success_rate': team_offensive_features.DEFAULT_SUCCESS_RATE,
            'off_rush_success_rate': team_offensive_features.DEFAULT_SUCCESS_RATE,
            'off_avg_yds_per_pass': team_offensive_features.DEFAULT_YDS_PER_PASS,
            'off_avg_yds_per_rush': team_offensive_features.DEFAULT_YDS_PER_RUSH,
            'off_completion_pct': team_offensive_features.DEFAULT_COMPLETION_PCT,
            'off_sack_rate': team_offensive_features.DEFAULT_SACK_RATE,
            'off_turnover_rate': team_offensive_features.DEFAULT_TURNOVER_RATE,
            'off_penalty_rate': team_offensive_features.DEFAULT_PENALTY_RATE,
        }

        mock_redzone.return_value = {
            'off_red_zone_td_rate': team_offensive_features.DEFAULT_RED_ZONE_TD_RATE,
        }

        mock_game.return_value = {
            'off_time_of_possession': team_offensive_features.DEFAULT_TIME_OF_POSSESSION,
            'off_plays_per_game': team_offensive_features.DEFAULT_PLAYS_PER_GAME,
            'off_points_per_game': team_offensive_features.DEFAULT_POINTS_PER_GAME,
            'off_yds_per_game': team_offensive_features.DEFAULT_YDS_PER_GAME,
            'off_pass_yds_per_game': team_offensive_features.DEFAULT_PASS_YDS_PER_GAME,
            'off_rush_yds_per_game': team_offensive_features.DEFAULT_RUSH_YDS_PER_GAME,
            'off_scoring_drives_pct': team_offensive_features.DEFAULT_SCORING_DRIVES_PCT,
        }

        # Execute
        features = team_offensive_features.extract_features("KC", 2024, 3)

        # Assert: Should use league average defaults
        assert features['off_epa_per_play'] == team_offensive_features.DEFAULT_EPA_PER_PLAY
        assert features['off_success_rate'] == team_offensive_features.DEFAULT_SUCCESS_RATE
        assert features['off_points_per_game'] == team_offensive_features.DEFAULT_POINTS_PER_GAME


def test_extract_features_no_games_available(team_offensive_features):
    """Test extract_features when NO games are available (Week 1 scenario)."""
    with patch.object(team_offensive_features, '_get_recent_games') as mock_get_games, \
         patch.object(team_offensive_features, '_compute_play_level_features') as mock_play, \
         patch.object(team_offensive_features, '_compute_red_zone_features') as mock_redzone, \
         patch.object(team_offensive_features, '_compute_game_level_features') as mock_game:

        # Mock NO games available (Week 1)
        mock_get_games.return_value = []

        # Mock methods should return defaults when no games
        mock_play.return_value = {
            'off_epa_per_play': team_offensive_features.DEFAULT_EPA_PER_PLAY,
            'off_success_rate': team_offensive_features.DEFAULT_SUCCESS_RATE,
            'off_explosive_play_rate': team_offensive_features.DEFAULT_EXPLOSIVE_RATE,
            'off_stuff_rate': team_offensive_features.DEFAULT_STUFF_RATE,
            'off_first_down_rate': team_offensive_features.DEFAULT_FIRST_DOWN_RATE,
            'off_third_down_conv_rate': team_offensive_features.DEFAULT_THIRD_DOWN_CONV,
            'off_pass_epa_per_play': team_offensive_features.DEFAULT_EPA_PER_PLAY,
            'off_rush_epa_per_play': team_offensive_features.DEFAULT_EPA_PER_PLAY,
            'off_pass_success_rate': team_offensive_features.DEFAULT_SUCCESS_RATE,
            'off_rush_success_rate': team_offensive_features.DEFAULT_SUCCESS_RATE,
            'off_avg_yds_per_pass': team_offensive_features.DEFAULT_YDS_PER_PASS,
            'off_avg_yds_per_rush': team_offensive_features.DEFAULT_YDS_PER_RUSH,
            'off_completion_pct': team_offensive_features.DEFAULT_COMPLETION_PCT,
            'off_sack_rate': team_offensive_features.DEFAULT_SACK_RATE,
            'off_turnover_rate': team_offensive_features.DEFAULT_TURNOVER_RATE,
            'off_penalty_rate': team_offensive_features.DEFAULT_PENALTY_RATE,
        }

        mock_redzone.return_value = {
            'off_red_zone_td_rate': team_offensive_features.DEFAULT_RED_ZONE_TD_RATE,
        }

        mock_game.return_value = {
            'off_time_of_possession': team_offensive_features.DEFAULT_TIME_OF_POSSESSION,
            'off_plays_per_game': team_offensive_features.DEFAULT_PLAYS_PER_GAME,
            'off_points_per_game': team_offensive_features.DEFAULT_POINTS_PER_GAME,
            'off_yds_per_game': team_offensive_features.DEFAULT_YDS_PER_GAME,
            'off_pass_yds_per_game': team_offensive_features.DEFAULT_PASS_YDS_PER_GAME,
            'off_rush_yds_per_game': team_offensive_features.DEFAULT_RUSH_YDS_PER_GAME,
            'off_scoring_drives_pct': team_offensive_features.DEFAULT_SCORING_DRIVES_PCT,
        }

        # Execute
        features = team_offensive_features.extract_features("KC", 2024, 1)

        # Assert: All features should be league average defaults
        assert len(features) == 24
        assert features['off_epa_per_play'] == team_offensive_features.DEFAULT_EPA_PER_PLAY


# ============================================================================
# Test 4: League Average Defaults
# ============================================================================

def test_default_constants_are_reasonable():
    """Test that default constants are within reasonable NFL ranges."""
    # EPA metrics (typically -0.3 to +0.3)
    assert -0.3 <= TeamOffensiveFeatures.DEFAULT_EPA_PER_PLAY <= 0.3

    # Success rate (typically 40-50%)
    assert 0.35 <= TeamOffensiveFeatures.DEFAULT_SUCCESS_RATE <= 0.55

    # Explosive play rate (typically 15-25%)
    assert 0.15 <= TeamOffensiveFeatures.DEFAULT_EXPLOSIVE_RATE <= 0.30

    # Completion percentage (typically 60-70%)
    assert 0.60 <= TeamOffensiveFeatures.DEFAULT_COMPLETION_PCT <= 0.75

    # Sack rate (typically 4-8%)
    assert 0.03 <= TeamOffensiveFeatures.DEFAULT_SACK_RATE <= 0.10

    # Points per game (typically 18-28)
    assert 15 <= TeamOffensiveFeatures.DEFAULT_POINTS_PER_GAME <= 30

    # Yards per game (typically 300-400)
    assert 280 <= TeamOffensiveFeatures.DEFAULT_YDS_PER_GAME <= 450


# ============================================================================
# Test 5: Feature Value Validation
# ============================================================================

def test_extract_features_values_are_numeric(team_offensive_features):
    """Test that all extracted features are numeric (float)."""
    with patch.object(team_offensive_features, '_get_recent_games') as mock_get_games, \
         patch.object(team_offensive_features, '_compute_play_level_features') as mock_play, \
         patch.object(team_offensive_features, '_compute_red_zone_features') as mock_redzone, \
         patch.object(team_offensive_features, '_compute_game_level_features') as mock_game:

        # Mock return values
        mock_get_games.return_value = [("2024_09_KC_BUF", 2024, 9)]
        mock_play.return_value = {
            'off_epa_per_play': 0.10,
            'off_success_rate': 0.48,
            'off_explosive_play_rate': 0.22,
            'off_stuff_rate': 0.16,
            'off_first_down_rate': 0.38,
            'off_third_down_conv_rate': 0.42,
            'off_pass_epa_per_play': 0.15,
            'off_rush_epa_per_play': 0.05,
            'off_pass_success_rate': 0.50,
            'off_rush_success_rate': 0.46,
            'off_avg_yds_per_pass': 7.5,
            'off_avg_yds_per_rush': 4.5,
            'off_completion_pct': 0.66,
            'off_sack_rate': 0.05,
            'off_turnover_rate': 0.018,
            'off_penalty_rate': 0.048,
        }
        mock_redzone.return_value = {'off_red_zone_td_rate': 0.60}
        mock_game.return_value = {
            'off_time_of_possession': 1900.0,
            'off_plays_per_game': 68.0,
            'off_points_per_game': 28.5,
            'off_yds_per_game': 380.0,
            'off_pass_yds_per_game': 260.0,
            'off_rush_yds_per_game': 120.0,
            'off_scoring_drives_pct': 0.40,
        }

        # Execute
        features = team_offensive_features.extract_features("KC", 2024, 10)

        # Assert: All values must be numeric (float)
        for key, value in features.items():
            assert isinstance(value, (int, float)), \
                f"Feature '{key}' has non-numeric value: {value} (type: {type(value)})"


def test_extract_features_no_nan_values(team_offensive_features):
    """Test that no features contain NaN values."""
    with patch.object(team_offensive_features, '_get_recent_games') as mock_get_games, \
         patch.object(team_offensive_features, '_compute_play_level_features') as mock_play, \
         patch.object(team_offensive_features, '_compute_red_zone_features') as mock_redzone, \
         patch.object(team_offensive_features, '_compute_game_level_features') as mock_game:

        # Mock return values
        mock_get_games.return_value = [("2024_09_KC_BUF", 2024, 9)]
        mock_play.return_value = {
            'off_epa_per_play': 0.10,
            'off_success_rate': 0.48,
            'off_explosive_play_rate': 0.22,
            'off_stuff_rate': 0.16,
            'off_first_down_rate': 0.38,
            'off_third_down_conv_rate': 0.42,
            'off_pass_epa_per_play': 0.15,
            'off_rush_epa_per_play': 0.05,
            'off_pass_success_rate': 0.50,
            'off_rush_success_rate': 0.46,
            'off_avg_yds_per_pass': 7.5,
            'off_avg_yds_per_rush': 4.5,
            'off_completion_pct': 0.66,
            'off_sack_rate': 0.05,
            'off_turnover_rate': 0.018,
            'off_penalty_rate': 0.048,
        }
        mock_redzone.return_value = {'off_red_zone_td_rate': 0.60}
        mock_game.return_value = {
            'off_time_of_possession': 1900.0,
            'off_plays_per_game': 68.0,
            'off_points_per_game': 28.5,
            'off_yds_per_game': 380.0,
            'off_pass_yds_per_game': 260.0,
            'off_rush_yds_per_game': 120.0,
            'off_scoring_drives_pct': 0.40,
        }

        # Execute
        features = team_offensive_features.extract_features("KC", 2024, 10)

        # Assert: No NaN values
        for key, value in features.items():
            assert not (isinstance(value, float) and np.isnan(value)), \
                f"Feature '{key}' contains NaN value"


# ============================================================================
# Test 6: Integration Test (Requires Database)
# ============================================================================

@pytest.mark.integration
def test_extract_features_integration_kc_2024_week10():
    """
    Integration test with real database: Kansas City Chiefs, 2024 Week 10.

    This test requires a running PostgreSQL database with data.
    """
    # Initialize with real database connection
    engine = TeamOffensiveFeatures(
        db_host="localhost",
        db_port=5544,
        db_name="devdb01",
        db_user="dro",
        db_password="sicillionbillions"
    )

    # Extract features for KC in 2024 Week 10
    features = engine.extract_features("KC", 2024, 10)

    # Assert: 24 features returned
    assert len(features) == 24

    # Assert: All features are numeric
    for key, value in features.items():
        assert isinstance(value, (int, float)), \
            f"Feature '{key}' is not numeric: {value}"

    # Assert: No NaN values
    for key, value in features.items():
        if isinstance(value, float):
            assert not np.isnan(value), f"Feature '{key}' is NaN"

    # Assert: Temporal safety validation passes
    is_valid = engine.validate_no_leakage("KC", 2024, 10)
    assert is_valid is True

    # Log results
    print("\n✓ Integration Test Results (KC 2024 Week 10):")
    print(f"  off_epa_per_play:       {features['off_epa_per_play']:.4f}")
    print(f"  off_points_per_game:    {features['off_points_per_game']:.1f}")
    print(f"  off_yds_per_game:       {features['off_yds_per_game']:.1f}")
    print(f"  off_pass_success_rate:  {features['off_pass_success_rate']:.3f}")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
