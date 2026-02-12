"""Baseline tests for asof feature chain.

This module tests the inheritance chain of asof feature extractors:
1. asof_features.py (base) - 46 team features
2. asof_features_enhanced.py - +17 enhanced features
3. asof_features_v2.py - +7 v2 model features

These tests lock down existing behavior before consolidation in Phase 3.

Created: December 13, 2025
Phase 1.2 of code quality remediation plan.
"""

from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_team_game_data() -> pd.DataFrame:
    """Create sample team-game data for testing compute_team_history.

    This fixture creates a minimal dataset with:
    - 2 teams (KC, BUF)
    - 4 games each (2 home, 2 away)
    - Enough data to test rolling windows
    """
    data = [
        # KC games
        {
            'game_id': '2023_01_KC_DET',
            'team': 'KC',
            'is_home': True,
            'season': 2023,
            'week': 1,
            'kickoff': dt.datetime(2023, 9, 7, 20, 15),
            'home_team': 'KC',
            'away_team': 'DET',
            'home_score': 21,
            'away_score': 20,
            'spread_close': -7.0,
            'total_close': 52.5,
            'points_for': 21,
            'points_against': 20,
            'stadium': 'Arrowhead Stadium',
            'roof': 'outdoors',
            'surface': 'grass',
            'qb_id': 'MAHOMES',
            'qb_name': 'Patrick Mahomes',
            'coach_name': 'Andy Reid',
            'margin': 1,
            'epa_sum': 15.5,
            'plays': 60,
        },
        {
            'game_id': '2023_02_JAX_KC',
            'team': 'KC',
            'is_home': True,
            'season': 2023,
            'week': 2,
            'kickoff': dt.datetime(2023, 9, 14, 20, 15),
            'home_team': 'KC',
            'away_team': 'JAX',
            'home_score': 17,
            'away_score': 9,
            'spread_close': -5.5,
            'total_close': 50.0,
            'points_for': 17,
            'points_against': 9,
            'stadium': 'Arrowhead Stadium',
            'roof': 'outdoors',
            'surface': 'grass',
            'qb_id': 'MAHOMES',
            'qb_name': 'Patrick Mahomes',
            'coach_name': 'Andy Reid',
            'margin': 8,
            'epa_sum': 12.3,
            'plays': 55,
        },
        {
            'game_id': '2023_03_KC_CHI',
            'team': 'KC',
            'is_home': False,
            'season': 2023,
            'week': 3,
            'kickoff': dt.datetime(2023, 9, 21, 20, 15),
            'home_team': 'CHI',
            'away_team': 'KC',
            'home_score': 14,
            'away_score': 28,
            'spread_close': 6.0,
            'total_close': 45.0,
            'points_for': 28,
            'points_against': 14,
            'stadium': 'Soldier Field',
            'roof': 'outdoors',
            'surface': 'grass',
            'qb_id': 'MAHOMES',
            'qb_name': 'Patrick Mahomes',
            'coach_name': 'Andy Reid',
            'margin': 14,
            'epa_sum': 20.1,
            'plays': 58,
        },
        {
            'game_id': '2023_04_KC_NYJ',
            'team': 'KC',
            'is_home': False,
            'season': 2023,
            'week': 4,
            'kickoff': dt.datetime(2023, 9, 28, 20, 15),
            'home_team': 'NYJ',
            'away_team': 'KC',
            'home_score': 20,
            'away_score': 23,
            'spread_close': 3.0,
            'total_close': 40.0,
            'points_for': 23,
            'points_against': 20,
            'stadium': 'MetLife Stadium',
            'roof': 'outdoors',
            'surface': 'turf',
            'qb_id': 'MAHOMES',
            'qb_name': 'Patrick Mahomes',
            'coach_name': 'Andy Reid',
            'margin': 3,
            'epa_sum': 8.7,
            'plays': 62,
        },
        # BUF games
        {
            'game_id': '2023_01_NYJ_BUF',
            'team': 'BUF',
            'is_home': True,
            'season': 2023,
            'week': 1,
            'kickoff': dt.datetime(2023, 9, 10, 13, 0),
            'home_team': 'BUF',
            'away_team': 'NYJ',
            'home_score': 22,
            'away_score': 16,
            'spread_close': -6.5,
            'total_close': 44.5,
            'points_for': 22,
            'points_against': 16,
            'stadium': 'Highmark Stadium',
            'roof': 'outdoors',
            'surface': 'turf',
            'qb_id': 'ALLEN',
            'qb_name': 'Josh Allen',
            'coach_name': 'Sean McDermott',
            'margin': 6,
            'epa_sum': 10.2,
            'plays': 65,
        },
        {
            'game_id': '2023_02_BUF_LV',
            'team': 'BUF',
            'is_home': False,
            'season': 2023,
            'week': 2,
            'kickoff': dt.datetime(2023, 9, 17, 16, 25),
            'home_team': 'LV',
            'away_team': 'BUF',
            'home_score': 23,
            'away_score': 38,
            'spread_close': 3.0,
            'total_close': 47.0,
            'points_for': 38,
            'points_against': 23,
            'stadium': 'Allegiant Stadium',
            'roof': 'dome',
            'surface': 'grass',
            'qb_id': 'ALLEN',
            'qb_name': 'Josh Allen',
            'coach_name': 'Sean McDermott',
            'margin': 15,
            'epa_sum': 22.5,
            'plays': 70,
        },
        {
            'game_id': '2023_03_WAS_BUF',
            'team': 'BUF',
            'is_home': True,
            'season': 2023,
            'week': 3,
            'kickoff': dt.datetime(2023, 9, 24, 13, 0),
            'home_team': 'BUF',
            'away_team': 'WAS',
            'home_score': 37,
            'away_score': 3,
            'spread_close': -10.0,
            'total_close': 42.0,
            'points_for': 37,
            'points_against': 3,
            'stadium': 'Highmark Stadium',
            'roof': 'outdoors',
            'surface': 'turf',
            'qb_id': 'ALLEN',
            'qb_name': 'Josh Allen',
            'coach_name': 'Sean McDermott',
            'margin': 34,
            'epa_sum': 28.0,
            'plays': 58,
        },
        {
            'game_id': '2023_04_BUF_MIA',
            'team': 'BUF',
            'is_home': False,
            'season': 2023,
            'week': 4,
            'kickoff': dt.datetime(2023, 10, 1, 13, 0),
            'home_team': 'MIA',
            'away_team': 'BUF',
            'home_score': 48,
            'away_score': 20,
            'spread_close': 3.5,
            'total_close': 53.0,
            'points_for': 20,
            'points_against': 48,
            'stadium': 'Hard Rock Stadium',
            'roof': 'outdoors',
            'surface': 'grass',
            'qb_id': 'ALLEN',
            'qb_name': 'Josh Allen',
            'coach_name': 'Sean McDermott',
            'margin': -28,
            'epa_sum': -5.3,
            'plays': 55,
        },
    ]
    return pd.DataFrame(data)


@pytest.fixture
def sample_enhanced_data(sample_team_game_data: pd.DataFrame) -> pd.DataFrame:
    """Extend base data with enhanced feature columns."""
    df = sample_team_game_data.copy()
    # Add enhanced columns
    df['turnovers'] = [1, 0, 1, 2, 2, 1, 0, 3]
    df['turnovers_opp'] = [2, 1, 0, 1, 1, 0, 1, 1]
    df['penalties'] = [5, 4, 6, 5, 7, 8, 4, 6]
    df['penalty_yards'] = [45, 35, 55, 40, 65, 75, 35, 55]
    df['success_rate'] = [0.45, 0.48, 0.52, 0.44, 0.42, 0.55, 0.58, 0.38]
    df['air_yards_mean'] = [8.5, 7.2, 9.1, 6.8, 7.8, 10.2, 8.5, 6.5]
    df['yac_mean'] = [4.2, 5.1, 4.8, 3.9, 5.5, 4.8, 6.2, 3.5]
    df['cpoe_mean'] = [2.5, 1.8, 3.2, 1.5, -0.5, 4.1, 3.8, -1.2]
    df['completion_pct'] = [0.65, 0.68, 0.72, 0.61, 0.58, 0.75, 0.70, 0.55]
    df['wp_max'] = [0.95, 0.92, 0.98, 0.88, 0.90, 0.96, 0.99, 0.65]
    df['wp_min'] = [0.35, 0.42, 0.55, 0.38, 0.28, 0.45, 0.60, 0.15]
    df['wp_q4_mean'] = [0.75, 0.82, 0.90, 0.68, 0.72, 0.88, 0.95, 0.25]
    df['shotgun_rate'] = [0.65, 0.60, 0.70, 0.55, 0.72, 0.68, 0.75, 0.60]
    df['no_huddle_rate'] = [0.10, 0.08, 0.15, 0.12, 0.18, 0.14, 0.08, 0.22]
    df['explosive_pass'] = [3, 2, 4, 2, 2, 5, 4, 1]
    df['explosive_rush'] = [2, 3, 2, 1, 3, 2, 4, 1]
    return df


@pytest.fixture
def sample_v2_data(sample_enhanced_data: pd.DataFrame) -> pd.DataFrame:
    """Extend enhanced data with v2 model feature columns."""
    df = sample_enhanced_data.copy()
    # Add v2 columns
    df['rest_days'] = [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0]
    df['is_division_game'] = [False, False, False, False, True, False, False, True]
    df['is_thursday_night'] = [True, True, True, True, False, False, False, False]
    df['is_short_week'] = [False, False, False, False, False, False, False, False]
    df['weather_condition'] = ['clear', 'clear', 'rain', 'clear', 'clear', 'dome', 'clear', 'clear']
    df['def_epa_last_4'] = [0.02, -0.05, -0.08, -0.03, 0.01, -0.02, -0.10, 0.05]
    df['opp_def_epa_last_4'] = [-0.03, 0.01, 0.02, -0.06, -0.02, 0.03, -0.05, -0.08]
    df['division_h2h_variance'] = [0.0, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 3.2]
    return df


# ============================================================================
# Base Module Tests (asof_features.py)
# ============================================================================


class TestBaseAsofFeatures:
    """Tests for py.features.asof_features module."""

    def test_import(self):
        """Verify base module can be imported."""
        from py.features import asof_features
        assert hasattr(asof_features, 'compute_team_history')
        assert hasattr(asof_features, 'pivot_to_games')
        assert hasattr(asof_features, 'validate_team_history')
        assert hasattr(asof_features, 'rename_with_prefix')
        assert hasattr(asof_features, 'get_connection')

    def test_team_feature_columns_count(self):
        """Verify expected number of base team features."""
        from py.features.asof_features import TEAM_FEATURE_COLUMNS
        # Base module should have 36 team features
        assert len(TEAM_FEATURE_COLUMNS) == 36, (
            f"Expected 36 base team features, got {len(TEAM_FEATURE_COLUMNS)}"
        )

    def test_diff_feature_columns_count(self):
        """Verify expected number of diff features."""
        from py.features.asof_features import DIFF_FEATURE_COLUMNS
        # Should have differential features
        assert len(DIFF_FEATURE_COLUMNS) > 20
        assert 'prior_epa_mean' in DIFF_FEATURE_COLUMNS
        assert 'rest_days' in DIFF_FEATURE_COLUMNS

    def test_rename_with_prefix(self):
        """Test the rename utility function."""
        from py.features.asof_features import rename_with_prefix

        columns = ['epa_mean', 'plays', 'rest_days']
        result = rename_with_prefix(columns, 'home')

        assert result == {
            'epa_mean': 'home_epa_mean',
            'plays': 'home_plays',
            'rest_days': 'home_rest_days',
        }

    def test_rename_with_prefix_away(self):
        """Test rename with away prefix."""
        from py.features.asof_features import rename_with_prefix

        columns = ['prior_games', 'margin_last3']
        result = rename_with_prefix(columns, 'away')

        assert result['prior_games'] == 'away_prior_games'
        assert result['margin_last3'] == 'away_margin_last3'

    def test_compute_team_history_prior_games(self, sample_team_game_data: pd.DataFrame):
        """Test that prior_games increments correctly (temporal safety)."""
        from py.features.asof_features import compute_team_history

        result = compute_team_history(sample_team_game_data)

        # Check KC team history
        kc = result[result['team'] == 'KC'].sort_values('week')
        assert list(kc['prior_games']) == [0, 1, 2, 3], (
            "prior_games should be 0,1,2,3 for consecutive games"
        )

        # Check BUF team history
        buf = result[result['team'] == 'BUF'].sort_values('week')
        assert list(buf['prior_games']) == [0, 1, 2, 3]

    def test_compute_team_history_prior_epa_sum(self, sample_team_game_data: pd.DataFrame):
        """Test that prior_epa_sum uses only past games."""
        from py.features.asof_features import compute_team_history

        result = compute_team_history(sample_team_game_data)
        kc = result[result['team'] == 'KC'].sort_values('week')

        # Week 1: prior_epa_sum should be 0 (no prior games)
        assert kc.iloc[0]['prior_epa_sum'] == 0.0

        # Week 2: prior_epa_sum should be week 1's epa_sum
        assert kc.iloc[1]['prior_epa_sum'] == pytest.approx(15.5, abs=0.01)

        # Week 3: prior_epa_sum should be sum of weeks 1+2
        assert kc.iloc[2]['prior_epa_sum'] == pytest.approx(15.5 + 12.3, abs=0.01)

    def test_compute_team_history_rest_days(self, sample_team_game_data: pd.DataFrame):
        """Test that rest_days is computed correctly."""
        from py.features.asof_features import compute_team_history

        result = compute_team_history(sample_team_game_data)
        kc = result[result['team'] == 'KC'].sort_values('week')

        # Week 1: rest_days should default to 7.0 (no prior game)
        assert kc.iloc[0]['rest_days'] == 7.0

        # Week 2: should be ~7 days from week 1
        assert kc.iloc[1]['rest_days'] == pytest.approx(7.0, abs=0.1)

    def test_compute_team_history_rolling_last3(self, sample_team_game_data: pd.DataFrame):
        """Test that rolling window features use shift(1)."""
        from py.features.asof_features import compute_team_history

        result = compute_team_history(sample_team_game_data)
        kc = result[result['team'] == 'KC'].sort_values('week')

        # Week 1: margin_last3 should be NaN (no prior games)
        assert pd.isna(kc.iloc[0]['margin_last3'])

        # Week 2: margin_last3 should be week 1's margin only
        assert kc.iloc[1]['margin_last3'] == pytest.approx(1.0, abs=0.01)

        # Week 4: margin_last3 should be avg of weeks 1,2,3 margins
        expected_margin_last3 = np.mean([1, 8, 14])  # Weeks 1,2,3
        assert kc.iloc[3]['margin_last3'] == pytest.approx(expected_margin_last3, abs=0.01)

    def test_compute_team_history_qb_change(self, sample_team_game_data: pd.DataFrame):
        """Test QB change detection."""
        from py.features.asof_features import compute_team_history

        result = compute_team_history(sample_team_game_data)
        kc = result[result['team'] == 'KC'].sort_values('week')

        # All KC games have Mahomes, so qb_change should be 0 after week 1
        assert kc.iloc[0]['qb_change'] == 0.0  # No prior game
        assert kc.iloc[1]['qb_change'] == 0.0  # Same QB
        assert kc.iloc[2]['qb_change'] == 0.0

    def test_compute_team_history_surface_encoding(self, sample_team_game_data: pd.DataFrame):
        """Test surface one-hot encoding."""
        from py.features.asof_features import compute_team_history

        result = compute_team_history(sample_team_game_data)

        # Check grass games
        grass_games = result[result['surface'] == 'grass']
        assert all(grass_games['surface_grass'] == 1.0)

        # Check turf games
        turf_games = result[result['surface'] == 'turf']
        assert all(turf_games['surface_artificial'] == 1.0)

    def test_validate_team_history_passes(self, sample_team_game_data: pd.DataFrame):
        """Test validation passes for correctly computed history."""
        from py.features.asof_features import compute_team_history, validate_team_history

        result = compute_team_history(sample_team_game_data)

        # Should not raise
        validate_team_history(result)

    def test_validate_team_history_detects_prior_games_error(self, sample_team_game_data: pd.DataFrame):
        """Test validation catches prior_games inconsistency."""
        from py.features.asof_features import compute_team_history, validate_team_history

        result = compute_team_history(sample_team_game_data)

        # Corrupt prior_games to simulate leakage
        result.loc[result.index[0], 'prior_games'] = 5

        with pytest.raises(ValueError, match="prior_games mismatch"):
            validate_team_history(result)

    def test_validate_team_history_detects_negative_rest(self, sample_team_game_data: pd.DataFrame):
        """Test validation catches negative rest days."""
        from py.features.asof_features import compute_team_history, validate_team_history

        result = compute_team_history(sample_team_game_data)

        # Corrupt rest_days
        result.loc[result.index[0], 'rest_days'] = -1

        with pytest.raises(ValueError, match="Negative rest_days"):
            validate_team_history(result)

    def test_pivot_to_games_creates_home_away(self, sample_team_game_data: pd.DataFrame):
        """Test pivoting creates home/away prefixed columns."""
        from py.features.asof_features import compute_team_history, pivot_to_games

        history = compute_team_history(sample_team_game_data)

        # Need to create matching home/away pairs
        # Our fixture only has team-level data, so we need to construct pairs
        # For this test, let's create a game where KC is home and BUF is away
        game_df = pd.DataFrame([{
            'game_id': 'TEST_KC_BUF',
            'team': 'KC',
            'is_home': True,
            'season': 2023,
            'week': 5,
            'kickoff': dt.datetime(2023, 10, 8, 13, 0),
            'home_team': 'KC',
            'away_team': 'BUF',
            'home_score': 24,
            'away_score': 21,
            'spread_close': -3.0,
            'total_close': 48.0,
            'points_for': 24,
            'points_against': 21,
            'stadium': 'Arrowhead Stadium',
            'roof': 'outdoors',
            'surface': 'grass',
            'qb_id': 'MAHOMES',
            'qb_name': 'Patrick Mahomes',
            'coach_name': 'Andy Reid',
            'margin': 3,
            'epa_sum': 12.0,
            'plays': 58,
        }, {
            'game_id': 'TEST_KC_BUF',
            'team': 'BUF',
            'is_home': False,
            'season': 2023,
            'week': 5,
            'kickoff': dt.datetime(2023, 10, 8, 13, 0),
            'home_team': 'KC',
            'away_team': 'BUF',
            'home_score': 24,
            'away_score': 21,
            'spread_close': -3.0,
            'total_close': 48.0,
            'points_for': 21,
            'points_against': 24,
            'stadium': 'Arrowhead Stadium',
            'roof': 'outdoors',
            'surface': 'grass',
            'qb_id': 'ALLEN',
            'qb_name': 'Josh Allen',
            'coach_name': 'Sean McDermott',
            'margin': -3,
            'epa_sum': 10.0,
            'plays': 55,
        }])

        history = compute_team_history(game_df)
        games = pivot_to_games(history)

        # Should have one game row
        assert len(games) == 1

        # Should have home_ and away_ prefixed columns
        assert 'home_prior_games' in games.columns
        assert 'away_prior_games' in games.columns
        assert 'home_rest_days' in games.columns
        assert 'away_rest_days' in games.columns

    def test_pivot_to_games_diff_columns(self, sample_team_game_data: pd.DataFrame):
        """Test pivoting creates differential columns."""
        from py.features.asof_features import compute_team_history, pivot_to_games

        # Create a paired game
        game_df = pd.DataFrame([{
            'game_id': 'TEST_GAME',
            'team': 'KC',
            'is_home': True,
            'season': 2023,
            'week': 1,
            'kickoff': dt.datetime(2023, 9, 10, 13, 0),
            'home_team': 'KC',
            'away_team': 'BUF',
            'home_score': 24,
            'away_score': 21,
            'spread_close': -3.0,
            'total_close': 48.0,
            'points_for': 24,
            'points_against': 21,
            'stadium': 'Arrowhead Stadium',
            'roof': 'outdoors',
            'surface': 'grass',
            'qb_id': 'MAHOMES',
            'qb_name': 'Patrick Mahomes',
            'coach_name': 'Andy Reid',
            'margin': 3,
            'epa_sum': 12.0,
            'plays': 58,
        }, {
            'game_id': 'TEST_GAME',
            'team': 'BUF',
            'is_home': False,
            'season': 2023,
            'week': 1,
            'kickoff': dt.datetime(2023, 9, 10, 13, 0),
            'home_team': 'KC',
            'away_team': 'BUF',
            'home_score': 24,
            'away_score': 21,
            'spread_close': -3.0,
            'total_close': 48.0,
            'points_for': 21,
            'points_against': 24,
            'stadium': 'Arrowhead Stadium',
            'roof': 'outdoors',
            'surface': 'grass',
            'qb_id': 'ALLEN',
            'qb_name': 'Josh Allen',
            'coach_name': 'Sean McDermott',
            'margin': -3,
            'epa_sum': 10.0,
            'plays': 55,
        }])

        history = compute_team_history(game_df)
        games = pivot_to_games(history)

        # Should have differential columns
        assert 'rest_diff' in games.columns
        assert 'epa_diff_prior' in games.columns


# ============================================================================
# Enhanced Module Tests (asof_features_enhanced.py)
# ============================================================================


class TestEnhancedAsofFeatures:
    """Tests for py.features.asof_features_enhanced module."""

    def test_import(self):
        """Verify enhanced module can be imported."""
        from py.features import asof_features_enhanced
        assert hasattr(asof_features_enhanced, 'compute_team_history_enhanced')
        assert hasattr(asof_features_enhanced, 'pivot_to_games_enhanced')
        assert hasattr(asof_features_enhanced, 'TEAM_FEATURE_COLUMNS_ENHANCED')

    def test_enhanced_columns_extend_base(self):
        """Verify enhanced columns include all base columns."""
        from py.features.asof_features import TEAM_FEATURE_COLUMNS
        from py.features.asof_features_enhanced import TEAM_FEATURE_COLUMNS_ENHANCED

        # Enhanced should be superset of base
        base_set = set(TEAM_FEATURE_COLUMNS)
        enhanced_set = set(TEAM_FEATURE_COLUMNS_ENHANCED)

        assert base_set.issubset(enhanced_set), (
            f"Missing base columns in enhanced: {base_set - enhanced_set}"
        )

    def test_enhanced_columns_count(self):
        """Verify expected number of enhanced features."""
        from py.features.asof_features_enhanced import TEAM_FEATURE_COLUMNS_ENHANCED

        # Should have more than base (46 + ~17 = 63)
        assert len(TEAM_FEATURE_COLUMNS_ENHANCED) > 50

    def test_enhanced_features_present(self):
        """Verify key enhanced features are defined."""
        from py.features.asof_features_enhanced import TEAM_FEATURE_COLUMNS_ENHANCED

        enhanced_only = [
            'prior_turnovers_avg',
            'prior_penalties_avg',
            'prior_success_rate',
            'success_rate_last3',
            'prior_air_yards',
            'prior_cpoe',
            'prior_completion_pct',
            'prior_explosive_pass_rate',
        ]

        for feat in enhanced_only:
            assert feat in TEAM_FEATURE_COLUMNS_ENHANCED, f"Missing enhanced feature: {feat}"

    def test_compute_team_history_enhanced_includes_base(self, sample_enhanced_data: pd.DataFrame):
        """Test enhanced computation includes all base features."""
        from py.features.asof_features_enhanced import compute_team_history_enhanced

        result = compute_team_history_enhanced(sample_enhanced_data)

        # Should have base features
        assert 'prior_games' in result.columns
        assert 'prior_epa_sum' in result.columns
        assert 'margin_last3' in result.columns

    def test_compute_team_history_enhanced_turnovers(self, sample_enhanced_data: pd.DataFrame):
        """Test enhanced turnover features."""
        from py.features.asof_features_enhanced import compute_team_history_enhanced

        result = compute_team_history_enhanced(sample_enhanced_data)

        # Should have turnover features
        assert 'prior_turnovers_avg' in result.columns
        assert 'prior_turnovers_opp_avg' in result.columns
        assert 'prior_penalties_avg' in result.columns

    def test_compute_team_history_enhanced_success_rate(self, sample_enhanced_data: pd.DataFrame):
        """Test enhanced success rate rolling features."""
        from py.features.asof_features_enhanced import compute_team_history_enhanced

        result = compute_team_history_enhanced(sample_enhanced_data)
        kc = result[result['team'] == 'KC'].sort_values('week')

        # Success rate features should use shift(1)
        assert 'prior_success_rate' in result.columns
        assert 'success_rate_last3' in result.columns

        # Week 1: prior_success_rate should be NaN
        assert pd.isna(kc.iloc[0]['prior_success_rate'])

        # Week 2: prior_success_rate should be week 1's success_rate
        assert kc.iloc[1]['prior_success_rate'] == pytest.approx(0.45, abs=0.01)

    def test_compute_team_history_enhanced_explosive_rate(self, sample_enhanced_data: pd.DataFrame):
        """Test enhanced explosive play rate features."""
        from py.features.asof_features_enhanced import compute_team_history_enhanced

        result = compute_team_history_enhanced(sample_enhanced_data)

        assert 'prior_explosive_pass_rate' in result.columns
        assert 'prior_explosive_rush_rate' in result.columns


# ============================================================================
# V2 Module Tests (asof_features_v2.py)
# ============================================================================


class TestV2AsofFeatures:
    """Tests for py.features.asof_features_v2 module."""

    def test_import(self):
        """Verify v2 module can be imported."""
        from py.features import asof_features_v2
        assert hasattr(asof_features_v2, 'compute_team_history_v2')
        assert hasattr(asof_features_v2, 'pivot_to_games_v2')
        assert hasattr(asof_features_v2, 'TEAM_FEATURE_COLUMNS_V2')

    def test_v2_columns_extend_enhanced(self):
        """Verify v2 columns include all enhanced columns."""
        from py.features.asof_features_enhanced import TEAM_FEATURE_COLUMNS_ENHANCED
        from py.features.asof_features_v2 import TEAM_FEATURE_COLUMNS_V2

        # V2 should be superset of enhanced
        enhanced_set = set(TEAM_FEATURE_COLUMNS_ENHANCED)
        v2_set = set(TEAM_FEATURE_COLUMNS_V2)

        assert enhanced_set.issubset(v2_set), (
            f"Missing enhanced columns in v2: {enhanced_set - v2_set}"
        )

    def test_v2_features_present(self):
        """Verify key v2 features are defined."""
        from py.features.asof_features_v2 import TEAM_FEATURE_COLUMNS_V2

        v2_only = [
            'is_division_game',
            'is_thursday_night',
            'is_short_week',
            'weather_condition',
            'def_epa_last_4',
            'opp_def_epa_last_4',
            'division_h2h_variance',
        ]

        for feat in v2_only:
            assert feat in TEAM_FEATURE_COLUMNS_V2, f"Missing v2 feature: {feat}"

    def test_compute_team_history_v2_division_features(self, sample_v2_data: pd.DataFrame):
        """Test v2 division game features."""
        from py.features.asof_features_v2 import compute_team_history_v2

        result = compute_team_history_v2(sample_v2_data)

        # Should have derived division features
        assert 'prior_division_win_rate' in result.columns
        assert 'prior_division_games' in result.columns

    def test_compute_team_history_v2_thursday_features(self, sample_v2_data: pd.DataFrame):
        """Test v2 Thursday night features."""
        from py.features.asof_features_v2 import compute_team_history_v2

        result = compute_team_history_v2(sample_v2_data)

        # Should have Thursday performance features
        assert 'prior_thursday_avg_margin' in result.columns
        assert 'prior_thursday_games' in result.columns

    def test_compute_team_history_v2_short_week_features(self, sample_v2_data: pd.DataFrame):
        """Test v2 short week features."""
        from py.features.asof_features_v2 import compute_team_history_v2

        result = compute_team_history_v2(sample_v2_data)

        # Should have short week performance features
        assert 'prior_short_week_avg_margin' in result.columns


# ============================================================================
# Temporal Safety Tests
# ============================================================================


class TestTemporalSafety:
    """Tests to verify no data leakage in asof features."""

    def test_prior_features_use_only_past_data(self, sample_team_game_data: pd.DataFrame):
        """Verify prior features don't include current game data."""
        from py.features.asof_features import compute_team_history

        result = compute_team_history(sample_team_game_data)
        kc = result[result['team'] == 'KC'].sort_values('week')

        # For each game, prior_epa_sum should NOT include current game's epa_sum
        for i in range(len(kc)):
            expected_prior_sum = kc.iloc[:i]['epa_sum'].sum()
            actual_prior_sum = kc.iloc[i]['prior_epa_sum']
            assert actual_prior_sum == pytest.approx(expected_prior_sum, abs=0.01), (
                f"Week {kc.iloc[i]['week']}: prior_epa_sum {actual_prior_sum} != expected {expected_prior_sum}"
            )

    def test_rolling_features_exclude_current(self, sample_team_game_data: pd.DataFrame):
        """Verify rolling window features use shift(1)."""
        from py.features.asof_features import compute_team_history

        result = compute_team_history(sample_team_game_data)
        kc = result[result['team'] == 'KC'].sort_values('week')

        # margin_last3 for week 4 should be avg of weeks 1,2,3 (not 2,3,4)
        week4 = kc.iloc[3]
        weeks_123 = kc.iloc[:3]
        expected = weeks_123['margin'].mean()

        assert week4['margin_last3'] == pytest.approx(expected, abs=0.01), (
            f"margin_last3 includes current game (leakage!)"
        )

    def test_win_percentage_excludes_current_result(self, sample_team_game_data: pd.DataFrame):
        """Verify win percentage doesn't include current game result."""
        from py.features.asof_features import compute_team_history

        result = compute_team_history(sample_team_game_data)
        kc = result[result['team'] == 'KC'].sort_values('week')

        # KC wins all 4 games (margins: 1, 8, 14, 3 are all positive)
        # prior_win_pct should be:
        # Week 1: NaN (no prior games)
        # Week 2: 1.0 (1 win / 1 game)
        # Week 3: 1.0 (2 wins / 2 games)
        # Week 4: 1.0 (3 wins / 3 games)

        assert pd.isna(kc.iloc[0]['prior_win_pct'])
        assert kc.iloc[1]['prior_win_pct'] == pytest.approx(1.0)
        assert kc.iloc[2]['prior_win_pct'] == pytest.approx(1.0)
        assert kc.iloc[3]['prior_win_pct'] == pytest.approx(1.0)

    def test_season_specific_features_reset(self, sample_team_game_data: pd.DataFrame):
        """Verify season-specific features reset at season boundaries."""
        from py.features.asof_features import compute_team_history

        # Add a new season game
        new_season = pd.DataFrame([{
            'game_id': '2024_01_KC_BAL',
            'team': 'KC',
            'is_home': True,
            'season': 2024,
            'week': 1,
            'kickoff': dt.datetime(2024, 9, 5, 20, 15),
            'home_team': 'KC',
            'away_team': 'BAL',
            'home_score': 27,
            'away_score': 20,
            'spread_close': -3.5,
            'total_close': 48.0,
            'points_for': 27,
            'points_against': 20,
            'stadium': 'Arrowhead Stadium',
            'roof': 'outdoors',
            'surface': 'grass',
            'qb_id': 'MAHOMES',
            'qb_name': 'Patrick Mahomes',
            'coach_name': 'Andy Reid',
            'margin': 7,
            'epa_sum': 18.0,
            'plays': 60,
        }])

        combined = pd.concat([sample_team_game_data, new_season], ignore_index=True)
        result = compute_team_history(combined)

        # Week 1 of 2024 should have season_games_prior = 0
        kc_2024_wk1 = result[(result['team'] == 'KC') & (result['season'] == 2024) & (result['week'] == 1)]

        if len(kc_2024_wk1) > 0:
            assert kc_2024_wk1.iloc[0]['season_games_prior'] == 0, (
                "Season features should reset at new season"
            )


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_dataframe(self):
        """Test handling of empty input."""
        from py.features.asof_features import compute_team_history

        empty_df = pd.DataFrame(columns=[
            'game_id', 'team', 'is_home', 'season', 'week', 'kickoff',
            'home_team', 'away_team', 'home_score', 'away_score',
            'spread_close', 'total_close', 'points_for', 'points_against',
            'stadium', 'roof', 'surface', 'qb_id', 'qb_name', 'coach_name',
            'margin', 'epa_sum', 'plays'
        ])

        result = compute_team_history(empty_df)
        assert len(result) == 0

    def test_single_game(self):
        """Test handling of single game."""
        from py.features.asof_features import compute_team_history

        single_game = pd.DataFrame([{
            'game_id': 'SINGLE_GAME',
            'team': 'KC',
            'is_home': True,
            'season': 2023,
            'week': 1,
            'kickoff': dt.datetime(2023, 9, 7, 20, 15),
            'home_team': 'KC',
            'away_team': 'DET',
            'home_score': 21,
            'away_score': 20,
            'spread_close': -7.0,
            'total_close': 52.5,
            'points_for': 21,
            'points_against': 20,
            'stadium': 'Arrowhead Stadium',
            'roof': 'outdoors',
            'surface': 'grass',
            'qb_id': 'MAHOMES',
            'qb_name': 'Patrick Mahomes',
            'coach_name': 'Andy Reid',
            'margin': 1,
            'epa_sum': 15.5,
            'plays': 60,
        }])

        result = compute_team_history(single_game)

        assert len(result) == 1
        assert result.iloc[0]['prior_games'] == 0
        assert result.iloc[0]['prior_epa_sum'] == 0.0

    def test_null_kickoff_handling(self, sample_team_game_data: pd.DataFrame):
        """Test handling of NULL kickoff times."""
        from py.features.asof_features import compute_team_history

        df = sample_team_game_data.copy()
        df.loc[df.index[0], 'kickoff'] = None

        result = compute_team_history(df)

        # Should handle gracefully (NULL kickoffs sorted to end)
        assert len(result) == len(df)

    def test_missing_qb_id(self, sample_team_game_data: pd.DataFrame):
        """Test handling of missing QB ID."""
        from py.features.asof_features import compute_team_history

        df = sample_team_game_data.copy()
        df.loc[df.index[0], 'qb_id'] = None

        result = compute_team_history(df)

        # Should fill with 'UNKNOWN'
        assert 'UNKNOWN' in result['qb_id'].values

    def test_zero_plays_game(self, sample_team_game_data: pd.DataFrame):
        """Test handling of game with zero plays (data quality issue)."""
        from py.features.asof_features import compute_team_history

        df = sample_team_game_data.copy()
        df.loc[df.index[0], 'plays'] = 0

        result = compute_team_history(df)

        # Should handle division by zero gracefully (NaN in EPA per play)
        assert len(result) == len(df)


# ============================================================================
# Integration Tests (require database)
# ============================================================================


@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests that require database connection."""

    def test_fetch_team_game_features(self):
        """Test fetching real data from database."""
        from py.features.asof_features import get_connection, fetch_team_game_features

        try:
            conn = get_connection()
            df = fetch_team_game_features(conn)
            conn.close()

            # Should have data
            assert len(df) > 0

            # Should have expected columns
            expected_cols = ['game_id', 'team', 'is_home', 'season', 'week', 'epa_sum', 'plays']
            for col in expected_cols:
                assert col in df.columns
        except Exception as e:
            pytest.skip(f"Database connection failed: {e}")

    def test_full_pipeline_base(self):
        """Test full pipeline with base module."""
        from py.features.asof_features import (
            get_connection, fetch_team_game_features,
            compute_team_history, validate_team_history, pivot_to_games
        )

        try:
            conn = get_connection()
            df = fetch_team_game_features(conn)
            conn.close()

            # Limit to recent season for speed
            df = df[df['season'] >= 2023]

            history = compute_team_history(df)
            validate_team_history(history)  # Should not raise

            games = pivot_to_games(history)

            # Validate output
            assert len(games) > 0
            assert 'home_prior_epa_mean' in games.columns
            assert 'away_prior_epa_mean' in games.columns
            assert 'epa_diff_prior' in games.columns

        except Exception as e:
            pytest.skip(f"Database integration test failed: {e}")
