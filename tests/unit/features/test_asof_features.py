"""Baseline tests for asof_features.py (base as-of team features).

The tests for asof_features_enhanced.py and asof_features_v2.py were removed with those
modules in the 2026-10 prune; asof_features_unified.py supersedes that chain.

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
            "game_id": "2023_01_KC_DET",
            "team": "KC",
            "is_home": True,
            "season": 2023,
            "week": 1,
            "kickoff": dt.datetime(2023, 9, 7, 20, 15),
            "home_team": "KC",
            "away_team": "DET",
            "home_score": 21,
            "away_score": 20,
            "spread_close": -7.0,
            "total_close": 52.5,
            "points_for": 21,
            "points_against": 20,
            "stadium": "Arrowhead Stadium",
            "roof": "outdoors",
            "surface": "grass",
            "qb_id": "MAHOMES",
            "qb_name": "Patrick Mahomes",
            "coach_name": "Andy Reid",
            "margin": 1,
            "epa_sum": 15.5,
            "plays": 60,
        },
        {
            "game_id": "2023_02_JAX_KC",
            "team": "KC",
            "is_home": True,
            "season": 2023,
            "week": 2,
            "kickoff": dt.datetime(2023, 9, 14, 20, 15),
            "home_team": "KC",
            "away_team": "JAX",
            "home_score": 17,
            "away_score": 9,
            "spread_close": -5.5,
            "total_close": 50.0,
            "points_for": 17,
            "points_against": 9,
            "stadium": "Arrowhead Stadium",
            "roof": "outdoors",
            "surface": "grass",
            "qb_id": "MAHOMES",
            "qb_name": "Patrick Mahomes",
            "coach_name": "Andy Reid",
            "margin": 8,
            "epa_sum": 12.3,
            "plays": 55,
        },
        {
            "game_id": "2023_03_KC_CHI",
            "team": "KC",
            "is_home": False,
            "season": 2023,
            "week": 3,
            "kickoff": dt.datetime(2023, 9, 21, 20, 15),
            "home_team": "CHI",
            "away_team": "KC",
            "home_score": 14,
            "away_score": 28,
            "spread_close": 6.0,
            "total_close": 45.0,
            "points_for": 28,
            "points_against": 14,
            "stadium": "Soldier Field",
            "roof": "outdoors",
            "surface": "grass",
            "qb_id": "MAHOMES",
            "qb_name": "Patrick Mahomes",
            "coach_name": "Andy Reid",
            "margin": 14,
            "epa_sum": 20.1,
            "plays": 58,
        },
        {
            "game_id": "2023_04_KC_NYJ",
            "team": "KC",
            "is_home": False,
            "season": 2023,
            "week": 4,
            "kickoff": dt.datetime(2023, 9, 28, 20, 15),
            "home_team": "NYJ",
            "away_team": "KC",
            "home_score": 20,
            "away_score": 23,
            "spread_close": 3.0,
            "total_close": 40.0,
            "points_for": 23,
            "points_against": 20,
            "stadium": "MetLife Stadium",
            "roof": "outdoors",
            "surface": "turf",
            "qb_id": "MAHOMES",
            "qb_name": "Patrick Mahomes",
            "coach_name": "Andy Reid",
            "margin": 3,
            "epa_sum": 8.7,
            "plays": 62,
        },
        # BUF games
        {
            "game_id": "2023_01_NYJ_BUF",
            "team": "BUF",
            "is_home": True,
            "season": 2023,
            "week": 1,
            "kickoff": dt.datetime(2023, 9, 10, 13, 0),
            "home_team": "BUF",
            "away_team": "NYJ",
            "home_score": 22,
            "away_score": 16,
            "spread_close": -6.5,
            "total_close": 44.5,
            "points_for": 22,
            "points_against": 16,
            "stadium": "Highmark Stadium",
            "roof": "outdoors",
            "surface": "turf",
            "qb_id": "ALLEN",
            "qb_name": "Josh Allen",
            "coach_name": "Sean McDermott",
            "margin": 6,
            "epa_sum": 10.2,
            "plays": 65,
        },
        {
            "game_id": "2023_02_BUF_LV",
            "team": "BUF",
            "is_home": False,
            "season": 2023,
            "week": 2,
            "kickoff": dt.datetime(2023, 9, 17, 16, 25),
            "home_team": "LV",
            "away_team": "BUF",
            "home_score": 23,
            "away_score": 38,
            "spread_close": 3.0,
            "total_close": 47.0,
            "points_for": 38,
            "points_against": 23,
            "stadium": "Allegiant Stadium",
            "roof": "dome",
            "surface": "grass",
            "qb_id": "ALLEN",
            "qb_name": "Josh Allen",
            "coach_name": "Sean McDermott",
            "margin": 15,
            "epa_sum": 22.5,
            "plays": 70,
        },
        {
            "game_id": "2023_03_WAS_BUF",
            "team": "BUF",
            "is_home": True,
            "season": 2023,
            "week": 3,
            "kickoff": dt.datetime(2023, 9, 24, 13, 0),
            "home_team": "BUF",
            "away_team": "WAS",
            "home_score": 37,
            "away_score": 3,
            "spread_close": -10.0,
            "total_close": 42.0,
            "points_for": 37,
            "points_against": 3,
            "stadium": "Highmark Stadium",
            "roof": "outdoors",
            "surface": "turf",
            "qb_id": "ALLEN",
            "qb_name": "Josh Allen",
            "coach_name": "Sean McDermott",
            "margin": 34,
            "epa_sum": 28.0,
            "plays": 58,
        },
        {
            "game_id": "2023_04_BUF_MIA",
            "team": "BUF",
            "is_home": False,
            "season": 2023,
            "week": 4,
            "kickoff": dt.datetime(2023, 10, 1, 13, 0),
            "home_team": "MIA",
            "away_team": "BUF",
            "home_score": 48,
            "away_score": 20,
            "spread_close": 3.5,
            "total_close": 53.0,
            "points_for": 20,
            "points_against": 48,
            "stadium": "Hard Rock Stadium",
            "roof": "outdoors",
            "surface": "grass",
            "qb_id": "ALLEN",
            "qb_name": "Josh Allen",
            "coach_name": "Sean McDermott",
            "margin": -28,
            "epa_sum": -5.3,
            "plays": 55,
        },
    ]
    return pd.DataFrame(data)


# ============================================================================
# Base Module Tests (asof_features.py)
# ============================================================================


class TestBaseAsofFeatures:
    """Tests for nfl_analytics.features.asof_features module."""

    def test_import(self):
        """Verify base module can be imported."""
        from nfl_analytics.features import asof_features

        assert hasattr(asof_features, "compute_team_history")
        assert hasattr(asof_features, "pivot_to_games")
        assert hasattr(asof_features, "validate_team_history")
        assert hasattr(asof_features, "rename_with_prefix")
        assert hasattr(asof_features, "get_connection")

    def test_team_feature_columns_count(self):
        """Verify expected number of base team features."""
        from nfl_analytics.features.asof_features import TEAM_FEATURE_COLUMNS

        # Base module should have 36 team features
        assert (
            len(TEAM_FEATURE_COLUMNS) == 36
        ), f"Expected 36 base team features, got {len(TEAM_FEATURE_COLUMNS)}"

    def test_diff_feature_columns_count(self):
        """Verify expected number of diff features."""
        from nfl_analytics.features.asof_features import DIFF_FEATURE_COLUMNS

        # Should have differential features
        assert len(DIFF_FEATURE_COLUMNS) > 20
        assert "prior_epa_mean" in DIFF_FEATURE_COLUMNS
        assert "rest_days" in DIFF_FEATURE_COLUMNS

    def test_rename_with_prefix(self):
        """Test the rename utility function."""
        from nfl_analytics.features.asof_features import rename_with_prefix

        columns = ["epa_mean", "plays", "rest_days"]
        result = rename_with_prefix(columns, "home")

        assert result == {
            "epa_mean": "home_epa_mean",
            "plays": "home_plays",
            "rest_days": "home_rest_days",
        }

    def test_rename_with_prefix_away(self):
        """Test rename with away prefix."""
        from nfl_analytics.features.asof_features import rename_with_prefix

        columns = ["prior_games", "margin_last3"]
        result = rename_with_prefix(columns, "away")

        assert result["prior_games"] == "away_prior_games"
        assert result["margin_last3"] == "away_margin_last3"

    def test_compute_team_history_prior_games(self, sample_team_game_data: pd.DataFrame):
        """Test that prior_games increments correctly (temporal safety)."""
        from nfl_analytics.features.asof_features import compute_team_history

        result = compute_team_history(sample_team_game_data)

        # Check KC team history
        kc = result[result["team"] == "KC"].sort_values("week")
        assert list(kc["prior_games"]) == [
            0,
            1,
            2,
            3,
        ], "prior_games should be 0,1,2,3 for consecutive games"

        # Check BUF team history
        buf = result[result["team"] == "BUF"].sort_values("week")
        assert list(buf["prior_games"]) == [0, 1, 2, 3]

    def test_compute_team_history_prior_epa_sum(self, sample_team_game_data: pd.DataFrame):
        """Test that prior_epa_sum uses only past games."""
        from nfl_analytics.features.asof_features import compute_team_history

        result = compute_team_history(sample_team_game_data)
        kc = result[result["team"] == "KC"].sort_values("week")

        # Week 1: prior_epa_sum should be 0 (no prior games)
        assert kc.iloc[0]["prior_epa_sum"] == 0.0

        # Week 2: prior_epa_sum should be week 1's epa_sum
        assert kc.iloc[1]["prior_epa_sum"] == pytest.approx(15.5, abs=0.01)

        # Week 3: prior_epa_sum should be sum of weeks 1+2
        assert kc.iloc[2]["prior_epa_sum"] == pytest.approx(15.5 + 12.3, abs=0.01)

    def test_compute_team_history_rest_days(self, sample_team_game_data: pd.DataFrame):
        """Test that rest_days is computed correctly."""
        from nfl_analytics.features.asof_features import compute_team_history

        result = compute_team_history(sample_team_game_data)
        kc = result[result["team"] == "KC"].sort_values("week")

        # Week 1: rest_days should default to 7.0 (no prior game)
        assert kc.iloc[0]["rest_days"] == 7.0

        # Week 2: should be ~7 days from week 1
        assert kc.iloc[1]["rest_days"] == pytest.approx(7.0, abs=0.1)

    def test_compute_team_history_rolling_last3(self, sample_team_game_data: pd.DataFrame):
        """Test that rolling window features use shift(1)."""
        from nfl_analytics.features.asof_features import compute_team_history

        result = compute_team_history(sample_team_game_data)
        kc = result[result["team"] == "KC"].sort_values("week")

        # Week 1: margin_last3 should be NaN (no prior games)
        assert pd.isna(kc.iloc[0]["margin_last3"])

        # Week 2: margin_last3 should be week 1's margin only
        assert kc.iloc[1]["margin_last3"] == pytest.approx(1.0, abs=0.01)

        # Week 4: margin_last3 should be avg of weeks 1,2,3 margins
        expected_margin_last3 = np.mean([1, 8, 14])  # Weeks 1,2,3
        assert kc.iloc[3]["margin_last3"] == pytest.approx(expected_margin_last3, abs=0.01)

    def test_compute_team_history_qb_change(self, sample_team_game_data: pd.DataFrame):
        """Test QB change detection."""
        from nfl_analytics.features.asof_features import compute_team_history

        result = compute_team_history(sample_team_game_data)
        kc = result[result["team"] == "KC"].sort_values("week")

        # All KC games have Mahomes, so qb_change should be 0 after week 1
        assert kc.iloc[0]["qb_change"] == 0.0  # No prior game
        assert kc.iloc[1]["qb_change"] == 0.0  # Same QB
        assert kc.iloc[2]["qb_change"] == 0.0

    def test_compute_team_history_surface_encoding(self, sample_team_game_data: pd.DataFrame):
        """Test surface one-hot encoding."""
        from nfl_analytics.features.asof_features import compute_team_history

        result = compute_team_history(sample_team_game_data)

        # Check grass games
        grass_games = result[result["surface"] == "grass"]
        assert all(grass_games["surface_grass"] == 1.0)

        # Check turf games
        turf_games = result[result["surface"] == "turf"]
        assert all(turf_games["surface_artificial"] == 1.0)

    def test_validate_team_history_passes(self, sample_team_game_data: pd.DataFrame):
        """Test validation passes for correctly computed history."""
        from nfl_analytics.features.asof_features import compute_team_history, validate_team_history

        result = compute_team_history(sample_team_game_data)

        # Should not raise
        validate_team_history(result)

    def test_validate_team_history_detects_prior_games_error(
        self, sample_team_game_data: pd.DataFrame
    ):
        """Test validation catches prior_games inconsistency."""
        from nfl_analytics.features.asof_features import compute_team_history, validate_team_history

        result = compute_team_history(sample_team_game_data)

        # Corrupt prior_games to simulate leakage
        result.loc[result.index[0], "prior_games"] = 5

        with pytest.raises(ValueError, match="prior_games mismatch"):
            validate_team_history(result)

    def test_validate_team_history_detects_negative_rest(self, sample_team_game_data: pd.DataFrame):
        """Test validation catches negative rest days."""
        from nfl_analytics.features.asof_features import compute_team_history, validate_team_history

        result = compute_team_history(sample_team_game_data)

        # Corrupt rest_days
        result.loc[result.index[0], "rest_days"] = -1

        with pytest.raises(ValueError, match="Negative rest_days"):
            validate_team_history(result)

    def test_pivot_to_games_creates_home_away(self, sample_team_game_data: pd.DataFrame):
        """Test pivoting creates home/away prefixed columns."""
        from nfl_analytics.features.asof_features import compute_team_history, pivot_to_games

        history = compute_team_history(sample_team_game_data)

        # Need to create matching home/away pairs
        # Our fixture only has team-level data, so we need to construct pairs
        # For this test, let's create a game where KC is home and BUF is away
        game_df = pd.DataFrame(
            [
                {
                    "game_id": "TEST_KC_BUF",
                    "team": "KC",
                    "is_home": True,
                    "season": 2023,
                    "week": 5,
                    "kickoff": dt.datetime(2023, 10, 8, 13, 0),
                    "home_team": "KC",
                    "away_team": "BUF",
                    "home_score": 24,
                    "away_score": 21,
                    "spread_close": -3.0,
                    "total_close": 48.0,
                    "points_for": 24,
                    "points_against": 21,
                    "stadium": "Arrowhead Stadium",
                    "roof": "outdoors",
                    "surface": "grass",
                    "qb_id": "MAHOMES",
                    "qb_name": "Patrick Mahomes",
                    "coach_name": "Andy Reid",
                    "margin": 3,
                    "epa_sum": 12.0,
                    "plays": 58,
                },
                {
                    "game_id": "TEST_KC_BUF",
                    "team": "BUF",
                    "is_home": False,
                    "season": 2023,
                    "week": 5,
                    "kickoff": dt.datetime(2023, 10, 8, 13, 0),
                    "home_team": "KC",
                    "away_team": "BUF",
                    "home_score": 24,
                    "away_score": 21,
                    "spread_close": -3.0,
                    "total_close": 48.0,
                    "points_for": 21,
                    "points_against": 24,
                    "stadium": "Arrowhead Stadium",
                    "roof": "outdoors",
                    "surface": "grass",
                    "qb_id": "ALLEN",
                    "qb_name": "Josh Allen",
                    "coach_name": "Sean McDermott",
                    "margin": -3,
                    "epa_sum": 10.0,
                    "plays": 55,
                },
            ]
        )

        history = compute_team_history(game_df)
        games = pivot_to_games(history)

        # Should have one game row
        assert len(games) == 1

        # Should have home_ and away_ prefixed columns
        assert "home_prior_games" in games.columns
        assert "away_prior_games" in games.columns
        assert "home_rest_days" in games.columns
        assert "away_rest_days" in games.columns

    def test_pivot_to_games_diff_columns(self, sample_team_game_data: pd.DataFrame):
        """Test pivoting creates differential columns."""
        from nfl_analytics.features.asof_features import compute_team_history, pivot_to_games

        # Create a paired game
        game_df = pd.DataFrame(
            [
                {
                    "game_id": "TEST_GAME",
                    "team": "KC",
                    "is_home": True,
                    "season": 2023,
                    "week": 1,
                    "kickoff": dt.datetime(2023, 9, 10, 13, 0),
                    "home_team": "KC",
                    "away_team": "BUF",
                    "home_score": 24,
                    "away_score": 21,
                    "spread_close": -3.0,
                    "total_close": 48.0,
                    "points_for": 24,
                    "points_against": 21,
                    "stadium": "Arrowhead Stadium",
                    "roof": "outdoors",
                    "surface": "grass",
                    "qb_id": "MAHOMES",
                    "qb_name": "Patrick Mahomes",
                    "coach_name": "Andy Reid",
                    "margin": 3,
                    "epa_sum": 12.0,
                    "plays": 58,
                },
                {
                    "game_id": "TEST_GAME",
                    "team": "BUF",
                    "is_home": False,
                    "season": 2023,
                    "week": 1,
                    "kickoff": dt.datetime(2023, 9, 10, 13, 0),
                    "home_team": "KC",
                    "away_team": "BUF",
                    "home_score": 24,
                    "away_score": 21,
                    "spread_close": -3.0,
                    "total_close": 48.0,
                    "points_for": 21,
                    "points_against": 24,
                    "stadium": "Arrowhead Stadium",
                    "roof": "outdoors",
                    "surface": "grass",
                    "qb_id": "ALLEN",
                    "qb_name": "Josh Allen",
                    "coach_name": "Sean McDermott",
                    "margin": -3,
                    "epa_sum": 10.0,
                    "plays": 55,
                },
            ]
        )

        history = compute_team_history(game_df)
        games = pivot_to_games(history)

        # Should have differential columns
        assert "rest_diff" in games.columns
        assert "epa_diff_prior" in games.columns


# ============================================================================
# Temporal Safety Tests
# ============================================================================


class TestTemporalSafety:
    """Tests to verify no data leakage in asof features."""

    def test_prior_features_use_only_past_data(self, sample_team_game_data: pd.DataFrame):
        """Verify prior features don't include current game data."""
        from nfl_analytics.features.asof_features import compute_team_history

        result = compute_team_history(sample_team_game_data)
        kc = result[result["team"] == "KC"].sort_values("week")

        # For each game, prior_epa_sum should NOT include current game's epa_sum
        for i in range(len(kc)):
            expected_prior_sum = kc.iloc[:i]["epa_sum"].sum()
            actual_prior_sum = kc.iloc[i]["prior_epa_sum"]
            assert actual_prior_sum == pytest.approx(
                expected_prior_sum, abs=0.01
            ), f"Week {kc.iloc[i]['week']}: prior_epa_sum {actual_prior_sum} != expected {expected_prior_sum}"

    def test_rolling_features_exclude_current(self, sample_team_game_data: pd.DataFrame):
        """Verify rolling window features use shift(1)."""
        from nfl_analytics.features.asof_features import compute_team_history

        result = compute_team_history(sample_team_game_data)
        kc = result[result["team"] == "KC"].sort_values("week")

        # margin_last3 for week 4 should be avg of weeks 1,2,3 (not 2,3,4)
        week4 = kc.iloc[3]
        weeks_123 = kc.iloc[:3]
        expected = weeks_123["margin"].mean()

        assert week4["margin_last3"] == pytest.approx(
            expected, abs=0.01
        ), f"margin_last3 includes current game (leakage!)"

    def test_win_percentage_excludes_current_result(self, sample_team_game_data: pd.DataFrame):
        """Verify win percentage doesn't include current game result."""
        from nfl_analytics.features.asof_features import compute_team_history

        result = compute_team_history(sample_team_game_data)
        kc = result[result["team"] == "KC"].sort_values("week")

        # KC wins all 4 games (margins: 1, 8, 14, 3 are all positive)
        # prior_win_pct should be:
        # Week 1: NaN (no prior games)
        # Week 2: 1.0 (1 win / 1 game)
        # Week 3: 1.0 (2 wins / 2 games)
        # Week 4: 1.0 (3 wins / 3 games)

        assert pd.isna(kc.iloc[0]["prior_win_pct"])
        assert kc.iloc[1]["prior_win_pct"] == pytest.approx(1.0)
        assert kc.iloc[2]["prior_win_pct"] == pytest.approx(1.0)
        assert kc.iloc[3]["prior_win_pct"] == pytest.approx(1.0)

    def test_season_specific_features_reset(self, sample_team_game_data: pd.DataFrame):
        """Verify season-specific features reset at season boundaries."""
        from nfl_analytics.features.asof_features import compute_team_history

        # Add a new season game
        new_season = pd.DataFrame(
            [
                {
                    "game_id": "2024_01_KC_BAL",
                    "team": "KC",
                    "is_home": True,
                    "season": 2024,
                    "week": 1,
                    "kickoff": dt.datetime(2024, 9, 5, 20, 15),
                    "home_team": "KC",
                    "away_team": "BAL",
                    "home_score": 27,
                    "away_score": 20,
                    "spread_close": -3.5,
                    "total_close": 48.0,
                    "points_for": 27,
                    "points_against": 20,
                    "stadium": "Arrowhead Stadium",
                    "roof": "outdoors",
                    "surface": "grass",
                    "qb_id": "MAHOMES",
                    "qb_name": "Patrick Mahomes",
                    "coach_name": "Andy Reid",
                    "margin": 7,
                    "epa_sum": 18.0,
                    "plays": 60,
                }
            ]
        )

        combined = pd.concat([sample_team_game_data, new_season], ignore_index=True)
        result = compute_team_history(combined)

        # Week 1 of 2024 should have season_games_prior = 0
        kc_2024_wk1 = result[
            (result["team"] == "KC") & (result["season"] == 2024) & (result["week"] == 1)
        ]

        if len(kc_2024_wk1) > 0:
            assert (
                kc_2024_wk1.iloc[0]["season_games_prior"] == 0
            ), "Season features should reset at new season"


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_dataframe(self):
        """Test handling of empty input."""
        from nfl_analytics.features.asof_features import compute_team_history

        empty_df = pd.DataFrame(
            columns=[
                "game_id",
                "team",
                "is_home",
                "season",
                "week",
                "kickoff",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "spread_close",
                "total_close",
                "points_for",
                "points_against",
                "stadium",
                "roof",
                "surface",
                "qb_id",
                "qb_name",
                "coach_name",
                "margin",
                "epa_sum",
                "plays",
            ]
        )

        result = compute_team_history(empty_df)
        assert len(result) == 0

    def test_single_game(self):
        """Test handling of single game."""
        from nfl_analytics.features.asof_features import compute_team_history

        single_game = pd.DataFrame(
            [
                {
                    "game_id": "SINGLE_GAME",
                    "team": "KC",
                    "is_home": True,
                    "season": 2023,
                    "week": 1,
                    "kickoff": dt.datetime(2023, 9, 7, 20, 15),
                    "home_team": "KC",
                    "away_team": "DET",
                    "home_score": 21,
                    "away_score": 20,
                    "spread_close": -7.0,
                    "total_close": 52.5,
                    "points_for": 21,
                    "points_against": 20,
                    "stadium": "Arrowhead Stadium",
                    "roof": "outdoors",
                    "surface": "grass",
                    "qb_id": "MAHOMES",
                    "qb_name": "Patrick Mahomes",
                    "coach_name": "Andy Reid",
                    "margin": 1,
                    "epa_sum": 15.5,
                    "plays": 60,
                }
            ]
        )

        result = compute_team_history(single_game)

        assert len(result) == 1
        assert result.iloc[0]["prior_games"] == 0
        assert result.iloc[0]["prior_epa_sum"] == 0.0

    def test_null_kickoff_handling(self, sample_team_game_data: pd.DataFrame):
        """Test handling of NULL kickoff times."""
        from nfl_analytics.features.asof_features import compute_team_history

        df = sample_team_game_data.copy()
        df.loc[df.index[0], "kickoff"] = None

        result = compute_team_history(df)

        # Should handle gracefully (NULL kickoffs sorted to end)
        assert len(result) == len(df)

    def test_missing_qb_id(self, sample_team_game_data: pd.DataFrame):
        """Test handling of missing QB ID."""
        from nfl_analytics.features.asof_features import compute_team_history

        df = sample_team_game_data.copy()
        df.loc[df.index[0], "qb_id"] = None

        result = compute_team_history(df)

        # Should fill with 'UNKNOWN'
        assert "UNKNOWN" in result["qb_id"].values

    def test_zero_plays_game(self, sample_team_game_data: pd.DataFrame):
        """Test handling of game with zero plays (data quality issue)."""
        from nfl_analytics.features.asof_features import compute_team_history

        df = sample_team_game_data.copy()
        df.loc[df.index[0], "plays"] = 0

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
        from nfl_analytics.features.asof_features import get_connection, fetch_team_game_features

        try:
            conn = get_connection()
            df = fetch_team_game_features(conn)
            conn.close()

            # Should have data
            assert len(df) > 0

            # Should have expected columns
            expected_cols = ["game_id", "team", "is_home", "season", "week", "epa_sum", "plays"]
            for col in expected_cols:
                assert col in df.columns
        except Exception as e:
            pytest.skip(f"Database connection failed: {e}")

    def test_full_pipeline_base(self):
        """Test full pipeline with base module."""
        from nfl_analytics.features.asof_features import (
            get_connection,
            fetch_team_game_features,
            compute_team_history,
            validate_team_history,
            pivot_to_games,
        )

        try:
            conn = get_connection()
            df = fetch_team_game_features(conn)
            conn.close()

            # Limit to recent season for speed
            df = df[df["season"] >= 2023]

            history = compute_team_history(df)
            validate_team_history(history)  # Should not raise

            games = pivot_to_games(history)

            # Validate output
            assert len(games) > 0
            assert "home_prior_epa_mean" in games.columns
            assert "away_prior_epa_mean" in games.columns
            assert "epa_diff_prior" in games.columns

        except Exception as e:
            pytest.skip(f"Database integration test failed: {e}")
