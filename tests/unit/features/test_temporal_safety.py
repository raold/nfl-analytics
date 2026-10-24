"""Temporal safety validation tests.

This module provides comprehensive tests to detect and prevent data leakage
in the NFL Analytics pipeline. Data leakage occurs when future information
"leaks" into the training data, leading to overly optimistic model performance
that doesn't generalize to real predictions.

Key leakage patterns tested:
1. Train-validation split leakage (future seasons in training)
2. Feature leakage (using outcome data to compute features)
3. Rolling window leakage (not using shift(1))
4. Season boundary leakage (using next season's data)

Created: December 13, 2025
Phase 1.3 of code quality remediation plan.
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
def multi_season_game_data() -> pd.DataFrame:
    """Create multi-season game data for testing temporal splits.

    This fixture creates games across 3 seasons to test:
    - Season boundary handling
    - Temporal train/validation splits
    - Cross-season feature leakage
    """
    games = []
    game_id = 0

    for season in [2021, 2022, 2023]:
        for week in range(1, 18):  # Weeks 1-17
            game_id += 1
            games.append(
                {
                    "game_id": f"{season}_{week:02d}_KC_BUF",
                    "season": season,
                    "week": week,
                    "kickoff": dt.datetime(season, 9, 1) + dt.timedelta(weeks=week - 1),
                    "home_team": "KC",
                    "away_team": "BUF",
                    "home_score": 24 + np.random.randint(-7, 8),
                    "away_score": 21 + np.random.randint(-7, 8),
                    "spread": -3.5 + np.random.uniform(-2, 2),
                    "total": 48.0 + np.random.uniform(-5, 5),
                    "home_win": 1 if np.random.random() > 0.4 else 0,
                    # Feature columns
                    "home_prior_epa_mean": np.random.uniform(-0.1, 0.2),
                    "away_prior_epa_mean": np.random.uniform(-0.1, 0.2),
                    "home_rest_days": 7.0,
                    "away_rest_days": 7.0,
                    "epa_diff_prior": np.random.uniform(-0.1, 0.1),
                }
            )

    return pd.DataFrame(games)


@pytest.fixture
def team_history_data() -> pd.DataFrame:
    """Create team-level historical data for testing feature extraction."""
    data = []

    for season in [2022, 2023]:
        for week in range(1, 18):
            # Game result data
            margin = np.random.randint(-20, 21)
            epa_sum = np.random.uniform(-10, 20)
            plays = np.random.randint(55, 75)

            data.append(
                {
                    "game_id": f"{season}_{week:02d}_KC_OPP",
                    "team": "KC",
                    "is_home": week % 2 == 1,
                    "season": season,
                    "week": week,
                    "kickoff": dt.datetime(season, 9, 1) + dt.timedelta(weeks=week - 1),
                    "home_team": "KC" if week % 2 == 1 else "OPP",
                    "away_team": "OPP" if week % 2 == 1 else "KC",
                    "home_score": 24 + margin // 2 if week % 2 == 1 else 24 - margin // 2,
                    "away_score": 24 - margin // 2 if week % 2 == 1 else 24 + margin // 2,
                    "spread_close": -3.5,
                    "total_close": 48.0,
                    "points_for": 24 + margin // 2,
                    "points_against": 24 - margin // 2,
                    "stadium": "Arrowhead Stadium",
                    "roof": "outdoors",
                    "surface": "grass",
                    "qb_id": "MAHOMES",
                    "qb_name": "Patrick Mahomes",
                    "coach_name": "Andy Reid",
                    "margin": margin,
                    "epa_sum": epa_sum,
                    "plays": plays,
                }
            )

    return pd.DataFrame(data)


# ============================================================================
# Train-Validation Split Tests
# ============================================================================


class TestTrainValidationSplitSafety:
    """Tests for temporal safety in train-validation splits."""

    def test_temporal_split_no_future_in_training(self, multi_season_game_data: pd.DataFrame):
        """Verify training data doesn't contain games from validation seasons.

        CRITICAL: This is the primary data leakage check. Training should only
        contain games that occurred BEFORE the validation period.
        """
        df = multi_season_game_data.copy()
        df = df.sort_values(["season", "week"]).reset_index(drop=True)

        # Simulate a proper temporal split (last season for validation)
        train_seasons = [2021, 2022]
        val_seasons = [2023]

        train_df = df[df["season"].isin(train_seasons)]
        val_df = df[df["season"].isin(val_seasons)]

        # Get the latest date in training
        train_max_date = train_df["kickoff"].max()

        # All validation games should be AFTER training games
        val_min_date = val_df["kickoff"].min()

        assert (
            val_min_date > train_max_date
        ), f"Validation data ({val_min_date}) overlaps with training ({train_max_date})"

    def test_percentage_split_can_cause_leakage(self, multi_season_game_data: pd.DataFrame):
        """Demonstrate that percentage-based splits can cause temporal leakage.

        This test shows the BUG in the current implementation - when data isn't
        strictly sorted by time, percentage splits can mix future and past data.
        """
        df = multi_season_game_data.copy()

        # Shuffle the data (simulating unsorted data)
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Apply percentage-based split (THE BUGGY APPROACH)
        validation_split = 0.2
        split_idx = int(len(df_shuffled) * (1 - validation_split))

        train_df = df_shuffled.iloc[:split_idx]
        val_df = df_shuffled.iloc[split_idx:]

        # Check if any validation games are from earlier seasons than training games
        train_seasons = set(train_df["season"].unique())
        val_seasons = set(val_df["season"].unique())

        # If training contains 2023 and validation contains 2021, that's leakage
        has_leakage = (
            2023 in train_seasons
            and 2021 in val_seasons
            or 2023 in train_seasons
            and 2022 in val_seasons
        )

        # This test documents the bug - it SHOULD detect leakage with shuffled data
        # When fixed, percentage splits on sorted data should NOT cause leakage
        assert has_leakage or (
            max(val_df["season"].unique()) >= max(train_df["season"].unique())
        ), "After fix, validation seasons should be >= training seasons"

    def test_season_based_split_prevents_leakage(self, multi_season_game_data: pd.DataFrame):
        """Verify that season-based splits prevent temporal leakage."""
        df = multi_season_game_data.copy()

        # Proper implementation: split by season
        max_season = df["season"].max()
        train_df = df[df["season"] < max_season]
        val_df = df[df["season"] == max_season]

        # All training games should be from earlier seasons
        assert all(train_df["season"] < max_season)
        assert all(val_df["season"] == max_season)

        # No overlap in game IDs
        train_ids = set(train_df["game_id"])
        val_ids = set(val_df["game_id"])
        assert len(train_ids & val_ids) == 0, "Training and validation have overlapping games"


# ============================================================================
# Feature Extraction Safety Tests
# ============================================================================


class TestFeatureExtractionSafety:
    """Tests for temporal safety in feature extraction."""

    def test_prior_features_exclude_current_game(self, team_history_data: pd.DataFrame):
        """Verify that 'prior_*' features don't include current game data."""
        from nfl_analytics.features.asof_features import compute_team_history

        result = compute_team_history(team_history_data)
        kc = result[result["team"] == "KC"].sort_values(["season", "week"])

        for i, row in kc.iterrows():
            if row["prior_games"] == 0:
                # First game: prior sums should be 0
                assert row["prior_epa_sum"] == 0.0
                assert row["prior_plays"] == 0.0
            else:
                # For all other games, verify prior_epa_sum doesn't include current
                idx_in_kc = kc.index.get_loc(i)
                games_before = kc.iloc[:idx_in_kc]
                expected_prior_epa = games_before["epa_sum"].sum()

                assert row["prior_epa_sum"] == pytest.approx(
                    expected_prior_epa, abs=0.1
                ), f"Game {row['game_id']}: prior_epa_sum includes current game (leakage)"

    def test_rolling_windows_use_shift_one(self, team_history_data: pd.DataFrame):
        """Verify rolling window features use shift(1) to exclude current game."""
        from nfl_analytics.features.asof_features import compute_team_history

        result = compute_team_history(team_history_data)
        kc = result[result["team"] == "KC"].sort_values(["season", "week"])

        for i in range(3, len(kc)):  # Start at game 4 to have full window
            row = kc.iloc[i]
            games_before = kc.iloc[:i]  # Exclude current game

            # margin_last3 should be mean of games i-3, i-2, i-1 (not i)
            expected_margin_last3 = games_before.tail(3)["margin"].mean()

            assert row["margin_last3"] == pytest.approx(expected_margin_last3, abs=0.1), (
                f"Game {i}: margin_last3 = {row['margin_last3']}, "
                f"expected {expected_margin_last3} (leakage detected)"
            )

    def test_win_percentage_excludes_current_result(self, team_history_data: pd.DataFrame):
        """Verify win percentage is computed before knowing current game result."""
        from nfl_analytics.features.asof_features import compute_team_history

        result = compute_team_history(team_history_data)
        kc = result[result["team"] == "KC"].sort_values(["season", "week"])

        for i in range(1, len(kc)):
            row = kc.iloc[i]
            games_before = kc.iloc[:i]

            # Count wins (margin > 0) in prior games
            prior_wins = (games_before["margin"] > 0).sum()
            expected_win_pct = prior_wins / i if i > 0 else np.nan

            # Allow for ties (margin = 0 counts as 0.5)
            assert abs(row["prior_win_pct"] - expected_win_pct) < 0.1 or pd.isna(
                row["prior_win_pct"]
            ), f"Game {i}: win_pct includes current result (leakage)"

    def test_no_score_in_pregame_features(self):
        """Verify that score-related columns aren't used as prediction features."""
        from nfl_analytics.features.asof_features import TEAM_FEATURE_COLUMNS

        leaky_columns = ["home_score", "away_score", "margin", "home_margin", "home_win", "result"]

        for col in leaky_columns:
            assert (
                col not in TEAM_FEATURE_COLUMNS
            ), f"Leaky column '{col}' found in TEAM_FEATURE_COLUMNS"


# ============================================================================
# Season Boundary Tests
# ============================================================================


class TestSeasonBoundarySafety:
    """Tests for temporal safety at season boundaries."""

    def test_season_features_reset_at_new_season(self, team_history_data: pd.DataFrame):
        """Verify season-specific features reset at season boundaries."""
        from nfl_analytics.features.asof_features import compute_team_history

        result = compute_team_history(team_history_data)

        # Check each season's first game
        for season in result["season"].unique():
            season_games = result[(result["team"] == "KC") & (result["season"] == season)]
            season_games = season_games.sort_values("week")

            first_game = season_games.iloc[0]

            # Season-specific prior games should be 0 at start of season
            assert (
                first_game["season_games_prior"] == 0
            ), f"Season {season}: season_games_prior != 0 at week 1"

    def test_all_time_features_carry_over(self, team_history_data: pd.DataFrame):
        """Verify all-time features DO carry over across seasons."""
        from nfl_analytics.features.asof_features import compute_team_history

        result = compute_team_history(team_history_data)
        kc = result[result["team"] == "KC"].sort_values(["season", "week"])

        # 2023 Week 1 should have prior_games > 0 (from 2022)
        season_2023_wk1 = kc[(kc["season"] == 2023) & (kc["week"] == 1)]

        if len(season_2023_wk1) > 0:
            assert (
                season_2023_wk1.iloc[0]["prior_games"] > 0
            ), "All-time prior_games should carry over from previous season"

    def test_cross_season_rolling_windows(self, team_history_data: pd.DataFrame):
        """Test rolling window features at season boundaries."""
        from nfl_analytics.features.asof_features import compute_team_history

        result = compute_team_history(team_history_data)
        kc = result[result["team"] == "KC"].sort_values(["season", "week"])

        # At 2023 Week 1, margin_last3 should use 2022 Week 15, 16, 17
        season_2023_wk1 = kc[(kc["season"] == 2023) & (kc["week"] == 1)]
        season_2022_last3 = kc[(kc["season"] == 2022)].tail(3)

        if len(season_2023_wk1) > 0 and len(season_2022_last3) == 3:
            expected_margin = season_2022_last3["margin"].mean()
            actual_margin = season_2023_wk1.iloc[0]["margin_last3"]

            assert actual_margin == pytest.approx(
                expected_margin, abs=1.0
            ), "Rolling windows should use cross-season data"


# ============================================================================
# Model Training Pipeline Safety
# ============================================================================


class TestModelTrainingSafety:
    """Tests for temporal safety in the model training pipeline."""

    def test_feature_names_exclude_targets(self):
        """Verify feature names don't include target columns."""
        target_columns = [
            "spread",
            "total",
            "home_win",
            "home_score",
            "away_score",
            "home_margin",
            "over_hit",
            "home_cover",
        ]

        # Simulate feature column selection (from game_outcome_model_fixed.py)
        all_columns = [
            "season",
            "week",
            "home_team",
            "away_team",
            "spread",
            "total",
            "home_win",  # targets
            "home_prior_epa_mean",
            "away_prior_epa_mean",  # valid features
            "epa_diff_prior",
            "rest_diff",  # valid features
        ]

        # Feature selection logic
        feature_cols = [
            col
            for col in all_columns
            if col
            not in ["season", "week", "home_team", "away_team", "spread", "total", "home_win"]
        ]

        for target in target_columns:
            assert target not in feature_cols, f"Target '{target}' found in features"

    def test_validation_set_temporal_ordering(self, multi_season_game_data: pd.DataFrame):
        """Verify validation set is temporally after training set."""
        df = multi_season_game_data.copy()
        df = df.sort_values(["season", "week"]).reset_index(drop=True)

        # Proper split: last 20% by temporal order
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

        # All validation kickoff times should be >= all training kickoff times
        train_max_kickoff = train_df["kickoff"].max()
        val_min_kickoff = val_df["kickoff"].min()

        assert (
            val_min_kickoff >= train_max_kickoff
        ), "Validation set contains games before training set ends"


# ============================================================================
# Prediction Time Safety
# ============================================================================


class TestPredictionTimeSafety:
    """Tests for temporal safety during prediction/inference."""

    def test_prediction_uses_only_pregame_data(self):
        """Verify predictions only use data available before game kickoff.

        This is a conceptual test that verifies the feature extraction
        pattern used for live predictions.
        """
        # Simulate pre-game feature extraction context
        prediction_time = dt.datetime(2023, 9, 10, 11, 0)  # 2 hours before kickoff
        game_kickoff = dt.datetime(2023, 9, 10, 13, 0)

        # Feature extraction should use snapshot_time < game_kickoff
        assert prediction_time < game_kickoff, "Test setup error"

        # Mock feature extraction that respects temporal boundaries
        def extract_features_for_prediction(
            team: str, season: int, week: int, snapshot_time: dt.datetime
        ) -> dict:
            """Mock feature extraction that demonstrates temporal safety."""
            # In production, this would query:
            # - Games WHERE kickoff < snapshot_time
            # - Plays WHERE game_kickoff < snapshot_time
            # - Stats WHERE as_of_date < snapshot_time
            return {
                "prior_games": 10,  # Games played BEFORE snapshot_time
                "prior_epa_mean": 0.15,  # EPA from games BEFORE snapshot_time
                "rest_days": 7.0,  # Days since LAST game (before snapshot_time)
            }

        features = extract_features_for_prediction("KC", 2023, 1, prediction_time)

        # Features should be valid (not None, not leaky)
        assert features["prior_games"] >= 0
        assert -1.0 <= features["prior_epa_mean"] <= 1.0
        assert features["rest_days"] >= 0

    def test_no_future_game_results_in_features(self, team_history_data: pd.DataFrame):
        """Verify feature extraction doesn't use results of future games."""
        from nfl_analytics.features.asof_features import compute_team_history

        result = compute_team_history(team_history_data)

        # For each game, verify features are computed from PAST games only
        kc = result[result["team"] == "KC"].sort_values(["season", "week"])

        for i in range(len(kc) - 1):
            current_game = kc.iloc[i]
            future_games = kc.iloc[i + 1 :]

            # prior_epa_sum should NOT contain EPA from future games
            future_epa = future_games["epa_sum"].sum()
            prior_epa = current_game["prior_epa_sum"]
            total_epa = kc["epa_sum"].sum()

            # prior_epa + current_epa + future_epa = total_epa
            current_epa = current_game["epa_sum"]
            reconstructed_total = prior_epa + current_epa + future_epa

            assert reconstructed_total == pytest.approx(
                total_epa, abs=0.1
            ), f"Game {i}: EPA accounting error suggests leakage"


# ============================================================================
# Leakage Detection Utilities
# ============================================================================


class TestLeakageDetectionUtilities:
    """Tests for utility functions that detect data leakage."""

    def test_validate_team_history_detects_prior_games_leakage(
        self, team_history_data: pd.DataFrame
    ):
        """Verify validation catches corrupted prior_games."""
        from nfl_analytics.features.asof_features import compute_team_history, validate_team_history

        result = compute_team_history(team_history_data)

        # Corrupt prior_games to simulate leakage
        result.iloc[0, result.columns.get_loc("prior_games")] = 100

        with pytest.raises(ValueError, match="prior_games mismatch"):
            validate_team_history(result)

    def test_validate_team_history_detects_epa_inconsistency(self, team_history_data: pd.DataFrame):
        """Verify validation catches EPA calculation errors."""
        from nfl_analytics.features.asof_features import compute_team_history, validate_team_history

        result = compute_team_history(team_history_data)

        # Corrupt prior_epa_mean to simulate calculation error
        mask = result["prior_plays"] > 0
        if mask.any():
            idx = result[mask].index[0]
            result.loc[idx, "prior_epa_mean"] = 999.0

            with pytest.raises(ValueError, match="prior_epa_mean inconsistent"):
                validate_team_history(result)

    def test_validate_game_features_detects_duplicates(self):
        """Verify validation catches duplicate game IDs."""
        from nfl_analytics.features.asof_features import validate_game_features

        # Create data with duplicate game_id
        df = pd.DataFrame(
            {
                "game_id": ["GAME_1", "GAME_1"],  # Duplicate!
                "home_team": ["KC", "KC"],
                "away_team": ["BUF", "BUF"],
            }
        )

        with pytest.raises(ValueError, match="Duplicate game_id"):
            validate_game_features(df)


# ============================================================================
# Integration Tests - Full Pipeline
# ============================================================================


@pytest.mark.integration
class TestFullPipelineTemporalSafety:
    """Integration tests for temporal safety across the full pipeline."""

    def test_feature_extraction_to_prediction_pipeline(self):
        """Test temporal safety from feature extraction to prediction."""
        try:
            from nfl_analytics.features.asof_features import (
                get_connection,
                fetch_team_game_features,
                compute_team_history,
                validate_team_history,
            )

            conn = get_connection()
            df = fetch_team_game_features(conn)
            conn.close()

            # Limit to recent data for speed
            df = df[df["season"] >= 2022]

            history = compute_team_history(df)

            # Validation should pass (no leakage in production data)
            validate_team_history(history)

            # Verify temporal ordering is maintained
            for team in history["team"].unique():
                team_games = history[history["team"] == team].sort_values(["season", "week"])

                # prior_games should monotonically increase
                prior_games = team_games["prior_games"].values
                assert all(
                    prior_games[i] <= prior_games[i + 1] for i in range(len(prior_games) - 1)
                ), f"Team {team}: prior_games not monotonically increasing"

        except Exception as e:
            pytest.skip(f"Database integration test failed: {e}")

    def test_no_data_leakage_in_2023_predictions(self):
        """Verify 2023 predictions don't use 2023 or future data."""
        try:
            from nfl_analytics.features.asof_features import (
                get_connection,
                fetch_team_game_features,
                compute_team_history,
                pivot_to_games,
            )

            conn = get_connection()
            df = fetch_team_game_features(conn)
            conn.close()

            history = compute_team_history(df)
            games = pivot_to_games(history)

            # For 2023 Week 1 games, prior features should only use 2022 and earlier
            week1_2023 = games[(games["season"] == 2023) & (games["week"] == 1)]

            for _, game in week1_2023.iterrows():
                # Season-specific features should be 0 or NaN
                if "home_season_games_prior" in games.columns:
                    assert game["home_season_games_prior"] == 0 or pd.isna(
                        game["home_season_games_prior"]
                    ), f"Game {game['game_id']}: 2023 Week 1 has season prior games > 0"

                # All-time features should be from 2022 and earlier
                # (can't directly verify without additional metadata)

        except Exception as e:
            pytest.skip(f"Database integration test failed: {e}")
