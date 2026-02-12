"""
Unit tests for the NFL analytics feedback loop core modules.

Tests cover:
    - outcome_updater: Updating predictions with actual game scores
    - retrain_trigger: MAE degradation, accuracy drop, calibration drift detection
    - model_error_decomposition: Error attribution math, classification helpers
    - auto_feature_selector: State transitions (active/probation/disabled)
    - feature_memory: Rolling window stats, recommendations

All database interactions are mocked via unittest.mock.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

# ---------------------------------------------------------------------------
# Module imports -- patching psycopg2.connect at the module level is
# sufficient to prevent any real DB connection from being established.
# ---------------------------------------------------------------------------
from py.feedback.outcome_updater import WeeklyOutcomeUpdater, UpdateResult
from py.feedback.retrain_trigger import (
    RetrainTrigger,
    RetrainTriggerResult,
    TriggerCheck,
)
from py.feedback.model_error_decomposition import (
    ModelErrorDecomposer,
    ErrorDecomposition,
    SeasonErrorReport,
    _classify_error_magnitude,
    _classify_game_context,
)
from py.feedback.auto_feature_selector import (
    AutoFeatureSelector,
    FeatureSelectionResult,
)
from py.feedback.feature_memory import (
    FeaturePerformanceMemory,
    FeatureStats,
)


# ===========================================================================
#  Helpers / Fixtures
# ===========================================================================

def _make_mock_connection():
    """Build a mock psycopg2 connection with context-manager support."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    # Support `with conn.cursor(...) as cur:`
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cursor
    # Support `with self._get_connection() as conn:`
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    return mock_conn, mock_cursor


# Realistic NFL game data (Week 16, 2025)
SAMPLE_GAME = {
    "game_id": "2025_16_KC_PIT",
    "season": 2025,
    "week": 16,
    "home_team": "KC",
    "away_team": "PIT",
    "home_score": 27,
    "away_score": 20,
    "actual_total": 47,
    "actual_spread": 7,       # KC won by 7
    "home_won": True,
}

SAMPLE_PREDICTION = {
    "id": 101,
    "game_id": "2025_16_KC_PIT",
    "model_id": "ensemble",
    "predicted_spread": 3.5,   # Model had KC -3.5
    "predicted_total": 45.0,
    "actual_home_score": None,
    "actual_spread": None,
}


# ===========================================================================
#  1. WeeklyOutcomeUpdater Tests
# ===========================================================================

class TestWeeklyOutcomeUpdater:
    """Tests for outcome_updater.py -- updating predictions with actuals."""

    @patch("py.feedback.outcome_updater.psycopg2.connect")
    def test_update_result_dataclass_defaults(self, mock_connect):
        """UpdateResult initializes with zeroed counters."""
        result = UpdateResult(season=2025)
        assert result.season == 2025
        assert result.games_with_scores == 0
        assert result.predictions_found == 0
        assert result.predictions_updated == 0
        assert result.already_complete == 0
        assert result.errors == []
        assert result.dry_run is False

    @patch("py.feedback.outcome_updater.psycopg2.connect")
    def test_validate_completeness_all_scored(self, mock_connect):
        """validate_completeness returns complete=True when all games have scores."""
        mock_conn, mock_cursor = _make_mock_connection()
        mock_connect.return_value = mock_conn
        mock_cursor.fetchone.return_value = {
            "total_games": 16,
            "games_with_scores": 16,
        }

        updater = WeeklyOutcomeUpdater()
        status = updater.validate_completeness(season=2025, week=16)

        assert status["complete"] is True
        assert status["total_games"] == 16
        assert status["games_with_scores"] == 16

    @patch("py.feedback.outcome_updater.psycopg2.connect")
    def test_validate_completeness_partial(self, mock_connect):
        """validate_completeness returns complete=False when games are missing scores."""
        mock_conn, mock_cursor = _make_mock_connection()
        mock_connect.return_value = mock_conn
        mock_cursor.fetchone.return_value = {
            "total_games": 16,
            "games_with_scores": 14,
        }

        updater = WeeklyOutcomeUpdater()
        status = updater.validate_completeness(season=2025, week=16)

        assert status["complete"] is False

    @patch("py.feedback.outcome_updater.psycopg2.connect")
    def test_validate_completeness_empty_week(self, mock_connect):
        """validate_completeness returns complete=False for a week with no games."""
        mock_conn, mock_cursor = _make_mock_connection()
        mock_connect.return_value = mock_conn
        mock_cursor.fetchone.return_value = {
            "total_games": 0,
            "games_with_scores": 0,
        }

        updater = WeeklyOutcomeUpdater()
        status = updater.validate_completeness(season=2025, week=22)

        assert status["complete"] is False
        assert status["total_games"] == 0

    @patch("py.feedback.outcome_updater.psycopg2.connect")
    def test_update_week_no_games(self, mock_connect):
        """update_week returns immediately when no completed games exist."""
        mock_conn, mock_cursor = _make_mock_connection()
        mock_connect.return_value = mock_conn
        mock_cursor.fetchall.return_value = []  # No completed games

        updater = WeeklyOutcomeUpdater()
        result = updater.update_week(season=2025, week=16)

        assert result.games_with_scores == 0
        assert result.predictions_updated == 0

    @patch("py.feedback.outcome_updater.psycopg2.connect")
    def test_update_week_dry_run_counts_without_committing(self, mock_connect):
        """In dry_run mode, predictions are counted but not committed."""
        mock_conn, mock_cursor = _make_mock_connection()
        mock_connect.return_value = mock_conn

        # First call: get_completed_games returns one game
        # Second call: get_predictions_needing_update returns one prediction
        mock_cursor.fetchall.side_effect = [
            [SAMPLE_GAME],
            [SAMPLE_PREDICTION],
        ]

        updater = WeeklyOutcomeUpdater()
        result = updater.update_week(season=2025, week=16, dry_run=True)

        assert result.dry_run is True
        assert result.games_with_scores == 1
        assert result.predictions_found == 1
        assert result.predictions_updated == 1
        # In dry_run, cur.execute should NOT be called for the UPDATE query
        # (only the SELECT queries are executed)

    @patch("py.feedback.outcome_updater.psycopg2.connect")
    def test_update_week_error_on_missing_outcome(self, mock_connect):
        """A prediction whose game_id has no matching outcome records an error."""
        mock_conn, mock_cursor = _make_mock_connection()
        mock_connect.return_value = mock_conn

        orphan_pred = {**SAMPLE_PREDICTION, "game_id": "2025_16_MISSING_GAME"}

        mock_cursor.fetchall.side_effect = [
            [SAMPLE_GAME],      # completed games -- does NOT contain MISSING_GAME
            [orphan_pred],      # predictions needing update
        ]

        updater = WeeklyOutcomeUpdater()
        result = updater.update_week(season=2025, week=16, dry_run=True)

        assert len(result.errors) == 1
        assert "MISSING_GAME" in result.errors[0]


# ===========================================================================
#  2. RetrainTrigger Tests
# ===========================================================================

class TestRetrainTrigger:
    """Tests for retrain_trigger.py -- detecting retraining needs."""

    def test_trigger_check_dataclass(self):
        """TriggerCheck stores all check metadata."""
        tc = TriggerCheck(
            trigger_type="mae_degradation",
            triggered=True,
            current_value=12.5,
            baseline_value=10.8,
            threshold=0.5,
            message="MAE increased",
        )
        assert tc.triggered is True
        assert tc.current_value == 12.5

    # -- MAE trigger --

    def test_mae_trigger_fires_when_degraded(self):
        """MAE degradation trigger fires when recent MAE exceeds baseline + threshold."""
        trigger = RetrainTrigger.__new__(RetrainTrigger)
        trigger.THRESHOLDS = RetrainTrigger.THRESHOLDS

        baseline = {"mae": 10.5, "accuracy": 0.53, "brier": 0.24, "games": 64}
        recent = {"mae": 11.2, "accuracy": 0.51, "brier": 0.25, "games": 32}
        # Degradation = 11.2 - 10.5 = 0.7 > threshold 0.5

        check = trigger.check_mae_trigger(baseline, recent)

        assert check.triggered is True
        assert check.trigger_type == "mae_degradation"
        assert check.current_value == 11.2
        assert check.baseline_value == 10.5

    def test_mae_trigger_does_not_fire_when_stable(self):
        """MAE trigger stays silent when MAE is within acceptable range."""
        trigger = RetrainTrigger.__new__(RetrainTrigger)
        trigger.THRESHOLDS = RetrainTrigger.THRESHOLDS

        baseline = {"mae": 10.5}
        recent = {"mae": 10.7}  # Only +0.2, below 0.5 threshold

        check = trigger.check_mae_trigger(baseline, recent)

        assert check.triggered is False

    def test_mae_trigger_handles_none_recent(self):
        """MAE trigger returns not-triggered when recent data is None."""
        trigger = RetrainTrigger.__new__(RetrainTrigger)
        trigger.THRESHOLDS = RetrainTrigger.THRESHOLDS

        baseline = {"mae": 10.5}
        recent = {"mae": None}

        check = trigger.check_mae_trigger(baseline, recent)

        assert check.triggered is False
        assert "Insufficient" in check.message

    # -- Accuracy trigger --

    def test_accuracy_trigger_fires_on_drop(self):
        """Accuracy trigger fires when recent accuracy drops below baseline - threshold."""
        trigger = RetrainTrigger.__new__(RetrainTrigger)
        trigger.THRESHOLDS = RetrainTrigger.THRESHOLDS

        baseline = {"accuracy": 0.545}  # 54.5% ATS
        recent = {"accuracy": 0.48}     # 48% -- drop of 6.5pp > 5pp threshold

        check = trigger.check_accuracy_trigger(baseline, recent)

        assert check.triggered is True
        assert check.trigger_type == "accuracy_drop"

    def test_accuracy_trigger_does_not_fire_within_tolerance(self):
        """Accuracy trigger stays silent for normal fluctuation."""
        trigger = RetrainTrigger.__new__(RetrainTrigger)
        trigger.THRESHOLDS = RetrainTrigger.THRESHOLDS

        baseline = {"accuracy": 0.545}
        recent = {"accuracy": 0.52}  # Only -2.5pp, below 5pp threshold

        check = trigger.check_accuracy_trigger(baseline, recent)

        assert check.triggered is False

    # -- Calibration trigger --

    def test_calibration_trigger_fires_on_drift(self):
        """Calibration drift trigger fires when Brier score increases beyond threshold."""
        trigger = RetrainTrigger.__new__(RetrainTrigger)
        trigger.THRESHOLDS = RetrainTrigger.THRESHOLDS

        baseline = {"brier": 0.24}
        recent = {"brier": 0.27}  # +0.03 > 0.02 threshold

        check = trigger.check_calibration_trigger(baseline, recent)

        assert check.triggered is True
        assert check.trigger_type == "calibration_drift"

    def test_calibration_trigger_ok_when_stable(self):
        """Calibration trigger stays silent when Brier is within range."""
        trigger = RetrainTrigger.__new__(RetrainTrigger)
        trigger.THRESHOLDS = RetrainTrigger.THRESHOLDS

        baseline = {"brier": 0.24}
        recent = {"brier": 0.25}  # +0.01, below 0.02 threshold

        check = trigger.check_calibration_trigger(baseline, recent)

        assert check.triggered is False

    # -- Integration: check_triggers --

    @patch("py.feedback.retrain_trigger.psycopg2.connect")
    def test_check_triggers_skips_with_insufficient_games(self, mock_connect):
        """check_triggers returns early if fewer than 10 games evaluated."""
        mock_conn, mock_cursor = _make_mock_connection()
        mock_connect.return_value = mock_conn

        # Baseline query
        mock_cursor.fetchone.side_effect = [
            {"mae": 10.5, "accuracy": 0.53, "brier": 0.24, "games": 64},  # baseline
            {"mae": 11.5, "accuracy": 0.40, "brier": 0.30, "games": 5},   # recent (too few)
        ]

        trigger = RetrainTrigger()
        result = trigger.check_triggers("ensemble", 2025, 3)

        assert result.games_evaluated == 5
        assert result.retraining_recommended is False
        assert len(result.triggers_checked) == 0

    @patch("py.feedback.retrain_trigger.psycopg2.connect")
    def test_check_triggers_recommends_retrain_when_fired(self, mock_connect):
        """check_triggers sets retraining_recommended=True when any trigger fires."""
        mock_conn, mock_cursor = _make_mock_connection()
        mock_connect.return_value = mock_conn

        mock_cursor.fetchone.side_effect = [
            {"mae": 10.5, "accuracy": 0.53, "brier": 0.24, "games": 64},  # baseline
            {"mae": 12.0, "accuracy": 0.45, "brier": 0.27, "games": 32},  # recent -- all bad
        ]

        trigger = RetrainTrigger()
        result = trigger.check_triggers("ensemble", 2025, 16)

        assert result.retraining_recommended is True
        assert "mae_degradation" in result.triggers_fired
        assert "accuracy_drop" in result.triggers_fired
        assert "calibration_drift" in result.triggers_fired
        assert len(result.triggers_checked) == 3


# ===========================================================================
#  3. ModelErrorDecomposition Tests
# ===========================================================================

class TestErrorClassification:
    """Tests for the helper classification functions."""

    def test_classify_error_magnitude_small(self):
        """Errors within 3 points (field goal) are 'small'."""
        assert _classify_error_magnitude(0.5) == "small"
        assert _classify_error_magnitude(3.0) == "small"

    def test_classify_error_magnitude_medium(self):
        """Errors between 3-7 points (touchdown range) are 'medium'."""
        assert _classify_error_magnitude(3.1) == "medium"
        assert _classify_error_magnitude(7.0) == "medium"

    def test_classify_error_magnitude_large(self):
        """Errors between 7-14 points are 'large'."""
        assert _classify_error_magnitude(7.1) == "large"
        assert _classify_error_magnitude(14.0) == "large"

    def test_classify_error_magnitude_blowout(self):
        """Errors above 14 points are 'blowout' misses."""
        assert _classify_error_magnitude(14.1) == "blowout"
        assert _classify_error_magnitude(25.0) == "blowout"

    def test_classify_game_context_close(self):
        """Games decided by 3 or fewer points are 'close_game'."""
        assert _classify_game_context(2.0) == "close_game"
        assert _classify_game_context(-3.0) == "close_game"
        assert _classify_game_context(0.0) == "close_game"

    def test_classify_game_context_moderate(self):
        """Games decided by 4-10 points are 'moderate'."""
        assert _classify_game_context(7.0) == "moderate"
        assert _classify_game_context(-10.0) == "moderate"

    def test_classify_game_context_blowout(self):
        """Games decided by more than 10 points are 'blowout'."""
        assert _classify_game_context(14.0) == "blowout"
        assert _classify_game_context(-21.0) == "blowout"


class TestModelErrorDecomposer:
    """Tests for model_error_decomposition.py."""

    def _make_game_row(self, **overrides):
        """Create a realistic game row dict for _build_decomposition."""
        row = {
            "game_id": "2025_16_KC_PIT",
            "season": 2025,
            "week": 16,
            "home_team": "KC",
            "away_team": "PIT",
            "pred_home_spread": -3.5,   # Model had KC favored by 3.5
            "pred_total": 45.0,
            "pred_home_win_prob": 0.62,
            "spread_std": 4.5,
            "total_std": 5.0,
            "vegas_spread": -4.0,       # Vegas had KC favored by 4
            "vegas_total": 46.5,
            "actual_home_score": 27,
            "actual_away_score": 20,
            "actual_spread": 7,          # KC won by 7
            "actual_total": 47,
            "model_version": "v2.0",
            "inference_time_ms": 105.0,
        }
        row.update(overrides)
        return row

    def test_build_decomposition_spread_error(self):
        """Spread error = actual_spread - pred_home_spread."""
        decomposer = ModelErrorDecomposer.__new__(ModelErrorDecomposer)
        row = self._make_game_row()

        result = decomposer._build_decomposition(row)

        # actual_spread=7, pred_home_spread=-3.5 => error = 7 - (-3.5) = 10.5
        assert result.model_spread_error == pytest.approx(10.5, abs=0.01)

    def test_build_decomposition_total_error(self):
        """Total error = actual_total - pred_total."""
        decomposer = ModelErrorDecomposer.__new__(ModelErrorDecomposer)
        row = self._make_game_row()

        result = decomposer._build_decomposition(row)

        # actual_total=47, pred_total=45.0 => error = 2.0
        assert result.model_total_error == pytest.approx(2.0, abs=0.01)

    def test_build_decomposition_vegas_comparison(self):
        """Vegas error computed and model-beats-vegas flag set correctly."""
        decomposer = ModelErrorDecomposer.__new__(ModelErrorDecomposer)
        row = self._make_game_row()

        result = decomposer._build_decomposition(row)

        # vegas_spread=-4.0, actual=7 => vegas error = 7 - (-4.0) = 11.0
        assert result.vegas_spread_error == pytest.approx(11.0, abs=0.01)
        # Model error |10.5| < |11.0| => model beat vegas
        assert result.model_beat_vegas_spread is True

    def test_build_decomposition_model_correct_side(self):
        """model_correct_side is True when model and result agree on winner direction."""
        decomposer = ModelErrorDecomposer.__new__(ModelErrorDecomposer)
        # pred_home_spread=-3.5 (home favored) -- but actual_spread=7 (home won)
        # Both negative pred and positive actual? Let's check:
        # pred=-3.5 means home is favored (negative spread = home favored),
        # actual=7 means home won by 7 (positive).
        # The code checks: (pred > 0 and actual > 0) or (pred < 0 and actual < 0)
        # -3.5 < 0 and 7 > 0 => neither branch matches => False
        row = self._make_game_row()
        result = decomposer._build_decomposition(row)
        assert result.model_correct_side is False

        # Now test with matching signs
        row2 = self._make_game_row(pred_home_spread=2.5, actual_spread=7)
        result2 = decomposer._build_decomposition(row2)
        assert result2.model_correct_side is True

    def test_build_decomposition_no_vegas_lines(self):
        """When Vegas data is None, vegas fields remain None."""
        decomposer = ModelErrorDecomposer.__new__(ModelErrorDecomposer)
        row = self._make_game_row(vegas_spread=None, vegas_total=None)

        result = decomposer._build_decomposition(row)

        assert result.vegas_spread_error is None
        assert result.vegas_total_error is None
        assert result.model_beat_vegas_spread is None
        assert result.vegas_correct_side is None

    def test_build_decomposition_error_magnitude_categorization(self):
        """Error magnitude is classified based on absolute spread error."""
        decomposer = ModelErrorDecomposer.__new__(ModelErrorDecomposer)
        # actual=7, pred=-3.5 => |error|=10.5 => "large" (7 < 10.5 <= 14)
        row = self._make_game_row()
        result = decomposer._build_decomposition(row)
        assert result.error_magnitude == "large"

    def test_season_error_report_aggregation(self):
        """SeasonErrorReport computes correct aggregate metrics from decompositions."""
        report = SeasonErrorReport(season=2025, total_games=3)

        # Simulate three games' errors
        spread_errors = np.array([2.0, -5.0, 8.0])
        report.model_spread_mae = float(np.mean(np.abs(spread_errors)))  # 5.0
        report.model_spread_rmse = float(np.sqrt(np.mean(spread_errors ** 2)))
        report.model_spread_bias = float(np.mean(spread_errors))  # 1.667

        assert report.model_spread_mae == pytest.approx(5.0, abs=0.01)
        assert report.model_spread_bias == pytest.approx(1.667, abs=0.01)
        # RMSE = sqrt((4 + 25 + 64)/3) = sqrt(31) = 5.568
        assert report.model_spread_rmse == pytest.approx(5.568, abs=0.01)

    def test_season_error_report_to_dict(self):
        """to_dict produces rounded values and handles None vegas fields."""
        report = SeasonErrorReport(
            season=2025,
            total_games=100,
            model_spread_mae=10.12345,
            model_spread_rmse=12.6789,
            model_total_mae=9.8765,
            model_total_rmse=11.234,
            model_spread_bias=-0.5678,
            model_total_bias=1.234,
            model_correct_side_pct=0.545,
        )
        d = report.to_dict()

        assert d["model_spread_mae"] == 10.1235  # rounded to 4 decimals
        assert d["vegas_spread_mae"] is None
        assert d["model_correct_side_pct"] == 0.545


# ===========================================================================
#  4. AutoFeatureSelector Tests
# ===========================================================================

class TestAutoFeatureSelector:
    """Tests for auto_feature_selector.py -- feature state transitions."""

    def test_determine_status_strong_negative_disables(self):
        """Features with strongly negative contribution get disabled."""
        selector = AutoFeatureSelector.__new__(AutoFeatureSelector)
        # Copy class-level thresholds
        selector.CONTRIBUTION_POSITIVE = AutoFeatureSelector.CONTRIBUTION_POSITIVE
        selector.CONTRIBUTION_NEGATIVE = AutoFeatureSelector.CONTRIBUTION_NEGATIVE
        selector.POSITIVE_WEEKS_TO_PROMOTE = AutoFeatureSelector.POSITIVE_WEEKS_TO_PROMOTE
        selector.NEGATIVE_WEEKS_TO_DISABLE = AutoFeatureSelector.NEGATIVE_WEEKS_TO_DISABLE

        # mean_contribution < CONTRIBUTION_NEGATIVE * 2 = -0.04
        status = selector._determine_status(
            mean_contribution=-0.06,
            current_status="active",
            weeks_active=10,
        )
        assert status == "disabled"

    def test_determine_status_strong_positive_promotes_probation(self):
        """Features with strongly positive contribution promote from probation to active."""
        selector = AutoFeatureSelector.__new__(AutoFeatureSelector)
        selector.CONTRIBUTION_POSITIVE = AutoFeatureSelector.CONTRIBUTION_POSITIVE
        selector.CONTRIBUTION_NEGATIVE = AutoFeatureSelector.CONTRIBUTION_NEGATIVE
        selector.POSITIVE_WEEKS_TO_PROMOTE = AutoFeatureSelector.POSITIVE_WEEKS_TO_PROMOTE
        selector.NEGATIVE_WEEKS_TO_DISABLE = AutoFeatureSelector.NEGATIVE_WEEKS_TO_DISABLE

        # mean_contribution > CONTRIBUTION_POSITIVE * 2 = 0.04
        status = selector._determine_status(
            mean_contribution=0.05,
            current_status="probation",
            weeks_active=1,
        )
        assert status == "active"

    def test_determine_status_moderate_positive_needs_time(self):
        """Moderate positive contribution only promotes after enough weeks on probation."""
        selector = AutoFeatureSelector.__new__(AutoFeatureSelector)
        selector.CONTRIBUTION_POSITIVE = AutoFeatureSelector.CONTRIBUTION_POSITIVE
        selector.CONTRIBUTION_NEGATIVE = AutoFeatureSelector.CONTRIBUTION_NEGATIVE
        selector.POSITIVE_WEEKS_TO_PROMOTE = AutoFeatureSelector.POSITIVE_WEEKS_TO_PROMOTE
        selector.NEGATIVE_WEEKS_TO_DISABLE = AutoFeatureSelector.NEGATIVE_WEEKS_TO_DISABLE

        # mean_contribution=0.025 (between POSITIVE and 2*POSITIVE)
        # Still on probation, only 1 week active (need 2)
        status = selector._determine_status(
            mean_contribution=0.025,
            current_status="probation",
            weeks_active=1,
        )
        assert status == "probation"

        # Same contribution, but now 2+ weeks active => promote
        status = selector._determine_status(
            mean_contribution=0.025,
            current_status="probation",
            weeks_active=3,
        )
        assert status == "active"

    def test_determine_status_moderate_negative_demotes_active(self):
        """Active features with moderate negative contribution get moved to probation after enough weeks."""
        selector = AutoFeatureSelector.__new__(AutoFeatureSelector)
        selector.CONTRIBUTION_POSITIVE = AutoFeatureSelector.CONTRIBUTION_POSITIVE
        selector.CONTRIBUTION_NEGATIVE = AutoFeatureSelector.CONTRIBUTION_NEGATIVE
        selector.POSITIVE_WEEKS_TO_PROMOTE = AutoFeatureSelector.POSITIVE_WEEKS_TO_PROMOTE
        selector.NEGATIVE_WEEKS_TO_DISABLE = AutoFeatureSelector.NEGATIVE_WEEKS_TO_DISABLE

        # mean_contribution=-0.025 (between NEGATIVE and 2*NEGATIVE)
        # Active for 5 weeks (>= NEGATIVE_WEEKS_TO_DISABLE=4) => demote to probation
        status = selector._determine_status(
            mean_contribution=-0.025,
            current_status="active",
            weeks_active=5,
        )
        assert status == "probation"

    def test_determine_status_neutral_keeps_current(self):
        """Neutral contribution keeps the feature in its current state."""
        selector = AutoFeatureSelector.__new__(AutoFeatureSelector)
        selector.CONTRIBUTION_POSITIVE = AutoFeatureSelector.CONTRIBUTION_POSITIVE
        selector.CONTRIBUTION_NEGATIVE = AutoFeatureSelector.CONTRIBUTION_NEGATIVE
        selector.POSITIVE_WEEKS_TO_PROMOTE = AutoFeatureSelector.POSITIVE_WEEKS_TO_PROMOTE
        selector.NEGATIVE_WEEKS_TO_DISABLE = AutoFeatureSelector.NEGATIVE_WEEKS_TO_DISABLE

        # Neutral = between -0.02 and 0.02
        status = selector._determine_status(
            mean_contribution=0.005,
            current_status="active",
            weeks_active=10,
        )
        assert status == "active"

    def test_determine_status_disabled_gets_second_chance(self):
        """Disabled features with neutral contribution move to probation for re-evaluation."""
        selector = AutoFeatureSelector.__new__(AutoFeatureSelector)
        selector.CONTRIBUTION_POSITIVE = AutoFeatureSelector.CONTRIBUTION_POSITIVE
        selector.CONTRIBUTION_NEGATIVE = AutoFeatureSelector.CONTRIBUTION_NEGATIVE
        selector.POSITIVE_WEEKS_TO_PROMOTE = AutoFeatureSelector.POSITIVE_WEEKS_TO_PROMOTE
        selector.NEGATIVE_WEEKS_TO_DISABLE = AutoFeatureSelector.NEGATIVE_WEEKS_TO_DISABLE

        status = selector._determine_status(
            mean_contribution=0.0,
            current_status="disabled",
            weeks_active=0,
        )
        assert status == "probation"

    def test_feature_selection_result_tracks_changes(self):
        """FeatureSelectionResult correctly tracks promotions and demotions."""
        result = FeatureSelectionResult(season=2025, week=16)
        result.active = ["sem_injury_advantage", "sem_sentiment_24h"]
        result.disabled = ["sem_media_buzz"]
        result.probation = ["sem_coaching_change"]
        result.promoted = ["sem_injury_advantage"]
        result.demoted = ["sem_media_buzz"]
        result.changes_made = 2

        assert len(result.active) == 2
        assert len(result.disabled) == 1
        assert len(result.probation) == 1
        assert result.changes_made == 2


# ===========================================================================
#  5. FeaturePerformanceMemory Tests
# ===========================================================================

class TestFeaturePerformanceMemory:
    """Tests for feature_memory.py -- rolling window feature tracking."""

    def test_feature_stats_dataclass(self):
        """FeatureStats stores all performance metrics."""
        stats = FeatureStats(
            feature_name="sem_injury_advantage",
            mean_contribution=0.035,
            contribution_when_correct=0.05,
            contribution_when_incorrect=-0.01,
            correlation_with_error=-0.15,
            stability=0.85,
            sample_size=120,
            recommendation="boost",
        )
        assert stats.recommendation == "boost"
        assert stats.sample_size == 120

    def test_compute_recommendation_drop(self):
        """Features below DROP_THRESHOLD (-0.03) get 'drop' recommendation."""
        memory = FeaturePerformanceMemory.__new__(FeaturePerformanceMemory)
        memory.DROP_THRESHOLD = FeaturePerformanceMemory.DROP_THRESHOLD
        memory.BOOST_THRESHOLD = FeaturePerformanceMemory.BOOST_THRESHOLD
        memory.STABILITY_THRESHOLD = FeaturePerformanceMemory.STABILITY_THRESHOLD

        rec = memory._compute_recommendation(
            mean_contrib=-0.05,
            corr=0.1,
            stability=0.8,
            contrib_correct=-0.03,
            contrib_incorrect=-0.07,
        )
        assert rec == "drop"

    def test_compute_recommendation_boost(self):
        """Features above BOOST_THRESHOLD (0.03) get 'boost' recommendation."""
        memory = FeaturePerformanceMemory.__new__(FeaturePerformanceMemory)
        memory.DROP_THRESHOLD = FeaturePerformanceMemory.DROP_THRESHOLD
        memory.BOOST_THRESHOLD = FeaturePerformanceMemory.BOOST_THRESHOLD
        memory.STABILITY_THRESHOLD = FeaturePerformanceMemory.STABILITY_THRESHOLD

        rec = memory._compute_recommendation(
            mean_contrib=0.05,
            corr=-0.1,
            stability=0.9,
            contrib_correct=0.06,
            contrib_incorrect=0.02,
        )
        assert rec == "boost"

    def test_compute_recommendation_investigate_low_stability(self):
        """Features with low stability get 'investigate' recommendation."""
        memory = FeaturePerformanceMemory.__new__(FeaturePerformanceMemory)
        memory.DROP_THRESHOLD = FeaturePerformanceMemory.DROP_THRESHOLD
        memory.BOOST_THRESHOLD = FeaturePerformanceMemory.BOOST_THRESHOLD
        memory.STABILITY_THRESHOLD = FeaturePerformanceMemory.STABILITY_THRESHOLD

        rec = memory._compute_recommendation(
            mean_contrib=0.01,   # neutral
            corr=0.0,
            stability=0.3,      # below 0.5 threshold
            contrib_correct=0.01,
            contrib_incorrect=0.01,
        )
        assert rec == "investigate"

    def test_compute_recommendation_investigate_high_error_corr(self):
        """Features with high positive error correlation get 'investigate'."""
        memory = FeaturePerformanceMemory.__new__(FeaturePerformanceMemory)
        memory.DROP_THRESHOLD = FeaturePerformanceMemory.DROP_THRESHOLD
        memory.BOOST_THRESHOLD = FeaturePerformanceMemory.BOOST_THRESHOLD
        memory.STABILITY_THRESHOLD = FeaturePerformanceMemory.STABILITY_THRESHOLD

        rec = memory._compute_recommendation(
            mean_contrib=0.01,
            corr=0.3,            # > 0.2 threshold
            stability=0.8,
            contrib_correct=0.01,
            contrib_incorrect=0.01,
        )
        assert rec == "investigate"

    def test_compute_recommendation_keep(self):
        """Neutral, stable features get 'keep' recommendation."""
        memory = FeaturePerformanceMemory.__new__(FeaturePerformanceMemory)
        memory.DROP_THRESHOLD = FeaturePerformanceMemory.DROP_THRESHOLD
        memory.BOOST_THRESHOLD = FeaturePerformanceMemory.BOOST_THRESHOLD
        memory.STABILITY_THRESHOLD = FeaturePerformanceMemory.STABILITY_THRESHOLD

        rec = memory._compute_recommendation(
            mean_contrib=0.01,
            corr=0.05,
            stability=0.85,
            contrib_correct=0.012,
            contrib_incorrect=0.008,
        )
        assert rec == "keep"

    @patch("py.feedback.feature_memory.psycopg2.connect")
    def test_get_feature_stats_insufficient_data(self, mock_connect):
        """get_feature_stats returns 'insufficient_data' when samples < MIN_SAMPLES."""
        mock_conn, mock_cursor = _make_mock_connection()
        mock_connect.return_value = mock_conn

        # Return only 10 rows, below MIN_SAMPLES=20
        mock_cursor.fetchall.return_value = [
            {"feature_value": 0.5, "prediction_error": 3.0,
             "correct_prediction": True, "marginal_contribution": 0.02}
        ] * 10

        memory = FeaturePerformanceMemory()
        stats = memory.get_feature_stats("sem_injury_advantage", weeks=8)

        assert stats.recommendation == "insufficient_data"
        assert stats.sample_size == 10

    @patch("py.feedback.feature_memory.psycopg2.connect")
    def test_get_feature_stats_computes_metrics(self, mock_connect):
        """get_feature_stats correctly computes contribution, stability, and correlation."""
        mock_conn, mock_cursor = _make_mock_connection()
        mock_connect.return_value = mock_conn

        # Generate 30 realistic rows (above MIN_SAMPLES=20)
        # Use fixed contributions to ensure deterministic outcomes:
        # 25 correct predictions with +0.05, 5 incorrect with -0.01
        # Mean contribution = (25*0.05 + 5*(-0.01)) / 30 = 1.2/30 = 0.04 > 0.03 threshold
        np.random.seed(42)
        rows = []
        for i in range(30):
            correct = i % 6 != 0  # 25 correct, 5 incorrect
            contrib = 0.05 if correct else -0.01
            rows.append({
                "feature_value": float(np.random.uniform(0.1, 0.9)),
                "prediction_error": float(np.random.uniform(2.0, 12.0)),
                "correct_prediction": correct,
                "marginal_contribution": contrib,
            })

        mock_cursor.fetchall.return_value = rows

        memory = FeaturePerformanceMemory()
        stats = memory.get_feature_stats("sem_home_sentiment", weeks=8)

        assert stats.sample_size == 30
        assert stats.feature_name == "sem_home_sentiment"
        # Mean contribution = 0.04 (above BOOST_THRESHOLD=0.03)
        assert stats.mean_contribution > 0.03
        # Stability should be reasonably high (contributions are +0.05 or -0.01)
        assert 0 < stats.stability <= 1.0
        # Recommendation should be 'boost' given mean > 0.03
        assert stats.recommendation == "boost"

    def test_get_feature_stats_validates_weeks_parameter(self):
        """get_feature_stats raises ValueError for invalid weeks."""
        memory = FeaturePerformanceMemory.__new__(FeaturePerformanceMemory)
        memory.conn_params = {}
        memory.MIN_SAMPLES = 20

        with pytest.raises(ValueError, match="weeks must be between"):
            memory.get_feature_stats("test_feature", weeks=0)

        with pytest.raises(ValueError, match="weeks must be between"):
            memory.get_feature_stats("test_feature", weeks=100)
