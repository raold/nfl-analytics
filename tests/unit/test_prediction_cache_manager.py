"""
Unit tests for PredictionCacheManager - PostgreSQL cache management for R models.

Tests cover:
- Cache management (bulk inserts, deduplication)
- Performance optimization (lookup speed, hit rate tracking)
- Data integrity (range checks, NaN detection)
- Migration support (rpy2 → PostgreSQL)
- Index optimization and VACUUM
- Validation and error handling

Author: Claude Code (Sonnet 4.5)
Created: November 22, 2025
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from py.utils.prediction_cache_manager import (
    CacheStats,
    PredictionCacheManager,
    ValidationReport,
)


@pytest.fixture
def mock_db_connection():
    """Mock database connection."""
    conn = MagicMock()
    conn.closed = False
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    return conn, cursor


@pytest.fixture
def cache_manager(mock_db_connection):
    """Create cache manager with mocked database."""
    conn, cursor = mock_db_connection

    with patch("psycopg2.connect", return_value=conn):
        manager = PredictionCacheManager(db_url="postgresql://test/testdb")
        return manager


@pytest.fixture
def sample_predictions():
    """Sample predictions DataFrame for testing."""
    return pd.DataFrame(
        {
            "player_id": ["P001", "P002", "P003", "P001", "P002"],
            "season": [2024, 2024, 2024, 2024, 2024],
            "week": [1, 1, 1, 2, 2],
            "prediction": [250.5, 180.3, 120.7, 245.2, 175.8],
        }
    )


class TestCachePredictions:
    """Test cache_predictions() method."""

    def test_cache_predictions_success(self, cache_manager, sample_predictions):
        """Test successful bulk insert of predictions."""
        result = cache_manager.cache_predictions(
            sample_predictions, model_name="state_space_v2", stat_type="passing_yards"
        )

        assert result == 5
        assert cache_manager._get_connection().commit.called

    def test_cache_predictions_missing_columns(self, cache_manager):
        """Test error on missing required columns."""
        invalid_df = pd.DataFrame({"player_id": ["P001"], "season": [2024]})

        with pytest.raises(ValueError, match="must have columns"):
            cache_manager.cache_predictions(
                invalid_df, model_name="test", stat_type="passing_yards"
            )

    def test_cache_predictions_deduplication(self, cache_manager):
        """Test deduplication of identical predictions."""
        # Create DataFrame with duplicates
        df = pd.DataFrame(
            {
                "player_id": ["P001", "P001"],  # Same player
                "season": [2024, 2024],  # Same season
                "week": [1, 1],  # Same week
                "prediction": [250.5, 255.0],  # Different predictions
            }
        )

        result = cache_manager.cache_predictions(df, model_name="test", stat_type="passing_yards")

        # Should keep only the latest (most recent)
        assert result == 1

    def test_cache_predictions_validation_removes_invalid(self, cache_manager):
        """Test validation removes invalid predictions."""
        df = pd.DataFrame(
            {
                "player_id": ["P001", "P002", "P003"],
                "season": [2024, 2024, 2024],
                "week": [1, 1, 1],
                "prediction": [250.5, np.nan, 700.0],  # NaN and out of range
            }
        )

        result = cache_manager.cache_predictions(
            df, model_name="test", stat_type="passing_yards", validate=True
        )

        # Should only cache the valid prediction (250.5)
        assert result == 1

    def test_cache_predictions_batch_processing(self, cache_manager):
        """Test batch processing for large datasets."""
        # Create large DataFrame (2500 predictions)
        large_df = pd.DataFrame(
            {
                "player_id": [f"P{i:04d}" for i in range(2500)],
                "season": [2024] * 2500,
                "week": [1] * 2500,
                "prediction": np.random.uniform(100, 400, 2500),
            }
        )

        result = cache_manager.cache_predictions(
            large_df, model_name="test", stat_type="passing_yards"
        )

        assert result == 2500

    def test_cache_predictions_handles_database_error(self, cache_manager):
        """Test rollback on database error."""
        conn = cache_manager._get_connection()
        cursor = conn.cursor.return_value
        cursor.execute.side_effect = Exception("Database error")

        df = pd.DataFrame(
            {
                "player_id": ["P001"],
                "season": [2024],
                "week": [1],
                "prediction": [250.5],
            }
        )

        with pytest.raises(Exception, match="Database error"):
            cache_manager.cache_predictions(df, model_name="test", stat_type="passing_yards")

        # Should rollback on error
        assert conn.rollback.called


class TestGetCachedPrediction:
    """Test get_cached_prediction() method."""

    def test_get_cached_prediction_hit(self, cache_manager):
        """Test successful cache hit."""
        cursor = cache_manager._get_connection().cursor.return_value
        cursor.fetchone.return_value = (250.5, datetime.now())

        result = cache_manager.get_cached_prediction(
            player_id="P001", stat_type="passing_yards", season=2024, week=1
        )

        assert result == 250.5
        assert cache_manager._hit_count == 1
        assert cache_manager._miss_count == 0

    def test_get_cached_prediction_miss(self, cache_manager):
        """Test cache miss."""
        cursor = cache_manager._get_connection().cursor.return_value
        cursor.fetchone.return_value = None

        result = cache_manager.get_cached_prediction(
            player_id="P001", stat_type="passing_yards", season=2024, week=1
        )

        assert result is None
        assert cache_manager._hit_count == 0
        assert cache_manager._miss_count == 1

    def test_get_cached_prediction_with_model_filter(self, cache_manager):
        """Test lookup with model name filter."""
        cursor = cache_manager._get_connection().cursor.return_value
        cursor.fetchone.return_value = (250.5, datetime.now())

        result = cache_manager.get_cached_prediction(
            player_id="P001",
            stat_type="passing_yards",
            season=2024,
            week=1,
            model_name="state_space_v2",
        )

        assert result == 250.5

        # Check that model_name was included in query
        call_args = cursor.execute.call_args[0]
        assert "model_name" in call_args[0]

    def test_get_cached_prediction_with_max_age(self, cache_manager):
        """Test lookup with maximum age constraint."""
        cursor = cache_manager._get_connection().cursor.return_value
        cursor.fetchone.return_value = (250.5, datetime.now())

        result = cache_manager.get_cached_prediction(
            player_id="P001",
            stat_type="passing_yards",
            season=2024,
            week=1,
            max_age_hours=24,
        )

        assert result == 250.5

        # Check that last_updated constraint was included
        call_args = cursor.execute.call_args[0]
        assert "last_updated" in call_args[0]

    def test_get_cached_prediction_tracks_lookup_time(self, cache_manager):
        """Test that lookup time is tracked."""
        cursor = cache_manager._get_connection().cursor.return_value
        cursor.fetchone.return_value = (250.5, datetime.now())

        cache_manager.get_cached_prediction(
            player_id="P001", stat_type="passing_yards", season=2024, week=1
        )

        assert len(cache_manager._lookup_times) == 1
        assert cache_manager._lookup_times[0] > 0  # Positive time

    def test_get_cached_prediction_limits_lookup_times(self, cache_manager):
        """Test that lookup times list is limited to 1000 entries."""
        cursor = cache_manager._get_connection().cursor.return_value
        cursor.fetchone.return_value = (250.5, datetime.now())

        # Perform 1500 lookups
        for i in range(1500):
            cache_manager.get_cached_prediction(
                player_id=f"P{i:04d}", stat_type="passing_yards", season=2024, week=1
            )

        # Should keep only last 1000
        assert len(cache_manager._lookup_times) == 1000


class TestInvalidateCache:
    """Test invalidate_cache() method."""

    def test_invalidate_cache_all(self, cache_manager):
        """Test invalidating all cached predictions."""
        cursor = cache_manager._get_connection().cursor.return_value
        cursor.rowcount = 1000

        result = cache_manager.invalidate_cache()

        assert result == 1000
        assert cache_manager._get_connection().commit.called

    def test_invalidate_cache_by_model(self, cache_manager):
        """Test invalidating predictions for specific model."""
        cursor = cache_manager._get_connection().cursor.return_value
        cursor.rowcount = 250

        result = cache_manager.invalidate_cache(model_name="state_space_v2")

        assert result == 250

        # Check that model_name filter was applied
        call_args = cursor.execute.call_args[0]
        assert "model_name" in call_args[0]

    def test_invalidate_cache_by_stat_type(self, cache_manager):
        """Test invalidating predictions for specific stat type."""
        cursor = cache_manager._get_connection().cursor.return_value
        cursor.rowcount = 180

        result = cache_manager.invalidate_cache(stat_type="rushing_yards")

        assert result == 180

    def test_invalidate_cache_by_season(self, cache_manager):
        """Test invalidating predictions for specific season."""
        cursor = cache_manager._get_connection().cursor.return_value
        cursor.rowcount = 500

        result = cache_manager.invalidate_cache(season=2023)

        assert result == 500

    def test_invalidate_cache_older_than(self, cache_manager):
        """Test invalidating old predictions."""
        cursor = cache_manager._get_connection().cursor.return_value
        cursor.rowcount = 75

        result = cache_manager.invalidate_cache(older_than_hours=24)

        assert result == 75

        # Check that last_updated filter was applied
        call_args = cursor.execute.call_args[0]
        assert "last_updated" in call_args[0]

    def test_invalidate_cache_handles_error(self, cache_manager):
        """Test rollback on invalidation error."""
        conn = cache_manager._get_connection()
        cursor = conn.cursor.return_value
        cursor.execute.side_effect = Exception("Database error")

        with pytest.raises(Exception):
            cache_manager.invalidate_cache()

        assert conn.rollback.called


class TestGetCacheStats:
    """Test get_cache_stats() method."""

    def test_get_cache_stats_success(self, cache_manager):
        """Test successful retrieval of cache statistics."""
        cursor = cache_manager._get_connection().cursor.return_value

        # Mock database responses
        cursor.fetchone.side_effect = [
            (1000,),  # Total predictions
            (150,),  # Unique players
            (4,),  # Unique models
            (25.5,),  # Cache size MB
            (datetime(2024, 1, 1), datetime(2024, 11, 22)),  # Oldest/newest
        ]
        cursor.fetchall.side_effect = [
            [("state_space_v2", 400), ("hierarchical_v2", 300)],  # By model
            [("passing_yards", 600), ("rushing_yards", 400)],  # By stat type
        ]

        # Set some lookup stats
        cache_manager._hit_count = 80
        cache_manager._miss_count = 20
        cache_manager._lookup_times = [0.5, 0.6, 0.7, 0.8]

        stats = cache_manager.get_cache_stats()

        assert stats.total_predictions == 1000
        assert stats.unique_players == 150
        assert stats.unique_models == 4
        assert stats.cache_size_mb == 25.5
        assert stats.hit_rate == 80.0  # 80/100
        assert stats.avg_lookup_time_ms == 0.65  # Mean of [0.5, 0.6, 0.7, 0.8]
        assert "state_space_v2" in stats.predictions_by_model
        assert "passing_yards" in stats.predictions_by_stat_type

    def test_get_cache_stats_empty_cache(self, cache_manager):
        """Test stats for empty cache."""
        cursor = cache_manager._get_connection().cursor.return_value

        cursor.fetchone.side_effect = [
            (0,),  # Total predictions
            (0,),  # Unique players
            (0,),  # Unique models
            (0.0,),  # Cache size MB
            (None, None),  # Oldest/newest
        ]
        cursor.fetchall.side_effect = [[], []]

        stats = cache_manager.get_cache_stats()

        assert stats.total_predictions == 0
        assert stats.unique_players == 0
        assert stats.hit_rate == 0.0
        assert stats.avg_lookup_time_ms == 0.0


class TestValidateCacheIntegrity:
    """Test validate_cache_integrity() method."""

    def test_validate_cache_all_valid(self, cache_manager):
        """Test validation with all valid predictions."""
        cursor = cache_manager._get_connection().cursor.return_value

        # Mock valid predictions
        cursor.fetchall.side_effect = [
            [
                (
                    "P001",
                    2024,
                    1,
                    "state_space_v2",
                    "passing_yards",
                    250.5,
                    datetime(2024, 11, 1),
                    datetime(2024, 11, 22),
                ),
                (
                    "P002",
                    2024,
                    1,
                    "state_space_v2",
                    "rushing_yards",
                    120.3,
                    datetime(2024, 11, 1),
                    datetime(2024, 11, 22),
                ),
            ],  # Predictions
            [],  # No duplicates
        ]
        cursor.fetchone.return_value = (0,)  # No orphaned predictions

        report = cache_manager.validate_cache_integrity()

        assert report.total_checked == 2
        assert report.valid_predictions == 2
        assert report.invalid_predictions == 0
        assert report.is_valid()

    def test_validate_cache_with_nan(self, cache_manager):
        """Test validation detects NaN predictions."""
        cursor = cache_manager._get_connection().cursor.return_value

        cursor.fetchall.side_effect = [
            [
                (
                    "P001",
                    2024,
                    1,
                    "test",
                    "passing_yards",
                    None,  # NaN prediction
                    datetime(2024, 11, 1),
                    datetime(2024, 11, 22),
                )
            ],
            [],  # No duplicates
        ]
        cursor.fetchone.return_value = (0,)

        report = cache_manager.validate_cache_integrity()

        assert report.nan_predictions == 1
        assert not report.is_valid()
        assert any("NaN prediction" in issue for issue in report.issues)

    def test_validate_cache_out_of_range(self, cache_manager):
        """Test validation detects out-of-range predictions."""
        cursor = cache_manager._get_connection().cursor.return_value

        cursor.fetchall.side_effect = [
            [
                (
                    "P001",
                    2024,
                    1,
                    "test",
                    "passing_yards",
                    700.0,  # Out of range (max 600)
                    datetime(2024, 11, 1),
                    datetime(2024, 11, 22),
                )
            ],
            [],
        ]
        cursor.fetchone.return_value = (0,)

        report = cache_manager.validate_cache_integrity()

        assert report.out_of_range_predictions == 1
        assert not report.is_valid()

    def test_validate_cache_duplicates(self, cache_manager):
        """Test validation detects duplicate predictions."""
        cursor = cache_manager._get_connection().cursor.return_value

        cursor.fetchall.side_effect = [
            [],  # No predictions to check individually
            [("P001", 2024, 1, "test", "passing_yards", 2)],  # 2 duplicates
        ]
        cursor.fetchone.return_value = (0,)

        report = cache_manager.validate_cache_integrity()

        assert report.duplicate_predictions == 1

    def test_validate_cache_orphaned(self, cache_manager):
        """Test validation detects orphaned predictions."""
        cursor = cache_manager._get_connection().cursor.return_value

        cursor.fetchall.side_effect = [[], []]
        cursor.fetchone.return_value = (5,)  # 5 orphaned predictions

        report = cache_manager.validate_cache_integrity()

        assert report.orphaned_predictions == 5

    def test_validate_cache_sample_size(self, cache_manager):
        """Test validation with sample size limit."""
        cursor = cache_manager._get_connection().cursor.return_value

        cursor.fetchall.side_effect = [[], []]
        cursor.fetchone.return_value = (0,)

        cache_manager.validate_cache_integrity(sample_size=100)

        # Check that LIMIT was applied
        call_args = cursor.execute.call_args_list[0][0]
        assert "LIMIT" in call_args[0]


class TestOptimizeIndexes:
    """Test optimize_indexes() method."""

    def test_optimize_indexes_success(self, cache_manager):
        """Test successful index creation."""
        results = cache_manager.optimize_indexes()

        assert len(results) == 4  # 4 indexes
        assert all(results.values())  # All successful

    def test_optimize_indexes_handles_error(self, cache_manager):
        """Test handling of index creation errors."""
        conn = cache_manager._get_connection()
        cursor = conn.cursor.return_value
        cursor.execute.side_effect = [
            None,  # First index succeeds
            Exception("Index error"),  # Second index fails
            None,  # Third index succeeds
            None,  # Fourth index succeeds
        ]

        results = cache_manager.optimize_indexes()

        # Should have mixed results
        assert not all(results.values())
        assert any(results.values())


class TestVacuumAnalyze:
    """Test vacuum_analyze() method."""

    def test_vacuum_analyze_success(self, cache_manager):
        """Test successful VACUUM ANALYZE."""
        result = cache_manager.vacuum_analyze()

        assert result is True

    def test_vacuum_analyze_handles_error(self, cache_manager):
        """Test handling of VACUUM error."""
        cursor = cache_manager._get_connection().cursor.return_value
        cursor.execute.side_effect = Exception("VACUUM error")

        result = cache_manager.vacuum_analyze()

        assert result is False


class TestBackfillFromRpy2:
    """Test backfill_from_rpy2() method."""

    def test_backfill_success(self, cache_manager):
        """Test successful backfill from rpy2."""

        # Mock rpy2 prediction function
        def mock_rpy2_predictions(season, stat_type, player_ids=None):
            return pd.DataFrame(
                {
                    "player_id": ["P001", "P002"],
                    "season": [season, season],
                    "week": [1, 1],
                    "prediction": [250.5, 180.3],
                }
            )

        total, cached = cache_manager.backfill_from_rpy2(
            model_name="state_space_v2",
            stat_type="passing_yards",
            seasons=[2023, 2024],
            rpy2_prediction_function=mock_rpy2_predictions,
        )

        assert total == 4  # 2 seasons × 2 predictions
        assert cached == 4

    def test_backfill_handles_empty_season(self, cache_manager):
        """Test backfill handles seasons with no predictions."""

        def mock_rpy2_predictions(season, stat_type, player_ids=None):
            if season == 2023:
                return None  # No predictions for 2023
            return pd.DataFrame(
                {
                    "player_id": ["P001"],
                    "season": [season],
                    "week": [1],
                    "prediction": [250.5],
                }
            )

        total, cached = cache_manager.backfill_from_rpy2(
            model_name="test",
            stat_type="passing_yards",
            seasons=[2023, 2024],
            rpy2_prediction_function=mock_rpy2_predictions,
        )

        assert total == 1  # Only 2024
        assert cached == 1


class TestValidatePredictions:
    """Test _validate_predictions() internal method."""

    def test_validate_predictions_removes_nan(self, cache_manager):
        """Test NaN removal."""
        df = pd.DataFrame(
            {
                "player_id": ["P001", "P002", "P003"],
                "season": [2024, 2024, 2024],
                "week": [1, 1, 1],
                "prediction": [250.5, np.nan, 180.3],
            }
        )

        result = cache_manager._validate_predictions(df, "passing_yards")

        assert len(result) == 2
        assert not result["prediction"].isna().any()

    def test_validate_predictions_removes_out_of_range(self, cache_manager):
        """Test out-of-range removal."""
        df = pd.DataFrame(
            {
                "player_id": ["P001", "P002", "P003"],
                "season": [2024, 2024, 2024],
                "week": [1, 1, 1],
                "prediction": [250.5, 700.0, 180.3],  # 700 is out of range
            }
        )

        result = cache_manager._validate_predictions(df, "passing_yards")

        assert len(result) == 2
        assert (result["prediction"] <= 600).all()

    def test_validate_predictions_removes_invalid_player_ids(self, cache_manager):
        """Test invalid player ID removal."""
        df = pd.DataFrame(
            {
                "player_id": ["P001", "", None],
                "season": [2024, 2024, 2024],
                "week": [1, 1, 1],
                "prediction": [250.5, 180.3, 120.7],
            }
        )

        result = cache_manager._validate_predictions(df, "passing_yards")

        assert len(result) == 1
        assert result["player_id"].iloc[0] == "P001"

    def test_validate_predictions_removes_invalid_seasons(self, cache_manager):
        """Test invalid season removal."""
        df = pd.DataFrame(
            {
                "player_id": ["P001", "P002", "P003"],
                "season": [2024, 2010, 2040],  # 2010 and 2040 out of range
                "week": [1, 1, 1],
                "prediction": [250.5, 180.3, 120.7],
            }
        )

        result = cache_manager._validate_predictions(df, "passing_yards")

        assert len(result) == 1
        assert result["season"].iloc[0] == 2024

    def test_validate_predictions_removes_invalid_weeks(self, cache_manager):
        """Test invalid week removal."""
        df = pd.DataFrame(
            {
                "player_id": ["P001", "P002", "P003"],
                "season": [2024, 2024, 2024],
                "week": [1, 0, 25],  # 0 and 25 out of range
                "prediction": [250.5, 180.3, 120.7],
            }
        )

        result = cache_manager._validate_predictions(df, "passing_yards")

        assert len(result) == 1
        assert result["week"].iloc[0] == 1


class TestCacheStatsDataclass:
    """Test CacheStats dataclass."""

    def test_cache_stats_initialization(self):
        """Test CacheStats initialization."""
        stats = CacheStats(
            total_predictions=1000,
            unique_players=150,
            unique_models=4,
            cache_size_mb=25.5,
            hit_rate=80.0,
            avg_lookup_time_ms=0.65,
            oldest_prediction=datetime(2024, 1, 1),
            newest_prediction=datetime(2024, 11, 22),
            predictions_by_model={"state_space_v2": 400},
            predictions_by_stat_type={"passing_yards": 600},
        )

        assert stats.total_predictions == 1000
        assert stats.hit_rate == 80.0


class TestValidationReportDataclass:
    """Test ValidationReport dataclass."""

    def test_validation_report_is_valid(self):
        """Test is_valid() method."""
        report = ValidationReport(
            total_checked=100,
            valid_predictions=100,
            invalid_predictions=0,
            duplicate_predictions=0,
            nan_predictions=0,
            out_of_range_predictions=0,
            orphaned_predictions=0,
            issues=[],
        )

        assert report.is_valid()

    def test_validation_report_is_invalid(self):
        """Test is_valid() returns False for invalid cache."""
        report = ValidationReport(
            total_checked=100,
            valid_predictions=95,
            invalid_predictions=5,
            duplicate_predictions=0,
            nan_predictions=5,
            out_of_range_predictions=0,
            orphaned_predictions=0,
            issues=["5 NaN predictions found"],
        )

        assert not report.is_valid()

    def test_validation_report_summary(self):
        """Test summary() method."""
        report = ValidationReport(
            total_checked=100,
            valid_predictions=95,
            invalid_predictions=5,
            duplicate_predictions=2,
            nan_predictions=3,
            out_of_range_predictions=2,
            orphaned_predictions=1,
            issues=["Issue 1", "Issue 2"],
        )

        summary = report.summary()

        assert "Total Checked: 100" in summary
        assert "Valid: 95" in summary
        assert "Issue 1" in summary
