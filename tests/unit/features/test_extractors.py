"""Tests for the feature extractor Strategy pattern implementation.

This module tests the base classes and registry that form the foundation
of the feature extraction architecture refactoring.

Created: December 13, 2025
Phase 2 tests for Strategy pattern implementation.
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from nfl_analytics.features.extractors.base import (
    ExtractionLevel,
    FeatureExtractionError,
    FeatureExtractionResult,
    FeatureExtractorStrategy,
    FeaturePriority,
    GameLevelExtractorStrategy,
)
from nfl_analytics.features.extractors.registry import FeatureRegistry

# ============================================================================
# Test Fixtures - Mock Extractors
# ============================================================================


class MockOffensiveExtractor(FeatureExtractorStrategy):
    """Mock offensive feature extractor for testing."""

    @property
    def name(self) -> str:
        return "offensive_features"

    @property
    def prefix(self) -> str:
        return "off"

    @property
    def priority(self) -> FeaturePriority:
        return FeaturePriority.CRITICAL

    def extract(self, team, season, week, conn=None, opponent=None):
        return {
            "epa_per_play": 0.15,
            "success_rate": 0.45,
            "points_per_game": 24.5,
        }

    def get_feature_names(self):
        return ["epa_per_play", "success_rate", "points_per_game"]


class MockDefensiveExtractor(FeatureExtractorStrategy):
    """Mock defensive feature extractor for testing."""

    @property
    def name(self) -> str:
        return "defensive_features"

    @property
    def prefix(self) -> str:
        return "def"

    @property
    def priority(self) -> FeaturePriority:
        return FeaturePriority.CRITICAL

    def extract(self, team, season, week, conn=None, opponent=None):
        return {
            "epa_allowed_per_play": -0.05,
            "sack_rate": 0.08,
            "points_allowed_per_game": 20.5,
        }

    def get_feature_names(self):
        return ["epa_allowed_per_play", "sack_rate", "points_allowed_per_game"]


class MockNextGenExtractor(FeatureExtractorStrategy):
    """Mock NextGen stats extractor for testing."""

    @property
    def name(self) -> str:
        return "nextgen_stats"

    @property
    def prefix(self) -> str:
        return "ngs"

    @property
    def priority(self) -> FeaturePriority:
        return FeaturePriority.HIGH

    def extract(self, team, season, week, conn=None, opponent=None):
        return {
            "time_to_throw": 2.8,
            "cpoe": 3.5,
        }

    def get_feature_names(self):
        return ["time_to_throw", "cpoe"]


class MockFailingExtractor(FeatureExtractorStrategy):
    """Mock extractor that always fails for testing error handling."""

    @property
    def name(self) -> str:
        return "failing_extractor"

    @property
    def prefix(self) -> str:
        return "fail"

    @property
    def priority(self) -> FeaturePriority:
        return FeaturePriority.LOW

    def extract(self, team, season, week, conn=None, opponent=None):
        raise FeatureExtractionError(self.name, "Intentional test failure")

    def get_feature_names(self):
        return ["feature1", "feature2"]


class MockGameLevelExtractor(GameLevelExtractorStrategy):
    """Mock game-level extractor for testing."""

    @property
    def name(self) -> str:
        return "game_features"

    @property
    def prefix(self) -> str:
        return ""  # No prefix for game-level features

    @property
    def priority(self) -> FeaturePriority:
        return FeaturePriority.MEDIUM

    def extract_game(self, home_team, away_team, season, week, conn=None):
        return {
            "offensive_mismatch": 0.10,
            "defensive_mismatch": -0.05,
            "pace_difference": 2.5,
        }

    def get_feature_names(self):
        return ["offensive_mismatch", "defensive_mismatch", "pace_difference"]


# ============================================================================
# Base Class Tests
# ============================================================================


class TestFeatureExtractionResult:
    """Tests for FeatureExtractionResult dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        result = FeatureExtractionResult()
        assert result.features == {}
        assert result.success is True
        assert result.error_message is None
        assert result.feature_count == 0

    def test_feature_count_auto_computed(self):
        """Test that feature_count is computed from features dict."""
        result = FeatureExtractionResult(features={"a": 1.0, "b": 2.0, "c": 3.0})
        assert result.feature_count == 3

    def test_failed_result(self):
        """Test creating a failed result."""
        result = FeatureExtractionResult(
            success=False, error_message="Test error", extractor_name="test_extractor"
        )
        assert not result.success
        assert result.error_message == "Test error"
        assert result.extractor_name == "test_extractor"


class TestFeatureExtractionError:
    """Tests for FeatureExtractionError exception."""

    def test_error_message_format(self):
        """Test error message includes extractor name."""
        error = FeatureExtractionError("test_extractor", "Something went wrong")
        assert "[test_extractor]" in str(error)
        assert "Something went wrong" in str(error)

    def test_original_error_preserved(self):
        """Test original exception is preserved."""
        original = ValueError("Original error")
        error = FeatureExtractionError("test", "Wrapped", original_error=original)
        assert error.original_error is original


class TestFeatureExtractorStrategy:
    """Tests for the FeatureExtractorStrategy base class."""

    def test_mock_extractor_properties(self):
        """Test that mock extractor implements required properties."""
        extractor = MockOffensiveExtractor()
        assert extractor.name == "offensive_features"
        assert extractor.prefix == "off"
        assert extractor.level == ExtractionLevel.TEAM
        assert extractor.priority == FeaturePriority.CRITICAL

    def test_extract_returns_features(self):
        """Test that extract returns expected features."""
        extractor = MockOffensiveExtractor()
        features = extractor.extract("KC", 2023, 10)

        assert "epa_per_play" in features
        assert "success_rate" in features
        assert features["epa_per_play"] == 0.15

    def test_get_feature_names(self):
        """Test get_feature_names returns correct list."""
        extractor = MockOffensiveExtractor()
        names = extractor.get_feature_names()

        assert len(names) == 3
        assert "epa_per_play" in names
        assert "success_rate" in names

    def test_get_prefixed_feature_names(self):
        """Test prefixed feature names are correct."""
        extractor = MockOffensiveExtractor()
        names = extractor.get_prefixed_feature_names()

        assert "off_epa_per_play" in names
        assert "off_success_rate" in names
        assert "off_points_per_game" in names

    def test_get_default_values(self):
        """Test default values are zeros."""
        extractor = MockOffensiveExtractor()
        defaults = extractor.get_default_values()

        assert len(defaults) == 3
        assert all(v == 0.0 for v in defaults.values())

    def test_validate_inputs_valid(self):
        """Test input validation with valid inputs."""
        extractor = MockOffensiveExtractor()
        # Should not raise
        extractor.validate_inputs("KC", 2023, 10)

    def test_validate_inputs_invalid_team(self):
        """Test input validation catches invalid team."""
        extractor = MockOffensiveExtractor()
        with pytest.raises(ValueError, match="Invalid team"):
            extractor.validate_inputs("", 2023, 10)

    def test_validate_inputs_invalid_season(self):
        """Test input validation catches invalid season."""
        extractor = MockOffensiveExtractor()
        with pytest.raises(ValueError, match="Invalid season"):
            extractor.validate_inputs("KC", 1900, 10)

    def test_validate_inputs_invalid_week(self):
        """Test input validation catches invalid week."""
        extractor = MockOffensiveExtractor()
        with pytest.raises(ValueError, match="Invalid week"):
            extractor.validate_inputs("KC", 2023, 25)

    def test_safe_extract_success(self):
        """Test safe_extract with successful extraction."""
        extractor = MockOffensiveExtractor()
        result = extractor.safe_extract("KC", 2023, 10)

        assert result.success
        assert result.error_message is None
        assert result.feature_count == 3
        assert result.extraction_time_ms >= 0

    def test_safe_extract_handles_error(self):
        """Test safe_extract handles errors gracefully."""
        extractor = MockFailingExtractor()
        result = extractor.safe_extract("KC", 2023, 10)

        assert not result.success
        assert result.error_message is not None
        # Should have default values
        assert result.feature_count == 2
        assert all(v == 0.0 for v in result.features.values())


class TestGameLevelExtractorStrategy:
    """Tests for GameLevelExtractorStrategy base class."""

    def test_level_is_game(self):
        """Test that game-level extractor has GAME level."""
        extractor = MockGameLevelExtractor()
        assert extractor.level == ExtractionLevel.GAME

    def test_extract_calls_extract_game(self):
        """Test that extract() delegates to extract_game()."""
        extractor = MockGameLevelExtractor()
        features = extractor.extract("KC", 2023, 10, opponent="BUF")

        assert "offensive_mismatch" in features
        assert "defensive_mismatch" in features

    def test_extract_requires_opponent(self):
        """Test that extract() requires opponent for game-level."""
        extractor = MockGameLevelExtractor()
        with pytest.raises(FeatureExtractionError, match="require opponent"):
            extractor.extract("KC", 2023, 10)


# ============================================================================
# Registry Tests
# ============================================================================


class TestFeatureRegistry:
    """Tests for FeatureRegistry class."""

    def test_register_extractor(self):
        """Test registering an extractor."""
        registry = FeatureRegistry()
        extractor = MockOffensiveExtractor()

        registry.register(extractor)

        assert "offensive_features" in registry.list_extractors()
        assert registry.get_extractor("offensive_features") is extractor

    def test_register_duplicate_raises(self):
        """Test that registering duplicate name raises error."""
        registry = FeatureRegistry()
        extractor1 = MockOffensiveExtractor()
        extractor2 = MockOffensiveExtractor()

        registry.register(extractor1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(extractor2)

    def test_unregister_extractor(self):
        """Test unregistering an extractor."""
        registry = FeatureRegistry()
        extractor = MockOffensiveExtractor()

        registry.register(extractor)
        registry.unregister("offensive_features")

        assert "offensive_features" not in registry.list_extractors()

    def test_extraction_order_by_priority(self):
        """Test that extractors are ordered by priority."""
        registry = FeatureRegistry()

        # Register in non-priority order
        registry.register(MockNextGenExtractor())  # HIGH
        registry.register(MockFailingExtractor())  # LOW
        registry.register(MockOffensiveExtractor())  # CRITICAL

        order = registry.list_extractors()

        # Critical should be first, Low should be last
        assert order.index("offensive_features") < order.index("nextgen_stats")
        assert order.index("nextgen_stats") < order.index("failing_extractor")

    def test_get_all_feature_names(self):
        """Test getting all feature names with prefixes."""
        registry = FeatureRegistry()
        registry.register(MockOffensiveExtractor())
        registry.register(MockDefensiveExtractor())

        names = registry.get_all_feature_names(include_home_away=True)

        # Should have home and away versions
        assert "home_off_epa_per_play" in names
        assert "away_off_epa_per_play" in names
        assert "home_def_epa_allowed_per_play" in names
        assert "away_def_epa_allowed_per_play" in names

    def test_get_all_feature_names_without_home_away(self):
        """Test getting feature names without home/away prefix."""
        registry = FeatureRegistry()
        registry.register(MockOffensiveExtractor())

        names = registry.get_all_feature_names(include_home_away=False)

        assert "off_epa_per_play" in names
        assert "home_off_epa_per_play" not in names

    @patch("psycopg2.connect")
    def test_extract_all_team_features(self, mock_connect):
        """Test extracting all features for a game."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        registry = FeatureRegistry(parallel=False)
        registry.register(MockOffensiveExtractor())
        registry.register(MockDefensiveExtractor())

        summary = registry.extract_all("KC", "BUF", 2023, 10)

        assert summary.success
        assert summary.total_features > 0

        # Check home features exist
        assert "home_off_epa_per_play" in summary.features
        assert "home_def_epa_allowed_per_play" in summary.features

        # Check away features exist
        assert "away_off_epa_per_play" in summary.features
        assert "away_def_epa_allowed_per_play" in summary.features

    @patch("psycopg2.connect")
    def test_extract_all_with_game_level(self, mock_connect):
        """Test extracting with game-level extractors."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        registry = FeatureRegistry(parallel=False)
        registry.register(MockGameLevelExtractor())

        summary = registry.extract_all("KC", "BUF", 2023, 10)

        assert summary.success
        # Game-level features should not have home/away prefix
        assert "offensive_mismatch" in summary.features
        assert "defensive_mismatch" in summary.features

    @patch("psycopg2.connect")
    def test_extract_all_handles_failures(self, mock_connect):
        """Test that extraction handles non-critical failures gracefully."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        registry = FeatureRegistry(parallel=False, fail_on_critical=False)
        registry.register(MockOffensiveExtractor())
        registry.register(MockFailingExtractor())

        summary = registry.extract_all("KC", "BUF", 2023, 10)

        # Should still succeed overall (failing extractor is LOW priority)
        assert summary.success
        assert "failing_extractor" in summary.failed_extractors

        # Should have default values for failed extractor
        assert "home_fail_feature1" in summary.features
        assert summary.features["home_fail_feature1"] == 0.0

    @patch("psycopg2.connect")
    def test_extraction_time_recorded(self, mock_connect):
        """Test that extraction time is recorded."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        registry = FeatureRegistry(parallel=False)
        registry.register(MockOffensiveExtractor())

        summary = registry.extract_all("KC", "BUF", 2023, 10)

        assert summary.extraction_time_ms >= 0

    @patch("psycopg2.connect")
    def test_extractor_results_recorded(self, mock_connect):
        """Test that individual extractor results are recorded."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        registry = FeatureRegistry(parallel=False)
        registry.register(MockOffensiveExtractor())
        registry.register(MockDefensiveExtractor())

        summary = registry.extract_all("KC", "BUF", 2023, 10)

        # Should have results for home and away for each extractor
        assert len(summary.extractor_results) == 4  # 2 extractors × 2 teams


# ============================================================================
# Integration Tests
# ============================================================================


class TestRegistryIntegration:
    """Integration tests for the registry with real extractors."""

    def test_registry_with_multiple_extractors(self):
        """Test registry handles multiple extractors correctly."""
        registry = FeatureRegistry(parallel=False)

        # Register multiple extractors
        registry.register(MockOffensiveExtractor())
        registry.register(MockDefensiveExtractor())
        registry.register(MockNextGenExtractor())
        registry.register(MockGameLevelExtractor())

        # Verify all registered
        assert len(registry.list_extractors()) == 4

        # Verify feature names
        names = registry.get_all_feature_names()
        assert len(names) > 0

        # Count expected features:
        # Offensive: 3 × 2 (home/away) = 6
        # Defensive: 3 × 2 = 6
        # NextGen: 2 × 2 = 4
        # Game: 3 (no home/away)
        assert len(names) == 6 + 6 + 4 + 3
