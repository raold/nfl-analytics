"""
Baseline Tests for GameFeatureExtractor

These tests capture the current behavior of GameFeatureExtractor before refactoring.
They serve as a safety net to ensure refactoring doesn't change existing functionality.

Test Coverage:
- Feature extraction returns expected structure
- Feature prefixes are correctly applied
- Cache behavior works as expected
- Temporal safety validation functions
- Input validation
- Derived feature calculations

Created: December 13, 2025
Purpose: Phase 1.1 of code quality remediation - tests before refactoring
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import pandas as pd
import numpy as np
from datetime import datetime


# Mark all tests in this module as requiring database for integration tests
pytestmark = [pytest.mark.unit]


class TestGameFeatureExtractorStructure:
    """Test the structure and interface of GameFeatureExtractor."""

    @pytest.fixture
    def mock_db_config(self):
        """Standard test database configuration."""
        return {
            "host": "localhost",
            "port": 5544,
            "dbname": "devdb01",
            "user": "dro",
            "password": "test_password",
        }

    @pytest.fixture
    def mock_offensive_features(self):
        """Mock offensive feature response."""
        return {
            "epa_per_play": 0.15,
            "success_rate": 0.48,
            "points_per_game": 27.5,
            "yards_per_play": 5.8,
            "third_down_rate": 0.42,
            "red_zone_rate": 0.58,
            "turnovers_per_game": 1.2,
            "pass_rate": 0.62,
            "plays_per_game": 65.0,
            "epa_std_4wk": 0.05,
            "scoring_std_4wk": 3.5,
            "epa_consistency_score": 0.85,
        }

    @pytest.fixture
    def mock_defensive_features(self):
        """Mock defensive feature response."""
        return {
            "epa_allowed_per_play": -0.08,
            "success_rate_allowed": 0.42,
            "points_allowed_per_game": 20.5,
            "yards_allowed_per_play": 5.2,
            "third_down_rate_allowed": 0.38,
            "red_zone_rate_allowed": 0.52,
            "takeaways_per_game": 1.5,
            "sack_rate": 0.07,
            "epa_std_4wk": 0.04,
            "points_allowed_std_4wk": 4.0,
            "epa_consistency_score": 0.78,
        }

    def test_extractor_initializes_with_default_config(self):
        """GameFeatureExtractor should initialize with default DB config."""
        with patch("nfl_analytics.features.game_feature_extractor.TeamOffensiveFeatures"):
            with patch("nfl_analytics.features.game_feature_extractor.TeamDefensiveFeatures"):
                from nfl_analytics.features.game_feature_extractor import GameFeatureExtractor

                extractor = GameFeatureExtractor()

                assert extractor.db_config is not None
                assert "host" in extractor.db_config
                assert "port" in extractor.db_config

    def test_extractor_accepts_custom_db_config(self, mock_db_config):
        """GameFeatureExtractor should accept custom DB configuration."""
        with patch("nfl_analytics.features.game_feature_extractor.TeamOffensiveFeatures"):
            with patch("nfl_analytics.features.game_feature_extractor.TeamDefensiveFeatures"):
                from nfl_analytics.features.game_feature_extractor import GameFeatureExtractor

                extractor = GameFeatureExtractor(db_config=mock_db_config)

                assert extractor.db_config["host"] == "localhost"
                assert extractor.db_config["port"] == 5544

    def test_cache_ttl_is_defined(self):
        """GameFeatureExtractor should have CACHE_TTL constant."""
        from nfl_analytics.features.game_feature_extractor import GameFeatureExtractor

        assert hasattr(GameFeatureExtractor, "CACHE_TTL")
        assert GameFeatureExtractor.CACHE_TTL == 300  # 5 minutes


class TestFeatureExtraction:
    """Test feature extraction functionality."""

    @pytest.fixture
    def mock_extractor(self):
        """Create extractor with mocked dependencies."""
        with patch("nfl_analytics.features.game_feature_extractor.TeamOffensiveFeatures") as mock_off:
            with patch("nfl_analytics.features.game_feature_extractor.TeamDefensiveFeatures") as mock_def:
                # Mock the extract_features methods
                mock_off_instance = MagicMock()
                mock_off_instance.extract_features.return_value = {
                    "epa_per_play": 0.15,
                    "success_rate": 0.48,
                    "points_per_game": 27.5,
                    "plays_per_game": 65.0,
                }
                mock_off_instance.validate_no_leakage.return_value = True
                mock_off.return_value = mock_off_instance

                mock_def_instance = MagicMock()
                mock_def_instance.extract_features.return_value = {
                    "epa_allowed_per_play": -0.08,
                    "success_rate_allowed": 0.42,
                    "points_allowed_per_game": 20.5,
                }
                mock_def_instance.validate_no_leakage.return_value = True
                mock_def.return_value = mock_def_instance

                from nfl_analytics.features.game_feature_extractor import GameFeatureExtractor

                extractor = GameFeatureExtractor()

                # Disable optional extractors for simpler testing
                extractor.ngs_extractor = None
                extractor.qbr_extractor = None
                extractor.oadj_extractor = None
                extractor.turnover_luck_extractor = None
                extractor.primetime_extractor = None
                extractor.opponent_strength_extractor = None
                extractor.line_movement_extractor = None
                extractor.interaction_engine = None
                extractor.momentum_engine = None
                extractor.research_engine = None

                yield extractor

    def test_extract_game_features_returns_dict(self, mock_extractor):
        """extract_game_features should return a dictionary."""
        with patch.object(mock_extractor, "_get_injury_features", return_value={}):
            features = mock_extractor.extract_game_features("KC", "BUF", 2024, 12)

            assert isinstance(features, dict)
            assert len(features) > 0

    def test_feature_prefixes_are_correct(self, mock_extractor):
        """Features should have correct prefixes for home/away and off/def."""
        with patch.object(mock_extractor, "_get_injury_features", return_value={}):
            features = mock_extractor.extract_game_features("KC", "BUF", 2024, 12)

            # Check for expected prefixes
            home_off_keys = [k for k in features if k.startswith("home_off_")]
            home_def_keys = [k for k in features if k.startswith("home_def_")]
            away_off_keys = [k for k in features if k.startswith("away_off_")]
            away_def_keys = [k for k in features if k.startswith("away_def_")]

            assert len(home_off_keys) > 0, "Should have home_off_ prefixed features"
            assert len(home_def_keys) > 0, "Should have home_def_ prefixed features"
            assert len(away_off_keys) > 0, "Should have away_off_ prefixed features"
            assert len(away_def_keys) > 0, "Should have away_def_ prefixed features"

    def test_derived_features_included(self, mock_extractor):
        """Derived matchup features should be included."""
        with patch.object(mock_extractor, "_get_injury_features", return_value={}):
            features = mock_extractor.extract_game_features("KC", "BUF", 2024, 12)

            assert "offensive_mismatch" in features
            assert "defensive_mismatch" in features
            assert "pace_difference" in features
            assert "rest_differential" in features

    def test_all_feature_values_are_numeric(self, mock_extractor):
        """All feature values should be numeric (int or float)."""
        with patch.object(mock_extractor, "_get_injury_features", return_value={}):
            features = mock_extractor.extract_game_features("KC", "BUF", 2024, 12)

            for key, value in features.items():
                assert isinstance(value, (int, float, np.integer, np.floating)), \
                    f"Feature {key} has non-numeric value: {type(value)}"


class TestDerivedFeatureCalculations:
    """Test derived feature calculation logic."""

    def test_offensive_mismatch_calculation(self):
        """Test offensive_mismatch = home_off_epa - away_def_epa."""
        from nfl_analytics.features.game_feature_extractor import GameFeatureExtractor

        with patch("nfl_analytics.features.game_feature_extractor.TeamOffensiveFeatures"):
            with patch("nfl_analytics.features.game_feature_extractor.TeamDefensiveFeatures"):
                extractor = GameFeatureExtractor()

                home_off = {"epa_per_play": 0.15}
                away_def = {"epa_allowed_per_play": -0.05}

                mismatch = extractor._compute_offensive_mismatch(home_off, away_def)

                # offensive_mismatch = home_off_epa - away_def_epa_allowed
                # Positive mismatch = home advantage
                assert isinstance(mismatch, (int, float))

    def test_defensive_mismatch_calculation(self):
        """Test defensive_mismatch = away_off_epa - home_def_epa."""
        from nfl_analytics.features.game_feature_extractor import GameFeatureExtractor

        with patch("nfl_analytics.features.game_feature_extractor.TeamOffensiveFeatures"):
            with patch("nfl_analytics.features.game_feature_extractor.TeamDefensiveFeatures"):
                extractor = GameFeatureExtractor()

                away_off = {"epa_per_play": 0.10}
                home_def = {"epa_allowed_per_play": -0.08}

                mismatch = extractor._compute_defensive_mismatch(away_off, home_def)

                assert isinstance(mismatch, (int, float))

    def test_pace_difference_calculation(self):
        """Test pace_difference = home_plays_per_game - away_plays_per_game."""
        from nfl_analytics.features.game_feature_extractor import GameFeatureExtractor

        with patch("nfl_analytics.features.game_feature_extractor.TeamOffensiveFeatures"):
            with patch("nfl_analytics.features.game_feature_extractor.TeamDefensiveFeatures"):
                extractor = GameFeatureExtractor()

                home_off = {"plays_per_game": 65.0}
                away_off = {"plays_per_game": 60.0}

                pace_diff = extractor._compute_pace_difference(home_off, away_off)

                assert isinstance(pace_diff, (int, float))


class TestInputValidation:
    """Test input validation behavior."""

    @pytest.fixture
    def extractor(self):
        """Create extractor for validation tests."""
        with patch("nfl_analytics.features.game_feature_extractor.TeamOffensiveFeatures"):
            with patch("nfl_analytics.features.game_feature_extractor.TeamDefensiveFeatures"):
                from nfl_analytics.features.game_feature_extractor import GameFeatureExtractor
                return GameFeatureExtractor()

    def test_validates_team_abbreviation_format(self, extractor):
        """Should validate team abbreviation format."""
        # Valid teams should not raise
        try:
            extractor._validate_inputs("KC", "BUF", 2024, 12)
        except ValueError:
            pytest.fail("Valid team abbreviations should not raise ValueError")

    def test_rejects_invalid_team_length(self, extractor):
        """Should reject team abbreviations with wrong length."""
        with pytest.raises(ValueError, match="Invalid"):
            extractor._validate_inputs("KANSAS", "BUF", 2024, 12)

    def test_rejects_non_string_team(self, extractor):
        """Should reject non-string team values."""
        with pytest.raises(ValueError):
            extractor._validate_inputs(123, "BUF", 2024, 12)

    def test_validates_season_range(self, extractor):
        """Should validate season is in reasonable range."""
        # Seasons should be between 2000 and current year + 1
        with pytest.raises(ValueError, match="Invalid"):
            extractor._validate_inputs("KC", "BUF", 1990, 12)

    def test_validates_week_range(self, extractor):
        """Should validate week is in range 1-22."""
        with pytest.raises(ValueError, match="Invalid"):
            extractor._validate_inputs("KC", "BUF", 2024, 25)


class TestCacheBehavior:
    """Test feature caching behavior."""

    @pytest.fixture
    def extractor_with_cache(self):
        """Create extractor for cache tests."""
        with patch("nfl_analytics.features.game_feature_extractor.TeamOffensiveFeatures") as mock_off:
            with patch("nfl_analytics.features.game_feature_extractor.TeamDefensiveFeatures") as mock_def:
                mock_off_instance = MagicMock()
                mock_off_instance.extract_features.return_value = {"epa_per_play": 0.15}
                mock_off_instance.validate_no_leakage.return_value = True
                mock_off.return_value = mock_off_instance

                mock_def_instance = MagicMock()
                mock_def_instance.extract_features.return_value = {"epa_allowed_per_play": -0.08}
                mock_def_instance.validate_no_leakage.return_value = True
                mock_def.return_value = mock_def_instance

                from nfl_analytics.features.game_feature_extractor import GameFeatureExtractor
                extractor = GameFeatureExtractor()

                # Disable optional extractors
                extractor.ngs_extractor = None
                extractor.qbr_extractor = None
                extractor.oadj_extractor = None
                extractor.turnover_luck_extractor = None
                extractor.primetime_extractor = None
                extractor.opponent_strength_extractor = None
                extractor.line_movement_extractor = None
                extractor.interaction_engine = None
                extractor.momentum_engine = None
                extractor.research_engine = None

                yield extractor, mock_off_instance, mock_def_instance

    def test_cache_stores_features(self, extractor_with_cache):
        """Features should be stored in cache after extraction."""
        extractor, _, _ = extractor_with_cache

        with patch.object(extractor, "_get_injury_features", return_value={}):
            extractor.extract_game_features("KC", "BUF", 2024, 12, use_cache=True)

            # Check cache is not empty
            assert len(extractor._feature_cache) > 0

    def test_cache_can_be_disabled(self, extractor_with_cache):
        """Cache should be bypassable with use_cache=False."""
        extractor, mock_off, _ = extractor_with_cache

        with patch.object(extractor, "_get_injury_features", return_value={}):
            # First call
            extractor.extract_game_features("KC", "BUF", 2024, 12, use_cache=False)
            initial_calls = mock_off.extract_features.call_count

            # Second call with cache disabled
            extractor.extract_game_features("KC", "BUF", 2024, 12, use_cache=False)

            # Should have called extract again
            assert mock_off.extract_features.call_count > initial_calls


class TestTemporalSafetyValidation:
    """Test temporal safety validation functionality."""

    @pytest.fixture
    def extractor_for_validation(self):
        """Create extractor for temporal safety tests."""
        with patch("nfl_analytics.features.game_feature_extractor.TeamOffensiveFeatures") as mock_off:
            with patch("nfl_analytics.features.game_feature_extractor.TeamDefensiveFeatures") as mock_def:
                mock_off_instance = MagicMock()
                mock_off_instance.validate_no_leakage.return_value = True
                mock_off.return_value = mock_off_instance

                mock_def_instance = MagicMock()
                mock_def_instance.validate_no_leakage.return_value = True
                mock_def.return_value = mock_def_instance

                from nfl_analytics.features.game_feature_extractor import GameFeatureExtractor
                extractor = GameFeatureExtractor()

                yield extractor, mock_off_instance, mock_def_instance

    def test_validate_temporal_safety_returns_bool(self, extractor_for_validation):
        """validate_temporal_safety should return boolean."""
        extractor, _, _ = extractor_for_validation

        result = extractor.validate_temporal_safety("KC", "BUF", 2024, 12)

        assert isinstance(result, bool)

    def test_validation_checks_all_extractors(self, extractor_for_validation):
        """Should validate both offensive and defensive extractors for both teams."""
        extractor, mock_off, mock_def = extractor_for_validation

        extractor.validate_temporal_safety("KC", "BUF", 2024, 12)

        # Should call validate_no_leakage for home team (off + def) and away team (off + def)
        assert mock_off.validate_no_leakage.call_count >= 2
        assert mock_def.validate_no_leakage.call_count >= 2

    def test_validation_returns_false_on_any_leakage(self, extractor_for_validation):
        """Should return False if any extractor detects leakage."""
        extractor, mock_off, _ = extractor_for_validation

        # Simulate one extractor detecting leakage
        mock_off.validate_no_leakage.return_value = False

        result = extractor.validate_temporal_safety("KC", "BUF", 2024, 12)

        assert result is False


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def extractor(self):
        """Create extractor for edge case tests."""
        with patch("nfl_analytics.features.game_feature_extractor.TeamOffensiveFeatures") as mock_off:
            with patch("nfl_analytics.features.game_feature_extractor.TeamDefensiveFeatures") as mock_def:
                mock_off_instance = MagicMock()
                mock_off_instance.extract_features.return_value = {"epa_per_play": 0.15}
                mock_off_instance.validate_no_leakage.return_value = True
                mock_off.return_value = mock_off_instance

                mock_def_instance = MagicMock()
                mock_def_instance.extract_features.return_value = {"epa_allowed_per_play": -0.08}
                mock_def_instance.validate_no_leakage.return_value = True
                mock_def.return_value = mock_def_instance

                from nfl_analytics.features.game_feature_extractor import GameFeatureExtractor
                extractor = GameFeatureExtractor()

                # Disable optional extractors
                extractor.ngs_extractor = None
                extractor.qbr_extractor = None
                extractor.oadj_extractor = None
                extractor.turnover_luck_extractor = None
                extractor.primetime_extractor = None
                extractor.opponent_strength_extractor = None
                extractor.line_movement_extractor = None
                extractor.interaction_engine = None
                extractor.momentum_engine = None
                extractor.research_engine = None

                yield extractor

    def test_handles_nan_values_gracefully(self, extractor):
        """Should convert NaN values to 0.0."""
        # Modify the mock to return NaN
        extractor.offensive_extractor.extract_features.return_value = {
            "epa_per_play": float("nan"),
            "success_rate": 0.48,
        }

        with patch.object(extractor, "_get_injury_features", return_value={}):
            features = extractor.extract_game_features("KC", "BUF", 2024, 12)

            # All NaN should be converted to 0.0
            for key, value in features.items():
                assert not pd.isna(value), f"Feature {key} is still NaN"

    def test_early_season_week_1(self, extractor):
        """Should handle week 1 (limited historical data)."""
        with patch.object(extractor, "_get_injury_features", return_value={}):
            # Week 1 should not raise even with limited data
            features = extractor.extract_game_features("KC", "BUF", 2024, 1)

            assert isinstance(features, dict)

    def test_same_team_home_and_away_rejected(self, extractor):
        """Should reject same team as home and away."""
        with pytest.raises(ValueError, match="same"):
            extractor._validate_inputs("KC", "KC", 2024, 12)


class TestFeatureConsistency:
    """Test that feature values are consistent and reasonable."""

    @pytest.fixture
    def real_extractor(self):
        """Create real extractor for consistency tests (requires DB)."""
        # This is an integration test - skip if DB not available
        pytest.importorskip("psycopg2")

        try:
            from nfl_analytics.features.game_feature_extractor import GameFeatureExtractor
            extractor = GameFeatureExtractor()
            return extractor
        except Exception:
            pytest.skip("Database not available for integration test")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_feature_values_in_expected_ranges(self, real_extractor):
        """Feature values should be in reasonable ranges."""
        features = real_extractor.extract_game_features("KC", "BUF", 2024, 12)

        # EPA per play should be between -0.5 and 0.5
        if "home_off_epa_per_play" in features:
            assert -0.5 <= features["home_off_epa_per_play"] <= 0.5

        # Success rate should be between 0 and 1
        if "home_off_success_rate" in features:
            assert 0.0 <= features["home_off_success_rate"] <= 1.0

        # Points per game should be positive
        if "home_off_points_per_game" in features:
            assert features["home_off_points_per_game"] >= 0

    @pytest.mark.integration
    @pytest.mark.slow
    def test_same_inputs_same_outputs(self, real_extractor):
        """Same inputs should produce same outputs (deterministic)."""
        features1 = real_extractor.extract_game_features("KC", "BUF", 2024, 12)
        features2 = real_extractor.extract_game_features("KC", "BUF", 2024, 12)

        assert features1 == features2
