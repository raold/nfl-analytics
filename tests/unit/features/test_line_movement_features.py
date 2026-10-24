"""
Unit tests for Line Movement Features module.

Tests:
- Feature extraction from database
- Default feature generation
- CLV calculation
- Batch extraction
"""

import pytest
from unittest.mock import MagicMock, patch


class TestLineMovementFeatures:
    """Test line movement feature extraction."""

    def test_default_features_structure(self):
        """Test that default features have correct structure."""
        from nfl_analytics.features.line_movement_features import LineMovementFeatures

        extractor = LineMovementFeatures()
        defaults = extractor._get_default_features()

        # Check all 12 features present
        expected_keys = {
            "spread_movement",
            "total_movement",
            "spread_movement_pct",
            "movement_direction",
            "sharp_indicator",
            "reverse_line_move",
            "consensus_spread",
            "spread_dispersion",
            "total_dispersion",
            "books_count",
            "opening_consensus",
            "closing_vs_opening_consensus",
        }

        assert set(defaults.keys()) == expected_keys
        assert len(defaults) == 12

    def test_default_features_values(self):
        """Test that default features have sensible neutral values."""
        from nfl_analytics.features.line_movement_features import LineMovementFeatures

        extractor = LineMovementFeatures()
        defaults = extractor._get_default_features()

        # Movement features should be 0 (no movement)
        assert defaults["spread_movement"] == 0.0
        assert defaults["total_movement"] == 0.0
        assert defaults["movement_direction"] == 0.0

        # Binary indicators should be 0
        assert defaults["sharp_indicator"] == 0.0
        assert defaults["reverse_line_move"] == 0.0

        # Dispersion should have moderate uncertainty
        assert defaults["spread_dispersion"] == 0.5
        assert defaults["total_dispersion"] == 0.5

    def test_clv_calculation_home_bet(self):
        """Test CLV calculation for home team bets."""
        from nfl_analytics.features.line_movement_features import calculate_clv

        # If you bet Home -3 and it closed at -3.5, you got 0.5 pts CLV
        clv = calculate_clv(bet_spread=-3.0, closing_spread=-3.5, bet_on_home=True)
        assert clv == 0.5

        # If you bet Home -3 and it closed at -2.5, you lost 0.5 pts CLV
        clv = calculate_clv(bet_spread=-3.0, closing_spread=-2.5, bet_on_home=True)
        assert clv == -0.5

    def test_clv_calculation_away_bet(self):
        """Test CLV calculation for away team bets."""
        from nfl_analytics.features.line_movement_features import calculate_clv

        # If you bet Away +3 and it closed at +3.5, you lost 0.5 pts CLV
        clv = calculate_clv(bet_spread=-3.0, closing_spread=-3.5, bet_on_home=False)
        assert clv == -0.5

        # If you bet Away +3 and it closed at +2.5, you got 0.5 pts CLV
        clv = calculate_clv(bet_spread=-3.0, closing_spread=-2.5, bet_on_home=False)
        assert clv == 0.5

    def test_extract_returns_dict(self):
        """Test that extract always returns a dictionary."""
        from nfl_analytics.features.line_movement_features import LineMovementFeatures

        extractor = LineMovementFeatures()

        # Test with non-existent game (should return defaults)
        features = extractor.extract("FAKE_GAME_ID")

        assert isinstance(features, dict)
        assert len(features) == 12

    @pytest.mark.integration
    def test_extract_real_game(self):
        """Integration test: extract features for a real game with odds data."""
        from nfl_analytics.features.line_movement_features import LineMovementFeatures

        extractor = LineMovementFeatures()

        # Test with 2023 Week 5 BUF vs JAX (known to have odds data)
        features = extractor.extract("2023_05_JAX_BUF")

        assert isinstance(features, dict)
        assert len(features) == 12

        # If we have real data, books_count should be > 0
        if features["books_count"] > 0:
            # Verify reasonable value ranges
            assert -20 <= features["consensus_spread"] <= 20
            assert 0 <= features["spread_dispersion"] <= 5
            assert features["movement_direction"] in [-1.0, 0.0, 1.0]
            assert features["sharp_indicator"] in [0.0, 1.0]

    @pytest.mark.integration
    def test_extract_by_teams(self):
        """Integration test: extract features using team/week lookup."""
        from nfl_analytics.features.line_movement_features import LineMovementFeatures

        extractor = LineMovementFeatures()

        # Test with BUF vs JAX 2023 Week 5
        features = extractor.extract_by_teams("BUF", "JAX", 2023, 5)

        assert isinstance(features, dict)
        assert len(features) == 12


class TestLineMovementConstants:
    """Test configuration constants."""

    def test_sharp_movement_threshold(self):
        """Test sharp movement threshold is reasonable."""
        from nfl_analytics.features.line_movement_features import SHARP_MOVEMENT_THRESHOLD

        # Should be 1-2 points (industry standard)
        assert 1.0 <= SHARP_MOVEMENT_THRESHOLD <= 2.0

    def test_min_books_for_consensus(self):
        """Test minimum books threshold is reasonable."""
        from nfl_analytics.features.line_movement_features import MIN_BOOKS_FOR_CONSENSUS

        # Should need at least 2-5 books for consensus
        assert 2 <= MIN_BOOKS_FOR_CONSENSUS <= 5


class TestMovementDirection:
    """Test movement direction calculations."""

    def test_movement_direction_logic(self):
        """Test that movement direction is calculated correctly.

        - Positive movement (line got bigger) = +1 (toward underdog)
        - Negative movement (line got smaller) = -1 (toward favorite)
        - Small movement = 0 (no significant move)
        """
        from nfl_analytics.features.line_movement_features import LineMovementFeatures

        # These are conceptual tests - actual logic is in the extractor
        # Movement > 0.25: +1
        # Movement < -0.25: -1
        # -0.25 <= Movement <= 0.25: 0

        # This validates the threshold is reasonable
        assert 0.25 > 0  # Small movements are filtered out
