#!/usr/bin/env python3
"""
Unit Tests for DC (Defensive Coordinator) Features Module
==========================================================

Tests the DCFeatureGenerator class for correct feature computation,
exponential decay for new DC adjustment, continuity logic, and cache behavior.

Test Coverage:
1. Exponential decay computation (H=4 halflife)
2. DC continuity logic (returning vs new coordinator)
3. Feature extraction with valid data
4. Tenure calculation across seasons
5. Edge cases (new team, missing DC data)
6. LRU cache behavior
7. Opponent DC features

Author: Claude Code (Opus 4.5)
Date: November 27, 2025
"""

import math

# Add project root to path
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from py.features.dc_features import DCFeatureGenerator


class TestExponentialDecayAdjustment(unittest.TestCase):
    """Test exponential decay computation for new DC adjustment."""

    def setUp(self):
        """Initialize generator."""
        self.generator = DCFeatureGenerator(halflife_weeks=4.0)

    def test_new_dc_week1_adjustment(self):
        """New DC at week 1 should have 0.8 adjustment."""
        # games_with_dc = 0 at week 1
        # adjustment = 0.8 * exp(-0 / 4) = 0.8 * 1.0 = 0.8
        with patch.object(
            self.generator, "get_dc_continuity", return_value=(0, "New DC", "Old DC")
        ):
            features = self.generator.get_features("BUF", 2024, 1)

        self.assertAlmostEqual(
            features["new_dc_adjustment"],
            0.8,
            places=4,
            msg="New DC at week 1 should have 0.8 adjustment",
        )

    def test_new_dc_week5_adjustment(self):
        """New DC at week 5 should have decayed adjustment (~0.294)."""
        # games_with_dc = 4 at week 5
        # adjustment = 0.8 * exp(-4 / 4) = 0.8 * exp(-1) ≈ 0.294
        expected = 0.8 * math.exp(-4 / 4.0)

        with patch.object(
            self.generator, "get_dc_continuity", return_value=(0, "Bobby Babich", "Sean McDermott")
        ):
            with patch.object(self.generator, "get_dc_tenure", return_value=1):
                features = self.generator.get_features("BUF", 2024, 5)

        self.assertAlmostEqual(
            features["new_dc_adjustment"],
            round(expected, 4),
            places=4,
            msg=f"New DC at week 5 should have ~{expected:.3f} adjustment",
        )

    def test_returning_dc_no_adjustment(self):
        """Returning DC should have 0.0 adjustment."""
        with patch.object(
            self.generator,
            "get_dc_continuity",
            return_value=(1, "Steve Spagnuolo", "Steve Spagnuolo"),
        ):
            with patch.object(self.generator, "get_dc_tenure", return_value=6):
                features = self.generator.get_features("KC", 2024, 1)

        self.assertAlmostEqual(
            features["new_dc_adjustment"],
            0.0,
            places=4,
            msg="Returning DC should have 0.0 adjustment",
        )

    def test_decay_formula_halflife(self):
        """Verify half-life decay: at H=4 weeks, adjustment should halve."""
        # At week 5 (4 games played), adjustment = 0.8 * exp(-1) ≈ 0.294
        # At week 9 (8 games played), adjustment = 0.8 * exp(-2) ≈ 0.108

        with patch.object(
            self.generator, "get_dc_continuity", return_value=(0, "New DC", "Old DC")
        ):
            with patch.object(self.generator, "get_dc_tenure", return_value=1):
                features_week5 = self.generator.get_features("TEST", 2024, 5)
                features_week9 = self.generator.get_features("TEST", 2024, 9)

        # Ratio should be approximately exp(-1) ≈ 0.368
        ratio = features_week9["new_dc_adjustment"] / features_week5["new_dc_adjustment"]
        expected_ratio = math.exp(-1)

        self.assertAlmostEqual(
            ratio,
            expected_ratio,
            places=3,
            msg="Decay ratio should follow exponential half-life formula",
        )


class TestDCContinuityLogic(unittest.TestCase):
    """Test DC continuity determination."""

    def setUp(self):
        """Initialize generator with mocked database."""
        self.generator = DCFeatureGenerator()

    @patch("py.features.dc_features.psycopg2.connect")
    def test_continuity_same_dc(self, mock_connect):
        """Same DC as last season should return continuity=1."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("Steve Spagnuolo", "Steve Spagnuolo", 1)
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Clear LRU cache
        self.generator.get_dc_continuity.cache_clear()

        continuity, current_dc, prior_dc = self.generator.get_dc_continuity("KC", 2024)

        self.assertEqual(continuity, 1, "Same DC should return continuity=1")
        self.assertEqual(current_dc, "Steve Spagnuolo")
        self.assertEqual(prior_dc, "Steve Spagnuolo")

    @patch("py.features.dc_features.psycopg2.connect")
    def test_continuity_new_dc(self, mock_connect):
        """Different DC from last season should return continuity=0."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("Bobby Babich", "Sean McDermott", 0)
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Clear LRU cache
        self.generator.get_dc_continuity.cache_clear()

        continuity, current_dc, prior_dc = self.generator.get_dc_continuity("BUF", 2024)

        self.assertEqual(continuity, 0, "New DC should return continuity=0")
        self.assertEqual(current_dc, "Bobby Babich")
        self.assertEqual(prior_dc, "Sean McDermott")

    @patch("py.features.dc_features.psycopg2.connect")
    def test_continuity_no_prior_dc_data(self, mock_connect):
        """Missing prior DC data should return continuity=0."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("New DC", None, 0)
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Clear LRU cache
        self.generator.get_dc_continuity.cache_clear()

        continuity, current_dc, prior_dc = self.generator.get_dc_continuity("NEW", 2024)

        self.assertEqual(continuity, 0, "Missing prior DC should return continuity=0")
        self.assertEqual(prior_dc, "Unknown")


class TestDCTenureCalculation(unittest.TestCase):
    """Test DC tenure calculation across seasons."""

    def setUp(self):
        """Initialize generator with mocked database."""
        self.generator = DCFeatureGenerator()

    @patch("py.features.dc_features.psycopg2.connect")
    def test_tenure_long_serving_dc(self, mock_connect):
        """Long-serving DC should return high tenure count."""
        mock_cursor = MagicMock()
        # KC's Steve Spagnuolo: 6 seasons (2019-2024)
        mock_cursor.fetchone.return_value = (6,)
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Clear LRU cache
        self.generator.get_dc_tenure.cache_clear()

        tenure = self.generator.get_dc_tenure("KC", 2024)

        self.assertEqual(tenure, 6, "Long-serving DC should have 6-year tenure")

    @patch("py.features.dc_features.psycopg2.connect")
    def test_tenure_new_dc(self, mock_connect):
        """New DC should return tenure=1."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Clear LRU cache
        self.generator.get_dc_tenure.cache_clear()

        tenure = self.generator.get_dc_tenure("BUF", 2024)

        self.assertEqual(tenure, 1, "New DC should have tenure=1")

    @patch("py.features.dc_features.psycopg2.connect")
    def test_tenure_no_data_returns_1(self, mock_connect):
        """Missing tenure data should default to 1."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (None,)
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Clear LRU cache
        self.generator.get_dc_tenure.cache_clear()

        tenure = self.generator.get_dc_tenure("UNKNOWN", 2024)

        self.assertEqual(tenure, 1, "Missing tenure data should default to 1")


class TestFeatureExtraction(unittest.TestCase):
    """Test complete feature extraction."""

    def setUp(self):
        """Initialize generator with mocked methods."""
        self.generator = DCFeatureGenerator()

    def test_get_features_returns_all_4_features(self):
        """get_features should return all 4 expected features."""
        with patch.object(
            self.generator, "get_dc_continuity", return_value=(1, "DC Name", "DC Name")
        ):
            with patch.object(self.generator, "get_dc_tenure", return_value=3):
                features = self.generator.get_features("KC", 2024, 5)

        expected_keys = ["dc_continuity", "games_with_dc", "dc_tenure_seasons", "new_dc_adjustment"]

        self.assertEqual(len(features), 4, "Should return exactly 4 features")
        for key in expected_keys:
            self.assertIn(key, features, f"Feature '{key}' should be present")

    def test_get_features_with_opponent(self):
        """get_features with opponent should return 5 features."""
        with patch.object(
            self.generator, "get_dc_continuity", return_value=(1, "DC Name", "DC Name")
        ):
            with patch.object(self.generator, "get_dc_tenure", return_value=3):
                features = self.generator.get_features("KC", 2024, 5, opponent="BUF")

        self.assertIn(
            "opponent_dc_continuity",
            features,
            "Opponent DC continuity should be included when opponent provided",
        )
        self.assertEqual(len(features), 5, "Should return 5 features with opponent")

    def test_games_with_dc_calculation(self):
        """games_with_dc should be week - 1."""
        with patch.object(self.generator, "get_dc_continuity", return_value=(1, "DC", "DC")):
            with patch.object(self.generator, "get_dc_tenure", return_value=2):
                features_week1 = self.generator.get_features("TEST", 2024, 1)
                features_week5 = self.generator.get_features("TEST", 2024, 5)
                features_week10 = self.generator.get_features("TEST", 2024, 10)

        self.assertEqual(features_week1["games_with_dc"], 0, "Week 1 should have 0 games with DC")
        self.assertEqual(features_week5["games_with_dc"], 4, "Week 5 should have 4 games with DC")
        self.assertEqual(features_week10["games_with_dc"], 9, "Week 10 should have 9 games with DC")


class TestOpponentAdjustedFeatures(unittest.TestCase):
    """Test opponent-adjusted feature extraction."""

    def setUp(self):
        """Initialize generator."""
        self.generator = DCFeatureGenerator()

    def test_get_opponent_adjusted_features_returns_all_features(self):
        """get_opponent_adjusted_features should return team + opponent features."""
        with patch.object(self.generator, "get_dc_continuity") as mock_cont:
            with patch.object(self.generator, "get_dc_tenure") as mock_tenure:
                # KC: continuity=1, tenure=6 (established DC)
                # BUF: continuity=0, tenure=1 (new DC)
                def continuity_side_effect(team, season):
                    if team == "KC":
                        return (1, "Steve Spagnuolo", "Steve Spagnuolo")
                    else:
                        return (0, "Bobby Babich", "Sean McDermott")

                def tenure_side_effect(team, season):
                    return 6 if team == "KC" else 1

                mock_cont.side_effect = continuity_side_effect
                mock_tenure.side_effect = tenure_side_effect

                features = self.generator.get_opponent_adjusted_features("KC", "BUF", 2024, 5)

        # Should have team features + opponent features
        self.assertIn("dc_continuity", features)
        self.assertIn("opp_dc_continuity", features)
        self.assertIn("opp_dc_tenure_seasons", features)
        self.assertIn("opp_new_dc_adjustment", features)

        # Verify opponent values
        self.assertEqual(features["opp_dc_continuity"], 0, "BUF should have new DC")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Initialize generator."""
        self.generator = DCFeatureGenerator()

    @patch("py.features.dc_features.psycopg2.connect")
    def test_no_dc_data_returns_defaults(self, mock_connect):
        """Missing DC data should return continuity=0, Unknown DC names."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None  # No data found
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Clear LRU cache
        self.generator.get_dc_continuity.cache_clear()

        continuity, current_dc, prior_dc = self.generator.get_dc_continuity("UNKNOWN", 2024)

        self.assertEqual(continuity, 0, "Missing data should return continuity=0")
        self.assertEqual(current_dc, "Unknown")
        self.assertEqual(prior_dc, "Unknown")

    def test_week_1_games_with_dc_is_zero(self):
        """Week 1 should always have games_with_dc=0."""
        with patch.object(self.generator, "get_dc_continuity", return_value=(1, "DC", "DC")):
            with patch.object(self.generator, "get_dc_tenure", return_value=1):
                features = self.generator.get_features("TEST", 2024, 1)

        self.assertEqual(features["games_with_dc"], 0, "Week 1 should have 0 games with DC")

    def test_custom_halflife(self):
        """Custom halflife should affect decay calculation."""
        generator_h2 = DCFeatureGenerator(halflife_weeks=2.0)
        generator_h8 = DCFeatureGenerator(halflife_weeks=8.0)

        self.assertEqual(generator_h2.halflife_weeks, 2.0)
        self.assertEqual(generator_h8.halflife_weeks, 8.0)

    def test_database_config_default(self):
        """Default database config should be set correctly."""
        generator = DCFeatureGenerator()

        self.assertEqual(generator.db_config["host"], "localhost")
        self.assertEqual(generator.db_config["port"], 5544)
        self.assertEqual(generator.db_config["database"], "devdb01")

    def test_database_config_custom(self):
        """Custom database config should override defaults."""
        custom_config = {
            "host": "remote-db.example.com",
            "port": 5432,
            "database": "testdb",
            "user": "testuser",
            "password": "testpass",
        }
        generator = DCFeatureGenerator(db_config=custom_config)

        self.assertEqual(generator.db_config["host"], "remote-db.example.com")
        self.assertEqual(generator.db_config["port"], 5432)


class TestCacheBehavior(unittest.TestCase):
    """Test LRU cache behavior for performance."""

    def setUp(self):
        """Initialize generator."""
        self.generator = DCFeatureGenerator()

    @patch("py.features.dc_features.psycopg2.connect")
    def test_continuity_cache_hit(self, mock_connect):
        """Second call with same args should hit cache (no DB query)."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("DC", "DC", 1)
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Clear cache first
        self.generator.get_dc_continuity.cache_clear()

        # First call - should hit DB
        result1 = self.generator.get_dc_continuity("KC", 2024)

        # Second call - should hit cache (no DB query)
        result2 = self.generator.get_dc_continuity("KC", 2024)

        # Both results should be identical
        self.assertEqual(result1, result2)

        # Database should only be called once (due to cache)
        # Note: cursor.execute is called once per query
        self.assertEqual(mock_cursor.execute.call_count, 1, "Second call should hit cache, not DB")

    @patch("py.features.dc_features.psycopg2.connect")
    def test_tenure_cache_hit(self, mock_connect):
        """Second tenure call with same args should hit cache."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (5,)
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Clear cache first
        self.generator.get_dc_tenure.cache_clear()

        # First call
        result1 = self.generator.get_dc_tenure("KC", 2024)

        # Second call
        result2 = self.generator.get_dc_tenure("KC", 2024)

        self.assertEqual(result1, result2)
        self.assertEqual(
            mock_cursor.execute.call_count, 1, "Tenure cache should prevent second DB query"
        )


class TestIntegrationPatterns(unittest.TestCase):
    """Test integration patterns and constants."""

    def setUp(self):
        """Initialize generator."""
        self.generator = DCFeatureGenerator()

    def test_halflife_default_value(self):
        """Default halflife should be 4.0 weeks."""
        self.assertEqual(self.generator.halflife_weeks, 4.0, "Default halflife should be 4.0 weeks")

    def test_cache_size_default(self):
        """Default cache size should be 1024."""
        self.assertEqual(self.generator._cache_size, 1024, "Default cache size should be 1024")

    def test_connection_lazy_initialization(self):
        """Connection should be None until first query."""
        generator = DCFeatureGenerator()
        self.assertIsNone(generator._conn, "Connection should be None until first query")

    def test_close_method_exists(self):
        """close() method should exist for cleanup."""
        self.assertTrue(hasattr(self.generator, "close"), "close() method should exist")


if __name__ == "__main__":
    unittest.main()
