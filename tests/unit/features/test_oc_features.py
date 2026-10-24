#!/usr/bin/env python3
"""
Unit Tests for OC (Offensive Coordinator) Features Module
==========================================================

Tests the OCFeatureGenerator class for correct feature computation,
exponential decay for new OC adjustment, continuity logic, and cache behavior.

Test Coverage:
1. Exponential decay computation (H=4 halflife)
2. OC continuity logic (returning vs new coordinator)
3. Feature extraction with valid data
4. Games with OC calculation
5. Tenure calculation across seasons
6. Edge cases (new team, missing OC data)
7. LRU cache behavior
8. Bulk feature extraction
9. Metadata inclusion

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

from nfl_analytics.features.oc_features import OCFeatureGenerator


class TestExponentialDecayAdjustment(unittest.TestCase):
    """Test exponential decay computation for new OC adjustment."""

    def setUp(self):
        """Initialize generator."""
        self.generator = OCFeatureGenerator()

    def test_new_oc_week1_adjustment(self):
        """New OC at week 1 should have 0.8 adjustment."""
        with patch.object(
            self.generator, "get_oc_continuity", return_value=(0, "New OC", "Old OC")
        ):
            adjustment = self.generator.get_new_oc_adjustment("TEST", 2024, 1)

        self.assertAlmostEqual(
            adjustment, 0.8, places=4, msg="New OC at week 1 should have 0.8 adjustment"
        )

    def test_new_oc_week5_adjustment(self):
        """New OC at week 5 should have decayed adjustment (~0.294)."""
        # games_with_oc = 4 at week 5
        # adjustment = 0.8 * exp(-4 / 4) = 0.8 * exp(-1) ≈ 0.294
        expected = 0.8 * math.exp(-4 / 4.0)

        with patch.object(
            self.generator, "get_oc_continuity", return_value=(0, "New OC", "Old OC")
        ):
            adjustment = self.generator.get_new_oc_adjustment("TEST", 2024, 5)

        self.assertAlmostEqual(
            adjustment,
            expected,
            places=4,
            msg=f"New OC at week 5 should have ~{expected:.3f} adjustment",
        )

    def test_returning_oc_no_adjustment(self):
        """Returning OC should have 0.0 adjustment."""
        with patch.object(
            self.generator, "get_oc_continuity", return_value=(1, "Same OC", "Same OC")
        ):
            adjustment = self.generator.get_new_oc_adjustment("KC", 2024, 1)

        self.assertAlmostEqual(
            adjustment, 0.0, places=4, msg="Returning OC should have 0.0 adjustment"
        )

    def test_custom_halflife(self):
        """Custom halflife should affect decay calculation."""
        with patch.object(self.generator, "get_oc_continuity", return_value=(0, "New", "Old")):
            # H=2: 0.8 * exp(-4/2) = 0.8 * exp(-2) ≈ 0.108
            adj_h2 = self.generator.get_new_oc_adjustment("TEST", 2024, 5, halflife_weeks=2.0)
            # H=8: 0.8 * exp(-4/8) = 0.8 * exp(-0.5) ≈ 0.485
            adj_h8 = self.generator.get_new_oc_adjustment("TEST", 2024, 5, halflife_weeks=8.0)

        expected_h2 = 0.8 * math.exp(-4 / 2.0)
        expected_h8 = 0.8 * math.exp(-4 / 8.0)

        self.assertAlmostEqual(adj_h2, expected_h2, places=3)
        self.assertAlmostEqual(adj_h8, expected_h8, places=3)
        self.assertLess(adj_h2, adj_h8, "Shorter halflife should decay faster")


class TestOCContinuityLogic(unittest.TestCase):
    """Test OC continuity determination."""

    def setUp(self):
        """Initialize generator with mocked database."""
        self.generator = OCFeatureGenerator()

    @patch("nfl_analytics.features.oc_features.psycopg2.connect")
    def test_continuity_same_oc(self, mock_connect):
        """Same OC as last season should return continuity=1."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("Matt Nagy", "Matt Nagy", 1)
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Clear LRU cache
        self.generator.get_oc_continuity.cache_clear()

        continuity, current_oc, prior_oc = self.generator.get_oc_continuity("KC", 2024)

        self.assertEqual(continuity, 1, "Same OC should return continuity=1")
        self.assertEqual(current_oc, "Matt Nagy")
        self.assertEqual(prior_oc, "Matt Nagy")

    @patch("nfl_analytics.features.oc_features.psycopg2.connect")
    def test_continuity_new_oc(self, mock_connect):
        """Different OC from last season should return continuity=0."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("New Coordinator", "Old Coordinator", 0)
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Clear LRU cache
        self.generator.get_oc_continuity.cache_clear()

        continuity, current_oc, prior_oc = self.generator.get_oc_continuity("TEST", 2024)

        self.assertEqual(continuity, 0, "New OC should return continuity=0")
        self.assertEqual(current_oc, "New Coordinator")
        self.assertEqual(prior_oc, "Old Coordinator")

    @patch("nfl_analytics.features.oc_features.psycopg2.connect")
    def test_continuity_no_data_returns_defaults(self, mock_connect):
        """Missing OC data should return continuity=0 and None names."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None  # No data found
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Clear LRU cache
        self.generator.get_oc_continuity.cache_clear()

        continuity, current_oc, prior_oc = self.generator.get_oc_continuity("UNKNOWN", 2024)

        self.assertEqual(continuity, 0, "Missing data should return continuity=0")
        self.assertIsNone(current_oc)
        self.assertIsNone(prior_oc)


class TestGamesWithOCCalculation(unittest.TestCase):
    """Test games_with_oc calculation."""

    def setUp(self):
        """Initialize generator."""
        self.generator = OCFeatureGenerator()

    def test_week1_games_with_oc(self):
        """Week 1 should have 0 games with OC."""
        games = self.generator.get_games_with_oc("KC", 2024, 1)
        self.assertEqual(games, 0, "Week 1 should have 0 games completed")

    def test_week5_games_with_oc(self):
        """Week 5 should have 4 games with OC."""
        games = self.generator.get_games_with_oc("KC", 2024, 5)
        self.assertEqual(games, 4, "Week 5 should have 4 games completed")

    def test_week18_games_with_oc(self):
        """Week 18 should have 17 games with OC (capped)."""
        games = self.generator.get_games_with_oc("KC", 2024, 18)
        self.assertEqual(games, 17, "Week 18 should be capped at 17 games")

    def test_week20_games_with_oc_capped(self):
        """Week 20 (playoffs) should still be capped at 17."""
        games = self.generator.get_games_with_oc("KC", 2024, 20)
        self.assertEqual(games, 17, "Playoffs should be capped at 17 regular season games")

    def test_negative_week_returns_zero(self):
        """Negative week should return 0."""
        games = self.generator.get_games_with_oc("KC", 2024, -1)
        self.assertEqual(games, 0, "Negative week should return 0")


class TestOCTenureCalculation(unittest.TestCase):
    """Test OC tenure calculation across seasons."""

    def setUp(self):
        """Initialize generator with mocked database."""
        self.generator = OCFeatureGenerator()

    @patch("nfl_analytics.features.oc_features.psycopg2.connect")
    def test_tenure_long_serving_oc(self, mock_connect):
        """Long-serving OC should return high tenure count."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (5,)  # 5 seasons
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Clear LRU cache
        self.generator.get_oc_tenure.cache_clear()

        tenure = self.generator.get_oc_tenure("KC", 2024)

        self.assertEqual(tenure, 5, "Long-serving OC should have 5-year tenure")

    @patch("nfl_analytics.features.oc_features.psycopg2.connect")
    def test_tenure_new_oc(self, mock_connect):
        """New OC should return tenure=1."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Clear LRU cache
        self.generator.get_oc_tenure.cache_clear()

        tenure = self.generator.get_oc_tenure("TEST", 2024)

        self.assertEqual(tenure, 1, "New OC should have tenure=1")

    @patch("nfl_analytics.features.oc_features.psycopg2.connect")
    def test_tenure_no_data_returns_1(self, mock_connect):
        """Missing tenure data should default to 1."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Clear LRU cache
        self.generator.get_oc_tenure.cache_clear()

        tenure = self.generator.get_oc_tenure("UNKNOWN", 2024)

        self.assertEqual(tenure, 1, "Missing tenure data should default to 1")


class TestFeatureExtraction(unittest.TestCase):
    """Test complete feature extraction."""

    def setUp(self):
        """Initialize generator with mocked methods."""
        self.generator = OCFeatureGenerator()

    def test_get_features_returns_all_4_features(self):
        """get_features should return all 4 expected features."""
        with patch.object(
            self.generator, "get_oc_continuity", return_value=(1, "OC Name", "OC Name")
        ):
            with patch.object(self.generator, "get_oc_tenure", return_value=3):
                features = self.generator.get_features("KC", 2024, 5)

        expected_keys = ["oc_continuity", "games_with_oc", "oc_tenure_seasons", "new_oc_adjustment"]

        self.assertEqual(len(features), 4, "Should return exactly 4 features without metadata")
        for key in expected_keys:
            self.assertIn(key, features, f"Feature '{key}' should be present")

    def test_get_features_with_metadata(self):
        """get_features with include_metadata=True should include OC names."""
        with patch.object(
            self.generator, "get_oc_continuity", return_value=(1, "Current OC", "Prior OC")
        ):
            with patch.object(self.generator, "get_oc_tenure", return_value=2):
                features = self.generator.get_features("KC", 2024, 5, include_metadata=True)

        self.assertEqual(len(features), 6, "Should return 6 features with metadata")
        self.assertIn("current_oc", features)
        self.assertIn("prior_oc", features)
        self.assertEqual(features["current_oc"], "Current OC")
        self.assertEqual(features["prior_oc"], "Prior OC")

    def test_games_with_oc_in_features(self):
        """Features should correctly calculate games_with_oc."""
        with patch.object(self.generator, "get_oc_continuity", return_value=(1, "OC", "OC")):
            with patch.object(self.generator, "get_oc_tenure", return_value=2):
                features_week1 = self.generator.get_features("TEST", 2024, 1)
                features_week5 = self.generator.get_features("TEST", 2024, 5)
                features_week10 = self.generator.get_features("TEST", 2024, 10)

        self.assertEqual(features_week1["games_with_oc"], 0)
        self.assertEqual(features_week5["games_with_oc"], 4)
        self.assertEqual(features_week10["games_with_oc"], 9)

    def test_new_oc_adjustment_rounded(self):
        """new_oc_adjustment should be rounded to 4 decimal places."""
        with patch.object(self.generator, "get_oc_continuity", return_value=(0, "New", "Old")):
            with patch.object(self.generator, "get_oc_tenure", return_value=1):
                features = self.generator.get_features("TEST", 2024, 5)

        # Check that it's properly rounded
        adjustment = features["new_oc_adjustment"]
        self.assertEqual(adjustment, round(adjustment, 4))


class TestBulkFeatures(unittest.TestCase):
    """Test bulk feature extraction."""

    def setUp(self):
        """Initialize generator."""
        self.generator = OCFeatureGenerator()

    def test_bulk_features_returns_all_keys(self):
        """get_bulk_features should return features for all requested team-game combos."""
        test_cases = [
            ("KC", 2024, 1),
            ("KC", 2024, 5),
            ("BUF", 2024, 3),
        ]

        with patch.object(self.generator, "get_oc_continuity", return_value=(1, "OC", "OC")):
            with patch.object(self.generator, "get_oc_tenure", return_value=2):
                bulk_results = self.generator.get_bulk_features(test_cases)

        # Verify all keys present
        for key in test_cases:
            self.assertIn(key, bulk_results, f"Key {key} should be in results")

        # Verify each result has 4 features
        for key, features in bulk_results.items():
            self.assertEqual(len(features), 4, "Each result should have 4 features")

    def test_bulk_features_with_metadata(self):
        """Bulk features with metadata should include OC names."""
        test_cases = [("KC", 2024, 1)]

        with patch.object(
            self.generator, "get_oc_continuity", return_value=(1, "OC Name", "OC Name")
        ):
            with patch.object(self.generator, "get_oc_tenure", return_value=2):
                bulk_results = self.generator.get_bulk_features(test_cases, include_metadata=True)

        features = bulk_results[("KC", 2024, 1)]
        self.assertEqual(len(features), 6, "Should have 6 features with metadata")
        self.assertIn("current_oc", features)
        self.assertIn("prior_oc", features)


class TestCacheBehavior(unittest.TestCase):
    """Test LRU cache behavior for performance."""

    def setUp(self):
        """Initialize generator."""
        self.generator = OCFeatureGenerator()

    @patch("nfl_analytics.features.oc_features.psycopg2.connect")
    def test_continuity_cache_hit(self, mock_connect):
        """Second call with same args should hit cache (no DB query)."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("OC", "OC", 1)
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Clear cache first
        self.generator.get_oc_continuity.cache_clear()

        # First call - should hit DB
        result1 = self.generator.get_oc_continuity("KC", 2024)

        # Second call - should hit cache (no DB query)
        result2 = self.generator.get_oc_continuity("KC", 2024)

        # Both results should be identical
        self.assertEqual(result1, result2)

        # Database cursor should only execute once (due to cache)
        self.assertEqual(mock_cursor.execute.call_count, 1, "Second call should hit cache, not DB")

    @patch("nfl_analytics.features.oc_features.psycopg2.connect")
    def test_tenure_cache_hit(self, mock_connect):
        """Second tenure call with same args should hit cache."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (4,)
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Clear cache first
        self.generator.get_oc_tenure.cache_clear()

        result1 = self.generator.get_oc_tenure("KC", 2024)
        result2 = self.generator.get_oc_tenure("KC", 2024)

        self.assertEqual(result1, result2)
        self.assertEqual(mock_cursor.execute.call_count, 1)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Initialize generator."""
        self.generator = OCFeatureGenerator()

    def test_database_config_default(self):
        """Default database config should be set correctly."""
        generator = OCFeatureGenerator()

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
        generator = OCFeatureGenerator(db_config=custom_config)

        self.assertEqual(generator.db_config["host"], "remote-db.example.com")
        self.assertEqual(generator.db_config["port"], 5432)

    def test_connection_lazy_initialization(self):
        """Connection should be None until first query."""
        generator = OCFeatureGenerator()
        self.assertIsNone(generator._conn, "Connection should be None until first query")

    def test_close_method_exists(self):
        """close() method should exist for cleanup."""
        self.assertTrue(hasattr(self.generator, "close"), "close() method should exist")

    def test_cache_size_default(self):
        """Default cache size should be 1024."""
        self.assertEqual(self.generator._cache_size, 1024, "Default cache size should be 1024")


class TestIntegrationPatterns(unittest.TestCase):
    """Test integration patterns and expected values."""

    def setUp(self):
        """Initialize generator."""
        self.generator = OCFeatureGenerator()

    def test_returning_oc_feature_values(self):
        """Returning OC should produce expected feature values."""
        with patch.object(self.generator, "get_oc_continuity", return_value=(1, "OC", "OC")):
            with patch.object(self.generator, "get_oc_tenure", return_value=5):
                features = self.generator.get_features("KC", 2024, 10)

        self.assertEqual(features["oc_continuity"], 1)
        self.assertEqual(features["games_with_oc"], 9)
        self.assertEqual(features["oc_tenure_seasons"], 5)
        self.assertEqual(features["new_oc_adjustment"], 0.0)

    def test_new_oc_feature_values(self):
        """New OC should produce expected feature values."""
        with patch.object(
            self.generator, "get_oc_continuity", return_value=(0, "New OC", "Old OC")
        ):
            with patch.object(self.generator, "get_oc_tenure", return_value=1):
                features = self.generator.get_features("TEST", 2024, 5)

        self.assertEqual(features["oc_continuity"], 0)
        self.assertEqual(features["games_with_oc"], 4)
        self.assertEqual(features["oc_tenure_seasons"], 1)
        # new_oc_adjustment should be ~0.294 (0.8 * exp(-1))
        expected_adj = round(0.8 * math.exp(-1), 4)
        self.assertEqual(features["new_oc_adjustment"], expected_adj)


if __name__ == "__main__":
    unittest.main()
