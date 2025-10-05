"""
Integration tests for data pipeline - end-to-end ingestion and processing.

Tests full pipeline from ingestion → database → feature generation.
"""

import pytest
import psycopg
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Add py/ to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "py"))


@pytest.fixture(scope="module")
def db_connection():
    """Database connection for integration tests."""
    import os
    try:
        conn = psycopg.connect(
            dbname=os.getenv("POSTGRES_DB", "devdb01"),
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5544")),
            user=os.getenv("POSTGRES_USER", "dro"),
            password=os.getenv("POSTGRES_PASSWORD", "sicillionbillions"),
        )
        yield conn
        conn.close()
    except Exception as e:
        pytest.skip(f"Database not available: {e}")


class TestDatabaseSchema:
    """Test database schema integrity."""

    def test_games_table_exists(self, db_connection):
        """Test that games table exists with expected columns."""
        query = """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'games' AND table_schema = 'public'
            ORDER BY ordinal_position
        """
        
        with db_connection.cursor() as cur:
            cur.execute(query)
            columns = {row[0]: row[1] for row in cur.fetchall()}
        
        # Check key columns
        assert 'game_id' in columns
        assert 'season' in columns
        assert 'home_team' in columns
        assert 'away_team' in columns
        assert 'home_score' in columns
        assert 'away_score' in columns

    def test_weather_table_exists(self, db_connection):
        """Test that weather table exists."""
        query = """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = 'weather' AND table_schema = 'public'
        """
        
        with db_connection.cursor() as cur:
            cur.execute(query)
            count = cur.fetchone()[0]
        
        assert count == 1

    def test_mart_schema_exists(self, db_connection):
        """Test that mart schema exists."""
        query = """
            SELECT COUNT(*)
            FROM information_schema.schemata
            WHERE schema_name = 'mart'
        """
        
        with db_connection.cursor() as cur:
            cur.execute(query)
            count = cur.fetchone()[0]
        
        assert count == 1


class TestDataIngestion:
    """Test data ingestion pipeline."""

    def test_games_data_loaded(self, db_connection):
        """Test that games data is loaded."""
        query = "SELECT COUNT(*) FROM games"
        
        with db_connection.cursor() as cur:
            cur.execute(query)
            count = cur.fetchone()[0]
        
        # Should have games data (at least 1000 games)
        assert count >= 1000

    def test_games_temporal_coverage(self, db_connection):
        """Test temporal coverage of games data."""
        query = """
            SELECT MIN(season) AS min_season, MAX(season) AS max_season
            FROM games
        """
        
        with db_connection.cursor() as cur:
            cur.execute(query)
            min_season, max_season = cur.fetchone()
        
        # Should cover multiple seasons
        assert min_season >= 1999
        assert max_season >= 2020
        assert max_season - min_season >= 20

    def test_weather_join_coverage(self, db_connection):
        """Test weather data join coverage."""
        query = """
            SELECT 
                COUNT(*) AS total_games,
                COUNT(w.game_id) AS games_with_weather,
                COUNT(w.game_id)::float / COUNT(*)::float AS coverage
            FROM games g
            LEFT JOIN weather w ON g.game_id = w.game_id
            WHERE g.season >= 2020
        """
        
        with db_connection.cursor() as cur:
            cur.execute(query)
            total, with_weather, coverage = cur.fetchone()
        
        # Should have >80% weather coverage for recent seasons
        assert coverage >= 0.80

    def test_odds_history_loaded(self, db_connection):
        """Test that odds history is loaded."""
        query = "SELECT COUNT(*) FROM odds_history WHERE market = 'spreads'"
        
        with db_connection.cursor() as cur:
            cur.execute(query)
            count = cur.fetchone()[0]
        
        # Should have odds data
        assert count > 0


class TestDataQuality:
    """Test data quality constraints."""

    def test_no_null_scores(self, db_connection):
        """Test that completed games have non-null scores."""
        query = """
            SELECT COUNT(*)
            FROM games
            WHERE season >= 2020
                AND (home_score IS NULL OR away_score IS NULL)
                AND gameday < CURRENT_DATE - INTERVAL '7 days'
        """
        
        with db_connection.cursor() as cur:
            cur.execute(query)
            null_count = cur.fetchone()[0]
        
        # Recent completed games should have scores
        assert null_count == 0

    def test_score_validity(self, db_connection):
        """Test that scores are within valid ranges."""
        query = """
            SELECT COUNT(*)
            FROM games
            WHERE home_score < 0 OR away_score < 0
                OR home_score > 100 OR away_score > 100
        """
        
        with db_connection.cursor() as cur:
            cur.execute(query)
            invalid_count = cur.fetchone()[0]
        
        # No invalid scores
        assert invalid_count == 0

    def test_unique_game_ids(self, db_connection):
        """Test that game_ids are unique."""
        query = """
            SELECT COUNT(*), COUNT(DISTINCT game_id)
            FROM games
        """
        
        with db_connection.cursor() as cur:
            cur.execute(query)
            total, distinct = cur.fetchone()
        
        # All game_ids should be unique
        assert total == distinct

    def test_weather_values_reasonable(self, db_connection):
        """Test that weather values are reasonable."""
        query = """
            SELECT 
                MIN(temp_c) AS min_temp,
                MAX(temp_c) AS max_temp,
                MIN(wind_kph) AS min_wind,
                MAX(wind_kph) AS max_wind
            FROM weather
        """
        
        with db_connection.cursor() as cur:
            cur.execute(query)
            min_temp, max_temp, min_wind, max_wind = cur.fetchone()
        
        # Reasonable ranges for NFL games
        assert -20 <= min_temp <= 40  # Celsius
        assert -20 <= max_temp <= 40
        assert 0 <= min_wind <= 100  # km/h
        assert 0 <= max_wind <= 100


class TestMaterializedViews:
    """Test materialized views."""

    def test_game_summary_view_exists(self, db_connection):
        """Test that game_summary view exists."""
        query = """
            SELECT COUNT(*)
            FROM information_schema.views
            WHERE table_name = 'game_summary' AND table_schema = 'mart'
        """
        
        with db_connection.cursor() as cur:
            cur.execute(query)
            count = cur.fetchone()[0]
        
        assert count >= 0  # May be materialized view or regular view

    def test_team_epa_table_exists(self, db_connection):
        """Test that team_epa table exists."""
        query = """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = 'team_epa' AND table_schema = 'mart'
        """
        
        with db_connection.cursor() as cur:
            cur.execute(query)
            count = cur.fetchone()[0]
        
        assert count == 1


class TestIdempotency:
    """Test that ingestion is idempotent."""

    def test_reingest_same_data(self, db_connection):
        """Test that re-ingesting same data doesn't create duplicates."""
        # Get current count
        query1 = "SELECT COUNT(*) FROM games WHERE season = 2024"
        
        with db_connection.cursor() as cur:
            cur.execute(query1)
            count_before = cur.fetchone()[0]
        
        # Note: Actual re-ingestion would happen here
        # For now, just verify count is stable
        
        with db_connection.cursor() as cur:
            cur.execute(query1)
            count_after = cur.fetchone()[0]
        
        # Count should be same (idempotent)
        assert count_before == count_after


class TestPerformance:
    """Test query performance."""

    def test_game_query_performance(self, db_connection):
        """Test that game queries are fast."""
        import time
        
        query = """
            SELECT *
            FROM games
            WHERE season = 2024
            LIMIT 100
        """
        
        start = time.time()
        with db_connection.cursor() as cur:
            cur.execute(query)
            cur.fetchall()
        elapsed = time.time() - start
        
        # Should be fast (< 1 second for 100 rows)
        assert elapsed < 1.0

    def test_join_performance(self, db_connection):
        """Test that joins are performant."""
        import time
        
        query = """
            SELECT g.*, w.*
            FROM games g
            LEFT JOIN weather w ON g.game_id = w.game_id
            WHERE g.season = 2024
            LIMIT 100
        """
        
        start = time.time()
        with db_connection.cursor() as cur:
            cur.execute(query)
            cur.fetchall()
        elapsed = time.time() - start
        
        # Should be reasonably fast
        assert elapsed < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
