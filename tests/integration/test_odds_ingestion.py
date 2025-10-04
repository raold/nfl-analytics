"""
Integration tests for odds ingestion pipeline.

Tests the complete flow: mock API response → parsing → database insertion.
"""

import sys
import importlib.util
from pathlib import Path
import datetime as dt
from typing import Generator

import pytest
import psycopg
import responses

# Load the module directly from file path to avoid conflicts with pytest's 'py' package
project_root = Path(__file__).parent.parent.parent
ingest_odds_path = project_root / "py" / "ingest_odds_history.py"

spec = importlib.util.spec_from_file_location("ingest_odds_history", ingest_odds_path)
ingest_odds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingest_odds)

flatten_events = ingest_odds.flatten_events
build_request = ingest_odds.build_request

# Import test fixtures
sys.path.insert(0, str(project_root))
from tests.fixtures.sample_odds import SAMPLE_ODDS_RESPONSE_SINGLE_GAME


@pytest.fixture
def test_db_with_schema(test_db_dsn: str) -> Generator[psycopg.Connection, None, None]:
    """
    Create a test database with full schema applied.
    
    This fixture applies all migrations and yields a connection.
    Changes are rolled back after the test.
    """
    conn = psycopg.connect(test_db_dsn)
    conn.autocommit = False
    cursor = conn.cursor()
    
    # Apply schema (simplified for testing)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS odds_history (
            event_id TEXT NOT NULL,
            sport_key TEXT,
            commence_time TIMESTAMPTZ,
            home_team TEXT,
            away_team TEXT,
            bookmaker_key TEXT NOT NULL,
            bookmaker_title TEXT,
            market_key TEXT NOT NULL,
            market_last_update TIMESTAMPTZ,
            outcome_name TEXT NOT NULL,
            outcome_price NUMERIC,
            outcome_point NUMERIC,
            snapshot_at TIMESTAMPTZ NOT NULL,
            book_last_update TIMESTAMPTZ,
            PRIMARY KEY (event_id, bookmaker_key, market_key, outcome_name, snapshot_at)
        );
    """)
    conn.commit()
    
    try:
        yield conn
    finally:
        conn.rollback()
        cursor.execute("DROP TABLE IF EXISTS odds_history CASCADE;")
        conn.commit()
        conn.close()


@pytest.mark.integration
class TestOddsIngestionFlow:
    """Integration tests for the full odds ingestion pipeline."""

    def test_flatten_and_insert_single_game(
        self,
        test_db_with_schema: psycopg.Connection,
        sample_snapshot_time: dt.datetime
    ):
        """Test flattening API response and inserting into database."""
        # Flatten the sample response
        rows = flatten_events([SAMPLE_ODDS_RESPONSE_SINGLE_GAME], sample_snapshot_time)
        
        # Should have 7 total rows: 2 spreads + 2 totals + 2 h2h + 1 more spread from DraftKings
        assert len(rows) == 7
        
        # Insert into database
        cursor = test_db_with_schema.cursor()
        for row in rows:
            cursor.execute("""
                INSERT INTO odds_history (
                    event_id, sport_key, commence_time, home_team, away_team,
                    bookmaker_key, bookmaker_title, market_key, market_last_update,
                    outcome_name, outcome_price, outcome_point, snapshot_at, book_last_update
                ) VALUES (
                    %(event_id)s, %(sport_key)s, %(commence_time)s, %(home_team)s, %(away_team)s,
                    %(bookmaker_key)s, %(bookmaker_title)s, %(market_key)s, %(market_last_update)s,
                    %(outcome_name)s, %(outcome_price)s, %(outcome_point)s, %(snapshot_at)s, %(book_last_update)s
                ) ON CONFLICT DO NOTHING;
            """, row)
        test_db_with_schema.commit()
        
        # Verify insertion
        cursor.execute("SELECT COUNT(*) FROM odds_history;")
        count = cursor.fetchone()[0]
        assert count == 7

    def test_idempotent_insertion(
        self,
        test_db_with_schema: psycopg.Connection,
        sample_snapshot_time: dt.datetime
    ):
        """Test that inserting the same data twice doesn't create duplicates."""
        rows = flatten_events([SAMPLE_ODDS_RESPONSE_SINGLE_GAME], sample_snapshot_time)
        
        cursor = test_db_with_schema.cursor()
        
        # Insert once
        for row in rows:
            cursor.execute("""
                INSERT INTO odds_history (
                    event_id, sport_key, commence_time, home_team, away_team,
                    bookmaker_key, bookmaker_title, market_key, market_last_update,
                    outcome_name, outcome_price, outcome_point, snapshot_at, book_last_update
                ) VALUES (
                    %(event_id)s, %(sport_key)s, %(commence_time)s, %(home_team)s, %(away_team)s,
                    %(bookmaker_key)s, %(bookmaker_title)s, %(market_key)s, %(market_last_update)s,
                    %(outcome_name)s, %(outcome_price)s, %(outcome_point)s, %(snapshot_at)s, %(book_last_update)s
                ) ON CONFLICT DO NOTHING;
            """, row)
        test_db_with_schema.commit()
        
        cursor.execute("SELECT COUNT(*) FROM odds_history;")
        count_after_first = cursor.fetchone()[0]
        
        # Insert again (should be no-op due to ON CONFLICT DO NOTHING)
        for row in rows:
            cursor.execute("""
                INSERT INTO odds_history (
                    event_id, sport_key, commence_time, home_team, away_team,
                    bookmaker_key, bookmaker_title, market_key, market_last_update,
                    outcome_name, outcome_price, outcome_point, snapshot_at, book_last_update
                ) VALUES (
                    %(event_id)s, %(sport_key)s, %(commence_time)s, %(home_team)s, %(away_team)s,
                    %(bookmaker_key)s, %(bookmaker_title)s, %(market_key)s, %(market_last_update)s,
                    %(outcome_name)s, %(outcome_price)s, %(outcome_point)s, %(snapshot_at)s, %(book_last_update)s
                ) ON CONFLICT DO NOTHING;
            """, row)
        test_db_with_schema.commit()
        
        cursor.execute("SELECT COUNT(*) FROM odds_history;")
        count_after_second = cursor.fetchone()[0]
        
        # Count should be the same
        assert count_after_first == count_after_second

    def test_query_by_bookmaker(
        self,
        test_db_with_schema: psycopg.Connection,
        sample_snapshot_time: dt.datetime
    ):
        """Test querying odds by bookmaker."""
        rows = flatten_events([SAMPLE_ODDS_RESPONSE_SINGLE_GAME], sample_snapshot_time)
        
        cursor = test_db_with_schema.cursor()
        for row in rows:
            cursor.execute("""
                INSERT INTO odds_history (
                    event_id, sport_key, commence_time, home_team, away_team,
                    bookmaker_key, bookmaker_title, market_key, market_last_update,
                    outcome_name, outcome_price, outcome_point, snapshot_at, book_last_update
                ) VALUES (
                    %(event_id)s, %(sport_key)s, %(commence_time)s, %(home_team)s, %(away_team)s,
                    %(bookmaker_key)s, %(bookmaker_title)s, %(market_key)s, %(market_last_update)s,
                    %(outcome_name)s, %(outcome_price)s, %(outcome_point)s, %(snapshot_at)s, %(book_last_update)s
                ) ON CONFLICT DO NOTHING;
            """, row)
        test_db_with_schema.commit()
        
        # Query FanDuel odds only
        cursor.execute("""
            SELECT COUNT(*) FROM odds_history WHERE bookmaker_key = 'fanduel';
        """)
        fanduel_count = cursor.fetchone()[0]
        assert fanduel_count == 6  # 2 spreads + 2 totals + 2 h2h
        
        # Query DraftKings odds only
        cursor.execute("""
            SELECT COUNT(*) FROM odds_history WHERE bookmaker_key = 'draftkings';
        """)
        draftkings_count = cursor.fetchone()[0]
        assert draftkings_count == 2  # 2 spreads only

    def test_query_by_market(
        self,
        test_db_with_schema: psycopg.Connection,
        sample_snapshot_time: dt.datetime
    ):
        """Test querying odds by market type."""
        rows = flatten_events([SAMPLE_ODDS_RESPONSE_SINGLE_GAME], sample_snapshot_time)
        
        cursor = test_db_with_schema.cursor()
        for row in rows:
            cursor.execute("""
                INSERT INTO odds_history (
                    event_id, sport_key, commence_time, home_team, away_team,
                    bookmaker_key, bookmaker_title, market_key, market_last_update,
                    outcome_name, outcome_price, outcome_point, snapshot_at, book_last_update
                ) VALUES (
                    %(event_id)s, %(sport_key)s, %(commence_time)s, %(home_team)s, %(away_team)s,
                    %(bookmaker_key)s, %(bookmaker_title)s, %(market_key)s, %(market_last_update)s,
                    %(outcome_name)s, %(outcome_price)s, %(outcome_point)s, %(snapshot_at)s, %(book_last_update)s
                ) ON CONFLICT DO NOTHING;
            """, row)
        test_db_with_schema.commit()
        
        # Count by market
        cursor.execute("""
            SELECT market_key, COUNT(*) 
            FROM odds_history 
            GROUP BY market_key 
            ORDER BY market_key;
        """)
        market_counts = cursor.fetchall()
        
        market_dict = {row[0]: row[1] for row in market_counts}
        assert market_dict["h2h"] == 2      # 2 moneylines (Bills + Cardinals)
        assert market_dict["spreads"] == 4  # 2 from FanDuel + 2 from DraftKings
        assert market_dict["totals"] == 2   # 2 from FanDuel (Over + Under)


@pytest.mark.integration
@pytest.mark.api
class TestOddsAPIIntegration:
    """Integration tests with mocked API responses."""

    @responses.activate
    def test_api_request_success(self, mock_env_vars):
        """Test successful API request with mocked response."""
        snapshot = dt.datetime(2023, 9, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
        
        # Mock the API response
        responses.add(
            responses.GET,
            "https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/odds",
            json={"data": [SAMPLE_ODDS_RESPONSE_SINGLE_GAME]},
            status=200,
            headers={"x-requests-remaining": "19999"}
        )
        
        # Make request
        response = build_request(
            api_key="test_key",
            sport_key="americanfootball_nfl",
            snapshot_at=snapshot,
            regions="us",
            markets="spreads,totals",
            bookmakers=None
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) == 1

    @responses.activate
    def test_api_rate_limit(self, mock_env_vars):
        """Test handling of rate limit error."""
        snapshot = dt.datetime(2023, 9, 1, 12, 0, 0, tzinfo=dt.timezone.utc)
        
        # Mock rate limit response
        responses.add(
            responses.GET,
            "https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/odds",
            json={"message": "Rate limit exceeded"},
            status=429,
            headers={"x-requests-reset": "1693569600"}
        )
        
        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="rate limit"):
            build_request(
                api_key="test_key",
                sport_key="americanfootball_nfl",
                snapshot_at=snapshot,
                regions="us",
                markets="spreads",
                bookmakers=None
            )
