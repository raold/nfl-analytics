"""
Pytest configuration and shared fixtures.

This file is automatically loaded by pytest and provides:
- Database connection fixtures
- Mock API fixtures
- Sample data fixtures
- Environment setup
"""

import os
import datetime as dt
from typing import Generator
import pytest
import psycopg


@pytest.fixture(scope="session")
def test_db_dsn() -> str:
    """
    Return database connection string for tests.
    
    Uses TEST_DATABASE_URL if set, otherwise falls back to a test database.
    """
    return os.getenv(
        "TEST_DATABASE_URL",
        "postgresql://testuser:testpass@localhost:5432/testdb"
    )


@pytest.fixture(scope="function")
def db_connection(test_db_dsn: str) -> Generator[psycopg.Connection, None, None]:
    """
    Provide a fresh database connection for each test.
    
    Automatically rolls back changes after the test completes.
    """
    conn = psycopg.connect(test_db_dsn)
    conn.autocommit = False
    
    try:
        yield conn
        conn.rollback()  # Rollback any changes
    finally:
        conn.close()


@pytest.fixture(scope="function")
def db_cursor(db_connection: psycopg.Connection) -> Generator[psycopg.Cursor, None, None]:
    """Provide a database cursor for direct SQL execution."""
    cursor = db_connection.cursor()
    try:
        yield cursor
    finally:
        cursor.close()


@pytest.fixture
def sample_snapshot_time() -> dt.datetime:
    """Return a consistent timestamp for testing."""
    return dt.datetime(2023, 9, 1, 12, 0, 0, tzinfo=dt.timezone.utc)


@pytest.fixture
def sample_odds_api_response() -> dict:
    """
    Return a sample response from The Odds API.
    
    Represents a single NFL game with multiple bookmakers and markets.
    """
    return {
        "id": "abc123def456",
        "sport_key": "americanfootball_nfl",
        "sport_title": "NFL",
        "commence_time": "2023-09-07T17:00:00Z",
        "home_team": "Buffalo Bills",
        "away_team": "Arizona Cardinals",
        "bookmakers": [
            {
                "key": "fanduel",
                "title": "FanDuel",
                "last_update": "2023-09-01T12:00:00Z",
                "markets": [
                    {
                        "key": "spreads",
                        "last_update": "2023-09-01T12:00:00Z",
                        "outcomes": [
                            {
                                "name": "Buffalo Bills",
                                "price": 1.91,
                                "point": -6.5
                            },
                            {
                                "name": "Arizona Cardinals",
                                "price": 1.91,
                                "point": 6.5
                            }
                        ]
                    },
                    {
                        "key": "totals",
                        "last_update": "2023-09-01T12:00:00Z",
                        "outcomes": [
                            {
                                "name": "Over",
                                "price": 1.87,
                                "point": 47.5
                            },
                            {
                                "name": "Under",
                                "price": 1.95,
                                "point": 47.5
                            }
                        ]
                    },
                    {
                        "key": "h2h",
                        "last_update": "2023-09-01T12:00:00Z",
                        "outcomes": [
                            {
                                "name": "Buffalo Bills",
                                "price": 1.25
                            },
                            {
                                "name": "Arizona Cardinals",
                                "price": 4.20
                            }
                        ]
                    }
                ]
            },
            {
                "key": "draftkings",
                "title": "DraftKings",
                "last_update": "2023-09-01T12:05:00Z",
                "markets": [
                    {
                        "key": "spreads",
                        "last_update": "2023-09-01T12:05:00Z",
                        "outcomes": [
                            {
                                "name": "Buffalo Bills",
                                "price": 1.87,
                                "point": -6.0
                            },
                            {
                                "name": "Arizona Cardinals",
                                "price": 1.95,
                                "point": 6.0
                            }
                        ]
                    }
                ]
            }
        ]
    }


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up environment variables for testing."""
    test_vars = {
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "POSTGRES_DB": "testdb",
        "POSTGRES_USER": "testuser",
        "POSTGRES_PASSWORD": "testpass",
        "ODDS_API_KEY": "test_api_key_123",
    }
    for key, value in test_vars.items():
        monkeypatch.setenv(key, value)
    return test_vars


# Pytest command-line options
def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run slow tests (default: skip)"
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command-line options."""
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
