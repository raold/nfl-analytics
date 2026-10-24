"""
Pytest configuration and shared fixtures.

This file is automatically loaded by pytest and provides:
- Database connection fixtures
- Mock API fixtures
- Sample data fixtures
- Environment setup
"""

import datetime as dt
import os
from collections.abc import Generator
from pathlib import Path

import psycopg
import pytest


@pytest.fixture(scope="session")
def test_db_dsn() -> str:
    """
    Return database connection string for tests.

    Uses TEST_DATABASE_URL if set, otherwise falls back to a test database.
    """
    return os.getenv("TEST_DATABASE_URL", "postgresql://testuser:testpass@localhost:5432/testdb")


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
    return dt.datetime(2023, 9, 1, 12, 0, 0, tzinfo=dt.UTC)


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
                            {"name": "Buffalo Bills", "price": 1.91, "point": -6.5},
                            {"name": "Arizona Cardinals", "price": 1.91, "point": 6.5},
                        ],
                    },
                    {
                        "key": "totals",
                        "last_update": "2023-09-01T12:00:00Z",
                        "outcomes": [
                            {"name": "Over", "price": 1.87, "point": 47.5},
                            {"name": "Under", "price": 1.95, "point": 47.5},
                        ],
                    },
                    {
                        "key": "h2h",
                        "last_update": "2023-09-01T12:00:00Z",
                        "outcomes": [
                            {"name": "Buffalo Bills", "price": 1.25},
                            {"name": "Arizona Cardinals", "price": 4.20},
                        ],
                    },
                ],
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
                            {"name": "Buffalo Bills", "price": 1.87, "point": -6.0},
                            {"name": "Arizona Cardinals", "price": 1.95, "point": 6.0},
                        ],
                    }
                ],
            },
        ],
    }


@pytest.fixture(autouse=True)
def _db_password_placeholder(request, monkeypatch):
    """Unit tests construct DB-backed classes with mocked connections, but the classes now
    refuse to start without DB_PASSWORD (no hardcoded fallback). Give them a placeholder;
    integration tests keep the real environment."""
    if "DB_PASSWORD" not in os.environ and request.node.get_closest_marker("integration") is None:
        monkeypatch.setenv("DB_PASSWORD", "unit-test-placeholder")


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
        "--runslow", action="store_true", default=False, help="Run slow tests (default: skip)"
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    # Existing markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )

    # P0 Test Suite markers (Nov 15, 2025)
    config.addinivalue_line(
        "markers", "critical: marks tests as critical (P0 priority, prevents expensive failures)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require database)"
    )
    config.addinivalue_line("markers", "benchmark: marks tests as performance benchmarks")


# Directories whose tests need live services (Postgres, Redis, Docker, the API).
# Everything in them is marked `integration` so `-m "not integration"` runs offline.
_INTEGRATION_DIRS = ("integration", "infrastructure", "e2e")


def pytest_ignore_collect(collection_path, config):
    """With -m "not integration", don't import service-dependent test modules at all.

    Several of them connect to Postgres or need heavy extras at import time, which would
    turn an offline run into collection errors before marker deselection applies.
    """
    if "not integration" not in (config.getoption("markexpr") or ""):
        return None
    try:
        top = collection_path.relative_to(Path(__file__).parent).parts[0]
    except (ValueError, IndexError):
        return None
    return True if top in _INTEGRATION_DIRS else None


def pytest_collection_modifyitems(config, items):
    """Auto-mark service-dependent tests; skip slow tests unless --runslow."""
    tests_root = Path(__file__).parent
    for item in items:
        try:
            top = item.path.relative_to(tests_root).parts[0]
        except ValueError:
            continue
        if top in _INTEGRATION_DIRS:
            item.add_marker(pytest.mark.integration)

    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return

    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
