#!/usr/bin/env python3
"""
Unit tests for BaseRepository with temporal leakage protection and retry logic.

Tests critical fixes:
- Task 1: Security (environment-based credentials)
- Task 2: Temporal leakage protection
- Task 9: Database connection retry logic

Author: NFL Analytics Team
Date: November 9, 2025
"""

import os
from unittest.mock import Mock, patch

import psycopg2
import pytest

from nfl_analytics.data.repository.base import (
    BaseRepository,
    ConnectionPoolError,
    DatabaseConfig,
    RepositoryError,
)


class TestDatabaseConfig:
    """Test DatabaseConfig security and initialization."""

    def test_requires_password_from_environment(self):
        """Test that password must be set (no hardcoded default)."""
        # Clear environment
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="DB_PASSWORD"):
                DatabaseConfig()

    def test_loads_from_environment_variables(self):
        """Test that config loads from environment variables."""
        env = {
            "DB_HOST": "test-host",
            "DB_PORT": "5555",
            "DB_NAME": "test-db",
            "DB_USER": "test-user",
            "DB_PASSWORD": "test-password",
            "DB_MIN_CONNECTIONS": "3",
            "DB_MAX_CONNECTIONS": "15",
        }

        with patch.dict(os.environ, env, clear=False):
            config = DatabaseConfig()
            assert config.host == "test-host"
            assert config.port == 5555
            assert config.database == "test-db"
            assert config.user == "test-user"
            assert config.password == "test-password"
            assert config.min_connections == 3
            assert config.max_connections == 15

    def test_uses_defaults_when_env_vars_missing(self):
        """Test that defaults are used when env vars not set."""
        env = {"DB_PASSWORD": "required-password"}

        with patch.dict(os.environ, env, clear=False):
            config = DatabaseConfig()
            assert config.host == "localhost"
            assert config.port == 5544
            assert config.database == "devdb01"
            assert config.user == "dro"
            assert config.min_connections == 5
            assert config.max_connections == 50


class TestTemporalLeakageProtection:
    """Test temporal leakage protection (Task 2)."""

    @pytest.fixture
    def mock_pool(self):
        """Mock connection pool for testing."""
        pool = Mock()
        conn = Mock()
        pool.getconn.return_value = conn
        return pool

    @pytest.fixture
    def repository(self, mock_pool):
        """Repository with mocked pool."""
        # Mock DatabaseConfig to bypass password validation
        mock_config = Mock()
        mock_config.host = "localhost"
        mock_config.port = 5544
        mock_config.database = "devdb01"
        mock_config.user = "test_user"
        mock_config.password = "test_password"

        repo = BaseRepository(config=mock_config)
        repo._pool = mock_pool
        repo._cache = {}
        return repo

    def test_blocks_deprecated_game_date_column(self, repository):
        """Test that queries using 'game_date' are blocked."""
        sql = "SELECT * FROM games WHERE game_date > '2024-01-01'"

        with pytest.raises(RepositoryError, match="TEMPORAL LEAKAGE.*kickoff"):
            repository.query(sql)

    def test_allows_kickoff_column(self, repository, mock_pool):
        """Test that queries using 'kickoff' are allowed."""
        sql = "SELECT * FROM games WHERE kickoff > '2024-01-01'"

        # Mock successful query execution
        mock_pool.getconn.return_value.cursor.return_value.__enter__.return_value.fetchall.return_value = [
            (1, "2024-01-01")
        ]
        mock_pool.getconn.return_value.cursor.return_value.__enter__.return_value.description = [
            ("id",),
            ("kickoff",),
        ]

        # Should not raise
        result = repository.query(sql)
        assert result.success

    def test_warns_on_missing_temporal_filter(self, repository, mock_pool, caplog):
        """Test warning when query lacks temporal filter."""
        import logging

        caplog.set_level(logging.WARNING)

        sql = "SELECT * FROM games"  # No WHERE clause with temporal filter

        # Mock successful execution
        mock_pool.getconn.return_value.cursor.return_value.__enter__.return_value.fetchall.return_value = (
            []
        )
        mock_pool.getconn.return_value.cursor.return_value.__enter__.return_value.description = [
            ("id",)
        ]

        repository.query(sql)

        # Should log warning
        assert any("TEMPORAL SAFETY WARNING" in record.message for record in caplog.records)

    def test_no_warning_with_asof_join(self, repository, mock_pool, caplog):
        """Test no warning when ASOF join is used."""
        import logging

        caplog.set_level(logging.WARNING)

        sql = """
        SELECT * FROM games g
        ASOF JOIN player_stats ps ON g.kickoff >= ps.stat_date
        """

        mock_pool.getconn.return_value.cursor.return_value.__enter__.return_value.fetchall.return_value = (
            []
        )
        mock_pool.getconn.return_value.cursor.return_value.__enter__.return_value.description = [
            ("id",)
        ]

        repository.query(sql)

        # Should not log warning (ASOF join indicates temporal awareness)
        assert not any("TEMPORAL SAFETY WARNING" in record.message for record in caplog.records)

    def test_no_warning_with_lag_window_function(self, repository, mock_pool, caplog):
        """Test no warning when LAG window function is used."""
        import logging

        caplog.set_level(logging.WARNING)

        sql = """
        SELECT player_id, LAG(passing_yards, 1) OVER (PARTITION BY player_id ORDER BY kickoff)
        FROM player_stats
        """

        mock_pool.getconn.return_value.cursor.return_value.__enter__.return_value.fetchall.return_value = (
            []
        )
        mock_pool.getconn.return_value.cursor.return_value.__enter__.return_value.description = [
            ("player_id",)
        ]

        repository.query(sql)

        # LAG indicates temporal awareness
        assert not any("TEMPORAL SAFETY WARNING" in record.message for record in caplog.records)

    def test_validation_can_be_disabled(self, repository, mock_pool):
        """Test that temporal validation can be disabled if needed."""
        sql = "SELECT * FROM games WHERE game_date > '2024-01-01'"

        # Mock execution
        mock_pool.getconn.return_value.cursor.return_value.__enter__.return_value.fetchall.return_value = (
            []
        )
        mock_pool.getconn.return_value.cursor.return_value.__enter__.return_value.description = [
            ("id",)
        ]

        # Should not raise when validation disabled
        result = repository.query(sql, validate_temporal=False)
        assert result.success


class TestConnectionRetryLogic:
    """Test database connection retry logic (Task 9)."""

    @pytest.fixture
    def repository(self):
        """Repository for testing retry logic."""
        with patch.dict(os.environ, {"DB_PASSWORD": "test"}):
            repo = BaseRepository()
            repo._cache = {}
            return repo

    def test_retries_on_operational_error(self, repository):
        """Test that OperationalError triggers retry."""
        with patch.object(repository, "_pool") as mock_pool:
            # First two calls fail, third succeeds
            mock_pool.getconn.side_effect = [
                psycopg2.OperationalError("Connection refused"),
                psycopg2.OperationalError("Connection refused"),
                Mock(),  # Success on third try
            ]

            # Should succeed after retries
            with repository.get_connection() as conn:
                assert conn is not None

            # Should have tried 3 times
            assert mock_pool.getconn.call_count == 3

    def test_retries_on_interface_error(self, repository):
        """Test that InterfaceError triggers retry."""
        with patch.object(repository, "_pool") as mock_pool:
            # First call fails, second succeeds
            mock_pool.getconn.side_effect = [
                psycopg2.InterfaceError("Interface error"),
                Mock(),  # Success on second try
            ]

            with repository.get_connection() as conn:
                assert conn is not None

            assert mock_pool.getconn.call_count == 2

    def test_retries_on_connection_pool_error(self, repository):
        """Test that ConnectionPoolError triggers retry."""
        with patch.object(repository, "_pool") as mock_pool:
            # First call fails, second succeeds
            mock_pool.getconn.side_effect = [
                ConnectionPoolError("Pool exhausted"),
                Mock(),
            ]

            with repository.get_connection() as conn:
                assert conn is not None

            assert mock_pool.getconn.call_count == 2

    def test_max_retry_attempts(self, repository):
        """Test that retry stops after max attempts (3)."""
        with patch.object(repository, "_pool") as mock_pool:
            # Always fail
            mock_pool.getconn.side_effect = psycopg2.OperationalError("Connection refused")

            # Should raise after 3 attempts
            with pytest.raises(RepositoryError):
                with repository.get_connection():
                    pass

            # Should have tried exactly 3 times
            assert mock_pool.getconn.call_count == 3

    def test_connection_returned_to_pool_on_success(self, repository):
        """Test that connection is returned to pool after use."""
        with patch.object(repository, "_pool") as mock_pool:
            mock_conn = Mock()
            mock_pool.getconn.return_value = mock_conn

            with repository.get_connection() as conn:
                assert conn == mock_conn

            # Connection should be returned to pool
            mock_pool.putconn.assert_called_once_with(mock_conn)

    def test_connection_returned_to_pool_on_error(self, repository):
        """Test that connection is returned even if query fails."""
        with patch.object(repository, "_pool") as mock_pool:
            mock_conn = Mock()
            mock_pool.getconn.return_value = mock_conn

            try:
                with repository.get_connection():
                    raise ValueError("Query failed")
            except ValueError:
                pass

            # Connection should still be returned to pool
            mock_pool.putconn.assert_called_once_with(mock_conn)

    def test_rollback_on_error(self, repository):
        """Test that connection is rolled back on error."""
        with patch.object(repository, "_pool") as mock_pool:
            mock_conn = Mock()
            mock_pool.getconn.return_value = mock_conn

            try:
                with repository.get_connection():
                    raise psycopg2.Error("Database error")
            except RepositoryError:
                pass

            # Connection should be rolled back
            mock_conn.rollback.assert_called_once()


class TestQueryCaching:
    """Test query result caching."""

    @pytest.fixture
    def repository(self):
        """Repository for caching tests."""
        with patch.dict(os.environ, {"DB_PASSWORD": "test"}):
            repo = BaseRepository()
            repo._pool = Mock()
            repo._cache = {}
            return repo

    def test_caches_query_results(self, repository):
        """Test that query results are cached."""
        mock_conn = Mock()
        repository._pool.getconn.return_value = mock_conn
        mock_cursor = mock_conn.cursor.return_value.__enter__.return_value
        mock_cursor.fetchall.return_value = [(1, "test")]
        mock_cursor.description = [("id",), ("name",)]

        # First query
        result1 = repository.query("SELECT * FROM test", cache_key="test_query", cache_ttl=60)

        # Second query with same cache key
        result2 = repository.query("SELECT * FROM test", cache_key="test_query", cache_ttl=60)

        # Should only execute query once (cached on second call)
        assert mock_cursor.execute.call_count == 1
        assert result1.rows == result2.rows

    def test_cache_respects_ttl(self, repository):
        """Test that cache expires after TTL."""
        mock_conn = Mock()
        repository._pool.getconn.return_value = mock_conn
        mock_cursor = mock_conn.cursor.return_value.__enter__.return_value
        mock_cursor.fetchall.return_value = [(1, "test")]
        mock_cursor.description = [("id",), ("name",)]

        # First query
        repository.query("SELECT * FROM test", cache_key="test_query", cache_ttl=0)

        # Wait for cache to expire (TTL=0 means immediate expiry)
        import time

        time.sleep(0.01)

        # Second query should execute again
        repository.query("SELECT * FROM test", cache_key="test_query", cache_ttl=0)

        # Should execute query twice
        assert mock_cursor.execute.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
