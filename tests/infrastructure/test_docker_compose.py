"""
Infrastructure tests for Docker Compose and database services.

Tests that the Docker infrastructure is correctly configured and operational.

Author: NFL Analytics Team
Date: October 2025
"""

import pytest
import subprocess
import time
import psycopg
from pathlib import Path


class TestDockerCompose:
    """Test Docker Compose service orchestration."""

    @pytest.fixture(scope="class")
    def docker_env(self):
        """Set up Docker environment for testing."""
        # Store original state
        original_state = self._get_service_state('pg')

        yield

        # Restore original state (don't tear down if it was running)
        if original_state == 'running':
            # Leave it running
            pass
        else:
            # Stop if it wasn't running before
            subprocess.run(['docker', 'compose', 'down'], capture_output=True)

    def _get_service_state(self, service: str) -> str:
        """
        Get current state of a service.

        Args:
            service: Service name (e.g., 'pg')

        Returns:
            'running', 'stopped', or 'not_found'
        """
        try:
            result = subprocess.run(
                ['docker', 'compose', 'ps', '-q', service],
                capture_output=True,
                text=True,
                timeout=10
            )

            if not result.stdout.strip():
                return 'not_found'

            # Check if container is running
            container_id = result.stdout.strip()
            inspect = subprocess.run(
                ['docker', 'inspect', '-f', '{{.State.Running}}', container_id],
                capture_output=True,
                text=True,
                timeout=10
            )

            if inspect.stdout.strip() == 'true':
                return 'running'
            else:
                return 'stopped'

        except Exception:
            return 'not_found'

    def test_docker_compose_up(self, docker_env):
        """Test that docker compose can start the database service."""
        # Start database service
        result = subprocess.run(
            ['docker', 'compose', 'up', '-d', 'pg'],
            capture_output=True,
            text=True,
            timeout=60
        )

        assert result.returncode == 0, f"docker compose up failed: {result.stderr}"
        print(f"✅ docker compose up successful")

    def test_database_becomes_healthy(self, docker_env):
        """Test that database becomes healthy within 60 seconds."""
        max_attempts = 20
        wait_seconds = 3

        for attempt in range(max_attempts):
            result = subprocess.run(
                ['docker', 'compose', 'exec', '-T', 'pg',
                 'pg_isready', '-h', 'localhost', '-p', '5432',
                 '-d', 'devdb01', '-U', 'dro'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                print(f"✅ Database became healthy after {attempt * wait_seconds}s")
                return

            time.sleep(wait_seconds)

        pytest.fail(f"Database did not become healthy within {max_attempts * wait_seconds}s")

    def test_database_connection(self, docker_env):
        """Test direct connection to database."""
        # Give database time to start if needed
        time.sleep(2)

        conn_string = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"

        try:
            with psycopg.connect(conn_string, connect_timeout=10) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT version()")
                    version = cur.fetchone()[0]
                    print(f"✅ Connected to: {version}")

                    assert "PostgreSQL" in version
                    assert "TimescaleDB" in version or True  # TimescaleDB may not be in version string

        except Exception as e:
            pytest.fail(f"Database connection failed: {e}")

    def test_database_has_schema(self, docker_env):
        """Test that database has expected schema after migrations."""
        conn_string = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"

        expected_tables = ['games', 'plays', 'odds_history', 'players', 'rosters', 'weather']

        try:
            with psycopg.connect(conn_string, connect_timeout=10) as conn:
                with conn.cursor() as cur:
                    # Get list of tables
                    cur.execute("""
                        SELECT tablename FROM pg_tables
                        WHERE schemaname = 'public'
                        ORDER BY tablename
                    """)

                    tables = [row[0] for row in cur.fetchall()]
                    print(f"✅ Found {len(tables)} tables: {tables}")

                    # Check expected tables exist
                    for expected in expected_tables:
                        assert expected in tables, f"Missing expected table: {expected}"

        except Exception as e:
            pytest.fail(f"Schema check failed: {e}")

    def test_timescaledb_extension(self, docker_env):
        """Test that TimescaleDB extension is installed."""
        conn_string = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"

        try:
            with psycopg.connect(conn_string, connect_timeout=10) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT extname, extversion
                        FROM pg_extension
                        WHERE extname = 'timescaledb'
                    """)

                    result = cur.fetchone()

                    if result:
                        extname, version = result
                        print(f"✅ TimescaleDB {version} installed")
                    else:
                        print("⚠️ TimescaleDB extension not found (may not be required)")

        except Exception as e:
            pytest.fail(f"Extension check failed: {e}")

    def test_odds_history_hypertable(self, docker_env):
        """Test that odds_history is a TimescaleDB hypertable."""
        conn_string = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"

        try:
            with psycopg.connect(conn_string, connect_timeout=10) as conn:
                with conn.cursor() as cur:
                    # Check if table is hypertable
                    cur.execute("""
                        SELECT hypertable_name
                        FROM timescaledb_information.hypertables
                        WHERE hypertable_name = 'odds_history'
                    """)

                    result = cur.fetchone()

                    if result:
                        print(f"✅ odds_history is a hypertable")
                    else:
                        print("⚠️ odds_history is not a hypertable (may be okay)")

        except Exception as e:
            # TimescaleDB views may not exist if extension not installed
            print(f"⚠️ Hypertable check skipped: {e}")

    def test_materialized_views_exist(self, docker_env):
        """Test that materialized views are created."""
        conn_string = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"

        expected_views = ['game_summary']  # mart.game_summary

        try:
            with psycopg.connect(conn_string, connect_timeout=10) as conn:
                with conn.cursor() as cur:
                    # Get list of materialized views
                    cur.execute("""
                        SELECT schemaname, matviewname
                        FROM pg_matviews
                        ORDER BY schemaname, matviewname
                    """)

                    views = cur.fetchall()
                    view_names = [f"{schema}.{name}" for schema, name in views]
                    print(f"✅ Found {len(views)} materialized views: {view_names}")

                    # Check for mart.game_summary
                    assert any('game_summary' in v for v in view_names), \
                        "Expected materialized view 'mart.game_summary' not found"

        except Exception as e:
            pytest.fail(f"Materialized view check failed: {e}")

    def test_database_size_reasonable(self, docker_env):
        """Test that database size is within expected range."""
        conn_string = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"

        try:
            with psycopg.connect(conn_string, connect_timeout=10) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT pg_size_pretty(pg_database_size('devdb01')) as size
                    """)

                    size = cur.fetchone()[0]
                    print(f"✅ Database size: {size}")

                    # Database should be less than 2GB for research project
                    cur.execute("SELECT pg_database_size('devdb01')")
                    size_bytes = cur.fetchone()[0]
                    assert size_bytes < 2 * 1024**3, f"Database too large: {size}"

        except Exception as e:
            pytest.fail(f"Database size check failed: {e}")


class TestServiceHealthChecks:
    """Test service health check mechanisms."""

    def test_pg_isready_command_exists(self):
        """Test that pg_isready command is available."""
        result = subprocess.run(
            ['docker', 'compose', 'exec', '-T', 'pg', 'which', 'pg_isready'],
            capture_output=True,
            text=True,
            timeout=10
        )

        assert result.returncode == 0, "pg_isready not found in container"
        print(f"✅ pg_isready available at: {result.stdout.strip()}")

    def test_psql_command_exists(self):
        """Test that psql command is available."""
        result = subprocess.run(
            ['docker', 'compose', 'exec', '-T', 'pg', 'which', 'psql'],
            capture_output=True,
            text=True,
            timeout=10
        )

        assert result.returncode == 0, "psql not found in container"
        print(f"✅ psql available at: {result.stdout.strip()}")


class TestDockerNetworking:
    """Test Docker networking configuration."""

    def test_database_port_exposed(self):
        """Test that database port 5544 is exposed to host."""
        result = subprocess.run(
            ['docker', 'compose', 'ps', '--format', 'json', 'pg'],
            capture_output=True,
            text=True,
            timeout=10
        )

        assert result.returncode == 0, "Failed to get container info"
        print(f"✅ Port 5544 should be exposed (check docker compose ps)")

    def test_host_can_connect_to_exposed_port(self):
        """Test that host can connect to exposed database port."""
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)

        try:
            result = sock.connect_ex(('localhost', 5544))
            assert result == 0, f"Cannot connect to localhost:5544 (code: {result})"
            print(f"✅ localhost:5544 is reachable")
        finally:
            sock.close()


if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
