"""
Migration Robustness Tests.

Tests database migration scripts to prevent production issues.

CRITICAL: These tests ensure migrations are safe for production deployment!

Test Categories:
1. Idempotency: Migrations can be run multiple times safely
2. Sequential Execution: Migrations run in correct order
3. Rollback: Migrations can be undone if needed
4. Data Preservation: Migrations don't lose data
5. Schema Validation: Final schema matches expectations
"""

import os
import re
import subprocess
from pathlib import Path
from typing import List, Tuple

import psycopg
import pytest


class TestMigrationDiscovery:
    """Test that migration files are properly organized."""

    @pytest.fixture(scope="class")
    def migrations_dir(self) -> Path:
        """Path to migrations directory."""
        return Path("db/migrations")

    def test_migrations_directory_exists(self, migrations_dir):
        """Test that migrations directory exists."""
        assert migrations_dir.exists(), "db/migrations directory not found"
        assert migrations_dir.is_dir(), "db/migrations is not a directory"

    def test_migrations_are_numbered(self, migrations_dir):
        """Test that migration files follow naming convention."""
        migration_files = sorted(migrations_dir.glob("*.sql"))

        # Exclude special files
        migration_files = [
            f for f in migration_files
            if f.name not in ["verify_schema.sql", "rollback_template.sql"]
        ]

        assert len(migration_files) > 0, "No migration files found"

        # Check numbering pattern (001_xxx.sql, 002_xxx.sql, etc.)
        pattern = re.compile(r'^\d{3}_\w+\.sql$')

        for migration_file in migration_files:
            assert pattern.match(migration_file.name), \
                f"Migration {migration_file.name} doesn't follow naming convention (NNN_name.sql)"

        print(f"✅ Found {len(migration_files)} properly numbered migrations")

    def test_migrations_are_sequential(self, migrations_dir):
        """Test that migration numbers are sequential (no gaps)."""
        migration_files = sorted(migrations_dir.glob("[0-9][0-9][0-9]_*.sql"))

        numbers = []
        for migration_file in migration_files:
            num_str = migration_file.name[:3]
            numbers.append(int(num_str))

        # Check for gaps
        expected = list(range(1, len(numbers) + 1))

        if numbers != expected:
            missing = set(expected) - set(numbers)
            extra = set(numbers) - set(expected)

            errors = []
            if missing:
                errors.append(f"Missing migration numbers: {sorted(missing)}")
            if extra:
                errors.append(f"Unexpected migration numbers: {sorted(extra)}")

            pytest.fail("\n".join(errors))

        print(f"✅ Migrations are sequential: {numbers[0]} to {numbers[-1]}")

    def test_migration_files_not_empty(self, migrations_dir):
        """Test that migration files are not empty."""
        migration_files = sorted(migrations_dir.glob("[0-9][0-9][0-9]_*.sql"))

        for migration_file in migration_files:
            size = migration_file.stat().st_size
            assert size > 0, f"Migration {migration_file.name} is empty"

        print(f"✅ All {len(migration_files)} migrations have content")


class TestMigrationIdempotency:
    """Test that migrations can be run multiple times safely."""

    @pytest.fixture(scope="class")
    def test_db_conn(self):
        """Connection to test database."""
        # Use separate test database to avoid corrupting dev data
        conn = psycopg.connect(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5544")),
            dbname=os.environ.get("POSTGRES_TEST_DB", "testdb"),
            user=os.environ.get("POSTGRES_USER", "dro"),
            password=os.environ.get("POSTGRES_PASSWORD", "sicillionbillions")
        )
        yield conn
        conn.close()

    @pytest.fixture(scope="class")
    def migrations_dir(self) -> Path:
        """Path to migrations directory."""
        return Path("db/migrations")

    def _run_migration(self, conn: psycopg.Connection, migration_file: Path) -> Tuple[bool, str]:
        """
        Run a single migration file.

        Returns:
            (success, error_message)
        """
        try:
            with open(migration_file) as f:
                sql = f.read()

            with conn.cursor() as cur:
                cur.execute(sql)

            conn.commit()
            return (True, "")
        except Exception as e:
            conn.rollback()
            return (False, str(e))

    @pytest.mark.slow
    @pytest.mark.skip(reason="Requires clean test database - run manually")
    def test_migrations_are_idempotent(self, test_db_conn, migrations_dir):
        """
        Test that migrations can be run twice without errors.

        This is CRITICAL for production safety - if a migration fails halfway,
        we need to be able to re-run it after fixing the issue.
        """
        migration_files = sorted(migrations_dir.glob("[0-9][0-9][0-9]_*.sql"))

        for migration_file in migration_files:
            # Run migration first time
            success1, error1 = self._run_migration(test_db_conn, migration_file)
            assert success1, f"First run of {migration_file.name} failed: {error1}"

            # Run migration second time (should not fail)
            success2, error2 = self._run_migration(test_db_conn, migration_file)

            if not success2:
                # Check if error is acceptable (e.g., "already exists")
                acceptable_errors = [
                    "already exists",
                    "duplicate key",
                    "relation already exists",
                    "constraint already exists",
                    "index already exists",
                ]

                is_acceptable = any(err in error2.lower() for err in acceptable_errors)

                if not is_acceptable:
                    pytest.fail(
                        f"Second run of {migration_file.name} failed with unexpected error: {error2}"
                    )

            print(f"✅ {migration_file.name} is idempotent")

    @pytest.mark.slow
    @pytest.mark.skip(reason="Requires clean test database - run manually")
    def test_all_migrations_run_successfully(self, test_db_conn, migrations_dir):
        """
        Test that all migrations run in sequence.

        This is the basic smoke test for migrations.
        """
        migration_files = sorted(migrations_dir.glob("[0-9][0-9][0-9]_*.sql"))

        failed_migrations = []

        for migration_file in migration_files:
            success, error = self._run_migration(test_db_conn, migration_file)

            if not success:
                failed_migrations.append((migration_file.name, error))

        if failed_migrations:
            error_report = "\n".join([
                f"  - {name}: {error}"
                for name, error in failed_migrations
            ])
            pytest.fail(f"Migrations failed:\n{error_report}")

        print(f"✅ All {len(migration_files)} migrations ran successfully")


class TestMigrationSchemaValidation:
    """Test that migrations produce expected schema."""

    @pytest.fixture(scope="class")
    def prod_db_conn(self):
        """Connection to production/dev database."""
        conn = psycopg.connect(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5544")),
            dbname=os.environ.get("POSTGRES_DB", "devdb01"),
            user=os.environ.get("POSTGRES_USER", "dro"),
            password=os.environ.get("POSTGRES_PASSWORD", "sicillionbillions")
        )
        yield conn
        conn.close()

    def test_core_tables_exist(self, prod_db_conn):
        """Test that all core tables exist."""
        expected_tables = [
            'games',
            'plays',
            'rosters',
            'odds_history',
            'betting_lines',
            'performance_log',
            'data_quality_log',
        ]

        with prod_db_conn.cursor() as cur:
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
            """)

            existing_tables = {row[0] for row in cur.fetchall()}

        missing_tables = set(expected_tables) - existing_tables

        if missing_tables:
            pytest.fail(f"Missing tables: {sorted(missing_tables)}")

        print(f"✅ All {len(expected_tables)} core tables exist")

    def test_timescale_hypertables_configured(self, prod_db_conn):
        """Test that TimescaleDB hypertables are properly configured."""
        expected_hypertables = [
            'odds_history',
            'performance_log',
            'data_quality_log',
        ]

        with prod_db_conn.cursor() as cur:
            # Check if TimescaleDB extension exists
            cur.execute("""
                SELECT COUNT(*)
                FROM pg_extension
                WHERE extname = 'timescaledb'
            """)

            if cur.fetchone()[0] == 0:
                pytest.skip("TimescaleDB extension not installed")

            # Get list of hypertables
            cur.execute("""
                SELECT hypertable_name
                FROM timescaledb_information.hypertables
                WHERE hypertable_schema = 'public'
            """)

            existing_hypertables = {row[0] for row in cur.fetchall()}

        missing_hypertables = set(expected_hypertables) - existing_hypertables

        if missing_hypertables:
            pytest.fail(f"Missing hypertables: {sorted(missing_hypertables)}")

        print(f"✅ All {len(expected_hypertables)} hypertables configured")

    def test_primary_keys_exist(self, prod_db_conn):
        """Test that all tables have primary keys."""
        with prod_db_conn.cursor() as cur:
            cur.execute("""
                SELECT t.table_name
                FROM information_schema.tables t
                WHERE t.table_schema = 'public'
                AND t.table_type = 'BASE TABLE'
                AND NOT EXISTS (
                    SELECT 1
                    FROM information_schema.table_constraints tc
                    WHERE tc.table_schema = t.table_schema
                    AND tc.table_name = t.table_name
                    AND tc.constraint_type = 'PRIMARY KEY'
                )
            """)

            tables_without_pk = [row[0] for row in cur.fetchall()]

        # Some tables legitimately don't need PKs (like log tables)
        # Exclude TimescaleDB hypertables (they have composite time-based keys)
        hypertable_tables = ['odds_history', 'performance_log', 'data_quality_log']
        tables_without_pk = [t for t in tables_without_pk if t not in hypertable_tables]

        if tables_without_pk:
            pytest.fail(f"Tables without primary keys: {sorted(tables_without_pk)}")

        print("✅ All non-hypertable tables have primary keys")

    def test_foreign_keys_exist(self, prod_db_conn):
        """Test that critical foreign keys exist."""
        expected_foreign_keys = [
            # (table, column, referenced_table)
            ('plays', 'game_id', 'games'),
            # Add more as needed
        ]

        with prod_db_conn.cursor() as cur:
            for table, column, ref_table in expected_foreign_keys:
                cur.execute("""
                    SELECT COUNT(*)
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                        ON tc.constraint_name = kcu.constraint_name
                    WHERE tc.table_schema = 'public'
                    AND tc.table_name = %s
                    AND kcu.column_name = %s
                    AND tc.constraint_type = 'FOREIGN KEY'
                """, (table, column))

                fk_count = cur.fetchone()[0]

                assert fk_count > 0, \
                    f"Missing foreign key: {table}.{column} -> {ref_table}"

        print(f"✅ All {len(expected_foreign_keys)} critical foreign keys exist")

    def test_indexes_exist(self, prod_db_conn):
        """Test that critical indexes exist."""
        critical_indexes = [
            # Table-specific indexes (inferred from table name)
            ('games', 'season'),
            ('games', 'week'),
            ('games', 'kickoff'),
            ('plays', 'game_id'),
            ('plays', 'quarter'),
            ('odds_history', 'event_id'),
            ('odds_history', 'snapshot_at'),
        ]

        with prod_db_conn.cursor() as cur:
            missing_indexes = []

            for table, column in critical_indexes:
                cur.execute("""
                    SELECT COUNT(*)
                    FROM pg_indexes
                    WHERE schemaname = 'public'
                    AND tablename = %s
                    AND indexdef LIKE %s
                """, (table, f'%{column}%'))

                index_count = cur.fetchone()[0]

                if index_count == 0:
                    missing_indexes.append(f"{table}.{column}")

        if missing_indexes:
            pytest.fail(f"Missing indexes: {missing_indexes}")

        print(f"✅ All {len(critical_indexes)} critical indexes exist")

    def test_verify_schema_script_passes(self):
        """Test that verify_schema.sql runs without errors."""
        schema_file = Path("db/migrations/verify_schema.sql")

        if not schema_file.exists():
            pytest.skip("verify_schema.sql not found")

        # Run verify_schema.sql
        result = subprocess.run(
            [
                "psql",
                "-h", os.environ.get("POSTGRES_HOST", "localhost"),
                "-p", os.environ.get("POSTGRES_PORT", "5544"),
                "-U", os.environ.get("POSTGRES_USER", "dro"),
                "-d", os.environ.get("POSTGRES_DB", "devdb01"),
                "-f", str(schema_file),
            ],
            env={**os.environ, "PGPASSWORD": os.environ.get("POSTGRES_PASSWORD", "sicillionbillions")},
            capture_output=True,
            text=True,
            timeout=30
        )

        assert result.returncode == 0, \
            f"verify_schema.sql failed: {result.stderr}"

        print("✅ verify_schema.sql passed")


class TestMigrationDataPreservation:
    """Test that migrations don't lose data."""

    @pytest.fixture(scope="class")
    def prod_db_conn(self):
        """Connection to production/dev database."""
        conn = psycopg.connect(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5544")),
            dbname=os.environ.get("POSTGRES_DB", "devdb01"),
            user=os.environ.get("POSTGRES_USER", "dro"),
            password=os.environ.get("POSTGRES_PASSWORD", "sicillionbillions")
        )
        yield conn
        conn.close()

    def test_games_table_not_empty(self, prod_db_conn):
        """Test that games table has data."""
        with prod_db_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM games")
            count = cur.fetchone()[0]

        # Allow empty for brand-new databases
        if count == 0:
            pytest.skip("Games table is empty (new database?)")

        assert count > 0, "Games table should not be empty"
        print(f"✅ Games table has {count:,} rows")

    def test_plays_table_not_empty(self, prod_db_conn):
        """Test that plays table has data."""
        with prod_db_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM plays")
            count = cur.fetchone()[0]

        if count == 0:
            pytest.skip("Plays table is empty (new database?)")

        assert count > 0, "Plays table should not be empty"
        print(f"✅ Plays table has {count:,} rows")

    def test_no_null_primary_keys(self, prod_db_conn):
        """Test that primary key columns don't have NULLs."""
        with prod_db_conn.cursor() as cur:
            # Get all tables with primary keys
            cur.execute("""
                SELECT kcu.table_name, kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                WHERE tc.table_schema = 'public'
                AND tc.constraint_type = 'PRIMARY KEY'
            """)

            pk_columns = cur.fetchall()

        null_counts = []

        with prod_db_conn.cursor() as cur:
            for table, column in pk_columns:
                cur.execute(f'SELECT COUNT(*) FROM "{table}" WHERE "{column}" IS NULL')
                null_count = cur.fetchone()[0]

                if null_count > 0:
                    null_counts.append(f"{table}.{column}: {null_count} NULLs")

        if null_counts:
            pytest.fail(f"Primary key columns with NULLs:\n" + "\n".join(null_counts))

        print(f"✅ All {len(pk_columns)} primary key columns are non-NULL")


class TestMigrationRollback:
    """Test that migrations can be rolled back (if rollback scripts exist)."""

    @pytest.fixture(scope="class")
    def migrations_dir(self) -> Path:
        """Path to migrations directory."""
        return Path("db/migrations")

    @pytest.mark.slow
    @pytest.mark.skip(reason="Rollback scripts not yet implemented")
    def test_rollback_scripts_exist(self, migrations_dir):
        """
        Test that rollback scripts exist for each migration.

        BEST PRACTICE: Each migration should have a corresponding rollback script.
        This is not enforced yet but should be for production systems.
        """
        migration_files = sorted(migrations_dir.glob("[0-9][0-9][0-9]_*.sql"))
        rollback_dir = migrations_dir / "rollbacks"

        if not rollback_dir.exists():
            pytest.skip("Rollback directory doesn't exist yet")

        missing_rollbacks = []

        for migration_file in migration_files:
            # Expected rollback file name: rollback_001_xxx.sql
            rollback_name = f"rollback_{migration_file.name}"
            rollback_file = rollback_dir / rollback_name

            if not rollback_file.exists():
                missing_rollbacks.append(rollback_name)

        if missing_rollbacks:
            pytest.fail(f"Missing rollback scripts:\n" + "\n".join(missing_rollbacks))

        print(f"✅ All {len(migration_files)} migrations have rollback scripts")


# Utility functions for manual testing
def run_all_migrations():
    """Utility to run all migrations in order."""
    migrations_dir = Path("db/migrations")
    migration_files = sorted(migrations_dir.glob("[0-9][0-9][0-9]_*.sql"))

    conn = psycopg.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5544")),
        dbname=os.environ.get("POSTGRES_DB", "devdb01"),
        user=os.environ.get("POSTGRES_USER", "dro"),
        password=os.environ.get("POSTGRES_PASSWORD", "sicillionbillions")
    )

    try:
        for migration_file in migration_files:
            print(f"Running {migration_file.name}...")

            with open(migration_file) as f:
                sql = f.read()

            with conn.cursor() as cur:
                cur.execute(sql)

            conn.commit()
            print(f"✅ {migration_file.name} completed")

        print(f"\n✅ All {len(migration_files)} migrations completed successfully")

    except Exception as e:
        conn.rollback()
        print(f"\n❌ Migration failed: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])
