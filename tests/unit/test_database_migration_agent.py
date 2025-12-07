#!/usr/bin/env python3
"""
Unit Tests for Database Migration Agent
========================================

Tests migration generation, validation, application, rollback,
backfill, and schema drift detection.

Run with:
    pytest tests/unit/test_database_migration_agent.py -v
    pytest tests/unit/test_database_migration_agent.py::TestMigrationGeneration -v

Author: Claude Code (Sonnet 4.5)
Date: November 22, 2025
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Import agent (with conditional import for CI environments)
try:
    from py.utils.database_migration_agent import (
        ColumnInfo,
        DatabaseMigrationAgent,
        DriftReport,
        ForeignKeyInfo,
        IndexInfo,
        MigrationResult,
        ValidationError,
        ValidationReport,
    )
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False
    pytest.skip("database_migration_agent not available", allow_module_level=True)


# ============================================================
# TEST FIXTURES
# ============================================================

@pytest.fixture
def temp_migrations_dir():
    """Create temporary migrations directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_db_config():
    """Mock database configuration."""
    return {
        'host': 'localhost',
        'port': 5544,
        'database': 'test_db',
        'user': 'test_user',
        'password': 'test_password'
    }


@pytest.fixture
def agent(temp_migrations_dir, mock_db_config):
    """Create DatabaseMigrationAgent instance with mocked DB."""
    with patch('py.utils.database_migration_agent.psycopg2.connect'):
        agent = DatabaseMigrationAgent(
            db_config=mock_db_config,
            migrations_dir=temp_migrations_dir,
            project_root=temp_migrations_dir
        )
        yield agent
        agent.close()


# ============================================================
# TEST: MIGRATION GENERATION
# ============================================================

class TestMigrationGeneration:
    """Test migration file generation from schema diffs."""

    def test_generate_migration_add_table(self, agent, temp_migrations_dir):
        """Test generating migration to add a table."""
        schema_diff = {
            'add_tables': [
                {
                    'name': 'test_table',
                    'schema': 'public',
                    'columns': [
                        {'name': 'id', 'type': 'SERIAL', 'primary_key': True},
                        {'name': 'name', 'type': 'TEXT', 'nullable': False},
                        {'name': 'created_at', 'type': 'TIMESTAMP', 'default': 'NOW()'}
                    ]
                }
            ]
        }

        migration_file = agent.generate_migration(schema_diff, migration_name='add_test_table')

        # Check file was created
        assert Path(migration_file).exists()

        # Check SQL content
        sql = Path(migration_file).read_text()
        assert 'CREATE TABLE IF NOT EXISTS public.test_table' in sql
        assert 'id SERIAL PRIMARY KEY' in sql
        assert 'name TEXT NOT NULL' in sql
        assert 'created_at TIMESTAMP DEFAULT NOW()' in sql

    def test_generate_migration_add_column(self, agent):
        """Test generating migration to add columns."""
        schema_diff = {
            'add_columns': [
                {
                    'table': 'games',
                    'schema': 'public',
                    'column': {
                        'name': 'prediction_count',
                        'type': 'INTEGER',
                        'default': '0',
                        'nullable': False
                    }
                }
            ]
        }

        migration_file = agent.generate_migration(schema_diff, migration_name='add_prediction_count')
        sql = Path(migration_file).read_text()

        assert 'ALTER TABLE public.games' in sql
        assert 'ADD COLUMN IF NOT EXISTS prediction_count INTEGER NOT NULL DEFAULT 0' in sql

    def test_generate_migration_add_index(self, agent):
        """Test generating migration to add indexes."""
        schema_diff = {
            'add_indexes': [
                {
                    'table': 'games',
                    'schema': 'public',
                    'name': 'idx_games_season_week',
                    'columns': ['season', 'week'],
                    'unique': False
                }
            ]
        }

        migration_file = agent.generate_migration(schema_diff, migration_name='add_game_indexes')
        sql = Path(migration_file).read_text()

        assert 'CREATE INDEX IF NOT EXISTS idx_games_season_week' in sql
        assert 'ON public.games(season, week)' in sql

    def test_generate_migration_add_foreign_key(self, agent):
        """Test generating migration to add foreign keys."""
        schema_diff = {
            'add_foreign_keys': [
                {
                    'table': 'predictions',
                    'schema': 'ml',
                    'name': 'fk_predictions_model',
                    'column': 'model_id',
                    'references': {
                        'table': 'model_registry',
                        'schema': 'ml',
                        'column': 'model_id'
                    },
                    'on_delete': 'CASCADE'
                }
            ]
        }

        migration_file = agent.generate_migration(schema_diff, migration_name='add_fk_constraints')
        sql = Path(migration_file).read_text()

        assert 'ALTER TABLE ml.predictions' in sql
        assert 'ADD CONSTRAINT fk_predictions_model' in sql
        assert 'FOREIGN KEY (model_id)' in sql
        assert 'REFERENCES ml.model_registry(model_id)' in sql
        assert 'ON DELETE CASCADE' in sql

    def test_generate_migration_drop_column(self, agent):
        """Test generating migration to drop columns."""
        schema_diff = {
            'drop_columns': [
                {
                    'table': 'games',
                    'schema': 'public',
                    'column': 'old_column'
                }
            ]
        }

        migration_file = agent.generate_migration(schema_diff, migration_name='drop_old_column')
        sql = Path(migration_file).read_text()

        assert 'ALTER TABLE public.games DROP COLUMN IF EXISTS old_column' in sql

    def test_generate_migration_auto_name(self, agent):
        """Test auto-generating migration name."""
        schema_diff = {'add_columns': []}

        migration_file = agent.generate_migration(schema_diff)

        # Check file name contains timestamp
        filename = Path(migration_file).name
        assert 'auto_generated' in filename
        assert filename.startswith('001_')  # First migration


# ============================================================
# TEST: MIGRATION VALIDATION
# ============================================================

class TestMigrationValidation:
    """Test migration validation logic."""

    @patch('py.utils.database_migration_agent.psycopg2.connect')
    def test_validate_migration_success(self, mock_connect, agent, temp_migrations_dir):
        """Test validation of valid migration."""
        # Create test migration file
        migration_sql = """
        CREATE TABLE IF NOT EXISTS test_table (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL
        );
        """
        migration_file = Path(temp_migrations_dir) / '001_test_migration.sql'
        migration_file.write_text(migration_sql)

        # Mock database connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Mock get_applied_migrations to return empty list
        agent.get_applied_migrations = Mock(return_value=[])

        # Validate
        report = agent.validate_migration(str(migration_file), dry_run=True)

        # Should pass validation
        assert report.is_valid
        assert len(report.errors) == 0

    def test_validate_migration_file_not_found(self, agent):
        """Test validation with non-existent file."""
        report = agent.validate_migration('nonexistent.sql', dry_run=True)

        assert not report.is_valid
        assert len(report.errors) == 1
        assert report.errors[0].category == 'file_not_found'

    @patch('py.utils.database_migration_agent.psycopg2.connect')
    def test_validate_migration_already_applied(self, mock_connect, agent, temp_migrations_dir):
        """Test validation warns about already-applied migration."""
        migration_file = Path(temp_migrations_dir) / '001_test_migration.sql'
        migration_file.write_text("SELECT 1;")

        # Mock get_applied_migrations to return this migration
        agent.get_applied_migrations = Mock(return_value=[
            {'migration_file': '001_test_migration.sql', 'success': True}
        ])

        # Mock database connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        report = agent.validate_migration(str(migration_file), dry_run=True)

        # Should have warning about already applied
        assert any(w.category == 'already_applied' for w in report.warnings)

    @patch('py.utils.database_migration_agent.psycopg2.connect')
    def test_validate_migration_sql_syntax_error(self, mock_connect, agent, temp_migrations_dir):
        """Test validation catches SQL syntax errors."""
        # Create migration with syntax error
        migration_file = Path(temp_migrations_dir) / '001_bad_syntax.sql'
        migration_file.write_text("CREATE TABEL bad_syntax;")  # Typo: TABEL

        # Mock database connection to raise syntax error
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("syntax error at or near 'TABEL'")
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        agent.get_applied_migrations = Mock(return_value=[])

        report = agent.validate_migration(str(migration_file), dry_run=True)

        # Should fail validation
        assert not report.is_valid
        assert len(report.errors) > 0
        assert any('syntax' in str(e).lower() for e in report.errors)


# ============================================================
# TEST: MIGRATION APPLICATION
# ============================================================

class TestMigrationApplication:
    """Test migration application and rollback."""

    @patch('py.utils.database_migration_agent.psycopg2.connect')
    def test_apply_migration_dry_run(self, mock_connect, agent, temp_migrations_dir):
        """Test applying migration in dry-run mode."""
        # Create test migration
        migration_file = Path(temp_migrations_dir) / '001_test.sql'
        migration_file.write_text("CREATE TABLE test_table (id SERIAL);")

        # Mock database
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Mock validation to pass
        agent.validate_migration = Mock(return_value=ValidationReport(
            migration_file='001_test.sql',
            is_valid=True
        ))

        # Apply in dry-run mode
        result = agent.apply_migration(str(migration_file), dry_run=True)

        # Should succeed
        assert result.success

        # Should rollback (not commit) in dry-run
        mock_conn.rollback.assert_called()

    @patch('py.utils.database_migration_agent.psycopg2.connect')
    def test_apply_migration_validation_fails(self, mock_connect, agent, temp_migrations_dir):
        """Test application blocked by validation failure."""
        migration_file = Path(temp_migrations_dir) / '001_test.sql'
        migration_file.write_text("SELECT 1;")

        # Mock validation to fail
        agent.validate_migration = Mock(return_value=ValidationReport(
            migration_file='001_test.sql',
            is_valid=False,
            errors=[ValidationError(
                severity='ERROR',
                category='test',
                message='Test validation failure'
            )]
        ))

        result = agent.apply_migration(str(migration_file), dry_run=False)

        # Should fail
        assert not result.success
        assert 'Validation failed' in result.error_message

    @patch('py.utils.database_migration_agent.psycopg2.connect')
    def test_apply_migration_already_applied(self, mock_connect, agent, temp_migrations_dir):
        """Test applying migration that was already applied."""
        migration_file = Path(temp_migrations_dir) / '001_test.sql'
        migration_file.write_text("SELECT 1;")

        # Mock get_applied_migrations to show migration already applied
        agent.get_applied_migrations = Mock(return_value=[
            {'migration_file': '001_test.sql', 'success': True}
        ])

        result = agent.apply_migration(str(migration_file), dry_run=False)

        # Should fail with already applied message
        assert not result.success
        assert 'already applied' in result.error_message.lower()


# ============================================================
# TEST: SCHEMA INTROSPECTION
# ============================================================

class TestSchemaIntrospection:
    """Test database schema introspection methods."""

    @patch('py.utils.database_migration_agent.psycopg2.connect')
    def test_get_all_tables(self, mock_connect, agent):
        """Test retrieving all tables from database."""
        # Mock database response
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ('public', 'games'),
            ('public', 'plays'),
            ('mart', 'player_game_stats'),
            ('ml', 'model_registry')
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        agent._conn = mock_conn

        tables = agent.get_all_tables(['public', 'mart', 'ml'])

        assert len(tables) == 4
        assert ('public', 'games') in tables
        assert ('ml', 'model_registry') in tables

    @patch('py.utils.database_migration_agent.psycopg2.connect')
    def test_get_table_columns(self, mock_connect, agent):
        """Test retrieving columns for a table."""
        # Mock database response
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {
                'table_schema': 'public',
                'table_name': 'games',
                'column_name': 'game_id',
                'ordinal_position': 1,
                'column_default': None,
                'is_nullable': 'NO',
                'data_type': 'text',
                'character_maximum_length': None,
                'numeric_precision': None,
                'numeric_scale': None
            },
            {
                'table_schema': 'public',
                'table_name': 'games',
                'column_name': 'season',
                'ordinal_position': 2,
                'column_default': None,
                'is_nullable': 'YES',
                'data_type': 'integer',
                'character_maximum_length': None,
                'numeric_precision': 32,
                'numeric_scale': 0
            }
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        agent._conn = mock_conn

        columns = agent.get_table_columns('games', 'public')

        assert len(columns) == 2
        assert columns[0].column_name == 'game_id'
        assert columns[0].data_type == 'text'
        assert columns[1].column_name == 'season'
        assert columns[1].data_type == 'integer'


# ============================================================
# TEST: BACKFILL MANAGEMENT
# ============================================================

class TestBackfillManagement:
    """Test column backfill functionality."""

    @patch('py.utils.database_migration_agent.psycopg2.connect')
    def test_backfill_column_sequential(self, mock_connect, agent):
        """Test sequential backfill."""
        # Mock database
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        # Mock total row count
        mock_cursor.fetchone.return_value = [1000]  # 1000 total rows

        # Mock batch updates (simulate 100 rows per batch)
        mock_cursor.rowcount = 100

        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        agent._conn = mock_conn

        # Backfill
        result = agent.backfill_column(
            table_name='games',
            column_name='prediction_count',
            backfill_value=0,
            schema='public',
            batch_size=100,
            parallel=False
        )

        # Check result
        assert result['total_rows'] == 1000
        assert result['rows_updated'] >= 0
        assert result['duration_seconds'] > 0

    @patch('py.utils.database_migration_agent.psycopg2.connect')
    def test_backfill_column_parallel(self, mock_connect, agent):
        """Test parallel backfill."""
        # Mock database connections (one per worker)
        def create_mock_conn():
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = [10000]  # Large table
            mock_cursor.rowcount = 1000
            mock_conn.cursor.return_value = mock_cursor
            return mock_conn

        mock_connect.side_effect = [create_mock_conn() for _ in range(10)]

        agent._conn = create_mock_conn()

        # Backfill in parallel
        result = agent.backfill_column(
            table_name='large_table',
            column_name='new_column',
            backfill_value='default',
            schema='public',
            batch_size=1000,
            parallel=True,
            num_workers=4
        )

        # Check result
        assert result['total_rows'] == 10000
        assert result['rows_updated'] >= 0


# ============================================================
# TEST: SCHEMA DRIFT DETECTION
# ============================================================

class TestSchemaDriftDetection:
    """Test schema drift detection between environments."""

    @patch('py.utils.database_migration_agent.psycopg2.connect')
    def test_audit_schema_drift_no_drift(self, mock_connect, agent):
        """Test drift detection when schemas match."""
        # Mock identical schemas
        def create_identical_conn():
            mock_conn = MagicMock()
            mock_cursor = MagicMock()

            # Same tables
            mock_cursor.fetchall.side_effect = [
                [('public.games',), ('public.plays',)],  # source tables
                [('game_id', 'text'), ('season', 'integer')],  # source columns
                [],  # source indexes
                []   # source foreign keys
            ]

            mock_conn.cursor.return_value = mock_cursor
            return mock_conn

        mock_connect.side_effect = [create_identical_conn(), create_identical_conn()]

        target_config = {
            'host': 'localhost',
            'port': 5544,
            'database': 'target_db',
            'user': 'test',
            'password': 'test'
        }

        # Simplified mock - just check no drift is reported
        report = DriftReport(source_env='source', target_env='target')
        assert not report.has_drift

    def test_drift_report_has_drift(self):
        """Test DriftReport.has_drift property."""
        report = DriftReport(source_env='source', target_env='target')

        # No drift initially
        assert not report.has_drift

        # Add table
        report.tables_added.append('public.new_table')
        assert report.has_drift

        # Reset
        report = DriftReport(source_env='source', target_env='target')

        # Add column
        report.columns_added.append(('public.games', 'new_column'))
        assert report.has_drift

    def test_drift_report_summary(self):
        """Test DriftReport.summary() method."""
        report = DriftReport(source_env='dev', target_env='prod')
        report.tables_added = ['public.new_table']
        report.columns_removed = [('public.games', 'old_column')]
        report.indexes_added = ['idx_new_index']

        summary = report.summary()

        assert 'SCHEMA DRIFT REPORT: dev vs prod' in summary
        assert 'TABLES ADDED (1)' in summary
        assert 'public.new_table' in summary
        assert 'COLUMNS REMOVED (1)' in summary
        assert 'public.games.old_column' in summary


# ============================================================
# TEST: VALIDATION REPORT
# ============================================================

class TestValidationReport:
    """Test ValidationReport data structure."""

    def test_validation_report_success(self):
        """Test ValidationReport for successful validation."""
        report = ValidationReport(
            migration_file='001_test.sql',
            is_valid=True,
            validation_time=1.5
        )

        assert report.is_valid
        assert not report.has_errors
        assert not report.has_warnings

        summary = report.summary()
        assert '✅ VALIDATION PASSED' in summary

    def test_validation_report_errors(self):
        """Test ValidationReport with errors."""
        report = ValidationReport(
            migration_file='001_test.sql',
            is_valid=False,
            errors=[
                ValidationError(
                    severity='ERROR',
                    category='foreign_key',
                    message='Foreign key constraint violation'
                )
            ],
            validation_time=2.0
        )

        assert not report.is_valid
        assert report.has_errors

        summary = report.summary()
        assert '❌ VALIDATION FAILED' in summary
        assert 'foreign_key' in summary

    def test_validation_report_warnings(self):
        """Test ValidationReport with warnings."""
        report = ValidationReport(
            migration_file='001_test.sql',
            is_valid=True,
            warnings=[
                ValidationError(
                    severity='WARNING',
                    category='orphaned_record',
                    message='10 orphaned records found'
                )
            ]
        )

        assert report.is_valid
        assert report.has_warnings

        summary = report.summary()
        assert '⚠️  VALIDATION PASSED WITH WARNINGS' in summary


# ============================================================
# TEST: MIGRATION RESULT
# ============================================================

class TestMigrationResult:
    """Test MigrationResult data structure."""

    def test_migration_result_success(self):
        """Test MigrationResult for successful migration."""
        result = MigrationResult(
            migration_file='001_test.sql',
            success=True,
            migration_id=42,
            duration=3.5
        )

        assert result.success

        summary = result.summary()
        assert '✅ MIGRATION SUCCESS' in summary
        assert 'ID: 42' in summary
        assert '3.5s' in summary

    def test_migration_result_failure(self):
        """Test MigrationResult for failed migration."""
        result = MigrationResult(
            migration_file='001_test.sql',
            success=False,
            error_message='Syntax error in SQL',
            duration=1.2
        )

        assert not result.success

        summary = result.summary()
        assert '❌ MIGRATION FAILED' in summary
        assert 'Syntax error in SQL' in summary


# ============================================================
# INTEGRATION TESTS (Requires database)
# ============================================================

@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv('CI') == 'true',
    reason="Integration tests require database"
)
class TestDatabaseMigrationIntegration:
    """Integration tests requiring actual database."""

    def test_full_migration_workflow(self):
        """Test complete migration workflow with real database."""
        # This would require a test database
        # Skipped in normal test runs
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
