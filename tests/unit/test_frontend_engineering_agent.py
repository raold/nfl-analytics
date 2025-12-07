"""
Unit tests for Frontend Engineering Agent

Tests schema validation, API contract enforcement, drift detection, and migration suggestions.

Author: Claude Code
Created: November 22, 2025
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from py.utils.frontend_engineering_agent import (
    FrontendEngineeringAgent,
    SchemaDrift,
    SchemaValidationReport,
    Severity,
    ValidationIssue,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_db_config():
    """Mock database configuration"""
    return {
        "host": "localhost",
        "port": 5544,
        "dbname": "testdb",
        "user": "testuser",
        "password": "testpass",  # pragma: allowlist secret
    }


@pytest.fixture
def sample_schema():
    """Sample database schema for testing"""
    return {
        "tables": {
            "public.games": {
                "columns": ["game_id", "season", "week", "home_team", "away_team", "kickoff"],
                "column_types": {
                    "game_id": "text",
                    "season": "integer",
                    "week": "integer",
                    "home_team": "text",
                    "away_team": "text",
                    "kickoff": "date",
                },
                "primary_keys": ["game_id"],
                "nullable": {
                    "game_id": False,
                    "season": False,
                    "week": False,
                    "home_team": False,
                    "away_team": False,
                    "kickoff": True,
                },
                "defaults": {},
            },
            "mart.player_game_stats": {
                "columns": ["player_id", "season", "week", "current_team", "passing_yards"],
                "column_types": {
                    "player_id": "text",
                    "season": "integer",
                    "week": "integer",
                    "current_team": "text",
                    "passing_yards": "integer",
                },
                "primary_keys": ["player_id", "season", "week"],
                "nullable": {
                    "player_id": False,
                    "season": False,
                    "week": False,
                    "current_team": False,
                    "passing_yards": True,
                },
                "defaults": {},
            },
        },
        "views": {},
        "materialized_views": {},
    }


@pytest.fixture
def agent(mock_db_config, tmp_path):
    """Create FrontendEngineeringAgent with mock config"""
    dashboard_path = tmp_path / "dashboard" / "app.py"
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_path.touch()

    snapshot_path = tmp_path / "schema_snapshot.json"

    agent = FrontendEngineeringAgent(
        db_config=mock_db_config, dashboard_path=dashboard_path, schema_snapshot_path=snapshot_path
    )

    return agent


# ============================================================================
# Schema Introspection Tests
# ============================================================================


def test_get_database_schema_no_psycopg(agent):
    """Test schema introspection when psycopg unavailable"""
    with patch("py.utils.frontend_engineering_agent.PSYCOPG_AVAILABLE", False):
        schema = agent.get_database_schema()

    assert schema == {"tables": {}}


def test_get_database_schema_connection_error(agent, mock_db_config):
    """Test schema introspection with connection error"""
    with patch("psycopg.connect", side_effect=Exception("Connection failed")):
        schema = agent.get_database_schema()

    assert "tables" in schema
    assert len(schema["tables"]) == 0


def test_save_schema_snapshot(agent, sample_schema):
    """Test saving schema snapshot"""
    agent._schema_cache = sample_schema

    snapshot_path = agent.save_schema_snapshot()

    assert snapshot_path.exists()

    with open(snapshot_path) as f:
        snapshot = json.load(f)

    assert "timestamp" in snapshot
    assert "schema" in snapshot
    assert snapshot["schema"] == sample_schema


def test_load_schema_snapshot_not_found(agent):
    """Test loading snapshot when file doesn't exist"""
    snapshot = agent.load_schema_snapshot()

    assert snapshot is None


def test_load_schema_snapshot_success(agent, sample_schema):
    """Test loading schema snapshot successfully"""
    # Save snapshot first
    agent._schema_cache = sample_schema
    agent.save_schema_snapshot()

    # Clear cache and reload
    agent._schema_cache = None
    loaded_schema = agent.load_schema_snapshot()

    assert loaded_schema == sample_schema


# ============================================================================
# SQL Query Extraction Tests
# ============================================================================


def test_extract_sql_queries_multiline(agent, tmp_path):
    """Test extracting multi-line SQL queries"""
    python_file = tmp_path / "test.py"
    python_file.write_text(
        '''
query = """
    SELECT game_id, season, week
    FROM public.games
    WHERE season = 2024
"""
'''
    )

    queries = agent.extract_sql_queries(python_file)

    assert len(queries) == 1
    query, line_num = queries[0]
    assert "SELECT" in query
    assert "FROM public.games" in query
    assert line_num == 2


def test_extract_sql_queries_single_line(agent, tmp_path):
    """Test extracting single-line SQL queries"""
    python_file = tmp_path / "test.py"
    python_file.write_text(
        """
result = conn.execute("SELECT * FROM games WHERE season = 2024")
"""
    )

    queries = agent.extract_sql_queries(python_file)

    assert len(queries) == 1
    query, line_num = queries[0]
    assert "SELECT * FROM games" in query


def test_extract_sql_queries_execute_pattern(agent, tmp_path):
    """Test extracting queries from execute() calls"""
    python_file = tmp_path / "test.py"
    python_file.write_text(
        """
cur.execute("SELECT COUNT(*) FROM games")
conn.execute("SELECT player_id FROM mart.player_game_stats")
"""
    )

    queries = agent.extract_sql_queries(python_file)

    assert len(queries) >= 2
    assert any("games" in q[0] for q in queries)
    assert any("player_game_stats" in q[0] for q in queries)


# ============================================================================
# SQL Query Validation Tests
# ============================================================================


def test_validate_sql_query_table_not_found(agent, sample_schema):
    """Test validation when table doesn't exist"""
    query = "SELECT * FROM nonexistent_table"

    issues = agent.validate_sql_query(query, sample_schema)

    assert len(issues) > 0
    assert any("does not exist" in issue.message for issue in issues)
    assert any(issue.severity == Severity.ERROR for issue in issues)


def test_validate_sql_query_column_not_found(agent, sample_schema):
    """Test validation when column doesn't exist"""
    # This is the CRITICAL bug that occurred DOZENS of times
    query = "SELECT pgs.game_id FROM mart.player_game_stats pgs"

    issues = agent.validate_sql_query(query, sample_schema)

    # Should detect that game_id doesn't exist in player_game_stats
    assert len(issues) > 0
    assert any("game_id" in issue.message and "does not exist" in issue.message for issue in issues)


def test_validate_sql_query_select_star_warning(agent, sample_schema):
    """Test warning for SELECT * queries"""
    query = "SELECT * FROM public.games"

    issues = agent.validate_sql_query(query, sample_schema)

    assert any(issue.severity == Severity.WARNING for issue in issues)
    assert any("SELECT *" in issue.message for issue in issues)


def test_validate_sql_query_missing_where_large_table(agent, sample_schema):
    """Test warning for queries on large tables without WHERE"""
    # Add plays table to schema (large table)
    sample_schema["tables"]["public.plays"] = {
        "columns": ["play_id", "game_id", "play_type"],
        "column_types": {"play_id": "bigint", "game_id": "text", "play_type": "text"},
        "primary_keys": ["play_id"],
        "nullable": {"play_id": False, "game_id": False, "play_type": True},
        "defaults": {},
    }

    query = "SELECT * FROM plays"

    issues = agent.validate_sql_query(query, sample_schema)

    assert any(issue.severity == Severity.WARNING for issue in issues)
    assert any("WHERE" in issue.message or "LIMIT" in issue.message for issue in issues)


def test_validate_sql_query_valid(agent, sample_schema):
    """Test validation of valid query"""
    query = """
        SELECT g.game_id, g.season, g.week
        FROM public.games g
        WHERE g.season = 2024
        LIMIT 100
    """

    issues = agent.validate_sql_query(query, sample_schema)

    # Should only have warnings (SELECT *), no errors
    assert not any(issue.severity == Severity.ERROR for issue in issues)


# ============================================================================
# Schema Validation Report Tests
# ============================================================================


def test_validate_dashboard_schema_file_not_found(agent):
    """Test validation when dashboard file doesn't exist"""
    agent.dashboard_path = Path("/nonexistent/dashboard.py")

    report = agent.validate_dashboard_schema()

    assert not report.is_valid
    assert any(issue.severity == Severity.CRITICAL for issue in report.issues)


def test_validate_dashboard_schema_success(agent, sample_schema, tmp_path):
    """Test successful dashboard validation"""
    # Create dashboard with valid queries
    dashboard_path = tmp_path / "dashboard" / "app.py"
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_path.write_text(
        '''
result = conn.execute("""
    SELECT g.game_id, g.season
    FROM public.games g
    WHERE g.season = 2024
    LIMIT 100
""")
'''
    )

    agent.dashboard_path = dashboard_path
    agent._schema_cache = sample_schema

    report = agent.validate_dashboard_schema()

    assert report.queries_checked > 0
    # May have warnings but no errors
    assert not any(issue.severity in [Severity.ERROR, Severity.CRITICAL] for issue in report.issues)


def test_validate_dashboard_schema_with_errors(agent, sample_schema, tmp_path):
    """Test dashboard validation with schema errors"""
    # Create dashboard with INVALID queries (the DOZENS of times bug)
    dashboard_path = tmp_path / "dashboard" / "app.py"
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_path.write_text(
        '''
# This is the bug that occurred DOZENS of times
result = conn.execute("""
    SELECT pgs.game_id
    FROM mart.player_game_stats pgs
    WHERE pgs.player_id = '00-0033873'
""")
'''
    )

    agent.dashboard_path = dashboard_path
    agent._schema_cache = sample_schema

    report = agent.validate_dashboard_schema()

    assert report.queries_checked > 0
    assert not report.is_valid
    assert any(issue.severity == Severity.ERROR for issue in report.issues)
    assert any("game_id" in issue.message for issue in report.issues)


def test_schema_validation_report_generate(agent):
    """Test SchemaValidationReport.generate_report()"""
    report = SchemaValidationReport()
    report.queries_checked = 10
    report.queries_valid = 8
    report.queries_invalid = 2

    report.add_issue(
        ValidationIssue(
            severity=Severity.ERROR,
            category="schema_validation",
            message="Column 'game_id' does not exist",
            file_path="/dashboard/app.py",
            line_number=42,
        )
    )

    text_report = report.generate_report()

    assert "FRONTEND SCHEMA VALIDATION REPORT" in text_report
    assert "Queries Checked: 10" in text_report
    assert "Valid: 8 | Invalid: 2" in text_report
    assert "❌ FAILED" in text_report
    assert "game_id" in text_report


# ============================================================================
# Schema Drift Detection Tests
# ============================================================================


def test_detect_schema_drift_no_snapshot(agent):
    """Test drift detection when no snapshot exists"""
    drift = agent.detect_schema_drift()

    # Should return empty drift
    assert not drift.has_any_changes
    assert not drift.has_breaking_changes


def test_detect_schema_drift_added_tables(agent, sample_schema):
    """Test detection of added tables"""
    # Save baseline snapshot
    agent._schema_cache = sample_schema
    agent.save_schema_snapshot()

    # Add new table to current schema
    new_schema = sample_schema.copy()
    new_schema["tables"]["public.new_table"] = {
        "columns": ["id", "name"],
        "column_types": {"id": "integer", "name": "text"},
        "primary_keys": ["id"],
        "nullable": {"id": False, "name": True},
        "defaults": {},
    }

    agent._schema_cache = new_schema

    drift = agent.detect_schema_drift()

    assert drift.has_any_changes
    assert not drift.has_breaking_changes  # Added tables are non-breaking
    assert "public.new_table" in drift.added_tables


def test_detect_schema_drift_removed_tables(agent, sample_schema):
    """Test detection of removed tables (BREAKING)"""
    # Save baseline snapshot
    agent._schema_cache = sample_schema
    agent.save_schema_snapshot()

    # Remove table from current schema
    new_schema = {"tables": {}, "views": {}, "materialized_views": {}}

    agent._schema_cache = new_schema

    drift = agent.detect_schema_drift()

    assert drift.has_any_changes
    assert drift.has_breaking_changes  # Removed tables are BREAKING
    assert len(drift.removed_tables) > 0


def test_detect_schema_drift_added_columns(agent, sample_schema):
    """Test detection of added columns"""
    # Save baseline snapshot
    agent._schema_cache = sample_schema
    agent.save_schema_snapshot()

    # Add column to existing table
    new_schema = sample_schema.copy()
    new_schema["tables"]["public.games"]["columns"].append("new_column")
    new_schema["tables"]["public.games"]["column_types"]["new_column"] = "text"
    new_schema["tables"]["public.games"]["nullable"]["new_column"] = True

    agent._schema_cache = new_schema

    drift = agent.detect_schema_drift()

    assert drift.has_any_changes
    assert not drift.has_breaking_changes  # Added columns are non-breaking
    assert "public.games" in drift.added_columns
    assert "new_column" in drift.added_columns["public.games"]


def test_detect_schema_drift_removed_columns(agent, sample_schema):
    """Test detection of removed columns (BREAKING)"""
    # Save baseline snapshot
    agent._schema_cache = sample_schema
    agent.save_schema_snapshot()

    # Remove column from existing table
    new_schema = sample_schema.copy()
    new_schema["tables"]["public.games"]["columns"].remove("kickoff")
    del new_schema["tables"]["public.games"]["column_types"]["kickoff"]
    del new_schema["tables"]["public.games"]["nullable"]["kickoff"]

    agent._schema_cache = new_schema

    drift = agent.detect_schema_drift()

    assert drift.has_any_changes
    assert drift.has_breaking_changes  # Removed columns are BREAKING
    assert "public.games" in drift.removed_columns
    assert "kickoff" in drift.removed_columns["public.games"]


def test_detect_schema_drift_type_changes(agent, sample_schema):
    """Test detection of column type changes"""
    # Save baseline snapshot
    agent._schema_cache = sample_schema
    agent.save_schema_snapshot()

    # Change column type
    new_schema = sample_schema.copy()
    new_schema["tables"]["public.games"]["column_types"]["season"] = "bigint"

    agent._schema_cache = new_schema

    drift = agent.detect_schema_drift()

    assert drift.has_any_changes
    assert drift.has_breaking_changes  # Type changes can be BREAKING
    assert "public.games" in drift.type_changes
    assert any(change[0] == "season" for change in drift.type_changes["public.games"])


# ============================================================================
# API Contract Validation Tests
# ============================================================================


def test_validate_api_contract_no_requests(agent):
    """Test API validation when requests library unavailable"""
    with patch("py.utils.frontend_engineering_agent.REQUESTS_AVAILABLE", False):
        report = agent.validate_api_contract()

    assert report.has_violations
    assert any("requests" in issue.message for issue in report.violations)


def test_validate_api_contract_api_unavailable(agent):
    """Test API validation when API is down"""
    with patch("requests.get", side_effect=Exception("Connection refused")):
        report = agent.validate_api_contract()

    assert report.has_violations
    assert not report.api_available
    assert any(issue.severity == Severity.CRITICAL for issue in report.violations)


def test_validate_api_contract_missing_endpoint(agent):
    """Test detection of missing API endpoints"""
    # Mock /health response
    mock_health_response = MagicMock()
    mock_health_response.json.return_value = {"status": "healthy", "version": "1.0.0"}

    # Mock OpenAPI schema WITHOUT /predict endpoint
    mock_schema_response = MagicMock()
    mock_schema_response.json.return_value = {
        "paths": {
            "/health": {"get": {}},
            "/models": {"get": {}},
            # Missing /predict and /metrics
        }
    }

    with patch("requests.get", side_effect=[mock_health_response, mock_schema_response]):
        report = agent.validate_api_contract()

    assert report.api_available
    assert report.has_violations
    assert any("/predict" in issue.message for issue in report.violations)


def test_validate_api_contract_success(agent):
    """Test successful API contract validation"""
    # Mock /health response
    mock_health_response = MagicMock()
    mock_health_response.json.return_value = {"status": "healthy", "version": "1.0.0"}

    # Mock complete OpenAPI schema
    mock_schema_response = MagicMock()
    mock_schema_response.json.return_value = {
        "paths": {
            "/health": {"get": {}},
            "/predict": {
                "post": {
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "properties": {
                                            "prediction": {"type": "number"},
                                            "confidence": {"type": "number"},
                                            "lower_bound": {"type": "number"},
                                            "upper_bound": {"type": "number"},
                                            "models_used": {"type": "array"},
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/models": {"get": {}},
            "/metrics": {"get": {}},
        }
    }

    with patch("requests.get", side_effect=[mock_health_response, mock_schema_response]):
        report = agent.validate_api_contract()

    assert report.api_available
    assert report.endpoints_checked > 0
    assert not report.has_violations


# ============================================================================
# Migration Update Tests
# ============================================================================


def test_suggest_frontend_updates_removed_columns(agent):
    """Test migration suggestions for removed columns"""
    drift = SchemaDrift()
    drift.removed_columns = {"mart.player_game_stats": ["game_id", "team"]}

    updates = agent.suggest_frontend_updates(drift)

    assert len(updates) > 0
    assert all(u.priority == Severity.CRITICAL for u in updates)
    assert any("game_id" in u.new_code for u in updates)


def test_suggest_frontend_updates_type_changes(agent):
    """Test migration suggestions for type changes"""
    drift = SchemaDrift()
    drift.type_changes = {"public.games": [("season", "integer", "bigint")]}

    updates = agent.suggest_frontend_updates(drift)

    assert len(updates) > 0
    assert any(u.priority == Severity.WARNING for u in updates)


def test_suggest_frontend_updates_added_tables(agent):
    """Test migration suggestions for added tables"""
    drift = SchemaDrift()
    drift.added_tables = ["public.new_feature_table"]

    updates = agent.suggest_frontend_updates(drift)

    assert len(updates) > 0
    assert all(u.priority == Severity.INFO for u in updates)


# ============================================================================
# Type Generation Tests
# ============================================================================


def test_generate_schema_validators(agent, sample_schema):
    """Test Pydantic validator generation"""
    agent._schema_cache = sample_schema

    validators_code = agent.generate_schema_validators()

    assert "from pydantic import BaseModel" in validators_code
    assert "class PublicGames(BaseModel)" in validators_code
    assert "game_id: str" in validators_code
    assert "season: int" in validators_code


def test_sql_type_to_python():
    """Test SQL to Python type mapping"""
    assert FrontendEngineeringAgent._sql_type_to_python("integer") == "int"
    assert FrontendEngineeringAgent._sql_type_to_python("bigint") == "int"
    assert FrontendEngineeringAgent._sql_type_to_python("text") == "str"
    assert FrontendEngineeringAgent._sql_type_to_python("boolean") == "bool"
    assert FrontendEngineeringAgent._sql_type_to_python("date") == "date"
    assert FrontendEngineeringAgent._sql_type_to_python("timestamp") == "datetime"
    assert FrontendEngineeringAgent._sql_type_to_python("numeric") == "float"
    assert FrontendEngineeringAgent._sql_type_to_python("unknown_type") == "str"


# ============================================================================
# Validation Issue Tests
# ============================================================================


def test_validation_issue_str():
    """Test ValidationIssue string representation"""
    issue = ValidationIssue(
        severity=Severity.ERROR,
        category="schema_validation",
        message="Column 'game_id' does not exist",
        file_path="/dashboard/app.py",
        line_number=42,
        suggestion="Use composite key (season, week) instead",
    )

    issue_str = str(issue)

    assert "[ERROR]" in issue_str
    assert "schema_validation" in issue_str
    assert "game_id" in issue_str
    assert "File: /dashboard/app.py" in issue_str
    assert "Line: 42" in issue_str
    assert "💡 Suggestion" in issue_str


# ============================================================================
# Integration Tests
# ============================================================================


def test_end_to_end_validation_workflow(agent, sample_schema, tmp_path):
    """Test complete validation workflow"""
    # 1. Save schema snapshot
    agent._schema_cache = sample_schema
    agent.save_schema_snapshot()

    # 2. Create dashboard with valid and invalid queries
    dashboard_path = tmp_path / "dashboard" / "app.py"
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_path.write_text(
        '''
# Valid query
valid_result = conn.execute("""
    SELECT g.game_id, g.season
    FROM public.games g
    WHERE g.season = 2024
""")

# INVALID query - game_id doesn't exist in player_game_stats
invalid_result = conn.execute("""
    SELECT pgs.game_id
    FROM mart.player_game_stats pgs
""")
'''
    )

    agent.dashboard_path = dashboard_path

    # 3. Validate dashboard
    report = agent.validate_dashboard_schema()

    assert report.queries_checked >= 2
    assert not report.is_valid  # Should fail due to invalid query
    assert any("game_id" in issue.message for issue in report.issues)

    # 4. Modify schema (remove a column)
    new_schema = sample_schema.copy()
    new_schema["tables"]["public.games"]["columns"].remove("kickoff")

    agent._schema_cache = new_schema

    # 5. Detect drift
    drift = agent.detect_schema_drift()

    assert drift.has_breaking_changes
    assert "public.games" in drift.removed_columns

    # 6. Generate migration updates
    updates = agent.suggest_frontend_updates(drift)

    assert len(updates) > 0
    assert any(u.priority == Severity.CRITICAL for u in updates)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
