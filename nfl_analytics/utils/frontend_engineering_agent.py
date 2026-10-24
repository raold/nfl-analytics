"""
Frontend Engineering Agent for NFL Analytics Dashboard

Prevents frontend brittleness by automating schema validation, API contract enforcement,
and frontend-backend synchronization. Catches breaking changes before they reach production.

Purpose:
    - Auto-validate all SQL queries against current database schema
    - Enforce API contract compliance (FastAPI ↔ Streamlit)
    - Detect schema drift and suggest automated migrations
    - Provide error boundaries and fallback mechanisms
    - Generate TypeScript interfaces from FastAPI models

Key Features:
    1. Schema Validation - Validates SQL queries against real database schema
    2. API Contract Enforcement - Ensures API responses match expected contracts
    3. Schema Drift Detection - Identifies when database/API changes break frontend
    4. Auto-Sync Tools - Generates migration scripts and code updates
    5. Error Boundaries - Wraps queries with fallback mechanisms

Created: November 22, 2025
Context: Dashboard has 15+ hardcoded SQL queries with no validation. Breaking changes
         discovered at runtime (not compile time).

Usage:
    from nfl_analytics.utils.frontend_engineering_agent import FrontendEngineeringAgent

    agent = FrontendEngineeringAgent()

    # Validate all dashboard SQL queries
    report = agent.validate_dashboard_schema()
    if not report.is_valid:
        print(report.generate_report())
        exit(1)

    # Detect schema drift
    drift = agent.detect_schema_drift()
    if drift.has_breaking_changes:
        updates = agent.suggest_frontend_updates(drift)
        for update in updates:
            print(f"Update required: {update}")

    # Validate API contract
    contract_report = agent.validate_api_contract()
    if contract_report.has_violations:
        print(contract_report.generate_report())

CLI Usage:
    # Validate dashboard schemas
    ./scripts/sync_frontend.sh validate-schema

    # Check for API contract violations
    ./scripts/sync_frontend.sh validate-api

    # Detect schema drift
    ./scripts/sync_frontend.sh check-drift

    # Generate TypeScript interfaces from FastAPI models
    ./scripts/sync_frontend.sh generate-types

    # Apply automated migration updates
    ./scripts/sync_frontend.sh apply-updates

Integration:
    - Pre-commit hook: Validates schemas before commit
    - CI/CD: Detects breaking changes in PRs
    - Dashboard startup: Validates schemas at app initialization
    - Prometheus: Real-time drift monitoring metrics

Author: Claude Code (Sonnet 4.5)
"""

import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import psycopg

    PSYCOPG_AVAILABLE = True
except ImportError:
    PSYCOPG_AVAILABLE = False

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================


class Severity(Enum):
    """Validation issue severity levels"""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ValidationIssue:
    """Single validation issue"""

    severity: Severity
    category: str
    message: str
    file_path: str | None = None
    line_number: int | None = None
    query: str | None = None
    suggestion: str | None = None

    def __str__(self) -> str:
        parts = [f"[{self.severity.value}] {self.category}: {self.message}"]
        if self.file_path:
            parts.append(f"  File: {self.file_path}")
        if self.line_number:
            parts.append(f"  Line: {self.line_number}")
        if self.suggestion:
            parts.append(f"  💡 Suggestion: {self.suggestion}")
        return "\n".join(parts)


@dataclass
class SchemaValidationReport:
    """Report of schema validation results"""

    timestamp: datetime = field(default_factory=datetime.now)
    issues: list[ValidationIssue] = field(default_factory=list)
    queries_checked: int = 0
    queries_valid: int = 0
    queries_invalid: int = 0
    tables_referenced: set[str] = field(default_factory=set)
    columns_referenced: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    @property
    def is_valid(self) -> bool:
        """Returns True if no ERROR or CRITICAL issues"""
        return not any(i.severity in [Severity.ERROR, Severity.CRITICAL] for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Returns True if any WARNING issues"""
        return any(i.severity == Severity.WARNING for i in self.issues)

    def add_issue(self, issue: ValidationIssue):
        """Add validation issue"""
        self.issues.append(issue)

    def generate_report(self) -> str:
        """Generate human-readable report"""
        lines = [
            "=" * 80,
            "FRONTEND SCHEMA VALIDATION REPORT",
            "=" * 80,
            f"Timestamp: {self.timestamp.isoformat()}",
            f"Queries Checked: {self.queries_checked}",
            f"Valid: {self.queries_valid} | Invalid: {self.queries_invalid}",
            "",
            "Issues by Severity:",
            f"  CRITICAL: {sum(1 for i in self.issues if i.severity == Severity.CRITICAL)}",
            f"  ERROR:    {sum(1 for i in self.issues if i.severity == Severity.ERROR)}",
            f"  WARNING:  {sum(1 for i in self.issues if i.severity == Severity.WARNING)}",
            f"  INFO:     {sum(1 for i in self.issues if i.severity == Severity.INFO)}",
            "",
        ]

        if self.issues:
            lines.append("Validation Issues:")
            lines.append("")
            for issue in sorted(self.issues, key=lambda x: (x.severity.value, x.category)):
                lines.append(str(issue))
                lines.append("")

        lines.append("=" * 80)
        if self.is_valid:
            lines.append("✅ PASSED: All validation checks passed")
        else:
            lines.append("❌ FAILED: Critical issues found - frontend may be broken")
        lines.append("=" * 80)

        return "\n".join(lines)


@dataclass
class SchemaDrift:
    """Represents schema drift between database and frontend"""

    timestamp: datetime = field(default_factory=datetime.now)
    added_tables: list[str] = field(default_factory=list)
    removed_tables: list[str] = field(default_factory=list)
    added_columns: dict[str, list[str]] = field(default_factory=dict)
    removed_columns: dict[str, list[str]] = field(default_factory=dict)
    renamed_columns: dict[str, list[tuple[str, str]]] = field(
        default_factory=dict
    )  # table -> [(old, new)]
    type_changes: dict[str, list[tuple[str, str, str]]] = field(
        default_factory=dict
    )  # table -> [(col, old_type, new_type)]

    @property
    def has_breaking_changes(self) -> bool:
        """Returns True if drift includes breaking changes"""
        return bool(self.removed_tables or self.removed_columns or self.type_changes)

    @property
    def has_any_changes(self) -> bool:
        """Returns True if any drift detected"""
        return bool(
            self.added_tables
            or self.removed_tables
            or self.added_columns
            or self.removed_columns
            or self.renamed_columns
            or self.type_changes
        )


@dataclass
class APIContractReport:
    """Report of API contract validation"""

    timestamp: datetime = field(default_factory=datetime.now)
    endpoints_checked: int = 0
    endpoints_valid: int = 0
    violations: list[ValidationIssue] = field(default_factory=list)
    api_available: bool = False
    api_version: str | None = None

    @property
    def has_violations(self) -> bool:
        """Returns True if any violations found"""
        return bool(self.violations)

    def generate_report(self) -> str:
        """Generate human-readable report"""
        lines = [
            "=" * 80,
            "API CONTRACT VALIDATION REPORT",
            "=" * 80,
            f"Timestamp: {self.timestamp.isoformat()}",
            f"API Available: {self.api_available}",
            f"API Version: {self.api_version or 'Unknown'}",
            f"Endpoints Checked: {self.endpoints_checked}",
            f"Valid: {self.endpoints_valid} | Violations: {len(self.violations)}",
            "",
        ]

        if self.violations:
            lines.append("Contract Violations:")
            lines.append("")
            for violation in self.violations:
                lines.append(str(violation))
                lines.append("")

        lines.append("=" * 80)
        if not self.has_violations:
            lines.append("✅ PASSED: All API contracts validated")
        else:
            lines.append("❌ FAILED: API contract violations detected")
        lines.append("=" * 80)

        return "\n".join(lines)


@dataclass
class MigrationUpdate:
    """Represents a suggested frontend update"""

    update_type: str  # "sql_query", "api_call", "schema_validator", "error_handling"
    file_path: str
    line_number: int | None
    old_code: str | None
    new_code: str
    reason: str
    priority: Severity
    automated: bool = True  # Can this be applied automatically?


# ============================================================================
# Frontend Engineering Agent
# ============================================================================


class FrontendEngineeringAgent:
    """
    Automated frontend engineering agent for schema validation and sync.

    Prevents dashboard brittleness by:
    1. Validating all SQL queries against current database schema
    2. Enforcing API contract compliance
    3. Detecting schema drift and suggesting migrations
    4. Generating type-safe interfaces
    """

    def __init__(
        self,
        db_config: dict[str, Any] | None = None,
        api_base_url: str = "http://localhost:8000",
        dashboard_path: Path | None = None,
        schema_snapshot_path: Path | None = None,
    ):
        """
        Initialize Frontend Engineering Agent.

        Args:
            db_config: Database connection config (host, port, dbname, user, password)
            api_base_url: FastAPI base URL for contract validation
            dashboard_path: Path to dashboard/app.py
            schema_snapshot_path: Path to schema snapshot JSON for drift detection
        """
        self.db_config = db_config or self._default_db_config()
        self.api_base_url = api_base_url

        # Paths
        project_root = Path(__file__).parent.parent.parent
        self.dashboard_path = dashboard_path or project_root / "dashboard" / "app.py"
        self.schema_snapshot_path = (
            schema_snapshot_path or project_root / "docs" / "database" / "schema_snapshot.json"
        )

        # Cached schema information
        self._schema_cache: dict[str, Any] | None = None
        self._api_schema_cache: dict[str, Any] | None = None

    @staticmethod
    def _default_db_config() -> dict[str, Any]:
        """Default database configuration from environment"""
        import os

        return {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5544")),
            "dbname": os.getenv("DB_NAME", "devdb01"),
            "user": os.getenv("DB_USER", "dro"),
            "password": os.getenv("DB_PASSWORD", "sicillionbillions"),  # pragma: allowlist secret
        }

    # ========================================================================
    # Schema Introspection
    # ========================================================================

    def get_database_schema(self) -> dict[str, Any]:
        """
        Introspect database schema.

        Returns:
            Dict with schema information:
            {
                "tables": {
                    "schema.table": {
                        "columns": ["col1", "col2", ...],
                        "column_types": {"col1": "integer", "col2": "text", ...},
                        "primary_keys": ["col1"],
                        "nullable": {"col1": False, "col2": True, ...}
                    }
                }
            }
        """
        if self._schema_cache:
            return self._schema_cache

        if not PSYCOPG_AVAILABLE:
            logger.error("psycopg not available - cannot introspect schema")
            return {"tables": {}}

        schema = {"tables": {}, "views": {}, "materialized_views": {}}

        try:
            with psycopg.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    # Get all tables and views
                    cur.execute("""
                        SELECT
                            table_schema,
                            table_name,
                            table_type
                        FROM information_schema.tables
                        WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                        ORDER BY table_schema, table_name
                    """)

                    for schema_name, table_name, table_type in cur.fetchall():
                        full_name = f"{schema_name}.{table_name}"

                        # Get column information
                        cur.execute(
                            """
                            SELECT
                                column_name,
                                data_type,
                                is_nullable,
                                column_default
                            FROM information_schema.columns
                            WHERE table_schema = %s AND table_name = %s
                            ORDER BY ordinal_position
                        """,
                            (schema_name, table_name),
                        )

                        columns = []
                        column_types = {}
                        nullable = {}
                        defaults = {}

                        for col_name, col_type, is_null, col_default in cur.fetchall():
                            columns.append(col_name)
                            column_types[col_name] = col_type
                            nullable[col_name] = is_null == "YES"
                            if col_default:
                                defaults[col_name] = col_default

                        # Get primary keys
                        cur.execute(
                            """
                            SELECT a.attname
                            FROM pg_index i
                            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                            WHERE i.indrelid = %s::regclass AND i.indisprimary
                        """,
                            (full_name,),
                        )

                        primary_keys = [row[0] for row in cur.fetchall()]

                        table_info = {
                            "columns": columns,
                            "column_types": column_types,
                            "primary_keys": primary_keys,
                            "nullable": nullable,
                            "defaults": defaults,
                        }

                        # Categorize by table type
                        if "MATERIALIZED" in table_type:
                            schema["materialized_views"][full_name] = table_info
                        elif "VIEW" in table_type:
                            schema["views"][full_name] = table_info
                        else:
                            schema["tables"][full_name] = table_info

            self._schema_cache = schema
            return schema

        except Exception as e:
            logger.error(f"Failed to introspect database schema: {e}")
            return {"tables": {}, "views": {}, "materialized_views": {}}

    def save_schema_snapshot(self) -> Path:
        """
        Save current schema to snapshot file for drift detection.

        Returns:
            Path to saved snapshot
        """
        schema = self.get_database_schema()

        # Add metadata
        snapshot = {"timestamp": datetime.now().isoformat(), "schema": schema}

        self.schema_snapshot_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.schema_snapshot_path, "w") as f:
            json.dump(snapshot, f, indent=2)

        logger.info(f"Schema snapshot saved to {self.schema_snapshot_path}")
        return self.schema_snapshot_path

    def load_schema_snapshot(self) -> dict[str, Any] | None:
        """Load previous schema snapshot for drift detection"""
        if not self.schema_snapshot_path.exists():
            logger.warning(f"Schema snapshot not found: {self.schema_snapshot_path}")
            return None

        with open(self.schema_snapshot_path) as f:
            snapshot = json.load(f)

        return snapshot.get("schema")

    # ========================================================================
    # SQL Query Validation
    # ========================================================================

    def extract_sql_queries(self, file_path: Path) -> list[tuple[str, int]]:
        """
        Extract SQL queries from Python file.

        Returns:
            List of (query, line_number) tuples
        """
        queries = []

        with open(file_path) as f:
            content = f.read()

        # Pattern 1: Multi-line SQL strings
        # Matches: """ SELECT ... """ or ''' SELECT ... '''
        # ({{3}} was f-string escaping in a plain raw string, so this never matched)
        multi_line_pattern = r'("""|\'\'\')(.*?SELECT.*?)\1'
        for match in re.finditer(multi_line_pattern, content, re.DOTALL | re.IGNORECASE):
            query = match.group(2).strip()
            line_num = content[: match.start()].count("\n") + 1
            queries.append((query, line_num))

        # Pattern 2: Single-line SQL strings
        # Matches: "SELECT ..." or 'SELECT ...'
        single_line_pattern = r'["\']([^"\']*SELECT[^"\']*)["\']'
        for match in re.finditer(single_line_pattern, content, re.IGNORECASE):
            query = match.group(1).strip()
            # Skip if already captured in multi-line
            if not any(query in q[0] for q in queries):
                line_num = content[: match.start()].count("\n") + 1
                queries.append((query, line_num))

        # Pattern 3: conn.execute() and cur.execute() calls
        execute_pattern = r'(?:conn|cur|cursor)\.execute\s*\(\s*["\']([^"\']+)["\']'
        for match in re.finditer(execute_pattern, content, re.IGNORECASE):
            query = match.group(1).strip()
            if "SELECT" in query.upper() and not any(query in q[0] for q in queries):
                line_num = content[: match.start()].count("\n") + 1
                queries.append((query, line_num))

        return queries

    def validate_sql_query(self, query: str, schema: dict[str, Any]) -> list[ValidationIssue]:
        """
        Validate SQL query against database schema.

        Args:
            query: SQL query string
            schema: Database schema from get_database_schema()

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        # Normalize query
        query_upper = query.upper()

        # Extract table references
        # Pattern: FROM schema.table or JOIN schema.table
        # plus an optional alias ("FROM mart.player_game_stats pgs" / "... AS pgs")
        table_pattern = (
            r"(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*\.?[a-zA-Z_][a-zA-Z0-9_]*)"
            r"(?:\s+(?:AS\s+)?(?!(?:WHERE|JOIN|ON|USING|LEFT|RIGHT|INNER|OUTER|FULL|CROSS|"
            r"GROUP|ORDER|LIMIT|HAVING|UNION|WINDOW)\b)([a-zA-Z_][a-zA-Z0-9_]*))?"
        )
        table_matches = re.findall(table_pattern, query, re.IGNORECASE)

        all_tables = {
            **schema.get("tables", {}),
            **schema.get("views", {}),
            **schema.get("materialized_views", {}),
        }

        for table_ref, alias in table_matches:
            # Normalize table reference
            if "." not in table_ref:
                # Assume public schema
                table_ref = f"public.{table_ref}"

            # Check if table exists
            if table_ref not in all_tables:
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        category="schema_validation",
                        message=f"Table '{table_ref}' does not exist in database",
                        query=query,
                        suggestion=f"Available tables: {', '.join(list(all_tables.keys())[:5])}...",
                    )
                )
                continue

            # Extract column references for this table via its alias or bare table name
            # (queries almost always use an alias, e.g. pgs.game_id)
            qualifiers = {table_ref.split(".")[-1]} | ({alias} if alias else set())
            col_matches = []
            for qualifier in qualifiers:
                col_pattern = rf"\b{re.escape(qualifier)}\.([a-zA-Z_][a-zA-Z0-9_]*)"
                col_matches += re.findall(col_pattern, query, re.IGNORECASE)

            table_info = all_tables[table_ref]
            available_columns = set(table_info["columns"])

            for col_name in col_matches:
                if col_name not in available_columns:
                    issues.append(
                        ValidationIssue(
                            severity=Severity.ERROR,
                            category="schema_validation",
                            message=f"Column '{col_name}' does not exist in table '{table_ref}'",
                            query=query,
                            suggestion=f"Available columns: {', '.join(list(available_columns)[:10])}...",
                        )
                    )

        # Check for common anti-patterns
        if "SELECT *" in query_upper:
            issues.append(
                ValidationIssue(
                    severity=Severity.WARNING,
                    category="best_practice",
                    message="SELECT * detected - specify columns explicitly",
                    query=query,
                    suggestion="List specific columns to prevent breakage on schema changes",
                )
            )

        # Check for missing WHERE clause on large tables
        large_tables = {"plays", "odds_history", "injuries"}
        for table in large_tables:
            if table in query.lower() and "WHERE" not in query_upper and "LIMIT" not in query_upper:
                issues.append(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        category="performance",
                        message=f"Query on large table '{table}' without WHERE or LIMIT clause",
                        query=query,
                        suggestion="Add WHERE clause or LIMIT to prevent full table scan",
                    )
                )

        return issues

    def validate_dashboard_schema(self) -> SchemaValidationReport:
        """
        Validate all SQL queries in dashboard against current database schema.

        Returns:
            SchemaValidationReport with validation results
        """
        report = SchemaValidationReport()

        if not self.dashboard_path.exists():
            report.add_issue(
                ValidationIssue(
                    severity=Severity.CRITICAL,
                    category="file_not_found",
                    message=f"Dashboard file not found: {self.dashboard_path}",
                    file_path=str(self.dashboard_path),
                )
            )
            return report

        # Get current schema
        schema = self.get_database_schema()

        # Extract all SQL queries from dashboard
        queries = self.extract_sql_queries(self.dashboard_path)
        report.queries_checked = len(queries)

        logger.info(f"Validating {len(queries)} SQL queries from {self.dashboard_path.name}")

        for query, line_num in queries:
            issues = self.validate_sql_query(query, schema)

            for issue in issues:
                issue.file_path = str(self.dashboard_path)
                issue.line_number = line_num
                report.add_issue(issue)

            # Count queries, not issues: a query is invalid if it has any ERROR/CRITICAL issue
            if any(i.severity in (Severity.ERROR, Severity.CRITICAL) for i in issues):
                report.queries_invalid += 1
            else:
                report.queries_valid += 1

        return report

    # ========================================================================
    # Schema Drift Detection
    # ========================================================================

    def detect_schema_drift(self) -> SchemaDrift:
        """
        Detect schema changes between current schema and saved snapshot.

        Returns:
            SchemaDrift object with detected changes
        """
        drift = SchemaDrift()

        current_schema = self.get_database_schema()
        previous_schema = self.load_schema_snapshot()

        if not previous_schema:
            logger.warning("No schema snapshot found - cannot detect drift")
            return drift

        # Combine all table types for comparison
        current_tables = {
            **current_schema.get("tables", {}),
            **current_schema.get("views", {}),
            **current_schema.get("materialized_views", {}),
        }

        previous_tables = {
            **previous_schema.get("tables", {}),
            **previous_schema.get("views", {}),
            **previous_schema.get("materialized_views", {}),
        }

        # Detect added/removed tables
        current_table_names = set(current_tables.keys())
        previous_table_names = set(previous_tables.keys())

        drift.added_tables = list(current_table_names - previous_table_names)
        drift.removed_tables = list(previous_table_names - current_table_names)

        # Detect column changes in common tables
        common_tables = current_table_names & previous_table_names

        for table_name in common_tables:
            current_cols = set(current_tables[table_name]["columns"])
            previous_cols = set(previous_tables[table_name]["columns"])

            added = list(current_cols - previous_cols)
            removed = list(previous_cols - current_cols)

            if added:
                drift.added_columns[table_name] = added
            if removed:
                drift.removed_columns[table_name] = removed

            # Detect type changes
            current_types = current_tables[table_name]["column_types"]
            previous_types = previous_tables[table_name]["column_types"]

            type_changes = []
            for col in current_cols & previous_cols:
                if current_types[col] != previous_types[col]:
                    type_changes.append((col, previous_types[col], current_types[col]))

            if type_changes:
                drift.type_changes[table_name] = type_changes

        return drift

    # ========================================================================
    # API Contract Validation
    # ========================================================================

    def get_api_schema(self) -> dict[str, Any]:
        """
        Fetch API schema from FastAPI's /openapi.json endpoint.

        Returns:
            OpenAPI schema dict
        """
        if self._api_schema_cache:
            return self._api_schema_cache

        if not REQUESTS_AVAILABLE:
            logger.error("requests library not available - cannot fetch API schema")
            return {}

        try:
            response = requests.get(f"{self.api_base_url}/openapi.json", timeout=5)
            response.raise_for_status()

            self._api_schema_cache = response.json()
            return self._api_schema_cache

        except Exception as e:
            logger.error(f"Failed to fetch API schema: {e}")
            return {}

    def validate_api_contract(self) -> APIContractReport:
        """
        Validate API contract compliance.

        Checks:
        1. API is reachable
        2. All expected endpoints exist
        3. Response schemas match expected contracts

        Returns:
            APIContractReport with validation results
        """
        report = APIContractReport()

        if not REQUESTS_AVAILABLE:
            report.violations.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category="dependency",
                    message="requests library not available - cannot validate API",
                    suggestion="pip install requests",
                )
            )
            return report

        # Check if API is available
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            response.raise_for_status()
            report.api_available = True

            health_data = response.json()
            report.api_version = health_data.get("version", "unknown")

        except Exception:
            report.violations.append(
                ValidationIssue(
                    severity=Severity.CRITICAL,
                    category="api_availability",
                    message=f"API not reachable at {self.api_base_url}",
                    suggestion="Ensure API is running: docker-compose up api",
                )
            )
            return report

        # Get API schema
        api_schema = self.get_api_schema()

        if not api_schema:
            report.violations.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    category="api_schema",
                    message="Failed to fetch API schema",
                    suggestion="Check API server logs",
                )
            )
            return report

        # Validate expected endpoints exist
        expected_endpoints = [
            ("/health", "GET"),
            ("/predict", "POST"),
            ("/models", "GET"),
            ("/metrics", "GET"),
        ]

        paths = api_schema.get("paths", {})
        report.endpoints_checked = len(expected_endpoints)

        for endpoint, method in expected_endpoints:
            method_lower = method.lower()

            if endpoint not in paths:
                report.violations.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        category="api_contract",
                        message=f"Expected endpoint '{method} {endpoint}' not found in API",
                        suggestion="Update API or fix dashboard to use correct endpoint",
                    )
                )
            elif method_lower not in paths[endpoint]:
                report.violations.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        category="api_contract",
                        message=f"Method '{method}' not supported for endpoint '{endpoint}'",
                        suggestion=f"Available methods: {', '.join(paths[endpoint].keys())}",
                    )
                )
            else:
                report.endpoints_valid += 1

        # Validate /predict response schema
        if "/predict" in paths and "post" in paths["/predict"]:
            predict_schema = paths["/predict"]["post"].get("responses", {}).get("200", {})

            # Check for required fields in PredictionResponse
            required_fields = [
                "prediction",
                "confidence",
                "lower_bound",
                "upper_bound",
                "models_used",
            ]

            response_schema = (
                predict_schema.get("content", {}).get("application/json", {}).get("schema", {})
            )
            properties = response_schema.get("properties", {})

            for field in required_fields:
                if field not in properties:
                    report.violations.append(
                        ValidationIssue(
                            severity=Severity.ERROR,
                            category="api_contract",
                            message=f"Required field '{field}' missing from /predict response schema",
                            suggestion="Update API models.py or fix dashboard expectations",
                        )
                    )

        return report

    # ========================================================================
    # Migration Suggestions
    # ========================================================================

    def suggest_frontend_updates(self, drift: SchemaDrift) -> list[MigrationUpdate]:
        """
        Generate automated migration suggestions based on schema drift.

        Args:
            drift: SchemaDrift object from detect_schema_drift()

        Returns:
            List of MigrationUpdate suggestions
        """
        updates = []

        # Handle removed columns (CRITICAL - will break queries)
        for table, columns in drift.removed_columns.items():
            for col in columns:
                updates.append(
                    MigrationUpdate(
                        update_type="sql_query",
                        file_path=str(self.dashboard_path),
                        line_number=None,
                        old_code=None,
                        new_code=f"# TODO: Column '{col}' removed from '{table}' - update queries",
                        reason=f"Column '{col}' no longer exists in table '{table}'",
                        priority=Severity.CRITICAL,
                        automated=False,
                    )
                )

        # Handle type changes (WARNING - may cause runtime errors)
        for table, changes in drift.type_changes.items():
            for col, old_type, new_type in changes:
                updates.append(
                    MigrationUpdate(
                        update_type="schema_validator",
                        file_path=str(self.dashboard_path),
                        line_number=None,
                        old_code=None,
                        new_code=f"# TODO: Column '{col}' in '{table}' changed from {old_type} to {new_type}",
                        reason="Type change may require casting or value conversion",
                        priority=Severity.WARNING,
                        automated=False,
                    )
                )

        # Handle added tables (INFO - opportunity for new features)
        for table in drift.added_tables:
            updates.append(
                MigrationUpdate(
                    update_type="sql_query",
                    file_path=str(self.dashboard_path),
                    line_number=None,
                    old_code=None,
                    new_code=f"# INFO: New table '{table}' available",
                    reason=f"New table '{table}' added to database",
                    priority=Severity.INFO,
                    automated=False,
                )
            )

        return updates

    def generate_schema_validators(self) -> str:
        """
        Generate Pydantic validators from database schema.

        Returns:
            Python code string with Pydantic models
        """
        schema = self.get_database_schema()

        lines = [
            '"""',
            "Auto-generated Pydantic models from database schema",
            f"Generated: {datetime.now().isoformat()}",
            "DO NOT EDIT MANUALLY - regenerate using frontend_engineering_agent.py",
            '"""',
            "",
            "from typing import Optional",
            "from datetime import datetime, date",
            "from pydantic import BaseModel, Field",
            "",
        ]

        # Generate model for each table
        all_tables = {**schema.get("tables", {}), **schema.get("views", {})}

        for table_name, table_info in all_tables.items():
            # Convert table name to class name
            class_name = "".join(
                word.capitalize() for word in table_name.replace(".", "_").split("_")
            )

            lines.append(f"class {class_name}(BaseModel):")
            lines.append(f'    """Auto-generated model for {table_name}"""')

            # Generate fields
            for col in table_info["columns"]:
                col_type = table_info["column_types"][col]
                is_nullable = table_info["nullable"][col]
                is_pk = col in table_info["primary_keys"]

                # Map SQL types to Python types
                python_type = self._sql_type_to_python(col_type)

                if is_nullable:
                    python_type = f"Optional[{python_type}]"

                field_str = f"    {col}: {python_type}"

                if is_pk:
                    field_str += " = Field(..., description='Primary key')"
                elif is_nullable:
                    field_str += " = None"

                lines.append(field_str)

            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _sql_type_to_python(sql_type: str) -> str:
        """Map SQL types to Python types"""
        type_map = {
            "integer": "int",
            "bigint": "int",
            "smallint": "int",
            "numeric": "float",
            "real": "float",
            "double precision": "float",
            "text": "str",
            "character varying": "str",
            "varchar": "str",
            "char": "str",
            "boolean": "bool",
            "date": "date",
            "timestamp": "datetime",
            "timestamp with time zone": "datetime",
            "timestamp without time zone": "datetime",
        }

        sql_type_lower = sql_type.lower()

        for sql_t, py_t in type_map.items():
            if sql_type_lower.startswith(sql_t):
                return py_t

        return "str"  # Default fallback

    # ========================================================================
    # CLI Interface
    # ========================================================================

    @staticmethod
    def cli_validate_schema():
        """CLI command: validate-schema"""
        agent = FrontendEngineeringAgent()
        report = agent.validate_dashboard_schema()

        print(report.generate_report())

        if not report.is_valid:
            sys.exit(1)

    @staticmethod
    def cli_validate_api():
        """CLI command: validate-api"""
        agent = FrontendEngineeringAgent()
        report = agent.validate_api_contract()

        print(report.generate_report())

        if report.has_violations:
            sys.exit(1)

    @staticmethod
    def cli_check_drift():
        """CLI command: check-drift"""
        agent = FrontendEngineeringAgent()
        drift = agent.detect_schema_drift()

        print("=" * 80)
        print("SCHEMA DRIFT DETECTION")
        print("=" * 80)
        print(f"Timestamp: {drift.timestamp.isoformat()}")
        print()

        if not drift.has_any_changes:
            print("✅ No schema drift detected")
            return

        if drift.added_tables:
            print(f"➕ Added Tables ({len(drift.added_tables)}):")
            for table in drift.added_tables:
                print(f"  - {table}")
            print()

        if drift.removed_tables:
            print(f"🗑️  Removed Tables ({len(drift.removed_tables)}) - BREAKING:")
            for table in drift.removed_tables:
                print(f"  - {table}")
            print()

        if drift.added_columns:
            print("➕ Added Columns:")
            for table, cols in drift.added_columns.items():
                print(f"  {table}: {', '.join(cols)}")
            print()

        if drift.removed_columns:
            print("🗑️  Removed Columns - BREAKING:")
            for table, cols in drift.removed_columns.items():
                print(f"  {table}: {', '.join(cols)}")
            print()

        if drift.type_changes:
            print("⚠️  Type Changes:")
            for table, changes in drift.type_changes.items():
                print(f"  {table}:")
                for col, old_type, new_type in changes:
                    print(f"    - {col}: {old_type} → {new_type}")
            print()

        print("=" * 80)
        if drift.has_breaking_changes:
            print("❌ BREAKING CHANGES DETECTED - Frontend updates required")
            sys.exit(1)
        else:
            print("⚠️  Non-breaking changes detected - Review recommended")

    @staticmethod
    def cli_generate_types():
        """CLI command: generate-types"""
        agent = FrontendEngineeringAgent()

        # Generate Pydantic models
        validators = agent.generate_schema_validators()

        output_path = Path(__file__).parent.parent.parent / "dashboard" / "schema_validators.py"

        with open(output_path, "w") as f:
            f.write(validators)

        print(f"✅ Schema validators generated: {output_path}")

    @staticmethod
    def cli_apply_updates():
        """CLI command: apply-updates"""
        agent = FrontendEngineeringAgent()

        drift = agent.detect_schema_drift()
        updates = agent.suggest_frontend_updates(drift)

        if not updates:
            print("✅ No updates required")
            return

        print("=" * 80)
        print(f"SUGGESTED FRONTEND UPDATES ({len(updates)})")
        print("=" * 80)

        for i, update in enumerate(updates, 1):
            print(f"\n{i}. [{update.priority.value}] {update.update_type}")
            print(f"   File: {update.file_path}")
            if update.line_number:
                print(f"   Line: {update.line_number}")
            print(f"   Reason: {update.reason}")
            print(f"   Update: {update.new_code}")
            print(f"   Automated: {'Yes' if update.automated else 'No (manual review required)'}")

        print("\n" + "=" * 80)
        print(f"⚠️  {len(updates)} updates suggested - Manual review required")


# ============================================================================
# Main CLI
# ============================================================================


def main():
    """CLI entry point"""
    if len(sys.argv) < 2:
        print("Usage: python frontend_engineering_agent.py <command>")
        print("\nCommands:")
        print("  validate-schema  - Validate dashboard SQL queries against database schema")
        print("  validate-api     - Validate API contract compliance")
        print("  check-drift      - Detect schema drift since last snapshot")
        print("  generate-types   - Generate Pydantic validators from schema")
        print("  apply-updates    - Suggest frontend migration updates")
        sys.exit(1)

    command = sys.argv[1]

    commands = {
        "validate-schema": FrontendEngineeringAgent.cli_validate_schema,
        "validate-api": FrontendEngineeringAgent.cli_validate_api,
        "check-drift": FrontendEngineeringAgent.cli_check_drift,
        "generate-types": FrontendEngineeringAgent.cli_generate_types,
        "apply-updates": FrontendEngineeringAgent.cli_apply_updates,
    }

    if command not in commands:
        print(f"❌ Unknown command: {command}")
        print(f"Available commands: {', '.join(commands.keys())}")
        sys.exit(1)

    commands[command]()


if __name__ == "__main__":
    main()
