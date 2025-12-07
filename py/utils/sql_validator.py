#!/usr/bin/env python3
"""
SQL Query Validator for NFL Analytics Database Schema

Purpose:
    Validate SQL queries against actual database schema to catch common errors:
    - References to non-existent columns (e.g., pgs.game_id)
    - Wrong column names (e.g., game_date vs kickoff)
    - Missing indexes on frequently-joined columns
    - Incorrect join patterns

Usage:
    from py.utils.sql_validator import validate_query, SQLValidator

    # Quick validation
    errors = validate_query(
        query="SELECT pgs.game_id FROM mart.player_game_stats pgs",
        db_config={'host': 'localhost', 'port': 5544, 'database': 'devdb01',
                   'user': 'dro', 'password': 'sicillionbillions'}
    )
    if errors:
        print("\\n".join(errors))

    # Full validation with recommendations
    validator = SQLValidator(db_config)
    report = validator.validate_query_full(query)
    print(report['summary'])

Author: Claude Code (Sonnet 4.5)
Created: November 14, 2025
"""

import re

import psycopg2
from psycopg2.extensions import connection as Connection

# Known schema conventions (prevent common mistakes)
COLUMN_NAMING_CONVENTIONS = {
    'games': {
        'game_date': 'kickoff',
        'gameday': 'kickoff',
        'spread_line': 'spread_close',
        'total_line': 'total_close',
    },
    'mart.player_game_stats': {
        'game_id': '(season, week)',  # Composite key, not a column
    }
}

# Tables that definitely do NOT have game_id
NO_GAME_ID_TABLES = {
    'mart.player_game_stats',
    'mart.player_hierarchy',
}

# Team abbreviation corrections
TEAM_ABBREV_CORRECTIONS = {
    'LA': 'Use LAR (Rams) or LAC (Chargers) - LA is ambiguous'
}


class SQLValidator:
    """Validates SQL queries against NFL Analytics database schema."""

    def __init__(self, db_config: dict[str, str] | None = None):
        """
        Initialize SQL validator.

        Args:
            db_config: Database connection config (host, port, database, user, password)
                      If None, uses default localhost:5544/devdb01
        """
        self.db_config = db_config or {
            'host': 'localhost',
            'port': 5544,
            'database': 'devdb01',
            'user': 'dro',
            'password': 'sicillionbillions'
        }
        self._conn: Connection | None = None
        self._schema_cache: dict[str, set[str]] = {}

    def _get_connection(self) -> Connection:
        """Get or create database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(**self.db_config)
        return self._conn

    def close(self):
        """Close database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def get_table_columns(self, table_name: str, schema: str = 'public') -> set[str]:
        """
        Get all column names for a table or materialized view.

        Args:
            table_name: Name of the table/view
            schema: Schema name (default: 'public')

        Returns:
            Set of column names (lowercase)

        Example:
            >>> validator = SQLValidator()
            >>> columns = validator.get_table_columns('games')
            >>> 'game_id' in columns
            True
            >>> 'game_date' in columns  # Wrong name
            False
        """
        cache_key = f"{schema}.{table_name}"

        if cache_key in self._schema_cache:
            return self._schema_cache[cache_key]

        conn = self._get_connection()
        cursor = conn.cursor()

        # Use pg_attribute instead of information_schema.columns
        # This works for tables, views, and materialized views
        query = """
            SELECT a.attname
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = %s
                AND c.relname = %s
                AND a.attnum > 0
                AND NOT a.attisdropped;
        """

        cursor.execute(query, (schema, table_name))
        columns = {row[0].lower() for row in cursor.fetchall()}
        cursor.close()

        self._schema_cache[cache_key] = columns
        return columns

    def extract_table_aliases(self, query: str) -> dict[str, str]:
        """
        Extract table aliases from SQL query.

        Args:
            query: SQL query string

        Returns:
            Dict mapping alias to table name (e.g., {'pgs': 'mart.player_game_stats'})

        Example:
            >>> query = "SELECT pgs.* FROM mart.player_game_stats pgs"
            >>> extract_table_aliases(query)
            {'pgs': 'mart.player_game_stats'}
        """
        aliases = {}

        # Pattern: FROM table_name alias
        from_pattern = r'\bFROM\s+([a-z_]+\.[a-z_]+|[a-z_]+)\s+([a-z_]+)\b'
        for match in re.finditer(from_pattern, query, re.IGNORECASE):
            table, alias = match.groups()
            aliases[alias.lower()] = table.lower()

        # Pattern: JOIN table_name alias
        join_pattern = r'\bJOIN\s+([a-z_]+\.[a-z_]+|[a-z_]+)\s+([a-z_]+)\b'
        for match in re.finditer(join_pattern, query, re.IGNORECASE):
            table, alias = match.groups()
            aliases[alias.lower()] = table.lower()

        return aliases

    def extract_column_references(self, query: str) -> list[tuple[str, str]]:
        """
        Extract column references from SQL query.

        Args:
            query: SQL query string

        Returns:
            List of (table_alias, column_name) tuples

        Example:
            >>> query = "SELECT pgs.game_id, g.spread_close FROM ..."
            >>> extract_column_references(query)
            [('pgs', 'game_id'), ('g', 'spread_close')]
        """
        references = []

        # Pattern: alias.column_name
        pattern = r'\b([a-z_]+)\.([a-z_][a-z0-9_]*)\b'

        for match in re.finditer(pattern, query, re.IGNORECASE):
            alias, column = match.groups()
            references.append((alias.lower(), column.lower()))

        return references

    def check_column_exists(
        self, table_name: str, column_name: str, schema: str = 'public'
    ) -> bool:
        """
        Check if a column exists in a table.

        Args:
            table_name: Name of the table
            column_name: Name of the column
            schema: Schema name (default: 'public')

        Returns:
            True if column exists, False otherwise
        """
        columns = self.get_table_columns(table_name, schema)
        return column_name.lower() in columns

    def validate_query(self, query: str) -> list[str]:
        """
        Validate SQL query against database schema.

        Args:
            query: SQL query string

        Returns:
            List of error messages (empty if valid)

        Example:
            >>> validator = SQLValidator()
            >>> errors = validator.validate_query(
            ...     "SELECT pgs.game_id FROM mart.player_game_stats pgs"
            ... )
            >>> print(errors[0])
            ERROR: Column 'game_id' does not exist in 'mart.player_game_stats'
        """
        errors = []

        # Extract table aliases
        aliases = self.extract_table_aliases(query)

        # Extract column references
        references = self.extract_column_references(query)

        # Check each column reference
        for alias, column in references:
            if alias not in aliases:
                # Alias not found (may be using full table name)
                continue

            table_name = aliases[alias]

            # Parse schema.table
            if '.' in table_name:
                schema, table = table_name.split('.')
            else:
                schema = 'public'
                table = table_name

            # Check if column exists
            if not self.check_column_exists(table, column, schema):
                full_table = f"{schema}.{table}" if schema != 'public' else table

                # Check for known naming conventions
                if full_table in COLUMN_NAMING_CONVENTIONS:
                    convention = COLUMN_NAMING_CONVENTIONS[full_table]
                    if column in convention:
                        correct_name = convention[column]
                        errors.append(
                            f"ERROR: Column '{column}' does not exist in '{full_table}'. "
                            f"Did you mean '{correct_name}'?"
                        )
                        continue

                # Check for game_id mistake
                if column == 'game_id' and full_table in NO_GAME_ID_TABLES:
                    errors.append(
                        f"ERROR: Column 'game_id' does not exist in '{full_table}'. "
                        f"Join via (season, week) composite key. "
                        f"See docs/database/SQL_PATTERNS.md"
                    )
                    continue

                errors.append(
                    f"ERROR: Column '{column}' does not exist in '{full_table}'"
                )

        # Check for team abbreviation issues
        if re.search(r"['\"]LA['\"]", query):
            errors.append(
                "WARNING: Team abbreviation 'LA' is ambiguous. "
                "Use 'LAR' (Rams) or 'LAC' (Chargers)"
            )

        # Check for common anti-patterns
        if 'SELECT *' in query.upper() and 'COUNT(*)' not in query.upper():
            errors.append(
                "WARNING: Using 'SELECT *' in production queries. "
                "Specify columns explicitly for better performance."
            )

        return errors

    def validate_query_full(self, query: str) -> dict:
        """
        Full validation with detailed report.

        Args:
            query: SQL query string

        Returns:
            Dict with keys:
                - valid (bool): True if no errors
                - errors (List[str]): Error messages
                - warnings (List[str]): Warning messages
                - recommendations (List[str]): Optimization recommendations
                - summary (str): Human-readable summary

        Example:
            >>> validator = SQLValidator()
            >>> report = validator.validate_query_full(query)
            >>> if not report['valid']:
            ...     print(report['summary'])
        """
        all_messages = self.validate_query(query)

        errors = [msg for msg in all_messages if msg.startswith('ERROR:')]
        warnings = [msg for msg in all_messages if msg.startswith('WARNING:')]
        recommendations = []

        # Check for recommended indexes
        if 'mart.player_game_stats' in query.lower():
            if 'where' in query.lower() and 'stat_category' not in query.lower():
                recommendations.append(
                    "RECOMMENDATION: Add 'WHERE stat_category = ...' to use partial index "
                    "idx_player_game_stats_lookup"
                )

        # Check for join optimization
        if 'join games' in query.lower():
            if 'season' not in query.lower() or 'week' not in query.lower():
                recommendations.append(
                    "RECOMMENDATION: Join games via (season, week) composite key "
                    "for optimal performance"
                )

        # Build summary
        if errors:
            summary = f"❌ VALIDATION FAILED: {len(errors)} error(s)\n"
            summary += "\n".join(errors)
        elif warnings:
            summary = f"⚠️  VALIDATION PASSED WITH WARNINGS: {len(warnings)} warning(s)\n"
            summary += "\n".join(warnings)
        else:
            summary = "✅ VALIDATION PASSED"

        if recommendations:
            summary += "\n\n" + "\n".join(recommendations)

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'recommendations': recommendations,
            'summary': summary
        }


def validate_query(query: str, db_config: dict[str, str] | None = None) -> list[str]:
    """
    Quick validation function (convenience wrapper).

    Args:
        query: SQL query string
        db_config: Database connection config (optional)

    Returns:
        List of error messages (empty if valid)

    Example:
        >>> errors = validate_query(
        ...     "SELECT pgs.game_id FROM mart.player_game_stats pgs",
        ...     db_config={'host': 'localhost', 'port': 5544, ...}
        ... )
        >>> if errors:
        ...     print("\\n".join(errors))
    """
    with SQLValidator(db_config) as validator:
        return validator.validate_query(query)


# CLI interface
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python sql_validator.py <query>")
        print("\nExample:")
        print('  python sql_validator.py "SELECT pgs.game_id FROM mart.player_game_stats pgs"')
        sys.exit(1)

    query = sys.argv[1]

    print("Validating query:")
    print(query)
    print()

    with SQLValidator() as validator:
        report = validator.validate_query_full(query)
        print(report['summary'])

    sys.exit(0 if report['valid'] else 1)
