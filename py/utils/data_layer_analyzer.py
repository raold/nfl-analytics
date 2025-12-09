#!/usr/bin/env python3
"""
Data Layer Analyzer - Column Lineage and Impact Analysis

Tracks column usage across Python, R, and SQL files to:
- Build column lineage (which code uses which columns)
- Analyze impact of schema changes (add/modify/remove)
- Detect unused columns (dead code)
- Identify feature opportunities (new columns)
- Validate migrations before deployment

Critical for real-money betting systems where silent failures = lost bets.

Author: Claude Code
Date: November 13, 2025
"""

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ColumnUsage:
    """Represents a single usage of a database column in code."""

    table_name: str
    column_name: str
    file_path: str
    line_number: int
    usage_type: str  # 'select', 'insert', 'update', 'delete', 'filter', 'groupby', 'orderby'
    context: str  # Code snippet showing usage
    consumer_type: str  # 'feature', 'model', 'backtest', 'validation', 'production', 'migration'


@dataclass
class ImpactReport:
    """Impact analysis report for schema changes."""

    column: str
    change_type: str  # 'add', 'modify', 'remove'
    breaking_changes: list[ColumnUsage] = field(default_factory=list)
    affected_models: set[str] = field(default_factory=set)
    affected_features: set[str] = field(default_factory=set)
    affected_backtests: set[str] = field(default_factory=set)
    opportunities: list[str] = field(default_factory=list)
    estimated_impact_hours: float = 0.0
    risk_level: str = "low"  # 'low', 'medium', 'high', 'critical'

    def has_breaking_changes(self) -> bool:
        """Check if change would break existing code."""
        return len(self.breaking_changes) > 0

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Impact Analysis: {self.column} ({self.change_type})",
            f"Risk Level: {self.risk_level.upper()}",
            f"Breaking Changes: {len(self.breaking_changes)}",
            f"Affected Models: {len(self.affected_models)}",
            f"Affected Features: {len(self.affected_features)}",
            f"Affected Backtests: {len(self.affected_backtests)}",
        ]

        if self.change_type == "add":
            lines.append(f"Opportunities: {len(self.opportunities)}")

        if self.estimated_impact_hours > 0:
            lines.append(f"Estimated Integration Effort: {self.estimated_impact_hours:.1f} hours")

        return "\n".join(lines)


class DataLayerAnalyzer:
    """
    Analyzes data layer dependencies across Python, R, and SQL code.

    Tracks which columns are used where, enabling:
    - Impact analysis before schema changes
    - Dead column detection
    - Feature opportunity identification
    - Dependency graph visualization
    """

    def __init__(self, project_root: str = "/Users/dro/rice/nfl-analytics"):
        self.project_root = Path(project_root)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Column lineage: {table.column: [usages]}
        self.lineage: dict[str, list[ColumnUsage]] = defaultdict(list)

        # File patterns to scan
        self.python_dirs = ["py/", "scripts/", "tests/", "etl/"]
        self.r_dirs = ["R/", "scripts/analysis/"]
        self.sql_dirs = ["db/migrations/", "db/queries/", "db/views/"]

        # SQL patterns for different languages
        self.python_sql_patterns = [
            r'read_sql\([\'"](.+?)[\'"]\s*,',  # pd.read_sql("SELECT ...", conn)
            r'execute\([\'"](.+?)[\'"]\)',  # cursor.execute("SELECT ...")
            r'dbGetQuery\(.*?[\'"](.+?)[\'"]\)',  # R-style in Python (rare)
            r'"""(.+?)"""',  # Triple-quoted SQL
            r"'''(.+?)'''",  # Triple-quoted SQL
        ]

        self.r_sql_patterns = [
            r'dbGetQuery\(.+?,\s*[\'"](.+?)[\'"]\)',  # dbGetQuery(con, "SELECT ...")
            r'dbReadTable\(.+?,\s*[\'"](.+?)[\'"]\)',  # dbReadTable(con, "table_name")
            r'dbSendQuery\(.+?,\s*[\'"](.+?)[\'"]\)',
            r'"(.+?)"',  # Quoted SQL strings in R
        ]

        # Column reference patterns
        self.column_patterns = [
            r"\b(\w+)\.(\w+)\b",  # table.column
            r"SELECT\s+(.+?)\s+FROM",  # SELECT columns
            r"WHERE\s+(.+?)(?:ORDER|GROUP|LIMIT|$)",  # WHERE clause
            r"GROUP\s+BY\s+(.+?)(?:ORDER|HAVING|LIMIT|$)",  # GROUP BY
            r"ORDER\s+BY\s+(.+?)(?:LIMIT|$)",  # ORDER BY
        ]

        # Known table names (will be populated from database or hardcoded)
        self.known_tables = {
            # Core tables
            "games",
            "plays",
            "players",
            "rosters",
            "teams",
            "injuries",
            "schedules",
            "weather",
            "betting_lines",
            # Mart tables
            "game_summary",
            "team_rolling_stats",
            "team_matchup_history",
            "player_game_stats",
            "player_season_stats",
            # ML tables
            "model_predictions",
            "model_metadata",
            "ensemble_predictions",
            "backtest_results",
            "feature_importance",
            # Staging tables
            "stg_nflverse_pbp",
            "stg_nflverse_rosters",
            "stg_odds",
        }

    def scan_codebase(self) -> None:
        """Scan entire codebase and build column lineage."""
        self.logger.info("Starting codebase scan...")

        # Scan Python files
        python_files = []
        for dir_pattern in self.python_dirs:
            pattern_path = self.project_root / dir_pattern
            if pattern_path.exists():
                python_files.extend(pattern_path.rglob("*.py"))

        self.logger.info(f"Scanning {len(python_files)} Python files...")
        for file_path in python_files:
            self._scan_python_file(file_path)

        # Scan R files
        r_files = []
        for dir_pattern in self.r_dirs:
            pattern_path = self.project_root / dir_pattern
            if pattern_path.exists():
                r_files.extend(pattern_path.rglob("*.R"))

        self.logger.info(f"Scanning {len(r_files)} R files...")
        for file_path in r_files:
            self._scan_r_file(file_path)

        # Scan SQL files
        sql_files = []
        for dir_pattern in self.sql_dirs:
            pattern_path = self.project_root / dir_pattern
            if pattern_path.exists():
                sql_files.extend(pattern_path.rglob("*.sql"))

        self.logger.info(f"Scanning {len(sql_files)} SQL files...")
        for file_path in sql_files:
            self._scan_sql_file(file_path)

        self.logger.info(f"Scan complete. Found {len(self.lineage)} unique columns with usages.")

    def _scan_python_file(self, file_path: Path) -> None:
        """Scan a Python file for SQL queries and column references."""
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Extract SQL queries using various patterns
            sql_queries = []

            # Multi-line triple-quoted strings
            for match in re.finditer(r'("""(.+?)""")|' + r"('''(.+?)''')", content, re.DOTALL):
                query = match.group(2) or match.group(4)
                if query and ("SELECT" in query.upper() or "FROM" in query.upper()):
                    sql_queries.append((query, match.start()))

            # Single-line queries in function calls
            for pattern in self.python_sql_patterns[:3]:  # read_sql, execute, dbGetQuery
                for match in re.finditer(pattern, content, re.IGNORECASE | re.DOTALL):
                    sql_queries.append((match.group(1), match.start()))

            # Parse SQL queries for column references
            for query, char_pos in sql_queries:
                line_number = content[:char_pos].count("\n") + 1
                self._extract_columns_from_sql(query, file_path, line_number)

        except Exception as e:
            self.logger.warning(f"Error scanning {file_path}: {e}")

    def _scan_r_file(self, file_path: Path) -> None:
        """Scan an R file for SQL queries and column references."""
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Extract SQL queries from R database calls
            sql_queries = []

            # dbGetQuery, dbReadTable, etc.
            for pattern in self.r_sql_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE | re.DOTALL):
                    query = match.group(1)
                    line_number = content[: match.start()].count("\n") + 1
                    sql_queries.append((query, line_number))

            # Parse SQL queries for column references
            for query, line_number in sql_queries:
                self._extract_columns_from_sql(query, file_path, line_number)

        except Exception as e:
            self.logger.warning(f"Error scanning {file_path}: {e}")

    def _scan_sql_file(self, file_path: Path) -> None:
        """Scan a raw SQL file for column references."""
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # SQL files may have multiple statements
            statements = content.split(";")

            for stmt in statements:
                if stmt.strip():
                    # Rough line number estimate
                    line_number = content[: content.find(stmt)].count("\n") + 1
                    self._extract_columns_from_sql(stmt, file_path, line_number)

        except Exception as e:
            self.logger.warning(f"Error scanning {file_path}: {e}")

    def _extract_columns_from_sql(self, query: str, file_path: Path, line_number: int) -> None:
        """Extract column references from SQL query."""
        query_upper = query.upper()

        # Determine consumer type based on file path
        consumer_type = self._classify_consumer_type(file_path)

        # Extract table.column patterns
        for match in re.finditer(r"\b(\w+)\.(\w+)\b", query):
            table_name = match.group(1).lower()
            column_name = match.group(2).lower()

            # Filter to known tables (reduces false positives from aliases)
            if (
                table_name in self.known_tables
                or table_name.startswith("stg_")
                or table_name.startswith("mv_")
            ):
                usage = ColumnUsage(
                    table_name=table_name,
                    column_name=column_name,
                    file_path=str(file_path.relative_to(self.project_root)),
                    line_number=line_number,
                    usage_type=self._infer_usage_type(query_upper, column_name),
                    context=self._extract_context(query, column_name),
                    consumer_type=consumer_type,
                )

                key = f"{table_name}.{column_name}"
                self.lineage[key].append(usage)

        # Extract bare column names from SELECT clause
        select_match = re.search(r"SELECT\s+(.+?)\s+FROM", query, re.IGNORECASE | re.DOTALL)
        if select_match:
            columns_str = select_match.group(1)
            # Parse individual columns (naive parsing, can be improved)
            for col_name in re.findall(r"\b(\w+)\b", columns_str):
                if col_name.upper() not in ["SELECT", "FROM", "WHERE", "AS", "DISTINCT", "ALL"]:
                    # Try to infer table from FROM clause
                    from_match = re.search(r"FROM\s+(\w+)", query, re.IGNORECASE)
                    if from_match:
                        table_name = from_match.group(1).lower()
                        if table_name in self.known_tables:
                            usage = ColumnUsage(
                                table_name=table_name,
                                column_name=col_name.lower(),
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=line_number,
                                usage_type="select",
                                context=self._extract_context(query, col_name),
                                consumer_type=consumer_type,
                            )

                            key = f"{table_name}.{col_name.lower()}"
                            self.lineage[key].append(usage)

    def _classify_consumer_type(self, file_path: Path) -> str:
        """Classify the type of code consuming this column."""
        path_str = str(file_path).lower()

        if "feature" in path_str:
            return "feature"
        elif "model" in path_str or "train" in path_str:
            return "model"
        elif "backtest" in path_str:
            return "backtest"
        elif "production" in path_str or "deploy" in path_str:
            return "production"
        elif "test" in path_str or "validation" in path_str:
            return "validation"
        elif "migration" in path_str or "db/" in path_str:
            return "migration"
        else:
            return "other"

    def _infer_usage_type(self, query: str, column_name: str) -> str:
        """Infer how the column is being used in the query."""
        if "SELECT" in query and column_name in query:
            if "WHERE" in query.split(column_name)[1]:
                return "filter"
            elif "GROUP BY" in query and column_name in query.split("GROUP BY")[1]:
                return "groupby"
            elif "ORDER BY" in query and column_name in query.split("ORDER BY")[1]:
                return "orderby"
            else:
                return "select"
        elif "INSERT" in query:
            return "insert"
        elif "UPDATE" in query:
            return "update"
        elif "DELETE" in query:
            return "delete"
        else:
            return "unknown"

    def _extract_context(self, query: str, column_name: str) -> str:
        """Extract relevant context around column usage (max 80 chars)."""
        # Find the column in the query
        pattern = re.compile(rf"\b{re.escape(column_name)}\b", re.IGNORECASE)
        match = pattern.search(query)

        if match:
            start = max(0, match.start() - 40)
            end = min(len(query), match.end() + 40)
            context = query[start:end].strip()
            return context[:80]

        return query[:80]

    def trace_column(self, column: str) -> list[ColumnUsage]:
        """
        Get all usages of a specific column.

        Args:
            column: Format 'table.column' (e.g., 'nfl_plays.passing_yards')

        Returns:
            List of ColumnUsage objects
        """
        return self.lineage.get(column.lower(), [])

    def analyze_impact(self, column: str, change_type: str) -> ImpactReport:
        """
        Analyze the impact of a schema change.

        Args:
            column: Format 'table.column' (e.g., 'injuries.report_status')
            change_type: 'add', 'modify', or 'remove'

        Returns:
            ImpactReport with breaking changes, affected components, opportunities
        """
        report = ImpactReport(column=column, change_type=change_type)
        usages = self.trace_column(column)

        if change_type == "remove":
            # All usages are breaking changes
            report.breaking_changes = usages
            report.risk_level = "critical" if usages else "low"

            # Classify affected components
            for usage in usages:
                if usage.consumer_type == "model":
                    report.affected_models.add(usage.file_path)
                elif usage.consumer_type == "feature":
                    report.affected_features.add(usage.file_path)
                elif usage.consumer_type == "backtest":
                    report.affected_backtests.add(usage.file_path)

            # Estimate impact hours (rough heuristic)
            report.estimated_impact_hours = len(usages) * 0.5  # 30 min per usage

        elif change_type == "modify":
            # Some usages may break depending on modification type
            report.breaking_changes = usages  # Conservative: treat all as potentially breaking
            report.risk_level = "high" if len(usages) > 5 else "medium" if usages else "low"

            for usage in usages:
                if usage.consumer_type == "model":
                    report.affected_models.add(usage.file_path)
                elif usage.consumer_type == "feature":
                    report.affected_features.add(usage.file_path)
                elif usage.consumer_type == "backtest":
                    report.affected_backtests.add(usage.file_path)

            report.estimated_impact_hours = len(usages) * 0.25  # 15 min per usage

        elif change_type == "add":
            # No breaking changes, but identify opportunities
            report.risk_level = "low"
            report.opportunities = self._identify_opportunities(column)
            report.estimated_impact_hours = (
                len(report.opportunities) * 1.0
            )  # 1 hour per opportunity

        return report

    def _identify_opportunities(self, new_column: str) -> list[str]:
        """
        Identify code that could benefit from a new column.

        Args:
            new_column: Format 'table.column' (e.g., 'weather.humidity')

        Returns:
            List of file paths that could use this column
        """
        opportunities = []
        table_name = new_column.split(".")[0]
        column_name = new_column.split(".")[1]

        # Heuristic: Find files that already use the same table
        for existing_column, usages in self.lineage.items():
            if existing_column.startswith(f"{table_name}."):
                for usage in usages:
                    # Check if the new column is semantically related
                    if self._is_semantically_related(column_name, usage.file_path):
                        opportunities.append(usage.file_path)

        # Deduplicate
        return list(set(opportunities))

    def _is_semantically_related(self, column_name: str, file_path: str) -> bool:
        """
        Heuristic to determine if a file might benefit from a column.

        Examples:
        - weather_humidity → files with 'weather' or 'environmental' in path
        - injury_severity → files with 'injury' or 'health' in path
        """
        column_lower = column_name.lower()
        path_lower = file_path.lower()

        # Extract keywords from column name
        keywords = column_lower.split("_")

        # Check if any keyword appears in file path
        return any(keyword in path_lower for keyword in keywords if len(keyword) > 3)

    def find_unused_columns(
        self, known_columns: dict[str, list[str]] | None = None
    ) -> dict[str, list[str]]:
        """
        Find database columns that are never used in code.

        Args:
            known_columns: Dict of {table_name: [column_names]} from database schema.
                          If None, uses self.known_tables (less accurate).

        Returns:
            Dict of {table_name: [unused_column_names]}
        """
        if known_columns is None:
            # Fallback: can't determine unused columns without schema info
            self.logger.warning("No schema provided. Cannot accurately detect unused columns.")
            return {}

        unused = defaultdict(list)

        for table_name, columns in known_columns.items():
            for column_name in columns:
                key = f"{table_name}.{column_name}"
                if key.lower() not in self.lineage:
                    unused[table_name].append(column_name)

        return dict(unused)

    def generate_dependency_graph(self, output_format: str = "graphviz") -> str:
        """
        Generate a dependency graph showing column → consumer relationships.

        Args:
            output_format: 'graphviz', 'mermaid', or 'json'

        Returns:
            Graph representation as string
        """
        if output_format == "graphviz":
            lines = ["digraph column_lineage {", "  rankdir=LR;", "  node [shape=box];"]

            for column, usages in self.lineage.items():
                for usage in usages:
                    # Shorten file paths
                    short_path = Path(usage.file_path).name
                    lines.append(f'  "{column}" -> "{short_path}:{usage.line_number}";')

            lines.append("}")
            return "\n".join(lines)

        elif output_format == "mermaid":
            lines = ["graph LR"]

            for column, usages in self.lineage.items():
                col_id = column.replace(".", "_")
                for i, usage in enumerate(usages):
                    short_path = Path(usage.file_path).name
                    consumer_id = f"{short_path}_{i}".replace(".", "_").replace("-", "_")
                    lines.append(f"  {col_id}[{column}] --> {consumer_id}[{short_path}]")

            return "\n".join(lines)

        elif output_format == "json":
            graph = {"nodes": [], "edges": []}

            # Nodes: columns + consumers
            node_ids = set()
            for column, usages in self.lineage.items():
                if column not in node_ids:
                    graph["nodes"].append({"id": column, "type": "column"})
                    node_ids.add(column)

                for usage in usages:
                    consumer_id = f"{usage.file_path}:{usage.line_number}"
                    if consumer_id not in node_ids:
                        graph["nodes"].append(
                            {
                                "id": consumer_id,
                                "type": "consumer",
                                "file": usage.file_path,
                                "line": usage.line_number,
                            }
                        )
                        node_ids.add(consumer_id)

                    graph["edges"].append(
                        {"source": column, "target": consumer_id, "usage_type": usage.usage_type}
                    )

            return json.dumps(graph, indent=2)

        else:
            raise ValueError(f"Unsupported format: {output_format}")

    def save_lineage(self, output_path: str) -> None:
        """Save lineage to JSON file for later analysis."""
        data = {
            column: [
                {
                    "table": u.table_name,
                    "column": u.column_name,
                    "file": u.file_path,
                    "line": u.line_number,
                    "usage_type": u.usage_type,
                    "consumer_type": u.consumer_type,
                    "context": u.context,
                }
                for u in usages
            ]
            for column, usages in self.lineage.items()
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Lineage saved to {output_path}")

    def load_lineage(self, input_path: str) -> None:
        """Load lineage from JSON file (faster than re-scanning)."""
        with open(input_path) as f:
            data = json.load(f)

        self.lineage.clear()
        for column, usages_data in data.items():
            self.lineage[column] = [
                ColumnUsage(
                    table_name=u["table"],
                    column_name=u["column"],
                    file_path=u["file"],
                    line_number=u["line"],
                    usage_type=u["usage_type"],
                    context=u["context"],
                    consumer_type=u["consumer_type"],
                )
                for u in usages_data
            ]

        self.logger.info(f"Lineage loaded from {input_path} ({len(self.lineage)} columns)")


def main():
    """Example usage."""
    analyzer = DataLayerAnalyzer()

    # Scan codebase
    analyzer.scan_codebase()

    # Trace a column
    print("\n=== Tracing 'plays.passing_yards' ===")
    usages = analyzer.trace_column("plays.passing_yards")
    print(f"Found {len(usages)} usages")
    for usage in usages[:5]:  # Show first 5
        print(f"  {usage.file_path}:{usage.line_number} ({usage.consumer_type})")

    # Analyze impact of removing a column
    print("\n=== Impact Analysis: Remove 'injuries.report_status' ===")
    report = analyzer.analyze_impact("injuries.report_status", "remove")
    print(report.summary())

    # Analyze impact of adding a column
    print("\n=== Impact Analysis: Add 'weather.humidity' ===")
    report = analyzer.analyze_impact("weather.humidity", "add")
    print(report.summary())
    if report.opportunities:
        print(f"\nOpportunities ({len(report.opportunities)}):")
        for opp in report.opportunities[:5]:
            print(f"  - {opp}")

    # Save lineage
    analyzer.save_lineage("/tmp/column_lineage.json")


if __name__ == "__main__":
    main()
