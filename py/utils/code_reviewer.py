"""
Multi-Language Code Reviewer for NFL Analytics Project

This module provides automated code review capabilities specifically designed for
the NFL analytics betting system, with emphasis on:

1. Multi-language support (Python 68%, TeX 19%, R 7%, SQL 3%, JS 2%, Shell 1%)
2. Data leakage detection (temporal validation, future data usage)
3. Bayesian modeling patterns (prior validation, MCMC diagnostics)
4. Security vulnerabilities (API keys, credentials, SQL injection)
5. SQL schema validation (mart.player_game_stats.game_id bug detection)
6. Code quality (type hints, docstrings, conventions)

Usage:
    from py.utils.code_reviewer import CodeReviewer

    reviewer = CodeReviewer()

    # Review Python file
    issues = reviewer.review_file("py/models/train_bnn_passing.py")

    # Review all changed files
    issues = reviewer.review_changes(git_diff=True)

    # Generate review report
    report = reviewer.generate_report()

SQL Schema Validation:
    The reviewer detects recurring database schema bugs, particularly:
    - mart.player_game_stats.game_id (DOES NOT EXIST - use season+week composite key)
    - Slow JOIN patterns (OR conditions in JOIN predicates)
    - Missing indexes on date columns

    Example:
        # This will be flagged as CRITICAL error:
        query = "SELECT * FROM mart.player_game_stats pgs JOIN games g ON g.game_id = pgs.game_id"

        # Correct pattern:
        query = "SELECT * FROM mart.player_game_stats pgs JOIN games g ON g.season = pgs.season AND g.week = pgs.week"

Integration:
    - Pre-commit hook: Blocks commits with critical issues
    - CI/CD: GitHub Actions runs on all PRs
    - Manual: CLI tool for on-demand reviews (./scripts/review_code.sh --check-sql)
"""

import ast
import logging
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class Severity(Enum):
    """Issue severity levels."""
    CRITICAL = "CRITICAL"  # Blocks commit/merge
    HIGH = "HIGH"          # Should fix before merge
    MEDIUM = "MEDIUM"      # Should fix soon
    LOW = "LOW"            # Nice to have
    INFO = "INFO"          # Informational


class IssueCategory(Enum):
    """Categories of code review issues."""
    DATA_LEAKAGE = "data_leakage"
    SECURITY = "security"
    BAYESIAN_MODELING = "bayesian_modeling"
    CODE_QUALITY = "code_quality"
    PERFORMANCE = "performance"
    TESTING = "testing"
    DOCUMENTATION = "documentation"


@dataclass
class ReviewIssue:
    """Represents a code review finding."""
    file_path: str
    line_number: int
    severity: Severity
    category: IssueCategory
    title: str
    description: str
    suggestion: str | None = None
    code_snippet: str | None = None


@dataclass
class ReviewReport:
    """Comprehensive code review report."""
    issues: list[ReviewIssue] = field(default_factory=list)
    files_reviewed: int = 0
    lines_reviewed: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0

    def add_issue(self, issue: ReviewIssue):
        """Add issue and update counters."""
        self.issues.append(issue)
        if issue.severity == Severity.CRITICAL:
            self.critical_count += 1
        elif issue.severity == Severity.HIGH:
            self.high_count += 1
        elif issue.severity == Severity.MEDIUM:
            self.medium_count += 1
        elif issue.severity == Severity.LOW:
            self.low_count += 1

    def has_blocking_issues(self) -> bool:
        """Check if report contains blocking issues."""
        return self.critical_count > 0


class CodeReviewer:
    """
    Multi-language code reviewer for NFL analytics project.

    Performs automated code review with focus on:
    - Data leakage prevention (temporal, player-level)
    - Security vulnerabilities
    - Bayesian modeling correctness
    - Code quality and conventions
    """

    # Patterns for data leakage detection
    LEAKAGE_PATTERNS = {
        # Temporal leakage indicators
        "future_data": [
            r"\.shift\(-\d+\)",  # Pandas shift with negative (future data)
            r"df\[.*>\s*date\]",  # Filter for future dates
            r"merge.*how=['\"]outer['\"]",  # Outer joins can leak
        ],

        # Missing asof joins (should use for temporal data)
        "missing_asof": [
            r"pd\.merge.*on=['\"]date['\"]",  # Regular merge on date
            r"\.merge\(.*left_on=['\"].*date",  # Date merge without asof
        ],

        # Target leakage
        "target_leakage": [
            r"train.*test.*split.*shuffle\s*=\s*True",  # Shuffling temporal data
            r"feature.*actual_.*yards",  # Using actual outcome in features
            r"feature.*result",  # Using game result in features
        ],
    }

    # Security vulnerability patterns
    SECURITY_PATTERNS = {
        "hardcoded_credentials": [
            r"password\s*=\s*['\"][^'\"]+['\"]",
            r"sicillionbillions",  # Project-specific password
            r"api_key\s*=\s*['\"][^'\"]+['\"]",
            r"secret\s*=\s*['\"][^'\"]+['\"]",
        ],

        "sql_injection": [
            r"execute\(f['\"]",  # f-string in SQL execute
            r"execute\(['\"].*\{",  # String formatting in SQL
            r"%\s*%.*execute",  # % formatting in SQL
        ],

        "command_injection": [
            r"subprocess.*shell\s*=\s*True",  # Shell=True is dangerous
            r"os\.system\(",  # os.system is vulnerable
        ],
    }

    # SQL schema-specific anti-patterns (project-specific)
    SQL_ANTI_PATTERNS = [
        {
            "pattern": r"player_game_stats.*\.game_id|game_id.*player_game_stats|pgs\.game_id",
            "severity": Severity.CRITICAL,
            "message": "mart.player_game_stats does NOT have game_id column",
            "description": (
                "RECURRING BUG: mart.player_game_stats only has (season, week) keys, not game_id. "
                "This error has occurred DOZENS of times in this project."
            ),
            "fix": (
                "Use composite join: JOIN games g ON g.season = pgs.season AND g.week = pgs.week "
                "AND g.home_team = pgs.current_team (or similar team matching)"
            ),
            "examples": [
                "❌ BAD:  JOIN games g ON g.game_id = pgs.game_id",
                "✅ GOOD: JOIN games g ON g.season = pgs.season AND g.week = pgs.week",
            ],
        },
        {
            "pattern": r"SELECT\s+COUNT\(\*\)\s+FROM.*WHERE.*=.*AND.*=",
            "severity": Severity.MEDIUM,
            "message": "Consider using COUNT(*) with proper indexes",
            "description": "COUNT(*) on large tables can be slow without proper indexes",
            "fix": "Ensure indexes exist on filter columns or use EXPLAIN ANALYZE",
        },
        {
            "pattern": r"LEFT\s+JOIN.*ON.*OR|RIGHT\s+JOIN.*ON.*OR",
            "severity": Severity.HIGH,
            "message": "OR conditions in JOIN can be extremely slow",
            "description": "OR in JOIN predicates prevents index usage and causes table scans",
            "fix": "Split into UNION of two joins or restructure query",
        },
    ]

    # Bayesian modeling patterns
    BAYESIAN_PATTERNS = {
        "prior_issues": [
            r"prior\s*=\s*None",  # Missing prior specification
            r"normal\(0,\s*1000\)",  # Unreasonably wide prior
            r"cauchy\(0,\s*100\)",  # Very wide Cauchy prior
        ],

        "mcmc_config": [
            r"chains\s*=\s*1\b",  # Single chain (should be 4+)
            r"tune\s*=\s*\d{1,3}\b",  # Very short tuning (<1000)
            r"draws\s*=\s*\d{1,3}\b",  # Very few samples (<1000)
        ],

        "missing_diagnostics": [
            r"sample\(",  # PyMC sampling
            r"brm\(",  # brms model
        ],
    }

    def __init__(self, project_root: Path | None = None):
        """
        Initialize code reviewer.

        Args:
            project_root: Root directory of project (default: auto-detect)
        """
        if project_root is None:
            # Auto-detect project root
            current = Path.cwd()
            while current != current.parent:
                if (current / "CLAUDE.md").exists():
                    project_root = current
                    break
                current = current.parent
            else:
                project_root = Path.cwd()

        self.project_root = project_root
        self.report = ReviewReport()

        logger.info(f"CodeReviewer initialized for {project_root}")

    def review_file(self, file_path: str) -> list[ReviewIssue]:
        """
        Review a single file.

        Args:
            file_path: Path to file to review

        Returns:
            List of issues found
        """
        path = Path(file_path)

        if not path.exists():
            logger.warning(f"File not found: {file_path}")
            return []

        # Route to appropriate language-specific reviewer
        if path.suffix == ".py":
            return self._review_python(path)
        elif path.suffix == ".R":
            return self._review_r(path)
        elif path.suffix == ".sql":
            return self._review_sql(path)
        elif path.suffix == ".tex":
            return self._review_latex(path)
        elif path.suffix in [".js", ".jsx"]:
            return self._review_javascript(path)
        elif path.suffix == ".sh":
            return self._review_shell(path)
        else:
            logger.debug(f"Skipping unsupported file type: {file_path}")
            return []

    def _review_python(self, path: Path) -> list[ReviewIssue]:
        """Review Python file for issues."""
        issues = []

        try:
            content = path.read_text()
            lines = content.split("\n")

            # Check data leakage patterns
            issues.extend(self._check_data_leakage(path, content, lines))

            # Check security vulnerabilities
            issues.extend(self._check_security(path, content, lines))

            # Check SQL queries embedded in Python
            if "SELECT" in content or "FROM" in content or "JOIN" in content:
                issues.extend(self._check_sql_schema_errors(path, content, lines))

            # Check Bayesian modeling patterns
            if "models/" in str(path) or "pymc" in content.lower() or "stan" in content.lower():
                issues.extend(self._check_bayesian(path, content, lines))

            # Check code quality
            issues.extend(self._check_python_quality(path, content, lines))

            # AST-based checks
            try:
                tree = ast.parse(content)
                issues.extend(self._check_python_ast(path, tree))
            except SyntaxError as e:
                issues.append(ReviewIssue(
                    file_path=str(path),
                    line_number=e.lineno or 0,
                    severity=Severity.CRITICAL,
                    category=IssueCategory.CODE_QUALITY,
                    title="Syntax error",
                    description=f"Python syntax error: {e.msg}",
                ))

        except Exception as e:
            logger.error(f"Error reviewing {path}: {e}")

        return issues

    def _check_data_leakage(self, path: Path, content: str, lines: list[str]) -> list[ReviewIssue]:
        """Check for data leakage patterns."""
        issues = []

        # Check future data usage
        for i, line in enumerate(lines, 1):
            for pattern in self.LEAKAGE_PATTERNS["future_data"]:
                if re.search(pattern, line):
                    issues.append(ReviewIssue(
                        file_path=str(path),
                        line_number=i,
                        severity=Severity.CRITICAL,
                        category=IssueCategory.DATA_LEAKAGE,
                        title="Potential future data leakage",
                        description=f"Pattern '{pattern}' suggests using future data",
                        suggestion="Use asof join or ensure temporal ordering",
                        code_snippet=line.strip(),
                    ))

            # Check for missing asof joins on dates
            for pattern in self.LEAKAGE_PATTERNS["missing_asof"]:
                if re.search(pattern, line) and "asof" not in line.lower():
                    issues.append(ReviewIssue(
                        file_path=str(path),
                        line_number=i,
                        severity=Severity.HIGH,
                        category=IssueCategory.DATA_LEAKAGE,
                        title="Date merge without asof join",
                        description="Merging on date without asof join risks temporal leakage",
                        suggestion="Use pd.merge_asof() for temporal joins",
                        code_snippet=line.strip(),
                    ))

            # Check target leakage
            for pattern in self.LEAKAGE_PATTERNS["target_leakage"]:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(ReviewIssue(
                        file_path=str(path),
                        line_number=i,
                        severity=Severity.CRITICAL,
                        category=IssueCategory.DATA_LEAKAGE,
                        title="Potential target leakage",
                        description="Using outcome variable in features or shuffling temporal data",
                        suggestion="Remove outcome from features, use temporal split",
                        code_snippet=line.strip(),
                    ))

        return issues

    def _check_security(self, path: Path, content: str, lines: list[str]) -> list[ReviewIssue]:
        """Check for security vulnerabilities."""
        issues = []

        for i, line in enumerate(lines, 1):
            # Hardcoded credentials
            for pattern in self.SECURITY_PATTERNS["hardcoded_credentials"]:
                if re.search(pattern, line) and "os.getenv" not in line and "env" not in line.lower():
                    issues.append(ReviewIssue(
                        file_path=str(path),
                        line_number=i,
                        severity=Severity.CRITICAL,
                        category=IssueCategory.SECURITY,
                        title="Hardcoded credentials detected",
                        description="Credentials should be in environment variables, not code",
                        suggestion="Use os.getenv() or .env file",
                        code_snippet=line.strip(),
                    ))

            # SQL injection
            for pattern in self.SECURITY_PATTERNS["sql_injection"]:
                if re.search(pattern, line):
                    issues.append(ReviewIssue(
                        file_path=str(path),
                        line_number=i,
                        severity=Severity.CRITICAL,
                        category=IssueCategory.SECURITY,
                        title="Potential SQL injection vulnerability",
                        description="String formatting in SQL execute() can allow injection",
                        suggestion="Use parameterized queries with %s placeholders",
                        code_snippet=line.strip(),
                    ))

            # Command injection
            for pattern in self.SECURITY_PATTERNS["command_injection"]:
                if re.search(pattern, line):
                    issues.append(ReviewIssue(
                        file_path=str(path),
                        line_number=i,
                        severity=Severity.HIGH,
                        category=IssueCategory.SECURITY,
                        title="Potential command injection",
                        description="shell=True or os.system() can allow command injection",
                        suggestion="Use subprocess with shell=False and list arguments",
                        code_snippet=line.strip(),
                    ))

        return issues

    def _check_bayesian(self, path: Path, content: str, lines: list[str]) -> list[ReviewIssue]:
        """Check Bayesian modeling patterns."""
        issues = []

        has_sampling = False
        has_diagnostics = False

        for i, line in enumerate(lines, 1):
            # Check prior specifications
            for pattern in self.BAYESIAN_PATTERNS["prior_issues"]:
                if re.search(pattern, line):
                    issues.append(ReviewIssue(
                        file_path=str(path),
                        line_number=i,
                        severity=Severity.MEDIUM,
                        category=IssueCategory.BAYESIAN_MODELING,
                        title="Questionable prior specification",
                        description="Prior may be too wide or missing",
                        suggestion="Use informative priors based on domain knowledge",
                        code_snippet=line.strip(),
                    ))

            # Check MCMC configuration
            for pattern in self.BAYESIAN_PATTERNS["mcmc_config"]:
                if re.search(pattern, line):
                    issues.append(ReviewIssue(
                        file_path=str(path),
                        line_number=i,
                        severity=Severity.HIGH,
                        category=IssueCategory.BAYESIAN_MODELING,
                        title="Insufficient MCMC sampling",
                        description="MCMC config may lead to poor convergence",
                        suggestion="Use chains=4, tune=1000, draws=2000 minimum",
                        code_snippet=line.strip(),
                    ))

            # Track if model uses sampling
            if "sample(" in line or "brm(" in line:
                has_sampling = True

            # Track if diagnostics are run
            if any(diag in line.lower() for diag in ["rhat", "ess", "divergences", "summary", "diagnose"]):
                has_diagnostics = True

        # Check for missing diagnostics
        if has_sampling and not has_diagnostics:
            issues.append(ReviewIssue(
                file_path=str(path),
                line_number=0,
                severity=Severity.HIGH,
                category=IssueCategory.BAYESIAN_MODELING,
                title="Missing MCMC diagnostics",
                description="Bayesian model trained without diagnostic checks",
                suggestion="Add R-hat, ESS, divergence checks after sampling",
            ))

        return issues

    def _check_python_quality(self, path: Path, content: str, lines: list[str]) -> list[ReviewIssue]:
        """Check Python code quality."""
        issues = []

        # Check for type hints in function definitions
        for i, line in enumerate(lines, 1):
            if line.strip().startswith("def ") and "(" in line:
                if "->" not in line and "__init__" not in line and "__" not in line:
                    issues.append(ReviewIssue(
                        file_path=str(path),
                        line_number=i,
                        severity=Severity.LOW,
                        category=IssueCategory.CODE_QUALITY,
                        title="Missing return type hint",
                        description="Function lacks return type annotation",
                        suggestion="Add -> Type annotation to function",
                        code_snippet=line.strip(),
                    ))

        # Check docstrings for public functions/classes
        func_pattern = r"^(def|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)"
        for i, line in enumerate(lines, 1):
            match = re.match(func_pattern, line.strip())
            if match and not match.group(2).startswith("_"):
                # Check if next non-empty line is docstring
                for j in range(i, min(i + 3, len(lines))):
                    next_line = lines[j].strip()
                    if next_line and not next_line.startswith("#"):
                        if '"""' not in next_line and "'''" not in next_line:
                            issues.append(ReviewIssue(
                                file_path=str(path),
                                line_number=i,
                                severity=Severity.LOW,
                                category=IssueCategory.DOCUMENTATION,
                                title="Missing docstring",
                                description=f"Public {match.group(1)} lacks docstring",
                                suggestion="Add Google-style docstring",
                                code_snippet=line.strip(),
                            ))
                        break

        return issues

    def _check_python_ast(self, path: Path, tree: ast.AST) -> list[ReviewIssue]:
        """AST-based Python checks."""
        issues = []

        for node in ast.walk(tree):
            # Check for bare except clauses
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                issues.append(ReviewIssue(
                    file_path=str(path),
                    line_number=node.lineno,
                    severity=Severity.MEDIUM,
                    category=IssueCategory.CODE_QUALITY,
                    title="Bare except clause",
                    description="Catching all exceptions can hide bugs",
                    suggestion="Catch specific exceptions (ValueError, KeyError, etc.)",
                ))

        return issues

    def _review_r(self, path: Path) -> list[ReviewIssue]:
        """Review R file for issues."""
        issues = []

        try:
            content = path.read_text()
            lines = content.split("\n")

            for i, line in enumerate(lines, 1):
                # Check for assignment operator (should use <-)
                if re.search(r"\w+\s*=\s*\w+", line) and not re.search(r"[=!<>]=", line):
                    issues.append(ReviewIssue(
                        file_path=str(path),
                        line_number=i,
                        severity=Severity.LOW,
                        category=IssueCategory.CODE_QUALITY,
                        title="Use <- for assignment",
                        description="R style guide recommends <- over =",
                        suggestion="Replace = with <-",
                        code_snippet=line.strip(),
                    ))

                # Check for data leakage in brms models
                if "brm(" in line and "prior" not in content[max(0, content.find(line)-500):content.find(line)]:
                    issues.append(ReviewIssue(
                        file_path=str(path),
                        line_number=i,
                        severity=Severity.MEDIUM,
                        category=IssueCategory.BAYESIAN_MODELING,
                        title="Missing prior specification in brms",
                        description="brms model may be using default priors",
                        suggestion="Specify priors explicitly with set_prior()",
                        code_snippet=line.strip(),
                    ))

        except Exception as e:
            logger.error(f"Error reviewing R file {path}: {e}")

        return issues

    def _review_sql(self, path: Path) -> list[ReviewIssue]:
        """Review SQL file for issues."""
        issues = []

        try:
            content = path.read_text()
            lines = content.split("\n")

            # Check SQL anti-patterns (schema-specific)
            issues.extend(self._check_sql_schema_errors(path, content, lines))

            for i, line in enumerate(lines, 1):
                # Check for SELECT *
                if re.search(r"SELECT\s+\*", line, re.IGNORECASE):
                    issues.append(ReviewIssue(
                        file_path=str(path),
                        line_number=i,
                        severity=Severity.LOW,
                        category=IssueCategory.PERFORMANCE,
                        title="Avoid SELECT *",
                        description="SELECT * can be slow and fragile",
                        suggestion="List specific columns",
                        code_snippet=line.strip(),
                    ))

                # Check for missing indexes on date columns
                if re.search(r"WHERE.*date", line, re.IGNORECASE) and "CREATE INDEX" not in content:
                    issues.append(ReviewIssue(
                        file_path=str(path),
                        line_number=i,
                        severity=Severity.MEDIUM,
                        category=IssueCategory.PERFORMANCE,
                        title="Consider index on date column",
                        description="Filtering by date without index can be slow",
                        suggestion="Add CREATE INDEX ON table(date_column)",
                    ))

        except Exception as e:
            logger.error(f"Error reviewing SQL file {path}: {e}")

        return issues

    def _check_sql_schema_errors(self, path: Path, content: str, lines: list[str]) -> list[ReviewIssue]:
        """
        Check for project-specific SQL schema errors.

        This function detects recurring bugs specific to the NFL analytics database schema,
        particularly the mart.player_game_stats table which DOES NOT have a game_id column.

        Args:
            path: Path to SQL file
            content: Full file content
            lines: List of lines

        Returns:
            List of schema-related issues found

        Example:
            >>> reviewer = CodeReviewer()
            >>> # This will be flagged as CRITICAL error:
            >>> bad_sql = "SELECT * FROM mart.player_game_stats pgs JOIN games g ON g.game_id = pgs.game_id"
            >>> issues = reviewer._check_sql_schema_errors(Path("test.sql"), bad_sql, bad_sql.split("\\n"))
            >>> assert len(issues) > 0
            >>> assert issues[0].severity == Severity.CRITICAL
        """
        issues = []

        for i, line in enumerate(lines, 1):
            # Check each anti-pattern
            for anti_pattern in self.SQL_ANTI_PATTERNS:
                pattern = anti_pattern["pattern"]
                if re.search(pattern, line, re.IGNORECASE):
                    # Build detailed error message
                    error_msg = anti_pattern["description"]

                    # Add examples if available
                    suggestion_parts = [anti_pattern["fix"]]
                    if "examples" in anti_pattern:
                        suggestion_parts.append("\n\n" + "\n".join(anti_pattern["examples"]))

                    issues.append(ReviewIssue(
                        file_path=str(path),
                        line_number=i,
                        severity=anti_pattern["severity"],
                        category=IssueCategory.DATA_LEAKAGE,  # Schema errors can cause data issues
                        title=anti_pattern["message"],
                        description=error_msg,
                        suggestion="\n".join(suggestion_parts),
                        code_snippet=line.strip(),
                    ))

        return issues

    def _review_latex(self, path: Path) -> list[ReviewIssue]:
        """Review LaTeX file for issues."""
        issues = []

        try:
            content = path.read_text()
            lines = content.split("\n")

            for i, line in enumerate(lines, 1):
                # Check for undefined references
                if "???" in line or "ref{}" in line:
                    issues.append(ReviewIssue(
                        file_path=str(path),
                        line_number=i,
                        severity=Severity.MEDIUM,
                        category=IssueCategory.DOCUMENTATION,
                        title="Undefined reference",
                        description="LaTeX reference may be undefined",
                        suggestion="Check label exists and compile twice",
                        code_snippet=line.strip(),
                    ))

        except Exception as e:
            logger.error(f"Error reviewing LaTeX file {path}: {e}")

        return issues

    def _review_javascript(self, path: Path) -> list[ReviewIssue]:
        """Review JavaScript file for issues."""
        # Placeholder for JavaScript review
        return []

    def _review_shell(self, path: Path) -> list[ReviewIssue]:
        """Review shell script for issues."""
        issues = []

        try:
            content = path.read_text()
            lines = content.split("\n")

            for i, line in enumerate(lines, 1):
                # Check for unquoted variables
                if re.search(r"\$\w+", line) and not re.search(r'"\$\w+"', line):
                    issues.append(ReviewIssue(
                        file_path=str(path),
                        line_number=i,
                        severity=Severity.MEDIUM,
                        category=IssueCategory.CODE_QUALITY,
                        title="Unquoted variable",
                        description="Unquoted variables can cause word splitting",
                        suggestion='Use "$VAR" instead of $VAR',
                        code_snippet=line.strip(),
                    ))

        except Exception as e:
            logger.error(f"Error reviewing shell script {path}: {e}")

        return issues

    def review_changes(self, git_diff: bool = True) -> ReviewReport:
        """
        Review all changed files in repository.

        Args:
            git_diff: If True, only review git-tracked changes

        Returns:
            Comprehensive review report
        """
        if git_diff:
            # Get changed files from git
            try:
                result = subprocess.run(
                    ["git", "diff", "--name-only", "HEAD"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                )
                changed_files = result.stdout.strip().split("\n")
            except Exception as e:
                logger.error(f"Error getting git changes: {e}")
                changed_files = []
        else:
            # Review all files (not implemented yet)
            changed_files = []

        for file_path in changed_files:
            if file_path:
                issues = self.review_file(str(self.project_root / file_path))
                for issue in issues:
                    self.report.add_issue(issue)
                self.report.files_reviewed += 1

        return self.report

    def generate_report(self) -> str:
        """Generate formatted review report."""
        report_lines = []

        report_lines.append("=" * 80)
        report_lines.append("CODE REVIEW REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"\nFiles Reviewed: {self.report.files_reviewed}")
        report_lines.append(f"Lines Reviewed: {self.report.lines_reviewed}")
        report_lines.append("\nIssues Found:")
        report_lines.append(f"  CRITICAL: {self.report.critical_count}")
        report_lines.append(f"  HIGH:     {self.report.high_count}")
        report_lines.append(f"  MEDIUM:   {self.report.medium_count}")
        report_lines.append(f"  LOW:      {self.report.low_count}")

        if self.report.has_blocking_issues():
            report_lines.append("\n⛔ CRITICAL ISSUES FOUND - BLOCKING COMMIT/MERGE")

        # Group issues by severity
        for severity in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]:
            severity_issues = [i for i in self.report.issues if i.severity == severity]

            if severity_issues:
                report_lines.append(f"\n{severity.value} Issues ({len(severity_issues)}):")
                report_lines.append("-" * 80)

                for issue in severity_issues:
                    report_lines.append(f"\n{issue.file_path}:{issue.line_number}")
                    report_lines.append(f"  [{issue.category.value}] {issue.title}")
                    report_lines.append(f"  {issue.description}")

                    if issue.code_snippet:
                        report_lines.append(f"  Code: {issue.code_snippet}")

                    if issue.suggestion:
                        report_lines.append(f"  💡 {issue.suggestion}")

        report_lines.append("\n" + "=" * 80)

        return "\n".join(report_lines)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    reviewer = CodeReviewer()

    # Review a specific file
    test_file = "py/models/train_bnn_passing.py"
    if Path(test_file).exists():
        issues = reviewer.review_file(test_file)
        print(f"\nReviewed {test_file}: {len(issues)} issues found")

        for issue in issues:
            print(f"\n{issue.severity.value}: {issue.title}")
            print(f"  {issue.description}")

    # Review all changes
    report = reviewer.review_changes(git_diff=True)
    print("\n" + reviewer.generate_report())
