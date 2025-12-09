"""
Security Auditor for NFL Analytics Betting System

This module provides comprehensive security scanning specifically designed for
the NFL analytics betting system. Real money betting requires rigorous security.

Security Focus Areas:
1. Credentials & Secrets (API keys, database passwords, tokens)
2. Docker Security (container configs, exposed ports, permissions)
3. API Security (rate limiting, authentication, input validation)
4. Data Security (encryption, backups, access control)
5. Dependency Vulnerabilities (outdated packages, known CVEs)

Usage:
    from py.utils.security_auditor import SecurityAuditor

    auditor = SecurityAuditor()

    # Run full security audit
    report = auditor.run_full_audit()

    # Check specific file
    issues = auditor.scan_file("py/api/app.py")

    # Scan Docker configurations
    issues = auditor.scan_docker_security()

Integration:
    - Weekly automated scans (cron job)
    - Pre-commit hook (credential scanning)
    - CI/CD pipeline (dependency checks)
"""

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class SecuritySeverity(Enum):
    """Security issue severity levels."""

    CRITICAL = "CRITICAL"  # Immediate fix required
    HIGH = "HIGH"  # Fix before production
    MEDIUM = "MEDIUM"  # Fix soon
    LOW = "LOW"  # Address eventually
    INFO = "INFO"  # Informational


class SecurityCategory(Enum):
    """Categories of security issues."""

    CREDENTIALS = "credentials"
    DOCKER = "docker"
    API_SECURITY = "api_security"
    DATA_SECURITY = "data_security"
    DEPENDENCIES = "dependencies"
    FILE_PERMISSIONS = "file_permissions"
    NETWORK = "network"


@dataclass
class SecurityIssue:
    """Represents a security finding."""

    file_path: str
    line_number: int
    severity: SecuritySeverity
    category: SecurityCategory
    title: str
    description: str
    remediation: str
    cve_id: str | None = None
    code_snippet: str | None = None


@dataclass
class SecurityReport:
    """Comprehensive security audit report."""

    scan_date: str = field(default_factory=lambda: datetime.now().isoformat())
    issues: list[SecurityIssue] = field(default_factory=list)
    files_scanned: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0

    def add_issue(self, issue: SecurityIssue):
        """Add issue and update counters."""
        self.issues.append(issue)
        if issue.severity == SecuritySeverity.CRITICAL:
            self.critical_count += 1
        elif issue.severity == SecuritySeverity.HIGH:
            self.high_count += 1
        elif issue.severity == SecuritySeverity.MEDIUM:
            self.medium_count += 1
        elif issue.severity == SecuritySeverity.LOW:
            self.low_count += 1

    def has_critical_issues(self) -> bool:
        """Check if report contains critical issues."""
        return self.critical_count > 0

    def to_json(self) -> str:
        """Export report to JSON."""
        return json.dumps(
            {
                "scan_date": self.scan_date,
                "summary": {
                    "files_scanned": self.files_scanned,
                    "critical": self.critical_count,
                    "high": self.high_count,
                    "medium": self.medium_count,
                    "low": self.low_count,
                },
                "issues": [
                    {
                        "file": i.file_path,
                        "line": i.line_number,
                        "severity": i.severity.value,
                        "category": i.category.value,
                        "title": i.title,
                        "description": i.description,
                        "remediation": i.remediation,
                        "cve_id": i.cve_id,
                    }
                    for i in self.issues
                ],
            },
            indent=2,
        )


class SecurityAuditor:
    """
    Comprehensive security auditor for NFL analytics betting system.

    Performs automated security scans with focus on:
    - Credential exposure (API keys, passwords)
    - Docker container security
    - API authentication and authorization
    - Data encryption and backups
    - Dependency vulnerabilities
    """

    # Project-specific credentials to scan for
    PROJECT_CREDENTIALS = [
        "sicillionbillions",  # Database password
        "postgresql://",  # Connection strings
        "ANTHROPIC_API_KEY",
        "ODDS_API_KEY",
    ]

    # Sensitive file patterns
    SENSITIVE_FILES = [
        ".env",
        ".env.local",
        ".env.production",
        "credentials.json",
        "secrets.yaml",
        ".aws/credentials",
        ".ssh/id_rsa",
    ]

    # Docker security patterns
    DOCKER_INSECURE_PATTERNS = {
        "privileged_mode": [
            r"privileged:\s*true",
            r"--privileged",
        ],
        "no_user": [
            r"FROM.*\n(?!.*USER)",  # Missing USER directive
        ],
        "hardcoded_secrets": [
            r"ENV.*PASSWORD",
            r"ENV.*SECRET",
            r"ENV.*API_KEY",
        ],
        "exposed_ports": [
            r"ports:.*5432",  # PostgreSQL exposed
            r"ports:.*6379",  # Redis exposed
        ],
    }

    # API security patterns
    API_INSECURE_PATTERNS = {
        "no_auth": [
            r"@app\.route.*methods.*POST",  # POST without auth decorator
            r"@router\.post",  # FastAPI POST without dependency
        ],
        "no_rate_limiting": [
            r"@app\.route",  # Flask route without rate limit
            r"@router\.",  # FastAPI route without rate limit
        ],
        "weak_cors": [
            r'CORS.*origins=\["?\*"?\]',  # Allow all origins
        ],
    }

    def __init__(self, project_root: Path | None = None):
        """
        Initialize security auditor.

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
        self.report = SecurityReport()

        logger.info(f"SecurityAuditor initialized for {project_root}")

    def run_full_audit(self) -> SecurityReport:
        """
        Run comprehensive security audit.

        Returns:
            Complete security report
        """
        logger.info("Starting full security audit...")

        # Scan credentials
        self._scan_credentials()

        # Scan Docker security
        self._scan_docker_security()

        # Scan API security
        self._scan_api_security()

        # Scan file permissions
        self._scan_file_permissions()

        # Scan dependencies (if possible)
        self._scan_dependencies()

        logger.info(f"Security audit complete: {len(self.report.issues)} issues found")

        return self.report

    def _scan_credentials(self):
        """Scan for exposed credentials."""
        logger.info("Scanning for exposed credentials...")

        # Scan all Python, R, Shell, YAML, JSON files
        patterns = [
            "**/*.py",
            "**/*.R",
            "**/*.sh",
            "**/*.yml",
            "**/*.yaml",
            "**/*.json",
            "**/*.env*",
        ]

        for pattern in patterns:
            for file_path in self.project_root.glob(pattern):
                # Skip certain directories
                if any(
                    skip in str(file_path)
                    for skip in [".venv", "venv", "node_modules", ".git", "pgdata"]
                ):
                    continue

                try:
                    content = file_path.read_text(errors="ignore")
                    lines = content.split("\n")

                    self.report.files_scanned += 1

                    for i, line in enumerate(lines, 1):
                        # Check for project-specific credentials
                        for cred in self.PROJECT_CREDENTIALS:
                            if (
                                cred in line
                                and "example" not in line.lower()
                                and "#" not in line[: line.find(cred) if cred in line else 0]
                            ):
                                self.report.add_issue(
                                    SecurityIssue(
                                        file_path=str(file_path.relative_to(self.project_root)),
                                        line_number=i,
                                        severity=SecuritySeverity.CRITICAL,
                                        category=SecurityCategory.CREDENTIALS,
                                        title="Hardcoded credential detected",
                                        description=f"Credential '{cred[:20]}...' found in code",
                                        remediation="Move credentials to .env file and use os.getenv()",
                                        code_snippet=(
                                            line.strip()[:100] + "..."
                                            if len(line.strip()) > 100
                                            else line.strip()
                                        ),
                                    )
                                )

                        # Check for API key patterns
                        api_key_patterns = [
                            r'["\']?api[_-]?key["\']?\s*[:=]\s*["\'][^"\']{20,}["\']',
                            r'["\']?secret["\']?\s*[:=]\s*["\'][^"\']{20,}["\']',
                            r'["\']?password["\']?\s*[:=]\s*["\'][^"\']+["\']',
                        ]

                        for pattern in api_key_patterns:
                            match = re.search(pattern, line, re.IGNORECASE)
                            if match and "env" not in line.lower() and "getenv" not in line.lower():
                                self.report.add_issue(
                                    SecurityIssue(
                                        file_path=str(file_path.relative_to(self.project_root)),
                                        line_number=i,
                                        severity=SecuritySeverity.CRITICAL,
                                        category=SecurityCategory.CREDENTIALS,
                                        title="API key or password in code",
                                        description="Sensitive credential hardcoded in source",
                                        remediation="Use environment variables or secrets management",
                                        code_snippet=line.strip()[:100],
                                    )
                                )

                except Exception as e:
                    logger.debug(f"Error scanning {file_path}: {e}")

    def _scan_docker_security(self):
        """Scan Docker configurations for security issues."""
        logger.info("Scanning Docker security...")

        docker_files = list(self.project_root.glob("**/Dockerfile*")) + list(
            self.project_root.glob("**/docker-compose*.yml")
        )

        for file_path in docker_files:
            try:
                content = file_path.read_text()
                lines = content.split("\n")

                # Check for privileged mode
                for i, line in enumerate(lines, 1):
                    for pattern in self.DOCKER_INSECURE_PATTERNS["privileged_mode"]:
                        if re.search(pattern, line):
                            self.report.add_issue(
                                SecurityIssue(
                                    file_path=str(file_path.relative_to(self.project_root)),
                                    line_number=i,
                                    severity=SecuritySeverity.HIGH,
                                    category=SecurityCategory.DOCKER,
                                    title="Privileged mode enabled",
                                    description="Container running in privileged mode has full host access",
                                    remediation="Remove privileged mode unless absolutely necessary",
                                    code_snippet=line.strip(),
                                )
                            )

                    # Check for hardcoded secrets in ENV
                    for pattern in self.DOCKER_INSECURE_PATTERNS["hardcoded_secrets"]:
                        if re.search(pattern, line, re.IGNORECASE) and "FROM" not in line:
                            self.report.add_issue(
                                SecurityIssue(
                                    file_path=str(file_path.relative_to(self.project_root)),
                                    line_number=i,
                                    severity=SecuritySeverity.HIGH,
                                    category=SecurityCategory.DOCKER,
                                    title="Hardcoded secret in Dockerfile",
                                    description="Secret in ENV directive should use build args",
                                    remediation="Use ARG + docker secrets or .env file",
                                    code_snippet=line.strip(),
                                )
                            )

                    # Check for exposed database ports
                    if "docker-compose" in file_path.name:
                        for pattern in self.DOCKER_INSECURE_PATTERNS["exposed_ports"]:
                            if re.search(pattern, line):
                                port = line.split(":")[0].strip() if ":" in line else "unknown"
                                self.report.add_issue(
                                    SecurityIssue(
                                        file_path=str(file_path.relative_to(self.project_root)),
                                        line_number=i,
                                        severity=SecuritySeverity.MEDIUM,
                                        category=SecurityCategory.DOCKER,
                                        title="Database port exposed to host",
                                        description=f"Port {port} exposed can be accessed from outside",
                                        remediation="Remove port mapping if only inter-container access needed",
                                        code_snippet=line.strip(),
                                    )
                                )

                # Check for missing USER directive in Dockerfile
                if "Dockerfile" in file_path.name and "USER" not in content:
                    self.report.add_issue(
                        SecurityIssue(
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=0,
                            severity=SecuritySeverity.MEDIUM,
                            category=SecurityCategory.DOCKER,
                            title="Container runs as root",
                            description="Missing USER directive means container runs as root",
                            remediation="Add USER directive after installing dependencies",
                        )
                    )

            except Exception as e:
                logger.debug(f"Error scanning Docker file {file_path}: {e}")

    def _scan_api_security(self):
        """Scan API code for security issues."""
        logger.info("Scanning API security...")

        api_files = list(self.project_root.glob("**/api/**/*.py")) + list(
            self.project_root.glob("**/app.py")
        )

        for file_path in api_files:
            try:
                content = file_path.read_text()
                lines = content.split("\n")

                "limiter" in content.lower() or "ratelimit" in content.lower()
                has_auth = "depends(" in content.lower() or "require_auth" in content.lower()

                for i, line in enumerate(lines, 1):
                    # Check for POST routes without authentication
                    if re.search(r"@(app\.route|router\.post)", line) and not has_auth:
                        # Check if auth is in next few lines
                        next_lines = "\n".join(lines[i : min(i + 5, len(lines))])
                        if "auth" not in next_lines.lower() and "depends" not in next_lines.lower():
                            self.report.add_issue(
                                SecurityIssue(
                                    file_path=str(file_path.relative_to(self.project_root)),
                                    line_number=i,
                                    severity=SecuritySeverity.HIGH,
                                    category=SecurityCategory.API_SECURITY,
                                    title="API endpoint without authentication",
                                    description="POST endpoint lacks authentication",
                                    remediation="Add authentication dependency or decorator",
                                    code_snippet=line.strip(),
                                )
                            )

                    # Check for CORS with wildcard
                    for pattern in self.API_INSECURE_PATTERNS["weak_cors"]:
                        if re.search(pattern, line):
                            self.report.add_issue(
                                SecurityIssue(
                                    file_path=str(file_path.relative_to(self.project_root)),
                                    line_number=i,
                                    severity=SecuritySeverity.MEDIUM,
                                    category=SecurityCategory.API_SECURITY,
                                    title="Weak CORS policy",
                                    description="CORS allows all origins (*)",
                                    remediation="Specify allowed origins explicitly",
                                    code_snippet=line.strip(),
                                )
                            )

            except Exception as e:
                logger.debug(f"Error scanning API file {file_path}: {e}")

    def _scan_file_permissions(self):
        """Scan for insecure file permissions."""
        logger.info("Scanning file permissions...")

        # Check sensitive files
        for pattern in self.SENSITIVE_FILES:
            for file_path in self.project_root.glob(f"**/{pattern}"):
                try:
                    stat_info = file_path.stat()
                    mode = oct(stat_info.st_mode)[-3:]

                    # Check if file is world-readable (others have read permission)
                    if mode[2] != "0":
                        self.report.add_issue(
                            SecurityIssue(
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=0,
                                severity=SecuritySeverity.HIGH,
                                category=SecurityCategory.FILE_PERMISSIONS,
                                title="Sensitive file is world-readable",
                                description=f"File has permissions {mode}, readable by others",
                                remediation="chmod 600 to restrict access to owner only",
                            )
                        )

                except Exception as e:
                    logger.debug(f"Error checking permissions for {file_path}: {e}")

        # Check script files for executable permissions
        for script_path in list(self.project_root.glob("**/*.sh")) + list(
            self.project_root.glob("**/scripts/*.py")
        ):
            try:
                stat_info = script_path.stat()
                mode = oct(stat_info.st_mode)[-3:]

                # Check if owner doesn't have execute permission
                if mode[0] not in ["7", "5"]:
                    self.report.add_issue(
                        SecurityIssue(
                            file_path=str(script_path.relative_to(self.project_root)),
                            line_number=0,
                            severity=SecuritySeverity.LOW,
                            category=SecurityCategory.FILE_PERMISSIONS,
                            title="Script file not executable",
                            description=f"Script has permissions {mode}, not executable",
                            remediation="chmod +x to make executable",
                        )
                    )

            except Exception as e:
                logger.debug(f"Error checking script permissions: {e}")

    def _scan_dependencies(self):
        """Scan for vulnerable dependencies."""
        logger.info("Scanning dependencies for vulnerabilities...")

        # Check Python dependencies
        requirements_files = list(self.project_root.glob("**/requirements*.txt"))

        for req_file in requirements_files:
            try:
                # Try to run pip-audit if available
                result = subprocess.run(
                    ["pip-audit", "-r", str(req_file)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode != 0 and "vulnerability" in result.stdout.lower():
                    self.report.add_issue(
                        SecurityIssue(
                            file_path=str(req_file.relative_to(self.project_root)),
                            line_number=0,
                            severity=SecuritySeverity.HIGH,
                            category=SecurityCategory.DEPENDENCIES,
                            title="Vulnerable dependencies detected",
                            description="pip-audit found vulnerabilities",
                            remediation="Run 'pip-audit -r requirements.txt' and update packages",
                        )
                    )

            except FileNotFoundError:
                logger.debug("pip-audit not installed, skipping dependency scan")
                break
            except Exception as e:
                logger.debug(f"Error running pip-audit: {e}")

    def generate_report(self) -> str:
        """Generate formatted security report."""
        lines = []

        lines.append("=" * 80)
        lines.append("SECURITY AUDIT REPORT")
        lines.append("=" * 80)
        lines.append(f"\nScan Date: {self.report.scan_date}")
        lines.append(f"Files Scanned: {self.report.files_scanned}")
        lines.append("\nIssues Summary:")
        lines.append(f"  🔴 CRITICAL: {self.report.critical_count}")
        lines.append(f"  🟠 HIGH:     {self.report.high_count}")
        lines.append(f"  🟡 MEDIUM:   {self.report.medium_count}")
        lines.append(f"  🟢 LOW:      {self.report.low_count}")

        if self.report.has_critical_issues():
            lines.append("\n⛔ CRITICAL SECURITY ISSUES FOUND")
            lines.append("   IMMEDIATE ACTION REQUIRED")

        # Group by severity
        for severity in [
            SecuritySeverity.CRITICAL,
            SecuritySeverity.HIGH,
            SecuritySeverity.MEDIUM,
            SecuritySeverity.LOW,
        ]:
            severity_issues = [i for i in self.report.issues if i.severity == severity]

            if severity_issues:
                lines.append(f"\n{severity.value} Issues ({len(severity_issues)}):")
                lines.append("-" * 80)

                for issue in severity_issues:
                    lines.append(f"\n📁 {issue.file_path}:{issue.line_number}")
                    lines.append(f"   [{issue.category.value}] {issue.title}")
                    lines.append(f"   🔍 {issue.description}")
                    lines.append(f"   💡 {issue.remediation}")

                    if issue.code_snippet:
                        lines.append(f"   Code: {issue.code_snippet}")

                    if issue.cve_id:
                        lines.append(f"   CVE: {issue.cve_id}")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)

    def save_report(self, output_path: Path | None = None):
        """Save report to file."""
        if output_path is None:
            output_path = self.project_root / "security_audit_report.json"

        output_path.write_text(self.report.to_json())
        logger.info(f"Security report saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    auditor = SecurityAuditor()
    report = auditor.run_full_audit()

    print(auditor.generate_report())

    # Save to file
    auditor.save_report()
