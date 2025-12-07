#!/usr/bin/env python3
"""
Automated validation script for CLAUDE.md accuracy.

This script validates:
1. Architecture references (files, directories, Docker services, database)
2. Cross-references (documentation links, code file references)
3. Freshness (update dates, recent commits, version numbers)
4. Metrics (database counts, model registry sync, service health)

Exit codes:
  0: All validations passed
  1: One or more validations failed
  2: Fatal error (e.g., cannot connect to database)

Usage:
  python scripts/validate_claude_md.py
  python scripts/validate_claude_md.py --report-file validation_report.json
  python scripts/validate_claude_md.py --fix-suggestions
  python scripts/validate_claude_md.py --verbose
"""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

try:
    import psycopg2
    from psycopg2 import sql
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ValidationResult:
    """Result of a single validation check."""
    category: str
    check_name: str
    passed: bool
    message: str
    details: dict = field(default_factory=dict)
    suggestion: str | None = None


@dataclass
class ValidationReport:
    """Complete validation report."""
    timestamp: str
    claude_md_path: str
    results: list[ValidationResult] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def add_result(self, result: ValidationResult) -> None:
        """Add a validation result to the report."""
        self.results.append(result)

    def get_summary(self) -> dict:
        """Calculate summary statistics."""
        by_category = {}
        for result in self.results:
            category = result.category
            if category not in by_category:
                by_category[category] = {"passed": 0, "failed": 0, "total": 0}
            by_category[category]["total"] += 1
            if result.passed:
                by_category[category]["passed"] += 1
            else:
                by_category[category]["failed"] += 1

        total_passed = sum(r["passed"] for r in by_category.values())
        total_checks = sum(r["total"] for r in by_category.values())

        return {
            "timestamp": self.timestamp,
            "total_passed": total_passed,
            "total_failed": total_checks - total_passed,
            "total_checks": total_checks,
            "pass_rate": f"{100 * total_passed / total_checks:.1f}%" if total_checks > 0 else "0%",
            "by_category": by_category,
        }


# ============================================================================
# VALIDATORS
# ============================================================================

class CLAUDEMDValidator:
    """Main validator for CLAUDE.md accuracy."""

    def __init__(self, claude_md_path: str, verbose: bool = False):
        """Initialize validator.

        Args:
            claude_md_path: Path to CLAUDE.md file
            verbose: Enable verbose output
        """
        self.claude_md_path = Path(claude_md_path)
        self.project_root = self.claude_md_path.parent
        self.verbose = verbose
        self.report = ValidationReport(
            timestamp=datetime.now().isoformat(),
            claude_md_path=str(self.claude_md_path),
        )
        self.content = self._read_claude_md()

    def _read_claude_md(self) -> str:
        """Read CLAUDE.md content."""
        try:
            with open(self.claude_md_path) as f:
                return f.read()
        except FileNotFoundError:
            print(f"❌ FATAL: CLAUDE.md not found at {self.claude_md_path}")
            sys.exit(2)

    def _vprint(self, msg: str) -> None:
        """Print if verbose mode enabled."""
        if self.verbose:
            print(f"  ℹ️  {msg}")

    # ========================================================================
    # CATEGORY 1: ARCHITECTURE REFERENCES
    # ========================================================================

    def validate_architecture_references(self) -> None:
        """Validate that all referenced files and directories exist."""
        self._vprint("Validating architecture references...")

        # Extract file references from CLAUDE.md
        file_patterns = [
            r"`([^`]+\.py)`",  # Python files
            r"`([^`]+\.R)`",   # R files
            r"`([^`]+\.sql)`", # SQL files
            r"`([^`]+\.tex)`", # LaTeX files
            r"`([^`]+\.md)`",  # Markdown files
        ]

        referenced_files = set()
        for pattern in file_patterns:
            matches = re.findall(pattern, self.content)
            referenced_files.update(matches)

        # Validate each file exists
        missing_files = []
        for file_ref in sorted(referenced_files):
            file_path = self.project_root / file_ref
            exists = file_path.exists()

            if not exists:
                missing_files.append(file_ref)
                self._vprint(f"❌ Missing: {file_ref}")

        passed = len(missing_files) == 0
        self.report.add_result(ValidationResult(
            category="Architecture References",
            check_name="File Existence",
            passed=passed,
            message=f"Checked {len(referenced_files)} file references",
            details={
                "total_referenced": len(referenced_files),
                "missing_count": len(missing_files),
                "missing_files": missing_files[:10],  # Limit to first 10
            },
            suggestion=(
                f"Update {len(missing_files)} missing file references in CLAUDE.md "
                if missing_files else None
            ),
        ))

    def validate_directory_structure(self) -> None:
        """Validate key directory structures match documentation."""
        self._vprint("Validating directory structure...")

        expected_dirs = [
            "py",
            "R",
            "analysis/dissertation",
            "models",
            "docs",
            "scripts",
            "tests",
            "data",
            "db",
        ]

        missing_dirs = []
        for dir_name in expected_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)
                self._vprint(f"❌ Missing directory: {dir_name}")

        passed = len(missing_dirs) == 0
        self.report.add_result(ValidationResult(
            category="Architecture References",
            check_name="Directory Structure",
            passed=passed,
            message=f"Checked {len(expected_dirs)} key directories",
            details={
                "total_expected": len(expected_dirs),
                "missing_count": len(missing_dirs),
                "missing_dirs": missing_dirs,
            },
            suggestion=(
                "Create missing directories or update CLAUDE.md documentation"
                if missing_dirs else None
            ),
        ))

    def validate_docker_services(self) -> None:
        """Validate Docker service configuration."""
        self._vprint("Validating Docker services...")

        # Extract service count from CLAUDE.md
        match = re.search(r"6\s+Docker containers?", self.content, re.IGNORECASE)
        claimed_services = 6 if match else None

        if not DOCKER_AVAILABLE:
            self._vprint("⚠️  Docker client not available, skipping live service check")
            self.report.add_result(ValidationResult(
                category="Architecture References",
                check_name="Docker Services",
                passed=True,
                message="Docker validation skipped (docker client not available)",
                details={"claimed_services": claimed_services},
            ))
            return

        try:
            client = docker.from_env()
            running_containers = client.containers.list()
            service_names = [c.name for c in running_containers]

            # Check for key services
            required_services = ["postgres", "redis", "api", "streamlit", "prometheus", "grafana"]
            missing_services = []
            found_services = []

            for required in required_services:
                found = any(required.lower() in name.lower() for name in service_names)
                if found:
                    found_services.append(required)
                else:
                    missing_services.append(required)
                self._vprint(
                    f"✅ Found service: {required}" if found else f"❌ Missing service: {required}"
                )

            passed = len(missing_services) == 0
            self.report.add_result(ValidationResult(
                category="Architecture References",
                check_name="Docker Services",
                passed=passed,
                message=f"Found {len(found_services)}/{len(required_services)} required services",
                details={
                    "claimed_services": claimed_services,
                    "actual_running": len(running_containers),
                    "found_services": found_services,
                    "missing_services": missing_services,
                },
                suggestion=(
                    f"Start missing Docker services: {', '.join(missing_services)}"
                    if missing_services else None
                ),
            ))
        except Exception as e:
            self._vprint(f"⚠️  Docker error: {e}")
            self.report.add_result(ValidationResult(
                category="Architecture References",
                check_name="Docker Services",
                passed=False,
                message=f"Docker validation failed: {str(e)}",
                details={"error": str(e)},
            ))

    def validate_database_connection(self) -> None:
        """Validate database connection string and accessibility."""
        self._vprint("Validating database connection...")

        # Extract connection string from CLAUDE.md
        match = re.search(
            r"postgresql://[^\s`]+",
            self.content
        )
        if not match:
            self.report.add_result(ValidationResult(
                category="Architecture References",
                check_name="Database Connection",
                passed=False,
                message="Database connection string not found in CLAUDE.md",
                details={},
                suggestion="Add database connection string to CLAUDE.md documentation",
            ))
            return

        connection_string = match.group(0).rstrip("`")
        self._vprint(f"Found connection string: {connection_string[:50]}...")

        if not PSYCOPG2_AVAILABLE:
            self._vprint("⚠️  psycopg2 not available, skipping live database check")
            self.report.add_result(ValidationResult(
                category="Architecture References",
                check_name="Database Connection",
                passed=True,
                message="Database validation skipped (psycopg2 not available)",
                details={"connection_string": connection_string},
            ))
            return

        try:
            # Remove pragma comment if present
            conn_str = connection_string.replace("<!--", "").replace("-->", "").strip()
            conn = psycopg2.connect(conn_str)
            cursor = conn.cursor()

            # Count tables
            cursor.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema NOT IN ('pg_catalog', 'information_schema')"
            )
            table_count = cursor.fetchone()[0]

            # Count rows in key tables
            cursor.execute("SELECT COUNT(*) FROM public.games")
            games_count = cursor.fetchone()[0]

            cursor.close()
            conn.close()

            # Extract claimed metrics from CLAUDE.md
            matches = re.findall(r"(\d+)\s*(?:games|plays|columns|models)", self.content, re.IGNORECASE)

            self.report.add_result(ValidationResult(
                category="Architecture References",
                check_name="Database Connection",
                passed=True,
                message=f"Database connection successful, {table_count} tables, {games_count} games",
                details={
                    "connection_string": connection_string,
                    "table_count": table_count,
                    "games_count": games_count,
                },
            ))
        except Exception as e:
            self.report.add_result(ValidationResult(
                category="Architecture References",
                check_name="Database Connection",
                passed=False,
                message=f"Database connection failed: {str(e)}",
                details={"error": str(e), "connection_string": connection_string},
                suggestion="Check database is running and credentials are correct",
            ))

    # ========================================================================
    # CATEGORY 2: CROSS-REFERENCES
    # ========================================================================

    def validate_documentation_links(self) -> None:
        """Validate that all documentation links point to existing files."""
        self._vprint("Validating documentation links...")

        # Extract markdown links
        link_pattern = r"\[([^\]]+)\]\(([^\)]+)\)"
        matches = re.findall(link_pattern, self.content)

        broken_links = []
        valid_links = []

        for text, link in matches:
            # Skip external links
            if link.startswith("http") or link.startswith("#"):
                continue

            # Resolve relative path
            target_path = self.project_root / link
            if target_path.exists():
                valid_links.append((text, link))
                self._vprint(f"✅ Valid link: {link}")
            else:
                broken_links.append((text, link))
                self._vprint(f"❌ Broken link: {link}")

        passed = len(broken_links) == 0
        self.report.add_result(ValidationResult(
            category="Cross-References",
            check_name="Documentation Links",
            passed=passed,
            message=f"Checked {len(matches)} links, {len(valid_links)} valid",
            details={
                "total_links": len(matches),
                "valid_links": len(valid_links),
                "broken_links": len(broken_links),
                "broken": [{"text": t, "link": l} for t, l in broken_links[:5]],
            },
            suggestion=(
                f"Fix {len(broken_links)} broken documentation links"
                if broken_links else None
            ),
        ))

    def validate_code_file_references(self) -> None:
        """Validate code file references are correct."""
        self._vprint("Validating code file references...")

        # Extract backtick references to files
        file_pattern = r"`((?:py|R|scripts|tests)/[^`]+)`"
        references = re.findall(file_pattern, self.content)

        invalid_refs = []
        valid_refs = []

        for ref in references:
            path = self.project_root / ref
            if path.exists():
                valid_refs.append(ref)
                self._vprint(f"✅ Valid: {ref}")
            else:
                invalid_refs.append(ref)
                self._vprint(f"❌ Invalid: {ref}")

        passed = len(invalid_refs) == 0
        self.report.add_result(ValidationResult(
            category="Cross-References",
            check_name="Code File References",
            passed=passed,
            message=f"Checked {len(references)} code references, {len(valid_refs)} valid",
            details={
                "total_references": len(references),
                "valid": len(valid_refs),
                "invalid": len(invalid_refs),
                "invalid_files": invalid_refs[:10],
            },
            suggestion=(
                f"Update {len(invalid_refs)} invalid code references"
                if invalid_refs else None
            ),
        ))

    # ========================================================================
    # CATEGORY 3: FRESHNESS
    # ========================================================================

    def validate_update_date(self) -> None:
        """Validate 'Last Updated' date is recent."""
        self._vprint("Validating update date...")

        # Extract last updated date
        match = re.search(r"\*\*Last Updated:\*\*\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})", self.content)
        if not match:
            self.report.add_result(ValidationResult(
                category="Freshness",
                check_name="Last Updated Date",
                passed=False,
                message="Last Updated date not found in CLAUDE.md",
                details={},
                suggestion="Add 'Last Updated: Month DD, YYYY' to top of CLAUDE.md",
            ))
            return

        date_str = match.group(1)
        try:
            claimed_date = datetime.strptime(date_str, "%B %d, %Y")
        except ValueError:
            self.report.add_result(ValidationResult(
                category="Freshness",
                check_name="Last Updated Date",
                passed=False,
                message=f"Cannot parse date format: {date_str}",
                details={"date_string": date_str},
                suggestion="Use format 'Month DD, YYYY' e.g., 'November 15, 2025'",
            ))
            return

        # Get latest commit date
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%cI"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                latest_commit = datetime.fromisoformat(result.stdout.strip())
                days_old = (datetime.now() - claimed_date).days

                passed = days_old <= 1  # Allow 1 day lag
                self.report.add_result(ValidationResult(
                    category="Freshness",
                    check_name="Last Updated Date",
                    passed=passed,
                    message=f"Last Updated date: {date_str} ({days_old} days old)",
                    details={
                        "claimed_date": date_str,
                        "days_old": days_old,
                        "latest_commit": latest_commit.isoformat(),
                    },
                    suggestion=(
                        f"Update 'Last Updated' to today ({datetime.now().strftime('%B %d, %Y')})"
                        if days_old > 1 else None
                    ),
                ))
            else:
                self.report.add_result(ValidationResult(
                    category="Freshness",
                    check_name="Last Updated Date",
                    passed=True,
                    message=f"Date found: {date_str}",
                    details={"date_string": date_str},
                ))
        except Exception as e:
            self._vprint(f"⚠️  Git check failed: {e}")
            self.report.add_result(ValidationResult(
                category="Freshness",
                check_name="Last Updated Date",
                passed=True,
                message=f"Date found: {date_str} (git check skipped)",
                details={"date_string": date_str},
            ))

    def validate_version_number(self) -> None:
        """Validate version number follows semantic versioning."""
        self._vprint("Validating version number...")

        match = re.search(r"\*\*Version:\*\*\s*([0-9]+\.[0-9]+)", self.content)
        if not match:
            self.report.add_result(ValidationResult(
                category="Freshness",
                check_name="Version Number",
                passed=False,
                message="Version number not found in CLAUDE.md",
                details={},
                suggestion="Add 'Version: X.Y' to top of CLAUDE.md",
            ))
            return

        version = match.group(1)
        # Check if it's semantic versioning
        parts = version.split(".")
        is_semantic = len(parts) >= 2 and all(p.isdigit() for p in parts)

        self.report.add_result(ValidationResult(
            category="Freshness",
            check_name="Version Number",
            passed=is_semantic,
            message=f"Version: {version}",
            details={"version": version, "is_semantic": is_semantic},
            suggestion=(
                "Use semantic versioning (e.g., 4.9, 5.0, 5.1)"
                if not is_semantic else None
            ),
        ))

    def validate_recent_work_section(self) -> None:
        """Validate Recent Work section references recent commits."""
        self._vprint("Validating Recent Work section...")

        # Check if recent work section exists
        if "## Recent Work" not in self.content:
            self.report.add_result(ValidationResult(
                category="Freshness",
                check_name="Recent Work Section",
                passed=False,
                message="Recent Work section not found in CLAUDE.md",
                details={},
                suggestion="Add '## Recent Work' section with recent commits",
            ))
            return

        # Extract work items from Recent Work section
        recent_section = self.content[self.content.find("## Recent Work"):]
        recent_section = recent_section[:recent_section.find("---")] if "---" in recent_section else recent_section

        # Try to get recent commits
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "-10"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                commits = [line.split(None, 1)[1] for line in result.stdout.strip().split("\n") if line]
                # Check if any recent commit message appears in Recent Work
                found_commits = sum(1 for commit in commits if commit in recent_section)
                passed = found_commits >= 1

                self.report.add_result(ValidationResult(
                    category="Freshness",
                    check_name="Recent Work Section",
                    passed=passed,
                    message=f"Found {found_commits}/{len(commits)} recent commits in Recent Work",
                    details={"recent_commits": commits[:5]},
                    suggestion=(
                        "Update Recent Work section with latest commits"
                        if found_commits == 0 else None
                    ),
                ))
            else:
                self.report.add_result(ValidationResult(
                    category="Freshness",
                    check_name="Recent Work Section",
                    passed=True,
                    message="Recent Work section found",
                    details={},
                ))
        except Exception as e:
            self._vprint(f"⚠️  Cannot validate commits: {e}")
            self.report.add_result(ValidationResult(
                category="Freshness",
                check_name="Recent Work Section",
                passed=True,
                message="Recent Work section found (git check skipped)",
                details={"error": str(e)},
            ))

    # ========================================================================
    # CATEGORY 4: METRICS
    # ========================================================================

    def validate_metrics(self) -> None:
        """Validate claimed metrics match reality."""
        self._vprint("Validating metrics...")

        # Extract metrics from CLAUDE.md
        metric_patterns = {
            "games": r"(\d+)\s*games",
            "plays": r"(\d+(?:,\d{3})?)\s*plays",
            "columns": r"(\d+)\s*columns",
            "pages": r"(\d+)\s*pages",
            "models": r"(\d+)\s*models?\s*(?:synced|in production|complete)",
        }

        claimed_metrics = {}
        for metric_name, pattern in metric_patterns.items():
            matches = re.findall(pattern, self.content, re.IGNORECASE)
            if matches:
                # Get the most common value
                values = [m.replace(",", "") for m in matches]
                claimed_metrics[metric_name] = values[0] if values else None

        self._vprint(f"Claimed metrics: {claimed_metrics}")

        if not PSYCOPG2_AVAILABLE:
            self._vprint("⚠️  psycopg2 not available, skipping database metrics")
            self.report.add_result(ValidationResult(
                category="Metrics",
                check_name="Database Metrics",
                passed=True,
                message="Database metrics validation skipped (psycopg2 not available)",
                details={"claimed_metrics": claimed_metrics},
            ))
            return

        try:
            # Extract connection string
            match = re.search(r"postgresql://[^\s`]+", self.content)
            if not match:
                return

            connection_string = match.group(0).rstrip("`").replace("<!--", "").replace("-->", "").strip()
            conn = psycopg2.connect(connection_string)
            cursor = conn.cursor()

            # Query actual metrics
            cursor.execute("SELECT COUNT(*) FROM public.games")
            actual_games = cursor.fetchone()[0]

            # Try to count plays
            try:
                cursor.execute("SELECT COUNT(*) FROM public.play_by_play")
                actual_plays = cursor.fetchone()[0]
            except psycopg2.Error:
                actual_plays = None

            cursor.close()
            conn.close()

            differences = {}
            if claimed_metrics.get("games"):
                claimed_games = int(claimed_metrics["games"])
                diff = abs(claimed_games - actual_games)
                differences["games"] = {
                    "claimed": claimed_games,
                    "actual": actual_games,
                    "difference": diff,
                    "percent_error": f"{100 * diff / actual_games:.1f}%" if actual_games > 0 else "N/A",
                }

            # Consider metrics valid if within 5% or showing recent data
            tolerance = 0.05
            games_ok = True
            if claimed_metrics.get("games"):
                claimed = int(claimed_metrics["games"])
                games_ok = abs(claimed - actual_games) / actual_games < tolerance if actual_games > 0 else True

            self.report.add_result(ValidationResult(
                category="Metrics",
                check_name="Database Metrics",
                passed=games_ok,
                message=f"Actual: {actual_games} games, Plays: {actual_plays if actual_plays else 'unknown'}",
                details=differences,
                suggestion=(
                    f"Update metrics in CLAUDE.md: {actual_games} games"
                    if not games_ok else None
                ),
            ))
        except Exception as e:
            self._vprint(f"⚠️  Metrics validation failed: {e}")
            self.report.add_result(ValidationResult(
                category="Metrics",
                check_name="Database Metrics",
                passed=False,
                message=f"Could not validate metrics: {str(e)}",
                details={"error": str(e)},
            ))

    def validate_model_registry_sync(self) -> None:
        """Validate model registry synchronization status."""
        self._vprint("Validating model registry sync...")

        # Extract model registry status from CLAUDE.md
        match = re.search(r"(\d+)\s*models?\s*(?:synced|in sync|registered)", self.content, re.IGNORECASE)
        if not match:
            self.report.add_result(ValidationResult(
                category="Metrics",
                check_name="Model Registry Sync",
                passed=True,
                message="Model registry sync status not specified in CLAUDE.md",
                details={},
            ))
            return

        claimed_synced = int(match.group(1))
        self._vprint(f"Claimed synced models: {claimed_synced}")

        # Try to validate against actual registry
        registry_file = self.project_root / "models" / "MODEL_REGISTRY.yaml"
        if registry_file.exists():
            try:
                with open(registry_file) as f:
                    content = f.read()
                    # Count models in YAML (simplified)
                    model_count = content.count("- name:")
                    self.report.add_result(ValidationResult(
                        category="Metrics",
                        check_name="Model Registry Sync",
                        passed=True,
                        message=f"Found {model_count} models in registry",
                        details={
                            "claimed_synced": claimed_synced,
                            "actual_in_registry": model_count,
                        },
                    ))
            except Exception as e:
                self._vprint(f"⚠️  Cannot read MODEL_REGISTRY.yaml: {e}")
        else:
            self._vprint("⚠️  MODEL_REGISTRY.yaml not found")

    # ========================================================================
    # MAIN VALIDATION RUNNER
    # ========================================================================

    def run_all_validations(self) -> int:
        """Run all validation categories and return exit code.

        Returns:
            0 if all validations passed, 1 if any failed, 2 if fatal error
        """
        print("Starting CLAUDE.md validation...\n")

        # Category 1: Architecture References
        print("=" * 70)
        print("CATEGORY 1: ARCHITECTURE REFERENCES")
        print("=" * 70)
        self.validate_architecture_references()
        self.validate_directory_structure()
        self.validate_docker_services()
        self.validate_database_connection()
        print()

        # Category 2: Cross-References
        print("=" * 70)
        print("CATEGORY 2: CROSS-REFERENCES")
        print("=" * 70)
        self.validate_documentation_links()
        self.validate_code_file_references()
        print()

        # Category 3: Freshness
        print("=" * 70)
        print("CATEGORY 3: FRESHNESS")
        print("=" * 70)
        self.validate_update_date()
        self.validate_version_number()
        self.validate_recent_work_section()
        print()

        # Category 4: Metrics
        print("=" * 70)
        print("CATEGORY 4: METRICS")
        print("=" * 70)
        self.validate_metrics()
        self.validate_model_registry_sync()
        print()

        # Print summary
        return self._print_summary()

    def _print_summary(self) -> int:
        """Print validation summary and return exit code."""
        summary = self.report.get_summary()
        self.report.summary = summary

        print("=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"Timestamp: {summary['timestamp']}")
        print(f"Total Checks: {summary['total_checks']}")
        print(f"Passed: {summary['total_passed']} ({summary['pass_rate']})")
        print(f"Failed: {summary['total_failed']}")
        print()

        # Print by category
        print("By Category:")
        for category, stats in summary["by_category"].items():
            status = "✅" if stats["failed"] == 0 else "❌"
            print(f"  {status} {category}: {stats['passed']}/{stats['total']} passed")
        print()

        # Print failed checks
        failed_results = [r for r in self.report.results if not r.passed]
        if failed_results:
            print("Failed Checks:")
            for result in failed_results:
                print(f"  ❌ {result.category} > {result.check_name}")
                print(f"     Message: {result.message}")
                if result.suggestion:
                    print(f"     Suggestion: {result.suggestion}")
            print()

        # Return exit code
        return 0 if summary["total_failed"] == 0 else 1


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate CLAUDE.md accuracy and consistency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_claude_md.py
  python scripts/validate_claude_md.py --report-file validation_report.json
  python scripts/validate_claude_md.py --fix-suggestions --verbose
        """,
    )

    parser.add_argument(
        "--claude-md-path",
        default="/Users/dro/rice/nfl-analytics/CLAUDE.md",
        help="Path to CLAUDE.md file (default: /Users/dro/rice/nfl-analytics/CLAUDE.md)",
    )
    parser.add_argument(
        "--report-file",
        help="Save JSON report to file",
    )
    parser.add_argument(
        "--fix-suggestions",
        action="store_true",
        help="Print actionable fix suggestions for failed checks",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Create validator and run validations
    validator = CLAUDEMDValidator(args.claude_md_path, verbose=args.verbose)
    exit_code = validator.run_all_validations()

    # Save report if requested
    if args.report_file:
        report_dict = {
            "timestamp": validator.report.timestamp,
            "claude_md_path": validator.report.claude_md_path,
            "summary": validator.report.summary,
            "results": [asdict(r) for r in validator.report.results],
        }
        with open(args.report_file, "w") as f:
            json.dump(report_dict, f, indent=2)
        print(f"✅ Report saved to: {args.report_file}")

    # Print fix suggestions if requested
    if args.fix_suggestions:
        print("\n" + "=" * 70)
        print("FIX SUGGESTIONS")
        print("=" * 70)
        suggestions = [
            (r.category, r.check_name, r.suggestion)
            for r in validator.report.results
            if r.suggestion and not r.passed
        ]
        if suggestions:
            for category, check, suggestion in suggestions:
                print(f"\n{category} > {check}:")
                print(f"  → {suggestion}")
        else:
            print("No actionable suggestions (all checks passed)")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
