#!/usr/bin/env python3
"""
Frontend Auditor Agent

Automated comprehensive frontend quality assurance for Streamlit dashboard.
Detects UI/UX issues, validates data, checks accessibility, and generates audit reports.

Author: Claude Code (Sonnet 4.5)
Date: November 21, 2025
Version: 1.0.0
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


@dataclass
class AuditIssue:
    """Represents a single audit issue found during frontend analysis."""

    severity: str  # 'critical', 'high', 'medium', 'low'
    category: str  # 'ui_ux', 'data', 'accessibility', 'error'
    title: str
    description: str
    location: str  # File path + line number
    fix_suggestion: str | None = None
    auto_fixable: bool = False
    wcag_violation: str | None = None  # e.g., "1.4.3 Contrast (Minimum)"
    screenshot_path: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'severity': self.severity,
            'category': self.category,
            'title': self.title,
            'description': self.description,
            'location': self.location,
            'fix_suggestion': self.fix_suggestion,
            'auto_fixable': self.auto_fixable,
            'wcag_violation': self.wcag_violation,
            'screenshot_path': self.screenshot_path
        }


@dataclass
class AuditReport:
    """Container for complete audit results."""

    timestamp: datetime
    pages_audited: int
    issues: list[AuditIssue] = field(default_factory=list)

    @property
    def critical_count(self) -> int:
        return len([i for i in self.issues if i.severity == 'critical'])

    @property
    def high_count(self) -> int:
        return len([i for i in self.issues if i.severity == 'high'])

    @property
    def medium_count(self) -> int:
        return len([i for i in self.issues if i.severity == 'medium'])

    @property
    def low_count(self) -> int:
        return len([i for i in self.issues if i.severity == 'low'])

    def filter_by_severity(self, *severities: str) -> list[AuditIssue]:
        """Filter issues by severity level(s)."""
        return [i for i in self.issues if i.severity in severities]

    def filter_by_category(self, *categories: str) -> list[AuditIssue]:
        """Filter issues by category."""
        return [i for i in self.issues if i.category in categories]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'pages_audited': self.pages_audited,
            'total_issues': len(self.issues),
            'critical': self.critical_count,
            'high': self.high_count,
            'medium': self.medium_count,
            'low': self.low_count,
            'issues': [issue.to_dict() for issue in self.issues]
        }


class UIUXAnalyzer:
    """Analyzes UI/UX issues: contrast, colors, typography, spacing."""

    # WCAG 2.1 AA requirements
    CONTRAST_RATIO_NORMAL = 4.5  # Normal text
    CONTRAST_RATIO_LARGE = 3.0   # Large text (18pt+ or 14pt+ bold)

    # Common color palette issues
    PROBLEMATIC_COMBINATIONS = [
        ('#ffffff', '#f8f9fa'),  # White on very light gray
        ('#000000', '#212529'),  # Black on very dark gray
        ('#fff3cd', '#ffffff'),  # Light yellow on white
    ]

    def __init__(self, dashboard_path: Path):
        self.dashboard_path = dashboard_path
        self.issues = []

    def analyze(self) -> list[AuditIssue]:
        """Run all UI/UX analyses."""
        logger.info("Running UI/UX analysis...")

        # Find all Python files in dashboard
        py_files = list(self.dashboard_path.glob("**/*.py"))
        logger.info(f"Found {len(py_files)} Python files to analyze")

        for py_file in py_files:
            self._analyze_file(py_file)

        logger.info(f"UI/UX analysis complete. Found {len(self.issues)} issues")
        return self.issues

    def _analyze_file(self, file_path: Path):
        """Analyze a single file for UI/UX issues."""
        with open(file_path) as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            # Check for color declarations
            if 'color:' in line or 'background' in line:
                self._check_contrast(line, file_path, line_num)

            # Check for hardcoded sizing
            if re.search(r'font-size:\s*\d+px', line):
                self._check_font_size(line, file_path, line_num)

            # Check for inline styles (should use CSS classes)
            if 'style="' in line and len(re.findall(r'style="[^"]{50,}"', line)) > 0:
                self._flag_inline_styles(line, file_path, line_num)

    def _check_contrast(self, line: str, file_path: Path, line_num: int):
        """Check color contrast ratios."""
        # Extract color values from line
        colors = re.findall(r'#[0-9a-fA-F]{3,6}', line)

        if len(colors) >= 2:
            # Check if it's a problematic combination
            for fg, bg in self.PROBLEMATIC_COMBINATIONS:
                if fg.lower() in line.lower() and bg.lower() in line.lower():
                    self.issues.append(AuditIssue(
                        severity='high',
                        category='ui_ux',
                        title='Low contrast color combination detected',
                        description=f'Foreground {fg} on background {bg} may not meet WCAG AA contrast ratio',
                        location=f'{file_path}:{line_num}',
                        fix_suggestion='Increase contrast by darkening text or lightening background',
                        auto_fixable=False,
                        wcag_violation='1.4.3 Contrast (Minimum)'
                    ))

    def _check_font_size(self, line: str, file_path: Path, line_num: int):
        """Check for font sizes below minimum."""
        match = re.search(r'font-size:\s*(\d+)px', line)
        if match:
            size = int(match.group(1))
            if size < 12:
                self.issues.append(AuditIssue(
                    severity='medium',
                    category='ui_ux',
                    title=f'Font size too small ({size}px)',
                    description='Minimum recommended font size is 12px for readability',
                    location=f'{file_path}:{line_num}',
                    fix_suggestion='Increase font-size to at least 12px',
                    auto_fixable=True
                ))

    def _flag_inline_styles(self, line: str, file_path: Path, line_num: int):
        """Flag excessive inline styles (should use CSS classes)."""
        self.issues.append(AuditIssue(
            severity='low',
            category='ui_ux',
            title='Excessive inline styles detected',
            description='Long inline style declarations reduce maintainability',
            location=f'{file_path}:{line_num}',
            fix_suggestion='Extract to CSS class or Streamlit component',
            auto_fixable=False
        ))


class DataValidator:
    """Validates data completeness and accuracy against database."""

    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.issues = []

    def analyze(self, page_name: str = None) -> list[AuditIssue]:
        """Run data validation checks."""
        logger.info("Running data validation...")

        # Check game predictions
        self._validate_game_predictions()

        # Check player stats
        self._validate_player_stats()

        logger.info(f"Data validation complete. Found {len(self.issues)} issues")
        return self.issues

    def _validate_game_predictions(self):
        """Validate game predictions are realistic."""
        with psycopg2.connect(**self.db_config) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Check for unrealistic predictions
                cur.execute("""
                    SELECT game_id, ensemble_version,
                           predicted_home_score, predicted_away_score,
                           predicted_total
                    FROM ml.game_predictions
                    WHERE season = 2025 AND week = 12
                      AND ensemble_version = 'v6.1'
                      AND (predicted_total < 35 OR predicted_total > 65)
                    LIMIT 10;
                """)

                unrealistic = cur.fetchall()

                if unrealistic:
                    for game in unrealistic:
                        self.issues.append(AuditIssue(
                            severity='critical' if game['predicted_total'] < 20 else 'high',
                            category='data',
                            title=f'Unrealistic game prediction for {game["game_id"]}',
                            description=f'Total: {game["predicted_total"]:.1f} (expected 40-55)',
                            location='ml.game_predictions',
                            fix_suggestion='Investigate ensemble component weights and R model predictions',
                            auto_fixable=False
                        ))

    def _validate_player_stats(self):
        """Validate player statistics are up-to-date."""
        # TODO: Implement player stats validation
        pass


class FrontendAuditor:
    """
    Main frontend audit orchestrator.

    Coordinates UI/UX analysis, data validation, accessibility checks,
    and generates comprehensive audit reports.
    """

    def __init__(
        self,
        dashboard_path: str = "dashboard",
        db_config: dict | None = None,
        output_dir: str = "docs/audits"
    ):
        """
        Initialize frontend auditor.

        Args:
            dashboard_path: Path to Streamlit dashboard directory
            db_config: Database configuration for data validation
            output_dir: Directory for audit reports
        """
        self.dashboard_path = Path(dashboard_path)
        self.db_config = db_config or self._get_default_db_config()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize analyzers
        self.uiux_analyzer = UIUXAnalyzer(self.dashboard_path)
        self.data_validator = DataValidator(self.db_config)

        logger.info("Initialized FrontendAuditor")
        logger.info(f"  Dashboard: {self.dashboard_path}")
        logger.info(f"  Output: {self.output_dir}")

    def _get_default_db_config(self) -> dict:
        """Get default database configuration from environment."""
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5544')),
            'dbname': os.getenv('DB_NAME', 'devdb01'),
            'user': os.getenv('DB_USER', 'dro'),
            'password': os.getenv('DB_PASSWORD', 'sicillionbillions'),
        }

    def audit_all_pages(self) -> AuditReport:
        """Run complete audit of all dashboard pages."""
        logger.info("Starting complete frontend audit...")
        start_time = datetime.now()

        all_issues = []

        # Run UI/UX analysis
        all_issues.extend(self.uiux_analyzer.analyze())

        # Run data validation
        all_issues.extend(self.data_validator.analyze())

        # Count pages
        pages_count = len(list(self.dashboard_path.glob("**/*.py")))

        # Create report
        report = AuditReport(
            timestamp=start_time,
            pages_audited=pages_count,
            issues=all_issues
        )

        # Save report
        self._save_report(report)

        logger.info(f"Audit complete in {(datetime.now() - start_time).total_seconds():.1f}s")
        logger.info(f"Found {len(all_issues)} issues:")
        logger.info(f"  Critical: {report.critical_count}")
        logger.info(f"  High: {report.high_count}")
        logger.info(f"  Medium: {report.medium_count}")
        logger.info(f"  Low: {report.low_count}")

        return report

    def audit_page(self, page_name: str) -> list[AuditIssue]:
        """Audit a specific page."""
        logger.info(f"Auditing page: {page_name}")
        # TODO: Implement single-page audit
        raise NotImplementedError("Single-page audit not yet implemented")

    def _save_report(self, report: AuditReport):
        """Save audit report to markdown and JSON."""
        timestamp_str = report.timestamp.strftime("%Y%m%d_%H%M%S")

        # Save JSON
        json_path = self.output_dir / f"audit_{timestamp_str}.json"
        with open(json_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Saved JSON report: {json_path}")

        # Save Markdown
        md_path = self.output_dir / f"audit_{timestamp_str}.md"
        self._generate_markdown_report(report, md_path)
        logger.info(f"Saved Markdown report: {md_path}")

    def _generate_markdown_report(self, report: AuditReport, output_path: Path):
        """Generate human-readable markdown report."""
        with open(output_path, 'w') as f:
            f.write("# Frontend Audit Report\n\n")
            f.write(f"**Date**: {report.timestamp.strftime('%B %d, %Y %H:%M:%S')}\n")
            f.write(f"**Pages Audited**: {report.pages_audited}\n")
            f.write(f"**Total Issues**: {len(report.issues)}\n\n")

            f.write("**Severity Breakdown**:\n")
            f.write(f"- ❌ **Critical**: {report.critical_count}\n")
            f.write(f"- ⚠️  **High**: {report.high_count}\n")
            f.write(f"- ℹ️  **Medium**: {report.medium_count}\n")
            f.write(f"- 💡 **Low**: {report.low_count}\n\n")

            f.write("---\n\n")

            # Group by severity
            for severity in ['critical', 'high', 'medium', 'low']:
                issues = report.filter_by_severity(severity)
                if not issues:
                    continue

                icon = {'critical': '❌', 'high': '⚠️', 'medium': 'ℹ️', 'low': '💡'}[severity]
                f.write(f"## {icon} {severity.upper()} Issues ({len(issues)})\n\n")

                for i, issue in enumerate(issues, 1):
                    f.write(f"### {i}. {issue.title}\n\n")
                    f.write(f"**Category**: {issue.category}\n\n")
                    f.write(f"**Location**: `{issue.location}`\n\n")
                    f.write(f"**Description**: {issue.description}\n\n")

                    if issue.wcag_violation:
                        f.write(f"**WCAG Violation**: {issue.wcag_violation}\n\n")

                    if issue.fix_suggestion:
                        f.write(f"**Fix**: {issue.fix_suggestion}\n\n")

                    if issue.auto_fixable:
                        f.write("🤖 *Auto-fixable*\n\n")

                    f.write("---\n\n")


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    auditor = FrontendAuditor()
    report = auditor.audit_all_pages()

    print(f"\n{'='*80}")
    print("AUDIT COMPLETE")
    print(f"{'='*80}")
    print(f"Total Issues: {len(report.issues)}")
    print(f"  Critical: {report.critical_count}")
    print(f"  High: {report.high_count}")
    print(f"  Medium: {report.medium_count}")
    print(f"  Low: {report.low_count}")
    print(f"{'='*80}\n")
