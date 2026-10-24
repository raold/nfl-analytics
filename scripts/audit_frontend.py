#!/usr/bin/env python3
"""
Frontend Audit CLI

Command-line interface for the Frontend Auditor Agent.

Usage:
    python scripts/audit_frontend.py                    # Full audit
    python scripts/audit_frontend.py --page "System Status"  # Specific page
    python scripts/audit_frontend.py --severity critical,high  # Filter by severity
    python scripts/audit_frontend.py --ci --fail-on critical  # CI/CD mode

Author: Claude Code (Sonnet 4.5)
Date: November 21, 2025
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nfl_analytics.agents.frontend_auditor import FrontendAuditor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Frontend Audit Agent - Automated UI/UX, data, and accessibility checks"
    )
    parser.add_argument("--page", help="Audit specific page only")
    parser.add_argument(
        "--severity", help="Filter by severity (comma-separated: critical,high,medium,low)"
    )
    parser.add_argument("--ci", action="store_true", help="CI/CD mode - machine-readable output")
    parser.add_argument(
        "--fail-on", help="Fail if issues of this severity found (critical, high, medium, low)"
    )
    parser.add_argument(
        "--output-dir", default="docs/audits", help="Output directory for audit reports"
    )

    args = parser.parse_args()

    # Initialize auditor
    auditor = FrontendAuditor(output_dir=args.output_dir)

    # Run audit
    if args.page:
        logger.info(f"Auditing specific page: {args.page}")
        auditor.audit_page(args.page)
        report = None  # TODO: Create report from issues
    else:
        logger.info("Running full frontend audit...")
        report = auditor.audit_all_pages()

    # Filter by severity if requested
    if args.severity and report:
        severities = [s.strip() for s in args.severity.split(",")]
        filtered_issues = report.filter_by_severity(*severities)
        logger.info(f"Filtered to {len(filtered_issues)} issues with severity: {severities}")
    else:
        filtered_issues = report.issues if report else []

    # CI mode output
    if args.ci:
        print(f"::set-output name=total_issues::{len(filtered_issues)}")
        print(f"::set-output name=critical::{report.critical_count if report else 0}")
        print(f"::set-output name=high::{report.high_count if report else 0}")

    # Check fail condition
    if args.fail_on and report:
        fail_severities = [s.strip() for s in args.fail_on.split(",")]
        fail_issues = report.filter_by_severity(*fail_severities)

        if fail_issues:
            logger.error(f"AUDIT FAILED: Found {len(fail_issues)} {args.fail_on} severity issues")
            for issue in fail_issues[:5]:  # Show first 5
                logger.error(f"  - {issue.title} ({issue.location})")
            return 1

    logger.info("✅ Audit complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
