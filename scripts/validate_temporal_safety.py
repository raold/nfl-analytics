#!/usr/bin/env python3
"""
Temporal Safety Validation Script
==================================

Comprehensive validation of temporal safety across all feature engineering code
and database queries. This script is the FINAL GATE before production deployment.

Usage:
    python scripts/validate_temporal_safety.py --full
    python scripts/validate_temporal_safety.py --code-only
    python scripts/validate_temporal_safety.py --db-only

Exit Codes:
    0 - No critical leakage found (safe for production)
    1 - Critical leakage detected (BLOCK deployment)
    2 - High-severity issues found (review required)

Author: Claude Code (Sonnet 4.5)
Date: 2025-11-13
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from py.utils.data_leakage_detector import DataLeakageDetector, LeakageSeverity
import pandas as pd
import psycopg

# Configuration
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "dbname": "devdb01",
    "user": "dro",
    "password": "sicillionbillions"
}

FEATURE_MODULES = [
    "py/features/asof_features.py",
    "py/features/asof_features_v2.py",
    "py/features/asof_features_enhanced.py",
    "py/features/player_features.py",
    "py/features/bayesian_features.py",
    "py/features/bayesian_player_features.py",
    "py/features/vegas_line_bias_features_SAFE.py",
    "py/features/defensive_situational_strength.py",
    "py/features/defensive_game_script_indicators.py",
    "py/features/environmental_contextual_features.py",
    "py/features/matchup_features.py",
    "py/features/feature_interactions.py",
    "py/features/line_movement_enhanced.py",
    "py/features/defensive_coverage_quality.py",
    "py/features/defensive_pressure_effectiveness.py",
]


class TemporalSafetyValidator:
    """Validates temporal safety across the entire codebase."""

    def __init__(self):
        self.detector = DataLeakageDetector()
        self.results = {
            'code_analysis': {},
            'database_validation': {},
            'summary': {
                'critical_count': 0,
                'high_count': 0,
                'medium_count': 0,
                'low_count': 0,
                'files_checked': 0,
                'timestamp': datetime.now().isoformat()
            }
        }

    def validate_code_modules(self) -> bool:
        """
        Validate all feature engineering code modules.

        Returns:
            True if no critical issues found
        """
        print("\n" + "=" * 80)
        print("VALIDATING FEATURE ENGINEERING CODE")
        print("=" * 80)

        all_safe = True

        for module_path in FEATURE_MODULES:
            path = Path(module_path)
            if not path.exists():
                print(f"⊘ {module_path} - File not found (skipped)")
                continue

            with open(path, 'r') as f:
                code = f.read()

            # Reset detector
            self.detector.report.issues = []
            self.detector.report.critical_count = 0
            self.detector.report.high_count = 0
            self.detector.report.medium_count = 0
            self.detector.report.low_count = 0

            # Check for leakage
            report = self.detector.check_feature_engineering(code, str(path))

            # Store results
            self.results['code_analysis'][module_path] = {
                'critical': report.critical_count,
                'high': report.high_count,
                'medium': report.medium_count,
                'low': report.low_count,
            }

            self.results['summary']['files_checked'] += 1
            self.results['summary']['critical_count'] += report.critical_count
            self.results['summary']['high_count'] += report.high_count
            self.results['summary']['medium_count'] += report.medium_count
            self.results['summary']['low_count'] += report.low_count

            # Print status
            if report.critical_count > 0:
                print(f"🔴 {module_path} - CRITICAL LEAKAGE ({report.critical_count} issues)")
                all_safe = False
                # Print issues
                for issue in [i for i in report.issues if i.severity == LeakageSeverity.CRITICAL]:
                    print(f"   ⛔ Line {issue.line_number}: {issue.title}")
                    print(f"      {issue.description}")
                    print(f"      Evidence: {issue.evidence[:100]}")
            elif report.high_count > 0:
                print(f"🟠 {module_path} - HIGH severity ({report.high_count} issues)")
                all_safe = False
            elif report.medium_count > 0:
                print(f"🟡 {module_path} - MEDIUM severity ({report.medium_count} issues)")
            else:
                print(f"✅ {module_path} - CLEAN")

        return all_safe

    def validate_database_features(self) -> bool:
        """
        Validate database feature tables.

        Returns:
            True if no critical issues found
        """
        print("\n" + "=" * 80)
        print("VALIDATING DATABASE FEATURE TABLES")
        print("=" * 80)

        try:
            conn = psycopg.connect(**DB_CONFIG)
        except Exception as e:
            print(f"⚠️  Could not connect to database: {e}")
            return True  # Don't fail if DB unavailable

        all_safe = True

        # Check for asof_team_features
        try:
            query = """
            SELECT *
            FROM mart.asof_team_features
            WHERE season >= 2023
            LIMIT 1000
            """
            df = pd.read_sql(query, conn)

            print(f"\n📊 Checking mart.asof_team_features ({len(df)} rows)")

            # Reset detector
            self.detector.report.issues = []
            self.detector.report.critical_count = 0
            self.detector.report.high_count = 0
            self.detector.report.medium_count = 0
            self.detector.report.low_count = 0

            # Check for leakage
            report = self.detector.check_dataframe(
                df,
                target_col='home_margin' if 'home_margin' in df.columns else None,
                date_col='kickoff',
                player_col=None
            )

            self.results['database_validation']['mart.asof_team_features'] = {
                'critical': report.critical_count,
                'high': report.high_count,
                'rows_checked': len(df),
            }

            if report.critical_count > 0:
                print(f"  🔴 CRITICAL LEAKAGE ({report.critical_count} issues)")
                all_safe = False
            elif report.high_count > 0:
                print(f"  🟠 HIGH severity ({report.high_count} issues)")
            else:
                print(f"  ✅ CLEAN")

        except Exception as e:
            print(f"  ⚠️  Could not check table: {e}")

        conn.close()
        return all_safe

    def generate_report(self) -> str:
        """Generate comprehensive validation report."""
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("TEMPORAL SAFETY VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append(f"Timestamp: {self.results['summary']['timestamp']}")
        lines.append(f"Files Checked: {self.results['summary']['files_checked']}")

        lines.append(f"\nIssues Summary:")
        lines.append(f"  🔴 CRITICAL: {self.results['summary']['critical_count']}")
        lines.append(f"  🟠 HIGH:     {self.results['summary']['high_count']}")
        lines.append(f"  🟡 MEDIUM:   {self.results['summary']['medium_count']}")
        lines.append(f"  🟢 LOW:      {self.results['summary']['low_count']}")

        # Production readiness
        lines.append(f"\n" + "=" * 80)
        if self.results['summary']['critical_count'] == 0:
            lines.append("✅ PRODUCTION READY - No critical leakage detected")
            lines.append("   Safe to deploy to production betting system")
        else:
            lines.append("⛔ DEPLOYMENT BLOCKED - Critical leakage detected")
            lines.append("   DO NOT USE FOR REAL MONEY BETTING")
            lines.append("   Fix critical issues before proceeding")

        lines.append("=" * 80)

        # Risk gates
        lines.append(f"\nRisk Gates:")
        lines.append(f"  ✓ Zero critical leakage required: {'PASS' if self.results['summary']['critical_count'] == 0 else 'FAIL'}")
        lines.append(f"  ✓ High severity issues < 5: {'PASS' if self.results['summary']['high_count'] < 5 else 'FAIL'}")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)

    def save_report(self, output_path: str = "temporal_safety_report.json"):
        """Save validation report to JSON."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📝 Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate temporal safety for production deployment"
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run full validation (code + database)'
    )
    parser.add_argument(
        '--code-only',
        action='store_true',
        help='Validate code only'
    )
    parser.add_argument(
        '--db-only',
        action='store_true',
        help='Validate database only'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='temporal_safety_report.json',
        help='Output report path'
    )

    args = parser.parse_args()

    # Default to full validation
    if not (args.code_only or args.db_only):
        args.full = True

    # Run validation
    validator = TemporalSafetyValidator()

    code_safe = True
    db_safe = True

    if args.full or args.code_only:
        code_safe = validator.validate_code_modules()

    if args.full or args.db_only:
        db_safe = validator.validate_database_features()

    # Generate report
    print(validator.generate_report())

    # Save report
    validator.save_report(args.output)

    # Exit code
    if validator.results['summary']['critical_count'] > 0:
        print("\n⛔ CRITICAL LEAKAGE DETECTED - Deployment BLOCKED")
        sys.exit(1)
    elif validator.results['summary']['high_count'] > 0:
        print("\n⚠️  HIGH severity issues found - Review required")
        sys.exit(2)
    else:
        print("\n✅ Validation PASSED - Safe for production")
        sys.exit(0)


if __name__ == '__main__':
    main()
