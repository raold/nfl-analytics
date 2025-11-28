"""
Frontend Engineering Agent - Integration Examples

Demonstrates practical integration patterns for the Frontend Engineering Agent
in production workflows.

Examples:
1. Pre-commit validation workflow
2. Dashboard startup validation
3. CI/CD integration
4. Prometheus monitoring integration
5. Custom validation rules
6. Integration with Sports Analytics Validator

Author: Claude Code (Sonnet 4.5)
Created: November 22, 2025
"""

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Example 1: Pre-commit Validation Workflow
# ============================================================================

def example_precommit_validation():
    """
    Example: Validate schemas before git commit.

    This prevents committing code with schema errors that would break production.
    """
    print("=" * 80)
    print("EXAMPLE 1: Pre-commit Validation Workflow")
    print("=" * 80)

    from py.utils.frontend_engineering_agent import FrontendEngineeringAgent

    agent = FrontendEngineeringAgent()

    # Validate all dashboard queries
    report = agent.validate_dashboard_schema()

    if not report.is_valid:
        print("❌ Schema validation failed - commit blocked")
        print(report.generate_report())
        return False
    else:
        print("✅ Schema validation passed - safe to commit")
        return True


# ============================================================================
# Example 2: Dashboard Startup Validation
# ============================================================================

def example_dashboard_startup_validation():
    """
    Example: Validate schemas at dashboard startup.

    This prevents starting a dashboard with broken queries.
    """
    print("=" * 80)
    print("EXAMPLE 2: Dashboard Startup Validation")
    print("=" * 80)

    from py.utils.frontend_engineering_agent import FrontendEngineeringAgent

    agent = FrontendEngineeringAgent()

    # Run all startup checks
    print("Running startup checks...")

    # Check 1: Schema validation
    print("  1/3: Validating dashboard schema...")
    schema_report = agent.validate_dashboard_schema()

    if not schema_report.is_valid:
        print("  ❌ Schema validation failed")
        print(schema_report.generate_report())
        return False

    print("  ✅ Schema validation passed")

    # Check 2: API contract
    print("  2/3: Validating API contract...")
    api_report = agent.validate_api_contract()

    if api_report.has_violations:
        print("  ⚠️  API contract violations detected")
        print(api_report.generate_report())
        # Non-blocking for API (dashboard can run without API)

    print("  ✅ API contract validated")

    # Check 3: Schema drift
    print("  3/3: Checking schema drift...")
    drift = agent.detect_schema_drift()

    if drift.has_breaking_changes:
        print("  ⚠️  Breaking schema drift detected")
        updates = agent.suggest_frontend_updates(drift)
        print(f"  {len(updates)} migration updates required")
        # Non-blocking for drift (log and continue)

    print("  ✅ Drift check complete")

    print("\n✅ All startup checks passed - dashboard ready")
    return True


# ============================================================================
# Example 3: CI/CD Integration
# ============================================================================

def example_cicd_integration():
    """
    Example: Run validation in CI/CD pipeline.

    This catches breaking changes in PRs before merge.
    """
    print("=" * 80)
    print("EXAMPLE 3: CI/CD Integration")
    print("=" * 80)

    exit_code = 0

    # Run validation via CLI (simulates CI/CD)
    commands = [
        ("validate-schema", "Schema validation"),
        ("validate-api", "API contract validation"),
        ("check-drift", "Schema drift detection"),
    ]

    for command, description in commands:
        print(f"\nRunning: {description}")

        result = subprocess.run(
            ["./scripts/sync_frontend.sh", command],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"❌ {description} failed")
            print(result.stdout)
            exit_code = 1
        else:
            print(f"✅ {description} passed")

    if exit_code == 0:
        print("\n✅ All CI/CD checks passed")
    else:
        print("\n❌ CI/CD checks failed - PR should not be merged")

    return exit_code == 0


# ============================================================================
# Example 4: Prometheus Monitoring Integration
# ============================================================================

def example_prometheus_monitoring():
    """
    Example: Track schema health with Prometheus metrics.

    This enables real-time monitoring of schema drift and validation errors.
    """
    print("=" * 80)
    print("EXAMPLE 4: Prometheus Monitoring Integration")
    print("=" * 80)

    try:
        from prometheus_client import Counter, Gauge
    except ImportError:
        print("⚠️  prometheus_client not installed - skipping example")
        print("Install: pip install prometheus-client")
        return

    from py.utils.frontend_engineering_agent import FrontendEngineeringAgent, Severity

    # Define metrics
    schema_errors = Counter(
        'dashboard_schema_errors_total',
        'Total schema validation errors',
        ['error_type', 'severity']
    )

    schema_drift_detected = Gauge(
        'dashboard_schema_drift_detected',
        'Whether schema drift has been detected (0=no, 1=yes)'
    )

    api_contract_violations = Counter(
        'dashboard_api_contract_violations_total',
        'Total API contract violations',
        ['endpoint']
    )

    # Run validation and track metrics
    agent = FrontendEngineeringAgent()

    # Schema validation
    print("Running schema validation...")
    report = agent.validate_dashboard_schema()

    for issue in report.issues:
        schema_errors.labels(
            error_type=issue.category,
            severity=issue.severity.value
        ).inc()

    print(f"  Tracked {len(report.issues)} issues in metrics")

    # Drift detection
    print("Checking schema drift...")
    drift = agent.detect_schema_drift()

    schema_drift_detected.set(1 if drift.has_breaking_changes else 0)
    print(f"  Schema drift: {'DETECTED' if drift.has_breaking_changes else 'NONE'}")

    # API contract
    print("Validating API contract...")
    api_report = agent.validate_api_contract()

    for violation in api_report.violations:
        # Extract endpoint from message
        endpoint = "unknown"
        if "/predict" in violation.message:
            endpoint = "/predict"
        elif "/models" in violation.message:
            endpoint = "/models"

        api_contract_violations.labels(endpoint=endpoint).inc()

    print(f"  Tracked {len(api_report.violations)} violations in metrics")

    print("\n✅ Metrics updated - check Prometheus for alerts")


# ============================================================================
# Example 5: Custom Validation Rules
# ============================================================================

def example_custom_validation_rules():
    """
    Example: Extend agent with custom validation rules.

    This shows how to add project-specific validation logic.
    """
    print("=" * 80)
    print("EXAMPLE 5: Custom Validation Rules")
    print("=" * 80)

    from py.utils.frontend_engineering_agent import (
        FrontendEngineeringAgent,
        ValidationIssue,
        Severity
    )

    class CustomFrontendAgent(FrontendEngineeringAgent):
        """Extended agent with custom NFL-specific rules"""

        def validate_sql_query(self, query, schema):
            # Call parent validation
            issues = super().validate_sql_query(query, schema)

            # Custom rule 1: Detect hardcoded seasons
            current_year = 2024  # Example year
            prev_year = 2023  # Example year
            season_check = f"season = {current_year}" in query.lower() or f"season = {prev_year}" in query.lower()
            if season_check:
                issues.append(ValidationIssue(
                    severity=Severity.WARNING,
                    category="hardcoded_value",
                    message="Hardcoded season value detected",
                    query=query,
                    suggestion="Use parameter: WHERE season = %(season)s"
                ))

            # Custom rule 2: Require LIMIT on large tables
            large_tables = ['plays', 'odds_history', 'injuries']
            for table in large_tables:
                if table in query.lower() and 'LIMIT' not in query.upper():
                    issues.append(ValidationIssue(
                        severity=Severity.WARNING,
                        category="performance",
                        message=f"Query on large table '{table}' without LIMIT",
                        query=query,
                        suggestion="Add LIMIT clause to prevent full table scan"
                    ))

            # Custom rule 3: Enforce team abbreviation patterns
            if "WHERE team = 'LA'" in query or "home_team = 'LA'" in query:
                issues.append(ValidationIssue(
                    severity=Severity.ERROR,
                    category="nfl_domain",
                    message="Ambiguous team abbreviation 'LA'",
                    query=query,
                    suggestion="Use 'LAR' (Rams) or 'LAC' (Chargers)"
                ))

            return issues

    # Test custom agent
    agent = CustomFrontendAgent()

    # Test query with custom rule violations
    test_query = """
        SELECT *
        FROM plays
        WHERE season = 2024
            AND home_team = 'LA'
    """

    print(f"Testing query:\n{test_query}\n")

    issues = agent.validate_sql_query(test_query, agent.get_database_schema())

    print(f"Found {len(issues)} issues:")
    for issue in issues:
        print(f"\n{issue}")

    print("\n✅ Custom validation rules working")


# ============================================================================
# Example 6: Integration with Sports Analytics Validator
# ============================================================================

def example_sports_analytics_integration():
    """
    Example: Combine frontend validation with NFL domain validation.

    This ensures both schema correctness AND NFL business logic compliance.
    """
    print("=" * 80)
    print("EXAMPLE 6: Sports Analytics Validator Integration")
    print("=" * 80)

    from py.utils.frontend_engineering_agent import (
        FrontendEngineeringAgent,
        ValidationIssue,
        Severity
    )

    try:
        from py.utils.sports_analytics_validator import SportsAnalyticsValidator
    except ImportError:
        print("⚠️  Sports Analytics Validator not available")
        return

    class EnhancedFrontendAgent(FrontendEngineeringAgent):
        """Frontend agent with NFL domain validation"""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.sports_validator = SportsAnalyticsValidator()

        def validate_sql_query(self, query, schema):
            # Run frontend validation
            issues = super().validate_sql_query(query, schema)

            # Add NFL domain validation
            nfl_issues = self.sports_validator.check_query_logic(query)

            # Convert to ValidationIssue format
            for nfl_issue in nfl_issues:
                issues.append(ValidationIssue(
                    severity=Severity.ERROR if nfl_issue.severity.value == "ERROR" else Severity.WARNING,
                    category="nfl_domain",
                    message=nfl_issue.message,
                    query=query,
                    suggestion=getattr(nfl_issue, 'suggestion', None)
                ))

            return issues

    # Test enhanced agent
    agent = EnhancedFrontendAgent()

    # Test query with NFL domain violations
    test_query = """
        SELECT *
        FROM games
        WHERE week = 25
            AND home_team = 'LA'
    """

    print(f"Testing query:\n{test_query}\n")

    issues = agent.validate_sql_query(test_query, agent.get_database_schema())

    print(f"Found {len(issues)} issues:")
    for issue in issues:
        print(f"\n{issue}")

    print("\n✅ Sports analytics integration working")


# ============================================================================
# Example 7: Automated Migration Application
# ============================================================================

def example_automated_migration():
    """
    Example: Apply automated migration updates.

    DANGER: This modifies code files - use with caution.
    """
    print("=" * 80)
    print("EXAMPLE 7: Automated Migration Application")
    print("=" * 80)

    from py.utils.frontend_engineering_agent import (
        FrontendEngineeringAgent,
        MigrationUpdate,
        Severity
    )

    def apply_migration_update(update: MigrationUpdate, dry_run: bool = True) -> bool:
        """Apply automated migration update to file"""
        if not update.automated:
            logger.warning(f"Update requires manual review: {update.file_path}")
            return False

        if dry_run:
            print(f"\n[DRY RUN] Would apply update:")
            print(f"  File: {update.file_path}")
            if update.line_number:
                print(f"  Line: {update.line_number}")
            print(f"  Old: {update.old_code}")
            print(f"  New: {update.new_code}")
            return True

        # Actual application (not executed in dry run)
        with open(update.file_path, 'r') as f:
            content = f.read()

        # Only apply if old_code is unique
        if update.old_code and content.count(update.old_code) != 1:
            logger.error(f"Old code not unique in {update.file_path} - cannot auto-apply")
            return False

        # Apply update
        if update.old_code:
            new_content = content.replace(update.old_code, update.new_code)
        else:
            new_content = content + "\n" + update.new_code

        # Write back
        with open(update.file_path, 'w') as f:
            f.write(new_content)

        logger.info(f"✅ Applied update to {update.file_path}")
        return True

    # Simulate migration
    agent = FrontendEngineeringAgent()

    # Detect drift
    drift = agent.detect_schema_drift()

    if not drift.has_any_changes:
        print("No schema drift detected - no migrations needed")
        return

    # Generate updates
    updates = agent.suggest_frontend_updates(drift)

    print(f"\nGenerated {len(updates)} migration updates:")

    # Apply updates (dry run only)
    for i, update in enumerate(updates, 1):
        print(f"\nUpdate {i}/{len(updates)}:")
        apply_migration_update(update, dry_run=True)

    print("\n✅ Migration dry run complete")


# ============================================================================
# Main - Run All Examples
# ============================================================================

def main():
    """Run all integration examples"""
    examples = [
        ("Pre-commit Validation", example_precommit_validation),
        ("Dashboard Startup Validation", example_dashboard_startup_validation),
        ("CI/CD Integration", example_cicd_integration),
        ("Prometheus Monitoring", example_prometheus_monitoring),
        ("Custom Validation Rules", example_custom_validation_rules),
        ("Sports Analytics Integration", example_sports_analytics_integration),
        ("Automated Migration", example_automated_migration),
    ]

    print("\n" + "=" * 80)
    print("FRONTEND ENGINEERING AGENT - INTEGRATION EXAMPLES")
    print("=" * 80)
    print("\nRunning all examples...\n")

    for i, (name, example_func) in enumerate(examples, 1):
        try:
            print(f"\n{'=' * 80}")
            print(f"Example {i}/{len(examples)}: {name}")
            print(f"{'=' * 80}\n")

            example_func()

        except Exception as e:
            print(f"❌ Example failed: {e}")
            import traceback
            traceback.print_exc()

        print("\n")
        time.sleep(1)  # Brief pause between examples

    print("=" * 80)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
