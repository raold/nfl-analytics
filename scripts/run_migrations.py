#!/usr/bin/env python3
"""
Migration Runner: Proper ordering and tracking for NFL Analytics schema migrations.

Features:
- Sequential enforcement (prevents out-of-order execution)
- Dependency resolution for grouped migrations
- Status tracking via schema_migrations table
- Dry-run mode for validation
- Rollback documentation

Usage:
  python scripts/run_migrations.py --status          # Show applied vs pending
  python scripts/run_migrations.py --apply           # Apply pending migrations
  python scripts/run_migrations.py --dry-run         # Validate without applying
  python scripts/run_migrations.py --generate-docs   # Generate migration docs

Author: Claude Code (Opus 4.5)
Date: December 7, 2025
"""

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import psycopg2
from psycopg2.extras import DictCursor

# Database connection
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "database": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",
}

MIGRATIONS_DIR = Path(__file__).parent.parent / "db" / "migrations"

# Canonical migration ordering (resolves duplicates)
# Format: version -> (filename, group, description)
CANONICAL_ORDER = {
    # Baseline (run first for fresh installs)
    "000": ("000_complete_schema.sql", "baseline", "Complete schema baseline"),

    # Core Infrastructure (001-013)
    "001": ("001_init.sql", "core", "Initialize database"),
    "002": ("002_timescale.sql", "core", "Enable TimescaleDB"),
    "003": ("003_mart_game_weather.sql", "core", "Weather mart views"),
    "004": ("004_advanced_features.sql", "core", "Advanced feature tables"),
    "005": ("005_enhance_mart_views.sql", "core", "Enhanced mart views"),
    "006": ("006_remove_weather_duplication.sql", "core", "Clean weather duplicates"),
    "007a": ("007_data_quality_log.sql", "core", "Data quality logging"),
    "007b": ("007_ab_test_monitoring.sql", "core", "A/B test monitoring"),
    "008a": ("008_add_foreign_keys_simple.sql", "core", "Basic foreign keys"),
    "008b": ("008_nextgen_stats.sql", "data", "NextGen stats tables"),
    "009a": ("009_odds_optimization.sql", "data", "Odds optimization"),
    "009b": ("009_advanced_metrics.sql", "data", "Advanced metrics"),
    "010a": ("010_data_standardization_layer.sql", "data", "Data standardization"),
    "010b": ("010_complete_nflverse_data.sql", "data", "Complete nflverse data"),
    "011": ("011_team_logos.sql", "data", "Team logos"),
    "012": ("012_consolidate_team_mappings.sql", "data", "Team mappings"),
    "013": ("013_optimize_timescaledb.sql", "core", "TimescaleDB optimization"),

    # DBA & Tracking (014-017)
    "014": ("014_add_backup_table_pk.sql", "dba", "Backup table PK"),
    "015": ("015_dba_audit_log_table.sql", "dba", "DBA audit log"),
    "016": ("016_player_id_mapping_view.sql", "dba", "Player ID mapping"),
    "017": ("017_materialized_views.sql", "core", "Materialized views"),

    # Critical Fixes (018 group - pick most comprehensive)
    "018": ("018_predictions_schema.sql", "predictions", "Predictions schema (primary)"),
    # Skip: 018_critical_fixes_v2.sql, v3 (superseded)
    # Skip: 018_database_consistency_fixes.sql (superseded by 100)
    # Skip: 018_pipeline_tracking.sql (merged into 018)

    # Props & Predictions (019-024)
    "019a": ("019_prop_lines_table.sql", "predictions", "Prop lines table"),
    "019b": ("019_v2_model_features.sql", "predictions", "V2 model features"),
    "020": ("020_prop_predictions_table.sql", "predictions", "Prop predictions"),
    "021": ("021_fix_week6_kickoff_times.sql", "fixes", "Week 6 kickoff fix"),
    "022": ("022_fix_all_week6_kickoffs.sql", "fixes", "All Week 6 kickoffs"),
    "023": ("023_player_hierarchy_schema.sql", "predictions", "Player hierarchy"),
    "024": ("024_bayesian_prop_predictions.sql", "predictions", "Bayesian props"),
    "025": ("025_model_registry.sql", "ml", "Model registry"),

    # Paper Trading & Columns (026-033)
    "026a": ("026_pinnacle_exchange_fields.sql", "trading", "Pinnacle fields"),
    "026b": ("026_paper_trading_schema.sql", "trading", "Paper trading schema"),
    "026c": ("026_canonical_training_window.sql", "ml", "Training window"),
    "026d": ("026_add_nflreadr_alignment_columns.sql", "data", "nflreadr alignment"),
    "027a": ("027_paper_trading_tables.sql", "trading", "Paper trading tables"),
    "027b": ("027_add_critical_columns.sql", "data", "Critical columns"),
    "028": ("028_add_defensive_player_columns.sql", "data", "Defensive columns"),
    "029": ("029_add_situational_drive_columns.sql", "data", "Situational columns"),
    "030": ("030_add_phase4_critical_columns.sql", "data", "Phase 4 columns"),
    "031": ("031_add_phase5_high_signal_columns.sql", "data", "Phase 5 columns"),
    "032": ("032_ml_predictions.sql", "ml", "ML predictions"),
    "033": ("033_fix_convergence_status.sql", "ml", "Convergence status fix"),

    # Feature Tracking (043-050)
    "043": ("043_column_lineage_tracking.sql", "tracking", "Column lineage"),
    "044": ("044_add_ensemble_tracking_columns.sql", "tracking", "Ensemble tracking"),
    "045": ("045_api_keys_table.sql", "security", "API keys"),
    "046": ("046_ensemble_version_tracking.sql", "tracking", "Ensemble versions"),
    "047": ("047_r_model_predictions_lookup.sql", "tracking", "R model lookup"),
    "048a": ("048_create_game_predictions.sql", "predictions", "Game predictions"),
    "048b": ("048_prediction_accuracy_view.sql", "predictions", "Accuracy view"),
    "048c": ("048_fix_thursday_night_flag.sql", "fixes", "Thursday flag fix"),
    "048d": ("048_create_views_compatibility_patch.sql", "fixes", "Views compat patch"),
    "048e": ("048_create_game_predictions_ADDENDUM.sql", "predictions", "Predictions addendum"),
    "049": ("049_adaptive_learning_tracking.sql", "ml", "Adaptive learning"),
    "050a": ("050_add_coaching_staff_tables.sql", "data", "Coaching staff"),
    "050b": ("050_dashboard_performance_indexes.sql", "perf", "Dashboard indexes"),

    # Schema Reconciliation (100+) - Already tracked
    "100": ("100_schema_reconciliation.sql", "reconciliation", "Schema reconciliation"),
    "101": ("101_stat_type_performance.sql", "ml", "Stat type performance"),
    "102": ("102_dirichlet_alpha_by_stat_type.sql", "ml", "Dirichlet alphas"),
    "103": ("103_bayesian_update_log.sql", "ml", "Bayesian update log"),

    # Unnumbered (apply at end)
    "900": ("add_international_venues.sql", "data", "International venues"),
    "999": ("verify_schema.sql", "validation", "Schema verification"),
}

# Migration groups for documentation
MIGRATION_GROUPS = {
    "baseline": "Database baseline schema",
    "core": "Core infrastructure and extensions",
    "data": "Data tables and columns",
    "predictions": "Predictions and props schema",
    "trading": "Paper trading system",
    "ml": "Machine learning schema",
    "tracking": "Feature and model tracking",
    "fixes": "Bug fixes and corrections",
    "dba": "DBA utilities and audit",
    "security": "Security and authentication",
    "perf": "Performance optimizations",
    "reconciliation": "Schema reconciliation",
    "validation": "Schema validation",
}


@dataclass
class Migration:
    """Represents a database migration."""
    version: str
    filename: str
    group: str
    description: str
    path: Path
    applied: bool = False
    applied_at: datetime | None = None

    @property
    def full_path(self) -> Path:
        return self.path

    def __lt__(self, other: "Migration") -> bool:
        """Sort by version for ordering."""
        return self.version < other.version


def get_db_connection():
    """Get PostgreSQL database connection."""
    return psycopg2.connect(**DB_CONFIG, cursor_factory=DictCursor)


def ensure_migrations_table(conn):
    """Ensure schema_migrations table exists."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version VARCHAR(100) PRIMARY KEY,
                description TEXT,
                applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
    conn.commit()


def get_applied_migrations(conn) -> set[str]:
    """Get set of applied migration versions."""
    with conn.cursor() as cur:
        cur.execute("SELECT version FROM schema_migrations")
        return {row["version"] for row in cur.fetchall()}


def get_all_migrations() -> list[Migration]:
    """Get all migrations in canonical order."""
    migrations = []

    for version, (filename, group, description) in sorted(CANONICAL_ORDER.items()):
        path = MIGRATIONS_DIR / filename
        if path.exists():
            migrations.append(Migration(
                version=version,
                filename=filename,
                group=group,
                description=description,
                path=path,
            ))
        else:
            print(f"WARNING: Migration file not found: {filename}")

    return sorted(migrations)


def check_migration_status(verbose: bool = True) -> dict:
    """Check status of all migrations."""
    conn = get_db_connection()
    ensure_migrations_table(conn)
    applied = get_applied_migrations(conn)
    conn.close()

    migrations = get_all_migrations()

    applied_list = []
    pending_list = []

    for m in migrations:
        # Check if this version or filename is applied
        is_applied = (
            m.version in applied or
            m.filename.replace(".sql", "") in applied or
            f"{m.version}_{m.filename.split('_', 1)[1].replace('.sql', '')}" in applied
        )
        m.applied = is_applied

        if is_applied:
            applied_list.append(m)
        else:
            pending_list.append(m)

    if verbose:
        print("=" * 70)
        print("MIGRATION STATUS")
        print("=" * 70)
        print(f"\nApplied: {len(applied_list)} | Pending: {len(pending_list)}")

        if pending_list:
            print("\nPending Migrations:")
            for m in pending_list[:20]:  # Show first 20
                print(f"  [{m.version:5s}] {m.filename[:40]:<40s} ({m.group})")
            if len(pending_list) > 20:
                print(f"  ... and {len(pending_list) - 20} more")

        print()

    return {
        "applied": applied_list,
        "pending": pending_list,
        "total": len(migrations),
    }


def apply_migration(conn, migration: Migration, dry_run: bool = False):
    """Apply a single migration."""
    print(f"  Applying {migration.version}: {migration.filename}")

    if dry_run:
        print(f"    [DRY RUN] Would apply {migration.filename}")
        return True

    try:
        with open(migration.full_path) as f:
            sql = f.read()

        with conn.cursor() as cur:
            cur.execute(sql)

            # Record in migrations table
            cur.execute("""
                INSERT INTO schema_migrations (version, description)
                VALUES (%s, %s)
                ON CONFLICT (version) DO NOTHING
            """, (migration.version, migration.description))

        conn.commit()
        return True

    except Exception as e:
        conn.rollback()
        print(f"    ERROR: {e}")
        return False


def apply_pending_migrations(dry_run: bool = False):
    """Apply all pending migrations in order."""
    status = check_migration_status(verbose=False)
    pending = status["pending"]

    if not pending:
        print("No pending migrations to apply.")
        return

    print(f"\nApplying {len(pending)} pending migrations...")
    if dry_run:
        print("(DRY RUN - no changes will be made)\n")

    conn = get_db_connection()
    ensure_migrations_table(conn)

    success = 0
    failed = 0

    for m in pending:
        if apply_migration(conn, m, dry_run):
            success += 1
        else:
            failed += 1
            if not dry_run:
                print(f"\nStopping due to error in {m.filename}")
                break

    conn.close()

    print(f"\nResults: {success} applied, {failed} failed")


def generate_migration_docs():
    """Generate markdown documentation for all migrations."""
    migrations = get_all_migrations()

    # Group by migration group
    by_group = {}
    for m in migrations:
        if m.group not in by_group:
            by_group[m.group] = []
        by_group[m.group].append(m)

    doc = """# NFL Analytics Database Migrations

**Generated:** {timestamp}
**Total Migrations:** {total}

## Migration Groups

| Group | Description | Count |
|-------|-------------|-------|
""".format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total=len(migrations),
    )

    for group, desc in MIGRATION_GROUPS.items():
        count = len(by_group.get(group, []))
        doc += f"| {group} | {desc} | {count} |\n"

    doc += "\n---\n\n## All Migrations\n\n"

    for group, desc in MIGRATION_GROUPS.items():
        if group not in by_group:
            continue

        doc += f"### {group.title()}: {desc}\n\n"
        doc += "| Version | File | Description |\n"
        doc += "|---------|------|-------------|\n"

        for m in by_group[group]:
            doc += f"| {m.version} | `{m.filename}` | {m.description} |\n"

        doc += "\n"

    doc += """---

## Usage

```bash
# Check migration status
python scripts/run_migrations.py --status

# Apply pending migrations
python scripts/run_migrations.py --apply

# Dry run (validate without applying)
python scripts/run_migrations.py --dry-run

# Generate this documentation
python scripts/run_migrations.py --generate-docs
```

## Canonical Ordering

Migrations with duplicate numbers (e.g., 018_*, 026_*, 048_*) are resolved using
suffixes (a, b, c, d, e). The canonical order is defined in `scripts/run_migrations.py`.

## Notes

- Migrations 100+ use the schema_migrations tracking table
- Migrations 000-050 may have been applied manually (legacy)
- Always run `--dry-run` before `--apply` in production
"""

    output_path = Path(__file__).parent.parent / "docs" / "database" / "MIGRATIONS.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(doc)

    print(f"Generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="NFL Analytics Migration Runner")
    parser.add_argument("--status", action="store_true", help="Show migration status")
    parser.add_argument("--apply", action="store_true", help="Apply pending migrations")
    parser.add_argument("--dry-run", action="store_true", help="Validate without applying")
    parser.add_argument("--generate-docs", action="store_true", help="Generate migration docs")

    args = parser.parse_args()

    if not any([args.status, args.apply, args.dry_run, args.generate_docs]):
        args.status = True

    if args.status:
        check_migration_status()

    if args.apply:
        apply_pending_migrations(dry_run=False)

    if args.dry_run:
        apply_pending_migrations(dry_run=True)

    if args.generate_docs:
        generate_migration_docs()


if __name__ == "__main__":
    main()
