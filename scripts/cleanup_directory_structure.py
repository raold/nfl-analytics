#!/usr/bin/env python3
"""
Directory Cleanup Script for NFL Analytics Project
Organizes scattered root-level files into logical directory structure.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

# Root directory
ROOT = Path('/Users/dro/rice/nfl-analytics')

# Define organizational structure
CLEANUP_PLAN = {
    # Python utility scripts → scripts/utils/
    'scripts/utils/': [
        'add_dois.py',
        'audit_bibliography.py',
        'check_notation_consistency.py',
        'clean_bibliography.py',
        'fix_latex_tables.py',
        'fix_references.py',
        'generate_missing_tables.py',
    ],

    # Dissertation-specific scripts → scripts/dissertation/
    'scripts/dissertation/': [
        # (none identified yet - will check in analysis run)
    ],

    # Compute system scripts → scripts/compute/
    'scripts/compute/': [
        'generate_heavy_tasks.py',
        'monitor_and_upgrade.py',
        'run_compute.py',
        'watch_and_restart.sh',
    ],

    # Status/progress documentation → docs/reports/
    'docs/reports/': [
        'DISSERTATION_RESULTS_SUMMARY.md',
        'FINAL_COMPLETION_REPORT.md',
        'FINAL_PRODUCTION_STATUS.md',
        'FIGURE_6_3_COMPLETION.md',
        'IMPROVEMENTS_IMPLEMENTED.md',
        'PHASE_2_COMPLETE.md',
        'PRODUCTION_READY_CHECKLIST.md',
        'REAL_RESULTS_SUMMARY.md',
        'SESSION_2_SUMMARY.md',
        'SESSION_COMPLETION_SUMMARY.md',
        'SPRINT_COMPLETED.md',
        'TABLE_10_3_FIXED.md',
        'THREE_AGENT_SYNTHESIS_REPORT.md',
        'WEATHER_ANALYSIS_COMPLETE.md',
        'WEATHER_INFRASTRUCTURE_ASSESSMENT.md',
    ],

    # Research/reference documentation → docs/research/
    'docs/research/': [
        'ETL_DEVOPS_BEST_PRACTICES_RESEARCH_2025.md',
    ],

    # System documentation → docs/
    'docs/': [
        'AGENTS.md',
        'COMPUTE_SYSTEM.md',
        'REPRODUCIBILITY.md',
    ],

    # Log files → logs/
    'logs/': [
        'cleanup_report.txt',
        'reliability_generation.log',
        'value_monitor.log',
        'worker1_high.log',
        'worker2_high.log',
    ],

    # Database/cache files → data/cache/
    'data/cache/': [
        'compute_queue.db',
        'compute_queue.db-shm',
        'compute_queue.db-wal',
        'dump.rdb',  # Redis dump
    ],

    # Config files → config/
    'config/': [
        'compute_odometer.json',
        'machine_info.json',
    ],
}

# Files to keep at root
KEEP_AT_ROOT = {
    'README.md',
    'requirements.txt',
    'Makefile',
    '.gitignore',
    '.env',
    'docker-compose.yml',
    'pyproject.toml',
    'setup.py',
}


def create_backup():
    """Create timestamped backup of current state."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = ROOT / 'archive' / f'pre_cleanup_backup_{timestamp}'
    backup_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating backup at: {backup_dir}")

    # Backup all files we're about to move
    for dest_dir, files in CLEANUP_PLAN.items():
        for filename in files:
            src = ROOT / filename
            if src.exists():
                dst = backup_dir / filename
                shutil.copy2(src, dst)
                print(f"  Backed up: {filename}")

    return backup_dir


def execute_cleanup(dry_run=True):
    """Execute the cleanup plan."""
    if dry_run:
        print("\n" + "="*60)
        print("DRY RUN - No files will be moved")
        print("="*60 + "\n")
    else:
        backup_dir = create_backup()
        print(f"\nBackup created: {backup_dir}\n")

    moves_executed = []
    files_not_found = []

    for dest_dir, files in CLEANUP_PLAN.items():
        dest_path = ROOT / dest_dir

        # Create destination directory
        if not dry_run:
            dest_path.mkdir(parents=True, exist_ok=True)

        for filename in files:
            src = ROOT / filename
            dst = dest_path / filename

            if src.exists():
                if dry_run:
                    print(f"WOULD MOVE: {filename:40s} → {dest_dir}")
                else:
                    shutil.move(str(src), str(dst))
                    print(f"MOVED: {filename:40s} → {dest_dir}")
                moves_executed.append((filename, dest_dir))
            else:
                files_not_found.append(filename)

    # Report
    print("\n" + "="*60)
    print(f"CLEANUP SUMMARY")
    print("="*60)
    print(f"Files to move: {len(moves_executed)}")
    print(f"Files not found: {len(files_not_found)}")

    if files_not_found:
        print("\nFiles not found (may have been moved already):")
        for f in files_not_found:
            print(f"  - {f}")

    print("\n" + "="*60)
    print("PROPOSED ROOT STRUCTURE AFTER CLEANUP:")
    print("="*60)
    print("""
    /Users/dro/rice/nfl-analytics/
    ├── README.md                 (project overview)
    ├── requirements.txt          (Python dependencies)
    ├── .gitignore
    ├── analysis/                 (dissertation LaTeX + results)
    ├── config/                   (system configuration files)
    ├── data/                     (datasets + cache)
    ├── docs/                     (system documentation)
    │   ├── reports/              (status/progress reports)
    │   └── research/             (research notes)
    ├── etl/                      (data pipelines)
    ├── logs/                     (execution logs)
    ├── models/                   (trained models)
    ├── py/                       (core Python codebase)
    ├── R/                        (R analysis scripts)
    ├── scripts/                  (utility scripts)
    │   ├── compute/              (distributed compute)
    │   ├── dissertation/         (dissertation tooling)
    │   └── utils/                (general utilities)
    └── tests/                    (test suite)
    """)

    return moves_executed, files_not_found


def analyze_remaining_files():
    """Identify any files still at root that aren't in the plan."""
    root_files = [f for f in ROOT.iterdir() if f.is_file() and not f.name.startswith('.')]

    # Files in cleanup plan
    planned_files = set()
    for files in CLEANUP_PLAN.values():
        planned_files.update(files)

    # Additional files to keep at root (discovered during analysis)
    additional_keep = {
        'Dockerfile',
        'pytest.ini',
        'renv.lock',
        'requirements-dev.txt',
        'requirements-lock.txt',
        'uv.lock',
        'setup_packages.R',
        'coverage.xml',
        'cleanup_directory_structure.py',  # This script itself
    }

    # Files to keep
    keep_files = KEEP_AT_ROOT | additional_keep

    # Unaccounted files
    unaccounted = []
    for f in root_files:
        if f.name not in planned_files and f.name not in keep_files:
            unaccounted.append(f.name)

    if unaccounted:
        print("\n" + "="*60)
        print("UNACCOUNTED FILES (not in cleanup plan):")
        print("="*60)
        for f in sorted(unaccounted):
            print(f"  - {f}")
        print("\nThese files may need manual review.")

    return unaccounted


if __name__ == '__main__':
    import sys

    # Default to dry run
    dry_run = '--execute' not in sys.argv

    print("NFL Analytics Directory Cleanup")
    print("="*60)

    # Analyze current state
    print("\nAnalyzing current directory structure...")
    unaccounted = analyze_remaining_files()

    # Execute cleanup
    print("\n")
    moves_executed, files_not_found = execute_cleanup(dry_run=dry_run)

    if dry_run:
        print("\n" + "="*60)
        print("To execute cleanup, run:")
        print("  python cleanup_directory_structure.py --execute")
        print("="*60)
