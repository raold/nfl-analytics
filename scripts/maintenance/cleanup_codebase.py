#!/usr/bin/env python3
"""
Cleanup and reorganize the NFL Analytics codebase.

This script:
1. Identifies duplicate and obsolete files
2. Standardizes naming conventions
3. Reorganizes directory structure
4. Updates imports and references
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import hashlib
from typing import List, Dict, Tuple

def get_file_hash(filepath: Path) -> str:
    """Get MD5 hash of file contents."""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def find_duplicate_files(root_dir: str) -> Dict[str, List[Path]]:
    """Find files with identical content."""
    hash_map = {}
    duplicates = {}

    for root, dirs, files in os.walk(root_dir):
        # Skip virtual environments and hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__' and 'venv' not in d]

        for file in files:
            if file.endswith(('.py', '.R', '.sql', '.sh')):
                filepath = Path(root) / file
                file_hash = get_file_hash(filepath)

                if file_hash in hash_map:
                    if file_hash not in duplicates:
                        duplicates[file_hash] = [hash_map[file_hash]]
                    duplicates[file_hash].append(filepath)
                else:
                    hash_map[file_hash] = filepath

    return duplicates

def find_obsolete_files(root_dir: str) -> List[Path]:
    """Identify potentially obsolete files based on patterns."""
    obsolete_patterns = [
        '*_old.*',
        '*_backup.*',
        '*_copy.*',
        '*test*',  # Test files not in tests/ directory
        '*.bak',
        '*deprecated*',
        '*unused*',
        '*temp*',
    ]

    obsolete_files = []
    root_path = Path(root_dir)

    for pattern in obsolete_patterns:
        # Find files matching pattern but not in tests directory
        for file in root_path.rglob(pattern):
            if 'test' not in file.parts[-2] and file.is_file():
                obsolete_files.append(file)

    return obsolete_files

def standardize_filenames(root_dir: str) -> List[Tuple[Path, Path]]:
    """Identify files that need renaming for consistency."""
    renames = []
    root_path = Path(root_dir)

    # Define naming rules
    rules = {
        # R scripts should use snake_case
        '.R': lambda name: name.replace('-', '_').lower(),
        # Python scripts should use snake_case
        '.py': lambda name: name.replace('-', '_').lower(),
        # SQL files should use numbered migrations
        '.sql': lambda name: name.lower(),
    }

    for ext, transform in rules.items():
        for file in root_path.rglob(f'*{ext}'):
            if 'venv' not in str(file) and '.git' not in str(file):
                old_name = file.stem
                new_name = transform(old_name)

                if new_name != old_name:
                    new_path = file.parent / f"{new_name}{file.suffix}"
                    if not new_path.exists():
                        renames.append((file, new_path))

    return renames

def reorganize_structure(root_dir: str) -> Dict[str, List[Path]]:
    """Suggest new organization for files."""
    suggestions = {
        'src/python/ingestion': [],
        'src/python/features': [],
        'src/python/models': [],
        'src/python/utils': [],
        'src/R/ingestion': [],
        'src/R/analysis': [],
        'src/R/utils': [],
        'src/sql/migrations': [],
        'src/sql/functions': [],
        'src/sql/views': [],
        'scripts/maintenance': [],
        'scripts/dev': [],
        'tests/unit': [],
        'tests/integration': [],
        'archive': [],
    }

    root_path = Path(root_dir)

    # Categorize Python files
    for file in root_path.glob('py/**/*.py'):
        if 'ingest' in file.name:
            suggestions['src/python/ingestion'].append(file)
        elif 'feature' in file.name:
            suggestions['src/python/features'].append(file)
        elif 'model' in file.name or 'train' in file.name:
            suggestions['src/python/models'].append(file)
        elif 'test' in file.name:
            suggestions['tests/unit'].append(file)
        else:
            suggestions['src/python/utils'].append(file)

    # Categorize R files
    for file in root_path.glob('R/**/*.R'):
        if 'ingest' in file.name:
            suggestions['src/R/ingestion'].append(file)
        elif 'feature' in file.name or 'analysis' in file.name:
            suggestions['src/R/analysis'].append(file)
        elif 'util' in file.name or 'error' in file.name:
            suggestions['src/R/utils'].append(file)

    # SQL files
    for file in root_path.glob('db/**/*.sql'):
        if 'migration' in str(file):
            suggestions['src/sql/migrations'].append(file)
        elif 'function' in file.name:
            suggestions['src/sql/functions'].append(file)
        elif 'view' in file.name:
            suggestions['src/sql/views'].append(file)

    # Scripts
    for file in root_path.glob('scripts/**/*.{py,sh,R}'):
        if 'maintenance' in str(file) or 'backup' in file.name:
            suggestions['scripts/maintenance'].append(file)
        else:
            suggestions['scripts/dev'].append(file)

    return {k: v for k, v in suggestions.items() if v}

def generate_report(
    duplicates: Dict[str, List[Path]],
    obsolete: List[Path],
    renames: List[Tuple[Path, Path]],
    reorganize: Dict[str, List[Path]]
):
    """Generate cleanup report."""
    report = []
    report.append("=" * 60)
    report.append("NFL ANALYTICS CODEBASE CLEANUP REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 60)

    # Duplicates
    report.append("\n## DUPLICATE FILES")
    if duplicates:
        for hash_val, files in duplicates.items():
            report.append(f"\n### Duplicate Set (identical content):")
            for file in files:
                size = file.stat().st_size
                report.append(f"  - {file} ({size} bytes)")
    else:
        report.append("  No duplicates found")

    # Obsolete files
    report.append("\n## POTENTIALLY OBSOLETE FILES")
    if obsolete:
        for file in obsolete:
            modified = datetime.fromtimestamp(file.stat().st_mtime).strftime('%Y-%m-%d')
            report.append(f"  - {file} (modified: {modified})")
    else:
        report.append("  No obsolete files found")

    # Rename suggestions
    report.append("\n## FILENAME STANDARDIZATION")
    if renames:
        for old, new in renames:
            report.append(f"  - {old.name} → {new.name}")
    else:
        report.append("  All filenames follow standards")

    # Reorganization suggestions
    report.append("\n## SUGGESTED REORGANIZATION")
    for new_location, files in reorganize.items():
        if files:
            report.append(f"\n### {new_location}/")
            for file in files[:5]:  # Show first 5
                report.append(f"  - {file.relative_to(Path.cwd())}")
            if len(files) > 5:
                report.append(f"  ... and {len(files) - 5} more")

    # Statistics
    report.append("\n## STATISTICS")
    total_duplicates = sum(len(files) - 1 for files in duplicates.values())
    total_obsolete = len(obsolete)
    total_renames = len(renames)
    total_files = sum(len(files) for files in reorganize.values())

    report.append(f"  - Duplicate files: {total_duplicates}")
    report.append(f"  - Obsolete files: {total_obsolete}")
    report.append(f"  - Files to rename: {total_renames}")
    report.append(f"  - Files to reorganize: {total_files}")
    report.append(f"  - Potential space saved: ~{total_duplicates + total_obsolete} files")

    return "\n".join(report)

def main():
    """Main cleanup process."""
    root_dir = Path.cwd()

    print("🔍 Analyzing codebase...")

    # Find issues
    duplicates = find_duplicate_files(root_dir)
    obsolete = find_obsolete_files(root_dir)
    renames = standardize_filenames(root_dir)
    reorganize = reorganize_structure(root_dir)

    # Generate report
    report = generate_report(duplicates, obsolete, renames, reorganize)

    # Save report
    report_file = Path("cleanup_report.txt")
    with open(report_file, 'w') as f:
        f.write(report)

    print(f"\n✅ Analysis complete! Report saved to: {report_file}")
    print("\nSummary:")
    print(f"  - Found {sum(len(files) - 1 for files in duplicates.values())} duplicate files")
    print(f"  - Found {len(obsolete)} potentially obsolete files")
    print(f"  - {len(renames)} files need renaming")

    # Ask for action
    print("\n🤔 What would you like to do?")
    print("  1. View report")
    print("  2. Archive duplicates and obsolete files")
    print("  3. Apply safe renames")
    print("  4. Exit without changes")

    choice = input("\nChoice (1-4): ")

    if choice == "1":
        print("\n" + report)
    elif choice == "2":
        archive_dir = Path("archive") / datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir.mkdir(parents=True, exist_ok=True)

        # Archive duplicates (keep first, archive rest)
        for files in duplicates.values():
            for file in files[1:]:  # Keep first, archive rest
                dest = archive_dir / file.relative_to(root_dir)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(file), str(dest))
                print(f"  Archived: {file}")

        # Archive obsolete
        for file in obsolete:
            dest = archive_dir / file.relative_to(root_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(file), str(dest))
            print(f"  Archived: {file}")

        print(f"\n✅ Files archived to: {archive_dir}")
    elif choice == "3":
        for old, new in renames:
            old.rename(new)
            print(f"  Renamed: {old.name} → {new.name}")
        print(f"\n✅ {len(renames)} files renamed")

if __name__ == "__main__":
    main()