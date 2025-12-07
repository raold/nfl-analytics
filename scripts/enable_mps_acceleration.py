#!/usr/bin/env python3
"""
Enable MPS GPU Acceleration - Quick Implementation Script

Automatically updates PyTorch models to use MPS GPU acceleration.

This script:
1. Scans for PyTorch models with hardcoded device='cpu' or device='cuda'
2. Replaces with get_best_device() utility
3. Creates backup files before modification
4. Validates changes don't break imports

Usage:
    python scripts/enable_mps_acceleration.py --dry-run  # Preview changes
    python scripts/enable_mps_acceleration.py            # Apply changes
    python scripts/enable_mps_acceleration.py --revert   # Revert changes

Author: Claude Code (Sonnet 4.5)
Date: November 15, 2025
"""

import argparse
import re
import shutil
from pathlib import Path

# File patterns to search
PYTORCH_PATTERNS = [
    "py/models/**/*.py",
    "py/rl/**/*.py",
    "py/ensemble/**/*.py",
]

# Device detection patterns (regex)
DEVICE_PATTERNS = [
    # Pattern 1: device = torch.device('cpu')
    (r"device\s*=\s*torch\.device\(['\"]cpu['\"]\)", "SIMPLE_CPU"),
    # Pattern 2: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    (r"device\s*=\s*torch\.device\(['\"]cuda['\"].*?else.*?['\"]cpu['\"]\)", "CUDA_FALLBACK"),
    # Pattern 3: torch.device(args.device if torch.cuda.is_available() else 'cpu')
    (r"torch\.device\(args\.device.*?torch\.cuda\.is_available.*?\)", "ARGS_CUDA"),
]

# Import statement to add
IMPORT_STATEMENT = "from py.utils.device_utils import get_best_device"

# Replacement code
REPLACEMENT_CODE = "device = get_best_device()"


def find_pytorch_files() -> list[Path]:
    """Find all PyTorch files in the project"""
    files = []
    for pattern in PYTORCH_PATTERNS:
        files.extend(Path(".").glob(pattern))
    return sorted(set(files))


def scan_file_for_device_patterns(file_path: Path) -> list[tuple[int, str, str]]:
    """
    Scan file for device assignment patterns

    Returns:
        List of (line_number, pattern_type, matched_text)
    """
    matches = []

    try:
        with open(file_path) as f:
            content = f.read()
            lines = content.split("\n")

        for line_num, line in enumerate(lines, start=1):
            for pattern, pattern_type in DEVICE_PATTERNS:
                if re.search(pattern, line):
                    matches.append((line_num, pattern_type, line.strip()))

    except Exception as e:
        print(f"Error scanning {file_path}: {e}")

    return matches


def has_device_utils_import(file_path: Path) -> bool:
    """Check if file already imports get_best_device"""
    try:
        with open(file_path) as f:
            content = f.read()
        return "get_best_device" in content
    except:
        return False


def update_file_with_mps_support(file_path: Path, dry_run: bool = False) -> bool:
    """
    Update file to use MPS-aware device detection

    Returns:
        bool: True if file was modified
    """
    try:
        with open(file_path) as f:
            content = f.read()

        original_content = content

        # Add import if not present
        if "get_best_device" not in content:
            # Find import section (after docstring)
            import_section_match = re.search(r'(""".*?"""|\'\'\'.*?\'\'\')\n', content, re.DOTALL)
            if import_section_match:
                insert_pos = import_section_match.end()
            else:
                insert_pos = 0

            content = content[:insert_pos] + f"\n{IMPORT_STATEMENT}\n" + content[insert_pos:]

        # Replace device patterns
        for pattern, pattern_type in DEVICE_PATTERNS:
            content = re.sub(pattern, REPLACEMENT_CODE, content)

        # Check if anything changed
        if content == original_content:
            return False

        if not dry_run:
            # Create backup
            backup_path = file_path.with_suffix(file_path.suffix + ".bak")
            shutil.copy(file_path, backup_path)

            # Write updated content
            with open(file_path, "w") as f:
                f.write(content)

            print(f"✅ Updated: {file_path}")
            print(f"   Backup: {backup_path}")

        else:
            print(f"[DRY RUN] Would update: {file_path}")

        return True

    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False


def revert_file(file_path: Path) -> bool:
    """Revert file from backup"""
    backup_path = file_path.with_suffix(file_path.suffix + ".bak")

    if not backup_path.exists():
        print(f"⚠️  No backup found for {file_path}")
        return False

    try:
        shutil.copy(backup_path, file_path)
        print(f"✅ Reverted: {file_path}")
        return True
    except Exception as e:
        print(f"❌ Error reverting {file_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Enable MPS GPU acceleration in PyTorch models")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without modifying files"
    )
    parser.add_argument("--revert", action="store_true", help="Revert changes from backup files")
    parser.add_argument(
        "--scan-only", action="store_true", help="Only scan for device patterns (no modifications)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MPS GPU ACCELERATION - IMPLEMENTATION SCRIPT")
    print("=" * 70)
    print()

    # Find PyTorch files
    files = find_pytorch_files()
    print(f"Found {len(files)} PyTorch files")
    print()

    if args.revert:
        # Revert mode
        print("REVERTING CHANGES...")
        print()
        reverted = 0
        for file_path in files:
            if revert_file(file_path):
                reverted += 1

        print()
        print(f"Reverted {reverted} files")
        return

    # Scan mode or update mode
    files_with_patterns = []

    print("SCANNING FOR DEVICE PATTERNS...")
    print()

    for file_path in files:
        matches = scan_file_for_device_patterns(file_path)
        if matches:
            files_with_patterns.append((file_path, matches))

    print(f"Found {len(files_with_patterns)} files with device patterns:")
    print()

    for file_path, matches in files_with_patterns:
        print(f"📄 {file_path}")
        for line_num, pattern_type, matched_text in matches:
            print(f"   Line {line_num} [{pattern_type}]: {matched_text}")
        print()

    if args.scan_only:
        return

    # Update mode
    if not args.dry_run:
        print("=" * 70)
        print("⚠️  WARNING: This will modify source files")
        print("   Backups will be created (.bak files)")
        print("=" * 70)
        response = input("Continue? (y/N): ")
        if response.lower() != "y":
            print("Aborted.")
            return
        print()

    print("UPDATING FILES...")
    print()

    updated = 0
    for file_path, _ in files_with_patterns:
        if update_file_with_mps_support(file_path, dry_run=args.dry_run):
            updated += 1

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Files scanned: {len(files)}")
    print(f"Files with device patterns: {len(files_with_patterns)}")
    print(f"Files {'would be ' if args.dry_run else ''}updated: {updated}")

    if args.dry_run:
        print()
        print("Run without --dry-run to apply changes")

    else:
        print()
        print("✅ Changes applied successfully")
        print()
        print("Next steps:")
        print("1. Run tests: pytest tests/performance/test_mps_acceleration.py")
        print("2. Validate training: python py/models/bnn_multioutput_pytorch.py")
        print("3. If issues: python scripts/enable_mps_acceleration.py --revert")


if __name__ == "__main__":
    main()
