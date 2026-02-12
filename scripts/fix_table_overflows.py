#!/usr/bin/env python3
"""
Fix Table Overflows: Automatically wrap wide tables with adjustbox

Scans dissertation for tables without adjustbox wrappers and adds them.
Handles different table environments (table, longtable, etc.)

Usage:
  python scripts/fix_table_overflows.py --check          # Dry run, show what would change
  python scripts/fix_table_overflows.py --fix            # Actually fix tables
  python scripts/fix_table_overflows.py --file PATH      # Fix single file

Author: Claude Code (Sonnet 4.5)
Date: 2025-11-03
"""

import argparse
import re
from pathlib import Path


def has_adjustbox(content: str, tabular_start: int) -> bool:
    """Check if tabular environment already wrapped in adjustbox"""
    # Look backwards from tabular to find if there's an adjustbox
    before_tabular = content[:tabular_start]
    lines_before = before_tabular.split("\n")

    # Check last 5 lines before tabular
    for line in lines_before[-5:]:
        if "adjustbox" in line or "resizebox" in line or "scalebox" in line:
            return True
    return False


def count_columns(tabular_spec: str) -> int:
    """Count number of columns in tabular specification"""
    # Extract spec from \begin{tabular}{...}
    match = re.search(r"\\begin{tabular}{([^}]+)}", tabular_spec)
    if not match:
        return 0

    spec = match.group(1)
    # Count column specifiers (l, c, r, p, etc.)
    col_count = len(re.findall(r"[lcrp]", spec))
    return col_count


def is_wide_table(content: str, tabular_start: int) -> bool:
    """Heuristically determine if table is likely too wide"""
    tabular_line = content[tabular_start : tabular_start + 200]
    col_count = count_columns(tabular_line)

    # Tables with 7+ columns are often problematic
    if col_count >= 7:
        return True

    # Check for long column headers
    # Extract first few lines of table
    table_start = content[tabular_start : tabular_start + 500]
    if "textbf{" in table_start:
        headers = re.findall(r"\\textbf{([^}]+)}", table_start)
        if any(len(h) > 20 for h in headers):
            return True

    return False


def fix_table_in_content(content: str) -> tuple[str, list[str]]:
    """Add adjustbox wrappers to tables that need them"""
    changes = []

    # Find all tabular environments
    pattern = r"(\\begin{tabular}{[^}]+})"
    matches = list(re.finditer(pattern, content))

    # Process in reverse to preserve offsets
    for match in reversed(matches):
        start_pos = match.start()

        # Skip if already wrapped
        if has_adjustbox(content, start_pos):
            continue

        # Skip if not wide (optional - could wrap all)
        if not is_wide_table(content, start_pos):
            continue

        # Find the end of this tabular environment
        end_pattern = r"\\end{tabular}"
        end_match = re.search(end_pattern, content[start_pos:])
        if not end_match:
            continue

        end_pos = start_pos + end_match.end()

        # Extract the table content
        table_content = content[start_pos:end_pos]

        # Wrap with adjustbox
        wrapped = (
            f"\\begin{{adjustbox}}{{max width=\\textwidth}}\n{table_content}\n\\end{{adjustbox}}"
        )

        # Replace in content
        content = content[:start_pos] + wrapped + content[end_pos:]

        # Log change
        line_num = content[:start_pos].count("\n") + 1
        changes.append(f"Line {line_num}: Added adjustbox wrapper")

    return content, changes


def process_file(file_path: Path, dry_run: bool = True) -> tuple[bool, list[str]]:
    """Process a single LaTeX file"""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Skip if no tables
        if "\\begin{tabular}" not in content:
            return False, []

        # Fix tables
        new_content, changes = fix_table_in_content(content)

        if not changes:
            return False, []

        # Write changes if not dry run
        if not dry_run:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

        return True, changes

    except Exception as e:
        return False, [f"ERROR: {str(e)}"]


def main():
    parser = argparse.ArgumentParser(description="Fix table overflows in dissertation")
    parser.add_argument("--check", action="store_true", help="Dry run (show changes)")
    parser.add_argument("--fix", action="store_true", help="Actually apply fixes")
    parser.add_argument("--file", type=str, help="Fix single file")
    parser.add_argument("--dir", type=str, default="analysis/dissertation", help="Base directory")

    args = parser.parse_args()

    if not any([args.check, args.fix]):
        args.check = True  # Default to dry run

    dry_run = not args.fix

    if args.file:
        # Single file
        file_path = Path(args.file)
        modified, changes = process_file(file_path, dry_run=dry_run)

        if modified:
            mode = "Would modify" if dry_run else "Modified"
            print(f"{mode}: {file_path}")
            for change in changes:
                print(f"  {change}")
        else:
            print(f"No changes needed: {file_path}")

    else:
        # Scan all .tex files
        base_dir = Path(args.dir)
        tex_files = list(base_dir.rglob("*.tex"))

        print(f"Scanning {len(tex_files)} LaTeX files...")

        modified_files = []
        for tex_file in tex_files:
            modified, changes = process_file(tex_file, dry_run=dry_run)
            if modified:
                modified_files.append((tex_file, changes))

        # Summary
        print(f"\n{'=' * 70}")
        print("TABLE OVERFLOW FIX SUMMARY")
        print(f"{'=' * 70}")
        print(f"Mode: {'DRY RUN (no changes made)' if dry_run else 'FIXING (changes applied)'}")
        print(f"Files scanned: {len(tex_files)}")
        print(f"Files modified: {len(modified_files)}")

        if modified_files:
            print("\nModified files:")
            for file_path, changes in modified_files:
                rel_path = file_path.relative_to(base_dir)
                print(f"\n  {rel_path}:")
                for change in changes:
                    print(f"    {change}")

        if dry_run and modified_files:
            print("\nRun with --fix to apply changes")


if __name__ == "__main__":
    main()
