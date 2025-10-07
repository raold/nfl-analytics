#!/usr/bin/env python3
"""
Fix LaTeX table files with incorrect tablenotes environment.
Convert \begin{tablenotes}...\end{tablenotes} to be wrapped in threeparttable.
"""

import os
import re
from pathlib import Path

def fix_table_file(filepath):
    """Fix a single LaTeX table file."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Check if file needs fixing
    if 'tablenotes' in content and 'threeparttable' not in content:
        # Pattern to match table with tablenotes but no threeparttable
        pattern = r'(\\begin{table}.*?\n.*?\\begin{tabular}{.*?})(.*?)(\\end{tabular})\s*(\\begin{tablenotes}.*?\\end{tablenotes})\s*(\\end{table})'

        def replace_func(match):
            return (match.group(1).replace('\\begin{tabular}', '\\begin{threeparttable}\n  \\begin{tabular}') +
                   match.group(2) +
                   match.group(3) + '\n  ' +
                   match.group(4) + '\n  \\end{threeparttable}\n' +
                   match.group(5))

        # Apply fix
        fixed_content = re.sub(pattern, replace_func, content, flags=re.DOTALL)

        # Write back if changed
        if fixed_content != content:
            with open(filepath, 'w') as f:
                f.write(fixed_content)
            print(f"Fixed: {filepath}")
            return True
    return False

def main():
    # Directory containing LaTeX table files
    tables_dir = Path('/Users/dro/rice/nfl-analytics/analysis/dissertation/figures/out')

    # Find all .tex files
    tex_files = list(tables_dir.glob('*.tex'))

    fixed_count = 0
    for tex_file in tex_files:
        if fix_table_file(tex_file):
            fixed_count += 1

    print(f"\nFixed {fixed_count} files out of {len(tex_files)} total .tex files")

if __name__ == '__main__':
    main()