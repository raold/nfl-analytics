#!/usr/bin/env python3
"""
Pre-commit hook to check file organization.

Prevents .md and .txt files from being added to root directory
unless they are in the permitted list.
"""
import os
import sys

# Permitted files in root directory
PERMITTED = [
    'README.md',
    'CLAUDE.md',
    'SETUP.md',
    'CONTRIBUTING.md',
    'FILE_ORGANIZATION.md',
    'LICENSE',
    '.gitignore',
    '.env.example'
]

def main():
    # Get files from command line arguments
    files_to_check = sys.argv[1:]

    # Filter to only .md and .txt files in root directory
    root_docs = [
        f for f in files_to_check
        if os.path.dirname(f) in ('', '.')
        and (f.endswith('.md') or f.endswith('.txt'))
    ]

    # Check if any are not permitted
    bad_files = [
        f for f in root_docs
        if os.path.basename(f) not in PERMITTED
    ]

    if bad_files:
        print("\n⛔ Files in wrong location (must go in docs/ subdirectories):")
        for f in bad_files:
            print(f"   - {f}")
        print("\nSee FILE_ORGANIZATION.md for correct locations")
        print("Run: ./scripts/cleanup_file_organization.sh --execute")
        sys.exit(1)

    sys.exit(0)

if __name__ == '__main__':
    main()
