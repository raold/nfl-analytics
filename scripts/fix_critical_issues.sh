#!/bin/bash
# Critical Code Quality Fixes
# Generated: 2026-01-09
# Purpose: Auto-fix critical Python code quality issues

set -e
cd /home/dro/rice/nfl-analytics

echo "========================================================"
echo "  NFL Analytics - Critical Code Quality Fixes"
echo "========================================================"
echo ""

# Check we're in the right directory
if [ ! -f "CLAUDE.md" ]; then
    echo "ERROR: Not in nfl-analytics directory"
    exit 1
fi

# Check for clean git state
if ! git diff-index --quiet HEAD --; then
    echo "WARNING: You have uncommitted changes."
    echo "Consider committing or stashing before running fixes."
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Step 1/5: Fixing undefined names (F821) - CRITICAL"
echo "------------------------------------------------------"
ruff check py/ --select F821 --fix
echo "✓ Fixed undefined name errors"
echo ""

echo "Step 2/5: Formatting Python files with Black"
echo "------------------------------------------------------"
black py/
echo "✓ All files formatted"
echo ""

echo "Step 3/5: Removing unused imports (F401)"
echo "------------------------------------------------------"
ruff check py/ --select F401 --fix
echo "✓ Unused imports removed"
echo ""

echo "Step 4/5: Fixing import ordering (I001, E402)"
echo "------------------------------------------------------"
ruff check py/ --select I001 --fix
echo "✓ Imports organized"
echo ""

echo "Step 5/5: Identifying R seq_len bugs (MANUAL FIX REQUIRED)"
echo "------------------------------------------------------"
grep -rn "1:nrow\|1:ncol\|1:length" R/ --include="*.R" > /tmp/r_seq_bugs.txt || true
COUNT=$(wc -l < /tmp/r_seq_bugs.txt || echo "0")
echo "Found $COUNT occurrences needing manual review"
if [ "$COUNT" -gt 0 ]; then
    echo "Locations saved to: /tmp/r_seq_bugs.txt"
    echo ""
    echo "Sample issues:"
    head -5 /tmp/r_seq_bugs.txt
fi
echo ""

echo "========================================================"
echo "  AUTO-FIXES COMPLETE!"
echo "========================================================"
echo ""
echo "Summary of changes:"
git diff --stat || echo "(No changes detected)"
echo ""

echo "Remaining work:"
echo "  - Review and fix $COUNT R seq_len bugs manually (2 hours)"
echo "  - See /tmp/r_seq_bugs.txt for locations"
echo ""

echo "Next steps:"
echo "  1. Review changes: git diff"
echo "  2. Test code: pytest tests/unit -m unit"
echo "  3. Commit: git add -A && git commit -m 'fix: critical code quality issues'"
echo "  4. Fix R bugs: edit files in /tmp/r_seq_bugs.txt"
echo ""

echo "Verification commands:"
echo "  ruff check py/ --select F821          # Should show 0 errors"
echo "  black --check py/                     # Should show 0 files to reformat"
echo "  ruff check py/ | head -20             # Check remaining issues"
echo ""
