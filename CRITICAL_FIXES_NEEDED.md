# CRITICAL CODE FIXES NEEDED - ACTION REQUIRED

**Date:** January 9, 2026
**Priority:** HIGH - Some issues cause runtime crashes

---

## 1. CRITICAL: Undefined Names (14 occurrences) - WILL CRASH

### Files with missing imports:

```python
# py/backtest/pinnacle_historical_simulation.py:78
"password": os.getenv("DB_PASSWORD", "sicillionbillions")
# ERROR: 'os' not imported

# py/causal/did_injury_analysis.py:137
"password": os.getenv("DB_PASSWORD", "sicillionbillions")
# ERROR: 'os' not imported

# py/rl/experiment_feature_extractor.py:166
# ERROR: Name "os" is not defined
```

### Quick Fix (5 minutes):

```bash
cd /home/dro/rice/nfl-analytics

# Find all F821 errors
ruff check py/ --select F821

# Auto-fix what we can
ruff check py/ --select F821 --fix

# Manually add 'import os' to any remaining files
```

**Why this matters:** These WILL crash at runtime when the code paths execute.

---

## 2. HIGH: R Edge Case Bugs (25 occurrences) - INCORRECT RESULTS

### Problem:

```r
# WRONG: When nrow(data) == 0, this becomes 1:0 = c(1, 0)
# Loop runs TWICE with invalid indices
for (i in 1:nrow(data)) {
  process_row(data[i, ])  # data[1,] and data[0,] are invalid!
}

# CORRECT: seq_len(0) = integer(0), loop doesn't run
for (i in seq_len(nrow(data))) {
  process_row(data[i, ])
}
```

### Files affected (25 total):

```bash
R/analysis/line_movement_analysis.R:652
R/backfill_advanced_metrics.R:417
R/backfill_advanced_metrics.R:440
R/backfill_advanced_metrics.R:462
# ... 21 more
```

### Find all occurrences:

```bash
cd /home/dro/rice/nfl-analytics
grep -rn "1:nrow\|1:ncol\|1:length" R/ --include="*.R" > /tmp/r_seq_bugs.txt
cat /tmp/r_seq_bugs.txt
```

### Manual Fix Required (2 hours):

Replace each `1:nrow(x)` with `seq_len(nrow(x))`
Replace each `1:length(x)` with `seq_along(x)`

**Why this matters:** These cause incorrect results when data frames are empty (edge case in backtesting, data gaps).

---

## 3. MEDIUM: 67 Unformatted Python Files

### One-line fix:

```bash
cd /home/dro/rice/nfl-analytics
black py/
```

**Result:** All 573 Python files will be consistently formatted (100% compliance)

**Time:** 30 seconds

---

## 4. MEDIUM: 77 Unused Imports

### Auto-fix:

```bash
cd /home/dro/rice/nfl-analytics
ruff check py/ --select F401 --fix
```

**Result:** Removes 77 unused imports, slightly faster module loading

**Time:** 1 minute

---

## 5. MEDIUM: 135 Imports Not at Top of File

### Auto-fix:

```bash
cd /home/dro/rice/nfl-analytics
ruff check py/ --select E402,I001 --fix
```

**Result:** Cleaner import organization

**Time:** 1 minute

---

## Quick Win Script (Week 1 - 3 hours)

```bash
#!/bin/bash
# File: /home/dro/rice/nfl-analytics/scripts/fix_critical_issues.sh

set -e
cd /home/dro/rice/nfl-analytics

echo "=== CRITICAL CODE QUALITY FIXES ==="
echo ""

echo "Step 1: Fix undefined names (F821)..."
ruff check py/ --select F821 --fix
echo "✓ Fixed"
echo ""

echo "Step 2: Format all Python files with Black..."
black py/
echo "✓ Fixed"
echo ""

echo "Step 3: Remove unused imports..."
ruff check py/ --select F401 --fix
echo "✓ Fixed"
echo ""

echo "Step 4: Fix import ordering..."
ruff check py/ --select I001,E402 --fix
echo "✓ Fixed"
echo ""

echo "Step 5: Finding R seq_len bugs (MANUAL FIX REQUIRED)..."
grep -rn "1:nrow\|1:ncol\|1:length" R/ --include="*.R" > /tmp/r_seq_bugs.txt
echo "Found $(wc -l < /tmp/r_seq_bugs.txt) occurrences"
echo "See /tmp/r_seq_bugs.txt for locations"
echo ""

echo "=== AUTO-FIXES COMPLETE ==="
echo "Remaining: 25 R seq_len bugs need manual review"
echo "Estimated time: 2 hours"
echo ""
echo "Next steps:"
echo "1. Review /tmp/r_seq_bugs.txt"
echo "2. Replace 1:nrow(x) with seq_len(nrow(x))"
echo "3. Replace 1:length(x) with seq_along(x)"
echo "4. Test affected R scripts"
```

---

## Before Running Fixes: Verify Clean Git State

```bash
cd /home/dro/rice/nfl-analytics

# Check current git status
git status

# Create a backup branch
git checkout -b code-quality-fixes-2026-01-09

# Run fixes
bash scripts/fix_critical_issues.sh

# Review changes
git diff

# Commit
git add -A
git commit -m "fix: critical code quality issues

- Fix F821 undefined names (14 files)
- Format all Python files with Black (67 files)
- Remove unused imports (77 occurrences)
- Fix import ordering (135 files)

Remaining: 25 R seq_len bugs require manual review
"
```

---

## Verification

### Python:

```bash
# Verify no F821 errors remain
ruff check py/ --select F821
# Should output: All checks passed!

# Verify formatting
black --check py/
# Should output: All done! ✨ 🍰 ✨
# 572 files would be left unchanged.

# Check remaining issues
ruff check py/ | head -20
```

### R:

```bash
# Check how many seq_len issues remain
grep -rn "1:nrow\|1:ncol\|1:length" R/ --include="*.R" | wc -l
# Target: 0 (after manual fixes)
```

---

## Impact Summary

| Fix | Files | Time | Impact |
|-----|-------|------|--------|
| F821 undefined names | 14 | 5 min | Prevents crashes |
| Black formatting | 67 | 30 sec | Consistency |
| Unused imports | 77 | 1 min | Cleanup |
| Import ordering | 135 | 1 min | Readability |
| R seq_len bugs | 25 | 2 hours | Prevents edge case bugs |
| **TOTAL** | **~300** | **~3 hours** | **Production-ready** |

---

## After Fixes: Setup Prevention

```bash
# Install pre-commit hooks to prevent future issues
cd /home/dro/rice/nfl-analytics
pre-commit install

# This will automatically run Black, Ruff, and MyPy on every commit
```

---

## Questions?

See full assessment:
- `/home/dro/rice/nfl-analytics/CODE_QUALITY_ASSESSMENT.md`
- `/tmp/code_quality_assessment_report.md`

Or run individual checks:
```bash
ruff check py/                       # Python linting
black --check py/                    # Python formatting
mypy py/ --ignore-missing-imports    # Python type checking
Rscript -e "library(lintr); lint_dir('R/')"  # R linting
```
