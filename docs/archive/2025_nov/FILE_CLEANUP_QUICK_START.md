# File Organization Cleanup - Quick Start

**Date:** November 15, 2025
**Status:** Ready to execute
**Time Required:** 5-10 minutes

---

## Problem

15+ documentation files scattered in root directory violating FILE_ORGANIZATION.md rules.

**Root directory should only contain:**
- README.md, CLAUDE.md, SETUP.md, CONTRIBUTING.md
- FILE_ORGANIZATION.md, LICENSE, .gitignore, .env.example

---

## Solution

Automated cleanup script that moves files to proper locations:
- Session summaries → `docs/archive/2025_nov/`
- Experiment docs → `docs/experiments/` (new)
- Delivery reports → `docs/deployment/` or `docs/archive/2025_nov/`
- Quick references → `docs/guides/`

---

## Quick Start

### Option 1: Preview Changes (Dry Run)

```bash
cd /Users/dro/rice/nfl-analytics
chmod +x scripts/cleanup_root_directory.sh
./scripts/cleanup_root_directory.sh
```

**Output:** Shows what would be moved (no actual changes)

### Option 2: Execute Cleanup

```bash
cd /Users/dro/rice/nfl-analytics
./scripts/cleanup_root_directory.sh --execute
```

**Output:** Moves files, creates directories, shows summary

### Option 3: Manual Verification

```bash
# See survey report
cat docs/archive/2025_nov/FILE_ORGANIZATION_SURVEY_NOV15.md

# Check current root violations
ls -1 *.md *.txt | grep -v -E "^(README|CLAUDE|SETUP|CONTRIBUTING|FILE_ORGANIZATION)\\.md$"
```

---

## What Gets Moved

### From Root → Archive (7 files)
- `SESSION_STATUS_NOV14.md` → `docs/archive/2025_nov/SESSION_STATUS_2025-11-14.md`
- `TRAINING_SCRIPT_CREATION_SUMMARY.md` → `docs/archive/2025_nov/TRAINING_SCRIPT_SUMMARY_2025-11-14.md`
- `FEATURE_ANALYSIS_DELIVERY.md` → `docs/archive/2025_nov/FEATURE_ANALYSIS_DELIVERY_2025-11-14.md`

### From Root → Experiments (5 files)
- `EXPERIMENT_2_DELIVERY.md` → `docs/experiments/XGB_EXP2_DELIVERY.md`
- `EXPERIMENT_2_FILES_INDEX.txt` → `docs/experiments/XGB_EXP2_FILES_INDEX.txt`
- `BFS_EXPERIMENT_5_CHECKLIST.md` → `docs/experiments/BFS_EXP5_CHECKLIST.md`
- `BFS_EXPERIMENTS_EXECUTION_SUMMARY.md` → `docs/experiments/BFS_EXPERIMENTS_SUMMARY.md`
- `XGBOOST_V3_1_EXECUTION_SUMMARY.md` → `docs/experiments/XGBOOST_V3_1_SUMMARY.md`

### From Root → Deployment (1 file)
- `AB_TEST_DELIVERY_REPORT.md` → `docs/deployment/AB_TEST_DELIVERY_REPORT.md`

### From Root → Guides (1 file)
- `QUICK_RECOVERY_COMMANDS.txt` → `docs/guides/DATABASE_RECOVERY_QUICK_REFERENCE.txt`

---

## After Cleanup

### Verify

```bash
# Should only show 8 permitted files (5 .md + 3 others)
ls -1 *.md *.txt

# Check git status
git status

# Preview changes
git diff --stat
```

### Commit

```bash
git add -A
git commit -m "docs: reorganize 15 root files per FILE_ORGANIZATION.md

- Moved session summaries to docs/archive/2025_nov/
- Created docs/experiments/ for XGBoost/BFS experiment docs
- Moved delivery reports to docs/deployment/
- Moved quick references to docs/guides/
- Root directory now compliant (8 permitted files only)

Ref: docs/archive/2025_nov/FILE_ORGANIZATION_SURVEY_NOV15.md"
```

---

## Rollback (If Needed)

```bash
# Undo all changes
git reset --hard HEAD~1

# Or undo specific file
git checkout HEAD~1 -- path/to/file.md
```

---

## Optional: Phase 2-4 (Future)

Phase 1 (above) cleans root directory. Optional future work:

**Phase 2:** Consolidate 6 AB test docs → 3 canonical docs
**Phase 3:** Archive old ensemble version docs
**Phase 4:** Update CLAUDE.md path references

See `docs/archive/2025_nov/FILE_ORGANIZATION_SURVEY_NOV15.md` for details.

---

## Files Created

1. **Survey Report:** `docs/archive/2025_nov/FILE_ORGANIZATION_SURVEY_NOV15.md` (comprehensive analysis)
2. **Cleanup Script:** `scripts/cleanup_root_directory.sh` (automated execution)
3. **Quick Start:** `docs/archive/2025_nov/FILE_CLEANUP_QUICK_START.md` (this file)

---

## Status

- ✅ Survey complete
- ✅ Script ready
- ⏳ Awaiting user approval/execution
- ⏳ Phase 1 execution (5 minutes)
- ⏳ Git commit
- ⏳ Optional Phase 2-4 (future)

---

**Questions?** See full survey: `docs/archive/2025_nov/FILE_ORGANIZATION_SURVEY_NOV15.md`
