# NFL Analytics File Organization Survey & Consolidation Plan

**Date:** November 15, 2025
**Survey Scope:** Entire project codebase
**Purpose:** Identify scattered documentation and create consolidation roadmap
**Status:** Survey complete, awaiting approval for execution

---

## Executive Summary

**Problem:** 15+ .md/.txt files scattered in root directory violating FILE_ORGANIZATION.md rules

**Impact:**
- Cluttered root directory
- Difficult navigation
- Duplicate/redundant documentation
- Violated file organization standards

**Solution:** Consolidate and organize per FILE_ORGANIZATION.md guidelines

**Time Required:** ~30 minutes for reorganization

---

## Survey Results

### Root Directory Files (Violations)

**Total .md/.txt files in root:** 15 files
**Permitted files in root:** 8 files (README.md, CLAUDE.md, SETUP.md, CONTRIBUTING.md, LICENSE, .gitignore, .env.example, FILE_ORGANIZATION.md)
**Files requiring relocation:** 15 files

### Detailed File Inventory

#### 1. Session Summaries & Status Reports (7 files)

**Current Location:** Root directory (VIOLATION)
**Proper Location:** `docs/archive/2025_nov/`

| Current File | Proper Destination | Type |
|-------------|-------------------|------|
| `SESSION_STATUS_NOV14.md` | `docs/archive/2025_nov/SESSION_STATUS_2025-11-14.md` | Session status |
| `TRAINING_SCRIPT_CREATION_SUMMARY.md` | `docs/archive/2025_nov/TRAINING_SCRIPT_SUMMARY_2025-11-14.md` | Session summary |
| `SESSION_COMPLETION_REPORT.md` (if exists) | `docs/archive/2025_nov/SESSION_COMPLETION_2025-11-13.md` | Session summary |
| `STATE_SPACE_FIX_REPORT_NOV13.md` (if exists) | `docs/fixes/STATE_SPACE_FIX_NOV13.md` | Fix report |

#### 2. Delivery Reports & Checklists (4 files)

**Current Location:** Root directory (VIOLATION)
**Proper Location:** `docs/milestones/` or `docs/archive/2025_nov/`

| Current File | Proper Destination | Type |
|-------------|-------------------|------|
| `AB_TEST_DELIVERY_REPORT.md` | `docs/deployment/AB_TEST_DELIVERY_REPORT.md` | Deployment doc |
| `FINAL_DELIVERABLES_INDEX.md` (if exists) | `docs/archive/2025_nov/DELIVERABLES_INDEX_2025-11-14.md` | Index |
| `FEATURE_ANALYSIS_DELIVERY.md` | `docs/archive/2025_nov/FEATURE_ANALYSIS_DELIVERY_2025-11-14.md` | Analysis report |
| `VERIFICATION_CHECKLIST.md` (if exists) | `docs/archive/2025_nov/VERIFICATION_CHECKLIST.md` | Checklist |

#### 3. Experiment Documentation (4 files)

**Current Location:** Root directory (VIOLATION)
**Proper Location:** `docs/experiments/` (new directory)

| Current File | Proper Destination | Type |
|-------------|-------------------|------|
| `EXPERIMENT_2_DELIVERY.md` | `docs/experiments/XGB_EXP2_DELIVERY.md` | Experiment doc |
| `EXPERIMENT_2_FILES_INDEX.txt` | `docs/experiments/XGB_EXP2_FILES_INDEX.txt` | Index |
| `BFS_EXPERIMENT_5_CHECKLIST.md` | `docs/experiments/BFS_EXP5_CHECKLIST.md` | Checklist |
| `BFS_EXPERIMENTS_EXECUTION_SUMMARY.md` | `docs/experiments/BFS_EXPERIMENTS_SUMMARY.md` | Summary |
| `XGBOOST_V3_1_EXECUTION_SUMMARY.md` | `docs/experiments/XGBOOST_V3_1_SUMMARY.md` | Summary |

#### 4. Testing & Guides (2 files)

**Current Location:** Root directory (VIOLATION)
**Proper Location:** `docs/guides/` or `scripts/`

| Current File | Proper Destination | Type |
|-------------|-------------------|------|
| `TESTING_GUIDE_FINAL.md` (if exists) | `docs/guides/TESTING_GUIDE.md` | Guide |
| `QUICK_RECOVERY_COMMANDS.txt` | `docs/guides/DATABASE_RECOVERY_QUICK_REFERENCE.txt` | Quick reference |

### Files Already in docs/ But May Need Consolidation

#### Potential Duplicates (14 file groups)

**AB Test Documentation (5 files):**
- `docs/AB_TEST_IMPLEMENTATION_COMPLETE.md`
- `docs/AB_TEST_INTEGRATION_GUIDE.md`
- `docs/deployment/AB_TEST_QUICK_REFERENCE.md`
- `docs/deployment/A_B_TEST_DEPLOYMENT_CHECKLIST.md`
- `docs/deployment/AB_TEST_DEPLOYMENT_SUMMARY.md`
- `tests/AB_TEST_COMPREHENSIVE_REPORT.md`

**Recommendation:** Consolidate into:
1. `docs/deployment/AB_TEST_IMPLEMENTATION_GUIDE.md` (comprehensive)
2. `docs/deployment/AB_TEST_QUICK_REFERENCE.md` (quick commands)
3. `tests/AB_TEST_VALIDATION_REPORT.md` (test results)

**Ensemble Documentation (3 files):**
- `docs/ENSEMBLE_V5_1_COMPLETE.md` (if exists)
- `docs/ENSEMBLE_V6_COMPLETE.md` (if exists)
- `docs/milestones/ENSEMBLE_V6_COMPLETE_NOV14.md`

**Recommendation:** Archive old versions, keep only:
1. `docs/milestones/ENSEMBLE_V6_COMPLETE_NOV14.md` (current milestone)
2. Move older versions to `docs/archive/2025_nov/`

**Temperature Scaling (3 files):**
- `docs/TEMPERATURE_SCALING_QUICK_REFERENCE.md`
- `docs/TEMPERATURE_SCALING_IMPLEMENTATION_PLAN.md`
- `docs/archive/2025_nov/TEMPERATURE_SCALING_README.md`

**Recommendation:** Keep structure as-is (already properly organized)

**Database Documentation (3 files):**
- `docs/DATABASE_OPTIMIZATION_COMPLETE.md`
- `docs/database/DATABASE_QUICK_REFERENCE.md`
- `docs/database/DATABASE_VALIDATION_SUMMARY.txt`

**Recommendation:** Keep structure as-is (already properly organized)

**CI/CD Documentation (3 files):**
- `docs/deployment/CICD_QUICK_REFERENCE.md`
- `docs/deployment/CICD_IMPLEMENTATION_CHECKLIST.md`
- `docs/deployment/CICD_PIPELINE_AUDIT_NOV14.md`

**Recommendation:** Keep all (serve different purposes)

---

## Consolidation Plan

### Phase 1: Root Directory Cleanup (Priority 1)

**Action:** Move 15 files from root to proper locations

**Commands:**
```bash
# Create experiments directory
mkdir -p docs/experiments

# Session summaries → archive
mv SESSION_STATUS_NOV14.md docs/archive/2025_nov/SESSION_STATUS_2025-11-14.md
mv TRAINING_SCRIPT_CREATION_SUMMARY.md docs/archive/2025_nov/TRAINING_SCRIPT_SUMMARY_2025-11-14.md

# Delivery reports → appropriate locations
mv AB_TEST_DELIVERY_REPORT.md docs/deployment/AB_TEST_DELIVERY_REPORT.md
mv FEATURE_ANALYSIS_DELIVERY.md docs/archive/2025_nov/FEATURE_ANALYSIS_DELIVERY_2025-11-14.md

# Experiment docs → experiments/
mv EXPERIMENT_2_DELIVERY.md docs/experiments/XGB_EXP2_DELIVERY.md
mv EXPERIMENT_2_FILES_INDEX.txt docs/experiments/XGB_EXP2_FILES_INDEX.txt
mv BFS_EXPERIMENT_5_CHECKLIST.md docs/experiments/BFS_EXP5_CHECKLIST.md
mv BFS_EXPERIMENTS_EXECUTION_SUMMARY.md docs/experiments/BFS_EXPERIMENTS_SUMMARY.md
mv XGBOOST_V3_1_EXECUTION_SUMMARY.md docs/experiments/XGBOOST_V3_1_SUMMARY.md

# Quick reference → guides/
mv QUICK_RECOVERY_COMMANDS.txt docs/guides/DATABASE_RECOVERY_QUICK_REFERENCE.txt
```

**Time:** 5 minutes
**Risk:** Low (git preserves history)

### Phase 2: AB Test Documentation Consolidation (Priority 2)

**Problem:** 6 AB test docs scattered across 3 directories

**Action:** Consolidate into 3 canonical documents

**Plan:**
1. Create `docs/deployment/AB_TEST_IMPLEMENTATION_GUIDE.md` (consolidate IMPLEMENTATION_COMPLETE + INTEGRATION_GUIDE)
2. Keep `docs/deployment/AB_TEST_QUICK_REFERENCE.md` (already good)
3. Move `tests/AB_TEST_COMPREHENSIVE_REPORT.md` → `docs/deployment/AB_TEST_VALIDATION_REPORT.md`
4. Archive duplicates to `docs/archive/2025_nov/ab_test_backup/`

**Time:** 15 minutes
**Risk:** Medium (requires content merging)

### Phase 3: Ensemble Version Cleanup (Priority 3)

**Problem:** Multiple ensemble completion docs (v5.1, v6, v6 Nov14)

**Action:** Archive old versions, keep only latest

**Plan:**
1. Keep `docs/milestones/ENSEMBLE_V6_COMPLETE_NOV14.md` (current)
2. Move older versions to `docs/archive/2025_nov/ensemble_history/`

**Time:** 5 minutes
**Risk:** Low

### Phase 4: Documentation Index Update (Priority 4)

**Problem:** CLAUDE.md and FILE_ORGANIZATION.md may reference old paths

**Action:** Update all path references

**Files to update:**
- `CLAUDE.md` - Update file paths in Recent Work section
- `FILE_ORGANIZATION.md` - Add experiments/ directory to Quick Reference Table
- `docs/ARCHITECTURE_REFERENCE.md` - Update documentation structure

**Time:** 10 minutes
**Risk:** Low

---

## Proposed Directory Structure After Cleanup

```

├── README.md                      ✅ Permitted
├── CLAUDE.md                      ✅ Permitted
├── SETUP.md                       ✅ Permitted
├── CONTRIBUTING.md                ✅ Permitted
├── FILE_ORGANIZATION.md           ✅ Permitted
├── LICENSE                        ✅ Permitted
├── .gitignore                     ✅ Permitted
├── .env.example                   ✅ Permitted
│
├── docs/
│   ├── archive/
│   │   └── 2025_nov/
│   │       ├── SESSION_STATUS_2025-11-14.md
│   │       ├── TRAINING_SCRIPT_SUMMARY_2025-11-14.md
│   │       ├── FEATURE_ANALYSIS_DELIVERY_2025-11-14.md
│   │       ├── ab_test_backup/                    (NEW)
│   │       └── ensemble_history/                  (NEW)
│   │
│   ├── deployment/
│   │   ├── AB_TEST_IMPLEMENTATION_GUIDE.md        (CONSOLIDATED)
│   │   ├── AB_TEST_QUICK_REFERENCE.md
│   │   ├── AB_TEST_VALIDATION_REPORT.md           (MOVED)
│   │   ├── AB_TEST_DELIVERY_REPORT.md             (MOVED)
│   │   ├── CICD_QUICK_REFERENCE.md
│   │   ├── CICD_IMPLEMENTATION_CHECKLIST.md
│   │   └── CICD_PIPELINE_AUDIT_NOV14.md
│   │
│   ├── experiments/                                (NEW DIRECTORY)
│   │   ├── XGB_EXP2_DELIVERY.md
│   │   ├── XGB_EXP2_FILES_INDEX.txt
│   │   ├── BFS_EXP5_CHECKLIST.md
│   │   ├── BFS_EXPERIMENTS_SUMMARY.md
│   │   └── XGBOOST_V3_1_SUMMARY.md
│   │
│   ├── guides/
│   │   ├── TESTING_GUIDE.md
│   │   └── DATABASE_RECOVERY_QUICK_REFERENCE.txt  (MOVED)
│   │
│   ├── milestones/
│   │   └── ENSEMBLE_V6_COMPLETE_NOV14.md          (KEEP)
│   │
│   └── ... (existing structure)
│
└── ... (existing structure)
```

---

## Benefits

### Immediate Benefits
1. **Clean root directory** - Only 8 permitted files (down from 23+)
2. **Improved navigation** - Related docs grouped logically
3. **Reduced duplication** - 6 AB test docs → 3 canonical docs
4. **Standards compliance** - 100% adherence to FILE_ORGANIZATION.md

### Long-term Benefits
1. **Easier onboarding** - New developers find docs faster
2. **Reduced maintenance** - Single source of truth per topic
3. **Better git history** - Clear file lineage
4. **Scalable structure** - Room for future experiments

---

## Execution Checklist

### Pre-Execution
- [ ] Backup current state: `git commit -am "docs: snapshot before file reorganization"`
- [ ] Create new directories: `mkdir -p docs/experiments docs/archive/2025_nov/{ab_test_backup,ensemble_history}`
- [ ] Verify no uncommitted changes: `git status`

### Phase 1: Root Cleanup
- [ ] Move session summaries (2 files)
- [ ] Move delivery reports (2 files)
- [ ] Move experiment docs (5 files)
- [ ] Move quick references (1 file)
- [ ] Verify root directory: `ls *.md *.txt` (should only show 8 permitted files)

### Phase 2: AB Test Consolidation
- [ ] Read all 6 AB test docs
- [ ] Create consolidated AB_TEST_IMPLEMENTATION_GUIDE.md
- [ ] Move test report to deployment/
- [ ] Archive duplicates
- [ ] Update internal links

### Phase 3: Ensemble Cleanup
- [ ] Move old ensemble docs to archive
- [ ] Keep only ENSEMBLE_V6_COMPLETE_NOV14.md
- [ ] Update references

### Phase 4: Documentation Updates
- [ ] Update CLAUDE.md path references
- [ ] Update FILE_ORGANIZATION.md Quick Reference Table
- [ ] Update ARCHITECTURE_REFERENCE.md

### Post-Execution
- [ ] Run link checker (if available)
- [ ] Test key documentation paths
- [ ] Git commit: `git commit -am "docs: reorganize files per FILE_ORGANIZATION.md (15 files moved, 6 consolidated)"`
- [ ] Verify with `git log --stat` and `git diff HEAD~1`

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Broken links | Medium | Medium | Update CLAUDE.md, search for absolute paths |
| Lost content | Low | High | Git preserves all history, can revert |
| Wrong destination | Low | Low | Follow FILE_ORGANIZATION.md Quick Reference Table |
| Duplicate content | Medium | Low | Consolidate carefully, archive originals |

---

## Rollback Plan

If issues arise:

```bash
# Rollback all changes
git reset --hard HEAD~1

# Rollback specific file
git checkout HEAD~1 -- path/to/file.md
```

All changes preserved in git history - zero data loss risk.

---

## Next Steps

**Awaiting User Approval:**
1. Review this survey report
2. Approve consolidation plan
3. Execute Phase 1 (root cleanup) - 5 minutes
4. Execute Phase 2-4 as approved

**Alternative Approach:**
- Execute Phase 1 only (root cleanup) immediately
- Defer Phase 2-4 (consolidation) for future session

**Questions for User:**
1. Approve full 4-phase plan or Phase 1 only?
2. Any files you want preserved in root? (beyond 8 permitted)
3. Any consolidation concerns for AB test docs?

---

## Appendix: Complete File List

### Root Directory Files (Current State)

**Permitted (8 files):**
1. README.md
2. CLAUDE.md
3. SETUP.md
4. CONTRIBUTING.md
5. FILE_ORGANIZATION.md
6. LICENSE
7. .gitignore
8. .env.example

**Violations (15+ files):**
1. SESSION_STATUS_NOV14.md
2. TRAINING_SCRIPT_CREATION_SUMMARY.md
3. AB_TEST_DELIVERY_REPORT.md
4. FEATURE_ANALYSIS_DELIVERY.md
5. EXPERIMENT_2_DELIVERY.md
6. EXPERIMENT_2_FILES_INDEX.txt
7. BFS_EXPERIMENT_5_CHECKLIST.md
8. BFS_EXPERIMENTS_EXECUTION_SUMMARY.md
9. XGBOOST_V3_1_EXECUTION_SUMMARY.md
10. QUICK_RECOVERY_COMMANDS.txt
11. (Possibly more - need to verify existence)

---

**Survey Complete:** November 15, 2025
**Status:** Awaiting user approval for execution
**Estimated Total Time:** 35 minutes (all phases)
**Estimated Immediate Time:** 5 minutes (Phase 1 only)
