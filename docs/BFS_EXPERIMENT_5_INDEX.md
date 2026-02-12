# BFS Experiment 5: Index & Quick Navigation

**Date:** November 14, 2025
**Status:** ✅ Complete - Ready for Execution
**Estimated Runtime:** 15-25 minutes

---

## What Is BFS Experiment 5?

**Goal:** Fix XGBoost v3.1 performance regression using L1 regularization

**The Problem:**
- XGBoost v2 (no opponent features): 40.20 MAE ✓
- XGBoost v3.1 (with opponent features): 57.37 MAE ✗ (-42.71% worse)
- Root cause: Redundant opponent features corrupt model training

**The Solution:**
- Use L1 regularization (alpha parameter) to eliminate redundant features
- Grid search over 24 hyperparameter combinations
- 5-fold cross-validation for robust selection
- Final evaluation on 2025 hold-out set

---

## Files Created

### Main Script (Ready to Run)
**`scripts/xgb_v3_2_l1_regularization.py`** (999 lines, 35 KB)
- Complete grid search implementation
- Data loading & feature engineering
- Cross-validation pipeline
- Feature importance analysis
- Decision gate logic
- Model saving & metadata export

**Syntax Status:** ✅ Validated

### Documentation (4 Files, 54 KB)

#### 1. Quick Start Guide
**File:** `docs/BFS_EXPERIMENT_5_QUICK_START.md` (7.3 KB)

**For:** People who want to run the script and understand results
**Time:** 10 minutes to read
**Contains:**
- TL;DR summary
- How to run
- Expected outcomes table
- Loading & using the model
- Interpreting results
- Common issues & solutions

**When to Read:** First, if you're new to this experiment

#### 2. Comprehensive Technical Guide
**File:** `docs/BFS_EXPERIMENT_5_L1_REGULARIZATION.md` (19 KB)

**For:** Technical deep dive, understanding architecture
**Time:** 20 minutes to read
**Contains:**
- Executive summary
- Root cause analysis
- Script architecture (5 sections)
- Key features & implementation details
- Usage examples
- Interpreting results (4 scenarios)
- Next steps based on outcomes
- Technical details (hyperparameters, CV config)
- References

**When to Read:** If you want to understand how/why it works

#### 3. Expected Outputs & Interpretation
**File:** `docs/BFS_EXPERIMENT_5_EXPECTED_OUTPUTS.md` (23 KB)

**For:** Understanding what the script output means
**Time:** 15 minutes to reference
**Contains:**
- Example outputs for all 9 script sections
- What each output means
- 4 different possible final outcomes
- Feature importance progression
- How to interpret grid search results
- Reading the metadata JSON
- Troubleshooting guide
- Summary table of all 24 grid points

**When to Read:** While running, or to interpret your results

#### 4. Execution Checklist
**File:** `BFS_EXPERIMENT_5_CHECKLIST.md` (5.5 KB)

**For:** Step-by-step execution guide
**Time:** 5 minutes to use
**Contains:**
- Pre-execution checks (5 items)
- How to run script
- Real-time progress monitoring
- Post-execution analysis (5 steps)
- Decision matrix
- If/then action items for DEPLOY/INVESTIGATE/ABANDON
- Troubleshooting quick fixes
- Time estimates
- Key success indicators

**When to Read:** Before running script, for pre-flight checklist

---

## How to Use These Files

### Scenario 1: "I Just Want to Run It"

1. Read: **BFS_EXPERIMENT_5_CHECKLIST.md** (5 min)
2. Run: `python scripts/xgb_v3_2_l1_regularization.py` (20 min)
3. Check: Grep for "Decision:" in output
4. Done!

### Scenario 2: "I Want to Understand Before Running"

1. Read: **BFS_EXPERIMENT_5_QUICK_START.md** (10 min)
2. Skim: **BFS_EXPERIMENT_5_EXPECTED_OUTPUTS.md** - sections 1-3 (10 min)
3. Run: Script (20 min)
4. Reference: EXPECTED_OUTPUTS.md for interpretation

### Scenario 3: "I Need Complete Understanding"

1. Read: **BFS_EXPERIMENT_5_L1_REGULARIZATION.md** (20 min)
2. Read: **BFS_EXPERIMENT_5_QUICK_START.md** (10 min)
3. Understand: Root cause analysis, architecture
4. Run: Script (20 min)
5. Interpret: Using EXPECTED_OUTPUTS.md

### Scenario 4: "Something Failed or I'm Confused"

1. Check: **BFS_EXPERIMENT_5_CHECKLIST.md** - Troubleshooting section
2. Reference: **BFS_EXPERIMENT_5_EXPECTED_OUTPUTS.md** - what should happen
3. Search: Your specific error in QUICK_START.md
4. Debug: See technical details in L1_REGULARIZATION.md

---

## Quick Navigation by Topic

### "How do I run this?"
→ **BFS_EXPERIMENT_5_CHECKLIST.md** - "Execution" section

### "What will the output look like?"
→ **BFS_EXPERIMENT_5_EXPECTED_OUTPUTS.md** - "Section 1-9" examples

### "What do these results mean?"
→ **BFS_EXPERIMENT_5_EXPECTED_OUTPUTS.md** - "Interpreting the Results" OR
→ **BFS_EXPERIMENT_5_QUICK_START.md** - "Interpreting the Results"

### "What happens if MAE = 41.23?"
→ **BFS_EXPERIMENT_5_EXPECTED_OUTPUTS.md** - "Scenario 2: DEPLOY"

### "Why is it taking so long?"
→ **BFS_EXPERIMENT_5_QUICK_START.md** - "Monitor Progress" OR
→ **BFS_EXPERIMENT_5_CHECKLIST.md** - "Troubleshooting"

### "What's the next step after it finishes?"
→ **BFS_EXPERIMENT_5_QUICK_START.md** - "Next Steps Based on Results" OR
→ **BFS_EXPERIMENT_5_L1_REGULARIZATION.md** - "Interpreting Results" section

### "What's L1 regularization anyway?"
→ **BFS_EXPERIMENT_5_L1_REGULARIZATION.md** - "Root Cause Analysis" & "Script Architecture"

### "How do I load and use the model?"
→ **BFS_EXPERIMENT_5_QUICK_START.md** - "Loading & Using the Model" OR
→ **BFS_EXPERIMENT_5_EXPECTED_OUTPUTS.md** - "Reading the Metadata JSON"

### "What if I get an error?"
→ **BFS_EXPERIMENT_5_CHECKLIST.md** - "Troubleshooting" OR
→ **BFS_EXPERIMENT_5_QUICK_START.md** - "Troubleshooting"

### "What are all the grid search results?"
→ **BFS_EXPERIMENT_5_EXPECTED_OUTPUTS.md** - "Summary Table: All 24 Grid Points"

---

## Decision Matrix Quick Reference

| Final MAE | Decision | Next Action |
|-----------|----------|------------|
| **< 40.2** | ✓ BREAKTHROUGH | Deploy to ensemble immediately |
| **40.2-42.0** | ✓ DEPLOY | Integrate ensemble v6.2 (20% weight) |
| **42.0-45.0** | ⚠ INVESTIGATE | Run feature ablation study |
| **≥ 45.0** | ✗ ABANDON | Return to v2 baseline |

---

## File Locations

### Script
```
scripts/xgb_v3_2_l1_regularization.py
```

### Documentation
```
docs/BFS_EXPERIMENT_5_L1_REGULARIZATION.md
docs/BFS_EXPERIMENT_5_QUICK_START.md
docs/BFS_EXPERIMENT_5_EXPECTED_OUTPUTS.md
BFS_EXPERIMENT_5_CHECKLIST.md
```

### Output (Created After Running)
```
models/xgboost/passing_xgboost_v3_2_l1.pkl
models/xgboost/passing_xgboost_v3_2_l1_metadata.json
```

---

## Quick Facts

**Script Size:** 999 lines of Python
**Documentation:** 54 KB across 4 files
**Total Combinations Tested:** 24 (8 alphas × 3 lambdas)
**Models Trained:** 144 total (24 combos × 5 CV folds + 24 final models)
**Expected Runtime:** 15-25 minutes
**Test Set Size:** 148 games from 2025 season
**Training Set Size:** 6,099 games from 2018-2024
**Features:** 16 (13 baseline + 3 opponent)
**Cross-Validation:** 5-fold

---

## What to Expect

### During Execution
1. Data loading (~2-3 min) - quiet
2. Feature engineering (~1-2 min) - quiet
3. Grid search (~10-15 min) - verbose logging of each combination
4. Feature analysis (~2-3 min) - detailed feature importance output
5. Final training (~1-2 min) - training progress
6. Results (~1 min) - summary statistics

### After Completion
1. Decision statement (DEPLOY, INVESTIGATE, or ABANDON)
2. Model file saved
3. Metadata JSON with complete results
4. Log file with all details for review

---

## Success Criteria

Script execution is successful if:

- [x] Completes without errors
- [x] Grid search converges (top 3 results similar)
- [x] Feature importance shows opponent features diminished at high alpha
- [x] Final MAE in reasonable range (38-50 yards)
- [x] Model and metadata saved successfully
- [x] Decision gate produces clear recommendation

---

## Connection to Other Work

This is part of a larger research framework:

- **BFS 1:** Data validation
- **BFS 2:** Baseline establishment (v2)
- **BFS 3:** Feature exploration
- **BFS 4:** Opponent feature engineering (v3.1 → regression)
- **BFS 5:** L1 regularization (this experiment) ← FIX REGRESSION
- **BFS 6:** Ensemble integration (if BFS 5 succeeds)
- **BFS 7:** Production deployment

---

## Support & Questions

**For questions about:** → **Check:**

Syntax/imports → Try: `python -m py_compile scripts/xgb_v3_2_l1_regularization.py`

Database connection → Check: `pg_isready -h localhost -p 5544`

Feature availability → Check: `ls data/processed/features/opponent_strength_features.csv`

Output interpretation → Read: `docs/BFS_EXPERIMENT_5_EXPECTED_OUTPUTS.md`

Pre-execution → Read: `BFS_EXPERIMENT_5_CHECKLIST.md`

Post-execution → Read: `docs/BFS_EXPERIMENT_5_QUICK_START.md`

Technical deep-dive → Read: `docs/BFS_EXPERIMENT_5_L1_REGULARIZATION.md`

---

## Summary

You have everything needed to:

1. ✅ **Run** the L1 regularization experiment (complete script)
2. ✅ **Understand** what it's doing (4 documentation files)
3. ✅ **Interpret** the results (expected output examples)
4. ✅ **Make decisions** based on outcomes (decision matrix & next steps)

**Next action:** Choose your reading path above, then run the script!

---

**Created:** November 14, 2025
**Status:** ✅ Ready for Execution
**Questions?** Check the Quick Navigation section above
