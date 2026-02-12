# CLAUDE.md Update Plan - November 7, 2025

**Date Prepared:** November 7, 2025
**Current CLAUDE.md Version:** 4.3 (Last Updated: November 3, 2025)

---

## Executive Summary

The CLAUDE.md file requires updates to reflect critical validation findings from November 7, 2025. A comprehensive historical backtest of the 5-way ensemble revealed production blockers that must be documented in the Current Status section. A new "Recent Work" section entry is required to document the investigation.

---

## Sections Requiring Updates

### 1. Header Metadata (Lines 1-5)

**Current:**
```
**Last Updated:** November 3, 2025
**Version:** 4.3
**Project Status:** Phase 5 Feature Engineering Complete, v3.1 Multi-Output BNN Complete, BNN Calibration Complete, Production Deployment Ready
```

**Required Change:**
```
**Last Updated:** November 7, 2025
**Version:** 4.4
**Project Status:** Phase 5 Feature Engineering Complete, v3.1 Multi-Output BNN Complete, BNN Calibration Complete, Phase 2 (5-way Ensemble) BLOCKED - Production Validation Backtest Failed
```

---

### 2. Ensemble Methods Section (Lines 341-365)

**Current Status:** Lists "4-way ensemble" with architecture and usage examples

**Required Updates:**
1. Change "4-way ensemble" to "5-way ensemble" (add BNN Calibrated as 5th model)
2. Update architecture list to include 5 models instead of 4
3. Add note about calibration failure blockers

**New Architecture Section:**

```markdown
### 3. Ensemble Methods

**Location**: `py/ensemble/enhanced_ensemble_v3.py`, `py/backtest/historical_validation_5way.py`
**Status**: ✅ v3.0 Complete (Ensemble Built), ❌ BLOCKED - Validation Failures (Nov 7, 2025)

**Architecture**: 5-way ensemble
1. Bayesian Hierarchical (brms/Stan) - `models/bayesian/passing_informative_priors_v1.rds`
2. XGBoost (GPU-accelerated) - `models/xgboost/v2_1.pkl`
3. Bayesian Neural Network (PyMC) - `models/bayesian/bnn_passing_v1.pkl`
4. BNN Calibrated (Temperature-Scaled) - `py/models/bnn_calibrated_v2.py`
5. State-Space Model (Time-varying ratings) - `models/bayesian/player_skill_trajectories_v1.csv`

**Additional Model:**
- **QB-WR Chemistry** (v3.0, receiving only): `receiving_qb_chemistry_v1.rds` (210MB)
  - Dyadic random effects for QB-WR interaction effects
  - Separate model for receiving yards predictions

**Combination Methods**:
- **Default**: Inverse variance weighting (no meta-learner required)
- **Advanced**: Stacked meta-learner with cross-validation
- **Portfolio**: Correlation-adjusted Kelly criterion
- **CQL Bet Sizer**: Conservative Q-Learning bet sizing (+24% improvement vs Kelly)

**Critical Issues (Nov 7, 2025)**:
- ⚠️ **Edge Calculation Bug**: Ensemble predictions not aligned with actual odds
- ⚠️ **Calibration Failure**: 11.3% calibration coverage (target: 85-95%)
- ⚠️ **State-Space Coverage**: 10% coverage (target: 90%+)
- ⚠️ **Severe Drawdown**: -326.7% bankroll loss on historical test (2022-2025, 2,120 bets)

**Historical Validation Results (Nov 7, 2025)**:
- **Test Period:** 2022-2025 Week 10 (2,120 bets, 4 seasons)
- **Win Rate:** 52.3% (above random but below professional baseline)
- **ROI:** -1.14% (loss territory)
- **Bankroll Drawdown:** -90% (from 10,000 to 1,005)
- **Sharpe Ratio:** -0.019 (negative)
- **Calibration (90% CI):** 11.3% (should be 85-95%)
- **Diagnostic:** Multiple model failure modes identified

**Detailed Breakdown:**
- Total Wagered: 791,149
- Total P&L: -8,995
- Wins: 1,094 | Losses: 997 | Pushes: 29
- Max Drawdown: -32,669

**Status:** Phase 2 (5-way ensemble production deployment) **BLOCKED** pending root cause analysis and fixes.

---

**Usage** (deprecated - do not use for production):
```bash
uv run python py/backtest/historical_validation_5way.py \
  --start-season 2022 \
  --end-season 2025 \
  --use-cql \
  --bankroll 10000 \
  --output predictions/ensemble_v3_week8.csv
```
```

---

### 3. Current Status Section (Lines 593-639)

**Location:** Lines 593-639

**Current Content:**
- Recent Completions (October 2025)
- Active Work
- Known Issues

**Required Changes:**

**A. Update "Recent Completions" subsection - Modify the "Known Issues" part:**

**Current Known Issues (Lines 634-639):**
```
### Known Issues

1. **State-space model**: Database insert logging issue (model saved successfully)
2. **BNN calibration**: Under-calibration (~26% vs 90% target) - architectural, not prior-related
3. **Database inserts**: Some models trained successfully but DB insert failed (logging errors)
```

**Replace with:**
```
### Known Issues (CRITICAL - Blocking Production)

**Phase 2 (5-Way Ensemble) - BLOCKED Nov 7, 2025:**
1. **Edge Calculation Bug**: Predictions not aligned with Vegas odds
   - Impact: Ensemble betting edge is zero or negative
   - Root Cause: Under investigation
   - Fix: Requires re-architecture of prediction to odds pipeline

2. **BNN Calibration Failure**: 11.3% coverage (target: 85-95%)
   - Note: Temperature scaling (T=0.75) achieved 89.5% on synthetic validation
   - Issue: Real data shows different failure mode
   - Root Cause: Data distribution mismatch or target leakage
   - Impact: Confidence intervals not trustworthy for risk management

3. **State-Space Model Coverage**: 10% (target: 90%+)
   - Impact: Time-varying ratings unreliable
   - Status: Under investigation

**Immediate Actions:**
- ⚠️ Do NOT deploy 5-way ensemble to production
- ⚠️ Disable ensemble predictions in betting pipeline
- ⚠️ Investigate edge calculation and recalibrate models
- Historical validation results: `analysis/results/historical_validation_5way_2022_2025_metrics.json`
```

---

### 4. Recent Work Section - Add New Entry (After line 843)

**Location:** After the existing "November 2-3, 2025" section (after line 841)

**New Section to Add:**

```markdown
---

### November 7, 2025: Historical Validation Backtest - 5-Way Ensemble BLOCKED

**Context**: Executed comprehensive historical validation of the 5-way ensemble (Bayesian Hierarchical, XGBoost, BNN, BNN Calibrated, State-Space) with CQL bet sizing on 2022-2025 data. Critical production blockers identified.

**Work Performed**:
1. **5-Way Ensemble Validation** (2022-2025 Week 10, 4 seasons, 2,120 bets)
   - Test Period: Complete season blocks from 2022 through Week 10 of 2025
   - Ensemble Architecture: Bayesian + XGBoost + BNN + BNN Calibrated + State-Space
   - Bet Sizing: CQL agent with risk gates
   - Bankroll: $10,000 starting
   - **Results**:
     * Win Rate: 52.3% (above random, below professional baseline)
     * ROI: **-1.14%** (loss territory)
     * Total P&L: **-$8,995** (net loss)
     * Bankroll Remaining: **$1,005** (-90% drawdown)
     * Max Drawdown: -32,669 (-326.7%)
     * Sharpe Ratio: -0.019 (negative)
     * Total Wagered: $791,149

2. **Calibration Analysis** - Multiple Failures Identified
   - **BNN Calibration (90% CI):** 11.3% coverage (target: 85-95%)
     * Gap Analysis: Real data shows different failure mode than synthetic validation
     * Note: Temperature scaling (T=0.75) achieved 89.5% on held-out training data
     * Hypothesis: Target leakage or data distribution mismatch in live validation
   - **State-Space Coverage:** 10% (target: 90%+)
     * Indicates time-varying player ratings not capturing true dynamics
   - **Ensemble Calibration:** Combined model shows poor uncertainty quantification

3. **Root Cause Investigation** - Multiple Failure Modes Identified
   - **Edge Calculation Bug**: Ensemble predictions not properly aligned with Vegas odds
     * Impact: Actual betting edge is zero or negative
     * Symptom: Win rate barely above random despite model accuracy
   - **Model Disagreement**: 5 models may be contradicting each other
     * Variance weighting might be masking poor individual predictions
   - **CQL Bet Sizing Failure**: Kelly-style bet sizing not compensating for poor predictions
     * Issue: CQL model trained on synthetic data, not live conditions
   - **Potential Target Leakage**: Under investigation
     * Hypothesis: Some form of forward-looking data in features

4. **Diagnostic Outputs** Created
   - `analysis/results/historical_validation_5way_2022_2025.csv` - Trade-by-trade detailed results
   - `analysis/results/historical_validation_5way_2022_2025_metrics.json` - Aggregate metrics
   - Trade-level analysis: Individual bet outcomes, predicted vs actual

**Files Created** (Nov 7, 2025):
- `py/backtest/historical_validation_5way.py` - Main validation script (536 lines)
- `analysis/results/historical_validation_5way_2022_2025.csv` - Trade results
- `analysis/results/historical_validation_5way_2022_2025_metrics.json` - Metrics summary
- `CLAUDE_MD_UPDATE_PLAN_NOV7.md` - This update plan

**Key Findings**:
1. **Production Blockers Identified**: 5-way ensemble NOT ready for deployment
2. **Calibration Failure**: Real-world validation shows different failure mode than lab tests
3. **Edge Problem**: Betting edge appears to be zero or negative despite model accuracy
4. **Multiple Root Causes**: Edge calculation, model disagreement, CQL, and potential leakage

**Status**: Phase 2 (5-way ensemble deployment) **BLOCKED** pending:
- Root cause analysis on edge calculation
- Re-validation with corrected feature pipeline
- BNN recalibration on production data
- State-space model diagnostic

**Result**: Critical validation findings documenting production blockers. Do NOT deploy ensemble to production. Requires investigation and architectural fixes before advancement to Phase 3 (real-money deployment).

---
```

---

## Summary of Changes

### File: `CLAUDE.md`

**Changes Required:**

1. **Header (lines 1-5):** Update version to 4.4, last updated date to Nov 7, add status note about Phase 2 BLOCKED

2. **Ensemble Methods (lines 341-365):**
   - Change 4-way to 5-way ensemble
   - Add detailed architecture with 5 models + QB-WR Chemistry model
   - Add critical issues section documenting Nov 7 findings
   - Add historical validation results with metrics

3. **Current Status - Known Issues (lines 634-639):**
   - Replace with new critical blockers section
   - Document Phase 2 BLOCKED status
   - List immediate actions

4. **Recent Work Section (after line 841):**
   - Add new "November 7, 2025" section with comprehensive findings

---

## Implementation Notes

**Do NOT make these edits yet.** This document serves as a specification for the exact changes needed.

**Why Changes Are Critical:**

1. **Version Management**: Update version to 4.4 to track this major status change
2. **AI Assistant Guidance**: Claude must know about production blockers before suggesting changes
3. **Traceability**: Link backtest results to specific metrics and recommendations
4. **Risk Mitigation**: Explicit warnings prevent accidental production deployment
5. **Future Context**: Documents the investigation timeline for future phases

**Validation:**

The metrics in this plan are sourced from:
- File: `analysis/results/historical_validation_5way_2022_2025_metrics.json`
- Test: 2,120 bets from 2022-2025 Week 10
- Models: 5-way ensemble with CQL bet sizing
- Bankroll: $10,000 starting, $1,005 ending (-90%)

---

**Prepared by:** Claude Code
**Date:** November 7, 2025
**Status:** Ready for implementation
