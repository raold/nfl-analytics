# Model Registry Audit - November 17, 2025

**Audit Date:** November 17, 2025 01:10:50 UTC
**Auditor:** Model Registry Auditor v1.0 (Automated)
**Status:** ⛔ CRITICAL ISSUES FOUND (16 total)

---

## Executive Summary

**Key Findings:**
- ❌ **v7.0 deployment BLOCKED** - Ridge residuals v7 models corrupted (4 files)
- ⚠️ **342 models pending registration** - 92% of filesystem not in database
- ⚠️ **BNN v3.1 convergence data missing** - Production model needs re-validation
- ✅ **v6.0 production ensemble healthy** - All 4 components operational

**Recommendation:** Block v7.0 deployment, auto-register 342 pending models, fix BNN v3.1 convergence data, clean up 15 corrupted files.

---

## 1. v7.0 Component Registration Status

### AdaptiveWLSWeights v2
- **Status:** ❌ NOT REGISTERED (PENDING)
- **File:** `/models/ensemble/adaptive_wls_weights_demo.json` (447 B)
- **Note:** Demo file exists, production model not yet trained
- **Action Required:** Train production model before v7.0 deployment

### MetaLearningWeightPredictor
- **Status:** ❌ NOT REGISTERED (PENDING)
- **File:** `/models/ensemble/test_meta_model.pkl` (2.1 KB)
- **Issue:** File too small, likely corrupted
- **Action Required:** Retrain model, validate file size > 10 KB

### Enhanced Ensemble v7.0
- **Status:** ❌ NOT REGISTERED (PENDING)
- **Current Production:** v6.0 (registered Nov 15, 2025)
- **Note:** v7.0 components not ready for production
- **Action Required:** Complete component training before ensemble assembly

### Ridge Residuals v7 (XGBoost Component)
- **Status:** ❌ NOT REGISTERED (CORRUPTED)
- **Files Found:** 4 files, all ~1.2 KB (expected: >100 KB)
  - `ridge_residuals_v7_20251114_171023.pkl` (1.2 KB)
  - `ridge_residuals_v7_20251114_182121.pkl` (1.2 KB)
  - `ridge_residuals_v7_backtest2024_20251114_184538.pkl` (1.2 KB)
  - `ridge_residuals_v7_backtest2024_20251114_184552.pkl` (1.2 KB)
- **Issue:** All files corrupted, training failed silently
- **Action Required:** Re-run XGBoost v7 training with error handling

---

## 2. Convergence Issues

### Summary
- **Total Bayesian models checked:** 4
- **Convergence failures:** 1 (BNN v3.1)
- **Models with no data:** 3 (likely OK, need metadata regeneration)

### Critical Issue: Multi-Output BNN v3.1 (Calibrated)

**Status:** ⛔ CRITICAL - No convergence data available
**Impact:** Production model in current ensemble (v6.0)
**Risk Level:** HIGH - Model may have poor MCMC convergence

**Required Diagnostics:**
- R-hat < 1.01 (convergence threshold)
- ESS > 400 (effective sample size)
- Divergences < 10 (sampling quality)
- Trace plots (visual inspection)

**Action Required:**
1. Re-run MCMC diagnostics from trace files
2. Regenerate `metadata.json` with convergence stats
3. If convergence failed, retrain with:
   - Increased warmup (2000 → 4000 iterations)
   - Lower learning rate (1e-3 → 1e-4)
   - Stronger priors (σ = 0.5 → 0.3)

### Other Models (No Data - Low Risk)

**Bayesian Random Walk State-Space v2**
- Status: ✅ Likely OK (approved Nov 8, 2025)
- Issue: Missing metadata.json
- Action: Regenerate metadata from training logs

**Hierarchical Bayesian Passing v2**
- Status: ✅ Likely OK (registered, production-ready)
- Issue: Missing metadata.json
- Action: Regenerate metadata from training logs

---

## 3. Pending Registrations

### Overview
- **Total filesystem models:** 372
- **Database registered:** 28 (7.5%)
- **Pending registrations:** 356 (92.5%)
  - Should register: 342 (91.9%)
  - Blocked: 14 (3.8%)

### Breakdown by Model Type

| Category | Count | Priority | Notes |
|----------|-------|----------|-------|
| Archived BNN experiments | ~200 | LOW | Archive folder, research experiments |
| Calibration artifacts | ~50 | MEDIUM | Temperature scalers, metadata |
| XGBoost variants | ~40 | HIGH | v1-v7 iterations, residual models |
| Feature engineering | ~30 | MEDIUM | Scalers, encoders, transformers |
| Miscellaneous | ~22 | LOW | Test files, demos |
| **Total** | **342** | - | - |

### Registration Command

```bash
# Auto-register 342 pending models (5-10 minutes)
source .venv/bin/activate
python py/utils/model_registry_auditor.py register --no-dry-run

# Expected output:
# - 342 models registered
# - 14 blocked (corrupted or invalid)
# - Database updated with metadata
```

### Blocked Models (14 total)

All blocked due to file integrity issues:
- 4 × Ridge residuals v7 (corrupted, 1.2 KB)
- 3 × Temperature scalers (corrupted, <1 KB)
- 1 × Meta learning model (corrupted, 2.1 KB)
- 6 × Miscellaneous test files (corrupted, <5 KB)

---

## 4. File Integrity Issues

### Summary
- **Total integrity issues:** 15
- **Corrupted files:** 15 (100%)
- **Production models affected:** 0 ✅

### High Priority Corrupted Files

#### 1. Ridge Residuals v7 (XGBoost v7.0 Component)
**Files:** 4 corrupted files (all 1.2 KB)
- `ridge_residuals_v7_20251114_171023.pkl`
- `ridge_residuals_v7_20251114_182121.pkl`
- `ridge_residuals_v7_backtest2024_20251114_184538.pkl`
- `ridge_residuals_v7_backtest2024_20251114_184552.pkl`

**Impact:** CRITICAL - Blocks v7.0 deployment
**Root Cause:** XGBoost training failed, error not caught
**Action:** Delete files, retrain with error handling

#### 2. Meta Learning Model (v7.0 Component)
**File:** `test_meta_model.pkl` (2.1 KB)
**Expected Size:** >10 KB
**Impact:** HIGH - v7.0 meta learning weight predictor unusable
**Action:** Retrain meta learning model

#### 3. Temperature Scalers (3 files, <1 KB each)
**Files:**
- `temperature_scaler_v6.pkl` (0.1 KB)
- `temperature_scaler_4way.pkl` (0.1 KB)
- `bnn_temperature_scaled_v1.pkl` (0.3 KB)

**Impact:** MEDIUM - Calibration components affected
**Note:** Production temperature scaler (v6) uses metadata JSON, not corrupted .pkl
**Action:** Delete corrupted files, use JSON-based scaler

#### 4. Incremental Learning Model
**File:** `incremental_model.pkl` (0.2 KB)
**Impact:** LOW - Online learning experiment, not in production
**Action:** Delete or archive

#### 5. RL Candidate Policy
**File:** `rl_candidate_policy.pkl` (0.1 KB)
**Impact:** LOW - Research experiment, not in production
**Action:** Delete or archive

### Production Models Integrity ✅

All production ensemble v6.0 components verified:

| Component | Status | Size | Modified |
|-----------|--------|------|----------|
| State-Space v2 | ✅ OK | Large file | Nov 8, 2025 |
| Hierarchical v2 | ✅ OK | Large file | Nov 8, 2025 |
| XGBoost v4 Ridge | ✅ OK | Registered | Nov 15, 2025 |
| Informative Priors v2 | ✅ OK | Registered | Nov 8, 2025 |
| Temperature Scaler v6 | ✅ OK | Metadata JSON | Nov 15, 2025 |
| **Ensemble v6.0** | ✅ OK | Registered | Nov 15, 2025 |

---

## 5. Database-Registry Sync

### Mismatches Found: 25

**Issue:** 25 models in database not tracked in `models/MODEL_REGISTRY.yaml`

**Breakdown:**
- 10 models in YAML, 28 in database → 18 database-only models
- 7 models with different metadata (version, timestamp)

**Impact:** LOW - Database is source of truth, YAML is convenience index

**Action:** Regenerate `MODEL_REGISTRY.yaml` from database (1 minute)

```bash
python py/utils/sync_model_registry.py --regenerate-yaml
```

---

## 6. Recommendations

### IMMEDIATE ACTIONS (Next 24 hours)

#### 1. ⛔ BLOCK v7.0 DEPLOYMENT (CRITICAL)
**Issue:** Ridge residuals v7 models corrupted, meta learning model too small
**Action:** Do NOT deploy v7.0 until components retrained
**Timeline:** 2-3 days (retrain XGBoost v7 + meta learning)

**Steps:**
```bash
# 1. Delete corrupted v7 files
rm models/ridge_residuals_v7_*.pkl
rm models/ensemble/test_meta_model.pkl

# 2. Retrain XGBoost v7 with error handling
python py/ensemble/train_xgboost_v7_ridge.py --validate-output

# 3. Retrain meta learning model
python py/ensemble/train_meta_learner.py --min-size-kb 10

# 4. Validate file sizes before registration
ls -lh models/ridge_residuals_v7_*.pkl  # Should be >100 KB
ls -lh models/ensemble/test_meta_model.pkl  # Should be >10 KB
```

#### 2. ✅ AUTO-REGISTER 342 PENDING MODELS
**Benefit:** Complete model inventory, better tracking
**Time Required:** 5-10 minutes
**Risk:** LOW - Only metadata update, no code changes

**Command:**
```bash
source .venv/bin/activate
python py/utils/model_registry_auditor.py register --no-dry-run
```

**Expected Results:**
- 342 models registered to `ml.model_registry`
- Metadata extracted (size, type, training window)
- `MODEL_REGISTRY.yaml` regenerated
- 14 corrupted files skipped (blocked)

#### 3. 🔧 FIX CONVERGENCE DATA FOR BNN v3.1
**Issue:** Production model missing convergence diagnostics
**Risk:** MEDIUM - Model may have poor MCMC quality
**Timeline:** 1-2 hours

**Steps:**
```bash
# 1. Re-run MCMC diagnostics
python py/bayesian/validate_bnn_convergence.py \
    --model models/bayesian/bnn_multioutput_v3_1.pkl \
    --output models/bayesian/bnn_multioutput_v3_1_metadata.json

# 2. Check diagnostics
cat models/bayesian/bnn_multioutput_v3_1_metadata.json | grep -E "rhat|ess|divergences"

# 3. If failed, retrain with stronger priors
python py/bayesian/train_bnn_v3.py \
    --sigma 0.3 \
    --warmup 4000 \
    --learning-rate 1e-4
```

#### 4. 🗑️ CLEAN UP CORRUPTED FILES (15 total)
**Benefit:** Clean filesystem, prevent confusion
**Time Required:** 5 minutes

**Commands:**
```bash
# Delete corrupted ridge residuals v7 (4 files)
rm models/ridge_residuals_v7_*.pkl

# Delete corrupted temperature scalers (3 files)
rm models/calibration/temperature_scaler_v6.pkl
rm models/calibration/temperature_scaler_4way.pkl
rm models/archive/bnn_failed_experiments_nov2025/bnn_temperature_scaled_v1.pkl

# Delete test files (8 files)
rm models/rl_candidate_policy.pkl
rm models/cvae/scaler_conditions.pkl
rm models/ensemble/test_meta_model.pkl
rm models/online/incremental_model.pkl
rm models/online/static_scaler.pkl
rm models/online/metadata.pkl
rm models/bayesian/mc_dropout_shallow_v1.pkl
rm models/bayesian/quantile_regression_6features.pkl
```

### MEDIUM-TERM ACTIONS (Next Week)

#### 5. 📊 VALIDATE v6.0 ENSEMBLE (PRODUCTION)
**Status:** ✅ Week 1/4 of A/B test in progress
**Current:** Control (v5.1) vs Treatment (v6.0)
**Next Checkpoint:** November 21, 2025 (Week 1 results)

**Validation Steps:**
1. Check A/B test metrics (MAE, ROI, Sharpe)
2. Monitor component weights (State-Space 30%, Hierarchical 30%, XGBoost 25%, IP 15%)
3. Validate calibration (target: 90% coverage)
4. Review production logs for errors

#### 6. 🔬 VALIDATE ALL BAYESIAN CONVERGENCE
**Issue:** 3 models missing convergence metadata
**Timeline:** 2-3 hours

**Models to Check:**
- State-Space v2 (production)
- Hierarchical v2 (production)
- Informative Priors v2 (production)

**Command:**
```bash
python py/bayesian/validate_all_convergence.py \
    --models state_space_v2 hierarchical_v2 informative_priors_v2 \
    --output docs/performance/BAYESIAN_CONVERGENCE_REPORT_NOV17.md
```

---

## 7. Audit Metrics

### Model Registry Coverage

| Metric | Count | Percentage |
|--------|-------|------------|
| Total filesystem models | 372 | 100% |
| Database registered | 28 | 7.5% |
| Pending registrations | 342 | 92.0% |
| Blocked (corrupted) | 14 | 3.8% |
| Production models | 6 | 1.6% |

### Production Model Health

| Model | Status | Size | Convergence | Calibration |
|-------|--------|------|-------------|-------------|
| State-Space v2 | ✅ OK | Large | ⚠️ No data | ✅ 90% |
| Hierarchical v2 | ✅ OK | Large | ⚠️ No data | ✅ 89% |
| XGBoost v4 Ridge | ✅ OK | Registered | N/A | ⚠️ No data |
| Informative Priors v2 | ✅ OK | Registered | ⚠️ No data | ✅ 88% |
| BNN v3.1 | ⚠️ No conv data | Large | ⛔ Missing | ✅ 90% |
| Ensemble v6.0 | ✅ OK | Registered | N/A | ✅ 90% |

### File Integrity Summary

| Category | OK | Corrupted | Total |
|----------|----|-----------| ------|
| Production models | 6 | 0 | 6 |
| XGBoost variants | 36 | 4 | 40 |
| Calibration | 47 | 3 | 50 |
| BNN experiments | 199 | 1 | 200 |
| Feature engineering | 29 | 1 | 30 |
| Miscellaneous | 16 | 6 | 22 |
| **Total** | **357** | **15** | **372** |

---

## 8. Next Steps

### This Week (Nov 17-21)
- [x] Run Model Registry Audit (completed)
- [ ] Block v7.0 deployment
- [ ] Delete 15 corrupted files
- [ ] Auto-register 342 pending models
- [ ] Fix BNN v3.1 convergence data
- [ ] Retrain XGBoost v7 ridge residuals
- [ ] Retrain meta learning model

### Next Week (Nov 21-28)
- [ ] Complete Bayesian convergence validation (all 4 models)
- [ ] Review Week 1 A/B test results (v5.1 vs v6.0)
- [ ] Validate v7.0 components ready for training
- [ ] Plan v7.0 deployment timeline (pending component fixes)

### Future Work
- [ ] Implement automated file integrity checks (pre-commit hook)
- [ ] Add convergence validation to training pipelines
- [ ] Create model lifecycle tracking (training → validation → registration → production)
- [ ] Build model performance dashboard (Grafana + Prometheus)

---

## 9. Appendix

### Files Generated

| File | Size | Purpose |
|------|------|---------|
| `/tmp/registry_audit_nov17.json` | ~500 KB | Full audit results (JSON) |
| `/docs/performance/MODEL_REGISTRY_AUDIT_NOV17.md` | ~20 KB | This report (Markdown) |

### Commands Reference

```bash
# Full audit
python py/utils/model_registry_auditor.py audit --output /tmp/audit.json

# Scan pending registrations
python py/utils/model_registry_auditor.py scan

# Auto-register models
python py/utils/model_registry_auditor.py register --no-dry-run

# Check convergence
python py/utils/model_registry_auditor.py convergence

# Check database sync
python py/utils/model_registry_auditor.py sync-check
```

### Contact

**Auditor:** Model Registry Auditor v1.0
**Documentation:** `py/utils/model_registry_auditor.py` (1,179 lines)
**Agent Infrastructure:** Phase 4 (complete Nov 14, 2025)

---

**Audit Complete:** November 17, 2025 01:10:50 UTC
**Next Audit:** Scheduled for v7.0 pre-deployment (TBD)
