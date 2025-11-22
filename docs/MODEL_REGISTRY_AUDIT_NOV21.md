# Model Registry Audit (November 21, 2025)

**Status**: ⚠️ REGISTRY OUT OF SYNC WITH PRODUCTION
**Production Version**: Ensemble v6.1 (PostgreSQL R Lookup)
**Registry Version**: Ensemble v5.0

---

## Issues Found (4 Critical)

### 1. ❌ XGBoost v4 Ridge: Not Marked as Production
**Status**: `is_production = false` (should be `true`)
**Current**: `status = 'active'`, `is_production = false`
**Expected**: `status = 'production'`, `is_production = true`
**Model ID**: 40
**Impact**: Production model not marked as production-ready in registry

### 2. ❌ Ensemble v6.1: Not Registered
**Status**: Missing from registry
**Current**: Only `ensemble_v5.0` (model_id 38) is registered
**Expected**: `ensemble_v6.1` should be registered
**Impact**: Latest production ensemble not tracked

### 3. ❌ Ensemble v6.0: Not Registered
**Status**: Missing from registry
**Current**: v6.0 was deployed Nov 15, but never registered
**Expected**: Should be marked as superseded by v6.1
**Impact**: Missing production deployment history

### 4. ⚠️ R Models: Using PostgreSQL Lookup (Architecture Change)
**Status**: R models now accessed via PostgreSQL, not direct file loading
**Current Registry**: Points to .rds files
**Actual Usage**: PostgreSQL table `ml.r_model_predictions`
**Impact**: Model paths are technically correct (batch generation still uses .rds files), but architecture has fundamentally changed

---

## Production Models (Ensemble v6.1 Components)

| Model | Registry Status | is_production | Actual Status | Issue |
|-------|----------------|---------------|---------------|-------|
| state_space_v2 | ✅ Registered (ID 11) | ✅ true | ✅ Production | None |
| hierarchical_v2 | ✅ Registered (ID 13) | ✅ true | ✅ Production | None |
| xgboost_v4_ridge | ⚠️ Registered (ID 40) | ❌ **false** | ✅ Production | **Not marked production** |
| informative_priors_v2 | ✅ Registered (ID 16) | ✅ true | ✅ Production | None |
| hybrid_static_incremental_v1 | ✅ Registered (ID 37) | ✅ true | ✅ Production | None |

---

## Recommended Fixes

### Fix 1: Mark XGBoost v4 Ridge as Production
```sql
UPDATE ml.model_registry
SET
  is_production = true,
  status = 'production',
  updated_at = NOW()
WHERE model_id = 40;
```

### Fix 2: Register Ensemble v6.1
```sql
INSERT INTO ml.model_registry (
  model_name, model_type, model_family, version,
  model_path, status, is_production,
  description
) VALUES (
  'ensemble_v6.1',
  'ensemble',
  'bayesian_ensemble',
  'v6.1',
  'py/ensemble/enhanced_ensemble_v6_production.py',
  'production',
  true,
  'Enhanced 5-way ensemble with PostgreSQL R model lookup (200-400x faster than rpy2). Components: State-Space v2 (30%), Hierarchical v2 (30%), XGBoost v4 Ridge (25%), Informative Priors v2 (15%), Hybrid Static-Incremental (10%). Temperature scaling with T=1.0771.'
);
```

### Fix 3: Deprecate Ensemble v5.0
```sql
UPDATE ml.model_registry
SET
  is_production = false,
  status = 'deprecated',
  notes = 'Superseded by ensemble_v6.1 on November 21, 2025',
  updated_at = NOW()
WHERE model_id = 38;  -- ensemble_v5.0
```

### Fix 4: Add Note About PostgreSQL Lookup Architecture
```sql
UPDATE ml.model_registry
SET
  notes = 'As of Nov 21, 2025: Predictions accessed via PostgreSQL table ml.r_model_predictions (<1ms) instead of rpy2 runtime calls (2-4s). Batch generation still uses this .rds file.',
  updated_at = NOW()
WHERE model_id IN (11, 13, 16);  -- state_space_v2, hierarchical_v2, informative_priors_v2
```

---

## Additional Observations

### Missing Models (227 pending from CLAUDE.md)
The registry shows only 15 production models, but CLAUDE.md mentions "227 pending (auto-registration available)". This suggests a large backlog of unregistered models.

### Registry Last Updated
Most recent production registration appears to be `hybrid_static_incremental_v1` and `temperature_scaler_4way` (no timestamp available in query).

---

## Action Items

1. **Immediate** (Critical):
   - [ ] Mark XGBoost v4 Ridge as production
   - [ ] Register Ensemble v6.1
   - [ ] Deprecate Ensemble v5.0

2. **This Week** (Important):
   - [ ] Add PostgreSQL lookup architecture notes to R models
   - [ ] Audit and register pending 227 models
   - [ ] Verify all model paths are accurate

3. **This Month** (Nice to Have):
   - [ ] Add performance metrics to registered models (MAE, calibration, etc.)
   - [ ] Set up automated registry updates on deployment
   - [ ] Create model lineage tracking (which model supersedes which)

---

**Generated**: November 21, 2025
**Auditor**: Claude Code (Sonnet 4.5)
