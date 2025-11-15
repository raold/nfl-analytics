# API Consolidation Analysis - EnhancedEnsemble6Way Cleanup

**Date:** November 15, 2025
**Issue:** Multiple conflicting versions of `EnhancedEnsemble6Way` causing TypeError
**Root Cause:** Duplicate class definitions with incompatible __init__ signatures

---

## Files Found

### 1. Canonical Version (Production)
**File:** `py/ensemble/enhanced_ensemble_v6.py`
**Class:** `EnhancedEnsemble4Way` (NOT 6Way!)
**Size:** 41 KB
**Modified:** 2025-11-14 21:38:40
**Status:** ✅ CANONICAL (most recent, in production)

**__init__ Parameters:**
- `state_space_path`
- `hierarchical_path`
- `informative_path`
- `ridge_path` (XGBoost v4 Ridge Residuals)
- `xgboost_path`
- `weights`
- `temperature_scaler_path`
- `db_host`, `db_port`, `db_name`, `db_user`, `db_password`

**Architecture:** 4-way ensemble with XGBoost v4 Ridge Residuals (NEW)

---

### 2. Deprecated Backup
**File:** `py/ensemble/enhanced_ensemble_v6_DEPRECATED_BAK.py`
**Class:** `EnhancedEnsemble6Way`
**Size:** 27 KB
**Modified:** 2025-11-14 14:06:56
**Status:** ⚠️ DEPRECATED (backup, older version)

**__init__ Parameters:**
- `state_space_path`
- `hierarchical_path`
- `bnn_path` (old architecture)
- `hybrid_xgboost_path` (old architecture)
- `db_host`, `db_port`, `db_name`, `db_user`, `db_password`

**Architecture:** 4-way ensemble with BNN and hybrid XGBoost (DEPRECATED)

---

### 3. NEW Version (Experimental)
**File:** `py/ensemble/enhanced_ensemble_v6_NEW.py`
**Class:** `EnhancedEnsemble6Way`
**Size:** 28 KB
**Modified:** 2025-11-14 13:59:23
**Status:** ⚠️ EXPERIMENTAL (not in production)

**__init__ Parameters:**
- `state_space_path`
- `hierarchical_path`
- `informative_path`
- `xgboost_path`
- `weights`
- `temperature_scaler_path`
- `db_host`, `db_port`, `db_name`, `db_user`, `db_password`

**Architecture:** 5-way ensemble (State-Space 30%, Hierarchical 30%, XGBoost 25%, Informative 15%)

---

### 4. ELO + WLS Version (Experimental)
**File:** `py/ensemble/enhanced_ensemble_v6_elo.py`
**Class:** `EnhancedEnsemble6Way`
**Size:** 39 KB
**Modified:** 2025-11-15 09:20:03  (MOST RECENT)
**Status:** ⚠️ EXPERIMENTAL (ELO + adaptive WLS research)

**__init__ Parameters:**
- `state_space_path`
- `hierarchical_path`
- `informative_path`
- `xgboost_path`
- `elo_ratings_path` (NEW - ELO ratings)
- `weights`
- `adaptive_weights_path` (NEW - WLS weights)
- `temperature_scaler_path`
- `db_host`, `db_port`, `db_name`, `db_user`, `db_password`

**Architecture:** 5-way ensemble with ELO (27%, 27%, 23%, 13%, 10% ELO)

---

## Import Analysis

### Files Importing `EnhancedEnsemble6Way`

1. **test_ensemble_v6.py** (root directory)
   - Import: `from py.ensemble.enhanced_ensemble_v6 import EnhancedEnsemble6Way`
   - Issue: ❌ Canonical file exports `EnhancedEnsemble4Way`, not `EnhancedEnsemble6Way`

2. **scripts/elo_wls_multi_season_validation.py** (line 224)
   - Import: `from py.ensemble.enhanced_ensemble_v6_elo import EnhancedEnsemble6Way`
   - Issue: ⚠️ Importing experimental version with `adaptive_weights_path` parameter

3. **scripts/test_elo_wls_fixes.py** (lines 41, 68)
   - Import: `from py.ensemble.enhanced_ensemble_v6_elo import EnhancedEnsemble6Way`
   - Issue: ⚠️ Importing experimental version

4. **py/production/ab_test_router.py** (line 55)
   - Import: `from py.ensemble.enhanced_ensemble_v6 import EnhancedEnsemble4Way as EnsembleV6`
   - Status: ✅ CORRECT (using canonical version with correct class name)

---

## API Signature Comparison

| File | Class Name | Key Parameters | Architecture |
|------|-----------|----------------|-------------|
| `enhanced_ensemble_v6.py` | `EnhancedEnsemble4Way` | `ridge_path`, `xgboost_path` | XGBoost v4 Ridge Residuals |
| `enhanced_ensemble_v6_DEPRECATED_BAK.py` | `EnhancedEnsemble6Way` | `bnn_path`, `hybrid_xgboost_path` | BNN + Hybrid XGB (OLD) |
| `enhanced_ensemble_v6_NEW.py` | `EnhancedEnsemble6Way` | `informative_path`, `xgboost_path` | 5-way baseline |
| `enhanced_ensemble_v6_elo.py` | `EnhancedEnsemble6Way` | `elo_ratings_path`, `adaptive_weights_path` | 5-way + ELO + WLS |

---

## Conflicts and Issues

### Issue 1: Class Name Mismatch
- **Canonical file** (`enhanced_ensemble_v6.py`) exports `EnhancedEnsemble4Way`
- **Test file** (`test_ensemble_v6.py`) imports `EnhancedEnsemble6Way`
- **Result:** ImportError (class not found)

### Issue 2: Parameter Mismatch
- **Validation script** (`elo_wls_multi_season_validation.py`) uses `adaptive_weights_path`
- **TypeError:** `__init__() got an unexpected keyword argument 'use_adaptive_weights'`
- **Root cause:** Script uses `adaptive_weights_path` from experimental version, but canonical version doesn't support it

### Issue 3: Duplicate Definitions
- 4 files define ensemble classes with same or similar names
- No clear versioning or deprecation strategy
- Ambiguous which version is "canonical"

---

## Canonical Version Decision

**Criteria:**
1. ✅ Most recent modification date: `enhanced_ensemble_v6.py` (Nov 14 21:38:40)
2. ✅ Used by production code: `ab_test_router.py` imports from this file
3. ✅ In standard location: `py/ensemble/` (not `scripts/`)
4. ✅ Has comprehensive methods: Full predict, calibrate, fallback logic
5. ✅ XGBoost v4 Ridge Residuals (latest architecture)

**Decision:** `py/ensemble/enhanced_ensemble_v6.py` is canonical

---

## Recommended Actions

### Phase 1: Archive Deprecated Files (IMMEDIATE)
1. Move `enhanced_ensemble_v6_DEPRECATED_BAK.py` to `archive/ensemble/`
2. Move `enhanced_ensemble_v6_NEW.py` to `archive/ensemble/`
3. Move `enhanced_ensemble_v6_elo.py` to `archive/ensemble/` (or keep for research)

### Phase 2: Fix Import References (IMMEDIATE)
1. **test_ensemble_v6.py**
   - Change: `from py.ensemble.enhanced_ensemble_v6 import EnhancedEnsemble6Way`
   - To: `from py.ensemble.enhanced_ensemble_v6 import EnhancedEnsemble4Way`
   - Or rename canonical class to `EnhancedEnsemble6Way` for consistency

2. **elo_wls_multi_season_validation.py**
   - Option A: Keep using experimental version (`enhanced_ensemble_v6_elo.py`)
   - Option B: Update canonical version to support ELO + WLS
   - Recommendation: Option A (keep research separate)

3. **test_elo_wls_fixes.py**
   - Same as above

### Phase 3: Rename Canonical Class (OPTIONAL)
- Rename `EnhancedEnsemble4Way` → `EnhancedEnsemble6Way` in canonical file
- Update all production imports accordingly
- Justification: Version number (v6) should match class name (6Way)

### Phase 4: Create API Contract Test (REQUIRED)
```python
# tests/unit/test_ensemble_v6_api.py
def test_ensemble_v6_api_signature():
    """Ensure API stability for production ensemble."""
    from py.ensemble.enhanced_ensemble_v6 import EnhancedEnsemble4Way

    import inspect
    sig = inspect.signature(EnhancedEnsemble4Way.__init__)
    params = list(sig.parameters.keys())

    # Required params
    assert 'state_space_path' in params
    assert 'hierarchical_path' in params
    assert 'informative_path' in params
    assert 'ridge_path' in params
    assert 'xgboost_path' in params

    # Should NOT have deprecated params
    assert 'bnn_path' not in params
    assert 'hybrid_xgboost_path' not in params
    assert 'use_adaptive_weights' not in params
```

---

## Success Criteria

✅ Only ONE canonical ensemble file in `py/ensemble/`
✅ All deprecated versions archived to `archive/ensemble/`
✅ All production imports point to canonical version
✅ API contract test passes
✅ No `use_adaptive_weights` parameter anywhere
✅ Clear separation between production and research code

---

## Time Estimate

- Phase 1 (Archive): 30 minutes
- Phase 2 (Fix imports): 1 hour
- Phase 3 (Rename class): 1 hour (optional)
- Phase 4 (API tests): 30 minutes

**Total:** 2-3 hours

---

## Next Steps

1. Archive deprecated files
2. Fix import in `test_ensemble_v6.py`
3. Keep ELO/WLS research separate (don't merge into canonical yet)
4. Create API contract test
5. Validate all tests pass
6. Document canonical API in CLAUDE.md
