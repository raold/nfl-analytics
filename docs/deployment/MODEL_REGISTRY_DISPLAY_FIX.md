# Model Registry Display Fix ✅

**Date**: November 21, 2025
**Duration**: ~5 minutes
**Status**: ✅ FIXED

---

## Problem

Dashboard Model Registry section showed "No models found in registry. Models will appear here after training." despite having **181 models** in the database.

---

## Root Cause

**SQL Column Name Error**: Dashboard query referenced non-existent column `framework`

**File**: `dashboard/app.py`
**Lines**: 1067 (query), 1314 (display)

**Error**:
```sql
-- WRONG (column doesn't exist)
SELECT model_id, model_name, version, model_type, framework, status
FROM ml.model_registry

-- ERROR: column "framework" does not exist
```

**Correct Column**: The table uses `model_family`, not `framework`

---

## Fix

Updated `dashboard/app.py` in 2 locations:

### 1. Query Function (Line 1067)
```python
# OLD
SELECT
    model_id,
    model_name,
    version as model_version,
    model_type,
    framework,  # ❌ Column doesn't exist
    status,
    ...
FROM ml.model_registry

# NEW
SELECT
    model_id,
    model_name,
    version as model_version,
    model_type,
    model_family,  # ✅ Correct column name
    status,
    ...
FROM ml.model_registry
```

### 2. Display Columns (Line 1314)
```python
# OLD
df_models[[
    'model_id', 'model_name', 'model_version', 'model_type',
    'framework', 'status', 'updated_at', 'description'  # ❌
]]

# NEW
df_models[[
    'model_id', 'model_name', 'model_version', 'model_type',
    'model_family', 'status', 'updated_at', 'description'  # ✅
]]
```

---

## Verification

### Database Query
```bash
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -c "SELECT COUNT(*) FROM ml.model_registry;"
```

**Result**: 181 models ✅

### Sample Models
| ID | Model Name | Version | Type | Family | Status |
|----|------------|---------|------|--------|--------|
| 39 | ensemble_v6.0 | 6.0 | ensemble | bayesian_ensemble | deprecated |
| 16 | informative_priors_v2 | v2.0 | hierarchical_bayesian | bayesian_hierarchical | production |
| 13 | hierarchical_v2 | v2.0 | hierarchical_bayesian | bayesian_hierarchical | production |
| 11 | state_space_v2 | v2.0 | state_space | bayesian_state_space | production |

---

## Restart

```bash
docker restart nfl-analytics-streamlit
```

Dashboard now displays all 181 models in the registry.

---

## Schema Reference

**Table**: `ml.model_registry`
**Key Columns**:
- `model_id` (PK)
- `model_name` (unique)
- `version` (text)
- `model_type` (text)
- `model_family` (text) ← **This is the correct column**
- `status` (text)
- `is_production` (boolean)
- `created_at` / `updated_at` (timestamps)

**Note**: No `framework` column exists in the table.

---

## Related Fixes

This session also fixed:
1. **Odds Fetcher**: Player props API integration (98 props from 10 games)
2. **Docker Health Checks**: Container-aware service monitoring
3. **Latest Game Display**: Filter out future scheduled games

See: `docs/deployment/PLAYER_PROPS_ODDS_FIX_COMPLETE.md`

---

**Status**: ✅ COMPLETE
**Dashboard**: http://localhost:8501 (Model Registry now displays 181 models)
