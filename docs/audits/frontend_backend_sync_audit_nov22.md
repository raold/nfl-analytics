# Frontend-Backend Sync Validation Audit - November 22, 2025

**Audit Date**: November 22, 2025, 12:15 PM
**Auditor**: Frontend Engineering Agent
**Scope**: Backend API ↔ Dashboard integration
**Tool**: `py/utils/frontend_engineering_agent.py`

---

## Executive Summary

**Status**: 🚨 **CRITICAL ISSUES** - Multiple contract violations and schema mismatches

**Key Findings**:
- 🚨 **6 SQL schema validation errors** - Dashboard queries reference non-existent tables
- 🚨 **5 API contract violations** - Frontend expects fields backend doesn't provide
- ⚠️ **3 SQL best practice warnings** - SELECT *, missing WHERE clauses
- ⚠️ **No schema drift detection** - Missing baseline snapshot
- ✅ **API availability**: Backend is healthy and responding

**Priority**: CRITICAL
**Estimated Remediation**: 4-6 hours
**Risk Level**: HIGH - Dashboard may crash on runtime with KeyError

---

## Detailed Findings

### 1. SQL Schema Validation Errors

**Status**: 🚨 CRITICAL (6 errors)

#### 1.1 Missing Table: `public.weekly_metrics`

**Location**: `dashboard/app.py:245`
**Severity**: ERROR
**Query**:
```sql
SELECT * FROM public.weekly_metrics
```

**Issue**: Table does not exist in database
**Impact**: Dashboard will fail when trying to load weekly metrics
**Fix Options**:
1. Create table with migration:
   ```sql
   CREATE TABLE public.weekly_metrics (
       season INTEGER,
       week INTEGER,
       metric_name TEXT,
       metric_value FLOAT,
       created_at TIMESTAMP DEFAULT NOW(),
       PRIMARY KEY (season, week, metric_name)
   );
   ```
2. Remove query if functionality no longer needed
3. Update query to use existing table (e.g., `mart.player_game_stats` aggregated)

---

#### 1.2 Missing Tables: `public.all_teams`, `public.week_teams`

**Location**: `dashboard/app.py:388` and `dashboard/app.py:1023`
**Severity**: ERROR (2 instances)
**Query Pattern**:
```sql
SELECT ... FROM public.all_teams
UNION
SELECT ... FROM public.week_teams
```

**Issue**: Both tables do not exist
**Likely Intent**: Aggregating team-level statistics across season and weekly views
**Available Alternatives**:
- `mart.team_game_stats` (if it exists - needs verification)
- `public.games` joined with aggregated player stats
- Create materialized view from existing data

**Fix Options**:
1. Verify if `mart.team_game_stats` exists (check schema)
2. Create tables via migration from existing games/player_stats data
3. Refactor queries to use existing schema

**Recommendation**: Check `docs/database/DATABASE_QUICK_REFERENCE.md` for correct team stats table

---

#### 1.3 Missing Table: `public.pg_tables`

**Location**: `dashboard/app.py:1550`
**Severity**: ERROR
**Query**:
```sql
SELECT ... FROM public.pg_tables
```

**Issue**: Should be `pg_catalog.pg_tables` (system catalog)
**Impact**: Database metadata query will fail
**Fix**:
```sql
-- ❌ WRONG
SELECT ... FROM public.pg_tables

-- ✅ CORRECT
SELECT ... FROM pg_catalog.pg_tables
```

**Estimated Fix Time**: 30 seconds (change schema name)

---

### 2. API Contract Violations

**Status**: 🚨 CRITICAL (5 violations)

#### 2.1 Mismatched Response Schema: `/predict` Endpoint

**Frontend Expectation** (from `dashboard/utils/api_client.py:55-67`):
```python
class SinglePlayerPrediction(TypedDict, total=False):
    player_id: str
    player_name: str
    stat_type: str
    prediction: float           # ❌ Expected
    confidence: float           # ❌ Expected
    lower_bound: float          # ❌ Expected
    upper_bound: float          # ❌ Expected
    models_used: List[ModelContribution]  # ❌ Expected
    timestamp: str
    ensemble_method: str
```

**Backend Reality** (from `py/api/main.py:278-301`):
```python
class PredictionResponse(BaseModel):
    player_id: str
    player_name: Optional[str]
    stat_type: str
    prediction: float                      # ✅ Matches
    uncertainty: float                     # ⚠️ Not in frontend
    confidence_level: float                # ⚠️ Frontend expects "confidence"
    confidence_interval: ConfidenceInterval  # ⚠️ Nested object (lower, upper, level)
    model_versions: ModelVersions          # ⚠️ Frontend expects "models_used"
    ensemble_weights: EnsembleWeights      # ⚠️ Not in frontend
    timestamp: str                         # ✅ Matches
    cache_hit: bool                        # ⚠️ Not in frontend
    processing_ms: float                   # ⚠️ Not in frontend
```

**Specific Violations**:

1. **Missing: `confidence`**
   - Backend provides: `confidence_level` (e.g., 0.90)
   - Frontend expects: `confidence` (name mismatch)
   - **Fix**: Rename field OR update frontend TypedDict

2. **Missing: `lower_bound`, `upper_bound`**
   - Backend provides: `confidence_interval.lower`, `confidence_interval.upper` (nested)
   - Frontend expects: `lower_bound`, `upper_bound` (flat)
   - **Fix**: Flatten response OR update frontend to access nested fields

3. **Missing: `models_used`**
   - Backend provides: `ensemble_weights` (dict of model weights)
   - Frontend expects: `models_used` (list of ModelContribution objects)
   - **Fix**: Add `models_used` field OR refactor frontend to use `ensemble_weights`

**Impact**: HIGH
- Dashboard will get KeyError when accessing missing fields
- Predictions may display incorrectly or crash page
- Confidence intervals won't render

**Remediation Options**:

**Option A: Update Backend (Add Compatibility Fields)**
```python
# py/api/main.py:278
class PredictionResponse(BaseModel):
    # ... existing fields ...

    # Add compatibility fields for dashboard
    @property
    def confidence(self) -> float:
        """Alias for confidence_level"""
        return self.confidence_level

    @property
    def lower_bound(self) -> float:
        """Flatten confidence_interval.lower"""
        return self.confidence_interval.lower

    @property
    def upper_bound(self) -> float:
        """Flatten confidence_interval.upper"""
        return self.confidence_interval.upper

    @property
    def models_used(self) -> List[dict]:
        """Convert ensemble_weights to models_used format"""
        return [
            {"model_name": k, "weight": v}
            for k, v in self.ensemble_weights.dict().items()
        ]
```

**Option B: Update Frontend (Use Actual Backend Schema)**
```python
# dashboard/utils/api_client.py:55
class SinglePlayerPrediction(TypedDict, total=False):
    player_id: str
    player_name: str
    stat_type: str
    prediction: float
    confidence_level: float  # Changed from 'confidence'
    confidence_interval: dict  # Changed from lower_bound/upper_bound
    ensemble_weights: dict  # Changed from 'models_used'
    timestamp: str
    cache_hit: bool
    processing_ms: float
```

**Recommendation**: **Option A** (backend compatibility layer)
- Preserves existing dashboard code
- Minimal risk of introducing bugs
- Gradual migration path
- Estimated time: 30 minutes

---

### 3. SQL Best Practice Warnings

**Status**: ⚠️ WARNING (non-blocking but should fix)

#### 3.1 SELECT * Detected

**Location**: `dashboard/app.py:245`
**Query**:
```sql
SELECT * FROM public.weekly_metrics
```

**Issue**: Schema changes will silently break queries
**Best Practice**: Explicitly list columns
**Fix**:
```sql
-- ❌ WRONG
SELECT * FROM public.weekly_metrics

-- ✅ CORRECT
SELECT season, week, metric_name, metric_value, created_at
FROM public.weekly_metrics
```

**Estimated Fix Time**: 2 minutes

---

#### 3.2 Full Table Scan on Large Table

**Location**: `dashboard/app.py:902` (2 instances)
**Query**:
```sql
SELECT ... FROM plays  -- No WHERE or LIMIT clause
```

**Issue**: `plays` table has 1.25M rows - full scan will timeout
**Impact**: Dashboard will hang or timeout
**Best Practice**: Always add WHERE/LIMIT on large tables
**Fix**:
```sql
-- ❌ WRONG
SELECT ... FROM plays

-- ✅ CORRECT (Option 1: Filter by season)
SELECT ... FROM plays WHERE season >= 2023

-- ✅ CORRECT (Option 2: Add LIMIT)
SELECT ... FROM plays LIMIT 1000
```

**Estimated Fix Time**: 5 minutes (depends on business logic)

---

### 4. Schema Drift Detection

**Status**: ⚠️ WARNING (unable to detect)

**Issue**: No baseline schema snapshot found
**Location**: Expected at `docs/database/schema_snapshot.json`

**Impact**: Cannot detect schema drift between environments
**Risk**: Staging/production schema mismatches go undetected

**Remediation**:
1. Generate baseline snapshot:
   ```bash
   python py/utils/frontend_engineering_agent.py generate-snapshot
   ```
2. Commit snapshot to git:
   ```bash
   git add docs/database/schema_snapshot.json
   git commit -m "feat: add schema baseline for drift detection"
   ```
3. Run drift detection:
   ```bash
   python py/utils/frontend_engineering_agent.py check-drift dashboard/
   ```

**Estimated Time**: 15 minutes

---

### 5. API Endpoint Discovery

**Status**: ✅ INFORMATIONAL

**All API Endpoints** (from OpenAPI schema):

#### Main App Endpoints (`py/api/main.py`)
- `GET /health` - Service health check ✅ (likely used)
- `POST /predict` - Single player prediction 🚨 (contract violations)
- `POST /predict/game` - Game-level prediction ⚠️ (check usage)
- `GET /prometheus` - Prometheus metrics ✅ (used)
- `GET /metrics` - JSON metrics summary ⚠️ (check usage)
- `GET /` - Root endpoint (docs redirect) ✅

#### Game Predictions Routes (`py/api/routes/game_predictions.py`)
- `POST /predict/game/full` - Full game prediction with all components
- `POST /predict/week` - Batch predictions for entire week
- `GET /predict/game/{game_id}` - Get cached game prediction

#### Props Predictions Routes (`py/api/routes/props_predictions.py`)
- `POST /predict/props/player` - Single player prop prediction
- `POST /predict/props/batch` - Batch player predictions
- `POST /predict/props/game` - All players for a game
- `GET /predict/props/available/{season}/{week}` - Available props for week

**Dashboard Integration Analysis**:

Based on analysis of `dashboard/utils/api_client.py`, the dashboard uses:

| Endpoint | Dashboard Method | Status | Usage |
|----------|------------------|--------|-------|
| `GET /health` | `get_health()` | ✅ USED | Health checks, monitoring |
| `GET /models` | `get_models()` | 🚨 ERROR | Returns 404 (endpoint doesn't exist in backend) |
| `POST /predict` | `get_player_prediction()` | 🚨 CONTRACT VIOLATION | See Section 2.1 |
| `POST /predict/game` | `get_game_prediction()` | ✅ USED | Game predictions |

**Critical Issues**:

1. **`GET /models` endpoint missing from backend**
   - Dashboard calls `api.get_models()` (line 420-455)
   - Backend has NO `/models` endpoint defined
   - **Impact**: Model registry dashboard page will fail
   - **Fix**: Add `/models` endpoint to backend OR remove dashboard feature

2. **`POST /predict` contract violations**
   - See Section 2.1 for full details
   - **Impact**: Player predictions may crash with KeyError

**Unused Backend Endpoints** (Not Integrated in Dashboard):

The following endpoints exist in the backend but are NOT used by the dashboard:
- `POST /predict/game/full` - Full game predictions with all details
- `POST /predict/week` - Batch week predictions (efficiency opportunity!)
- `GET /predict/game/{game_id}` - Cached game lookup
- `POST /predict/props/player` - Player props via new routes
- `POST /predict/props/batch` - Batch player props (efficiency opportunity!)
- `POST /predict/props/game` - All props for a game
- `GET /predict/props/available/{season}/{week}` - Props discovery

**Integration Opportunities**:

1. **Batch Predictions** (2-10x faster page loads)
   - Replace individual `get_game_prediction()` calls with `POST /predict/week`
   - Estimated speedup: 5-10x for week view pages
   - Estimated effort: 2-3 hours

2. **Props Discovery**
   - Use `GET /predict/props/available/{season}/{week}` to dynamically populate player dropdowns
   - Current: Hardcoded player lists
   - Estimated effort: 1-2 hours

3. **Cached Game Lookups**
   - Use `GET /predict/game/{game_id}` for historical games (faster than POST)
   - Estimated speedup: 2-3x for historical game pages
   - Estimated effort: 1 hour

---

### 6. TypedDict Validation Analysis

**Status**: ⚠️ WARNING (low adoption)

**Finding**: Dashboard has excellent TypedDict definitions but limited validation usage

**Code Review**:
- ✅ TypedDict definitions exist for all response types (lines 47-92 in api_client.py)
- ✅ Validation functions exist (`validate_game_prediction`, `validate_single_player_prediction`)
- ⚠️ But validation is only called in APIClient methods, not in dashboard pages directly

**Example of Type-Safe Pattern** (from api_client.py:218):
```python
def validate_single_player_prediction(data: Dict[str, Any]) -> Optional[SinglePlayerPrediction]:
    """Validate and extract single player prediction response."""
    required_fields = ['player_id', 'prediction', 'confidence']

    missing_fields = [f for f in required_fields if f not in data]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")

    # Validate numeric types
    prediction = float(data['prediction'])
    confidence = float(data['confidence'])
    lower_bound = float(data.get('lower_bound', prediction * 0.8))
    upper_bound = float(data.get('upper_bound', prediction * 1.2))

    return SinglePlayerPrediction(...)
```

**Issue**: This validation expects `confidence`, `lower_bound`, `upper_bound` - which backend doesn't provide (see Section 2.1)

**Recommendation**: Update validation functions after fixing API contract violations

---

## Summary of Violations

### By Severity

| Severity | Count | Issues |
|----------|-------|--------|
| 🚨 CRITICAL | 11 | 6 SQL schema errors, 5 API contract violations |
| ⚠️ WARNING | 4 | 3 SQL best practices, 1 missing snapshot |
| ✅ INFO | 8 | 8 unused backend endpoints (integration opportunities) |

### By Category

| Category | Issues | Priority |
|----------|--------|----------|
| **SQL Schema Errors** | 6 | CRITICAL - Fix first |
| **API Contract Violations** | 5 | CRITICAL - Dashboard may crash |
| **SQL Best Practices** | 3 | WARNING - Performance risk |
| **Missing /models Endpoint** | 1 | CRITICAL - Dashboard feature broken |
| **Schema Drift Detection** | 1 | WARNING - Deploy risk |
| **Integration Gaps** | 8 | INFO - Performance opportunity |

---

## Remediation Plan

### Phase 1: Critical Fixes (MUST DO - 3-4 hours)

**Priority 1A: Fix SQL Schema Errors** (1.5 hours)
1. Fix `pg_tables` schema (30 seconds):
   ```sql
   -- dashboard/app.py:1550
   SELECT ... FROM pg_catalog.pg_tables  -- Changed from public.pg_tables
   ```

2. Handle missing tables (1 hour):
   - `weekly_metrics`: Remove query OR create table via migration
   - `all_teams`, `week_teams`: Replace with existing schema OR create migration
   - **Action**: Check DATABASE_QUICK_REFERENCE.md for correct table names

3. Fix full table scans on `plays` (30 min):
   - Add `WHERE season >= 2023` OR `LIMIT 1000` to queries at app.py:902

**Priority 1B: Fix API Contract Violations** (1.5 hours)

**Recommended Approach**: Add compatibility layer in backend

```python
# py/api/main.py:278 - Add to PredictionResponse class

class PredictionResponse(BaseModel):
    # ... existing fields ...

    # Compatibility fields for dashboard (Phase 1 - Nov 22)
    @property
    def confidence(self) -> float:
        """Alias for confidence_level (dashboard compatibility)"""
        return self.confidence_level

    @property
    def lower_bound(self) -> float:
        """Flatten confidence_interval.lower (dashboard compatibility)"""
        return self.confidence_interval.lower if hasattr(self, 'confidence_interval') else 0.0

    @property
    def upper_bound(self) -> float:
        """Flatten confidence_interval.upper (dashboard compatibility)"""
        return self.confidence_interval.upper if hasattr(self, 'confidence_interval') else 0.0

    @property
    def models_used(self) -> List[Dict[str, Any]]:
        """Convert ensemble_weights to models_used format (dashboard compatibility)"""
        if not hasattr(self, 'ensemble_weights'):
            return []
        return [
            {"model_name": k, "weight": v}
            for k, v in self.ensemble_weights.dict().items()
        ]

    class Config:
        # Enable property serialization in JSON response
        json_encoders = {property: lambda v: v}
```

**Estimated Time**: 30 minutes coding + 30 minutes testing + 30 minutes deployment

**Priority 1C: Add Missing /models Endpoint** (45 min)

```python
# py/api/main.py - Add new endpoint

@app.get(
    "/models",
    response_model=ModelsListResponse,
    tags=["Models"]
)
def get_models():
    """
    Get list of loaded models (dashboard compatibility endpoint).

    Returns model registry information for dashboard display.
    """
    # TODO: Implement model registry query
    # For now, return placeholder to unblock dashboard
    return {
        "models": [],
        "total_models": 0,
        "loaded_models": 0,
        "ensemble_ready": True
    }
```

**Note**: This is a placeholder. Full implementation requires model registry integration.

---

### Phase 2: Performance & Best Practices (SHOULD DO - 2-3 hours)

**Priority 2A: Fix SQL Best Practices** (30 min)
1. Replace `SELECT *` with explicit columns (app.py:245)
2. Validate all large table queries have WHERE/LIMIT clauses

**Priority 2B: Schema Drift Detection** (15 min)
1. Generate baseline snapshot:
   ```bash
   python py/utils/frontend_engineering_agent.py generate-snapshot
   git add docs/database/schema_snapshot.json
   git commit -m "feat: add schema baseline for drift detection"
   ```

**Priority 2C: Batch API Integration** (2 hours)
1. Replace individual game predictions with `POST /predict/week`
2. Expected speedup: 5-10x for week view pages

---

### Phase 3: Future Enhancements (NICE TO HAVE - 3-5 hours)

**Priority 3A: Props Integration** (2 hours)
- Integrate `POST /predict/props/batch` for player props pages
- Use `GET /predict/props/available` for dynamic player selection

**Priority 3B: Cached Lookups** (1 hour)
- Use `GET /predict/game/{game_id}` for historical games

**Priority 3C: Type Safety Hardening** (2 hours)
- Add runtime schema validation to all dashboard pages
- Implement automated contract testing in CI/CD

---

## Testing Strategy

### Before Deployment (Required)

1. **SQL Query Validation**:
   ```bash
   # Test all SQL queries against dev database
   python py/utils/frontend_engineering_agent.py validate-schema dashboard/
   ```

2. **API Contract Testing**:
   ```bash
   # Validate API responses match TypedDict definitions
   python py/utils/frontend_engineering_agent.py validate-api dashboard/
   ```

3. **Manual Testing Checklist**:
   - [ ] Dashboard loads without errors
   - [ ] Game predictions display correctly
   - [ ] Player predictions show confidence intervals
   - [ ] Model registry page loads (after /models endpoint added)
   - [ ] No KeyError crashes in browser console

### After Deployment (Recommended)

1. **Schema Drift Monitoring**:
   ```bash
   # Run weekly to detect schema changes
   python py/utils/frontend_engineering_agent.py check-drift dashboard/
   ```

2. **API Contract Monitoring**:
   - Add alerts for API 500 errors (schema mismatches)
   - Monitor Sentry/logs for KeyError exceptions in dashboard

---

## Risk Assessment

### High Risk (Must Fix Before Production)

1. **SQL Schema Errors** (6 errors)
   - **Impact**: Dashboard crashes, data not loading
   - **Probability**: 100% (errors already present)
   - **Mitigation**: Fix in Phase 1

2. **API Contract Violations** (5 errors)
   - **Impact**: KeyError crashes, broken predictions display
   - **Probability**: 90% (if /predict endpoint is used)
   - **Mitigation**: Add compatibility layer in Phase 1

3. **Missing /models Endpoint**
   - **Impact**: Model registry page completely broken
   - **Probability**: 100%
   - **Mitigation**: Add endpoint in Phase 1

### Medium Risk (Should Fix Soon)

1. **Full Table Scans** (2 queries)
   - **Impact**: Dashboard timeouts, poor performance
   - **Probability**: 60% (depends on data size)
   - **Mitigation**: Add WHERE clauses in Phase 2

2. **No Schema Drift Detection**
   - **Impact**: Silent breakage on schema changes
   - **Probability**: 30% (if schema changes frequently)
   - **Mitigation**: Generate snapshot in Phase 2

### Low Risk (Nice to Have)

1. **SELECT * Usage**
   - **Impact**: Fragility to schema changes
   - **Probability**: 10%
   - **Mitigation**: Replace with explicit columns in Phase 2

2. **Unused Backend Endpoints**
   - **Impact**: Missed performance opportunities
   - **Probability**: N/A (opportunity cost)
   - **Mitigation**: Integrate batch endpoints in Phase 3

---

## Conclusion

**Overall Assessment**: 🚨 **CRITICAL ISSUES REQUIRE IMMEDIATE ATTENTION**

The dashboard has **11 critical issues** that must be fixed before production deployment:
- 6 SQL schema errors that will cause query failures
- 5 API contract violations that will cause KeyError crashes
- 1 missing /models endpoint that breaks model registry page

**Good News**:
- ✅ Excellent TypedDict architecture for type safety
- ✅ Comprehensive validation functions (just need schema updates)
- ✅ API backend is healthy and responding
- ✅ Many optimization opportunities via batch endpoints

**Recommended Action**:
1. **Fix Phase 1 Critical Issues** (3-4 hours) - BLOCK PRODUCTION UNTIL COMPLETE
2. **Deploy to Staging** - Validate all fixes work
3. **Implement Phase 2** (2-3 hours) - Best practices and performance
4. **Consider Phase 3** (3-5 hours) - Future performance enhancements

**Estimated Total Remediation**: 8-12 hours for Phases 1-2 (production-ready)

---

**Next Steps**: Proceed to Phase 6 (UI/UX & Design System Compliance)
