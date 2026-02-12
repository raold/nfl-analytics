# Dashboard Placeholder Values - Root Cause Analysis

**Date:** November 23, 2025
**Issue:** Dashboard showing 240 ± 50 yard predictions for ALL games
**Status:** ❌ CRITICAL - Placeholder fallback values instead of real ensemble predictions

---

## Root Cause

The `/api/v1/predict/game` endpoint (`py/api/routes/game_predictions.py:1190-1295`) is **completely disconnected from the ensemble models** and uses placeholder fallback values.

### Code Location

**File:** `py/api/routes/game_predictions.py`
**Lines:** 1227-1239

```python
cursor.execute(qb_query, (request.game_id,))
qbs = cursor.fetchall()

# If game hasn't happened yet, use season averages for team's likely starter
if not qbs:
    logger.warning(f"No QB data for {request.game_id}, using season averages")

    # Fallback to season averages (realistic estimates based on 2024 averages)
    # This provides reasonable predictions even for future games
    home_yards = 240.0  # Median QB passing yards  ← HARDCODED PLACEHOLDER
    away_yards = 240.0
    home_uncertainty = 50.0  # Typical std dev  ← HARDCODED PLACEHOLDER
    away_uncertainty = 50.0
```

### Why It Fails

1. **Line 1219:** Query looks for plays from THIS SPECIFIC game:
   ```sql
   WHERE game_id = %s AND pass_attempt = TRUE
   ```

2. **For Future Games:** 2025 Week 18 games haven't been played yet, so:
   - `plays` table has NO records for game_id `2025_18_DAL_NYG`
   - `qbs` variable is empty
   - Code hits `if not qbs:` fallback (line 1231)
   - Returns hardcoded `240.0 ± 50.0` for BOTH teams

3. **TODO Comment (Line 1243):**
   ```python
   # TODO: Integrate with EnhancedEnsemble5Way for full prediction
   ```
   The integration was never implemented.

---

## Evidence

### API Test - 2025 Week 18 DAL@NYG
```bash
curl -X POST 'http://api:8000/api/v1/predict/game' \
  -H 'Content-Type: application/json' \
  -d '{"game_id": "2025_18_DAL_NYG", "season": 2025, "week": 18}'
```

**Response:**
```json
{
  "game_id": "2025_18_DAL_NYG",
  "home_prediction": {
    "predicted_passing_yards": 240.0,  ← PLACEHOLDER
    "predicted_passing_yards_std": 50.0  ← PLACEHOLDER
  },
  "away_prediction": {
    "predicted_passing_yards": 240.0,  ← PLACEHOLDER
    "predicted_passing_yards_std": 50.0  ← PLACEHOLDER
  },
  "total_predicted_yards": 480.0,
  "processing_ms": 24.24
}
```

### Working Endpoint - `/api/v1/predict/game/full`
```bash
curl -X POST 'http://localhost:8000/api/v1/predict/game/full' \
  -H 'Content-Type: application/json' \
  -d '{"game_id": "2025_18_DAL_NYG", "season": 2025, "week": 18}'
```

**Response:**
```json
{
  "game_id": "2025_18_DAL_NYG",
  "predicted_home_score": 21.8,  ← REAL ensemble v6.1 prediction
  "predicted_away_score": 21.7,  ← REAL ensemble v6.1 prediction
  "predicted_total": 43.6,
  "predicted_spread": 0.1,
  "ensemble_version": "v6.1",
  "processing_ms": 236.87
}
```

**Key Difference:**
- `/predict/game/full` calls ensemble v6.1 → Returns REAL predictions (21.8, 21.7 scores)
- `/predict/game` queries historical plays → Returns PLACEHOLDERS (240 ± 50 yards)

---

## Why Dashboard Shows Placeholders

1. **dashboard/app.py:487** calls `get_game_predictions(row['game_id'], season, week)`
2. **dashboard/app.py:532** sends POST to `/api/v1/predict/game`
3. API returns placeholder values (240 ± 50)
4. **dashboard/app.py:1887-1899** displays the placeholders on dashboard cards

The dashboard code is **correct** - it's fetching and displaying API responses. The API is **wrong** - it's returning placeholders instead of ensemble predictions.

---

## Solution

### Option 1: Fix `/api/v1/predict/game` to call ensemble
Modify `py/api/routes/game_predictions.py` to:
1. Import `EnhancedEnsemble6Production`
2. Call ensemble's `predict_player_props()` method for QB passing yards
3. Remove placeholder fallback (lines 1236-1239)
4. Use ensemble uncertainty estimates instead of hardcoded 50.0

### Option 2: Change dashboard to call `/api/v1/predict/game/full`
Modify `dashboard/app.py` to:
1. Call `/api/v1/predict/game/full` instead of `/api/v1/predict/game`
2. Calculate QB passing yards from home/away scores (rough approximation)
3. This is NOT ideal - we lose player-level granularity

### Recommended: Option 1
Fix the API endpoint to actually call the ensemble. This preserves:
- Player-level QB predictions (not just team scores)
- Full Bayesian uncertainty quantification
- Consistent ensemble versioning (v6.1)

---

## Next Steps

1. ✅ Root cause identified
2. ⏳ Implement Option 1 fix
3. ⏳ Test with 2025 Week 18 games
4. ⏳ Verify ensemble predictions load in dashboard
5. ⏳ Git commit with fix
6. ⏳ Verify in browser

---

## Files Involved

- **Broken Endpoint:** `py/api/routes/game_predictions.py:1190-1295`
- **Working Endpoint:** `py/api/routes/game_predictions.py:predict_game_full()`
- **Dashboard Caller:** `dashboard/app.py:504-614`
- **Dashboard Display:** `dashboard/app.py:1887-1912`

---

**Impact:** ❌ CRITICAL - Dashboard shows placeholder values for ALL future games, completely defeating the purpose of the ensemble models.

**Effort:** ~1-2 hours to implement Option 1 (import ensemble, modify endpoint, test)

**Priority:** P0 - This is the core functionality of the entire project
