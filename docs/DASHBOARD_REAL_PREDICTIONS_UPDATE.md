# Dashboard Real Predictions Integration

**Date:** November 14, 2025
**Status:** ✅ Code Complete, Testing Required
**Updated Files:** 1 (`dashboard/app.py`)

## Summary

Updated the Streamlit dashboard to display **REAL predictions** from the FastAPI ensemble v5.1 service instead of mock/hardcoded values. The dashboard now shows live QB passing yards predictions, total passing yards, and win probabilities for all games.

---

## Changes Made

### 1. New Helper Function: `get_game_predictions()`

**Location:** `dashboard/app.py` lines 462-531

**Purpose:** Fetch real-time predictions from FastAPI `/predict/game` endpoint

**Returns:**
```python
{
    'home_qb_yards': float,           # Predicted passing yards (home QB)
    'home_qb_yards_std': float,       # Uncertainty (std dev)
    'away_qb_yards': float,           # Predicted passing yards (away QB)
    'away_qb_yards_std': float,       # Uncertainty (std dev)
    'total_yards': float,             # Combined passing yards
    'home_win_prob': float,           # Estimated win probability (0.1-0.9)
    'inference_time_ms': float        # API processing time
}
```

**Features:**
- Configurable API URL via `FASTAPI_URL` env var (default: `http://localhost:8000`)
- 10-second timeout for API calls
- Graceful error handling (returns `None` if API fails)
- Simple win probability heuristic based on QB yards advantage

**Error Handling:**
- Network errors → logged warning, returns None
- HTTP errors → logged warning, returns None
- Parse errors → logged error, returns None

---

### 2. Updated `fetch_current_week_games()`

**Location:** `dashboard/app.py` lines 394-466

**Changes:**
- **Week Selection:** Changed from 2025 Week 10 to **2024 Week 12** (for demonstration with completed games)
- **Prediction Integration:** Calls `get_game_predictions()` for each game in the loop
- **New Columns Added:**
  - `predicted_home_qb_yards`
  - `predicted_home_qb_std`
  - `predicted_away_qb_yards`
  - `predicted_away_qb_std`
  - `predicted_total_yards`
  - `predicted_home_win_prob`

**Performance Note:** Sequential API calls for 10-15 games may take 5-15 seconds total. Consider parallelizing with `ThreadPoolExecutor` for production.

---

### 3. Updated Game Display Sections

Updated prediction display logic in **3 sections**:

#### A. Live Games (lines 1524-1554)
- **Visual:** Yellow background with orange border
- **Display:** QB yards with uncertainty bands (e.g., "KC QB: 268 ± 32 yds")
- **Format:** HTML with `<strong>` tags for emphasis

#### B. Completed Games (lines 1637-1667)
- **Visual:** Green (correct) or red (incorrect) background
- **Display:** Same format as live games
- **Status:** Shows "FINAL" with actual scores

#### C. Upcoming Games (lines 1716-1746)
- **Visual:** Blue background
- **Display:** Same format as live/completed
- **Status:** Shows kickoff time

**Common Format:**
```
Home QB: 268 ± 32 yds | Away QB: 245 ± 28 yds | Total Passing: 513 yds | Winner: KC (62.3%)
```

---

### 4. Dynamic Week Header

**Location:** `dashboard/app.py` lines 1593-1599

**Before:** Hardcoded "Week 6"
**After:** Dynamic "2024 Week 12" (or whatever week is queried)

---

## Testing Instructions

### Prerequisites

1. **FastAPI must be running** with ensemble v5.1:
   ```bash
   # Start API (from project root)
   uvicorn py.api.main:app --reload --port 8000
   ```

2. **Database must be accessible:**
   ```bash
   psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT COUNT(*) FROM games WHERE season=2024 AND week=12;"
   ```
   Should return ~15 games.

3. **Ensemble models must be loaded:**
   - State-space v2
   - Hierarchical v2
   - Hybrid XGBoost v1
   - Informative Priors v2

### Test Steps

#### 1. Start Dashboard
```bash
streamlit run dashboard/app.py
```

#### 2. Navigate to Predictions Page
- Sidebar → "Predictions" (default page)
- Should see "2024 Week 12" header

#### 3. Verify Predictions Display

**Expected Output (per game):**
```
📊 Sportsbooks: Spread: LAC -3.0 | Total: O/U 51.0
🎯 Our Predictions: LAC QB: 245 ± 28 yds | BAL QB: 268 ± 32 yds | Total Passing: 513 yds | Winner: BAL (58.3%)
```

**Look for:**
- ✅ QB predictions show actual values (not "~45" placeholder)
- ✅ Uncertainty bands (± std dev) displayed
- ✅ Total passing yards = sum of both QBs
- ✅ Win probabilities between 10-90%
- ✅ No "Fetching predictions..." or "No predictions available" messages

#### 4. Test Error Handling

**Stop FastAPI:**
```bash
# Kill uvicorn process
pkill -f "uvicorn py.api.main"
```

**Refresh Dashboard:**
- Should show: "Fetching predictions..." or "No predictions available"
- No error popups or crashes
- Dashboard remains functional

#### 5. Check Performance

**Open browser DevTools → Network tab:**
- Each game prediction: `POST /predict/game` should complete in <2 seconds
- Total page load for 15 games: <30 seconds (sequential calls)

**Optimization Needed?**
- If >30 seconds, implement parallel API calls (see Recommendations below)

---

## API Contract

### Request Format
```json
POST http://localhost:8000/predict/game
Content-Type: application/json

{
  "game_id": "2024_12_BAL_LAC",
  "season": 2024,
  "week": 12
}
```

### Response Format
```json
{
  "game_id": "2024_12_BAL_LAC",
  "season": 2024,
  "week": 12,
  "home_team": "LAC",
  "away_team": "BAL",
  "home_prediction": {
    "team": "LAC",
    "qb_id": "00-0036389",
    "qb_name": "Justin Herbert",
    "predicted_passing_yards": 245.3,
    "predicted_passing_yards_std": 28.1
  },
  "away_prediction": {
    "team": "BAL",
    "qb_id": "00-0036945",
    "qb_name": "Lamar Jackson",
    "predicted_passing_yards": 268.2,
    "predicted_passing_yards_std": 32.4
  },
  "total_predicted_yards": 513.5,
  "processing_ms": 145.3,
  "timestamp": "2025-11-14T14:50:40.353Z"
}
```

---

## Database Queries

### Games Available for Testing
```sql
-- 2024 Week 12 (Thanksgiving week)
SELECT game_id, home_team, away_team, kickoff, home_score, away_score
FROM games
WHERE season = 2024 AND week = 12
ORDER BY kickoff DESC
LIMIT 10;
```

**Expected Output:**
```
     game_id     | home_team | away_team |        kickoff         | home_score | away_score
-----------------+-----------+-----------+------------------------+------------+------------
 2024_12_BAL_LAC | LAC       | BAL       | 2024-11-26 01:11:00+00 |         23 |         30
 2024_12_TEN_HOU | HOU       | TEN       | 2024-11-25 01:11:00+00 |         27 |         32
 2024_12_ARI_SEA | SEA       | ARI       | 2024-11-24 17:56:00+00 |         16 |          6
 2024_12_DAL_WAS | WAS       | DAL       | 2024-11-24 17:56:00+00 |         26 |         34
 2024_12_DEN_LV  | LV        | DEN       | 2024-11-24 17:56:00+00 |         19 |         29
```

---

## Known Limitations

### 1. **Sequential API Calls**
- **Issue:** Fetches predictions one game at a time (blocking)
- **Impact:** Page load time scales linearly (15 games × 2s = 30s)
- **Workaround:** Use smaller week (fewer games) for faster testing

### 2. **No Caching of Predictions**
- **Issue:** Every dashboard refresh makes fresh API calls
- **Impact:** Increased API load, slower UX
- **Workaround:** Redis cache implemented in API (1-hour TTL)

### 3. **Simplified Win Probability**
- **Issue:** Heuristic based only on QB yards (not full game model)
- **Formula:** `win_prob = 0.5 + (yards_diff / 500)`
- **Impact:** Less accurate than proper game outcome model
- **Workaround:** Acceptable for demonstration; improve in Phase 2

### 4. **Hardcoded Week (2024 Week 12)**
- **Issue:** Not dynamically fetching current week
- **Impact:** Shows historical games (not live NFL week)
- **Workaround:** Change query in `fetch_current_week_games()` to:
  ```sql
  WHERE g.season = 2025 AND g.week = CURRENT_WEEK
  ```

---

## Recommendations

### Phase 1: Performance Optimization

**Parallelize API Calls:**
```python
from concurrent.futures import ThreadPoolExecutor

def fetch_predictions_parallel(games_df):
    """Fetch predictions for all games in parallel."""
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(
                get_game_predictions,
                row['game_id'],
                row['season'],
                row['week']
            ): idx
            for idx, row in games_df.iterrows()
        }

        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            predictions = future.result()
            if predictions:
                games_df.at[idx, 'predicted_home_qb_yards'] = predictions.get('home_qb_yards')
                # ... (set other columns)

    return games_df
```

**Expected Speedup:**
- Sequential: 15 games × 2s = 30s
- Parallel (5 workers): 15 games / 5 × 2s = 6s
- **5x faster!**

### Phase 2: Enhanced Predictions

1. **Add Over/Under Recommendations:**
   ```python
   if total_yards > total_close:
       recommendation = "OVER"
   else:
       recommendation = "UNDER"
   ```

2. **Confidence Intervals:**
   - Show 90% CI for QB yards: `[210-327 yds]`
   - Color-code by confidence (green = high, yellow = medium, red = low)

3. **Model Breakdown:**
   - Tooltip showing individual model predictions
   - Ensemble weights used

### Phase 3: Real-Time Updates

1. **Auto-Refresh During Live Games:**
   ```python
   if game_status == 'in_progress':
       st.rerun()  # Refresh every 30 seconds
   ```

2. **Live Prediction Updates:**
   - Re-fetch predictions as game progresses
   - Show prediction accuracy in real-time

---

## Files Modified

| File | Lines Changed | Description |
|------|---------------|-------------|
| `dashboard/app.py` | +130 | Added `get_game_predictions()`, updated prediction display logic, dynamic week header |

**Total:** 1 file, 130 lines added/modified

---

## Next Steps

1. **Restart FastAPI** with `/predict/game` endpoint:
   ```bash
   uvicorn py.api.main:app --reload --port 8000
   ```

2. **Test Dashboard:**
   ```bash
   streamlit run dashboard/app.py
   ```

3. **Verify Predictions Load:**
   - Navigate to Predictions page
   - Check 2024 Week 12 games show real QB yards
   - Confirm no errors in browser console

4. **Performance Test:**
   - Measure page load time (should be <30s for 15 games)
   - Consider implementing parallel API calls if >30s

5. **Deploy to Production:**
   - Switch to current week (2025 Week 11+)
   - Enable auto-refresh for live games
   - Monitor API performance under load

---

## Questions & Issues

### Q: Why 2024 Week 12 instead of current week?
**A:** Current date is Nov 14, 2025. The 2025 season games are scheduled but haven't been played yet (no scores). Week 12 2024 has completed games with actual outcomes, allowing us to validate predictions against reality.

### Q: What if FastAPI is down?
**A:** Dashboard gracefully degrades:
- Shows "Fetching predictions..." message
- No crashes or errors
- Rest of dashboard remains functional

### Q: How to add more stats (rushing, receiving)?
**A:** Extend `get_game_predictions()` to call `/predict` endpoint for individual players:
```python
rush_pred = requests.post(
    f"{api_url}/predict",
    json={
        "player_id": "00-0037077",  # RB
        "stat_type": "rushing_yards",
        "week": week,
        "season": season
    }
)
```

---

**Author:** Claude Code (Sonnet 4.5)
**Review Required:** Yes (API integration testing)
**Production Ready:** No (needs performance optimization for >15 games)
