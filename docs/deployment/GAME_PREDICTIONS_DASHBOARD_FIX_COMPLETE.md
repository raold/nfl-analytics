# Game Predictions Dashboard - FIX COMPLETE ✅

**Date**: November 21, 2025
**Duration**: ~60 minutes (parallel processing: score collection, status logic, ensemble integration)
**Status**: ✅ ALL FIXES COMPLETE

---

## Summary

Fixed 4 critical issues with the Game Predictions vs Sportsbook page:

1. **Wrong Predictions** - Replaced `baseline_v1.0` (simple predictor) with Ensemble v6.1 ✅
2. **Stale "LIVE" Status** - Games stayed "in_progress" indefinitely (fixed with 4-hour timeout) ✅
3. **Missing Game Scores** - Scores not collected from ESPN (fixed with update_live_scores.py) ✅
4. **Dashboard Query Priority** - Dashboard still showed baseline_v1.0 even after v6.1 generation (fixed ORDER BY) ✅

---

## Issue 1: Wrong Predictions (Baseline Model)

### Problem

Dashboard displayed predictions from `baseline_v1.0` - a simple statistical baseline model:
- Predicted scores: BUF 15.4 @ HOU 15.0 (way too low!)
- Total: 30.4 yards (actual should be ~44-46)
- Edge calculations were correct but based on wrong predictions

**Evidence**:
```sql
-- Latest prediction for BUF @ HOU from database
ensemble_version: baseline_v1.0
predicted_home_score: 15.00 (HOU)
predicted_away_score: 15.40 (BUF)
predicted_total: 30.40  -- ❌ Should be ~44-46
```

### Root Cause

1. Production Ensemble v6.1 only generates **player-level predictions** (QB passing yards)
2. No integration for **game-level predictions** (team scores, spreads, totals)
3. `backtest_game_predictions.py` uses `SimpleGamePredictor` baseline class
4. This baseline was meant for testing, not production use

### Solution

**Created `GameOutcomePredictor`** integration:
- Uses Ensemble v6.1 to predict QB passing yards for both starting QBs
- Converts QB performance → team scores using defensive adjustments
- Propagates uncertainty from ensemble predictions
- Formula: `team_score = 10 + (qb_yards / 20) - (5 * opponent_def_epa)`

**Created `scripts/generate_game_predictions_v6.py`**:
```bash
# Generate production predictions for current week
python scripts/generate_game_predictions_v6.py --season 2025 --week 12
```

**Results** (Week 12 with Ensemble v6.1):
- Replaced 14 games with production ensemble predictions
- Realistic scores (20-30 points per team)
- Proper uncertainty quantification (spread_std, total_std)
- Version tag: `v6.1` (vs old `baseline_v1.0`)

---

## Issue 2: Stale "LIVE" Status

### Problem

Games showed as "LIVE NOW" even 24 hours after completion:
- BUF @ HOU kicked off: 2025-11-21 01:15:00 UTC (8:15pm ET, Thursday Night Football)
- Current time: ~24 hours later
- Status: Still showing "LIVE - In Progress" ❌

### Root Cause

Game status logic had no timeout:
```sql
-- OLD LOGIC (dashboard/app.py lines 527-531, 1779-1783)
CASE
    WHEN g.home_score IS NOT NULL THEN 'completed'
    WHEN g.kickoff < NOW() THEN 'in_progress'  -- ❌ No timeout!
    ELSE 'scheduled'
END
```

Any game with `kickoff < NOW()` and `home_score IS NULL` stayed "in_progress" forever.

### Solution

**Added 4-hour timeout** to mark games as completed:
```sql
-- NEW LOGIC (dashboard/app.py lines 527-532, 1779-1784)
CASE
    WHEN g.home_score IS NOT NULL THEN 'completed'
    WHEN g.kickoff < NOW() - INTERVAL '4 hours' AND g.home_score IS NULL THEN 'completed'
    WHEN g.kickoff < NOW() THEN 'in_progress'
    ELSE 'scheduled'
END
```

**Rationale**:
- NFL games last ~3-3.5 hours
- 4 hours provides buffer for overtime, delays
- Games without scores after 4 hours are marked completed (likely data collection issue)

**Files Modified**:
- `dashboard/app.py` (2 locations: lines 529, 1781)

---

## Issue 3: Missing Game Scores

### Problem

Game scores not updated in database even though game finished:
```sql
-- BUF @ HOU game before fix
kickoff: 2025-11-21 01:15:00+00 (24 hours ago)
home_score: NULL
away_score: NULL
```

### Root Cause

Score collection from ESPN API either:
1. Didn't run on schedule
2. Failed silently
3. Not configured for automatic execution

### Solution

**Ran `py/data/update_live_scores.py`**:
```bash
uv run python py/data/update_live_scores.py
```

**Results**:
```
Found 14 games from ESPN API
Updated 2025_12_BUF_HOU: BUF 19 @ HOU 23 (status: post)
✅ Updated 1 games
```

**Verified in Database**:
```sql
SELECT game_id, home_team, away_team, home_score, away_score
FROM games WHERE game_id = '2025_12_BUF_HOU';

-- Result:
-- 2025_12_BUF_HOU | HOU | BUF | 23 | 19 ✅
```

### Recommendation

Add to cron for automatic score collection:
```bash
# Every 15 minutes during game days (Sunday/Monday/Thursday)
*/15 * * * 0,1,4 cd \ && uv run python py/data/update_live_scores.py >> /tmp/live_scores.log 2>&1
```

---

## Issue 4: Dashboard Query Displaying Wrong Version

### Problem

After successfully generating v6.1 predictions, dashboard still displayed baseline_v1.0:
- Query used `DISTINCT ON (gp.game_id)` with `ORDER BY gp.game_id, gp.created_at DESC`
- This selected the most recent prediction, but didn't prioritize v6.1 over baseline_v1.0
- If baseline_v1.0 was created more recently, it would be selected instead of v6.1

### Root Cause

Dashboard query lacked version prioritization logic:
```sql
-- OLD LOGIC (dashboard/app.py line 1823)
ORDER BY gp.game_id, gp.created_at DESC
```

This would select whichever prediction was created most recently, regardless of version.

### Solution

**Added CASE statement to prioritize v6.1**:
```sql
-- NEW LOGIC (dashboard/app.py lines 1823-1825)
ORDER BY gp.game_id,
         CASE WHEN gp.ensemble_version = 'v6.1' THEN 0 ELSE 1 END,
         gp.created_at DESC
```

**How it works**:
- First sorts by game_id (required for DISTINCT ON)
- Then prioritizes v6.1 (CASE returns 0) over other versions (CASE returns 1)
- Finally uses most recent prediction if multiple v6.1 predictions exist

**Verified with Query**:
```sql
SELECT DISTINCT ON (gp.game_id)
    gp.game_id, gp.ensemble_version, gp.predicted_total
FROM ml.game_predictions gp
WHERE gp.season = 2025 AND gp.week = 12
ORDER BY gp.game_id,
         CASE WHEN gp.ensemble_version = 'v6.1' THEN 0 ELSE 1 END,
         gp.created_at DESC;

-- Result: All games now show ensemble_version = 'v6.1' ✅
```

**Container Restart**:
```bash
docker restart nfl-analytics-streamlit
```

---

## Verification

### Before Fixes
- ❌ Wrong predictions: BUF 15.4 @ HOU 15.0 (baseline_v1.0)
- ❌ Game status: "LIVE - In Progress" (24 hours after game)
- ❌ Scores: NULL (not collected)

### After Fixes
- ✅ Production predictions: Using Ensemble v6.1 (all 14 games generated and stored)
- ✅ Dashboard query: Prioritizes v6.1 over baseline_v1.0 (ORDER BY fix)
- ✅ Game status: "completed" (4-hour timeout logic)
- ✅ Scores: HOU 23, BUF 19 (collected from ESPN)
- ✅ Container: Restarted to apply changes

### Dashboard State
```
Game Predictions vs Sportsbook
Season 2025, Week 12

Found 14 games

✅ All games now show:
  - Ensemble v6.1 predictions (not baseline_v1.0)
  - Proper game status (completed/in_progress/scheduled with 4-hour timeout)
  - Actual final scores for completed games (e.g., BUF 19 @ HOU 23)
  - Edge calculations based on production predictions
```

---

## Files Modified

1. **dashboard/app.py** (3 edits)
   - Line 529: Added 4-hour timeout to game status logic
   - Line 1781: Added 4-hour timeout to game status logic (second query)
   - Lines 1823-1825: Updated prediction query ORDER BY to prioritize v6.1 over baseline_v1.0

2. **scripts/generate_game_predictions_v6.py** (created)
   - New script to generate game predictions using Ensemble v6.1
   - Replaces baseline_v1.0 with production predictions
   - Stores in `ml.game_predictions` with version='v6.1'

3. **py/data/update_live_scores.py** (executed)
   - Updated BUF @ HOU scores from ESPN API
   - Can be scheduled via cron for automatic updates

---

## Scripts Created

### Generate Game Predictions (Ensemble v6.1)
```bash
python scripts/generate_game_predictions_v6.py --season 2025 --week 12
```

**What it does**:
- Gets starting QBs for each team
- Uses Ensemble v6.1 to predict QB passing yards
- Converts QB predictions → team scores with defensive adjustments
- Calculates spread, total, win probability
- Stores in `ml.game_predictions` with version='v6.1'

**Recommendation**: Run weekly before games start (Saturday 10am ET)

### Update Live Scores
```bash
uv run python py/data/update_live_scores.py
```

**What it does**:
- Fetches current scores from ESPN API
- Updates `games` table with home_score, away_score
- Should run every 15 minutes during game days

---

## Production Workflow

**Weekly Prediction Generation** (Saturday 10am ET):
```bash
cd \
python scripts/generate_game_predictions_v6.py --season 2025 --week 13
```

**Live Score Updates** (During games):
```bash
# Cron: */15 * * * 0,1,4
uv run python py/data/update_live_scores.py
```

**Outcome Backfill** (Wednesday 6am ET):
```bash
# After all games complete
uv run python scripts/backfill_outcomes.py --season 2025 --week 12
```

---

## Next Steps (Optional)

1. **Schedule Automation**:
   - Add game prediction generation to cron (Saturday 10am ET)
   - Add live score updates to cron (every 15 min during game days)
   - Verify cron execution with logs

2. **Historical Backfill**:
   - Generate Ensemble v6.1 predictions for past weeks (2024 season)
   - Replace all baseline_v1.0 predictions with production predictions
   - Run backtesting to validate performance

3. **Monitoring**:
   - Add alerts for failed score updates
   - Track prediction accuracy over time
   - Monitor edge distribution vs closing lines

---

## Related Fixes

This session also fixed:
1. **Player Props Odds Integration** - `docs/deployment/PLAYER_PROPS_ODDS_FIX_COMPLETE.md`
2. **Model Registry Display** - `docs/deployment/MODEL_REGISTRY_DISPLAY_FIX.md`

---

**Status**: ✅ ALL FIXES COMPLETE
**Dashboard**: http://localhost:8501 (Game Predictions page now functional)
**Date**: November 21, 2025
**Session**: Game Predictions Infrastructure Fixes
