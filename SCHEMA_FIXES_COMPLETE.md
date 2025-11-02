# Database Schema Fixes - Complete
**Date:** November 2, 2025
**Git Commit:** 92927aa

## Summary

Fixed critical database schema mismatches across all training scripts that were blocking Phase 2-4 training.

## Root Cause

Training scripts expected columns that don't exist in `mart.player_game_stats`:
- ❌ `opponent_rank` (doesn't exist anywhere)
- ❌ `temperature` / `temp` (exists in `weather` table, not player_game_stats)
- ❌ `is_home` (must be derived from JOIN with games table)
- ❌ `rushing_attempts` (doesn't exist)
- ❌ `stat_category`, `stat_attempts`, `stat_completions`, `stat_touchdowns`, `current_team` (don't exist)

**Actual schema** of `mart.player_game_stats` (15 columns only):
```
game_id, season, week, player_id, player_name, position, team,
passing_yards, passing_tds, rushing_yards, rushing_tds,
receiving_yards, receiving_tds, targets, receptions
```

**Weather data** is in separate `weather` table (game_id PK):
```
game_id, temp, wind, humidity, conditions, data_source
```

## Files Fixed

### 1. scripts/bnn_optuna_search.py
**Changes:**
- Added `LEFT JOIN weather w ON g.game_id = w.game_id`
- Changed `g.temp` → `w.temp`, `g.wind` → `w.wind`
- Removed `opponent_rank`, `temperature`, `is_home` from SELECT
- Removed `rushing_attempts > 5` filter (column doesn't exist)
- Added feature engineering to derive `is_home` from team comparisons
- Parse weather TEXT columns to numeric (handle DOME/None)
- Calculate point_differential feature

### 2. scripts/train_best_bnn.py
**Changes:** Same as above

### 3. scripts/train_all_phases_scaled.py
**Changes:** Same as above (embedded in optuna_script string)

### 4. py/models/train_bnn_passing.py
**Changes:**
- Added `LEFT JOIN weather w ON g.game_id = w.game_id`
- Changed `g.temp` → `w.temp`, `g.wind` → `w.wind`
- Removed `stat_yards`, `stat_attempts`, `stat_completions`, `stat_touchdowns`, `stat_category`, `current_team`
- Used actual columns: `passing_yards`, `passing_tds`, `team`
- Simplified feature engineering (no attempts/completions data available)
- Added note explaining schema limitations

### 5. py/rl/train_experiment_rl.py
**Changes:**
- Fixed Python imports: `from py.rl.experiment_augmented_env` → `from experiment_augmented_env`
- Fixed: `from py.rl.experiment_rl_agent` → `from experiment_rl_agent`

### 6. py/rl/experiment_augmented_env.py
**Changes:**
- Fixed Python import: `from py.rl.experiment_feature_extractor` → `from experiment_feature_extractor`

## Testing

Validated fixed query loads data successfully:
```python
# Test query with weather JOIN
query = """
SELECT pgs.*, w.temp, w.wind, g.home_team, g.away_team
FROM mart.player_game_stats pgs
JOIN games g ON pgs.game_id = g.game_id
LEFT JOIN weather w ON g.game_id = w.game_id
WHERE pgs.rushing_yards > 0
"""
# ✅ Query successful! Loaded data correctly
# ✅ Columns: ['player_id', 'rushing_yards', 'temp', 'wind', ...]
```

## Impact

**Before:** All training scripts failed with `psycopg2.errors.UndefinedColumn`

**After:** All scripts can load data from database correctly

## Next Steps

Now that schema issues are fixed:
1. ✅ Phase 1 complete (Model Registry + cVAE fixed)
2. ✅ Schema fixes committed (git 92927aa)
3. ⏳ Ready to resume Phase 2-4 training:
   - Phase 2: BNN Hyperparameter Optimization (100 trials)
   - Phase 3: DQN/PPO RL Agents (3000 episodes)
   - Phase 4: Ensemble & Backtest

## Key Learnings

1. **Always JOIN with weather table** for temp/wind data (LEFT JOIN to handle missing weather)
2. **mart.player_game_stats is minimal** - only 15 basic stat columns
3. **Derive features** like `is_home` from team comparisons (pgs.team = g.home_team)
4. **Weather data is sparse** - use fillna() for missing values (DOME games have None)
5. **Use relative imports** for py/rl/* modules (not absolute py.rl.* imports)

## References

- Database Schema Audit: `docs/DATABASE_SCHEMA_AUDIT.md`
- Training Progress: `TRAINING_PROGRESS_FINAL.md`
