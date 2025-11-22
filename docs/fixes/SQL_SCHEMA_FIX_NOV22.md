# SQL Schema Fix - November 22, 2025

**Status:** ✅ Complete
**Duration:** 15 minutes
**Files Modified:** 1
**False Positives:** 1

---

## Summary

Fixed SQL schema error in production code where `mart.player_game_stats.game_id` was incorrectly referenced (column doesn't exist in schema). Replaced with correct composite key join pattern using `(season, week, team)`.

---

## Files Modified

### 1. `/Users/dro/rice/nfl-analytics/py/production/bayesian_online_updater.py` (line 450)

**Issue:** Incorrect join using non-existent `game_id` column
**Severity:** HIGH (production code, would fail at runtime)

**Before:**
```sql
JOIN mart.player_game_stats pgs
    ON pr.player_id = pgs.player_id
    AND pr.stat_type = pgs.stat_type
    AND pr.season = pgs.season
    AND pr.week = pgs.week
JOIN games g ON pgs.game_id = g.game_id  -- ❌ WRONG: pgs doesn't have game_id
```

**After:**
```sql
JOIN mart.player_game_stats pgs
    ON pr.player_id = pgs.player_id
    AND pr.stat_type = pgs.stat_type
    AND pr.season = pgs.season
    AND pr.week = pgs.week
JOIN games g
    ON pgs.season = g.season
    AND pgs.week = g.week
    AND pgs.team IN (g.home_team, g.away_team)  -- ✅ CORRECT: composite key
```

**Rationale:**
- `mart.player_game_stats` has NO `game_id` column (per schema audit)
- Correct join uses composite key: `(season, week, team)`
- Games table has `home_team` and `away_team`, so use `IN` clause to match either

---

## False Positives

### `/Users/dro/rice/nfl-analytics/py/models/train_xgboost_touchdowns.py` (lines 180, 205)

**Status:** ✅ No fix needed - false positive

**Why False Positive:**
- The `pgs` alias in this file refers to a **CTE** (Common Table Expression), not `mart.player_game_stats`
- The CTE is defined starting at line 95 and includes `p.game_id` from the `plays` table (line 97)
- Therefore, `pgs.game_id` is valid in this context

**CTE Definition (line 95-97):**
```sql
WITH player_game_stats AS (
    SELECT
        p.game_id,  -- ✅ Valid: comes from plays table
        g.season,
        g.week,
        ...
```

**Usage in defense_stats CTE (line 180):**
```sql
SELECT
    pgs.game_id,  -- ✅ Valid: pgs is CTE, not mart table
    pgs.opponent,
    pgs.stat_type,
    AVG(pgs.scored_td::numeric) as def_td_rate_allowed
FROM player_game_stats pgs  -- CTE reference
```

---

## Validation

**Grep Search Results:**
- Total instances of `pgs.game_id`: 89 across codebase
- Production-critical errors: 1 (bayesian_online_updater.py)
- False positives: 1 (train_xgboost_touchdowns.py)
- Documentation/examples: 87 (no action needed)

**Schema Validation:**
- ✅ Verified `mart.player_game_stats` schema has NO `game_id` column
- ✅ Verified correct join pattern from `docs/database/SQL_PATTERNS.md`
- ✅ Confirmed composite key `(season, week, team)` is standard

---

## Impact

**Before Fix:**
- Runtime error when bayesian_online_updater.py executes
- Error message: `column pgs.game_id does not exist`
- Online learning system would fail to update posteriors

**After Fix:**
- Query executes correctly
- Joins `mart.player_game_stats` to `games` using composite key
- Online learning system can update posteriors for recent games

---

## Related Documentation

- Schema reference: `docs/database/DATABASE_QUICK_REFERENCE.md`
- SQL patterns: `docs/database/SQL_PATTERNS.md`
- Original audit: `docs/audits/ROUND1_CONSOLIDATED_FINDINGS.md`

---

## Next Steps

1. ✅ SQL schema error fixed
2. ⏳ Proceed to Quick Wins (connection pooling, batch queries, indexes)
3. ⏳ Add schema validation to prevent future errors

---

**Fix completed:** November 22, 2025
**Next phase:** Quick Wins (Phase 1b) - 5.3 hours estimated
