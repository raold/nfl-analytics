# SQL Schema Audit - November 14, 2025

## Executive Summary

**Total queries auditing:** 121 files reference `mart.player_game_stats`
**Broken queries found:** 6 files (5 unique query patterns)
**Documentation errors:** 2 files
**Status:** ❌ CRITICAL - Production training scripts affected

## The Bug

**Incorrect Assumption:** Queries assume `mart.player_game_stats.game_id` exists
**Reality:** The view only has `game_id_pattern` (a computed string, not a join key)
**Correct Join:** Use composite key: `(season, week, team)` to join with `games` table

### Schema Definition (db/migrations/023_player_hierarchy_schema.sql:136-212)

```sql
CREATE MATERIALIZED VIEW mart.player_game_stats AS
...
SELECT
    a.*,
    ph.position_group,
    ph.current_team,
    ph.hierarchy_position,
    ph.hierarchy_team_position,
    -- Game identifier for joining with games table
    CASE
        WHEN ph.current_team IS NOT NULL THEN
            season || '_' || LPAD(week::TEXT, 2, '0') || '_' ||
            ph.current_team || '_OPP'
        ELSE NULL
    END as game_id_pattern  -- ⚠️ NOT game_id!
FROM all_stats a
LEFT JOIN mart.player_hierarchy ph ON a.player_id = ph.player_id;
```

**Key Columns:**
- `player_id` (text)
- `season` (integer)
- `week` (integer)
- `stat_category` (text: 'passing', 'rushing', 'receiving')
- `position_group` (text: 'QB', 'RB', 'WR', 'TE')
- `current_team` (text: team abbreviation)
- `game_id_pattern` (computed text, NOT for joins)

---

## Broken Queries (MUST FIX)

### Priority 1: Production Training Scripts (3 files)

#### 1. scripts/train_hybrid_xgboost_v1.py
**Line:** 148
**Status:** ❌ BROKEN - Production training script
**Impact:** HIGH - Used for hybrid XGBoost training (Week 2 integration)
**Issue:**
```python
LEFT JOIN games g ON pgs.game_id = g.game_id
```
**Fix:**
```python
LEFT JOIN games g ON
    g.season = pgs.season
    AND g.week = pgs.week
    AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
```
**Notes:** Requires `mart.player_hierarchy` join for `current_team`. Already has `ph` alias in query (line 147).

---

#### 2. scripts/train_all_phases_scaled.py
**Lines:** 121, 306
**Status:** ❌ BROKEN - Two identical queries in same file
**Impact:** HIGH - Used for multi-phase model training
**Issue:**
```python
JOIN games g ON pgs.game_id = g.game_id
LEFT JOIN weather w ON g.game_id = w.game_id
WHERE pgs.rushing_yards > 0
```
**Fix:**
```python
JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
LEFT JOIN games g ON
    g.season = pgs.season
    AND g.week = pgs.week
    AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
LEFT JOIN weather w ON g.game_id = w.game_id
WHERE pgs.rushing_yards > 0
```
**Notes:**
- Query is duplicated at line 121 and line 306 (same bug twice)
- Uses `pgs.rushing_yards` instead of `pgs.stat_yards` (correct for rushing-specific query)
- Must add player_hierarchy join to get `current_team`

---

#### 3. scripts/bnn_optuna_search.py
**Line:** 35
**Status:** ❌ BROKEN - Hyperparameter search script
**Impact:** MEDIUM - Used for BNN optimization (research)
**Issue:**
```python
JOIN games g ON pgs.game_id = g.game_id
LEFT JOIN weather w ON g.game_id = w.game_id
WHERE pgs.rushing_yards > 0
```
**Fix:** Same as train_all_phases_scaled.py above

---

### Priority 2: Production Pipeline (1 file)

#### 4. py/production/bayesian_online_updater.py
**Line:** 450
**Status:** ❌ BROKEN - Production online learning updater
**Impact:** HIGH - Used for Bayesian online updates (Week 3 feature)
**Issue:**
```python
JOIN games g ON pgs.game_id = g.game_id
WHERE pr.model_version = 'informative_priors_v2.5'
  AND pgs.stat_yards IS NOT NULL
  AND g.gameday >= CURRENT_DATE - INTERVAL '7 days'
```
**Fix:**
```python
LEFT JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
LEFT JOIN games g ON
    g.season = pgs.season
    AND g.week = pgs.week
    AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
WHERE pr.model_version = 'informative_priors_v2.5'
  AND pgs.stat_yards IS NOT NULL
  AND g.gameday >= CURRENT_DATE - INTERVAL '7 days'
```
**Notes:**
- Query joins `mart.bayesian_player_ratings` with `player_game_stats`
- Already filters by season/week (lines 448-449), can validate join correctness
- Must add player_hierarchy to get team

---

### Priority 3: Legacy/Research Scripts (1 file)

#### 5. scripts/train_best_bnn.py
**Line:** 41
**Status:** ❌ BROKEN - Legacy BNN training script
**Impact:** LOW - BNN research line abandoned (Nov 9, 2025)
**Issue:**
```python
JOIN games g ON pgs.game_id = g.game_id
LEFT JOIN weather w ON g.game_id = w.game_id
WHERE pgs.rushing_yards > 0
AND pgs.season >= 2020
```
**Fix:** Same as train_all_phases_scaled.py
**Notes:**
- May not need fix (abandoned research)
- Consider deleting file if no longer used

---

## Documentation Errors (2 files)

### 1. docs/database/DATABASE_QUICK_REFERENCE.md
**Line:** 128
**Status:** ⚠️ INCORRECT DOCUMENTATION
**Issue:**
```markdown
2. Join `mart.player_game_stats` on `player_id` and `game_id`
```
**Fix:**
```markdown
2. Join `mart.player_game_stats` on `player_id`, then join `games` on (`season`, `week`, `team`)
```

### 2. docs/archive/2025_nov/PHASE1_PINNACLE_COMPLETE.md
**Line:** 429
**Status:** ⚠️ INCORRECT EXAMPLE
**Issue:**
```sql
SELECT COUNT(*) FROM mart.player_game_stats WHERE game_id = 'YOUR_GAME_ID';
```
**Fix:**
```sql
SELECT COUNT(*) FROM mart.player_game_stats WHERE season = 2024 AND week = 1;
```

---

## Correct Query Examples (Reference)

### Example 1: Enhanced Ensemble v6 (✅ CORRECT)
**File:** py/ensemble/enhanced_ensemble_v6.py
**Lines:** 218-223

```python
FROM mart.player_game_stats pgs
JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
LEFT JOIN games g ON
  g.season = pgs.season
  AND g.week = pgs.week
  AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
```

**Pattern:**
1. Join `player_hierarchy` to get `current_team`
2. Join `games` on `(season, week, team)` composite key
3. Use `OR` condition for home/away matching

---

### Example 2: XGBoost Canonical Training (✅ CORRECT)
**File:** py/models/train_xgboost_canonical.py
**Lines:** 150-155

```python
FROM mart.player_game_stats pgs
JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
LEFT JOIN games g ON
    g.season = pgs.season
    AND g.week = pgs.week
    AND (g.home_team = pgs.current_team OR g.away_team = pgs.current_team)
```

**Pattern:** Identical to Example 1, uses `pgs.current_team` (denormalized from hierarchy)

---

### Example 3: Hierarchical v2 Training (R) (✅ CORRECT)
**File:** R/train_hierarchical_v2_production.R
**Lines:** 81-86

```r
FROM mart.player_game_stats pgs
JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
LEFT JOIN games g ON
  g.season = pgs.season
  AND g.week = pgs.week
  AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
```

**Pattern:** Same composite key pattern works in R

---

### Example 4: State-Space v2 Canonical (R) (✅ CORRECT)
**File:** R/state_space/train_state_space_v2_canonical.R
**Lines:** 73-78

```r
FROM mart.player_game_stats pgs
JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
LEFT JOIN games g ON
  g.season = pgs.season
  AND g.week = pgs.week
  AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
```

**Pattern:** Canonical training window script (Nov 8, 2025 APPROVED)

---

## Why `game_id` Doesn't Exist

### Database Design Decision

The `mart.player_game_stats` view aggregates from three NextGen tables:
- `nextgen_passing`
- `nextgen_rushing`
- `nextgen_receiving`

These tables use `(player_id, season, week)` as the natural key, NOT `game_id`, because:

1. **Player-centric data:** NextGen stats are indexed by player, not game
2. **Multiple games per week:** Players can only play one game per week, but week/season is sufficient key
3. **Team changes:** Players can change teams mid-season, so team is stored in `player_hierarchy`
4. **Normalization:** `game_id` would be redundant given `(season, week, team)` composite key

### The `game_id_pattern` Column

The view computes `game_id_pattern` as:
```sql
season || '_' || LPAD(week::TEXT, 2, '0') || '_' || ph.current_team || '_OPP'
```

**Purpose:** Display/debugging only (not for joins)
**Example:** `2024_01_KC_OPP` (2024, week 1, Kansas City as team)

**Why not use for joins?**
- Opponent is simplified to 'OPP' (not actual opponent)
- Games table uses different `game_id` format (e.g., `2024_01_KC_BUF`)
- String matching is slower and error-prone vs. integer joins

---

## Impact Analysis

### Queries by Status

| Status | Count | Impact | Files |
|--------|-------|--------|-------|
| ✅ Correct (season/week join) | 115+ | None | Most production code |
| ❌ Broken (game_id join) | 6 | HIGH | 4 production scripts |
| ⚠️ Documentation errors | 2 | LOW | Confusing for developers |

### Affected Components

1. **Hybrid XGBoost Training** (Week 2 priority) - ❌ BLOCKED
2. **Multi-Phase Training** - ❌ BLOCKED
3. **Bayesian Online Updater** (Week 3 feature) - ❌ BLOCKED
4. **BNN Research** (abandoned) - ⚠️ Low priority

### Risk Assessment

**Immediate Risk:** MEDIUM
- Production training scripts will fail when executed
- Most queries already use correct pattern (115/121 files)
- Current production models (State-Space v2, Hierarchical v2) use correct queries

**Long-Term Risk:** HIGH if not fixed
- Week 2 Hybrid XGBoost integration will fail
- Week 3 Bayesian online learning will fail
- New developers may copy incorrect patterns from broken scripts

---

## Fix Checklist

### Phase 1: Critical Fixes (2 hours)

- [ ] **scripts/train_hybrid_xgboost_v1.py** (line 148)
  - Add player_hierarchy join
  - Replace `ON pgs.game_id = g.game_id`
  - Test with sample query

- [ ] **scripts/train_all_phases_scaled.py** (lines 121, 306)
  - Fix both queries (identical bug)
  - Add player_hierarchy join
  - Test with rushing data (pgs.rushing_yards > 0)

- [ ] **py/production/bayesian_online_updater.py** (line 450)
  - Add player_hierarchy join
  - Verify week 3 integration not blocked
  - Test with recent games (7-day window)

### Phase 2: Cleanup (30 minutes)

- [ ] **scripts/bnn_optuna_search.py** (line 35)
  - Fix or delete (BNN research abandoned)
  - Archive if no longer needed

- [ ] **scripts/train_best_bnn.py** (line 41)
  - Fix or delete (BNN research abandoned)
  - Archive if no longer needed

### Phase 3: Documentation (15 minutes)

- [ ] **docs/database/DATABASE_QUICK_REFERENCE.md** (line 128)
  - Correct join pattern documentation
  - Add composite key example

- [ ] **docs/archive/2025_nov/PHASE1_PINNACLE_COMPLETE.md** (line 429)
  - Fix example query
  - Update to use season/week filter

### Phase 4: Validation (1 hour)

- [ ] Run all fixed scripts with small test dataset
- [ ] Verify no other references to `pgs.game_id` exist
- [ ] Update CLAUDE.md with lessons learned
- [ ] Create regression test to catch this pattern

---

## Testing Protocol

### 1. Validate Fix (per file)

```bash
# Test query returns data
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 << EOF
[PASTE FIXED QUERY]
LIMIT 10;
EOF
```

### 2. Verify Row Counts

```sql
-- Before fix (will error)
SELECT COUNT(*) FROM mart.player_game_stats pgs
JOIN games g ON pgs.game_id = g.game_id;
-- ERROR: column pgs.game_id does not exist

-- After fix (should return ~1.2M rows)
SELECT COUNT(*) FROM mart.player_game_stats pgs
JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
LEFT JOIN games g ON
    g.season = pgs.season
    AND g.week = pgs.week
    AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
WHERE pgs.season >= 2015;
-- Expected: ~1,254,173 rows (from Nov 8 validation)
```

### 3. Full Training Run

```bash
# Test hybrid XGBoost training (most critical)
python scripts/train_hybrid_xgboost_v1.py --season 2024 --test-run

# Verify no errors in logs
# Check model file created successfully
```

---

## Lessons Learned

### Root Causes

1. **Schema misunderstanding:** Assumed `player_game_stats` had `game_id` (never existed)
2. **Copy-paste errors:** Broken pattern copied across multiple scripts
3. **Incomplete testing:** Scripts not validated against actual database schema
4. **Documentation lag:** Quick reference guide had incorrect example

### Prevention Strategies

1. **Schema validation:** Always check `\d mart.player_game_stats` before writing queries
2. **Use working examples:** Copy from production files (ensemble_v6, xgboost_canonical)
3. **Automated testing:** Add SQL linting to pre-commit hooks
4. **Documentation sync:** Update quick reference when schema changes

### Adding to Project Standards

Add to `CLAUDE.md` Important Patterns:

```markdown
### Database Query Patterns

**Joining player_game_stats with games:**
```sql
-- ✅ CORRECT
FROM mart.player_game_stats pgs
JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
LEFT JOIN games g ON
  g.season = pgs.season
  AND g.week = pgs.week
  AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)

-- ❌ WRONG (game_id doesn't exist)
FROM mart.player_game_stats pgs
JOIN games g ON pgs.game_id = g.game_id
```
```

---

## Summary

**Files Requiring Immediate Fixes:** 4 production scripts
**Estimated Fix Time:** 3.5 hours (including testing)
**Blocking:** Week 2 Hybrid XGBoost integration, Week 3 Bayesian online learning
**Status:** Ready to fix (all patterns identified, correct examples documented)

**Next Steps:**
1. Fix critical production scripts (Phase 1)
2. Test fixes with sample queries
3. Update documentation (Phase 3)
4. Add schema validation to CI/CD

---

**Audit Completed:** November 14, 2025
**Auditor:** Claude Code (Sonnet 4.5)
**Files Scanned:** 121 files with `mart.player_game_stats` references
**Methodology:** grep search + manual code review + schema analysis
