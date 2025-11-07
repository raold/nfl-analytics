# Schema Alignment Solution - nflreadr vs Database
## Date: November 7, 2025

## Root Cause Analysis

**Problem**: Recurring schema mismatches between nflreadr package and our PostgreSQL database schema have been causing ingestion failures for months.

**Current Error**: `column "play_type" of relation "plays" does not exist`

**Why This Keeps Happening**:
- nflverse/nflreadr is an **upstream data source** maintained by nflverse community
- They add new columns, rename columns, and update schemas regularly
- Our database schema was created at a point in time and hasn't kept up
- Ingestion script tries to select columns that exist in nflreadr but not in our DB

## Schema Comparison Analysis

### Database Schema (71 columns)
Our `plays` table has 71 columns, missing several nflreadr columns including:
- `play_type` (the current blocker)
- `drive` (we have `fixed_drive` but not `drive`)
- Many advanced metrics added to nflreadr over time

### nflreadr Provides (369 columns)
nflreadr's play-by-play data has **369 columns** including:
- Core play data (game_id, play_id, down, ydstogo, etc.)
- EPA/WPA metrics (epa, wp, wpa, etc.)
- Player tracking (passer_player_id, rusher_player_id, etc.)
- Advanced metrics (xyac_epa, cpoe, etc.)
- Drive-level data (drive, series, etc.)
- **play_type** - categorical play type (pass, run, kickoff, punt, etc.)

### Key Missing Columns in Database

| nflreadr Column | Database Status | Impact | Priority |
|----------------|----------------|--------|----------|
| `play_type` | ❌ Missing | **HIGH** - Currently blocking ingestion | **CRITICAL** |
| `drive` | ❌ Missing (have `fixed_drive`) | Medium - Used for drive-level analysis | High |
| `series` | ❌ Missing | Low - Series tracking | Medium |
| `series_success` | ❌ Missing | Low - Series outcomes | Medium |
| `play_type_nfl` | ❌ Missing | Low - NFL's categorization | Low |
| `st_play_type` | ❌ Missing | Low - Special teams detail | Low |

## Recommended Solution: **Hybrid Approach**

After analyzing the situation, I recommend a **hybrid approach** that balances maintainability with stability:

### Phase 1: Add Missing Critical Columns (IMMEDIATE)
**Action**: Add the most important missing columns to database schema via SQL migration

**Rationale**:
1. nflverse is mostly stable - `play_type` has existed for years
2. Adding a few critical columns is a one-time fix
3. Doesn't require transforming 369 columns on every ingestion
4. Preserves our carefully designed schema constraints

**Columns to Add**:
```sql
-- Migration 026: Add nflreadr alignment columns
ALTER TABLE plays ADD COLUMN IF NOT EXISTS play_type TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS drive DOUBLE PRECISION;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS series INTEGER;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS series_success BOOLEAN;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS series_result TEXT;
```

**Benefits**:
- ✅ Immediate fix for ingestion blocker
- ✅ Minimal changes to existing schema
- ✅ Preserves data types and constraints
- ✅ No transformation overhead on every ingestion

### Phase 2: Robust Column Mapping (MAINTENANCE)
**Action**: Create a defensive column mapping in the ingestion script

**Rationale**:
1. Even after adding critical columns, nflreadr may add new ones
2. Script should handle missing columns gracefully
3. Log warnings when new columns appear in nflreadr

**Implementation**:
```r
# Define our canonical column set
CANONICAL_COLUMNS <- c(
  # Core play identification
  "game_id", "play_id", "posteam", "defteam", "quarter",
  "time_seconds", "game_seconds_remaining", "half_seconds_remaining",
  "down", "ydstogo", "yardline_100",

  # Play type and outcome
  "play_type", "yards_gained", "first_down",
  "third_down_converted", "fourth_down_converted",

  # Play category flags
  "pass", "rush", "special", "touchdown",
  "pass_touchdown", "rush_touchdown", "return_touchdown",

  # Kicking plays
  "extra_point_attempt", "two_point_attempt",
  "field_goal_attempt", "kickoff_attempt", "punt_attempt",

  # Turnovers
  "fumble", "interception", "fumble_lost",

  # Other events
  "timeout", "penalty", "penalty_yards",
  "qb_spike", "qb_kneel",

  # Advanced metrics
  "epa", "wp", "wpa", "vegas_wp", "vegas_wpa", "success",

  # Passing metrics
  "air_yards", "yards_after_catch", "cpoe",
  "comp_air_epa", "comp_yac_epa",
  "complete_pass", "incomplete_pass",
  "pass_length", "pass_location",
  "qb_hit", "qb_scramble", "sack",

  # Running metrics
  "run_location", "run_gap",

  # Player IDs
  "passer_player_id", "passer_player_name",
  "rusher_player_id", "rusher_player_name",
  "receiver_player_id", "receiver_player_name",

  # Formation
  "shotgun", "no_huddle", "qb_dropback",

  # Score tracking
  "posteam_score", "defteam_score", "score_differential",
  "posteam_score_post", "defteam_score_post",

  # Drive tracking
  "drive", "fixed_drive", "fixed_drive_result",

  # Series tracking
  "series", "series_success", "series_result"
)

# Defensive column selection
plays_df <- pbp %>%
  select(any_of(CANONICAL_COLUMNS))

# Log missing columns (for monitoring)
missing_from_nflreadr <- setdiff(CANONICAL_COLUMNS, names(pbp))
if (length(missing_from_nflreadr) > 0) {
  log_message(
    sprintf("Warning: nflreadr missing expected columns: %s",
            paste(missing_from_nflreadr, collapse=", ")),
    level = "WARN"
  )
}

# Log new columns in nflreadr (for discovery)
extra_in_nflreadr <- setdiff(names(pbp), CANONICAL_COLUMNS)
if (length(extra_in_nflreadr) > 0) {
  log_message(
    sprintf("Info: nflreadr has %d additional columns not in our schema",
            length(extra_in_nflreadr)),
    level = "INFO"
  )
}
```

### Phase 3: Automated Schema Drift Detection (FUTURE)
**Action**: Create monitoring to detect when nflreadr schema changes

**Implementation**:
1. Store nflreadr column list on each ingestion
2. Compare to previous ingestion
3. Alert on schema changes
4. Log to database: `data_quality.schema_drift_log`

## Why NOT Full Schema Sync?

**Option Rejected**: Automatically sync database to match all 369 nflreadr columns

**Why Rejected**:
1. **Overwhelming**: Most of nflreadr's 369 columns are not needed (tackle_for_loss_2_player_name, etc.)
2. **Semantic Differences**: Some nflreadr columns have unclear definitions or quality issues
3. **Performance**: Storing 369 columns per play = massive storage overhead for minimal value
4. **Constraints**: Our schema has carefully designed constraints (CHECK, NOT NULL) that wouldn't map
5. **Feature Engineering**: We compute our own derived features optimized for betting
6. **Maintenance**: Would require ALTER TABLE for every nflreadr update (frequent)

## Why NOT Transform in Script?

**Option Rejected**: Create transformation layer to rename all columns in ingestion script

**Why Rejected**:
1. **Unnecessary**: Most columns already match (game_id, play_id, epa, etc.)
2. **Performance**: Transforming 369 columns on 50K+ plays per season = overhead
3. **Error-Prone**: Complex mapping logic = bugs
4. **The Real Issue**: We're not renaming columns, we're just missing a few critical ones

## Implementation Plan

### Step 1: Create SQL Migration (IMMEDIATE)
```sql
-- db/migrations/026_add_nflreadr_alignment_columns.sql
BEGIN;

-- Add missing columns from nflreadr that are commonly used
ALTER TABLE plays ADD COLUMN IF NOT EXISTS play_type TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS drive DOUBLE PRECISION;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS series INTEGER;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS series_success BOOLEAN;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS series_result TEXT;

-- Add indexes for commonly queried columns
CREATE INDEX IF NOT EXISTS idx_plays_play_type ON plays(play_type);
CREATE INDEX IF NOT EXISTS idx_plays_drive ON plays(game_id, drive);
CREATE INDEX IF NOT EXISTS idx_plays_series ON plays(game_id, series);

-- Add comments
COMMENT ON COLUMN plays.play_type IS 'Play type from nflreadr (pass, run, kickoff, punt, etc.)';
COMMENT ON COLUMN plays.drive IS 'Drive number from nflreadr';
COMMENT ON COLUMN plays.series IS 'Series number (set of downs) from nflreadr';
COMMENT ON COLUMN plays.series_success IS 'Whether series resulted in first down';
COMMENT ON COLUMN plays.series_result IS 'Outcome of series (First down, Touchdown, etc.)';

COMMIT;
```

### Step 2: Apply Migration
```bash
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -f db/migrations/026_add_nflreadr_alignment_columns.sql
```

### Step 3: Update Ingestion Script
- Replace ad-hoc `select(any_of(...))` with `CANONICAL_COLUMNS` constant
- Add logging for missing/new columns
- Document which columns we intentionally exclude

### Step 4: Run Ingestion
```bash
Rscript R/ingestion/ingest_current_season.R --season 2025
```

### Step 5: Verify Success
```sql
-- Check new columns exist
SELECT
  COUNT(*) as total_plays,
  COUNT(play_type) as plays_with_type,
  COUNT(DISTINCT play_type) as unique_types,
  COUNT(drive) as plays_with_drive,
  COUNT(series) as plays_with_series
FROM plays
WHERE game_id LIKE '2025_%';
```

## Monitoring & Maintenance

### Quarterly Review
Every 3 months, compare nflreadr schema to our schema:
```r
# Compare schemas
nflreadr_cols <- names(nflreadr::load_pbp(2025))
db_cols <- DBI::dbListFields(conn, "plays")

new_in_nflreadr <- setdiff(nflreadr_cols, db_cols)
if (length(new_in_nflreadr) > 0) {
  message("New columns in nflreadr: ", paste(new_in_nflreadr, collapse=", "))
  # Evaluate if any should be added to database
}
```

### Documentation
- Document intentionally excluded columns (tackle_with_assist_2_player_id, etc.)
- Document column mapping decisions
- Update this document when adding new columns

## Expected Outcome

After implementing this solution:

1. ✅ **Immediate**: Ingestion succeeds, no more `column does not exist` errors
2. ✅ **Short-term**: Robust handling of missing columns via `any_of()`
3. ✅ **Long-term**: Monitoring alerts when nflreadr adds important new columns
4. ✅ **Maintainability**: Clear documentation of schema alignment strategy
5. ✅ **Performance**: No transformation overhead, minimal storage overhead

## Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Sync all 369 columns** | Complete alignment | Massive storage, unclear semantics | ❌ Rejected |
| **Transform in script** | No schema changes | Complex mapping, performance hit | ❌ Rejected |
| **Add critical columns** | Simple, targeted | Requires occasional migrations | ✅ **CHOSEN** |
| **Ignore nflreadr changes** | No work | Recurring ingestion failures | ❌ Rejected |

## Summary

**Root Cause**: nflreadr evolves (adds columns), our schema doesn't, ingestion script breaks

**Solution**: Hybrid approach
1. Add 5 critical missing columns via SQL migration
2. Use defensive `any_of()` selection in ingestion script
3. Monitor for new columns quarterly

**Result**: Permanent fix that balances stability with flexibility

---

**Created**: November 7, 2025
**Author**: Claude Sonnet 4.5
**Status**: Ready for Implementation
