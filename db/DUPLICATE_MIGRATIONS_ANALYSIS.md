# Duplicate Migration Prefix Analysis

> **Historical (December 2025).** The consolidation it prepared was never done. `db/migrations/`
> is now history and the schema snapshot in `db/schema/` is the bootstrap ([db/README.md](README.md)).

**Date:** December 10, 2025
**Purpose:** Detailed analysis of duplicate migration prefixes
**Status:** Technical deep-dive for consolidation decisions

---

## Overview

Out of 70 migration files, **21 files share 6 duplicate prefixes** (018, 026, 048, 050, 101, 102). This represents **30% of all migrations** being duplicate or overlapping work.

---

## Duplicate Group 1: Prefix 018 (5 files)

### Context
The 018 prefix represents **critical database consistency fixes** that evolved through multiple iterations as schema issues were discovered during production use.

### Files Analysis

#### File 1: `018_predictions_schema.sql` (384 lines)
**Created:** Early in project (migration tracking infrastructure)
**Purpose:** RL-based prediction tracking system
**Key Components:**
- `predictions.game_predictions` table
- `predictions.prediction_versions` table
- `predictions.retrospectives` table (post-game analysis)
- `predictions.learning_loop` table (pattern discovery)
- Views: latest_predictions, accuracy_summary, learning_opportunities
- Functions: get_prediction_evolution(), games_needing_retrospective()

**Assessment:**
- Comprehensive prediction infrastructure
- Well-designed schema for ML feedback loops
- No overlaps with other 018 files (different purpose)
- **Recommendation:** Extract to separate migration `017_predictions_schema.sql`

---

#### File 2: `018_pipeline_tracking.sql` (44 lines)
**Created:** During production automation setup
**Purpose:** Track automated refresh pipeline runs
**Key Components:**
- `pipeline_refresh_log` table (season, week, success, duration)
- Basic indexes
- Minimal permissions

**Assessment:**
- Simple, focused table
- Complements prediction schema but separate concern
- **Recommendation:** Merge into `018_database_consistency_complete.sql` (monitoring section)

---

#### File 3: `018_critical_fixes_v2.sql` (215 lines)
**Created:** After discovering team abbreviation issues
**Purpose:** Fix team codes (LA→LAR) and player ID fragmentation
**Key Components:**
- `reference.team_abbreviations` table (32 teams)
- `normalize_team_abbr()` function
- `lookup_player_ids()` function
- Data type fixes (injuries.gsis_id, bayesian_player_ratings.player_id)
- Primary key fix for `mart.rolling_features`
- `v_rosters_with_player_info` view
- `v_games_normalized` view

**Issues Identified:**
- View creation depended on functions
- Column name inconsistency: `player_name` vs actual schema
- **Recommendation:** Superseded by v3, archive

---

#### File 4: `018_critical_fixes_v3.sql` (222 lines)
**Created:** Immediately after v2 failed in production
**Purpose:** Idempotent version of v2 with proper dependency ordering
**Key Improvements over v2:**
- Drops dependent views before recreating functions (fixes dependency issue)
- Uses `CASCADE` on function drops
- Fixed column name: `players_name` instead of `player_name`
- Otherwise identical to v2

**Assessment:**
- Proper fix for v2's issues
- Production-tested and working
- **Recommendation:** Use as canonical version, archive v2

---

#### File 5: `018_database_consistency_fixes.sql` (319 lines)
**Created:** Comprehensive consolidation attempt
**Purpose:** Most complete version combining all consistency fixes
**Key Components:**
- Everything from v3
- **Additional:** player_id_mapping comprehensive view
- **Additional:** full team metadata (city, division, conference)
- **Additional:** position groups table references
- **Additional:** Migration verification view
- **Additional:** schema_migrations tracking table

**Assessment:**
- Most comprehensive solution
- Includes migration tracking infrastructure
- Well-documented with verification
- **Recommendation:** USE AS BASE for consolidated migration

---

### Consolidation Decision for 018 Series

**Strategy:** Multi-file split based on concern separation

1. **`017_predictions_schema.sql`**
   - Source: `018_predictions_schema.sql` (rename)
   - Purpose: ML prediction tracking infrastructure
   - Size: ~384 lines

2. **`018_database_consistency_complete.sql`**
   - Base: `018_database_consistency_fixes.sql`
   - Add from: `018_pipeline_tracking.sql` (monitoring section)
   - Purpose: Team codes, player IDs, primary keys, data types
   - Size: ~350 lines

**Archive:**
- `018_critical_fixes_v2.sql` (superseded by v3)
- `018_critical_fixes_v3.sql` (incorporated into complete)

**Rationale:** Prediction schema is separate concern from database consistency fixes

---

## Duplicate Group 2: Prefix 026 (4 files)

### Context
Prefix 026 represents a **collision of unrelated features** added around same time (November 2025).

### Files Analysis

#### File 1: `026_canonical_training_window.sql` (180 lines)
**Created:** November 8, 2025
**Purpose:** Establish canonical training windows for ML models
**Key Components:**
- `ml.models` table registration for state-space v2
- `ml.canonical_training_windows` table
- Training window metadata (2015-2024, H=4 decay)
- Validation methodology documentation
- Foreign key linking models to windows

**Assessment:**
- Critical ML infrastructure
- Well-documented (references ablation study)
- No overlap with other 026 files
- **Recommendation:** KEEP as core, merge nflreadr alignment into it

---

#### File 2: `026_add_nflreadr_alignment_columns.sql` (47 lines)
**Created:** November 7, 2025
**Purpose:** Fix schema mismatch between nflreadr and database
**Key Components:**
- Add `play_type` column (was missing!)
- Add `drive`, `series`, `series_success`, `series_result` columns
- Indexes on new columns
- Documentation comments

**Schema Issue Fixed:**
- nflreadr has had these columns for years
- Database was missing them (ETL failure)
- Queries were breaking on these columns

**Assessment:**
- Small, focused fix
- Logically related to data layer (not training)
- **Recommendation:** Merge into `026_model_training_config.sql` (schema fixes section)

---

#### File 3: `026_paper_trading_schema.sql` (227 lines)
**Created:** November 7, 2025
**Purpose:** Complete paper trading system for ensemble validation
**Key Components:**
- `paper_trading` schema
- `positions` table (individual bets)
- `performance` table (daily metrics)
- `bankroll_history` table
- `odds_snapshots` table
- Views: v_active_positions, v_performance_summary, v_model_performance
- Triggers: auto-update timestamps
- Initial $10,000 bankroll

**Assessment:**
- Comprehensive standalone system
- No overlap with other 026 files
- Large enough to warrant own migration
- **Recommendation:** KEEP SEPARATE, renumber to `027_paper_trading.sql`

---

#### File 4: `026_pinnacle_exchange_fields.sql` (194 lines)
**Created:** November 6, 2025
**Purpose:** Add low-vig bookmaker tracking and analysis
**Key Components:**
- Alter `prop_lines_history` table (exchange fields)
- `is_exchange`, `back_price`, `lay_price`, `liquidity` columns
- `bookmaker_tier` classification
- Update existing records (Pinnacle=sharp, DK/FD=square, Betfair=exchange)
- Views: best_pinnacle_lines, pinnacle_vs_square_comparison
- Materialized view: pinnacle_historical_performance

**Assessment:**
- Props-related infrastructure
- Logically belongs with props system
- Created same day as paper trading (explains numbering collision)
- **Recommendation:** Merge into `019_props_infrastructure.sql`

---

### Consolidation Decision for 026 Series

**Strategy:** Split by functional area

1. **`026_model_training_config.sql`**
   - Base: `026_canonical_training_window.sql`
   - Add: `026_add_nflreadr_alignment_columns.sql`
   - Purpose: ML training configuration + schema alignment
   - Size: ~230 lines

2. **`027_paper_trading.sql`**
   - Source: `026_paper_trading_schema.sql` (rename)
   - Purpose: Paper trading system
   - Size: ~227 lines

**Archive:**
- None (all components used)

**Move:**
- `026_pinnacle_exchange_fields.sql` → merge into `019_props_infrastructure.sql`

---

## Duplicate Group 3: Prefix 048 (5 files)

### Context
The **most problematic duplicate group**. Prefix 048 evolved as the game predictions infrastructure grew in complexity. Multiple attempts to patch and enhance led to 5 overlapping files.

### Files Analysis

#### File 1: `048_create_game_predictions.sql` (560 lines)
**Created:** November 21, 2025
**Purpose:** Complete game-level prediction infrastructure
**Key Components:**
- `ml.game_predictions` table (46 columns!)
  - Game identification (game_id, model_id, season, week)
  - Predictions (scores, spread, total, win probability)
  - Uncertainty quantification (std deviations)
  - Sportsbook lines (spread, total, moneylines)
  - Edge calculations (model vs sportsbook)
  - Actual results (post-game)
  - Performance metrics (MAE, Brier, log loss)
  - Bet outcomes (cover, hit, correct)
  - Quality indicators (confidence, outlier_flag)
- `ml.game_prediction_metrics` table (aggregated stats)
- Materialized views (3):
  - mv_current_week_predictions
  - mv_model_performance_summary
  - mv_high_edge_opportunities
- Triggers (2):
  - Auto-calculate edges on insert/update
  - Auto-calculate metrics when actuals filled
- Functions (3):
  - calculate_prediction_edges()
  - calculate_prediction_metrics()
  - refresh_game_prediction_metrics()
  - refresh_game_prediction_views()
- 14 indexes

**Assessment:**
- Extremely comprehensive
- Well-designed with triggers for automation
- **Issue:** Some columns referenced don't exist initially
- **Recommendation:** USE AS BASE, merge ADDENDUM columns

---

#### File 2: `048_create_game_predictions_ADDENDUM.sql` (526 lines)
**Created:** November 21, 2025 (same day as main)
**Purpose:** Add missing components discovered after main migration
**Key Components:**
- ALTER TABLE to add missing columns:
  - `predicted_at` (timestamp)
  - `win_prob_std` (uncertainty)
  - `sportsbook_source` (bookmaker tracking)
  - `actual_home_win` (outcome boolean)
  - Performance metrics (spread_mae, total_mae, brier_score, log_loss)
  - Bet outcomes (spread_cover, total_hit, ml_correct)
  - Quality indicators (prediction_confidence, outlier_flag, notes)
- Creates `ml.game_prediction_metrics` table (same as main! Duplicate!)
- Recreates all 3 materialized views (same as main! Duplicate!)
- Recreates triggers and functions (same as main! Duplicate!)

**Issues Identified:**
- Massive duplication with main file
- Suggests main file was incomplete at creation
- Uses `IF NOT EXISTS` / `DO $$` blocks for idempotency
- **Problem:** If both run, you get duplicate function/trigger definitions

**Assessment:**
- Created to patch incomplete main migration
- Half the content is duplicate code
- **Recommendation:** Merge missing columns into main, delete duplicates

---

#### File 3: `048_create_views_compatibility_patch.sql` (160 lines)
**Created:** November 21, 2025 (same day!)
**Purpose:** Fix materialized views to work with actual table schema
**Key Components:**
- Drops and recreates all 3 materialized views
- Uses actual column names from deployed table:
  - `abs_spread_error` instead of `spread_mae`
  - `home_won` instead of `actual_home_win`
  - Hardcoded current week (2025, week 12)
- Simplified queries without model_registry joins
- Fixed refresh function

**Issues Identified:**
- Suggests table was deployed with different column names than migration specified
- Hardcoded season/week (breaks in 2026!)
- Missing model_registry join (data gap)

**Assessment:**
- Emergency patch for production deployment
- **Root cause:** Main migration had wrong column names
- **Recommendation:** Use correct column names in consolidated version, archive this

---

#### File 4: `048_fix_thursday_night_flag.sql` (130 lines)
**Created:** November 22, 2025 (day after predictions)
**Purpose:** Populate is_thursday_night column with actual data
**Key Components:**
- UPDATE statement: `SET is_thursday_night = TRUE WHERE EXTRACT(DOW FROM kickoff) = 4`
- Research documentation (TNF analysis)
- Validation queries
- Rollback instructions
- **NOT a schema change - a DATA UPDATE**

**Assessment:**
- Completely different purpose than other 048 files
- Should NOT share prefix (no schema changes)
- Well-documented with research findings
- **Recommendation:** KEEP SEPARATE as `049_thursday_night_data_fix.sql`

---

#### File 5: `048_prediction_accuracy_view.sql` (138 lines)
**Created:** November 21, 2025
**Purpose:** Create comprehensive accuracy tracking view
**Key Components:**
- `ml.prediction_accuracy` view
  - Joins predictions with actual outcomes
  - Calculates all error metrics (MAE components)
  - Signed errors (bias detection)
  - Boolean accuracy indicators
  - Spread coverage analysis
  - Edge accuracy validation
  - Uncertainty calibration checks
- Indexes on game_id, ensemble_version, season/week

**Assessment:**
- Useful analytics view
- No overlap with other 048 files
- Complements main predictions table
- **Recommendation:** Merge into consolidated 048

---

### Consolidation Decision for 048 Series

**Strategy:** Merge schema changes, separate data updates

1. **`048_game_predictions_complete.sql`**
   - Base: `048_create_game_predictions.sql`
   - Add missing columns from: ADDENDUM
   - Add view from: `048_prediction_accuracy_view.sql`
   - **Remove duplicates:** Don't recreate functions/views twice
   - **Fix column names:** Use production-tested names
   - Purpose: Complete game predictions infrastructure
   - Size: ~650 lines

2. **`049_thursday_night_data_fix.sql`**
   - Source: `048_fix_thursday_night_flag.sql` (rename)
   - Purpose: Data update, not schema
   - Size: ~130 lines

**Archive:**
- `048_create_game_predictions_ADDENDUM.sql` (merged into main)
- `048_create_views_compatibility_patch.sql` (emergency patch, superseded)

**Critical Fix Required:**
- Verify all column names match production schema
- Remove duplicate function/trigger definitions
- Fix hardcoded season/week in views (use dynamic calculation)

---

## Duplicate Group 4: Prefix 050 (2 files)

### Files Analysis

#### File 1: `050_add_coaching_staff_tables.sql`
**Purpose:** Add coaching staff tracking
**Assessment:** Core schema addition, keep as primary

#### File 2: `050_dashboard_performance_indexes.sql`
**Purpose:** Add indexes for dashboard queries
**Assessment:** Related to coaching data, merge into main

**Consolidation:** Merge both → `050_coaching_staff.sql`

---

## Duplicate Group 5: Prefix 101 (2 files)

### Files Analysis

#### File 1: `101_add_foreign_key_constraints.sql`
**Purpose:** Add referential integrity
**Assessment:** Database housekeeping

#### File 2: `101_stat_type_performance.sql`
**Purpose:** Track model performance by stat type
**Assessment:** Analytics tracking

**Consolidation:** Merge both → `101_analytics_tracking.sql`

---

## Duplicate Group 6: Prefix 102 (2 files)

### Files Analysis

#### File 1: `102_causal_feature_tables.sql`
**Purpose:** Causal inference framework
**Assessment:** Research feature

#### File 2: `102_dirichlet_alpha_by_stat_type.sql`
**Purpose:** Dirichlet optimization parameters
**Assessment:** Related to ensemble weights

**Consolidation:** Merge both → `102_dirichlet_optimization.sql`

---

## Root Cause Analysis

### Why Do We Have Duplicates?

1. **Rapid Development Cycle** (018, 048)
   - Features deployed to production
   - Issues discovered in production
   - Emergency patches created
   - Patches not consolidated back into main

2. **Parallel Development** (026)
   - Multiple features developed simultaneously
   - Developers used next available number
   - No coordination on numbering
   - All merged to main on same day

3. **Iterative Fixes** (018 v2 → v3)
   - Fix deployed
   - Bug found
   - New version created
   - Old version not removed

4. **Scope Creep** (048 ADDENDUM)
   - Migration created incomplete
   - Missing columns discovered
   - Addendum created instead of fixing original
   - Duplication ensued

### Process Improvements

1. **Migration Numbering Policy**
   - Reserve number before development
   - Use feature branches
   - Consolidate before merge to main

2. **Schema Review Process**
   - Pre-deployment schema validation
   - Column name consistency check
   - Duplicate detection

3. **Emergency Patch Protocol**
   - Document why patch needed
   - Plan consolidation date
   - Track technical debt

---

## Consolidation Checklist

For each duplicate group:

- [ ] Identify canonical base version
- [ ] List unique features from each variant
- [ ] Merge unique features into base
- [ ] Remove duplicate code
- [ ] Fix any bugs/issues from variants
- [ ] Test idempotency
- [ ] Test on empty database
- [ ] Test on production backup
- [ ] Verify schema matches production
- [ ] Document consolidation in header
- [ ] Archive old variants with explanation
- [ ] Update references in code/docs

---

## Appendix: Column Name Inconsistencies

### 048 Series Issues

| Migration File | Column Name | Production Schema | Status |
|----------------|-------------|-------------------|--------|
| Main | `spread_mae` | `abs_spread_error` | ❌ Mismatch |
| Main | `total_mae` | `abs_total_error` | ❌ Mismatch |
| Main | `actual_home_win` | `home_won` | ❌ Mismatch |
| Compatibility Patch | Fixed names | Matches production | ✅ Correct |

**Resolution:** Use compatibility patch column names in consolidated migration

---

**Document Status:** Technical Deep-Dive Complete
**Next Step:** Execute consolidation plan
**Updated:** December 10, 2025
