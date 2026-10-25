# Database Migration Consolidation Plan

> **Historical (December 2025): never carried out.** `db/migrations/` is now history and the
> schema snapshot in `db/schema/` is the bootstrap ([db/README.md](README.md)).

**Date:** December 10, 2025
**Analyst:** Legacy System Modernization Expert (Claude Sonnet 4.5)
**Current State:** 70 migration files
**Target State:** ~30 canonical migrations
**Status:** RESEARCH COMPLETE - AWAITING APPROVAL

---

## Executive Summary

The NFL Analytics database has accumulated **70 migration files** over time, with significant duplication and redundancy. This document outlines a comprehensive consolidation plan to reduce this to **~30 canonical migrations** while preserving all schema changes and maintaining backward compatibility.

### Key Findings

- **5 duplicate prefixes identified:** 018 (5 files), 026 (4 files), 048 (5 files), 050 (2 files), 101-102 (duplicates)
- **Redundant migrations:** Multiple schema reconciliation attempts (018, 100)
- **Addendum pattern:** Several migrations are addenda/patches to earlier migrations
- **Safe consolidation potential:** 40-45 files can be safely merged

---

## Migration Inventory by Category

### Category 1: Core Schema Foundations (Keep Separate)
**Files: 3 → 3 (No consolidation needed)**

| Current File | Lines | Purpose | Action |
|--------------|-------|---------|--------|
| `000_complete_schema.sql` | 372 | Complete initial schema setup | **KEEP** - Foundation |
| `001_init.sql` | 83 | Original basic init | **KEEP** - Historical |
| `002_timescale.sql` | 31 | TimescaleDB setup | **KEEP** - Extension setup |

**Rationale:** These form the foundational schema layer and should remain independent.

---

### Category 2: Core Features & Mart Schema (Consolidate to 1)
**Files: 4 → 1**

| Current Files | Consolidate To |
|---------------|----------------|
| `003_mart_game_weather.sql` | `010_mart_analytics_layer.sql` |
| `004_advanced_features.sql` | ↑ (merge into) |
| `005_enhance_mart_views.sql` | ↑ (merge into) |
| `006_remove_weather_duplication.sql` | ↑ (merge into) |

**New File:** `010_mart_analytics_layer.sql`
**Purpose:** Unified mart schema with weather, advanced features, and analytics views
**Size:** ~350 lines (combined)

---

### Category 3: Data Quality & Monitoring (Consolidate to 1)
**Files: 3 → 1**

| Current Files | Consolidate To |
|---------------|----------------|
| `007_data_quality_log.sql` | `011_monitoring_infrastructure.sql` |
| `007_ab_test_monitoring.sql` | ↑ (merge into) |
| `015_dba_audit_log_table.sql` | ↑ (merge into) |

**New File:** `011_monitoring_infrastructure.sql`
**Purpose:** Complete monitoring, auditing, and data quality framework

---

### Category 4: Data Layer Enhancements (Consolidate to 2)
**Files: 7 → 2**

#### Group A: NextGen & Advanced Stats → `012_nextgen_stats.sql`
- `008_nextgen_stats.sql`
- `009_advanced_metrics.sql`

#### Group B: NFLverse Data Layer → `013_nflverse_complete.sql`
- `010_complete_nflverse_data.sql`
- `010_data_standardization_layer.sql`
- `009_odds_optimization.sql`
- `008_add_foreign_keys_simple.sql`

---

### Category 5: Team & Player Mapping (Consolidate to 1)
**Files: 3 → 1**

| Current Files | Consolidate To |
|---------------|----------------|
| `011_team_logos.sql` | `014_team_player_mappings.sql` |
| `012_consolidate_team_mappings.sql` | ↑ (merge into) |
| `016_player_id_mapping_view.sql` | ↑ (merge into) |

---

### Category 6: Database Consistency Fixes (★ HIGH PRIORITY)
**Files: 5 → 1**

**DUPLICATE PREFIX: 018 (5 versions!)**

| File | Size | Purpose | Status |
|------|------|---------|--------|
| `018_predictions_schema.sql` | 384 lines | RL prediction tracking (comprehensive) | **CANONICAL** |
| `018_pipeline_tracking.sql` | 44 lines | Basic refresh tracking | Merge features into canonical |
| `018_critical_fixes_v2.sql` | 215 lines | Team abbreviations, player IDs | Merge into canonical |
| `018_critical_fixes_v3.sql` | 222 lines | Idempotent version of v2 | Superseded by v3 |
| `018_database_consistency_fixes.sql` | 319 lines | Most comprehensive | **USE AS BASE** |

**Action:** Consolidate to `018_database_consistency_complete.sql`
**Approach:**
1. Use `018_database_consistency_fixes.sql` as base
2. Add prediction schema from `018_predictions_schema.sql`
3. Add pipeline tracking from `018_pipeline_tracking.sql`
4. Ensure idempotency

---

### Category 7: Props & Predictions Infrastructure (Consolidate to 2)
**Files: 8 → 2**

#### Group A: Props Lines → `019_props_infrastructure.sql`
- `019_prop_lines_table.sql`
- `019_v2_model_features.sql`
- `020_prop_predictions_table.sql`
- `024_bayesian_prop_predictions.sql`

#### Group B: Model Registry & ML → `025_ml_model_registry.sql`
- `025_model_registry.sql`
- `032_ml_predictions.sql`

---

### Category 8: Duplicate 026 Series (★ HIGH PRIORITY)
**Files: 4 → 2**

**DUPLICATE PREFIX: 026 (4 versions!)**

| File | Purpose | Action |
|------|---------|--------|
| `026_canonical_training_window.sql` | ML model training windows | **KEEP** as `026_model_training_config.sql` |
| `026_add_nflreadr_alignment_columns.sql` | Schema alignment | **MERGE** into 026 |
| `026_paper_trading_schema.sql` | Paper trading system | **KEEP SEPARATE** as `027_paper_trading.sql` |
| `026_pinnacle_exchange_fields.sql` | Bookmaker fields | **MERGE** into `019_props_infrastructure.sql` |

**Consolidated:**
- `026_model_training_config.sql` (training windows + schema alignment)
- `027_paper_trading.sql` (renamed from 026)

---

### Category 9: Data Column Additions (Consolidate to 2)
**Files: 6 → 2**

#### Phase 4-5 Features → `030_advanced_feature_columns.sql`
- `027_add_critical_columns.sql`
- `028_add_defensive_player_columns.sql`
- `029_add_situational_drive_columns.sql`
- `030_add_phase4_critical_columns.sql`
- `031_add_phase5_high_signal_columns.sql`

#### Model Infrastructure → `031_model_infrastructure.sql`
- `033_fix_convergence_status.sql`
- `043_column_lineage_tracking.sql`
- `044_add_ensemble_tracking_columns.sql`
- `045_api_keys_table.sql`
- `046_ensemble_version_tracking.sql`

---

### Category 10: Duplicate 048 Series (★ CRITICAL)
**Files: 5 → 1**

**DUPLICATE PREFIX: 048 (5 versions!)**

| File | Size | Purpose | Status |
|------|------|---------|--------|
| `048_create_game_predictions.sql` | 560 lines | Complete game predictions infra | **CANONICAL** |
| `048_create_game_predictions_ADDENDUM.sql` | 526 lines | Adds missing columns/views | Features merged into main |
| `048_create_views_compatibility_patch.sql` | 160 lines | View compatibility layer | Superseded |
| `048_fix_thursday_night_flag.sql` | 130 lines | TNF flag population | **CRITICAL - KEEP SEPARATE** |
| `048_prediction_accuracy_view.sql` | 138 lines | Accuracy tracking view | Merge into main |

**Action:**
- **Main:** `048_game_predictions_complete.sql` (merge main + addendum + accuracy view)
- **Separate:** `049_thursday_night_fix.sql` (data update, not schema)

---

### Category 11: Duplicate 050 Series
**Files: 2 → 1**

| File | Purpose | Action |
|------|---------|--------|
| `050_add_coaching_staff_tables.sql` | Coaching data | **KEEP** as `050_coaching_staff.sql` |
| `050_dashboard_performance_indexes.sql` | Performance indexes | **MERGE** into `050_coaching_staff.sql` |

---

### Category 12: Advanced Infrastructure (Keep Mostly Separate)
**Files: 7 → 5**

| Current | Keep/Consolidate |
|---------|------------------|
| `047_r_model_predictions_lookup.sql` | **KEEP** |
| `049_adaptive_learning_tracking.sql` | **KEEP** |
| `051_game_outcome_predictions_history.sql` | **KEEP** |
| `100_schema_reconciliation.sql` | **MERGE** with 018 series |
| `101_add_foreign_key_constraints.sql` | **KEEP** |
| `101_stat_type_performance.sql` | **MERGE** → `101_analytics_tracking.sql` |
| `102_causal_feature_tables.sql` | **KEEP** |
| `102_dirichlet_alpha_by_stat_type.sql` | **MERGE** → `102_dirichlet_optimization.sql` |

**Consolidate:**
- `101_analytics_tracking.sql` (foreign keys + stat performance)
- `102_dirichlet_optimization.sql` (causal tables + alpha config)

---

### Category 13: Special Cases
**Files: 4**

| File | Action | Rationale |
|------|--------|-----------|
| `103_bayesian_update_log.sql` | **KEEP** | Latest Bayesian infrastructure |
| `1000_international_games_table.sql` | **KEEP** | Future feature (numbering suggests post-v1) |
| `add_international_venues.sql` | **MERGE** into 1000 | Related content |
| `verify_schema.sql` | **KEEP** | Utility script |

---

## Proposed Final Migration Sequence

### Tier 1: Foundation (3 files)
1. `000_complete_schema.sql` - Complete initial schema
2. `001_init.sql` - Historical init
3. `002_timescale.sql` - TimescaleDB extension

### Tier 2: Core Infrastructure (5 files)
4. `010_mart_analytics_layer.sql` - Mart schema + weather + features
5. `011_monitoring_infrastructure.sql` - DQ + auditing + monitoring
6. `012_nextgen_stats.sql` - NextGen + advanced metrics
7. `013_nflverse_complete.sql` - NFLverse + standardization + odds
8. `014_team_player_mappings.sql` - Team/player reference data

### Tier 3: Critical Fixes (1 file)
9. **`018_database_consistency_complete.sql`** - Team abbr + player IDs + predictions schema + pipeline tracking

### Tier 4: Props & ML Infrastructure (2 files)
10. `019_props_infrastructure.sql` - Props lines + Bayesian predictions + Pinnacle
11. `025_ml_model_registry.sql` - Model registry + ML predictions

### Tier 5: Model Training & Trading (2 files)
12. `026_model_training_config.sql` - Training windows + nflreadr alignment
13. `027_paper_trading.sql` - Paper trading system

### Tier 6: Feature Engineering (2 files)
14. `030_advanced_feature_columns.sql` - Phase 4-5 features + critical columns
15. `031_model_infrastructure.sql` - Convergence + lineage + ensemble tracking + API keys

### Tier 7: Game Predictions (2 files)
16. **`048_game_predictions_complete.sql`** - Complete predictions infrastructure
17. `049_thursday_night_fix.sql` - TNF flag data update

### Tier 8: Additional Infrastructure (5 files)
18. `047_r_model_predictions_lookup.sql` - R model integration
19. `049_adaptive_learning_tracking.sql` - Adaptive learning
20. `050_coaching_staff.sql` - Coaching + dashboard indexes
21. `051_game_outcome_predictions_history.sql` - Prediction history

### Tier 9: Analytics & Optimization (3 files)
22. `101_analytics_tracking.sql` - Foreign keys + stat performance
23. `102_dirichlet_optimization.sql` - Causal features + Dirichlet alpha
24. `103_bayesian_update_log.sql` - Bayesian update logging

### Tier 10: Future Features (3 files)
25. `1000_international_games.sql` - International games + venues
26. `verify_schema.sql` - Schema validation utility

**TOTAL: 26 canonical migrations** (down from 70)

---

## Duplicate Resolution Matrix

### Critical Duplicates

| Prefix | Count | Files | Resolution |
|--------|-------|-------|------------|
| **018** | 5 | predictions_schema, pipeline_tracking, critical_fixes_v2/v3, database_consistency | Consolidate to `018_database_consistency_complete.sql` |
| **026** | 4 | training_window, nflreadr_alignment, paper_trading, pinnacle | Split: 026_model_training_config + 027_paper_trading |
| **048** | 5 | create_predictions, ADDENDUM, compatibility_patch, thursday_fix, accuracy_view | Split: 048_complete + 049_thursday_fix |

### Minor Duplicates

| Prefix | Count | Resolution |
|--------|-------|------------|
| **050** | 2 | Merge coaching + dashboard indexes → `050_coaching_staff.sql` |
| **101** | 2 | Merge foreign keys + stat performance → `101_analytics_tracking.sql` |
| **102** | 2 | Merge causal + dirichlet → `102_dirichlet_optimization.sql` |

---

## Risk Assessment & Mitigation

### High Risk Areas

#### 1. **018 Series Consolidation** (Highest Risk)
**Risk:** Multiple schema fixes applied in production may have order dependencies
**Mitigation:**
- Use `018_database_consistency_fixes.sql` as base (most comprehensive)
- Add idempotency guards from v3
- Test against current production schema
- Create rollback script

#### 2. **048 Series Consolidation** (High Risk)
**Risk:** Game predictions table has evolved through 5 migrations
**Mitigation:**
- Combine main + ADDENDUM first
- Verify all columns present in final schema
- Test triggers and functions
- Keep thursday_night_fix separate (data update)

#### 3. **Mart Schema Dependencies** (Medium Risk)
**Risk:** Materialized views may have refresh order dependencies
**Mitigation:**
- Document refresh order in consolidation
- Add comments for view dependencies
- Test `REFRESH MATERIALIZED VIEW CONCURRENTLY`

### Low Risk Areas

- Category 1 (Foundation): No consolidation = no risk
- Category 13 (Special cases): Mostly kept separate
- Number-only migrations (100, 101, 102, 103): Clear sequence

---

## Implementation Strategy

### Phase 1: Planning & Documentation (Current)
- ✅ Complete inventory
- ✅ Identify duplicates
- ✅ Create consolidation plan
- ⏳ Get stakeholder approval

### Phase 2: Consolidation Script Creation (Week 1)
1. Create new consolidated migration files
2. Add comprehensive comments
3. Include "Consolidates:" header listing source files
4. Test idempotency

### Phase 3: Testing (Week 2)
1. Test on empty database (fresh init)
2. Test on production-like backup
3. Verify schema matches current production
4. Document any discrepancies

### Phase 4: Validation (Week 3)
1. SQL validator checks
2. Schema comparison tool
3. Cross-reference with `DATABASE_QUICK_REFERENCE.md`
4. Update documentation

### Phase 5: Cleanup (Week 4)
1. Move old migrations to `db/migrations/archive/`
2. Update migration tracking
3. Update `CLAUDE.md` references
4. Create migration guide

---

## Consolidation Rules

### Include in Header
```sql
-- ============================================================
-- Migration XXX: [Descriptive Name]
-- Consolidates: [list of original migrations]
-- Created: [Date]
-- Purpose: [Clear purpose statement]
-- Dependencies: [Required prior migrations]
-- Safe to run multiple times: [YES/NO]
-- ============================================================
```

### Idempotency Patterns
- `CREATE TABLE IF NOT EXISTS`
- `ALTER TABLE ADD COLUMN IF NOT EXISTS`
- `DROP ... IF EXISTS` before `CREATE OR REPLACE`
- `ON CONFLICT DO UPDATE` for data inserts
- Check for existence before operations

### Testing Requirements
Each consolidated migration must:
1. Run successfully on empty database
2. Run successfully when re-run (idempotent)
3. Produce identical schema to original sequence
4. Pass SQL validator
5. Have rollback script

---

## Open Questions for Stakeholder

1. **Backward Compatibility:** Do we need to maintain migration history for audit purposes?
   - **Option A:** Archive old migrations, new deployments use consolidated
   - **Option B:** Keep both, mark old as deprecated
   - **Recommendation:** Option A (cleaner)

2. **Migration Numbering:**
   - Keep current numbers (018, 026, 048) or renumber sequentially (010-035)?
   - **Recommendation:** Keep current numbers for traceability

3. **Deployment Timeline:**
   - Test in dev → staging → production over 2-4 weeks?
   - **Recommendation:** 3-week phased rollout

4. **Rollback Strategy:**
   - Create explicit rollback migrations or rely on backup/restore?
   - **Recommendation:** Explicit rollbacks for critical migrations (018, 048)

---

## Estimated Effort

| Phase | Estimated Hours | Dependencies |
|-------|----------------|--------------|
| Planning (Current) | 4 hours | Complete ✅ |
| Script Creation | 16 hours | Approval needed |
| Testing | 12 hours | Scripts complete |
| Documentation | 6 hours | Testing complete |
| Deployment | 4 hours | Validation complete |
| **TOTAL** | **42 hours** | ~1 week FTE |

---

## Success Criteria

- ✅ Reduce migration count from 70 to ≤30
- ✅ Zero duplicate prefixes (018, 026, 048, 050, 101, 102)
- ✅ All migrations idempotent
- ✅ 100% schema parity with current production
- ✅ All tests passing
- ✅ Documentation updated
- ✅ Rollback scripts created

---

## Appendix A: File-by-File Mapping

### Files to Archive (40 files)

**Category 2-3 Merges (7 files):**
- `003_mart_game_weather.sql` → 010
- `004_advanced_features.sql` → 010
- `005_enhance_mart_views.sql` → 010
- `006_remove_weather_duplication.sql` → 010
- `007_data_quality_log.sql` → 011
- `007_ab_test_monitoring.sql` → 011
- `015_dba_audit_log_table.sql` → 011

**Category 4 Merges (5 files):**
- `008_add_foreign_keys_simple.sql` → 013
- `009_odds_optimization.sql` → 013
- `010_data_standardization_layer.sql` → 013

**Category 5 Merges (3 files):**
- `011_team_logos.sql` → 014
- `012_consolidate_team_mappings.sql` → 014
- `016_player_id_mapping_view.sql` → 014

**Category 6: 018 Series (4 of 5 files):**
- `018_pipeline_tracking.sql` → 018_complete
- `018_critical_fixes_v2.sql` → 018_complete
- `018_critical_fixes_v3.sql` → 018_complete
- (Keep `018_predictions_schema.sql` as source for merge)

**Category 7 Merges (6 files):**
- `019_v2_model_features.sql` → 019
- `020_prop_predictions_table.sql` → 019
- `024_bayesian_prop_predictions.sql` → 019
- `032_ml_predictions.sql` → 025

**Category 8: 026 Series (2 files):**
- `026_pinnacle_exchange_fields.sql` → 019
- (Keep others for renumbering)

**Category 9 Merges (11 files):**
- `027_add_critical_columns.sql` → 030
- `028_add_defensive_player_columns.sql` → 030
- `029_add_situational_drive_columns.sql` → 030
- `031_add_phase5_high_signal_columns.sql` → 030
- `033_fix_convergence_status.sql` → 031
- `043_column_lineage_tracking.sql` → 031
- `044_add_ensemble_tracking_columns.sql` → 031
- `045_api_keys_table.sql` → 031
- `046_ensemble_version_tracking.sql` → 031

**Category 10: 048 Series (4 of 5 files):**
- `048_create_game_predictions_ADDENDUM.sql` → 048_complete
- `048_create_views_compatibility_patch.sql` → 048_complete
- `048_prediction_accuracy_view.sql` → 048_complete
- (Keep thursday_night_fix separate)

**Category 11 Merge (1 file):**
- `050_dashboard_performance_indexes.sql` → 050

**Category 12 Merges (3 files):**
- `100_schema_reconciliation.sql` → merge with 018
- `101_add_foreign_key_constraints.sql` → 101
- `102_causal_feature_tables.sql` → 102

**Special Cases (1 file):**
- `add_international_venues.sql` → 1000

### Files to Keep/Rename (30 files)

See "Proposed Final Migration Sequence" section above.

---

## Appendix B: Known Schema Issues

Based on analysis, the following schema issues were discovered across migrations:

1. **Team Abbreviation Inconsistency** (018 series)
   - LA vs LAR for Rams
   - WSH vs WAS for Commanders
   - **Fixed in:** 018_database_consistency_fixes.sql

2. **Player ID Fragmentation** (018 series)
   - Multiple ID systems: player_id, gsis_id, pfr_id, espn_id
   - **Fixed in:** player_id_mapping and lookup functions

3. **Missing Primary Keys** (018 series)
   - `mart.rolling_features` had no PK
   - `mart.bayesian_team_ratings` had no PK
   - **Fixed in:** 018 series

4. **Thursday Night Flag** (048 series)
   - Column existed but was always FALSE
   - **Fixed in:** 048_fix_thursday_night_flag.sql

5. **Duplicate Tables** (100 series)
   - `rosters` vs `rosters_weekly`
   - **Fixed in:** 100_schema_reconciliation.sql (deprecation notice)

6. **Empty Mart Tables** (100 series)
   - `mart.team_epa` was empty
   - `mart.team_playoff_context` was empty
   - **Fixed in:** 100_schema_reconciliation.sql

---

## Appendix C: Dependencies Graph

```
000_complete_schema.sql (foundation)
  ↓
001_init.sql, 002_timescale.sql (extensions)
  ↓
010_mart_analytics_layer.sql (mart schema)
  ↓
011_monitoring_infrastructure.sql (DQ)
  ↓
012_nextgen_stats.sql, 013_nflverse_complete.sql (data layer)
  ↓
014_team_player_mappings.sql (reference data)
  ↓
018_database_consistency_complete.sql (fixes + predictions)
  ↓
019_props_infrastructure.sql, 025_ml_model_registry.sql (ML)
  ↓
026_model_training_config.sql, 027_paper_trading.sql (training)
  ↓
030_advanced_feature_columns.sql, 031_model_infrastructure.sql (features)
  ↓
048_game_predictions_complete.sql (game predictions)
  ↓
[remaining infrastructure migrations]
```

---

## Next Steps

1. **Stakeholder Review:** Review this plan and answer open questions
2. **Approval:** Get formal approval to proceed
3. **Create Consolidation Branch:** `git checkout -b consolidate-migrations`
4. **Begin Script Creation:** Start with low-risk consolidations (Categories 2-5)
5. **Incremental Testing:** Test each consolidated migration as it's created
6. **Documentation Update:** Update all references to migration numbers

---

**Document Status:** DRAFT - Awaiting Stakeholder Approval
**Last Updated:** December 10, 2025
**Maintained By:** Legacy System Modernization Expert
