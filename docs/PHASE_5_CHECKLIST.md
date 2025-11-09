# Phase 5 Implementation Checklist

**Status**: READY FOR IMPLEMENTATION  
**Created**: November 7, 2025  
**Expected Duration**: 12 days (48-72 engineering hours)  
**Expected ROI**: +26.7% (95 HIGH-rated columns)

---

## Phase 5.1: P0 High-Impact (Days 1-3) - +12.2% ROI

### Day 1: Setup & Personnel Groupings (+5.7% ROI, 16 columns)

- [ ] **SQL Migration**
  - [ ] Create `db/migrations/026_phase5_high_impact.sql`
  - [ ] Add 16 personnel columns (offense/defense_personnel, defenders_in_box, etc.)
  - [ ] Add COMMENT ON COLUMN for each with population rates
  - [ ] Create indexes (personnel, defenders_in_box, coverage_type, JSONB GIN)
  - [ ] Test migration on dev database
  - [ ] Verify column types (TEXT, INTEGER, JSONB)

- [ ] **R Data Ingestion**
  - [ ] Update `data/ingest_pbp.R` 
  - [ ] Add personnel column mappings (offense_personnel, defense_personnel)
  - [ ] Convert player lists to JSONB (toJSON for offense/defense_players)
  - [ ] Handle NAs gracefully (pre-2018 data has no personnel)
  - [ ] Test ingestion on 2024 season (sample)
  - [ ] Verify population rates match nflreadr (92.8% for personnel)

- [ ] **Feature Engineering**
  - [ ] Create `py/features/personnel_features.py`
  - [ ] Implement `compute_personnel_features()` function
  - [ ] Add `parse_personnel()` helper (e.g., "1 RB, 1 TE, 3 WR" → dict)
  - [ ] Compute personnel_mismatch_score (-1 to +1)
  - [ ] Compute expected_rush_rate by personnel matchup
  - [ ] Add blitz_indicator (defenders_in_box > 7)
  - [ ] Add coverage_shell_2_high (5+ DBs)

- [ ] **Testing**
  - [ ] Create `tests/unit/test_personnel_features.py`
  - [ ] Test parse_personnel() with various inputs
  - [ ] Test heavy_run_personnel detection (2+ RB or 2+ TE)
  - [ ] Test personnel_mismatch_score calculation
  - [ ] Test edge cases (missing personnel, NULL defenders_in_box)

### Day 2: Return Game Metrics (+4.5% ROI, 16 columns)

- [ ] **SQL Migration** (continue 026)
  - [ ] Add 16 return game columns (return_team, return_touchdown, etc.)
  - [ ] Add kickoff returner player IDs (kickoff_returner_player_id/name)
  - [ ] Add punt returner player IDs (punt_returner_player_id/name)
  - [ ] Add kickoff quality flags (inside_twenty, in_endzone, downed, fair_catch, out_of_bounds)
  - [ ] Add punt quality flags (inside_twenty, in_endzone, downed, fair_catch, out_of_bounds, blocked)
  - [ ] Create index on return_team

- [ ] **R Data Ingestion** (update ingest_pbp.R)
  - [ ] Add return game column mappings
  - [ ] Convert booleans (as.logical for flags)
  - [ ] Handle player IDs (TEXT for returner IDs)
  - [ ] Test on 2024 kickoffs/punts
  - [ ] Verify population rates (96.9% flags, 2-4% player IDs)

- [ ] **Feature Engineering**
  - [ ] Create `py/features/return_game_features.py`
  - [ ] Implement `compute_return_features()` function
  - [ ] Compute expected_field_position (touchback vs return)
  - [ ] Compute punter_quality_rating (inside_twenty rate)
  - [ ] Compute return_danger_score (player TD rate × coverage quality)
  - [ ] Add special_teams_quality_score (team-level)

- [ ] **Testing**
  - [ ] Create `tests/unit/test_return_game_features.py`
  - [ ] Test expected_field_position calculation
  - [ ] Test punter_quality_rating (sample player data)
  - [ ] Test return_danger_score with mock returner stats
  - [ ] Test kickoff vs punt differences

### Day 3: Game Script Indicators (+2.0% ROI, 5 columns) & Integration Tests

- [ ] **SQL Migration** (continue 026)
  - [ ] Add 5 game script columns (home_wp, away_wp, home_wp_post, away_wp_post, vegas_home_wp)
  - [ ] Create index on home_wp
  - [ ] COMMIT migration 026

- [ ] **R Data Ingestion** (update ingest_pbp.R)
  - [ ] Add game script column mappings
  - [ ] Convert to numeric (as.numeric for WP values)
  - [ ] Test on 2024 data
  - [ ] Verify population rates (100% pre-play, 92.9% post-play)

- [ ] **Feature Engineering**
  - [ ] Create `py/features/game_script_features.py`
  - [ ] Implement `compute_game_script_features()` function
  - [ ] Compute wp_delta (home_wp_post - home_wp = leverage)
  - [ ] Compute game_script_category (desperate, conservative, neutral)
  - [ ] Compute market_efficiency (vegas_home_wp vs home_wp)
  - [ ] Add high_leverage indicator (WP 0.45-0.55)

- [ ] **Integration Testing**
  - [ ] Create `tests/integration/test_phase5_ingestion.py`
  - [ ] Test personnel columns populated correctly
  - [ ] Test return game columns populated
  - [ ] Test game script columns populated
  - [ ] Verify population rates match expectations
  - [ ] Check JSONB validity (offense/defense_players parse as JSON)
  - [ ] Test referential integrity (player IDs exist in mart.player_hierarchy)

- [ ] **Backtest Phase 5.1**
  - [ ] Run baseline backtest (pre-Phase 5.1 features)
  - [ ] Run Phase 5.1 backtest (with personnel, return game, game script)
  - [ ] Compare ROI improvement (expected: +12.2%)
  - [ ] Generate backtest report
  - [ ] Document results

---

## Phase 5.2: P1 Medium-Impact (Days 4-6) - +7.9% ROI

### Day 4: Tackle Details (+3.0% ROI, 14 columns)

- [ ] **SQL Migration**
  - [ ] Create `db/migrations/027_phase5_medium_impact.sql`
  - [ ] Add 14 tackle columns (tackle_with_assist, assist_tackle, etc.)
  - [ ] Add tackle_with_assist player IDs (1 and 2)
  - [ ] Add assist_tackle player IDs (2 only, 1 already exists)
  - [ ] Add tackle_for_loss_2 player IDs
  - [ ] Add solo_tackle_2 player IDs
  - [ ] Create index on tackle_with_assist

- [ ] **R Data Ingestion**
  - [ ] Update `data/ingest_pbp.R` for tackle columns
  - [ ] Convert booleans (as.logical)
  - [ ] Handle player IDs (TEXT)
  - [ ] Test on 2024 data
  - [ ] Verify population rates (96.9% flags, 0-18% player IDs)

- [ ] **Feature Engineering**
  - [ ] Create `py/features/tackle_features.py`
  - [ ] Implement `compute_tackle_features()` function
  - [ ] Compute gang_tackle_rate (% plays with 2+ tacklers)
  - [ ] Compute tfl_depth (penetration rate)
  - [ ] Compute yac_allowed (solo vs assisted differential)
  - [ ] Add player_tackling_skill (hierarchical rating)

- [ ] **Testing**
  - [ ] Create `tests/unit/test_tackle_features.py`
  - [ ] Test gang_tackle_rate calculation
  - [ ] Test tfl_depth computation
  - [ ] Test yac_allowed differential

### Day 5: Drive Details (+1.8% ROI, 7 columns) & Total EPA (+1.8% ROI, 6 columns)

- [ ] **SQL Migration** (continue 027)
  - [ ] Add 7 drive detail columns (drive_start/end_yard_line, transitions, times)
  - [ ] Add drive_real_start_time (TIMESTAMP)
  - [ ] Add drive_game_clock_start/end (TEXT)
  - [ ] Add 6 total EPA columns (total_home/away_epa, pass/rush EPA)
  - [ ] Create indexes (drive_start_transition, total_home_epa)

- [ ] **R Data Ingestion**
  - [ ] Update `data/ingest_pbp.R` for drive + EPA columns
  - [ ] Convert timestamps (as.POSIXct)
  - [ ] Convert numerics (as.numeric for EPA)
  - [ ] Test on 2024 data
  - [ ] Verify population rates (98.7-98.9% drives, 100% EPA)

- [ ] **Feature Engineering**
  - [ ] Create `py/features/drive_features.py`
  - [ ] Implement `compute_drive_features()` function
  - [ ] Compute field_position_score (OWN 25 = -25, OPP 40 = +40)
  - [ ] Compute drive_momentum (turnover → short field → high TD prob)
  - [ ] Compute seconds_per_play (clock management)
  
  - [ ] Create `py/features/total_epa_features.py`
  - [ ] Implement `compute_epa_features()` function
  - [ ] Compute epa_momentum (rolling 10-play average)
  - [ ] Compute pass_rush_balance (total_pass_epa / total_epa)
  - [ ] Compute epa_trend (slope of last 20 plays)

- [ ] **Testing**
  - [ ] Create `tests/unit/test_drive_features.py`
  - [ ] Test field_position_score parsing ("OWN 25" → -25)
  - [ ] Test drive_momentum calculation
  
  - [ ] Create `tests/unit/test_total_epa_features.py`
  - [ ] Test epa_momentum rolling window
  - [ ] Test pass_rush_balance edge cases (division by zero)

### Day 6: QB Pressure (+1.1% ROI, 4 columns) & Integration Tests

- [ ] **SQL Migration** (continue 027)
  - [ ] Add 4 QB pressure columns (was_pressure, number_of_pass_rushers, qb_hit_2)
  - [ ] Create index on was_pressure
  - [ ] COMMIT migration 027

- [ ] **R Data Ingestion**
  - [ ] Update `data/ingest_pbp.R` for pressure columns
  - [ ] Convert booleans (as.logical for was_pressure)
  - [ ] Convert integers (as.integer for number_of_pass_rushers)
  - [ ] Test on 2024 data
  - [ ] Verify population rates (92.8% pressure, 0.3-6.1% player IDs)

- [ ] **Feature Engineering**
  - [ ] Create `py/features/qb_pressure_features.py`
  - [ ] Implement `compute_pressure_features()` function
  - [ ] Compute pressure_rate (% by team/QB)
  - [ ] Compute blitz_tendency (number_of_pass_rushers > 4)
  - [ ] Compute coverage_sack (qb_hit_2 != NULL → delayed pressure)

- [ ] **Testing**
  - [ ] Create `tests/unit/test_qb_pressure_features.py`
  - [ ] Test pressure_rate calculation
  - [ ] Test blitz_tendency threshold
  
  - [ ] Update `tests/integration/test_phase5_ingestion.py`
  - [ ] Test tackle columns populated
  - [ ] Test drive columns populated
  - [ ] Test total EPA columns populated
  - [ ] Test QB pressure columns populated

- [ ] **Backtest Phase 5.2**
  - [ ] Run Phase 5.1 baseline backtest
  - [ ] Run Phase 5.2 backtest (with tackle, drive, EPA, pressure)
  - [ ] Compare ROI improvement (expected: +7.9% incremental)
  - [ ] Generate backtest report
  - [ ] Document results

---

## Phase 5.3-5.4: P2-P3 Low-Impact (Days 7-12) - +6.6% ROI

### Day 7-8: Play Timing, Environment, Kick Quality (+3.6% ROI, 13 columns)

- [ ] **SQL Migration**
  - [ ] Create `db/migrations/028_phase5_low_impact.sql`
  - [ ] Add 4 play timing columns (start_time, time_of_day, play_clock, time_to_throw)
  - [ ] Add 5 environment columns (roof, surface, stadium, stadium_id, location)
  - [ ] Add 4 kick quality columns (punter/kicker player IDs)

- [ ] **R Data Ingestion**
  - [ ] Update `data/ingest_pbp.R` for timing, environment, kick columns
  - [ ] Convert timestamps (as.POSIXct for start_time)
  - [ ] Convert numerics (as.numeric for time_to_throw)
  - [ ] Test on 2024 data
  - [ ] Verify population rates

- [ ] **Feature Engineering**
  - [ ] Create `py/features/play_timing_features.py`
  - [ ] Create `py/features/environment_features.py`
  - [ ] Create `py/features/kick_quality_features.py`

- [ ] **Testing**
  - [ ] Create unit tests for each module
  - [ ] Test timestamp parsing, environment categorization, kicker ratings

### Day 9-10: Sack, Interception, Fumble, Pass Defense, Route (+3.0% ROI, 16 columns)

- [ ] **SQL Migration** (continue 028)
  - [ ] Add 4 sack detail columns (half_sack_1/2 player IDs)
  - [ ] Add 2 interception columns (interception_player_id/name)
  - [ ] Add 3 fumble detail columns (fumble_not_forced, out_of_bounds, forced_fumble_team)
  - [ ] Add 2 pass defense columns (pass_defense_2 player IDs)
  - [ ] Add 1 route column (route)

- [ ] **R Data Ingestion**
  - [ ] Update `data/ingest_pbp.R` for remaining columns
  - [ ] Convert booleans, TEXT types
  - [ ] Test on 2024 data

- [ ] **Feature Engineering**
  - [ ] Create `py/features/sack_features.py`
  - [ ] Create `py/features/interception_features.py`
  - [ ] Create `py/features/fumble_features.py`
  - [ ] Create `py/features/pass_defense_features.py`
  - [ ] Create `py/features/route_features.py`

- [ ] **Testing**
  - [ ] Create unit tests for each module

### Day 11: Replay (+0.4% ROI, 2 columns) & Final Integration

- [ ] **SQL Migration** (continue 028)
  - [ ] Add 2 replay columns (replay_or_challenge, replay_or_challenge_result)
  - [ ] COMMIT migration 028

- [ ] **R Data Ingestion**
  - [ ] Update `data/ingest_pbp.R` for replay columns
  - [ ] Convert booleans (as.logical)
  - [ ] Test on 2024 data

- [ ] **Integration Testing**
  - [ ] Update `tests/integration/test_phase5_ingestion.py`
  - [ ] Test all Phase 5.3-5.4 columns populated
  - [ ] Verify population rates across all 95 columns
  - [ ] Check data quality (no unexpected NULLs, valid JSONB, referential integrity)

### Day 12: Full Backtest & Validation

- [ ] **Full Backtest**
  - [ ] Run pre-Phase 5 baseline (142 features)
  - [ ] Run post-Phase 5 backtest (237 features)
  - [ ] Compare ROI improvement (expected: +26.7% total)
  - [ ] Generate comprehensive backtest report
  - [ ] Validate Sharpe ratio improvement (+0.6 expected)
  - [ ] Validate win rate improvement (+6 pp expected)

- [ ] **Data Quality Validation**
  - [ ] Create `py/validation/phase5_data_quality.py`
  - [ ] Check population rates match nflreadr (±5%)
  - [ ] Validate JSONB columns parse correctly
  - [ ] Check referential integrity (all player IDs exist)
  - [ ] Verify no unexpected NULL spikes
  - [ ] Run data quality report

- [ ] **Documentation**
  - [ ] Update `CLAUDE.md` with Phase 5 completion
  - [ ] Update `PROJECT_STATUS.md` with new feature count (237)
  - [ ] Update `README.md` with Phase 5 achievements
  - [ ] Update `docs/DATABASE_SCHEMA_AUDIT.md` with new columns
  - [ ] Create `docs/milestones/PHASE_5_COMPLETE_SUMMARY.md`

---

## Post-Implementation

- [ ] **Model Retraining**
  - [ ] Retrain Bayesian hierarchical models with 237 features
  - [ ] Update XGBoost hyperparameters for expanded feature set
  - [ ] Update BNN architecture for JSONB embeddings
  - [ ] Validate model improvements match backtest expectations

- [ ] **Monitoring Setup**
  - [ ] Add Phase 5 columns to data quality dashboard
  - [ ] Set up alerts for population rate drops
  - [ ] Monitor JSONB parse errors
  - [ ] Track model performance metrics (ROI, Sharpe, win rate)

- [ ] **Rollback Preparation**
  - [ ] Create database backup: `backup_pre_phase5.dump`
  - [ ] Document rollback procedure
  - [ ] Test rollback on dev database
  - [ ] Prepare emergency revert commits

---

## Success Criteria (ALL must be met)

- [ ] ✅ All 95 HIGH-rated columns added to `plays` table
- [ ] ✅ Population rates match nflreadr expectations (±5%)
- [ ] ✅ Backtest ROI improves by **+20% minimum** (conservative vs +26.7% expected)
- [ ] ✅ No data quality regressions (NULL rates, referential integrity maintained)
- [ ] ✅ Model training time increases <25% (acceptable for +26.7% ROI)
- [ ] ✅ All unit tests pass (100% coverage for new modules)
- [ ] ✅ All integration tests pass (database population validated)
- [ ] ✅ Documentation updated (CLAUDE.md, PROJECT_STATUS.md, README.md, schema audit)

---

## Rollback Triggers (if ANY are met, consider rollback)

- [ ] ❌ Backtest ROI **decreases** or improves <10% (vs +20% expected)
- [ ] ❌ Data quality issues (>10% population rate drop, >5% unexpected NULLs)
- [ ] ❌ JSONB parse errors >0.1%
- [ ] ❌ Referential integrity violations (player IDs don't exist)
- [ ] ❌ Model training time increases >50%
- [ ] ❌ Production inference latency unacceptable (>500ms per prediction)

---

**Status**: READY FOR IMPLEMENTATION  
**Next Action**: Begin Day 1 with SQL migration 026  
**Owner**: [Assign engineer]  
**Target Completion**: [Start date] + 12 days

**Version**: 1.0  
**Last Updated**: November 7, 2025
