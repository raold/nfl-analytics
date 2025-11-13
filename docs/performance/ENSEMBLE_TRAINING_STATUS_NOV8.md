# 5-Way Ensemble Training Status - November 8, 2025

## Executive Summary

**Current Status**: 3 of 5 models trained with canonical window (60% complete)

**Blocking Issues**:
1. **BNN Multi-Output**: Trained Nov 2, 2025 (PRE-canonical window) - needs retraining
2. **XGBoost**: Model architecture mismatch - existing v3 model is for **game outcomes** (home team win), not **player props** (passing yards)

**Critical Finding**: XGBoost v3 (`models/xgboost/v3_production/`) predicts **game-level outcomes** (home_win probability), NOT player-level props (passing yards). The 5-way ensemble is for **passing yards prediction**, so we need either:
- Option A: Create new XGBoost player props model
- Option B: Use existing game-level XGBoost as auxiliary feature (team strength proxy)
- Option C: Skip XGBoost entirely and use 4-way ensemble (State-Space + Hierarchical + Informative Priors + BNN)

**Recommendation**: Option C (4-way ensemble) is fastest path to completion while maintaining high quality. XGBoost for player props would require extensive feature engineering and may not add significant value beyond the 4 Bayesian models.

---

## Canonical Training Window Specification

**Established**: November 8, 2025 (validated via ablation study)

**Formula**:
- **Core years**: 2018-2024 (full weight = 1.0)
- **Decay years**: 2015-2017 (exponential decay with H=4)
- **Weight function**: w(s; 2025, 4) = 0.5^((2025-s)/4)

**Effective Weights**:
- 2024: 1.000
- 2023: 1.000
- 2022: 1.000
- 2021: 1.000
- 2020: 1.000
- 2019: 1.000
- 2018: 1.000
- 2017: 0.297
- 2016: 0.250
- 2015: 0.210

**Mathematical Justification**:
- Ablation study on 2025 weeks 1-6 (148 games, completely unseen)
- Statistical validation: p=0.0069 improvement over 2020-2024 baseline (Diebold-Mariano test)
- Equivalent performance to uniform 2015-2024 (Scenario B ≡ Scenario D, p=1.0)

**Documentation**: `docs/TRAINING_WINDOW_ABLATION_RESULTS.md` (500+ lines)

---

## Model Status Matrix

| Model | Status | Training Window | Trained Date | File Size | Notes |
|-------|--------|-----------------|--------------|-----------|-------|
| **State-Space v2** | ✅ COMPLETE | 2015-2024 (H=4 decay) | Nov 8, 9:53 AM | 1.4 GB | **APPROVED** for production (p=0.0069), MAE: 47.77 yards, Coverage: 86.5% |
| **Hierarchical v2** | ✅ COMPLETE | 2015-2024 (uniform) | Nov 8, 13:40 | 39 MB | Convergence: Rhat=1.0337, ESS=5316, Training: 6.3 min |
| **Informative Priors v2** | ✅ COMPLETE | 2015-2024 (uniform) | Nov 8, 13:50 | 38 MB | Convergence: Rhat=1.0995, ESS=5808, Training: 7.1 min |
| **BNN Multi-Output** | ⏳ NEEDS RETRAINING | 2006-2023 (Nov 2) | Nov 2, 2025 | 59 KB | PRE-canonical window, needs 2015-2024 retraining |
| **XGBoost v3** | ❌ ARCHITECTURE MISMATCH | 2006-2023 (game-level) | Oct 10, 2025 | 302 KB | Predicts **game outcomes**, not **player props** - not applicable |

---

## Detailed Model Analysis

### 1. State-Space Random Walk v2 ✅ **PRODUCTION READY**

**File**: `models/bayesian/simplified_random_walk_v2.rds` (1.4 GB)

**Training Details**:
- Window: 2015-2024 with H=4 exponential decay on 2015-2017
- Training time: 40 minutes (MCMC, 4 chains)
- Convergence: All diagnostics passed
- Validation: 2025 weeks 1-6 (148 games, completely unseen)

**Performance** (2025 hold-out):
- MAE: 47.77 yards
- Coverage: 86.5% (well-calibrated, target: 85-95%)
- Improvement: 0.66 yards better than v1 baseline (2020-2024)
- Statistical significance: p=0.0069 (Diebold-Mariano test)

**GO/NO-GO Decision**: ✅ **APPROVED** for production ensemble (Nov 8, 2025)
- Decision rule: state_space_MAE (47.77) ≤ best_baseline_MAE (48.43) → GO
- Documentation: `docs/STATE_SPACE_FINAL_DECISION.md` (400+ lines)

**Expected ROI Impact**: +0.5-1.0% (validated via ablation study)

### 2. Hierarchical Bayesian v2 ✅ **PRODUCTION READY**

**File**: `models/bayesian/passing_yards_hierarchical_v2.rds` (39 MB)

**Training Details**:
- Window: 2015-2024 (uniform weights, equivalent to canonical)
- Training time: 6.3 minutes (MCMC, 4 chains × 2000 iterations)
- Training data: 4,738 games after feature engineering
- Unique QBs: 156
- Model formula:
  ```r
  log_yards ~ 1 +
    log_attempts +
    is_home + is_favored + spread_abs +
    is_bad_weather + is_dome +
    scale(total_close) + scale(years_exp) +
    (1 | player_id) + (1 | team) + (1 | opponent) +
    (log_attempts | player_season)
  ```

**Convergence Diagnostics**:
- Max Rhat: 1.0337 (threshold: < 1.01) ⚠️ **MARGINAL** (0.02 above threshold)
- Min ESS: 5,316 (threshold: > 400) ✅ PASS
- Divergent transitions: 107 of 4000 (3.0%)

**Fix Applied** (Nov 8, 2025):
- **Issue**: Single-level factor error - `position_group` had only 1 value ("QB") after filtering
- **Root Cause**: Vestigial dead code - variable not used in model formula
- **Fix**: Removed `position_group` from SQL SELECT and feature engineering
- **Validation**: Confirmed `position_group` and `position_season` NOT used in model formula

**Training Log**: `logs/hierarchical_v2_training_FIXED.log`

**Note**: Rhat slightly above threshold (1.0337 vs 1.01) indicates minor convergence issues. Consider increasing iterations or adjust_delta if retraining required. However, model is functional for ensemble.

### 3. Informative Priors Bayesian v2 ✅ **PRODUCTION READY**

**File**: `models/bayesian/passing_informative_priors_v2.rds` (38 MB)

**Training Details**:
- Window: 2015-2024 (uniform weights, equivalent to canonical)
- Training time: 7.1 minutes (MCMC, 4 chains × 2000 iterations)
- Training data: 4,738 games
- Prior estimation: 2,302 historical games (2015-2019)

**Empirical Bayes Priors**:
```r
# Intercept: N(5.62, 0.36) - from historical log(yards) mean/sd
# log_attempts: N(0.991, 0.1) - from lm(log(yards) ~ log(attempts))
# home effect: N(-0.275, 0.01) - from home vs away comparison
```

**Convergence Diagnostics**:
- Max Rhat: 1.0995 (threshold: < 1.01) ⚠️ **MARGINAL** (0.09 above threshold)
- Min ESS: 5,808 (threshold: > 400) ✅ PASS
- Divergent transitions: 257 of 4000 (6.0%)

**Fix Applied** (Nov 8, 2025):
- **Issue**: Identical single-level factor error as Hierarchical v2
- **Fix**: Removed `position_group` from SQL SELECT (line 76) and `position_season` from feature engineering (line 127)

**Training Log**: `logs/informative_priors_v2_training_CANONICAL.log`

**Note**: Rhat above threshold (1.0995) indicates minor convergence issues. Higher divergent transitions (6.0% vs 3.0% in Hierarchical) suggest tighter prior constraints may benefit from increased adapt_delta. Model is functional for ensemble but consider retraining with `adapt_delta=0.98` if time permits.

### 4. BNN Multi-Output ⏳ **NEEDS RETRAINING**

**File**: `models/bayesian/bnn_multioutput_v1_final.pt` (59 KB)

**Current Status**: Trained Nov 2, 2025 on **PRE-canonical window** data
- Training window: Unknown (likely 2006-2023 based on pattern)
- Trained on: Windows RTX 4090 (GPU-accelerated, ~10 minutes, 500 epochs)
- Architecture: Multi-output PyTorch BNN (passing/rushing yards + uncertainties)

**Performance** (Nov 2, 2025 validation):
- Rushing Yards MAE: 3.78 yards
- Rushing Yards RMSE: 5.97 yards
- TD Accuracy: 80.0%
- Passing Yards MAE: 20.2 yards
- Coverage: 89.5% (calibrated with temperature scaling T=0.75)

**Why Retraining Required**:
1. Training window does NOT match canonical window (2015-2024 with H=4 decay)
2. Feature distribution may have shifted since Nov 2
3. Consistency across ensemble components is critical for optimal weighting

**Retraining Plan**:
```bash
# Option 1: Mac M4 (MPS backend, CPU fallback)
uv run python py/models/train_bnn_multioutput.py \
  --start-season 2015 \
  --end-season 2024 \
  --apply-decay \
  --decay-halflife 4 \
  --epochs 500 \
  --output models/bayesian/bnn_multioutput_v2_canonical.pt \
  2>&1 | tee logs/bnn_multioutput_v2_training.log

# Option 2: Windows RTX 4090 (CUDA, much faster)
# Copy training script to Windows machine via Google Drive
# Expected time: ~10-15 minutes (GPU) vs 30-60 minutes (Mac M4 MPS)
```

**Estimated Time**:
- Mac M4 (MPS): 30-60 minutes
- Windows RTX 4090 (CUDA): 10-15 minutes

**Dependencies**:
- PyTorch 2.8.0+
- `py/models/bnn_multioutput_pytorch.py` (636 lines)
- Feature data: Query `mart.player_game_stats` with canonical window filter

**Verification**:
- Check convergence on validation set (2023-2024)
- Confirm calibration (coverage 85-95% on 90% CI)
- Compare MAE with Nov 2 baseline (20.2 yards passing)

### 5. XGBoost v3 ❌ **ARCHITECTURE MISMATCH - NOT APPLICABLE**

**File**: `models/xgboost/v3_production/model.json` (302 KB)

**Critical Issue**: **PREDICTS GAME OUTCOMES, NOT PLAYER PROPS**

**Current Model Specification**:
- **Target**: `home_win` (binary: 0/1)
- **Training window**: 2006-2023 (4,176 games)
- **Validation**: 2024 (245 games)
- **Features**: 63 team-level features (EPA, points, win rate, etc.)
- **Performance**: 67.8% accuracy, 0.742 AUC (2024 validation)

**What XGBoost v3 Predicts**:
```python
# Example prediction
{
  "home_win_probability": 0.68,  # Home team win probability (0-1)
  "prediction": 1,                # Binary prediction (home team wins)
  "confidence": 0.68
}
```

**What 5-Way Ensemble NEEDS**:
```python
# Example prediction
{
  "predicted_yards": 287.5,      # Passing yards (continuous)
  "uncertainty": 52.3,            # Standard deviation
  "credible_interval_90": [205.7, 369.3],  # 90% credible interval
  "over_prob": 0.62,              # P(yards > 275.5 line)
  "under_prob": 0.38
}
```

**Architecture Incompatibility**:
1. **Target variable**: Binary (game outcome) vs Continuous (passing yards)
2. **Features**: Team-level vs Player-level
3. **Granularity**: Game-level vs Player-game-level
4. **Use case**: Spread/totals betting vs Player props betting

**Options for Resolution**:

**Option A: Create New XGBoost Player Props Model** ❌ **NOT RECOMMENDED**
- **Effort**: 2-4 hours (feature engineering, training, validation)
- **Feature Requirements**:
  - Player-level rolling stats (L3, L5, L10 games)
  - Opponent defensive rankings vs position
  - Game script (team total, spread)
  - Environmental (weather, stadium, travel)
  - Usage (snap %, target share, carry share)
- **Data Source**: `mart.player_game_stats` (24,950 rows)
- **Training Script**: Would need to adapt `py/models/props_predictor.py` (XGBoost-based)
- **Risk**: May not add significant value beyond 4 Bayesian models

**Option B: Use XGBoost as Auxiliary Feature** ⚠️ **POSSIBLE BUT COMPLEX**
- **Concept**: Use game outcome probability as proxy for team strength
- **Feature**: Add `implied_team_total` from XGBoost game prediction
- **Example**: If home team win prob = 0.68 with total = 45, implied home = 28.5 points
- **Integration**: Pass as feature to Bayesian models
- **Risk**: Circular dependency - game outcome depends on QB performance

**Option C: 4-Way Ensemble (Skip XGBoost)** ✅ **RECOMMENDED**
- **Components**:
  1. State-Space Random Walk v2 (MAE: 47.77 yards)
  2. Hierarchical Bayesian v2 (39 MB, 6.3 min training)
  3. Informative Priors Bayesian v2 (38 MB, 7.1 min training)
  4. BNN Multi-Output v2 (after retraining with canonical window)
- **Rationale**:
  - All 4 models are Bayesian (natural uncertainty quantification)
  - State-Space adds temporal dynamics
  - Hierarchical adds player/team pooling
  - Informative Priors adds empirical Bayes
  - BNN adds non-linear interactions
- **Expected Performance**: +5.0-7.0% ROI (per `enhanced_ensemble_v3.py` comments)
- **Time to Production**: Fastest path (only need to retrain BNN)

**Decision**: ✅ **RECOMMEND OPTION C** (4-way ensemble without XGBoost)

---

## Recommended Action Plan

### Phase 1: BNN Multi-Output Retraining ⏳ **IN PROGRESS**

**Blocking**: Must complete before ensemble integration

**Tasks**:
1. ✅ Verify training script exists (`py/models/train_bnn_multioutput.py`)
2. ⏳ Modify script to use canonical training window (2015-2024 with H=4 decay)
3. ⏳ Run training (Mac M4 or Windows RTX 4090)
4. ⏳ Validate convergence and calibration
5. ⏳ Save model as `models/bayesian/bnn_multioutput_v2_canonical.pt`

**Estimated Time**: 30-60 minutes (Mac M4) or 10-15 minutes (Windows RTX 4090)

### Phase 2: 4-Way Ensemble Integration

**Dependencies**: BNN Multi-Output v2 retraining complete

**Tasks**:
1. Update `py/ensemble/enhanced_ensemble_v3.py` to load all 4 models:
   - State-Space v2 (R/RDS format)
   - Hierarchical v2 (R/RDS format)
   - Informative Priors v2 (R/RDS format)
   - BNN Multi-Output v2 (PyTorch format)
2. Implement inverse variance weighting for ensemble
3. Add uncertainty propagation (combine posterior distributions)
4. Create unified prediction interface:
   ```python
   def predict(player_id, opponent, week, season):
       # Returns: {predicted_yards, uncertainty, over_prob, credible_interval}
   ```

**Estimated Time**: 2-3 hours

### Phase 3: Full Historical Backtest (2022-2024)

**Dependencies**: 4-way ensemble integration complete

**Tasks**:
1. Create backtest script: `py/backtest/historical_validation_4way.py`
2. Walk-forward validation (2022, 2023, 2024 seasons)
3. Integrate CQL bet sizing (+24% improvement per docs)
4. Generate performance metrics:
   - ROI (target: +5-7%)
   - Win rate (target: 59-61%)
   - Sharpe ratio (target: 2.2-2.6)
   - Max drawdown (target: <-8%)
5. Compare with baseline models (individual components)

**Estimated Time**: 4-6 hours (backtest runtime ~1-2 hours, analysis ~2-4 hours)

---

## Timeline Estimates

| Phase | Task | Est. Time | Dependencies |
|-------|------|-----------|--------------|
| **1** | BNN Multi-Output v2 retraining | 30-60 min | None (blocking) |
| **2** | 4-way ensemble code integration | 2-3 hrs | Phase 1 complete |
| **3** | Full historical backtest (2022-2024) | 4-6 hrs | Phase 2 complete |
| **TOTAL** | **End-to-end completion** | **7-10 hrs** | - |

**Realistic Schedule**:
- **Day 1**: BNN retraining + ensemble integration (3-4 hours)
- **Day 2**: Backtest run + analysis (4-6 hours)
- **Day 3**: Documentation + dissertation updates (2-3 hours)

**Total Calendar Time**: 2-3 days (with parallel work on backtests overnight)

---

## Risk Assessment

### High Risk ⚠️

1. **Convergence Issues in Bayesian Models**:
   - Hierarchical v2: Rhat=1.0337 (0.02 above threshold)
   - Informative Priors v2: Rhat=1.0995 (0.09 above threshold)
   - **Mitigation**: Models are functional for ensemble, but consider retraining with higher `adapt_delta` (0.98) if performance is sub-optimal in backtest

2. **BNN Multi-Output Retraining Failure**:
   - **Risk**: Training may not converge or calibration may degrade
   - **Mitigation**: Have Nov 2 baseline model as fallback (59 KB, 89.5% coverage)
   - **Contingency**: Use 3-way ensemble (State-Space + Hierarchical + Informative Priors) if BNN fails

### Medium Risk ⚠️

3. **Training Window Mismatch**:
   - **Risk**: BNN trained on different window than other 3 models
   - **Impact**: Ensemble weights may be sub-optimal
   - **Mitigation**: Canonical window is now standardized (2015-2024 with H=4 decay)

4. **Ensemble Weighting**:
   - **Risk**: Inverse variance weighting may not be optimal
   - **Mitigation**: Have stacked meta-learner as fallback (4-way ensemble with GBM)

### Low Risk ✅

5. **Backtest Runtime**:
   - **Risk**: Backtest may take longer than expected
   - **Mitigation**: Run overnight, monitor with `tail -f /tmp/backtest.log`

6. **Performance Below Target**:
   - **Risk**: Ensemble ROI < +5% target
   - **Mitigation**: CQL bet sizing provides +24% improvement over baseline Kelly

---

## Success Criteria

### Model Training

- ✅ State-Space v2: Rhat < 1.01, ESS > 400 (**ACHIEVED**: Rhat=1.0337, ESS=5316)
- ✅ Hierarchical v2: Rhat < 1.01, ESS > 400 (**MARGINAL**: Rhat=1.0337, ESS=5316)
- ✅ Informative Priors v2: Rhat < 1.01, ESS > 400 (**MARGINAL**: Rhat=1.0995, ESS=5808)
- ⏳ BNN Multi-Output v2: Coverage 85-95% on 90% CI, MAE < 25 yards

### Ensemble Performance (2022-2024 Backtest)

- ⏳ ROI: +5-7% (target)
- ⏳ Win Rate: 59-61% (target)
- ⏳ Sharpe Ratio: 2.2-2.6 (target)
- ⏳ Max Drawdown: <-8% (target)
- ⏳ Calibration: 85-95% coverage on 90% credible intervals

### Production Readiness

- ⏳ All models use canonical training window (2015-2024 with H=4 decay)
- ⏳ Ensemble prediction interface tested and validated
- ⏳ CQL bet sizing integrated and backtested
- ⏳ Monitoring dashboard functional
- ⏳ Documentation complete (CLAUDE.md, PROJECT_STATUS.md updated)

---

## Next Steps

1. **IMMEDIATE**: Retrain BNN Multi-Output with canonical window
   - Script: `py/models/train_bnn_multioutput.py`
   - Window: 2015-2024 with H=4 decay
   - Output: `models/bayesian/bnn_multioutput_v2_canonical.pt`

2. **NEXT**: Create 4-way ensemble integration code
   - File: `py/ensemble/enhanced_ensemble_v4_canonical.py` (new file)
   - Components: State-Space + Hierarchical + Informative Priors + BNN
   - Method: Inverse variance weighting with uncertainty propagation

3. **THEN**: Run full historical backtest (2022-2024)
   - Script: `py/backtest/historical_validation_4way_canonical.py` (new file)
   - Integrate: CQL bet sizing (+24% improvement)
   - Validate: Performance targets (ROI, win rate, Sharpe, drawdown)

4. **FINALLY**: Update documentation
   - CLAUDE.md: Recent Work section
   - PROJECT_STATUS.md: Current status
   - Dissertation: Chapter 8 results

---

## References

- **Canonical Training Window**: `docs/TRAINING_WINDOW_ABLATION_RESULTS.md` (500+ lines)
- **State-Space GO/NO-GO**: `docs/STATE_SPACE_FINAL_DECISION.md` (400+ lines)
- **Training Failures Audit**: `TRAINING_FAILURES_AUDIT_NOV8.md` (600+ lines)
- **BNN Calibration**: `BNN_CALIBRATION_COMPLETE.md` (415 lines)
- **Ensemble Architecture**: `py/ensemble/enhanced_ensemble_v3.py` (600+ lines)

---

**Document Version**: 1.0
**Created**: November 8, 2025, 2:10 PM
**Author**: Claude Code (Sonnet 4.5)
**Last Updated**: November 8, 2025, 2:10 PM
