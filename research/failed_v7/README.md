# Ensemble v7.0 - Failed Experiment Archive

**Status**: ❌ ABANDONED (November 19, 2025)
**Reason**: Validation failure - delivered +0.01 yards vs target +2.0 yards improvement
**Decision**: ADR-009 (see `docs/adr/009-abandon-ensemble-v7-meta-learning.md`)

---

## Executive Summary

Ensemble v7.0 attempted to add context-aware meta-learning to predict optimal ensemble weights dynamically based on game conditions (QB experience, roster stability, market dispersion, etc.). After 60+ hours of implementation (Phases A, B, C) and comprehensive multi-season validation (2022-2024), the approach showed **NO meaningful improvement** over v6.0 baseline.

**Final Validation Results (Nov 18-19, 2025)**:
- **v7.0 Full Ensemble**: 40.63 MAE (2022-2024 average)
- **v6.0 Baseline**: 40.64 MAE
- **Improvement**: +0.01 yards (0.02% gain vs 5.0% target = **99.6% miss**)

---

## What Was v7.0?

### Core Innovation: Meta-Learning Weight Prediction

Instead of static weights (v6.0: State-Space 30%, Hierarchical 30%, XGBoost 25%, IP 15%), v7.0 used a **meta-model** to dynamically adjust weights per game based on:

1. **QB Experience Features** (5 features):
   - One-hot encoding: rookie, developing, veteran, elite veteran
   - Log(games played)

2. **Roster Stability** (1 feature):
   - L3 (3-game) rolling average of returning starters

3. **Market Dispersion** (2 features):
   - Spread variance across 5 sportsbooks
   - High dispersion flag (>1.5 SD)

4. **Game Context** (7 features):
   - Home/away, dome, division game
   - Temperature, wind, cold (<40°F), windy (>15mph)

**Meta-Model Architecture**:
- 5 separate LightGBM regressors (one per component weight)
- Input: 15 context features
- Output: 5 weights (normalized to sum to 1.0)
- Training: Optimal weights computed retrospectively via random simplex sampling

### Integration with v6.0 Base

v7.0 built on v6.0's 5-component ensemble:
- State-Space v2 (30% static → 10-50% adaptive)
- Hierarchical v2 (30% → 10-50%)
- XGBoost v4 Ridge (25% → 15-35%)
- Informative Priors v2 (15% → 5-30%)
- ELO+WLS (0% → 0-15% for rookies)

**Hypothesis**: Context-aware weighting would improve performance in edge cases (e.g., up-weight ELO for rookies, down-weight State-Space for unstable rosters).

---

## Why Did It Fail?

### Root Cause Analysis (Postmortem)

**See**: `docs/postmortems/ENSEMBLE_V7_VALIDATION_FAILURE_NOV18.md`

1. **Insufficient Signal-to-Noise Ratio**:
   - Context features weakly correlated with optimal weights (R² < 0.15)
   - Meta-model essentially learned to predict mean weights ≈ v6.0 static

2. **Overfitting to Historical Noise**:
   - "Optimal" weights computed per game were unstable (high variance)
   - Meta-model couldn't generalize beyond training distribution

3. **Component Correlation**:
   - State-Space, Hierarchical, XGBoost predictions highly correlated (ρ > 0.85)
   - Changing weights had minimal impact on final ensemble prediction

4. **Baseline Already Near-Optimal**:
   - v6.0's static weights (30/30/25/15) derived from robust CVX optimization
   - Adaptive adjustments couldn't improve on well-tuned baseline

### Validation Timeline

- **Nov 18, 2025 (Phase C)**: Initial validation showed +0.01 yards (within noise)
- **Nov 19, 2025 (Multi-Season)**: Confirmed across 2022-2024 (no statistical significance)
- **Nov 19, 2025 (Decision)**: Abandoned per ADR-009

---

## Archived Files (Complete Inventory)

### Core Ensemble Code (3,981 lines)

**Primary Implementation**:
- `code/enhanced_ensemble_v7_production.py` (1,258 lines) - Production-ready v7.0 ensemble
- `code/enhanced_ensemble_v7.py` (1,147 lines) - Initial v7.0 implementation
- `code/enhanced_ensemble_v7_base.py` (920 lines) - Base class with WLS integration

**Meta-Learning Components**:
- `code/meta_learning_weight_predictor.py` (560 lines) - LightGBM meta-model
- `code/meta_learning_context.py` (376 lines) - Context feature extraction
- `code/stacked_meta_learner.py` (285 lines) - Alternative stacked approach
- `code/thompson_sampling_meta.py` (218 lines) - Bayesian bandits approach

**WLS Adjustment** (Reusable, NOT deleted):
- `py/ensemble/adaptive_wls_weights.py` (retained in main codebase)
- `py/ensemble/adaptive_wls_weights_v2.py` (retained)

### Scripts (7 files)

**Validation Scripts**:
- `scripts/backtest_ensemble_v7_multi_season.py` (485 lines) - Multi-season backtest
- `scripts/validate_ensemble_v7_multiyear.py` (412 lines) - Three-year validation
- `scripts/test_ensemble_v7_integration.py` (312 lines) - Integration tests
- `scripts/test_ensemble_v7_wls_adjustment.py` (201 lines) - WLS component tests
- `scripts/test_v7_single_prediction.py` (156 lines) - Single prediction test
- `scripts/aggregate_batch_results_v7.py` (134 lines) - Batch aggregation
- `scripts/populate_test_predictions_v7.py` (98 lines) - Test data generation

**Shell Scripts**:
- `scripts/run_batched_backtest_v7.sh` (78 lines) - Batch backtest runner
- `scripts/run_batched_backtest_v7_parallel.sh` (92 lines) - Parallel runner

**Validation Helpers**:
- `scripts/validate_v7_sports_analytics.py` (267 lines) - Sports analytics checks

### Tests (4 files)

**Unit Tests**:
- `tests/unit/test_ensemble_v7_components.py` (389 lines) - Component unit tests
- `tests/unit/test_stacked_meta_learner.py` (142 lines) - Meta-learner tests
- `tests/unit/test_thompson_sampling_meta.py` (118 lines) - Thompson sampling tests

**Integration Tests**:
- `tests/integration/test_ensemble_v7_integration.py` (512 lines) - Full integration
- `tests/integration/test_ensemble_v7_sql.py` (278 lines) - SQL validation

### Models (7 files, 0.5 GB)

**XGBoost Ridge Residuals**:
- `models/xgb_ridge_residuals_v7_20251114_171023.json` (12 MB)
- `models/xgb_ridge_residuals_v7_20251114_182121.json` (12 MB)
- `models/xgb_ridge_residuals_v7_backtest2024_20251114_184538.json` (11 MB)
- `models/xgb_ridge_residuals_v7_backtest2024_20251114_184552.json` (11 MB)
- `models/xgb_ridge_residuals_v7_backtest2024_20251117_140011.json` (12 MB)

**Ridge Models**:
- `models/ridge_residuals_v7_backtest2024_20251117_140011.pkl` (2.3 MB)
- `models/ridge_residuals_v7_backtest2024_20251114_184552.pkl` (2.3 MB)

### Results (18 CSV files)

**Validation Results (Nov 17-19)**:
- `results/ensemble_v7_validation/v7_full_2022_20251119_010640.csv` (1.2 MB)
- `results/ensemble_v7_validation/v7_full_2023_20251119_010640.csv` (1.3 MB)
- `results/ensemble_v7_validation/v7_full_2024_20251119_010640.csv` (1.1 MB)
- `results/ensemble_v7_validation/v7_wls_only_2022_20251119_010640.csv` (1.2 MB)
- `results/ensemble_v7_validation/v7_wls_only_2023_20251119_010640.csv` (1.3 MB)
- `results/ensemble_v7_validation/v7_wls_only_2024_20251119_010640.csv` (1.1 MB)
- `results/ensemble_v7_validation/v7_meta_only_2022_20251119_010640.csv` (1.2 MB)
- `results/ensemble_v7_validation/v7_meta_only_2023_20251119_010640.csv` (1.3 MB)
- `results/ensemble_v7_validation/v7_meta_only_2024_20251119_010640.csv` (1.1 MB)

_(9 additional CSVs from Nov 17 runs)_

**Logs**:
- `results/backtest_v7_run.log` (4.2 MB)
- `results/validation_v7_nov18_current.log` (2.8 MB)

### Documentation (11 files)

**Session Logs**:
- `docs/archive/2025_nov/SESSION_NOV18_ENSEMBLE_V7_PHASES_ABC_COMPLETE.md` (8,412 lines)
- `docs/archive/2025_nov/SESSION_NOV17_V7_VALIDATION_BLOCKER.md` (2,134 lines)

**Postmortems**:
- `docs/postmortems/ENSEMBLE_V7_VALIDATION_FAILURE_NOV18.md` (512 lines)
- `docs/milestones/postmortems/2025_11_18_ensemble_v7_validation_failure.md` (387 lines)

**Research**:
- `docs/research/ENSEMBLE_V7_FAILURE_INVESTIGATION.md` (678 lines)

**Knowledge Synthesis**:
- `docs/knowledge/ENSEMBLE_V7_SESSION_NOV18_SYNTHESIS.md` (1,245 lines)
- `docs/knowledge/ENSEMBLE_V7_PROCESSING_COMPLETE.md` (892 lines)

**Testing Guides**:
- `docs/testing/ENSEMBLE_V7_TEST_GUIDE.md` (487 lines)
- `docs/testing/ENSEMBLE_V7_PYTEST_COMMANDS.md` (156 lines)

**Milestones**:
- `docs/milestones/ENSEMBLE_V7_VALIDATION_COMPLETE.md` (734 lines)

**Root Documentation**:
- `SPORTS_ANALYTICS_VALIDATION_V7_REPORT.md` (312 lines)
- `SPORTS_ANALYTICS_VALIDATION_V7_SUMMARY.txt` (89 lines)
- `ENSEMBLE_V7_TEST_SUMMARY.md` (267 lines)

### Architecture Decision Record

**See**: `docs/adr/009-abandon-ensemble-v7-meta-learning.md` (retained in main codebase)

---

## What Was Reusable (NOT Archived)

The following components remain in the main codebase because they have value beyond v7.0:

1. **WLS Adaptive Weights** (`py/ensemble/adaptive_wls_weights_v2.py`):
   - ELO+WLS blending for dynamic Elo adjustments
   - Used in future ensemble versions

2. **Context Feature Extractors** (partial retention):
   - `py/features/qb_experience_features.py` - QB career stage classification
   - `py/features/roster_stability_features.py` - Roster turnover metrics
   - `py/features/market_dispersion_features.py` - Betting market uncertainty

3. **Test Infrastructure**:
   - Generic ensemble test harness (`tests/integration/test_ensemble_integration.py`)
   - SQL validation patterns (Rule 19)

4. **R Model Interface** (`py/ensemble/r_model_interface.py`):
   - Generic R↔Python bridge for State-Space/Hierarchical models
   - Shared by v6.0, v7.0, and future versions

---

## Lessons Learned

### Technical Insights

1. **Beware Minimal Gains in Mature Systems**:
   - v6.0's CVX-optimized weights already near optimal
   - Incremental improvements (<0.5%) extremely difficult without new data sources

2. **Context Features Need Strong Signal**:
   - Weak correlations (R² < 0.15) insufficient for meta-learning
   - Would need richer context (e.g., injury reports, coaching changes)

3. **Component Diversity Matters**:
   - High correlation (ρ > 0.85) between State-Space, Hierarchical, XGBoost limits gains
   - Adaptive weighting only helps if components disagree

4. **Validation Before Production**:
   - Multi-season validation (2022-2024) caught failure before deployment
   - Saved ~20 hours of production integration work

### Process Insights

1. **Clear Success Criteria Upfront**:
   - Target: +2.0 yards MAE improvement (5%)
   - Reality: +0.01 yards (0.02%)
   - Early abandonment saved 100+ hours

2. **Incremental Development Pays Off**:
   - Phase A (WLS): Validated independently (working)
   - Phase B (Meta): Validated independently (minimal gain)
   - Phase C (Integration): Confirmed Phase B failure

3. **Document Failures as Thoroughly as Successes**:
   - This archive prevents future re-invention
   - ADR-009 records decision rationale for long-term reference

---

## Related Documentation

- **ADR-009**: `docs/adr/009-abandon-ensemble-v7-meta-learning.md`
- **Postmortem**: `docs/postmortems/ENSEMBLE_V7_VALIDATION_FAILURE_NOV18.md`
- **Session Log**: `docs/archive/2025_nov/SESSION_NOV18_ENSEMBLE_V7_PHASES_ABC_COMPLETE.md`
- **Baseline (v6.0)**: `docs/archive/2025_nov/SESSION_NOV15_PRODUCTION_DEPLOYMENT_COMPLETE.md`

---

## Future Research Directions

If revisiting context-aware ensembles, consider:

1. **Richer Context Features**:
   - Injury severity (not just binary)
   - Coaching staff changes
   - Travel distance/time zone effects

2. **Online Learning**:
   - Update weights incrementally (not batch)
   - Explore multi-armed bandits (Thompson sampling showed promise)

3. **Component Diversification**:
   - Add fundamentally different models (e.g., neural networks, causal models)
   - Reduce correlation between components

4. **Longer Validation Windows**:
   - 5+ years to capture regime shifts
   - Test across rule changes (e.g., roughing the passer, PI reviews)

---

**Archive Created**: November 19, 2025
**By**: Claude Sonnet 4.5
**Total Files**: 67 (code + scripts + tests + models + results + docs)
**Total Size**: ~1.2 GB (mostly results CSVs and model artifacts)
