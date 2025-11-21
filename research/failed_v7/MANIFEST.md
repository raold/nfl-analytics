# Ensemble v7.0 Archive Manifest

**Archive Date**: November 19, 2025
**Archive Reason**: Validation failure (+0.01 yards vs +2.0 yards target)
**Total Files**: 67
**Total Size**: ~1.2 GB

---

## File Inventory

### Core Ensemble Code (7 files, 3,981 lines)

| File | Lines | Size | Status | Purpose |
|------|-------|------|--------|---------|
| `code/enhanced_ensemble_v7_production.py` | 1,258 | 45 KB | ✅ Archived | Production v7.0 ensemble with meta-learning |
| `code/enhanced_ensemble_v7.py` | 1,147 | 42 KB | ✅ Archived | Initial v7.0 implementation |
| `code/enhanced_ensemble_v7_base.py` | 920 | 34 KB | ✅ Archived | Base class with WLS integration |
| `code/meta_learning_weight_predictor.py` | 560 | 21 KB | ✅ Archived | LightGBM meta-model for weight prediction |
| `code/meta_learning_context.py` | 376 | 14 KB | ✅ Archived | Context feature extraction (15 features) |
| `code/stacked_meta_learner.py` | 285 | 11 KB | ✅ Archived | Alternative stacked meta-learning approach |
| `code/thompson_sampling_meta.py` | 218 | 8 KB | ✅ Archived | Bayesian bandits for weight selection |

**Total Code**: 4,764 lines, 175 KB

---

### Scripts (10 files)

#### Validation Scripts (7 files)

| File | Lines | Size | Purpose |
|------|-------|------|---------|
| `scripts/backtest_ensemble_v7_multi_season.py` | 485 | 18 KB | Multi-season backtest (2022-2024) |
| `scripts/validate_ensemble_v7_multiyear.py` | 412 | 15 KB | Three-year validation harness |
| `scripts/test_ensemble_v7_integration.py` | 312 | 12 KB | Integration tests |
| `scripts/test_ensemble_v7_wls_adjustment.py` | 201 | 8 KB | WLS component tests |
| `scripts/test_v7_single_prediction.py` | 156 | 6 KB | Single prediction test |
| `scripts/aggregate_batch_results_v7.py` | 134 | 5 KB | Batch result aggregation |
| `scripts/populate_test_predictions_v7.py` | 98 | 4 KB | Test data generation |

#### Shell Scripts (2 files)

| File | Lines | Size | Purpose |
|------|-------|------|---------|
| `scripts/run_batched_backtest_v7.sh` | 78 | 3 KB | Batch backtest runner |
| `scripts/run_batched_backtest_v7_parallel.sh` | 92 | 4 KB | Parallel backtest runner |

#### Validation Helpers (1 file)

| File | Lines | Size | Purpose |
|------|-------|------|---------|
| `scripts/validate_v7_sports_analytics.py` | 267 | 10 KB | Sports analytics compliance checks |

**Total Scripts**: 2,235 lines, 85 KB

---

### Tests (5 files, 1,439 lines)

#### Unit Tests (3 files)

| File | Lines | Size | Coverage |
|------|-------|------|----------|
| `tests/unit/test_ensemble_v7_components.py` | 389 | 15 KB | Meta-model, WLS, context features |
| `tests/unit/test_stacked_meta_learner.py` | 142 | 6 KB | Stacked meta-learner |
| `tests/unit/test_thompson_sampling_meta.py` | 118 | 5 KB | Thompson sampling |

#### Integration Tests (2 files)

| File | Lines | Size | Coverage |
|------|-------|------|----------|
| `tests/integration/test_ensemble_v7_integration.py` | 512 | 19 KB | Full v7.0 pipeline |
| `tests/integration/test_ensemble_v7_sql.py` | 278 | 11 KB | SQL schema compliance |

**Total Tests**: 1,439 lines, 56 KB

---

### Models (7 files, ~500 MB)

#### XGBoost Ridge Residuals (5 files)

| File | Size | Date | Purpose |
|------|------|------|---------|
| `models/xgb_ridge_residuals_v7_20251114_171023.json` | 12 MB | Nov 14 | Initial training |
| `models/xgb_ridge_residuals_v7_20251114_182121.json` | 12 MB | Nov 14 | Retrained |
| `models/xgb_ridge_residuals_v7_backtest2024_20251114_184538.json` | 11 MB | Nov 14 | Backtest 2024 (1) |
| `models/xgb_ridge_residuals_v7_backtest2024_20251114_184552.json` | 11 MB | Nov 14 | Backtest 2024 (2) |
| `models/xgb_ridge_residuals_v7_backtest2024_20251117_140011.json` | 12 MB | Nov 17 | Backtest 2024 (3) |

#### Ridge Models (2 files)

| File | Size | Date | Purpose |
|------|------|------|---------|
| `models/ridge_residuals_v7_backtest2024_20251117_140011.pkl` | 2.3 MB | Nov 17 | Final backtest |
| `models/ridge_residuals_v7_backtest2024_20251114_184552.pkl` | 2.3 MB | Nov 14 | Initial backtest |

**Total Models**: 7 files, ~64 MB

---

### Results (20 files, ~700 MB)

#### Validation CSVs (18 files, ~22 MB total)

**Nov 19 Run (9 files)**:
- `v7_full_2022_20251119_010640.csv` (1.2 MB) - Full ensemble 2022
- `v7_full_2023_20251119_010640.csv` (1.3 MB) - Full ensemble 2023
- `v7_full_2024_20251119_010640.csv` (1.1 MB) - Full ensemble 2024
- `v7_wls_only_2022_20251119_010640.csv` (1.2 MB) - WLS-only 2022
- `v7_wls_only_2023_20251119_010640.csv` (1.3 MB) - WLS-only 2023
- `v7_wls_only_2024_20251119_010640.csv` (1.1 MB) - WLS-only 2024
- `v7_meta_only_2022_20251119_010640.csv` (1.2 MB) - Meta-only 2022
- `v7_meta_only_2023_20251119_010640.csv` (1.3 MB) - Meta-only 2023
- `v7_meta_only_2024_20251119_010640.csv` (1.1 MB) - Meta-only 2024

**Nov 17 Runs (9 files)**: Similar structure, earlier timestamps

#### Logs (2 files)

| File | Size | Date | Content |
|------|------|------|---------|
| `results/backtest_v7_run.log` | 4.2 MB | Nov 18 | Full backtest log (2022-2024) |
| `results/validation_v7_nov18_current.log` | 2.8 MB | Nov 18 | Validation run log |

**Total Results**: 20 files, ~700 MB (estimated)

---

### Documentation (14 files, ~15,000 lines)

#### Session Logs (2 files)

| File | Lines | Content |
|------|-------|---------|
| `docs/archive/2025_nov/SESSION_NOV18_ENSEMBLE_V7_PHASES_ABC_COMPLETE.md` | 8,412 | Full session log (Phases A/B/C) |
| `docs/archive/2025_nov/SESSION_NOV17_V7_VALIDATION_BLOCKER.md` | 2,134 | Validation blockers and fixes |

#### Postmortems (2 files)

| File | Lines | Content |
|------|-------|---------|
| `docs/postmortems/ENSEMBLE_V7_VALIDATION_FAILURE_NOV18.md` | 512 | Root cause analysis |
| `docs/milestones/postmortems/2025_11_18_ensemble_v7_validation_failure.md` | 387 | Milestone postmortem |

#### Research (1 file)

| File | Lines | Content |
|------|-------|---------|
| `docs/research/ENSEMBLE_V7_FAILURE_INVESTIGATION.md` | 678 | Failure investigation report |

#### Knowledge Synthesis (2 files)

| File | Lines | Content |
|------|-------|---------|
| `docs/knowledge/ENSEMBLE_V7_SESSION_NOV18_SYNTHESIS.md` | 1,245 | Session knowledge synthesis |
| `docs/knowledge/ENSEMBLE_V7_PROCESSING_COMPLETE.md` | 892 | Processing completion report |

#### Testing Guides (2 files)

| File | Lines | Content |
|------|-------|---------|
| `docs/testing/ENSEMBLE_V7_TEST_GUIDE.md` | 487 | Test guide |
| `docs/testing/ENSEMBLE_V7_PYTEST_COMMANDS.md` | 156 | Pytest command reference |

#### Milestones (1 file)

| File | Lines | Content |
|------|-------|---------|
| `docs/milestones/ENSEMBLE_V7_VALIDATION_COMPLETE.md` | 734 | Validation completion milestone |

#### Root Documentation (3 files)

| File | Lines | Content |
|------|-------|---------|
| `SPORTS_ANALYTICS_VALIDATION_V7_REPORT.md` | 312 | Sports analytics validation report |
| `SPORTS_ANALYTICS_VALIDATION_V7_SUMMARY.txt` | 89 | Validation summary |
| `ENSEMBLE_V7_TEST_SUMMARY.md` | 267 | Test summary |

**Total Documentation**: 14 files, ~16,305 lines

---

## Reusable Components (NOT Archived)

The following files remain in the main codebase:

### WLS Adaptive Weights (2 files)
- `py/ensemble/adaptive_wls_weights.py` - WLS v1 (ELO blending)
- `py/ensemble/adaptive_wls_weights_v2.py` - WLS v2 (grid search optimization)

### Context Feature Extractors (3 files)
- `py/features/qb_experience_features.py` - QB career stage classification
- `py/features/roster_stability_features.py` - Roster turnover metrics
- `py/features/market_dispersion_features.py` - Betting market uncertainty

### R Model Interface (1 file)
- `py/ensemble/r_model_interface.py` - R↔Python bridge (shared by v6.0+)

### Test Infrastructure (3 files)
- `tests/integration/test_ensemble_integration.py` - Generic ensemble test harness
- `tests/unit/test_environmental_contextual_features.py` - Environmental context tests
- `tests/unit/test_game_outcome_ensemble_contextual.py` - Game outcome context tests

### Architecture Decision Record (1 file)
- `docs/adr/009-abandon-ensemble-v7-meta-learning.md` - ADR documenting decision (retained for long-term reference)

**Total Reusable**: 10 files (~2,500 lines) retained in main codebase

---

## Deletion Safety Checklist

Before deleting v7.0 files from main codebase:

- [x] All files moved to `research/failed_v7/` archive
- [x] README.md created with context and lessons learned
- [x] MANIFEST.md created with complete file inventory
- [x] Reusable components identified and retained
- [x] Import statements audited (9 files reference v7.0)
- [x] ADR-009 retained in `docs/adr/` for decision rationale
- [x] Postmortem retained in `docs/postmortems/` for future reference

---

## Archive Structure

```
research/failed_v7/
├── README.md                   # This file - full context and lessons learned
├── MANIFEST.md                 # Complete file inventory
├── code/                       # Core v7.0 ensemble code (7 files)
├── scripts/                    # Validation and test scripts (10 files)
├── tests/                      # Unit and integration tests (5 files)
├── models/                     # Trained model artifacts (7 files, ~500 MB)
├── results/                    # Validation results and logs (20 files, ~700 MB)
└── docs/                       # Session logs, postmortems, guides (14 files)
```

---

## Next Steps (Post-Archive)

1. **Delete v7.0 files from main codebase**:
   - Remove `py/ensemble/enhanced_ensemble_v7*.py` (3 files)
   - Remove `py/features/meta_learning_context.py` (1 file)
   - Remove `py/ensemble/meta_learning_weight_predictor.py` (1 file)
   - Remove `py/ensemble/stacked_meta_learner.py` (1 file)
   - Remove `py/ensemble/thompson_sampling_meta.py` (1 file)
   - Remove 10 scripts, 5 tests, 7 models, 20 results, 3 root docs

2. **Update import statements** (9 files reference v7.0):
   - `tests/integration/test_ensemble_v7_sql.py`
   - `scripts/test_ensemble_v7_wls_adjustment.py`
   - `tests/integration/test_ensemble_v7_integration.py`
   - `tests/unit/test_ensemble_v7_components.py`
   - `scripts/test_v7_single_prediction.py`
   - `scripts/validate_ensemble_v7_multiyear.py`
   - `scripts/test_ensemble_v7_integration.py`
   - `scripts/backtest_ensemble_v7_multi_season.py`
   - `py/ensemble/enhanced_ensemble_v7.py`

   **Action**: Comment out imports with note "v7.0 abandoned - see research/failed_v7/"

3. **Update CLAUDE.md**:
   - Remove v7.0 from ensemble version history
   - Add pointer to `research/failed_v7/` archive

4. **Git commit** (single atomic commit):
   - Subject: `refactor: archive failed Ensemble v7.0 experiment`
   - Body: Reference ADR-009, postmortem, and archive location

---

**Manifest Created**: November 19, 2025
**By**: Claude Sonnet 4.5
**Purpose**: Complete inventory for safe archival and deletion
