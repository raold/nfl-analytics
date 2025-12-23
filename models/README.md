# Models Directory Structure

**Last Updated:** December 23, 2025
**Reorganized:** Phase 0.2 of XGBoost Integration Plan

## Structure Overview

```
models/
├── bayesian/          # PRODUCTION: R models (4.8GB)
├── calibration/       # PRODUCTION: Temperature scalers
├── ensemble/          # PRODUCTION: Ensemble configs
├── props/             # PRODUCTION: Player props models
├── state_space/       # PRODUCTION: State-space artifacts
├── team_strength/     # PRODUCTION: Team strength models
├── xgboost/           # PRODUCTION: XGBoost production models
├── rag/               # PRODUCTION: RAG/semantic models
├── staging/           # STAGING: Models under validation
├── experiments/       # EXPERIMENTS: Active development
└── archive/           # ARCHIVE: Deprecated/old models
```

## Production Models (Root Level)

These models are actively used in the production ensemble (v6.3):

| Directory | Description | Weight in Ensemble |
|-----------|-------------|-------------------|
| `bayesian/` | R models: state_space_v2, hierarchical_v2, informative_priors_v2 | 85% combined |
| `calibration/` | Temperature scalers for probability calibration | N/A (post-processing) |
| `ensemble/` | Ensemble configurations and backtest results | N/A (orchestration) |
| `xgboost/` | XGBoost models (v4 Ridge DISABLED due to leakage) | 0% (was 23%) |
| `props/` | Player props models | Separate pipeline |
| `team_strength/` | Team offense/defense strength models | Used as features |
| `state_space/` | State-space model artifacts | Part of bayesian pipeline |
| `rag/` | RAG and semantic stack models | New (Dec 2025) |

## Staging (`staging/`)

Models under validation before promotion to production:

| Model | Created | Status | Notes |
|-------|---------|--------|-------|
| `game_outcome_v1/` | Nov 29, 2025 | Pre-semantic | 293 features |
| `game_outcome_week15/` | Dec 14, 2025 | Pre-semantic | 335 features |

**Next:** `game_outcome_v2/` will be trained with 38 semantic features.

## Experiments (`experiments/`)

Active research and development:

| Directory | Description |
|-----------|-------------|
| `xgboost_semantic/` | XGBoost with semantic features (planned) |
| `gnn/` | Graph Neural Network experiments |
| `cvae/` | Conditional VAE experiments |
| `evidential/` | Evidential deep learning |
| `feature_exploration/` | Feature optimization/weighting tests |

## Archive (`archive/`)

Deprecated or old models kept for reproducibility:

| Directory | Contents | Archived Date |
|-----------|----------|---------------|
| `rl_experiments/` | CQL, IQL, fourth_down, simulator | Dec 23, 2025 |
| `ablation_experiments/` | Dec 14 ablation (with/without synthetic) | Dec 23, 2025 |
| `game_outcome_legacy/` | Early game_outcome experiments | Dec 23, 2025 |
| `loose_files/` | Old .pth, .pkl, .json files from root | Dec 23, 2025 |
| `baselines/` | Early baseline models | Dec 23, 2025 |
| `deprecated/` | Previously deprecated models | Dec 23, 2025 |
| `elo/` | Elo rating experiments | Dec 23, 2025 |
| `stan/` | Stan model experiments | Dec 23, 2025 |

## Key Files

| File | Description |
|------|-------------|
| `MODEL_REGISTRY.yaml` | Comprehensive model documentation |
| `registry.json` | JSON model registry (API compatible) |
| `README.md` | This file |

## Critical Notes

### XGBoost v4 Ridge Leakage (Dec 17, 2025)

**DISABLED** - The XGBoost v4 Ridge model had target leakage:
- `residual_feature` (actual - ridge_pred) was used as INPUT
- Feature importance showed 91.4% on residual_feature
- Model was "seeing the answer" during training

**Fix available:** `scripts/xgb_exp7_fixed_leakage.py`

### Model Promotion Workflow

1. Train in `experiments/`
2. Validate on holdout data
3. Move to `staging/`
4. A/B test against production
5. Promote to root level (production)
6. Update `MODEL_REGISTRY.yaml` and `registry.json`

## Size Summary

| Directory | Size | Tracked in Git |
|-----------|------|----------------|
| `bayesian/` | 4.8GB | Partial (.rds, .csv) |
| `archive/rl_experiments/` | ~471MB | Partial |
| `experiments/gnn/` | 56MB | Partial |
| `xgboost/` | 31MB | Partial |
| Other | <50MB total | Varies |

**Note:** Large .pth and .pkl files are gitignored. Only metadata and small artifacts are tracked.
