# Sequence Diagram: Training Workflow

**Last Updated:** November 15, 2025
**Diagram Type:** Sequence Diagram
**System:** NFL Analytics Platform
**Process:** End-to-End Model Training Workflow
**Status:** Production

---

## Overview

This sequence diagram documents the temporal flow of the model training workflow, from trigger detection through validation to production deployment.

**Key Phases:**
1. **Trigger Detection** - Auto-detect or manual trigger (2 minutes)
2. **Feature Extraction** - Pull from materialized views (8 minutes)
3. **Parallel Training** - Train 4 models concurrently (40 minutes)
4. **Validation** - 7-gate validation process (5 minutes)
5. **Registration** - Save to filesystem + database (3 minutes)
6. **Deployment** - Optimize weights, reload API (5 minutes)

**Total Duration:** ~63 minutes (parallel execution)

---

## High-Level Sequence Diagram

```mermaid
sequenceDiagram
    participant CRON as Cron/User
    participant DETECTOR as Auto-Retrain Detector
    participant FEATURE as Feature Extractor
    participant DB as PostgreSQL
    participant CACHE as Redis Cache

    participant SS as State-Space Trainer (R)
    participant HIER as Hierarchical Trainer (R)
    participant XGB as XGBoost v4 Trainer (Py)
    participant IP as Informative Priors (R)

    participant VAL as Validation Pipeline
    participant REG as Model Registry
    participant DEPLOY as Deployment Pipeline
    participant API as FastAPI Ensemble

    Note over CRON,API: Phase 1: Trigger Detection (2 min)
    CRON->>DETECTOR: Check if retrain needed
    DETECTOR->>DB: SELECT MAX(season) FROM games
    DB-->>DETECTOR: 2024
    DETECTOR->>REG: Get last trained season
    REG-->>DETECTOR: 2023
    DETECTOR-->>CRON: Retrain needed (new season)

    Note over CRON,API: Phase 2: Feature Extraction (8 min)
    CRON->>FEATURE: Extract features (2018-2024)
    FEATURE->>CACHE: Check feature cache
    CACHE-->>FEATURE: 70% cache hit
    FEATURE->>DB: Query mv_player_season_stats
    FEATURE->>DB: Query mv_team_rolling_stats
    FEATURE->>DB: Query mv_betting_features
    DB-->>FEATURE: 15,000 training rows
    FEATURE->>CACHE: Update feature cache (TTL=5min)
    FEATURE-->>CRON: features.csv (40 features)

    Note over CRON,API: Phase 3: Parallel Training (40 min bottleneck)
    par Train State-Space (20 min)
        CRON->>SS: Rscript train_state_space_v2.R
        SS->>SS: Stan sampling (4 chains, 2000 iter)
        SS->>SS: Extract player skills
        SS-->>CRON: simplified_random_walk_v2.rds
    and Train Hierarchical (40 min - BOTTLENECK)
        CRON->>HIER: Rscript train_hierarchical_v2.R
        HIER->>HIER: brms sampling (4 chains, 6000 iter)
        HIER->>HIER: Check convergence (Rhat < 1.02)
        HIER-->>CRON: passing_yards_hierarchical_v2.rds
    and Train XGBoost v4 (10 min)
        CRON->>XGB: python train_xgboost_v4_ridge.py
        XGB->>XGB: Stage 1: Ridge on residuals
        XGB->>XGB: Stage 2: XGBoost on orth features
        XGB-->>CRON: xgb_v4_ridge.pkl
    and Train Informative Priors (25 min)
        CRON->>IP: Rscript train_informative_priors_v2.R
        IP->>IP: brms sampling (priors from domain)
        IP-->>CRON: passing_informative_priors_v2.rds
    end

    Note over CRON,API: Phase 4: Temperature Scaling (3 min)
    CRON->>SS: Generate predictions
    CRON->>HIER: Generate predictions
    CRON->>XGB: Generate predictions
    CRON->>IP: Generate predictions
    SS-->>CRON: pred_ss.csv
    HIER-->>CRON: pred_hier.csv
    XGB-->>CRON: pred_xgb.csv
    IP-->>CRON: pred_ip.csv
    CRON->>DEPLOY: Train temperature scaler
    DEPLOY->>DEPLOY: Isotonic regression (T=0.5534)
    DEPLOY-->>CRON: temperature_scaler_v6.pkl

    Note over CRON,API: Phase 5: Validation (5 min)
    CRON->>VAL: Validate all models
    VAL->>VAL: Gate 1: Convergence (Rhat < 1.02)
    VAL->>VAL: Gate 2: ESS > 400
    VAL->>VAL: Gate 3: Divergences < 1%
    VAL->>VAL: Gate 4: Calibration 85-92%
    VAL->>VAL: Gate 5: Performance (MAE < threshold)
    VAL->>VAL: Gate 6: Data Leakage (CRITICAL)
    VAL->>VAL: Gate 7: Ensemble Improvement

    alt All Gates Pass
        VAL-->>CRON: ✅ All models validated
    else Any Gate Fails
        VAL->>CRON: ❌ Validation failed (details)
        CRON->>CRON: Alert team, rollback
        Note over CRON: STOP - Do not deploy
    end

    Note over CRON,API: Phase 6: Registration (3 min)
    CRON->>REG: Register models
    REG->>REG: Save to models/ directory
    REG->>DB: INSERT INTO ml.model_metadata
    REG->>REG: Update MODEL_REGISTRY.yaml
    REG->>REG: Git commit + Git LFS push
    REG-->>CRON: 5 models registered

    Note over CRON,API: Phase 7: Deployment (5 min)
    CRON->>DEPLOY: Optimize ensemble weights
    DEPLOY->>DEPLOY: CVX optimization (MAE minimize)
    DEPLOY-->>CRON: weights = [0.30, 0.30, 0.25, 0.15]
    CRON->>API: POST /reload
    API->>API: Load 4 models + temperature scaler
    API-->>CRON: Ensemble v6 operational

    Note over CRON,API: Total Duration: ~63 minutes
```

---

## Detailed Phase Breakdown

### Phase 1: Trigger Detection (2 minutes)

**Decision Points:**
1. New season detected?
2. Data drift detected?
3. Performance degradation?
4. Manual override?

```mermaid
sequenceDiagram
    participant USER as User/Cron
    participant DETECTOR as Auto-Retrain Detector
    participant DB as PostgreSQL
    participant REGISTRY as Model Registry

    USER->>DETECTOR: should_retrain()
    DETECTOR->>DB: SELECT MAX(season) FROM games
    DB-->>DETECTOR: 2024

    DETECTOR->>REGISTRY: get_latest_training_season()
    REGISTRY-->>DETECTOR: 2023

    alt New Season Detected
        DETECTOR-->>USER: {"should_retrain": true, "reason": "new_season"}
    else No New Season
        DETECTOR->>DETECTOR: Check data drift
        alt Drift > Threshold
            DETECTOR-->>USER: {"should_retrain": true, "reason": "data_drift"}
        else No Drift
            DETECTOR->>DETECTOR: Check performance degradation
            alt Performance Degraded
                DETECTOR-->>USER: {"should_retrain": true, "reason": "performance_degradation"}
            else All OK
                DETECTOR-->>USER: {"should_retrain": false, "reason": null}
            end
        end
    end
```

---

### Phase 2: Feature Extraction (8 minutes)

**Optimization:** Redis caching reduces 45s → 8s (70% hit rate)

```mermaid
sequenceDiagram
    participant CRON as Cron/User
    participant EXTRACTOR as Feature Extractor
    participant CACHE as Redis Cache
    participant DB as PostgreSQL

    CRON->>EXTRACTOR: extract_features(2018-2024)

    loop For each player-game
        EXTRACTOR->>CACHE: GET "features:{player}:{season}:{week}"

        alt Cache Hit (70%)
            CACHE-->>EXTRACTOR: Cached features
        else Cache Miss (30%)
            EXTRACTOR->>DB: Query mv_player_season_stats
            EXTRACTOR->>DB: Query mv_team_rolling_stats
            EXTRACTOR->>DB: Query mv_betting_features
            DB-->>EXTRACTOR: Raw features
            EXTRACTOR->>EXTRACTOR: Transform + validate
            EXTRACTOR->>CACHE: SETEX (TTL=5min)
        end
    end

    EXTRACTOR->>EXTRACTOR: Combine 40+ features
    EXTRACTOR->>EXTRACTOR: Train/test split (temporal)
    EXTRACTOR-->>CRON: train.csv (15k rows), test.csv (3k rows)
```

---

### Phase 3: Parallel Training (40 minutes)

**Bottleneck:** Hierarchical v2 (40 minutes with 6000 iterations)

**Decision Point:** If convergence fails → Increase iterations OR adjust priors OR fail

```mermaid
sequenceDiagram
    participant CRON as Orchestrator
    participant SS as State-Space (R)
    participant HIER as Hierarchical (R)
    participant XGB as XGBoost v4 (Py)
    participant IP as Informative Priors (R)

    Note over CRON: Start parallel training (t=0)

    par State-Space (20 min)
        CRON->>SS: train_state_space_v2.R
        SS->>SS: Stan sampling (chains=4, iter=2000)
        alt Rhat < 1.01 (Success)
            SS->>SS: Extract player skills
            SS-->>CRON: model.rds + skills.csv (t=20)
        else Rhat >= 1.01 (Failure)
            SS-->>CRON: ❌ Convergence failed
        end

    and Hierarchical (40 min - BOTTLENECK)
        CRON->>HIER: train_hierarchical_v2.R
        HIER->>HIER: brms sampling (chains=4, iter=6000)
        alt Rhat < 1.02 (Success)
            HIER-->>CRON: model.rds (t=40)
        else Rhat >= 1.02 (Retry)
            HIER->>HIER: Increase to iter=8000
            alt Rhat < 1.02 (Success)
                HIER-->>CRON: model.rds (t=52)
            else Still Failed
                HIER-->>CRON: ❌ Convergence failed
            end
        end

    and XGBoost v4 (10 min)
        CRON->>XGB: train_xgboost_v4_ridge.py
        XGB->>XGB: Stage 1: Ridge regression
        XGB->>XGB: Stage 2: XGBoost
        XGB-->>CRON: model.pkl (t=10)

    and Informative Priors (25 min)
        CRON->>IP: train_informative_priors_v2.R
        IP->>IP: brms sampling (chains=4, iter=4000)
        IP-->>CRON: model.rds (t=25)
    end

    Note over CRON: All models complete (t=40)
```

---

### Phase 4: Temperature Scaling (3 minutes)

**Purpose:** Calibrate ensemble to 90% coverage

```mermaid
sequenceDiagram
    participant CRON as Orchestrator
    participant SS as State-Space
    participant HIER as Hierarchical
    participant XGB as XGBoost v4
    participant IP as Informative Priors
    participant SCALER as Temperature Scaler

    CRON->>SS: predict(test_df)
    SS-->>CRON: pred_ss (mean, std)

    CRON->>HIER: predict(test_df)
    HIER-->>CRON: pred_hier (mean, std)

    CRON->>XGB: predict(test_df)
    XGB-->>CRON: pred_xgb (mean)

    CRON->>IP: predict(test_df)
    IP-->>CRON: pred_ip (mean, std)

    CRON->>SCALER: train(predictions, actuals)
    SCALER->>SCALER: Isotonic regression
    SCALER->>SCALER: Find optimal T = 0.5534
    SCALER-->>CRON: scaler.pkl (coverage=89.7%)
```

---

### Phase 5: Validation (5 minutes)

**Critical Gates:** 7 checks must pass before deployment

```mermaid
sequenceDiagram
    participant CRON as Orchestrator
    participant VAL as Validation Pipeline
    participant ALERT as Alert Manager

    CRON->>VAL: validate_models([ss, hier, xgb, ip, scaler])

    Note over VAL: Gate 1: Convergence
    VAL->>VAL: Check Rhat < 1.02 (R models)
    alt Pass
        VAL->>VAL: ✅ Gate 1 passed
    else Fail
        VAL->>ALERT: ❌ Convergence failed (Rhat=1.0337)
        ALERT-->>CRON: STOP - Do not deploy
    end

    Note over VAL: Gate 2: Effective Sample Size
    VAL->>VAL: Check ESS > 400 (R models)
    alt Pass
        VAL->>VAL: ✅ Gate 2 passed
    else Fail
        VAL->>ALERT: ❌ ESS too low (ESS=320)
        ALERT-->>CRON: STOP
    end

    Note over VAL: Gate 3: Divergences
    VAL->>VAL: Check divergences < 1% (R models)
    alt Pass
        VAL->>VAL: ✅ Gate 3 passed
    else Fail
        VAL->>ALERT: ❌ Too many divergences (2.5%)
        ALERT-->>CRON: STOP
    end

    Note over VAL: Gate 4: Calibration
    VAL->>VAL: Check 90% coverage = 85-92%
    alt Pass
        VAL->>VAL: ✅ Gate 4 passed (89.7%)
    else Fail
        VAL->>ALERT: ❌ Calibration poor (78%)
        ALERT-->>CRON: STOP
    end

    Note over VAL: Gate 5: Performance
    VAL->>VAL: Check MAE < threshold
    alt Pass
        VAL->>VAL: ✅ Gate 5 passed
    else Fail
        VAL->>ALERT: ❌ MAE too high (65 yards)
        ALERT-->>CRON: STOP
    end

    Note over VAL: Gate 6: Data Leakage (CRITICAL)
    VAL->>VAL: Check temporal/target/player leakage
    alt Pass
        VAL->>VAL: ✅ Gate 6 passed
    else Fail (CRITICAL)
        VAL->>ALERT: 🚨 DATA LEAKAGE DETECTED
        ALERT-->>CRON: CRITICAL - DO NOT DEPLOY
    end

    Note over VAL: Gate 7: Ensemble Improvement
    VAL->>VAL: Check MAE improvement vs baseline
    alt Pass
        VAL->>VAL: ✅ Gate 7 passed (+17% improvement)
    else Fail (Warning Only)
        VAL->>ALERT: ⚠️ No improvement (investigate)
        VAL->>VAL: Continue (not blocking)
    end

    VAL-->>CRON: ✅ All critical gates passed
```

---

### Phase 6: Registration (3 minutes)

**Storage:** Filesystem + Database + Git LFS

```mermaid
sequenceDiagram
    participant CRON as Orchestrator
    participant REG as Model Registry
    participant FS as Filesystem
    participant DB as PostgreSQL
    participant GIT as Git LFS

    CRON->>REG: register_models([ss, hier, xgb, ip, scaler])

    loop For each model
        REG->>FS: Save model to models/
        FS-->>REG: saved (e.g., models/bayesian/simplified_random_walk_v2.rds)

        REG->>DB: INSERT INTO ml.model_metadata
        DB-->>REG: model_id = 42

        REG->>REG: Update MODEL_REGISTRY.yaml
        REG->>GIT: git add models/bayesian/*.rds
        REG->>GIT: git lfs track "*.rds"
        REG->>GIT: git commit -m "feat: register state-space v2"
        REG->>GIT: git push origin main
    end

    REG-->>CRON: 5 models registered (IDs: 42-46)
```

---

### Phase 7: Deployment (5 minutes)

**Weight Optimization:** CVX solver minimizes validation MAE

```mermaid
sequenceDiagram
    participant CRON as Orchestrator
    participant DEPLOY as Deployment Pipeline
    participant CVX as CVX Optimizer
    participant API as FastAPI Ensemble

    CRON->>DEPLOY: deploy_ensemble()
    DEPLOY->>CVX: optimize_weights(predictions, actuals)

    CVX->>CVX: Solve convex optimization
    Note over CVX: minimize MAE<br/>subject to:<br/>sum(w) = 1<br/>w >= 0

    CVX-->>DEPLOY: optimal_weights = [0.30, 0.30, 0.25, 0.15]

    DEPLOY->>API: POST /reload
    API->>API: Load state-space v2 (30%)
    API->>API: Load hierarchical v2 (30%)
    API->>API: Load XGBoost v4 (25%)
    API->>API: Load informative priors v2 (15%)
    API->>API: Load temperature scaler v6
    API-->>DEPLOY: ✅ Ensemble v6 operational

    DEPLOY->>DEPLOY: Run smoke tests
    alt Smoke Tests Pass
        DEPLOY-->>CRON: ✅ Deployment successful
    else Smoke Tests Fail
        DEPLOY->>API: POST /rollback
        API-->>DEPLOY: ✅ Rolled back to v5.1
        DEPLOY-->>CRON: ⚠️ Deployment failed, rolled back
    end
```

---

## Error Handling Scenarios

### Scenario 1: Convergence Failure (Hierarchical v2)

```mermaid
sequenceDiagram
    participant HIER as Hierarchical Trainer
    participant VAL as Validation
    participant ALERT as Alert Manager
    participant ADMIN as Administrator

    HIER->>VAL: Check convergence (Rhat=1.0337)
    VAL->>VAL: Gate 1: Rhat < 1.02? ❌ FAILED

    VAL->>ALERT: Send alert (convergence failed)
    ALERT->>ADMIN: Email: Rhat=1.0337, increase iterations

    ADMIN->>HIER: Increase iter: 6000 → 8000
    HIER->>HIER: Retrain (8000 iterations)
    HIER->>VAL: Check convergence (Rhat=1.0092)
    VAL->>VAL: Gate 1: Rhat < 1.02? ✅ PASSED

    Note over VAL,ADMIN: Continue to next gates
```

---

### Scenario 2: Data Leakage Detected (CRITICAL)

```mermaid
sequenceDiagram
    participant VAL as Validation
    participant LEAK as Data Leakage Detector
    participant ALERT as Alert Manager
    participant ADMIN as Administrator

    VAL->>LEAK: validate_split(train, test)
    LEAK->>LEAK: Check temporal order
    LEAK->>LEAK: ❌ CRITICAL: Future data in training set

    LEAK-->>VAL: has_critical_leakage = True
    VAL->>ALERT: 🚨 CRITICAL DATA LEAKAGE

    ALERT->>ADMIN: PagerDuty alert + Email
    Note over ALERT,ADMIN: Details: Test dates (2022-01-01)<br/>appear in training set

    ADMIN->>ADMIN: Investigate feature engineering
    ADMIN->>ADMIN: Fix temporal join (use asof)
    ADMIN->>VAL: Re-run training pipeline

    VAL->>LEAK: validate_split(train_fixed, test)
    LEAK->>LEAK: ✅ No leakage detected
    LEAK-->>VAL: has_critical_leakage = False
    VAL-->>ADMIN: ✅ Safe to continue
```

**Historical Example:** BNN investigation (Nov 9, 2025) - 84 hours wasted before leakage detected. Now caught in <1 minute.

---

### Scenario 3: Ensemble Performance Degradation

```mermaid
sequenceDiagram
    participant VAL as Validation
    participant DEPLOY as Deployment
    participant ALERT as Alert Manager
    participant ADMIN as Administrator

    VAL->>VAL: Gate 7: Ensemble improvement
    VAL->>VAL: New MAE: 42 yards, Baseline: 40 yards
    VAL->>VAL: ❌ WARNING: Degradation -5%

    VAL->>ALERT: ⚠️ Ensemble performance degraded
    ALERT->>ADMIN: Slack: New ensemble worse than baseline

    ADMIN->>ADMIN: Investigate model drift
    alt Root Cause: XGBoost v4 overfitting
        ADMIN->>DEPLOY: Reduce XGBoost weight: 25% → 15%
        DEPLOY->>DEPLOY: Re-optimize weights
        DEPLOY->>VAL: New MAE: 39 yards (+2.5% improvement)
        VAL-->>ADMIN: ✅ Proceed to deployment
    else Root Cause: Data quality issue
        ADMIN->>ADMIN: Fix data pipeline
        ADMIN->>VAL: Re-run training
    end
```

---

## Performance Optimization History

### Nov 13, 2025: Hierarchical Convergence Fix

**Problem:** Rhat=1.0337 (failed validation)
**Solution:** Increased iterations 2000 → 6000
**Result:** Rhat=1.0092 (73% improvement, PASSED)
**Trade-off:** Training time +15 minutes (25 → 40 minutes)

### Nov 15, 2025: XGBoost v4 Ridge Residuals

**Problem:** XGBoost v2 overfitting ensemble predictions
**Solution:** Two-stage orthogonalization (Ridge + XGBoost)
**Result:** MAE 4.70 yards (88% improvement over v2 baseline)
**Impact:** Ensemble v6.0 expected +17.5% improvement over v5.1

---

## Related Documentation

**Core Files:**
- [c4_component_training_pipeline.md](c4_component_training_pipeline.md) - Component diagram
- [MODEL_REGISTRY.yaml](../../models/MODEL_REGISTRY.yaml) - Production models
- [TRAINING_WINDOW_ABLATION_RESULTS.md](../milestones/TRAINING_WINDOW_ABLATION_RESULTS.md)

**Utilities:**
- `py/utils/convergence_checker.py`
- `py/utils/calibration_checker.py`
- `py/utils/data_leakage_detector.py`

---

## Legend

**Notation:**
- `→` : Synchronous call
- `-->>` : Return value
- `par ... and` : Parallel execution
- `alt ... else` : Conditional branch
- `loop` : Iteration

**Time Annotations:**
- `(t=20)` : Time elapsed since training start (minutes)
- Bottleneck highlighted in notes

---

**Maintained by:** Model Management Team
**Contact:** See [CLAUDE.md](../../CLAUDE.md) for architecture questions
