# Comprehensive Parallel Workflow Guide

**Last Updated**: 2025-10-10
**Purpose**: Master guide for parallel processing workflows across all NFL analytics operations
**Audience**: All agents (DevOps, ETL, Research, Academic Publishing)

---

## 🎯 Overview

This guide documents parallel processing patterns and workflows that enable **3-6x performance improvements** across dissertation writing, data processing, and model training operations.

### Parallel Agent Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PARALLEL ORCHESTRATION LAYER                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │
│  │   Academic       │  │  ETL Parallel    │  │  Distributed     │      │
│  │   Publishing     │  │  Orchestrator    │  │  Training        │      │
│  │   Agent          │  │  Agent           │  │  Coordinator     │      │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘      │
│           │                     │                      │                 │
│           │                     │                      │                 │
│  ┌────────▼─────────┐  ┌────────▼─────────┐  ┌────────▼─────────┐      │
│  │  DevOps Parallel │  │    Research      │  │      ETL          │      │
│  │  Orchestrator    │  │    Agent         │  │     Agent         │      │
│  └──────────────────┘  └──────────────────┘  └───────────────────┘      │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### Performance Gains Summary

| Workflow | Sequential Time | Parallel Time | Speedup |
|----------|----------------|---------------|---------|
| **Dissertation Build** | 45-60 min | 8-12 min | 4-6x |
| **Full ETL Refresh** | 60-90 min | 15-20 min | 3-4x |
| **Model Training (4 models)** | 12-16 hours | 3-4 hours | 4x |
| **Database Migration Test** | 15-20 min | 4-5 min | 3-4x |
| **Hyperparameter Search (324 configs)** | 8-12 days | 1.5-2 days | 5-6x |

---

## 📚 Complete Workflow Catalog

### 1. Dissertation Compilation Workflow

**Agent**: Academic Publishing Agent
**Trigger**: User requests dissertation build or chapter update

#### Full Parallel Build
```
┌─────────────────────────────────────────────────────────────┐
│ START: Dissertation Build Request                           │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
     ┌─────────────────────┐
     │ Check Dependencies  │  (Which chapters changed?)
     └──────────┬──────────┘
                │
                ▼
     ┌──────────────────────────────────────────────┐
     │ PARALLEL: Render Quarto Notebooks (j=4)      │
     ├──────────────────────────────────────────────┤
     │ [1] notebooks/04_score_validation.qmd        │
     │ [2] notebooks/05_copula_gof.qmd              │
     │ [3] notebooks/12_risk_sizing.qmd             │
     │ [4] notebooks/80_rl_ablation.qmd             │
     └──────────┬───────────────────────────────────┘
                │ (wait for all)
                ▼
     ┌────────────────────────────┐
     │ Validate Generated Tables  │  (LaTeX syntax check)
     └──────────┬─────────────────┘
                │
                ▼
     ┌──────────────────────────────────────────────┐
     │ PARALLEL: Compile Individual Chapters (j=4)  │
     ├──────────────────────────────────────────────┤
     │ [1] chapter_4_baseline_modeling.tex          │
     │ [2] chapter_5_rl_design.tex                  │
     │ [3] chapter_6_uncertainty_risk_betting.tex   │
     │ [4] chapter_7_simulation.tex                 │
     └──────────┬───────────────────────────────────┘
                │ (wait for all)
                ▼
     ┌────────────────────────────┐
     │ Compile main.tex           │  (3 passes: pdflatex → bibtex → pdflatex × 2)
     └──────────┬─────────────────┘
                │
                ▼
     ┌────────────────────────────┐
     │ Validate PDF Output        │  (page count, TOC, references)
     └──────────┬─────────────────┘
                │
                ▼
     ┌────────────────────────────┐
     │ Generate Build Report      │
     └──────────┬─────────────────┘
                │
                ▼
       ┌───────────────────┐
       │ ✅ SUCCESS        │
       │ main.pdf ready    │
       └───────────────────┘
```

**Command**:
```bash
bash scripts/dissertation/parallel_build.sh --full
```

**Expected Duration**: 8-12 minutes (vs 45-60 minutes sequential)

#### Incremental Chapter Update
```
User: "Update Chapter 5 with new RL results"
  ↓
Academic Publishing Agent:
  1. Identifies dependencies: notebooks/80_rl_ablation.qmd
  2. Renders only affected notebook (30s)
  3. Validates tables: rl_vs_baseline_table.tex, ess_table.tex
  4. Compiles Chapter 5 standalone (45s)
  5. Quick rebuild main.tex (60s)
  ↓
Total: ~2-3 minutes (vs ~8-10 minutes full build)
```

---

### 2. ETL Parallel Refresh Workflow

**Agent**: ETL Parallel Orchestrator Agent
**Trigger**: Weekly data refresh or user request

#### Full Data Pipeline
```
┌─────────────────────────────────────────────────────────────┐
│ START: Full ETL Refresh                                     │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
     ┌─────────────────────┐
     │ Build Dependency DAG │
     └──────────┬───────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────┐
│ LEVEL 0: Independent Sources (PARALLEL j=3)                   │
├───────────────────────────────────────────────────────────────┤
│ [M4 CPU] nflverse_schedules    (R/ingestion/ingest_schedules.R)│
│ [M4 CPU] nflverse_players      (R/backfill_rosters.R)         │
│ [M4 CPU] stadium_reference     (db/seeds/stadiums.sql)        │
└──────────┬────────────────────────────────────────────────────┘
           │ (wait for schedules - critical path)
           ▼
┌───────────────────────────────────────────────────────────────┐
│ LEVEL 1: Depends on Schedules (PARALLEL j=3)                  │
├───────────────────────────────────────────────────────────────┤
│ [M4 CPU]  nflverse_pbp        (R/ingestion/ingest_pbp.R)      │
│ [4090 GPU] odds_history       (py/ingest_odds_history.py)     │
│ [M4 CPU]  weather_fetch       (py/weather_meteostat.py)       │
└──────────┬────────────────────────────────────────────────────┘
           │ (wait for pbp - critical path)
           ▼
┌───────────────────────────────────────────────────────────────┐
│ LEVEL 2: Feature Aggregation (PARALLEL j=2)                   │
├───────────────────────────────────────────────────────────────┤
│ [M4 CPU] epa_aggregation      (R/features/features_epa.R)     │
│ [M4 CPU] play_classification  (R/features/features_play_types.R)│
└──────────┬────────────────────────────────────────────────────┘
           │ (wait for all)
           ▼
     ┌────────────────────────────┐
     │ LEVEL 3: Feature Generation│  (Sequential - depends on all)
     │ asof_features_unified.py  │
     └──────────┬─────────────────┘
                │
                ▼
     ┌────────────────────────────┐
     │ Refresh Materialized Views │
     └──────────┬─────────────────┘
                │
                ▼
     ┌────────────────────────────┐
     │ Validate Data Consistency  │  (Referential integrity, no orphans)
     └──────────┬─────────────────┘
                │
                ▼
       ┌───────────────────┐
       │ ✅ SUCCESS        │
       │ Features ready    │
       └───────────────────┘
```

**Command**:
```bash
python etl/orchestration/parallel_executor.py etl/config/pipeline_dag.yaml
```

**Expected Duration**: 15-20 minutes (vs 60-90 minutes sequential)

---

### 3. Distributed Model Training Workflow

**Agent**: Distributed Training Coordinator Agent
**Trigger**: Research agent requests multi-model comparison

#### Ensemble Training Example
```
User: "Train GLM, XGBoost, RandomForest, NeuralNet for Chapter 4"
  ↓
Distributed Training Coordinator:
  1. Analyzes requirements
  2. Routes tasks based on hardware
  3. Launches parallel training jobs

┌────────────────────────────────────────────────────────────────┐
│ PARALLEL TRAINING (4 jobs)                                     │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│ ┌─────────────────┐              ┌─────────────────┐          │
│ │ M4 MacBook      │              │ RTX 4090        │          │
│ ├─────────────────┤              ├─────────────────┤          │
│ │                 │              │                 │          │
│ │ [Job 1]         │              │ [Job 3]         │          │
│ │ GLM Baseline    │              │ XGBoost (GPU)   │          │
│ │ R tidymodels    │              │ tree_method=    │          │
│ │ 2 hours         │              │ gpu_hist        │          │
│ │                 │              │ 1.5 hours       │          │
│ │ [Job 2]         │              │                 │          │
│ │ Random Forest   │              │ [Job 4]         │          │
│ │ n_jobs=8        │              │ Neural Network  │          │
│ │ 2.5 hours       │              │ PyTorch GPU     │          │
│ │                 │              │ 2 hours         │          │
│ └─────────────────┘              └─────────────────┘          │
│                                                                 │
└────────┬───────────────────────────────────────────────────────┘
         │ (wait for all to complete)
         ▼
  ┌──────────────────────┐
  │ Aggregate Results    │  (Collect metrics from all jobs)
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ Generate Comparison  │  (LaTeX table for dissertation)
  │ Table                │
  └──────────┬───────────┘
             │
             ▼
    ┌────────────────┐
    │ ✅ SUCCESS     │
    │ 4 models ready │
    └────────────────┘
```

**Command**:
```bash
bash scripts/training/parallel_ensemble_train.sh
```

**Expected Duration**: 3-4 hours (vs 12-16 hours sequential)

---

### 4. Hyperparameter Search Workflow

**Agent**: Distributed Training Coordinator Agent
**Trigger**: Research agent requests hyperparameter optimization

#### XGBoost Grid Search Example (324 configurations)
```
Parameter Grid:
  max_depth: [3, 5, 7, 9]               → 4 values
  learning_rate: [0.01, 0.05, 0.1]      → 3 values
  n_estimators: [100, 300, 500]         → 3 values
  subsample: [0.7, 0.8, 0.9]            → 3 values
  colsample_bytree: [0.7, 0.8, 0.9]     → 3 values

Total: 4 × 3 × 3 × 3 × 3 = 324 configurations

┌─────────────────────────────────────────────────────────────┐
│ Task Distribution (Hardware-Aware)                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ RTX 4090 (Priority - GPU accelerated):                      │
│   ├─ 250 configs (GPU training)                             │
│   ├─ 2 concurrent jobs                                      │
│   └─ ~30 hours wall time (125 iterations @ 15 min each)    │
│                                                              │
│ M4 MacBook (Fallback - CPU):                                │
│   ├─ 74 configs (CPU training)                              │
│   ├─ 4 concurrent jobs                                      │
│   └─ ~5 hours wall time (19 iterations @ 15 min each)      │
│                                                              │
└─────────┬───────────────────────────────────────────────────┘
          │ (parallel execution)
          ▼
   ┌──────────────────┐
   │ Aggregate 324    │
   │ Experiment       │
   │ Results          │
   └──────┬───────────┘
          │
          ▼
   ┌──────────────────┐
   │ Identify Best    │
   │ Configuration    │
   │ (by val AUC)     │
   └──────┬───────────┘
          │
          ▼
    ┌─────────────┐
    │ ✅ SUCCESS  │
    │ Best config │
    │ found       │
    └─────────────┘
```

**Command**:
```bash
python py/compute/parallel_hyperparam_search.py \
  --model-type XGBoost \
  --param-grid configs/hyperparam_grids/xgboost_grid.yaml \
  --parallel
```

**Expected Duration**: 1.5-2 days (vs 8-12 days sequential)

---

### 5. Database Migration Testing Workflow

**Agent**: DevOps Parallel Orchestrator Agent
**Trigger**: New migration needs validation before production

#### Parallel Multi-Environment Testing
```
New Migration: db/migrations/008_add_advanced_metrics.sql

┌──────────────────────────────────────────────────────────────┐
│ STEP 1: Snapshot Databases (PARALLEL j=3)                    │
├──────────────────────────────────────────────────────────────┤
│ [1] dev_local → dev_test_1        (2 min)                    │
│ [2] staging_clone → staging_test_1 (5 min)                   │
│ [3] prod_snapshot → prod_test_1    (8 min)                   │
└──────────┬───────────────────────────────────────────────────┘
           │ (wait for all snapshots)
           ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 2: Apply Migrations (PARALLEL j=3)                      │
├──────────────────────────────────────────────────────────────┤
│ [1] dev_test_1: Apply 008_add_advanced_metrics.sql  (1 min)  │
│ [2] staging_test_1: Apply 008_add_advanced_metrics.sql (1 min)│
│ [3] prod_test_1: Apply 008_add_advanced_metrics.sql  (2 min) │
└──────────┬───────────────────────────────────────────────────┘
           │ (check for errors)
           ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 3: Validate Schemas (PARALLEL j=3)                      │
├──────────────────────────────────────────────────────────────┤
│ [1] dev_test_1: Check table structure, indexes  (30s)        │
│ [2] staging_test_1: Check table structure, indexes (30s)     │
│ [3] prod_test_1: Check table structure, indexes (30s)        │
└──────────┬───────────────────────────────────────────────────┘
           │ (all pass?)
           ▼
┌──────────────────────────────────────────────────────────────┐
│ STEP 4: Run Integration Tests (PARALLEL j=3)                 │
├──────────────────────────────────────────────────────────────┤
│ [1] dev_test_1: ETL pipeline smoke test    (1 min)           │
│ [2] staging_test_1: Research query test    (1 min)           │
│ [3] prod_test_1: Performance benchmark     (2 min)           │
└──────────┬───────────────────────────────────────────────────┘
           │ (all pass?)
           ▼
    ┌──────────────────┐
    │ Cleanup Snapshots│  (PARALLEL j=3)
    └──────┬───────────┘
           │
           ▼
     ┌─────────────┐
     │ ✅ SAFE     │
     │ Apply to    │
     │ production  │
     └─────────────┘
```

**Command**:
```bash
python infrastructure/parallel_migration_executor.py \
  infrastructure/parallel_migration_test.yaml
```

**Expected Duration**: 4-5 minutes (vs 15-20 minutes sequential)

---

## 🔄 End-to-End Research Workflow Example

**Scenario**: User requests full dissertation update with new 2025 Week 10 data

```
┌─────────────────────────────────────────────────────────────────┐
│ USER REQUEST: "Update dissertation with Week 10 data"          │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
     ┌─────────────────────┐
     │ Orchestrator        │  (Determines workflow)
     │ Analyzes Request    │
     └──────────┬──────────┘
                │
                ├─────────────────┬──────────────────┬─────────────┐
                ▼                 ▼                  ▼             ▼
        ┌───────────────┐  ┌──────────────┐  ┌──────────┐  ┌────────────┐
        │ ETL Parallel  │  │ Distributed  │  │ Academic │  │ DevOps     │
        │ Orchestrator  │  │ Training     │  │ Publishing│  │ Parallel   │
        │ Agent         │  │ Coordinator  │  │ Agent    │  │ Orchestrator│
        └───────┬───────┘  └──────┬───────┘  └────┬─────┘  └──────┬─────┘
                │                 │                │               │
                │                 │                │               │

PHASE 1: DATA REFRESH (ETL - 15 min)
├─ Level 0: nflverse_schedules, players (parallel)
├─ Level 1: nflverse_pbp, odds, weather (parallel)
├─ Level 2: EPA aggregation (parallel)
└─ Level 3: asof_features_unified.py

PHASE 2: MODEL UPDATES (Training - 3 hours, parallel with Phase 3)
├─ M4: GLM retrain with Week 10 data
├─ 4090: XGBoost retrain with Week 10 data
├─ M4: RandomForest retrain
└─ 4090: NeuralNet retrain

PHASE 3: ANALYSIS NOTEBOOKS (Academic - 10 min, parallel with Phase 2)
├─ Render 04_score_validation.qmd
├─ Render 05_copula_gof.qmd
├─ Render 12_risk_sizing.qmd
└─ Render 80_rl_ablation.qmd

PHASE 4: DISSERTATION BUILD (Academic - 8 min)
├─ Validate LaTeX tables
├─ Compile chapters 4, 5, 6, 7 (parallel)
└─ Build main.tex

                │
                ▼
       ┌────────────────┐
       │ ✅ COMPLETE    │
       │ Updated PDF    │
       │ with Week 10   │
       │ results        │
       └────────────────┘

Total Time: ~3.5 hours (vs ~16 hours sequential)
```

---

## 🛠 Practical Examples

### Example 1: Quick Chapter Update (Dissertation)

**Scenario**: Updated RL OPE table needs integration into Chapter 5

```bash
# Traditional way (sequential): ~10 minutes
quarto render notebooks/80_rl_ablation.qmd           # 3 min
cd analysis/dissertation/chapter_5_rl_design
pdflatex chapter_5_rl_design.tex                     # 2 min
cd ../main
pdflatex main.tex                                     # 5 min
# Total: ~10 minutes

# Parallel way: ~2 minutes
bash scripts/dissertation/incremental_chapter_build.sh chapter_5_rl_design
# - Renders only affected notebook (parallel with validation)
# - Compiles only Chapter 5
# - Quick rebuild main.tex (with cached aux files)
# Total: ~2 minutes
```

### Example 2: Weekly ETL Refresh (Data)

**Scenario**: Monday morning data refresh after weekend games

```bash
# Traditional way (sequential): ~60 minutes
Rscript R/ingestion/ingest_schedules.R               # 2 min
Rscript R/ingestion/ingest_pbp.R                     # 15 min
python py/ingest_odds_history.py ...                 # 20 min
python py/weather_meteostat.py ...                   # 10 min
Rscript R/features/features_epa.R                    # 5 min
python py/features/asof_features_unified.py ...     # 8 min
# Total: ~60 minutes

# Parallel way: ~18 minutes
python etl/orchestration/parallel_executor.py etl/config/pipeline_dag.yaml
# - Level 0 (schedules, players): 2 min (parallel)
# - Level 1 (pbp, odds, weather): 15 min (parallel, limited by pbp)
# - Level 2 (EPA, play types): 3 min (parallel)
# - Level 3 (features): 8 min (sequential)
# Total: ~18 minutes (3.3x speedup)
```

### Example 3: Multi-Model Comparison (Research)

**Scenario**: Chapter 4 needs comparison of 5 models

```bash
# Traditional way (sequential): ~16 hours
python py/backtest/baseline_glm.py ...               # 3 hours
python py/backtest/xgb_classifier.py ...             # 4 hours
python py/backtest/rf_classifier.py ...              # 3 hours
python py/backtest/nn_classifier.py ...              # 4 hours
python py/backtest/ensemble.py ...                   # 2 hours
# Total: ~16 hours

# Parallel way: ~4 hours
bash scripts/training/parallel_ensemble_train.sh
# M4 jobs (parallel):
#   - GLM: 3 hours
#   - RandomForest: 3 hours
# 4090 jobs (parallel):
#   - XGBoost (GPU): 2.5 hours
#   - NeuralNet (GPU): 3.5 hours
# Ensemble (after all): 30 min
# Total: ~4 hours (4x speedup, limited by longest job)
```

---

## 📊 Performance Monitoring

### Tracking Parallel Execution

All parallel workflows log to:
```
logs/
├── dissertation/
│   ├── parallel_build_20251010_143000.json
│   └── performance_trends.csv
├── etl/
│   ├── parallel_run_20251010_060000.json
│   └── dag_execution_metrics.csv
├── training/
│   ├── parallel_search_20251010_100000.json
│   └── hardware_utilization.csv
└── devops/
    ├── migration_test_20251010_150000.json
    └── deployment_validation.csv
```

### Performance Dashboards

**ETL Performance**:
```bash
python etl/monitoring/performance_tracker.py --analyze --last-n 10
```

**Training Efficiency**:
```bash
python py/compute/experiment_tracker.py --platform-utilization --last-n-days 7
```

**Dissertation Build Times**:
```bash
python scripts/dissertation/analyze_build_times.py --last-n 20
```

---

## 🚨 Troubleshooting Parallel Workflows

### Common Issues

#### Issue 1: Resource Contention
**Symptom**: Parallel jobs slower than expected
**Cause**: Oversubscribed CPU/memory/GPU
**Solution**:
```bash
# Check resource usage
python etl/orchestration/resource_monitor.py --check-capacity

# Reduce parallelism
python etl/orchestration/parallel_executor.py --max-parallel 2
```

#### Issue 2: Dependency Deadlock
**Symptom**: Jobs waiting indefinitely
**Cause**: Circular dependencies in DAG
**Solution**:
```bash
# Validate DAG structure
python etl/orchestration/validate_dag.py etl/config/pipeline_dag.yaml

# Visualize dependencies
python etl/orchestration/visualize_dag.py --output dag_graph.png
```

#### Issue 3: Failed Parallel Job Doesn't Fail Whole Workflow
**Symptom**: Workflow continues despite critical failure
**Cause**: Missing `critical: true` flag
**Solution**:
```yaml
# In pipeline_dag.yaml
level_1_parallel:
  critical: true  # ← Add this
  tasks:
    - nflverse_pbp
```

---

## 📖 Best Practices

### 1. Design for Parallelism from the Start
- Identify independent tasks
- Minimize shared state
- Make operations idempotent

### 2. Monitor and Optimize
- Track execution times
- Identify bottlenecks
- Optimize critical path

### 3. Fail Gracefully
- Implement retries
- Provide fallback to sequential
- Clear error messages

### 4. Document Dependencies
- Maintain DAG configurations
- Update when adding new tasks
- Version control all configs

### 5. Test Parallel Workflows
- Validate on small datasets first
- Check for race conditions
- Ensure reproducibility

---

## 🎯 Quick Reference Commands

```bash
# Dissertation (full parallel build)
bash scripts/dissertation/parallel_build.sh --full

# Dissertation (incremental chapter)
bash scripts/dissertation/incremental_chapter_build.sh chapter_5_rl_design

# ETL (full parallel refresh)
python etl/orchestration/parallel_executor.py etl/config/pipeline_dag.yaml

# Training (parallel ensemble)
bash scripts/training/parallel_ensemble_train.sh

# Training (hyperparameter search)
python py/compute/parallel_hyperparam_search.py --model-type XGBoost --parallel

# DevOps (migration testing)
python infrastructure/parallel_migration_executor.py infrastructure/parallel_migration_test.yaml

# DevOps (parallel backups)
bash scripts/devops/parallel_backup.sh

# Performance analysis
python etl/monitoring/performance_tracker.py --analyze --last-n 10
```

---

## 🔗 Related Documentation

- [Academic Publishing Agent](../agent_context/SUBAGENT_ACADEMIC_PUBLISHING.md)
- [ETL Parallel Orchestrator Agent](../agent_context/SUBAGENT_ETL_PARALLEL_ORCHESTRATOR.md)
- [Distributed Training Coordinator Agent](../agent_context/SUBAGENT_DISTRIBUTED_TRAINING_COORDINATOR.md)
- [DevOps Parallel Orchestrator Agent](../agent_context/SUBAGENT_DEVOPS_PARALLEL_ORCHESTRATOR.md)
- [Subagent Coordination Framework](../agent_context/SUBAGENT_COORDINATION.md)

---

**Last Updated**: 2025-10-10
**Maintained By**: All Parallel Orchestrator Agents
**Questions?**: Review agent-specific documentation or handoff protocols
