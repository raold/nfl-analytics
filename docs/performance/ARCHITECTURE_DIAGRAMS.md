# NFL Analytics - Architecture Diagrams (Mac M4 MPS Optimization)

**Date:** November 15, 2025
**Purpose:** Visual architecture diagrams for MPS optimization analysis

---

## Diagram 1: Current System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     NFL ANALYTICS SYSTEM ARCHITECTURE                       │
│                         (Current: Nov 15, 2025)                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                        │
├────────────────────────────────────────────────────────────────────────────┤
│  PostgreSQL 14 + TimescaleDB                                               │
│  ├─ public schema: games, plays, weather, injuries, odds_history          │
│  ├─ mart schema: player_game_stats, game_summary (materialized views)     │
│  ├─ ml schema: model_registry, training_runs                              │
│  └─ Optimizations (Nov 13): Pooling, caching, composite indexes           │
│     ├─ Connection pool: 50-100x faster queries                            │
│     ├─ Feature cache: 70-85% hit rate                                     │
│     └─ Query time: 200ms → 8ms (25x speedup)                              │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                      FEATURE ENGINEERING LAYER                              │
├────────────────────────────────────────────────────────────────────────────┤
│  Pandas DataFrames (CPU-bound)                                             │
│  ├─ Rolling windows (lag features) - 10-60s                                │
│  ├─ Group-by aggregations - 5-20s                                          │
│  ├─ Weather features, game context - 2-5s                                  │
│  └─ 13 ensemble-compatible features                                        │
│                                                                             │
│  ⚠️  BOTTLENECK: Single-threaded, no parallelization                       │
│  💡 MPS NOT APPLICABLE (Pandas is CPU-optimized)                           │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        MODEL TRAINING LAYER                                 │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────┐     │
│  │ R MODELS (CPU-Only, 74% of training time)                        │     │
│  ├──────────────────────────────────────────────────────────────────┤     │
│  │ 1. State-Space v2 (Stan/cmdstanr)         30-120 min │ Mac M4 CPU│     │
│  │ 2. Hierarchical v2 (brms)                 45-180 min │ Mac M4 CPU│     │
│  │ 3. Informative Priors v2 (brms)           20-90 min  │ Mac M4 CPU│     │
│  │                                                                   │     │
│  │ ❌ MPS NOT SUPPORTED (MCMC algorithms are CPU-only)              │     │
│  └──────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────┐     │
│  │ XGBOOST MODELS (CUDA Required)                                   │     │
│  ├──────────────────────────────────────────────────────────────────┤     │
│  │ 4. XGBoost v2 Canonical                   3-8 min   │ Win GPU   │     │
│  │ 5. Hybrid XGBoost (70% static + 30% incremental)    │           │     │
│  │    ├─ Static component:                  5-12 min  │ Win GPU   │     │
│  │    └─ Incremental component:             2-3 min   │ Mac M4 CPU│     │
│  │                                                                   │     │
│  │ ❌ MPS NOT SUPPORTED (XGBoost GPU requires CUDA)                 │     │
│  │ ⚠️  DISTRIBUTED COMPUTE REQUIRED (Windows RTX 4090)              │     │
│  └──────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────┐     │
│  │ PYTORCH MODELS (MPS-Compatible) ⚠️ UNDERUTILIZED                 │     │
│  ├──────────────────────────────────────────────────────────────────┤     │
│  │ 6. CQL Betting Agent                     60-120 min │ Mac M4 CPU│     │
│  │    ⚠️  Current: CPU (90 min)                        │ 5% MPS    │     │
│  │    ✅ With MPS: 35 min (2.6x faster)                │           │     │
│  │                                                                   │     │
│  │ 7. Evidential Regression                  10-30 min │ Mac M4 CPU│     │
│  │    ⚠️  Current: CPU (25 min)                        │ 10% MPS   │     │
│  │    ✅ With MPS: 12 min (2.1x faster)                │           │     │
│  │                                                                   │     │
│  │ 8. BNN Multioutput (not in ensemble)     20-60 min │ Mac M4 CPU│     │
│  │    ⚠️  Current: CPU (45 min)                        │ 5% MPS    │     │
│  │    ✅ With MPS: 22 min (2.0x faster)                │           │     │
│  │                                                                   │     │
│  │ ✅ MPS SUPPORTED BUT UNDERUTILIZED (5-10% usage)                 │     │
│  │ 💡 OPTIMIZATION OPPORTUNITY: Enable MPS → 2-3x speedup           │     │
│  └──────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  TOTAL TRAINING TIME: 305 min (CPU-only)                                   │
│  WITH MPS OPTIMIZATION: 255 min (16% reduction) ← PHASE 1 TARGET           │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        ENSEMBLE PREDICTION LAYER                            │
├────────────────────────────────────────────────────────────────────────────┤
│  4-Way Ensemble (enhanced_ensemble_v5.py)                                  │
│                                                                             │
│  Sequential Prediction (Single Game):                                      │
│  ├─ 1. State-Space v2:           50ms  (NumPy vectorized, CPU)            │
│  ├─ 2. Hierarchical v2:         180ms  (rpy2 subprocess) ⚠️ BOTTLENECK    │
│  ├─ 3. XGBoost v2:               20ms  (DMatrix prediction, CPU)           │
│  ├─ 4. Informative Priors v2:   150ms  (rpy2 subprocess) ⚠️ BOTTLENECK    │
│  └─ 5. Ensemble Combination:     10ms  (weighted avg + variance pooling)  │
│  TOTAL: 410ms per prediction                                               │
│                                                                             │
│  💡 OPTIMIZATION OPPORTUNITY: R Batching                                   │
│  Batch Prediction (100 games):                                             │
│  ├─ Database fetch:              50ms  (connection pool)                   │
│  ├─ State-Space v2 (batch):      80ms  (vectorized)                        │
│  ├─ Hierarchical v2 (batch):    300ms  (single R call) ← 60x faster       │
│  ├─ XGBoost v2 (batch):          50ms  (batch DMatrix)                     │
│  ├─ Informative Priors v2 (batch): 250ms (single R call) ← 60x faster     │
│  └─ Ensemble Combination:        20ms  (NumPy)                             │
│  TOTAL: 750ms for 100 predictions (7.5ms each) ← 55x SPEEDUP              │
│                                                                             │
│  ❌ MPS NOT APPLICABLE (all CPU/R operations)                              │
│  ✅ R BATCHING IS THE BIGGEST WIN (55x speedup)                            │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                      POST-PROCESSING PIPELINE                               │
├────────────────────────────────────────────────────────────────────────────┤
│  1. Temperature Scaling (T=0.5534)    5-15ms   (NumPy, CPU)               │
│  2. CQL Bet Sizing (+24% vs baseline) 10-30ms  (PyTorch, could use MPS)   │
│  3. Risk Gates Validation            2-10ms   (NumPy, CPU)                │
│  4. JSON Serialization               2-5ms    (CPU)                        │
│                                                                             │
│  💡 MINOR MPS OPPORTUNITY: CQL bet sizing (10-30ms → 5-15ms)               │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Diagram 2: Distributed Compute Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              DISTRIBUTED COMPUTE: Mac M4 + Windows RTX 4090                 │
│                    (SETI@home Pattern via Google Drive)                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│                       MAC M4 (COORDINATOR + WORKER)                         │
│  Hardware: M4 (10-core CPU, GPU with Metal), 64GB Unified RAM              │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  COORDINATOR ROLE:                                                          │
│  ├─ Task Scheduling (adaptive_scheduler.py)                                │
│  │   ├─ Multi-armed bandit (UCB1, Thompson Sampling)                       │
│  │   └─ Hardware-aware routing (task_router.py)                            │
│  ├─ Task Queue (SQLite-based)                                              │
│  ├─ Performance Tracking (PerformanceTracker)                              │
│  └─ Result Aggregation                                                     │
│                                                                             │
│  LOCAL WORKER ROLE (70% of development time):                              │
│  ├─ R Models (ALWAYS local, CPU-only)                                      │
│  │   ├─ State-Space v2:         30-120 min │ CPU │ 100% local             │
│  │   ├─ Hierarchical v2:        45-180 min │ CPU │ 100% local             │
│  │   └─ Informative Priors v2:  20-90 min  │ CPU │ 100% local             │
│  │                                                                          │
│  ├─ PyTorch Models (CURRENT: CPU, SHOULD BE: MPS)                          │
│  │   ├─ CQL Agent:              90 min → 35 min │ CPU → MPS ⚠️            │
│  │   ├─ DQN Agent:              75 min → 30 min │ CPU → MPS ⚠️            │
│  │   ├─ Evidential Regression:  25 min → 12 min │ CPU → MPS ⚠️            │
│  │   └─ BNN Multioutput:        45 min → 22 min │ CPU → MPS ⚠️            │
│  │                                                                          │
│  ├─ Ensemble Inference (ALWAYS local)                                      │
│  │   └─ 410ms per game (7.5ms with R batching)                             │
│  │                                                                          │
│  └─ Database Queries (ALWAYS local)                                        │
│      └─ Optimized with pooling + caching (Nov 13)                          │
│                                                                             │
│  TASK ROUTING DECISION (current):                                          │
│  ├─ Mac M4 Score for PyTorch: 0.70 (CPU-only assumption) ❌                │
│  ├─ Windows Score for PyTorch: 0.95 (CUDA) ✅                              │
│  └─ Decision: Route to Windows (30-90s sync overhead) ⚠️ SUBOPTIMAL        │
│                                                                             │
│  TASK ROUTING DECISION (with MPS):                                         │
│  ├─ Mac M4 Score for PyTorch: 0.88 (MPS-accelerated) ✅                    │
│  ├─ Windows Score for PyTorch: 0.95 (CUDA, but sync overhead) ⚠️           │
│  └─ Decision: Run locally (no sync, 2x faster than CPU) ✅ OPTIMAL         │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Google Drive Sync
                                    │ (Task queue, data, results)
                                    │ Overhead: 15-95s per task
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    WINDOWS RTX 4090 (GPU WORKER)                           │
│  Hardware: RTX 4090 (24GB VRAM), CUDA 12.x, 128GB RAM                     │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  WORKER ROLE (30% of compute time):                                        │
│  ├─ XGBoost GPU Training (CUDA REQUIRED)                                   │
│  │   ├─ XGBoost v2:             3-8 min   │ CUDA │ 100% remote            │
│  │   └─ Hybrid XGBoost (static): 5-12 min │ CUDA │ 100% remote            │
│  │                                                                          │
│  ├─ PyTorch GPU Training (CURRENT: 75%, SHOULD BE: 5%)                     │
│  │   ├─ CQL Agent:              20 min │ CUDA │ 75% remote ⚠️             │
│  │   ├─ DQN Agent:              18 min │ CUDA │ 75% remote ⚠️             │
│  │   └─ Evidential Regression:   8 min │ CUDA │ 10% remote ⚠️             │
│  │                                                                          │
│  └─ Hyperparameter Sweeps (parallel tasks)                                 │
│      └─ Optuna, grid search (10-100 tasks in parallel)                     │
│                                                                             │
│  POLLING WORKFLOW:                                                          │
│  1. Watch Google Drive for new tasks (every 5s)                            │
│  2. Download task config + data (10-60s)                                   │
│  3. Execute training job (3-120 min)                                       │
│  4. Upload trained model + metrics (5-30s)                                 │
│  5. Mark task as complete                                                  │
│                                                                             │
│  ⚠️  SYNC OVERHEAD: 15-95s per task (non-negligible)                       │
│  💡 WITH MPS: 40-60% of tasks can run locally (save sync overhead)         │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│                     SYNC LAYER (Google Drive)                              │
├────────────────────────────────────────────────────────────────────────────┤
│  Tasks:                                                                     │
│  ├─ Task configs (JSON, 1-10 KB)          Upload: 1-5s                     │
│  ├─ Training data (Pickled DataFrames)    Upload: 10-60s (10-50 MB)       │
│  └─ Trained models (.pkl, .rds)           Download: 5-30s (5-100 MB)      │
│                                                                             │
│  Reliability:                                                              │
│  ├─ Auto-retry on network failure (3 attempts)                             │
│  ├─ Checkpointing (resume interrupted tasks)                               │
│  └─ Timeout detection (mark stale tasks as failed after 4 hours)           │
└────────────────────────────────────────────────────────────────────────────┘

IMPACT OF MPS OPTIMIZATION:
├─ BEFORE: 60% of PyTorch tasks → Windows (sync overhead: 30-90s each)
├─ AFTER:  10% of PyTorch tasks → Windows (only large-scale sweeps)
└─ BENEFIT: 40-50% reduction in distributed sync overhead
            15-25% faster overall pipeline
            Simpler local-first development workflow
```

---

## Diagram 3: Training Pipeline Flow (Before vs After MPS)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│           TRAINING PIPELINE: BEFORE MPS OPTIMIZATION                        │
│                      (Current as of Nov 15, 2025)                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Database │───▶│ Features │───▶│ R Models │───▶│ XGBoost  │───▶│ PyTorch  │
│ Query    │    │ Engineer │    │ Training │    │ (Windows)│    │ (CPU)    │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
   5 min           15 min          180 min         7 min           90 min
   CPU             CPU             CPU             CUDA            CPU ⚠️

                                                   │              │
                                                   │              ▼
                                                   │         ┌──────────┐
                                                   │         │ Sync     │
                                                   │         │ Overhead │
                                                   │         └──────────┘
                                                   │           30-90s
                                                   │
                                                   └─────────▶┌──────────┐
                                                              │ Evaluate │
                                                              └──────────┘
                                                                  8 min
                                                                  CPU

TOTAL TIME: 305 minutes (~5 hours)
SYNC OVERHEAD: 30-90 seconds per XGBoost task
GPU UTILIZATION: Mac M4 MPS = 0%, Windows CUDA = 2.3% of total

┌─────────────────────────────────────────────────────────────────────────────┐
│           TRAINING PIPELINE: AFTER MPS OPTIMIZATION                         │
│                      (Projected with Phase 1)                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Database │───▶│ Features │───▶│ R Models │───▶│ XGBoost  │───▶│ PyTorch  │
│ Pooling  │    │ Caching  │    │ Training │    │ (Windows)│    │ (MPS)    │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
   2 min ✅       8 min ✅         180 min         7 min           40 min ✅
   CPU             CPU             CPU             CUDA            MPS

                                                   │              │
                                                   │              ▼
                                                   │         ┌──────────┐
                                                   │         │ No Sync  │
                                                   │         │ (Local)  │
                                                   │         └──────────┘
                                                   │           0s ✅
                                                   │
                                                   └─────────▶┌──────────┐
                                                              │ Evaluate │
                                                              └──────────┘
                                                                  5 min ✅
                                                                  CPU

TOTAL TIME: 242 minutes (~4 hours) ← 21% REDUCTION
SYNC OVERHEAD: 0 seconds (PyTorch runs locally)
GPU UTILIZATION: Mac M4 MPS = 16.5%, Windows CUDA = 2.9%

IMPROVEMENTS:
✅ Database queries: 5 min → 2 min (2.5x, Nov 13 optimization)
✅ Feature engineering: 15 min → 8 min (1.9x, Nov 13 caching)
✅ PyTorch training: 90 min → 40 min (2.25x, MPS acceleration)
✅ Evaluation: 8 min → 5 min (1.6x, parallel metrics)
✅ Sync overhead: 30-90s → 0s (local execution)
❌ R models: 180 min → 180 min (unchanged, CPU-only)
❌ XGBoost: 7 min → 7 min (unchanged, keep CUDA)
```

---

## Diagram 4: Inference Pipeline (Before vs After R Batching)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│        ENSEMBLE INFERENCE: SEQUENTIAL (Current)                             │
│                    Single Game Prediction                                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐
│ Game        │
│ Request     │
└─────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Database Fetch Features                                    20-50ms          │
│ ├─ Query: player_id, season, week                                          │
│ └─ Returns: 13 ensemble features                                            │
└─────────────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Model 1: State-Space v2 (NumPy)                           30-80ms          │
│ ├─ Vectorized computation (CPU)                                            │
│ └─ Returns: (mean, std)                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Model 2: Hierarchical v2 (rpy2) ⚠️ BOTTLENECK              100-200ms       │
│ ├─ Python → R subprocess call                                              │
│ ├─ Load brms model (cached)                                                │
│ ├─ Bayesian posterior sampling (1000 draws)                                │
│ └─ Returns: (mean, std)                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Model 3: XGBoost v2 (DMatrix)                              10-30ms         │
│ ├─ Construct DMatrix (CPU)                                                 │
│ ├─ Tree inference (fast)                                                   │
│ └─ Returns: (mean, fixed_std)                                              │
└─────────────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Model 4: Informative Priors v2 (rpy2) ⚠️ BOTTLENECK       100-200ms       │
│ ├─ Python → R subprocess call                                              │
│ ├─ Load brms model (cached)                                                │
│ ├─ Bayesian posterior sampling (1000 draws)                                │
│ └─ Returns: (mean, std)                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Ensemble Combination (NumPy)                               5-15ms          │
│ ├─ Weighted average: ensemble_mean = Σ(w_i × μ_i)                          │
│ ├─ Variance pooling: σ² = within_var + between_var                         │
│ └─ Temperature scaling: σ_calib = σ × √T                                   │
└─────────────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────┐
│ Response    │
│ (JSON)      │
└─────────────┘

TOTAL LATENCY: 250-550ms per game
BOTTLENECK: rpy2 overhead (200-400ms = 36-73% of total)
❌ FAILS RISK GATE: Target <500ms, actual 250-550ms (marginal)

┌─────────────────────────────────────────────────────────────────────────────┐
│        ENSEMBLE INFERENCE: BATCHED (Optimized)                              │
│                 Batch of 100 Games                                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐
│ Batch       │
│ Request     │
│ (100 games) │
└─────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Database Batch Fetch (connection pool)                     50ms            │
│ ├─ Single query with WHERE player_id IN (...)                              │
│ └─ Returns: 100 × 13 features                                              │
└─────────────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Model 1: State-Space v2 Batch (vectorized NumPy)           80ms ✅         │
│ ├─ Skills lookup (vectorized)                                              │
│ ├─ Matrix operations (BLAS-optimized)                                      │
│ └─ Returns: 100 × (mean, std)                                              │
└─────────────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Model 2: Hierarchical v2 Batch (single R call) ✅          300ms ← 60x ✅  │
│ ├─ Python → R with DataFrame (100 rows)                                    │
│ ├─ Single brms posterior_predict() call                                    │
│ ├─ Vectorized posterior sampling                                           │
│ └─ Returns: 100 × (mean, std)                                              │
└─────────────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Model 3: XGBoost v2 Batch (batch DMatrix)                  50ms ✅         │
│ ├─ Construct DMatrix for all rows                                          │
│ ├─ Batch tree inference                                                    │
│ └─ Returns: 100 × (mean, fixed_std)                                        │
└─────────────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Model 4: Informative Priors v2 Batch (single R call) ✅    250ms ← 60x ✅  │
│ ├─ Python → R with DataFrame (100 rows)                                    │
│ ├─ Single brms posterior_predict() call                                    │
│ ├─ Vectorized posterior sampling                                           │
│ └─ Returns: 100 × (mean, std)                                              │
└─────────────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Ensemble Combination (vectorized NumPy)                    20ms ✅         │
│ ├─ Broadcast operations for all 100 games                                  │
│ ├─ Variance pooling (vectorized)                                           │
│ └─ Temperature scaling (vectorized)                                        │
└─────────────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────┐
│ Response    │
│ (JSON array)│
└─────────────┘

TOTAL LATENCY: 750ms for 100 games = 7.5ms per game ← 55x SPEEDUP
BOTTLENECK ELIMINATED: R overhead 300ms (40% of total, down from 73%)
✅ PASSES RISK GATE: 7.5ms << 500ms target

BACKTEST IMPACT (4,440 predictions):
├─ BEFORE: 410ms × 4,440 = 30.3 minutes
├─ AFTER:  7.5ms × 4,440 = 33 seconds
└─ SPEEDUP: 55x faster (from 30 min to 33 sec)
```

---

## Diagram 5: MPS Device Placement Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              PYTORCH MODEL: DEVICE PLACEMENT STRATEGY                       │
│                    (CQL Agent Example)                                      │
└─────────────────────────────────────────────────────────────────────────────┘

INITIALIZATION PHASE
┌─────────────────────────────────────────────────────────────────────────────┐
│ def get_device(device_arg: str = "auto") -> torch.device:                  │
│     if device_arg == "auto":                                                │
│         if torch.cuda.is_available():                                       │
│             return torch.device("cuda")      # Priority 1: CUDA             │
│         elif torch.backends.mps.is_available():                             │
│             return torch.device("mps")       # Priority 2: MPS ← NEW        │
│         else:                                                               │
│             return torch.device("cpu")       # Priority 3: CPU              │
│     # ... manual device selection                                           │
│                                                                             │
│ device = get_device()  # Auto-select best available device                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
MODEL INITIALIZATION
┌─────────────────────────────────────────────────────────────────────────────┐
│ q_network = QNetwork(state_dim, n_actions, hidden_dims)                    │
│ q_network.to(device)  # Move model to selected device                      │
│                                                                             │
│ target_network = QNetwork(state_dim, n_actions, hidden_dims)               │
│ target_network.to(device)                                                  │
│                                                                             │
│ optimizer = torch.optim.Adam(q_network.parameters(), lr=3e-4)              │
│ # Optimizer automatically tracks device from model                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
TRAINING LOOP (KEEP DATA ON DEVICE)
┌─────────────────────────────────────────────────────────────────────────────┐
│ for batch in dataloader:                                                    │
│     # Move batch to device (ONE-TIME TRANSFER)                             │
│     states = batch['states'].to(device)        # CPU → MPS (5-10ms)        │
│     actions = batch['actions'].to(device)                                  │
│     rewards = batch['rewards'].to(device)                                  │
│     next_states = batch['next_states'].to(device)                          │
│     dones = batch['dones'].to(device)                                      │
│                                                                             │
│     # ALL OPERATIONS STAY ON DEVICE (NO TRANSFERS)                         │
│     q_values = q_network(states)               # MPS (10-30ms)             │
│     next_q_values = target_network(next_states) # MPS (10-30ms)            │
│                                                                             │
│     # Loss computation (stays on device)                                   │
│     td_targets = rewards + (1 - dones) * gamma * next_q_values.max(1)[0]  │
│     td_errors = q_values.gather(1, actions) - td_targets                   │
│     loss = (td_errors ** 2).mean()                                         │
│                                                                             │
│     # Backpropagation (all on device)                                      │
│     optimizer.zero_grad()                                                  │
│     loss.backward()                                                        │
│     optimizer.step()                                                       │
│                                                                             │
│     # ONLY TRANSFER TO CPU FOR LOGGING (minimize)                          │
│     if step % 100 == 0:                                                    │
│         loss_cpu = loss.cpu().item()  # MPS → CPU (2-5ms)                 │
│         print(f"Loss: {loss_cpu:.4f}")                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
EVALUATION PHASE (MINIMAL TRANSFERS)
┌─────────────────────────────────────────────────────────────────────────────┐
│ q_network.eval()  # Set to evaluation mode                                 │
│                                                                             │
│ with torch.no_grad():  # Disable gradient computation                      │
│     for eval_batch in eval_dataloader:                                     │
│         states = eval_batch['states'].to(device)  # CPU → MPS              │
│         q_values = q_network(states)              # MPS                    │
│                                                                             │
│         # Only transfer results to CPU for metrics                         │
│         q_values_cpu = q_values.cpu().numpy()     # MPS → CPU              │
│         # Calculate metrics on CPU (scikit-learn)                          │
│         mae = mean_absolute_error(y_true, q_values_cpu)                    │
└─────────────────────────────────────────────────────────────────────────────┘

DATA FLOW SUMMARY:
┌─────────────┐     to(device)    ┌─────────────┐     Operations   ┌──────────┐
│   CPU       │ ─────────────────▶│     MPS     │ ────────────────▶│   MPS    │
│ (NumPy)     │    5-20ms         │  (Tensors)  │     10-50ms      │ (Model)  │
└─────────────┘                   └─────────────┘                  └──────────┘
                                                                          │
                                                                          │
                                                                          ▼
                                                  ┌──────────────────────────┐
                                                  │   STAY ON MPS            │
                                                  │   (entire training loop) │
                                                  └──────────────────────────┘
                                                          │
                                                          │ .cpu().item()
                                                          │ (only for logging)
                                                          ▼
                                                  ┌──────────────┐
                                                  │   CPU        │
                                                  │ (Logging)    │
                                                  └──────────────┘

PERFORMANCE IMPACT:
├─ Transfer overhead: 5-20ms per batch (unavoidable)
├─ Computation speedup: 20-100ms per batch (2-3x faster)
└─ NET BENEFIT: +10-60ms per batch (1.5-2x overall)

OPTIMIZATION TIPS:
✅ Use larger batch sizes (256-512) to amortize transfer overhead
✅ Keep data on device as long as possible (entire training loop)
✅ Only transfer to CPU for logging/metrics (minimize)
❌ Avoid ping-ponging between CPU and MPS (kills performance)
```

---

## Diagram 6: Optimization Priority Matrix

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   OPTIMIZATION PRIORITY MATRIX                              │
│              (Impact vs Effort for Mac M4 MPS Integration)                 │
└─────────────────────────────────────────────────────────────────────────────┘

     HIGH IMPACT
         ▲
         │
         │  ┌────────────────────┐
         │  │ R BATCHING API     │ ← BIGGEST WIN (but high effort)
         │  │ • 55x speedup      │   Estimated: 8-12 hours
         │  │ • Inference 30→0.5│   Risk: High (rpy2 refactoring)
    85%  │  │ • Backtest 45→18  │   Priority: PHASE 3
         │  └────────────────────┘
         │
         │
         │                          ┌───────────────────┐
         │                          │ ENABLE MPS (CQL)  │ ← QUICK WIN
         │                          │ • 2.6x speedup    │   Estimated: 30 min
    50%  │                          │ • CQL 90→35 min   │   Risk: Low
         │                          │ • Local-first dev │   Priority: PHASE 1
         │                          └───────────────────┘
         │
         │  ┌──────────────────┐
         │  │ TASK ROUTER      │   Estimated: 4-6 hours
    35%  │  │ • 15-25% sync↓   │   Risk: Medium
         │  │ • Smarter routing│   Priority: PHASE 2
         │  └──────────────────┘
         │
         │                      ┌────────────────┐
         │                      │ ENABLE MPS     │   Estimated: 20 min each
    20%  │                      │ (DQN, Evid)    │   Risk: Low
         │                      │ • 1.5-2x speed │   Priority: PHASE 1
         │                      └────────────────┘
         │
         │
    LOW IMPACT
         │
         └────────────────────────────────────────────────────────────────▶
            LOW EFFORT                                         HIGH EFFORT

LEGEND:
├─ 🟢 GREEN ZONE (Quick Wins): High impact, low effort → DO FIRST
├─ 🟡 YELLOW ZONE (Strategic): High impact, high effort → DO LATER
├─ 🔵 BLUE ZONE (Nice-to-Have): Low impact, low effort → DO IF TIME
└─ 🔴 RED ZONE (Avoid): Low impact, high effort → DON'T DO

OPTIMIZATION ROADMAP:
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 1 (Quick Wins): Enable MPS in PyTorch Models              │
│ ├─ CQL Agent (30 min, 2.6x speedup) 🟢                          │
│ ├─ Evidential Regression (20 min, 2.1x speedup) 🟢              │
│ ├─ DQN Agent (30 min, 2.5x speedup) 🟢                          │
│ └─ BNN Multioutput (20 min, 2.0x speedup) 🟢                    │
│ Total: 2-4 hours, 2-3x speedup on 40-50% of workload            │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 2 (Strategic): Optimize Task Router                       │
│ ├─ Update TaskCharacteristics for MPS (1 hour)                  │
│ ├─ Add MPS scoring logic (2 hours)                              │
│ ├─ Test distributed workflow (1-2 hours)                        │
│ └─ Validate routing decisions (1 hour)                          │
│ Total: 4-6 hours, 15-25% reduction in sync overhead             │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 3 (High-Impact Refactor): R Batching Integration          │
│ ├─ Verify r_model_interface_batched.py (1 hour)                 │
│ ├─ Refactor enhanced_ensemble_v5.py (3-4 hours)                 │
│ ├─ Test batch vs sequential (2-3 hours)                         │
│ ├─ Update historical_validation_4way.py (2 hours)               │
│ └─ Production validation (1-2 hours)                            │
│ Total: 8-12 hours, 55x speedup on ensemble inference            │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 4 (Validation): Production Readiness                      │
│ ├─ Full 2022-2024 backtest (2 hours)                            │
│ ├─ Risk gate validation (1 hour)                                │
│ ├─ A/B test MPS vs CUDA (1 hour)                                │
│ └─ Deployment preparation (1-2 hours)                           │
│ Total: 4-6 hours, production-ready deployment                   │
└──────────────────────────────────────────────────────────────────┘

TOTAL TIMELINE: 18-28 hours
EXPECTED BENEFIT: 2-3x overall speedup, 40-50% less Windows dependency
```

---

**END OF DIAGRAMS**

**Related Documents:**
- [ARCHITECTURE_COMPUTE_ANALYSIS.md](ARCHITECTURE_COMPUTE_ANALYSIS.md) - Full analysis (13,500 lines)
- [MPS_OPTIMIZATION_QUICKSTART.md](MPS_OPTIMIZATION_QUICKSTART.md) - Quick start guide
