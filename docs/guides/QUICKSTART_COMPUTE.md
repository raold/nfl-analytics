# Distributed Compute Infrastructure - Quick Start

This guide will get you up and running with the distributed compute infrastructure for GPU-accelerated NFL betting model training.

## Architecture Overview

```
┌─────────────────┐
│  Submit Script  │  (You: submit_sweep.py)
└────────┬────────┘
         │ submits tasks
         ▼
┌─────────────────┐
│  Redis Queue    │  (Priority queue: high → low)
└────────┬────────┘
         │ claims tasks
         ▼
┌─────────────────┬─────────────────┐
│  M4 Worker      │  RTX Worker     │  (Auto-detect device, claim by capability)
│  (priority ≥1)  │  (priority ≥5)  │
└────────┬────────┴────────┬────────┘
         │                 │
         ▼                 ▼
┌─────────────────────────────────┐
│      Model Registry             │  (Shared checkpoint storage)
└─────────────────────────────────┘
```

## Prerequisites

1. **Redis** (already installed via Homebrew)
2. **Python packages** (already installed):
   - redis
   - psutil
   - torch
   - pyyaml

## Step-by-Step Walkthrough

### 1. Start Redis Server

```bash
# Check if Redis is running
redis-cli ping
# Should return: PONG

# If not running, start it:
redis-server &
```

### 2. Test Infrastructure

Run the end-to-end test to verify everything works:

```bash
python py/compute/test_infrastructure.py
```

Expected output:
```
ALL TESTS PASSED ✓
Infrastructure is ready for production use!
```

### 3. Start Worker(s)

**On M4 MacBook** (current machine):
```bash
# Terminal 1: Start M4 worker
python py/compute/worker_enhanced.py \
    --worker-id macbook_m4 \
    --min-priority 1
```

**On RTX 5090 Workstation** (when available):
```bash
# Start RTX worker (claims high-priority tasks)
python py/compute/worker_enhanced.py \
    --worker-id rtx_gpu0 \
    --device cuda:0 \
    --min-priority 5
```

Worker will auto-detect device and log:
```
Worker macbook_m4 using device: Apple Silicon (MPS) (16 GB)
Worker macbook_m4 starting (min_priority=1)
```

### 4. Submit Tasks

**Option A: Submit Test Sweep**

```bash
# Terminal 2: Dry run first
python py/compute/submit_sweep.py \
    --config sweeps/test_sweep.yaml \
    --dry-run

# If looks good, submit for real
python py/compute/submit_sweep.py \
    --config sweeps/test_sweep.yaml
# Confirm with 'y' when prompted
```

**Option B: Submit Custom Sweep**

```bash
python py/compute/submit_sweep.py \
    --model test \
    --base-config '{"sleep_per_epoch": 2}' \
    --param epochs 5 10 15 20 \
    --priority 3
```

**Option C: Submit Single Task**

```python
from compute.tasks.training_task import TrainingTask, get_redis_client

task = TrainingTask(
    model_type="test",
    config={"epochs": 10, "sleep_per_epoch": 1},
    priority=5
)
redis_client = get_redis_client()
task.submit(redis_client)
```

### 5. Monitor Progress

**Option A: Watch Queue**

```bash
# Terminal 3: Monitor queue in real-time
watch -n 1 'redis-cli zrange training_queue 0 -1 withscores'
```

**Option B: Check Worker Logs**

Worker logs show claimed tasks and progress:
```
✓ Worker macbook_m4 claimed task abc123 (test, priority 3)
Executing task abc123 (test)
  Epoch 1/10... loss=1.0000, acc=0.5500
  ...
✓ Task abc123 completed successfully
```

**Option C: Query Task Status**

```bash
# List all tasks
redis-cli keys 'task:*'

# Get specific task
redis-cli hgetall task:abc123
```

### 6. Inspect Results

**List completed runs:**

```bash
python py/compute/model_registry.py list test
```

**Show run details:**

```bash
python py/compute/model_registry.py show test abc123
```

**Find best run by metric:**

```bash
python py/compute/model_registry.py best test accuracy
```

**Export results to CSV:**

```bash
python py/compute/model_registry.py export test results.csv
```

## Common Workflows

### Workflow 1: Quick Test (5 minutes)

```bash
# Terminal 1: Start worker
python py/compute/worker_enhanced.py --worker-id test1 --min-priority 1

# Terminal 2: Submit single test task
python -c "
from compute.tasks.training_task import TrainingTask, get_redis_client
task = TrainingTask(model_type='test', config={'epochs': 3}, priority=5)
task.submit(get_redis_client())
"

# Worker logs will show task execution
# Check results in checkpoints/ directory
```

### Workflow 2: Hyperparameter Sweep

```bash
# 1. Create sweep config (see sweeps/test_sweep.yaml)

# 2. Dry run to estimate resources
python py/compute/submit_sweep.py --config sweeps/my_sweep.yaml --dry-run

# 3. Start workers (can run multiple)
python py/compute/worker_enhanced.py --worker-id macbook_m4 &
python py/compute/worker_enhanced.py --worker-id rtx_gpu0 --device cuda:0 &

# 4. Submit sweep
python py/compute/submit_sweep.py --config sweeps/my_sweep.yaml

# 5. Monitor until all tasks complete
watch -n 1 'redis-cli zcard training_queue'

# 6. Find best hyperparameters
python py/compute/model_registry.py best cql win_rate
```

### Workflow 3: Multi-Machine Setup

**M4 MacBook (low priority):**
```bash
python py/compute/worker_enhanced.py \
    --worker-id macbook_m4 \
    --min-priority 1 \
    --redis-host localhost
```

**RTX Workstation (high priority):**
```bash
# Update redis-host to M4's IP if running Redis on MacBook
python py/compute/worker_enhanced.py \
    --worker-id rtx_gpu0 \
    --device cuda:0 \
    --min-priority 5 \
    --redis-host 192.168.1.100
```

Now submit tasks from anywhere:
```bash
python py/compute/submit_sweep.py \
    --config sweeps/cql_sweep.yaml \
    --redis-host 192.168.1.100
```

Tasks will be automatically distributed:
- Priority 1-4: Claimed by M4
- Priority 5-10: Claimed by RTX (10x faster)

## Directory Structure

```
nfl-analytics/
├── py/
│   ├── compute/
│   │   ├── tasks/
│   │   │   └── training_task.py      # Task specification
│   │   ├── worker_enhanced.py         # Worker process
│   │   ├── model_registry.py          # Checkpoint management
│   │   ├── submit_sweep.py            # Sweep submission
│   │   └── test_infrastructure.py     # End-to-end test
│   └── rl/
│       └── cql_agent.py               # CQL implementation (TODO)
├── sweeps/
│   ├── test_sweep.yaml                # Example sweep config
│   └── cql_alpha_sweep.yaml           # CQL sweep (TODO)
├── models/
│   └── {model_type}/
│       └── {run_id}/
│           ├── metadata.json
│           ├── checkpoint_epoch_*.pth
│           ├── best_checkpoint.pth
│           └── metrics_history.jsonl
└── checkpoints/
    └── {task_id}_epoch*.pth           # Worker checkpoints
```

## Task Priority Guidelines

| Priority | Use Case | Workers | Example |
|----------|----------|---------|---------|
| 1-2 | Exploration | M4 only | Test new features |
| 3-4 | Development | M4 or RTX | Debug model code |
| 5-7 | Production | RTX preferred | Hyperparameter sweeps |
| 8-10 | Critical | RTX only | Final evaluation runs |

## Troubleshooting

### Worker Not Claiming Tasks

```bash
# Check queue has tasks
redis-cli zrange training_queue 0 -1 withscores

# Check worker min_priority matches task priority
# Worker with --min-priority 5 won't claim priority 3 tasks

# Check GPU memory requirement
# Task with min_gpu_memory=16 won't run on 8GB GPU
```

### Redis Connection Error

```bash
# Verify Redis is running
redis-cli ping

# If not, start it
redis-server &

# Check port (default 6379)
redis-cli -p 6379 ping
```

### No Module Named 'redis'

```bash
# Install missing dependencies
python -m pip install --break-system-packages redis psutil pyyaml torch
```

### Task Stuck in 'claimed' Status

```bash
# Worker may have crashed mid-execution
# Tasks don't auto-retry to prevent duplicates
# Manually resubmit if needed

# Or implement retry logic in worker
```

## Next Steps

1. **Implement CQL Agent**: `py/rl/cql_agent.py` (see GPU_PROFITABILITY_PLAN.md)
2. **Create CQL Sweep Config**: `sweeps/cql_alpha_sweep.yaml`
3. **Run Phase 1 Sweeps**: CQL/IQL hyperparameter optimization (Week 2-4)
4. **Setup RTX Workers**: Connect 2× RTX 5090 workstation
5. **Scale to Full Pipeline**: Phases 2-7 (uncertainty filtering, GNN, transformers)

## Resources

- **Full Plan**: `GPU_PROFITABILITY_PLAN.md` (21-page roadmap)
- **TODO List**: `analysis/dissertation/appendix/master_todos.tex`
- **Sweep Configs**: `sweeps/README.md`

## Performance Targets

From GPU_PROFITABILITY_PLAN.md:

| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| Win Rate | 51.0% | 52.4% | CQL/IQL + uncertainty filtering |
| ROI | -7.5% | +2-4% | Selective betting (top 20% confidence) |
| Brier | 0.2515 | <0.250 | Ensemble + calibration |
| Sharpe | 0.01 | >0.5 | Risk-aware position sizing |

Expected compute: **1,410 RTX hours + 210 M4 hours** over 12 weeks.
