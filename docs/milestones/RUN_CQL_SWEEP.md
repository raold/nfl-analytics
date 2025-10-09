# Running CQL Hyperparameter Sweep - Step-by-Step Guide

This guide walks through running your first CQL hyperparameter sweep from start to finish.

## Prerequisites Check

```bash
# 1. Verify Redis is running
redis-cli ping
# Should return: PONG

# 2. Verify dataset exists
ls -lh data/rl_logged.csv
# Should show: -rw-r--r--@ 327K Oct  7 16:55 data/rl_logged.csv

# 3. Verify Python packages
python -c "import torch, redis, psutil, yaml, pandas, numpy; print('All packages installed')"
# Should print: All packages installed
```

## Option 1: Quick Test (Recommended First)

### Step 1: Review the sweep configuration

```bash
cat sweeps/cql_quick_test.yaml
```

You should see:
- Model: cql
- Configs: 4 (2 alphas × 2 learning rates)
- Estimated time: ~2 RTX hours, ~20 M4 hours

### Step 2: Dry run to verify

```bash
python py/compute/submit_sweep.py \
    --config sweeps/cql_quick_test.yaml \
    --dry-run
```

Expected output:
```
============================================================
HYPERPARAMETER SWEEP SUMMARY
============================================================
Model Type: cql
Priority: 4
...
Total Configurations: 4
Total GPU-Hours: 2.0
...
[DRY RUN] Not submitting tasks
```

### Step 3: Start worker (Terminal 1)

```bash
# On M4 MacBook
python py/compute/worker_enhanced.py \
    --worker-id macbook_m4 \
    --min-priority 1

# Worker will log:
# Worker macbook_m4 using device: Apple Silicon (MPS) (16 GB)
# Worker macbook_m4 starting (min_priority=1)
```

**Keep this terminal open.** Worker will claim and execute tasks automatically.

### Step 4: Submit sweep (Terminal 2)

```bash
python py/compute/submit_sweep.py \
    --config sweeps/cql_quick_test.yaml
```

When prompted:
```
Submit this sweep? (y/N): y
```

Expected output:
```
Submitting 4 tasks for cql hyperparameter sweep...
✓ Submitted task abc123 (cql) with priority 4
✓ Submitted task def456 (cql) with priority 4
✓ Submitted task ghi789 (cql) with priority 4
✓ Submitted task jkl012 (cql) with priority 4
✓ Successfully submitted 4 tasks

Queue status after submission:
  Pending tasks: 4
  Priority breakdown: {4: 4}
```

### Step 5: Monitor progress (Terminal 3, optional)

```bash
# Watch queue size
watch -n 1 'redis-cli zcard training_queue'

# Or watch queue contents
watch -n 1 'redis-cli zrange training_queue 0 -1 withscores'
```

### Step 6: Watch worker logs (Terminal 1)

Worker will automatically:
1. Claim task from queue
2. Load dataset
3. Train CQL agent
4. Log progress every epoch
5. Save checkpoints
6. Save final model to registry
7. Claim next task

Expected logs:
```
✓ Worker macbook_m4 claimed task abc123 (cql, priority 4)
Executing task abc123 (cql)
Starting CQL training...
Config: alpha=0.5, lr=0.0001, epochs=50, batch_size=128
Loaded 1675 samples, state_dim=6, n_actions=4
Populated replay buffer with 1675 transitions
Epoch 1/50 | Loss: 1.6351 | TD: 0.2266 | CQL: 1.4085 | Q: -0.1259
...
Epoch 50/50 | Loss: 0.8234 | TD: 0.0543 | CQL: 0.7691 | Q: 0.2134
Saved checkpoint: checkpoints/abc123_epoch50.pth
CQL training completed. Model saved to registry: cql/abc123
✓ Task abc123 completed successfully
```

### Step 7: Analyze results (after all tasks complete)

```bash
# List all CQL runs
python py/compute/model_registry.py list cql

# Find best run by CQL loss
python py/compute/model_registry.py best cql cql_loss --minimize

# Show details of best run
python py/compute/model_registry.py show cql <run_id>

# Export all results to CSV
python py/compute/model_registry.py export cql results/cql_quick_test.csv
```

### Step 8: Analyze hyperparameters

```python
import pandas as pd
import json

# Load all run metadata
runs = []
registry = ModelRegistry("models")
for run_metadata in registry.list_runs("cql"):
    runs.append({
        "run_id": run_metadata["run_id"],
        "alpha": run_metadata["config"]["alpha"],
        "lr": run_metadata["config"]["lr"],
        "final_loss": run_metadata["latest_metrics"]["loss"],
        "final_cql_loss": run_metadata["latest_metrics"]["cql_loss"],
        "final_q_mean": run_metadata["latest_metrics"]["q_mean"],
    })

df = pd.DataFrame(runs)
print(df.sort_values("final_loss"))

# Group by alpha
print("\nResults by alpha:")
print(df.groupby("alpha")["final_loss"].agg(["mean", "std", "min"]))
```

## Option 2: Full Production Sweep

**⚠️ WARNING**: This will consume ~225 RTX hours or ~2,250 M4 hours!

### Only run after:
1. Quick test completes successfully
2. Results look reasonable
3. You've connected RTX 5090 workers
4. You've reviewed the cost estimates

### Steps:

```bash
# 1. Dry run
python py/compute/submit_sweep.py \
    --config sweeps/cql_alpha_sweep.yaml \
    --dry-run

# 2. Review estimates
# Total Configurations: 45
# Total GPU-Hours: 225.0
# Wall-Clock: ~112 hours on 2× RTX, ~11 days on 1× RTX

# 3. Start RTX workers (priority ≥5)
# On RTX workstation:
python py/compute/worker_enhanced.py \
    --worker-id rtx_gpu0 \
    --device cuda:0 \
    --min-priority 5

python py/compute/worker_enhanced.py \
    --worker-id rtx_gpu1 \
    --device cuda:1 \
    --min-priority 5

# 4. Submit sweep
python py/compute/submit_sweep.py \
    --config sweeps/cql_alpha_sweep.yaml

# Confirm with 'y'
```

## Troubleshooting

### Issue: Worker not claiming tasks

**Symptom**: Worker logs "No tasks available"

**Cause**: Task priority doesn't match worker min_priority

**Fix**:
```bash
# Check task priority
redis-cli zrange training_queue 0 -1 withscores

# If tasks have priority 4 but worker has min_priority 5:
# Option 1: Restart worker with lower min_priority
python py/compute/worker_enhanced.py --worker-id macbook_m4 --min-priority 1

# Option 2: Resubmit tasks with higher priority (edit sweep YAML first)
```

### Issue: Worker crashes during training

**Symptom**: Worker exits mid-task

**Check**:
```bash
# 1. Memory usage
htop

# 2. Disk space
df -h

# 3. Dataset accessible
ls -lh data/rl_logged.csv

# 4. Python packages
python -c "import torch; print(torch.__version__)"
```

**Fix**: Restart worker, task will remain in "claimed" state but can be manually reset

### Issue: Low GPU utilization

**Symptom**: Training slow, GPU usage <50%

**Cause**: Batch size too small or CPU bottleneck

**Fix**: Edit sweep config to increase batch_size:
```yaml
base_config:
  batch_size: 256  # Increase from 128
```

### Issue: NaN losses during training

**Symptom**: Loss becomes NaN after few epochs

**Cause**: Learning rate too high or numerical instability

**Fix**: Reduce learning rate in sweep:
```yaml
param_grid:
  lr: [1e-5, 5e-5, 1e-4]  # Remove 1e-3 if unstable
```

## Expected Results

After quick test (4 configs):

**Best configuration** (example):
```
alpha: 1.0
lr: 1e-4
final_loss: 0.82
final_cql_loss: 0.77
final_q_mean: 0.21
```

**Key metrics to compare**:
- Lower `final_loss` = better overall
- Lower `final_cql_loss` = more conservative (good for offline RL)
- `final_q_mean` should be moderate (not too high = overestimation)

**Typical ranges**:
- Loss: 0.5-2.0
- CQL loss: 0.4-1.5
- Q mean: -0.5 to 0.5 (CQL pushes this down)

## Next Steps After Sweep

1. **Identify best alpha**:
   - Group results by alpha
   - Find alpha with lowest loss
   - Check if Q-values are conservative

2. **Evaluate on test set**:
   ```bash
   python py/rl/cql_agent.py \
       --dataset data/rl_logged_test.csv \
       --load models/cql/<best_run_id>/best_checkpoint.pth \
       --evaluate
   ```

3. **Deploy best policy**:
   - Load best checkpoint
   - Run on live data
   - Compare win rate vs historical

4. **Implement Phase 2**:
   - Train ensemble of best models (10 seeds)
   - Use ensemble disagreement for uncertainty
   - Filter to top 20% confidence predictions

## Clean Up

After sweep completes:

```bash
# Remove temporary checkpoints
rm -rf checkpoints/

# Keep models directory (registry)
# models/ contains all run metadata and checkpoints

# Clear Redis queue (if needed)
redis-cli del training_queue
redis-cli keys 'task:*' | xargs redis-cli del
redis-cli keys 'checkpoint:*' | xargs redis-cli del
```

## Summary

**Quick test workflow:**
1. Start worker (Terminal 1)
2. Submit sweep (Terminal 2)
3. Monitor progress (Terminal 3, optional)
4. Wait for completion (~2 hours RTX, ~20 hours M4)
5. Analyze results
6. Pick best hyperparameters
7. Run full sweep with best settings

**Full sweep workflow:**
1. Validate quick test first
2. Start 2× RTX workers
3. Submit full sweep
4. Monitor over ~5 days (2× RTX) or ~2 weeks (1× RTX)
5. Analyze 45 configurations
6. Select optimal alpha for production

---

**Ready to begin!** Start with the quick test to validate everything works, then scale to full sweep.
