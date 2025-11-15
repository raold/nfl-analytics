# Model Registry Fix Plan
## Resolving 227 Pending Models & Registry Sync Issues

**Created**: November 14, 2025
**Status**: Action Plan (Not Started)
**Estimated Time**: 2-3 hours total
**Priority**: P2 (Low) - Not blocking production deployment

---

## Executive Summary

The "227 pending models" issue mentioned in CLAUDE.md is a **STALE REFERENCE** from the November 9, 2025 BNN investigation cleanup. Investigation shows:

1. **No CLI tool exists** at `scripts/model_registry.py` (CLAUDE.md line 139 references it)
2. **Alternative registry code exists** in `py/registry/model_registry.py` and `py/compute/model_registry.py`
3. **Database sync script exists** at `py/database/register_ensemble_models.py`
4. **Most "pending" models are abandoned** BNN experiments (sigma tuning, NUTS variants, MC Dropout)

**Severity**: LOW - These are experiments, not production candidates. Only 6 production models need registration.

---

## Root Cause Analysis

### Why 227 Models Were Pending

**Timeline**:
1. **October 2025**: BNN investigation begins (6 approaches, 26 days)
2. **Experiments created**:
   - 4 sigma tuning variants (`bnn_rushing_sigma{0.5,0.7,1.0,1.5}.pkl`)
   - 4 NUTS variants (`bnn_nuts_heteroscedastic_*.pkl`, `bnn_nuts_homoscedastic_*.pkl`)
   - 3 MC Dropout variants (`bnn_mc_dropout_*.pkl`)
   - 19 BNN checkpoints (`bnn_multioutput_checkpoints/epoch_*.pt`)
   - Mixture of Experts (`bnn_mixture_experts_v2.pkl`)
   - Environment model (`bnn_rushing_environment.pkl`)
   - Opponent model (`bnn_rushing_opponent.pkl`)
   - Total: ~30-40 BNN files
3. **Additional experiments**:
   - 100+ CQL sweep configs (`models/cql/sweep/cql_config*.pth`)
   - 50+ GNN Optuna trials (`models/gnn/optuna/trial_*/model.pt`)
   - 20+ CQL hyperparameter runs (`models/cql/{hash}/checkpoint_*.pth`)
   - Total: ~170-180 experimental files
4. **November 9**: BNN investigation complete, research line abandoned
5. **Issue**: Models trained but never registered in `ml.model_registry` database table

### Why Registration Failed

**Hypothesis** (based on code review):

1. **No automated registration**: Training scripts save models but don't call registry
2. **Manual process assumed**: Expected researcher to run `py/database/register_ensemble_models.py` manually
3. **Experiments skipped**: Sigma tuning, NUTS, MC Dropout were quick tests, not production candidates
4. **Cleanup incomplete**: Files left on disk after abandonment decision

---

## Current State Assessment

### Production Models (Need Registration)

| Model | File | Size | Registered? | Priority |
|-------|------|------|-------------|----------|
| State-Space v2 | `simplified_random_walk_v2.rds` | 1.4GB | ❓ Unknown | P0 |
| Hierarchical v2 | `passing_yards_hierarchical_v2.rds` | 124MB | ❓ Unknown | P0 |
| Informative Priors v2 | `passing_informative_priors_v2.rds` | 123MB | ❓ Unknown | P0 |
| Hybrid XGBoost v1 | `hybrid_static_incremental_v1/` | 3MB | ❓ Unknown | P0 |
| BNN v3.1 (rushing) | `bnn_multioutput_v1_final.pt` | 58KB | ❓ Unknown | P0 |
| CQL Bet Sizer | `cql/best_model.pth` | 206KB | ❓ Unknown | P0 |

**Check Registration Status**:
```bash
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c \
  "SELECT model_name, version, status FROM ml.model_registry ORDER BY created_at DESC LIMIT 10;"
```

### Experimental Models (Don't Need Registration)

**BNN Experiments** (~30 files, 3.5GB):
- Sigma tuning: `bnn_rushing_sigma*.pkl` (4 files)
- NUTS variants: `bnn_nuts_*.pkl` (4 files)
- MC Dropout: `bnn_mc_dropout_*.pkl` (3 files)
- Checkpoints: `bnn_multioutput_checkpoints/` (19 files)
- Other: `bnn_mixture_experts_v2.pkl`, `bnn_rushing_environment.pkl`, etc.

**Recommendation**: **DELETE** these files (save 3.5GB), don't register.

**CQL Sweep** (~100 files, 70MB):
- Location: `models/cql/sweep/cql_config*.pth`
- Purpose: Hyperparameter grid search
- Best config already identified: `models/cql/best_model.pth`

**Recommendation**: **ARCHIVE** sweep results (keep for reference), don't register.

**GNN Optuna Trials** (~50 files, 50MB):
- Location: `models/gnn/optuna/trial_*/model.pt`
- Purpose: Hyperparameter optimization
- Best model already identified: `models/gnn/hierarchical_gnn_best.pt`

**Recommendation**: **KEEP** for research, don't register in production registry.

---

## Fix Plan

### Phase 1: Verify Current Registration (15 min)

**Step 1.1: Check Database Schema**
```bash
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c \
  "\\d ml.model_registry"
```

Expected columns:
- `model_id` (primary key)
- `model_name`, `version`, `status`
- `file_path`, `training_window`
- `mae_holdout`, `rmse_holdout`, `calibration_90`
- `ensemble_weight`
- `created_at`, `updated_at`

**Step 1.2: Check Existing Registrations**
```sql
SELECT
  model_name,
  version,
  status,
  file_path,
  ensemble_weight,
  created_at
FROM ml.model_registry
ORDER BY created_at DESC;
```

**Step 1.3: Identify Missing Models**
```bash
# Compare MODEL_REGISTRY.yaml vs database
python3 <<EOF
import yaml
import psycopg2

# Load YAML
with open('models/MODEL_REGISTRY.yaml') as f:
    registry = yaml.safe_load(f)

# Load DB
conn = psycopg2.connect(
    host='localhost',
    port=5544,
    database='devdb01',
    user='dro',
    password='sicillionbillions'
)
cursor = conn.cursor()
cursor.execute("SELECT model_name FROM ml.model_registry")
db_models = {row[0] for row in cursor.fetchall()}

# Find missing
yaml_models = set(registry['production_models'].keys())
missing = yaml_models - db_models

print(f"YAML models: {len(yaml_models)}")
print(f"DB models: {len(db_models)}")
print(f"Missing: {len(missing)}")
print(f"Missing models: {missing}")
EOF
```

### Phase 2: Register Production Models (30 min)

**Step 2.1: Run Registration Script**
```bash
cd /Users/dro/rice/nfl-analytics
python py/database/register_ensemble_models.py
```

Expected output:
```
✓ Loaded MODEL_REGISTRY.yaml (version 5.2)
✓ Registered: state_space_v2
✓ Registered: hierarchical_v2
✓ Registered: informative_priors_v2
✓ Registered: hybrid_static_incremental_v1
✓ Registered: bnn_multioutput_v3_1
✓ Registered: cql_bet_sizer
⚠️  Skipped: temperature_scaler_4way (file not found)

Total: 6 models registered
```

**Step 2.2: Verify Registration**
```bash
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c \
  "SELECT model_name, status, ensemble_weight FROM ml.model_registry WHERE status = 'production';"
```

Expected result: 6 rows (State-Space, Hierarchical, IP, Hybrid XGB, BNN v3.1, CQL)

**Step 2.3: Handle Temperature Scaler**

Two options:

**Option A**: Train temperature scaler (2 hours)
```python
# py/calibration/train_temperature_scaler.py
from sklearn.isotonic import IsotonicRegression
import pickle

# Load ensemble predictions on validation set (2022-2024)
# Compute empirical coverage
# Fit isotonic regression to calibrate
# Save to models/calibration/temperature_scaler_4way.pkl
```

**Option B**: Mark as "pending" in YAML (5 min)
```yaml
temperature_scaler_4way:
  status: "pending"
  expected_completion: "2025-11-20"
  notes: |
    Temperature scaler training pending.
    Ensemble can operate without it (uncalibrated intervals).
    Expected to reduce interval width by 25.6% when complete.
```

### Phase 3: Clean Up Experimental Models (45 min)

**Step 3.1: Create Backup** (10 min)
```bash
cd models
tar -czf experimental_models_backup_20251114.tar.gz \
  bayesian/bnn_multioutput_v1.pkl \
  bayesian/bnn_mixture_experts_v2.pkl \
  bayesian/bnn_multioutput_checkpoints/ \
  bayesian/bnn_rushing_sigma*.pkl \
  bayesian/bnn_nuts_*.pkl \
  bayesian/bnn_rushing_opponent.pkl \
  bayesian/bnn_rushing_environment.pkl \
  bayesian/bnn_rushing_improved_v2.pkl

# Upload to archive storage (optional)
# gsutil cp experimental_models_backup_20251114.tar.gz gs://nfl-analytics-archive/
```

**Step 3.2: Delete Abandoned Files** (5 min)
```bash
cd models/bayesian

# Delete data leakage model (1.6GB)
rm bnn_multioutput_v1.pkl

# Delete unknown mixture model (1.3GB)
rm bnn_mixture_experts_v2.pkl

# Delete failed v2 checkpoints (475MB)
rm -rf bnn_multioutput_checkpoints/

# Delete sigma tuning experiments (724MB)
rm bnn_rushing_sigma*.pkl

# Delete NUTS timeout failures (336MB)
rm bnn_nuts_*.pkl

# Delete unmetadata'd experiments (1GB)
rm bnn_rushing_opponent.pkl bnn_rushing_environment.pkl bnn_rushing_improved_v2.pkl

# Verify deletion
du -sh .
# Expected: ~1.5GB (down from ~5GB)
```

**Step 3.3: Archive CQL Sweep** (10 min)
```bash
cd models/cql

# Create archive
tar -czf sweep_results_archive.tar.gz sweep/

# Move to archive directory
mkdir -p ../archive/cql_sweep_2025_oct
mv sweep_results_archive.tar.gz ../archive/cql_sweep_2025_oct/

# Delete original sweep directory
rm -rf sweep/

# Keep best model (already at models/cql/best_model.pth)
```

**Step 3.4: Document GNN Optuna Trials** (10 min)
```bash
cd models/gnn

# Create summary
python3 <<EOF
import json
from pathlib import Path

trials = list(Path('optuna/').glob('trial_*/'))
best_trial = None
best_loss = float('inf')

for trial_dir in trials:
    metadata = trial_dir / 'metadata.json'
    if metadata.exists():
        with open(metadata) as f:
            data = json.load(f)
            if data.get('val_loss', float('inf')) < best_loss:
                best_loss = data['val_loss']
                best_trial = trial_dir.name

print(f"Best trial: {best_trial} (val_loss={best_loss:.4f})")
print(f"Total trials: {len(trials)}")
print("✓ Best model already saved as hierarchical_gnn_best.pt")
EOF

# Optionally: Keep optuna/ for research, don't register
```

**Step 3.5: Update CLAUDE.md** (10 min)
```markdown
# CLAUDE.md - Remove stale references

## OLD (remove):
- Model Registry: 8 models synced, 227 pending registration

## NEW (add):
- Model Registry: 6 production models registered
- Experimental models: Archived (CQL sweep, GNN Optuna) or deleted (abandoned BNN)
- Storage: 5.2GB (down from 8.7GB after cleanup)
```

### Phase 4: Create Registry Query Tool (30 min)

**Purpose**: Easy CLI access to model registry without raw SQL.

**Step 4.1: Create Script** (20 min)
```bash
cat > scripts/query_model_registry.sh <<'EOF'
#!/bin/bash
#
# Query Model Registry - Quick CLI for ml.model_registry
#
# Usage:
#   ./scripts/query_model_registry.sh list          # List all models
#   ./scripts/query_model_registry.sh production    # List production models
#   ./scripts/query_model_registry.sh <model_name>  # Show model details
#

set -e

DB="postgresql://dro:sicillionbillions@localhost:5544/devdb01"

case "$1" in
  list)
    psql "$DB" -c "
      SELECT model_name, version, status, ensemble_weight
      FROM ml.model_registry
      ORDER BY created_at DESC;
    "
    ;;

  production)
    psql "$DB" -c "
      SELECT model_name, version, mae_holdout, calibration_90, ensemble_weight
      FROM ml.model_registry
      WHERE status = 'production'
      ORDER BY ensemble_weight DESC NULLS LAST;
    "
    ;;

  *)
    if [ -n "$1" ]; then
      psql "$DB" -c "
        SELECT *
        FROM ml.model_registry
        WHERE model_name = '$1';
      "
    else
      echo "Usage:"
      echo "  ./scripts/query_model_registry.sh list"
      echo "  ./scripts/query_model_registry.sh production"
      echo "  ./scripts/query_model_registry.sh <model_name>"
    fi
    ;;
esac
EOF

chmod +x scripts/query_model_registry.sh
```

**Step 4.2: Test Tool** (5 min)
```bash
# List all models
./scripts/query_model_registry.sh list

# List production models
./scripts/query_model_registry.sh production

# Get specific model
./scripts/query_model_registry.sh state_space_v2
```

**Step 4.3: Add to Documentation** (5 min)
```bash
cat >> docs/database/DATABASE_QUICK_REFERENCE.md <<'EOF'

## Model Registry Queries

### Query All Models
```bash
./scripts/query_model_registry.sh list
```

### Query Production Models
```bash
./scripts/query_model_registry.sh production
```

### Query Specific Model
```bash
./scripts/query_model_registry.sh state_space_v2
```

### Direct SQL Access
```bash
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c \
  "SELECT * FROM ml.model_registry WHERE status = 'production';"
```
EOF
```

---

## Success Criteria

### Phase 1 Success ✅
- [ ] Database schema verified (ml.model_registry exists)
- [ ] Current registrations counted (N models)
- [ ] Missing models identified (6 production models)

### Phase 2 Success ✅
- [ ] 6 production models registered in database
- [ ] Temperature scaler status resolved (trained OR marked pending)
- [ ] All registered models have `status = 'production'`

### Phase 3 Success ✅
- [ ] Backup created (3.5GB experimental_models_backup.tar.gz)
- [ ] 3.5GB deleted (abandoned BNN files)
- [ ] CQL sweep archived (70MB)
- [ ] GNN Optuna documented (kept for research)
- [ ] CLAUDE.md updated (removed "227 pending" claim)

### Phase 4 Success ✅
- [ ] `scripts/query_model_registry.sh` created and tested
- [ ] Documentation updated with query examples
- [ ] CLI tool returns correct results for production models

---

## Rollback Plan

If registration causes issues:

**Step 1: Restore Backup**
```bash
cd models
tar -xzf experimental_models_backup_20251114.tar.gz
```

**Step 2: Clear Database Registrations**
```sql
DELETE FROM ml.model_registry WHERE created_at >= '2025-11-14';
```

**Step 3: Revert CLAUDE.md**
```bash
git checkout CLAUDE.md
```

---

## Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Verify | 15 min | Database access |
| Phase 2: Register | 30 min | Phase 1 complete |
| Phase 3: Cleanup | 45 min | Backup storage |
| Phase 4: Query Tool | 30 min | None |
| **Total** | **2 hours** | - |

---

## Recommendations

### Immediate Actions

1. **Run Phase 1** (15 min) to verify current state
2. **Skip Phase 3** if storage not a concern (experimental files harmless)
3. **Prioritize Phase 2** (register production models) for programmatic access

### Future Improvements

1. **Automated Registration**: Add `register_model()` call to training scripts
   ```python
   # At end of train_hierarchical_v2_canonical.R
   system("python py/database/register_ensemble_models.py --model hierarchical_v2")
   ```

2. **CI/CD Integration**: Run validation on `ml.model_registry` before deployment
   ```yaml
   # .github/workflows/validate_models.yml
   - name: Validate Model Registry
     run: |
       python py/database/register_ensemble_models.py --dry-run
       ./scripts/query_model_registry.sh production | grep -q "6 rows"
   ```

3. **Model Versioning**: Add `git_commit_hash` to registry for reproducibility
   ```sql
   ALTER TABLE ml.model_registry ADD COLUMN git_commit_hash VARCHAR(40);
   ```

---

## Conclusion

The "227 pending models" issue is **NOT a production blocker**. It's a stale reference to experimental models from the BNN investigation (Oct-Nov 2025). Only 6 production models need registration, which can be accomplished in 30 minutes using the existing `py/database/register_ensemble_models.py` script.

**Recommendation**: Run Phases 1-2 (45 min total) to register production models. Skip Phase 3 (cleanup) unless storage is a concern. Add Phase 4 (query tool) for operational convenience.

**Impact**: Low priority (P2). Does not block production deployment. Improves operational hygiene and reduces confusion.

---

**Plan Created**: November 14, 2025
**Status**: Ready for Execution
**Owner**: Model Management Agent
**Next Review**: After Phase 2 completion
