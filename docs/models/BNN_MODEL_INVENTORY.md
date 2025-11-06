# BNN Model Inventory

**Last Updated:** November 3, 2025
**Total BNN Models:** 35 files (5.2 GB total)
**Purpose:** Comprehensive inventory of all Bayesian Neural Network models

---

## Production Models (3 models, 290M)

| Model | Size | Date | Status | Description |
|-------|------|------|--------|-------------|
| `bnn_multioutput_v1_final.pt` | 58 KB | Nov 2, 2025 | ✅ Production | PyTorch multi-output BNN (passing/rushing yards), MAE: 20.2 yards, 89.5% calibration |
| `bnn_passing_v1.pkl` | 80 MB | Oct 16, 2024 | ✅ Production | PyMC BNN for passing yards, MAE: 58.70 yards, 86.8% calibration |
| `receiving_qb_chemistry_v1.rds` | 210 MB | Nov 2, 2025 | ✅ Production | QB-WR dyadic effects model, 2,168 QB-WR pairs |

**Database Registration:**
- All production models registered in PostgreSQL `ml.model_registry`
- Query: `SELECT * FROM ml.model_registry WHERE is_production = true AND model_type = 'bnn';`

---

## R Hierarchical Models (2 models, 27M)

| Model | Size | Date | Type | Description |
|-------|------|------|------|-------------|
| `passing_informative_priors_v1.rds` | 5.2 MB | Oct 24, 2025 | brms/Stan | Informative priors model, 25.3s training |
| `passing_yards_hierarchical_v1.rds` | 22 MB | Oct 24, 2025 | brms/Stan | v1.0 baseline hierarchical model |

---

## Experimental BNN Models (14 models, 4.9 GB)

### Large Experimental Models (>100M)
| Model | Size | Date | Focus | Status |
|-------|------|------|-------|--------|
| `bnn_multioutput_v1.pkl` | 1.6 GB | Oct 21, 2024 | Multi-output (old version) | ⚠️ **DEPRECATED** - Use `bnn_multioutput_v1_final.pt` instead |
| `bnn_mixture_experts_v2.pkl` | 1.3 GB | Oct 19, 2024 | Mixture of experts | 🧪 Experimental |
| `bnn_rushing_opponent.pkl` | 372 MB | Oct 16, 2024 | Opponent-specific features | 🧪 Experimental |
| `bnn_rushing_improved_v2.pkl` | 343 MB | Oct 16, 2024 | Improved rushing model | 🧪 Experimental |
| `bnn_rushing_environment.pkl` | 302 MB | Oct 16, 2024 | Environmental features | 🧪 Experimental |
| `bnn_rushing_vegas_v1.pkl` | 298 MB | Oct 16, 2024 | Vegas-derived features | ⚠️ **CONTAINS LEAKAGE** - Do not use |

### Sigma Parameter Sweep (4 models, 724M)
| Model | Size | Sigma | Coverage |
|-------|------|-------|----------|
| `bnn_rushing_sigma1.5.pkl` | 181 MB | 1.5 | Wide intervals |
| `bnn_rushing_sigma1.0.pkl` | 181 MB | 1.0 | Baseline |
| `bnn_rushing_sigma0.7.pkl` | 181 MB | 0.7 | Medium |
| `bnn_rushing_sigma0.5.pkl` | 181 MB | 0.5 | Narrow intervals |

**Note:** Sigma sweep superseded by temperature scaling (T=0.75) in production.

### Other Experimental Models
| Model | Size | Date | Description |
|-------|------|------|-------------|
| `bnn_rushing_v1.pkl` | 53 MB | Oct 16, 2024 | v1.0 baseline rushing |
| `bnn_simpler_v2.pkl` | 10 MB | Oct 19, 2024 | Simplified architecture |

---

## Training Checkpoints (20 files, 503M)

### BNN Multi-Output Training Checkpoints
**Location:** `models/bayesian/bnn_multioutput_checkpoints/`
**Training Date:** November 2, 2025 (RTX 4090, 10 minutes, 500 epochs)
**Purpose:** Analyze training dynamics, convergence, overfitting

| Checkpoint | Size | Epoch | Notes |
|------------|------|-------|-------|
| `best_model.pt` | 3.4 MB | N/A | Best validation loss (PRODUCTION) |
| `epoch_50.pt` | 25 MB | 50 | Early training |
| `epoch_100.pt` | 25 MB | 100 | |
| `epoch_150.pt` | 25 MB | 150 | |
| `epoch_200.pt` | 25 MB | 200 | |
| `epoch_250.pt` | 25 MB | 250 | Mid-training |
| `epoch_300.pt` | 25 MB | 300 | |
| `epoch_350.pt` | 25 MB | 350 | |
| `epoch_400.pt` | 25 MB | 400 | |
| `epoch_450.pt` | 25 MB | 450 | Late training |
| `epoch_500.pt` | 25 MB | 500 | Final epoch |
| `epoch_550.pt` - `epoch_950.pt` | 25 MB each | 550-950 | Additional checkpoints |

**Total:** 20 checkpoints × 25 MB = 500 MB
**Recommendation:** Keep for dissertation analysis (Figure 5.3: Training dynamics), delete after publication

---

## Database Integration

### Registered Models (7 total, 3 BNN)

**Query all BNN models:**
```sql
SELECT
    model_id,
    model_name,
    model_type,
    version,
    is_production,
    model_path,
    description
FROM ml.model_registry
WHERE model_type = 'bnn'
ORDER BY is_production DESC, created_at DESC;
```

**Production BNN models in database:**
1. `bnn_multioutput_v1` (model_id=1)
2. `bnn_passing_v1` (model_id=2)
3. `bnn_calibrated_v2` (model_id=3) - Production class wrapper

---

## Cleanup Recommendations

### High Priority (Delete After Verification)

1. **Deprecated Models (1.6 GB saved):**
   - `bnn_multioutput_v1.pkl` (1.6 GB) - Old version, use `bnn_multioutput_v1_final.pt`
   - Verify new model works, then delete

2. **Data Leakage Models (298 MB saved):**
   - `bnn_rushing_vegas_v1.pkl` - Contains Vegas-derived features (circular logic)
   - Delete immediately

3. **Sigma Sweep Models (724 MB saved):**
   - `bnn_rushing_sigma*.pkl` (4 files) - Superseded by temperature scaling
   - Keep only `sigma1.0` as baseline, delete others

**Potential savings: 2.6 GB**

### Medium Priority (Archive)

1. **Large Experimental Models (2.0 GB):**
   - `bnn_mixture_experts_v2.pkl` (1.3 GB)
   - `bnn_rushing_opponent.pkl` (372 MB)
   - `bnn_rushing_environment.pkl` (302 MB)
   - Move to `models/archive/experimental/` for future reference

2. **Training Checkpoints (500 MB):**
   - Keep until dissertation published
   - Archive after publication (dissertation defense complete)

### Low Priority (Keep)

- All production models
- R hierarchical models
- `best_model.pt` checkpoint
- `bnn_simpler_v2.pkl` (small, useful baseline)

---

## Model Metadata

### File Format Standards

**PyTorch models (.pt, .pth):**
- Modern BNN models (Oct 2024+)
- Fast loading, GPU-compatible
- Preferred format for production

**PyMC models (.pkl):**
- Legacy BNN models (pre-Oct 2024)
- Contains full MCMC traces
- Large file sizes (80-1600 MB)

**R/brms models (.rds):**
- Hierarchical Bayesian models
- Stan MCMC backend
- Requires R/cmdstanr to load

### Loading Models

**PyTorch:**
```python
import torch
model = torch.load('models/bayesian/bnn_multioutput_v1_final.pt')
model.eval()
```

**PyMC:**
```python
import pickle
with open('models/bayesian/bnn_passing_v1.pkl', 'rb') as f:
    trace = pickle.load(f)
```

**R/brms:**
```r
library(brms)
model <- readRDS('models/bayesian/passing_informative_priors_v1.rds')
```

---

## Training Hardware

| Model Type | Hardware | Training Time | Notes |
|------------|----------|---------------|-------|
| PyTorch BNN | Windows RTX 4090 | 10 min (500 epochs) | GPU-accelerated |
| PyMC BNN | Mac M4 CPU | 30-60 min | CPU-only |
| R/brms | Mac M4 CPU | 25s - 30 min | Stan MCMC |

**GPU Speedup:** ~24x for suitable workloads (RTX 4090 vs M4 CPU)

---

## Version History

### v3.1 (November 2025) - Production BNN
- `bnn_multioutput_v1_final.pt` (58 KB)
- Temperature-scaled calibration (T=0.75, 89.5% coverage)
- Multi-output architecture (passing + rushing yards)
- RTX 4090 training (10 minutes)

### v3.0 (October 2024) - Advanced Features
- `bnn_passing_v1.pkl` (80 MB)
- QB-WR chemistry model (210 MB)
- Informative priors (5.2 MB)

### v2.5 (Earlier) - Enhancements
- Hierarchical baseline models

### v1.0 (2023-2024) - Baseline
- Initial BNN implementations

---

## Automated Inventory

Use the model registry sync script to keep this inventory current:

```bash
# Check sync status
uv run python scripts/sync_model_registry.py --check

# Generate fresh inventory
uv run python scripts/sync_model_registry.py --export-md

# Scan filesystem
uv run python scripts/sync_model_registry.py --scan
```

**Last sync:** November 3, 2025

---

## Summary Statistics

| Category | Count | Total Size | Status |
|----------|-------|------------|--------|
| **Production** | 3 | 290 MB | ✅ Active |
| **R Hierarchical** | 2 | 27 MB | ✅ Active |
| **Experimental** | 14 | 4.9 GB | 🧪 Research |
| **Checkpoints** | 20 | 503 MB | 📦 Archive pending |
| **TOTAL** | 39 | **5.7 GB** | |

**Cleanup potential:** 2.6 GB (46% reduction)
**After cleanup:** 3.1 GB (production + research-grade models)

---

**Maintained by:** Model Registry Automation
**Contact:** See `scripts/sync_model_registry.py` for programmatic access
**Database:** PostgreSQL `ml.model_registry` (single source of truth)
