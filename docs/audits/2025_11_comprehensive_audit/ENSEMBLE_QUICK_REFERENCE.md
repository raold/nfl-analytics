# Ensemble Health Monitor - Quick Reference Card

**Last Updated:** November 14, 2025
**Ensemble Version:** v5.1 (Production)

---

## 30-Second Overview

**Health Score:** 7.8/10 (GOOD)
**Status:** ✅ Functional, ⚠️ Validation Incomplete
**Approved For:** Manual betting (< 100 predictions/day)
**Blocked For:** Automated betting (requires backtest validation)

---

## Production Architecture (v5.1)

```
┌─────────────────────────────────────────────────┐
│         4-Way Ensemble v5.1                     │
├─────────────────────────────────────────────────┤
│ 1. State-Space v2 (R/Stan)         → 30%       │
│ 2. Hierarchical v2 (R/brms)        → 30%       │
│ 3. Hybrid XGBoost v1 (Python)      → 25%       │
│ 4. Informative Priors v2 (R/brms)  → 15%       │
├─────────────────────────────────────────────────┤
│ Post-Processing:                                │
│ - Temperature Scaling: T = 0.5534               │
│ - CQL Bet Sizer: Kelly + Risk Gates             │
└─────────────────────────────────────────────────┘

Training Window: 2018-2024 + H=4 decay (2015-2017)
Expected ROI: +5.5-7.5% (theoretical, not validated)
Expected MAE: 39-42 yards
Calibration: 89.7% coverage (with temperature scaling)
```

---

## Critical Files

```
Production Ensemble:
├── py/ensemble/enhanced_ensemble_v5_1.py (749 lines) ✅ USE THIS
├── models/MODEL_REGISTRY.yaml (v5.1 config)          ✅ TRUTH SOURCE
└── models/calibration/temperature_scaler_4way.pkl    ✅ REQUIRED

Deprecated (DO NOT USE):
├── py/ensemble/enhanced_ensemble_v3.py       ❌ STALE (Oct 2025)
├── py/ensemble/enhanced_ensemble_v5.py       ❌ SUPERSEDED
├── py/ensemble/enhanced_ensemble_v5_canonical.py ❌ SUPERSEDED
└── py/ensemble/enhanced_ensemble_v6.py       ❌ WRONG ARCHITECTURE

Model Files:
├── models/bayesian/simplified_random_walk_v2.rds (1.4 GB)
├── models/bayesian/passing_yards_hierarchical_v2.rds (124 MB)
├── models/bayesian/passing_informative_priors_v2.rds (123 MB)
└── models/xgboost/hybrid_static_incremental_v1/ (5 files, 294 KB)

Health Monitor:
└── py/utils/ensemble_health_monitor.py (850 lines) ✅ USE BEFORE BACKTEST
```

---

## Usage Examples

### Single Prediction
```python
from py.ensemble.enhanced_ensemble_v5_1 import EnhancedEnsemble4Way

ensemble = EnhancedEnsemble4Way()

# Patrick Mahomes, 2024 Week 1
result = ensemble.predict(
    player_id="00-0033873",
    season=2024,
    week=1
)

print(f"Prediction: {result['ensemble_mean']:.1f} ± {result['ensemble_std']:.1f} yards")
print(f"90% CI: [{result['quantile_10']:.1f}, {result['quantile_90']:.1f}]")
print(f"Inference time: {result['inference_time_ms']:.1f}ms")
```

**Expected Output:**
```
Prediction: 295.2 ± 96.9 yards
90% CI: [135.9, 454.5]
Inference time: 4628.8ms
```

### Pre-Backtest Health Check
```python
from py.utils.ensemble_health_monitor import EnsembleHealthMonitor

monitor = EnsembleHealthMonitor()

# CRITICAL: Run before 18-hour backtests
report = monitor.check_ensemble_health()

if not report.is_healthy:
    print("⛔ Ensemble unhealthy - do not run backtest")
    exit(1)

print("✅ Ensemble healthy - safe to run backtest")
```

### Batch Predictions
```python
ensemble = EnhancedEnsemble4Way()

games = [
    {"player_id": "00-0033873", "season": 2024, "week": 1},
    {"player_id": "00-0036355", "season": 2024, "week": 1},
]

results_df = ensemble.predict_batch(
    games=games,
    output_csv="predictions/week1_2024.csv"
)
```

---

## Blocking Issues (Do NOT Deploy Until Fixed)

### P0-1: No Full Backtest 🚨
**Why BLOCKING:** Expected ROI +5.5-7.5% is theoretical (not validated on holdout data)
**Action:** Run backtest on 450 games (2022-2024)
**Effort:** 8 hours runtime + 2 hours analysis
**Command:**
```bash
python py/backtest/ensemble_5way_backtest.py \
  --start-date 2022-01-01 \
  --end-date 2024-12-31 \
  --output analysis/backtest/v5_1_validation.json
```

### P0-2: CLAUDE.md Conflict 🚨
**Why BLOCKING:** Developers may use wrong ensemble architecture
**Action:** Update CLAUDE.md line 60
**Current:** "4-way (State-Space v2 + BNN v3.1 Rushing + Hierarchical v2 + XGBoost)"
**Correct:** "4-way (State-Space v2 + Hierarchical v2 + Hybrid XGBoost v1 + Informative Priors v2)"

### P0-3: Missing Temperature Scaler Check 🚨
**Why BLOCKING:** Uncalibrated predictions if file missing (97.3% coverage instead of 90%)
**Action:** Add validation in `__init__` (enhanced_ensemble_v5_1.py, line 165)
```python
if not self.temperature_scaler:
    raise FileNotFoundError("Temperature scaler required for production")
```

---

## Configuration Conflicts Resolved

### Conflict 1: BNN v3.1 Inclusion
**CLAUDE.md line 60:** Says BNN v3.1 is in ensemble ❌ WRONG
**MODEL_REGISTRY.yaml:** Excludes BNN v3.1 ✅ CORRECT
**Truth:** BNN v3.1 is **EXCLUDED** (rushing-only, not for passing yards)

### Conflict 2: Multiple Versions
**Production:** enhanced_ensemble_v5_1.py ✅
**Deprecated:** v3, v5, v5_canonical, v6 ❌
**Action:** Add deprecation warnings to v3, v5, v5_canonical

---

## Performance Specs

### Latency (Single Prediction)
```
Current (no caching):     4,628ms  ⚠️  TOO SLOW FOR API
With pre-loading:           770ms  ✅  OK FOR MANUAL
With Redis caching:       5-10ms  ✅  OK FOR AUTOMATED

Target for production:    < 500ms
```

### Bottleneck Analysis
```
R model loading:   3,236ms (70%)  ← CRITICAL BOTTLENECK
Posterior draws:     770ms (16%)
Python models:       236ms (5%)
Overhead:            386ms (9%)
─────────────────────────────────
TOTAL:             4,628ms (100%)
```

### Optimization Roadmap
1. **Pre-load models on startup** → 770ms (4.6s → 770ms)
2. **Add Redis caching** → 5-10ms (15x speedup)
3. **Reduce posterior draws (1000 → 500)** → 400ms saved

---

## Model Loading Checklist

```bash
# Check all files exist
ls -lh models/bayesian/simplified_random_walk_v2.rds          # 1.4 GB ✅
ls -lh models/bayesian/passing_yards_hierarchical_v2.rds     # 124 MB ✅
ls -lh models/bayesian/passing_informative_priors_v2.rds     # 123 MB ✅
ls -lh models/xgboost/hybrid_static_incremental_v1/           # 5 files ✅
ls -lh models/calibration/temperature_scaler_4way.pkl         # 84 B ✅

# Test ensemble loading
python scripts/test_ensemble_v5_1.py  # Should take ~5s, all tests pass ✅

# Check R/rpy2 integration
python -c "import rpy2.robjects as ro; print(ro.r('R.version.string'))"
# Expected: R version 4.5.1 (2025-06-13) ✅
```

---

## Common Errors & Solutions

### Error 1: "No module named 'rpy2'"
**Solution:**
```bash
uv pip install rpy2
# OR
pip install rpy2==3.5.11
```

### Error 2: "R model loading failed"
**Check:**
1. R installed? `which R`
2. brms installed? `Rscript -e 'library(brms)'`
3. cmdstanr installed? `Rscript -e 'library(cmdstanr)'`

**Solution:**
```bash
Rscript -e 'install.packages("brms", repos="https://cloud.r-project.org")'
```

### Error 3: "Temperature scaler not found"
**Impact:** Predictions uncalibrated (97.3% coverage instead of 90%)
**Solution:**
```bash
# Check file exists
ls -lh models/calibration/temperature_scaler_4way.pkl

# If missing, retrain temperature scaler
python py/calibration/train_temperature_scaler.py
```

### Error 4: "Hybrid XGBoost model not found"
**Solution:**
```bash
# Check directory exists
ls -lh models/xgboost/hybrid_static_incremental_v1/

# If missing, retrain hybrid model
python scripts/train_hybrid_xgboost_v1.py
```

---

## Risk Gates (CQL Bet Sizer)

```python
# Applied AFTER ensemble prediction
risk_gates = {
    "min_edge": 0.02,              # 2% minimum edge
    "max_bet_size": 0.05,          # 5% of bankroll
    "daily_stop_loss": -0.10,      # -10% daily drawdown
    "max_uncertainty": 80.0,       # 80 yards max std
}

# Example
if ensemble_std > 80.0:
    print("⛔ Uncertainty too high - skip bet")
elif expected_roi < 0.02:
    print("⛔ Edge too small - skip bet")
else:
    kelly_fraction = 0.25 * expected_roi  # Conservative Kelly
    bet_size = min(kelly_fraction, 0.05) * bankroll
```

---

## Monitoring Metrics (To Be Implemented)

**P2-1: Prometheus Metrics**
```python
nfl_ensemble_prediction_latency_seconds (histogram)
nfl_ensemble_model_load_failures_total (counter)
nfl_ensemble_calibration_coverage (gauge)
nfl_ensemble_between_model_variance (gauge)
```

**P2-2: Grafana Dashboard**
- Latency p50/p95/p99
- Model load success rate (should be 100%)
- Calibration drift (rolling 100 predictions)
- Model agreement (between-model variance)

---

## Version History (Condensed)

```
v3.0 (Oct 2025)    → STALE (legacy BNN + stacking)
v5.0 (Nov 9)       → SUPERSEDED (static XGBoost)
v5_canonical       → SUPERSEDED (5-way, BNN passing included)
v5.1 (Nov 13) ✅   → PRODUCTION (hybrid XGBoost, correct 4-way)
v6.0 (Nov 12)      → DEPRECATED (BNN rushing in passing ensemble)
```

**v5.1 Key Changes:**
- Hybrid XGBoost (+4.2% improvement over static)
- Excluded BNN v3.1 (rushing-only)
- Temperature scaling (T = 0.5534)

---

## Contact & Support

**Audit Report:** [docs/audits/2025_11_comprehensive_audit/13_ensemble_health_monitor.md](13_ensemble_health_monitor.md)

**Model Registry:** [models/MODEL_REGISTRY.yaml](../../../models/MODEL_REGISTRY.yaml)

**Health Monitor:** [py/utils/ensemble_health_monitor.py](../../../py/utils/ensemble_health_monitor.py)

**Next Audit:** After P0-1 backtest (expected Nov 15-16, 2025)

---

**Remember:** Real money at stake. Do NOT skip P0-1 backtest validation before automated betting.
