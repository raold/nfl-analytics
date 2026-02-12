# Calibration Fix - Quick Start Guide

## Problem
Ensemble calibration is **18.61%** instead of target **85-95%** (71.4 pp gap).

## Solution
Apply 2 immediate fixes to improve coverage by ~47 pp to **65-70%**.

---

## Fix #1: Aggressive Temperature Scaling (5 minutes)

### File
`py/models/player_props_ensemble.py`

### Lines
423-428

### Change FROM:
```python
temperature_map = {
    "bayesian": 0.75,
    "xgboost": 0.80,
    "bnn": 0.75,
    "bnn_calibrated": 1.0,
    "state_space": 0.85
}
```

### Change TO:
```python
temperature_map = {
    "bayesian": 0.35,       # Was 0.75, now ~2.86x widening
    "xgboost": 0.40,        # Was 0.80, now ~2.5x widening
    "bnn": 0.35,            # Was 0.75, now ~2.86x widening
    "bnn_calibrated": 0.50, # Was 1.0, now ~2x widening
    "state_space": 0.45     # Was 0.85, now ~2.22x widening
}
```

**Expected Impact:** +40 pp coverage

---

## Fix #2: Equal Weighting (2 minutes)

### File
`py/models/player_props_ensemble.py`

### Lines
924-926

### Change FROM:
```python
# Inverse variance weighting
inv_variances = np.array([1.0 / v for v in variances])
weights = inv_variances / inv_variances.sum()
```

### Change TO:
```python
# Equal weighting (robust to miscalibration)
weights = np.ones(len(predictions)) / len(predictions)
```

**Expected Impact:** +7 pp coverage

---

## Validation (30 minutes)

### Rerun Backtest
```bash
cd \

python py/backtest/historical_validation_5way.py \
  --start-season 2022 \
  --end-season 2024 \
  --output analysis/results/historical_validation_4way_TEMP_FIX.csv
```

### Check Coverage
```bash
python3 << 'EOF'
import pandas as pd
import numpy as np

df = pd.read_csv("analysis/results/historical_validation_4way_TEMP_FIX.csv")
ensemble_df = df[df['prediction_mean'].notna()].copy()

actual = ensemble_df['actual_yards'].values
q05 = ensemble_df['prediction_q05'].values
q95 = ensemble_df['prediction_q95'].values

in_interval = (actual >= q05) & (actual <= q95)
coverage = in_interval.mean() * 100

print(f"Ensemble Coverage: {coverage:.2f}% (target: 85-95%)")
print(f"Coverage Gap: {90 - coverage:+.2f} pp")
print()
if coverage >= 85:
    print("✅ SUCCESS - Calibration achieved!")
elif coverage >= 60:
    print("⚠️  PARTIAL - Apply isotonic calibration next")
else:
    print("❌ INSUFFICIENT - Investigate model training")
EOF
```

### Expected Output
```
Ensemble Coverage: 65-70% (target: 85-95%)
Coverage Gap: +20-25 pp
⚠️  PARTIAL - Apply isotonic calibration next
```

---

## If Coverage Still < 85% (Long-term)

### Option A: Isotonic Calibration (Recommended)
- Fit isotonic regression to map predicted probabilities to calibrated probabilities
- Works well with moderate miscalibration
- Implementation: ~2 hours

### Option B: Retrain Models (If overfitting suspected)
- Increase L2 regularization (XGBoost lambda: 1.0 → 5.0)
- Validation-based early stopping
- Cross-validation (5-fold)
- Implementation: ~1-2 weeks

---

## Files

### Modified
- `py/models/player_props_ensemble.py`

### Analyzed
- `analysis/results/historical_validation_4way_EMERGENCY_FIX.csv`

### Documentation
- `docs/diagnostics/calibration_diagnostic_nov7.md` (full report)
- `docs/diagnostics/CALIBRATION_DIAGNOSTIC_SUMMARY_NOV7.md` (summary)

---

## Summary

| Fix | Impact | Time | Priority |
|-----|--------|------|----------|
| Temperature Scaling | +40 pp | 5 min | **NOW** |
| Equal Weighting | +7 pp | 2 min | **NOW** |
| Isotonic Calibration | +15 pp | 2 hrs | If < 85% |
| Model Retraining | +10-20 pp | 1-2 wks | If < 70% |

**Total Expected:** 65-70% coverage (up from 18.61%)

---

**Quick Start Date:** November 7, 2025
**Analysis by:** Claude Code (Sonnet 4.5)
