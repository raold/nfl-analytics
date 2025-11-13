# Temperature Scaling - Quick Reference Card
## One-Page Guide for Implementation

---

## Problem → Solution → Outcome

```
PROBLEM                      SOLUTION                    OUTCOME
97.3% coverage              Temperature Scaling         90% coverage
(too wide intervals)        σ_cal = σ_raw * √T          (optimal)
↓                           ↓                           ↓
Missed bets                 Learn T from data           +$5-10K/season
Suboptimal ROI              2-5 sec fitting time        3-4 hour impl.
```

---

## Implementation Checklist

### Phase 1: Code (1.5 hours)
- [ ] Add `temperature` param to `__init__` (5 lines)
- [ ] Implement `fit_temperature()` method (200 lines)
- [ ] Modify `predict()` to apply T (2 lines)
- [ ] Add `save_temperature()` and `load_temperature()` (20 lines)

### Phase 2: Testing (1.0 hours)
- [ ] Unit tests pass (synthetic data)
- [ ] Integration tests pass (2025 data)
- [ ] Coverage in 85-95% range ✓
- [ ] MAE unchanged ✓

### Phase 3: Validation (0.5 hours)
- [ ] Fit on 182 games (2025 weeks 1-10)
- [ ] Log optimal T value
- [ ] Save to configuration file
- [ ] Document results

### Phase 4: Documentation (0.5 hours)
- [ ] Update docstrings
- [ ] Add usage examples
- [ ] Create README section

**Total: 3.5 hours**

---

## Code Snippets

### 1. Add to __init__()
```python
# Lines 90-101 in enhanced_ensemble_v5.py
def __init__(
    self,
    # ... existing params ...
    temperature: float = 1.0,  # NEW
    temperature_fitted: bool = False,  # NEW
):
    # ... existing code ...
    self.temperature = temperature
    self.temperature_fitted = temperature_fitted
```

### 2. Add fit_temperature() Method
```python
# Full implementation in TECHNICAL_SPEC.md
# ~200 lines, uses scipy.optimize.minimize_scalar
# Returns dict with 'temperature', 'coverage_before/after', etc.

def fit_temperature(self, predictions_df, ...):
    """Fit T to minimize NLL, achieve ~90% coverage."""
    from scipy.optimize import minimize_scalar
    # ... optimize T ...
    return {'temperature': T_optimal, 'coverage_after': ..., ...}
```

### 3. Modify predict() Method
```python
# After line 596 (variance pooling)
# Add 3 lines:

if self.temperature_fitted and self.temperature != 1.0:
    ensemble_std = ensemble_std * np.sqrt(self.temperature)
    logger.debug(f"Applied T={self.temperature:.4f}")
```

### 4. Save/Load Temperature
```python
# Add two methods (~20 lines total)
def save_temperature(self, path):
    import json
    json.dump({'temperature': self.temperature}, open(path, 'w'))

def load_temperature(self, path):
    import json
    data = json.load(open(path, 'r'))
    self.temperature = data['temperature']
```

---

## Quick Start: Fitting Temperature

```python
import pandas as pd
from py.ensemble.enhanced_ensemble_v5 import EnhancedEnsemble4Way

# Load predictions
preds = pd.read_csv('predictions_2025.csv')
preds['actual_yards'] = [...]  # Add actuals

# Fit temperature
ensemble = EnhancedEnsemble4Way()
result = ensemble.fit_temperature(preds)

# Results
print(f"T = {result['temperature']:.4f}")
print(f"Coverage: {result['coverage_after']:.1%}")

# Save & use
ensemble.save_temperature('models/temperature.json')
ensemble.load_temperature('models/temperature.json')
```

---

## Expected Results

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| **Coverage** | 97.3% | ~90% | 90% |
| **Width** | 491.5 yd | ~450 yd | - |
| **T value** | 1.0 | ~0.83 | - |
| **Acceptable?** | ❌ | ✅ | - |

---

## Key Parameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| **temperature** | 1.0 | 0.1-5.0 | Main parameter, learned from data |
| **bounds** | (0.1, 5.0) | - | Optimization search range |
| **verbose** | True | - | Log optimization details |
| **z_score** | 1.645 | - | 90th percentile (fixed) |

---

## Testing Examples

### Test 1: Synthetic Data
```python
# Should learn T < 1.0 from overconfident predictions
import numpy as np
np.random.seed(42)
means = np.random.normal(250, 50, 100)
actuals = means + np.random.normal(0, 30, 100)
stds = np.sqrt(30**2 + 50**2) * np.ones(100)  # Overestimate

result = ensemble.fit_temperature(df)
assert result['temperature'] < 1.0  # Should shrink
assert 0.85 <= result['coverage_after'] <= 0.95
```

### Test 2: Real Data (2025 holdout)
```python
# Load 2025 weeks 1-10
preds = ensemble.predict_batch(games_2025)
preds['actual_yards'] = actuals

# Fit
result = ensemble.fit_temperature(preds)
assert result['acceptable'] == True
```

---

## Success Criteria

### Green Light (✅ Proceed)
- T in [0.5, 2.0] range
- Coverage: 85-95%
- Fitting time < 30 sec
- No errors in testing

### Yellow Light (⚠️ Investigate)
- T in [0.2, 5.0] range
- Coverage: 80-98%
- Larger-than-expected fitting time
- Minor test failures

### Red Light (❌ Fallback)
- T outside [0.1, 5.0]
- Coverage <80% or >98%
- Optimization failed to converge
- Test failures indicate bugs

---

## Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| T very large (>2) | Undercalibrated | Check if std too small |
| T very small (<0.3) | Overcalibrated | Data distribution shift? |
| Won't converge | Bad bounds | Widen to (0.05, 10) |
| Wrong column names | Input df error | Check column names match |
| Fitting very slow | Large dataset | Normal for >1000 samples |

---

## Files to Modify

| File | Changes | Lines |
|------|---------|-------|
| `py/ensemble/enhanced_ensemble_v5.py` | Main implementation | ~230 |
| `py/ensemble/test_ensemble_v5.py` | Tests | ~80 |
| `models/temperature.json` | Configuration (new) | ~5 |

**No other files modified**

---

## Mathematical Foundation

```
Temperature Scaling: σ_cal = σ_raw * √T

Objective: Minimize NLL = Σ(0.5*log(σ²) + 0.5*(y-ŷ)²/σ²)

Method: Bounded optimization (scipy.optimize.minimize_scalar)

Theory: Guo et al. (2017), ICML
"On Calibration of Modern Neural Networks"
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Wrong T | Use bounded optimization with alerts |
| Coverage still bad | Log all metrics, can retrain |
| Ensemble breaks | Extensive unit + integration tests |
| Performance degradation | Can revert (set T=1.0) instantly |
| Data leakage | Using holdout 2025 data, no train data |

---

## Integration Timeline

```
Nov 13, 14:00  Implementation starts
Nov 13, 15:00  Coding complete
Nov 13, 15:45  Testing complete
Nov 13, 16:15  Fit on 2025 data
Nov 13, 16:45  Save results
Nov 13, 17:00  Documentation done
Nov 14, 08:00  Ready for production
```

---

## Production Deployment

### Step 1: Load Temperature
```python
ensemble = EnhancedEnsemble4Way()
ensemble.load_temperature('models/temperature.json')
```

### Step 2: Generate Predictions
```python
pred = ensemble.predict(player_id, season, week)
# std automatically scaled by T
```

### Step 3: Monitor Calibration
```python
# Track weekly: coverage should stay 85-95%
# Alert if <80% or >98%
```

---

## Documentation Links

1. **Full Implementation Plan:** TEMPERATURE_SCALING_IMPLEMENTATION_PLAN.md
2. **Technical Spec:** TEMPERATURE_SCALING_TECHNICAL_SPEC.md
3. **Executive Summary:** TEMPERATURE_SCALING_EXECUTIVE_SUMMARY.md
4. **This Quick Reference:** TEMPERATURE_SCALING_QUICK_REFERENCE.md

---

## Key Insight

Temperature scaling is **NOT** a hack or workaround.

It's the **industry standard** post-hoc calibration method:
- Used at Google, Meta, Tesla
- Published in ICML (top ML conference)
- Mathematically sound and theoretically justified
- Single parameter = minimal overfitting risk
- Reversible = zero financial downside

**It's the RIGHT way to fix calibration.**

---

**Quick Reference v1.0 | November 13, 2025 | Ready to Implement**
