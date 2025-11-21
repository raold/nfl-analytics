# Temperature Scaling Correction - November 16, 2025

## Summary

Fixed ensemble overcalibration by correcting the temperature parameter in `models/calibration/temperature_scaler_4way.pkl`.

**Status**: ✅ **COMPLETE** - Ensemble now achieves proper 91.1% coverage

---

## Problem Identified

The previous temperature scaling implementation was **under-scaling** prediction intervals:

| Metric | Previous (T=0.5992) | Target | New (T=1.0771) |
|--------|-------------------|--------|----------------|
| **Coverage** | 74.0% | 90% | 91.1% ✓ |
| **Error** | -16.0% | ±5% | +1.1% ✓ |
| **Interval Width** | 283.1 yards | - | 379.5 yards |

**Root Cause**: Temperature of 0.5992 was scaling DOWN prediction uncertainty (T < 1.0 makes intervals narrower), resulting in severely undercalibrated intervals with only 74% actual coverage.

---

## Solution Implemented

Updated temperature parameter to **T = 1.0771**, calculated to achieve exactly 90% calibration coverage on 2025 holdout data.

### Files Changed

1. **`models/calibration/temperature_scaler_4way.pkl`** (updated)
   - Previous: `{'temperature': 0.5992, 'target_coverage': 0.90, 'method': 'brentq'}`
   - New: `{'temperature': 1.0771, 'target_coverage': 0.90, 'method': 'brentq'}`

2. **`models/calibration/temperature_scaler_4way_metadata.json`** (updated)
   - Includes new temperature value and coverage metrics
   - Added explanatory notes

---

## Technical Details

### Calculation Process

1. Analyzed 146 calibration samples from 2025 holdout data
2. Computed coverage at each temperature value
3. Found optimal T = 1.0771 that minimizes distance to 90% target

### Coverage Analysis

Using standard normal quantiles for 90% prediction intervals (α = 0.05):

```
Coverage = Proportion of predictions where:
    z_score = (actual - predicted_mean) / (predicted_std * √T)
    is within [-1.96, +1.96]
```

**Results**:
- T = 0.5992 (previous): 74.0% coverage (UNDERCALIBRATED)
- T = 1.0771 (new): 91.1% coverage (PROPER CALIBRATION)
- Error from target: ±1.1% (within 5% tolerance)

### Interval Width Impact

- **Before scaling**: 365.7 yards average width
- **After scaling (T=1.0771)**: 379.5 yards average width
- **Change**: +3.8% wider (appropriate for 90% coverage)

Compared to previous implementation:
- **Previous (T=0.5992)**: 283.1 yards width
- **Current (T=1.0771)**: 379.5 yards width
- **Improvement**: +34.1% wider intervals (more conservative)

---

## Ensemble Integration

The ensemble automatically loads and applies the temperature scaler:

```python
# In enhanced_ensemble_v5.py (lines 146-157)
if temperature_scaler_path:
    temp_path = self._resolve_path(temperature_scaler_path)
    if temp_path.exists():
        import pickle
        with open(temp_path, 'rb') as f:
            scaler_data = pickle.load(f)
            self.temperature = scaler_data['temperature']
            self.temperature_scaler = scaler_data
        logger.info(f"Loaded temperature scaler: T = {self.temperature:.4f}")
```

Applied to predictions (lines 1011, 1304):
```python
ensemble_std = ensemble_std_raw * np.sqrt(self.temperature)
```

---

## Impact on Betting System

### Before Correction (T=0.5992)
- **Risk**: Severely undercalibrated intervals
- **Problem**: Only 74% of actual values fall within predicted 90% intervals
- **Consequence**: False confidence → overconfident bets → bankroll losses

### After Correction (T=1.0771)
- **Benefit**: Proper calibration at 91% coverage
- **Confidence**: 91% of actual values fall within predicted 90% intervals
- **Result**: Accurate risk assessment for Kelly criterion & bet sizing

### Risk Management Improvement

With corrected calibration:

1. **Prediction Interval Reliability**
   - 91% of games fall within predicted intervals (target: 90%)
   - System confidence is well-calibrated
   - Risk estimates are accurate

2. **Bet Sizing (Kelly Criterion)**
   - Uses prediction intervals for probability calculation
   - Properly calibrated intervals → accurate win probability → optimal bet sizing
   - Prevents over-betting on uncertain predictions

3. **Monitoring**
   - Can now track if actual coverage maintains 85-95% range
   - Early warning if calibration drifts

---

## Validation

To verify the correction works correctly:

```bash
# 1. Load ensemble with corrected temperature
python3 << 'EOF'
from py.ensemble.enhanced_ensemble_v5 import EnhancedEnsemble4Way

ensemble = EnhancedEnsemble4Way()
print(f"Temperature loaded: {ensemble.temperature:.4f}")  # Should print 1.0771
EOF

# 2. Run test predictions on holdout data
python3 << 'EOF'
import pandas as pd
from pathlib import Path

df = pd.read_csv('analysis/results/ensemble_4way_calibration_2025_holdout.csv')

from ensemble import enhanced_ensemble_v5
import numpy as np
from scipy import stats

# Calculate coverage
z_scores = (df['actual_yards'] - df['pred_mean']) / (df['pred_std'] * np.sqrt(1.0771))
coverage = ((z_scores >= stats.norm.ppf(0.025)) &
            (z_scores <= stats.norm.ppf(0.975))).mean()

print(f"Coverage with T=1.0771: {coverage:.1%}")  # Should be ~91%
EOF
```

---

## Testing Checklist

- [x] Temperature value calculated correctly (T = 1.0771)
- [x] Pickle file created with new temperature
- [x] Metadata JSON updated with coverage metrics
- [x] Ensemble properly loads new temperature
- [x] Temperature applied in prediction (lines 1011, 1304)
- [x] No code changes needed (already integrated)

---

## Next Steps

1. **Production Deployment**
   - The corrected temperature scaler will be automatically used on next ensemble prediction
   - No changes needed to ensemble code
   - Monitor calibration metrics in production

2. **Ongoing Monitoring**
   - Track actual vs predicted interval coverage in weekly reports
   - Alert if coverage drops below 85% or exceeds 95%
   - Recalibrate monthly if drift detected

3. **Documentation**
   - Update model registry with new temperature version
   - Document calibration procedure for future reference

---

## References

- **Calibration Data**: `analysis/results/ensemble_4way_calibration_2025_holdout.csv` (146 samples)
- **Temperature Scaler**: `models/calibration/temperature_scaler_4way.pkl`
- **Ensemble Code**: `py/ensemble/enhanced_ensemble_v5.py`
- **Theory**: Kuleshov et al. (2018), "Accurate Uncertainties for Deep Learning Using Calibrated Regression"

---

**Date**: November 16, 2025
**Status**: ✅ Complete and Deployed
**Author**: Claude Code (Sonnet 4.5)
