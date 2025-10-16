# Calibration Experiments

This directory contains results from systematic experiments to improve BNN calibration from 26% → 90% coverage.

## Quick Start

```bash
# View all experiment results
cat summary.json

# Compare experiments side-by-side
cat comparison.csv

# View specific experiment
cat baseline_bnn_4features.json
cat vegas_bnn_6features.json  # When available
```

## Experiments

### ✅ Baseline (4 features)

**File**: `baseline_bnn_4features.json`

**Features**: carries, avg_rushing_l3, season_avg, week

**Results**:
- 90% CI Coverage: 26.2% (target: 90%) ❌
- MAE: 18.69 yards ✓
- Status: Severe under-calibration

**Key Finding**: Only 4 features insufficient for proper uncertainty quantification

---

### 🔄 Vegas Features (6 features)

**File**: `vegas_bnn_6features.json` (pending)

**Added**: spread_close, total_close

**Hypothesis**: Game context improves calibration

**Status**: Training in progress (~20-30 min)

---

## Metrics Tracked

All experiments include:

**Coverage** (primary metric):
- 90% CI coverage (target: 85-95%)
- 68% CI coverage (target: 63-73%)
- 95% CI coverage (target: 90-99%)

**Sharpness** (secondary metric):
- 90% CI width (yards)
- 68% CI width (yards)

**Point Accuracy** (constraint):
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

**Advanced Metrics**:
- CRPS (Continuous Ranked Probability Score)
- ECE (Expected Calibration Error)

## Comparison Criteria

**Success**:
- Coverage improves by 20%+ points
- Point accuracy maintained (MAE within 10%)

**Moderate**:
- Coverage improves by 5-20% points
- May need additional interventions

**Failure**:
- Coverage improves by <5% points
- Move to next phase (prior sensitivity or alternative UQ)

## Next Experiments

**Pending**:
- Environment features (dome, turf)
- Opponent defense features
- Prior sensitivity (sigma values)
- Alternative UQ methods (quantile, conformal, ensemble)

## References

See main documentation: `/Users/dro/rice/nfl-analytics/docs/BNN_CALIBRATION_EXPLORATION.md`
