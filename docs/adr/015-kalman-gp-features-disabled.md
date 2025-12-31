# ADR-015: Kalman EPA and GP Injury Features Disabled

**Status:** ACCEPTED
**Date:** 2025-12-31
**Decision Makers:** dro, Claude Code

## Context

In ADR-013 (Dec 28), we validated two new feature sets:

1. **Kalman EPA Filter (18 features)**: State-space model for EPA trend detection with CUSUM regime change indicators
2. **GP Injury Recovery (17 features)**: Gaussian Process model for uncertainty-aware injury recovery curves

These were integrated into XGBoost v2.1 and tested against v2 baseline on the 2025 holdout (163 games).

## Decision

**DISABLE both feature groups (weight=0.0) via configuration, preserving code for future iteration.**

### Implementation

Created `py/config/feature_weights.yaml` with weight=0.0 for both groups:
- Features are extracted only when `is_feature_enabled("group_name")` returns True
- Code remains in place for easy re-enabling
- Reason and date documented in config

## Evidence

### A/B Test Results (v2.1 vs v2)

| Metric | v2 Baseline | v2.1 (Kalman+GP) | Impact |
|--------|-------------|------------------|--------|
| Spread MAE | 10.45 | 10.70 | **+0.25 worse** |
| Moneyline Accuracy | 69.1% | 58.9% | **-10.2pp worse** |
| AUC | 0.691 | 0.627 | **-0.064 worse** |

### Root Cause Analysis

**Kalman EPA - NEGATIVE SIGNAL:**
1. EPA already captured by existing features (redundant)
2. CUSUM regime changes don't correlate with game outcomes
3. 2-season lookback may be insufficient for stable estimates
4. Adds 18 features that dilute model signal

**GP Injury - PURE NOISE:**
1. `self._trained = False` ALWAYS - GP never actually trains
2. Uses `exponential_decay_baseline()` with arbitrary tau=14.0
3. `days_since_return` is ESTIMATED from severity index, NOT real data:
   ```python
   home_days = max(0, int(14 - home_severity * 2))  # FAKE
   ```
4. Uncertainty bounds are heuristic (15.0 magic number), not calibrated
5. All injury type flags (muscle/structural/concussion) are zero when unknown

### Zero Features Analysis

120/423 features (28.4%) were zeros, broken down:

| Category | Count | Root Cause |
|----------|-------|------------|
| Possession Value | 8 | Schema bug: `fixed_drive` column missing |
| QBR | 20 | `espn_qbr` table empty for 2024/2025 |
| NextGen | 46 | `nextgen_*` tables sparse |
| Turnover Luck | 25 | Insufficient data in query |
| Kalman | 8 | Expected zeros for Week 1 (no history) |
| GP | 5 | Binary flags when injury_type="unknown" |

## Alternatives Considered

1. **Delete the code**: Rejected - preserves option to iterate
2. **Environment variable toggle**: Rejected - YAML config more flexible
3. **Keep enabled with regularization**: Rejected - would still dilute signal
4. **Fix GP training**: Deferred - requires real player-level injury data

## Consequences

### Positive
- Model accuracy restored to v2 baseline levels
- Eliminated 35 features (18 Kalman + 17 GP) that added noise
- Feature extraction faster without Kalman DB queries
- Clear documentation of why features failed

### Negative
- Research time spent on features that didn't pan out
- Need to revisit if/when proper injury data available

### Neutral
- Code preserved for future iteration
- Configuration pattern established for feature toggling

## Files Modified

| File | Change |
|------|--------|
| `py/config/feature_weights.yaml` | Created - feature group weights |
| `py/config/feature_weights.py` | Created - config loader |
| `py/config/__init__.py` | Created - package exports |
| `py/features/game_feature_extractor.py` | Added `is_feature_enabled()` checks |
| `py/features/possession_value.py` | Fixed schema bug (returns defaults) |

## Lessons Learned

1. **Validate features BEFORE training**: Signal:noise analysis should happen before full model training
2. **Real data matters**: GP recovery using estimated `days_since_return` is worse than no GP at all
3. **Feature count isn't value**: 423 features underperformed 382 features
4. **Config > code for experiments**: Feature weights config enables rapid A/B testing

## Related

- ADR-013: Feature Validation Results
- ADR-014: XGBoost v2.1 A/B Test Results

---

**Author:** Claude Code
**Review:** dro
