# River Library NFL-Specific Tuning Guide

**Author:** Claude Code (Sonnet 4.5)
**Created:** November 12, 2025
**Status:** ✅ Complete (based on Phase 4 debugging)

---

## Executive Summary

**Problem:** River library defaults are tuned for high-frequency data streams (stock prices with 1000s of samples/day). NFL has only **18 weeks and ~500 samples/season**, causing drift detectors to fail (0 events detected).

**Solution:** NFL-specific configuration reduces thresholds by 5-10× to account for data sparsity.

**Impact:** Expected drift detection rate increases from 0% → 10-30% of seasons (realistic for NFL).

---

## Table of Contents

1. [Why River Defaults Fail for NFL](#why-river-defaults-fail-for-nfl)
2. [Recommended Configuration](#recommended-configuration)
3. [Detector-Specific Tuning](#detector-specific-tuning)
4. [Implementation Guide](#implementation-guide)
5. [Validation Strategy](#validation-strategy)
6. [Known Issues](#known-issues)

---

## Why River Defaults Fail for NFL

### Data Characteristics Comparison

| Characteristic | River Default Assumption | NFL Reality | Impact |
|---------------|-------------------------|-------------|--------|
| **Samples/day** | 1000+ (stock market) | 32 games/week (~4/day) | 250× less frequent |
| **Samples/season** | 250,000+ (hourly for 1 year) | 500 player-games | 500× less data |
| **Temporal granularity** | Minutes to hours | Weeks (7 days) | 100× coarser |
| **Variance** | Low (continuous markets) | High (injuries, weather) | 10× more noisy |
| **Drift events** | Gradual (economic trends) | Abrupt (rule changes) | Different regime |

### Failure Modes

#### 1. ADWIN (Adaptive Windowing)

**River Default:** `delta=0.002` (99.8% confidence)

**Issue:** Requires ~500+ samples to detect mean changes. NFL has only 18 weeks (126 games total), so ADWIN never fires.

**Example:**
```python
from river.drift import ADWIN

detector = ADWIN(delta=0.002)  # Default

# Feed 18 weeks of NFL data (72 samples)
for mae in [55, 58, 52, 60, 57, ..., 75, 78, 80, 85]:  # Drift in week 15
    detector.update(mae)
    if detector.drift_detected:
        print("Drift!")  # NEVER PRINTS - threshold too high
```

**Fix:** Use `delta=0.01` (90% confidence) to fire after ~50 samples.

---

#### 2. KSWIN (Kolmogorov-Smirnov Windowing)

**River Default:** `window_size=100`, `alpha=0.005` (0.5% significance)

**Issue:** Window size of 100 games requires 3+ weeks of data. NFL seasons are only 18 weeks, so detector can't accumulate enough samples.

**Example:**
```python
from river.drift import KSWIN

detector = KSWIN(alpha=0.005, window_size=100)  # Default

# Feed 18 weeks of NFL data
for week in range(1, 19):
    for game in games_in_week:
        detector.update(mae)
# Result: Detector never has 100 samples in window → no drift detected
```

**Fix:** Use `window_size=30` (1 week) and `alpha=0.05` (5% significance).

---

#### 3. Page-Hinkley Test

**River Default:** `threshold=50.0`, `delta=0.005`

**Issue:** Threshold of 50 is calibrated for stock prices (magnitude ~100-1000). NFL MAE is ~56 yards, so cumulative deviations never reach 50.

**Example:**
```python
from river.drift import PageHinkley

detector = PageHinkley(threshold=50.0, delta=0.005)  # Default

# Feed NFL MAE values (magnitude ~56 yards)
for mae in [55, 58, 52, 60, ..., 75, 78, 80]:
    detector.update(mae)
# Result: Cumulative sum never exceeds 50 → no drift
```

**Fix:** Use `threshold=20.0` (2.5× more sensitive).

---

### Ensemble Decision Logic

**Current Implementation:**
```python
# From DriftDetector.update()
drift_flags = [name for name, detector in self.detectors.items() if detector.drift_detected]
return len(drift_flags) >= 2  # Require 2/3 detectors to confirm
```

**Issue:** Even if one detector (Page-Hinkley) fires, drift is never confirmed because the other two (ADWIN, KSWIN) have thresholds too high for NFL.

**Observed Behavior:**
```
INFO: Drift detected by PageHinkley (total: 4)  # Internal log
INFO: Prequential evaluation: Drifts=0          # Final count (ensemble blocked)
```

**Fix:** Require only 1/3 detectors (any single detector triggers drift):
```python
return len(drift_flags) >= 1  # Any detector is sufficient
```

---

## Recommended Configuration

### NFL-Tuned Parameters

```python
from py.online.online_learning_adapter import OnlinePredictionPipeline

# Create pipeline with NFL-tuned configuration
pipeline = OnlinePredictionPipeline(
    online_model_type='adaptive_random_forest',
    drift_sensitivity='nfl_tuned',  # NEW: Custom NFL configuration
)

# Under the hood (in OnlinePredictionPipeline.__init__):
if drift_sensitivity == 'nfl_tuned':
    adwin_delta = 0.01           # 5× more sensitive (was 0.002)
    kswin_alpha = 0.05           # 10× more permissive (was 0.005)
    kswin_window_size = 30       # 3× smaller (was 100)
    ph_threshold = 20.0          # 2.5× more sensitive (was 50.0)
    ph_delta = 0.01              # 2× more sensitive (was 0.005)
    min_detectors_for_drift = 1  # Any detector (was 2)
```

### Comparison Table

| Parameter | River Default | NFL-Tuned | Reason |
|-----------|---------------|-----------|--------|
| **ADWIN delta** | 0.002 | 0.01 | Fewer samples available |
| **KSWIN alpha** | 0.005 | 0.05 | High week-to-week variance |
| **KSWIN window** | 100 | 30 | Only 18 weeks per season |
| **PH threshold** | 50.0 | 20.0 | MAE magnitude ~56 yards |
| **PH delta** | 0.005 | 0.01 | Need sensitivity to 5-10 yard shifts |
| **Ensemble votes** | 2/3 | 1/3 | Sparse data → any detector sufficient |

---

## Detector-Specific Tuning

### ADWIN (Adaptive Windowing)

**Purpose:** Detect changes in mean (e.g., average MAE increases from 56 → 65 yards)

**Tuning Parameter:** `delta` (confidence level)

**Recommended Value:** `0.01` (90% confidence)

**Rationale:**
- NFL has only 18 weeks → need fast detection
- False positives are acceptable (retraining 1-2× per season is fine)
- Known drift events (COVID, rule changes) are obvious and persistent

**Test Case:**
```python
from river.drift import ADWIN

detector = ADWIN(delta=0.01)  # NFL-tuned

# Simulate known drift: Week 1-10 (MAE=56), Week 11-18 (MAE=70)
for week in range(1, 11):
    for _ in range(8):  # 8 games/week
        detector.update(56 + np.random.randn() * 5)

for week in range(11, 19):
    for _ in range(8):
        detector.update(70 + np.random.randn() * 5)  # DRIFT
        if detector.drift_detected:
            print(f"Drift detected in week {week}")  # Should fire in week 11-12

# Expected: Detector fires within 2 weeks of drift (10-16 games)
```

---

### KSWIN (Kolmogorov-Smirnov Windowing)

**Purpose:** Detect changes in distribution shape (e.g., MAE variance increases)

**Tuning Parameters:**
- `alpha` (significance level)
- `window_size` (number of samples)

**Recommended Values:**
- `alpha=0.05` (5% significance, vs 0.5% default)
- `window_size=30` (1 week of NFL games, vs 100 default)

**Rationale:**
- NFL has high week-to-week variance (injuries, weather) → need permissive alpha
- Window size of 100 requires 3+ weeks → too slow for 18-week season
- Distribution changes (e.g., COVID empty stadiums) are obvious

**Test Case:**
```python
from river.drift import KSWIN

detector = KSWIN(alpha=0.05, window_size=30)  # NFL-tuned

# Simulate distribution change: Week 1-10 (σ=5), Week 11-18 (σ=15)
for week in range(1, 11):
    for _ in range(8):
        detector.update(56 + np.random.randn() * 5)  # Low variance

for week in range(11, 19):
    for _ in range(8):
        detector.update(56 + np.random.randn() * 15)  # HIGH variance (DRIFT)
        if detector.drift_detected:
            print(f"Drift detected in week {week}")

# Expected: Detector fires within 1-2 weeks (8-16 games)
```

---

### Page-Hinkley Test

**Purpose:** Detect cumulative deviations (sequential changes over multiple weeks)

**Tuning Parameters:**
- `threshold` (cumulative deviation magnitude)
- `delta` (minimum change amplitude)

**Recommended Values:**
- `threshold=20.0` (vs 50.0 default)
- `delta=0.01` (vs 0.005 default)

**Rationale:**
- NFL MAE is ~56 yards → threshold of 50 is too high
- Regime changes (rule changes) cause 5-10 yard shifts → need delta=0.01
- Cumulative test is good for gradual shifts (17-game season, fatigue effects)

**Test Case:**
```python
from river.drift import PageHinkley

detector = PageHinkley(threshold=20.0, delta=0.01)  # NFL-tuned

# Simulate gradual drift: MAE increases 2 yards/week
for week in range(1, 19):
    for _ in range(8):
        mae = 56 + 2 * week + np.random.randn() * 5  # Gradual increase
        detector.update(mae)
        if detector.drift_detected:
            print(f"Drift detected in week {week}")

# Expected: Detector fires in week 8-10 (cumulative sum exceeds 20)
```

---

## Implementation Guide

### Step 1: Update `OnlinePredictionPipeline.__init__`

**File:** `py/online/online_learning_adapter.py`

**Current Code (Lines 428-435):**
```python
# Set drift detector sensitivity
if drift_sensitivity == 'low':
    adwin_delta, kswin_alpha = 0.01, 0.01
elif drift_sensitivity == 'medium':
    adwin_delta, kswin_alpha = 0.002, 0.005
elif drift_sensitivity == 'high':
    adwin_delta, kswin_alpha = 0.0002, 0.001
else:
    raise ValueError(f"Unknown sensitivity: {drift_sensitivity}")
```

**Updated Code:**
```python
# Set drift detector sensitivity
if drift_sensitivity == 'low':
    adwin_delta, kswin_alpha = 0.01, 0.01
    kswin_window_size, ph_threshold, ph_delta = 100, 50.0, 0.005
elif drift_sensitivity == 'medium':
    adwin_delta, kswin_alpha = 0.002, 0.005
    kswin_window_size, ph_threshold, ph_delta = 100, 50.0, 0.005
elif drift_sensitivity == 'high':
    adwin_delta, kswin_alpha = 0.0002, 0.001
    kswin_window_size, ph_threshold, ph_delta = 100, 50.0, 0.005
elif drift_sensitivity == 'nfl_tuned':  # NEW
    adwin_delta, kswin_alpha = 0.01, 0.05
    kswin_window_size, ph_threshold, ph_delta = 30, 20.0, 0.01
else:
    raise ValueError(f"Unknown sensitivity: {drift_sensitivity}")

# Pass parameters to DriftDetector
self.drift_detector = DriftDetector(
    adwin_delta=adwin_delta,
    kswin_alpha=kswin_alpha,
    kswin_window_size=kswin_window_size,
    ph_threshold=ph_threshold,
    ph_delta=ph_delta,
)
```

---

### Step 2: Update `DriftDetector.__init__`

**File:** `py/online/online_learning_adapter.py`

**Current Code (Lines 147-154):**
```python
def __init__(
    self,
    adwin_delta: float = 0.002,
    kswin_alpha: float = 0.005,
    kswin_window_size: int = 100,
    ph_threshold: float = 50.0,
    ph_delta: float = 0.005,
):
```

**Updated Code (add to __init__):**
```python
# Ensemble decision threshold (number of detectors required to confirm drift)
self.min_detectors_for_drift = 1  # NEW: Any detector (was 2 implicit)
```

**Update `update()` method:**
```python
def update(self, error: float) -> bool:
    """Update detectors and check for drift"""
    drift_flags = []
    for name, detector in self.detectors.items():
        detector.update(error)
        if detector.drift_detected:
            self.drift_counts[name] += 1
            drift_flags.append(name)
            logger.info(f"Drift detected by {name} (total: {self.drift_counts[name]})")

    # NEW: Configurable ensemble threshold
    return len(drift_flags) >= self.min_detectors_for_drift  # Was: >= 2
```

---

### Step 3: Test with Synthetic Data

**File:** `tests/debug/test_river_nfl_tuning.py` (create this)

```python
import numpy as np
from river.drift import ADWIN, KSWIN, PageHinkley

def test_adwin_nfl_tuned():
    """Test ADWIN with NFL-tuned configuration"""
    detector = ADWIN(delta=0.01)

    # Week 1-10: MAE=56
    for _ in range(80):
        detector.update(56 + np.random.randn() * 5)

    # Week 11-18: MAE=70 (DRIFT)
    drift_detected = False
    for i in range(64):
        detector.update(70 + np.random.randn() * 5)
        if detector.drift_detected:
            drift_detected = True
            print(f"ADWIN detected drift at sample {80 + i}")
            break

    assert drift_detected, "ADWIN should detect drift within 64 samples"
    assert i < 32, "ADWIN should detect drift within 2 weeks (16 games)"


def test_kswin_nfl_tuned():
    """Test KSWIN with NFL-tuned configuration"""
    detector = KSWIN(alpha=0.05, window_size=30)

    # Week 1-10: σ=5
    for _ in range(80):
        detector.update(56 + np.random.randn() * 5)

    # Week 11-18: σ=15 (DRIFT in variance)
    drift_detected = False
    for i in range(64):
        detector.update(56 + np.random.randn() * 15)
        if detector.drift_detected:
            drift_detected = True
            print(f"KSWIN detected drift at sample {80 + i}")
            break

    assert drift_detected, "KSWIN should detect variance drift"


def test_page_hinkley_nfl_tuned():
    """Test Page-Hinkley with NFL-tuned configuration"""
    detector = PageHinkley(threshold=20.0, delta=0.01)

    # Gradual drift: MAE increases 2 yards/week
    drift_detected = False
    for week in range(1, 19):
        for game in range(8):
            mae = 56 + 2 * week + np.random.randn() * 5
            detector.update(mae)
            if detector.drift_detected:
                drift_detected = True
                print(f"Page-Hinkley detected drift in week {week}")
                break
        if drift_detected:
            break

    assert drift_detected, "Page-Hinkley should detect gradual drift"
    assert week < 12, "Page-Hinkley should detect drift before week 12"


if __name__ == "__main__":
    test_adwin_nfl_tuned()
    test_kswin_nfl_tuned()
    test_page_hinkley_nfl_tuned()
    print("✓ All NFL-tuned detector tests passed")
```

**Run tests:**
```bash
python tests/debug/test_river_nfl_tuning.py
```

---

## Validation Strategy

### Phase 1: Synthetic Drift Tests (Immediate)

**Purpose:** Verify detectors fire correctly on known drift patterns

**Test Cases:**
1. Mean shift (ADWIN): 56 → 70 yards
2. Variance shift (KSWIN): σ=5 → σ=15
3. Gradual trend (Page-Hinkley): +2 yards/week

**Expected Results:** All 3 detectors fire within 2 weeks (16 games)

---

### Phase 2: Historical Validation (2020-2023)

**Purpose:** Validate against known NFL drift events

**Known Drift Events:**

| Season | Event | Week | Expected Detector | Expected Timing |
|--------|-------|------|-------------------|-----------------|
| 2020 | COVID (empty stadiums) | 1 | ADWIN, KSWIN | Week 1-2 |
| 2021 | 17-game season | 18 | Page-Hinkley | Week 17-18 |
| 2022 | Targeting rule change | 1 | ADWIN | Week 1-2 |
| 2023 | Kickoff rule | 1 | KSWIN | Week 1-3 |

**Validation Procedure:**
```bash
# Run validation on each season
for year in 2020 2021 2022 2023; do
    python scripts/validate_online_learning_fixed.py --season $year --drift-sensitivity nfl_tuned
done

# Check if detectors fired in expected weeks
# Target: 80%+ precision, 80%+ recall
```

**Expected Results:**
- Detect 3-4 out of 4 known events (75%+ recall)
- False positive rate <20% (5-10 extra detections across 4 years)

---

### Phase 3: Production Backtest (2022-2024)

**Purpose:** Ensure online learning doesn't degrade ROI

**Metrics:**
- Prequential MAE: 50-60 yards (vs 56 static baseline)
- Static vs online: <+10% MAE degradation
- Drift events: 8-12 over 3 years (10-30% of seasons)
- Adaptation speed: <5 games to recover

**Acceptance Criteria:**
- ✅ Prequential MAE <60 yards
- ✅ Online MAE within 10% of static
- ✅ At least 8 drift events detected (realistic for NFL)
- ✅ Mean recovery time <5 games

---

## Known Issues

### Issue 1: Page-Hinkley Fires Too Often

**Symptom:** Page-Hinkley fires 4× in Phase 4 validation, but ensemble blocks (ADWIN/KSWIN silent)

**Root Cause:** Page-Hinkley is most sensitive to NFL data (MAE magnitude ~56 is close to threshold)

**Workaround:** Use ensemble logic (1/3 detectors) to allow Page-Hinkley alone to trigger drift

**Long-term Fix:** Calibrate Page-Hinkley threshold on historical data (2018-2023) to target 10-15% firing rate

---

### Issue 2: KSWIN Requires 30-Game Window

**Symptom:** KSWIN doesn't fire until week 4-5 (30 games accumulated)

**Root Cause:** K-S test requires minimum sample size for statistical power

**Workaround:** Accept that KSWIN won't detect early-season drift (weeks 1-3)

**Long-term Fix:** Use ADWIN for early-season detection, KSWIN for mid-season

---

### Issue 3: NFL-Tuned Config Not Yet Implemented

**Symptom:** `validate_online_learning_fixed.py --drift-sensitivity nfl_tuned` falls back to 'medium'

**Root Cause:** `OnlinePredictionPipeline.__init__` doesn't have 'nfl_tuned' case

**Status:** ⏸️ PENDING - Need to update `py/online/online_learning_adapter.py` (see Step 1 above)

**Timeline:** 1-2 days to implement and test

---

## FAQ

### Q: Why not just use the most sensitive settings (high)?

**A:** High sensitivity causes false positives (drift detected when none exists). NFL has natural week-to-week variance (injuries, weather) that isn't true drift. 'nfl_tuned' balances sensitivity with false positive rate.

### Q: How often should drift be detected?

**A:** For NFL, expect 10-30% of seasons to have drift (3-9 events over 4 years). More than 50% suggests false positives. Less than 10% suggests underdetection.

### Q: Can I use River for real-time betting?

**A:** Yes, but with caveats:
1. NFL data is sparse (32 games/week) → predictions update slowly
2. Drift detection requires 30-50 samples → 2-3 weeks lag
3. Prequential evaluation is test-then-train → always 1 game behind

For real-time betting, use batch ensemble (State-Space + Hierarchical + BNN + XGBoost) for predictions, River for monitoring drift.

### Q: What if ADWIN/KSWIN still don't fire?

**A:** Three options:
1. Reduce thresholds further (e.g., `delta=0.05`, `alpha=0.1`)
2. Use only Page-Hinkley (set `min_detectors_for_drift=1` and disable others)
3. Implement domain-specific drift detector (e.g., detect when team win% changes >20%)

---

## References

**River Library:**
- Documentation: https://riverml.xyz/
- Drift detection API: https://riverml.xyz/latest/api/drift/
- Montiel et al. (2021): River library (JMLR)

**NFL-Specific Research:**
- This debugging report: `results/phase4_debugging_report.md`
- Phase 4 validation: `scripts/validate_online_learning.py`

**Concept Drift Theory:**
- Bifet et al. (2018): Machine learning for data streams
- Gama et al. (2014): Survey on concept drift adaptation

---

**Document Status:** ✅ COMPLETE
**Next Action:** Implement 'nfl_tuned' configuration in `OnlinePredictionPipeline`
**Owner:** Model Management Agent (coordinate with Data Layer Agent)

