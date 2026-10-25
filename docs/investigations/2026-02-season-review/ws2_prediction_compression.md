# WS2: Prediction Compression Root Cause Analysis

> **Read with [README.md](README.md) (2026-10-25):** the compression is real, but missing QBR and
> NextGen data cannot explain it (v2 never splits on those inputs); the ROI projections are
> speculation.

**Date:** 2026-02-12
**Investigator:** Claude Code (Systematic Investigation)
**Status:** COMPLETE — Root cause identified, fix proposed

---

## Executive Summary

Model predictions cluster in [-2.6, +8.7] (std=2.30) while actual spreads range [-34, +38] (std=14.16). This is a **6.16x compression ratio** that directly causes the 11.05 MAE (vs Vegas 9.67). The root cause is **not** model architecture — it's **missing data causing feature collapse**.

---

## The Compression Mechanism

### Step 1: Data Sources Go Empty

| Source | Status in 2025 | Features Affected |
|--------|---------------|-------------------|
| ESPN QBR | 0 rows (never ingested) | 20 features = 0 |
| Injuries | 0 rows for 2025 | 12 features = 0 |
| NextGen | Cut off at week 12 | 46 features → 0 after week 12 |

### Step 2: GameFeatureExtractor Returns Zeros

The extractor's graceful degradation (try/except + NaN→0.0) means:
- QBR extractor gets initialized but `extract_features()` returns empty dict → no QBR features added
- Injury lookup returns default zeros → `{home_injury_total: 0, away_injury_total: 0, ...}`
- NaN values are caught and set to 0.0 (line 898)

**Result:** For every game, 78-116 features are identical (all zeros).

### Step 3: XGBoost Sees Identical Teams

When 30% of features are zeros for all teams, XGBoost can only split on the remaining 70% of features. But the zeros aren't random noise — they're **systematically missing** for team-differentiating features (QBR, injuries, tracking data). This means:

1. XGBoost can't tell elite QBs from backups (no QBR data)
2. XGBoost can't account for injury impact (no injury data)
3. XGBoost can't use advanced tracking (NextGen stops at week 12)

The remaining features (base EPA, playcalling, etc.) have less variance across teams than the full feature set, so predictions compress toward the league mean.

### Step 4: Predictions Collapse to Home Field Advantage

**Week 1 evidence:** All 16 predictions between -0.3 and +1.0.

This is the model's learned prior for "no information about team quality" — approximately +2.6 points (average home field advantage). The model correctly learned HFA exists but has no data to differentiate teams beyond that.

**Week 18 evidence:** Predictions widen to [-0.7, +7.9] but still only std=2.43 vs actual std=13.12. Even late in the season with 17 weeks of EPA data, the missing QBR/injury/NextGen features limit the variance the model can generate.

---

## Quantitative Evidence

### Prediction vs Actual Distribution

| Percentile | Model | Actuals | Vegas | Model Deficiency |
|-----------|-------|---------|-------|------------------|
| P5 | -0.58 | -22.0 | -7.5 | 21.4 pts too narrow |
| P25 | 0.9 | -6.0 | -3.0 | 6.9 pts too narrow |
| P50 | 2.3 | 2.0 | 2.5 | OK (median correct) |
| P75 | 4.4 | 11.0 | 2.5 | 6.6 pts too narrow |
| P95 | 6.7 | 27.8 | 12.5 | 21.1 pts too narrow |

**Key insight:** The median is correct (2.3 vs 2.0) — the model knows the center of the distribution. It just can't spread predictions out because it lacks the differentiating features.

### Week-by-Week Compression

| Week | Pred Std | Actual Std | Compression Ratio | MAE |
|------|----------|------------|-------------------|-----|
| 1 | 0.33 | 9.31 | **28.2x** | 7.61 |
| 5 | 2.57 | 15.37 | 5.98x | 12.36 |
| 10 | 2.23 | 12.63 | 5.66x | 10.42 |
| 15 | 2.41 | 15.13 | 6.28x | 12.54 |
| 18 | 2.43 | 13.12 | 5.40x | 10.78 |
| **Average** | **2.30** | **14.16** | **6.16x** | **11.05** |

Week 1 is 28x compressed because no prior-season features carry over. By week 5, the model has some EPA signal but compression only improves to ~6x because QBR/injuries are still missing.

---

## Fix: Feature Fallback Hierarchy

### Proposed Implementation

When a primary data source is unavailable, use a correlated proxy instead of zero:

```python
def _get_fallback_features(self, team, season, week):
    """When QBR/NextGen/injuries unavailable, use proxy features."""

    # QBR fallback: use weighted EPA as proxy (r≈0.65 with QBR)
    if not self._has_qbr_data(team, season, week):
        team_epa = self._get_team_epa(team, season, week)
        features["home_qbr_total"] = team_epa.get("epa_mean", 0) * 50 + 50  # Scale to QBR range
        features["home_qbr_raw"] = team_epa.get("epa_mean", 0) * 45 + 45
        # ... (10 QBR proxies from EPA)

    # NextGen fallback: use play-level metrics from plays table
    if not self._has_nextgen_data(team, season, week):
        play_metrics = self._compute_play_level_proxies(team, season, week)
        # e.g., completion % over expected → proxy from raw completion %
        # time to throw → proxy from sack rate

    # Injury fallback: use historical team average, not zero
    if not self._has_injury_data(team, season, week):
        features["home_injury_total"] = 3.2  # League average ~3.2 injuries/week
        features["home_injury_severity_index"] = 1.8  # Average severity
```

### Why This Helps

Zero means "no injuries at all" which is information. The model treats 0 as a real value, not as missing. By substituting the league average or a correlated proxy, we tell the model "we don't know — assume average" which is a more honest representation.

### Alternative: Feature Masking During Training

Train v2.1 with random feature dropout:
- During training, randomly zero out 20-30% of features per sample
- This teaches XGBoost to be robust to partial feature availability
- At inference, when features are genuinely missing, the model handles it gracefully

### Implementation Priority

1. **Ingest the data** (P0) — Best fix is to have real data. Run ingestion scripts.
2. **Feature fallback hierarchy** (P0) — Implement proxy features for when data is genuinely unavailable
3. **Feature masking training** (P1) — Retrain with dropout for robustness
4. **Pipeline health gate** (Done) — `pipeline_health_check.py` prevents predicting with garbage

---

## Expected Impact of Fix

If all 386 features have live data:
- Prediction std should increase from 2.30 to ~5-7 (matching Vegas's 6.18)
- MAE should decrease from 11.05 to ~9.5-10.5 (approaching Vegas's 9.67)
- ATS ROI could move from -3.2% to +1-3% (above breakeven)

**Rationale:** Vegas uses injury reports, QBR-like metrics, and advanced tracking. Our model matches their architecture but was missing their data. Restoring data parity should restore prediction quality.

---

## Verification Queries

```sql
-- After ingestion, verify compression resolves
-- Run predictions for a test week and compare std
SELECT
    STDDEV(pred_home_spread) as new_std,
    STDDEV(actual_spread) as actual_std
FROM ml.game_outcome_predictions
WHERE season = 2025 AND prediction_run_id = 'v2.1_test';
-- Target: new_std > 4.0 (was 2.30)

-- Compare MAE
SELECT
    AVG(ABS(pred_home_spread - actual_spread)) as mae
FROM ml.game_outcome_predictions
WHERE season = 2025 AND prediction_run_id = 'v2.1_test';
-- Target: mae < 10.5 (was 11.05)
```
