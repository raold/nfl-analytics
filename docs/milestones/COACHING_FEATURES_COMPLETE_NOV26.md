# Coaching Features Integration Complete

**Date:** November 26, 2025
**Status:** COMPLETE
**Author:** Claude Code (Opus 4.5)

## Summary

Completed full implementation of OC (Offensive Coordinator) coaching features for XGBoost v5 integration:
1. Wikipedia coaching data scrape (2018-2025)
2. PostgreSQL data loading
3. OC feature generation module
4. XGBoost v5 integration

## Components Delivered

### 1. Wikipedia Coaching Scraper
- **Script:** `scripts/scrape_coaching_wikipedia.py` (existing, extended)
- **Output:** `data/coaching_staff_2018_2025_wikipedia_full.csv`
- **Coverage:**
  - 263 team-seasons (2018-2025)
  - 97.7% Head Coaches
  - 96.1% Offensive Coordinators
  - 93.4% Defensive Coordinators
  - 97.3% ST Coordinators

### 2. Data Loading Script
- **Script:** `scripts/load_coaching_data.py` (NEW - 290 lines)
- **Functions:**
  - `load_coaching_data()` - CSV parsing with name cleaning
  - `insert_coach_metadata()` - Coach biographical records
  - `insert_coaching_staff()` - Team-season coaching assignments
  - `refresh_materialized_view()` - `coaching_continuity` view refresh
  - `validate_load()` - Data quality checks

### 3. OC Features Module
- **Script:** `py/features/oc_features.py` (NEW - 341 lines)
- **Class:** `OCFeatureGenerator`
- **Features Generated:**

| Feature | Description | Range |
|---------|-------------|-------|
| `oc_continuity` | Binary: 1 if same OC as last season | 0-1 |
| `games_with_oc` | Games played with current OC | 0-17 |
| `oc_tenure_seasons` | Consecutive seasons with OC | 1-10+ |
| `new_oc_adjustment` | Learning curve decay factor | 0.0-0.8 |

**Adjustment Formula:**
```python
adjustment = 0.8 * exp(-games / halflife_weeks)
# halflife_weeks default = 4.0
# Starts at 0.8 for week 1, decays to ~0 by mid-season
```

### 4. XGBoost v5 Integration
- **Script:** `scripts/train_xgboost_v5_integrated.py` (MODIFIED)
- **Changes:**
  - Added `from py.features.oc_features import OCFeatureGenerator`
  - Initialized `self.oc_features` in constructor
  - Added Track D: OC Coaching Features section in `engineer_features()`
  - Added OC features to `select_feature_columns()`

## Validation Results

### OC Features Direct Test
```
KC 2024 W1: continuity=1, games=0, tenure=2, adj=0.000 | OC: Matt Nagy
KC 2024 W5: continuity=1, games=4, tenure=2, adj=0.000 | OC: Matt Nagy
BUF 2024 W1: continuity=0, games=0, tenure=1, adj=0.800 | OC: Joe Brady
BUF 2024 W5: continuity=0, games=4, tenure=1, adj=0.294 | OC: Joe Brady
PHI 2024 W1: continuity=0, games=0, tenure=1, adj=0.800 | OC: Kellen Moore
SF 2024 W1: continuity=1, games=0, tenure=6, adj=0.000 | OC: Kyle Shanahan
```

### Key Insights
- **KC (Matt Nagy):** OC retained from 2023 → no adjustment penalty
- **BUF (Joe Brady):** New OC replacing Ken Dorsey → 0.8 penalty at week 1, decaying to 0.29 by week 5
- **PHI (Kellen Moore):** New OC → same 0.8 learning curve penalty
- **SF (Kyle Shanahan):** 6-year tenure → maximum continuity benefit

## Database Schema

### coaching_staff table
```sql
CREATE TABLE coaching_staff (
    team VARCHAR(3),
    season INTEGER,
    week INTEGER,  -- NULL for season-level records
    head_coach VARCHAR(100),
    offensive_coordinator VARCHAR(100),
    defensive_coordinator VARCHAR(100),
    special_teams_coordinator VARCHAR(100)
);
```

### coaching_continuity materialized view
Pre-computes year-over-year continuity metrics for efficient queries.

## Expected Impact

### Model Performance
- Early-season predictions: +2-3 yard MAE improvement (weeks 1-4)
- New OC teams: Better calibration of uncertainty
- Continuity bonus: Stable teams predicted with tighter intervals

### Feature Importance (Expected)
- `oc_continuity`: High importance for week 1-4 predictions
- `new_oc_adjustment`: Captures learning curve variance
- `oc_tenure_seasons`: Long-term system familiarity

## Files Changed/Created

| File | Status | Lines |
|------|--------|-------|
| `scripts/load_coaching_data.py` | NEW | 290 |
| `py/features/oc_features.py` | NEW | 341 |
| `scripts/train_xgboost_v5_integrated.py` | MODIFIED | +35 |
| `data/coaching_staff_2018_2025_wikipedia_full.csv` | NEW | 264 |

## Next Steps

1. Run full XGBoost v5 training with OC features
2. Measure MAE improvement vs baseline
3. Analyze OC feature SHAP importance
4. Add DC (Defensive Coordinator) features if beneficial

## Related Work

- **Bug #2 Fix:** Early-season fallback in ensemble_v6_production.py (same session)
- **CQL Training:** 4th down agent grid search (running in background)
