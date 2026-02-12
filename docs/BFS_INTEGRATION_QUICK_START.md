# BFS Feature Engineering - Quick Start Guide

**Status:** Ready for implementation
**Recommendation:** Option 1 - CSV-Based Inline (follows existing pattern)
**Timeline:** 1-2 weeks
**Complexity:** Moderate (mirror `opponent_strength.py`)

---

## Architecture Overview

```
plays table (raw)
     │
     ├─→ BFS Computation (new)
     │       ├─ Snap-level EPA
     │       ├─ Rolling averages (L3, L5)
     │       ├─ Exponential weighting (halflife=4)
     │       └─ Opponent strength calculation
     │
     └─→ CSV Export
             └─ data/processed/features/bfs_features.csv
                    │
                    ├─ Load in XGBoost training
                    ├─ Merge on [game_id, team]
                    ├─ Coverage audit (>80% required)
                    └─ Integrate into ensemble
```

---

## Step 1: Create Feature Generator

**File:** `py/features/bfs_features.py`

**Template from:** `py/features/opponent_strength.py` (use as reference)

**Key Methods:**

```python
class BFSFeatureEngine:
    """
    Compute team-level Big Friendly Score (BFS) metrics.

    BFS represents team offensive efficiency with:
    - EPA per snap
    - Rolling averages (L3, L5, L10)
    - Exponential recency weighting (halflife=4 games)
    - Temporal safety (as-of joins, shift by 1)
    """

    def __init__(self, db_host="localhost", db_port=5544, ...):
        """Initialize with DB connection config."""
        pass

    def load_play_data(self, start_season=2018, end_season=2024):
        """Load play-by-play data grouped by team-game."""
        # Query: SELECT game_id, season, week, team, epa, yards, ...
        # FROM plays JOIN games WHERE season >= 2018
        pass

    def compute_bfs_metrics(self, df):
        """
        Compute BFS offensive metrics:
        - bfs_epa_exp: Exponentially weighted EPA/snap
        - bfs_yards_exp: Exponentially weighted yards/snap
        - bfs_efficiency: Composite efficiency score

        Use only PRIOR games (shift by 1 to prevent lookahead).
        """
        pass

    def compute_opponent_strength(self, df):
        """
        For each game, compute opponent's BFS metrics.

        Example:
        - Home team (QB) vs Away team defense
        - Use away_team's bfs_epa_exp as opp_bfs_epa
        """
        pass

    def save_features(self, df, output_path):
        """Export to CSV."""
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} records to {output_path}")
```

**Output Format:**

```
game_id, season, week, team, opponent,
bfs_epa_exp, bfs_yards_exp, bfs_efficiency,
opp_bfs_epa_exp, opp_bfs_yards_exp, opp_bfs_efficiency,
n_snaps_prior, coverage_pct
```

---

## Step 2: Generate Features

**Command:**

```bash
python py/features/bfs_features.py \
    --output data/processed/features/bfs_features.csv \
    --start-season 2018 \
    --end-season 2024 \
    --verbose
```

**Expected Output:**

```
Loaded 16,500+ records (2 teams × ~8 games × 7 seasons)
Features: [bfs_epa_exp, bfs_yards_exp, bfs_efficiency, ...]
Coverage:
  bfs_epa_exp: 95.2% non-null
  bfs_yards_exp: 94.8% non-null
  ...
```

---

## Step 3: Integrate into XGBoost Training

**File:** `py/models/train_xgboost_v3_1_aligned_window.py` (modify)

**Change 1: Load BFS features (after line 332)**

```python
def load_bfs_features():
    """Load pre-computed BFS features from CSV."""
    features_path = Path('data/processed/features/bfs_features.csv')
    if not features_path.exists():
        raise FileNotFoundError(
            f"BFS features not found at {features_path}. "
            f"Run py/features/bfs_features.py first."
        )

    bfs_df = pd.read_csv(features_path)
    logger.info(f"✓ Loaded {len(bfs_df):,} BFS records")

    # Select only needed features
    bfs_features = bfs_df[[
        'game_id', 'team',
        'bfs_epa_exp', 'bfs_yards_exp', 'bfs_efficiency',
        'opp_bfs_epa_exp', 'opp_bfs_yards_exp', 'opp_bfs_efficiency'
    ]].copy()

    return bfs_features
```

**Change 2: Merge BFS features in engineer_features() (after line 366)**

```python
# Load BFS features
bfs_features = load_bfs_features()

# Merge into main dataframe
df = df.merge(
    bfs_features,
    on=['game_id', 'team'],
    how='left'
)

logger.info(f"✓ Merged BFS features, {len(df):,} rows preserved")
```

**Change 3: Add BFS features to feature list (line 442-461)**

```python
feature_cols = [
    # ... existing 16 features ...

    # New BFS features (v4)
    'bfs_epa_exp',
    'bfs_yards_exp',
    'bfs_efficiency',
    'opp_bfs_epa_exp',
    'opp_bfs_yards_exp',
    'opp_bfs_efficiency'
]
```

**Change 4: Update feature count in metadata (line 646)**

```python
metadata = {
    'model_name': 'XGBoost Passing v4 (Aligned Window 2018-2024, 22 Features)',
    # ... existing fields ...
    'num_features': 22,  # was 16
    'bfs_features': [
        'bfs_epa_exp', 'bfs_yards_exp', 'bfs_efficiency',
        'opp_bfs_epa_exp', 'opp_bfs_yards_exp', 'opp_bfs_efficiency'
    ],
}
```

---

## Step 4: Test Coverage Audit

**The coverage audit will automatically validate BFS features:**

```python
# Lines 480-487 already run audit
coverage_report = audit_feature_coverage(
    train_df,
    feature_cols,  # Now includes BFS features
    threshold=0.80,
    fail_on_low_coverage=True
)
```

**Expected Result:**

```
FEATURE COVERAGE AUDIT (CRITICAL GATE)
Threshold: 80% (minimum required)
Target: 95% (clean features)

  bfs_epa_exp                 92.5%  ✓ PASS   (missing: 1,240 / 16,800)
  bfs_yards_exp               91.8%  ✓ PASS   (missing: 1,380 / 16,800)
  bfs_efficiency              91.2%  ✓ PASS   (missing: 1,486 / 16,800)
  opp_bfs_epa_exp             88.3%  ✓ PASS   (missing: 1,976 / 16,800)
  ...
```

---

## Step 5: Backtest Integration

**File:** `py/backtest/harness.py` (optional, for GLM baseline)

**Add to DEFAULT_CONFIGS:**

```python
Config(
    name="core_plus_bfs",
    features=[
        "epa_diff_prior",
        "prior_epa_mean_diff",
        "epa_pp_last3_diff",
        "rest_diff",
        # NEW BFS features
        "bfs_epa_exp",
        "bfs_yards_exp",
        "opp_bfs_epa_exp",
    ],
    threshold=0.5,
),
```

**Usage:**

```bash
python py/backtest/harness.py \
    --features-csv data/processed/features/bfs_features.csv \
    --start-season 2018 \
    --end-season 2024 \
    --output-csv results/bfs_ablation.csv
```

---

## Step 6: Ensemble Integration

**File:** `py/ensemble/enhanced_ensemble_v6.py` (future)

**Expected pattern:**

```python
# Load BFS-augmented XGBoost v4
xgb_v4 = load_xgboost_v4()  # 22 features including BFS

# In ensemble:
ensemble_pred = (
    0.30 * state_space_v2.predict(X) +
    0.25 * bnn_v3_1_rushing.predict(X) +
    0.25 * hierarchical_v2.predict(X) +
    0.20 * xgb_v4.predict(X)  # Now with BFS features
)
```

---

## Temporal Safety Checklist

**CRITICAL: Ensure no lookahead bias**

- [ ] All rolling windows use `shift(1)` (exclude current game)
- [ ] Opponent metrics computed from PRIOR games only
- [ ] As-of joins: Use `game_id` + `kickoff` timestamp
- [ ] Coverage audit passes (>80%) before training
- [ ] Test on 2025 season (not in training window)

---

## Key Files to Reference

| File | Purpose | Key Pattern |
|------|---------|------------|
| `py/features/opponent_strength.py` | Template | `_exponential_weights()`, `_compute_exponential_rolling_epa()` |
| `py/models/train_xgboost_v3_1_aligned_window.py` | Integration | `load_opponent_strength_features()`, `audit_feature_coverage()` |
| `py/features/materialized_view_features.py` | Feature generation | CSV export, rolling stats computation |
| `db/migrations/025_model_registry.sql` | Schema ref | If moving to Option 2/3 later |

---

## Success Criteria

- [ ] BFS CSV generated with >80% coverage on all features
- [ ] XGBoost v4 trains without coverage audit failures
- [ ] v4 MAE < v3.1 baseline (validates BFS features add value)
- [ ] Feature importance shows BFS features >5% contribution
- [ ] Backtest GLM ablation shows BFS features improve ROI
- [ ] No temporal leakage detected (2025 season validation)

---

## Timeline

1. **Day 1-2:** Create `py/features/bfs_features.py` (mirror opponent_strength.py)
2. **Day 3:** Generate CSV, run coverage audit
3. **Day 4-5:** Integrate into XGBoost training
4. **Day 6:** Test on 2025 holdout, measure improvement
5. **Day 7:** Backtest integration (if needed)
6. **Day 8-10:** Ensemble integration + production validation

---

## Common Pitfalls to Avoid

1. **Missing `shift(1)`** - Will cause lookahead bias and inflated performance
2. **Filling NaN with median** - BFS features should fail coverage audit, not be imputed
3. **Not running coverage audit** - Will miss data quality issues (lesson from XGBoost v3)
4. **Using future games in opponent strength** - Use as-of joins only
5. **Inconsistent merge keys** - Must match exactly with main feature dataframe

---

## Questions?

See full analysis: `docs/FEATURE_ENGINEERING_PATTERN_ANALYSIS.md`

Key contacts:
- Feature generation: Data Layer Agent
- Model integration: Prediction Engine Agent
- Backtest validation: Model Management Agent
