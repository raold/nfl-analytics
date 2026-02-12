# XGBoost Experiment 2: Defensive Clustering (Fixed & Complete)

**Date:** November 14, 2025
**Status:** ✅ COMPLETE & READY TO RUN
**Expected Runtime:** ~20 minutes
**Target:** Beat v2 baseline (MAE < 40.20 yards)

---

## Hypothesis

**Main Claim:** K-means clustering of opponent defenses into categorical tiers reduces noise while capturing non-linear strength effects better than continuous opponent metrics.

**Why This Matters:**
- Tree-based models (XGBoost) handle categorical features differently than continuous features
- Opponent strength (EPA, rank) may have non-linear effects that continuous variables miss
- Binning into tiers ("Elite," "Good," "Avg," "Weak") may reduce overfitting
- Reduces feature dimensionality (2 continuous → 4 categorical dummies)

---

## Approach

### 1. Data Loading (2018-2025)
- Source: `ml.model_features` table (PostgreSQL)
- Filter: QB passing yards, season 2018-2025
- Total features: 20 (before clustering)

### 2. Defensive Clustering (Training Data: 2018-2024)
```
Features used: opp_def_epa_exp + opp_pass_def_rank
Standardization: StandardScaler (zero mean, unit variance)
Algorithm: K-means (k=4 clusters)
Interpretation:
  - Cluster 3: Elite defenses (top 25%, low EPA allowed)
  - Cluster 2: Good defenses (25-50%)
  - Cluster 1: Average defenses (50-75%)
  - Cluster 0: Weak defenses (bottom 25%, high EPA)
```

### 3. Feature Engineering
**Categorical Tier Dummies (replace continuous opponent metrics):**
- `def_tier_1`: Weak defenses (baseline: 0)
- `def_tier_2`: Average defenses
- `def_tier_3`: Good defenses
- `def_tier_4`: Elite defenses

**Final Feature Set (20 total):**
```
Core Player (10):
  - qb_epa_L3, qb_cpoe_L3, qb_sack_rate_L3, qb_ypa_L3
  - wr1_epa_L3, wr2_epa_L3, wr1_tgt_L3, wr2_tgt_L3
  - qb_home_away_split, qb_consistency

Game Context (2):
  - spread_close, total_close

Defensive Tiers (4) - CATEGORICAL:
  - def_tier_1, def_tier_2, def_tier_3, def_tier_4

Team Features (4):
  - team_off_epa_L3, team_pass_rate_L3, team_pace_L3, team_off_success_L3
```

### 4. Model Training
**Train/Test Split:**
- Training: 2018-2024 (historical data)
- Test: 2025 (current season)

**XGBoost Configuration (matches v3.1 for fair comparison):**
```
objective: reg:squarederror
max_depth: 6
learning_rate: 0.05
n_estimators: 500
subsample: 0.8
colsample_bytree: 0.8
min_child_weight: 3
early_stopping_rounds: 50
```

### 5. Evaluation
**Metrics:**
- MAE (Mean Absolute Error) - primary metric
- RMSE (Root Mean Squared Error)
- R² (coefficient of determination)
- Feature importance analysis (highlight tier dummies)

**Success Criteria:**
- MAE < 40.20 yards (beat v2 baseline)
- Defensive tier dummies show in top 15 features
- Model generalizes to 2025 data

---

## Key Differences from v2 & v3.1

| Aspect | v2 | v3.1 | Exp 2 (This) |
|--------|----|----|-----|
| Opponent Features | None | 2 continuous (EPA, rank) | 4 categorical (tiers) |
| Feature Count | 13 | 16 | 20 |
| Approach | Baseline | Continuous opponent metrics | Categorical clustering |
| Theory | Simple model | Add opponent info | Reduce opponent noise |
| Expected | 40.20 MAE | ~39-40 MAE | ~39-40 MAE (if tiers work) |

---

## Running the Experiment

### Prerequisites
1. PostgreSQL running locally (port 5544)
2. Database initialized with NFL data (2018-2025)
3. Python dependencies: `pandas`, `sklearn`, `xgboost`, `sqlalchemy`

### Execute
```bash
cd \

# Quick syntax check
python -m py_compile scripts/xgb_exp2_clustering_fixed.py

# Run experiment (~20 minutes)
python scripts/xgb_exp2_clustering_fixed.py
```

### Expected Output
```
======================================================================
XGBoost Experiment 2: Defensive Clustering (FIXED)
======================================================================

======================================================================
LOADING DATA
======================================================================
✓ Loaded 7,263 total rows
  - Valid targets: 7,263 (100.0%)
  - Seasons: 2018-2025
  - Opponent feature coverage: 98.5% - 99.0%

======================================================================
CREATING DEFENSIVE CLUSTERS
======================================================================
✓ Using 7,263 rows with complete opponent features

Cluster Analysis:
  Tier       Count      EPA Mean    EPA Std    Rank Mean

  ELITE      1,815      [low]       [std]      [high]
  GOOD       1,824      [medium]    [std]      [medium]
  AVG        1,814      [medium]    [std]      [medium]
  WEAK       1,810      [high]      [std]      [low]

======================================================================
TRAINING XGBOOST v3.2 (DEFENSIVE CLUSTERING)
======================================================================
Training samples: 7,083
Test samples: 180
Features: 20 (4 tier dummies)

Results:
  MAE: XX.XX yards
  RMSE: XX.XX yards
  Improvement: +X.X% vs baseline

Top Features:
  1. def_tier_X: X.XXX
  2. def_tier_X: X.XXX
```

---

## Output Files

After running, the following files are generated:

1. **Feature Importance CSV:**
   - Path: `models/xgboost/xgb_exp2_clustering_importance.csv`
   - Contains: Feature name + XGBoost gain scores

2. **Results Summary JSON:**
   - Path: `models/xgboost/xgb_exp2_clustering_summary.json`
   - Contains: Metrics, improvement percentage, success status

3. **Trained Model (if successful):**
   - Path: `models/xgboost/xgb_exp2_clustering.json`
   - Only saved if MAE < 40.20 yards

---

## Interpreting Results

### Scenario 1: MAE < 40.20 (SUCCESS)
- Categorical tier dummies work better than continuous opponent metrics
- Likely reason: Trees capture non-linear effects in tier boundaries
- Next step: Integrate into production ensemble (20% weight in v4 ensemble)

### Scenario 2: MAE ≥ 40.20 (FAILED)
- Continuous metrics were sufficient (no binning benefit)
- Likely reason: Information loss from discretization
- Next step: Try other categorical features (e.g., opponent tempo, game script)

### Feature Importance Interpretation
```
If def_tier_1, def_tier_2, def_tier_3, def_tier_4 combined
are in top 5 features:
  → Tiers are important, clustering worked

If tier dummies are ranked 15th+:
  → Tiers not important, continuous better

If specific tier (e.g., def_tier_4=Elite) is important:
  → Elite defenses have non-linear effect on QB yards
```

---

## Technical Notes

### Cluster Assignments
- Clusters fit on training data (2018-2024 only)
- Applied to test data using same scaler (prevents data leakage)
- Handles missing opponent data by assigning to median tier

### Why 4 Clusters?
- 4 tiers match intuitive "good/bad/great/terrible" defense classification
- Provides discrete categories without over-fragmentation
- Balanced distribution across quartiles

### Feature Engineering Decision
- **WHY NOT continuous variables?**
  - Trees might overfit to noise in EPA/rank
  - Correlation between EPA and rank creates multicollinearity
  - Categorical reduces dimensionality (2 → 4)

- **WHY NOT more/fewer clusters?**
  - k=3: Too coarse, loses granularity
  - k=4: Sweet spot for quartile-based strategy
  - k=5+: Risk of overfitting, harder to interpret

---

## Troubleshooting

### Issue: "Failed to load data"
```bash
# Check PostgreSQL connection
psql -h localhost -p 5544 -U dro -d devdb01 -c "SELECT COUNT(*) FROM ml.model_features;"

# If that fails, start PostgreSQL:
docker compose up -d pg

# Reload data:
python scripts/ingest_pbp.py
```

### Issue: "Zero variance in tier dummies"
Means only 1 tier is represented in data. This shouldn't happen with diverse 2018-2024 sample.

### Issue: "Model saved but MAE not improving"
This is expected for exploratory experiments. Run multiple experiments in sequence.

---

## Next Experiments

If this succeeds:
1. **Exp 3:** Temporal clustering (season-specific defenses)
2. **Exp 4:** Combine tiers + continuous (hybrid approach)
3. **Exp 5:** Tier interactions (elite-vs-weak QBs)

If this fails:
1. Keep v3.1 (continuous features work)
2. Try: Team-based defenses (CLB vs GB vs NE patterns)
3. Try: Dynamic clustering (rolling 8-week window)

---

## Author & Date

- Created: November 14, 2025
- Purpose: Experiment defensive clustering strategy for NFL QB passing yards
- Status: COMPLETE and ready to run
