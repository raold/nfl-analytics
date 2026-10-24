# Feature Engineering

**Location:** `nfl_analytics/features/`
**Purpose:** Feature engineering modules for NFL analytics and betting models
**Total Modules:** 30+ Python files

---

## 📋 Overview

This directory contains feature engineering modules that transform raw NFL data into predictive features for betting models. Features are designed to be **leakage-safe** using as-of-date logic to prevent look-ahead bias.

---

## 🎯 Key Principles

1. **No Data Leakage:** All features use as-of-date logic (no future information)
2. **Database Schema Standards:** Follow [DATABASE_SCHEMA_AUDIT.md](../../docs/database/DATABASE_SCHEMA_AUDIT.md)
   - Use `kickoff` not `game_date`
   - Use `spread_close` not `spread_line`
3. **Modular Design:** Each module focuses on a specific feature category
4. **Validation:** Built-in data quality checks and validation

---

## 📦 Feature Categories

### 1. Core As-Of Features
Leakage-safe features with proper temporal logic.

| Module | Features | Description |
|--------|----------|-------------|
| **asof_features.py** | Rolling stats, opponent strength | Core as-of-date features |
| **asof_features_enhanced.py** | 157 features | Enhanced feature set (v2.5) |
| **asof_features_v2.py** | Updated version | Improved as-of logic |
| **asof_snapshot.py** | Point-in-time | Snapshot features at specific dates |

**Usage:**
```python
from features.asof_features_enhanced import build_asof_features

# Build features as of specific date
df = build_asof_features(
    season_start=2020,
    season_end=2024,
    as_of_date='2024-12-01'
)
```

### 2. Bayesian Features
Features derived from Bayesian model posteriors.

| Module | Description |
|--------|-------------|
| **bayesian_features.py** | General Bayesian feature extraction |
| **bayesian_player_features.py** | Player-level Bayesian ratings |

**Usage:**
```python
from features.bayesian_player_features import extract_player_ratings

# Extract player ratings from Bayesian posteriors
ratings = extract_player_ratings(
    model_path='models/bayesian/passing_informative_priors_v1.rds',
    players=['Patrick Mahomes', 'Josh Allen']
)
```

### 3. Defensive Features (Phase 2, Oct 2025)
Advanced defensive matchup and situational features.

| Module | Features | Description |
|--------|----------|-------------|
| **defensive_coverage_quality.py** | 6 | Coverage metrics by zone/man |
| **defensive_pressure_effectiveness.py** | 8 | QB pressure and sack rates |
| **defensive_run_disruption.py** | 7 | Run defense metrics |
| **defensive_situational_strength.py** | 7 | Red zone, goal line, down conversions |
| **defensive_game_script_indicators.py** | 5 | Game script and garbage time |

**Usage:**
```python
from features.defensive_pressure_effectiveness import build_pressure_features

# Build QB pressure features
pressure_df = build_pressure_features(
    season_start=2020,
    season_end=2024
)
```

### 4. Matchup Features (Phase 2)
Player-specific matchup advantages.

| Module | Features | Description |
|--------|----------|-------------|
| **qb_pressure_matchups.py** | QB vs pass rush | QB performance under pressure |
| **receiver_coverage_matchups.py** | WR vs CB | Receiver-cornerback matchups |
| **line_matchup_strength.py** | OL vs DL | Offensive/defensive line strength |
| **target_share_coverage.py** | WR target share | Coverage-adjusted target metrics |

### 5. Environmental & Contextual (Phase 3, Oct 2025)
Weather, travel, venue, and situational features.

| Module | Features | Description |
|--------|----------|-------------|
| **environmental_contextual_features.py** | 15+ | Weather, travel, venue, rivalry |
| **vegas_line_bias_features.py** | 10+ | ⚠️ **Data leakage risk** - vegas-derived |
| **feature_interactions.py** | 5 categories | Weather×Travel, Rest×Travel, etc. |

**⚠️ Vegas Features Warning:**
The `vegas_line_bias_features.py` module includes implied probabilities derived from betting lines. This creates **circular dependency** and can cause data leakage. Use with caution and validate carefully.

### 6. Player-Specific Features

| Module | Description |
|--------|-------------|
| **player_features.py** | General player feature engineering |
| **merge_advanced_features.py** | Combine multiple feature sets |
| **matchup_features.py** | Player matchup features |

### 7. Utilities

| Module | Purpose |
|--------|---------|
| **catalog.py** | Feature catalog management |
| **catalog_validator.py** | Validate feature catalog |
| **feature_loader.py** | Load features from database |
| **feature_loader_parallel.py** | Parallel feature loading |

---

## 🚀 Quick Start

### Build Core Features

```python
from features.asof_features_enhanced import build_asof_features

# Build enhanced feature set
features = build_asof_features(
    season_start=2020,
    season_end=2024,
    validate=True  # Run validation checks
)

# Save to CSV
features.to_csv('analysis/features/asof_team_features.csv', index=False)
```

### Build Defensive Features

```python
from features.defensive_situational_strength import build_situational_defense_features

# Build red zone, goal line, 3rd/4th down features
defense_features = build_situational_defense_features(
    season_start=2020,
    season_end=2024
)
```

### Build Feature Interactions

```python
from features.feature_interactions import build_all_interaction_features

# Build weather×travel, rest×travel, altitude×rest
interactions = build_all_interaction_features(
    season_start=2020,
    season_end=2024
)
```

---

## 📊 Feature Engineering Pipeline

### Standard Pipeline

1. **Extract raw data** from PostgreSQL database
2. **Build as-of features** (no look-ahead)
3. **Add Bayesian features** (from model posteriors)
4. **Add defensive features** (matchup-specific)
5. **Create interactions** (weather×travel, etc.)
6. **Validate** (check for leakage, nulls, outliers)
7. **Save** to CSV or database

### Example: Full Pipeline

```python
import pandas as pd
from features.asof_features_enhanced import build_asof_features
from features.defensive_situational_strength import build_situational_defense_features
from features.feature_interactions import build_all_interaction_features

# Step 1: Core features
core = build_asof_features(2020, 2024)

# Step 2: Defensive features
defense = build_situational_defense_features(2020, 2024)

# Step 3: Interactions
interactions = build_all_interaction_features(2020, 2024)

# Step 4: Merge
full_features = core.merge(defense, on=['game_id', 'team'], how='left')
full_features = full_features.merge(interactions, on=['game_id', 'team'], how='left')

# Step 5: Validate
assert full_features.isnull().sum().sum() < 1000, "Too many nulls"
assert not (full_features.columns.str.contains('_future_')).any(), "Future leakage detected"

# Step 6: Save
full_features.to_csv('analysis/features/full_feature_set.csv', index=False)
```

---

## ⚠️ Data Leakage Prevention

### Critical Rules

1. **Never use future data:** All features must use data available before game time
2. **As-of-date logic:** Use `WHERE kickoff < as_of_date` in queries
3. **Rolling windows:** Use only past N games, not future games
4. **Vegas features:** Avoid derived implied probabilities (circular dependency)
5. **Validation:** Always run leakage detection tests

### Leakage Detection

```python
def detect_leakage(df, game_date_col='kickoff'):
    """Check for potential data leakage."""

    # Check for future-looking column names
    future_cols = [c for c in df.columns if 'future' in c.lower()]
    if future_cols:
        print(f"⚠️ Future columns detected: {future_cols}")

    # Check for impossibly high correlations
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    high_corr = (corr_matrix > 0.95) & (corr_matrix < 1.0)
    if high_corr.any().any():
        print(f"⚠️ Suspiciously high correlations detected")

    # Check for perfect predictions
    if 'target' in df.columns:
        for col in df.select_dtypes(include=[np.number]).columns:
            if col != 'target':
                r2 = np.corrcoef(df['target'], df[col])[0,1]**2
                if r2 > 0.90:
                    print(f"⚠️ {col} has R²={r2:.3f} with target")
```

---

## 📂 Database Schema Integration

All feature modules query from PostgreSQL database following schema standards:

**Connection:**
```python
import psycopg2

conn = psycopg2.connect(
    dbname='devdb01',
    user='dro',
    password='sicillionbillions',
    host='localhost',
    port=5544
)
```

**Key Tables:**
- `games` - Game metadata and betting lines
- `plays` - Play-by-play with EPA
- `mart.game_summary` - Materialized view with enriched data
- `mart.team_epa` - Per-game EPA summaries
- `mart.player_game_stats` - Player-level statistics

**Schema Standards:**
- Use `g.kickoff` not `g.game_date` or `g.gameday`
- Use `g.spread_close` not `g.spread_line`
- Weather stored as TEXT, requires parsing
- Team abbreviations: LAR (Rams), LAC (Chargers), not LA

See: [docs/database/DATABASE_SCHEMA_AUDIT.md](../../docs/database/DATABASE_SCHEMA_AUDIT.md)

---

## 🧪 Testing

### Unit Tests

```bash
# Test individual feature modules
pytest nfl_analytics/tests/test_asof_features.py -v

# Test all feature modules
pytest nfl_analytics/tests/test_features/ -v
```

### Integration Tests

```bash
# Test feature engineering pipeline end-to-end
pytest nfl_analytics/tests/test_feature_engineering_integration.py -v
```

---

## 📈 Feature Importance

Track which features contribute most to model performance:

```python
import xgboost as xgb

# Train model
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Get feature importance
importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance.head(20))
```

**Common Top Features:**
1. Recent performance (rolling averages)
2. Opponent defensive strength
3. Rest days
4. Home/away status
5. Weather conditions (for passing games)

---

## 🔄 Feature Updates

### Recent Work (Oct 2025)

**Phase 2.4:** Defensive Situational Strength (7 features)
- Red zone defense
- Goal line defense
- 3rd/4th down conversions
- Time-sensitive situations

**Phase 2.5:** Defensive Game Script Indicators (5 features)
- Game script classification
- Garbage time detection
- High leverage situations

**Phase 3:** Vegas + Environmental + Rivalry (30+ features)
- ⚠️ Vegas line bias (data leakage risk)
- Weather impact features
- Travel/rest features
- Divisional rivalry indicators

**Phase 3.5:** Feature Interactions (5 categories)
- Weather × Travel × Venue
- Rest × Travel
- Altitude × Rest
- Division × Rivalry
- Playoff Implications

---

## 📚 Related Documentation

- **[docs/database/DATABASE_SCHEMA_AUDIT.md](../../docs/database/DATABASE_SCHEMA_AUDIT.md)** - Schema standards
- **analysis/DATA_LEAKAGE_AUDIT_REPORT_UPDATED.md (no longer in the repo; see [HISTORY](../../docs/HISTORY.md))** - Data leakage findings
- **PHASE_5_COMPLETION_SUMMARY.md (no longer in the repo; see [HISTORY](../../docs/HISTORY.md))** - Phase 5 feature engineering summary
- **[CLAUDE.md](../../CLAUDE.md)** - Complete project documentation

---

## 🆘 Troubleshooting

**"Column 'game_date' does not exist"**
- Use `kickoff` instead (see DATABASE_SCHEMA_AUDIT.md)

**"High null count in features"**
- Check data availability for date range
- Verify materialized views are refreshed

**"Features correlate too highly with target"**
- Check for data leakage
- Verify as-of-date logic is correct

**"Different results on re-run"**
- Ensure deterministic data ordering
- Use explicit `ORDER BY` in queries
- Set random seeds if using sampling

---

**Last Updated:** November 1, 2025
**Maintained by:** NFL Analytics Team
**Total Modules:** 30 Python files
