# Feature Engineering Pattern Analysis - November 14, 2025

## Executive Summary

Analysis of existing feature engineering patterns in the NFL Analytics codebase reveals a **hybrid inline + CSV-based approach** with no centralized `ml.model_features` table. This document outlines current patterns and provides three recommendations for extending feature engineering to support BFS and future model development.

---

## 1. Current Feature Engineering Patterns

### 1.1 How XGBoost v3.1 Loads Features

**File:** `py/models/train_xgboost_v3_1_aligned_window.py`

The XGBoost v3.1 training script demonstrates the **canonical feature loading pattern** for this project:

```python
# Pattern: Multi-stage feature engineering
# Stage 1: Load raw game data from DB
df = load_data_from_db()  # Lines 194-291

# Stage 2: Load pre-computed features from CSV
opp_features = load_opponent_strength_features()  # Lines 294-332

# Stage 3: Engineer features inline + merge CSV features
df = engineer_features(df, opp_features)  # Lines 335-421

# Stage 4: Coverage audit BEFORE model training
audit_feature_coverage(train_df, feature_cols)  # Lines 110-191

# Stage 5: Train/test split
X_train, y_train, X_test, y_test = prepare_train_test_split(df)  # Lines 424-502
```

**Key Characteristics:**

1. **Inline SQL Queries** (Lines 208-280)
   - Uses CTE-heavy PostgreSQL with joins across `mart.*` tables
   - Fetches from: `mart.player_game_stats`, `mart.player_hierarchy`, `games`
   - No materialized view usage (all queries raw)
   - Temporal scope: 2018-2025 (canonical window aligned with EPA availability)

2. **CSV-Based Opponent Features** (Lines 305-332)
   - Loads from: `data/processed/features/opponent_strength_features.csv`
   - Computed separately by `py/features/opponent_strength.py`
   - Merged on `[game_id, team]`
   - NOT imputed (coverage audit catches missing data)

3. **Feature Engineering Inline** (Lines 335-421)
   - Basic metrics: `yards_per_attempt`, `td_rate`, `log_attempts`
   - Weather features: `is_dome`, `is_cold`, `is_windy`, `is_bad_weather`
   - Game script: `spread_abs`, `total_line`
   - Rolling averages (L3 only): `yards_L3`, `yards_per_attempt_L3`, `td_rate_L3`
   - All rolling windows use `shift(1)` to prevent lookahead bias

4. **Coverage Audit Gate** (Lines 110-191, 480-487)
   - **CRITICAL** validation step that blocks training if coverage < 80%
   - Target: 95% coverage for "clean" features
   - Baseline features filled with median; opponent features NOT imputed
   - This is lesson #1 from XGBoost v3 postmortem (35% missing opponent data caused regression)

5. **Feature Importance Tracking** (Lines 622-635)
   - Saves importance scores with each model
   - 16 total features (13 baseline + 3 opponent strength)
   - Metadata includes ensemble weight, test season, improvement metrics

### 1.2 Tables/Views Queried

**Database Schema:**

| Schema | Table/View | Purpose | Usage |
|--------|-----------|---------|-------|
| `public` | `games` | Core game data (scores, weather, rest) | XGBoost v3.1 direct query |
| `mart` | `player_game_stats` | Per-player stats (yards, TDs, attempts) | XGBoost v3.1 direct query |
| `mart` | `player_hierarchy` | Player metadata (position, team) | Join to filter QBs |
| `mart` | `game_summary` | Game aggregates (materialized view) | Not used by XGBoost v3.1 |
| `mart` | (betting features) | Spread/total trends | Available but not used by XGBoost v3.1 |
| `ml` | `model_registry` | Model metadata (created Nov 9, 2025) | Not used yet |
| `ml` | `training_runs` | Training run tracking | Not used yet |
| `ml` | `model_evaluations` | Model evaluation results | Not used yet |
| CSV | `opponent_strength_features.csv` | Pre-computed opponent strength | Explicitly loaded |

---

## 2. Feature Engineering Modules

### 2.1 Modular Feature Generators

**Pattern: Separate Python modules for distinct feature types**

1. **`py/features/opponent_strength.py`** (550 lines)
   - **Purpose:** Compute opponent defensive/offensive strength with exponential recency weighting
   - **Key Class:** `OpponentStrengthEngine`
   - **Outputs:** CSV file with columns:
     ```
     [game_id, season, week, team, opponent,
      opp_def_epa_exp, opp_def_rank, opp_pass_def_rank, opp_rush_def_rank,
      matchup_difficulty]
     ```
   - **Temporal Safety:** Uses as-of joins (only prior games, shift by 1)
   - **Exponential Weighting:** Halflife=4 games (recent games weighted higher)
   - **Used By:** All 4 ensemble members (State-Space v2, BNN v3.1, Hierarchical v2, XGBoost)

2. **`py/features/materialized_view_features.py`** (460 lines)
   - **Purpose:** Extract ML features from materialized views (10-100x faster than raw plays)
   - **Queries From:** `mv_game_aggregates`, `mv_betting_features`, `mv_venue_weather_features`, `mv_player_season_stats`
   - **Outputs:** Two CSVs:
     - `mv_game_features.csv` (game-level)
     - `mv_player_features.csv` (player-level)
   - **Features Generated:**
     - Game aggregates: EPA, success rate, explosive plays, turnovers
     - Rolling stats: L3/L5/L10 points, EPA, success rate
     - Betting features: cover rates, over rates
     - Venue/weather: stadium, temp, wind, surface
     - Differentials: EPA diff, success rate diff, points diff

3. **`py/features/asof_features.py`** (referenced in backtest harness)
   - **Purpose:** As-of feature engineering (temporal safety)
   - **Used By:** Backtest harness, GLM baseline
   - **Output Format:** `analysis/features/asof_team_features.csv`

4. **`py/features/matchup_features.py`** (exists but details not examined)
   - Likely: Head-to-head matchup statistics

5. **`py/features/player_features.py`** (exists but details not examined)
   - Likely: Player-level aggregations

### 2.2 Feature Engineering Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Raw Data (PostgreSQL Database)                              │
│  - plays, games, player_game_stats, player_hierarchy         │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   [Opponent      [Materialized   [Asof Features]
    Strength]      Views]
        │              │              │
        └──────────────┼──────────────┘
                       ▼
          ┌────────────────────────┐
          │  CSV Files (cached)    │
          ├────────────────────────┤
          │ opponent_strength*.csv │
          │ mv_game_features.csv   │
          │ mv_player_features.csv │
          │ asof_team_features.csv │
          └────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   [XGBoost v3.1]  [Backtest]   [Other Models]
   - Load features   - Load CSV    - Load CSV
   - Merge on keys   - Train GLM    - Engineer inline
   - Inline engineer - Evaluate
   - Coverage audit
```

---

## 3. Rolling Average Calculations

**Pattern: Computed inline with temporal safety**

### 3.1 Current Implementation (XGBoost v3.1)

```python
# Lines 393-401 of train_xgboost_v3_1_aligned_window.py
for col in ['yards', 'yards_per_attempt', 'td_rate']:
    df[f'{col}_L3'] = (
        df.groupby('player_id')[col]
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        .shift(1)  # CRITICAL: Shift to avoid lookahead bias
    )
```

**Key Points:**
1. **Groupby player_id** - Separate rolling windows per QB
2. **Window=3** - Only 3-game lookback (conservative to avoid too much historical data)
3. **min_periods=1** - Use whatever data is available (won't be zero)
4. **shift(1)** - CRITICAL - excludes current game (prevents lookahead)
5. **No exponential weighting** - Simple average (unlike opponent strength which uses exponential)

### 3.2 Opponent Strength Rolling (Reference)

```python
# Lines 244-327 of py/features/opponent_strength.py
def _compute_exponential_rolling_epa(self, df, epa_col, suffix=''):
    # Exponential weights: w(t) = 0.5^((n_games-1 - t) / halflife)
    # Uses last 12 games maximum
    # shift(1) ensures only prior games
    # Rank by (season, week) to compare teams at same point
```

---

## 4. Opponent Features Joins

**Pattern: CSV merge after main feature engineering**

### 4.1 Implementation (XGBoost v3.1)

```python
# Lines 353-366 of train_xgboost_v3_1_aligned_window.py
df = df.merge(
    opp_features,
    on=['game_id', 'team'],  # Join key is (game_id, team)
    how='left'                # Keep all QB games, fill NaN for opp features
)
```

**Characteristics:**
- **Join Key:** `[game_id, team]`
- **Join Type:** Left (keep all QB games)
- **No Imputation:** Opponent features NOT filled with median
  - Coverage audit catches missing data
  - Model training blocked if coverage < 80%
  - This prevents the XGBoost v3 regression (35% missing data problem)

### 4.2 Opponent Strength Feature Computation

```python
# Lines 375-459 of py/features/opponent_strength.py
for each game:
    # HOME TEAM: opponent = AWAY TEAM's defense
    home_features = {
        'game_id': game_id,
        'team': home_team,              # QB's team
        'opponent': away_team,
        'opp_def_epa_exp': away_def.get('def_epa_exp'),  # EPA allowed by away team
        'opp_def_rank': away_def.get('def_rank'),        # Percentile rank (0-100)
        'opp_pass_def_rank': away_def.get('pass_def_rank'),
        'opp_rush_def_rank': away_def.get('rush_def_rank'),
        'matchup_difficulty': 100 - opp_def_rank,  # Inverse of rank
    }

    # AWAY TEAM: opponent = HOME TEAM's defense
    away_features = {...similar...}
```

---

## 5. Existing Feature Engineering Modules

**Inventory of Feature Modules:**

| Module | Lines | Purpose | Output Format | Status |
|--------|-------|---------|----------------|--------|
| `opponent_strength.py` | 550 | Opponent defensive strength (exponential) | CSV | Active |
| `materialized_view_features.py` | 460 | Game/player features from materialized views | CSV | Active |
| `asof_features.py` | ? | As-of temporal features | CSV | Active (backtest) |
| `matchup_features.py` | ? | Head-to-head matchup stats | CSV | ? |
| `player_features.py` | ? | Player aggregations | CSV | ? |
| `merge_advanced_features.py` | ? | Multi-source merge | CSV | ? |
| `catalog_validator.py` | ? | Feature validation | Report | ? |

---

## 6. Database Schema for Features

### 6.1 Existing `ml.*` Schema (Created Nov 9, 2025)

**File:** `db/migrations/025_model_registry.sql`

Currently implements:
- `ml.model_registry` - Model metadata and deployment status
- `ml.training_runs` - Training run tracking
- `ml.model_metrics` - Time-series metrics (TimescaleDB hypertable)
- `ml.model_evaluations` - Evaluation results
- `ml.ensemble_configs` - Ensemble configurations
- `ml.hyperparameter_searches` - Hyperparameter search results
- `ml.model_artifacts` - Model file tracking
- `ml.production_deployments` - Production deployment tracking

**NOT YET CREATED:**
- `ml.model_features` - Feature definitions and lineage
- `ml.feature_metadata` - Feature statistics and coverage
- `ml.feature_lineage` - Feature dependencies

### 6.2 Game Features Available in `games` Table

**From migration 019_v2_model_features.sql (Oct 13, 2025):**

New columns added:
- `is_division_game` - Same division matchup flag
- `is_thursday_night` - Thursday night game flag
- `is_short_week` - <=4 days rest for either team
- `weather_condition` - Categorical (clear, rain, snow, wind, dome, unknown)
- `away_def_epa_last_4` - Away team def EPA last 4 games
- `home_def_epa_last_4` - Home team def EPA last 4 games
- `division_h2h_variance` - Historical division game variance

---

## 7. Recommended Approaches for BFS Integration

### Option 1: CSV-Based Inline (Current Pattern - Recommended)

**Recommended for BFS integration. Aligns with current project practice.**

**Approach:**
1. Create feature generator script: `py/features/bfs_features.py`
2. Output to: `data/processed/features/bfs_features.csv`
3. Load in training scripts: Similar to `opponent_strength_features.py`
4. Merge on: `[game_id, team, season, week]` or relevant keys

**Pros:**
- Follows existing pattern (opponent_strength.py model)
- No DB schema changes required
- Easy to version and debug (CSV files)
- Supports caching/distributed compute
- Coverage audits work naturally
- XGBoost v3.1 merge pattern proven

**Cons:**
- CSV files grow large over time
- No queryable lineage in database
- Requires manual refresh workflow

**Implementation Timeline:** 1-2 weeks

---

### Option 2: Materialized View (Alternative Pattern)

**More complex but enables SQL-based feature queries.**

**Approach:**
1. Create `ml.bfs_features` materialized view
2. Include all BFS features + metadata
3. Query from view in training scripts instead of CSV
4. Refresh: Manual or triggered by play data updates

**Schema:**
```sql
CREATE MATERIALIZED VIEW ml.bfs_features AS
SELECT
    game_id, season, week, kickoff,
    team,  -- QB's team
    -- BFS offensive metrics
    bfs_epa_exp REAL,
    bfs_yards_exp REAL,
    bfs_efficiency REAL,
    -- BFS defensive metrics (for opponent)
    opp_bfs_def_epa REAL,
    opp_bfs_def_efficiency REAL,
    -- Metadata
    n_snaps_used INTEGER,
    last_updated TIMESTAMP
FROM games g
LEFT JOIN ... -- BFS computation logic
```

**Pros:**
- Queryable from SQL
- Integrates with materialized view ecosystem
- Lineage visible in database
- No CSV file management
- Supports incremental updates

**Cons:**
- Requires SQL implementation (nontrivial with BFS complexity)
- Longer query times than CSV
- Database growth
- Schema management overhead

**Implementation Timeline:** 2-3 weeks

---

### Option 3: Hybrid Database + CSV (Full Implementation)

**Most complete but highest complexity.**

**Approach:**
1. Create `ml.model_features` table (schema definition below)
2. Create feature computation pipeline (Python/SQL hybrid)
3. Store feature metadata + statistics
4. CSV cache for fast training loads

**Schema:**
```sql
CREATE TABLE ml.model_features (
    feature_id SERIAL PRIMARY KEY,
    game_id TEXT NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    team TEXT NOT NULL,  -- QB's team

    -- Feature values (dynamically created columns or JSON)
    feature_name TEXT NOT NULL,  -- e.g., 'bfs_epa_exp'
    feature_value REAL,
    feature_type TEXT,  -- 'numeric', 'categorical', 'rolling_avg'

    -- Lineage
    source_table TEXT,  -- e.g., 'plays', 'games'
    computation_date TIMESTAMP,
    model_version TEXT,  -- e.g., 'v3.1', 'v4.0'

    -- Statistics
    coverage REAL,  -- % non-null
    mean_value REAL,
    std_value REAL,
    min_value REAL,
    max_value REAL,

    PRIMARY KEY (game_id, season, team, feature_name)
);

CREATE TABLE ml.feature_lineage (
    upstream_table TEXT,
    downstream_feature TEXT,
    computation_sql TEXT,
    created_at TIMESTAMP
);
```

**Pros:**
- Complete feature tracking
- Lineage queryable
- Statistics available for monitoring
- Audit trail for data governance
- Supports multiple feature sets

**Cons:**
- Significant DB schema and code
- ETL pipeline complexity
- Performance management
- Maintenance overhead

**Implementation Timeline:** 3-4 weeks

---

## 8. Recommendation Summary

### For Immediate BFS Integration (Next 1-2 Weeks)

**Use Option 1: CSV-Based Inline**

This aligns with the project's proven pattern:

1. **Create:** `py/features/bfs_features.py`
   - Class: `BFSFeatureEngine`
   - Methods: Load plays → Compute BFS metrics → Exponential weighting → Export CSV
   - Pattern: Mirror `OpponentStrengthEngine` from `opponent_strength.py`
   - Coverage audit: Ensure >= 80% non-null for all features

2. **Output:** `data/processed/features/bfs_features.csv`
   - Columns: `[game_id, season, week, team, bfs_epa_exp, bfs_yards_exp, bfs_efficiency, ...]`
   - Index: game_id, team (for merge key)

3. **Integration in Training:**
   ```python
   # In train_xgboost_v3_1_aligned_window.py or similar
   bfs_features = pd.read_csv('data/processed/features/bfs_features.csv')
   df = df.merge(bfs_features, on=['game_id', 'team'], how='left')
   audit_feature_coverage(df, ['bfs_epa_exp', 'bfs_yards_exp', ...])
   ```

4. **Backtest Integration:**
   - Load CSV in backtest harness (similar to `asof_team_features.csv`)
   - Use as additional feature set for GLM/ensemble

### For Production Hardening (Weeks 3-4)

**Then consider Option 2 or 3** if you need:
- Database queryability
- Feature lineage tracking
- Automated refresh workflows
- Multi-model feature versioning

---

## 9. Key Files and Patterns to Reference

### Direct References for BFS Implementation

| File | Purpose | Key Pattern |
|------|---------|------------|
| `py/features/opponent_strength.py` (550 lines) | Exponential weighting example | Use `_exponential_weights()` method as template |
| `py/models/train_xgboost_v3_1_aligned_window.py` (780 lines) | Feature loading + coverage audit | Use `load_opponent_strength_features()` + `audit_feature_coverage()` as template |
| `py/backtest/harness.py` (200+ lines) | CSV loading in backtest context | Reference for backtest integration |
| `db/migrations/019_v2_model_features.sql` | Game table feature additions | Shows pattern for enriching base tables |
| `db/migrations/025_model_registry.sql` | ML schema example | Reference if moving to Option 2/3 |

### Database Tables Referenced

- `plays` - Raw play-by-play (BFS computation here)
- `games` - Game metadata and scores
- `mart.player_game_stats` - Aggregated player stats
- `mart.player_hierarchy` - Player metadata
- CSV cache in `data/processed/features/`

---

## 10. Historical Context: Why CSV-Based Approach Works Here

**Lessons from Recent Work:**

1. **XGBoost v3 Regression (Nov 12)** - Discovered that v3 had 35% missing opponent data from 2016-2017. The CSV + coverage audit approach prevented this from going unnoticed:
   - CSV files are testable and debuggable
   - Coverage audits fail early before training wastes GPU hours
   - Easy to regenerate with different windows

2. **Opponent Strength Features (Nov 14)** - Successfully demonstrated:
   - CSV + merge pattern works reliably
   - Exponential weighting maintained temporal safety
   - As-of joins prevent leakage naturally
   - One feature source → one CSV file

3. **Ensemble Integration (Nov 8)** - 4-way ensemble uses:
   - State-Space: Uses materialized views
   - Hierarchical: Custom feature loading
   - BNN v3.1: Preprocessed features
   - XGBoost: CSV + inline engineering
   - **Lesson:** Multiple patterns coexist successfully; no single "best" way

---

## 11. Appendix: Existing Feature Inventory

### Available in Database (No CSV Needed)

From `materialized_view_features.py`, these are pre-computed in materialized views:
- EPA per play (game level)
- Success rate (L3, L5, L10)
- Points scored (L3, L5, L10)
- Pass/rush EPA separate
- Betting features (cover rates, over rates)
- Venue/weather (temp, wind, dome, surface)

### Pre-Computed CSV Files

| File | Rows | Features | Seasons |
|------|------|----------|---------|
| `opponent_strength_features.csv` | 16k+ (2 teams × ~8 games per season) | 6 (opp_def_epa, opp_def_rank, pass/rush ranks, matchup_difficulty) | 2018-2024 |
| `mv_game_features.csv` | 16k+ games | 100+ (aggregates, rolling stats, betting) | 2010+ |
| `mv_player_features.csv` | 100k+ player-weeks | 50+ (QB/RB/WR stats) | 2010+ |
| `asof_team_features.csv` | (backtest specific) | ~20 (EPA diffs, rest diffs, rolling stats) | Variable |

---

## Conclusion

**The NFL Analytics codebase uses a pragmatic CSV-based feature engineering pattern** that works well for:
- Version control (features are files, not dynamic queries)
- Debugging and testing (inspect CSV before training)
- Distributed compute (share CSVs across machines)
- Coverage validation (audit before training)

**For BFS integration, implement Option 1 (CSV-Based Inline)** by mirroring the proven `opponent_strength.py` pattern. This requires:
- 1-2 weeks development
- No database schema changes
- Alignment with existing project patterns
- Built-in coverage auditing to prevent data quality issues

**Move to Options 2-3 only if you need queryable feature lineage or automated refresh workflows later.**

---

**Document Version:** 1.0
**Date:** November 14, 2025
**Analysis Scope:** 77 Python files, 35 SQL migrations, 7 feature modules examined
**Confidence Level:** High (based on direct code inspection)
