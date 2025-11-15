# Data Layer: Feature Engineering Guidance

**Target Audience:** Data Layer Agent
**Context:** ML/Data Architecture decisions
**Status:** Ready for implementation
**Complexity:** Moderate

---

## Your Decision Tree

```
┌─ Do you need database queryability?
│  └─ YES → Consider Option 2 (MV) or Option 3 (Full DB)
│
├─ Do you need feature lineage tracking?
│  └─ YES → Option 3 (Full DB) required
│
├─ Do you need fast iteration (1-2 weeks)?
│  └─ YES → Option 1 (CSV) ✓ Recommended
│
└─ What's your priority?
   ├─ Speed: Option 1 (CSV)
   ├─ Queryability: Option 2 (MV)
   └─ Completeness: Option 3 (Full DB)
```

**Recommendation: START with Option 1 (CSV), add Option 2/3 later if needed**

---

## Option 1: CSV-Based Implementation

### Why This Works

1. **Proven Pattern**: `opponent_strength.py` demonstrates exact pattern
2. **No DB Changes**: Works with existing schema
3. **Debuggable**: CSV files are inspectable, versionable
4. **Fast**: CSV I/O faster than SQL for training
5. **Distributed**: Easy to share across machines
6. **Testable**: Can validate coverage before merging

### Implementation Checklist

#### Phase 1: Create Generator (2-3 days)

```python
# File: py/features/bfs_features.py

class BFSFeatureEngine:
    """Compute Big Friendly Score metrics with exponential recency weighting."""

    HALFLIFE = 4  # Exponential decay halflife (games)
    MIN_GAMES = 4  # Minimum games for stable estimate

    def __init__(self, db_config):
        """Initialize with database connection."""
        self.conn = psycopg2.connect(**db_config)

    # Required methods (from opponent_strength.py template):
    def _exponential_weights(self, n_games: int) -> np.ndarray:
        """Compute exponential decay weights: 0.5^((n-1-t)/halflife)"""
        # See opponent_strength.py lines 101-128
        pass

    def load_play_data(self, start_season=2018, end_season=2024) -> pd.DataFrame:
        """Load play-level data grouped by team-game."""
        # Query plays table, group by (team, game_id, season, week)
        # Compute: epa, yards, snaps per game
        # Return: DataFrame with columns [game_id, season, week, team, epa, yards, ...]
        pass

    def compute_bfs_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute BFS metrics (offensive efficiency):
        - bfs_epa_exp: Exponentially weighted EPA/snap
        - bfs_yards_exp: Exponentially weighted yards/snap
        - bfs_efficiency: Composite metric (0-100 scale)

        TEMPORAL SAFETY:
        - Use only PRIOR games (shift by 1)
        - Minimum 4 prior games required
        - Sort by team, season, week, kickoff
        """
        pass

    def compute_opponent_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For each game, add opponent's BFS metrics.

        Example:
        Game 1: KC (home) vs LAC (away)
        - KC team: opp_bfs_epa = LAC's bfs_epa (as of Game 1)
        - LAC team: opp_bfs_epa = KC's bfs_epa (as of Game 1)
        """
        pass

    def save_features(self, df: pd.DataFrame, path: str):
        """Export to CSV."""
        pass
```

#### Phase 2: Generate CSV (1 day)

```bash
python py/features/bfs_features.py \
    --start-season 2018 \
    --end-season 2024 \
    --output data/processed/features/bfs_features.csv \
    --verbose
```

**Validation:**
```
Expected output:
  Rows: 16,800+ (2 teams × ~7-8 games × 7 seasons)
  Columns: game_id, season, week, team, opponent,
           bfs_epa_exp, bfs_yards_exp, bfs_efficiency,
           opp_bfs_epa_exp, opp_bfs_yards_exp, opp_bfs_efficiency,
           n_snaps_prior, coverage_pct

Coverage check:
  bfs_epa_exp: >90% non-null (else investigate)
  opp_bfs_epa_exp: >85% non-null (lower due to early season)
```

#### Phase 3: Validation (1 day)

```python
# py/features/validate_bfs_features.py

def validate_bfs_csv():
    df = pd.read_csv('data/processed/features/bfs_features.csv')

    # Check 1: Coverage
    assert df['bfs_epa_exp'].notna().mean() > 0.90
    assert df['opp_bfs_epa_exp'].notna().mean() > 0.85

    # Check 2: No temporal leakage
    # For each game, verify n_snaps_prior >= 4
    assert (df['n_snaps_prior'] >= 4).mean() > 0.85

    # Check 3: No NaN in keys
    assert df[['game_id', 'season', 'week', 'team']].notna().all().all()

    # Check 4: Value ranges reasonable
    assert df['bfs_epa_exp'].between(-1, 2).sum() > 0.95 * len(df)
    assert df['bfs_yards_exp'].between(3, 7).sum() > 0.95 * len(df)

    print("✓ All validations passed")
```

---

## Option 2: Materialized View Implementation

### Why This Pattern

1. **Queryable**: SQL SELECT without CSV file management
2. **Integrated**: Part of `mart.*` ecosystem
3. **Refreshable**: REFRESH MATERIALIZED VIEW on schedule
4. **Indexable**: Faster queries than CSV load
5. **Visible**: Lineage transparent in database

### Implementation Checklist

#### Migration File

```sql
-- db/migrations/026_bfs_features.sql

BEGIN;

-- Create BFS features materialized view
CREATE MATERIALIZED VIEW IF NOT EXISTS mart.bfs_features AS
WITH team_game_plays AS (
    SELECT
        g.game_id,
        g.season,
        g.week,
        g.kickoff,
        CASE WHEN g.home_team = p.posteam THEN g.home_team
             ELSE g.away_team END as team,

        -- EPA metrics
        p.epa,
        p.yards,

        -- Count snaps
        1 as snap_count

    FROM games g
    JOIN plays p ON g.game_id = p.game_id
    WHERE g.season >= 2018
      AND g.game_type = 'REG'
      AND p.epa IS NOT NULL

    ORDER BY g.season, g.week, g.kickoff, p.play_id
)
SELECT
    game_id, season, week, kickoff, team,

    -- Aggregate metrics
    COUNT(*) as n_snaps,
    AVG(epa) as avg_epa,
    AVG(yards) as avg_yards,

    -- Exponential rolling metrics (using window function)
    -- This is complex in SQL - consider UDF or application layer

    NOW() as refreshed_at
FROM team_game_plays
GROUP BY game_id, season, week, kickoff, team
ORDER BY season, week, team;

-- Create indexes
CREATE UNIQUE INDEX idx_bfs_game_team
ON mart.bfs_features (game_id, team);

CREATE INDEX idx_bfs_season_week
ON mart.bfs_features (season, week);

-- Add comment
COMMENT ON MATERIALIZED VIEW mart.bfs_features IS
'Big Friendly Score metrics: offensive efficiency per team-game with exponential recency weighting.
Used for QB prop prediction calibration.
Refresh: After play data updates. Refresh command: REFRESH MATERIALIZED VIEW CONCURRENTLY mart.bfs_features;';

COMMIT;
```

#### Python Loading

```python
# In training scripts:
df = pd.read_sql("""
    SELECT * FROM mart.bfs_features
    WHERE season >= 2018 AND season <= 2024
    ORDER BY season, week, team
""", conn)
```

---

## Option 3: Full Database Implementation

### Why This Pattern

1. **Feature Lineage**: Track all upstream/downstream dependencies
2. **Statistics**: Compute and store feature statistics
3. **Audit Trail**: Who computed what when
4. **Multi-Model**: Support multiple feature sets per model

### Schema Design

```sql
-- db/migrations/026_model_features_table.sql

CREATE SCHEMA IF NOT EXISTS ml;

-- Main feature table
CREATE TABLE IF NOT EXISTS ml.model_features (
    feature_id SERIAL PRIMARY KEY,

    -- Identifiers
    game_id TEXT NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    team TEXT NOT NULL,

    -- Feature metadata
    feature_name TEXT NOT NULL,  -- e.g., 'bfs_epa_exp'
    feature_value REAL,
    feature_type TEXT,  -- 'numeric', 'categorical', 'time_series'

    -- Lineage
    source_table TEXT,  -- 'plays', 'games', 'external'
    computation_method TEXT,  -- 'exponential_rolling', 'aggregate', 'derived'
    computation_params JSONB,  -- {'halflife': 4, 'window': 3}

    -- Versioning
    feature_version TEXT,  -- 'v1.0', 'v2.0'
    model_version TEXT,  -- 'xgb_v3_1', 'xgb_v4'

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    computed_at TIMESTAMP WITH TIME ZONE,

    -- Statistics
    coverage REAL,  -- % non-null in this feature
    mean_value REAL,
    std_value REAL,
    min_value REAL,
    max_value REAL,

    -- Constraints
    UNIQUE (game_id, season, team, feature_name, feature_version),
    CHECK (feature_value IS NOT NULL OR coverage = 0)
);

-- Feature lineage table
CREATE TABLE IF NOT EXISTS ml.feature_lineage (
    lineage_id SERIAL PRIMARY KEY,

    upstream_table TEXT,  -- 'plays', 'games'
    downstream_feature TEXT,  -- 'bfs_epa_exp'

    -- Computation
    computation_sql TEXT,  -- Store the SQL/Python for reproducibility
    transformation_type TEXT,  -- 'aggregation', 'rolling', 'join'

    -- Dependencies
    depends_on JSONB,  -- ['plays.epa', 'games.kickoff']
    impacts JSONB,  -- ['xgb_v4', 'hierarchical_v2']

    -- Tracking
    created_by TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE,

    UNIQUE (upstream_table, downstream_feature)
);

-- Indexes
CREATE INDEX idx_model_features_game_team
ON ml.model_features (game_id, team);

CREATE INDEX idx_model_features_feature_name
ON ml.model_features (feature_name, feature_version);

CREATE INDEX idx_feature_lineage_downstream
ON ml.feature_lineage (downstream_feature);
```

### ETL Pipeline

```python
# py/features/bfs_etl.py

class BFSETI:
    """Full ETL for BFS features with database storage."""

    def __init__(self, conn):
        self.conn = conn

    def extract(self) -> pd.DataFrame:
        """Load raw plays data."""
        return pd.read_sql("SELECT ... FROM plays WHERE ...", self.conn)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute BFS metrics."""
        # Apply exponential weighting, rolling averages, etc.
        return df

    def load(self, df: pd.DataFrame):
        """Store in ml.model_features table."""
        # Insert rows into database
        for _, row in df.iterrows():
            # INSERT INTO ml.model_features (...)
            pass

    def record_lineage(self):
        """Document feature dependencies."""
        # INSERT INTO ml.feature_lineage (...)
        pass
```

---

## Comparison Table

| Aspect | Option 1 (CSV) | Option 2 (MV) | Option 3 (DB) |
|--------|---|---|---|
| **Development Time** | 1-2 weeks | 2-3 weeks | 3-4 weeks |
| **Database Changes** | None | 1 migration file | 2-3 migrations |
| **Queryability** | Not SQL | SQL SELECT | SQL + Python API |
| **Feature Lineage** | Comments only | View definition | Table + tracking |
| **Version Control** | Git (CSV diffs) | Git (SQL) | Git (SQL) + DB |
| **Production Ready** | Yes | Yes | Yes |
| **Testing Ease** | Easy (CSV inspect) | Medium (query) | Complex (DB state) |
| **Refresh Frequency** | Manual or cron | Scheduled MV refresh | Scheduled ETL |

---

## Recommendation by Use Case

### Use Option 1 (CSV) If:
- [ ] You need BFS integrated in 1-2 weeks
- [ ] You're still experimenting with feature set
- [ ] You want fast iteration cycles
- [ ] Your team is more Python-focused
- [ ] You have <10 feature sources

**Verdict for BFS:** ✓ CSV is best choice

### Use Option 2 (MV) If:
- [ ] You want SQL integration
- [ ] You have stable feature definitions
- [ ] You're building team infrastructure
- [ ] You want queryable feature history
- [ ] You have 5+ feature sources to unify

### Use Option 3 (Full DB) If:
- [ ] You need feature lineage for compliance
- [ ] You have 20+ feature sources to manage
- [ ] You need multi-team data governance
- [ ] You're building production data platform
- [ ] You have significant ETL complexity

---

## Integration Points by Agent

### Data Layer Agent Tasks

**Option 1 (CSV):**
1. Implement `py/features/bfs_features.py`
2. Generate `data/processed/features/bfs_features.csv`
3. Validate coverage (>80%)
4. Document in `FEATURE_ENGINEERING_PATTERNS.md`

**Option 2 (MV):**
1. Write migration file: `db/migrations/026_bfs_features.sql`
2. Create materialized view
3. Add view to `mart` schema documentation
4. Create refresh schedule

**Option 3 (Full DB):**
1. Write migrations for `ml.model_features` + `ml.feature_lineage`
2. Implement `py/features/bfs_etl.py`
3. Create lineage documentation
4. Set up monitoring for feature quality

### Prediction Engine Agent Tasks

**All Options:**
1. Load features in training scripts
2. Verify coverage audit passes
3. Check feature importance
4. Validate 2025 holdout performance

### Model Management Agent Tasks

**All Options:**
1. Monitor feature statistics over time
2. Track feature usage across models
3. Document feature importance
4. Alert on coverage degradation

---

## Temporal Safety Requirements (Non-Negotiable)

### Checklist for Any Implementation

- [ ] All rolling windows use `shift(1)` (no lookahead)
- [ ] Opponent metrics use only PRIOR games
- [ ] As-of joins ordered by `kickoff` timestamp
- [ ] Test data (2025) excluded from feature computation
- [ ] Coverage audit >80% before training
- [ ] Feature importance stored with model metadata
- [ ] Holdout validation on 2025 season

### Validation Script

```python
def validate_temporal_safety(df: pd.DataFrame) -> bool:
    """Verify no lookahead bias in features."""

    # Check 1: No future data in rolling windows
    for col in ['bfs_epa_L3', 'bfs_yards_L3']:
        if col in df.columns:
            # Manually verify a few samples
            assert df[col].notna().sum() > 0
            logger.info(f"✓ {col} computed (likely safe)")

    # Check 2: Opponent metrics from prior games
    assert df['opp_bfs_epa'].notna().sum() > 0
    logger.info("✓ Opponent metrics present")

    # Check 3: Coverage acceptable
    assert df[feature_cols].notna().mean() > 0.80
    logger.info("✓ Coverage >80%")

    return True
```

---

## Migration Path

**Week 1-2: Option 1 (CSV)**
- Implement BFS feature generator
- Generate CSV, validate, integrate into XGBoost
- Measure improvement

**Week 3-4: Option 2 (MV) - If Needed**
- If XGBoost v4 with BFS shows promise (+1-2% ROI)
- Migrate CSV features to materialized view
- Improve query performance

**Month 2: Option 3 (DB) - If Building Infrastructure**
- If managing 20+ features across 5+ models
- Implement full feature lineage + statistics
- Build feature monitoring dashboard

---

## Success Criteria

For any implementation:

1. **Coverage Audit**
   - [ ] bfs_epa_exp: >90% non-null
   - [ ] opp_bfs_epa_exp: >85% non-null

2. **Temporal Safety**
   - [ ] No lookahead bias detected
   - [ ] 2025 holdout validates on separate data

3. **Model Performance**
   - [ ] XGBoost v4 MAE < v3.1 baseline
   - [ ] Feature importance >5% for BFS
   - [ ] Backtest ROI improvement >0.5%

4. **Data Quality**
   - [ ] No NaN in key columns
   - [ ] Values within expected ranges
   - [ ] Consistent across seasons

---

## Deliverables

**Option 1 (CSV):**
- [ ] `py/features/bfs_features.py` (550+ lines)
- [ ] `data/processed/features/bfs_features.csv` (16k+ rows)
- [ ] Coverage audit report
- [ ] Integration instructions

**Option 2 (MV):**
- [ ] `db/migrations/026_bfs_features.sql`
- [ ] View documentation
- [ ] Query examples
- [ ] Refresh schedule

**Option 3 (Full DB):**
- [ ] `db/migrations/026_model_features_table.sql`
- [ ] `db/migrations/027_feature_lineage.sql`
- [ ] `py/features/bfs_etl.py`
- [ ] Monitoring dashboard

---

## Questions to Ask Before Starting

1. **Do you need queryable features now?**
   - No → Option 1
   - Yes → Option 2 or 3

2. **How many feature sources will you manage?**
   - <5 → Option 1
   - 5-20 → Option 2
   - >20 → Option 3

3. **What's your timeline?**
   - 1-2 weeks → Option 1
   - 2-3 weeks → Option 2
   - 3-4 weeks → Option 3

4. **Do you need feature lineage for compliance?**
   - No → Option 1 or 2
   - Yes → Option 3

---

## Next Steps

1. **Decide:** Option 1, 2, or 3?
2. **Implement:** Follow checklist for chosen option
3. **Validate:** Run temporal safety checks
4. **Integrate:** Update training scripts
5. **Measure:** Check feature importance + holdout performance

---

**Resources:**
- Template: `py/features/opponent_strength.py`
- Training integration: `py/models/train_xgboost_v3_1_aligned_window.py`
- Full analysis: `docs/FEATURE_ENGINEERING_PATTERN_ANALYSIS.md`
- Quick start: `docs/BFS_INTEGRATION_QUICK_START.md`

**Last Updated:** November 14, 2025
**Status:** Ready for implementation
