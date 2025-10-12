# NFL Analytics Feature Integration Plan
## Comprehensive Guide to Integrating Advanced Metrics

**Status**: Ready for Implementation
**Date**: 2025-10-10
**Estimated Timeline**: 2-3 days full implementation

---

## 📊 Current Database State

### Successfully Loaded ✅
| Dataset | Records | Coverage | Size | Key Metrics |
|---------|---------|----------|------|-------------|
| **Next Gen Passing** | 5,521 | 2016-2025 | 2.3 MB | CPOE, time to throw, aggressiveness |
| **Next Gen Rushing** | 5,613 | 2016-2025 | 2.0 MB | Efficiency, yards over expected, stacked box % |
| **Next Gen Receiving** | 13,816 | 2016-2025 | 5.7 MB | Separation, YAC above expectation |
| **PFR Defense** | 56,620 | 2018-2025 | TBD | Coverage, pressures, missed tackles |
| **Snap Counts** | 305,373 | 2012-2025 | TBD | Participation % (off/def/ST) |
| **Injuries** | Existing | 2009+ | 5.3 MB | Game status, practice participation |
| **Plays (EPA/WPA)** | Existing | 1999+ | Large | Play-level EPA, WPA, Success Rate, CPOE |

### Pending Fix 🔧
- **ESPN QBR**: Data loaded but all filtered out (investigate "Season Total" filter)
- **Depth Charts**: Column name mismatch (needs `game_type` not `season_type`)

---

## 🎯 Integration Strategy

### Phase 1: Database Foundation (Day 1 Morning)
**Goal**: Ensure all data is properly loaded and indexed

#### Tasks:
1. **Fix remaining backfill issues**
   - [x] Depth charts column mapping
   - [ ] ESPN QBR "Season Total" filter (may need to keep weekly data differently)
   - [ ] Verify all foreign key relationships

2. **Add computed indexes**
   ```sql
   -- Team-level aggregation indexes
   CREATE INDEX idx_nextgen_passing_team_week ON nextgen_passing(season, week);
   CREATE INDEX idx_pfr_defense_team_week ON pfr_defense(team, season, week);
   CREATE INDEX idx_snap_counts_starter_pct ON snap_counts(team, season, week)
     WHERE offense_pct >= 50 OR defense_pct >= 50;
   ```

3. **Create materialized views for common aggregations**
   ```sql
   -- Example: Team offensive efficiency by week
   CREATE MATERIALIZED VIEW mart.team_offense_weekly AS
   SELECT
     season,
     week,
     team,
     AVG(qbr_total) as avg_qbr,
     AVG(completion_percentage_above_expectation) as avg_cpoe,
     AVG(rush_yards_over_expected_per_att) as avg_ryoe,
     AVG(avg_separation) as avg_receiver_separation
   FROM games g
   LEFT JOIN nextgen_passing np ON ...
   LEFT JOIN nextgen_rushing nr ON ...
   LEFT JOIN nextgen_receiving nre ON ...
   GROUP BY season, week, team;
   ```

---

### Phase 2: Feature Engineering Pipeline (Day 1 Afternoon - Day 2)
**Goal**: Integrate new metrics into `py/features/asof_features_enhanced.py`

#### Current Feature Structure
Your existing feature pipeline in `asof_features_enhanced.py` likely has sections for:
- Team stats (offense/defense)
- Recent performance windows (L3, L5, L10 games)
- Head-to-head history
- Situational factors (home/away, rest days, etc.)

#### New Feature Categories to Add

##### 1. **QB Performance Features**
*Sources: ESPN QBR, Next Gen Passing, Plays (EPA)*

```python
# Example feature additions
def generate_qb_features(game_id, team, asof_date):
    """
    Generate QB-related features for a team going into a game.
    Uses weighted rolling averages (more recent = higher weight).
    """
    features = {}

    # ESPN QBR (if we fix the data load)
    features['qb_qbr_l3'] = get_qbr_last_n_games(team, asof_date, n=3)
    features['qb_qbr_l5'] = get_qbr_last_n_games(team, asof_date, n=5)
    features['qb_qbr_season'] = get_qbr_season_avg(team, asof_date)

    # Next Gen Stats - Passing
    features['qb_cpoe_l5'] = get_ngs_metric(team, 'completion_percentage_above_expectation', n=5)
    features['qb_time_to_throw_l5'] = get_ngs_metric(team, 'avg_time_to_throw', n=5)
    features['qb_aggressiveness_l5'] = get_ngs_metric(team, 'aggressiveness', n=5)
    features['qb_air_yards_l5'] = get_ngs_metric(team, 'avg_intended_air_yards', n=5)

    # Play-level EPA (already in database)
    features['qb_epa_per_play_l3'] = get_epa_per_play(team, asof_date, play_type='pass', n=3)
    features['qb_success_rate_l5'] = get_success_rate(team, asof_date, play_type='pass', n=5)

    # QB stability (is starting QB available?)
    features['qb_is_starter'] = check_qb_starter_status(team, asof_date)  # from depth_charts
    features['qb_injury_status'] = get_injury_status(team, 'QB', asof_date)  # from injuries

    return features
```

##### 2. **Offensive Skill Position Features**
*Sources: Next Gen Rushing, Next Gen Receiving, Snap Counts*

```python
def generate_skill_position_features(team, asof_date):
    """
    RB and WR/TE advanced metrics.
    """
    features = {}

    # Rushing efficiency
    features['rb_yards_over_expected_l5'] = get_ngs_rushing_metric(
        team, 'rush_yards_over_expected_per_att', n=5
    )
    features['rb_efficiency_l5'] = get_ngs_rushing_metric(team, 'efficiency', n=5)
    features['rb_vs_stacked_box_pct'] = get_ngs_rushing_metric(
        team, 'percent_attempts_gte_eight_defenders', n=5
    )

    # Receiving playmaking
    features['wr_avg_separation_l5'] = get_ngs_receiving_metric(team, 'avg_separation', n=5)
    features['wr_yac_above_exp_l5'] = get_ngs_receiving_metric(
        team, 'avg_yac_above_expectation', n=5
    )
    features['wr_catch_pct_l5'] = get_ngs_receiving_metric(team, 'catch_percentage', n=5)

    # Snap count context (who's actually playing?)
    features['rb1_snap_pct_l3'] = get_rb1_snap_share(team, asof_date, n=3)
    features['wr_top3_snap_pct'] = get_top_wr_snap_share(team, asof_date, n=3)

    return features
```

##### 3. **Defensive Features**
*Sources: PFR Defense, Plays (EPA allowed)*

```python
def generate_defensive_features(team, asof_date):
    """
    Defensive performance metrics.
    """
    features = {}

    # Coverage metrics
    features['def_pass_rating_allowed_l5'] = get_pfr_def_metric(
        team, 'def_passer_rating_allowed', n=5
    )
    features['def_completion_pct_allowed_l5'] = get_pfr_def_metric(
        team, 'def_completion_pct', n=5
    )
    features['def_yards_per_target_l5'] = get_pfr_def_metric(
        team, 'def_yards_allowed_per_tgt', n=5
    )

    # Pass rush
    features['def_pressure_rate_l5'] = get_pressure_rate(team, asof_date, n=5)
    features['def_sack_rate_l5'] = get_sack_rate(team, asof_date, n=5)
    features['def_qb_hits_l5'] = get_pfr_def_metric(team, 'def_times_hitqb', n=5)

    # Tackling efficiency
    features['def_missed_tackle_pct_l5'] = get_pfr_def_metric(
        team, 'def_missed_tackle_pct', n=5
    )

    # EPA allowed (from plays table)
    features['def_epa_per_play_allowed_l3'] = get_epa_allowed_per_play(team, asof_date, n=3)
    features['def_success_rate_allowed_l5'] = get_success_rate_allowed(team, asof_date, n=5)

    return features
```

##### 4. **Matchup-Specific Features**
*Cross-reference offensive and defensive metrics*

```python
def generate_matchup_features(home_team, away_team, asof_date):
    """
    Matchup-specific advantages/disadvantages.
    """
    features = {}

    # QB vs Pass Defense
    features['matchup_home_cpoe_vs_away_pass_def'] = (
        get_qb_cpoe(home_team, n=5) - get_pass_def_rating_allowed(away_team, n=5)
    )

    # RB efficiency vs Run Defense
    features['matchup_away_ryoe_vs_home_run_def'] = (
        get_rb_yards_over_expected(away_team, n=5) -
        get_rush_yards_allowed_over_exp(home_team, n=5)
    )

    # WR separation vs DB coverage
    features['matchup_home_wr_sep_vs_away_db_coverage'] = (
        get_wr_separation(home_team, n=5) -
        get_yards_per_target_allowed(away_team, n=5)
    )

    # Pressure rate vs time to throw
    features['matchup_away_pressure_vs_home_ttt'] = (
        get_pressure_rate(away_team, n=5) /
        get_time_to_throw(home_team, n=5)
    )

    return features
```

##### 5. **Injury/Availability Context**
*Sources: Injuries, Depth Charts, Snap Counts*

```python
def generate_availability_features(team, asof_date):
    """
    Context on who's playing and at what capacity.
    """
    features = {}

    # Key position availability
    features['qb_available'] = is_position_healthy(team, 'QB', asof_date)
    features['rb1_available'] = is_position_healthy(team, 'RB', asof_date, starter_only=True)
    features['wr_top3_healthy_count'] = count_healthy_starters(team, 'WR', asof_date, top_n=3)

    # Defensive availability
    features['def_starters_out_count'] = count_injured_starters(team, 'Defense', asof_date)

    # Depth concerns (backups getting significant snaps)
    features['offense_backup_snap_pct_l3'] = get_backup_snap_pct(team, 'offense', n=3)

    return features
```

---

### Phase 3: Feature Pipeline Integration (Day 2)

#### Step-by-step Integration

1. **Add helper functions to `py/features/asof_features_enhanced.py`**

```python
import psycopg2
import pandas as pd
from datetime import datetime, timedelta

class NFLFeatureGenerator:
    def __init__(self, db_conn):
        self.conn = db_conn

    def get_ngs_metric(self, team, metric_name, asof_date, n_games=5):
        """
        Get Next Gen Stats metric averaged over last N games.
        Handles passing, rushing, or receiving based on metric name.
        """
        # Determine which NGS table based on metric
        if metric_name in ['completion_percentage_above_expectation', 'avg_time_to_throw']:
            table = 'nextgen_passing'
            join_condition = 'ON p.passer_player_id = ngs.player_id'
        elif metric_name in ['rush_yards_over_expected_per_att', 'efficiency']:
            table = 'nextgen_rushing'
            join_condition = 'ON p.rusher_player_id = ngs.player_id'
        elif metric_name in ['avg_separation', 'avg_yac_above_expectation']:
            table = 'nextgen_receiving'
            join_condition = 'ON p.receiver_player_id = ngs.player_id'

        query = f"""
        WITH team_games AS (
            SELECT game_id, season, week
            FROM games
            WHERE (home_team = %s OR away_team = %s)
              AND gameday < %s
            ORDER BY gameday DESC
            LIMIT %s
        )
        SELECT AVG(ngs.{metric_name}) as avg_metric
        FROM team_games tg
        JOIN plays p ON tg.game_id = p.game_id
        JOIN {table} ngs {join_condition}
          AND ngs.season = tg.season
          AND ngs.week = tg.week
        WHERE p.posteam = %s
        """

        df = pd.read_sql(query, self.conn, params=[team, team, asof_date, n_games, team])
        return df['avg_metric'].iloc[0] if not df.empty else None

    def get_pressure_rate(self, team, asof_date, n_games=5):
        """
        Calculate defensive pressure rate from PFR defense stats.
        """
        query = """
        WITH team_games AS (
            SELECT game_id
            FROM games
            WHERE (home_team = %s OR away_team = %s)
              AND gameday < %s
            ORDER BY gameday DESC
            LIMIT %s
        )
        SELECT
            SUM(def_pressures) * 1.0 / NULLIF(SUM(def_times_blitzed +
                CASE WHEN def_targets > 0 THEN def_targets ELSE 0 END), 0) as pressure_rate
        FROM team_games tg
        JOIN pfr_defense pd ON tg.game_id = pd.game_id
        WHERE pd.team = %s
        """

        df = pd.read_sql(query, self.conn, params=[team, team, asof_date, n_games, team])
        return df['pressure_rate'].iloc[0] if not df.empty else 0.0

    def is_position_healthy(self, team, position, asof_date):
        """
        Check if key position players are healthy based on injury report.
        """
        query = """
        SELECT
            CASE
                WHEN COUNT(*) FILTER (WHERE report_status IN ('Out', 'Doubtful')) > 0
                THEN FALSE
                ELSE TRUE
            END as is_healthy
        FROM injuries
        WHERE team = %s
          AND position = %s
          AND week = (
              SELECT MAX(week) FROM injuries
              WHERE season = EXTRACT(YEAR FROM %s::date)
                AND date_modified <= %s
          )
        """

        df = pd.read_sql(query, self.conn, params=[team, position, asof_date, asof_date])
        return df['is_healthy'].iloc[0] if not df.empty else True  # Default to healthy if no data

    def generate_all_features(self, game_id, asof_date):
        """
        Master function to generate all features for a game.
        """
        # Get teams
        teams_query = "SELECT home_team, away_team FROM games WHERE game_id = %s"
        teams = pd.read_sql(teams_query, self.conn, params=[game_id])
        home_team = teams['home_team'].iloc[0]
        away_team = teams['away_team'].iloc[0]

        features = {}

        # Generate for both teams
        for prefix, team in [('home', home_team), ('away', away_team)]:
            features.update(self._add_prefix(
                self.generate_qb_features(team, asof_date), prefix
            ))
            features.update(self._add_prefix(
                self.generate_skill_position_features(team, asof_date), prefix
            ))
            features.update(self._add_prefix(
                self.generate_defensive_features(team, asof_date), prefix
            ))
            features.update(self._add_prefix(
                self.generate_availability_features(team, asof_date), prefix
            ))

        # Add matchup features
        features.update(self.generate_matchup_features(home_team, away_team, asof_date))

        return features

    def _add_prefix(self, feature_dict, prefix):
        """Helper to add prefix to feature names."""
        return {f"{prefix}_{k}": v for k, v in feature_dict.items()}
```

2. **Update main feature generation script**

```python
# In your main training/prediction script
from features.asof_features_enhanced import NFLFeatureGenerator
import psycopg2

# Connect to database
conn = psycopg2.connect(
    host='localhost',
    port=5544,
    dbname='devdb01',
    user='dro',
    password='sicillionbillions'
)

# Initialize feature generator
fg = NFLFeatureGenerator(conn)

# Generate features for all games
feature_data = []
for game_id in training_game_ids:
    asof_date = get_game_date(game_id) - timedelta(days=1)  # Day before game
    features = fg.generate_all_features(game_id, asof_date)
    features['game_id'] = game_id
    feature_data.append(features)

# Convert to DataFrame
X = pd.DataFrame(feature_data)
```

---

### Phase 4: Model Retraining & Validation (Day 3)

#### Tasks:
1. **Feature importance analysis**
   - Run feature importance on XGBoost model
   - Identify which new features are most predictive
   - Remove redundant/low-value features

2. **Cross-validation with new features**
   - Retrain model with expanded feature set
   - Compare performance metrics:
     - AUC-ROC
     - Log loss
     - Calibration plots
     - Sharpe ratio (if using for betting)

3. **Feature ablation study**
   - Test impact of feature groups:
     - Baseline (existing features only)
     - + Next Gen Stats
     - + PFR Defense
     - + Snap Counts/Injuries
     - Full feature set

4. **Backtest on 2024 season**
   - Run predictions on all 2024 games
   - Compare against actual results
   - Calculate ROI if applied to betting strategy

---

## 🚀 Quick Start Commands

### Verify Data Load
```bash
# Check record counts
PGPASSWORD=sicillionbillions psql -h localhost -p 5544 -U dro devdb01 -c "
SELECT
  'nextgen_passing' as table_name, COUNT(*) as records FROM nextgen_passing
UNION ALL
SELECT 'nextgen_rushing', COUNT(*) FROM nextgen_rushing
UNION ALL
SELECT 'nextgen_receiving', COUNT(*) FROM nextgen_receiving
UNION ALL
SELECT 'pfr_defense', COUNT(*) FROM pfr_defense
UNION ALL
SELECT 'snap_counts', COUNT(*) FROM snap_counts
UNION ALL
SELECT 'depth_charts', COUNT(*) FROM depth_charts;
"
```

### Generate Sample Features
```python
# Test feature generation on single game
python -c "
from features.asof_features_enhanced import NFLFeatureGenerator
import psycopg2

conn = psycopg2.connect(
    host='localhost', port=5544, dbname='devdb01',
    user='dro', password='sicillionbillions'
)
fg = NFLFeatureGenerator(conn)

# Test on a 2024 game
features = fg.generate_all_features('2024_01_BUF_JAX', '2024-09-23')
print(f'Generated {len(features)} features')
print('Sample features:', list(features.keys())[:10])
"
```

---

## 📈 Expected Impact

### Performance Improvements (Estimated)
- **AUC-ROC**: +2-4% (from baseline ~0.59 to ~0.61-0.63)
- **Log Loss**: -5-8% improvement
- **Betting ROI**: +3-5% if current ROI is positive
- **Calibration**: Better probability estimates with more granular data

### Most Valuable Features (Predicted)
1. **CPOE (Completion % Above Expectation)** - Strong QB performance signal
2. **Pressure Rate** - Direct impact on QB effectiveness
3. **Yards Over Expected (RB)** - Offensive efficiency independent of blocking
4. **WR Separation** - Playmaking ability indicator
5. **Missed Tackle %** - Defensive execution quality

---

## ⚠️ Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Overfitting with 100+ new features | High | Regularization, feature selection, cross-validation |
| Data recency bias (NGS only 2016+) | Medium | Use 2016+ for training, pre-2016 as validation only |
| Missing data for injured/backup players | Medium | Impute with team averages, add "missing data" flags |
| Computation time increase | Low | Cache aggregated features, use materialized views |

---

## 📝 Next Steps

1. [ ] Fix remaining data load issues (ESPN QBR, Depth Charts)
2. [ ] Create materialized views for common aggregations
3. [ ] Implement helper functions in `asof_features_enhanced.py`
4. [ ] Generate features for 2024 season as test case
5. [ ] Retrain XGBoost model with new feature set
6. [ ] Compare performance metrics vs baseline
7. [ ] Deploy updated model to prediction pipeline

**Estimated Completion**: 2-3 days (depends on data quality issues)
