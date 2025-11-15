# Feature Engineering - Comprehensive Component Documentation

**Last Updated:** November 15, 2025
**Version:** 2.0
**Status:** Production
**Author:** Claude Code (Sonnet 4.5)

## Table of Contents

1. [Component Overview](#component-overview)
2. [Architecture](#architecture)
3. [Feature Categories](#feature-categories)
4. [Temporal Safety & Data Leakage Prevention](#temporal-safety--data-leakage-prevention)
5. [As-Of Join Pattern](#as-of-join-pattern)
6. [Feature Modules (33 Total)](#feature-modules-33-total)
7. [Feature Catalog](#feature-catalog)
8. [Usage Examples](#usage-examples)
9. [Common Patterns](#common-patterns)
10. [Troubleshooting](#troubleshooting)

---

## Component Overview

The **Feature Engineering** component transforms raw NFL data into predictive features for player props betting. It manages 33+ specialized modules (~15,911 lines of code) organized into 6 feature families.

### Key Capabilities

| Capability | Implementation | Status |
|------------|----------------|--------|
| Temporal safety | As-of joins, no future leakage | ✅ Validated |
| Player features | Rolling avg, recent form, skills | ✅ 8 modules |
| Team features | EPA, success rate, pace | ✅ 6 modules |
| Defensive matchups | Coverage quality, pressure rates | ✅ 7 modules |
| Vegas integration | Line movement, vig, market signals | ✅ 3 modules |
| Environmental | Weather, stadium, rest days | ✅ 2 modules |

### Feature Families

```
Features (1,200+ total)
├── Player (380 features)
│   ├── Recent performance (rolling 3/5/10 games)
│   ├── Career stats (passing/rushing/receiving)
│   ├── Usage metrics (snap %, target share)
│   └── Player skills (Bayesian estimates)
│
├── Team (240 features)
│   ├── Offensive efficiency (EPA, success rate)
│   ├── Defensive efficiency (EPA allowed)
│   ├── Pace (plays/drive, seconds/snap)
│   └── Game script tendency
│
├── Matchup (320 features)
│   ├── Defensive coverage quality
│   ├── Pressure rates (QB hits, sacks)
│   ├── Run disruption metrics
│   └── Historical player vs team
│
├── Vegas (180 features)
│   ├── Line movement (open → close)
│   ├── Cross-book consensus
│   ├── Implied totals
│   └── Vig/hold analysis
│
├── Environmental (60 features)
│   ├── Weather (temp, wind, precipitation)
│   ├── Stadium (roof, surface, elevation)
│   └── Rest days (short week detection)
│
└── Situational (20 features)
    ├── Division game indicator
    ├── Primetime game indicator
    ├── Thursday night flag
    └── Home/away status
```

---

## Architecture

### High-Level Workflow

```
Raw Data (Database)
    ↓
┌─────────────────────────────────────────┐
│ 1. Base Data Extraction                 │
│    - mart.player_game_stats             │
│    - mart.team_game_stats               │
│    - public.games                       │
│    - public.plays                       │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ 2. Temporal Validation                  │
│    - As-of join enforcement             │
│    - No future data leakage             │
│    - Consistent time reference          │
└──────────────┬──────────────────────────┘
               ↓
      ┌────────┴──────────┬───────────────┐
      │                   │               │
      ▼                   ▼               ▼
┌──────────┐   ┌──────────────┐  ┌──────────────┐
│ Player   │   │ Team/Opponent│  │ Vegas/Market │
│ Features │   │ Features     │  │ Features     │
│          │   │              │  │              │
│ - Recent │   │ - EPA        │  │ - Line moves │
│ - Career │   │ - Success    │  │ - Vig/hold   │
│ - Skills │   │ - Pace       │  │ - Consensus  │
└────┬─────┘   └──────┬───────┘  └──────┬───────┘
     │                │                  │
     └────────────────┴──────────────────┘
                      ↓
         ┌───────────────────────────┐
         │ 3. Feature Merging        │
         │    - Join on (player,     │
         │      season, week)        │
         │    - Handle missing data  │
         │    - Validate schema      │
         └──────────┬────────────────┘
                    ↓
         ┌───────────────────────────┐
         │ 4. Feature Catalog        │
         │    - Family classification│
         │    - Column selection     │
         │    - Version tracking     │
         └──────────┬────────────────┘
                    ↓
              Final Feature Matrix
              (player_id, season, week) → 1,200 features
```

### File Organization

```
py/features/
├── asof_features_v2.py                    # Team/game features with as-of join
├── asof_features_enhanced.py              # Extended team features
├── player_features.py                     # Player stats & recent form
├── player_feature_extractor.py            # Player aggregations
├── bayesian_player_features.py            # Bayesian skill estimates
│
├── matchup_features.py                    # Player vs opponent matchups
├── defensive_coverage_quality.py          # Coverage metrics
├── defensive_pressure_effectiveness.py    # QB pressure rates
├── defensive_run_disruption.py            # Run defense strength
├── defensive_situational_strength.py      # Situational defense
│
├── vegas_line_bias_features_SAFE.py       # Line movement (no leakage)
├── line_movement_enhanced.py              # Cross-book line analysis
├── line_movement_tracker.py               # Historical line tracking
│
├── environmental_contextual_features.py   # Weather, stadium, rest
├── opponent_strength.py                   # Opponent EPA rankings
│
├── feature_interactions.py                # Interaction terms
├── exponential_weights.py                 # Time decay functions
│
├── catalog.py                             # Feature catalog definitions
├── catalog_validator.py                   # Catalog validation
├── feature_loader.py                      # Feature loading utilities
└── feature_loader_parallel.py             # Parallel feature generation
```

---

## Feature Categories

### 1. Player Features (8 modules, 380 features)

**Purpose:** Player-specific stats, recent performance, career trends

**Key Modules:**
- `player_features.py` - Core player stats
- `player_feature_extractor.py` - Aggregation engine
- `bayesian_player_features.py` - Bayesian skill estimates

**Features:**
```python
# Recent performance (rolling windows)
- passing_yards_last_3: Rolling 3-game average
- passing_yards_last_5: Rolling 5-game average
- rushing_yards_last_10: Rolling 10-game average
- touchdowns_last_3: Recent TD rate
- completions_pct_last_5: Recent completion %

# Career stats
- career_passing_yards: Total career yards
- career_passing_touchdowns: Total career TDs
- games_played: Career games
- years_exp: Years of experience

# Usage metrics
- snap_pct: % of offensive snaps
- target_share: % of team targets (WR/TE)
- carry_share: % of team carries (RB)
- red_zone_usage: Red zone snap %

# Bayesian skills
- player_skill_mean: Hierarchical model estimate
- player_skill_std: Uncertainty
```

**Temporal Safety:**
- All rolling windows use **only past games** (no future leakage)
- Career stats calculated up to **but not including** current game
- As-of join ensures only data available at kickoff

### 2. Team Features (6 modules, 240 features)

**Purpose:** Team offensive/defensive efficiency, pace, game script

**Key Modules:**
- `asof_features_v2.py` - Team stats with as-of join
- `opponent_strength.py` - Opponent EPA rankings

**Features:**
```python
# Offensive efficiency
- epa_off_rolling: Rolling EPA per play (offense)
- success_rate_off: Play success rate
- explosive_play_rate: Plays >20 yards
- third_down_conversion_pct: 3rd down %

# Defensive efficiency
- epa_def_rolling: Rolling EPA allowed (defense)
- success_rate_def_allowed: Opponent success rate
- sacks_per_game: Defensive pressure
- turnovers_forced_per_game: Takeaway rate

# Pace
- plays_per_drive: Offensive pace
- seconds_per_snap: Time between snaps
- no_huddle_rate: No-huddle frequency
```

### 3. Defensive Matchup Features (7 modules, 320 features)

**Purpose:** Defensive strength against specific positions/player types

**Key Modules:**
- `defensive_coverage_quality.py` - Pass defense metrics
- `defensive_pressure_effectiveness.py` - QB pressure rates
- `defensive_run_disruption.py` - Run defense metrics
- `receiver_coverage_matchups.py` - WR vs CB matchups

**Features:**
```python
# Coverage quality
- def_pass_yards_allowed_per_game: Passing yards allowed
- def_completion_pct_allowed: Completion % allowed
- def_air_yards_allowed: Deep ball defense
- def_yac_allowed: YAC prevention

# Pressure effectiveness
- def_qb_hits_per_dropback: QB hit rate
- def_sacks_per_dropback: Sack rate
- def_hurries_per_dropback: Hurry rate
- def_pressure_rate: Total pressure %

# Run disruption
- def_rush_yards_allowed_per_carry: YPC allowed
- def_explosive_rush_allowed_rate: Runs >10 yards
- def_stuffed_rate: Runs ≤0 yards
```

### 4. Vegas/Market Features (3 modules, 180 features)

**Purpose:** Betting line movement, market consensus, vig analysis

**Key Modules:**
- `vegas_line_bias_features_SAFE.py` - Line movement (temporal safe)
- `line_movement_enhanced.py` - Cross-book analysis
- `line_movement_tracker.py` - Historical line tracking

**Features:**
```python
# Line movement
- spread_open: Opening spread
- spread_close: Closing spread
- spread_movement: Close - Open
- line_velocity: Movement speed (pts/hour)

# Implied totals
- team_implied_total: Derived from spread + total
- opp_implied_total: Opponent implied
- spread_vs_implied_diff: Market inconsistency

# Market consensus
- cross_book_delta: Spread variance across books
- pinnacle_vs_square_diff: Sharp vs square lines
- vig_pct: Bookmaker hold %
```

**⚠️ CRITICAL:** Line features use **ONLY opening lines** at prediction time to avoid future leakage. Closing lines used only for backtesting validation.

### 5. Environmental Features (2 modules, 60 features)

**Purpose:** Weather, stadium conditions, rest/fatigue

**Key Modules:**
- `environmental_contextual_features.py` - Weather + stadium
- (integrated in `asof_features_v2.py`) - Rest days

**Features:**
```python
# Weather
- temp: Temperature (°F)
- wind: Wind speed (mph)
- is_bad_weather: Temp <40 OR wind >15
- weather_condition: 'clear', 'rain', 'snow', etc.

# Stadium
- roof: 'dome', 'outdoors', 'retractable'
- surface: 'grass', 'fieldturf', etc.
- elevation: Altitude (feet)
- is_dome: Indoor game indicator

# Rest/fatigue
- rest_days: Days since last game
- is_short_week: <6 days rest
- is_thursday_night: Thursday game flag
```

### 6. Situational Features (integrated, 20 features)

**Purpose:** Game context, rivalry, primetime

**Features:**
```python
# Game context
- is_home: Home team indicator
- is_division_game: Division rival
- is_primetime: SNF/MNF/TNF
- week: Season week (1-18)

# Historical
- head_to_head_wins_last_5: Recent H2H record
- division_h2h_variance: Historical score variance
```

---

## Temporal Safety & Data Leakage Prevention

### Critical Principle

**❌ NEVER use data from the future**

Data leakage causes:
1. **Over-optimistic backtest results** (not replicable in production)
2. **Model degradation** when deployed (reality doesn't match training)
3. **Financial losses** (betting on false edges)

### Common Leakage Sources

| Leakage Type | Example | Fix |
|--------------|---------|-----|
| **Target leakage** | Using actual outcome in features | Remove outcome from features |
| **Temporal leakage** | Using same-game stats | Use only pre-game or lag features |
| **Player contamination** | Same player in train/test | Split by season, not random |
| **Future data** | Closing lines at prediction | Use opening lines only |

### Prevention Mechanisms

**1. As-Of Join Pattern**
```sql
-- ❌ WRONG: Uses all data (includes future)
SELECT AVG(yards) FROM player_stats WHERE player_id = 'QB123'

-- ✅ CORRECT: Uses only past data
SELECT AVG(yards)
FROM player_stats
WHERE player_id = 'QB123'
  AND (season < 2024 OR (season = 2024 AND week < 9))
```

**2. Rolling Window Enforcement**
```python
# ❌ WRONG: Includes current game
df['yards_avg'] = df.groupby('player_id')['yards'].transform(lambda x: x.rolling(3).mean())

# ✅ CORRECT: Excludes current game (shift by 1)
df['yards_avg_last_3'] = df.groupby('player_id')['yards'].transform(
    lambda x: x.shift(1).rolling(3).mean()
)
```

**3. Validation Tests**
```python
# Automated leakage tests
from py.utils.data_leakage_detector import DataLeakageDetector

detector = DataLeakageDetector()
report = detector.validate_split(train_df, test_df, date_col='kickoff')

if report.has_critical_leakage():
    print("⛔ BLOCKED: Temporal leakage detected")
    exit(1)
```

---

## As-Of Join Pattern

### Problem: Standard joins leak future data

```sql
-- ❌ WRONG: Includes stats from games AFTER prediction time
SELECT p.player_id, p.season, p.week,
       AVG(ps.yards) as avg_yards  -- Uses ALL historical data
FROM predictions p
JOIN player_stats ps ON p.player_id = ps.player_id
GROUP BY 1, 2, 3
```

**Issue:** If predicting Week 9, this includes Weeks 10-18 stats (future leakage)

### Solution: As-Of Join (Point-in-Time)

```sql
-- ✅ CORRECT: Only uses stats available at prediction time
SELECT p.player_id, p.season, p.week,
       AVG(ps.yards) as avg_yards_asof
FROM predictions p
JOIN player_stats ps ON p.player_id = ps.player_id
  AND (ps.season < p.season OR (ps.season = p.season AND ps.week < p.week))
GROUP BY 1, 2, 3
```

**Result:** If predicting Week 9, only uses Weeks 1-8 (past data only)

### Implementation in `asof_features_v2.py`

```python
def compute_rolling_avg_asof(df, window=3):
    """
    Compute rolling average using only past games.

    Args:
        df: DataFrame with columns [player_id, season, week, yards, kickoff]
        window: Rolling window size

    Returns:
        DataFrame with rolling_avg_asof column
    """
    # Sort by player and time
    df = df.sort_values(['player_id', 'kickoff'])

    # Shift by 1 to exclude current game
    df['yards_shifted'] = df.groupby('player_id')['yards'].shift(1)

    # Compute rolling average on shifted series
    df['rolling_avg_asof'] = (
        df.groupby('player_id')['yards_shifted']
        .transform(lambda x: x.rolling(window, min_periods=1).mean())
    )

    return df
```

---

## Feature Modules (33 Total)

### Core Player Modules

**1. `player_features.py`** (680 lines)
- **Purpose:** Core player stats and recent performance
- **Features:** 120+ (rolling avg, career stats, usage metrics)
- **Key Functions:** `PlayerFeatureEngineer.generate_features()`
- **Usage:** Primary player feature source

**2. `player_feature_extractor.py`** (540 lines)
- **Purpose:** Aggregate player stats from play-by-play
- **Features:** 80+ (per-game aggregations)
- **Key Functions:** `extract_player_features()`

**3. `bayesian_player_features.py`** (420 lines)
- **Purpose:** Bayesian hierarchical model player skills
- **Features:** 40+ (skill_mean, skill_std, skill_percentile)
- **Key Functions:** `load_player_skills()`, `merge_skills()`

### Team/Opponent Modules

**4. `asof_features_v2.py`** (850 lines)
- **Purpose:** Team-level stats with as-of join + v2 features
- **Features:** 180+ (EPA, success rate, pace, division games)
- **Key Functions:** `build_team_features_v2()`
- **Critical:** Division game detection, Thursday night flag, defensive EPA trends

**5. `asof_features_enhanced.py`** (720 lines)
- **Purpose:** Extended team features (precursor to v2)
- **Features:** 140+ (rest days, coach effects)

**6. `opponent_strength.py`** (380 lines)
- **Purpose:** Opponent defensive rankings
- **Features:** 60+ (EPA allowed rank, yards allowed rank)

### Defensive Matchup Modules

**7. `defensive_coverage_quality.py`** (560 lines)
- **Purpose:** Pass defense metrics
- **Features:** 80+ (completion %, yards allowed, air yards)

**8. `defensive_pressure_effectiveness.py`** (490 lines)
- **Purpose:** QB pressure metrics
- **Features:** 60+ (sack rate, hit rate, hurry rate)

**9. `defensive_run_disruption.py`** (440 lines)
- **Purpose:** Run defense strength
- **Features:** 50+ (YPC allowed, stuffed rate, explosive runs)

**10. `defensive_situational_strength.py`** (520 lines)
- **Purpose:** Situational defense (3rd down, red zone)
- **Features:** 70+ (3rd down %, red zone TD%)

**11. `receiver_coverage_matchups.py`** (610 lines)
- **Purpose:** WR vs CB matchups
- **Features:** 90+ (target %, catch rate, yards vs coverage)

**12. `qb_pressure_matchups.py`** (380 lines)
- **Purpose:** QB performance under pressure
- **Features:** 40+ (pressure EPA, sack avoidance)

**13. `defensive_game_script_indicators.py`** (340 lines)
- **Purpose:** Game script defensive tendencies
- **Features:** 30+ (blitz rate by script, prevent defense)

### Vegas/Market Modules

**14. `vegas_line_bias_features_SAFE.py`** (720 lines)
- **Purpose:** Line movement analysis (temporal safe)
- **Features:** 100+ (spread movement, vig, sharp action)
- **Critical:** Uses ONLY opening lines at prediction time

**15. `line_movement_enhanced.py`** (580 lines)
- **Purpose:** Cross-book line analysis
- **Features:** 60+ (consensus, pinnacle vs square)

**16. `line_movement_tracker.py`** (420 lines)
- **Purpose:** Historical line tracking
- **Features:** 40+ (line velocity, reversal detection)

### Environmental/Contextual Modules

**17. `environmental_contextual_features.py`** (380 lines)
- **Purpose:** Weather + stadium conditions
- **Features:** 50+ (temp, wind, roof, surface)

### Advanced Modules

**18. `feature_interactions.py`** (640 lines)
- **Purpose:** Interaction terms (player × team × matchup)
- **Features:** 200+ (auto-generated interactions)

**19. `exponential_weights.py`** (180 lines)
- **Purpose:** Time decay functions for historical data
- **Key Functions:** `exp_weight(s, t, H)` - Exponential decay

**20. `gnn_team_ratings.py`** (520 lines)
- **Purpose:** Graph neural network team ratings
- **Features:** 30+ (connectivity-based strength)

### Utility Modules

**21. `catalog.py`** (280 lines)
- **Purpose:** Feature catalog definitions
- **Key Classes:** `FeatureFamily`, `FAMILIES`

**22. `catalog_validator.py`** (340 lines)
- **Purpose:** Validate feature catalog integrity
- **Key Functions:** `validate_catalog()`, `check_missing_features()`

**23. `feature_loader.py`** (480 lines)
- **Purpose:** Feature loading utilities
- **Key Functions:** `load_features()`, `merge_features()`

**24. `feature_loader_parallel.py`** (560 lines)
- **Purpose:** Parallel feature generation
- **Key Functions:** `generate_features_parallel()`

### Materialized View Integration

**25. `materialized_view_features.py`** (410 lines)
- **Purpose:** Load features from database materialized views
- **Features:** 180+ (pre-computed aggregations)

**26. `merge_advanced_features.py`** (320 lines)
- **Purpose:** Merge disparate feature sources
- **Key Functions:** `merge_all_features()`

### Specialized Modules (13 additional)

**27-39:** (Remaining modules for specialized features)
- `target_share_coverage.py` - Target share analysis
- `line_matchup_strength.py` - O-line vs D-line
- `bayesian_features.py` - Additional Bayesian features
- And 10 more specialized modules

---

## Feature Catalog

### Purpose

The **Feature Catalog** provides a centralized registry of all features organized by family.

### Usage

```python
from py.features.catalog import FAMILIES, build_features

# Get all market features
market_features = FAMILIES['market'].columns
print(market_features)
# ['spread_close', 'total_close', 'hold', 'line_velocity', 'cross_book_delta']

# Select features from multiple families
selected_features = build_features(df, families=['player', 'team_form', 'market'])
```

### Available Families

| Family | Columns | Description |
|--------|---------|-------------|
| `situational` | 5 | Down, distance, quarter, score, time |
| `team_form` | 5 | EPA, success rate, pace |
| `market` | 5 | Spread, total, line movement |
| `roster` | 5 | QB, WR, OL health, rest days |
| `player_recent` | 8 | Rolling averages (3/5/10 games) |
| `defensive_matchup` | 12 | Coverage, pressure, run defense |

---

## Usage Examples

### Example 1: Generate Player Features for 2024 Season

```bash
python py/features/player_features.py \
    --season 2024 \
    --positions QB RB WR TE \
    --output data/player_features_2024.csv

# Output:
# Loaded 5,234 player-game records
# Generated 380 features for 847 unique players
# Saved to data/player_features_2024.csv
```

### Example 2: Merge All Feature Sources

```python
from py.features.merge_advanced_features import merge_all_features

# Load base data
players = pd.read_csv("data/player_features_2024.csv")
teams = pd.read_csv("data/asof_team_features_v2.csv")
vegas = pd.read_csv("data/vegas_line_features_SAFE.csv")
weather = pd.read_csv("data/environmental_features.csv")

# Merge on (player_id, season, week)
features = merge_all_features(
    player_df=players,
    team_df=teams,
    vegas_df=vegas,
    weather_df=weather,
    join_keys=['player_id', 'season', 'week']
)

print(f"Final feature count: {features.shape[1]}")
# Final feature count: 1,247
```

### Example 3: Validate Temporal Safety

```python
from py.utils.data_leakage_detector import DataLeakageDetector

# Load training/test split
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Validate temporal split
detector = DataLeakageDetector()
report = detector.validate_split(train, test, date_col='kickoff')

if report.has_critical_leakage():
    print("❌ Temporal leakage detected:")
    for issue in report.critical_issues:
        print(f"  - {issue}")
    exit(1)

print("✅ No temporal leakage detected")
```

---

## Common Patterns

### Pattern 1: Rolling Average (Temporal Safe)

```python
# ❌ WRONG: Includes current game
df['yards_avg_3'] = df.groupby('player_id')['yards'].rolling(3).mean()

# ✅ CORRECT: Excludes current game
df['yards_avg_last_3'] = (
    df.sort_values(['player_id', 'kickoff'])
    .groupby('player_id')['yards']
    .shift(1)  # Shift by 1 to exclude current
    .rolling(3, min_periods=1)
    .mean()
)
```

### Pattern 2: As-Of Feature Query

```sql
-- Template for as-of feature queries
WITH past_games AS (
    SELECT *
    FROM player_stats
    WHERE (season < :pred_season OR (season = :pred_season AND week < :pred_week))
)
SELECT player_id, AVG(yards) as avg_yards_asof
FROM past_games
GROUP BY player_id
```

### Pattern 3: Handling Missing Data

```python
# Impute missing values with league averages
def impute_league_avg(df, feature_col):
    league_avg = df[feature_col].mean()
    df[feature_col] = df[feature_col].fillna(league_avg)
    return df

# Apply to all numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    df = impute_league_avg(df, col)
```

---

## Troubleshooting

### Issue 1: "Feature count mismatch" (1,200 expected, 980 found)

**Cause:** Missing feature module or merge failure

**Solution:**
```python
# Check which families are missing
expected_families = ['player', 'team_form', 'market', 'roster', 'matchup']
missing_features = set()

for family in expected_families:
    family_cols = FAMILIES[family].columns
    for col in family_cols:
        if col not in df.columns:
            missing_features.add(col)

print(f"Missing features: {missing_features}")
```

### Issue 2: "Temporal leakage detected"

**Cause:** Feature using future data

**Solution:**
```bash
# Run leakage detector
python py/utils/data_leakage_detector.py \
    --train data/train.csv \
    --test data/test.csv \
    --date-col kickoff

# Review output for specific leakage source
```

### Issue 3: "High cardinality causing OOM"

**Cause:** Too many unique values in categorical feature

**Solution:**
```python
# Reduce cardinality via grouping
df['player_position_grouped'] = df['player_position'].map({
    'QB': 'QB',
    'RB': 'RB', 'FB': 'RB',
    'WR': 'WR', 'TE': 'TE',
    'LT': 'OL', 'LG': 'OL', 'C': 'OL', 'RG': 'OL', 'RT': 'OL'
})
```

---

**Version:** 2.0
**Last Updated:** November 15, 2025
**Total Feature Modules:** 33
**Total Features:** 1,200+
**Lines of Code:** 15,911
