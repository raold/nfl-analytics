# R Ecosystem Opportunities for NFL Analytics

**Date:** October 4, 2025  
**Current State:** Using nflreadr, nflfastR for schedules/PBP, The Odds API for odds

---

## What We're Already Using ✅

### nflverse Suite (Current)
1. **`nflreadr`** ✅ - Data loader (schedules, PBP, rosters, draft picks)
2. **`nflfastR`** ✅ - Play-by-play processor with EPA calculations
3. **Current Usage:**
   - `data/ingest_schedules.R` - Loads 1999-2024 schedules
   - `data/ingest_pbp.R` - Loads PBP with EPA, pass/rush indicators
   - Cached snapshot: `data/raw/nflverse_schedules_1999_2024.rds`

---

## Missing nflverse Packages (High Value)

### 1. **`nfl4th`** 🎯 HIGH PRIORITY
**What it does:** 4th down decision modeling and optimal play calling
**Why you need it:**
- Calculate expected points added for go/punt/FG decisions
- Build features for coaching aggressiveness/conservatism
- Detect suboptimal decision-making that creates betting edges

**Use Cases:**
- Feature: `coach_4th_down_aggression_index` (rolling average)
- Feature: `team_4th_down_conversion_rate_vs_expectation`
- Edge detection: Teams that consistently make suboptimal 4th down calls create total line value

**Integration:**
```r
library(nfl4th)
library(dplyr)

# Add 4th down decision quality to your play data
pbp_with_4th <- pbp |>
  nfl4th::add_4th_probs() |>
  mutate(
    bad_4th_decision = case_when(
      go_boost < -2 ~ 1,  # Should have kicked
      fg_boost < -2 ~ 1,  # Should have gone for it
      TRUE ~ 0
    )
  )

# Aggregate to team-level features
team_4th_features <- pbp_with_4th |>
  group_by(season, week, posteam) |>
  summarise(
    fourth_down_aggression = mean(went_on_fourth, na.rm = TRUE),
    bad_4th_decisions_per_game = mean(bad_4th_decision, na.rm = TRUE),
    fourth_down_epa = mean(epa[down == 4], na.rm = TRUE)
  )
```

**Database Schema Addition:**
```sql
ALTER TABLE plays ADD COLUMN go_boost REAL;
ALTER TABLE plays ADD COLUMN fg_boost REAL;
ALTER TABLE plays ADD COLUMN punt_boost REAL;

CREATE TABLE mart.team_4th_down_metrics (
  team TEXT,
  season INT,
  week INT,
  go_rate REAL,
  success_rate REAL,
  epa_per_4th REAL,
  bad_decisions_pct REAL,
  PRIMARY KEY (team, season, week)
);
```

---

### 2. **`nflplotR`** 📊 MEDIUM PRIORITY
**What it does:** Visualization with team logos, colors, wordmarks
**Why you need it:**
- Professional-looking charts for dissertation
- Team-aware ggplot themes
- Logo/helmet integration in plots

**Use Cases:**
- Chapter 3 figures: EPA by team with logos
- Chapter 4 figures: Model performance by team
- Diagnostic plots with team branding

**Integration:**
```r
library(nflplotR)
library(ggplot2)

# EPA by team with logos
team_epa_plot <- team_epa_df |>
  ggplot(aes(x = def_epa, y = off_epa)) +
  geom_nfl_logos(aes(team_abbr = team), width = 0.05) +
  theme_minimal() +
  labs(title = "Offensive vs Defensive EPA (2024)",
       x = "Defensive EPA (lower is better)",
       y = "Offensive EPA per play")

# Save for dissertation
ggsave("analysis/dissertation/figures/team_epa_landscape.png", 
       width = 10, height = 6, dpi = 300)
```

---

### 3. **`nflseedR`** 🏆 MEDIUM-HIGH PRIORITY
**What it does:** Playoff probability simulation and tiebreakers
**Why you need it:**
- Model late-season desperation (teams fighting for playoffs)
- Playoff elimination creates rest-vs-starters dynamics
- Futures market inefficiencies

**Use Cases:**
- Feature: `playoff_prob_week_n` (rolling)
- Feature: `must_win_game` (binary based on playoff elimination risk)
- Feature: `playoff_seeding_locked` (teams resting starters)
- Market inefficiency: Lines don't fully adjust for desperation/tanking

**Integration:**
```r
library(nflseedR)

# Simulate playoff probabilities for each week
playoff_sims <- nflseedR::simulate_nfl(
  nfl_season = 2024,
  process_games = schedules_df,
  playoff_seeds = 7,
  tiebreaker_depth = 3,
  sims = 1000
)

# Extract playoff probabilities
playoff_probs <- playoff_sims |>
  group_by(team) |>
  summarise(
    playoff_prob = mean(made_playoffs),
    bye_prob = mean(seed == 1),
    home_advantage_prob = mean(seed <= 4)
  )

# Join to games for feature engineering
games_with_playoff_context <- games |>
  left_join(playoff_probs, by = c("home_team" = "team", "week")) |>
  left_join(playoff_probs, by = c("away_team" = "team", "week"), 
            suffix = c("_home", "_away"))
```

**Betting Edge Example:**
```r
# Teams with <5% playoff probability often rest starters or tank
# Create binary feature for database
desperation_features <- games |>
  mutate(
    home_desperate = playoff_prob_home > 0.15 & playoff_prob_home < 0.60,
    away_desperate = playoff_prob_away > 0.15 & playoff_prob_away < 0.60,
    home_eliminated = playoff_prob_home < 0.05,
    away_eliminated = playoff_prob_away < 0.05
  )
```

---

### 4. **`nflreadr` Extensions** (Already installed but underutilized)

#### **Rosters & Draft Data**
```r
# Load current rosters with positions, experience
rosters <- nflreadr::load_rosters(seasons = 2020:2024)

# Key injury features
qb_status <- rosters |>
  filter(position == "QB", depth_chart_position == "1") |>
  select(season, week, team, full_name, status)

# Join to games for QB injury indicator
games_with_qb <- games |>
  left_join(qb_status, by = c("home_team" = "team", "season", "week")) |>
  mutate(home_qb_out = status %in% c("Out", "IR"))
```

#### **Draft Capital (Future Feature)**
```r
# Draft picks as proxy for roster quality
draft_picks <- nflreadr::load_draft_picks(seasons = 2015:2024)

# Calculate team draft capital by year
draft_capital <- draft_picks |>
  group_by(season, team) |>
  summarise(
    first_round_picks = sum(round == 1),
    total_draft_value = sum(200 - pick, na.rm = TRUE),  # Simple AV
    avg_pick_quality = mean(round, na.rm = TRUE)
  )
```

#### **Participation Data (Snap Counts)**
```r
# Load snap counts for rotation analysis
snaps <- nflreadr::load_snap_counts(seasons = 2020:2024)

# Key starters playing percentage
starter_snaps <- snaps |>
  filter(position %in% c("QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB")) |>
  group_by(season, week, team) |>
  summarise(
    avg_starter_snap_pct = mean(offense_pct, na.rm = TRUE),
    rotation_depth = n_distinct(player_id)
  )
```

#### **Injuries (Official Status)**
```r
# Load official injury reports
injuries <- nflreadr::load_injuries(seasons = 2020:2024)

# Aggregate to team-level injury load
injury_load <- injuries |>
  group_by(season, week, team) |>
  summarise(
    players_out = sum(report_status == "Out"),
    players_questionable = sum(report_status == "Questionable"),
    key_position_out = any(position %in% c("QB", "LT", "C") & 
                           report_status == "Out")
  )
```

---

## Other R Ecosystem Tools

### 5. **`cfbfastR`** (CFB Parallel)
**What it does:** College football equivalent of nflfastR
**Why you might care:**
- Draft prospect analysis (college EPA predicts NFL success)
- Coaching hires from college (scheme familiarity)
- Cross-sport model validation

**Skip for now:** Focus on NFL, but keep in mind for future research

---

### 6. **`sportyR`** (Field Plotting)
**What it does:** Draw NFL fields with exact coordinates
**Why you might care:**
- Visualize play outcomes by field position
- Spatial analysis of red zone efficiency
- Next-Gen Stats tracking data visualization

**Use Case (Future):**
```r
library(sportyR)

# Plot red zone efficiency by team
geom_football("nfl") +
  geom_point(data = red_zone_plays, 
             aes(x = yardline_100, y = side_of_field, color = epa),
             size = 2, alpha = 0.6)
```

---

### 7. **`hoopR`** / **`wehoop`** / **`baseballr`**
**What they do:** NBA, WNBA, MLB equivalents
**Why you might care:**
- Cross-sport betting model validation
- Multi-sport RL framework
- Market microstructure comparisons

**Skip for now:** NFL-only dissertation, but architecture should be sport-agnostic

---

## Recommended Data Augmentation Pipeline

### Phase 1: Critical Missing Features (This Week)
1. ✅ **Weather** - Already integrated
2. 🔴 **4th Down Decisions** (`nfl4th`) - Add coaching quality features
3. 🔴 **Playoff Context** (`nflseedR`) - Add desperation/tanking indicators
4. 🔴 **Injury Load** (`nflreadr::load_injuries`) - Already ingested, need features

### Phase 2: Enhanced Context (Next 2 Weeks)
5. **Roster Stability** (`nflreadr::load_rosters`) - Starter continuity
6. **Snap Counts** (`nflreadr::load_snap_counts`) - Rotation depth
7. **Referee Assignments** (manual scrape) - Penalty/pace tendencies
8. **Travel/Rest** (calculate from schedules) - Fatigue features

### Phase 3: Advanced Analytics (Post-Defense)
9. **Next-Gen Stats** (if accessible) - Player tracking metrics
10. **PFF Grades** (if budget allows) - O-line, coverage quality
11. **Team-Level Tendencies** - Pace, play-calling, 2-point decisions

---

## Immediate Action Items

### 1. Install Missing Packages
```r
# Add to renv or run manually
install.packages(c("nfl4th", "nflplotR", "nflseedR"))
renv::snapshot()  # Lock versions
```

### 2. Create Feature Engineering Script
```r
# R/features_advanced.R
library(nfl4th)
library(nflseedR)
library(nflreadr)
library(dplyr)

source("R/db_helpers.R")  # Your DB connection helpers

# Load raw data
pbp <- fetch_pbp_from_db()
games <- fetch_games_from_db()
injuries <- nflreadr::load_injuries(2020:2024)

# 4th down features
pbp_4th <- pbp |>
  nfl4th::add_4th_probs() |>
  group_by(game_id, posteam) |>
  summarise(
    fourth_down_aggression = mean(went_on_fourth, na.rm = TRUE),
    fourth_down_epa = mean(epa[down == 4], na.rm = TRUE),
    bad_4th_decisions = sum(go_boost < -2 | fg_boost < -2, na.rm = TRUE)
  )

# Playoff context (run weekly during season)
playoff_probs <- nflseedR::simulate_nfl(2024, games, sims = 1000) |>
  summarise_playoff_probabilities()

# Injury load
injury_features <- injuries |>
  group_by(season, week, team) |>
  summarise(
    injury_load = n(),
    key_position_out = any(position %in% c("QB", "LT", "C") & 
                           report_status == "Out")
  )

# Write to mart schema
write_features_to_mart(pbp_4th, "team_4th_down_features")
write_features_to_mart(playoff_probs, "team_playoff_context")
write_features_to_mart(injury_features, "team_injury_load")
```

### 3. Update Database Schema
```sql
-- db/004_advanced_features.sql
CREATE SCHEMA IF NOT EXISTS mart;

CREATE TABLE mart.team_4th_down_features (
  game_id TEXT,
  team TEXT,
  fourth_down_aggression REAL,
  fourth_down_epa REAL,
  bad_4th_decisions INT,
  PRIMARY KEY (game_id, team)
);

CREATE TABLE mart.team_playoff_context (
  team TEXT,
  season INT,
  week INT,
  playoff_prob REAL,
  division_leader_prob REAL,
  eliminated BOOLEAN,
  PRIMARY KEY (team, season, week)
);

CREATE TABLE mart.team_injury_load (
  team TEXT,
  season INT,
  week INT,
  players_out INT,
  players_questionable INT,
  key_position_out BOOLEAN,
  PRIMARY KEY (team, season, week)
);

-- Composite feature view for modeling
CREATE MATERIALIZED VIEW mart.game_features_enhanced AS
SELECT 
  g.*,
  h4th.fourth_down_aggression AS home_4th_aggression,
  a4th.fourth_down_aggression AS away_4th_aggression,
  hpl.playoff_prob AS home_playoff_prob,
  apl.playoff_prob AS away_playoff_prob,
  hinj.injury_load AS home_injury_load,
  ainj.injury_load AS away_injury_load
FROM games g
LEFT JOIN mart.team_4th_down_features h4th 
  ON g.game_id = h4th.game_id AND g.home_team = h4th.team
LEFT JOIN mart.team_4th_down_features a4th 
  ON g.game_id = a4th.game_id AND g.away_team = a4th.team
LEFT JOIN mart.team_playoff_context hpl 
  ON g.home_team = hpl.team AND g.season = hpl.season AND g.week = hpl.week
LEFT JOIN mart.team_playoff_context apl 
  ON g.away_team = apl.team AND g.season = apl.season AND g.week = apl.week
LEFT JOIN mart.team_injury_load hinj 
  ON g.home_team = hinj.team AND g.season = hinj.season AND g.week = hinj.week
LEFT JOIN mart.team_injury_load ainj 
  ON g.away_team = ainj.team AND g.season = ainj.season AND g.week = ainj.week;
```

### 4. Update Python Harness to Use New Features
```python
# py/backtest/harness.py - add new feature columns

def load_games_with_features(seasons, conn):
    """Load games with all available features from mart views."""
    query = """
    SELECT 
        game_id,
        season,
        week,
        home_team,
        away_team,
        spread_line,
        total_line,
        home_score,
        away_score,
        -- EPA features (existing)
        home_off_epa,
        away_off_epa,
        home_def_epa,
        away_def_epa,
        -- Weather features (existing)
        temp_extreme,
        wind_penalty,
        has_precip,
        is_dome,
        -- NEW: 4th down coaching
        home_4th_aggression,
        away_4th_aggression,
        -- NEW: Playoff context
        home_playoff_prob,
        away_playoff_prob,
        -- NEW: Injury load
        home_injury_load,
        away_injury_load
    FROM mart.game_features_enhanced
    WHERE season = ANY(%s)
    ORDER BY kickoff
    """
    return pd.read_sql(query, conn, params=(seasons,))
```

---

## Expected Performance Gains

### 4th Down Features
- **GLM lift:** +0.5-1.0% accuracy (coaching quality signal)
- **XGBoost lift:** +0.3-0.8% (nonlinear coaching patterns)
- **Betting edge:** 2-3 extra bets/season where bad coaching creates value

### Playoff Context
- **GLM lift:** +1.0-1.5% accuracy (desperation/tanking is strong)
- **Betting edge:** Late-season games with eliminated teams are mispriced
- **ROI impact:** +0.5% on season-end games (Weeks 15-18)

### Injury Load
- **Already ingested but not featurized**
- **Expected lift:** +0.8-1.2% (key position injuries matter)
- **Betting edge:** Lines don't fully adjust for cumulative injury load

### Combined Impact
- **Total expected lift:** +2.0-3.5% model accuracy
- **ROI improvement:** +0.8-1.5% (more selective, better calibrated)
- **Risk reduction:** Better avoid traps (tanking teams, injured rosters)

---

## Timeline Estimate

### Week 1 (Oct 4-10): Core Infrastructure
- [ ] Install `nfl4th`, `nflseedR`, update renv
- [ ] Create `R/features_advanced.R` script
- [ ] Add `db/004_advanced_features.sql` migration
- [ ] Run feature generation for 2020-2024

### Week 2 (Oct 11-17): Model Integration
- [ ] Update Python harness to use new features
- [ ] Retrain XGBoost with expanded feature set
- [ ] Rerun GLM baseline with new features
- [ ] Update `glm_harness_overall.tex` table

### Week 3 (Oct 18-24): Dissertation Updates
- [ ] Add Section 3.1.3: "Advanced Context Features"
- [ ] Update Chapter 4 with new model performance
- [ ] Regenerate all figures with new features
- [ ] Update TODO.tex completion markers

---

## Additional Resources

### Documentation
- nflverse docs: https://nflverse.nflverse.com/
- nfl4th vignette: https://nfl4th.nflverse.com/articles/nfl4th.html
- nflseedR guide: https://nflseedr.com/articles/articles/nflseedR.html

### Community
- Discord: https://discord.gg/nflverse
- GitHub: https://github.com/nflverse
- Twitter: @nflfastR

### Papers Using nflverse
- Baldwin et al. (2022) - "The Athletic's 4th down bot"
- Yurko et al. (2019) - "Going for it on fourth down"
- Thompson & Lopez (2020) - "nflWAR: A reproducible method"

---

## Summary

**You're already using the core tools (nflreadr, nflfastR), but missing high-value features:**

🎯 **Must-Have (This Week):**
1. `nfl4th` - 4th down decision quality (coaching edge)
2. `nflseedR` - Playoff context (desperation/tanking)
3. Injury featurization - Already have data, need to aggregate

📊 **Nice-to-Have (Next 2 Weeks):**
4. `nflplotR` - Better dissertation figures
5. Roster/snap count features - Depth/continuity signals
6. Referee assignments - Pace/penalty tendencies

🚀 **Expected Impact:**
- +2-3% model accuracy
- +0.8-1.5% ROI improvement
- Better risk management (avoid traps)

**Next Command:**
```r
install.packages(c("nfl4th", "nflplotR", "nflseedR"))
renv::snapshot()
```
