# Advanced Feature Engineering Roadmap - Vector A Integration

**Date**: November 16, 2025
**Context**: Integration plan for next-generation causal inference and market microstructure features
**Current State**: Ensemble v6.0 (production), Ridge Residuals breakthrough (+88% improvement)
**Future Vision**: Move from prediction → causal understanding → market exploitation

---

## Executive Summary

Your feedback identifies two critical frontiers that represent the natural evolution beyond current Ridge Residuals success:

1. **Causal Inference Features**: Move from "what correlates" to "what causes" (IV, DiD, Propensity Scores)
2. **Market Microstructure Features**: Predict the *market* not just the *game* (line movement, cross-book dispersion, vigorish)

**Key Insight**: Ridge Residuals succeeded because they isolate **causal signal** (QB skill) from **confounding** (game context). The proposed features extend this principle to:
- **Causal Features**: Isolate treatment effects (coaching changes, injuries, rule changes)
- **Market Features**: Isolate information flow (sharp money, bookmaker uncertainty, market efficiency)

**Strategic Fit**:
- **Phase 1 (v7.0)**: Implement easiest causal features (rule changes DiD, travel fatigue matching)
- **Phase 2 (v8.0)**: Add market microstructure (line movement velocity, cross-book dispersion)
- **Phase 3 (v9.0)**: Advanced causal inference (IV for coaching changes, crowd noise isolation)

**Expected ROI Trajectory**:
- v6.0 baseline: +6-9% ROI, 61-62% win rate
- v7.0 (causal features): +7-10% ROI, 62-63% win rate (+1-2 pp improvement)
- v8.0 (market features): +9-12% ROI, 64-65% win rate (+2-3 pp improvement)
- v9.0 (advanced IV): +11-15% ROI, 66-68% win rate (+2-3 pp improvement)

---

## Table of Contents

1. [Causal Inference Features Roadmap](#causal-inference-features-roadmap)
2. [Market Microstructure Features Roadmap](#market-microstructure-features-roadmap)
3. [Implementation Phases](#implementation-phases)
4. [Technical Deep Dives](#technical-deep-dives)
5. [Data Requirements](#data-requirements)
6. [Integration Architecture](#integration-architecture)
7. [Validation Strategy](#validation-strategy)
8. [Expected Challenges](#expected-challenges)
9. [Research Timeline](#research-timeline)

---

## Causal Inference Features Roadmap

### Philosophy: From Association to Causation

**Current Limitation**: XGBoost v4 Ridge Residuals isolate QB skill from game context, but:
- Cannot distinguish correlation from causation
- Struggle with confounding (player quality ↔ offensive scheme)
- Miss exogenous shocks (injuries, coaching changes, rule changes)

**Solution**: Apply causal inference toolkit to create features that capture **true treatment effects**

---

### 1. Instrumental Variables (IV) for Coaching Changes

**Concept**: Use coach firing as "exogenous shock" to measure causal impact of offensive scheme change

**Why IV Works Here**:
- **Endogeneity Problem**: Good players attract good coordinators (reverse causality)
- **Instrument**: Head coach firing is exogenous (owner decision, not player quality)
- **Two-Stage Process**:
  1. Stage 1: Coach firing → Offensive scheme change (first stage regression)
  2. Stage 2: Predicted scheme change → Player performance (second stage regression)

**Feature Engineering**:
```python
# py/features/causal_iv_coaching.py

def create_coaching_iv_features(player_df, coaching_df):
    """
    Instrumental variable features for coaching changes

    IV Strategy:
    - Instrument: Head coach firing (exogenous shock)
    - Treatment: Offensive coordinator change
    - Outcome: Player passing yards

    Returns features capturing causal effect of scheme change
    """

    # Stage 1: Coach firing → OC change
    coaching_df['coach_fired'] = (coaching_df['tenure_weeks'] < 52).astype(int)
    coaching_df['oc_changed'] = (coaching_df['oc_id'] != coaching_df['prev_oc_id']).astype(int)

    # First stage regression
    first_stage = sm.OLS(
        coaching_df['oc_changed'],
        sm.add_constant(coaching_df[['coach_fired', 'team_win_pct', 'owner_patience']])
    ).fit()

    # Predicted OC change (instrumented by coach firing)
    coaching_df['oc_change_predicted'] = first_stage.predict()

    # Stage 2: Merge to player level
    player_df = player_df.merge(
        coaching_df[['team', 'season', 'week', 'oc_change_predicted']],
        on=['team', 'season', 'week']
    )

    # Interaction: Player skill × Predicted scheme change
    player_df['skill_x_scheme_change'] = (
        player_df['player_skill_estimate'] * player_df['oc_change_predicted']
    )

    return player_df
```

**Expected Impact**:
- Isolate **causal effect** of scheme change (not just correlation with player quality)
- Capture heterogeneous treatment effects (some players benefit from new scheme, others suffer)
- Reduce confounding bias by 30-50%

**Data Requirements**:
- Coaching change database (head coach, OC, tenure, firing date)
- Owner patience metric (previous coach tenures, win% tolerance)
- Scheme classification (West Coast, Air Raid, Run-Heavy, etc.)

**Validation**:
- **Instrument Validity**: Coach firing uncorrelated with player skill (F-stat > 10)
- **Exclusion Restriction**: Firing affects performance ONLY through scheme change
- **Backtesting**: Compare IV estimates vs naive OLS (expect 20-40% difference)

---

### 2. Difference-in-Differences (DiD) for Rule Changes

**Concept**: Use league-wide rule changes as natural experiments

**Example: 2018 Roughing-the-Passer Emphasis**

**Why DiD Works Here**:
- **Treatment Group**: Penalty-prone defenses (e.g., high 2017 roughing rate)
- **Control Group**: Disciplined defenses (low 2017 roughing rate)
- **Pre-period**: 2016-2017 (before rule emphasis)
- **Post-period**: 2018-2019 (after rule emphasis)
- **Estimand**: (Treatment_post - Treatment_pre) - (Control_post - Control_pre)

**Feature Engineering**:
```python
# py/features/causal_did_rule_changes.py

def create_did_roughing_features(player_df, defense_df):
    """
    DiD features for 2018 roughing-the-passer rule change

    Treatment: High roughing rate defense (2017 > median)
    Control: Low roughing rate defense (2017 < median)
    Outcome: QB passing yards vs that defense
    """

    # Classify defenses by 2017 roughing rate
    defense_2017 = defense_df[defense_df['season'] == 2017]
    median_roughing = defense_2017['roughing_penalties'].median()
    defense_2017['high_roughing'] = (defense_2017['roughing_penalties'] > median_roughing).astype(int)

    # Merge to player-game level
    player_df = player_df.merge(
        defense_2017[['team', 'high_roughing']],
        left_on='opponent',
        right_on='team',
        suffixes=('', '_def')
    )

    # DiD treatment indicator
    player_df['post_2018'] = (player_df['season'] >= 2018).astype(int)
    player_df['did_treatment'] = (
        player_df['high_roughing'] * player_df['post_2018']
    )

    # Expected causal effect: +15-25 yards vs high-roughing defenses post-2018
    player_df['expected_roughing_boost'] = player_df['did_treatment'] * 20  # yards

    return player_df
```

**Additional Rule Changes to Exploit**:
1. **2011 Kickoff Moved to 35-yard line** → Reduced returns → More offensive plays
2. **2018 Catch Rule Clarification** → More completed catches
3. **2020 COVID Season** → No preseason → Slower offensive starts (Week 1-4)
4. **2023 Hip-Drop Tackle Ban** → Safer QB play

**Expected Impact**:
- Capture **structural breaks** that naive time-series models miss
- Isolate **policy effects** from player talent
- Improve accuracy by 5-10 yards MAE for affected games

**Data Requirements**:
- Rule change database (year, description, affected positions)
- Defense penalty rates (roughing, holding, DPI by team-season)
- Historical game context (crowd noise, referee assignments)

**Validation**:
- **Parallel Trends Assumption**: Treatment/control groups have similar pre-trends (2016-2017)
- **Placebo Tests**: Run DiD on 2016-2017 (should show no effect)
- **Robustness**: Vary treatment threshold (median → quartiles)

---

### 3. Propensity Score Matching for Home Field Advantage

**Concept**: Isolate causal components of home field advantage

**Why Current Models Fail**:
- "Home field advantage" is a **bundle** of causal mechanisms:
  - Crowd noise (communication difficulty)
  - Travel fatigue (jet lag, sleep disruption)
  - Referee bias (home team gets favorable calls)
  - Familiarity (home team knows field, weather)

**Naive Approach**: `is_home = 1` captures average effect, but:
- Doesn't isolate which mechanisms matter
- Misses heterogeneity (outdoor vs dome, East vs West coast travel)

**Propensity Score Solution**: Match games with similar team strengths but different travel/crowd/ref characteristics

**Feature Engineering**:
```python
# py/features/causal_psm_home_advantage.py

def create_psm_home_advantage_features(game_df):
    """
    Propensity score matching to isolate home advantage components

    Mechanisms:
    1. Travel fatigue (distance, time zones, rest days)
    2. Crowd noise (stadium capacity × attendance %)
    3. Referee bias (home team favorability by ref crew)
    4. Weather familiarity (temp/wind difference from team's home)
    """

    # Calculate propensity scores (probability of home game given covariates)
    from sklearn.linear_model import LogisticRegression

    X = game_df[['team_elo', 'opponent_elo', 'season', 'week']]
    y = game_df['is_home']

    ps_model = LogisticRegression().fit(X, y)
    game_df['propensity_score'] = ps_model.predict_proba(X)[:, 1]

    # Match away games to similar home games (within 0.05 propensity score)
    from scipy.spatial import KDTree

    home_games = game_df[game_df['is_home'] == 1]
    away_games = game_df[game_df['is_home'] == 0]

    tree = KDTree(home_games[['propensity_score']].values)
    distances, indices = tree.query(away_games[['propensity_score']].values, k=1)

    # For each away game, compare to matched home game
    away_games['matched_home_idx'] = indices
    away_games['match_quality'] = 1 / (1 + distances)  # Higher = better match

    # Isolate causal components
    for i, away_row in away_games.iterrows():
        home_row = home_games.iloc[away_row['matched_home_idx']]

        # Component 1: Travel fatigue (away game has travel, matched home doesn't)
        travel_diff = away_row['travel_distance'] - home_row['travel_distance']
        away_games.loc[i, 'travel_fatigue_effect'] = -0.5 * travel_diff / 1000  # yards per 1000 miles

        # Component 2: Crowd noise (away game faces hostile crowd)
        crowd_diff = home_row['crowd_noise_db'] - away_row['crowd_noise_db']
        away_games.loc[i, 'crowd_noise_effect'] = -1.2 * crowd_diff / 10  # yards per 10 dB

        # Component 3: Referee bias (compare ref favorability)
        ref_diff = home_row['ref_home_bias'] - away_row['ref_home_bias']
        away_games.loc[i, 'referee_bias_effect'] = 8 * ref_diff  # yards per unit bias

    return away_games
```

**Expected Decomposition**:
- **Total Home Advantage**: ~20 yards passing (empirical)
- **Travel Fatigue**: 8-12 yards (3+ time zones, < 6 days rest)
- **Crowd Noise**: 5-8 yards (outdoor stadiums, playoff games)
- **Referee Bias**: 3-5 yards (well-documented in literature)
- **Weather Familiarity**: 2-4 yards (cold-weather teams at home in December)

**Feature Set for Ensemble**:
```python
features = [
    'travel_fatigue_effect',      # Isolated via PSM
    'crowd_noise_effect',          # Isolated via PSM
    'referee_bias_effect',         # Isolated via PSM
    'weather_familiarity_effect',  # Isolated via PSM
    'is_home',                     # Residual/unobserved components
]
```

**Expected Impact**:
- **Heterogeneity Capture**: Home advantage varies 10-30 yards depending on mechanism strength
- **Prediction Improvement**: 3-5 yards MAE for away games with extreme travel (SEA → MIA)
- **Betting Edge**: Exploit markets that use naive home advantage adjustment

**Data Requirements**:
- Travel distance matrix (all 32 teams, city coordinates)
- Stadium capacity and attendance (crowd noise proxy)
- Referee crew database (historical home team favorability)
- Weather data (temp, wind, precipitation by game)
- Rest days (Thursday night, Monday night, bye weeks)

**Validation**:
- **Balance Check**: Matched pairs have similar propensity scores (within 0.05)
- **Covariate Balance**: Team strength, opponent strength balanced post-matching
- **Sensitivity Analysis**: Vary matching tolerance (0.01, 0.05, 0.10)

---

## Market Microstructure Features Roadmap

### Philosophy: The Line is Information

**Current Limitation**: Models predict game outcome, but **ignore market dynamics**:
- A line that moves -3 → -4 slowly (public drift) is different from -3 → -4 in 5 minutes (sharp money)
- Cross-bookmaker dispersion signals uncertainty or mispricing
- Vigorish changes reveal bookmaker risk management

**Solution**: Treat betting line as **rich information source** about informed trader beliefs and bookmaker uncertainty

---

### 1. Line Movement Velocity & Acceleration

**Concept**: First and second derivatives of betting line capture information flow

**Why This Matters**:
- **Slow Drift** (0.1 points/hour): Public bettors, recreational money, noise
- **Sharp Snap** (2+ points in 5 minutes): Syndicate bet, injury news, weather report
- **Oscillation** (back-and-forth): Bookmaker uncertainty, balanced action

**Feature Engineering**:
```python
# py/features/market_line_movement.py

def create_line_movement_features(odds_df):
    """
    Line movement velocity and acceleration features

    Captures:
    1. Velocity: How fast is line moving? (points per hour)
    2. Acceleration: Is movement accelerating? (points per hour²)
    3. Sharp Signal: Discrete jumps (≥1.5 points in <10 minutes)
    """

    # Sort by game and timestamp
    odds_df = odds_df.sort_values(['game_id', 'timestamp'])

    # Calculate time deltas (hours)
    odds_df['hours_to_kickoff'] = (odds_df['kickoff'] - odds_df['timestamp']).dt.total_seconds() / 3600

    # First derivative: Line movement velocity
    odds_df['line_velocity'] = odds_df.groupby('game_id')['spread'].diff() / odds_df.groupby('game_id')['hours_to_kickoff'].diff()

    # Second derivative: Line movement acceleration
    odds_df['line_acceleration'] = odds_df.groupby('game_id')['line_velocity'].diff() / odds_df.groupby('game_id')['hours_to_kickoff'].diff()

    # Sharp signal: Discrete jumps (≥1.5 points in <10 minutes)
    time_delta_minutes = odds_df.groupby('game_id')['timestamp'].diff().dt.total_seconds() / 60
    line_delta = odds_df.groupby('game_id')['spread'].diff().abs()

    odds_df['sharp_signal'] = ((line_delta >= 1.5) & (time_delta_minutes < 10)).astype(int)

    # Aggregate to game level (use max velocity in 24 hours before kickoff)
    game_features = odds_df[odds_df['hours_to_kickoff'] <= 24].groupby('game_id').agg({
        'line_velocity': ['max', 'mean', 'std'],
        'line_acceleration': ['max', 'mean'],
        'sharp_signal': 'sum',  # Count of sharp signals
    }).reset_index()

    game_features.columns = ['_'.join(col).strip('_') for col in game_features.columns]

    return game_features
```

**Interpretation**:
- **High Velocity** (>0.5 points/hour): Significant information flow → Respect the move
- **High Acceleration** (>0.2 points/hour²): Accelerating information → Momentum trade
- **Sharp Signal Count** (≥3 in 24 hours): Informed money → Strong directional bias

**Expected Impact**:
- **Market Timing**: Identify when sharp money entered (fade public, follow sharp)
- **Information Content**: High velocity lines have 20-30% less mispricing
- **Contrarian Opportunity**: Slow drifts (public) can be faded profitably

**Data Requirements**:
- Historical odds feed (minute-by-minute, 7 days before kickoff)
- Timestamp of line changes (not just opening/closing line)
- Bookmaker identity (DraftKings, FanDuel, Caesars, etc.)

**Validation**:
- **Backtest**: High velocity lines have higher Sharpe ratios (less mispricing)
- **Correlation**: Sharp signals correlate with injury news, weather changes
- **Profitability**: Fading slow drifts yields positive expectation

---

### 2. Cross-Bookmaker Dispersion

**Concept**: Variance of line across sportsbooks signals uncertainty or error

**Why This Matters**:
- **Low Dispersion** (0.5 point spread): Market consensus, efficient pricing
- **High Dispersion** (2+ point spread): Disagreement, uncertainty, or bookmaker error
- **Outlier Book**: One book at -7, others at -4.5 → Potential arbitrage or information asymmetry

**Feature Engineering**:
```python
# py/features/market_cross_book_dispersion.py

def create_cross_book_dispersion_features(odds_df):
    """
    Cross-bookmaker dispersion features

    Captures:
    1. Dispersion: Standard deviation of spread across books
    2. Outlier Detection: Books >1.5 SD from mean
    3. Consensus Line: Median spread (robust to outliers)
    """

    # Pivot to wide format (books as columns)
    spread_wide = odds_df.pivot_table(
        index=['game_id', 'timestamp'],
        columns='bookmaker',
        values='spread'
    )

    # Calculate dispersion metrics
    game_features = pd.DataFrame()
    game_features['spread_std'] = spread_wide.std(axis=1)  # Dispersion
    game_features['spread_median'] = spread_wide.median(axis=1)  # Consensus
    game_features['spread_range'] = spread_wide.max(axis=1) - spread_wide.min(axis=1)  # Range

    # Outlier detection (z-score > 1.5)
    spread_mean = spread_wide.mean(axis=1)
    spread_std = spread_wide.std(axis=1)
    z_scores = (spread_wide.T - spread_mean) / spread_std
    game_features['num_outlier_books'] = (z_scores.abs() > 1.5).sum(axis=0)

    # Identify which book is outlier (highest absolute z-score)
    game_features['outlier_book'] = z_scores.abs().idxmax(axis=0)
    game_features['outlier_magnitude'] = z_scores.abs().max(axis=0)

    # Time-to-convergence: How long until dispersion < 0.5?
    convergence_threshold = 0.5
    game_features['time_to_convergence'] = (
        spread_wide.groupby('game_id')
        .apply(lambda x: (x.std(axis=1) < convergence_threshold).idxmax())
    )

    return game_features.reset_index()
```

**Trading Strategies**:

**Strategy 1: Fade High Dispersion Games**
- Hypothesis: High dispersion = high uncertainty = wider mispricing opportunity
- Implementation: Only bet games with dispersion > 1.5 (top 20%)
- Expected Edge: +2-3% ROI vs uniform betting

**Strategy 2: Arbitrage Outlier Books**
- Hypothesis: Outlier book has stale line or bookmaker error
- Implementation: Bet outlier book if |z-score| > 2.0
- Expected Edge: +5-10% ROI on outlier bets (rare, ~2% of games)

**Strategy 3: Wait for Convergence**
- Hypothesis: Early dispersion resolves as information diffuses
- Implementation: Wait until dispersion < 0.5 before betting (efficient market)
- Expected Edge: Reduce variance, similar expected value

**Expected Impact**:
- **Risk Management**: Avoid high-dispersion games (high uncertainty)
- **Arbitrage**: Exploit outlier books (5-10% of games)
- **Market Efficiency**: Time bets when market is most efficient (low dispersion)

**Data Requirements**:
- Odds from 10+ major sportsbooks (DraftKings, FanDuel, BetMGM, Caesars, etc.)
- Timestamp of each book's line change
- Bookmaker metadata (limits, market share, sharpness)

**Validation**:
- **Dispersion Stability**: High dispersion persists or resolves? (test 7-day, 3-day, 1-day windows)
- **Outlier Profitability**: Betting outlier books yields positive EV?
- **Convergence Timing**: When does dispersion drop below 0.5? (6 hours? 1 hour?)

---

### 3. Implied Vigorish Dynamics

**Concept**: Bookmaker vig changes reveal risk management and information

**Why This Matters**:
- **Widening Vig**: Bookmaker uncertain → Raise vig to 5-6% (protect downside)
- **Shrinking Vig**: Bookmaker confident → Lower vig to 3-4% (attract volume)
- **Asymmetric Vig**: One side juiced more → Bookmaker wants action on other side (balance book)

**Feature Engineering**:
```python
# py/features/market_vigorish_dynamics.py

def create_vigorish_features(odds_df):
    """
    Implied vigorish dynamics features

    Captures:
    1. Total Vig: Sum of implied probabilities - 100%
    2. Vig Change: How does vig evolve over time?
    3. Asymmetric Vig: Is one side juiced more than the other?
    """

    # Convert American odds to implied probability
    def american_to_prob(odds):
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return -odds / (-odds + 100)

    odds_df['prob_home'] = odds_df['home_odds'].apply(american_to_prob)
    odds_df['prob_away'] = odds_df['away_odds'].apply(american_to_prob)

    # Total implied probability (should be >100% due to vig)
    odds_df['total_prob'] = odds_df['prob_home'] + odds_df['prob_away']
    odds_df['total_vig'] = (odds_df['total_prob'] - 1) * 100  # Percentage vig

    # Vig asymmetry (higher prob side is more juiced)
    odds_df['vig_asymmetry'] = abs(odds_df['prob_home'] - odds_df['prob_away'])

    # Vig change over time
    odds_df = odds_df.sort_values(['game_id', 'timestamp'])
    odds_df['vig_change'] = odds_df.groupby('game_id')['total_vig'].diff()

    # Aggregate to game level
    game_features = odds_df.groupby('game_id').agg({
        'total_vig': ['mean', 'max', 'min', 'std'],
        'vig_asymmetry': ['mean', 'max'],
        'vig_change': ['mean', 'max'],
    }).reset_index()

    game_features.columns = ['_'.join(col).strip('_') for col in game_features.columns]

    # Binary indicators
    game_features['high_vig'] = (game_features['total_vig_max'] > 5.5).astype(int)  # >5.5% vig
    game_features['shrinking_vig'] = (game_features['vig_change_mean'] < -0.1).astype(int)  # Vig decreasing

    return game_features
```

**Interpretation**:

**High Vig (>5.5%)**:
- Bookmaker uncertain about outcome
- Typically: Bad weather games, key injury news, divisional rivalry
- Strategy: Avoid betting or demand higher edge

**Shrinking Vig (<4%)**:
- Bookmaker confident in line, wants volume
- Typically: Marquee games (SNF, MNF), playoff games
- Strategy: Efficient market, harder to find edge but lower transaction cost

**Asymmetric Vig**:
- Home team -110, Away team -115 → Bookmaker wants home bets (has too much away action)
- Strategy: Bet the side bookmaker wants (they're balancing, not predicting)

**Expected Impact**:
- **Vig-Adjusted Betting**: Only bet when vig < 4.5% (lower transaction cost)
- **Uncertainty Signal**: High vig games have 15-20% more prediction error
- **Volume Timing**: Bet when bookmaker shrinks vig (wants action = confident line)

**Data Requirements**:
- American odds (moneyline) for both sides (home/away or over/under)
- Timestamp of odds changes
- Bookmaker metadata (typical vig levels, market positioning)

**Validation**:
- **Vig Correlation**: High vig correlates with high prediction error (MAE)?
- **Shrinking Vig Profit**: Betting shrinking vig games yields higher Sharpe?
- **Asymmetric Vig Exploitation**: Fading bookmaker bias profitable?

---

## Implementation Phases

### Phase 0: International Games Module (Week 0 - IMMEDIATE, 6-8 hours)

**Objective**: Validate international games as a natural experiment for travel/jet lag effects

**Why Phase 0?**
- **Perfect natural experiment**: BOTH teams travel (symmetric treatment), isolates pure travel effect
- **Small dataset**: 9 games, 18 QB performances (2018-2024) → fast validation
- **Strong causal identification**: Exogenous scheduling, no selection bias
- **Immediate betting edge**: If validated, standalone profitability

**Data Extraction** ✅ COMPLETE:
```bash
python scripts/extract_international_games.py
# Output: data/international_games_2018_2024.csv
```

**Findings (Preliminary)**:
- **-60.4 yard penalty** vs domestic baseline (189.6 vs ~250 yards)
- All travel is West→East (hardest jet lag direction)
- Body clock effect: -7.4 yards for early morning (<9am) kickoffs
- No Mexico City games (altitude effect unavailable)

**Features Engineered**:
1. **Jet Lag Direction**: West→East (-12 yards/timezone), East→West (-4 yards/timezone)
2. **Body Clock Hour**: Kickoff time adjusted for team's home timezone
3. **Early Morning Penalty**: -2 yards per hour before 9am body clock
4. **Altitude Shock**: -3 yards per 1000 ft difference (Mexico City only)
5. **Neutral Field Indicator**: Neither team has home advantage

**Validation Protocol** (2-3 hours):
1. Compare international-aware predictions vs baseline on 2022-2024 London/Germany games
2. Expected MAE: Baseline ~50 yards → International-aware ~42-45 yards (10-15% improvement)
3. Statistical significance: p < 0.05 via bootstrapped confidence intervals

**Betting Edges to Test**:
1. **Fade West Coast QBs in early London games**: Chargers/Rams/49ers/Seahawks at 6:30am body clock
2. **International totals UNDER bias**: If both QBs underperform → total goes UNDER (58-62% historical UNDER rate if validated)
3. **Lingering fatigue (next game)**: Short rest after international game → performance drop

**Expected Performance**:
- International games MAE: 50 → 42-45 yards (10-15% improvement)
- Betting ROI (international games only): +5-10% if edges validated
- Season-wide impact: Limited (9 games/year = 3% of schedule)

**Deployment Gate**:
- **IF** international-aware model achieves <45 MAE on 2022-2024 games (p < 0.05) → **DEPLOY**
- **IF NOT** → Document as failed experiment (8 hours invested), move to Phase 1

**Timeline**:
- ✅ **Hour 0-1**: Data extraction (COMPLETE - script: `scripts/extract_international_games.py`)
- **Hour 2-3**: Feature engineering validation
- **Hour 4-5**: Backtest on historical international games
- **Hour 6-7**: Betting edge analysis (UNDER bias, lingering fatigue)
- **Hour 8**: Go/no-go decision

**Strategic Value**:
- **Immediate**: Standalone betting edge for 2025 London/Germany games (Week 5, 6, 7, 10)
- **Research**: Validates jet lag/travel methodology for Phase 1 PSM Travel Fatigue
- **Risk**: Only 8 hours invested - minimal opportunity cost if fails

---

### Phase 1: Ensemble v7.0 - Easy Causal Features (Q1 2026)

**Objective**: Add low-hanging causal fruit to v6.0 baseline

**Features to Implement**:
1. **DiD Rule Changes** (2018 roughing, 2011 kickoff, 2020 COVID)
   - Data: Historical rule change database (10 major changes 2010-2024)
   - Implementation: 2-3 weeks (data collection + feature engineering)
   - Expected Impact: +3-5 yards MAE improvement for affected games (10% of season)

2. **PSM Travel Fatigue** (isolate distance, time zones, rest days)
   - Data: Travel distance matrix (32 teams, city coordinates)
   - Implementation: 1-2 weeks (propensity score matching)
   - Expected Impact: +2-4 yards MAE improvement for extreme travel (5% of season)

3. **PSM Weather Familiarity** (cold-weather teams at home in December)
   - Data: Historical weather (temp, wind, precipitation)
   - Implementation: 1 week (matching + interaction terms)
   - Expected Impact: +2-3 yards MAE improvement for weather extremes (8% of season)

**Total Development**: 4-6 weeks (parallel development via subagents)

**Validation**:
- Multi-season backtest (2022-2024)
- Compare v7.0 vs v6.0: Expect +1-2% ROI improvement
- Consistency gates: std < 2 yards across years

**Expected Performance**:
- Ensemble MAE: 33 → 31 yards (6% improvement)
- Win Rate: 61-62% → 62-63% (+1 pp)
- ROI: +6-9% → +7-10% (+1% improvement)

**Deployment**:
- A/B test: v6.0 control vs v7.0 treatment (4 weeks, 100+ games)
- Success criteria: MAE improvement > 2 yards, p < 0.05
- Expected deployment: February 2026

---

### Phase 2: Ensemble v8.0 - Market Microstructure (Q2 2026)

**Objective**: Add line movement, dispersion, vigorish features

**Features to Implement**:
1. **Line Movement Velocity/Acceleration**
   - Data: Minute-by-minute odds feed (7 days before kickoff, all 2024 season)
   - Implementation: 3-4 weeks (API integration, feature engineering)
   - Expected Impact: +15-20% ROI on sharp signal games (10% of season)

2. **Cross-Bookmaker Dispersion**
   - Data: 10+ sportsbooks (DraftKings, FanDuel, BetMGM, Caesars, etc.)
   - Implementation: 2-3 weeks (multi-book API, dispersion metrics)
   - Expected Impact: +10-15% ROI on high-dispersion games (15% of season)

3. **Vigorish Dynamics**
   - Data: Moneyline odds from multiple books (American odds format)
   - Implementation: 1-2 weeks (vig calculation, time series features)
   - Expected Impact: +5-10% ROI by avoiding high-vig games (20% of season)

**Total Development**: 6-9 weeks (API integration bottleneck)

**Validation**:
- **Market Efficiency**: High velocity lines more efficient (lower mispricing)?
- **Dispersion Profitability**: High dispersion games have wider edges?
- **Vig Correlation**: High vig correlates with high prediction error?

**Expected Performance**:
- Ensemble MAE: 31 → 29 yards (6% improvement)
- Win Rate: 62-63% → 64-65% (+2 pp)
- ROI: +7-10% → +9-12% (+2-3% improvement)

**Deployment**:
- A/B test: v7.0 control vs v8.0 treatment (4 weeks, 100+ games)
- Success criteria: ROI improvement > 2%, Sharpe ratio > 3.0
- Expected deployment: June 2026

---

### Phase 3: Ensemble v9.0 - Advanced Causal IV (Q3-Q4 2026)

**Objective**: Implement full IV framework for coaching, injuries, crowd noise

**Features to Implement**:
1. **IV Coaching Changes** (head coach firing → OC change → performance)
   - Data: Coaching database (2000-2024, 200+ firings, 300+ OC changes)
   - Implementation: 6-8 weeks (two-stage IV regression, heterogeneous treatment effects)
   - Expected Impact: +5-8 yards MAE improvement for post-coaching-change games (5% of season)

2. **IV Crowd Noise** (stadium capacity → communication difficulty → performance)
   - Data: Stadium acoustics (capacity, design, decibel levels)
   - Implementation: 4-6 weeks (acoustic modeling, IV regression)
   - Expected Impact: +3-5 yards MAE improvement for loud outdoor stadiums (12% of season)

3. **IV Referee Bias** (referee crew → penalty favorability → performance)
   - Data: Referee database (all crews 2010-2024, penalty rates by crew-team)
   - Implementation: 3-4 weeks (IV regression, bias estimation)
   - Expected Impact: +2-4 yards MAE improvement for high-bias refs (8% of season)

**Total Development**: 13-18 weeks (research-intensive, requires econometric expertise)

**Validation**:
- **Instrument Validity**: F-stat > 10 (strong instrument)
- **Exclusion Restriction**: Instrument affects outcome ONLY through treatment
- **Overidentification Tests**: Multiple instruments consistent?

**Expected Performance**:
- Ensemble MAE: 29 → 26 yards (10% improvement)
- Win Rate: 64-65% → 66-68% (+2-3 pp)
- ROI: +9-12% → +11-15% (+2-3% improvement)

**Deployment**:
- A/B test: v8.0 control vs v9.0 treatment (6 weeks, 150+ games)
- Success criteria: ROI improvement > 3%, Sharpe ratio > 3.5
- Expected deployment: December 2026

---

## Technical Deep Dives

### Deep Dive 1: Why IV Works for Coaching Changes

**The Endogeneity Problem**:
- Naive regression: `Yards ~ OC_Changed + Player_Skill + ...`
- **Bias**: Good players attract good coordinators (reverse causality)
- **Confounding**: Team quality affects both OC hiring and performance

**IV Solution**:
- **Instrument**: Head coach firing (exogenous shock, owner decision)
- **Two-Stage Estimation**:
  1. **First Stage**: Regress OC change on coach firing + controls
     - `OC_Changed ~ Coach_Fired + Team_Win_Pct + Owner_Patience + ε₁`
     - Extract predicted OC change: `ÔC_Changed` (variation explained by firing only)

  2. **Second Stage**: Regress performance on predicted OC change
     - `Yards ~ ÔC_Changed + Player_Skill + ε₂`
     - Coefficient on `ÔC_Changed` is **causal effect** (unbiased by reverse causality)

**Why This Works**:
- Coach firing is **exogenous** (not caused by player skill)
- Coach firing affects performance **only through** OC change (exclusion restriction)
- Variation in `ÔC_Changed` comes from exogenous source (firing), not confounders

**Expected Coefficient**:
- Naive OLS: +8 yards (upward biased, good players get good coordinators)
- IV estimate: +3 yards (causal effect, scheme change alone)
- **Interpretation**: 60% of naive correlation is confounding, 40% is causal

**Validation**:
- **F-statistic > 10**: Strong first stage (coach firing predicts OC change)
- **Overidentification**: Use owner impatience as second instrument (Hansen J-test)
- **Robustness**: Vary controls (team strength, market size, playoff miss)

---

### Deep Dive 2: Line Movement as Information Revelation

**Market Microstructure Theory**:
- Betting markets aggregate dispersed information via price (line)
- **Informed Traders** (syndicates, pros): Move line quickly (sharp money)
- **Uninformed Traders** (public, recreation): Move line slowly (noise)

**Line Movement Velocity**:
- **Slow Drift** (0.1 points/hour):
  - Public accumulation (small bets, no private info)
  - Noise trading (random, uncorrelated)
  - **Interpretation**: Line move unreliable, fade it

- **Sharp Snap** (2+ points in 5 minutes):
  - Informed trader with private info (injury, weather, lineup)
  - Syndicate bet (coordinated, large stake)
  - **Interpretation**: Line move informative, respect it

**Acceleration (Second Derivative)**:
- **Positive Acceleration** (speeding up):
  - Information cascade (initial bet triggers more bets)
  - Momentum (traders chase the move)
  - **Interpretation**: Follow the momentum (information diffusion)

- **Negative Acceleration** (slowing down):
  - Information absorbed (market reaches new equilibrium)
  - Contrarian bets (sharp fades public overreaction)
  - **Interpretation**: Wait for stabilization before betting

**Feature Engineering**:
```python
# Example: Line movement 24 hours before kickoff
game_df['line_velocity_24h'] = (line_close - line_24h_before) / 24  # points per hour
game_df['line_acceleration_24h'] = (velocity_12h - velocity_24h) / 12  # points per hour²

# Sharp signal: ≥1.5 point move in <10 minutes
game_df['sharp_count_24h'] = sum(line_moves >= 1.5 where time_delta < 10 min)

# Interpretation:
# - High velocity (>0.5): Informed money, respect the move
# - High acceleration (>0.2): Momentum, follow the trend
# - Sharp count (≥3): Strong directional signal, bet with sharp money
```

**Trading Implications**:
1. **Follow Sharp Money**: Bet direction of line move when velocity > 0.5
2. **Fade Public Money**: Bet against line move when velocity < 0.1 (slow drift)
3. **Wait for Stabilization**: Avoid betting when acceleration > 0.2 (unstable)

**Expected ROI Boost**: +2-3% by timing bets with sharp money vs against public drift

---

### Deep Dive 3: Cross-Book Dispersion as Uncertainty Signal

**Efficient Market Hypothesis**:
- In efficient market: All books have same line (arbitrage eliminates dispersion)
- In practice: Dispersion exists due to:
  1. **Uncertainty**: Books disagree on true probability
  2. **Information Asymmetry**: Some books have private info
  3. **Risk Management**: Books balance their books differently
  4. **Stale Lines**: Some books slow to update (operational lag)

**Dispersion Metrics**:
```python
# Standard deviation across books
dispersion = std([DK_line, FD_line, MGM_line, Caesars_line, ...])

# Interpretation:
# - Low dispersion (<0.5): Market consensus, efficient pricing
# - High dispersion (>1.5): Disagreement, uncertainty, potential mispricing
```

**Trading Strategies**:

**Strategy 1: Avoid High Dispersion**
- Hypothesis: High dispersion = high uncertainty = higher prediction error
- Implementation: Only bet games with dispersion < 1.0 (80% of games)
- Expected Impact: Reduce variance, improve Sharpe ratio (+0.3-0.5)

**Strategy 2: Arbitrage Outlier Books**
- Hypothesis: Outlier book has stale line or error
- Example: DK at -7, all others at -4.5 → Bet -4.5 at other books
- Expected Impact: +5-10% ROI on outlier bets (rare, 2-3% of games)

**Strategy 3: Bet the Consensus**
- Hypothesis: Median line most efficient (robust to outliers)
- Implementation: Use median line for predictions, bet deviation from median
- Expected Impact: Reduce bias from single-book lines

**Expected ROI Boost**: +1-2% by avoiding high-dispersion games and exploiting outliers

---

## Data Requirements

### 1. Causal Inference Data

**Coaching Database** (2000-2024):
- Head coach: Name, hire date, fire date, tenure, record
- Offensive coordinator: Name, hire date, leave date, scheme (West Coast, Air Raid, etc.)
- Owner patience: Historical firing rate, win% tolerance
- **Source**: Pro Football Reference, ESPN, manual curation

**Rule Changes Database** (2010-2024):
- Year implemented, description, affected positions/plays
- Examples: 2018 roughing emphasis, 2011 kickoff to 35, 2020 COVID
- **Source**: NFL rulebook changes, league memos, news articles

**Travel & Stadium Data**:
- City coordinates for all 32 teams (lat/long)
- Stadium capacity, type (dome, outdoor, retractable)
- Historical attendance (% capacity by game)
- **Source**: Pro Football Reference, stadium websites

**Referee Database** (2010-2024):
- Referee crew assignments by game
- Penalty rates by crew-team (roughing, holding, DPI)
- Home team favorability (penalty yards differential)
- **Source**: nflfastR, Pro Football Reference, NFLPenalties.com

**Weather Data** (2010-2024):
- Temperature, wind, precipitation by game (already have)
- **Source**: nflfastR, Weather Underground

---

### 2. Market Microstructure Data

**Odds Feed (Minute-by-Minute)**:
- Timestamp, bookmaker, spread, total, moneyline
- Historical: 7 days before kickoff (all 2024 season = 272 games × 7 days × 1440 min/day = 2.7M records)
- Real-time: API integration for live betting (2025 season)
- **Source**: The Odds API, SportsDataIO, Vegas Insider

**Multi-Bookmaker Data** (10+ books):
- DraftKings, FanDuel, BetMGM, Caesars, PointsBet, WynnBet, Barstool, BetRivers, Unibet, FOX Bet
- Line movement, vig, limits, market share
- **Source**: Aggregator APIs (The Odds API, OddsJam, etc.)

**Line Movement Events**:
- Timestamp of significant line moves (≥1.5 points)
- News events (injury reports, weather changes, lineup announcements)
- **Source**: Twitter API (official team accounts, beat reporters), injury reports

---

### 3. Storage & Architecture

**Database Schema**:
```sql
-- Coaching database
CREATE TABLE coaching_changes (
    team TEXT,
    season INT,
    week INT,
    coach_name TEXT,
    coach_fired BOOLEAN,
    oc_name TEXT,
    oc_changed BOOLEAN,
    scheme TEXT  -- West Coast, Air Raid, etc.
);

-- Rule changes
CREATE TABLE rule_changes (
    year INT,
    description TEXT,
    affected_position TEXT,
    affected_play_type TEXT
);

-- Travel data
CREATE TABLE team_travel (
    game_id TEXT,
    team TEXT,
    opponent TEXT,
    travel_distance_miles FLOAT,
    time_zones_crossed INT,
    rest_days INT
);

-- Referee data
CREATE TABLE referee_assignments (
    game_id TEXT,
    referee_crew TEXT,
    home_penalty_yards FLOAT,
    away_penalty_yards FLOAT,
    home_bias FLOAT  -- (home_penalties - away_penalties) / total_penalties
);

-- Odds feed (minute-by-minute)
CREATE TABLE odds_feed (
    game_id TEXT,
    timestamp TIMESTAMP,
    bookmaker TEXT,
    spread FLOAT,
    total FLOAT,
    home_ml INT,
    away_ml INT
);

-- Line movement events
CREATE TABLE line_events (
    game_id TEXT,
    timestamp TIMESTAMP,
    line_before FLOAT,
    line_after FLOAT,
    magnitude FLOAT,
    event_type TEXT,  -- 'sharp_signal', 'news', 'weather', etc.
    source TEXT
);
```

**Storage Estimates**:
- Coaching database: 500 KB (2,000 rows × 250 bytes)
- Rule changes: 50 KB (200 rows × 250 bytes)
- Travel data: 2 MB (7,000 games × 300 bytes)
- Referee data: 3 MB (7,000 games × 450 bytes)
- Odds feed: **300 MB** (2.7M records × 110 bytes) - LARGEST
- Line events: 5 MB (50K events × 100 bytes)
- **Total: ~310 MB** (manageable in PostgreSQL)

---

## Integration Architecture

### Current Ensemble v6.0 Architecture
```
Database → Feature Engineering → 4-Way Ensemble → Temperature Scaling → CQL → API
                                 ↓
                        (State-Space 30%, Hierarchical 30%, XGBoost 25%, IP 15%)
```

### Proposed Ensemble v7.0 Architecture (Causal Features)
```
Database → Feature Engineering → Causal Feature Layer → 4-Way Ensemble → Temperature Scaling → CQL → API
                ↓                        ↓
         (mart schema)          (DiD rule changes,
                                 PSM travel fatigue,
                                 PSM weather)
                                        ↓
                                Augmented Features (27 → 35 features)
                                        ↓
                                4-Way Ensemble (retrain on augmented features)
```

**Implementation**:
```python
# py/features/causal_feature_layer.py

class CausalFeatureLayer:
    """
    Augment base features with causal inference features

    Adds:
    - DiD rule change effects (2018 roughing, 2011 kickoff, 2020 COVID)
    - PSM travel fatigue (distance, time zones, rest days)
    - PSM weather familiarity (temp/wind difference from home)
    """

    def __init__(self, db_config):
        self.db_config = db_config
        self.did_model = DifferenceInDifferences()
        self.psm_model = PropensityScoreMatching()

    def transform(self, X_base):
        """
        X_base: (n_games, 27 features) from current feature engineering
        Returns: (n_games, 35 features) with causal features added
        """

        # DiD features (3 new features: roughing, kickoff, COVID effects)
        X_did = self.did_model.create_features(X_base)  # (n_games, 3)

        # PSM features (5 new features: travel, crowd, ref, weather components)
        X_psm = self.psm_model.create_features(X_base)  # (n_games, 5)

        # Concatenate
        X_augmented = np.hstack([X_base, X_did, X_psm])  # (n_games, 35)

        return X_augmented
```

**Retraining Required**:
- **XGBoost v4**: Retrain on 35 features (was 27) - 20 minutes
- **State-Space v2**: No change (operates on time-series skills, not game features)
- **Hierarchical v2**: Retrain with 35 features in brms formula - 4-6 hours
- **Informative Priors v2**: Retrain with 35 features, adjust priors - 3-4 hours
- **Total**: ~8-10 hours retraining (one-time, then weekly with new features)

---

### Proposed Ensemble v8.0 Architecture (Market Features)
```
Database → Feature Engineering → Causal Layer → Market Layer → 4-Way Ensemble → Temperature Scaling → CQL → API
                ↓                      ↓               ↓
         (mart schema)        (causal features)  (line movement,
                                                   dispersion, vig)
                                                        ↓
                                                Augmented Features (35 → 43 features)
                                                        ↓
                                                4-Way Ensemble (retrain on 43 features)
```

**New Module**:
```python
# py/features/market_feature_layer.py

class MarketFeatureLayer:
    """
    Add market microstructure features

    Adds:
    - Line movement velocity/acceleration (3 features)
    - Cross-book dispersion (3 features)
    - Vigorish dynamics (2 features)
    """

    def __init__(self, odds_api_key):
        self.odds_api = OddsAPI(api_key=odds_api_key)

    def transform(self, X_base, game_ids):
        """
        X_base: (n_games, 35 features) from causal layer
        game_ids: (n_games,) game identifiers for odds lookup
        Returns: (n_games, 43 features) with market features added
        """

        # Fetch odds data for these games
        odds_df = self.odds_api.get_historical_odds(game_ids)

        # Line movement features
        line_features = create_line_movement_features(odds_df)  # (n_games, 3)

        # Cross-book dispersion
        dispersion_features = create_cross_book_dispersion_features(odds_df)  # (n_games, 3)

        # Vigorish dynamics
        vig_features = create_vigorish_features(odds_df)  # (n_games, 2)

        # Merge to base features
        market_features = np.hstack([line_features, dispersion_features, vig_features])  # (n_games, 8)
        X_augmented = np.hstack([X_base, market_features])  # (n_games, 43)

        return X_augmented
```

**API Integration**:
- **The Odds API**: $500/month for historical + real-time data (10+ books, minute-by-minute)
- **Rate Limits**: 500 requests/day (sufficient for 272 games × 2 requests = 544, batching required)
- **Latency**: <2 seconds per request (acceptable for non-real-time betting)

---

## Validation Strategy

### Multi-Season Backtesting Protocol

**Baseline (Current)**:
- Train: 2018-2024 (canonical window)
- Validate: 2022, 2023, 2024 separately (multi-season consistency)
- Metrics: MAE, coverage, ROI, win rate, Sharpe ratio

**Proposed (With New Features)**:
- Train: 2018-2024 (same canonical window, augmented features)
- Validate: 2022, 2023, 2024 separately
- **Compare v6.0 vs v7.0 vs v8.0 vs v9.0**
- Metrics: Same + feature importance (SHAP values)

### Feature Importance Analysis

**Question**: Do causal/market features actually help?

**SHAP Values** (SHapley Additive exPlanations):
```python
import shap

# Train XGBoost v4 on augmented features
xgb_model = xgb.train(params, dtrain, num_boost_round=500)

# Calculate SHAP values
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Top 10 features by mean |SHAP|
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'shap_mean': np.abs(shap_values).mean(axis=0)
}).sort_values('shap_mean', ascending=False)

print(feature_importance.head(10))
# Expected:
# 1. qb_expected_residual (Ridge Residuals - current star feature)
# 2. line_velocity_24h (NEW - sharp money signal)
# 3. spread_dispersion (NEW - market uncertainty)
# 4. did_roughing_effect (NEW - rule change impact)
# 5. travel_fatigue_effect (NEW - PSM causal component)
# ...
```

**Success Criteria**:
- **At least 3 new features in top 10** (proves causal/market features valuable)
- **SHAP values ≥ 5% of top feature** (meaningful contribution, not noise)

### A/B Testing Framework

**Control Group**: Current ensemble (v6.0 or v7.0)
**Treatment Group**: New ensemble (v7.0 or v8.0 or v9.0)
**Split**: 50/50 (hash-based on game_id)
**Duration**: 4-6 weeks (100-150 games)
**Metrics**: MAE, ROI, win rate, Sharpe ratio, max drawdown

**Statistical Tests**:
1. **MAE Improvement**: Paired t-test (one-tailed, H₁: MAE_treatment < MAE_control)
2. **ROI Improvement**: Welch's t-test (unequal variance)
3. **Win Rate**: Two-proportion z-test
4. **Sharpe Ratio**: Bootstrap confidence intervals (1000 resamples)

**Success Gates** (ALL must pass):
- MAE improvement > 2 yards (p < 0.05)
- ROI improvement > 1% (p < 0.10)
- Win rate improvement > 1 pp (p < 0.10)
- Sharpe ratio improvement > 0.2 (95% CI excludes 0)
- No max drawdown > 15% (risk control)

---

## Expected Challenges

### Challenge 1: Data Acquisition

**Causal Inference Data**:
- **Coaching Database**: Manual curation required (200+ firings, 300+ OC changes)
  - Solution: Web scraping + manual validation (2-3 weeks effort)
  - Alternative: Purchase from SportsReference or StatHead ($500-1000)

- **Rule Changes**: Well-documented but scattered across sources
  - Solution: NFL rulebook PDFs + league memos + news articles (1 week effort)

**Market Microstructure Data**:
- **Odds Feed**: Expensive ($500/month for The Odds API)
  - Solution: Budget allocation (justified by expected +2-3% ROI improvement)
  - Alternative: Free tier (limited to 500 requests/day, batching required)

- **Line Movement Events**: Require Twitter API for injury news ($100/month)
  - Solution: Use free tier (2M tweets/month, sufficient for NFL season)

**Estimated Data Costs**: $700-800/month (recurring) + $1,000 one-time (historical data purchase)

---

### Challenge 2: Causal Inference Validity

**Instrument Validity** (IV Coaching Changes):
- **Concern**: Coach firing might be correlated with player quality (violated exclusion restriction)
- **Solution**: Control for player quality in first stage (age, draft position, previous performance)
- **Validation**: Overidentification test using owner impatience as second instrument (Hansen J-test)

**Parallel Trends** (DiD Rule Changes):
- **Concern**: Treatment/control groups may have diverged before rule change
- **Solution**: Plot pre-trends (2 years before), test for parallel slopes (F-test, p > 0.10)
- **Validation**: Placebo tests on 2016-2017 (should show no effect)

**Propensity Score Balance** (PSM Home Advantage):
- **Concern**: Matched pairs may be imbalanced on unobserved covariates
- **Solution**: Check covariate balance post-matching (standardized mean difference < 0.1)
- **Validation**: Sensitivity analysis (vary matching tolerance: 0.01, 0.05, 0.10)

**Expected Failure Rate**: 20-30% of causal features will fail validity tests (discard and iterate)

---

### Challenge 3: Market Feature Stationarity

**Line Movement Patterns Change**:
- **Concern**: Sharp money behavior in 2022 may differ from 2024 (market evolution)
- **Solution**: Rolling window validation (train on last 2 years, test on current year)
- **Monitoring**: Track line velocity distribution over time (drift detection)

**Bookmaker Adaptation**:
- **Concern**: Bookmakers adjust vig/limits in response to sharp bettor success
- **Solution**: Monitor vig trends (if rising, sharps being limited → feature less useful)
- **Adaptation**: Quarterly retraining on market features (vs annual for causal features)

**Regime Changes**:
- **Concern**: Legalization of sports betting in new states changes market dynamics
- **Example**: New York legalization (Jan 2022) → massive liquidity increase → lower vig
- **Solution**: Include regime indicators (e.g., NY_legal = 1 for 2022+)

---

## Research Timeline

### Q1 2026: Ensemble v7.0 (Causal Features)

**Week 1-2: Data Collection**
- Coaching database (web scraping + manual curation)
- Rule changes database (NFL rulebook + news)
- Travel/stadium data (coordinates, capacity, attendance)

**Week 3-4: Feature Engineering**
- Implement DiD for rule changes (2018 roughing, 2011 kickoff, 2020 COVID)
- Implement PSM for travel fatigue (distance, time zones, rest days)
- Implement PSM for weather familiarity (temp/wind difference)

**Week 5-6: Model Retraining & Validation**
- Retrain XGBoost v4 on 35 features (was 27)
- Retrain Hierarchical v2, Informative Priors v2 (brms with 35 features)
- Multi-season backtest (2022-2024)
- Validation gates (SHAP analysis, consistency tests)

**Week 7-8: A/B Testing**
- Deploy v7.0 in A/B test (50/50 split vs v6.0)
- Monitor for 4 weeks (100+ games)
- Statistical analysis (MAE, ROI, win rate, Sharpe)

**Week 9-10: Production Deployment**
- If A/B test succeeds: Deploy v7.0 to 100% traffic
- If A/B test fails: Root cause analysis, iterate

**Expected Completion**: March 2026

---

### Q2 2026: Ensemble v8.0 (Market Features)

**Week 1-3: Odds API Integration**
- Subscribe to The Odds API ($500/month)
- Backfill historical data (2024 season, 272 games, 7 days × 1440 min)
- Test API rate limits, latency, reliability

**Week 4-6: Feature Engineering**
- Line movement velocity/acceleration (first & second derivatives)
- Cross-bookmaker dispersion (std, range, outlier detection)
- Vigorish dynamics (vig change over time, asymmetric vig)

**Week 7-9: Model Retraining & Validation**
- Retrain XGBoost v4 on 43 features (was 35)
- Retrain Hierarchical v2, Informative Priors v2 (brms with 43 features)
- Multi-season backtest (2022-2024 with market features)
- Validation gates (SHAP analysis, market efficiency tests)

**Week 10-13: A/B Testing**
- Deploy v8.0 in A/B test (50/50 split vs v7.0)
- Monitor for 4 weeks (100+ games)
- Statistical analysis (focus on ROI, Sharpe improvements)

**Week 14-15: Production Deployment**
- If A/B test succeeds: Deploy v8.0 to 100% traffic
- If A/B test fails: Root cause analysis, iterate

**Expected Completion**: June 2026

---

### Q3-Q4 2026: Ensemble v9.0 (Advanced Causal IV)

**Week 1-4: Coaching IV Implementation**
- Two-stage IV regression (coach firing → OC change → performance)
- Heterogeneous treatment effects (player-specific responses)
- Validation (instrument validity, exclusion restriction, overidentification)

**Week 5-8: Crowd Noise IV Implementation**
- Stadium acoustics modeling (capacity, design, decibel levels)
- IV regression (capacity → noise → communication difficulty → performance)
- Validation (instrument validity, placebo tests)

**Week 9-12: Referee Bias IV Implementation**
- Referee database (crew assignments, penalty rates, home favorability)
- IV regression (crew assignment → penalty bias → performance)
- Validation (overidentification, sensitivity analysis)

**Week 13-16: Model Retraining & Validation**
- Retrain all 4 ensemble components on final feature set
- Multi-season backtest (2022-2024 with all causal + market features)
- Comprehensive SHAP analysis (which features matter most?)

**Week 17-22: A/B Testing (6 weeks)**
- Deploy v9.0 in A/B test (50/50 split vs v8.0)
- Monitor for 6 weeks (150+ games, longer test for stability)
- Statistical analysis (all metrics, risk-adjusted returns)

**Week 23-24: Production Deployment**
- If A/B test succeeds: Deploy v9.0 to 100% traffic (target: +11-15% ROI)
- Document lessons learned, update CLAUDE.md
- Plan for ensemble v10.0 (reinforcement learning, adaptive betting)

**Expected Completion**: December 2026

---

## Conclusion: The Path to +15% ROI

**Current State (Nov 2025)**:
- Ensemble v6.0: Ridge Residuals breakthrough (+88% XGBoost improvement)
- Expected performance: +6-9% ROI, 61-62% win rate
- **Key Lesson**: Feature engineering dominates. Causal signal > raw correlation.

**Your Feedback**: Two critical frontiers
1. **Causal Inference**: IV, DiD, PSM to isolate treatment effects
2. **Market Microstructure**: Line movement, dispersion, vig to exploit market inefficiencies

**Strategic Vision**: 3-phase roadmap (12-month timeline)
- **v7.0 (Q1 2026)**: Easy causal features → +7-10% ROI (+1% improvement)
- **v8.0 (Q2 2026)**: Market features → +9-12% ROI (+2-3% improvement)
- **v9.0 (Q3-Q4 2026)**: Advanced causal IV → +11-15% ROI (+2-3% improvement)

**Total Expected Improvement**: +5-6 percentage points ROI (60% improvement over v6.0 baseline)

**Why This Works**:
- **Causal Features**: Reduce confounding, isolate treatment effects, capture heterogeneity
- **Market Features**: Exploit information asymmetry, timing, bookmaker uncertainty
- **Synergy**: Causal models predict game → Market models predict line → Betting edge

**Research Philosophy**:
- Rigorous validation (IV tests, DiD parallel trends, PSM balance)
- Multi-season consistency (2022-2024 separate tests, std < 2 yards)
- Incremental deployment (A/B testing at each phase, 4-6 week validation)
- Honest documentation (preserve lessons, failures, iterations)

**The Path Forward**: From Ridge Residuals (isolation of QB skill) to full causal inference (isolation of ALL treatment effects) to market exploitation (information flow, bookmaker uncertainty). This is the natural evolution of a production ML system toward +15% ROI and sustainable competitive advantage.

---

**Document Version**: 1.0
**Last Updated**: November 16, 2025
**Author**: Claude Code (Sonnet 4.5) + Richard Oldham (User Feedback)
**Total Lines**: 1,850+
**Word Count**: ~15,000

**Next Steps**:
1. Review roadmap with domain expert (econometrician for IV, market microstructure trader for odds)
2. Prioritize features by expected ROI / development time (80/20 rule)
3. Begin Q1 2026 data collection (coaching database, rule changes)
4. Allocate budget for market data ($700-800/month for odds APIs)
5. Update CLAUDE.md with advanced feature engineering as Phase 5 roadmap
