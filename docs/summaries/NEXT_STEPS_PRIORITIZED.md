# Next Steps: Prioritized Development Roadmap

**Last Updated**: October 11, 2025
**Status**: Post-Infrastructure Build - Ready for Deployment & Enhancement

---

## 🎯 Mission

Deploy the predictions system for live betting, validate the RL feedback loop with historical data, and systematically implement high-ROI improvements to compound gains over time.

---

## 📊 Development Phases

### **PHASE 1: VALIDATION & DEPLOYMENT** (Week 1-2, Immediate Priority)

**Goal**: Prove the system works on historical data and deploy for live betting (paper trading).

#### 1.1 Historical Backtest (Priority: CRITICAL)
**Estimated Time**: 4-6 hours
**Dependencies**: Existing 2022-2023 predictions from prior XGBoost runs

**Tasks**:
```bash
# Step 1: Generate historical predictions with versioning
python py/predictions/generate_predictions.py \
  --season 2022 --all-remaining --version "final" \
  --xgb-model models/xgboost/v2_trained_on_2006_2021.json

python py/predictions/generate_predictions.py \
  --season 2023 --all-remaining --version "final" \
  --xgb-model models/xgboost/v2_trained_on_2006_2022.json

# Step 2: Run retrospective analysis
python py/predictions/retrospective_analyzer.py --auto

# Step 3: Extract learning patterns
python py/predictions/extract_learning_patterns.py \
  --min-sample-size 10 --verbose

# Step 4: Validate results
psql -h localhost -p 5544 -U dro devdb01 -c "
  SELECT outcome_type, COUNT(*), AVG(abs_margin_error)
  FROM predictions.retrospectives
  WHERE game_id LIKE '2022%' OR game_id LIKE '2023%'
  GROUP BY outcome_type;
"
```

**Success Metrics**:
- [ ] 500+ games analyzed (2022-2023 full seasons)
- [ ] Learning patterns extracted with statistical significance
- [ ] Win rate ≥ 52% on retrospective bets
- [ ] At least 3 high-priority learnings identified

**Deliverables**:
- Populated `predictions.retrospectives` table (2022-2023)
- Populated `predictions.learning_loop` table with patterns
- Validation report: `analysis/predictions/backtest_2022_2023_report.md`

---

#### 1.2 2025 Season Prediction Generation (Priority: HIGH)
**Estimated Time**: 2 hours
**Dependencies**: Bayesian models trained on 2015-2024, XGBoost v3

**Tasks**:
```bash
# Generate predictions for all remaining 2025 games
python py/predictions/generate_predictions.py \
  --season 2025 \
  --all-remaining \
  --version "final" \
  --xgb-model models/xgboost/v3_best.json \
  --features-csv data/processed/features/asof_team_features_v3_bayesian.csv \
  --output analysis/predictions/2025_season_predictions.json
```

**Success Metrics**:
- [ ] Predictions generated for all remaining 2025 games
- [ ] Edge estimates > 0.02 for 20-30% of games
- [ ] Ensemble agreement rate ≈ 70-80% (Bayesian + XGBoost)
- [ ] Predictions stored in database with full metadata

**Deliverables**:
- Database records in `predictions.game_predictions`
- JSON export for review: `analysis/predictions/2025_season_predictions.json`
- Betting recommendations CSV for paper trading

---

#### 1.3 Paper Trading Deployment (Priority: HIGH)
**Estimated Time**: 1 day
**Dependencies**: 2025 predictions generated

**Setup**:
1. Create paper trading spreadsheet/dashboard
2. Track recommendations vs actual outcomes
3. Monitor Closing Line Value (CLV) weekly
4. Calculate realized ROI

**Paper Trading Protocol**:
```
Week N (e.g., Week 7):
- Wednesday: Review predictions for upcoming week
- Wednesday: Record opening lines (Pinnacle/DraftKings)
- Sunday morning: Record closing lines
- Sunday: Place "paper bets" based on recommendations
- Monday: Record actual outcomes
- Tuesday: Update performance metrics
- Tuesday: Run retrospective analysis
- Tuesday: Check for new learning patterns
```

**Success Metrics**:
- [ ] Track 8-12 bets per week
- [ ] Target win rate ≥ 54%
- [ ] CLV > 0 (beating closing line)
- [ ] Realized ROI ≥ +1.5% after 4 weeks

**Deliverables**:
- Paper trading tracker: `analysis/predictions/paper_trading_log.csv`
- Weekly performance summary
- CLV tracking report

---

#### 1.4 Automated Weekly Workflow (Priority: MEDIUM)
**Estimated Time**: 4 hours
**Dependencies**: Validated backtesting results

**Components**:

**1. Tuesday Morning Retraining Script** (`scripts/weekly_retraining.sh`):
```bash
#!/bin/bash
# Run every Tuesday at 8am

echo "=== Tuesday Weekly Retraining ==="
date

# 1. Data quality check
echo "Checking data..."
psql -h localhost -p 5544 -U dro devdb01 -c "
  SELECT COUNT(*) as monday_games
  FROM games
  WHERE season = 2025
    AND EXTRACT(DOW FROM kickoff) = 1  -- Monday
    AND kickoff < NOW()
    AND home_score IS NOT NULL;
"

# 2. Bayesian retraining
echo "Retraining Bayesian models..."
Rscript R/models/bayesian_hierarchical_weekly.R \
  --current-season 2025 \
  --lookback-seasons 5 \
  --output models/bayesian/weekly_ratings_$(date +%Y%m%d).rds

# 3. Feature generation with Bayesian
echo "Regenerating features with Bayesian ratings..."
python py/features/bayesian_features.py \
  --input data/processed/features/asof_team_features_v3.csv \
  --output data/processed/features/asof_team_features_v3_bayesian.csv \
  --add-predictions

# 4. Generate predictions for upcoming week
NEXT_WEEK=$(($(date +%V) + 1))
echo "Generating predictions for Week $NEXT_WEEK..."
python py/predictions/generate_predictions.py \
  --season 2025 \
  --week $NEXT_WEEK \
  --version "5_days_out"

# 5. Retrospective analysis on last week
LAST_WEEK=$(($(date +%V) - 1))
echo "Running retrospective analysis for Week $LAST_WEEK..."
python py/predictions/retrospective_analyzer.py --auto

# 6. Learning pattern extraction (monthly)
if [ $(date +%d) -le 7 ]; then
  echo "Monthly pattern extraction..."
  python py/predictions/extract_learning_patterns.py \
    --min-sample-size 10 --verbose \
    > logs/learning_patterns_$(date +%Y%m).txt
fi

echo "=== Tuesday Retraining Complete ==="
```

**2. Cron Job Setup**:
```bash
# Edit crontab
crontab -e

# Add weekly Tuesday job at 8am
0 8 * * 2 /Users/dro/rice/nfl-analytics/scripts/weekly_retraining.sh >> /Users/dro/rice/nfl-analytics/logs/weekly_$(date +\%Y\%m\%d).log 2>&1
```

**3. Email Alerts** (`scripts/send_alerts.py`):
```python
# Monitor performance and send alerts
# Trigger alerts for:
# - Win rate < 52% for 3 weeks
# - CLV < -5 bps for 2 weeks
# - Bayesian convergence failures
# - High-priority learnings discovered
```

**Success Metrics**:
- [ ] Cron job runs successfully every Tuesday
- [ ] All steps complete in < 30 minutes
- [ ] Email alerts configured and tested
- [ ] Logs stored in `logs/` directory

**Deliverables**:
- `scripts/weekly_retraining.sh` (production ready)
- `scripts/send_alerts.py` (email notifications)
- Cron job configured and documented
- Log rotation setup

---

### **PHASE 2: FIRST IMPROVEMENTS** (Week 3-4, High Priority)

**Goal**: Implement first learning patterns discovered from backtesting and validate improvement.

#### 2.1 Thursday Night Effect Feature (Priority: HIGH)
**Estimated Time**: 3-4 hours
**Expected Gain**: +0.5% ROI

**If backtest shows Thursday Night home bias:**

```python
# Add to py/features/asof_features_enhanced.py

def add_thursday_night_feature(games_df):
    """Add Thursday Night Football home advantage indicator."""
    games_df['thursday_night'] = (
        games_df['kickoff'].dt.dayofweek == 3  # Thursday = 3
    ).astype(int)

    # Adjust home advantage for TNF
    # Add ~2 points home advantage on Thursday nights
    games_df['home_adv_adjusted'] = np.where(
        games_df['thursday_night'] == 1,
        games_df['home_adv_base'] + 2.0,
        games_df['home_adv_base']
    )

    return games_df
```

**Validation**:
1. Retrain XGBoost with new feature on 2006-2023
2. Test on 2024 holdout (exclude from training)
3. Compare Brier score before/after
4. Validate on TNF games specifically

**Success Metrics**:
- [ ] Feature implemented and tested
- [ ] Brier improvement ≥ 0.001 on 2024 holdout
- [ ] TNF game accuracy improves by ≥ 2 pp
- [ ] Mark pattern as `implemented` in `predictions.learning_loop`

---

#### 2.2 Monitoring Dashboard (Priority: MEDIUM)
**Estimated Time**: 6-8 hours
**Tool**: Streamlit

**Dashboard Components**:

```python
# streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import psycopg

st.set_page_config(page_title="NFL Predictions Dashboard", layout="wide")

# Sidebar: Date range filter
season = st.sidebar.selectbox("Season", [2022, 2023, 2024, 2025])
week_range = st.sidebar.slider("Week Range", 1, 18, (1, 18))

# Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Win Rate", "54.2%", "+2.2pp")
with col2:
    st.metric("Expected ROI", "+2.1%", "+0.5pp")
with col3:
    st.metric("Avg CLV", "+3.2 bps", "+1.1 bps")
with col4:
    st.metric("Brier Score", "0.251", "-0.004")

# Charts
st.subheader("Performance Over Time")
# Line chart: Win rate by week
# Bar chart: Bets placed vs recommended
# Scatter: Predicted prob vs actual outcome

st.subheader("Learning Opportunities")
# Table: Top 5 unimplemented patterns from learning_loop
# With sample size, p-value, expected improvement

st.subheader("Recent Predictions")
# Table: Last 10 games with predictions, outcomes, profit/loss
```

**Success Metrics**:
- [ ] Dashboard deployed locally (localhost:8501)
- [ ] Real-time data from PostgreSQL
- [ ] Interactive filters working
- [ ] Performance metrics accurate

---

### **PHASE 3: MODEL ENHANCEMENT** (Week 5-8, Medium Priority)

**Goal**: Retrain XGBoost with Bayesian features, validate improvement, deploy enhanced model.

#### 3.1 XGBoost v3 Retraining with Bayesian Features
**Estimated Time**: 1 day (6-8 hours including hyperparameter sweep)
**Expected Gain**: +0.3-0.5% Brier improvement

```bash
# Full retraining with Bayesian features
uv run python py/models/xgboost_gpu_v3.py \
  --features-csv data/processed/features/asof_team_features_v3_bayesian.csv \
  --start-season 2006 \
  --end-season 2023 \
  --test-seasons 2024 \
  --sweep \
  --output-dir models/xgboost/v3_bayesian_$(date +%Y%m%d) \
  --device cuda
```

**Validation**:
- Compare to v3 baseline (no Bayesian features)
- Require Brier ≤ 0.255 on 2024 holdout
- ATS accuracy ≥ 52.5%
- Feature importance analysis: Are Bayesian features used?

**Success Metrics**:
- [ ] Training completes without errors
- [ ] Validation metrics meet thresholds
- [ ] Bayesian features in top 20 importance
- [ ] Deploy as `models/xgboost/v3_best.json` (replace current)

---

#### 3.2 Implement Additional Learning Patterns
**Estimated Time**: 2-3 hours per pattern
**Candidates** (based on expected backtest results):

1. **Backup QB Uncertainty** (+0.4% ROI expected)
   - Add `backup_qb_start` indicator
   - Increase Bayesian SD when backup starts
   - Reduce bet sizing (lower Kelly fraction)

2. **Divisional Game Variance** (+0.3% ROI expected)
   - Add `divisional_game` indicator
   - Adjust prediction confidence downward
   - Historical divisional matchup features

3. **Weather Impact Refinement** (+0.2% ROI expected)
   - Interaction: `weather_severity * pass_rate`
   - Dome vs outdoor adjustments
   - Wind speed thresholds (>15 mph)

**Implementation Priority**:
1. Validate pattern on 2022-2023 holdout
2. Implement feature engineering
3. Retrain model (or adjust post-prediction)
4. Backtest on 2024
5. Mark as implemented in database

---

### **PHASE 4: ADVANCED MODELS** (Month 3-4, Lower Priority)

**Goal**: Implement Dixon-Coles Phase 2 and begin NLP pipeline.

#### 4.1 Dixon-Coles Bivariate Poisson Phase 2
**Estimated Time**: 8-12 hours
**Expected Gain**: +0.3-0.8% ROI on spread/total bets

**Full EM Algorithm Implementation**:
```python
# py/models/dixon_coles_em.py

class DixonColesEM:
    def __init__(self, rho_init=-0.1):
        self.rho = rho_init
        self.alpha = {}  # attack strengths
        self.delta = {}  # defense strengths
        self.gamma = 0.14  # home advantage

    def fit(self, games_df, max_iter=100, tol=1e-4):
        """Fit Dixon-Coles with EM algorithm."""
        # E-step: Compute expected sufficient statistics
        # M-step: Update parameters
        # Iterate until convergence
        pass

    def predict_score_distribution(self, home_team, away_team):
        """Return full score distribution PMF."""
        pass
```

**Key Features**:
- Low-score correlation adjustment ($\tau$ function)
- Time-weighted training (half-life = 4 weeks)
- Attack/defense decomposition
- Exact score probabilities

**Use Cases**:
- Same-game parlays (spread + total correlation)
- Exact score betting
- Push risk analysis at key numbers

---

#### 4.2 NLP Sentiment Pipeline Architecture
**Estimated Time**: 2-3 weeks
**Expected Gain**: +1.5-3.0% ROI

**Phase 6.1 Plan**:

**Data Sources**:
1. ESPN articles (RSS feeds)
2. Twitter/X (team accounts, beat reporters)
3. Reddit (/r/nfl, team subreddits)
4. Press conferences (transcripts via YouTube API)

**Pipeline**:
```python
# 1. Data Collection (daily)
scrape_espn_articles(teams=['all'])
scrape_twitter_sentiment(accounts=BEAT_REPORTERS)
scrape_reddit_threads(subreddits=['nfl'] + TEAM_SUBS)

# 2. Sentiment Analysis (BERT)
from transformers import pipeline
sentiment_analyzer = pipeline("sentiment-analysis",
                               model="bert-base-uncased")

# 3. Feature Engineering
features = {
    'media_sentiment_home': mean_sentiment_last_7_days,
    'media_sentiment_away': mean_sentiment_last_7_days,
    'narrative_momentum_home': trend_direction,
    'locker_room_stability': controversy_mentions,
}

# 4. Integration
merge_into_xgboost_features()
```

**12 New Features**:
- `media_sentiment_home/away` (ESPN + Twitter)
- `reddit_sentiment_home/away` (fan sentiment)
- `narrative_momentum` (trend direction)
- `coaching_drama_indicator` (mentions of "fired", "hot seat")
- `locker_room_stability` (controversy keywords)
- `injury_narrative_load` (media attention on injuries)

**Success Metrics**:
- [ ] Sentiment scores correlate with outcomes (r > 0.1)
- [ ] Improves XGBoost Brier by ≥ 0.003
- [ ] Identifies games with narrative edge

---

### **PHASE 5: SUPER-ENSEMBLE** (Month 5-6, Research Priority)

**Goal**: Explore GNN, GAN, Deep RL architectures for ensemble integration.

#### 5.1 Technical Specification Document
**Estimated Time**: 1 week
**Deliverable**: `docs/SUPER_ENSEMBLE_TECHNICAL_SPEC.md`

**Contents**:
1. **Bayesian-GNN Hybrid**
   - Use Bayesian posteriors as node priors
   - Message passing with uncertainty weighting
   - Graph structure: Teams as nodes, games as edges

2. **GAN Scenario Simulator**
   - Generate 10K synthetic games
   - Conditioned on Bayesian team strengths
   - Data augmentation for rare scenarios (e.g., snow games)

3. **Bayesian Deep RL**
   - Thompson Sampling with Bayesian priors
   - Explore-exploit betting strategy
   - Multi-armed bandit framework

4. **Hierarchical Meta-Learner**
   - Stack: XGBoost, Bayesian, GNN, GAN, Deep RL
   - Learn optimal weights per game context
   - Expected performance: 57-58% win rate, +5-6% ROI

**Success Metrics**:
- [ ] Complete technical specification
- [ ] Proof-of-concept for each component
- [ ] Computational cost analysis
- [ ] Expected ROI projections

---

## 📋 Summary Checklist

### Week 1-2: Validation & Deployment
- [ ] Backtest on 2022-2023 (500+ games analyzed)
- [ ] Generate 2025 season predictions
- [ ] Deploy paper trading tracker
- [ ] Set up automated weekly workflow

### Week 3-4: First Improvements
- [ ] Implement Thursday Night Effect feature
- [ ] Build monitoring dashboard (Streamlit)
- [ ] Validate first learning pattern
- [ ] Update production model

### Week 5-8: Model Enhancement
- [ ] Retrain XGBoost v3 with Bayesian features
- [ ] Implement 2-3 additional learning patterns
- [ ] Monitor live performance
- [ ] Iterate based on feedback

### Month 3-4: Advanced Models
- [ ] Dixon-Coles Phase 2 with EM algorithm
- [ ] NLP sentiment pipeline (Phase 6.1)
- [ ] Deploy sentiment features

### Month 5-6: Super-Ensemble
- [ ] Technical specification document
- [ ] Proof-of-concept implementations
- [ ] ROI projections and cost analysis

---

## 🎯 Success Criteria

**End of Phase 1** (Week 2):
- Historical backtest validates RL feedback loop
- 2025 predictions generated and stored
- Paper trading deployed
- Automated workflow running

**End of Phase 2** (Week 4):
- First learning pattern implemented and validated
- Monitoring dashboard operational
- Live performance tracked

**End of Phase 3** (Week 8):
- XGBoost v3 with Bayesian features deployed
- 2-3 learning patterns implemented
- Win rate ≥ 54%, ROI ≥ +2.0%

**End of Phase 4** (Month 4):
- Dixon-Coles Phase 2 deployed
- NLP sentiment features integrated
- Win rate ≥ 55%, ROI ≥ +3.0%

**End of Phase 5** (Month 6):
- Super-ensemble specification complete
- Proof-of-concept validated
- Roadmap for 57-58% win rate, +5-6% ROI

---

**Last Updated**: October 11, 2025
**Next Review**: After Phase 1 completion (Week 2)
