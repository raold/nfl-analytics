# NFL Player Props Betting System - Complete Guide

**Last Updated:** November 1, 2025
**Status:** ✅ Core Infrastructure Complete, Ready for Production
**Consolidated From:** PROPS_BETTING_SYSTEM.md, PROPS_QUICK_START.md, PROPS_IMPLEMENTATION_SUMMARY.md, PROPS_SYSTEM_STATUS.md

---

## 📋 Table of Contents

1. [Quick Start](#quick-start) - Get running in 10 minutes
2. [System Overview](#system-overview) - Architecture and approach
3. [Current Status](#current-status) - What's built and what's next
4. [Setup](#setup) - Detailed installation
5. [Usage](#usage) - Daily workflow
6. [Components](#components) - Technical details
7. [Performance](#performance) - Expected ROI and metrics
8. [Advanced Features](#advanced-features) - Correlation adjustment, Kelly sizing

---

## 🚀 Quick Start

Get up and running with the NFL player props betting system in under 10 minutes.

### Prerequisites

- PostgreSQL database running (with NFL data loaded)
- Python 3.11+ with `uv` installed
- The Odds API key ([Get one free](https://the-odds-api.com))

### 1. Install Dependencies

```bash
uv pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Add to .env file
echo "ODDS_API_KEY=your_key_here" >> .env
```

### 3. Fetch Latest Props

```bash
uv run python py/data/fetch_prop_odds.py
```

### 4. Generate Predictions

```bash
uv run python py/production/generate_prop_predictions.py
```

### 5. View Recommendations

Check `predictions/props_recommendations_YYYY-MM-DD.csv` for today's best bets.

---

## 🎯 System Overview

Complete end-to-end system for profitable NFL player props betting using machine learning, EV-based bet selection, and correlation-adjusted position sizing.

### Why Player Props?

Player props betting differs from game-level betting:

- **Higher book holds** (7-10% vs 4-5% for games)
- **Lower betting limits** (typically $500-2000 per bet)
- **More line movement volatility**
- **Injury status critical** (one injury = prop void)
- **Correlation between props** (e.g., passing_yards ↔ passing_tds)

### Our Approach

1. **XGBoost predictions** with uncertainty quantification
2. **EV-based selection** with 3% minimum edge
3. **Correlation-adjusted sizing** to manage exposure
4. **Injury checking** to avoid void bets
5. **Multi-book comparison** to find best lines

### Architecture

```
┌─────────────────────┐
│  The Odds API       │  ← Fetch prop lines from sportsbooks
│  (DK, FD, MGM, etc) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ prop_lines_history  │  ← TimescaleDB table (time-series optimized)
│ (PostgreSQL)        │
└──────────┬──────────┘
           │
           ├────────────────────────────┐
           │                            │
           ▼                            ▼
┌─────────────────────┐      ┌─────────────────────┐
│ Player Features     │      │  Latest Prop Lines  │
│ - Rolling averages  │      │  - Best odds        │
│ - Opponent defense  │      │  - Book hold        │
│ - Game script       │      │  - Line movement    │
│ - Weather/injuries  │      └──────────┬──────────┘
└──────────┬──────────┘                 │
           │                            │
           ▼                            │
┌─────────────────────┐                 │
│ XGBoost Models      │                 │
│ (8 prop types)      │                 │
│ + Uncertainty       │                 │
└──────────┬──────────┘                 │
           │                            │
           └────────────┬───────────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │  EV Bet Selector    │
              │  - Min edge: 3%     │
              │  - Injury checks    │
              │  - Correlation adj  │
              │  - Kelly sizing     │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Recommended Bets    │
              │ (CSV + Summary)     │
              └─────────────────────┘
```

---

## 📊 Current Status

**Last Updated:** October 12, 2025

### ✅ Complete

**Data Layer:**
- ✅ PostgreSQL schema with TimescaleDB for prop lines
- ✅ 63,055 player-game feature records (2020-2025)
- ✅ All offensive positions: QB, RB, WR, TE
- ✅ The Odds API integration
- ✅ Schema mismatches resolved

**Feature Engineering:**
- ✅ Rolling performance averages (3, 5, 10 games)
- ✅ Opponent defensive rankings
- ✅ Game script projections
- ✅ Weather impact features
- ✅ Injury status integration

**Models:**
- ✅ XGBoost models for 8 prop types
- ✅ Uncertainty quantification
- ✅ Model validation framework

**Betting Infrastructure:**
- ✅ EV calculation engine
- ✅ Multi-book line comparison
- ✅ Correlation matrix estimation
- ✅ Kelly criterion sizing
- ✅ Injury status checking

**Production Integration:**
- ✅ Automated prediction pipeline
- ✅ CSV output for bet recommendations
- ✅ Dashboard integration ready

### 🔄 In Progress

- Real-time line monitoring
- Automated bet placement (manual verification required)
- Historical performance tracking dashboard

### 📋 Roadmap

**Short-term (Next 2 weeks):**
1. Add live injury status API integration
2. Implement bet tracking database
3. Create performance analytics dashboard

**Medium-term (Next month):**
4. Add defensive props (sacks, INTs, tackles)
5. Implement closing line value (CLV) tracking
6. Build automated reporting system

**Long-term (Next quarter):**
7. Multi-sport expansion (NBA, MLB)
8. Ensemble model combining XGBoost + Bayesian
9. Advanced bankroll management strategies

---

## 🛠️ Setup

### Detailed Installation

#### 1. Database Setup

```bash
# Initialize database (if not already done)
bash scripts/dev/init_dev.sh

# Create props-specific tables
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -f db/migrations/015_props_schema.sql
```

#### 2. Python Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

#### 3. API Configuration

Get your free API key from [The Odds API](https://the-odds-api.com):

```bash
# Add to .env
cat >> .env << EOF
ODDS_API_KEY=your_actual_key_here
ODDS_API_REGION=us
ODDS_API_MARKETS=player_pass_yds,player_rush_yds,player_receptions
EOF
```

#### 4. Data Ingestion

```bash
# Fetch historical prop lines (if available)
uv run python py/data/fetch_prop_odds.py --historical --start-date 2024-09-01

# Fetch current week's lines
uv run python py/data/fetch_prop_odds.py
```

#### 5. Model Training (Optional)

If you want to retrain models:

```bash
# Train all prop type models
uv run python py/models/train_props_models.py --all

# Or train specific prop type
uv run python py/models/train_props_models.py --prop-type passing_yards
```

---

## 💼 Usage

### Daily Workflow

#### Morning (9 AM ET)

1. **Fetch latest lines:**
```bash
uv run python py/data/fetch_prop_odds.py
```

2. **Check injury reports:**
```bash
uv run python py/data/fetch_injury_updates.py
```

3. **Generate predictions:**
```bash
uv run python py/production/generate_prop_predictions.py
```

4. **Review recommendations:**
```bash
cat predictions/props_recommendations_$(date +%Y-%m-%d).csv
```

#### Afternoon (1 PM ET)

5. **Update for any line movement:**
```bash
uv run python py/data/fetch_prop_odds.py --update
uv run python py/production/generate_prop_predictions.py --refresh
```

#### Pre-Game (30 minutes before kickoff)

6. **Final injury check:**
```bash
uv run python py/production/verify_active_players.py
```

7. **Place bets** (manual or via API)

#### Post-Game

8. **Update results:**
```bash
uv run python py/production/update_bet_results.py
```

### Command Reference

```bash
# Fetch props for specific games
uv run python py/data/fetch_prop_odds.py --games "KC@BUF,SF@DAL"

# Generate predictions with custom thresholds
uv run python py/production/generate_prop_predictions.py \
  --min-edge 0.05 \
  --max-bets 10 \
  --bankroll 10000

# Backtest models
uv run python py/backtests/props_backtest.py \
  --start-date 2024-09-01 \
  --end-date 2024-12-31

# View EV distribution
uv run python py/analysis/props_ev_analysis.py
```

---

## 🔧 Components

### 1. Data Layer

**Tables:**
- `prop_lines_history` - TimescaleDB hypertable for prop line history
- `player_game_stats` - Historical player performance
- `injuries` - Injury status and impact
- `bet_tracking` - Placed bets and results

**Key Columns (prop_lines_history):**
```sql
CREATE TABLE prop_lines_history (
  id BIGSERIAL PRIMARY KEY,
  fetched_at TIMESTAMPTZ NOT NULL,
  game_id INTEGER REFERENCES games(game_id),
  player_name TEXT NOT NULL,
  player_id INTEGER,
  position TEXT,
  prop_type TEXT NOT NULL,  -- passing_yards, rushing_yards, etc.
  line_value NUMERIC,        -- Over/Under line (e.g., 249.5)
  over_odds INTEGER,         -- American odds for Over
  under_odds INTEGER,        -- American odds for Under
  sportsbook TEXT NOT NULL,  -- DraftKings, FanDuel, etc.
  CONSTRAINT valid_prop_type CHECK (prop_type IN (...))
);

-- Hypertable for time-series optimization
SELECT create_hypertable('prop_lines_history', 'fetched_at');
```

### 2. Feature Engineering

**Player Features (`py/features/player_features.py`):**
- Rolling averages (3, 5, 10 game windows)
- Home/Away splits
- Division opponent performance
- Rest days impact
- Weather conditions

**Opponent Features:**
- Defensive rankings by position
- Recent defensive performance
- Injuries to key defensive players

**Game Script Features:**
- Projected point spread
- Over/Under total
- Expected game flow (pass-heavy vs run-heavy)

### 3. Models

**XGBoost Models (`models/props/`):**
- 8 prop types: passing_yards, rushing_yards, receptions, receiving_yards, passing_tds, rushing_tds, receiving_tds, anytime_td
- Features: ~50 per model
- Training data: 2020-2024 seasons
- Validation: 2024 holdout set

**Uncertainty Quantification:**
- Quantile regression (10th, 50th, 90th percentiles)
- Prediction intervals for risk assessment

### 4. EV Bet Selector

**Logic (`py/production/bet_selector.py`):**

```python
# Simplified EV calculation
def calculate_ev(prediction, line, over_odds, under_odds):
    """
    prediction: Model's predicted value (e.g., 265.3 yards)
    line: Sportsbook's over/under (e.g., 249.5 yards)
    over_odds: American odds for Over (e.g., -110)
    under_odds: American odds for Under (e.g., -110)
    """
    # Convert American odds to probabilities
    over_prob = american_to_prob(over_odds)
    under_prob = american_to_prob(under_odds)

    # Model's implied probability
    model_over_prob = percentile_rank(prediction, line)  # e.g., 0.65

    # Expected value for betting Over
    ev_over = (model_over_prob * (1/over_prob - 1)) - ((1-model_over_prob) * 1)

    # Only recommend if EV > 3%
    if ev_over > 0.03:
        return "Over", ev_over

    # Check Under side
    ev_under = ((1-model_over_prob) * (1/under_prob - 1)) - (model_over_prob * 1)

    if ev_under > 0.03:
        return "Under", ev_under

    return None, 0
```

**Selection Criteria:**
1. Minimum 3% edge
2. Player must be probable/healthy
3. Reasonable line (no extreme outliers)
4. Sufficient liquidity (at major books)

### 5. Position Sizing

**Correlation-Adjusted Kelly:**

```python
def kelly_fraction(ev, win_prob, correlation_factor=0.5):
    """
    ev: Expected value (e.g., 0.05 for 5% edge)
    win_prob: Probability of winning (from model)
    correlation_factor: Adjustment for correlated bets (0-1)
    """
    # Basic Kelly
    kelly = (win_prob * (1 + ev) - 1) / ev

    # Adjust for correlation
    adjusted_kelly = kelly * (1 - correlation_factor)

    # Cap at 2% of bankroll (quarter-Kelly)
    return min(adjusted_kelly, 0.02)
```

---

## 📈 Performance

### Expected Metrics

| Metric | Target | Conservative |
|--------|--------|--------------|
| ROI | +7-12% | +5-8% |
| Win Rate | 55-58% | 53-55% |
| Bets/Week | 10-20 | 5-10 |
| Avg Bet Size | $200-500 | $100-300 |
| Weekly Profit | $200-400 | $100-200 |

**Assumptions:**
- $10,000 bankroll
- 17-week NFL season
- Conservative bet sizing (quarter-Kelly)

### Validation Results (2024 Holdout)

**Passing Yards Model:**
- MAE: 25.6 yards
- R²: 0.78
- Calibration: 94% (predictions within 1σ interval)

**Rushing Yards Model:**
- MAE: 18.3 yards
- R²: 0.71
- Calibration: 91%

**Receptions Model:**
- MAE: 1.2 receptions
- R²: 0.82
- Calibration: 96%

---

## 🚀 Advanced Features

### 1. Correlation Adjustment

Props are often correlated (e.g., QB passing yards ↔ WR receiving yards). We estimate correlation matrices and adjust position sizing:

```python
# Correlation matrix (example)
corr_matrix = {
    ('QB_passing_yards', 'WR1_receiving_yards'): 0.45,
    ('QB_passing_yards', 'WR2_receiving_yards'): 0.32,
    ('RB_rushing_yards', 'QB_passing_yards'): -0.28,  # Negative correlation
}

# Adjust total exposure
max_correlated_exposure = 0.10  # Max 10% of bankroll on correlated props
```

### 2. Multi-Book Line Shopping

Automatically find best available odds across sportsbooks:

```python
# Find best Over line
best_over = max(lines, key=lambda x: american_to_decimal(x['over_odds']))

# Find best Under line
best_under = max(lines, key=lambda x: american_to_decimal(x['under_odds']))
```

### 3. Closing Line Value (CLV)

Track how our picks compare to closing lines:

```python
# CLV = (Our line - Closing line) for Over bets
# Positive CLV indicates we beat the closing line
clv = our_line - closing_line
```

### 4. Injury Impact Modeling

Adjust predictions based on missing teammates:

```python
# Example: WR production with backup QB
if qb_status == 'OUT' and backup_qb_quality < starter_quality:
    wr_yards_prediction *= 0.85  # 15% reduction
```

---

## 📂 File Structure

```
py/
├── data/
│   ├── fetch_prop_odds.py          # API integration
│   └── fetch_injury_updates.py     # Injury status
├── features/
│   └── player_features.py          # Feature engineering
├── models/
│   └── train_props_models.py       # Model training
├── production/
│   ├── generate_prop_predictions.py # Main prediction pipeline
│   ├── bet_selector.py              # EV-based selection
│   └── verify_active_players.py     # Pre-game checks
├── backtests/
│   └── props_backtest.py            # Historical validation
└── analysis/
    └── props_ev_analysis.py         # Performance analysis

db/migrations/
└── 015_props_schema.sql             # Database schema

models/props/
├── passing_yards_v1.pkl
├── rushing_yards_v1.pkl
└── ... (8 prop types)

predictions/
└── props_recommendations_YYYY-MM-DD.csv  # Daily output
```

---

## 🆘 Troubleshooting

### Common Issues

**"API rate limit exceeded"**
- The Odds API has limits (500 requests/month on free tier)
- Use `--cache` flag to reduce requests
- Upgrade to paid tier if needed

**"No predictions generated"**
- Check that models are trained: `ls models/props/`
- Verify player names match: Check `mart.player_hierarchy`
- Ensure recent data: Run `fetch_prop_odds.py`

**"High correlation warning"**
- System detected correlated bets exceeding exposure limit
- Reduce bet sizes or skip some props
- Review correlation matrix

**"Player marked inactive"**
- Check injury report manually
- Update injury status in database
- Re-run predictions after verification

---

## 📚 Related Documentation

- **[CLAUDE.md](../CLAUDE.md)** - Complete project documentation
- **[docs/milestones/BAYESIAN_PROPS_COMPLETE.md](milestones/BAYESIAN_PROPS_COMPLETE.md)** - Bayesian methodology
- **[PROJECT_STATUS.md](../PROJECT_STATUS.md)** - Current project status
- **[docs/database/schema.md](database/schema.md)** - Full database schema

---

## 📝 Changelog

**November 1, 2025:**
- Consolidated 4 PROPS docs into single comprehensive guide
- Added troubleshooting section
- Updated status to reflect October 2025 progress

**October 12, 2025:**
- Core infrastructure complete
- 63,055 player-game records generated
- All 8 prop type models trained

**October 11, 2025:**
- Data layer validation complete
- Schema mismatches resolved

---

**This guide consolidates:**
- `PROPS_BETTING_SYSTEM.md` (500 lines) - Architecture and technical details
- `PROPS_QUICK_START.md` (314 lines) - Quick start guide
- `PROPS_IMPLEMENTATION_SUMMARY.md` (301 lines) - What was built
- `PROPS_SYSTEM_STATUS.md` (255 lines) - Current status

**Maintained by:** NFL Analytics Team
**Last Updated:** November 1, 2025
