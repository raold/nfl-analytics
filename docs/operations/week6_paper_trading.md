# Week 6 (2025) Paper Trading Guide

**Mission**: Trial the production betting system with virtual money ($10k paper bankroll) during NFL Week 6 to validate the system before going live with real money in Week 7.

## 📋 Overview

### What is Paper Trading?

Paper trading means placing **virtual bets** with **fake money** to test the system without financial risk. All bets are tracked in the database with `is_paper_trade = TRUE` flag, keeping them separate from future real-money bets.

### Goals for Week 6

1. **Validate the workflow** - Ensure the betting system, data pipeline, and monitoring work smoothly
2. **Test Positron interface** - Try Positron IDE to see if it's a good fit for this workflow
3. **Assess model performance** - See how models perform on current-season data
4. **Build confidence** - Get comfortable with the process before risking real money
5. **Make go/no-go decision** - Decide whether to proceed to live betting in Week 7

### Success Criteria

After Week 6, the system should:
- Generate sensible betting recommendations
- Track performance accurately
- Show positive expected value (ROI > 0%)
- Have win rate ≥ 53% (above breakeven at -110 odds)
- Run without major technical issues

## 🛠️ Setup (Before Week 6 Games)

### 1. Database Setup

First, apply the production schema to create the database tables:

```bash
# Option 1: Using pgAdmin (GUI)
# 1. Open pgAdmin
# 2. Connect to devdb01
# 3. Open Query Tool
# 4. Open sql/production_schema.sql
# 5. Execute

# Option 2: Using psql (CLI)
psql -h localhost -p 5544 -U dro -d devdb01 -f sql/production_schema.sql

# Option 3: Using Python
python -c "
from sqlalchemy import create_engine
engine = create_engine('postgresql://dro:sicillionbillions@localhost:5544/devdb01')
with open('sql/production_schema.sql') as f:
    with engine.connect() as conn:
        conn.execute(f.read())
        conn.commit()
"
```

### 2. Prepare Week 6 Game Data

You'll need a CSV file with Week 6 games and features. The file should be named `data/week6_games.csv` with these columns:

**Required columns**:
- `game_id`: Unique game identifier (e.g., "2025_06_KC_BUF")
- `season`: 2025
- `week`: 6
- `home_team`: Home team abbreviation (e.g., "KC")
- `away_team`: Away team abbreviation (e.g., "BUF")
- `spread_close`: Closing spread (negative = home favored)
- `total_close`: Over/under total points
- All model feature columns (see models for required features)

**How to get this data**:

1. **Scrape from The Odds API** (you have a subscription):
   ```bash
   python py/data/fetch_odds.py --week 6 --season 2025 --output data/week6_odds.csv
   ```

2. **Get team features** from nflreadr:
   ```r
   library(nflreadr)
   library(dplyr)

   # Get play-by-play through Week 5
   pbp <- load_pbp(2025) %>% filter(week <= 5)

   # Calculate team features (EPA, etc.)
   # ... (run your feature engineering pipeline)

   # Merge with Week 6 games
   write.csv(week6_data, "data/week6_games.csv")
   ```

3. **Alternatively**, use the feature pipeline:
   ```bash
   Rscript R/features/asof_features.R --season 2025 --week 6
   ```

### 3. Verify Models are Ready

Check that trained models exist:

```bash
# Check for model files
ls models/xgboost/v2_sweep/xgb_config18_season2024.json
ls models/cql/sweep/cql_config4.pth
ls models/iql/baseline_model.pth
```

If models are missing, train them on 2010-2024 data:

```bash
# XGBoost
python py/models/xgboost_gpu.py \
  --features-csv data/processed/features/asof_team_features_v2.csv \
  --start-season 2010 \
  --end-season 2024 \
  --test-seasons 2024 \
  --sweep

# CQL
python py/rl/cql_sweep.py \
  --dataset data/rl_logged_2006_2024.csv \
  --output-dir models/cql/sweep \
  --epochs 500 \
  --device cuda

# IQL
python py/rl/iql_agent.py \
  --dataset data/rl_logged_2006_2024.csv \
  --output models/iql/baseline_model.pth \
  --expectile 0.9 \
  --temperature 3.0 \
  --epochs 500 \
  --device cuda
```

## 🎯 Week 6 Workflow

### Monday/Tuesday: Generate Recommendations

Once Week 6 games and lines are available:

```bash
python py/production/paper_trade.py recommend --week 6
```

This will:
1. Load Week 6 games from `data/week6_games.csv`
2. Run majority voting ensemble (XGBoost + CQL + IQL)
3. Calculate Kelly bet sizes (quarter Kelly = 0.25)
4. Filter bets by minimum edge (2% minimum)
5. Save recommendations to `paper_trades/week6_recommendations.json`

**Review the recommendations**:

```bash
cat paper_trades/week6_recommendations.json | jq .
```

Look for:
- **Edge**: Model probability - market probability (want ≥ 2%)
- **Odds**: American odds (e.g., -110)
- **Bet amount**: Kelly-sized bet (max 5% of bankroll)
- **Uncertainty**: Model disagreement (lower is better)
- **Ensemble agreement**: How many models agree

### Tuesday-Saturday: Log Paper Bets

For each bet you want to place, log it to the database:

```bash
python py/production/paper_trade.py log \
  --game-id "2025_06_KC_BUF" \
  --week 6 \
  --season 2025 \
  --bet-type spread \
  --side home \
  --line -3.5 \
  --odds -110 \
  --stake 250 \
  --prediction 0.58
```

**TIP**: The `paper_trade.py log` command automatically adds `--paper-trade` flag, so all bets are marked as virtual.

You can also use the full monitor_performance.py command with `--paper-trade`:

```bash
python py/production/monitor_performance.py log \
  --game-id "2025_06_KC_BUF" \
  --week 6 \
  --season 2025 \
  --bet-type spread \
  --side home \
  --line -3.5 \
  --odds -110 \
  --stake 250 \
  --prediction 0.58 \
  --paper-trade
```

### Sunday-Monday: Update Results

After games finish, update each bet with the actual outcome:

```bash
python py/production/paper_trade.py update \
  --bet-id 1 \
  --result win \
  --home-score 28 \
  --away-score 24 \
  --closing-line -3.0
```

The system will:
1. Calculate payout (win/loss/push)
2. Calculate CLV (closing line value)
3. Update virtual bankroll

### Monday Night: Generate Report

After all Week 6 games are complete:

```bash
python py/production/paper_trade.py report
```

This shows:
- Total bets placed and settled
- Win/loss record and win rate
- Total staked, payout, ROI
- Final bankroll ($10,000 starting)
- **Go/no-go recommendation** for Week 7

## 📊 Monitoring & Analysis

### List All Paper Trades

```bash
python py/production/paper_trade.py list
```

### View in Database

```sql
-- All paper trades
SELECT * FROM bets WHERE is_paper_trade = TRUE;

-- Settled paper trades only
SELECT * FROM bets
WHERE is_paper_trade = TRUE
AND result IS NOT NULL;

-- Paper trade summary
SELECT
  COUNT(*) as n_bets,
  SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as wins,
  SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END) as losses,
  SUM(stake) as total_staked,
  SUM(payout) as total_payout,
  SUM(payout) / SUM(stake) as roi
FROM bets
WHERE is_paper_trade = TRUE
AND result IS NOT NULL;
```

### Streamlit Dashboard (Optional)

Launch the production dashboard to monitor in real-time:

```bash
streamlit run py/production/viz/production_dashboard.py
```

Filter to show only paper trades using the sidebar.

## 🧪 Testing Positron IDE

As part of this trial, we're also testing **Positron** (VSCode-based IDE for R + Python data science).

### Install Positron

Download from: https://github.com/posit-dev/positron/releases

### Open Project in Positron

```bash
# From command line
positron nfl-analytics/

# Or use File -> Open Folder in Positron
```

### Recommended Workspace Setup

Create `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": ".venv/Scripts/python.exe",
  "r.rpath.windows": "C:\\Program Files\\R\\R-4.5.1\\bin\\R.exe",
  "files.exclude": {
    "**/__pycache__": true,
    "**/.pytest_cache": true,
    "**/*.pyc": true
  },
  "python.analysis.typeCheckingMode": "basic",
  "editor.rulers": [88, 120],
  "editor.formatOnSave": true,
  "python.formatting.provider": "black"
}
```

### Things to Test

During Week 6, assess Positron for:

1. **Python console**: Does it work well for running `paper_trade.py` commands?
2. **R console**: Can you easily pull data with nflreadr?
3. **Data explorer**: How well does it visualize bets dataframe?
4. **SQL integration**: Can you query PostgreSQL directly?
5. **Notebook support**: Does it handle .ipynb files for analysis?
6. **Terminal**: Is the integrated terminal better than plain VSCode?

**Keep notes** on what you like/dislike for the Week 7 debrief.

## 🚦 Week 7 Go/No-Go Decision

After Week 6 paper trading completes, use this framework to decide:

### ✅ GREEN LIGHT (Go Live in Week 7)

Proceed with **real money** (starting small) if:
- ROI > 2% (system showing edge)
- Win rate ≥ 53% (above breakeven)
- At least 8-10 bets settled (enough sample)
- No major technical issues
- Comfortable with workflow

**Starting strategy**:
- Start with **smaller stakes** (50% of Kelly)
- Limit to **3-5 bets per week** initially
- Monitor closely for 2-3 weeks before scaling up

### 🟡 YELLOW LIGHT (Iterate)

Paper trade **another week** (Week 7 virtual) if:
- ROI between -5% and +2% (inconclusive)
- Win rate between 50-53% (borderline)
- Technical issues but fixable
- Uncomfortable with workflow
- Want more data before risking money

### 🔴 RED LIGHT (Do Not Go Live)

**Stop and reassess** if:
- ROI < -5% (system losing money)
- Win rate < 50% (clearly underperforming)
- Major technical issues
- Models clearly broken
- Not confident in system

In this case:
1. Review model predictions vs outcomes
2. Check for data quality issues
3. Validate feature engineering
4. Consider retraining models on 2025 data
5. Paper trade until issues resolved

## 📝 Week 6 Checklist

### Pre-Game (Monday-Wednesday)

- [ ] Database schema applied (`sql/production_schema.sql`)
- [ ] Week 6 game data prepared (`data/week6_games.csv`)
- [ ] Models verified (XGBoost, CQL, IQL)
- [ ] Positron IDE installed and configured
- [ ] Paper trading recommendations generated
- [ ] Recommendations reviewed for sanity
- [ ] Paper bets logged to database

### Post-Game (Sunday-Monday)

- [ ] All bet results updated
- [ ] CLV calculated for each bet
- [ ] Paper trading report generated
- [ ] Performance assessed (ROI, win rate)
- [ ] Technical issues documented (if any)
- [ ] Positron evaluation notes written
- [ ] Week 7 go/no-go decision made
- [ ] Debrief completed (see `paper_trade_debrief.md`)

## 🆘 Troubleshooting

### Issue: `data/week6_games.csv` not found

**Solution**: You need to create this file manually. See "Prepare Week 6 Game Data" section above.

### Issue: Model files missing

**Solution**: Train models using commands in "Verify Models are Ready" section.

### Issue: Database connection error

**Solution**:
```bash
# Check PostgreSQL is running
docker ps | grep postgres

# If not running, start it
docker start nfl-postgres  # or your container name

# Test connection
psql -h localhost -p 5544 -U dro -d devdb01 -c "SELECT 1"
```

### Issue: Paper trade bets mixed with real bets

**Solution**: This should never happen if you use `paper_trade.py`. Always use:
```bash
python py/production/paper_trade.py log ...  # Correct (auto-adds --paper-trade)
```

NOT:
```bash
python py/production/monitor_performance.py log ...  # Wrong (missing --paper-trade)
```

To verify bets are marked correctly:
```sql
SELECT bet_id, game_id, is_paper_trade FROM bets ORDER BY bet_id DESC LIMIT 10;
```

### Issue: Recommendations look weird

**Solution**:
1. Check Week 6 data quality
2. Verify features match what models expect
3. Look at model predictions directly
4. Compare to market odds for sanity check

## 📖 Additional Resources

- [Production schema](../../sql/production_schema.sql) - Database structure
- [Majority betting system](../../py/production/majority_betting_system.py) - Core betting logic
- [Performance monitor](../../py/production/monitor_performance.py) - Bet tracking
- [Paper trade orchestrator](../../py/production/paper_trade.py) - Week 6 wrapper
- [Virginia sportsbooks guide](virginia_sportsbooks.md) - When you go live

## 📅 Timeline

**Week 6 Schedule** (typical NFL week):

- **Monday**: Lines released, prepare `week6_games.csv`
- **Tuesday**: Generate recommendations, review, log virtual bets
- **Wednesday**: Odds movement, potentially add more bets
- **Thursday**: TNF game (if any bets)
- **Friday**: Final odds check
- **Saturday**: College football (no NFL)
- **Sunday**: Most games (1pm, 4pm, SNF)
- **Monday**: MNF game, update all results, generate report, decide on Week 7

**After Week 6**:

- Write debrief (see `paper_trade_debrief.md`)
- Make Week 7 decision (go/no-go/iterate)
- If going live: Set up real sportsbook accounts
- If iterating: Prepare for Week 7 paper trading
- If stopping: Review and fix issues before resuming

---

**Remember**: This is a **trial run**. The goal is to learn and validate, not to be perfect. Take notes on what works and what doesn't. You'll refine the system based on what you learn this week.

**Good luck with Week 6 paper trading!** 🏈
