# Game-Level Predictions Implementation Plan

## Current State Analysis

### ✅ What You Already Have

**1. Sportsbook Data (in `games` table):**
```sql
-- Moneyline odds
home_moneyline REAL        -- e.g., -150 (favorite), +130 (underdog)
away_moneyline REAL

-- Spread
spread_close REAL          -- e.g., -3.5 (home favored by 3.5)
home_spread_odds REAL      -- Usually -110
away_spread_odds REAL      -- Usually -110

-- Totals (Over/Under)
total_close REAL           -- e.g., 47.5
over_odds REAL             -- Usually -110
under_odds REAL            -- Usually -110
```

**2. Player Prop Predictions (in `ml.predictions` table):**
- Player-level predictions (passing yards only)
- 486 predictions for 2024 season (261 games, 72 QBs)
- Stores: prediction, actual, MAE, status, P&L

**3. Ensemble v6.1 Model:**
- 5 components (State-Space, Hierarchical, XGBoost, Informative Priors, Hybrid)
- Currently predicts QB passing yards
- FastAPI endpoint: `/predict/game`

---

## ❌ What's Missing for Game-Level Predictions

### **Gap 1: Game-Level Prediction Models**

**You need models that predict:**
1. **Spread** (predicted margin: home_score - away_score)
2. **Total** (predicted combined score: home_score + away_score)
3. **Moneyline** (win probability → implied odds)

**Current:** You only have QB passing yards predictions (player props)

---

### **Gap 2: Database Schema for Game Predictions**

**You need a new table:** `ml.game_predictions`

```sql
CREATE TABLE ml.game_predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    game_id VARCHAR(20) NOT NULL REFERENCES games(game_id),
    model_id INTEGER NOT NULL REFERENCES ml.model_registry(model_id),
    predicted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Game outcome predictions
    predicted_home_score NUMERIC(5,2),
    predicted_away_score NUMERIC(5,2),
    predicted_total NUMERIC(5,2),          -- home + away
    predicted_spread NUMERIC(5,2),         -- home - away
    predicted_home_win_prob NUMERIC(5,4),  -- 0.0 to 1.0

    -- Uncertainty quantification
    spread_std NUMERIC(5,2),
    total_std NUMERIC(5,2),
    win_prob_std NUMERIC(5,4),

    -- Sportsbook comparison (snapshot at prediction time)
    sportsbook_spread REAL,
    sportsbook_total REAL,
    sportsbook_home_ml REAL,
    sportsbook_away_ml REAL,

    -- Edge calculation
    spread_edge NUMERIC(8,4),              -- model vs sportsbook spread
    total_edge NUMERIC(8,4),               -- model vs sportsbook total
    ml_edge NUMERIC(8,4),                  -- model vs implied moneyline prob

    -- Actual results (filled after game completes)
    actual_home_score INTEGER,
    actual_away_score INTEGER,
    actual_total INTEGER,
    actual_spread NUMERIC(5,2),
    actual_home_win BOOLEAN,

    -- Performance metrics
    spread_mae NUMERIC(5,2),               -- |predicted - actual| spread
    total_mae NUMERIC(5,2),                -- |predicted - actual| total
    brier_score NUMERIC(8,4),              -- (win_prob - actual)^2
    log_loss NUMERIC(8,4),

    -- Metadata
    prediction_confidence VARCHAR(20),     -- 'high', 'medium', 'low'
    notes TEXT,

    UNIQUE(game_id, model_id, predicted_at)
);

CREATE INDEX idx_game_predictions_game_id ON ml.game_predictions(game_id);
CREATE INDEX idx_game_predictions_season_week ON ml.game_predictions USING btree
    ((substring(game_id, 1, 4)::int), (substring(game_id, 6, 2)::int));
CREATE INDEX idx_game_predictions_predicted_at ON ml.game_predictions(predicted_at DESC);
```

**Why this structure:**
- Stores predictions AND sportsbook lines at prediction time (for edge calculation)
- Links to existing `games` table and `ml.model_registry`
- Captures uncertainty (std dev)
- Allows MAE calculation after game completes
- Supports multiple predictions per game (from different models)

---

### **Gap 3: API Endpoints**

**You need new FastAPI routes:**

```python
# py/api/main.py

@app.post("/predict/game/full")
async def predict_game_full(request: GamePredictionRequest):
    """
    Predict spread, total, and moneyline for a single game.

    Returns:
    {
        "game_id": "2025_12_KC_LAC",
        "predicted_home_score": 28.3,
        "predicted_away_score": 24.1,
        "predicted_total": 52.4,
        "predicted_spread": 4.2,  # home favored by 4.2
        "predicted_home_win_prob": 0.657,
        "spread_std": 7.8,
        "total_std": 9.2,
        "sportsbook_spread": -3.5,
        "sportsbook_total": 47.5,
        "spread_edge": 0.7,  # model says home by 4.2, book says 3.5
        "total_edge": 4.9    # model says 52.4, book says 47.5
    }
    """
    # Implementation:
    # 1. Fetch game context (teams, QBs, rest, weather, etc.)
    # 2. Run ensemble models for home/away scores
    # 3. Calculate derived metrics (total, spread, win prob)
    # 4. Fetch sportsbook lines from games table
    # 5. Calculate edge
    # 6. Store in ml.game_predictions
    # 7. Return prediction

@app.post("/predict/week")
async def predict_week(season: int, week: int):
    """
    Generate predictions for all games in a given week.
    Returns list of game predictions.
    """
    # Batch prediction for all games in week

@app.get("/predictions/game/{game_id}")
async def get_game_prediction(game_id: str):
    """
    Retrieve stored predictions for a game (with actuals if available).
    """
    # Query ml.game_predictions table
```

---

### **Gap 4: Prediction Logic**

**You need to build game outcome models.** Options:

**Option A: Aggregate Player Props** (Quick & Easy)
```python
# Use your existing QB passing yards predictions
home_qb_yards = predict_qb_yards(home_qb_id)
away_qb_yards = predict_qb_yards(away_qb_id)

# Regression: passing yards → team score
# Historical correlation: ~0.65 between QB yards and team points
# Simple model: Score ≈ 10 + (QB_yards / 20)
predicted_home_score = 10 + (home_qb_yards / 20)
predicted_away_score = 10 + (away_qb_yards / 20)

# Derive spread and total
predicted_spread = predicted_home_score - predicted_away_score
predicted_total = predicted_home_score + predicted_away_score
```

**Option B: Train Game Outcome Models** (Better)
```python
# Train new models for:
# 1. Home team score (given: home team stats, away def stats, rest, weather)
# 2. Away team score (given: away team stats, home def stats, rest, weather)

# Features:
# - Team offensive EPA/play (last 4 games)
# - Team defensive EPA/play (last 4 games)
# - Home/away splits
# - Rest days
# - QB experience
# - Injuries
# - Weather (temp, wind, dome)
# - Division game flag
# - Historical H2H

# Target: actual scores from games table
```

**Option C: Hybrid** (Recommended)
```python
# Combine both:
# 1. Use existing player props (QB yards, rushing yards, etc.)
# 2. Use team-level features (defensive EPA, rest, etc.)
# 3. Ensemble multiple approaches (regression, XGBoost, Bayesian)

# Example ensemble:
# - Component 1: Player prop aggregation (30%)
# - Component 2: Team EPA regression (25%)
# - Component 3: XGBoost on historical games (25%)
# - Component 4: Bayesian hierarchical model (20%)
```

---

### **Gap 5: Dashboard Display**

**You need UI components in `dashboard/app.py`:**

```python
def render_game_predictions_comparison(game_id):
    """
    Display sportsbook lines vs model predictions side-by-side.

    Layout:
    ┌─────────────────────────────────────────────────┐
    │ Team Matchup: KC @ LAC                          │
    ├─────────────────────────────────────────────────┤
    │                 Sportsbook    Our Model   Edge  │
    │ Spread (LAC)      -3.5          -4.2     +0.7  │
    │ Total            47.5          52.4     +4.9   │
    │ Moneyline (LAC)  -150          -180     +30    │
    │                                                 │
    │ Predicted Score: KC 24.1  LAC 28.3             │
    │ Win Probability: LAC 65.7%                     │
    └─────────────────────────────────────────────────┘
    """

    # Fetch prediction from ml.game_predictions
    prediction = fetch_game_prediction(game_id)

    # Fetch sportsbook from games table
    sportsbook = fetch_sportsbook_lines(game_id)

    # Display comparison table with color coding:
    # - Green: Model has edge (prediction differs favorably)
    # - Red: No edge
    # - Yellow: Marginal edge
```

**Enhanced dashboard page:**

```python
# In main() function, Predictions page:

st.header("🎯 Game Predictions vs Sportsbook")

for game in current_week_games:
    with st.expander(f"{game['away_team']} @ {game['home_team']}"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Sportsbook")
            st.metric("Spread", f"{game['spread_close']:+.1f}")
            st.metric("Total", f"{game['total_close']:.1f}")
            st.metric("Moneyline", f"{game['home_moneyline']:+.0f}")

        with col2:
            st.subheader("Our Model")
            prediction = get_game_prediction(game['game_id'])
            st.metric("Spread", f"{prediction['predicted_spread']:+.1f}")
            st.metric("Total", f"{prediction['predicted_total']:.1f}")
            st.metric("Win Prob", f"{prediction['predicted_home_win_prob']:.1%}")

        with col3:
            st.subheader("Edge")
            spread_edge = prediction['spread_edge']
            total_edge = prediction['total_edge']
            st.metric("Spread Edge", f"{spread_edge:+.1f}",
                     delta_color="normal" if abs(spread_edge) > 2 else "off")
            st.metric("Total Edge", f"{total_edge:+.1f}",
                     delta_color="normal" if abs(total_edge) > 3 else "off")

            # Color-coded recommendation
            if abs(spread_edge) > 2.5 or abs(total_edge) > 4:
                st.success("✅ STRONG EDGE - Consider bet")
            elif abs(spread_edge) > 1.5 or abs(total_edge) > 2.5:
                st.warning("⚠️ MODERATE EDGE")
            else:
                st.info("ℹ️ No significant edge")
```

---

## 📋 Implementation Checklist

### **Phase 1: Database Schema** (1 hour)
- [ ] Create migration: `db/migrations/048_create_game_predictions.sql`
- [ ] Add `ml.game_predictions` table
- [ ] Add indexes
- [ ] Add foreign key constraints
- [ ] Run migration: `psql < db/migrations/048_create_game_predictions.sql`

### **Phase 2: Prediction Models** (4-8 hours)
- [ ] **Quick Path:** Build score prediction from QB yards aggregation
  - [ ] Create `py/models/game_outcomes/score_from_props.py`
  - [ ] Historical regression: QB yards → team score
  - [ ] Add uncertainty estimation
- [ ] **Better Path:** Train game outcome models
  - [ ] Feature engineering: team EPA, rest, weather, injuries
  - [ ] Train home/away score models separately
  - [ ] Cross-validate on 2022-2024 games
  - [ ] Save models to `models/game_outcomes/`

### **Phase 3: API Endpoints** (2-3 hours)
- [ ] Add `/predict/game/full` endpoint in `py/api/main.py`
- [ ] Add `/predict/week` batch endpoint
- [ ] Add `/predictions/game/{game_id}` retrieval endpoint
- [ ] Add logic to store predictions in `ml.game_predictions`
- [ ] Add logic to update actuals after game completes

### **Phase 4: Dashboard UI** (2-3 hours)
- [ ] Add `render_game_predictions_comparison()` function
- [ ] Update Predictions page to show sportsbook vs model
- [ ] Add edge calculation and color coding
- [ ] Add filtering: show only games with >2pt spread edge or >3pt total edge

### **Phase 5: Automated Prediction Generation** (1-2 hours)
- [ ] Create cron job: `scripts/generate_weekly_predictions.py`
- [ ] Schedule to run Wednesdays at 9am (before lines move)
- [ ] Batch predict all games for upcoming week
- [ ] Store in database
- [ ] Send notification if high-edge games found

### **Phase 6: Performance Tracking** (1-2 hours)
- [ ] Create `scripts/update_game_prediction_actuals.py`
- [ ] Run Monday morning after games complete
- [ ] Update `actual_*` columns in `ml.game_predictions`
- [ ] Calculate MAE, Brier score, log loss
- [ ] Generate weekly performance report

---

## 🎯 Minimal Viable Product (MVP)

**If you want to get this working TODAY:**

### **1-Hour MVP:**

```sql
-- Create simplified table
CREATE TABLE ml.game_predictions (
    game_id VARCHAR(20) PRIMARY KEY,
    predicted_spread NUMERIC(5,2),
    predicted_total NUMERIC(5,2),
    predicted_home_win_prob NUMERIC(5,4),
    sportsbook_spread REAL,
    sportsbook_total REAL,
    actual_spread NUMERIC(5,2),
    actual_total INTEGER,
    predicted_at TIMESTAMPTZ DEFAULT NOW()
);
```

```python
# py/api/main.py - Add quick endpoint
@app.post("/predict/game/simple")
async def predict_game_simple(game_id: str):
    # Use existing QB yards predictions
    home_qb_pred = predict_qb_yards(home_qb_id)
    away_qb_pred = predict_qb_yards(away_qb_id)

    # Simple regression
    home_score = 10 + (home_qb_pred / 20)
    away_score = 10 + (away_qb_pred / 20)

    spread = home_score - away_score
    total = home_score + away_score
    win_prob = 1 / (1 + exp(-spread / 7))  # Logistic

    # Store and return
    return {
        "predicted_spread": spread,
        "predicted_total": total,
        "predicted_home_win_prob": win_prob
    }
```

```python
# dashboard/app.py - Add to game cards
prediction = get_game_prediction_simple(game['game_id'])

st.markdown(f"""
**Sportsbook:** Spread {game['spread_close']:+.1f} | Total {game['total_close']:.1f}
**Our Model:** Spread {prediction['predicted_spread']:+.1f} | Total {prediction['predicted_total']:.1f}
**Edge:** Spread {prediction['predicted_spread'] - game['spread_close']:+.1f} | Total {prediction['predicted_total'] - game['total_close']:+.1f}
""")
```

---

## 📊 Expected Outcomes

**After Full Implementation:**

1. **Dashboard shows:**
   - Sportsbook lines vs model predictions side-by-side
   - Calculated edge for each bet type
   - Color-coded recommendations
   - Historical accuracy (MAE on spread/total predictions)

2. **Database stores:**
   - All game predictions with timestamps
   - Sportsbook snapshots at prediction time
   - Actual results after games complete
   - Performance metrics (MAE, Brier score)

3. **Weekly workflow:**
   - Wednesday: Generate predictions for upcoming week
   - Sunday/Monday: Games complete, actuals stored
   - Tuesday: Performance report generated
   - Continuous improvement: retrain models with new data

4. **Performance tracking:**
   ```sql
   -- Example MAE query
   SELECT
       AVG(spread_mae) as avg_spread_mae,
       AVG(total_mae) as avg_total_mae,
       AVG(brier_score) as avg_brier_score,
       COUNT(*) as games_predicted
   FROM ml.game_predictions
   WHERE predicted_at > NOW() - INTERVAL '30 days'
     AND actual_spread IS NOT NULL;
   ```

---

## 🚀 Next Steps

**Recommended order:**

1. **Start with MVP** (1-2 hours) - Get something working today
2. **Validate predictions** (1 week) - Run in parallel with real games, track accuracy
3. **Build full schema** (if MVP works) - Add proper uncertainty, edge calculation
4. **Train dedicated models** (if needed) - Replace simple regression with ML models
5. **Automate** - Cron jobs for weekly predictions and actuals updates

**Question for you:**
- Do you want to start with the **1-hour MVP** (simple QB yards → score regression)?
- Or go straight to **full implementation** with proper models and schema?
