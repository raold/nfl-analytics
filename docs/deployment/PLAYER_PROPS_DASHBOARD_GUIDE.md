# Player Props Dashboard Guide

**Location**: `dashboard/pages/player_props.py`
**URL**: http://localhost:8501/player_props (when Streamlit is running)

---

## Overview

The Player Props Dashboard displays **QB passing yards predictions vs sportsbook lines** with real-time edge analysis. It integrates:

1. **Our predictions** from `ml.predictions` (Ensemble v6.1)
2. **Sportsbook odds** from `prop_lines_history` (The Odds API)
3. **Edge calculation** (our prediction - sportsbook line)
4. **Line movement tracking** (historical odds snapshots)

---

## Features

### 1. Summary Metrics

**Top Row Metrics**:
- **Total Props**: Number of QB props with odds available
- **Avg Edge**: Average difference between our prediction and sportsbook line
- **Props w/ 10+ Edge**: Number of props with at least 10 yards edge
- **Avg Book Hold**: Average bookmaker profit margin (vig)

### 2. Edge Distribution

**Histogram Chart**: Shows distribution of edge across all props
- Positive edge → Our prediction is OVER the sportsbook line
- Negative edge → Our prediction is UNDER the sportsbook line
- Large edges (±15+ yards) indicate potential value bets

### 3. Player Props Table

**Expandable Cards** for each QB with:
- **Game Info**: Matchup, kickoff time, actual result (if completed)
- **Our Prediction**: Mean, std dev, 80% CI, model version
- **Sportsbook**: Line, over/under odds, hold percentage, edge
- **Line Movement Chart**: Historical line changes over last 7 days

### 4. Best Edges Summary

**Two Columns**:
- **Biggest OVER Edges**: Our prediction significantly above sportsbook line
- **Biggest UNDER Edges**: Our prediction significantly below sportsbook line

---

## Filters (Sidebar)

### Season/Week Selection
- **Season**: Select 2024, 2023, or 2022
- **Week**: Slider from 1-18

### Edge Filter
- **Min Edge**: Only show props with at least X yards edge
- Default: 10 yards (filters out small edges)
- Range: 0-50 yards

### Bookmaker Filter
- Select which sportsbooks to display
- Default: DraftKings + FanDuel
- Options: DraftKings, FanDuel, BetMGM, Caesars, PointsBet

### Refresh Data
- Clears cache and reloads from database
- Use after new predictions or odds are fetched

---

## How to Interpret Edge

**Edge** = Our Prediction - Sportsbook Line

**Examples**:

1. **Positive Edge (+15 yards)**
   - Our prediction: 285 yards
   - Sportsbook line: 270 yards
   - **Interpretation**: We think QB will significantly EXCEED the line
   - **Potential play**: OVER 270 yards

2. **Negative Edge (-12 yards)**
   - Our prediction: 248 yards
   - Sportsbook line: 260 yards
   - **Interpretation**: We think QB will fall SHORT of the line
   - **Potential play**: UNDER 260 yards

3. **Small Edge (±5 yards)**
   - Our prediction: 272 yards
   - Sportsbook line: 267 yards
   - **Interpretation**: Close to market consensus, no strong edge
   - **Recommendation**: Pass or small bet only

---

## Line Movement Tracking

The dashboard shows **7-day line movement history** for each QB:

**Key Patterns**:

1. **Line moving TOWARD our prediction**
   - Market agreeing with our model
   - Sharp money may be following similar logic
   - Good sign for our edge

2. **Line moving AWAY from our prediction**
   - Market disagrees with our model
   - Injury news, weather, or other information
   - Re-evaluate our edge

3. **Stable line**
   - Market has settled on a price
   - Less likely to have new information
   - Edge may persist

**Chart Elements**:
- Colored lines: Different sportsbooks
- Green dashed line: Our prediction (horizontal)
- Convergence/divergence visible at a glance

---

## Workflow

### Daily Routine (10-15 minutes)

**Morning (10am ET)** - After first odds fetch:
1. Open dashboard: http://localhost:8501/player_props
2. Select current week
3. Review edge distribution chart
4. Check "Best Edges" summary
5. Expand top 5-10 props with largest edges
6. Compare our prediction vs line movement
7. Note any props with 15+ yards edge

**Afternoon (4pm ET)** - After second odds fetch:
1. Refresh data
2. Check for line movement on flagged props
3. Re-evaluate edge (did line move toward or away from us?)

**Evening (10pm ET)** - Before closing lines:
1. Final refresh
2. Review closing lines vs our predictions
3. Document final edges for post-game analysis

### Weekly Analysis (Wednesday morning)

After outcomes backfilled:
1. Go to main dashboard: http://localhost:8501
2. Review "📊 Summary Metrics" for actual results
3. Compare predicted vs actual for high-edge props
4. Analyze calibration (were actuals within our 80% CI?)
5. Calculate realized P&L on flagged props

---

## Data Sources

### Predictions Table (ml.predictions)
```sql
SELECT
    player_name,
    prediction_mean,
    prediction_std,
    prediction_q10,
    prediction_q90,
    ensemble_version
FROM ml.predictions
WHERE season = 2024 AND week = 12
  AND stat_type = 'passing_yards'
```

### Odds Table (prop_lines_history)
```sql
SELECT
    player_name,
    line_value,
    over_odds,
    under_odds,
    book_hold,
    bookmaker_title,
    snapshot_at
FROM prop_lines_history
WHERE prop_type = 'passing_yards'
  AND snapshot_at >= NOW() - INTERVAL '7 days'
ORDER BY snapshot_at DESC
```

### JOIN Logic
- Predictions matched to odds by `player_name`
- Uses **latest odds snapshot** per bookmaker
- Calculates edge: `prediction_mean - line_value`

---

## Troubleshooting

### Issue: "No predictions found"
**Cause**: Prediction generation hasn't run yet for this week
**Solution**:
```bash
# Generate predictions manually
python scripts/generate_weekly_predictions.py --season 2024 --week 12
```

### Issue: "No odds available"
**Cause**: Odds haven't been fetched yet
**Solution**:
```bash
# Fetch odds manually (requires API key)
export ODDS_API_KEY='your_key_here'
python scripts/fetch_odds.py
```

### Issue: No line movement charts
**Cause**: Only one odds snapshot exists
**Solution**: Wait for multiple odds fetches (3x daily schedule)

### Issue: Large edges on all props
**Cause**: Stale odds or model miscalibration
**Solution**:
1. Check odds freshness: `snapshot_at` should be recent
2. Refresh odds: `python scripts/fetch_odds.py`
3. Review model performance on recent weeks

---

## Edge Calculation Details

### Formula
```python
edge = prediction_mean - book_line

prob_over = 1.0 - norm.cdf(
    (book_line - prediction_mean) / prediction_std
)

expected_value = (
    prob_over * (payout_over) -
    (1 - prob_over) * (stake)
)
```

### Example Calculation

**Given**:
- Prediction: 275 ± 18 yards (normal distribution)
- Sportsbook line: 260.5 yards
- Over odds: -110 (American)

**Edge**: 275 - 260.5 = **+14.5 yards**

**Probability OVER**:
```
Z = (260.5 - 275) / 18 = -0.806
P(OVER) = 1 - norm.cdf(-0.806) = 0.79 = 79%
```

**Expected Value** (betting OVER):
```
Payout = 100 / 110 = 0.909 (bet $110 to win $100)
EV = 0.79 * 0.909 - 0.21 * 1.0 = 0.51 = +51%
```

**Interpretation**: Strong OVER bet with +51% expected value

---

## Best Practices

1. **Don't chase small edges** (<10 yards)
   - Noise dominates signal
   - Book hold eats into edge
   - Focus on 15+ yard edges

2. **Cross-check multiple books**
   - Different bookmakers may have different lines
   - Shop for best price
   - Beware of "soft" books with wider holds

3. **Monitor line movement**
   - Line moving toward our prediction = confirmation
   - Line moving away = new information (injury, weather)
   - Stable line = market consensus

4. **Consider context**
   - Weather: Heavy rain/wind reduces passing
   - Opponent: Strong pass defense lowers totals
   - Home/Away: QBs typically perform better at home
   - Rest: Short week (Thursday games) affects performance

5. **Track results**
   - Document all flagged props
   - Compare actual vs predicted
   - Calculate realized P&L
   - Analyze calibration over time

---

## Quick Start

```bash
# 1. Ensure predictions exist for current week
python scripts/generate_weekly_predictions.py

# 2. Fetch latest odds
export ODDS_API_KEY='your_key_here'
python scripts/fetch_odds.py

# 3. Start dashboard
streamlit run dashboard/app.py

# 4. Navigate to Player Props page
# http://localhost:8501/player_props
```

---

## Support

For issues or questions:
- Check logs: `/tmp/weekly_predictions.log`, `/tmp/odds_fetcher.log`
- Verify database: `psql postgresql://dro:sicillionbillions@localhost:5544/devdb01`
- Refresh cache: Click "🔄 Refresh Data" in sidebar

---

**Last Updated**: November 21, 2025
**Version**: 1.0 (Phase 1 Complete)
