# Game Predictions Backtest - Usage Guide

**Last Updated:** November 21, 2025
**Script:** `scripts/backtest_game_predictions.py`
**Database Table:** `ml.game_predictions`

---

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Run backtest (2023-2025 by default)
python scripts/backtest_game_predictions.py

# Custom parameters
python scripts/backtest_game_predictions.py \
    --start-season 2024 \
    --end-season 2025 \
    --ensemble-version "my_model_v1.0" \
    --output-dir my_results
```

---

## CLI Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--start-season` | int | 2023 | First season to backtest (inclusive) |
| `--end-season` | int | 2025 | Last season to backtest (inclusive) |
| `--ensemble-version` | str | baseline_v1.0 | Version tag for predictions |
| `--output-dir` | str | results | Output directory for reports |

---

## Output Files

### 1. Backtest Report (Markdown)

**Location:** `{output_dir}/backtest_report_YYYY_MM_DD_HHMMSS.md`

**Contents:**
- Overall metrics table (Spread MAE, Total MAE, Brier, Cover Rate)
- Season-by-season breakdown
- Edge analysis (ROI simulation)
- Interpretation guide
- Next steps recommendations

**Example:**
```markdown
## Overall Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Spread MAE | 10.69 pts | <10 pts | ❌ FAIL |
| Total MAE | 10.98 pts | <12 pts | ✅ PASS |
| Cover Rate | 60.1% | >52% | ✅ PASS |
```

### 2. Metrics JSON

**Location:** `{output_dir}/backtest_metrics_YYYY_MM_DD_HHMMSS.json`

**Structure:**
```json
{
  "overall": {
    "spread_mae": 10.69,
    "total_mae": 10.98,
    "brier_score": 0.2352,
    "n_games": 680,
    "cover_rate": 0.601,
    "ou_rate": 0.50
  },
  "edge_analysis": {
    "n_edge_opportunities": 423,
    "edge_cover_rate": 0.539,
    "estimated_roi": 2.9
  },
  "by_season": [
    {
      "season": 2023,
      "n_games": 272,
      "spread_mae": 10.87,
      "cover_rate": 0.592
    }
  ]
}
```

### 3. Log File

**Location:** `backtest_game_predictions.log`

**Contents:**
- Initialization messages
- Game-by-game processing logs
- Error messages
- Database commit confirmations

---

## Database Integration

### Predictions Table

**Table:** `ml.game_predictions`

**Key Columns:**
- `game_id` - Game identifier (e.g., "2023_01_ARI_WAS")
- `season`, `week` - Temporal identifiers
- `predicted_spread`, `predicted_total` - Model predictions
- `actual_spread`, `actual_total` - Actual outcomes
- `abs_spread_error`, `abs_total_error` - Absolute errors
- `brier_score` - Win probability calibration metric
- `spread_edge`, `total_edge` - Edge vs sportsbook
- `ensemble_version` - Model version tag

### Verification Queries

**Count predictions:**
```sql
SELECT COUNT(*) FROM ml.game_predictions
WHERE ensemble_version = 'baseline_v1.0';
```

**Performance by season:**
```sql
SELECT
    season,
    COUNT(*) as games,
    ROUND(AVG(abs_spread_error), 2) as spread_mae,
    ROUND(AVG(abs_total_error), 2) as total_mae,
    ROUND(AVG(brier_score), 4) as brier
FROM ml.game_predictions
WHERE ensemble_version = 'baseline_v1.0'
GROUP BY season
ORDER BY season;
```

**Edge analysis:**
```sql
SELECT
    CASE WHEN spread_edge > 2 THEN 'Home Edge >2'
         WHEN spread_edge < -2 THEN 'Away Edge >2'
         ELSE 'No Edge'
    END as edge_category,
    COUNT(*) as games,
    ROUND(AVG(CASE
        WHEN (predicted_spread > 0 AND actual_spread > 0) OR
             (predicted_spread < 0 AND actual_spread < 0)
        THEN 1.0 ELSE 0.0
    END) * 100, 1) as cover_rate_pct
FROM ml.game_predictions
WHERE ensemble_version = 'baseline_v1.0' AND spread_edge IS NOT NULL
GROUP BY edge_category;
```

---

## Performance Metrics

### Spread MAE (Mean Absolute Error)

**Definition:** Average absolute difference between predicted and actual point spread.

**Formula:** `MAE = (1/n) * Σ |predicted_spread - actual_spread|`

**Target:** <10 points

**Interpretation:**
- <8 pts: Excellent
- 8-10 pts: Good
- 10-12 pts: Average
- >12 pts: Poor

### Total MAE

**Definition:** Average absolute difference between predicted and actual total points.

**Formula:** `MAE = (1/n) * Σ |predicted_total - actual_total|`

**Target:** <12 points

**Interpretation:**
- <10 pts: Excellent
- 10-12 pts: Good
- 12-14 pts: Average
- >14 pts: Poor

### Brier Score

**Definition:** Measures calibration of win probabilities.

**Formula:** `Brier = (1/n) * Σ (predicted_win_prob - actual_win)^2`

**Target:** <0.25

**Interpretation:**
- <0.20: Excellent calibration
- 0.20-0.25: Good calibration
- 0.25-0.30: Moderate calibration
- >0.30: Poor calibration

### Cover Rate

**Definition:** Percentage of games where model correctly predicts spread direction.

**Formula:** `Cover Rate = Correct Direction Predictions / Total Games`

**Target:** >52% (breakeven after -110 vig)

**Interpretation:**
- >55%: Strong edge
- 52-55%: Profitable
- 50-52%: Marginal
- <50%: Unprofitable

### ROI (Return on Investment)

**Definition:** Estimated profit per dollar wagered (assuming -110 odds).

**Formula:** `ROI = (Wins * 100 - Losses * 110) / (Total Games * 110)`

**Target:** >5%

**Interpretation:**
- >7%: Excellent
- 5-7%: Good
- 3-5%: Marginal
- <3%: Unprofitable

---

## Common Use Cases

### 1. Test New Model

```bash
# Create predictions with new model
python scripts/backtest_game_predictions.py \
    --ensemble-version "my_new_model_v2.0"

# Compare to baseline
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "
SELECT
    ensemble_version,
    COUNT(*) as games,
    ROUND(AVG(abs_spread_error), 2) as spread_mae,
    ROUND(AVG(abs_total_error), 2) as total_mae
FROM ml.game_predictions
WHERE ensemble_version IN ('baseline_v1.0', 'my_new_model_v2.0')
GROUP BY ensemble_version;
"
```

### 2. Single Season Analysis

```bash
# Backtest 2024 only
python scripts/backtest_game_predictions.py \
    --start-season 2024 \
    --end-season 2024 \
    --ensemble-version "2024_only"
```

### 3. Recent Performance

```bash
# Last 2 weeks (partial 2025 season)
python scripts/backtest_game_predictions.py \
    --start-season 2025 \
    --end-season 2025 \
    --ensemble-version "recent_2025"
```

---

## Troubleshooting

### Issue: ModuleNotFoundError

**Error:** `ModuleNotFoundError: No module named 'numpy'`

**Fix:** Activate virtual environment
```bash
source .venv/bin/activate
python scripts/backtest_game_predictions.py
```

### Issue: Database Connection Failed

**Error:** `psycopg2.OperationalError: could not connect to server`

**Fix:** Check Docker services
```bash
docker ps | grep postgres
# If not running:
docker-compose up -d postgres
```

### Issue: Empty Results

**Error:** `KeyError: 'abs_spread_error'`

**Fix:** Check log file for prediction errors
```bash
tail -n 50 backtest_game_predictions.log
```

### Issue: Transaction Aborted

**Error:** `current transaction is aborted, commands ignored until end of transaction block`

**Fix:** Script now automatically rolls back failed transactions (v1.0+)

---

## Performance Benchmarks

| Metric | Baseline v1.0 | Ensemble v6.1 (Expected) |
|--------|---------------|--------------------------|
| **Spread MAE** | 10.69 pts | 7-9 pts |
| **Total MAE** | 10.98 pts | 9-11 pts |
| **Cover Rate** | 60.1% | 62-65% |
| **ROI** | +2.9% | +5-7% |
| **Processing Time** | 1.3s (680 games) | 3-5s (includes ensemble inference) |

---

## Next Steps

### 1. Replace SimpleGamePredictor

**File:** `scripts/backtest_game_predictions.py`

**Change:**
```python
# Line 464: Replace SimpleGamePredictor
from py.ensemble.enhanced_ensemble_v6_production import EnhancedEnsemble5Way

self.predictor = EnhancedEnsemble5Way()  # Instead of SimpleGamePredictor
```

### 2. Add Calibration Plotting

**Create:** `scripts/plot_calibration.py`

**Logic:**
- Bin predicted probabilities (0-10%, 10-20%, ..., 90-100%)
- Calculate actual win rate in each bin
- Plot predicted vs actual (should follow diagonal)

### 3. Integrate with Live API

**Endpoint:** `POST /predict/week`

**Logic:**
- Generate predictions for upcoming week
- Store in ml.game_predictions with status='pending'
- Update to status='completed' after game finishes
- Calculate errors and update metrics

---

**Author:** Claude Code (Sonnet 4.5)
**Date:** November 21, 2025
**Version:** 1.0.0
**Related:** GAME_PREDICTIONS_BACKTEST_COMPLETE_NOV21.md, DATABASE_QUICK_REFERENCE.md
