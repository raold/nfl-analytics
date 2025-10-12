# NFL Analytics Monitoring Dashboard

Real-time monitoring dashboard for NFL prediction performance, CLV analysis, and model metrics.

## Features

- **Summary Metrics**: Total games, win rate, MAE, and ATS accuracy at a glance
- **Performance Over Time**: Win rate and MAE trends across seasons and weeks
- **ATS Accuracy**: Against-the-spread performance vs breakeven threshold (52.4%)
- **Error Distribution**: Histogram of prediction errors
- **Calibration Curve**: Model calibration vs perfect calibration
- **Model Comparison**: Compare different model versions (xgb_v3_backtest, xgb_v2_backtest)
- **Recent Predictions**: Table of the latest 20 predictions with outcomes
- **Filters**: Season and model version filters in sidebar
- **Auto-refresh**: Data refreshes every 5 minutes

## Running with Docker

### Start the dashboard

From the `containers/` directory:

```bash
cd containers
docker-compose up dashboard
```

The dashboard will be available at: http://localhost:8501

### Start the entire stack (worker + dashboard)

```bash
docker-compose up
```

### Rebuild the dashboard container

```bash
docker-compose build dashboard
docker-compose up dashboard
```

### Stop the dashboard

```bash
docker-compose down dashboard
```

## Running Locally (without Docker)

### Install dependencies

```bash
cd dashboard
pip install -r requirements.txt
```

### Run the dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at http://localhost:8501

## Environment Variables

The dashboard uses the following environment variables for database connection:

- `DB_HOST`: Database host (default: localhost)
- `DB_PORT`: Database port (default: 5544)
- `DB_NAME`: Database name (default: devdb01)
- `DB_USER`: Database user (default: dro)
- `DB_PASSWORD`: Database password (default: sicillionbillions)

When running in Docker, these are automatically configured in `containers/docker-compose.yml`.

## Database Requirements

The dashboard requires access to the following database tables:

- `predictions.retrospectives`: Post-game prediction evaluation data
- `predictions.game_predictions`: Pre-game predictions with model versions
- `games`: Game information (scores, spread, totals)

## Metrics Explained

### Win Rate
Percentage of games where the predicted winner matches the actual winner. Must exceed 52.4% to break even on -110 odds.

### MAE (Mean Absolute Error)
Average absolute difference between predicted and actual margins. Lower is better.

### ATS Accuracy
Against-the-spread accuracy. Percentage of games where the prediction beats the closing spread. Must exceed 52.4% for profitability.

### Brier Score
Measure of prediction calibration. Lower is better (perfect = 0, random = 0.25).

### CLV (Closing Line Value)
Average difference between model-implied edge and market-implied edge. Positive CLV indicates the model predicts better than the closing line.

### Calibration
How well predicted probabilities match actual frequencies. Perfect calibration means a prediction of 60% wins 60% of the time.

## Performance

- Data is cached for 5 minutes to reduce database load
- Queries are optimized with indexes on key columns
- Dashboard auto-refreshes data when filters change

## Troubleshooting

### Dashboard won't connect to database

Check that:
1. PostgreSQL is running on the host machine (port 5544)
2. Database credentials are correct
3. Docker container can access host via `host.docker.internal`

### Dashboard shows no data

Verify that:
1. Backtest has been run and populated `predictions.retrospectives`
2. Model predictions exist in `predictions.game_predictions`
3. Database filters (season, model version) are not too restrictive

### Dashboard is slow

Try:
1. Reducing the query limit (currently 1000 games)
2. Adding database indexes on frequently queried columns
3. Increasing the cache TTL (currently 300 seconds)
