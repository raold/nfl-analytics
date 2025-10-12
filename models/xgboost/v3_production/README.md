# v3 Production Model Deployment Guide

## Model Overview

**Version:** v3.0.0
**Created:** October 10, 2025
**Model Type:** XGBoost Binary Classifier
**Task:** NFL Game Winner Prediction (Home Team Win Probability)

## Performance Summary

### Validation Performance (2024 Season)
- **Brier Score:** 0.2050
- **Accuracy:** 67.8%
- **AUC:** 0.7417

### Multi-Season Test Performance (2022-2025)
- **Mean Brier Score:** 0.2057 (±0.0055)
- **Mean Accuracy:** 68.6%
- **Consistency:** 84% more stable than v2.1

### Comparison to Baseline (v2.1)
- **Improvement:** 2.14% better Brier score
- **Stability:** 83.9% lower variance across seasons
- **Features:** 63 features vs 11 in v2.1

## Model Architecture

### Hyperparameters (Optimized via 432-config sweep)
```python
{
    "max_depth": 2,
    "learning_rate": 0.07,
    "num_boost_round": 400,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "objective": "binary:logistic",
    "device": "cpu"
}
```

### Training Data
- **Training Period:** 2006-2023 seasons (4,176 games)
- **Validation Period:** 2024 season (245 games)
- **Total Games:** 4,421 completed games

### Feature Set (63 Features)

#### Rolling Window Differentials (10 features)
- `epa_per_play_l3_diff`, `epa_per_play_l5_diff`, `epa_per_play_l10_diff`
- `points_l3_diff`, `points_l5_diff`, `points_l10_diff`
- `success_rate_l3_diff`, `success_rate_l5_diff`
- `pass_epa_l5_diff`, `win_pct_diff`

#### Betting Features (3 features)
- `spread_close` - Closing point spread
- `total_close` - Closing over/under total
- `home_over_rate_l10` - Home team over hit rate (last 10 games)

#### Rolling Stats - Last 3 Games (6 features)
- `home_points_l3`, `away_points_l3`
- `home_points_against_l3`, `away_points_against_l3`
- `home_success_rate_l3`, `away_success_rate_l3`

#### Rolling Stats - Last 5 Games (12 features)
- EPA: `home_epa_per_play_l5`, `away_epa_per_play_l5`
- Points: `home_points_l5`, `away_points_l5`, `home_points_against_l5`, `away_points_against_l5`
- Pass/Rush EPA: `home_pass_epa_l5`, `away_pass_epa_l5`, `home_rush_epa_l5`, `away_rush_epa_l5`
- Success Rate: `home_success_rate_l5`, `away_success_rate_l5`

#### Rolling Stats - Last 10 Games (6 features)
- `home_epa_per_play_l10`, `away_epa_per_play_l10`
- `home_points_l10`, `away_points_l10`
- `home_points_against_l10`, `away_points_against_l10`

#### Season-to-Date Stats (8 features)
- `home_epa_per_play_season`, `away_epa_per_play_season`
- `home_points_season`, `away_points_season`
- `home_points_against_season`, `away_points_against_season`
- `home_success_rate_season`, `away_success_rate_season`

#### Win Rates & Records (6 features)
- `home_win_pct`, `away_win_pct`
- `home_wins`, `away_wins`, `home_losses`, `away_losses`

#### Home/Away Splits (4 features)
- `home_epa_home_avg`, `home_epa_away_avg`
- `away_epa_home_avg`, `away_epa_away_avg`

#### Venue & Weather (NEW in v3) (7 features)
- `venue_home_win_rate` - Historical home win rate at venue
- `venue_avg_margin` - Average margin at venue
- `venue_avg_total` - Average total points at venue
- `is_dome` - Indoor stadium indicator
- `is_outdoor` - Outdoor stadium indicator
- `is_cold_game` - Cold weather game (temp < 40°F)
- `is_windy_game` - Windy conditions (> 15 mph)

#### Context (1 feature)
- `week` - Week number in season

## File Structure

```
models/xgboost/v3_production/
├── README.md                    # This file
├── model.json                   # Trained XGBoost model
├── metadata.json                # Model metadata and configuration
└── feature_importance.csv       # Feature importance rankings
```

## Usage

### 1. Inference (Single Prediction)

```python
from py.predict.v3_inference import V3Predictor

# Initialize predictor
predictor = V3Predictor()

# Predict specific games
results = predictor.predict(game_ids=['2025_05_KC_NO'])

# Predict all games in a week
results = predictor.predict(season=2025, week=5)

# Save predictions
predictor.predict_and_save(
    output_path='data/predictions/week5.csv',
    season=2025,
    week=5
)
```

### 2. Ensemble Predictions (v3 + v2.1)

```python
from py.predict.ensemble import EnsemblePredictor

# Weighted ensemble (v3: 70%, v2.1: 30%)
predictor = EnsemblePredictor(method='weighted')
results = predictor.predict(season=2025, week=5)

# Fallback ensemble (use v2.1 if v3 has low confidence)
predictor = EnsemblePredictor(method='fallback')
results = predictor.predict(season=2025, week=5)
```

### 3. Command Line

```bash
# Generate v3 predictions
uv run python py/predict/v3_inference.py --season 2025 --week 5 --output predictions.csv

# Generate ensemble predictions
uv run python py/predict/ensemble.py --season 2025 --week 5 --method weighted

# Monitor production performance
uv run python py/monitor/production_monitor.py --predictions-file predictions.csv --season 2025 --week 5
```

## Production Monitoring

### Monitoring Infrastructure

The model includes comprehensive production monitoring:

1. **Predictions Logging** (`monitoring.predictions`)
   - Every prediction is logged with metadata
   - Actual outcomes are back-filled after games complete
   - Tracks prediction correctness and confidence

2. **Performance Metrics** (`monitoring.model_metrics`)
   - Weekly/monthly/seasonal performance metrics
   - Tracks Brier score, accuracy, AUC, calibration
   - Enables performance trending over time

3. **Feature Drift Detection** (`monitoring.feature_drift`)
   - Monitors feature distributions for drift
   - Alerts on significant changes from baseline
   - Helps identify when retraining is needed

4. **Alerts System** (`monitoring.alerts`)
   - Automated alerting for performance degradation
   - Tracks alert resolution and ownership

### Weekly Monitoring Workflow

```bash
# 1. Initialize monitoring tables (one-time)
uv run python py/monitor/production_monitor.py --init-tables

# 2. Generate predictions for upcoming week
uv run python py/predict/v3_inference.py --season 2025 --week 5 --output week5_predictions.csv

# 3. After week completes, evaluate performance
uv run python py/monitor/production_monitor.py \
    --predictions-file week5_predictions.csv \
    --season 2025 \
    --week 5 \
    --model-version v3.0.0

# 4. Generate weekly report
uv run python py/monitor/production_monitor.py --weekly-report --season 2025 --week 5
```

## Model Retraining Guidelines

### When to Retrain

Retrain the model if any of the following conditions are met:

1. **Performance Degradation**
   - Brier score increases by >5% for 3+ consecutive weeks
   - Accuracy drops below 60%
   - Calibration error exceeds 0.10

2. **Significant Data Drift**
   - Multiple features show drift z-score > 5
   - New venue/team dynamics not captured

3. **Seasonal Schedule**
   - Beginning of each new season (incorporate previous season data)
   - Mid-season (after week 8-10) if early performance is poor

### Retraining Process

```bash
# 1. Update feature data with latest games
uv run python py/features/materialized_view_features.py \
    --output data/processed/features/asof_team_features_v3.csv \
    --games-only

# 2. Run hyperparameter sweep (if exploring new configurations)
uv run python py/models/xgboost_gpu_v3.py \
    --features-csv data/processed/features/asof_team_features_v3.csv \
    --sweep \
    --output-dir models/xgboost/v3_sweep_YYYYMMDD

# 3. Train production model with updated data
# Update training script to include latest season
# Save new model with incremented version (v3.1.0)

# 4. Validate new model performance on holdout data

# 5. Deploy new model and update monitoring
```

## Feature Engineering

Features are extracted from 6 materialized views:

1. `mv_game_aggregates` - Game-level EPA, success rate, explosive plays
2. `mv_team_rolling_stats` - Rolling windows (L3, L5, L10)
3. `mv_team_matchup_history` - Head-to-head records
4. `mv_player_season_stats` - Player aggregates (QB/RB/WR)
5. `mv_betting_features` - Spread coverage, over/under trends
6. `mv_venue_weather_features` - Stadium and weather conditions

Features are computed **as-of** each game, ensuring no data leakage.

## Known Limitations

1. **Feature Availability**
   - Requires at least 3 completed games per team for L3 rolling stats
   - Early season predictions (weeks 1-3) may have higher uncertainty
   - Weather features only available for outdoor games

2. **Data Dependencies**
   - Depends on timely updates to `games` table
   - Materialized views must be refreshed after new games
   - Betting lines must be populated

3. **Model Assumptions**
   - Assumes relatively stable team strength within season
   - Does not account for player injuries (future enhancement)
   - Home field advantage assumed constant (could be team-specific)

## Future Enhancements

1. **Player Impact Model**
   - Incorporate QB, key position player availability
   - Adjust predictions based on injury reports

2. **In-Season Adaptation**
   - Online learning to adapt to within-season dynamics
   - Weighted recent games more heavily

3. **Opponent-Specific Modeling**
   - Head-to-head historical performance
   - Matchup-specific adjustments

4. **Alternative Targets**
   - Point spread coverage prediction
   - Total points over/under prediction
   - Margin of victory regression

## Support

For questions or issues:
- Review monitoring dashboard: `monitoring.model_metrics`
- Check logs: `data/logs/`
- Model training code: `py/models/xgboost_gpu_v3.py`
- Inference code: `py/predict/v3_inference.py`

---

**Model Version:** v3.0.0
**Last Updated:** October 10, 2025
**Next Review:** Start of 2026 Season
