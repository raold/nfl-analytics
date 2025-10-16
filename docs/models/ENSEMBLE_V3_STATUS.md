# Ensemble v3.0 Development Status

**Date**: October 13, 2025
**Version**: v3.0 (4-way ensemble with BNN)
**Status**: ⚠️ Development - Backtest Issues Identified

---

## Executive Summary

The v3.0 ensemble framework has been implemented with 4-way model combination (Bayesian + XGBoost + BNN + State-space), but comprehensive backtesting revealed critical methodology issues that must be addressed before production deployment.

## Ensemble Architecture

### Component Models
```
Model Type              Version    Weight Method           Status
────────────────────────────────────────────────────────────────────
Bayesian Hierarchical   v2.5       Inverse Variance        ✓ Trained
XGBoost                 v2.1       Inverse Variance        ✓ Trained
BNN (Neural Net)        v2.0       Inverse Variance        ✓ Training
State-space             v1.0       Inverse Variance        ✓ Trained
```

### Ensemble Methods
1. **Inverse Variance Weighting**: Default, no meta-learner required
2. **Stacking** (Optional): Meta-learner for optimal combination
3. **Portfolio Optimization**: Kelly criterion for bet sizing

### Code Structure
```
py/ensemble/
├── enhanced_ensemble_v3.py       ✓ Implemented (450 lines)
├── stacking_meta_learner.py      ✓ Implemented
└── correlation_analysis.py       ✓ Implemented

py/backtests/
├── comprehensive_ensemble_backtest.py   ✓ Implemented (480 lines)
└── bayesian_props_multiyear_backtest.py ✓ Existing
```

---

## Backtest Results (October 13, 2025)

### Test Configuration
```
Period: 2022-2024
Initial Bankroll: $10,000
Kelly Fraction: 0.25
Min Edge: 2%
Models Tested: hierarchical_v1.0, informative_priors_v2.5, ensemble_v3.0
```

### Results Summary ❌
```
Model                   Total Bets  Win Rate  ROI      Bankroll    Status
────────────────────────────────────────────────────────────────────────────
hierarchical_v1.0       582        51.9%     -3.75%   $4,101      ❌ FAILED
informative_priors_v2.5 0          0.0%      0.00%    $10,000     ⚠️ NO DATA
ensemble_v3.0           0          0.0%      0.00%    $10,000     ⚠️ NO DATA
```

**Target**: +5-7% ROI
**Achieved**: -3.75% ROI (v1.0) / No bets (v2.5, v3.0)
**Gap**: 8.75% below minimum target

---

## Critical Issues Identified

### Issue #1: Missing Historical Predictions ❌
**Problem**: Database lacks historical predictions for backtesting

**Details**:
```sql
-- Query attempted
SELECT player_id, season, rating_mean, rating_sd
FROM mart.bayesian_player_ratings
WHERE model_version = 'hierarchical_v1.0' AND season = 2022;

-- Results
hierarchical_v1.0:       0 predictions (2022-2023), 118 predictions (2024)
informative_priors_v2.5: 0 predictions (all seasons)
ensemble_v3.0:           0 predictions (all seasons)
```

**Impact**:
- Backtest can only run on 2024 data (118 predictions)
- No year-over-year validation
- No statistical significance

**Root Cause**:
- Models trained on current data only
- No historical predictions saved to database
- Predictions not generated retrospectively

### Issue #2: Look-Ahead Bias ❌
**Problem**: Current backtest methodology has temporal leakage

**Flawed Approach**:
```python
# WRONG: Uses models trained on ALL data including future
predictions = load_model_predictions("v2.5", season=2022)
actual_results = load_actual_data(season=2022)
evaluate_bets(predictions, actual_results)
```

**Correct Approach** (Walk-Forward):
```python
# RIGHT: Train on past data only, predict forward
for season in [2022, 2023, 2024]:
    train_data = load_data(start=2006, end=season-1)
    model = train_model(train_data)  # No future data!
    predictions = model.predict(season)
    actual_results = load_actual_data(season)
    evaluate_bets(predictions, actual_results)
```

**Impact**:
- Results are unreliable
- Overstates model performance
- Not production-representative

### Issue #3: Poor Baseline Performance ❌
**Problem**: v1.0 model lost 59% of bankroll when predictions existed

**Metrics**:
```
Win Rate: 51.9% (barely above coin flip)
ROI: -3.75% (losing money)
Max Drawdown: -83.5% (catastrophic)
Sharpe Ratio: -1.08 (negative risk-adjusted returns)
Final Bankroll: $4,101 (lost $5,899)
```

**Analysis**:
- 582 bets placed (only in 2024 where predictions existed)
- Edge calculation may be flawed
- Kelly sizing may be too aggressive
- Lines simulated, not real betting markets

### Issue #4: Database Schema Corrections ✅
**Problem**: Multiple column name mismatches (NOW FIXED)

**Fixes Applied**:
```sql
-- plays table: Use correct column names
SELECT passer_player_id, rusher_player_id, receiver_player_id
-- NOT: passer_id, rusher_id, receiver_id

-- games table: Use correct date column
SELECT kickoff as game_date
-- NOT: game_date (column doesn't exist)
```

**Status**: ✅ All schema issues resolved

---

## Recommendations

### Immediate Actions

#### Option A: Walk-Forward Backtest (Rigorous)
**Pros**:
- Scientifically sound
- No look-ahead bias
- Production-representative

**Cons**:
- Requires retraining models for each historical period
- Time-intensive (~2-3 days development)
- Computationally expensive

**Implementation**:
```python
class WalkForwardBacktest:
    def run(self, start_season=2022, end_season=2024):
        for season in range(start_season, end_season + 1):
            # Train on past data only
            train_data = self.load_data(end_season=season - 1)
            model = self.train_model(train_data)

            # Predict current season
            predictions = model.predict(season)

            # Evaluate against actual
            results = self.evaluate(predictions, season)

            # Update bankroll
            self.update_portfolio(results)
```

#### Option B: Cross-Validation Approach (Faster)
**Pros**:
- Faster to implement
- Tests ensemble logic
- Validates weights

**Cons**:
- Less rigorous than walk-forward
- May still have some temporal leakage
- Doesn't test betting simulation

**Implementation**:
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=3)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train ensemble
    ensemble.fit(X_train, y_train)

    # Evaluate
    score = ensemble.score(X_test, y_test)
```

#### Option C: Simplified Validation (Quickest)
**Pros**:
- Fast to implement (<1 hour)
- Tests ensemble weights
- Validates uncertainty quantification

**Cons**:
- No betting simulation
- No ROI metrics
- Limited insights

**Implementation**:
```python
# Just test ensemble combination logic
y_pred_ensemble = ensemble.predict(X_test)
mae = mean_absolute_error(y_test, y_pred_ensemble)
calibration = check_calibration(y_test, y_pred_ensemble, uncertainty)
```

### Short-term Priorities

1. **Fix BNN Integration** ✅ (In Progress)
   - Improved BNN training completing with excellent convergence
   - Integrate into ensemble with proper weighting

2. **Validate Ensemble Weights**
   - Use Option C (simplified validation) first
   - Ensure BNN contribution is positive
   - Adjust weights if calibration is poor

3. **Generate 2024 Predictions**
   - Use ensemble v3.0 to predict remaining 2024 games
   - Compare against actual results as they happen
   - Paper trade without real money

4. **Document Methodology**
   - Write up proper backtest requirements
   - Get alignment on validation approach
   - Plan production deployment timeline

### Long-term Roadmap

**Phase 1: Model Validation** (2 weeks)
- [ ] Complete BNN training and evaluation
- [ ] Integrate BNN into ensemble v3.0
- [ ] Run simplified ensemble validation (Option C)
- [ ] Verify calibration meets standards

**Phase 2: Proper Backtesting** (3-4 weeks)
- [ ] Implement walk-forward backtest framework
- [ ] Retrain models for historical periods (2022-2024)
- [ ] Save all historical predictions to database
- [ ] Run comprehensive 3-year backtest
- [ ] Target: Validate +5-7% ROI

**Phase 3: Production Deployment** (2 weeks)
- [ ] Real-time prediction pipeline
- [ ] Monitoring dashboards
- [ ] Alert system for anomalies
- [ ] Paper trading for Week 8-10
- [ ] Live betting (if metrics hold)

---

## Database Schema Reference

### Required Tables
```sql
-- Historical predictions (MISSING - needs creation)
CREATE TABLE mart.model_predictions_history (
    prediction_id SERIAL PRIMARY KEY,
    model_version VARCHAR(50),
    player_id VARCHAR(50),
    season INTEGER,
    week INTEGER,
    stat_type VARCHAR(50),
    prediction_mean FLOAT,
    prediction_std FLOAT,
    prediction_q05 FLOAT,
    prediction_q95 FLOAT,
    created_at TIMESTAMP,
    UNIQUE(model_version, player_id, season, week, stat_type)
);

-- Actual results (EXISTS)
-- Uses: plays table, mart.player_game_stats
```

### Column Name Reference
```sql
-- plays table
plays.passer_player_id   (NOT passer_id)
plays.rusher_player_id   (NOT rusher_id)
plays.receiver_player_id (NOT receiver_id)

-- games table
games.kickoff           (NOT game_date)
```

---

## Key Metrics & Targets

### Model Performance
```
Metric                  Current     Target      Status
──────────────────────────────────────────────────────
90% CI Coverage         TBD         85-92%      ⏳
MAE                     TBD         <20 yds     ⏳
RMSE                    TBD         <30 yds     ⏳
```

### Betting Performance
```
Metric                  Current     Target      Status
──────────────────────────────────────────────────────
ROI                     -3.75%      +5-7%       ❌
Win Rate                51.9%       >53%        ⚠️
Sharpe Ratio            -1.08       >1.0        ❌
Max Drawdown            -83.5%      <30%        ❌
```

---

## Files Reference

### Ensemble Code
```
py/ensemble/enhanced_ensemble_v3.py:314           Main ensemble class
py/ensemble/stacking_meta_learner.py:150          Stacking implementation
py/backtests/comprehensive_ensemble_backtest.py:480   Backtest framework
```

### Configuration
```
py/ensemble/enhanced_ensemble_v3.py:84-89
Default weights:
- Bayesian: 0.35
- XGBoost: 0.30
- BNN: 0.20
- State-space: 0.15
```

---

## Conclusion

The ensemble v3.0 framework is **structurally sound** but requires:

1. ✅ BNN integration (in progress - excellent convergence)
2. ❌ Proper historical validation (walk-forward backtest needed)
3. ⚠️ Improved baseline models (v1.0 lost money)
4. ✅ Database schema fixes (completed)

**Recommended Path**:
- Complete BNN integration
- Run simplified validation (Option C)
- Plan walk-forward backtest for proper validation
- Do NOT deploy to production without proper validation

---

**Last Updated**: October 13, 2025
**Next Review**: After BNN training completion
**Owner**: Ensemble Development Team
