# Ensemble v3.0 Validation & Production Deployment Roadmap

**Date**: October 13, 2025
**Target**: Production deployment by November 1, 2025
**Status**: ⚠️ Critical validation issues identified - deployment paused

---

## Executive Summary

This document outlines the comprehensive validation and deployment plan for Ensemble v3.0, addressing critical methodology issues discovered during initial backtesting. **Production deployment is blocked pending proper walk-forward validation**.

### Current Blockers

1. ❌ **Missing Historical Predictions** - No database predictions for 2022-2023
2. ❌ **Look-Ahead Bias** - Current backtest trains on future data
3. ⚠️ **Poor v1.0 Baseline** - Lost money (-3.75% ROI) vs +5-7% target

### Recommendation

**DO NOT DEPLOY** to production until:
- ✅ Walk-forward validation framework implemented
- ✅ 3-year backtest achieves +5-7% ROI
- ✅ BNN calibration verified (85-92% coverage)
- ✅ Real market lines integrated (not simulated)

---

## Phase 1: Model Validation (Week 1-2)

### Objective
Verify BNN improvements and integrate into ensemble with proper uncertainty quantification.

### Tasks

#### 1.1 BNN v2.0 Completion ⏳
**Status**: Training in progress
**ETA**: October 13 (today)
**Owner**: Model team

**Steps**:
```bash
# Training running with ID: 594d67
# Monitor: models/bayesian/bnn_rushing_improved_v2_training.log

# When complete, verify:
uv run python py/validation/verify_bnn.py \
  --model models/bayesian/bnn_rushing_improved_v2.pkl \
  --test-data data/test_rushing.csv
```

**Success Criteria**:
- ✅ Divergences: 0 (0.00%)
- ✅ R-hat max: <1.01
- ✅ ESS mean: >1000
- ⏳ 90% CI coverage: 85-92% (vs 19.8% in v1.0)
- ⏳ MAE: ~18-20 yards (similar to v1.0)

#### 1.2 Ensemble Integration
**Status**: Pending BNN completion
**ETA**: October 14
**Owner**: Ensemble team

**Implementation**:
```python
# py/ensemble/enhanced_ensemble_v3.py
from py.ensemble.enhanced_ensemble_v3 import EnhancedEnsembleV3

# Initialize with BNN
ensemble = EnhancedEnsembleV3(
    use_bnn=True,
    use_stacking=False,  # Start with inverse variance
    use_portfolio_opt=True,
    kelly_fraction=0.10  # Reduced from 0.25
)

# Load all models
ensemble.load_models(
    bayesian_path="models/bayesian/informative_priors_v2.5.pkl",
    xgb_path="models/xgboost/v2_1.pkl",
    bnn_path="models/bayesian/bnn_rushing_improved_v2.pkl",
    statespace_path="models/bayesian/state_space_v1.rds"
)

# Verify weights (should sum to 1.0)
print(ensemble.get_weights())
# Expected: {bayesian: 0.35, xgb: 0.30, bnn: 0.20, statespace: 0.15}
```

**Success Criteria**:
- ✅ All models load successfully
- ✅ Weights sum to 1.0
- ✅ Uncertainty properly combined (inverse variance weighting)
- ✅ Predictions faster than 500ms per game

#### 1.3 Simplified Validation
**Status**: Not started
**ETA**: October 14
**Effort**: 2 hours

**Purpose**: Quick validation of ensemble logic without full backtest simulation.

**Implementation**:
```python
# Test on 2024 holdout set (no betting simulation)
from py.validation.simplified_validation import SimplifiedValidator

validator = SimplifiedValidator()
results = validator.validate(
    model=ensemble,
    test_data=X_test_2024,
    y_true=y_test_2024
)

# Check metrics
print(f"MAE: {results['mae']:.2f} yards")
print(f"RMSE: {results['rmse']:.2f} yards")
print(f"90% CI Coverage: {results['ci_coverage']:.1f}%")
print(f"Calibration ECE: {results['ece']:.4f}")
```

**Success Criteria**:
- ✅ MAE < 20 yards
- ✅ RMSE < 30 yards
- ✅ 90% CI coverage: 85-92%
- ✅ ECE < 0.05 (well-calibrated)

---

## Phase 2: Walk-Forward Backtest (Week 2-3)

### Objective
Implement proper temporal validation without look-ahead bias to verify real-world performance.

### 2.1 Framework Design
**Status**: Design complete (below)
**ETA**: October 15-16
**Effort**: 1 day design, 1 day implementation

**Methodology**:
```
Year 2022 Test:
├─ Train: 2006-2021 (16 years)
├─ Test: 2022 (all games)
└─ Save predictions & results

Year 2023 Test:
├─ Train: 2006-2022 (17 years)
├─ Test: 2023 (all games)
└─ Save predictions & results

Year 2024 Test:
├─ Train: 2006-2023 (18 years)
├─ Test: 2024 (all games)
└─ Save predictions & results
```

**Key Principle**: NO FUTURE DATA LEAKAGE
- Model at time T sees only data before T
- Mimics real-world deployment scenario

### 2.2 Database Schema
**Status**: Schema designed (not created)
**ETA**: October 15
**Effort**: 1 hour

**Required Table**:
```sql
CREATE TABLE mart.model_predictions_history (
    prediction_id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50),
    player_id VARCHAR(50),
    game_id VARCHAR(50),
    season INTEGER NOT NULL,
    week INTEGER,
    stat_type VARCHAR(50) NOT NULL,
    prediction_mean FLOAT NOT NULL,
    prediction_std FLOAT,
    prediction_q05 FLOAT,
    prediction_q50 FLOAT,
    prediction_q95 FLOAT,
    actual_value FLOAT,
    trained_through_season INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(model_version, player_id, game_id, stat_type)
);

CREATE INDEX idx_model_pred_history_lookup
    ON mart.model_predictions_history(model_version, season, week);
```

### 2.3 Implementation
**Status**: Skeleton code exists (needs walk-forward logic)
**ETA**: October 16-17
**Effort**: 2 days

**Pseudocode**:
```python
class WalkForwardBacktest:
    def __init__(self, models, start_season=2022, end_season=2024):
        self.models = models
        self.test_seasons = range(start_season, end_season + 1)
        self.bankroll_history = []

    def run(self):
        bankroll = 10000
        all_bets = []

        for test_season in self.test_seasons:
            # 1. Load training data (only past)
            train_data = self.load_data(
                start_season=2006,
                end_season=test_season - 1
            )

            # 2. Retrain ALL models on this training data
            for model_name, model in self.models.items():
                print(f"Training {model_name} on 2006-{test_season-1}")
                model.fit(train_data)

            # 3. Generate predictions for test season
            test_data = self.load_data(
                start_season=test_season,
                end_season=test_season
            )
            predictions = self.ensemble.predict(test_data)

            # 4. Save predictions to database
            self.save_predictions(predictions, test_season, test_season - 1)

            # 5. Simulate betting on test season
            season_bets, season_bankroll = self.simulate_betting(
                predictions, test_data, bankroll
            )

            bankroll = season_bankroll
            all_bets.extend(season_bets)

            print(f"{test_season}: Bankroll = ${bankroll:,.0f}")

        return self.calculate_metrics(all_bets, bankroll)
```

**File**: `py/backtests/walk_forward_backtest.py`

**Usage**:
```bash
uv run python py/backtests/walk_forward_backtest.py \
  --start-season 2022 \
  --end-season 2024 \
  --initial-bankroll 10000 \
  --kelly-fraction 0.10 \
  --min-edge 0.02 \
  --output reports/walk_forward_results.json
```

**Success Criteria**:
- ✅ ROI: +5-7% over 3 years
- ✅ Win rate: >53%
- ✅ Sharpe ratio: >1.0
- ✅ Max drawdown: <30%
- ✅ No year loses >10% of bankroll

---

## Phase 3: Enhanced Validation (Week 3-4)

### Objective
Address remaining validation gaps and prepare for production deployment.

### 3.1 Real Market Lines Integration
**Status**: Not started
**ETA**: October 18-19
**Effort**: 2 days

**Problem**: Current backtest uses simulated lines (implied probability from our models). Need real historical lines.

**Solution**:
```python
# Pull historical lines from Odds API history
import requests

def get_historical_lines(game_id, stat_type="passing_yards"):
    """Fetch real closing lines for game"""
    response = requests.get(
        "https://api.the-odds-api.com/v4/historical/...",
        params={"game_id": game_id, "stat_type": stat_type},
        headers={"apiKey": os.getenv("ODDS_API_KEY")}
    )
    return response.json()
```

**Database Table**:
```sql
CREATE TABLE odds_history.prop_lines_historical (
    line_id SERIAL PRIMARY KEY,
    game_id VARCHAR(50),
    player_id VARCHAR(50),
    bookmaker VARCHAR(50),
    stat_type VARCHAR(50),
    line_value FLOAT,
    over_odds FLOAT,
    under_odds FLOAT,
    closing_time TIMESTAMP,
    UNIQUE(game_id, player_id, bookmaker, stat_type, closing_time)
);
```

### 3.2 Edge Calculation Review
**Status**: Not started
**ETA**: October 19
**Effort**: 4 hours

**Current Method**:
```python
# May be flawed
edge = our_prob - implied_prob_from_line
```

**Need to Verify**:
- Are odds-to-probability conversions correct?
- Is vig being properly removed?
- Are we comparing apples-to-apples (same stat definition)?

**Validation Test**:
```python
# Test on known profitable scenarios
assert calculate_edge(0.55, -110) > 0  # Should have edge
assert calculate_edge(0.50, -110) < 0  # Should have no edge
```

### 3.3 Kelly Fraction Optimization
**Status**: Reduced to 0.10 (was 0.25)
**ETA**: October 20
**Effort**: 2 hours

**Current Issue**: Kelly fraction of 0.25 may be too aggressive (led to -83.5% drawdown).

**Solution**: Grid search for optimal fraction
```python
kelly_fractions = [0.05, 0.10, 0.15, 0.20, 0.25]
for fraction in kelly_fractions:
    results = run_backtest(kelly_fraction=fraction)
    print(f"Fraction: {fraction}, ROI: {results['roi']:.2f}%, Drawdown: {results['max_dd']:.1f}%")
```

**Target**: Find fraction that maximizes Sharpe ratio while keeping drawdown <30%.

---

## Phase 4: Production Deployment (Week 4-5)

### Objective
Deploy validated ensemble to production with monitoring and safeguards.

### 4.1 Pre-Deployment Checklist

**Model Quality** ✅/❌:
- [ ] BNN convergence verified (R-hat < 1.01)
- [ ] BNN calibration verified (90% CI: 85-92%)
- [ ] Walk-forward backtest complete
- [ ] ROI +5-7% achieved over 3 years
- [ ] Max drawdown <30%
- [ ] Real market lines integrated
- [ ] Edge calculation validated

**Infrastructure** ✅/❌:
- [ ] Database schema deployed
- [ ] Prediction API tested
- [ ] Monitoring dashboards configured
- [ ] Alert thresholds set
- [ ] Rollback plan documented
- [ ] Paper trading tested (Week 8-10)

### 4.2 Deployment Timeline

**Week 1 (Oct 14-20)**: Validation
- Complete BNN integration
- Run simplified validation
- Implement walk-forward framework
- Run 3-year backtest

**Week 2 (Oct 21-27)**: Enhancement
- Integrate real market lines
- Optimize Kelly fraction
- Review edge calculation
- Final validation run

**Week 3 (Oct 28 - Nov 3)**: Pre-Production
- Deploy infrastructure
- Paper trading (Week 8-10)
- Monitor without real money
- Validate live predictions

**Week 4 (Nov 4-10)**: Production Launch
- Start live betting (if metrics hold)
- Small bet sizes initially ($10-50)
- Scale up gradually

### 4.3 Production Monitoring

**Real-Time Dashboards**:
```python
# Monitor key metrics every game
metrics = {
    "predictions_generated": count(),
    "average_edge": mean(edge),
    "bets_placed": count(),
    "current_bankroll": sum(bankroll),
    "daily_roi": calculate_roi(today),
    "model_uncertainty": mean(std)
}
```

**Alert Triggers**:
- 🔴 Bankroll drops >10% in single day
- 🔴 Win rate <45% over 10+ bets
- 🟡 Edge suddenly drops <1%
- 🟡 Uncertainty spikes >2 standard deviations
- 🟡 Missing predictions for scheduled games

**Kill Switch**: Auto-disable betting if:
- Bankroll drops >30%
- Win rate <40% over 20+ bets
- Model uncertainty exceeds historical maximum

---

## Risk Assessment & Mitigation

### High-Risk Items 🔴

**1. Look-Ahead Bias Not Fixed**
- **Risk**: Results still unreliable even with walk-forward
- **Mitigation**: Code review by 2nd engineer, test with toy dataset
- **Owner**: Lead engineer

**2. Real Lines Unavailable**
- **Risk**: Cannot validate on real markets
- **Mitigation**: Start collecting lines now for future validation
- **Owner**: Data team

**3. Poor Performance on New Season**
- **Risk**: Models trained on 2006-2023 fail on 2024
- **Mitigation**: Paper trade first, monitor closely, kill switch
- **Owner**: Ops team

### Medium-Risk Items 🟡

**1. BNN Calibration Not Improved**
- **Risk**: 90% CI still <85% even with fixes
- **Mitigation**: Reduce BNN weight in ensemble (0.20 → 0.10)
- **Owner**: Model team

**2. Infrastructure Failure**
- **Risk**: Database/API down during game
- **Mitigation**: Health checks, automatic failover, manual backup
- **Owner**: DevOps

---

## Success Metrics

### Model Performance
```
Metric                  Current     Target      Priority
────────────────────────────────────────────────────────────
BNN Convergence         ✅ 1.0027   <1.01       P0
BNN Calibration         ⏳ TBD      85-92%      P0
Ensemble MAE            ⏳ TBD      <18 yds     P1
Ensemble RMSE           ⏳ TBD      <28 yds     P1
```

### Backtest Performance
```
Metric                  Current     Target      Priority
────────────────────────────────────────────────────────────
3-Year ROI              ❌ -3.75%   +5-7%       P0
Win Rate                ⚠️ 51.9%    >53%        P0
Sharpe Ratio            ❌ -1.08    >1.0        P0
Max Drawdown            ❌ -83.5%   <30%        P0
```

### Production Performance
```
Metric                  Target      Monitoring
────────────────────────────────────────────────
Weekly ROI              +1-2%       Daily
Win Rate                >53%        Weekly
Uptime                  >99.5%      Real-time
Prediction Latency      <500ms      Real-time
```

---

## Rollback Plan

### Trigger Conditions
- Bankroll drops >30%
- Win rate <40% over 20+ bets
- Critical bug discovered
- Model drift detected

### Rollback Procedure
```bash
# 1. Stop all automated betting
python py/ops/kill_switch.py --reason "performance_degradation"

# 2. Rollback to previous model version
python py/ops/rollback_model.py --version v2.5

# 3. Notify team
python py/ops/send_alert.py --level critical --message "Production rollback executed"

# 4. Investigate root cause
python py/analysis/investigate_failure.py --start-date 2024-11-01

# 5. Re-validate before redeployment
python py/backtests/walk_forward_backtest.py --quick-check
```

---

## Timeline Summary

```
Week 1 (Oct 14-20): Validation
├─ Oct 14: BNN completion + ensemble integration
├─ Oct 15-16: Walk-forward framework implementation
├─ Oct 17-19: 3-year backtest run
└─ Oct 20: Results review

Week 2 (Oct 21-27): Enhancement
├─ Oct 21-22: Real market lines integration
├─ Oct 23-24: Edge calculation review
├─ Oct 25-26: Kelly fraction optimization
└─ Oct 27: Final backtest run

Week 3 (Oct 28 - Nov 3): Pre-Production
├─ Oct 28-29: Infrastructure deployment
├─ Oct 30 - Nov 1: Paper trading (Week 8-10)
└─ Nov 2-3: Go/no-go decision

Week 4 (Nov 4-10): Production Launch (If Approved)
├─ Nov 4: Start live betting (small size)
├─ Nov 5-7: Monitor closely
├─ Nov 8-10: Scale up if stable
└─ Nov 11+: Regular operations
```

---

## Conclusion

**Current Status**: ⚠️ Development complete, validation blocked

**Key Blockers**:
1. Walk-forward backtest not implemented
2. Historical predictions missing from database
3. Real market lines not integrated

**Recommendation**: **DO NOT DEPLOY** until all P0 items complete and 3-year backtest achieves +5-7% ROI.

**Best Case**: Production deployment by November 4, 2025
**Realistic Case**: Production deployment by November 11, 2025
**Worst Case**: Back to drawing board if walk-forward backtest fails

---

**Document Version**: 1.0
**Last Updated**: October 13, 2025
**Next Review**: October 14, 2025 (after BNN completion)
**Owner**: Model Development Team

**Related Documents**:
- [ENSEMBLE_V3_STATUS.md](ENSEMBLE_V3_STATUS.md) - Current status
- [MODEL_REGISTRY.md](MODEL_REGISTRY.md) - Model inventory
- [BNN_IMPROVEMENT_REPORT.md](BNN_IMPROVEMENT_REPORT.md) - BNN analysis
