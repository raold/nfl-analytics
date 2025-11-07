# Phase 2: Player Props Ensemble - Implementation Plan

**Date**: November 6, 2025
**Status**: 🚀 **READY TO START**
**Context**: Phase 1 (Pinnacle + Paper Trading) Complete ✅

---

## Executive Summary

**Phase 2** deploys a **4-way ensemble** for NFL player props betting, combining:
1. ✅ **Bayesian Hierarchical Models** (QB-WR chemistry, informative priors)
2. ✅ **XGBoost** (gradient boosting baseline)
3. ✅ **Bayesian Neural Networks** (non-linear interactions)
4. ⏳ **Stacked Meta-Learner** (optimal weighting)

**Expected Performance**: +5-7% ROI (vs +1.1% from Pinnacle alone)

**Timeline**: 1-2 weeks (parallel development)

---

## 🎯 Phase 2 Objectives

### **Primary Goals**
1. Deploy 4-way ensemble for player props (passing, rushing, receiving)
2. Integrate with Pinnacle paper trading (Phase 1 infrastructure)
3. Validate on Week 10-11 games (2025 season)
4. Achieve 52-55% win rate, +2-4% ROI (ensemble effect)

### **Success Criteria**
- **Minimum**: Win rate 52%+, ROI ≥ +1% (better than Pinnacle alone)
- **Target**: Win rate 54%+, ROI +2-4% (ensemble working)
- **Excellent**: Win rate 56%+, ROI +4-6% (ready for Phase 3)

---

## 📊 Current Status Assessment

### ✅ **What's Already Built**

#### **1. Trained Models Available**
```bash
models/bayesian/
├── passing_informative_priors_v1.rds       # 5.2 MB - Bayesian passing
├── receiving_qb_chemistry_v1.rds           # 210 MB - QB-WR chemistry
├── bnn_passing_v1.pkl                      # 80 MB - BNN passing
├── bnn_multioutput_v1_final.pt             # 59 KB - Multi-output BNN
└── bnn_rushing_v1.pkl                      # 53 MB - BNN rushing
```

**Status**:
- ✅ Passing models: Bayesian (brms) + BNN (PyMC + PyTorch)
- ✅ Receiving models: QB-WR chemistry (brms)
- ✅ Rushing models: BNN (PyMC)
- ⏳ XGBoost models: Need to check if trained

#### **2. Infrastructure Complete**
- ✅ `py/ensemble/enhanced_ensemble_v3.py` (4-way ensemble)
- ✅ `py/models/props_predictor.py` (XGBoost props)
- ✅ `py/production/props_production_pipeline.py` (pipeline)
- ✅ `py/production/paper_trader.py` (paper trading)
- ✅ Database: `prop_lines_history` table with 627 live lines

#### **3. Data Pipeline**
- ✅ Odds API: 483/500 requests remaining
- ✅ Pinnacle lines: 4.5-5.7% hold (validated)
- ✅ 627 prop lines collected (Week 10, 14 games)
- ✅ `mart.player_game_stats` table (24,950 records)
- ✅ `mart.bayesian_player_ratings` table (predictions stored)

### ⏳ **What Needs to Be Done**

1. **Check XGBoost Models** - Do we have trained XGBoost prop models?
2. **Integrate Ensemble** - Wire up 4-way ensemble to production pipeline
3. **Test on Historical Data** - Validate ensemble on 2024 season
4. **Paper Trade Week 10** - Deploy on live games with paper trading
5. **Monitor & Calibrate** - Track performance, adjust weights

---

## 📋 Phase 2 Implementation Tasks

### **Step 1: Model Inventory (30 minutes)**

**Goal**: Identify what models are trained and ready to use

**Tasks**:
1. ✅ List all Bayesian models (done above)
2. ⏳ Check if XGBoost prop models exist:
   ```bash
   ls -lh models/xgboost/props_* models/props/*
   ```
3. ⏳ Test loading each model to ensure compatibility
4. ⏳ Document model metadata (training date, performance, features)

**Deliverable**: `PHASE2_MODEL_INVENTORY.md`

---

### **Step 2: Ensemble Integration (2-3 hours)**

**Goal**: Wire up 4-way ensemble to production pipeline

**Tasks**:
1. ⏳ Create `py/models/player_props_ensemble.py`:
   - Load all 4 models (Bayesian, XGBoost, BNN, State-space)
   - Implement inverse variance weighting (no meta-learner initially)
   - Return ensemble prediction + uncertainty

2. ⏳ Modify `props_production_pipeline.py`:
   - Replace single-model predictions with ensemble
   - Pass ensemble predictions to EV selector
   - Maintain paper trading integration

3. ⏳ Test on single player (dry run):
   ```bash
   python py/models/player_props_ensemble.py \
     --player-id "00-0036355" \
     --prop-type passing_yards \
     --opponent "KC" \
     --line 275.5
   ```

**Deliverable**: Working ensemble predictor

---

### **Step 3: Historical Validation (1-2 hours)**

**Goal**: Validate ensemble on 2024 season (out-of-sample)

**Tasks**:
1. ⏳ Run backtest on 2024 season:
   ```bash
   python py/backtests/comprehensive_ensemble_backtest.py \
     --season 2024 \
     --use-ensemble \
     --output analysis/results/phase2_ensemble_backtest_2024.csv
   ```

2. ⏳ Compare to baseline:
   - XGBoost alone: Expected +3-4% ROI
   - Bayesian alone: Expected +2.5-3.5% ROI
   - **Ensemble**: Target +5-7% ROI

3. ⏳ Analyze results:
   - Win rate by prop type (passing, rushing, receiving)
   - Edge calibration (predicted EV vs actual outcome)
   - Sharpe ratio, max drawdown

**Deliverable**: `analysis/results/phase2_ensemble_validation.txt`

---

### **Step 4: Paper Trading Deployment (30 minutes)**

**Goal**: Deploy ensemble on Week 10 games with paper trading

**Tasks**:
1. ⏳ Run production pipeline with ensemble:
   ```bash
   uv run python py/production/props_production_pipeline.py \
     --paper-trade \
     --bankroll 10000 \
     --api-key dd966684be820b93588d19993ac1abc7 \
     --bookmakers pinnacle \
     --use-ensemble \
     --min-edge 0.025
   ```

2. ⏳ Monitor dashboard (http://localhost:8501 → Paper Trading)
3. ⏳ Track metrics:
   - Number of bets placed
   - Edge distribution
   - Bet sizing (should use 35% Kelly for Pinnacle)

**Deliverable**: Week 10 paper trades in database

---

### **Step 5: Settlement & Analysis (1 hour)**

**Goal**: Settle trades after Sunday games and analyze results

**Tasks**:
1. ⏳ Wait for Sunday games to complete (Nov 10)
2. ⏳ Run settlement:
   ```bash
   uv run python py/production/settle_paper_trades.py
   ```
3. ⏳ Analyze results:
   - Win rate by prop type
   - ROI vs baseline (XGBoost alone)
   - Ensemble effect (improvement over single models)

4. ⏳ Generate report:
   ```bash
   python py/analysis/paper_trading_summary.py \
     --session-id 1 \
     --output PHASE2_WEEK10_RESULTS.md
   ```

**Deliverable**: `PHASE2_WEEK10_RESULTS.md`

---

### **Step 6: Calibration & Tuning (1-2 hours)**

**Goal**: Optimize ensemble weights and edge thresholds

**Tasks**:
1. ⏳ If ensemble underperforms:
   - Check model weights (should auto-adjust via inverse variance)
   - Increase min_edge from 2.5% to 3.0%
   - Reduce Kelly fraction from 35% to 25%

2. ⏳ If ensemble overperforms:
   - Consider meta-learner (stacked ensemble)
   - Add state-space model (time-varying player ratings)
   - Increase bet sizing (up to 40% Kelly)

3. ⏳ Update `py/models/pinnacle_calibration.py` with ensemble-specific thresholds

**Deliverable**: Calibrated ensemble parameters

---

## 🎓 Technical Details

### **Ensemble Architecture**

```python
# Phase 2 Architecture (Simplified)
class PlayerPropsEnsemble:
    def __init__(self):
        # Load 3 base models
        self.bayesian_model = load_brms_model('passing_informative_priors_v1.rds')
        self.xgboost_model = load_xgboost('props_passing_v1.json')
        self.bnn_model = load_pytorch_bnn('bnn_multioutput_v1_final.pt')

    def predict(self, player_id, opponent, prop_type, line):
        # Get predictions from each model
        bayesian_pred = self.bayesian_model.predict(player_id, opponent)
        xgb_pred = self.xgboost_model.predict(player_id, opponent)
        bnn_pred = self.bnn_model.predict(player_id, opponent)

        # Inverse variance weighting
        weights = [
            1 / bayesian_pred['var'],
            1 / xgb_pred['var'],
            1 / bnn_pred['var']
        ]
        weights = weights / sum(weights)

        # Ensemble prediction
        ensemble_mean = (
            weights[0] * bayesian_pred['mean'] +
            weights[1] * xgb_pred['mean'] +
            weights[2] * bnn_pred['mean']
        )

        # Ensemble uncertainty (average variance)
        ensemble_var = (
            weights[0]**2 * bayesian_pred['var'] +
            weights[1]**2 * xgb_pred['var'] +
            weights[2]**2 * bnn_pred['var']
        )

        # Calculate edge
        over_prob = 1 - norm.cdf(line, ensemble_mean, sqrt(ensemble_var))
        market_prob = american_to_prob(-110)  # Standard -110 odds
        edge = over_prob - market_prob

        return {
            'mean': ensemble_mean,
            'std': sqrt(ensemble_var),
            'over_prob': over_prob,
            'edge': edge,
            'recommend': 'OVER' if edge > 0.025 else 'PASS'
        }
```

### **Expected Improvements**

| Component | ROI Contribution | Confidence |
|-----------|------------------|------------|
| Pinnacle lower vig | +1.1% | High (validated) |
| Bayesian models | +0.5-1.0% | Medium (QB-WR chemistry) |
| BNN non-linearity | +0.3-0.8% | Medium (calibration issues) |
| XGBoost baseline | +1.0-1.5% | High (proven) |
| Ensemble effect | +0.5-1.0% | Medium (variance reduction) |
| **Total** | **+3.4-5.4%** | **Target: +5-7%** |

---

## 📞 Support & Troubleshooting

### **Common Issues**

#### **Issue 1: Model Loading Fails**
```python
# Error: Cannot load .rds file in Python
# Solution: Use rpy2 or pre-export predictions to database
```

**Fix**:
```bash
# Export Bayesian predictions from R to database
Rscript R/export_bayesian_ratings.R \
  --model models/bayesian/passing_informative_priors_v1.rds \
  --output mart.bayesian_player_ratings
```

#### **Issue 2: XGBoost Models Missing**
If XGBoost prop models don't exist, train them:
```bash
python py/models/props_predictor.py \
  --train-seasons 2020-2023 \
  --test-season 2024 \
  --prop-type passing_yards \
  --output models/props/passing_yards_xgb.json
```

#### **Issue 3: Ensemble Underperforms**
- Check individual model performance first
- Ensure proper inverse variance weighting
- Validate edge calculations (market efficiency)
- Consider increasing min_edge threshold

---

## 🎯 Phase 2 Success Definition

**Proceed to Phase 3 IF**:
- Win rate ≥ 54% (2-4 pp above baseline)
- ROI ≥ +2% (double Pinnacle alone)
- Sharpe ratio ≥ 1.5 (consistent returns)
- Max drawdown ≤ 10% (risk management)

**Extend Phase 2 IF**:
- Win rate 52-54% (marginal improvement)
- ROI +1-2% (Pinnacle + small ensemble effect)
- Need more data (sample size < 50 bets)

**Revert to Phase 1 IF**:
- Win rate < 52% (worse than baseline)
- ROI < +1% (ensemble hurting performance)
- Risk gates frequently triggered

---

## 📈 Phase 3 Preview

**Once Phase 2 succeeds** → Phase 3: CQL Bet Sizing

Phase 3 enhancements:
1. **Conservative Q-Learning (CQL)** for bet sizing (+24% improvement)
2. **Correlation-adjusted Kelly** (portfolio optimization)
3. **Real-time Bayesian updating** (live odds adjustments)
4. **A/B testing framework** (Thompson sampling)

**Expected ROI**: +7-10% (Phase 2 + CQL + portfolio optimization)

---

**Generated**: November 6, 2025
**Status**: ✅ Ready to implement
**Next Action**: Start Step 1 (Model Inventory)
