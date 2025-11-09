# 5-Way Ensemble Integration: Implementation Checklist

**Date**: November 8, 2025
**Timeline**: 2-3 weeks
**Full Plan**: See `docs/5WAY_ENSEMBLE_INTEGRATION_PLAN.md`

---

## Week 1: Model Retraining ✅ Ready to Start

### Day 1-2: Verify Existing R Models

- [ ] Load `models/bayesian/passing_yards_hierarchical_v2.rds`
  ```r
  model <- readRDS("models/bayesian/passing_yards_hierarchical_v2.rds")
  # Check: training window 2015-2024? Rhat < 1.01? ESS > 400?
  ```

- [ ] Load `models/bayesian/passing_informative_priors_v2.rds`
  ```r
  model <- readRDS("models/bayesian/passing_informative_priors_v2.rds")
  # Check: training window 2015-2024? Rhat < 1.01? ESS > 400?
  ```

- [ ] **If not trained on canonical window**, retrain both:
  ```bash
  Rscript --vanilla R/train_hierarchical_v2_canonical.R  # 15-20 min
  Rscript --vanilla R/train_informative_priors_v2_canonical.R  # 20-25 min
  ```

### Day 3: XGBoost Training

- [ ] Create `py/models/train_xgboost_canonical.py` (200-250 lines)
  - Load data from `mart.game_features_enhanced` (157 features)
  - Apply exponential decay weights: `0.5**((2025 - season) / 4)` for 2015-2017
  - Train with GPU: `tree_method='gpu_hist'`
  - Validate on 2025 weeks 1-6

- [ ] Train XGBoost v2:
  ```bash
  uv run python py/models/train_xgboost_canonical.py \
    --output models/xgboost/passing_yards_v2.pkl \
    --validate-on-2025
  # Expected time: 10-15 minutes on RTX 4090
  ```

- [ ] Verify outputs:
  - [ ] Model file: `models/xgboost/passing_yards_v2.pkl` (~5-10 MB)
  - [ ] Validation MAE on 2025 < 48 yards ✅
  - [ ] Feature importance: `analysis/results/xgb_feature_importance.csv`

### Day 4: BNN Multi-Output Training

- [ ] Create `py/models/train_bnn_canonical.py` (300-350 lines)
  - Load data from `mart.player_game_stats`
  - Apply exponential decay weights
  - Architecture: hidden_dim=448, lr=0.00654764, dropout=0.116
  - Apply temperature scaling (T=0.75) for calibration

- [ ] Train BNN v2:
  ```bash
  uv run python py/models/train_bnn_canonical.py \
    --output models/bayesian/bnn_multioutput_v2_final.pt \
    --epochs 500 \
    --validate-on-2025
  # Expected time: 30-40 minutes on RTX 4090
  ```

- [ ] Verify outputs:
  - [ ] Model file: `models/bayesian/bnn_multioutput_v2_final.pt` (~60-80 KB)
  - [ ] Validation MAE on 2025 < 25 yards ✅
  - [ ] Coverage on 90% CI: 85-95% ✅

### Day 5: Model Registry

- [ ] Create `py/database/register_ensemble_models.py` (150-200 lines)
  - Register all 5 models in `ml.model_registry`
  - Insert ensemble config in `ml.ensemble_configs`

- [ ] Run registration:
  ```bash
  uv run python py/database/register_ensemble_models.py
  ```

- [ ] Verify database:
  ```sql
  SELECT model_name, version, status, is_production
  FROM ml.model_registry
  WHERE version = 'v2';
  -- Should show 5 rows

  SELECT ensemble_name, version, is_active
  FROM ml.ensemble_configs;
  -- Should show 1 row: '5way_ensemble', 'v1', TRUE
  ```

### Day 6-7: 2025 Hold-Out Validation

- [ ] Generate predictions from all 5 models on 2025 weeks 1-6:
  ```bash
  # R models (hierarchical, informative, state-space)
  Rscript R/generate_2025_predictions_v2.R  # Create this script

  # Python models (XGBoost, BNN)
  uv run python py/models/generate_2025_predictions_v2.py  # Create this script
  ```

- [ ] Compute metrics for each model:
  - [ ] MAE (yards)
  - [ ] RMSE (yards)
  - [ ] Coverage (90% CI)
  - [ ] Brier score
  - [ ] Mean interval width

- [ ] Create comparison table:
  ```bash
  uv run python py/analysis/compare_2025_models.py  # Create this script
  # Output: analysis/results/2025_model_comparison.csv
  ```

- [ ] Verify inverse variance weighting is optimal:
  - [ ] Compute weights: `w_i = 1 / (σᵢ² + 1e-6)`
  - [ ] Ensemble mean: `Σ(w_i · μᵢ) / Σw_i`
  - [ ] Ensemble MAE should be < best individual MAE

### ✅ Week 1 Milestone Checklist

- [ ] All 5 models trained on canonical window (2015-2024)
- [ ] All models registered in `ml.model_registry`
- [ ] 2025 validation complete (MAE, coverage computed)
- [ ] Ensemble weights optimized (inverse variance)
- [ ] Decision: Proceed to Week 2? ✅ YES / ❌ NO

---

## Week 2: Ensemble Integration

### Day 8-9: Ensemble Code

- [ ] Create `py/ensemble/enhanced_ensemble_v5.py` (400-500 lines)
  - Class: `EnhancedEnsembleV5`
  - Load all 5 models
  - Implement inverse variance weighting
  - Integrate CQL bet sizer
  - Return comprehensive predictions

- [ ] Write unit tests:
  ```bash
  # Create tests/unit/test_ensemble_v5.py
  pytest tests/unit/test_ensemble_v5.py -v
  ```

- [ ] Test predictions:
  ```python
  from py.ensemble.enhanced_ensemble_v5 import EnhancedEnsembleV5

  ensemble = EnhancedEnsembleV5()
  pred = ensemble.predict(
      player_id="00-0019596",  # Patrick Mahomes
      player_name="P.Mahomes",
      prop_type="passing_yards",
      line=275.5,
      opponent="LV",
      is_home=True,
      season=2025,
      week=7
  )

  # Verify output structure
  assert 'ensemble_mean' in pred
  assert 'ensemble_std' in pred
  assert 'weights' in pred
  assert len(pred['model_means']) == 5
  ```

### Day 10: Database Schema

- [ ] Create `ml.predictions` table:
  ```sql
  -- See full schema in docs/5WAY_ENSEMBLE_INTEGRATION_PLAN.md
  CREATE TABLE ml.predictions (...);
  CREATE INDEX idx_predictions_game ON ml.predictions(game_id);
  CREATE INDEX idx_predictions_player ON ml.predictions(player_id);
  -- etc.
  ```

- [ ] Create `py/database/store_ensemble_predictions.py` (200-250 lines)

- [ ] Test prediction storage:
  ```bash
  # Insert test prediction
  uv run python py/database/store_ensemble_predictions.py \
    --predictions test_predictions.csv \
    --ensemble-id 1 \
    --mode test

  # Verify insert
  psql -c "SELECT COUNT(*) FROM ml.predictions;"
  ```

### Day 11-12: Backtest Preparation

- [ ] Update `py/backtest/historical_validation_5way.py`:
  - Replace state-space placeholder with actual v2 model
  - Update ensemble to 5-way (currently 4-way + placeholder)
  - Add logging for each model prediction

- [ ] Create `py/backtest/ensemble_5way_backtest.py` (300-400 lines)
  - Load ensemble
  - Fetch 2022-2024 data
  - Simulate betting with CQL
  - Track bankroll evolution
  - Compute metrics

- [ ] Test on small dataset:
  ```bash
  uv run python py/backtest/ensemble_5way_backtest.py \
    --start-season 2022 \
    --end-season 2022 \
    --end-week 4 \
    --use-cql \
    --output analysis/results/test_backtest.csv
  # Should complete in < 5 minutes
  ```

### Day 13-14: Full Backtest (2022-2024)

- [ ] Run full backtest:
  ```bash
  uv run python py/backtest/ensemble_5way_backtest.py \
    --start-season 2022 \
    --end-season 2024 \
    --use-cql \
    --bankroll 10000 \
    --output analysis/results/ensemble_5way_backtest.csv
  # Expected time: 2-3 hours (850-900 games)
  ```

- [ ] Monitor progress:
  - Check logs every 30 minutes
  - Verify predictions are reasonable
  - Track bankroll (should be increasing if ROI > 0)

- [ ] Store predictions in database:
  ```bash
  uv run python py/database/store_ensemble_predictions.py \
    --predictions analysis/results/ensemble_5way_backtest.csv \
    --ensemble-id 1 \
    --mode backtest
  ```

- [ ] Generate metrics:
  - [ ] ROI: +5-7% ✅ Target met?
  - [ ] Win rate: 59-61% ✅ Target met?
  - [ ] Sharpe ratio: 2.2-2.6 ✅ Target met?
  - [ ] Max drawdown: -8-10% ✅ Acceptable?
  - [ ] Calibration: 85-95% ✅ Target met?

### ✅ Week 2 Milestone Checklist

- [ ] 5-way ensemble code complete and tested
- [ ] `ml.predictions` table created
- [ ] Full backtest on 2022-2024 complete
- [ ] Metrics meet targets (ROI +5-7%, win rate 59-61%, etc.)
- [ ] Decision: Proceed to Week 3? ✅ YES / ❌ NO

---

## Week 3: Validation & Documentation

### Day 15-16: Ablation Studies

- [ ] Create `py/backtest/ensemble_ablation.py` (200-300 lines)

- [ ] Test ensemble vs each individual model:
  ```bash
  for model in hierarchical informative xgboost bnn state_space; do
    uv run python py/backtest/ensemble_ablation.py \
      --model $model \
      --start-season 2022 \
      --end-season 2024 \
      --output analysis/results/ablation_${model}.csv
  done
  ```

- [ ] Test with/without CQL:
  ```bash
  uv run python py/backtest/ensemble_5way_backtest.py --use-cql  # CQL
  uv run python py/backtest/ensemble_5way_backtest.py            # Kelly only
  ```

- [ ] Test different Kelly fractions:
  ```bash
  for kelly in 0.15 0.25 0.35; do
    uv run python py/backtest/ensemble_5way_backtest.py \
      --kelly-fraction $kelly \
      --output analysis/results/kelly_${kelly}.csv
  done
  ```

- [ ] Statistical significance tests (Diebold-Mariano):
  ```bash
  uv run python py/analysis/diebold_mariano_tests.py \
    --ensemble analysis/results/ensemble_5way_backtest.csv \
    --individuals analysis/results/ablation_*.csv \
    --output analysis/results/dm_tests.csv
  # Verify p-values < 0.05 for all comparisons
  ```

- [ ] Create ablation comparison table:
  - [ ] `analysis/results/ensemble_ablation_summary.csv`

### Day 17: Plots & Visualizations

- [ ] Create `py/analysis/generate_ensemble_plots.py` (300-400 lines)

- [ ] Generate plots:
  ```bash
  uv run python py/analysis/generate_ensemble_plots.py \
    --backtest analysis/results/ensemble_5way_backtest.csv \
    --output analysis/figures/
  ```

- [ ] Plots to create:
  - [ ] `ensemble_5way_bankroll_evolution.pdf` - Bankroll over time
  - [ ] `ensemble_5way_calibration_plot.pdf` - Predicted vs actual
  - [ ] `ensemble_5way_model_weights.pdf` - Weight distribution
  - [ ] `ensemble_5way_ablation_comparison.pdf` - Bar chart comparison

- [ ] Verify plots for dissertation quality:
  - [ ] High resolution (300 DPI)
  - [ ] Clear labels
  - [ ] Consistent color scheme
  - [ ] Legends included

### Day 18-19: Documentation

- [ ] Create `docs/CANONICAL_TRAINING_WINDOW.md` (300-400 lines)
  - Mathematical justification
  - Implementation guide (R and Python)
  - When to use decay vs uniform
  - Ablation study summary

- [ ] Create `docs/ENSEMBLE_WEIGHTING_METHODS.md` (200-300 lines)
  - Inverse variance weighting theory
  - Alternative methods (equal, stacked meta-learner)
  - When to use each method
  - Empirical comparison

- [ ] Create `docs/PRODUCTION_MONITORING_GUIDE.md` (300-400 lines)
  - Dashboard setup (Grafana)
  - Alert configuration (Prometheus)
  - Response protocols
  - Weekly maintenance procedures

- [ ] Update `CLAUDE.md`:
  - Add canonical training window section
  - Update ensemble status to 5-way
  - Update model inventory
  - Add state-space approval details

- [ ] Update `README.md`:
  - Add 5-way ensemble quick start
  - Update performance metrics

- [ ] Update `PROJECT_STATUS.md`:
  - Add Phase 3 completion
  - Add backtest results summary
  - Update next priorities

### Day 20-21: Dissertation Updates

- [ ] Update Chapter 4, Section 4.6 (state-space):
  ```latex
  % Add 4 attempts narrative
  \subsection{State-Space Journey: 4 Attempts}
  \begin{enumerate}
      \item LOESS Approximation: Failed (not true Bayesian)
      \item AR(1) Hierarchical: Failed (convergence issues)
      \item AR(1) Pooled: Failed (identifiability)
      \item Random Walk Pooled: Success ✅ (R >> Q, simple, converges)
  \end{enumerate}

  % Add training window ablation
  \subsection{Training Window Selection}
  % Table: 4 scenarios, MAE, coverage, p-values
  ```

- [ ] Update Chapter 5, Section 5.4 (informative priors):
  ```latex
  % Add empirical Bayes prior estimation
  \subsection{Empirical Bayes Priors}
  % Use 2015-2019 to calibrate priors for 2015-2024 training
  ```

- [ ] Update Chapter 8, Section 8.5 (ensemble results):
  ```latex
  % Add 5-way ensemble results
  \subsection{5-Way Ensemble Performance}
  \begin{table}
      \caption{5-Way Ensemble Results (2022-2024)}
      % ROI, win rate, Sharpe, max drawdown, calibration
  \end{table}

  % Add ablation studies
  \subsection{Ablation Studies}
  % Table: Ensemble vs each individual model, p-values
  ```

- [ ] Create Appendix C (ablation study):
  ```latex
  % appendix/appendix_c_ablation_study.tex
  \chapter{Training Window Ablation Study}
  % Full ablation study from docs/TRAINING_WINDOW_ABLATION_RESULTS.md
  % Include: 4 scenarios, convergence diagnostics, statistical tests
  ```

- [ ] Compile dissertation:
  ```bash
  cd analysis/dissertation/main
  pdflatex main.tex
  bibtex main
  pdflatex main.tex
  pdflatex main.tex
  # Check for errors
  grep -i "warning\|error" main.log
  ```

### ✅ Week 3 Milestone Checklist

- [ ] Ablation studies complete (ensemble beats all individuals, p < 0.05)
- [ ] All plots generated and dissertation-ready
- [ ] All 3 technical docs written (canonical window, weighting, monitoring)
- [ ] All 6 updates complete (CLAUDE.md, README, PROJECT_STATUS, dissertation)
- [ ] Dissertation compiles without errors
- [ ] Decision: Proceed to production deployment? ✅ YES / ❌ NO

---

## Week 4+: Production Deployment

### Phase 1: Shadow Mode (Days 22-28)

- [ ] Create `api/ensemble_service.py` (200-300 lines)
  - FastAPI endpoints: `/predict`, `/health`, `/metrics`
  - Load 5-way ensemble
  - Return predictions (but don't bet)

- [ ] Create `docker-compose.yml` service:
  ```yaml
  services:
    prediction-service:
      build: .
      ports:
        - "8000:8000"
      volumes:
        - ./models:/app/models
      environment:
        - DATABASE_URL=postgresql://...
      command: uvicorn api.ensemble_service:app
  ```

- [ ] Deploy:
  ```bash
  docker compose up -d prediction-service
  ```

- [ ] Monitor shadow mode:
  - [ ] Latency < 1 second (p95) ✅
  - [ ] Error rate < 1% ✅
  - [ ] Predictions match backtest quality ✅

### Phase 2: A/B Testing (Days 29-42)

- [ ] Create `py/production/ab_test_manager.py` (200-300 lines)
  - Thompson sampling for dynamic allocation
  - Track performance: 4-way vs 5-way
  - Decision rule: If 5-way ROI > 4-way after 100+ bets → full rollout

- [ ] Start A/B test:
  ```bash
  uv run python py/production/ab_test_manager.py \
    --allocation 0.9,0.1 \
    --min-bets 100 \
    --confidence 0.95
  # 90% bankroll: 4-way ensemble
  # 10% bankroll: 5-way ensemble
  ```

- [ ] Monitor daily:
  ```bash
  uv run python py/production/ab_test_report.py
  # Shows: ROI (4-way), ROI (5-way), p-value, decision
  ```

- [ ] Decision criteria (after 100+ bets):
  - [ ] If 5-way ROI > 4-way ROI (p < 0.05) → Full rollout ✅
  - [ ] If 5-way ROI ≈ 4-way ROI → Continue A/B test
  - [ ] If 5-way ROI < 4-way ROI (p < 0.05) → Investigate issues ❌

### Phase 3: Full Deployment (Day 43+)

- [ ] Full rollout:
  ```bash
  uv run python py/production/ab_test_manager.py \
    --allocation 0.0,1.0
  # 0% bankroll: 4-way ensemble
  # 100% bankroll: 5-way ensemble
  ```

- [ ] Set up Grafana dashboards:
  ```bash
  docker compose up -d grafana prometheus
  # Access: http://localhost:3000
  ```

- [ ] Configure dashboards:
  - [ ] Real-Time Performance (ROI, win rate, P&L, bankroll)
  - [ ] Model Health (calibration, MAE, weights, latency)
  - [ ] Betting Activity (bets/day, edge, risk gates)

- [ ] Configure Prometheus alerts:
  ```yaml
  # prometheus/alerts.yml
  groups:
    - name: ensemble_alerts
      rules:
        - alert: ROI_Too_Low
          expr: rolling_roi_7d < 0.03
          for: 1h
          labels:
            severity: warning
          annotations:
            summary: "ROI dropped below 3% (7-day rolling)"

        - alert: Calibration_Out_Of_Range
          expr: coverage_90 < 0.80 OR coverage_90 > 0.97
          for: 6h
          labels:
            severity: warning
          annotations:
            summary: "90% CI coverage out of range 80-97%"
  ```

- [ ] Create weekly retraining script:
  ```bash
  # scripts/weekly_retrain.sh
  #!/bin/bash
  # Fetch latest data
  Rscript data/ingest_pbp.R
  # Refresh views
  psql -c "REFRESH MATERIALIZED VIEW mart.game_summary;"
  # Retrain all 5 models
  Rscript R/train_hierarchical_v2_canonical.R
  Rscript R/train_informative_priors_v2_canonical.R
  uv run python py/models/train_xgboost_canonical.py
  uv run python py/models/train_bnn_canonical.py
  # Validate
  uv run python py/validation/validate_ensemble_v5.py
  # Restart service
  docker compose restart prediction-service
  ```

- [ ] Schedule weekly retraining:
  ```bash
  # crontab -e
  0 6 * * 1 /path/to/scripts/weekly_retrain.sh >> /var/log/weekly_retrain.log 2>&1
  # Runs every Monday at 6 AM
  ```

### ✅ Production Deployment Checklist

- [ ] Shadow mode complete (latency < 1s, error rate < 1%)
- [ ] A/B testing complete (5-way ROI > 4-way ROI, p < 0.05)
- [ ] Full rollout complete (100% bankroll on 5-way)
- [ ] Grafana dashboards operational
- [ ] Prometheus alerts configured
- [ ] Weekly retraining automated
- [ ] Real-time monitoring active

---

## Quick Commands Reference

### Training Commands

```bash
# R models (M4 CPU)
Rscript --vanilla R/train_hierarchical_v2_canonical.R           # 15-20 min
Rscript --vanilla R/train_informative_priors_v2_canonical.R     # 20-25 min
# State-space already trained ✅

# Python models (RTX 4090 GPU)
uv run python py/models/train_xgboost_canonical.py              # 10-15 min
uv run python py/models/train_bnn_canonical.py                  # 30-40 min
```

### Backtest Commands

```bash
# Full backtest (2022-2024)
uv run python py/backtest/ensemble_5way_backtest.py \
  --start-season 2022 --end-season 2024 --use-cql \
  --output analysis/results/ensemble_5way_backtest.csv          # 2-3 hours

# Ablation (individual models)
uv run python py/backtest/ensemble_ablation.py --model xgboost  # Per model
```

### Database Commands

```bash
# Register models
uv run python py/database/register_ensemble_models.py

# Store predictions
uv run python py/database/store_ensemble_predictions.py \
  --predictions analysis/results/ensemble_5way_backtest.csv \
  --ensemble-id 1

# Query predictions
psql -c "SELECT COUNT(*) FROM ml.predictions;"
```

### Production Commands

```bash
# Start API service
docker compose up -d prediction-service

# A/B testing
uv run python py/production/ab_test_manager.py --allocation 0.9,0.1

# Weekly retraining
bash scripts/weekly_retrain.sh
```

---

## Success Metrics

### Performance (Must Meet)

| Metric | Target | Status |
|--------|--------|--------|
| ROI | +5-7% | [ ] |
| Win Rate | 59-61% | [ ] |
| Sharpe Ratio | 2.2-2.6 | [ ] |
| Max Drawdown | -8-10% | [ ] |
| Calibration | 85-95% | [ ] |
| Ensemble MAE | < Best Individual | [ ] |

### Ablation (Must Pass)

| Test | Target | Status |
|------|--------|--------|
| Ensemble vs Hierarchical | ROI +1pp | [ ] |
| Ensemble vs Informative | ROI +1pp | [ ] |
| Ensemble vs XGBoost | ROI +1pp | [ ] |
| Ensemble vs BNN | ROI +1pp | [ ] |
| Ensemble vs State-Space | ROI +1pp | [ ] |
| Statistical Significance | p < 0.05 | [ ] |

### Production (Must Achieve)

| Metric | Target | Status |
|--------|--------|--------|
| Uptime | 99.5%+ | [ ] |
| Latency (p95) | < 1 second | [ ] |
| Error Rate | < 1% | [ ] |
| Dashboards | Operational | [ ] |
| Alerts | Configured | [ ] |
| Weekly Retraining | Automated | [ ] |

---

## Contact

**Document**: Implementation Checklist for 5-Way Ensemble
**Version**: 1.0
**Date**: November 8, 2025
**Author**: Claude Code (Sonnet 4.5)

**Related Documents**:
- Full Plan: `docs/5WAY_ENSEMBLE_INTEGRATION_PLAN.md`
- Executive Summary: `docs/5WAY_ENSEMBLE_EXECUTIVE_SUMMARY.md`

**Status**: ✅ Ready for user approval to proceed with Week 1
