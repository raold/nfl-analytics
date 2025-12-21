# Research to Output Mapping

**Last Updated:** 2025-10-31
**Purpose:** Map research models to production frontend and dissertation chapters

---

## Frontend Integration (dashboard/app.py)

### Currently Powering Frontend (PRODUCTION)

| Frontend Feature | Model | Status | Data Flow |
|-----------------|-------|--------|-----------|
| **Live Game Predictions** | Game Outcome Ensemble v2025 | ✅ LIVE | Model → `predictions.game_predictions` → Dashboard |
| **Win Rate Tracking** | Game Outcome Ensemble v2025 | ✅ LIVE | Model → `predictions.retrospectives` → Dashboard charts |
| **Calibration Curves** | Game Outcome Ensemble v2025 | ✅ LIVE | `home_win_prob` → Calibration plot |
| **MAE/Error Distributions** | Game Outcome Ensemble v2025 | ✅ LIVE | `margin_error`, `abs_margin_error` → Charts |
| **ATS Accuracy** | Game Outcome Ensemble v2025 | ✅ LIVE | `actual_margin + spread_close` → ATS tracking |
| **Model Comparison** | Multiple versions | ✅ LIVE | `model_version` field allows version comparison |

### Not Yet Integrated (Research Models)

| Research Model | Potential Frontend Feature | Integration Effort | Priority |
|----------------|---------------------------|-------------------|----------|
| **BNN Multioutput v1** | Uncertainty visualization (confidence bands) | Medium (2-3h) | HIGH |
| **BNN Mixture/Simpler** | Alternative prediction sources for comparison | Low (1-2h) | MEDIUM |
| **BNN Environment** | Contextual factors display (weather, rest impact) | Medium (2-3h) | HIGH |
| **GNN Hierarchical** | Team relationship graph visualization | High (4-6h) | LOW |
| **cVAE** | Game trajectory simulations (animated predictions) | Very High (8-10h) | LOW |
| **RL Agents** | Optimal bet sizing recommendations | Medium (3-4h) | HIGH |
| **Causal Framework** | Treatment effect analysis (e.g., rest impact) | Medium (3-4h) | MEDIUM |

### Proposed Frontend Enhancements

**High Priority** (Models ready, clear value):
1. **Uncertainty Bands (BNN Multioutput)**
   - Display: Add confidence intervals to predictions
   - Value: Users see prediction confidence, not just point estimates
   - Integration: Add `prediction_lower`, `prediction_upper` columns to database

2. **Optimal Bet Sizing (RL Policy)**
   - Display: Show recommended bet size based on RL agent
   - Value: Users get adaptive bet sizing, not just fixed Kelly
   - Integration: Add `recommended_bet_size` column from RL policy

3. **Contextual Factors (BNN Environment)**
   - Display: Show impact of weather, rest days on prediction
   - Value: Users understand situational factors affecting game
   - Integration: Add `weather_impact`, `rest_impact` columns

**Medium Priority** (Needs evaluation first):
4. **Model Ensemble Comparison**
   - Display: Side-by-side comparison of Ensemble, BNN, GNN predictions
   - Value: Users see model agreement/disagreement
   - Integration: Multi-model prediction pipeline

5. **Causal Analysis Page**
   - Display: Interactive treatment effect analysis
   - Value: Users understand WHY factors matter, not just correlation
   - Integration: New dashboard page with causal graphs

**Low Priority** (Exploratory):
6. **Game Trajectory Simulation (cVAE)**
   - Display: Animated game progression with multiple scenarios
   - Value: Entertainment, scenario planning
   - Integration: cVAE inference endpoint, animation library

7. **Team Relationship Graph (GNN)**
   - Display: Interactive network graph of team similarities
   - Value: Visual exploration of team dynamics
   - Integration: Graph visualization library (e.g., Plotly network)

---

## Dissertation Chapter Mapping

### Chapter Structure

| Chapter | Focus | Models/Findings | Status |
|---------|-------|----------------|--------|
| 1. Introduction | Motivation, problem statement | N/A | ✅ Written |
| 2. Literature Review | Prior work in sports analytics | N/A | ✅ Written |
| 3. Data Foundation | Data pipeline, features | All models use this | ✅ Written |
| 4. Baseline Modeling | Traditional ML + Ensemble | Game Outcome Ensemble, State-Space, XGBoost | ✅ Partially written |
| 5. RL Design | Reinforcement learning for betting | CQL, IQL, DQN, PPO | ⚠️ RL not recently trained |
| 6. Uncertainty & Risk | Bayesian methods, betting | BNN variants, Conformal Prediction | ⚠️ Needs BNN consolidation |
| 7. Simulation | Generative models | cVAE | ⚠️ Needs use case clarity |
| 8. Results & Discussion | Empirical findings | All models | ⚠️ Needs updates |
| 9. Production Deployment | Engineering, monitoring | Game Outcome Ensemble v2025 | ✅ Recent deployment |
| 10. Conclusion | Summary, limitations | N/A | ❓ To be written |
| 11. Market Optimization | Advanced betting strategies | RL, Causal | ❓ Future work |
| 12. Profit Optimization | ROI maximization | Ensemble + RL | ❓ Future work |

### Detailed Model-to-Chapter Mapping

#### Chapter 3: Data Foundation
**All Models** use features documented here:
- Team season statistics (prior season only)
- State-space ratings (Kalman filter)
- Phase 5 advanced features:
  - ✅ `away_short_rest` (legitimate)
  - ✅ `is_conference_game` (legitimate)
  - ❌ Removed: `extreme_cold`, `high_wind`, `dome_team_outdoor_wind`

**Dissertation Content:**
- Feature engineering methodology
- Look-ahead bias prevention
- Master features table (`table_master_features.tex`)

---

#### Chapter 4: Baseline Modeling

**Section 4.1: State-Space Model**
- **Model:** Kalman filter team ratings
- **File:** `py/models/state_space.py`
- **Performance:** Component of ensemble (60.8% train accuracy standalone)
- **Dissertation Content:** `section_state_space_model.tex`
- **Findings:** Dynamic ratings capture team evolution over season

**Section 4.2: XGBoost Model**
- **Model:** Regularized XGBoost (α=5.0, λ=5.0)
- **File:** `py/models/xgboost_gpu_v3.py`
- **Performance:** 70.5% train accuracy (strongest single model)
- **Dissertation Content:** Part of ensemble section
- **Findings:** Strong regularization prevents overfitting

**Section 4.3: Game Outcome Ensemble**
- **Model:** v2025 Optimized Ensemble
- **File:** `py/ensemble/game_outcome_ensemble_realistic.py`
- **Performance:** ✅ **60.7% test accuracy, 2.8% calibration error**
- **Dissertation Content:** `section_game_ensemble.tex`
- **Key Findings:**
  - Regularization study: α/λ 2.0 → 5.0 improves calibration by 70%
  - Feature ablation: Removing 3 harmful features improves performance
  - Platt Scaling: Reduces calibration error from 7.0% → 2.8%
  - Temporal validation: No degradation over 2021-2024 test periods
- **Tables/Figures:**
  - Regularization comparison table
  - Feature ablation results
  - Calibration curves (before/after Platt Scaling)
  - Temporal validation results

**Section 4.4: GNN Architecture & Ablation**
- **Models:** 9 GNN configurations
- **Files:**
  - `py/models/hierarchical_gnn_v1.py`
  - `py/models/gnn_ablation_study.py`
- **Performance:** ⚠️ **46.9-50.2% test accuracy (poor)**
- **Dissertation Content:**
  - `section_gnn_ablation_study.tex`
  - `section_gnn_optuna_hyperparameter_search.tex`
  - `section_gnn_gpu_optimization.tex`
- **Key Findings:**
  - Best config: Large Model (embed=128, hidden=256, msg_rounds=5, val Brier 0.215)
  - All configs underperform simple ensemble
  - Graph structure may not be directly predictive for game outcomes
  - **Recommendation:** Use GNN embeddings as features, not standalone model
- **Tables/Figures:**
  - Ablation study results (9 configs)
  - Hyperparameter search convergence
  - GPU optimization speedup

**Section 4.5: Ensemble v3.0 (Enhanced)**
- **Model:** Enhanced Ensemble v3.0
- **File:** `py/ensemble/enhanced_ensemble_v3.py`
- **Performance:** ❌ **0% ROI (0 predictions generated in backtest)**
- **Dissertation Content:** `v3_ensemble_section.tex`
- **Status:** NEEDS DEBUGGING
- **Findings:** Failure case study - what went wrong and how to detect

---

#### Chapter 5: RL Design

**All RL Agents:**
- **Models:** CQL, IQL, DQN, PPO
- **Files:** `py/models/cql_agent.py`, etc.
- **Performance:** ⚠️ **Status unclear** (not recently trained)
- **Dissertation Content:** `chapter_5_rl_design.tex`
- **Status:** ❌ **INCOMPLETE** - RL agents need fresh training

**Next Steps for Chapter 5:**
1. Audit existing RL code (still functional?)
2. Define RL problem properly:
   - State: Bankroll, ensemble predictions, market odds, week, team stats
   - Action: Bet size (0-100% bankroll), bet side (home/away/none)
   - Reward: Bankroll change (log for risk-adjusted), drawdown penalty
3. Train fresh RL agents on 2022-2024 data
4. Backtest RL policy vs fixed Kelly criterion
5. Document in dissertation:
   - RL formulation
   - Training methodology
   - Comparison to baseline betting strategies
   - Adaptive bankroll management results

---

#### Chapter 6: Uncertainty, Risk & Betting

**Section 6.1: Bayesian Neural Networks**

**BNN Multioutput v1:**
- **File:** `py/models/bnn_multioutput_v1.py`
- **Performance:** ✅ **92% CI coverage (rushing yards), 58.3% TD accuracy**
- **Status:** Trained, needs integration
- **Dissertation Content:**
  - Multi-task learning (yards + TDs jointly)
  - Uncertainty quantification via Bayesian inference
  - Calibration study (90% CI coverage validation)
- **Findings:**
  - Multi-output modeling preserves calibration
  - Mixture of experts provides expert specialization
  - Uncertainty useful for bet sizing
- **Tables/Figures:**
  - Calibration curves (90% CI coverage)
  - Multi-output correlation analysis
  - Expert utilization (for Mixture variant)

**BNN Mixture of Experts v2:**
- **File:** `py/models/bnn_mixture_experts_v2.py`
- **Performance:** ❓ **Not yet evaluated**
- **Status:** Trained (1.3 GB model), needs evaluation
- **Dissertation Content:**
  - Expert specialization analysis
  - Comparison to Multioutput v1
  - When do different experts activate?

**BNN Environment:**
- **File:** `py/models/train_bnn_environment.py`
- **Performance:** ❌ **29.7% CI coverage (target 90%)**
- **Status:** Trained, poor calibration
- **Dissertation Content:**
  - Failure case study: Why poor calibration?
  - Repurposing as feature extractor (not standalone model)
  - Contextual factor extraction
- **Findings:**
  - Poor standalone calibration
  - May still provide useful contextual features for ensemble

**BNN Simpler v2:**
- **File:** `py/models/bnn_simpler_v2.py`
- **Performance:** ❓ **Not yet evaluated**
- **Status:** Trained (10 MB, lightweight)
- **Dissertation Content:**
  - Speed vs accuracy tradeoff
  - Real-time inference suitability

**BNN Prior Sensitivity Study:**
- **Files:** `bnn_rushing_sigma{0.5,0.7,1.0,1.5}.pkl`
- **Performance:** Research study (varying prior std)
- **Status:** ✅ Complete
- **Dissertation Content:** `section_bnn_calibration_study.tex`
- **Findings:**
  - Prior std sensitivity analysis
  - Calibration robustness to hyperparameters
- **Tables/Figures:**
  - `figures/out/bnn_prior_sensitivity_table.tex`

**Section 6.2: Conformal Prediction**
- **Status:** ⚠️ Partially implemented
- **Dissertation Content:**
  - Uncertainty quantification framework
  - Comparison to Bayesian CI
  - Coverage guarantees

**Section 6.3: Betting Strategy**
- **Models:** Game Outcome Ensemble v2025 + Conservative criteria
- **Dissertation Content:** `section_profit_optimization_results.tex`
- **Key Findings:**
  - Conservative betting criteria:
    - 5% minimum edge (up from 3%)
    - 10% max Kelly fraction (down from 25%)
    - Probability clipping [15%, 85%]
  - Reduces bet frequency from 67% → 33% of games
  - Expected 10-15% annual ROI, Sharpe 1.2-1.5
- **Tables/Figures:**
  - Betting criteria comparison
  - ROI vs bet frequency tradeoff
  - Sharpe ratio optimization

---

#### Chapter 7: Simulation

**Section 7.1: cVAE Game Trajectories**
- **Model:** Game Trajectory cVAE
- **File:** `py/models/train_cvae.py`
- **Performance:** Test loss 62.55, reconstruction adequate
- **Status:** ⚠️ **USE CASE UNDEFINED**
- **Dissertation Content:** `chapter_7_simulation.tex`
- **Possible Findings** (if use case defined):
  - Game trajectory generation for Monte Carlo uncertainty
  - Rare event modeling (blowouts, overtime)
  - Counterfactual analysis ("what if" scenarios)
- **Tables/Figures:**
  - Reconstruction quality metrics
  - Latent space visualization
  - Trajectory samples

**Next Steps for Chapter 7:**
1. Define 1-2 concrete cVAE experiments:
   - Option A: Use for uncertainty (alternative to BNN)
   - Option B: Use for rare event simulation
   - Option C: Use for counterfactual analysis
2. Run experiments
3. Document findings

---

#### Chapter 8: Results & Discussion

**Section 8.1: State-Space Results**
- **Content:** `section_state_space_results.tex`
- **Models:** Kalman filter team ratings
- **Findings:** Dynamic team strength evolution

**Section 8.2: GNN Empirical Results**
- **Content:** `section_gnn_empirical_results.tex`
- **Models:** 9 GNN configurations
- **Findings:** Poor standalone performance, potential as feature extractor

**Section 8.3: BNN Calibration Study**
- **Content:** `section_bnn_calibration_study.tex`
- **Models:** BNN variants (Multioutput, Mixture, Environment, Prior sensitivity)
- **Findings:**
  - Multioutput v1: 92% CI coverage ✅
  - Environment: 29.7% CI coverage ❌
  - Prior sensitivity: Robust to hyperparameters

**Section 8.4: Causal Inference**
- **Content:** `section_causal_inference.tex`
- **Models:** Causal framework (`py/causal/`)
- **Status:** ⚠️ **NOT YET APPLIED**
- **Potential Findings:**
  - Does short rest CAUSE worse performance?
  - Dome team outdoors CAUSAL effect
  - Extreme cold CAUSAL impact
- **Tables/Figures:**
  - `figures/out/causal_dag_rushing.tex`
  - `figures/out/causal_treatment_effects_table.tex`
  - `figures/out/causal_betting_backtest_table.tex`

**Section 8.5: Ensemble Comparison**
- **Models:** v1.0 vs v2.0 vs v2025 vs v3.0
- **Findings:**
  - v2025 optimized: 60.7% accuracy, 2.8% calibration error ✅
  - v3.0 enhanced: 0% ROI (failed) ❌
  - Importance of regularization, feature ablation, calibration

---

#### Chapter 9: Production Deployment

**Section 9.1: Deployment Architecture**
- **Model:** Game Outcome Ensemble v2025
- **Content:** `chapter_9_production_deployment.tex`
- **Findings:**
  - Deployed 2025-10-30 for weeks 1-7 predictions
  - 107 games predicted, 35 bets recommended (32.7%)
  - Production-ready: 60.7% accuracy, 2.8% calibration error
  - Conservative betting: 5% min edge, 10% max Kelly
  - Monitoring dashboard (Streamlit)

**Section 9.2: Monitoring & Observability**
- **Frontend:** `dashboard/app.py`
- **Features:**
  - Live game tracking with predictions
  - Calibration monitoring
  - Performance tracking (win rate, MAE, ATS)
  - Model version comparison

**Section 9.3: Engineering Challenges**
- **Docker workflow** (to be documented)
- **Mac-to-Desktop training pipeline**
- **Model versioning and registry**

---

#### Chapter 10: Conclusion

**Content:** `chapter_10_conclusion.tex`
- Summary of all findings
- Limitations:
  - GNN poor standalone performance
  - cVAE use case undefined
  - RL agents need fresh training
- Future work:
  - BNN integration with ensemble
  - RL policy optimization
  - Causal analysis application

---

#### Chapter 11: Market Optimization (Future Work)

**Content:** `chapter_11_market_optimization.tex`
- Advanced betting strategies
- Market inefficiency detection
- Closing line value (CLV) optimization
- **Models:** RL agents (when trained)

---

#### Chapter 12: Profit Optimization (Future Work)

**Content:** `chapter_12_profit_optimization.tex`
- ROI maximization
- Bankroll management
- Multi-objective optimization (ROI vs risk)
- **Models:** Ensemble + RL + BNN uncertainty

---

## Summary: Dissertation Readiness by Model

| Model | Dissertation Ready? | Chapter | What's Missing |
|-------|-------------------|---------|----------------|
| **Game Outcome Ensemble v2025** | ✅ YES | 4, 8, 9 | Nothing - deployed and documented |
| **State-Space Ratings** | ✅ YES | 4, 8 | Nothing - component of ensemble |
| **XGBoost** | ✅ YES | 4 | Nothing - component of ensemble |
| **GNN Ablation** | ✅ YES | 4, 8 | Nothing - complete study |
| **BNN Multioutput v1** | ⚠️ PARTIAL | 6, 8 | Needs integration testing, ensemble comparison |
| **BNN Mixture v2** | ❌ NO | 6 | Needs evaluation |
| **BNN Simpler v2** | ❌ NO | 6 | Needs evaluation |
| **BNN Environment** | ⚠️ PARTIAL | 6, 8 | Documented as failure case, needs repurposing plan |
| **BNN Prior Sensitivity** | ✅ YES | 6, 8 | Complete, table generated |
| **cVAE** | ❌ NO | 7 | Needs use case definition and experiments |
| **RL Agents** | ❌ NO | 5, 11, 12 | Needs fresh training and evaluation |
| **Causal Framework** | ❌ NO | 8 | Needs application to research questions |
| **Ensemble v3.0** | ⚠️ PARTIAL | 4, 8 | Documented as failure case (0% ROI) |
| **Conformal Prediction** | ❌ NO | 6 | Partial implementation only |

---

## Action Plan: Prioritize by Output

### For Frontend (Production)
**Priority Order:**
1. BNN Multioutput uncertainty bands (HIGH value, medium effort)
2. BNN Environment contextual factors (HIGH value, medium effort)
3. RL optimal bet sizing (HIGH value, medium effort)
4. Causal analysis page (MEDIUM value, medium effort)
5. Model ensemble comparison (MEDIUM value, low effort)

### For Dissertation (Graduation)
**Priority Order:**
1. **Evaluate BNN variants** (Mixture, Simpler) - Need for Chapter 6
2. **Define cVAE experiments** - Critical for Chapter 7
3. **Apply causal framework** - Critical for Chapter 8.4
4. **Train RL agents** - Critical for Chapter 5
5. **Debug Ensemble v3.0** - Failure case study for Chapter 4

### Quick Wins (Both Frontend + Dissertation)
1. BNN Multioutput evaluation → Chapter 6 + Frontend uncertainty
2. BNN Environment repurposing → Chapter 6 + Frontend context
3. Causal analysis (1-2 questions) → Chapter 8 + Frontend insights

---

**Last Updated:** 2025-10-31
**Maintained By:** Research Team
**Related Documents:**
- `MODEL_MATRIX.md` - Comprehensive model status tracking
- `RESEARCH_VECTORS.md` - Detailed research vector plans (to be created)
- `INTEGRATION_MATRIX.json` - Machine-readable integration opportunities (to be created)
