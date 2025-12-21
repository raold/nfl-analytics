# NFL Analytics Model Matrix

**Last Updated:** 2025-10-31
**Purpose:** Comprehensive tracking of all models across research vectors

---

## Quick Navigation

- [Status Matrix](#status-matrix) - What's done, what's not
- [Strengths & Weaknesses Matrix](#strengths--weaknesses-matrix) - Capabilities of each model
- [Combination Opportunities Matrix](#combination-opportunities-matrix) - Integration potential
- [Hardware Requirements](#hardware-requirements) - M4 Mac vs 4090 Desktop needs

---

## Status Matrix

### Production Models (The Hub)

| Model | Training | Calibration | Ablation | Ensemble Integration | Production | Next Steps |
|-------|----------|-------------|----------|---------------------|------------|------------|
| **Game Outcome Ensemble v2025** | ✅ Complete | ✅ Excellent (2.8%) | ✅ Complete | N/A (is hub) | ✅ **LIVE** | Monitor week 8+ performance |

**v2025 Details:**
- **Performance:** 60.7% accuracy, 2.8% calibration error, 10-15% ROI expected
- **Components:** State-Space (Kalman), XGBoost (α=5.0, λ=5.0), Bayesian LogReg, Neural Net, Meta-Learner
- **Training Data:** 1,390 games (2019-2023), calibrated on 285 games (2024)
- **Deployment:** 2025-10-30, generating predictions for weeks 1-7
- **File:** `py/ensemble/game_outcome_ensemble_realistic.py`
- **Artifacts:** `analysis/OPTIMIZED_ENSEMBLE_2025_DEPLOYMENT.md`

---

### Bayesian Neural Networks (Research Spoke - ACTIVE)

| Model | Size | Training | Calibration | Ablation | Integration | Production | Next Steps |
|-------|------|----------|-------------|----------|-------------|------------|------------|
| **BNN Multioutput v1** | 1.6 GB | ✅ Complete | ⚠️ Partial (92% CI rushing) | ❌ Needed | ❌ Not tested | ❌ No | 1. Full ablation study<br>2. Test as ensemble feature<br>3. Compare to Mixture v2 |
| **BNN Mixture of Experts v2** | 1.3 GB | ✅ Complete | ❌ Not done | ❌ Needed | ❌ Not tested | ❌ No | 1. **Evaluate performance**<br>2. Compare to Multioutput<br>3. Calibration study |
| **BNN Environment** | 302 MB | ✅ Complete | ❌ Failed (29.7% CI) | ❌ Needed | ❌ Not tested | ❌ No | 1. Fix calibration OR<br>2. **Test as ensemble feature** |
| **BNN Simpler v2** | 10 MB | ✅ Complete | ❌ Not done | ❌ Needed | ❌ Not tested | ❌ No | 1. **Evaluate speed/accuracy**<br>2. Test for real-time inference |
| **BNN Deeper v2** | ? | ⚠️ Unclear | ❌ Not done | ❌ Needed | ❌ Not tested | ❌ No | 1. Verify training status<br>2. Compare to other variants |
| **BNN Prior Sensitivity** | 181 MB × 4 | ✅ Research | N/A | ✅ Complete | N/A | ❌ No | Research artifact (σ=0.5/0.7/1.0/1.5) |

**BNN Summary:**
- **Strengths:** Uncertainty quantification, multi-task learning, principled Bayesian approach
- **Current State:** Multiple variants trained, unclear which is best
- **Blocker:** Need systematic comparison and integration testing
- **Priority:** HIGH - Multioutput v1 shows promise (92% CI coverage for rushing yards)

---

### Graph Neural Networks (Research Spoke - NEEDS REVISION)

| Model | Training | Calibration | Ablation | Integration | Production | Next Steps |
|-------|----------|-------------|----------|-------------|------------|------------|
| **GNN Hierarchical (9 configs)** | ✅ Complete | ❌ Not done | ✅ **Complete** | ❌ Not tested | ❌ No | 1. Test as feature extractor<br>2. Redesign architecture OR<br>3. Pivot to different graph |

**GNN Ablation Results (models/gnn/ablation/ablation_results.json):**
- **Best Config:** Large Model (embed=128, hidden=256, msg_rounds=5)
  - Val Brier: 0.215 (7.9% better than baseline)
  - **Test Accuracy: 46.9%** ⚠️ (worse than ensemble 60.7%)
  - Test AUC: 0.548
- **Baseline:** Full Model (embed=64, hidden=128, msg_rounds=3)
  - Val Brier: 0.233
  - Test Accuracy: 50.2%
  - Test AUC: 0.555

**GNN Analysis:**
- **Strengths:** Captures relational structure between teams
- **Weaknesses:** Poor standalone performance (< 51% accuracy on all configs)
- **Insight:** Graph structure may not be directly predictive
- **Pivot Options:**
  1. Use GNN node embeddings as features for ensemble (**RECOMMENDED NEXT**)
  2. Redesign: Player-level graphs instead of team-level
  3. Redesign: Play-by-play temporal graphs
  4. Shelve until BNN consolidation complete

---

### Conditional Variational Autoencoder (Research Spoke - USE CASE UNDEFINED)

| Model | Training | Calibration | Ablation | Integration | Production | Next Steps |
|-------|----------|-------------|----------|-------------|------------|------------|
| **cVAE Game Trajectory** | ✅ Complete | N/A | ❌ Needed | ❌ Not tested | ❌ No | 1. **Define concrete use case**<br>2. Pilot experiments |

**cVAE Training Results (models/cvae/training_history.json):**
- **Test Loss:** 62.55
- **Best Val Loss:** 60.47 (epoch 94)
- **Metrics:**
  - Reconstruction: 57.86
  - KL Divergence: 4.70
  - Score: 47.92
  - Event: 0.26
  - Momentum: 9.68
- **Training Data:** 1,343 games (2020-2024), 100 epochs

**Possible Use Cases:**
1. **Game Trajectory Simulation:** Generate synthetic game progressions for Monte Carlo uncertainty
2. **Rare Event Modeling:** Generate training data for uncommon game scenarios
3. **Counterfactual Analysis:** "What if" scenario generation (e.g., different weather, different rest)
4. **Uncertainty Quantification:** Alternative to conformal prediction
5. **Anomaly Detection:** Identify unusual game patterns

**Recommendation:** Define 1-2 concrete experiments before further development

---

### Reinforcement Learning (Research Spoke - DORMANT, HIGH POTENTIAL)

| Model | Training | Calibration | Ablation | Integration | Production | Next Steps |
|-------|----------|-------------|----------|-------------|------------|------------|
| **CQL Agent** | ⚠️ Old (unclear) | N/A | ❌ Not done | ❌ Not tested | ❌ No | 1. **Audit existing code**<br>2. Define state/action/reward<br>3. Fresh training |
| **IQL Agent** | ⚠️ Old (unclear) | N/A | ❌ Not done | ❌ Not tested | ❌ No | Same as CQL |
| **DQN Agent** | ⚠️ Old (unclear) | N/A | ❌ Not done | ❌ Not tested | ❌ No | Same as CQL |
| **PPO Agent** | ⚠️ Old (unclear) | N/A | ❌ Not done | ❌ Not tested | ❌ No | Same as CQL |

**Why RL Has High Potential:**
- **Perfect Fit:** Betting is sequential decision-making under uncertainty
- **Adaptive:** Can learn optimal bankroll management, Kelly fraction adjustments
- **Market Adaptation:** Can adapt to changing betting market conditions
- **Policy Optimization:** Better than fixed rules (e.g., fixed 5% edge, 10% Kelly)

**Why Dormant:**
- We pivoted to BNN/GNN research instead
- Unclear if existing code still works
- Need to properly define RL problem:
  - **State:** Current bankroll, ensemble predictions, market odds, week, team stats
  - **Action:** Bet size (0-100% of bankroll), bet side (home/away/none)
  - **Reward:** Bankroll change (log for risk-adjusted), drawdown penalty

**Priority:** MEDIUM-HIGH (high potential, but needs investigation first)

**Next Steps:**
1. Audit `models/cql/`, `models/iql/`, `models/dqn/` - do they work?
2. Review `py/models/` for RL training scripts
3. Define RL problem properly (state, action, reward)
4. Train fresh RL agents on 2022-2024 data
5. Backtest RL policy with current ensemble predictions
6. **Estimated Effort:** 8-12 hours

---

### Causal Inference (Research Spoke - CODE COMPLETE, NOT APPLIED)

| Component | Status | Next Steps |
|-----------|--------|------------|
| **Structural Causal Models** | ✅ Code complete (`py/causal/structural_causal_models.py`) | Apply to research questions |
| **Diff-in-Diff** | ✅ Code complete (`py/causal/diff_in_diff.py`) | Apply to treatment effects |
| **Synthetic Control** | ✅ Code complete (`py/causal/synthetic_control.py`) | Apply to policy evaluation |
| **Panel Constructor** | ✅ Code complete (`py/causal/panel_constructor.py`) | Generate panel data |
| **Confounder Identification** | ✅ Code complete (`py/causal/confounder_identification.py`) | Identify confounders |
| **Validation** | ✅ Code complete (`py/causal/validation.py`) | Validate causal claims |

**Use Cases for Causal Inference:**
1. **Treatment Effect Analysis:**
   - Does short rest **cause** worse performance? (vs correlation)
   - Does dome team playing outdoors **cause** worse performance?
   - Does extreme cold **cause** lower scoring?
2. **Deconfounding:**
   - Remove confounding variables for fair team comparisons
   - Adjust for schedule strength causally (not just correlational)
3. **Feature Selection:**
   - Identify causal features (vs spurious correlations)
   - Build causal DAG for feature engineering

**Priority:** MEDIUM (framework is ready, needs application)

**Next Steps:**
1. Pick 1-2 concrete causal questions
2. Run diff-in-diff or synthetic control analysis
3. Document findings (e.g., `analysis/CAUSAL_ANALYSIS_SHORT_REST.md`)
4. **Estimated Effort:** 4-6 hours per study

---

### Other Ensembles (Variants of the Hub)

| Model | Status | Performance | Next Steps |
|-------|--------|-------------|------------|
| **Ensemble v1.0 (Hierarchical)** | Old baseline | 53.8% win rate, 1.29% ROI | Superseded by v2025, keep for comparison |
| **Ensemble v2.0 (Realistic)** | Intermediate | No vegas leakage | Evolved into v2025 optimized |
| **Ensemble v3.0 (Enhanced)** | ⚠️ Failed backtest | 0% ROI (0 predictions generated) | **Needs debugging** (may be training issue) |

**v3.0 Analysis:**
- **Issue:** Backtest generated 0 predictions (models/ensemble/enhanced_ensemble_v3.py)
- **Possible Causes:**
  1. Training never completed successfully
  2. Code bug preventing prediction generation
  3. Missing dependencies or data
- **Recommendation:** Investigate and fix OR document as failed experiment
- **Estimated Effort:** 3-4 hours to debug

---

## Strengths & Weaknesses Matrix

| Model | Strengths | Weaknesses | Standalone Viability | Combination Potential |
|-------|-----------|------------|---------------------|----------------------|
| **Game Outcome Ensemble v2025** | Production-ready, well-calibrated (2.8%), 60.7% accuracy, 10-15% ROI | No uncertainty quantification | ✅ **HIGH** (deployed) | N/A (is hub) |
| **BNN Multioutput v1** | 92% CI coverage (rushing), multi-task (yards + TDs), uncertainty | Calibration partial, slow inference (1.6GB) | ⚠️ MEDIUM (needs work) | ✅ **HIGH**: Uncertainty for ensemble |
| **BNN Mixture of Experts** | Expert specialization, robust to distribution shift | Not yet evaluated | ❓ UNKNOWN | ⚠️ MEDIUM: Needs evaluation first |
| **BNN Environment** | Captures contextual factors (weather, rest, etc.) | 29.7% CI (target 90%), poor calibration | ❌ LOW (poor calibration) | ✅ **HIGH**: Feature extractor for ensemble |
| **BNN Simpler v2** | Fast inference (10MB), lightweight | Not yet evaluated, may sacrifice accuracy | ❓ UNKNOWN (likely MEDIUM) | ⚠️ MEDIUM: Real-time inference |
| **GNN Hierarchical** | Relational structure, captures team dynamics | 46-50% accuracy (poor), worse than baseline | ❌ LOW (< 51%) | ⚠️ MEDIUM: Use embeddings as features |
| **cVAE** | Trajectory modeling, generative capabilities | No clear use case yet | ❌ LOW (undefined purpose) | ⚠️ LOW-MEDIUM: Needs use case |
| **RL Agents** | Adaptive strategy, optimal bankroll management | Not recently trained, unclear if code works | ❓ UNKNOWN (old) | ✅ **HIGH**: Policy optimization for betting |
| **Causal Framework** | Principled causal inference, deconfounding | Not yet applied to real questions | N/A (framework) | ⚠️ MEDIUM: Feature selection, treatment analysis |

---

## Combination Opportunities Matrix

### High-Priority Integrations (Test Next)

| ID | Combination | Purpose | Status | Effort | Expected Benefit | Success Criteria |
|----|-------------|---------|--------|--------|-----------------|------------------|
| **INT-001** | **Ensemble + BNN Multioutput Uncertainty** | Confidence-weighted predictions | ❌ Untested | 4h | Better bet sizing, risk management | Sharpe ratio ↑ 10%+, lower variance |
| **INT-002** | **Ensemble + BNN Environment Features** | Contextual adaptation | ❌ Untested | 3h | Capture game conditions (weather, rest) | Accuracy ↑ OR calibration ↑ |
| **INT-003** | **Ensemble + RL Policy** | Adaptive betting strategy | ❌ Untested | 8h | Optimal bankroll mgmt, market adaptation | ROI > 15%, better drawdown |

### Medium-Priority Integrations

| ID | Combination | Purpose | Status | Effort | Expected Benefit | Success Criteria |
|----|-------------|---------|--------|--------|-----------------|------------------|
| **INT-004** | **BNN Variants Stacking** | Multi-model Bayesian ensemble | ❌ Untested | 6h | Robust uncertainty quantification | 90% CI coverage, better tails |
| **INT-005** | **Ensemble + GNN Embeddings** | Richer team representations | ❌ Untested | 5h | Capture relational structure | Accuracy ↑ 1-2% |
| **INT-006** | **Causal + Feature Selection** | Identify causal features | ❌ Untested | 4-6h | Remove spurious correlations | More robust to distribution shift |

### Low-Priority / Exploratory

| ID | Combination | Purpose | Status | Effort | Expected Benefit | Success Criteria |
|----|-------------|---------|--------|--------|-----------------|------------------|
| **INT-007** | **Ensemble + cVAE Simulations** | Monte Carlo uncertainty | ❌ Untested | 6-8h | Alternative uncertainty quantification | Comparable to BNN uncertainty |
| **INT-008** | **GNN + Player-Level Graph** | Redesign GNN architecture | ❌ Not started | 12-16h | Better graph structure for prediction | Accuracy > 55% |
| **INT-009** | **RL + Conformal Prediction** | Adaptive confidence intervals | ❌ Not started | 8-10h | Dynamic uncertainty based on RL state | Better tail risk management |

---

## Hardware Requirements

### M4 Mac (Development Environment)

**Best For:**
- Code development and testing
- Light experiments and feature engineering
- Ensemble training (~5 min)
- BNN inference
- Data exploration
- Quick ablation studies (<1 hour)

**Limitations:**
- BNN training: 3-4 hours (vs 30-45 min on 4090)
- GNN ablation: 12-16 hours on CPU (vs 2-3 hours on 4090)
- RL training: Not practical for large hyperparameter sweeps
- Memory: 16-32 GB may limit batch sizes

**M4 Mac Training Times:**
| Model | Time |
|-------|------|
| Game Outcome Ensemble v2025 | ~5 min |
| BNN Multioutput v1 | 3-4 hours |
| BNN Simpler v2 | <1 hour |
| GNN Ablation (9 configs × 20 epochs) | 12-16 hours (CPU) |
| cVAE (100 epochs) | 30-40 min |

---

### 4090 Desktop (Training Environment)

**Best For:**
- Heavy BNN training (3-4h → 30-45 min)
- GNN ablation (12-16h → 2-3h)
- RL agent training (8-12h → 1-2h)
- Large hyperparameter sweeps
- Batch inference on large datasets

**Speedup:**
- BNN training: **~5-8x faster**
- GNN training: **~4-6x faster**
- RL training: **~6-8x faster**

**4090 Desktop Training Times (Estimated):**
| Model | Time |
|-------|------|
| BNN Multioutput v1 | 30-45 min |
| BNN Mixture v2 | 30-45 min |
| GNN Ablation (9 configs × 20 epochs) | 2-3 hours |
| RL Agents (full training) | 1-2 hours |
| Hyperparameter Search (50 trials) | 4-6 hours |

---

### Sync Strategy (Mac ↔ Desktop)

**Code:** Git (push from Mac, pull on Desktop)
**Data:** Shared NFS OR docker volume mount
**Models:** rsync OR S3 bucket OR git LFS
**Results:** Commit to `models/` and `logs/`, push to git

**Docker Setup:** See `docker/README.md` (to be created)

---

## Model File Inventory

### Bayesian Models (`models/bayesian/`)
```
bnn_multioutput_v1.pkl         1.6 GB   ✅ Trained   ⚠️ Needs eval
bnn_mixture_experts_v2.pkl     1.3 GB   ✅ Trained   ❌ Needs eval
bnn_environment.pkl            302 MB   ✅ Trained   ❌ Poor calibration
bnn_simpler_v2.pkl              10 MB   ✅ Trained   ❌ Needs eval
bnn_rushing_v1.pkl              53 MB   ✅ Trained   Research artifact
bnn_passing_v1.pkl              80 MB   ✅ Trained   Research artifact
bnn_rushing_sigma0.5.pkl       181 MB   ✅ Trained   Prior sensitivity study
bnn_rushing_sigma0.7.pkl       181 MB   ✅ Trained   Prior sensitivity study
bnn_rushing_sigma1.0.pkl       181 MB   ✅ Trained   Prior sensitivity study
bnn_rushing_sigma1.5.pkl       181 MB   ✅ Trained   Prior sensitivity study
bnn_demo_v1.pkl                 10 MB   ✅ Trained   Demo/tutorial
```

### GNN Models (`models/gnn/`)
```
hierarchical_gnn_best.pt       306 KB   ✅ Trained   Test accuracy 50.2%
ablation/*/best_model.pt         Various ✅ Trained   9 configs tested
```

### cVAE Models (`models/cvae/`)
```
best_model.pt                  ~5-10 MB ✅ Trained   Needs use case
config.json                             ✅ Saved     Training config
training_history.json                   ✅ Saved     Loss curves
```

### Ensemble Models (`models/ensemble/`)
```
game_outcome/                           ✅ v2025     Production
phase5_comparison/                      ✅ Ablation  Feature studies
realistic_backtest/                     ✅ Backtest  Validation
```

### RL Models (`models/cql/`, `models/iql/`, `models/dqn/`)
```
Status: ⚠️ UNCLEAR - Need to audit existing artifacts
```

---

## Summary & Recommendations

### What's Working (Keep)
1. ✅ **Game Outcome Ensemble v2025** - Production, well-calibrated, generating predictions
2. ✅ **Causal Inference Framework** - Code complete, ready for application
3. ✅ **BNN Multioutput v1** - 92% CI coverage, promising for integration
4. ✅ **GNN Ablation Study** - Complete analysis, clear next steps

### What Needs Work (Prioritize)
1. ⚠️ **BNN Consolidation** - Evaluate Mixture v2, Simpler v2, compare all variants
2. ⚠️ **Environment BNN** - Fix calibration OR repurpose as feature extractor
3. ⚠️ **Ensemble v3.0** - Debug why 0 predictions generated
4. ⚠️ **RL Investigation** - Audit existing code, define problem properly

### What's Exploratory (Define)
1. ❓ **cVAE Use Case** - Define concrete experiments before further work
2. ❓ **GNN Redesign** - Decide: embeddings, new architecture, or shelve?

### High-Priority Next Steps (This Week)
1. **Evaluate BNN Mixture v2** (30 min)
2. **Evaluate BNN Simpler v2** (30 min)
3. **Test Environment BNN as ensemble feature** (2-3 hours)
4. **INT-001: Ensemble + BNN Uncertainty** (4 hours)
5. **Create Docker workflow** (2-3 hours)

### Medium-Priority Next Steps (Next Week)
6. **INT-003: RL Investigation** (8-12 hours)
7. **Causal Analysis: Short Rest Effect** (4-6 hours)
8. **BNN Variants Stacking** (6 hours)
9. **Debug Ensemble v3.0** (3-4 hours)

---

**Last Updated:** 2025-10-31
**Maintained By:** Research Team
**Related Documents:**
- `RESEARCH_VECTORS.md` - Detailed status of each research spoke
- `INCOMPLETE_MODELS.md` - What's missing for each model
- `INTEGRATION_MATRIX.json` - Machine-readable integration opportunities
- `experiments/manifest.yaml` - Experiment tracking
- `OPTIMIZED_ENSEMBLE_2025_DEPLOYMENT.md` - Production ensemble details
