# BNN Implementation Exploration Report
## Phase 1 (Conformal Prediction) & Phase 5 (Evidential Deep Learning) Enhancement Plan

**Date:** November 9, 2025  
**Thoroughness Level:** Medium  
**Project Status:** 3-way ensemble (hierarchical_v2 excluded due to convergence failure)

---

## EXECUTIVE SUMMARY

The NFL Analytics codebase has **multiple BNN implementations** across two major frameworks:

1. **PyMC-based BNN** (`bayesian_neural_network.py`):
   - Production-ready PyMC implementation with ADVI/NUTS
   - Used in legacy pipeline but NOT in current 3-way ensemble
   - Full uncertainty quantification via posterior sampling

2. **PyTorch-based BNN** (Multi-Output models):
   - GPU-optimized variational inference with Bayesian layers
   - Current production variant: `bnn_multioutput_v3.1` (rushing yards only, 89.5% calibrated)
   - Mixture-of-Experts architecture with 3 experts
   - **CRITICAL:** Passing yards variant abandoned after 6 consecutive failures (NaN losses)

3. **Conformal Prediction Baseline** (already exists!):
   - MAPIE-based split conformal prediction
   - 90% coverage guarantee (theoretical)
   - Baseline for Phase 1 comparison

### Current Ensemble Configuration
- **State-Space v2** (30% weight): R/brms, MAE 47.77, approved Nov 8
- **BNN Multi-Output v3.1** (25% weight): PyTorch rushing yards, 89.5% calibrated
- **XGBoost v2** (45% weight): Increased from 20% after hierarchical_v2 exclusion
- **Hierarchical_v2 & Informative_Priors_v2**: Excluded due to Rhat convergence failures

---

## 1. CURRENT BNN FILES

### Production/Near-Production Models

#### A. PyMC-Based (Scikit-Learn Style API)
**File:** `py/models/bayesian_neural_network.py` (415 lines)

**Architecture:**
```
Input (n_features) 
  → Dense(64, HalfNormal priors) 
  → ReLU 
  → Dense(32) 
  → ReLU 
  → Output mean + optional output std
```

**Framework:** PyMC 5.x (probabilistic programming)

**Inference Methods:**
- ADVI: Variational inference (fast, approximate)
- NUTS: MCMC sampling (slow, exact)

**Uncertainty Quantification:**
- Heteroscedastic option: Input-dependent noise via second output head
- Posterior sampling for predictions
- Full posterior distributions available

**Key Features:**
- Hierarchical priors (HalfNormal on weight scales)
- Automatic feature scaling (StandardScaler)
- Save/load via joblib
- Demo function shows quick start

**Status:** ⚠️ Legacy, not currently used in ensemble

**Integration Point:** `py/models/bnn_temperature_scaling.py` wraps this for calibration

---

#### B. PyTorch Multi-Output Models (Current Production Rushing)

**Primary File:** `py/models/bnn_multioutput_pytorch.py` (280+ lines)

**Architecture:**
```
3 Expert Networks (in parallel):
  Each expert:
    Input (features)
      → BayesianLinear (weight uncertainty via reparameterization trick)
      → ReLU
      → BayesianLinear
      → ReLU
      → Output: yards head (continuous) + TD head (binary)

Gating Network:
  Input (features) → Linear → Softmax(3 experts)

Outputs:
  - weighted_yards = sum(gate_weights * expert_yards)
  - weighted_td_logits = sum(gate_weights * expert_td_logits)
```

**Framework:** PyTorch + CUDA/MPS acceleration

**Inference Method:** 
- Variational inference (reparameterization trick)
- Stochastic forward passes for MC sampling
- KL divergence regularization

**Uncertainty Quantification:**
- MC dropout behavior from weight sampling
- Multiple forward passes to estimate epistemic uncertainty
- Can generate confidence intervals

**Status:** ✅ Production (rushing yards v3.1, 89.5% calibrated)

**Current Usage:** MODEL_REGISTRY.yaml shows:
```yaml
ensemble_weight: 0.25
file_path: models/bayesian/bnn_multioutput_v1_final.pt
temperature: 0.75  # Post-hoc calibration
```

---

#### C. Other PyTorch Variants (Archived/Failed)

**Attempted approaches (all failed):**
1. `bnn_mc_dropout.py` - Gradient explosion (NaN losses)
2. `bnn_laplace.py` - Hessian dimension errors (9/9 configurations)
3. `bnn_nuts_comparison.py` - Timeout (6+ hours)
4. `train_bnn_passing.py` - NaN losses, week 0 contamination
5. `train_multioutput_passing_bnn.py` - NaN losses despite canonical window + Week 0 fix
6. `bnn_multioutput_passing_v2` - 6th consecutive failure, archived Nov 9

**Root Cause Analysis (from dissertation Appendix I):**
- Passing yards has higher variance → harder to model joint distribution
- Lag/rolling features create data leakage
- Architecture fundamentally limited for passing variance (not rushing)
- No fundamental fix found after 84 person-hours, 26 days

---

### Calibration & Temperature Scaling

**File:** `py/models/bnn_temperature_scaling.py` (200+ lines)

**Method:** Post-hoc temperature scaling (Guo et al. 2017)

**Key Points:**
- Loads pre-trained BNN
- Tunes temperature T on validation set
- Applies: σ_calibrated = σ_raw × T
- BNN v3.1 reaching used **T = 0.75** for 89.5% → 68% calibration fix

**Status:** ✅ Currently deployed

---

### Data/Feature Engineering for BNN

**Rushers:** `py/models/train_bnn_rushing.py` (200+ lines)

**Receivers:** `py/models/train_bnn_receiving.py`

**Vegas-aware:** `py/models/train_bnn_rushing_vegas.py`

**Common Pattern:**
- Load from PostgreSQL (5544)
- Feature engineering: carries, avg_rushing_l3, season_avg, week, spread_close, total_close
- Train/test split: 2020-2023 train, 2024 test
- StandardScaler normalization

---

## 2. CURRENT BNN ARCHITECTURE SUMMARY

### Framework Comparison

| Aspect | PyMC | PyTorch |
|--------|------|---------|
| **Status** | Legacy | Production |
| **Inference** | ADVI/NUTS | Variational (reparameterization) |
| **UQ Method** | Posterior sampling | MC dropout + weight sampling |
| **Hardware** | CPU | GPU/MPS accelerated |
| **Training Time** | Hours-days | Minutes-hours |
| **Calibration** | 93.8% → 68% (T=0.75) | 89.5% (T=0.75) |
| **Framework** | Probabilistic prog. | Deep learning |
| **Production Use** | ❌ No | ✅ Rushing v3.1 |

### Uncertainty Quantification Methods

**Current in Production (BNN v3.1):**
1. Bayesian weight uncertainty → epistemic UQ
2. Temperature scaling (T=0.75) → post-hoc calibration
3. MC dropout behavior from stochastic weights
4. Ensemble prediction intervals from multiple forward passes

**NOT Currently Used:**
- Aleatoric uncertainty head (input-dependent noise)
- Full posterior sampling
- Evidence lower bound (ELBO) analysis
- Calibration curves

---

## 3. INTEGRATION POINTS WITH ENSEMBLE

### Current Ensemble Architecture (`enhanced_ensemble_v5_canonical.py`)

**3-way ensemble (Nov 9, 2025 revision):**

```python
predictions = {
    "state_space": (mean, std),        # 30% weight
    "bnn_multioutput": (mean, std),    # 25% weight
    "xgboost": (mean, None)            # 45% weight (no uncertainty)
}

# Weighted combination (inverse variance weighting)
weights = 1.0 / (stds ** 2)
weights /= weights.sum()
ensemble_mean = sum(weights * means)
ensemble_var = sum((weights ** 2) * (stds ** 2))
ensemble_std = sqrt(ensemble_var)
```

### Load/Predict Flow

1. **Load BNN:** `BayesianNeuralNetwork.load()` (PyMC) OR directly load PyTorch `.pt`
2. **Predict:** Call `.predict()` → returns (mean, std)
3. **Aggregate:** Inverse-variance weighting
4. **Output:** Ensemble (mean, std) tuple

### Missing Components

- ❌ No conformal prediction wrapper
- ❌ No evidential output heads
- ❌ No calibration curves generated during inference
- ❌ No prediction interval diagnostics

---

## 4. PHASE 1: CONFORMAL PREDICTION GAPS

### What Exists

**File:** `py/models/conformal_prediction_baseline.py` (410 lines)

**Implementation:**
- MAPIE SplitConformalRegressor
- Random Forest base model (100 trees)
- Alpha = 0.10 for 90% coverage guarantee
- Coverage: 90% CI, ±1σ (~68%)
- Sharpness: Average interval width

**Current Status:**
- ✅ Standalone baseline exists
- ❌ NOT integrated with BNN
- ❌ NOT in ensemble pipeline

### Phase 1 Requirements (Not Met)

**Gap 1: BNN-Wrapped Conformal Predictor**
- Need: Wrap BNN predictions with conformal intervals
- Current: Only Random Forest baseline
- Missing: `py/models/bnn_conformal.py`

```python
# DESIRED but missing:
class BNNConformalPredictor:
    def __init__(self, bnn_model_path, alpha=0.10):
        self.bnn = BayesianNeuralNetwork.load(bnn_model_path)
        self.conformal = SplitConformalRegressor(...)
    
    def predict(self, X):
        # BNN predictions
        bnn_mean, bnn_std = self.bnn.predict(X)
        
        # Conformal wrapper
        conf_lower, conf_upper = self.conformal.predict_interval(X)
        
        # Return both: BNN uncertainty + conformal guarantee
        return {
            'bnn_mean': bnn_mean,
            'bnn_std': bnn_std,
            'conformal_lower': conf_lower,
            'conformal_upper': conf_upper,
            'coverage_guarantee': 0.90
        }
```

**Gap 2: Ensemble Integration**
- Need: Conformal intervals for 3-way ensemble
- Current: Point estimates + inverse-variance weighting
- Missing: Unified uncertainty framework

**Gap 3: Calibration Diagnostics**
- Need: Runtime calibration monitoring
- Missing: `py/monitoring/calibration_tracker.py` for conformal predictions
- Missing: Coverage plots per prediction pool

---

## 5. PHASE 5: EVIDENTIAL DL GAPS

### Current Status: ❌ ZERO Implementation

No evidential outputs found in codebase.

### Phase 5 Requirements

**What is Evidential DL?**

Evidential deep learning quantifies uncertainty via Dirichlet/Beta distributions:
- Uses softmax confidence outputs to estimate aleatoric + epistemic uncertainty
- Outputs: (μ, σ, λ) where λ is "evidence strength"
- Automatic calibration without temperature scaling

**Architecture Needed:**

```python
# DESIRED (doesn't exist):
class EvidentialBNN(nn.Module):
    """
    BNN with evidential outputs for uncertainty calibration
    
    Standard output: Single prediction + std
    Evidential output: (mean, variance, lambda)
    where lambda measures evidence for the prediction
    """
    
    def __init__(self, input_dim, hidden_dims=(64, 32)):
        super().__init__()
        self.experts = nn.ModuleList([...])  # 3 experts
        
    def forward(self, x, return_evidential=True):
        # Expert outputs
        expert_outputs = [e(x) for e in self.experts]
        
        if return_evidential:
            # Evidential head: outputs (μ, σ, λ)
            mu, sigma, lambda_evi = self.evidential_head(expert_outputs)
            return {
                'mu': mu,
                'sigma': sigma,
                'lambda': lambda_evi,  # Evidence/confidence
                'uncertainty_type': 'epistemic+aleatoric'
            }
        else:
            # Standard Bayesian output
            return mu, sigma
```

### Phase 5 Research Questions

1. **Does evidential output improve calibration vs temperature scaling?**
   - Current: T=0.75 → 89.5% (after tuning)
   - Evidential: Could achieve automatic calibration?

2. **Can we port PyTorch BNN to Evidential framework?**
   - Current: Variational Bayesian layers
   - Needed: Evidential regression heads

3. **Computational overhead?**
   - Additional forward pass for evidential head?
   - Training time increase?

---

## 6. RECOMMENDED INTEGRATION ARCHITECTURE

### High-Level Proposal

```
Enhanced Uncertainty Pipeline
├─ Phase 1: Conformal Prediction Wrappers
│  ├─ BNN → SplitConformalRegressor
│  ├─ Ensemble → Conformal aggregation
│  └─ Monitoring → Runtime calibration tracking
│
├─ Phase 5: Evidential Outputs
│  ├─ Port PyTorch BNN to evidential regression
│  ├─ Comparison: Evidential vs. Temperature scaling
│  └─ Production decision: Which to use?
│
└─ Integration Points
   ├─ Load: enhanced_ensemble_v5_canonical.py (updated)
   ├─ Predict: BNNConformalPredictor | EvidentialBNN
   ├─ Aggregate: Ensemble with guaranteed coverage
   └─ Monitor: Calibration tracker (Phase 1 + Phase 5)
```

---

## 7. FILE LOCATIONS & ORGANIZATION

### Current BNN Code

```
py/models/
├─ bayesian_neural_network.py          # PyMC BNN (legacy, 415 lines)
├─ bnn_multioutput_pytorch.py           # PyTorch BNN (current prod, 280 lines)
├─ train_bnn_rushing.py                 # Training script (rushing yards)
├─ train_bnn_receiving.py               # Training script (receiving yards)
├─ train_bnn_rushing_vegas.py           # Vegas-aware variant
├─ train_multioutput_passing_bnn.py     # FAILED: Passing yards
├─ bnn_temperature_scaling.py           # Post-hoc calibration (120 lines)
├─ conformal_prediction_baseline.py     # MAPIE baseline (410 lines) ✅ EXISTING
├─ validate_bnn_calibration.py          # Calibration metrics
└─ improve_bnn_calibration.py           # Temperature search utils

py/ensemble/
├─ enhanced_ensemble_v5_canonical.py    # 3-way ensemble (current production)
├─ enhanced_ensemble_v5.py              # (older variant)
├─ enhanced_ensemble_v3.py              # (archived)
└─ ensemble_predictor.py                # Generic ensemble class

py/utils/
├─ calibration.py                       # Calibration metrics library
└─ ensemble_orchestrator.py             # Training coordination

models/
├─ MODEL_REGISTRY.yaml                  # Production manifest (Nov 9, 2025)
├─ bayesian/
│  ├─ bnn_multioutput_v1_final.pt       # Current production (PyTorch)
│  ├─ simplified_random_walk_v2.rds     # State-Space (R/brms)
│  └─ *.rds                              # R models
└─ xgboost/
   └─ passing_xgboost_v2_canonical.pkl  # XGBoost

tests/unit/
├─ test_bayesian_neural_network.py      # PyMC BNN tests
├─ test_bnn_multioutput_pytorch.py      # PyTorch BNN tests
├─ test_bnn_temperature_scaling.py      # Calibration tests
└─ test_conformal_prediction_baseline.py # Conformal tests
```

### Proposed New Files (Phase 1)

```
py/models/
├─ bnn_conformal.py                     # BNN + conformal wrapper (NEW)
├─ ensemble_conformal.py                # Ensemble + conformal aggregation (NEW)

py/monitoring/
├─ calibration_tracker.py               # Runtime monitoring (partly exists)
└─ conformal_coverage_monitor.py        # Coverage tracking (NEW)

examples/
├─ phase1_bnn_conformal_example.py      # Usage example (NEW)
└─ phase1_calibration_comparison.py     # Conformal vs. temperature (NEW)

tests/
├─ test_bnn_conformal.py                # Unit tests (NEW)
└─ test_ensemble_conformal.py           # Integration tests (NEW)
```

### Proposed New Files (Phase 5)

```
py/models/
├─ evidential_bnn.py                    # Evidential regression layer (NEW)
├─ evidential_pytorch_bnn.py            # Full evidential BNN (NEW)
├─ train_evidential_bnn_rushing.py      # Training script (NEW)

py/analysis/
├─ evidential_vs_temperature_study.py   # Comparison analysis (NEW)

examples/
├─ phase5_evidential_example.py         # Usage example (NEW)

tests/
├─ test_evidential_bnn.py               # Unit tests (NEW)
└─ test_evidential_calibration.py       # Calibration tests (NEW)
```

---

## 8. KEY BLOCKERS & CONCERNS

### Blocker 1: PyMC vs PyTorch Decision
**Status:** Unresolved

- PyMC BNN exists but legacy (not used)
- PyTorch BNN exists and in production
- **Question:** Should Phase 1/5 enhancements target PyTorch or both?
- **Recommendation:** Focus on PyTorch (current production path)
- **Risk:** PyMC still better for Bayesian inference (posterior sampling)

### Blocker 2: Passing Yards BNN Failure
**Status:** Documented, research line abandoned

- 6 consecutive BNN failures for passing yards
- Only rushing yards variant (v3.1) works
- **Question:** Why does evidential learning help passing but not current BNN?
- **Hypothesis:** Evidential framework better suited for high-variance prediction
- **Action:** Phase 5 should explicitly target passing yards recovery

### Blocker 3: Temperature Scaling Already Works
**Status:** Effective but suboptimal

- Current solution: T=0.75 → 89.5% calibration
- **Question:** Is conformal prediction + evidential actually better?
- **Concern:** May need extensive tuning/retraining
- **Recommendation:** Phase 1 should show >5% improvement to justify switch

### Blocker 4: Ensemble Model Failures
**Status:** Recent, Nov 9 discovery

- Hierarchical_v2: Rhat 1.0337 > 1.01 (excluded)
- Informative_Priors_v2: Rhat 1.0995 > 1.01 (excluded)
- **Impact:** Ensemble reduced from 4-way to 3-way
- **Question:** Should Phase 1/5 fix these models or assume they stay excluded?
- **Recommendation:** Assume current 3-way ensemble; separate issue

### Blocker 5: Dependencies Not in requirements.txt
**Status:** Missing

Current requirements.txt lacks:
- ❌ `mapie` (for conformal prediction)
- ❌ `pymc` (for PyMC BNN)
- ❌ `arviz` (for MCMC diagnostics)

**Action:** Need to add to requirements.txt before Phase 1/5 training

---

## 9. RECOMMENDATIONS FOR PHASE 1 & 5

### Phase 1: Conformal Prediction (PRIORITY: HIGH)

**Step 1: Wrap Current BNN with Conformal (2-3 days)**
```python
# py/models/bnn_conformal.py
class BNNConformalPredictor:
    """Wraps trained BNN with conformal prediction intervals"""
    - Load: BNN v3.1 or v2 (test both)
    - Calibration: Split conformal on 2023 validation set
    - Output: (bnn_mean, bnn_std, conf_lower, conf_upper)
```

**Step 2: Integrate with Ensemble (1-2 days)**
```python
# py/ensemble/ensemble_conformal.py
def ensemble_conformal_predict(X):
    """3-way ensemble with conformal guarantees"""
    - Conformal from each model (where available)
    - Aggregate intervals via Bonferroni correction or pooling
    - Return: Ensemble mean + guaranteed 90% coverage
```

**Step 3: Add Monitoring (1 day)**
```python
# py/monitoring/conformal_coverage_monitor.py
- Track: Daily/weekly coverage rates
- Alert: If coverage < 87% or > 93% (drift detection)
- Log: Prediction pool statistics
```

**Step 4: Validation Study (2-3 days)**
- Compare: Temperature scaling vs. Conformal on test set
- Hypothesis: Conformal should match or exceed temperature scaling
- Report: Coverage rates, interval widths, computational cost

**Expected Outcome:**
- ✅ BNN conformal wrapper operational
- ✅ Ensemble with guaranteed coverage
- ✅ Runtime monitoring
- ✅ Decision: Use conformal, temperature, or both?

---

### Phase 5: Evidential DL (PRIORITY: MEDIUM)

**Step 1: Implement Evidential Regression Layer (2-3 days)**
```python
# py/models/evidential_bnn.py
class EvidentialRegressionLayer(nn.Module):
    """Output head with evidential uncertainty"""
    - Inputs: expert features
    - Outputs: (μ, σ, λ) where λ = evidence/confidence
    - Loss: Negative log marginal likelihood
    - Calibration: Automatic (no temperature scaling needed)
```

**Step 2: Integrate with PyTorch BNN (2-3 days)**
```python
# py/models/evidential_pytorch_bnn.py
class EvidentialBNN(nn.Module):
    """Multi-expert BNN with evidential outputs"""
    - Architecture: 3 experts + gating (like current)
    - Outputs: Add evidential head alongside yards/TD heads
    - Training: KL annealing + evidential loss
    - Inference: Can use standard or evidential mode
```

**Step 3: Train on Rushing (1 day)**
```python
# py/models/train_evidential_bnn_rushing.py
- Start: From BNN v3.1 weights
- Epochs: 30-50 (longer than current)
- Loss: MSE + KL + evidential regularization
- Target: >90% calibration without temperature scaling
```

**Step 4: Try on Passing Yards (2-3 days)**
```python
# py/models/train_evidential_bnn_passing.py
- Goal: Recover passing yards (currently abandoned)
- Hypothesis: Evidential better for high-variance
- Success metric: MAE <65 + >85% calibration
- Risk: Might still fail
```

**Step 5: Comparison Study (2-3 days)**
```python
# py/analysis/evidential_vs_temperature_study.py
- Hypothesis test: Evidential ≥ Temperature scaling?
- Metrics: Calibration, sharpness, MAE, training time
- Decision: Which to use in production?
```

**Expected Outcome:**
- ✅ Evidential BNN implemented
- ✅ Rushing yards: Better or similar calibration (no temp scaling)
- ⚠️ Passing yards: Might recover or might still fail (risky)
- ✅ Production decision: Use evidential, temperature, or hybrid?

---

## 10. STRATEGIC RECOMMENDATIONS

### Short-term (Phase 1 - 2 weeks)
1. ✅ **Create `bnn_conformal.py`** - Wrap existing BNN
2. ✅ **Create `ensemble_conformal.py`** - Integrate with 3-way
3. ✅ **Add calibration monitoring** - Track coverage drift
4. ✅ **Validate against temperature scaling** - Show improvement/parity

**Decision Gate:** Only proceed to Phase 5 if Phase 1 shows clear benefit (>3% improvement or lower cost).

### Medium-term (Phase 5 - 3-4 weeks)
1. ⚠️ **Implement evidential regression** - New layer type
2. ⚠️ **Port to rushing yards first** - Lower risk
3. ⚠️ **Attempt passing yards** - Known failure point
4. ⚠️ **Comparison study** - Evidential vs. current methods

**Decision Gate:** Only deploy evidential if measurably better AND production-ready.

### Strategic Decisions Needed

| Decision | Current | Phase 1 | Phase 5 |
|----------|---------|---------|---------|
| UQ Method | Temperature scaling | Conformal prediction | Evidential DL |
| PyMC vs PyTorch | PyTorch only | PyTorch only | PyTorch only |
| Passing Yards | Abandoned | No change | Attempt recovery |
| Computational Cost | Low | Medium | High |
| Production Risk | Low | Low | Medium |
| Timeline | Done | 2 weeks | 3-4 weeks |

---

## 11. SUMMARY MATRIX

### By File

| File Path | Type | Status | Framework | UQ Method | Production? |
|-----------|------|--------|-----------|-----------|-------------|
| `bayesian_neural_network.py` | BNN | Legacy | PyMC | Posterior sampling | ❌ |
| `bnn_multioutput_pytorch.py` | BNN | Production | PyTorch | Weight uncertainty | ✅ |
| `bnn_temperature_scaling.py` | Calibration | Active | Post-hoc | Temperature T | ✅ |
| `conformal_prediction_baseline.py` | Baseline | Exists | MAPIE+RF | Split conformal | ❌ Standalone |
| `train_bnn_rushing.py` | Training | Reference | PyTorch | - | ✅ |
| `train_multioutput_passing_bnn.py` | Training | Failed | PyTorch | - | ❌ |

### By Gap

| Gap | Type | Phase 1 | Phase 5 | Blocker? |
|-----|------|---------|---------|----------|
| BNN conformal wrapper | Code | Missing | N/A | Yes |
| Ensemble conformal | Code | Missing | N/A | Yes |
| Calibration monitoring | Code | Partial | N/A | No |
| Evidential layer | Code | N/A | Missing | Yes |
| Evidential BNN | Code | N/A | Missing | Yes |
| Passing yards recovery | Research | N/A | Risky | No |
| Dependencies in requirements.txt | Config | Missing | Missing | Yes |

---

## 12. QUICK START CHECKLIST

### Prerequisites
- [ ] Add to requirements.txt: `mapie>=0.7.0`, `pymc>=5.0.0`, `arviz>=0.17.0`
- [ ] Review MODEL_REGISTRY.yaml (current 3-way ensemble config)
- [ ] Review dissertation Appendix I (BNN failure analysis)
- [ ] Understand current temperature scaling approach (T=0.75)

### Phase 1 Execution
- [ ] Create `py/models/bnn_conformal.py` (200 lines)
- [ ] Create `py/ensemble/ensemble_conformal.py` (300 lines)
- [ ] Create `py/monitoring/conformal_coverage_monitor.py` (150 lines)
- [ ] Create `py/analysis/conformal_vs_temperature_study.py` (400 lines)
- [ ] Add tests: 3-4 unit test files
- [ ] Validate: 2022-2024 backtest
- [ ] Document: Add to docs/PHASE_1_CONFORMAL.md

### Phase 5 Execution
- [ ] Create `py/models/evidential_bnn.py` (250 lines)
- [ ] Create `py/models/evidential_pytorch_bnn.py` (350 lines)
- [ ] Create `py/models/train_evidential_bnn_rushing.py` (300 lines)
- [ ] Create `py/analysis/evidential_vs_temperature_study.py` (450 lines)
- [ ] Add tests: 3-4 unit test files
- [ ] Validate: 2024 test set
- [ ] Optional: `train_evidential_bnn_passing.py` (risky, might fail)
- [ ] Document: Add to docs/PHASE_5_EVIDENTIAL.md

---

**Report Generated:** November 9, 2025  
**Next Steps:** Review blockers, confirm Phase 1 priority, schedule 2-week sprint

