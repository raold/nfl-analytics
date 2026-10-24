# Component Deep Dives

**Purpose**: Comprehensive technical guide for the 6 major components of the NFL Analytics system. Each section provides detailed usage examples, architecture patterns, configuration options, and integration points.

**Last Updated**: November 8, 2025
**Version**: 1.0

---

## Table of Contents

1. [Causal Inference Framework](#causal-inference-framework)
2. [Bayesian Hierarchical Models](#bayesian-hierarchical-models)
3. [Ensemble Methods](#ensemble-methods)
4. [Reinforcement Learning (CQL)](#reinforcement-learning-cql)
5. [Distributed Compute System](#distributed-compute-system)
6. [Dissertation](#dissertation)

---

## Causal Inference Framework

### Overview

**Location**: `nfl_analytics/causal/`
**Status**: ✅ Complete (Phase 6)
**Documentation**: `docs/causal/CAUSAL_INFERENCE_FRAMEWORK.md`

**Purpose**: Move beyond correlation to true causal understanding for shock events (injuries, coaching changes, trades). This framework implements 8 modules for end-to-end causal analysis from data preparation through validation.

### Module Architecture

The framework consists of 8 integrated modules:

1. **`panel_constructor.py`** - Build player × week panel datasets
   - Constructs longitudinal panel data from play-by-play records
   - Handles temporal alignment and missing data
   - Output: CSV/Parquet with player-week observations

2. **`treatment_definitions.py`** - Define shock events
   - Injury treatments (severity levels)
   - Coaching changes (regime shifts)
   - Trades (team/role changes)
   - Weather shocks (extreme conditions)
   - Output: Boolean treatment indicators with timing

3. **`confounder_identification.py`** - Identify confounders, check balance
   - Implements backdoor criterion for DAG analysis
   - Computes standardized mean differences (SMD)
   - Balance diagnostics post-matching
   - Output: Confounder sets, balance tables

4. **`synthetic_control.py`** - Counterfactual estimation
   - Synthetic control weights via convex optimization
   - Pre-treatment matching period validation
   - Post-treatment counterfactual trajectories
   - Output: Weights, synth trajectories, prediction errors

5. **`diff_in_diff.py`** - Difference-in-differences estimators
   - Two-way fixed effects (TWFE) with clustering
   - Heterogeneous treatment effects (HTE)
   - Event study plots
   - Output: Treatment effect estimates with confidence intervals

6. **`structural_causal_models.py`** - Causal DAGs
   - Implements causal directed acyclic graphs
   - Backdoor criterion enforcement
   - Front-door criterion for mediation
   - Output: DAG validation, covariate selection

7. **`model_integration.py`** - Integrate causal adjustments with BNN
   - Extract treatment effect posteriors
   - Combine with BNN predictions
   - Uncertainty propagation
   - Output: Adjusted predictions with confidence bounds

8. **`validation.py`** - Placebo tests, sensitivity analysis, backtesting
   - Placebo tests (fake treatments)
   - Robustness to specification changes
   - Backtesting on historical data
   - Output: Validation reports with p-values

### Usage Examples

#### Basic Difference-in-Differences

```python
from causal.diff_in_diff import DifferenceInDifferences

# Initialize estimator with clustering on player_id
did = DifferenceInDifferences(cluster_var='player_id')

# Fit the model
did.fit(
    df=panel,
    outcome_col='stat_yards',
    treatment_col='injury_treatment',
    unit_col='player_id',
    time_col='week'
)

# Extract results
print(f"Injury effect: {did.treatment_effect_['estimate']:.1f} yards")
print(f"95% CI: [{did.treatment_effect_['ci_lower']:.1f}, "
      f"{did.treatment_effect_['ci_upper']:.1f}]")
print(f"Standard Error: {did.treatment_effect_['se']:.2f}")
print(f"p-value: {did.treatment_effect_['p_value']:.4f}")

# Heterogeneous effects by position
het_effects = did.get_het_effects(group_var='position')
print(het_effects)
```

#### Synthetic Control Analysis

```python
from causal.synthetic_control import SyntheticControl
from causal.panel_constructor import PanelConstructor

# Build panel data
panel_builder = PanelConstructor()
panel = panel_builder.construct_panel(
    start_season=2018,
    end_season=2024,
    outcome_vars=['passing_yards', 'completion_pct', 'td_passes']
)

# Apply synthetic control to a specific injury
sc = SyntheticControl(
    pre_period_end=20,  # Treatment occurs in week 21
    post_period_end=32
)

# Fit synthetic control weights
sc.fit(
    df=panel,
    outcome_col='passing_yards',
    unit_col='player_id',
    time_col='week',
    treatment_unit='QB_12345'  # Patrick Mahomes example
)

# Extract counterfactual and effect
counterfactual = sc.counterfactual_
effect = sc.treatment_effect_
effect_se = sc.treatment_effect_se_

print(f"Average treatment effect: {effect:.1f} yards")
print(f"95% CI: [{effect - 1.96*effect_se:.1f}, {effect + 1.96*effect_se:.1f}]")

# Plot the trajectory
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(panel.loc[panel['player_id']=='QB_12345', 'week'],
         panel.loc[panel['player_id']=='QB_12345', 'passing_yards'],
         label='Actual', marker='o')
plt.plot(counterfactual.index, counterfactual,
         label='Counterfactual', linestyle='--', marker='s')
plt.axvline(x=21, color='red', linestyle=':', label='Treatment')
plt.xlabel('Week')
plt.ylabel('Passing Yards')
plt.legend()
plt.title('Injury Impact: Synthetic Control Analysis')
plt.savefig('analysis/results/injury_synthetic_control.pdf')
```

#### Causal DAG and Backdoor Criterion

```python
from causal.structural_causal_models import CausalDAG

# Define causal relationships
dag = CausalDAG()

# Add nodes (variables)
dag.add_node('injury', 'Injury Status')
dag.add_node('yards', 'Passing Yards')
dag.add_node('experience', 'Player Experience')
dag.add_node('team_strength', 'Team Strength')
dag.add_node('opponent', 'Opponent Defense')

# Add edges (causal relationships)
dag.add_edge('experience', 'yards')  # Experience → Yards
dag.add_edge('experience', 'injury')  # Experience → Injury (confounder!)
dag.add_edge('injury', 'yards')       # Injury → Yards (treatment effect)
dag.add_edge('team_strength', 'yards')
dag.add_edge('opponent', 'yards')

# Check backdoor criterion
backdoor_paths = dag.get_backdoor_paths('injury', 'yards')
print(f"Backdoor paths: {backdoor_paths}")

# Get covariate set to close backdoors
adjustment_set = dag.get_backdoor_adjustment_set('injury', 'yards')
print(f"Adjust for: {adjustment_set}")
# Output: {'experience'} to block the confounding path
```

#### Model Integration with BNN

```python
from causal.model_integration import CausalAdjustedPredictor
from nfl_analytics.models.bnn_calibrated_v2 import CalibratedBNN

# Initialize BNN
bnn = CalibratedBNN.load('models/bayesian/bnn_multioutput_v1_final.pt')

# Wrap with causal adjustment
causal_predictor = CausalAdjustedPredictor(
    bnn_model=bnn,
    causal_effects='models/causal/injury_effects_posterior.pkl'
)

# Get prediction with causal adjustment
player_features = {...}  # Player features before injury
injury_status = True

# Get BNN prediction (baseline)
baseline_pred = bnn.predict(player_features)  # μ=275 yards, σ=35

# Get causal-adjusted prediction
causal_pred = causal_predictor.predict(
    features=player_features,
    treatment=injury_status
)
# Output: μ=258 yards (275 - 17 yard injury effect), σ=45 (uncertainty propagated)

print(f"Baseline: {baseline_pred.mean:.1f} ± {baseline_pred.std:.1f}")
print(f"Injury-adjusted: {causal_pred.mean:.1f} ± {causal_pred.std:.1f}")
print(f"Causal adjustment: {causal_pred.mean - baseline_pred.mean:.1f} yards")
```

#### Validation and Backtesting

```python
from causal.validation import CausalValidator

validator = CausalValidator()

# Run placebo tests (apply fake treatments)
placebo_results = validator.run_placebo_tests(
    df=panel,
    outcome_col='passing_yards',
    treatment_col='injury_treatment',
    unit_col='player_id',
    time_col='week',
    n_placebos=1000
)

# Sensitivity analysis (how much unobserved confounding would change results?)
sensitivity = validator.sensitivity_analysis(
    treatment_effect=did.treatment_effect_['estimate'],
    treatment_effect_se=did.treatment_effect_['se'],
    correlation_strength=0.8  # How correlated is unobserved confounder?
)

print(f"Treatment effect robust if |R²|_z < {sensitivity['robustness_bound']:.2f}")

# Backtest on historical injuries
backtest = validator.backtest_causal_model(
    df=panel,
    treatment_col='injury_treatment',
    outcome_col='passing_yards',
    model=causal_predictor,
    test_period_start=2022,
    test_period_end=2024
)

print(f"Backtest MAE: {backtest['mae']:.1f} yards")
print(f"Backtest Coverage (90% CI): {backtest['coverage_90']:.1%}")
```

### Integration Points

- **Input**: Play-by-play data from database (`mart.player_game_stats`)
- **Output**: Causal effect estimates, adjusted predictions
- **Integration with BNN**: `nfl_analytics/models/bnn_calibrated_v2.py` accepts causal adjustments
- **Validation**: Backtesting against 2022-2024 season

### Documentation References

- Detailed module documentation: `docs/causal/CAUSAL_INFERENCE_FRAMEWORK.md`
- Theoretical background: `analysis/dissertation/chapter_8_results_discussion/`

---

## Bayesian Hierarchical Models

### Overview

**Location**: `R/` and `models/bayesian/`
**Status**: ✅ v3.0 Complete
**Documentation**: `docs/ADVANCED_BAYESIAN_V3.md`

**Core Principle**: Hierarchical partial pooling leverages information across players, teams, and positions to reduce overfitting and provide principled uncertainty quantification.

### Canonical Training Window

All models use the established canonical training window (Nov 8, 2025):

**Training Window Specification**:
- **Core years**: 2018-2024 (full weight = 1.0)
- **Decay years**: 2015-2017 (exponential decay with half-life H=4 years)
- **Weight formula**: w(s; 2025, 4) = 0.5^((2025-s)/4)
- **Effective weights**:
  - 2017: 0.297
  - 2016: 0.250
  - 2015: 0.210
- **Total games**: 4,504 (2015-2024, weighted)
- **Validation**: Ablation study on 2025 weeks 1-6 (148 games, 37 QBs)
- **Statistical significance**: p=0.0069 improvement over 2020-2024 baseline (Diebold-Mariano test)

**Documentation**:
- `docs/TRAINING_WINDOW_ABLATION_RESULTS.md` (comprehensive ablation study)
- `docs/STATE_SPACE_FINAL_DECISION.md` (GO/NO-GO decision)

### Trained Models

#### 1. State-Space Random Walk (v2.0) - APPROVED

**File**: `models/bayesian/simplified_random_walk_v2.rds`
**Status**: ✅ **APPROVED for production ensemble (Nov 8, 2025)**

**Training Details**:
- Training time: 14.2 minutes
- Training data: 4,504 games (2015-2024, weighted with H=4 decay)
- Algorithm: Dynamic linear model with random walk state evolution
- MCMC: cmdstanr, 4 chains × 2000 iterations

**Performance Metrics**:
- MAE: 47.77 yards (2025 hold-out validation on 148 games)
- Coverage: 86.5% (well-calibrated on 90% credible intervals)
- Win rate: Expected +0.5-1.0% ROI improvement

**Validation**:
- Tested on completely unseen 2025 weeks 1-6 data
- Statistically significant improvement vs baseline (p=0.0069)
- Diebold-Mariano test confirms predictive power

**Loading and Using**:
```r
# Load state-space model
library(tidyverse)

model <- readRDS('models/bayesian/simplified_random_walk_v2.rds')

# Generate predictions for week 8
predictions <- predict(
  model,
  newdata = test_data,
  re.form = NA  # Population-level predictions
)

# Extract uncertainty
pred_df <- tibble(
  pred_mean = predictions$estimate,
  pred_lower = predictions$lower_ci_90,
  pred_upper = predictions$upper_ci_90
)
```

**Expected ROI Impact**: +0.5-1.0%

#### 2. Informative Priors (v3.0)

**File**: `models/bayesian/passing_informative_priors_v1.rds`
**Size**: 5.2 MB
**Status**: ⚠️ Needs v2 retraining with canonical window

**Training Details**:
- Training time: 25.3 seconds (fast!)
- Training data: 2,302 historical games (2015-2019)
- Priors: Empirical Bayes learned from historical data
- MCMC: 4 chains × 1000 iterations

**Performance**:
- MAE: ~52 yards (estimated)
- Expected ROI: +0.2-0.5%

**Status Note**: This model was trained on pre-2020 data. Retraining with the canonical window (2015-2024, H=4 decay) is recommended for consistency.

#### 3. QB-WR Chemistry (v3.0)

**File**: `models/bayesian/receiving_qb_chemistry_v1.rds`
**Size**: 210 MB
**Status**: ✅ Complete

**Training Details**:
- Training time: 30.5 minutes (most computationally intensive)
- Training data: 13,218 games with 2,168 unique QB-WR pairs
- Dyadic effects: Random slopes for QB-WR interactions
- MCMC: 4 chains × 2000 iterations

**Hierarchy**:
```
League
├── Position Group (QBs, WRs)
│   ├── Position (QB, WR, TE, RB-receiving)
│   │   ├── Team
│   │   │   └── Player
│   │   │       └── Game
```

**Key Features**:
- Models QB-WR chemistry as dyadic random effect
- Accounts for receiver skill variation by QB
- Partial pooling shares information across pairs

**Expected ROI Impact**: +0.5-1.0%

**Usage**:
```r
# Load QB-WR chemistry model
qb_wr_model <- readRDS('models/bayesian/receiving_qb_chemistry_v1.rds')

# Predict receiving yards with QB-WR specific effect
pred <- predict(qb_wr_model,
  newdata = tibble(
    qb_id = 'KB',
    receiver_id = 'JJ',
    air_yards = 12,
    epa = 0.5
  )
)

# Extract QB-WR chemistry effect
ranef_qb_wr <- ranef(qb_wr_model)$`qb_id:receiver_id`
chemistry_effect <- ranef_qb_wr[paste0('KB:JJ'), 'Estimate']

cat(f"KB-JJ chemistry effect: {chemistry_effect:.1f} yards\n")
```

#### 4. Bayesian Neural Network (v3.0)

**File**: `models/bayesian/bnn_passing_v1.pkl`
**Size**: 80 MB
**Status**: ✅ Complete (PyMC-based)

**Training Details**:
- Training time: ~2-3 hours (PyMC NUTS sampling)
- Training data: 2,163 games with 157-feature set
- Framework: PyMC with Aesara backend
- MCMC: 4 chains × 1000 iterations, 2000 tuning steps

**Architecture**:
- Input layer: 157 features
- Hidden layer 1: 128 neurons (ReLU)
- Hidden layer 2: 64 neurons (ReLU)
- Output layer: Passing yards (Gaussian likelihood)

**Performance**:
- MAE: 58.70 yards
- Coverage: 86.8% on 90% CI
- Expected ROI: +0.3-0.8%

**Uncertainty Quantification**:
```python
from nfl_analytics.models.bnn_calibrated_v2 import CalibratedBNN

# Load BNN
bnn = CalibratedBNN.load('models/bayesian/bnn_passing_v1.pkl')

# Get prediction with full posterior
features = np.array([...])  # 157-feature vector
pred = bnn.predict(features)

# Extract intervals
print(f"Point estimate: {pred.mean:.1f} yards")
print(f"90% credible interval: [{pred.lower_90:.1f}, {pred.upper_90:.1f}]")
print(f"Interval width: {pred.upper_90 - pred.lower_90:.1f} yards")
print(f"Posterior samples: {len(pred.posterior_samples)}")
```

#### 5. BNN Multi-Output (v3.1) - PRODUCTION READY

**File**: `models/bayesian/bnn_multioutput_v1_final.pt`
**Size**: 59 KB (model weights) + 500 MB (20 checkpoints)
**Status**: ✅ Complete, Production-ready
**Trained**: November 2, 2025 (Windows RTX 4090)

**Training Details**:
- Training time: ~10 minutes (500 epochs on RTX 4090)
- Framework: PyTorch with Bayesian uncertainty
- Batch size: 32
- Optimizer: AdamW (lr=0.00654764)
- Dropout: 0.116244 (for uncertainty)
- Hidden dimension: 448 neurons

**Multi-Output Architecture**:
```python
Input Features (157 dimensions)
    ↓
Hidden Layer 1: 448 neurons (ReLU) + Dropout(0.116)
    ↓
Hidden Layer 2: 224 neurons (ReLU) + Dropout(0.116)
    ↓
Output Head 1: Passing Yards (Gaussian)
Output Head 2: Rushing Yards (Gaussian)
Output Head 3: TD Prediction (Bernoulli)
```

**Performance Metrics**:
- **Passing Yards**: MAE=20.2 yards, RMSE=29.1 yards
- **Rushing Yards**: MAE=3.78 yards, RMSE=5.97 yards
- **TD Prediction**: Accuracy=80.0%
- **Calibration**: 89.5% coverage on 90% CI (temperature-scaled, T=0.75)
- **Interval width**: 86.4 yards (25% tighter than uncalibrated)

**Checkpoints Available**:
- `bnn_multioutput_v1_final.pt` - Final trained model
- `bnn_checkpoint_best.pt` - Best validation performance
- `bnn_checkpoint_epoch_{50,100,...,950}.pt` - Intermediate checkpoints for analysis

**Production Usage**:
```python
from nfl_analytics.models.bnn_calibrated_v2 import CalibratedBNN
import torch

# Load model
bnn = CalibratedBNN.load('models/bayesian/bnn_multioutput_v1_final.pt')

# Prepare batch of features
features = torch.randn(32, 157)  # Batch of 32 games

# Predict passing yards with uncertainty
passing_pred = bnn.predict_passing(features)
print(f"Passing yards: {passing_pred['mean']}")
print(f"90% CI: [{passing_pred['lower_90']}, {passing_pred['upper_90']}]")

# Predict rushing yards with uncertainty
rushing_pred = bnn.predict_rushing(features)
print(f"Rushing yards: {rushing_pred['mean']}")

# Predict TD probability
td_prob = bnn.predict_td(features)
print(f"TD probability: {td_prob['prob']}")
```

**Platform Support**:
- **Windows 11 + RTX 4090**: CUDA 12.1 (native GPU acceleration)
- **Mac M4 Pro**: PyTorch MPS backend (auto-fallback to CPU if needed)
- **Linux**: CUDA or CPU

#### 6. Calibrated BNN (v2.0) - PRODUCTION CLASS

**File**: `nfl_analytics/models/bnn_calibrated_v2.py`
**Status**: ✅ Production-ready with monitoring hooks

**Key Features**:
- Temperature scaling (T=0.75) for optimal calibration
- Coverage: 89.5% on 90% CI (target: 85-95%) ✅ **PERFECT**
- Interval width: 86.4 yards (25% tighter than baseline)
- Built-in monitoring hooks for production deployment

**Implementation Details**:
```python
from nfl_analytics.models.bnn_calibrated_v2 import CalibratedBNN

class CalibratedBNN:
    """Production-ready Bayesian Neural Network with calibration.

    Features:
    - Temperature scaling (default T=0.75)
    - Monitoring hooks for production tracking
    - Batch prediction support
    - Uncertainty quantification
    - Calibration validation
    """

    def __init__(self, model_path, temperature=0.75):
        self.model = self._load_model(model_path)
        self.temperature = temperature
        self.monitor_hooks = {}

    def predict(self, features, return_uncertainty=True):
        """Make predictions with optional uncertainty quantification.

        Args:
            features: Input features [batch_size, n_features]
            return_uncertainty: If True, return credible intervals

        Returns:
            PredictionResult with mean, lower_90, upper_90, posterior_samples
        """
        # ... implementation

    def predict_with_monitoring(self, features, request_id=None):
        """Make prediction and log to monitoring system.

        Args:
            features: Input features
            request_id: Unique identifier for tracking

        Returns:
            PredictionResult with monitoring metadata
        """
        # Calls self.monitor_hooks['prediction'](request_id, features, result)
```

**Calibration Validation**:
```python
# Verify 89.5% coverage on validation set
from nfl_analytics.models.bnn_calibrated_v2 import validate_calibration

results = validate_calibration(
    model=bnn,
    validation_data=val_loader,
    ci_level=0.90
)

print(f"Coverage: {results['coverage']:.1%}")  # Should be 89.5%
print(f"Target range: [85%, 95%]")
print(f"Status: {'✅ PERFECT' if 85 <= results['coverage']*100 <= 95 else '❌ NEEDS ADJUSTMENT'}")
```

### Training Instructions

#### Training a New State-Space Model

```r
# R script: R/train_state_space_model.R
library(tidyverse)
library(brms)
library(cmdstanr)

# Load data with canonical training window
games <- readRDS('data/games_with_weights.rds')

# Apply canonical weights (2015-2024 with H=4 decay on 2015-2017)
games <- games %>%
  mutate(
    weight = if_else(
      season >= 2018,
      1.0,
      0.5^((2025 - season) / 4)
    )
  )

# Fit state-space model
model <- brm(
  passing_yards ~ 1 + s(week, bs='gp') + (1 | qb_id) + (1 | team_id),
  data = games,
  family = gaussian(),
  chains = 4,
  iter = 2000,
  backend = 'cmdstanr',
  prior = c(
    prior(normal(220, 50), class = 'b'),
    prior(normal(0, 30), class = 'b', coef = 'weekweek'),
    prior(exponential(1), class = 'sigma')
  ),
  weights = games$weight
)

# Save model
saveRDS(model, 'models/bayesian/state_space_vX.rds')
```

#### Training Passing Model with Canonical Window

```bash
# From command line
Rscript R/train_and_save_passing_model.R \
  --start-year 2015 \
  --end-year 2024 \
  --apply-canonical-weights true \
  --output models/bayesian/passing_model_canonical_v1.rds
```

### Database Integration

**Data Source**: `mart.player_game_stats` and derived tables

**Feature Requirements**:
- Player statistics (passing_yards, completion_pct, etc.)
- Team context (season, team_id)
- Game metadata (week, season)
- EPA and advanced metrics

**SQL Query Example**:
```sql
-- Get training data for Bayesian models
SELECT
  pgs.game_id,
  pgs.player_id,
  pgs.passing_yards,
  pgs.week,
  pgs.season,
  p.position,
  p.team_id,
  gf.epa_offense,
  gf.win_probability_added
FROM mart.player_game_stats pgs
JOIN mart.player_hierarchy p ON pgs.player_id = p.player_id
JOIN mart.game_features_enhanced gf ON pgs.game_id = gf.game_id
WHERE pgs.season >= 2015
  AND p.position = 'QB'
ORDER BY pgs.season, pgs.week;
```

### Documentation References

- **Technical details**: `docs/ADVANCED_BAYESIAN_V3.md`
- **Training window ablation**: `docs/TRAINING_WINDOW_ABLATION_RESULTS.md`
- **State-space decision**: `docs/STATE_SPACE_FINAL_DECISION.md`
- **Dissertation chapters**: `analysis/dissertation/chapter_3/`, `chapter_4/`

---

## Ensemble Methods

### Overview

**Location**: `nfl_analytics/ensemble/enhanced_ensemble_v3.py`
**Status**: ✅ v3.0 Complete
**Documentation**: See this file + `docs/ADVANCED_BAYESIAN_V3.md`

**Architecture**: 4-way ensemble combining complementary modeling approaches:

1. **Bayesian Hierarchical** (brms/Stan) - Captures parametric structure
2. **XGBoost** (GPU-accelerated) - Captures non-linear patterns
3. **Bayesian Neural Network** (PyMC/PyTorch) - Deep learning with uncertainty
4. **State-Space Random Walk** (v2.0) ⭐ **APPROVED** - Temporal dynamics

### Component Models

| Component | File | Status | MAE | Key Strength |
|-----------|------|--------|-----|--------------|
| Bayesian Hierarchical | `informative_priors_v1.rds` | ✅ | ~52 yards | Principled priors |
| XGBoost GPU | `xgboost_gpu_v3.pkl` | ✅ | ~55 yards | Non-linear features |
| BNN PyMC | `bnn_passing_v1.pkl` | ✅ | 58.70 yards | Deep learning |
| State-Space | `simplified_random_walk_v2.rds` | ✅ | **47.77 yards** | Temporal patterns |
| **Ensemble (IVW)** | Combined | ✅ | **~47.5 yards** | Robust average |

**State-Space Approval** (Nov 8, 2025):
- MAE: 47.77 yards on 2025 hold-out (148 games)
- Coverage: 86.5% (well-calibrated)
- Statistical significance: p=0.0069 (Diebold-Mariano test)
- Decision: ✅ **APPROVED for production ensemble**

### Combination Methods

#### 1. Inverse Variance Weighting (Default)

**Advantage**: No meta-learner training required
**Disadvantage**: Assumes independence, doesn't capture correlations

```python
from nfl_analytics.ensemble.enhanced_ensemble_v3 import EnsemblePredictor
import numpy as np

# Initialize with component models
ensemble = EnsemblePredictor(
    bayesian_model_path='models/bayesian/informative_priors_v1.rds',
    xgboost_model_path='models/xgboost/v2_1.pkl',
    bnn_model_path='models/bayesian/bnn_passing_v1.pkl',
    state_space_model_path='models/bayesian/simplified_random_walk_v2.rds'
)

# Get predictions from all components
features = {...}  # 157-feature vector
predictions = ensemble.get_component_predictions(features)

# predictions = {
#   'bayesian': (mu=265, sigma=35),
#   'xgboost': (mu=268, sigma=40),
#   'bnn': (mu=270, sigma=45),
#   'state_space': (mu=272, sigma=32)
# }

# Combine via inverse variance weighting
ensemble_pred = ensemble.predict_inverse_variance(features)
# Weights proportional to 1/sigma²
# Result: mu=269.2 yards, sigma=19.8 yards

print(f"Ensemble prediction: {ensemble_pred.mean:.1f} ± {ensemble_pred.std:.1f}")
```

**Formula**:
```
μ_ensemble = Σ(w_i × μ_i) / Σ(w_i)
σ²_ensemble = 1 / Σ(w_i)

where w_i = 1/σ_i²
```

#### 2. Stacked Meta-Learner

**Advantage**: Learns optimal weights from data
**Disadvantage**: Requires training data, risk of overfitting

```python
from nfl_analytics.ensemble.enhanced_ensemble_v3 import StackedMetaLearner
from sklearn.model_selection import cross_val_score

# Prepare meta-features (component predictions on training set)
meta_features = ensemble.get_meta_features(X_train)
# Shape: (n_train, 8) - [bayesian_mean, bayesian_std, xgboost_mean, ...]

# Train meta-learner (simple linear regression)
meta_learner = StackedMetaLearner(
    meta_model='ridge',  # Ridge regression to avoid overfitting
    alpha=1.0  # Regularization strength
)

meta_learner.fit(meta_features, y_train)

# Extract learned weights
weights = meta_learner.get_weights()
print(f"Learned weights:")
print(f"  Bayesian: {weights['bayesian']:.3f}")
print(f"  XGBoost: {weights['xgboost']:.3f}")
print(f"  BNN: {weights['bnn']:.3f}")
print(f"  State-Space: {weights['state_space']:.3f}")

# Make predictions on test set
test_meta_features = ensemble.get_meta_features(X_test)
ensemble_pred = meta_learner.predict(test_meta_features)
```

#### 3. Portfolio Optimization (Kelly Criterion)

**Advantage**: Incorporates correlation structure, optimizes for profitability
**Disadvantage**: Requires precise return estimates

```python
from nfl_analytics.ensemble.enhanced_ensemble_v3 import PortfolioOptimizer

# Get correlation matrix of component predictions
pred_matrix = ensemble.get_prediction_matrix(X_backtest)  # (n_samples, 4)
# Columns: [bayesian, xgboost, bnn, state_space]

# Compute correlation
corr_matrix = np.corrcoef(pred_matrix.T)
print("Component correlations:")
print(corr_matrix)

# Expected returns from each component (based on backtest)
expected_returns = np.array([0.035, 0.038, 0.032, 0.042])  # +3.5%, +3.8%, etc.

# Optimize using correlation-adjusted Kelly criterion
optimizer = PortfolioOptimizer(
    correlation_matrix=corr_matrix,
    expected_returns=expected_returns,
    kelly_factor=0.25  # Fractional Kelly (safer)
)

optimal_weights = optimizer.optimize()
print(f"Optimal weights: {optimal_weights}")

# Apply weights
ensemble_pred = (
    optimal_weights['bayesian'] * predictions['bayesian'] +
    optimal_weights['xgboost'] * predictions['xgboost'] +
    optimal_weights['bnn'] * predictions['bnn'] +
    optimal_weights['state_space'] * predictions['state_space']
)
```

### Usage Examples

#### Basic Ensemble Prediction

```bash
# Command-line usage
uv run python nfl_analytics/ensemble/enhanced_ensemble_v3.py \
  --bayesian models/bayesian/informative_priors_v1.rds \
  --xgboost models/xgboost/v2_1.pkl \
  --bnn models/bayesian/bnn_passing_v1.pkl \
  --state-space models/bayesian/simplified_random_walk_v2.rds \
  --features analysis/features/asof_team_features.csv \
  --output predictions/ensemble_v3_week8.csv \
  --combination-method inverse_variance
```

#### Python API

```python
from nfl_analytics.ensemble.enhanced_ensemble_v3 import EnsemblePredictor
import pandas as pd

# Load features
features_df = pd.read_csv('analysis/features/asof_team_features.csv')

# Initialize ensemble
ensemble = EnsemblePredictor(
    bayesian_model_path='models/bayesian/informative_priors_v1.rds',
    xgboost_model_path='models/xgboost/v2_1.pkl',
    bnn_model_path='models/bayesian/bnn_passing_v1.pkl',
    state_space_model_path='models/bayesian/simplified_random_walk_v2.rds',
    combination_method='inverse_variance'
)

# Make predictions
predictions = ensemble.predict(features_df)

# Output: DataFrame with columns
# - game_id, player_id, prediction, lower_90, upper_90, confidence
# - component_bayesian, component_xgboost, component_bnn, component_state_space

predictions.to_csv('predictions/ensemble_predictions.csv', index=False)
```

#### Full Backtest

```python
from nfl_analytics.ensemble.enhanced_ensemble_v3 import EnsembleBacktest

backtest = EnsembleBacktest(
    start_season=2022,
    end_season=2024,
    ensemble=ensemble
)

results = backtest.run()
# Returns: ROI, win_rate, max_drawdown, sharpe_ratio, calibration_metrics

print(f"Ensemble Backtest Results (2022-2024):")
print(f"  ROI: {results['roi']:.2%}")
print(f"  Win Rate: {results['win_rate']:.1%}")
print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"  Max Drawdown: {results['max_drawdown']:.1%}")
```

### Integration with Production

**Bet Sizing**: Works with CQL bet sizer in `nfl_analytics/production/cql_bet_sizer.py`

```python
from nfl_analytics.production.cql_bet_sizer import CQLBetSizer

bet_sizer = CQLBetSizer(
    ensemble_predictor=ensemble,
    kelly_factor=0.25  # Fractional Kelly (conservative)
)

# Get optimal bet size
game = {
    'player_id': 'KB',
    'prop_type': 'passing_yards',
    'over_under': 260.5,
    'implied_prob': 0.515
}

bet = bet_sizer.size_bet(
    ensemble_pred=ensemble_pred,  # From ensemble
    game=game,
    bankroll=10000,
    max_risk_per_game=250
)

print(f"Optimal bet: {bet['size']:.2f} on {'OVER' if bet['action']=='over' else 'UNDER'}")
```

---

## Reinforcement Learning (CQL)

### Overview

**Location**: `nfl_analytics/rl/`
**Status**: ✅ Complete (Oct 9, 2025)
**Documentation**: `docs/milestones/CQL_COMPLETE_SUMMARY.md`

**Purpose**: Learn optimal bet sizing policy from historical betting data using Conservative Q-Learning (CQL) - an offline RL algorithm that avoids overestimating value of out-of-distribution actions.

### Training Results

**Environment**: Windows 11 + RTX 4090 GPU (CUDA 12.9)
**Training Time**: ~9 minutes (2000 epochs)
**Dataset**: 5,146 games (2006-2024 NFL seasons)

**Performance Metrics**:

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| Match Rate | 98.5% | Policy matches logged behavior 98.5% of the time |
| Estimated Reward | 1.75% | Expected ROI vs 1.41% baseline (**24% improvement**) |
| Q-value Range | [-0.5, 0.8] | Stable value estimates |
| Policy Divergence | 0.12 | KL-divergence from behavior policy |

**Comparison to Baseline**:
- **Baseline (Kelly Criterion)**: 1.41% ROI
- **CQL Policy**: 1.75% ROI
- **Improvement**: +0.34 pp (+24% relative)

### CQL Algorithm

**Conservative Q-Learning** addresses the offline RL problem:

1. **Offline Dataset**: Logged betting decisions with outcomes
2. **Pessimism**: Avoid overestimating value of high OOD actions
3. **Conservative Penalty**: -α × max_a Q(s,a) added to loss

**Implementation**:
```python
# Simplified CQL update rule
Q_loss = (expected_return - Q(s,a))^2  # TD loss
Q_loss -= α * max_a' Q(s, a')  # Conservative penalty
Q_loss += β * KL(π_learned, π_behavior)  # Regularization
```

### Model Details

**File**: `models/cql/best_model.pth`
**Size**: 207 KB
**Framework**: PyTorch

**Network Architecture**:
```
Input: [bankroll, edge, uncertainty, odds] (4 dimensions)
    ↓
Hidden Layer 1: 256 neurons (ReLU)
    ↓
Hidden Layer 2: 128 neurons (ReLU)
    ↓
Output: Q-values for 5 bet sizing actions
  - SKIP (0%)
  - CONSERVATIVE (10%)
  - MODERATE (25%)
  - AGGRESSIVE (50%)
  - MAXIMUM (100% of bankroll)
```

**State Space**:
```python
state = {
    'bankroll': float,  # Current bankroll
    'edge': float,      # Estimated edge vs market (-0.1 to 0.1)
    'uncertainty': float, # Model uncertainty (std of ensemble)
    'odds': float       # Implied probability from line
}
# Total: 4 continuous dimensions
```

**Action Space**:
```python
actions = {
    0: {'bet_size': 0.00, 'name': 'SKIP'},
    1: {'bet_size': 0.10, 'name': 'CONSERVATIVE'},
    2: {'bet_size': 0.25, 'name': 'MODERATE'},
    3: {'bet_size': 0.50, 'name': 'AGGRESSIVE'},
    4: {'bet_size': 1.00, 'name': 'MAXIMUM'}
}
```

### Training Script

**File**: `nfl_analytics/rl/train_cql_agent.py`

```python
from nfl_analytics.rl.cql_agent import CQLAgent
from nfl_analytics.rl.dataset import OfflineBettingDataset
import torch

# Load offline dataset
dataset = OfflineBettingDataset(
    start_season=2006,
    end_season=2024,
    min_edge=0.01  # Only include games with >1% edge
)

print(f"Dataset: {len(dataset)} games, {dataset.mean_return:.2%} avg return")

# Initialize agent
agent = CQLAgent(
    state_dim=4,
    action_dim=5,
    hidden_dim=256,
    learning_rate=1e-3,
    alpha=0.1,  # Conservative penalty weight
    beta=0.5    # Behavior regularization
)

# Train
agent.train(
    dataset=dataset,
    batch_size=32,
    epochs=2000,
    val_fraction=0.2,
    save_best=True,
    save_path='models/cql/best_model.pth'
)

# Evaluate
eval_results = agent.evaluate(dataset)
print(f"Evaluation Results:")
print(f"  Match Rate: {eval_results['match_rate']:.1%}")
print(f"  Estimated Return: {eval_results['estimated_return']:.2%}")
print(f"  Policy Divergence: {eval_results['policy_divergence']:.3f}")
```

### Usage in Production

**File**: `nfl_analytics/production/cql_bet_sizer.py`

```python
from nfl_analytics.production.cql_bet_sizer import CQLBetSizer
import torch

# Load trained CQL model
cql_agent = torch.load('models/cql/best_model.pth')

# Initialize bet sizer
bet_sizer = CQLBetSizer(
    cql_agent=cql_agent,
    bankroll=10000,
    risk_gates={
        'min_edge': 0.01,  # Minimum 1% edge
        'max_uncertainty': 50,  # Max model uncertainty (yards)
        'min_confidence': 0.55,  # Min win probability
        'odds_requirement': 0.90  # Min odds quality
    }
)

# Example: Size bet for a game
state = {
    'bankroll': 10000,
    'edge': 0.045,  # 4.5% edge vs market
    'uncertainty': 35,  # Ensemble std
    'odds': 0.52  # 52% implied prob
}

action = bet_sizer.decide(state)
# Returns: {'action': 2, 'bet_size': 0.25, 'amount': 2500, 'rationale': '...'}

print(f"Recommended bet: ${action['amount']:.0f}")
print(f"Rationale: {action['rationale']}")
```

### Platform Support

**Automatic Detection**:
```python
# Detects device automatically
device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')

# Training locations
# - Windows 11 + RTX 4090: CUDA 12.9 (recommended, fastest)
# - Mac M4 Pro: MPS backend (good performance)
# - Fallback: CPU (slowest)
```

### Validation

**Off-Policy Evaluation** (`nfl_analytics/rl/ope_gate.py`):

```python
from nfl_analytics.rl.ope_gate import OffPolicyEvaluator

ope = OffPolicyEvaluator(
    behavior_policy='logged',  # Historical decisions
    eval_policy=agent,         # Trained CQL policy
    dataset=test_dataset
)

# Estimate true return of new policy
estimated_return = ope.estimate_return(
    method='wdr',  # Weighted doubly robust
    bootstrap_samples=1000
)

print(f"Estimated return: {estimated_return['mean']:.2%}")
print(f"95% CI: [{estimated_return['ci_lower']:.2%}, {estimated_return['ci_upper']:.2%}]")
```

---

## Distributed Compute System

### Overview

**Location**: `nfl_analytics/compute/`, `run_compute.py`
**Status**: ✅ Complete
**Documentation**: README.md section 7

**Purpose**: Enable SETI@home-style distributed training between MacBook M4 (CPU-optimized) and Windows RTX 4090 (GPU-optimized) with automatic hardware routing and Google Drive synchronization.

### Architecture

**Three-Layer System**:

1. **Scheduling Layer** (`nfl_analytics/compute/adaptive_scheduler.py`)
   - Multi-armed bandit for task allocation
   - Learns which hardware excels at which tasks
   - Epsilon-greedy exploration vs exploitation

2. **Routing Layer** (`nfl_analytics/compute/hardware/task_router.py`)
   - CPU tasks → M4 (MCMC, state-space modeling, statistical tests)
   - GPU tasks → 4090 (BNN training, XGBoost, RL)
   - Fallback logic if hardware unavailable

3. **Synchronization Layer** (`nfl_analytics/compute/sync/`)
   - Google Drive bidirectional sync
   - Task queue: SQLite with WAL mode
   - Automatic conflict resolution
   - Cross-platform file locking

### Supported Hardware

#### MacBook M4 Pro (Development)
- **Best for**: CPU-intensive modeling
- **Supported tasks**:
  - MCMC (Stan, cmdstanr, pymc)
  - State-space modeling
  - Statistical testing (Diebold-Mariano, etc.)
  - Feature engineering
  - Data validation

#### Windows 11 RTX 4090 (Compute)
- **Best for**: GPU-intensive training
- **Supported tasks**:
  - BNN training (PyTorch, PyMC)
  - XGBoost with CUDA
  - RL (CQL agent training)
  - Hyperparameter optimization
  - Large-scale backtest

### Task Queue Structure

**Database**: `nfl_analytics/compute/task_queue.db` (SQLite, WAL mode)

**Tables**:
```sql
CREATE TABLE tasks (
    task_id TEXT PRIMARY KEY,
    task_type TEXT,  -- 'mcmc', 'bnn_train', 'xgboost', etc.
    hardware_hint TEXT,  -- 'cpu', 'gpu', 'any'
    priority INTEGER DEFAULT 5,  -- 1=critical, 5=normal, 10=low
    status TEXT,  -- 'pending', 'in_progress', 'completed', 'failed'
    assigned_hardware TEXT,
    created_at TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    parameters JSON,  -- Task-specific configuration
    output_path TEXT  -- Where to save results
);

CREATE TABLE hardware_metrics (
    hardware_id TEXT,
    task_type TEXT,
    avg_duration_seconds REAL,
    success_rate REAL,
    last_update TIMESTAMP
);
```

### Setup Instructions

#### Initial Setup

```bash
# Clone repo and create virtual environment
git clone https://github.com/raold/nfl-analytics.git
cd nfl-analytics

# Python environment
uv venv .venv && source .venv/bin/activate
uv sync --extra dev

# Initialize compute system
python run_compute.py --init

# Check system readiness
python run_compute.py --health-check
```

#### Configure Google Drive Sync

```bash
# Authorize Google Drive access
python -c "from nfl_analytics.compute.sync.google_drive_sync import setup_auth; setup_auth()"

# Verify sync
python -c "from nfl_analytics.compute.sync.google_drive_sync import GoogleDriveSync; \
  sync = GoogleDriveSync('nfl-analytics-compute'); \
  print(f'Root folder: {sync.root_folder_id}')"
```

### Command Reference

#### Run Compute System

```bash
# Start with default intensity (medium)
python run_compute.py --intensity medium

# Options: low, medium, high
# - low: 1-2 tasks/hour, minimal resource usage
# - medium: 4-6 tasks/hour, normal usage
# - high: 10+ tasks/hour, maximize throughput
```

#### Check Status

```bash
# Quick status
python run_compute.py --status

# Full dashboard
python run_compute.py --dashboard

# Specific hardware
python run_compute.py --hardware-status m4  # or 'rtx4090'
```

#### Task Management

```bash
# Queue a new task
python run_compute.py --queue-task \
  --type mcmc \
  --hardware-hint cpu \
  --priority 5 \
  --config '{"model": "state_space", "season_start": 2015}'

# Monitor specific task
python run_compute.py --monitor-task abc123def456

# Retry failed task
python run_compute.py --retry-task abc123def456

# Cancel task
python run_compute.py --cancel-task abc123def456
```

#### Sync Control

```bash
# Manual sync to Google Drive
python run_compute.py --sync-push

# Manual sync from Google Drive
python run_compute.py --sync-pull

# Verify sync integrity
python run_compute.py --verify-sync
```

### Task Router Configuration

**File**: `nfl_analytics/compute/hardware/task_router.py`

```python
from nfl_analytics.compute.hardware.task_router import task_router

# View routing decisions
routing = task_router.get_routing_report()
print(routing)
# Output:
# Task Type           CPU Score  GPU Score  Assigned
# ────────────────────────────────────────────────
# state_space_mcmc    9.8        2.1        → M4 (CPU)
# bnn_train           3.2        9.7        → RTX4090 (GPU)
# xgboost_train       5.5        8.9        → RTX4090 (GPU)
# feature_eng         8.5        3.2        → M4 (CPU)
# backtest            7.2        6.8        → Either (slightly favor M4)
```

**Customization**:
```python
# Override default routing
task_router.override_routing(
    task_type='bnn_train',
    hardware='m4'  # Force CPU training (slow, for testing)
)

# Reset to defaults
task_router.reset_routing()
```

### Example Workflow

**Scenario**: Train state-space model on M4, BNN on RTX 4090, synchronize results

```bash
# Step 1: M4 queues tasks
python run_compute.py --queue-task \
  --type mcmc \
  --hardware-hint cpu \
  --priority 8  # High priority
  --config '{
    "model": "state_space",
    "season_start": 2015,
    "season_end": 2024,
    "chains": 4,
    "output": "models/bayesian/state_space_v3.rds"
  }'

# Step 2: M4 starts compute
python run_compute.py --intensity high
# Runs MCMC sampling on M4 (14-15 minutes)

# Step 3: When complete, sync to Google Drive
python run_compute.py --sync-push

# Step 4: Windows RTX 4090 pulls and queues BNN training
python run_compute.py --sync-pull
python run_compute.py --queue-task \
  --type bnn_train \
  --hardware-hint gpu \
  --priority 8
  --config '{
    "model": "multioutput",
    "batch_size": 32,
    "epochs": 500,
    "output": "models/bayesian/bnn_v4.pt"
  }'

# Step 5: RTX 4090 trains BNN (8-10 minutes)
python run_compute.py --intensity high

# Step 6: Results sync back to M4
python run_compute.py --sync-push  # From Windows
python run_compute.py --sync-pull  # To M4
```

### Monitoring and Logging

**Logs Location**: `logs/compute_system.log`

```bash
# Tail logs
tail -f logs/compute_system.log

# Filter by component
grep "TaskRouter" logs/compute_system.log
grep "GoogleDriveSync" logs/compute_system.log

# Analyze performance
python -c "from nfl_analytics.compute.analytics import ComputeAnalytics; \
  analytics = ComputeAnalytics('logs/'); \
  print(analytics.generate_report())"
```

**Metrics Dashboard**:
```bash
# View Prometheus metrics (if monitoring stack deployed)
curl http://localhost:9090/api/v1/query?query=compute_task_duration_seconds
```

---

## Dissertation

### Overview

**Location**: `analysis/dissertation/main/main.tex`
**Status**: ✅ Complete - 324 pages, PDF compiled
**Last Updated**: October 19, 2025

**Purpose**: Comprehensive technical documentation of the entire NFL analytics system from first principles through production deployment.

### Structure

**Main Document**: `analysis/dissertation/main/main.tex`

**324 Pages Total**:
- 12 main chapters
- 8 appendix chapters
- 45+ figures and tables
- 200+ references

### Chapter Overview

| Chapter | Title | Pages | Focus |
|---------|-------|-------|-------|
| 1 | Introduction | 15 | Motivation, research questions |
| 2 | Literature Review | 28 | Bayesian methods, causal inference, RL, ensemble methods |
| 3 | Data Foundation | 22 | Database schema, feature engineering, data quality |
| 4 | Baseline Models | 35 | GLM baseline, hierarchical Bayesian, informative priors |
| 5 | Reinforcement Learning | 18 | CQL algorithm, training results, policy validation |
| 6 | Advanced Bayesian Methods | 25 | State-space modeling, QB-WR chemistry, BNNs |
| 7 | Causal Inference | 30 | Panel data, treatment definitions, DiD, synthetic control |
| 8 | Results & Discussion | 40 | Performance comparison, calibration, leakage analysis |
| 9 | Production Deployment | 28 | Monitoring, A/B testing, risk management |
| 10 | Conclusion | 12 | Summary, future work |
| 11 | Offline RL | 15 | Offline RL for bet sizing, policy evaluation |
| 12 | Profit Optimization | 18 | Portfolio optimization, Kelly criterion |

**Appendices**:
- A: Data Leakage Analysis
- B: Causal DAGs (Technical Details)
- C: BNN Calibration
- D: Feature Engineering Details
- E: Code Snippets (Reproducibility)
- F: Experimental Protocols
- G: Benchmark Details
- H: Future Work & Extensions

### Figure Standardization

**Style**: `analysis/dissertation/style/tikz_diagram_style.tex`

**TikZ Specifications**:
- **Color Palette**: Accent blue (RGB: 31,119,180)
- **Backgrounds**: Light blue (`accent!3`)
- **Borders**: Blue (`accent!60`)
- **Corners**: Rounded (5pt radius)
- **Arrows**: Thick with arrowheads

**Usage**:
```latex
% In any chapter .tex file
\usepackage{tikz}
\usetikzlibrary{shapes, arrows, positioning}
\input{../style/tikz_diagram_style.tex}

% Create diagram
\begin{tikzpicture}[node distance=2cm]
  \node[process] (n1) {Data Ingestion};
  \node[process, right of=n1] (n2) {Feature Engineering};
  \node[process, right of=n2] (n3) {Model Training};

  \draw[->,thick] (n1) -- (n2);
  \draw[->,thick] (n2) -- (n3);
\end{tikzpicture}
```

### Compilation Instructions

#### Full Compilation (All Chapters + Index)

```bash
cd analysis/dissertation/main

# Step 1: Clean previous build
rm -f main.pdf main.aux main.log main.out

# Step 2: Run pdflatex (creates aux file)
pdflatex main.tex

# Step 3: Run bibtex (processes bibliography)
bibtex main

# Step 4: Run pdflatex again (resolves references)
pdflatex main.tex

# Step 5: Run pdflatex final time (updates index, page numbers)
pdflatex main.tex

# Output: main.pdf (324 pages, 5.48 MB)
```

#### Fast Compilation (No Index Update)

```bash
cd analysis/dissertation/main

# For editing: skip bibtex
pdflatex main.tex
pdflatex main.tex
# Output: main.pdf (faster, 1-2 minutes)
```

#### Check for Warnings/Errors

```bash
# View all warnings
grep -i "warning" main.log | head -20

# View all errors
grep -i "error" main.log

# View specific warnings
grep "Overfull\|Underfull" main.log  # Line fitting issues
grep "Citation" main.log              # Missing citations
grep "Reference" main.log             # Undefined references
```

### File Organization

```
analysis/dissertation/
├── main/
│   ├── main.tex                    # Master document
│   └── main.pdf                    # Compiled output (324 pages)
├── chapter_1_introduction/
│   └── chapter_1_introduction.tex
├── chapter_2_literature_review/
│   └── chapter_2_literature_review.tex
├── chapter_3_data_foundation/
│   └── chapter_3_data_foundation.tex
├── chapter_4_baseline_modeling/
│   └── chapter_4_baseline_modeling.tex
├── ... (chapters 5-10)
├── chapter_11_offline_rl/
│   └── chapter_11_offline_rl.tex
├── chapter_12_profit_optimization/
│   └── chapter_12_profit_optimization.tex
├── appendix/
│   ├── appendix_a_data_leakage.tex
│   ├── appendix_b_causal_dags.tex
│   ├── appendix_c_bnn_calibration.tex
│   ├── ... (appendices D-H)
├── figures/
│   ├── out/                        # Generated outputs
│   │   ├── *.pdf                   # LaTeX-generated figures
│   │   ├── *.tex                   # TikZ source
│   │   └── *.csv                   # Table data
│   └── R/                          # R scripts to generate figures
│       ├── fig_roi_comparison.R
│       ├── fig_calibration_curves.R
│       └── ... (30+ figure scripts)
├── style/
│   ├── tikz_diagram_style.tex      # Standardized TikZ style
│   ├── dissertation_preamble.tex   # Global preamble
│   └── preamble/
│       ├── packages.tex             # Package includes
│       ├── custom_commands.tex      # Custom LaTeX commands
│       └── formatting.tex           # Formatting rules
├── docs/
│   ├── DISSERTATION_COMPILATION_GUIDE.md
│   ├── FIGURE_GENERATION_GUIDE.md
│   └── CONTRIBUTING_TO_DISSERTATION.md
└── bibliography/
    └── references.bib              # Bibliography database
```

### Editing Workflow

#### Adding New Figure

```bash
# Step 1: Create R script to generate figure
cat > analysis/dissertation/figures/R/fig_new_result.R << 'EOF'
library(ggplot2)
library(dplyr)

# Query data or load CSV
data <- read_csv('analysis/results/new_metric.csv')

# Create plot
p <- ggplot(data, aes(x = week, y = metric)) +
  geom_point() +
  geom_smooth(method = 'loess') +
  theme_minimal() +
  labs(title = 'New Result', x = 'Week', y = 'Metric')

# Save as PDF
ggsave('../out/fig_new_result.pdf', p, width = 6, height = 4)
EOF

# Step 2: Run script to generate PDF
Rscript analysis/dissertation/figures/R/fig_new_result.R

# Step 3: Include in chapter LaTeX
cat >> analysis/dissertation/chapter_8_results_discussion/chapter_8_results_discussion.tex << 'EOF'

\section{New Result}

\begin{figure}[h]
  \centering
  \includegraphics[width=0.8\textwidth]{../figures/out/fig_new_result.pdf}
  \caption{Description of new result.}
  \label{fig:new_result}
\end{figure}

See Figure \ref{fig:new_result} for details.
EOF

# Step 4: Recompile
cd analysis/dissertation/main
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

#### Adding New Chapter

```bash
# Step 1: Create chapter directory
mkdir analysis/dissertation/chapter_13_new_topic

# Step 2: Create chapter file
cat > analysis/dissertation/chapter_13_new_topic/chapter_13_new_topic.tex << 'EOF'
\chapter{New Topic}
\label{ch:new_topic}

\section{Introduction}

\section{Methods}

\section{Results}

\section{Discussion}

\section{Conclusion}

EOF

# Step 3: Add to main.tex
# Edit analysis/dissertation/main/main.tex
# Add: \input{../chapter_13_new_topic/chapter_13_new_topic.tex}

# Step 4: Recompile main.tex (see Compilation Instructions above)
```

#### Updating Bibliography

```bash
# Step 1: Add citation to bibliography/references.bib
cat >> analysis/dissertation/bibliography/references.bib << 'EOF'

@article{AuthorYear,
  title={Title of Paper},
  author={Author, A and Author, B},
  journal={Journal Name},
  year={2024},
  volume={1},
  pages={1--10}
}
EOF

# Step 2: Use citation in LaTeX
# In chapter .tex: This result is important \cite{AuthorYear}.

# Step 3: Recompile with bibtex
cd analysis/dissertation/main
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Key Chapters for Different Audiences

**For Statisticians**: Chapters 2, 4, 6, Appendix C (Calibration)
**For ML Engineers**: Chapters 2, 5, 6, 9, 11
**For Decision-Makers**: Chapters 1, 8, 9, 10, 12
**For Reproducibility**: Appendices E, F, G

### References and Citations

**Bibliography File**: `analysis/dissertation/bibliography/references.bib`

**Citation Count**: 200+ references covering:
- Bayesian methods (20+)
- Causal inference (25+)
- Reinforcement learning (15+)
- Sports analytics (30+)
- Ensemble methods (20+)
- Production systems (15+)

**Citation Format**: BibTeX (automatically formatted)

### Troubleshooting

**Missing Figures**:
```bash
# Regenerate all figures
for script in analysis/dissertation/figures/R/*.R; do
  Rscript "$script"
done

# Then recompile dissertation
cd analysis/dissertation/main
pdflatex main.tex
```

**Compilation Errors**:
```bash
# Check main.log for errors
tail -50 analysis/dissertation/main/main.log

# Common issues:
# - Missing packages: Add to preamble/packages.tex
# - Undefined references: Run pdflatex + bibtex + pdflatex again
# - File not found: Check relative paths in chapter files
```

**Slow Compilation**:
```bash
# Use fast compilation (skip index)
pdflatex -interaction=nonstopmode main.tex

# Or compile individual chapters first
pdflatex -jobname=chapter_4 chapter_4_baseline_modeling/chapter_4_baseline_modeling.tex
```

---

## Cross-Component Integration

### Data Flow

```
Raw NFL Data (schedules, plays)
    ↓
PostgreSQL Database (public schema)
    ↓
Feature Engineering (157 features)
    ↓
Materialized Views (mart schema)
    ↓
┌─────────────────────────────────────────────────────────┐
│  4-Way Ensemble Prediction                              │
├─────────────────────────────────────────────────────────┤
│ ├─ Bayesian Hierarchical (brms/Stan)                   │
│ ├─ XGBoost (GPU-accelerated)                           │
│ ├─ BNN (PyTorch with Calibration)                      │
│ └─ State-Space Random Walk v2 (APPROVED)               │
└─────────────────────────────────────────────────────────┘
    ↓
Inverse Variance Weighting (or Meta-learner)
    ↓
CQL Bet Sizer (RL policy)
    ↓
Final Predictions + Bet Sizes
    ↓
Production Deployment (Monitoring, A/B Testing)
```

### Testing Integration

**Unit Tests**: `tests/unit/`
**Integration Tests**: `tests/integration/`
**E2E Tests**: `tests/e2e/`

```bash
# Run all tests
pytest tests/ -v

# Run specific component tests
pytest tests/integration/test_ensemble.py -v
pytest tests/integration/test_causal_framework.py -v
pytest tests/integration/test_bnn_calibration.py -v
```

---

## Quick Reference

### Common Commands

```bash
# Training
Rscript R/train_and_save_passing_model.R
uv run python nfl_analytics/models/train_bnn_passing.py
python nfl_analytics/rl/train_cql_agent.py

# Prediction
uv run python nfl_analytics/ensemble/enhanced_ensemble_v3.py --help
python nfl_analytics/production/cql_bet_sizer.py

# Analysis
uv run python nfl_analytics/causal/validation.py

# Dissertation
cd analysis/dissertation/main && pdflatex main.tex && bibtex main && pdflatex main.tex

# Compute System
python run_compute.py --status
python run_compute.py --dashboard
```

### Documentation Hub

- **CLAUDE.md**: System-wide documentation
- **PROJECT_STATUS.md**: Current status and recent work
- **DOCUMENTATION_INDEX.md**: Complete documentation catalog
- **docs/**: Component-specific documentation
- **analysis/dissertation/**: Comprehensive technical reference

---

**Version**: 1.0
**Created**: November 8, 2025
**Maintainer**: Claude Code (Sonnet 4.5)
