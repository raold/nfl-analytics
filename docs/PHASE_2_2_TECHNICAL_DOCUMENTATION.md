# Phase 2.2: Mixture-of-Experts Bayesian Neural Network
## Technical Documentation

**Version:** 2.2
**Date:** October 2025
**Status:** Production-Ready
**Author:** Richard Oldham

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Model Training](#model-training)
4. [Feature Engineering](#feature-engineering)
5. [Uncertainty Quantification](#uncertainty-quantification)
6. [Calibration Metrics](#calibration-metrics)
7. [Production Implementation](#production-implementation)
8. [Performance Analysis](#performance-analysis)
9. [Limitations & Future Work](#limitations--future-work)

---

## Overview

### Motivation

Phase 1 baseline models achieved strong point estimate accuracy (MAE ~18 yards) but suffered from severe under-calibration:
- **90% Confidence Interval Coverage:** 26.0% (target: 90%)
- **Problem:** Uncertainty estimates were too narrow, leading to overconfident predictions
- **Impact:** Cannot reliably use uncertainty for bet sizing or risk management

### Solution: Mixture-of-Experts Architecture

Phase 2.2 introduces a Mixture-of-Experts (MoE) Bayesian Neural Network that:
- Models **heterogeneous uncertainty** across different player types
- Uses **expert specialization** for low/medium/high variance regimes
- Achieves **92.2% calibration** (90% CI coverage)
- Maintains **excellent point estimates** (MAE 18.5 yards)

### Key Innovation

Traditional BNNs assume homogeneous uncertainty (single σ parameter). Real rushing yards have heterogeneous variance:
- **Bench players:** Low variance (20-40 yards, consistent role)
- **Starters:** Medium variance (50-80 yards, game flow dependent)
- **Stars:** High variance (80-150 yards, high ceiling/floor)

The MoE architecture learns **which expert to use** for each prediction via a gating network.

---

## Architecture

### Model Structure

```
Input (4 features)
    │
    ├───► GATING NETWORK ────► Softmax ────► Expert Weights [g₀, g₁, g₂]
    │
    ├───► EXPERT 0 (Low Variance)    ────► Output₀, σ₀ ~ HalfNormal(10)
    ├───► EXPERT 1 (Medium Variance) ────► Output₁, σ₁ ~ HalfNormal(15)
    └───► EXPERT 2 (High Variance)   ────► Output₂, σ₂ ~ HalfNormal(20)
          │
          └───► Weighted Combination
                 μ = Σ(gᵢ × Outputᵢ)
                 σ = Σ(gᵢ × σᵢ)
                 │
                 └───► Likelihood: y ~ Normal(μ, σ)
```

### Gating Network

Maps input features to expert selection probabilities:

```python
W_gate: [D × K] where D=4 features, K=3 experts
gate_logits = X @ W_gate
gates = softmax(gate_logits)  # Sum to 1 for each sample
```

**Prior:** `W_gate ~ Normal(0, 0.5)` (weakly informative)

**Interpretation:**
- `gates[i, 0]` = probability sample `i` uses Expert 0
- `gates[i, 1]` = probability sample `i` uses Expert 1
- `gates[i, 2]` = probability sample `i` uses Expert 2

### Expert Networks

Each expert is a simple 2-layer neural network:

```python
# Expert k (k ∈ {0, 1, 2})
W_h: [D × H] where H=16 hidden units
b_h: [H]
h = tanh(X @ W_h + b_h)

W_out: [H × 1]
b_out: scalar
output_k = (h @ W_out).flatten() + b_out

# Priors
W_h ~ Normal(0, 1)
b_h ~ Normal(0, 1)
W_out ~ Normal(0, 1)
b_out ~ Normal(0, 5)  # Allow for larger intercepts
```

### Expert-Specific Uncertainty

Each expert has its own noise parameter:

```python
σ₀ ~ HalfNormal(10)   # Low variance expert
σ₁ ~ HalfNormal(15)   # Medium variance expert
σ₂ ~ HalfNormal(20)   # High variance expert
```

**Interpretation:**
- Bench players (low carries) → Expert 0 → tight uncertainty
- Starters (medium carries) → Expert 1 → moderate uncertainty
- Stars (high carries) → Expert 2 → wide uncertainty

### Weighted Output

The final prediction is a weighted combination:

```python
μ = Σᵢ (gates[:, i] × output_i)
σ = Σᵢ (gates[:, i] × σᵢ)

y_obs ~ Normal(μ, σ)
```

**Key Property:** Each sample gets a **personalized uncertainty estimate** based on which experts it activates.

---

## Model Training

### Training Data

```sql
SELECT
    player_id,
    season,
    week,
    stat_yards,
    stat_attempts as carries,
    AVG(stat_yards) OVER (
        PARTITION BY player_id
        ORDER BY season, week
        ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
    ) as avg_rushing_l3,
    AVG(stat_yards) OVER (
        PARTITION BY player_id, season
        ORDER BY week
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) as season_avg
FROM mart.player_game_stats
WHERE season BETWEEN 2020 AND 2024
  AND stat_category = 'rushing'
  AND position_group IN ('RB', 'FB', 'HB')
  AND stat_attempts >= 5
  AND stat_yards IS NOT NULL
```

**Training Set:**
- **Seasons:** 2020-2024 (Week 1-6 of 2024)
- **Samples:** 2,663 rushing performances
- **Players:** ~300 unique running backs
- **Date Range:** Sept 2020 - Oct 2024

**Test Set:**
- **Season:** 2024 (Week 7+)
- **Samples:** 374 rushing performances
- **Purpose:** Out-of-sample calibration validation

### Target Transformation

Rushing yards have a heavily right-skewed distribution. We apply log-transformation for training:

```python
y_train = np.log1p(stat_yards)  # log(1 + yards)
```

**Rationale:**
- Stabilizes variance (homoscedasticity)
- Makes distribution more Gaussian
- Prevents negative predictions

**Inverse transform for predictions:**
```python
yards_pred = np.expm1(y_pred_log)  # exp(y) - 1
```

### Feature Standardization

Features are standardized (zero mean, unit variance) before training:

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Scaler parameters (fitted on training data):
means = [30.92, 151.78, 252.69, 8.54]
stds =  [51.54, 170.76, 233.23, 6.02]
```

**Critical:** The same scaler must be used at prediction time (`transform()`, not `fit_transform()`).

### MCMC Sampling

We use PyMC's NUTS sampler for Bayesian inference:

```python
pm.sample(
    draws=2000,          # Posterior samples per chain
    tune=1000,           # Burn-in samples per chain
    chains=4,            # Independent Markov chains
    cores=1,             # Sequential (avoid race conditions)
    target_accept=0.95,  # High acceptance rate (careful exploration)
    init='adapt_diag'    # Adaptive diagonal initialization
)
```

**Total posterior samples:** 2000 × 4 = 8,000

**Training time:** 154 minutes (2 hours 34 minutes) on M1 Pro

**Convergence:**
- **Divergences:** 5 / 8,000 (0.06%) ✓ Excellent
- **R̂:** < 1.01 for all parameters ✓ Converged
- **ESS:** > 1000 for key parameters ✓ Good mixing

### Model Size

- **Saved model:** 1.3 GB (includes full posterior trace)
- **Posterior samples:** 8,000 × ~500 parameters = ~4M values
- **Inference:** Uses all 8,000 samples for uncertainty quantification

---

## Feature Engineering

### Input Features (4)

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `carries` | Integer | 5-30 | Number of rushing attempts in game |
| `avg_rushing_l3` | Float | 0-200 | Rolling 3-game average rushing yards |
| `season_avg` | Float | 0-300 | Season-to-date rushing average |
| `week` | Integer | 1-18 | Week number (captures seasonality) |

### Feature Importance

From model analysis (posterior gate weights):

1. **carries** (35%): Most important - workload predicts volume
2. **season_avg** (30%): Recent season form
3. **avg_rushing_l3** (25%): Short-term momentum
4. **week** (10%): Slight seasonality effect

### Missing Value Imputation

```python
# If player has no L3 average (first 3 games)
avg_rushing_l3 = df.groupby('season')['stat_yards'].transform('median')

# If player has no season average (first game)
season_avg = df.groupby('season')['stat_yards'].transform('median')
```

**Rationale:** Use league-wide median for season as neutral baseline.

### Feature Engineering Rationale

**Why these features?**
- **carries**: Direct workload indicator (more carries = more yards opportunity)
- **avg_rushing_l3**: Recent form matters (hot/cold streaks)
- **season_avg**: Season-long quality indicator
- **week**: Some seasonality (weather, playoffs)

**What we DON'T include:**
- Opponent strength (incomplete data)
- Weather (tested in Phase 1, no significant effect)
- Game location (home/away - marginal effect)
- Teammate effects (too sparse)

**Simplicity is key:** More features ≠ better predictions. The 4 features above capture 95% of predictive signal.

---

## Uncertainty Quantification

### Bayesian Inference

Unlike point estimate models (XGBoost, Neural Nets), Bayesian models provide **full posterior distributions**:

```
Point estimate model:  y_pred = f(x)
Bayesian model:        y_pred ~ P(y | x, Data)
```

**Advantage:** We get **uncertainty intervals** for free, calibrated from data.

### Prediction Process

For each test sample, we generate predictions from all 8,000 posterior samples:

```python
for i in range(8000):  # Each posterior sample
    # 1. Compute gating weights
    gate_logits = X @ W_gate[i]
    gates = softmax(gate_logits)

    # 2. Compute expert outputs
    for k in range(3):
        h = tanh(X @ W_h[k][i] + b_h[k][i])
        output[k] = h @ W_out[k][i] + b_out[k][i]

    # 3. Weighted combination
    μ = Σ(gates[:, k] × output[k])
    σ = Σ(gates[:, k] × σ_k[i])

    # 4. Sample from likelihood
    y_pred[i] = μ + N(0, σ)
```

**Result:** 8,000 predictions per sample → full posterior distribution

### Uncertainty Estimates

From posterior samples, we extract:

```python
predictions = {
    'mean': y_pred_samples.mean(axis=0),           # Point estimate
    'std': y_pred_samples.std(axis=0),             # Uncertainty (σ)
    'lower_90': np.percentile(y_pred_samples, 5),  # 90% CI lower
    'upper_90': np.percentile(y_pred_samples, 95), # 90% CI upper
}
```

**Interpretation:**
- **mean**: Best point estimate (expected value)
- **std**: Average prediction error (1 standard deviation)
- **90% CI**: 90% chance true value falls in this range

### Heterogeneous Uncertainty Example

```python
# Bench player (low carries)
Player A: 12 carries, 45 yards avg
  Prediction: 48 ± 21 yards (90% CI: 22-87)
  Expert usage: [75%, 20%, 5%]  # Mostly Expert 0 (low variance)

# Starter (medium carries)
Player B: 18 carries, 75 yards avg
  Prediction: 78 ± 29 yards (90% CI: 38-142)
  Expert usage: [15%, 70%, 15%]  # Mostly Expert 1 (medium variance)

# Star (high carries)
Player C: 25 carries, 120 yards avg
  Prediction: 118 ± 36 yards (90% CI: 65-195)
  Expert usage: [5%, 25%, 70%]  # Mostly Expert 2 (high variance)
```

**Key insight:** Model learns that high-volume backs have wider uncertainty (game script dependent).

---

## Calibration Metrics

### What is Calibration?

A model is **well-calibrated** if its stated confidence matches empirical frequency:

> "If I give 90% confidence intervals, 90% of true values should fall inside."

### Evaluation on Test Set (Week 7+, 2024)

| Metric | Target | Phase 2.2 | Phase 1 | Improvement |
|--------|--------|-----------|---------|-------------|
| **90% CI Coverage** | 90.0% | **92.2%** | 26.0% | +66.2 pp |
| **±1σ Coverage** | 68.0% | **78.3%** | ~35% | +43.3 pp |
| **MAE** | Lower | **18.5 yards** | 18.7 yards | -0.2 yards |
| **RMSE** | Lower | **24.8 yards** | 25.3 yards | -0.5 yards |

### 90% CI Coverage (Primary Metric)

```python
in_interval = (y_true >= lower_90) & (y_true <= upper_90)
coverage_90 = in_interval.mean() * 100
```

**Result:** 92.2% (target: 90%)

**Interpretation:**
- **Under-calibrated:** Coverage < 75% → intervals too narrow (overconfident)
- **Well-calibrated:** Coverage 85-95% → intervals match reality ✓
- **Over-calibrated:** Coverage > 95% → intervals too wide (under-confident)

### ±1σ Coverage (Secondary Metric)

```python
in_interval = (y_true >= mean - std) & (y_true <= mean + std)
coverage_68 = in_interval.mean() * 100
```

**Result:** 78.3% (target: 68%)

**Interpretation:** Slightly over-calibrated at 1σ level, but excellent for risk management (conservative).

### Point Estimate Accuracy

Despite focusing on calibration, we maintain strong point estimates:

- **MAE:** 18.5 yards (equivalent to Phase 1)
- **RMSE:** 24.8 yards (equivalent to Phase 1)

**Takeaway:** We improved calibration without sacrificing accuracy!

### Calibration Plot

```
Predicted Probability vs Observed Frequency
┌─────────────────────────────────────────┐
│ 100%│                               /   │
│     │                           /       │
│  90%│                       /••         │ ← Phase 2.2 (well-calibrated)
│     │                   /               │
│  80%│               /                   │
│     │           /                       │
│  70%│       /                           │
│     │   /                               │
│  60%│/••←Phase 1 (under-calibrated)     │
│     │                                   │
│     └───────────────────────────────────┘
│      60%  70%  80%  90% 100%
│      Predicted Confidence
```

**Ideal:** Dots lie on diagonal (predicted = observed)
**Phase 1:** Below diagonal (overconfident)
**Phase 2.2:** On diagonal (well-calibrated) ✓

---

## Production Implementation

### Pipeline Architecture

```
┌──────────────────────────────────────────────────────────┐
│                 Production Pipeline                       │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  1. Database Query (PostgreSQL)                           │
│     └─► Fetch rushing features for upcoming week         │
│                                                           │
│  2. Feature Engineering                                   │
│     └─► Calculate L3 avg, season avg                     │
│     └─► Standardize using training scaler                │
│                                                           │
│  3. BNN Prediction (Phase 2.2 MoE)                        │
│     └─► Load model (1.3 GB)                               │
│     └─► Generate 8,000 posterior samples                  │
│     └─► Extract mean, std, 90% CI                        │
│                                                           │
│  4. Prop Lines Integration                                │
│     └─► Fetch rushing yards O/U lines                    │
│     └─► Match players to lines                           │
│                                                           │
│  5. Bet Selection (Kelly Criterion)                       │
│     └─► Calculate P(over) and P(under)                   │
│     └─► Compare to implied probabilities                 │
│     └─► Filter by edge (≥3%) and confidence (≥80%)       │
│     └─► Size bets using 15% Kelly                        │
│                                                           │
│  6. Output Generation                                     │
│     └─► Save predictions CSV                             │
│     └─► Save bet recommendations CSV                     │
│     └─► Generate human-readable summary                  │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### Kelly Criterion Bet Sizing

For each bet opportunity, calculate optimal bet size:

```python
def kelly_criterion(p, odds_american):
    """
    Kelly formula: f* = (bp - q) / b

    where:
      p = win probability (model)
      q = 1 - p (loss probability)
      b = decimal odds - 1
    """
    # Convert American odds to decimal
    if odds_american > 0:
        b = odds_american / 100.0
    else:
        b = 100.0 / abs(odds_american)

    # Full Kelly
    kelly_full = (b * p - (1-p)) / b

    # Fractional Kelly (conservative)
    kelly_frac = kelly_full * 0.15  # 15% Kelly

    # Cap at 3% of bankroll
    return min(kelly_frac, 0.03)
```

**Example:**
```python
# Bet: OVER 75.5 yards @ -110
p_over = 0.58  # Model probability
implied_prob = 0.5238  # Market probability (from -110)
edge = 0.58 - 0.5238 = 0.0562  # 5.62% edge

# Kelly sizing
b = 100 / 110 = 0.909
kelly_full = (0.909 * 0.58 - 0.42) / 0.909 = 0.116  # 11.6%
kelly_15 = 0.116 * 0.15 = 0.0174  # 1.74% of bankroll

# For $10,000 bankroll
bet_amount = $10,000 * 0.0174 = $174
```

### Confidence Filtering

We require **statistical confidence** before betting:

```python
def confidence_check(pred_lower_90, pred_upper_90, line, bet_side):
    """
    Check if 90% CI supports the bet direction
    """
    if bet_side == 'over':
        # For OVER: lower bound should be close to line
        # (within 1 std below line is acceptable)
        confidence = (pred_lower_90 - line) / pred_std
        return confidence > -1.0

    elif bet_side == 'under':
        # For UNDER: upper bound should be close to line
        confidence = (line - pred_upper_90) / pred_std
        return confidence > -1.0
```

**Interpretation:**
- **High confidence:** 90% CI doesn't overlap line significantly
- **Low confidence:** 90% CI spans line widely (uncertain)

**Example:**
```python
# OVER 75.5 yards
Prediction: 82 ± 28 yards (90% CI: 38-142)
Confidence: (38 - 75.5) / 28 = -1.3σ  # REJECT (too uncertain)

# OVER 55.5 yards
Prediction: 82 ± 28 yards (90% CI: 38-142)
Confidence: (38 - 55.5) / 28 = -0.6σ  # ACCEPT (confident)
```

### Output Format

```csv
player_name,team,rushing_line,bet_side,prediction_mean,prediction_std,
prediction_lower_90,prediction_upper_90,prob_win,edge,confidence,
odds,bet_amount,expected_value

Saquon Barkley,PHI,75.5,over,82.3,28.1,38.2,142.1,0.581,0.057,0.82,
-110,174.00,9.93
```

---

## Performance Analysis

### Computational Performance

| Operation | Time | Memory |
|-----------|------|--------|
| **Model Loading** | 2-3 seconds | 1.3 GB |
| **Single Prediction** | 1-2 seconds | +200 MB |
| **Batch (30 players)** | 30-60 seconds | +500 MB |
| **Full Pipeline** | ~2 minutes | 2 GB peak |

### Prediction Latency Breakdown

```
Total: 1.8 seconds per prediction
├─ Posterior sampling (1.5s) ← Bottleneck
│  └─ 8,000 × (forward pass + softmax + sampling)
├─ Transform back to yards (0.2s)
└─ Percentile calculation (0.1s)
```

**Optimization opportunities:**
- Reduce posterior samples (8000 → 2000) for faster inference
- GPU acceleration (currently CPU-only)
- Batch processing (30 players in parallel)

### Prediction Quality by Carry Volume

| Carry Range | Count | MAE | RMSE | 90% Coverage |
|-------------|-------|-----|------|--------------|
| 5-10 | 89 | 12.3 | 16.8 | 94.4% |
| 11-15 | 142 | 15.8 | 21.2 | 92.3% |
| 16-20 | 98 | 19.7 | 26.4 | 90.8% |
| 21-25 | 32 | 24.1 | 31.9 | 90.6% |
| 26+ | 13 | 29.8 | 38.2 | 92.3% |

**Interpretation:**
- MAE scales with carries (expected)
- Calibration holds across all carry ranges ✓
- High-volume backs have wider errors (but well-calibrated)

### Expert Usage Analysis

| Expert | σ Prior | Usage % | Typical Players |
|--------|---------|---------|-----------------|
| 0 (Low) | HalfNormal(10) | 32% | Backup RBs, 8-12 carries |
| 1 (Med) | HalfNormal(15) | 32% | Starters, 13-18 carries |
| 2 (High) | HalfNormal(20) | 36% | Bellcows, 19+ carries |

**Interpretation:**
- Balanced expert usage (no expert dominates) ✓
- Model learns carry volume correlates with variance
- Gating network successfully differentiates regimes

---

## Limitations & Future Work

### Current Limitations

1. **Conservative for Elite Performances**
   - **Issue:** Model under-predicts 150+ yard games (stars going off)
   - **Example:** Saquon 176 yards → predicted 67 yards
   - **Cause:** Prioritizes calibration over capturing extreme upside
   - **Impact:** May miss some +EV OVER bets on stars in favorable matchups

2. **No Opponent Adjustment**
   - **Missing:** Defensive strength, run defense DVOA
   - **Impact:** Treats all matchups equally (overestimates tough matchups)
   - **Challenge:** Historical opponent data is noisy/incomplete

3. **Limited Context**
   - **Missing:** Game script (leading/trailing), weather, injuries
   - **Impact:** Doesn't adapt to game-specific factors
   - **Trade-off:** Simplicity vs comprehensiveness

4. **Static Scaler**
   - **Issue:** Scaler fitted on 2020-2024 data (may drift over time)
   - **Impact:** If NFL rushing trends change significantly, recalibration needed
   - **Solution:** Retrain annually or use rolling window

### Future Research Directions

#### Phase 3: Multi-Output Models
- **Idea:** Predict rushing yards + receptions + TDs jointly
- **Advantage:** Capture correlations (high rushers get more TDs)
- **Challenge:** Multi-dimensional calibration

#### Phase 4: Sequential Models
- **Idea:** LSTM/Transformer for play-by-play prediction
- **Advantage:** Model within-game dynamics (momentum, fatigue)
- **Challenge:** Data availability (play-by-play features)

#### Phase 5: Hierarchical Models
- **Idea:** Player-specific parameters (random effects)
- **Advantage:** Better capture player styles (breakaway threat vs grinder)
- **Challenge:** Sparse data per player (need strong priors)

#### Phase 6: Ensemble Methods
- **Idea:** Combine Phase 2.2 MoE with XGBoost and GLMs
- **Advantage:** Robust to model mis-specification
- **Challenge:** How to ensemble uncertainties?

---

## References

### Key Papers

1. **Bayesian Neural Networks:**
   - Blundell et al. (2015). "Weight Uncertainty in Neural Networks"
   - Gal & Ghahramani (2016). "Dropout as a Bayesian Approximation"

2. **Mixture-of-Experts:**
   - Jacobs et al. (1991). "Adaptive Mixtures of Local Experts"
   - Shazeer et al. (2017). "Outrageously Large Neural Networks" (Sparse MoE)

3. **Calibration:**
   - Gneiting & Raftery (2007). "Strictly Proper Scoring Rules"
   - Kuleshov et al. (2018). "Accurate Uncertainties for Deep Learning"

4. **Sports Analytics:**
   - Lopez & Matthews (2015). "Building Win Probability Models"
   - Yurko et al. (2019). "Going Deep: Models for Continuous NFL Outcomes"

### Software

- **PyMC:** Probabilistic programming (NUTS sampler)
- **Arviz:** Bayesian diagnostics and visualization
- **Scikit-learn:** Feature preprocessing
- **PostgreSQL:** Data warehouse

---

## Appendix A: Model Code

See `/py/models/bnn_mixture_experts_v2.py` for full implementation.

**Key methods:**
- `build_model(X_train, y_train)`: Construct PyMC model
- `train(X_train, y_train)`: MCMC sampling
- `predict(X_test)`: Generate posterior predictive samples
- `evaluate_calibration(X_test, y_test)`: Compute calibration metrics
- `save(filepath)`: Serialize model + scaler
- `load(filepath)`: Restore model + scaler

## Appendix B: Production Code

See `/py/production/bnn_moe_production_pipeline.py` for full implementation.

**Key methods:**
- `predict_with_uncertainty(players_df)`: Generate predictions
- `select_bets(predictions_df, prop_lines_df)`: Kelly criterion bet sizing
- `run_pipeline(season, week)`: End-to-end workflow

---

**Document Version:** 1.0
**Last Updated:** October 17, 2025
**Status:** Production-Ready ✓
