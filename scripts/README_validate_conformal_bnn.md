# BNN Conformal Prediction Validation Script

**Script:** `scripts/validate_conformal_bnn.py`
**Created:** November 9, 2025
**Purpose:** Comprehensive validation of BNN Conformal Prediction wrapper with finite-sample coverage guarantees

---

## Overview

This script validates the BNN Conformal Prediction wrapper (`nfl_analytics/models/bnn_conformal.py`) by:

1. **Loading NFL rushing yards data** (2018-2024) from the database
2. **Splitting data temporally** into train/calib/test (60/20/20)
3. **Loading existing BNN model** (or falling back to Random Forest)
4. **Calibrating conformal prediction** on the calibration set
5. **Validating 90% coverage** on the test set
6. **Comparing to temperature scaling** baseline (89.5%)
7. **Generating performance report** with metrics and diagnostic plots

---

## Requirements

### Dependencies

All dependencies are declared in `pyproject.toml`:

```bash
# Core dependencies
numpy>=2.0.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.8.0
psycopg[binary]>=3.1.0

# Conformal prediction
mapie>=1.1.0

# Optional (for BNN support)
torch>=2.0.0
```

### Database

Requires PostgreSQL database running at:
- **Host:** localhost
- **Port:** 5544
- **Database:** devdb01
- **User:** dro
- **Password:** sicillionbillions

If database is not available, script will attempt to use cached data.

---

## Usage

### Basic Usage (Random Forest fallback)

```bash
# Activate virtual environment
source .venv/bin/activate

# Run validation with Random Forest base model
python scripts/validate_conformal_bnn.py --use-rf-fallback
```

### With BNN Model (if available)

```bash
# Activate virtual environment
source .venv/bin/activate

# Run validation with BNN base model (attempts to load from checkpoints)
python scripts/validate_conformal_bnn.py
```

### Custom Coverage Level

```bash
# 95% coverage instead of 90%
python scripts/validate_conformal_bnn.py --use-rf-fallback --alpha 0.05
```

### Help

```bash
python scripts/validate_conformal_bnn.py --help
```

---

## Data Split Strategy

**Temporal split** to avoid data leakage:

| Split | Seasons | Weeks | Percentage | Purpose |
|-------|---------|-------|------------|---------|
| **Train** | 2018-2021 | All | 60% | Train base model (RF only) |
| **Calibration** | 2022, 2023 (weeks 1-9) | 2022: All, 2023: 1-9 | 20% | Compute nonconformity scores |
| **Test** | 2023 (weeks 10+), 2024 | 2023: 10+, 2024: All | 20% | Validate coverage guarantee |

**No overlap** between splits (validated by assertions).

---

## Features

Uses same 6 features as baseline models:

1. **carries** - Number of rushing attempts
2. **avg_rushing_l3** - 3-game rolling average
3. **season_avg** - Season-to-date average
4. **week** - Game week
5. **spread_close** - Vegas closing spread
6. **total_close** - Vegas closing total

---

## Outputs

### 1. Performance Report (JSON)

**File:** `results/conformal_bnn_validation.json`

```json
{
  "experiment": "BNN Conformal Prediction Validation",
  "timestamp": "2025-11-09T15:30:00",
  "config": {
    "base_model": "Random Forest",
    "conformal_method": "split conformal (Jackknife+)",
    "target_coverage": 0.90,
    "n_features": 6
  },
  "results": {
    "point_accuracy": {
      "mae": 45.23,
      "rmse": 58.67,
      "mape": 42.1
    },
    "calibration": {
      "coverage_90": 0.895,
      "target_coverage": 0.90,
      "coverage_gap": -0.005,
      "within_2pct": true
    },
    "sharpness": {
      "avg_width": 148.5,
      "median_width": 142.3,
      "width_std": 32.1,
      "width_cv": 0.22
    }
  },
  "comparison": {
    "baseline": {
      "method": "Temperature Scaling",
      "coverage_90": 0.895,
      "temperature": 0.75
    },
    "conformal": {
      "coverage_90": 0.895,
      "avg_width": 148.5
    },
    "comparison": {
      "coverage_diff": 0.0,
      "conformal_better": false
    }
  }
}
```

### 2. Comparison Report (Markdown)

**File:** `results/conformal_vs_temperature.md`

Human-readable comparison including:
- Test set performance metrics
- Calibration results
- Comparison table
- Diagnostic plot references
- Conclusion

### 3. Diagnostic Plots (PNG)

**Directory:** `results/figures/`

#### a) Calibration Curve

**File:** `conformal_calibration_curve.png`

Shows predicted vs empirical coverage at different confidence levels (50%-99%).

- **X-axis:** Target coverage
- **Y-axis:** Empirical coverage
- **Perfect calibration:** Points lie on diagonal line

#### b) Interval Width Distribution

**File:** `conformal_interval_width_distribution.png`

Histogram of prediction interval widths.

- Shows distribution of interval widths across all predictions
- Includes mean and median lines
- Indicates adaptive intervals (varying width)

#### c) Coverage by Prediction Value

**File:** `conformal_coverage_by_prediction.png`

Coverage stratified by predicted rushing yards (0-200 yards in 20-yard bins).

- **X-axis:** Predicted rushing yards
- **Y-axis:** Empirical coverage
- **Target:** 90% coverage (red dashed line)
- Shows if coverage is uniform across prediction range

---

## Validation Criteria

### ✓ PASS Criteria

1. **Coverage within ±2% of target:** `|coverage - 0.90| ≤ 0.02`
2. **No systematic bias:** Coverage similar across prediction range
3. **Finite interval widths:** Average width < 200 yards
4. **Better than temperature scaling:** Coverage ≥ 89.5%

### ↗ ACCEPTABLE Criteria

1. **Coverage within ±5% of target:** `|coverage - 0.90| ≤ 0.05`
2. **Small sample effects:** May occur with limited calibration data

### ⚠ INVESTIGATE Criteria

1. **Coverage deviates >5% from target**
2. **Infinite interval widths:** Suggests calibration failure
3. **Non-uniform coverage:** Coverage varies significantly by prediction range

---

## Troubleshooting

### 1. Database Connection Failed

**Error:**
```
psycopg2.OperationalError: could not connect to server
```

**Solution:**
- Ensure PostgreSQL is running: `docker compose up -d pg`
- Check connection details in script
- Script will automatically use cached data if available

### 2. MAPIE Not Installed

**Error:**
```
ERROR:root:MAPIE not installed
```

**Solution:**
```bash
# Install MAPIE
uv pip install mapie

# Or with regular pip
pip install mapie
```

### 3. PyTorch Not Available

**Warning:**
```
WARNING:root:PyTorch not available - will use Random Forest fallback
```

**Solution:**
- Use `--use-rf-fallback` flag (no action needed)
- Or install PyTorch: `uv pip install torch`

### 4. BNN Model Not Found

**Warning:**
```
WARNING:root:Could not load BNN model from any path
```

**Solution:**
- Use `--use-rf-fallback` flag
- Or ensure BNN model exists at:
  - `models/bayesian/bnn_multioutput_v1_final.pt`
  - `models/bayesian/bnn_multioutput_passing_v1_canonical.pt`

### 5. Insufficient Calibration Data

**Warning:**
```
WARNING:root:Coverage below target (may need more calibration data)
```

**Solution:**
- Increase calibration set size by adjusting split strategy
- Currently: 20% of data (~1,000+ samples)
- For tighter guarantees: 30% calibration recommended

---

## Technical Details

### Conformal Prediction Method

**Algorithm:** Split Conformal Prediction (Vovk et al., 2005)

**Steps:**

1. **Train base model** (BNN or Random Forest) on training set
2. **Compute nonconformity scores** on calibration set:
   ```
   scores = |y_calib - f(X_calib)|
   ```
3. **Calculate conformal quantile** at level `1-alpha`:
   ```
   q = quantile(scores, 1-alpha)
   ```
4. **Construct prediction intervals** on test set:
   ```
   [f(X_test) - q, f(X_test) + q]
   ```

**Guarantee:** For any `alpha`, the marginal coverage is at least `1-alpha`:
```
P(y_test ∈ [lower, upper]) ≥ 1 - alpha
```

### MAPIE Implementation

Uses **MAPIE 1.1.0** `SplitConformalRegressor`:

```python
from mapie.regression import SplitConformalRegressor

# Create conformal predictor
cp = SplitConformalRegressor(
    estimator=base_model,
    confidence_level=0.90,  # 90% coverage
    conformity_score='absolute',  # Absolute residual
    prefit=True,  # Base model already trained
    n_jobs=-1
)

# Calibrate (compute nonconformity scores)
cp.conformalize(X_calib, y_calib)

# Predict with intervals
y_pred, y_intervals = cp.predict_interval(X_test)
# intervals shape: (n_samples, 2, 1)
# intervals[:, 0, 0] = lower bounds
# intervals[:, 1, 0] = upper bounds
```

### Comparison to Temperature Scaling

**Temperature Scaling (Baseline):**
- Method: Scale BNN variance by temperature `T`
- Implementation: `sigma_scaled = T * sigma_bnn`
- Result: 89.5% coverage on test set (BNN v3.1 Rushing)

**Conformal Prediction (This Script):**
- Method: Distribution-free finite-sample guarantee
- Implementation: Split conformal with absolute conformity score
- Expected: ≥90% coverage (theoretical guarantee)

**Key Difference:**
- Temperature scaling requires **distributional assumptions** (Gaussian)
- Conformal prediction is **distribution-free** (no assumptions)

---

## Expected Runtime

| Step | Time | Notes |
|------|------|-------|
| Database loading | 5-10s | Depends on network |
| Data preparation | <1s | Fast preprocessing |
| RF training | 10-20s | 100 trees, 10k+ samples |
| Conformal calibration | <1s | Compute quantile |
| Test set prediction | <1s | Vectorized operations |
| Plot generation | 5-10s | 3 plots with calibration curve |
| **Total** | **~30-45s** | Full validation pipeline |

---

## Next Steps

### 1. Integrate with Production Ensemble

Once validated, integrate conformal prediction into production ensemble:

```python
from nfl_analytics.models.bnn_conformal import BNNConformalPredictor

# Load BNN
bnn = torch.load('models/bayesian/bnn_multioutput_v1_final.pt')

# Create conformal predictor
conformal = BNNConformalPredictor(
    bnn_model=bnn,
    alpha=0.10,  # 90% coverage
    device='cuda'
)

# Calibrate on validation set
conformal.calibrate(X_calib, y_calib)

# Use in ensemble
predictions = conformal.predict(X_test)
# Returns: {'mean': ..., 'q05': ..., 'q95': ..., 'coverage_guarantee': 0.90}
```

### 2. Extend to Other Stat Types

Apply conformal prediction to:
- **Passing yards** (QB props)
- **Receiving yards** (WR/TE props)
- **Touchdowns** (classification with conformal sets)

### 3. Adaptive Conformal Prediction

Implement **Adaptive Conformal Inference** (Gibbs & Candès, 2021) for:
- Time-varying coverage under distribution shift
- Online updating of nonconformity scores
- Tracking miscoverage and adapting `alpha`

---

## References

1. **Conformal Prediction Theory:**
   - Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
   - Angelopoulos, A. N., & Bates, S. (2022). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv:2107.07511.

2. **Conformalized Quantile Regression:**
   - Romano, Y., Patterson, E., & Candès, E. (2019). *Conformalized Quantile Regression*. NeurIPS 2019.

3. **MAPIE Library:**
   - MAPIE Documentation: https://mapie.readthedocs.io/
   - GitHub: https://github.com/scikit-learn-contrib/MAPIE

4. **Temperature Scaling:**
   - Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks*. ICML 2017.

---

## Maintenance

**Last Updated:** November 9, 2025
**Author:** Claude Code
**Version:** 1.0
**Script Size:** 35 KB (1,000+ lines)

**Change Log:**
- **v1.0 (Nov 9, 2025):** Initial implementation
  - Full validation pipeline
  - Random Forest fallback
  - 3 diagnostic plots
  - Comparison to temperature scaling

**Known Issues:**
- BNN loading requires specific checkpoint format (direct model, not state dict)
- MAPIE 1.1.0 API differs from older versions (use `SplitConformalRegressor`, not `MapieRegressor`)

**Future Enhancements:**
- [ ] Add cross-validation for more robust calibration
- [ ] Implement Conformalized Quantile Regression (CQR)
- [ ] Support adaptive conformal prediction (online updating)
- [ ] Add stratified coverage analysis (by position, team, weather)

---

**For questions or issues, see:**
- **Project Documentation:** `CLAUDE.md`
- **BNN Conformal Wrapper:** `nfl_analytics/models/bnn_conformal.py`
- **Baseline Conformal:** `nfl_analytics/models/conformal_prediction_baseline.py`
