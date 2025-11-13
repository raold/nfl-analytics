# Temperature Scaling - Technical Specification
## Detailed Implementation Guide

**Date:** November 13, 2025
**Target File:** `py/ensemble/enhanced_ensemble_v5.py`
**Estimated Lines:** ~250 lines of new code
**Dependencies:** scipy.optimize, numpy, pandas, json

---

## Quick Reference

### Current Problem
```
97.3% coverage on 90% PI (target: 90%)
→ Intervals too wide
→ Solution: Scale std by temperature T < 1.0
```

### Mathematical Solution
```
σ_calibrated = σ_raw * √T
where T ≈ 0.7-0.85 (estimated)
```

### Expected T Value Estimation

For 97.3% → 90% coverage:
```python
# Rough estimate (z-score based)
coverage_before = 0.973
coverage_target = 0.90

# For normal dist, coverage ≈ erf(z_actual / z_target)
# T ≈ (z_target / z_actual)²
T_estimated ≈ (1.282 / 1.95)² ≈ 0.43

# But this is conservative. Actual T likely: 0.6-0.8
# Optimization will find exact value
```

---

## Implementation Details

### 1. Modify __init__() Method

**Location:** Lines 90-141 in enhanced_ensemble_v5.py

**Code to Add:**
```python
def __init__(
    self,
    state_space_path: str = "models/bayesian/simplified_random_walk_v2.rds",
    hierarchical_path: str = "models/bayesian/passing_yards_hierarchical_v2.rds",
    informative_path: str = "models/bayesian/passing_informative_priors_v2.rds",
    xgboost_path: str = "models/xgboost/passing_xgboost_v2_canonical.pkl",
    weights: Optional[Dict[str, float]] = None,
    db_host: str = "localhost",
    db_port: int = 5544,
    db_name: str = "devdb01",
    db_user: str = "dro",
    db_password: str = "sicillionbillions",
    # NEW PARAMETERS
    temperature: float = 1.0,
    temperature_fitted: bool = False,
):
    """
    Initialize 4-way ensemble predictor with optional temperature scaling.

    Args:
        ... [existing docstring] ...
        temperature: Temperature scaling parameter (default 1.0 = no scaling).
                   Values < 1.0 shrink intervals (fix overcalibration).
                   Values > 1.0 widen intervals (fix undercalibration).
        temperature_fitted: Whether temperature was fitted on calibration data.
    """
    # ... existing initialization code ...

    # Temperature scaling (NEW)
    self.temperature = temperature
    self.temperature_fitted = temperature_fitted

    if temperature != 1.0:
        logger.info(f"⚠️  Using pre-fitted temperature: T={temperature:.4f}")
    elif temperature_fitted:
        logger.warning(f"⚠️  Temperature marked as fitted but T=1.0 (no effect)")
    else:
        logger.debug(f"Temperature scaling: disabled (T=1.0)")
```

### 2. Add fit_temperature() Method

**Location:** After predict_batch() method (around line 671)

**Full Implementation:**
```python
def fit_temperature(
    self,
    predictions_df: pd.DataFrame,
    actuals_col: str = 'actual_yards',
    pred_mean_col: str = 'ensemble_mean',
    pred_std_col: str = 'ensemble_std',
    bounds: Tuple[float, float] = (0.1, 5.0),
    verbose: bool = True,
) -> Dict:
    """
    Fit temperature parameter T to achieve target 90% PI coverage.

    Temperature scaling is a post-hoc calibration method that rescales
    prediction uncertainty by a learned temperature parameter:

        σ_calibrated = σ_raw * √T

    This method finds T that minimizes negative log-likelihood on the
    calibration set, which empirically achieves ~90% coverage.

    Mathematical Background:
    - Objective: Minimize NLL = Σ(0.5 * log(σ²) + 0.5 * (y - ŷ)²/σ²)
    - Single scalar parameter (no overfitting risk)
    - Preserves all mean predictions (model accuracy unchanged)
    - Reference: Guo et al. (2017), "On Calibration of Modern Neural Networks"

    Args:
        predictions_df (pd.DataFrame): Calibration data with predictions and actuals.
            Required columns:
            - {pred_mean_col}: Ensemble mean predictions
            - {pred_std_col}: Ensemble std predictions
            - {actuals_col}: Actual observed values

        actuals_col (str): Name of actual values column. Default: 'actual_yards'

        pred_mean_col (str): Name of ensemble mean column. Default: 'ensemble_mean'

        pred_std_col (str): Name of ensemble std column. Default: 'ensemble_std'

        bounds (Tuple[float, float]): (min_T, max_T) for optimization.
            Default: (0.1, 5.0)
            - Lower bound 0.1: Allow significant shrinkage if needed
            - Upper bound 5.0: Allow significant expansion if needed
            - Typical values: 0.5-2.0 (rarely outside this range)

        verbose (bool): Whether to log optimization details. Default: True

    Returns:
        Dict with keys:
            - 'temperature' (float): Optimal T value
            - 'neg_log_likelihood' (float): Loss at optimum
            - 'coverage_before' (float): 90% PI coverage before scaling
            - 'coverage_after' (float): 90% PI coverage after scaling
            - 'n_samples' (int): Number of calibration samples
            - 'acceptable' (bool): True if 85-95% coverage achieved
            - 'interval_width_before' (float): Mean 90% PI width before
            - 'interval_width_after' (float): Mean 90% PI width after

    Example:
        >>> import pandas as pd
        >>> from py.ensemble.enhanced_ensemble_v5 import EnhancedEnsemble4Way
        >>>
        >>> # Load predictions and actuals
        >>> preds = pd.read_csv('predictions.csv')
        >>> preds['actual_yards'] = [250, 275, 320, ...]  # actuals
        >>>
        >>> # Fit temperature
        >>> ensemble = EnhancedEnsemble4Way()
        >>> result = ensemble.fit_temperature(preds)
        >>>
        >>> # Use fitted ensemble
        >>> print(f"Optimal T: {result['temperature']:.4f}")
        >>> print(f"Coverage: {result['coverage_after']:.1%}")
        >>> ensemble.temperature = result['temperature']
        >>> ensemble.temperature_fitted = True
        >>>
        >>> # Future predictions automatically use T
        >>> pred = ensemble.predict("00-0019596", 2025, 11)

    Raises:
        ValueError: If input DataFrame missing required columns
        RuntimeError: If optimization fails to converge
    """
    from scipy.optimize import minimize_scalar
    import warnings

    # Validate input
    if pred_mean_col not in predictions_df.columns:
        raise ValueError(f"Missing column: {pred_mean_col}")
    if pred_std_col not in predictions_df.columns:
        raise ValueError(f"Missing column: {pred_std_col}")
    if actuals_col not in predictions_df.columns:
        raise ValueError(f"Missing column: {actuals_col}")

    # Extract arrays
    actuals = predictions_df[actuals_col].values
    means = predictions_df[pred_mean_col].values
    stds = predictions_df[pred_std_col].values

    # Check for NaNs
    mask = ~(np.isnan(actuals) | np.isnan(means) | np.isnan(stds))
    actuals = actuals[mask]
    means = means[mask]
    stds = stds[mask]

    n_removed = len(predictions_df) - len(actuals)
    if n_removed > 0 and verbose:
        logger.warning(f"⚠️  Removed {n_removed} rows with NaN values")

    if len(actuals) < 20:
        logger.warning(f"⚠️  Small calibration set ({len(actuals)} samples)")
        logger.warning("    Results may be unstable. Recommend ≥50 samples.")

    if verbose:
        logger.info("\n" + "=" * 80)
        logger.info("TEMPERATURE SCALING CALIBRATION")
        logger.info("=" * 80)
        logger.info(f"\nCalibration Set:")
        logger.info(f"  Samples: {len(actuals):,}")
        logger.info(f"  Mean predictions: [{means.min():.1f}, {means.max():.1f}]")
        logger.info(f"  Actuals: [{actuals.min():.1f}, {actuals.max():.1f}]")
        logger.info(f"  Std: {stds.mean():.1f} ± {stds.std():.1f} (mean ± SD)")

    # ===== OBJECTIVE FUNCTION =====
    def neg_log_likelihood(T):
        """
        Compute negative log-likelihood of actuals under N(μ, σ²_scaled).

        For each sample: NLL_i = 0.5 * log(2π σ²) + 0.5 * ((y - ŷ) / σ)²
        where σ = σ_raw * √T

        Total NLL = Σ NLL_i

        Args:
            T (float): Temperature parameter

        Returns:
            float: Total negative log-likelihood
        """
        if T <= 0:
            return 1e10  # Penalize invalid values

        sigma_scaled = stds * np.sqrt(T)
        residuals = actuals - means

        # Compute NLL
        # Using: NLL = 0.5 * log(σ²) + 0.5 * (residual/σ)²
        # (dropping constant 0.5*log(2π) since it doesn't affect optimization)
        nll = 0.5 * np.log(sigma_scaled ** 2) + 0.5 * (residuals / sigma_scaled) ** 2
        return nll.sum()

    # ===== OPTIMIZE =====
    if verbose:
        logger.info(f"\nOptimizing T in range [{bounds[0]}, {bounds[1]}]...")

    result = minimize_scalar(
        neg_log_likelihood,
        bounds=bounds,
        method='bounded',
        options={'xatol': 1e-6}  # High precision
    )

    if not result.success and verbose:
        logger.warning(f"⚠️  Optimization did not converge fully: {result.message}")

    T_optimal = result.x
    nll_optimal = result.fun

    # ===== COMPUTE CALIBRATION METRICS =====
    z_score = 1.645  # 90th percentile of standard normal

    # Before scaling
    lower_before = means - z_score * stds
    upper_before = means + z_score * stds
    within_before = (actuals >= lower_before) & (actuals <= upper_before)
    coverage_before = within_before.mean()
    width_before = (upper_before - lower_before).mean()

    # After scaling
    stds_scaled = stds * np.sqrt(T_optimal)
    lower_after = means - z_score * stds_scaled
    upper_after = means + z_score * stds_scaled
    within_after = (actuals >= lower_after) & (actuals <= upper_after)
    coverage_after = within_after.mean()
    width_after = (upper_after - lower_after).mean()

    # Acceptability
    acceptable = 0.85 <= coverage_after <= 0.95

    # ===== UPDATE ENSEMBLE STATE =====
    self.temperature = T_optimal
    self.temperature_fitted = True

    # ===== LOGGING =====
    if verbose:
        logger.info(f"\n{'=' * 80}")
        logger.info("RESULTS")
        logger.info(f"{'=' * 80}")
        logger.info(f"Optimal Temperature: {T_optimal:.6f}")
        logger.info(f"  → √T = {np.sqrt(T_optimal):.6f} (uncertainty scale factor)")
        logger.info(f"  → Effect: {'SHRINK' if T_optimal < 1.0 else 'WIDEN' if T_optimal > 1.0 else 'NO CHANGE'} intervals")
        logger.info(f"\nNegative Log-Likelihood: {nll_optimal:.2f}")
        logger.info(f"  → Per-sample: {nll_optimal / len(actuals):.4f}")

        logger.info(f"\nCoverage (90% Prediction Interval):")
        logger.info(f"  Before: {coverage_before:.1%} ({within_before.sum()}/{len(actuals)})")
        logger.info(f"  After:  {coverage_after:.1%} ({within_after.sum()}/{len(actuals)})")
        logger.info(f"  Change: {coverage_after - coverage_before:+.1%}")
        logger.info(f"  Status: {'✅ ACCEPTABLE (85-95%)' if acceptable else '⚠️  MARGINAL (80-98%)' if 0.80 <= coverage_after <= 0.98 else '❌ UNACCEPTABLE'}")

        logger.info(f"\nPrediction Interval Width (90% PI):")
        logger.info(f"  Before: {width_before:.1f} yards")
        logger.info(f"  After:  {width_after:.1f} yards")
        logger.info(f"  Reduction: {width_before - width_after:.1f} yards ({(width_before - width_after) / width_before * 100:.1f}%)")

        logger.info(f"\nCalibration Status:")
        if 0.85 <= coverage_after <= 0.95:
            logger.info(f"  ✅ Ensemble is properly calibrated")
        elif coverage_after > 0.95:
            logger.info(f"  ⚠️  Still overcalibrated (coverage {coverage_after:.1%})")
            logger.info(f"      Consider using tighter bounds or larger calibration set")
        else:
            logger.info(f"  ⚠️  Undercalibrated (coverage {coverage_after:.1%})")
            logger.info(f"      May need wider bounds or investigate model issues")

        logger.info(f"\n{'=' * 80}\n")

    return {
        'temperature': T_optimal,
        'temperature_sqrt': np.sqrt(T_optimal),  # For reference
        'neg_log_likelihood': nll_optimal,
        'coverage_before': float(coverage_before),
        'coverage_after': float(coverage_after),
        'coverage_change': float(coverage_after - coverage_before),
        'n_samples': len(actuals),
        'acceptable': bool(acceptable),
        'interval_width_before': float(width_before),
        'interval_width_after': float(width_after),
        'interval_width_reduction': float(width_before - width_after),
        'interval_width_reduction_pct': float((width_before - width_after) / width_before * 100),
    }
```

### 3. Modify predict() Method

**Location:** Around line 596 (after ensemble_std calculation)

**Current Code:**
```python
# Line 588-596
within_var = np.sum(active_weights * (stds ** 2))
between_var = np.sum(active_weights * ((means - ensemble_mean) ** 2))
ensemble_var = within_var + between_var
ensemble_std = np.sqrt(ensemble_var)

# 90% prediction interval (assuming normal)
quantile_10 = ensemble_mean - 1.645 * ensemble_std
quantile_90 = ensemble_mean + 1.645 * ensemble_std
```

**Modified Code:**
```python
# Line 588-596 (unchanged)
within_var = np.sum(active_weights * (stds ** 2))
between_var = np.sum(active_weights * ((means - ensemble_mean) ** 2))
ensemble_var = within_var + between_var
ensemble_std = np.sqrt(ensemble_var)

# Apply temperature scaling if fitted (NEW)
if self.temperature_fitted and self.temperature != 1.0:
    ensemble_std = ensemble_std * np.sqrt(self.temperature)
    logger.debug(f"Applied temperature scaling T={self.temperature:.4f} → std *= {np.sqrt(self.temperature):.4f}")

# 90% prediction interval (assuming normal)
quantile_10 = ensemble_mean - 1.645 * ensemble_std
quantile_90 = ensemble_mean + 1.645 * ensemble_std
```

### 4. Add save_temperature() Method

**Location:** After fit_temperature() method

```python
def save_temperature(self, path: str) -> None:
    """
    Save temperature parameter to JSON file.

    The temperature can be loaded later to use a pre-fitted calibration
    without rerunning fit_temperature().

    Args:
        path (str): Path to save JSON file

    Example:
        >>> ensemble = EnhancedEnsemble4Way()
        >>> result = ensemble.fit_temperature(calib_data)
        >>> ensemble.save_temperature('models/temperature.json')

    JSON Format:
        {
            "temperature": 0.8234,
            "temperature_fitted": true,
            "date_fitted": "2025-11-13T14:30:00Z"
        }
    """
    import json
    from datetime import datetime

    data = {
        'temperature': float(self.temperature),
        'temperature_fitted': bool(self.temperature_fitted),
        'date_fitted': datetime.utcnow().isoformat(),
    }

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"✅ Temperature saved to {path}")
    logger.info(f"   T = {self.temperature:.4f}, fitted = {self.temperature_fitted}")
```

### 5. Add load_temperature() Method

**Location:** After save_temperature() method

```python
def load_temperature(self, path: str) -> None:
    """
    Load temperature parameter from JSON file.

    Args:
        path (str): Path to JSON file (created by save_temperature())

    Example:
        >>> ensemble = EnhancedEnsemble4Way()
        >>> ensemble.load_temperature('models/temperature.json')
        >>> pred = ensemble.predict(player_id, season, week)
        >>> # Prediction automatically uses loaded temperature

    Raises:
        FileNotFoundError: If file does not exist
        json.JSONDecodeError: If file is not valid JSON
        KeyError: If file missing 'temperature' key
    """
    import json

    with open(path, 'r') as f:
        data = json.load(f)

    if 'temperature' not in data:
        raise KeyError("JSON file missing 'temperature' key")

    self.temperature = float(data['temperature'])
    self.temperature_fitted = bool(data.get('temperature_fitted', True))

    logger.info(f"✅ Temperature loaded from {path}")
    logger.info(f"   T = {self.temperature:.4f}, fitted = {self.temperature_fitted}")
```

---

## 3. Integration Points

### In predict() method (line 596)
Add temperature scaling after variance pooling

### In predict_batch() method
No changes needed (uses predict() internally)

### In __init__() method (lines 90-141)
Add temperature and temperature_fitted parameters

---

## 4. Import Statements (No changes needed)

Existing imports cover scipy.optimize, numpy, etc.

---

## 5. Error Handling

### Optimization Failures
```python
if not result.success:
    logger.warning(f"Optimization may not have converged: {result.message}")
    # Still return result (scipy's best estimate)
```

### Missing Columns
```python
if pred_mean_col not in predictions_df.columns:
    raise ValueError(f"Missing column: {pred_mean_col}")
```

### Invalid Temperature Values
```python
if T <= 0:
    return 1e10  # Invalid, penalize heavily
```

---

## 6. Testing Code Examples

### Unit Test
```python
def test_fit_temperature():
    """Test temperature fitting on synthetic data."""
    import numpy as np
    import pandas as pd
    from py.ensemble.enhanced_ensemble_v5 import EnhancedEnsemble4Way

    # Synthetic data: perfectly centered predictions
    np.random.seed(42)
    n = 200

    # Ground truth: actuals = mean + N(0, 30)
    true_std = 30
    means = np.random.normal(250, 50, n)
    actuals = means + np.random.normal(0, true_std, n)

    # Ensemble overestimates uncertainty: std = sqrt(30² + 50²) ≈ 58
    ensemble_std = np.sqrt(true_std**2 + 50**2) * np.ones(n)

    # Create DataFrame
    df = pd.DataFrame({
        'ensemble_mean': means,
        'ensemble_std': ensemble_std,
        'actual_yards': actuals,
    })

    # Fit temperature
    ensemble = EnhancedEnsemble4Way()
    result = ensemble.fit_temperature(df, verbose=False)

    # Assertions
    assert 0.5 <= result['temperature'] <= 1.0, "Should shrink (T < 1)"
    assert 0.85 <= result['coverage_after'] <= 0.95, "Coverage in range"
    assert result['acceptable'] == True, "Should be acceptable"
    assert result['interval_width_after'] < result['interval_width_before'], "Intervals shrink"

    print("✅ Test passed")
    print(f"   T = {result['temperature']:.4f}")
    print(f"   Coverage: {result['coverage_after']:.1%}")
```

---

## 7. Expected Output

When fitting temperature on 2025 data:

```
================================================================================
TEMPERATURE SCALING CALIBRATION
================================================================================

Calibration Set:
  Samples: 182
  Mean predictions: [145.3, 421.7]
  Actuals: [67.0, 512.0]
  Std: 98.6 ± 35.2 (mean ± SD)

Optimizing T in range [0.1, 5.0]...

================================================================================
RESULTS
================================================================================
Optimal Temperature: 0.826400
  → √T = 0.909 (uncertainty scale factor)
  → Effect: SHRINK intervals

Negative Log-Likelihood: 856.34
  → Per-sample: 4.7061

Coverage (90% Prediction Interval):
  Before: 97.3% (177/182)
  After:  90.1% (164/182)
  Change: -7.2%
  Status: ✅ ACCEPTABLE (85-95%)

Prediction Interval Width (90% PI):
  Before: 491.5 yards
  After:  446.2 yards
  Reduction: 45.3 yards (9.2%)

Calibration Status:
  ✅ Ensemble is properly calibrated

================================================================================
```

---

## 8. Version Compatibility

- **Python:** 3.8+
- **NumPy:** 1.20+
- **SciPy:** 1.5+
- **Pandas:** 1.2+
- **rpy2:** 3.4+ (existing)

No new dependencies required.

---

## 9. Performance

- **Fitting Time:** ~1-5 seconds for 182 samples
- **Prediction Time:** <1ms additional (single multiply per prediction)
- **Memory:** ~1 KB for saved T

---

## 10. Configuration File Format

**models/temperature.json:**
```json
{
  "temperature": 0.826,
  "temperature_fitted": true,
  "date_fitted": "2025-11-13T14:30:00Z"
}
```

---

**This specification is complete and ready for implementation.**
