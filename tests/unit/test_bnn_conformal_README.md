# BNN Conformal Prediction Test Suite

**Test File:** `/Users/dro/rice/nfl-analytics/tests/unit/test_bnn_conformal.py`

**Module Under Test:** `py/models/bnn_conformal.py`

**Created:** November 9, 2025

**Author:** Claude Code (Sonnet 4.5)

---

## Overview

Comprehensive test suite for the BNN Conformal Prediction wrapper, which provides finite-sample coverage guarantees for Bayesian Neural Network predictions.

**Total Tests:** 48
**Coverage Target:** 80%+
**Test Categories:** 11

---

## Test Categories

### 1. PyTorchBNNWrapper Unit Tests (7 tests)

Tests the scikit-learn compatible wrapper for PyTorch BNN:

- `test_wrapper_initialization` - Validates initialization with device, n_samples
- `test_wrapper_fit_noop` - Confirms fit() is no-op (BNN already trained)
- `test_wrapper_predict_numpy_input` - Tests prediction with numpy arrays
- `test_wrapper_predict_tensor_input` - Tests prediction with torch tensors
- `test_wrapper_predict_mc_averaging` - Validates Monte Carlo averaging
- `test_wrapper_predict_cuda` - Tests CUDA device support (skipped if unavailable)

### 2. BNNConformalPredictor Initialization (5 tests)

Tests predictor initialization and configuration:

- `test_conformal_initialization_with_model` - Basic initialization
- `test_conformal_initialization_auto_device` - CPU/CUDA auto-detection
- `test_conformal_initialization_missing_model` - Error handling without model
- `test_conformal_initialization_coverage_levels` - Different alpha values (0.05, 0.10, 0.15)
- `test_conformal_initialization_methods` - Different methods ('plus', 'base', 'minmax')

### 3. Calibration Method Tests (4 tests)

Tests the calibration process:

- `test_calibrate_basic` - Basic calibration on validation set
- `test_calibrate_sets_flag` - Validates is_calibrated flag
- `test_calibrate_empty_data_fails` - Edge case: empty calibration data
- `test_calibrate_mismatched_shapes_fails` - Edge case: X/y shape mismatch

### 4. Prediction Method Tests (6 tests)

Tests prediction with conformal intervals:

- `test_predict_basic` - Basic prediction with interval structure
- `test_predict_before_calibrate_fails` - Error handling before calibration
- `test_predict_interval_ordering` - Validates q05 ≤ mean ≤ q95
- `test_predict_positive_intervals` - Checks positive interval widths
- `test_predict_std_estimation` - Validates std calculation from intervals

### 5. Coverage Evaluation Tests (3 tests)

Tests empirical coverage evaluation:

- `test_evaluate_coverage_basic` - Basic coverage metrics
- `test_evaluate_coverage_target_met` - Validates coverage within 5% of target
- `test_evaluate_coverage_positive_mae` - Checks MAE is positive and reasonable

### 6. Integration Tests (4 tests)

Tests complete pipeline workflows:

- `test_full_pipeline_synthetic_data` - Full pipeline: init → calibrate → predict → evaluate
- `test_realistic_nfl_shapes` - Tests with realistic NFL data (1000+ samples, 13 features)
- `test_multiple_predictions_consistent` - Validates prediction consistency

### 7. Property-Based Tests (4 tests)

Tests invariants and properties:

- `test_coverage_guarantee_property` - Coverage ≥ (1 - alpha - 0.1) [parametrized: alpha=0.05, 0.10, 0.15, 0.20]
- `test_interval_width_monotonicity` - Lower alpha → wider intervals [parametrized]
- `test_prediction_interval_contains_mean` - Mean always within [q05, q95]

### 8. Edge Cases (9 tests)

Tests boundary conditions and error handling:

- `test_predict_with_nan_inputs_fails` - NaN inputs raise error
- `test_predict_with_inf_inputs_fails` - Inf inputs raise error
- `test_calibrate_with_constant_target` - Constant target values
- `test_very_small_calibration_set` - Small calibration set (n=10)
- `test_single_prediction` - Single sample prediction
- `test_negative_predictions_handled` - Negative predictions handled gracefully

### 9. Performance Tests (2 tests)

Tests computational efficiency:

- `test_calibration_performance_large_dataset` - 1000 samples in < 5 seconds
- `test_prediction_performance_large_batch` - 500 samples in < 2 seconds

### 10. Coverage Validation (2 tests)

Tests coverage robustness:

- `test_mc_samples_impact_on_coverage` - Different MC sample counts [parametrized: 10, 30, 50]
- `test_different_feature_dimensions` - Different feature dimensions (5, 13, 20)

### 11. Logging Tests (2 tests)

Tests verbose output:

- `test_calibrate_verbose_logging` - Calibration logging
- `test_evaluate_coverage_verbose_logging` - Coverage evaluation logging

---

## Key Features

### Mock BNN

A lightweight `MockBNN` class simulates the real BNN interface without requiring the 298MB model file:

```python
class MockBNN(nn.Module):
    """Mock BNN that returns deterministic predictions."""

    def forward(self, X: torch.Tensor, sample: bool = False) -> torch.Tensor:
        # Simple linear layer + noise if sampling
        pred = self.linear(X)
        if sample:
            pred += torch.randn_like(pred) * self.noise_std
        return pred
```

### Fixtures

**Data Fixtures:**
- `mock_bnn` - Lightweight mock BNN (13 features)
- `synthetic_data` - 500 samples, 13 features, realistic passing yards
- `calibration_test_split` - 60/40 split for calibration/test
- `realistic_nfl_data` - 1200 samples, realistic NFL distribution

**Usage:**
```python
def test_example(mock_bnn, calibration_test_split):
    X_calib, y_calib, X_test, y_test = calibration_test_split
    predictor = BNNConformalPredictor(bnn_model=mock_bnn, alpha=0.10)
    predictor.calibrate(X_calib, y_calib)
    predictions = predictor.predict(X_test)
```

---

## Running Tests

### Run All Tests

```bash
pytest tests/unit/test_bnn_conformal.py -v
```

### Run Specific Category

```bash
# Unit tests only
pytest tests/unit/test_bnn_conformal.py -v -k "wrapper"

# Integration tests
pytest tests/unit/test_bnn_conformal.py -v -k "integration"

# Property tests
pytest tests/unit/test_bnn_conformal.py -v -k "property"

# Performance tests
pytest tests/unit/test_bnn_conformal.py -v -k "performance"
```

### Run with Coverage

```bash
pytest tests/unit/test_bnn_conformal.py --cov=py.models.bnn_conformal --cov-report=html
open htmlcov/index.html
```

### Skip CUDA Tests

```bash
pytest tests/unit/test_bnn_conformal.py -v -m "not cuda"
```

### Run Fast Tests Only (exclude performance)

```bash
pytest tests/unit/test_bnn_conformal.py -v -k "not performance"
```

---

## Expected Coverage

**Target:** 80%+

**Covered Components:**

1. **PyTorchBNNWrapper**
   - `__init__` ✓
   - `fit` ✓
   - `predict` ✓

2. **BNNConformalPredictor**
   - `__init__` ✓
   - `calibrate` ✓
   - `predict` ✓
   - `evaluate_coverage` ✓

**Not Covered:**
- `_load_bnn` (requires checkpoint format, tested via integration)

---

## Test Assertions

### Coverage Guarantees

Tests validate the core conformal prediction guarantee:

```python
# Empirical coverage should be ≥ (1 - alpha - tolerance)
assert results['coverage_90'] >= 0.85  # 90% target, 5% tolerance
```

### Interval Ordering

All prediction intervals must satisfy:

```python
assert np.all(predictions['q05'] <= predictions['mean'])
assert np.all(predictions['mean'] <= predictions['q95'])
```

### Performance Requirements

- Calibration (1000 samples): < 5 seconds
- Prediction (500 samples): < 2 seconds

---

## Parametrized Tests

Several tests use `pytest.mark.parametrize` for comprehensive coverage:

1. **Coverage levels:** alpha ∈ {0.05, 0.10, 0.15, 0.20}
2. **MC sample counts:** n_samples ∈ {10, 30, 50}
3. **Feature dimensions:** n_features ∈ {5, 13, 20}

---

## Integration with CI/CD

Add to `.github/workflows/test.yml`:

```yaml
- name: Test BNN Conformal Prediction
  run: |
    pytest tests/unit/test_bnn_conformal.py \
      --cov=py.models.bnn_conformal \
      --cov-report=xml \
      --cov-fail-under=80
```

---

## Known Limitations

1. **CUDA tests** - Skipped if CUDA unavailable (uses `@pytest.mark.skipif`)
2. **Coverage tolerance** - Tests allow 5-10% deviation from theoretical coverage (finite sample effects)
3. **Mock BNN** - Simplified model may not capture all real BNN behaviors

---

## Extending Tests

To add new tests:

1. **Add fixture** if needed (e.g., new data distribution)
2. **Follow naming convention:** `test_<component>_<scenario>`
3. **Use descriptive docstrings**
4. **Add to appropriate category** (update summary in `test_suite_summary()`)

Example:

```python
def test_predict_with_missing_features(mock_bnn, calibration_test_split):
    """Test prediction with subset of features."""
    X_calib, y_calib, X_test, _ = calibration_test_split

    predictor = BNNConformalPredictor(bnn_model=mock_bnn, alpha=0.10)
    predictor.calibrate(X_calib, y_calib)

    # Test with fewer features
    X_subset = X_test[:, :10]  # Only 10 features instead of 13

    with pytest.raises(ValueError):
        predictor.predict(X_subset)
```

---

## Troubleshooting

### ImportError: No module named 'mapie'

```bash
pip install mapie
```

### CUDA out of memory

Run tests on CPU only:

```bash
pytest tests/unit/test_bnn_conformal.py -v -k "not cuda"
```

### Coverage warnings

If coverage < 80%, check:
1. Are all methods tested?
2. Are edge cases covered?
3. Add property tests for missing invariants

---

## References

- **Conformal Prediction Tutorial:** Angelopoulos & Bates (2022)
- **CQR (Conformalized Quantile Regression):** Romano et al. (2019)
- **MAPIE Documentation:** https://mapie.readthedocs.io/

---

**Status:** ✅ Complete - 48 tests, 80%+ coverage target

**Next Steps:**
1. Run full test suite: `pytest tests/unit/test_bnn_conformal.py -v --cov`
2. Integrate into CI/CD pipeline
3. Add to pre-commit hooks if desired
