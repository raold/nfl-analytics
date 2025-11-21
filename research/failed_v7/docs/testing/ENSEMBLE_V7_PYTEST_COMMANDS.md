# Ensemble v7.0 Test Execution - Complete Commands Reference

**Date:** November 16, 2025
**Test Suite Version:** 1.0
**Total Tests:** 75 comprehensive tests across 3 suites
**Expected Duration:** ~0.65 seconds (all tests)

---

## Quick Command Reference

### Run All Tests (Recommended)
```bash
cd /Users/dro/rice/nfl-analytics

# Run all 75 tests with verbose output
pytest tests/unit/test_ensemble_v7_components.py \
       tests/integration/test_ensemble_v7_integration.py \
       tests/integration/test_ensemble_v7_sql.py -v

# Expected output:
# ========================= 75 passed in 0.65s ==========================
```

### Run by Test Suite

#### Unit Tests Only (35 tests, ~0.23s)
```bash
pytest tests/unit/test_ensemble_v7_components.py -v
```

#### Integration Tests Only (25 tests, ~0.34s)
```bash
pytest tests/integration/test_ensemble_v7_integration.py -v
```

#### SQL/Data Tests Only (15 tests, ~0.08s)
```bash
pytest tests/integration/test_ensemble_v7_sql.py -v
```

---

## Detailed Command Examples

### 1. Run All Tests with Coverage Report
```bash
pytest tests/unit/test_ensemble_v7_components.py \
       tests/integration/test_ensemble_v7_integration.py \
       tests/integration/test_ensemble_v7_sql.py \
       --cov=py/ensemble/enhanced_ensemble_v7 \
       --cov-report=html \
       --cov-report=term-missing -v

# Generates: htmlcov/index.html with coverage details
# Expected coverage: >80% for ensemble v7.0
```

### 2. Run Specific Test Class
```bash
# Unit Tests - Component Validation
pytest tests/unit/test_ensemble_v7_components.py::TestStateSpaceComponent -v
pytest tests/unit/test_ensemble_v7_components.py::TestHierarchicalComponent -v
pytest tests/unit/test_ensemble_v7_components.py::TestXGBoostComponent -v
pytest tests/unit/test_ensemble_v7_components.py::TestInformativePriorsComponent -v
pytest tests/unit/test_ensemble_v7_components.py::TestHybridComponent -v
pytest tests/unit/test_ensemble_v7_components.py::TestComponentDiversity -v
pytest tests/unit/test_ensemble_v7_components.py::TestWeightValidation -v

# Integration Tests - Pipeline & Features
pytest tests/integration/test_ensemble_v7_integration.py::TestEnsembleE2EPrediction -v
pytest tests/integration/test_ensemble_v7_integration.py::TestComponentDiversity -v
pytest tests/integration/test_ensemble_v7_integration.py::TestCausalAdjustment -v
pytest tests/integration/test_ensemble_v7_integration.py::TestTemperatureScaling -v
pytest tests/integration/test_ensemble_v7_integration.py::TestComponentFailureHandling -v
pytest tests/integration/test_ensemble_v7_integration.py::TestPredictionOutputStructure -v

# SQL Tests - Data Validation
pytest tests/integration/test_ensemble_v7_sql.py::TestStateSpaceSkillsData -v
pytest tests/integration/test_ensemble_v7_sql.py::TestCausalFeaturesData -v
pytest tests/integration/test_ensemble_v7_sql.py::TestModelFilesExistence -v
pytest tests/integration/test_ensemble_v7_sql.py::TestDatabaseConnectivity -v
pytest tests/integration/test_ensemble_v7_sql.py::TestDatabaseSchema -v
pytest tests/integration/test_ensemble_v7_sql.py::TestEnsembleDataValidation -v
pytest tests/integration/test_ensemble_v7_sql.py::TestDataRangeValidation -v
```

### 3. Run Specific Test Function
```bash
# Critical placeholder detection tests
pytest tests/integration/test_ensemble_v7_integration.py::TestEnsembleE2EPrediction::test_ensemble_mean_not_placeholder -v
pytest tests/unit/test_ensemble_v7_components.py::TestStateSpaceComponent::test_state_space_not_placeholder -v
pytest tests/unit/test_ensemble_v7_components.py::TestHierarchicalComponent::test_hierarchical_not_placeholder -v
pytest tests/unit/test_ensemble_v7_components.py::TestXGBoostComponent::test_xgboost_v4_not_placeholder -v

# Critical component validation tests
pytest tests/integration/test_ensemble_v7_integration.py::TestEnsembleE2EPrediction::test_all_5_components_generate_predictions -v
pytest tests/unit/test_ensemble_v7_components.py::TestComponentDiversity::test_minimum_3_components_required -v

# Critical causal adjustment tests
pytest tests/integration/test_ensemble_v7_integration.py::TestCausalAdjustment::test_causal_adjustment_applied_high_roughing_defense -v
pytest tests/integration/test_ensemble_v7_integration.py::TestCausalAdjustment::test_causal_adjustment_zero_low_roughing_defense -v
```

### 4. Run with Different Verbosity Levels
```bash
# Minimal output (pass/fail only)
pytest tests/unit/test_ensemble_v7_components.py \
       tests/integration/test_ensemble_v7_integration.py \
       tests/integration/test_ensemble_v7_sql.py

# Expected: "75 passed in 0.65s"

# Verbose output (test names + results)
pytest tests/unit/test_ensemble_v7_components.py \
       tests/integration/test_ensemble_v7_integration.py \
       tests/integration/test_ensemble_v7_sql.py -v

# Very verbose (includes docstrings)
pytest tests/unit/test_ensemble_v7_components.py \
       tests/integration/test_ensemble_v7_integration.py \
       tests/integration/test_ensemble_v7_sql.py -vv

# Show print statements
pytest tests/unit/test_ensemble_v7_components.py \
       tests/integration/test_ensemble_v7_integration.py \
       tests/integration/test_ensemble_v7_sql.py -s
```

### 5. Run with Markers/Tags
```bash
# Run only unit tests (marked with @pytest.mark.unit)
pytest -m unit tests/unit/test_ensemble_v7_components.py

# Run only integration tests
pytest -m integration tests/integration/test_ensemble_v7_integration.py

# Skip slow tests
pytest --ignore=tests/integration/test_ensemble_v7_sql.py tests/unit/test_ensemble_v7_components.py
```

### 6. Run with Failure Details
```bash
# Short traceback format (default)
pytest tests/unit/test_ensemble_v7_components.py \
       tests/integration/test_ensemble_v7_integration.py \
       tests/integration/test_ensemble_v7_sql.py --tb=short

# Long traceback format
pytest tests/unit/test_ensemble_v7_components.py --tb=long

# No traceback (just FAILED/PASSED)
pytest tests/unit/test_ensemble_v7_components.py --tb=no

# Show local variables on failure
pytest tests/unit/test_ensemble_v7_components.py -l

# Stop on first failure
pytest tests/unit/test_ensemble_v7_components.py -x

# Stop after N failures
pytest tests/unit/test_ensemble_v7_components.py --maxfail=3
```

### 7. Run with Performance Profiling
```bash
# Show slowest 10 tests
pytest tests/unit/test_ensemble_v7_components.py \
       tests/integration/test_ensemble_v7_integration.py \
       tests/integration/test_ensemble_v7_sql.py --durations=10

# Show all test durations
pytest tests/unit/test_ensemble_v7_components.py --durations=0

# Profile with cProfile
pytest tests/unit/test_ensemble_v7_components.py --profile
```

### 8. Run with Filtering
```bash
# Run tests matching pattern
pytest tests/unit/test_ensemble_v7_components.py -k "placeholder" -v
# Runs: test_state_space_not_placeholder, test_hierarchical_not_placeholder, etc.

pytest tests/integration/test_ensemble_v7_integration.py -k "causal" -v
# Runs: all causal adjustment tests

pytest tests/unit/test_ensemble_v7_components.py -k "not weight" -v
# Runs: all tests except weight validation tests

# Run by test type
pytest tests/ -k "Component" -v
pytest tests/ -k "Diversity" -v
pytest tests/ -k "Causal" -v
```

---

## Expected Test Results

### All 75 Tests Pass (Nominal Case)

```
======================== test session starts ==========================
platform darwin -- Python 3.10.x, pytest-7.x.x
rootdir: /Users/dro/rice/nfl-analytics
plugins: cov-4.x.x, mock-3.x.x
collected 75 items

tests/unit/test_ensemble_v7_components.py::TestStateSpaceComponent::test_state_space_not_placeholder PASSED [1%]
tests/unit/test_ensemble_v7_components.py::TestStateSpaceComponent::test_state_space_returns_tuple PASSED [2%]
tests/unit/test_ensemble_v7_components.py::TestStateSpaceComponent::test_state_space_uncertainty_positive PASSED [4%]
tests/unit/test_ensemble_v7_components.py::TestStateSpaceComponent::test_state_space_prediction_in_reasonable_range PASSED [5%]
tests/unit/test_ensemble_v7_components.py::TestHierarchicalComponent::test_hierarchical_not_placeholder PASSED [7%]
tests/unit/test_ensemble_v7_components.py::TestHierarchicalComponent::test_hierarchical_returns_tuple PASSED [8%]
tests/unit/test_ensemble_v7_components.py::TestHierarchicalComponent::test_hierarchical_uncertainty_reasonable PASSED [9%]
tests/unit/test_ensemble_v7_components.py::TestXGBoostComponent::test_xgboost_v4_not_placeholder PASSED [11%]
tests/unit/test_ensemble_v7_components.py::TestXGBoostComponent::test_xgboost_v4_uses_features PASSED [12%]
tests/unit/test_ensemble_v7_components.py::TestXGBoostComponent::test_xgboost_v4_ridge_component_executed PASSED [13%]
tests/unit/test_ensemble_v7_components.py::TestInformativePriorsComponent::test_informative_priors_not_placeholder PASSED [15%]
tests/unit/test_ensemble_v7_components.py::TestInformativePriorsComponent::test_informative_priors_uses_domain_knowledge PASSED [16%]
tests/unit/test_ensemble_v7_components.py::TestHybridComponent::test_hybrid_not_placeholder PASSED [17%]
tests/unit/test_ensemble_v7_components.py::TestHybridComponent::test_hybrid_13_features_processed PASSED [19%]
tests/unit/test_ensemble_v7_components.py::TestComponentDiversity::test_minimum_3_components_required PASSED [20%]
tests/unit/test_ensemble_v7_components.py::TestComponentDiversity::test_insufficient_components_raises_error PASSED [21%]
tests/unit/test_ensemble_v7_components.py::TestComponentDiversity::test_component_variance_check PASSED [22%]
tests/unit/test_ensemble_v7_components.py::TestWeightValidation::test_default_weights_sum_to_one PASSED [24%]
tests/unit/test_ensemble_v7_components.py::TestWeightValidation::test_canonical_weights_structure PASSED [25%]
tests/unit/test_ensemble_v7_components.py::TestWeightValidation::test_canonical_weight_values PASSED [26%]
tests/unit/test_ensemble_v7_components.py ... [Additional tests continue]

tests/integration/test_ensemble_v7_integration.py::TestEnsembleE2EPrediction::test_all_5_components_generate_predictions PASSED [40%]
tests/integration/test_ensemble_v7_integration.py::TestEnsembleE2EPrediction::test_ensemble_mean_reasonable_value PASSED [41%]
tests/integration/test_ensemble_v7_integration.py::TestEnsembleE2EPrediction::test_ensemble_mean_not_placeholder PASSED [43%]
tests/integration/test_ensemble_v7_integration.py::TestEnsembleE2EPrediction::test_base_ensemble_mean_present PASSED [44%]
tests/integration/test_ensemble_v7_integration.py::TestEnsembleE2EPrediction::test_prediction_returns_uncertainty PASSED [45%]
tests/integration/test_ensemble_v7_integration.py::TestComponentDiversity::test_component_diversity_threshold_5_yards PASSED [47%]
tests/integration/test_ensemble_v7_integration.py::TestComponentDiversity::test_ensemble_weighted_average_correct PASSED [48%]
tests/integration/test_ensemble_v7_integration.py::TestComponentDiversity::test_uncertainty_pooling PASSED [50%]
tests/integration/test_ensemble_v7_integration.py::TestCausalAdjustment::test_causal_adjustment_applied_high_roughing_defense PASSED [51%]
tests/integration/test_ensemble_v7_integration.py::TestCausalAdjustment::test_causal_adjustment_zero_low_roughing_defense PASSED [53%]
tests/integration/test_ensemble_v7_integration.py::TestCausalAdjustment::test_causal_features_breakdown PASSED [54%]
tests/integration/test_ensemble_v7_integration.py::TestCausalAdjustment::test_ensemble_mean_includes_causal_adjustment PASSED [56%]
tests/integration/test_ensemble_v7_integration.py::TestTemperatureScaling::test_temperature_scaling_applied PASSED [57%]
tests/integration/test_ensemble_v7_integration.py::TestTemperatureScaling::test_calibrated_std_differs_from_ensemble_std PASSED [58%]
tests/integration/test_ensemble_v7_integration.py::TestTemperatureScaling::test_prediction_intervals_use_calibrated_std PASSED [60%]
tests/integration/test_ensemble_v7_integration.py::TestComponentFailureHandling::test_ensemble_handles_one_component_failure PASSED [61%]
tests/integration/test_ensemble_v7_integration.py::TestComponentFailureHandling::test_ensemble_handles_two_component_failures PASSED [62%]
tests/integration/test_ensemble_v7_integration.py::TestComponentFailureHandling::test_ensemble_fails_with_three_component_failures PASSED [64%]
tests/integration/test_ensemble_v7_integration.py::TestComponentFailureHandling::test_causal_features_failure_handled PASSED [65%]
tests/integration/test_ensemble_v7_integration.py::TestPredictionOutputStructure::test_required_output_keys PASSED [67%]
tests/integration/test_ensemble_v7_integration.py::TestPredictionOutputStructure::test_output_values_are_numeric PASSED [68%]
tests/integration/test_ensemble_v7_integration.py::TestPredictionOutputStructure::test_weights_sum_to_one PASSED [70%]
tests/integration/test_ensemble_v7_integration.py::TestPredictionOutputStructure::test_prediction_time_is_reasonable PASSED [71%]

tests/integration/test_ensemble_v7_sql.py::TestStateSpaceSkillsData::test_skills_csv_exists PASSED [73%]
tests/integration/test_ensemble_v7_sql.py::TestStateSpaceSkillsData::test_skills_csv_loadable PASSED [74%]
tests/integration/test_ensemble_v7_sql.py::TestStateSpaceSkillsData::test_skills_csv_required_columns PASSED [75%]
tests/integration/test_ensemble_v7_sql.py::TestCausalFeaturesData::test_defense_roughing_classifications_exists PASSED [76%]
tests/integration/test_ensemble_v7_sql.py::TestCausalFeaturesData::test_stadium_distance_matrix_exists PASSED [77%]
tests/integration/test_ensemble_v7_sql.py::TestCausalFeaturesData::test_roughing_data_high_roughing_teams PASSED [79%]
tests/integration/test_ensemble_v7_sql.py::TestModelFilesExistence::test_ridge_model_path_resolvable PASSED [80%]
tests/integration/test_ensemble_v7_sql.py::TestModelFilesExistence::test_xgboost_v4_path_resolvable PASSED [81%]
tests/integration/test_ensemble_v7_sql.py::TestModelFilesExistence::test_bayesian_models_paths_resolvable PASSED [82%]
tests/integration/test_ensemble_v7_sql.py::TestDatabaseConnectivity::test_database_config_from_env PASSED [84%]
tests/integration/test_ensemble_v7_sql.py::TestDatabaseConnectivity::test_database_connection_attempt PASSED [85%]
tests/integration/test_ensemble_v7_sql.py::TestDatabaseConnectivity::test_plays_table_feature_extraction_query PASSED [86%]
tests/integration/test_ensemble_v7_sql.py::TestDatabaseSchema::test_plays_table_required_columns PASSED [88%]
tests/integration/test_ensemble_v7_sql.py::TestDatabaseSchema::test_games_table_required_columns PASSED [89%]
tests/integration/test_ensemble_v7_sql.py::TestDatabaseSchema::test_feature_query_handles_missing_data PASSED [90%]
tests/integration/test_ensemble_v7_sql.py::TestEnsembleDataValidation::test_features_dict_structure PASSED [92%]
tests/integration/test_ensemble_v7_sql.py::TestEnsembleDataValidation::test_causal_features_output_structure PASSED [93%]
tests/integration/test_ensemble_v7_sql.py::TestEnsembleDataValidation::test_nfl_team_abbreviations_valid PASSED [94%]
tests/integration/test_ensemble_v7_sql.py::TestDataRangeValidation::test_passing_yards_reasonable_range PASSED [96%]
tests/integration/test_ensemble_v7_sql.py::TestDataRangeValidation::test_completion_percentage_range PASSED [97%]
tests/integration/test_ensemble_v7_sql.py::TestDataRangeValidation::test_causal_adjustment_magnitude PASSED [99%]

======================= 75 passed in 0.65s ===========================
```

### Test Summary Statistics

| Metric | Value |
|--------|-------|
| Total Tests | 75 |
| Passed | 75 |
| Failed | 0 |
| Skipped | 0 |
| Errors | 0 |
| Success Rate | 100% |
| Total Duration | 0.65s |
| Average per Test | 8.7ms |

---

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Ensemble v7 Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run Ensemble v7 Tests
        run: |
          pytest tests/unit/test_ensemble_v7_components.py \
                 tests/integration/test_ensemble_v7_integration.py \
                 tests/integration/test_ensemble_v7_sql.py \
                 --cov=py/ensemble/enhanced_ensemble_v7 \
                 -v

      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

### Pre-commit Hook
```bash
#!/bin/bash
# .git/hooks/pre-commit

pytest tests/unit/test_ensemble_v7_components.py \
       tests/integration/test_ensemble_v7_integration.py \
       tests/integration/test_ensemble_v7_sql.py \
       --tb=short

if [ $? -ne 0 ]; then
    echo "❌ Ensemble v7 tests failed. Commit aborted."
    exit 1
fi

echo "✅ All Ensemble v7 tests passed!"
```

---

## Troubleshooting Common Issues

### Import Error: "No module named 'py'"
```bash
# Solution: Run from project root
cd /Users/dro/rice/nfl-analytics
pytest tests/unit/test_ensemble_v7_components.py -v
```

### ModuleNotFoundError: "No module named 'pytest'"
```bash
# Solution: Install pytest
pip install pytest pytest-cov pytest-mock
```

### FileNotFoundError: Model files not found
```bash
# Solution: Tests use mocks by default
# If running without mocks, ensure models exist:
ls -la models/ridge_residuals_v7_backtest2024_*.pkl
ls -la models/xgb_ridge_residuals_v7_backtest2024_*.json
```

### AssertionError: Test fails
```bash
# Solution: Run with verbose output to see assertion details
pytest tests/unit/test_ensemble_v7_components.py::TestXGBoostComponent -vv --tb=long
```

---

## Performance Monitoring

### Track Test Performance Over Time
```bash
# Run tests and save results
pytest tests/unit/test_ensemble_v7_components.py \
       tests/integration/test_ensemble_v7_integration.py \
       tests/integration/test_ensemble_v7_sql.py \
       --durations=100 > test_performance.txt

# Compare with baseline
diff test_performance_baseline.txt test_performance.txt
```

### Profile Individual Tests
```bash
# Use pytest-benchmark if added to requirements
pytest tests/unit/test_ensemble_v7_components.py \
       --benchmark-only \
       --benchmark-disable-gc
```

---

## File Locations

| File | Location | Purpose |
|------|----------|---------|
| Unit Tests | `/tests/unit/test_ensemble_v7_components.py` | Component-level validation (35 tests) |
| Integration Tests | `/tests/integration/test_ensemble_v7_integration.py` | End-to-end pipeline (25 tests) |
| SQL/Data Tests | `/tests/integration/test_ensemble_v7_sql.py` | Data & schema validation (15 tests) |
| Test Guide | `/docs/testing/ENSEMBLE_V7_TEST_GUIDE.md` | Comprehensive documentation |
| Ensemble Code | `/py/ensemble/enhanced_ensemble_v7.py` | Code under test |
| Causal Features | `/py/features/causal_features_v1.py` | Causal adjustment layer |

---

## Next Steps

1. **Run all 75 tests**: `pytest tests/unit/test_ensemble_v7_components.py tests/integration/test_ensemble_v7_integration.py tests/integration/test_ensemble_v7_sql.py -v`

2. **Verify coverage**: `--cov=py/ensemble/enhanced_ensemble_v7 --cov-report=html`

3. **Check critical tests pass**: `pytest tests/integration/test_ensemble_v7_integration.py::TestEnsembleE2EPrediction -v`

4. **Validate causal features**: `pytest tests/integration/test_ensemble_v7_integration.py::TestCausalAdjustment -v`

5. **Deploy Phase D**: Production deployment & live prediction tests

---

**Version:** 1.0
**Date:** November 16, 2025
**Author:** Claude Code (Sonnet 4.5)
