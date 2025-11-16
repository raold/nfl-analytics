# Test Generation Plan to Reach 80% Coverage

**Current Coverage:** 62-68%
**Target Coverage:** 80%+
**Gap to Close:** 12-18 percentage points
**Estimated Effort:** 71 hours over 6 weeks
**Expected ROI:** 105+ hours saved

---

## Priority Roadmap

### Week 1: P0 Critical Gaps (12 hours → 19 hours ROI)

#### Task 1.1: Feature Repository Caching Tests (4 hours)
**File:** `tests/integration/test_feature_repository_caching.py`
**Tests:** 12 tests, ~300 lines
**Coverage Target:** 25% → 70%

```python
"""
Feature Repository Caching Critical Tests

Validates:
- Cache hit/miss rates (70-85% target)
- TTL expiration (stale data prevention)
- Memory eviction (LRU policy)
- Cache coherency (fresh data guarantee)
"""

class TestCacheHitMissRates:
    def test_cache_hit_rate_repeated_queries()
    def test_cache_miss_on_first_query()
    def test_cache_hit_after_warm_up()

class TestTTLExpiration:
    def test_cache_expires_after_ttl()
    def test_fresh_data_after_expiration()
    def test_ttl_refresh_on_access()

class TestMemoryEviction:
    def test_lru_evicts_least_recently_used()
    def test_cache_size_limit_enforced()
    def test_memory_stays_below_threshold()

class TestCacheCoherency:
    def test_cache_invalidation_on_model_update()
    def test_no_stale_predictions()
    def test_concurrent_cache_access()
```

**ROI:** Prevents stale feature bugs (6 hours saved per incident)

---

#### Task 1.2: SQL Query Validation Tests (5 hours)
**File:** `tests/integration/test_sql_query_validation.py`
**Tests:** 16 tests, ~400 lines
**Coverage Target:** 40% → 80%

```python
"""
SQL Query Validation Critical Tests

Validates:
- Schema enforcement (mart.player_game_stats columns)
- Team abbreviation correctness (LAR/LAC not LA)
- Position eligibility (QB/RB/WR/TE)
- Stat range validation (passing <600, rushing <300)
"""

class TestSchemaEnforcement:
    def test_player_game_stats_schema()
    def test_no_game_id_column_used()  # Common mistake
    def test_required_columns_present()
    def test_column_types_correct()

class TestTeamAbbreviations:
    def test_lar_lac_not_la()
    def test_all_teams_valid()
    def test_historical_team_renames()

class TestPositionEligibility:
    def test_qb_position_group()
    def test_rb_wr_te_positions()
    def test_invalid_positions_rejected()

class TestStatRanges:
    def test_passing_yards_bounds()
    def test_rushing_yards_bounds()
    def test_receiving_yards_bounds()
    def test_negative_stats_rejected()
```

**ROI:** Prevents query failures from schema evolution (8 hours saved per incident)

---

#### Task 1.3: Model Serialization Tests (3 hours)
**File:** `tests/integration/test_model_serialization.py`
**Tests:** 10 tests, ~250 lines
**Coverage Target:** 25% → 75%

```python
"""
Model Serialization Critical Tests

Validates:
- Save/load round-trip consistency
- Model versioning
- Corrupted file recovery
- Forward compatibility
"""

class TestSaveLoadRoundTrip:
    def test_xgboost_model_round_trip()
    def test_r_model_round_trip()
    def test_temperature_scaler_round_trip()

class TestModelVersioning:
    def test_version_metadata_saved()
    def test_version_mismatch_detected()
    def test_version_upgrade_path()

class TestCorruptedFileRecovery:
    def test_corrupted_pkl_handled()
    def test_corrupted_rds_handled()
    def test_fallback_to_default_model()

class TestForwardCompatibility:
    def test_old_model_loads_in_new_code()
```

**ROI:** Prevents model loading failures (5 hours saved per incident)

---

### Week 2: P1 Risk Management (10 hours → 15 hours ROI)

#### Task 2.1: CVaR LP Edge Cases (4 hours)
**File:** `tests/unit/test_cvar_lp_edge_cases.py`
**Tests:** 10 tests, ~250 lines
**Coverage Target:** 40% → 75%

```python
"""
CVaR Linear Programming Edge Case Tests

Validates:
- Infeasible problems (budget < required staking)
- Degenerate solutions (zero stakes)
- Solver convergence
- Large portfolios (1000+ positions)
"""

class TestInfeasibleProblems:
    def test_insufficient_budget()
    def test_conflicting_constraints()
    def test_infeasible_detection()

class TestDegenerateSolutions:
    def test_zero_stake_solutions()
    def test_all_bets_rejected()
    def test_corner_case_solutions()

class TestSolverConvergence:
    def test_solver_timeout()
    def test_numerical_stability()

class TestLargePortfolios:
    def test_1000_position_portfolio()
```

**ROI:** Prevents bankruptcy scenarios (8 hours saved per incident)

---

#### Task 2.2: Kelly Criterion Edge Cases (3 hours)
**File:** `tests/unit/test_kelly_criterion_edge_cases.py`
**Tests:** 12 tests, ~300 lines
**Coverage Target:** 35% → 80%

```python
"""
Kelly Criterion Edge Case Tests

Validates:
- Win probability bounds (0-1)
- Fractional Kelly edge cases
- Negative Kelly (reduce position)
- Bankruptcy scenarios
"""

class TestProbabilityBounds:
    def test_p_win_greater_than_1()
    def test_p_win_equal_to_0()
    def test_p_win_negative()

class TestFractionalKelly:
    def test_half_kelly()
    def test_quarter_kelly()
    def test_fractional_kelly_edge_cases()

class TestNegativeKelly:
    def test_negative_kelly_reduces_position()
    def test_negative_kelly_prevents_bet()

class TestBankruptcyScenarios:
    def test_kelly_prevents_bankruptcy()
    def test_max_kelly_fraction()
    def test_losing_streak_recovery()
```

**ROI:** Prevents Kelly sizing errors (4 hours saved per incident)

---

#### Task 2.3: State-Space Integration (3 hours)
**File:** `tests/integration/test_state_space_integration.py`
**Tests:** 8 tests, ~200 lines
**Coverage Target:** 45% → 75%

```python
"""
State-Space Model Integration Tests

Validates:
- Prediction correctness (MAE <50 yards)
- Skills DataFrame loading
- Prediction interval calibration
- Multi-season trajectory modeling
"""

class TestPredictionCorrectness:
    def test_state_space_mae_validation()
    def test_prediction_bounds()

class TestSkillsDataFrame:
    def test_skills_df_loaded()
    def test_player_skill_lookup()

class TestPredictionIntervals:
    def test_calibration_coverage()
    def test_interval_width()

class TestMultiSeasonTrajectories:
    def test_trajectory_continuity()
```

**ROI:** Validates state-space v2 (47.77 MAE) correctness (3 hours saved per incident)

---

### Week 3: P1 XGBoost & BNN (8 hours → 10 hours ROI)

#### Task 3.1: XGBoost v4 Ridge Residuals (4 hours)
**File:** `tests/integration/test_xgboost_v4_ridge_residuals.py`
**Tests:** 14 tests, ~350 lines
**Coverage Target:** 65% → 85%

```python
"""
XGBoost v4 Ridge Residuals Critical Tests

Validates:
- Residual feature=0 case (QB-specific expected residuals)
- Ensemble integration with residuals
- Ridge regression component
"""

class TestResidualFeature:
    def test_residual_feature_zero_case()
    def test_qb_specific_expected_residuals()
    def test_residual_calculation_correctness()

class TestEnsembleIntegration:
    def test_xgboost_v4_in_ensemble()
    def test_residual_propagation()
    def test_ensemble_aggregation_with_residuals()

class TestRidgeRegression:
    def test_ridge_coefficient_validation()
    def test_ridge_regularization_strength()
    def test_ridge_predictions_bounded()

class TestPredictionAccuracy:
    def test_mae_below_5_yards()
    def test_prediction_distribution()
```

**ROI:** Validates XGBoost v4 (4.7 MAE, most accurate component) (6 hours saved per incident)

---

#### Task 3.2: BNN Calibration Edge Cases (4 hours)
**File:** `tests/unit/test_bnn_calibration_edge_cases.py`
**Tests:** 15 tests, ~375 lines
**Coverage Target:** 55% → 75%

```python
"""
BNN Calibration Edge Case Tests

Validates:
- Conformal prediction intervals
- Quantile regression baselines
- Temperature scaling edge cases
"""

class TestConformalPrediction:
    def test_conformal_intervals_coverage()
    def test_conformal_interval_width()
    def test_conformal_calibration()

class TestQuantileRegression:
    def test_quantile_predictions()
    def test_quantile_crossing_prevention()

class TestTemperatureScaling:
    def test_temperature_above_2()
    def test_temperature_below_0_3()
    def test_temperature_optimization()

class TestCalibrationCurves:
    def test_calibration_curve_generation()
    def test_reliability_diagram()
```

**ROI:** Improves calibration coverage (target 90%) (4 hours saved per incident)

---

### Week 4: P2 Error Handling (6 hours → 8 hours ROI)

#### Task 4.1: Database Failure Recovery (3 hours)
**File:** `tests/integration/test_database_failure_recovery.py`
**Tests:** 10 tests, ~250 lines
**Coverage Target:** 40% → 75%

```python
"""
Database Failure Recovery Tests

Validates:
- Connection failure handling
- Query retry logic
- Graceful degradation
"""

class TestConnectionFailure:
    def test_connection_retry()
    def test_max_retries_exceeded()
    def test_fallback_to_cache()

class TestQueryRetry:
    def test_transient_error_retry()
    def test_permanent_error_abort()

class TestGracefulDegradation:
    def test_database_unavailable_fallback()
    def test_partial_data_handling()
```

**ROI:** Prevents production failures (5 hours saved per incident)

---

#### Task 4.2: Network Timeout Handling (3 hours)
**File:** `tests/integration/test_network_timeout_handling.py`
**Tests:** 8 tests, ~200 lines
**Coverage Target:** 30% → 70%

```python
"""
Network Timeout Handling Tests

Validates:
- API request timeouts
- Database query timeouts
- R model prediction timeouts
"""

class TestAPITimeouts:
    def test_prediction_api_timeout()
    def test_timeout_error_message()

class TestDatabaseTimeouts:
    def test_query_timeout()
    def test_connection_timeout()

class TestRModelTimeouts:
    def test_r_prediction_timeout()
    def test_rpy2_timeout_handling()
```

**ROI:** Prevents timeout cascades (3 hours saved per incident)

---

### Week 5: P2 Performance Regression (5 hours → 6 hours ROI)

#### Task 5.1: Inference Latency Tracking (3 hours)
**File:** `tests/performance/test_inference_latency_tracking.py`
**Tests:** 8 tests, ~200 lines
**Coverage Target:** 30% → 70%

```python
"""
Inference Latency Tracking Tests

Validates:
- <400ms SLA for single predictions
- Latency percentiles (p50, p95, p99)
- Regression detection
"""

class TestLatencySLA:
    def test_single_prediction_under_400ms()
    def test_batch_prediction_throughput()

class TestLatencyPercentiles:
    def test_p50_latency()
    def test_p95_latency()
    def test_p99_latency()

class TestRegressionDetection:
    def test_latency_increase_alert()
    def test_latency_trending()
```

**ROI:** Catches performance degradation early (3 hours saved per incident)

---

#### Task 5.2: Memory Usage Monitoring (2 hours)
**File:** `tests/performance/test_memory_usage_monitoring.py`
**Tests:** 6 tests, ~150 lines
**Coverage Target:** 25% → 65%

```python
"""
Memory Usage Monitoring Tests

Validates:
- <2GB RSS limit
- No memory leaks
- Memory growth tracking
"""

class TestMemoryLimits:
    def test_memory_under_2gb()
    def test_batch_prediction_memory()

class TestMemoryLeaks:
    def test_no_leak_after_1000_predictions()

class TestMemoryGrowth:
    def test_memory_growth_tracking()
    def test_memory_regression_detection()
```

**ROI:** Prevents OOM errors (3 hours saved per incident)

---

### Week 6: P2 Property-Based Testing (8 hours → 12 hours ROI)

#### Task 6.1: Hypothesis Integration (8 hours)
**File:** `tests/property/test_ensemble_properties.py`
**Tests:** 20 tests, ~500 lines
**Coverage Target:** New (0% → 60%)

```python
"""
Property-Based Testing for Ensemble

Uses Hypothesis library for adversarial input generation.

Validates:
- Ensemble behavior under random inputs
- Invariants hold across all inputs
- No crashes on edge cases
"""

import hypothesis
from hypothesis import strategies as st

class TestEnsembleInvariants:
    @hypothesis.given(
        player_id=st.text(min_size=10, max_size=15),
        season=st.integers(min_value=2015, max_value=2030),
        week=st.integers(min_value=1, max_value=18)
    )
    def test_prediction_always_bounded(player_id, season, week):
        """Predictions always in [0, 600] yards."""
        pass

    @hypothesis.given(
        predictions=st.lists(
            st.tuples(st.floats(0, 600), st.floats(5, 100)),
            min_size=3, max_size=5
        )
    )
    def test_ensemble_mean_within_component_bounds(predictions):
        """Ensemble mean always within component bounds."""
        pass

    @hypothesis.given(
        weights=st.lists(st.floats(0, 1), min_size=4, max_size=4)
    )
    def test_weight_normalization(weights):
        """Weights always sum to 1.0."""
        pass
```

**ROI:** Discovers edge cases not found by manual tests (12 hours saved from bug discovery)

---

## ELO Component - Comprehensive Test Suite

### Week 1.5: ELO Component Tests (6 hours → 8 hours ROI)

#### Task 1.5.1: ELO Rating Accuracy (3 hours)
**File:** `tests/unit/test_elo_rating_accuracy.py`
**Tests:** 8 tests, ~200 lines
**Coverage Target:** 60% → 75%

```python
"""
ELO Rating Accuracy Tests

Validates:
- ELO ratings match expected values for known players
- ELO rating updates after wins/losses
- ELO regression to mean
"""

class TestELORatingValues:
    def test_elite_qb_rating():
        """Patrick Mahomes should be ~1650+ (elite QB)."""
        pass

    def test_backup_qb_rating():
        """Backup QB should be ~1400-1450 (below average)."""
        pass

    def test_rookie_qb_rating():
        """Rookie QB should start at ~1500 (league average)."""
        pass

class TestELORatingUpdates:
    def test_rating_increases_after_win():
        """QB rating increases after good performance."""
        pass

    def test_rating_decreases_after_loss():
        """QB rating decreases after bad performance."""
        pass

class TestELORegression:
    def test_seasonal_decay():
        """5% decay toward 1500 each season."""
        pass

    def test_regression_to_mean():
        """Extreme ratings regress to mean over time."""
        pass
```

**ROI:** Validates ELO component accuracy (4 hours saved per incident)

---

#### Task 1.5.2: WLS Integration (3 hours)
**File:** `tests/integration/test_wls_integration.py`
**Tests:** 10 tests, ~250 lines
**Coverage Target:** 60% → 80%

```python
"""
Weighted Least Squares (WLS) Integration Tests

Validates:
- Adaptive weight calculation
- Weight persistence
- Weight rebalancing
"""

class TestAdaptiveWeights:
    def test_weight_optimization():
        """Weights optimized via CVX."""
        pass

    def test_weights_sum_to_one():
        """Constraint: sum(weights) = 1.0."""
        pass

    def test_weights_non_negative():
        """Constraint: all weights >= 0."""
        pass

class TestWeightPersistence:
    def test_save_adaptive_weights():
        """Weights saved to JSON."""
        pass

    def test_load_adaptive_weights():
        """Weights loaded from JSON."""
        pass

class TestWeightRebalancing:
    def test_rebalance_on_component_failure():
        """6-way → 5-way when component fails."""
        pass

    def test_better_models_get_higher_weights():
        """Lower MAE → higher weight."""
        pass
```

**ROI:** Validates adaptive weighting (4 hours saved per incident)

---

## Test Execution Infrastructure

### CI/CD Integration (2 hours setup)

**GitHub Actions Workflow:**
```yaml
name: P0 Critical Tests

on: [push, pull_request]

jobs:
  test-p0:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: timescale/timescaledb:latest-pg14
        env:
          POSTGRES_DB: testdb
          POSTGRES_USER: testuser
          POSTGRES_PASSWORD: testpass
        ports:
          - 5432:5432

      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-xdist

      - name: Run P0 critical tests
        run: |
          pytest tests/unit/test_data_leakage_prevention.py \
                 tests/integration/test_ensemble_v6_integration.py \
                 tests/integration/test_backtest_framework.py \
                 tests/integration/test_api_security.py \
                 tests/integration/test_batch_prediction_perf.py \
                 tests/integration/test_database_pooling.py \
                 tests/integration/test_ensemble_v5_smoke.py \
                 tests/integration/test_r_model_interface_critical.py \
                 tests/integration/test_temporal_leakage_critical.py \
                 -m critical \
                 -n auto \
                 --cov=py \
                 --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          fail_ci_if_error: true
          flags: critical
```

---

## Coverage Monitoring

### Codecov Integration (1 hour setup)

**codecov.yml:**
```yaml
coverage:
  status:
    project:
      default:
        target: 80%
        threshold: 2%

    patch:
      default:
        target: 70%
        threshold: 5%

  ignore:
    - "tests/*"
    - "scripts/*"
    - "examples/*"

comment:
  layout: "header, diff, files"
  behavior: default
```

---

## Summary

### Total Investment
- Week 1 (P0): 12 hours
- Week 2 (P1 Risk): 10 hours
- Week 3 (P1 XGBoost/BNN): 8 hours
- Week 4 (P2 Error): 6 hours
- Week 5 (P2 Performance): 5 hours
- Week 6 (P2 Property): 8 hours
- ELO (Week 1.5): 6 hours
- Infrastructure: 3 hours

**Total:** 58 hours

### Total ROI
- Week 1: 19 hours
- Week 2: 15 hours
- Week 3: 10 hours
- Week 4: 8 hours
- Week 5: 6 hours
- Week 6: 12 hours
- ELO: 8 hours
- Infrastructure: Ongoing quality assurance

**Total:** 78+ hours saved

**Net ROI:** +20 hours (34% return)

### Coverage Trajectory

| Week | Coverage | Milestone |
|------|----------|-----------|
| Current | 62-68% | Baseline |
| Week 1 | 67-72% | P0 gaps closed |
| Week 2 | 70-74% | Risk management complete |
| Week 3 | 72-76% | XGBoost/BNN complete |
| Week 4 | 74-78% | Error handling complete |
| Week 5 | 76-79% | Performance monitoring complete |
| Week 6 | 78-82% | Property-based testing complete |

**Target:** 80%+ overall, 95%+ critical path

---

**Status:** Ready for execution. Start with Week 1 P0 gaps for maximum immediate ROI.
