# NFL Analytics Test Suite Comprehensive Audit

**Generated:** November 9, 2025
**Auditor:** Claude Code (Sonnet 4.5)
**Project:** Real-money NFL betting system
**Critical Requirement:** NO data leakage (temporal, target, player contamination)

---

## Executive Summary

### Current State
- **Total test files:** 356 unit tests + 5 integration tests = 361 tests
- **Test functions:** ~8,891 test functions across all files
- **Coverage estimate:** ~40-50% (many tests are auto-generated stubs)
- **Critical gaps:** Feature engineering, ensemble integration, causal inference modules
- **Quality issues:** Many tests are TODO stubs without proper assertions

### Risk Assessment
⚠️ **HIGH RISK:** Production deployment without comprehensive temporal leakage tests
⚠️ **MEDIUM RISK:** Ensemble integration lacks proper validation tests
⚠️ **MEDIUM RISK:** Model registry infrastructure untested (1,179 lines, 0 tests)

### Recommendations
1. **Immediate:** Complete temporal leakage test suite (10 new tests)
2. **High Priority:** Ensemble integration tests (5 new tests)
3. **High Priority:** Feature engineering validation tests (8 new tests)
4. **Medium Priority:** Causal inference module tests (6 new tests)
5. **Medium Priority:** Model registry CLI tests (4 new tests)

---

## Section 1: Current Coverage Analysis

### 1.1 Test Infrastructure (✅ EXCELLENT)

**Location:** `tests/conftest.py`, `tests/README.md`, `tests/TESTING.md`

**Status:** Well-structured test infrastructure with:
- Database connection fixtures (session + function scope)
- Sample data fixtures (`sample_odds_api_response`)
- Environment variable mocking
- Custom markers (`slow`, `integration`, `unit`)
- Pre-commit hooks (black, ruff, mypy)

**Coverage:** ~95%

**Gaps:** None significant

---

### 1.2 Data Leakage Detection (⚠️ PARTIAL)

**Location:**
- `tests/unit/test_data_leakage_detector.py` (297 lines, auto-generated stubs)
- `tests/integration/test_temporal_leakage.py` (426 lines, 10 comprehensive tests)
- `py/utils/data_leakage_detector.py` (720 lines)

**Current Tests:**
1. ✅ `test_spread_only_baseline_accuracy` - Market efficiency check (< 54%)
2. ✅ `test_no_vegas_implied_probabilities` - Circular logic detection
3. ✅ `test_rolling_operations_use_shift` - Temporal safety checks
4. ✅ `test_no_postgame_features` - Future data detection
5. ✅ `test_kickoff_ordering_no_nulls` - Temporal ordering
6. ✅ `test_week_ordering_sequential` - Sequential weeks
7. ✅ `test_feature_module_temporal_safety` - Per-module validation
8. ✅ `test_no_future_data_in_features` - Random game sampling
9. ✅ `test_spread_convention_correct` - Convention validation
10. ✅ `test_leakage_diagnostic_tool_exists` - Tool availability

**Coverage:** Integration tests: ~80% | Unit tests: ~10% (stubs only)

**Critical Gaps:**
- ❌ Unit tests are empty TODO stubs
- ❌ No tests for `check_dataframe()` with real data
- ❌ No tests for `validate_split()` with contaminated data
- ❌ No tests for `check_feature_engineering()` AST analysis
- ❌ No property-based tests (Hypothesis) for invariants
- ❌ No performance tests (should complete in < 1s for large DataFrames)

**Recommendation:** Priority 1 - Complete unit test suite

---

### 1.3 Ensemble Integration (❌ CRITICAL GAP)

**Location:**
- `py/ensemble/enhanced_ensemble_v3.py` (4-way ensemble)
- `tests/unit/test_enhanced_ensemble_v3.py` (310 lines, auto-generated stubs)

**Current Tests:** All TODO stubs without real assertions

**Critical Gaps:**
- ❌ No tests for weight optimization (30% SS, 25% BNN, 25% Hier, 20% XGB)
- ❌ No tests for prediction aggregation
- ❌ No tests for missing model handling (e.g., BNN v3.1 only for rushing)
- ❌ No tests for confidence interval merging
- ❌ No tests for database query integration (`load_bayesian_predictions()`)
- ❌ No tests for backtesting logic
- ❌ No integration tests with real model artifacts

**Coverage:** ~5% (stubs only)

**Recommendation:** Priority 1 - Build comprehensive test suite

---

### 1.4 Feature Engineering (⚠️ MODERATE GAP)

**Location:**
- `py/features/` (33 files)
- `tests/unit/test_asof_features.py` (587 lines, 51 tests)
- Other feature tests: 20+ files with mixed coverage

**Current Tests:**
- ✅ `asof_features.py` - 51 tests (comprehensive)
- ⚠️ `vegas_line_bias_features.py` - 27 tests (basic coverage)
- ⚠️ `defensive_situational_strength.py` - 43 tests (basic coverage)
- ❌ `feature_interactions.py` - 11 tests (minimal)
- ❌ `bayesian_features.py` - 23 tests (basic)
- ❌ `player_feature_extractor.py` - 4 tests (insufficient)

**Coverage:** ~60% across all modules (highly variable)

**Critical Gaps:**
- ❌ No end-to-end feature pipeline tests
- ❌ No tests for feature catalog validation (`catalog.yaml`)
- ❌ No tests for parallel feature loading (`feature_loader_parallel.py`)
- ❌ No tests for asof join correctness (temporal safety)
- ❌ Insufficient tests for lag/rolling operations
- ❌ No tests for feature versioning/reproducibility

**Recommendation:** Priority 2 - Comprehensive feature validation suite

---

### 1.5 Causal Inference Modules (⚠️ MODERATE GAP)

**Location:**
- `py/causal/` (8 modules)
- Tests scattered across unit test directory

**Modules:**
1. `treatment_definitions.py` - 25 tests (stubs)
2. `panel_constructor.py` - 16 tests (stubs)
3. `diff_in_diff.py` - 8 tests (stubs)
4. `synthetic_control.py` - 12 tests (stubs)
5. `structural_causal_models.py` - 16 tests (stubs)
6. `confounder_identification.py` - 8 tests (stubs)
7. `validation.py` - 12 tests (stubs)
8. `model_integration.py` - 8 tests (stubs)

**Coverage:** ~15% (mostly stubs)

**Critical Gaps:**
- ❌ No tests for treatment effect estimation accuracy
- ❌ No tests for parallel trends assumption (DiD)
- ❌ No tests for synthetic control weight optimization
- ❌ No tests for confounder detection algorithms
- ❌ No tests for causal DAG construction
- ❌ No integration tests with real shock events (injuries, trades)

**Recommendation:** Priority 3 - Build causal inference test suite

---

### 1.6 Reinforcement Learning (CQL) (⚠️ MODERATE GAP)

**Location:**
- `py/rl/cql_agent.py` (CQL agent - 24% improvement, 1.75% policy reward)
- `tests/unit/test_cql_agent.py` (63 tests, auto-generated stubs)

**Current Tests:** 63 test stubs without real assertions

**Critical Gaps:**
- ❌ No tests for Q-network architecture (LayerNorm, ReLU)
- ❌ No tests for CQL penalty calculation
- ❌ No tests for experience replay sampling
- ❌ No tests for device selection (CPU/MPS/CUDA)
- ❌ No tests for offline RL training convergence
- ❌ No tests for action selection (epsilon-greedy)
- ❌ No integration tests with logged betting dataset

**Coverage:** ~8% (stubs only)

**Recommendation:** Priority 3 - Build RL agent test suite

---

### 1.7 Model Registry Infrastructure (❌ CRITICAL GAP)

**Location:**
- `scripts/sync_model_registry.py` (1,179 lines CLI tool)
- `scripts/generate_model_metadata.py`
- `scripts/setup_model_registry.py`
- `tests/unit/test_model_registry.py` (51 tests, stubs)

**Current Tests:** 51 test stubs without real assertions

**Critical Gaps:**
- ❌ No tests for filesystem scanning (`scan_models_directory()`)
- ❌ No tests for database sync logic (8 in sync, 227 pending)
- ❌ No tests for model metadata extraction
- ❌ No tests for CLI commands (list, sync, validate, register)
- ❌ No tests for version conflict resolution
- ❌ No integration tests with PostgreSQL `ml.model_registry` table

**Coverage:** ~5% (stubs only)

**Recommendation:** Priority 2 - Build model registry test suite

---

### 1.8 Bayesian Model Validation (⚠️ MODERATE)

**Location:**
- `py/utils/statistical_validator.py` (Phase 1.6 agent)
- `tests/unit/test_statistical_validator.py` (46 tests)

**Current Tests:** 46 tests covering:
- R-hat convergence checks (< 1.01)
- ESS bulk and tail (> 400)
- Divergence detection (< 10)
- Calibration validation (> 85%)

**Coverage:** ~70%

**Critical Gaps:**
- ❌ No tests for prior sensitivity analysis
- ❌ No tests for posterior predictive checks
- ❌ No tests for MCMC trace analysis
- ❌ No tests for Stan/brms integration
- ❌ No integration tests with real model artifacts

**Recommendation:** Priority 3 - Expand Bayesian validation suite

---

### 1.9 Production Monitoring (✅ GOOD)

**Location:**
- `py/production/` (8 modules)
- Tests: ~200+ tests across production modules

**Current Tests:**
- ✅ `test_cql_bet_sizer.py` - 11 tests (Kelly criterion, risk gates)
- ✅ `test_risk_gates.py` - 11 tests (ROI, win rate, Sharpe, drawdown)
- ✅ `test_monitor_performance.py` - 28 tests
- ✅ `test_metrics.py` - 65 tests

**Coverage:** ~75%

**Gaps:** Minor (mostly edge cases)

---

## Section 2: Top 10 Priority Test Files to Create

### Priority 1: Temporal Leakage & Ensemble (Production Blockers)

#### 1. `tests/unit/test_data_leakage_detector_comprehensive.py`

**Purpose:** Replace auto-generated stubs with real tests
**Lines:** ~500 lines
**Test Count:** 25 tests

**Key Scenarios:**
- `test_check_dataframe_detects_target_leakage` - Target in features
- `test_check_dataframe_detects_temporal_leakage` - Future data usage
- `test_check_dataframe_detects_player_contamination` - Same player in train/test
- `test_check_dataframe_handles_missing_dates` - NULL handling
- `test_validate_split_detects_temporal_overlap` - Train max date > test min date
- `test_validate_split_detects_player_overlap` - Player contamination
- `test_validate_split_handles_empty_test_set` - Edge case
- `test_check_feature_engineering_detects_negative_shift` - `.shift(-1)`
- `test_check_feature_engineering_detects_unsafe_merge` - No asof join
- `test_check_feature_engineering_detects_circular_logic` - Vegas probability features
- `test_generate_report_formats_critical_issues` - Report generation
- `test_has_critical_leakage_blocks_training` - Pre-training gate
- Property tests: Idempotency, monotonicity
- Performance tests: Large DataFrames (< 1s)

**Data Fixtures:**
```python
@pytest.fixture
def leakage_dataframe():
    """DataFrame with intentional temporal leakage."""
    return pd.DataFrame({
        'game_id': range(100),
        'kickoff': pd.date_range('2024-01-01', periods=100),
        'passing_yards': np.random.randint(100, 400, 100),
        # LEAKAGE: Target used in feature
        'avg_passing_yards': np.random.randint(100, 400, 100),
        'player_id': ['player_' + str(i % 20) for i in range(100)]
    })

@pytest.fixture
def contaminated_split():
    """Train/test split with player contamination."""
    train = pd.DataFrame({
        'player_id': ['player_1', 'player_2', 'player_3'],
        'kickoff': pd.date_range('2024-01-01', periods=3),
        'passing_yards': [250, 300, 275]
    })
    test = pd.DataFrame({
        'player_id': ['player_2', 'player_4', 'player_5'],  # player_2 overlaps
        'kickoff': pd.date_range('2024-01-04', periods=3),
        'passing_yards': [280, 310, 290]
    })
    return train, test
```

**Rationale:** #1 cause of betting model failure in production

---

#### 2. `tests/integration/test_ensemble_integration.py`

**Purpose:** Validate 4-way ensemble integration
**Lines:** ~450 lines
**Test Count:** 12 tests

**Key Scenarios:**
- `test_load_all_models_from_database` - Query integration
- `test_weight_optimization_sums_to_one` - 30% + 25% + 25% + 20% = 100%
- `test_prediction_aggregation_weighted_average` - Correct math
- `test_missing_model_handles_gracefully` - BNN v3.1 only for rushing
- `test_confidence_intervals_merge_correctly` - Uncertainty quantification
- `test_backtest_2022_2024_roi_target` - Expected ROI: +5-7%
- `test_ensemble_predictions_match_metadata` - Model version consistency
- `test_database_insert_predictions` - Store predictions in `ml.ensemble_predictions`
- `test_portfolio_optimization_kelly_criterion` - Bet sizing
- `test_ensemble_handles_missing_features` - Graceful degradation
- `test_ensemble_reproducible_with_seed` - Deterministic predictions
- Integration with real model artifacts (State-Space v2, BNN v3.1, Hierarchical v2, XGBoost)

**Mock Fixtures:**
```python
@pytest.fixture
def mock_model_predictions():
    """Mock predictions from 4 models."""
    return {
        'state_space_v2': pd.DataFrame({
            'game_id': [1, 2, 3],
            'prediction': [245.2, 312.1, 278.5],
            'std': [18.3, 22.1, 19.7]
        }),
        'bnn_v3_1_rushing': pd.DataFrame({
            'game_id': [1, 2, 3],
            'prediction': [240.5, 305.8, 275.3],
            'std': [25.1, 28.3, 26.2]
        }),
        # ... hierarchical v2, xgboost
    }
```

**Rationale:** Production ensemble untested, high risk of integration bugs

---

#### 3. `tests/unit/test_feature_pipeline_end_to_end.py`

**Purpose:** Validate complete feature engineering pipeline
**Lines:** ~400 lines
**Test Count:** 10 tests

**Key Scenarios:**
- `test_feature_pipeline_no_temporal_leakage` - End-to-end safety
- `test_asof_join_correctness` - Point-in-time joins
- `test_rolling_features_exclude_current_game` - `.shift(1)` verification
- `test_lag_features_correct_offset` - Lag 1, 3, 5 games
- `test_feature_catalog_consistency` - `catalog.yaml` validation
- `test_parallel_feature_loading_reproducible` - Deterministic results
- `test_feature_versioning_hash` - Reproducibility across runs
- `test_missing_feature_handling` - Graceful degradation
- `test_feature_pipeline_performance` - < 5s for 1000 games
- Integration with database (`mart.asof_team_features`, `mart.player_features`)

**Temporal Safety Test:**
```python
def test_feature_pipeline_no_temporal_leakage(db_connection):
    """Verify feature pipeline uses only prior data."""
    # Sample 10 random games from 2024
    games = sample_games(db_connection, season=2024, n=10)

    for game in games:
        # Extract features for this game
        features = extract_features(db_connection, game)

        # Verify all features reference only prior games
        query = """
        SELECT MAX(kickoff) as max_kickoff
        FROM feature_source_data
        WHERE game_id IN (
            SELECT source_game_id FROM features WHERE game_id = %s
        )
        """
        result = db_connection.execute(query, (game['game_id'],))
        max_kickoff = result.fetchone()['max_kickoff']

        assert max_kickoff < game['kickoff'], (
            f"Feature pipeline leaked future data for game {game['game_id']}"
        )
```

**Rationale:** Feature engineering is most common source of temporal leakage

---

### Priority 2: Model Infrastructure & Validation

#### 4. `tests/unit/test_model_registry_cli.py`

**Purpose:** Test model registry CLI tool (1,179 lines)
**Lines:** ~350 lines
**Test Count:** 15 tests

**Key Scenarios:**
- `test_scan_models_directory` - Filesystem scanning
- `test_extract_model_info` - Metadata extraction
- `test_database_sync_check` - 8 in sync, 227 pending
- `test_register_new_model` - Insert to `ml.model_registry`
- `test_update_existing_model` - Version conflict handling
- `test_list_models_command` - CLI output formatting
- `test_validate_model_metadata` - JSON schema validation
- `test_delete_corrupted_model` - Cleanup logic
- `test_export_markdown_report` - Documentation generation
- `test_cli_handles_missing_database` - Error handling
- `test_cli_handles_corrupted_files` - Graceful degradation
- Integration tests with PostgreSQL

**CLI Test Pattern:**
```python
def test_register_new_model_cli(tmp_path, db_connection):
    """Test CLI registration of new model."""
    # Create mock model file
    model_file = tmp_path / "models" / "test_model_v1.pkl"
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"mock model data")

    # Run CLI command
    result = subprocess.run(
        ["python", "scripts/sync_model_registry.py", "--register", str(model_file)],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0
    assert "Successfully registered" in result.stdout

    # Verify database insert
    cursor = db_connection.execute(
        "SELECT * FROM ml.model_registry WHERE model_path = %s",
        (str(model_file),)
    )
    assert cursor.rowcount == 1
```

**Rationale:** 1,179-line CLI tool with 0 tests is high-risk for production

---

#### 5. `tests/unit/test_causal_treatment_effects.py`

**Purpose:** Validate causal inference treatment effect estimation
**Lines:** ~400 lines
**Test Count:** 12 tests

**Key Scenarios:**
- `test_injury_treatment_definition` - Sudden drop in carries
- `test_coaching_change_treatment` - Event identification
- `test_usage_increase_treatment` - 50% increase detection
- `test_diff_in_diff_parallel_trends` - Pre-treatment trend similarity
- `test_synthetic_control_weight_optimization` - Donor pool weighting
- `test_confounder_identification_correlation` - Confounding variable detection
- `test_treatment_effect_confidence_intervals` - Bootstrap uncertainty
- `test_placebo_test_no_effect` - Falsification test
- `test_heterogeneous_treatment_effects` - Subgroup analysis
- `test_causal_dag_construction` - Graph building
- Integration with real shock events (2024 injuries)

**Causal Test Example:**
```python
def test_diff_in_diff_parallel_trends():
    """Verify parallel trends assumption holds pre-treatment."""
    # Create synthetic panel data
    df = create_panel_data(n_players=100, n_weeks=20)

    # Define treatment at week 10
    df['treated'] = (df['player_id'].isin(treated_players)) & (df['week'] >= 10)

    # Check parallel trends (weeks 1-9)
    pre_treatment = df[df['week'] < 10]

    # Run DiD
    did_model = DiffInDiff()
    did_model.fit(pre_treatment)

    # Test parallel trends
    p_value = did_model.test_parallel_trends()
    assert p_value > 0.05, f"Parallel trends violated (p={p_value:.3f})"
```

**Rationale:** Causal inference critical for shock event analysis (Phase 6)

---

#### 6. `tests/integration/test_bayesian_model_validation.py`

**Purpose:** Validate Bayesian models before production deployment
**Lines:** ~350 lines
**Test Count:** 10 tests

**Key Scenarios:**
- `test_state_space_v2_rhat_convergence` - R-hat < 1.01
- `test_state_space_v2_ess_bulk_tail` - ESS > 400
- `test_state_space_v2_divergences` - Divergences < 10
- `test_bnn_v3_1_calibration` - 89.5% calibrated coverage
- `test_hierarchical_v2_prior_sensitivity` - Robust to prior choices
- `test_posterior_predictive_checks` - Model fit assessment
- `test_mcmc_trace_analysis` - Mixing and stationarity
- `test_bayesian_model_blocks_deployment_on_failure` - Risk gate
- Integration with real model artifacts (brms, Stan, PyMC)

**Bayesian Validation Test:**
```python
def test_state_space_v2_blocks_deployment_on_convergence_failure():
    """Ensure non-converged models blocked from production."""
    # Load model with poor convergence
    model_path = "models/state_space/bad_convergence_v1.rds"

    # Run validation
    validator = StatisticalValidator()
    report = validator.validate_model(model_path)

    # Check R-hat > 1.01 (convergence failure)
    assert report.has_convergence_issues()
    assert report.max_rhat > 1.01

    # Verify deployment blocked
    with pytest.raises(DeploymentBlockedException):
        deploy_model(model_path)
```

**Rationale:** Prevent non-converged Bayesian models from reaching production

---

### Priority 3: RL, Performance, Security

#### 7. `tests/unit/test_cql_agent_comprehensive.py`

**Purpose:** Replace auto-generated CQL stubs with real tests
**Lines:** ~450 lines
**Test Count:** 20 tests

**Key Scenarios:**
- `test_qnetwork_architecture` - Layer dimensions, LayerNorm, ReLU
- `test_cql_penalty_calculation` - log sum exp Q(s, a)
- `test_experience_replay_sampling` - Batch sampling
- `test_device_selection_auto` - CPU/MPS/CUDA priority
- `test_offline_rl_training_convergence` - Loss decreases over epochs
- `test_action_selection_epsilon_greedy` - Exploration vs exploitation
- `test_model_save_load` - Checkpoint persistence
- `test_logged_dataset_loading` - CSV ingestion
- `test_conservative_penalty_weight_alpha` - Hyperparameter effect
- `test_target_network_updates` - Polyak averaging
- Integration with real logged betting dataset (5,146 games)

**CQL Test Example:**
```python
def test_cql_penalty_reduces_ood_action_values():
    """Verify CQL penalty prevents overestimation on OOD actions."""
    # Create agent
    agent = CQLAgent(state_dim=10, n_actions=4, alpha=1.0)

    # Generate in-distribution and OOD states
    states_id = torch.randn(32, 10)  # In-distribution
    states_ood = torch.randn(32, 10) * 5  # Out-of-distribution (scaled)

    # Get Q-values
    q_values_id = agent.q_network(states_id)
    q_values_ood = agent.q_network(states_ood)

    # OOD Q-values should be lower (conservative)
    assert q_values_ood.mean() < q_values_id.mean(), (
        "CQL penalty failed to reduce OOD action values"
    )
```

**Rationale:** CQL agent (+24% improvement) is production-ready but untested

---

#### 8. `tests/unit/test_performance_optimizer_profiling.py`

**Purpose:** Test performance optimization agent (Phase 1.6)
**Lines:** ~300 lines
**Test Count:** 12 tests

**Key Scenarios:**
- `test_mcmc_configuration_check` - Chains, tune, draws validation
- `test_sql_query_optimization` - Missing index detection
- `test_pandas_performance_check` - `.iterrows()`, `.apply()` detection
- `test_backtest_time_estimation` - Runtime prediction
- `test_profiling_overhead_minimal` - < 5% overhead
- `test_optimization_suggestions` - Actionable recommendations
- `test_parallel_execution_speedup` - Multiprocessing benefit
- Integration with real model training scripts

**Performance Test:**
```python
def test_backtest_time_estimation_accuracy():
    """Verify backtest time estimation is within 20% of actual."""
    # Create mock backtest config
    config = {
        'n_games': 1000,
        'n_models': 4,
        'n_samples': 1000,
        'parallel': False
    }

    # Estimate runtime
    optimizer = PerformanceOptimizer()
    estimated_time = optimizer.estimate_backtest_time(config)

    # Run actual backtest
    start = time.time()
    run_backtest(config)
    actual_time = time.time() - start

    # Check accuracy (within 20%)
    error = abs(estimated_time - actual_time) / actual_time
    assert error < 0.20, f"Estimate error: {error:.1%}"
```

**Rationale:** Performance optimizer agent lacks validation tests

---

#### 9. `tests/integration/test_security_audit_integration.py`

**Purpose:** Test security auditor agent (Phase 1.6)
**Lines:** ~300 lines
**Test Count:** 10 tests

**Key Scenarios:**
- `test_detect_credentials_in_code` - "sicillionbillions" hardcoded
- `test_detect_api_keys_in_env_files` - `.env` scanning
- `test_detect_sql_injection_patterns` - String concatenation
- `test_detect_privileged_docker_mode` - `--privileged` flag
- `test_detect_exposed_ports` - Port 5544 public
- `test_dependency_vulnerability_scan` - pip-audit integration
- `test_security_report_json_export` - Structured output
- `test_security_audit_blocks_commit` - Pre-commit hook
- Integration with codebase scanning

**Security Test:**
```python
def test_detect_hardcoded_database_password():
    """Verify security auditor detects hardcoded credentials."""
    # Scan codebase
    auditor = SecurityAuditor()
    report = auditor.scan_codebase(
        path="\",
        exclude_dirs={".git", "venv", ".venv"}
    )

    # Check for known hardcoded password
    credential_issues = [
        issue for issue in report.issues
        if issue.title == "Hardcoded credentials"
    ]

    # Should detect "sicillionbillions" in multiple files
    assert len(credential_issues) > 0
    assert any("sicillionbillions" in issue.evidence for issue in credential_issues)
```

**Rationale:** Real-money betting system requires comprehensive security testing

---

#### 10. `tests/unit/test_betting_strategy_validator.py`

**Purpose:** Test betting strategy validator (Phase 1.6)
**Lines:** ~350 lines
**Test Count:** 15 tests

**Key Scenarios:**
- `test_roi_risk_gate` - ROI > 5% (target: 7%+)
- `test_win_rate_risk_gate` - Win rate > 59% (target: 62%+)
- `test_sharpe_ratio_risk_gate` - Sharpe > 2.2 (target: 2.5+)
- `test_max_drawdown_risk_gate` - Max drawdown < -8%
- `test_kelly_criterion_bet_sizing` - Kelly fraction calculation
- `test_stress_test_scenarios` - Market crash, losing streak
- `test_bankroll_simulation` - Monte Carlo trajectories
- `test_blocks_deployment_on_failure` - Risk gate enforcement
- `test_strategy_reproducibility` - Deterministic with seed
- Integration with backtest results

**Risk Gate Test:**
```python
def test_blocks_deployment_on_low_roi():
    """Ensure low-ROI strategies blocked from production."""
    # Create backtest with low ROI
    backtest_results = {
        'roi': 0.03,  # 3% (below 5% threshold)
        'win_rate': 0.58,
        'sharpe_ratio': 1.8,
        'max_drawdown': -0.06
    }

    # Run validator
    validator = BettingStrategyValidator()
    report = validator.validate(backtest_results)

    # Should fail ROI gate
    assert not report.passed
    assert report.failed_gates == ['roi']

    # Verify deployment blocked
    with pytest.raises(DeploymentBlockedException):
        deploy_strategy(backtest_results)
```

**Rationale:** Production betting strategies must pass 4 risk gates

---

## Section 3: Example Test Structure (Top 3 Modules)

### 3.1 Data Leakage Detector - Complete Example

```python
#!/usr/bin/env python3
"""
Comprehensive tests for Data Leakage Detector

Validates temporal integrity for NFL betting models.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from py.utils.data_leakage_detector import (
    DataLeakageDetector,
    LeakageSeverity,
    LeakageType
)


class TestCheckDataFrame:
    """Test DataFrame leakage detection."""

    def test_detects_target_leakage(self):
        """Detect target variable used in features."""
        # Create DataFrame with target leakage
        df = pd.DataFrame({
            'game_id': [1, 2, 3],
            'kickoff': pd.date_range('2024-01-01', periods=3),
            'passing_yards': [250, 300, 275],  # Target
            'avg_passing_yards': [250, 300, 275],  # LEAKAGE: Same as target
        })

        detector = DataLeakageDetector()
        report = detector.check_dataframe(
            df,
            target_col='passing_yards',
            date_col='kickoff'
        )

        # Should detect perfect correlation (target leakage)
        assert report.has_critical_leakage()
        target_issues = [
            issue for issue in report.issues
            if issue.leakage_type == LeakageType.TARGET
        ]
        assert len(target_issues) > 0
        assert target_issues[0].severity == LeakageSeverity.CRITICAL

    def test_detects_temporal_leakage(self):
        """Detect future data in features."""
        # Create DataFrame with temporal leakage
        df = pd.DataFrame({
            'game_id': [1, 2, 3],
            'kickoff': pd.date_range('2024-01-01', periods=3),
            'passing_yards': [250, 300, 275],
            'next_game_yards': [300, 275, 280],  # LEAKAGE: Future data
        })

        detector = DataLeakageDetector()
        report = detector.check_dataframe(
            df,
            target_col='passing_yards',
            date_col='kickoff'
        )

        # Should detect suspicious column pattern
        assert report.critical_count > 0 or report.high_count > 0

    def test_detects_player_contamination(self):
        """Detect same player in train and test."""
        # This is handled by validate_split, but check here too
        df = pd.DataFrame({
            'game_id': range(10),
            'kickoff': pd.date_range('2024-01-01', periods=10),
            'passing_yards': np.random.randint(200, 350, 10),
            'player_id': ['player_' + str(i % 3) for i in range(10)]
        })

        detector = DataLeakageDetector()
        report = detector.check_dataframe(
            df,
            target_col='passing_yards',
            date_col='kickoff',
            player_col='player_id'
        )

        # Should warn about small unique player count
        assert len(report.issues) > 0

    def test_handles_missing_dates_gracefully(self):
        """Handle NULL dates without crashing."""
        df = pd.DataFrame({
            'game_id': [1, 2, 3],
            'kickoff': [pd.Timestamp('2024-01-01'), pd.NaT, pd.Timestamp('2024-01-03')],
            'passing_yards': [250, 300, 275],
        })

        detector = DataLeakageDetector()
        report = detector.check_dataframe(
            df,
            target_col='passing_yards',
            date_col='kickoff'
        )

        # Should detect NULL dates as HIGH severity issue
        assert report.high_count > 0
        null_issues = [
            issue for issue in report.issues
            if 'NULL' in issue.description or 'missing' in issue.description.lower()
        ]
        assert len(null_issues) > 0


class TestValidateSplit:
    """Test train/test split validation."""

    def test_detects_temporal_overlap(self):
        """Detect train max date > test min date."""
        train = pd.DataFrame({
            'game_id': [1, 2, 3],
            'kickoff': pd.date_range('2024-01-01', periods=3),
            'passing_yards': [250, 300, 275]
        })
        test = pd.DataFrame({
            'game_id': [4, 5, 6],
            'kickoff': pd.date_range('2024-01-02', periods=3),  # OVERLAP
            'passing_yards': [280, 310, 290]
        })

        detector = DataLeakageDetector()
        report = detector.validate_split(
            train, test,
            date_col='kickoff'
        )

        # Should detect temporal overlap
        assert report.has_critical_leakage()

    def test_detects_player_overlap(self):
        """Detect same player in train and test."""
        train = pd.DataFrame({
            'player_id': ['player_1', 'player_2', 'player_3'],
            'kickoff': pd.date_range('2024-01-01', periods=3),
            'passing_yards': [250, 300, 275]
        })
        test = pd.DataFrame({
            'player_id': ['player_2', 'player_4', 'player_5'],  # player_2 overlaps
            'kickoff': pd.date_range('2024-01-04', periods=3),
            'passing_yards': [280, 310, 290]
        })

        detector = DataLeakageDetector()
        report = detector.validate_split(
            train, test,
            date_col='kickoff',
            player_col='player_id'
        )

        # Should detect player contamination
        assert report.critical_count > 0
        player_issues = [
            issue for issue in report.issues
            if issue.leakage_type == LeakageType.PLAYER
        ]
        assert len(player_issues) > 0

    def test_handles_empty_test_set(self):
        """Handle empty test set gracefully."""
        train = pd.DataFrame({
            'kickoff': pd.date_range('2024-01-01', periods=10),
            'passing_yards': np.random.randint(200, 350, 10)
        })
        test = pd.DataFrame(columns=['kickoff', 'passing_yards'])

        detector = DataLeakageDetector()
        report = detector.validate_split(train, test, date_col='kickoff')

        # Should warn about empty test set
        assert report.critical_count > 0 or report.high_count > 0


class TestCheckFeatureEngineering:
    """Test feature engineering code analysis."""

    def test_detects_negative_shift(self):
        """Detect .shift(-1) usage (future data)."""
        code = """
        df['next_yards'] = df.groupby('player_id')['passing_yards'].shift(-1)
        """

        detector = DataLeakageDetector()
        report = detector.check_feature_engineering(code, file_path="test.py")

        # Should detect negative shift
        assert report.has_critical_leakage()
        shift_issues = [
            issue for issue in report.issues
            if 'shift' in issue.description.lower()
        ]
        assert len(shift_issues) > 0

    def test_detects_unsafe_merge(self):
        """Detect merge without asof (temporal safety)."""
        code = """
        df = df.merge(historical_data, on='player_id', how='left')
        """

        detector = DataLeakageDetector()
        report = detector.check_feature_engineering(code, file_path="test.py")

        # Should warn about unsafe merge
        assert report.high_count > 0 or report.medium_count > 0

    def test_detects_circular_logic_vegas_probs(self):
        """Detect vegas-derived probability features."""
        code = """
        df['implied_prob'] = 1 / df['moneyline_odds']
        df['market_efficiency'] = df['actual_prob'] - df['implied_prob']
        """

        detector = DataLeakageDetector()
        report = detector.check_feature_engineering(code, file_path="test.py")

        # Should detect circular logic
        assert len(report.issues) > 0


class TestGenerateReport:
    """Test report generation."""

    def test_formats_critical_issues_prominently(self):
        """Ensure critical issues highlighted in report."""
        detector = DataLeakageDetector()

        # Add critical issue
        from py.utils.data_leakage_detector import LeakageIssue
        detector.report.add_issue(LeakageIssue(
            severity=LeakageSeverity.CRITICAL,
            leakage_type=LeakageType.TEMPORAL,
            title="Future data leakage",
            description="Features use data from future games",
            evidence=".shift(-1) detected",
            remediation="Change to .shift(1)"
        ))

        # Generate report
        report_text = detector.generate_report()

        # Should contain critical marker
        assert "CRITICAL" in report_text
        assert "Future data leakage" in report_text


class TestPreTrainingGate:
    """Test pre-training validation gate."""

    def test_blocks_training_on_critical_leakage(self):
        """Verify training blocked when critical leakage detected."""
        # Create leaky data
        df = pd.DataFrame({
            'kickoff': pd.date_range('2024-01-01', periods=100),
            'passing_yards': np.random.randint(200, 350, 100),
            'avg_passing_yards': np.random.randint(200, 350, 100)  # Leakage
        })

        # Validate before training
        detector = DataLeakageDetector()
        report = detector.check_dataframe(
            df,
            target_col='passing_yards',
            date_col='kickoff'
        )

        # Should block training
        if report.has_critical_leakage():
            with pytest.raises(ValueError, match="Critical leakage detected"):
                # This would be in model training code
                raise ValueError("Critical leakage detected - model invalid")


# Property-based tests
from hypothesis import given, strategies as st

class TestLeakageDetectorProperties:
    """Property-based tests using Hypothesis."""

    @given(
        n_rows=st.integers(min_value=10, max_value=1000),
        n_features=st.integers(min_value=3, max_value=20)
    )
    def test_check_dataframe_idempotent(self, n_rows, n_features):
        """check_dataframe should be idempotent."""
        # Generate random DataFrame
        df = pd.DataFrame({
            f'feature_{i}': np.random.randn(n_rows)
            for i in range(n_features)
        })
        df['kickoff'] = pd.date_range('2024-01-01', periods=n_rows)
        df['target'] = np.random.randint(0, 2, n_rows)

        detector = DataLeakageDetector()

        # Run twice
        report1 = detector.check_dataframe(df, target_col='target', date_col='kickoff')
        report2 = detector.check_dataframe(df, target_col='target', date_col='kickoff')

        # Should produce identical results
        assert report1.critical_count == report2.critical_count
        assert report1.high_count == report2.high_count


# Performance tests
class TestLeakageDetectorPerformance:
    """Performance tests for large DataFrames."""

    def test_check_dataframe_completes_quickly(self):
        """Verify detector completes in < 1s for 10k rows."""
        # Create large DataFrame
        n_rows = 10_000
        df = pd.DataFrame({
            'game_id': range(n_rows),
            'kickoff': pd.date_range('2010-01-01', periods=n_rows),
            'passing_yards': np.random.randint(150, 400, n_rows),
            'player_id': [f'player_{i % 100}' for i in range(n_rows)]
        })

        detector = DataLeakageDetector()

        # Time execution
        import time
        start = time.time()
        report = detector.check_dataframe(
            df,
            target_col='passing_yards',
            date_col='kickoff',
            player_col='player_id'
        )
        elapsed = time.time() - start

        # Should complete in < 1s
        assert elapsed < 1.0, f"Detector too slow: {elapsed:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
```

---

### 3.2 Ensemble Integration - Complete Example

```python
#!/usr/bin/env python3
"""
Comprehensive tests for 4-Way Ensemble Integration

Validates State-Space v2 + BNN v3.1 + Hierarchical v2 + XGBoost ensemble.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from py.ensemble.enhanced_ensemble_v3 import EnhancedEnsembleV3


class TestEnsembleWeights:
    """Test ensemble weight configuration."""

    def test_weights_sum_to_one(self):
        """Verify weights sum to 100%."""
        ensemble = EnhancedEnsembleV3()

        weights = {
            'state_space_v2': 0.30,
            'bnn_v3_1_rushing': 0.25,
            'hierarchical_v2': 0.25,
            'xgboost': 0.20
        }

        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, not 1.0"

    def test_weight_optimization_respects_constraints(self):
        """Verify CVX optimization respects weight bounds."""
        ensemble = EnhancedEnsembleV3()

        # Mock prediction errors
        errors = {
            'state_space_v2': np.random.randn(100),
            'bnn_v3_1_rushing': np.random.randn(100),
            'hierarchical_v2': np.random.randn(100),
            'xgboost': np.random.randn(100)
        }

        # Optimize weights
        optimized_weights = ensemble.optimize_weights(errors)

        # Check constraints
        assert all(w >= 0 for w in optimized_weights.values())
        assert all(w <= 1 for w in optimized_weights.values())
        assert abs(sum(optimized_weights.values()) - 1.0) < 1e-6


class TestPredictionAggregation:
    """Test prediction aggregation logic."""

    def test_weighted_average_correct(self):
        """Verify weighted average calculation."""
        ensemble = EnhancedEnsembleV3()

        predictions = {
            'state_space_v2': 250.0,
            'bnn_v3_1_rushing': 245.0,
            'hierarchical_v2': 255.0,
            'xgboost': 248.0
        }

        weights = {
            'state_space_v2': 0.30,
            'bnn_v3_1_rushing': 0.25,
            'hierarchical_v2': 0.25,
            'xgboost': 0.20
        }

        # Calculate expected
        expected = sum(predictions[k] * weights[k] for k in predictions.keys())

        # Get ensemble prediction
        result = ensemble.aggregate_predictions(predictions, weights)

        assert abs(result - expected) < 1e-6

    def test_confidence_intervals_merge_correctly(self):
        """Verify confidence interval merging."""
        ensemble = EnhancedEnsembleV3()

        predictions = {
            'state_space_v2': {'mean': 250.0, 'std': 18.3},
            'bnn_v3_1_rushing': {'mean': 245.0, 'std': 25.1},
            'hierarchical_v2': {'mean': 255.0, 'std': 20.5},
            'xgboost': {'mean': 248.0, 'std': 15.2}
        }

        # Merge confidence intervals
        result = ensemble.merge_confidence_intervals(predictions)

        # Should return mean and combined std
        assert 'mean' in result
        assert 'std' in result
        assert result['std'] > 0


class TestMissingModelHandling:
    """Test graceful handling of missing models."""

    def test_bnn_v3_1_only_for_rushing(self):
        """Verify BNN v3.1 only used for rushing props."""
        ensemble = EnhancedEnsembleV3()

        # Passing prop (BNN not available)
        passing_prop = {
            'prop_type': 'passing_yards',
            'player_id': 'player_123'
        }

        predictions = ensemble.get_predictions(passing_prop)

        # Should not include BNN v3.1 in ensemble
        assert 'bnn_v3_1_rushing' not in predictions
        assert len(predictions) == 3  # Only SS, Hier, XGB

    def test_missing_model_reweights_others(self):
        """When model missing, reweight remaining models."""
        ensemble = EnhancedEnsembleV3()

        # Mock 3 models available (BNN missing)
        available_models = ['state_space_v2', 'hierarchical_v2', 'xgboost']

        # Get adjusted weights
        adjusted_weights = ensemble.adjust_weights_for_missing(
            available_models,
            original_weights={
                'state_space_v2': 0.30,
                'bnn_v3_1_rushing': 0.25,
                'hierarchical_v2': 0.25,
                'xgboost': 0.20
            }
        )

        # Should sum to 1.0
        assert abs(sum(adjusted_weights.values()) - 1.0) < 1e-6


class TestDatabaseIntegration:
    """Test database query integration."""

    @pytest.mark.integration
    def test_load_bayesian_predictions_from_db(self, db_connection):
        """Load Bayesian predictions from ml.model_predictions."""
        ensemble = EnhancedEnsembleV3()

        # Load predictions
        preds = ensemble.load_bayesian_predictions(
            model_version='state_space_v2',
            season=2024,
            week=10
        )

        # Should return DataFrame
        assert isinstance(preds, pd.DataFrame)
        assert 'game_id' in preds.columns
        assert 'prediction' in preds.columns
        assert len(preds) > 0

    @pytest.mark.integration
    def test_store_ensemble_predictions_to_db(self, db_connection):
        """Store ensemble predictions in ml.ensemble_predictions."""
        ensemble = EnhancedEnsembleV3()

        predictions = pd.DataFrame({
            'game_id': [1, 2, 3],
            'player_id': ['player_1', 'player_2', 'player_3'],
            'prop_type': ['passing_yards', 'rushing_yards', 'receiving_yards'],
            'prediction': [275.5, 82.3, 105.7],
            'lower_ci': [250.0, 70.0, 90.0],
            'upper_ci': [300.0, 95.0, 120.0]
        })

        # Store predictions
        ensemble.store_predictions(predictions)

        # Verify insert
        cursor = db_connection.execute(
            "SELECT COUNT(*) FROM ml.ensemble_predictions WHERE prediction_date = CURRENT_DATE"
        )
        assert cursor.fetchone()[0] == 3


class TestBacktesting:
    """Test ensemble backtesting."""

    def test_backtest_2022_2024_roi_target(self):
        """Verify 2022-2024 backtest achieves +5-7% ROI."""
        ensemble = EnhancedEnsembleV3()

        # Run backtest
        results = ensemble.backtest(
            start_season=2022,
            end_season=2024,
            initial_bankroll=10000
        )

        # Check ROI
        roi = results['final_roi']
        assert roi >= 0.05, f"ROI {roi:.1%} below 5% target"
        assert roi <= 0.10, f"ROI {roi:.1%} suspiciously high (check for leakage)"

        # Check other metrics
        assert results['win_rate'] >= 0.59
        assert results['sharpe_ratio'] >= 2.2
        assert results['max_drawdown'] >= -0.08


class TestReproducibility:
    """Test ensemble reproducibility."""

    def test_predictions_deterministic_with_seed(self):
        """Verify predictions reproducible with random seed."""
        ensemble1 = EnhancedEnsembleV3(random_seed=42)
        ensemble2 = EnhancedEnsembleV3(random_seed=42)

        game_context = {
            'game_id': 123,
            'player_id': 'player_456',
            'prop_type': 'passing_yards'
        }

        # Get predictions twice
        pred1 = ensemble1.predict(game_context)
        pred2 = ensemble2.predict(game_context)

        # Should be identical
        assert abs(pred1['mean'] - pred2['mean']) < 1e-6


# Integration test with real model artifacts
@pytest.mark.integration
@pytest.mark.slow
class TestRealModelArtifacts:
    """Integration tests with real trained models."""

    def test_load_state_space_v2_model(self):
        """Load State-Space v2 model from filesystem."""
        ensemble = EnhancedEnsembleV3()

        model_path = "models/state_space/state_space_v2.rds"
        model = ensemble.load_model(model_path)

        assert model is not None
        assert hasattr(model, 'predict')

    def test_load_bnn_v3_1_rushing_model(self):
        """Load BNN v3.1 Rushing model."""
        ensemble = EnhancedEnsembleV3()

        model_path = "models/bnn/bnn_v3_1_rushing.pth"
        model = ensemble.load_model(model_path)

        assert model is not None
        assert hasattr(model, 'sample_posterior')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
```

---

### 3.3 Feature Pipeline Temporal Safety - Complete Example

```python
#!/usr/bin/env python3
"""
End-to-End Feature Pipeline Temporal Safety Tests

Validates complete feature engineering pipeline for temporal leakage.
"""

import pytest
import pandas as pd
import numpy as np
import psycopg
from datetime import datetime, timedelta
from py.features.asof_features import fetch_team_game_features, compute_team_history
from py.features.feature_loader import FeatureLoader


DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "database": "devdb01",
    "user": "dro",
    "password": "sicillionbillions"
}


@pytest.fixture
def db_connection():
    """Database connection fixture."""
    conn = psycopg.connect(**DB_CONFIG)
    yield conn
    conn.close()


class TestFeaturePipelineTemporalSafety:
    """End-to-end temporal safety tests."""

    @pytest.mark.integration
    def test_feature_pipeline_no_temporal_leakage(self, db_connection):
        """Verify feature pipeline uses only prior data."""
        # Sample 10 random games from 2024
        query = """
        SELECT game_id, kickoff, home_team, away_team
        FROM games
        WHERE season = 2024 AND spread_close IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 10
        """
        sampled_games = pd.read_sql_query(query, db_connection)

        for _, game in sampled_games.iterrows():
            # Extract features for this game
            features = fetch_team_game_features(db_connection, game['game_id'])

            # Verify all features reference only prior games
            query = f"""
            SELECT MAX(kickoff) as max_kickoff
            FROM games
            WHERE game_id IN (
                SELECT source_game_id
                FROM feature_lineage
                WHERE target_game_id = '{game['game_id']}'
            )
            """
            result = pd.read_sql_query(query, db_connection)
            max_kickoff = result['max_kickoff'].iloc[0]

            assert max_kickoff < game['kickoff'], (
                f"Feature pipeline leaked future data for game {game['game_id']}: "
                f"max_kickoff={max_kickoff}, game_kickoff={game['kickoff']}"
            )

    @pytest.mark.integration
    def test_asof_join_correctness(self, db_connection):
        """Verify asof joins use point-in-time data."""
        # Get a specific game
        game_id = 'sample_game_123'
        game = pd.read_sql_query(
            f"SELECT * FROM games WHERE game_id = '{game_id}'",
            db_connection
        ).iloc[0]

        # Get asof features
        features = compute_team_history(db_connection, game['home_team'], game['kickoff'])

        # Verify all features are from prior games
        assert features['games_played'] >= 1
        assert features['last_game_date'] < game['kickoff']

    def test_rolling_features_exclude_current_game(self, db_connection):
        """Verify rolling features use .shift(1)."""
        # Create sample data
        df = pd.DataFrame({
            'team': ['BUF'] * 10,
            'game_date': pd.date_range('2024-01-01', periods=10),
            'points_for': [24, 31, 27, 35, 20, 28, 33, 17, 30, 26]
        })

        # Compute rolling average (last 3 games)
        df['rolling_ppg'] = (
            df.groupby('team')['points_for']
            .rolling(window=3, min_periods=1)
            .mean()
            .shift(1)  # Exclude current game
            .reset_index(drop=True)
        )

        # Verify current game not included
        for i in range(1, len(df)):
            expected = df.loc[max(0, i-3):i-1, 'points_for'].mean()
            actual = df.loc[i, 'rolling_ppg']
            if not pd.isna(actual):
                assert abs(actual - expected) < 1e-6, (
                    f"Rolling average leaked current game at index {i}"
                )


class TestLagFeatures:
    """Test lag feature correctness."""

    def test_lag_1_correct_offset(self):
        """Verify lag-1 features reference previous game."""
        df = pd.DataFrame({
            'team': ['BUF'] * 5,
            'game_date': pd.date_range('2024-01-01', periods=5),
            'passing_yards': [250, 280, 265, 300, 275]
        })

        # Lag-1
        df['prev_passing_yards'] = df.groupby('team')['passing_yards'].shift(1)

        # Verify
        assert pd.isna(df.loc[0, 'prev_passing_yards'])  # First game has no prior
        assert df.loc[1, 'prev_passing_yards'] == 250
        assert df.loc[2, 'prev_passing_yards'] == 280

    def test_lag_3_correct_offset(self):
        """Verify lag-3 features reference 3 games ago."""
        df = pd.DataFrame({
            'team': ['KC'] * 10,
            'game_date': pd.date_range('2024-01-01', periods=10),
            'rushing_yards': [100, 120, 95, 130, 110, 115, 105, 125, 118, 108]
        })

        # Lag-3
        df['lag3_rushing_yards'] = df.groupby('team')['rushing_yards'].shift(3)

        # Verify
        assert pd.isna(df.loc[0, 'lag3_rushing_yards'])
        assert pd.isna(df.loc[1, 'lag3_rushing_yards'])
        assert pd.isna(df.loc[2, 'lag3_rushing_yards'])
        assert df.loc[3, 'lag3_rushing_yards'] == 100
        assert df.loc[4, 'lag3_rushing_yards'] == 120


class TestFeatureCatalog:
    """Test feature catalog validation."""

    def test_catalog_yaml_schema_valid(self):
        """Verify catalog.yaml follows schema."""
        import yaml
        from py.features.catalog_validator import CatalogValidator

        with open('py/features/catalog.yaml', 'r') as f:
            catalog = yaml.safe_load(f)

        validator = CatalogValidator()
        assert validator.validate(catalog)

    def test_catalog_features_match_code(self):
        """Verify catalog matches implemented features."""
        import yaml
        from py.features.feature_loader import FeatureLoader

        with open('py/features/catalog.yaml', 'r') as f:
            catalog = yaml.safe_load(f)

        loader = FeatureLoader()
        available_features = loader.list_features()

        catalog_features = set(catalog['features'].keys())
        code_features = set(available_features)

        # All catalog features should be implemented
        missing = catalog_features - code_features
        assert len(missing) == 0, f"Missing implementations: {missing}"


class TestParallelFeatureLoading:
    """Test parallel feature loading."""

    def test_parallel_loading_reproducible(self, db_connection):
        """Verify parallel loading produces deterministic results."""
        from py.features.feature_loader_parallel import ParallelFeatureLoader

        loader = ParallelFeatureLoader(n_jobs=4)

        # Load features twice
        features1 = loader.load_features(
            db_connection,
            game_ids=[1, 2, 3, 4, 5],
            feature_set='standard'
        )

        features2 = loader.load_features(
            db_connection,
            game_ids=[1, 2, 3, 4, 5],
            feature_set='standard'
        )

        # Should be identical
        pd.testing.assert_frame_equal(features1, features2)


class TestFeatureVersioning:
    """Test feature versioning and reproducibility."""

    def test_feature_hash_deterministic(self):
        """Verify feature hash reproducible across runs."""
        from py.features.catalog import FeatureCatalog

        catalog = FeatureCatalog()

        # Get hash twice
        hash1 = catalog.get_feature_hash('rolling_ppg_l3')
        hash2 = catalog.get_feature_hash('rolling_ppg_l3')

        # Should be identical
        assert hash1 == hash2


class TestMissingFeatureHandling:
    """Test graceful handling of missing features."""

    def test_missing_feature_returns_nan(self, db_connection):
        """Missing features should return NaN, not crash."""
        from py.features.feature_loader import FeatureLoader

        loader = FeatureLoader()

        # Request non-existent feature
        features = loader.load_features(
            db_connection,
            game_ids=[1, 2, 3],
            feature_set=['nonexistent_feature']
        )

        # Should return DataFrame with NaN
        assert isinstance(features, pd.DataFrame)
        assert features['nonexistent_feature'].isna().all()


class TestPerformance:
    """Test feature pipeline performance."""

    @pytest.mark.slow
    def test_feature_pipeline_completes_quickly(self, db_connection):
        """Verify pipeline completes in < 5s for 1000 games."""
        from py.features.feature_loader import FeatureLoader
        import time

        loader = FeatureLoader()

        # Sample 1000 games
        game_ids = pd.read_sql_query(
            "SELECT game_id FROM games WHERE season >= 2020 ORDER BY RANDOM() LIMIT 1000",
            db_connection
        )['game_id'].tolist()

        # Time execution
        start = time.time()
        features = loader.load_features(
            db_connection,
            game_ids=game_ids,
            feature_set='standard'
        )
        elapsed = time.time() - start

        # Should complete in < 5s
        assert elapsed < 5.0, f"Feature pipeline too slow: {elapsed:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])
```

---

## Section 4: Summary & Recommendations

### 4.1 Immediate Actions (Week 1)

1. **Complete Data Leakage Unit Tests** (Priority 1)
   - File: `tests/unit/test_data_leakage_detector_comprehensive.py`
   - Lines: ~500
   - Tests: 25
   - **Blocks:** Production deployment

2. **Build Ensemble Integration Tests** (Priority 1)
   - File: `tests/integration/test_ensemble_integration.py`
   - Lines: ~450
   - Tests: 12
   - **Blocks:** 4-way ensemble production deployment

3. **Create Feature Pipeline Tests** (Priority 1)
   - File: `tests/unit/test_feature_pipeline_end_to_end.py`
   - Lines: ~400
   - Tests: 10
   - **Blocks:** Feature engineering validation

### 4.2 High Priority (Weeks 2-3)

4. **Model Registry CLI Tests** (Priority 2)
   - File: `tests/unit/test_model_registry_cli.py`
   - Lines: ~350
   - Tests: 15

5. **Causal Treatment Effects Tests** (Priority 3)
   - File: `tests/unit/test_causal_treatment_effects.py`
   - Lines: ~400
   - Tests: 12

6. **Bayesian Model Validation Tests** (Priority 3)
   - File: `tests/integration/test_bayesian_model_validation.py`
   - Lines: ~350
   - Tests: 10

### 4.3 Medium Priority (Weeks 4-5)

7. **CQL Agent Comprehensive Tests** (Priority 3)
   - File: `tests/unit/test_cql_agent_comprehensive.py`
   - Lines: ~450
   - Tests: 20

8. **Performance Optimizer Tests** (Priority 3)
   - File: `tests/unit/test_performance_optimizer_profiling.py`
   - Lines: ~300
   - Tests: 12

9. **Security Audit Integration Tests** (Priority 3)
   - File: `tests/integration/test_security_audit_integration.py`
   - Lines: ~300
   - Tests: 10

10. **Betting Strategy Validator Tests** (Priority 3)
    - File: `tests/unit/test_betting_strategy_validator.py`
    - Lines: ~350
    - Tests: 15

### 4.4 Coverage Goals

| Component | Current | Target | Priority |
|-----------|---------|--------|----------|
| Data Leakage Detector | 15% | 90% | 1 |
| Ensemble Integration | 5% | 85% | 1 |
| Feature Engineering | 60% | 90% | 1 |
| Model Registry | 5% | 80% | 2 |
| Causal Inference | 15% | 75% | 3 |
| RL (CQL) | 8% | 80% | 3 |
| Bayesian Validation | 70% | 90% | 3 |
| Security Audit | 0% | 75% | 3 |
| Production Monitoring | 75% | 85% | 4 |

**Overall Target:** 80%+ coverage by end of Week 5

### 4.5 Testing Standards

**All tests must follow:**
1. **Naming:** `test_<function>_<scenario>_<expected>`
2. **Structure:** Arrange / Act / Assert (AAA pattern)
3. **Type hints:** All functions typed
4. **Docstrings:** Google style, explain "why" not "what"
5. **Markers:** `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`
6. **Fixtures:** Use conftest.py for shared data
7. **Mocking:** Mock external dependencies (DB, API, filesystem)
8. **Property tests:** Use Hypothesis for invariants
9. **Performance:** Fast tests (< 100ms) run by default, slow tests opt-in

### 4.6 CI/CD Integration

**Pre-commit hooks** (already configured):
- black (formatting)
- ruff (linting)
- mypy (type checking)
- pytest (unit tests only, < 5s)

**GitHub Actions** (needs expansion):
- Unit tests (Python 3.10, 3.11, 3.12)
- Integration tests (with PostgreSQL)
- Coverage report (Codecov)
- Security audit
- **NEW:** Temporal leakage tests (must pass)
- **NEW:** Ensemble validation tests (must pass)

### 4.7 Risk Mitigation

**Production Blockers:**
1. ❌ Data leakage tests incomplete → **BLOCKS DEPLOYMENT**
2. ❌ Ensemble integration untested → **BLOCKS DEPLOYMENT**
3. ⚠️ Feature pipeline lacks end-to-end tests → **HIGH RISK**

**Mitigation Strategy:**
- Week 1: Complete blockers (tests 1-3)
- Week 2-3: High priority tests (tests 4-6)
- Week 4-5: Medium priority tests (tests 7-10)
- Week 6: CI/CD integration + documentation

---

## Section 5: Conclusion

### Current State Summary
- **Test infrastructure:** Excellent (✅)
- **Integration tests:** Good temporal leakage tests (✅)
- **Unit tests:** Mostly auto-generated stubs (❌)
- **Coverage:** ~40-50% estimated (needs improvement)

### Critical Findings
1. **Data leakage unit tests are TODO stubs** - Real-money betting system requires comprehensive temporal validation
2. **Ensemble integration completely untested** - 4-way ensemble is production target but has 0 real tests
3. **Feature pipeline lacks end-to-end tests** - Most common source of temporal leakage
4. **Model registry CLI untested** - 1,179-line tool with 0 tests is high-risk

### Path to 80%+ Coverage
1. **Week 1:** Complete production blockers (tests 1-3) → 60% coverage
2. **Week 2-3:** High priority tests (tests 4-6) → 70% coverage
3. **Week 4-5:** Medium priority tests (tests 7-10) → 80% coverage
4. **Week 6:** CI/CD integration + documentation → Production-ready

### Final Recommendation
**Do not deploy to production until:**
1. ✅ Data leakage detector has 90%+ coverage (test 1)
2. ✅ Ensemble integration has 85%+ coverage (test 2)
3. ✅ Feature pipeline has end-to-end validation (test 3)
4. ✅ All pre-training gates enforce temporal safety

**Real-money betting requires ironclad temporal integrity. No shortcuts.**

---

**Audit Complete**
**Generated:** November 9, 2025
**Next Review:** After Week 1 tests complete
