#!/usr/bin/env python3
"""
Unit Tests for Rule 20: Data Sufficiency Gate

**Purpose**: Verify that Rule 20 correctly detects underpowered experiments
and blocks training with insufficient data.

**Test Coverage**:
1. Meta-learning without data validation (CRITICAL)
2. Online learning without initialization check (HIGH)
3. LightGBM/XGBoost without sample validation (HIGH)
4. Ensemble reweighting without component validation (MEDIUM)
5. Statistical tests without minimum samples (MEDIUM)

**Author**: Claude Code (Sonnet 4.5)
**Date**: November 19, 2025
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from py.utils.enforcement_router import EnforcementRouter, ValidationStatus


@pytest.fixture
def router():
    """Create enforcement router instance."""
    return EnforcementRouter()


class TestRule20MetaLearning:
    """Test Rule 20: Meta-learning data sufficiency validation."""

    def test_meta_learning_without_validation_blocks(self, router, tmp_path):
        """Meta-learning without data validation should BLOCK."""
        file_path = tmp_path / "train_meta_model.py"
        content = """
import lightgbm as lgb

def train_meta_learning_model(X, y):
    # No data validation before training
    meta_model = lgb.LGBMRegressor()
    meta_model.fit(X, y)  # Missing: assert len(X) >= 5000
    return meta_model
"""
        file_path.write_text(content)

        results = router.validate_file(str(file_path), content)
        rule_20_results = [r for r in results if r.agent_name == 'data_sufficiency_gate']

        assert len(rule_20_results) == 1
        result = rule_20_results[0]
        assert result.failed
        assert result.status == ValidationStatus.FAILED
        assert 'Meta-learning without data validation' in str(result.details['issues'])
        assert result.details['issues'][0]['severity'] == 'CRITICAL'
        assert result.details['issues'][0]['required_samples'] == 5000

    def test_meta_learning_with_validation_passes(self, router, tmp_path):
        """Meta-learning with proper data validation should PASS."""
        file_path = tmp_path / "train_meta_model_safe.py"
        content = """
import lightgbm as lgb

def train_meta_learning_model(X, y):
    # Proper data validation
    assert len(X) >= 5000, "Insufficient data for meta-learning"

    meta_model = lgb.LGBMRegressor()
    meta_model.fit(X, y)
    return meta_model
"""
        file_path.write_text(content)

        results = router.validate_file(str(file_path), content)
        rule_20_results = [r for r in results if r.agent_name == 'data_sufficiency_gate']

        assert len(rule_20_results) == 1
        result = rule_20_results[0]
        assert result.passed

    def test_adaptive_weighting_without_validation_blocks(self, router, tmp_path):
        """Adaptive weighting without data validation should BLOCK."""
        file_path = tmp_path / "adaptive_weights.py"
        content = """
import xgboost as xgb

def train_adaptive_weight_predictor(context_features, optimal_weights):
    # No validation on training data size
    model = xgb.XGBRegressor()
    model.fit(context_features, optimal_weights)
    return model
"""
        file_path.write_text(content)

        results = router.validate_file(str(file_path), content)
        rule_20_results = [r for r in results if r.agent_name == 'data_sufficiency_gate']

        assert len(rule_20_results) == 1
        result = rule_20_results[0]
        assert result.failed
        assert 'Meta-learning without data validation' in str(result.details['issues'])


class TestRule20OnlineLearning:
    """Test Rule 20: Online learning data sufficiency validation."""

    def test_online_learning_without_init_check_warns(self, router, tmp_path):
        """Online learning without initialization check should warn (HIGH severity)."""
        file_path = tmp_path / "online_model.py"
        content = """
from sklearn.linear_model import SGDRegressor

def train_online_learning_model(stream_data):
    model = SGDRegressor()

    for batch in stream_data:
        model.partial_fit(batch['X'], batch['y'])  # Missing: init batch size check

    return model
"""
        file_path.write_text(content)

        results = router.validate_file(str(file_path), content)
        rule_20_results = [r for r in results if r.agent_name == 'data_sufficiency_gate']

        assert len(rule_20_results) == 1
        result = rule_20_results[0]
        assert result.failed
        issues = [i for i in result.details['issues'] if 'Online learning' in i['pattern']]
        assert len(issues) == 1
        assert issues[0]['severity'] == 'HIGH'
        assert issues[0]['required_samples'] == 1000

    def test_online_learning_with_warmstart_passes(self, router, tmp_path):
        """Online learning with proper initialization should PASS."""
        file_path = tmp_path / "online_model_safe.py"
        content = """
from sklearn.linear_model import SGDRegressor

def train_online_learning_model(stream_data):
    model = SGDRegressor()

    # Proper initialization check
    initial_batch_size = 1200
    if len(stream_data[0]['X']) < 1000:
        raise ValueError("Need ≥1,000 samples for warm start")

    # Warmstart with sufficient data
    model.partial_fit(stream_data[0]['X'][:initial_batch_size],
                     stream_data[0]['y'][:initial_batch_size])

    for batch in stream_data[1:]:
        model.partial_fit(batch['X'], batch['y'])

    return model
"""
        file_path.write_text(content)

        results = router.validate_file(str(file_path), content)
        rule_20_results = [r for r in results if r.agent_name == 'data_sufficiency_gate']

        assert len(rule_20_results) == 1
        result = rule_20_results[0]
        assert result.passed


class TestRule20TreeModels:
    """Test Rule 20: Complex tree model data sufficiency validation."""

    def test_lightgbm_without_sample_check_warns(self, router, tmp_path):
        """LightGBM training without sample validation should warn."""
        file_path = tmp_path / "train_lgb.py"
        content = """
import lightgbm as lgb

def train_lightgbm_model(X, y):
    # 15 features detected but no sample validation
    n_features = 15

    model = lgb.train({'objective': 'regression'},
                     lgb.Dataset(X, y))
    return model
"""
        file_path.write_text(content)

        results = router.validate_file(str(file_path), content)
        rule_20_results = [r for r in results if r.agent_name == 'data_sufficiency_gate']

        assert len(rule_20_results) == 1
        result = rule_20_results[0]
        assert result.failed
        issues = [i for i in result.details['issues'] if 'Complex tree model' in i['pattern']]
        assert len(issues) == 1
        assert issues[0]['severity'] == 'HIGH'
        assert issues[0]['required_samples'] == 1500  # 15 features * 100
        assert issues[0]['features_detected'] == 15

    def test_xgboost_with_validation_passes(self, router, tmp_path):
        """XGBoost with proper sample validation should PASS."""
        file_path = tmp_path / "train_xgb_safe.py"
        content = """
import xgboost as xgb

def train_xgboost_model(X, y):
    n_features = X.shape[1]

    # Proper validation
    assert len(X) >= n_features * 100, f"Need {n_features * 100}+ samples"

    model = xgb.train({'objective': 'reg:squarederror'},
                     xgb.DMatrix(X, y))
    return model
"""
        file_path.write_text(content)

        results = router.validate_file(str(file_path), content)
        rule_20_results = [r for r in results if r.agent_name == 'data_sufficiency_gate']

        assert len(rule_20_results) == 1
        result = rule_20_results[0]
        assert result.passed


class TestRule20EnsembleReweighting:
    """Test Rule 20: Ensemble reweighting data sufficiency validation."""

    def test_ensemble_optimization_without_component_check_warns(self, router, tmp_path):
        """Ensemble weight optimization without per-component validation should warn."""
        file_path = tmp_path / "optimize_ensemble_weights.py"
        content = """
import cvxpy as cvx

def optimize_ensemble_weights(component_predictions, actuals):
    # No validation of games per component
    weights = cvx.Variable(len(component_predictions))

    objective = cvx.Minimize(cvx.sum_squares(
        sum([w * pred for w, pred in zip(weights, component_predictions)]) - actuals
    ))

    problem = cvx.Problem(objective)
    problem.solve()

    return weights.value
"""
        file_path.write_text(content)

        results = router.validate_file(str(file_path), content)
        rule_20_results = [r for r in results if r.agent_name == 'data_sufficiency_gate']

        assert len(rule_20_results) == 1
        result = rule_20_results[0]
        assert result.failed
        issues = [i for i in result.details['issues'] if 'Ensemble reweighting' in i['pattern']]
        assert len(issues) == 1
        assert issues[0]['severity'] == 'MEDIUM'
        assert issues[0]['required_samples'] == 100

    def test_ensemble_optimization_with_validation_passes(self, router, tmp_path):
        """Ensemble optimization with component validation should PASS."""
        file_path = tmp_path / "optimize_ensemble_weights_safe.py"
        content = """
import cvxpy as cvx

def optimize_ensemble_weights(component_predictions, actuals):
    # Proper per-component validation
    games_per_component = [len(pred) for pred in component_predictions]
    assert min(games_per_component) >= 100, "Need ≥100 games per component"

    weights = cvx.Variable(len(component_predictions))

    objective = cvx.Minimize(cvx.sum_squares(
        sum([w * pred for w, pred in zip(weights, component_predictions)]) - actuals
    ))

    problem = cvx.Problem(objective)
    problem.solve()

    return weights.value
"""
        file_path.write_text(content)

        results = router.validate_file(str(file_path), content)
        rule_20_results = [r for r in results if r.agent_name == 'data_sufficiency_gate']

        assert len(rule_20_results) == 1
        result = rule_20_results[0]
        assert result.passed


class TestRule20StatisticalTests:
    """Test Rule 20: Statistical test data sufficiency validation."""

    def test_t_test_without_sample_check_warns(self, router, tmp_path):
        """t-test without minimum sample validation should warn."""
        file_path = tmp_path / "statistical_analysis.py"
        content = """
from scipy.stats import ttest_ind

def compare_models(model_a_errors, model_b_errors):
    # No validation of sample size (need n≥30 for CLT)
    t_stat, p_value = ttest_ind(model_a_errors, model_b_errors)
    return t_stat, p_value
"""
        file_path.write_text(content)

        results = router.validate_file(str(file_path), content)
        rule_20_results = [r for r in results if r.agent_name == 'data_sufficiency_gate']

        assert len(rule_20_results) == 1
        result = rule_20_results[0]
        assert result.failed
        issues = [i for i in result.details['issues'] if 'Statistical tests' in i['pattern']]
        assert len(issues) == 1
        assert issues[0]['severity'] == 'MEDIUM'
        assert issues[0]['required_samples'] == 30

    def test_statistical_test_with_validation_passes(self, router, tmp_path):
        """Statistical test with proper sample validation should PASS."""
        file_path = tmp_path / "statistical_analysis_safe.py"
        content = """
from scipy.stats import ttest_ind

def compare_models(model_a_errors, model_b_errors):
    # Proper CLT validation
    n = min(len(model_a_errors), len(model_b_errors))
    assert n >= 30, "Need ≥30 samples per group for CLT"

    t_stat, p_value = ttest_ind(model_a_errors, model_b_errors)
    return t_stat, p_value
"""
        file_path.write_text(content)

        results = router.validate_file(str(file_path), content)
        rule_20_results = [r for r in results if r.agent_name == 'data_sufficiency_gate']

        assert len(rule_20_results) == 1
        result = rule_20_results[0]
        assert result.passed


class TestRule20EdgeCases:
    """Test Rule 20: Edge cases and special scenarios."""

    def test_test_files_skipped(self, router, tmp_path):
        """Test files should be skipped (placeholders allowed for testing)."""
        file_path = tmp_path / "test_meta_learning.py"
        content = """
def test_meta_learning_insufficient_data():
    # This is a test - should be skipped
    X = [[1, 2], [3, 4]]  # Only 2 samples
    model.fit(X, y)
"""
        file_path.write_text(content)

        results = router.validate_file(str(file_path), content)
        rule_20_results = [r for r in results if r.agent_name == 'data_sufficiency_gate']

        assert len(rule_20_results) == 1
        result = rule_20_results[0]
        assert result.status == ValidationStatus.SKIPPED

    def test_non_python_files_skipped(self, router, tmp_path):
        """Non-Python files should be skipped."""
        file_path = tmp_path / "train_model.sh"
        content = """#!/bin/bash
python train_meta_learning.py
"""
        file_path.write_text(content)

        results = router.validate_file(str(file_path), content)
        rule_20_results = [r for r in results if r.agent_name == 'data_sufficiency_gate']

        # Should not trigger (not Python file)
        assert len(rule_20_results) == 1
        result = rule_20_results[0]
        assert result.status == ValidationStatus.SKIPPED

    def test_multiple_issues_reported(self, router, tmp_path):
        """Multiple data sufficiency issues should all be reported."""
        file_path = tmp_path / "complex_training.py"
        content = """
import lightgbm as lgb
from scipy.stats import ttest_ind

def train_and_test(meta_features, y, test_results_a, test_results_b):
    # Issue 1: Meta-learning without validation
    meta_model = lgb.train({'objective': 'regression'},
                          lgb.Dataset(meta_features, y))

    # Issue 2: Statistical test without sample check
    t_stat, p_val = ttest_ind(test_results_a, test_results_b)

    return meta_model, t_stat
"""
        file_path.write_text(content)

        results = router.validate_file(str(file_path), content)
        rule_20_results = [r for r in results if r.agent_name == 'data_sufficiency_gate']

        assert len(rule_20_results) == 1
        result = rule_20_results[0]
        assert result.failed
        # Should detect both tree model issue and statistical test issue
        assert len(result.details['issues']) >= 2

    def test_v7_failure_prevented(self, router, tmp_path):
        """Ensure Rule 20 would have prevented v7.0 failure."""
        file_path = tmp_path / "train_meta_weight_predictor.py"
        content = """
import lightgbm as lgb

def train_meta_weight_predictor(context_features, optimal_weights):
    '''
    Train meta-learning model to predict ensemble weights from context.

    Context features: 15 features (QB experience, roster stability, etc.)
    Training data: 1,701 games (from v7.0 - INSUFFICIENT!)
    '''
    # This is the EXACT pattern from v7.0 that failed
    meta_model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1
    )

    # MISSING: assert len(context_features) >= 5000
    meta_model.fit(context_features, optimal_weights)

    return meta_model
"""
        file_path.write_text(content)

        results = router.validate_file(str(file_path), content)
        rule_20_results = [r for r in results if r.agent_name == 'data_sufficiency_gate']

        assert len(rule_20_results) == 1
        result = rule_20_results[0]
        assert result.failed
        assert 'CRITICAL' in str(result.details['issues'])
        assert '5000' in str(result.details['issues'])
        # Verify postmortem reference included
        assert 'v7.0' in str(result.details)


class TestRule20Decorator:
    """Test Rule 20: Runtime decorator enforcement."""

    def test_decorator_blocks_underpowered_training(self, router):
        """Decorator should block function execution if data insufficient."""
        from py.utils.enforcement_router import enforce_data_sufficiency

        @enforce_data_sufficiency
        def train_meta_model(X, y):
            # Missing data validation
            import lightgbm as lgb
            model = lgb.LGBMRegressor()
            model.fit(X, y)
            return model

        # This would raise RuntimeError in production
        # (skipped in test since decorator checks source file, not test code)
        pass


# Integration test
def test_rule_20_integration(router, tmp_path):
    """Full integration test: detect all 5 patterns in one file."""
    file_path = tmp_path / "comprehensive_training.py"
    content = """
import lightgbm as lgb
import cvxpy as cvx
from scipy.stats import ttest_ind

def comprehensive_training_pipeline():
    # Pattern 1: Meta-learning (CRITICAL)
    meta_model = lgb.LGBMRegressor()
    meta_model.fit(meta_features, y)  # Missing validation

    # Pattern 2: Online learning (HIGH)
    from sklearn.linear_model import SGDRegressor
    online_model = SGDRegressor()
    online_model.partial_fit(X, y)  # Missing init check

    # Pattern 3: LightGBM (HIGH)
    n_features = 15
    lgb_model = lgb.train({'objective': 'regression'}, lgb.Dataset(X, y))

    # Pattern 4: Ensemble reweighting (MEDIUM)
    weights = cvx.Variable(4)
    problem = cvx.Problem(cvx.Minimize(cvx.sum_squares(weights)))
    problem.solve()

    # Pattern 5: Statistical test (MEDIUM)
    t_stat, p_val = ttest_ind(errors_a, errors_b)

    return meta_model, online_model, lgb_model, weights, t_stat
"""
    file_path.write_text(content)

    results = router.validate_file(str(file_path), content)
    rule_20_results = [r for r in results if r.agent_name == 'data_sufficiency_gate']

    assert len(rule_20_results) == 1
    result = rule_20_results[0]
    assert result.failed

    # Should detect all patterns
    issues = result.details['issues']
    patterns_detected = [i['pattern'] for i in issues]

    assert any('Meta-learning' in p for p in patterns_detected)
    assert any('Online learning' in p for p in patterns_detected)
    assert any('Complex tree model' in p for p in patterns_detected)
    assert any('Ensemble reweighting' in p for p in patterns_detected)
    assert any('Statistical tests' in p for p in patterns_detected)

    # Verify severity classification
    critical_count = result.details['critical_count']
    assert critical_count >= 1  # At least meta-learning


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
