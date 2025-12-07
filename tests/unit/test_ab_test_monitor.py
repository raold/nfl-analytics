"""
Unit tests for A/B Test Monitor.

Tests guardrail enforcement, statistical significance calculations,
and decision logic using simulated A/B test scenarios.

Author: Claude Code (Sonnet 4.5)
Created: November 22, 2025
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml
from py.utils.ab_test_monitor import (
    ABTestMonitor,
    Decision,
    GuardrailConfig,
    GuardrailStatus,
    SignificanceReport,
    TestStatus,
)

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Mock A/B test configuration."""
    return {
        'test_config': {
            'name': 'ensemble_v5_vs_v6',
            'start_date': '2025-11-14',
            'end_date': '2025-12-12',
            'traffic_split': {'control': 50, 'treatment': 50},
            'min_sample_size': 50
        },
        'success_criteria': {
            'mae_improvement_yards': 5.0,
            'roi_improvement_pct': 1.0,
            'statistical_significance': {'p_value': 0.05}
        },
        'guardrails': {
            'max_mae_degradation_pct': 10.0,
            'max_roi_drop_pct': 2.0,
            'min_win_rate_drop_pct': 2.0,
            'max_calibration_drop_pct': 5.0
        }
    }


@pytest.fixture
def mock_config_file(mock_config, tmp_path):
    """Create temporary config file."""
    config_path = tmp_path / "ab_test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(mock_config, f)
    return str(config_path)


@pytest.fixture
def mock_db_connection():
    """Mock database connection."""
    with patch('py.utils.ab_test_monitor.psycopg2.connect') as mock_conn:
        yield mock_conn


@pytest.fixture
def sample_ab_results():
    """Generate sample A/B test results."""
    np.random.seed(42)
    n_control = 100
    n_treatment = 100

    # Control group (baseline)
    control = pd.DataFrame({
        'game_id': range(n_control),
        'prediction_date': [datetime.now() - timedelta(days=i) for i in range(n_control)],
        'group': 'control',
        'predicted_value': np.random.normal(150, 40, n_control),
        'actual_value': np.random.normal(150, 40, n_control)
    })

    # Treatment group (improved by 7 yards MAE)
    treatment = pd.DataFrame({
        'game_id': range(n_control, n_control + n_treatment),
        'prediction_date': [datetime.now() - timedelta(days=i) for i in range(n_treatment)],
        'group': 'treatment',
        'predicted_value': np.random.normal(150, 33, n_treatment),  # Tighter predictions
        'actual_value': np.random.normal(150, 40, n_treatment)
    })

    df = pd.concat([control, treatment], ignore_index=True)

    # Calculate error, win, roi
    df['error'] = np.abs(df['predicted_value'] - df['actual_value'])
    df['win'] = (np.random.random(len(df)) > 0.45).astype(int)  # 55% win rate
    df['roi'] = df['win'].map({1: 1.0, 0: -1.1})

    return df


@pytest.fixture
def degraded_ab_results():
    """Generate A/B results where treatment is worse (guardrail violation)."""
    np.random.seed(42)
    n_control = 100
    n_treatment = 100

    # Control group
    control = pd.DataFrame({
        'game_id': range(n_control),
        'prediction_date': [datetime.now() - timedelta(days=i) for i in range(n_control)],
        'group': 'control',
        'predicted_value': np.random.normal(150, 40, n_control),
        'actual_value': np.random.normal(150, 40, n_control)
    })

    # Treatment group (degraded by 15% MAE)
    treatment = pd.DataFrame({
        'game_id': range(n_control, n_control + n_treatment),
        'prediction_date': [datetime.now() - timedelta(days=i) for i in range(n_treatment)],
        'group': 'treatment',
        'predicted_value': np.random.normal(150, 50, n_treatment),  # Worse predictions
        'actual_value': np.random.normal(150, 40, n_treatment)
    })

    df = pd.concat([control, treatment], ignore_index=True)

    # Calculate error, win, roi
    df['error'] = np.abs(df['predicted_value'] - df['actual_value'])
    df['win'] = (np.random.random(len(df)) > 0.60).astype(int)  # 40% win rate (worse)
    df['roi'] = df['win'].map({1: 1.0, 0: -1.1})

    return df


# ============================================================================
# Test ABTestMonitor Initialization
# ============================================================================

def test_monitor_initialization(mock_config_file, mock_db_connection):
    """Test monitor initializes correctly."""
    monitor = ABTestMonitor(config_path=mock_config_file)

    assert monitor.config['test_config']['name'] == 'ensemble_v5_vs_v6'
    assert monitor.guardrails.max_mae_degradation_pct == 10.0
    assert monitor.guardrails.min_sample_size == 50


def test_monitor_loads_config(mock_config_file, mock_db_connection):
    """Test configuration loading."""
    monitor = ABTestMonitor(config_path=mock_config_file)

    assert 'test_config' in monitor.config
    assert 'success_criteria' in monitor.config
    assert 'guardrails' in monitor.config


def test_monitor_custom_guardrails(mock_config_file, mock_db_connection):
    """Test custom guardrail configuration."""
    custom_guardrails = GuardrailConfig(
        max_mae_degradation_pct=5.0,
        max_roi_drop_pct=1.0,
        min_sample_size=100
    )

    monitor = ABTestMonitor(config_path=mock_config_file, guardrails=custom_guardrails)

    assert monitor.guardrails.max_mae_degradation_pct == 5.0
    assert monitor.guardrails.max_roi_drop_pct == 1.0
    assert monitor.guardrails.min_sample_size == 100


# ============================================================================
# Test Data Retrieval
# ============================================================================

def test_get_current_metrics(mock_config_file, mock_db_connection, sample_ab_results):
    """Test current metrics calculation."""
    monitor = ABTestMonitor(config_path=mock_config_file)

    with patch.object(monitor, 'get_ab_test_results', return_value=sample_ab_results):
        metrics = monitor.get_current_metrics()

    assert 'control' in metrics
    assert 'treatment' in metrics

    # Check control metrics
    assert 'mae' in metrics['control']
    assert 'win_rate' in metrics['control']
    assert 'roi' in metrics['control']
    assert 'sample_size' in metrics['control']

    # Check treatment metrics
    assert metrics['treatment']['sample_size'] == 100


def test_get_current_metrics_empty_data(mock_config_file, mock_db_connection):
    """Test metrics calculation with no data."""
    monitor = ABTestMonitor(config_path=mock_config_file)

    with patch.object(monitor, 'get_ab_test_results', return_value=pd.DataFrame()):
        metrics = monitor.get_current_metrics()

    assert metrics['control'] == {}
    assert metrics['treatment'] == {}


# ============================================================================
# Test Guardrail Enforcement
# ============================================================================

def test_check_guardrails_pass(mock_config_file, mock_db_connection, sample_ab_results):
    """Test guardrails pass with good data."""
    monitor = ABTestMonitor(config_path=mock_config_file)

    with patch.object(monitor, 'get_ab_test_results', return_value=sample_ab_results):
        status = monitor.check_guardrails()

    # Should pass (sample data has similar performance)
    assert not status.violated
    assert len(status.violations) == 0


def test_check_guardrails_mae_violation(mock_config_file, mock_db_connection, degraded_ab_results):
    """Test guardrail violation on MAE degradation."""
    monitor = ABTestMonitor(config_path=mock_config_file)

    with patch.object(monitor, 'get_ab_test_results', return_value=degraded_ab_results):
        status = monitor.check_guardrails()

    # Should violate MAE degradation guardrail (>10%)
    assert status.violated
    assert any('MAE degradation' in v for v in status.violations)


def test_check_guardrails_sample_size_warning(mock_config_file, mock_db_connection, sample_ab_results):
    """Test warning for insufficient sample size."""
    monitor = ABTestMonitor(config_path=mock_config_file)

    # Take only 30 treatment samples (below min_sample_size=50)
    small_sample = sample_ab_results[sample_ab_results['group'] == 'treatment'].head(30)
    small_sample = pd.concat([
        sample_ab_results[sample_ab_results['group'] == 'control'],
        small_sample
    ])

    with patch.object(monitor, 'get_ab_test_results', return_value=small_sample):
        status = monitor.check_guardrails()

    # Should have warning (not violation)
    assert len(status.warnings) > 0
    assert any('Sample size' in w for w in status.warnings)


def test_auto_stop_on_violation(mock_config_file, mock_db_connection, degraded_ab_results):
    """Test auto-stop when guardrails violated."""
    monitor = ABTestMonitor(config_path=mock_config_file)

    with patch.object(monitor, 'get_ab_test_results', return_value=degraded_ab_results):
        with patch.object(monitor, '_update_test_status') as mock_update:
            stopped = monitor.auto_stop_if_violated()

    assert stopped
    mock_update.assert_called_once()
    call_args = mock_update.call_args
    assert call_args[0][0] == TestStatus.STOPPED_GUARDRAIL


def test_auto_stop_no_violation(mock_config_file, mock_db_connection, sample_ab_results):
    """Test no auto-stop when guardrails pass."""
    monitor = ABTestMonitor(config_path=mock_config_file)

    with patch.object(monitor, 'get_ab_test_results', return_value=sample_ab_results):
        with patch.object(monitor, '_update_test_status') as mock_update:
            stopped = monitor.auto_stop_if_violated()

    assert not stopped
    mock_update.assert_not_called()


# ============================================================================
# Test Statistical Analysis
# ============================================================================

def test_calculate_significance_ttest(mock_config_file, mock_db_connection):
    """Test t-test significance calculation."""
    monitor = ABTestMonitor(config_path=mock_config_file)

    # Create data with known difference
    np.random.seed(42)
    control = pd.DataFrame({
        'group': 'control',
        'error': np.random.normal(40, 10, 100),
        'win': np.random.binomial(1, 0.55, 100),
        'roi': np.random.normal(5, 2, 100)
    })

    treatment = pd.DataFrame({
        'group': 'treatment',
        'error': np.random.normal(33, 10, 100),  # 7 yard improvement
        'win': np.random.binomial(1, 0.60, 100),
        'roi': np.random.normal(7, 2, 100)
    })

    df = pd.concat([control, treatment])

    with patch.object(monitor, 'get_ab_test_results', return_value=df):
        sig_report = monitor.calculate_significance('mae', test_type='t-test')

    assert sig_report.metric == 'mae'
    assert sig_report.control_mean > sig_report.treatment_mean  # Treatment better
    assert sig_report.difference < 0  # Negative = improvement
    assert sig_report.test_type == 't-test'
    assert 0 <= sig_report.p_value <= 1


def test_calculate_significance_bootstrap(mock_config_file, mock_db_connection):
    """Test bootstrap significance calculation."""
    monitor = ABTestMonitor(config_path=mock_config_file)

    np.random.seed(42)
    control = pd.DataFrame({
        'group': 'control',
        'error': np.random.normal(40, 10, 100),
        'win': np.random.binomial(1, 0.55, 100),
        'roi': np.random.normal(5, 2, 100)
    })

    treatment = pd.DataFrame({
        'group': 'treatment',
        'error': np.random.normal(33, 10, 100),
        'win': np.random.binomial(1, 0.60, 100),
        'roi': np.random.normal(7, 2, 100)
    })

    df = pd.concat([control, treatment])

    with patch.object(monitor, 'get_ab_test_results', return_value=df):
        sig_report = monitor.calculate_significance('mae', test_type='bootstrap')

    assert sig_report.test_type == 'bootstrap'
    assert sig_report.confidence_interval_95[0] < sig_report.confidence_interval_95[1]


def test_calculate_significance_bayesian(mock_config_file, mock_db_connection):
    """Test Bayesian significance calculation."""
    monitor = ABTestMonitor(config_path=mock_config_file)

    np.random.seed(42)
    control = pd.DataFrame({
        'group': 'control',
        'error': np.random.normal(40, 10, 100),
        'win': np.random.binomial(1, 0.55, 100),
        'roi': np.random.normal(5, 2, 100)
    })

    treatment = pd.DataFrame({
        'group': 'treatment',
        'error': np.random.normal(33, 10, 100),
        'win': np.random.binomial(1, 0.60, 100),
        'roi': np.random.normal(7, 2, 100)
    })

    df = pd.concat([control, treatment])

    with patch.object(monitor, 'get_ab_test_results', return_value=df):
        sig_report = monitor.calculate_significance('mae', test_type='bayesian')

    assert sig_report.test_type == 'bayesian'
    assert sig_report.confidence_interval_95[0] < sig_report.confidence_interval_95[1]


def test_calculate_significance_roi_metric(mock_config_file, mock_db_connection):
    """Test significance calculation for ROI metric."""
    monitor = ABTestMonitor(config_path=mock_config_file)

    np.random.seed(42)
    control = pd.DataFrame({
        'group': 'control',
        'error': np.random.normal(40, 10, 100),
        'win': np.random.binomial(1, 0.55, 100),
        'roi': np.random.normal(5, 2, 100)
    })

    treatment = pd.DataFrame({
        'group': 'treatment',
        'error': np.random.normal(33, 10, 100),
        'win': np.random.binomial(1, 0.60, 100),
        'roi': np.random.normal(7, 2, 100)  # 2pp improvement
    })

    df = pd.concat([control, treatment])

    with patch.object(monitor, 'get_ab_test_results', return_value=df):
        sig_report = monitor.calculate_significance('roi', test_type='t-test')

    assert sig_report.metric == 'roi'
    assert sig_report.difference > 0  # Positive = improvement


# ============================================================================
# Test Decision Logic
# ============================================================================

def test_recommend_decision_insufficient_data(mock_config_file, mock_db_connection):
    """Test decision with insufficient data."""
    monitor = ABTestMonitor(config_path=mock_config_file)

    # Only 20 samples (below min_sample_size=50)
    small_sample = pd.DataFrame({
        'group': ['control'] * 10 + ['treatment'] * 10,
        'error': np.random.normal(40, 10, 20),
        'win': np.random.binomial(1, 0.55, 20),
        'roi': np.random.normal(5, 2, 20)
    })

    with patch.object(monitor, 'get_ab_test_results', return_value=small_sample):
        decision = monitor.recommend_decision()

    assert decision == Decision.INSUFFICIENT_DATA


def test_recommend_decision_promote(mock_config_file, mock_db_connection):
    """Test recommendation to promote treatment."""
    monitor = ABTestMonitor(config_path=mock_config_file)

    # Treatment significantly better (7 yard MAE improvement, 2pp ROI improvement)
    np.random.seed(42)
    control = pd.DataFrame({
        'group': 'control',
        'error': np.random.normal(40, 10, 100),
        'win': np.random.binomial(1, 0.55, 100),
        'roi': np.random.normal(5, 2, 100)
    })

    treatment = pd.DataFrame({
        'group': 'treatment',
        'error': np.random.normal(33, 8, 100),  # 7 yard improvement
        'win': np.random.binomial(1, 0.60, 100),
        'roi': np.random.normal(7, 2, 100)  # 2pp improvement
    })

    df = pd.concat([control, treatment])

    with patch.object(monitor, 'get_ab_test_results', return_value=df):
        with patch.object(monitor, 'check_guardrails', return_value=GuardrailStatus(violated=False)):
            decision = monitor.recommend_decision()

    # Should recommend promotion (meets success criteria)
    assert decision == Decision.PROMOTE_TREATMENT


def test_recommend_decision_stop_degradation(mock_config_file, mock_db_connection, degraded_ab_results):
    """Test recommendation to stop due to degradation."""
    monitor = ABTestMonitor(config_path=mock_config_file)

    with patch.object(monitor, 'get_ab_test_results', return_value=degraded_ab_results):
        decision = monitor.recommend_decision()

    assert decision == Decision.STOP_DEGRADATION


def test_recommend_decision_continue(mock_config_file, mock_db_connection):
    """Test recommendation to continue (no clear winner yet)."""
    monitor = ABTestMonitor(config_path=mock_config_file)

    # Treatment slightly better but not significant
    np.random.seed(42)
    control = pd.DataFrame({
        'group': 'control',
        'error': np.random.normal(40, 10, 100),
        'win': np.random.binomial(1, 0.55, 100),
        'roi': np.random.normal(5, 2, 100)
    })

    treatment = pd.DataFrame({
        'group': 'treatment',
        'error': np.random.normal(38, 10, 100),  # Only 2 yard improvement (not enough)
        'win': np.random.binomial(1, 0.56, 100),
        'roi': np.random.normal(5.5, 2, 100)  # Only 0.5pp improvement (not enough)
    })

    df = pd.concat([control, treatment])

    with patch.object(monitor, 'get_ab_test_results', return_value=df):
        with patch.object(monitor, 'check_guardrails', return_value=GuardrailStatus(violated=False)):
            decision = monitor.recommend_decision()

    # Should continue (insufficient evidence)
    assert decision == Decision.CONTINUE


# ============================================================================
# Test Export and Reporting
# ============================================================================

def test_export_results(mock_config_file, mock_db_connection, sample_ab_results):
    """Test results export."""
    monitor = ABTestMonitor(config_path=mock_config_file)

    with patch.object(monitor, 'get_ab_test_results', return_value=sample_ab_results):
        df = monitor.export_results(format='dataframe')

    assert isinstance(df, pd.DataFrame)
    assert 'group' in df.columns
    assert 'mae' in df.columns
    assert len(df) == 2  # control and treatment rows


def test_generate_report(mock_config_file, mock_db_connection, sample_ab_results):
    """Test report generation."""
    monitor = ABTestMonitor(config_path=mock_config_file)

    with patch.object(monitor, 'get_ab_test_results', return_value=sample_ab_results):
        report = monitor.generate_report()

    assert isinstance(report, str)
    assert 'A/B TEST REPORT' in report
    assert 'CURRENT METRICS' in report
    assert 'STATISTICAL SIGNIFICANCE' in report
    assert 'GUARDRAILS' in report
    assert 'RECOMMENDATION' in report


# ============================================================================
# Test Data Classes
# ============================================================================

def test_guardrail_status_to_dict():
    """Test GuardrailStatus serialization."""
    status = GuardrailStatus(
        violated=True,
        violations=['MAE degradation 15% exceeds limit 10%'],
        warnings=['Sample size below minimum'],
        metrics={'mae_degradation_pct': 15.0}
    )

    d = status.to_dict()

    assert d['violated'] == True
    assert len(d['violations']) == 1
    assert len(d['warnings']) == 1
    assert d['metrics']['mae_degradation_pct'] == 15.0


def test_significance_report_to_dict():
    """Test SignificanceReport serialization."""
    report = SignificanceReport(
        metric='mae',
        control_mean=40.0,
        treatment_mean=33.0,
        difference=-7.0,
        difference_pct=-17.5,
        p_value=0.001,
        significant=True,
        confidence_interval_95=(-10.0, -4.0),
        test_type='t-test',
        sample_size_control=100,
        sample_size_treatment=100,
        effect_size=0.7
    )

    d = report.to_dict()

    assert d['metric'] == 'mae'
    assert d['difference'] == -7.0
    assert d['p_value'] == 0.001
    assert d['significant'] == True
    assert d['ci_lower'] == -10.0
    assert d['ci_upper'] == -4.0


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
def test_full_monitoring_workflow(mock_config_file, mock_db_connection, sample_ab_results):
    """Test complete monitoring workflow."""
    monitor = ABTestMonitor(config_path=mock_config_file)

    with patch.object(monitor, 'get_ab_test_results', return_value=sample_ab_results):
        # 1. Check guardrails
        guardrail_status = monitor.check_guardrails()
        assert not guardrail_status.violated

        # 2. Calculate significance
        mae_sig = monitor.calculate_significance('mae')
        roi_sig = monitor.calculate_significance('roi')
        assert mae_sig.p_value >= 0
        assert roi_sig.p_value >= 0

        # 3. Get recommendation
        decision = monitor.recommend_decision()
        assert decision in [Decision.CONTINUE, Decision.PROMOTE_TREATMENT, Decision.INSUFFICIENT_DATA]

        # 4. Export results
        results_df = monitor.export_results()
        assert len(results_df) == 2

        # 5. Generate report
        report = monitor.generate_report()
        assert 'A/B TEST REPORT' in report


@pytest.mark.integration
def test_degradation_auto_stop_workflow(mock_config_file, mock_db_connection, degraded_ab_results):
    """Test auto-stop workflow on degradation."""
    monitor = ABTestMonitor(config_path=mock_config_file)

    with patch.object(monitor, 'get_ab_test_results', return_value=degraded_ab_results):
        with patch.object(monitor, '_update_test_status') as mock_update:
            # Check guardrails
            status = monitor.check_guardrails()
            assert status.violated

            # Auto-stop
            stopped = monitor.auto_stop_if_violated()
            assert stopped
            mock_update.assert_called_once()

            # Verify status updated
            call_args = mock_update.call_args
            assert call_args[0][0] == TestStatus.STOPPED_GUARDRAIL


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
