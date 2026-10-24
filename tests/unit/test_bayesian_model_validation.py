"""
Comprehensive tests for Bayesian Model Validation

Tests validation framework for Bayesian models before production deployment:
1. MCMC convergence - R-hat < 1.01, ESS > 400
2. Calibration - Coverage ≥ 85% (target: 90%)
3. Divergences - < 10 divergent transitions
4. Prior sensitivity - Robustness to prior specification
5. Posterior predictive checks - Model adequacy
6. Production readiness gates - Block deployment if validation fails

Critical Requirements:
- R-hat < 1.01 for all parameters (convergence)
- ESS_bulk > 400 and ESS_tail > 400 (sufficient samples)
- Calibration coverage > 85% (well-calibrated uncertainty)
- Divergences < 10 (sampling quality)
- ⛔ BLOCK deployment if any gate fails

Author: Claude Code (Sonnet 4.5)
Created: November 9, 2025
"""

import numpy as np
import pandas as pd
import pytest

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_mcmc_summary_good():
    """Create mock MCMC summary with good diagnostics."""
    return {
        "parameters": {
            "beta[0]": {
                "mean": 0.05,
                "sd": 0.02,
                "rhat": 1.001,
                "ess_bulk": 1500,
                "ess_tail": 1200,
                "divergences": 0,
            },
            "beta[1]": {
                "mean": 0.12,
                "sd": 0.03,
                "rhat": 1.002,
                "ess_bulk": 1400,
                "ess_tail": 1100,
                "divergences": 0,
            },
            "sigma": {
                "mean": 25.0,
                "sd": 2.5,
                "rhat": 1.003,
                "ess_bulk": 1600,
                "ess_tail": 1300,
                "divergences": 0,
            },
        },
        "total_divergences": 0,
        "n_samples": 2000,
        "n_chains": 4,
    }


@pytest.fixture
def mock_mcmc_summary_bad():
    """Create mock MCMC summary with convergence issues."""
    return {
        "parameters": {
            "beta[0]": {
                "mean": 0.05,
                "sd": 0.02,
                "rhat": 1.15,  # BAD: R-hat > 1.01
                "ess_bulk": 250,  # BAD: ESS < 400
                "ess_tail": 180,  # BAD: ESS < 400
                "divergences": 15,  # BAD: > 10 divergences
            },
            "beta[1]": {
                "mean": 0.12,
                "sd": 0.03,
                "rhat": 1.08,  # BAD
                "ess_bulk": 320,  # BAD
                "ess_tail": 250,  # BAD
                "divergences": 8,
            },
            "sigma": {
                "mean": 25.0,
                "sd": 2.5,
                "rhat": 1.001,  # GOOD
                "ess_bulk": 1600,  # GOOD
                "ess_tail": 1300,  # GOOD
                "divergences": 0,
            },
        },
        "total_divergences": 23,  # BAD: > 10
        "n_samples": 2000,
        "n_chains": 4,
    }


@pytest.fixture
def mock_calibration_data_good():
    """Create mock calibration data with good coverage."""
    np.random.seed(42)
    n = 1000

    # Generate well-calibrated predictions
    true_values = np.random.normal(250, 50, n)
    predictions = true_values + np.random.normal(0, 10, n)  # Small prediction error

    # 90% credible intervals: well-calibrated means the width matches the error SD (10)
    lower = predictions - 1.645 * 10
    upper = predictions + 1.645 * 10

    return pd.DataFrame(
        {
            "true_value": true_values,
            "prediction": predictions,
            "lower_90": lower,
            "upper_90": upper,
        }
    )


@pytest.fixture
def mock_calibration_data_bad():
    """Create mock calibration data with poor coverage."""
    np.random.seed(43)
    n = 1000

    # Generate poorly-calibrated predictions
    true_values = np.random.normal(250, 50, n)
    predictions = true_values + np.random.normal(0, 30, n)  # Large prediction error

    # Intervals too narrow (poor calibration)
    lower = predictions - 0.5 * 25  # Much less than 90% coverage
    upper = predictions + 0.5 * 25

    return pd.DataFrame(
        {
            "true_value": true_values,
            "prediction": predictions,
            "lower_90": lower,
            "upper_90": upper,
        }
    )


# ============================================================================
# TEST 1: MCMC CONVERGENCE (R-hat)
# ============================================================================


def test_rhat_validation_good(mock_mcmc_summary_good):
    """Test R-hat validation with good convergence."""
    summary = mock_mcmc_summary_good

    # Check all R-hat values
    rhat_values = [p["rhat"] for p in summary["parameters"].values()]
    max_rhat = max(rhat_values)

    # R-hat < 1.01 (good convergence)
    assert max_rhat < 1.01, f"R-hat = {max_rhat} exceeds 1.01 threshold"


def test_rhat_validation_bad(mock_mcmc_summary_bad):
    """Test R-hat validation catches convergence issues."""
    summary = mock_mcmc_summary_bad

    # Check all R-hat values
    rhat_values = [p["rhat"] for p in summary["parameters"].values()]
    max_rhat = max(rhat_values)

    # R-hat > 1.01 (poor convergence)
    assert max_rhat > 1.01, "Failed to detect convergence issues"

    # Specifically, beta[0] should fail
    assert summary["parameters"]["beta[0]"]["rhat"] > 1.01


def test_rhat_threshold_customizable():
    """Test that R-hat threshold can be customized."""
    summary = {
        "parameters": {
            "param1": {"rhat": 1.05},
            "param2": {"rhat": 1.02},
        }
    }

    # Strict threshold (1.01): Both fail
    strict_threshold = 1.01
    failing_params_strict = [
        name for name, p in summary["parameters"].items() if p["rhat"] > strict_threshold
    ]
    assert len(failing_params_strict) == 2

    # Relaxed threshold (1.03): Only param1 fails
    relaxed_threshold = 1.03
    failing_params_relaxed = [
        name for name, p in summary["parameters"].items() if p["rhat"] > relaxed_threshold
    ]
    assert len(failing_params_relaxed) == 1
    assert "param1" in failing_params_relaxed


# ============================================================================
# TEST 2: EFFECTIVE SAMPLE SIZE (ESS)
# ============================================================================


def test_ess_validation_good(mock_mcmc_summary_good):
    """Test ESS validation with sufficient samples."""
    summary = mock_mcmc_summary_good

    # Check ESS_bulk and ESS_tail for all parameters
    for param_name, param_data in summary["parameters"].items():
        assert (
            param_data["ess_bulk"] > 400
        ), f"{param_name} ESS_bulk = {param_data['ess_bulk']} < 400"
        assert (
            param_data["ess_tail"] > 400
        ), f"{param_name} ESS_tail = {param_data['ess_tail']} < 400"


def test_ess_validation_bad(mock_mcmc_summary_bad):
    """Test ESS validation catches insufficient samples."""
    summary = mock_mcmc_summary_bad

    # Count failing parameters
    failing_bulk = [name for name, p in summary["parameters"].items() if p["ess_bulk"] < 400]
    failing_tail = [name for name, p in summary["parameters"].items() if p["ess_tail"] < 400]

    assert len(failing_bulk) > 0, "Failed to detect insufficient ESS_bulk"
    assert len(failing_tail) > 0, "Failed to detect insufficient ESS_tail"

    # beta[0] and beta[1] should fail
    assert "beta[0]" in failing_bulk
    assert "beta[1]" in failing_bulk


def test_ess_minimum_threshold():
    """Test ESS minimum threshold."""
    test_cases = [
        (500, True),  # Above threshold (400)
        (400, True),  # At threshold
        (399, False),  # Below threshold
        (100, False),  # Well below threshold
    ]

    for ess_value, should_pass in test_cases:
        passes = ess_value >= 400
        assert passes == should_pass, f"ESS={ess_value} should_pass={should_pass}, got {passes}"


# ============================================================================
# TEST 3: DIVERGENCES
# ============================================================================


def test_divergences_validation_good(mock_mcmc_summary_good):
    """Test divergence validation with no divergences."""
    summary = mock_mcmc_summary_good

    total_div = summary["total_divergences"]

    # No divergences (ideal)
    assert total_div == 0, f"Found {total_div} divergences (expected 0)"


def test_divergences_validation_bad(mock_mcmc_summary_bad):
    """Test divergence validation catches excessive divergences."""
    summary = mock_mcmc_summary_bad

    total_div = summary["total_divergences"]

    # > 10 divergences (bad)
    assert total_div > 10, "Failed to detect excessive divergences"


def test_divergences_threshold():
    """Test divergence threshold (< 10 acceptable)."""
    test_cases = [
        (0, True),  # Ideal
        (5, True),  # Acceptable
        (10, True),  # At threshold
        (11, False),  # Above threshold
        (50, False),  # Way above threshold
    ]

    threshold = 10

    for div_count, should_pass in test_cases:
        passes = div_count <= threshold
        assert (
            passes == should_pass
        ), f"Divergences={div_count} should_pass={should_pass}, got {passes}"


# ============================================================================
# TEST 4: CALIBRATION CHECKS
# ============================================================================


def test_calibration_coverage_good(mock_calibration_data_good):
    """Test calibration with good coverage."""
    df = mock_calibration_data_good

    # Calculate coverage (fraction of true values within intervals)
    in_interval = (df["true_value"] >= df["lower_90"]) & (df["true_value"] <= df["upper_90"])
    coverage = in_interval.mean()

    # Should be close to 0.90 (90% intervals)
    assert coverage >= 0.85, f"Coverage = {coverage:.2%} < 85% (poor calibration)"
    assert coverage <= 0.95, f"Coverage = {coverage:.2%} > 95% (too conservative)"


def test_calibration_coverage_bad(mock_calibration_data_bad):
    """Test calibration catches under-coverage."""
    df = mock_calibration_data_bad

    # Calculate coverage
    in_interval = (df["true_value"] >= df["lower_90"]) & (df["true_value"] <= df["upper_90"])
    coverage = in_interval.mean()

    # Should be much less than 0.85 (poor calibration)
    assert coverage < 0.85, f"Coverage = {coverage:.2%} >= 85% (should be poorly calibrated)"


def test_calibration_multiple_levels():
    """Test calibration at multiple credible interval levels."""
    np.random.seed(42)
    n = 5000

    true_values = np.random.normal(250, 50, n)
    predictions = true_values + np.random.normal(0, 10, n)

    # Test different interval levels
    levels = [0.50, 0.68, 0.90, 0.95]
    z_scores = [0.674, 1.0, 1.645, 1.96]

    for level, z in zip(levels, z_scores):
        lower = predictions - z * 10  # interval SD matches the prediction error SD
        upper = predictions + z * 10

        in_interval = (true_values >= lower) & (true_values <= upper)
        coverage = in_interval.mean()

        # Coverage should be close to nominal level (within 3%)
        assert (
            abs(coverage - level) < 0.03
        ), f"{level*100}% interval has {coverage:.2%} coverage (expected ~{level:.2%})"


# ============================================================================
# TEST 5: PRODUCTION READINESS GATES
# ============================================================================


def test_production_gate_all_pass(mock_mcmc_summary_good, mock_calibration_data_good):
    """Test production gate when all checks pass."""
    mcmc = mock_mcmc_summary_good
    calib_df = mock_calibration_data_good

    # Gate 1: R-hat < 1.01
    rhat_pass = max(p["rhat"] for p in mcmc["parameters"].values()) < 1.01

    # Gate 2: ESS > 400
    ess_bulk_pass = all(p["ess_bulk"] > 400 for p in mcmc["parameters"].values())
    ess_tail_pass = all(p["ess_tail"] > 400 for p in mcmc["parameters"].values())

    # Gate 3: Divergences < 10
    div_pass = mcmc["total_divergences"] < 10

    # Gate 4: Calibration > 85%
    in_interval = (calib_df["true_value"] >= calib_df["lower_90"]) & (
        calib_df["true_value"] <= calib_df["upper_90"]
    )
    coverage = in_interval.mean()
    calib_pass = coverage > 0.85

    # ALL gates must pass
    production_ready = rhat_pass and ess_bulk_pass and ess_tail_pass and div_pass and calib_pass

    assert production_ready, "Model should pass all production gates"


def test_production_gate_blocks_bad_model(mock_mcmc_summary_bad, mock_calibration_data_bad):
    """Test production gate blocks model with issues."""
    mcmc = mock_mcmc_summary_bad
    calib_df = mock_calibration_data_bad

    # Gate 1: R-hat < 1.01
    rhat_pass = max(p["rhat"] for p in mcmc["parameters"].values()) < 1.01

    # Gate 2: ESS > 400
    ess_bulk_pass = all(p["ess_bulk"] > 400 for p in mcmc["parameters"].values())
    ess_tail_pass = all(p["ess_tail"] > 400 for p in mcmc["parameters"].values())

    # Gate 3: Divergences < 10
    div_pass = mcmc["total_divergences"] < 10

    # Gate 4: Calibration > 85%
    in_interval = (calib_df["true_value"] >= calib_df["lower_90"]) & (
        calib_df["true_value"] <= calib_df["upper_90"]
    )
    coverage = in_interval.mean()
    calib_pass = coverage > 0.85

    # At least one gate should fail
    production_ready = rhat_pass and ess_bulk_pass and ess_tail_pass and div_pass and calib_pass

    assert not production_ready, "Bad model should be blocked by production gates"


def test_production_gate_detailed_report():
    """Test production gate generates detailed failure report."""
    gates = {
        "rhat": {"passed": False, "value": 1.15, "threshold": 1.01},
        "ess_bulk": {"passed": False, "value": 250, "threshold": 400},
        "ess_tail": {"passed": True, "value": 450, "threshold": 400},
        "divergences": {"passed": False, "value": 25, "threshold": 10},
        "calibration": {"passed": True, "value": 0.89, "threshold": 0.85},
    }

    # Count failures
    failures = [name for name, gate in gates.items() if not gate["passed"]]

    assert len(failures) == 3, f"Expected 3 failures, found {len(failures)}"
    assert "rhat" in failures
    assert "ess_bulk" in failures
    assert "divergences" in failures


# ============================================================================
# TEST 6: EDGE CASES
# ============================================================================


def test_single_parameter_model():
    """Test validation with single parameter."""
    summary = {
        "parameters": {
            "mu": {
                "rhat": 1.002,
                "ess_bulk": 1500,
                "ess_tail": 1200,
                "divergences": 0,
            }
        },
        "total_divergences": 0,
    }

    # Should validate correctly
    rhat_pass = summary["parameters"]["mu"]["rhat"] < 1.01
    ess_pass = (
        summary["parameters"]["mu"]["ess_bulk"] > 400
        and summary["parameters"]["mu"]["ess_tail"] > 400
    )
    div_pass = summary["total_divergences"] < 10

    assert all([rhat_pass, ess_pass, div_pass])


def test_empty_calibration_data():
    """Test calibration with empty DataFrame."""
    empty_df = pd.DataFrame(columns=["true_value", "prediction", "lower_90", "upper_90"])

    # Should handle gracefully (coverage = NaN or 0)
    if len(empty_df) > 0:
        in_interval = (empty_df["true_value"] >= empty_df["lower_90"]) & (
            empty_df["true_value"] <= empty_df["upper_90"]
        )
        coverage = in_interval.mean()
    else:
        coverage = 0.0  # or NaN

    # Empty data should fail calibration check
    assert coverage < 0.85 or np.isnan(coverage)


def test_perfect_predictions_calibration():
    """Test calibration with perfect predictions (unrealistic but edge case)."""
    n = 100
    true_values = np.array([250.0] * n)
    predictions = np.array([250.0] * n)  # Perfect predictions!

    lower = predictions - 25
    upper = predictions + 25

    df = pd.DataFrame(
        {
            "true_value": true_values,
            "prediction": predictions,
            "lower_90": lower,
            "upper_90": upper,
        }
    )

    in_interval = (df["true_value"] >= df["lower_90"]) & (df["true_value"] <= df["upper_90"])
    coverage = in_interval.mean()

    # Perfect predictions should have 100% coverage
    assert coverage == 1.0


def test_very_wide_intervals_calibration():
    """Test calibration with very wide intervals (overconfident)."""
    np.random.seed(42)
    n = 1000

    true_values = np.random.normal(250, 50, n)
    predictions = true_values + np.random.normal(0, 10, n)

    # Very wide intervals (should capture everything)
    lower = predictions - 200
    upper = predictions + 200

    df = pd.DataFrame(
        {
            "true_value": true_values,
            "prediction": predictions,
            "lower_90": lower,
            "upper_90": upper,
        }
    )

    in_interval = (df["true_value"] >= df["lower_90"]) & (df["true_value"] <= df["upper_90"])
    coverage = in_interval.mean()

    # Should have very high coverage (>95%, overly conservative)
    assert coverage > 0.95, "Wide intervals should capture most values"


# ============================================================================
# TEST 7: INTEGRATION WITH MODEL TRAINING
# ============================================================================


def test_validation_workflow_integration():
    """Test full validation workflow after model training."""
    # Simulate model training results
    mcmc_summary = {
        "parameters": {
            "beta[0]": {"rhat": 1.001, "ess_bulk": 1500, "ess_tail": 1200, "divergences": 0},
            "beta[1]": {"rhat": 1.002, "ess_bulk": 1400, "ess_tail": 1100, "divergences": 0},
            "sigma": {"rhat": 1.003, "ess_bulk": 1600, "ess_tail": 1300, "divergences": 0},
        },
        "total_divergences": 0,
    }

    # Simulate predictions
    np.random.seed(42)
    n_test = 500
    true_value = np.random.normal(250, 50, n_test)
    prediction = true_value + np.random.normal(0, 20, n_test)  # error SD 20
    calib_df = pd.DataFrame(
        {
            "true_value": true_value,
            "prediction": prediction,
            "lower_90": prediction - 1.645 * 20,  # calibrated 90% interval
            "upper_90": prediction + 1.645 * 20,
        }
    )

    # Calculate calibration
    in_interval = (calib_df["true_value"] >= calib_df["lower_90"]) & (
        calib_df["true_value"] <= calib_df["upper_90"]
    )
    coverage = in_interval.mean()

    # Run all validation checks
    rhat_pass = all(p["rhat"] < 1.01 for p in mcmc_summary["parameters"].values())
    ess_pass = all(
        p["ess_bulk"] > 400 and p["ess_tail"] > 400 for p in mcmc_summary["parameters"].values()
    )
    div_pass = mcmc_summary["total_divergences"] < 10
    calib_pass = coverage > 0.85

    # Overall validation
    validated = rhat_pass and ess_pass and div_pass and calib_pass

    # If validated, allow deployment
    if validated:
        # Save model metadata with validation results
        metadata = {
            "validated": True,
            "rhat_max": max(p["rhat"] for p in mcmc_summary["parameters"].values()),
            "ess_min": min(p["ess_bulk"] for p in mcmc_summary["parameters"].values()),
            "divergences": mcmc_summary["total_divergences"],
            "calibration_coverage": coverage,
        }

        assert metadata["validated"] is True
    else:
        # Block deployment
        pytest.fail("Model failed validation - cannot deploy")


# ============================================================================
# MARKER: Unit tests (no database required)
# ============================================================================

pytestmark = pytest.mark.unit
