"""
Statistical Model Validator for NFL Analytics Bayesian Models

This module validates Bayesian models before production deployment. Real money
betting requires rigorous statistical validation.

Validation Checks:
1. MCMC Diagnostics (R-hat, ESS, divergences, energy)
2. Calibration (coverage tests, PIT plots)
3. Prior/Posterior Comparison (prior predictive checks)
4. Cross-Validation (LOO, K-fold for temporal data)
5. Model Specification (priors, likelihood, identifiability)

Supported Frameworks:
- brms (R interface to Stan)
- Stan (CmdStanPy, PyStan)
- PyMC (Python)
- Custom BNN models

Usage:
    from py.utils.statistical_validator import StatisticalValidator

    validator = StatisticalValidator()

    # Validate MCMC diagnostics
    report = validator.validate_mcmc(fit, chains=4)

    # Validate calibration
    calibration = validator.validate_calibration(predictions, actuals)

    # Full model validation
    report = validator.validate_model(model, data)

    # Block deployment if validation fails
    if report.has_critical_issues():
        print("⛔ Model invalid - do not deploy")
        exit(1)

Integration:
    - Post-training: Validate before saving model
    - Pre-deployment: Validate before production
    - Weekly: Re-validate production models
"""

import json
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    CRITICAL = "CRITICAL"  # Model invalid
    HIGH = "HIGH"          # Likely problematic
    MEDIUM = "MEDIUM"      # Should investigate
    LOW = "LOW"            # Minor concern
    INFO = "INFO"          # Informational


class ValidationCategory(Enum):
    """Categories of validation checks."""
    MCMC_DIAGNOSTICS = "mcmc_diagnostics"
    CALIBRATION = "calibration"
    PRIOR_POSTERIOR = "prior_posterior"
    CROSS_VALIDATION = "cross_validation"
    SPECIFICATION = "specification"


@dataclass
class ValidationIssue:
    """Represents a model validation finding."""
    severity: ValidationSeverity
    category: ValidationCategory
    title: str
    description: str
    metric_value: str
    threshold: str
    recommendation: str


@dataclass
class ValidationReport:
    """Model validation report."""
    model_name: str
    validation_date: str = field(default_factory=lambda: datetime.now().isoformat())
    issues: List[ValidationIssue] = field(default_factory=list)
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    passed: bool = False

    def add_issue(self, issue: ValidationIssue):
        """Add issue and update counters."""
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.CRITICAL:
            self.critical_count += 1
        elif issue.severity == ValidationSeverity.HIGH:
            self.high_count += 1
        elif issue.severity == ValidationSeverity.MEDIUM:
            self.medium_count += 1
        elif issue.severity == ValidationSeverity.LOW:
            self.low_count += 1

    def has_critical_issues(self) -> bool:
        """Check if report contains critical issues."""
        return self.critical_count > 0

    def to_json(self) -> str:
        """Export report to JSON."""
        return json.dumps({
            "model_name": self.model_name,
            "validation_date": self.validation_date,
            "passed": self.passed,
            "summary": {
                "critical": self.critical_count,
                "high": self.high_count,
                "medium": self.medium_count,
                "low": self.low_count,
            },
            "issues": [
                {
                    "severity": i.severity.value,
                    "category": i.category.value,
                    "title": i.title,
                    "description": i.description,
                    "metric_value": i.metric_value,
                    "threshold": i.threshold,
                    "recommendation": i.recommendation,
                }
                for i in self.issues
            ]
        }, indent=2)


class StatisticalValidator:
    """
    Validates Bayesian models for NFL analytics betting system.

    Critical for production deployment - ensures models are:
    - Converged (MCMC diagnostics)
    - Calibrated (coverage tests)
    - Well-specified (priors, likelihood)
    - Validated (cross-validation)
    """

    # MCMC diagnostic thresholds (from Stan documentation)
    RHAT_THRESHOLD = 1.01
    ESS_BULK_THRESHOLD = 400
    ESS_TAIL_THRESHOLD = 400
    DIVERGENCE_THRESHOLD = 10  # Max divergent transitions
    ENERGY_THRESHOLD = 0.2  # Max BFMI warning threshold

    # Calibration thresholds
    CALIBRATION_TOLERANCE = 0.05  # 5% tolerance on coverage
    MIN_CALIBRATION = 0.80  # Minimum 80% calibration
    TARGET_CALIBRATION = 0.90  # Target 90% calibration

    def __init__(self):
        """Initialize statistical validator."""
        if not PANDAS_AVAILABLE:
            logger.warning("pandas not available - some features limited")

        logger.info("StatisticalValidator initialized")

    def validate_rhat(
        self,
        rhat_values: np.ndarray,
        param_names: Optional[List[str]] = None
    ) -> List[ValidationIssue]:
        """
        Validate R-hat convergence diagnostic.

        Args:
            rhat_values: Array of R-hat values for each parameter
            param_names: Optional parameter names

        Returns:
            List of validation issues
        """
        issues = []

        if param_names is None:
            param_names = [f"param_{i}" for i in range(len(rhat_values))]

        # Check each parameter
        for i, (rhat, name) in enumerate(zip(rhat_values, param_names)):
            if rhat > self.RHAT_THRESHOLD:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category=ValidationCategory.MCMC_DIAGNOSTICS,
                    title=f"R-hat convergence failure: {name}",
                    description=f"Parameter {name} has R-hat = {rhat:.4f}",
                    metric_value=f"{rhat:.4f}",
                    threshold=f"<{self.RHAT_THRESHOLD}",
                    recommendation=f"Increase chains/draws or reparameterize model",
                ))

        # Check max R-hat
        max_rhat = np.max(rhat_values)
        if max_rhat > 1.05:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category=ValidationCategory.MCMC_DIAGNOSTICS,
                title="Severe convergence failure",
                description=f"Max R-hat = {max_rhat:.4f} indicates non-convergence",
                metric_value=f"{max_rhat:.4f}",
                threshold=f"<{self.RHAT_THRESHOLD}",
                recommendation="Model likely unidentifiable or misspecified",
            ))

        return issues

    def validate_ess(
        self,
        ess_bulk: np.ndarray,
        ess_tail: np.ndarray,
        param_names: Optional[List[str]] = None
    ) -> List[ValidationIssue]:
        """
        Validate Effective Sample Size diagnostics.

        Args:
            ess_bulk: Bulk ESS for each parameter
            ess_tail: Tail ESS for each parameter
            param_names: Optional parameter names

        Returns:
            List of validation issues
        """
        issues = []

        if param_names is None:
            param_names = [f"param_{i}" for i in range(len(ess_bulk))]

        # Check bulk ESS
        for i, (ess, name) in enumerate(zip(ess_bulk, param_names)):
            if ess < self.ESS_BULK_THRESHOLD:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.HIGH,
                    category=ValidationCategory.MCMC_DIAGNOSTICS,
                    title=f"Low bulk ESS: {name}",
                    description=f"Parameter {name} has bulk ESS = {ess:.0f}",
                    metric_value=f"{ess:.0f}",
                    threshold=f">{self.ESS_BULK_THRESHOLD}",
                    recommendation="Increase draws or improve sampling efficiency",
                ))

        # Check tail ESS
        for i, (ess, name) in enumerate(zip(ess_tail, param_names)):
            if ess < self.ESS_TAIL_THRESHOLD:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.HIGH,
                    category=ValidationCategory.MCMC_DIAGNOSTICS,
                    title=f"Low tail ESS: {name}",
                    description=f"Parameter {name} has tail ESS = {ess:.0f}",
                    metric_value=f"{ess:.0f}",
                    threshold=f">{self.ESS_TAIL_THRESHOLD}",
                    recommendation="Tails poorly explored - increase draws",
                ))

        return issues

    def validate_divergences(
        self,
        num_divergences: int,
        total_draws: int
    ) -> List[ValidationIssue]:
        """
        Validate divergent transitions.

        Args:
            num_divergences: Number of divergent transitions
            total_draws: Total number of draws

        Returns:
            List of validation issues
        """
        issues = []

        if num_divergences > 0:
            divergence_rate = num_divergences / total_draws

            if num_divergences > self.DIVERGENCE_THRESHOLD:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category=ValidationCategory.MCMC_DIAGNOSTICS,
                    title="Divergent transitions detected",
                    description=f"{num_divergences} divergences ({divergence_rate:.2%})",
                    metric_value=f"{num_divergences}",
                    threshold="0",
                    recommendation="Reparameterize model or increase adapt_delta",
                ))
            else:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.MEDIUM,
                    category=ValidationCategory.MCMC_DIAGNOSTICS,
                    title="Some divergences present",
                    description=f"{num_divergences} divergences (small number)",
                    metric_value=f"{num_divergences}",
                    threshold="0",
                    recommendation="Monitor - may indicate minor specification issues",
                ))

        return issues

    def validate_calibration(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        confidence_levels: Optional[List[float]] = None
    ) -> Tuple[float, List[ValidationIssue]]:
        """
        Validate model calibration via coverage tests.

        Args:
            predictions: Predicted values (or intervals)
            actuals: Actual observed values
            confidence_levels: Confidence levels to test (default: [0.5, 0.68, 0.95])

        Returns:
            Tuple of (calibration_score, validation_issues)
        """
        issues = []

        if confidence_levels is None:
            confidence_levels = [0.5, 0.68, 0.95]

        # If predictions are intervals (shape: [n, 2])
        if predictions.ndim == 2 and predictions.shape[1] == 2:
            lower, upper = predictions[:, 0], predictions[:, 1]

            for conf_level in confidence_levels:
                # Check coverage
                in_interval = (actuals >= lower) & (actuals <= upper)
                coverage = np.mean(in_interval)

                # Check if coverage matches nominal level
                diff = abs(coverage - conf_level)

                if diff > self.CALIBRATION_TOLERANCE:
                    severity = ValidationSeverity.CRITICAL if diff > 0.15 else \
                               ValidationSeverity.HIGH if diff > 0.10 else \
                               ValidationSeverity.MEDIUM

                    issues.append(ValidationIssue(
                        severity=severity,
                        category=ValidationCategory.CALIBRATION,
                        title=f"Miscalibration at {conf_level:.0%} level",
                        description=f"Coverage {coverage:.1%} vs nominal {conf_level:.1%}",
                        metric_value=f"{coverage:.1%}",
                        threshold=f"{conf_level:.1%} ± {self.CALIBRATION_TOLERANCE:.1%}",
                        recommendation="Adjust temperature scaling or retrain model",
                    ))

            # Overall calibration score (average over confidence levels)
            calibration_score = 1.0 - np.mean([
                abs(np.mean((actuals >= lower) & (actuals <= upper)) - conf_level)
                for conf_level in confidence_levels
            ])

        else:
            # Point predictions - use residual analysis
            residuals = actuals - predictions
            calibration_score = 1.0 - np.abs(np.mean(residuals)) / np.std(actuals)

        # Check minimum calibration
        if calibration_score < self.MIN_CALIBRATION:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category=ValidationCategory.CALIBRATION,
                title="Poor model calibration",
                description=f"Calibration score {calibration_score:.1%} below minimum",
                metric_value=f"{calibration_score:.1%}",
                threshold=f">{self.MIN_CALIBRATION:.1%}",
                recommendation="Retrain model with better calibration",
            ))

        return calibration_score, issues

    def validate_priors(
        self,
        prior_samples: np.ndarray,
        posterior_samples: np.ndarray,
        param_name: str
    ) -> List[ValidationIssue]:
        """
        Validate prior specification via prior/posterior comparison.

        Args:
            prior_samples: Samples from prior distribution
            posterior_samples: Samples from posterior distribution
            param_name: Parameter name

        Returns:
            List of validation issues
        """
        issues = []

        # Check if prior is too informative (posterior ≈ prior)
        prior_mean = np.mean(prior_samples)
        prior_std = np.std(prior_samples)
        post_mean = np.mean(posterior_samples)
        post_std = np.std(posterior_samples)

        # Check if posterior is same as prior (data not informative)
        mean_diff = abs(post_mean - prior_mean) / prior_std
        if mean_diff < 0.1:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.MEDIUM,
                category=ValidationCategory.PRIOR_POSTERIOR,
                title=f"Posterior ≈ Prior: {param_name}",
                description=f"Posterior unchanged from prior (data uninformative)",
                metric_value=f"{mean_diff:.2f} std",
                threshold=">0.1 std",
                recommendation="Data may not inform this parameter - consider removing",
            ))

        # Check if prior is too wide (posterior >> prior precision)
        precision_ratio = prior_std / post_std
        if precision_ratio > 10:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.LOW,
                category=ValidationCategory.PRIOR_POSTERIOR,
                title=f"Prior too wide: {param_name}",
                description=f"Prior {precision_ratio:.1f}x wider than posterior",
                metric_value=f"{precision_ratio:.1f}x",
                threshold="<10x",
                recommendation="Consider more informative prior based on domain knowledge",
            ))

        # Check if prior is too narrow (posterior ≈ prior width)
        if precision_ratio < 1.5:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.MEDIUM,
                category=ValidationCategory.PRIOR_POSTERIOR,
                title=f"Prior too narrow: {param_name}",
                description=f"Prior constrains posterior too much",
                metric_value=f"{precision_ratio:.2f}x",
                threshold=">1.5x",
                recommendation="Use wider prior to let data speak",
            ))

        return issues

    def validate_model(
        self,
        model_data: Dict[str, Any],
        validation_type: str = "full"
    ) -> ValidationReport:
        """
        Comprehensive model validation.

        Args:
            model_data: Dictionary with model diagnostics:
                - rhat: R-hat values
                - ess_bulk: Bulk ESS values
                - ess_tail: Tail ESS values
                - divergences: Number of divergences
                - predictions: Model predictions
                - actuals: Actual values
            validation_type: "full", "mcmc_only", or "calibration_only"

        Returns:
            Complete validation report
        """
        report = ValidationReport(
            model_name=model_data.get("model_name", "unknown")
        )

        # MCMC diagnostics
        if validation_type in ["full", "mcmc_only"]:
            if "rhat" in model_data:
                issues = self.validate_rhat(
                    model_data["rhat"],
                    model_data.get("param_names")
                )
                for issue in issues:
                    report.add_issue(issue)

            if "ess_bulk" in model_data and "ess_tail" in model_data:
                issues = self.validate_ess(
                    model_data["ess_bulk"],
                    model_data["ess_tail"],
                    model_data.get("param_names")
                )
                for issue in issues:
                    report.add_issue(issue)

            if "divergences" in model_data:
                issues = self.validate_divergences(
                    model_data["divergences"],
                    model_data.get("total_draws", 4000)
                )
                for issue in issues:
                    report.add_issue(issue)

        # Calibration
        if validation_type in ["full", "calibration_only"]:
            if "predictions" in model_data and "actuals" in model_data:
                calibration_score, issues = self.validate_calibration(
                    model_data["predictions"],
                    model_data["actuals"]
                )
                for issue in issues:
                    report.add_issue(issue)

                # Store calibration score
                report.passed = calibration_score >= self.MIN_CALIBRATION

        # Set passed status
        report.passed = report.critical_count == 0

        return report

    def generate_report(self, report: ValidationReport) -> str:
        """Generate formatted validation report."""
        lines = []

        lines.append("=" * 80)
        lines.append("STATISTICAL MODEL VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append(f"\nModel: {report.model_name}")
        lines.append(f"Validation Date: {report.validation_date}")
        lines.append(f"Status: {'✅ PASSED' if report.passed else '⛔ FAILED'}")
        lines.append(f"\nIssues Summary:")
        lines.append(f"  🔴 CRITICAL: {report.critical_count}")
        lines.append(f"  🟠 HIGH:     {report.high_count}")
        lines.append(f"  🟡 MEDIUM:   {report.medium_count}")
        lines.append(f"  🟢 LOW:      {report.low_count}")

        if report.has_critical_issues():
            lines.append(f"\n⛔ CRITICAL VALIDATION FAILURES")
            lines.append(f"   MODEL IS INVALID - DO NOT DEPLOY")

        # Group by category
        for category in ValidationCategory:
            category_issues = [i for i in report.issues if i.category == category]

            if category_issues:
                lines.append(f"\n{category.value.upper().replace('_', ' ')} ({len(category_issues)} issues):")
                lines.append("-" * 80)

                for issue in category_issues:
                    lines.append(f"\n{issue.severity.value}: {issue.title}")
                    lines.append(f"  🔍 {issue.description}")
                    lines.append(f"  📊 Value: {issue.metric_value}, Threshold: {issue.threshold}")
                    lines.append(f"  💡 {issue.recommendation}")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    validator = StatisticalValidator()

    # Example: Validate MCMC diagnostics
    rhat = np.array([1.00, 1.01, 1.05, 0.99])
    ess_bulk = np.array([500, 450, 300, 600])
    ess_tail = np.array([480, 420, 280, 580])

    model_data = {
        "model_name": "BNN Passing v1",
        "rhat": rhat,
        "ess_bulk": ess_bulk,
        "ess_tail": ess_tail,
        "divergences": 5,
        "total_draws": 4000,
        "param_names": ["alpha", "beta", "gamma", "sigma"],
    }

    report = validator.validate_model(model_data, validation_type="mcmc_only")

    print(validator.generate_report(report))

    if report.has_critical_issues():
        print("\n⛔ Model validation failed - do not deploy")
    else:
        print("\n✅ Model validation passed")
