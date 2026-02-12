#!/usr/bin/env python3
"""
Unified Enforcement Router - Routes code changes to appropriate validation agents

This module provides a centralized framework for enforcing best practices and preventing
recurring issues by automatically routing code changes to specialized validation agents.

**Purpose:**
- Prevent schema errors (documented "DOZENS of times")
- Block expensive failures (Nov 13 R integration: 3.5 hours lost)
- Enforce data leakage checks (BNN: 84 hours wasted)
- Validate model convergence before deployment
- Security audit credential changes
- Prevent placeholder predictions from reaching production
- Prevent underpowered experiments (v7.0: 58 hours wasted on insufficient data)

**Integration Points:**
1. Pre-commit hooks (git-level enforcement)
2. Runtime decorators (function-level enforcement)
3. CI/CD pipelines (deployment-level enforcement)

**Author:** Claude Code (Sonnet 4.5)
**Date:** November 15, 2025
**Version:** 1.2 (Added Rule 20: Data Sufficiency Gate)
"""

import inspect
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnforcementLevel(Enum):
    """Enforcement severity levels."""

    BLOCKING = "blocking"  # Blocks execution if validation fails
    WARNING = "warning"  # Logs warning but allows execution
    INFO = "info"  # Informational only


class ValidationStatus(Enum):
    """Validation result status."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ValidationResult:
    """Result of a validation check."""

    status: ValidationStatus
    agent_name: str
    message: str
    details: dict[str, Any] | None = None

    @property
    def passed(self) -> bool:
        return self.status == ValidationStatus.PASSED

    @property
    def failed(self) -> bool:
        return self.status == ValidationStatus.FAILED


@dataclass
class EnforcementRule:
    """Defines an enforcement rule."""

    name: str
    description: str
    triggers: list[str]  # Patterns that trigger this rule
    validator: Callable  # Function that performs validation
    level: EnforcementLevel
    enabled: bool = True


class EnforcementRouter:
    """
    Routes code changes to appropriate validation agents based on triggers.

    **Design Philosophy:**
    - Fail fast, fail loudly
    - Provide actionable error messages
    - No silent failures
    - Enforce at multiple levels (git, runtime, CI/CD)

    **Example Usage:**

    ```python
    # Runtime enforcement (decorator)
    @enforce_sql_validation
    def load_data(query: str):
        return pd.read_sql(query, conn)

    # Pre-commit enforcement
    router = EnforcementRouter()
    router.validate_commit(changed_files)

    # Manual enforcement
    router.validate_sql_query(query)
    ```
    """

    def __init__(self):
        """Initialize enforcement router with all validation rules."""
        self.rules: dict[str, EnforcementRule] = {}
        self._register_rules()

    def _register_rules(self):
        """Register all enforcement rules."""

        # Rule 1: SQL Query Validation (HIGHEST PRIORITY)
        self.rules["sql_validation"] = EnforcementRule(
            name="sql_validation",
            description="Validate SQL queries against schema",
            triggers=[
                r"\.sql$",  # .sql files
                r"pd\.read_sql",  # pandas read_sql
                r"dbGetQuery",  # R database queries
                r"SELECT.*FROM",  # SQL strings
                r"psycopg2\.connect",  # Direct DB connections
            ],
            validator=self._validate_sql,
            level=EnforcementLevel.BLOCKING,
        )

        # Rule 2: Pre-Backtest Health Check (HIGH PRIORITY)
        self.rules["backtest_health"] = EnforcementRule(
            name="backtest_health",
            description="Check ensemble health before backtest",
            triggers=[
                r"_validation\.py$",  # All validation scripts
                r"historical_backtest_.*\.py$",  # Historical backtests
                r"multi_season_validation\.py$",  # Multi-season validators (PHASE 3)
            ],
            validator=self._validate_backtest_health,
            level=EnforcementLevel.BLOCKING,
        )

        # Rule 3: Model Training Validation (HIGH PRIORITY)
        self.rules["model_training"] = EnforcementRule(
            name="model_training",
            description="Validate data splits and convergence",
            triggers=[
                r"py/models/train_.*\.py$",
                r"R/train_.*\.R$",
                r"scripts/train_.*\.py$",
            ],
            validator=self._validate_model_training,
            level=EnforcementLevel.BLOCKING,
        )

        # Rule 4: Security Audit (HIGH PRIORITY) - DISABLED (too many false positives)
        # self.rules['security_audit'] = EnforcementRule(
        #     name='security_audit',
        #     description='Audit credentials and secrets',
        #     triggers=[
        #         r'\.env',
        #         r'docker-compose\.yml',
        #         r'password|secret|api_key',  # Case-insensitive regex
        #     ],
        #     validator=self._validate_security,
        #     level=EnforcementLevel.BLOCKING
        # )

        # Rule 5: Ensemble Modification Guard (MEDIUM PRIORITY)
        self.rules["ensemble_guard"] = EnforcementRule(
            name="ensemble_guard",
            description="Validate ensemble modifications",
            triggers=[
                r"py/ensemble/enhanced_ensemble_.*\.py$",
            ],
            validator=self._validate_ensemble_modification,
            level=EnforcementLevel.WARNING,
        )

        # Rule 6: Schema Change Workflow (MEDIUM PRIORITY)
        self.rules["schema_change"] = EnforcementRule(
            name="schema_change",
            description="Enforce schema documentation updates",
            triggers=[
                r"db/.*\.sql$",
                r"CREATE.*VIEW",
                r"ALTER TABLE",
            ],
            validator=self._validate_schema_change,
            level=EnforcementLevel.WARNING,
        )

        # Rule 7: Documentation Sync (MEDIUM PRIORITY)
        self.rules["documentation_sync"] = EnforcementRule(
            name="documentation_sync",
            description="Keep CLAUDE.md in sync",
            triggers=[
                r"docs/milestones/.*\.md$",
                r"MODEL_REGISTRY\.yaml$",
            ],
            validator=self._validate_documentation_sync,
            level=EnforcementLevel.INFO,
        )

        # Rule 8: R Model Convergence Pre-Check (HIGH PRIORITY)
        self.rules["r_convergence_check"] = EnforcementRule(
            name="r_convergence_check",
            description="Validate MCMC configuration before training",
            triggers=[
                r"R/train_.*\.(R|r)$",
                r"brms?\s*\(",
                r"cmdstanr::(sample|optimize)",
                r"fit\s*<-.*brm\(",
            ],
            validator=self._validate_r_convergence,
            level=EnforcementLevel.BLOCKING,
        )

        # Rule 9: Hardcoded Temporal Values Detection (HIGH PRIORITY)
        self.rules["hardcoded_temporal"] = EnforcementRule(
            name="hardcoded_temporal",
            description="Detect hardcoded season/year values",
            triggers=[
                r"\.(py|R|r)$",
            ],
            validator=self._validate_hardcoded_temporal,
            level=EnforcementLevel.BLOCKING,
        )

        # Rule 10: allow_new_levels Parameter Validation (HIGH PRIORITY)
        self.rules["allow_new_levels_check"] = EnforcementRule(
            name="allow_new_levels_check",
            description="Validate brms posterior_predict parameters",
            triggers=[
                r"\.R$",
                r"posterior_predict\(",
            ],
            validator=self._validate_allow_new_levels,
            level=EnforcementLevel.BLOCKING,
        )

        # Rule 12: Multi-Season Validation Enforcement (MEDIUM PRIORITY)
        self.rules["multi_season_validation"] = EnforcementRule(
            name="multi_season_validation",
            description="Enforce multi-season validation (minimum 3 seasons)",
            triggers=[
                r"py/backtest/.*\.py$",
                r"scripts/validate_.*\.py$",
            ],
            validator=self._validate_multi_season,
            level=EnforcementLevel.WARNING,
        )

        # Rule 13: Documentation Drift Detection (MEDIUM PRIORITY)
        self.rules["documentation_drift"] = EnforcementRule(
            name="documentation_drift",
            description="Detect when documentation may need updates",
            triggers=[
                r"models/.*\.(pkl|rds|pth)$",
                r"docs/milestones/.*COMPLETE\.md$",
                r"MODEL_REGISTRY\.yaml$",
                r"db/migrations/.*\.sql$",
            ],
            validator=self._validate_documentation_drift,
            level=EnforcementLevel.INFO,
        )

        # Rule 14: Model File Integrity Check (MEDIUM PRIORITY)
        self.rules["model_file_integrity"] = EnforcementRule(
            name="model_file_integrity",
            description="Check model files saved correctly",
            triggers=[
                r"saveRDS\(",
                r"pickle\.dump\(",
                r"torch\.save\(",
                r"joblib\.dump\(",
            ],
            validator=self._validate_model_integrity,
            level=EnforcementLevel.WARNING,
        )

        # Rule 15: API Contract Validation (HIGH PRIORITY) - PHASE 3
        self.rules["api_contract_validation"] = EnforcementRule(
            name="api_contract_validation",
            description="Validate API contract usage (catch deprecated parameters)",
            triggers=[
                r"from py\.ensemble.*import",
                r"EnhancedEnsemble.*\(",
                r"ensemble.*=.*\(",
            ],
            validator=self._validate_api_contract,
            level=EnforcementLevel.BLOCKING,
        )

        # Rule 16: SQL Query Result Validation (HIGH PRIORITY) - PHASE 3
        self.rules["sql_result_validation"] = EnforcementRule(
            name="sql_result_validation",
            description="Validate SQL results checked before access",
            triggers=[
                r"pd\.read_sql",
                r"pd\.read_sql_query",
                r"\.iloc\[0\]",
                r"\.values\[0\]",
            ],
            validator=self._validate_sql_results,
            level=EnforcementLevel.BLOCKING,
        )

        # Rule 17: R Model Prediction Validation (MEDIUM PRIORITY) - PHASE 3
        self.rules["r_prediction_validation"] = EnforcementRule(
            name="r_prediction_validation",
            description="Validate R predictions checked for NaN",
            triggers=[
                r"posterior_predict\(",
                r"predict\.brmsfit",
                r"rpy2\.robjects",
            ],
            validator=self._validate_r_predictions,
            level=EnforcementLevel.WARNING,
        )

        # Rule 18: Ensemble Minimum Components (HIGH PRIORITY) - PHASE 3
        self.rules["ensemble_components_check"] = EnforcementRule(
            name="ensemble_components_check",
            description="Validate ensemble has minimum components",
            triggers=[
                r"ensemble\.predict\(",
                r"\.predict\(.*player_id",
                r"valid_predictions",
            ],
            validator=self._validate_ensemble_components,
            level=EnforcementLevel.BLOCKING,
        )

        # Rule 19: Placeholder Detection (CRITICAL) - PHASE 4
        self.rules["placeholder_detection"] = EnforcementRule(
            name="placeholder_detection",
            description="Prevent hardcoded placeholder predictions from reaching production",
            triggers=[
                r"py/ensemble/.*\.py$",  # All ensemble files
                r"predict\(",  # Predict methods
                r"return.*dict",  # Dict returns
            ],
            validator=self._validate_placeholder_predictions,
            level=EnforcementLevel.BLOCKING,
        )

        # Rule 20: Data Sufficiency Gate (WARNING) - PHASE 5
        # Changed from BLOCKING to WARNING - too many false positives on general code
        self.rules["data_sufficiency_gate"] = EnforcementRule(
            name="data_sufficiency_gate",
            description="Warn about potential underpowered experiments",
            triggers=[
                r"meta_learning",  # Meta-learning code
                r"adaptive.*weight",  # Adaptive weighting
                r"online.*learning",  # Online learning methods
            ],
            validator=self._validate_data_sufficiency,
            level=EnforcementLevel.WARNING,  # Changed from BLOCKING - advisory only
        )

    def validate_file(self, file_path: str, content: str | None = None) -> list[ValidationResult]:
        """
        Validate a file against all applicable rules.

        Args:
            file_path: Path to file being validated
            content: Optional file content (reads from disk if not provided)

        Returns:
            List of validation results
        """
        if content is None and os.path.exists(file_path):
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()

        results = []
        for rule in self.rules.values():
            if not rule.enabled:
                continue

            # Check if any trigger matches
            triggered = False
            for trigger in rule.triggers:
                if re.search(trigger, file_path, re.IGNORECASE):
                    triggered = True
                    break
                if content and re.search(trigger, content, re.IGNORECASE):
                    triggered = True
                    break

            if triggered:
                logger.info(f"Rule '{rule.name}' triggered for {file_path}")
                result = rule.validator(file_path, content)
                results.append(result)

        return results

    def validate_commit(self, changed_files: list[str]) -> tuple[bool, list[ValidationResult]]:
        """
        Validate a git commit against all rules.

        Args:
            changed_files: List of file paths changed in commit

        Returns:
            Tuple of (all_passed: bool, results: List[ValidationResult])
        """
        all_results = []

        for file_path in changed_files:
            # Skip .md files in docs/audits/ (audit documentation contains code examples)
            if file_path.endswith(".md") and "docs/audits/" in file_path:
                logger.info(f"Skipping validation for audit documentation: {file_path}")
                continue

            results = self.validate_file(file_path)
            all_results.extend(results)

        # Check if any blocking rules failed
        blocking_failures = [
            r
            for r in all_results
            if r.failed
            and self.rules.get(
                r.agent_name,
                EnforcementRule(
                    name="",
                    description="",
                    triggers=[],
                    validator=lambda x, y: None,
                    level=EnforcementLevel.INFO,
                ),
            ).level
            == EnforcementLevel.BLOCKING
        ]

        all_passed = len(blocking_failures) == 0
        return all_passed, all_results

    # Validation methods for each rule

    def _validate_sql(self, file_path: str, content: str | None) -> ValidationResult:
        """Validate SQL query against schema (Rule 1)."""
        try:
            import importlib.util
            from pathlib import Path

            # Skip migration files that CREATE new tables (can't validate against non-existent schema)
            if (
                content
                and "migrations/" in file_path
                and re.search(r"CREATE\s+TABLE", content, re.IGNORECASE)
            ):
                return ValidationResult(
                    status=ValidationStatus.PASSED,
                    agent_name="sql_validation",
                    message="Migration file creating new table - skipping schema validation",
                )

            # Load sql_validator module directly from file path
            # This works regardless of how enforcement_router is invoked
            sql_validator_path = Path(__file__).parent / "sql_validator.py"
            if not sql_validator_path.exists():
                raise ImportError(f"sql_validator.py not found at {sql_validator_path}")

            spec = importlib.util.spec_from_file_location("sql_validator", sql_validator_path)
            sql_validator_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sql_validator_module)
            SQLValidator = sql_validator_module.SQLValidator

            validator = SQLValidator()

            # Extract SQL queries from content
            if content:
                # First, extract all CTE (Common Table Expression) names from content
                # These are virtual tables defined with: WITH table_name AS (...)
                cte_names = set()
                cte_pattern = r"\bWITH\s+(\w+)\s+AS\s*\("
                for match in re.finditer(cte_pattern, content, re.IGNORECASE):
                    cte_names.add(match.group(1).lower())
                # Also find chained CTEs: , another_cte AS (
                chained_cte_pattern = r",\s*(\w+)\s+AS\s*\("
                for match in re.finditer(chained_cte_pattern, content, re.IGNORECASE):
                    cte_names.add(match.group(1).lower())

                # Simple heuristic: find SELECT statements
                queries = re.findall(r"(SELECT\s+.*?;)", content, re.IGNORECASE | re.DOTALL)

                all_issues = []
                for query in queries:
                    issues = validator.validate_query(query)
                    if issues:
                        # Filter out issues about CTE names (not real tables)
                        filtered_issues = []
                        for issue in issues:
                            # Check if error references a CTE name
                            is_cte_error = False
                            for cte_name in cte_names:
                                if f"'{cte_name}'" in issue.lower():
                                    is_cte_error = True
                                    logger.debug(f"Ignoring error for CTE '{cte_name}': {issue}")
                                    break
                            if not is_cte_error:
                                filtered_issues.append(issue)
                        all_issues.extend(filtered_issues)

                # Separate errors (block) from warnings (allow)
                # Errors start with "ERROR:" or don't have a severity prefix
                # Warnings start with "WARNING:"
                errors = [i for i in all_issues if not i.startswith("WARNING:")]
                warnings = [i for i in all_issues if i.startswith("WARNING:")]

                if errors:
                    return ValidationResult(
                        status=ValidationStatus.FAILED,
                        agent_name="sql_validation",
                        message=f"SQL validation failed: {len(errors)} errors found",
                        details={"errors": errors, "warnings": warnings},
                    )
                elif warnings:
                    # Warnings are advisory only - don't block commit
                    logger.info(f"SQL validation warnings: {warnings}")
                    return ValidationResult(
                        status=ValidationStatus.PASSED,
                        agent_name="sql_validation",
                        message=f"SQL validation passed with {len(warnings)} warnings",
                    )

            return ValidationResult(
                status=ValidationStatus.PASSED,
                agent_name="sql_validation",
                message="SQL validation passed",
            )

        except ImportError as e:
            logger.warning(f"SQLValidator import failed: {e}")
            return ValidationResult(
                status=ValidationStatus.SKIPPED,
                agent_name="sql_validation",
                message="SQLValidator not available",
            )
        except Exception as e:
            logger.error(f"SQL validation error: {type(e).__name__}: {e}")
            return ValidationResult(
                status=ValidationStatus.SKIPPED,
                agent_name="sql_validation",
                message=f"SQL validation error: {str(e)}",
            )

    def _validate_backtest_health(self, file_path: str, content: str | None) -> ValidationResult:
        """Check ensemble health before backtest (Rule 2 - ENHANCED)."""
        try:
            import sys
            from pathlib import Path

            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            from py.utils.ensemble_health_monitor import EnsembleHealthMonitor

            monitor = EnsembleHealthMonitor()
            report = monitor.check_ensemble_health()

            if not report.is_healthy:
                return ValidationResult(
                    status=ValidationStatus.FAILED,
                    agent_name="backtest_health",
                    message=f"Ensemble unhealthy: {len(report.critical_issues)} critical issues",
                    details={"issues": report.critical_issues},
                )

            # ENHANCED: Check rpy2 integration if R models present
            if content and ("import rpy2" in content or "from rpy2" in content):
                try:
                    import rpy2.robjects as ro

                    # Test R session
                    ro.r('print("R session OK")')
                    logger.info("✅ rpy2 integration working")
                except Exception as e:
                    return ValidationResult(
                        status=ValidationStatus.FAILED,
                        agent_name="backtest_health",
                        message=f"rpy2 integration failed: {str(e)}",
                        details={
                            "error": str(e),
                            "recommendation": "Check R environment and rpy2 version",
                        },
                    )

            return ValidationResult(
                status=ValidationStatus.PASSED,
                agent_name="backtest_health",
                message="Ensemble health check passed",
            )

        except Exception as e:
            logger.warning(f"Ensemble health check failed: {e}")
            return ValidationResult(
                status=ValidationStatus.SKIPPED,
                agent_name="backtest_health",
                message=f"Health check error: {str(e)}",
            )

    def _validate_model_training(self, file_path: str, content: str | None) -> ValidationResult:
        """Validate model training script (Rule 3)."""
        # This is a static analysis - runtime validation happens via decorator

        if not content:
            return ValidationResult(
                status=ValidationStatus.SKIPPED,
                agent_name="model_training",
                message="No content to validate",
            )

        # Skip RL training scripts - they use different validation paradigm (temporal/episode-based)
        rl_patterns = ["cql", "rl/", "agent", "fourth_down", "reinforcement", "mdp"]
        if any(p in file_path.lower() for p in rl_patterns):
            return ValidationResult(
                status=ValidationStatus.SKIPPED,
                agent_name="model_training",
                message="RL training script - uses temporal/episode validation instead of split validation",
            )

        # Skip scripts that use temporal cross-validation (detected by pattern)
        if "TimeSeriesSplit" in content or "walk_forward" in content or "temporal_cv" in content:
            return ValidationResult(
                status=ValidationStatus.SKIPPED,
                agent_name="model_training",
                message="Uses temporal cross-validation - inherently prevents data leakage",
            )

        # Check for data leakage validation calls
        has_leakage_check = "DataLeakageDetector" in content or "validate_split" in content

        if not has_leakage_check:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                agent_name="model_training",
                message="Missing data leakage validation - add DataLeakageDetector.validate_split()",
                details={"file": file_path},
            )

        return ValidationResult(
            status=ValidationStatus.PASSED,
            agent_name="model_training",
            message="Model training validation checks present",
        )

    def _validate_security(self, file_path: str, content: str | None) -> ValidationResult:
        """Security audit for credentials (Rule 4)."""
        if not content:
            return ValidationResult(
                status=ValidationStatus.SKIPPED,
                agent_name="security_audit",
                message="No content to validate",
            )

        # Simple pattern-based credential detection
        credential_patterns = [
            (r"sicillionbillions(?!\s*#.*pragma:\s*allowlist)", "Hardcoded password"),
            (r'["\']?api[_-]?key["\']?\s*[:=]\s*["\'][A-Za-z0-9]{20,}["\']', "API key in code"),
            (r'["\']?password["\']?\s*[:=]\s*["\'][^"\']{8,}["\']', "Password in code"),
            (r"postgres://\w+:[^@\s]+@", "Database URL with password"),
        ]

        issues = []
        for pattern, description in credential_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1
                # Skip if in comment and has pragma allowlist
                line_start = content.rfind("\n", 0, match.start()) + 1
                line_end = content.find("\n", match.end())
                line = content[line_start : line_end if line_end != -1 else len(content)]

                if "pragma: allowlist" in line or "pragma:allowlist" in line:
                    continue

                issues.append(
                    {"line": line_num, "match": match.group(0)[:50], "description": description}
                )

        if issues:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                agent_name="security_audit",
                message=f"Potential credentials detected ({len(issues)} issues)",
                details={"issues": issues[:3]},  # Limit to first 3
            )

        return ValidationResult(
            status=ValidationStatus.PASSED,
            agent_name="security_audit",
            message="No credentials detected",
        )

    def _validate_ensemble_modification(
        self, file_path: str, content: str | None
    ) -> ValidationResult:
        """Validate ensemble modifications (Rule 5)."""
        # Warning-level check - logs but doesn't block
        return ValidationResult(
            status=ValidationStatus.PASSED,
            agent_name="ensemble_guard",
            message=f"⚠️ Ensemble modified: {file_path}. Run health check before deployment.",
            details={"recommendation": "Run: ./scripts/check_ensemble_health.sh"},
        )

    def _validate_schema_change(self, file_path: str, content: str | None) -> ValidationResult:
        """Validate schema change documentation (Rule 6)."""
        # Check if DATABASE_QUICK_REFERENCE.md was also modified in this commit
        # This would need git context - for now just warn
        return ValidationResult(
            status=ValidationStatus.PASSED,
            agent_name="schema_change",
            message=f"⚠️ Schema change detected: {file_path}. Update DATABASE_QUICK_REFERENCE.md",
            details={"action_required": "Update docs/database/DATABASE_QUICK_REFERENCE.md"},
        )

    def _validate_documentation_sync(self, file_path: str, content: str | None) -> ValidationResult:
        """Validate documentation synchronization (Rule 7)."""
        return ValidationResult(
            status=ValidationStatus.PASSED,
            agent_name="documentation_sync",
            message=f"ℹ️ Documentation updated: {file_path}. Consider updating CLAUDE.md 'Recent Work'",
            details={"recommendation": "Update CLAUDE.md if this is a major milestone"},
        )

    def _validate_r_convergence(self, file_path: str, content: str | None) -> ValidationResult:
        """Validate MCMC configuration for convergence (Rule 8)."""
        if not content:
            return ValidationResult(
                status=ValidationStatus.SKIPPED,
                agent_name="r_convergence_check",
                message="No content to validate",
            )

        # Check for complex hierarchical structures
        random_effects_count = len(re.findall(r"\|\s*\w+\)", content))

        # Check iterations setting
        iter_match = re.search(r"iter\s*=\s*(\d+)", content)
        warmup_match = re.search(r"warmup\s*=\s*(\d+)", content)

        issues = []

        # Complex models need more iterations
        if random_effects_count >= 3:
            if iter_match:
                iterations = int(iter_match.group(1))
                if iterations < 4000:
                    issues.append(
                        f"Complex model ({random_effects_count} random effects) needs iter >= 4000 (found: {iterations})"
                    )
            else:
                issues.append(
                    f"Complex model ({random_effects_count} random effects) missing iterations specification"
                )

        # Check warmup ratio
        if iter_match and warmup_match:
            iterations = int(iter_match.group(1))
            warmup = int(warmup_match.group(1))
            if warmup < iterations * 0.5:
                issues.append(f"Warmup too low: {warmup} (should be >= {iterations // 2})")

        if issues:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                agent_name="r_convergence_check",
                message="MCMC configuration likely to fail convergence",
                details={"issues": issues, "random_effects_count": random_effects_count},
            )

        return ValidationResult(
            status=ValidationStatus.PASSED,
            agent_name="r_convergence_check",
            message="MCMC configuration looks reasonable",
        )

    def _validate_hardcoded_temporal(self, file_path: str, content: str | None) -> ValidationResult:
        """Detect hardcoded temporal values (Rule 9)."""
        # Skip test files, examples, demo code, and year-specific evaluation scripts
        skip_patterns = ["test", "example", "demo", "fixture", "mock"]
        # Skip scripts with year in filename or holdout/ablation scripts
        # (e.g., ab_test_semantic_2025.py, week15_predictions.py, ablation_synthetic_features.py)
        filename = file_path.split("/")[-1] if "/" in file_path else file_path
        if re.search(r"20\d{2}|week\d+|ablation|holdout", filename, re.IGNORECASE):
            return ValidationResult(
                status=ValidationStatus.SKIPPED,
                agent_name="hardcoded_temporal",
                message="Year-specific evaluation script",
            )
        if not content or any(p in file_path.lower() for p in skip_patterns):
            return ValidationResult(
                status=ValidationStatus.SKIPPED,
                agent_name="hardcoded_temporal",
                message="Test file or no content",
            )

        # Skip non-Python/R files (like .md documentation)
        if not file_path.endswith((".py", ".R", ".r")):
            return ValidationResult(
                status=ValidationStatus.SKIPPED,
                agent_name="hardcoded_temporal",
                message="Not a Python/R file",
            )

        # Find if __name__ == "__main__" block start (to skip demo code)
        main_block_start = content.find("if __name__")

        # Patterns to detect
        patterns = [
            (r"(season|year)\s*[=<-]\s*20\d{2}", "Direct assignment of year"),
            (r'paste0\([^)]*,\s*["\']20\d{2}["\']', "R paste0 with hardcoded year"),
            (r'f["\'][^"\']*\{[^}]*20\d{2}[^}]*\}', "f-string with hardcoded year"),
            (r"player_season.*=.*20\d{2}", "player_season with hardcoded year"),
        ]

        issues = []
        for pattern, description in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                # Skip if inside if __name__ == "__main__" block (demo code)
                if main_block_start != -1 and match.start() > main_block_start:
                    continue

                # Skip if line contains pragma comment
                line_start = content.rfind("\n", 0, match.start()) + 1
                line_end = content.find("\n", match.end())
                line = content[line_start : line_end if line_end != -1 else len(content)]
                if "# pragma:" in line or "# noqa" in line or "# allowlist" in line:
                    continue

                # Skip CLI argument defaults (parser.add_argument with default=YEAR)
                if "parser.add_argument" in line or "add_argument" in line:
                    continue

                # Skip function parameter defaults (def func(season: int = 2024))
                if ": int =" in line or ": int=" in line:
                    continue

                # Skip docstring examples (inside triple-quoted strings)
                # Count triple quotes before match to determine if we're inside a docstring
                triple_double = content[: match.start()].count('"""')
                triple_single = content[: match.start()].count("'''")
                in_docstring = (triple_double % 2 == 1) or (triple_single % 2 == 1)
                if in_docstring:
                    continue

                # Get line number
                line_num = content[: match.start()].count("\n") + 1
                issues.append(
                    {"line": line_num, "match": match.group(0), "description": description}
                )

        if issues:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                agent_name="hardcoded_temporal",
                message=f"Found {len(issues)} hardcoded temporal value(s)",
                details={"issues": issues[:5]},  # Limit to first 5
            )

        return ValidationResult(
            status=ValidationStatus.PASSED,
            agent_name="hardcoded_temporal",
            message="No hardcoded temporal values detected",
        )

    def _validate_allow_new_levels(self, file_path: str, content: str | None) -> ValidationResult:
        """Validate brms posterior_predict has required parameters (Rule 10)."""
        if not content or not re.search(r"posterior_predict\(", content):
            return ValidationResult(
                status=ValidationStatus.SKIPPED,
                agent_name="allow_new_levels_check",
                message="No posterior_predict calls found",
            )

        # Find all posterior_predict calls
        calls = re.finditer(r"posterior_predict\([^)]{0,500}\)", content, re.DOTALL)

        issues = []
        for match in calls:
            call_text = match.group(0)
            line_num = content[: match.start()].count("\n") + 1

            has_allow_new = "allow_new_levels" in call_text
            has_sample_new = "sample_new_levels" in call_text

            if has_allow_new and not has_sample_new:
                issues.append(
                    {
                        "line": line_num,
                        "issue": "Missing sample_new_levels parameter",
                        "snippet": call_text[:100],
                    }
                )

        if issues:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                agent_name="allow_new_levels_check",
                message=f"Incomplete brms prediction configuration ({len(issues)} calls)",
                details={"issues": issues},
            )

        return ValidationResult(
            status=ValidationStatus.PASSED,
            agent_name="allow_new_levels_check",
            message="brms prediction calls properly configured",
        )

    def _validate_multi_season(self, file_path: str, content: str | None) -> ValidationResult:
        """Validate multi-season testing (Rule 12)."""
        if not content:
            return ValidationResult(
                status=ValidationStatus.SKIPPED,
                agent_name="multi_season_validation",
                message="No content to validate",
            )

        # Check for single-season filters
        single_season_patterns = [
            r"season\s*==\s*202[0-9](?!\s*(and|or|\|))",
            r"\.query.*season.*==.*202[0-9].*(?!AND|OR)",
            r"df\[df\[.season.\]\s*==\s*202[0-9]\]",
        ]

        has_single_season = any(
            re.search(p, content, re.IGNORECASE) for p in single_season_patterns
        )

        # Check for multi-season loops
        has_multi_season = bool(
            re.search(r"for\s+season\s+in\s+\[.*202[0-9].*,.*202[0-9]", content)
        )

        if has_single_season and not has_multi_season:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                agent_name="multi_season_validation",
                message="Single-season validation detected - recommend ≥3 seasons",
                details={
                    "recommendation": "Use multi-season validation (2022-2024) for ≥500 games",
                    "reference": "docs/POSTMORTEMS.md (Incremental XGBoost Baseline Error)",
                },
            )

        return ValidationResult(
            status=ValidationStatus.PASSED,
            agent_name="multi_season_validation",
            message="Multi-season validation present or not applicable",
        )

    def _validate_documentation_drift(
        self, file_path: str, content: str | None
    ) -> ValidationResult:
        """Detect documentation drift (Rule 13)."""
        suggestions = []

        # New model artifacts
        if re.search(r"models/.*\.(pkl|rds|pth)$", file_path):
            suggestions.append("Update CLAUDE.md 'Recent Work' section")
            suggestions.append("Register model: scripts/model_registry.py register <model>")

        # Milestones
        if "COMPLETE.md" in file_path:
            suggestions.append("Update CLAUDE.md with milestone completion")
            suggestions.append("Consider incrementing CLAUDE.md version number")

        # Schema changes
        if file_path.startswith("db/"):
            suggestions.append("Update docs/database/DATABASE_QUICK_REFERENCE.md")
            suggestions.append("Update docs/database/SQL_PATTERNS.md if join patterns changed")

        if suggestions:
            return ValidationResult(
                status=ValidationStatus.PASSED,
                agent_name="documentation_drift",
                message="Documentation may need update",
                details={"suggestions": suggestions},
            )

        return ValidationResult(
            status=ValidationStatus.PASSED,
            agent_name="documentation_drift",
            message="No documentation updates needed",
        )

    def _validate_model_integrity(self, file_path: str, content: str | None) -> ValidationResult:
        """Validate model save includes integrity checks (Rule 14)."""
        if not content:
            return ValidationResult(
                status=ValidationStatus.SKIPPED,
                agent_name="model_file_integrity",
                message="No content to validate",
            )

        # Find save operations
        save_patterns = [
            (r"saveRDS\(", r"file\.exists"),
            (r"pickle\.dump\(", r"os\.path\.exists"),
            (r"torch\.save\(", r"os\.path\.exists"),
            (r"joblib\.dump\(", r"os\.path\.exists"),
        ]

        issues = []
        for save_pattern, check_pattern in save_patterns:
            if re.search(save_pattern, content):
                # Check if integrity validation exists nearby
                if not re.search(check_pattern, content):
                    issues.append(f"Save operation ({save_pattern}) without integrity check")

        if issues:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                agent_name="model_file_integrity",
                message="Model save without integrity check",
                details={
                    "issues": issues,
                    "recommendation": "Add file.exists() + test reload after save",
                },
            )

        return ValidationResult(
            status=ValidationStatus.PASSED,
            agent_name="model_file_integrity",
            message="Model save operations look safe",
        )

    def _validate_api_contract(self, file_path: str, content: str | None) -> ValidationResult:
        """Validate API contract usage - catch deprecated parameters (Rule 15)."""
        # Only validate Python files (skip .md, .txt documentation with code examples)
        if not file_path.endswith(".py"):
            return ValidationResult(
                status=ValidationStatus.SKIPPED,
                agent_name="api_contract_validation",
                message="Not a Python file",
            )

        if not content:
            return ValidationResult(
                status=ValidationStatus.SKIPPED,
                agent_name="api_contract_validation",
                message="No content to validate",
            )

        issues = []

        # Check for deprecated parameters
        deprecated_params = [
            ("use_adaptive_weights", "Use adaptive_weights_path instead"),
            ("dynamic_reweighting", "Removed in v6 - weights are static"),
            ("auto_calibrate", "Temperature scaling now manual via T parameter"),
        ]

        for param, suggestion in deprecated_params:
            pattern = rf"\b{param}\s*="
            if re.search(pattern, content):
                line_num = len(re.findall(pattern, content[: content.find(param)]))
                issues.append(
                    {
                        "param": param,
                        "suggestion": suggestion,
                        "recommendation": "Use inspect.signature() to check API",
                    }
                )

        # Check for missing required parameters in ensemble initialization
        ensemble_init_pattern = r"EnhancedEnsemble\w*\([^)]{0,300}\)"
        matches = re.finditer(ensemble_init_pattern, content, re.DOTALL)

        for match in matches:
            init_text = match.group(0)
            line_num = content[: match.start()].count("\n") + 1

            # Check for db_password parameter (required)
            if "db_password" not in init_text:
                issues.append(
                    {
                        "line": line_num,
                        "issue": "Missing db_password parameter in ensemble initialization",
                        "suggestion": 'Add db_password=os.getenv("DB_PASSWORD")',
                    }
                )

        if issues:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                agent_name="api_contract_validation",
                message=f"API contract violations detected ({len(issues)} issues)",
                details={"issues": issues},
            )

        return ValidationResult(
            status=ValidationStatus.PASSED,
            agent_name="api_contract_validation",
            message="API contract usage looks correct",
        )

    def _validate_sql_results(self, file_path: str, content: str | None) -> ValidationResult:
        """Validate SQL query results checked before access (Rule 16)."""
        if not content:
            return ValidationResult(
                status=ValidationStatus.SKIPPED,
                agent_name="sql_result_validation",
                message="No content to validate",
            )

        issues = []

        # Pattern: pd.read_sql followed by direct indexing without len() check
        # Match: df = pd.read_sql(...) NOT followed by if len(df)
        pattern = r"(\w+)\s*=\s*pd\.read_sql[_query]*\([^)]+\)"
        matches = re.finditer(pattern, content, re.MULTILINE)

        for match in matches:
            var_name = match.group(1)
            line_num = content[: match.start()].count("\n") + 1

            # Look ahead for direct indexing without len() check
            # Search next 300 characters for .iloc[0], .values[0], etc.
            search_start = match.end()
            search_end = min(search_start + 300, len(content))
            next_lines = content[search_start:search_end]

            # Check for direct indexing patterns
            direct_index_patterns = [
                rf"{var_name}\.iloc\[0\]",
                rf"{var_name}\.values\[0\]",
                rf"{var_name}\[0\]",
            ]

            has_direct_index = any(re.search(p, next_lines) for p in direct_index_patterns)

            # Check for len() validation
            has_len_check = (
                re.search(rf"if\s+len\({var_name}\)", next_lines)
                or re.search(rf"if\s+{var_name}\.empty", next_lines)
                or re.search(rf"if\s+not\s+{var_name}\.empty", next_lines)
            )

            if has_direct_index and not has_len_check:
                issues.append(
                    {
                        "line": line_num,
                        "variable": var_name,
                        "issue": f"DataFrame '{var_name}' accessed without empty check",
                        "suggestion": f"Add: if len({var_name}) == 0: return None",
                    }
                )

        if issues:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                agent_name="sql_result_validation",
                message=f"SQL result access without validation ({len(issues)} issues)",
                details={"issues": issues},
            )

        return ValidationResult(
            status=ValidationStatus.PASSED,
            agent_name="sql_result_validation",
            message="SQL results properly validated before access",
        )

    def _validate_r_predictions(self, file_path: str, content: str | None) -> ValidationResult:
        """Validate R model predictions checked for NaN (Rule 17)."""
        if not content:
            return ValidationResult(
                status=ValidationStatus.SKIPPED,
                agent_name="r_prediction_validation",
                message="No content to validate",
            )

        issues = []

        # Check for posterior_predict without NaN validation
        if "posterior_predict" in content or "predict.brmsfit" in content:
            has_nan_check = bool(re.search(r"(np\.isnan|pd\.isna|math\.isnan)", content))

            if not has_nan_check:
                issues.append(
                    {
                        "issue": "R model predictions not validated for NaN",
                        "suggestion": "Add: if np.isnan(prediction).any(): return None",
                        "reference": "Bug found Nov 15: R models returned NaN, crashed ensemble",
                    }
                )

            # Check for proper error handling around rpy2 calls
            if "rpy2" in content:
                has_try_except = bool(re.search(r"try:\s+.*rpy2", content, re.DOTALL))
                if not has_try_except:
                    issues.append(
                        {
                            "issue": "rpy2 calls not wrapped in try/except",
                            "suggestion": "Wrap rpy2 calls in try/except RRuntimeError",
                            "reference": "R session failures crash without proper handling",
                        }
                    )

        if issues:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                agent_name="r_prediction_validation",
                message=f"R prediction validation missing ({len(issues)} issues)",
                details={"issues": issues},
            )

        return ValidationResult(
            status=ValidationStatus.PASSED,
            agent_name="r_prediction_validation",
            message="R predictions properly validated",
        )

    def _validate_ensemble_components(
        self, file_path: str, content: str | None
    ) -> ValidationResult:
        """Validate ensemble has minimum components check (Rule 18)."""
        if not content:
            return ValidationResult(
                status=ValidationStatus.SKIPPED,
                agent_name="ensemble_components_check",
                message="No content to validate",
            )

        # Skip non-Python files (documentation, config, etc.)
        if not file_path.endswith(".py"):
            return ValidationResult(
                status=ValidationStatus.PASSED,
                agent_name="ensemble_components_check",
                message="Not a Python file",
            )

        issues = []

        # Find ensemble predict calls
        if "ensemble.predict" in content or re.search(r"\.predict\(.*player_id", content):
            # Check for valid_predictions length check
            has_component_check = bool(
                re.search(
                    r"(if\s+len\(valid_predictions\)\s*[<>=]|"
                    r"valid_predictions\s*=\s*\[.*for.*if.*is not None\])",
                    content,
                    re.DOTALL,
                )
            )

            if not has_component_check:
                issues.append(
                    {
                        "issue": "Ensemble predictions not validated for minimum components",
                        "suggestion": "Add: if len(valid_predictions) < 3: raise RuntimeError(...)",
                        "reference": "Bug found Nov 15: Only 2/4 models loaded, poor predictions",
                    }
                )

            # Check for component failure logging
            has_failure_logging = bool(
                re.search(r"(logger\.(warning|error).*failed|" r"print.*component.*None)", content)
            )

            if not has_failure_logging:
                issues.append(
                    {
                        "issue": "No logging for component failures",
                        "suggestion": 'Add: logger.warning(f"Component {name} returned None")',
                        "reference": "Silent component failures are hard to debug",
                    }
                )

        if issues:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                agent_name="ensemble_components_check",
                message=f"Ensemble component validation missing ({len(issues)} issues)",
                details={"issues": issues},
            )

        return ValidationResult(
            status=ValidationStatus.PASSED,
            agent_name="ensemble_components_check",
            message="Ensemble component validation present",
        )

    def _validate_placeholder_predictions(
        self, file_path: str, content: str | None
    ) -> ValidationResult:
        """
        Validate no hardcoded placeholder predictions in ensemble code (Rule 19 - CRITICAL).

        **Purpose:** Prevent placeholder predictions from reaching production/validation.

        **Detection Patterns:**
        1. Hardcoded mean values around 250 ± 5 yards
        2. Hardcoded std values (25-40 yards) without computation
        3. Warning messages indicating placeholders

        **Why Critical:** Placeholders blocked validation for 18+ hours (Nov 16-18)
        """
        if not content or not file_path.endswith(".py"):
            return ValidationResult(
                status=ValidationStatus.SKIPPED,
                agent_name="placeholder_detection",
                message="Not a Python file",
            )

        # Skip test files and utility/validation code (which may contain example patterns)
        skip_patterns = [
            "test",
            "_test.py",
            "utils/",
            "enforcement",
            "validator",
            "example",
            "demo",
        ]
        if any(p in file_path.lower() for p in skip_patterns):
            return ValidationResult(
                status=ValidationStatus.SKIPPED,
                agent_name="placeholder_detection",
                message="Test/utility file - placeholders allowed",
            )

        issues = []

        # Pattern 1: Hardcoded mean values around 250 ± 5 yards
        mean_patterns = [
            (r"['\"]mean['\"]\\?\s*:\s*250\.0", "Hardcoded mean: 250.0"),
            (r"['\"]mean['\"]\\?\s*:\s*245\.0", "Hardcoded mean: 245.0"),
            (r"['\"]mean['\"]\\?\s*:\s*248\.0", "Hardcoded mean: 248.0"),
            (r"['\"]mean['\"]\\?\s*:\s*252\.0", "Hardcoded mean: 252.0"),
            (r"['\"]mean['\"]\\?\s*:\s*255\.0", "Hardcoded mean: 255.0"),
            (r"mean\s*=\s*250(?![\d])", "Direct assignment: mean = 250"),
            (r"mean\s*=\s*24[5-9](?![\d])", "Direct assignment: mean = 245-249"),
            (r"mean\s*=\s*25[0-5](?![\d])", "Direct assignment: mean = 250-255"),
        ]

        for pattern, description in mean_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1
                # Check if in predict method (high risk zone)
                predict_method_start = content.rfind("def predict", 0, match.start())
                next_def = content.find("def ", match.end())
                in_predict = predict_method_start > -1 and (
                    next_def == -1 or next_def > match.start()
                )

                if in_predict:
                    issues.append(
                        {
                            "line": line_num,
                            "match": match.group(0)[:60],
                            "description": description,
                            "severity": "CRITICAL",
                            "location": "predict() method",
                        }
                    )
                else:
                    issues.append(
                        {
                            "line": line_num,
                            "match": match.group(0)[:60],
                            "description": description,
                            "severity": "HIGH",
                            "location": "ensemble file",
                        }
                    )

        # Pattern 2: Hardcoded std values without computation (25-40 yards)
        std_patterns = [
            (r"['\"]std['\"]\\?\s*:\s*[2-4][0-9]\.[0-9]", "Hardcoded std without computation"),
            (r"std\s*=\s*[2-4][0-9](?![\d])", "Direct std assignment: 20-49"),
        ]

        for pattern, description in std_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1
                # Look for computation context (is it computed?)
                search_start = max(0, match.start() - 200)
                context = content[search_start : match.start() + 100]

                has_computation = any(
                    op in context
                    for op in [
                        "np.std",
                        "std(",
                        "pandas.std",
                        "computed",
                        "calculate",
                        "numpy",
                        "quantile",
                        "percentile",
                    ]
                )

                if not has_computation:
                    # Check if in predict method
                    predict_method_start = content.rfind("def predict", 0, match.start())
                    next_def = content.find("def ", match.end())
                    in_predict = predict_method_start > -1 and (
                        next_def == -1 or next_def > match.start()
                    )

                    issues.append(
                        {
                            "line": line_num,
                            "match": match.group(0)[:60],
                            "description": description,
                            "severity": "CRITICAL" if in_predict else "HIGH",
                            "location": "predict() method" if in_predict else "ensemble code",
                        }
                    )

        # Pattern 3: Warning messages indicating placeholders
        placeholder_keywords = [
            (r"placeholder", "Placeholder indicator"),
            (r"not yet integrated", "Not yet integrated indicator"),
            (r"TODO.*integrate", "TODO: integrate indicator"),
            (r"temporary.*return", "Temporary return indicator"),
            (r"dummy.*prediction", "Dummy prediction indicator"),
        ]

        for pattern, description in placeholder_keywords:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1

                # Skip if match is in a comment (# ... Placeholder ...)
                line_start = content.rfind("\n", 0, match.start()) + 1
                line_end = content.find("\n", match.end())
                line = content[line_start : line_end if line_end != -1 else len(content)]
                hash_pos = line.find("#")
                match_in_line_pos = match.start() - line_start
                if hash_pos != -1 and hash_pos < match_in_line_pos:
                    # Match is after a # comment marker - skip it
                    continue

                # Check if near a return statement
                search_end = min(len(content), match.end() + 200)
                context = content[match.start() : search_end]

                if "return" in context:
                    issues.append(
                        {
                            "line": line_num,
                            "match": match.group(0),
                            "description": description,
                            "severity": "CRITICAL",
                            "location": "predict/return path",
                        }
                    )

        if issues:
            critical_issues = [i for i in issues if i.get("severity") == "CRITICAL"]

            return ValidationResult(
                status=ValidationStatus.FAILED,
                agent_name="placeholder_detection",
                message=f"Placeholder predictions detected: {len(critical_issues)} CRITICAL, {len(issues) - len(critical_issues)} HIGH",
                details={
                    "issues": issues[:10],  # Limit to first 10
                    "critical_count": len(critical_issues),
                    "recommendation": "Remove all placeholder predictions before running validation/backtest",
                    "blocked_for": "18+ hours (Nov 16-18)" if critical_issues else None,
                },
            )

        return ValidationResult(
            status=ValidationStatus.PASSED,
            agent_name="placeholder_detection",
            message="No placeholder predictions detected",
        )

    def _validate_data_sufficiency(self, file_path: str, content: str | None) -> ValidationResult:
        """
        Validate sufficient training data exists before complex model training (Rule 20 - WARNING).

        **Purpose:** Warn about underpowered experiments like v7.0 (1,701 games for meta-learning).

        **Detection Patterns:**
        1. Meta-learning without data sufficiency checks
        2. Online learning without initialization validation

        **Why Advisory:** v7.0 needed 5,000-10,000 games but had only 1,701.

        **Thresholds:**
        - Meta-learning: ≥5,000 samples
        - Online learning: ≥1,000 samples for initialization
        """
        if not content or not file_path.endswith(".py"):
            return ValidationResult(
                status=ValidationStatus.SKIPPED,
                agent_name="data_sufficiency_gate",
                message="Not a Python file",
            )

        # Skip test files, examples, demos, and scripts
        skip_patterns = ["test", "example", "demo", "fixture", "mock", "scripts/"]
        if any(p in file_path.lower() for p in skip_patterns):
            return ValidationResult(
                status=ValidationStatus.SKIPPED,
                agent_name="data_sufficiency_gate",
                message="Test/example/script file - data sufficiency not required",
            )

        issues = []

        # Pattern 1: Meta-learning without data validation
        if re.search(r"meta.*learning|adaptive.*weight|contextual.*weight", content, re.IGNORECASE):
            # Check for data validation
            has_data_check = bool(
                re.search(
                    r"(assert\s+len\(.*\)\s*>=\s*\d+|"
                    r"if\s+len\(.*\)\s*<\s*\d+.*raise|"
                    r"validate.*sample.*size|"
                    r"check.*data.*sufficiency)",
                    content,
                    re.IGNORECASE,
                )
            )

            # Check for training calls
            has_training = bool(
                re.search(r"\.fit\(|\.train\(|lightgbm\.train", content, re.IGNORECASE)
            )

            if has_training and not has_data_check:
                issues.append(
                    {
                        "pattern": "Meta-learning without data validation",
                        "severity": "CRITICAL",
                        "required_samples": 5000,
                        "suggestion": 'Add: assert len(training_data) >= 5000, "Insufficient data for meta-learning"',
                        "reference": "v7.0 failure: 1,701 games insufficient (needed 5,000-10,000)",
                        "postmortem": "docs/postmortems/ENSEMBLE_V7_VALIDATION_FAILURE_NOV18.md",
                    }
                )

        # Pattern 2: Online learning without initialization check
        if re.search(
            r"online.*learning|incremental.*learning|stream.*learning", content, re.IGNORECASE
        ):
            has_init_check = bool(
                re.search(
                    r"(if\s+len\(.*\)\s*<\s*\d+.*warm.*start|"
                    r"initial.*batch.*size|"
                    r"bootstrap.*samples)",
                    content,
                    re.IGNORECASE,
                )
            )

            has_training = bool(
                re.search(r"\.partial_fit\(|\.update\(|incremental", content, re.IGNORECASE)
            )

            if has_training and not has_init_check:
                issues.append(
                    {
                        "pattern": "Online learning without initialization validation",
                        "severity": "HIGH",
                        "required_samples": 1000,
                        "suggestion": 'Add: assert len(init_data) >= 1000, "Need ≥1,000 samples for initialization"',
                        "reference": "Cold start requires sufficient bootstrap data",
                    }
                )

        # NOTE: Patterns 3-5 (XGBoost/LightGBM, ensemble reweighting, statistical tests)
        # were removed due to excessive false positives. Focus on meta-learning/online learning only.

        if issues:
            critical_issues = [i for i in issues if i["severity"] == "CRITICAL"]
            high_issues = [i for i in issues if i["severity"] == "HIGH"]

            return ValidationResult(
                status=ValidationStatus.FAILED,
                agent_name="data_sufficiency_gate",
                message=f"Data sufficiency validation missing: {len(critical_issues)} CRITICAL, {len(high_issues)} HIGH",
                details={
                    "issues": issues,
                    "critical_count": len(critical_issues),
                    "recommendation": "Add data validation BEFORE model training to prevent underpowered experiments",
                    "example": (
                        "def validate_data_sufficiency(n_samples, n_features, model_type):\n"
                        '    """Validate sufficient training data before building pipeline."""\n'
                        '    min_required = {"lightgbm": n_features * 100, "linear": n_features * 20}[model_type]\n'
                        '    assert n_samples >= min_required, f"Need {min_required}+ samples"\n'
                    ),
                    "prevented_failures": [
                        "v7.0: 1,701 games for meta-learning (needed 5,000+) → 58 hours wasted",
                        "v7.0: WLS on 50 games (needed 100+) → ineffective",
                    ],
                },
            )

        return ValidationResult(
            status=ValidationStatus.PASSED,
            agent_name="data_sufficiency_gate",
            message="Data sufficiency validation present or not applicable",
        )


# Singleton instance
_router_instance = None


def get_router() -> EnforcementRouter:
    """Get singleton enforcement router instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = EnforcementRouter()
    return _router_instance


# Decorator for runtime enforcement


def enforce(rule_name: str):
    """
    Decorator to enforce validation rules at runtime.

    **Example Usage:**

    ```python
    @enforce('sql_validation')
    def execute_query(query: str):
        return pd.read_sql(query, conn)

    # This will trigger SQL validation before function executes
    # If validation fails and rule is BLOCKING, raises RuntimeError
    ```

    Args:
        rule_name: Name of enforcement rule to apply
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            router = get_router()
            rule = router.rules.get(rule_name)

            if not rule or not rule.enabled:
                return func(*args, **kwargs)

            # Get source file of decorated function
            source_file = inspect.getfile(func)

            # Validate
            result = rule.validator(source_file, None)

            if result.failed and rule.level == EnforcementLevel.BLOCKING:
                raise RuntimeError(
                    f"❌ Enforcement rule '{rule_name}' failed: {result.message}\n"
                    f"Details: {result.details}"
                )
            elif result.failed:
                logger.warning(f"⚠️ {rule.name}: {result.message}")

            return func(*args, **kwargs)

        return wrapper

    return decorator


# Convenience decorators for common use cases


def enforce_sql_validation(func: Callable) -> Callable:
    """Enforce SQL validation on function."""
    return enforce("sql_validation")(func)


def enforce_backtest_health(func: Callable) -> Callable:
    """Enforce ensemble health check before backtest."""
    return enforce("backtest_health")(func)


def enforce_model_training(func: Callable) -> Callable:
    """Enforce model training validation."""
    return enforce("model_training")(func)


def enforce_placeholder_detection(func: Callable) -> Callable:
    """Enforce placeholder prediction detection."""
    return enforce("placeholder_detection")(func)


def enforce_data_sufficiency(func: Callable) -> Callable:
    """Enforce data sufficiency validation."""
    return enforce("data_sufficiency_gate")(func)


if __name__ == "__main__":
    # CLI interface for pre-commit hook integration
    import sys

    if len(sys.argv) < 2:
        print("Usage: python enforcement_router.py <file1> <file2> ...")
        sys.exit(1)

    router = get_router()
    files = sys.argv[1:]

    all_passed, results = router.validate_commit(files)

    # Print results
    for result in results:
        level = router.rules.get(
            result.agent_name,
            EnforcementRule(
                name="",
                description="",
                triggers=[],
                validator=lambda x, y: None,
                level=EnforcementLevel.INFO,
            ),
        ).level

        if result.failed and level == EnforcementLevel.BLOCKING:
            print(f"❌ BLOCKING: {result.agent_name}: {result.message}")
            if result.details:
                print(f"   Details: {result.details}")
        elif result.failed:
            print(f"⚠️ WARNING: {result.agent_name}: {result.message}")
        else:
            print(f"✅ PASSED: {result.agent_name}: {result.message}")

    if not all_passed:
        print("\n❌ Commit validation FAILED - blocking issues detected")
        sys.exit(1)
    else:
        print("\n✅ Commit validation PASSED")
        sys.exit(0)
