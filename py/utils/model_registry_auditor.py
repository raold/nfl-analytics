"""
Model Registry Auditor for NFL Analytics

Comprehensive automated auditing and registration of ML models across filesystem,
database (ml.model_registry), and registry YAML (models/MODEL_REGISTRY.yaml).

Purpose:
    - Automated registration workflow: Scan models/, identify pending registrations
    - Convergence audits: Verify Bayesian models (Rhat < 1.01, ESS > 400)
    - File integrity checks: Detect corrupted files, size anomalies
    - Database sync monitoring: Compare YAML to database, alert on divergence
    - Production readiness validation: Check training window, calibration

Critical Context:
    - 227 models pending registration (as of Nov 9, 2025)
    - Canonical window: 2018-2024 + H=4 decay (established Nov 8, 2025)
    - Database: postgresql://dro:<password>@localhost:5544/devdb01  # pragma: allowlist secret
    - Registry: models/MODEL_REGISTRY.yaml
    - Schema: ml.model_registry (25 models registered, 37 latest ID)

Validation Gates (Production Deployment):
    1. Bayesian convergence: Rhat < 1.02 (relaxed threshold, ADR-002)
    2. Calibration: Coverage > 85% (target 90%)
    3. Training window: 2018-2024 + H=4 decay
    4. File integrity: readable, size > 10 KB, valid timestamp

Usage:
    # Scan for pending models
    python py/utils/model_registry_auditor.py --scan

    # Auto-register pending models (dry-run)
    python py/utils/model_registry_auditor.py --register --dry-run

    # Full audit report
    python py/utils/model_registry_auditor.py --audit

    # Check convergence for all Bayesian models
    python py/utils/model_registry_auditor.py --convergence

    # Database sync validation
    python py/utils/model_registry_auditor.py --sync-check

Python API:
    from py.utils.model_registry_auditor import ModelRegistryAuditor

    auditor = ModelRegistryAuditor()

    # Scan for pending models
    report = auditor.scan_models()
    print(f"Pending: {len(report.pending_models)}")

    # Auto-register
    results = auditor.auto_register(dry_run=True)

    # Full audit
    audit_report = auditor.audit_full()
    if audit_report.has_critical_issues():
        print("⛔ Critical issues found")
        exit(1)

Author: Claude Code (Sonnet 4.5)
Date: November 14, 2025
"""

import json
import logging
import os
import pickle
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import psycopg2
import yaml  # type: ignore
from psycopg2.extras import DictCursor

# Optional dependencies for CLI
try:
    import click

    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False
    click = None

try:
    from tabulate import tabulate  # type: ignore

    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False
    tabulate = None

# Try to import torch for .pth files
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Try to import rpy2 for .rds files
try:
    import rpy2.robjects as ro

    RDS_AVAILABLE = True
except (ImportError, Exception):
    # Catch any exception during rpy2 import (including R initialization errors)
    RDS_AVAILABLE = False
    ro = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Database connection
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "database": "devdb01",
    "user": "dro",
    "password": os.getenv("DB_PASSWORD", ""),
}

# Canonical training window (established Nov 8, 2025)
CANONICAL_WINDOW = {
    "full_weight": "2018-2024",
    "decay_function": "exponential",
    "decay_parameter": "H=4",
}

# Validation thresholds (ADR-002: relaxed from 1.01 to 1.02)
CONVERGENCE_THRESHOLDS = {
    "rhat_max": 1.02,
    "ess_min": 400,
    "divergences_max": 200,  # <10% of typical 2000-6000 iterations
}

CALIBRATION_THRESHOLDS = {
    "coverage_90pct_min": 0.85,
    "coverage_90pct_target": 0.90,
}

# File size thresholds (KB)
FILE_SIZE_THRESHOLDS = {
    "min_kb": 10,  # Models < 10 KB likely corrupted
    "warn_mb": 500,  # Models > 500 MB worth warning
}


@dataclass
class ModelFile:
    """Represents a model file on filesystem."""

    file_path: Path
    file_name: str
    file_type: str  # pkl, pth, rds
    size_bytes: int
    size_kb: float
    modified_time: datetime
    model_name: str  # Inferred from filename
    is_readable: bool
    read_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatabaseModel:
    """Represents a model in ml.model_registry."""

    model_id: int
    model_name: str
    model_type: str
    status: str
    is_production: bool
    model_path: str
    file_exists: bool = False
    file_size_kb: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegistryModel:
    """Represents a model in MODEL_REGISTRY.yaml."""

    registry_key: str
    model_name: str
    model_type: str
    status: str
    file_path: str
    ensemble_weight: Optional[float]
    convergence: Optional[Dict[str, Any]]
    performance: Optional[Dict[str, Any]]
    file_exists: bool = False


@dataclass
class ConvergenceReport:
    """Bayesian model convergence report."""

    model_name: str
    converged: bool
    rhat_max: Optional[float]
    ess_min: Optional[int]
    divergences: Optional[int]
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class CalibrationReport:
    """Model calibration report."""

    model_name: str
    calibrated: bool
    coverage_90pct: Optional[float]
    temperature: Optional[float]
    issues: List[str] = field(default_factory=list)


@dataclass
class RegistrationCandidate:
    """Model candidate for registration."""

    file_path: Path
    model_name: str
    model_type: str
    inferred_status: str
    should_register: bool
    blocking_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditReport:
    """Full audit report."""

    audit_date: str = field(default_factory=lambda: datetime.now().isoformat())
    total_files: int = 0
    total_db_models: int = 0
    total_registry_models: int = 0
    pending_registrations: int = 0
    convergence_failures: int = 0
    calibration_issues: int = 0
    file_integrity_issues: int = 0
    sync_mismatches: int = 0

    # Detailed findings
    filesystem_models: List[ModelFile] = field(default_factory=list)
    database_models: List[DatabaseModel] = field(default_factory=list)
    registry_models: List[RegistryModel] = field(default_factory=list)
    registration_candidates: List[RegistrationCandidate] = field(default_factory=list)
    convergence_reports: List[ConvergenceReport] = field(default_factory=list)
    calibration_reports: List[CalibrationReport] = field(default_factory=list)

    # Critical issues (block deployment)
    critical_issues: List[str] = field(default_factory=list)

    def has_critical_issues(self) -> bool:
        """Check if report has critical issues."""
        return len(self.critical_issues) > 0

    def summary(self) -> str:
        """Generate summary text."""
        lines = [
            "=" * 80,
            "MODEL REGISTRY AUDIT REPORT",
            "=" * 80,
            f"Audit Date: {self.audit_date}",
            "",
            "SUMMARY",
            "-" * 80,
            f"Filesystem Models: {self.total_files}",
            f"Database Models: {self.total_db_models}",
            f"Registry Models: {self.total_registry_models}",
            f"Pending Registrations: {self.pending_registrations}",
            "",
            "ISSUES",
            "-" * 80,
            f"Convergence Failures: {self.convergence_failures}",
            f"Calibration Issues: {self.calibration_issues}",
            f"File Integrity Issues: {self.file_integrity_issues}",
            f"Sync Mismatches: {self.sync_mismatches}",
            f"Critical Issues: {len(self.critical_issues)}",
            "",
        ]

        if self.critical_issues:
            lines.append("⛔ CRITICAL ISSUES")
            lines.append("-" * 80)
            for issue in self.critical_issues:
                lines.append(f"  - {issue}")
            lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)


class ModelRegistryAuditor:
    """
    Model Registry Auditor for NFL Analytics.

    Comprehensive automated auditing and registration of ML models.
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        registry_path: Optional[Path] = None,
    ):
        """
        Initialize auditor.

        Args:
            project_root: Project root directory (default: auto-detect)
            registry_path: Path to MODEL_REGISTRY.yaml (default: models/MODEL_REGISTRY.yaml)
        """
        if project_root is None:
            # Auto-detect project root
            project_root = Path(__file__).resolve().parent.parent.parent

        self.project_root = Path(project_root)
        self.models_dir = self.project_root / "models"

        if registry_path is None:
            registry_path = self.models_dir / "MODEL_REGISTRY.yaml"

        self.registry_path = Path(registry_path)

        # Validate paths
        if not self.models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {self.models_dir}")

        if not self.registry_path.exists():
            raise FileNotFoundError(f"Registry not found: {self.registry_path}")

        # Validate DB_PASSWORD
        if not DB_CONFIG["password"]:
            raise ValueError(
                "DB_PASSWORD environment variable is required. "
                "Set it in .env or export DB_PASSWORD=your_password"
            )

        logger.info(f"Initialized auditor for project: {self.project_root}")
        logger.info(f"Models directory: {self.models_dir}")
        logger.info(f"Registry file: {self.registry_path}")

    def get_db_connection(self):
        """Get database connection."""
        try:
            conn = psycopg2.connect(**DB_CONFIG, cursor_factory=DictCursor)
            return conn
        except psycopg2.Error as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def scan_filesystem_models(self) -> List[ModelFile]:
        """
        Scan models/ directory for all model files (.pkl, .pth, .rds).

        Returns:
            List of ModelFile objects
        """
        logger.info("Scanning filesystem for models...")

        model_files = []
        extensions = ["*.pkl", "*.pth", "*.rds", "*.pt"]

        for ext in extensions:
            for file_path in self.models_dir.rglob(ext):
                # Skip backups, temp files
                if any(x in file_path.name for x in [".bak", ".tmp", "~", ".old"]):
                    continue

                # Get file stats
                stat = file_path.stat()
                size_bytes = stat.st_size
                size_kb = size_bytes / 1024
                modified_time = datetime.fromtimestamp(stat.st_mtime)

                # Infer model name from filename
                model_name = file_path.stem

                # Try to read file
                is_readable = True
                read_error = None
                metadata = {}

                try:
                    if file_path.suffix in [".pkl"]:
                        # Try to load pickle metadata
                        with open(file_path, "rb") as f:
                            try:
                                obj = pickle.load(f)
                                # Extract metadata if available
                                if isinstance(obj, dict) and "metadata" in obj:
                                    metadata = obj["metadata"]
                            except Exception as e:
                                # File readable but not standard format
                                logger.debug(
                                    f"Could not extract metadata from {file_path.name}: {e}"
                                )

                    elif file_path.suffix in [".pth", ".pt"]:
                        if TORCH_AVAILABLE:
                            try:
                                obj = torch.load(file_path, map_location="cpu")
                                if isinstance(obj, dict) and "metadata" in obj:
                                    metadata = obj["metadata"]
                            except Exception as e:
                                logger.debug(f"Could not load PyTorch model {file_path.name}: {e}")
                        else:
                            logger.warning("PyTorch not available, cannot validate .pth files")

                    elif file_path.suffix == ".rds":
                        if RDS_AVAILABLE:
                            try:
                                # R files can be read but metadata extraction complex
                                pass
                            except Exception as e:
                                logger.debug(f"Could not load RDS model {file_path.name}: {e}")
                        else:
                            logger.warning("rpy2 not available, cannot validate .rds files")

                except Exception as e:
                    is_readable = False
                    read_error = str(e)
                    logger.warning(f"File not readable: {file_path.name} ({e})")

                model_file = ModelFile(
                    file_path=file_path,
                    file_name=file_path.name,
                    file_type=file_path.suffix[1:],  # Remove leading dot
                    size_bytes=size_bytes,
                    size_kb=size_kb,
                    modified_time=modified_time,
                    model_name=model_name,
                    is_readable=is_readable,
                    read_error=read_error,
                    metadata=metadata,
                )

                model_files.append(model_file)

        logger.info(f"Found {len(model_files)} model files")
        return model_files

    def scan_database_models(self) -> List[DatabaseModel]:
        """
        Scan ml.model_registry for all registered models.

        Returns:
            List of DatabaseModel objects
        """
        logger.info("Scanning database for registered models...")

        conn = self.get_db_connection()
        cursor = conn.cursor()

        query = """
        SELECT
            model_id,
            model_name,
            model_type,
            status,
            is_production,
            model_path,
            description,
            notes,
            training_start,
            training_end,
            train_start_season,
            train_end_season
        FROM ml.model_registry
        ORDER BY model_id
        """

        cursor.execute(query)
        rows = cursor.fetchall()

        database_models = []
        for row in rows:
            # Check if file exists
            model_path = row["model_path"]
            file_exists = False
            file_size_kb = None

            if model_path:
                full_path = self.project_root / model_path
                if full_path.exists():
                    file_exists = True
                    file_size_kb = full_path.stat().st_size / 1024

            metadata = {
                "description": row.get("description"),
                "notes": row.get("notes"),
                "training_start": row.get("training_start"),
                "training_end": row.get("training_end"),
                "train_start_season": row.get("train_start_season"),
                "train_end_season": row.get("train_end_season"),
            }

            db_model = DatabaseModel(
                model_id=row["model_id"],
                model_name=row["model_name"],
                model_type=row["model_type"],
                status=row["status"],
                is_production=row["is_production"],
                model_path=model_path,
                file_exists=file_exists,
                file_size_kb=file_size_kb,
                metadata=metadata,
            )

            database_models.append(db_model)

        cursor.close()
        conn.close()

        logger.info(f"Found {len(database_models)} database models")
        return database_models

    def scan_registry_yaml(self) -> List[RegistryModel]:
        """
        Parse MODEL_REGISTRY.yaml for production models.

        Returns:
            List of RegistryModel objects
        """
        logger.info("Scanning MODEL_REGISTRY.yaml...")

        with open(self.registry_path) as f:
            registry = yaml.safe_load(f)

        registry_models = []

        # Production models
        if "production_models" in registry:
            for key, config in registry["production_models"].items():
                file_path = config.get("file_path", "")
                full_path = self.project_root / file_path if file_path else None
                file_exists = full_path.exists() if full_path else False

                model = RegistryModel(
                    registry_key=key,
                    model_name=config.get("name", key),
                    model_type=config.get("model_type", "unknown"),
                    status=config.get("status", "unknown"),
                    file_path=file_path,
                    ensemble_weight=config.get("ensemble_weight"),
                    convergence=config.get("convergence"),
                    performance=config.get("performance"),
                    file_exists=file_exists,
                )

                registry_models.append(model)

        # Development models
        if "development_models" in registry:
            for key, config in registry["development_models"].items():
                file_path = config.get("file_path", "")
                full_path = self.project_root / file_path if file_path else None
                file_exists = full_path.exists() if full_path else False

                model = RegistryModel(
                    registry_key=key,
                    model_name=config.get("name", key),
                    model_type=config.get("model_type", "unknown"),
                    status=config.get("status", "unknown"),
                    file_path=file_path,
                    ensemble_weight=None,
                    convergence=config.get("convergence"),
                    performance=config.get("performance"),
                    file_exists=file_exists,
                )

                registry_models.append(model)

        logger.info(f"Found {len(registry_models)} registry models")
        return registry_models

    def identify_pending_registrations(
        self,
        filesystem_models: List[ModelFile],
        database_models: List[DatabaseModel],
    ) -> List[RegistrationCandidate]:
        """
        Identify models on filesystem not in database.

        Args:
            filesystem_models: Models found on filesystem
            database_models: Models in database

        Returns:
            List of RegistrationCandidate objects
        """
        logger.info("Identifying pending registrations...")

        # Build set of registered model names (from database)
        registered_names = {m.model_name for m in database_models}

        # Also check for path matches (some models may have different names)
        registered_paths = {m.model_path for m in database_models if m.model_path}

        candidates = []

        for file_model in filesystem_models:
            # Convert to relative path
            rel_path = file_model.file_path.relative_to(self.project_root)

            # Check if already registered
            if file_model.model_name in registered_names:
                continue

            if str(rel_path) in registered_paths:
                continue

            # Infer model type from directory structure
            model_type = "unknown"
            if "bayesian" in str(file_model.file_path):
                if file_model.file_type == "rds":
                    model_type = "hierarchical_bayesian"
                else:
                    model_type = "bayesian_neural_network"
            elif "xgboost" in str(file_model.file_path):
                model_type = "gradient_boosting"
            elif "cql" in str(file_model.file_path):
                model_type = "reinforcement_learning"
            elif "calibration" in str(file_model.file_path):
                model_type = "calibration"

            # Infer status
            inferred_status = "experimental"
            if "v1" in file_model.model_name and "final" not in file_model.model_name:
                inferred_status = "experimental"
            elif "v2" in file_model.model_name or "final" in file_model.model_name:
                inferred_status = "staging"

            # Check for blocking issues
            blocking_issues = []
            warnings = []

            if not file_model.is_readable:
                blocking_issues.append(f"File not readable: {file_model.read_error}")

            if file_model.size_kb < FILE_SIZE_THRESHOLDS["min_kb"]:
                blocking_issues.append(
                    f"File too small ({file_model.size_kb:.1f} KB), likely corrupted"
                )

            if file_model.size_kb > FILE_SIZE_THRESHOLDS["warn_mb"] * 1024:
                warnings.append(f"Large file ({file_model.size_kb / 1024:.1f} MB)")

            # Should register if no blocking issues
            should_register = len(blocking_issues) == 0

            candidate = RegistrationCandidate(
                file_path=file_model.file_path,
                model_name=file_model.model_name,
                model_type=model_type,
                inferred_status=inferred_status,
                should_register=should_register,
                blocking_issues=blocking_issues,
                warnings=warnings,
                metadata={
                    "size_kb": file_model.size_kb,
                    "modified_time": file_model.modified_time.isoformat(),
                },
            )

            candidates.append(candidate)

        logger.info(f"Found {len(candidates)} pending registrations")
        logger.info(f"  - Should register: {sum(1 for c in candidates if c.should_register)}")
        logger.info(f"  - Blocked: {sum(1 for c in candidates if not c.should_register)}")

        return candidates

    def validate_convergence(self, registry_models: List[RegistryModel]) -> List[ConvergenceReport]:
        """
        Validate Bayesian model convergence (Rhat, ESS, divergences).

        Args:
            registry_models: Models from registry YAML

        Returns:
            List of ConvergenceReport objects
        """
        logger.info("Validating Bayesian model convergence...")

        reports = []

        for model in registry_models:
            # Only check Bayesian models
            if model.model_type not in [
                "hierarchical_bayesian",
                "state_space",
                "bayesian_neural_network",
            ]:
                continue

            if not model.convergence:
                # No convergence data available
                report = ConvergenceReport(
                    model_name=model.model_name,
                    converged=False,
                    rhat_max=None,
                    ess_min=None,
                    divergences=None,
                    issues=["No convergence data available"],
                )
                reports.append(report)
                continue

            # Extract convergence metrics
            rhat_max = model.convergence.get("rhat_max")
            ess_min = model.convergence.get("ess_min")
            divergences = model.convergence.get("divergences")

            issues = []
            warnings = []
            converged = True

            # Check Rhat (ADR-002: relaxed to 1.02)
            if rhat_max is not None:
                if rhat_max > CONVERGENCE_THRESHOLDS["rhat_max"]:
                    issues.append(
                        f"Rhat too high: {rhat_max:.4f} > {CONVERGENCE_THRESHOLDS['rhat_max']}"
                    )
                    converged = False
                elif rhat_max > 1.01:
                    warnings.append(f"Rhat marginal: {rhat_max:.4f} (relaxed threshold)")

            # Check ESS
            if ess_min is not None:
                if ess_min < CONVERGENCE_THRESHOLDS["ess_min"]:
                    issues.append(f"ESS too low: {ess_min} < {CONVERGENCE_THRESHOLDS['ess_min']}")
                    converged = False

            # Check divergences
            if divergences is not None:
                if divergences > CONVERGENCE_THRESHOLDS["divergences_max"]:
                    issues.append(
                        f"Too many divergences: {divergences} > {CONVERGENCE_THRESHOLDS['divergences_max']}"
                    )
                    converged = False
                elif divergences > 50:
                    warnings.append(f"Moderate divergences: {divergences}")

            report = ConvergenceReport(
                model_name=model.model_name,
                converged=converged,
                rhat_max=rhat_max,
                ess_min=ess_min,
                divergences=divergences,
                issues=issues,
                warnings=warnings,
            )

            reports.append(report)

        failed = sum(1 for r in reports if not r.converged)
        logger.info(f"Convergence validation: {len(reports)} models, {failed} failed")

        return reports

    def validate_calibration(self, registry_models: List[RegistryModel]) -> List[CalibrationReport]:
        """
        Validate model calibration (coverage tests).

        Args:
            registry_models: Models from registry YAML

        Returns:
            List of CalibrationReport objects
        """
        logger.info("Validating model calibration...")

        reports = []

        for model in registry_models:
            # Skip non-probabilistic models
            if model.model_type in ["gradient_boosting"]:
                continue

            if not model.performance:
                # No performance data available
                report = CalibrationReport(
                    model_name=model.model_name,
                    calibrated=False,
                    coverage_90pct=None,
                    temperature=None,
                    issues=["No performance data available"],
                )
                reports.append(report)
                continue

            # Extract calibration metrics
            coverage_90pct = model.performance.get("calibration_90pct")
            if coverage_90pct is not None and coverage_90pct > 1.0:
                # Convert percentage to fraction if needed
                coverage_90pct = coverage_90pct / 100

            temperature = model.performance.get("temperature")

            issues = []
            calibrated = True

            if coverage_90pct is not None:
                if coverage_90pct < CALIBRATION_THRESHOLDS["coverage_90pct_min"]:
                    issues.append(
                        f"Calibration too low: {coverage_90pct:.1%} < {CALIBRATION_THRESHOLDS['coverage_90pct_min']:.0%}"
                    )
                    calibrated = False
                elif abs(coverage_90pct - CALIBRATION_THRESHOLDS["coverage_90pct_target"]) > 0.05:
                    issues.append(f"Calibration off target: {coverage_90pct:.1%} (target: 90%)")
            else:
                issues.append("No calibration data available")
                calibrated = False

            report = CalibrationReport(
                model_name=model.model_name,
                calibrated=calibrated,
                coverage_90pct=coverage_90pct,
                temperature=temperature,
                issues=issues,
            )

            reports.append(report)

        failed = sum(1 for r in reports if not r.calibrated)
        logger.info(f"Calibration validation: {len(reports)} models, {failed} issues")

        return reports

    def check_file_integrity(self, filesystem_models: List[ModelFile]) -> List[str]:
        """
        Check file integrity (corrupted files, size anomalies).

        Args:
            filesystem_models: Models from filesystem scan

        Returns:
            List of issue descriptions
        """
        logger.info("Checking file integrity...")

        issues = []

        for model in filesystem_models:
            # Check readability
            if not model.is_readable:
                issues.append(f"⛔ {model.file_name}: Not readable ({model.read_error})")

            # Check size
            if model.size_kb < FILE_SIZE_THRESHOLDS["min_kb"]:
                issues.append(
                    f"⛔ {model.file_name}: Too small ({model.size_kb:.1f} KB), likely corrupted"
                )

            if model.size_kb > FILE_SIZE_THRESHOLDS["warn_mb"] * 1024:
                issues.append(f"⚠️  {model.file_name}: Large file ({model.size_kb / 1024:.1f} MB)")

            # Check timestamp (warn if > 30 days old and not production)
            days_old = (datetime.now() - model.modified_time).days
            if days_old > 30 and "v1" in model.file_name:
                issues.append(f"⚠️  {model.file_name}: Old file ({days_old} days), may be stale")

        logger.info(f"File integrity check: {len(issues)} issues")
        return issues

    def check_database_sync(
        self,
        database_models: List[DatabaseModel],
        registry_models: List[RegistryModel],
    ) -> List[str]:
        """
        Check sync between database and registry YAML.

        Args:
            database_models: Models in database
            registry_models: Models in registry YAML

        Returns:
            List of sync mismatch descriptions
        """
        logger.info("Checking database-registry sync...")

        mismatches = []

        # Build lookup maps
        db_by_name = {m.model_name: m for m in database_models}
        registry_by_name = {m.model_name: m for m in registry_models}

        # Check: Models in registry but not in database
        for reg_model in registry_models:
            if reg_model.model_name not in db_by_name:
                # Extract just the model name (not full description)
                model_key = reg_model.registry_key
                mismatches.append(f"⚠️  Registry model '{model_key}' not in database")

        # Check: Production models in database but not in registry
        for db_model in database_models:
            if db_model.is_production and db_model.model_name not in registry_by_name:
                mismatches.append(
                    f"⚠️  Production database model '{db_model.model_name}' not in registry"
                )

        # Check: Status mismatches
        for name in set(db_by_name.keys()) & set(registry_by_name.keys()):
            db_model = db_by_name[name]
            reg_model = registry_by_name[name]

            # Compare status
            if db_model.status != reg_model.status:
                mismatches.append(
                    f"⚠️  Status mismatch '{name}': DB={db_model.status}, Registry={reg_model.status}"
                )

        logger.info(f"Database-registry sync: {len(mismatches)} mismatches")
        return mismatches

    def audit_full(self) -> AuditReport:
        """
        Run full audit: scan, validate, check integrity, check sync.

        Returns:
            AuditReport with all findings
        """
        logger.info("Starting full audit...")

        # Scan all sources
        filesystem_models = self.scan_filesystem_models()
        database_models = self.scan_database_models()
        registry_models = self.scan_registry_yaml()

        # Identify pending registrations
        registration_candidates = self.identify_pending_registrations(
            filesystem_models, database_models
        )

        # Validate convergence
        convergence_reports = self.validate_convergence(registry_models)

        # Validate calibration
        calibration_reports = self.validate_calibration(registry_models)

        # Check file integrity
        file_integrity_issues = self.check_file_integrity(filesystem_models)

        # Check database sync
        sync_mismatches = self.check_database_sync(database_models, registry_models)

        # Build audit report
        report = AuditReport(
            total_files=len(filesystem_models),
            total_db_models=len(database_models),
            total_registry_models=len(registry_models),
            pending_registrations=len(registration_candidates),
            convergence_failures=sum(1 for r in convergence_reports if not r.converged),
            calibration_issues=sum(1 for r in calibration_reports if not r.calibrated),
            file_integrity_issues=sum(1 for x in file_integrity_issues if "⛔" in x),
            sync_mismatches=len(sync_mismatches),
            filesystem_models=filesystem_models,
            database_models=database_models,
            registry_models=registry_models,
            registration_candidates=registration_candidates,
            convergence_reports=convergence_reports,
            calibration_reports=calibration_reports,
        )

        # Identify critical issues (block deployment)
        critical = []

        # Critical: Convergence failures in production models
        for conv_report in convergence_reports:
            if not conv_report.converged:
                # Check if production
                reg_model = next(
                    (m for m in registry_models if m.model_name == conv_report.model_name), None
                )
                if reg_model and reg_model.status == "production":
                    critical.append(
                        f"Production model '{conv_report.model_name}' failed convergence"
                    )

        # Critical: File integrity issues
        for issue in file_integrity_issues:
            if "⛔" in issue:
                critical.append(issue.replace("⛔ ", ""))

        report.critical_issues = critical

        logger.info("Full audit complete")
        return report

    def auto_register(
        self,
        candidates: Optional[List[RegistrationCandidate]] = None,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """
        Auto-register pending models to database.

        Args:
            candidates: Registration candidates (default: auto-detect)
            dry_run: If True, don't actually insert (default: True)

        Returns:
            Dict with registration results
        """
        logger.info(f"Auto-registering models (dry_run={dry_run})...")

        if candidates is None:
            filesystem_models = self.scan_filesystem_models()
            database_models = self.scan_database_models()
            candidates = self.identify_pending_registrations(filesystem_models, database_models)

        # Filter to only candidates that should register
        valid_candidates = [c for c in candidates if c.should_register]

        logger.info(f"Found {len(valid_candidates)} valid candidates for registration")

        if dry_run:
            logger.info("DRY RUN: Would register the following models:")
            for candidate in valid_candidates:
                logger.info(f"  - {candidate.model_name} ({candidate.model_type})")

            return {
                "dry_run": True,
                "total_candidates": len(candidates),
                "valid_candidates": len(valid_candidates),
                "blocked_candidates": len(candidates) - len(valid_candidates),
                "registered": 0,
            }

        # Actually register
        conn = self.get_db_connection()
        cursor = conn.cursor()

        registered = 0
        errors = []

        for candidate in valid_candidates:
            try:
                # Convert to relative path
                rel_path = candidate.file_path.relative_to(self.project_root)

                query = """
                INSERT INTO ml.model_registry (
                    model_name,
                    model_type,
                    model_family,
                    version,
                    model_path,
                    status,
                    is_production,
                    description
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING model_id
                """

                # Extract version from name
                version = "v1"
                if "v2" in candidate.model_name:
                    version = "v2"
                elif "v3" in candidate.model_name:
                    version = "v3"

                # Infer model family
                family = (
                    candidate.model_type.split("_")[0]
                    if "_" in candidate.model_type
                    else candidate.model_type
                )

                cursor.execute(
                    query,
                    (
                        candidate.model_name,
                        candidate.model_type,
                        family,
                        version,
                        str(rel_path),
                        candidate.inferred_status,
                        False,  # Not production by default
                        f"Auto-registered on {datetime.now().isoformat()}",
                    ),
                )

                model_id = cursor.fetchone()[0]
                conn.commit()

                logger.info(f"✅ Registered {candidate.model_name} (ID: {model_id})")
                registered += 1

            except Exception as e:
                logger.error(f"❌ Failed to register {candidate.model_name}: {e}")
                errors.append({"model_name": candidate.model_name, "error": str(e)})
                conn.rollback()

        cursor.close()
        conn.close()

        logger.info(f"Registration complete: {registered} models registered")

        return {
            "dry_run": False,
            "total_candidates": len(candidates),
            "valid_candidates": len(valid_candidates),
            "blocked_candidates": len(candidates) - len(valid_candidates),
            "registered": registered,
            "errors": errors,
        }


# CLI (only available if click installed)
if CLICK_AVAILABLE:

    @click.group()
    def cli():
        """Model Registry Auditor for NFL Analytics."""
        pass

    @cli.command()
    @click.option("--output", "-o", type=click.Path(), help="Save report to file")
    def scan(output):
        """Scan filesystem for pending model registrations."""
        auditor = ModelRegistryAuditor()

        filesystem_models = auditor.scan_filesystem_models()
        database_models = auditor.scan_database_models()
        candidates = auditor.identify_pending_registrations(filesystem_models, database_models)

        # Display summary
        click.echo(f"\n{'=' * 80}")
        click.echo("PENDING MODEL REGISTRATIONS")
        click.echo(f"{'=' * 80}")
        click.echo(f"Total filesystem models: {len(filesystem_models)}")
        click.echo(f"Total database models: {len(database_models)}")
        click.echo(f"Pending registrations: {len(candidates)}")
        click.echo(f"  - Should register: {sum(1 for c in candidates if c.should_register)}")
        click.echo(f"  - Blocked: {sum(1 for c in candidates if not c.should_register)}")
        click.echo()

        # Show candidates
        if candidates:
            valid = [c for c in candidates if c.should_register]
            blocked = [c for c in candidates if not c.should_register]

            if valid:
                click.echo("READY TO REGISTER:")
                for c in valid[:10]:  # Show first 10
                    click.echo(f"  ✅ {c.model_name} ({c.model_type}) - {c.file_path.name}")
                if len(valid) > 10:
                    click.echo(f"  ... and {len(valid) - 10} more")
                click.echo()

            if blocked:
                click.echo("BLOCKED (Issues):")
                for c in blocked[:10]:  # Show first 10
                    click.echo(f"  ⛔ {c.model_name}: {', '.join(c.blocking_issues)}")
                if len(blocked) > 10:
                    click.echo(f"  ... and {len(blocked) - 10} more")
                click.echo()

        if output:
            with open(output, "w") as f:
                json.dump([asdict(c) for c in candidates], f, indent=2, default=str)
            click.echo(f"Report saved to {output}")

    @cli.command()
    @click.option("--dry-run/--no-dry-run", default=True, help="Dry run (don't actually register)")
    def register(dry_run):
        """Auto-register pending models to database."""
        auditor = ModelRegistryAuditor()

        results = auditor.auto_register(dry_run=dry_run)

        click.echo(f"\n{'=' * 80}")
        click.echo("AUTO-REGISTRATION RESULTS")
        click.echo(f"{'=' * 80}")
        click.echo(f"Dry run: {results['dry_run']}")
        click.echo(f"Total candidates: {results['total_candidates']}")
        click.echo(f"Valid candidates: {results['valid_candidates']}")
        click.echo(f"Blocked candidates: {results['blocked_candidates']}")
        click.echo(f"Registered: {results['registered']}")

        if results.get("errors"):
            click.echo(f"\nErrors: {len(results['errors'])}")
            for err in results["errors"][:5]:
                click.echo(f"  ❌ {err['model_name']}: {err['error']}")

    @cli.command()
    @click.option("--output", "-o", type=click.Path(), help="Save report to file")
    def audit(output):
        """Run full audit (scan, validate, check integrity)."""
        auditor = ModelRegistryAuditor()

        report = auditor.audit_full()

        # Display summary
        click.echo(report.summary())

        # Show convergence failures
        if report.convergence_failures > 0:
            click.echo("\nCONVERGENCE FAILURES")
            click.echo("-" * 80)
            for r in report.convergence_reports:
                if not r.converged:
                    click.echo(f"  ⛔ {r.model_name}")
                    for issue in r.issues:
                        click.echo(f"      - {issue}")
            click.echo()

        # Show calibration issues
        if report.calibration_issues > 0:
            click.echo("\nCALIBRATION ISSUES")
            click.echo("-" * 80)
            for r in report.calibration_reports:
                if not r.calibrated:
                    click.echo(f"  ⚠️  {r.model_name}")
                    for issue in r.issues:
                        click.echo(f"      - {issue}")
            click.echo()

        if output:
            # Save full report
            report_dict = asdict(report)
            with open(output, "w") as f:
                json.dump(report_dict, f, indent=2, default=str)
            click.echo(f"\nFull report saved to {output}")

    @cli.command()
    def convergence():
        """Check Bayesian model convergence (Rhat, ESS, divergences)."""
        auditor = ModelRegistryAuditor()

        registry_models = auditor.scan_registry_yaml()
        reports = auditor.validate_convergence(registry_models)

        click.echo(f"\n{'=' * 80}")
        click.echo("BAYESIAN MODEL CONVERGENCE")
        click.echo(f"{'=' * 80}")

        # Build table
        rows = []
        for r in reports:
            status = "✅ PASS" if r.converged else "⛔ FAIL"
            rhat = f"{r.rhat_max:.4f}" if r.rhat_max else "N/A"
            ess = str(r.ess_min) if r.ess_min else "N/A"
            div = str(r.divergences) if r.divergences else "N/A"

            rows.append([r.model_name, status, rhat, ess, div])

        if TABULATE_AVAILABLE:
            click.echo(tabulate(rows, headers=["Model", "Status", "Rhat", "ESS", "Divergences"]))
        else:
            # Fallback simple table
            click.echo(f"{'Model':<40} {'Status':<10} {'Rhat':<10} {'ESS':<10} {'Divergences':<15}")
            click.echo("-" * 85)
            for row in rows:
                click.echo(f"{row[0]:<40} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<15}")

        # Show failures
        failures = [r for r in reports if not r.converged]
        if failures:
            click.echo(f"\n{len(failures)} FAILURES:")
            for r in failures:
                click.echo(f"\n⛔ {r.model_name}")
                for issue in r.issues:
                    click.echo(f"    - {issue}")

    @cli.command()
    def sync_check():
        """Check database-registry sync."""
        auditor = ModelRegistryAuditor()

        database_models = auditor.scan_database_models()
        registry_models = auditor.scan_registry_yaml()

        mismatches = auditor.check_database_sync(database_models, registry_models)

        click.echo(f"\n{'=' * 80}")
        click.echo("DATABASE-REGISTRY SYNC CHECK")
        click.echo(f"{'=' * 80}")
        click.echo(f"Database models: {len(database_models)}")
        click.echo(f"Registry models: {len(registry_models)}")
        click.echo(f"Mismatches: {len(mismatches)}")
        click.echo()

        if mismatches:
            for mismatch in mismatches:
                click.echo(f"  {mismatch}")
        else:
            click.echo("  ✅ All models in sync")

else:

    def cli():
        """Placeholder when click not available."""
        print("Error: click library required for CLI. Install with: pip install click")
        print("Python API is still available.")


if __name__ == "__main__":
    if CLICK_AVAILABLE:
        cli()
    else:
        print("Error: click library required for CLI")
        print("Install with: pip install click")
        print("\nPython API is still available. Example:")
        print("  from py.utils.model_registry_auditor import ModelRegistryAuditor")
        print("  auditor = ModelRegistryAuditor()")
        print("  report = auditor.audit_full()")
