#!/usr/bin/env python3
"""
Component Architecture Enforcer - Prevent monolithic code smells

Enforces architectural best practices: module size limits, circular dependency
detection, coupling metrics, and automated refactoring suggestions.

Purpose:
- Enforce 2k-line module limits (prevent 91k-line dashboard/app.py!)
- Detect circular dependencies between modules
- Track coupling metrics (afferent/efferent coupling)
- Suggest refactoring opportunities
- Auto-generate component breakdown plans

Usage:
    from nfl_analytics.agents.architecture_enforcer import ArchitectureEnforcer

    enforcer = ArchitectureEnforcer()

    # Check if file violates size limit
    violations = enforcer.check_module_size("dashboard/app.py")
    if violations:
        print(f"File too large: {violations[0].lines} lines")

    # Detect circular dependencies
    cycles = enforcer.detect_circular_dependencies()

    # Get refactoring plan for large file
    plan = enforcer.generate_refactoring_plan("dashboard/app.py")

Author: Claude Code (Sonnet 4.5)
Date: November 22, 2025
Phase: 6 (Frontend Agents - Week 4)
"""

import ast
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import networkx as nx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ViolationType(Enum):
    """Types of architectural violations."""

    MODULE_SIZE = "module_size"  # File >2k lines
    CIRCULAR_DEPENDENCY = "circular_dependency"
    HIGH_COUPLING = "high_coupling"  # Ce + Ca > 30
    GOD_CLASS = "god_class"  # Class with >10 methods
    LONG_METHOD = "long_method"  # Method >50 lines
    DEEP_NESTING = "deep_nesting"  # Nesting depth >4


class RefactoringStrategy(Enum):
    """Recommended refactoring strategies."""

    EXTRACT_MODULE = "extract_module"
    EXTRACT_CLASS = "extract_class"
    EXTRACT_METHOD = "extract_method"
    SPLIT_RESPONSIBILITIES = "split_responsibilities"
    INTRODUCE_FACADE = "introduce_facade"


@dataclass
class ArchitecturalViolation:
    """Represents an architectural rule violation."""

    violation_type: ViolationType
    severity: str  # CRITICAL, WARNING, INFO
    file_path: str
    line_number: int | None
    metric_value: float
    threshold: float
    message: str


@dataclass
class ModuleMetrics:
    """Metrics for a single module."""

    file_path: str
    lines_of_code: int
    num_classes: int
    num_functions: int
    num_imports: int
    afferent_coupling: int  # Number of modules that depend on this
    efferent_coupling: int  # Number of modules this depends on
    instability: float  # Ce / (Ca + Ce)
    violations: list[ArchitecturalViolation] = field(default_factory=list)


@dataclass
class CircularDependency:
    """Represents a circular dependency cycle."""

    cycle: list[str]  # Module paths in cycle
    length: int
    severity: str


@dataclass
class ComponentExtractionPlan:
    """Plan for extracting a component from a large module."""

    source_file: str
    target_file: str
    strategy: RefactoringStrategy
    lines_to_move: list[tuple[int, int]]  # (start_line, end_line) ranges
    estimated_reduction: int  # Lines of code reduction
    dependencies: list[str]  # Modules that will need updates
    code_example: str


@dataclass
class RefactoringPlan:
    """Complete refactoring plan for a module."""

    source_file: str
    current_lines: int
    target_lines: int
    num_extractions: int
    extractions: list[ComponentExtractionPlan]
    estimated_effort_hours: float
    priority: str  # HIGH, MEDIUM, LOW


class ArchitectureEnforcer:
    """
    Enforces architectural best practices and prevents code smells.

    Monitors module sizes, dependencies, coupling, and suggests refactoring.
    """

    def __init__(
        self,
        codebase_path: str = ".",
        max_module_lines: int = 2000,
        max_class_methods: int = 10,
        max_method_lines: int = 50,
        max_coupling: int = 30,
    ):
        self.codebase_path = Path(codebase_path)
        self.max_module_lines = max_module_lines
        self.max_class_methods = max_class_methods
        self.max_method_lines = max_method_lines
        self.max_coupling = max_coupling

        # Dependency graph (module -> list of dependencies)
        self.dependency_graph: dict[str, set[str]] = defaultdict(set)

    def check_module_size(self, file_path: str) -> list[ArchitecturalViolation]:
        """
        Check if a module exceeds size limit.

        Args:
            file_path: Path to Python file

        Returns:
            List of violations (empty if compliant)
        """
        violations = []
        path = Path(file_path)

        if not path.exists():
            return violations

        try:
            with open(path) as f:
                lines = len(f.readlines())

            if lines > self.max_module_lines:
                severity = "CRITICAL" if lines > 5000 else "WARNING"

                violations.append(
                    ArchitecturalViolation(
                        violation_type=ViolationType.MODULE_SIZE,
                        severity=severity,
                        file_path=str(path),
                        line_number=None,
                        metric_value=lines,
                        threshold=self.max_module_lines,
                        message=(
                            f"Module has {lines} lines (limit: {self.max_module_lines}). "
                            f"Consider splitting into smaller components."
                        ),
                    )
                )

        except Exception as e:
            logger.warning(f"Error checking {file_path}: {e}")

        return violations

    def analyze_module(self, file_path: str) -> ModuleMetrics:
        """
        Analyze a module and compute architectural metrics.

        Args:
            file_path: Path to Python file

        Returns:
            ModuleMetrics with detailed analysis
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Module not found: {file_path}")

        try:
            with open(path) as f:
                content = f.read()
                lines_of_code = len(content.splitlines())
                tree = ast.parse(content)

            # Count components
            num_classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
            num_functions = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
            num_imports = sum(
                1 for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
            )

            # Extract dependencies
            dependencies = self._extract_dependencies(tree, str(path))

            # Calculate coupling metrics
            afferent = self._calculate_afferent_coupling(str(path))
            efferent = len(dependencies)
            instability = efferent / (afferent + efferent) if (afferent + efferent) > 0 else 0.0

            # Check for violations
            violations = []

            # Size violation
            size_violations = self.check_module_size(file_path)
            violations.extend(size_violations)

            # Coupling violation
            total_coupling = afferent + efferent
            if total_coupling > self.max_coupling:
                violations.append(
                    ArchitecturalViolation(
                        violation_type=ViolationType.HIGH_COUPLING,
                        severity="WARNING",
                        file_path=str(path),
                        line_number=None,
                        metric_value=total_coupling,
                        threshold=self.max_coupling,
                        message=(
                            f"High coupling detected (Ca={afferent}, Ce={efferent}). "
                            f"Consider reducing dependencies."
                        ),
                    )
                )

            # God class violation
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                    if len(methods) > self.max_class_methods:
                        violations.append(
                            ArchitecturalViolation(
                                violation_type=ViolationType.GOD_CLASS,
                                severity="WARNING",
                                file_path=str(path),
                                line_number=node.lineno,
                                metric_value=len(methods),
                                threshold=self.max_class_methods,
                                message=(
                                    f"Class '{node.name}' has {len(methods)} methods "
                                    f"(limit: {self.max_class_methods})"
                                ),
                            )
                        )

            return ModuleMetrics(
                file_path=str(path),
                lines_of_code=lines_of_code,
                num_classes=num_classes,
                num_functions=num_functions,
                num_imports=num_imports,
                afferent_coupling=afferent,
                efferent_coupling=efferent,
                instability=instability,
                violations=violations,
            )

        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            raise

    def detect_circular_dependencies(
        self, directory: str | None = None
    ) -> list[CircularDependency]:
        """
        Detect circular dependencies in codebase.

        Args:
            directory: Directory to scan (default: entire codebase)

        Returns:
            List of circular dependency cycles
        """
        scan_path = Path(directory) if directory else self.codebase_path

        # Build dependency graph
        self._build_dependency_graph(scan_path)

        # Convert to NetworkX graph for cycle detection
        G = nx.DiGraph()
        for module, deps in self.dependency_graph.items():
            for dep in deps:
                G.add_edge(module, dep)

        # Find cycles
        cycles = []
        try:
            simple_cycles = list(nx.simple_cycles(G))

            for cycle in simple_cycles:
                severity = "CRITICAL" if len(cycle) > 2 else "WARNING"

                cycles.append(CircularDependency(cycle=cycle, length=len(cycle), severity=severity))

        except Exception as e:
            logger.warning(f"Error detecting cycles: {e}")

        return cycles

    def generate_refactoring_plan(self, file_path: str) -> RefactoringPlan:
        """
        Generate a refactoring plan for a large module.

        Args:
            file_path: Path to oversized module

        Returns:
            RefactoringPlan with extraction strategies
        """
        metrics = self.analyze_module(file_path)

        if metrics.lines_of_code <= self.max_module_lines:
            return RefactoringPlan(
                source_file=str(file_path),
                current_lines=metrics.lines_of_code,
                target_lines=metrics.lines_of_code,
                num_extractions=0,
                extractions=[],
                estimated_effort_hours=0.0,
                priority="LOW",
            )

        # Analyze structure to suggest extractions
        extractions = self._suggest_extractions(file_path, metrics)

        # Calculate target lines
        lines_to_extract = sum(e.estimated_reduction for e in extractions)
        target_lines = metrics.lines_of_code - lines_to_extract

        # Estimate effort (1 hour per 500 lines moved)
        estimated_effort_hours = lines_to_extract / 500.0

        # Determine priority
        if metrics.lines_of_code > 10000:
            priority = "HIGH"
        elif metrics.lines_of_code > 5000:
            priority = "MEDIUM"
        else:
            priority = "LOW"

        return RefactoringPlan(
            source_file=str(file_path),
            current_lines=metrics.lines_of_code,
            target_lines=target_lines,
            num_extractions=len(extractions),
            extractions=extractions,
            estimated_effort_hours=estimated_effort_hours,
            priority=priority,
        )

    def check_all_modules(self, directory: str | None = None) -> dict[str, ModuleMetrics]:
        """
        Analyze all Python modules in directory.

        Args:
            directory: Directory to scan (default: entire codebase)

        Returns:
            Dictionary mapping file paths to metrics
        """
        scan_path = Path(directory) if directory else self.codebase_path
        metrics = {}

        for py_file in scan_path.rglob("*.py"):
            # Skip test files and virtual environments
            if ".venv" in str(py_file) or "test_" in py_file.name:
                continue

            try:
                metrics[str(py_file)] = self.analyze_module(str(py_file))
            except Exception as e:
                logger.warning(f"Error analyzing {py_file}: {e}")

        return metrics

    def get_top_violations(
        self, limit: int = 10, violation_type: ViolationType | None = None
    ) -> list[ArchitecturalViolation]:
        """
        Get top architectural violations across codebase.

        Args:
            limit: Maximum number of violations to return
            violation_type: Optional filter by violation type

        Returns:
            List of most severe violations
        """
        all_metrics = self.check_all_modules()
        all_violations = []

        for metrics in all_metrics.values():
            for violation in metrics.violations:
                if violation_type is None or violation.violation_type == violation_type:
                    all_violations.append(violation)

        # Sort by severity and metric value
        severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
        all_violations.sort(key=lambda v: (severity_order[v.severity], -v.metric_value))

        return all_violations[:limit]

    # Private helper methods

    def _extract_dependencies(self, tree: ast.AST, file_path: str) -> set[str]:
        """Extract module dependencies from AST."""
        dependencies = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    dependencies.add(node.module.split(".")[0])

        # Update dependency graph
        self.dependency_graph[file_path] = dependencies

        return dependencies

    def _build_dependency_graph(self, directory: Path) -> None:
        """Build complete dependency graph for directory."""
        for py_file in directory.rglob("*.py"):
            if ".venv" in str(py_file):
                continue

            try:
                with open(py_file) as f:
                    tree = ast.parse(f.read())
                self._extract_dependencies(tree, str(py_file))
            except Exception as e:
                logger.warning(f"Error parsing {py_file}: {e}")

    def _calculate_afferent_coupling(self, module_path: str) -> int:
        """Calculate number of modules that depend on this module."""
        # Count how many other modules import this one
        afferent = 0
        module_name = Path(module_path).stem

        for other_module, deps in self.dependency_graph.items():
            if other_module != module_path and module_name in deps:
                afferent += 1

        return afferent

    def _suggest_extractions(
        self, file_path: str, metrics: ModuleMetrics
    ) -> list[ComponentExtractionPlan]:
        """Suggest component extractions for large module."""
        extractions = []

        try:
            with open(file_path) as f:
                tree = ast.parse(f.read())

            # Strategy 1: Extract large classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_lines = (
                        node.end_lineno - node.lineno if hasattr(node, "end_lineno") else 100
                    )

                    if class_lines > 200:  # Large class
                        target_file = f"{Path(file_path).parent}/{node.name.lower()}.py"

                        extractions.append(
                            ComponentExtractionPlan(
                                source_file=file_path,
                                target_file=target_file,
                                strategy=RefactoringStrategy.EXTRACT_CLASS,
                                lines_to_move=[(node.lineno, node.end_lineno)],
                                estimated_reduction=class_lines,
                                dependencies=[],
                                code_example=f"# Extract {node.name} to {target_file}",
                            )
                        )

        except Exception as e:
            logger.warning(f"Error suggesting extractions for {file_path}: {e}")

        return extractions


if __name__ == "__main__":
    # Example usage
    enforcer = ArchitectureEnforcer()

    # Check dashboard/app.py (known to be 91k lines)
    violations = enforcer.check_module_size("dashboard/app.py")

    if violations:
        print("\n⚠️  Module Size Violations:")
        for v in violations:
            print(f"  {v.severity}: {v.message}")

        # Generate refactoring plan
        plan = enforcer.generate_refactoring_plan("dashboard/app.py")

        print("\n=== Refactoring Plan ===")
        print(f"Current size: {plan.current_lines} lines")
        print(f"Target size: {plan.target_lines} lines")
        print(f"Extractions: {plan.num_extractions}")
        print(f"Estimated effort: {plan.estimated_effort_hours:.1f} hours")
        print(f"Priority: {plan.priority}")

    # Detect circular dependencies
    cycles = enforcer.detect_circular_dependencies("nfl_analytics/")

    if cycles:
        print("\n⚠️  Circular Dependencies:")
        for cycle in cycles[:5]:
            print(f"  {cycle.severity}: {' → '.join(cycle.cycle)}")

    # Top violations
    top_violations = enforcer.get_top_violations(limit=5)

    print("\n=== Top 5 Violations ===")
    for v in top_violations:
        print(f"{v.severity}: {v.file_path}")
        print(f"  {v.message}")
