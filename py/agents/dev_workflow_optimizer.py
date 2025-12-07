#!/usr/bin/env python3
"""
Development Workflow Optimizer - Maximize developer productivity

Monitors iteration cycle times, auto-detects rebuild triggers, suggests Docker
layer caching improvements, and formalizes best practices from earlier work.

Purpose:
- Monitor actual iteration cycle times (time from code change to result visible)
- Auto-detect rebuild triggers (what changes force full rebuild)
- Suggest Docker layer caching improvements
- Track developer experience metrics (builds/hour, avg cycle time)
- Formalize docker-compose.dev.yml best practices

Usage:
    from py.agents.dev_workflow_optimizer import DevWorkflowOptimizer

    optimizer = DevWorkflowOptimizer()

    # Monitor a development session
    session = optimizer.start_monitoring_session()
    # ... make changes, rebuild, test ...
    optimizer.end_monitoring_session(session)

    # Get cycle time report
    report = optimizer.analyze_cycle_times()
    print(f"Avg cycle time: {report.avg_cycle_time_seconds:.1f}s")

    # Get optimization suggestions
    suggestions = optimizer.suggest_optimizations()

Author: Claude Code (Sonnet 4.5)
Date: November 22, 2025
Phase: 6 (Frontend Agents - Week 4)
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import docker
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Types of code changes."""

    PYTHON_CODE = "python_code"
    PYTHON_DEPS = "python_deps"  # requirements.txt
    R_CODE = "r_code"
    R_DEPS = "r_deps"  # DESCRIPTION, renv.lock
    SQL = "sql"
    DOCKER_CONFIG = "docker_config"
    ENV_VARS = "env_vars"


class RebuildType(Enum):
    """Types of rebuilds."""

    NONE = "none"  # Hot reload (no rebuild)
    SERVICE_RESTART = "service_restart"  # docker-compose restart
    IMAGE_REBUILD = "image_rebuild"  # docker-compose build
    FULL_REBUILD = "full_rebuild"  # docker-compose down + build + up


@dataclass
class DevelopmentCycle:
    """Represents one development iteration cycle."""

    cycle_id: int
    started_at: datetime
    ended_at: datetime | None
    change_type: ChangeType
    rebuild_type: RebuildType
    duration_seconds: float | None
    files_changed: list[str]
    success: bool
    error: str | None


@dataclass
class DevelopmentSession:
    """Represents a development session (multiple cycles)."""

    session_id: str
    started_at: datetime
    ended_at: datetime | None
    cycles: list[DevelopmentCycle] = field(default_factory=list)


@dataclass
class CycleTimeReport:
    """Analysis of cycle times."""

    total_cycles: int
    avg_cycle_time_seconds: float
    median_cycle_time_seconds: float
    p95_cycle_time_seconds: float
    cycles_by_type: dict[ChangeType, int]
    rebuilds_by_type: dict[RebuildType, int]
    hot_reload_rate: float  # % of cycles with no rebuild
    generated_at: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationSuggestion:
    """Suggested workflow optimization."""

    category: str  # "docker", "caching", "tooling"
    title: str
    description: str
    expected_improvement: str
    implementation: str
    priority: str  # HIGH, MEDIUM, LOW


class DevWorkflowOptimizer:
    """
    Optimizes development workflow for maximum productivity.

    Monitors cycle times, detects bottlenecks, suggests improvements.
    """

    def __init__(
        self,
        project_root: str = ".",
        docker_compose_file: str = "docker-compose.yml",
        docker_compose_dev_file: str = "docker-compose.dev.yml"
    ):
        self.project_root = Path(project_root)
        self.docker_compose_file = self.project_root / docker_compose_file
        self.docker_compose_dev_file = self.project_root / docker_compose_dev_file

        self.sessions: list[DevelopmentSession] = []
        self.current_session: DevelopmentSession | None = None

        self.docker_client = docker.from_env()

    def start_monitoring_session(self) -> DevelopmentSession:
        """
        Start monitoring a development session.

        Returns:
            DevelopmentSession object to track cycles
        """
        session = DevelopmentSession(
            session_id=f"session_{int(time.time())}",
            started_at=datetime.now(),
            ended_at=None
        )

        self.current_session = session
        self.sessions.append(session)

        logger.info(f"Started monitoring session: {session.session_id}")
        return session

    def end_monitoring_session(self, session: DevelopmentSession) -> None:
        """
        End a monitoring session.

        Args:
            session: Session to end
        """
        session.ended_at = datetime.now()
        logger.info(f"Ended session: {session.session_id}")

    def track_cycle(
        self,
        change_type: ChangeType,
        files_changed: list[str]
    ) -> DevelopmentCycle:
        """
        Track a development cycle (code change → result visible).

        Args:
            change_type: Type of change made
            files_changed: List of file paths changed

        Returns:
            DevelopmentCycle object
        """
        if not self.current_session:
            self.start_monitoring_session()

        cycle = DevelopmentCycle(
            cycle_id=len(self.current_session.cycles) + 1,
            started_at=datetime.now(),
            ended_at=None,
            change_type=change_type,
            rebuild_type=RebuildType.NONE,  # Will be determined
            duration_seconds=None,
            files_changed=files_changed,
            success=False,
            error=None
        )

        # Determine rebuild type
        cycle.rebuild_type = self._determine_rebuild_type(change_type, files_changed)

        self.current_session.cycles.append(cycle)

        return cycle

    def complete_cycle(
        self,
        cycle: DevelopmentCycle,
        success: bool,
        error: str | None = None
    ) -> None:
        """
        Mark a cycle as complete.

        Args:
            cycle: Cycle to complete
            success: Whether cycle succeeded
            error: Optional error message
        """
        cycle.ended_at = datetime.now()
        cycle.success = success
        cycle.error = error

        if cycle.started_at and cycle.ended_at:
            duration = (cycle.ended_at - cycle.started_at).total_seconds()
            cycle.duration_seconds = duration

            logger.info(
                f"Cycle {cycle.cycle_id} complete: "
                f"{duration:.1f}s ({cycle.rebuild_type.value})"
            )

    def analyze_cycle_times(self) -> CycleTimeReport:
        """
        Analyze cycle times across all sessions.

        Returns:
            CycleTimeReport with statistics
        """
        all_cycles = []
        for session in self.sessions:
            all_cycles.extend(session.cycles)

        # Filter to completed cycles
        completed = [c for c in all_cycles if c.duration_seconds is not None]

        if not completed:
            return CycleTimeReport(
                total_cycles=0,
                avg_cycle_time_seconds=0.0,
                median_cycle_time_seconds=0.0,
                p95_cycle_time_seconds=0.0,
                cycles_by_type={},
                rebuilds_by_type={},
                hot_reload_rate=0.0
            )

        durations = sorted(c.duration_seconds for c in completed)

        # Calculate stats
        avg = sum(durations) / len(durations)
        median = durations[len(durations) // 2]
        p95_idx = int(len(durations) * 0.95)
        p95 = durations[p95_idx] if p95_idx < len(durations) else durations[-1]

        # Group by type
        cycles_by_type = {}
        for cycle in completed:
            cycles_by_type[cycle.change_type] = cycles_by_type.get(cycle.change_type, 0) + 1

        rebuilds_by_type = {}
        for cycle in completed:
            rebuilds_by_type[cycle.rebuild_type] = rebuilds_by_type.get(cycle.rebuild_type, 0) + 1

        # Hot reload rate
        hot_reload_count = rebuilds_by_type.get(RebuildType.NONE, 0)
        hot_reload_rate = hot_reload_count / len(completed) if completed else 0.0

        return CycleTimeReport(
            total_cycles=len(completed),
            avg_cycle_time_seconds=avg,
            median_cycle_time_seconds=median,
            p95_cycle_time_seconds=p95,
            cycles_by_type=cycles_by_type,
            rebuilds_by_type=rebuilds_by_type,
            hot_reload_rate=hot_reload_rate
        )

    def suggest_optimizations(self) -> list[OptimizationSuggestion]:
        """
        Suggest workflow optimizations based on analysis.

        Returns:
            List of actionable suggestions
        """
        suggestions = []
        report = self.analyze_cycle_times()

        # Suggestion 1: Enable hot reload if not already
        if report.hot_reload_rate < 0.5:
            suggestions.append(OptimizationSuggestion(
                category="docker",
                title="Enable hot reload for faster iteration",
                description=(
                    f"Only {report.hot_reload_rate:.0%} of cycles use hot reload. "
                    "Enabling hot reload can reduce cycle time from 45s to 5s."
                ),
                expected_improvement="9x faster iteration (45s → 5s)",
                implementation=(
                    "Use docker-compose.dev.yml with volume mounts and --reload flags. "
                    "See docs/guides/FRONTEND_DEV_WORKFLOW.md"
                ),
                priority="HIGH"
            ))

        # Suggestion 2: Optimize Docker layer caching
        if report.avg_cycle_time_seconds > 30:
            suggestions.append(OptimizationSuggestion(
                category="docker",
                title="Optimize Docker layer caching",
                description=(
                    f"Average cycle time is {report.avg_cycle_time_seconds:.0f}s. "
                    "Proper layer caching can reduce rebuild time by 50-80%."
                ),
                expected_improvement="2-5x faster rebuilds",
                implementation=(
                    "1. Copy requirements.txt before source code\n"
                    "2. Use multi-stage builds\n"
                    "3. Minimize COPY commands\n"
                    "4. Use .dockerignore"
                ),
                priority="HIGH"
            ))

        # Suggestion 3: Use BuildKit
        suggestions.append(OptimizationSuggestion(
            category="docker",
            title="Enable Docker BuildKit",
            description=(
                "BuildKit provides faster builds with improved caching and parallelization."
            ),
            expected_improvement="20-50% faster builds",
            implementation=(
                "Set DOCKER_BUILDKIT=1 environment variable or add to .bashrc:\n"
                "export DOCKER_BUILDKIT=1"
            ),
            priority="MEDIUM"
        ))

        # Suggestion 4: Pre-commit hooks
        suggestions.append(OptimizationSuggestion(
            category="tooling",
            title="Use pre-commit hooks for faster feedback",
            description=(
                "Catch linting/formatting errors before Docker rebuild cycle."
            ),
            expected_improvement="Catch errors in <1s vs 30-45s rebuild cycle",
            implementation=(
                "Install pre-commit hooks: pre-commit install\n"
                "Runs Black, flake8, mypy locally before commit"
            ),
            priority="MEDIUM"
        ))

        # Suggestion 5: Watch mode for tests
        suggestions.append(OptimizationSuggestion(
            category="tooling",
            title="Use pytest-watch for continuous testing",
            description=(
                "Automatically re-run tests on file changes for immediate feedback."
            ),
            expected_improvement="Instant test feedback vs manual pytest runs",
            implementation=(
                "Install: pip install pytest-watch\n"
                "Run: ptw tests/ --clear"
            ),
            priority="LOW"
        ))

        return suggestions

    def check_docker_dev_config(self) -> dict[str, Any]:
        """
        Check if docker-compose.dev.yml is configured correctly.

        Returns:
            Dictionary with configuration status
        """
        if not self.docker_compose_dev_file.exists():
            return {
                "configured": False,
                "message": "docker-compose.dev.yml not found",
                "recommendations": [
                    "Create docker-compose.dev.yml with hot reload enabled",
                    "See docs/guides/FRONTEND_DEV_WORKFLOW.md"
                ]
            }

        try:
            with open(self.docker_compose_dev_file) as f:
                config = yaml.safe_load(f)

            # Check for key optimizations
            has_volumes = False
            has_reload = False

            for service_name, service_config in config.get("services", {}).items():
                # Check for volume mounts
                if "volumes" in service_config:
                    has_volumes = True

                # Check for reload flags
                command = service_config.get("command", [])
                if isinstance(command, list) and "--reload" in command:
                    has_reload = True
                elif isinstance(command, str) and "--reload" in command:
                    has_reload = True

            configured = has_volumes and has_reload

            return {
                "configured": configured,
                "has_volumes": has_volumes,
                "has_reload": has_reload,
                "message": (
                    "Docker dev config looks good!" if configured
                    else "Docker dev config needs optimization"
                ),
                "recommendations": [] if configured else [
                    "Add volume mounts for hot reload" if not has_volumes else "",
                    "Add --reload flag to uvicorn" if not has_reload else ""
                ]
            }

        except Exception as e:
            return {
                "configured": False,
                "message": f"Error parsing docker-compose.dev.yml: {e}",
                "recommendations": ["Fix YAML syntax errors"]
            }

    # Private helper methods

    def _determine_rebuild_type(
        self,
        change_type: ChangeType,
        files_changed: list[str]
    ) -> RebuildType:
        """Determine what type of rebuild is required."""

        # Check docker-compose.dev.yml for hot reload support
        dev_config = self.check_docker_dev_config()

        if change_type == ChangeType.PYTHON_CODE and dev_config["has_reload"]:
            return RebuildType.NONE  # Hot reload
        elif change_type == ChangeType.PYTHON_DEPS:
            return RebuildType.IMAGE_REBUILD
        elif change_type == ChangeType.DOCKER_CONFIG:
            return RebuildType.FULL_REBUILD
        elif change_type == ChangeType.ENV_VARS:
            return RebuildType.SERVICE_RESTART
        else:
            return RebuildType.SERVICE_RESTART


if __name__ == "__main__":
    # Example usage
    optimizer = DevWorkflowOptimizer()

    # Check dev configuration
    dev_config = optimizer.check_docker_dev_config()
    print("\n=== Docker Dev Configuration ===")
    print(f"Configured: {dev_config['configured']}")
    print(f"Message: {dev_config['message']}")

    if dev_config["recommendations"]:
        print("\nRecommendations:")
        for rec in dev_config["recommendations"]:
            if rec:
                print(f"  - {rec}")

    # Simulate monitoring a session
    session = optimizer.start_monitoring_session()

    # Track a cycle
    cycle = optimizer.track_cycle(
        change_type=ChangeType.PYTHON_CODE,
        files_changed=["dashboard/app.py"]
    )
    time.sleep(0.1)  # Simulate work
    optimizer.complete_cycle(cycle, success=True)

    # Get report
    report = optimizer.analyze_cycle_times()
    print("\n=== Cycle Time Report ===")
    print(f"Total cycles: {report.total_cycles}")
    print(f"Avg cycle time: {report.avg_cycle_time_seconds:.1f}s")
    print(f"Hot reload rate: {report.hot_reload_rate:.0%}")

    # Get suggestions
    suggestions = optimizer.suggest_optimizations()
    print("\n=== Optimization Suggestions ===")
    for sug in suggestions[:3]:
        print(f"\n{sug.priority}: {sug.title}")
        print(f"  {sug.description}")
        print(f"  Expected: {sug.expected_improvement}")
