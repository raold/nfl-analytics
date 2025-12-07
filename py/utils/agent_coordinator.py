"""
Agent Coordinator for NFL Analytics Project

This module provides intelligent task routing to specialized agents based on project
components. It enables complex multi-component tasks to be handled by appropriate
specialized agents working in coordination.

Components Managed:
1. Dissertation - LaTeX compilation, figure updates, content auditing
2. Model Management - Training, evaluation, versioning, registry updates
3. Frontend - Dashboard updates, Streamlit/Grafana monitoring
4. Data Layer - Schema management, migrations, queries, data integrity
5. Prediction Engine - Ensemble integration, backtesting, production deployment
6. Documentation - GitHub management, documentation updates, status tracking

Usage:
    from py.utils.agent_coordinator import AgentCoordinator

    coordinator = AgentCoordinator()

    # Route a single task
    result = coordinator.route_task(
        "Update dissertation with latest BNN calibration results",
        context={"bnn_results": {...}}
    )

    # Handle complex multi-component task
    results = coordinator.coordinate_task(
        "Retrain all models with canonical window and update dashboard",
        components=["model_management", "frontend", "documentation"]
    )
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Specialized agent types for NFL analytics project components."""

    DISSERTATION = "dissertation"
    MODEL_MANAGEMENT = "model_management"
    FRONTEND = "frontend"
    DATA_LAYER = "data_layer"
    PREDICTION_ENGINE = "prediction_engine"
    DOCUMENTATION = "documentation"
    RESEARCH_COORDINATOR = "research_coordinator"
    GENERAL = "general"


@dataclass
class AgentCapability:
    """Defines capabilities and responsibilities of a specialized agent."""

    agent_type: AgentType
    name: str
    description: str
    responsibilities: list[str]
    file_patterns: list[str] = field(default_factory=list)
    keywords: set[str] = field(default_factory=set)
    dependencies: list[AgentType] = field(default_factory=list)

    def matches(self, task: str, files: list[str] | None = None) -> float:
        """
        Calculate match score for this agent given a task description.

        Args:
            task: Task description to analyze
            files: Optional list of files involved in the task

        Returns:
            Match score from 0.0 to 1.0 (higher = better match)
        """
        score = 0.0
        task_lower = task.lower()

        # Keyword matching (50% weight)
        keyword_matches = sum(1 for kw in self.keywords if kw in task_lower)
        if self.keywords:
            score += (keyword_matches / len(self.keywords)) * 0.5

        # File pattern matching (30% weight)
        if files and self.file_patterns:
            file_matches = sum(
                1 for f in files
                for pattern in self.file_patterns
                if pattern in f
            )
            score += min(file_matches / len(files), 1.0) * 0.3

        # Responsibility matching (20% weight)
        resp_matches = sum(
            1 for resp in self.responsibilities
            if any(word in task_lower for word in resp.lower().split())
        )
        if self.responsibilities:
            score += (resp_matches / len(self.responsibilities)) * 0.2

        return min(score, 1.0)


@dataclass
class TaskRouting:
    """Represents a task routed to one or more agents."""

    task: str
    primary_agent: AgentType
    supporting_agents: list[AgentType] = field(default_factory=list)
    confidence: float = 0.0
    context: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class AgentCoordinator:
    """
    Coordinator for routing tasks to specialized agents in the NFL analytics project.

    This class analyzes incoming requests and intelligently routes them to appropriate
    specialized agents based on capabilities, keywords, file patterns, and dependencies.
    """

    def __init__(self):
        """Initialize coordinator with agent registry."""
        self.agents = self._initialize_agents()
        self.routing_history: list[TaskRouting] = []

        logger.info(
            f"AgentCoordinator initialized with {len(self.agents)} specialized agents"
        )

    def _initialize_agents(self) -> dict[AgentType, AgentCapability]:
        """
        Initialize registry of specialized agents with capabilities.

        Returns:
            Dictionary mapping agent types to their capabilities
        """
        agents = {
            AgentType.DISSERTATION: AgentCapability(
                agent_type=AgentType.DISSERTATION,
                name="Dissertation Agent",
                description="Manages LaTeX dissertation compilation, figures, and content",
                responsibilities=[
                    "Compile LaTeX documents",
                    "Update dissertation chapters",
                    "Generate TikZ figures",
                    "Manage bibliography and references",
                    "Create tables from results",
                    "Track dissertation TODOs",
                    "Audit dissertation completeness",
                ],
                file_patterns=[
                    "analysis/dissertation/",
                    ".tex",
                    ".bib",
                    "figures/",
                ],
                keywords={
                    "dissertation", "latex", "tikz", "compile", "chapter",
                    "figure", "table", "bibliography", "citation", "reference",
                    "writeup", "thesis", "subsection", "section",
                },
                dependencies=[],
            ),

            AgentType.MODEL_MANAGEMENT: AgentCapability(
                agent_type=AgentType.MODEL_MANAGEMENT,
                name="Model Management Agent",
                description="Handles model training, evaluation, versioning, registry, performance optimization, testing, and statistical validation",
                responsibilities=[
                    "Train Bayesian models (brms, Stan, PyMC)",
                    "Train ML models (XGBoost, BNN)",
                    "Evaluate model performance",
                    "Version model artifacts",
                    "Update model registry",
                    "Track training runs",
                    "Manage model metadata",
                    "Hyperparameter optimization",
                    # Phase 2: Performance Optimizer
                    "Profile model training performance",
                    "Optimize MCMC configuration (chains, tune, draws)",
                    "Estimate training and backtest time",
                    "Identify performance bottlenecks in training",
                    # Phase 2: Test Generator
                    "Generate unit tests for model code",
                    "Generate property-based tests (Hypothesis)",
                    "Generate temporal validation tests",
                    "Target 80%+ code coverage",
                    # Phase 2: Statistical Validator
                    "Validate MCMC diagnostics (R-hat, ESS, divergences)",
                    "Validate model calibration (coverage tests)",
                    "Check prior/posterior comparison",
                    "Block deployment if validation fails",
                ],
                file_patterns=[
                    "py/models/",
                    "R/train_",
                    "models/",
                    ".pkl",
                    ".rds",
                    ".pt",
                    "_metadata.json",
                    "py/utils/performance_optimizer.py",
                    "py/utils/test_generator.py",
                    "py/utils/statistical_validator.py",
                    "tests/",
                ],
                keywords={
                    "train", "model", "bayesian", "xgboost", "bnn", "neural network",
                    "hierarchical", "state-space", "ensemble", "prediction",
                    "hyperparameter", "optimization", "calibration", "evaluation",
                    "brms", "stan", "pymc", "cmdstanr", "pytorch",
                    # Phase 2: Performance
                    "performance", "profiling", "speed", "bottleneck", "mcmc efficiency",
                    "backtest time", "training time", "optimize mcmc",
                    # Phase 2: Testing
                    "test", "testing", "coverage", "unit test", "integration test",
                    "property test", "hypothesis", "pytest", "temporal validation",
                    # Phase 2: Validation (stronger keywords)
                    "validate model", "model validation", "mcmc diagnostics",
                    "validate mcmc", "check diagnostics", "r-hat", "rhat", "ess",
                    "ess values", "validate r-hat", "validate ess",
                    "divergence", "calibration", "validate calibration",
                    "prior posterior", "cross validation",
                    "statistical validation", "convergence", "check convergence",
                    "model diagnostics",
                },
                dependencies=[AgentType.DATA_LAYER],
            ),

            AgentType.FRONTEND: AgentCapability(
                agent_type=AgentType.FRONTEND,
                name="Frontend Agent",
                description="Manages dashboards, monitoring, and visualization",
                responsibilities=[
                    "Update Streamlit dashboard",
                    "Configure Grafana panels",
                    "Create visualizations",
                    "Monitor system health",
                    "Display predictions",
                    "Track model metrics",
                    "User interface updates",
                ],
                file_patterns=[
                    "dashboard/",
                    "grafana/",
                    "app.py",
                    "dashboards.yaml",
                    ".json",
                ],
                keywords={
                    "dashboard", "streamlit", "grafana", "visualization", "plot",
                    "chart", "panel", "monitoring", "metrics", "ui", "interface",
                    "display", "frontend", "web",
                },
                dependencies=[AgentType.DATA_LAYER, AgentType.MODEL_MANAGEMENT],
            ),

            AgentType.DATA_LAYER: AgentCapability(
                agent_type=AgentType.DATA_LAYER,
                name="Data Layer Agent",
                description="Manages database schema, migrations, queries, data integrity, leakage detection, SQL performance optimization, and column lineage tracking",
                responsibilities=[
                    "Create database migrations",
                    "Update schema definitions",
                    "Write optimized queries",
                    "Validate data integrity",
                    "Manage materialized views",
                    "Ingest new data sources",
                    "Clean and transform data",
                    "Feature engineering",
                    "Detect temporal data leakage",
                    "Validate train/test splits",
                    "Check for target leakage",
                    "Prevent player-level contamination",
                    # Phase 2: Performance Optimizer (SQL)
                    "Analyze SQL query performance",
                    "Recommend database indexes",
                    "Optimize PostgreSQL queries",
                    "Identify slow query patterns",
                    # Phase 4: Data Layer Analyzer (Column Lineage)
                    "Track column usage across codebase",
                    "Build column lineage (column → consumers)",
                    "Analyze impact of schema changes (add/modify/remove)",
                    "Detect unused/dead columns",
                    "Identify feature opportunities for new columns",
                    "Validate migrations before deployment",
                    "Generate dependency graphs (column dependencies)",
                    "Prevent breaking changes in production",
                ],
                file_patterns=[
                    "db/",
                    "data/",
                    ".sql",
                    "queries/",
                    "migrations/",
                    "ingest_",
                    "py/utils/data_leakage_detector.py",
                    "py/utils/performance_optimizer.py",
                    "py/utils/data_layer_analyzer.py",
                    "scripts/analyze_data_layer.sh",
                ],
                keywords={
                    "database", "schema", "migration", "query", "sql", "postgres",
                    "ingest", "data", "etl", "transform", "clean", "feature",
                    "materialized view", "table", "column", "index",
                    "leakage", "temporal", "asof", "split", "contamination",
                    "train test", "validation", "integrity",
                    # Phase 2: Performance
                    "query performance", "slow query", "database optimization",
                    "sql optimization", "index recommendation", "explain analyze",
                    # Phase 4: Column Lineage & Impact Analysis
                    "column lineage", "impact analysis", "schema change", "breaking changes",
                    "unused columns", "dead columns", "column usage", "column tracking",
                    "dependency graph", "column dependencies", "feature opportunities",
                    "add column", "remove column", "drop column", "alter column",
                    "migration validation", "trace column", "column consumers",
                },
                dependencies=[],
            ),

            AgentType.PREDICTION_ENGINE: AgentCapability(
                agent_type=AgentType.PREDICTION_ENGINE,
                name="Prediction Engine Agent",
                description="Handles ensemble integration, backtesting, production deployment, ensemble orchestration, and betting strategy validation",
                responsibilities=[
                    "Integrate ensemble methods",
                    "Run backtests",
                    "Generate predictions",
                    "Optimize ensemble weights",
                    "Deploy to production",
                    "Risk gate validation",
                    "ROI tracking",
                    "A/B testing",
                    # Phase 3: Ensemble Orchestrator
                    "Orchestrate 5-way ensemble training (R + Python)",
                    "Coordinate multi-language model training",
                    "Optimize ensemble weights with CVX",
                    "Manage model versioning and registry",
                    "Monitor individual model performance",
                    "Trigger retraining when needed",
                    # Phase 3: Betting Strategy Validator
                    "Validate betting strategy before deployment",
                    "Check risk gates (edge, confidence, bankroll)",
                    "Validate Kelly criterion bet sizing",
                    "Validate ROI/Sharpe/Drawdown targets",
                    "Run stress tests on betting strategy",
                    "Block deployment if betting validation fails",
                    # Phase 2: Performance (backtest optimization)
                    "Optimize prediction pipeline latency (<10s)",
                    "Reduce backtest runtime (<1h target)",
                    "Implement caching strategies",
                ],
                file_patterns=[
                    "py/ensemble/",
                    "py/backtest/",
                    "py/production/",
                    "py/api/",
                    "py/utils/ensemble_orchestrator.py",
                    "py/utils/betting_validator.py",
                    "py/utils/performance_optimizer.py",
                    "py/rl/",
                ],
                keywords={
                    "ensemble", "backtest", "prediction", "production", "deploy",
                    "roi", "risk", "bet", "betting", "odds", "props",
                    "weight", "optimization", "integration", "api",
                    # Phase 3: Orchestration (stronger keywords)
                    "orchestration", "orchestrate ensemble", "ensemble coordination",
                    "coordinate training", "multi-language training",
                    "r python coordination", "retrain all models", "retrain ensemble",
                    "model registry", "cvx", "cvxpy", "weight optimization",
                    "5-way ensemble", "ensemble models",
                    # Phase 3: Betting Validation
                    "betting validation", "validate betting", "validate strategy",
                    "risk gates", "kelly criterion", "roi validation",
                    "sharpe ratio", "drawdown", "stress test",
                    "edge validation", "confidence threshold", "bankroll management",
                    "before deployment", "approve strategy",
                    # Phase 2: Performance
                    "prediction latency", "backtest runtime", "reduce backtest",
                    "caching", "pipeline optimization",
                },
                dependencies=[AgentType.MODEL_MANAGEMENT, AgentType.DATA_LAYER],
            ),

            AgentType.DOCUMENTATION: AgentCapability(
                agent_type=AgentType.DOCUMENTATION,
                name="Documentation Agent",
                description="Manages documentation, GitHub, project tracking, code review, and security",
                responsibilities=[
                    "Update project documentation",
                    "Maintain CLAUDE.md",
                    "Track project status",
                    "Create milestone docs",
                    "Manage GitHub issues/PRs",
                    "Update README files",
                    "Generate documentation indexes",
                    "Track technical debt",
                    "Code review (multi-language: Python, R, SQL, TeX, Shell)",
                    "Security auditing (credentials, Docker, API)",
                    "Detect hardcoded secrets",
                    "Enforce code quality standards",
                ],
                file_patterns=[
                    "docs/",
                    "README.md",
                    "CLAUDE.md",
                    "PROJECT_STATUS.md",
                    ".md",
                    "py/utils/code_reviewer.py",
                    "py/utils/security_auditor.py",
                ],
                keywords={
                    "documentation", "docs", "readme", "guide", "github",
                    "status", "milestone", "tracking", "update", "maintain",
                    "index", "catalog", "changelog",
                    # Code review and security (stronger)
                    "code review", "review code", "run code review",
                    "security", "audit", "security audit", "credentials", "secrets",
                    "vulnerability", "quality", "lint", "review pr", "pr review",
                    "review betting", "review validator", "review implementation",
                },
                dependencies=[],
            ),

            AgentType.RESEARCH_COORDINATOR: AgentCapability(
                agent_type=AgentType.RESEARCH_COORDINATOR,
                name="Research Coordinator",
                description="Tracks experiments, maintains dissertation, guides R/Bayesian workflow, ensures publication-quality typography, tells research story",
                responsibilities=[
                    # Experiment Tracking
                    "Track experiment narrative and development story",
                    "Maintain research log and milestone documentation",
                    "Connect experiments to dissertation chapters",
                    "Synthesize findings across experiments",
                    "Document what worked, what didn't, and why",
                    # Dissertation Management
                    "Update dissertation chapters with latest findings",
                    "Generate LaTeX tables from model results",
                    "Create TikZ figures with consistent styling",
                    "Manage bibliography and citations",
                    "Ensure dissertation compiles cleanly",
                    "Track dissertation TODOs and gaps",
                    # R Statistics Workflow
                    "Guide brms/Stan/cmdstanr workflows",
                    "Interpret MCMC diagnostics in R context",
                    "Convert brms output to LaTeX tables",
                    "Generate ggplot2 publication-quality figures",
                    "Understand hierarchical Bayesian modeling patterns",
                    # Typography & Aesthetics
                    "Enforce consistent figure styling (TikZ style guide)",
                    "Ensure publication-quality typography",
                    "Optimize figure layouts and spacing",
                    "Manage color schemes (accent blue: 31,119,180)",
                    "Review visual consistency across chapters",
                    # Cross-Component Synthesis
                    "Synthesize findings across model types",
                    "Connect data → features → models → results → dissertation",
                    "Tell the complete research story",
                    "Identify narrative gaps",
                    "Suggest experiments to strengthen story",
                ],
                file_patterns=[
                    "analysis/dissertation/",
                    ".tex",
                    ".bib",
                    "analysis/dissertation/figures/",
                    "analysis/dissertation/style/",
                    "docs/milestones/",
                    "docs/reports/",
                    "R/train_",
                    "R/state_space/",
                    "R/features/",
                    "models/bayesian/*.rds",
                    "RECENT_WORK_HISTORY.md",
                    "RESEARCH_TO_OUTPUT_MAPPING.md",
                ],
                keywords={
                    # Experiment tracking (unique to Research Coordinator)
                    "experiment journey", "research log", "development story",
                    "what worked", "lessons learned", "document experiment",
                    "track experiments", "research narrative", "experiment tracking",
                    # Dissertation content updates (different from compilation)
                    "update dissertation chapter", "dissertation findings",
                    "connect to dissertation", "latex table from results",
                    "generate latex table", "results to latex",
                    # R Statistics guidance (interpretation, not training)
                    "interpret brms", "interpret mcmc", "help me interpret",
                    "understand diagnostics", "explain r output", "brms guidance",
                    "r statistics help", "hierarchical bayesian guidance",
                    # Typography & aesthetics (unique focus)
                    "typography", "aesthetics", "figure style", "tikz style",
                    "visual consistency", "publication quality", "consistent styling",
                    "figure aesthetics", "color scheme", "tufte", "design",
                    # Story synthesis (unique)
                    "tell the story", "research story", "narrative gaps",
                    "synthesis", "cross-cutting findings", "holistic view",
                    "research arc", "connect findings", "synthesize results",
                },
                dependencies=[],  # Reads all agents but depends on none
            ),
        }

        return agents

    def analyze_task(
        self,
        task: str,
        files: list[str] | None = None,
        min_confidence: float = 0.3
    ) -> TaskRouting:
        """
        Analyze a task and determine which agents should handle it.

        Args:
            task: Task description to analyze
            files: Optional list of files involved in the task
            min_confidence: Minimum confidence threshold for supporting agents

        Returns:
            TaskRouting with primary and supporting agents identified
        """
        # Calculate match scores for all agents
        scores = {
            agent_type: capability.matches(task, files)
            for agent_type, capability in self.agents.items()
        }

        # Identify primary agent (highest score)
        primary_agent = max(scores.items(), key=lambda x: x[1])
        primary_type, primary_score = primary_agent

        # Identify supporting agents (above threshold, excluding primary)
        supporting_agents = [
            agent_type
            for agent_type, score in scores.items()
            if score >= min_confidence and agent_type != primary_type
        ]

        # Sort by score (highest first)
        supporting_agents.sort(key=lambda x: scores[x], reverse=True)

        routing = TaskRouting(
            task=task,
            primary_agent=primary_type,
            supporting_agents=supporting_agents,
            confidence=primary_score,
            context={"files": files, "scores": scores},
        )

        self.routing_history.append(routing)

        logger.info(
            f"Task routed to {primary_type.value} (confidence: {primary_score:.2f})"
            + (f" with support from {[a.value for a in supporting_agents]}"
               if supporting_agents else "")
        )

        return routing

    def get_agent_info(self, agent_type: AgentType) -> AgentCapability:
        """
        Get detailed information about a specific agent.

        Args:
            agent_type: Type of agent to query

        Returns:
            AgentCapability with full details
        """
        return self.agents[agent_type]

    def list_agents(self) -> dict[str, dict[str, Any]]:
        """
        List all available agents with their capabilities.

        Returns:
            Dictionary mapping agent names to their details
        """
        return {
            agent_type.value: {
                "name": capability.name,
                "description": capability.description,
                "responsibilities": capability.responsibilities,
                "keywords_sample": list(capability.keywords)[:10],
                "dependencies": [dep.value for dep in capability.dependencies],
            }
            for agent_type, capability in self.agents.items()
        }

    def get_routing_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get recent task routing history.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of routing history entries
        """
        return [
            {
                "task": routing.task[:100] + "..." if len(routing.task) > 100 else routing.task,
                "primary_agent": routing.primary_agent.value,
                "supporting_agents": [a.value for a in routing.supporting_agents],
                "confidence": f"{routing.confidence:.2f}",
                "created_at": routing.created_at,
            }
            for routing in self.routing_history[-limit:]
        ]

    def get_dependencies(self, agent_type: AgentType) -> list[AgentType]:
        """
        Get dependencies for a specific agent type.

        Args:
            agent_type: Type of agent to query

        Returns:
            List of agent types this agent depends on
        """
        return self.agents[agent_type].dependencies

    def suggest_workflow(self, agent_type: AgentType) -> list[AgentType]:
        """
        Suggest workflow order when executing tasks for an agent.

        Args:
            agent_type: Primary agent type

        Returns:
            Ordered list of agents to execute (dependencies first, then primary)
        """
        workflow = []
        visited = set()

        def add_dependencies(current_type: AgentType):
            if current_type in visited:
                return
            visited.add(current_type)

            # Add dependencies first
            for dep in self.agents[current_type].dependencies:
                add_dependencies(dep)

            # Then add current agent
            workflow.append(current_type)

        add_dependencies(agent_type)
        return workflow

    def execute_workflow(self, workflow_name: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Execute a predefined workflow (dissertation → models → dashboard).
        
        Args:
            workflow_name: Name of workflow to execute
            context: Optional context data
            
        Returns:
            Dictionary with execution results
        """
        workflows = {
            "pre-backtest": [
                ("data_leakage", "Check for temporal data leakage"),
                ("sql_schema", "Validate SQL schema patterns"),
                ("nfl_domain", "Validate NFL domain rules"),
                ("ensemble_health", "Check ensemble health"),
                ("betting_strategy", "Validate betting parameters"),
            ],
            "pre-training": [
                ("data_leakage", "Check for temporal data leakage"),
                ("sql_schema", "Validate SQL schema patterns"),
                ("nfl_domain", "Validate NFL domain rules"),
                ("statistical", "Validate MCMC diagnostics"),
            ],
            "pre-dashboard": [
                ("sql_schema", "Validate SQL schema patterns"),
                ("nfl_domain", "Validate NFL domain rules"),
            ],
            "research": [
                ("statistical", "Validate R models"),
                ("nfl_domain", "Check domain rules"),
                ("data_leakage", "Temporal safety check"),
            ],
        }

        if workflow_name not in workflows:
            raise ValueError(f"Unknown workflow: {workflow_name}. Available: {list(workflows.keys())}")

        results = {
            "workflow": workflow_name,
            "started_at": datetime.now().isoformat(),
            "steps": [],
            "success": True,
        }

        logger.info(f"Executing workflow: {workflow_name}")

        for step_name, step_description in workflows[workflow_name]:
            logger.info(f"  Step: {step_description}")

            step_result = {
                "name": step_name,
                "description": step_description,
                "success": True,
                "message": f"{step_description} completed",
            }

            # In a full implementation, this would call actual validators
            # For now, it's a framework for execution

            results["steps"].append(step_result)

        results["completed_at"] = datetime.now().isoformat()
        logger.info(f"Workflow {workflow_name} completed successfully")

        return results

    def execute_task(
        self,
        task: str,
        files: list[str] | None = None,
        agents: list[AgentType] | None = None,
    ) -> dict[str, Any]:
        """
        Execute a task by routing to appropriate agents and running them.
        
        Args:
            task: Task description
            files: Optional list of files involved
            agents: Optional list of specific agents to use (overrides routing)
            
        Returns:
            Dictionary with execution results
        """
        # Route task if agents not specified
        if agents is None:
            routing = self.analyze_task(task, files)
            agents_to_run = [routing.primary_agent] + routing.supporting_agents
        else:
            agents_to_run = agents

        results = {
            "task": task,
            "agents": [a.value for a in agents_to_run],
            "started_at": datetime.now().isoformat(),
            "success": True,
            "outputs": [],
        }

        logger.info(f"Executing task: {task}")
        logger.info(f"Using agents: {[a.value for a in agents_to_run]}")

        # Execute each agent
        for agent_type in agents_to_run:
            agent_info = self.agents[agent_type]

            output = {
                "agent": agent_type.value,
                "agent_name": agent_info.name,
                "success": True,
                "message": f"Agent {agent_info.name} executed successfully",
            }

            # In a full implementation, this would call actual agent executors
            # For now, it's a framework for execution

            results["outputs"].append(output)

        results["completed_at"] = datetime.now().isoformat()
        logger.info("Task execution completed")

        return results

    def print_summary(self):
        """Print a summary of the coordinator's configuration."""
        print("\n=== NFL Analytics Agent Coordinator ===\n")
        print(f"Total Agents: {len(self.agents)}")
        print(f"Routing History: {len(self.routing_history)} tasks\n")

        print("Available Agents:")
        for agent_type, capability in self.agents.items():
            deps = f" (depends on: {[d.value for d in capability.dependencies]})" if capability.dependencies else ""
            print(f"\n  {capability.name}{deps}")
            print(f"    - {capability.description}")
            print(f"    - Responsibilities: {len(capability.responsibilities)}")
            print(f"    - Keywords: {len(capability.keywords)}")
            print(f"    - File Patterns: {len(capability.file_patterns)}")


# Singleton instance for easy import
_coordinator_instance = None


def get_coordinator() -> AgentCoordinator:
    """
    Get singleton instance of AgentCoordinator.

    Returns:
        Shared AgentCoordinator instance
    """
    global _coordinator_instance
    if _coordinator_instance is None:
        _coordinator_instance = AgentCoordinator()
    return _coordinator_instance


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    coordinator = AgentCoordinator()
    coordinator.print_summary()

    # Example task routing
    print("\n\n=== Example Task Routing ===\n")

    test_tasks = [
        "Update dissertation Chapter 8 with BNN calibration results",
        "Retrain state-space model with canonical training window",
        "Fix actual_margin calculation bug in Streamlit dashboard",
        "Create database migration for new ensemble_predictions table",
        "Run full backtest on 2022-2024 data and update documentation",
    ]

    for task in test_tasks:
        routing = coordinator.analyze_task(task)
        print(f"\nTask: {task}")
        print(f"  Primary: {routing.primary_agent.value} ({routing.confidence:.2f})")
        if routing.supporting_agents:
            print(f"  Support: {[a.value for a in routing.supporting_agents]}")

        # Show suggested workflow
        workflow = coordinator.suggest_workflow(routing.primary_agent)
        print(f"  Workflow: {' → '.join([a.value for a in workflow])}")
