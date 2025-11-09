# Agent Coordinator Guide

**Last Updated:** November 9, 2025
**Purpose:** Guide for using the Agent Coordinator system to intelligently route tasks to specialized agents

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Specialized Agents](#specialized-agents)
4. [Usage](#usage)
5. [CLI Tool](#cli-tool)
6. [Python API](#python-api)
7. [Integration Patterns](#integration-patterns)
8. [Examples](#examples)

---

## Overview

The Agent Coordinator is an intelligent task routing system that analyzes complex requests and delegates them to appropriate specialized agents based on:

- **Keywords** in the task description
- **File patterns** involved in the task
- **Component responsibilities** and capabilities
- **Agent dependencies** and workflow ordering

### Key Benefits

- **Intelligent Routing**: Automatically identifies which components are involved in a task
- **Multi-Agent Coordination**: Handles complex tasks requiring multiple specialized agents
- **Dependency Management**: Ensures tasks execute in correct order based on dependencies
- **Context Preservation**: Maintains routing history for analysis and debugging
- **Scalable**: Easy to add new agents or modify existing capabilities

---

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                    AgentCoordinator                          │
│  - Analyzes incoming tasks                                   │
│  - Routes to appropriate agents                              │
│  - Manages dependencies and workflows                        │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
    ┌────────────────────┐      ┌────────────────────┐
    │  Agent Registry    │      │  Routing Engine    │
    │  - Capabilities    │      │  - Keyword match   │
    │  - Responsibilities│      │  - File pattern    │
    │  - Dependencies    │      │  - Score calculate │
    └────────────────────┘      └────────────────────┘
                │
                ▼
    ┌──────────────────────────────────────────────┐
    │          Specialized Agents                   │
    │  1. Dissertation                              │
    │  2. Model Management                          │
    │  3. Frontend                                  │
    │  4. Data Layer                                │
    │  5. Prediction Engine                         │
    │  6. Documentation                             │
    └──────────────────────────────────────────────┘
```

### Routing Algorithm

1. **Keyword Analysis**: Scans task description for agent-specific keywords (50% weight)
2. **File Pattern Matching**: Matches involved files to agent file patterns (30% weight)
3. **Responsibility Matching**: Analyzes task against agent responsibilities (20% weight)
4. **Score Calculation**: Combines weighted scores to identify primary and supporting agents
5. **Workflow Generation**: Orders agents based on dependencies

---

## Specialized Agents

### 1. Dissertation Agent

**Purpose:** Manages LaTeX dissertation compilation, figures, and content

**Responsibilities:**
- Compile LaTeX documents
- Update dissertation chapters
- Generate TikZ figures
- Manage bibliography and references
- Create tables from results
- Track dissertation TODOs
- Audit dissertation completeness

**File Patterns:**
- `analysis/dissertation/`
- `*.tex`
- `*.bib`
- `figures/`

**Keywords:**
- dissertation, latex, tikz, compile, chapter
- figure, table, bibliography, citation, reference
- writeup, thesis, subsection, section

**Dependencies:** None

---

### 2. Model Management Agent

**Purpose:** Handles model training, evaluation, versioning, and registry

**Responsibilities:**
- Train Bayesian models (brms, Stan, PyMC)
- Train ML models (XGBoost, BNN)
- Evaluate model performance
- Version model artifacts
- Update model registry
- Track training runs
- Manage model metadata
- Hyperparameter optimization

**File Patterns:**
- `py/models/`
- `R/train_*`
- `models/`
- `*.pkl`, `*.rds`, `*.pt`
- `*_metadata.json`

**Keywords:**
- train, model, bayesian, xgboost, bnn, neural network
- hierarchical, state-space, ensemble, prediction
- hyperparameter, optimization, calibration, evaluation
- brms, stan, pymc, cmdstanr, pytorch

**Dependencies:** Data Layer

---

### 3. Frontend Agent

**Purpose:** Manages dashboards, monitoring, and visualization

**Responsibilities:**
- Update Streamlit dashboard
- Configure Grafana panels
- Create visualizations
- Monitor system health
- Display predictions
- Track model metrics
- User interface updates

**File Patterns:**
- `dashboard/`
- `grafana/`
- `app.py`
- `dashboards.yaml`
- `*.json`

**Keywords:**
- dashboard, streamlit, grafana, visualization, plot
- chart, panel, monitoring, metrics, ui, interface
- display, frontend, web

**Dependencies:** Data Layer, Model Management

---

### 4. Data Layer Agent

**Purpose:** Manages database schema, migrations, queries, and data integrity

**Responsibilities:**
- Create database migrations
- Update schema definitions
- Write optimized queries
- Validate data integrity
- Manage materialized views
- Ingest new data sources
- Clean and transform data
- Feature engineering

**File Patterns:**
- `db/`
- `data/`
- `*.sql`
- `queries/`
- `migrations/`
- `ingest_*`

**Keywords:**
- database, schema, migration, query, sql, postgres
- ingest, data, etl, transform, clean, feature
- materialized view, table, column, index

**Dependencies:** None

---

### 5. Prediction Engine Agent

**Purpose:** Handles ensemble integration, backtesting, and production deployment

**Responsibilities:**
- Integrate ensemble methods
- Run backtests
- Generate predictions
- Optimize ensemble weights
- Deploy to production
- Risk gate validation
- ROI tracking
- A/B testing

**File Patterns:**
- `py/ensemble/`
- `py/backtest/`
- `py/production/`
- `py/api/`

**Keywords:**
- ensemble, backtest, prediction, production, deploy
- roi, risk, bet, betting, odds, props
- weight, optimization, integration, api

**Dependencies:** Model Management, Data Layer

---

### 6. Documentation Agent

**Purpose:** Manages documentation, GitHub, and project tracking

**Responsibilities:**
- Update project documentation
- Maintain CLAUDE.md
- Track project status
- Create milestone docs
- Manage GitHub issues/PRs
- Update README files
- Generate documentation indexes
- Track technical debt

**File Patterns:**
- `docs/`
- `README.md`
- `CLAUDE.md`
- `PROJECT_STATUS.md`
- `*.md`

**Keywords:**
- documentation, docs, readme, guide, github
- status, milestone, tracking, update, maintain
- index, catalog, changelog

**Dependencies:** None

---

## Usage

### CLI Tool

The coordinator includes a CLI tool for interactive task analysis.

#### Installation

```bash
# Already available in scripts/coordinate_task.py
chmod +x scripts/coordinate_task.py
```

#### Commands

**1. Analyze a Task**
```bash
python scripts/coordinate_task.py analyze "Your task description here"
```

**2. List All Agents**
```bash
python scripts/coordinate_task.py list
python scripts/coordinate_task.py list -v  # verbose mode
```

**3. Get Agent Details**
```bash
python scripts/coordinate_task.py info model_management
```

**4. Show Workflow**
```bash
python scripts/coordinate_task.py workflow prediction_engine
```

**5. View Routing History**
```bash
python scripts/coordinate_task.py history
python scripts/coordinate_task.py history --limit 20
```

---

### Python API

#### Basic Usage

```python
from py.utils.agent_coordinator import get_coordinator, AgentType

# Get coordinator instance
coordinator = get_coordinator()

# Analyze a task
routing = coordinator.analyze_task(
    "Retrain BNN with canonical training window",
    files=["py/models/train_bnn_passing.py", "models/bayesian/"]
)

print(f"Primary: {routing.primary_agent.value}")
print(f"Confidence: {routing.confidence:.1%}")
print(f"Supporting: {[a.value for a in routing.supporting_agents]}")

# Get suggested workflow
workflow = coordinator.suggest_workflow(routing.primary_agent)
print(f"Workflow: {[a.value for a in workflow]}")
```

#### Advanced Usage

```python
from py.utils.agent_coordinator import get_coordinator, AgentType

coordinator = get_coordinator()

# Get agent capabilities
dissertation_agent = coordinator.get_agent_info(AgentType.DISSERTATION)
print(f"Responsibilities: {dissertation_agent.responsibilities}")

# List all agents
agents = coordinator.list_agents()
for name, info in agents.items():
    print(f"{name}: {info['description']}")

# Check dependencies
deps = coordinator.get_dependencies(AgentType.PREDICTION_ENGINE)
print(f"Dependencies: {[d.value for d in deps]}")

# View routing history
history = coordinator.get_routing_history(limit=5)
for entry in history:
    print(f"{entry['task']}: {entry['primary_agent']}")
```

---

## Integration Patterns

### Pattern 1: Pre-Task Analysis

Use before starting complex work to understand which agents will be involved:

```python
from py.utils.agent_coordinator import get_coordinator

coordinator = get_coordinator()

# Analyze task before starting
task = "Update all models with canonical window and refresh dashboard"
routing = coordinator.analyze_task(task)

# Get workflow
workflow = coordinator.suggest_workflow(routing.primary_agent)

# Execute in order
for agent_type in workflow:
    agent = coordinator.get_agent_info(agent_type)
    print(f"Executing: {agent.name}")
    # ... perform agent-specific work ...
```

### Pattern 2: File-Based Routing

Route tasks based on files being modified:

```python
from py.utils.agent_coordinator import get_coordinator

coordinator = get_coordinator()

# Files being modified
files = [
    "analysis/dissertation/chapter_8_results_discussion/chapter_8_results_discussion.tex",
    "models/bayesian/bnn_passing_v1.pkl",
]

routing = coordinator.analyze_task(
    "Update results chapter with new BNN calibration metrics",
    files=files
)

# Will correctly identify Dissertation as primary, Model Management as supporting
```

### Pattern 3: Multi-Component Orchestration

Coordinate complex tasks across multiple components:

```python
from py.utils.agent_coordinator import get_coordinator, AgentType

coordinator = get_coordinator()

# Complex task requiring multiple agents
task = "Full production deployment: retrain models, update API, refresh dashboard"

routing = coordinator.analyze_task(task)

# Get full workflow including dependencies
workflow = coordinator.suggest_workflow(routing.primary_agent)

print("Execution plan:")
for i, agent_type in enumerate(workflow, 1):
    agent = coordinator.get_agent_info(agent_type)
    deps = coordinator.get_dependencies(agent_type)
    deps_str = f" (depends on: {[d.value for d in deps]})" if deps else ""
    print(f"{i}. {agent.name}{deps_str}")
```

---

## Examples

### Example 1: Dissertation Update

```bash
$ python scripts/coordinate_task.py analyze \
    "Update Chapter 8 with BNN calibration results and regenerate figures"

================================================================================
TASK ROUTING ANALYSIS
================================================================================

Task: Update Chapter 8 with BNN calibration results and regenerate figures

PRIMARY AGENT: Dissertation Agent
  Confidence: 78.5%
  Description: Manages LaTeX dissertation compilation, figures, and content
  Responsibilities:
    • Compile LaTeX documents
    • Update dissertation chapters
    • Generate TikZ figures
    • Manage bibliography and references
    • Create tables from results

SUPPORTING AGENTS:
  • Model Management Agent (confidence: 42.3%)

SUGGESTED WORKFLOW:
  Dissertation Agent

================================================================================
```

### Example 2: Model Retraining

```bash
$ python scripts/coordinate_task.py analyze \
    "Retrain all Bayesian models with canonical training window (2018-2024)" \
    --files models/bayesian/*.rds py/models/train_*.py

================================================================================
TASK ROUTING ANALYSIS
================================================================================

Task: Retrain all Bayesian models with canonical training window (2018-2024)

PRIMARY AGENT: Model Management Agent
  Confidence: 91.2%
  Description: Handles model training, evaluation, versioning, and registry
  Responsibilities:
    • Train Bayesian models (brms, Stan, PyMC)
    • Train ML models (XGBoost, BNN)
    • Evaluate model performance
    • Version model artifacts
    • Update model registry

SUPPORTING AGENTS:
  • Documentation Agent (confidence: 35.1%)

SUGGESTED WORKFLOW:
  Data Layer Agent → Model Management Agent

================================================================================
```

### Example 3: Full Production Pipeline

```bash
$ python scripts/coordinate_task.py analyze \
    "Deploy production pipeline: retrain ensemble, update API, refresh Grafana dashboard, update docs"

================================================================================
TASK ROUTING ANALYSIS
================================================================================

Task: Deploy production pipeline: retrain ensemble, update API, refresh dashboard, update docs

PRIMARY AGENT: Prediction Engine Agent
  Confidence: 85.7%
  Description: Handles ensemble integration, backtesting, and production deployment
  Responsibilities:
    • Integrate ensemble methods
    • Run backtests
    • Generate predictions
    • Optimize ensemble weights
    • Deploy to production

SUPPORTING AGENTS:
  • Model Management Agent (confidence: 68.4%)
  • Frontend Agent (confidence: 54.2%)
  • Documentation Agent (confidence: 41.8%)

SUGGESTED WORKFLOW:
  Data Layer Agent → Model Management Agent → Prediction Engine Agent

================================================================================
```

### Example 4: Python API Usage

```python
#!/usr/bin/env python3
"""Example: Using coordinator in a training script"""

from py.utils.agent_coordinator import get_coordinator

def main():
    coordinator = get_coordinator()

    # Analyze what you're about to do
    task = "Train BNN passing model with updated features"
    routing = coordinator.analyze_task(
        task,
        files=["py/models/train_bnn_passing.py"]
    )

    print(f"Task: {task}")
    print(f"Primary Agent: {routing.primary_agent.value}")

    # Get workflow
    workflow = coordinator.suggest_workflow(routing.primary_agent)

    print(f"\nRecommended workflow:")
    for i, agent_type in enumerate(workflow, 1):
        agent = coordinator.get_agent_info(agent_type)
        print(f"  {i}. {agent.name}")

    # Check dependencies are met
    deps = coordinator.get_dependencies(routing.primary_agent)
    if deps:
        print(f"\nEnsure these dependencies are ready:")
        for dep in deps:
            dep_agent = coordinator.get_agent_info(dep)
            print(f"  • {dep_agent.name}")

    # Proceed with training...
    print(f"\nProceeding with model training...")

if __name__ == "__main__":
    main()
```

### Example 5: Viewing Agent Capabilities

```bash
$ python scripts/coordinate_task.py info prediction_engine

================================================================================
AGENT: Prediction Engine Agent
================================================================================

Type: prediction_engine

Description:
  Handles ensemble integration, backtesting, and production deployment

Responsibilities (8):
  1. Integrate ensemble methods
  2. Run backtests
  3. Generate predictions
  4. Optimize ensemble weights
  5. Deploy to production
  6. Risk gate validation
  7. ROI tracking
  8. A/B testing

Dependencies:
  • Model Management Agent
  • Data Layer Agent

File Patterns (4):
  • py/ensemble/
  • py/backtest/
  • py/production/
  • py/api/

Keywords (13):
  ensemble, backtest, prediction, production, deploy
  roi, risk, bet, betting, odds
  props, weight, optimization

================================================================================
```

---

## Best Practices

### 1. Use for Task Planning

Before starting complex work, analyze the task to understand all components involved:

```python
routing = coordinator.analyze_task("Your complex task")
workflow = coordinator.suggest_workflow(routing.primary_agent)
# Now you know the full scope and order of work
```

### 2. Provide File Context

When analyzing tasks, include files for better routing accuracy:

```python
routing = coordinator.analyze_task(
    "Update model performance",
    files=["models/bayesian/bnn_v1.pkl", "dashboard/app.py"]
)
# More accurate routing based on actual files involved
```

### 3. Check Dependencies

Before executing agent-specific work, verify dependencies are ready:

```python
deps = coordinator.get_dependencies(AgentType.PREDICTION_ENGINE)
for dep in deps:
    # Verify dependency is ready before proceeding
    pass
```

### 4. Monitor Routing History

Review routing history to improve coordinator configuration:

```python
history = coordinator.get_routing_history(limit=20)
# Analyze patterns to refine agent capabilities
```

### 5. Extend Agent Capabilities

As project evolves, update agent keywords, file patterns, and responsibilities in `py/utils/agent_coordinator.py`.

---

## Troubleshooting

### Low Confidence Scores

If routing confidence is low (<50%), consider:
- Adding more specific keywords to task description
- Providing file context
- Checking if task spans multiple components (should have supporting agents)

### Wrong Agent Selected

If routing selects unexpected agent:
- Review agent keywords and file patterns
- Update coordinator configuration
- Provide more specific task description

### Missing Dependencies

If workflow doesn't include expected dependencies:
- Check agent dependency configuration
- Verify dependency relationships are correctly defined

---

## Future Enhancements

Planned improvements to the coordinator:

1. **ML-Based Routing**: Train classifier on historical routing decisions
2. **Real-Time Coordination**: Integration with Claude Code Task tool
3. **Performance Tracking**: Monitor agent execution times and success rates
4. **Auto-Configuration**: Learn optimal agent configurations from usage patterns
5. **Parallel Execution**: Support for parallel agent execution when dependencies allow

---

## Related Documentation

- [CLAUDE.md](../CLAUDE.md) - Project overview and component details
- [COMMON_TASKS_REFERENCE.md](guides/COMMON_TASKS_REFERENCE.md) - Task-specific guides
- [ARCHITECTURE_REFERENCE.md](ARCHITECTURE_REFERENCE.md) - System architecture
- [COMPONENT_DEEP_DIVES.md](COMPONENT_DEEP_DIVES.md) - Detailed component guides

---

**Maintainer:** Claude Code (Sonnet 4.5)
**Version:** 1.0
**Last Updated:** November 9, 2025
