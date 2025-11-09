#!/usr/bin/env python3
"""
Practical Integration Examples for Agent Coordinator

This script demonstrates real-world integration patterns for using the
Agent Coordinator system in the NFL analytics project.

Examples:
1. Pre-task planning and workflow analysis
2. Automated task routing based on file changes
3. Multi-component orchestration
4. Integration with existing training scripts
5. Production deployment coordination

Usage:
    python examples/agent_coordinator_integration.py example1
    python examples/agent_coordinator_integration.py example2
    python examples/agent_coordinator_integration.py example3
"""

import sys
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from py.utils.agent_coordinator import (
    AgentCoordinator,
    AgentType,
    get_coordinator,
)


def example1_pre_task_planning():
    """
    Example 1: Pre-Task Planning

    Use coordinator before starting complex work to understand scope and dependencies.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Pre-Task Planning")
    print("=" * 80 + "\n")

    coordinator = get_coordinator()

    # Complex task to analyze
    task = """
    Retrain all ensemble models with canonical training window (2018-2024),
    update the prediction API with new models, refresh Grafana dashboard
    with latest metrics, and document changes in milestone doc.
    """

    print(f"Task: {task.strip()}\n")

    # Analyze task
    routing = coordinator.analyze_task(task)

    print(f"Analysis:")
    print(f"  Primary Agent: {routing.primary_agent.value}")
    print(f"  Confidence: {routing.confidence:.1%}")

    if routing.supporting_agents:
        print(f"  Supporting Agents:")
        for agent_type in routing.supporting_agents:
            agent = coordinator.get_agent_info(agent_type)
            print(f"    • {agent.name}")

    # Get recommended workflow
    workflow = coordinator.suggest_workflow(routing.primary_agent)

    print(f"\nRecommended Execution Order:")
    for i, agent_type in enumerate(workflow, 1):
        agent = coordinator.get_agent_info(agent_type)
        is_primary = agent_type == routing.primary_agent
        marker = " ← PRIMARY" if is_primary else ""
        print(f"  {i}. {agent.name}{marker}")

    print(f"\nExecution Plan:")
    for agent_type in workflow:
        agent = coordinator.get_agent_info(agent_type)
        print(f"\n  {agent.name}:")

        if agent_type == AgentType.DATA_LAYER:
            print(f"    • Validate training data (2018-2024)")
            print(f"    • Check feature engineering pipeline")
            print(f"    • Verify materialized views are current")

        elif agent_type == AgentType.MODEL_MANAGEMENT:
            print(f"    • Retrain Bayesian hierarchical model")
            print(f"    • Retrain XGBoost model")
            print(f"    • Retrain BNN model")
            print(f"    • Update model registry")

        elif agent_type == AgentType.PREDICTION_ENGINE:
            print(f"    • Integrate new models into ensemble")
            print(f"    • Optimize ensemble weights")
            print(f"    • Update prediction API")
            print(f"    • Run validation tests")

        elif agent_type == AgentType.FRONTEND:
            print(f"    • Update Grafana panels with new metrics")
            print(f"    • Refresh Streamlit dashboard")
            print(f"    • Verify visualization accuracy")

        elif agent_type == AgentType.DOCUMENTATION:
            print(f"    • Create milestone document")
            print(f"    • Update PROJECT_STATUS.md")
            print(f"    • Document model versions")


def example2_file_based_routing():
    """
    Example 2: File-Based Task Routing

    Automatically route tasks based on files being modified.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: File-Based Task Routing")
    print("=" * 80 + "\n")

    coordinator = get_coordinator()

    # Simulate detecting changed files
    scenarios = [
        {
            "task": "Update model performance metrics",
            "files": [
                "models/bayesian/bnn_passing_v1.pkl",
                "models/bayesian/bnn_passing_v1_metadata.json"
            ]
        },
        {
            "task": "Fix visualization bug",
            "files": [
                "dashboard/app.py",
                "grafana/provisioning/dashboards/system.json"
            ]
        },
        {
            "task": "Add new feature columns",
            "files": [
                "db/migrations/027_new_features.sql",
                "data/feature_engineering.py"
            ]
        },
    ]

    for scenario in scenarios:
        print(f"Scenario: {scenario['task']}")
        print(f"Files: {', '.join(scenario['files'])}")

        routing = coordinator.analyze_task(
            scenario['task'],
            files=scenario['files']
        )

        agent = coordinator.get_agent_info(routing.primary_agent)
        print(f"Routed to: {agent.name} (confidence: {routing.confidence:.1%})")
        print()


def example3_multi_component_orchestration():
    """
    Example 3: Multi-Component Task Orchestration

    Coordinate complex tasks requiring multiple specialized agents.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Multi-Component Task Orchestration")
    print("=" * 80 + "\n")

    coordinator = get_coordinator()

    # Complex multi-component task
    task = "Full production deployment pipeline"

    print(f"Task: {task}\n")

    # Define subtasks for each component
    subtasks = [
        ("Data Layer", [
            "Validate production data integrity",
            "Run feature engineering pipeline",
            "Refresh materialized views",
        ]),
        ("Model Management", [
            "Load trained models",
            "Validate model checksums",
            "Update model registry",
        ]),
        ("Prediction Engine", [
            "Initialize ensemble predictor",
            "Run validation predictions",
            "Deploy to production API",
        ]),
        ("Frontend", [
            "Update Grafana dashboards",
            "Refresh Streamlit displays",
            "Enable production monitoring",
        ]),
        ("Documentation", [
            "Create deployment milestone doc",
            "Update production checklist",
            "Log deployment metadata",
        ]),
    ]

    # Get workflow for prediction engine (our primary)
    workflow = coordinator.suggest_workflow(AgentType.PREDICTION_ENGINE)

    print("Orchestrated Execution Plan:\n")

    for agent_type in workflow:
        agent = coordinator.get_agent_info(agent_type)
        print(f"STEP: {agent.name}")

        # Find matching subtasks
        for component_name, tasks in subtasks:
            if component_name.lower().replace(" ", "_") in agent.name.lower():
                for task in tasks:
                    print(f"  ✓ {task}")
                break

        print()


def example4_training_script_integration():
    """
    Example 4: Integration with Training Scripts

    Show how to integrate coordinator into existing training workflows.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Training Script Integration")
    print("=" * 80 + "\n")

    coordinator = get_coordinator()

    # Simulate training script context
    model_type = "BNN Passing"
    training_window = "2018-2024"

    task = f"Train {model_type} model with {training_window} data"
    files = [
        "py/models/train_bnn_passing.py",
        "models/bayesian/bnn_passing_v2.pkl"
    ]

    print(f"Training Script: train_bnn_passing.py")
    print(f"Task: {task}\n")

    # Analyze task
    routing = coordinator.analyze_task(task, files=files)

    print(f"Pre-Training Analysis:")
    print(f"  Primary Agent: {routing.primary_agent.value}")

    # Check dependencies
    deps = coordinator.get_dependencies(routing.primary_agent)

    if deps:
        print(f"\n  Dependencies Required:")
        for dep in deps:
            dep_agent = coordinator.get_agent_info(dep)
            print(f"    • {dep_agent.name}")

    # Simulate dependency checks
    print(f"\n  Checking Dependencies:")
    print(f"    ✓ Database connection available")
    print(f"    ✓ Training data validated (2018-2024)")
    print(f"    ✓ Feature engineering complete")
    print(f"    ✓ GPU resources available")

    print(f"\n  Proceeding with model training...")
    print(f"    [Simulated training output]")
    print(f"    Epoch 1/100: loss=45.2, val_loss=47.8")
    print(f"    Epoch 2/100: loss=43.1, val_loss=46.2")
    print(f"    ...")

    # Post-training coordination
    print(f"\n  Post-Training Coordination:")
    print(f"    ✓ Model saved: models/bayesian/bnn_passing_v2.pkl")
    print(f"    ✓ Metadata saved: models/bayesian/bnn_passing_v2_metadata.json")
    print(f"    ⚠ Consider updating:")
    print(f"      • Prediction Engine: Integrate new model into ensemble")
    print(f"      • Frontend: Update dashboard with new metrics")
    print(f"      • Documentation: Log training results")


def example5_agent_capabilities_query():
    """
    Example 5: Querying Agent Capabilities

    Dynamically query agent capabilities for runtime decision-making.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Agent Capabilities Query")
    print("=" * 80 + "\n")

    coordinator = get_coordinator()

    # Query all agents
    agents = coordinator.list_agents()

    print(f"Available Agents: {len(agents)}\n")

    # Find agents that can handle specific responsibilities
    print("Finding agents for specific tasks:\n")

    queries = [
        ("Who can compile LaTeX?", "latex"),
        ("Who handles model training?", "train"),
        ("Who manages database schema?", "schema"),
        ("Who can update dashboards?", "dashboard"),
    ]

    for query, keyword in queries:
        print(f"Q: {query}")
        matching_agents = []

        for agent_type in AgentType:
            if agent_type == AgentType.GENERAL:
                continue

            capability = coordinator.get_agent_info(agent_type)

            # Check if keyword in agent keywords or responsibilities
            if (keyword in capability.keywords or
                any(keyword in resp.lower() for resp in capability.responsibilities)):
                matching_agents.append(capability.name)

        print(f"A: {', '.join(matching_agents)}\n")


def main():
    """Main entry point for examples."""

    if len(sys.argv) < 2:
        print("NFL Analytics Agent Coordinator - Integration Examples\n")
        print("Usage: python examples/agent_coordinator_integration.py <example_number>")
        print("\nAvailable examples:")
        print("  example1 - Pre-Task Planning")
        print("  example2 - File-Based Task Routing")
        print("  example3 - Multi-Component Task Orchestration")
        print("  example4 - Training Script Integration")
        print("  example5 - Agent Capabilities Query")
        print("  all      - Run all examples")
        sys.exit(1)

    example = sys.argv[1].lower()

    examples = {
        'example1': example1_pre_task_planning,
        'example2': example2_file_based_routing,
        'example3': example3_multi_component_orchestration,
        'example4': example4_training_script_integration,
        'example5': example5_agent_capabilities_query,
    }

    if example == 'all':
        for func in examples.values():
            func()
            print("\n" + "-" * 80 + "\n")
    elif example in examples:
        examples[example]()
    else:
        print(f"Error: Unknown example '{example}'")
        print(f"Available: {', '.join(examples.keys())}, all")
        sys.exit(1)


if __name__ == "__main__":
    main()
