#!/usr/bin/env python3
"""
CLI tool for NFL Analytics Agent Coordinator

This script provides a command-line interface to the AgentCoordinator for
analyzing tasks and getting agent routing recommendations.

Usage:
    # Analyze a task
    python scripts/coordinate_task.py analyze "Update dissertation with BNN results"

    # List all agents
    python scripts/coordinate_task.py list

    # Get agent details
    python scripts/coordinate_task.py info model_management

    # Show routing history
    python scripts/coordinate_task.py history

    # Suggest workflow for an agent
    python scripts/coordinate_task.py workflow prediction_engine

Examples:
    # Complex task analysis
    python scripts/coordinate_task.py analyze \
        "Retrain all models with canonical window and update dashboard" \
        --files models/bayesian/*.rds dashboard/app.py

    # Get workflow for a component
    python scripts/coordinate_task.py workflow model_management
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from py.utils.agent_coordinator import (
    AgentType,
    get_coordinator,
)


def analyze_task(args):
    """Analyze a task and show routing recommendation."""
    coordinator = get_coordinator()

    files = args.files if args.files else None
    routing = coordinator.analyze_task(args.task, files=files)

    print("\n" + "=" * 80)
    print("TASK ROUTING ANALYSIS")
    print("=" * 80)
    print(f"\nTask: {routing.task}\n")

    # Primary agent
    primary = coordinator.get_agent_info(routing.primary_agent)
    print(f"PRIMARY AGENT: {primary.name}")
    print(f"  Confidence: {routing.confidence:.1%}")
    print(f"  Description: {primary.description}")
    print("  Responsibilities:")
    for resp in primary.responsibilities[:5]:
        print(f"    • {resp}")
    if len(primary.responsibilities) > 5:
        print(f"    ... and {len(primary.responsibilities) - 5} more")

    # Supporting agents
    if routing.supporting_agents:
        print("\nSUPPORTING AGENTS:")
        for agent_type in routing.supporting_agents:
            agent = coordinator.get_agent_info(agent_type)
            score = routing.context["scores"][agent_type]
            print(f"  • {agent.name} (confidence: {score:.1%})")

    # Suggested workflow
    workflow = coordinator.suggest_workflow(routing.primary_agent)
    print("\nSUGGESTED WORKFLOW:")
    print(f"  {' → '.join([coordinator.get_agent_info(a).name for a in workflow])}")

    # Detailed scores (if verbose)
    if args.verbose:
        print("\nDETAILED SCORES:")
        for agent_type, score in sorted(
            routing.context["scores"].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            agent = coordinator.get_agent_info(agent_type)
            print(f"  {agent.name:30s} {score:.1%}")

    print("\n" + "=" * 80 + "\n")


def list_agents(args):
    """List all available agents."""
    coordinator = get_coordinator()
    agents_info = coordinator.list_agents()

    print("\n" + "=" * 80)
    print("AVAILABLE AGENTS")
    print("=" * 80 + "\n")

    for agent_key, info in agents_info.items():
        print(f"{info['name']}")
        print(f"  Type: {agent_key}")
        print(f"  Description: {info['description']}")
        print(f"  Responsibilities: {len(info['responsibilities'])}")
        if info['dependencies']:
            print(f"  Dependencies: {', '.join(info['dependencies'])}")
        if args.verbose:
            print(f"  Sample Keywords: {', '.join(info['keywords_sample'])}")
        print()

    print("=" * 80 + "\n")


def show_agent_info(args):
    """Show detailed information about a specific agent."""
    coordinator = get_coordinator()

    try:
        agent_type = AgentType(args.agent_type)
    except ValueError:
        print(f"Error: Unknown agent type '{args.agent_type}'")
        print(f"Available types: {', '.join([t.value for t in AgentType])}")
        sys.exit(1)

    capability = coordinator.get_agent_info(agent_type)

    print("\n" + "=" * 80)
    print(f"AGENT: {capability.name}")
    print("=" * 80 + "\n")

    print(f"Type: {capability.agent_type.value}")
    print(f"\nDescription:\n  {capability.description}\n")

    print(f"Responsibilities ({len(capability.responsibilities)}):")
    for i, resp in enumerate(capability.responsibilities, 1):
        print(f"  {i}. {resp}")

    if capability.dependencies:
        deps = [coordinator.get_agent_info(d).name for d in capability.dependencies]
        print("\nDependencies:")
        for dep in deps:
            print(f"  • {dep}")

    print(f"\nFile Patterns ({len(capability.file_patterns)}):")
    for pattern in capability.file_patterns[:10]:
        print(f"  • {pattern}")
    if len(capability.file_patterns) > 10:
        print(f"  ... and {len(capability.file_patterns) - 10} more")

    print(f"\nKeywords ({len(capability.keywords)}):")
    keywords_list = sorted(list(capability.keywords))
    for i in range(0, min(len(keywords_list), 30), 5):
        print(f"  {', '.join(keywords_list[i:i+5])}")
    if len(keywords_list) > 30:
        print(f"  ... and {len(keywords_list) - 30} more")

    print("\n" + "=" * 80 + "\n")


def show_workflow(args):
    """Show suggested workflow for an agent."""
    coordinator = get_coordinator()

    try:
        agent_type = AgentType(args.agent_type)
    except ValueError:
        print(f"Error: Unknown agent type '{args.agent_type}'")
        print(f"Available types: {', '.join([t.value for t in AgentType])}")
        sys.exit(1)

    workflow = coordinator.suggest_workflow(agent_type)

    print("\n" + "=" * 80)
    print(f"WORKFLOW FOR: {coordinator.get_agent_info(agent_type).name}")
    print("=" * 80 + "\n")

    print("Execution Order (dependencies first):\n")
    for i, step_type in enumerate(workflow, 1):
        agent = coordinator.get_agent_info(step_type)
        is_primary = step_type == agent_type
        marker = " ← PRIMARY AGENT" if is_primary else ""
        print(f"  {i}. {agent.name}{marker}")
        if not is_primary:
            print("     └─ Dependency: provides required data/services")

    print("\n" + "=" * 80 + "\n")


def show_history(args):
    """Show routing history."""
    coordinator = get_coordinator()
    history = coordinator.get_routing_history(limit=args.limit)

    if not history:
        print("\nNo routing history available.\n")
        return

    print("\n" + "=" * 80)
    print("ROUTING HISTORY")
    print("=" * 80 + "\n")

    for i, entry in enumerate(reversed(history), 1):
        print(f"{i}. {entry['task']}")
        print(f"   Primary: {entry['primary_agent']} (confidence: {entry['confidence']})")
        if entry['supporting_agents']:
            print(f"   Support: {', '.join(entry['supporting_agents'])}")
        print(f"   Created: {entry['created_at']}")
        print()

    print("=" * 80 + "\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="NFL Analytics Agent Coordinator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a task and get routing recommendation"
    )
    analyze_parser.add_argument("task", help="Task description to analyze")
    analyze_parser.add_argument(
        "--files",
        nargs="+",
        help="Files involved in the task"
    )
    analyze_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed scores for all agents"
    )

    # List command
    list_parser = subparsers.add_parser(
        "list",
        help="List all available agents"
    )
    list_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show additional details"
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show detailed information about a specific agent"
    )
    info_parser.add_argument(
        "agent_type",
        help="Agent type (e.g., model_management, dissertation)"
    )

    # Workflow command
    workflow_parser = subparsers.add_parser(
        "workflow",
        help="Show suggested workflow for an agent"
    )
    workflow_parser.add_argument(
        "agent_type",
        help="Agent type (e.g., model_management, prediction_engine)"
    )

    # History command
    history_parser = subparsers.add_parser(
        "history",
        help="Show recent routing history"
    )
    history_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of entries to show (default: 10)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Route to appropriate handler
    handlers = {
        "analyze": analyze_task,
        "list": list_agents,
        "info": show_agent_info,
        "workflow": show_workflow,
        "history": show_history,
    }

    handlers[args.command](args)


if __name__ == "__main__":
    main()
