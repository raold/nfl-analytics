#!/usr/bin/env python3
"""
Workflow Automation Examples - Single-User Edition

Demonstrates how to integrate agents into your dissertation → models → dashboard pipeline.

Author: Rovo Dev
Date: December 2025
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from py.utils.agent_coordinator import AgentCoordinator

try:
    from rich.console import Console
    from rich.panel import Panel
    console = Console()
    RICH = True
except ImportError:
    RICH = False


def example_1_pre_backtest_workflow():
    """Example 1: Full pre-backtest validation."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Pre-Backtest Workflow")
    print("="*60 + "\n")
    
    coordinator = AgentCoordinator()
    
    # Execute pre-backtest workflow
    results = coordinator.execute_workflow("pre-backtest")
    
    print(f"✓ Workflow: {results['workflow']}")
    print(f"✓ Started: {results['started_at']}")
    print(f"✓ Steps completed: {len(results['steps'])}")
    
    for step in results['steps']:
        icon = "✅" if step['success'] else "❌"
        print(f"  {icon} {step['description']}")
    
    print(f"✓ Completed: {results['completed_at']}")
    print(f"✓ Success: {results['success']}")
    
    return results


def example_2_pre_training_workflow():
    """Example 2: Pre-training validation."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Pre-Training Workflow")
    print("="*60 + "\n")
    
    coordinator = AgentCoordinator()
    
    # Execute pre-training workflow
    results = coordinator.execute_workflow("pre-training")
    
    print(f"✓ Workflow: {results['workflow']}")
    print(f"✓ Steps completed: {len(results['steps'])}")
    
    return results


def example_3_research_workflow():
    """Example 3: Research workflow (R model validation)."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Research Workflow (R Models)")
    print("="*60 + "\n")
    
    coordinator = AgentCoordinator()
    
    # Execute research workflow
    results = coordinator.execute_workflow("research")
    
    print(f"✓ Workflow: {results['workflow']}")
    print(f"✓ Validates: MCMC diagnostics, NFL domain rules, temporal safety")
    
    return results


def example_4_task_execution():
    """Example 4: Execute specific task with routing."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Task Execution with Auto-Routing")
    print("="*60 + "\n")
    
    coordinator = AgentCoordinator()
    
    # Execute a specific task (coordinator will route to correct agents)
    results = coordinator.execute_task(
        "Validate R models before ensemble backtest",
        files=["R/state_space_team_ratings.R"]
    )
    
    print(f"✓ Task: {results['task']}")
    print(f"✓ Agents used: {results['agents']}")
    print(f"✓ Outputs: {len(results['outputs'])}")
    
    for output in results['outputs']:
        icon = "✅" if output['success'] else "❌"
        print(f"  {icon} {output['agent_name']}: {output['message']}")
    
    return results


def example_5_full_pipeline():
    """Example 5: Complete dissertation → dashboard pipeline."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Full Research Pipeline")
    print("="*60 + "\n")
    
    coordinator = AgentCoordinator()
    
    pipeline = [
        ("research", "Validate R models"),
        ("pre-training", "Validate before Python training"),
        ("pre-backtest", "Validate before backtest"),
        ("pre-dashboard", "Validate before dashboard update"),
    ]
    
    print("Executing full pipeline:")
    print("  Dissertation → R Models → Python Ensemble → Backtest → Dashboard\n")
    
    all_results = []
    for workflow_name, description in pipeline:
        print(f"→ {description}...")
        results = coordinator.execute_workflow(workflow_name)
        all_results.append(results)
        
        if results['success']:
            print(f"  ✅ {workflow_name}: PASSED ({len(results['steps'])} steps)")
        else:
            print(f"  ❌ {workflow_name}: FAILED")
            break
    
    print(f"\n✓ Pipeline completed: {len(all_results)}/{len(pipeline)} workflows")
    
    return all_results


def example_6_integration_in_training():
    """Example 6: How to integrate into your training script."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Integration in Training Script")
    print("="*60 + "\n")
    
    print("Copy this code into your training script:\n")
    
    code = '''
# At top of your training script
from py.utils.agent_coordinator import AgentCoordinator

# Before training
coordinator = AgentCoordinator()
validation = coordinator.execute_workflow("pre-training")

if not validation["success"]:
    print("❌ Validation failed - aborting training")
    print(validation)
    exit(1)

print("✅ Validation passed - proceeding with training")

# Your training code here
model.fit(X_train, y_train)

# After training, validate model
model_validation = coordinator.execute_task(
    "Validate trained model",
    files=["models/xgboost/my_model.json"]
)
'''
    
    print(code)


def example_7_integration_in_backtest():
    """Example 7: How to integrate into backtest script."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Integration in Backtest Script")
    print("="*60 + "\n")
    
    print("Copy this code into your backtest script:\n")
    
    code = '''
# At top of your backtest script
from py.utils.agent_coordinator import AgentCoordinator

# Before backtest
coordinator = AgentCoordinator()
validation = coordinator.execute_workflow("pre-backtest")

if not validation["success"]:
    print("❌ Pre-backtest validation failed")
    for step in validation["steps"]:
        if not step["success"]:
            print(f"  ❌ {step['description']}: {step['message']}")
    exit(1)

print("✅ Pre-backtest validation passed")
print(f"   Validated: {len(validation['steps'])} checks")

# Your backtest code here
results = run_backtest()
'''
    
    print(code)


def main():
    """Run all examples."""
    if RICH:
        console.print(Panel.fit(
            "[bold cyan]Workflow Automation Examples[/bold cyan]\n"
            "[dim]Single-User Edition - NFL Analytics[/dim]",
            border_style="cyan"
        ))
    else:
        print("\n" + "="*60)
        print("Workflow Automation Examples")
        print("Single-User Edition - NFL Analytics")
        print("="*60)
    
    examples = [
        example_1_pre_backtest_workflow,
        example_2_pre_training_workflow,
        example_3_research_workflow,
        example_4_task_execution,
        example_5_full_pipeline,
        example_6_integration_in_training,
        example_7_integration_in_backtest,
    ]
    
    for example in examples:
        try:
            example()
            input("\nPress Enter to continue to next example...")
        except KeyboardInterrupt:
            print("\n\nExamples interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Error in example: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60 + "\n")
    
    print("Next steps:")
    print("  1. Try: python scripts/quick_validate.py --quick")
    print("  2. Read: docs/AGENTS_SINGLE_USER.md")
    print("  3. Integrate into your workflows!")


if __name__ == "__main__":
    main()
