#!/usr/bin/env python3
"""
Generate hyperparameter sensitivity heatmap for RL agents.

Creates a grid visualization showing how different hyperparameter combinations
affect agent performance. Used for dissertation Figure in Chapter 5.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def load_sweep_results(results_path: Path) -> Dict:
    """Load hyperparameter sweep results from JSON."""
    with open(results_path) as f:
        return json.load(f)


def extract_grid_data(
    results: Dict,
    param1: str,
    param2: str,
    metric: str = "final_reward"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract 2D grid data for heatmap.

    Args:
        results: Sweep results dict or list
        param1: First hyperparameter name (x-axis)
        param2: Second hyperparameter name (y-axis)
        metric: Performance metric to plot

    Returns:
        param1_vals, param2_vals, metric_grid (2D array)
    """
    # Handle different result formats
    if isinstance(results, dict) and "experiments" in results:
        experiments = results["experiments"]
    elif isinstance(results, list):
        experiments = results
    else:
        raise ValueError("Unknown results format")

    # Extract unique parameter values
    param1_vals = sorted(set(exp["params"][param1] for exp in experiments if param1 in exp["params"]))
    param2_vals = sorted(set(exp["params"][param2] for exp in experiments if param2 in exp["params"]))

    # Build grid
    grid = np.full((len(param2_vals), len(param1_vals)), np.nan)

    for exp in experiments:
        if param1 not in exp["params"] or param2 not in exp["params"]:
            continue

        p1_val = exp["params"][param1]
        p2_val = exp["params"][param2]

        i = param2_vals.index(p2_val)
        j = param1_vals.index(p1_val)

        # Extract metric value
        if metric in exp:
            grid[i, j] = exp[metric]
        elif "metrics" in exp and metric in exp["metrics"]:
            grid[i, j] = exp["metrics"][metric]
        elif "final_metrics" in exp and metric in exp["final_metrics"]:
            grid[i, j] = exp["final_metrics"][metric]

    return np.array(param1_vals), np.array(param2_vals), grid


def plot_sensitivity_heatmap(
    results_path: Path,
    output_path: Path,
    param1: str = "learning_rate",
    param2: str = "entropy_coef",
    metric: str = "final_reward",
    title: Optional[str] = None,
) -> None:
    """
    Create hyperparameter sensitivity heatmap.

    Args:
        results_path: Path to sweep results JSON
        output_path: Path to save output PNG
        param1: First hyperparameter (x-axis)
        param2: Second hyperparameter (y-axis)
        metric: Performance metric to visualize
        title: Plot title (auto-generated if None)
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        print("ERROR: matplotlib not installed. Install with: pip install matplotlib")
        sys.exit(1)

    # Load results
    results = load_sweep_results(results_path)

    # Extract grid data
    param1_vals, param2_vals, grid = extract_grid_data(results, param1, param2, metric)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    # Determine if we should use log scale for parameters
    use_log_x = param1_vals.max() / param1_vals.min() > 100 if len(param1_vals) > 1 else False
    use_log_y = param2_vals.max() / param2_vals.min() > 100 if len(param2_vals) > 1 else False

    # Create heatmap
    im = ax.imshow(
        grid,
        aspect='auto',
        origin='lower',
        cmap='RdYlGn',
        interpolation='nearest',
        extent=[
            param1_vals[0], param1_vals[-1],
            param2_vals[0], param2_vals[-1]
        ]
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric.replace('_', ' ').title(), fontsize=10)

    # Set labels
    ax.set_xlabel(param1.replace('_', ' ').title(), fontsize=11)
    ax.set_ylabel(param2.replace('_', ' ').title(), fontsize=11)

    # Set title
    if title is None:
        title = f'Hyperparameter Sensitivity: {metric.replace("_", " ").title()}'
    ax.set_title(title, fontsize=12)

    # Apply log scale if needed
    if use_log_x:
        ax.set_xscale('log')
    if use_log_y:
        ax.set_yscale('log')

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Annotate cells with values
    for i in range(len(param2_vals)):
        for j in range(len(param1_vals)):
            if not np.isnan(grid[i, j]):
                text = ax.text(
                    param1_vals[j], param2_vals[i],
                    f'{grid[i, j]:.3f}',
                    ha='center', va='center',
                    fontsize=8,
                    color='black' if grid[i, j] > np.nanmean(grid) else 'white'
                )

    # Tight layout
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved sensitivity heatmap to: {output_path}")
    print(f"   {param1}: {len(param1_vals)} values ({param1_vals[0]:.2e} to {param1_vals[-1]:.2e})")
    print(f"   {param2}: {len(param2_vals)} values ({param2_vals[0]:.2e} to {param2_vals[-1]:.2e})")
    print(f"   Metric range: {np.nanmin(grid):.4f} to {np.nanmax(grid):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot hyperparameter sensitivity heatmap"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("models/hparam_sweep_results.json"),
        help="Path to sweep results JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/dissertation/figures/hparam_sensitivity.png"),
        help="Output PNG path",
    )
    parser.add_argument(
        "--param1",
        default="learning_rate",
        help="First hyperparameter (x-axis)",
    )
    parser.add_argument(
        "--param2",
        default="entropy_coef",
        help="Second hyperparameter (y-axis)",
    )
    parser.add_argument(
        "--metric",
        default="final_reward",
        help="Performance metric to visualize",
    )
    parser.add_argument(
        "--title",
        help="Custom plot title",
    )

    args = parser.parse_args()

    # Check input exists
    if not args.results.exists():
        print(f"ERROR: Results file not found: {args.results}")
        print("\nExpected JSON format:")
        print("""
{
  "experiments": [
    {
      "params": {"learning_rate": 0.001, "entropy_coef": 0.01},
      "final_reward": 0.523
    },
    ...
  ]
}

OR

[
  {
    "params": {"learning_rate": 0.001, "entropy_coef": 0.01},
    "metrics": {"final_reward": 0.523}
  },
  ...
]
        """)
        return 1

    # Generate plot
    try:
        plot_sensitivity_heatmap(
            results_path=args.results,
            output_path=args.output,
            param1=args.param1,
            param2=args.param2,
            metric=args.metric,
            title=args.title,
        )
        return 0
    except Exception as e:
        print(f"ERROR: Failed to generate plot: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
