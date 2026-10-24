"""
Generate CQL training curves visualization.

Usage:
    python py/viz/plot_cql_curves.py \
      --input models/cql/cql_training_log.json \
      --output analysis/dissertation/figures/out/cql_training_curves.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CQL training curves")
    parser.add_argument("--input", required=True, help="Path to CQL training log JSON")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI (default: 300)")
    return parser.parse_args()


def plot_training_curves(log_path: Path, output_path: Path, dpi: int = 300):
    """Generate multi-panel training curves plot."""
    # Load training log
    with open(log_path) as f:
        epochs = json.load(f)

    # Extract metrics
    epoch_nums = [e["epoch"] for e in epochs]
    total_loss = [e["loss"] for e in epochs]
    td_loss = [e["td_loss"] for e in epochs]
    cql_loss = [e["cql_loss"] for e in epochs]
    q_mean = [e["q_mean"] for e in epochs]

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("CQL Training Curves (2000 Epochs)", fontsize=14, fontweight="bold")

    # Define consistent styling
    color_total = "#2E86AB"
    color_td = "#A23B72"
    color_cql = "#F18F01"
    color_q = "#C73E1D"

    # Plot 1: Total Loss
    axes[0, 0].plot(epoch_nums, total_loss, color=color_total, linewidth=1.5, alpha=0.8)
    axes[0, 0].set_title("Total Loss", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3, linestyle="--")
    axes[0, 0].set_ylim(bottom=0)

    # Plot 2: TD Error
    axes[0, 1].plot(epoch_nums, td_loss, color=color_td, linewidth=1.5, alpha=0.8)
    axes[0, 1].set_title("TD Error", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("TD Loss")
    axes[0, 1].grid(True, alpha=0.3, linestyle="--")
    axes[0, 1].set_ylim(bottom=0)

    # Plot 3: CQL Penalty
    axes[1, 0].plot(epoch_nums, cql_loss, color=color_cql, linewidth=1.5, alpha=0.8)
    axes[1, 0].set_title("CQL Penalty", fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("CQL Loss")
    axes[1, 0].grid(True, alpha=0.3, linestyle="--")
    axes[1, 0].set_ylim(bottom=0)

    # Plot 4: Mean Q-Value
    axes[1, 1].plot(epoch_nums, q_mean, color=color_q, linewidth=1.5, alpha=0.8)
    axes[1, 1].set_title("Mean Q-Value", fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Q Mean")
    axes[1, 1].grid(True, alpha=0.3, linestyle="--")
    axes[1, 1].axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    # Add final values as text annotations
    for ax, values, label in zip(
        axes.flatten(), [total_loss, td_loss, cql_loss, q_mean], ["Total", "TD", "CQL", "Q Mean"]
    ):
        final_val = values[-1]
        initial_val = values[0]
        reduction = (initial_val - final_val) / initial_val * 100 if initial_val != 0 else 0
        ax.text(
            0.98,
            0.98,
            f"Final: {final_val:.4f}\n({reduction:+.1f}% from start)",
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=8,
        )

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"[plot_cql_curves] Generated training curves -> {output_path}")
    print(f"  Epochs: {len(epochs):,}")
    print(
        f"  Total Loss: {total_loss[0]:.4f} -> {total_loss[-1]:.4f} ({(total_loss[0] - total_loss[-1]) / total_loss[0] * 100:.1f}% reduction)"
    )
    print(f"  TD Error: {td_loss[0]:.4f} -> {td_loss[-1]:.4f}")
    print(f"  CQL Penalty: {cql_loss[0]:.4f} -> {cql_loss[-1]:.4f}")


def main():
    args = parse_args()
    plot_training_curves(Path(args.input), Path(args.output), args.dpi)


if __name__ == "__main__":
    main()
