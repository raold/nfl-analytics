#!/usr/bin/env python3
"""
Generate BNN Calibration Improvement Figures for Dissertation

Creates visualization figures showing BNN calibration improvement progression:
1. phase_improvements_breakdown: Bar chart showing coverage improvement across phases
2. improvement_hierarchy: Stacked/grouped visualization of improvement sources

Referenced in: Chapter 8, section_bnn_calibration_study.tex

Output: analysis/dissertation/figures/out/

Usage:
  python scripts/generate_bnn_improvement_figures.py

Author: Claude Code (Sonnet 4.5)
Date: 2025-11-03
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Dissertation styling
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "baseline": "#E63946",  # Red
    "features": "#457B9D",  # Blue
    "architecture": "#2A9D8F",  # Teal
    "target": "#F4A261",  # Orange
}


def generate_phase_improvements_breakdown():
    """Generate bar chart showing coverage improvements across phases"""

    # Data from BNN calibration study
    phases = [
        "Baseline\n(4 features)",
        "Phase 1:\n+Vegas Features",
        "Phase 1:\n+Opponent",
        "Phase 2:\n+Prior Tuning",
        "Phase 3:\nMulti-Output\nArchitecture",
    ]
    coverage = [26.2, 29.7, 31.3, 31.8, 92.0]  # 90% coverage percentages
    target = 90.0

    # Incremental improvements
    improvements = [coverage[0]] + [coverage[i] - coverage[i - 1] for i in range(1, len(coverage))]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar positions
    x = np.arange(len(phases))
    width = 0.7

    # Colors for each phase
    colors = [
        COLORS["baseline"],
        COLORS["features"],
        COLORS["features"],
        COLORS["features"],
        COLORS["architecture"],
    ]

    # Create bars
    bars = ax.bar(x, coverage, width, color=colors, alpha=0.85, edgecolor="black", linewidth=1.5)

    # Add target line
    ax.axhline(
        y=target,
        color=COLORS["target"],
        linestyle="--",
        linewidth=2.5,
        label=f"Target: {target}%",
        alpha=0.9,
    )

    # Add value labels on bars
    for i, (bar, cov) in enumerate(zip(bars, coverage)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1.5,
            f"{cov:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

        # Add improvement delta for non-baseline
        if i > 0:
            delta = improvements[i]
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height / 2,
                f"+{delta:.1f}pp",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
            )

    # Formatting
    ax.set_xlabel("Calibration Improvement Phase", fontsize=13, fontweight="bold")
    ax.set_ylabel("90% Coverage (%)", fontsize=13, fontweight="bold")
    ax.set_title(
        "BNN Calibration Improvement Progression\nCoverage Gains Across Enhancement Phases",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, axis="y")

    # Add annotation
    ax.annotate(
        "Architectural\nBreakthrough\n(+60.2pp)",
        xy=(4, 92),
        xytext=(3.5, 75),
        arrowprops=dict(arrowstyle="->", lw=2, color="black"),
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
    )

    plt.tight_layout()
    output_path = Path("analysis/dissertation/figures/out/phase_improvements_breakdown.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def generate_improvement_hierarchy():
    """Generate stacked visualization showing improvement source hierarchy"""

    # Improvement sources
    categories = ["Feature\nEngineering", "Prior\nTuning", "Architectural\nInnovation"]
    max_improvements = [5.1, 0.5, 60.2]  # percentage points
    final_coverage = [31.3, 31.8, 92.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: Improvement magnitudes
    x = np.arange(len(categories))
    bars = ax1.barh(
        x,
        max_improvements,
        color=[COLORS["features"], COLORS["features"], COLORS["architecture"]],
        alpha=0.85,
        edgecolor="black",
        linewidth=1.5,
    )

    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars, max_improvements)):
        width = bar.get_width()
        ax1.text(
            width + 1,
            bar.get_y() + bar.get_height() / 2,
            f"+{imp:.1f}pp",
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    ax1.set_yticks(x)
    ax1.set_yticklabels(categories, fontsize=11)
    ax1.set_xlabel("Coverage Improvement (percentage points)", fontsize=12, fontweight="bold")
    ax1.set_title("Maximum Impact by\nImprovement Category", fontsize=13, fontweight="bold")
    ax1.set_xlim(0, 70)
    ax1.grid(True, alpha=0.3, axis="x")

    # Right plot: Cumulative effect
    baseline = 26.2
    cumulative = [
        baseline,
        baseline + max_improvements[0],
        baseline + max_improvements[0] + max_improvements[1],
        baseline + sum(max_improvements),
    ]
    stages = ["Baseline", "+ Features", "+ Prior", "+ Architecture"]

    ax2.plot(
        range(len(stages)),
        cumulative,
        "o-",
        linewidth=3,
        markersize=10,
        color=COLORS["architecture"],
        markerfacecolor="white",
        markeredgewidth=2.5,
    )

    # Target line
    ax2.axhline(
        y=90, color=COLORS["target"], linestyle="--", linewidth=2.5, label="Target (90%)", alpha=0.9
    )

    # Add value labels
    for i, (stage, cov) in enumerate(zip(stages, cumulative)):
        ax2.text(
            i, cov + 3, f"{cov:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    ax2.set_xticks(range(len(stages)))
    ax2.set_xticklabels(stages, fontsize=10, rotation=15, ha="right")
    ax2.set_ylabel("90% Coverage (%)", fontsize=12, fontweight="bold")
    ax2.set_title("Cumulative Calibration\nImprovement", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path("analysis/dissertation/figures/out/improvement_hierarchy.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def main():
    print("Generating BNN calibration improvement figures...")

    # Generate both figures
    fig1 = generate_phase_improvements_breakdown()
    print(f"✅ Generated: {fig1}")

    fig2 = generate_improvement_hierarchy()
    print(f"✅ Generated: {fig2}")

    print(f"\n{'='*70}")
    print("BNN IMPROVEMENT FIGURES GENERATION COMPLETE")
    print(f"{'='*70}")
    print("Output directory: analysis/dissertation/figures/out/")
    print("\nFigures ready for dissertation inclusion:")
    print("  - phase_improvements_breakdown.pdf")
    print("  - improvement_hierarchy.pdf")


if __name__ == "__main__":
    main()
