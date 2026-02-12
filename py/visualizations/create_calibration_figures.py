#!/usr/bin/env python3
"""
Generate publication-quality figures for BNN calibration study.

Creates:
1. Coverage comparison bar chart
2. Calibration-sharpness trade-off scatter plot

Figures are saved as PDF for LaTeX inclusion.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set publication-quality style
plt.style.use("seaborn-v0_8-paper")
sns.set_palette("colorblind")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern Roman"]
plt.rcParams["text.usetex"] = False  # Set to True if LaTeX is installed
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.dpi"] = 300

# Output directory
OUT_DIR = Path("/Users/dro/rice/nfl-analytics/analysis/dissertation/figures/out")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def coverage_comparison_bar_chart():
    """Figure 1: Coverage comparison across all methods"""

    # Data
    methods = [
        "BNN Baseline",
        "BNN + Vegas",
        "BNN + Environment",
        "BNN + Opponent",
        "Conformal Prediction",
        "Quantile Regression",
        "Multi-output BNN",
    ]

    coverage = np.array([26.2, 29.7, 29.7, 31.3, 84.5, 89.4, 92.0])
    target = 90.0

    # Color code by performance
    colors = []
    for cov in coverage:
        if cov < 50:
            colors.append("#d62728")  # Red - severely under-calibrated
        elif cov < 80:
            colors.append("#ff7f0e")  # Orange - under-calibrated
        elif cov < 88:
            colors.append("#bcbd22")  # Yellow - approaching target
        else:
            colors.append("#2ca02c")  # Green - well-calibrated

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Horizontal bars
    y_pos = np.arange(len(methods))
    bars = ax.barh(y_pos, coverage, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

    # Target line
    ax.axvline(target, color="red", linestyle="--", linewidth=2, label="Target (90%)", zorder=0)

    # Annotations
    for i, (bar, cov) in enumerate(zip(bars, coverage)):
        ax.text(cov + 2, i, f"{cov:.1f}%", va="center", fontsize=10, fontweight="bold")

    # Labels and formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=10)
    ax.set_xlabel("90% CI Coverage (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Calibration Performance Across UQ Methods", fontsize=12, fontweight="bold", pad=15
    )
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.3, linestyle=":")

    # Legend
    legend_elements = [
        mpatches.Patch(
            facecolor="#d62728", edgecolor="black", label="Severely Under-calibrated (<50%)"
        ),
        mpatches.Patch(facecolor="#ff7f0e", edgecolor="black", label="Under-calibrated (50-80%)"),
        mpatches.Patch(facecolor="#bcbd22", edgecolor="black", label="Approaching Target (80-88%)"),
        mpatches.Patch(facecolor="#2ca02c", edgecolor="black", label="Well-calibrated (≥88%)"),
        plt.Line2D([0], [0], color="red", linestyle="--", linewidth=2, label="Target (90%)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)

    plt.tight_layout()

    # Save
    output_path = OUT_DIR / "coverage_comparison_bar_chart.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")

    # Also save PNG for quick viewing
    plt.savefig(str(output_path).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()


def calibration_sharpness_scatter():
    """Figure 2: Calibration vs. sharpness trade-off"""

    # Data
    methods = [
        "BNN Baseline",
        "BNN + Vegas",
        "BNN + Opponent",
        "Conformal",
        "Quantile Reg",
        "Multi-output BNN",
    ]

    # (CI width, coverage, training time in minutes)
    data = np.array(
        [
            [17.0, 26.2, 25],  # BNN Baseline
            [17.0, 29.7, 25],  # BNN + Vegas
            [17.0, 31.3, 30],  # BNN + Opponent
            [66.0, 84.5, 2],  # Conformal
            [106.0, 89.4, 2],  # Quantile Reg
            [50.0, 92.0, 240],  # Multi-output (estimated width)
        ]
    )

    ci_width = data[:, 0]
    coverage = data[:, 1]
    train_time = data[:, 2]

    # Marker size scaled by training time (log scale for visibility)
    marker_sizes = 100 + 200 * np.log1p(train_time) / np.log1p(train_time.max())

    # Color by method type
    method_colors = [
        "#1f77b4",  # BNN Baseline (blue)
        "#1f77b4",  # BNN + Vegas (blue)
        "#1f77b4",  # BNN + Opponent (blue)
        "#ff7f0e",  # Conformal (orange)
        "#2ca02c",  # Quantile (green)
        "#d62728",  # Multi-output BNN (red)
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot
    for i, (w, c, s, color, method) in enumerate(
        zip(ci_width, coverage, marker_sizes, method_colors, methods)
    ):
        ax.scatter(
            w,
            c,
            s=s,
            color=color,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
            label=method,
            zorder=3,
        )

    # Target coverage line
    ax.axhline(
        90, color="red", linestyle="--", linewidth=2, label="Target Coverage", alpha=0.7, zorder=1
    )

    # Ideal region (shaded)
    ax.axhspan(88, 95, xmin=0, xmax=0.3, alpha=0.1, color="green", zorder=0)
    ax.text(
        20,
        93,
        "Ideal:\nCalibrated\n+ Sharp",
        fontsize=9,
        ha="center",
        style="italic",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    # Annotate points with method names
    for i, (w, c, method) in enumerate(zip(ci_width, coverage, methods)):
        if "Multi-output" in method:
            ax.annotate(
                method,
                (w, c),
                xytext=(10, -10),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", lw=1.5),
            )
        elif "BNN Baseline" in method:
            ax.annotate(
                method, (w, c), xytext=(-60, 5), textcoords="offset points", fontsize=8, alpha=0.8
            )

    # Labels
    ax.set_xlabel("Average 90% CI Width (yards)", fontsize=11, fontweight="bold")
    ax.set_ylabel("90% CI Coverage (%)", fontsize=11, fontweight="bold")
    ax.set_title("Calibration-Sharpness Trade-off", fontsize=12, fontweight="bold", pad=15)

    # Grid
    ax.grid(True, alpha=0.3, linestyle=":")

    # Limits
    ax.set_xlim(0, 120)
    ax.set_ylim(20, 100)

    # Legend (without duplicates)
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicate labels
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="lower right", framealpha=0.9, fontsize=9)

    # Add text box explaining marker size
    ax.text(
        0.02,
        0.98,
        "Marker size ∝ log(training time)",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    # Save
    output_path = OUT_DIR / "calibration_sharpness_scatter.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")

    # Also save PNG
    plt.savefig(str(output_path).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()


def feature_ablation_progression():
    """Figure 3: Feature ablation progression chart"""

    # Data
    stages = [
        "Baseline\n(4 features)",
        "+ Vegas\n(6 features)",
        "+ Environment\n(10 features)",
        "+ Opponent\n(9 features)",
    ]
    coverage = [26.2, 29.7, 29.7, 31.3]
    delta_from_baseline = [0, 3.5, 0, 1.6]  # Incremental improvement

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Line plot
    x = np.arange(len(stages))
    ax.plot(
        x, coverage, marker="o", markersize=10, linewidth=2.5, color="#1f77b4", label="Coverage"
    )

    # Target line
    ax.axhline(90, color="red", linestyle="--", linewidth=2, label="Target (90%)", alpha=0.7)

    # Annotate points
    for i, (cov, delta) in enumerate(zip(coverage, delta_from_baseline)):
        # Coverage value
        ax.text(i, cov + 2, f"{cov:.1f}%", ha="center", fontsize=10, fontweight="bold")
        # Delta from baseline
        if delta > 0:
            ax.text(
                i,
                cov - 3,
                f"+{delta:.1f}pp",
                ha="center",
                fontsize=8,
                style="italic",
                color="green",
            )

    # Labels
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=10)
    ax.set_ylabel("90% CI Coverage (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Feature Ablation Study: Incremental Improvements", fontsize=12, fontweight="bold", pad=15
    )
    ax.set_ylim(20, 100)

    # Grid
    ax.grid(True, alpha=0.3, linestyle=":")

    # Legend
    ax.legend(loc="upper left", framealpha=0.9)

    # Add annotation box
    ax.text(
        0.98,
        0.02,
        "Total improvement: +5.1pp\nFar from 90% target (58.7pp gap)",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="#ffcccc", alpha=0.7),
    )

    plt.tight_layout()

    # Save
    output_path = OUT_DIR / "feature_ablation_progression.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")

    plt.savefig(str(output_path).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Generate all figures"""
    print("Generating BNN calibration study figures...")
    print()

    # Create figures
    coverage_comparison_bar_chart()
    calibration_sharpness_scatter()
    feature_ablation_progression()

    print()
    print("✓ All figures generated successfully!")
    print(f"✓ Output directory: {OUT_DIR}")
    print()
    print("LaTeX inclusion example:")
    print(r"\begin{figure}[t]")
    print(r"  \centering")
    print(r"  \includegraphics[width=\textwidth]{figures/out/coverage_comparison_bar_chart.pdf}")
    print(r"  \caption{90\% CI coverage across all methods...}")
    print(r"  \label{fig:coverage-comparison}")
    print(r"\end{figure}")


if __name__ == "__main__":
    main()
