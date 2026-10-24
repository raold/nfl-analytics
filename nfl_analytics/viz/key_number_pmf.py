#!/usr/bin/env python3
"""
Generate Figure 2.3: Integer-Margin PMF comparison.

Shows baseline (Skellam/discretized Gaussian) vs key-number reweighted
distribution matching empirical masses at 3, 6, 7, and 10.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# =============================================================================
# DISSERTATION COLOR PALETTE (synchronized with main.tex / tikz_diagram_style.tex)
# =============================================================================
ACCENT_BLUE = "#1f77b4"  # RGB(31,119,180) - Primary
ACCENT_ORANGE = "#ff7f0e"  # RGB(255,127,14) - Secondary/contrast
ACCENT_GREEN = "#2ca02c"  # RGB(44,160,44) - Success
ACCENT_RED = "#d62728"  # RGB(214,39,40) - Warning
GREY_DARK = "#4d4d4d"  # grey3 = 0.3 - Axes, labels
GREY_MED = "#808080"  # grey5 = 0.5 - Secondary text
GREY_LIGHT = "#b3b3b3"  # grey7 = 0.7 - Gridlines

# Set publication-quality style (academic/serif)
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "axes.edgecolor": GREY_DARK,
        "axes.labelcolor": GREY_DARK,
        "xtick.color": GREY_MED,
        "ytick.color": GREY_MED,
        "grid.color": GREY_LIGHT,
        "grid.linewidth": 0.5,
    }
)


def discretized_gaussian_pmf(margins: np.ndarray, mu: float = 0, sigma: float = 13.5):
    """
    Discretized Gaussian PMF for integer margins.

    Uses continuity correction: P(M = m) ≈ Φ((m+0.5-μ)/σ) - Φ((m-0.5-μ)/σ)
    """
    pmf = norm.cdf((margins + 0.5 - mu) / sigma) - norm.cdf((margins - 0.5 - mu) / sigma)
    return pmf / pmf.sum()  # Normalize


def apply_key_number_reweighting(
    base_pmf: np.ndarray,
    margins: np.ndarray,
    key_numbers: list[int],
    target_masses: dict[int, float],
):
    """
    Apply IPF-style reweighting to match key number masses.

    Multiplies probabilities at key margins by target/current, then renormalizes.
    """
    reweighted = base_pmf.copy()

    for key, target in target_masses.items():
        idx = np.where(margins == key)[0]
        if len(idx) > 0 and base_pmf[idx[0]] > 0:
            # Also apply to negative key number (symmetric)
            neg_idx = np.where(margins == -key)[0]

            factor = target / base_pmf[idx[0]]
            reweighted[idx[0]] *= factor
            if len(neg_idx) > 0:
                reweighted[neg_idx[0]] *= factor

    return reweighted / reweighted.sum()  # Renormalize


def generate_pmf_figure():
    """Generate the integer-margin PMF comparison figure."""

    # Margin range (typical NFL score differentials)
    margins = np.arange(-35, 36)

    # Baseline discretized Gaussian (σ ≈ 13.5 for NFL)
    baseline_pmf = discretized_gaussian_pmf(margins, mu=0, sigma=13.5)

    # Key number targets (empirical masses from NFL historical data)
    # These are approximate targets for 3, 6, 7, 10 point margins
    key_numbers = [3, 6, 7, 10]
    target_masses = {
        3: 0.085,  # Touchdowns with missed XP or field goals
        6: 0.055,  # Two field goals
        7: 0.080,  # Touchdown with XP
        10: 0.045,  # TD + FG
    }

    # Reweighted PMF
    reweighted_pmf = apply_key_number_reweighting(baseline_pmf, margins, key_numbers, target_masses)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

    # Bar width for side-by-side bars
    width = 0.4

    # Baseline PMF (blue)
    ax.bar(
        margins - width / 2,
        baseline_pmf,
        width=width,
        color=ACCENT_BLUE,
        alpha=0.7,
        label="Baseline PMF",
        edgecolor="white",
        linewidth=0.3,
    )

    # Reweighted PMF (orange)
    ax.bar(
        margins + width / 2,
        reweighted_pmf,
        width=width,
        color=ACCENT_ORANGE,
        alpha=0.7,
        label="Reweighted PMF (keys)",
        edgecolor="white",
        linewidth=0.3,
    )

    # Highlight key numbers with subtle markers
    for key in key_numbers:
        for m in [key, -key]:
            idx = np.where(margins == m)[0][0]
            ax.axvline(m, color=GREY_LIGHT, linestyle=":", linewidth=0.8, alpha=0.6, zorder=0)

    # Add key number labels at top
    for key in key_numbers:
        ax.annotate(
            f"±{key}",
            xy=(key, max(baseline_pmf.max(), reweighted_pmf.max()) * 1.02),
            ha="center",
            fontsize=8,
            color=GREY_MED,
        )

    # Labels (academic style)
    ax.set_xlabel("Margin (points)")
    ax.set_ylabel("Probability")

    # Limits and grid
    ax.set_xlim(-32, 32)
    ax.set_ylim(0, max(baseline_pmf.max(), reweighted_pmf.max()) * 1.15)
    ax.grid(True, axis="y", alpha=0.4, linestyle="-", linewidth=0.5)

    # Legend
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        edgecolor=GREY_LIGHT,
        fancybox=False,
    )

    # Spine styling
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color(GREY_DARK)

    plt.tight_layout()

    return fig


def main():
    """Generate and save the figure."""

    print("Generating Figure 2.3: Integer-margin PMF comparison...")

    fig = generate_pmf_figure()

    # Save to figures directory
    output_path = (
        Path(__file__).parent.parent.parent
        / "analysis"
        / "dissertation"
        / "figures"
        / "key_number_pmf.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved to {output_path}")

    plt.close()
    print("\n✓ Figure 2.3 complete")


if __name__ == "__main__":
    main()
