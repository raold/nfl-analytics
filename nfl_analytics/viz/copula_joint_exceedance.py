#!/usr/bin/env python3
"""
Generate Figure 2.1: Gaussian copula joint exceedance.

Shows the probability of joint exceedance P(Z1>0, Z2>0) as a function of
correlation rho, comparing empirical simulation to analytic formula:
P = 1/4 + (1/2π) * arcsin(rho)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# DISSERTATION COLOR PALETTE (synchronized with main.tex / tikz_diagram_style.tex)
# =============================================================================
ACCENT_BLUE = "#1f77b4"     # RGB(31,119,180) - Primary
ACCENT_ORANGE = "#ff7f0e"   # RGB(255,127,14) - Secondary/contrast
ACCENT_GREEN = "#2ca02c"    # RGB(44,160,44) - Success
ACCENT_RED = "#d62728"      # RGB(214,39,40) - Warning
GREY_DARK = "#4d4d4d"       # grey3 = 0.3 - Axes, labels
GREY_MED = "#808080"        # grey5 = 0.5 - Secondary text
GREY_LIGHT = "#b3b3b3"      # grey7 = 0.7 - Gridlines

# Set publication-quality style (academic/serif)
plt.rcParams.update({
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
})


def analytic_joint_exceedance(rho):
    """
    Analytic formula for P(Z1>0, Z2>0) under Gaussian copula.

    For bivariate standard normal with correlation rho:
    P(Z1>0, Z2>0) = 1/4 + (1/2π) * arcsin(rho)
    """
    return 0.25 + (1 / (2 * np.pi)) * np.arcsin(rho)


def empirical_joint_exceedance(rho, n_samples=100000, seed=42):
    """
    Empirical estimate via simulation.
    """
    np.random.seed(seed)

    # Generate correlated bivariate normal samples
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]

    samples = np.random.multivariate_normal(mean, cov, size=n_samples)

    # Count joint exceedances (both > 0)
    joint_exceedance = np.mean((samples[:, 0] > 0) & (samples[:, 1] > 0))

    return joint_exceedance


def generate_copula_figure(rho_range=None, n_samples=100000, seed=42):
    """Generate the copula joint exceedance figure with publication-quality styling."""

    if rho_range is None:
        rho_range = np.linspace(-0.99, 0.99, 100)

    # Analytic curve
    analytic_probs = analytic_joint_exceedance(rho_range)

    # Empirical points (sample at fewer points for speed)
    rho_empirical = np.linspace(-0.9, 0.9, 20)
    empirical_probs = [
        empirical_joint_exceedance(rho, n_samples=n_samples, seed=seed) for rho in rho_empirical
    ]

    # Create figure with academic proportions
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    # Plot analytic curve (primary blue)
    ax.plot(
        rho_range,
        analytic_probs,
        color=ACCENT_BLUE,
        linewidth=2,
        label=r"Analytic: $P = \frac{1}{4} + \frac{1}{2\pi}\arcsin(\rho)$",
        zorder=2,
    )

    # Plot empirical points (orange for contrast)
    ax.scatter(
        rho_empirical,
        empirical_probs,
        color=ACCENT_ORANGE,
        s=40,
        alpha=0.8,
        edgecolors="white",
        linewidth=0.5,
        label=f"Empirical ({n_samples:,} samples)",
        zorder=3,
    )

    # Reference lines (subtle, informative)
    # Independence (ρ=0)
    ax.axhline(0.25, color=GREY_MED, linestyle="--", linewidth=1, alpha=0.6, zorder=1)
    ax.text(0.92, 0.26, r"$P=0.25$ (indep.)", fontsize=8, color=GREY_MED, ha="right")

    # Perfect positive (ρ=1)
    ax.axhline(0.5, color=GREY_MED, linestyle=":", linewidth=1, alpha=0.6, zorder=1)
    ax.text(0.92, 0.51, r"$P=0.5$ ($\rho=1$)", fontsize=8, color=GREY_MED, ha="right")

    # Perfect negative (ρ=-1)
    ax.axhline(0.0, color=GREY_MED, linestyle=":", linewidth=1, alpha=0.6, zorder=1)
    ax.text(-0.92, 0.01, r"$P=0$ ($\rho=-1$)", fontsize=8, color=GREY_MED, ha="left")

    # Mark key points with minimal annotations (no tooltip boxes)
    special_rhos = [0.3]  # Just the NFL-relevant point
    for rho in special_rhos:
        prob = analytic_joint_exceedance(rho)
        ax.plot(rho, prob, "o", color=ACCENT_RED, markersize=8, zorder=4)
        ax.annotate(
            rf"$\rho={rho}$" + "\n" + f"$P={prob:.3f}$",
            xy=(rho, prob),
            xytext=(15, -20),
            textcoords="offset points",
            fontsize=9,
            color=GREY_DARK,
            arrowprops=dict(arrowstyle="-", color=GREY_MED, lw=0.8),
        )

    # Labels (no bold, academic style)
    ax.set_xlabel(r"Correlation ($\rho$)")
    ax.set_ylabel(r"$\mathbb{P}(Z_1 > 0, Z_2 > 0)$")

    # Grid and limits
    ax.grid(True, alpha=0.4, linestyle="-", linewidth=0.5)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.02, 0.55)

    # Legend (clean, outside interference)
    ax.legend(
        loc="upper left",
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

    print("Generating Figure 2.1: Copula joint exceedance...")

    # Generate figure
    fig = generate_copula_figure(rho_range=np.linspace(-0.99, 0.99, 100), n_samples=100000, seed=42)

    # Save to figures directory
    output_path = (
        Path(__file__).parent.parent.parent
        / "analysis"
        / "dissertation"
        / "figures"
        / "joint_exceedance_vs_rho.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved to {output_path}")

    # Verification: print some analytic values
    print("\nCalibration check:")
    test_rhos = [-0.9, -0.5, 0, 0.5, 0.9]
    for rho in test_rhos:
        analytic = analytic_joint_exceedance(rho)
        empirical = empirical_joint_exceedance(rho, n_samples=100000)
        print(
            f"  rho={rho:5.2f}: Analytic={analytic:.4f}, Empirical={empirical:.4f}, Diff={abs(analytic-empirical):.4f}"
        )

    plt.close()
    print("\n✓ Figure 2.1 complete")


if __name__ == "__main__":
    main()
