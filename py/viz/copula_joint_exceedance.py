#!/usr/bin/env python3
"""
Generate Figure 4.1: Gaussian copula joint exceedance.

Shows the probability of joint exceedance P(Z1>0, Z2>0) as a function of
correlation rho, comparing empirical simulation to analytic formula:
P = 1/4 + (1/2π) * arcsin(rho)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality style
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"


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
    """Generate the copula joint exceedance figure."""

    if rho_range is None:
        rho_range = np.linspace(-0.99, 0.99, 100)

    # Analytic curve
    analytic_probs = analytic_joint_exceedance(rho_range)

    # Empirical points (sample at fewer points for speed)
    rho_empirical = np.linspace(-0.9, 0.9, 20)
    empirical_probs = [
        empirical_joint_exceedance(rho, n_samples=n_samples, seed=seed) for rho in rho_empirical
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    # Plot analytic curve
    ax.plot(
        rho_range,
        analytic_probs,
        "b-",
        linewidth=3,
        label=r"Analytic: $P = \frac{1}{4} + \frac{1}{2\pi}\arcsin(\rho)$",
        zorder=2,
    )

    # Plot empirical points
    ax.scatter(
        rho_empirical,
        empirical_probs,
        color="red",
        s=60,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
        label=f"Empirical ({n_samples:,} samples)",
        zorder=3,
    )

    # Mark special points
    special_rhos = [-0.5, 0, 0.5]
    special_probs = analytic_joint_exceedance(np.array(special_rhos))

    for rho, prob in zip(special_rhos, special_probs):
        ax.axvline(rho, color="gray", linestyle=":", alpha=0.4, zorder=1)
        ax.axhline(prob, color="gray", linestyle=":", alpha=0.4, zorder=1)

        # Annotate
        ax.annotate(
            f"rho={rho:.1f}\nP={prob:.3f}",
            xy=(rho, prob),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", lw=1),
        )

    # Independence line
    ax.axhline(
        0.25,
        color="green",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=r"Independence ($\rho=0$): $P=0.25$",
    )

    # Perfect positive correlation
    ax.axhline(
        0.5,
        color="orange",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=r"Perfect positive ($\rho=1$): $P=0.5$",
    )

    # Perfect negative correlation
    ax.axhline(
        0.0,
        color="purple",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=r"Perfect negative ($\rho=-1$): $P=0$",
    )

    # Labels and title
    ax.set_xlabel("Correlation (rho)", fontsize=14, fontweight="bold")
    ax.set_ylabel("P(Z1>0, Z2>0)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Gaussian Copula Joint Exceedance\n" "Calibration Check for Dependence Modeling",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Grid and styling
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.05, 0.55)

    # Legend
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95, edgecolor="black")

    # Add context box
    context_text = (
        "Application: NFL spread-total dependence\n"
        "Typical correlation: rho in [0.2, 0.4]\n"
        "Joint exceedance P in [0.28, 0.32]"
    )

    ax.text(
        0.98,
        0.03,
        context_text,
        transform=ax.transAxes,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
        fontsize=9,
        family="monospace",
    )

    plt.tight_layout()

    return fig


def main():
    """Generate and save the figure."""

    print("Generating Figure 4.1: Copula joint exceedance...")

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
    print("\n✓ Figure 4.1 complete")


if __name__ == "__main__":
    main()
