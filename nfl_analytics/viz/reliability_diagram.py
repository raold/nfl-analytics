#!/usr/bin/env python3
"""
Generate Figure 2.2: Baseline calibration reliability diagram.

Shows probability calibration with 95% binomial confidence intervals.
Perfect calibration follows the diagonal line.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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


def simulate_predictions_and_outcomes(
    n_games: int = 2500, calibration_error: float = 0.02, seed: int = 42
):
    """
    Simulate model predictions and outcomes with realistic calibration.

    Parameters match dissertation: 2500 games (roughly 10 seasons of 250 games each),
    slight miscalibration (~2%) typical for NFL models.
    """
    np.random.seed(seed)

    # Generate predicted probabilities (model forecasts)
    # Concentrated around 0.45-0.55 (close games) with some spread
    predicted_probs = np.concatenate(
        [
            np.random.beta(8, 8, size=int(n_games * 0.6)),  # Close games
            np.random.beta(10, 5, size=int(n_games * 0.2)),  # Favorites
            np.random.beta(5, 10, size=int(n_games * 0.2)),  # Underdogs
        ]
    )

    # Clip to reasonable range
    predicted_probs = np.clip(predicted_probs, 0.05, 0.95)

    # Generate outcomes with slight miscalibration
    # True probability slightly different from predicted
    true_probs = predicted_probs + np.random.normal(0, calibration_error, n_games)
    true_probs = np.clip(true_probs, 0, 1)

    # Generate actual outcomes
    outcomes = (np.random.random(n_games) < true_probs).astype(int)

    return predicted_probs, outcomes


def calculate_calibration_bins(predicted_probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10):
    """
    Calculate calibration statistics by binning predictions.

    Returns:
        bin_centers: Center of each probability bin
        observed_freqs: Observed frequency of positive outcomes in each bin
        counts: Number of predictions in each bin
        lower_ci: Lower 95% confidence interval
        upper_ci: Upper 95% confidence interval
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    observed_freqs = []
    counts = []
    lower_ci = []
    upper_ci = []

    for i in range(n_bins):
        # Find predictions in this bin
        mask = (predicted_probs >= bin_edges[i]) & (predicted_probs < bin_edges[i + 1])

        if i == n_bins - 1:  # Include upper edge in last bin
            mask = (predicted_probs >= bin_edges[i]) & (predicted_probs <= bin_edges[i + 1])

        bin_outcomes = outcomes[mask]
        count = len(bin_outcomes)
        counts.append(count)

        if count > 0:
            freq = bin_outcomes.mean()
            observed_freqs.append(freq)

            # Binomial confidence interval (Wilson score interval)
            # More accurate than normal approximation for proportions
            n = count
            p = freq
            z = 1.96  # 95% CI

            denominator = 1 + z**2 / n
            center = (p + z**2 / (2 * n)) / denominator
            margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator

            lower_ci.append(max(0, center - margin))
            upper_ci.append(min(1, center + margin))
        else:
            observed_freqs.append(np.nan)
            lower_ci.append(np.nan)
            upper_ci.append(np.nan)

    return (
        bin_centers,
        np.array(observed_freqs),
        np.array(counts),
        np.array(lower_ci),
        np.array(upper_ci),
    )


def generate_reliability_diagram(
    n_games: int = 2500, n_bins: int = 10, calibration_error: float = 0.02, seed: int = 42
):
    """Generate the reliability diagram figure with publication-quality styling."""

    # Simulate data
    predicted_probs, outcomes = simulate_predictions_and_outcomes(
        n_games=n_games, calibration_error=calibration_error, seed=seed
    )

    # Calculate calibration
    bin_centers, observed_freqs, counts, lower_ci, upper_ci = calculate_calibration_bins(
        predicted_probs, outcomes, n_bins=n_bins
    )

    # Create figure with academic proportions
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    # Perfect calibration diagonal
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        color=GREY_MED,
        alpha=0.8,
        label="Perfect calibration",
        zorder=1,
    )

    # Observed calibration with error bars
    valid_mask = ~np.isnan(observed_freqs)

    ax.errorbar(
        bin_centers[valid_mask],
        observed_freqs[valid_mask],
        yerr=[
            observed_freqs[valid_mask] - lower_ci[valid_mask],
            upper_ci[valid_mask] - observed_freqs[valid_mask],
        ],
        fmt="o",
        markersize=8,
        color=ACCENT_BLUE,
        ecolor=ACCENT_BLUE,
        elinewidth=1.5,
        capsize=4,
        capthick=1.5,
        label="Observed frequency (95% CI)",
        zorder=3,
    )

    # Connect points
    ax.plot(
        bin_centers[valid_mask],
        observed_freqs[valid_mask],
        "-",
        color=ACCENT_BLUE,
        linewidth=1.2,
        alpha=0.7,
        zorder=2,
    )

    # Add subtle histogram showing distribution of predictions (secondary axis)
    ax2 = ax.twinx()
    ax2.hist(
        predicted_probs,
        bins=n_bins,
        alpha=0.12,
        color=GREY_MED,
        edgecolor=GREY_LIGHT,
        linewidth=0.5,
    )
    ax2.set_ylabel("Frequency", fontsize=9, color=GREY_MED)
    ax2.tick_params(axis="y", labelcolor=GREY_MED, labelsize=8)
    ax2.grid(False)
    ax2.spines["right"].set_color(GREY_LIGHT)

    # Calculate calibration metrics
    brier_score = np.mean((predicted_probs - outcomes) ** 2)

    # Expected Calibration Error (ECE)
    ece = np.nansum(
        counts[valid_mask] * np.abs(bin_centers[valid_mask] - observed_freqs[valid_mask])
    ) / np.sum(counts[valid_mask])

    # Add metrics as clean text (no box, serif font)
    metrics_text = f"Brier = {brier_score:.4f}\n" f"ECE = {ece:.4f}\n" f"$n$ = {n_games:,}"

    ax.text(
        0.04,
        0.96,
        metrics_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        fontsize=9,
        color=GREY_DARK,
        linespacing=1.4,
    )

    # Labels (academic style, no bold)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4, linestyle="-", linewidth=0.5)

    # Legend (clean, minimal)
    ax.legend(
        loc="lower right",
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

    return fig, brier_score, ece


def main():
    """Generate and save the figure."""

    print("Generating Figure 2.2: Baseline calibration diagram...")

    # Generate figure with dissertation-realistic parameters
    fig, brier, ece = generate_reliability_diagram(
        n_games=2500,  # ~10 seasons
        n_bins=10,
        calibration_error=0.02,  # Slight miscalibration typical for NFL
        seed=42,
    )

    # Save to figures directory
    output_path = (
        Path(__file__).parent.parent.parent
        / "analysis"
        / "dissertation"
        / "figures"
        / "reliability_diagram.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved to {output_path}")

    # Print metrics
    print("\nCalibration Metrics:")
    print(f"  Brier Score: {brier:.4f}")
    print(f"  Expected Calibration Error (ECE): {ece:.4f}")

    plt.close()
    print("\n✓ Figure 2.2 complete")


if __name__ == "__main__":
    main()
