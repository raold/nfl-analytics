#!/usr/bin/env python3
"""
Generate Figure 8.2: Fractional Kelly bankroll trajectories.

Shows simulated bankroll paths under different Kelly fractions with median,
50% and 90% credible envelopes highlighting growth vs drawdown tradeoff.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Set publication-quality style
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

def simulate_trajectory(
    n_bets: int,
    initial_bankroll: float,
    kelly_fraction: float,
    base_edge: float,
    win_prob: float,
    seed: int
):
    """Simulate a single bankroll trajectory."""

    np.random.seed(seed)
    bankroll = initial_bankroll
    trajectory = [bankroll]

    for _ in range(n_bets):
        # Model uncertainty in edge
        edge = np.random.normal(base_edge, 0.01)

        # Kelly stake
        if edge > 0:
            stake = kelly_fraction * edge * bankroll
            stake = min(stake, 0.05 * bankroll)  # 5% cap
        else:
            stake = 0

        # Outcome at -110 odds
        if np.random.random() < win_prob:
            bankroll += stake * (100/110)
        else:
            bankroll -= stake

        trajectory.append(bankroll)

    return np.array(trajectory)

def generate_trajectories(
    kelly_fractions=[0.1, 0.25, 0.5, 1.0],
    n_sims: int = 500,
    n_bets: int = 250,
    initial_bankroll: float = 10000,
    base_edge: float = 0.0149,
    win_prob: float = 0.51,
    seed: int = 42
):
    """Generate bankroll trajectories for multiple Kelly fractions."""

    np.random.seed(seed)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    axes = axes.flatten()

    colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson']

    for idx, (kelly_frac, ax) in enumerate(zip(kelly_fractions, axes)):

        # Generate trajectories
        trajectories = []
        for sim_idx in range(n_sims):
            traj = simulate_trajectory(
                n_bets=n_bets,
                initial_bankroll=initial_bankroll,
                kelly_fraction=kelly_frac,
                base_edge=base_edge,
                win_prob=win_prob,
                seed=seed + sim_idx
            )
            trajectories.append(traj)

        trajectories = np.array(trajectories)  # Shape: (n_sims, n_bets+1)

        # Calculate percentiles
        median = np.median(trajectories, axis=0)
        p25 = np.percentile(trajectories, 25, axis=0)
        p75 = np.percentile(trajectories, 75, axis=0)
        p5 = np.percentile(trajectories, 5, axis=0)
        p95 = np.percentile(trajectories, 95, axis=0)

        x = np.arange(n_bets + 1)

        # Plot envelopes
        ax.fill_between(x, p5, p95, alpha=0.2, color=colors[idx], label='90% CI')
        ax.fill_between(x, p25, p75, alpha=0.3, color=colors[idx], label='50% CI')

        # Plot median
        ax.plot(x, median, color=colors[idx], linewidth=2.5, label='Median')

        # Initial bankroll line
        ax.axhline(
            initial_bankroll,
            color='black',
            linestyle='--',
            linewidth=1,
            alpha=0.6,
            label='Initial'
        )

        # Final stats
        final_median = median[-1]
        final_mean = np.mean(trajectories[:, -1])
        max_dd = np.min(median / initial_bankroll - 1) * 100

        # Title with stats
        ax.set_title(
            f'Kelly Fraction = {kelly_frac}\\n'
            f'Final Median: ${final_median:,.0f} | Max DD: {max_dd:.1f}%',
            fontsize=11,
            fontweight='bold',
            pad=10
        )

        ax.set_xlabel('Bet Number', fontsize=10)
        ax.set_ylabel('Bankroll ($)', fontsize=10)
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

    # Main title
    fig.suptitle(
        'Fractional Kelly Bankroll Trajectories\\n'
        f'({n_sims} simulations, Win Rate = {win_prob:.1%}, Edge = {base_edge:.2%})',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.985])

    return fig

def main():
    """Generate and save the figure."""

    print("Generating Figure 8.2: Bankroll trajectories...")

    # Generate figure
    fig = generate_trajectories(
        kelly_fractions=[0.1, 0.25, 0.5, 1.0],
        n_sims=500,
        n_bets=250,
        initial_bankroll=10000,
        base_edge=0.0149,  # CLV = +14.9 bps
        win_prob=0.51,  # 51% win rate from dissertation
        seed=42
    )

    # Save to figures directory
    output_path = Path(__file__).parent.parent.parent / 'analysis' / 'dissertation' / 'figures' / 'bankroll_trajectories.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved to {output_path}")

    plt.close()
    print("\\n✓ Figure 8.2 complete")

if __name__ == '__main__':
    main()
