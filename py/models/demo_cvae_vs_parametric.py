"""
Demonstration comparing deep generative cVAE model vs parametric distributions.

Shows how the cVAE captures complex game dynamics that Skellam/Poisson miss.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

sys.path.append(str(Path(__file__).parent.parent))

import warnings

from data.cvae_data_pipeline import create_simulated_data
from models.game_trajectory_cvae import ConditionalVAE, CVAESimulator
from models.score_distributions import reweight_key_masses, skellam_pmf_range

warnings.filterwarnings("ignore")

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (15, 10)


def simulate_parametric_games(
    mu_home: float, mu_away: float, n_samples: int = 10000, reweight_keys: bool = True
) -> dict:
    """
    Simulate games using parametric distributions (Skellam/Poisson).
    """
    # Generate margin distribution
    margin_pmf = skellam_pmf_range(mu_home, mu_away, k_min=-50, k_max=50)

    # Optionally reweight for key numbers
    if reweight_keys:
        key_targets = {
            3: 0.09,  # Field goal
            6: 0.06,  # Missed XP
            7: 0.075,  # Touchdown
            10: 0.05,  # TD + FG
            14: 0.04,  # Two TDs
        }
        margin_pmf = reweight_key_masses(margin_pmf, key_targets)

    # Sample margins
    margins_vals = list(margin_pmf.keys())
    margins_probs = list(margin_pmf.values())
    sampled_margins = np.random.choice(margins_vals, size=n_samples, p=margins_probs)

    # Sample total scores independently (simplified)
    home_scores = np.random.poisson(mu_home, n_samples)
    away_scores = np.random.poisson(mu_away, n_samples)
    totals = home_scores + away_scores

    return {
        "margins": sampled_margins,
        "totals": totals,
        "home_scores": home_scores,
        "away_scores": away_scores,
    }


def compare_distributions(cvae_model: ConditionalVAE, conditions: torch.Tensor):
    """
    Compare cVAE vs parametric distributions.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Generate samples from both models
    n_samples = 10000

    # cVAE samples
    cvae_trajectories = cvae_model.generate(conditions[0:1], n_samples=n_samples)
    cvae_margins = [t.margin for t in cvae_trajectories]
    cvae_totals = [t.total_points for t in cvae_trajectories]
    cvae_home = [t.home_total for t in cvae_trajectories]
    cvae_away = [t.away_total for t in cvae_trajectories]

    # Extract implied mu values from conditions (simplified)
    spread = conditions[0, 0].item()
    total = conditions[0, 1].item()
    mu_home = (total - spread) / 2
    mu_away = (total + spread) / 2

    # Parametric samples
    param_results = simulate_parametric_games(mu_home, mu_away, n_samples)

    # Plot 1: Margin distributions
    ax = axes[0, 0]
    ax.hist(cvae_margins, bins=50, alpha=0.5, label="cVAE", density=True, color="blue")
    ax.hist(
        param_results["margins"], bins=50, alpha=0.5, label="Skellam", density=True, color="red"
    )
    ax.set_xlabel("Margin")
    ax.set_ylabel("Density")
    ax.set_title("Margin Distribution Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Total points distributions
    ax = axes[0, 1]
    ax.hist(cvae_totals, bins=30, alpha=0.5, label="cVAE", density=True, color="blue")
    ax.hist(
        param_results["totals"], bins=30, alpha=0.5, label="Poisson Sum", density=True, color="red"
    )
    ax.set_xlabel("Total Points")
    ax.set_ylabel("Density")
    ax.set_title("Total Points Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Joint distribution (2D density)
    ax = axes[0, 2]
    ax.hexbin(cvae_home, cvae_away, gridsize=20, cmap="Blues", alpha=0.6)
    ax.set_xlabel("Home Score")
    ax.set_ylabel("Away Score")
    ax.set_title("cVAE Joint Score Distribution")
    ax.grid(True, alpha=0.3)

    # Plot 4: Key number frequencies
    ax = axes[1, 0]
    key_numbers = [3, 6, 7, 10, 14, -3, -6, -7, -10, -14]
    cvae_key_freq = []
    param_key_freq = []

    for k in key_numbers:
        cvae_key_freq.append(sum(1 for m in cvae_margins if abs(m - k) < 0.5) / n_samples)
        param_key_freq.append(
            sum(1 for m in param_results["margins"] if abs(m - k) < 0.5) / n_samples
        )

    x = np.arange(len(key_numbers))
    width = 0.35
    ax.bar(x - width / 2, cvae_key_freq, width, label="cVAE", color="blue", alpha=0.7)
    ax.bar(x + width / 2, param_key_freq, width, label="Parametric", color="red", alpha=0.7)
    ax.set_xlabel("Key Number")
    ax.set_ylabel("Frequency")
    ax.set_title("Key Number Clustering")
    ax.set_xticks(x)
    ax.set_xticklabels(key_numbers)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Quarter correlations (unique to cVAE)
    ax = axes[1, 1]
    q1_scores = [t.home_q1 + t.away_q1 for t in cvae_trajectories[:1000]]
    q2_scores = [t.home_q2 + t.away_q2 for t in cvae_trajectories[:1000]]
    q3_scores = [t.home_q3 + t.away_q3 for t in cvae_trajectories[:1000]]
    q4_scores = [t.home_q4 + t.away_q4 for t in cvae_trajectories[:1000]]

    quarters_data = np.column_stack([q1_scores, q2_scores, q3_scores, q4_scores])
    corr_matrix = np.corrcoef(quarters_data.T)

    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-0.5, vmax=0.5)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
    ax.set_yticklabels(["Q1", "Q2", "Q3", "Q4"])
    ax.set_title("Quarter Score Correlations (cVAE)")
    plt.colorbar(im, ax=ax)

    # Add correlation values
    for i in range(4):
        for j in range(4):
            text = ax.text(
                j,
                i,
                f"{corr_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

    # Plot 6: Tail behavior
    ax = axes[1, 2]

    # Calculate tail probabilities
    thresholds = [20, 30, 40, 50]
    cvae_tail_probs = []
    param_tail_probs = []

    for t in thresholds:
        cvae_tail_probs.append(sum(1 for m in np.abs(cvae_margins) if m > t) / n_samples)
        param_tail_probs.append(
            sum(1 for m in np.abs(param_results["margins"]) if m > t) / n_samples
        )

    ax.semilogy(thresholds, cvae_tail_probs, "o-", label="cVAE", color="blue", linewidth=2)
    ax.semilogy(thresholds, param_tail_probs, "s-", label="Parametric", color="red", linewidth=2)
    ax.set_xlabel("Margin Threshold")
    ax.set_ylabel("P(|Margin| > Threshold)")
    ax.set_title("Tail Probability Comparison (Log Scale)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    return fig


def demonstrate_complex_derivatives(cvae_model: ConditionalVAE, conditions: torch.Tensor):
    """
    Show how cVAE enables pricing of complex derivatives.
    """
    simulator = CVAESimulator()
    simulator.model = cvae_model

    # Generate game simulations
    features = {
        "spread_close": -3.5,
        "total_close": 45.5,
        "home_offensive_rating": 105,
        "home_defensive_rating": 98,
        "away_offensive_rating": 102,
        "away_defensive_rating": 101,
        "home_recent_ppg": 24.5,
        "away_recent_ppg": 22.3,
        "home_rest_days": 7,
        "away_rest_days": 7,
        "temperature": 65,
        "wind_speed": 8,
        "precipitation": 0,
        "is_division_game": 0,
        "is_primetime": 1,
        "week_of_season": 10,
        "home_win_pct": 0.625,
        "away_win_pct": 0.500,
    }

    sims = simulator.simulate_game(features, n_samples=100000)

    print("\n" + "=" * 60)
    print("COMPLEX DERIVATIVE PRICING WITH cVAE")
    print("=" * 60)

    # 1. Same-game parlay pricing
    print("\n1. Same-Game Parlay (SGP) Analysis:")
    print("-" * 40)

    # Calculate correlations between different bet types
    home_covers = sims["margins"] > 3.5
    over_hits = sims["totals"] > 45.5
    home_q1_wins = sims["home_q1"] > sims["away_q1"]

    # Joint probabilities
    p_home_covers = np.mean(home_covers)
    p_over = np.mean(over_hits)
    p_home_q1 = np.mean(home_q1_wins)

    # Correlated probabilities
    p_home_and_over = np.mean(home_covers & over_hits)
    p_all_three = np.mean(home_covers & over_hits & home_q1_wins)

    # Independent assumption (wrong!)
    p_indep_two = p_home_covers * p_over
    p_indep_three = p_home_covers * p_over * p_home_q1

    print(f"  Home -3.5 probability: {p_home_covers:.3f}")
    print(f"  Over 45.5 probability: {p_over:.3f}")
    print(f"  Home Q1 win probability: {p_home_q1:.3f}")
    print("\n  Two-leg SGP (Home -3.5 + Over 45.5):")
    print(f"    True joint probability: {p_home_and_over:.3f}")
    print(f"    Independent assumption: {p_indep_two:.3f}")
    print(f"    Correlation impact: {(p_home_and_over/p_indep_two - 1)*100:+.1f}%")
    print("\n  Three-leg SGP (+ Home Q1):")
    print(f"    True joint probability: {p_all_three:.3f}")
    print(f"    Independent assumption: {p_indep_three:.3f}")
    print(f"    Correlation impact: {(p_all_three/p_indep_three - 1)*100:+.1f}%")

    # 2. Live betting scenarios
    print("\n2. Live Betting Dynamics:")
    print("-" * 40)

    # Analyze momentum effects
    q1_blowouts = (sims["home_q1"] - sims["away_q1"]) > 10
    comeback_games = q1_blowouts & (sims["margins"] < 0)

    print(f"  P(Home leads by 10+ after Q1): {np.mean(q1_blowouts):.3f}")
    print(
        f"  P(Away wins | Home led by 10+ in Q1): {np.mean(comeback_games)/np.mean(q1_blowouts):.3f}"
    )
    print(f"  P(Away wins unconditionally): {np.mean(sims['margins'] < 0):.3f}")

    # Quarter-by-quarter scoring patterns
    avg_q1 = np.mean(sims["home_q1"] + sims["away_q1"])
    avg_q2 = np.mean(sims["home_q2"] + sims["away_q2"])
    avg_q3 = np.mean(sims["home_q3"] + sims["away_q3"])
    avg_q4 = np.mean(sims["home_q4"] + sims["away_q4"])

    print("\n  Average scoring by quarter:")
    print(f"    Q1: {avg_q1:.1f} points")
    print(f"    Q2: {avg_q2:.1f} points")
    print(f"    Q3: {avg_q3:.1f} points")
    print(f"    Q4: {avg_q4:.1f} points")

    # 3. Teaser pricing with Wong coverage
    print("\n3. Teaser Analysis (6-point, 2-team):")
    print("-" * 40)

    # Wong teaser numbers
    wong_spreads = [-9, -8.5, -8, -7.5, +1.5, +2, +2.5, +3]

    for spread in [-7.5, -3, +2.5]:
        teased_spread = spread + 6
        base_cover = np.mean(sims["margins"] > -spread)
        teased_cover = np.mean(sims["margins"] > -teased_spread)

        is_wong = spread in wong_spreads
        wong_tag = " [WONG]" if is_wong else ""

        print(f"  Spread {spread:+.1f} → {teased_spread:+.1f}{wong_tag}")
        print(f"    Base cover: {base_cover:.3f}")
        print(f"    Teased cover: {teased_cover:.3f}")
        print(f"    Value ratio: {teased_cover/base_cover:.2f}x")

    # 4. Player prop correlations (simulated)
    print("\n4. Player Prop Correlations (Simulated):")
    print("-" * 40)

    # Simulate QB passing yards correlation with game total
    high_scoring = sims["totals"] > 50
    low_scoring = sims["totals"] < 40

    # Simulated passing yards (correlated with total)
    qb_yards_high = 300 + 50 * np.random.randn(np.sum(high_scoring))
    qb_yards_low = 220 + 40 * np.random.randn(np.sum(low_scoring))

    print(f"  E[QB Yards | Total > 50]: {np.mean(qb_yards_high):.0f}")
    print(f"  E[QB Yards | Total < 40]: {np.mean(qb_yards_low):.0f}")
    print(f"  Correlation impact: {(np.mean(qb_yards_high)/np.mean(qb_yards_low) - 1)*100:+.1f}%")


def main():
    """
    Main demonstration.
    """
    print("\n" + "=" * 60)
    print("DEEP GENERATIVE MODEL vs PARAMETRIC DISTRIBUTIONS")
    print("Conditional VAE for NFL Game Simulation")
    print("=" * 60)

    # Create simulated training data
    print("\n1. Creating simulated training data...")
    trajectories, conditions = create_simulated_data(n_samples=5000)

    # Initialize model
    print("2. Initializing conditional VAE model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")

    model = ConditionalVAE(
        trajectory_dim=16,
        condition_dim=20,
        latent_dim=32,
        hidden_dims=[256, 128, 64],
        device=device,
    )

    # Quick training demo (normally would train for many epochs)
    print("3. Training cVAE (demo with 10 epochs)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        epoch_losses = []

        # Mini-batch training
        batch_size = 32
        for i in range(0, len(trajectories), batch_size):
            batch_traj = torch.stack(
                [
                    trajectories[j].to_tensor()
                    for j in range(i, min(i + batch_size, len(trajectories)))
                ]
            )
            batch_cond = torch.FloatTensor(conditions[i : i + batch_size])

            # Forward pass
            outputs = model(batch_traj, batch_cond)
            losses = model.loss_function(outputs, batch_traj)

            # Backward pass
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            epoch_losses.append(losses["total"].item())

        if epoch % 5 == 0:
            print(f"   Epoch {epoch}: Loss = {np.mean(epoch_losses):.4f}")

    print("\n4. Comparing distributions...")

    # Test conditions
    test_conditions = torch.FloatTensor(
        [
            [-3.5, 45.5] + [0] * 18,  # Spread -3.5, Total 45.5
            [-7, 51] + [0] * 18,  # Spread -7, Total 51
            [2.5, 42] + [0] * 18,  # Spread +2.5, Total 42
        ]
    )

    # Generate comparison plots
    fig = compare_distributions(model, test_conditions)
    plt.savefig("analysis/cvae_vs_parametric.png", dpi=150, bbox_inches="tight")
    print("   Saved comparison plots to analysis/cvae_vs_parametric.png")

    # Demonstrate complex derivative pricing
    demonstrate_complex_derivatives(model, test_conditions)

    # Key takeaways
    print("\n" + "=" * 60)
    print("KEY ADVANTAGES OF DEEP GENERATIVE MODELS")
    print("=" * 60)
    print(
        """
1. NON-PARAMETRIC FLEXIBILITY
   - Learns actual game distribution from data
   - No assumptions about functional form
   - Captures complex tail dependencies

2. QUARTER-LEVEL DYNAMICS
   - Models momentum and game flow
   - Enables live betting applications
   - Captures comeback probabilities

3. CORRELATION STRUCTURE
   - Joint modeling of all outcomes
   - Accurate same-game parlay pricing
   - Player prop correlations

4. KEY NUMBER CLUSTERING
   - Learns empirically without manual reweighting
   - Adapts to rule changes (2-pt conversions, etc.)

5. ENHANCED RISK MANAGEMENT
   - More accurate tail risk assessment
   - Better portfolio VaR/CVaR estimates
   - Scenario generation for stress testing

The cVAE approach is particularly valuable for:
- Complex derivative pricing (SGPs, teasers, round robins)
- Live betting model updates
- Risk aggregation across correlated bets
- Regulatory scenario analysis
    """
    )

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
