#!/usr/bin/env python3
"""
Generate key-number calibration tables with chi-square tests.

Compares observed margin distribution at key numbers (3, 6, 7, 10, 14)
against fitted discrete distributions with and without IPF reweighting.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import psycopg
from scipy import stats


def get_connection() -> psycopg.Connection:
    """Connect to PostgreSQL database."""
    import os
    return psycopg.connect(
        dbname=os.getenv("POSTGRES_DB", "devdb01"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5544")),
        user=os.getenv("POSTGRES_USER", "dro"),
        password=os.getenv("POSTGRES_PASSWORD", "sicillionbillions"),
    )


def get_margin_distribution(
    conn: psycopg.Connection,
    season_start: int,
    season_end: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Fetch observed margin distribution from database.
    
    Returns:
        margins: Array of margin values
        counts: Array of counts for each margin
    """
    query = """
        SELECT 
            home_score - away_score AS margin,
            COUNT(*) AS count
        FROM games
        WHERE season BETWEEN %s AND %s
            AND home_score IS NOT NULL
            AND away_score IS NOT NULL
        GROUP BY margin
        ORDER BY margin
    """
    
    with conn.cursor() as cur:
        cur.execute(query, (season_start, season_end))
        results = cur.fetchall()
    
    margins = np.array([r[0] for r in results])
    counts = np.array([r[1] for r in results])
    
    return margins, counts


def fit_discrete_normal(margins: np.ndarray, counts: np.ndarray) -> Dict[int, float]:
    """Fit discrete normal distribution to observed margins.
    
    Returns:
        Dictionary mapping margin -> probability
    """
    # Calculate empirical mean and std
    total = counts.sum()
    probs = counts / total
    mean = (margins * probs).sum()
    std = np.sqrt(((margins - mean) ** 2 * probs).sum())
    
    # Fit discrete normal
    fitted_probs = {}
    for m in margins:
        # Use continuity correction: P(X=m) ≈ Φ((m+0.5-μ)/σ) - Φ((m-0.5-μ)/σ)
        p_upper = stats.norm.cdf((m + 0.5 - mean) / std)
        p_lower = stats.norm.cdf((m - 0.5 - mean) / std)
        fitted_probs[m] = p_upper - p_lower
    
    # Normalize
    total_prob = sum(fitted_probs.values())
    fitted_probs = {k: v / total_prob for k, v in fitted_probs.items()}
    
    return fitted_probs


def apply_ipf_reweighting(
    base_probs: Dict[int, float],
    key_numbers: List[int],
    target_masses: Dict[int, float],
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> Dict[int, float]:
    """Apply Iterative Proportional Fitting to match key number masses.
    
    Args:
        base_probs: Base probability distribution
        key_numbers: List of key margins to reweight
        target_masses: Target probabilities for key numbers
        max_iterations: Maximum IPF iterations
        tolerance: Convergence tolerance
    
    Returns:
        Reweighted probability distribution
    """
    reweighted = base_probs.copy()
    
    for iteration in range(max_iterations):
        old_probs = reweighted.copy()
        
        # Update each key number
        for key in key_numbers:
            if key not in target_masses or key not in reweighted:
                continue
            
            current_mass = reweighted[key]
            target_mass = target_masses[key]
            
            if current_mass > 0:
                # Scale this key number
                scale = target_mass / current_mass
                reweighted[key] *= scale
                
                # Compensate by scaling other margins proportionally
                other_total = sum(p for m, p in reweighted.items() if m not in key_numbers)
                if other_total > 0:
                    compensation = (1 - sum(target_masses.values())) / other_total
                    for m in reweighted:
                        if m not in key_numbers:
                            reweighted[m] *= compensation
        
        # Check convergence
        max_change = max(abs(reweighted[m] - old_probs[m]) for m in reweighted)
        if max_change < tolerance:
            break
    
    # Final normalization
    total = sum(reweighted.values())
    reweighted = {k: v / total for k, v in reweighted.items()}
    
    return reweighted


def chi_square_test(
    observed: np.ndarray,
    expected: np.ndarray,
    min_expected: float = 5.0
) -> Tuple[float, float, int]:
    """Perform chi-square goodness-of-fit test.
    
    Args:
        observed: Observed counts
        expected: Expected counts
        min_expected: Minimum expected count for valid test
    
    Returns:
        chi2_stat: Chi-square test statistic
        p_value: P-value
        df: Degrees of freedom
    """
    # Filter out cells with low expected counts
    mask = expected >= min_expected
    obs_filtered = observed[mask]
    exp_filtered = expected[mask]
    
    if len(obs_filtered) < 2:
        return np.nan, np.nan, 0
    
    # Chi-square test
    chi2_stat = ((obs_filtered - exp_filtered) ** 2 / exp_filtered).sum()
    df = len(obs_filtered) - 1  # -1 for normalization constraint
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)
    
    return chi2_stat, p_value, df


def generate_calibration_table(
    margins: np.ndarray,
    counts: np.ndarray,
    key_numbers: List[int],
    output_path: Path
):
    """Generate LaTeX table comparing observed vs fitted at key numbers."""
    
    total = counts.sum()
    observed_probs = counts / total
    margin_to_prob = {m: p for m, p in zip(margins, observed_probs)}
    margin_to_count = {m: c for m, c in zip(margins, counts)}
    
    # Fit base distribution
    base_probs = fit_discrete_normal(margins, counts)
    
    # Calculate empirical key masses for reweighting target
    key_masses = {k: margin_to_prob.get(k, 0) for k in key_numbers}
    
    # Apply IPF reweighting
    reweighted_probs = apply_ipf_reweighting(
        base_probs, key_numbers, key_masses
    )
    
    # Chi-square tests at key numbers only
    key_observed = np.array([margin_to_count.get(k, 0) for k in key_numbers])
    key_base = np.array([base_probs.get(k, 0) * total for k in key_numbers])
    key_reweighted = np.array([reweighted_probs.get(k, 0) * total for k in key_numbers])
    
    chi2_base, p_base, df_base = chi_square_test(key_observed, key_base)
    chi2_reweighted, p_reweighted, df_reweighted = chi_square_test(
        key_observed, key_reweighted
    )
    
    # Generate LaTeX table
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \caption{Key-number calibration: $\chi^2$ goodness-of-fit at key margins.}",
        r"  \begin{tabular}{lcccc}",
        r"    \toprule",
        r"    Margin & Observed & Base Fit & Reweighted & Abs. Error \\",
        r"    \midrule",
    ]
    
    # Add key number rows
    for key in key_numbers:
        obs = margin_to_prob.get(key, 0) * 100
        base = base_probs.get(key, 0) * 100
        rw = reweighted_probs.get(key, 0) * 100
        error = abs(obs - rw)
        
        lines.append(
            f"    {key:+3d} & {obs:5.2f}\\% & {base:5.2f}\\% & "
            f"{rw:5.2f}\\% & {error:5.2f}\\% \\\\"
        )
    
    lines.extend([
        r"    \midrule",
        f"    \\multicolumn{{5}}{{l}}{{Base: $\\chi^2$={chi2_base:.2f}, "
        f"$p$={p_base:.3f}, $df$={df_base}}} \\\\",
        f"    \\multicolumn{{5}}{{l}}{{Reweighted: $\\chi^2$={chi2_reweighted:.2f}, "
        f"$p$={p_reweighted:.3f}, $df$={df_reweighted}}} \\\\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"✓ Generated: {output_path}")
    
    return {
        "chi2_base": chi2_base,
        "p_base": p_base,
        "df_base": df_base,
        "chi2_reweighted": chi2_reweighted,
        "p_reweighted": p_reweighted,
        "df_reweighted": df_reweighted,
    }


def generate_reweighting_ablation_table(
    margins: np.ndarray,
    counts: np.ndarray,
    key_numbers: List[int],
    output_path: Path
):
    """Generate ablation table: with/without key-mass reweighting impact."""
    
    total = counts.sum()
    observed_probs = counts / total
    margin_to_count = {m: c for m, c in zip(margins, counts)}
    
    # Base fit
    base_probs = fit_discrete_normal(margins, counts)
    
    # Reweighted fit
    key_masses = {
        k: observed_probs[np.where(margins == k)[0][0]]
        if k in margins else 0
        for k in key_numbers
    }
    reweighted_probs = apply_ipf_reweighting(base_probs, key_numbers, key_masses)
    
    # Full distribution chi-square tests
    expected_base = np.array([base_probs.get(m, 0) * total for m in margins])
    expected_rw = np.array([reweighted_probs.get(m, 0) * total for m in margins])
    
    chi2_base, p_base, df_base = chi_square_test(counts, expected_base)
    chi2_rw, p_rw, df_rw = chi_square_test(counts, expected_rw)
    
    # MAE at key numbers
    mae_base = np.mean([
        abs(margin_to_count.get(k, 0) - base_probs.get(k, 0) * total)
        for k in key_numbers
    ])
    mae_rw = np.mean([
        abs(margin_to_count.get(k, 0) - reweighted_probs.get(k, 0) * total)
        for k in key_numbers
    ])
    
    # Generate LaTeX table
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \small",
        r"  \caption{Reweighting ablation: impact of key-mass adjustment.}",
        r"  \begin{tabular}{lccc}",
        r"    \toprule",
        r"    Method & $\chi^2$ (full) & $p$-value & MAE at keys \\",
        r"    \midrule",
        f"    Base (no reweight) & {chi2_base:.2f} & {p_base:.3f} & {mae_base:.2f} \\\\",
        f"    IPF reweighted & {chi2_rw:.2f} & {p_rw:.3f} & {mae_rw:.2f} \\\\",
        r"    \midrule",
        f"    Improvement & {chi2_base - chi2_rw:+.2f} & "
        f"{p_rw - p_base:+.3f} & {mae_base - mae_rw:+.2f} \\\\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"✓ Generated: {output_path}")
    
    return {
        "mae_improvement": mae_base - mae_rw,
        "chi2_improvement": chi2_base - chi2_rw,
        "p_value_improvement": p_rw - p_base,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate key-number calibration tables with chi-square tests"
    )
    parser.add_argument(
        "--season-start", type=int, default=1999,
        help="Start season for analysis"
    )
    parser.add_argument(
        "--season-end", type=int, default=2024,
        help="End season for analysis"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("analysis/dissertation/figures/out"),
        help="Output directory for TeX tables"
    )
    parser.add_argument(
        "--key-numbers", type=int, nargs="+",
        default=[3, 6, 7, 10, 14],
        help="Key margins to test"
    )
    
    args = parser.parse_args()
    
    print(f"Analyzing key-number calibration: {args.season_start}-{args.season_end}")
    print(f"Key numbers: {args.key_numbers}")
    
    # Connect and fetch data
    with get_connection() as conn:
        margins, counts = get_margin_distribution(
            conn, args.season_start, args.season_end
        )
    
    print(f"Loaded {len(margins)} unique margins, {counts.sum()} total games")
    
    # Generate tables
    calibration_stats = generate_calibration_table(
        margins, counts, args.key_numbers,
        args.output_dir / "keymass_chisq_table.tex"
    )
    
    ablation_stats = generate_reweighting_ablation_table(
        margins, counts, args.key_numbers,
        args.output_dir / "reweighting_ablation_table.tex"
    )
    
    # Save stats to JSON
    stats_path = args.output_dir / "keymass_calibration_stats.json"
    stats = {
        "season_range": [args.season_start, args.season_end],
        "key_numbers": args.key_numbers,
        "n_games": int(counts.sum()),
        "calibration": calibration_stats,
        "ablation": ablation_stats,
    }
    
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"✓ Saved stats: {stats_path}")
    
    print("\n✅ Key-number calibration tables generated successfully!")


if __name__ == "__main__":
    main()
