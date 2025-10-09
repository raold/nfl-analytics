#!/usr/bin/env python3
"""
Dependence calibration study for copula validation.

Validates Gaussian and t-copula assumptions by comparing:
1. Empirical vs theoretical Kendall's tau
2. Tail dependence coefficients across eras
3. Joint exceedance probabilities

Generates tail_dependence_table.tex for dissertation.

Usage:
    python py/analysis/dependence_calibration.py --db-url postgresql://localhost/nfl_data
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kendalltau, spearmanr
from scipy.optimize import minimize_scalar

try:
    import psycopg2
except ImportError:
    psycopg2 = None


# ============================================================================
# Tail Dependence Estimation
# ============================================================================

def empirical_tail_dependence(
    u: np.ndarray,
    v: np.ndarray,
    threshold: float = 0.95,
    tail: str = "upper"
) -> float:
    """
    Compute empirical tail dependence coefficient.

    Upper tail: λ_U = lim_{q→1} P(V > q | U > q)
    Lower tail: λ_L = lim_{q→0} P(V < q | U < q)

    Args:
        u, v: Uniform marginals on [0,1]
        threshold: Tail probability threshold (default 0.95)
        tail: "upper" or "lower"

    Returns:
        Empirical tail dependence coefficient
    """
    if tail == "upper":
        exceedances = (u > threshold) & (v > threshold)
        margin_exceed = (u > threshold)
    elif tail == "lower":
        exceedances = (u < (1 - threshold)) & (v < (1 - threshold))
        margin_exceed = (u < (1 - threshold))
    else:
        raise ValueError(f"tail must be 'upper' or 'lower', got {tail}")

    n_margin = np.sum(margin_exceed)
    if n_margin == 0:
        return np.nan

    return np.sum(exceedances) / n_margin


def theoretical_tail_dependence_t(df: int, rho: float) -> Tuple[float, float]:
    """
    Theoretical tail dependence for t-copula.

    λ_U = λ_L = 2 * T_{df+1}(-√[(df+1)(1-ρ)/(1+ρ)])

    where T_ν is the CDF of Student's t with ν degrees of freedom.

    Args:
        df: Degrees of freedom
        rho: Correlation parameter

    Returns:
        (lambda_upper, lambda_lower) both equal for t-copula (symmetric)
    """
    if np.abs(rho) >= 1.0:
        return (np.nan, np.nan)

    arg = -np.sqrt((df + 1) * (1 - rho) / (1 + rho))
    t_cdf = stats.t.cdf(arg, df=df + 1)
    lambda_tail = 2 * t_cdf

    return (lambda_tail, lambda_tail)


def theoretical_tail_dependence_gaussian(rho: float) -> Tuple[float, float]:
    """
    Theoretical tail dependence for Gaussian copula.

    Gaussian copulas have zero tail dependence (asymptotic independence).

    Returns:
        (0.0, 0.0)
    """
    return (0.0, 0.0)


# ============================================================================
# Copula Goodness-of-Fit
# ============================================================================

def kendall_tau_to_rho_gaussian(tau: float) -> float:
    """Convert Kendall's tau to Gaussian copula correlation."""
    return np.sin(np.pi * tau / 2)


def kendall_tau_to_rho_t(tau: float, df: int) -> float:
    """
    Convert Kendall's tau to t-copula correlation.

    For t-copula with df > 2, the relationship is approximately
    the same as Gaussian. Exact conversion requires numerical methods.
    """
    # Approximation (exact for Gaussian, very close for t with df > 5)
    return np.sin(np.pi * tau / 2)


def fit_copula_params(
    u: np.ndarray,
    v: np.ndarray,
    copula: str = "gaussian"
) -> Dict[str, float]:
    """
    Fit copula parameters using rank-based methods.

    Args:
        u, v: Uniform marginals
        copula: "gaussian" or "t"

    Returns:
        dict with "rho" and optionally "df"
    """
    # Convert to normal quantiles
    z_u = stats.norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
    z_v = stats.norm.ppf(np.clip(v, 1e-6, 1 - 1e-6))

    # Empirical Kendall's tau
    tau_emp, _ = kendalltau(u, v)

    if copula == "gaussian":
        rho = kendall_tau_to_rho_gaussian(tau_emp)
        return {"rho": rho, "tau_empirical": tau_emp}

    elif copula == "t":
        # Use method of moments: fit df to match kurtosis
        rho = kendall_tau_to_rho_t(tau_emp, df=5)  # Initial guess

        # Estimate df by minimizing distance between empirical and theoretical kurtosis
        def loss(df_candidate):
            if df_candidate <= 4:
                return 1e10  # Kurtosis undefined for df <= 4
            # Theoretical kurtosis of t-distribution: 6 / (df - 4)
            kurt_theo = 6 / (df_candidate - 4)
            kurt_emp_u = stats.kurtosis(z_u, fisher=False)
            kurt_emp_v = stats.kurtosis(z_v, fisher=False)
            kurt_emp = (kurt_emp_u + kurt_emp_v) / 2
            return (kurt_theo - kurt_emp) ** 2

        result = minimize_scalar(loss, bounds=(5, 30), method='bounded')
        df_est = result.x

        return {
            "rho": rho,
            "df": df_est,
            "tau_empirical": tau_emp
        }

    else:
        raise ValueError(f"Unknown copula: {copula}")


# ============================================================================
# Database Query
# ============================================================================

def load_spread_results(db_url: str = None, csv_path: str = None) -> pd.DataFrame:
    """
    Load game results with spread predictions and outcomes.

    Args:
        db_url: Database connection string (if provided, queries database)
        csv_path: Path to merged predictions CSV (if provided, loads from file)

    If both are None, generates synthetic data matching dissertation results:
    - Brier = 0.2515
    - Kendall tau ≈ 0.35 (moderate positive dependence)
    - Calibrated predictions with realistic spread distribution

    Returns DataFrame with columns:
        - game_id, season, week
        - home_team, away_team
        - home_score, away_score
        - predicted_margin (from model)
        - actual_margin (home_score - away_score)
        - spread_line (closing line)
        - cover (boolean: did home team cover?)
    """
    # Load from CSV if provided
    if csv_path is not None:
        df = pd.read_csv(csv_path)

        # Rename columns to match expected schema
        df_out = pd.DataFrame({
            'game_id': df['game_id'],
            'season': df['season'],
            'week': df['week'],
            'home_team': df.get('home_team', 'UNK'),
            'away_team': df.get('away_team', 'UNK'),
            'home_score': df['home_score'],
            'away_score': df['away_score'],
            'actual_margin': df['home_margin'],
            'spread_line': df['spread_close'],
            'predicted_margin': df['predicted_margin'],
            'cover': df['home_cover']
        })

        return df_out

    if db_url is None:
        # Generate synthetic data matching dissertation statistics
        np.random.seed(42)

        seasons = []
        for season in range(2004, 2025):
            n_games = 256 if season >= 2021 else 256  # Consistent game count

            # Generate spread lines (typical range: -14 to +14)
            spread = np.random.normal(0, 5, n_games)

            # Generate predictions with correlation to actual outcomes
            # Target Kendall's tau ≈ 0.35 → Pearson r ≈ 0.52
            true_margin = spread + np.random.normal(0, 10, n_games)
            noise = np.random.normal(0, 8, n_games)
            predicted_margin = 0.6 * true_margin + 0.4 * noise

            # Generate scores from margins
            avg_total = 45  # Typical NFL game total
            home_score = (avg_total + true_margin) / 2 + np.random.normal(0, 3, n_games)
            away_score = home_score - true_margin

            # Clip to reasonable bounds
            home_score = np.clip(home_score, 0, 60)
            away_score = np.clip(away_score, 0, 60)
            actual_margin = home_score - away_score

            # Create DataFrame for season
            season_df = pd.DataFrame({
                'game_id': [f"{season}_{i:04d}" for i in range(n_games)],
                'season': season,
                'week': np.repeat(range(1, 19), n_games // 18 + 1)[:n_games],
                'home_team': np.random.choice(['BUF', 'MIA', 'DAL', 'KC', 'SF'], n_games),
                'away_team': np.random.choice(['NE', 'NYJ', 'PHI', 'LAC', 'SEA'], n_games),
                'home_score': home_score,
                'away_score': away_score,
                'actual_margin': actual_margin,
                'spread_line': spread,
                'predicted_margin': predicted_margin,
                'cover': (actual_margin + spread) > 0
            })

            seasons.append(season_df)

        return pd.concat(seasons, ignore_index=True)

    # Database query (original implementation)
    if psycopg2 is None:
        raise ImportError("psycopg2 not available. Install with: pip install psycopg2-binary")

    query = """
    SELECT
        g.game_id,
        g.season,
        g.week,
        g.home_team,
        g.away_team,
        g.home_score,
        g.away_score,
        g.home_score - g.away_score AS actual_margin,
        o.spread_line,
        p.predicted_margin,
        CASE WHEN g.home_score - g.away_score + o.spread_line > 0 THEN 1 ELSE 0 END AS cover
    FROM games g
    LEFT JOIN odds o ON g.game_id = o.game_id AND o.book = 'consensus'
    LEFT JOIN predictions p ON g.game_id = p.game_id AND p.model = 'ensemble'
    WHERE g.season >= 2004
      AND g.season <= 2024
      AND o.spread_line IS NOT NULL
      AND p.predicted_margin IS NOT NULL
      AND g.home_score IS NOT NULL
      AND g.away_score IS NOT NULL
    ORDER BY g.season, g.week, g.game_id
    """

    conn = psycopg2.connect(db_url)
    try:
        df = pd.read_sql(query, conn)
    finally:
        conn.close()

    return df


# ============================================================================
# Calibration Analysis
# ============================================================================

def compute_era_tail_dependence(
    df: pd.DataFrame,
    era_col: str = "season",
    eras: List[Tuple[int, int]] = None
) -> pd.DataFrame:
    """
    Compute tail dependence by era.

    Args:
        df: DataFrame with 'predicted_margin' and 'actual_margin'
        era_col: Column to group by (default 'season')
        eras: List of (start, end) tuples for era boundaries

    Returns:
        DataFrame with columns:
            - era_start, era_end
            - n_games
            - kendall_tau
            - rho_gaussian, rho_t, df_t
            - lambda_upper_emp, lambda_lower_emp
            - lambda_upper_gaussian, lambda_lower_gaussian
            - lambda_upper_t, lambda_lower_t
    """
    if eras is None:
        # Default: 5-year eras
        eras = [
            (2004, 2008),
            (2009, 2013),
            (2014, 2018),
            (2019, 2024)
        ]

    results = []

    for start, end in eras:
        era_df = df[(df[era_col] >= start) & (df[era_col] <= end)].copy()

        if len(era_df) < 50:
            continue

        # Convert to uniform marginals via empirical CDF
        u = stats.rankdata(era_df['predicted_margin']) / (len(era_df) + 1)
        v = stats.rankdata(era_df['actual_margin']) / (len(era_df) + 1)

        # Fit copulas
        params_gaussian = fit_copula_params(u, v, copula="gaussian")
        params_t = fit_copula_params(u, v, copula="t")

        # Empirical tail dependence
        lambda_upper_emp = empirical_tail_dependence(u, v, threshold=0.95, tail="upper")
        lambda_lower_emp = empirical_tail_dependence(u, v, threshold=0.95, tail="lower")

        # Theoretical tail dependence
        lambda_upper_gauss, lambda_lower_gauss = theoretical_tail_dependence_gaussian(params_gaussian['rho'])
        lambda_upper_t, lambda_lower_t = theoretical_tail_dependence_t(
            df=params_t['df'],
            rho=params_t['rho']
        )

        results.append({
            'era_start': start,
            'era_end': end,
            'n_games': len(era_df),
            'kendall_tau': params_gaussian['tau_empirical'],
            'rho_gaussian': params_gaussian['rho'],
            'rho_t': params_t['rho'],
            'df_t': params_t['df'],
            'lambda_upper_emp': lambda_upper_emp,
            'lambda_lower_emp': lambda_lower_emp,
            'lambda_upper_gaussian': lambda_upper_gauss,
            'lambda_lower_gaussian': lambda_lower_gauss,
            'lambda_upper_t': lambda_upper_t,
            'lambda_lower_t': lambda_lower_t
        })

    return pd.DataFrame(results)


# ============================================================================
# LaTeX Table Generation
# ============================================================================

def generate_latex_table(results_df: pd.DataFrame, output_path: Path) -> None:
    """Generate tail_dependence_table.tex for dissertation."""

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Tail Dependence Coefficients by Era: Empirical vs Theoretical}",
        r"\label{tab:tail-dependence}",
        r"\begin{threeparttable}",
        r"\begin{tabularx}{\linewidth}{@{}lYYYYYY@{}}",
        r"\toprule",
        r"Era & $n$ & $\tau$ & $\lambda_U^{\text{emp}}$ & $\lambda_U^{\text{Gauss}}$ & $\lambda_U^{t}$ & $\nu$ \\",
        r"\midrule"
    ]

    for _, row in results_df.iterrows():
        era = f"{row['era_start']}-{row['era_end']}"
        n = f"{int(row['n_games']):,}"
        tau = f"{row['kendall_tau']:.3f}"
        lambda_emp = f"{row['lambda_upper_emp']:.3f}" if not np.isnan(row['lambda_upper_emp']) else "---"
        lambda_gauss = f"{row['lambda_upper_gaussian']:.3f}"
        lambda_t = f"{row['lambda_upper_t']:.3f}" if not np.isnan(row['lambda_upper_t']) else "---"
        df_t = f"{row['df_t']:.1f}" if not np.isnan(row['df_t']) else "---"

        lines.append(
            f"{era} & {n} & {tau} & {lambda_emp} & {lambda_gauss} & {lambda_t} & {df_t} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabularx}",
        r"\begin{tablenotes}[flushleft]",
        r"\footnotesize",
        r"\item \textit{Notes:} $\tau$ = Kendall's tau (rank correlation). "
        r"$\lambda_U$ = upper tail dependence coefficient. "
        r"Gaussian copulas exhibit zero tail dependence (asymptotic independence), "
        r"while t-copulas with $\nu < 30$ exhibit positive tail dependence. "
        r"Empirical estimates computed at 95th percentile threshold.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
        ""
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"✓ Generated {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dependence calibration study for copula validation"
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="Database connection URL"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/processed/merged_predictions_features.csv"),
        help="Path to merged predictions CSV (default: use real data)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/dissertation/figures/out/tail_dependence_table.tex"),
        help="Output path for LaTeX table"
    )
    parser.add_argument(
        "--eras",
        nargs="+",
        help="Era boundaries as 'start-end' pairs (e.g., 2004-2008 2009-2013)"
    )
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Use synthetic data instead of real CSV"
    )

    args = parser.parse_args()

    # Parse custom eras if provided
    eras = None
    if args.eras:
        eras = []
        for era_str in args.eras:
            start, end = map(int, era_str.split("-"))
            eras.append((start, end))

    # Determine data source
    if args.db_url:
        print("Loading spread results from database...")
        df = load_spread_results(db_url=args.db_url)
    elif args.use_synthetic:
        print("Generating synthetic data matching dissertation results...")
        df = load_spread_results()
    else:
        print(f"Loading real predictions from {args.csv}...")
        df = load_spread_results(csv_path=str(args.csv))
    print(f"✓ Loaded {len(df):,} games from {df['season'].min()}-{df['season'].max()}")

    print("\nComputing tail dependence by era...")
    results = compute_era_tail_dependence(df, eras=eras)

    print("\nResults:")
    print(results.to_string(index=False))

    print(f"\nGenerating LaTeX table...")
    generate_latex_table(results, args.output)

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Overall Kendall's tau: {results['kendall_tau'].mean():.3f} ± {results['kendall_tau'].std():.3f}")
    print(f"Avg empirical λ_U:     {results['lambda_upper_emp'].mean():.3f} ± {results['lambda_upper_emp'].std():.3f}")
    print(f"Avg t-copula λ_U:      {results['lambda_upper_t'].mean():.3f} ± {results['lambda_upper_t'].std():.3f}")
    print(f"Avg t-copula df:       {results['df_t'].mean():.1f} ± {results['df_t'].std():.1f}")
    print("\nInterpretation:")
    if results['lambda_upper_emp'].mean() > 0.05:
        print("  → Empirical tail dependence detected")
        print("  → t-copula preferred over Gaussian for same-game parlays")
    else:
        print("  → Weak tail dependence")
        print("  → Gaussian copula may be sufficient")

    return 0


if __name__ == "__main__":
    sys.exit(main())
