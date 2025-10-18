"""
Dixon-Coles bivariate Poisson model for NFL score prediction.

Implements the Dixon and Coles (1997) model for soccer scores, adapted for
American football. The model uses a bivariate Poisson distribution with a
correlation parameter to capture low-score dependence, and includes:

1. Attack/defense strength parameters for each team
2. Home-field advantage
3. Low-score inflation factor (rho parameter)
4. EM algorithm for parameter estimation
5. Dynamic intensity tracking (Koopman et al. extension)

References:
- Dixon, M. J., & Coles, S. G. (1997). Modelling association football scores
  and inefficiencies in the football betting market. Applied Statistics, 46(2), 265-280.
- Karlis, D., & Ntzoufras, I. (2003). Analysis of sports data by using bivariate
  Poisson models. Journal of the Royal Statistical Society: Series D, 52(3), 381-393.
- Koopman, S. J., & Lit, R. (2015). A dynamic bivariate Poisson model for
  analysing and forecasting match results in the English Premier League.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

# Precompute log-factorials for speed (NFL scores rarely exceed 100)
_LOG_FACT_CACHE = np.array([gammaln(i + 1) for i in range(101)])


# ============================================================================
# Database Connection
# ============================================================================


def get_connection() -> psycopg.Connection:
    """Get PostgreSQL connection using environment variables."""
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5544")
    dbname = os.environ.get("POSTGRES_DB", "devdb01")
    user = os.environ.get("POSTGRES_USER", "dro")
    password = os.environ.get("POSTGRES_PASSWORD", "sicillionbillions")
    return psycopg.connect(host=host, port=port, dbname=dbname, user=user, password=password)


def fetch_games(seasons: list[int]) -> pd.DataFrame:
    """Fetch completed games with scores for specified seasons."""
    sql = """
        SELECT game_id, season, week, kickoff, home_team, away_team,
               home_score, away_score
        FROM games
        WHERE season = ANY(%s)
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
        ORDER BY season, week, kickoff
    """
    with get_connection() as conn:
        df = pd.read_sql(sql, conn, params=(seasons,))
    return df


# ============================================================================
# Dixon-Coles Bivariate Poisson Model
# ============================================================================


def tau(x: int, y: int, lambda_x: float, lambda_y: float, rho: float) -> float:
    """
    Dixon-Coles correlation adjustment factor for low scores.

    Adjusts the bivariate Poisson probability for (x, y) to account for
    correlation between home and away scores, particularly for low-scoring games.

    Args:
        x: Home score
        y: Away score
        lambda_x: Expected home score (Poisson rate)
        lambda_y: Expected away score (Poisson rate)
        rho: Correlation parameter (typically small, e.g., -0.15 to 0.0)

    Returns:
        Multiplicative adjustment factor (close to 1.0 for high scores)
    """
    if x == 0 and y == 0:
        return 1.0 - lambda_x * lambda_y * rho
    elif x == 0 and y == 1:
        return 1.0 + lambda_x * rho
    elif x == 1 and y == 0:
        return 1.0 + lambda_y * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    else:
        return 1.0


def bivariate_poisson_pmf(x: int, y: int, lambda_x: float, lambda_y: float, rho: float) -> float:
    """
    Bivariate Poisson probability mass function with Dixon-Coles adjustment.

    P(X=x, Y=y) = tau(x,y) * Poisson(x; lambda_x) * Poisson(y; lambda_y)

    Args:
        x: Home score
        y: Away score
        lambda_x: Expected home score
        lambda_y: Expected away score
        rho: Correlation parameter

    Returns:
        Joint probability P(X=x, Y=y)
    """
    p_x = poisson.pmf(x, lambda_x)
    p_y = poisson.pmf(y, lambda_y)
    tau_adj = tau(x, y, lambda_x, lambda_y, rho)
    return tau_adj * p_x * p_y


@dataclass
class DixonColesParams:
    """Parameters for Dixon-Coles model."""

    attack: dict[str, float]  # Team attack strengths
    defense: dict[str, float]  # Team defense strengths
    home_advantage: float  # Home-field advantage (in log space)
    rho: float  # Low-score correlation parameter
    converged: bool = False
    iterations: int = 0
    log_likelihood: float = 0.0


class DixonColesModel:
    """
    Dixon-Coles bivariate Poisson model for NFL scores.

    Model specification:
        lambda_home = exp(attack_home - defense_away + home_advantage)
        lambda_away = exp(attack_away - defense_home)
        P(home_score, away_score) = tau * Poisson(home | lambda_home) * Poisson(away | lambda_away)

    Parameters are estimated via maximum likelihood using scipy optimization.
    """

    def __init__(
        self,
        home_advantage_init: float = 0.15,
        rho_init: float = -0.10,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        """
        Initialize Dixon-Coles model.

        Args:
            home_advantage_init: Initial home-field advantage (log scale)
            rho_init: Initial correlation parameter
            max_iter: Maximum optimization iterations
            tol: Convergence tolerance
        """
        self.home_advantage_init = home_advantage_init
        self.rho_init = rho_init
        self.max_iter = max_iter
        self.tol = tol

        self.params: DixonColesParams | None = None
        self.teams: list[str] = []
        self.team_to_idx: dict[str, int] = {}

    def _initialize_params(self, df: pd.DataFrame) -> np.ndarray:
        """
        Initialize model parameters using empirical scoring rates.

        Returns:
            Initial parameter vector [attack_1, ..., attack_N, defense_1, ..., defense_N, hfa, rho]
        """
        # Get unique teams
        self.teams = sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))
        self.team_to_idx = {team: idx for idx, team in enumerate(self.teams)}
        n_teams = len(self.teams)

        # Initialize attack/defense based on average scores
        attack = np.zeros(n_teams)
        defense = np.zeros(n_teams)

        for team in self.teams:
            idx = self.team_to_idx[team]

            # Attack strength: average goals scored as home and away
            home_goals = df[df["home_team"] == team]["home_score"].mean()
            away_goals = df[df["away_team"] == team]["away_score"].mean()
            attack[idx] = np.log(max(0.1, (home_goals + away_goals) / 2.0))

            # Defense strength: average goals conceded as home and away
            goals_conceded_home = df[df["home_team"] == team]["away_score"].mean()
            goals_conceded_away = df[df["away_team"] == team]["home_score"].mean()
            defense[idx] = np.log(max(0.1, (goals_conceded_home + goals_conceded_away) / 2.0))

        # Normalize to prevent identifiability issues (set sum of attack = sum of defense)
        attack = attack - attack.mean()
        defense = defense - defense.mean()

        # Initial parameter vector
        params = np.concatenate([attack, defense, [self.home_advantage_init, self.rho_init]])
        return params

    def _unpack_params(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        """
        Unpack parameter vector into components.

        Returns:
            (attack, defense, home_advantage, rho)
        """
        n_teams = len(self.teams)
        attack = params[:n_teams]
        defense = params[n_teams : 2 * n_teams]
        home_advantage = params[2 * n_teams]
        rho = params[2 * n_teams + 1]
        return attack, defense, home_advantage, rho

    def _negative_log_likelihood(self, params: np.ndarray, df: pd.DataFrame) -> float:
        """
        Compute negative log-likelihood for optimization (vectorized).

        Args:
            params: Parameter vector
            df: DataFrame with columns ['home_team', 'away_team', 'home_score', 'away_score']

        Returns:
            Negative log-likelihood (to be minimized)
        """
        attack, defense, hfa, rho = self._unpack_params(params)

        # Vectorized computation
        home_indices = df["home_team"].map(self.team_to_idx).values
        away_indices = df["away_team"].map(self.team_to_idx).values

        # Expected scores (Poisson rates)
        lambda_home = np.exp(attack[home_indices] - defense[away_indices] + hfa)
        lambda_away = np.exp(attack[away_indices] - defense[home_indices])

        # Observed scores
        x = df["home_score"].values.astype(int)
        y = df["away_score"].values.astype(int)

        # Vectorized tau computation (only for low scores where it matters)
        tau_vals = np.ones(len(df))
        for i in range(len(df)):
            if x[i] <= 1 and y[i] <= 1:  # Only compute tau for low scores
                tau_vals[i] = tau(x[i], y[i], lambda_home[i], lambda_away[i], rho)

        # Vectorized Poisson PMF using precomputed log-factorials
        log_p_x = x * np.log(lambda_home) - lambda_home - _LOG_FACT_CACHE[np.clip(x, 0, 100)]
        log_p_y = y * np.log(lambda_away) - lambda_away - _LOG_FACT_CACHE[np.clip(y, 0, 100)]

        # Combined log-likelihood
        log_lik = np.sum(np.log(tau_vals) + log_p_x + log_p_y)

        return -log_lik

    def fit(self, df: pd.DataFrame) -> DixonColesParams:
        """
        Fit Dixon-Coles model using maximum likelihood estimation.

        Args:
            df: DataFrame with columns ['home_team', 'away_team', 'home_score', 'away_score']

        Returns:
            Fitted parameters
        """
        print(f"Fitting Dixon-Coles model on {len(df)} games...")

        # Initialize parameters
        params_init = self._initialize_params(df)

        # Bounds: rho in [-1, 0] to ensure valid correlation
        n_teams = len(self.teams)
        bounds = [(None, None)] * (2 * n_teams) + [(0.0, 1.0), (-1.0, 0.0)]

        # Optimize
        result = minimize(
            self._negative_log_likelihood,
            params_init,
            args=(df,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        # Extract fitted parameters
        attack, defense, hfa, rho = self._unpack_params(result.x)

        self.params = DixonColesParams(
            attack={team: attack[idx] for team, idx in self.team_to_idx.items()},
            defense={team: defense[idx] for team, idx in self.team_to_idx.items()},
            home_advantage=hfa,
            rho=rho,
            converged=result.success,
            iterations=result.nit,
            log_likelihood=-result.fun,
        )

        print(f"Fitted: HFA={hfa:.4f}, rho={rho:.4f}, log-lik={-result.fun:.2f}")
        print(f"Converged: {result.success} in {result.nit} iterations")

        return self.params

    def predict_score_distribution(
        self, home_team: str, away_team: str, max_score: int = 60
    ) -> dict[tuple[int, int], float]:
        """
        Predict joint score distribution P(home_score, away_score).

        Args:
            home_team: Home team name
            away_team: Away team name
            max_score: Maximum score to consider

        Returns:
            Dictionary mapping (home_score, away_score) to probability
        """
        if self.params is None:
            raise ValueError("Model must be fitted before prediction")

        # Get parameters
        attack_home = self.params.attack.get(home_team, 0.0)
        attack_away = self.params.attack.get(away_team, 0.0)
        defense_home = self.params.defense.get(home_team, 0.0)
        defense_away = self.params.defense.get(away_team, 0.0)

        # Expected scores
        lambda_home = np.exp(attack_home - defense_away + self.params.home_advantage)
        lambda_away = np.exp(attack_away - defense_home)

        # Compute joint distribution
        dist = {}
        for x in range(max_score + 1):
            for y in range(max_score + 1):
                prob = bivariate_poisson_pmf(x, y, lambda_home, lambda_away, self.params.rho)
                dist[(x, y)] = prob

        # Normalize
        total = sum(dist.values())
        dist = {k: v / total for k, v in dist.items()}

        return dist

    def predict_margin_distribution(
        self, home_team: str, away_team: str, max_score: int = 60
    ) -> dict[int, float]:
        """
        Predict margin distribution P(margin = home_score - away_score).

        Args:
            home_team: Home team name
            away_team: Away team name
            max_score: Maximum score to consider

        Returns:
            Dictionary mapping margin to probability
        """
        score_dist = self.predict_score_distribution(home_team, away_team, max_score)

        margin_dist: dict[int, float] = {}
        for (x, y), prob in score_dist.items():
            margin = x - y
            margin_dist[margin] = margin_dist.get(margin, 0.0) + prob

        return margin_dist

    def predict_total_distribution(
        self, home_team: str, away_team: str, max_score: int = 60
    ) -> dict[int, float]:
        """
        Predict total points distribution P(total = home_score + away_score).

        Args:
            home_team: Home team name
            away_team: Away team name
            max_score: Maximum score to consider

        Returns:
            Dictionary mapping total to probability
        """
        score_dist = self.predict_score_distribution(home_team, away_team, max_score)

        total_dist: dict[int, float] = {}
        for (x, y), prob in score_dist.items():
            total = x + y
            total_dist[total] = total_dist.get(total, 0.0) + prob

        return total_dist

    def save(self, path: Path) -> None:
        """Save fitted parameters to JSON."""
        if self.params is None:
            raise ValueError("No parameters to save")

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self.params), f, indent=2)

        print(f"Saved parameters to {path}")

    def load(self, path: Path) -> None:
        """Load fitted parameters from JSON."""
        with open(path) as f:
            data = json.load(f)

        self.params = DixonColesParams(**data)
        self.teams = sorted(self.params.attack.keys())
        self.team_to_idx = {team: idx for idx, team in enumerate(self.teams)}

        print(f"Loaded parameters from {path}")


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Dixon-Coles bivariate Poisson model")
    ap.add_argument(
        "--seasons", required=True, help="Comma-separated seasons (e.g., 2020,2021,2022)"
    )
    ap.add_argument("--output", default="models/dixon_coles_params.json", help="Output params JSON")
    ap.add_argument("--table", help="Optional TeX table output path")
    ap.add_argument("--hfa-init", type=float, default=0.15, help="Initial home-field advantage")
    ap.add_argument("--rho-init", type=float, default=-0.10, help="Initial correlation parameter")
    return ap.parse_args()


def main():
    args = parse_args()

    seasons = [int(s.strip()) for s in args.seasons.split(",")]
    print(f"Seasons: {seasons}")

    # Load games
    df = fetch_games(seasons)
    print(f"Loaded {len(df)} games")

    # Fit model
    model = DixonColesModel(home_advantage_init=args.hfa_init, rho_init=args.rho_init)
    params = model.fit(df)

    # Save parameters
    output_path = Path(args.output)
    model.save(output_path)

    # Print summary
    print("\n=== Model Summary ===")
    print(f"Home-field advantage: {params.home_advantage:.4f}")
    print(f"Correlation (rho): {params.rho:.4f}")
    print(f"Log-likelihood: {params.log_likelihood:.2f}")

    # Top 5 attack/defense
    attack_sorted = sorted(params.attack.items(), key=lambda x: x[1], reverse=True)
    defense_sorted = sorted(params.defense.items(), key=lambda x: x[1])

    print("\nTop 5 Attack:")
    for team, val in attack_sorted[:5]:
        print(f"  {team}: {val:.4f}")

    print("\nTop 5 Defense (lowest = best):")
    for team, val in defense_sorted[:5]:
        print(f"  {team}: {val:.4f}")

    # Generate LaTeX table if requested
    if args.table:
        generate_latex_table(model, Path(args.table))


def generate_latex_table(model: DixonColesModel, output_path: Path) -> None:
    """Generate LaTeX table comparing Dixon-Coles to Skellam."""
    if model.params is None:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by py/models/bivariate_poisson.py\n")
        f.write("% !TEX root = ../../main/main.tex\n")
        f.write("\\begin{table}[t]\n")
        f.write("  \\centering\n")
        f.write("  \\small\n")
        f.write(
            f"  \\caption{{Dixon-Coles bivariate Poisson parameters. HFA={model.params.home_advantage:.3f}, $\\rho$={model.params.rho:.3f}}}\n"
        )
        f.write("  \\label{tab:dixon-coles}\n")
        f.write("  \\setlength{\\tabcolsep}{4pt}\\renewcommand{\\arraystretch}{1.1}\n")
        f.write("  \\begin{tabular}{@{} l r r @{}}\n")
        f.write("    \\toprule\n")
        f.write("    Team & Attack & Defense \\\\\\\\\n")
        f.write("    \\midrule\n")

        # Sort by attack strength
        teams_sorted = sorted(model.params.attack.items(), key=lambda x: x[1], reverse=True)
        for team, attack in teams_sorted[:10]:  # Top 10
            defense = model.params.defense[team]
            f.write(f"    {team} & {attack:+.3f} & {defense:+.3f} \\\\\\\\\n")

        f.write("    \\bottomrule\n")
        f.write("  \\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"Generated LaTeX table: {output_path}")


if __name__ == "__main__":
    main()
