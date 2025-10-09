"""
State-space ratings models for NFL team strength estimation.

Implements:
1. Glickman-Stern (1994) dynamic rating system with Kalman filter
2. Weekly team strength posteriors (θ_t) with uncertainty
3. Home-field advantage (HFA) estimation
4. Integration with backtest harness

Model:
  θ_t = θ_{t-1} + w_t,  w_t ~ N(0, Q)  (state evolution)
  y_t = H·θ_t + v_t,     v_t ~ N(0, R)  (observation)

where:
  - θ_t: team strength vector at week t
  - Q: process noise covariance (weekly variance in strength)
  - R: observation noise (margin variance given strength differential)
  - y_t: observed margin (home_score - away_score)
  - H: design matrix ([home_idx] - [away_idx] + [hfa])

Usage:
  # Fit model on historical data
  python py/models/state_space.py --seasons 2020,2021,2022,2023 --output models/state_space_ratings.csv

  # Evaluate predictions
  python py/models/state_space.py --seasons 2024 --load models/state_space_ratings.csv --evaluate
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from scipy.stats import norm
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error

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
    """Fetch games with scores and spreads for specified seasons."""
    sql = """
        SELECT game_id, season, week, kickoff, home_team, away_team,
               home_score, away_score, spread_close
        FROM games
        WHERE season = ANY(%s)
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
        ORDER BY season, week, kickoff
    """
    with get_connection() as conn:
        df = pd.read_sql(sql, conn, params=(seasons,))
    df["margin"] = df["home_score"] - df["away_score"]
    return df


# ============================================================================
# Kalman Filter State-Space Model
# ============================================================================


class StateSpaceRatings:
    """
    Dynamic team ratings via Kalman filter (Glickman-Stern approach).

    Parameters:
    - q: process noise std (weekly strength volatility, default 3.0 points)
    - r: observation noise std (margin residual std, default 13.5 points)
    - init_rating: initial team strength (default 0.0)
    - init_variance: initial uncertainty (default 100.0)
    - hfa_init: initial home-field advantage (default 2.5 points)
    - hfa_variance: HFA uncertainty (default 4.0)
    """

    def __init__(
        self,
        q: float = 3.0,
        r: float = 13.5,
        init_rating: float = 0.0,
        init_variance: float = 100.0,
        hfa_init: float = 2.5,
        hfa_variance: float = 4.0,
    ):
        self.q = q
        self.r = r
        self.init_rating = init_rating
        self.init_variance = init_variance
        self.hfa_init = hfa_init
        self.hfa_variance = hfa_variance

        # State: {team: (mean, variance)}
        self.ratings: dict[str, tuple[float, float]] = {}
        self.hfa = hfa_init

        # History: [(week, team, rating, variance)]
        self.history: list[tuple[int, int, str, float, float]] = []

    def initialize_team(self, team: str):
        """Initialize team rating if not seen before."""
        if team not in self.ratings:
            self.ratings[team] = (self.init_rating, self.init_variance)

    def predict_step(self):
        """
        Time update: add process noise to all teams.
        θ_{t|t-1} = θ_{t-1|t-1}
        P_{t|t-1} = P_{t-1|t-1} + Q
        """
        for team in self.ratings:
            mean, var = self.ratings[team]
            self.ratings[team] = (mean, var + self.q**2)

    def update_game(self, home_team: str, away_team: str, margin: float):
        """
        Measurement update: incorporate observed margin.

        Observation: y = θ_home - θ_away + hfa + v, v ~ N(0, r^2)

        Kalman gain: K = P·H^T / (H·P·H^T + R)
        Innovation: ε = y - (θ_home - θ_away + hfa)
        Update: θ_new = θ_old + K·ε
                P_new = (I - K·H)·P
        """
        self.initialize_team(home_team)
        self.initialize_team(away_team)

        mean_h, var_h = self.ratings[home_team]
        mean_a, var_a = self.ratings[away_team]

        # Predicted margin
        margin_pred = mean_h - mean_a + self.hfa

        # Innovation (prediction error)
        innovation = margin - margin_pred

        # Observation variance: var(y) = var_h + var_a + hfa_var + r^2
        obs_var = var_h + var_a + self.hfa_variance + self.r**2

        # Kalman gains
        k_h = var_h / obs_var
        k_a = var_a / obs_var

        # Update ratings
        mean_h_new = mean_h + k_h * innovation
        mean_a_new = mean_a - k_a * innovation  # Subtract for away team

        var_h_new = var_h * (1 - k_h)
        var_a_new = var_a * (1 - k_a)

        # Store updated ratings
        self.ratings[home_team] = (mean_h_new, var_h_new)
        self.ratings[away_team] = (mean_a_new, var_a_new)

    def fit(self, df: pd.DataFrame):
        """
        Fit model on DataFrame with columns: season, week, home_team, away_team, margin.

        Process:
        1. Group by (season, week)
        2. For each week:
           a. Predict step (add process noise)
           b. Update step (incorporate all games)
           c. Log ratings
        """
        print(f"Fitting state-space model on {len(df)} games...")

        for (season, week), week_df in df.groupby(["season", "week"], sort=True):
            # Predict step (start of week)
            self.predict_step()

            # Update step (observe games)
            for _, row in week_df.iterrows():
                self.update_game(row["home_team"], row["away_team"], row["margin"])

            # Log ratings at end of week
            for team, (mean, var) in self.ratings.items():
                self.history.append((season, week, team, mean, var))

        print(f"Fitted {len(self.ratings)} teams over {len(self.history)} team-weeks")

    def predict_margin(self, home_team: str, away_team: str) -> tuple[float, float]:
        """
        Predict margin with uncertainty.

        Returns:
            (mean_margin, std_margin)
        """
        self.initialize_team(home_team)
        self.initialize_team(away_team)

        mean_h, var_h = self.ratings[home_team]
        mean_a, var_a = self.ratings[away_team]

        mean_margin = mean_h - mean_a + self.hfa
        std_margin = np.sqrt(var_h + var_a + self.hfa_variance + self.r**2)

        return mean_margin, std_margin

    def predict_prob_home_win(self, home_team: str, away_team: str) -> float:
        """Predict P(home wins) using normal CDF."""
        mean_margin, std_margin = self.predict_margin(home_team, away_team)
        return norm.cdf(mean_margin / std_margin)

    def predict_prob_ats(self, home_team: str, away_team: str, spread: float) -> float:
        """
        Predict P(home covers spread).

        Home covers if: home_score - away_score + spread > 0
        """
        mean_margin, std_margin = self.predict_margin(home_team, away_team)
        adjusted_margin = mean_margin + spread
        return norm.cdf(adjusted_margin / std_margin)

    def get_ratings_df(self) -> pd.DataFrame:
        """Return history as DataFrame."""
        return pd.DataFrame(self.history, columns=["season", "week", "team", "rating", "variance"])


# ============================================================================
# Evaluation
# ============================================================================


def evaluate_model(model: StateSpaceRatings, df: pd.DataFrame) -> dict[str, float]:
    """
    Evaluate model on held-out games.

    Metrics:
    - MAE (margin prediction)
    - Brier score (win probability)
    - Log loss (win probability)
    - ATS accuracy (if spread available)
    """
    margins_pred = []
    margins_true = []
    probs_win = []
    actuals_win = []
    probs_ats = []
    actuals_ats = []

    for _, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        margin_true = row["margin"]

        # Predict margin
        margin_pred, _ = model.predict_margin(home_team, away_team)
        margins_pred.append(margin_pred)
        margins_true.append(margin_true)

        # Predict win probability
        prob_win = model.predict_prob_home_win(home_team, away_team)
        probs_win.append(prob_win)
        actuals_win.append(1.0 if margin_true > 0 else 0.0)

        # Predict ATS if spread available
        if pd.notna(row.get("spread_close")):
            spread = float(row["spread_close"])
            prob_ats = model.predict_prob_ats(home_team, away_team, spread)
            probs_ats.append(prob_ats)
            actuals_ats.append(1.0 if (margin_true + spread) > 0 else 0.0)

    metrics = {
        "n_games": len(df),
        "mae_margin": mean_absolute_error(margins_true, margins_pred),
        "brier_win": brier_score_loss(actuals_win, probs_win),
        "logloss_win": log_loss(actuals_win, probs_win),
        "accuracy_win": float(
            np.mean([(p >= 0.5) == bool(a) for p, a in zip(probs_win, actuals_win)])
        ),
    }

    if probs_ats:
        metrics["brier_ats"] = brier_score_loss(actuals_ats, probs_ats)
        metrics["logloss_ats"] = log_loss(actuals_ats, probs_ats)
        metrics["accuracy_ats"] = float(
            np.mean([(p >= 0.5) == bool(a) for p, a in zip(probs_ats, actuals_ats)])
        )

    return metrics


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="State-space team ratings (Glickman-Stern)")
    ap.add_argument(
        "--seasons", required=True, help="Comma-separated seasons (e.g., 2020,2021,2022)"
    )
    ap.add_argument("--output", default="models/state_space_ratings.csv", help="Output ratings CSV")
    ap.add_argument("--load", help="Load pre-trained ratings CSV")
    ap.add_argument("--evaluate", action="store_true", help="Evaluation mode")
    ap.add_argument(
        "--q", type=float, default=3.0, help="Process noise std (weekly strength volatility)"
    )
    ap.add_argument("--r", type=float, default=13.5, help="Observation noise std (margin residual)")
    ap.add_argument("--hfa-init", type=float, default=2.5, help="Initial home-field advantage")
    ap.add_argument(
        "--init-variance", type=float, default=100.0, help="Initial team rating variance"
    )
    return ap.parse_args()


def main():
    args = parse_args()

    seasons = [int(s.strip()) for s in args.seasons.split(",")]
    print(f"Seasons: {seasons}")

    # Load games
    df = fetch_games(seasons)
    print(f"Loaded {len(df)} games")

    if args.evaluate and args.load:
        # Evaluation mode: load pre-trained ratings and evaluate
        print("\n=== Evaluation Mode ===")
        print(f"Loading ratings from {args.load}...")

        # Load pre-trained ratings from CSV
        ratings_df = pd.read_csv(args.load)
        print(f"Loaded {len(ratings_df)} team ratings from {args.load}")

        # Reconstruct model state
        model = StateSpaceRatings(
            q=args.q, r=args.r, init_variance=args.init_variance, hfa_init=args.hfa_init
        )

        # Restore ratings from CSV
        for _, row in ratings_df.iterrows():
            team = row["team"]
            rating = row["rating"]
            variance = row.get("variance", args.init_variance)  # Default if not in CSV
            model.ratings[team] = rating
            model.variances[team] = variance

        print(f"Restored ratings for {len(model.ratings)} teams")

        # Evaluate on provided seasons (typically held-out test set)
        print(f"\nEvaluating on seasons {seasons}...")
        metrics = evaluate_model(model, df)

        # Save metrics to JSON
        metrics_file = args.load.replace(".csv", "_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {metrics_file}")
        print(json.dumps(metrics, indent=2))

    else:
        # Training mode
        print("\n=== Training Mode ===")
        model = StateSpaceRatings(
            q=args.q, r=args.r, init_variance=args.init_variance, hfa_init=args.hfa_init
        )

        # Split: train on first N-1 seasons, test on last season
        if len(seasons) > 1:
            train_seasons = seasons[:-1]
            test_seasons = [seasons[-1]]

            train_df = df[df["season"].isin(train_seasons)]
            test_df = df[df["season"].isin(test_seasons)]

            print(f"\nTrain: seasons {train_seasons} ({len(train_df)} games)")
            print(f"Test:  seasons {test_seasons} ({len(test_df)} games)")

            # Fit on training data
            model.fit(train_df)

            # Evaluate on test data
            print("\n=== Test Set Evaluation ===")
            metrics = evaluate_model(model, test_df)
            print(json.dumps(metrics, indent=2))

            # Save metrics
            metrics_path = Path(args.output).parent / "state_space_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"\nMetrics saved to {metrics_path}")
        else:
            # No test set, just fit on all data
            print("\nFitting on all data (no test set)...")
            model.fit(df)

        # Save ratings history
        ratings_df = model.get_ratings_df()
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        ratings_df.to_csv(args.output, index=False)
        print(f"Ratings saved to {args.output}")

        # Print summary stats
        print("\n=== Final Ratings Summary ===")
        final_week = ratings_df[ratings_df["season"] == seasons[-1]]["week"].max()
        final_ratings = ratings_df[
            (ratings_df["season"] == seasons[-1]) & (ratings_df["week"] == final_week)
        ].sort_values("rating", ascending=False)
        print(final_ratings[["team", "rating", "variance"]].head(10).to_string(index=False))
        print(f"\nHome-field advantage: {model.hfa:.2f} points")


if __name__ == "__main__":
    main()
