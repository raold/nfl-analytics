"""
Build a logged dataset for offline RL/OPE using mart features.

Pulls games + mart.game_summary (EPA features), fits a simple logistic baseline
to estimate win probability, compares to market-implied probabilities to define
an edge, and derives a soft behavior propensity and target policy probability.

Output CSV columns (one-step per game):
  game_id, season, week, spread_close, total_close, home_score, away_score,
  home_epa_mean, away_epa_mean, epa_gap, market_prob, p_hat,
  action, r, b_prob, pi_prob, edge

Usage:
  python py/rl/dataset.py --output data/rl_logged.csv --season-start 2019 --season-end 2024
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd
import psycopg
from sklearn.linear_model import LogisticRegression


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build minimal logged dataset for OPE")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument("--season-start", type=int, default=2019)
    ap.add_argument("--season-end", type=int, default=2024)
    ap.add_argument(
        "--b-propensity", type=float, default=0.2, help="Behavior propensity for action=1"
    )
    ap.add_argument(
        "--price-decimal", type=float, default=1.91, help="Decimal odds for spread (-110≈1.91)"
    )
    return ap.parse_args()


def get_connection() -> psycopg.Connection:
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5544")
    dbname = os.environ.get("POSTGRES_DB", "devdb01")
    user = os.environ.get("POSTGRES_USER", "dro")
    password = os.environ.get("POSTGRES_PASSWORD", "sicillionbillions")
    return psycopg.connect(host=host, port=port, dbname=dbname, user=user, password=password)


def fetch_games(season_start: int, season_end: int) -> pd.DataFrame:
    sql = """
        SELECT g.game_id, g.season, g.week, g.home_team, g.away_team,
               g.spread_close, g.total_close, g.home_score, g.away_score,
               ms.home_epa_mean, ms.away_epa_mean
        FROM games g
        LEFT JOIN mart.game_summary ms ON ms.game_id = g.game_id
        WHERE g.season BETWEEN %s AND %s
    """
    with get_connection() as conn:
        df = pd.read_sql(sql, conn, params=(season_start, season_end))
    return df


def _american_to_decimal(ml: float) -> float:
    if pd.isna(ml):
        return np.nan
    if ml >= 0:
        return 1.0 + ml / 100.0
    return 1.0 + 100.0 / abs(ml)


def _dejuice(p1: float, p2: float) -> tuple[float, float]:
    """Remove overround from two-sided implied probs by scaling to sum=1."""
    s = p1 + p2
    if s <= 0:
        return p1, p2
    return p1 / s, p2 / s


def _market_prob(row: pd.Series) -> float:
    # Prefer moneylines if present; else approximate from spread via normal margin sd
    h_ml = row.get("home_moneyline")
    a_ml = row.get("away_moneyline")
    if pd.notna(h_ml) and pd.notna(a_ml):
        d_h = _american_to_decimal(h_ml)
        d_a = _american_to_decimal(a_ml)
        p_h = 1.0 / d_h if d_h and d_h > 0 else np.nan
        p_a = 1.0 / d_a if d_a and d_a > 0 else np.nan
        p_h, _ = _dejuice(p_h, p_a)
        return p_h
    # Spread-based prob (Stern approx): P(home win) ~ Phi(spread/sigma)
    sigma = 13.5
    sp = float(row.get("spread_close", 0.0))
    return 0.5 * (1.0 + math.erf(sp / (sigma * math.sqrt(2))))


def build_logged_dataset(
    df: pd.DataFrame, b_propensity: float, price_decimal: float
) -> pd.DataFrame:
    b = price_decimal - 1.0  # net odds used for reward

    # Features and target for a quick logistic baseline
    df_feat = df.copy()
    if "home_epa_mean" not in df_feat.columns:
        df_feat["home_epa_mean"] = np.nan
    if "away_epa_mean" not in df_feat.columns:
        df_feat["away_epa_mean"] = np.nan
    df_feat["epa_gap"] = df_feat["home_epa_mean"].astype(float).fillna(0.0) - df_feat[
        "away_epa_mean"
    ].astype(float).fillna(0.0)
    # Target: home win
    y = (df_feat["home_score"] > df_feat["away_score"]).astype(int)
    X = pd.DataFrame(
        {
            "spread_close": df_feat["spread_close"].fillna(0.0).astype(float),
            "total_close": df_feat["total_close"]
            .fillna(
                df_feat["total_close"].median() if df_feat["total_close"].notna().any() else 44.0
            )
            .astype(float),
            "epa_gap": df_feat["epa_gap"].astype(float),
        }
    )
    # Train simple logistic (fit on all rows with outcomes available)
    mask_fit = y.notna() & X.notna().all(axis=1)
    clf = LogisticRegression(max_iter=1000)
    if mask_fit.sum() >= 10:
        clf.fit(X.loc[mask_fit], y.loc[mask_fit])
        p_hat = clf.predict_proba(X)[:, 1]
    else:
        p_hat = np.full(len(X), 0.5)

    # Market implied probability
    if "home_moneyline" not in df_feat.columns:
        df_feat["home_moneyline"] = np.nan
        df_feat["away_moneyline"] = np.nan
    market_prob = df_feat.apply(_market_prob, axis=1)

    edge = p_hat - market_prob

    # Soft behavior propensity: more likely to bet when |edge| is large and spread within teaser band
    b_prob = 1.0 / (1.0 + np.exp(-6.0 * (edge - 0.01)))
    b_prob *= (df_feat["spread_close"].abs() <= 7.5).astype(float)
    b_prob = np.clip(b_prob + 0.05, 0.05, 0.95)

    # Target policy probability: slightly more aggressive than behavior
    pi_prob = 1.0 / (1.0 + np.exp(-8.0 * (edge - 0.0)))
    pi_prob *= (df_feat["spread_close"].abs() <= 7.5).astype(float)
    pi_prob = np.clip(pi_prob + 0.05, 0.05, 0.99)

    # Logged action: stochastic draw under behavior propensity (for scaffold reproducibility, use threshold)
    action = ((edge > 0.0) & (df_feat["spread_close"].abs() <= 7.5)).astype(int)

    # Realized cover for home at closing line
    margin = (df_feat["home_score"] - df_feat["away_score"]).astype(float)
    cover = (margin > -df_feat["spread_close"].astype(float)).astype(int)
    r = action * (cover * b + (1 - cover) * (-1.0))

    out = df_feat[
        [
            "game_id",
            "season",
            "week",
            "spread_close",
            "total_close",
            "home_score",
            "away_score",
            "home_epa_mean",
            "away_epa_mean",
        ]
    ].copy()
    out["epa_gap"] = X["epa_gap"].to_numpy()
    out["market_prob"] = market_prob
    out["p_hat"] = p_hat
    out["edge"] = edge
    out["action"] = action
    out["r"] = r
    out["b_prob"] = b_prob
    out["pi_prob"] = pi_prob
    return out


def main() -> None:
    args = parse_args()
    df = fetch_games(args.season_start, args.season_end)
    if df.empty:
        raise SystemExit("No games found in DB; run schedule ingestion first")
    out = build_logged_dataset(df, args.b_propensity, args.price_decimal)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"[dataset] wrote {len(out)} rows -> {args.output}")


if __name__ == "__main__":
    main()
