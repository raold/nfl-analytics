"""
Benchmark metrics: against-the-spread record, ROI at real prices, accuracy versus the closing
line, probabilistic skill versus the de-vigged market, and whether the model adds information
the line does not already contain.

Every headline number comes with an uncertainty estimate. ATS results are noisy: at 208 bets
the 95% interval on ROI is roughly +/-13 points.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from nfl_analytics.analysis.statistical_tests import DieboldMarianoTest, apply_holm_bonferroni
from nfl_analytics.evaluation.market_benchmark.odds import (
    STANDARD_PRICE,
    breakeven_win_rate,
    profit_per_unit,
)
from nfl_analytics.utils.calibration import brier_score, log_loss
from nfl_analytics.utils.spread_convention import home_cover_margin, model_edge

BLOCK = ["season", "week"]  # games in the same week share information; resample weeks together


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (math.nan, math.nan)
    p = wins / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (centre - half, centre + half)


def grade_ats(
    df: pd.DataFrame, pred_col: str = "pred_home_margin", min_edge: float = 1.0
) -> pd.DataFrame:
    """One row per game where |pred - line| >= min_edge, graded against the closing spread.

    Pushes are kept (result 'push', profit 0) so they can be counted, and excluded from rates.
    """
    edge = model_edge(df[pred_col], df["spread_close"])
    mask = edge.abs().ge(min_edge) & edge.ne(0) & df[pred_col].notna()
    bets = df.loc[mask].copy()
    bets["edge"] = edge[mask]
    bets["side"] = np.where(bets["edge"] > 0, "home", "away")
    cover = home_cover_margin(bets["home_margin"], bets["spread_close"])
    side_margin = np.where(bets["side"] == "home", cover, -cover)
    bets["result"] = np.select([side_margin > 0, side_margin < 0], ["win", "loss"], "push")

    price = np.where(bets["side"] == "home", bets["home_spread_odds"], bets["away_spread_odds"])
    price = pd.Series(price, index=bets.index, dtype=float).fillna(STANDARD_PRICE)
    bets["price"] = price
    bets["profit"] = np.select(
        [bets["result"] == "win", bets["result"] == "loss"],
        [price.map(profit_per_unit), -1.0],
        0.0,
    )
    return bets


def _block_bootstrap_mean(
    df: pd.DataFrame, col: str, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% interval for the mean of ``col``, resampling whole weeks (NaN with fewer than 2)."""
    groups = df.groupby(BLOCK)[col].agg(["sum", "count"]).to_numpy()
    if len(groups) < 2:  # one week resampled is always itself: no interval
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    means = groups[draws, 0].sum(axis=1) / groups[draws, 1].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def block_bootstrap_roi(
    bets: pd.DataFrame, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% interval for ROI (profit per settled bet), resampling whole weeks."""
    settled = bets[bets["result"] != "push"]
    if settled.empty:
        return (math.nan, math.nan)
    return _block_bootstrap_mean(settled, "profit", n_boot=n_boot, seed=seed)


def ats_summary(bets: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """Win-loss-push record, win rate with Wilson interval, ROI with block-bootstrap interval."""
    wins = int((bets["result"] == "win").sum())
    losses = int((bets["result"] == "loss").sum())
    pushes = int((bets["result"] == "push").sum())
    settled = wins + losses
    roi = float(bets.loc[bets["result"] != "push", "profit"].mean()) if settled else math.nan
    breakeven = (
        float(bets.loc[bets["result"] != "push", "price"].map(breakeven_win_rate).mean())
        if settled
        else breakeven_win_rate()
    )
    p_value = (
        float(stats.binomtest(wins, settled, breakeven, alternative="greater").pvalue)
        if settled
        else math.nan
    )
    return {
        "bets": settled,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": wins / settled if settled else math.nan,
        "win_rate_ci95": wilson_interval(wins, settled),
        "breakeven_win_rate": breakeven,
        "p_value_beats_breakeven": p_value,
        "roi": roi,
        "roi_ci95": block_bootstrap_roi(bets, n_boot=n_boot, seed=seed),
    }


def edge_threshold_sweep(
    df: pd.DataFrame,
    thresholds=(0.5, 1, 2, 3, 4),
    pred_col: str = "pred_home_margin",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """ATS by minimum edge, with Holm correction: trying several thresholds and reporting the
    best one is how mined filters ("2-4 pt edge: +25.7%") get through."""
    rows = []
    for t in thresholds:
        s = ats_summary(grade_ats(df, pred_col, t), n_boot=500)
        rows.append(
            {
                "min_edge": t,
                "bets": s["bets"],
                "win_rate": s["win_rate"],
                "roi": s["roi"],
                "p_value_one_sided": s["p_value_beats_breakeven"],
            }
        )
    out = pd.DataFrame(rows)
    valid = out.dropna(subset=["p_value_one_sided"])
    if valid.empty:
        out["significant_holm"] = False
        return out
    corrected = apply_holm_bonferroni(valid.copy(), alpha=alpha)
    return out.merge(corrected[["min_edge", "significant_holm"]], on="min_edge", how="left")


def spread_accuracy(df: pd.DataFrame, pred_col: str = "pred_home_margin") -> dict:
    """Model vs closing-line MAE on the home margin, with a Diebold-Mariano test."""
    d = df.dropna(subset=[pred_col])
    actual = d["home_margin"].to_numpy(float)
    model = d[pred_col].to_numpy(float)
    market = d["spread_close"].to_numpy(float)
    model_mae = float(np.mean(np.abs(actual - model)))
    market_mae = float(np.mean(np.abs(actual - market)))
    loss_diff = np.abs(actual - model) - np.abs(actual - market)
    if np.allclose(loss_diff, loss_diff[0]):  # zero variance: DM statistic undefined
        p_two = 1.0 if np.isclose(loss_diff[0], 0) else 0.0
    else:
        p_two = float(
            DieboldMarianoTest(loss_fn="absolute").test(actual, model, market)["p_value_two_sided"]
        )
    return {
        "games": len(d),
        "model_mae": model_mae,
        "market_mae": market_mae,
        "mae_skill": 1 - model_mae / market_mae,
        "dm_p_value_two_sided": p_two,
    }


def _regression_problem(line: np.ndarray, gap: np.ndarray, n_clusters: int) -> str | None:
    """Why home_margin ~ 1 + line + (pred - line) cannot be estimated, or None if it can."""
    n = len(gap)
    if n_clusters < 2 or n <= 3:
        return (
            f"not estimable: {n} games in {n_clusters} week(s); week-clustered errors need "
            "at least 2 weeks"
        )
    if np.allclose(gap, gap[0]):
        return (
            "not identified: pred - line is constant (the prediction is the line plus a "
            "constant), so it carries no information beyond the line"
        )
    if np.ptp(line) == 0:
        return "not identified: the line is constant"
    Z = np.column_stack([np.ones(n), line])
    coef, *_ = np.linalg.lstsq(Z, gap, rcond=None)
    resid = gap - Z @ coef
    if np.sqrt(np.mean(resid**2)) <= 1e-8 * max(1.0, float(np.sqrt(np.mean(gap**2)))):
        return (
            "not identified: pred - line is an exact linear function of the line (for example "
            "a constant prediction), so the model cannot be separated from the line"
        )
    return None


def information_regression(df: pd.DataFrame, pred_col: str = "pred_home_margin") -> dict:
    """OLS: home_margin = a + b * line + c * (pred - line), week-clustered standard errors.

    c is what the model knows that the line does not. If its interval covers 0, the model
    adds nothing to the market. When the design is rank deficient (pred - line constant, as
    for the line itself, or collinear with the line, as for a constant prediction) or there
    are fewer than 2 weeks, the coefficients are NaN and ``note`` says why; otherwise ``note``
    is None.
    """
    d = df.dropna(subset=[pred_col]).reset_index(drop=True)
    gap = (d[pred_col] - d["spread_close"]).to_numpy(float)
    clusters = d.groupby(BLOCK).indices.values()
    note = _regression_problem(d["spread_close"].to_numpy(float), gap, len(clusters))
    if note:
        return {
            "line_coef": math.nan,
            "model_gap_coef": math.nan,
            "model_gap_se": math.nan,
            "model_gap_ci95": (math.nan, math.nan),
            "model_gap_p_value": math.nan,
            "clusters": len(clusters),
            "note": note,
        }
    X = np.column_stack([np.ones(len(d)), d["spread_close"].to_numpy(float), gap])
    y = d["home_margin"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    xtx_inv = np.linalg.inv(X.T @ X)
    meat = np.zeros((3, 3))
    for idx in clusters:
        s = X[idx].T @ resid[idx]
        meat += np.outer(s, s)
    g = len(clusters)
    meat *= g / (g - 1) * (len(d) - 1) / (len(d) - 3)  # small-sample correction
    se = np.sqrt(np.diag(xtx_inv @ meat @ xtx_inv))
    t = beta[2] / se[2]
    return {
        "line_coef": float(beta[1]),
        "model_gap_coef": float(beta[2]),
        "model_gap_se": float(se[2]),
        "model_gap_ci95": (float(beta[2] - 1.96 * se[2]), float(beta[2] + 1.96 * se[2])),
        "model_gap_p_value": float(2 * stats.t.sf(abs(t), df=g - 1)),
        "clusters": g,
        "note": None,
    }


def win_prob_skill(
    df: pd.DataFrame,
    prob_col: str = "pred_home_win_prob",
    market_col: str = "market_home_win_prob",
    n_boot: int = 2000,
    seed: int = 0,
) -> dict | None:
    """Log-loss and Brier of the model vs the de-vigged closing moneyline (ties excluded)."""
    if prob_col not in df or market_col not in df:
        return None
    d = df.dropna(subset=[prob_col, market_col])
    d = d[d["home_margin"] != 0]
    if d.empty:
        return None
    y = (d["home_margin"] > 0).astype(int).to_numpy()
    pm, pk = d[prob_col].to_numpy(float), d[market_col].to_numpy(float)
    eps = 1e-12
    ll = lambda p: -(
        y * np.log(np.clip(p, eps, 1 - eps)) + (1 - y) * np.log(np.clip(1 - p, eps, 1 - eps))
    )  # noqa: E731
    diff = ll(pm) - ll(pk)  # per-game log-loss difference (negative = model better)
    blocks = d.assign(_diff=diff).groupby(BLOCK)["_diff"].agg(["sum", "count"]).to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(blocks), size=(n_boot, len(blocks)))
    boot = blocks[draws, 0].sum(axis=1) / blocks[draws, 1].sum(axis=1)
    return {
        "games": len(d),
        "model_log_loss": log_loss(y, pm),
        "market_log_loss": log_loss(y, pk),
        "model_brier": brier_score(y, pm),
        "market_brier": brier_score(y, pk),
        "log_loss_diff_ci95": (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))),
    }


def closing_line_value(
    bets: pd.DataFrame, line_at_bet_col: str = "spread_at_bet"
) -> pd.Series | None:
    """Points of closing-line value per bet (positive = beat the close).

    Lines are nflverse (expected home margin): backing home at 3 when it closes at 4 gains a point.
    Real example, 2025_15_LAC_KC: KC (home) was 4.5 on 2025-12-07 and closed 5.5, so KC backed
    at 4.5 gained +1 and LAC backed at 4.5 lost 1; 2025_15_BUF_NE moved from 1.5 (NE favored)
    to -2.5 (BUF favored), so BUF (away) backed at 1.5 gained +4.
    """
    if line_at_bet_col not in bets:
        return None
    gained = bets["spread_close"] - bets[line_at_bet_col]
    return pd.Series(
        np.where(bets["side"] == "home", gained, -gained), index=bets.index, name="clv_points"
    )


def bets_at_prediction(
    df: pd.DataFrame,
    pred_col: str = "pred_home_margin",
    line_col: str = "spread_at_prediction",
    min_edge: float = 1.0,
) -> pd.DataFrame:
    """The bets a model would have placed when it predicted: |pred - line then| >= min_edge.

    The side comes from the edge against the line available at prediction time, not the close:
    choosing the side with the closing line would use information from after the prediction.
    """
    d = df.dropna(subset=[pred_col, line_col])
    edge = model_edge(d[pred_col], d[line_col])
    mask = edge.abs().ge(min_edge) & edge.ne(0)
    bets = d.loc[mask].copy()
    bets["edge_at_prediction"] = edge[mask]
    bets["side"] = np.where(bets["edge_at_prediction"] > 0, "home", "away")
    return bets


def clv_summary(
    bets: pd.DataFrame,
    line_at_bet_col: str = "spread_at_prediction",
    n_boot: int = 2000,
    seed: int = 0,
) -> dict | None:
    """Mean closing-line value per bet (points) with a week-block bootstrap interval."""
    clv = closing_line_value(bets, line_at_bet_col)
    if clv is None:
        return None
    b = bets.assign(_clv=clv).dropna(subset=["_clv"])
    if b.empty:
        return {
            "bets": 0,
            "mean_points": math.nan,
            "mean_points_ci95": (math.nan, math.nan),
            "share_positive": math.nan,
            "share_negative": math.nan,
        }
    return {
        "bets": len(b),
        "mean_points": float(b["_clv"].mean()),
        "mean_points_ci95": _block_bootstrap_mean(b, "_clv", n_boot=n_boot, seed=seed),
        "share_positive": float((b["_clv"] > 0).mean()),
        "share_negative": float((b["_clv"] < 0).mean()),
    }
