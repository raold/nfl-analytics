#!/usr/bin/env python3
"""
Regrade the Oct-2025 walk-forward ATS harness (2004-2024, 5,529 games).

Inputs (read-only):
  analysis/results/multimodel_predictions.csv  home-cover probabilities per game: glm, xgb, state
                                               and the equal-weight / stacked ensembles
  analysis/results/glm_baseline_preds.csv      the same GLM run, with spread_close and labels
  devdb01 games                                scores and closing spreads (read-only session)

What it checks and recomputes:
  1. Labels: `actual` against home_cover(margin, spread_close) from the DB.
  2. The `state` column came from StateSpaceRatings.predict_prob_ats, which returned
     Phi((mu + spread_close) / sd). Home covers when margin - spread_close > 0, so the
     probability is Phi((mu - spread_close) / sd). The state-space model is refit exactly as
     nfl_analytics/backtest/harness_multimodel.py did (one fit per test season on every earlier
     season from 2003, pushes excluded, ratings frozen during the test season). The old formula
     must reproduce the published column before the fixed one is trusted.
  3. The six ensembles that include state are rebuilt (equal-weight means, and logistic stacks
     refit walk-forward by season as in add_stack_ensembles), after checking that the same
     procedure reproduces the published ensemble columns.
  4. Every column is graded the same way:
       side = home if p > 0.5, away if p < 0.5 (no bet at exactly 0.5), so every game is a bet;
       ATS win rate with a Wilson 95% interval; ROI at -110 (+100/110 per win, -1 per loss)
       with a season-block bootstrap 95% interval; Brier vs a constant 0.5 and vs the
       full-sample base rate; log-loss vs ln 2.
  5. The dissertation's printed tables are reproduced with their own rules, then restated
     with the correct payoff and the sign-fixed state column.

Outputs: docs/investigations/2026-10-final-evaluation/results/walkforward_2004_2024_*.csv

Usage:
  uv run --no-sync python scripts/final_evaluation/regrade_walkforward_2004_2024.py
  (DB connection from the DB_* environment variables; see nfl_analytics/utils/db_config.py)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from scipy import stats
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from nfl_analytics.models.state_space import StateSpaceRatings
from nfl_analytics.utils.db_config import get_db_config
from nfl_analytics.utils.spread_convention import home_cover, home_cover_margin

REPO = Path(__file__).resolve().parents[2]
MM_CSV = REPO / "analysis/results/multimodel_predictions.csv"
GLM_CSV = REPO / "analysis/results/glm_baseline_preds.csv"
OUT_DIR = REPO / "docs/investigations/2026-10-final-evaluation/results"
PREFIX = "walkforward_2004_2024"

WIN_PROFIT = 100 / 110  # net profit on a winning 1-unit bet at -110
BREAKEVEN = 110 / 210  # 52.38%
SEED = 20261024
N_BOOT = 10_000
FIRST_TRAIN_SEASON = 2003  # the harness ran with --seasons 2003-2024
TEST_SEASONS = list(range(2004, 2025))
BASE = ["glm", "xgb", "state"]
STATE_Q, STATE_R = 3.0, 13.5  # harness StateSpaceModel defaults

GAMES_SQL = """
SELECT game_id, season, week, game_type, home_team, away_team, home_score, away_score,
       spread_close
FROM games
WHERE season BETWEEN %(lo)s AND %(hi)s
  AND home_score IS NOT NULL AND away_score IS NOT NULL AND spread_close IS NOT NULL
"""


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------


def load_games() -> pd.DataFrame:
    """Completed games 2003-2024 from devdb01 in a read-only session."""
    cfg = get_db_config()
    with psycopg.connect(**cfg, options="-c default_transaction_read_only=on") as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute(GAMES_SQL, {"lo": FIRST_TRAIN_SEASON, "hi": max(TEST_SEASONS)})
            cols = [c.name for c in cur.description]
            games = pd.DataFrame(cur.fetchall(), columns=cols)
    for c in ("home_score", "away_score", "spread_close"):
        games[c] = games[c].astype(float)
    games["margin"] = games["home_score"] - games["away_score"]
    games["cover_margin"] = home_cover_margin(games["margin"], games["spread_close"])
    games["truth"] = home_cover(games["margin"], games["spread_close"])
    return games


# ----------------------------------------------------------------------------
# 1. Label check
# ----------------------------------------------------------------------------


def label_check(mm: pd.DataFrame, glm: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    g = games.set_index("game_id")
    m = mm.join(g[["margin", "spread_close", "truth", "game_type"]], on="game_id")
    matched = m["margin"].notna()
    push = matched & (m["margin"] - m["spread_close"] == 0)
    ok = matched & ~push
    old_rule = (m["margin"] + m["spread_close"] > 0).astype(int)  # sign-ok: reproduces the retracted rule on purpose
    straight_up = (m["margin"] > 0).astype(int)

    gl = glm.join(g[["spread_close", "truth"]].add_suffix("_db"), on="game_id")
    gl["truth"] = gl.pop("truth_db")
    gl_ok = gl["truth"].notna() & gl["home_cover"].notna()
    glm_prob = mm.set_index("game_id")["glm"].reindex(gl["game_id"]).to_numpy()

    rows = [
        ("rows in multimodel_predictions.csv", len(mm), ""),
        ("rows matched to devdb01 games", int(matched.sum()), ""),
        ("pushes among those rows (margin - spread_close == 0)", int(push.sum()), ""),
        (
            "agreement of `actual` with margin - spread_close > 0",
            float((m.loc[ok, "actual"] == m.loc[ok, "truth"]).mean()),
            f"n={int(ok.sum())}",
        ),
        (
            "agreement of `actual` with the old rule margin + spread_close > 0",  # sign-ok: reproduces the retracted rule on purpose
            float((m.loc[ok, "actual"] == old_rule[ok]).mean()),
            f"n={int(ok.sum())}",
        ),
        (
            "agreement of `actual` with home straight-up win",
            float((m.loc[ok, "actual"] == straight_up[ok]).mean()),
            f"n={int(ok.sum())}",
        ),
        ("pooled home cover rate (actual)", float(m["actual"].mean()), f"n={len(m)}"),
        (
            "playoff games included (current games.game_type != REG)",
            int((m["game_type"] != "REG").sum()),
            "",
        ),
        (
            "seasons",
            f"{int(m['season'].min())}-{int(m['season'].max())}",
            f"{m['season'].nunique()}",
        ),
        ("rows in glm_baseline_preds.csv", len(glm), ""),
        (
            "glm_baseline_preds spread_close vs devdb01 (max abs diff)",
            float((gl["spread_close"] - gl["spread_close_db"]).abs().max()),
            "",
        ),
        (
            "glm_baseline_preds home_cover agreement with margin - spread_close > 0",
            float((gl.loc[gl_ok, "home_cover"] == gl.loc[gl_ok, "truth"]).mean()),
            f"n={int(gl_ok.sum())}",
        ),
        (
            "glm_baseline_preds pred_cover_prob vs multimodel glm (max abs diff)",
            float(np.nanmax(np.abs(gl["pred_cover_prob"].to_numpy() - glm_prob))),
            "same GLM run",
        ),
    ]
    return pd.DataFrame(rows, columns=["check", "value", "note"])


# ----------------------------------------------------------------------------
# 2. State-space refit
# ----------------------------------------------------------------------------


def refit_state_space(games: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward state-space predictions, as harness_multimodel.walk_forward_predictions."""
    pool = games[games["cover_margin"] != 0].sort_values(["season", "week", "game_id"])
    cols = ["season", "week", "home_team", "away_team", "margin"]
    rows = []
    for season in TEST_SEASONS:
        model = StateSpaceRatings(q=STATE_Q, r=STATE_R)
        with contextlib.redirect_stdout(io.StringIO()):  # fit() prints progress
            model.fit(pool.loc[pool["season"] < season, cols].copy())
        test = pool[pool["season"] == season]
        for gid, home, away, line in zip(
            test["game_id"], test["home_team"], test["away_team"], test["spread_close"]
        ):
            mu, sd = model.predict_margin(home, away)
            rows.append((gid, mu, sd, line))
    out = pd.DataFrame(rows, columns=["game_id", "state_mu", "state_sd", "spread_close"])
    out["state_old_formula"] = norm.cdf((out["state_mu"] + out["spread_close"]) / out["state_sd"])
    out["state_fixed"] = norm.cdf((out["state_mu"] - out["spread_close"]) / out["state_sd"])
    return out


# ----------------------------------------------------------------------------
# 3. Ensembles (same procedure as harness_multimodel.add_*_ensembles)
# ----------------------------------------------------------------------------


def combos(items):
    for r in range(2, len(items) + 1):
        yield from itertools.combinations(items, r)


def stack_walk_forward(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Logistic stack fit on all earlier test seasons; the first season falls back to the mean."""
    seasons = sorted(df["season"].unique())
    out = np.full(len(df), np.nan)
    for i in range(1, len(seasons)):
        train = df["season"].isin(seasons[:i]).to_numpy()
        test = (df["season"] == seasons[i]).to_numpy()
        model = LogisticRegression(max_iter=2000, solver="lbfgs")
        model.fit(df.loc[train, cols], df.loc[train, "actual"])
        out[test] = model.predict_proba(df.loc[test, cols])[:, 1]
    mean = df[cols].mean(axis=1).to_numpy()
    return np.where(np.isnan(out), mean, out)


def build_ensembles(df: pd.DataFrame, base: dict[str, str]) -> pd.DataFrame:
    """Equal-weight and stacked ensembles for every subset of the base columns.

    ``base`` maps the harness name (glm/xgb/state) to the column holding it in ``df``.
    """
    out = pd.DataFrame(index=df.index)
    for combo in combos(list(base)):
        tag = "_".join(combo)
        frame = df[["season", "actual"]].copy()
        for name in combo:
            frame[name] = df[base[name]]
        out[f"ens_mean_{tag}"] = frame[list(combo)].mean(axis=1)
        out[f"ens_stack_{tag}"] = stack_walk_forward(frame, list(combo))
    return out


# ----------------------------------------------------------------------------
# 4. Grading
# ----------------------------------------------------------------------------


def wilson(wins: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = wins / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (centre - half, centre + half)


def season_block_ci(num: np.ndarray, den: np.ndarray, season: np.ndarray) -> tuple[float, float]:
    """95% percentile interval of sum(num)/sum(den), resampling whole seasons."""
    keys = np.unique(season)
    num_s = np.array([num[season == s].sum() for s in keys])
    den_s = np.array([den[season == s].sum() for s in keys])
    rng = np.random.default_rng(SEED)  # same draws for every column
    idx = rng.integers(0, len(keys), size=(N_BOOT, len(keys)))
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = num_s[idx].sum(axis=1) / den_s[idx].sum(axis=1)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return (float(lo), float(hi))


def log_loss_terms(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q))


RULES = {  # name: (bet home above, bet away below)
    "all games, side = sign(p - 0.5)": (0.5, 0.5),
    "p > 0.524 or p < 0.476 (dissertation betting rule)": (0.524, 0.476),
}


def grade(p, y, season, hi: float = 0.5, lo: float = 0.5, with_ci: bool = True) -> dict:
    """ATS (home if p > hi, away if p < lo), ROI at -110, Brier and log-loss, season-block CIs."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    season = np.asarray(season)
    bet = (p > hi) | (p < lo)
    won = np.where(p > hi, y == 1, y == 0) & bet
    n = int(bet.sum())
    wins = int(won.sum())
    profit = np.where(bet, np.where(won, WIN_PROFIT, -1.0), 0.0)
    returns = pd.Series(profit[bet])
    sq = (p - y) ** 2
    ll = log_loss_terms(p, y)
    out = {
        "games": len(p),
        "bets": n,
        "wins": wins,
        "losses": n - wins,
        "home_bet_share": float((p[bet] > hi).mean()) if n else math.nan,
        "win_rate": wins / n if n else math.nan,
        "roi": profit.sum() / n if n else math.nan,
        "sharpe_x_sqrt272": (
            float(returns.mean() / returns.std() * math.sqrt(272)) if n > 1 else math.nan
        ),
        "brier": sq.mean(),
        "brier_base_rate": float(((y.mean() - y) ** 2).mean()),
        "log_loss": ll.mean(),
        "accuracy_p_ge_0_5": float(((p >= 0.5) == (y == 1)).mean()),
    }
    if not with_ci or n == 0:
        return out
    ones = np.ones_like(p)
    roi_ci = season_block_ci(profit, bet.astype(float), season)
    brier_ci = season_block_ci(sq - 0.25, ones, season)
    ll_ci = season_block_ci(ll - math.log(2), ones, season)
    out.update(
        {
            "win_rate_lo": wilson(wins, n)[0],
            "win_rate_hi": wilson(wins, n)[1],
            "p_two_sided_vs_50": float(stats.binomtest(wins, n, 0.5).pvalue),
            "p_one_sided_vs_breakeven": float(
                stats.binomtest(wins, n, BREAKEVEN, alternative="greater").pvalue
            ),
            "roi_lo": roi_ci[0],
            "roi_hi": roi_ci[1],
            "brier_minus_coin": sq.mean() - 0.25,
            "brier_minus_coin_lo": brier_ci[0],
            "brier_minus_coin_hi": brier_ci[1],
            "log_loss_minus_ln2": ll.mean() - math.log(2),
            "log_loss_minus_ln2_lo": ll_ci[0],
            "log_loss_minus_ln2_hi": ll_ci[1],
        }
    )
    return out


# ----------------------------------------------------------------------------
# 5. The dissertation's own rules
# ----------------------------------------------------------------------------


def betting_table_rule(p, y, win_profit: float, loss: float) -> dict:
    """nfl_analytics/analysis/generate_betting_metrics.py: bet home if p > 0.524, away if
    p < 0.476. The script paid +0.91 per win and -1.1 per loss (the vig twice)."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    bet = (p > 0.524) | (p < 0.476)
    won = np.where(p > 0.524, y == 1, y == 0)[bet]
    r = pd.Series(np.where(won, win_profit, loss))
    return {
        "bets": int(bet.sum()),
        "win_rate": float(won.mean()),
        "roi": float(r.mean()),
        "sharpe": float(r.mean() / r.std() * math.sqrt(272)),
    }


def harness_rule(p, y) -> dict:
    """harness_multimodel.overall_metrics: accuracy and ROI at threshold 0.55, home bets only,
    decimal payout 1.91; probabilities clipped at 1e-9 for Brier and log-loss."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    q = np.clip(p, 1e-9, 1 - 1e-9)
    bet = p >= 0.55
    wins = y[bet].sum()
    return {
        "brier": float(((q - y) ** 2).mean()),
        "log_loss": float(log_loss_terms(q, y).mean()),
        "accuracy": float(((p >= 0.55).astype(int) == y).mean()),
        "home_bets": int(bet.sum()),
        "roi": float((wins * 0.91 - (bet.sum() - wins)) / bet.sum()) if bet.sum() else 0.0,
    }


def pseudo_clv(p, y) -> dict:
    """generate_betting_metrics.calculate_clv: (p - 0.5) signed by the outcome, in bps."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    v = pd.Series(np.where(y == 1, p - 0.5, 0.5 - p) * 1e4)
    t = v.mean() / (v.std() / math.sqrt(len(v)))
    return {
        "mean_bps": float(v.mean()),
        "median_bps": float(v.median()),
        "std_bps": float(v.std()),
        "t": float(t),
        "p_two_sided": float(2 * stats.t.sf(abs(t), df=len(v) - 1)),
    }


PRINTED_BETTING = [  # analysis/dissertation/figures/out/betting_performance_table.tex
    ("GLM", "glm", 3826, 0.510, -7.45, -1.22),
    ("MEAN(GLM+XGB)", "ens_mean_glm_xgb", 3963, 0.507, -8.00, -1.31),
    ("XGB", "xgb", 4419, 0.501, -9.39, -1.54),
    ("GLM+XGB (stack)", "ens_stack_glm_xgb", 1139, 0.500, -9.41, -1.54),
    ("GLM+STATE (stack)", "ens_stack_glm_state", 1100, 0.500, -9.50, -1.56),
]
PRINTED_MULTIMODEL = [  # analysis/dissertation/figures/out/multimodel_comparison_table.tex
    ("Stack(GLM+XGB+State)", "ens_stack_glm_xgb_state", 0.2515, 0.6966, 0.511, -13.5),
    ("Stack(GLM+XGB)", "ens_stack_glm_xgb", 0.2517, 0.6973, 0.512, -11.5),
    ("Stack(GLM+State)", "ens_stack_glm_state", 0.2517, 0.6971, 0.512, -7.9),
    ("Stack(XGB+State)", "ens_stack_xgb_state", 0.2519, 0.6976, 0.510, -17.2),
    ("GLM (baseline)", "glm", 0.2552, 0.7055, 0.510, -6.3),
    ("Mean(GLM+XGB)", "ens_mean_glm_xgb", 0.2567, 0.7078, 0.513, -4.5),
    ("Mean(GLM+XGB+State)", "ens_mean_glm_xgb_state", 0.2615, 0.7176, 0.497, -8.3),
    ("XGBoost", "xgb", 0.2643, 0.7260, 0.514, -4.2),
]
PRINTED_CLV = [  # analysis/dissertation/figures/out/clv_distribution_table.tex
    ("GLM", "glm", 14.9, 8.1, 820.6),
    ("MEAN(GLM+XGB)", "ens_mean_glm_xgb", 8.7, -9.1, 867.7),
    ("XGB", "xgb", 2.5, 2.5, 1208.4),
    ("GLM+XGB (stack)", "ens_stack_glm_xgb", 4.3, 20.9, 456.5),
    ("GLM+STATE (stack)", "ens_stack_glm_state", 3.6, 27.4, 453.9),
]
# analysis/dissertation/figures/out/per_season_top3_table.tex (oos_record_table.tex prints the
# same Brier, accuracy and ROI): season, column, Brier, log-loss, accuracy %, ROI %
PRINTED_PER_SEASON = {
    "glm": [
        (2015, 0.2539, 0.7011, 53.3, -4.5),
        (2016, 0.2459, 0.6844, 54.2, 29.4),
        (2017, 0.2530, 0.6983, 50.2, 3.5),
        (2018, 0.2552, 0.7037, 50.4, -22.4),
        (2019, 0.2508, 0.6948, 53.7, -25.0),
        (2020, 0.2554, 0.7067, 50.9, 0.0),
        (2021, 0.2502, 0.6937, 49.8, -22.2),
        (2022, 0.2537, 0.7005, 50.4, -7.6),
        (2023, 0.2539, 0.7010, 49.1, -19.6),
        (2024, 0.2546, 0.7023, 48.0, -4.5),
    ],
    "xgb": [
        (2015, 0.2639, 0.7221, 51.0, -14.7),
        (2016, 0.2550, 0.7050, 51.5, 3.5),
        (2017, 0.2585, 0.7132, 53.7, 14.6),
        (2018, 0.2586, 0.7123, 51.6, -9.4),
        (2019, 0.2534, 0.7006, 54.5, -14.1),
        (2020, 0.2636, 0.7214, 49.4, -11.5),
        (2021, 0.2529, 0.6999, 55.9, 20.4),
        (2022, 0.2546, 0.7029, 52.2, 4.6),
        (2023, 0.2537, 0.7010, 50.6, -2.4),
        (2024, 0.2602, 0.7141, 49.8, 5.2),
    ],
    "ens_stack_glm_xgb_state": [
        (2015, 0.2491, 0.6914, 53.3, 0.0),
        (2016, 0.2498, 0.6927, 50.0, 0.0),
        (2017, 0.2503, 0.6937, 48.6, 0.0),
        (2018, 0.2498, 0.6928, 52.7, 0.0),
        (2019, 0.2486, 0.6904, 56.0, 0.0),
        (2020, 0.2504, 0.6939, 50.6, 0.0),
        (2021, 0.2499, 0.6929, 51.6, 0.0),
        (2022, 0.2503, 0.6938, 50.7, 0.0),
        (2023, 0.2505, 0.6941, 50.2, 0.0),
        (2024, 0.2511, 0.6953, 48.0, 0.0),
    ],
}


def fixed_name(col: str) -> str:
    return f"{col}_fixed" if "state" in col else col


def dissertation_check(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = d["actual"]

    def add(table, label, col, metric, printed, reproduced, corrected, note="", tol=None):
        if tol is None:  # half a unit in the last printed digit
            tol = {"bets": 0.5, "win_rate": 5e-4, "accuracy_at_0.55": 5e-4, "sharpe": 5e-3}.get(
                metric, 5e-5 if metric in ("brier", "log_loss") else 0.05
            )
        rows.append(
            {
                "table": table,
                "row": label,
                "column_graded": col,
                "metric": metric,
                "printed": printed,
                "reproduced_from_csv": reproduced,
                "matches_printed": bool(abs(printed - reproduced) <= tol + 1e-9),
                "corrected": corrected,
                "note": note,
            }
        )

    t = "betting_performance_table.tex"
    for label, col, n, win, roi, sharpe in PRINTED_BETTING:
        script = betting_table_rule(d[col], y, 0.91, -1.1)
        right = betting_table_rule(d[fixed_name(col)], y, WIN_PROFIT, -1.0)
        note = "state column sign-fixed and stack refit" if "state" in col else "payoff fixed"
        add(t, label, col, "bets", n, script["bets"], right["bets"], note)
        add(t, label, col, "win_rate", win, script["win_rate"], right["win_rate"], note)
        add(t, label, col, "roi_pct", roi, 100 * script["roi"], 100 * right["roi"], note, 5e-3)
        add(t, label, col, "sharpe", sharpe, script["sharpe"], right["sharpe"], note)

    t = "multimodel_comparison_table.tex"
    for label, col, brier, ll, acc, roi in PRINTED_MULTIMODEL:
        pub = harness_rule(d[col], y)
        fix = harness_rule(d[fixed_name(col)], y)
        note = "state column sign-fixed and stack refit" if "state" in col else "unchanged"
        add(t, label, col, "brier", brier, pub["brier"], fix["brier"], note)
        add(t, label, col, "log_loss", ll, pub["log_loss"], fix["log_loss"], note)
        add(t, label, col, "accuracy_at_0.55", acc, pub["accuracy"], fix["accuracy"], note)
        add(
            t,
            label,
            col,
            "roi_pct_home_bets_p>=0.55",
            roi,
            100 * pub["roi"],
            100 * fix["roi"],
            f"{note}; home bets {pub['home_bets']} -> {fix['home_bets']}",
        )

    t = "clv_distribution_table.tex"
    for label, col, mean, median, std in PRINTED_CLV:
        pub = pseudo_clv(d[col], y)
        fix = pseudo_clv(d[fixed_name(col)], y)
        note = (
            "outcome-signed (p - 0.5), not closing-line value; "
            f"t={pub['t']:.2f}, p={pub['p_two_sided']:.2f} as published"
        )
        add(t, label, col, "mean_bps", mean, pub["mean_bps"], fix["mean_bps"], note)
        add(t, label, col, "median_bps", median, pub["median_bps"], fix["median_bps"], note)
        add(t, label, col, "std_bps", std, pub["std_bps"], fix["std_bps"], note)

    t = "per_season_top3_table.tex / oos_record_table.tex"
    for col, printed in PRINTED_PER_SEASON.items():
        for season, brier, ll, acc, roi in printed:
            s = d[d["season"] == season]
            pub = harness_rule(s[col], s["actual"])
            fix = harness_rule(s[fixed_name(col)], s["actual"])
            note = f"home bets at p>=0.55: {pub['home_bets']} -> {fix['home_bets']}"
            if "state" in col:
                note += "; state column sign-fixed and stack refit"
            label = f"{season} {col}"
            add(t, label, col, "brier", brier, pub["brier"], fix["brier"], note)
            add(t, label, col, "log_loss", ll, pub["log_loss"], fix["log_loss"], note)
            add(
                t,
                label,
                col,
                "accuracy_pct",
                acc,
                100 * pub["accuracy"],
                100 * fix["accuracy"],
                note,
            )
            add(t, label, col, "roi_pct", roi, 100 * pub["roi"], 100 * fix["roi"], note)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mm = pd.read_csv(MM_CSV)
    glm = pd.read_csv(GLM_CSV)
    games = load_games()
    print(f"multimodel_predictions.csv: {len(mm)} rows; devdb01 games 2003-2024: {len(games)}")

    labels = label_check(mm, glm, games)
    labels.to_csv(args.out_dir / f"{PREFIX}_label_check.csv", index=False)
    print("\n== Label check")
    print(labels.to_string(index=False))

    # State-space refit and reproduction of the published column
    ss = refit_state_space(games)
    d = mm.merge(ss[["game_id", "state_mu", "state_sd", "state_old_formula", "state_fixed"]])
    if len(d) != len(mm):
        raise RuntimeError(f"state-space refit matched {len(d)} of {len(mm)} games")
    repro = [
        (
            "state",
            "max abs diff: refit + old formula Phi((mu + spread)/sd) vs published",
            float((d["state"] - d["state_old_formula"]).abs().max()),
        )
    ]
    line = d[["game_id"]].merge(games[["game_id", "spread_close"]])["spread_close"].to_numpy()
    for col in (
        "state",
        "state_fixed",
        "glm",
        "xgb",
    ):  # a cover probability should not track the line
        repro.append((col, "corr(p, spread_close)", np.corrcoef(d[col], line)[0, 1]))

    # Ensembles: the procedure must reproduce the published columns from the published inputs
    rebuilt = build_ensembles(d, {"glm": "glm", "xgb": "xgb", "state": "state"})
    for col in rebuilt.columns:
        repro.append(
            (
                col,
                "max abs diff: rebuilt from published base columns",
                float((rebuilt[col] - d[col]).abs().max()),
            )
        )
    fixed = build_ensembles(d, {"glm": "glm", "xgb": "xgb", "state": "state_fixed"})
    for col in fixed.columns:
        if "state" in col:
            d[f"{col}_fixed"] = fixed[col]
    repro = pd.DataFrame(repro, columns=["column", "check", "value"])
    repro.to_csv(args.out_dir / f"{PREFIX}_reproduction.csv", index=False)
    print("\n== Reproduction of published columns")
    print(repro.to_string(index=False))

    # Label-only references: the home cover rate is below 0.5, so these always back the road team
    pool = games[games["cover_margin"] != 0]
    prior_rate = {s: float(pool.loc[pool["season"] < s, "truth"].mean()) for s in TEST_SEASONS}
    d["ref_base_rate_walk_forward"] = d["season"].map(prior_rate)
    d["ref_base_rate_in_sample"] = d["actual"].mean()

    published = BASE + [c for c in mm.columns if c.startswith("ens_")]
    corrected = [f"{c}_fixed" for c in published if "state" in c]
    references = ["ref_base_rate_walk_forward", "ref_base_rate_in_sample"]
    graded = published + corrected + references
    reg = d[["game_id"]].merge(games[["game_id", "game_type"]])["game_type"].eq("REG").to_numpy()
    subsets = [  # (rule, games): the second rule and the regular-season cut are robustness checks
        (rule, "all games", np.ones(len(d), dtype=bool)) for rule in RULES
    ] + [(next(iter(RULES)), "regular season only", reg)]
    metrics = []
    for rule, subset, mask in subsets:
        hi, lo = RULES[rule]
        for col in graded:
            if col.startswith("ref_"):
                status = "reference (labels only)"
            elif col.endswith("_fixed"):
                status = "corrected (state sign fixed)"
            elif "state" in col:
                status = "published, invalid (state sign bug)"
            else:
                status = "published, valid"
            s = d[mask]
            g = grade(s[col], s["actual"], s["season"], hi=hi, lo=lo)
            metrics.append(
                {"rule": rule, "games_subset": subset, "column": col, "status": status, **g}
            )
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(args.out_dir / f"{PREFIX}_metrics.csv", index=False)
    show = [
        "column",
        "bets",
        "home_bet_share",
        "win_rate",
        "win_rate_lo",
        "win_rate_hi",
        "roi",
        "roi_lo",
        "roi_hi",
        "brier",
        "log_loss",
    ]
    for rule, subset, _ in subsets:
        print(f"\n== {rule}, {subset}; ROI at -110 with season-block bootstrap CI")
        sel = (metrics["rule"] == rule) & (metrics["games_subset"] == subset)
        print(metrics.loc[sel, show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    per_season = [
        {"season": season, "column": col, **grade(grp[col], grp["actual"], season, with_ci=False)}
        for season, grp in d.groupby("season")
        for col in graded
    ]
    pd.DataFrame(per_season).to_csv(args.out_dir / f"{PREFIX}_per_season.csv", index=False)

    diss = dissertation_check(d)
    diss.to_csv(args.out_dir / f"{PREFIX}_dissertation_check.csv", index=False)
    print("\n== Dissertation tables: printed values reproduced from the published CSV")
    print(diss.groupby("table")["matches_printed"].agg(["size", "sum"]).to_string())
    misses = diss[~diss["matches_printed"]]
    if len(misses):
        print(misses.drop(columns=["note"]).to_string(index=False))
    main_rows = diss[~diss["table"].str.startswith("per_season")].drop(columns=["note"])
    print("\n== Printed vs corrected (season tables in the CSV)")
    print(main_rows.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    keep = ["game_id", "season", "week", "actual", "state_mu", "state_sd", "state"]
    keep += ["state_fixed"] + [c for c in corrected if c != "state_fixed"]
    pred = d[keep].rename(columns={"state": "state_published"})
    pred.to_csv(
        args.out_dir / f"{PREFIX}_predictions_corrected.csv", index=False, float_format="%.6f"
    )
    print(f"\nWrote {PREFIX}_*.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
