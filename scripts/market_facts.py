#!/usr/bin/env python3
"""
Market facts from public nflverse schedules: how NFL closing lines have behaved in 1999-2025,
with the current season's completed games as an out-of-sample check.

Every number is a historical association measured after the fact against closing lines. None
of them is a betting edge: a strategy needs the line, the price and the inputs that were
available when the bet was placed, a rule fixed in advance, and a prospective sample (see the
README's "What a prospective test would need").

Usage:
    uv run --no-sync python scripts/market_facts.py --games <nflverse games.csv> \\
        --out docs/investigations/2026-10-market-facts/results

Data: https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv
Sign convention: spread_line is the expected HOME margin (positive = home favoured), so the home
team covers when home_margin - spread_line > 0 (nfl_analytics/utils/spread_convention.py).
Pushes are excluded from every rate. Seasons before --holdout-season form the historical window;
the holdout season contributes its completed games only.

Outputs (in --out): one CSV per table, tests_holm.csv (every p-value in the test family with its
Holm adjustment and a week-clustered sensitivity check), prospective_power.csv, provenance.csv
and summary.md (the printed report).
"""

from __future__ import annotations

import argparse
import hashlib
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Only the leaf module spread_convention is imported, so the script keeps working while the
# rest of the package is refactored. The odds and Wilson formulas below mirror
# nfl_analytics/evaluation/market_benchmark/odds.py and metrics.py.
from nfl_analytics.utils.spread_convention import (
    assert_nflverse_convention,
    book_home_line,
    home_cover,
    home_cover_margin,
)

SOURCE_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
REQUIRED_COLUMNS = (
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "location",
    "home_score",
    "away_score",
    "result",
    "total",
    "spread_line",
    "total_line",
    "home_moneyline",
    "away_moneyline",
    "home_spread_odds",
    "away_spread_odds",
    "over_odds",
    "under_odds",
    "roof",
    "wind",
    "stadium",
)
# Sign anchor from the project brief: BAL (home) -7.5 against BUF, so spread_line = +7.5
ANCHOR_GAME, ANCHOR_SPREAD = "2018_01_BUF_BAL", 7.5

ERA_STARTS = (1999, 2007, 2014, 2020)  # blocks of 6-8 seasons; the last runs to the holdout
HFA_STARTS = (1999, 2007, 2014, 2020, 2021)  # the same eras with the 2020 no-crowd season apart
PAT_RULE_CHANGE = 2015  # first season with extra points snapped from the 15-yard line
KEY_MARGINS = (3, 7, 6, 10, 1, 2, 8)

OUTDOOR_ROOFS = ("outdoors", "open")
WIND_THRESHOLD = 15.0  # mph; a rule of thumb that the survey picked after seeing the data
WIND_EDGES = (0.0, 6.0, 11.0, 15.0, 20.0, np.inf)
WIND_LABELS = ("0-5", "6-10", "11-14", "15-19", "20+")
WIND_IMPLAUSIBLE = 40.0  # sensitivity check: three games list 44, 70 and 71 mph

TEASER_POINTS = 6.0
WONG_FAVOURITE = (-8.5, -7.5)  # sportsbook line of the leg; teased through 7 and 3
WONG_UNDERDOG = (1.5, 2.5)
TEASER_PRICES = (-110, -120, -130)  # price of a two-team, six-point teaser

PROB_EDGES = tuple(np.round(np.linspace(0, 1, 11), 1))  # reliability bins of width 0.1
ROI_EDGES = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)  # side's de-vigged win probability

STANDARD_PRICE = -110  # price assumed when none is listed
ALPHA, POWER = 0.05, 0.80


# ---------------------------------------------------------------------------------------------
# Odds and statistics helpers
# ---------------------------------------------------------------------------------------------


def american_to_decimal(odds: float) -> float:
    """Decimal odds, stake included: -110 -> 1.909, +130 -> 2.30."""
    if odds == 0 or not math.isfinite(odds):
        raise ValueError(f"invalid American odds: {odds}")
    return 1 + (odds / 100 if odds > 0 else 100 / -odds)


def american_to_implied(odds: float) -> float:
    """Raw implied probability, bookmaker margin included."""
    return 1 / american_to_decimal(odds)


def profit_per_unit(odds: float) -> float:
    """Net profit on a winning 1-unit stake."""
    return american_to_decimal(odds) - 1


def breakeven_win_rate(odds: float) -> float:
    """Win rate needed to break even at these odds (-110 -> 52.38%)."""
    return american_to_implied(odds)


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (math.nan, math.nan)
    p = wins / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (centre - half, centre + half)


def proportion(k: int, n: int) -> tuple[float, float, float]:
    """Rate with a Wilson 95% interval (NaN when n = 0)."""
    lo, hi = wilson_interval(int(k), int(n))
    return (k / n if n else math.nan, lo, hi)


def mean_ci(values) -> tuple[float, float, float]:
    """Mean with a t-based 95% interval."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (math.nan, math.nan, math.nan)
    m = float(x.mean())
    if len(x) < 2:
        return (m, math.nan, math.nan)
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return (m, m - half, m + half)


def welch_ci(x, y) -> tuple[float, float, float]:
    """Difference of means x - y with a Welch 95% interval."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return (math.nan, math.nan, math.nan)
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    dof = (vx + vy) ** 2 / (vx**2 / (len(x) - 1) + vy**2 / (len(y) - 1))
    d = float(x.mean() - y.mean())
    half = stats.t.ppf(0.975, dof) * math.sqrt(vx + vy)
    return (d, d - half, d + half)


def newcombe_ci(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """Difference of proportions p1 - p2 with Newcombe's hybrid score interval."""
    if n1 == 0 or n2 == 0:
        return (math.nan, math.nan, math.nan)
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson_interval(k1, n1)
    l2, u2 = wilson_interval(k2, n2)
    d = p1 - p2
    return (d, d - math.hypot(p1 - l1, u2 - p2), d + math.hypot(u1 - p1, p2 - l2))


def week_blocks(frame: pd.DataFrame) -> np.ndarray:
    """One block per (season, week): games in the same week share weather, news and markets."""
    return frame["season"].to_numpy(dtype=np.int64) * 100 + frame["week"].to_numpy(dtype=np.int64)


def clustered_rate_se(successes, clusters) -> float:
    """Week-clustered (CR1) standard error of a rate: checks the binomial tests' independence."""
    y = np.asarray(successes, dtype=float)
    codes, uniques = pd.factorize(np.asarray(clusters))
    g = len(uniques)
    if len(y) < 2 or g < 2:
        return math.nan
    sums = np.bincount(codes, weights=y - y.mean(), minlength=g)
    return math.sqrt((sums**2).sum() * g / (g - 1)) / len(y)


def block_bootstrap_means(values, blocks, n_boot: int, seed: int):
    """95% percentile intervals for the column means of `values`, resampling whole weeks."""
    v = np.asarray(values, dtype=float)
    if v.ndim == 1:
        v = v[:, None]
    if len(v) == 0:
        return np.full(v.shape[1], math.nan), np.full(v.shape[1], math.nan)
    codes, uniques = pd.factorize(np.asarray(blocks))
    sums = np.zeros((len(uniques), v.shape[1]))
    np.add.at(sums, codes, v)
    counts = np.bincount(codes, minlength=len(uniques)).astype(float)
    draws = np.random.default_rng(seed).integers(0, len(uniques), size=(n_boot, len(uniques)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)[:, None]
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return lo, hi


def clustered_ols(y, x, clusters) -> dict:
    """OLS y = a + b x with week-clustered (CR1) standard errors; t tests on G - 1 df."""
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    codes, uniques = pd.factorize(np.asarray(clusters))
    n, g = len(y), len(uniques)
    nan = {"n": n, "clusters": g, "intercept": math.nan, "slope": math.nan}
    nan.update({"slope_se": math.nan, "slope_ci_low": math.nan, "slope_ci_high": math.nan})
    nan["slope_p_two_sided"] = math.nan
    if n < 3 or g < 2 or np.ptp(x) == 0:
        return nan
    design = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    bread = np.linalg.inv(design.T @ design)
    scores = np.zeros((g, 2))
    np.add.at(scores, codes, design * resid[:, None])
    meat = scores.T @ scores * (g / (g - 1)) * ((n - 1) / (n - 2))
    se = math.sqrt((bread @ meat @ bread)[1, 1])
    half = stats.t.ppf(0.975, g - 1) * se
    return {
        "n": n,
        "clusters": g,
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "slope_se": se,
        "slope_ci_low": float(beta[1] - half),
        "slope_ci_high": float(beta[1] + half),
        "slope_p_two_sided": float(2 * stats.t.sf(abs(beta[1] / se), g - 1)),
    }


def logistic_recalibration(y, p, clusters) -> dict:
    """Fit logit P(y = 1) = a + b logit(p) by Newton-Raphson, week-clustered sandwich SEs.

    b = 1 and a = 0 means the probabilities are calibrated. b > 1 means the market is
    under-confident (favourites win more often than priced, the favourite-longshot pattern).
    """
    y, p = np.asarray(y, dtype=float), np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    codes, uniques = pd.factorize(np.asarray(clusters))
    n, g = len(y), len(uniques)
    out = {"n": n, "clusters": g}
    if n < 10 or g < 2:
        return out | {k: math.nan for k in ("a", "b", "b_se", "b_ci_low", "b_ci_high")}
    design = np.column_stack([np.ones(n), np.log(p / (1 - p))])
    beta = np.array([0.0, 1.0])
    for _ in range(100):
        mu = 1 / (1 + np.exp(-(design @ beta)))
        hessian = design.T @ (design * (mu * (1 - mu))[:, None])
        step = np.linalg.solve(hessian, design.T @ (y - mu))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    mu = 1 / (1 + np.exp(-(design @ beta)))
    bread = np.linalg.inv(design.T @ (design * (mu * (1 - mu))[:, None]))
    scores = np.zeros((g, 2))
    np.add.at(scores, codes, design * (y - mu)[:, None])
    se = math.sqrt((bread @ (scores.T @ scores * g / (g - 1)) @ bread)[1, 1])
    half = stats.t.ppf(0.975, g - 1) * se
    return out | {
        "a": float(beta[0]),
        "b": float(beta[1]),
        "b_se": se,
        "b_ci_low": float(beta[1] - half),
        "b_ci_high": float(beta[1] + half),
    }


def stratified_permutation_p(values, in_group, strata, n_perm: int, seed: int):
    """Two-sided permutation p-value for mean(group) - mean(rest), shuffling labels within strata.

    Group sizes are fixed within each stratum, so only the group's sum varies under the null.
    """
    v = np.asarray(values, dtype=float)
    grp = np.asarray(in_group, dtype=bool)
    strata = np.asarray(strata)
    k_total = int(grp.sum())
    if k_total == 0 or k_total == len(v):
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    group_sum = np.zeros(n_perm)
    for s in np.unique(strata):
        vs = v[strata == s]
        k = int(grp[strata == s].sum())
        if k == 0:
            continue
        if k == len(vs):
            group_sum += vs.sum()
            continue
        pick = np.argpartition(rng.random((n_perm, len(vs))), k - 1, axis=1)[:, :k]
        group_sum += vs[pick].sum(axis=1)
    rest = len(v) - k_total
    diffs = group_sum / k_total - (v.sum() - group_sum) / rest
    observed = float(v[grp].mean() - v[~grp].mean())
    p = (np.sum(np.abs(diffs) >= abs(observed) - 1e-12) + 1) / (n_perm + 1)
    return float(p), observed


def holm_adjust(p_values) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone; reject where adjusted p < alpha)."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def required_n(p0: float, p1: float, alpha: float = ALPHA, power: float = POWER) -> float:
    """Sample size for a one-sided test that a rate exceeds p0 when the true rate is p1."""
    if not p1 > p0:
        return math.inf
    za, zb = stats.norm.ppf(1 - alpha), stats.norm.ppf(power)
    root = za * math.sqrt(p0 * (1 - p0)) + zb * math.sqrt(p1 * (1 - p1))
    return float(math.ceil((root / (p1 - p0)) ** 2))


def two_team_leg_breakeven(price: float) -> float:
    """Per-leg win rate at which a two-team teaser at `price` breaks even (independent legs)."""
    return math.sqrt(breakeven_win_rate(price))


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------


def load_games(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All schedule rows, and the completed games with derived columns, after sanity checks."""
    raw = pd.read_csv(path, low_memory=False)
    missing = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    done = raw[raw["home_score"].notna() & raw["away_score"].notna()].copy()
    done["home_margin"] = done["home_score"] - done["away_score"]
    done["points"] = done["home_score"] + done["away_score"]
    if not (
        np.allclose(done["home_margin"], done["result"])
        and np.allclose(done["points"], done["total"])
    ):
        raise ValueError("nflverse result/total columns disagree with the scores")
    done["phase"] = np.where(done["game_type"] == "REG", "REG", "POST")
    assert_nflverse_convention(done.dropna(subset=["spread_line"]), spread_col="spread_line")
    anchor = done.loc[done["game_id"] == ANCHOR_GAME, "spread_line"].to_numpy()
    if len(anchor) and anchor[0] != ANCHOR_SPREAD:
        raise ValueError(f"{ANCHOR_GAME} spread_line is {anchor[0]}, expected {ANCHOR_SPREAD}")
    return raw, done


def eras(starts, last: int) -> list[tuple[str, int, int]]:
    """Consecutive blocks beginning at each start; the final block ends at `last`."""
    out = []
    for i, lo in enumerate(starts):
        if lo > last:
            break
        hi = min(starts[i + 1] - 1, last) if i + 1 < len(starts) else last
        out.append((f"{lo}-{hi}" if hi > lo else str(lo), lo, hi))
    return out


def windows(hist: pd.DataFrame, oos: pd.DataFrame, holdout: int, starts) -> list[tuple]:
    """(label, kind, frame) for each era, the pooled historical window and the holdout."""
    last = holdout - 1
    out = [
        (label, "era", hist[hist["season"].between(lo, hi)]) for label, lo, hi in eras(starts, last)
    ]
    out.append((f"{int(hist['season'].min())}-{last}", "pooled", hist))
    out.append((f"{holdout} holdout", "holdout", oos))
    return out


def provenance(path: Path, raw, done, hist, oos, holdout: int) -> pd.DataFrame:
    data = path.read_bytes()
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    scheduled = raw[raw["season"] == holdout]
    items = [
        ("source_url", SOURCE_URL),
        ("file_name", path.name),
        ("sha256", hashlib.sha256(data).hexdigest()),
        ("bytes", len(data)),
        ("file_mtime_utc", mtime.isoformat(timespec="seconds")),
        ("rows", len(raw)),
        ("seasons_in_file", f"{raw['season'].min()}-{raw['season'].max()}"),
        ("completed_games", len(done)),
        ("historical_window", f"{hist['season'].min()}-{hist['season'].max()}"),
        ("historical_completed_games", len(hist)),
        ("holdout_season", holdout),
        ("holdout_scheduled_games", len(scheduled)),
        ("holdout_completed_games", len(oos)),
        ("holdout_last_completed_week", int(oos["week"].max()) if len(oos) else ""),
        ("holdout_last_completed_gameday", oos["gameday"].max() if len(oos) else ""),
    ]
    return pd.DataFrame(items, columns=["item", "value"])


# ---------------------------------------------------------------------------------------------
# 1. Closing-spread calibration
# ---------------------------------------------------------------------------------------------


def cover_summary(frame: pd.DataFrame) -> dict:
    cover = home_cover(frame["home_margin"], frame["spread_line"])
    decided = cover.dropna()
    rate, lo, hi = proportion(int(decided.sum()), len(decided))
    return {
        "games": len(frame),
        "pushes": int(cover.isna().sum()),
        "decided": len(decided),
        "home_covers": int(decided.sum()),
        "home_cover_rate": rate,
        "ci_low": lo,
        "ci_high": hi,
    }


def line_error_summary(frame: pd.DataFrame) -> dict:
    err = home_cover_margin(frame["home_margin"], frame["spread_line"]).to_numpy(dtype=float)
    mae, mae_lo, mae_hi = mean_ci(np.abs(err))
    bias, bias_lo, bias_hi = mean_ci(err)
    fit = clustered_ols(frame["home_margin"], frame["spread_line"], week_blocks(frame))
    return {
        "games": len(frame),
        "mae": mae,
        "mae_ci_low": mae_lo,
        "mae_ci_high": mae_hi,
        "mean_error": bias,
        "mean_error_ci_low": bias_lo,
        "mean_error_ci_high": bias_hi,
        "rmse": float(np.sqrt(np.mean(err**2))) if len(err) else math.nan,
        "slope_margin_on_line": fit["slope"],
        "slope_ci_low": fit["slope_ci_low"],
        "slope_ci_high": fit["slope_ci_high"],
    }


def spread_calibration(hist, oos, holdout):
    cover_rows, error_rows = [], []
    for label, kind, d in windows(hist, oos, holdout, ERA_STARTS):
        lined = d[d["spread_line"].notna()]
        home_site = lined[lined["location"] == "Home"]
        cover_rows.append(
            {"window": label, "sample": kind, "sites": "non-neutral"} | cover_summary(home_site)
        )
        error_rows.append({"window": label, "sample": kind} | line_error_summary(lined))
        if kind == "pooled":
            all_sites = {"window": label, "sample": kind, "sites": "all, incl. neutral"}
            cover_rows.append(all_sites | cover_summary(lined))
    by_season = []
    for season, d in pd.concat([hist, oos]).groupby("season"):
        lined = d[d["spread_line"].notna()]
        row = {"season": season, "holdout": season == holdout} | line_error_summary(lined)
        by_season.append(row)
    neutral = hist[hist["location"] != "Home"]
    sites = neutral.groupby(["game_type", "stadium"]).size().rename("games").reset_index()
    sites = sites.sort_values(["games", "game_type", "stadium"], ascending=[False, True, True])
    return (
        pd.DataFrame(cover_rows),
        pd.DataFrame(error_rows),
        pd.DataFrame(by_season),
        sites.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------------------------
# 2. Key numbers
# ---------------------------------------------------------------------------------------------


def key_numbers(hist, oos, holdout):
    first, last = int(hist["season"].min()), holdout - 1
    before = hist[hist["season"] < PAT_RULE_CHANGE]
    after = hist[hist["season"] >= PAT_RULE_CHANGE]
    periods = {
        f"{first}-{PAT_RULE_CHANGE - 1}": before,
        f"{PAT_RULE_CHANGE}-{last}": after,
        f"{holdout} holdout": oos,
    }
    labels = list(periods)
    rows = []
    for k in KEY_MARGINS:
        row = {"abs_margin": k}
        counts = {}
        for label, d in periods.items():
            hits = int((d["home_margin"].abs() == k).sum())
            rate, lo, hi = proportion(hits, len(d))
            counts[label] = (hits, len(d))
            row |= {f"n_{label}": len(d), f"share_{label}": rate}
            row |= {f"ci_low_{label}": lo, f"ci_high_{label}": hi}
        d, lo, hi = newcombe_ci(*counts[labels[1]], *counts[labels[0]])
        row |= {"change_after_minus_before": d, "change_ci_low": lo, "change_ci_high": hi}
        rows.append(row)
    dist = []
    for label, d in periods.items():
        margin_counts = d["home_margin"].abs().astype(int).value_counts().sort_index()
        for margin, c in margin_counts.items():
            dist.append({"period": label, "abs_margin": margin, "games": c, "share": c / len(d)})
    return pd.DataFrame(rows), pd.DataFrame(dist), labels


# ---------------------------------------------------------------------------------------------
# 3. Home-field advantage
# ---------------------------------------------------------------------------------------------


def hfa_summary(frame: pd.DataFrame) -> dict:
    margin = mean_ci(frame["home_margin"])
    line = mean_ci(frame["spread_line"])
    gap = mean_ci(home_cover_margin(frame["home_margin"], frame["spread_line"]))
    cover = cover_summary(frame)
    out = {"games": len(frame)}
    for name, (m, lo, hi) in (
        ("home_margin", margin),
        ("spread", line),
        ("margin_minus_spread", gap),
        ("home_cover_rate", (cover["home_cover_rate"], cover["ci_low"], cover["ci_high"])),
    ):
        out |= {name: m, f"{name}_ci_low": lo, f"{name}_ci_high": hi}
    return out | {"cover_decided": cover["decided"]}


def home_field(hist, oos, holdout):
    rows = []
    for label, kind, d in windows(hist, oos, holdout, HFA_STARTS):
        h = d[(d["location"] == "Home") & d["spread_line"].notna()]
        rows.append({"window": label, "sample": kind} | hfa_summary(h))
    by_season = []
    for season, d in pd.concat([hist, oos]).groupby("season"):
        h = d[(d["location"] == "Home") & d["spread_line"].notna()]
        by_season.append({"season": season, "holdout": season == holdout} | hfa_summary(h))
    return pd.DataFrame(rows), pd.DataFrame(by_season)


# ---------------------------------------------------------------------------------------------
# 4. Moneyline calibration and the favourite-longshot pattern
# ---------------------------------------------------------------------------------------------


def with_moneyline(frame: pd.DataFrame) -> pd.DataFrame:
    """Games with both moneylines, ties dropped (a tie refunds a two-way moneyline bet)."""
    d = frame[frame["home_moneyline"].notna() & frame["away_moneyline"].notna()]
    d = d[d["home_margin"] != 0].copy()
    raw_home = np.array([american_to_implied(x) for x in d["home_moneyline"]])
    raw_away = np.array([american_to_implied(x) for x in d["away_moneyline"]])
    d["p_home"] = raw_home / (raw_home + raw_away)  # proportional de-vig
    d["overround"] = raw_home + raw_away - 1
    d["home_won"] = (d["home_margin"] > 0).astype(int)
    return d


def log_loss_terms(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def probability_skill(d: pd.DataFrame, constant: float, n_boot: int, seed: int) -> dict:
    y = d["home_won"].to_numpy(dtype=float)
    p = d["p_home"].to_numpy(dtype=float)
    c = np.full(len(y), constant)
    terms = np.column_stack(
        [
            (p - y) ** 2,
            (c - y) ** 2,
            log_loss_terms(y, p),
            log_loss_terms(y, c),
        ]
    )
    terms = np.column_stack([terms, terms[:, 1] - terms[:, 0], terms[:, 3] - terms[:, 2]])
    lo, hi = block_bootstrap_means(terms, week_blocks(d), n_boot, seed)
    mean = terms.mean(axis=0) if len(terms) else np.full(6, math.nan)
    names = (
        "brier",
        "brier_constant",
        "log_loss",
        "log_loss_constant",
        "brier_gain",
        "log_loss_gain",
    )
    out = {"games": len(d), "constant": constant, "mean_overround": float(d["overround"].mean())}
    for i, name in enumerate(names):
        out |= {
            name: float(mean[i]),
            f"{name}_ci_low": float(lo[i]),
            f"{name}_ci_high": float(hi[i]),
        }
    return out | logistic_recalibration(y, p, week_blocks(d))


def moneyline_sides(d: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game: backing home and backing away at the listed moneyline."""
    home = pd.DataFrame(
        {
            "season": d["season"].to_numpy(),
            "week": d["week"].to_numpy(),
            "p": d["p_home"].to_numpy(),
            "price": d["home_moneyline"].to_numpy(dtype=float),
            "won": d["home_won"].to_numpy(),
        }
    )
    away = home.assign(
        p=1 - home["p"], price=d["away_moneyline"].to_numpy(dtype=float), won=1 - home["won"]
    )
    sides = pd.concat([home, away], ignore_index=True)
    decimal = np.array([american_to_decimal(x) for x in sides["price"]])
    sides["profit"] = np.where(sides["won"] == 1, decimal - 1, -1.0)
    sides["fair_roi"] = sides["p"] * decimal - 1  # expected ROI if the de-vigged p is exact
    return sides


def roi_summary(s: pd.DataFrame, n_boot: int, seed: int) -> dict:
    rate, lo, hi = proportion(int(s["won"].sum()), len(s))
    roi_lo, roi_hi = block_bootstrap_means(s["profit"], week_blocks(s), n_boot, seed)
    return {
        "bets": len(s),
        "mean_p": float(s["p"].mean()) if len(s) else math.nan,
        "win_rate": rate,
        "win_ci_low": lo,
        "win_ci_high": hi,
        "roi": float(s["profit"].mean()) if len(s) else math.nan,
        "roi_ci_low": float(roi_lo[0]),
        "roi_ci_high": float(roi_hi[0]),
        "roi_if_devigged_price_exact": float(s["fair_roi"].mean()) if len(s) else math.nan,
    }


def moneyline(hist, oos, holdout, n_boot, seed):
    h, o = with_moneyline(hist), with_moneyline(oos)
    base_rate = float(h["home_won"].mean())
    first, last = int(h["season"].min()), int(h["season"].max())
    skill = [
        {
            "window": f"{first}-{last}",
            "sample": "pooled",
            "constant_source": "in-sample home win rate",
        }
        | probability_skill(h, base_rate, n_boot, seed),
        {
            "window": f"{holdout} holdout",
            "sample": "holdout",
            "constant_source": f"{first}-{last} home win rate",
        }
        | probability_skill(o, base_rate, n_boot, seed),
    ]
    reliability = []
    h_bins = pd.cut(h["p_home"], PROB_EDGES, right=False)
    for interval, d in h.groupby(h_bins, observed=True):
        rate, lo, hi = proportion(int(d["home_won"].sum()), len(d))
        reliability.append(
            {
                "p_home_bin": f"[{interval.left:.1f}, {interval.right:.1f})",
                "games": len(d),
                "mean_p_home": float(d["p_home"].mean()),
                "home_win_rate": rate,
                "ci_low": lo,
                "ci_high": hi,
                "observed_minus_priced": rate - float(d["p_home"].mean()),
            }
        )
    roi = []
    for label, frame in ((f"{first}-{last}", h), (f"{holdout} holdout", o)):
        sides = moneyline_sides(frame)
        buckets = pd.cut(sides["p"], ROI_EDGES, right=False)
        for interval, s in sides.groupby(buckets, observed=True):
            bucket = f"[{interval.left:.1f}, {interval.right:.1f})"
            roi.append({"window": label, "bucket": bucket} | roi_summary(s, n_boot, seed))
        for name, s in (
            ("all favourites (p > 0.5)", sides[sides["p"] > 0.5]),
            ("all underdogs (p < 0.5)", sides[sides["p"] < 0.5]),
        ):
            roi.append({"window": label, "bucket": name} | roi_summary(s, n_boot, seed))

    def ties(frame):
        priced = frame["home_moneyline"].notna() & frame["away_moneyline"].notna()
        return int(((frame["home_margin"] == 0) & priced).sum())

    extra = {
        "ties_excluded": ties(hist),
        "ties_excluded_holdout": ties(oos),
        "negative_overround_games": ", ".join(h.loc[h["overround"] < 0, "game_id"]) or "none",
        "both_moneylines_negative": int(
            ((h["home_moneyline"] < 0) & (h["away_moneyline"] < 0)).sum()
        ),
        # p(home) = 0.5 exactly: both sides land in the [0.5, 0.6) bucket
        "games_priced_exactly_even": int((h["home_moneyline"] == h["away_moneyline"]).sum()),
    }
    return pd.DataFrame(skill), pd.DataFrame(reliability), pd.DataFrame(roi), extra


def price_overround(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean bookmaker margin of the nflverse prices by season; a jump marks a change of source."""
    markets = (
        ("moneyline", "home_moneyline", "away_moneyline"),
        ("spread", "home_spread_odds", "away_spread_odds"),
        ("total", "over_odds", "under_odds"),
    )
    rows = []
    for season, d in frame.groupby("season"):
        row = {"season": season}
        for name, a, b in markets:
            both = d[d[a].notna() & d[b].notna() & (d[a] != 0) & (d[b] != 0)]
            margin = [
                american_to_implied(x) + american_to_implied(y) - 1
                for x, y in zip(both[a], both[b])
            ]
            row |= {
                f"{name}_games": len(both),
                f"{name}_overround": float(np.mean(margin)) if margin else math.nan,
            }
        if row["moneyline_games"]:
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# 5. Totals
# ---------------------------------------------------------------------------------------------


def totals_summary(frame: pd.DataFrame) -> dict:
    diff = (frame["points"] - frame["total_line"]).to_numpy(dtype=float)
    overs, unders = int((diff > 0).sum()), int((diff < 0).sum())
    rate, lo, hi = proportion(overs, overs + unders)
    m, m_lo, m_hi = mean_ci(diff)
    return {
        "games": len(frame),
        "pushes": int((diff == 0).sum()),
        "overs": overs,
        "unders": unders,
        "over_rate": rate,
        "over_ci_low": lo,
        "over_ci_high": hi,
        "mean_total_minus_line": m,
        "mean_ci_low": m_lo,
        "mean_ci_high": m_hi,
        "mae_total_line": float(np.mean(np.abs(diff))) if len(diff) else math.nan,
    }


def totals(hist, oos, holdout):
    rows = []
    for label, kind, d in windows(hist, oos, holdout, ERA_STARTS):
        rows.append({"window": label, "sample": kind} | totals_summary(d[d["total_line"].notna()]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# 6. Wind and the closing total
# ---------------------------------------------------------------------------------------------


def outdoor_with_wind(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame[
        frame["roof"].isin(OUTDOOR_ROOFS) & frame["total_line"].notna() & frame["wind"].notna()
    ]
    return d.assign(diff=d["points"] - d["total_line"])


def under_summary(frame: pd.DataFrame) -> dict:
    decided = frame[frame["diff"] != 0]
    under = (decided["diff"] < 0).to_numpy()
    rate, lo, hi = proportion(int(under.sum()), len(decided))
    m, m_lo, m_hi = mean_ci(frame["diff"])
    price = decided["under_odds"].fillna(STANDARD_PRICE).to_numpy(dtype=float)
    profit = np.where(under, [profit_per_unit(x) for x in price], -1.0) if len(decided) else []
    profit_std = np.where(under, profit_per_unit(STANDARD_PRICE), -1.0)
    return {
        "games": len(frame),
        "pushes": len(frame) - len(decided),
        "decided": len(decided),
        "unders": int(under.sum()),
        "under_rate": rate,
        "under_ci_low": lo,
        "under_ci_high": hi,
        "under_se_week_clustered": clustered_rate_se(under, week_blocks(decided)),
        "mean_total_minus_line": m,
        "mean_ci_low": m_lo,
        "mean_ci_high": m_hi,
        "mean_closing_total": float(frame["total_line"].mean()) if len(frame) else math.nan,
        "mean_points": float(frame["points"].mean()) if len(frame) else math.nan,
        "flat_roi_unders_at_110": float(np.mean(profit_std)) if len(decided) else math.nan,
        "flat_roi_unders_listed_price": float(np.mean(profit)) if len(decided) else math.nan,
    }


def wind(hist, oos, holdout):
    coverage = []
    for season, d in pd.concat([hist, oos]).groupby("season"):
        out = d[d["roof"].isin(OUTDOOR_ROOFS) & d["total_line"].notna()]
        coverage.append(
            {
                "season": season,
                "outdoor_open_games": len(out),
                "with_wind": int(out["wind"].notna().sum()),
                "share_with_wind": float(out["wind"].notna().mean()) if len(out) else math.nan,
                "wind_ge_threshold": int((out["wind"] >= WIND_THRESHOLD).sum()),
            }
        )
    threshold, regression = [], []
    for label, kind, d in windows(hist, oos, holdout, ERA_STARTS):
        w = outdoor_with_wind(d)
        threshold.append(
            {"window": label, "sample": kind, "wind_mph": f">= {WIND_THRESHOLD:g}"}
            | under_summary(w[w["wind"] >= WIND_THRESHOLD])
        )
        if kind == "pooled":
            threshold.append(
                {"window": label, "sample": kind, "wind_mph": "all"} | under_summary(w)
            )
        fit = clustered_ols(w["diff"], w["wind"], week_blocks(w))
        regression.append({"window": label, "sample": kind, "games": "all with wind"} | fit)
        if kind == "pooled":
            for name, sub in (
                (f"wind <= {WIND_IMPLAUSIBLE:g} mph", w[w["wind"] <= WIND_IMPLAUSIBLE]),
                ("pushes dropped", w[w["diff"] != 0]),
            ):
                fit = clustered_ols(sub["diff"], sub["wind"], week_blocks(sub))
                regression.append({"window": label, "sample": kind, "games": name} | fit)
    w = outdoor_with_wind(hist)
    bins = []
    for interval_label, d in w.groupby(
        pd.cut(w["wind"], WIND_EDGES, right=False, labels=WIND_LABELS), observed=True
    ):
        bins.append({"wind_mph": interval_label} | under_summary(d))
    every = outdoor_with_wind(pd.concat([hist, oos]))
    columns = ["game_id", "gameday", "wind", "points", "total_line"]
    flagged = every.loc[every["wind"] > WIND_IMPLAUSIBLE, columns].sort_values("wind")
    # Which holdout weeks have a wind value yet (the latest weeks can still be blank upstream)
    out = oos[oos["roof"].isin(OUTDOOR_ROOFS) & oos["total_line"].notna()]
    holdout_weeks = out.groupby("week").agg(
        outdoor_open_games=("wind", "size"), with_wind=("wind", "count")
    )
    return (
        pd.DataFrame(coverage),
        pd.DataFrame(threshold),
        pd.DataFrame(bins),
        pd.DataFrame(regression),
        flagged,
        holdout_weeks.reset_index(),
    )


# ---------------------------------------------------------------------------------------------
# 7. Wong teaser legs
# ---------------------------------------------------------------------------------------------


def teaser_legs(frame: pd.DataFrame) -> pd.DataFrame:
    """Two legs per game, each teased TEASER_POINTS toward the bettor, graded on the close."""
    lined = frame[frame["spread_line"].notna()]
    cover = home_cover_margin(lined["home_margin"], lined["spread_line"]).to_numpy(dtype=float)
    home_line = book_home_line(lined["spread_line"]).to_numpy(dtype=float)
    seasons, weeks = lined["season"].to_numpy(), lined["week"].to_numpy()
    legs = pd.DataFrame(
        {
            "season": np.concatenate([seasons, seasons]),
            "week": np.concatenate([weeks, weeks]),
            "side": ["home"] * len(lined) + ["away"] * len(lined),
            "book_line": np.concatenate([home_line, -home_line]),
            "teased_cover": np.concatenate([cover, -cover]) + TEASER_POINTS,
        }
    )
    fav = legs["book_line"].between(*WONG_FAVOURITE)
    dog = legs["book_line"].between(*WONG_UNDERDOG)
    legs["leg_type"] = np.where(fav, "wong_favourite", np.where(dog, "wong_underdog", "other"))
    return legs


def leg_summary(legs: pd.DataFrame) -> dict:
    decided = legs[legs["teased_cover"] != 0]
    won = (decided["teased_cover"] > 0).to_numpy()
    rate, lo, hi = proportion(int(won.sum()), len(decided))
    out = {
        "legs": len(legs),
        "pushes": len(legs) - len(decided),
        "decided": len(decided),
        "wins": int(won.sum()),
        "win_rate": rate,
        "ci_low": lo,
        "ci_high": hi,
        "win_se_week_clustered": clustered_rate_se(won, week_blocks(decided)),
    }
    for price in TEASER_PRICES:
        out[f"two_team_roi_at_{price}"] = rate**2 * american_to_decimal(price) - 1
    return out


def teasers(hist, oos, holdout):
    rows = []
    for label, kind, d in windows(hist, oos, holdout, ERA_STARTS):
        legs = teaser_legs(d)
        wong = legs[legs["leg_type"] != "other"]
        rows.append(
            {"window": label, "sample": kind, "legs_type": "Wong (all)"} | leg_summary(wong)
        )
        if kind in ("pooled", "holdout"):
            for name, key in (
                ("Wong favourites", "wong_favourite"),
                ("Wong underdogs", "wong_underdog"),
            ):
                sub = legs[legs["leg_type"] == key]
                rows.append({"window": label, "sample": kind, "legs_type": name} | leg_summary(sub))
            other = legs[legs["leg_type"] == "other"]
            rows.append(
                {"window": label, "sample": kind, "legs_type": "all other legs"}
                | leg_summary(other)
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# 8. Playoffs vs regular season
# ---------------------------------------------------------------------------------------------


def phase_comparison(frame: pd.DataFrame, n_perm: int, seed: int) -> dict:
    lined = frame[frame["spread_line"].notna()]
    err = home_cover_margin(lined["home_margin"], lined["spread_line"]).abs().to_numpy(dtype=float)
    post = (lined["phase"] == "POST").to_numpy()
    reg_mae, reg_lo, reg_hi = mean_ci(err[~post])
    post_mae, post_lo, post_hi = mean_ci(err[post])
    diff, d_lo, d_hi = welch_ci(err[post], err[~post])
    p, _ = stratified_permutation_p(err, post, lined["season"].to_numpy(), n_perm, seed)
    return {
        "reg_games": int((~post).sum()),
        "reg_mae": reg_mae,
        "reg_ci_low": reg_lo,
        "reg_ci_high": reg_hi,
        "post_games": int(post.sum()),
        "post_mae": post_mae,
        "post_ci_low": post_lo,
        "post_ci_high": post_hi,
        "post_minus_reg": diff,
        "diff_ci_low": d_lo,
        "diff_ci_high": d_hi,
        "permutation_p_two_sided": p,
    }


def random_subset_probability(
    reg_err: np.ndarray, k: int, threshold: float, n_draws: int, seed: int
):
    """P(a random k-game subset of regular-season games has MAE <= threshold)."""
    if k == 0 or k > len(reg_err):
        return math.nan
    keys = np.random.default_rng(seed).random((n_draws, len(reg_err)))
    pick = np.argpartition(keys, k - 1, axis=1)[:, :k]
    return float(np.mean(reg_err[pick].mean(axis=1) <= threshold))


def playoffs(hist, oos, holdout, n_perm, seed):
    last = holdout - 1
    final = hist[hist["season"] == last]
    rows = [
        {"window": f"{int(hist['season'].min())}-{last}", "sample": "pooled", "strata": "season"}
        | phase_comparison(hist, n_perm, seed),
        {"window": str(last), "sample": "last historical season", "strata": "none"}
        | phase_comparison(final, n_perm, seed),
        {"window": f"{holdout} holdout", "sample": "holdout", "strata": "none"}
        | phase_comparison(oos, n_perm, seed),
    ]
    lined = final[final["spread_line"].notna()]
    err = home_cover_margin(lined["home_margin"], lined["spread_line"]).abs()
    reg_err = err[lined["phase"] == "REG"].to_numpy(dtype=float)
    post_err = err[lined["phase"] == "POST"].to_numpy(dtype=float)
    subset_p = random_subset_probability(
        reg_err, len(post_err), float(post_err.mean()), n_perm, seed
    )
    rows[1]["p_random_reg_subset_mae_le_post"] = subset_p
    by_season = []
    for (season, phase), d in pd.concat([hist, oos]).groupby(["season", "phase"]):
        lined = d[d["spread_line"].notna()]
        mae, lo, hi = mean_ci(home_cover_margin(lined["home_margin"], lined["spread_line"]).abs())
        by_season.append(
            {
                "season": season,
                "phase": phase,
                "games": len(lined),
                "mae": mae,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(by_season)


# ---------------------------------------------------------------------------------------------
# Tests, Holm correction and prospective sample sizes
# ---------------------------------------------------------------------------------------------


def test_family(wind_threshold, wind_regression, teaser_rows, playoff_rows) -> pd.DataFrame:
    """The declared test family. Everything else in the report is descriptive (estimate + CI).

    The exact binomial tests treat games as independent. As a sensitivity check, each is
    repeated as a one-sided Wald z test with a week-clustered standard error; W3 is already
    week-clustered and P1/P2 are permutation tests, so their check is the same p-value.
    """
    pooled = wind_threshold[
        (wind_threshold["sample"] == "pooled") & (wind_threshold["wind_mph"] != "all")
    ]
    w = pooled.to_dict("records")[0]
    slope = wind_regression[
        (wind_regression["sample"] == "pooled") & (wind_regression["games"] == "all with wind")
    ].to_dict("records")[0]
    wong = teaser_rows[
        (teaser_rows["sample"] == "pooled") & (teaser_rows["legs_type"] == "Wong (all)")
    ]
    t = wong.to_dict("records")[0]
    clustered = "Wald z, week-clustered SE, one-sided"
    tests = []
    for tid, null in (("W1", 0.5), ("W2", breakeven_win_rate(STANDARD_PRICE))):
        p = stats.binomtest(w["unders"], w["decided"], null, alternative="greater").pvalue
        z = (w["under_rate"] - null) / w["under_se_week_clustered"]
        tests.append(
            {
                "id": tid,
                "family": "wind",
                "hypothesis": f"under rate at wind >= {WIND_THRESHOLD:g} mph > {null:.4f}",
                "window": w["window"],
                "n": w["decided"],
                "estimate": w["under_rate"],
                "null_value": null,
                "test": "exact binomial, one-sided",
                "p_value": float(p),
                "sensitivity_test": clustered,
                "p_sensitivity": float(stats.norm.sf(z)),
            }
        )
    tests.append(
        {
            "id": "W3",
            "family": "wind",
            "hypothesis": "slope of (total - closing total) on wind != 0 (points per mph)",
            "window": slope["window"],
            "n": slope["n"],
            "estimate": slope["slope"],
            "null_value": 0.0,
            "test": "OLS, week-clustered SE, two-sided",
            "p_value": slope["slope_p_two_sided"],
            "sensitivity_test": "same test (SE already week-clustered)",
            "p_sensitivity": slope["slope_p_two_sided"],
        }
    )
    for i, price in enumerate(TEASER_PRICES, start=1):
        null = two_team_leg_breakeven(price)
        p = stats.binomtest(t["wins"], t["decided"], null, alternative="greater").pvalue
        z = (t["win_rate"] - null) / t["win_se_week_clustered"]
        tests.append(
            {
                "id": f"T{i}",
                "family": "teaser",
                "hypothesis": f"Wong leg win rate > two-team break-even at {price} ({null:.4f})",
                "window": t["window"],
                "n": t["decided"],
                "estimate": t["win_rate"],
                "null_value": null,
                "test": "exact binomial, one-sided",
                "p_value": float(p),
                "sensitivity_test": clustered,
                "p_sensitivity": float(stats.norm.sf(z)),
            }
        )
    for tid, row in (
        ("P1", playoff_rows.to_dict("records")[0]),
        ("P2", playoff_rows.to_dict("records")[1]),
    ):
        tests.append(
            {
                "id": tid,
                "family": "playoffs",
                "hypothesis": "closing-line MAE differs between playoffs and regular season",
                "window": row["window"],
                "n": row["reg_games"] + row["post_games"],
                "estimate": row["post_minus_reg"],
                "null_value": 0.0,
                "test": f"permutation (strata: {row['strata']}), two-sided",
                "p_value": row["permutation_p_two_sided"],
                "sensitivity_test": "same test (permutation)",
                "p_sensitivity": row["permutation_p_two_sided"],
            }
        )
    out = pd.DataFrame(tests)
    out["p_holm"] = holm_adjust(out["p_value"])
    out["reject_at_5pct_holm"] = out["p_holm"] < ALPHA
    out["p_holm_sensitivity"] = holm_adjust(out["p_sensitivity"])
    out["reject_at_5pct_holm_sensitivity"] = out["p_holm_sensitivity"] < ALPHA
    return out


def prospective_power(wind_threshold, teaser_rows, holdout) -> pd.DataFrame:
    """How many qualifying bets a prospective test would need, and how many seasons that takes."""
    last_era = eras(ERA_STARTS, holdout - 1)[-1]
    seasons_in_era = last_era[2] - last_era[1] + 1
    rows = []
    wt = wind_threshold[wind_threshold["wind_mph"] != "all"].set_index("window")
    wong = teaser_rows[teaser_rows["legs_type"] == "Wong (all)"].set_index("window")
    pooled_label = wt.index[wt["sample"] == "pooled"][0]
    cases = [
        (
            "wind unders",
            wt,
            "under_rate",
            "decided",
            STANDARD_PRICE,
            breakeven_win_rate(STANDARD_PRICE),
        ),
    ]
    cases += [
        ("Wong teaser legs", wong, "win_rate", "decided", price, two_team_leg_breakeven(price))
        for price in TEASER_PRICES[:2]
    ]
    for name, table, rate_col, n_col, price, breakeven in cases:
        per_season = table.loc[last_era[0], n_col] / seasons_in_era
        for source in (pooled_label, last_era[0]):
            p1 = float(table.loc[source, rate_col])
            n = required_n(breakeven, p1)
            rows.append(
                {
                    "bet": name,
                    "price": price,
                    "break_even": breakeven,
                    "assumed_true_rate": p1,
                    "rate_source": source,
                    "n_needed_alpha05_power80": n,
                    "qualifying_per_season": per_season,
                    "per_season_source": last_era[0],
                    "seasons_needed": n / per_season if per_season else math.inf,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------------------------


def pct(x, lo=None, hi=None, digits=1) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    text = f"{100 * x:.{digits}f}%"
    if lo is not None and np.isfinite(lo) and np.isfinite(hi):
        text += f" [{100 * lo:.{digits}f}, {100 * hi:.{digits}f}]"
    return text


def num(x, lo=None, hi=None, digits=2, sign=False) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    spec = f"+.{digits}f" if sign else f".{digits}f"
    text = format(x, spec)
    if lo is not None and np.isfinite(lo) and np.isfinite(hi):
        text += f" [{format(lo, spec)}, {format(hi, spec)}]"
    return text


def pval(p) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


def md(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False, disable_numparse=True)


def report(tables: dict, labels: dict, extra: dict) -> str:
    prov = tables["provenance"].set_index("item")["value"]
    out = ["# Market facts: raw output of scripts/market_facts.py", ""]
    out += [f"- {k}: {v}" for k, v in prov.items()]
    out += [
        "",
        "Brackets are 95% intervals: Wilson for rates, t for means, Newcombe for differences of",
        "rates, Welch for differences of means, week-block bootstrap for Brier, log-loss and ROI,",
        "week-clustered standard errors for slopes. Rates exclude pushes.",
        "",
    ]

    c = tables["01_spread_cover_rate"]
    out += ["## 1. Closing-spread calibration", "", "### 1a. Home cover rate", ""]
    view = pd.DataFrame(
        {
            "Window": c["window"],
            "Sites": c["sites"],
            "Games": c["games"],
            "Pushes": c["pushes"],
            "Home cover": [pct(*r) for r in c[["home_cover_rate", "ci_low", "ci_high"]].to_numpy()],
        }
    )
    out += [md(view), ""]
    ns = tables["01_neutral_sites"]
    sb = int(ns.loc[ns["game_type"] == "SB", "games"].sum())
    top = ns[ns["game_type"] != "SB"].head(6)
    out += [
        f"Neutral-site games, {labels['pooled']}: {int(ns['games'].sum())} ({sb} Super Bowls). "
        "Most common other venues: "
        + ", ".join(f"{s} {n}" for s, n in zip(top["stadium"], top["games"])),
        "",
    ]
    e = tables["01_spread_error"]
    out += ["### 1b. Closing spread vs final home margin (all games with a line)", ""]
    view = pd.DataFrame(
        {
            "Window": e["window"],
            "Games": e["games"],
            "MAE (pts)": [num(*r) for r in e[["mae", "mae_ci_low", "mae_ci_high"]].to_numpy()],
            "Mean of margin - line": [
                num(*r, sign=True)
                for r in e[["mean_error", "mean_error_ci_low", "mean_error_ci_high"]].to_numpy()
            ],
            "Slope of margin on line": [
                num(*r, digits=3)
                for r in e[["slope_margin_on_line", "slope_ci_low", "slope_ci_high"]].to_numpy()
            ],
        }
    )
    out += [md(view), ""]
    s = tables["01_spread_error_by_season"]
    out += ["### 1c. Closing-line MAE by season", ""]
    view = pd.DataFrame(
        {
            "Season": [
                f"{x} (holdout)" if h else str(x) for x, h in zip(s["season"], s["holdout"])
            ],
            "Games": s["games"],
            "MAE (pts)": [num(*r) for r in s[["mae", "mae_ci_low", "mae_ci_high"]].to_numpy()],
            "Mean of margin - line": [
                num(*r, sign=True)
                for r in s[["mean_error", "mean_error_ci_low", "mean_error_ci_high"]].to_numpy()
            ],
        }
    )
    out += [md(view), ""]

    k = tables["02_key_numbers"]
    names = labels["key_periods"]
    out += ["## 2. Key numbers: share of games decided by exactly |margin| points", ""]
    view = pd.DataFrame({"abs(margin)": k["abs_margin"]})
    for name in names:
        n = int(k[f"n_{name}"].max())
        view[f"{name} (n={n})"] = [
            pct(*r) for r in k[[f"share_{name}", f"ci_low_{name}", f"ci_high_{name}"]].to_numpy()
        ]
    view[f"Change, {names[1]} minus {names[0]} (pct pts)"] = [
        num(100 * a, 100 * b, 100 * c, digits=2, sign=True)
        for a, b, c in k[
            ["change_after_minus_before", "change_ci_low", "change_ci_high"]
        ].to_numpy()
    ]
    view = view[
        [view.columns[0], view.columns[1], view.columns[2], view.columns[4], view.columns[3]]
    ]
    out += [md(view), ""]

    h = tables["03_home_field"]
    out += ["## 3. Home-field advantage (non-neutral games, points)", ""]
    view = pd.DataFrame(
        {
            "Window": h["window"],
            "Games": h["games"],
            "Mean home margin": [
                num(*r)
                for r in h[["home_margin", "home_margin_ci_low", "home_margin_ci_high"]].to_numpy()
            ],
            "Mean closing spread": [
                num(*r) for r in h[["spread", "spread_ci_low", "spread_ci_high"]].to_numpy()
            ],
            "Margin - spread": [
                num(*r, sign=True)
                for r in h[
                    [
                        "margin_minus_spread",
                        "margin_minus_spread_ci_low",
                        "margin_minus_spread_ci_high",
                    ]
                ].to_numpy()
            ],
            "Home cover": [
                pct(*r)
                for r in h[
                    ["home_cover_rate", "home_cover_rate_ci_low", "home_cover_rate_ci_high"]
                ].to_numpy()
            ],
        }
    )
    out += [md(view), ""]

    m = tables["04_moneyline_skill"]
    out += ["## 4. Moneyline calibration (proportional de-vig, ties excluded)", ""]
    view = pd.DataFrame(
        {
            "Window": m["window"],
            "Games": m["games"],
            "Constant": [f"{x:.3f} ({src})" for x, src in zip(m["constant"], m["constant_source"])],
            "Brier: market": [
                num(*r, digits=4) for r in m[["brier", "brier_ci_low", "brier_ci_high"]].to_numpy()
            ],
            "Brier: constant": [num(x, digits=4) for x in m["brier_constant"]],
            "Log-loss: market": [
                num(*r, digits=4)
                for r in m[["log_loss", "log_loss_ci_low", "log_loss_ci_high"]].to_numpy()
            ],
            "Log-loss: constant": [num(x, digits=4) for x in m["log_loss_constant"]],
            "Log-loss gain vs constant": [
                num(*r, digits=4)
                for r in m[
                    ["log_loss_gain", "log_loss_gain_ci_low", "log_loss_gain_ci_high"]
                ].to_numpy()
            ],
            "Recalibration slope b": [
                num(*r, digits=3) for r in m[["b", "b_ci_low", "b_ci_high"]].to_numpy()
            ],
        }
    )
    out += [md(view), ""]
    out += [f"- {k}: {v}" for k, v in extra.items()] + [""]
    r = tables["04_moneyline_reliability"]
    out += ["### 4b. Reliability of the de-vigged home win probability", ""]
    view = pd.DataFrame(
        {
            "p(home) bin": r["p_home_bin"],
            "Games": r["games"],
            "Mean p(home)": [f"{x:.3f}" for x in r["mean_p_home"]],
            "Home win rate": [
                pct(*x) for x in r[["home_win_rate", "ci_low", "ci_high"]].to_numpy()
            ],
            "Observed - priced (pct pts)": [
                num(100 * x, digits=1, sign=True) for x in r["observed_minus_priced"]
            ],
        }
    )
    out += [md(view), ""]
    roi = tables["04_moneyline_roi_by_bucket"]
    out += ["### 4c. Flat moneyline ROI by the side's de-vigged win probability", ""]
    view = pd.DataFrame(
        {
            "Window": roi["window"],
            "p(side)": roi["bucket"],
            "Bets": roi["bets"],
            "Mean p": [f"{x:.3f}" for x in roi["mean_p"]],
            "Win rate": [
                pct(*x) for x in roi[["win_rate", "win_ci_low", "win_ci_high"]].to_numpy()
            ],
            "Flat ROI": [pct(*x) for x in roi[["roi", "roi_ci_low", "roi_ci_high"]].to_numpy()],
            "ROI if de-vigged price exact": [pct(x) for x in roi["roi_if_devigged_price_exact"]],
        }
    )
    out += [md(view), ""]
    ov = tables["04_price_overround_by_season"]
    out += ["### 4d. Bookmaker overround of the nflverse prices by season", ""]
    view = pd.DataFrame(
        {
            "Season": ov["season"],
            "Games": ov["moneyline_games"],
            "Moneyline": [pct(x, digits=2) for x in ov["moneyline_overround"]],
            "Spread": [pct(x, digits=2) for x in ov["spread_overround"]],
            "Total": [pct(x, digits=2) for x in ov["total_overround"]],
        }
    )
    out += [md(view), ""]

    t = tables["05_totals"]
    out += ["## 5. Totals: final total points vs the closing total", ""]
    view = pd.DataFrame(
        {
            "Window": t["window"],
            "Games": t["games"],
            "Pushes": t["pushes"],
            "Over rate": [
                pct(*x) for x in t[["over_rate", "over_ci_low", "over_ci_high"]].to_numpy()
            ],
            "Mean total - line (pts)": [
                num(*x, sign=True)
                for x in t[["mean_total_minus_line", "mean_ci_low", "mean_ci_high"]].to_numpy()
            ],
        }
    )
    out += [md(view), ""]

    w = tables["06_wind_threshold"]
    out += [
        "## 6. Wind (outdoor/open roofs) and the closing total",
        "",
        "### 6a. Unders by window",
        "",
    ]
    view = pd.DataFrame(
        {
            "Window": w["window"],
            "Wind (mph)": w["wind_mph"],
            "Games": w["games"],
            "Pushes": w["pushes"],
            "Under rate": [
                pct(*x) for x in w[["under_rate", "under_ci_low", "under_ci_high"]].to_numpy()
            ],
            "Mean total - line (pts)": [
                num(*x, sign=True)
                for x in w[["mean_total_minus_line", "mean_ci_low", "mean_ci_high"]].to_numpy()
            ],
            "Flat ROI on unders at -110": [pct(x) for x in w["flat_roi_unders_at_110"]],
            "At listed price": [pct(x) for x in w["flat_roi_unders_listed_price"]],
        }
    )
    out += [md(view), "", "Listed price: nflverse under_odds (from 2006), -110 where missing.", ""]
    b = tables["06_wind_bins"]
    out += [f"### 6b. By wind speed, {labels['pooled']}", ""]
    view = pd.DataFrame(
        {
            "Wind (mph)": b["wind_mph"],
            "Games": b["games"],
            "Mean closing total": [num(x, digits=1) for x in b["mean_closing_total"]],
            "Mean points": [num(x, digits=1) for x in b["mean_points"]],
            "Mean total - line (pts)": [
                num(*x, sign=True)
                for x in b[["mean_total_minus_line", "mean_ci_low", "mean_ci_high"]].to_numpy()
            ],
            "Under rate": [
                pct(*x) for x in b[["under_rate", "under_ci_low", "under_ci_high"]].to_numpy()
            ],
        }
    )
    out += [md(view), ""]
    g = tables["06_wind_regression"]
    out += ["### 6c. OLS of (total - closing total) on wind, points per mph", ""]
    view = pd.DataFrame(
        {
            "Window": g["window"],
            "Games": [f"{n} ({sub})" for n, sub in zip(g["n"], g["games"])],
            "Slope": [
                num(*x, digits=3, sign=True)
                for x in g[["slope", "slope_ci_low", "slope_ci_high"]].to_numpy()
            ],
        }
    )
    out += [md(view), ""]
    cov = tables["06_wind_coverage_by_season"]
    low = cov[cov["share_with_wind"] < 0.9]
    out += [
        "Seasons where under 90% of outdoor/open games have a wind value: "
        + ", ".join(f"{s} ({100 * x:.0f}%)" for s, x in zip(low["season"], low["share_with_wind"])),
        "",
    ]
    hw = tables["06_wind_coverage_holdout_by_week"]
    out += [
        f"{labels['holdout']} outdoor/open games with a wind value, by week: "
        + ", ".join(
            f"week {wk} {a}/{n}"
            for wk, a, n in zip(hw["week"], hw["with_wind"], hw["outdoor_open_games"])
        ),
        "",
    ]
    flagged = tables["06_wind_above_40mph"]
    out += [
        f"Games listing wind above {WIND_IMPLAUSIBLE:g} mph (kept in the main estimates): "
        + ", ".join(f"{g_id} ({x:g} mph)" for g_id, x in zip(flagged["game_id"], flagged["wind"])),
        "",
    ]

    tz = tables["07_teaser_legs"]
    out += ["## 7. Six-point teaser legs graded on the closing spread", ""]
    out += [
        "Two-team break-even per leg: "
        + ", ".join(f"{p}: {pct(two_team_leg_breakeven(p), digits=2)}" for p in TEASER_PRICES),
        "",
    ]
    view = pd.DataFrame(
        {
            "Window": tz["window"],
            "Legs": tz["legs_type"],
            "Decided": tz["decided"],
            "Pushes": tz["pushes"],
            "Win rate": [pct(*x) for x in tz[["win_rate", "ci_low", "ci_high"]].to_numpy()],
        }
    )
    for price in TEASER_PRICES:
        view[f"2-team ROI at {price}"] = [pct(x) for x in tz[f"two_team_roi_at_{price}"]]
    out += [md(view), "", "ROI columns: win_rate^2 x decimal price - 1, i.e. independent legs.", ""]

    p = tables["08_playoff_vs_regular"]
    out += ["## 8. Closing-line MAE: playoffs vs regular season", ""]
    view = pd.DataFrame(
        {
            "Window": p["window"],
            "REG games": p["reg_games"],
            "REG MAE": [num(*x) for x in p[["reg_mae", "reg_ci_low", "reg_ci_high"]].to_numpy()],
            "POST games": p["post_games"],
            "POST MAE": [
                num(*x) for x in p[["post_mae", "post_ci_low", "post_ci_high"]].to_numpy()
            ],
            "POST - REG": [
                num(*x, sign=True)
                for x in p[["post_minus_reg", "diff_ci_low", "diff_ci_high"]].to_numpy()
            ],
            "Permutation p": [pval(x) for x in p["permutation_p_two_sided"]],
        }
    )
    out += [md(view), ""]
    subset = p["p_random_reg_subset_mae_le_post"].dropna()
    if len(subset):
        out += [
            "P(random regular-season subset of the same size has MAE <= playoff MAE), "
            f"{labels['last']}: {subset.max():.3f}",
            "",
        ]

    tests = tables["tests_holm"]
    out += ["## Test family with Holm correction", ""]
    view = pd.DataFrame(
        {
            "ID": tests["id"],
            "Hypothesis": tests["hypothesis"],
            "Window": tests["window"],
            "n": tests["n"],
            "Estimate": [f"{x:.4f}" for x in tests["estimate"]],
            "Null": [f"{x:.4f}" for x in tests["null_value"]],
            "p": [pval(x) for x in tests["p_value"]],
            "Holm p": [pval(x) for x in tests["p_holm"]],
            "Reject (Holm, 5%)": ["yes" if x else "no" for x in tests["reject_at_5pct_holm"]],
        }
    )
    out += [md(view), ""]
    out += ["### Sensitivity: week-clustered standard errors", ""]
    view = pd.DataFrame(
        {
            "ID": tests["id"],
            "Sensitivity test": tests["sensitivity_test"],
            "p": [pval(x) for x in tests["p_sensitivity"]],
            "Holm p": [pval(x) for x in tests["p_holm_sensitivity"]],
            "Reject (Holm, 5%)": [
                "yes" if x else "no" for x in tests["reject_at_5pct_holm_sensitivity"]
            ],
        }
    )
    out += [md(view), ""]

    pw = tables["prospective_power"]
    out += ["## Prospective sample sizes (one-sided alpha 0.05, power 0.80)", ""]
    view = pd.DataFrame(
        {
            "Bet": pw["bet"],
            "Price": pw["price"],
            "Break-even": [pct(x, digits=2) for x in pw["break_even"]],
            "Assumed true rate": [
                f"{pct(x, digits=1)} ({src})"
                for x, src in zip(pw["assumed_true_rate"], pw["rate_source"])
            ],
            "Bets needed": [
                f"{x:,.0f}" if np.isfinite(x) else "never (rate below break-even)"
                for x in pw["n_needed_alpha05_power80"]
            ],
            "Qualifying per season": [
                f"{x:.1f} ({src})"
                for x, src in zip(pw["qualifying_per_season"], pw["per_season_source"])
            ],
            "Seasons needed": [
                f"{x:,.0f}" if np.isfinite(x) else "n/a" for x in pw["seasons_needed"]
            ],
        }
    )
    out += [md(view), ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--games", required=True, type=Path, help="nflverse schedules games.csv")
    parser.add_argument(
        "--out", required=True, type=Path, help="directory for the CSVs and summary.md"
    )
    parser.add_argument(
        "--holdout-season", type=int, default=2026, help="season reported out of sample"
    )
    parser.add_argument("--n-boot", type=int, default=2000, help="week-block bootstrap draws")
    parser.add_argument("--n-perm", type=int, default=20000, help="permutation / subset draws")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    raw, done = load_games(args.games)
    holdout = args.holdout_season
    hist = done[done["season"] < holdout].copy()
    oos = done[done["season"] == holdout].copy()
    if hist.empty:
        raise SystemExit(f"no completed games before {holdout}")

    tables = {"provenance": provenance(args.games, raw, done, hist, oos, holdout)}
    cover, error, error_by_season, neutral_sites = spread_calibration(hist, oos, holdout)
    tables |= {"01_spread_cover_rate": cover, "01_spread_error": error}
    tables |= {"01_spread_error_by_season": error_by_season, "01_neutral_sites": neutral_sites}
    key, dist, key_periods = key_numbers(hist, oos, holdout)
    tables |= {"02_key_numbers": key, "02_margin_distribution": dist}
    hfa, hfa_by_season = home_field(hist, oos, holdout)
    tables |= {"03_home_field": hfa, "03_home_field_by_season": hfa_by_season}
    skill, reliability, roi, ml_extra = moneyline(hist, oos, holdout, args.n_boot, args.seed)
    tables |= {"04_moneyline_skill": skill, "04_moneyline_reliability": reliability}
    tables["04_moneyline_roi_by_bucket"] = roi
    tables["04_price_overround_by_season"] = price_overround(pd.concat([hist, oos]))
    tables["05_totals"] = totals(hist, oos, holdout)
    coverage, threshold, bins, regression, flagged, holdout_weeks = wind(hist, oos, holdout)
    tables |= {"06_wind_coverage_by_season": coverage, "06_wind_threshold": threshold}
    tables |= {"06_wind_bins": bins, "06_wind_regression": regression}
    tables |= {"06_wind_above_40mph": flagged, "06_wind_coverage_holdout_by_week": holdout_weeks}
    tables["07_teaser_legs"] = teasers(hist, oos, holdout)
    phase, phase_by_season = playoffs(hist, oos, holdout, args.n_perm, args.seed)
    tables |= {"08_playoff_vs_regular": phase, "08_mae_by_season_phase": phase_by_season}
    tables["tests_holm"] = test_family(threshold, regression, tables["07_teaser_legs"], phase)
    tables["prospective_power"] = prospective_power(threshold, tables["07_teaser_legs"], holdout)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(args.out / f"{name}.csv", index=False, float_format="%.6g")
    labels = {
        "key_periods": key_periods,
        "pooled": f"{int(hist['season'].min())}-{holdout - 1}",
        "last": str(holdout - 1),
        "holdout": f"{holdout} holdout",
    }
    text = report(tables, labels, ml_extra).rstrip("\n")
    (args.out / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
