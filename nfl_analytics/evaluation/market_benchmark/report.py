"""Run a full benchmark for one set of predictions and render it."""

from __future__ import annotations

import datetime as dt
import json
import math

import numpy as np
import pandas as pd

from nfl_analytics.evaluation.market_benchmark.guardrails import (
    HOME_COVER_P0,
    SIGN_CHECK_MIN_GAMES,
    check_market_sanity,
    check_plausibility,
    placebo_win_rate,
)
from nfl_analytics.evaluation.market_benchmark.metrics import (
    ats_summary,
    bets_at_prediction,
    clv_summary,
    edge_threshold_sweep,
    grade_ats,
    information_regression,
    spread_accuracy,
    win_prob_skill,
)

# corr(pred, line) <= 0 is refused as a sign error; below this the report carries a warning
MIN_PRED_LINE_CORR = 0.2
LINE_AT_PREDICTION = "spread_at_prediction"  # written by predict mode (see predict.py)


def _pred_line_corr(d: pd.DataFrame, pred_col: str) -> float | None:
    """corr(pred, spread_close); None when pred - line is constant (the line plus a constant)."""
    gap = d[pred_col] - d["spread_close"]
    if np.allclose(gap, gap.iloc[0]):
        return None
    if d[pred_col].nunique() < 2 or d["spread_close"].nunique() < 2:
        return math.nan
    return float(np.corrcoef(d[pred_col], d["spread_close"])[0, 1])


def run_benchmark(
    df: pd.DataFrame,
    name: str,
    pred_col: str = "pred_home_margin",
    min_edge: float = 1.0,
    allow_implausible: bool = False,
) -> dict:
    """Benchmark predictions (expected home margin, nflverse sign) against the closing market.

    ``df`` is a market frame (see market.prepare_market) with ``pred_col`` attached. Optional
    columns: ``pred_home_win_prob`` (log-loss vs market), ``prediction_timestamp`` (flags
    predictions made at or after kickoff as retroactive) and ``spread_at_prediction`` (the line
    when the prediction was made: adds closing-line value).
    """
    sanity = check_market_sanity(df)
    d = df.dropna(subset=[pred_col])
    if d.empty:
        raise ValueError(f"No games with a {pred_col} prediction to grade")
    warnings = []
    if not sanity["sign_checked"]:
        warnings.append(
            f"Only {sanity['corr_games']} graded games: the line-sign check (corr of "
            f"spread_close with the home margin) needs {SIGN_CHECK_MIN_GAMES}"
        )

    corr = _pred_line_corr(d, pred_col)
    if corr is not None and math.isnan(corr):
        warnings.append(f"{pred_col} or the line is constant; corr(pred, line) is undefined")
    elif corr is not None and corr <= 0:
        raise ValueError(
            f"{pred_col} correlates {corr:+.2f} with spread_close. Predictions must be the "
            "expected HOME margin (positive = home favored); a correlation at or below zero means "
            "sportsbook sign. Convert with nfl_analytics.utils.spread_convention."
        )
    elif corr is not None and corr < MIN_PRED_LINE_CORR:
        warnings.append(
            f"{pred_col} correlates only {corr:+.2f} with the closing line (below "
            f"{MIN_PRED_LINE_CORR}): the sign is plausible, but the predictions share little "
            "with the market. Check that they are the expected home margin in points"
        )

    bets = grade_ats(d, pred_col, min_edge)
    summary = ats_summary(bets)
    problems = check_plausibility(summary, allow_implausible=allow_implausible)

    retroactive = None
    if "prediction_timestamp" in d and "kickoff" in d:
        made = pd.to_datetime(d["prediction_timestamp"], utc=True, errors="coerce")
        kick = pd.to_datetime(d["kickoff"], utc=True, errors="coerce")
        known = made.notna() & kick.notna()
        if known.any():
            retroactive = float((made[known] >= kick[known]).mean())
        if not known.all():
            warnings.append(
                f"{int((~known).sum())} predictions lack a prediction timestamp or kickoff "
                "and are left out of the retroactive share"
            )

    clv = None
    if LINE_AT_PREDICTION in d:
        clv = clv_summary(bets_at_prediction(d, pred_col, LINE_AT_PREDICTION, min_edge))

    return {
        "name": name,
        "games": len(d),
        "seasons": sorted(int(s) for s in d["season"].unique()),
        "min_edge": min_edge,
        "ats": summary,
        "plausibility_problems": problems,
        "home_cover_rate": sanity["home_cover_rate"],
        "market_sanity": sanity,
        "pred_line_corr": corr,
        "placebo_win_rate": placebo_win_rate(d, pred_col, min_edge),
        "spread_accuracy": spread_accuracy(d, pred_col),
        "information_regression": information_regression(d, pred_col),
        "win_prob_skill": win_prob_skill(d),
        "edge_threshold_sweep": edge_threshold_sweep(d, pred_col=pred_col).to_dict("records"),
        "retroactive_share": retroactive,
        "closing_line_value": clv,
        "warnings": warnings,
    }


def _fmt(x, pct: bool = False, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if pct:
        return f"{x:+.1%}" if signed else f"{x:.1%}"
    return f"{x:.3f}"


def season_span(seasons) -> str:
    """The seasons as written in a report: '2019-2024', '2025', or '2019, 2022-2024' with gaps."""
    runs: list[list[int]] = []
    for s in sorted({int(s) for s in seasons}):
        if runs and s == runs[-1][1] + 1:
            runs[-1][1] = s
        else:
            runs.append([s, s])
    return ", ".join(f"{a}-{b}" if b > a else f"{a}" for a, b in runs)


def to_markdown(result: dict) -> str:
    """Human-readable summary of run_benchmark output."""
    a, s, m = result["ats"], result["spread_accuracy"], result["market_sanity"]
    corr = m["line_margin_corr"]
    corr_txt = "n/a" if corr is None or math.isnan(corr) else f"{corr:+.2f}"
    seasons = result["seasons"]
    lines = [
        f"## {result['name']} vs closing market",
        "",
        f"{'Season' if len(seasons) == 1 else 'Seasons'} {season_span(seasons)}, "
        f"{result['games']} games, bets at |edge| >= {result['min_edge']} pts.",
        "",
        "| Metric | Value | 95% CI |",
        "|---|---|---|",
        f"| ATS record | {a['wins']}-{a['losses']}-{a['pushes']} | |",
        f"| Win rate | {_fmt(a['win_rate'], pct=True)} | {_fmt(a['win_rate_ci95'][0], pct=True)} to "
        f"{_fmt(a['win_rate_ci95'][1], pct=True)} |",
        f"| Breakeven | {_fmt(a['breakeven_win_rate'], pct=True)} | |",
        f"| ROI | {_fmt(a['roi'], pct=True, signed=True)} | {_fmt(a['roi_ci95'][0], pct=True, signed=True)} to "
        f"{_fmt(a['roi_ci95'][1], pct=True, signed=True)} |",
        f"| P(win rate > breakeven) | p = {_fmt(a['p_value_beats_breakeven'])} | |",
        f"| Spread MAE: model / market | {s['model_mae']:.2f} / {s['market_mae']:.2f} | "
        f"DM p = {_fmt(s['dm_p_value_two_sided'])} |",
        f"| Placebo win rate (shuffled) | {_fmt(result['placebo_win_rate'], pct=True)} | should be ~50% |",
        f"| Pooled home cover rate | {_fmt(result['home_cover_rate'], pct=True)} | "
        f"{m['home_covers']}/{m['decided_games']}, binomial p = {_fmt(m['home_cover_p_value'])} "
        f"vs {HOME_COVER_P0:.0%} |",
        f"| Line vs home margin correlation | {corr_txt} | sign check: must be > 0 from "
        f"{SIGN_CHECK_MIN_GAMES} games |",
    ]
    reg = result["information_regression"]
    if reg["note"]:
        lines.append(f"| Model info beyond the line (coef on pred - line) | n/a | {reg['note']} |")
    else:
        lines.append(
            f"| Model info beyond the line (coef on pred - line) | {reg['model_gap_coef']:+.3f} | "
            f"{reg['model_gap_ci95'][0]:+.3f} to {reg['model_gap_ci95'][1]:+.3f} |"
        )
    wp = result["win_prob_skill"]
    if wp:
        lo, hi = wp["log_loss_diff_ci95"]
        ci = "n/a (one week)" if math.isnan(lo) else f"diff {lo:+.4f} to {hi:+.4f}"
        lines.append(
            f"| Log-loss: model / market | {wp['model_log_loss']:.4f} / {wp['market_log_loss']:.4f} | "
            f"{ci} |"
        )
    clv = result["closing_line_value"]
    if clv and clv["bets"]:
        lo, hi = clv["mean_points_ci95"]
        ci = "n/a (one week)" if math.isnan(lo) else f"{lo:+.2f} to {hi:+.2f}"
        lines.append(
            f"| Closing-line value, pts per bet ({clv['bets']} bets at the line when predicted) | "
            f"{clv['mean_points']:+.2f} | {ci} |"
        )
    elif clv:
        lines.append("| Closing-line value | no bets at the line when predicted | |")
    share = result["retroactive_share"]
    if share:
        lines += [
            "",
            f"**{share:.0%} of predictions were made at or after kickoff "
            "(retroactive); this is not a prospective test.**",
        ]
    elif share == 0:
        lines += ["", "All timestamped predictions were made before kickoff (prospective)."]
    if result["warnings"]:
        lines += ["", "Warnings:"] + [f"- {w}" for w in result["warnings"]]
    return "\n".join(lines) + "\n"


def _jsonable(o):
    """Plain JSON types: NaN and +-inf become None, numpy scalars Python scalars."""
    if o is None or o is pd.NaT or o is pd.NA:
        return None
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, (bool, int, str)):
        return o
    if isinstance(o, (dt.datetime, dt.date)):
        return o.isoformat()
    return str(o)


def to_json(result: dict) -> str:
    """Strict JSON (RFC 8259): no NaN or Infinity tokens."""
    return json.dumps(_jsonable(result), indent=2, allow_nan=False)
