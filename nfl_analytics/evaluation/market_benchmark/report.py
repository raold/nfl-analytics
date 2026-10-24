"""Run a full benchmark for one set of predictions and render it."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from nfl_analytics.evaluation.market_benchmark.guardrails import (
    check_home_cover_rate,
    check_plausibility,
    placebo_win_rate,
)
from nfl_analytics.evaluation.market_benchmark.metrics import (
    ats_summary,
    edge_threshold_sweep,
    grade_ats,
    information_regression,
    spread_accuracy,
    win_prob_skill,
)

MIN_PRED_LINE_CORR = 0.2


def run_benchmark(
    df: pd.DataFrame,
    name: str,
    pred_col: str = "pred_home_margin",
    min_edge: float = 1.0,
    allow_implausible: bool = False,
) -> dict:
    """Benchmark predictions (expected home margin, nflverse sign) against the closing market.

    ``df`` is a market frame (see market.prepare_market) with ``pred_col`` attached. Optional
    columns: ``pred_home_win_prob`` (log-loss vs market) and ``prediction_timestamp`` (flags
    predictions made at or after kickoff as retroactive).
    """
    home_cover_rate = check_home_cover_rate(df)
    d = df.dropna(subset=[pred_col])
    gap = d[pred_col] - d["spread_close"]
    if not np.allclose(gap, gap.iloc[0]):
        corr = float(np.corrcoef(d[pred_col], d["spread_close"])[0, 1])
        if corr < MIN_PRED_LINE_CORR:
            raise ValueError(
                f"{pred_col} correlates {corr:+.2f} with spread_close. Predictions must be the "
                "expected HOME margin (positive = home favored); a negative correlation means "
                "sportsbook sign. Convert with nfl_analytics.utils.spread_convention."
            )

    bets = grade_ats(d, pred_col, min_edge)
    summary = ats_summary(bets)
    problems = check_plausibility(summary, allow_implausible=allow_implausible)

    retroactive = None
    if "prediction_timestamp" in d and "kickoff" in d:
        made = pd.to_datetime(d["prediction_timestamp"], utc=True, errors="coerce")
        kick = pd.to_datetime(d["kickoff"], utc=True, errors="coerce")
        retroactive = float((made >= kick).mean())

    return {
        "name": name,
        "games": len(d),
        "seasons": sorted(int(s) for s in d["season"].unique()),
        "min_edge": min_edge,
        "ats": summary,
        "plausibility_problems": problems,
        "home_cover_rate": home_cover_rate,
        "placebo_win_rate": placebo_win_rate(d, pred_col, min_edge),
        "spread_accuracy": spread_accuracy(d, pred_col),
        "information_regression": information_regression(d, pred_col),
        "win_prob_skill": win_prob_skill(d),
        "edge_threshold_sweep": edge_threshold_sweep(d, pred_col=pred_col).to_dict("records"),
        "retroactive_share": retroactive,
    }


def _fmt(x, pct: bool = False, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if pct:
        return f"{x:+.1%}" if signed else f"{x:.1%}"
    return f"{x:.3f}"


def to_markdown(result: dict) -> str:
    """Human-readable summary of run_benchmark output."""
    a, s = result["ats"], result["spread_accuracy"]
    lines = [
        f"## {result['name']} vs closing market",
        "",
        f"Seasons {result['seasons'][0]}-{result['seasons'][-1]}, {result['games']} games, "
        f"bets at |edge| >= {result['min_edge']} pts.",
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
        f"| Pooled home cover rate | {_fmt(result['home_cover_rate'], pct=True)} | sanity: 46-53% |",
    ]
    reg = result["information_regression"]
    if reg:
        lines.append(
            f"| Model info beyond the line (coef on pred - line) | {reg['model_gap_coef']:+.3f} | "
            f"{reg['model_gap_ci95'][0]:+.3f} to {reg['model_gap_ci95'][1]:+.3f} |"
        )
    wp = result["win_prob_skill"]
    if wp:
        lines.append(
            f"| Log-loss: model / market | {wp['model_log_loss']:.4f} / {wp['market_log_loss']:.4f} | "
            f"diff {wp['log_loss_diff_ci95'][0]:+.4f} to {wp['log_loss_diff_ci95'][1]:+.4f} |"
        )
    if result["retroactive_share"]:
        lines += [
            "",
            f"**{result['retroactive_share']:.0%} of predictions were made at or after kickoff "
            "(retroactive); this is not a prospective test.**",
        ]
    return "\n".join(lines) + "\n"


def to_json(result: dict) -> str:
    return json.dumps(
        result, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)
    )
