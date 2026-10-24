"""
Guardrails that refuse results which are too good to be true.

A sustained NFL ATS win rate above ~60%, or ROI above ~15%, has never been documented with
public information. When an evaluation produces one, the evaluation is wrong (sign error,
leakage, post-hoc selection) far more often than the model is right. The Dec 2025
"74.9% ATS / +43% ROI" claim was a spread sign error that these checks would have stopped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_analytics.evaluation.market_benchmark.metrics import grade_ats
from nfl_analytics.utils.spread_convention import home_cover

MAX_PLAUSIBLE_WIN_RATE = 0.60
MAX_PLAUSIBLE_ROI = 0.15
MIN_BETS_FOR_CHECK = 100
HOME_COVER_RANGE = (0.46, 0.53)


class ImplausibleResultError(RuntimeError):
    """Raised when a benchmark result is too good to be true."""


def check_plausibility(summary: dict, allow_implausible: bool = False) -> list[str]:
    """Raise if win rate or ROI is beyond anything credible on a meaningful sample."""
    problems = []
    if summary.get("bets", 0) >= MIN_BETS_FOR_CHECK:
        if summary["win_rate"] > MAX_PLAUSIBLE_WIN_RATE:
            problems.append(
                f"ATS win rate {summary['win_rate']:.1%} on {summary['bets']} bets exceeds "
                f"{MAX_PLAUSIBLE_WIN_RATE:.0%}"
            )
        if summary["roi"] > MAX_PLAUSIBLE_ROI:
            problems.append(f"ROI {summary['roi']:+.1%} exceeds {MAX_PLAUSIBLE_ROI:.0%}")
    if problems and not allow_implausible:
        raise ImplausibleResultError(
            "; ".join(problems)
            + ". Check the spread sign convention, look-ahead in features, and post-hoc filters "
            "(docs/POSTMORTEMS.md section 4). Pass allow_implausible=True only after that."
        )
    return problems


def check_home_cover_rate(market: pd.DataFrame) -> float:
    """The pooled home cover rate must look like a fair market (~49%)."""
    rate = float(home_cover(market["home_margin"], market["spread_close"]).dropna().mean())
    lo, hi = HOME_COVER_RANGE
    if not lo <= rate <= hi:
        raise ImplausibleResultError(
            f"Pooled home cover rate {rate:.1%} is outside {lo:.0%}-{hi:.0%}; the cover "
            "grading or the spread sign is wrong"
        )
    return rate


def placebo_win_rate(
    df: pd.DataFrame, pred_col: str = "pred_home_margin", min_edge: float = 1.0, seed: int = 0
) -> float:
    """ATS win rate after shuffling predictions within each week. Should be ~50%: if the
    shuffled model still 'wins', the grading itself is biased."""
    rng = np.random.default_rng(seed)
    shuffled = df.copy()
    shuffled[pred_col] = shuffled.groupby(["season", "week"])[pred_col].transform(
        lambda s: rng.permutation(s.to_numpy())
    )
    bets = grade_ats(shuffled, pred_col, min_edge)
    settled = bets[bets["result"] != "push"]
    return float((settled["result"] == "win").mean()) if len(settled) else float("nan")
