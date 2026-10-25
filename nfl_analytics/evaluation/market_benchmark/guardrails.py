"""
Guardrails that refuse results which are too good to be true, and market data that cannot be
graded correctly.

A sustained NFL ATS win rate above ~60%, or ROI above ~15%, has never been documented with
public information. When an evaluation produces one, the evaluation is wrong (sign error,
leakage, post-hoc selection) far more often than the model is right. The Dec 2025
"74.9% ATS / +43% ROI" claim was a spread sign error that these checks would have stopped.

The market checks look at the closing lines before any model is graded:

- Pooled home cover rate. nflverse 1999-2025 (all game types) has 3,473 home covers in 7,081
  decided games (0.4905; 195 pushes excluded). A two-sided exact binomial test against
  HOME_COVER_P0 refuses only when p < HOME_COVER_MIN_P, so short samples are judged by what
  they can show: 2026 weeks 1-7 (40/88 = 45.5%, p = 0.52) passes, while 2019-2024 graded with
  the lines negated (55.9% on 1,577 decided games, p = 4e-8) is refused.
- Line sign. Lines in nflverse sign (expected home margin) correlate positively with the home
  margin (+0.43 over 1999-2025 regular seasons); sportsbook-signed lines correlate just as
  strongly negatively. The binomial test can miss sportsbook sign on a small sample (2026 weeks
  1-7 negated: 53/92 = 57.6%, p = 0.12); the correlation does not (-0.33 on 94 games).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from nfl_analytics.evaluation.market_benchmark.metrics import grade_ats
from nfl_analytics.utils.spread_convention import home_cover

MAX_PLAUSIBLE_WIN_RATE = 0.60
MAX_PLAUSIBLE_ROI = 0.15
MIN_BETS_FOR_CHECK = 100
HOME_COVER_P0 = 0.49  # nflverse 1999-2025 pooled home cover rate, 3,473 / 7,081 = 0.4905
HOME_COVER_MIN_P = 0.001  # refuse when the two-sided exact binomial p-value is below this
SIGN_CHECK_MIN_GAMES = 30  # corr(spread_close, home_margin) must be > 0 from this many games
NFLVERSE_LINE_MARGIN_CORR = 0.43  # 1999-2025 regular seasons, quoted in the refusal message


class ImplausibleResultError(RuntimeError):
    """Raised when a benchmark result is too good to be true, or the market data is broken."""


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


def market_sanity(market: pd.DataFrame) -> dict:
    """Pooled home cover rate with an exact binomial test, and the line / home margin correlation.

    Pushes and games without a score or line are excluded from the cover count.
    """
    cover = home_cover(market["home_margin"], market["spread_close"]).dropna()
    covers, decided = int(cover.sum()), int(len(cover))
    pair = market[["spread_close", "home_margin"]].astype(float).dropna()
    corr = math.nan
    if len(pair) > 2 and pair["spread_close"].nunique() > 1 and pair["home_margin"].nunique() > 1:
        corr = float(np.corrcoef(pair["spread_close"], pair["home_margin"])[0, 1])
    p_value = float(stats.binomtest(covers, decided, HOME_COVER_P0).pvalue) if decided else math.nan
    return {
        "home_cover_rate": covers / decided if decided else math.nan,
        "home_covers": covers,
        "decided_games": decided,
        "home_cover_p0": HOME_COVER_P0,
        "home_cover_p_value": p_value,
        "line_margin_corr": corr,
        "corr_games": len(pair),
        "sign_checked": len(pair) >= SIGN_CHECK_MIN_GAMES,
    }


def check_market_sanity(market: pd.DataFrame) -> dict:
    """Refuse market data whose line sign or home cover base rate cannot be right.

    Returns the market_sanity statistics when both checks pass. Not overridable: if the market
    itself fails, every number graded against it is wrong.
    """
    s = market_sanity(market)
    problems = []
    if s["sign_checked"] and math.isnan(s["line_margin_corr"]):
        problems.append(
            f"spread_close or the home margin is constant over {s['corr_games']} games; the "
            "lines are probably placeholders, not closing lines"
        )
    elif s["sign_checked"] and s["line_margin_corr"] <= 0:
        problems.append(
            f"spread_close correlates {s['line_margin_corr']:+.2f} with the home margin over "
            f"{s['corr_games']} games. Lines in nflverse sign (expected home margin, positive = "
            f"home favored) correlate positively (+{NFLVERSE_LINE_MARGIN_CORR:.2f} over nflverse "
            "1999-2025), so these lines are probably in sportsbook sign (home favorite negative). "
            "Negate them before grading (nfl_analytics/utils/spread_convention.py)"
        )
    if s["decided_games"] and s["home_cover_p_value"] < HOME_COVER_MIN_P:
        problems.append(
            f"Pooled home cover rate {s['home_cover_rate']:.1%} ({s['home_covers']}/"
            f"{s['decided_games']} decided games) is not what a fair market produces: two-sided "
            f"exact binomial p = {s['home_cover_p_value']:.1g} against {HOME_COVER_P0:.0%} "
            f"(refused below {HOME_COVER_MIN_P}). The cover grading is probably wrong: the home "
            "team covers when home_margin - spread_close > 0 with nflverse lines "
            "(spread_convention.home_cover); otherwise the lines are in sportsbook sign, or the "
            "scores and lines come from different games"
        )
    if problems:
        raise ImplausibleResultError("; ".join(problems))
    return s


def check_home_cover_rate(market: pd.DataFrame) -> float:
    """check_market_sanity, returning only the pooled home cover rate."""
    return check_market_sanity(market)["home_cover_rate"]


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
