"""
Spread sign convention for ``games.spread_close``.

``games.spread_close`` is copied verbatim from nflverse ``spread_line``
(R/ingestion/ingest_schedules.R, R/ingestion/ingest_current_season.R), which is
expressed as the expected HOME margin: positive means the home team is favored.
Example: 2018_01_BUF_BAL has spread_close = 7.5 because BAL (home) was a 7.5-point
favorite (sportsbook notation: BAL -7.5).

Sportsbook notation is the mirror image (home -7.5), so the sportsbook cover rule
``margin + line > 0`` becomes ``margin - spread_close > 0`` here. Using the
sportsbook rule on nflverse data inflates ATS accuracy to ~74% for any model that
agrees with the favorite (see docs/POSTMORTEMS.md, "Spread sign convention").

Always compute covers, edges, and cover probabilities through these helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Fraction of home-moneyline-favorite games that must have spread_close > 0
_MIN_AGREEMENT = 0.90


def home_cover_margin(home_margin, spread_close):
    """Points by which the home team beat the closing spread (negative = failed to cover)."""
    return home_margin - spread_close


def home_cover(home_margin, spread_close):
    """Home ATS result: 1.0 cover, 0.0 no cover, NaN for a push or a missing line.

    Accepts scalars, numpy arrays, or pandas Series (index is preserved).
    """
    cover_margin = home_cover_margin(home_margin, spread_close)
    result = np.where(cover_margin > 0, 1.0, np.where(cover_margin < 0, 0.0, np.nan))
    if isinstance(cover_margin, pd.Series):
        return pd.Series(result, index=cover_margin.index, name="home_cover")
    if np.ndim(result) == 0:
        return float(result)
    return result


def model_edge(predicted_home_margin, spread_close):
    """Model edge in points for the HOME side (positive = model likes home vs the line)."""
    return predicted_home_margin - spread_close


def book_home_line(spread_close):
    """Convert spread_close to sportsbook notation for the home team (favorite negative)."""
    return -spread_close


def implied_home_win_prob(spread_close, scale: float = 7.0):
    """Rough home win probability implied by the spread (logistic, ~7 points per logit)."""
    return 1.0 / (1.0 + np.exp(-np.asarray(spread_close, dtype=float) / scale))


def assert_nflverse_convention(
    df: pd.DataFrame,
    spread_col: str = "spread_close",
    home_ml_col: str = "home_moneyline",
) -> None:
    """Fail fast if ``spread_col`` does not follow the nflverse (home-favored positive) convention.

    Uses moneylines when present (home favorite <=> home_moneyline < 0); otherwise
    falls back to outcomes (spread_close must correlate positively with home margin).
    """
    spread = pd.to_numeric(df[spread_col], errors="coerce")

    if home_ml_col in df.columns:
        ml = pd.to_numeric(df[home_ml_col], errors="coerce")
        mask = (ml < 0) & spread.notna() & (spread != 0)
        if mask.sum() >= 20:
            agreement = float((spread[mask] > 0).mean())
            if agreement < _MIN_AGREEMENT:
                raise ValueError(
                    f"{spread_col} is not in nflverse convention: only {agreement:.1%} of "
                    f"home-favorite games have {spread_col} > 0 (expected >= {_MIN_AGREEMENT:.0%})"
                )
            return

    if {"home_score", "away_score"}.issubset(df.columns):
        margin = df["home_score"] - df["away_score"]
        mask = spread.notna() & margin.notna()
        if mask.sum() >= 20:
            corr = float(np.corrcoef(spread[mask], margin[mask])[0, 1])
            if corr <= 0:
                raise ValueError(
                    f"{spread_col} correlates {corr:+.2f} with home margin; nflverse convention "
                    "requires a positive correlation"
                )
            return

    raise ValueError(f"Not enough data to verify the {spread_col} sign convention")
