"""
Odds conversions and vig removal.

American odds: -150 means risk 150 to win 100; +130 means risk 100 to win 130.
"""

from __future__ import annotations

import math

import numpy as np

STANDARD_PRICE = -110  # default spread price when a game has no recorded odds


def american_to_decimal(odds: float) -> float:
    """Decimal odds (total return per unit staked, stake included)."""
    if odds == 0 or not math.isfinite(odds):
        raise ValueError(f"invalid American odds: {odds}")
    return 1 + (odds / 100 if odds > 0 else 100 / -odds)


def american_to_implied(odds: float) -> float:
    """Raw implied probability (includes the bookmaker margin)."""
    return 1 / american_to_decimal(odds)


def profit_per_unit(odds: float) -> float:
    """Net profit on a winning 1-unit stake."""
    return american_to_decimal(odds) - 1


def breakeven_win_rate(odds: float = STANDARD_PRICE) -> float:
    """Win rate needed to break even at these odds (-110 -> 52.38%)."""
    return american_to_implied(odds)


def devig_proportional(p_home: float, p_away: float) -> tuple[float, float]:
    """Remove the margin by normalizing the two implied probabilities to sum to 1."""
    total = p_home + p_away
    return p_home / total, p_away / total


def devig_shin(p_home: float, p_away: float, tol: float = 1e-12) -> tuple[float, float]:
    """Shin (1993) de-vigging for a two-outcome market.

    Models the margin as protection against insider trading, which shades favorites less
    than longshots. Solves for the insider fraction z by bisection.
    """
    raw = np.array([p_home, p_away], dtype=float)
    total = raw.sum()
    if total <= 1 + tol:
        return devig_proportional(p_home, p_away)

    def fair(z: float) -> np.ndarray:
        return (np.sqrt(z**2 + 4 * (1 - z) * raw**2 / total) - z) / (2 * (1 - z))

    lo, hi = 0.0, 0.999
    for _ in range(200):
        mid = (lo + hi) / 2
        if fair(mid).sum() > 1:
            lo = mid
        else:
            hi = mid
    probs = fair((lo + hi) / 2)
    probs = probs / probs.sum()
    return float(probs[0]), float(probs[1])
