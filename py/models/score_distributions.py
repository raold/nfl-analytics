"""
Score distribution models for NFL pricing.

Implements:
- Skellam margin PMF from Poisson home/away scoring rates
- Integer-margin reweighting to match key-number masses (3, 6, 7, 10)
- Cover/push/total probabilities from a discrete margin distribution

Notes
- Reweighting here is a simple multiplicative adjustment with renormalization.
  The dissertation discusses preserving location/scale; that refinement can be
  added by projecting the adjusted pmf back to the original mean/variance.

References
- Skellam (1946) distribution for score differences
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple


def _bessel_i(k: int, x: float) -> float:
    """Modified Bessel function I_k(x) via series (sufficient for moderate x).

    For production use, prefer scipy.special.iv. We avoid the dependency here.
    """
    # Truncated series for I_k(x) = sum_{m=0}^inf (1/m!Gamma(m+k+1)) (x/2)^{2m+k}
    # Use few terms for typical NFL intensities.
    terms = 0.0
    m = 0
    # Cap iterations for safety
    for m in range(50):
        num = (x / 2.0) ** (2 * m + k)
        den = math.factorial(m) * math.gamma(m + k + 1)
        term = num / den
        terms += term
        if term < 1e-14:
            break
    return terms


def skellam_pmf(mu_home: float, mu_away: float, k: int) -> float:
    """P(D = k) for Skellam difference of two independent Poissons.

    D = Home - Away, with Home~Poisson(mu_home), Away~Poisson(mu_away).
    """
    if mu_home < 0 or mu_away < 0:
        return 0.0
    lam = 2.0 * math.sqrt(mu_home * mu_away)
    return math.exp(-(mu_home + mu_away)) * ((mu_home / mu_away) ** (k / 2.0)) * _bessel_i(abs(k), lam)


def skellam_pmf_range(mu_home: float, mu_away: float, k_min: int = -80, k_max: int = 80) -> Dict[int, float]:
    """Compute Skellam pmf over integer margins in [k_min, k_max]."""
    pmf = {k: skellam_pmf(mu_home, mu_away, k) for k in range(k_min, k_max + 1)}
    # Normalize to mitigate truncation error
    s = sum(pmf.values())
    if s <= 0:
        return pmf
    return {k: v / s for k, v in pmf.items()}


def reweight_key_masses(pmf: Dict[int, float], targets: Dict[int, float]) -> Dict[int, float]:
    """Multiply probabilities at key margins by factors so masses match targets.

    Args:
        pmf: dict margin->prob
        targets: dict key_margin->target_prob (e.g., {3: 0.09, 6: 0.06, 7: 0.07, 10: 0.05})

    Returns: new normalized pmf.

    Notes: This is a simple one-pass multiplicative reweight + renormalize. For
    tighter control over mean/variance, use iterative proportional fitting with
    constraints on low-order moments.
    """
    adj = dict(pmf)
    for k, target in targets.items():
        current = adj.get(k, 0.0)
        if current <= 0 or target <= 0:
            # If either is zero or missing, skip; leave mass as-is
            continue
        factor = target / current
        adj[k] = current * factor
    # Renormalize
    s = sum(adj.values())
    if s <= 0:
        return adj
    return {k: v / s for k, v in adj.items()}


def cover_push_probs(pmf: Dict[int, float], spread: float) -> Tuple[float, float, float]:
    """Compute (cover, push, fail) for home vs given spread.

    For half-point spreads, push mass is zero. For integer spreads, push is mass
    at margin == -spread (home margin equals negative spread threshold).
    """
    # Home covers if margin > -spread
    cover = sum(p for m, p in pmf.items() if m > -spread)
    # Push if integer spread and margin equals -spread exactly
    push = 0.0
    if float(spread).is_integer():
        key = int(-spread)
        push = pmf.get(key, 0.0)
    fail = 1.0 - cover - push
    return cover, push, max(0.0, fail)


def total_over_prob(pmf_home: Dict[int, float], pmf_away: Dict[int, float], total: float) -> float:
    """Approximate P(total points > total) via independent Poisson marginals.

    This helper assumes separate Poisson scoring for home and away. For broader
    use with margin-only pmf, prefer simulation from bivariate score models.
    """
    # Construct supports
    home_scores = sorted(pmf_home.keys())
    away_scores = sorted(pmf_away.keys())
    p = 0.0
    for h in home_scores:
        for a in away_scores:
            s = h + a
            if s > total:
                p += pmf_home[h] * pmf_away[a]
    return p


__all__ = [
    "skellam_pmf",
    "skellam_pmf_range",
    "reweight_key_masses",
    "cover_push_probs",
    "total_over_prob",
]

