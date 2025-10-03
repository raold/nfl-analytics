"""
Teaser and middle pricing utilities.

Uses per-leg success probabilities and a dependence model (Gaussian copula)
to compute basket success and expected value under given payouts.
"""
from __future__ import annotations

from typing import Tuple

from ..models.copulas import joint_success_prob_gaussian


def teaser_ev(q1: float, q2: float, payout_decimal: float, rho: float = 0.0) -> Tuple[float, float]:
    """Expected value and joint success for a 2-leg teaser.

    Args:
        q1, q2: Leg success probabilities (marginals)
        payout_decimal: Decimal payout d (net profit is d-1 on success)
        rho: Correlation for Gaussian copula between leg indicators

    Returns:
        (ev, joint_success)
    """
    # Convert marginal success probs to uniform thresholds
    u1 = 1 - q1
    u2 = 1 - q2
    p_both = joint_success_prob_gaussian(u1_thresh=u1, u2_thresh=u2, rho=rho)
    ev = p_both * (payout_decimal - 1.0) - (1.0 - p_both)
    return ev, p_both


def middle_breakeven(required_mass: float, pmf_margin: dict[int, float], n: int) -> bool:
    """Check if middle around integer n has sufficient mass.

    Args:
        required_mass: Threshold mass needed at integer margin n to justify a middle
        pmf_margin: Discrete pmf over integer margins
        n: Integer margin

    Returns: True if feasible (mass >= required_mass)
    """
    mass = pmf_margin.get(n, 0.0)
    return mass >= required_mass


__all__ = ["teaser_ev", "middle_breakeven"]

