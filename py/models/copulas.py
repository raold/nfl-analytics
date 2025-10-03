"""
Copula utilities for dependence modeling between spread and total (or other legs).

Implements minimal Gaussian copula fit and sampling on pseudo-observations.
For production use, consider statsmodels or scipy; here we avoid heavy deps.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Tuple


def _phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _Phi_inv(u: float) -> float:
    """Inverse standard normal CDF via Acklam's approximation."""
    # Coeffs from Peter J. Acklam (2003)
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1 - plow
    if u < plow:
        q = math.sqrt(-2 * math.log(u))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    if phigh < u:
        q = math.sqrt(-2 * math.log(1 - u))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    q = u - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    ) / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


def fit_gaussian_copula(u: Iterable[float], v: Iterable[float]) -> float:
    """Estimate Gaussian copula correlation via Pearson corr in Gaussian scores."""
    xs = [_Phi_inv(min(max(1e-9, ui), 1 - 1e-9)) for ui in u]
    ys = [_Phi_inv(min(max(1e-9, vi), 1 - 1e-9)) for vi in v]
    n = len(xs)
    if n == 0:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return 0.0
    r = num / (denx * deny)
    return max(-0.999, min(0.999, r))


def joint_success_prob_gaussian(u1_thresh: float, u2_thresh: float, rho: float) -> float:
    """P(U1>u1_thresh, U2>u2_thresh) under Gaussian copula with corr rho.

    Inputs are thresholds in [0,1] for each marginal CDF.
    """
    z1 = _Phi_inv(u1_thresh)
    z2 = _Phi_inv(u2_thresh)
    # P(Z1>z1, Z2>z2) for bivariate normal -> use survival copula formula.
    # Here approximate via Gaussian tail bound using Mehler's formula-inspired approx.
    # For simplicity, we use a rough approximation; replace with mvn CDF if available.
    # Approximate by assuming independence plus a correlation adjustment term.
    p_ind = (1 - 0.5 * (1 + math.erf(z1 / math.sqrt(2)))) * (
        1 - 0.5 * (1 + math.erf(z2 / math.sqrt(2)))
    )
    adj = rho * _phi(z1) * _phi(z2)
    p = max(0.0, min(1.0, p_ind + adj))
    return p


__all__ = ["fit_gaussian_copula", "joint_success_prob_gaussian"]

