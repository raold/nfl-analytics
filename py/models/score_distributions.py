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
    return (
        math.exp(-(mu_home + mu_away)) * ((mu_home / mu_away) ** (k / 2.0)) * _bessel_i(abs(k), lam)
    )


def skellam_pmf_range(
    mu_home: float, mu_away: float, k_min: int = -80, k_max: int = 80
) -> dict[int, float]:
    """Compute Skellam pmf over integer margins in [k_min, k_max]."""
    pmf = {k: skellam_pmf(mu_home, mu_away, k) for k in range(k_min, k_max + 1)}
    # Normalize to mitigate truncation error
    s = sum(pmf.values())
    if s <= 0:
        return pmf
    return {k: v / s for k, v in pmf.items()}


def reweight_key_masses(pmf: dict[int, float], targets: dict[int, float]) -> dict[int, float]:
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


def reweight_with_moments(
    pmf: dict[int, float],
    targets: dict[int, float],
    mu: float,
    var: float,
    *,
    iters: int = 200,
    eta: float = 1e-3,
    tol: float = 1e-6,
) -> dict[int, float]:
    """Moment-preserving reweighting to match key masses.

    Minimizes squared error on key masses with nonnegativity, while projecting
    onto the affine set that preserves normalization, mean, and variance.

    Args:
        pmf: baseline integer pmf (margin -> prob)
        targets: desired masses at keys (e.g., {3:0.09,6:0.06,7:0.07,10:0.05})
        mu, var: target mean and variance to preserve
        iters: max iterations
        eta: gradient step size
        tol: convergence tolerance on key-mass deltas

    Returns: adjusted pmf dict
    """
    # Support and baseline arrays
    ks = sorted(pmf.keys())
    q = {k: max(0.0, float(pmf[k])) for k in ks}
    # Initialize weights
    w = {k: 1.0 for k in ks}

    # Precompute q-weighted sums for projection basis
    def _moments_under_q() -> dict[str, float]:
        s0 = sum(q.values())
        s1 = sum(k * q[k] for k in ks)
        s2 = sum((k**2) * q[k] for k in ks)
        c2 = sum(((k - mu) ** 2) * q[k] for k in ks)
        d_c2 = sum(k * ((k - mu) ** 2) * q[k] for k in ks)
        c4 = sum(((k - mu) ** 4) * q[k] for k in ks)
        return {"s0": s0, "s1": s1, "s2": s2, "c2": c2, "d_c2": d_c2, "c4": c4}

    const = _moments_under_q()

    def _project_weights() -> None:
        # Current p and moments
        p = {k: q[k] * w[k] for k in ks}
        S0 = sum(p.values())
        S1 = sum(k * p[k] for k in ks)
        S2 = sum(((k - mu) ** 2) * p[k] for k in ks)
        # Desired deltas
        d0 = 1.0 - S0
        d1 = mu - S1
        d2 = var - S2
        # Solve 3x3 for (alpha, beta, gamma)
        # Matrix M using precomputed q-weighted sums
        # Row order corresponds to constraints on (sum, mean, variance)

        M = [
            [const["s0"], const["s1"], const["c2"]],
            [const["s1"], const["s2"], const["d_c2"]],
            [const["c2"], const["d_c2"], const["c4"]],
        ]
        b = [d0, d1, d2]
        # Small ridge for numerical stability
        lam = 1e-10
        for i in range(3):
            M[i][i] += lam
        # Manual 3x3 solve (no numpy hard dependency)
        A = [M[0] + [b[0]], M[1] + [b[1]], M[2] + [b[2]]]
        n = 3
        for i in range(n):
            piv = i
            for r in range(i + 1, n):
                if abs(A[r][i]) > abs(A[piv][i]):
                    piv = r
            A[i], A[piv] = A[piv], A[i]
            if abs(A[i][i]) < 1e-18:
                continue
            fac = A[i][i]
            for c in range(i, n + 1):
                A[i][c] /= fac
            for r in range(n):
                if r == i:
                    continue
                fac = A[r][i]
                for c in range(i, n + 1):
                    A[r][c] -= fac * A[i][c]
        alpha, beta, gamma = (A[0][n], A[1][n], A[2][n])
        # Apply affine adjustment in weight space: w <- w + a + b*d + c*(d-mu)^2
        for k in ks:
            w[k] = max(0.0, w[k] + alpha + beta * k + gamma * ((k - mu) ** 2))

    # Iterative projected updates
    for _ in range(max(1, iters)):
        # Gradient step on key masses
        max_err = 0.0
        for k, t in targets.items():
            if k not in q:
                continue
            cur = q[k] * w[k]
            err = cur - t
            max_err = max(max_err, abs(err))
            # d/dw_k (cur - t)^2 = 2*(cur - t)*q_k
            grad = 2.0 * err * q[k]
            w[k] = max(0.0, w[k] - eta * grad)
        # Project to match normalization/mean/variance
        _project_weights()
        if max_err < tol:
            break

    # Build adjusted pmf and normalize for safety
    out = {k: q[k] * max(0.0, w[k]) for k in ks}
    s = sum(out.values())
    if s > 0:
        out = {k: v / s for k, v in out.items()}
    return out


def cover_push_probs(pmf: dict[int, float], spread: float) -> tuple[float, float, float]:
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


def total_over_prob(pmf_home: dict[int, float], pmf_away: dict[int, float], total: float) -> float:
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
    "reweight_with_moments",
    "cover_push_probs",
    "total_over_prob",
]
