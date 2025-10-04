"""
Depth/limit curve impact model (stub implementation).

Fits a simple quadratic impact curve with velocity interaction per
(book, market, tau_bucket):

  E[Δp | q, τ, book] = β0 + β1 q + β2 q^2 + β3 sign(velocity) * q

where q is stake fraction of posted limit in [0, 1], Δp is executed - quoted
price (e.g., ticks). We also estimate a residual rmse per bucket.

This module contains pure-Python least-squares without external deps and is
intended as a scaffold. Replace with your preferred regression library.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class DepthParams:
    beta: tuple[float, float, float, float]
    rmse: float


def tau_bucket(tau_min: float) -> str:
    if tau_min is None:
        return ">180"
    m = float(tau_min)
    if m > 180:
        return ">180"
    if m > 60:
        return "60-180"
    if m > 15:
        return "15-60"
    return "<15"


def _normal_eq(X: list[list[float]], y: list[float]) -> tuple[float, float, float, float]:
    # Solve (X'X)β = X'y for 4-dim β using closed-form 4x4 solver (Gaussian elim)

    # Build XtX and Xty
    XtX = [[0.0] * 4 for _ in range(4)]
    Xty = [0.0] * 4
    for i, row in enumerate(X):
        for a in range(4):
            Xty[a] += row[a] * y[i]
            for b in range(4):
                XtX[a][b] += row[a] * row[b]
    # Gaussian elimination
    n = 4
    # augment
    A = [XtX[i] + [Xty[i]] for i in range(n)]
    for i in range(n):
        # pivot
        piv = i
        for r in range(i + 1, n):
            if abs(A[r][i]) > abs(A[piv][i]):
                piv = r
        A[i], A[piv] = A[piv], A[i]
        if abs(A[i][i]) < 1e-12:
            return (0.0, 0.0, 0.0, 0.0)
        # normalize
        fac = A[i][i]
        for c in range(i, n + 1):
            A[i][c] /= fac
        # eliminate
        for r in range(n):
            if r == i:
                continue
            fac = A[r][i]
            for c in range(i, n + 1):
                A[r][c] -= fac * A[i][c]
    beta = tuple(A[i][n] for i in range(n))
    return beta  # type: ignore


def fit_depth(rows: Iterable[dict[str, str]]) -> dict[tuple[str, str, str], DepthParams]:
    buckets: dict[tuple[str, str, str], list[tuple[list[float], float]]] = defaultdict(list)
    for r in rows:
        try:
            book = r.get("book", "unknown")
            market = r.get("market", "spread")
            tau = tau_bucket(float(r.get("tau_min", "999")))
            q = float(r.get("q_req", r.get("q", "0")))
            quoted = float(r.get("quoted_price", "0"))
            executed = float(r.get("executed_price", quoted))
            vel = float(r.get("velocity", r.get("dline_dt", "0")))
            sign_v = 1.0 if vel >= 0 else -1.0
            dp = executed - quoted
            Xrow = [1.0, q, q * q, sign_v * q]
            buckets[(book, market, tau)].append((Xrow, dp))
        except Exception:
            continue
    out: dict[tuple[str, str, str], DepthParams] = {}
    for key, pairs in buckets.items():
        if len(pairs) < 4:
            out[key] = DepthParams((0.0, 0.0, 0.0, 0.0), 0.0)
            continue
        X = [p[0] for p in pairs]
        y = [p[1] for p in pairs]
        beta = _normal_eq(X, y)
        # RMSE
        import math

        resid = []
        for i in range(len(X)):
            yhat = sum(beta[j] * X[i][j] for j in range(4))
            resid.append(y[i] - yhat)
        rmse = math.sqrt(sum(r * r for r in resid) / max(1, len(resid)))
        out[key] = DepthParams(beta, rmse)
    return out


def sample_depth(beta: tuple[float, float, float, float], q: float, velocity: float) -> float:
    sign_v = 1.0 if velocity >= 0 else -1.0
    b0, b1, b2, b3 = beta
    return b0 + b1 * q + b2 * q * q + b3 * sign_v * q
