"""
Fill probability model (stub implementation).

Estimates P(fill >= q | q, τ, book, velocity sign) with a simple logistic
regression per (book, market, tau_bucket). This is a lightweight Newton solver
over features [1, q, q^2, |velocity|, same_sign], where same_sign indicates
velocity moving against our side.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .depth_model import tau_bucket


@dataclass
class FillParams:
    theta: Tuple[float, float, float, float, float]


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logistic_fit(X: List[List[float]], y: List[int], iters: int = 10) -> Tuple[float, ...]:
    p = len(X[0])
    theta = [0.0] * p
    for _ in range(iters):
        # Compute gradient and Hessian (p x p)
        g = [0.0] * p
        H = [[0.0] * p for _ in range(p)]
        for i in range(len(X)):
            z = sum(theta[j] * X[i][j] for j in range(p))
            p_i = _sigmoid(z)
            diff = y[i] - p_i
            for a in range(p):
                g[a] += X[i][a] * diff
                for b in range(p):
                    H[a][b] -= X[i][a] * X[i][b] * p_i * (1 - p_i)
        # Solve H Δ = g (Newton step); add small ridge for stability
        for d in range(p):
            H[d][d] -= 1e-4
        # Gaussian elim
        A = [H[i] + [g[i]] for i in range(p)]
        for i in range(p):
            piv = i
            for r in range(i + 1, p):
                if abs(A[r][i]) > abs(A[piv][i]):
                    piv = r
            A[i], A[piv] = A[piv], A[i]
            if abs(A[i][i]) < 1e-12:
                return tuple(theta)
            fac = A[i][i]
            for c in range(i, p + 1):
                A[i][c] /= fac
            for r in range(p):
                if r == i:
                    continue
                fac = A[r][i]
                for c in range(i, p + 1):
                    A[r][c] -= fac * A[i][c]
        delta = [A[i][p] for i in range(p)]
        for j in range(p):
            theta[j] += delta[j]
    return tuple(theta)


def fit_fill(rows: Iterable[Dict[str, str]]) -> Dict[Tuple[str, str, str], FillParams]:
    groups: Dict[Tuple[str, str, str], Tuple[List[List[float]], List[int]]] = defaultdict(lambda: ([], []))
    for r in rows:
        try:
            book = r.get("book", "unknown")
            market = r.get("market", "spread")
            tau = tau_bucket(float(r.get("tau_min", "999")))
            q = float(r.get("q_req", r.get("q", "0")))
            q_exec = float(r.get("executed_q", r.get("q_filled", "0")))
            vel = float(r.get("velocity", r.get("dline_dt", "0")))
            side = float(r.get("side", "1"))  # +1/-1
            same_sign = 1.0 if vel * side > 0 else 0.0
            x = [1.0, q, q * q, abs(vel), same_sign]
            y = 1 if q_exec + 1e-6 >= q else 0
            X, Y = groups[(book, market, tau)]
            X.append(x)
            Y.append(y)
        except Exception:
            continue
    out: Dict[Tuple[str, str, str], FillParams] = {}
    for key, (X, Y) in groups.items():
        if not X:
            out[key] = FillParams((0, 0, 0, 0, 0))
            continue
        theta = _logistic_fit(X, Y, iters=8)
        out[key] = FillParams(theta)
    return out


def prob_fill(theta: Tuple[float, float, float, float, float], q: float, vel: float, side: float) -> float:
    same_sign = 1.0 if vel * side > 0 else 0.0
    z = theta[0] + theta[1] * q + theta[2] * (q * q) + theta[3] * abs(vel) + theta[4] * same_sign
    return _sigmoid(z)

