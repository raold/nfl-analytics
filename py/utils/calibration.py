"""
Calibration utilities: reliability curves, ECE, Brier score, log loss.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Tuple


def brier_score(y_true: Iterable[int], p_hat: Iterable[float]) -> float:
    num = 0.0
    n = 0
    for y, p in zip(y_true, p_hat):
        num += (y - p) ** 2
        n += 1
    return num / max(1, n)


def log_loss(y_true: Iterable[int], p_hat: Iterable[float], eps: float = 1e-12) -> float:
    s = 0.0
    n = 0
    for y, p in zip(y_true, p_hat):
        p = min(1 - eps, max(eps, p))
        s += -(y * math.log(p) + (1 - y) * math.log(1 - p))
        n += 1
    return s / max(1, n)


def reliability_curve(y_true: List[int], p_hat: List[float], bins: int = 10) -> List[Tuple[float, float, int]]:
    """Return list of (bin_center, observed_rate, count)."""
    assert len(y_true) == len(p_hat)
    n = len(y_true)
    if n == 0:
        return []
    # Create bins [0,1]
    edges = [i / bins for i in range(bins + 1)]
    sums = [0.0] * bins
    counts = [0] * bins
    for y, p in zip(y_true, p_hat):
        k = min(bins - 1, max(0, int(p * bins)))
        sums[k] += y
        counts[k] += 1
    out: List[Tuple[float, float, int]] = []
    for k in range(bins):
        c = counts[k]
        rate = (sums[k] / c) if c > 0 else 0.0
        center = 0.5 * (edges[k] + edges[k + 1])
        out.append((center, rate, c))
    return out


def expected_calibration_error(y_true: List[int], p_hat: List[float], bins: int = 10) -> float:
    curve = reliability_curve(y_true, p_hat, bins)
    n = len(y_true)
    ece = 0.0
    for k, (center, obs, count) in enumerate(curve):
        if count == 0:
            continue
        ece += (count / n) * abs(obs - center)
    return ece


__all__ = ["brier_score", "log_loss", "reliability_curve", "expected_calibration_error"]

