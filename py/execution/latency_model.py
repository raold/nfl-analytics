"""
Latency model (stub implementation).

Fits a lognormal to observed latency ℓ = executed_time - signal_time (seconds)
per (book, market, tau_bucket). Assumes timestamps are either seconds since
epoch or ISO-8601 parseable; this stub prefers seconds.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .depth_model import tau_bucket


@dataclass
class LatencyParams:
    mu_log: float
    sigma_log: float


def _to_seconds(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def fit_latency(rows: Iterable[Dict[str, str]]) -> Dict[Tuple[str, str, str], LatencyParams]:
    groups: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for r in rows:
        try:
            book = r.get("book", "unknown")
            market = r.get("market", "spread")
            tau = tau_bucket(float(r.get("tau_min", "999")))
            sig = _to_seconds(r.get("signal_ts", "0"))
            exe = _to_seconds(r.get("exec_ts", r.get("executed_ts", "0")))
            ell = max(1e-6, exe - sig)
            groups[(book, market, tau)].append(ell)
        except Exception:
            continue
    out: Dict[Tuple[str, str, str], LatencyParams] = {}
    for key, vals in groups.items():
        if not vals:
            out[key] = LatencyParams(0.0, 0.0)
            continue
        logs = [math.log(v) for v in vals if v > 0]
        mu = sum(logs) / len(logs)
        var = sum((x - mu) ** 2 for x in logs) / max(1, len(logs) - 1)
        out[key] = LatencyParams(mu, math.sqrt(max(var, 1e-12)))
    return out


def sample_latency(mu_log: float, sigma_log: float) -> float:
    # Simple lognormal sample via Box-Muller
    import random

    u1, u2 = random.random(), random.random()
    z = math.sqrt(-2.0 * math.log(max(u1, 1e-12))) * math.cos(2.0 * math.pi * u2)
    return math.exp(mu_log + sigma_log * z)

