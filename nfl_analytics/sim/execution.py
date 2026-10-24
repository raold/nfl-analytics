"""
Execution engine (stub) combining depth, fill, latency.

Loads execution priors JSON and provides a simulate_order() function that
returns filled fraction and executed price given requested q, book, market,
time-to-kickoff, velocity, and side.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..execution.depth_model import sample_depth, tau_bucket
from ..execution.fill_model import prob_fill
from ..execution.latency_model import sample_latency


@dataclass
class BucketParams:
    beta: tuple[float, float, float, float]
    rmse: float
    theta: tuple[float, float, float, float, float]
    mu_log: float
    sigma_log: float


class ExecutionEngine:
    def __init__(self, priors_path: str):
        with open(priors_path, encoding="utf-8") as f:
            raw = json.load(f)
        self.params: dict[tuple[str, str, str], BucketParams] = {}
        for k, v in raw.items():
            book, market, tau = k.split("::")
            beta = tuple(v.get("depth", {}).get("beta", [0, 0, 0, 0]))  # type: ignore
            rmse = float(v.get("depth", {}).get("rmse", 0.0))
            theta = tuple(v.get("fill", {}).get("theta", [0, 0, 0, 0, 0]))  # type: ignore
            mu = float(v.get("latency", {}).get("mu_log", 0.0))
            sg = float(v.get("latency", {}).get("sigma_log", 0.0))
            self.params[(book, market, tau)] = BucketParams(beta, rmse, theta, mu, sg)

    def simulate_order(
        self,
        book: str,
        market: str,
        tau_min: float,
        q_req: float,
        quoted_price: float,
        velocity: float,
        side: float,
    ) -> tuple[float, float, dict[str, float]]:
        tau = tau_bucket(tau_min)
        p = self.params.get((book, market, tau))
        if not p:
            return 0.0, quoted_price, {"fill_prob": 0.0, "latency": 0.0, "impact": 0.0}
        # Fill probability
        pfill = prob_fill(p.theta, q_req, velocity, side)
        q_filled = min(q_req, max(0.0, pfill) * q_req)
        # Depth impact
        impact = sample_depth(p.beta, q_filled, velocity)
        # Latency impact
        ell = sample_latency(p.mu_log, p.sigma_log)
        impact += velocity * ell
        executed_price = quoted_price + impact
        stats = {"fill_prob": pfill, "latency": ell, "impact": impact}
        return q_filled, executed_price, stats
