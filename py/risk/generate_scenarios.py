"""
Generate scenario returns for a set of bets using Skellam score assumptions.

Input CSV: rows with columns
  game_id, mu_home, mu_away, spread, side, price_decimal
  - side: +1 for home -spread (bet home to cover), -1 for away +spread

Writes a CSV matrix of shape B x N (B scenarios, N bets) suitable for CVaR LP.

Usage:
  python py/risk/generate_scenarios.py --bets data/bets.csv --output data/scenarios.csv --sims 20000
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import List

import numpy as np

from ..models.score_distributions import skellam_pmf_range


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate scenario returns for CVaR sizing")
    ap.add_argument("--bets", required=True, help="CSV of bets (game_id, mu_home, mu_away, spread, side, price_decimal)")
    ap.add_argument("--output", required=True, help="Output CSV (B x N matrix)")
    ap.add_argument("--sims", type=int, default=20000, help="Number of Monte Carlo scenarios")
    return ap.parse_args()


def load_bets(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        return list(rdr)


def simulate_returns(bets: List[dict], sims: int) -> np.ndarray:
    # Precompute pmfs and sample margins
    rng = np.random.default_rng(42)
    margins_samples: List[np.ndarray] = []
    for b in bets:
        mu_h = float(b["mu_home"])
        mu_a = float(b["mu_away"])
        pmf = skellam_pmf_range(mu_h, mu_a, -80, 80)
        ks = np.array(list(pmf.keys()))
        ps = np.array([pmf[k] for k in ks])
        ps = ps / ps.sum()
        margins = rng.choice(ks, size=sims, p=ps)
        margins_samples.append(margins)
    # Compute per-bet returns per scenario
    B = sims
    N = len(bets)
    R = np.zeros((B, N), dtype=float)
    for j, b in enumerate(bets):
        spread = float(b["spread"])  # book spread wrt home
        side = float(b.get("side", 1.0))
        price = float(b.get("price_decimal", 1.91))
        net = price - 1.0
        margins = margins_samples[j]
        # Home covers if margin > -spread; away cover if margin < -spread
        cover = (margins > -spread) if side >= 0 else (margins < -spread)
        R[:, j] = np.where(cover, net, -1.0)
    return R


def main() -> None:
    args = parse_args()
    bets = load_bets(args.bets)
    if not bets:
        raise SystemExit("No bets provided")
    R = simulate_returns(bets, args.sims)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for i in range(R.shape[0]):
            w.writerow([f"{x:.6f}" for x in R[i, :]])
    print(f"[scenarios] wrote {R.shape[0]}x{R.shape[1]} -> {args.output}")


if __name__ == "__main__":
    main()

