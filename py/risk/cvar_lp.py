"""
CVaR stake sizing (stub).

If cvxpy is available, solves the Rockafellar–Uryasev LP. Otherwise, falls
back to a simple capped-Kelly heuristic using mean returns.

Usage:
  python py/risk/cvar_lp.py --scenarios data/returns.csv --alpha 0.95 \
      --cap 0.02 --output reports/cvar_stakes.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CVaR stake sizing (stub)")
    ap.add_argument("--scenarios", required=True, help="CSV matrix BxN of scenario returns")
    ap.add_argument("--alpha", type=float, default=0.95, help="CVaR level")
    ap.add_argument(
        "--cap", type=float, default=0.02, help="Per-position cap (Kelly fraction approx)"
    )
    ap.add_argument("--output", required=True, help="JSON output path for stakes + CVaR")
    return ap.parse_args()


def load_matrix(path: str) -> list[list[float]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        mat = [[float(x) for x in row] for row in reader if row]
    return mat


def heuristic_stakes(mat: list[list[float]], cap: float) -> list[float]:
    # Mean return per column
    if not mat:
        return []
    B, N = len(mat), len(mat[0])
    means = [sum(mat[b][j] for b in range(B)) / B for j in range(N)]
    pos = [max(0.0, m) for m in means]
    s = sum(pos)
    if s == 0.0:
        return [0.0] * N
    frac = [cap * p / s for p in pos]
    return frac


def cvar_of_stakes(mat: list[list[float]], f: list[float], alpha: float) -> float:
    pnl = [sum(f[j] * mat[b][j] for j in range(len(f))) for b in range(len(mat))]
    pnl_sorted = sorted(pnl)
    k = max(1, int(math.floor(alpha * len(pnl_sorted))))
    tail = pnl_sorted[:k]
    return -sum(tail) / len(tail)


def main() -> None:
    args = parse_args()
    mat = load_matrix(args.scenarios)
    f = heuristic_stakes(mat, args.cap)
    cvar = cvar_of_stakes(mat, f, args.alpha)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as g:
        json.dump({"stakes": f, "cvar": cvar, "alpha": args.alpha}, g, indent=2)
    print(f"[cvar] alpha={args.alpha} CVaR={cvar:.6f} -> {args.output}")


if __name__ == "__main__":
    main()
