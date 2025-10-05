"""
CVaR stake sizing via Rockafellar-Uryasev LP.

Solves the CVaR optimization problem:
    minimize CVaR_α(-PnL) subject to sum(stakes) ≤ budget and bounds

Uses cvxpy if available, otherwise falls back to a capped-Kelly heuristic.

Usage:
  python py/risk/cvar_lp.py --scenarios data/returns.csv --alpha 0.95 \
      --cap 0.02 --budget 1.0 --output reports/cvar_stakes.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os

try:
    import cvxpy as cp
    import numpy as np
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    np = None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="CVaR stake sizing via Rockafellar-Uryasev LP"
    )
    ap.add_argument("--scenarios", required=True, help="CSV matrix BxN of scenario returns")
    ap.add_argument("--alpha", type=float, default=0.95, help="CVaR level (e.g., 0.95)")
    ap.add_argument(
        "--cap", type=float, default=0.02, help="Per-position stake cap (fraction)"
    )
    ap.add_argument(
        "--budget", type=float, default=1.0, help="Total budget (sum of stakes ≤ budget)"
    )
    ap.add_argument("--output", required=True, help="JSON output path for stakes + CVaR")
    ap.add_argument(
        "--solver", default="CLARABEL", help="cvxpy solver (CLARABEL, OSQP, SCS, ECOS)"
    )
    return ap.parse_args()


def load_matrix(path: str) -> list[list[float]]:
    """Load scenario return matrix from CSV (rows=scenarios, cols=bets)."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        mat = [[float(x) for x in row] for row in reader if row]
    return mat


def solve_cvar_lp(
    returns: np.ndarray,
    alpha: float,
    cap: float,
    budget: float,
    solver: str = "CLARABEL"
) -> tuple[np.ndarray, float, float]:
    """
    Solve CVaR optimization using Rockafellar-Uryasev LP formulation.

    Minimize:
        VaR + (1/(1-α)) * E[max(0, -PnL - VaR)]

    Subject to:
        0 ≤ stakes ≤ cap
        sum(stakes) ≤ budget

    Args:
        returns: BxN matrix of scenario returns (rows=scenarios, cols=bets)
        alpha: CVaR confidence level (e.g., 0.95 for CVaR95)
        cap: Maximum stake per position
        budget: Maximum total stake
        solver: cvxpy solver name

    Returns:
        stakes: Optimal stake allocation (length N)
        cvar: CVaR value at optimal stakes
        var: VaR value at optimal stakes
    """
    B, N = returns.shape

    # Decision variables
    f = cp.Variable(N, nonneg=True)  # stakes
    z = cp.Variable()  # VaR (Value at Risk)
    u = cp.Variable(B, nonneg=True)  # auxiliary variables for CVaR

    # Portfolio PnL for each scenario: returns @ f
    pnl = returns @ f

    # Rockafellar-Uryasev formulation:
    # CVaR_α(-PnL) = z + (1/(1-α)) * mean(u)
    # where u_i ≥ -pnl_i - z  and  u_i ≥ 0
    constraints = [
        u >= -pnl - z,  # u_i ≥ -pnl_i - z for all scenarios
        f <= cap,  # per-position cap
        cp.sum(f) <= budget,  # total budget constraint
    ]

    # Objective: minimize CVaR of losses
    cvar_term = z + (1.0 / (1.0 - alpha)) * cp.sum(u) / B
    objective = cp.Minimize(cvar_term)

    # Solve
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=solver, verbose=False)
    except Exception:
        # Fallback to ECOS if specified solver fails
        try:
            problem.solve(solver="ECOS", verbose=False)
        except Exception:
            # Last resort: SCS
            problem.solve(solver="SCS", verbose=False)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"CVaR LP solver failed with status: {problem.status}")

    stakes = f.value
    var_value = z.value
    cvar_value = cvar_term.value

    return stakes, cvar_value, var_value


def heuristic_stakes(mat: list[list[float]], cap: float, budget: float) -> list[float]:
    """
    Fallback heuristic when cvxpy is not available.

    Allocates stakes proportional to positive mean returns, respecting cap and budget.
    """
    if not mat:
        return []
    B, N = len(mat), len(mat[0])
    means = [sum(mat[b][j] for b in range(B)) / B for j in range(N)]
    pos = [max(0.0, m) for m in means]
    s = sum(pos)
    if s == 0.0:
        return [0.0] * N
    # Scale to respect budget
    scale = min(budget / s, 1.0)
    frac = [min(cap, scale * p) for p in pos]
    return frac


def cvar_of_stakes(mat: list[list[float]], f: list[float], alpha: float) -> tuple[float, float]:
    """
    Calculate CVaR and VaR for given stakes.

    Returns:
        cvar: CVaR_α of portfolio losses
        var: VaR_α of portfolio losses
    """
    pnl = [sum(f[j] * mat[b][j] for j in range(len(f))) for b in range(len(mat))]
    losses = [-p for p in pnl]  # Convert PnL to losses
    losses_sorted = sorted(losses)

    # VaR: α-quantile of loss distribution
    var_idx = int(math.ceil(alpha * len(losses_sorted))) - 1
    var_value = losses_sorted[var_idx]

    # CVaR: expected loss beyond VaR
    tail = [l for l in losses if l >= var_value]
    cvar_value = sum(tail) / len(tail) if tail else var_value

    return cvar_value, var_value


def main() -> None:
    args = parse_args()
    mat = load_matrix(args.scenarios)

    if not mat or not mat[0]:
        raise ValueError("Empty scenario matrix")

    B, N = len(mat), len(mat[0])

    if HAS_CVXPY:
        # Optimal CVaR LP solution
        returns_np = np.array(mat)
        stakes, cvar, var = solve_cvar_lp(
            returns_np, args.alpha, args.cap, args.budget, args.solver
        )
        stakes_list = stakes.tolist()
        method = "cvxpy_lp"
    else:
        # Fallback heuristic
        stakes_list = heuristic_stakes(mat, args.cap, args.budget)
        cvar, var = cvar_of_stakes(mat, stakes_list, args.alpha)
        method = "heuristic"

    # Write output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result = {
        "method": method,
        "stakes": stakes_list,
        "cvar": float(cvar),
        "var": float(var),
        "alpha": args.alpha,
        "cap": args.cap,
        "budget": args.budget,
        "n_bets": N,
        "n_scenarios": B,
        "total_stake": sum(stakes_list),
    }

    with open(args.output, "w", encoding="utf-8") as g:
        json.dump(result, g, indent=2)

    print(f"[cvar] method={method} α={args.alpha} CVaR={cvar:.6f} VaR={var:.6f} "
          f"total_stake={sum(stakes_list):.4f} -> {args.output}")


if __name__ == "__main__":
    main()
