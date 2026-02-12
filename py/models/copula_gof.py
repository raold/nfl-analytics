"""
Copula GOF (Gaussian vs t) and tail-dependence estimator from spread/total pairs.

Reads a CSV with closing spread/total (defaults to analysis/features/asof_team_features.csv),
computes pseudo-observations via empirical CDF, fits Gaussian and t copulas, and outputs:

- GOF table using a tail-focused CvM-like statistic (SSE over joint tail exceedance rates
  at thresholds {0.80, 0.90, 0.95} between empirical and fitted-copula simulation), with
  parametric bootstrap p-values.
- Tail dependence estimates (empirical) with bootstrap 95% CIs for upper/lower tails.

Example:
  python py/models/copula_gof.py \
    --csv analysis/features/asof_team_features.csv \
    --start 2020 --end 2024 \
    --tex-gof analysis/dissertation/figures/out/copula_gof_table.tex \
    --tex-tail analysis/dissertation/figures/out/tail_dependence_table.tex
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import norm, t


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Copula GOF + tail dependence from spread/total pairs")
    ap.add_argument("--csv", default="analysis/features/asof_team_features.csv")
    ap.add_argument("--start", type=int, default=2020)
    ap.add_argument("--end", type=int, default=2024)
    ap.add_argument("--tex-gof", required=True)
    ap.add_argument("--tex-tail", required=True)
    ap.add_argument("--nsim", type=int, default=50000)
    ap.add_argument("--nboot", type=int, default=300)
    return ap.parse_args()


def load_pairs(path: str, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    df = df[(df["season"] >= start) & (df["season"] <= end)].copy()
    df = df[["spread_close", "total_close"]].dropna()
    x = df["spread_close"].to_numpy(dtype=float)
    y = df["total_close"].to_numpy(dtype=float)
    return x, y


def rank_pobs(x: np.ndarray) -> np.ndarray:
    n = x.size
    ranks = pd.Series(x).rank(method="average").to_numpy()
    u = (ranks - 0.5) / n
    u = np.clip(u, 1e-6, 1 - 1e-6)
    return u


@dataclass
class FitGaussian:
    rho: float


@dataclass
class FitTCopula:
    rho: float
    nu: float


def fit_gaussian(u: np.ndarray, v: np.ndarray) -> FitGaussian:
    z = norm.ppf(u)
    w = norm.ppf(v)
    rho = float(np.corrcoef(z, w)[0, 1])
    rho = float(np.clip(rho, -0.999, 0.999))
    return FitGaussian(rho=rho)


def kendall_tau(u: np.ndarray, v: np.ndarray) -> float:
    # Simple Kendall tau-b via numpy (O(n^2)), fallback to approx for large n
    n = len(u)
    if n > 4000:
        idx = np.random.default_rng(42).choice(n, size=4000, replace=False)
        u = u[idx]
        v = v[idx]
        n = len(u)
    conc = 0
    disc = 0
    for i in range(n):
        du = u[i] - u[i + 1 :]
        dv = v[i] - v[i + 1 :]
        s = du * dv
        conc += np.count_nonzero(s > 0)
        disc += np.count_nonzero(s < 0)
    denom = conc + disc
    return (conc - disc) / denom if denom > 0 else 0.0


def fit_t(u: np.ndarray, v: np.ndarray) -> FitTCopula:
    tau = kendall_tau(u, v)
    rho = float(np.sin(np.pi * tau / 2.0))  # valid for elliptical copulas
    rho = float(np.clip(rho, -0.999, 0.999))
    # Grid search nu by pseudo-likelihood
    nus = np.arange(3, 31)  # 3..30
    best_ll = -np.inf
    best_nu = 5
    for nu in nus:
        ll = t_copula_loglik(u, v, rho, nu)
        if np.isfinite(ll) and ll > best_ll:
            best_ll = ll
            best_nu = int(nu)
    return FitTCopula(rho=rho, nu=best_nu)


def t_copula_loglik(u: np.ndarray, v: np.ndarray, rho: float, nu: int) -> float:
    x = t.ppf(u, df=nu)
    y = t.ppf(v, df=nu)
    # Sigma and inverse
    det = 1.0 - rho * rho
    if det <= 1e-12:
        return -np.inf
    inv = (1.0 / det) * np.array([[1.0, -rho], [-rho, 1.0]])
    # Quadratic form for each observation
    a = inv[0, 0]
    b = inv[0, 1]
    c = inv[1, 1]
    quad = a * (x * x) + 2.0 * b * (x * y) + c * (y * y)
    # log joint t density in 2D
    d = 2
    lg = (
        gammaln((nu + d) / 2.0)
        - gammaln(nu / 2.0)
        - (d / 2.0) * (np.log(nu) + np.log(np.pi))
        - 0.5 * np.log(det)
        - ((nu + d) / 2.0) * np.log(1.0 + quad / nu)
    )
    # log marginals t density
    lmx = (
        gammaln((nu + 1) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * (np.log(nu) + np.log(np.pi))
        - ((nu + 1) / 2.0) * np.log(1.0 + (x * x) / nu)
    )
    lmy = (
        gammaln((nu + 1) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * (np.log(nu) + np.log(np.pi))
        - ((nu + 1) / 2.0) * np.log(1.0 + (y * y) / nu)
    )
    ll = float(np.sum(lg - (lmx + lmy)))
    return ll


def sim_gaussian(n: int, rho: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    z1 = rng.standard_normal(n)
    z2 = rho * z1 + np.sqrt(max(1e-12, 1.0 - rho * rho)) * rng.standard_normal(n)
    return norm.cdf(z1), norm.cdf(z2)


def sim_t_copula(
    n: int, rho: float, nu: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    # Draw bivariate normal with corr rho
    z1 = rng.standard_normal(n)
    z2 = rho * z1 + np.sqrt(max(1e-12, 1.0 - rho * rho)) * rng.standard_normal(n)
    # Scale by sqrt(S) with S ~ Chi^2_nu / nu
    s = rng.chisquare(df=nu, size=n) / float(nu)
    x1 = z1 / np.sqrt(s)
    x2 = z2 / np.sqrt(s)
    return t.cdf(x1, df=nu), t.cdf(x2, df=nu)


def tail_gof_stat(
    u: np.ndarray,
    v: np.ndarray,
    model: str,
    rho: float,
    nu: int | None,
    nsim: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    # Empirical tail exceedance at thresholds
    ths = np.array([0.80, 0.90, 0.95])

    def emp_tail(th: float) -> float:
        return float(np.mean((u > th) & (v > th)))

    emp = np.array([emp_tail(t0) for t0 in ths])
    # Model tail via simulation
    if model == "gaussian":
        uu, vv = sim_gaussian(nsim, rho, rng)
    else:
        assert nu is not None
        uu, vv = sim_t_copula(nsim, rho, int(nu), rng)
    mod = np.array([float(np.mean((uu > t0) & (vv > t0))) for t0 in ths])
    sse = float(np.sum((emp - mod) ** 2))
    # Parametric bootstrap p-value: replicate SSE under model
    B = 200
    reps = []
    n = len(u)
    for _ in range(B):
        if model == "gaussian":
            r1, r2 = sim_gaussian(n, rho, rng)
        else:
            r1, r2 = sim_t_copula(n, rho, int(nu), rng)
        # compute empirical tail of replicate and compare to model tail
        emp_r = np.array([float(np.mean((r1 > t0) & (r2 > t0))) for t0 in ths])
        reps.append(float(np.sum((emp_r - mod) ** 2)))
    reps = np.array(reps)
    pval = float(np.mean(reps >= sse))
    return sse, pval


def tail_dependence(
    u: np.ndarray, v: np.ndarray, nboot: int, rng: np.random.Generator
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    # Empirical lambda_U/L using threshold t=0.95
    t0 = 0.95

    def lam_u(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean((a > t0) & (b > t0)) / (1.0 - t0))

    def lam_l(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean((a <= (1.0 - t0)) & (b <= (1.0 - t0))) / (1.0 - t0))

    lu = lam_u(u, v)
    ll = lam_l(u, v)
    nus = u.size
    bu = []
    bl = []
    for _ in range(nboot):
        idx = rng.integers(0, nus, size=nus)
        uu = u[idx]
        vv = v[idx]
        bu.append(lam_u(uu, vv))
        bl.append(lam_l(uu, vv))
    lo_u, hi_u = float(np.percentile(bu, 2.5)), float(np.percentile(bu, 97.5))
    lo_l, hi_l = float(np.percentile(bl, 2.5)), float(np.percentile(bl, 97.5))
    return (lu, lo_u, hi_u), (ll, lo_l, hi_l)


def write_tex_gof(
    path: str,
    sse_gauss: float,
    p_gauss: float,
    sse_t: float,
    p_t: float,
    rho_g: float,
    rho_t: float,
    nu_t: int,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tex = (
        "\\begin{table}[t]\n"
        "  \\centering\n"
        "  \\small\n"
        "  \\caption{Copula GOF (tail CvM; thresholds 0.80/0.90/0.95).}\n"
        "  \\begin{tabular}{lccc}\n"
        "    \\toprule\n"
        "    Copula & CvM stat & p-value & params \\\\\n"
        "    \\midrule\n"
        f"    Gaussian & {sse_gauss:.4f} & {p_gauss:.3f} & $\\rho={rho_g:.2f}$ \\\\\n"
        f"    $t$ & {sse_t:.4f} & {p_t:.3f} & $\\rho={rho_t:.2f},\\,\\nu={nu_t}$ \\\\\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)


def write_tex_tail(
    path: str, upper: tuple[float, float, float], lower: tuple[float, float, float]
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lu, lu_lo, lu_hi = upper
    ll, ll_lo, ll_hi = lower
    tex = (
        "\\begin{table}[t]\n"
        "  \\centering\n"
        "  \\small\n"
        "  \\caption{Tail dependence estimates with 95\\% CIs.}\n"
        "  \\begin{tabular}{lcc}\n"
        "    \\toprule\n"
        "    Tail & estimate & 95\\% CI \\\\\n"
        "    \\midrule\n"
        f"    Upper ($\\lambda_U$) & {lu:.3f} & [{lu_lo:.3f},\\,{lu_hi:.3f}] \\\\\n"
        f"    Lower ($\\lambda_L$) & {ll:.3f} & [{ll_lo:.3f},\\,{ll_hi:.3f}] \\\\\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(12345)
    x, y = load_pairs(args.csv, args.start, args.end)
    # Pseudo-observations
    u = rank_pobs(x)
    v = rank_pobs(y)

    # Fit copulas
    g = fit_gaussian(u, v)
    tfit = fit_t(u, v)

    # Tail-focused GOF via simulation
    sse_g, p_g = tail_gof_stat(u, v, "gaussian", g.rho, None, args.nsim, rng)
    sse_t, p_t = tail_gof_stat(u, v, "t", tfit.rho, tfit.nu, args.nsim, rng)

    # Tail dependence (empirical)
    upper, lower = tail_dependence(u, v, nboot=args.nboot, rng=rng)

    write_tex_gof(args.tex_gof, sse_g, p_g, sse_t, p_t, g.rho, tfit.rho, int(tfit.nu))
    write_tex_tail(args.tex_tail, upper, lower)
    print(
        f"[copula] rho_G={g.rho:.3f}; rho_t={tfit.rho:.3f}, nu={tfit.nu}; "
        f"GOF sse_G={sse_g:.4f} p={p_g:.3f}; sse_t={sse_t:.4f} p={p_t:.3f} \n"
        f"[copula] tail: lambda_U={upper[0]:.3f} [{upper[1]:.3f},{upper[2]:.3f}], "
        f"lambda_L={lower[0]:.3f} [{lower[1]:.3f},{lower[2]:.3f}]"
    )


if __name__ == "__main__":
    main()
