"""
Teaser EV estimator: two-leg teasers via integer-margin PMF and dependence.

Features
- Build per-leg success probabilities from a discrete margin PMF (Skellam)
- Optional key-number reweighting (3, 6, 7, 10)
- Pair eligible basic-strategy legs and compute EV per bet
- Independence and Gaussian/t-copula joint success models
- Outputs:
  - TeX table of out-of-sample expected EV: analysis/dissertation/figures/out/teaser_ev_oos_table.tex
  - PNG heatmap of EV delta vs independence: analysis/dissertation/figures/out/teaser_pricing_copula_delta.png

Usage
  python py/pricing/teaser_ev.py \
    --start 2020 --end 2024 \
    --teaser 6.0 --price -120 \
    --tex analysis/dissertation/figures/out/teaser_ev_oos_table.tex \
    --png analysis/dissertation/figures/out/teaser_pricing_copula_delta.png
"""

from __future__ import annotations

import argparse
import os
import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import psycopg

from ..models.copula_gof import sim_t_copula
from ..models.copulas import joint_success_prob_gaussian
from ..models.score_distributions import (
    cover_push_probs,
    reweight_key_masses,
    reweight_with_moments,
    skellam_pmf_range,
)

# -------------------------
# Utility conversions
# -------------------------


def american_to_decimal(odds: float) -> float:
    if odds >= 100:
        return 1.0 + odds / 100.0
    if odds <= -100:
        return 1.0 + 100.0 / abs(odds)
    raise ValueError("American odds must be <= -100 or >= 100")


# -------------------------
# Database access
# -------------------------


def get_connection() -> psycopg.Connection:
    import os as _os

    host = _os.environ.get("POSTGRES_HOST", "localhost")
    port = int(_os.environ.get("POSTGRES_PORT", "5544"))
    dbname = _os.environ.get("POSTGRES_DB", "devdb01")
    user = _os.environ.get("POSTGRES_USER", "dro")
    password = _os.environ.get("POSTGRES_PASSWORD", "")
    return psycopg.connect(host=host, port=port, dbname=dbname, user=user, password=password)


def fetch_games(
    conn: psycopg.Connection, start_season: int, end_season: int
) -> list[dict[str, object]]:
    sql = """
        select season, week, home_team, away_team, spread_close, total_close, home_score, away_score
        from mart.game_summary
        where season between %s and %s
          and spread_close is not null and total_close is not null
        order by season, week, home_team
    """
    with conn.cursor() as cur:
        cur.execute(sql, (start_season, end_season))
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return rows


def fetch_key_targets(
    conn: psycopg.Connection, start_season: int, end_season: int
) -> dict[int, float]:
    """Empirical key-number masses from realized margins in a training window.

    Returns: dict like {3: p3, 6: p6, 7: p7, 10: p10}
    """
    sql = """
        select (home_score - away_score) as margin, count(*) as n
        from mart.game_summary
        where season between %s and %s
          and home_score is not null and away_score is not null
        group by 1
    """
    with conn.cursor() as cur:
        cur.execute(sql, (start_season, end_season))
        rows = cur.fetchall()
    counts: dict[int, int] = {}
    total = 0
    for m, n in rows:
        if m is None:
            continue
        mi = int(m)
        counts[mi] = counts.get(mi, 0) + int(n)
        total += int(n)
    targets = {}
    for k in (3, 6, 7, 10):
        targets[k] = (counts.get(k, 0) + counts.get(-k, 0)) / max(1, total)
    return targets


# -------------------------
# PMF and leg success
# -------------------------


def pmf_from_spread_total(
    total_close: float, spread_close: float, k_min: int = -80, k_max: int = 80
) -> dict[int, float]:
    """Construct integer-margin PMF via Skellam using (total, spread).

    Conventions
    - spread_close is the home spread; negative => home favorite, positive => home dog
    - Expected margin E[D] = -spread_close (home minus away)
    - Expected points total E[H+A] = total_close
    - Poisson means: mu_h = (total_close - spread_close) / 2, mu_a = (total_close + spread_close) / 2
    """
    mu_h = max(0.1, (total_close - spread_close) / 2.0)
    mu_a = max(0.1, (total_close + spread_close) / 2.0)
    return skellam_pmf_range(mu_h, mu_a, k_min=k_min, k_max=k_max)


def crosses_keys(th0: float, th1: float) -> bool:
    """Check if threshold moves across both 3 and 7 in its predominant direction.

    th0 is baseline success boundary; th1 is teased boundary for the same leg.
    We check that both integers at +/-{3,7} with the sign of the average threshold
    lie strictly between th0 and th1.
    """
    mid = 0.5 * (th0 + th1)
    sign = 1.0 if mid >= 0 else -1.0
    a, b = (th0, th1) if th0 < th1 else (th1, th0)
    return (a < sign * 3 < b) and (a < sign * 7 < b)


@dataclass
class Leg:
    game_idx: int
    side: str  # 'home' or 'away'
    spread: (
        float  # closing spread for that side (home spread if side=='home', away spread if 'away')
    )
    teased_spread: float
    q: float  # success probability under model


def leg_success_probs_for_game(
    pmf_margin: dict[int, float], spread_home: float, teaser: float
) -> list[tuple[str, float, float, float]]:
    """Return eligible legs with success probs: [(side, spread_side, teased, q)].

    Basic-strategy filter:
      - Home favorite teased down: include if home spread < 0 and crosses 3 & 7
      - Away underdog teased up: include if away spread > 0 and crosses 3 & 7
    """
    legs: list[tuple[str, float, float, float]] = []

    # Home side
    s_home = float(spread_home)
    s_home_tease = s_home + teaser
    th0_home = -s_home
    th1_home = -s_home_tease
    if s_home < 0 and crosses_keys(th0_home, th1_home):
        cover, push, _ = cover_push_probs(pmf_margin, s_home_tease)
        q_home = cover + 0.5 * push
        legs.append(("home", s_home, s_home_tease, q_home))

    # Away side
    s_away = -s_home
    s_away_tease = s_away + teaser
    # For away bets, success is D < s_away_tease (push handled at integer)
    th0_away = s_home  # since condition is D < s_away, threshold on D is s_home
    th1_away = s_home - teaser
    if s_away > 0 and crosses_keys(th0_away, th1_away):
        # P(D < s_away_tease) = 1 - P(D > s_away_tease) - P(D == s_away_tease)
        # Reuse cover/push with spread = -s_away_tease for home > threshold mapping
        cover_home, push_home, _ = cover_push_probs(pmf_margin, -s_away_tease)
        # Success prob: P(D < s_away_tease) + 0.5*P(D == s_away_tease)
        q_away = sum(p for m, p in pmf_margin.items() if m < s_away_tease)
        if float(s_away_tease).is_integer():
            q_away += 0.5 * pmf_margin.get(int(s_away_tease), 0.0)
        legs.append(("away", s_away, s_away_tease, q_away))

    return legs


# -------------------------
# Joint success under dependence
# -------------------------


def joint_success(
    q1: float,
    q2: float,
    model: str = "indep",
    rho: float = 0.0,
    nu: int = 5,
    nsim: int = 20000,
    rng: np.random.Generator | None = None,
) -> float:
    """P(both legs succeed) under selected dependence model.

    - indep: q1*q2
    - gaussian: Gaussian copula with correlation rho
    - t: Student-t copula with corr rho and dof nu (Monte Carlo)
    """
    if model == "indep":
        return q1 * q2
    if model == "gaussian":
        u1 = 1.0 - q1
        u2 = 1.0 - q2
        return joint_success_prob_gaussian(u1_thresh=u1, u2_thresh=u2, rho=rho)
    if model == "t":
        if rng is None:
            rng = np.random.default_rng(123)
        uu, vv = sim_t_copula(nsim, rho=float(rho), nu=int(nu), rng=rng)
        thr1 = 1.0 - q1
        thr2 = 1.0 - q2
        return float(np.mean((uu > thr1) & (vv > thr2)))
    raise ValueError(f"Unknown model: {model}")


def pairwise_ev(
    legs: Sequence[Leg],
    payout_decimal: float,
    dep_model: str = "indep",
    rho: float = 0.0,
    nu: int = 5,
) -> tuple[list[float], list[tuple[Leg, Leg]]]:
    """Pair legs sequentially and compute EV per pair. Drops the last if odd."""
    pairs: list[tuple[Leg, Leg]] = []
    evs: list[float] = []
    n = len(legs)
    for i in range(0, n - 1, 2):
        a = legs[i]
        b = legs[i + 1]
        p_both = joint_success(a.q, b.q, model=dep_model, rho=rho, nu=nu)
        ev = p_both * (payout_decimal - 1.0) - (1.0 - p_both)
        pairs.append((a, b))
        evs.append(ev)
    return evs, pairs


# -------------------------
# Reporting
# -------------------------


def write_tex_table(path: str, rows: list[tuple[str, float, float]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "% !TEX root = ../../main/main.tex",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\small",
        "  \\caption{Two-leg teaser EV on holdout.}",
        "  \\begin{tabular}{lrr}",
        "    \\toprule",
        r"    Model & Mean EV (bps) & ROI (\%) \\",
        "    \\midrule",
    ]
    for label, mean_ev, roi in rows:
        lines.append(f"    {label} & {mean_ev:.1f} & {roi:.2f} \\\\")
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_copula_delta_png(
    path: str, price_american: float, rho: float = 0.1, model: str = "gaussian"
) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = american_to_decimal(price_american)
    grid = np.linspace(0.60, 0.85, 101)
    Q1, Q2 = np.meshgrid(grid, grid)
    ev_indep = Q1 * Q2 * (d - 1.0) - (1.0 - Q1 * Q2)
    ev_dep = np.empty_like(ev_indep)
    rng = np.random.default_rng(123)
    for i in range(Q1.shape[0]):
        for j in range(Q1.shape[1]):
            q1 = float(Q1[i, j])
            q2 = float(Q2[i, j])
            if model == "gaussian":
                p_both = joint_success(q1, q2, model="gaussian", rho=rho)
            else:
                p_both = joint_success(q1, q2, model="t", rho=rho, nu=5, nsim=20000, rng=rng)
            ev_dep[i, j] = p_both * (d - 1.0) - (1.0 - p_both)
    delta = ev_dep - ev_indep
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    im = ax.imshow(
        delta,
        origin="lower",
        extent=[grid.min(), grid.max(), grid.min(), grid.max()],
        aspect="auto",
        cmap="coolwarm",
        vmin=-0.02,
        vmax=0.02,
    )
    ax.set_xlabel("Leg 1 win prob")
    ax.set_ylabel("Leg 2 win prob")
    ax.set_title(f"Copula EV delta (model={model}, rho={rho:+.2f}, price={int(price_american)})")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("EV delta (dep - indep)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_tex_sensitivity(
    path: str,
    indep_bps: float,
    indep_roi: float,
    gauss_rows: list[tuple[float, float, float]],
    t_rows: list[tuple[float, int, float, float]],
):
    """Write a sensitivity table covering indep baseline, Gaussian rho grid, and t-copula (rho, nu)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "% !TEX root = ../../main/main.tex",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\small",
        "  \\caption{Two-leg teaser EV sensitivity to dependence (Gaussian and t copulas).}",
        "  \\begin{tabular}{l l r r}",
        "    \\toprule",
        r"    Model & Param(s) & Mean EV (bps) & ROI (\%) \\",
        "    \\midrule",
        f"    Independence & -- & {indep_bps:.1f} & {indep_roi:.2f} \\\\",
        "    \\midrule",
    ]
    # Gaussian rows
    for rho, bps, roi in gauss_rows:
        lines.append(f"    Gaussian & $\\rho={rho:+.2f}$ & {bps:.1f} & {roi:.2f} \\\\")
    lines.append("    \\midrule")
    # t-copula rows
    for rho, nu, bps, roi in t_rows:
        lines.append(f"    $t$ & $\\rho={rho:+.2f},\\,\\nu={nu:d}$ & {bps:.1f} & {roi:.2f} \\\\")
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# -------------------------
# Main
# -------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Two-leg teaser EV estimator (OOS table + copula delta figure)"
    )
    ap.add_argument("--start", type=int, default=2020, help="Start season (inclusive)")
    ap.add_argument("--end", type=int, default=2024, help="End season (inclusive)")
    ap.add_argument("--train-start", type=int, default=1999, help="Training start for key masses")
    ap.add_argument("--train-end", type=int, default=2019, help="Training end for key masses")
    ap.add_argument("--teaser", type=float, default=6.0, help="Teaser points")
    ap.add_argument("--price", type=float, default=-120.0, help="American odds for 2-leg teaser")
    ap.add_argument(
        "--dep",
        type=str,
        default="indep",
        choices=["indep", "gaussian", "t"],
        help="Dependence model for pairing",
    )
    ap.add_argument("--rho", type=float, default=0.0, help="Correlation for copula models")
    ap.add_argument("--nu", type=int, default=5, help="DoF for t copula")
    ap.add_argument(
        "--tex", type=str, default="analysis/dissertation/figures/out/teaser_ev_oos_table.tex"
    )
    ap.add_argument(
        "--png",
        type=str,
        default="analysis/dissertation/figures/out/teaser_pricing_copula_delta.png",
    )
    ap.add_argument(
        "--sensitivity", action="store_true", help="Also emit sensitivity table over rho/nu grids"
    )
    ap.add_argument(
        "--rho-grid",
        default="-0.30,-0.20,-0.10,0.00,0.10,0.20,0.30",
        help="Comma-separated rho values for sensitivity",
    )
    ap.add_argument(
        "--nu-grid", default="3,5,10,30", help="Comma-separated nu values for t-copula sensitivity"
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    d = american_to_decimal(args.price)

    with get_connection() as conn:
        # Key mass targets from training data, symmetric over +/- margins
        targets = fetch_key_targets(conn, args.train_start, args.train_end)
        games = fetch_games(conn, args.start, args.end)

    legs_base: list[Leg] = []
    legs_rw: list[Leg] = []

    for idx, g in enumerate(games):
        s = float(g["spread_close"])  # home spread
        tclose = float(g["total_close"])
        pmf = pmf_from_spread_total(tclose, s)
        # Skellam parameters: mu_h = (total - spread)/2, mu_a = (total + spread)/2
        # E[D] = mu_h - mu_a = -spread, Var[D] = mu_h + mu_a = total
        mu_margin = -s
        var_margin = max(0.1, tclose)  # Skellam variance = sum of Poisson means
        pmf_rw = reweight_with_moments(pmf, targets, mu_margin, var_margin)
        # Eligible legs
        elig = leg_success_probs_for_game(pmf, s, args.teaser)
        elig_rw = leg_success_probs_for_game(pmf_rw, s, args.teaser)

        for side, s_side, s_tease, q in elig:
            legs_base.append(
                Leg(game_idx=idx, side=side, spread=s_side, teased_spread=s_tease, q=q)
            )
        for side, s_side, s_tease, q in elig_rw:
            legs_rw.append(Leg(game_idx=idx, side=side, spread=s_side, teased_spread=s_tease, q=q))

    # Stable ordering; then pair sequentially
    legs_base.sort(key=lambda x: (x.game_idx, x.side))
    legs_rw.sort(key=lambda x: (x.game_idx, x.side))

    # Compute EVs under independence baseline first
    evs_base_indep, pairs_base = pairwise_ev(legs_base, d, dep_model="indep", rho=0.0, nu=args.nu)
    evs_rw_indep, pairs_rw = pairwise_ev(legs_rw, d, dep_model="indep", rho=0.0, nu=args.nu)

    # If a dependence model is specified, compute EVs under that model
    evs_base_dep, pairs_base_dep = None, None
    if args.dep != "indep":
        evs_base_dep, pairs_base_dep = pairwise_ev(legs_base, d, dep_model=args.dep, rho=args.rho, nu=args.nu)

    def summarize(evs: Sequence[float]) -> tuple[float, float]:
        if not evs:
            return (0.0, 0.0)
        mean_ev = float(np.mean(evs))
        roi_pct = 100.0 * mean_ev
        bps = 10000.0 * mean_ev
        return bps, roi_pct

    bps_base_indep, roi_base_indep = summarize(evs_base_indep)
    bps_rw_indep, roi_rw_indep = summarize(evs_rw_indep)

    # Main table: always include independence baseline
    main_rows: list[tuple[str, float, float]] = [
        ("Independence", bps_base_indep, roi_base_indep),
        ("Independence + reweight", bps_rw_indep, roi_rw_indep),
    ]

    # Add dependence model row if specified
    if args.dep == "gaussian" and evs_base_dep is not None:
        bps_dep, roi_dep = summarize(evs_base_dep)
        main_rows.append((f"Gaussian (rho={args.rho:+.2f})", bps_dep, roi_dep))
    elif args.dep == "t" and evs_base_dep is not None:
        bps_dep, roi_dep = summarize(evs_base_dep)
        main_rows.append((f"t (rho={args.rho:+.2f}, nu={int(args.nu)})", bps_dep, roi_dep))
    write_tex_table(args.tex, rows=main_rows)

    # Copula delta heatmap figure (gaussian by default)
    model_for_png = "gaussian" if args.dep == "indep" else args.dep
    write_copula_delta_png(
        args.png, price_american=args.price, rho=args.rho or 0.1, model=model_for_png
    )

    # Optional sensitivity table
    if args.sensitivity:
        # Baseline indep values from independent pairing
        evs_indep, _ = pairwise_ev(legs_base, d, dep_model="indep", rho=0.0, nu=args.nu)
        indep_ev = float(np.mean(evs_indep)) if evs_indep else 0.0
        indep_bps = 10000.0 * indep_ev
        indep_roi = 100.0 * indep_ev
        # Gaussian grid
        rho_vals = [float(x) for x in args.rho_grid.split(",") if x.strip()]
        gauss_rows: list[tuple[float, float, float]] = []
        for r in rho_vals:
            evs_g, _ = pairwise_ev(legs_base, d, dep_model="gaussian", rho=r, nu=args.nu)
            m = float(np.mean(evs_g)) if evs_g else 0.0
            gauss_rows.append((r, 10000.0 * m, 100.0 * m))
        # t grid (cartesian of rho and nu)
        nu_vals = [int(float(x)) for x in args.nu_grid.split(",") if x.strip()]
        t_rows: list[tuple[float, int, float, float]] = []
        for r in rho_vals:
            for nu in nu_vals:
                evs_t, _ = pairwise_ev(legs_base, d, dep_model="t", rho=r, nu=nu)
                m = float(np.mean(evs_t)) if evs_t else 0.0
                t_rows.append((r, int(nu), 10000.0 * m, 100.0 * m))
        sens_path = os.path.join(os.path.dirname(args.tex), "teaser_ev_sensitivity_table.tex")
        write_tex_sensitivity(sens_path, indep_bps, indep_roi, gauss_rows, t_rows)


if __name__ == "__main__":
    main()
