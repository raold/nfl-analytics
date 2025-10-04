"""
Simulator acceptance test.

Computes acceptance metrics comparing simulated outputs to historical targets:
  - 1D EMD on margin pmf (integer lattice)
  - Max |delta| across key-number masses (3, 6, 7, 10)
  - Dependence metric: Kendall's tau on provided pairs or joint upper-tail rate
  - Friction RMSE on slippage
  - Fill shortfall (fractional short of requested quantity)

Writes a compact JSON report and (optionally) a small TeX table for Chapter 7.

Usage:
  python py/sim/acceptance.py --hist hist.json --sim sim.json \
    --output analysis/reports/sim_accept.json \
    --tex analysis/dissertation/results/sim_acceptance_table.tex
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Simulator acceptance test")
    ap.add_argument("--hist", required=True, help="Historical metrics JSON")
    ap.add_argument("--sim", required=True, help="Simulated metrics JSON")
    ap.add_argument("--output", required=True, help="Output JSON report")
    ap.add_argument("--tau-marg", type=float, default=0.05, help="Tolerance for EMD(margin)")
    ap.add_argument("--tau-key", type=float, default=0.01, help="Tolerance for max |Δ key mass|")
    ap.add_argument("--tau-fric", type=float, default=0.5, help="Tolerance for slippage RMSE")
    ap.add_argument(
        "--tau-fillshort", type=float, default=0.1, help="Tolerance for fill shortfall (fraction)"
    )
    ap.add_argument("--tex", help="Optional TeX table output path")
    return ap.parse_args()


def load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _mk_pmf(data: dict[str, Any], lo: int = -80, hi: int = 80) -> dict[int, float]:
    # Accept either explicit pmf or raw margins
    if "margin_pmf" in data and isinstance(data["margin_pmf"], dict):
        pmf = {int(k): float(v) for k, v in data["margin_pmf"].items()}
        s = sum(v for v in pmf.values() if v >= 0)
        return {k: (v / s if s > 0 else 0.0) for k, v in pmf.items()}
    margins = data.get("margins") or []
    # Build counts on integer lattice
    counts: dict[int, int] = {k: 0 for k in range(lo, hi + 1)}
    for m in margins:
        k = int(round(float(m)))
        if lo <= k <= hi:
            counts[k] += 1
    total = sum(counts.values())
    return {k: (counts[k] / total if total else 0.0) for k in range(lo, hi + 1)}


def _emd_1d(p: dict[int, float], q: dict[int, float]) -> float:
    # Earth Mover's Distance for 1D histograms on shared support
    supp = sorted(set(p.keys()) | set(q.keys()))
    cumsum = 0.0
    emd = 0.0
    for k in supp:
        cumsum += p.get(k, 0.0) - q.get(k, 0.0)
        emd += abs(cumsum)
    return emd


def _key_mass_delta(p: dict[int, float], q: dict[int, float], keys: list[int]) -> float:
    return max(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys) if keys else 0.0


def _kendall_tau(x: list[float], y: list[float]) -> float:
    n = min(len(x), len(y))
    if n <= 1:
        return 0.0
    # Subsample if very large
    if n > 3000:
        step = max(1, n // 3000)
        x = x[::step]
        y = y[::step]
        n = len(x)
    conc = 0
    disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (x[i] - x[j]) * (y[i] - y[j])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    denom = conc + disc
    return (conc - disc) / denom if denom > 0 else 0.0


def _dependence_metric(hist: dict[str, Any], sim: dict[str, Any]) -> tuple[str, float]:
    # Prefer Kendall's tau if paired arrays present; else compute joint upper-tail rate gap
    hx = hist.get("x") or hist.get("u1") or hist.get("spread_prob")
    hy = hist.get("y") or hist.get("u2") or hist.get("total_prob")
    sx = sim.get("x") or sim.get("u1") or sim.get("spread_prob")
    sy = sim.get("y") or sim.get("u2") or sim.get("total_prob")
    if isinstance(hx, list) and isinstance(hy, list):
        tau_h = _kendall_tau([float(v) for v in hx], [float(v) for v in hy])
        tau_s = _kendall_tau([float(v) for v in (sx or [])], [float(v) for v in (sy or [])])
        return ("kendall_tau_delta", abs(tau_h - tau_s))
    # Upper-tail (u>0.9) joint exceedance rate
    hu = hist.get("u_pairs")
    su = sim.get("u_pairs")

    def _tail_rate(pairs) -> float:
        if not isinstance(pairs, list):
            return 0.0
        c = 0
        n = 0
        for a, b in pairs:
            try:
                if float(a) > 0.9 and float(b) > 0.9:
                    c += 1
                n += 1
            except Exception:
                continue
        return c / n if n else 0.0

    return ("joint_tail_delta", abs(_tail_rate(hu) - _tail_rate(su)))


def compare(hist: dict[str, Any], sim: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    # Margins
    pmf_h = _mk_pmf(hist)
    pmf_s = _mk_pmf(sim)
    emd_margin = _emd_1d(pmf_h, pmf_s)
    key_delta = _key_mass_delta(pmf_h, pmf_s, [3, 6, 7, 10])

    # Dependence delta
    dep_name, dep_delta = _dependence_metric(hist, sim)

    # Friction RMSE
    def _rmse(a: list[float], b: list[float]) -> float:
        import math

        n = min(len(a), len(b))
        if n == 0:
            return float(sim.get("slippage_rmse", 0.0))
        s = 0.0
        for i in range(n):
            s += (float(a[i]) - float(b[i])) ** 2
        return math.sqrt(s / n)

    fric_rmse = float(sim.get("slippage_rmse", 0.0))
    if isinstance(hist.get("slippage"), list) and isinstance(sim.get("slippage"), list):
        fric_rmse = _rmse(hist["slippage"], sim["slippage"])

    # Fill shortfall
    fill_short = float(sim.get("fill_shortfall", 0.0))
    if isinstance(sim.get("requested"), list) and isinstance(sim.get("filled"), list):
        rq = sum(float(x) for x in sim["requested"]) or 1.0
        fl = sum(float(x) for x in sim["filled"]) or 0.0
        fill_short = max(0.0, 1.0 - (fl / rq))

    checks = {
        "emd_margin": {
            "value": emd_margin,
            "tau": args.tau_marg,
            "ok": emd_margin <= args.tau_marg,
        },
        "key_mass_max_delta": {
            "value": key_delta,
            "tau": args.tau_key,
            "ok": key_delta <= args.tau_key,
        },
        dep_name: {"value": dep_delta, "tau": args.tau_key, "ok": dep_delta <= args.tau_key},
        "slippage_rmse": {
            "value": fric_rmse,
            "tau": args.tau_fric,
            "ok": fric_rmse <= args.tau_fric,
        },
        "fill_shortfall": {
            "value": fill_short,
            "tau": args.tau_fillshort,
            "ok": fill_short <= args.tau_fillshort,
        },
    }
    passed = all(v["ok"] for v in checks.values())
    return {"pass": passed, "checks": checks}


def _write_tex(path: str, rep: dict[str, Any]) -> None:
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    order = [
        ("emd_margin", "EMD (margin)"),
        ("key_mass_max_delta", "Max |$\\Delta$ key mass|"),
        ("kendall_tau_delta", "Kendall's $\\tau$ $\\Delta$"),
        ("joint_tail_delta", "Joint tail $\\Delta$"),
        ("slippage_rmse", "Slippage RMSE"),
        ("fill_shortfall", "Fill shortfall"),
    ]
    rows = []
    for key, label in order:
        chk = rep["checks"].get(key)
        if not chk:
            continue
        rows.append((label, float(chk["value"]), float(chk["tau"]), "Yes" if chk["ok"] else "No"))
    with open(path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by py/sim/acceptance.py\n")
        f.write("\\begin{table}[t]\n  \\centering\n  \\footnotesize\n  ")
        f.write("\\begin{threeparttable}\n    ")
        f.write(
            "\\caption[Simulator acceptance checks]{Simulator acceptance metrics vs tolerances. Pass=\\textbf{%s}.}\n"
            % ("Yes" if rep.get("pass") else "No")
        )
        f.write("\\label{tab:sim-accept}\n    ")
        f.write("\\setlength{\\tabcolsep}{3pt}\\renewcommand{\\arraystretch}{1.1}\n")
        f.write("\\begin{tabularx}{\\linewidth}{@{} l r r c @{} }\n    \\toprule\n")
        f.write("    Metric & Value & Tolerance & Pass \\\\ \n    \\midrule\n")
        for label, val, tau, ok in rows:
            f.write(f"{label} & {val:.4f} & {tau:.4f} & {ok} \\\\ \n")
        f.write("    \\bottomrule\n")
        f.write("  \\end{tabularx}\n")
        f.write("\\end{threeparttable}\n\\end{table}\n")


def main() -> None:
    args = parse_args()
    hist = load_json(args.hist)
    sim = load_json(args.sim)
    rep = compare(hist, sim, args)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    if args.tex:
        _write_tex(args.tex, rep)
    print(
        f"[sim-accept] pass={rep['pass']} -> {args.output}"
        + (f"; tex -> {args.tex}" if args.tex else "")
    )


if __name__ == "__main__":
    main()
