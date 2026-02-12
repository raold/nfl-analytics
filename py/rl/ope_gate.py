"""
Offline RL promotion gate.

Reads a one-step logged dataset CSV and computes SNIS and DR estimates across
grids of clipping and shrinkage. Emits a JSON report with stability checks and
an accept/reject decision. Optionally writes a small TeX table summarizing the grid.

Dataset CSV must include columns: action, r, b_prob, pi_prob; optional: edge

Usage:
  python py/rl/ope_gate.py --dataset data/rl_logged.csv --output reports/ope_gate.json \
      --grid-clips 5,10,20 --grid-shrinks 0.0,0.2,0.5 --alpha 0.05 \
      --tex analysis/dissertation/results/ope_grid_table.tex
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import pandas as pd

try:
    from .ope import dr, snis
except Exception:  # allow running as a script without packages
    import importlib.util
    from pathlib import Path

    OPE_PATH = Path(__file__).resolve().parent / "ope.py"
    spec = importlib.util.spec_from_file_location("ope", str(OPE_PATH))
    _mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(_mod)
    snis, dr = _mod.snis, _mod.dr


@dataclass
class GateConfig:
    clips: list[float]
    shrinks: list[float]
    alpha: float = 0.05
    ess_threshold: float = 30.0  # Minimum ESS for acceptance
    n_bootstrap: int = 1000  # Bootstrap iterations for CI


@dataclass
class GateResult:
    accept: bool
    median_dr: float
    stable: bool
    grid: dict[str, dict[str, float]]
    note: str
    dr_lower_ci: float  # Lower confidence bound on DR
    dr_upper_ci: float  # Upper confidence bound on DR
    min_ess: float  # Minimum ESS across grid
    reason_codes: list[str]  # Rejection reasons if not accepted


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Offline RL OPE promotion gate (stub)")
    ap.add_argument("--dataset", required=True, help="Path to logged dataset (stub)")
    ap.add_argument("--policy", required=True, help="Path to candidate policy artifact (stub)")
    ap.add_argument("--output", required=True, help="Path to JSON report output")
    ap.add_argument("--grid-clips", default="5,10,20", help="Comma-separated clip thresholds")
    ap.add_argument("--grid-shrinks", default="0.5,0.8,1.0", help="Comma-separated shrink scales")
    ap.add_argument("--alpha", type=float, default=0.05, help="Lower-bound level")
    ap.add_argument("--tex", help="Optional TeX table output path")
    return ap.parse_args()


def _grid(values: str) -> list[float]:
    return [float(x.strip()) for x in values.split(",") if x.strip()]


def _bootstrap_dr_ci(
    df: pd.DataFrame, clip: float, shrink: float, n_boot: int, alpha: float
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for DR estimate."""
    import numpy as np

    boot_vals = []
    n = len(df)
    for _ in range(n_boot):
        # Resample with replacement
        boot_idx = np.random.choice(n, size=n, replace=True)
        df_boot = df.iloc[boot_idx].copy()
        de = dr(df_boot, clip=clip, shrink=shrink)
        boot_vals.append(de["value"])

    # Percentile method
    lower = float(np.percentile(boot_vals, 100 * alpha / 2))
    upper = float(np.percentile(boot_vals, 100 * (1 - alpha / 2)))
    return lower, upper


def evaluate_grid(
    df: pd.DataFrame, clips: list[float], shrinks: list[float], cfg: GateConfig
) -> tuple[float, dict[str, dict[str, float]], bool, float, float, float]:
    """Evaluate OPE grid with bootstrap CIs and ESS checks.

    Returns: (median_dr, grid, stable, dr_lower, dr_upper, min_ess)
    """
    grid: dict[str, dict[str, float]] = {}
    dr_vals: list[float] = []
    ess_vals: list[float] = []

    # Find best (clip, shrink) by median DR
    best_clip, best_shrink = clips[len(clips) // 2], shrinks[len(shrinks) // 2]
    best_dr = float("-inf")

    for c in clips:
        for s in shrinks:
            key = f"c{int(c)}_s{str(s).replace('.', '')}"
            sn = snis(df, clip=c, shrink=s)
            de = dr(df, clip=c, shrink=s)
            grid[key] = {"snis": sn["value"], "dr": de["value"], "ess": sn["ess"]}
            dr_vals.append(de["value"])
            ess_vals.append(sn["ess"])

            # Track best hyperparameters
            if de["value"] > best_dr:
                best_dr = de["value"]
                best_clip, best_shrink = c, s

    dr_vals_sorted = sorted(dr_vals)
    median_dr = dr_vals_sorted[len(dr_vals_sorted) // 2]
    min_ess = min(ess_vals) if ess_vals else 0.0

    # Stability: require all DR values around the top quartile to share sign
    top = [v for v in dr_vals_sorted[int(0.75 * len(dr_vals_sorted)) :]] or dr_vals_sorted
    signs = [v > 0.0 for v in top]
    stable = all(signs) or not any(signs)

    # Bootstrap CI on best hyperparameters
    dr_lower, dr_upper = _bootstrap_dr_ci(df, best_clip, best_shrink, cfg.n_bootstrap, cfg.alpha)

    return median_dr, grid, stable, dr_lower, dr_upper, min_ess


def run_gate(df: pd.DataFrame, cfg: GateConfig) -> GateResult:
    median_dr, grid, stable, dr_lower, dr_upper, min_ess = evaluate_grid(
        df, cfg.clips, cfg.shrinks, cfg
    )

    # Acceptance criteria with reason codes
    reason_codes = []
    accept = True

    if not stable:
        accept = False
        reason_codes.append("unstable_grid")

    if min_ess < cfg.ess_threshold:
        accept = False
        reason_codes.append(f"low_ess_{min_ess:.1f}")

    if dr_lower <= 0.0:
        accept = False
        reason_codes.append(f"negative_lcb_{dr_lower:.4f}")

    if median_dr <= 0.0:
        accept = False
        reason_codes.append(f"negative_median_{median_dr:.4f}")

    note = (
        f"OPE gate with bootstrap CI (alpha={cfg.alpha}), ESS threshold={cfg.ess_threshold}. "
        f"Accept requires: stable grid, ESS >= threshold, LCB > 0, median > 0."
    )

    return GateResult(
        accept=accept,
        median_dr=median_dr,
        stable=stable,
        grid=grid,
        note=note,
        dr_lower_ci=dr_lower,
        dr_upper_ci=dr_upper,
        min_ess=min_ess,
        reason_codes=reason_codes if not accept else [],
    )


def _write_tex_table(path: str, res: GateResult, cfg: GateConfig) -> None:
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    for c in cfg.clips:
        for s in cfg.shrinks:
            key = f"c{int(c)}_s{str(s).replace('.', '')}"
            cell = res.grid.get(key, {"snis": 0.0, "dr": 0.0, "ess": 0.0})
            rows.append((c, s, cell["snis"], cell["dr"], cell["ess"]))
    with open(path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by py/rl/ope_gate.py\n")
        f.write("% !TEX root = ../../main/main.tex\n")
        f.write("\\providecommand{\\opeGridLabel}{\\label{tab:ope-grid}}\n")
        f.write("\\begin{table}[t]\n  \\centering\n  \\footnotesize\n  ")
        f.write("\\begin{threeparttable}\n    ")
        caption = (
            "Off-policy evaluation grid: SNIS and DR values with effective sample sizes (ESS). "
            f"Accept=\\textbf{{{'Yes' if res.accept else 'No'}}}, "
            f"median DR={res.median_dr:.4f} [{res.dr_lower_ci:.4f}, {res.dr_upper_ci:.4f}], "
            f"min ESS={res.min_ess:.1f}."
        )
        f.write("\\caption[OPE grid (SNIS/DR/ESS)]{" + caption + "}\n")
        f.write("\\opeGridLabel\n    ")
        f.write("\\setlength{\\tabcolsep}{3pt}\\renewcommand{\\arraystretch}{1.1}\n")
        f.write("\\begin{tabularx}{\\linewidth}{@{} r r r r r @{} }\n    \\toprule\n")
        f.write("    Clip & Shrink & SNIS & DR & ESS \\\\ \n    \\midrule\n")
        for c, s, sn, drv, ess in rows:
            f.write(f"{c:.0f} & {s:.2f} & {sn:.4f} & {drv:.4f} & {ess:.1f} \\\\ \n")
        f.write("    \\bottomrule\n")
        f.write("  \\end{tabularx}\n")
        f.write("\\end{threeparttable}\n\\end{table}\n")


def main() -> None:
    args = parse_args()
    cfg = GateConfig(
        clips=_grid(args.grid_clips), shrinks=_grid(args.grid_shrinks), alpha=args.alpha
    )
    df = pd.read_csv(args.dataset)
    res = run_gate(df, cfg)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(asdict(res), f, indent=2)
    if args.tex:
        _write_tex_table(args.tex, res, cfg)
    print(
        f"[gate] accept={res.accept} median_dr={res.median_dr:.4f} "
        f"CI=[{res.dr_lower_ci:.4f}, {res.dr_upper_ci:.4f}] "
        f"min_ess={res.min_ess:.1f} stable={res.stable}"
    )
    if not res.accept:
        print(f"  Rejection reasons: {', '.join(res.reason_codes)}")
    print(f"  Output: {args.output}" + (f", tex: {args.tex}" if args.tex else ""))


if __name__ == "__main__":
    main()
