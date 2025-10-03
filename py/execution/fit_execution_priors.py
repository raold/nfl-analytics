"""
Fit execution priors (depth, fill, latency) from order logs (stub).

Input CSV columns expected (best-effort parsing):
  book,market,tau_min,q_req,executed_q,quoted_price,executed_price,
  velocity,side,signal_ts,exec_ts

Outputs JSON with per-(book,market,tau_bucket) parameters for:
  depth: beta0..beta3, rmse
  fill:  theta0..theta4
  latency: mu_log, sigma_log

Usage:
  python py/execution/fit_execution_priors.py --logs data/orders.csv \
      --out analysis/dissertation/results/execution_priors.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from typing import Dict, Any, Tuple

from .depth_model import fit_depth, tau_bucket
from .fill_model import fit_fill
from .latency_model import fit_latency


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fit execution priors (stub)")
    ap.add_argument("--logs", required=True, help="CSV order logs file")
    ap.add_argument("--out", required=True, help="Output JSON path")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.logs, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    depth = fit_depth(reader)
    fill = fit_fill(reader)
    lat = fit_latency(reader)
    out: Dict[str, Any] = {}
    for key in set(list(depth.keys()) + list(fill.keys()) + list(lat.keys())):
        book, market, tau = key
        kstr = f"{book}::{market}::{tau}"
        d = depth.get(key)
        fl = fill.get(key)
        lt = lat.get(key)
        out[kstr] = {
            "depth": {
                "beta": list(d.beta) if d else [0, 0, 0, 0],
                "rmse": getattr(d, "rmse", 0.0),
            },
            "fill": {
                "theta": list(fl.theta) if fl else [0, 0, 0, 0, 0],
            },
            "latency": {
                "mu_log": getattr(lt, "mu_log", 0.0),
                "sigma_log": getattr(lt, "sigma_log", 0.0),
            },
        }
    with open(args.out, "w", encoding="utf-8") as g:
        json.dump(out, g, indent=2)
    print(f"[exec-priors] wrote {len(out)} buckets -> {args.out}")


if __name__ == "__main__":
    main()

