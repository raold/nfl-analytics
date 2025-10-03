"""
Simulator acceptance test (stub).

Loads placeholder historical and simulated metrics and evaluates simple
criteria (margins, key masses, dependence, friction RMSE). Replace the
readers/comparators with real implementations.

Usage:
  python py/sim/acceptance.py --hist hist.json --sim sim.json --output report.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Simulator acceptance test (stub)")
    ap.add_argument("--hist", required=True, help="Historical metrics JSON")
    ap.add_argument("--sim", required=True, help="Simulated metrics JSON")
    ap.add_argument("--output", required=True, help="Output JSON report")
    ap.add_argument("--tau-marg", type=float, default=0.05, help="Tolerance for margin metric")
    ap.add_argument("--tau-key", type=float, default=0.01, help="Tolerance for key-mass deltas")
    ap.add_argument("--tau-fric", type=float, default=0.5, help="Tolerance for slippage RMSE")
    ap.add_argument("--tau-fillshort", type=float, default=0.1, help="Tolerance for fill shortfall (fraction)")
    return ap.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare(hist: Dict[str, Any], sim: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    # Placeholder comparisons
    marg_ok = abs(sim.get("emd_margin", 0.0) - hist.get("emd_margin", 0.0)) <= args.tau_marg
    key_ok = max(abs(sim.get("key3", 0.0) - hist.get("key3", 0.0)),
                 abs(sim.get("key6", 0.0) - hist.get("key6", 0.0)),
                 abs(sim.get("key7", 0.0) - hist.get("key7", 0.0)),
                 abs(sim.get("key10", 0.0) - hist.get("key10", 0.0))) <= args.tau_key
    fric_ok = sim.get("slippage_rmse", 0.0) <= args.tau_fric
    fill_short = sim.get("fill_shortfall", 0.0)
    fill_ok = fill_short <= args.tau_fillshort
    passed = marg_ok and key_ok and fric_ok and fill_ok
    return {
        "pass": passed,
        "checks": {
            "margin": marg_ok,
            "keys": key_ok,
            "friction": fric_ok,
            "fill_shortfall": fill_ok,
        },
    }


def main() -> None:
    args = parse_args()
    hist = load_json(args.hist)
    sim = load_json(args.sim)
    rep = compare(hist, sim, args)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(f"[sim-accept] pass={rep['pass']} -> {args.output}")


if __name__ == "__main__":
    main()
