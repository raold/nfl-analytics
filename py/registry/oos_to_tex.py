"""
Experiment Registry Writer: export OOS results CSV to TeX table rows.

Input CSV columns:
  season,model,clv_bp,brier,ece,roi_pct,mar,max_dd_pct,n_bets

Writes lines suitable for the Chapter 8 table:
  2023 & Ensemble & +12 & 0.217 & 0.029 & +1.9 & 0.48 & -10.2 & 1150 \\

Usage:
  python py/registry/oos_to_tex.py --csv results/oos.csv \
      --tex analysis/dissertation/results/oos_record_rows.tex
"""

from __future__ import annotations

import argparse
import csv
import os


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export OOS CSV to TeX rows")
    ap.add_argument("--csv", required=True, help="Input CSV with OOS results")
    ap.add_argument("--tex", required=True, help="Output TeX rows file")
    return ap.parse_args()


def fmt_int(x: str) -> str:
    try:
        n = int(float(x))
        return f"{n:,}".replace(",", "{,}")
    except Exception:
        return x


def main() -> None:
    args = parse_args()
    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            season = r["season"].strip()
            model = r["model"].strip()
            clv = r["clv_bp"].strip()
            brier = r["brier"].strip()
            ece = r["ece"].strip()
            roi = r["roi_pct"].strip()
            mar = r["mar"].strip()
            dd = r["max_dd_pct"].strip()
            n = fmt_int(r["n_bets"].strip())
            rows.append(
                f"{season} & {model} & {clv} & {brier} & {ece} & {roi} & {mar} & {dd} & {n} \\\n"
            )
    os.makedirs(os.path.dirname(args.tex), exist_ok=True)
    with open(args.tex, "w", encoding="utf-8") as g:
        for line in rows:
            g.write(line)
    print(f"[oos->tex] wrote {len(rows)} rows -> {args.tex}")


if __name__ == "__main__":
    main()
