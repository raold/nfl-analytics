# -*- coding: utf-8 -*-
"""
make_clv_sparkline.py — CLV sparkline (margin figure)

Summary
-------
Generates a compact sparkline showing cumulative expected value (CLV) across a
sequence of bets. Designed to be placed in the LaTeX margin via \marginfig{}.

Inputs
------
Optional CSV:
  expected_value : float per bet (EV per bet, positive/negative is fine)
  (optional) bet_id : any identifier, ignored by this plot

Outputs
-------
PDF: analysis/figures/out/clv_sparkline_2025.pdf  (default)

CLI
---
  python analysis/figures/make_clv_sparkline.py \
      --input analysis/figures/data/bets.csv \
      --output analysis/figures/out/clv_sparkline_2025.pdf
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DEFAULT = "analysis/figures/out/clv_sparkline_2025.pdf"


def main():
    ap = argparse.ArgumentParser(description="Generate CLV sparkline PDF.")
    ap.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional CSV with column 'expected_value'.",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=OUT_DEFAULT,
        help=f"Output PDF (default {OUT_DEFAULT})",
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.input:
        df = pd.read_csv(args.input)
        if "expected_value" not in df.columns:
            raise ValueError("CSV must contain 'expected_value' column.")
        ev = df["expected_value"].astype(float).values
    else:
        # Synthetic demo series
        rng = np.random.default_rng(7)
        ev = rng.normal(0.002, 0.02, 120)  # small noise around slight positive drift

    clv = np.cumsum(ev)

    fig = plt.figure(figsize=(1.6, 0.35), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(clv, lw=1.2, color="0.2")
    ax.set_axis_off()
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    main()
