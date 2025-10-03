# -*- coding: utf-8 -*-
"""
make_reliability_and_error.py — Reliability diagram + weekly error sparkline
Outputs:
  • reliability_calibration_wk01_wk18.pdf
  • calibration_error_sparkline.pdf (only if 'week' exists)
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REL_OUT_DEFAULT = (
    "analysis/dissertation/figures/out/reliability_calibration_wk01_wk18.pdf"
)
ERR_OUT_DEFAULT = "analysis/dissertation/figures/out/calibration_error_sparkline.pdf"


def bin_calibration(probs, outcomes, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    emp = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = (probs >= bins[i]) & (
            probs < bins[i + 1] if i < n_bins - 1 else probs <= bins[i + 1]
        )
        if m.any():
            emp[i] = outcomes[m].mean()
    return centers, emp


def ece_absolute(probs, outcomes, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(probs)
    ece = 0.0
    for i in range(n_bins):
        m = (probs >= bins[i]) & (
            probs < bins[i + 1] if i < n_bins - 1 else probs <= bins[i + 1]
        )
        if m.any():
            p_hat = outcomes[m].mean()
            p_bar = probs[m].mean()
            ece += (m.sum() / total) * abs(p_hat - p_bar)
    return ece


def main():
    ap = argparse.ArgumentParser(
        description="Reliability diagram + weekly calibration error sparkline."
    )
    ap.add_argument(
        "--input",
        type=str,
        default="analysis/dissertation/figures/data/preds.csv",
        help="CSV with columns prob,outcome[,week]",
    )
    ap.add_argument("--out_reliability", type=str, default=REL_OUT_DEFAULT)
    ap.add_argument("--out_error", type=str, default=ERR_OUT_DEFAULT)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if not {"prob", "outcome"}.issubset(df.columns):
        raise ValueError("CSV must have prob,outcome.")
    os.makedirs(os.path.dirname(args.out_reliability), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="0.6", label="Perfect")
    colors = ["0.2", "0.5"]
    drawn = 0
    if "week" in df.columns:
        for j, wk in enumerate([1, 18]):
            if (df["week"] == wk).any():
                m = df["week"] == wk
                c, emp = bin_calibration(
                    df.loc[m, "prob"].values, df.loc[m, "outcome"].values
                )
                ax.plot(c, emp, marker="o", lw=1.5, color=colors[j], label=f"Week {wk}")
                drawn += 1
    if drawn == 0:
        c, emp = bin_calibration(df["prob"].values, df["outcome"].values)
        ax.plot(c, emp, marker="o", lw=1.5, color="0.2", label="All")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_title("Reliability diagram")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out_reliability)
    plt.close(fig)

    if "week" in df.columns:
        weeks = sorted(df["week"].dropna().unique())
        errs = []
        for wk in weeks:
            m = df["week"] == wk
            errs.append(
                ece_absolute(df.loc[m, "prob"].values, df.loc[m, "outcome"].values)
            )
        fig2 = plt.figure(figsize=(1.6, 0.35), dpi=150)
        ax2 = fig2.add_axes([0, 0, 1, 1])
        ax2.plot(weeks, errs, lw=1.2, color="0.2")
        ax2.set_axis_off()
        os.makedirs(os.path.dirname(args.out_error), exist_ok=True)
        fig2.savefig(args.out_error, bbox_inches="tight", pad_inches=0)
        plt.close(fig2)


if __name__ == "__main__":
    main()
