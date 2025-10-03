# -*- coding: utf-8 -*-
"""
make_bankroll_ensemble.py — Bankroll ensemble simulator (fractional Kelly)
Outputs:
  • Envelope PDF → analysis/dissertation/figures/out/bankroll_ensemble.pdf
  • Optional histogram PDF (via --histout)
  • Printed summary (+ optional CSV via --summary)
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DEFAULT = "analysis/dissertation/figures/out/bankroll_ensemble.pdf"


def american_to_decimal(american: float) -> float:
    if american >= 100:
        return 1.0 + american / 100.0
    if american <= -100:
        return 1.0 + 100.0 / abs(american)
    raise ValueError("american_odds must be <= -100 or >= 100")


def series_american_to_decimal(arr: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr, dtype=float)
    for i, ao in enumerate(arr):
        out[i] = american_to_decimal(float(ao))
    return out


def kelly_fraction(p: float, b_net: float) -> float:
    if b_net <= 0:
        return 0.0
    return max(0.0, p - (1.0 - p) / b_net)


def load_bets(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    if "prob" not in cols:
        raise ValueError("CSV must contain 'prob'.")
    if "decimal_odds" in cols:
        dec = df[cols["decimal_odds"]].astype(float).values
    elif "american_odds" in cols:
        dec = series_american_to_decimal(df[cols["american_odds"]].values.astype(float))
    else:
        raise ValueError("CSV must contain decimal_odds or american_odds.")
    prob = df[cols["prob"]].astype(float).values
    if np.any((prob < 0) | (prob > 1)):
        raise ValueError("prob must be in [0,1].")
    b_net = np.clip(dec - 1.0, 0.0, None)
    return pd.DataFrame(
        {"prob": prob, "decimal_odds": dec, "b_net": b_net}
    ).reset_index(drop=True)


def simulate_bankroll_paths(
    bets_df: pd.DataFrame,
    n_paths=1000,
    frac=0.5,
    cap_fraction=1.0,
    seed=123,
    start_bankroll=1.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p, b = bets_df["prob"].values, bets_df["b_net"].values
    n_bets = len(p)
    f_star = np.array([kelly_fraction(pi, bi) for pi, bi in zip(p, b)], dtype=float)
    stake_mult = np.minimum(cap_fraction, np.maximum(0.0, frac * f_star))
    br = np.zeros((n_paths, n_bets + 1), dtype=float)
    br[:, 0] = float(start_bankroll)
    for i in range(n_bets):
        wins = rng.random(n_paths) < p[i]
        stake = stake_mult[i] * br[:, i]
        pnl = np.where(wins, stake * b[i], -stake)
        br[:, i + 1] = br[:, i] + pnl
    return br


def plot_envelope(bankrolls: np.ndarray, out_pdf: str, title: str, logy=False):
    q5 = np.quantile(bankrolls, 0.05, axis=0)
    q25 = np.quantile(bankrolls, 0.25, axis=0)
    q50 = np.quantile(bankrolls, 0.50, axis=0)
    q75 = np.quantile(bankrolls, 0.75, axis=0)
    q95 = np.quantile(bankrolls, 0.95, axis=0)
    x = np.arange(bankrolls.shape[1])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.fill_between(x, q25, q75, color="0.8", label="IQR")
    ax.fill_between(x, q5, q95, color="0.9", label="5–95%")
    ax.plot(x, q50, lw=1.8, color="0.2", label="Median")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Bets")
    ax.set_ylabel("Bankroll (units)")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_final_hist(bankrolls: np.ndarray, out_pdf: str, bins=60):
    final_br = bankrolls[:, -1]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(final_br, bins=bins, color="0.3", edgecolor="0.7")
    ax.set_xlabel("Final bankroll")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of final bankrolls")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


def summarize(bankrolls: np.ndarray) -> dict:
    def q(a: float) -> float:
        return float(np.quantile(bankrolls[:, -1], a))
    median_path = np.median(bankrolls, axis=0)
    run_max = np.maximum.accumulate(median_path)
    dd = (run_max - median_path) / np.maximum(run_max, 1e-12)
    return {
        "final_q05": q(0.05),
        "final_q25": q(0.25),
        "final_q50": q(0.50),
        "final_q75": q(0.75),
        "final_q95": q(0.95),
        "median_max_drawdown": float(np.max(dd)),
        "prob_final_lt_0_5": float(np.mean(bankrolls[:, -1] < 0.5)),
        "prob_final_lt_0_25": float(np.mean(bankrolls[:, -1] < 0.25)),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Bankroll ensemble simulator (fractional Kelly)."
    )
    ap.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Bets CSV with prob and decimal_odds OR american_odds.",
    )
    ap.add_argument("--paths", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--cap", type=float, default=1.0)
    ap.add_argument("--start", type=float, default=1.0)
    ap.add_argument("--out", type=str, default=OUT_DEFAULT)
    ap.add_argument("--logy", action="store_true")
    ap.add_argument("--histout", type=str, default=None)
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--summary", type=str, default=None)
    args = ap.parse_args()

    if args.csv:
        bets = load_bets(args.csv)
        title = "Bankroll trajectories (fractional Kelly)"
    else:
        n_bets, p_true = 200, 0.54
        dec = np.full(n_bets, 2.0)
        b_net = dec - 1.0
        bets = pd.DataFrame(
            {"prob": np.full(n_bets, p_true), "decimal_odds": dec, "b_net": b_net}
        )
        title = "Bankroll trajectories (demo, p=0.54, even money)"

    br = simulate_bankroll_paths(
        bets,
        n_paths=args.paths,
        frac=args.frac,
        cap_fraction=args.cap,
        seed=args.seed,
        start_bankroll=args.start,
    )
    plot_envelope(br, args.out, title, logy=args.logy)
    if args.histout:
        plot_final_hist(br, args.histout, bins=args.bins)

    summary = summarize(br)
    print("=== Bankroll Summary ===")
    [print(f"{k}: {v:.6f}") for k, v in summary.items()]
    if args.summary:
        os.makedirs(os.path.dirname(args.summary), exist_ok=True)
        pd.DataFrame([summary]).to_csv(args.summary, index=False)


if __name__ == "__main__":
    main()
