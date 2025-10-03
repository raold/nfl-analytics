# -*- coding: utf-8 -*-
"""
make_teaser_ev_figs.py — Teaser EV curve + heatmap
Curve PDF: analysis/dissertation/figures/out/teaser_ev_curve.pdf
Heatmap PDF: analysis/dissertation/figures/out/teaser_ev_heatmap.pdf
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

CURVE_OUT_DEFAULT = "analysis/dissertation/figures/out/teaser_ev_curve.pdf"
HEAT_OUT_DEFAULT = "analysis/dissertation/figures/out/teaser_ev_heatmap.pdf"


def dec_from_american(odds: float) -> float:
    if odds >= 100:
        return 1.0 + odds / 100.0
    if odds <= -100:
        return 1.0 + 100.0 / abs(odds)
    raise ValueError("American odds must be <= -100 or >= 100")


def ev_two_leg(q1: float, q2: float, american_odds: float) -> float:
    d = dec_from_american(american_odds)
    p_both = q1 * q2
    return p_both * (d - 1.0) - (1.0 - p_both)


def main():
    ap = argparse.ArgumentParser(description="Teaser EV curve + heatmap.")
    ap.add_argument(
        "--prices",
        type=str,
        default="-110,-120,-130",
        help="Comma-separated American odds list.",
    )
    ap.add_argument("--qmin", type=float, default=0.60)
    ap.add_argument("--qmax", type=float, default=0.85)
    ap.add_argument("--steps", type=int, default=126)
    ap.add_argument("--heat_price", type=float, default=-120.0)
    ap.add_argument("--out_curve", type=str, default=CURVE_OUT_DEFAULT)
    ap.add_argument("--out_heatmap", type=str, default=HEAT_OUT_DEFAULT)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_curve), exist_ok=True)

    qs = np.linspace(args.qmin, args.qmax, args.steps)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for token in args.prices.split(","):
        am = float(token.strip())
        evs = [ev_two_leg(q, q, am) for q in qs]
        ax.plot(qs, evs, lw=1.5, label=f"{int(am)}")
    ax.axhline(0, color="0.5", lw=1, ls="--")
    ax.set_xlabel("Per-leg win probability (q)")
    ax.set_ylabel("Expected value (units per 1 staked)")
    ax.set_title("Two-leg teaser EV across book pricing")
    ax.legend(title="American odds", frameon=False)
    fig.tight_layout()
    fig.savefig(args.out_curve)
    plt.close(fig)

    grid = np.linspace(args.qmin, args.qmax, 121)
    Q1, Q2 = np.meshgrid(grid, grid)
    EV = Q1 * Q2 * (dec_from_american(args.heat_price) - 1.0) - (1.0 - Q1 * Q2)
    fig2, ax2 = plt.subplots(figsize=(8.0, 4.8))
    im = ax2.imshow(
        EV,
        origin="lower",
        extent=[grid.min(), grid.max(), grid.min(), grid.max()],
        aspect="auto",
        cmap="Greys",
    )
    ax2.set_xlabel("Leg 1 win prob")
    ax2.set_ylabel("Leg 2 win prob")
    ax2.set_title(f"Teaser EV heatmap (odds {int(args.heat_price)})")
    cbar = fig2.colorbar(im, ax=ax2)
    cbar.set_label("EV (units per 1 staked)")
    fig2.tight_layout()
    fig2.savefig(args.out_heatmap)
    plt.close(fig2)


if __name__ == "__main__":
    main()
