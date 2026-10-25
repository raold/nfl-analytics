#!/usr/bin/env python3
"""
Grade the walk-forward baselines with market_benchmark: devdb01 2019-2024, 2025 and 2019-2025,
and the 2026 holdout from an nflverse games.csv.

For every window and baseline (market, elo, power_ratings) this runs the market_benchmark CLI,

    python -m nfl_analytics.evaluation.market_benchmark --baseline B --db --seasons S
    python -m nfl_analytics.evaluation.market_benchmark --baseline B --nflverse-csv GAMES \\
        --seasons 2026

in-process (the CLI's own main()), and saves its Markdown and JSON reports. The database
session is read-only (PGOPTIONS). Every baseline warms up on the season before the first one
graded, so 2025 alone starts from 2024, and 2019-2025 runs on from 2018 without a restart.
devdb01 grades regular-season games only (game_type 'REG', repaired from nflverse on
2026-10-24); the nflverse source does the same by default.

ROI here is at each side's listed nflverse closing spread price (-110 where none is listed),
which is what market_benchmark uses when the source has prices.

The README rows for 2019-2024 (power ratings 609-608-31, Elo 581-596-31) come from the bundled
data/historical_weather_2018_2024.csv. The script reconciles them with devdb01: the CSV writes
the Raiders as LV in 2018-2019, where nflverse and devdb01 have OAK (the baselines key ratings
by team code, so with devdb01 the Raiders restart at the mean in 2020), and it has no spread
prices (every bet at -110). Recoding OAK as LV and pricing every bet at -110 must give the CSV
rows exactly; the script stops if it does not.

Outputs (docs/investigations/2026-10-final-evaluation/results/):
  baselines_{db,nflverse}_<seasons>_<baseline>.{md,json}  one report per run
  baselines_summary.csv / baselines_summary.md            headline numbers of every run
  baselines_reconcile_2019_2024.csv                       bundled CSV vs devdb01
  baselines_provenance.csv                                input fingerprints

Usage (games.csv from
https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv):

  uv run --no-sync python scripts/final_evaluation/grade_baselines.py --nflverse-csv games.csv
  (DB connection from the DB_* environment variables; see nfl_analytics/utils/db_config.py)
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_analytics.evaluation.market_benchmark import (
    baseline_predictions,
    load_market_from_csv,
    load_market_from_db,
    load_market_from_nflverse,
    run_benchmark,
)
from nfl_analytics.evaluation.market_benchmark.__main__ import main as benchmark_cli
from nfl_analytics.utils.db_config import get_connection

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "docs/investigations/2026-10-final-evaluation/results"
BUNDLED_CSV = REPO / "data/historical_weather_2018_2024.csv"
BASELINES = ("market", "elo", "power_ratings")
DB_WINDOWS = ("2019-2024", "2025", "2019-2025")
HOLDOUT = "2026"
READ_ONLY = "-c default_transaction_read_only=on"
MARKET_COLS = [
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "spread_close",
    "home_moneyline",
    "away_moneyline",
    "home_spread_odds",
    "away_spread_odds",
]


def run_cli(argv: list[str]) -> tuple[dict, str]:
    """The market_benchmark CLI's JSON (parsed) and Markdown output for one set of arguments."""
    outputs = []
    for extra in (["--json"], []):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = benchmark_cli([*argv, *extra])
        if rc != 0:
            raise RuntimeError(f"market_benchmark {' '.join(argv)} returned {rc}")
        outputs.append(buf.getvalue())
    return json.loads(outputs[0]), outputs[1]


def summary_row(source: str, seasons: str, baseline: str, r: dict) -> dict:
    a, s, m = r["ats"], r["spread_accuracy"], r["market_sanity"]
    reg, wp = r["information_regression"], r["win_prob_skill"] or {}
    return {
        "source": source,
        "seasons": seasons,
        "baseline": baseline,
        "games": r["games"],
        "bets": a["bets"],
        "wins": a["wins"],
        "losses": a["losses"],
        "pushes": a["pushes"],
        "win_rate": a["win_rate"],
        "win_rate_lo": a["win_rate_ci95"][0],
        "win_rate_hi": a["win_rate_ci95"][1],
        "breakeven": a["breakeven_win_rate"],
        "p_beats_breakeven": a["p_value_beats_breakeven"],
        "roi": a["roi"],
        "roi_lo": a["roi_ci95"][0],
        "roi_hi": a["roi_ci95"][1],
        "model_mae": s["model_mae"],
        "market_mae": s["market_mae"],
        "dm_p": s["dm_p_value_two_sided"],
        "info_coef": reg["model_gap_coef"],
        "info_lo": reg["model_gap_ci95"][0],
        "info_hi": reg["model_gap_ci95"][1],
        "info_note": reg["note"],
        "model_log_loss": wp.get("model_log_loss"),
        "market_log_loss": wp.get("market_log_loss"),
        "log_loss_diff_lo": (wp.get("log_loss_diff_ci95") or [None, None])[0],
        "log_loss_diff_hi": (wp.get("log_loss_diff_ci95") or [None, None])[1],
        "home_cover_rate": m["home_cover_rate"],
        "home_cover_p": m["home_cover_p_value"],
        "line_margin_corr": m["line_margin_corr"],
        "placebo_win_rate": r["placebo_win_rate"],
        "warnings": " | ".join(r["warnings"]),
    }


def fingerprint(market: pd.DataFrame) -> str:
    """sha256 of the graded market rows, to tell later whether a source still holds the same data."""
    cols = [c for c in MARKET_COLS if c in market]
    data = market[cols].sort_values("game_id").to_csv(index=False, float_format="%.6g")
    return hashlib.sha256(data.encode()).hexdigest()


def reconcile(conn) -> pd.DataFrame:
    """2019-2024 baselines: bundled CSV vs devdb01, recoded and repriced step by step."""
    seasons = list(range(2018, 2025))
    db = load_market_from_db(seasons, conn)
    as_lv = db.replace({"home_team": {"OAK": "LV"}, "away_team": {"OAK": "LV"}})
    flat = as_lv.assign(home_spread_odds=np.nan, away_spread_odds=np.nan)
    frames = {
        "bundled CSV (README rows)": load_market_from_csv(BUNDLED_CSV, seasons),
        "devdb01": db,
        "devdb01, Raiders as LV in 2018-2019": as_lv,
        "devdb01, Raiders as LV, every bet at -110": flat,
    }
    rows = []
    for baseline in ("elo", "power_ratings"):
        for label, market in frames.items():
            m = market.join(baseline_predictions(baseline, market))
            m = m[m["season"].between(2019, 2024)]
            r = run_benchmark(m, baseline)
            a = r["ats"]
            rows.append(
                {
                    "baseline": baseline,
                    "market_frame": label,
                    "record": f"{a['wins']}-{a['losses']}-{a['pushes']}",
                    "win_rate": a["win_rate"],
                    "roi": a["roi"],
                    "breakeven": a["breakeven_win_rate"],
                    "model_mae": r["spread_accuracy"]["model_mae"],
                    "market_mae": r["spread_accuracy"]["market_mae"],
                }
            )
    out = pd.DataFrame(rows)
    for baseline, g in out.groupby("baseline"):
        g = g.set_index("market_frame")
        csv_row = g.loc["bundled CSV (README rows)"]
        flat_row = g.loc["devdb01, Raiders as LV, every bet at -110"]
        same = csv_row["record"] == flat_row["record"] and all(
            math.isclose(csv_row[c], flat_row[c], rel_tol=0, abs_tol=1e-12)
            for c in ("win_rate", "roi", "model_mae", "market_mae")
        )
        if not same:
            raise RuntimeError(f"{baseline}: recoded devdb01 does not reproduce the bundled CSV")
    return out


def pct(x, signed=False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:+.1%}" if signed else f"{x:.1%}"


def render(summary: pd.DataFrame, rec: pd.DataFrame, prov: pd.DataFrame) -> str:
    out = [
        "# Walk-forward baselines graded with market_benchmark",
        "",
        "Generated by `scripts/final_evaluation/grade_baselines.py`. ROI at each side's listed",
        "closing spread price; win-rate intervals Wilson 95%, ROI and log-loss intervals by week",
        "blocks; MAE p-value Diebold-Mariano; information = coefficient on (pred - line).",
        "",
        "| Source | Seasons | Baseline | Games | ATS W-L-P | Win rate [95% CI] | ROI [95% CI] | "
        "MAE model / line (DM p) | Log-loss model / market | Info coef [95% CI] |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary.itertuples():
        info = (
            "n/a (pred - line constant)"
            if isinstance(r.info_note, str) and r.info_note
            else f"{r.info_coef:+.3f} [{r.info_lo:+.3f}, {r.info_hi:+.3f}]"
        )
        ll = (
            ""
            if r.model_log_loss is None or pd.isna(r.model_log_loss)
            else f"{r.model_log_loss:.4f} / {r.market_log_loss:.4f}"
        )
        win = f"{pct(r.win_rate)} [{pct(r.win_rate_lo)}, {pct(r.win_rate_hi)}]"
        roi = f"{pct(r.roi, True)} [{pct(r.roi_lo, True)}, {pct(r.roi_hi, True)}]"
        if r.bets == 0:  # the market baseline never differs from the line
            win = roi = "no bets"
        out.append(
            f"| {r.source} | {r.seasons} | {r.baseline} | {r.games} | "
            f"{r.wins}-{r.losses}-{r.pushes} | {win} | {roi} | "
            f"{r.model_mae:.2f} / {r.market_mae:.2f} ({r.dm_p:.3f}) | {ll} | {info} |"
        )
    out += [
        "",
        "## 2019-2024: the bundled CSV (README rows) and devdb01",
        "",
        "| Baseline | Market frame | ATS W-L-P | Win rate | ROI | Mean break-even | "
        "MAE model / line |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rec.itertuples():
        out.append(
            f"| {r.baseline} | {r.market_frame} | {r.record} | {pct(r.win_rate)} | "
            f"{pct(r.roi, True)} | {pct(r.breakeven)} | {r.model_mae:.3f} / {r.market_mae:.3f} |"
        )
    out += ["", "## Inputs", "", "| Input | Detail |", "|---|---|"]
    for r in prov.itertuples():
        out.append(f"| {r.input} | {r.detail} |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--nflverse-csv", required=True, type=Path, help="nflverse games.csv")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    os.environ["PGOPTIONS"] = READ_ONLY  # libpq applies it to every connection below
    # pandas warns that psycopg2 is not SQLAlchemy; the queries are plain SELECTs
    warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
    games = args.nflverse_csv.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = [("db", s, b, ["--db"]) for s in DB_WINDOWS for b in BASELINES]
    runs += [("nflverse", HOLDOUT, b, ["--nflverse-csv", str(games)]) for b in BASELINES]
    rows = []
    for source, seasons, baseline, src_args in runs:
        argv = ["--baseline", baseline, *src_args, "--seasons", seasons]
        result, md = run_cli(argv)
        stem = f"baselines_{source}_{seasons}_{baseline}"
        (args.out_dir / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n")
        (args.out_dir / f"{stem}.md").write_text(md)
        rows.append(summary_row(source, seasons, baseline, result))
        a = result["ats"]
        print(
            f"{source:8s} {seasons:9s} {baseline:13s} {result['games']:5d} games  "
            f"{a['wins']}-{a['losses']}-{a['pushes']}  ROI {pct(a['roi'], True)}  MAE "
            f"{result['spread_accuracy']['model_mae']:.2f} / "
            f"{result['spread_accuracy']['market_mae']:.2f}"
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "baselines_summary.csv", index=False, float_format="%.6g")

    conn = get_connection()
    try:
        conn.set_session(readonly=True)
        rec = reconcile(conn)
        db_market = load_market_from_db(range(2018, 2026), conn)
    finally:
        conn.close()
    rec.to_csv(args.out_dir / "baselines_reconcile_2019_2024.csv", index=False, float_format="%.6g")

    holdout = load_market_from_nflverse(games, [int(HOLDOUT)])
    raw = games.read_bytes()
    prov = pd.DataFrame(
        [
            {
                "input": "devdb01 games, REG 2018-2025 with score and line",
                "detail": f"{len(db_market)} games, sha256 of graded columns "
                f"{fingerprint(db_market)}",
            },
            {
                "input": f"nflverse {games.name}",
                "detail": f"{len(raw)} bytes, sha256 {hashlib.sha256(raw).hexdigest()}; "
                f"{len(holdout)} completed 2026 REG games with a line (weeks "
                f"{holdout['week'].min()}-{holdout['week'].max()}), last kickoff "
                f"{holdout['kickoff'].max():%Y-%m-%d %H:%M} UTC",
            },
            {
                "input": "data/historical_weather_2018_2024.csv",
                "detail": f"sha256 {hashlib.sha256(BUNDLED_CSV.read_bytes()).hexdigest()}",
            },
        ]
    )
    prov.to_csv(args.out_dir / "baselines_provenance.csv", index=False)
    report = render(summary, rec, prov)
    (args.out_dir / "baselines_summary.md").write_text(report)
    print()
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
