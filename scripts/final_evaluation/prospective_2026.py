#!/usr/bin/env python3
"""
Prospective 2026 predictions with market_benchmark's predict mode, plus a SHA256SUMS file.

For each week and baseline (elo, power_ratings) this runs

    python -m nfl_analytics.evaluation.market_benchmark predict --baseline B \\
        --nflverse-csv GAMES --season 2026 --week W --out OUT_DIR/preds_2026_wWW_B<suffix>.csv

Predict mode fits the walk-forward baseline on every completed regular-season game of the
previous season and of the earlier weeks of the target season, and writes one row per game of
week W that has no score and has not kicked off, with the line in GAMES at that moment
(spread_at_prediction) and the time of the run (prediction_timestamp). The baselines run with
the fixed constants in nfl_analytics/evaluation/market_benchmark/baselines.py, which every row
records in baseline_params; nothing is fitted to the target season beyond the walk-forward
ratings themselves. The CLI refuses to replace an existing file, so a record cannot be
overwritten by accident; use --suffix for a second run of the same week.

Afterwards OUT_DIR/SHA256SUMS lists every preds_*.csv in OUT_DIR, so the files can be checked
with `cd OUT_DIR && sha256sum -c SHA256SUMS`.

Usage (games.csv from https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv):

    uv run --no-sync python scripts/final_evaluation/prospective_2026.py \\
        --nflverse-csv games.csv --weeks 7,8 \\
        --out-dir docs/investigations/2026-10-final-evaluation/prospective
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASELINES = ("elo", "power_ratings")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--nflverse-csv", required=True, type=Path, help="nflverse games.csv")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--weeks", required=True, help="comma-separated, e.g. 7,8")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--suffix", default="", help="appended to each file name, e.g. _rerun")
    args = ap.parse_args(argv)

    games = args.nflverse_csv.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"input  {games.name}  sha256 {sha256(games)}  {games.stat().st_size} bytes")

    failed = 0
    for week in [int(w) for w in args.weeks.split(",")]:
        for baseline in BASELINES:
            out = out_dir / f"preds_{args.season}_w{week:02d}_{baseline}{args.suffix}.csv"
            started = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
            cmd = [
                sys.executable,
                "-I",  # the input is a download: keep the cwd and PYTHON* variables off the path
                "-m",
                "nfl_analytics.evaluation.market_benchmark",
                "predict",
                "--baseline",
                baseline,
                "--nflverse-csv",
                str(games),
                "--season",
                str(args.season),
                "--week",
                str(week),
                "--out",
                str(out),
            ]
            run = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
            print(
                f"== {args.season} week {week} {baseline} (started {started}) rc={run.returncode}"
            )
            # the CLI prints counts on stderr and the two sha256 lines on stdout
            for line in (run.stderr + run.stdout).splitlines():
                print("   " + line.replace(str(games), games.name).replace(str(REPO) + "/", ""))
            failed += run.returncode != 0

    sums = sorted(out_dir.glob("preds_*.csv"))
    (out_dir / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in sums))
    print(f"SHA256SUMS: {len(sums)} files")
    for p in sums:
        print(f"   {sha256(p)}  {p.name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
