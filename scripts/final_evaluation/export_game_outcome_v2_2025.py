#!/usr/bin/env python3
"""
Export Game Outcome v2's 2025 predictions with the closing market to a tracked CSV.

Reads devdb01 in a read-only session: every row of ml.game_outcome_predictions (285 games, one
run created 2026-02-12, after the season) joined to games for the game type, the true kickoff,
the scores and the nflverse closing market (spread, total, moneylines, spread and total prices).
Writes data/game_outcome_v2_predictions_2025.csv, which grade_game_outcome_v2_2025.py grades
without the database.

Column semantics (docs/investigations/2026-10-final-evaluation/README.md has the full list):
  pred_home_spread  the model's expected HOME margin (home - away points), despite its name;
                    nflverse sign, the same as spread_close
  vegas_spread      equals spread_close for every row (checked below)
  kickoff           as stored by the run: the game date at midnight ET written as UTC (04:00 or
                    05:00), not the kickoff time. kickoff_utc is the true kickoff from games
  prediction_timestamp  copy of created_at under the name market_benchmark reads, so grading the
                    file with --predictions-csv reports the run as retroactive

With --nflverse-csv the market and score columns are also checked against an nflverse
games.csv (they are nflverse schedule data, not The Odds API data).

Usage:
  uv run --no-sync python scripts/final_evaluation/export_game_outcome_v2_2025.py \\
      [--nflverse-csv games.csv]
  (DB connection from the DB_* environment variables; see nfl_analytics/utils/db_config.py)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

from nfl_analytics.utils.db_config import get_db_config

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/game_outcome_v2_predictions_2025.csv"

# Model columns in table order, then the market columns from games
SQL = """
SELECT p.id, p.prediction_run_id, p.game_id, p.season, p.week, p.home_team, p.away_team,
       p.kickoff, p.pred_home_spread, p.pred_total, p.pred_home_win_prob, p.spread_std,
       p.total_std, p.vegas_spread, p.vegas_total, p.actual_home_score, p.actual_away_score,
       p.actual_spread, p.actual_total, p.model_version, p.inference_time_ms, p.created_at,
       g.game_type, g.kickoff AS kickoff_utc, g.home_score, g.away_score, g.spread_close,
       g.total_close, g.home_moneyline, g.away_moneyline, g.home_spread_odds,
       g.away_spread_odds, g.over_odds, g.under_odds
FROM ml.game_outcome_predictions p
JOIN games g USING (game_id)
ORDER BY g.kickoff, p.game_id
"""

# nflverse games.csv column for each exported market column
NFLVERSE = {
    "game_type": "game_type",
    "home_score": "home_score",
    "away_score": "away_score",
    "spread_close": "spread_line",
    "total_close": "total_line",
    "home_moneyline": "home_moneyline",
    "away_moneyline": "away_moneyline",
    "home_spread_odds": "home_spread_odds",
    "away_spread_odds": "away_spread_odds",
    "over_odds": "over_odds",
    "under_odds": "under_odds",
}


def fetch() -> pd.DataFrame:
    cfg = get_db_config()
    with psycopg.connect(**cfg, options="-c default_transaction_read_only=on") as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute("SHOW transaction_read_only")
            if cur.fetchone()[0] != "on":
                raise RuntimeError("could not open a read-only session")
            cur.execute(SQL)
            cols = [d.name for d in cur.description]
            rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


def check(df: pd.DataFrame) -> list[str]:
    """Consistency checks that the README relies on; returns failures."""
    margin = df["home_score"] - df["away_score"]
    checks = {
        "285 rows, one per game": len(df) == 285 and df["game_id"].is_unique,
        "one prediction run": df["prediction_run_id"].nunique() == 1,
        "vegas_spread == spread_close": bool((df["vegas_spread"] == df["spread_close"]).all()),
        "vegas_total == total_close": bool((df["vegas_total"] == df["total_close"]).all()),
        "actual scores == games scores": bool(
            (df["actual_home_score"] == df["home_score"]).all()
            and (df["actual_away_score"] == df["away_score"]).all()
        ),
        "actual_spread == home margin": bool((df["actual_spread"] == margin).all()),
        "created_at after the last kickoff": df["created_at"].min() > df["kickoff_utc"].max(),
    }
    for name, ok in checks.items():
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    return [name for name, ok in checks.items() if not ok]


def compare_nflverse(df: pd.DataFrame, path: Path) -> list[str]:
    nv = pd.read_csv(path, dtype={"gameday": "string", "gametime": "string"})
    nv = nv.set_index("game_id").loc[df["game_id"]]
    failures = []
    for ours, theirs in NFLVERSE.items():
        a, b = df[ours].to_numpy(), nv[theirs].to_numpy()
        if a.dtype.kind in "fi":
            same = np.isclose(a.astype(float), b.astype(float), rtol=0, atol=1e-9)
        else:
            same = a == b
        print(
            f"  {'ok  ' if same.all() else 'FAIL'} {ours} == nflverse {theirs}: {int(same.sum())}/{len(df)}"
        )
        if not same.all():
            failures.append(ours)
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--nflverse-csv", type=Path, help="also compare with an nflverse games.csv")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    df = fetch()
    df["prediction_timestamp"] = df["created_at"]
    print(f"{len(df)} rows from ml.game_outcome_predictions JOIN games")
    print(
        f"  run {df['prediction_run_id'].iloc[0]}, model_version {df['model_version'].iloc[0]}, "
        f"created_at {df['created_at'].min()} .. {df['created_at'].max()}"
    )
    print(f"  game types: {df['game_type'].value_counts().to_dict()}")
    print(f"  kickoff_utc {df['kickoff_utc'].min()} .. {df['kickoff_utc'].max()}")
    failures = check(df)
    if args.nflverse_csv:
        print(f"compared with {args.nflverse_csv.name}:")
        failures += compare_nflverse(df, args.nflverse_csv)
    if failures:
        print(f"NOT WRITTEN: {len(failures)} checks failed: {failures}", file=sys.stderr)
        return 1
    df.to_csv(args.out, index=False)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    shown = args.out.relative_to(REPO) if args.out.is_relative_to(REPO) else args.out
    print(f"wrote {shown}: {len(df)} rows, {len(df.columns)} columns, sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
