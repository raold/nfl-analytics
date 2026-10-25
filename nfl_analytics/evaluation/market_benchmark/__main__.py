"""
Benchmark NFL spread predictions against the closing market, or write prospective predictions.

Grading (the default command; "grade" may be given explicitly):

    # Baselines from the bundled CSV (no database needed)
    python -m nfl_analytics.evaluation.market_benchmark --baseline power_ratings \\
        --market-csv data/historical_weather_2018_2024.csv --seasons 2019-2024

    # A model's predictions (CSV with game_id + expected home margin) against devdb01
    python -m nfl_analytics.evaluation.market_benchmark --predictions-csv preds.csv \\
        --pred-col pred_home_spread --seasons 2025 --db

    # The current season straight from nflverse (games.csv from nflverse-data releases)
    python -m nfl_analytics.evaluation.market_benchmark --baseline elo \\
        --nflverse-csv games.csv --seasons 2026

Baselines are walk-forward and start from the season before the first one graded (warm-up),
whichever the source. Prospective predictions for one week (games not started yet):

    python -m nfl_analytics.evaluation.market_benchmark predict --baseline elo \\
        --nflverse-csv games.csv --season 2026 --week 8 --out preds_2026_w08.csv

Grade such a file after the games with --predictions-csv; it reports whether the predictions
were made before kickoff and their closing-line value.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path

import pandas as pd

from nfl_analytics.evaluation.market_benchmark.baselines import BASELINES, baseline_predictions
from nfl_analytics.evaluation.market_benchmark.guardrails import ImplausibleResultError
from nfl_analytics.evaluation.market_benchmark.market import (
    load_market_from_csv,
    load_market_from_db,
    load_market_from_nflverse,
    read_nflverse_games,
)
from nfl_analytics.evaluation.market_benchmark.predict import (
    PREDICT_BASELINES,
    predict_week,
    write_predictions,
)
from nfl_analytics.evaluation.market_benchmark.report import (
    LINE_AT_PREDICTION,
    run_benchmark,
    to_json,
    to_markdown,
)

PROG = "python -m nfl_analytics.evaluation.market_benchmark"


def _seasons(text: str) -> list[int]:
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in text.split(",")]


def _load_market(args, seasons: list[int]) -> pd.DataFrame:
    if args.db:
        from nfl_analytics.utils.db_config import get_connection

        conn = get_connection()
        try:
            return load_market_from_db(seasons, conn)
        finally:
            conn.close()
    if args.nflverse_csv:
        return load_market_from_nflverse(args.nflverse_csv, seasons, args.game_types or ["REG"])
    return load_market_from_csv(args.market_csv, seasons)


def grade_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog=PROG, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--baseline", choices=sorted(BASELINES))
    src.add_argument("--predictions-csv", help="CSV with game_id and a predicted home margin")
    ap.add_argument("--pred-col", default="pred_home_margin", help="column in --predictions-csv")
    mkt = ap.add_mutually_exclusive_group(required=True)
    mkt.add_argument("--market-csv", help="CSV export of games (scores + spread_close)")
    mkt.add_argument("--db", action="store_true", help="read games from devdb01 (DB_* env vars)")
    mkt.add_argument("--nflverse-csv", help="nflverse schedule CSV (games.csv)")
    ap.add_argument(
        "--game-types",
        type=lambda s: s.split(","),
        help="with --nflverse-csv: game types to grade, e.g. REG,WC,DIV,CON,SB (default REG)",
    )
    ap.add_argument("--seasons", required=True, type=_seasons, help="e.g. 2019-2024 or 2023,2024")
    ap.add_argument("--min-edge", type=float, default=1.0)
    ap.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    ap.add_argument("--allow-implausible", action="store_true")
    args = ap.parse_args(argv)
    if args.game_types and not args.nflverse_csv:
        ap.error("--game-types applies only to --nflverse-csv")

    # Baselines need the prior season for their preseason ratings, whatever the source
    seasons = [min(args.seasons) - 1, *args.seasons] if args.baseline else args.seasons
    market = _load_market(args, seasons)

    if args.baseline:
        market = market.join(baseline_predictions(args.baseline, market))
        market = market[market["season"].isin(args.seasons)]
        name, pred_col = args.baseline, "pred_home_margin"
    else:
        preds = pd.read_csv(args.predictions_csv)
        dupes = preds["game_id"][preds["game_id"].duplicated()].unique()
        if len(dupes):
            raise ValueError(
                f"{args.predictions_csv} has more than one prediction for {len(dupes)} games "
                f"(e.g. {dupes[0]}); keep one per game, such as the earliest"
            )
        keep = [
            c
            for c in [
                "game_id",
                args.pred_col,
                "pred_home_win_prob",
                "prediction_timestamp",
                LINE_AT_PREDICTION,
            ]
            if c in preds
        ]
        market = market.merge(preds[keep], on="game_id", how="inner")
        name, pred_col = args.predictions_csv, args.pred_col
        if market.empty:
            print(
                f"None of the {len(preds)} games in {args.predictions_csv} is a completed game "
                f"of {args.seasons[0]}-{args.seasons[-1]} with a closing line in this source yet",
                file=sys.stderr,
            )
            return 1

    try:
        result = run_benchmark(market, name, pred_col, args.min_edge, args.allow_implausible)
    except ImplausibleResultError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(to_json(result) if args.json else to_markdown(result))
    return 0


def predict_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog=f"{PROG} predict",
        description="Walk-forward baseline predictions for the regular-season games of one week "
        "that have not started, fit on every completed regular-season game of the earlier weeks "
        "and of the previous season. Grade the file later with --predictions-csv.",
    )
    ap.add_argument("--baseline", required=True, choices=PREDICT_BASELINES)
    ap.add_argument("--nflverse-csv", required=True, help="nflverse schedule CSV (games.csv)")
    ap.add_argument("--season", required=True, type=int)
    ap.add_argument("--week", required=True, type=int)
    ap.add_argument("--out", required=True, help="CSV to write")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing --out file")
    args = ap.parse_args(argv)
    if Path(args.out).exists() and not args.overwrite:
        ap.error(f"{args.out} exists; a prediction record is not replaced without --overwrite")

    data = Path(args.nflverse_csv).read_bytes()  # hash exactly the bytes the fit uses
    input_sha = hashlib.sha256(data).hexdigest()
    games = read_nflverse_games(io.BytesIO(data))
    preds = predict_week(games, args.baseline, args.season, args.week)
    skipped = preds.attrs["skipped"]
    if preds.empty:
        print(
            f"No game of {args.season} week {args.week} is still to be played "
            f"({len(skipped)} already started or completed); nothing written",
            file=sys.stderr,
        )
        return 1
    out_sha = write_predictions(preds, args.out, input_sha)
    print(
        f"{len(preds)} predictions for {args.season} week {args.week} ({args.baseline}, fit on "
        f"{preds.attrs['fit_games']} completed games); skipped {len(skipped)} games already "
        f"started or completed{': ' + ', '.join(skipped) if skipped else ''}",
        file=sys.stderr,
    )
    print(f"input  sha256 {input_sha}  {args.nflverse_csv}")
    print(f"output sha256 {out_sha}  {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "predict":
        return predict_main(argv[1:])
    if argv and argv[0] == "grade":
        argv = argv[1:]
    return grade_main(argv)


if __name__ == "__main__":
    sys.exit(main())
