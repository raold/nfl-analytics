"""
Benchmark NFL spread predictions against the closing market.

Examples:
    # Baselines from the bundled CSV (no database needed)
    python -m nfl_analytics.evaluation.market_benchmark --baseline power_ratings \\
        --market-csv data/historical_weather_2018_2024.csv --seasons 2019-2024

    # A model's predictions (CSV with game_id + expected home margin) against devdb01
    python -m nfl_analytics.evaluation.market_benchmark --predictions-csv preds.csv \\
        --pred-col pred_home_spread --seasons 2025 --db
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from nfl_analytics.evaluation.market_benchmark.baselines import BASELINES
from nfl_analytics.evaluation.market_benchmark.guardrails import ImplausibleResultError
from nfl_analytics.evaluation.market_benchmark.market import (
    load_market_from_csv,
    load_market_from_db,
)
from nfl_analytics.evaluation.market_benchmark.report import run_benchmark, to_json, to_markdown


def _seasons(text: str) -> list[int]:
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in text.split(",")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--baseline", choices=sorted(BASELINES))
    src.add_argument("--predictions-csv", help="CSV with game_id and a predicted home margin")
    ap.add_argument("--pred-col", default="pred_home_margin", help="column in --predictions-csv")
    mkt = ap.add_mutually_exclusive_group(required=True)
    mkt.add_argument("--market-csv", help="CSV export of games (scores + spread_close)")
    mkt.add_argument("--db", action="store_true", help="read games from devdb01 (DB_* env vars)")
    ap.add_argument("--seasons", required=True, type=_seasons, help="e.g. 2019-2024 or 2023,2024")
    ap.add_argument("--min-edge", type=float, default=1.0)
    ap.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    ap.add_argument("--allow-implausible", action="store_true")
    args = ap.parse_args(argv)

    if args.db:
        from nfl_analytics.utils.db_config import get_connection

        conn = get_connection()
        try:
            market = load_market_from_db(args.seasons, conn)
        finally:
            conn.close()
    else:
        # Baselines need the prior season for their preseason ratings
        seasons = [min(args.seasons) - 1, *args.seasons] if args.baseline else args.seasons
        market = load_market_from_csv(args.market_csv, seasons)

    if args.baseline:
        market["pred_home_margin"] = BASELINES[args.baseline](market)
        market = market[market["season"].isin(args.seasons)]
        name, pred_col = args.baseline, "pred_home_margin"
    else:
        preds = pd.read_csv(args.predictions_csv)
        keep = [
            c
            for c in ["game_id", args.pred_col, "pred_home_win_prob", "prediction_timestamp"]
            if c in preds
        ]
        market = market.merge(preds[keep], on="game_id", how="inner")
        name, pred_col = args.predictions_csv, args.pred_col

    try:
        result = run_benchmark(market, name, pred_col, args.min_edge, args.allow_implausible)
    except ImplausibleResultError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(to_json(result) if args.json else to_markdown(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
