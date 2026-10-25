"""Benchmark spread predictions against the closing market (see __main__ for the CLI)."""

from nfl_analytics.evaluation.market_benchmark.baselines import (
    BASELINES,
    baseline_params,
    baseline_predictions,
    elo_walkforward,
    market_baseline,
    power_ratings_walkforward,
)
from nfl_analytics.evaluation.market_benchmark.guardrails import (
    ImplausibleResultError,
    check_home_cover_rate,
    check_market_sanity,
    check_plausibility,
    market_sanity,
)
from nfl_analytics.evaluation.market_benchmark.market import (
    load_market_from_csv,
    load_market_from_db,
    load_market_from_nflverse,
    prepare_market,
    read_nflverse_games,
)
from nfl_analytics.evaluation.market_benchmark.predict import predict_week, write_predictions
from nfl_analytics.evaluation.market_benchmark.report import run_benchmark, to_json, to_markdown

__all__ = [
    "BASELINES",
    "ImplausibleResultError",
    "baseline_params",
    "baseline_predictions",
    "check_home_cover_rate",
    "check_market_sanity",
    "check_plausibility",
    "elo_walkforward",
    "load_market_from_csv",
    "load_market_from_db",
    "load_market_from_nflverse",
    "market_baseline",
    "market_sanity",
    "power_ratings_walkforward",
    "predict_week",
    "prepare_market",
    "read_nflverse_games",
    "run_benchmark",
    "to_json",
    "to_markdown",
    "write_predictions",
]
