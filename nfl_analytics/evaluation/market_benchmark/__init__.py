"""Benchmark spread predictions against the closing market (see __main__ for the CLI)."""

from nfl_analytics.evaluation.market_benchmark.baselines import (
    BASELINES,
    elo_walkforward,
    market_baseline,
    power_ratings_walkforward,
)
from nfl_analytics.evaluation.market_benchmark.guardrails import (
    ImplausibleResultError,
    check_home_cover_rate,
    check_plausibility,
)
from nfl_analytics.evaluation.market_benchmark.market import (
    load_market_from_csv,
    load_market_from_db,
    prepare_market,
)
from nfl_analytics.evaluation.market_benchmark.report import run_benchmark, to_markdown

__all__ = [
    "BASELINES",
    "ImplausibleResultError",
    "check_home_cover_rate",
    "check_plausibility",
    "elo_walkforward",
    "load_market_from_csv",
    "load_market_from_db",
    "market_baseline",
    "power_ratings_walkforward",
    "prepare_market",
    "run_benchmark",
    "to_markdown",
]
