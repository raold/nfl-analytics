"""
Prospective predictions for a prospective test.

Fits a walk-forward baseline on what is known now and writes one row per game of the target
week that has not started. Graded later with ``--predictions-csv``, the rows show whether they
were made before kickoff (``prediction_timestamp`` vs ``kickoff``) and, through
``spread_at_prediction``, how the line moved afterwards (closing-line value).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_analytics.evaluation.market_benchmark.baselines import (
    baseline_params,
    baseline_predictions,
)
from nfl_analytics.evaluation.market_benchmark.market import prepare_market

PREDICT_BASELINES = ("elo", "power_ratings")
OUTPUT_COLUMNS = [
    "game_id",
    "season",
    "week",
    "kickoff_utc",
    "home_team",
    "away_team",
    "pred_home_margin",
    "pred_home_win_prob",
    "spread_at_prediction",
    "prediction_timestamp",
    "baseline",
    "baseline_params",
    "input_sha256",
]


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predict_week(
    games: pd.DataFrame,
    baseline: str,
    season: int,
    week: int,
    as_of: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Predict the regular-season games of ``season`` week ``week`` that have not started.

    ``games`` holds every game, played or not (market.read_nflverse_games). The baseline is fit
    walk-forward on what was known at ``as_of`` (default now): completed regular-season games
    of the previous season (warm-up) and of the earlier weeks of ``season``, exactly what the
    grading path uses for the same games. Scores of games kicking off at or after ``as_of``
    are treated as unknown, and only games with no score that kick off after ``as_of`` are
    returned. ``prediction_timestamp`` is always the actual time of the call, so a prediction
    made for a game that has already kicked off is graded as retroactive.
    """
    if baseline not in PREDICT_BASELINES:
        raise ValueError(f"baseline must be one of {PREDICT_BASELINES}, not {baseline!r}")
    made = _utc_now()
    as_of = made if as_of is None else pd.Timestamp(as_of)
    as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")

    g = games[games["game_type"] == "REG"].copy()
    future = g["kickoff"].notna() & (g["kickoff"] >= as_of)
    g.loc[future, ["home_score", "away_score"]] = np.nan
    done = g["home_score"].notna() & g["away_score"].notna()
    earlier = (g["season"] == season - 1) | ((g["season"] == season) & (g["week"] < week))
    history = g[done & earlier & g["spread_close"].notna()]
    target = g[(g["season"] == season) & (g["week"] == week)]
    if target.empty:
        raise ValueError(f"No regular-season games in {season} week {week}")
    if history.empty:
        raise ValueError(f"No completed games before {season} week {week} to fit {baseline} on")
    upcoming = target.loc[~done[target.index] & (target["kickoff"] > as_of), "game_id"]

    frame = prepare_market(pd.concat([history, target]))
    frame = frame.join(baseline_predictions(baseline, frame))
    out = frame[frame["game_id"].isin(upcoming)].sort_values(["kickoff", "game_id"])
    result = pd.DataFrame(
        {
            "game_id": out["game_id"],
            "season": out["season"].astype(int),
            "week": out["week"].astype(int),
            "kickoff_utc": out["kickoff"].map(lambda t: t.isoformat()),
            "home_team": out["home_team"],
            "away_team": out["away_team"],
            "pred_home_margin": out["pred_home_margin"],
        }
    )
    if "pred_home_win_prob" in out:
        result["pred_home_win_prob"] = out["pred_home_win_prob"]
    result["spread_at_prediction"] = out["spread_close"].astype(float)
    result["prediction_timestamp"] = made.ceil("s").isoformat()
    result["baseline"] = baseline
    result["baseline_params"] = json.dumps(baseline_params(baseline), sort_keys=True)
    result = result.reset_index(drop=True)
    result.attrs["fit_games"] = len(history)
    result.attrs["skipped"] = sorted(set(target["game_id"]) - set(upcoming))
    return result


def write_predictions(preds: pd.DataFrame, path: str | Path, input_sha256: str) -> str:
    """Write predict_week output with the input file's sha256; return the output's sha256."""
    out = preds.assign(input_sha256=input_sha256)
    out[[c for c in OUTPUT_COLUMNS if c in out.columns]].to_csv(path, index=False)
    return sha256_file(path)
