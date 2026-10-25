"""
Baseline predictors of the home margin. Every baseline is walk-forward: a game's prediction
uses only games from earlier weeks, so these are honest yardsticks for any model.
"""

from __future__ import annotations

import inspect
import math
from collections import defaultdict

import numpy as np
import pandas as pd

HFA_POINTS = 2.5  # home-field advantage in points


def market_baseline(market: pd.DataFrame) -> pd.Series:
    """The closing line itself. A correct evaluation must find zero edge against it."""
    return market["spread_close"].astype(float).rename("pred_home_margin")


def power_ratings_walkforward(
    market: pd.DataFrame,
    hfa: float = HFA_POINTS,
    prior_games: float = 4.0,
    prior_shrink: float = 1 / 3,
) -> pd.Series:
    """Average point differential, blended with last season's rating early in the year.

    rating = (sum of this season's margins + prior_games * shrunk prior) / (n + prior_games),
    where the prior is last season's average margin pulled ``prior_shrink`` of the way to 0.
    """
    df = market.sort_values(["season", "week", "game_id"])
    final_prev: dict[tuple[int, str], float] = {}
    preds = pd.Series(np.nan, index=df.index, name="pred_home_margin")

    for season, sg in df.groupby("season", sort=True):
        sums: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)

        def rating(team: str) -> float:
            prior = final_prev.get((season - 1, team), 0.0) * (1 - prior_shrink)
            return (sums[team] + prior_games * prior) / (counts[team] + prior_games)

        for _, wk in sg.groupby("week", sort=True):
            for idx, g in wk.iterrows():
                preds[idx] = rating(g["home_team"]) - rating(g["away_team"]) + hfa
            for _, g in wk.iterrows():  # update only after the whole week is predicted
                m = g["home_margin"]
                sums[g["home_team"]] += m
                sums[g["away_team"]] -= m
                counts[g["home_team"]] += 1
                counts[g["away_team"]] += 1
        for team in counts:
            final_prev[(season, team)] = sums[team] / counts[team]

    return preds.reindex(market.index)


def elo_walkforward(
    market: pd.DataFrame,
    k: float = 20.0,
    hfa_elo: float = 48.0,
    revert: float = 1 / 3,
    points_per_elo: float = 1 / 25,
    mean_rating: float = 1505.0,
) -> pd.Series:
    """Elo with a margin-of-victory multiplier and preseason reversion to the mean."""
    return _elo_walkforward(market, k, hfa_elo, revert, points_per_elo, mean_rating)[0]


def _elo_walkforward(
    market: pd.DataFrame,
    k: float,
    hfa_elo: float,
    revert: float,
    points_per_elo: float,
    mean_rating: float,
) -> tuple[pd.Series, pd.Series]:
    """Expected home margin and home win probability (the Elo expectation) for every row."""
    df = market.sort_values(["season", "week", "game_id"])
    elo: dict[str, float] = defaultdict(lambda: mean_rating)
    preds = pd.Series(np.nan, index=df.index, name="pred_home_margin")
    probs = pd.Series(np.nan, index=df.index, name="pred_home_win_prob")
    current_season = None

    for (season, _week), wk in df.groupby(["season", "week"], sort=True):
        if season != current_season:
            for team in list(elo):
                elo[team] = elo[team] * (1 - revert) + mean_rating * revert
            current_season = season
        for idx, g in wk.iterrows():
            diff = elo[g["home_team"]] + hfa_elo - elo[g["away_team"]]
            preds[idx] = diff * points_per_elo
            probs[idx] = 1 / (1 + 10 ** (-diff / 400))
        for _, g in wk.iterrows():
            diff = elo[g["home_team"]] + hfa_elo - elo[g["away_team"]]
            expected = 1 / (1 + 10 ** (-diff / 400))
            margin = g["home_margin"]
            actual = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
            winner_diff = diff if margin >= 0 else -diff
            mov = math.log(abs(margin) + 1) * 2.2 / (winner_diff * 0.001 + 2.2)
            shift = k * mov * (actual - expected)
            elo[g["home_team"]] += shift
            elo[g["away_team"]] -= shift

    return preds.reindex(market.index), probs.reindex(market.index)


BASELINES = {
    "market": market_baseline,
    "power_ratings": power_ratings_walkforward,
    "elo": elo_walkforward,
}


def baseline_params(name: str) -> dict:
    """The constants a baseline runs with (its keyword defaults), for provenance records."""
    params = list(inspect.signature(BASELINES[name]).parameters.values())[1:]
    return {p.name: p.default for p in params}


def baseline_predictions(name: str, market: pd.DataFrame) -> pd.DataFrame:
    """``pred_home_margin`` for every row of ``market``, plus ``pred_home_win_prob`` when the
    baseline has one (Elo: the rating expectation 1 / (1 + 10 ** (-diff / 400)))."""
    if name == "elo":
        margin, prob = _elo_walkforward(market, **baseline_params("elo"))
        return pd.DataFrame({"pred_home_margin": margin, "pred_home_win_prob": prob})
    return BASELINES[name](market).rename("pred_home_margin").to_frame()
