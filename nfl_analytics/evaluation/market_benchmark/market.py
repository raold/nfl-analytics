"""
Closing-market data for benchmarking.

All spreads follow the nflverse convention used in ``games.spread_close``: the expected HOME
margin, positive when home is favored (see nfl_analytics/utils/spread_convention.py).

Three sources give the same frame: devdb01 ``games`` (load_market_from_db), a CSV export of
that table (load_market_from_csv), and an nflverse schedule file, games.csv from
https://github.com/nflverse/nflverse-data/releases/tag/schedules (load_market_from_nflverse),
which carries the current season up to the latest download.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_analytics.evaluation.market_benchmark.odds import (
    american_to_implied,
    devig_proportional,
    devig_shin,
)
from nfl_analytics.utils.spread_convention import assert_nflverse_convention

MARKET_COLUMNS = [
    "game_id",
    "season",
    "week",
    "kickoff",
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

_MARKET_SQL = """
SELECT game_id, season, week, kickoff, home_team, away_team, home_score, away_score,
       spread_close, home_moneyline, away_moneyline, home_spread_odds, away_spread_odds
FROM games
WHERE season = ANY(%(seasons)s)
  AND game_type = 'REG'
  AND home_score IS NOT NULL
  AND spread_close IS NOT NULL
ORDER BY kickoff, game_id
"""


def load_market_from_db(seasons: Iterable[int], conn) -> pd.DataFrame:
    """Completed regular-season games with closing lines from ``games``."""
    df = pd.read_sql_query(_MARKET_SQL, conn, params={"seasons": list(seasons)})
    return prepare_market(df)


def load_market_from_csv(path: str | Path, seasons: Iterable[int] | None = None) -> pd.DataFrame:
    """Same as load_market_from_db, from a CSV export with the games columns."""
    df = pd.read_csv(path)
    if seasons is not None:
        df = df[df["season"].isin(list(seasons))]
    df = df.dropna(subset=["home_score", "away_score", "spread_close"])
    return prepare_market(df)


NFLVERSE_TZ = "America/New_York"  # nflverse gameday + gametime are Eastern wall-clock time
_NFLVERSE_REQUIRED = [
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "gametime",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "spread_line",
]
_NFLVERSE_OPTIONAL = [
    "total_line",
    "home_moneyline",
    "away_moneyline",
    "home_spread_odds",
    "away_spread_odds",
    "over_odds",
    "under_odds",
]


def nflverse_kickoff(gameday: pd.Series, gametime: pd.Series) -> pd.Series:
    """UTC kickoff from nflverse gameday (YYYY-MM-DD) and gametime (HH:MM, Eastern).

    NaT when gametime is missing (all of 1999 upstream) or unparseable.
    """
    text = gameday.astype("string") + " " + gametime.astype("string").str.slice(0, 5)
    local = pd.to_datetime(text, format="%Y-%m-%d %H:%M", errors="coerce")
    return local.dt.tz_localize(NFLVERSE_TZ, ambiguous="NaT", nonexistent="NaT").dt.tz_convert(
        "UTC"
    )


def read_nflverse_games(path) -> pd.DataFrame:
    """Every game in an nflverse schedule CSV (a path or a file object), played or not, in
    ``games`` table columns.

    spread_line -> spread_close and total_line -> total_close (both already nflverse sign:
    expected home margin, positive = home favored); kickoff is the UTC instant.
    """
    raw = pd.read_csv(path, dtype={"gameday": "string", "gametime": "string"})
    missing = [c for c in _NFLVERSE_REQUIRED if c not in raw.columns]
    if missing:
        name = path if isinstance(path, (str, Path)) else "The input"
        raise ValueError(f"{name} is not an nflverse schedule (games.csv): no column {missing}")
    keep = _NFLVERSE_REQUIRED + [c for c in _NFLVERSE_OPTIONAL if c in raw.columns]
    df = raw[keep].rename(columns={"spread_line": "spread_close", "total_line": "total_close"})
    df["kickoff"] = nflverse_kickoff(df["gameday"], df["gametime"])
    return df


def load_market_from_nflverse(
    path: str | Path,
    seasons: Iterable[int] | None = None,
    game_types: Iterable[str] | None = ("REG",),
) -> pd.DataFrame:
    """Completed games with closing lines from an nflverse schedule CSV.

    Regular season only by default; ``game_types=None`` keeps every game type.
    """
    df = read_nflverse_games(path)
    if seasons is not None:
        df = df[df["season"].isin(list(seasons))]
    if game_types is not None:
        df = df[df["game_type"].isin(list(game_types))]
    df = df.dropna(subset=["home_score", "away_score", "spread_close"])
    return prepare_market(df)


def prepare_market(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize columns, add outcome and de-vigged market probabilities."""
    df = df.copy()
    for col in MARKET_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df.drop_duplicates("game_id").sort_values(["season", "week", "game_id"])
    df["home_margin"] = df["home_score"] - df["away_score"]
    assert_nflverse_convention(df)

    p_prop, p_shin = [], []
    for ml_home, ml_away in zip(df["home_moneyline"], df["away_moneyline"]):
        if pd.notna(ml_home) and pd.notna(ml_away) and ml_home != 0 and ml_away != 0:
            raw = (american_to_implied(ml_home), american_to_implied(ml_away))
            p_prop.append(devig_proportional(*raw)[0])
            p_shin.append(devig_shin(*raw)[0])
        else:
            p_prop.append(np.nan)
            p_shin.append(np.nan)
    df["market_home_win_prob"] = p_prop
    df["market_home_win_prob_shin"] = p_shin
    return df.reset_index(drop=True)
