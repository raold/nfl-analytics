#!/usr/bin/env python3
"""
Correctly timed, cleaned panel of public.odds_history, plus aggregate line-movement / CLV results.

public.odds_history holds The Odds API historical snapshots for 2020-2024, written by
scripts/backfill_historical_odds.py in Dec 2025. Its market_type labels are only the names of the
backfill's two API calls per game date, and neither is an opening or a closing line:

* 'closing' requested MIN(games.kickoff on that date) - 30 min. games.kickoff then held Eastern
  wall-clock time labelled UTC, so the request was 4.5 h (EDT) or 5.5 h (EST) before the first game.
* 'opening' requested 14:00 UTC on the game date (10:00 EDT / 09:00 EST).

No snapshot time was stored (captured_at is the insert time). This script rebuilds every row's
request time without API calls: rows split into API calls at captured_at gaps, and each call is
matched to the request schedule recomputed from the kickoffs the backfill read
(backups/local/games_before_repair_20261024.csv). True kickoffs come from games.kickoff (repaired
from nflverse on 2026-10-24). Lines are converted to the nflverse sign: line_home = -home_spread is
the expected home margin, positive when home is favored (nfl_analytics/utils/spread_convention.py).

The Odds API terms forbid redistributing the data as a standalone product, so row-level output goes
only to the gitignored backups/local/odds_history_panel/; the docs get aggregates only.

The database is read in a read-only session (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD).

Usage:
    uv run python scripts/odds_history_panel.py
    # optional: check the backfill read the pre-repair kickoffs (dump made right after it)
    docker exec -i nfl-analytics-pg-1 pg_restore -a -t games -f - \\
        < backups/nfl_analytics_20251208_odds_complete.dump > /tmp/games_20251208.copy
    uv run python scripts/odds_history_panel.py --dump-games-copy /tmp/games_20251208.copy
"""

from __future__ import annotations

import argparse
import functools
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_analytics.utils.db_config import get_db_config
from nfl_analytics.utils.spread_convention import assert_nflverse_convention, home_cover_margin

REPO = Path(__file__).resolve().parents[1]
OLD_KICKOFFS = REPO / "backups/local/games_before_repair_20261024.csv"
PRIVATE_DIR = REPO / "backups/local/odds_history_panel"
RESULTS_DIR = REPO / "docs/investigations/2026-10-line-movement"
PANEL_FILE = "odds_history_panel_2020_2024.parquet"
SNAPSHOT_FILE = "odds_history_snapshots_2020_2024.parquet"
COVERAGE_FILE = "coverage_2020_2024.csv"

SEASONS = (2020, 2021, 2022, 2023, 2024)
# Rows of one API call were inserted milliseconds apart; the backfill slept 0.3 s between calls.
CALL_GAP_SECONDS = 0.2
# A main-line price pair has no side above +125 and none below -200; anything else is an
# alternate line or a placeholder (checked against the distance from the same call's median line).
MAIN_LINE_MAX_PLUS = 125
MAIN_LINE_MIN_MINUS = -200
OUTLIER_POINTS = 3.0  # a normally priced quote this far from the same call's median line
STALE_CALL_MINUTES = 30  # call whose newest book update is this much older than the request
MIN_BOOKS = 5  # books needed for a snapshot consensus
GAME_DAY_HOURS = 12  # the latest pre-kickoff snapshot must be at most this far from kickoff
MOVE_POINTS = 1.0
MARGIN_SAMPLE_MIN = 200
STANDARD_PRICE = -110
N_BOOT = 10_000
SEED = 0

PULL_NAMES = {"closing": "first_kickoff_minus_30m", "opening": "fixed_1400_utc"}
# stored American price column -> decimal odds recovered from it (see decimal_odds)
DECIMAL_COLUMNS = {
    "home_spread_price": "home_spread_dec",
    "away_spread_price": "away_spread_dec",
    "over_price": "over_dec",
    "under_price": "under_dec",
    "home_moneyline": "home_ml_dec",
    "away_moneyline": "away_ml_dec",
}
PANEL_COLUMNS = [
    "game_id",
    "season",
    "week",
    "game_type",
    "home_team",
    "away_team",
    "kickoff_utc",
    "bookmaker",
    "source_label",
    "pull",
    "call_id",
    "request_at",
    "hours_to_kickoff",
    "last_update",
    "update_age_minutes",
    "captured_at",
    "call_last_update_max",
    "call_lag_minutes",
    "home_spread",
    "away_spread",
    "home_spread_price",
    "away_spread_price",
    "home_spread_dec",
    "away_spread_dec",
    "line_home",
    "over_under",
    "over_price",
    "under_price",
    "over_dec",
    "under_dec",
    "home_moneyline",
    "away_moneyline",
    "home_ml_dec",
    "away_ml_dec",
    "spread_close",
    "total_close",
    "home_score",
    "away_score",
    "home_margin",
    "flag_inplay",
    "flag_spread_price",
    "flag_spread_outlier",
    "flag_total_price",
    "flag_moneyline_price",
    "flag_stale_call",
    "clean_spread",
    "clean_total",
]
HOUR_BUCKETS = [0, 6, 12, 48, 96, 168, 336, np.inf]
HOUR_LABELS = ["0-6 h", "6-12 h", "12-48 h", "48-96 h", "96-168 h", "168-336 h", "336+ h"]

ODDS_SQL = """
SELECT captured_at, game_id, bookmaker, market_type, home_spread, away_spread,
       home_spread_price, away_spread_price, over_under, over_price, under_price,
       home_moneyline, away_moneyline, last_update
FROM odds_history
ORDER BY market_type, captured_at, game_id, bookmaker
"""

GAMES_SQL = """
SELECT game_id, season, week, game_type, home_team, away_team, kickoff,
       home_score, away_score, spread_close, total_close, home_moneyline, away_moneyline
FROM games
ORDER BY game_id
"""


# --------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------


def read_only_connection():
    import psycopg2

    conn = psycopg2.connect(**get_db_config(), options="-c default_transaction_read_only=on")
    conn.set_session(readonly=True)
    return conn


def query(conn, sql: str) -> pd.DataFrame:
    buf = io.StringIO()
    with conn.cursor() as cur:
        cur.copy_expert(f"COPY ({sql.strip()}) TO STDOUT WITH CSV HEADER", buf)
    buf.seek(0)
    return pd.read_csv(buf)


def utc(values) -> pd.Series:
    return pd.to_datetime(values, utc=True, format="ISO8601")


def hours(delta: pd.Series) -> pd.Series:
    return delta.dt.total_seconds() / 3600


def load_inputs(old_kickoffs: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conn = read_only_connection()
    try:
        odds = query(conn, ODDS_SQL)
        games = query(conn, GAMES_SQL)
    finally:
        conn.close()
    odds["captured_at"] = utc(odds["captured_at"])
    odds["last_update"] = utc(odds["last_update"])
    games["kickoff"] = utc(games["kickoff"])
    games["home_margin"] = games["home_score"] - games["away_score"]
    old = pd.read_csv(old_kickoffs, usecols=["game_id", "kickoff"])
    old = old.rename(columns={"kickoff": "old_kickoff"})
    old["old_kickoff"] = utc(old["old_kickoff"])
    old["season"] = old["game_id"].str[:4].astype(int)
    return odds, games, old


def dump_kickoff_check(copy_path: Path, old: pd.DataFrame) -> dict:
    """Compare the pre-repair kickoffs with a pg_restore COPY block of games from an older dump."""
    lines = copy_path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("COPY public.games "))
    cols = lines[start].split("(", 1)[1].split(")", 1)[0].split(", ")
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == "\\.")
    dump = pd.DataFrame([line.split("\t") for line in lines[start + 1 : end]], columns=cols)
    dump = dump[["game_id", "kickoff"]].replace("\\N", None)
    dump["kickoff"] = utc(dump["kickoff"])
    sub = old[old["season"].isin(SEASONS)].merge(dump, on="game_id", how="left")
    same = sub["old_kickoff"].eq(sub["kickoff"])
    return {"file": copy_path.name, "games": len(sub), "identical": int(same.sum())}


# --------------------------------------------------------------------------------------------
# Timing reconstruction
# --------------------------------------------------------------------------------------------


def request_schedule(old: pd.DataFrame) -> pd.DataFrame:
    """The two requests per game date in backfill_historical_odds.py (lines 359-378).

    get_nfl_game_dates() and get_earliest_kickoff() ran DATE(kickoff) and MIN(kickoff) in a UTC
    session on the pre-repair kickoffs (Eastern wall clock labelled UTC), so DATE() is the ET game
    date and the 'closing' request is that wall-clock time minus 30 min, read as UTC.
    """
    k = old[old["season"].isin(SEASONS)]
    if k["old_kickoff"].isna().any():
        raise ValueError("a 2020-2024 game has no pre-repair kickoff")
    k = k.assign(game_date=k["old_kickoff"].dt.date)
    first = k.groupby(["season", "game_date"], as_index=False)["old_kickoff"].min()
    day = pd.to_datetime(first["game_date"].astype(str)).dt.tz_localize("UTC")
    closing = first.assign(
        market_type="closing", request_at=first["old_kickoff"] - pd.Timedelta("30min")
    )
    opening = first.assign(market_type="opening", request_at=day + pd.Timedelta("14h"))
    sched = pd.concat([closing, opening], ignore_index=True)
    return sched.sort_values(["market_type", "request_at"]).reset_index(drop=True)


def split_calls(odds: pd.DataFrame, gap_seconds: float) -> pd.Series:
    """API-call number per row (rows sorted by label, captured_at): a jump > gap starts a call."""
    gap = odds.groupby("market_type")["captured_at"].diff().dt.total_seconds()
    return gap.gt(gap_seconds).groupby(odds["market_type"]).cumsum()


def match_calls(odds: pd.DataFrame, sched: pd.DataFrame, old: pd.DataFrame):
    """Assign every row the request time of its API call; validate the assignment."""
    odds = odds.sort_values(["market_type", "captured_at"], kind="stable").reset_index(drop=True)
    odds["call_seq"] = split_calls(odds, CALL_GAP_SECONDS)
    calls = (
        odds.groupby(["market_type", "call_seq"])
        .agg(
            captured_first=("captured_at", "min"),
            last_update_min=("last_update", "min"),
            last_update_max=("last_update", "max"),
            rows=("game_id", "size"),
            games=("game_id", "nunique"),
        )
        .reset_index()
    )
    old_dates = old.assign(game_date=old["old_kickoff"].dt.date)
    parts, check = [], {}
    for label, c in calls.groupby("market_type"):
        slots = pd.DatetimeIndex(
            sched.loc[sched["market_type"] == label, "request_at"]
        ).sort_values()
        # The API returns the latest snapshot at or before the requested time, and a book's
        # last_update is never after its snapshot, so a call's request is the first slot at or
        # after its newest last_update.
        pos = slots.searchsorted(pd.DatetimeIndex(c["last_update_max"]))
        if (pos >= len(slots)).any():
            raise ValueError(f"{label}: a call holds data newer than the last scheduled request")
        if pd.Series(pos).duplicated().any():
            raise ValueError(f"{label}: two API calls map to the same request")
        c = c.assign(request_at=pd.Series(slots[pos], index=c.index), slot=pos)
        c["lag_minutes"] = (c["request_at"] - c["last_update_max"]).dt.total_seconds() / 60
        order = c.sort_values("captured_first")["slot"].to_numpy()
        breaks = np.flatnonzero(np.diff(order) <= 0) + 1
        runs = []
        for seg in np.split(np.arange(len(order)), breaks):
            seg_slots = order[seg]
            runs.append(
                {
                    "calls": len(seg),
                    "first_request": str(slots[seg_slots.min()]),
                    "last_request": str(slots[seg_slots.max()]),
                }
            )
        unmatched = []
        for p in sorted(set(range(len(slots))) - set(pos)):
            req = slots[p]
            date = (req if label == "opening" else req + pd.Timedelta("30min")).date()
            ids = sorted(old_dates.loc[old_dates["game_date"] == date, "game_id"])
            unmatched.append({"request_at": str(req), "games_on_date": ids})
        robust = {
            str(g): int(split_calls(odds[odds["market_type"] == label], g).max()) + 1
            for g in (0.05, 0.5)
        }
        stale = c[c["lag_minutes"] > STALE_CALL_MINUTES]
        season = c["request_at"].dt.year - (c["request_at"].dt.month <= 6)
        check[label] = {
            "scheduled_requests": len(slots),
            "api_calls_found": len(c),
            "calls_at_other_gap_thresholds": robust,
            "requests_without_rows": unmatched,
            "capture_order_runs": runs,
            "lag_minutes_request_minus_newest_update": quantiles(c["lag_minutes"]),
            "lag_minutes_by_season": {
                int(s): quantiles(v, (0.0, 0.5, 1.0))
                for s, v in c.groupby(season)["lag_minutes"]
            },
            "calls_lag_over_30_min": [
                {"request_at": str(r.request_at), "lag_minutes": round(r.lag_minutes, 1)}
                for r in stale.itertuples()
            ],
        }
        parts.append(c)
    calls = pd.concat(parts, ignore_index=True)
    calls = calls.sort_values(["request_at", "market_type"]).reset_index(drop=True)
    calls["call_id"] = np.arange(1, len(calls) + 1)
    odds = odds.merge(
        calls[
            ["market_type", "call_seq", "call_id", "request_at", "last_update_max", "lag_minutes"]
        ],
        on=["market_type", "call_seq"],
        how="left",
        validate="many_to_one",
    ).rename(columns={"last_update_max": "call_last_update_max", "lag_minutes": "call_lag_minutes"})
    after = int((odds["last_update"] > odds["request_at"]).sum())
    if after:
        raise ValueError(f"{after} rows were updated after their reconstructed request time")
    check["rows_updated_after_request"] = after
    check["row_request_minus_last_update_minutes"] = quantiles(
        (odds["request_at"] - odds["last_update"]).dt.total_seconds() / 60
    )
    return odds.drop(columns="call_seq"), check


def label_timing_check(panel: pd.DataFrame) -> dict:
    """Per (game, label): newest last_update vs the newest request (the survey's proxy)."""
    g = panel.groupby(["game_id", "source_label"]).agg(
        lu=("last_update", "max"), req=("request_at", "max"), kick=("kickoff_utc", "first")
    )
    lag = (g["req"] - g["lu"]).dt.total_seconds() / 60
    w = g.unstack("source_label")
    later_lu = (w[("lu", "opening")] > w[("lu", "closing")]).mean()
    later_req = (w[("req", "opening")] > w[("req", "closing")]).mean()
    gap = hours(w[("lu", "opening")] - w[("lu", "closing")])
    out = {
        "pairs": len(g),
        "newest_request_minus_newest_update_minutes": quantiles(lag),
        "games_opening_newer_than_closing_by_last_update": round(float(later_lu), 4),
        "games_opening_newer_than_closing_by_request": round(float(later_req), 4),
        "opening_minus_closing_newest_update_hours": quantiles(gap),
    }
    for label in ("opening", "closing"):
        out[f"{label}_hours_before_kickoff_of_newest_update"] = quantiles(
            hours(w[("kick", label)] - w[("lu", label)])
        )
        out[f"{label}_hours_before_kickoff_of_newest_request"] = quantiles(
            hours(w[("kick", label)] - w[("req", label)])
        )
    return out


# --------------------------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------------------------


def backfill_american(decimal: float) -> int:
    """The backfill's price conversion, verbatim: int(american_odds(decimal)) (lines 162, 247-253)."""
    if decimal >= 2.0:
        american = (decimal - 1) * 100
    elif decimal > 1.0:
        american = -100 / (decimal - 1)
    else:
        american = 0
    return int(american)


@functools.lru_cache(maxsize=1)
def stored_price_preimages() -> dict[int, tuple[float, ...]]:
    """The two-place decimal odds (as the API returns them) behind each stored American price.

    int() truncates toward zero, so -110 (1.91 -> -109.9) is stored as -109, and +105
    (2.05 -> 104.99999 in floating point) as +104. A negative stored price comes from one decimal,
    a positive one from two (+104 from 2.04 or 2.05).
    """
    out: dict[int, list[float]] = {}
    for k in range(101, 100_001):  # decimal 1.01 .. 1000.00
        out.setdefault(backfill_american(k / 100), []).append(k / 100)
    return {a: tuple(d) for a, d in out.items()}


def decimal_odds(stored: pd.Series) -> pd.Series:
    """Decimal odds recovered from a stored American price (NaN when missing or 0).

    The mean of the two-place decimals the backfill would have stored as that value: exact for
    negative prices, within 0.005 for positive ones. A value that no two-place decimal produces
    (a quote with more precision) is converted with the exact formula.
    """
    pre = stored_price_preimages()
    values = pd.to_numeric(pd.Series(stored), errors="coerce").astype(float)
    lookup = {}
    for v in values.dropna().unique():
        if v == 0:
            lookup[v] = np.nan
        elif int(v) in pre:
            lookup[v] = float(np.mean(pre[int(v)]))
        else:
            lookup[v] = 1 + 100 / abs(v) if v < 0 else 1 + v / 100
    return values.map(lookup)


def american(decimal) -> np.ndarray:
    """Exact American odds of a decimal price (for display)."""
    d = np.asarray(decimal, dtype=float)
    return np.where(d < 2, -100 / (d - 1), (d - 1) * 100)


def price_encoding_summary(p: pd.DataFrame) -> dict:
    pre = stored_price_preimages()
    out = {}
    for market, cols in [
        ("spread", ["home_spread_price", "away_spread_price"]),
        ("total", ["over_price", "under_price"]),
        ("moneyline", ["home_moneyline", "away_moneyline"]),
    ]:
        v = pd.concat([p[c] for c in cols]).dropna()
        v = v[v != 0].astype(int)
        n = v.map(lambda a: len(pre.get(a, ())))
        out[market] = {
            "stored_prices": len(v),
            "from_one_two_place_decimal": int((n == 1).sum()),
            "from_two_two_place_decimals": int((n >= 2).sum()),
            "from_no_two_place_decimal": int((n == 0).sum()),
        }
    sp = pd.concat([p["home_spread_price"], p["away_spread_price"]]).dropna()
    out["spread prices stored as -109 / -110"] = [int(sp.eq(-109).sum()), int(sp.eq(-110).sum())]
    out["decimals behind stored values"] = {str(a): list(pre[a]) for a in (-109, -114, 100, 104)}
    return out


# --------------------------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------------------------


def price_flags(a: pd.Series, b: pd.Series, present: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(zero price, alternate-line price pair) for a two-sided market."""
    zero = present & (a.eq(0) | b.eq(0))
    hi, lo = np.maximum(a, b), np.minimum(a, b)
    alt = present & ~zero & ((hi > MAIN_LINE_MAX_PLUS) | (lo < MAIN_LINE_MIN_MINUS))
    return zero, alt


def build_panel(odds: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "game_id",
        "season",
        "week",
        "game_type",
        "home_team",
        "away_team",
        "kickoff",
        "spread_close",
        "total_close",
        "home_score",
        "away_score",
        "home_margin",
    ]
    p = odds.merge(games[keep], on="game_id", how="left", validate="many_to_one")
    if p["kickoff"].isna().any():
        raise ValueError("odds rows without a kickoff in games")
    p = p.rename(columns={"kickoff": "kickoff_utc", "market_type": "source_label"})
    p["pull"] = p["source_label"].map(PULL_NAMES)
    p["hours_to_kickoff"] = hours(p["kickoff_utc"] - p["request_at"])
    p["update_age_minutes"] = (p["request_at"] - p["last_update"]).dt.total_seconds() / 60
    p["line_home"] = -p["home_spread"]  # sportsbook home line -> nflverse expected home margin
    for col, dec in DECIMAL_COLUMNS.items():
        p[dec] = decimal_odds(p[col])

    spread = p["home_spread"].notna()
    total = p["over_under"].notna()
    p["flag_inplay"] = (p["request_at"] > p["kickoff_utc"]) | (p["last_update"] > p["kickoff_utc"])
    zero, alt = price_flags(p["home_spread_price"], p["away_spread_price"], spread)
    p["flag_spread_price"] = zero | alt
    p["_spread_zero"], p["_spread_alt"] = zero, alt
    tzero, talt = price_flags(p["over_price"], p["under_price"], total)
    p["flag_total_price"] = tzero | talt
    ml = p["home_moneyline"].notna()
    p["flag_moneyline_price"] = ml & (p["home_moneyline"].eq(0) | p["away_moneyline"].eq(0))
    p["flag_stale_call"] = p["call_lag_minutes"] > STALE_CALL_MINUTES

    base = spread & ~p["flag_inplay"] & ~p["flag_spread_price"]
    grp = p[base].groupby(["call_id", "game_id"])["line_home"]
    med = grp.transform("median")
    n = grp.transform("size")
    outlier = ((p.loc[base, "line_home"] - med).abs() >= OUTLIER_POINTS) & (n >= 3)
    p["flag_spread_outlier"] = False
    p.loc[outlier.index, "flag_spread_outlier"] = outlier
    p["clean_spread"] = base & ~p["flag_spread_outlier"]
    p["clean_total"] = total & ~p["flag_inplay"] & ~p["flag_total_price"]
    return p


def flag_summary(p: pd.DataFrame) -> dict:
    spread = p["home_spread"].notna()
    price = p[["home_spread_price", "away_spread_price"]]
    over200 = spread & price.abs().max(axis=1).gt(200)
    out = {}
    for name, mask in [
        ("flag_inplay", p["flag_inplay"]),
        ("flag_spread_price", p["flag_spread_price"]),
        ("flag_spread_price: zero price", p["_spread_zero"]),
        ("flag_spread_price: alternate-line price pair", p["_spread_alt"]),
        ("flag_spread_price: any |price| > 200", over200),
        ("flag_spread_outlier", p["flag_spread_outlier"]),
        ("flag_total_price", p["flag_total_price"]),
        ("flag_moneyline_price", p["flag_moneyline_price"]),
        ("flag_stale_call", p["flag_stale_call"]),
        ("clean_spread", p["clean_spread"]),
    ]:
        sub = p[mask]
        by = sub.groupby(["bookmaker", "season"]).size().sort_values(ascending=False)
        out[name] = {
            "rows": int(mask.sum()),
            "games": int(sub["game_id"].nunique()),
            "top_book_seasons": {f"{b} {s}": int(v) for (b, s), v in by.head(4).items()},
        }
    out["in-play rows by label"] = p[p["flag_inplay"]]["source_label"].value_counts().to_dict()
    ip = p[p["flag_inplay"]].drop_duplicates("game_id")
    out["in-play games by ET kickoff time"] = (
        ip["kickoff_utc"].dt.tz_convert("America/New_York").dt.strftime("%H:%M").value_counts()
    ).to_dict()
    sp = p[spread & ~p["flag_inplay"]]
    dev = (
        sp["line_home"] - sp.groupby(["call_id", "game_id"])["line_home"].transform("median")
    ).abs()
    out["alt-price rows: |line - call median| quantiles"] = quantiles(dev[sp["_spread_alt"]])
    out["main-price rows: |line - call median| quantiles"] = quantiles(
        dev[~sp["_spread_alt"] & ~sp["_spread_zero"]]
    )
    b = p[spread & p["bookmaker"].eq("betmgm") & p["season"].eq(2020)]
    out["betmgm 2020 spread rows"] = {
        "spread_rows": len(b),
        "home price |p| > 200": int(b["home_spread_price"].abs().gt(200).sum()),
        "either price |p| > 200": int(price.loc[b.index].abs().max(axis=1).gt(200).sum()),
        "alternate-line pair": int(b["_spread_alt"].sum()),
    }
    return out


# --------------------------------------------------------------------------------------------
# Snapshots and per-game lines
# --------------------------------------------------------------------------------------------


def build_snapshots(p: pd.DataFrame) -> pd.DataFrame:
    """Consensus (median of books) and best quotes per (game, API call), clean quotes only."""
    keys = ["game_id", "call_id"]
    q = p[p["clean_spread"]].sort_values(keys + ["bookmaker"], kind="stable")
    snap = q.groupby(keys).agg(
        season=("season", "first"),
        week=("week", "first"),
        game_type=("game_type", "first"),
        source_label=("source_label", "first"),
        pull=("pull", "first"),
        request_at=("request_at", "first"),
        kickoff_utc=("kickoff_utc", "first"),
        hours_to_kickoff=("hours_to_kickoff", "first"),
        n_books=("bookmaker", "nunique"),
        cons_line_home=("line_home", "median"),
        line_min=("line_home", "min"),
        line_max=("line_home", "max"),
        spread_close=("spread_close", "first"),
        home_margin=("home_margin", "first"),
    )
    # A home bettor wants the lowest expected-home-margin line, an away bettor the highest;
    # ties go to the better price.
    home = q.sort_values(["line_home", "home_spread_dec"], ascending=[True, False], kind="stable")
    away = q.sort_values(["line_home", "away_spread_dec"], ascending=[False, False], kind="stable")
    home = home.groupby(keys).head(1).set_index(keys)
    away = away.groupby(keys).head(1).set_index(keys)
    snap["best_home_line"] = home["line_home"]
    snap["best_home_price"] = home["home_spread_price"]
    snap["best_home_dec"] = home["home_spread_dec"]
    snap["best_home_book"] = home["bookmaker"]
    snap["best_away_line"] = away["line_home"]
    snap["best_away_price"] = away["away_spread_price"]
    snap["best_away_dec"] = away["away_spread_dec"]
    snap["best_away_book"] = away["bookmaker"]
    t = (
        p[p["clean_total"]]
        .groupby(keys)
        .agg(cons_total=("over_under", "median"), n_books_total=("bookmaker", "nunique"))
    )
    return snap.join(t).reset_index()


def pick(snap: pd.DataFrame, latest: bool) -> pd.DataFrame:
    s = snap.sort_values(["game_id", "hours_to_kickoff"], ascending=[True, latest], kind="stable")
    return s.groupby("game_id").head(1).set_index("game_id")


def per_game_lines(snap: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    pre = snap[snap["hours_to_kickoff"] > 0]
    out = games.set_index("game_id").loc[
        sorted(pre["game_id"].unique()),
        ["season", "week", "game_type", "spread_close", "home_margin"],
    ]
    picks = {
        "early_any": pick(pre, latest=False),
        "early3": pick(pre[pre["n_books"] >= 3], latest=False),
        "early5": pick(pre[pre["n_books"] >= MIN_BOOKS], latest=False),
        "late": pick(
            pre[(pre["hours_to_kickoff"] <= GAME_DAY_HOURS) & (pre["n_books"] >= MIN_BOOKS)],
            latest=True,
        ),
    }
    for name, df in picks.items():
        out[f"{name}_line"] = df["cons_line_home"]
        out[f"{name}_hours"] = df["hours_to_kickoff"]
        out[f"{name}_books"] = df["n_books"]
        out[f"{name}_call"] = df["call_id"]
    return out


def survey_lines(p: pd.DataFrame) -> pd.DataFrame:
    """The survey's definition: median over books of each book's first / last pre-kickoff quote
    (by last_update), all opening/closing rows with a spread, no flags."""
    pre = p[(p["last_update"] < p["kickoff_utc"]) & p["home_spread"].notna()]
    pre = pre.sort_values(["last_update", "request_at", "bookmaker"], kind="stable")
    by_book = pre.groupby(["game_id", "bookmaker"])["line_home"]
    early = by_book.first().groupby("game_id").median()
    late = by_book.last().groupby("game_id").median()
    g = pre.groupby("game_id")[["season", "week", "spread_close", "home_margin"]].first()
    return g.assign(early=early, late=late)


# --------------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------------


def quantiles(s: pd.Series, qs=(0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)) -> dict:
    s = pd.Series(s).dropna()
    if s.empty:
        return {}
    return {f"p{int(round(q * 100))}": round(float(s.quantile(q)), 3) for q in qs}


def block_draws(frame: pd.DataFrame, cols: list[str], n_boot: int, seed: int) -> dict:
    """Week-block bootstrap: column sums over (season, week) blocks resampled with replacement."""
    sums = frame.groupby(["season", "week"])[cols].sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    arr = sums.to_numpy(dtype=float)
    return {c: arr[idx, j].sum(axis=1) for j, c in enumerate(cols)}


def ci95(draws: np.ndarray) -> list[float]:
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def side_result(home_margin, line, side) -> pd.Series:
    """+1 win, -1 loss, 0 push for a bet on `side` (+1 home, -1 away) at an nflverse line."""
    return np.sign(home_cover_margin(home_margin, line) * side)


def clv_test(
    frame: pd.DataFrame,
    early_col: str,
    threshold: float,
    n_boot: int,
    seed: int,
    hours_col: str | None = None,
) -> dict:
    """Side the line moved toward (early -> close), graded at the early line and at the close.

    This prices a number `threshold` or more points better than the close; it is not a strategy,
    because the direction of the move is only known afterwards.
    """
    d = frame.dropna(subset=[early_col, "spread_close", "home_margin"])
    move = d["spread_close"] - d[early_col]
    d = d[move.abs() >= threshold].copy()
    d["side"] = np.sign(d["spread_close"] - d[early_col])  # +1: moved toward home
    res: dict = {"games": len(d), "weeks": int(d.groupby(["season", "week"]).ngroups)}
    res["mean_abs_move"] = round(float((d["spread_close"] - d[early_col]).abs().mean()), 3)
    res["early_hours_median"] = float(d[hours_col].median()) if hours_col else None
    cols = []
    for name, line in (("at_early_line", d[early_col]), ("at_close", d["spread_close"])):
        r = side_result(d["home_margin"], line, d["side"])
        d[f"{name}_w"], d[f"{name}_d"] = (r > 0).astype(float), (r != 0).astype(float)
        w, lo, pu = int((r > 0).sum()), int((r < 0).sum()), int((r == 0).sum())
        res[name] = {
            "wins": w,
            "losses": lo,
            "pushes": pu,
            "win_rate": w / (w + lo) if w + lo else math.nan,
            "roi_at_minus_110": (w * 100 / 110 - lo) / (w + lo) if w + lo else math.nan,
        }
        cols += [f"{name}_w", f"{name}_d"]
    b = block_draws(d, cols, n_boot, seed)
    r_e = b["at_early_line_w"] / b["at_early_line_d"]
    r_c = b["at_close_w"] / b["at_close_d"]
    res["at_early_line"]["win_rate_ci95"] = ci95(r_e)
    res["at_close"]["win_rate_ci95"] = ci95(r_c)
    res["difference_ci95"] = ci95(r_e - r_c)
    return res


class CoverModel:
    """P(home covers a line | fair line) from historical margins at a similar closing line.

    A game with close c and home margin m is the mirror image of close -c and margin -m, so both
    orientations are pooled. The window around the fair line (+/- 0.5 pt) widens in half points
    until it holds MARGIN_SAMPLE_MIN observations. Pushes are excluded, as in every ATS rate here.
    """

    def __init__(self, games: pd.DataFrame):
        d = games.dropna(subset=["spread_close", "home_margin"])
        self.close = np.concatenate([d["spread_close"], -d["spread_close"]]).astype(float)
        self.margin = np.concatenate([d["home_margin"], -d["home_margin"]]).astype(float)
        self.n_games = len(d)
        self.seasons = f"{int(d['season'].min())}-{int(d['season'].max())}"
        self._cache: dict = {}
        self.widened = 0

    def p_home(self, line: float, fair: float) -> float:
        key = (float(line), float(fair))
        if key not in self._cache:
            w = 0.5
            while True:
                sel = np.abs(self.close - fair) <= w
                if sel.sum() >= MARGIN_SAMPLE_MIN or w >= 3.0:
                    break
                w += 0.5
            self.widened += w > 0.5
            m = self.margin[sel]
            win, loss = (m > line).sum(), (m < line).sum()
            self._cache[key] = win / (win + loss)
        return self._cache[key]


def breakeven(price) -> np.ndarray:
    """Win probability needed to break even at an American price."""
    a = np.asarray(price, dtype=float)
    return np.where(a < 0, -a / (100 + np.abs(a)), 100 / (100 + np.abs(a)))


def dispersion(snaps: pd.DataFrame, model: CoverModel, n_boot: int, seed: int) -> dict:
    """Best available line vs the consensus at the same snapshot, both sides."""
    s = snaps.copy()
    s["range"] = s["line_max"] - s["line_min"]
    s["gain_home"] = s["cons_line_home"] - s["best_home_line"]
    s["gain_away"] = s["best_away_line"] - s["cons_line_home"]
    p_cons = [model.p_home(c, c) for c in s["cons_line_home"]]
    p_best_h = [
        model.p_home(b, c) for b, c in zip(s["best_home_line"], s["cons_line_home"], strict=True)
    ]
    p_best_a = [
        model.p_home(b, c) for b, c in zip(s["best_away_line"], s["cons_line_home"], strict=True)
    ]
    s["dp_home"] = np.array(p_best_h) - np.array(p_cons)
    s["dp_away"] = np.array(p_cons) - np.array(p_best_a)  # away covers with probability 1 - p
    gains = pd.concat([s["gain_home"], s["gain_away"]])
    dps = pd.concat([s["dp_home"], s["dp_away"]])
    dec = pd.concat([s["best_home_dec"], s["best_away_dec"]])
    # Net of price: the win probability gained minus the extra breakeven of the quote's price
    # over -110, the price assumed for the consensus. The recovered decimal is used because the
    # stored American price is truncated (a -110 quote is stored as -109).
    cost = 1 / dec - float(breakeven(STANDARD_PRICE))
    net = dps - cost
    # realized ATS at the best line vs the consensus, both sides of every game
    rh = side_result(s["home_margin"], s["best_home_line"], 1)
    ra = side_result(s["home_margin"], s["best_away_line"], -1)
    ch = side_result(s["home_margin"], s["cons_line_home"], 1)
    ca = side_result(s["home_margin"], s["cons_line_home"], -1)
    s["best_w"] = (rh > 0).astype(float) + (ra > 0).astype(float)
    s["best_d"] = (rh != 0).astype(float) + (ra != 0).astype(float)
    s["cons_w"] = (ch > 0).astype(float) + (ca > 0).astype(float)
    s["cons_d"] = (ch != 0).astype(float) + (ca != 0).astype(float)
    b = block_draws(s, ["best_w", "best_d", "cons_w", "cons_d"], n_boot, seed)
    best_rate = s["best_w"].sum() / s["best_d"].sum()
    cons_rate = s["cons_w"].sum() / s["cons_d"].sum()
    return {
        "snapshots": len(s),
        "hours_to_kickoff_median": round(float(s["hours_to_kickoff"].median()), 2),
        "books_median": float(s["n_books"].median()),
        "share_books_disagree": round(float((s["range"] > 0).mean()), 4),
        "share_range_ge_1": round(float((s["range"] >= 1).mean()), 4),
        "mean_range": round(float(s["range"].mean()), 3),
        "mean_gain_points_per_side": round(float(gains.mean()), 3),
        "share_sides_gain_ge_0_5": round(float((gains >= 0.5).mean()), 4),
        "share_sides_gain_ge_1": round(float((gains >= 1).mean()), 4),
        "implied_win_prob_gain_pp": round(float(100 * dps.mean()), 2),
        "implied_win_prob_gain_pp_when_gain_ge_0_5": round(
            float(100 * dps[gains >= 0.5].mean()), 2
        ),
        "best_quote_decimal_median": round(float(dec.median()), 3),
        "best_quote_american_median": round(float(american(dec.median())), 1),
        "share_best_quotes_at_minus_110_or_better": round(float((cost <= 1e-9).mean()), 4),
        "price_cost_vs_minus_110_pp": round(float(100 * cost.mean()), 2),
        "implied_gain_net_of_price_pp": round(float(100 * net.mean()), 2),
        "realized_win_rate_best": round(float(best_rate), 4),
        "realized_win_rate_best_ci95": ci95(b["best_w"] / b["best_d"]),
        "realized_win_rate_consensus": round(float(cons_rate), 4),
        "realized_gain_pp_ci95": [
            round(100 * x, 2) for x in ci95(b["best_w"] / b["best_d"] - b["cons_w"] / b["cons_d"])
        ],
    }


# --------------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------------


def coverage_table(games: pd.DataFrame, p: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    g = games[games["season"].isin(SEASONS)].copy()
    et = g["kickoff"].dt.tz_convert("America/New_York")
    g["kickoff_et"] = et.dt.strftime("%Y-%m-%d %H:%M %Z")
    g["weekday_et"] = et.dt.day_name()
    g["rams_game"] = g["home_team"].eq("LA") | g["away_team"].eq("LA")
    g["utc_date_after_et_date"] = g["kickoff"].dt.date != et.dt.date
    pre = p[p["hours_to_kickoff"] > 0]
    per = p.groupby("game_id").agg(
        n_rows=("bookmaker", "size"),
        n_calls=("call_id", "nunique"),
        earliest_hours=("hours_to_kickoff", "max"),
    )
    per["latest_pre_kickoff_hours"] = pre.groupby("game_id")["hours_to_kickoff"].min()
    g = g.merge(per, on="game_id", how="left")
    g["has_odds"] = g["n_rows"].notna()
    g["has_game_day_consensus"] = g["game_id"].isin(lines.index[lines["late_line"].notna()])
    g["coverage"] = np.select(
        [
            g["rams_game"],
            g["utc_date_after_et_date"] & g["has_odds"],
            g["utc_date_after_et_date"],
            g["has_game_day_consensus"],
        ],
        [
            "missing: Rams game",
            "early snapshots only: night kickoff",
            "missing: night kickoff",
            "covered",
        ],
        default="other",
    )
    cols = [
        "game_id",
        "season",
        "week",
        "game_type",
        "home_team",
        "away_team",
        "kickoff",
        "kickoff_et",
        "weekday_et",
        "rams_game",
        "utc_date_after_et_date",
        "has_odds",
        "has_game_day_consensus",
        "n_rows",
        "n_calls",
        "earliest_hours",
        "latest_pre_kickoff_hours",
        "coverage",
    ]
    return g[cols].rename(columns={"kickoff": "kickoff_utc"})


def slot_of(cov: pd.DataFrame) -> pd.Series:
    t = pd.to_datetime(cov["kickoff_utc"], utc=True).dt.tz_convert("America/New_York")
    hour = t.dt.hour + t.dt.minute / 60
    day = t.dt.day_name()
    return pd.Series(
        np.select(
            [
                cov["game_type"].ne("REG"),
                hour < 12,
                day.eq("Sunday") & (hour < 15),
                day.eq("Sunday") & (hour < 19),
                day.eq("Sunday"),
                day.eq("Monday"),
                day.eq("Thursday"),
                day.eq("Saturday"),
            ],
            [
                "playoffs",
                "international morning (09:30 ET)",
                "Sunday early (13:00 ET)",
                "Sunday late afternoon",
                "Sunday night",
                "Monday",
                "Thursday",
                "Saturday",
            ],
            default="other weekday",
        ),
        index=cov.index,
    )


def analyse(p, snap, lines, cov, model, n_boot, seed) -> dict:
    R: dict = {}
    u = lines[lines["late_line"].notna()].copy()  # games with a game-day multi-book consensus
    R["universe"] = {
        "games_with_odds": len(lines),
        "games_with_game_day_consensus": len(u),
        "excluded_no_game_day_consensus": sorted(lines.index[lines["late_line"].isna()]),
        "early5_is_game_day_snapshot": int((u["early5_hours"] <= GAME_DAY_HOURS).sum()),
        "hours": {
            name: quantiles(u[f"{name}_hours"], (0.1, 0.5, 0.9))
            for name in ("early_any", "early3", "early5", "late")
        },
        "books": {
            name: float(u[f"{name}_books"].median()) for name in ("early_any", "early5", "late")
        },
    }

    # sign convention: game-day consensus vs nflverse close
    d = u["late_line"] - u["spread_close"]
    flip = -u["late_line"] - u["spread_close"]
    R["sign_check"] = {
        "games": len(u),
        "mae": round(float(d.abs().mean()), 3),
        "share_exact": round(float((d.abs() < 0.01).mean()), 4),
        "share_within_0_5": round(float((d.abs() <= 0.5).mean()), 4),
        "share_within_1": round(float((d.abs() <= 1).mean()), 4),
        "corr": round(float(np.corrcoef(u["late_line"], u["spread_close"])[0, 1]), 4),
        "mae_without_sign_flip": round(float(flip.abs().mean()), 3),
        "corr_without_sign_flip": round(
            float(np.corrcoef(-u["late_line"], u["spread_close"])[0, 1]), 4
        ),
        "mae_by_season": {
            int(k): round(float(v), 3) for k, v in d.abs().groupby(u["season"]).mean().items()
        },
        "games_differing_by_3_or_more": int((d.abs() >= 3).sum()),
    }

    # (a) accuracy vs the final margin, same games for every line
    acc = u.dropna(
        subset=["early_any_line", "early5_line", "late_line", "spread_close", "home_margin"]
    )
    err = {
        name: (acc["home_margin"] - acc[col]).abs()
        for name, col in [
            ("earliest snapshot (any books)", "early_any_line"),
            (f"earliest snapshot with >= {MIN_BOOKS} books", "early5_line"),
            ("latest pre-kickoff snapshot (game day)", "late_line"),
            ("nflverse close (games.spread_close)", "spread_close"),
        ]
    }
    frame = acc[["season", "week"]].assign(**{f"e{i}": e for i, e in enumerate(err.values())})
    frame["one"] = 1.0
    b = block_draws(frame, [f"e{i}" for i in range(len(err))] + ["one"], n_boot, seed)
    close_draw = b[f"e{len(err) - 1}"] / b["one"]
    hours_col = {0: "early_any_hours", 1: "early5_hours", 2: "late_hours"}
    R["a_accuracy"] = {"games": len(acc), "rows": []}
    for i, (name, e) in enumerate(err.items()):
        is_close = i == len(err) - 1
        R["a_accuracy"]["rows"].append(
            {
                "line": name,
                "median_hours_before_kickoff": (
                    round(float(acc[hours_col[i]].median()), 1) if i in hours_col else None
                ),
                "mae": round(float(e.mean()), 3),
                "mae_minus_close": round(float(e.mean() - err[list(err)[-1]].mean()), 3),
                "mae_minus_close_ci95": (
                    None if is_close else ci95(b[f"e{i}"] / b["one"] - close_draw)
                ),
            }
        )

    # (b) how far snapshot consensus lines sit from the close, by hours before kickoff
    s = snap[snap["game_id"].isin(u.index) & (snap["hours_to_kickoff"] > 0)]
    s = s[s["n_books"] >= MIN_BOOKS].copy()
    s["abs_to_close"] = (s["cons_line_home"] - s["spread_close"]).abs()
    s["bucket"] = pd.cut(s["hours_to_kickoff"], HOUR_BUCKETS, labels=HOUR_LABELS)
    rows = []
    for label, grp in s.groupby("bucket", observed=True):
        a = grp["abs_to_close"]
        rows.append(
            {
                "hours_before_kickoff": str(label),
                "snapshots": len(grp),
                "games": int(grp["game_id"].nunique()),
                "median_books": float(grp["n_books"].median()),
                "mean_abs_diff": round(float(a.mean()), 3),
                "median_abs_diff": round(float(a.median()), 3),
                "share_ge_0_5": round(float((a >= 0.5).mean()), 4),
                "share_ge_1": round(float((a >= 1).mean()), 4),
                "share_ge_2": round(float((a >= 2).mean()), 4),
                "share_ge_3": round(float((a >= 3).mean()), 4),
            }
        )
    e5 = (u["early5_line"] - u["spread_close"]).abs()
    lt = (u["late_line"] - u["spread_close"]).abs()
    e5_lt = (u["early5_line"] - u["late_line"]).abs()
    R["b_movement"] = {
        "by_hours_bucket": rows,
        "per_game": {
            "games": len(u),
            "early5_hours_median": round(float(u["early5_hours"].median()), 1),
            "abs_early5_minus_close": quantiles(e5, (0.25, 0.5, 0.75, 0.9, 0.95)),
            "mean_abs_early5_minus_close": round(float(e5.mean()), 3),
            "share_early5_close_ge": {
                str(t): round(float((e5 >= t).mean()), 4) for t in (0.5, 1, 2, 3)
            },
            "mean_abs_early5_minus_late": round(float(e5_lt.mean()), 3),
            "mean_abs_late_minus_close": round(float(lt.mean()), 3),
            "share_late_equals_close": round(float((lt < 0.01).mean()), 4),
            "share_late_close_ge_1": round(float((lt >= 1).mean()), 4),
        },
    }

    # (c) what closing-line value is worth
    R["c_clv"] = {
        "main": clv_test(u, "early5_line", MOVE_POINTS, n_boot, seed, "early5_hours"),
        "sensitivity": [],
    }
    for label, name, t in [
        (f"earliest >= {MIN_BOOKS} books, move >= 0.5", "early5", 0.5),
        (f"earliest >= {MIN_BOOKS} books, move >= 2", "early5", 2.0),
        (f"earliest >= {MIN_BOOKS} books, move >= 3", "early5", 3.0),
        ("earliest snapshot (any books), move >= 1", "early_any", 1.0),
        ("earliest >= 3 books, move >= 1", "early3", 1.0),
    ]:
        res = clv_test(u, f"{name}_line", t, n_boot, seed, f"{name}_hours")
        R["c_clv"]["sensitivity"].append({"definition": label, **res})
    # nflverse 2020 closing lines produce no pushes at all (survey); check the result without them
    res = clv_test(u[u["season"] != 2020], "early5_line", MOVE_POINTS, n_boot, seed, "early5_hours")
    R["c_clv"]["sensitivity"].append(
        {"definition": f"earliest >= {MIN_BOOKS} books, move >= 1, without 2020", **res}
    )
    sv = survey_lines(p)
    sv_res = clv_test(sv, "early", MOVE_POINTS, n_boot, seed)
    R["c_clv"]["survey_definition"] = {
        "definition": "median of each book's first pre-kickoff quote by last_update, all games, "
        "no flags (survey odds_analysis.py)",
        **sv_res,
        "mae_vs_margin": {
            "early": round(float((sv["home_margin"] - sv["early"]).abs().mean()), 3),
            "late": round(float((sv["home_margin"] - sv["late"]).abs().mean()), 3),
            "close": round(float((sv["home_margin"] - sv["spread_close"]).abs().mean()), 3),
            "games": len(sv),
        },
    }

    # (d) book dispersion at the same snapshot
    late = (
        snap.set_index(["game_id", "call_id"])
        .loc[list(zip(u.index, u["late_call"].astype(int), strict=True))]
        .reset_index()
    )
    early = snap.set_index(["game_id", "call_id"]).loc[
        list(zip(u.index, u["early5_call"].astype(int), strict=True))
    ]
    early = early[early["hours_to_kickoff"] > GAME_DAY_HOURS].reset_index()
    R["d_dispersion"] = {
        "latest pre-kickoff snapshot": dispersion(late, model, n_boot, seed),
        f"earliest >= {MIN_BOOKS}-book snapshot (before game day)": dispersion(
            early, model, n_boot, seed
        ),
        "margin_model": {
            "games": model.n_games,
            "seasons": model.seasons,
            "lookups_with_widened_window": int(model.widened),
            "p_home_at_fair_line_examples": {
                str(c): round(float(model.p_home(c, c)), 4) for c in (1.0, 2.5, 3.0, 3.5, 7.0)
            },
            "value_of_half_point_examples_pp": {
                f"{c} -> {c - 0.5}": round(
                    float(100 * (model.p_home(c - 0.5, c) - model.p_home(c, c))), 2
                )
                for c in (1.5, 3.0, 3.5, 7.0, 7.5)
            },
        },
    }

    # coverage
    cov = cov.assign(slot=slot_of(cov))
    groups = [
        ("All 2020-2024 games", pd.Series(True, index=cov.index)),
        ("Rams games (games use 'LA'; the backfill mapped the Rams to 'LAR')", cov["rams_game"]),
        (
            "Other games kicking off on the next UTC date (>= 20:00 EDT / 19:00 EST)",
            ~cov["rams_game"] & cov["utc_date_after_et_date"],
        ),
        ("All other games", ~cov["rams_game"] & ~cov["utc_date_after_et_date"]),
    ]
    R["coverage"] = {
        "by_cause": [
            {
                "group": name,
                "games": int(m.sum()),
                "with_any_odds": int(cov.loc[m, "has_odds"].sum()),
                "with_game_day_consensus": int(cov.loc[m, "has_game_day_consensus"].sum()),
            }
            for name, m in groups
        ],
        "by_slot": [
            {
                "slot": slot,
                "games": len(grp),
                "with_any_odds": int(grp["has_odds"].sum()),
                "with_game_day_consensus": int(grp["has_game_day_consensus"].sum()),
            }
            for slot, grp in cov.groupby("slot", sort=False)
        ],
        "by_season": [
            {
                "season": int(season),
                "games": len(grp),
                "with_any_odds": int(grp["has_odds"].sum()),
                "with_game_day_consensus": int(grp["has_game_day_consensus"].sum()),
            }
            for season, grp in cov.groupby("season")
        ],
        "night_games_with_odds_latest_snapshot_hours": quantiles(
            cov.loc[
                cov["coverage"].eq("early snapshots only: night kickoff"),
                "latest_pre_kickoff_hours",
            ],
            (0.0, 0.5, 1.0),
        ),
        "by_slot_and_class": {
            slot: {k: int(v) for k, v in row.items() if v}
            for slot, row in pd.crosstab(cov["slot"], cov["coverage"]).iterrows()
        },
        "next_utc_date_kickoffs_et_non_rams": (
            pd.to_datetime(cov.loc[groups[2][1], "kickoff_utc"], utc=True)
            .dt.tz_convert("America/New_York")
            .dt.strftime("%H:%M %Z")
            .value_counts()
            .to_dict()
        ),
    }
    return R


def raw_inventory(odds: pd.DataFrame, games: pd.DataFrame) -> dict:
    o = odds.assign(game_season=odds["game_id"].str[:4].astype(int))
    by = o.groupby(["game_season", "market_type"]).size()
    gl = odds[odds["market_type"] == "game_lines"].merge(
        games[["game_id", "season", "home_team", "away_team", "kickoff"]], on="game_id", how="left"
    )
    gl["hours_after_kickoff"] = hours(gl["last_update"] - gl["kickoff"])
    wrong = gl[gl["hours_after_kickoff"] > 24 * 30]
    later = wrong.merge(
        games[["game_id", "home_team", "away_team", "kickoff"]],
        on=["home_team", "away_team"],
        suffixes=("", "_match"),
    )
    later = later[(later["kickoff_match"] - later["last_update"]).abs() <= pd.Timedelta("10D")]
    return {
        "rows": len(odds),
        "rows_by_label": odds["market_type"].value_counts().to_dict(),
        "rows_by_game_season_and_label": {f"{s} {m}": int(v) for (s, m), v in by.items()},
        "books": int(odds.loc[odds["market_type"] != "game_lines", "bookmaker"].nunique()),
        "captured_at_range": {
            m: [str(g["captured_at"].min()), str(g["captured_at"].max())]
            for m, g in odds.groupby("market_type")
        },
        "game_lines": {
            "rows": len(gl),
            "game_ids": int(gl["game_id"].nunique()),
            "rows_recorded_over_30_days_after_their_game": len(wrong),
            "game_ids_recorded_over_30_days_after_their_game": int(wrong["game_id"].nunique()),
            "hours_after_kickoff_range": [
                round(float(wrong["hours_after_kickoff"].min()), 1),
                round(float(wrong["hours_after_kickoff"].max()), 1),
            ],
            "same_matchup_within_10_days_of_snapshot": sorted(
                f"{a} -> {b}"
                for a, b in later[["game_id", "game_id_match"]].drop_duplicates().values
            ),
        },
        "games_2025_in_db": int((games["season"] == 2025).sum()),
        "games_2026_in_db": int((games["season"] == 2026).sum()),
    }


# --------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else round(float(x), 6)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def pct(x, digits: int = 1) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{100 * x:.{digits}f}%"


def pct_ci(ci) -> str:
    return f"{100 * ci[0]:.1f} to {100 * ci[1]:.1f}"


def table(header: list[str], rows: list[list]) -> str:
    def cell(v) -> str:
        return str(v).replace("|", "\\|")

    out = [
        "| " + " | ".join(cell(h) for h in header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    out += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def runs_text(runs: list[dict]) -> str:
    return "; ".join(
        f"{r['calls']} calls ({r['first_request'][:10]} to {r['last_request'][:10]})" for r in runs
    )


def lag_text(q: dict) -> str:
    return f"{q['p0']:.1f} / {q['p50']:.1f} / {q['p95']:.1f} / {q['p100']:.1f}"


def stale_text(T: dict) -> str:
    calls = T["closing"]["calls_lag_over_30_min"] + T["opening"]["calls_lag_over_30_min"]
    return "; ".join(f"{c['request_at']} ({c['lag_minutes']:.0f} min)" for c in calls) or "none"


def timing_table(T: dict) -> str:
    rows = []
    for name, key, fmt in [
        ("requests scheduled (game dates)", "scheduled_requests", str),
        (f"API calls found (captured_at gap > {CALL_GAP_SECONDS} s)", "api_calls_found", str),
        (
            "calls found at gaps 0.05 s / 0.5 s",
            "calls_at_other_gap_thresholds",
            lambda d: f"{d['0.05']} / {d['0.5']}",
        ),
        (
            "requests with no rows (games that day)",
            "requests_without_rows",
            lambda us: "; ".join(
                f"{u['request_at']} ({', '.join(u['games_on_date'])})" for u in us
            ),
        ),
        ("capture-order runs", "capture_order_runs", runs_text),
        (
            "request - newest update in the call, min (min / p50 / p95 / max)",
            "lag_minutes_request_minus_newest_update",
            lag_text,
        ),
        (
            "the same by season (min / p50 / max)",
            "lag_minutes_by_season",
            lambda d: "; ".join(
                f"{s}: {q['p0']:.1f} / {q['p50']:.1f} / {q['p100']:.1f}" for s, q in d.items()
            ),
        ),
        (f"calls > {STALE_CALL_MINUTES} min", "calls_lag_over_30_min", len),
    ]:
        rows.append([name, fmt(T["closing"][key]), fmt(T["opening"][key])])
    return table(
        ["check", f"'closing' ({PULL_NAMES['closing']})", f"'opening' ({PULL_NAMES['opening']})"],
        rows,
    )


def hdr(text: str) -> list[str]:
    return text.split("; ")


def count_rows(records: list[dict], first: str) -> list[list]:
    return [
        [r[first], r["games"], r["with_any_odds"], r["with_game_day_consensus"]] for r in records
    ]


def clv_row(name: str, r: dict) -> list:
    e, c, dci = r["at_early_line"], r["at_close"], r["difference_ci95"]
    diff = 100 * (e["win_rate"] - c["win_rate"])
    hours_med = r.get("early_hours_median")
    return [
        name,
        r["games"],
        r["weeks"],
        "" if hours_med is None else f"{hours_med:.0f}",
        r["mean_abs_move"],
        f"{e['wins']}-{e['losses']}-{e['pushes']}",
        f"{pct(e['win_rate'])} ({pct_ci(e['win_rate_ci95'])})",
        pct(e["roi_at_minus_110"]),
        f"{c['wins']}-{c['losses']}-{c['pushes']}",
        f"{pct(c['win_rate'])} ({pct_ci(c['win_rate_ci95'])})",
        pct(c["roi_at_minus_110"]),
        f"{diff:+.1f} ({100 * dci[0]:+.1f} to {100 * dci[1]:+.1f})",
    ]


def accuracy_row(r: dict) -> list:
    ci = r["mae_minus_close_ci95"]
    h = r["median_hours_before_kickoff"]
    return [
        r["line"],
        "" if h is None else h,
        f"{r['mae']:.3f}",
        "" if ci is None else f"{r['mae_minus_close']:+.3f}",
        "" if ci is None else f"{ci[0]:+.3f} to {ci[1]:+.3f}",
    ]


def price_rows(R: dict) -> list[list]:
    keys = ("stored_prices", "from_one_two_place_decimal", "from_two_two_place_decimals")
    return [
        [m] + [R["prices"][m][k] for k in keys + ("from_no_two_place_decimal",)]
        for m in ("spread", "total", "moneyline")
    ]


def results_markdown(R: dict) -> str:
    T, F, A, raw, P = R["timing"], R["flags"], R["analysis"], R["raw"], R["prices"]
    gl, U, S = raw["game_lines"], A["universe"], A["sign_check"]
    hrange = gl["hours_after_kickoff_range"]
    inputs = [
        ["odds_history rows", raw["rows"]],
        ["rows by label", raw["rows_by_label"]],
        ["panel rows (opening + closing)", R["panel_rows"]],
        ["panel games / books", f"{R['panel_games']} / {raw['books']}"],
        [
            "game_lines rows dropped",
            f"{gl['rows']} on {gl['game_ids']} game_ids; "
            f"{gl['rows_recorded_over_30_days_after_their_game']} rows on "
            f"{gl['game_ids_recorded_over_30_days_after_their_game']} 2024 game_ids recorded "
            f"{hrange[0]:,.0f}-{hrange[1]:,.0f} h after kickoff",
        ],
        [
            "same matchup within 10 days of those rows",
            ", ".join(gl["same_matchup_within_10_days_of_snapshot"]),
        ],
        ["pre-repair kickoffs vs 2025-12-08 dump", R.get("dump_check", "not checked")],
    ]
    sign = [
        ["games", S["games"]],
        ["MAE", S["mae"]],
        [
            "exact / within 0.5 / within 1",
            " / ".join(pct(S[k]) for k in ("share_exact", "share_within_0_5", "share_within_1")),
        ],
        ["correlation", S["corr"]],
        [
            "MAE / corr without the sign flip",
            f"{S['mae_without_sign_flip']} / {S['corr_without_sign_flip']}",
        ],
        ["MAE by season", S["mae_by_season"]],
        ["games differing by >= 3", S["games_differing_by_3_or_more"]],
    ]
    buckets = [
        [r["hours_before_kickoff"], r["snapshots"], r["games"], r["median_books"]]
        + [f"{r['mean_abs_diff']:.3f}", r["median_abs_diff"]]
        + [pct(r[f"share_ge_{k}"]) for k in ("0_5", "1", "2", "3")]
        for r in A["b_movement"]["by_hours_bucket"]
    ]
    C = A["c_clv"]
    clv = [clv_row(f"main: earliest >= {MIN_BOOKS} books, move >= 1", C["main"])]
    clv += [clv_row(r["definition"], r) for r in C["sensitivity"]]
    clv += [clv_row("survey definition, move >= 1", C["survey_definition"])]
    D = A["d_dispersion"]
    snaps = [k for k in D if k != "margin_model"]
    by_class = A["coverage"]["by_slot_and_class"]
    classes = sorted({k for row in by_class.values() for k in row})
    flags = [
        [k, v["rows"], v["games"], v["top_book_seasons"]]
        for k, v in F.items()
        if isinstance(v, dict) and "rows" in v
    ]
    L = [
        "# odds_history line-movement results (generated)",
        "",
        f"Generated by `scripts/odds_history_panel.py` on {R['generated_at']}. Do not edit by",
        "hand; README.md in this directory explains the results. Aggregates only: the row-level",
        "panel is private (The Odds API terms).",
        "",
        "## Inputs",
        "",
        table(["item", "value"], inputs),
        "",
        "## Timing reconstruction",
        "",
        timing_table(T),
        "",
        f"- Calls whose newest update is > {STALE_CALL_MINUTES} min before the request: "
        f"{stale_text(T)}",
        f"- Rows with last_update after their request time: {T['rows_updated_after_request']}",
        f"- Request - last_update per row (minutes): {T['row_request_minus_last_update_minutes']}",
        "",
        "Per (game, label) check (the survey used the newest last_update as the snapshot time):",
        "",
        table(["measure", "value"], [[k, v] for k, v in R["label_timing"].items()]),
        "",
        "## Flags",
        "",
        table(hdr("flag; rows; games; largest book-seasons"), flags),
        "",
        f"- In-play rows by label: {F['in-play rows by label']}; in-play games by ET kickoff: "
        f"{F['in-play games by ET kickoff time']}",
        "- |line - same-call median|, alternate-price rows: "
        f"{F['alt-price rows: |line - call median| quantiles']}",
        "- |line - same-call median|, main-price rows: "
        f"{F['main-price rows: |line - call median| quantiles']}",
        f"- betmgm 2020 spread rows: {F['betmgm 2020 spread rows']}",
        "",
        "## Prices",
        "",
        "The backfill stored int(American) of the API's two-place decimal odds (truncation toward",
        "zero). Stored values by how many two-place decimals produce them:",
        "",
        table(hdr("market; stored prices; from one decimal; from two; from none"), price_rows(R)),
        "",
        f"- Spread prices stored as -109 / -110: {P['spread prices stored as -109 / -110']}",
        f"- Decimals behind stored values: {P['decimals behind stored values']}",
        "",
        "## Analysis universe and sign check",
        "",
        f"- Games with odds: {U['games_with_odds']}; with a game-day consensus (>= {MIN_BOOKS}"
        f" books, <= {GAME_DAY_HOURS} h before kickoff): {U['games_with_game_day_consensus']};"
        f" excluded: {', '.join(U['excluded_no_game_day_consensus'])}",
        f"- Hours before kickoff (p10/p50/p90): {U['hours']}; median books: {U['books']}",
        f"- Earliest >= {MIN_BOOKS}-book snapshot is itself on game day: "
        f"{U['early5_is_game_day_snapshot']} games",
        "",
        table(["game-day consensus (-home_spread) vs spread_close", "value"], sign),
        "",
        "## (a) Line accuracy vs the final margin",
        "",
        f"{A['a_accuracy']['games']} games, the same games for every line. CI: week-block"
        f" bootstrap ({R['n_boot']} draws).",
        "",
        table(
            hdr("line; median h before kickoff; MAE; MAE - close; 95% CI"),
            [accuracy_row(r) for r in A["a_accuracy"]["rows"]],
        ),
        "",
        "## (b) How far the consensus sits from the close, by hours before kickoff",
        "",
        f"Snapshots with >= {MIN_BOOKS} books, games in the analysis universe.",
        "",
        table(
            hdr(
                "hours before kickoff; snapshots; games; median books; mean |line - close|;"
                " median; >= 0.5; >= 1; >= 2; >= 3"
            ),
            buckets,
        ),
        "",
        f"Per game, earliest >= {MIN_BOOKS}-book consensus vs close:",
        "",
        table(["measure", "value"], [[k, v] for k, v in A["b_movement"]["per_game"].items()]),
        "",
        "## (c) What closing-line value is worth",
        "",
        "Side the consensus moved toward between the early snapshot and the close, graded at the",
        "early line and at the close. Pushes excluded; ROI at -110; CIs: week-block bootstrap.",
        "",
        table(
            hdr(
                "definition; games; weeks; median h of early line; mean |move|;"
                " W-L-P at early line; win rate (95% CI); ROI; W-L-P at close; win rate (95% CI);"
                " ROI; difference, pp (95% CI)"
            ),
            clv,
        ),
        "",
        f"Survey-definition MAE vs margin: {C['survey_definition']['mae_vs_margin']}",
        "",
        "## (d) Book dispersion: best available line vs the consensus at the same snapshot",
        "",
        table(["measure"] + snaps, [[k] + [D[n][k] for n in snaps] for k in D[snaps[0]]]),
        "",
        f"Implied win probabilities: {D['margin_model']}",
        "",
        "## Coverage, 2020-2024",
        "",
        table(
            hdr("group; games; with any odds; with game-day consensus"),
            count_rows(A["coverage"]["by_cause"], "group"),
        ),
        "",
        table(
            hdr("kickoff slot (ET); games; with any odds; with game-day consensus"),
            count_rows(A["coverage"]["by_slot"], "slot"),
        ),
        "",
        table(
            hdr("season; games; with any odds; with game-day consensus"),
            count_rows(A["coverage"]["by_season"], "season"),
        ),
        "",
        table(
            ["kickoff slot (ET)"] + classes,
            [[slot] + [row.get(k, 0) for k in classes] for slot, row in by_class.items()],
        ),
        "",
        "Night games with odds, latest pre-kickoff snapshot (h): "
        f"{A['coverage']['night_games_with_odds_latest_snapshot_hours']}",
        "",
        "ET kickoff times of the non-Rams games whose UTC date is the next day: "
        f"{A['coverage']['next_utc_date_kickoffs_et_non_rams']}",
        "",
        "## (e) 2025 and 2026",
        "",
        f"odds_history rows by game season and label: {raw['rows_by_game_season_and_label']}.",
        f"games rows: 2025 {raw['games_2025_in_db']}, 2026 {raw['games_2026_in_db']}.",
        "",
    ]
    return "\n".join(L)


COLUMN_DOCS = [
    ("game_id, season, week, game_type, home_team, away_team", "from games (Rams = LA)"),
    ("kickoff_utc", "true kickoff instant (games.kickoff, repaired from nflverse 2026-10-24)"),
    ("bookmaker", "The Odds API bookmaker key"),
    ("source_label", "odds_history.market_type as stored: the API-call name only"),
    ("pull", f"{PULL_NAMES['opening']} ('opening') or {PULL_NAMES['closing']} ('closing')"),
    ("call_id", "API call, 1..N in request order; every row of a call shares request_at"),
    (
        "request_at",
        "reconstructed request time (UTC); the API returns its snapshot at or before it",
    ),
    ("hours_to_kickoff", "(kickoff_utc - request_at) in hours; negative = after kickoff"),
    ("last_update", "the book's own last update (UTC), as returned by the API"),
    ("update_age_minutes", "request_at - last_update"),
    ("captured_at", "insert time (Dec 2025); used only to split rows into API calls"),
    ("call_last_update_max, call_lag_minutes", "newest last_update in the call; request_at - it"),
    ("home_spread, away_spread", "raw, SPORTSBOOK sign (home -7.5 = home favored by 7.5)"),
    (
        "home_spread_price, away_spread_price",
        "American odds as stored: int() of the API's decimal odds, so -110 reads -109 (Prices)",
    ),
    ("home_spread_dec, away_spread_dec", "decimal odds recovered from the stored prices"),
    ("line_home", "-home_spread: nflverse sign, expected home margin, positive = home favored"),
    ("over_under, over_price, under_price", "total and its stored American prices"),
    ("over_dec, under_dec", "recovered decimal odds"),
    ("home_moneyline, away_moneyline", "stored American moneylines"),
    ("home_ml_dec, away_ml_dec", "recovered decimal odds"),
    ("spread_close, total_close", "nflverse closing lines from games (nflverse sign)"),
    ("home_score, away_score, home_margin", "final score; home_margin = home - away"),
    ("flag_inplay", "request_at or last_update after kickoff_utc"),
    ("flag_spread_price", "a spread price of 0, or a side above +125 or below -200 (alt line)"),
    (
        "flag_spread_outlier",
        f"normal price but >= {OUTLIER_POINTS:g} pts from the call's median line (>= 3 books)",
    ),
    ("flag_total_price", "a total price of 0, or a side above +125 or below -200"),
    ("flag_moneyline_price", "a moneyline of 0"),
    ("flag_stale_call", f"newest update in the call > {STALE_CALL_MINUTES} min before request_at"),
    ("clean_spread", "spread present; not flag_inplay, flag_spread_price or flag_spread_outlier"),
    ("clean_total", "total present; not flag_inplay or flag_total_price"),
]


def private_readme(R: dict) -> str:
    T, F, A, raw = R["timing"], R["flags"], R["analysis"], R["raw"]
    S, gl = A["sign_check"], raw["game_lines"]
    files = [
        [PANEL_FILE, R["panel_rows"], "one row per book quote per API call (opening + closing)"],
        [SNAPSHOT_FILE, R["snapshot_rows"], "per (game, call): median line, range, best quotes"],
        [COVERAGE_FILE, R["coverage_rows"], "every 2020-2024 game: coverage and cause (no prices)"],
    ]
    flags = [
        [k, v["rows"], v["games"], v["top_book_seasons"]]
        for k, v in F.items()
        if isinstance(v, dict) and "rows" in v
    ]
    L = [
        "# odds_history panel, 2020-2024 (PRIVATE)",
        "",
        "**Do not commit, publish or share these files.** They are The Odds API historical",
        'snapshots bought for this project. The terms say "Do not resell, repackage, or',
        'redistribute our data as a standalone data product"',
        "(https://the-odds-api.com/terms-and-conditions.html, read 2026-10-24 during the salvage",
        "survey; storing the data indefinitely is permitted). This directory is gitignored",
        "(`backups/*`). Publish only aggregates, as in",
        "`docs/investigations/2026-10-line-movement/`.",
        "",
        f"Built by `scripts/odds_history_panel.py` on {R['generated_at']} from devdb01",
        f"`public.odds_history` ({raw['rows']} rows) and `games` (kickoffs repaired 2026-10-24),",
        "read in a read-only session.",
        "",
        "## Files",
        "",
        table(["file", "rows", "contents"], files),
        "",
        "```python",
        "import pandas as pd",
        f'p = pd.read_parquet("backups/local/odds_history_panel/{PANEL_FILE}")',
        'q = p[p["clean_spread"]]  # line_home is the nflverse-sign line',
        "```",
        "",
        "## Sign convention",
        "",
        "`home_spread` is the sportsbook sign (negative = home favored).",
        "`line_home = -home_spread` is the nflverse sign used by `games.spread_close`: the",
        "expected home margin, positive = home favored. Home covers line L iff",
        "`home_margin - L > 0`. Check on game-day consensus",
        f"lines: MAE vs spread_close {S['mae']} (corr {S['corr']}); without the flip MAE",
        f"{S['mae_without_sign_flip']} (corr {S['corr_without_sign_flip']}).",
        "",
        "## Prices",
        "",
        "The API returned decimal odds with two places, and the backfill stored",
        "`int(american_odds(decimal))` (backfill_historical_odds.py:162, 247-253), which",
        "truncates toward zero. So a -110 quote (1.91) is stored as -109, -115 (1.87) as -114,",
        "and +105 (2.05, 104.99999 in floating point) as +104. Use the `*_dec` columns for",
        "breakevens and vig: each is the mean of the two-place decimals that produce the stored",
        "value, which is exact for negative prices and within 0.005 for positive ones (two",
        "candidates, e.g. +104 from 2.04 or 2.05). A stored value that no two-place decimal",
        "produces (a quote with more precision) is converted exactly. A stored 0 means no usable",
        "price (`american_odds` returns 0 for a missing decimal); its `*_dec` is NaN.",
        "",
        table(hdr("market; stored prices; from one decimal; from two; from none"), price_rows(R)),
        "",
        "## Timing reconstruction",
        "",
        "`scripts/backfill_historical_odds.py` made two requests per game date (lines 359-378),",
        "reading `games.kickoff` when it still held Eastern wall-clock time labelled UTC:",
        "",
        f"- 'closing' (`pull = {PULL_NAMES['closing']}`): MIN(kickoff on that date) - 30 min,",
        "  i.e. the ET wall-clock time read as UTC: 4.5 h (EDT) or 5.5 h (EST) before the true",
        "  first kickoff of the day.",
        f"- 'opening' (`pull = {PULL_NAMES['opening']}`): 14:00 UTC on the ET game date.",
        "",
        "Neither is an opening or a closing line. A game's earliest snapshot comes from the calls",
        "on earlier game dates (often a week or more ahead, with few books), so there is no true",
        "opener either. Use `spread_close` (nflverse) as the close.",
        "",
        "Reconstruction: rows split into API calls where `captured_at` jumps by more than",
        f"{CALL_GAP_SECONDS} s; each call gets the first scheduled request at or after its newest",
        "`last_update` (the API returns the latest snapshot at or before the requested time).",
        "",
        timing_table(T),
        "",
        f"Every call maps to a different request; {T['rows_updated_after_request']} rows have",
        "`last_update` after their request time. The newest book update in a call is a few",
        "minutes before the request: medians near 10 minutes in 2020-2021 and near 5 from 2023",
        "(by-season row above), as expected from the API's 10-minute snapshots until September",
        "2022 and 5-minute snapshots after. So the data show the market a few minutes before",
        "`request_at`.",
        f"`flag_stale_call` (newest update > {STALE_CALL_MINUTES} min before the request):",
        f"{stale_text(T)}. Pre-repair kickoffs vs the 2025-12-08 dump (made right after the",
        f"backfill): {R.get('dump_check', 'not checked')}.",
        "",
        "## Flags",
        "",
        table(hdr("flag; rows; games; largest book-seasons"), flags),
        "",
        f"In-play rows by label: {F['in-play rows by label']}; in-play games by ET kickoff:",
        f"{F['in-play games by ET kickoff time']} (the 14:00 UTC call comes after a 09:30 EDT",
        f"kickoff). The {gl['rows']} `market_type = 'game_lines'` rows (inserted",
        f"{raw['captured_at_range']['game_lines'][0][:16]} by a script that is not in the",
        "repo) are not in the panel:",
        f"{gl['rows_recorded_over_30_days_after_their_game']} of them sit on 2024 game_ids but",
        "were recorded more than 30 days after those games, within 10 days of the",
        f"same matchups in 2025 ({', '.join(gl['same_matchup_within_10_days_of_snapshot'])}).",
        "",
        "## Coverage holes",
        "",
        table(
            hdr("group; games; with any odds; with game-day consensus"),
            count_rows(A["coverage"]["by_cause"], "group"),
        ),
        "",
        "Both holes come from `find_game_id` (backfill_historical_odds.py:205-244), which could",
        "never accept these games: it mapped 'Los Angeles Rams' to 'LAR' (games uses 'LA'), and it",
        "compared the API's UTC commence date with DATE(kickoff), the ET date, so games starting",
        "at or after 20:00 EDT / 19:00 EST never matched. Whatever the API returned for them was",
        "dropped (no raw responses were stored). The night games that do have odds matched only",
        "in early snapshots, while the API still listed a commence time on the ET date; none has",
        f"a game-day snapshot. Per-game detail: `{COVERAGE_FILE}`. 2025 and 2026 have no",
        "snapshots at all.",
        "",
        "## Columns",
        "",
        table(["column", "meaning"], [[f"`{c}`", m] for c, m in COLUMN_DOCS]),
        "",
    ]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--old-kickoffs", type=Path, default=OLD_KICKOFFS)
    ap.add_argument("--private-dir", type=Path, default=PRIVATE_DIR)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--dump-games-copy", type=Path, help="pg_restore -a -t games output")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    odds, games, old = load_inputs(args.old_kickoffs)
    assert_nflverse_convention(games)
    R: dict = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_boot": args.n_boot,
        "seed": args.seed,
        "raw": raw_inventory(odds, games),
    }
    if args.dump_games_copy:
        c = dump_kickoff_check(args.dump_games_copy, old)
        R["dump_check"] = f"{c['identical']} of {c['games']} 2020-2024 kickoffs identical"

    sched = request_schedule(old)
    timed, R["timing"] = match_calls(odds[odds["market_type"].isin(PULL_NAMES)], sched, old)
    panel = build_panel(timed, games)
    R["label_timing"] = label_timing_check(panel)
    R["flags"] = flag_summary(panel)
    R["prices"] = price_encoding_summary(panel)
    snap = build_snapshots(panel)
    lines = per_game_lines(snap, games)
    cov = coverage_table(games, panel, lines)
    R["analysis"] = analyse(panel, snap, lines, cov, CoverModel(games), args.n_boot, args.seed)
    R["panel_rows"], R["panel_games"] = len(panel), int(panel["game_id"].nunique())
    R["snapshot_rows"], R["coverage_rows"] = len(snap), len(cov)

    args.private_dir.mkdir(parents=True, exist_ok=True)
    panel = panel[PANEL_COLUMNS].sort_values(["call_id", "game_id", "bookmaker"])
    panel.to_parquet(args.private_dir / PANEL_FILE, index=False)
    snap.to_parquet(args.private_dir / SNAPSHOT_FILE, index=False)
    cov.to_csv(args.private_dir / COVERAGE_FILE, index=False)
    (args.private_dir / "README.md").write_text(private_readme(R))

    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "results.json").write_text(json.dumps(jsonable(R), indent=2) + "\n")
    (args.results_dir / "results.md").write_text(results_markdown(R))

    main_c = R["analysis"]["c_clv"]["main"]
    print(f"panel: {R['panel_rows']} rows, {R['panel_games']} games -> {args.private_dir}")
    print(f"results -> {args.results_dir}")
    early, close = main_c["at_early_line"]["win_rate"], main_c["at_close"]["win_rate"]
    print(
        f"CLV main: {main_c['games']} games, {pct(early)} at the early line, {pct(close)} at close"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
