"""
As-of snapshot builder for pre-decision features (one row per team-game).

Joins across games, odds_history (<= cutoff), and weather to produce a
clean, reproducible snapshot that can be used for modeling or RL datasets.

Usage:
  python py/features/asof_snapshot.py \
    --cutoff 2024-12-31T23:59:59Z \
    --output data/asof_snapshot.csv

Environment (defaults shown):
  POSTGRES_HOST=localhost POSTGRES_PORT=5544 POSTGRES_DB=devdb01
  POSTGRES_USER=dro POSTGRES_PASSWORD=sicillionbillions
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import pandas as pd
import psycopg


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build as-of snapshot (team-game rows)")
    ap.add_argument("--cutoff", required=True, help="ISO timestamp (UTC) for as-of boundary")
    ap.add_argument("--output", required=True, help="Output CSV path")
    return ap.parse_args()


def get_connection() -> psycopg.Connection:
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5544")
    dbname = os.environ.get("POSTGRES_DB", "devdb01")
    user = os.environ.get("POSTGRES_USER", "dro")
    password = os.environ.get("POSTGRES_PASSWORD", "sicillionbillions")
    return psycopg.connect(host=host, port=port, dbname=dbname, user=user, password=password)


def build_snapshot(conn: psycopg.Connection, cutoff_iso: str) -> pd.DataFrame:
    sql = f"""
    with odds_ranked as (
      select
        home_team,
        away_team,
        date(commence_time) as gdate,
        market_key,
        outcome_name,
        outcome_point,
        outcome_price,
        snapshot_at,
        row_number() over (
          partition by home_team, away_team, date(commence_time), market_key, outcome_name
          order by snapshot_at desc
        ) as rk
      from odds_history
      where snapshot_at <= %(cutoff)s
    ), odds_latest as (
      select * from odds_ranked where rk = 1
    ), odds_pivot as (
      select
        home_team,
        away_team,
        gdate,
        max(case when market_key='spreads' then outcome_point end) as spread_point,
        max(case when market_key='spreads' then outcome_price end) as spread_price,
        max(case when market_key='totals' and lower(outcome_name)='over' then outcome_point end) as total_point,
        max(case when market_key='totals' and lower(outcome_name)='over' then outcome_price end) as over_price,
        max(case when market_key='totals' and lower(outcome_name)='under' then outcome_price end) as under_price
      from odds_latest
      group by 1,2,3
    )
    select
      g.game_id,
      g.season,
      g.week,
      g.home_team,
      g.away_team,
      w.temp_c, w.wind_kph, w.precip_mm,
      o.spread_point, o.spread_price, o.total_point, o.over_price, o.under_price,
      %(cutoff)s::timestamptz as as_of
    from games g
    left join weather w on w.game_id = g.game_id
    left join odds_pivot o
      on o.home_team = g.home_team and o.away_team = g.away_team
      and (g.kickoff is null or o.gdate = date(g.kickoff))
    where (g.kickoff is null or g.kickoff <= %(cutoff)s)
    order by g.season, g.week, g.game_id
    """
    df_games = pd.read_sql(sql, conn, params={"cutoff": cutoff_iso})

    # Expand to per-team rows with simple spread sign flip for away side
    rows = []
    for _, r in df_games.iterrows():
        # Home perspective
        rows.append(
            {
                "as_of": r["as_of"],
                "game_id": r["game_id"],
                "season": r["season"],
                "week": r["week"],
                "team": r["home_team"],
                "opponent": r["away_team"],
                "is_home": True,
                "spread_point": r["spread_point"],
                "spread_price": r["spread_price"],
                "total_point": r["total_point"],
                "over_price": r["over_price"],
                "under_price": r["under_price"],
                "temp_c": r["temp_c"],
                "wind_kph": r["wind_kph"],
                "precip_mm": r["precip_mm"],
            }
        )
        # Away perspective
        rows.append(
            {
                "as_of": r["as_of"],
                "game_id": r["game_id"],
                "season": r["season"],
                "week": r["week"],
                "team": r["away_team"],
                "opponent": r["home_team"],
                "is_home": False,
                "spread_point": -r["spread_point"] if pd.notna(r["spread_point"]) else None,
                "spread_price": r["spread_price"],
                "total_point": r["total_point"],
                "over_price": r["over_price"],
                "under_price": r["under_price"],
                "temp_c": r["temp_c"],
                "wind_kph": r["wind_kph"],
                "precip_mm": r["precip_mm"],
            }
        )
    snap = pd.DataFrame.from_records(rows)
    # Basic FK sanity: drop rows with missing game_id/team
    snap = snap.dropna(subset=["game_id", "team"])  # noqa: PD002
    return snap


def main() -> None:
    args = parse_args()
    with get_connection() as conn:
        snap = build_snapshot(conn, args.cutoff)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    snap.to_csv(args.output, index=False)
    print(f"[asof] wrote {len(snap)} rows -> {args.output}")


if __name__ == "__main__":
    main()

