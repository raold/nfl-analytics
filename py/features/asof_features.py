"""Build as-of team/game features for modeling and backtesting.

This script produces leakage-safe, pre-game features for every NFL matchup by
rolling forward through time and only aggregating information available prior
to kickoff. It emits a wide, game-level dataset with home and away features as
well as helpful differentials for modeling and backtests.

Example:
  python py/features/asof_features.py --output analysis/features/asof_team_features.csv \
      --write-table mart.asof_team_features --season-start 2003 --validate
"""

from __future__ import annotations

import argparse
import os
from io import StringIO
from typing import Iterable

import numpy as np
import pandas as pd
import psycopg


SQL_TEAM_BASE = """
WITH team_games AS (
    SELECT g.game_id,
           g.season,
           g.week,
           g.kickoff,
           g.home_team AS team,
           TRUE AS is_home,
           g.home_team,
           g.away_team,
           g.home_score,
           g.away_score,
           g.spread_close,
           g.total_close,
           g.home_score AS points_for,
           g.away_score AS points_against,
           g.stadium,
           g.roof,
           g.surface,
           g.home_qb_id AS qb_id,
           g.home_qb_name AS qb_name,
           g.home_coach AS coach_name
    FROM games g
    UNION ALL
    SELECT g.game_id,
           g.season,
           g.week,
           g.kickoff,
           g.away_team AS team,
           FALSE AS is_home,
           g.home_team,
           g.away_team,
           g.home_score,
           g.away_score,
           g.spread_close,
           g.total_close,
           g.away_score AS points_for,
           g.home_score AS points_against,
           g.stadium,
           g.roof,
           g.surface,
           g.away_qb_id AS qb_id,
           g.away_qb_name AS qb_name,
           g.away_coach AS coach_name
    FROM games g
),
team_stats AS (
    SELECT game_id,
           posteam AS team,
           COALESCE(SUM(epa), 0) AS epa_sum,
           COUNT(*) AS plays
    FROM plays
    WHERE posteam IS NOT NULL
    GROUP BY 1, 2
)
SELECT tg.game_id,
       tg.team,
       tg.is_home,
       tg.season,
       tg.week,
       tg.kickoff,
       tg.home_team,
       tg.away_team,
       tg.home_score,
       tg.away_score,
       tg.spread_close,
       tg.total_close,
       tg.points_for,
       tg.points_against,
       tg.stadium,
       tg.roof,
       tg.surface,
       tg.qb_id,
       tg.qb_name,
       tg.coach_name,
       tg.points_for - tg.points_against AS margin,
       COALESCE(ts.epa_sum, 0) AS epa_sum,
       COALESCE(ts.plays, 0) AS plays
FROM team_games tg
LEFT JOIN team_stats ts
  ON ts.game_id = tg.game_id AND ts.team = tg.team
ORDER BY tg.team, tg.season, tg.week, tg.kickoff NULLS FIRST, tg.game_id;
"""


TEAM_FEATURE_COLUMNS = [
    "prior_games",
    "prior_epa_sum",
    "prior_plays",
    "prior_epa_mean",
    "epa_pp_last3",
    "epa_pp_last5",
    "prior_margin_avg",
    "margin_last3",
    "margin_last5",
    "prior_points_for_avg",
    "prior_points_against_avg",
    "points_for_last3",
    "points_against_last3",
    "prior_win_pct",
    "win_pct_last5",
    "season_win_pct",
    "season_point_diff_avg",
    "rest_days",
    "rest_lt_6",
    "rest_gt_13",
    "travel_change",
    "prev_result",
    "prev_margin",
    "prev_points_for",
    "prev_points_against",
    "prev_spread_close",
    "qb_change",
    "qb_team_games",
    "qb_total_games",
    "coach_change",
    "coach_team_games",
    "coach_total_games",
    "surface_grass",
    "surface_artificial",
    "roof_dome",
    "roof_outdoors",
]


DIFF_FEATURE_COLUMNS = [
    "prior_epa_mean",
    "prior_games",
    "epa_pp_last3",
    "epa_pp_last5",
    "prior_margin_avg",
    "margin_last3",
    "margin_last5",
    "prior_points_for_avg",
    "prior_points_against_avg",
    "points_for_last3",
    "points_against_last3",
    "prior_win_pct",
    "win_pct_last5",
    "season_win_pct",
    "season_point_diff_avg",
    "rest_days",
    "qb_change",
    "qb_team_games",
    "qb_total_games",
    "coach_change",
    "coach_team_games",
    "coach_total_games",
    "surface_grass",
    "surface_artificial",
    "roof_dome",
    "roof_outdoors",
]


def get_connection() -> psycopg.Connection:
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = int(os.environ.get("POSTGRES_PORT", "5544"))
    dbname = os.environ.get("POSTGRES_DB", "devdb01")
    user = os.environ.get("POSTGRES_USER", "dro")
    password = os.environ.get("POSTGRES_PASSWORD", "sicillionbillions")
    return psycopg.connect(host=host, port=port, dbname=dbname, user=user, password=password)


def fetch_team_game_features(conn: psycopg.Connection) -> pd.DataFrame:
    return pd.read_sql(SQL_TEAM_BASE, conn)


def compute_team_history(df: pd.DataFrame) -> pd.DataFrame:
    team_df = df.copy()
    team_df["kickoff"] = pd.to_datetime(team_df["kickoff"], errors="coerce")
    team_df.sort_values([
        "team",
        "season",
        "week",
        "kickoff",
        "game_id",
    ], inplace=True)

    team_df["epa_sum"] = team_df["epa_sum"].astype(float)
    team_df["plays"] = team_df["plays"].astype(float)
    team_df["points_for"] = team_df["points_for"].astype(float)
    team_df["points_against"] = team_df["points_against"].astype(float)
    team_df["margin"] = team_df["margin"].astype(float)

    team_df["qb_id"] = team_df["qb_id"].fillna("UNKNOWN")
    team_df["coach_name"] = team_df["coach_name"].fillna("UNKNOWN")
    team_df["surface"] = team_df["surface"].fillna("").str.lower()
    team_df["roof"] = team_df["roof"].fillna("").str.lower()

    team_df["epa_per_play"] = np.where(team_df["plays"] > 0, team_df["epa_sum"] / team_df["plays"], np.nan)
    team_df["points_total"] = team_df["points_for"] + team_df["points_against"]

    result_values = np.where(team_df["margin"] > 0, 1.0, np.where(team_df["margin"] < 0, 0.0, np.nan))
    result_values = np.where(np.isnan(result_values) & team_df["margin"].notna(), 0.5, result_values)
    team_df["result_value"] = result_values

    team_df["_points_for_f"] = team_df["points_for"].fillna(0.0)
    team_df["_points_against_f"] = team_df["points_against"].fillna(0.0)
    team_df["_margin_f"] = team_df["margin"].fillna(0.0)
    team_df["_result_f"] = team_df["result_value"].fillna(0.0)

    team_group = team_df.groupby("team", group_keys=False)
    team_df["prior_games"] = team_group.cumcount()
    team_df["prior_epa_sum"] = team_group["epa_sum"].cumsum() - team_df["epa_sum"]
    team_df["prior_plays"] = team_group["plays"].cumsum() - team_df["plays"]
    team_df["prior_points_for"] = team_group["_points_for_f"].cumsum() - team_df["_points_for_f"]
    team_df["prior_points_against"] = team_group["_points_against_f"].cumsum() - team_df["_points_against_f"]
    team_df["prior_margin_sum"] = team_group["_margin_f"].cumsum() - team_df["_margin_f"]
    team_df["prior_wins"] = team_group["_result_f"].cumsum() - team_df["_result_f"]

    with np.errstate(divide="ignore", invalid="ignore"):
        team_df["prior_epa_mean"] = np.where(team_df["prior_plays"] > 0, team_df["prior_epa_sum"] / team_df["prior_plays"], np.nan)
        team_df["prior_margin_avg"] = np.where(team_df["prior_games"] > 0, team_df["prior_margin_sum"] / team_df["prior_games"], np.nan)
        team_df["prior_points_for_avg"] = np.where(team_df["prior_games"] > 0, team_df["prior_points_for"] / team_df["prior_games"], np.nan)
        team_df["prior_points_against_avg"] = np.where(team_df["prior_games"] > 0, team_df["prior_points_against"] / team_df["prior_games"], np.nan)
        team_df["prior_win_pct"] = np.where(team_df["prior_games"] > 0, team_df["prior_wins"] / team_df["prior_games"], np.nan)

    season_group = team_df.groupby(["team", "season"], group_keys=False)
    team_df["season_games_prior"] = season_group.cumcount()
    team_df["season_margin_prior_sum"] = season_group["_margin_f"].cumsum() - team_df["_margin_f"]
    team_df["season_wins_prior"] = season_group["_result_f"].cumsum() - team_df["_result_f"]
    with np.errstate(divide="ignore", invalid="ignore"):
        team_df["season_point_diff_avg"] = np.where(
            team_df["season_games_prior"] > 0,
            team_df["season_margin_prior_sum"] / team_df["season_games_prior"],
            np.nan,
        )
        team_df["season_win_pct"] = np.where(
            team_df["season_games_prior"] > 0,
            team_df["season_wins_prior"] / team_df["season_games_prior"],
            np.nan,
        )

    team_df["epa_pp_last3"] = team_group["epa_per_play"].transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
    team_df["epa_pp_last5"] = team_group["epa_per_play"].transform(lambda s: s.shift(1).rolling(window=5, min_periods=1).mean())
    team_df["margin_last3"] = team_group["_margin_f"].transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
    team_df["margin_last5"] = team_group["_margin_f"].transform(lambda s: s.shift(1).rolling(window=5, min_periods=1).mean())
    team_df["points_for_last3"] = team_group["points_for"].transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
    team_df["points_against_last3"] = team_group["points_against"].transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
    team_df["win_pct_last5"] = team_group["_result_f"].transform(lambda s: s.shift(1).rolling(window=5, min_periods=1).mean())

    team_df["prev_kickoff"] = team_group["kickoff"].shift(1)
    team_df["rest_days"] = (team_df["kickoff"] - team_df["prev_kickoff"]).dt.total_seconds() / (24 * 3600)
    team_df["rest_days"] = team_df["rest_days"].fillna(7.0)
    team_df["rest_lt_6"] = (team_df["rest_days"] < 6).astype(float)
    team_df["rest_gt_13"] = (team_df["rest_days"] > 13).astype(float)

    team_df["prev_result"] = team_group["_result_f"].shift(1)
    team_df["prev_margin"] = team_group["_margin_f"].shift(1)
    team_df["prev_points_for"] = team_group["_points_for_f"].shift(1)
    team_df["prev_points_against"] = team_group["_points_against_f"].shift(1)
    team_df["prev_spread_close"] = team_group["spread_close"].shift(1)

    prev_is_home = team_group["is_home"].shift(1)
    team_df["travel_change"] = np.where(team_df["prior_games"] > 0, (prev_is_home != team_df["is_home"]).astype(float), 0.0)

    prev_qb = team_group["qb_id"].shift(1)
    team_df["qb_change"] = np.where(team_df["prior_games"] > 0, (prev_qb != team_df["qb_id"]).astype(float), 0.0)
    team_df["qb_team_games"] = team_df.groupby(["team", "qb_id"]).cumcount().astype(float)
    team_df["qb_total_games"] = team_df.groupby("qb_id").cumcount().astype(float)

    prev_coach = team_group["coach_name"].shift(1)
    team_df["coach_change"] = np.where(team_df["prior_games"] > 0, (prev_coach != team_df["coach_name"]).astype(float), 0.0)
    team_df["coach_team_games"] = team_df.groupby(["team", "coach_name"]).cumcount().astype(float)
    team_df["coach_total_games"] = team_df.groupby("coach_name").cumcount().astype(float)

    surface_lower = team_df["surface"]
    team_df["surface_grass"] = surface_lower.str.contains("grass").astype(float)
    team_df["surface_artificial"] = surface_lower.str.contains("turf").astype(float)

    roof_lower = team_df["roof"]
    team_df["roof_dome"] = roof_lower.isin(["dome", "closed"]).astype(float)
    team_df["roof_outdoors"] = roof_lower.isin(["outdoors", "open", "retractable"]).astype(float)

    helper_cols = [
        "_points_for_f",
        "_points_against_f",
        "_margin_f",
        "_result_f",
    ]
    team_df.drop(columns=helper_cols, inplace=True)
    return team_df


def validate_team_history(df: pd.DataFrame) -> None:
    expected_prior = df.groupby("team").cumcount()
    if not np.array_equal(df["prior_games"].to_numpy(), expected_prior.to_numpy()):
        raise ValueError("prior_games mismatch detected (possible leakage).")

    mask = df["prior_plays"] > 0
    if mask.any():
        ratio = df.loc[mask, "prior_epa_sum"] / df.loc[mask, "prior_plays"]
        if not np.allclose(df.loc[mask, "prior_epa_mean"], ratio, rtol=1e-6, atol=1e-9):
            raise ValueError("prior_epa_mean inconsistent with sums/plays.")

    if (df["rest_days"] < 0).any():
        raise ValueError("Negative rest_days encountered.")


def rename_with_prefix(columns: Iterable[str], prefix: str) -> dict[str, str]:
    return {col: f"{prefix}_{col}" for col in columns}


def pivot_to_games(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["kickoff"] = pd.to_datetime(df["kickoff"], errors="coerce")

    common_cols = [
        "game_id",
        "season",
        "week",
        "kickoff",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "spread_close",
        "total_close",
    ]

    home = df[df["is_home"]].copy()
    away = df[~df["is_home"]].copy()

    home = home[common_cols + TEAM_FEATURE_COLUMNS].rename(columns=rename_with_prefix(TEAM_FEATURE_COLUMNS, "home"))
    away = away[common_cols + TEAM_FEATURE_COLUMNS].rename(columns=rename_with_prefix(TEAM_FEATURE_COLUMNS, "away"))

    merged = pd.merge(home, away, on=common_cols, how="inner", validate="one_to_one")

    for col in [
        "home_prior_epa_sum",
        "home_prior_plays",
        "away_prior_epa_sum",
        "away_prior_plays",
    ]:
        merged[col] = merged[col].fillna(0.0)

    for col in ["home_rest_days", "away_rest_days"]:
        merged[col] = merged[col].fillna(7.0)

    for feature in DIFF_FEATURE_COLUMNS:
        home_col = f"home_{feature}"
        away_col = f"away_{feature}"
        diff_col = f"{feature}_diff"
        merged[diff_col] = merged[home_col] - merged[away_col]

    merged["epa_diff_prior"] = merged["home_prior_epa_mean"] - merged["away_prior_epa_mean"]
    merged["plays_diff_prior"] = merged["home_prior_plays"] - merged["away_prior_plays"]
    merged["rest_diff"] = merged["home_rest_days"] - merged["away_rest_days"]

    merged["home_margin"] = merged["home_score"] - merged["away_score"]
    merged["home_cover"] = np.where(
        merged["home_margin"] - merged["spread_close"] > 0,
        1,
        np.where(merged["home_margin"] - merged["spread_close"] < 0, 0, np.nan),
    )
    merged["is_push"] = merged["home_cover"].isna()

    merged.sort_values(["season", "week", "game_id"], inplace=True)
    return merged


def validate_game_features(df: pd.DataFrame) -> None:
    if df["game_id"].duplicated().any():
        raise ValueError("Duplicate game_id rows detected in wide features.")


def write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def infer_sql_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "double precision"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "timestamptz"
    return "text"


def write_table(conn: psycopg.Connection, df: pd.DataFrame, table_name: str) -> None:
    schema, table = table_name.split(".") if "." in table_name else ("public", table_name)
    columns_sql = []
    for col in df.columns:
        sql_type = infer_sql_type(df[col])
        suffix = " PRIMARY KEY" if col == "game_id" else ""
        columns_sql.append(f"    {col} {sql_type}{suffix}")
    columns_definition = ",\n".join(columns_sql)
    create_sql = f"""
    CREATE SCHEMA IF NOT EXISTS {schema};
    DROP TABLE IF EXISTS {schema}.{table};
    CREATE TABLE {schema}.{table} (
{columns_definition}
    );
    """
    with conn.cursor() as cur:
        cur.execute(create_sql)
        buf = StringIO()
        df.to_csv(buf, index=False, header=False)
        buf.seek(0)
        with cur.copy(f"COPY {schema}.{table} FROM STDIN WITH (FORMAT CSV)") as copy:
            copy.write(buf.read())
    conn.commit()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate as-of team/game features")
    ap.add_argument("--output", default="analysis/features/asof_team_features.csv", help="CSV output path")
    ap.add_argument("--write-table", help="Optional schema.table to write results (overwrites)")
    ap.add_argument("--season-start", type=int, help="Earliest season to retain")
    ap.add_argument("--season-end", type=int, help="Latest season to retain")
    ap.add_argument("--validate", action="store_true", help="Perform leakage/consistency checks")
    return ap.parse_args()


def maybe_filter_season(df: pd.DataFrame, season_start: int | None, season_end: int | None) -> pd.DataFrame:
    filtered = df
    if season_start is not None:
        filtered = filtered[filtered["season"] >= season_start]
    if season_end is not None:
        filtered = filtered[filtered["season"] <= season_end]
    return filtered


def main() -> None:
    args = parse_args()
    with get_connection() as conn:
        team_df = fetch_team_game_features(conn)
    team_df = compute_team_history(team_df)
    if args.validate:
        validate_team_history(team_df)
    merged = pivot_to_games(team_df)
    if args.validate:
        validate_game_features(merged)
    merged = maybe_filter_season(merged, args.season_start, args.season_end)
    write_csv(merged, args.output)
    if args.write_table:
        with get_connection() as conn:
            write_table(conn, merged, args.write_table)
    print(f"[asof] wrote {len(merged)} rows -> {args.output}")


if __name__ == "__main__":
    main()
