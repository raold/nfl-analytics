"""Enhanced as-of team/game features incorporating new backfilled columns.

This extends asof_features.py to include:
- Advanced play-level metrics: success rate, air yards, completion %, win probability
- Turnover and penalty differentials from games table
- QB and coaching stability metrics

Example:
  python py/features/asof_features_enhanced.py --output analysis/features/asof_team_features_enhanced.csv \
      --season-start 2003 --validate
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

# Import base module for shared utilities
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from py.features import asof_features as base
except Exception:
    import importlib
    base = importlib.import_module("features.asof_features")


# Enhanced SQL with new play-level features
SQL_TEAM_BASE_ENHANCED = """
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
           g.home_coach AS coach_name,
           g.home_turnovers AS turnovers,
           g.away_turnovers AS turnovers_opp,
           g.home_penalties AS penalties,
           g.home_penalty_yards AS penalty_yards
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
           g.away_coach AS coach_name,
           g.away_turnovers AS turnovers,
           g.home_turnovers AS turnovers_opp,
           g.away_penalties AS penalties,
           g.away_penalty_yards AS penalty_yards
    FROM games g
),
team_stats AS (
    SELECT game_id,
           posteam AS team,
           -- Base metrics
           COALESCE(SUM(epa), 0) AS epa_sum,
           COUNT(*) AS plays,
           
           -- Success rate (EPA > 0)
           COALESCE(AVG(CASE WHEN success = 1.0 THEN 1.0 ELSE 0.0 END), 0) AS success_rate,
           
           -- Passing efficiency
           COALESCE(AVG(CASE WHEN pass THEN air_yards END), 0) AS air_yards_mean,
           COALESCE(AVG(CASE WHEN pass THEN yards_after_catch END), 0) AS yac_mean,
           COALESCE(AVG(CASE WHEN pass THEN cpoe END), 0) AS cpoe_mean,
           COALESCE(AVG(CASE WHEN pass AND complete_pass = 1.0 THEN 1.0 ELSE 0.0 END), 0) AS completion_pct,
           
           -- Win probability
           COALESCE(MAX(wp), 0.5) AS wp_max,
           COALESCE(MIN(wp), 0.5) AS wp_min,
           COALESCE(AVG(CASE WHEN quarter = 4 THEN wp END), 0.5) AS wp_q4_mean,
           
           -- Situational play-calling
           COALESCE(AVG(CASE WHEN shotgun = 1.0 THEN 1.0 ELSE 0.0 END), 0) AS shotgun_rate,
           COALESCE(AVG(CASE WHEN no_huddle = 1.0 THEN 1.0 ELSE 0.0 END), 0) AS no_huddle_rate,
           
           -- Explosive plays
           COUNT(CASE WHEN pass AND yards_gained >= 20 THEN 1 END) AS explosive_pass,
           COUNT(CASE WHEN rush AND yards_gained >= 10 THEN 1 END) AS explosive_rush
           
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
       tg.turnovers,
       tg.turnovers_opp,
       tg.penalties,
       tg.penalty_yards,
       tg.points_for - tg.points_against AS margin,
       
       -- Base EPA
       COALESCE(ts.epa_sum, 0) AS epa_sum,
       COALESCE(ts.plays, 0) AS plays,
       
       -- Advanced play metrics
       COALESCE(ts.success_rate, 0) AS success_rate,
       COALESCE(ts.air_yards_mean, 0) AS air_yards_mean,
       COALESCE(ts.yac_mean, 0) AS yac_mean,
       COALESCE(ts.cpoe_mean, 0) AS cpoe_mean,
       COALESCE(ts.completion_pct, 0) AS completion_pct,
       COALESCE(ts.wp_max, 0.5) AS wp_max,
       COALESCE(ts.wp_min, 0.5) AS wp_min,
       COALESCE(ts.wp_q4_mean, 0.5) AS wp_q4_mean,
       COALESCE(ts.shotgun_rate, 0) AS shotgun_rate,
       COALESCE(ts.no_huddle_rate, 0) AS no_huddle_rate,
       COALESCE(ts.explosive_pass, 0) AS explosive_pass,
       COALESCE(ts.explosive_rush, 0) AS explosive_rush
       
FROM team_games tg
LEFT JOIN team_stats ts
  ON ts.game_id = tg.game_id AND ts.team = tg.team
ORDER BY tg.team, tg.season, tg.week, tg.kickoff NULLS FIRST, tg.game_id;
"""


# Extended feature column list
TEAM_FEATURE_COLUMNS_ENHANCED = base.TEAM_FEATURE_COLUMNS + [
    # Game-level (from games table)
    "prior_turnovers_avg",
    "prior_turnovers_opp_avg",
    "prior_penalties_avg",
    "prior_penalty_yards_avg",
    
    # Play-level efficiency (from plays table)
    "prior_success_rate",
    "success_rate_last3",
    "success_rate_last5",
    "prior_air_yards",
    "prior_yac",
    "prior_cpoe",
    "prior_completion_pct",
    "prior_wp_q4_mean",
    "prior_shotgun_rate",
    "prior_no_huddle_rate",
    "prior_explosive_pass_rate",
    "prior_explosive_rush_rate",
]


# Features to diff home vs away
DIFF_FEATURE_COLUMNS_ENHANCED = base.DIFF_FEATURE_COLUMNS + [
    "prior_turnovers_avg",
    "prior_penalties_avg",
    "prior_success_rate",
    "success_rate_last3",
    "prior_air_yards",
    "prior_cpoe",
    "prior_completion_pct",
    "prior_shotgun_rate",
    "prior_explosive_pass_rate",
]


def fetch_team_game_features(conn: psycopg.Connection) -> pd.DataFrame:
    """Fetch team-game features with enhanced play-level metrics."""
    return pd.read_sql(SQL_TEAM_BASE_ENHANCED, conn)


def compute_team_history_enhanced(df: pd.DataFrame) -> pd.DataFrame:
    """Extend base history computation with new features."""
    
    # Start with base computation (handles all original features)
    team_df = base.compute_team_history(df)
    
    # Now add new rolling features
    team_df["turnovers"] = team_df["turnovers"].fillna(0.0)
    team_df["turnovers_opp"] = team_df["turnovers_opp"].fillna(0.0)
    team_df["penalties"] = team_df["penalties"].fillna(0.0)
    team_df["penalty_yards"] = team_df["penalty_yards"].fillna(0.0)
    
    team_group = team_df.groupby("team", group_keys=False)
    
    # Turnover/penalty priors
    team_df["prior_turnovers_sum"] = team_group["turnovers"].cumsum() - team_df["turnovers"]
    team_df["prior_turnovers_opp_sum"] = team_group["turnovers_opp"].cumsum() - team_df["turnovers_opp"]
    team_df["prior_penalties_sum"] = team_group["penalties"].cumsum() - team_df["penalties"]
    team_df["prior_penalty_yards_sum"] = team_group["penalty_yards"].cumsum() - team_df["penalty_yards"]
    
    with np.errstate(divide="ignore", invalid="ignore"):
        team_df["prior_turnovers_avg"] = np.where(
            team_df["prior_games"] > 0,
            team_df["prior_turnovers_sum"] / team_df["prior_games"],
            np.nan
        )
        team_df["prior_turnovers_opp_avg"] = np.where(
            team_df["prior_games"] > 0,
            team_df["prior_turnovers_opp_sum"] / team_df["prior_games"],
            np.nan
        )
        team_df["prior_penalties_avg"] = np.where(
            team_df["prior_games"] > 0,
            team_df["prior_penalties_sum"] / team_df["prior_games"],
            np.nan
        )
        team_df["prior_penalty_yards_avg"] = np.where(
            team_df["prior_games"] > 0,
            team_df["prior_penalty_yards_sum"] / team_df["prior_games"],
            np.nan
        )
    
    # Play-level efficiency priors (success rate, air yards, etc.)
    team_df["success_rate"] = team_df["success_rate"].fillna(0.0)
    team_df["air_yards_mean"] = team_df["air_yards_mean"].fillna(0.0)
    team_df["yac_mean"] = team_df["yac_mean"].fillna(0.0)
    team_df["cpoe_mean"] = team_df["cpoe_mean"].fillna(0.0)
    team_df["completion_pct"] = team_df["completion_pct"].fillna(0.0)
    team_df["wp_q4_mean"] = team_df["wp_q4_mean"].fillna(0.5)
    team_df["shotgun_rate"] = team_df["shotgun_rate"].fillna(0.0)
    team_df["no_huddle_rate"] = team_df["no_huddle_rate"].fillna(0.0)
    
    # Rolling averages for new metrics
    team_df["prior_success_rate"] = team_group["success_rate"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    team_df["success_rate_last3"] = team_group["success_rate"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
    )
    team_df["success_rate_last5"] = team_group["success_rate"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )
    
    team_df["prior_air_yards"] = team_group["air_yards_mean"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    team_df["prior_yac"] = team_group["yac_mean"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    team_df["prior_cpoe"] = team_group["cpoe_mean"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    team_df["prior_completion_pct"] = team_group["completion_pct"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    team_df["prior_wp_q4_mean"] = team_group["wp_q4_mean"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    team_df["prior_shotgun_rate"] = team_group["shotgun_rate"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    team_df["prior_no_huddle_rate"] = team_group["no_huddle_rate"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    
    # Explosive play rates
    team_df["explosive_pass"] = team_df["explosive_pass"].fillna(0.0)
    team_df["explosive_rush"] = team_df["explosive_rush"].fillna(0.0)
    
    with np.errstate(divide="ignore", invalid="ignore"):
        team_df["explosive_pass_rate"] = np.where(
            team_df["plays"] > 0,
            team_df["explosive_pass"] / team_df["plays"],
            0.0
        )
        team_df["explosive_rush_rate"] = np.where(
            team_df["plays"] > 0,
            team_df["explosive_rush"] / team_df["plays"],
            0.0
        )
    
    team_df["prior_explosive_pass_rate"] = team_group["explosive_pass_rate"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    team_df["prior_explosive_rush_rate"] = team_group["explosive_rush_rate"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    
    return team_df


def pivot_to_games_enhanced(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot enhanced team features to game-level home/away format."""
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

    home = home[common_cols + TEAM_FEATURE_COLUMNS_ENHANCED].rename(
        columns=base.rename_with_prefix(TEAM_FEATURE_COLUMNS_ENHANCED, "home")
    )
    away = away[common_cols + TEAM_FEATURE_COLUMNS_ENHANCED].rename(
        columns=base.rename_with_prefix(TEAM_FEATURE_COLUMNS_ENHANCED, "away")
    )

    merged = pd.merge(home, away, on=common_cols, how="inner", suffixes=("", "_dup"))
    merged = merged[[c for c in merged.columns if not c.endswith("_dup")]]

    # Fill NA values for key columns
    for col in [
        "home_prior_epa_sum",
        "home_prior_plays",
        "away_prior_epa_sum",
        "away_prior_plays",
    ]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)

    for col in ["home_rest_days", "away_rest_days"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(7.0)

    # Compute differentials for enhanced features
    for col in DIFF_FEATURE_COLUMNS_ENHANCED:
        home_col = f"home_{col}"
        away_col = f"away_{col}"
        diff_col = f"{col}_diff"
        if home_col in merged.columns and away_col in merged.columns:
            merged[diff_col] = merged[home_col] - merged[away_col]

    # Add manual differentials (for backwards compatibility with original feature names)
    if "home_prior_epa_mean" in merged.columns and "away_prior_epa_mean" in merged.columns:
        merged["epa_diff_prior"] = merged["home_prior_epa_mean"] - merged["away_prior_epa_mean"]
    if "home_prior_plays" in merged.columns and "away_prior_plays" in merged.columns:
        merged["plays_diff_prior"] = merged["home_prior_plays"] - merged["away_prior_plays"]
    if "home_rest_days" in merged.columns and "away_rest_days" in merged.columns:
        merged["rest_diff"] = merged["home_rest_days"] - merged["away_rest_days"]

    # Add betting outcomes
    merged["home_margin"] = merged["home_score"] - merged["away_score"]
    merged["home_win"] = (merged["home_margin"] > 0).astype(float)
    merged["home_cover"] = (merged["home_margin"] + merged["spread_close"] > 0).astype(float)
    merged["over_hit"] = ((merged["home_score"] + merged["away_score"]) > merged["total_close"]).astype(float)
    merged["is_push"] = merged["home_cover"].isna()

    merged.sort_values(["season", "week", "game_id"], inplace=True)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Generate enhanced as-of team features")
    parser.add_argument(
        "--output", 
        default="data/processed/features/asof_team_features_enhanced.csv",
        help="Output CSV path"
    )
    parser.add_argument("--season-start", type=int, default=2003, help="First season to include")
    parser.add_argument("--validate", action="store_true", help="Run validation checks")
    args = parser.parse_args()

    print("=== Enhanced As-Of Feature Generation ===")
    print(f"Connecting to database...")
    conn = base.get_connection()

    print("Fetching team-game data with enhanced metrics...")
    team_df = fetch_team_game_features(conn)
    print(f"Loaded {len(team_df)} team-game records")

    print("Computing team history with enhanced features...")
    team_history = compute_team_history_enhanced(team_df)
    
    if args.validate:
        print("Validating team history...")
        base.validate_team_history(team_history)
        print("✓ Validation passed")

    print("Pivoting to game-level format...")
    games_wide = pivot_to_games_enhanced(team_history)
    
    # Filter to requested seasons
    games_wide = games_wide[games_wide["season"] >= args.season_start].copy()
    print(f"Filtered to {len(games_wide)} games (seasons >= {args.season_start})")

    print(f"Writing to {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    games_wide.to_csv(args.output, index=False)

    print(f"✓ Feature generation complete: {games_wide.shape[0]} games, {games_wide.shape[1]} columns")
    
    # Print summary of new features
    new_features = [c for c in games_wide.columns if any(
        x in c for x in ["success_rate", "air_yards", "cpoe", "turnover", "penalty", "shotgun", "explosive"]
    )]
    print(f"\nNew enhanced features ({len(new_features)}):")
    for feat in sorted(new_features)[:20]:  # Show first 20
        print(f"  - {feat}")
    if len(new_features) > 20:
        print(f"  ... and {len(new_features) - 20} more")

    conn.close()


if __name__ == "__main__":
    main()
