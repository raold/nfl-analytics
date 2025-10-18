"""v2 as-of team/game features with Migration 019 features.

This extends asof_features_enhanced.py to include v2 model features:
- Division rivalry detection (is_division_game)
- Thursday night indicator (is_thursday_night, is_short_week)
- Weather categorization (weather_condition)
- Defensive EPA trends (away_def_epa_last_4, home_def_epa_last_4)
- Division H2H variance

These features address Week 6 2025 retrospective findings:
- PHI @ NYG division game miss (20.56 pt error)
- Thursday night volatility
- Low-scoring defensive games
- Tie prediction needs

Example:
  python py/features/asof_features_v2.py --output data/processed/features/asof_team_features_v2.csv \
      --season-start 2020 --validate
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

# Import enhanced module for shared utilities
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from py.features import asof_features_enhanced as enhanced
except Exception:
    import importlib

    enhanced = importlib.import_module("features.asof_features_enhanced")


# Extended SQL with v2 model features
SQL_TEAM_BASE_V2 = """
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
           g.home_penalty_yards AS penalty_yards,
           g.home_rest AS rest_days,
           -- v2 MODEL FEATURES
           g.is_division_game,
           g.is_thursday_night,
           g.is_short_week,
           g.weather_condition,
           g.home_def_epa_last_4 AS def_epa_last_4,
           g.away_def_epa_last_4 AS opp_def_epa_last_4,
           g.division_h2h_variance
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
           g.away_penalty_yards AS penalty_yards,
           g.away_rest AS rest_days,
           -- v2 MODEL FEATURES (away perspective)
           g.is_division_game,
           g.is_thursday_night,
           g.is_short_week,
           g.weather_condition,
           g.away_def_epa_last_4 AS def_epa_last_4,
           g.home_def_epa_last_4 AS opp_def_epa_last_4,
           g.division_h2h_variance
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

           -- Passing efficiency (pass and rush are BOOLEAN)
           COALESCE(AVG(CASE WHEN pass = TRUE THEN air_yards END), 0) AS air_yards_mean,
           COALESCE(AVG(CASE WHEN pass = TRUE THEN yards_after_catch END), 0) AS yac_mean,
           COALESCE(AVG(CASE WHEN pass = TRUE THEN cpoe END), 0) AS cpoe_mean,
           COALESCE(AVG(CASE WHEN pass = TRUE AND complete_pass = 1 THEN 1.0 ELSE 0.0 END), 0) AS completion_pct,

           -- Win probability
           COALESCE(MAX(wp), 0.5) AS wp_max,
           COALESCE(MIN(wp), 0.5) AS wp_min,
           COALESCE(AVG(CASE WHEN quarter = 4 THEN wp END), 0.5) AS wp_q4_mean,

           -- Situational play-calling
           COALESCE(AVG(CASE WHEN shotgun = 1.0 THEN 1.0 ELSE 0.0 END), 0) AS shotgun_rate,
           COALESCE(AVG(CASE WHEN no_huddle = 1.0 THEN 1.0 ELSE 0.0 END), 0) AS no_huddle_rate,

           -- Explosive plays (pass and rush are BOOLEAN)
           COUNT(CASE WHEN pass = TRUE AND yards_gained >= 20 THEN 1 END) AS explosive_pass,
           COUNT(CASE WHEN rush = TRUE AND yards_gained >= 10 THEN 1 END) AS explosive_rush

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
       tg.rest_days,
       tg.points_for - tg.points_against AS margin,

       -- v2 MODEL FEATURES
       tg.is_division_game,
       tg.is_thursday_night,
       tg.is_short_week,
       tg.weather_condition,
       tg.def_epa_last_4,
       tg.opp_def_epa_last_4,
       tg.division_h2h_variance,

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


# Extended feature column list with v2 features
TEAM_FEATURE_COLUMNS_V2 = enhanced.TEAM_FEATURE_COLUMNS_ENHANCED + [
    # v2 features (current game flags)
    "is_division_game",
    "is_thursday_night",
    "is_short_week",
    "weather_condition",
    "def_epa_last_4",
    "opp_def_epa_last_4",
    "division_h2h_variance",
]


# Features to diff home vs away (exclude game-level flags)
DIFF_FEATURE_COLUMNS_V2 = enhanced.DIFF_FEATURE_COLUMNS_ENHANCED + [
    "def_epa_last_4",  # Defensive strength differential
]


def fetch_team_game_features_v2(conn: psycopg.Connection) -> pd.DataFrame:
    """Fetch team-game features with v2 model features."""
    return pd.read_sql(SQL_TEAM_BASE_V2, conn)


def compute_team_history_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Extend enhanced history computation with v2 features."""

    # Start with enhanced computation (handles all enhanced features)
    team_df = enhanced.compute_team_history_enhanced(df)

    # v2 features don't need rolling computation as they're pre-computed in DB
    # But we do need to handle NULLs and create derived features

    # Fill NULLs for v2 features
    team_df["is_division_game"] = team_df["is_division_game"].fillna(False)
    team_df["is_thursday_night"] = team_df["is_thursday_night"].fillna(False)
    team_df["is_short_week"] = team_df["is_short_week"].fillna(False)
    team_df["weather_condition"] = team_df["weather_condition"].fillna("unknown")
    team_df["def_epa_last_4"] = team_df["def_epa_last_4"].fillna(0.0)
    team_df["opp_def_epa_last_4"] = team_df["opp_def_epa_last_4"].fillna(0.0)
    team_df["division_h2h_variance"] = team_df["division_h2h_variance"].fillna(0.0)

    # Compute derived features for v2
    team_group = team_df.groupby("team", group_keys=False)

    # Historical division game win rate (as-of)
    team_df["division_game_indicator"] = team_df["is_division_game"].astype(float)
    team_df["division_game_win"] = (team_df["is_division_game"] & (team_df["margin"] > 0)).astype(
        float
    )

    team_df["prior_division_games"] = team_group["division_game_indicator"].transform(
        lambda s: s.shift(1).cumsum()
    )
    team_df["prior_division_wins"] = team_group["division_game_win"].transform(
        lambda s: s.shift(1).cumsum()
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        team_df["prior_division_win_rate"] = np.where(
            team_df["prior_division_games"] > 0,
            team_df["prior_division_wins"] / team_df["prior_division_games"],
            0.5,  # Default 50% for no history
        )

    # Thursday night performance (as-of)
    team_df["thursday_game_indicator"] = team_df["is_thursday_night"].astype(float)
    team_df["thursday_game_margin"] = np.where(team_df["is_thursday_night"], team_df["margin"], 0.0)

    team_df["prior_thursday_games"] = team_group["thursday_game_indicator"].transform(
        lambda s: s.shift(1).cumsum()
    )
    team_df["prior_thursday_margin_sum"] = team_group["thursday_game_margin"].transform(
        lambda s: s.shift(1).cumsum()
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        team_df["prior_thursday_avg_margin"] = np.where(
            team_df["prior_thursday_games"] > 0,
            team_df["prior_thursday_margin_sum"] / team_df["prior_thursday_games"],
            0.0,
        )

    # Short week performance (as-of)
    team_df["short_week_indicator"] = team_df["is_short_week"].astype(float)
    team_df["short_week_margin"] = np.where(team_df["is_short_week"], team_df["margin"], 0.0)

    team_df["prior_short_week_games"] = team_group["short_week_indicator"].transform(
        lambda s: s.shift(1).cumsum()
    )
    team_df["prior_short_week_margin_sum"] = team_group["short_week_margin"].transform(
        lambda s: s.shift(1).cumsum()
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        team_df["prior_short_week_avg_margin"] = np.where(
            team_df["prior_short_week_games"] > 0,
            team_df["prior_short_week_margin_sum"] / team_df["prior_short_week_games"],
            0.0,
        )

    return team_df


def pivot_to_games_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot v2 team features to game-level home/away format."""
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

    # v2 game-level features (don't split by home/away)
    v2_game_features = [
        "is_division_game",
        "is_thursday_night",
        "is_short_week",
        "weather_condition",
        "division_h2h_variance",
    ]

    home = df[df["is_home"]].copy()
    away = df[~df["is_home"]].copy()

    # Team-specific features (split home/away)

    team_features = TEAM_FEATURE_COLUMNS_V2.copy()
    # Remove game-level features from team split (they're added at game level)
    for feat in v2_game_features:
        if feat in team_features:
            team_features.remove(feat)

    home = home[common_cols + v2_game_features + team_features].rename(
        columns=enhanced.base.rename_with_prefix(team_features, "home")
    )
    away = away[common_cols + team_features].rename(
        columns=enhanced.base.rename_with_prefix(team_features, "away")
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

    # Fill v2 features
    merged["is_division_game"] = merged["is_division_game"].fillna(False).astype(bool)
    merged["is_thursday_night"] = merged["is_thursday_night"].fillna(False).astype(bool)
    merged["is_short_week"] = merged["is_short_week"].fillna(False).astype(bool)
    merged["weather_condition"] = merged["weather_condition"].fillna("unknown")

    for col in [
        "home_def_epa_last_4",
        "away_def_epa_last_4",
        "home_prior_division_win_rate",
        "away_prior_division_win_rate",
        "home_prior_thursday_avg_margin",
        "away_prior_thursday_avg_margin",
        "home_prior_short_week_avg_margin",
        "away_prior_short_week_avg_margin",
        "division_h2h_variance",
    ]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)

    # Compute differentials for v2 features
    for col in DIFF_FEATURE_COLUMNS_V2:
        home_col = f"home_{col}"
        away_col = f"away_{col}"
        diff_col = f"{col}_diff"
        if home_col in merged.columns and away_col in merged.columns:
            merged[diff_col] = merged[home_col] - merged[away_col]

    # Add manual differentials (for backwards compatibility)
    if "home_prior_epa_mean" in merged.columns and "away_prior_epa_mean" in merged.columns:
        merged["epa_diff_prior"] = merged["home_prior_epa_mean"] - merged["away_prior_epa_mean"]
    if "home_prior_plays" in merged.columns and "away_prior_plays" in merged.columns:
        merged["plays_diff_prior"] = merged["home_prior_plays"] - merged["away_prior_plays"]
    if "home_rest_days" in merged.columns and "away_rest_days" in merged.columns:
        merged["rest_diff"] = merged["home_rest_days"] - merged["away_rest_days"]

    # v2 specific differentials
    if "home_def_epa_last_4" in merged.columns and "away_def_epa_last_4" in merged.columns:
        merged["def_epa_last_4_diff"] = (
            merged["home_def_epa_last_4"] - merged["away_def_epa_last_4"]
        )

    # One-hot encode weather_condition
    weather_dummies = pd.get_dummies(merged["weather_condition"], prefix="weather")
    merged = pd.concat([merged, weather_dummies], axis=1)

    # Add betting outcomes
    merged["home_margin"] = merged["home_score"] - merged["away_score"]
    merged["home_win"] = (merged["home_margin"] > 0).astype(float)
    merged["home_cover"] = (merged["home_margin"] + merged["spread_close"] > 0).astype(float)
    merged["over_hit"] = (
        (merged["home_score"] + merged["away_score"]) > merged["total_close"]
    ).astype(float)
    merged["is_push"] = merged["home_cover"].isna()

    # Tie detection (for v2 model)
    merged["is_tie"] = (merged["home_margin"] == 0).astype(bool)

    # Add tie probability indicator (for spreads < 1.5)
    merged["high_tie_prob"] = (merged["spread_close"].abs() < 1.5).fillna(False).astype(bool)

    merged.sort_values(["season", "week", "game_id"], inplace=True)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Generate v2 as-of team features")
    parser.add_argument(
        "--output",
        default="data/processed/features/asof_team_features_v2.csv",
        help="Output CSV path",
    )
    parser.add_argument("--season-start", type=int, default=2020, help="First season to include")
    parser.add_argument("--validate", action="store_true", help="Run validation checks")
    args = parser.parse_args()

    print("=== v2 As-Of Feature Generation (with Migration 019 features) ===")
    print("Connecting to database...")
    conn = enhanced.base.get_connection()

    print("Fetching team-game data with v2 features...")
    team_df = fetch_team_game_features_v2(conn)
    print(f"Loaded {len(team_df)} team-game records")

    print("Computing team history with v2 features...")
    team_history = compute_team_history_v2(team_df)

    if args.validate:
        print("Validating team history...")
        enhanced.base.validate_team_history(team_history)
        print("[PASS] Validation passed")

    print("Pivoting to game-level format...")
    games_wide = pivot_to_games_v2(team_history)

    # Filter to requested seasons
    games_wide = games_wide[games_wide["season"] >= args.season_start].copy()
    print(f"Filtered to {len(games_wide)} games (seasons >= {args.season_start})")

    print(f"Writing to {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    games_wide.to_csv(args.output, index=False)

    print(
        f"[SUCCESS] Feature generation complete: {games_wide.shape[0]} games, {games_wide.shape[1]} columns"
    )

    # Print summary of v2 features
    v2_features = [
        c
        for c in games_wide.columns
        if any(
            x in c
            for x in [
                "division_game",
                "thursday",
                "short_week",
                "weather",
                "def_epa_last_4",
                "division_h2h",
                "tie",
            ]
        )
    ]
    print(f"\nv2 model features ({len(v2_features)}):")
    for feat in sorted(v2_features):
        print(f"  - {feat}")

    # Print feature counts by category
    print("\nFeature category counts:")
    print(
        f"  - Game-level flags: {sum('is_' in c and 'prior' not in c for c in games_wide.columns)}"
    )
    print(f"  - Weather categories: {sum('weather_' in c for c in games_wide.columns)}")
    print(f"  - Defensive EPA: {sum('def_epa' in c for c in games_wide.columns)}")
    print(f"  - Division history: {sum('division' in c for c in games_wide.columns)}")
    print(
        f"  - Thursday/Short week history: {sum('thursday' in c or 'short_week' in c for c in games_wide.columns)}"
    )

    # Sample 2025 Week 6 features
    week6_2025 = games_wide[(games_wide["season"] == 2025) & (games_wide["week"] == 6)]
    if len(week6_2025) > 0:
        print("\n2025 Week 6 sample (first 5 games):")
        sample_cols = [
            "game_id",
            "home_team",
            "away_team",
            "is_division_game",
            "is_thursday_night",
            "is_short_week",
            "weather_condition",
            "home_def_epa_last_4",
            "away_def_epa_last_4",
        ]
        sample = week6_2025[sample_cols].head(5)
        print(sample.to_string(index=False))

    conn.close()


if __name__ == "__main__":
    main()
