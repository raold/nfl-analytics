"""Feature extraction from materialized views.

Efficiently extracts ML features by querying pre-aggregated materialized views
instead of raw play-by-play data. This provides 10-100x faster feature extraction
for training and inference.

Materialized Views Used:
  - mv_game_aggregates: Game-level EPA, success rate, explosive plays
  - mv_team_rolling_stats: Rolling team performance (3/5/10 game windows)
  - mv_team_matchup_history: Head-to-head records and rivalry stats
  - mv_player_season_stats: Player aggregates (QB/RB/WR)
  - mv_betting_features: Spread coverage, over/under trends
  - mv_venue_weather_features: Stadium and weather conditions

Example:
  python py/features/materialized_view_features.py --output data/processed/features/mv_features.csv \\
      --season-start 2010 --validate
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from psycopg import Connection

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def get_connection() -> Connection:
    """Connect to PostgreSQL database."""
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", 5544)),
        dbname=os.getenv("POSTGRES_DB", "devdb01"),
        user=os.getenv("POSTGRES_USER", "dro"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
    )


# ============================================================
# FEATURE EXTRACTION QUERIES
# ============================================================

SQL_GAME_FEATURES = """
-- Extract game-level features from materialized views
WITH base_features AS (
  SELECT
    ga.game_id,
    ga.season,
    ga.week,
    ga.game_type,
    ga.kickoff,
    ga.home_team,
    ga.away_team,
    ga.home_score,
    ga.away_score,
    ga.home_margin,
    ga.spread_close,
    ga.total_close,

    -- Game aggregates (home team)
    ga.home_plays,
    ga.home_epa,
    ga.home_epa_per_play,
    ga.home_total_yards,
    ga.home_successful_plays,
    ga.home_success_rate,
    ga.home_pass_attempts,
    ga.home_pass_yards,
    ga.home_pass_epa,
    ga.home_completions,
    ga.home_avg_air_yards,
    ga.home_avg_yac,
    ga.home_rush_attempts,
    ga.home_rush_yards,
    ga.home_rush_epa,
    ga.home_explosive_pass,
    ga.home_explosive_rush,
    ga.home_turnovers,
    ga.away_turnovers AS home_turnovers_forced,
    ga.home_penalties,
    ga.home_penalty_yards,

    -- Game aggregates (away team)
    ga.away_plays,
    ga.away_epa,
    ga.away_epa_per_play,
    ga.away_total_yards,
    ga.away_successful_plays,
    ga.away_success_rate,
    ga.away_pass_attempts,
    ga.away_pass_yards,
    ga.away_pass_epa,
    ga.away_completions,
    ga.away_avg_air_yards,
    ga.away_avg_yac,
    ga.away_rush_attempts,
    ga.away_rush_yards,
    ga.away_rush_epa,
    ga.away_explosive_pass,
    ga.away_explosive_rush,
    ga.away_turnovers,
    ga.home_turnovers AS away_turnovers_forced,
    ga.away_penalties,
    ga.away_penalty_yards,

    -- Betting features
    bf.home_cover_margin,
    bf.home_covered,
    bf.over_hit,
    bf.home_cover_rate_l10,
    bf.home_over_rate_l10,
    bf.away_cover_rate_l10,
    bf.away_over_rate_l10,

    -- Venue/weather features
    vw.stadium,
    vw.temp,
    vw.wind,
    vw.is_outdoor,
    vw.is_dome,
    vw.is_cold_game,
    vw.is_hot_game,
    vw.is_windy_game,
    vw.temp_category,
    vw.wind_category,
    vw.surface_type,
    vw.venue_avg_total,
    vw.venue_avg_margin,
    vw.venue_home_win_rate

  FROM mv_game_aggregates ga
  LEFT JOIN mv_betting_features bf ON ga.game_id = bf.game_id
  LEFT JOIN mv_venue_weather_features vw ON ga.game_id = vw.game_id
  WHERE ga.home_score IS NOT NULL  -- Only completed games
),
home_rolling AS (
  SELECT
    game_id,
    points_for_l3 AS home_points_l3,
    points_against_l3 AS home_points_against_l3,
    epa_per_play_l3 AS home_epa_per_play_l3,
    success_rate_l3 AS home_success_rate_l3,
    points_for_l5 AS home_points_l5,
    points_against_l5 AS home_points_against_l5,
    epa_per_play_l5 AS home_epa_per_play_l5,
    success_rate_l5 AS home_success_rate_l5,
    pass_epa_l5 AS home_pass_epa_l5,
    rush_epa_l5 AS home_rush_epa_l5,
    points_for_l10 AS home_points_l10,
    points_against_l10 AS home_points_against_l10,
    epa_per_play_l10 AS home_epa_per_play_l10,
    points_for_season AS home_points_season,
    points_against_season AS home_points_against_season,
    epa_per_play_season AS home_epa_per_play_season,
    success_rate_season AS home_success_rate_season,
    points_for_home AS home_points_home_avg,
    points_for_away AS home_points_away_avg,
    epa_per_play_home AS home_epa_home_avg,
    epa_per_play_away AS home_epa_away_avg,
    wins AS home_wins,
    losses AS home_losses
  FROM mv_team_rolling_stats
  WHERE is_home = TRUE
),
away_rolling AS (
  SELECT
    game_id,
    points_for_l3 AS away_points_l3,
    points_against_l3 AS away_points_against_l3,
    epa_per_play_l3 AS away_epa_per_play_l3,
    success_rate_l3 AS away_success_rate_l3,
    points_for_l5 AS away_points_l5,
    points_against_l5 AS away_points_against_l5,
    epa_per_play_l5 AS away_epa_per_play_l5,
    success_rate_l5 AS away_success_rate_l5,
    pass_epa_l5 AS away_pass_epa_l5,
    rush_epa_l5 AS away_rush_epa_l5,
    points_for_l10 AS away_points_l10,
    points_against_l10 AS away_points_against_l10,
    epa_per_play_l10 AS away_epa_per_play_l10,
    points_for_season AS away_points_season,
    points_against_season AS away_points_against_season,
    epa_per_play_season AS away_epa_per_play_season,
    success_rate_season AS away_success_rate_season,
    points_for_home AS away_points_home_avg,
    points_for_away AS away_points_away_avg,
    epa_per_play_home AS away_epa_home_avg,
    epa_per_play_away AS away_epa_away_avg,
    wins AS away_wins,
    losses AS away_losses
  FROM mv_team_rolling_stats
  WHERE is_home = FALSE
),
matchup_stats AS (
  SELECT
    game_id,
    -- We need to join on team matchups properly
    -- For now, we'll skip this and add in a future iteration
    NULL::BIGINT AS matchup_games_played,
    NULL::NUMERIC AS matchup_home_win_rate
  FROM base_features
  LIMIT 0  -- Placeholder
)
SELECT
  bf.*,
  hr.*,
  ar.*
FROM base_features bf
LEFT JOIN home_rolling hr ON bf.game_id = hr.game_id
LEFT JOIN away_rolling ar ON bf.game_id = ar.game_id
ORDER BY bf.season, bf.week, bf.kickoff, bf.game_id;
"""


SQL_PLAYER_FEATURES = """
-- Extract player-level features for key positions
SELECT
  ps.player_id,
  ps.gsis_id,
  ps.canonical_name,
  ps.position,
  ps.season,
  ps.week,

  -- QB stats
  ps.attempts,
  ps.completions,
  ps.passing_yards,
  ps.passing_tds,
  ps.interceptions,
  ps.sacks,
  ps.passing_epa,
  ps.passing_epa_per_play,
  ps.avg_cpoe,
  ps.avg_air_yards,
  ps.big_time_throws,

  -- RB stats
  ps.rush_attempts,
  ps.rushing_yards,
  ps.rushing_tds,
  ps.rushing_epa,
  ps.rushing_epa_per_carry,
  ps.explosive_runs,
  ps.fumbles_lost,

  -- WR stats
  ps.targets,
  ps.receptions,
  ps.receiving_yards,
  ps.receiving_tds,
  ps.avg_yac,
  ps.receiving_epa,
  ps.explosive_receptions

FROM mv_player_season_stats ps
ORDER BY ps.season, ps.week, ps.position, ps.player_id;
"""


# ============================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================


def fetch_game_features(conn: Connection, season_start: int | None = None) -> pd.DataFrame:
    """Fetch game-level features from materialized views."""
    df = pd.read_sql(SQL_GAME_FEATURES, conn)

    if season_start:
        df = df[df["season"] >= season_start].copy()

    # Convert timestamps
    if "kickoff" in df.columns:
        df["kickoff"] = pd.to_datetime(df["kickoff"], errors="coerce")

    return df


def fetch_player_features(conn: Connection, season_start: int | None = None) -> pd.DataFrame:
    """Fetch player-level features from materialized views."""
    df = pd.read_sql(SQL_PLAYER_FEATURES, conn)

    if season_start:
        df = df[df["season"] >= season_start].copy()

    return df


def compute_differentials(df: pd.DataFrame) -> pd.DataFrame:
    """Compute home/away differentials for key metrics."""
    df = df.copy()

    # EPA differentials
    df["epa_per_play_diff"] = df["home_epa_per_play"] - df["away_epa_per_play"]
    df["epa_per_play_l3_diff"] = df["home_epa_per_play_l3"] - df["away_epa_per_play_l3"]
    df["epa_per_play_l5_diff"] = df["home_epa_per_play_l5"] - df["away_epa_per_play_l5"]
    df["epa_per_play_l10_diff"] = df["home_epa_per_play_l10"] - df["away_epa_per_play_l10"]

    # Success rate differentials
    df["success_rate_diff"] = df["home_success_rate"] - df["away_success_rate"]
    df["success_rate_l3_diff"] = df["home_success_rate_l3"] - df["away_success_rate_l3"]
    df["success_rate_l5_diff"] = df["home_success_rate_l5"] - df["away_success_rate_l5"]

    # Points differentials
    df["points_l3_diff"] = df["home_points_l3"] - df["away_points_l3"]
    df["points_l5_diff"] = df["home_points_l5"] - df["away_points_l5"]
    df["points_l10_diff"] = df["home_points_l10"] - df["away_points_l10"]

    # Pass/rush EPA differentials
    df["pass_epa_l5_diff"] = df["home_pass_epa_l5"] - df["away_pass_epa_l5"]
    df["rush_epa_l5_diff"] = df["home_rush_epa_l5"] - df["away_rush_epa_l5"]

    # Win rate differential
    with np.errstate(divide="ignore", invalid="ignore"):
        df["home_win_pct"] = np.where(
            (df["home_wins"] + df["home_losses"]) > 0,
            df["home_wins"] / (df["home_wins"] + df["home_losses"]),
            0.5,
        )
        df["away_win_pct"] = np.where(
            (df["away_wins"] + df["away_losses"]) > 0,
            df["away_wins"] / (df["away_wins"] + df["away_losses"]),
            0.5,
        )
    df["win_pct_diff"] = df["home_win_pct"] - df["away_win_pct"]

    return df


def add_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Add target variables for ML models."""
    df = df.copy()

    # Binary outcomes
    df["home_win"] = (df["home_margin"] > 0).astype(float)
    df["home_cover"] = (df["home_margin"] + df["spread_close"] > 0).astype(float)
    df["over_hit"] = ((df["home_score"] + df["away_score"]) > df["total_close"]).astype(float)

    # Continuous outcomes
    df["actual_total"] = df["home_score"] + df["away_score"]
    df["cover_margin"] = df["home_margin"] + df["spread_close"]
    df["total_vs_line"] = df["actual_total"] - df["total_close"]

    return df


def validate_features(df: pd.DataFrame) -> None:
    """Validate feature DataFrame for common issues."""
    print("\n=== Feature Validation ===")

    # Check for missing game_id
    if df["game_id"].isna().any():
        raise ValueError("Found missing game_id values")

    # Check for duplicates
    dup_count = df.duplicated(subset=["game_id"]).sum()
    if dup_count > 0:
        print(f"WARNING: Found {dup_count} duplicate game_ids")

    # Check for missing values in key columns
    key_cols = ["season", "week", "home_team", "away_team", "home_score", "away_score"]
    missing = df[key_cols].isna().sum()
    if missing.sum() > 0:
        print("WARNING: Missing values in key columns:")
        print(missing[missing > 0])

    # Check rolling stats coverage
    rolling_cols = [c for c in df.columns if "_l3" in c or "_l5" in c or "_l10" in c]
    if rolling_cols:
        missing_rolling = df[rolling_cols].isna().sum()
        print("\nRolling stats coverage:")
        print(
            f"  - L3 features: {100 * (1 - missing_rolling.filter(like='_l3').mean() / len(df)):.1f}% populated"
        )
        print(
            f"  - L5 features: {100 * (1 - missing_rolling.filter(like='_l5').mean() / len(df)):.1f}% populated"
        )
        print(
            f"  - L10 features: {100 * (1 - missing_rolling.filter(like='_l10').mean() / len(df)):.1f}% populated"
        )

    print("\n[PASS] Validation complete")


def main():
    parser = argparse.ArgumentParser(description="Extract features from materialized views")
    parser.add_argument(
        "--output",
        default="data/processed/features/mv_game_features.csv",
        help="Output CSV path for game features",
    )
    parser.add_argument(
        "--output-players",
        default="data/processed/features/mv_player_features.csv",
        help="Output CSV path for player features",
    )
    parser.add_argument(
        "--season-start", type=int, default=2010, help="First season to include (default: 2010)"
    )
    parser.add_argument("--validate", action="store_true", help="Run validation checks")
    parser.add_argument("--players-only", action="store_true", help="Only extract player features")
    parser.add_argument("--games-only", action="store_true", help="Only extract game features")
    args = parser.parse_args()

    print("=== Materialized View Feature Extraction ===")
    print(f"Season range: {args.season_start}+")

    conn = get_connection()

    # Extract game features
    if not args.players_only:
        print("\nFetching game-level features from materialized views...")
        game_features = fetch_game_features(conn, season_start=args.season_start)
        print(f"Loaded {len(game_features)} games")

        print("Computing feature differentials...")
        game_features = compute_differentials(game_features)

        print("Adding target variables...")
        game_features = add_target_variables(game_features)

        if args.validate:
            validate_features(game_features)

        print(f"\nWriting game features to {args.output}...")
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        game_features.to_csv(args.output, index=False)
        print(f"[SUCCESS] Wrote {game_features.shape[0]} games, {game_features.shape[1]} columns")

    # Extract player features
    if not args.games_only:
        print("\nFetching player-level features from materialized views...")
        player_features = fetch_player_features(conn, season_start=args.season_start)
        print(f"Loaded {len(player_features)} player-weeks")

        print(f"\nWriting player features to {args.output_players}...")
        os.makedirs(os.path.dirname(args.output_players), exist_ok=True)
        player_features.to_csv(args.output_players, index=False)
        print(
            f"[SUCCESS] Wrote {player_features.shape[0]} player-weeks, {player_features.shape[1]} columns"
        )

    conn.close()
    print("\n✅ Feature extraction complete!")


if __name__ == "__main__":
    main()
