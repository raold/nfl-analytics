#!/usr/bin/env python3
"""
Generate game-specific Bayesian predictions for player props.

This script:
1. Loads player ratings from the Bayesian model
2. Adjusts for game-specific context (opponent, weather, spread, etc.)
3. Saves predictions to predictions.bayesian_prop_predictions table
"""

import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import psycopg

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database connection
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "dbname": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",
}

# Confidence interval inflation factor (2.5x for better coverage)
CI_INFLATION = 2.5


def get_upcoming_games(conn, season: int, week: int) -> pd.DataFrame:
    """Get games for the specified week."""
    query = """
    SELECT
        g.game_id,
        g.season,
        g.week,
        g.home_team,
        g.away_team,
        g.kickoff,
        g.spread_close as spread,
        g.total_close as total,
        g.temp,
        g.wind,
        g.stadium_id,
        g.roof
    FROM games g
    WHERE g.season = %s AND g.week = %s
    ORDER BY g.kickoff
    """

    return pd.read_sql_query(query, conn, params=(season, week))


def get_player_matchups(conn, games_df: pd.DataFrame) -> pd.DataFrame:
    """Get player matchups for the games."""
    game_ids = games_df["game_id"].tolist()

    query = """
    WITH active_players AS (
        -- Get players who have played recently
        SELECT DISTINCT
            rw.gsis_id as player_id,
            rw.player_name,
            rw.position,
            rw.team,
            MAX(rw.season) as latest_season,
            MAX(rw.week) as latest_week
        FROM rosters_weekly rw
        WHERE rw.season >= 2024
            AND rw.status = 'ACT'
            AND rw.position IN ('QB', 'RB', 'WR', 'TE')
        GROUP BY rw.gsis_id, rw.player_name, rw.position, rw.team
        HAVING MAX(rw.season * 100 + rw.week) >= 202401  -- Active in 2024
    )
    SELECT
        g.game_id,
        g.home_team,
        g.away_team,
        g.kickoff,
        ap.player_id,
        ap.player_name,
        ap.position,
        ap.team as player_team,
        CASE
            WHEN ap.team = g.home_team THEN g.away_team
            ELSE g.home_team
        END as opponent,
        CASE
            WHEN ap.team = g.home_team THEN 1
            ELSE 0
        END as is_home
    FROM games g
    JOIN active_players ap ON ap.team IN (g.home_team, g.away_team)
    WHERE g.game_id = ANY(%s)
    """

    return pd.read_sql_query(query, conn, params=(game_ids,))


def get_bayesian_ratings(conn) -> pd.DataFrame:
    """Get the latest Bayesian player ratings."""
    query = """
    SELECT
        player_id,
        stat_type,
        rating_mean,
        rating_sd,
        rating_q05,
        rating_q50,
        rating_q95,
        model_version,
        n_games_observed
    FROM mart.bayesian_player_ratings
    WHERE model_version = 'hierarchical_v1.1'
        AND stat_type = 'passing_yards'
    """

    return pd.read_sql_query(query, conn)


def get_defensive_adjustments(conn, season: int) -> dict[str, float]:
    """Get defensive strength adjustments for each team."""
    query = """
    WITH def_stats AS (
        SELECT
            posteam as opponent,
            AVG(CASE
                WHEN pass = 1 THEN yards_gained
                ELSE NULL
            END) as pass_yards_allowed,
            AVG(CASE
                WHEN rush = 1 THEN yards_gained
                ELSE NULL
            END) as rush_yards_allowed
        FROM plays
        WHERE season = %s
            AND week <= 17
            AND play_type IN ('pass', 'run')
        GROUP BY posteam
    )
    SELECT
        opponent,
        pass_yards_allowed / NULLIF(AVG(pass_yards_allowed) OVER(), 0) as pass_def_factor,
        rush_yards_allowed / NULLIF(AVG(rush_yards_allowed) OVER(), 0) as rush_def_factor
    FROM def_stats
    """

    df = pd.read_sql_query(query, conn, params=(season,))

    # Create adjustment factors (1.0 = average, >1.0 = allows more, <1.0 = allows less)
    pass_adj = dict(zip(df["opponent"], df["pass_def_factor"].fillna(1.0)))
    rush_adj = dict(zip(df["opponent"], df["rush_def_factor"].fillna(1.0)))

    return {"pass": pass_adj, "rush": rush_adj}


def apply_game_context(
    base_prediction: float,
    base_sd: float,
    is_home: bool,
    is_favored: bool,
    spread: float,
    total: float,
    weather_factor: float,
    opponent_adj: float,
) -> tuple[float, float]:
    """Apply game-specific adjustments to base prediction."""

    # Start with base prediction
    adjusted = base_prediction

    # Home field advantage (~3% boost)
    if is_home:
        adjusted *= 1.03

    # Favored teams tend to pass more when ahead
    if is_favored:
        adjusted *= 1.02

    # Game script adjustments
    # High totals = more passing
    if total and total > 45:
        adjusted *= 1.02
    elif total and total < 40:
        adjusted *= 0.98

    # Large spreads affect game script
    if spread:
        spread_abs = abs(spread)
        if spread_abs > 7:
            # Blowouts lead to less passing for favorites late
            if is_favored:
                adjusted *= 0.98
            else:
                # Trailing teams pass more
                adjusted *= 1.05

    # Weather adjustments
    adjusted *= weather_factor

    # Opponent defense adjustment
    adjusted *= opponent_adj

    # Adjust uncertainty based on context
    # More uncertainty in bad weather or extreme game scripts
    context_uncertainty = base_sd
    if weather_factor < 0.9:
        context_uncertainty *= 1.2
    if spread and abs(spread) > 10:
        context_uncertainty *= 1.1

    return adjusted, context_uncertainty


def generate_predictions(
    matchups_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    games_df: pd.DataFrame,
    def_adjustments: dict[str, dict[str, float]],
) -> list[dict]:
    """Generate game-specific predictions."""

    predictions = []

    for _, matchup in matchups_df.iterrows():
        # Get player's base rating
        player_rating = ratings_df[ratings_df["player_id"] == matchup["player_id"]]

        if player_rating.empty:
            # No rating for this player - skip
            continue

        rating = player_rating.iloc[0]

        # Get game context
        game = games_df[games_df["game_id"] == matchup["game_id"]].iloc[0]

        # Determine stat type based on position
        if matchup["position"] == "QB":
            stat_type = "passing_yards"
            def_type = "pass"
        elif matchup["position"] == "RB":
            stat_type = "rushing_yards"
            def_type = "rush"
        else:
            stat_type = "receiving_yards"
            def_type = "pass"

        # Skip if we don't have ratings for this stat type
        if rating["stat_type"] != stat_type:
            continue

        # Get defensive adjustment
        opponent_adj = def_adjustments[def_type].get(matchup["opponent"], 1.0)

        # Weather adjustment
        weather_factor = 1.0
        if game["wind"] and game["wind"] > 15:
            weather_factor *= 0.95
        if game["temp"] and game["temp"] < 32:
            weather_factor *= 0.97
        if game["roof"] in ["dome", "closed"]:
            weather_factor = 1.0  # No weather impact

        # Determine if player's team is favored
        is_favored = False
        spread = game["spread"]
        if spread:
            if matchup["is_home"] and spread < 0:
                is_favored = True
            elif not matchup["is_home"] and spread > 0:
                is_favored = True

        # Get base prediction (in log space)
        base_mean_log = rating["rating_mean"]
        base_sd_log = rating["rating_sd"] * CI_INFLATION  # Apply inflation

        # Convert to normal space for adjustments
        base_mean = np.exp(base_mean_log)

        # Apply game context adjustments
        adjusted_mean, adjusted_sd = apply_game_context(
            base_mean,
            base_sd_log * base_mean,  # Approximate SD in normal space
            matchup["is_home"],
            is_favored,
            spread,
            game["total"],
            weather_factor,
            opponent_adj,
        )

        # Convert back to log space for consistency
        adjusted_mean_log = np.log(adjusted_mean)
        adjusted_sd_log = adjusted_sd / adjusted_mean  # Approximate

        # Generate confidence intervals (with inflated uncertainty)
        q05 = np.exp(adjusted_mean_log - 1.645 * adjusted_sd_log)
        q95 = np.exp(adjusted_mean_log + 1.645 * adjusted_sd_log)

        prediction = {
            "game_id": matchup["game_id"],
            "player_id": matchup["player_id"],
            "stat_type": stat_type,
            "model_version": f"{rating['model_version']}_context",
            "rating_mean": adjusted_mean_log,
            "rating_sd": adjusted_sd_log,
            "rating_q05": np.log(q05),
            "rating_q50": adjusted_mean_log,
            "rating_q95": np.log(q95),
            "predicted_value": adjusted_mean,
            "predicted_q05": q05,
            "predicted_q95": q95,
            "league_intercept": 0.0,
            "position_group_effect": 0.0,
            "team_effect": 0.0,
            "vs_opponent_effect": np.log(opponent_adj),
            "player_effect": base_mean_log,
            "log_attempts_adjustment": 0.0,
            "home_field_adjustment": 0.03 if matchup["is_home"] else 0.0,
            "favorite_adjustment": 0.02 if is_favored else 0.0,
            "weather_adjustment": np.log(weather_factor),
            "total_line_adjustment": 0.0,
            "spread_adjustment": 0.0,
            "n_posterior_draws": 4000,
            "effective_sample_size": 1000.0,
            "rhat": 1.0,
            "game_context": {
                "is_home": bool(matchup["is_home"]),
                "opponent": matchup["opponent"],
                "spread": float(spread) if spread else None,
                "total": float(game["total"]) if game["total"] else None,
                "weather_factor": weather_factor,
            },
            "game_kickoff": game["kickoff"],
            "predicted_at": datetime.now(UTC),
        }

        predictions.append(prediction)

    return predictions


def save_predictions(conn, predictions: list[dict]) -> int:
    """Save predictions to database."""

    if not predictions:
        logger.warning("No predictions to save")
        return 0

    cursor = conn.cursor()

    # Clear old predictions for these games
    game_ids = list(set(p["game_id"] for p in predictions))
    cursor.execute(
        """
        DELETE FROM predictions.bayesian_prop_predictions
        WHERE game_id = ANY(%s)
            AND model_version LIKE '%_context'
        """,
        (game_ids,),
    )

    inserted = 0
    for pred in predictions:
        try:
            cursor.execute(
                """
                INSERT INTO predictions.bayesian_prop_predictions (
                    game_id, player_id, stat_type, model_version,
                    rating_mean, rating_sd, rating_q05, rating_q50, rating_q95,
                    predicted_value, predicted_q05, predicted_q95,
                    league_intercept, position_group_effect, team_effect,
                    vs_opponent_effect, player_effect, log_attempts_adjustment,
                    home_field_adjustment, favorite_adjustment, weather_adjustment,
                    total_line_adjustment, spread_adjustment,
                    n_posterior_draws, effective_sample_size, rhat,
                    game_context, game_kickoff, predicted_at, updated_at
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, NOW()
                )
                """,
                (
                    pred["game_id"],
                    pred["player_id"],
                    pred["stat_type"],
                    pred["model_version"],
                    pred["rating_mean"],
                    pred["rating_sd"],
                    pred["rating_q05"],
                    pred["rating_q50"],
                    pred["rating_q95"],
                    pred["predicted_value"],
                    pred["predicted_q05"],
                    pred["predicted_q95"],
                    pred["league_intercept"],
                    pred["position_group_effect"],
                    pred["team_effect"],
                    pred["vs_opponent_effect"],
                    pred["player_effect"],
                    pred["log_attempts_adjustment"],
                    pred["home_field_adjustment"],
                    pred["favorite_adjustment"],
                    pred["weather_adjustment"],
                    pred["total_line_adjustment"],
                    pred["spread_adjustment"],
                    pred["n_posterior_draws"],
                    pred["effective_sample_size"],
                    pred["rhat"],
                    pred["game_context"],
                    pred["game_kickoff"],
                    pred["predicted_at"],
                ),
            )
            inserted += 1

            if inserted % 50 == 0:
                logger.info(f"Inserted {inserted}/{len(predictions)} predictions")

        except Exception as e:
            logger.error(f"Error inserting prediction: {e}")
            logger.error(f"Prediction data: {pred}")
            conn.rollback()
            raise

    conn.commit()
    return inserted


def main():
    """Generate predictions for upcoming games."""

    # For now, generate for Week 7 of 2025
    season = 2025
    week = 7

    logger.info(f"Generating Bayesian predictions for {season} Week {week}")

    with psycopg.connect(**DB_CONFIG) as conn:
        # Get upcoming games
        games_df = get_upcoming_games(conn, season, week)
        logger.info(f"Found {len(games_df)} games")

        if games_df.empty:
            logger.warning("No games found")
            return

        # Get player matchups
        matchups_df = get_player_matchups(conn, games_df)
        logger.info(f"Found {len(matchups_df)} player matchups")

        # Get Bayesian ratings
        ratings_df = get_bayesian_ratings(conn)
        logger.info(f"Found {len(ratings_df)} player ratings")

        if ratings_df.empty:
            logger.error("No Bayesian ratings found - run save_bayesian_ratings.py first")
            return

        # Get defensive adjustments
        def_adjustments = get_defensive_adjustments(conn, season)

        # Generate predictions
        predictions = generate_predictions(matchups_df, ratings_df, games_df, def_adjustments)
        logger.info(f"Generated {len(predictions)} predictions")

        # Save to database
        if predictions:
            inserted = save_predictions(conn, predictions)
            logger.info(f"✅ Saved {inserted} predictions to database")
        else:
            logger.warning("No predictions generated")


if __name__ == "__main__":
    main()
