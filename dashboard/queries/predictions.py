"""
Prediction Queries

SQL queries for ml.predictions table and related player/game prediction analytics.

Author: Claude Code
Date: November 2025
"""

import logging

import pandas as pd
import psycopg

logger = logging.getLogger(__name__)

# Import schema validation
try:
    from dashboard.utils.schema_validator import (
        ValidationError,
        get_common_schemas,
        validate_schema,
    )

    SCHEMA_VALIDATION_AVAILABLE = True
except ImportError:
    SCHEMA_VALIDATION_AVAILABLE = False
    logger.warning("Schema validation not available - operating without validation")


class PredictionQueries:
    """Centralized queries for ml.predictions table and prediction analytics."""

    @staticmethod
    def retrospective_metrics(
        conn, model_version: str | None = None, season: int | None = None, limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch player passing yards predictions and actual performance with error handling.

        Args:
            conn: Database connection (psycopg)
            model_version: Optional model version filter (e.g., 'v5.1')
            season: Optional season filter (e.g., 2024)
            limit: Maximum rows to return (default: 1000)

        Returns:
            DataFrame with columns or empty DataFrame on error:
                - game_id, season, week, player_id, player_name, team, opponent, is_home
                - stat_type, predicted_value, actual_value, mae (abs_prediction_error)
                - status, pnl, edge, prob_win, calibrated
                - model_name, model_version, kickoff, spread_close, total_close
        """
        query = """
        SELECT
            p.game_id,
            p.season,
            p.week,
            p.player_id,
            p.player_name,
            p.team,
            p.opponent,
            p.is_home,
            p.stat_type,
            p.prediction_mean as predicted_value,
            p.actual_value,
            p.abs_prediction_error as mae,
            p.status,
            p.pnl,
            p.edge,
            p.prob_win,
            p.calibrated,
            mr.model_name,
            mr.version as model_version,
            g.kickoff,
            g.spread_close,
            g.total_close
        FROM ml.predictions p
        JOIN games g ON p.game_id = g.game_id
        JOIN ml.model_registry mr ON p.model_id = mr.model_id
        WHERE p.stat_type = 'passing_yards'
        """

        params = []
        if model_version:
            query += " AND mr.version = %s"
            params.append(model_version)
        if season:
            query += " AND p.season = %s"
            params.append(season)

        query += f" ORDER BY g.kickoff DESC LIMIT {limit}"

        try:
            df = pd.read_sql_query(query, conn, params=params if params else None)
            logger.info(f"Fetched {len(df)} retrospective metrics rows")
            return df

        except psycopg.Error as e:
            logger.error(f"Database error in retrospective_metrics: {e}", exc_info=True)
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Unexpected error in retrospective_metrics: {e}", exc_info=True)
            return pd.DataFrame()

    @staticmethod
    def weekly_performance_metrics(conn, min_season: int = 2022) -> pd.DataFrame:
        """
        Fetch weekly aggregated performance metrics for player predictions.

        Args:
            conn: Database connection
            min_season: Minimum season to include (default: 2022)

        Returns:
            DataFrame with columns:
                - season, week, total_player_games
                - mae (mean absolute error)
                - win_rate, calibration_rate
                - weekly_pnl (profit/loss sum)
        """
        query = f"""
        WITH weekly_metrics AS (
            SELECT
                p.season,
                p.week,
                COUNT(DISTINCT (p.player_id, p.game_id)) as total_player_games,
                ROUND(AVG(p.abs_prediction_error)::numeric, 2) as mae,
                ROUND(AVG(CASE WHEN p.status = 'win' THEN 1.0 ELSE 0.0 END)::numeric, 3) as win_rate,
                ROUND(AVG(CASE WHEN p.calibrated THEN 1.0 ELSE 0.0 END)::numeric, 3) as calibration_rate,
                ROUND(SUM(p.pnl)::numeric, 2) as weekly_pnl
            FROM ml.predictions p
            WHERE p.stat_type = 'passing_yards'
              AND p.actual_value IS NOT NULL
              AND p.season >= {min_season}
            GROUP BY p.season, p.week
            ORDER BY p.season, p.week
        )
        SELECT * FROM weekly_metrics
        """

        return pd.read_sql_query(query, conn)

    @staticmethod
    def bnn_uncertainty_metrics(conn, min_season: int = 2023, limit: int = 500) -> pd.DataFrame:
        """
        Fetch Bayesian Neural Network uncertainty metrics from predictions.

        Args:
            conn: Database connection
            min_season: Minimum season to include (default: 2023)
            limit: Maximum rows to return (default: 500)

        Returns:
            DataFrame with columns:
                - game_id, player_name, team, season, week
                - bnn_prediction, calibrated_prediction, ensemble_prediction
                - total_uncertainty (prediction_std)
                - ci_lower_5pct (q05), ci_upper_95pct (q95)
                - actual_value, mae, within_90pct_ci
        """
        query = f"""
        SELECT
            p.game_id,
            p.player_name,
            p.team,
            p.season,
            p.week,
            p.bnn_mean as bnn_prediction,
            p.bnn_calibrated_mean as calibrated_prediction,
            p.prediction_mean as ensemble_prediction,
            p.prediction_std as total_uncertainty,
            p.prediction_q05 as ci_lower_5pct,
            p.prediction_q95 as ci_upper_95pct,
            p.actual_value,
            p.abs_prediction_error as mae,
            CASE
                WHEN p.actual_value IS NOT NULL
                     AND p.actual_value >= p.prediction_q05
                     AND p.actual_value <= p.prediction_q95
                THEN true
                ELSE false
            END as within_90pct_ci
        FROM ml.predictions p
        WHERE p.stat_type = 'passing_yards'
          AND p.bnn_mean IS NOT NULL
          AND p.actual_value IS NOT NULL
          AND p.season >= {min_season}
        ORDER BY p.predicted_at DESC
        LIMIT {limit}
        """

        return pd.read_sql_query(query, conn)

    @staticmethod
    def predictions_with_odds(conn, season: int, week: int) -> pd.DataFrame:
        """
        Fetch QB passing predictions with latest sportsbook odds and edge analysis.

        Args:
            conn: Database connection
            season: Season to query
            week: Week to query

        Returns:
            DataFrame with columns:
                - player_id, player_name, team, opponent, is_home, game_id, kickoff
                - prediction_mean, prediction_std, prediction_q10, prediction_q90
                - ensemble_version, actual_value
                - book_line, over_odds, under_odds, over_implied_prob, under_implied_prob
                - book_hold, bookmaker_key, bookmaker_title, odds_updated_at
                - edge (prediction - line), prob_over, expected_value
        """
        query = """
        WITH latest_odds AS (
            SELECT DISTINCT ON (player_name, bookmaker_key)
                player_name,
                line_value,
                over_odds,
                under_odds,
                over_implied_prob,
                under_implied_prob,
                book_hold,
                bookmaker_key,
                bookmaker_title,
                snapshot_at
            FROM prop_lines_history
            WHERE prop_type = 'passing_yards'
              AND snapshot_at >= NOW() - INTERVAL '7 days'
            ORDER BY player_name, bookmaker_key, snapshot_at DESC
        )
        SELECT
            p.player_id,
            p.player_name,
            p.team,
            p.opponent,
            p.is_home,
            p.game_id,
            g.kickoff,
            p.prediction_mean,
            p.prediction_std,
            p.prediction_q10,
            p.prediction_q90,
            p.ensemble_version,
            p.actual_value,
            o.line_value as book_line,
            o.over_odds,
            o.under_odds,
            o.over_implied_prob,
            o.under_implied_prob,
            o.book_hold,
            o.bookmaker_key,
            o.bookmaker_title,
            o.snapshot_at as odds_updated_at,
            -- Calculate edge
            p.prediction_mean - o.line_value as edge,
            -- Probability our prediction is over the line
            CASE
                WHEN p.prediction_std > 0 THEN
                    1.0 - (0.5 * (1 + erf((o.line_value - p.prediction_mean) / (p.prediction_std * sqrt(2)))))
                ELSE 0.5
            END as prob_over,
            -- Expected value (simplified)
            CASE
                WHEN o.over_odds > 0 THEN
                    (1.0 - (0.5 * (1 + erf((o.line_value - p.prediction_mean) / (p.prediction_std * sqrt(2)))))) * (o.over_odds / 100.0) - (0.5 * (1 + erf((o.line_value - p.prediction_mean) / (p.prediction_std * sqrt(2))))) * 1.0
                ELSE
                    (1.0 - (0.5 * (1 + erf((o.line_value - p.prediction_mean) / (p.prediction_std * sqrt(2)))))) * (100.0 / abs(o.over_odds)) - (0.5 * (1 + erf((o.line_value - p.prediction_mean) / (p.prediction_std * sqrt(2))))) * 1.0
            END as expected_value
        FROM ml.predictions p
        JOIN games g ON p.game_id = g.game_id
        LEFT JOIN latest_odds o ON p.player_name = o.player_name
        WHERE p.season = %s
          AND p.week = %s
          AND p.stat_type = 'passing_yards'
          AND p.ensemble_version IS NOT NULL
        ORDER BY ABS(p.prediction_mean - o.line_value) DESC NULLS LAST
        """

        return pd.read_sql_query(query, conn, params=(season, week))

    @staticmethod
    def line_movement_history(conn, player_name: str, days_back: int = 7) -> pd.DataFrame:
        """
        Fetch historical line movement for a specific player.

        Args:
            conn: Database connection
            player_name: Player name to query
            days_back: Number of days to look back (default: 7)

        Returns:
            DataFrame with columns:
                - snapshot_at, line_value, over_odds, under_odds, bookmaker_title
        """
        query = f"""
        SELECT
            snapshot_at,
            line_value,
            over_odds,
            under_odds,
            bookmaker_title
        FROM prop_lines_history
        WHERE player_name = %s
          AND prop_type = 'passing_yards'
          AND snapshot_at >= NOW() - INTERVAL '{days_back} days'
        ORDER BY snapshot_at ASC
        """

        return pd.read_sql_query(query, conn, params=(player_name,))

    @staticmethod
    def model_comparison(conn) -> pd.DataFrame:
        """
        Compare different model versions for both player and game predictions.

        Returns:
            DataFrame with columns:
                - model_name, model_version, model_status, prediction_type
                - total_predictions, mae, std_error
                - win_rate, calibration_rate, total_pnl, avg_edge
        """
        query = """
        WITH player_metrics AS (
            SELECT
                mr.model_id,
                mr.model_name,
                mr.version as model_version,
                mr.status as model_status,
                'player' as prediction_type,
                COUNT(p.prediction_id) as total_predictions,
                ROUND(AVG(p.abs_prediction_error)::numeric, 2) as mae,
                ROUND(STDDEV(p.prediction_error)::numeric, 2) as std_error,
                ROUND(AVG(CASE WHEN p.status = 'win' THEN 1.0 ELSE 0.0 END)::numeric, 3) as win_rate,
                ROUND(AVG(CASE WHEN p.calibrated THEN 1.0 ELSE 0.0 END)::numeric, 3) as calibration_rate,
                ROUND(SUM(p.pnl)::numeric, 2) as total_pnl,
                ROUND(AVG(p.edge)::numeric, 4) as avg_edge
            FROM ml.model_registry mr
            LEFT JOIN ml.predictions p ON mr.model_id = p.model_id AND p.stat_type = 'passing_yards' AND p.actual_value IS NOT NULL
            WHERE mr.status IN ('production', 'staging', 'active')
            GROUP BY mr.model_id, mr.model_name, mr.version, mr.status
        ),
        game_metrics AS (
            SELECT
                NULL::integer as model_id,
                gp.ensemble_version as model_name,
                gp.ensemble_version as model_version,
                'active' as model_status,
                'game' as prediction_type,
                COUNT(gp.prediction_id) as total_predictions,
                ROUND(AVG(gp.abs_spread_error)::numeric, 2) as mae,
                ROUND(STDDEV(gp.abs_spread_error)::numeric, 2) as std_error,
                ROUND(AVG(CASE
                    WHEN gp.predicted_spread > 0 AND gp.actual_home_score - gp.actual_away_score > gp.sportsbook_spread THEN 1.0
                    WHEN gp.predicted_spread < 0 AND gp.actual_home_score - gp.actual_away_score < gp.sportsbook_spread THEN 1.0
                    WHEN gp.actual_home_score IS NULL THEN NULL
                    ELSE 0.0
                END)::numeric, 3) as win_rate,
                ROUND(AVG(CASE WHEN gp.brier_score < 0.25 THEN 1.0 ELSE 0.0 END)::numeric, 3) as calibration_rate,
                ROUND(SUM(CASE
                    WHEN gp.predicted_spread > 0 AND gp.actual_home_score - gp.actual_away_score > gp.sportsbook_spread THEN 100
                    WHEN gp.predicted_spread < 0 AND gp.actual_home_score - gp.actual_away_score < gp.sportsbook_spread THEN 100
                    WHEN gp.actual_home_score IS NULL THEN 0
                    ELSE -110
                END)::numeric, 2) as total_pnl,
                ROUND(AVG(gp.spread_edge)::numeric, 4) as avg_edge
            FROM ml.game_predictions gp
            WHERE gp.actual_home_score IS NOT NULL
            GROUP BY gp.ensemble_version
        ),
        combined_metrics AS (
            SELECT * FROM player_metrics
            UNION ALL
            SELECT * FROM game_metrics
        )
        SELECT
            model_name,
            model_version,
            model_status,
            prediction_type,
            total_predictions,
            mae,
            std_error,
            win_rate,
            calibration_rate,
            total_pnl,
            avg_edge
        FROM combined_metrics
        WHERE total_predictions > 0
        ORDER BY prediction_type DESC, mae ASC NULLS LAST
        """

        return pd.read_sql_query(query, conn)
