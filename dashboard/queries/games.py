"""
Game Queries

SQL queries for games table and game-level analytics (scheduling, matchups, live scores).

Author: Claude Code
Date: November 2025
"""

from typing import Optional, List
import pandas as pd
import logging
import psycopg

logger = logging.getLogger(__name__)

# Import schema validation
try:
    from dashboard.utils.schema_validator import validate_schema, get_common_schemas, ValidationError
    SCHEMA_VALIDATION_AVAILABLE = True
except ImportError:
    SCHEMA_VALIDATION_AVAILABLE = False
    logger.warning("Schema validation not available - operating without validation")


class GameQueries:
    """Centralized queries for games table and game analytics."""

    @staticmethod
    def current_week_games(
        conn,
        season: int,
        week: int
    ) -> pd.DataFrame:
        """
        Fetch current/upcoming week games with predictions and live scores with error handling.

        Args:
            conn: Database connection
            season: Season to query
            week: Week to query

        Returns:
            DataFrame with columns or empty DataFrame on error:
                - game_id, season, week, game_type, home_team, away_team
                - home_score, away_score, spread_close, total_close, kickoff
                - game_status ('completed', 'in_progress', 'scheduled')
                - predicted_spread, predicted_total, predicted_home_win_prob
                - spread_edge, total_edge, predicted_home_score, predicted_away_score
        """
        query = """
        SELECT DISTINCT ON (g.game_id)
            g.game_id,
            g.season,
            g.week,
            g.game_type,
            g.home_team,
            g.away_team,
            g.home_score,
            g.away_score,
            g.spread_close,
            g.total_close,
            g.kickoff,
            CASE
                WHEN g.home_score IS NOT NULL THEN 'completed'
                WHEN g.kickoff < NOW() - INTERVAL '4 hours' AND g.home_score IS NULL THEN 'completed'
                WHEN g.kickoff < NOW() THEN 'in_progress'
                ELSE 'scheduled'
            END as game_status,
            gp.predicted_spread,
            gp.predicted_total,
            gp.predicted_home_win_prob,
            gp.spread_edge,
            gp.total_edge,
            gp.predicted_home_score,
            gp.predicted_away_score
        FROM games g
        LEFT JOIN ml.game_predictions gp ON g.game_id = gp.game_id
        WHERE g.season = %s AND g.week = %s AND g.game_type = 'REG'
        ORDER BY g.game_id, gp.created_at DESC, g.kickoff DESC
        LIMIT 20
        """

        try:
            df = pd.read_sql_query(query, conn, params=(season, week))

            # Schema validation
            if SCHEMA_VALIDATION_AVAILABLE and not df.empty:
                schemas = get_common_schemas()
                df = validate_schema(df, schemas['games'], strict=False)

            logger.info(f"Fetched {len(df)} games for {season} Week {week}")
            return df

        except psycopg.Error as e:
            logger.error(f"Database error in current_week_games: {e}", exc_info=True)
            return pd.DataFrame()

        except ValidationError as e:
            logger.warning(f"Schema validation failed in current_week_games: {e}")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Unexpected error in current_week_games: {e}", exc_info=True)
            return pd.DataFrame()

    @staticmethod
    def teams_without_games(
        conn,
        season: int,
        week: int
    ) -> List[str]:
        """
        Get list of teams with bye weeks (no games scheduled).

        Args:
            conn: Database connection
            season: Season to query
            week: Week to query

        Returns:
            List of team abbreviations (e.g., ['KC', 'SF'])
        """
        query = """
        WITH all_teams AS (
            SELECT DISTINCT home_team as team FROM games WHERE season = %s
            UNION
            SELECT DISTINCT away_team as team FROM games WHERE season = %s
        ),
        week_teams AS (
            SELECT DISTINCT home_team as team FROM games WHERE season = %s AND week = %s AND game_type = 'REG'
            UNION
            SELECT DISTINCT away_team as team FROM games WHERE season = %s AND week = %s AND game_type = 'REG'
        )
        SELECT t.team
        FROM all_teams t
        LEFT JOIN week_teams w ON t.team = w.team
        WHERE w.team IS NULL
        ORDER BY t.team
        """

        df = pd.read_sql_query(
            query,
            conn,
            params=(season, season, season, week, season, week)
        )

        return df['team'].tolist() if not df.empty else []

    @staticmethod
    def season_schedule(
        conn,
        season: int,
        game_type: str = 'REG'
    ) -> pd.DataFrame:
        """
        Fetch complete season schedule.

        Args:
            conn: Database connection
            season: Season to query
            game_type: Game type filter ('REG', 'WC', 'DIV', 'CONF', 'SB')

        Returns:
            DataFrame with columns:
                - game_id, season, week, game_type, home_team, away_team
                - kickoff, spread_close, total_close, home_score, away_score
        """
        query = """
        SELECT
            game_id,
            season,
            week,
            game_type,
            home_team,
            away_team,
            kickoff,
            spread_close,
            total_close,
            home_score,
            away_score
        FROM games
        WHERE season = %s AND game_type = %s
        ORDER BY week ASC, kickoff ASC
        """

        return pd.read_sql_query(query, conn, params=(season, game_type))

    @staticmethod
    def recent_games(
        conn,
        limit: int = 20,
        completed_only: bool = True
    ) -> pd.DataFrame:
        """
        Fetch most recent games.

        Args:
            conn: Database connection
            limit: Number of games to return
            completed_only: Only return games with final scores (default: True)

        Returns:
            DataFrame with columns:
                - game_id, season, week, kickoff, home_team, away_team
                - home_score, away_score, spread_close, total_close
        """
        where_clause = ""
        if completed_only:
            where_clause = " AND home_score IS NOT NULL"

        query = f"""
        SELECT
            game_id,
            season,
            week,
            kickoff,
            home_team,
            away_team,
            home_score,
            away_score,
            spread_close,
            total_close
        FROM games
        WHERE kickoff IS NOT NULL {where_clause}
        ORDER BY kickoff DESC
        LIMIT {limit}
        """

        return pd.read_sql_query(query, conn)

    @staticmethod
    def data_coverage_by_season_week(conn) -> pd.DataFrame:
        """
        Fetch data coverage breakdown by season and week.

        Returns:
            DataFrame with columns:
                - season, week, game_type, games (count)
                - latest_kickoff (most recent game in that week)
        """
        query = """
        SELECT
            season,
            week,
            game_type,
            COUNT(*) as games,
            MAX(kickoff) as latest_kickoff
        FROM games
        WHERE season >= 2024
        GROUP BY season, week, game_type
        ORDER BY season DESC, week DESC
        LIMIT 10
        """

        return pd.read_sql_query(query, conn)

    @staticmethod
    def earliest_latest_games(conn) -> dict:
        """
        Get earliest and latest games in database.

        Returns:
            Dict with keys:
                - earliest_kickoff, earliest_season, earliest_week
                - latest_kickoff, latest_season, latest_week
        """
        query = """
        WITH date_range AS (
            SELECT
                MIN(kickoff) as earliest_kickoff,
                MAX(kickoff) as latest_kickoff
            FROM games
            WHERE kickoff IS NOT NULL AND home_score IS NOT NULL
        ),
        earliest AS (
            SELECT season, week FROM games
            WHERE kickoff = (SELECT earliest_kickoff FROM date_range)
            LIMIT 1
        ),
        latest AS (
            SELECT season, week FROM games
            WHERE kickoff = (SELECT latest_kickoff FROM date_range)
            LIMIT 1
        )
        SELECT
            dr.earliest_kickoff,
            e.season as earliest_season,
            e.week as earliest_week,
            dr.latest_kickoff,
            l.season as latest_season,
            l.week as latest_week
        FROM date_range dr
        CROSS JOIN earliest e
        CROSS JOIN latest l
        """

        df = pd.read_sql_query(query, conn)

        if not df.empty:
            row = df.iloc[0]
            return {
                'earliest_kickoff': row['earliest_kickoff'],
                'earliest_season': row['earliest_season'],
                'earliest_week': row['earliest_week'],
                'latest_kickoff': row['latest_kickoff'],
                'latest_season': row['latest_season'],
                'latest_week': row['latest_week'],
            }

        return {}

    @staticmethod
    def games_by_team(
        conn,
        team: str,
        season: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch all games for a specific team.

        Args:
            conn: Database connection
            team: Team abbreviation (e.g., 'KC', 'LAR')
            season: Optional season filter

        Returns:
            DataFrame with columns:
                - game_id, season, week, kickoff, opponent, is_home
                - team_score, opponent_score, spread_close, total_close
        """
        where_clause = "WHERE (home_team = %s OR away_team = %s)"
        params = [team, team]

        if season:
            where_clause += " AND season = %s"
            params.append(season)

        query = f"""
        SELECT
            game_id,
            season,
            week,
            kickoff,
            CASE WHEN home_team = %s THEN away_team ELSE home_team END as opponent,
            CASE WHEN home_team = %s THEN true ELSE false END as is_home,
            CASE WHEN home_team = %s THEN home_score ELSE away_score END as team_score,
            CASE WHEN home_team = %s THEN away_score ELSE home_score END as opponent_score,
            spread_close,
            total_close
        FROM games
        {where_clause}
        ORDER BY kickoff ASC
        """

        params = [team, team, team, team] + params
        return pd.read_sql_query(query, conn, params=params)
