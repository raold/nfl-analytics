"""
Roster Stability Features for Meta-Learning Context
====================================================

Calculate team roster turnover and stability metrics for context-aware weighting.

Features:
- roster_stability_l3: 3-week rolling average roster retention percentage (0-1)
- roster_turnover_l3: 3-week rolling average roster turnover (complement of stability)

Meta-Learning Logic:
- High roster stability (>85%) + Veteran QB → UP-weight State-Space/Hierarchical (consistent patterns)
- Low roster stability (<70%) → DOWN-weight historical models, UP-weight XGBoost (adaptive to change)
- Moderate stability (70-85%) → Balanced weights

Roster Stability Calculation:
- Week-over-week player retention: retained_players / total_players
- Rolling 3-week average for smoothing

Author: Enhanced Ensemble v7.0
Date: November 16, 2025
"""

import logging

import pandas as pd
import psycopg2

logger = logging.getLogger(__name__)


# ============================================================================
# ROSTER STABILITY EXTRACTOR
# ============================================================================


class RosterStabilityFeatures:
    """
    Extract roster stability features for a team.

    Workflow:
    1. Identify team's active roster for current week
    2. Identify team's active roster for previous 3 weeks
    3. Calculate week-over-week retention percentage
    4. Compute 3-week rolling average stability

    Note: "Active roster" = players with game stats (passing/rushing/receiving)
    """

    def __init__(
        self,
        db_host: str = "localhost",
        db_port: int = 5544,
        db_name: str = "devdb01",
        db_user: str = "dro",
        db_password: str = "sicillionbillions",
    ):
        """Initialize roster stability feature extractor."""
        self.db_config = {
            "host": db_host,
            "port": db_port,
            "database": db_name,
            "user": db_user,
            "password": db_password,
        }

    def _get_db_connection(self):
        """Get database connection."""
        return psycopg2.connect(**self.db_config)

    def extract(
        self,
        team: str,
        season: int,
        week: int,
    ) -> dict[str, float]:
        """
        Extract roster stability features for a team-game.

        Args:
            team: Team abbreviation (e.g., 'KC', 'BUF')
            season: NFL season (e.g., 2024)
            week: Week number (1-18)

        Returns:
            Dictionary with 2 features:
            - roster_stability_l3 (continuous, 0-1): 3-week average retention
            - roster_turnover_l3 (continuous, 0-1): 3-week average turnover
        """
        conn = self._get_db_connection()

        query = """
        WITH weekly_rosters AS (
            -- Get active roster for each team-week (players with stats)
            SELECT DISTINCT
                g.season,
                g.week,
                ph.current_team as team,
                pgs.player_id
            FROM mart.player_game_stats pgs
            JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
            LEFT JOIN games g ON
                g.season = pgs.season
                AND g.week = pgs.week
                AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
            WHERE pgs.stat_category IN ('passing', 'rushing', 'receiving')
              AND g.season = %s
              AND ph.current_team = %s
              AND g.week BETWEEN GREATEST(1, %s - 3) AND %s  -- Current week + 3 weeks back
        ),
        roster_changes AS (
            -- Calculate week-over-week retention
            SELECT
                curr.season,
                curr.week,
                curr.team,
                COUNT(DISTINCT curr.player_id) as roster_size_current,
                COUNT(DISTINCT prev.player_id) as roster_size_prev,
                COUNT(DISTINCT CASE
                    WHEN prev.player_id IS NOT NULL THEN curr.player_id
                END) as players_retained,

                -- Stability = fraction of current roster that was on previous week's roster
                CAST(COUNT(DISTINCT CASE
                    WHEN prev.player_id IS NOT NULL THEN curr.player_id
                END) AS FLOAT) / NULLIF(COUNT(DISTINCT curr.player_id), 0) as weekly_stability

            FROM weekly_rosters curr
            LEFT JOIN weekly_rosters prev ON
                prev.team = curr.team
                AND prev.season = curr.season
                AND prev.week = curr.week - 1
            GROUP BY curr.season, curr.week, curr.team
        ),
        rolling_stability AS (
            -- Compute 3-week rolling average
            SELECT
                season,
                week,
                team,
                weekly_stability,
                AVG(weekly_stability) OVER (
                    PARTITION BY team, season
                    ORDER BY week
                    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ) as roster_stability_l3
            FROM roster_changes
        )
        SELECT
            roster_stability_l3,
            1.0 - COALESCE(roster_stability_l3, 0.85) as roster_turnover_l3
        FROM rolling_stability
        WHERE season = %s
          AND week = %s
          AND team = %s
        LIMIT 1;
        """

        try:
            df = pd.read_sql_query(
                query, conn, params=(season, team, week, week, season, week, team)
            )
            conn.close()

            if len(df) == 0 or pd.isna(df["roster_stability_l3"].iloc[0]):
                logger.debug(
                    f"Roster stability: Team {team} W{week} S{season} - using default (0.85)"
                )
                return self._get_default_features()

            # Extract features
            features = {
                "roster_stability_l3": float(df["roster_stability_l3"].iloc[0]),
                "roster_turnover_l3": float(df["roster_turnover_l3"].iloc[0]),
            }

            logger.debug(
                f"Roster stability: {team} W{week} S{season} - "
                f"stability={features['roster_stability_l3']:.2%}"
            )

            return features

        except Exception as e:
            logger.warning(f"Roster stability extraction failed: {e}")
            conn.close()
            return self._get_default_features()

    def _get_default_features(self) -> dict[str, float]:
        """
        Return default features when extraction fails.

        Assumes average roster stability (85% retention).
        """
        return {
            "roster_stability_l3": 0.85,
            "roster_turnover_l3": 0.15,
        }


# ============================================================================
# MATERIALIZED VIEW CREATION (OPTIONAL OPTIMIZATION)
# ============================================================================


def create_roster_stability_view(db_config: dict | None = None) -> None:
    """
    Create materialized view for roster stability (speeds up queries).

    Run this once during Phase 2C setup, then refresh weekly.

    Usage:
        >>> create_roster_stability_view()
        >>> # Refresh weekly: REFRESH MATERIALIZED VIEW mart.roster_stability;
    """
    if db_config is None:
        db_config = {
            "host": "localhost",
            "port": 5544,
            "database": "devdb01",
            "user": "dro",
            "password": "sicillionbillions",
        }

    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    # Drop if exists
    cur.execute("DROP MATERIALIZED VIEW IF EXISTS mart.roster_stability CASCADE;")

    # Create materialized view
    create_view_sql = """
    CREATE MATERIALIZED VIEW mart.roster_stability AS
    WITH weekly_rosters AS (
        SELECT DISTINCT
            g.season,
            g.week,
            ph.current_team as team,
            pgs.player_id
        FROM mart.player_game_stats pgs
        JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
        LEFT JOIN games g ON
            g.season = pgs.season
            AND g.week = pgs.week
            AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
        WHERE pgs.stat_category IN ('passing', 'rushing', 'receiving')
    ),
    roster_changes AS (
        SELECT
            curr.season,
            curr.week,
            curr.team,
            COUNT(DISTINCT curr.player_id) as roster_size_current,
            COUNT(DISTINCT prev.player_id) as roster_size_prev,
            COUNT(DISTINCT CASE
                WHEN prev.player_id IS NOT NULL THEN curr.player_id
            END) as players_retained,
            CAST(COUNT(DISTINCT CASE
                WHEN prev.player_id IS NOT NULL THEN curr.player_id
            END) AS FLOAT) / NULLIF(COUNT(DISTINCT curr.player_id), 0) as weekly_stability
        FROM weekly_rosters curr
        LEFT JOIN weekly_rosters prev ON
            prev.team = curr.team
            AND prev.season = curr.season
            AND prev.week = curr.week - 1
        GROUP BY curr.season, curr.week, curr.team
    )
    SELECT
        season,
        week,
        team,
        weekly_stability,
        AVG(weekly_stability) OVER (
            PARTITION BY team, season
            ORDER BY week
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) as roster_stability_l3,
        1.0 - AVG(weekly_stability) OVER (
            PARTITION BY team, season
            ORDER BY week
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) as roster_turnover_l3
    FROM roster_changes
    ORDER BY season DESC, week DESC, team;
    """

    cur.execute(create_view_sql)
    conn.commit()

    # Create index for fast lookups
    cur.execute(
        """
        CREATE INDEX idx_roster_stability_lookup
        ON mart.roster_stability (team, season, week);
    """
    )
    conn.commit()

    logger.info("✅ Created materialized view: mart.roster_stability")
    logger.info("   Refresh weekly with: REFRESH MATERIALIZED VIEW mart.roster_stability;")

    conn.close()


# ============================================================================
# BATCH QUERY (FOR TRAINING DATA)
# ============================================================================


def query_roster_stability_batch(
    team_season_weeks: pd.DataFrame,
    db_config: dict | None = None,
) -> pd.DataFrame:
    """
    Batch extract roster stability for multiple team-games.

    Args:
        team_season_weeks: DataFrame with columns [team, season, week]
        db_config: Optional database configuration

    Returns:
        DataFrame with roster stability features
    """
    if db_config is None:
        db_config = {
            "host": "localhost",
            "port": 5544,
            "database": "devdb01",
            "user": "dro",
            "password": "sicillionbillions",
        }

    conn = psycopg2.connect(**db_config)

    # Create temp table
    create_temp = """
    CREATE TEMP TABLE input_team_games (
        team TEXT,
        season INT,
        week INT
    );
    """

    cur = conn.cursor()
    cur.execute(create_temp)

    # Insert input data
    for _, row in team_season_weeks.iterrows():
        cur.execute(
            "INSERT INTO input_team_games VALUES (%s, %s, %s)",
            (row["team"], row["season"], row["week"]),
        )

    conn.commit()

    # Batch query (use materialized view if available, else compute)
    query = """
    -- Try materialized view first
    SELECT
        itg.team,
        itg.season,
        itg.week,
        COALESCE(rs.roster_stability_l3, 0.85) as roster_stability_l3,
        COALESCE(rs.roster_turnover_l3, 0.15) as roster_turnover_l3
    FROM input_team_games itg
    LEFT JOIN mart.roster_stability rs ON
        rs.team = itg.team
        AND rs.season = itg.season
        AND rs.week = itg.week
    ORDER BY itg.season, itg.week;
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    logger.info(f"Batch roster stability: {len(df):,} team-games processed")

    return df
