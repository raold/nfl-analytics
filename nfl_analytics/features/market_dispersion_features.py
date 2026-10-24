"""
Market Dispersion Features for Meta-Learning Context
=====================================================

Calculate betting market dispersion metrics for context-aware ensemble weighting.

Features:
- market_dispersion: Standard deviation of closing spreads across bookmakers (0-3 points)
- high_dispersion_flag: Binary indicator for high uncertainty (dispersion > 1.5 points)
- books_count: Number of bookmakers with closing spread data

Meta-Learning Logic:
- High market dispersion (>1.5 points) → Bookmakers disagree → UP-weight XGBoost (most stable/consistent)
- Low market dispersion (<0.5 points) → Bookmakers agree → Balanced weights
- Moderate dispersion (0.5-1.5) → Standard allocation

Rationale:
When bookmakers disagree (high dispersion), the game is harder to predict → prefer
models with lowest variance (XGBoost v4 Ridge has tightest prediction intervals)

Data Source:
- odds_history table (closing spreads from last 2 hours before kickoff)
- Requires multiple sportsbooks per game (ideally 8-12)

Author: Enhanced Ensemble v7.0
Date: November 16, 2025
"""

import logging

import pandas as pd
import psycopg2

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

HIGH_DISPERSION_THRESHOLD = 1.5  # Standard deviation > 1.5 points
CLOSING_WINDOW_HOURS = 2  # Last 2 hours before kickoff


# ============================================================================
# MARKET DISPERSION EXTRACTOR
# ============================================================================


class MarketDispersionFeatures:
    """
    Extract betting market dispersion features for a game.

    Workflow:
    1. Query odds_history for closing spreads (last 2 hours)
    2. Calculate standard deviation across bookmakers
    3. Flag high dispersion games (std > 1.5 points)
    4. Count bookmakers for data quality assessment

    Note: Requires odds data integration (Phase 1-3 roadmap)
    For now, returns default values until odds API is integrated.
    """

    def __init__(
        self,
        db_host: str = "localhost",
        db_port: int = 5544,
        db_name: str = "devdb01",
        db_user: str = "dro",
        db_password: str = "sicillionbillions",
    ):
        """Initialize market dispersion feature extractor."""
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
        home_team: str,
        away_team: str,
        season: int,
        week: int,
    ) -> dict[str, float]:
        """
        Extract market dispersion features for a game.

        Args:
            home_team: Home team abbreviation (e.g., 'KC')
            away_team: Away team abbreviation (e.g., 'BUF')
            season: NFL season (e.g., 2024)
            week: Week number (1-18)

        Returns:
            Dictionary with 3 features:
            - market_dispersion (continuous, 0-3): Spread std dev across books
            - high_dispersion_flag (binary, 0/1): 1 if dispersion > 1.5
            - books_count (discrete, 0-20): Number of bookmakers
        """
        conn = self._get_db_connection()

        # Check if odds_history table exists
        check_table = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name = 'odds_history'
        );
        """

        cur = conn.cursor()
        cur.execute(check_table)
        table_exists = cur.fetchone()[0]
        cur.close()

        if not table_exists:
            logger.debug("Market dispersion: odds_history table not found (using defaults)")
            conn.close()
            return self._get_default_features()

        # Query closing spreads (last 2 hours before kickoff)
        query = """
        WITH game_context AS (
            -- Find game_id and kickoff time
            SELECT
                game_id,
                kickoff
            FROM games
            WHERE season = %s
              AND week = %s
              AND home_team = %s
              AND away_team = %s
            LIMIT 1
        ),
        closing_spreads AS (
            -- Get closing spreads (last 2 hours before kickoff)
            SELECT
                oh.bookmaker_key,
                oh.outcome_point as spread
            FROM odds_history oh
            JOIN game_context gc ON oh.event_id = gc.game_id::TEXT
            WHERE oh.market_key = 'spreads'
              AND oh.snapshot_at >= (gc.kickoff - INTERVAL '2 hours')
              AND oh.snapshot_at <= gc.kickoff
              AND oh.outcome_point IS NOT NULL
        )
        SELECT
            STDDEV(spread) as market_dispersion,
            COUNT(DISTINCT bookmaker_key) as books_count
        FROM closing_spreads;
        """

        try:
            df = pd.read_sql_query(query, conn, params=(season, week, home_team, away_team))
            conn.close()

            if len(df) == 0 or pd.isna(df["market_dispersion"].iloc[0]):
                logger.debug(f"Market dispersion: No odds data for {away_team} @ {home_team}")
                return self._get_default_features()

            # Extract features
            dispersion = float(df["market_dispersion"].iloc[0])
            books_count = int(df["books_count"].iloc[0])

            features = {
                "market_dispersion": dispersion,
                "high_dispersion_flag": 1.0 if dispersion > HIGH_DISPERSION_THRESHOLD else 0.0,
                "books_count": float(books_count),
            }

            logger.debug(
                f"Market dispersion: {away_team} @ {home_team} - "
                f"dispersion={dispersion:.2f}, books={books_count}"
            )

            return features

        except Exception as e:
            logger.warning(f"Market dispersion extraction failed: {e}")
            conn.close()
            return self._get_default_features()

    def _get_default_features(self) -> dict[str, float]:
        """
        Return default features when extraction fails.

        Assumes moderate market dispersion (0.75 points std dev).
        """
        return {
            "market_dispersion": 0.75,
            "high_dispersion_flag": 0.0,
            "books_count": 10.0,  # Assume ~10 books (typical)
        }


# ============================================================================
# MATERIALIZED VIEW CREATION (OPTIONAL OPTIMIZATION)
# ============================================================================


def create_market_dispersion_view(db_config: dict | None = None) -> None:
    """
    Create materialized view for market dispersion (speeds up queries).

    Run this once during Phase 2C setup, then refresh weekly.

    Usage:
        >>> create_market_dispersion_view()
        >>> # Refresh weekly: REFRESH MATERIALIZED VIEW mart.market_dispersion;
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

    # Check if odds_history exists
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name = 'odds_history'
        );
    """)
    table_exists = cur.fetchone()[0]

    if not table_exists:
        logger.warning("⚠️  odds_history table not found - skipping market dispersion view")
        conn.close()
        return

    # Drop if exists
    cur.execute("DROP MATERIALIZED VIEW IF EXISTS mart.market_dispersion CASCADE;")

    # Create materialized view
    create_view_sql = """
    CREATE MATERIALIZED VIEW mart.market_dispersion AS
    WITH closing_spreads AS (
        SELECT
            oh.event_id::INT as game_id,
            oh.bookmaker_key,
            oh.outcome_point as spread,
            oh.snapshot_at,
            g.kickoff
        FROM odds_history oh
        JOIN games g ON oh.event_id = g.game_id::TEXT
        WHERE oh.market_key = 'spreads'
          AND oh.snapshot_at >= (g.kickoff - INTERVAL '2 hours')
          AND oh.snapshot_at <= g.kickoff
          AND oh.outcome_point IS NOT NULL
    ),
    dispersion_by_game AS (
        SELECT
            game_id,
            STDDEV(spread) as market_dispersion,
            COUNT(DISTINCT bookmaker_key) as books_count,
            AVG(spread) as avg_closing_spread
        FROM closing_spreads
        GROUP BY game_id
    )
    SELECT
        g.game_id,
        g.season,
        g.week,
        g.home_team,
        g.away_team,
        g.kickoff,
        COALESCE(d.market_dispersion, 0.75) as market_dispersion,
        CASE
            WHEN COALESCE(d.market_dispersion, 0.75) > 1.5 THEN 1
            ELSE 0
        END as high_dispersion_flag,
        COALESCE(d.books_count, 10) as books_count,
        d.avg_closing_spread
    FROM games g
    LEFT JOIN dispersion_by_game d ON g.game_id = d.game_id
    ORDER BY g.season DESC, g.week DESC;
    """

    cur.execute(create_view_sql)
    conn.commit()

    # Create index for fast lookups
    cur.execute("""
        CREATE INDEX idx_market_dispersion_lookup
        ON mart.market_dispersion (season, week, home_team, away_team);
    """)
    conn.commit()

    logger.info("✅ Created materialized view: mart.market_dispersion")
    logger.info("   Refresh weekly with: REFRESH MATERIALIZED VIEW mart.market_dispersion;")

    conn.close()


# ============================================================================
# BATCH QUERY (FOR TRAINING DATA)
# ============================================================================


def query_market_dispersion_batch(
    game_ids: pd.DataFrame,
    db_config: dict | None = None,
) -> pd.DataFrame:
    """
    Batch extract market dispersion for multiple games.

    Args:
        game_ids: DataFrame with columns [game_id] or [season, week, home_team, away_team]
        db_config: Optional database configuration

    Returns:
        DataFrame with market dispersion features
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
    if "game_id" in game_ids.columns:
        # Use game_id directly
        create_temp = "CREATE TEMP TABLE input_games (game_id INT);"
        cur = conn.cursor()
        cur.execute(create_temp)

        for _, row in game_ids.iterrows():
            cur.execute("INSERT INTO input_games VALUES (%s)", (row["game_id"],))

        conn.commit()

        query = """
        SELECT
            ig.game_id,
            g.season,
            g.week,
            g.home_team,
            g.away_team,
            COALESCE(md.market_dispersion, 0.75) as market_dispersion,
            COALESCE(md.high_dispersion_flag, 0) as high_dispersion_flag,
            COALESCE(md.books_count, 10) as books_count
        FROM input_games ig
        JOIN games g ON ig.game_id = g.game_id
        LEFT JOIN mart.market_dispersion md ON md.game_id = ig.game_id
        ORDER BY g.season, g.week;
        """
    else:
        # Use season/week/teams
        create_temp = (
            "CREATE TEMP TABLE input_games (season INT, week INT, home_team TEXT, away_team TEXT);"
        )
        cur = conn.cursor()
        cur.execute(create_temp)

        for _, row in game_ids.iterrows():
            cur.execute(
                "INSERT INTO input_games VALUES (%s, %s, %s, %s)",
                (row["season"], row["week"], row["home_team"], row["away_team"]),
            )

        conn.commit()

        query = """
        SELECT
            ig.season,
            ig.week,
            ig.home_team,
            ig.away_team,
            COALESCE(md.market_dispersion, 0.75) as market_dispersion,
            COALESCE(md.high_dispersion_flag, 0) as high_dispersion_flag,
            COALESCE(md.books_count, 10) as books_count
        FROM input_games ig
        LEFT JOIN mart.market_dispersion md ON
            md.season = ig.season
            AND md.week = ig.week
            AND md.home_team = ig.home_team
            AND md.away_team = ig.away_team
        ORDER BY ig.season, ig.week;
        """

    df = pd.read_sql_query(query, conn)
    conn.close()

    logger.info(f"Batch market dispersion: {len(df):,} games processed")

    return df
