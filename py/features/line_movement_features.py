"""
Line Movement Features for NFL Betting Analysis
================================================

Extract betting line movement metrics between opening and closing lines
to detect sharp money, market consensus, and line value opportunities.

**Features (12 total):**
| Feature | Formula | Purpose |
|---------|---------|---------|
| spread_movement | close - open | Raw line change in points |
| total_movement | close_total - open_total | Total line change in points |
| spread_movement_pct | movement / abs(open) | Normalized change relative to open |
| movement_direction | sign(movement) | -1=toward favorite, 0=none, +1=toward dog |
| sharp_indicator | abs(movement) > 1.5 pts | Professional betting action flag |
| reverse_line_move | movement vs public | Indicates sharp counter-public action |
| consensus_spread | AVG across books | Market agreement (consensus price) |
| spread_dispersion | STDDEV across books | Market uncertainty/disagreement |
| total_dispersion | STDDEV(total) across books | Total market uncertainty |
| books_count | COUNT DISTINCT books | Market depth indicator |
| opening_consensus | AVG(open) across books | Opening market consensus |
| closing_vs_opening_consensus | closing - opening_consensus | Movement from opening consensus |

**Temporal Safety:**
All features use ONLY lines available BEFORE the game starts.
Opening lines represent early-week consensus; closing lines represent final market.

**Sharp Money Indicators:**
- Large movements (>1.5 pts) typically indicate sharp/professional action
- Movement AGAINST public betting patterns = sharps fading public
- Low dispersion + large movement = strong sharp consensus

**Data Source:**
- odds_history table with 'opening' and 'closing' market_type records
- 57K opening + 60K closing = 117K total odds records (2020-2024)
- Multiple bookmakers per game for consensus calculations

Author: Claude Code
Date: December 8, 2025
Task: Phase 1 Line Movement Feature Implementation
Reference: /Users/dro/.claude/plans/unified-crafting-babbage.md
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Configuration constants
SHARP_MOVEMENT_THRESHOLD = 1.5  # Points movement indicating sharp action
MIN_BOOKS_FOR_CONSENSUS = 3  # Minimum books needed for reliable consensus


class LineMovementFeatures:
    """
    Extract line movement features comparing opening to closing lines.

    Workflow:
    1. Query odds_history for opening lines (market_type='opening')
    2. Query odds_history for closing lines (market_type='closing')
    3. Calculate movement, consensus, and dispersion metrics
    4. Return feature dictionary for game-level prediction

    Note: This module requires the historical odds backfill to be complete.
    Without opening/closing data, returns default neutral values.
    """

    def __init__(
        self,
        db_host: str = "localhost",
        db_port: int = 5544,
        db_name: str = "devdb01",
        db_user: str = "dro",
        db_password: Optional[str] = None,
    ):
        """
        Initialize line movement feature extractor.

        Args:
            db_host: PostgreSQL host
            db_port: PostgreSQL port
            db_name: Database name
            db_user: Database user
            db_password: Database password (defaults to env var or hardcoded)
        """
        self.db_config = {
            "host": db_host,
            "port": db_port,
            "database": db_name,
            "user": db_user,
            "password": db_password or os.getenv("DB_PASSWORD", "sicillionbillions"),
        }

    def _get_db_connection(self):
        """Get database connection."""
        return psycopg2.connect(**self.db_config)

    def extract(
        self,
        game_id: str,
    ) -> dict[str, float]:
        """
        Extract line movement features for a specific game.

        Args:
            game_id: Game identifier (e.g., '2024_12_KC_BUF')

        Returns:
            Dictionary with 12 line movement features:
            - spread_movement (continuous): Change from open to close
            - total_movement (continuous): Total line change
            - spread_movement_pct (continuous): Percentage movement
            - movement_direction (categorical): -1/0/+1
            - sharp_indicator (binary): 1 if sharp movement detected
            - reverse_line_move (binary): 1 if movement against public
            - consensus_spread (continuous): Average closing spread
            - spread_dispersion (continuous): Closing spread std dev
            - total_dispersion (continuous): Closing total std dev
            - books_count (discrete): Number of bookmakers
            - opening_consensus (continuous): Average opening spread
            - closing_vs_opening_consensus (continuous): Movement from open consensus
        """
        conn = self._get_db_connection()

        try:
            # Check if odds_history table exists and has data
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = 'odds_history'
                    );
                """)
                table_exists = cur.fetchone()[0]

            if not table_exists:
                logger.debug("odds_history table not found, returning defaults")
                return self._get_default_features()

            # Query opening and closing lines for this game
            query = """
            WITH opening_lines AS (
                SELECT
                    bookmaker,
                    home_spread as open_spread,
                    over_under as open_total
                FROM odds_history
                WHERE game_id = %s
                  AND market_type = 'opening'
                  AND home_spread IS NOT NULL
            ),
            closing_lines AS (
                SELECT
                    bookmaker,
                    home_spread as close_spread,
                    over_under as close_total
                FROM odds_history
                WHERE game_id = %s
                  AND market_type = 'closing'
                  AND home_spread IS NOT NULL
            ),
            matched_lines AS (
                SELECT
                    o.bookmaker,
                    o.open_spread,
                    c.close_spread,
                    o.open_total,
                    c.close_total,
                    c.close_spread - o.open_spread as spread_movement,
                    c.close_total - o.open_total as total_movement
                FROM opening_lines o
                JOIN closing_lines c ON o.bookmaker = c.bookmaker
            )
            SELECT
                -- Opening consensus
                AVG(open_spread) as opening_consensus,
                STDDEV(open_spread) as opening_dispersion,
                -- Closing consensus
                AVG(close_spread) as closing_consensus,
                STDDEV(close_spread) as closing_dispersion,
                -- Total consensus
                AVG(close_total) as closing_total_consensus,
                STDDEV(close_total) as total_dispersion,
                -- Movement
                AVG(spread_movement) as avg_spread_movement,
                AVG(total_movement) as avg_total_movement,
                -- Market depth
                COUNT(DISTINCT bookmaker) as books_count
            FROM matched_lines;
            """

            df = pd.read_sql_query(query, conn, params=(game_id, game_id))

            if len(df) == 0 or pd.isna(df["closing_consensus"].iloc[0]):
                logger.debug(f"No line movement data for {game_id}")
                return self._get_default_features()

            # Extract raw values
            opening_consensus = float(df["opening_consensus"].iloc[0] or 0)
            closing_consensus = float(df["closing_consensus"].iloc[0] or 0)
            spread_dispersion = float(df["closing_dispersion"].iloc[0] or 0)
            total_dispersion = float(df["total_dispersion"].iloc[0] or 0)
            spread_movement = float(df["avg_spread_movement"].iloc[0] or 0)
            total_movement = float(df["avg_total_movement"].iloc[0] or 0)
            books_count = int(df["books_count"].iloc[0] or 0)

            # Calculate derived features
            # Movement percentage (relative to opening line magnitude)
            if abs(opening_consensus) > 0.5:
                spread_movement_pct = spread_movement / abs(opening_consensus)
            else:
                spread_movement_pct = 0.0

            # Movement direction: +1 = moved toward underdog, -1 = toward favorite, 0 = no move
            if abs(spread_movement) < 0.25:
                movement_direction = 0
            elif spread_movement > 0:
                movement_direction = 1  # Moved toward underdog (line got bigger)
            else:
                movement_direction = -1  # Moved toward favorite (line got smaller)

            # Sharp indicator: significant movement suggests professional action
            sharp_indicator = 1.0 if abs(spread_movement) > SHARP_MOVEMENT_THRESHOLD else 0.0

            # Reverse line move: movement + low dispersion = strong signal
            # High confidence when books agree and move together
            reverse_line_move = 1.0 if (
                abs(spread_movement) > 1.0 and
                spread_dispersion < 0.5 and
                books_count >= MIN_BOOKS_FOR_CONSENSUS
            ) else 0.0

            # Consensus movement from opening
            closing_vs_opening = closing_consensus - opening_consensus

            features = {
                "spread_movement": spread_movement,
                "total_movement": total_movement,
                "spread_movement_pct": spread_movement_pct,
                "movement_direction": float(movement_direction),
                "sharp_indicator": sharp_indicator,
                "reverse_line_move": reverse_line_move,
                "consensus_spread": closing_consensus,
                "spread_dispersion": spread_dispersion,
                "total_dispersion": total_dispersion,
                "books_count": float(books_count),
                "opening_consensus": opening_consensus,
                "closing_vs_opening_consensus": closing_vs_opening,
            }

            logger.debug(
                f"Line movement for {game_id}: "
                f"movement={spread_movement:+.1f}, "
                f"sharp={sharp_indicator}, "
                f"books={books_count}"
            )

            return features

        except Exception as e:
            logger.warning(f"Line movement extraction failed for {game_id}: {e}")
            return self._get_default_features()

        finally:
            conn.close()

    def extract_by_teams(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: int,
    ) -> dict[str, float]:
        """
        Extract line movement features by team/week lookup.

        Convenience method that looks up game_id and delegates to extract().

        Args:
            home_team: Home team abbreviation (e.g., 'KC')
            away_team: Away team abbreviation (e.g., 'BUF')
            season: NFL season (e.g., 2024)
            week: Week number (1-18)

        Returns:
            Dictionary with 12 line movement features
        """
        conn = self._get_db_connection()

        try:
            # Find game_id from teams and week
            query = """
            SELECT game_id FROM games
            WHERE home_team = %s
              AND away_team = %s
              AND season = %s
              AND week = %s
            LIMIT 1;
            """

            with conn.cursor() as cur:
                cur.execute(query, (home_team, away_team, season, week))
                row = cur.fetchone()

            if not row:
                logger.debug(
                    f"No game found for {away_team} @ {home_team} "
                    f"({season} Week {week})"
                )
                return self._get_default_features()

            game_id = row[0]
            return self.extract(game_id)

        finally:
            conn.close()

    def _get_default_features(self) -> dict[str, float]:
        """
        Return default features when extraction fails.

        Assumes neutral/zero line movement (no sharp action detected).
        """
        return {
            "spread_movement": 0.0,
            "total_movement": 0.0,
            "spread_movement_pct": 0.0,
            "movement_direction": 0.0,
            "sharp_indicator": 0.0,
            "reverse_line_move": 0.0,
            "consensus_spread": 0.0,
            "spread_dispersion": 0.5,  # Moderate uncertainty default
            "total_dispersion": 0.5,
            "books_count": 0.0,
            "opening_consensus": 0.0,
            "closing_vs_opening_consensus": 0.0,
        }


# ============================================================================
# BATCH EXTRACTION (FOR TRAINING DATA)
# ============================================================================


def extract_line_movement_batch(
    game_ids: list[str],
    db_config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Batch extract line movement features for multiple games.

    Args:
        game_ids: List of game_id strings
        db_config: Optional database configuration

    Returns:
        DataFrame with game_id and 12 line movement features
    """
    if db_config is None:
        db_config = {
            "host": "localhost",
            "port": 5544,
            "database": "devdb01",
            "user": "dro",
            "password": os.getenv("DB_PASSWORD", "sicillionbillions"),
        }

    extractor = LineMovementFeatures(**db_config)

    results = []
    for game_id in game_ids:
        features = extractor.extract(game_id)
        features["game_id"] = game_id
        results.append(features)

    df = pd.DataFrame(results)

    # Reorder columns
    cols = ["game_id"] + [c for c in df.columns if c != "game_id"]
    df = df[cols]

    logger.info(f"Batch line movement: {len(df):,} games processed")

    return df


def extract_line_movement_for_season(
    season: int,
    db_config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Extract line movement features for all games in a season.

    Args:
        season: NFL season year
        db_config: Optional database configuration

    Returns:
        DataFrame with line movement features for all games in season
    """
    if db_config is None:
        db_config = {
            "host": "localhost",
            "port": 5544,
            "database": "devdb01",
            "user": "dro",
            "password": os.getenv("DB_PASSWORD", "sicillionbillions"),
        }

    conn = psycopg2.connect(**db_config)

    try:
        # Get all game_ids for season
        query = """
        SELECT game_id
        FROM games
        WHERE season = %s
        ORDER BY week, kickoff;
        """

        df = pd.read_sql_query(query, conn, params=(season,))
        game_ids = df["game_id"].tolist()

    finally:
        conn.close()

    logger.info(f"Extracting line movement for {len(game_ids)} games in {season}")

    return extract_line_movement_batch(game_ids, db_config)


# ============================================================================
# CLV (CLOSING LINE VALUE) CALCULATION
# ============================================================================


def calculate_clv(
    bet_spread: float,
    closing_spread: float,
    bet_on_home: bool = True,
) -> float:
    """
    Calculate Closing Line Value for a bet.

    CLV measures how much better (or worse) your bet price was compared to
    the closing line. Positive CLV = long-term profitable bets.

    Args:
        bet_spread: The spread when you placed the bet
        closing_spread: The final closing spread
        bet_on_home: Whether bet was on home team

    Returns:
        CLV in points (positive = got better price)

    Example:
        If you bet Home -3 and it closed at Home -3.5, you got 0.5 pts CLV.
        If you bet Away +3 and it closed at Away +3.5, you lost 0.5 pts CLV.
    """
    movement = closing_spread - bet_spread

    if bet_on_home:
        # Home bettor benefits when line moves toward dog (closing more negative)
        clv = -movement
    else:
        # Away bettor benefits when line moves toward favorite (closing more positive)
        clv = movement

    return clv


# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================


def main():
    """Test line movement feature extraction."""
    extractor = LineMovementFeatures()

    # Test with a specific game
    test_game_ids = [
        "2024_12_KC_BUF",  # Week 12 2024
        "2024_11_DAL_PHI",  # Week 11 2024
        "2023_01_NYJ_BUF",  # Week 1 2023
    ]

    print("=" * 70)
    print("Line Movement Feature Extraction Test")
    print("=" * 70)

    for game_id in test_game_ids:
        print(f"\n{game_id}:")
        features = extractor.extract(game_id)

        for key, value in features.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

    # Test batch extraction
    print("\n" + "=" * 70)
    print("Batch Extraction Test (2024 Season Sample)")
    print("=" * 70)

    # Get sample of 2024 games
    conn = psycopg2.connect(
        host="localhost",
        port=5544,
        database="devdb01",
        user="dro",
        password=os.getenv("DB_PASSWORD", "sicillionbillions"),
    )

    try:
        # Use dynamic season: current year if after September, else previous year
        from datetime import datetime
        current_month = datetime.now().month
        current_year = datetime.now().year
        sample_season = current_year if current_month >= 9 else current_year - 1

        sample_query = """
        SELECT game_id FROM games
        WHERE season = %s AND week <= 5
        ORDER BY week, kickoff
        LIMIT 20;
        """
        sample_df = pd.read_sql_query(sample_query, conn, params=(sample_season,))
        sample_ids = sample_df["game_id"].tolist()
    finally:
        conn.close()

    if sample_ids:
        batch_df = extract_line_movement_batch(sample_ids)
        print(f"\nSample batch results ({len(batch_df)} games):")
        print(batch_df[["game_id", "spread_movement", "sharp_indicator", "books_count"]].head(10))

        # Summary statistics
        print("\n" + "=" * 70)
        print("Line Movement Statistics (2024 Weeks 1-5)")
        print("=" * 70)
        print(f"Average spread movement: {batch_df['spread_movement'].mean():.3f}")
        print(f"Std dev spread movement: {batch_df['spread_movement'].std():.3f}")
        print(f"Sharp indicator rate: {batch_df['sharp_indicator'].mean()*100:.1f}%")
        print(f"Average books per game: {batch_df['books_count'].mean():.1f}")
        print(f"Average spread dispersion: {batch_df['spread_dispersion'].mean():.3f}")

    return features


if __name__ == "__main__":
    main()
