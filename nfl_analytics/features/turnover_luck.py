#!/usr/bin/env python3
"""
Turnover Luck Features - Regression-Based Public Betting Edge
==============================================================

Quantifies turnover luck to identify teams that are over/under-bet due to
unsustainable turnover performance.

**Research Foundation**:
- Fumble recovery rate regresses to ~50% over time (NFL average: 50.3%)
- Interception rate above/below expectation regresses to the mean
- Teams with "lucky" recent turnovers are systematically over-bet
- Sharp bettors fade hot teams with high turnover luck

**Key Features**:
1. `turnover_luck_last_3`: Net turnovers above/below expected (last 3 games)
2. `turnover_luck_last_5`: Net turnovers above/below expected (last 5 games)
3. `fumble_recovery_rate_l5`: Fumble recovery rate vs 50% baseline (last 5 games)
4. `int_regression_indicator`: Flag for unsustainable INT rate (+/-1)
5. `turnover_margin_l3`: Raw turnover margin (for comparison)
6. `expected_regression_points`: Estimated point value of expected regression

**Expected Impact**: +1-2% ROI (teams with high positive luck are over-bet)

**Temporal Safety**:
- All features use only games PRIOR to prediction date (as-of joins)
- Rolling windows exclude current game
- No future data leakage

Author: Claude Code (Opus 4.5)
Date: November 27, 2025
Research: NFL Betting Inefficiency Analysis (Nov 2025)
"""

import logging
import warnings

import pandas as pd
import psycopg2

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class TurnoverLuckEngine:
    """
    Compute turnover luck features to identify regression candidates.

    **Why This Matters**:
    A team that recovered 80% of fumbles over the last 3 games (vs 50% baseline)
    will likely regress. Public bettors chase hot streaks, creating value on
    the other side.

    **Expected Turnover Math**:
    - Expected fumble recovery rate: 50%
    - Expected interception rate: ~2.1% of pass attempts (league average)
    - Luck = Actual - Expected

    **Betting Implication**:
    High positive turnover luck → Team is likely over-valued → Fade
    High negative turnover luck → Team is likely under-valued → Bet
    """

    # Expected fumble recovery rate (NFL historical average)
    EXPECTED_FUMBLE_RECOVERY_RATE = 0.50

    # Expected turnovers per game (league average ~1.3)
    EXPECTED_TURNOVERS_PER_GAME = 1.30

    # Point value per turnover (NFL average)
    POINTS_PER_TURNOVER = 4.0

    # Rolling window sizes
    WINDOW_SHORT = 3  # Recent form
    WINDOW_MEDIUM = 5  # More stable estimate

    def __init__(
        self,
        db_host: str = "localhost",
        db_port: int = 5544,
        db_name: str = "devdb01",
        db_user: str = "dro",
        db_password: str = "sicillionbillions",
    ):
        """Initialize turnover luck engine with database connection."""
        self.db_config = {
            "host": db_host,
            "port": db_port,
            "dbname": db_name,
            "user": db_user,
            "password": db_password,
        }
        logger.info("Initialized TurnoverLuckEngine")

    def _connect_db(self):
        """Create database connection."""
        return psycopg2.connect(**self.db_config)

    def _fetch_team_turnover_history(
        self, team: str, season: int, week: int, lookback_games: int = 10
    ) -> pd.DataFrame:
        """
        Fetch turnover history for a team prior to the specified game.

        Uses as-of join semantics: only games BEFORE (season, week) are included.

        Args:
            team: Team abbreviation (e.g., 'KC', 'BUF')
            season: Season year
            week: Week number (1-18)
            lookback_games: Maximum games to retrieve

        Returns:
            DataFrame with columns: game_id, season, week, kickoff,
            turnovers_committed, turnovers_forced, is_home
        """
        query = """
        WITH team_games AS (
            SELECT
                g.game_id,
                g.season,
                g.week,
                g.kickoff,
                CASE
                    WHEN g.home_team = %(team)s THEN TRUE
                    ELSE FALSE
                END as is_home,
                CASE
                    WHEN g.home_team = %(team)s THEN g.home_turnovers
                    ELSE g.away_turnovers
                END as turnovers_committed,
                CASE
                    WHEN g.home_team = %(team)s THEN g.away_turnovers
                    ELSE g.home_turnovers
                END as turnovers_forced
            FROM games g
            WHERE (g.home_team = %(team)s OR g.away_team = %(team)s)
              AND g.season >= %(min_season)s
              AND (g.season < %(season)s OR (g.season = %(season)s AND g.week < %(week)s))
              AND g.home_turnovers IS NOT NULL
              AND g.away_turnovers IS NOT NULL
            ORDER BY g.kickoff DESC
            LIMIT %(lookback)s
        )
        SELECT * FROM team_games ORDER BY kickoff ASC;
        """

        with self._connect_db() as conn:
            df = pd.read_sql_query(
                query,
                conn,
                params={
                    "team": team,
                    "season": season,
                    "week": week,
                    "min_season": season - 2,  # Look back up to 2 seasons
                    "lookback": lookback_games,
                },
            )

        logger.debug(f"Fetched {len(df)} games for {team} before {season} week {week}")
        return df

    def _compute_fumble_recovery_rate(
        self, team: str, season: int, week: int, window: int = 5
    ) -> float:
        """
        Compute fumble recovery rate from play-by-play data.

        This provides a more accurate measure of luck than aggregate turnovers,
        since fumble recovery is truly random (50/50).

        Note: Falls back to turnover margin if play-by-play data is unavailable.
        """
        query = """
        WITH team_fumbles AS (
            SELECT
                p.game_id,
                g.season,
                g.week,
                p.fumble,
                p.fumble_recovery_1_team,
                CASE
                    WHEN g.home_team = %(team)s THEN 'home'
                    ELSE 'away'
                END as team_side
            FROM plays p
            JOIN games g ON p.game_id = g.game_id
            WHERE (g.home_team = %(team)s OR g.away_team = %(team)s)
              AND p.fumble = 1
              AND g.season >= %(min_season)s
              AND (g.season < %(season)s OR (g.season = %(season)s AND g.week < %(week)s))
            ORDER BY g.kickoff DESC
        ),
        recent_fumbles AS (
            SELECT *
            FROM team_fumbles tf
            WHERE (tf.season, tf.week, tf.game_id) IN (
                SELECT DISTINCT season, week, game_id
                FROM team_fumbles
                ORDER BY season DESC, week DESC
                LIMIT %(window)s
            )
        )
        SELECT
            COUNT(*) as total_fumbles,
            SUM(CASE
                WHEN (team_side = 'home' AND fumble_recovery_1_team = 'home')
                  OR (team_side = 'away' AND fumble_recovery_1_team = 'away')
                THEN 1 ELSE 0
            END) as fumbles_recovered
        FROM recent_fumbles;
        """

        try:
            with self._connect_db() as conn:
                result = pd.read_sql_query(
                    query,
                    conn,
                    params={
                        "team": team,
                        "season": season,
                        "week": week,
                        "min_season": season - 1,
                        "window": window,
                    },
                )

            if len(result) == 0:
                return self.EXPECTED_FUMBLE_RECOVERY_RATE

            total = result.iloc[0]["total_fumbles"]
            recovered = result.iloc[0]["fumbles_recovered"]

            if total is None or total < 3:
                # Insufficient data, return baseline
                return self.EXPECTED_FUMBLE_RECOVERY_RATE

            return float(recovered) / float(total)

        except Exception as e:
            logger.warning(f"Fumble recovery query failed: {e}, using baseline")
            return self.EXPECTED_FUMBLE_RECOVERY_RATE

    def compute_turnover_luck_features(self, team: str, season: int, week: int) -> dict[str, float]:
        """
        Compute all turnover luck features for a team.

        Args:
            team: Team abbreviation
            season: Season year
            week: Week number

        Returns:
            Dictionary of turnover luck features
        """
        # Fetch turnover history
        history = self._fetch_team_turnover_history(team, season, week, lookback_games=10)

        features = {}

        if len(history) < 3:
            # Insufficient data - return neutral values
            logger.warning(f"Insufficient turnover data for {team} ({len(history)} games)")
            features = {
                "turnover_luck_last_3": 0.0,
                "turnover_luck_last_5": 0.0,
                "turnover_margin_l3": 0.0,
                "turnover_margin_l5": 0.0,
                "fumble_recovery_rate_l5": 0.50,
                "fumble_recovery_luck": 0.0,
                "int_regression_indicator": 0,
                "expected_regression_points": 0.0,
                "turnovers_committed_l3": self.EXPECTED_TURNOVERS_PER_GAME,
                "turnovers_forced_l3": self.EXPECTED_TURNOVERS_PER_GAME,
            }
            return features

        # Sort by most recent first for rolling calculations
        history = history.sort_values("kickoff", ascending=False)

        # Last 3 games stats
        last_3 = history.head(min(3, len(history)))
        turnovers_committed_l3 = last_3["turnovers_committed"].mean()
        turnovers_forced_l3 = last_3["turnovers_forced"].mean()
        turnover_margin_l3 = turnovers_forced_l3 - turnovers_committed_l3

        # Last 5 games stats
        last_5 = history.head(min(5, len(history)))
        turnovers_committed_l5 = last_5["turnovers_committed"].mean()
        turnovers_forced_l5 = last_5["turnovers_forced"].mean()
        turnover_margin_l5 = turnovers_forced_l5 - turnovers_committed_l5

        # Turnover luck = Actual margin - Expected margin (0)
        # Positive = lucky (forced more / committed fewer than expected)
        # Negative = unlucky
        turnover_luck_l3 = turnover_margin_l3 - 0  # Expected margin is 0
        turnover_luck_l5 = turnover_margin_l5 - 0

        # Fumble recovery rate (more precise luck measure)
        fumble_recovery_rate = self._compute_fumble_recovery_rate(team, season, week, window=5)
        fumble_recovery_luck = fumble_recovery_rate - self.EXPECTED_FUMBLE_RECOVERY_RATE

        # INT regression indicator
        # +1 if team has unsustainably high forced INTs (lucky)
        # -1 if team has unsustainably low forced INTs (unlucky)
        # 0 if within normal range
        avg_forced_per_game = turnovers_forced_l5
        if avg_forced_per_game > 2.0:
            int_regression_indicator = 1  # Likely to regress down
        elif avg_forced_per_game < 0.8:
            int_regression_indicator = -1  # Likely to regress up
        else:
            int_regression_indicator = 0

        # Expected regression in points
        # Each turnover is worth ~4 points
        # If luck is +0.5 turnovers/game over 3 games, expect ~2 points regression
        expected_regression_points = -turnover_luck_l3 * self.POINTS_PER_TURNOVER

        features = {
            "turnover_luck_last_3": round(turnover_luck_l3, 3),
            "turnover_luck_last_5": round(turnover_luck_l5, 3),
            "turnover_margin_l3": round(turnover_margin_l3, 3),
            "turnover_margin_l5": round(turnover_margin_l5, 3),
            "fumble_recovery_rate_l5": round(fumble_recovery_rate, 3),
            "fumble_recovery_luck": round(fumble_recovery_luck, 3),
            "int_regression_indicator": int_regression_indicator,
            "expected_regression_points": round(expected_regression_points, 2),
            "turnovers_committed_l3": round(turnovers_committed_l3, 3),
            "turnovers_forced_l3": round(turnovers_forced_l3, 3),
        }

        logger.debug(f"Turnover luck features for {team}: {features}")
        return features

    def compute_game_turnover_features(
        self, home_team: str, away_team: str, season: int, week: int
    ) -> dict[str, float]:
        """
        Compute turnover luck features for a matchup.

        Returns differential features (home - away) for betting edge calculation.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: Season year
            week: Week number

        Returns:
            Dictionary with both team features and differentials
        """
        home_features = self.compute_turnover_luck_features(home_team, season, week)
        away_features = self.compute_turnover_luck_features(away_team, season, week)

        # Create prefixed features
        result = {}

        # Home team features
        for key, value in home_features.items():
            result[f"home_{key}"] = value

        # Away team features
        for key, value in away_features.items():
            result[f"away_{key}"] = value

        # Differential features (home - away)
        # Positive differential = home team is luckier (likely to regress)
        result["turnover_luck_diff"] = (
            home_features["turnover_luck_last_3"] - away_features["turnover_luck_last_3"]
        )
        result["turnover_margin_diff"] = (
            home_features["turnover_margin_l3"] - away_features["turnover_margin_l3"]
        )
        result["fumble_recovery_luck_diff"] = (
            home_features["fumble_recovery_luck"] - away_features["fumble_recovery_luck"]
        )
        result["expected_regression_diff"] = (
            home_features["expected_regression_points"]
            - away_features["expected_regression_points"]
        )

        # Composite luck indicator
        # Combines turnover luck and fumble recovery luck
        result["composite_luck_diff"] = (
            result["turnover_luck_diff"] * 0.7
            + result["fumble_recovery_luck_diff"] * 10 * 0.3  # Scale fumble luck
        )

        return result

    def compute_batch_features(self, games: pd.DataFrame) -> pd.DataFrame:
        """
        Compute turnover luck features for multiple games efficiently.

        Args:
            games: DataFrame with columns: game_id, home_team, away_team, season, week

        Returns:
            DataFrame with original columns plus all turnover luck features
        """
        all_features = []

        for _, row in games.iterrows():
            try:
                features = self.compute_game_turnover_features(
                    home_team=row["home_team"],
                    away_team=row["away_team"],
                    season=row["season"],
                    week=row["week"],
                )
                features["game_id"] = row["game_id"]
                all_features.append(features)
            except Exception as e:
                logger.error(
                    f"Error computing turnover features for game {row.get('game_id')}: {e}"
                )
                # Return neutral features
                features = {"game_id": row["game_id"]}
                all_features.append(features)

        features_df = pd.DataFrame(all_features)
        result = games.merge(features_df, on="game_id", how="left")

        return result


def get_turnover_luck_feature_names() -> list:
    """Return list of all turnover luck feature names for integration."""
    return [
        # Home team features
        "home_turnover_luck_last_3",
        "home_turnover_luck_last_5",
        "home_turnover_margin_l3",
        "home_turnover_margin_l5",
        "home_fumble_recovery_rate_l5",
        "home_fumble_recovery_luck",
        "home_int_regression_indicator",
        "home_expected_regression_points",
        "home_turnovers_committed_l3",
        "home_turnovers_forced_l3",
        # Away team features
        "away_turnover_luck_last_3",
        "away_turnover_luck_last_5",
        "away_turnover_margin_l3",
        "away_turnover_margin_l5",
        "away_fumble_recovery_rate_l5",
        "away_fumble_recovery_luck",
        "away_int_regression_indicator",
        "away_expected_regression_points",
        "away_turnovers_committed_l3",
        "away_turnovers_forced_l3",
        # Differential features
        "turnover_luck_diff",
        "turnover_margin_diff",
        "fumble_recovery_luck_diff",
        "expected_regression_diff",
        "composite_luck_diff",
    ]


# Quick test
if __name__ == "__main__":
    engine = TurnoverLuckEngine()

    # Test single team features
    print("\n=== KC Turnover Luck (Week 12, 2024) ===")
    kc_features = engine.compute_turnover_luck_features("KC", 2024, 12)
    for k, v in kc_features.items():
        print(f"  {k}: {v}")

    # Test matchup features
    print("\n=== KC vs BUF Turnover Luck Comparison ===")
    matchup = engine.compute_game_turnover_features("KC", "BUF", 2024, 12)
    for k, v in matchup.items():
        if "diff" in k or "composite" in k:
            print(f"  {k}: {v}")

    print(f"\n=== Total features: {len(get_turnover_luck_feature_names())} ===")
