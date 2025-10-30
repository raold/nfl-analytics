"""
Vegas Line Bias Feature Engineering (Phase 3)

Measures how teams perform relative to Vegas expectations and betting market signals.
Part of the feature engineering pipeline for predictive modeling.

This module quantifies Vegas line characteristics and team performance vs expectations:
- Key number proximity (3, 7, 10, 14 point spreads)
- Implied probability from moneyline
- Favorite/underdog status and magnitude
- Historical line performance (beat/miss spread)
- Total (over/under) positioning
- Market efficiency signals from odds

Author: NFL Analytics System
Date: 2025-10-30
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
from sqlalchemy import create_engine
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VegasLineBiasEngine:
    """
    Engineer Vegas line bias and market expectation features.

    Tracks:
    - Key number proximity (important spread values: 3, 7, 10, 14)
    - Implied win probability from moneyline
    - Favorite/underdog magnitude
    - Historical line performance (ATS record)
    - Total positioning (over/under trends)
    - Market efficiency indicators
    """

    # Key numbers in NFL betting (most common margins of victory)
    KEY_NUMBERS = [3, 7, 10, 14]

    def __init__(
        self,
        db_url: str = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"
    ):
        """Initialize the Vegas line bias engine."""
        self.engine = create_engine(db_url)
        logger.info(f"Connected to database: {db_url}")

    def load_games_with_lines(
        self,
        start_season: int = 2024,
        end_season: int = 2025
    ) -> pd.DataFrame:
        """
        Load games with Vegas betting lines and results.

        Returns game-level data with spread, total, moneyline, and actual results.
        """
        logger.info(f"Loading games with Vegas lines ({start_season}-{end_season})...")

        query = f"""
        SELECT
            g.game_id,
            g.season,
            g.week,
            g.kickoff as game_date,
            g.home_team,
            g.away_team,
            g.home_score,
            g.away_score,
            g.spread_close,
            g.total_close,
            g.home_moneyline,
            g.away_moneyline,
            g.home_spread_odds,
            g.away_spread_odds,
            g.over_odds,
            g.under_odds
        FROM games g
        WHERE g.spread_close IS NOT NULL
            AND g.total_close IS NOT NULL
            AND g.season >= {start_season}
            AND g.season <= {end_season}
        ORDER BY g.kickoff
        """

        df = pd.read_sql(query, self.engine)
        logger.info(f"Loaded {len(df):,} games with Vegas lines")

        # Add derived metrics
        self._add_game_outcomes(df)
        self._add_implied_probabilities(df)
        self._add_key_number_features(df)

        return df

    def _add_game_outcomes(self, df: pd.DataFrame) -> None:
        """
        Add actual game outcomes and line performance.

        Calculates:
        - Final margin (home_score - away_score)
        - Whether home team covered spread
        - Whether total went over
        """
        # Final margin (positive = home won)
        df["final_margin"] = df["home_score"] - df["away_score"]

        # Spread is from home team perspective (negative = home favored)
        # Home covers if: final_margin + spread_close > 0
        # Example: Home -7 (spread_close = -7), Home wins by 10 (final_margin = 10)
        #          10 + (-7) = 3 > 0, so home covered
        df["home_covered"] = (df["final_margin"] + df["spread_close"]) > 0
        df["away_covered"] = ~df["home_covered"]

        # Push (tie against spread)
        df["spread_push"] = (df["final_margin"] + df["spread_close"]) == 0

        # Total
        df["actual_total"] = df["home_score"] + df["away_score"]
        df["over_hit"] = df["actual_total"] > df["total_close"]
        df["under_hit"] = df["actual_total"] < df["total_close"]
        df["total_push"] = df["actual_total"] == df["total_close"]

        # Margin vs expectation
        df["margin_vs_spread"] = df["final_margin"] - (-df["spread_close"])
        df["total_vs_line"] = df["actual_total"] - df["total_close"]

    def _add_implied_probabilities(self, df: pd.DataFrame) -> None:
        """
        Convert moneyline odds to implied win probabilities.

        American odds to probability conversion:
        - Negative odds (favorites): |odds| / (|odds| + 100)
        - Positive odds (underdogs): 100 / (odds + 100)
        """
        def moneyline_to_prob(odds):
            """Convert American moneyline odds to implied probability."""
            if pd.isna(odds):
                return np.nan
            if odds < 0:
                return abs(odds) / (abs(odds) + 100)
            else:
                return 100 / (odds + 100)

        df["home_implied_prob"] = df["home_moneyline"].apply(moneyline_to_prob)
        df["away_implied_prob"] = df["away_moneyline"].apply(moneyline_to_prob)

        # Vig (overround) - should sum to > 1.0
        df["total_implied_prob"] = df["home_implied_prob"] + df["away_implied_prob"]
        df["market_vig"] = df["total_implied_prob"] - 1.0

    def _add_key_number_features(self, df: pd.DataFrame) -> None:
        """
        Add features related to key numbers (3, 7, 10, 14).

        These spreads are particularly important in NFL betting due to
        the scoring system (3 for FG, 7 for TD+PAT).
        """
        # Distance to nearest key number
        df["dist_to_nearest_key"] = df["spread_close"].apply(
            lambda x: min([abs(x - kn) for kn in self.KEY_NUMBERS])
        )

        # Is the spread exactly on a key number?
        df["on_key_number"] = df["spread_close"].apply(
            lambda x: abs(x) in self.KEY_NUMBERS
        )

        # Specific key numbers
        for kn in self.KEY_NUMBERS:
            df[f"on_key_{kn}"] = (df["spread_close"].abs() == kn)

    def calculate_team_line_performance(
        self,
        games: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate historical line performance for each team.

        Tracks:
        - Against the Spread (ATS) record
        - Over/under record
        - Average margin vs spread
        - Favorite/underdog record
        """
        logger.info("Calculating team line performance...")

        results = []

        # Process home team performance
        for game_id, row in games.iterrows():
            # Home team metrics
            home_metrics = {
                "game_id": row["game_id"],
                "team": row["home_team"],
                "opponent": row["away_team"],
                "location": "home",
                "is_favorite": row["spread_close"] < 0,
                "spread_line": row["spread_close"],
                "covered_spread": row["home_covered"],
                "margin_vs_spread": row["margin_vs_spread"],
                "total_line": row["total_close"],
                "over_hit": row["over_hit"],
                "actual_total": row["actual_total"],
                "implied_win_prob": row["home_implied_prob"],
                "actual_won": row["final_margin"] > 0,
                "on_key_number": row["on_key_number"],
                "dist_to_key": row["dist_to_nearest_key"],
            }
            results.append(home_metrics)

            # Away team metrics
            away_metrics = {
                "game_id": row["game_id"],
                "team": row["away_team"],
                "opponent": row["home_team"],
                "location": "away",
                "is_favorite": row["spread_close"] > 0,
                "spread_line": -row["spread_close"],  # Flip for away team perspective
                "covered_spread": row["away_covered"],
                "margin_vs_spread": -row["margin_vs_spread"],  # Flip
                "total_line": row["total_close"],
                "over_hit": row["over_hit"],
                "actual_total": row["actual_total"],
                "implied_win_prob": row["away_implied_prob"],
                "actual_won": row["final_margin"] < 0,
                "on_key_number": row["on_key_number"],
                "dist_to_key": row["dist_to_nearest_key"],
            }
            results.append(away_metrics)

        df = pd.DataFrame(results)

        # Add rolling performance metrics
        df = df.sort_values(["team", "game_id"])

        # 4-game rolling ATS record
        df["ats_pct_last4"] = (
            df.groupby("team")["covered_spread"]
            .transform(lambda x: x.rolling(4, min_periods=1).mean() * 100)
        )

        # 8-game rolling ATS record
        df["ats_pct_last8"] = (
            df.groupby("team")["covered_spread"]
            .transform(lambda x: x.rolling(8, min_periods=1).mean() * 100)
        )

        # Season-to-date ATS record
        df["ats_pct_season"] = (
            df.groupby(["team"])["covered_spread"]
            .transform(lambda x: x.expanding().mean() * 100)
        )

        # Over/under trends
        df["over_pct_last4"] = (
            df.groupby("team")["over_hit"]
            .transform(lambda x: x.rolling(4, min_periods=1).mean() * 100)
        )

        # Average margin vs spread (last 4 games)
        df["avg_margin_vs_spread_last4"] = (
            df.groupby("team")["margin_vs_spread"]
            .transform(lambda x: x.rolling(4, min_periods=1).mean())
        )

        # Favorite/underdog splits
        df["as_favorite_ats_pct"] = (
            df[df["is_favorite"]].groupby("team")["covered_spread"]
            .transform(lambda x: x.expanding().mean() * 100)
        )

        df["as_underdog_ats_pct"] = (
            df[~df["is_favorite"]].groupby("team")["covered_spread"]
            .transform(lambda x: x.expanding().mean() * 100)
        )

        logger.info(f"Generated {len(df):,} team-game line performance records")

        return df

    def calculate_market_efficiency_indicators(
        self,
        games: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate market efficiency and betting value indicators.

        Tracks:
        - Vig (market overround)
        - Odds deviation from standard -110
        - Probability vs outcome alignment
        """
        logger.info("Calculating market efficiency indicators...")

        results = []

        for game_id, row in games.iterrows():
            # Home team perspective
            home_metrics = {
                "game_id": row["game_id"],
                "team": row["home_team"],
                "market_vig": row["market_vig"],
                "spread_odds": row["home_spread_odds"],
                "spread_odds_deviation": row["home_spread_odds"] - (-110),  # vs standard
                "implied_prob": row["home_implied_prob"],
                "prob_vs_outcome": row["home_implied_prob"] - (1 if row["final_margin"] > 0 else 0),
            }
            results.append(home_metrics)

            # Away team perspective
            away_metrics = {
                "game_id": row["game_id"],
                "team": row["away_team"],
                "market_vig": row["market_vig"],
                "spread_odds": row["away_spread_odds"],
                "spread_odds_deviation": row["away_spread_odds"] - (-110),
                "implied_prob": row["away_implied_prob"],
                "prob_vs_outcome": row["away_implied_prob"] - (1 if row["final_margin"] < 0 else 0),
            }
            results.append(away_metrics)

        df = pd.DataFrame(results)
        logger.info(f"Generated {len(df):,} market efficiency records")

        return df

    def calculate_key_number_performance(
        self,
        games: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate performance when spread is on or near key numbers.

        Key numbers (3, 7) are critical in NFL betting.
        """
        logger.info("Calculating key number performance...")

        # Filter to games on key numbers
        key_number_games = games[games["on_key_number"]].copy()

        if len(key_number_games) == 0:
            logger.warning("No games found on key numbers")
            return pd.DataFrame()

        results = []

        for key_num in self.KEY_NUMBERS:
            key_games = games[games[f"on_key_{key_num}"]].copy()

            if len(key_games) == 0:
                continue

            # Favorites (negative spread) vs underdogs on this key number
            fav_ats = key_games[key_games["spread_close"] == -key_num]["home_covered"].mean()
            dog_ats = key_games[key_games["spread_close"] == key_num]["away_covered"].mean()

            metrics = {
                "key_number": key_num,
                "n_games": len(key_games),
                "favorite_ats_pct": fav_ats * 100 if not pd.isna(fav_ats) else 0,
                "underdog_ats_pct": dog_ats * 100 if not pd.isna(dog_ats) else 0,
            }
            results.append(metrics)

        df = pd.DataFrame(results)
        logger.info(f"Generated {len(df):,} key number performance records")

        return df

    def generate_all_features(
        self,
        start_season: int = 2024,
        end_season: int = 2025
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate all Vegas line bias features.

        Returns:
            Tuple of (team_line_performance, market_efficiency, key_number_performance)
        """
        logger.info("=" * 70)
        logger.info("Vegas Line Bias Feature Engineering")
        logger.info("=" * 70)

        # Load games with lines
        games = self.load_games_with_lines(start_season, end_season)

        # Calculate all features
        team_line_performance = self.calculate_team_line_performance(games)
        market_efficiency = self.calculate_market_efficiency_indicators(games)
        key_number_performance = self.calculate_key_number_performance(games)

        logger.info("=" * 70)
        logger.info("Feature generation complete!")
        logger.info(f"  Team line performance: {len(team_line_performance):,} records")
        logger.info(f"  Market efficiency: {len(market_efficiency):,} records")
        logger.info(f"  Key number performance: {len(key_number_performance):,} records")
        logger.info("=" * 70)

        return (
            team_line_performance,
            market_efficiency,
            key_number_performance
        )


def main():
    """Test the Vegas line bias engine."""
    engine = VegasLineBiasEngine()

    # Generate ALL features for 2024-2025 seasons
    (
        team_line_performance,
        market_efficiency,
        key_number_performance
    ) = engine.generate_all_features(
        start_season=2024,
        end_season=2025
    )

    # Display sample statistics
    print("\n" + "=" * 70)
    print("TEAM LINE PERFORMANCE SAMPLE")
    print("=" * 70)
    if len(team_line_performance) > 0:
        print(team_line_performance.head(15).to_string())
        print("\nTEAM ATS PERFORMANCE STATISTICS")
        print(team_line_performance[[
            "ats_pct_last4",
            "ats_pct_last8",
            "ats_pct_season",
            "avg_margin_vs_spread_last4"
        ]].describe())

    print("\n" + "=" * 70)
    print("MARKET EFFICIENCY SAMPLE")
    print("=" * 70)
    if len(market_efficiency) > 0:
        print(market_efficiency.head(10).to_string())
        print("\nMARKET EFFICIENCY STATISTICS")
        print(market_efficiency[[
            "market_vig",
            "spread_odds_deviation",
            "implied_prob"
        ]].describe())

    print("\n" + "=" * 70)
    print("KEY NUMBER PERFORMANCE")
    print("=" * 70)
    if len(key_number_performance) > 0:
        print(key_number_performance.to_string(index=False))

    print("\n" + "=" * 70)
    print("TOP 10 TEAMS ATS (Last 4 Games)")
    print("=" * 70)
    if len(team_line_performance) > 0:
        # Get most recent entry for each team
        latest = team_line_performance.sort_values("game_id").groupby("team").tail(1)
        top_ats = latest.nlargest(10, "ats_pct_last4")[[
            "team", "ats_pct_last4", "ats_pct_season", "avg_margin_vs_spread_last4"
        ]]
        print(top_ats.to_string(index=False))

    print("\n" + "=" * 70)
    print("FAVORITE vs UNDERDOG ATS PERFORMANCE")
    print("=" * 70)
    if len(team_line_performance) > 0:
        fav_ats = team_line_performance[team_line_performance["is_favorite"]]["covered_spread"].mean() * 100
        dog_ats = team_line_performance[~team_line_performance["is_favorite"]]["covered_spread"].mean() * 100
        print(f"Favorites ATS: {fav_ats:.1f}%")
        print(f"Underdogs ATS: {dog_ats:.1f}%")

    print("\n" + "=" * 70)
    print("OVER/UNDER TRENDS")
    print("=" * 70)
    if len(team_line_performance) > 0:
        over_pct = team_line_performance["over_hit"].mean() * 100
        under_pct = team_line_performance["~over_hit"].mean() * 100 if "~over_hit" in team_line_performance.columns else (100 - over_pct)
        print(f"Over hit rate: {over_pct:.1f}%")
        print(f"Under hit rate: {100 - over_pct:.1f}%")

    return (
        team_line_performance,
        market_efficiency,
        key_number_performance
    )


if __name__ == "__main__":
    main()
