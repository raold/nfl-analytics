"""
Defensive Game Script Indicators Feature Engineering (Phase 2.5)

Measures defensive effectiveness across different game script scenarios.
Part of the defensive interaction feature engineering pipeline.

This module quantifies defensive performance based on game context:
- Leading by 1 score (protect mode)
- Leading by 2+ scores (comfortable)
- Trailing by 1 score (get ball back mode)
- Trailing by 2+ scores (desperate mode)
- Tied game (competitive)
- Garbage time (large differential late)
- Competitive 4th quarter (within 1 score)
- Game script velocity (score momentum)

Author: NFL Analytics System
Date: 2025-10-30
"""

import logging

import pandas as pd
from sqlalchemy import create_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DefensiveGameScriptEngine:
    """
    Engineer defensive game script indicators.

    Tracks defense performance based on game situation context:
    - Score differential categories (leading/trailing/tied)
    - Quarter-specific situations (4th quarter urgency)
    - Garbage time vs competitive situations
    - Game script velocity (momentum)
    """

    def __init__(
        self,
        db_url: str = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"
    ):
        """Initialize the game script engine."""
        self.engine = create_engine(db_url)
        logger.info(f"Connected to database: {db_url}")

    def load_all_plays(
        self,
        start_season: int = 2024,
        end_season: int = 2025
    ) -> pd.DataFrame:
        """
        Load all offensive plays with game script context.

        Includes score differential, quarter, and timing information.
        """
        logger.info(f"Loading plays for game script analysis ({start_season}-{end_season})...")

        query = f"""
        SELECT
            p.game_id,
            g.season,
            g.week,
            g.kickoff as game_date,
            p.defteam as defense_team,
            p.posteam as offense_team,
            p.pass,
            p.rush,
            p.yards_gained,
            p.epa,
            p.wpa,
            p.success,
            p.touchdown,
            p.down,
            p.ydstogo,
            p.yardline_100,
            p.quarter,
            p.half_seconds_remaining,
            p.game_seconds_remaining,
            p.score_differential,
            p.first_down,
            p.interception,
            p.fumble_lost,
            p.sack,
            p.play_id
        FROM plays p
        JOIN games g ON p.game_id = g.game_id
        WHERE (p.pass = true OR p.rush = true)
            AND p.defteam IS NOT NULL
            AND p.score_differential IS NOT NULL
            AND g.season >= {start_season}
            AND g.season <= {end_season}
        ORDER BY g.kickoff, p.game_id, p.play_id
        """

        df = pd.read_sql(query, self.engine)
        logger.info(f"Loaded {len(df):,} plays")

        # Derive game script indicators
        self._add_game_script_indicators(df)

        return df

    def _add_game_script_indicators(self, df: pd.DataFrame) -> None:
        """
        Add game script categorical indicators to the plays dataframe.

        Score differential is from the perspective of the offense (posteam).
        For defense metrics, we need to invert the interpretation:
        - Positive score_differential = defense is LOSING (offense winning)
        - Negative score_differential = defense is WINNING (offense losing)
        """
        # Basic score categories (from defense perspective)
        # Note: score_differential > 0 means offense is ahead, so defense is trailing
        df["def_leading_1score"] = (df["score_differential"] <= -1) & (df["score_differential"] >= -7)
        df["def_leading_2score"] = (df["score_differential"] <= -8) & (df["score_differential"] >= -14)
        df["def_leading_3plus"] = df["score_differential"] <= -15
        df["def_trailing_1score"] = (df["score_differential"] >= 1) & (df["score_differential"] <= 7)
        df["def_trailing_2score"] = (df["score_differential"] >= 8) & (df["score_differential"] <= 14)
        df["def_trailing_3plus"] = df["score_differential"] >= 15
        df["def_tied"] = df["score_differential"] == 0

        # Garbage time: Large differential (>21 points) in 4th quarter
        df["garbage_time"] = (
            (df["quarter"] == 4) &
            (df["score_differential"].abs() > 21)
        )

        # Competitive 4th quarter: Within 1 score in 4th quarter
        df["competitive_4th"] = (
            (df["quarter"] == 4) &
            (df["score_differential"].abs() <= 7)
        )

        # High leverage: Close game, late in game
        df["high_leverage"] = (
            (df["game_seconds_remaining"] <= 600) &  # Last 10 minutes
            (df["score_differential"].abs() <= 7)     # Within 1 score
        )

        # Negative play (defense wins)
        df["negative_play"] = df["yards_gained"] < 0
        df["turnover"] = (df["interception"] == 1) | (df["fumble_lost"] == 1)

    def calculate_score_situation_defense(
        self,
        plays: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate defensive performance by score situation.

        Categories:
        - Leading by 1 score (protect mode)
        - Leading by 2 scores (comfortable)
        - Leading by 3+ scores (commanding)
        - Trailing by 1 score (get ball back)
        - Trailing by 2 scores (desperate)
        - Trailing by 3+ scores (very desperate)
        - Tied (competitive)
        """
        logger.info("Calculating defense by score situation...")

        situations = {
            "leading_1score": plays[plays["def_leading_1score"]],
            "leading_2score": plays[plays["def_leading_2score"]],
            "leading_3plus": plays[plays["def_leading_3plus"]],
            "trailing_1score": plays[plays["def_trailing_1score"]],
            "trailing_2score": plays[plays["def_trailing_2score"]],
            "trailing_3plus": plays[plays["def_trailing_3plus"]],
            "tied": plays[plays["def_tied"]],
        }

        results = []

        for situation_name, situation_df in situations.items():
            if len(situation_df) == 0:
                logger.warning(f"No plays found for situation: {situation_name}")
                continue

            for (game_id, defense_team), group in situation_df.groupby(["game_id", "defense_team"]):
                if len(group) < 5:  # Minimum sample size
                    continue

                metrics = {
                    "game_id": game_id,
                    "defense_team": defense_team,
                    "script_type": "score_situation",
                    "script_value": situation_name,
                    "n_plays": len(group),
                    "epa_per_play_allowed": group["epa"].mean(),
                    "success_rate_allowed": group["success"].mean() * 100,
                    "yards_per_play_allowed": group["yards_gained"].mean(),
                    "turnover_rate": group["turnover"].mean() * 100,
                    "sack_rate": group["sack"].mean() * 100,
                    "td_rate_allowed": group["touchdown"].mean() * 100,
                    "negative_play_rate": group["negative_play"].mean() * 100,
                }

                results.append(metrics)

        df = pd.DataFrame(results)
        logger.info(f"Generated {len(df):,} score situation defense records")

        return df

    def calculate_garbage_time_defense(
        self,
        plays: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate defensive performance in garbage time vs competitive situations.

        Garbage time: Large lead (>21 points) in 4th quarter
        Competitive: All other situations
        """
        logger.info("Calculating garbage time vs competitive defense...")

        results = []

        for (game_id, defense_team, is_garbage), group in plays.groupby(
            ["game_id", "defense_team", "garbage_time"]
        ):
            if len(group) < 5:
                continue

            script_value = "garbage_time" if is_garbage else "competitive"

            metrics = {
                "game_id": game_id,
                "defense_team": defense_team,
                "script_type": "garbage_time",
                "script_value": script_value,
                "n_plays": len(group),
                "epa_per_play_allowed": group["epa"].mean(),
                "success_rate_allowed": group["success"].mean() * 100,
                "yards_per_play_allowed": group["yards_gained"].mean(),
                "turnover_rate": group["turnover"].mean() * 100,
            }

            results.append(metrics)

        df = pd.DataFrame(results)
        logger.info(f"Generated {len(df):,} garbage time defense records")

        return df

    def calculate_competitive_fourth_quarter_defense(
        self,
        plays: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate defensive performance in competitive 4th quarters.

        Competitive 4th: Within 1 score (7 points) in 4th quarter.
        This is the highest pressure situation for defenses.
        """
        logger.info("Calculating competitive 4th quarter defense...")

        competitive_4th = plays[plays["competitive_4th"]].copy()

        if len(competitive_4th) == 0:
            logger.warning("No competitive 4th quarter plays found")
            return pd.DataFrame()

        results = []

        for (game_id, defense_team), group in competitive_4th.groupby(["game_id", "defense_team"]):
            if len(group) < 3:  # Lower threshold for this rare situation
                continue

            metrics = {
                "game_id": game_id,
                "defense_team": defense_team,
                "script_type": "competitive_4th",
                "script_value": "within_1score",
                "n_plays": len(group),
                "epa_per_play_allowed": group["epa"].mean(),
                "success_rate_allowed": group["success"].mean() * 100,
                "yards_per_play_allowed": group["yards_gained"].mean(),
                "turnover_rate": group["turnover"].mean() * 100,
                "sack_rate": group["sack"].mean() * 100,
                "td_rate_allowed": group["touchdown"].mean() * 100,
            }

            results.append(metrics)

        df = pd.DataFrame(results)
        logger.info(f"Generated {len(df):,} competitive 4th quarter defense records")

        return df

    def calculate_high_leverage_defense(
        self,
        plays: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate defensive performance in high leverage situations.

        High leverage: Last 10 minutes of game, within 1 score.
        """
        logger.info("Calculating high leverage defense...")

        high_leverage = plays[plays["high_leverage"]].copy()

        if len(high_leverage) == 0:
            logger.warning("No high leverage plays found")
            return pd.DataFrame()

        results = []

        for (game_id, defense_team), group in high_leverage.groupby(["game_id", "defense_team"]):
            if len(group) < 3:
                continue

            metrics = {
                "game_id": game_id,
                "defense_team": defense_team,
                "script_type": "high_leverage",
                "script_value": "last_10min_close",
                "n_plays": len(group),
                "epa_per_play_allowed": group["epa"].mean(),
                "wpa_per_play_allowed": group["wpa"].mean(),
                "success_rate_allowed": group["success"].mean() * 100,
                "yards_per_play_allowed": group["yards_gained"].mean(),
                "turnover_rate": group["turnover"].mean() * 100,
                "sack_rate": group["sack"].mean() * 100,
            }

            results.append(metrics)

        df = pd.DataFrame(results)
        logger.info(f"Generated {len(df):,} high leverage defense records")

        return df

    def calculate_game_script_velocity(
        self,
        plays: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate how defensive performance changes as game script evolves.

        Tracks:
        - Score differential momentum (getting worse/better for defense)
        - EPA momentum (recent defensive performance trend)
        """
        logger.info("Calculating game script velocity metrics...")

        results = []

        for (game_id, defense_team), game_plays in plays.groupby(["game_id", "defense_team"]):
            if len(game_plays) < 15:  # Need enough plays to measure velocity
                continue

            # Sort by play order
            game_plays = game_plays.sort_values("play_id")

            # Calculate rolling metrics for velocity
            window = 5  # 5-play rolling window

            game_plays["score_diff_change"] = game_plays["score_differential"].diff()
            game_plays["epa_roll5"] = game_plays["epa"].rolling(window, min_periods=1).mean()

            # Determine if defense is improving or declining in performance
            # (from defense perspective: negative EPA = good)
            game_plays["def_improving"] = game_plays["epa_roll5"].diff() < 0

            # Score momentum: Is the game getting closer or further apart?
            # For defense, we want score differential moving toward negative (defense gaining lead)
            game_plays["score_momentum_favorable"] = game_plays["score_diff_change"] < 0

            # Aggregate for the game
            metrics = {
                "game_id": game_id,
                "defense_team": defense_team,
                "script_type": "velocity",
                "script_value": "game_trend",
                "n_plays": len(game_plays),
                "avg_epa_per_play": game_plays["epa"].mean(),
                "epa_momentum": game_plays["epa_roll5"].iloc[-1] if len(game_plays) >= window else game_plays["epa"].mean(),
                "pct_plays_improving": game_plays["def_improving"].mean() * 100,
                "avg_score_momentum": game_plays["score_diff_change"].mean(),
                "pct_favorable_momentum": game_plays["score_momentum_favorable"].mean() * 100,
            }

            results.append(metrics)

        df = pd.DataFrame(results)
        logger.info(f"Generated {len(df):,} game script velocity records")

        return df

    def add_rolling_averages(
        self,
        metrics: pd.DataFrame,
        window: int = 4
    ) -> pd.DataFrame:
        """
        Add rolling averages for trend analysis.
        """
        logger.info(f"Adding {window}-game rolling averages...")

        metrics = metrics.copy()
        metrics = metrics.sort_values(["defense_team", "game_id"])

        rolling_cols = [
            "epa_per_play_allowed",
            "success_rate_allowed",
            "yards_per_play_allowed",
            "turnover_rate",
            "sack_rate",
        ]

        for col in rolling_cols:
            if col in metrics.columns:
                metrics[f"{col}_roll{window}"] = (
                    metrics.groupby(["defense_team", "script_type", "script_value"])[col]
                    .transform(lambda x: x.rolling(window, min_periods=1).mean())
                )

        logger.info("Added rolling averages for game script metrics")

        return metrics

    def generate_all_features(
        self,
        start_season: int = 2024,
        end_season: int = 2025
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate all defensive game script features.

        Returns:
            Tuple of (score_situation_metrics, garbage_time_metrics,
                     competitive_4th_metrics, high_leverage_metrics, velocity_metrics)
        """
        logger.info("=" * 70)
        logger.info("Defensive Game Script Indicators Feature Engineering")
        logger.info("=" * 70)

        # Load data
        plays = self.load_all_plays(start_season, end_season)

        # Calculate all game script metrics
        score_situation_metrics = self.calculate_score_situation_defense(plays)
        garbage_time_metrics = self.calculate_garbage_time_defense(plays)
        competitive_4th_metrics = self.calculate_competitive_fourth_quarter_defense(plays)
        high_leverage_metrics = self.calculate_high_leverage_defense(plays)
        velocity_metrics = self.calculate_game_script_velocity(plays)

        # Add rolling averages where applicable
        if len(score_situation_metrics) > 0:
            score_situation_metrics = self.add_rolling_averages(score_situation_metrics, window=4)
        if len(garbage_time_metrics) > 0:
            garbage_time_metrics = self.add_rolling_averages(garbage_time_metrics, window=4)
        if len(competitive_4th_metrics) > 0:
            competitive_4th_metrics = self.add_rolling_averages(competitive_4th_metrics, window=4)
        if len(high_leverage_metrics) > 0:
            high_leverage_metrics = self.add_rolling_averages(high_leverage_metrics, window=4)

        logger.info("=" * 70)
        logger.info("Feature generation complete!")
        logger.info(f"  Score situation metrics: {len(score_situation_metrics):,} records")
        logger.info(f"  Garbage time metrics: {len(garbage_time_metrics):,} records")
        logger.info(f"  Competitive 4th metrics: {len(competitive_4th_metrics):,} records")
        logger.info(f"  High leverage metrics: {len(high_leverage_metrics):,} records")
        logger.info(f"  Velocity metrics: {len(velocity_metrics):,} records")
        logger.info("=" * 70)

        return (
            score_situation_metrics,
            garbage_time_metrics,
            competitive_4th_metrics,
            high_leverage_metrics,
            velocity_metrics
        )


def main():
    """Test the game script engine."""
    engine = DefensiveGameScriptEngine()

    # Generate ALL features for 2020-2025 seasons
    (
        score_situation_metrics,
        garbage_time_metrics,
        competitive_4th_metrics,
        high_leverage_metrics,
        velocity_metrics
    ) = engine.generate_all_features(
        start_season=2020,
        end_season=2025
    )

    # Display sample statistics
    print("\n" + "=" * 70)
    print("SCORE SITUATION DEFENSE SAMPLE")
    print("=" * 70)
    if len(score_situation_metrics) > 0:
        print(score_situation_metrics.head(15).to_string())
        print("\nSCORE SITUATION DEFENSE STATISTICS")
        print(score_situation_metrics[[
            "epa_per_play_allowed",
            "success_rate_allowed",
            "turnover_rate",
            "sack_rate"
        ]].describe())

    print("\n" + "=" * 70)
    print("COMPETITIVE 4TH QUARTER DEFENSE SAMPLE")
    print("=" * 70)
    if len(competitive_4th_metrics) > 0:
        print(competitive_4th_metrics.head(10).to_string())
        print("\nCOMPETITIVE 4TH QUARTER STATISTICS")
        print(competitive_4th_metrics[[
            "epa_per_play_allowed",
            "success_rate_allowed",
            "turnover_rate"
        ]].describe())

    print("\n" + "=" * 70)
    print("HIGH LEVERAGE DEFENSE SAMPLE")
    print("=" * 70)
    if len(high_leverage_metrics) > 0:
        print(high_leverage_metrics.head(10).to_string())

    print("\n" + "=" * 70)
    print("GAME SCRIPT VELOCITY SAMPLE")
    print("=" * 70)
    if len(velocity_metrics) > 0:
        print(velocity_metrics.head(10).to_string())

    print("\n" + "=" * 70)
    print("TOP 10 DEFENSES IN COMPETITIVE 4TH QUARTERS (by EPA)")
    print("=" * 70)
    if len(competitive_4th_metrics) > 0:
        top_clutch = competitive_4th_metrics.nsmallest(10, "epa_per_play_allowed")[[
            "defense_team", "game_id", "n_plays", "epa_per_play_allowed",
            "success_rate_allowed", "turnover_rate"
        ]]
        print(top_clutch.to_string(index=False))

    print("\n" + "=" * 70)
    print("DEFENSE PERFORMANCE BY SCORE SITUATION")
    print("=" * 70)
    if len(score_situation_metrics) > 0:
        summary = score_situation_metrics.groupby("script_value").agg({
            "epa_per_play_allowed": "mean",
            "success_rate_allowed": "mean",
            "turnover_rate": "mean",
            "n_plays": "sum"
        }).round(2)
        print(summary.to_string())

    return (
        score_situation_metrics,
        garbage_time_metrics,
        competitive_4th_metrics,
        high_leverage_metrics,
        velocity_metrics
    )


if __name__ == "__main__":
    main()
