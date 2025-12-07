"""
Defensive Situational Strength Feature Engineering (Phase 2.4)

Measures defensive effectiveness across critical game situations.
Part of the defensive interaction feature engineering pipeline.

This module quantifies overall defensive quality in high-leverage situations:
- Red zone defense (inside 20-yard line)
- Third down conversion prevention
- Two-minute drill defense
- Goal line stands (inside 5-yard line)
- Short yardage stops
- Performance by quarter (especially 4th)
- Performance by score differential

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


class DefensiveSituationalStrengthEngine:
    """
    Engineer defensive situational strength metrics.

    Tracks defense performance in high-leverage situations:
    - Red zone defense (inside 20)
    - Goal line defense (inside 5)
    - Third down stops
    - Fourth down stops
    - Two-minute drill defense
    - Performance by quarter
    - Performance by score differential
    """

    def __init__(
        self,
        db_url: str = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"
    ):
        """Initialize the situational strength engine."""
        self.engine = create_engine(db_url)
        logger.info(f"Connected to database: {db_url}")

    def load_all_plays(
        self,
        start_season: int = 2024,
        end_season: int = 2025
    ) -> pd.DataFrame:
        """
        Load all offensive plays from the database.

        Includes passing and rushing plays for comprehensive defensive analysis.
        """
        logger.info(f"Loading all plays for seasons {start_season}-{end_season}...")

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
            p.third_down_converted,
            p.fourth_down_converted,
            p.first_down,
            p.interception,
            p.fumble_lost,
            p.sack
        FROM plays p
        JOIN games g ON p.game_id = g.game_id
        WHERE (p.pass = true OR p.rush = true)
            AND p.defteam IS NOT NULL
            AND g.season >= {start_season}
            AND g.season <= {end_season}
        ORDER BY g.kickoff, p.game_id, p.play_id
        """

        df = pd.read_sql(query, self.engine)
        logger.info(f"Loaded {len(df):,} plays")

        # Derive situational indicators
        df["short_yardage"] = df["ydstogo"] <= 2
        df["two_minute_drill"] = (
            ((df["quarter"] == 2) & (df["half_seconds_remaining"] <= 120)) |
            ((df["quarter"] == 4) & (df["game_seconds_remaining"] <= 120))
        )
        df["negative_play"] = df["yards_gained"] < 0
        df["turnover"] = (df["interception"] == 1) | (df["fumble_lost"] == 1)

        # Field position indicators (yardline_100 is yards to opponent's end zone)
        df["red_zone"] = (df["yardline_100"] <= 20) & (df["yardline_100"].notna())
        df["goal_line"] = (df["yardline_100"] <= 5) & (df["yardline_100"].notna())

        return df

    def calculate_red_zone_defense(
        self,
        plays: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate red zone defense effectiveness (inside 20-yard line).

        Red zone defense is critical for preventing touchdowns.

        Metrics:
        - Red zone plays allowed
        - Red zone TD rate allowed
        - Red zone EPA allowed
        - Red zone success rate allowed
        - Red zone turnover rate (forced)
        """
        logger.info("Calculating red zone defense metrics...")

        red_zone = plays[plays["red_zone"]].copy()

        if len(red_zone) == 0:
            logger.warning("No red zone plays found")
            return pd.DataFrame()

        results = []

        for (game_id, defense_team), group in red_zone.groupby(["game_id", "defense_team"]):
            if len(group) < 3:  # Minimum sample size
                continue

            metrics = {
                "game_id": game_id,
                "defense_team": defense_team,
                "situation_type": "red_zone",
                "n_plays": len(group),
                "td_rate_allowed": group["touchdown"].mean() * 100,
                "epa_per_play_allowed": group["epa"].mean(),
                "success_rate_allowed": group["success"].mean() * 100,
                "turnover_rate": group["turnover"].mean() * 100,
                "yards_per_play_allowed": group["yards_gained"].mean(),
            }

            results.append(metrics)

        df = pd.DataFrame(results)
        logger.info(f"Generated {len(df):,} red zone defense records")

        return df

    def calculate_goal_line_defense(
        self,
        plays: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate goal line defense effectiveness (inside 5-yard line).

        Goal line stands are critical for momentum and points prevention.

        Metrics:
        - Goal line plays
        - TD rate allowed
        - Success rate allowed
        - Negative play rate (stuffs)
        """
        logger.info("Calculating goal line defense metrics...")

        goal_line = plays[plays["goal_line"]].copy()

        if len(goal_line) == 0:
            logger.warning("No goal line plays found")
            return pd.DataFrame()

        results = []

        for (game_id, defense_team), group in goal_line.groupby(["game_id", "defense_team"]):
            if len(group) < 2:
                continue

            metrics = {
                "game_id": game_id,
                "defense_team": defense_team,
                "situation_type": "goal_line",
                "n_plays": len(group),
                "td_rate_allowed": group["touchdown"].mean() * 100,
                "success_rate_allowed": group["success"].mean() * 100,
                "negative_play_rate": group["negative_play"].mean() * 100,
                "yards_per_play_allowed": group["yards_gained"].mean(),
            }

            results.append(metrics)

        df = pd.DataFrame(results)
        logger.info(f"Generated {len(df):,} goal line defense records")

        return df

    def calculate_third_down_defense(
        self,
        plays: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate third down conversion prevention.

        Third down stops are critical for getting off the field.

        Metrics:
        - Third down plays
        - Conversion rate allowed
        - EPA per play allowed
        - Sack rate
        - Turnover rate
        """
        logger.info("Calculating third down defense metrics...")

        third_downs = plays[plays["down"] == 3].copy()

        if len(third_downs) == 0:
            logger.warning("No third down plays found")
            return pd.DataFrame()

        results = []

        for (game_id, defense_team), group in third_downs.groupby(["game_id", "defense_team"]):
            if len(group) < 3:
                continue

            # Handle None values in third_down_converted
            conversions = group["third_down_converted"].fillna(0)

            metrics = {
                "game_id": game_id,
                "defense_team": defense_team,
                "situation_type": "third_down",
                "n_plays": len(group),
                "conversion_rate_allowed": conversions.mean() * 100,
                "epa_per_play_allowed": group["epa"].mean(),
                "sack_rate": group["sack"].mean() * 100,
                "turnover_rate": group["turnover"].mean() * 100,
            }

            results.append(metrics)

        df = pd.DataFrame(results)
        logger.info(f"Generated {len(df):,} third down defense records")

        return df

    def calculate_fourth_down_defense(
        self,
        plays: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate fourth down conversion prevention.

        Fourth down stops are critical high-leverage situations.
        """
        logger.info("Calculating fourth down defense metrics...")

        fourth_downs = plays[plays["down"] == 4].copy()

        if len(fourth_downs) == 0:
            logger.warning("No fourth down plays found")
            return pd.DataFrame()

        results = []

        for (game_id, defense_team), group in fourth_downs.groupby(["game_id", "defense_team"]):
            if len(group) < 1:
                continue

            # Handle None values
            conversions = group["fourth_down_converted"].fillna(0)

            metrics = {
                "game_id": game_id,
                "defense_team": defense_team,
                "situation_type": "fourth_down",
                "n_plays": len(group),
                "conversion_rate_allowed": conversions.mean() * 100,
                "epa_per_play_allowed": group["epa"].mean(),
                "turnover_rate": group["turnover"].mean() * 100,
            }

            results.append(metrics)

        df = pd.DataFrame(results)
        logger.info(f"Generated {len(df):,} fourth down defense records")

        return df

    def calculate_two_minute_defense(
        self,
        plays: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate two-minute drill defense effectiveness.

        Two-minute drills test defense's ability to prevent scores under pressure.
        Critical for end of half and end of game situations.
        """
        logger.info("Calculating two-minute drill defense metrics...")

        two_minute = plays[plays["two_minute_drill"]].copy()

        if len(two_minute) == 0:
            logger.warning("No two-minute drill plays found")
            return pd.DataFrame()

        results = []

        for (game_id, defense_team), group in two_minute.groupby(["game_id", "defense_team"]):
            if len(group) < 3:
                continue

            metrics = {
                "game_id": game_id,
                "defense_team": defense_team,
                "situation_type": "two_minute_drill",
                "n_plays": len(group),
                "td_rate_allowed": group["touchdown"].mean() * 100,
                "epa_per_play_allowed": group["epa"].mean(),
                "success_rate_allowed": group["success"].mean() * 100,
                "turnover_rate": group["turnover"].mean() * 100,
                "yards_per_play_allowed": group["yards_gained"].mean(),
            }

            results.append(metrics)

        df = pd.DataFrame(results)
        logger.info(f"Generated {len(df):,} two-minute defense records")

        return df

    def calculate_quarter_defense(
        self,
        plays: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate defensive performance by quarter.

        Fourth quarter performance is especially important for closing games.
        """
        logger.info("Calculating defense by quarter...")

        results = []

        for (game_id, defense_team, quarter), group in plays.groupby(
            ["game_id", "defense_team", "quarter"]
        ):
            if len(group) < 5:  # Minimum sample size
                continue

            metrics = {
                "game_id": game_id,
                "defense_team": defense_team,
                "situation_type": "quarter",
                "situation_value": f"Q{int(quarter)}",
                "n_plays": len(group),
                "epa_per_play_allowed": group["epa"].mean(),
                "success_rate_allowed": group["success"].mean() * 100,
                "yards_per_play_allowed": group["yards_gained"].mean(),
                "turnover_rate": group["turnover"].mean() * 100,
            }

            results.append(metrics)

        df = pd.DataFrame(results)
        logger.info(f"Generated {len(df):,} quarter defense records")

        return df

    def calculate_score_differential_defense(
        self,
        plays: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate defensive performance by score differential.

        Measures how defense performs when winning, tied, or losing.
        """
        logger.info("Calculating defense by score differential...")

        plays = plays.copy()
        plays["score_situation"] = pd.cut(
            plays["score_differential"],
            bins=[-100, -14, -7, 0, 7, 14, 100],
            labels=["losing_big", "losing", "tied_or_close_losing", "tied_or_close_winning", "winning", "winning_big"]
        )

        results = []

        for (game_id, defense_team, score_sit), group in plays.groupby(
            ["game_id", "defense_team", "score_situation"]
        ):
            if len(group) < 5:
                continue

            metrics = {
                "game_id": game_id,
                "defense_team": defense_team,
                "situation_type": "score_differential",
                "situation_value": str(score_sit),
                "n_plays": len(group),
                "epa_per_play_allowed": group["epa"].mean(),
                "success_rate_allowed": group["success"].mean() * 100,
                "yards_per_play_allowed": group["yards_gained"].mean(),
            }

            results.append(metrics)

        df = pd.DataFrame(results)
        logger.info(f"Generated {len(df):,} score differential defense records")

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
            "td_rate_allowed",
            "epa_per_play_allowed",
            "success_rate_allowed",
            "conversion_rate_allowed",
            "turnover_rate",
        ]

        for col in rolling_cols:
            if col in metrics.columns:
                metrics[f"{col}_roll{window}"] = (
                    metrics.groupby("defense_team")[col]
                    .transform(lambda x: x.rolling(window, min_periods=1).mean())
                )

        logger.info("Added rolling averages for situational metrics")

        return metrics

    def generate_all_features(
        self,
        start_season: int = 2024,
        end_season: int = 2025
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate all defensive situational strength features.

        Returns:
            Tuple of (red_zone_metrics, goal_line_metrics, third_down_metrics,
                     fourth_down_metrics, two_minute_metrics, quarter_metrics, score_diff_metrics)
        """
        logger.info("=" * 70)
        logger.info("Defensive Situational Strength Feature Engineering")
        logger.info("=" * 70)

        # Load data
        plays = self.load_all_plays(start_season, end_season)

        # Calculate situational metrics (ALL features - no skipping!)
        red_zone_metrics = self.calculate_red_zone_defense(plays)
        goal_line_metrics = self.calculate_goal_line_defense(plays)
        third_down_metrics = self.calculate_third_down_defense(plays)
        fourth_down_metrics = self.calculate_fourth_down_defense(plays)
        two_minute_metrics = self.calculate_two_minute_defense(plays)
        quarter_metrics = self.calculate_quarter_defense(plays)
        score_diff_metrics = self.calculate_score_differential_defense(plays)

        # Add rolling averages where applicable
        if len(red_zone_metrics) > 0:
            red_zone_metrics = self.add_rolling_averages(red_zone_metrics, window=4)
        if len(goal_line_metrics) > 0:
            goal_line_metrics = self.add_rolling_averages(goal_line_metrics, window=4)
        if len(third_down_metrics) > 0:
            third_down_metrics = self.add_rolling_averages(third_down_metrics, window=4)

        logger.info("=" * 70)
        logger.info("Feature generation complete!")
        logger.info(f"  Red zone metrics: {len(red_zone_metrics):,} records")
        logger.info(f"  Goal line metrics: {len(goal_line_metrics):,} records")
        logger.info(f"  Third down metrics: {len(third_down_metrics):,} records")
        logger.info(f"  Fourth down metrics: {len(fourth_down_metrics):,} records")
        logger.info(f"  Two-minute metrics: {len(two_minute_metrics):,} records")
        logger.info(f"  Quarter metrics: {len(quarter_metrics):,} records")
        logger.info(f"  Score diff metrics: {len(score_diff_metrics):,} records")
        logger.info("=" * 70)

        return (
            red_zone_metrics,
            goal_line_metrics,
            third_down_metrics,
            fourth_down_metrics,
            two_minute_metrics,
            quarter_metrics,
            score_diff_metrics
        )


def main():
    """Test the situational strength engine."""
    engine = DefensiveSituationalStrengthEngine()

    # Generate ALL features for 2020-2025 seasons
    (
        red_zone_metrics,
        goal_line_metrics,
        third_down_metrics,
        fourth_down_metrics,
        two_minute_metrics,
        quarter_metrics,
        score_diff_metrics
    ) = engine.generate_all_features(
        start_season=2020,
        end_season=2025
    )

    # Display sample statistics
    print("\n" + "=" * 70)
    print("RED ZONE DEFENSE SAMPLE")
    print("=" * 70)
    if len(red_zone_metrics) > 0:
        print(red_zone_metrics.head(10).to_string())
        print("\nRED ZONE DEFENSE STATISTICS")
        print(red_zone_metrics[[
            "td_rate_allowed",
            "epa_per_play_allowed",
            "success_rate_allowed",
            "turnover_rate"
        ]].describe())

    print("\n" + "=" * 70)
    print("GOAL LINE DEFENSE SAMPLE")
    print("=" * 70)
    if len(goal_line_metrics) > 0:
        print(goal_line_metrics.head(10).to_string())

    print("\n" + "=" * 70)
    print("THIRD DOWN DEFENSE SAMPLE")
    print("=" * 70)
    if len(third_down_metrics) > 0:
        print(third_down_metrics.head(10).to_string())
        print("\nTHIRD DOWN DEFENSE STATISTICS")
        print(third_down_metrics[[
            "conversion_rate_allowed",
            "epa_per_play_allowed",
            "sack_rate",
            "turnover_rate"
        ]].describe())

    print("\n" + "=" * 70)
    print("FOURTH DOWN DEFENSE SAMPLE")
    print("=" * 70)
    if len(fourth_down_metrics) > 0:
        print(fourth_down_metrics.head(10).to_string())

    print("\n" + "=" * 70)
    print("TWO-MINUTE DRILL DEFENSE SAMPLE")
    print("=" * 70)
    if len(two_minute_metrics) > 0:
        print(two_minute_metrics.head(10).to_string())

    print("\n" + "=" * 70)
    print("TOP 10 RED ZONE DEFENSES (by EPA)")
    print("=" * 70)
    if len(red_zone_metrics) > 0:
        top_rz_d = red_zone_metrics.nsmallest(10, "epa_per_play_allowed")[[
            "defense_team", "game_id", "n_plays", "td_rate_allowed",
            "epa_per_play_allowed", "turnover_rate"
        ]]
        print(top_rz_d.to_string(index=False))

    print("\n" + "=" * 70)
    print("TOP 10 THIRD DOWN DEFENSES (by conversion rate)")
    print("=" * 70)
    if len(third_down_metrics) > 0:
        top_3rd_d = third_down_metrics.nsmallest(10, "conversion_rate_allowed")[[
            "defense_team", "game_id", "n_plays", "conversion_rate_allowed",
            "epa_per_play_allowed", "sack_rate"
        ]]
        print(top_3rd_d.to_string(index=False))

    print("\n" + "=" * 70)
    print("TOP 10 TWO-MINUTE DRILL DEFENSES (by EPA)")
    print("=" * 70)
    if len(two_minute_metrics) > 0:
        top_2min_d = two_minute_metrics.nsmallest(10, "epa_per_play_allowed")[[
            "defense_team", "game_id", "n_plays", "td_rate_allowed",
            "epa_per_play_allowed", "turnover_rate"
        ]]
        print(top_2min_d.to_string(index=False))

    return (
        red_zone_metrics,
        goal_line_metrics,
        third_down_metrics,
        fourth_down_metrics,
        two_minute_metrics,
        quarter_metrics,
        score_diff_metrics
    )


if __name__ == "__main__":
    main()
