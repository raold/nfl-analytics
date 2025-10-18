#!/usr/bin/env python3
"""
Treatment Definitions for Causal Inference

Defines and identifies various treatment conditions in NFL data:
- Player injuries/absences
- Coaching changes
- Player trades/team changes
- Weather events
- Rule changes
"""

import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TreatmentDefiner:
    """
    Defines and identifies treatment conditions for causal analysis.

    Treatments are events that potentially cause changes in outcomes.
    We need to carefully define:
    - Treatment onset (when it starts)
    - Treatment duration (how long it lasts)
    - Treatment intensity (binary vs continuous)
    - Control group (untreated units for comparison)
    """

    def __init__(self):
        self.treatments = {}

    def define_injury_treatment(
        self, df: pd.DataFrame, min_games_missed: int = 1, star_players_only: bool = False
    ) -> pd.DataFrame:
        """
        Define injury as a treatment.

        Treatment occurs when a player misses games due to injury.
        Post-treatment period is when they return.

        Args:
            df: Panel dataframe with player-game observations
            min_games_missed: Minimum games missed to qualify as treatment
            star_players_only: Only consider injuries to star players

        Returns:
            DataFrame with injury treatment indicators
        """
        df = df.copy()

        # Sort by player and time
        df = df.sort_values(["player_id", "season", "week"])

        # Identify games missed (gaps in weekly sequence)
        df["expected_week"] = df.groupby(["player_id", "season"])["week"].shift(1) + 1
        df["games_missed_flag"] = (df["week"] - df["expected_week"] > 0).astype(int)
        df["consecutive_games_missed"] = df["week"] - df["expected_week"]
        df["consecutive_games_missed"] = df["consecutive_games_missed"].fillna(0).clip(lower=0)

        # Create treatment indicator
        df["injury_treatment"] = (df["consecutive_games_missed"] >= min_games_missed).astype(int)

        # Mark post-treatment period (return from injury)
        df["post_injury"] = df.groupby("player_id")["injury_treatment"].cumsum() > 0

        # Create treatment intensity (number of games missed)
        df["injury_intensity"] = df["consecutive_games_missed"]

        # If star players only, mask non-stars
        if star_players_only and "is_star" in df.columns:
            df.loc[df["is_star"] == 0, "injury_treatment"] = 0
            df.loc[df["is_star"] == 0, "post_injury"] = 0

        # Log treatment statistics
        n_treatments = df["injury_treatment"].sum()
        n_players_treated = df[df["injury_treatment"] == 1]["player_id"].nunique()

        logger.info(
            f"Identified {n_treatments} injury treatments affecting {n_players_treated} players"
        )

        return df

    def define_coaching_change_treatment(
        self, df: pd.DataFrame, coaching_changes: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """
        Define coaching change as a treatment.

        Treatment affects all players/games for a team after coaching change.

        Args:
            df: Panel dataframe
            coaching_changes: DataFrame with columns [team, season, change_week]

        Returns:
            DataFrame with coaching change treatment indicators
        """
        df = df.copy()

        # If no coaching changes provided, use known examples
        if coaching_changes is None:
            # Known coaching changes (would be loaded from database in production)
            coaching_changes = pd.DataFrame(
                [
                    {"team": "LV", "season": 2021, "change_week": 6},  # Raiders fired Gruden
                    {"team": "JAX", "season": 2021, "change_week": 14},  # Jaguars fired Meyer
                    {"team": "CAR", "season": 2022, "change_week": 6},  # Panthers fired Rhule
                    {"team": "IND", "season": 2022, "change_week": 9},  # Colts fired Reich
                    {"team": "DEN", "season": 2022, "change_week": 16},  # Broncos fired Hackett
                    {"team": "CAR", "season": 2023, "change_week": 11},  # Panthers fired Reich
                    {"team": "LV", "season": 2023, "change_week": 8},  # Raiders fired McDaniels
                ]
            )

        # Initialize treatment columns
        df["coaching_change_treatment"] = 0
        df["post_coaching_change"] = 0
        df["weeks_since_coaching_change"] = np.nan

        # Apply treatment indicators
        for _, change in coaching_changes.iterrows():
            mask = (df["team"] == change["team"]) & (df["season"] == change["season"])

            # Mark the week of change as treatment
            df.loc[mask & (df["week"] == change["change_week"]), "coaching_change_treatment"] = 1

            # Mark post-treatment period
            df.loc[mask & (df["week"] >= change["change_week"]), "post_coaching_change"] = 1

            # Calculate weeks since change
            df.loc[mask & (df["week"] >= change["change_week"]), "weeks_since_coaching_change"] = (
                df.loc[mask & (df["week"] >= change["change_week"]), "week"] - change["change_week"]
            )

        n_changes = len(coaching_changes)
        n_games_affected = df["post_coaching_change"].sum()

        logger.info(
            f"Identified {n_changes} coaching changes affecting {n_games_affected} team-games"
        )

        return df

    def define_trade_treatment(
        self, df: pd.DataFrame, min_games_before: int = 3, min_games_after: int = 3
    ) -> pd.DataFrame:
        """
        Define player trade/team change as a treatment.

        Treatment occurs when a player changes teams mid-season.

        Args:
            df: Panel dataframe
            min_games_before: Minimum games with old team to qualify
            min_games_after: Minimum games with new team to qualify

        Returns:
            DataFrame with trade treatment indicators
        """
        df = df.copy()
        df = df.sort_values(["player_id", "season", "week"])

        # Detect team changes within a season
        df["prev_team"] = df.groupby(["player_id", "season"])["team"].shift(1)
        df["team_changed"] = (df["team"] != df["prev_team"]) & df["prev_team"].notna()

        # Count games with each team
        df["games_with_current_team"] = df.groupby(["player_id", "season", "team"]).cumcount() + 1

        # Only consider mid-season trades (not offseason moves)
        df["trade_treatment"] = (
            df["team_changed"]
            & (df["week"] > 1)  # Not week 1
            & (df["week"] < 15)  # Not late season
        ).astype(int)

        # Mark post-trade period
        df["post_trade"] = df.groupby(["player_id", "season"])["trade_treatment"].cumsum() > 0

        # Weeks since trade
        trade_weeks = df[df["trade_treatment"] == 1].groupby("player_id")["week"].first()
        for player_id in trade_weeks.index:
            player_mask = df["player_id"] == player_id
            trade_week = trade_weeks[player_id]
            df.loc[player_mask & df["post_trade"], "weeks_since_trade"] = (
                df.loc[player_mask & df["post_trade"], "week"] - trade_week
            )

        n_trades = df["trade_treatment"].sum()
        n_players = df[df["trade_treatment"] == 1]["player_id"].nunique()

        logger.info(f"Identified {n_trades} trades affecting {n_players} players")

        return df

    def define_weather_treatment(
        self,
        df: pd.DataFrame,
        temp_threshold: float = 32.0,  # Freezing
        wind_threshold: float = 20.0,  # High wind
        precipitation_threshold: float = 0.5,  # Heavy rain/snow
    ) -> pd.DataFrame:
        """
        Define extreme weather as a treatment.

        Treatment occurs when games are played in extreme conditions.

        Args:
            df: Panel dataframe with weather data
            temp_threshold: Temperature threshold for cold weather
            wind_threshold: Wind speed threshold
            precipitation_threshold: Precipitation threshold

        Returns:
            DataFrame with weather treatment indicators
        """
        df = df.copy()

        # Define extreme weather conditions
        extreme_cold = (df.get("temp", 100) < temp_threshold).astype(int)
        high_wind = (df.get("wind", 0) > wind_threshold).astype(int)
        heavy_precip = (df.get("weather_category", "") == "Rain").astype(int) | (
            df.get("weather_category", "") == "Snow"
        ).astype(int)

        # Combined weather treatment (any extreme condition)
        df["weather_treatment"] = (extreme_cold | high_wind | heavy_precip).astype(int)

        # Specific weather treatments
        df["cold_weather_treatment"] = extreme_cold
        df["wind_treatment"] = high_wind
        df["precipitation_treatment"] = heavy_precip

        # Weather intensity (continuous treatment)
        if "temp" in df.columns:
            df["cold_intensity"] = np.maximum(0, temp_threshold - df["temp"])
        if "wind" in df.columns:
            df["wind_intensity"] = np.maximum(0, df["wind"] - wind_threshold)

        n_weather_games = df["weather_treatment"].sum()
        logger.info(f"Identified {n_weather_games} games with extreme weather treatment")

        return df

    def define_synthetic_treatment(
        self, df: pd.DataFrame, treatment_prob: float = 0.1, seed: int = 42
    ) -> pd.DataFrame:
        """
        Define synthetic/placebo treatment for validation.

        Randomly assigns treatment for placebo testing.

        Args:
            df: Panel dataframe
            treatment_prob: Probability of treatment assignment
            seed: Random seed for reproducibility

        Returns:
            DataFrame with synthetic treatment indicators
        """
        np.random.seed(seed)
        df = df.copy()

        # Random treatment assignment
        df["synthetic_treatment"] = np.random.binomial(1, treatment_prob, size=len(df))

        # Create synthetic post-treatment period
        df["post_synthetic"] = df.groupby("player_id")["synthetic_treatment"].cumsum() > 0

        n_synthetic = df["synthetic_treatment"].sum()
        logger.info(f"Created {n_synthetic} synthetic treatments for placebo testing")

        return df

    def create_treatment_windows(
        self, df: pd.DataFrame, treatment_col: str, pre_window: int = 3, post_window: int = 3
    ) -> pd.DataFrame:
        """
        Create pre/post treatment windows for analysis.

        Args:
            df: Panel dataframe
            treatment_col: Name of treatment column
            pre_window: Number of periods before treatment
            post_window: Number of periods after treatment

        Returns:
            DataFrame with treatment window indicators
        """
        df = df.copy()

        # Sort by unit and time
        if "player_id" in df.columns:
            df = df.sort_values(["player_id", "season", "week"])
            group_cols = ["player_id", "season"]
        else:
            df = df.sort_values(["team", "season", "week"])
            group_cols = ["team", "season"]

        # Find treatment periods for each unit
        treatment_periods = df[df[treatment_col] == 1].groupby(group_cols[0])["week"].min()

        # Initialize window columns
        df[f"{treatment_col}_pre_window"] = 0
        df[f"{treatment_col}_post_window"] = 0
        df[f"{treatment_col}_window_period"] = np.nan

        # Apply windows
        for unit, treatment_week in treatment_periods.items():
            unit_mask = df[group_cols[0]] == unit

            # Pre-treatment window
            pre_mask = (
                unit_mask
                & (df["week"] >= treatment_week - pre_window)
                & (df["week"] < treatment_week)
            )
            df.loc[pre_mask, f"{treatment_col}_pre_window"] = 1
            df.loc[pre_mask, f"{treatment_col}_window_period"] = (
                df.loc[pre_mask, "week"] - treatment_week
            )

            # Post-treatment window
            post_mask = (
                unit_mask
                & (df["week"] > treatment_week)
                & (df["week"] <= treatment_week + post_window)
            )
            df.loc[post_mask, f"{treatment_col}_post_window"] = 1
            df.loc[post_mask, f"{treatment_col}_window_period"] = (
                df.loc[post_mask, "week"] - treatment_week
            )

        return df

    def get_treatment_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for all treatments in the dataframe.

        Returns:
            Summary DataFrame with treatment statistics
        """
        treatment_cols = [
            col for col in df.columns if "treatment" in col.lower() and "post" not in col.lower()
        ]

        summaries = []
        for col in treatment_cols:
            if col in df.columns:
                summary = {
                    "treatment": col.replace("_treatment", ""),
                    "n_treated_observations": df[col].sum(),
                    "n_treated_units": (
                        df[df[col] == 1][
                            "player_id" if "player_id" in df.columns else "team"
                        ].nunique()
                        if col in df.columns
                        else 0
                    ),
                    "treatment_rate": df[col].mean() * 100,
                    "has_post_period": f'post_{col.replace("_treatment", "")}' in df.columns,
                }
                summaries.append(summary)

        summary_df = pd.DataFrame(summaries)
        return summary_df

    def validate_treatment_assignment(
        self, df: pd.DataFrame, treatment_col: str, covariates: list[str]
    ) -> dict:
        """
        Validate treatment assignment by checking balance on covariates.

        Args:
            df: Panel dataframe
            treatment_col: Treatment column name
            covariates: List of covariate columns to check

        Returns:
            Dictionary with balance statistics
        """
        treated = df[df[treatment_col] == 1]
        control = df[df[treatment_col] == 0]

        balance_stats = {}
        for cov in covariates:
            if cov in df.columns:
                treated_mean = treated[cov].mean()
                control_mean = control[cov].mean()
                std_diff = (treated_mean - control_mean) / np.sqrt(
                    (treated[cov].var() + control[cov].var()) / 2
                )

                balance_stats[cov] = {
                    "treated_mean": treated_mean,
                    "control_mean": control_mean,
                    "standardized_diff": std_diff,
                    "balanced": abs(std_diff) < 0.1,  # Common threshold
                }

        return balance_stats


def main():
    """Example usage of treatment definer"""

    # This would normally load from panel_constructor
    # Creating mock data for demonstration
    mock_data = pd.DataFrame(
        {
            "player_id": ["p1"] * 10 + ["p2"] * 10,
            "season": [2023] * 20,
            "week": list(range(1, 11)) * 2,
            "team": ["NYG"] * 10 + ["DAL"] * 10,
            "stat_yards": np.random.normal(75, 20, 20),
            "is_star": [1] * 10 + [0] * 10,
            "temp": np.random.normal(60, 20, 20),
            "wind": np.random.normal(10, 5, 20),
        }
    )

    # Skip week 5 for player 1 (injury)
    mock_data = mock_data[~((mock_data["player_id"] == "p1") & (mock_data["week"] == 5))]

    definer = TreatmentDefiner()

    # Define injury treatment
    mock_data = definer.define_injury_treatment(mock_data)

    # Define weather treatment
    mock_data = definer.define_weather_treatment(mock_data)

    # Create treatment windows
    mock_data = definer.create_treatment_windows(mock_data, "injury_treatment")

    # Get summary
    summary = definer.get_treatment_summary(mock_data)
    print("\nTreatment Summary:")
    print(summary)

    # Validate balance
    balance = definer.validate_treatment_assignment(
        mock_data, "injury_treatment", ["stat_yards", "temp", "wind"]
    )
    print("\nCovariate Balance:")
    for cov, stats in balance.items():
        print(
            f"  {cov}: Std Diff = {stats['standardized_diff']:.3f}, Balanced = {stats['balanced']}"
        )


if __name__ == "__main__":
    main()
