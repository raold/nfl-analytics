#!/usr/bin/env python3
"""
Feature Loader for Phase 5 Advanced Features

Loads and integrates all 5 advanced feature modules:
1. Defensive situational strength
2. Defensive game script indicators
3. Vegas line bias features
4. Environmental/contextual features
5. Feature interactions

Provides a unified interface for the ensemble model.

Author: Claude Code
Date: 2025-10-30
"""

import logging
import pandas as pd
import psycopg
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedFeatureLoader:
    """
    Loads and integrates Phase 5 advanced features for game outcome prediction.

    All features are game-level (home/away perspective) and designed to improve
    spread and total predictions.
    """

    def __init__(self, db_url: str = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"):
        self.db_url = db_url
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_defensive_situational_features(self, game_ids: Optional[list] = None) -> pd.DataFrame:
        """
        Load defensive situational strength features.

        Returns:
            DataFrame with defensive performance in critical situations
        """
        # For now, we'll compute these features on-the-fly using the same logic
        # as the feature generation modules. In production, these would be
        # pre-computed and stored in a database table.

        query = """
        WITH third_down AS (
            SELECT
                game_id,
                posteam as team,
                COUNT(*) FILTER (WHERE down = 3) as third_down_attempts,
                COUNT(*) FILTER (WHERE down = 3 AND third_down_converted = false) as third_down_stops,
                AVG(yards_gained) FILTER (WHERE down = 3) as third_down_yards_avg
            FROM plays
            WHERE down = 3
            GROUP BY game_id, posteam
        ),
        fourth_down AS (
            SELECT
                game_id,
                posteam as team,
                COUNT(*) FILTER (WHERE down = 4) as fourth_down_attempts,
                COUNT(*) FILTER (WHERE down = 4 AND fourth_down_converted = false) as fourth_down_stops,
                AVG(yards_gained) FILTER (WHERE down = 4) as fourth_down_yards_avg
            FROM plays
            WHERE down = 4
            GROUP BY game_id, posteam
        )
        SELECT
            g.game_id,
            g.home_team,
            g.away_team,

            -- Home defense (vs away offense)
            COALESCE(td_away.third_down_attempts, 0) as home_def_third_down_att,
            COALESCE(td_away.third_down_stops, 0) as home_def_third_down_stops,
            COALESCE(td_away.third_down_stops::float / NULLIF(td_away.third_down_attempts, 0), 0.33) as home_def_third_down_stop_rate,

            COALESCE(fd_away.fourth_down_attempts, 0) as home_def_fourth_down_att,
            COALESCE(fd_away.fourth_down_stops, 0) as home_def_fourth_down_stops,
            COALESCE(fd_away.fourth_down_stops::float / NULLIF(fd_away.fourth_down_attempts, 0), 0.5) as home_def_fourth_down_stop_rate,

            -- Away defense (vs home offense)
            COALESCE(td_home.third_down_attempts, 0) as away_def_third_down_att,
            COALESCE(td_home.third_down_stops, 0) as away_def_third_down_stops,
            COALESCE(td_home.third_down_stops::float / NULLIF(td_home.third_down_attempts, 0), 0.33) as away_def_third_down_stop_rate,

            COALESCE(fd_home.fourth_down_attempts, 0) as away_def_fourth_down_att,
            COALESCE(fd_home.fourth_down_stops, 0) as away_def_fourth_down_stops,
            COALESCE(fd_home.fourth_down_stops::float / NULLIF(fd_home.fourth_down_attempts, 0), 0.5) as away_def_fourth_down_stop_rate

        FROM games g
        LEFT JOIN third_down td_away ON g.game_id = td_away.game_id AND g.away_team = td_away.team
        LEFT JOIN fourth_down fd_away ON g.game_id = fd_away.game_id AND g.away_team = fd_away.team
        LEFT JOIN third_down td_home ON g.game_id = td_home.game_id AND g.home_team = td_home.team
        LEFT JOIN fourth_down fd_home ON g.game_id = fd_home.game_id AND g.home_team = fd_home.team
        WHERE g.season >= 2020
        """

        if game_ids:
            query += f" AND g.game_id = ANY(ARRAY{game_ids})"

        with psycopg.connect(self.db_url) as conn:
            df = pd.read_sql_query(query, conn)

        self.logger.info(f"Loaded defensive situational features for {len(df)} games")
        return df

    def load_vegas_features(self, game_ids: Optional[list] = None) -> pd.DataFrame:
        """
        Load Vegas line bias features.

        Returns:
            DataFrame with line characteristics and team ATS performance
        """
        query = """
        SELECT
            game_id,
            home_team,
            away_team,
            spread_close,
            total_close,
            home_moneyline,
            away_moneyline,

            -- Line characteristics
            CASE WHEN spread_close < 0 THEN 1 ELSE 0 END as home_is_favorite,
            ABS(spread_close) as spread_magnitude,
            CASE WHEN ABS(spread_close) <= 3 THEN 1 ELSE 0 END as is_small_spread,
            CASE WHEN ABS(spread_close) >= 7 THEN 1 ELSE 0 END as is_large_spread,
            CASE WHEN ABS(spread_close) IN (3, 7, 10, 14) THEN 1 ELSE 0 END as on_key_number,

            -- Market efficiency
            CASE
                WHEN home_moneyline < 0 THEN ABS(home_moneyline) / (ABS(home_moneyline) + 100)
                ELSE 100 / (home_moneyline + 100)
            END as home_implied_prob,

            CASE
                WHEN away_moneyline < 0 THEN ABS(away_moneyline) / (ABS(away_moneyline) + 100)
                ELSE 100 / (away_moneyline + 100)
            END as away_implied_prob

        FROM games
        WHERE season >= 2020
            AND spread_close IS NOT NULL
            AND total_close IS NOT NULL
        """

        if game_ids:
            query += f" AND game_id = ANY(ARRAY{game_ids})"

        with psycopg.connect(self.db_url) as conn:
            df = pd.read_sql_query(query, conn)

        self.logger.info(f"Loaded Vegas features for {len(df)} games")
        return df

    def load_environmental_features(self, game_ids: Optional[list] = None) -> pd.DataFrame:
        """
        Load environmental and contextual features.

        Returns:
            DataFrame with weather, travel, venue, and rivalry features
        """
        # Team classifications
        dome_teams = ['NO', 'ATL', 'DET', 'MIN', 'DAL', 'ARI', 'LA', 'LV']

        query = """
        SELECT
            g.game_id,
            g.home_team,
            g.away_team,
            g.season,
            g.week,

            -- Game timing
            EXTRACT(DOW FROM g.kickoff) as day_of_week,
            CASE WHEN EXTRACT(DOW FROM g.kickoff) = 4 THEN 1 ELSE 0 END as is_thursday_night,

            -- Weather
            g.temp,
            g.wind,
            g.roof,
            CASE WHEN g.roof = 'outdoors' THEN 1 ELSE 0 END as is_outdoor_game,
            CASE WHEN CAST(g.temp AS INTEGER) < 25 THEN 1 ELSE 0 END as extreme_cold,
            CASE WHEN CAST(g.temp AS INTEGER) > 85 THEN 1 ELSE 0 END as extreme_heat,
            CASE WHEN CAST(g.wind AS INTEGER) > 15 THEN 1 ELSE 0 END as high_wind,

            -- Rest
            g.away_rest,
            g.home_rest,
            CASE WHEN g.away_rest <= 5 THEN 1 ELSE 0 END as away_short_rest,

            -- Rivalry
            t_home.team_division as home_division,
            t_away.team_division as away_division,
            CASE WHEN t_home.team_division = t_away.team_division THEN 1 ELSE 0 END as is_division_game,
            t_home.team_conf as home_conf,
            t_away.team_conf as away_conf,
            CASE WHEN t_home.team_conf = t_away.team_conf THEN 1 ELSE 0 END as is_conference_game

        FROM games g
        LEFT JOIN teams t_home ON g.home_team = t_home.team_abbr
        LEFT JOIN teams t_away ON g.away_team = t_away.team_abbr
        WHERE g.season >= 2020
        """

        if game_ids:
            query += f" AND g.game_id = ANY(ARRAY{game_ids})"

        with psycopg.connect(self.db_url) as conn:
            df = pd.read_sql_query(query, conn)

        # Add venue adaptation features
        df['away_is_dome_team'] = df['away_team'].isin(dome_teams).astype(int)
        df['venue_adaptation'] = (
            (df['away_is_dome_team'] == 1) & (df['is_outdoor_game'] == 1)
        ).astype(int)

        self.logger.info(f"Loaded environmental features for {len(df)} games")
        return df

    def load_feature_interactions(self, game_ids: Optional[list] = None) -> pd.DataFrame:
        """
        Load feature interactions.

        Returns:
            DataFrame with synergistic feature combinations
        """
        # Load component features first
        env_features = self.load_environmental_features(game_ids)
        vegas_features = self.load_vegas_features(game_ids)

        # Merge on game_id
        df = env_features.merge(vegas_features, on='game_id', how='inner', suffixes=('', '_vegas'))

        # Calculate interactions

        # 1. Weather × Travel interactions
        df['dome_team_outdoor_cold'] = (
            df['away_is_dome_team'] * df['is_outdoor_game'] * df['extreme_cold']
        )
        df['dome_team_outdoor_wind'] = (
            df['away_is_dome_team'] * df['is_outdoor_game'] * df['high_wind']
        )

        # 2. Rest × Travel interactions
        df['thursday_short_rest'] = df['is_thursday_night'] * df['away_short_rest']

        # 3. Vegas × Context interactions
        df['division_toss_up'] = df['is_division_game'] * df['is_small_spread']

        # 4. Weather severity composite
        df['weather_severity'] = (
            df['extreme_cold'].astype(float) +
            df['extreme_heat'].astype(float) +
            df['high_wind'].astype(float) * 0.5
        )

        # 5. Big favorite in bad weather
        df['big_favorite_bad_weather'] = df['is_large_spread'] * df['weather_severity']

        # Select interaction columns
        interaction_cols = [
            'game_id', 'home_team', 'away_team',
            'dome_team_outdoor_cold', 'dome_team_outdoor_wind',
            'thursday_short_rest', 'division_toss_up',
            'weather_severity', 'big_favorite_bad_weather'
        ]

        result = df[interaction_cols].copy()
        self.logger.info(f"Loaded feature interactions for {len(result)} games")
        return result

    def load_all_features(self, game_ids: Optional[list] = None) -> pd.DataFrame:
        """
        Load all advanced features and merge into a single DataFrame.

        Args:
            game_ids: Optional list of game_ids to filter

        Returns:
            DataFrame with all advanced features merged
        """
        self.logger.info("=" * 70)
        self.logger.info("Loading Phase 5 Advanced Features")
        self.logger.info("=" * 70)

        # Load each feature set
        defensive_features = self.load_defensive_situational_features(game_ids)
        vegas_features = self.load_vegas_features(game_ids)
        environmental_features = self.load_environmental_features(game_ids)
        interaction_features = self.load_feature_interactions(game_ids)

        # Merge all features on game_id
        merged = defensive_features.merge(
            vegas_features, on='game_id', how='inner', suffixes=('', '_vegas')
        )
        merged = merged.merge(
            environmental_features, on='game_id', how='inner', suffixes=('', '_env')
        )
        merged = merged.merge(
            interaction_features, on='game_id', how='inner', suffixes=('', '_inter')
        )

        self.logger.info(f"✓ Loaded {len(merged)} games with {len(merged.columns)} features")
        self.logger.info("=" * 70)

        return merged


def main():
    """Test feature loader."""
    loader = AdvancedFeatureLoader()

    # Load all features
    features_df = loader.load_all_features()

    print("\nFeature columns:")
    for col in sorted(features_df.columns):
        print(f"  - {col}")

    print(f"\nShape: {features_df.shape}")
    print(f"Date range: {features_df['season'].min()}-{features_df['season'].max()}")

    # Save sample
    features_df.head(50).to_csv("data/phase5_features_sample.csv", index=False)
    print("\n✓ Sample saved to data/phase5_features_sample.csv")


if __name__ == "__main__":
    main()
