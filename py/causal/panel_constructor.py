#!/usr/bin/env python3
"""
Panel Data Constructor for Causal Inference

Builds player-game and team-game panel datasets suitable for causal analysis.
Includes treatment indicators, pre/post periods, and control group identification.
"""

import logging

import numpy as np
import pandas as pd
import psycopg2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PanelConstructor:
    """
    Constructs panel data for causal inference in NFL analytics.

    Panel types:
    - Player-game: Track individual players across games
    - Team-game: Track teams across games
    - Player-season: Aggregated player performance by season
    """

    def __init__(self, db_config: dict | None = None):
        """Initialize panel constructor with database connection"""
        self.db_config = db_config or {
            "host": "localhost",
            "port": 5544,
            "database": "devdb01",
            "user": "dro",
            "password": "sicillionbillions",
        }

    def build_player_game_panel(
        self, start_season: int = 2020, end_season: int = 2024, position_groups: list[str] = None
    ) -> pd.DataFrame:
        """
        Build player-game panel dataset with rich features for causal analysis.

        Returns DataFrame with:
        - Player identifiers and demographics
        - Game-level performance metrics
        - Injury/absence indicators (treatments)
        - Team context variables
        - Market expectations (spreads, totals)
        """
        conn = psycopg2.connect(**self.db_config)

        position_filter = ""
        if position_groups:
            positions = "','".join(position_groups)
            position_filter = f"AND pgs.position_group IN ('{positions}')"

        query = f"""
        WITH player_games AS (
            SELECT
                pgs.player_id,
                pgs.player_display_name as player_name,
                pgs.season,
                pgs.week,
                pgs.game_id,
                pgs.current_team as team,
                pgs.opponent,
                pgs.position,
                pgs.position_group,

                -- Performance metrics
                pgs.stat_yards,
                pgs.stat_attempts,
                pgs.stat_touchdowns,
                pgs.stat_receptions,
                pgs.stat_targets,
                pgs.fantasy_points,
                pgs.snap_count,
                pgs.snap_pct,

                -- Calculate rolling averages
                AVG(pgs.stat_yards) OVER (
                    PARTITION BY pgs.player_id
                    ORDER BY pgs.season, pgs.week
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) as avg_yards_l3,

                AVG(pgs.fantasy_points) OVER (
                    PARTITION BY pgs.player_id
                    ORDER BY pgs.season, pgs.week
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) as avg_fantasy_l3,

                -- Season-to-date stats
                AVG(pgs.stat_yards) OVER (
                    PARTITION BY pgs.player_id, pgs.season
                    ORDER BY pgs.week
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) as season_avg_yards,

                -- Career stats up to this point
                ROW_NUMBER() OVER (
                    PARTITION BY pgs.player_id
                    ORDER BY pgs.season, pgs.week
                ) as career_game_number,

                COUNT(*) OVER (
                    PARTITION BY pgs.player_id, pgs.season
                    ORDER BY pgs.week
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) as season_games_played

            FROM mart.player_game_stats pgs
            WHERE pgs.season BETWEEN {start_season} AND {end_season}
              {position_filter}
              AND pgs.stat_category = (
                  CASE
                      WHEN pgs.position_group IN ('QB') THEN 'passing'
                      WHEN pgs.position_group IN ('RB', 'FB', 'HB') THEN 'rushing'
                      WHEN pgs.position_group IN ('WR', 'TE') THEN 'receiving'
                      ELSE 'receiving'
                  END
              )
        ),
        injuries AS (
            SELECT DISTINCT
                player_id,
                season,
                week,
                1 as injury_flag,
                injury_status,
                injury_report_status
            FROM injuries
            WHERE season BETWEEN {start_season} AND {end_season}
              AND injury_status IN ('Out', 'Questionable', 'Doubtful')
        ),
        game_context AS (
            SELECT
                game_id,
                season,
                week,
                home_team,
                away_team,
                home_score,
                away_score,
                spread_close,
                total_close,
                weather_category,
                div_game,
                roof,
                surface
            FROM games
            WHERE season BETWEEN {start_season} AND {end_season}
        ),
        team_performance AS (
            SELECT
                team,
                season,
                week,
                AVG(epa) as team_epa,
                AVG(success_rate) as team_success_rate,
                SUM(CASE WHEN win = 1 THEN 1 ELSE 0 END) OVER (
                    PARTITION BY team, season
                    ORDER BY week
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) as wins_to_date,
                COUNT(*) OVER (
                    PARTITION BY team, season
                    ORDER BY week
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) as games_to_date
            FROM mart.team_game_stats
            WHERE season BETWEEN {start_season} AND {end_season}
            GROUP BY team, season, week
        )
        SELECT
            pg.*,

            -- Injury/treatment indicators
            COALESCE(i.injury_flag, 0) as injury_flag,
            i.injury_status,

            -- Identify if player missed previous game (potential treatment)
            LAG(pg.game_id) OVER (PARTITION BY pg.player_id ORDER BY pg.season, pg.week) as prev_game_id,
            CASE
                WHEN LAG(pg.week) OVER (PARTITION BY pg.player_id, pg.season ORDER BY pg.week) IS NULL
                  OR pg.week - LAG(pg.week) OVER (PARTITION BY pg.player_id, pg.season ORDER BY pg.week) > 1
                THEN 1
                ELSE 0
            END as missed_prev_game,

            -- Game context
            gc.spread_close,
            gc.total_close,
            gc.weather_category,
            gc.div_game,
            gc.roof,
            gc.surface,
            CASE
                WHEN pg.team = gc.home_team THEN gc.home_score
                ELSE gc.away_score
            END as team_score,
            CASE
                WHEN pg.team = gc.home_team THEN gc.away_score
                ELSE gc.home_score
            END as opponent_score,

            -- Team performance context
            tp.team_epa,
            tp.team_success_rate,
            COALESCE(tp.wins_to_date, 0) as team_wins_to_date,
            COALESCE(tp.games_to_date, 0) as team_games_to_date,
            CASE
                WHEN tp.games_to_date > 0 THEN tp.wins_to_date::float / tp.games_to_date
                ELSE 0.5
            END as team_win_pct_to_date,

            -- Create time index for panel
            pg.season * 100 + pg.week as time_index,

            -- Create unique panel ID
            pg.player_id || '_' || pg.season || '_' || pg.week as panel_id

        FROM player_games pg
        LEFT JOIN injuries i
            ON pg.player_id = i.player_id
            AND pg.season = i.season
            AND pg.week = i.week
        LEFT JOIN game_context gc
            ON pg.game_id = gc.game_id
        LEFT JOIN team_performance tp
            ON pg.team = tp.team
            AND pg.season = tp.season
            AND pg.week = tp.week
        ORDER BY pg.player_id, pg.season, pg.week
        """

        df = pd.read_sql(query, conn)
        conn.close()

        # Add derived features useful for causal analysis
        df = self._add_causal_features(df)

        logger.info(
            f"Built player-game panel: {len(df)} observations, {df['player_id'].nunique()} players"
        )
        return df

    def build_team_game_panel(
        self, start_season: int = 2020, end_season: int = 2024
    ) -> pd.DataFrame:
        """
        Build team-game panel dataset for team-level causal analysis.

        Useful for:
        - Coaching change effects
        - Home field advantage shifts
        - Weather impact analysis
        """
        conn = psycopg2.connect(**self.db_config)

        query = f"""
        WITH team_games AS (
            SELECT
                tgs.team,
                tgs.season,
                tgs.week,
                tgs.game_id,
                tgs.opponent,
                tgs.home,

                -- Performance metrics
                tgs.score,
                tgs.score_differential,
                tgs.total_yards,
                tgs.turnovers,
                tgs.time_of_possession_seconds,
                tgs.third_down_pct,
                tgs.red_zone_pct,
                tgs.epa,
                tgs.success_rate,
                tgs.win,

                -- Rolling performance
                AVG(tgs.score) OVER (
                    PARTITION BY tgs.team
                    ORDER BY tgs.season, tgs.week
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) as avg_score_l3,

                AVG(tgs.epa) OVER (
                    PARTITION BY tgs.team
                    ORDER BY tgs.season, tgs.week
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) as avg_epa_l3,

                -- Season performance
                SUM(tgs.win) OVER (
                    PARTITION BY tgs.team, tgs.season
                    ORDER BY tgs.week
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) as wins_to_date,

                COUNT(*) OVER (
                    PARTITION BY tgs.team, tgs.season
                    ORDER BY tgs.week
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) as games_to_date

            FROM mart.team_game_stats tgs
            WHERE tgs.season BETWEEN {start_season} AND {end_season}
        ),
        coaching_changes AS (
            -- This would come from a coaching changes table if available
            -- For now, using a placeholder
            SELECT
                'placeholder' as team,
                2023 as season,
                8 as change_week
            WHERE 1=0  -- Disabled for now
        ),
        game_context AS (
            SELECT
                game_id,
                spread_close,
                total_close,
                weather_category,
                temp,
                wind,
                div_game,
                roof,
                surface
            FROM games
            WHERE season BETWEEN {start_season} AND {end_season}
        )
        SELECT
            tg.*,

            -- Coaching change indicator (placeholder)
            CASE
                WHEN cc.change_week IS NOT NULL
                  AND tg.week >= cc.change_week
                THEN 1
                ELSE 0
            END as post_coaching_change,

            -- Game context
            gc.spread_close,
            gc.total_close,
            gc.weather_category,
            gc.temp,
            gc.wind,
            gc.div_game,
            gc.roof,
            gc.surface,

            -- Win percentage to date
            CASE
                WHEN tg.games_to_date > 0
                THEN tg.wins_to_date::float / tg.games_to_date
                ELSE 0.5
            END as win_pct_to_date,

            -- Create time index
            tg.season * 100 + tg.week as time_index,

            -- Panel ID
            tg.team || '_' || tg.season || '_' || tg.week as panel_id

        FROM team_games tg
        LEFT JOIN coaching_changes cc
            ON tg.team = cc.team
            AND tg.season = cc.season
        LEFT JOIN game_context gc
            ON tg.game_id = gc.game_id
        ORDER BY tg.team, tg.season, tg.week
        """

        df = pd.read_sql(query, conn)
        conn.close()

        logger.info(f"Built team-game panel: {len(df)} observations, {df['team'].nunique()} teams")
        return df

    def _add_causal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features useful for causal inference"""

        # Create treatment indicators
        df["games_missed"] = df.groupby("player_id")["missed_prev_game"].cumsum()

        # Identify "star" players (top performers who matter more)
        season_stats = df.groupby(["player_id", "season"])["stat_yards"].mean()
        top_percentile = season_stats.quantile(0.8)
        star_players = (
            season_stats[season_stats > top_percentile].index.get_level_values(0).unique()
        )
        df["is_star"] = df["player_id"].isin(star_players).astype(int)

        # Create pre/post treatment periods for injuries
        df["post_injury"] = df.groupby("player_id")["injury_flag"].cumsum() > 0

        # Market expectation vs actual (useful for identifying shocks)
        df["score_vs_spread"] = (df["team_score"] - df["opponent_score"]) - df["spread_close"]
        df["total_vs_ou"] = (df["team_score"] + df["opponent_score"]) - df["total_close"]

        # Experience/tenure features
        df["is_rookie"] = df["career_game_number"] <= 16
        df["is_veteran"] = df["career_game_number"] > 48

        return df

    def create_matched_pairs(
        self, df: pd.DataFrame, treatment_col: str, match_cols: list[str], caliper: float = 0.1
    ) -> pd.DataFrame:
        """
        Create matched pairs for causal analysis using propensity score matching.

        Args:
            df: Panel dataframe
            treatment_col: Column indicating treatment (0/1)
            match_cols: Columns to use for matching
            caliper: Maximum distance for matching

        Returns:
            DataFrame with matched pairs
        """
        from scipy.spatial.distance import cdist
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        # Prepare data
        X = df[match_cols].fillna(0)
        y = df[treatment_col]

        # Estimate propensity scores
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_scaled, y)

        propensity_scores = model.predict_proba(X_scaled)[:, 1]
        df["propensity_score"] = propensity_scores

        # Split into treatment and control
        treated = df[df[treatment_col] == 1].copy()
        control = df[df[treatment_col] == 0].copy()

        # Match each treated unit to closest control
        matches = []
        for idx, treated_row in treated.iterrows():
            # Find controls within caliper
            ps_dist = np.abs(control["propensity_score"] - treated_row["propensity_score"])
            valid_controls = control[ps_dist < caliper]

            if len(valid_controls) > 0:
                # Find closest match
                distances = cdist(
                    [treated_row[match_cols].fillna(0).values],
                    valid_controls[match_cols].fillna(0).values,
                )
                best_match_idx = valid_controls.iloc[distances.argmin()].name

                matches.append(
                    {
                        "treated_id": treated_row["panel_id"],
                        "control_id": control.loc[best_match_idx, "panel_id"],
                        "treated_idx": idx,
                        "control_idx": best_match_idx,
                        "ps_distance": ps_dist.loc[best_match_idx],
                    }
                )

        matches_df = pd.DataFrame(matches)

        logger.info(f"Created {len(matches)} matched pairs from {len(treated)} treated units")
        return matches_df

    def save_panel(self, df: pd.DataFrame, filename: str):
        """Save panel dataset to parquet for efficient storage"""
        output_path = f"data/panels/{filename}"
        df.to_parquet(output_path, compression="snappy")
        logger.info(f"Saved panel to {output_path}")

    def load_panel(self, filename: str) -> pd.DataFrame:
        """Load saved panel dataset"""
        input_path = f"data/panels/{filename}"
        return pd.read_parquet(input_path)


def main():
    """Example usage of panel constructor"""

    constructor = PanelConstructor()

    # Build player-game panel for RBs and WRs
    logger.info("Building player-game panel for skill positions...")
    player_panel = constructor.build_player_game_panel(
        start_season=2020, end_season=2024, position_groups=["RB", "WR", "TE"]
    )

    logger.info(f"Panel shape: {player_panel.shape}")
    logger.info(f"Columns: {list(player_panel.columns)[:10]}...")

    # Example: Create matched pairs for injury analysis
    injury_matches = constructor.create_matched_pairs(
        player_panel[player_panel["is_star"] == 1],  # Only star players
        treatment_col="injury_flag",
        match_cols=["avg_yards_l3", "season_avg_yards", "team_win_pct_to_date", "spread_close"],
    )

    logger.info(f"Created {len(injury_matches)} matched pairs for injury analysis")

    # Build team panel
    logger.info("\nBuilding team-game panel...")
    team_panel = constructor.build_team_game_panel()
    logger.info(f"Team panel shape: {team_panel.shape}")


if __name__ == "__main__":
    main()
