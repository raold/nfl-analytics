#!/usr/bin/env python3
"""
Opponent-Specific Matchup Features

Computes features for specific team matchups:
1. Divisional rival indicators
2. Historical head-to-head records
3. Recent matchup performance
4. Conference/division indicators

Usage:
    python py/features/matchup_features.py --output data/processed/features/matchup_features.csv
"""
import argparse
import logging

import pandas as pd
import psycopg2

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# NFL Divisions (2002-present alignment)
DIVISIONS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"],
    "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"],  # Historically JAC
    "AFC West": ["DEN", "KC", "LV", "LAC"],  # Historically OAK, SD
    "NFC East": ["DAL", "NYG", "PHI", "WAS"],
    "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"],
    "NFC West": ["ARI", "LA", "SF", "SEA"],  # Historically STL
}

# Legacy team codes
TEAM_ALIASES = {"JAC": "JAX", "OAK": "LV", "SD": "LAC", "STL": "LA"}


class MatchupFeatureGenerator:
    """Generate matchup-specific features."""

    def __init__(self):
        """Initialize generator."""
        self.db_config = {
            "dbname": "devdb01",
            "user": "dro",
            "password": "sicillionbillions",
            "host": "localhost",
            "port": 5544,
        }

    def connect_db(self):
        """Create database connection."""
        return psycopg2.connect(**self.db_config)

    def normalize_team(self, team: str) -> str:
        """Normalize team code to current name."""
        return TEAM_ALIASES.get(team, team)

    def get_division(self, team: str) -> str:
        """Get division for team."""
        team = self.normalize_team(team)
        for div_name, teams in DIVISIONS.items():
            if team in teams:
                return div_name
        return "Unknown"

    def is_divisional_matchup(self, home_team: str, away_team: str) -> bool:
        """Check if matchup is divisional."""
        home_div = self.get_division(home_team)
        away_div = self.get_division(away_team)
        return home_div == away_div and home_div != "Unknown"

    def is_conference_matchup(self, home_team: str, away_team: str) -> bool:
        """Check if matchup is within same conference."""
        home_div = self.get_division(home_team)
        away_div = self.get_division(away_team)
        home_conf = home_div.split()[0] if home_div != "Unknown" else "Unknown"
        away_conf = away_div.split()[0] if away_div != "Unknown" else "Unknown"
        return home_conf == away_conf and home_conf != "Unknown"

    def compute_matchup_features(self) -> pd.DataFrame:
        """
        Compute matchup features for all games.

        Returns:
            DataFrame with matchup features
        """
        logger.info("Computing matchup features...")

        conn = self.connect_db()

        query = """
        WITH game_list AS (
            SELECT
                game_id,
                season,
                week,
                game_type,
                home_team,
                away_team,
                home_score,
                away_score,
                kickoff
            FROM games
            WHERE season >= 2006
              AND game_type = 'REG'
            ORDER BY season, week
        ),
        historical_matchups AS (
            -- Historical head-to-head record (prior games only)
            SELECT
                g1.game_id,
                COUNT(g2.game_id) as h2h_games_played,
                COUNT(CASE WHEN g2.home_team = g1.home_team AND g2.home_score > g2.away_score THEN 1
                           WHEN g2.away_team = g1.home_team AND g2.away_score > g2.home_score THEN 1
                      END) as h2h_home_team_wins,
                COUNT(CASE WHEN g2.home_team = g1.away_team AND g2.home_score > g2.away_score THEN 1
                           WHEN g2.away_team = g1.away_team AND g2.away_score > g2.home_score THEN 1
                      END) as h2h_away_team_wins,
                AVG(CASE WHEN g2.home_team = g1.home_team THEN g2.home_score - g2.away_score
                         WHEN g2.away_team = g1.home_team THEN g2.away_score - g2.home_score
                    END) as h2h_home_team_avg_margin,
                AVG(g2.home_score + g2.away_score) as h2h_avg_total
            FROM game_list g1
            LEFT JOIN game_list g2 ON (
                (g2.home_team = g1.home_team AND g2.away_team = g1.away_team) OR
                (g2.home_team = g1.away_team AND g2.away_team = g1.home_team)
            )
            AND g2.kickoff < g1.kickoff
            AND g2.home_score IS NOT NULL
            AND g2.season >= g1.season - 5  -- Last 5 seasons only
            GROUP BY g1.game_id
        ),
        recent_matchups AS (
            -- Last 3 meetings
            SELECT
                g1.game_id,
                COUNT(CASE WHEN g2.home_team = g1.home_team AND g2.home_score > g2.away_score THEN 1
                           WHEN g2.away_team = g1.home_team AND g2.away_score > g2.home_score THEN 1
                      END) as h2h_l3_home_wins,
                AVG(CASE WHEN g2.home_team = g1.home_team THEN g2.home_score - g2.away_score
                         WHEN g2.away_team = g1.home_team THEN g2.away_score - g2.home_score
                    END) as h2h_l3_home_margin
            FROM game_list g1
            LEFT JOIN LATERAL (
                SELECT *
                FROM game_list g2
                WHERE ((g2.home_team = g1.home_team AND g2.away_team = g1.away_team) OR
                       (g2.home_team = g1.away_team AND g2.away_team = g1.home_team))
                  AND g2.kickoff < g1.kickoff
                  AND g2.home_score IS NOT NULL
                ORDER BY g2.kickoff DESC
                LIMIT 3
            ) g2 ON TRUE
            GROUP BY g1.game_id
        )
        SELECT
            gl.*,
            COALESCE(hm.h2h_games_played, 0) as h2h_games_played,
            COALESCE(hm.h2h_home_team_wins, 0) as h2h_home_wins,
            COALESCE(hm.h2h_away_team_wins, 0) as h2h_away_wins,
            COALESCE(hm.h2h_home_team_avg_margin, 0) as h2h_home_avg_margin,
            COALESCE(hm.h2h_avg_total, 0) as h2h_avg_total,
            COALESCE(rm.h2h_l3_home_wins, 0) as h2h_l3_home_wins,
            COALESCE(rm.h2h_l3_home_margin, 0) as h2h_l3_home_margin
        FROM game_list gl
        LEFT JOIN historical_matchups hm ON gl.game_id = hm.game_id
        LEFT JOIN recent_matchups rm ON gl.game_id = rm.game_id
        ORDER BY gl.season, gl.week;
        """

        logger.info("Executing matchup query...")
        df = pd.read_sql(query, conn)
        conn.close()

        logger.info(f"Loaded {len(df)} games")

        # Add division/conference indicators
        logger.info("Computing division/conference indicators...")
        df["is_divisional"] = df.apply(
            lambda row: self.is_divisional_matchup(row["home_team"], row["away_team"]), axis=1
        ).astype(int)

        df["is_conference"] = df.apply(
            lambda row: self.is_conference_matchup(row["home_team"], row["away_team"]), axis=1
        ).astype(int)

        df["home_division"] = df["home_team"].apply(self.get_division)
        df["away_division"] = df["away_team"].apply(self.get_division)

        # Compute derived features
        df["h2h_home_win_pct"] = df["h2h_home_wins"] / df["h2h_games_played"].replace(0, 1)
        df["h2h_l3_home_win_pct"] = df["h2h_l3_home_wins"] / 3.0

        # Rivalry intensity (more games = bigger rivalry)
        df["rivalry_intensity"] = df["h2h_games_played"] / 5.0  # Normalize to 5 years

        logger.info("\nMatchup Feature Statistics:")
        logger.info(
            f"  Divisional games: {df['is_divisional'].sum()} ({df['is_divisional'].mean():.1%})"
        )
        logger.info(
            f"  Conference games: {df['is_conference'].sum()} ({df['is_conference'].mean():.1%})"
        )
        logger.info(
            f"  Games with H2H history: {(df['h2h_games_played'] > 0).sum()} ({(df['h2h_games_played'] > 0).mean():.1%})"
        )

        return df

    def save_features(self, df: pd.DataFrame, output_path: str):
        """Save matchup features to CSV."""
        logger.info(f"Saving matchup features to {output_path}")
        df.to_csv(output_path, index=False)
        logger.info(f"✓ Saved {len(df)} rows with {len(df.columns)} columns")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate matchup features")
    parser.add_argument("--output", default="data/processed/features/matchup_features.csv")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Generate features
    generator = MatchupFeatureGenerator()
    df = generator.compute_matchup_features()
    generator.save_features(df, args.output)

    print("\n" + "=" * 80)
    print("MATCHUP FEATURES GENERATION COMPLETE")
    print("=" * 80)
    print(f"Output: {args.output}")
    print(f"Games: {len(df)}")
    print(f"Features: {len(df.columns)}")


if __name__ == "__main__":
    main()
