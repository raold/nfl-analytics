#!/usr/bin/env python3
"""
Simple Player Impact Adjustments for Injuries

Applies position-based win probability adjustments when key players are out.
Uses hardcoded impact values based on position importance.

Usage:
    python py/predict/injury_adjustments.py --season 2025 --week 5
"""
import argparse
import logging
from typing import Dict, List, Tuple
import psycopg2
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Position-based impact values (win probability deltas when starter is out)
# Based on NFL analytics research and historical data
POSITION_IMPACTS = {
    'QB': -0.050,      # Losing starting QB: -5% win probability
    'RB': -0.010,      # Losing RB1: -1%
    'WR': -0.015,      # Losing WR1: -1.5%
    'TE': -0.008,      # Losing TE1: -0.8%
    'T': -0.012,       # Losing starting tackle: -1.2%
    'G': -0.008,       # Losing starting guard: -0.8%
    'C': -0.010,       # Losing starting center: -1.0%
    'DE': -0.010,      # Losing starting DE: -1.0%
    'DT': -0.008,      # Losing starting DT: -0.8%
    'OLB': -0.008,     # Losing starting OLB: -0.8%
    'ILB': -0.008,     # Losing starting ILB: -0.8%
    'CB': -0.010,      # Losing starting CB: -1.0%
    'S': -0.008,       # Losing starting safety: -0.8%
    'FS': -0.008,
    'SS': -0.008,
}

# Depth chart importance multipliers
DEPTH_MULTIPLIERS = {
    1: 1.0,    # Starter
    2: 0.3,    # Backup (less impact if backup is out)
    3: 0.1,    # 3rd string
}


class InjuryAdjuster:
    """Adjust predictions based on player injuries."""

    def __init__(self):
        """Initialize adjuster."""
        self.db_config = {
            'dbname': 'devdb01',
            'user': 'dro',
            'password': 'sicillionbillions',
            'host': 'localhost',
            'port': 5544
        }

    def connect_db(self):
        """Create database connection."""
        return psycopg2.connect(**self.db_config)

    def get_injuries_for_week(self, season: int, week: int) -> pd.DataFrame:
        """
        Get all players listed as 'Out' for a given week.

        Args:
            season: NFL season
            week: Week number

        Returns:
            DataFrame with injury information
        """
        conn = self.connect_db()

        query = """
        SELECT
            i.season,
            i.week,
            i.team,
            i.gsis_id,
            i.full_name,
            i.position,
            i.report_status,
            i.report_primary_injury
        FROM injuries i
        WHERE i.season = %s
          AND i.week = %s
          AND i.report_status = 'Out'
        ORDER BY i.team, i.position;
        """

        df = pd.read_sql(query, conn, params=(season, week))
        conn.close()

        logger.info(f"Found {len(df)} players out for {season} Week {week}")
        return df

    def get_depth_chart_position(self, gsis_id: str, team: str, season: int, week: int) -> Tuple[str, int]:
        """
        Get player's position and depth from depth chart.

        Args:
            gsis_id: Player GSIS ID
            team: Team abbreviation
            season: Season
            week: Week

        Returns:
            (position, depth_position) tuple
        """
        conn = self.connect_db()

        query = """
        SELECT position, depth_position
        FROM depth_charts
        WHERE gsis_id = %s
          AND club_code = %s
          AND season = %s
          AND week <= %s
        ORDER BY week DESC
        LIMIT 1;
        """

        cur = conn.cursor()
        cur.execute(query, (gsis_id, team, season, week))
        result = cur.fetchone()
        cur.close()
        conn.close()

        if result:
            # Extract depth number from depth_position (e.g., "QB" -> 1, "WR2" -> 2)
            position, depth_str = result
            # Parse depth from string like "QB", "WR", "RB2", etc.
            depth = 1  # Default to starter
            if depth_str and len(depth_str) > len(position):
                try:
                    depth = int(depth_str[len(position):])
                except ValueError:
                    depth = 1
            return position, depth
        return None, None

    def calculate_team_impact(
        self,
        injuries: pd.DataFrame,
        team: str,
        season: int,
        week: int
    ) -> float:
        """
        Calculate total impact for a team based on injuries.

        Args:
            injuries: DataFrame of all injuries
            team: Team to analyze
            season: Season
            week: Week

        Returns:
            Total win probability impact (negative value)
        """
        team_injuries = injuries[injuries['team'] == team]

        if len(team_injuries) == 0:
            return 0.0

        total_impact = 0.0
        impacts_detail = []

        for _, injury in team_injuries.iterrows():
            # Get position from depth chart
            position, depth = self.get_depth_chart_position(
                injury['gsis_id'],
                team,
                season,
                week
            )

            if position is None:
                # Fallback to injury report position
                position = injury['position']
                depth = 1  # Assume starter if not in depth chart

            # Calculate impact
            base_impact = POSITION_IMPACTS.get(position, -0.005)  # Default -0.5%
            depth_multiplier = DEPTH_MULTIPLIERS.get(depth, 0.05)
            impact = base_impact * depth_multiplier

            total_impact += impact
            impacts_detail.append({
                'player': injury['full_name'],
                'position': position,
                'depth': depth,
                'impact': impact
            })

        # Log details
        if impacts_detail:
            logger.info(f"\n{team} Injury Impact:")
            for detail in impacts_detail:
                logger.info(f"  {detail['player']} ({detail['position']}{detail['depth']}): {detail['impact']:+.3f}")
            logger.info(f"  TOTAL: {total_impact:+.3f}")

        return total_impact

    def adjust_predictions(
        self,
        predictions: pd.DataFrame,
        season: int,
        week: int
    ) -> pd.DataFrame:
        """
        Adjust predictions based on injuries.

        Args:
            predictions: DataFrame with columns [home_team, away_team, base_win_prob]
            season: Season
            week: Week

        Returns:
            DataFrame with adjusted probabilities
        """
        logger.info(f"\nAdjusting predictions for injuries in {season} Week {week}...")

        # Get injuries
        injuries = self.get_injuries_for_week(season, week)

        if len(injuries) == 0:
            logger.info("No injuries reported as 'Out'")
            predictions['adjusted_home_win_prob'] = predictions['base_win_prob']
            predictions['injury_adjustment'] = 0.0
            return predictions

        # Calculate impacts for each team
        adjusted = []

        for _, pred in predictions.iterrows():
            home_team = pred['home_team']
            away_team = pred['away_team']
            base_prob = pred['base_win_prob']

            # Calculate impacts
            home_impact = self.calculate_team_impact(injuries, home_team, season, week)
            away_impact = self.calculate_team_impact(injuries, away_team, season, week)

            # Net impact (home perspective)
            # If home team has injuries: negative impact
            # If away team has injuries: positive impact for home
            net_impact = home_impact - away_impact

            # Adjust probability
            adjusted_prob = base_prob + net_impact

            # Clip to [0, 1]
            adjusted_prob = np.clip(adjusted_prob, 0.01, 0.99)

            adjusted.append({
                **pred.to_dict(),
                'home_injury_impact': home_impact,
                'away_injury_impact': away_impact,
                'net_injury_adjustment': net_impact,
                'adjusted_home_win_prob': adjusted_prob
            })

        adjusted_df = pd.DataFrame(adjusted)

        # Summary
        significant_adjustments = adjusted_df[abs(adjusted_df['net_injury_adjustment']) > 0.01]
        if len(significant_adjustments) > 0:
            logger.info(f"\n{len(significant_adjustments)} games with significant injury adjustments (>1%):")
            for _, game in significant_adjustments.iterrows():
                logger.info(f"  {game['away_team']} @ {game['home_team']}: "
                          f"{game['base_win_prob']:.1%} → {game['adjusted_home_win_prob']:.1%} "
                          f"({game['net_injury_adjustment']:+.1%})")

        return adjusted_df

    def adjust_from_csv(
        self,
        predictions_csv: str,
        season: int,
        week: int,
        output_csv: str
    ):
        """
        Load predictions from CSV, adjust for injuries, save to new CSV.

        Args:
            predictions_csv: Input predictions file
            season: Season
            week: Week
            output_csv: Output file path
        """
        logger.info(f"Loading predictions from {predictions_csv}")
        df = pd.read_csv(predictions_csv)

        # Ensure required columns
        if 'home_win_prob' in df.columns and 'base_win_prob' not in df.columns:
            df['base_win_prob'] = df['home_win_prob']

        # Adjust
        adjusted = self.adjust_predictions(df, season, week)

        # Save
        adjusted.to_csv(output_csv, index=False)
        logger.info(f"✓ Saved injury-adjusted predictions to {output_csv}")

        return adjusted


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Adjust predictions for player injuries')
    parser.add_argument('--season', type=int, required=True, help='Season')
    parser.add_argument('--week', type=int, required=True, help='Week')
    parser.add_argument('--predictions-csv', help='Input predictions CSV')
    parser.add_argument('--output', help='Output CSV path')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    adjuster = InjuryAdjuster()

    if args.predictions_csv:
        # Adjust existing predictions
        output = args.output or args.predictions_csv.replace('.csv', '_injury_adjusted.csv')
        adjusted = adjuster.adjust_from_csv(
            args.predictions_csv,
            args.season,
            args.week,
            output
        )

        print("\n" + "=" * 80)
        print("INJURY ADJUSTMENTS COMPLETE")
        print("=" * 80)
        print(f"Output: {output}")
        print(f"Games adjusted: {len(adjusted)}")
        print(f"Significant adjustments: {(abs(adjusted['net_injury_adjustment']) > 0.01).sum()}")

    else:
        # Just show injury report
        injuries = adjuster.get_injuries_for_week(args.season, args.week)
        print("\n" + "=" * 80)
        print(f"INJURY REPORT - {args.season} Week {args.week}")
        print("=" * 80)
        print(f"\nPlayers Listed as OUT: {len(injuries)}")
        print("\nBy Team:")
        for team in sorted(injuries['team'].unique()):
            team_injuries = injuries[injuries['team'] == team]
            print(f"\n{team}:")
            for _, inj in team_injuries.iterrows():
                print(f"  - {inj['full_name']} ({inj['position']}) - {inj['report_primary_injury']}")


if __name__ == '__main__':
    main()
