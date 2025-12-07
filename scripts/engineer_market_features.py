#!/usr/bin/env python3
"""
Market Microstructure Feature Engineering for NFL Betting Models.

This script engineers advanced market features from historical betting odds data
to capture sharp money signals and market uncertainty. These features are critical
for bet selection and expected value estimation.

Features Engineered:
1. Line Movement Velocity - Rate of line change (sharp money indicator)
2. Line Movement Acceleration - Second derivative of line movement
3. Cross-Book Dispersion - Variance across sportsbooks (uncertainty measure)
4. Time-to-Kickoff Decay - Lines sharpen as kickoff approaches
5. Sharp Move Detection - Binary flags for significant line movements
6. Consensus vs Outlier Lines - Deviation from consensus

Expected ROI Impact: +1-2% via improved bet selection (avoid high-variance games)

Environment Variables:
    DB_HOST - PostgreSQL host (default: localhost)
    DB_PORT - PostgreSQL port (default: 5544)
    DB_NAME - Database name (default: devdb01)
    DB_USER - Database user (default: dro)
    DB_PASSWORD - Database password

Usage:
    # Engineer features for 2024 season
    python scripts/engineer_market_features.py --season 2024

    # Custom date range
    python scripts/engineer_market_features.py --start-date 2024-09-01 --end-date 2025-01-31

    # Save to custom output file
    python scripts/engineer_market_features.py --season 2024 --output data/market_features_2024.csv

    # Enable validation checks
    python scripts/engineer_market_features.py --season 2024 --validate

Author: Claude Code (Sonnet 4.5)
Date: November 16, 2025
Version: 1.0
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# ============================================================================
# Logging Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Database Configuration
# ============================================================================


class DatabaseConfig:
    """Database configuration from environment."""

    def __init__(self):
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = int(os.getenv("DB_PORT", 5544))
        self.name = os.getenv("DB_NAME", "devdb01")
        self.user = os.getenv("DB_USER", "dro")
        self.password = os.getenv("DB_PASSWORD", "sicillionbillions")

    def get_connection_string(self) -> str:
        """Get PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    def get_connection(self) -> psycopg2.extensions.connection:
        """Get database connection."""
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=self.name,
            user=self.user,
            password=self.password,
        )


# ============================================================================
# Market Microstructure Feature Engineering
# ============================================================================


class MarketFeatureEngineer:
    """
    Engineer market microstructure features from historical odds data.

    Key Features:
    1. Line Movement Velocity (sharp money signal)
    2. Line Movement Acceleration (momentum detection)
    3. Cross-Book Dispersion (uncertainty measure)
    4. Time-to-Kickoff Decay (market efficiency)
    5. Sharp Move Detection (binary flags)
    """

    def __init__(self, db_config: DatabaseConfig):
        """Initialize feature engineer."""
        self.db_config = db_config
        self.conn = None

    def __enter__(self):
        """Context manager entry."""
        self.conn = self.db_config.get_connection()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.conn:
            self.conn.close()

    def fetch_historical_odds(
        self,
        start_date: str,
        end_date: str,
        markets: list[str] = None,
        min_bookmakers: int = 5,
    ) -> pd.DataFrame:
        """
        Fetch historical odds data from database.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            markets: List of markets to fetch (default: spreads, totals)
            min_bookmakers: Minimum bookmakers required per game

        Returns:
            DataFrame with historical odds data
        """
        if markets is None:
            markets = ["spreads", "totals"]

        logger.info(f"Fetching odds data from {start_date} to {end_date}")
        logger.info(f"Markets: {markets}")
        logger.info(f"Minimum bookmakers: {min_bookmakers}")

        query = """
        SELECT
            event_id,
            sport_key,
            commence_time,
            home_team,
            away_team,
            bookmaker_key,
            bookmaker_title,
            market_key,
            outcome_name,
            outcome_price,
            outcome_point,
            snapshot_at,
            market_last_update
        FROM odds_history
        WHERE snapshot_at >= %s
          AND snapshot_at <= %s
          AND market_key = ANY(%s)
          AND outcome_point IS NOT NULL  -- Filter out null point values
        ORDER BY event_id, market_key, bookmaker_key, snapshot_at
        """

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (start_date, end_date, markets))
            rows = cur.fetchall()

        if not rows:
            logger.warning("No data found for specified date range")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        logger.info(f"Fetched {len(df):,} odds records")
        logger.info(f"Unique events: {df['event_id'].nunique():,}")
        logger.info(f"Unique bookmakers: {df['bookmaker_key'].nunique()}")
        logger.info(f"Date range: {df['snapshot_at'].min()} to {df['snapshot_at'].max()}")

        # Filter to events with minimum bookmaker coverage
        event_bookmaker_counts = (
            df.groupby(["event_id", "market_key"])["bookmaker_key"]
            .nunique()
            .reset_index(name="bookmaker_count")
        )
        valid_events = event_bookmaker_counts[
            event_bookmaker_counts["bookmaker_count"] >= min_bookmakers
        ][["event_id", "market_key"]]

        df = df.merge(valid_events, on=["event_id", "market_key"], how="inner")
        logger.info(f"After filtering: {len(df):,} records, {df['event_id'].nunique():,} events")

        return df

    def calculate_line_movement_velocity(
        self, df: pd.DataFrame, window_hours: float = 1.0
    ) -> pd.DataFrame:
        """
        Calculate line movement velocity (points per hour).

        Velocity indicates sharp money moving the line. High velocity (>1.5 points/hour)
        is a strong signal of informed betting.

        Args:
            df: DataFrame with odds data
            window_hours: Time window for velocity calculation (hours)

        Returns:
            DataFrame with velocity features
        """
        logger.info(f"Calculating line movement velocity (window: {window_hours}h)")

        # Sort by event, market, bookmaker, and time
        df = df.sort_values(["event_id", "market_key", "bookmaker_key", "snapshot_at"])

        # Calculate time delta in hours
        df["time_delta"] = (
            df.groupby(["event_id", "market_key", "bookmaker_key"])["snapshot_at"]
            .diff()
            .dt.total_seconds()
            / 3600
        )

        # Calculate line change
        df["line_delta"] = df.groupby(["event_id", "market_key", "bookmaker_key"])[
            "outcome_point"
        ].diff()

        # Calculate velocity (points per hour)
        df["line_velocity"] = np.where(df["time_delta"] > 0, df["line_delta"] / df["time_delta"], 0)

        # Calculate absolute velocity
        df["line_velocity_abs"] = df["line_velocity"].abs()

        # Calculate rolling average velocity (smoothed)
        df["line_velocity_rolling_avg"] = (
            df.groupby(["event_id", "market_key", "bookmaker_key"])["line_velocity"]
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(level=[0, 1, 2], drop=True)
        )

        return df

    def calculate_line_movement_acceleration(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate line movement acceleration (second derivative).

        Acceleration detects momentum shifts. Positive acceleration indicates
        accelerating line movement (increasing sharp action).

        Args:
            df: DataFrame with velocity already calculated

        Returns:
            DataFrame with acceleration features
        """
        logger.info("Calculating line movement acceleration")

        # Calculate acceleration (change in velocity over time)
        df["velocity_delta"] = df.groupby(["event_id", "market_key", "bookmaker_key"])[
            "line_velocity"
        ].diff()

        df["line_acceleration"] = np.where(
            df["time_delta"] > 0, df["velocity_delta"] / df["time_delta"], 0
        )

        df["line_acceleration_abs"] = df["line_acceleration"].abs()

        return df

    def calculate_cross_book_dispersion(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cross-bookmaker dispersion for each event at each timestamp.

        High dispersion (std > 1.5 points) indicates market uncertainty and should
        trigger bet avoidance. Low dispersion indicates sharp, efficient markets.

        Args:
            df: DataFrame with odds data

        Returns:
            DataFrame with dispersion features
        """
        logger.info("Calculating cross-bookmaker dispersion")

        # Group by event, market, outcome, and snapshot time
        dispersion = (
            df.groupby(["event_id", "market_key", "outcome_name", "snapshot_at"])["outcome_point"]
            .agg(
                cross_book_std=("std"),
                cross_book_median=("median"),
                cross_book_mean=("mean"),
                cross_book_min=("min"),
                cross_book_max=("max"),
                cross_book_count=("count"),
            )
            .reset_index()
        )

        # Calculate range
        dispersion["cross_book_range"] = dispersion["cross_book_max"] - dispersion["cross_book_min"]

        # Merge back to main dataframe
        df = df.merge(
            dispersion,
            on=["event_id", "market_key", "outcome_name", "snapshot_at"],
            how="left",
        )

        # Create high dispersion flag (std > 1.5 points)
        df["high_dispersion"] = (df["cross_book_std"] > 1.5).astype(int)

        return df

    def calculate_time_to_kickoff(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate time to kickoff in hours.

        Lines become sharper (more efficient) as kickoff approaches. Time-to-kickoff
        is a critical feature for weighting line movements.

        Args:
            df: DataFrame with odds data

        Returns:
            DataFrame with time-to-kickoff features
        """
        logger.info("Calculating time-to-kickoff features")

        # Calculate hours to kickoff
        df["hours_to_kickoff"] = (df["commence_time"] - df["snapshot_at"]).dt.total_seconds() / 3600

        # Create time buckets (e.g., <1h, 1-6h, 6-24h, 24-48h, >48h)
        df["time_bucket"] = pd.cut(
            df["hours_to_kickoff"],
            bins=[-np.inf, 1, 6, 24, 48, np.inf],
            labels=["<1h", "1-6h", "6-24h", "24-48h", ">48h"],
        )

        # Calculate decay weight (lines within 6h of kickoff get full weight)
        df["time_decay_weight"] = np.where(
            df["hours_to_kickoff"] <= 6, 1.0, np.exp(-df["hours_to_kickoff"] / 24)
        )

        return df

    def detect_sharp_moves(
        self, df: pd.DataFrame, velocity_threshold: float = 1.5, window_minutes: int = 10
    ) -> pd.DataFrame:
        """
        Detect sharp line movements (rapid changes indicating informed betting).

        A sharp move is defined as:
        - Line movement > velocity_threshold points per hour
        - Within a short time window (default 10 minutes)

        Args:
            df: DataFrame with velocity already calculated
            velocity_threshold: Minimum velocity for sharp move (points/hour)
            window_minutes: Time window for detection (minutes)

        Returns:
            DataFrame with sharp move flags
        """
        logger.info(
            f"Detecting sharp moves (velocity > {velocity_threshold} pts/h, "
            f"window: {window_minutes} min)"
        )

        # Flag sharp moves based on velocity and time window
        window_hours = window_minutes / 60
        df["sharp_move"] = (
            (df["line_velocity_abs"] >= velocity_threshold) & (df["time_delta"] <= window_hours)
        ).astype(int)

        # Count sharp moves per event
        sharp_move_counts = (
            df.groupby(["event_id", "market_key"])["sharp_move"]
            .sum()
            .reset_index(name="sharp_move_count")
        )

        df = df.merge(sharp_move_counts, on=["event_id", "market_key"], how="left")

        # Flag events with multiple sharp moves (high signal strength)
        df["multiple_sharp_moves"] = (df["sharp_move_count"] >= 2).astype(int)

        return df

    def calculate_consensus_deviation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate deviation from consensus line.

        Identifies outlier bookmakers (off-market lines) which may indicate
        stale lines or sharp book positioning.

        Args:
            df: DataFrame with cross-book statistics already calculated

        Returns:
            DataFrame with consensus deviation features
        """
        logger.info("Calculating consensus deviation")

        # Calculate deviation from median (robust to outliers)
        df["consensus_deviation"] = df["outcome_point"] - df["cross_book_median"]

        # Calculate deviation from mean
        df["consensus_deviation_mean"] = df["outcome_point"] - df["cross_book_mean"]

        # Calculate absolute deviation
        df["consensus_deviation_abs"] = df["consensus_deviation"].abs()

        # Flag outlier lines (>2 std from median)
        df["outlier_line"] = (df["consensus_deviation_abs"] > 2).astype(int)

        return df

    def aggregate_to_event_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate features to event-market level (one row per event-market).

        This creates the final feature set for model training.

        Args:
            df: DataFrame with all engineered features

        Returns:
            Event-level aggregated DataFrame
        """
        logger.info("Aggregating features to event level")

        # Get the latest snapshot for each event
        latest_snapshots = (
            df.groupby(["event_id", "market_key"])["snapshot_at"]
            .max()
            .reset_index(name="latest_snapshot_at")
        )

        df = df.merge(latest_snapshots, on=["event_id", "market_key"], how="left")

        # Filter to latest snapshot
        df_latest = df[df["snapshot_at"] == df["latest_snapshot_at"]].copy()

        # Aggregate to event level
        event_features = (
            df.groupby(["event_id", "market_key"])
            .agg(
                {
                    # Event info
                    "commence_time": "first",
                    "home_team": "first",
                    "away_team": "first",
                    # Line movement features (max absolute values)
                    "line_velocity_abs": "max",
                    "line_acceleration_abs": "max",
                    # Dispersion features (from latest snapshot)
                    "cross_book_std": "last",
                    "cross_book_median": "last",
                    "cross_book_range": "last",
                    "cross_book_count": "last",
                    # Time features (from latest snapshot)
                    "hours_to_kickoff": "last",
                    "time_decay_weight": "last",
                    # Sharp move detection
                    "sharp_move_count": "first",
                    "multiple_sharp_moves": "first",
                    # High dispersion flag (from latest)
                    "high_dispersion": "last",
                }
            )
            .reset_index()
        )

        # Rename columns for clarity
        event_features.rename(
            columns={
                "line_velocity_abs": "max_line_velocity",
                "line_acceleration_abs": "max_line_acceleration",
            },
            inplace=True,
        )

        logger.info(f"Created {len(event_features):,} event-level feature rows")

        return event_features

    def engineer_all_features(
        self,
        start_date: str,
        end_date: str,
        markets: list[str] = None,
        min_bookmakers: int = 5,
    ) -> pd.DataFrame:
        """
        Engineer all market microstructure features.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            markets: List of markets (default: spreads, totals)
            min_bookmakers: Minimum bookmakers per event

        Returns:
            DataFrame with all engineered features
        """
        logger.info("=" * 80)
        logger.info("MARKET MICROSTRUCTURE FEATURE ENGINEERING")
        logger.info("=" * 80)

        # Step 1: Fetch historical data
        df = self.fetch_historical_odds(start_date, end_date, markets, min_bookmakers)

        if df.empty:
            logger.error("No data available - cannot engineer features")
            return pd.DataFrame()

        # Step 2: Calculate line movement velocity
        df = self.calculate_line_movement_velocity(df)

        # Step 3: Calculate line movement acceleration
        df = self.calculate_line_movement_acceleration(df)

        # Step 4: Calculate cross-book dispersion
        df = self.calculate_cross_book_dispersion(df)

        # Step 5: Calculate time-to-kickoff
        df = self.calculate_time_to_kickoff(df)

        # Step 6: Detect sharp moves
        df = self.detect_sharp_moves(df)

        # Step 7: Calculate consensus deviation
        df = self.calculate_consensus_deviation(df)

        # Step 8: Aggregate to event level
        event_features = self.aggregate_to_event_level(df)

        logger.info("=" * 80)
        logger.info("FEATURE ENGINEERING COMPLETE")
        logger.info("=" * 80)

        return event_features


# ============================================================================
# Data Quality Validation
# ============================================================================


def validate_features(df: pd.DataFrame) -> dict:
    """
    Validate engineered features for data quality issues.

    Args:
        df: DataFrame with engineered features

    Returns:
        Dictionary with validation results
    """
    logger.info("Validating feature quality...")

    validation = {
        "total_events": len(df),
        "date_range": {
            "start": df["commence_time"].min().isoformat(),
            "end": df["commence_time"].max().isoformat(),
        },
        "missing_data": {},
        "feature_stats": {},
        "quality_flags": {},
    }

    # Check for missing data
    for col in df.columns:
        missing_pct = (df[col].isna().sum() / len(df)) * 100
        if missing_pct > 0:
            validation["missing_data"][col] = f"{missing_pct:.2f}%"

    # Feature statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        validation["feature_stats"][col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
        }

    # Quality flags
    validation["quality_flags"]["high_dispersion_events"] = int(df["high_dispersion"].sum())
    validation["quality_flags"]["sharp_move_events"] = int((df["sharp_move_count"] > 0).sum())
    validation["quality_flags"]["multiple_sharp_moves"] = int(df["multiple_sharp_moves"].sum())

    # Overall quality score (% complete data)
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()
    validation["data_quality_pct"] = ((total_cells - missing_cells) / total_cells) * 100

    return validation


# ============================================================================
# CLI
# ============================================================================


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Engineer market microstructure features from odds data"
    )

    parser.add_argument(
        "--season",
        type=int,
        help="NFL season year (e.g., 2024)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: data/market_features_{season}.csv)",
    )
    parser.add_argument(
        "--min-bookmakers",
        type=int,
        default=5,
        help="Minimum bookmakers required per event (default: 5)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation checks and generate report",
    )

    args = parser.parse_args()

    # Determine date range
    if args.season:
        start_date = f"{args.season}-09-01"
        end_date = f"{args.season + 1}-02-28"
    elif args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
    else:
        logger.error("Must specify either --season or both --start-date and --end-date")
        return 1

    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        season = args.season if args.season else start_date[:4]
        output_file = Path(__file__).parent.parent / "data" / f"market_features_{season}.csv"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Initialize database config
    db_config = DatabaseConfig()

    # Engineer features
    with MarketFeatureEngineer(db_config) as engineer:
        features = engineer.engineer_all_features(
            start_date=start_date,
            end_date=end_date,
            markets=["spreads", "totals"],
            min_bookmakers=args.min_bookmakers,
        )

    if features.empty:
        logger.error("No features engineered - check date range and data availability")
        return 1

    # Save to CSV
    logger.info(f"Saving features to {output_file}")
    features.to_csv(output_file, index=False)
    logger.info(f"Saved {len(features):,} feature rows")

    # Print sample
    logger.info("\nSample of engineered features (first 5 rows):")
    print(features.head(10).to_string())

    # Validation
    if args.validate:
        logger.info("\n" + "=" * 80)
        logger.info("DATA QUALITY VALIDATION")
        logger.info("=" * 80)

        validation = validate_features(features)

        # Save validation report
        validation_file = output_file.with_suffix(".validation.json")
        with open(validation_file, "w") as f:
            json.dump(validation, f, indent=2)

        logger.info(f"\nValidation report saved to {validation_file}")
        logger.info(f"Data quality: {validation['data_quality_pct']:.2f}%")
        logger.info(
            f"Events with high dispersion: {validation['quality_flags']['high_dispersion_events']}"
        )
        logger.info(f"Events with sharp moves: {validation['quality_flags']['sharp_move_events']}")
        logger.info(
            f"Events with multiple sharp moves: {validation['quality_flags']['multiple_sharp_moves']}"
        )

    logger.info("\n" + "=" * 80)
    logger.info("SUCCESS: Market feature engineering complete")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
