#!/usr/bin/env python3
"""
Load coaching staff data from Wikipedia CSV into PostgreSQL.

This script loads the scraped coaching data and populates:
- coaching_staff table (historical coaching assignments)
- coach_metadata table (coach biographical info)
- Refreshes coaching_continuity materialized view

Usage:
    python scripts/load_coaching_data.py --input data/coaching_staff_2018_2025_wikipedia_full.csv

Author: Claude Code (Sonnet 4.5)
Date: November 26, 2025
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Database connection
DB_CONFIG = {
    "host": "localhost",
    "port": 5544,
    "database": "devdb01",
    "user": "dro",
    "password": "sicillionbillions",  # pragma: allowlist secret
}


def clean_coach_name(name: str) -> str:
    """Clean and standardize coach name."""
    if pd.isna(name) or name is None:
        return None

    name = str(name).strip()

    # Filter out obvious parsing errors (years, team names, etc.)
    if name.isdigit() or len(name) < 3:
        return None

    # Known bad parses from Wikipedia
    bad_names = {
        "2017",
        "2018",
        "2019",
        "2020",
        "2021",
        "2022",
        "2023",
        "2024",
        "2025",
        "Oakland Raiders",
        "Cincinnati Bengals",
        "Los Angeles Chargers",
        "Super Bowl LII",
        "None",
    }

    if name in bad_names:
        return None

    return name


def load_coaching_data(csv_path: Path) -> pd.DataFrame:
    """Load and clean coaching data from CSV."""
    logger.info(f"Loading coaching data from {csv_path}")

    df = pd.read_csv(csv_path)

    # Clean coach names
    for col in [
        "head_coach",
        "offensive_coordinator",
        "defensive_coordinator",
        "special_teams_coordinator",
    ]:
        df[col] = df[col].apply(clean_coach_name)

    # Count data quality
    total = len(df)
    hc_count = df["head_coach"].notna().sum()
    oc_count = df["offensive_coordinator"].notna().sum()
    dc_count = df["defensive_coordinator"].notna().sum()

    logger.info(f"Loaded {total} team-seasons")
    logger.info(f"  Head coaches: {hc_count}/{total} ({100*hc_count/total:.1f}%)")
    logger.info(f"  Offensive coordinators: {oc_count}/{total} ({100*oc_count/total:.1f}%)")
    logger.info(f"  Defensive coordinators: {dc_count}/{total} ({100*dc_count/total:.1f}%)")

    return df


def get_unique_coaches(df: pd.DataFrame) -> set[str]:
    """Extract unique coach names from dataframe."""
    coaches = set()

    for col in [
        "head_coach",
        "offensive_coordinator",
        "defensive_coordinator",
        "special_teams_coordinator",
    ]:
        coaches.update(df[col].dropna().unique())

    return coaches


def insert_coach_metadata(conn, coaches: set[str]):
    """Insert coach metadata records."""
    logger.info(f"Inserting {len(coaches)} coaches into coach_metadata")

    cur = conn.cursor()

    # Get existing coaches
    cur.execute("SELECT coach_name FROM coach_metadata")
    existing = {row[0] for row in cur.fetchall()}

    new_coaches = coaches - existing

    if not new_coaches:
        logger.info("All coaches already exist in coach_metadata")
        return

    # Insert new coaches with minimal metadata (will need manual enrichment)
    # Schema: coach_name, background, prior_positions, career_pass_tendency, scheme_family, quality_tier, notes
    insert_query = """
    INSERT INTO coach_metadata (coach_name, background, quality_tier)
    VALUES %s
    ON CONFLICT (coach_name) DO NOTHING
    """

    values = [(name, "unknown", "Unknown") for name in new_coaches]

    execute_values(cur, insert_query, values)
    conn.commit()

    logger.info(f"Inserted {len(new_coaches)} new coaches")


def insert_coaching_staff(conn, df: pd.DataFrame):
    """Insert coaching staff records."""
    logger.info(f"Inserting {len(df)} coaching staff records")

    cur = conn.cursor()

    # Clear existing data for clean reload
    cur.execute("TRUNCATE TABLE coaching_staff CASCADE")

    # Prepare records - schema is: team, season, week, head_coach, offensive_coordinator, defensive_coordinator, special_teams_coordinator
    records = []
    for _, row in df.iterrows():
        records.append(
            (
                row["team"],
                int(row["season"]),
                None,  # week = NULL for season-level records
                row["head_coach"],
                row["offensive_coordinator"],
                row["defensive_coordinator"],
                row.get("special_teams_coordinator"),  # May not exist in all rows
            )
        )

    # Insert all records
    insert_query = """
    INSERT INTO coaching_staff (team, season, week, head_coach, offensive_coordinator, defensive_coordinator, special_teams_coordinator)
    VALUES %s
    """

    execute_values(cur, insert_query, records)
    conn.commit()

    logger.info(f"Inserted {len(records)} coaching staff records")


def refresh_materialized_view(conn):
    """Refresh coaching_continuity materialized view."""
    logger.info("Refreshing coaching_continuity materialized view...")

    cur = conn.cursor()
    cur.execute("REFRESH MATERIALIZED VIEW coaching_continuity")
    conn.commit()

    # Verify row count
    cur.execute("SELECT COUNT(*) FROM coaching_continuity")
    count = cur.fetchone()[0]

    logger.info(f"coaching_continuity refreshed with {count} rows")


def validate_load(conn):
    """Validate the data load."""
    logger.info("Validating data load...")

    cur = conn.cursor()

    # Count total records
    cur.execute("SELECT COUNT(*) FROM coaching_staff")
    total = cur.fetchone()[0]
    print(f"\n=== COACHING STAFF LOADED: {total} team-seasons ===")

    # Count non-null coaches
    cur.execute(
        """
        SELECT
            COUNT(*) as total,
            COUNT(head_coach) as hc_count,
            COUNT(offensive_coordinator) as oc_count,
            COUNT(defensive_coordinator) as dc_count,
            COUNT(special_teams_coordinator) as st_count
        FROM coaching_staff
    """
    )
    row = cur.fetchone()
    print(f"  Head coaches: {row[1]}/{row[0]}")
    print(f"  Offensive coordinators: {row[2]}/{row[0]}")
    print(f"  Defensive coordinators: {row[3]}/{row[0]}")
    print(f"  Special teams coordinators: {row[4]}/{row[0]}")

    # Sample recent data
    cur.execute(
        """
        SELECT team, season, head_coach, offensive_coordinator, defensive_coordinator
        FROM coaching_staff
        WHERE season = 2024
        ORDER BY team
        LIMIT 10
    """
    )

    print("\n=== SAMPLE 2024 COACHING DATA ===")
    for row in cur.fetchall():
        print(f"  {row[0]} {row[1]}: HC={row[2]}, OC={row[3]}, DC={row[4]}")

    # Check if materialized view exists and has data
    cur.execute(
        """
        SELECT COUNT(*) FROM information_schema.tables
        WHERE table_name = 'coaching_continuity'
    """
    )
    if cur.fetchone()[0] > 0:
        cur.execute("SELECT COUNT(*) FROM coaching_continuity")
        continuity_count = cur.fetchone()[0]
        print(f"\n=== COACHING_CONTINUITY VIEW: {continuity_count} rows ===")

    logger.info("Validation complete")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Load coaching staff data from Wikipedia CSV into PostgreSQL"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/coaching_staff_2018_2025_wikipedia_full.csv",
        help="Input CSV file path",
    )
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        logger.error(f"Input file not found: {csv_path}")
        return 1

    # Load data
    df = load_coaching_data(csv_path)

    # Extract unique coaches
    coaches = get_unique_coaches(df)
    logger.info(f"Found {len(coaches)} unique coaches")

    # Connect to database
    logger.info("Connecting to PostgreSQL...")
    conn = psycopg2.connect(**DB_CONFIG)

    try:
        # Insert coach metadata
        insert_coach_metadata(conn, coaches)

        # Insert coaching staff
        insert_coaching_staff(conn, df)

        # Refresh materialized view
        refresh_materialized_view(conn)

        # Validate load
        validate_load(conn)

        logger.info("Data load complete!")
        return 0

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        conn.rollback()
        return 1

    finally:
        conn.close()


if __name__ == "__main__":
    exit(main())
