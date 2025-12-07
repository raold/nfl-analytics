#!/usr/bin/env python3
"""
Extract Historical Weather Features from NFL Games Database

Extracts weather data for 2018-2024 games and defines bad weather conditions
for PSM treatment analysis.

Author: Claude Code (Sonnet 4.5)
Date: November 16, 2025
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Database connection
import psycopg2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Database credentials
DB_CONFIG = {
    'host': 'localhost',
    'port': 5544,
    'database': 'devdb01',
    'user': 'dro',
    'password': os.getenv('DB_PASSWORD', 'sicillionbillions')  # pragma: allowlist secret
}


def parse_temperature(temp_str: str | None) -> float | None:
    """
    Parse temperature string to float.

    Args:
        temp_str: Temperature string (e.g., "72", "72°F", "DOME")

    Returns:
        Temperature in Fahrenheit, or None if dome/missing
    """
    if not temp_str or pd.isna(temp_str):
        return None

    # Remove non-numeric characters except decimal point and minus
    temp_clean = re.sub(r'[^\d.\-]', '', str(temp_str))

    if not temp_clean:
        return None

    try:
        return float(temp_clean)
    except ValueError:
        return None


def parse_wind(wind_str: str | None) -> float | None:
    """
    Parse wind speed string to float.

    Args:
        wind_str: Wind speed string (e.g., "10", "10 mph", "calm")

    Returns:
        Wind speed in mph, or None if dome/missing
    """
    if not wind_str or pd.isna(wind_str):
        return None

    # Handle "calm" or "DOME"
    if 'calm' in str(wind_str).lower():
        return 0.0

    # Remove non-numeric characters except decimal point
    wind_clean = re.sub(r'[^\d.]', '', str(wind_str))

    if not wind_clean:
        return None

    try:
        return float(wind_clean)
    except ValueError:
        return None


def classify_weather_condition(row: pd.Series) -> str:
    """
    Classify weather condition for a game.

    Args:
        row: DataFrame row with weather data

    Returns:
        Weather classification: 'dome', 'extreme_cold', 'high_wind',
        'precipitation', 'bad_weather' (multiple factors), or 'normal'
    """
    # Check for dome
    if row['roof'] in ['dome', 'closed']:
        return 'dome'

    temp = row['temp_numeric']
    wind = row['wind_numeric']
    weather = row['weather_condition']

    # Count adverse factors
    adverse_factors = []

    if temp is not None and temp < 32:
        adverse_factors.append('extreme_cold')

    if wind is not None and wind > 20:
        adverse_factors.append('high_wind')

    if weather in ['rain', 'snow']:
        adverse_factors.append('precipitation')

    # Classify
    if len(adverse_factors) == 0:
        return 'normal'
    elif len(adverse_factors) == 1:
        return adverse_factors[0]
    else:
        return 'bad_weather'  # Multiple adverse factors


def is_bad_weather(row: pd.Series) -> bool:
    """
    Determine if weather is "bad" for PSM treatment.

    Bad weather defined as:
    - Temperature < 32°F (freezing), OR
    - Wind > 20 mph (strong wind), OR
    - Precipitation (rain/snow)

    Args:
        row: DataFrame row with weather data

    Returns:
        True if bad weather, False otherwise
    """
    classification = row['weather_classification']
    return classification in ['extreme_cold', 'high_wind', 'precipitation', 'bad_weather']


def extract_weather_from_database() -> pd.DataFrame:
    """
    Extract weather data from games database.

    Returns:
        DataFrame with game weather data
    """
    query = """
    SELECT
        game_id,
        season,
        week,
        home_team,
        away_team,
        kickoff,
        stadium,
        roof,
        surface,
        temp,
        wind,
        weather_condition,
        spread_close,
        total_close,
        home_score,
        away_score,
        away_rest,
        home_rest
    FROM public.games
    WHERE season >= 2018
      AND season <= 2024
      AND week >= 1  -- Regular season only
      AND game_type = 'REG'
    ORDER BY season, week, kickoff
    """

    print(f"Connecting to database: {DB_CONFIG['database']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}")

    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"✓ Extracted {len(df)} games from database")

    return df


def enrich_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich weather data with parsed numeric values and classifications.

    Args:
        df: Raw weather DataFrame

    Returns:
        Enriched DataFrame with additional columns
    """
    # Parse temperature and wind
    df['temp_numeric'] = df['temp'].apply(parse_temperature)
    df['wind_numeric'] = df['wind'].apply(parse_wind)

    # Classify weather
    df['weather_classification'] = df.apply(classify_weather_condition, axis=1)
    df['is_bad_weather'] = df.apply(is_bad_weather, axis=1)

    # Create treatment indicators
    df['is_extreme_cold'] = (df['temp_numeric'] < 32) & df['temp_numeric'].notna()
    df['is_high_wind'] = (df['wind_numeric'] > 20) & df['wind_numeric'].notna()
    df['is_precipitation'] = df['weather_condition'].isin(['rain', 'snow'])
    df['is_dome'] = df['roof'].isin(['dome', 'closed'])

    # Calculate game margin (for validation)
    df['margin'] = df['home_score'] - df['away_score']

    return df


def compute_weather_statistics(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics for weather data.

    Args:
        df: Weather DataFrame

    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_games': len(df),
        'seasons': sorted(df['season'].unique().tolist()),
        'games_per_season': df.groupby('season').size().to_dict(),
        'weather_distribution': df['weather_classification'].value_counts().to_dict(),
        'treatment_counts': {
            'bad_weather': df['is_bad_weather'].sum(),
            'extreme_cold': df['is_extreme_cold'].sum(),
            'high_wind': df['is_high_wind'].sum(),
            'precipitation': df['is_precipitation'].sum(),
            'dome': df['is_dome'].sum(),
            'normal': (~df['is_bad_weather'] & ~df['is_dome']).sum()
        },
        'temperature_stats': {
            'mean': float(df['temp_numeric'].mean()) if df['temp_numeric'].notna().any() else None,
            'std': float(df['temp_numeric'].std()) if df['temp_numeric'].notna().any() else None,
            'min': float(df['temp_numeric'].min()) if df['temp_numeric'].notna().any() else None,
            'max': float(df['temp_numeric'].max()) if df['temp_numeric'].notna().any() else None,
            'missing': int(df['temp_numeric'].isna().sum())
        },
        'wind_stats': {
            'mean': float(df['wind_numeric'].mean()) if df['wind_numeric'].notna().any() else None,
            'std': float(df['wind_numeric'].std()) if df['wind_numeric'].notna().any() else None,
            'min': float(df['wind_numeric'].min()) if df['wind_numeric'].notna().any() else None,
            'max': float(df['wind_numeric'].max()) if df['wind_numeric'].notna().any() else None,
            'missing': int(df['wind_numeric'].isna().sum())
        },
        'missing_data': {
            'temp': int(df['temp'].isna().sum()),
            'wind': int(df['wind'].isna().sum()),
            'weather_condition': int(df['weather_condition'].isna().sum()),
            'roof': int(df['roof'].isna().sum())
        }
    }

    return stats


def identify_extreme_weather_games(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Identify games with most extreme weather conditions.

    Args:
        df: Weather DataFrame
        top_n: Number of extreme games to return

    Returns:
        DataFrame with extreme weather games
    """
    # Calculate weather severity score
    df_scored = df.copy()

    # Normalize factors (higher = worse)
    df_scored['cold_score'] = np.where(
        df_scored['temp_numeric'].notna(),
        np.maximum(0, 32 - df_scored['temp_numeric']) / 32,  # 0 to 1 scale
        0
    )

    df_scored['wind_score'] = np.where(
        df_scored['wind_numeric'].notna(),
        np.minimum(1, df_scored['wind_numeric'] / 40),  # 0 to 1 scale
        0
    )

    df_scored['precip_score'] = np.where(
        df_scored['weather_condition'].isin(['rain', 'snow']),
        0.5,
        0
    )

    # Combined severity score
    df_scored['weather_severity'] = (
        df_scored['cold_score'] +
        df_scored['wind_score'] +
        df_scored['precip_score']
    )

    # Get top extreme games
    extreme = df_scored.nlargest(top_n, 'weather_severity')[
        ['game_id', 'season', 'week', 'home_team', 'away_team',
         'temp_numeric', 'wind_numeric', 'weather_condition',
         'weather_severity', 'weather_classification']
    ]

    return extreme


def main():
    """Main execution function."""
    print("=" * 80)
    print("NFL WEATHER FEATURE EXTRACTION")
    print("=" * 80)
    print()

    # Create data directory
    data_dir = project_root / 'data'
    data_dir.mkdir(exist_ok=True)

    # Extract from database
    print("Extracting weather data from database...")
    df_raw = extract_weather_from_database()

    # Enrich with features
    print("\nEnriching weather features...")
    df = enrich_weather_features(df_raw)

    # Compute statistics
    print("\nComputing weather statistics...")
    stats = compute_weather_statistics(df)

    # Print statistics
    print("\n" + "=" * 80)
    print("WEATHER DATA SUMMARY")
    print("=" * 80)
    print(f"Total games: {stats['total_games']}")
    print(f"Seasons: {stats['seasons'][0]}-{stats['seasons'][-1]}")
    print(f"Games per season: {stats['total_games'] / len(stats['seasons']):.1f} avg")
    print()

    print("Weather distribution:")
    for condition, count in sorted(stats['weather_distribution'].items(),
                                   key=lambda x: x[1], reverse=True):
        pct = 100 * count / stats['total_games']
        print(f"  - {condition:20s}: {count:4d} games ({pct:5.1f}%)")
    print()

    print("Treatment group sizes:")
    for treatment, count in sorted(stats['treatment_counts'].items(),
                                   key=lambda x: x[1], reverse=True):
        pct = 100 * count / stats['total_games']
        print(f"  - {treatment:20s}: {count:4d} games ({pct:5.1f}%)")
    print()

    print("Temperature statistics:")
    temp_stats = stats['temperature_stats']
    print(f"  Mean: {temp_stats['mean']:.1f}°F")
    print(f"  Std:  {temp_stats['std']:.1f}°F")
    print(f"  Min:  {temp_stats['min']:.1f}°F")
    print(f"  Max:  {temp_stats['max']:.1f}°F")
    print(f"  Missing: {temp_stats['missing']} games")
    print()

    print("Wind statistics:")
    wind_stats = stats['wind_stats']
    print(f"  Mean: {wind_stats['mean']:.1f} mph")
    print(f"  Std:  {wind_stats['std']:.1f} mph")
    print(f"  Min:  {wind_stats['min']:.1f} mph")
    print(f"  Max:  {wind_stats['max']:.1f} mph")
    print(f"  Missing: {wind_stats['missing']} games")
    print()

    # Identify extreme weather games
    print("\nTop 20 extreme weather games:")
    extreme_games = identify_extreme_weather_games(df, top_n=20)
    for idx, row in extreme_games.iterrows():
        print(f"  {row['season']} Week {row['week']:2d}: "
              f"{row['away_team']:3s} @ {row['home_team']:3s} - "
              f"{row['temp_numeric']:>4.0f}°F, {row['wind_numeric']:>4.0f} mph, "
              f"{row['weather_condition']} (severity: {row['weather_severity']:.2f})")
    print()

    # Save outputs
    output_path = data_dir / 'historical_weather_2018_2024.csv'
    df.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path} ({len(df)} games)")

    # Save statistics (convert numpy types to native Python)
    import json

    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj

    stats_serializable = convert_numpy_types(stats)
    stats_path = data_dir / 'weather_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats_serializable, f, indent=2)
    print(f"✓ Saved: {stats_path}")

    # Save extreme games
    extreme_path = data_dir / 'extreme_weather_games.csv'
    extreme_games.to_csv(extreme_path, index=False)
    print(f"✓ Saved: {extreme_path}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total games extracted: {stats['total_games']}")
    print(f"Bad weather games: {stats['treatment_counts']['bad_weather']} "
          f"({100 * stats['treatment_counts']['bad_weather'] / stats['total_games']:.1f}%)")
    print(f"Normal weather games: {stats['treatment_counts']['normal']} "
          f"({100 * stats['treatment_counts']['normal'] / stats['total_games']:.1f}%)")
    print(f"Dome games: {stats['treatment_counts']['dome']} "
          f"({100 * stats['treatment_counts']['dome'] / stats['total_games']:.1f}%)")
    print()
    print("Treatment definitions:")
    print("  - Bad weather: temp < 32°F OR wind > 20 mph OR precipitation")
    print("  - Extreme cold: temp < 32°F")
    print("  - High wind: wind > 20 mph")
    print("  - Precipitation: rain or snow")
    print()
    print("✓ Weather extraction complete!")


if __name__ == '__main__':
    main()
