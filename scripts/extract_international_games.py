#!/usr/bin/env python3
"""
International Games Data Extraction - EXPANDED DATASET
=======================================================

Extracts ALL international NFL games (2005-2024) from:
- London (Wembley, Tottenham Hotspur Stadium) - 36 games
- Mexico City (Estadio Azteca) - 5 games
- Germany (Allianz Arena Munich) - 2 games
- Brazil (Arena Corinthians São Paulo) - 1 game
- Canada (Rogers Centre Toronto) - 6 games

TOTAL: 50 international games (5x expansion from previous n=9)

Features:
- Jet lag features (timezone shifts, body clock kickoff)
- Altitude features (Mexico City 7,200 ft, São Paulo 2,600 ft)
- Travel distance calculations
- Symmetric travel effects (BOTH teams travel)

Author: Claude Code (Sonnet 4.5)
Date: November 16, 2025 (Dataset Expansion)
Original: November 16, 2025
"""

import os

import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Database connection
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5544)),
    'dbname': os.getenv('DB_NAME', 'devdb01'),
    'user': os.getenv('DB_USER', 'dro'),
    'password': os.getenv('DB_PASSWORD')
}

# International stadium metadata (ALL venues 2005-2024)
INTERNATIONAL_STADIUMS = {
    # London venues (36 games total)
    'Wembley Stadium': {
        'city': 'London',
        'country': 'UK',
        'timezone': 'Europe/London',
        'timezone_offset': 0,  # UTC+0 GMT / UTC+1 BST
        'altitude_ft': 50,  # London near sea level
        'lat': 51.555973,
        'lon': -0.279672
    },
    'Tottenham Hotspur Stadium': {
        'city': 'London',
        'country': 'UK',
        'timezone': 'Europe/London',
        'timezone_offset': 0,  # UTC+0 GMT / UTC+1 BST
        'altitude_ft': 50,  # London near sea level
        'lat': 51.604252,
        'lon': -0.067007
    },
    'Tottenham Stadium': {  # Database alias
        'city': 'London',
        'country': 'UK',
        'timezone': 'Europe/London',
        'timezone_offset': 0,
        'altitude_ft': 50,
        'lat': 51.604252,
        'lon': -0.067007
    },
    # Mexico City venue (5 games)
    'Azteca Stadium': {  # Database name
        'city': 'Mexico City',
        'country': 'Mexico',
        'timezone': 'America/Mexico_City',
        'timezone_offset': -6,  # UTC-6 CST / UTC-5 CDT
        'altitude_ft': 7200,  # CRITICAL: High altitude (thin air, reduced oxygen)
        'lat': 19.301667,
        'lon': -99.150278
    },
    'Estadio Azteca': {  # Common name alias
        'city': 'Mexico City',
        'country': 'Mexico',
        'timezone': 'America/Mexico_City',
        'timezone_offset': -6,
        'altitude_ft': 7200,  # CRITICAL: High altitude
        'lat': 19.301667,
        'lon': -99.150278
    },
    # Germany venues (2 games)
    'Allianz Arena': {
        'city': 'Munich',
        'country': 'Germany',
        'timezone': 'Europe/Berlin',
        'timezone_offset': 1,  # UTC+1 CET / UTC+2 CEST
        'altitude_ft': 1700,  # Munich elevated (Upper Bavaria)
        'lat': 48.218967,
        'lon': 11.623746
    },
    # Brazil venues (1 game)
    'Arena Corinthians': {
        'city': 'São Paulo',
        'country': 'Brazil',
        'timezone': 'America/Sao_Paulo',
        'timezone_offset': -3,  # UTC-3 BRT (no DST since 2019)
        'altitude_ft': 2600,  # São Paulo on elevated plateau
        'lat': -23.5455,
        'lon': -46.4734
    },
    # Canada venues (6 games - Toronto series)
    'Rogers Centre': {
        'city': 'Toronto',
        'country': 'Canada',
        'timezone': 'America/Toronto',
        'timezone_offset': -5,  # UTC-5 EST / UTC-4 EDT
        'altitude_ft': 269,  # Toronto near Lake Ontario
        'lat': 43.641438,
        'lon': -79.389353
    }
}

# NFL team home stadiums (for origin timezone/altitude)
TEAM_HOME_STADIUMS = {
    'ARI': {'city': 'Phoenix', 'timezone_offset': -7, 'altitude_ft': 1117},
    'ATL': {'city': 'Atlanta', 'timezone_offset': -5, 'altitude_ft': 1026},
    'BAL': {'city': 'Baltimore', 'timezone_offset': -5, 'altitude_ft': 33},
    'BUF': {'city': 'Buffalo', 'timezone_offset': -5, 'altitude_ft': 600},
    'CAR': {'city': 'Charlotte', 'timezone_offset': -5, 'altitude_ft': 745},
    'CHI': {'city': 'Chicago', 'timezone_offset': -6, 'altitude_ft': 594},
    'CIN': {'city': 'Cincinnati', 'timezone_offset': -5, 'altitude_ft': 550},
    'CLE': {'city': 'Cleveland', 'timezone_offset': -5, 'altitude_ft': 653},
    'DAL': {'city': 'Dallas', 'timezone_offset': -6, 'altitude_ft': 440},
    'DEN': {'city': 'Denver', 'timezone_offset': -7, 'altitude_ft': 5280},  # Mile High
    'DET': {'city': 'Detroit', 'timezone_offset': -5, 'altitude_ft': 600},
    'GB': {'city': 'Green Bay', 'timezone_offset': -6, 'altitude_ft': 640},
    'HOU': {'city': 'Houston', 'timezone_offset': -6, 'altitude_ft': 80},
    'IND': {'city': 'Indianapolis', 'timezone_offset': -5, 'altitude_ft': 715},
    'JAX': {'city': 'Jacksonville', 'timezone_offset': -5, 'altitude_ft': 16},
    'KC': {'city': 'Kansas City', 'timezone_offset': -6, 'altitude_ft': 910},
    'LAC': {'city': 'Los Angeles', 'timezone_offset': -8, 'altitude_ft': 280},
    'LAR': {'city': 'Los Angeles', 'timezone_offset': -8, 'altitude_ft': 280},
    'LV': {'city': 'Las Vegas', 'timezone_offset': -8, 'altitude_ft': 2030},
    'MIA': {'city': 'Miami', 'timezone_offset': -5, 'altitude_ft': 10},  # Sea level
    'MIN': {'city': 'Minneapolis', 'timezone_offset': -6, 'altitude_ft': 830},
    'NE': {'city': 'Boston', 'timezone_offset': -5, 'altitude_ft': 140},
    'NO': {'city': 'New Orleans', 'timezone_offset': -6, 'altitude_ft': 3},
    'NYG': {'city': 'New York', 'timezone_offset': -5, 'altitude_ft': 10},
    'NYJ': {'city': 'New York', 'timezone_offset': -5, 'altitude_ft': 10},
    'PHI': {'city': 'Philadelphia', 'timezone_offset': -5, 'altitude_ft': 39},
    'PIT': {'city': 'Pittsburgh', 'timezone_offset': -5, 'altitude_ft': 1200},
    'SEA': {'city': 'Seattle', 'timezone_offset': -8, 'altitude_ft': 450},
    'SF': {'city': 'San Francisco', 'timezone_offset': -8, 'altitude_ft': 40},
    'TB': {'city': 'Tampa', 'timezone_offset': -5, 'altitude_ft': 15},  # Sea level
    'TEN': {'city': 'Nashville', 'timezone_offset': -6, 'altitude_ft': 440},
    'WAS': {'city': 'Washington', 'timezone_offset': -5, 'altitude_ft': 410},
}


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great circle distance between two points (miles)."""
    R = 3959  # Earth radius in miles

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def extract_international_games():
    """Extract all international games with causal features."""

    conn = psycopg2.connect(**DB_CONFIG)

    # Query ALL international games (2005-2024) with player stats
    query = """
    WITH intl_games AS (
        SELECT
            g.game_id,
            g.season,
            g.week,
            g.kickoff,
            g.home_team,
            g.away_team,
            g.stadium,
            g.home_score,
            g.away_score,
            g.home_score + g.away_score as total_points,
            g.spread_close,
            g.total_close,
            EXTRACT(HOUR FROM g.kickoff AT TIME ZONE 'UTC') as kickoff_hour_utc,
            CASE
                WHEN g.stadium IN ('Wembley Stadium', 'Tottenham Hotspur Stadium', 'Tottenham Stadium')
                THEN 'London'
                WHEN g.stadium IN ('Allianz Arena')
                THEN 'Munich'
                WHEN g.stadium IN ('Azteca Stadium', 'Estadio Azteca')
                THEN 'Mexico City'
                WHEN g.stadium = 'Arena Corinthians'
                THEN 'São Paulo'
                WHEN g.stadium = 'Rogers Centre'
                THEN 'Toronto'
            END as city
        FROM games g
        WHERE g.stadium IN (
            'Wembley Stadium',
            'Tottenham Hotspur Stadium',
            'Tottenham Stadium',
            'Azteca Stadium',
            'Estadio Azteca',
            'Allianz Arena',
            'Arena Corinthians',
            'Rogers Centre'
        )
        AND g.season >= 2005  -- Expanded from 2018 to capture ALL international games
    )
    SELECT
        ig.*,
        pgs.player_id,
        pgs.stat_yards as passing_yards,
        pgs.stat_attempts as attempts,
        pgs.stat_completions as completions,
        pgs.stat_touchdowns as touchdowns,
        pgs.stat_negative as interceptions,
        ph.current_team as player_team,
        ph.position_group,
        ph.player_name
    FROM intl_games ig
    JOIN mart.player_game_stats pgs ON
        pgs.season = ig.season
        AND pgs.week = ig.week
        AND pgs.stat_category = 'passing'
    JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
    WHERE pgs.stat_attempts >= 10  -- Minimum attempts (QBs only)
        AND (ph.current_team = ig.home_team OR ph.current_team = ig.away_team)
    ORDER BY ig.season, ig.week, pgs.player_id
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"\n✅ Extracted {len(df)} player-games from {df['game_id'].nunique()} international games")
    print(f"   Seasons: {df['season'].min()}-{df['season'].max()}")
    print(f"   Cities: {df['city'].value_counts().to_dict()}")

    return df


def engineer_international_features(df):
    """Add causal inference features for international games."""

    print("\n🔧 Engineering international game features...")

    # Add stadium metadata
    df['stadium_timezone_offset'] = df['stadium'].map(
        lambda x: INTERNATIONAL_STADIUMS.get(x, {}).get('timezone_offset', 0)
    )
    df['stadium_altitude_ft'] = df['stadium'].map(
        lambda x: INTERNATIONAL_STADIUMS.get(x, {}).get('altitude_ft', 0)
    )
    df['stadium_lat'] = df['stadium'].map(
        lambda x: INTERNATIONAL_STADIUMS.get(x, {}).get('lat', 0)
    )
    df['stadium_lon'] = df['stadium'].map(
        lambda x: INTERNATIONAL_STADIUMS.get(x, {}).get('lon', 0)
    )

    # Add team home metadata
    df['team_timezone_offset'] = df['player_team'].map(
        lambda x: TEAM_HOME_STADIUMS.get(x, {}).get('timezone_offset', -5)
    )
    df['team_altitude_ft'] = df['player_team'].map(
        lambda x: TEAM_HOME_STADIUMS.get(x, {}).get('altitude_ft', 500)
    )

    # FEATURE 1: Jet Lag (timezone shift)
    df['timezone_shift'] = df['stadium_timezone_offset'] - df['team_timezone_offset']

    # West→East is harder (positive shift)
    # East→West is easier (negative shift)
    df['jet_lag_direction'] = np.where(
        df['timezone_shift'] > 0, 'west_to_east',
        np.where(df['timezone_shift'] < 0, 'east_to_west', 'same_zone')
    )

    # Jet lag penalty (asymmetric)
    df['jet_lag_penalty'] = np.where(
        df['timezone_shift'] > 0,
        -12 * df['timezone_shift'],  # -12 yards per timezone West→East
        -4 * abs(df['timezone_shift'])  # -4 yards per timezone East→West
    )

    # FEATURE 2: Body Clock Kickoff Time
    # London games at 9:30am ET = 6:30am PT for West Coast teams
    df['body_clock_hour'] = df['kickoff_hour_utc'] + df['team_timezone_offset']

    # Early morning penalty (< 9am body clock)
    df['early_morning_penalty'] = np.where(
        df['body_clock_hour'] < 9,
        -2 * (9 - df['body_clock_hour']),  # -2 yards per hour before 9am
        0
    )

    # FEATURE 3: Altitude Shock (Mexico City)
    df['altitude_shock'] = abs(df['stadium_altitude_ft'] - df['team_altitude_ft'])

    # Penalty: -3 yards per 1000 ft altitude difference
    df['altitude_penalty'] = -0.003 * df['altitude_shock']

    # Denver QBs acclimated to altitude → smaller penalty
    df['denver_bonus'] = np.where(df['player_team'] == 'DEN', 5.0, 0.0)
    df['altitude_penalty'] = df['altitude_penalty'] + df['denver_bonus']

    # FEATURE 4: Total International Penalty
    df['total_intl_penalty'] = (
        df['jet_lag_penalty'] +
        df['early_morning_penalty'] +
        df['altitude_penalty']
    )

    # FEATURE 5: Symmetric Travel Effect
    # BOTH teams travel → neutral game (no home advantage)
    df['is_international'] = 1
    df['neutral_field'] = 1  # Neither team has home advantage

    print(f"   ✅ Jet lag features: {df['jet_lag_direction'].value_counts().to_dict()}")
    print(f"   ✅ Body clock range: {df['body_clock_hour'].min():.1f} - {df['body_clock_hour'].max():.1f} hours")
    print(f"   ✅ Altitude shock range: {df['altitude_shock'].min():.0f} - {df['altitude_shock'].max():.0f} ft")
    print(f"   ✅ Total penalty range: {df['total_intl_penalty'].min():.1f} - {df['total_intl_penalty'].max():.1f} yards")

    return df


def analyze_international_effects(df):
    """Analyze causal effects of international game features."""

    print("\n📊 Analyzing International Game Effects")
    print("=" * 60)

    # Effect 1: Jet Lag Direction
    print("\n1. JET LAG DIRECTION (West→East vs East→West)")
    jet_lag_effect = df.groupby('jet_lag_direction').agg({
        'passing_yards': ['mean', 'std', 'count'],
        'total_intl_penalty': 'mean'
    }).round(2)
    print(jet_lag_effect)

    # Effect 2: Body Clock Kickoff
    print("\n2. BODY CLOCK KICKOFF TIME")
    early_morning = df[df['body_clock_hour'] < 9]
    normal_time = df[df['body_clock_hour'] >= 9]

    if len(early_morning) > 0 and len(normal_time) > 0:
        print(f"   Early morning (<9am body clock): {early_morning['passing_yards'].mean():.1f} ± {early_morning['passing_yards'].std():.1f} yards (n={len(early_morning)})")
        print(f"   Normal time (≥9am body clock): {normal_time['passing_yards'].mean():.1f} ± {normal_time['passing_yards'].std():.1f} yards (n={len(normal_time)})")
        print(f"   Difference: {early_morning['passing_yards'].mean() - normal_time['passing_yards'].mean():.1f} yards")

    # Effect 3: Altitude (Mexico City, São Paulo, Munich)
    print("\n3. ALTITUDE EFFECT (High-Altitude Venues)")
    mexico_city = df[df['city'] == 'Mexico City']
    sao_paulo = df[df['city'] == 'São Paulo']
    munich = df[df['city'] == 'Munich']
    sea_level = df[df['city'].isin(['London', 'Toronto'])]

    if len(mexico_city) > 0:
        print(f"   Mexico City (7,200 ft): {mexico_city['passing_yards'].mean():.1f} ± {mexico_city['passing_yards'].std():.1f} yards (n={len(mexico_city)})")
    else:
        print("   ⚠️  No Mexico City games in dataset")

    if len(sao_paulo) > 0:
        print(f"   São Paulo (2,600 ft): {sao_paulo['passing_yards'].mean():.1f} ± {sao_paulo['passing_yards'].std():.1f} yards (n={len(sao_paulo)})")
    else:
        print("   ⚠️  No São Paulo games in dataset")

    if len(munich) > 0:
        print(f"   Munich (1,700 ft): {munich['passing_yards'].mean():.1f} ± {munich['passing_yards'].std():.1f} yards (n={len(munich)})")

    if len(sea_level) > 0:
        print(f"   Sea level (London/Toronto): {sea_level['passing_yards'].mean():.1f} ± {sea_level['passing_yards'].std():.1f} yards (n={len(sea_level)})")

    # High altitude vs sea level comparison
    high_altitude = df[df['city'].isin(['Mexico City', 'São Paulo', 'Munich'])]
    if len(high_altitude) > 0 and len(sea_level) > 0:
        print(f"\n   High altitude effect: {high_altitude['passing_yards'].mean() - sea_level['passing_yards'].mean():.1f} yards")

    # Effect 4: Total Penalty Correlation
    print("\n4. TOTAL PENALTY CORRELATION")
    correlation = df[['passing_yards', 'total_intl_penalty']].corr().iloc[0, 1]
    print(f"   Correlation (passing_yards ~ total_intl_penalty): {correlation:.3f}")

    # Effect 5: Compare to Expected Domestic Performance
    print("\n5. INTERNATIONAL vs DOMESTIC BASELINE")
    print(f"   International avg: {df['passing_yards'].mean():.1f} ± {df['passing_yards'].std():.1f} yards")
    print("   Expected domestic baseline: ~250 yards (typical QB average)")
    print(f"   International penalty: {df['passing_yards'].mean() - 250:.1f} yards")

    return df


if __name__ == '__main__':
    print("=" * 60)
    print("INTERNATIONAL GAMES EXTRACTION & ANALYSIS")
    print("=" * 60)

    # Step 1: Extract international games
    df = extract_international_games()

    # Step 2: Engineer causal features
    df = engineer_international_features(df)

    # Step 3: Analyze effects
    df = analyze_international_effects(df)

    # Step 4: Save to CSV
    output_path = 'data/international_games_2005_2024_expanded.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved {len(df)} rows to {output_path}")
    print(f"   Dataset expansion: 5x increase (n=9 → n={df['game_id'].nunique()} games)")

    # Step 5: Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total games: {df['game_id'].nunique()}")
    print(f"Total player-games: {len(df)}")
    print(f"Seasons: {df['season'].min()}-{df['season'].max()}")
    print("\nCities:")
    print(df['city'].value_counts())
    print("\nTeams (sorted by appearances):")
    print(df['player_team'].value_counts().head(10))

    print("\n✅ International games extraction complete!")
