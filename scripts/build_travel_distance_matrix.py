#!/usr/bin/env python3
"""
Build Travel Distance Matrix for NFL Stadiums

Creates a 32x32 matrix of great-circle distances between all NFL team home stadiums.
Uses haversine formula for accurate distance calculation.

Author: Claude Code (Sonnet 4.5)
Date: November 16, 2025
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# NFL Stadium Coordinates (2018-2024 era, accounting for relocations)
# Source: Google Maps / Official NFL stadium data
# Format: {team_abbr: (latitude, longitude, stadium_name, timezone)}
NFL_STADIUMS = {
    'ARI': (33.5276, -112.2626, 'State Farm Stadium', 'America/Phoenix'),
    'ATL': (33.7553, -84.4008, 'Mercedes-Benz Stadium', 'America/New_York'),
    'BAL': (39.2780, -76.6227, 'M&T Bank Stadium', 'America/New_York'),
    'BUF': (42.7738, -78.7870, 'Highmark Stadium', 'America/New_York'),
    'CAR': (35.2258, -80.8528, 'Bank of America Stadium', 'America/New_York'),
    'CHI': (41.8623, -87.6167, 'Soldier Field', 'America/Chicago'),
    'CIN': (39.0954, -84.5160, 'Paycor Stadium', 'America/New_York'),
    'CLE': (41.5061, -81.6995, 'Cleveland Browns Stadium', 'America/New_York'),
    'DAL': (32.7473, -97.0945, 'AT&T Stadium', 'America/Chicago'),
    'DEN': (39.7439, -105.0201, 'Empower Field at Mile High', 'America/Denver'),
    'DET': (42.3400, -83.0456, 'Ford Field', 'America/New_York'),
    'GB': (44.5013, -88.0622, 'Lambeau Field', 'America/Chicago'),
    'HOU': (29.6847, -95.4107, 'NRG Stadium', 'America/Chicago'),
    'IND': (39.7601, -86.1639, 'Lucas Oil Stadium', 'America/New_York'),
    'JAX': (30.3239, -81.6373, 'TIAA Bank Field', 'America/New_York'),
    'KC': (39.0489, -94.4839, 'GEHA Field at Arrowhead Stadium', 'America/Chicago'),
    'LAC': (33.9535, -118.3390, 'SoFi Stadium', 'America/Los_Angeles'),
    'LAR': (33.9535, -118.3390, 'SoFi Stadium', 'America/Los_Angeles'),
    'LV': (36.0909, -115.1833, 'Allegiant Stadium', 'America/Los_Angeles'),
    'MIA': (25.9580, -80.2389, 'Hard Rock Stadium', 'America/New_York'),
    'MIN': (44.9738, -93.2577, 'U.S. Bank Stadium', 'America/Chicago'),
    'NE': (42.0909, -71.2643, 'Gillette Stadium', 'America/New_York'),
    'NO': (29.9511, -90.0812, 'Caesars Superdome', 'America/Chicago'),
    'NYG': (40.8128, -74.0742, 'MetLife Stadium', 'America/New_York'),
    'NYJ': (40.8128, -74.0742, 'MetLife Stadium', 'America/New_York'),
    'PHI': (39.9008, -75.1675, 'Lincoln Financial Field', 'America/New_York'),
    'PIT': (40.4468, -80.0158, 'Acrisure Stadium', 'America/New_York'),
    'SEA': (47.5952, -122.3316, 'Lumen Field', 'America/Los_Angeles'),
    'SF': (37.4032, -121.9698, "Levi's Stadium", 'America/Los_Angeles'),
    'TB': (27.9759, -82.5033, 'Raymond James Stadium', 'America/New_York'),
    'TEN': (36.1665, -86.7713, 'Nissan Stadium', 'America/Chicago'),
    'WAS': (38.9076, -76.8645, 'FedExField', 'America/New_York'),
}

# Timezone offsets (standard time, in hours from UTC)
TIMEZONE_OFFSETS = {
    'America/New_York': -5,
    'America/Chicago': -6,
    'America/Denver': -7,
    'America/Los_Angeles': -8,
    'America/Phoenix': -7,  # Arizona doesn't observe DST
}


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points using haversine formula.

    Args:
        lat1: Latitude of point 1 (degrees)
        lon1: Longitude of point 1 (degrees)
        lat2: Latitude of point 2 (degrees)
        lon2: Longitude of point 2 (degrees)

    Returns:
        Distance in miles
    """
    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of Earth in miles
    radius_miles = 3958.8

    return c * radius_miles


def calculate_timezone_shift(tz1: str, tz2: str) -> int:
    """
    Calculate timezone shift in hours.

    Args:
        tz1: Home team timezone
        tz2: Away team timezone

    Returns:
        Absolute timezone difference in hours
    """
    offset1 = TIMEZONE_OFFSETS.get(tz1, 0)
    offset2 = TIMEZONE_OFFSETS.get(tz2, 0)
    return abs(offset2 - offset1)


def build_distance_matrix() -> pd.DataFrame:
    """
    Build 32x32 distance matrix for all NFL team pairs.

    Returns:
        DataFrame with teams as rows/columns, distances in miles
    """
    teams = sorted(NFL_STADIUMS.keys())
    n_teams = len(teams)

    # Initialize distance matrix
    distance_matrix = np.zeros((n_teams, n_teams))

    # Calculate pairwise distances
    for i, team1 in enumerate(teams):
        lat1, lon1, _, _ = NFL_STADIUMS[team1]
        for j, team2 in enumerate(teams):
            lat2, lon2, _, _ = NFL_STADIUMS[team2]

            if i == j:
                distance_matrix[i, j] = 0.0
            else:
                distance = haversine_distance(lat1, lon1, lat2, lon2)
                distance_matrix[i, j] = distance

    # Create DataFrame
    df = pd.DataFrame(
        distance_matrix,
        index=teams,
        columns=teams
    )

    return df


def build_timezone_matrix() -> pd.DataFrame:
    """
    Build 32x32 timezone shift matrix for all NFL team pairs.

    Returns:
        DataFrame with teams as rows/columns, timezone shifts in hours
    """
    teams = sorted(NFL_STADIUMS.keys())
    n_teams = len(teams)

    # Initialize timezone matrix
    tz_matrix = np.zeros((n_teams, n_teams))

    # Calculate pairwise timezone shifts
    for i, team1 in enumerate(teams):
        _, _, _, tz1 = NFL_STADIUMS[team1]
        for j, team2 in enumerate(teams):
            _, _, _, tz2 = NFL_STADIUMS[team2]
            tz_shift = calculate_timezone_shift(tz1, tz2)
            tz_matrix[i, j] = tz_shift

    # Create DataFrame
    df = pd.DataFrame(
        tz_matrix,
        index=teams,
        columns=teams
    )

    return df


def export_stadium_reference() -> pd.DataFrame:
    """
    Export stadium reference table with coordinates and metadata.

    Returns:
        DataFrame with stadium details
    """
    data = []
    for team, (lat, lon, stadium, timezone) in NFL_STADIUMS.items():
        data.append({
            'team': team,
            'latitude': lat,
            'longitude': lon,
            'stadium_name': stadium,
            'timezone': timezone,
            'timezone_offset': TIMEZONE_OFFSETS[timezone]
        })

    df = pd.DataFrame(data)
    return df.sort_values('team')


def validate_matrix(df: pd.DataFrame) -> dict:
    """
    Validate distance matrix for consistency and known distances.

    Args:
        df: Distance matrix DataFrame

    Returns:
        Dictionary with validation results
    """
    results = {
        'is_symmetric': True,
        'diagonal_zeros': True,
        'known_distances': {},
        'max_distance': 0.0,
        'min_distance': 0.0,
        'coast_to_coast': {}
    }

    # Check symmetry
    diff = np.abs(df.values - df.values.T)
    results['is_symmetric'] = bool(np.max(diff) < 1.0)  # Allow 1 mile tolerance

    # Check diagonal is zero
    results['diagonal_zeros'] = bool(np.max(np.abs(np.diag(df.values))) < 0.1)

    # Check known distances (approximate)
    known_pairs = [
        ('SEA', 'MIA', 2724),  # Seattle to Miami (coast-to-coast)
        ('SF', 'NE', 2699),    # San Francisco to New England
        ('LAR', 'NYG', 2448),  # Los Angeles to New York
        ('GB', 'TB', 1200),    # Green Bay to Tampa Bay
    ]

    for team1, team2, expected_dist in known_pairs:
        if team1 in df.index and team2 in df.columns:
            actual_dist = df.loc[team1, team2]
            error = abs(actual_dist - expected_dist)
            results['known_distances'][f'{team1}-{team2}'] = {
                'expected': expected_dist,
                'actual': round(actual_dist, 1),
                'error': round(error, 1),
                'within_tolerance': bool(error < 100)  # 100 mile tolerance
            }

    # Find max/min distances (excluding diagonal)
    mask = ~np.eye(len(df), dtype=bool)
    results['max_distance'] = round(df.values[mask].max(), 1)
    results['min_distance'] = round(df.values[mask].min(), 1)

    # Coast-to-coast distances
    west_coast = ['SEA', 'SF', 'LAR', 'LAC', 'LV']
    east_coast = ['MIA', 'NE', 'NYG', 'NYJ', 'PHI', 'WAS', 'BAL']

    for west in west_coast:
        for east in east_coast:
            if west in df.index and east in df.columns:
                dist = df.loc[west, east]
                if dist > 2500:  # Coast-to-coast threshold
                    results['coast_to_coast'][f'{west}-{east}'] = round(dist, 1)

    return results


def main():
    """Main execution function."""
    print("=" * 80)
    print("NFL TRAVEL DISTANCE MATRIX BUILDER")
    print("=" * 80)
    print()

    # Create data directory
    data_dir = project_root / 'data'
    data_dir.mkdir(exist_ok=True)

    # Build distance matrix
    print("Building distance matrix...")
    distance_matrix = build_distance_matrix()

    # Build timezone matrix
    print("Building timezone shift matrix...")
    timezone_matrix = build_timezone_matrix()

    # Export stadium reference
    print("Exporting stadium reference table...")
    stadium_ref = export_stadium_reference()

    # Validate matrix
    print("\nValidating distance matrix...")
    validation = validate_matrix(distance_matrix)

    # Print validation results
    print("\nValidation Results:")
    print(f"  - Matrix is symmetric: {validation['is_symmetric']}")
    print(f"  - Diagonal is zero: {validation['diagonal_zeros']}")
    print(f"  - Max distance: {validation['max_distance']} miles")
    print(f"  - Min distance: {validation['min_distance']} miles")
    print()

    print("Known distance validations:")
    for pair, metrics in validation['known_distances'].items():
        status = "✓" if metrics['within_tolerance'] else "✗"
        print(f"  {status} {pair}: {metrics['actual']} miles "
              f"(expected {metrics['expected']}, error {metrics['error']} mi)")
    print()

    print("Coast-to-coast distances (>2500 miles):")
    for pair, dist in sorted(validation['coast_to_coast'].items(),
                            key=lambda x: x[1], reverse=True)[:10]:
        print(f"  - {pair}: {dist} miles")
    print()

    # Save outputs
    output_files = {
        'stadium_distance_matrix.csv': distance_matrix,
        'stadium_timezone_matrix.csv': timezone_matrix,
        'stadium_reference.csv': stadium_ref
    }

    for filename, df in output_files.items():
        output_path = data_dir / filename
        df.to_csv(output_path)
        print(f"✓ Saved: {output_path} ({df.shape[0]} x {df.shape[1]})")

    # Save validation results
    validation_path = data_dir / 'distance_matrix_validation.json'
    with open(validation_path, 'w') as f:
        json.dump(validation, f, indent=2)
    print(f"✓ Saved: {validation_path}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Stadium distance matrix: 32 x 32 teams")
    print(f"Longest distance: {validation['max_distance']} miles")
    print(f"Shortest distance: {validation['min_distance']} miles")
    print(f"Coast-to-coast pairs: {len(validation['coast_to_coast'])}")
    print()
    print("Files created:")
    for filename in output_files.keys():
        print(f"  - data/{filename}")
    print("  - data/distance_matrix_validation.json")
    print()
    print("✓ Distance matrix build complete!")


if __name__ == '__main__':
    main()
