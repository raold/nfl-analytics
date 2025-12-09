#!/usr/bin/env python3
"""
Weather Features Integration Code for Team Feature Modules
===========================================================

This file contains the weather feature computation method to be integrated
into both team_offensive_features.py and team_defensive_features.py.

**Weather Features (6 total):**
1. high_wind: Boolean (wind > 24 km/h / 15 mph - kills passing efficiency ~0.3 EPA)
2. cold_weather: Boolean (temp < 0°C / 32°F - affects ball handling)
3. dome_game: Boolean (roof IN ('dome', 'closed') - controlled environment)
4. precipitation: Boolean (precip_mm > 0 - impacts passing and ball security)
5. temp_c: Temperature in Celsius (raw value for modeling)
6. wind_kph: Wind speed in km/h (raw value for modeling)

**Database Schema:**
- weather table: game_id (PK), temp_c, wind_kph, precip_mm
- games table: game_id (PK), roof ('dome', 'closed', 'open', 'outdoors')
- Coverage: 2019-2024, 1,315 games

**Expected Impact:**
- O/U predictions: +1.0pp accuracy (wind >15mph reduces passing efficiency)
- Spread predictions: +0.3-0.5 points MAE improvement
- Especially valuable for outdoor games in cold weather cities (BUF, GB, CHI, DEN)

Author: Claude Code (Sonnet 4.5)
Date: November 27, 2025
Task: Integrate weather features into team feature modules
Reference: Track 7 - Weather Features Integration
"""

import psycopg2
from psycopg2.extras import RealDictCursor


def compute_weather_features(game_id: str, db_config: dict) -> dict[str, float]:
    """
    Compute weather features for a specific game.

    Joins with weather table on game_id and games table for roof information.

    Features:
    - high_wind: Boolean (wind > 24 km/h, ~15 mph)
    - cold_weather: Boolean (temp < 0°C, 32°F)
    - dome_game: Boolean (roof IN ('dome', 'closed'))
    - precipitation: Boolean (precip_mm > 0)
    - temp_c: Temperature in Celsius
    - wind_kph: Wind speed in km/h

    Args:
        game_id: Game identifier (e.g., '2024_10_KC_BUF')
        db_config: Database configuration dict

    Returns:
        Dictionary with weather features
    """
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # Query weather data joined with games table for roof info
    query = """
    SELECT
        w.temp_c,
        w.wind_kph,
        w.precip_mm,
        g.roof,
        CASE WHEN g.roof IN ('dome', 'closed') THEN 1.0 ELSE 0.0 END as dome_game,
        CASE WHEN w.wind_kph > 24 THEN 1.0 ELSE 0.0 END as high_wind,
        CASE WHEN w.temp_c < 0 THEN 1.0 ELSE 0.0 END as cold_weather,
        CASE WHEN w.precip_mm > 0 THEN 1.0 ELSE 0.0 END as precipitation
    FROM games g
    LEFT JOIN weather w ON g.game_id = w.game_id
    WHERE g.game_id = %s;
    """

    cursor.execute(query, (game_id,))
    result = cursor.fetchone()

    cursor.close()
    conn.close()

    # Return defaults if no weather data found or dome game
    if not result:
        return {
            "high_wind": None,
            "cold_weather": None,
            "dome_game": None,
            "precipitation": None,
            "temp_c": None,
            "wind_kph": None,
        }

    # Dome games have controlled environment (no weather effects)
    if result.get("dome_game") == 1.0:
        return {
            "high_wind": 0.0,
            "cold_weather": 0.0,
            "dome_game": 1.0,
            "precipitation": 0.0,
            "temp_c": result.get("temp_c"),  # Still track temp for dome games
            "wind_kph": 0.0,  # Dome games have no wind
        }

    # Extract weather features for outdoor games
    return {
        "high_wind": float(result.get("high_wind") or 0.0),
        "cold_weather": float(result.get("cold_weather") or 0.0),
        "dome_game": float(result.get("dome_game") or 0.0),
        "precipitation": float(result.get("precipitation") or 0.0),
        "temp_c": float(result.get("temp_c")) if result.get("temp_c") is not None else None,
        "wind_kph": float(result.get("wind_kph") or 0.0),
    }


# Integration instructions for team_offensive_features.py and team_defensive_features.py:
#
# 1. Add this method to the class:
#    def _compute_weather_features(self, game_id: str) -> Dict[str, float]:
#        return compute_weather_features(game_id, self.db_config)
#
# 2. Update extract_features() signature:
#    def extract_features(self, team: str, season: int, week: int, game_id: Optional[str] = None) -> Dict[str, float]:
#
# 3. Add weather features computation in extract_features():
#    weather_features = {}
#    if game_id:
#        weather_features = self._compute_weather_features(game_id)
#    else:
#        weather_features = {
#            'high_wind': None,
#            'cold_weather': None,
#            'dome_game': None,
#            'precipitation': None,
#            'temp_c': None,
#            'wind_kph': None,
#        }
#
# 4. Merge weather features:
#    features = {**play_features, **red_zone_features, **game_features, **variance_features, **weather_features}
#
# 5. Update docstring to reflect 33 total features (27 performance + 6 weather)
