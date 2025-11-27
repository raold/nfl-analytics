# Weather Features Integration Patch

**Date:** November 27, 2025
**Files to Modify:** 2 files (team_offensive_features.py, team_defensive_features.py)
**Lines to Add:** ~75 lines per file
**Status:** Ready to apply

---

## Patch for team_offensive_features.py

### Location 1: Update module docstring

**Find:** (Line ~20)
```python
**Features (27 total):**
```

**Replace with:**
```python
**Features (33 total):**
```

**And add after line ~56:**
```python

6. Weather Metrics (NEW - November 27, 2025):
   - high_wind: Boolean (wind > 15 mph / 24 km/h kills passing efficiency ~0.3 EPA)
   - cold_weather: Boolean (temp < 32°F / 0°C affects ball handling)
   - dome_game: Boolean (controlled environment, no weather effects)
   - precipitation: Boolean (rain/snow impacts passing and ball security)
   - temp_c: Temperature in Celsius (raw value for modeling)
   - wind_kph: Wind speed in km/h (raw value for modeling)
```

### Location 2: Add weather feature method

**Add before `extract_features()` method** (around line ~665):

```python
    def _compute_weather_features(
        self,
        game_id: str
    ) -> Dict[str, float]:
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

        Returns:
            Dictionary with weather features
        """
        from py.features.weather_features_integration import compute_weather_features
        return compute_weather_features(game_id, self.db_config)
```

### Location 3: Update extract_features() signature

**Find:** (around line ~665)
```python
    def extract_features(
        self,
        team: str,
        season: int,
        week: int
    ) -> Dict[str, float]:
        """
        Extract all 27 team offensive features for a specific game.
```

**Replace with:**
```python
    def extract_features(
        self,
        team: str,
        season: int,
        week: int,
        game_id: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Extract all 33 team offensive features for a specific game (including weather).
```

### Location 4: Add to extract_features() method docstring

**Find:** (in extract_features docstring)
```python
        Args:
            team: Team abbreviation (e.g., 'KC', 'BUF')
            season: Season year
            week: Week number

        Returns:
            Dictionary with 27 offensive features
```

**Replace with:**
```python
        Args:
            team: Team abbreviation (e.g., 'KC', 'BUF')
            season: Season year
            week: Week number
            game_id: Optional game_id for weather features. If not provided, weather features will be None.

        Returns:
            Dictionary with 33 offensive features (27 performance + 6 weather)
```

### Location 5: Add weather features computation

**Find:** (in extract_features() method, around line ~700)
```python
        # Compute features
        play_features = self._compute_play_level_features(team, recent_games)
        red_zone_features = self._compute_red_zone_features(team, recent_games)
        game_features = self._compute_game_level_features(team, recent_games)
        variance_features = self._compute_variance_features(team, recent_games)

        # Merge all features
        features = {**play_features, **red_zone_features, **game_features, **variance_features}

        return features
```

**Replace with:**
```python
        # Compute features
        play_features = self._compute_play_level_features(team, recent_games)
        red_zone_features = self._compute_red_zone_features(team, recent_games)
        game_features = self._compute_game_level_features(team, recent_games)
        variance_features = self._compute_variance_features(team, recent_games)

        # Compute weather features if game_id provided
        weather_features = {}
        if game_id:
            weather_features = self._compute_weather_features(game_id)
        else:
            # Return defaults if no game_id provided
            weather_features = {
                'high_wind': None,
                'cold_weather': None,
                'dome_game': None,
                'precipitation': None,
                'temp_c': None,
                'wind_kph': None,
            }

        # Merge all features
        features = {**play_features, **red_zone_features, **game_features, **variance_features, **weather_features}

        return features
```

---

## Patch for team_defensive_features.py

### Location 1: Update module docstring

**Find:** (Line ~9)
```python
Computes 24 team defensive features using rolling 4-game average with H=4 exponential decay.
```

**Replace with:**
```python
Computes 30 team defensive features using rolling 4-game average with H=4 exponential decay.
```

**Find:** (Line ~10)
```python
**Features Computed (24 total):**
```

**Replace with:**
```python
**Features Computed (30 total):**
```

**And add after the existing 5 feature groups (around line ~44):**
```python

6. Weather Metrics (NEW - November 27, 2025):
   - high_wind: Boolean (wind > 15 mph / 24 km/h kills passing efficiency)
   - cold_weather: Boolean (temp < 32°F / 0°C affects ball handling)
   - dome_game: Boolean (controlled environment, no weather effects)
   - precipitation: Boolean (rain/snow impacts passing and ball security)
   - temp_c: Temperature in Celsius (raw value for modeling)
   - wind_kph: Wind speed in km/h (raw value for modeling)
```

### Location 2: Add weather feature method

**Add before `extract_features()` method** (around line ~796):

```python
    def _compute_weather_features(
        self,
        game_id: str
    ) -> Dict[str, float]:
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

        Returns:
            Dictionary with weather features
        """
        from py.features.weather_features_integration import compute_weather_features
        return compute_weather_features(game_id, self.db_config)
```

### Location 3: Update extract_features() signature

**Find:** (around line ~796)
```python
    def extract_features(
        self,
        team: str,
        season: int,
        week: int
    ) -> Dict[str, float]:
        """
        Extract all 24 defensive features for a team at a specific game.
```

**Replace with:**
```python
    def extract_features(
        self,
        team: str,
        season: int,
        week: int,
        game_id: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Extract all 30 defensive features for a team at a specific game (including weather).
```

### Location 4: Update extract_features() docstring

**Find:** (in extract_features docstring)
```python
        Args:
            team: Team abbreviation (e.g., 'KC', 'BUF', 'BAL')
            season: Target season year
            week: Target week number

        Returns:
            Dictionary with 24 features:
```

**Replace with:**
```python
        Args:
            team: Team abbreviation (e.g., 'KC', 'BUF', 'BAL')
            season: Target season year
            week: Target week number
            game_id: Optional game_id for weather features. If not provided, weather features will be None.

        Returns:
            Dictionary with 30 features (24 defensive + 6 weather):
```

**And add at the end of the Returns dict (before the closing brace):**
```python
                # Weather metrics (6)
                'high_wind': float,
                'cold_weather': float,
                'dome_game': float,
                'precipitation': float,
                'temp_c': float,
                'wind_kph': float,
```

### Location 5: Add weather features computation

**Find:** (in extract_features() method, around line ~875)
```python
        # Compute all feature groups
        epa_features = self._compute_epa_features(team, recent_games, weights)
        explosive_features = self._compute_explosive_efficiency_features(
            team, recent_games, weights
        )
        passing_features = self._compute_passing_defense_features(
            team, recent_games, weights
        )
        rushing_features = self._compute_rushing_defense_features(
            team, recent_games, weights
        )
        game_features = self._compute_game_level_features(
            team, recent_games, weights
        )

        # Merge all features
        all_features = {
            **epa_features,
            **explosive_features,
            **passing_features,
            **rushing_features,
            **game_features,
        }

        return all_features
```

**Replace with:**
```python
        # Compute all feature groups
        epa_features = self._compute_epa_features(team, recent_games, weights)
        explosive_features = self._compute_explosive_efficiency_features(
            team, recent_games, weights
        )
        passing_features = self._compute_passing_defense_features(
            team, recent_games, weights
        )
        rushing_features = self._compute_rushing_defense_features(
            team, recent_games, weights
        )
        game_features = self._compute_game_level_features(
            team, recent_games, weights
        )

        # Compute weather features if game_id provided
        weather_features = {}
        if game_id:
            weather_features = self._compute_weather_features(game_id)
        else:
            # Return defaults if no game_id provided
            weather_features = {
                'high_wind': None,
                'cold_weather': None,
                'dome_game': None,
                'precipitation': None,
                'temp_c': None,
                'wind_kph': None,
            }

        # Merge all features
        all_features = {
            **epa_features,
            **explosive_features,
            **passing_features,
            **rushing_features,
            **game_features,
            **weather_features,
        }

        return all_features
```

---

## Testing After Integration

### Test team_offensive_features.py

```bash
source .venv/bin/activate

# Test with game_id (weather features should be populated)
python -c "
from py.features.team_offensive_features import TeamOffensiveFeatures
engine = TeamOffensiveFeatures()
features = engine.extract_features('KC', 2019, 19, game_id='2019_19_HOU_KC')
print('Features extracted:', len(features))
print('high_wind:', features['high_wind'])
print('cold_weather:', features['cold_weather'])
print('temp_c:', features['temp_c'])
"

# Expected output:
# Features extracted: 33
# high_wind: 1.0
# cold_weather: 1.0
# temp_c: -7.2
```

### Test team_defensive_features.py

```bash
source .venv/bin/activate

# Test with game_id (weather features should be populated)
python -c "
from py.features.team_defensive_features import TeamDefensiveFeatures
engine = TeamDefensiveFeatures()
features = engine.extract_features('KC', 2019, 19, game_id='2019_19_HOU_KC')
print('Features extracted:', len(features))
print('high_wind:', features['high_wind'])
print('dome_game:', features['dome_game'])
print('wind_kph:', features['wind_kph'])
"

# Expected output:
# Features extracted: 30
# high_wind: 1.0
# dome_game: 0.0
# wind_kph: 27.36
```

---

## Summary of Changes

| File | Lines Added | Features Added | Total Features |
|------|-------------|----------------|----------------|
| team_offensive_features.py | ~75 lines | 6 weather | 27 → 33 |
| team_defensive_features.py | ~75 lines | 6 weather | 24 → 30 |

**Total:** 150 lines added, 12 new features (6 per module, same features for offense and defense)

**Deployment:** Apply patches above, run tests, verify 33/30 features extracted correctly.

---

**Author:** Claude Code (Sonnet 4.5)
**Date:** November 27, 2025
**Status:** Ready to apply
