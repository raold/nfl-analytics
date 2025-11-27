#!/usr/bin/env python3
"""
Game Feature Extractor - Unified Offensive + Defensive Feature Pipeline
========================================================================

Combines offensive and defensive features for both teams in a matchup to create
comprehensive game-level features for outcome predictions (spreads, totals, moneylines).

**Purpose:**
Extract all 96 base features (24 offensive + 24 defensive × 2 teams) plus 4 derived
matchup features plus 12 injury features for game outcome predictions. Single entry point
that coordinates feature extraction from TeamOffensiveFeatures, TeamDefensiveFeatures,
and mart.team_injury_load.

**Features Extracted (112 total):**

1. Home Team Offensive (24):
   - Prefix: 'home_off_'
   - Source: TeamOffensiveFeatures.extract_features(home_team)
   - Examples: home_off_epa_per_play, home_off_success_rate, home_off_points_per_game

2. Home Team Defensive (24):
   - Prefix: 'home_def_'
   - Source: TeamDefensiveFeatures.extract_features(home_team)
   - Examples: home_def_epa_allowed_per_play, home_def_sack_rate, home_def_points_allowed_per_game

3. Away Team Offensive (24):
   - Prefix: 'away_off_'
   - Source: TeamOffensiveFeatures.extract_features(away_team)
   - Examples: away_off_epa_per_play, away_off_success_rate, away_off_points_per_game

4. Away Team Defensive (24):
   - Prefix: 'away_def_'
   - Source: TeamDefensiveFeatures.extract_features(away_team)
   - Examples: away_def_epa_allowed_per_play, away_def_sack_rate, away_def_points_allowed_per_game

5. Derived Matchup Features (4):
   - offensive_mismatch: home_off_epa - away_def_epa (positive = home advantage)
   - defensive_mismatch: away_off_epa - home_def_epa (positive = away advantage)
   - pace_difference: home_plays_per_game - away_plays_per_game (positive = home faster)
   - rest_differential: Currently 0 (future: days since last game difference)

6. Injury Features (12):
   - Home team (5): home_injury_total, home_injury_severity_index, home_qb_out,
     home_key_position_out, home_oline_injuries
   - Away team (5): away_injury_total, away_injury_severity_index, away_qb_out,
     away_key_position_out, away_oline_injuries
   - Derived (2): injury_severity_diff, injury_spread_adjustment
   - Source: mart.team_injury_load table (2,798 rows, 2020-2024)
   - Impact: QB injury = -5 pts, key position = -2 pts, oline = -0.5 pts/injury

**Temporal Safety:**
All features use ONLY games BEFORE target week. Both underlying extractors implement
temporal safety via shift(1) and explicit week filtering.

**Performance:**
- Target: <500ms for all 4 feature sets (4 × ~100ms base extraction)
- Caching: 5-minute TTL for recently extracted features
- Vectorized operations in underlying extractors minimize computation time

**Integration:**
Primary entry point for game outcome prediction models. Replaces manual feature
extraction in GameOutcomePredictor and other game-level prediction modules.

**Schema Compliance:**
- Uses canonical patterns from docs/database/SQL_PATTERNS.md
- Validated against docs/database/DATABASE_QUICK_REFERENCE.md
- Primary tables: plays (1.2M rows), games (6,991 rows)

**Expected Impact:**
- Simplifies feature extraction API (1 call vs 4)
- Ensures consistency in feature computation across models
- Reduces code duplication and maintenance burden
- Enables easy addition of new derived features

Author: Claude Code (Sonnet 4.5)
Date: November 21, 2025
Task: Create unified game feature extraction pipeline
Reference: CLAUDE.md § Feature Engineering Best Practices
"""

import logging
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from py.features.team_offensive_features import TeamOffensiveFeatures
from py.features.team_defensive_features import TeamDefensiveFeatures

import psycopg2
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class GameFeatureExtractor:
    """
    Extract comprehensive game-level features combining offensive and defensive metrics.

    **Why This Matters:**
    Game outcomes depend on matchup dynamics: how does Team A's offense match up against
    Team B's defense, and vice versa? This class provides a unified interface to extract
    all relevant features in a single call, ensuring consistency and temporal safety.

    **Usage:**
    >>> extractor = GameFeatureExtractor()
    >>> features = extractor.extract_game_features('KC', 'BUF', 2025, 12)
    >>> len(features)  # 100 features
    100
    >>> features['offensive_mismatch']  # Derived matchup feature
    0.15

    **Performance:**
    - Target: <500ms per game
    - Caching: 5-minute TTL for repeated extractions
    - Parallel extraction: Future optimization (async/await)
    """

    # Cache TTL in seconds (5 minutes)
    CACHE_TTL = 300

    def __init__(self, db_config: Optional[Dict] = None):
        """
        Initialize game feature extractor with database connection.

        Args:
            db_config: Database configuration dict. If None, uses default connection:
                {
                    'host': 'localhost',
                    'port': 5544,
                    'dbname': 'devdb01',
                    'user': 'dro',
                    'password': 'sicillionbillions'
                }
        """
        self.offensive_extractor = TeamOffensiveFeatures(db_config=db_config)
        self.defensive_extractor = TeamDefensiveFeatures(db_config=db_config)

        # Feature cache: {(team, season, week, feature_type): (features, timestamp)}
        self._feature_cache = {}

        logger.info("GameFeatureExtractor initialized successfully")

    def extract_game_features(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: int,
        use_cache: bool = True
    ) -> Dict[str, float]:
        """
        Extract all 112 features for a game matchup (includes injury features).

        This is the main entry point for game-level feature extraction.

        Args:
            home_team: Home team abbreviation (e.g., 'KC', 'BUF')
            away_team: Away team abbreviation (e.g., 'SF', 'PHI')
            season: Target season year
            week: Target week number
            use_cache: Whether to use cached features (5-minute TTL)

        Returns:
            Dictionary with 112 features:
            {
                # Home team offensive features (24)
                'home_off_epa_per_play': float,
                'home_off_success_rate': float,
                ... (24 total)

                # Home team defensive features (24)
                'home_def_epa_allowed_per_play': float,
                'home_def_success_rate_allowed': float,
                ... (24 total)

                # Away team offensive features (24)
                'away_off_epa_per_play': float,
                ... (24 total)

                # Away team defensive features (24)
                'away_def_epa_allowed_per_play': float,
                ... (24 total)

                # Matchup-specific features (4)
                'offensive_mismatch': float,  # home_off_epa - away_def_epa
                'defensive_mismatch': float,   # away_off_epa - home_def_epa
                'pace_difference': float,      # home_plays_per_game - away_plays_per_game
                'rest_differential': int,      # Currently 0 (future enhancement)

                # Injury features (12)
                'home_injury_total': int,
                'home_injury_severity_index': float,
                'home_qb_out': int,  # 0/1
                'home_key_position_out': int,  # 0/1
                'home_oline_injuries': int,
                'away_injury_total': int,
                'away_injury_severity_index': float,
                'away_qb_out': int,  # 0/1
                'away_key_position_out': int,  # 0/1
                'away_oline_injuries': int,
                'injury_severity_diff': float,  # home - away
                'injury_spread_adjustment': float,  # estimated spread impact (-5 for QB, etc.)
            }

        Raises:
            ValueError: If team abbreviations are invalid or season/week are out of range
            RuntimeError: If feature extraction fails for any team
        """
        start_time = time.time()

        # Validate inputs
        self._validate_inputs(home_team, away_team, season, week)

        # Extract base features (with caching)
        home_off = self._get_features(home_team, season, week, 'offensive', use_cache)
        home_def = self._get_features(home_team, season, week, 'defensive', use_cache)
        away_off = self._get_features(away_team, season, week, 'offensive', use_cache)
        away_def = self._get_features(away_team, season, week, 'defensive', use_cache)

        # Combine with prefixes
        features = {}
        features.update({f'home_off_{k}': v for k, v in home_off.items()})
        features.update({f'home_def_{k}': v for k, v in home_def.items()})
        features.update({f'away_off_{k}': v for k, v in away_off.items()})
        features.update({f'away_def_{k}': v for k, v in away_def.items()})

        # Add derived matchup features
        features['offensive_mismatch'] = self._compute_offensive_mismatch(home_off, away_def)
        features['defensive_mismatch'] = self._compute_defensive_mismatch(away_off, home_def)
        features['pace_difference'] = self._compute_pace_difference(home_off, away_off)
        features['rest_differential'] = 0  # Future: days_since_last_game

        # Add injury features
        try:
            injury_features = self._get_injury_features(home_team, away_team, season, week)
            features.update(injury_features)
            logger.debug(
                f"Added {len(injury_features)} injury features "
                f"(spread_adj={injury_features.get('injury_spread_adjustment', 0):.1f})"
            )
        except Exception as e:
            logger.warning(f"Failed to extract injury features: {e}. Using default values.")
            # Add default injury features (all zeros)
            features.update({
                'home_injury_total': 0,
                'home_injury_severity_index': 0.0,
                'home_qb_out': 0,
                'home_key_position_out': 0,
                'home_oline_injuries': 0,
                'away_injury_total': 0,
                'away_injury_severity_index': 0.0,
                'away_qb_out': 0,
                'away_key_position_out': 0,
                'away_oline_injuries': 0,
                'injury_severity_diff': 0.0,
                'injury_spread_adjustment': 0.0
            })

        # Validate: Check for NaN values
        nan_features = [k for k, v in features.items() if pd.isna(v)]
        if nan_features:
            logger.warning(
                f"Found {len(nan_features)} NaN values for {home_team} vs {away_team} "
                f"({season} Week {week}): {nan_features[:5]}... Setting to 0.0"
            )
            for key in nan_features:
                features[key] = 0.0

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Extracted {len(features)} features for {home_team} vs {away_team} "
            f"({season} Week {week}) in {elapsed_ms:.1f}ms"
        )

        return features

    def validate_temporal_safety(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: int
    ) -> bool:
        """
        Validate all features are from BEFORE target week (no leakage).

        This delegates to underlying extractors' validate_no_leakage methods.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: Target season year
            week: Target week number

        Returns:
            True if all features are temporally safe, False otherwise
        """
        checks = [
            self.offensive_extractor.validate_no_leakage(home_team, season, week),
            self.offensive_extractor.validate_no_leakage(away_team, season, week),
            self.defensive_extractor.validate_no_leakage(home_team, season, week),
            self.defensive_extractor.validate_no_leakage(away_team, season, week)
        ]

        if all(checks):
            logger.info(
                f"Temporal safety validated: All features for {home_team} vs {away_team} "
                f"({season} Week {week}) use ONLY games before target week"
            )
            return True
        else:
            logger.error(
                f"Temporal leakage detected for {home_team} vs {away_team} "
                f"({season} Week {week}): {checks}"
            )
            return False

    def clear_cache(self):
        """Clear the feature cache (useful for testing or manual refresh)."""
        cache_size = len(self._feature_cache)
        self._feature_cache.clear()
        logger.info(f"Cleared feature cache ({cache_size} entries)")

    def _get_features(
        self,
        team: str,
        season: int,
        week: int,
        feature_type: str,
        use_cache: bool
    ) -> Dict[str, float]:
        """
        Get features with caching.

        Args:
            team: Team abbreviation
            season: Season year
            week: Week number
            feature_type: 'offensive' or 'defensive'
            use_cache: Whether to use cached features

        Returns:
            Dictionary of features
        """
        cache_key = (team, season, week, feature_type)

        # Check cache
        if use_cache and cache_key in self._feature_cache:
            features, timestamp = self._feature_cache[cache_key]
            age_seconds = (datetime.now() - timestamp).total_seconds()

            if age_seconds < self.CACHE_TTL:
                logger.debug(
                    f"Cache hit: {team} {feature_type} features "
                    f"({season} Week {week}, age={age_seconds:.1f}s)"
                )
                return features
            else:
                logger.debug(
                    f"Cache expired: {team} {feature_type} features "
                    f"({season} Week {week}, age={age_seconds:.1f}s)"
                )
                del self._feature_cache[cache_key]

        # Extract features
        start_time = time.time()
        if feature_type == 'offensive':
            features = self.offensive_extractor.extract_features(team, season, week)
        elif feature_type == 'defensive':
            features = self.defensive_extractor.extract_features(team, season, week)
        else:
            raise ValueError(f"Invalid feature_type: {feature_type}")

        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(
            f"Extracted {team} {feature_type} features "
            f"({season} Week {week}) in {elapsed_ms:.1f}ms"
        )

        # Update cache
        if use_cache:
            self._feature_cache[cache_key] = (features, datetime.now())

        return features

    def _compute_offensive_mismatch(
        self,
        home_off: Dict[str, float],
        away_def: Dict[str, float]
    ) -> float:
        """
        Compute offensive mismatch: home offense vs away defense.

        Positive value = home offensive advantage
        Negative value = away defensive advantage

        Args:
            home_off: Home team offensive features
            away_def: Away team defensive features

        Returns:
            Offensive mismatch score (EPA scale)
        """
        home_off_epa = home_off.get('off_epa_per_play', 0.0)
        away_def_epa = away_def.get('def_epa_allowed_per_play', 0.0)

        mismatch = home_off_epa - away_def_epa
        return mismatch

    def _compute_defensive_mismatch(
        self,
        away_off: Dict[str, float],
        home_def: Dict[str, float]
    ) -> float:
        """
        Compute defensive mismatch: away offense vs home defense.

        Positive value = away offensive advantage
        Negative value = home defensive advantage

        Args:
            away_off: Away team offensive features
            home_def: Home team defensive features

        Returns:
            Defensive mismatch score (EPA scale)
        """
        away_off_epa = away_off.get('off_epa_per_play', 0.0)
        home_def_epa = home_def.get('def_epa_allowed_per_play', 0.0)

        mismatch = away_off_epa - home_def_epa
        return mismatch

    def _compute_pace_difference(
        self,
        home_off: Dict[str, float],
        away_off: Dict[str, float]
    ) -> float:
        """
        Compute pace difference: home plays/game - away plays/game.

        Positive value = home faster pace
        Negative value = away faster pace

        Args:
            home_off: Home team offensive features
            away_off: Away team offensive features

        Returns:
            Pace difference (plays per game)
        """
        home_pace = home_off.get('off_plays_per_game', 0.0)
        away_pace = away_off.get('off_plays_per_game', 0.0)

        pace_diff = home_pace - away_pace
        return pace_diff

    def _get_injury_features(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: int
    ) -> Dict[str, float]:
        """
        Extract injury load features for both teams.

        Queries mart.team_injury_load table for injury data and computes
        derived features including spread adjustments based on injury impact.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: Target season
            week: Target week

        Returns:
            Dictionary with injury features:
            {
                # Home team injury features
                'home_injury_total': int,
                'home_injury_severity_index': float,
                'home_qb_out': int (0/1),
                'home_key_position_out': int (0/1),
                'home_oline_injuries': int,

                # Away team injury features
                'away_injury_total': int,
                'away_injury_severity_index': float,
                'away_qb_out': int (0/1),
                'away_key_position_out': int (0/1),
                'away_oline_injuries': int,

                # Derived features
                'injury_severity_diff': float,  # home - away
                'injury_spread_adjustment': float,  # estimated spread impact
            }
        """
        # Connect to database
        conn = psycopg2.connect(
            host='localhost',
            port=5544,
            dbname='devdb01',
            user='dro',
            password=os.getenv('DB_PASSWORD', 'sicillionbillions')
        )

        try:
            with conn.cursor() as cur:
                # Query injury data for both teams
                query = """
                SELECT
                    team,
                    total_injuries,
                    injury_severity_index,
                    qb_out,
                    key_position_out,
                    oline_injuries
                FROM mart.team_injury_load
                WHERE season = %s
                  AND week = %s
                  AND team IN (%s, %s)
                """
                cur.execute(query, (season, week, home_team, away_team))
                rows = cur.fetchall()
        finally:
            conn.close()

        # Parse results
        injury_data = {}
        for row in rows:
            team = row[0]
            injury_data[team] = {
                'total_injuries': row[1] or 0,
                'injury_severity_index': row[2] or 0.0,
                'qb_out': 1 if row[3] else 0,
                'key_position_out': 1 if row[4] else 0,
                'oline_injuries': row[5] or 0
            }

        # Extract home and away data (default to zeros if missing)
        home_data = injury_data.get(home_team, {
            'total_injuries': 0,
            'injury_severity_index': 0.0,
            'qb_out': 0,
            'key_position_out': 0,
            'oline_injuries': 0
        })
        away_data = injury_data.get(away_team, {
            'total_injuries': 0,
            'injury_severity_index': 0.0,
            'qb_out': 0,
            'key_position_out': 0,
            'oline_injuries': 0
        })

        # Compute derived features
        injury_severity_diff = home_data['injury_severity_index'] - away_data['injury_severity_index']

        # Spread adjustment based on injury impact weights
        # Positive = home team more injured (worse for home team)
        # Negative = away team more injured (better for home team)
        spread_adjustment = 0.0

        # QB injury impact: -5 points
        spread_adjustment += home_data['qb_out'] * 5.0
        spread_adjustment -= away_data['qb_out'] * 5.0

        # Key position injury impact: -2 points
        spread_adjustment += home_data['key_position_out'] * 2.0
        spread_adjustment -= away_data['key_position_out'] * 2.0

        # O-line injury impact: -0.5 points per injury
        spread_adjustment += home_data['oline_injuries'] * 0.5
        spread_adjustment -= away_data['oline_injuries'] * 0.5

        return {
            # Home team
            'home_injury_total': home_data['total_injuries'],
            'home_injury_severity_index': home_data['injury_severity_index'],
            'home_qb_out': home_data['qb_out'],
            'home_key_position_out': home_data['key_position_out'],
            'home_oline_injuries': home_data['oline_injuries'],

            # Away team
            'away_injury_total': away_data['total_injuries'],
            'away_injury_severity_index': away_data['injury_severity_index'],
            'away_qb_out': away_data['qb_out'],
            'away_key_position_out': away_data['key_position_out'],
            'away_oline_injuries': away_data['oline_injuries'],

            # Derived
            'injury_severity_diff': injury_severity_diff,
            'injury_spread_adjustment': spread_adjustment
        }

    def _validate_inputs(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: int
    ):
        """
        Validate input parameters.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: Season year
            week: Week number

        Raises:
            ValueError: If any input is invalid
        """
        # Validate team abbreviations (3-letter codes)
        if not isinstance(home_team, str) or len(home_team) < 2 or len(home_team) > 3:
            raise ValueError(f"Invalid home_team: {home_team}")
        if not isinstance(away_team, str) or len(away_team) < 2 or len(away_team) > 3:
            raise ValueError(f"Invalid away_team: {away_team}")

        # Validate season (1999-2099)
        if not isinstance(season, int) or season < 1999 or season > 2099:
            raise ValueError(f"Invalid season: {season}")

        # Validate week (1-22: regular season 1-18, playoffs 19-22)
        if not isinstance(week, int) or week < 1 or week > 22:
            raise ValueError(f"Invalid week: {week}")

        # Validate teams are different
        if home_team == away_team:
            raise ValueError(f"Home and away teams cannot be the same: {home_team}")


def main():
    """
    Example usage and validation.

    This demonstrates how to use GameFeatureExtractor for a real game.
    """
    # Initialize extractor
    extractor = GameFeatureExtractor()

    # Example: Current season Week 12 - HOU vs BUF
    from datetime import datetime
    home_team = 'HOU'
    away_team = 'BUF'
    season = datetime.now().year if datetime.now().month >= 9 else datetime.now().year - 1
    week = 12

    logger.info(f"\n{'='*80}")
    logger.info(f"Example: {home_team} vs {away_team} ({season} Week {week})")
    logger.info(f"{'='*80}\n")

    # Extract features
    start_time = time.time()
    features = extractor.extract_game_features(home_team, away_team, season, week)
    elapsed_ms = (time.time() - start_time) * 1000

    # Display results
    logger.info(f"\nFeature Extraction Complete:")
    logger.info(f"- Total features: {len(features)}")
    logger.info(f"- Extraction time: {elapsed_ms:.1f}ms")
    logger.info(f"\nSample Features:")

    sample_keys = [
        'home_off_epa_per_play',
        'home_def_epa_allowed_per_play',
        'away_off_epa_per_play',
        'away_def_epa_allowed_per_play',
        'offensive_mismatch',
        'defensive_mismatch',
        'pace_difference',
        'rest_differential'
    ]

    for key in sample_keys:
        if key in features:
            logger.info(f"  {key}: {features[key]:.4f}")

    # Validate temporal safety
    logger.info(f"\nTemporal Safety Check:")
    is_safe = extractor.validate_temporal_safety(home_team, away_team, season, week)
    logger.info(f"  Temporal safety: {'✓ PASSED' if is_safe else '✗ FAILED'}")

    # Check for NaN values
    nan_count = sum(1 for v in features.values() if pd.isna(v))
    logger.info(f"\nData Quality:")
    logger.info(f"  NaN values: {nan_count}/{len(features)} {'✓' if nan_count == 0 else '✗'}")

    logger.info(f"\n{'='*80}\n")

    return features


if __name__ == "__main__":
    main()
