"""
NFL Analytics Domain Constants
==============================

All magic numbers should be documented here with:
1. Source/citation for the value
2. Date validated
3. Confidence level (empirical, expert opinion, placeholder)

This file centralizes all hardcoded values from feature extraction modules
to improve maintainability and enable empirical validation.
"""

from dataclasses import dataclass
from typing import Dict

# ==============================================================================
# INJURY IMPACT WEIGHTS
# ==============================================================================
# Source: Empirically validated via OLS regression (December 8, 2025)
# Location: nfl_analytics/features/game_feature_extractor.py:904-914
# Validation Script: analysis/validation/validate_injury_weights.py
#
# Validation Results (n=1,401 games, 2020-2024):
#   - R² = 0.0337 (low predictive power for ATS, as expected)
#   - QB injury: estimated 8.45 pts (using conservative 7.0)
#   - Key position: estimated 1.73 pts (keeping 2.0)
#   - O-line: estimated 0.05 pts (reducing from 0.5 to 0.1)
#
# Key Finding: Away QB out → Home covers 82.9% (n=82 games)


@dataclass(frozen=True)
class InjuryWeights:
    """Spread adjustment weights for injuries by position."""

    # Points subtracted from spread when QB is out
    # Empirical: 8.45 pts | Using conservative 7.0 (up from 5.0)
    # Rationale: Starting QB is most valuable position
    QB_OUT_SPREAD_IMPACT: float = 7.0

    # Points subtracted for key skill position player out
    # Empirical: 1.73 pts | Keeping 2.0 (within error margin)
    # Rationale: RB1, WR1, TE1 have measurable impact
    KEY_POSITION_SPREAD_IMPACT: float = 2.0

    # Points subtracted per offensive line injury
    # Empirical: 0.05 pts | Reducing to 0.1 (from 0.5)
    # Rationale: O-line injuries have minimal marginal ATS impact
    OLINE_INJURY_SPREAD_IMPACT: float = 0.1

    # Positions that qualify as "key position"
    KEY_POSITIONS: tuple = ("RB", "WR", "TE", "LT", "RT", "DE", "DT", "LB", "CB", "S")


INJURY_WEIGHTS = InjuryWeights()


# ==============================================================================
# FEATURE EXTRACTION PARAMETERS
# ==============================================================================
# Source: Industry standard for NFL analytics
# Validation: Tested against 2018-2024 data


@dataclass(frozen=True)
class FeatureParams:
    """Parameters for feature extraction algorithms."""

    # Number of past games to include in rolling averages
    # Rationale: 4 games balances recency with stability
    LOOKBACK_WINDOW: int = 4

    # Exponential decay halflife in games
    # Rationale: Recent performance weighted ~2x older games
    # Game weights: G1=54%, G2=27%, G3=13%, G4=6%
    DECAY_HALFLIFE: float = 4.0

    # Feature cache time-to-live in seconds (5 minutes)
    # Rationale: Prevents redundant DB queries during batch processing
    CACHE_TTL_SECONDS: int = 300

    # Opponent-adjusted features use longer window for stability
    OPPONENT_ADJ_LOOKBACK: int = 8


FEATURE_PARAMS = FeatureParams()


# ==============================================================================
# LEAGUE AVERAGE DEFAULTS
# ==============================================================================
# Source: nflfastR data 2018-2024 season averages
# Used when insufficient data for rolling calculations (early season)


@dataclass(frozen=True)
class LeagueAverages:
    """Default values based on NFL league averages (2018-2024)."""

    # Expected points added per play (league neutral = 0)
    EPA_PER_PLAY: float = 0.00

    # Percentage of plays considered "successful" (>= 0.5 EPA)
    SUCCESS_RATE: float = 0.45

    # Percentage of plays gaining 15+ yards
    EXPLOSIVE_RATE: float = 0.20

    # Percentage of runs stuffed for <= 1 yard
    STUFF_RATE: float = 0.18

    # First down conversion rate
    FIRST_DOWN_RATE: float = 0.35

    # Third down conversion rate
    THIRD_DOWN_CONVERSION: float = 0.35

    # Yards per pass attempt
    YARDS_PER_PASS: float = 6.5

    # Yards per rush attempt
    YARDS_PER_RUSH: float = 4.3

    # Points per game
    POINTS_PER_GAME: float = 23.0


LEAGUE_AVERAGES = LeagueAverages()


# ==============================================================================
# TURNOVER LUCK CONSTANTS
# ==============================================================================
# Source: NFL fumble recovery research (Sharp Football, Football Outsiders)
# Validation: Empirically validated across 1999-2024 data


@dataclass(frozen=True)
class TurnoverLuckParams:
    """Parameters for turnover luck regression analysis."""

    # Expected fumble recovery rate (50-50 is essentially random)
    # Source: NFL historical average is 50.3%
    EXPECTED_FUMBLE_RECOVERY_RATE: float = 0.50

    # Expected turnovers per game (league average)
    # Source: 2018-2024 NFL average
    EXPECTED_TURNOVERS_PER_GAME: float = 1.30

    # Point value of each turnover (field position + scoring opportunity)
    # Source: Football Outsiders research
    POINTS_PER_TURNOVER: float = 4.0


TURNOVER_PARAMS = TurnoverLuckParams()


# ==============================================================================
# PRIMETIME GAME EFFECTS
# ==============================================================================
# Source: Sharp Football historical ATS analysis
# Validation: Pending full backtesting across 2010-2024


@dataclass(frozen=True)
class PrimetimeEffects:
    """Betting market effects for primetime games."""

    # Thursday Night Football road team cover rate
    # Rationale: Short rest disadvantages home team prep
    TNF_ROAD_COVER_RATE: float = 0.63

    # Monday Night Football public betting bias
    # Rationale: Casual bettors overweight primetime exposure
    MNF_PUBLIC_BIAS: float = 0.02

    # Primetime games attract more public money
    PRIMETIME_PUBLIC_SHARE_INCREASE: float = 0.05


PRIMETIME_EFFECTS = PrimetimeEffects()


# ==============================================================================
# GAME SLOT ENCODING
# ==============================================================================
# Ordinal encoding for game time slots
# Higher values = more primetime exposure

GAME_SLOT_ENCODING: Dict[str, int] = {
    "UNKNOWN": -1,
    "OTHER": 0,
    "EARLY_SUNDAY": 1,  # 1pm ET
    "LATE_AFTERNOON": 2,  # 4pm ET
    "THURSDAY_NIGHT": 3,  # TNF
    "SUNDAY_NIGHT": 4,  # SNF
    "MONDAY_NIGHT": 5,  # MNF
}


# ==============================================================================
# EXPECTED POINTS BY FIELD POSITION
# ==============================================================================
# Source: nflfastR expected points model
# Bins by 10-yard increments from own goal line


EP_BY_FIELD_POSITION: Dict[int, float] = {
    # Own territory (yards from own end zone)
    1: -1.50,  # Own 1-yard line
    5: -1.30,  # Own 5-yard line
    10: -1.00,  # Own 10-yard line
    15: -0.70,  # Own 15-yard line
    20: -0.40,  # Own 20-yard line
    25: -0.10,  # Own 25-yard line
    30: 0.20,  # Own 30-yard line
    35: 0.50,  # Own 35-yard line
    40: 0.80,  # Own 40-yard line
    45: 1.10,  # Midfield
    50: 1.40,  # Opponent 50-yard line
    # Opponent territory
    55: 1.80,  # Opponent 45-yard line
    60: 2.20,  # Opponent 40-yard line
    65: 2.70,  # Opponent 35-yard line
    70: 3.20,  # Opponent 30-yard line
    75: 3.80,  # Opponent 25-yard line
    80: 4.40,  # Opponent 20-yard line (FG range)
    85: 5.00,  # Opponent 15-yard line
    90: 5.50,  # Opponent 10-yard line (Red zone)
    95: 6.00,  # Opponent 5-yard line
    99: 6.30,  # Goal line
}


# ==============================================================================
# WEATHER THRESHOLDS
# ==============================================================================


@dataclass(frozen=True)
class WeatherThresholds:
    """Weather condition thresholds affecting gameplay."""

    # Wind speed (mph) considered significant for passing
    HIGH_WIND_MPH: int = 15

    # Temperature (F) considered cold weather game
    COLD_WEATHER_F: int = 32

    # Temperature (F) considered hot weather game
    HOT_WEATHER_F: int = 85

    # Precipitation (inches) considered wet conditions
    PRECIPITATION_THRESHOLD: float = 0.1


WEATHER_THRESHOLDS = WeatherThresholds()


# ==============================================================================
# VALIDATION STATUS TRACKER
# ==============================================================================
# Track which constants have been empirically validated

VALIDATION_STATUS: Dict[str, dict] = {
    "INJURY_WEIGHTS": {
        "validated": True,
        "validation_script": "analysis/validation/validate_injury_weights.py",
        "last_validated": "2025-12-08",
        "confidence": "empirical",
        "n_games": 1401,
        "r_squared": 0.0337,
        "notes": "QB weight increased 5.0→7.0, O-line reduced 0.5→0.1",
    },
    "TURNOVER_PARAMS": {
        "validated": True,
        "validation_script": None,
        "last_validated": "2024-11-15",
        "confidence": "empirical",
        "source": "Football Outsiders fumble recovery analysis",
    },
    "PRIMETIME_EFFECTS": {
        "validated": False,
        "validation_script": "analysis/validation/validate_primetime_effects.py",
        "last_validated": None,
        "confidence": "expert_opinion",
    },
    "LEAGUE_AVERAGES": {
        "validated": True,
        "validation_script": None,
        "last_validated": "2024-11-01",
        "confidence": "empirical",
        "source": "nflfastR 2018-2024 aggregates",
    },
}
