#!/usr/bin/env python3
"""
Dashboard Configuration Constants

This module centralizes all magic numbers and configuration values used throughout
the dashboard. By extracting these constants, we improve maintainability, consistency,
and make it easier to tune dashboard behavior.

Categories:
  - Health Checks: Service connectivity timeouts
  - Prediction Thresholds: Edge detection for betting recommendations
  - Win Probability: Mathematical models for probability estimation
  - Score Estimation: Heuristics for predicting team scores from QB yards
  - Freshness Tracking: Time-based staleness detection
  - Cache Configuration: Redis TTL values
  - Display Configuration: Chart heights, limits, bins
  - Colors: Bootstrap-compatible color palette
  - Database: Connection and query defaults
  - Time: Time zone and period calculations

Author: Claude Code
Date: November 22, 2025
"""

# ============================================================================
# Health Checks
# ============================================================================

# Socket timeout for service connectivity checks (seconds)
SOCKET_TIMEOUT_SECONDS = 5

# Number of retries for failed health checks
SERVICE_CHECK_RETRY_COUNT = 2

# ============================================================================
# Prediction Thresholds
# ============================================================================

# Spread edge detection thresholds (points)
STRONG_SPREAD_EDGE_THRESHOLD = 2.0  # Confident recommendation (green)
MODERATE_SPREAD_EDGE_THRESHOLD = 1.0  # Moderate recommendation (yellow)

# Total edge detection thresholds (points)
STRONG_TOTAL_EDGE_THRESHOLD = 3.0  # Confident recommendation (green)
MODERATE_TOTAL_EDGE_THRESHOLD = 2.0  # Moderate recommendation (yellow)

# ============================================================================
# Win Probability Calculations
# ============================================================================

# Logistic regression coefficient for spread → win probability
# P(home win) = 1 / (1 + exp(-spread / SPREAD_TO_PROB_LOGISTIC_FACTOR))
# Calibrated on historical NFL data
SPREAD_TO_PROB_LOGISTIC_FACTOR = 7

# Historical correlation: each 50 yards QB difference ≈ 10% win probability difference
# P(home win) = 0.5 + (yards_diff / YARDS_PER_10PCT_WIN_PROB)
YARDS_PER_10PCT_WIN_PROB = 500.0

# Clamp win probability to reasonable bounds (avoid overconfident predictions)
MIN_WIN_PROB = 0.1  # At least 10% chance of losing
MAX_WIN_PROB = 0.9  # At most 90% confident

# ============================================================================
# Score Estimation
# ============================================================================

# Historical correlation: team score ≈ intercept + (QB_yards / slope)
# Based on 5+ seasons of NFL data analysis
QB_YARDS_TO_SCORE_INTERCEPT = 10.0  # Base team score (field goals, defense, etc.)
QB_YARDS_TO_SCORE_SLOPE = 20.0  # Yards per point (20 yards ≈ 1 team point)

# ============================================================================
# Freshness Tracking
# ============================================================================

# Time thresholds for prediction freshness (hours)
PREDICTION_FRESH_HOURS = 24  # Prediction is "fresh" if < 1 day old
PREDICTION_AGING_HOURS = 72  # Prediction is "aging" if > 3 days old
PREDICTION_STALE_HOURS = 168  # Prediction is "stale" if > 1 week old

# ============================================================================
# Cache Configuration
# ============================================================================

# Redis cache TTL (time-to-live) in seconds
# Used by @cache_data decorator for different data freshness requirements

# Live data: quotes, game status (5-20ms Redis hit vs 100-500ms query)
REDIS_CACHE_TTL_LIVE = 30  # 30 seconds

# Dynamic data: player props, current predictions (5 minutes)
REDIS_CACHE_TTL_SHORT = 300  # 5 minutes

# Stable data: historical stats, player baselines, models (1 hour)
REDIS_CACHE_TTL_MEDIUM = 1800  # 30 minutes

# Very stable data: schema info, historical stats (1+ hour)
REDIS_CACHE_TTL_LONG = 3600  # 1 hour

# ============================================================================
# Display Configuration
# ============================================================================

# Chart heights (pixels) - Plotly/Streamlit display
CHART_HEIGHT_DEFAULT = 400  # Standard chart height
CHART_HEIGHT_COMPACT = 300  # Compact display (sidebars, small sections)
CHART_HEIGHT_MINI = 200  # Very compact (summary boxes)

# Display limits (prevent overwhelming UI)
MAX_GAMES_DISPLAY = 50  # Max games shown in game list
MAX_PREDICTIONS_DISPLAY = 1000  # Max predictions in table
MAX_TRADES_DISPLAY = 20  # Max recent trades shown

# Histogram/histogram configuration
HISTOGRAM_BINS_DEFAULT = 30  # Default number of bins
HISTOGRAM_BINS_DENSE = 40  # Dense histogram (more detail)
HISTOGRAM_BINS_SPARSE = 20  # Sparse histogram (less detail)

# Probability calibration bins (for calibration plots)
CALIBRATION_BINS = 10  # Split probabilities into 10 deciles (0.0-0.1, 0.1-0.2, etc.)

# ============================================================================
# Colors - Bootstrap Compatible
# ============================================================================

# Standard Bootstrap color palette for consistent theming
COLOR_SUCCESS = "#28a745"  # Green (profitable, edge detected, healthy)
COLOR_WARNING = "#ffc107"  # Yellow (marginal, caution)
COLOR_NEUTRAL = "#6c757d"  # Gray (neutral, no edge, no status)
COLOR_DANGER = "#dc3545"  # Red (loss, unhealthy)
COLOR_PRIMARY = "#1f77b4"  # Blue (primary accent, used in charts)
COLOR_SECONDARY = "#ff7f0e"  # Orange (secondary accent)
COLOR_SUCCESS_BG = "#d4edda"  # Light green background
COLOR_WARNING_BG = "#fff3cd"  # Light yellow background
COLOR_DANGER_BG = "#f8d7da"  # Light red background
COLOR_NEUTRAL_BG = "#f8f9fa"  # Light gray background

# CSS color constants (for styling)
COLOR_TEXT_DARK = "#212529"  # Dark text on light backgrounds
COLOR_TEXT_LIGHT = "#ffffff"  # Light text on dark backgrounds
COLOR_BORDER_LIGHT = "#dee2e6"  # Light border color
COLOR_BORDER_DARK = "#495057"  # Dark border color

# ============================================================================
# Database Configuration
# ============================================================================

# Default API timeout (seconds)
API_TIMEOUT_DEFAULT = 10  # Standard API calls
API_TIMEOUT_LONG = 15  # Long-running API calls (game predictions)

# Database query limits
QUERY_LIMIT_DEFAULT = 1000  # Default SQL query limit
QUERY_LIMIT_RECENT = 100  # Fetch recent records (smaller limit)
QUERY_LIMIT_STATS = 500  # Statistical summaries

# Season filters (for stability)
MIN_SEASON_FOR_STATS = 2022  # Only use 2022+ data for performance calculations
MIN_SEASON_FOR_FULL_HISTORY = 2023  # 2023+ for comprehensive history

# ============================================================================
# Time Configuration
# ============================================================================

# NFL game schedule
MIN_WEEK_NUMBER = 1
MAX_WEEK_NUMBER = 18
GAMES_PER_TEAM_PER_SEASON = 17  # 2023 onward (previously 16)

# Game time slots (ET)
GAME_TIME_EARLY_MORNING_HOURS = 9  # Overseas games (~9:30 AM ET)
GAME_TIME_EARLY_SLOT_HOURS = 13  # Early games (1:00 PM ET)
GAME_TIME_AFTERNOON_SLOT_HOURS = 16  # Afternoon games (4:00 PM ET)
GAME_TIME_NIGHT_SLOT_HOURS = 20  # Sunday Night Football (8:00 PM ET)

# UTC offset for ET
TIMEZONE_OFFSET_ET_HOURS = 5

# Calibration confidence intervals
CALIBRATION_TARGET_COVERAGE = 0.90  # Target 90% coverage for intervals
CALIBRATION_UPPER_PERCENTILE = 0.95  # Upper bound (95th percentile)
CALIBRATION_LOWER_PERCENTILE = 0.05  # Lower bound (5th percentile)

# ============================================================================
# MD5 Hash Configuration
# ============================================================================

# MD5 hash truncation for cache keys (reduces collision risk)
MD5_HASH_LENGTH = 16  # Use first 16 chars of MD5 hash

# ============================================================================
# Breakeven Thresholds
# ============================================================================

# Sports betting profitability thresholds
BREAKEVEN_WIN_RATE = 0.524  # 52.4% win rate needed for breakeven at -110 odds
TARGET_WIN_RATE = 0.60  # Target: 60% win rate
TARGET_SHARPE_RATIO = 2.0  # Target Sharpe ratio for risk-adjusted returns

# ============================================================================
# Model Performance Targets
# ============================================================================

# Historical model targets (documented in CLAUDE.md)
TARGET_ROI_MIN = 0.05  # Minimum 5% ROI
TARGET_ROI_OPTIMAL = 0.07  # Target 7% ROI
TARGET_WIN_RATE_MIN = 0.59  # Minimum 59% win rate
TARGET_CALIBRATION = 0.90  # Target 90% calibration coverage

# Historical model performance (state-space v2)
STATE_SPACE_MAE = 47.77  # Mean absolute error (yards)
STATE_SPACE_COVERAGE = 0.865  # Calibration coverage
STATE_SPACE_P_VALUE = 0.0069  # Statistical significance

# ============================================================================
# Feature Engineering
# ============================================================================

# Training window configuration (documented in CLAUDE.md)
CANONICAL_TRAINING_START_YEAR = 2018  # Full weight from 2018 onward
CANONICAL_TRAINING_DECAY_START_YEAR = 2015  # Exponential decay for 2015-2017
CANONICAL_TRAINING_DECAY_HALFLIFE = 4  # Decay halflife (years)

# Decay weights for historical years
DECAY_WEIGHT_2017 = 0.297  # w(2017; 2025, 4) = 0.5^((2025-2017)/4)
DECAY_WEIGHT_2016 = 0.250  # w(2016; 2025, 4) = 0.5^((2025-2016)/4)
DECAY_WEIGHT_2015 = 0.210  # w(2015; 2025, 4) = 0.5^((2025-2015)/4)

# ============================================================================
# Test Set Configuration
# ============================================================================

# Validation test set (documented in CLAUDE.md)
TEST_SET_SEASON = 2025
TEST_SET_MIN_WEEK = 1
TEST_SET_MAX_WEEK = 6
TEST_SET_GAME_COUNT = 148  # Expected games in test set
TEST_SET_QB_COUNT = 37  # Expected unique QBs in test set
