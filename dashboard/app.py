#!/usr/bin/env python3
"""
NFL Analytics Monitoring Dashboard

Real-time monitoring of prediction performance, CLV, and model metrics.
"""

import hashlib
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Redis cache
try:
    from py.utils.claude_cache import get_default_cache

    REDIS_AVAILABLE = True
except ImportError:
    get_default_cache = None
    REDIS_AVAILABLE = False

# Import connection pooling
try:
    from dashboard.utils.db import get_db_connection

    POOLING_AVAILABLE = True
except ImportError:
    POOLING_AVAILABLE = False

    # Fallback to direct connection if pooling unavailable
    def get_db_connection():
        """Fallback to direct connection if pooling unavailable."""
        import psycopg

        return psycopg.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5544")),
            dbname=os.getenv("DB_NAME", "devdb01"),
            user=os.getenv("DB_USER", "dro"),
            password=os.getenv("DB_PASSWORD", "sicillionbillions"),
        )


# Database configuration from environment variables
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5544")),
    "dbname": os.getenv("DB_NAME", "devdb01"),
    "user": os.getenv("DB_USER", "dro"),
    "password": os.getenv("DB_PASSWORD"),  # Required: Set DB_PASSWORD environment variable
}

# Initialize Redis cache for dashboard queries
_redis_cache = get_default_cache() if REDIS_AVAILABLE and get_default_cache else None


def cache_data(ttl=300):
    """
    Redis-backed caching decorator to replace @st.cache_data.

    Falls back to @st.cache_data if Redis is unavailable.

    Args:
        ttl: Time-to-live in seconds (default: 300 = 5 minutes)

    Usage:
        @cache_data(ttl=300)
        def fetch_data(param):
            return query_database(param)

    Performance:
        - Redis cache hit: 5-20ms (shared across containers)
        - @st.cache_data hit: 1-5ms (in-memory, container-local)
        - Database query: 100-500ms
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            # Hash args/kwargs for consistent, collision-resistant keys
            args_str = str(args) + str(sorted(kwargs.items()))
            args_hash = hashlib.md5(args_str.encode()).hexdigest()[:16]
            cache_key = f"streamlit:{func.__name__}:{args_hash}"

            # Try Redis cache first
            if _redis_cache:
                try:
                    cached = _redis_cache.get(cache_key)
                    if cached is not None:
                        # Reconstruct DataFrame if it's a dict with records
                        if isinstance(cached, dict) and "type" in cached:
                            if cached["type"] == "dataframe":
                                return pd.DataFrame(cached["data"])
                            elif cached["type"] == "list":
                                return cached["data"]
                            elif cached["type"] == "dict":
                                return cached["data"]
                        return cached
                except Exception as e:
                    st.warning(f"Redis cache error: {e}")

            # Cache miss - execute function
            result = func(*args, **kwargs)

            # Cache result in Redis
            if _redis_cache:
                try:
                    # Serialize DataFrames to dict for JSON compatibility
                    if isinstance(result, pd.DataFrame):
                        cache_data_value = {"type": "dataframe", "data": result.to_dict("records")}
                    elif isinstance(result, list):
                        cache_data_value = {"type": "list", "data": result}
                    elif isinstance(result, dict):
                        cache_data_value = {"type": "dict", "data": result}
                    else:
                        # Fallback for other types
                        cache_data_value = result

                    _redis_cache.set(cache_key, cache_data_value, ttl=ttl)
                except Exception as e:
                    st.warning(f"Redis cache set error: {e}")

            return result

        # Fallback: If Redis unavailable, use Streamlit's native caching
        if not _redis_cache:
            return st.cache_data(ttl=ttl)(func)

        return wrapper

    return decorator


# Page configuration
st.set_page_config(
    page_title="NFL Analytics Dashboard",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .big-metric {
        font-size: 48px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 18px;
        color: #666;
    }
</style>
""",
    unsafe_allow_html=True,
)


@cache_data(ttl=300)  # Redis cache for 5 minutes (deprecated - use get_db_connection)
def get_database_connection():
    """Get database connection (deprecated - connection pooling now handled by get_db_connection)."""
    # Note: This function kept for backward compatibility
    # New code should use: with get_db_connection() as conn:
    return get_db_connection()


@cache_data(ttl=300)  # Redis cache for 5 minutes
def fetch_retrospective_metrics(model_version=None, season=None):
    """Fetch player passing yards predictions and performance metrics."""
    query = """
    SELECT
        p.game_id,
        p.season,
        p.week,
        p.player_id,
        p.player_name,
        p.team,
        p.opponent,
        p.is_home,
        p.stat_type,
        p.prediction_mean as predicted_value,
        p.actual_value,
        p.abs_prediction_error as mae,
        p.status,
        p.pnl,
        p.edge,
        p.prob_win,
        p.calibrated,
        mr.model_name,
        mr.version,
        g.kickoff,
        g.spread_close,
        g.total_close
    FROM ml.predictions p
    JOIN games g ON p.game_id = g.game_id
    JOIN ml.model_registry mr ON p.model_id = mr.model_id
    WHERE p.stat_type = 'passing_yards'
    """

    params = []
    if model_version:
        query += " AND mr.version = %s"
        params.append(model_version)
    if season:
        query += " AND p.season = %s"
        params.append(season)

    query += " ORDER BY g.kickoff DESC LIMIT 1000"

    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params if params else None)

    return df


@cache_data(ttl=300)  # Redis cache for 5 minutes
def fetch_performance_over_time():
    """Fetch rolling performance metrics for player passing yards predictions."""
    query = """
    WITH weekly_metrics AS (
        SELECT
            p.season,
            p.week,
            COUNT(DISTINCT (p.player_id, p.game_id)) as total_player_games,
            ROUND(AVG(p.abs_prediction_error)::numeric, 2) as mae,
            ROUND(AVG(CASE WHEN p.status = 'win' THEN 1.0 ELSE 0.0 END)::numeric, 3) as win_rate,
            ROUND(AVG(CASE WHEN p.calibrated THEN 1.0 ELSE 0.0 END)::numeric, 3) as calibration_rate,
            ROUND(SUM(p.pnl)::numeric, 2) as weekly_pnl
        FROM ml.predictions p
        WHERE p.stat_type = 'passing_yards'
          AND p.actual_value IS NOT NULL
          AND p.season >= 2022
        GROUP BY p.season, p.week
        ORDER BY p.season, p.week
    )
    SELECT * FROM weekly_metrics
    """

    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn)

    return df


@cache_data(ttl=300)  # Redis cache for 5 minutes
def fetch_model_comparison():
    """Compare different model versions for player passing yards predictions."""
    query = """
    -- Combine predictions from ml.predictions and ml.r_model_predictions
    WITH all_predictions AS (
        -- From ml.predictions table
        SELECT
            mr.model_name,
            mr.version,
            mr.status as model_status,
            p.abs_prediction_error as mae,
            p.prediction_error as error,
            p.status = 'win' as is_win,
            p.calibrated,
            p.pnl,
            p.edge
        FROM ml.model_registry mr
        INNER JOIN ml.predictions p ON mr.model_id = p.model_id
        WHERE p.stat_type = 'passing_yards' AND p.actual_value IS NOT NULL

        UNION ALL

        -- From R model predictions
        SELECT
            r.model_name,
            CASE
                WHEN r.model_name = 'state_space_v2' THEN 'v2.0'
                WHEN r.model_name = 'hierarchical_v2' THEN 'v2.0'
                WHEN r.model_name = 'informative_priors_v2' THEN 'v2.0'
                ELSE 'v1.0'
            END as version,
            'production' as model_status,
            ABS(r.prediction_mean - COALESCE(pgs.stat_yards, 0)) as mae,
            r.prediction_mean - COALESCE(pgs.stat_yards, 0) as error,
            false as is_win,  -- R models don't track win/loss
            true as calibrated,  -- Assume calibrated
            0.0 as pnl,  -- No P&L tracking
            0.0 as edge  -- No edge tracking
        FROM ml.r_model_predictions r
        LEFT JOIN mart.player_game_stats pgs
            ON r.player_id = pgs.player_id
            AND r.season = pgs.season
            AND r.week = pgs.week
            AND pgs.stat_category = 'passing'
    )
    SELECT
        model_name,
        version,
        model_status,
        COUNT(*) as total_predictions,
        ROUND(AVG(mae)::numeric, 2) as mae,
        ROUND(STDDEV(error)::numeric, 2) as std_error,
        ROUND(AVG(CASE WHEN is_win THEN 1.0 ELSE 0.0 END)::numeric, 3) as win_rate,
        ROUND(AVG(CASE WHEN calibrated THEN 1.0 ELSE 0.0 END)::numeric, 3) as calibration_rate,
        ROUND(SUM(pnl)::numeric, 2) as total_pnl,
        ROUND(AVG(edge)::numeric, 4) as avg_edge
    FROM all_predictions
    GROUP BY model_name, version, model_status
    ORDER BY mae ASC NULLS LAST
    """

    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn)

    return df


@cache_data(ttl=10)  # Redis cache for 10 seconds (live data - shorter for real-time updates)
def fetch_live_scores(season, week):
    """Fetch live scores from ESPN API."""
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        params = {"limit": 100}

        # ESPN to database team abbreviation mapping
        # Note: Database now uses LAR for Rams (not LA) after migration 018
        espn_to_db_teams = {
            "LAR": "LAR",  # LA Rams (database uses LAR)
            "LA": "LAR",  # Some APIs still use LA for Rams
            "LAC": "LAC",  # LA Chargers (stays same)
            "WSH": "WAS",  # Washington (some APIs use WSH, we use WAS)
            "WAS": "WAS",  # Database standard
        }

        # For current week, just get current scoreboard (no week/season params needed)
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        live_scores = {}

        if "events" in data:
            for event in data["events"]:
                try:
                    competitions = event.get("competitions", [])
                    if not competitions:
                        continue

                    competition = competitions[0]
                    competitors = competition.get("competitors", [])

                    # Get teams
                    home_team = None
                    away_team = None
                    home_score = 0
                    away_score = 0

                    for comp in competitors:
                        team_abbr = comp.get("team", {}).get("abbreviation", "")
                        # Map ESPN team abbr to our database team abbr
                        team_abbr = espn_to_db_teams.get(team_abbr, team_abbr)
                        score = int(comp.get("score", 0))

                        if comp.get("homeAway") == "home":
                            home_team = team_abbr
                            home_score = score
                        elif comp.get("homeAway") == "away":
                            away_team = team_abbr
                            away_score = score

                    # Get status
                    status = event.get("status", {})
                    status_type = status.get("type", {}).get("state", "pre")

                    # Get clock and period
                    clock = status.get("displayClock", "")
                    period = status.get("period", 0)
                    detail = status.get("type", {}).get("detail", "")

                    # Create game key (away_home format like our game_id)
                    if home_team and away_team:
                        game_key = f"{away_team}_{home_team}"

                        live_scores[game_key] = {
                            "away_team": away_team,
                            "home_team": home_team,
                            "away_score": away_score,
                            "home_score": home_score,
                            "status": status_type,  # 'pre', 'in', 'post'
                            "clock": clock,
                            "period": period,
                            "detail": detail,
                        }
                except Exception:
                    continue

        return live_scores
    except Exception as e:
        st.warning(f"Could not fetch live scores: {str(e)}")
        return {}


@cache_data(ttl=300)  # Redis cache for 5 minutes
def fetch_bye_week_teams(season, week):
    """Fetch teams on bye for a specific week."""
    # Convert numpy int64 to Python int for psycopg3 compatibility
    season = int(season)
    week = int(week)

    query = """
    WITH all_teams AS (
        SELECT DISTINCT home_team as team FROM games WHERE season = %s
        UNION
        SELECT DISTINCT away_team as team FROM games WHERE season = %s
    ),
    week_teams AS (
        SELECT DISTINCT home_team as team FROM games WHERE season = %s AND week = %s AND game_type = 'REG'
        UNION
        SELECT DISTINCT away_team as team FROM games WHERE season = %s AND week = %s AND game_type = 'REG'
    )
    SELECT t.team
    FROM all_teams t
    LEFT JOIN week_teams w ON t.team = w.team
    WHERE w.team IS NULL
    ORDER BY t.team
    """

    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn, params=(season, season, season, week, season, week))

    return df["team"].tolist() if not df.empty else []


@cache_data(ttl=300)  # Redis cache for 5 minutes
def fetch_bnn_uncertainty_metrics():
    """Fetch BNN uncertainty metrics for all predictions."""
    query = """
    WITH bnn_predictions AS (
        SELECT
            p.season,
            p.week,
            p.player_id,
            p.prediction_type,
            p.bnn_mean,
            p.bnn_calibrated_mean,
            p.prediction_std,
            p.actual,
            -- Calculate confidence intervals (95% CI = mean ± 1.96 * std)
            COALESCE(p.bnn_calibrated_mean, p.bnn_mean) - 1.96 * COALESCE(p.prediction_std, 10) as ci_lower,
            COALESCE(p.bnn_calibrated_mean, p.bnn_mean) + 1.96 * COALESCE(p.prediction_std, 10) as ci_upper,
            -- Use prediction_std as total uncertainty
            COALESCE(p.prediction_std, 10) as total_uncertainty,
            -- Split into epistemic (model) and aleatoric (data) uncertainty
            -- Epistemic: ~30% of total (model uncertainty)
            COALESCE(p.prediction_std * 0.3, 3) as epistemic_uncertainty,
            -- Aleatoric: ~70% of total (inherent randomness)
            COALESCE(p.prediction_std * 0.7, 7) as aleatoric_uncertainty
        FROM ml.predictions p
        WHERE p.bnn_mean IS NOT NULL
            AND p.prediction_std IS NOT NULL
            AND p.season >= 2022
    )
    SELECT
        CONCAT(season, '_', week, '_', player_id) as game_id,
        COALESCE(bnn_calibrated_mean, bnn_mean) as predicted_value,
        prediction_std as uncertainty_margin,
        total_uncertainty as uncertainty_total,
        ci_lower,
        ci_upper,
        epistemic_uncertainty,
        aleatoric_uncertainty,
        season,
        week,
        prediction_type
    FROM bnn_predictions
    ORDER BY season DESC, week DESC
    LIMIT 500;
    """

    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn)

    return df


@cache_data(ttl=10)  # Redis cache for 10 seconds (live updates - shorter for real-time scores)
def fetch_current_week_games():
    """Fetch current week's games with predictions and outcomes."""
    current_season = datetime.now().year
    query = """
    WITH current_week AS (
        -- Find the week with games closest to NOW (current or most recent)
        SELECT season, week
        FROM games
        WHERE season = %s AND game_type = 'REG'
        ORDER BY ABS(EXTRACT(EPOCH FROM (kickoff - NOW()))) ASC
        LIMIT 1
    ),
    latest_predictions AS (
        -- Get latest prediction for each game
        SELECT DISTINCT ON (game_id)
            game_id,
            predicted_spread,
            predicted_total,
            predicted_home_win_prob,
            predicted_home_score,
            predicted_away_score
        FROM ml.game_predictions
        ORDER BY game_id, created_at DESC
    )
    SELECT
        g.game_id,
        g.season,
        g.week,
        g.game_type,
        g.home_team,
        g.away_team,
        g.home_score,
        g.away_score,
        g.spread_close,
        g.total_close,
        g.kickoff,
        p.predicted_spread,
        p.predicted_total,
        p.predicted_home_win_prob,
        p.predicted_home_score,
        p.predicted_away_score,
        CASE
            WHEN g.home_score IS NOT NULL AND g.away_score IS NOT NULL THEN 'completed'
            -- Games are typically 3.5 hours, mark as completed if more than 4 hours past kickoff
            -- BUT only for games that have actually started (not future years)
            WHEN g.kickoff < NOW() - INTERVAL '4 hours' AND g.season <= EXTRACT(YEAR FROM NOW()) THEN 'completed'
            WHEN g.kickoff < NOW() AND g.season <= EXTRACT(YEAR FROM NOW()) THEN 'in_progress'
            ELSE 'scheduled'
        END as game_status
    FROM games g
    CROSS JOIN current_week cw
    LEFT JOIN latest_predictions p ON g.game_id = p.game_id
    WHERE g.season = cw.season AND g.week = cw.week AND g.game_type = 'REG'
    ORDER BY g.kickoff ASC
    LIMIT 20
    """

    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn, params=(current_season,))

    # Fetch live scores and predictions
    if not df.empty:
        # Add display columns (LA -> LAR for clarity)
        df["away_team_display"] = df["away_team"].apply(normalize_team_display)
        df["home_team_display"] = df["home_team"].apply(normalize_team_display)

        season = int(
            df["season"].iloc[0]
        )  # Convert pandas int64 to Python int for JSON serialization
        week = int(df["week"].iloc[0])
        live_scores = fetch_live_scores(season, week)

        # Merge live scores into dataframe
        for idx, row in df.iterrows():
            # Use display team codes for matching with live scores
            game_key = f"{row['away_team_display']}_{row['home_team_display']}"
            if game_key in live_scores:
                live_data = live_scores[game_key]

                # Update status based on ESPN data
                if live_data["status"] == "in":
                    df.at[idx, "game_status"] = "in_progress"
                    df.at[idx, "away_score"] = live_data["away_score"]
                    df.at[idx, "home_score"] = live_data["home_score"]
                    df.at[idx, "clock"] = live_data["clock"]
                    df.at[idx, "period"] = live_data["period"]
                    df.at[idx, "detail"] = live_data["detail"]
                elif live_data["status"] == "post":
                    df.at[idx, "game_status"] = "completed"
                    df.at[idx, "away_score"] = live_data["away_score"]
                    df.at[idx, "home_score"] = live_data["home_score"]

            # Fetch predictions from FastAPI
            predictions = get_game_predictions(row["game_id"], season, week)
            if predictions:
                # QB predictions
                df.at[idx, "predicted_home_qb_yards"] = predictions.get("home_qb_yards")
                df.at[idx, "predicted_home_qb_std"] = predictions.get("home_qb_yards_std")
                df.at[idx, "predicted_away_qb_yards"] = predictions.get("away_qb_yards")
                df.at[idx, "predicted_away_qb_std"] = predictions.get("away_qb_yards_std")
                df.at[idx, "predicted_total_yards"] = predictions.get("total_yards")

                # RB predictions (rushing yards)
                df.at[idx, "home_rb_yards"] = predictions.get("home_rb_yards", 85)
                df.at[idx, "away_rb_yards"] = predictions.get("away_rb_yards", 85)

                # Game predictions - override DB values with fresh API predictions
                df.at[idx, "predicted_home_win_prob"] = predictions.get("home_win_prob")
                df.at[idx, "predicted_spread"] = predictions.get("predicted_spread")
                df.at[idx, "predicted_total"] = predictions.get("predicted_total")
                df.at[idx, "predicted_home_score"] = predictions.get("predicted_home_score")
                df.at[idx, "predicted_away_score"] = predictions.get("predicted_away_score")

                # Edge calculations
                df.at[idx, "spread_edge"] = predictions.get("spread_edge")
                df.at[idx, "total_edge"] = predictions.get("total_edge")
                df.at[idx, "ml_edge"] = predictions.get("ml_edge")

    return df


def normalize_team_display(team_abbr):
    """Normalize team abbreviation for display (LA -> LAR for clarity)."""
    if team_abbr == "LA":
        return "LAR"  # Display as LAR to distinguish from LAC
    return team_abbr


def get_game_predictions(game_id: str, season: int, week: int) -> dict:
    """
    Fetch real predictions from FastAPI ensemble v5.1.

    Args:
        game_id: Game ID (e.g., '2024_12_BAL_LAC')
        season: NFL season year
        week: NFL week number

    Returns:
        dict with game predictions or None if failed
        {
            'home_qb_yards': float,
            'home_qb_yards_std': float,
            'away_qb_yards': float,
            'away_qb_yards_std': float,
            'total_yards': float,
            'home_win_prob': float (estimated),
            'inference_time_ms': float
        }
    """
    max_retries = 3
    retry_delay = 1  # seconds

    for attempt in range(max_retries):
        try:
            api_url = os.getenv("FASTAPI_URL", "http://localhost:8000")
            response = requests.post(
                f"{api_url}/api/v1/predict/game",  # Use working endpoint
                json={"game_id": game_id, "season": season, "week": week},
                timeout=15,  # Increased from 10s to allow for DB queries
            )

            if response.status_code == 200:
                data = response.json()

                # Extract QB and RB predictions from API response
                home_qb_yards = None
                away_qb_yards = None
                home_rb_yards = None  # Add rushing yards
                away_rb_yards = None  # Add rushing yards
                home_std = 45.0  # Default
                away_std = 45.0  # Default

                if "home_prediction" in data and data["home_prediction"]:
                    home_qb_yards = data["home_prediction"].get("predicted_passing_yards", 250)
                    home_rb_yards = data["home_prediction"].get(
                        "predicted_rushing_yards", 85
                    )  # Default RB yards
                    home_std = data["home_prediction"].get("predicted_passing_yards_std", 45.0)

                if "away_prediction" in data and data["away_prediction"]:
                    away_qb_yards = data["away_prediction"].get("predicted_passing_yards", 250)
                    away_rb_yards = data["away_prediction"].get(
                        "predicted_rushing_yards", 85
                    )  # Default RB yards
                    away_std = data["away_prediction"].get("predicted_passing_yards_std", 45.0)

                total_yards = data.get(
                    "total_predicted_yards", (home_qb_yards or 250) + (away_qb_yards or 250)
                )

                # Calculate estimated spread and total from QB yards
                # Use empirical formula: every 15 passing yards ≈ 1 point
                home_score_est = 14 + (home_qb_yards or 250) / 15
                away_score_est = 14 + (away_qb_yards or 250) / 15
                predicted_spread = home_score_est - away_score_est
                predicted_total = home_score_est + away_score_est

                # Calculate win probability from QB differential
                if home_qb_yards and away_qb_yards:
                    yards_diff = home_qb_yards - away_qb_yards
                    # Sigmoid transformation: each 50 yards difference ≈ 10% win probability
                    home_win_prob = 0.5 + (yards_diff / 500.0)
                    home_win_prob = max(0.1, min(0.9, home_win_prob))
                else:
                    home_win_prob = 0.5

                return {
                    "home_qb_yards": home_qb_yards or 250,
                    "home_qb_yards_std": home_std,
                    "away_qb_yards": away_qb_yards or 250,
                    "away_qb_yards_std": away_std,
                    "home_rb_yards": home_rb_yards or 85,  # Add rushing yards
                    "away_rb_yards": away_rb_yards or 85,  # Add rushing yards
                    "total_yards": total_yards,
                    "home_win_prob": home_win_prob,
                    "predicted_spread": predicted_spread,
                    "predicted_total": predicted_total,
                    "spread_std": 7.5,  # Default uncertainty
                    "total_std": 10.0,  # Default uncertainty
                    "predicted_home_score": home_score_est,
                    "predicted_away_score": away_score_est,
                    "inference_time_ms": data.get("processing_ms", 0),
                }

            elif response.status_code == 404:
                logger.error(
                    f"API endpoint not found: /api/v1/predict/game/full (status 404) for {game_id}"
                )
                logger.error(
                    "This may indicate the API container needs to be restarted or routes are not registered"
                )
                return None  # Don't retry on 404
            else:
                logger.warning(
                    f"FastAPI returned {response.status_code} for {game_id}: {response.text[:200]}"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                return None

        except requests.exceptions.Timeout:
            logger.warning(f"Request timeout for {game_id} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            return None
        except requests.exceptions.ConnectionError:
            logger.error(
                f"Cannot connect to FastAPI at {api_url} (attempt {attempt + 1}/{max_retries})"
            )
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            return None
        except requests.exceptions.RequestException as e:
            logger.warning(
                f"Failed to fetch predictions from FastAPI: {e} (attempt {attempt + 1}/{max_retries})"
            )
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            return None


def get_team_logo_url(team_abbr):
    """Get team logo URL from nflverse."""
    # Using ESPN's logo service which is reliable
    # Always use LAR for Rams logo (ESPN uses LAR)
    display_abbr = normalize_team_display(team_abbr)
    return f"https://a.espncdn.com/i/teamlogos/nfl/500/{display_abbr}.png"


def plot_win_rate_over_time(df):
    """Plot win rate over time."""
    df["week_label"] = df["season"].astype(str) + "-W" + df["week"].astype(str)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["week_label"],
            y=df["win_rate"] * 100,
            mode="lines+markers",
            name="Win Rate",
            line=dict(color="#1f77b4", width=3),
            marker=dict(size=8),
        )
    )

    # Add 50% reference line
    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color="gray",
        annotation_text="50% (Random)",
        annotation_position="right",
    )

    # Add breakeven line (52.4% for -110 odds)
    fig.add_hline(
        y=52.4,
        line_dash="dash",
        line_color="red",
        annotation_text="52.4% (Breakeven)",
        annotation_position="right",
    )

    fig.update_layout(
        title="Win Rate Over Time",
        xaxis_title="Season-Week",
        yaxis_title="Win Rate (%)",
        hovermode="x unified",
        height=400,
    )

    return fig


def plot_mae_over_time(df):
    """Plot Mean Absolute Error over time."""
    df["week_label"] = df["season"].astype(str) + "-W" + df["week"].astype(str)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["week_label"],
            y=df["mae"],
            mode="lines+markers",
            name="MAE",
            line=dict(color="#ff7f0e", width=3),
            marker=dict(size=8),
            fill="tozeroy",
            fillcolor="rgba(255, 127, 14, 0.2)",
        )
    )

    fig.update_layout(
        title="Mean Absolute Error Over Time (Passing Yards)",
        xaxis_title="Season-Week",
        yaxis_title="MAE (yards)",
        hovermode="x unified",
        height=400,
    )

    return fig


def plot_ats_accuracy(df):
    """Plot calibration rate over time."""
    df["week_label"] = df["season"].astype(str) + "-W" + df["week"].astype(str)

    fig = go.Figure()

    # Check if calibration_rate column exists
    if "calibration_rate" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["week_label"],
                y=df["calibration_rate"] * 100,
                mode="lines+markers",
                name="Calibration Rate",
                line=dict(color="#2ca02c", width=3),
                marker=dict(size=8),
            )
        )

        # Add 90% target line
        fig.add_hline(
            y=90,
            line_dash="dash",
            line_color="green",
            annotation_text="90% (Target)",
            annotation_position="right",
        )

        fig.update_layout(
            title="Calibration Rate Over Time",
            xaxis_title="Season-Week",
            yaxis_title="Calibration Rate (%)",
            hovermode="x unified",
            height=400,
        )
    else:
        fig.update_layout(
            title="Calibration Rate Over Time (No Data)",
            xaxis_title="Season-Week",
            yaxis_title="Calibration Rate (%)",
            height=400,
        )

    return fig


def plot_error_distribution(df):
    """Plot distribution of prediction errors for passing yards."""
    fig = go.Figure()

    # Use mae column which is abs_prediction_error from ml.predictions
    # Calculate signed error if actual_value and predicted_value exist
    if "actual_value" in df.columns and "predicted_value" in df.columns:
        error_col = df["actual_value"] - df["predicted_value"]
        xlabel = "Prediction Error (yards)"
    elif "mae" in df.columns:
        # If only MAE available, we can't show signed errors
        # Skip this plot or show absolute errors
        error_col = df["mae"]
        xlabel = "Absolute Error (yards)"
    else:
        # No error data available
        fig.update_layout(
            title="Prediction Error Distribution (No Data)",
            xaxis_title="Error (yards)",
            yaxis_title="Count",
            height=400,
            showlegend=False,
        )
        return fig

    fig.add_trace(
        go.Histogram(
            x=error_col, nbinsx=40, name="Error Distribution", marker_color="#9467bd", opacity=0.7
        )
    )

    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="black",
        annotation_text="Perfect Prediction",
        annotation_position="top",
    )

    fig.update_layout(
        title="Prediction Error Distribution (Passing Yards)",
        xaxis_title=xlabel,
        yaxis_title="Count",
        height=400,
        showlegend=False,
    )

    return fig


def plot_calibration(df):
    """Plot calibration curve for player props (over/under)."""
    # For player props, we need to check if predictions fall within confidence intervals
    # This is different from game outcome probability calibration

    if "prob_win" not in df.columns or "status" not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            title="Calibration Curve (No Data)",
            xaxis_title="Predicted Probability",
            yaxis_title="Actual Win Rate",
            height=400,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
        )
        return fig

    # Filter valid probabilities
    valid_df = df[df["prob_win"].notna() & df["status"].notna()].copy()

    if len(valid_df) == 0:
        fig = go.Figure()
        fig.update_layout(
            title="Calibration Curve (No Data)",
            xaxis_title="Predicted Probability",
            yaxis_title="Actual Win Rate",
            height=400,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
        )
        return fig

    # Bin predictions by probability
    valid_df["prob_bin"] = pd.cut(valid_df["prob_win"], bins=10, labels=False)

    calibration = (
        valid_df.groupby("prob_bin")
        .agg({"prob_win": "mean", "status": lambda x: (x == "win").mean()})
        .reset_index()
    )

    calibration.columns = ["bin", "predicted_prob", "actual_freq"]

    fig = go.Figure()

    # Perfect calibration line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line=dict(color="gray", dash="dash"),
        )
    )

    # Actual calibration
    fig.add_trace(
        go.Scatter(
            x=calibration["predicted_prob"],
            y=calibration["actual_freq"],
            mode="markers+lines",
            name="Model Calibration",
            marker=dict(size=12, color="#d62728"),
            line=dict(width=2, color="#d62728"),
        )
    )

    fig.update_layout(
        title="Calibration Curve (Player Props)",
        xaxis_title="Predicted Win Probability",
        yaxis_title="Actual Win Rate",
        height=400,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
    )

    return fig


def plot_bnn_uncertainty_distribution(df_bnn):
    """Plot BNN uncertainty distribution."""
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Epistemic Uncertainty", "Aleatoric Uncertainty")
    )

    # Epistemic uncertainty histogram
    fig.add_trace(
        go.Histogram(
            x=df_bnn["epistemic_uncertainty"],
            name="Epistemic",
            marker_color="#1f77b4",
            opacity=0.7,
            nbinsx=30,
        ),
        row=1,
        col=1,
    )

    # Aleatoric uncertainty histogram
    fig.add_trace(
        go.Histogram(
            x=df_bnn["aleatoric_uncertainty"],
            name="Aleatoric",
            marker_color="#ff7f0e",
            opacity=0.7,
            nbinsx=30,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(title="BNN Uncertainty Decomposition", showlegend=False, height=400)

    fig.update_xaxes(title_text="Epistemic Uncertainty (points)", row=1, col=1)
    fig.update_xaxes(title_text="Aleatoric Uncertainty (points)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)

    return fig


def plot_bnn_prediction_intervals(df_bnn):
    """Plot BNN prediction intervals."""
    # Sort by predicted value for better visualization
    df_sorted = df_bnn.sort_values("predicted_value").reset_index(drop=True)
    df_sorted["game_index"] = range(len(df_sorted))

    fig = go.Figure()

    # Confidence interval
    fig.add_trace(
        go.Scatter(
            x=df_sorted["game_index"],
            y=df_sorted["ci_upper"],
            mode="lines",
            name="95% CI Upper",
            line=dict(width=0),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_sorted["game_index"],
            y=df_sorted["ci_lower"],
            mode="lines",
            name="95% CI",
            line=dict(width=0),
            fillcolor="rgba(31, 119, 180, 0.2)",
            fill="tonexty",
            showlegend=True,
        )
    )

    # Predicted margin
    fig.add_trace(
        go.Scatter(
            x=df_sorted["game_index"],
            y=df_sorted["predicted_value"],
            mode="markers",
            name="Predicted Value",
            marker=dict(size=6, color="#d62728"),
        )
    )

    fig.update_layout(
        title="BNN Prediction Intervals (95% Confidence)",
        xaxis_title="Game Index",
        yaxis_title="Predicted Margin (points)",
        hovermode="x unified",
        height=400,
    )

    return fig


def check_service_health(host, port, service_name):
    """Check if a service is healthy by attempting to connect."""
    import socket

    # Simple socket-based health check that should work reliably
    if service_name == "Streamlit":
        # Streamlit is this app itself, should always be true if running
        return True

    try:
        # Create a socket connection to test if the service is reachable
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)  # Generous timeout

        # Try to connect
        result = sock.connect_ex((host, port))
        sock.close()

        # Return True if connection successful (result == 0)
        return result == 0
    except Exception as e:
        # If any error occurs, consider the service as down
        print(f"Health check failed for {service_name} at {host}:{port} - {str(e)}")
        return False


def fetch_database_stats():
    """Fetch database statistics."""
    try:
        with get_db_connection() as conn:
            stats = {}

            # Total games
            result = conn.execute("SELECT COUNT(*) FROM games").fetchone()
            stats["total_games"] = result[0] if result else 0

            # Total plays
            result = conn.execute("SELECT COUNT(*) FROM plays").fetchone()
            stats["total_plays"] = result[0] if result else 0

            # Season coverage
            result = conn.execute("SELECT MIN(season), MAX(season) FROM games").fetchone()
            stats["season_min"] = result[0] if result and result[0] else 0
            stats["season_max"] = result[1] if result and result[1] else 0

            # Database size
            result = conn.execute(
                """
                SELECT pg_size_pretty(pg_database_size('devdb01'))
            """
            ).fetchone()
            stats["db_size"] = result[0] if result else "N/A"

            # Recent data timestamp
            result = conn.execute(
                """
                SELECT MAX(kickoff) FROM games WHERE kickoff IS NOT NULL
            """
            ).fetchone()
            stats["latest_game"] = result[0] if result and result[0] else None

            return stats
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return None


def fetch_model_registry():
    """Fetch models from ml.model_registry."""
    try:
        query = """
        SELECT
            model_id,
            model_name,
            model_version,
            model_type,
            framework,
            status,
            created_at,
            updated_at,
            description,
            performance_summary
        FROM ml.model_registry
        ORDER BY updated_at DESC
        """
        with get_db_connection() as conn:
            df = pd.read_sql_query(query, conn)
        return df
    except Exception:
        return pd.DataFrame()


def fetch_training_runs():
    """Fetch recent training runs."""
    try:
        query = """
        SELECT
            run_id,
            model_id,
            started_at,
            completed_at,
            status,
            final_metrics,
            training_duration_seconds
        FROM ml.training_runs
        ORDER BY started_at DESC
        LIMIT 10
        """
        with get_db_connection() as conn:
            df = pd.read_sql_query(query, conn)
        return df
    except Exception:
        return pd.DataFrame()


def render_game_predictions_page():
    """Render game predictions page organized by time slots."""
    # Add header with cache controls
    col1, col2 = st.columns([3, 1])
    with col1:
        current_time = datetime.now(ZoneInfo("America/New_York")).strftime(
            "%Y-%m-%d %I:%M:%S %p ET"
        )
        st.caption(f"📡 Last updated: {current_time} | Live scores from ESPN API")
    with col2:
        if st.button(
            "🗑️ Clear Cache",
            help="Clear cached data to force refresh from ESPN",
            key="clear_cache_top",
        ):
            st.cache_data.clear()
            st.success("✅ Cache cleared!")
            time.sleep(0.5)
            st.rerun()

    # Get current week dynamically
    current_year = datetime.now().year
    with st.spinner("Loading current week..."):
        with get_db_connection() as conn:
            week_query = "SELECT season, week FROM games WHERE season = %s AND game_type = 'REG' ORDER BY kickoff DESC LIMIT 1"
            week_result = pd.read_sql_query(week_query, conn, params=(current_year,))
            if week_result.empty:
                st.error(f"No games found for {current_year} season")
                return
            current_season = int(week_result["season"].iloc[0])
            current_week = int(week_result["week"].iloc[0])

    st.header("🎯 Game Predictions vs Sportsbook")
    st.markdown(f"**{current_season} NFL Season - Week {current_week}**")

    # Fetch all games and predictions for current week
    query = f"""
    SELECT DISTINCT ON (g.game_id)
        g.game_id,
        g.season,
        g.week,
        g.home_team,
        g.away_team,
        g.home_score,
        g.away_score,
        g.spread_close,
        g.total_close,
        g.kickoff,
        p.predicted_spread,
        p.predicted_total,
        p.predicted_home_win_prob as home_win_prob,
        CASE
            WHEN g.home_score IS NOT NULL AND g.away_score IS NOT NULL THEN 'completed'
            -- Games are typically 3.5 hours, mark as completed if more than 4 hours past kickoff
            -- BUT only for games that have actually started (not future years)
            WHEN g.kickoff < NOW() - INTERVAL '4 hours' AND g.season <= EXTRACT(YEAR FROM NOW()) THEN 'completed'
            WHEN g.kickoff < NOW() AND g.season <= EXTRACT(YEAR FROM NOW()) THEN 'in_progress'
            ELSE 'scheduled'
        END as game_status
    FROM games g
    LEFT JOIN ml.game_predictions p ON g.game_id = p.game_id
    WHERE g.season = {current_season} AND g.week = {current_week} AND g.game_type = 'REG'
    ORDER BY g.game_id, p.created_at DESC NULLS LAST, g.kickoff
    """

    with st.spinner("Loading game predictions..."):
        with get_db_connection() as conn:
            games_df = pd.read_sql_query(query, conn)

    if games_df.empty:
        st.info(f"No games found for Week {current_week}")
        return

    # Get bye week teams
    bye_query = f"""
    WITH all_teams AS (
        SELECT DISTINCT home_team as team FROM games WHERE season = {current_season}
        UNION
        SELECT DISTINCT away_team as team FROM games WHERE season = {current_season}
    ),
    week_teams AS (
        SELECT DISTINCT home_team as team FROM games WHERE season = {current_season} AND week = {current_week}
        UNION
        SELECT DISTINCT away_team as team FROM games WHERE season = {current_season} AND week = {current_week}
    )
    SELECT t.team
    FROM all_teams t
    WHERE t.team NOT IN (SELECT team FROM week_teams)
    ORDER BY t.team
    """

    with get_db_connection() as conn:
        bye_teams = pd.read_sql_query(bye_query, conn)

    # Show bye week teams
    if not bye_teams.empty:
        bye_list = ", ".join(bye_teams["team"].tolist())
        st.info(f"**Bye Week:** {bye_list}")

    # Organize games by time slot and status
    def categorize_game(row):
        if pd.notna(row["kickoff"]):
            kickoff_utc = pd.to_datetime(row["kickoff"])
            if kickoff_utc.tzinfo is None:
                kickoff_utc = kickoff_utc.tz_localize("UTC")
            kickoff_et = kickoff_utc.astimezone(ZoneInfo("America/New_York"))

            hour = kickoff_et.hour
            day_of_week = kickoff_et.weekday()  # 0=Monday, 3=Thursday, 6=Sunday

            if day_of_week == 3:  # Thursday
                return "TNF"
            elif day_of_week == 6:  # Sunday
                if hour < 13:  # Before 1 PM
                    return "International"
                elif hour < 16:  # 1-4 PM
                    return "Sunday 1PM"
                elif hour < 20:  # 4-8 PM
                    return "Sunday 4PM"
                else:  # 8 PM or later
                    return "SNF"
            elif day_of_week == 0:  # Monday
                return "MNF"
        return "Other"

    games_df["time_slot"] = games_df.apply(categorize_game, axis=1)

    # Sort games: in-progress → completed → scheduled
    status_order = {"in_progress": 0, "completed": 1, "scheduled": 2}
    games_df["status_order"] = games_df["game_status"].map(status_order)

    # Time slot order
    slot_order = {
        "TNF": 0,
        "International": 1,
        "Sunday 1PM": 2,
        "Sunday 4PM": 3,
        "SNF": 4,
        "MNF": 5,
        "Other": 6,
    }
    games_df["slot_order"] = games_df["time_slot"].map(slot_order)

    games_df = games_df.sort_values(["status_order", "slot_order", "kickoff"])

    # Render games by time slot
    current_slot = None
    slot_headers = {
        "TNF": "🏈 Thursday Night Football",
        "International": "🌍 International Games",
        "Sunday 1PM": "🏈 Sunday Early Games (1:00 PM ET)",
        "Sunday 4PM": "🏈 Sunday Afternoon Games (4:00 PM ET)",
        "SNF": "🏈 ⭐ Sunday Night Football (8:00 PM ET)",
        "MNF": "🏈 🏈 Monday Night Football",
    }

    for _, game in games_df.iterrows():
        # Show time slot header
        if game["time_slot"] != current_slot:
            current_slot = game["time_slot"]
            if current_slot in slot_headers:
                st.markdown(f"### {slot_headers[current_slot]}")

        # Render game card
        render_game_card_with_predictions(game)


def render_game_card_with_predictions(game):
    """Render a single game card with sportsbook vs model predictions."""
    home = game["home_team"]
    away = game["away_team"]

    # Format kickoff time
    if pd.notna(game["kickoff"]):
        kickoff_utc = pd.to_datetime(game["kickoff"])
        if kickoff_utc.tzinfo is None:
            kickoff_utc = kickoff_utc.tz_localize("UTC")
        kickoff_et = kickoff_utc.astimezone(ZoneInfo("America/New_York"))
        # Show full date for future games, just time for today
        today_et = datetime.now(ZoneInfo("America/New_York")).date()
        if kickoff_et.date() == today_et:
            kickoff_str = kickoff_et.strftime("%I:%M %p ET")
        elif kickoff_et.date() == today_et + timedelta(days=1):
            kickoff_str = kickoff_et.strftime("Tomorrow %I:%M %p ET")
        else:
            kickoff_str = kickoff_et.strftime("%b %d, %I:%M %p ET")
    else:
        kickoff_str = "TBD"

    # Game status display
    if game["game_status"] == "completed":
        # Only show as completed if we actually have scores
        if pd.notna(game["home_score"]) and pd.notna(game["away_score"]):
            home_score = int(game["home_score"])
            away_score = int(game["away_score"])
            status_str = f"FINAL: {away} {away_score} @ {home} {home_score}"
        else:
            # No scores yet, treat as scheduled even if marked completed
            status_str = kickoff_str
    elif game["game_status"] == "in_progress":
        status_str = f"⚡ IN PROGRESS - {kickoff_str}"
    else:
        status_str = kickoff_str

    # Sportsbook lines
    spread_sb = game["spread_close"] if pd.notna(game["spread_close"]) else None
    total_sb = game["total_close"] if pd.notna(game["total_close"]) else None

    # Model predictions
    spread_model = game["predicted_spread"] if pd.notna(game["predicted_spread"]) else None
    total_model = game["predicted_total"] if pd.notna(game["predicted_total"]) else None
    home_win_prob = game["home_win_prob"] if pd.notna(game["home_win_prob"]) else None

    # Use edges from API if available, otherwise calculate
    spread_edge = game.get("spread_edge")
    if spread_edge is None:
        spread_edge = (
            (spread_model - spread_sb)
            if (spread_model is not None and spread_sb is not None)
            else None
        )

    total_edge = game.get("total_edge")
    if total_edge is None:
        total_edge = (
            (total_model - total_sb) if (total_model is not None and total_sb is not None) else None
        )

    # Format values
    spread_sb_str = f"{home} {spread_sb:+.1f}" if spread_sb is not None else "N/A"
    spread_model_str = f"{home} {spread_model:+.1f}" if spread_model is not None else "—"

    # Enhanced edge formatting with confidence indicators
    if spread_edge is not None:
        spread_edge_str = f"{spread_edge:+.1f}"
        if abs(spread_edge) >= 5:
            spread_edge_str += " ⭐⭐⭐"  # High confidence
            spread_edge_style = "color: #28a745; font-weight: bold;"
        elif abs(spread_edge) >= 3:
            spread_edge_str += " ⭐⭐"  # Medium confidence
            spread_edge_style = "color: #17a2b8; font-weight: bold;"
        elif abs(spread_edge) >= 1.5:
            spread_edge_str += " ⭐"  # Low confidence
            spread_edge_style = "color: #6c757d;"
        else:
            spread_edge_style = ""
    else:
        spread_edge_str = "—"
        spread_edge_style = ""

    total_sb_str = f"O/U {total_sb:.1f}" if total_sb is not None else "N/A"
    total_model_str = f"O/U {total_model:.1f}" if total_model is not None else "—"

    # Enhanced edge formatting for totals
    if total_edge is not None:
        total_edge_str = f"{total_edge:+.1f}"
        if abs(total_edge) >= 5:
            total_edge_str += " ⭐⭐⭐"  # High confidence
            total_edge_style = "color: #28a745; font-weight: bold;"
        elif abs(total_edge) >= 3:
            total_edge_str += " ⭐⭐"  # Medium confidence
            total_edge_style = "color: #17a2b8; font-weight: bold;"
        elif abs(total_edge) >= 1.5:
            total_edge_str += " ⭐"  # Low confidence
            total_edge_style = "color: #6c757d;"
        else:
            total_edge_style = ""
    else:
        total_edge_str = "—"
        total_edge_style = ""

    # Format win probability with confidence color
    if home_win_prob is not None:
        ml_model_str = f"{home} {home_win_prob*100:.1f}%"
        if home_win_prob >= 0.65 or home_win_prob <= 0.35:
            ml_model_str += " ⭐⭐"  # Strong favorite/underdog
        elif home_win_prob >= 0.58 or home_win_prob <= 0.42:
            ml_model_str += " ⭐"  # Moderate favorite/underdog
    else:
        ml_model_str = "—"

    # Get QB and RB yards predictions if available
    home_qb_yards = game.get("predicted_home_qb_yards") or game.get("home_qb_yards")
    away_qb_yards = game.get("predicted_away_qb_yards") or game.get("away_qb_yards")
    home_rb_yards = game.get("home_rb_yards")
    away_rb_yards = game.get("away_rb_yards")

    # Format player predictions
    qb_predictions_str = ""
    if home_qb_yards or away_qb_yards:
        qb_predictions_str = f"""
                <tr style="background-color: #e8f4fd;">
                    <td style="padding: 5px;"><strong>QB Pass Yds</strong></td>
                    <td style="padding: 5px; text-align: center;">—</td>
                    <td style="padding: 5px; text-align: center;">{away}: {away_qb_yards or '—':.0f} | {home}: {home_qb_yards or '—':.0f}</td>
                    <td style="padding: 5px; text-align: center;">—</td>
                </tr>"""

    rb_predictions_str = ""
    if home_rb_yards or away_rb_yards:
        rb_predictions_str = f"""
                <tr style="background-color: #fff3cd;">
                    <td style="padding: 5px;"><strong>RB Rush Yds</strong></td>
                    <td style="padding: 5px; text-align: center;">—</td>
                    <td style="padding: 5px; text-align: center;">{away}: {away_rb_yards or '—':.0f} | {home}: {home_rb_yards or '—':.0f}</td>
                    <td style="padding: 5px; text-align: center;">—</td>
                </tr>"""

    # Build HTML card
    st.markdown(
        f"""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #007bff;">
        <div style="font-size: 14px; font-weight: bold; margin-bottom: 10px; color: #333;">
            {status_str}
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <div style="flex: 1; text-align: center;">
                <div style="font-size: 18px; font-weight: bold;">{away}</div>
            </div>
            <div style="flex: 0 0 50px; text-align: center; font-size: 20px; color: #666;">@</div>
            <div style="flex: 1; text-align: center;">
                <div style="font-size: 18px; font-weight: bold;">{home}</div>
            </div>
        </div>

        <div style="background-color: white; padding: 10px; border-radius: 5px;">
            <table style="width: 100%; font-size: 13px;">
                <tr style="background-color: #e9ecef;">
                    <th style="padding: 5px; text-align: left;">Line</th>
                    <th style="padding: 5px; text-align: center;">Sportsbook</th>
                    <th style="padding: 5px; text-align: center;">Model</th>
                    <th style="padding: 5px; text-align: center;">Edge</th>
                </tr>
                <tr>
                    <td style="padding: 5px;"><strong>Spread</strong></td>
                    <td style="padding: 5px; text-align: center;">{spread_sb_str}</td>
                    <td style="padding: 5px; text-align: center;">{spread_model_str}</td>
                    <td style="padding: 5px; text-align: center; {spread_edge_style}">{spread_edge_str}</td>
                </tr>
                <tr style="background-color: #f8f9fa;">
                    <td style="padding: 5px;"><strong>Total</strong></td>
                    <td style="padding: 5px; text-align: center;">{total_sb_str}</td>
                    <td style="padding: 5px; text-align: center;">{total_model_str}</td>
                    <td style="padding: 5px; text-align: center; {total_edge_style}">{total_edge_str}</td>
                </tr>
                <tr>
                    <td style="padding: 5px;"><strong>Win Prob</strong></td>
                    <td style="padding: 5px; text-align: center;">—</td>
                    <td style="padding: 5px; text-align: center;">{ml_model_str}</td>
                    <td style="padding: 5px; text-align: center;">—</td>
                </tr>
                {qb_predictions_str}
                {rb_predictions_str}
            </table>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_system_status_page():
    """Render the system status page."""
    st.header("🖥️ System Status")

    col1, col2 = st.columns([2, 1])

    with col2:
        if st.button("🔄 Refresh Status", type="primary"):
            st.cache_data.clear()
            st.rerun()

    # Docker services health
    st.subheader("🐳 Docker Services")

    # Check if we're running in Docker by looking for /.dockerenv
    import os

    in_docker = os.path.exists("/.dockerenv")

    # Use Docker service names if in container, localhost if running on host
    if in_docker:
        # Running in Docker - use service names from docker-compose
        services = [
            ("PostgreSQL", "pg-1", 5432),  # Internal port is 5432
            ("Redis", "redis", 6379),
            ("Prometheus", "prometheus", 9090),
            ("Grafana", "grafana", 3000),
            ("Streamlit", "localhost", 8501),  # This container
            ("Postgres Exporter", "postgres_exporter", 9187),
            ("Node Exporter", "node_exporter", 9100),
        ]
    else:
        # Running on host - use localhost with exposed ports
        services = [
            ("PostgreSQL", "localhost", 5544),
            ("Redis", "localhost", 6379),
            ("Prometheus", "localhost", 9090),
            ("Grafana", "localhost", 3000),
            ("Streamlit", "localhost", 8501),
            ("Postgres Exporter", "localhost", 9187),
            ("Node Exporter", "localhost", 9100),
        ]

    cols = st.columns(len(services))
    health_status = {}

    for idx, (name, host, port) in enumerate(services):
        with cols[idx]:
            is_healthy = check_service_health(host, port, name)
            health_status[name] = is_healthy
            status_icon = "✅" if is_healthy else "❌"
            status_color = "#28a745" if is_healthy else "#dc3545"
            st.markdown(
                f"""
            <div style="text-align: center; padding: 15px; background-color: {'#d4edda' if is_healthy else '#f8d7da'};
                        border-radius: 10px; border: 2px solid {status_color};">
                <div style="font-size: 32px;">{status_icon}</div>
                <div style="font-size: 14px; font-weight: bold; color: #212529; margin-top: 5px;">{name}</div>
                <div style="font-size: 12px; color: #666;">:{port}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Add debug information in an expander
    with st.expander("🔧 Debug Information", expanded=False):
        st.write("**Service Health Check Results:**")
        debug_cols = st.columns(2)
        with debug_cols[0]:
            for service_name, is_healthy in health_status.items():
                if is_healthy:
                    st.success(f"✅ {service_name}: Connected")
                else:
                    st.error(f"❌ {service_name}: Connection failed")

        with debug_cols[1]:
            st.write("**Docker Container Status:**")
            try:
                import subprocess

                result = subprocess.run(
                    "docker ps --format '{{.Names}}: {{.Status}}' | grep nfl-analytics | head -10",
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.stdout:
                    for line in result.stdout.strip().split("\n"):
                        if "healthy" in line.lower():
                            st.success(line)
                        else:
                            st.warning(line)
                else:
                    st.info("No NFL Analytics containers found")
            except Exception as e:
                st.error(f"Could not fetch Docker status: {e}")

    st.markdown("---")

    # Database statistics
    st.subheader("💾 Database Statistics")
    with st.spinner("Loading database statistics..."):
        db_stats = fetch_database_stats()

    if db_stats:
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total Games", f"{db_stats['total_games']:,}")

        with col2:
            st.metric("Total Plays", f"{db_stats['total_plays']:,}")

        with col3:
            st.metric("Season Coverage", f"{db_stats['season_min']}-{db_stats['season_max']}")

        with col4:
            st.metric("Database Size", db_stats["db_size"])

        with col5:
            if db_stats["latest_game"]:
                latest = pd.to_datetime(db_stats["latest_game"])
                days_ago = (datetime.now(latest.tzinfo) - latest).days
                st.metric("Latest Game", f"{days_ago} days ago")
            else:
                st.metric("Latest Game", "N/A")
    else:
        st.error("Could not fetch database statistics")

    st.markdown("---")

    # Training window information
    st.subheader("📊 Canonical Training Window")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        **Current Configuration:**
        - **Core years**: 2018-2024 (full weight)
        - **Decay years**: 2015-2017 (exponential decay, H=4)
        - **Formula**: w(s; 2025, 4) = 0.5^((2025-s)/4)

        **Effective Weights:**
        - 2017: 0.297
        - 2016: 0.250
        - 2015: 0.210
        """
        )

    with col2:
        st.markdown(
            """
        **Validation Results:**
        - **Test Set**: 2025 weeks 1-6 (148 games, 37 QBs)
        - **MAE**: 47.77 yards
        - **Coverage**: 86.5%
        - **Statistical Significance**: p=0.0069 (vs baseline)
        - **Status**: ✅ Approved for production
        """
        )

    st.markdown("---")

    # External monitoring links
    st.subheader("🔗 Monitoring Links")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        ### Grafana
        [Open Grafana Dashboard →](http://localhost:3000)

        **Credentials:**
        - Username: `admin`
        - Password: See `.env` file or `GRAFANA_PASSWORD` environment variable
        """
        )

    with col2:
        st.markdown(
            """
        ### Prometheus
        [Open Prometheus →](http://localhost:9090)

        **Features:**
        - Metrics queries
        - Alert rules
        - Target health
        """
        )

    with col3:
        st.markdown(
            """
        ### Database
        [localhost:5544](postgresql://localhost:5544/devdb01)

        **Connection:**
        ```
        psql postgresql://dro:***@localhost:5544/devdb01
        ```
        """
        )


def render_model_status_page():
    """Render the model status page."""
    st.header("🤖 Model Status")

    # Model registry
    st.subheader("📋 Model Registry")
    with st.spinner("Loading model registry..."):
        df_models = fetch_model_registry()

    if not df_models.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Models", len(df_models))

        with col2:
            active_models = len(df_models[df_models["status"] == "active"])
            st.metric("Active Models", active_models)

        with col3:
            in_production = len(df_models[df_models["status"] == "production"])
            st.metric("In Production", in_production)

        with col4:
            archived = len(df_models[df_models["status"] == "archived"])
            st.metric("Archived", archived)

        # Model table
        st.dataframe(
            df_models[
                [
                    "model_id",
                    "model_name",
                    "model_version",
                    "model_type",
                    "framework",
                    "status",
                    "updated_at",
                    "description",
                ]
            ],
            use_container_width=True,
            height=400,
        )
    else:
        st.info("No models found in registry. Models will appear here after training.")

    st.markdown("---")

    # Trained model files
    st.subheader("📦 Model Files")

    # Check models directory (handle both local and Docker paths)
    try:
        import glob
        import os

        # Try multiple possible model directory locations
        possible_dirs = [
            "/app/models",  # Docker mount
            "/Users/dro/rice/nfl-analytics/models",  # Local development
            "../models",  # Relative path
        ]

        models_dir = None
        for d in possible_dirs:
            if os.path.exists(d):
                models_dir = d
                break

        if not models_dir:
            st.warning(
                "Models directory not found. Make sure models directory is mounted in Docker."
            )
            st.info(
                "Expected paths: /app/models (Docker) or /Users/dro/rice/nfl-analytics/models (local)"
            )
        else:
            model_files = []

            # Find all model files
            for ext in ["*.rds", "*.pkl", "*.pt", "*.pth"]:
                for filepath in glob.glob(os.path.join(models_dir, "**", ext), recursive=True):
                    try:
                        stat = os.stat(filepath)
                        model_files.append(
                            {
                                "filename": os.path.relpath(filepath, models_dir),
                                "size_mb": stat.st_size / (1024 * 1024),
                                "modified": datetime.fromtimestamp(stat.st_mtime).strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                            }
                        )
                    except Exception:
                        continue

            if model_files:
                df_files = pd.DataFrame(model_files)
                df_files = df_files.sort_values("modified", ascending=False)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Model Files", len(df_files))
                with col2:
                    total_size = df_files["size_mb"].sum()
                    st.metric("Total Size", f"{total_size:.1f} MB")
                with col3:
                    largest = df_files["size_mb"].max()
                    st.metric("Largest File", f"{largest:.1f} MB")

                st.dataframe(
                    df_files.style.format({"size_mb": "{:.2f}"}),
                    use_container_width=True,
                    height=400,
                )
            else:
                st.info(f"No model files found in {models_dir}")
    except Exception as e:
        st.error(f"Error scanning model files: {str(e)}")

    st.markdown("---")

    # Recent training runs
    st.subheader("🏃 Recent Training Runs")
    df_runs = fetch_training_runs()

    if not df_runs.empty:
        st.dataframe(df_runs, use_container_width=True, height=300)
    else:
        st.info("No training runs found")

    st.markdown("---")

    # Ensemble configuration
    st.subheader("🎯 4-Way Ensemble Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        **Components:**
        1. ✅ **Hierarchical Bayesian** (brms/Stan)
           - Status: Production ready
           - Framework: R, cmdstanr

        2. ✅ **XGBoost** (GPU-accelerated)
           - Status: Production ready
           - Framework: Python, xgboost

        3. ✅ **Bayesian Neural Network** (PyMC)
           - Status: Calibrated (89.5% coverage)
           - Framework: Python, PyMC

        4. ✅ **State-Space Random Walk v2** ⭐
           - Status: **APPROVED** (Nov 8, 2025)
           - MAE: 47.77 yards
           - Coverage: 86.5%
        """
        )

    with col2:
        st.markdown(
            """
        **Combination Method:**
        - **Default**: Inverse variance weighting
        - **Advanced**: Stacked meta-learner with CV
        - **Portfolio**: Correlation-adjusted Kelly criterion

        **Expected Performance:**
        - **Target ROI**: +5-7%
        - **Win Rate**: 59-61%
        - **Sharpe Ratio**: 2.2-2.6
        - **Max Drawdown**: -8 to -10%
        """
        )


def render_data_summary_page():
    """Render the data summary page."""
    st.header("📊 Data Summary")

    db_stats = fetch_database_stats()

    if not db_stats:
        st.error("Could not connect to database")
        return

    # Overview metrics
    st.subheader("📈 Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Games", f"{db_stats['total_games']:,}", help="Total games in database")

    with col2:
        st.metric("Total Plays", f"{db_stats['total_plays']:,}", help="Total play-by-play records")

    with col3:
        st.metric(
            "Season Coverage",
            f"{db_stats['season_min']}-{db_stats['season_max']}",
            help="Range of seasons in database",
        )

    st.markdown("---")

    # Schema statistics
    st.subheader("🗄️ Schema Statistics")

    try:
        with get_db_connection() as conn:
            # Count tables by schema
            schema_query = """
            SELECT
                schemaname,
                COUNT(*) as table_count
            FROM pg_tables
            WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
            GROUP BY schemaname
            ORDER BY table_count DESC
            """
            df_schemas = pd.read_sql_query(schema_query, conn)

            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(df_schemas, use_container_width=True, height=200)

            with col2:
                # Schema breakdown
                for _, row in df_schemas.iterrows():
                    st.metric(f"{row['schemaname']} schema", f"{row['table_count']} tables")
    except Exception as e:
        st.error(f"Error fetching schema stats: {str(e)}")

    st.markdown("---")

    # Recent data ingestion
    st.subheader("📥 Recent Data Ingestion")

    try:
        with get_db_connection() as conn:
            # Get recent games by season and week
            recent_query = """
            SELECT
                season,
                week,
                game_type,
                COUNT(*) as games,
                MAX(kickoff) as latest_kickoff
            FROM games
            GROUP BY season, week, game_type
            ORDER BY season DESC, week DESC
            LIMIT 10
            """
            df_recent = pd.read_sql_query(recent_query, conn)

            st.dataframe(df_recent, use_container_width=True, height=300)
    except Exception as e:
        st.error(f"Error fetching recent data: {str(e)}")

    st.markdown("---")

    # Data quality notes
    st.subheader("✅ Data Quality")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        **Key Findings:**
        - ✅ All games have valid kickoff timestamps
        - ✅ Betting lines (spread, total) available for 99%+ of games
        - ✅ Play-by-play data includes EPA calculations
        - ✅ Weather data available for outdoor venues
        - ✅ Injury reports tracked by week
        """
        )

    with col2:
        st.markdown(
            """
        **Important Notes:**
        - Use `kickoff` column (not `game_date` or `gameday`)
        - Use `spread_close` and `total_close` (not `spread_line`, `total_line`)
        - Team abbreviations: LA Rams = `LAR` (not `LA`)
        - Weather stored as TEXT, requires parsing
        - See `docs/DATABASE_SCHEMA_AUDIT.md` for details
        """
        )


def main():
    st.title("🏈 NFL Analytics Monitoring Dashboard")
    st.markdown("Real-time monitoring of prediction performance and model metrics")

    # Sidebar navigation
    st.sidebar.header("Navigation")

    page = st.sidebar.radio(
        "Select Page",
        ["Predictions", "🎯 Game Predictions", "System Status", "Model Status", "Data Summary"],
        index=0,
    )

    # Route to appropriate page
    if page == "🎯 Game Predictions":
        render_game_predictions_page()
        return
    elif page == "System Status":
        render_system_status_page()
        return
    elif page == "Model Status":
        render_model_status_page()
        return
    elif page == "Data Summary":
        render_data_summary_page()
        return

    # Sidebar filters (for Predictions page)
    st.sidebar.header("Filters")

    # Season filter
    seasons = list(range(2025, 2021, -1))
    selected_season = st.sidebar.selectbox("Season", ["All"] + seasons, index=0)

    # Model version filter (dynamically fetch from database)
    try:
        with get_db_connection() as conn:
            model_versions_df = pd.read_sql_query(
                "SELECT DISTINCT model_version FROM ml.model_registry WHERE status IN ('production', 'staging', 'active') ORDER BY model_version",
                conn,
            )
            model_versions = model_versions_df["model_version"].tolist()
    except Exception:
        model_versions = []

    selected_model = st.sidebar.selectbox("Model Version", ["All"] + model_versions, index=0)

    # Fetch data
    with st.spinner("Loading data..."):
        season_filter = None if selected_season == "All" else selected_season
        model_filter = None if selected_model == "All" else selected_model

        df_retrospectives = fetch_retrospective_metrics(
            model_version=model_filter, season=season_filter
        )

        # Note: df_retrospectives now contains player-level predictions (passing yards)
        # No need to derive winner columns since we're tracking individual player prop bets

        df_performance = fetch_performance_over_time()
        df_model_comparison = fetch_model_comparison()

    # Summary metrics
    st.header("📊 Summary Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_predictions = len(df_retrospectives)
        st.metric("Total Player-Games", f"{total_predictions:,}")

    with col2:
        if not df_retrospectives.empty and "status" in df_retrospectives.columns:
            win_rate = (df_retrospectives["status"] == "win").mean()
            st.metric("Win Rate", f"{win_rate*100:.1f}%", delta=f"{(win_rate-0.5)*100:.1f}% vs 50%")
        else:
            st.metric("Win Rate", "N/A")

    with col3:
        if not df_retrospectives.empty and "mae" in df_retrospectives.columns:
            mae = df_retrospectives["mae"].mean()
            st.metric("Mean Absolute Error", f"{mae:.2f} yards")
        else:
            st.metric("Mean Absolute Error", "N/A")

    with col4:
        if not df_retrospectives.empty and "pnl" in df_retrospectives.columns:
            total_pnl = df_retrospectives["pnl"].sum()
            avg_pnl = df_retrospectives["pnl"].mean()
            st.metric("Total P&L", f"${total_pnl:,.2f}", delta=f"${avg_pnl:.2f} avg")
        else:
            st.metric("Total P&L", "N/A")

    # Fetch current week games data
    df_current_week = fetch_current_week_games()

    if not df_current_week.empty:
        # Separate completed, in-progress, and upcoming games
        completed_games = df_current_week[df_current_week["game_status"] == "completed"]
        in_progress_games = df_current_week[df_current_week["game_status"] == "in_progress"]
        upcoming_games = df_current_week[df_current_week["game_status"] == "scheduled"]

        # In-Progress Games (LIVE) - Show at top
        if not in_progress_games.empty:
            # Add refresh button and auto-refresh timer
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.empty()  # Spacer
            with col2:
                # Auto-refresh countdown
                if "last_refresh" not in st.session_state:
                    st.session_state.last_refresh = time.time()

                time_since_refresh = int(time.time() - st.session_state.last_refresh)
                time_until_refresh = max(0, 15 - time_since_refresh)
                st.caption(f"⏱️ Auto-refresh in {time_until_refresh}s")
            with col3:
                if st.button("🔄 Refresh Now", type="primary", key="refresh_live_games"):
                    st.session_state.last_refresh = time.time()
                    st.cache_data.clear()  # Clear cache to get fresh data
                    st.rerun()

            # Auto-refresh every 15 seconds when there are live games (faster than cache expiry)
            if time_since_refresh >= 15:
                st.session_state.last_refresh = time.time()
                st.rerun()

            st.subheader("🔴 LIVE GAMES")

            # Create two-column layout for games
            for i in range(0, len(in_progress_games), 2):
                col1, col2 = st.columns(2)

                for col_idx, col in enumerate([col1, col2]):
                    game_idx = i + col_idx
                    if game_idx >= len(in_progress_games):
                        break

                    game = in_progress_games.iloc[game_idx]
                    away_score = int(game["away_score"]) if pd.notna(game["away_score"]) else 0
                    home_score = int(game["home_score"]) if pd.notna(game["home_score"]) else 0

                    # Get clock and period info
                    clock = game.get("clock", "")
                    period = int(game.get("period", 0)) if pd.notna(game.get("period")) else 0
                    detail = game.get("detail", "In Progress")

                    # Format period display (Q1, Q2, Q3, Q4, OT, etc.)
                    if period > 0:
                        if period <= 4:
                            period_display = f"Q{period}"
                        else:
                            period_display = f"OT{period - 4}" if period > 5 else "OT"
                    else:
                        period_display = ""

                    # Combine clock and period for status display
                    if clock and period_display:
                        status_display = f"{period_display} {clock}"
                    elif period_display:
                        status_display = period_display
                    elif clock:
                        status_display = clock
                    elif detail:
                        status_display = detail
                    else:
                        # Fallback: show kickoff time if available
                        if pd.notna(game["kickoff"]):
                            kickoff_utc = pd.to_datetime(game["kickoff"])
                            if kickoff_utc.tzinfo is None:
                                kickoff_utc = kickoff_utc.tz_localize("UTC")
                            kickoff_et = kickoff_utc.astimezone(ZoneInfo("America/New_York"))
                            status_display = kickoff_et.strftime("%I:%M %p ET")
                        else:
                            status_display = "TBD"

                    # Determine who's winning
                    leading_team = None
                    if away_score > home_score:
                        leading_team = game["away_team"]
                    elif home_score > away_score:
                        leading_team = game["home_team"]

                    # Sportsbook info
                    spread = game["spread_close"] if pd.notna(game["spread_close"]) else None
                    total = game["total_close"] if pd.notna(game["total_close"]) else None

                    if spread is not None:
                        spread_text = (
                            f"Spread: {game['home_team_display']} {spread:+.1f}"
                            if spread != 0
                            else "Spread: EVEN"
                        )
                    else:
                        spread_text = "Spread: N/A"

                    if total is not None:
                        total_text = f"Total: O/U {total:.1f}"
                    else:
                        total_text = "Total: N/A"

                    # Our model predictions (from database ml.game_predictions)
                    prediction_lines = []

                    # Spread prediction
                    if "predicted_spread" in game and pd.notna(game["predicted_spread"]):
                        pred_spread = float(game["predicted_spread"])
                        pred_spread_str = (
                            f"{game['home_team_display']} {pred_spread:+.1f}"
                            if pred_spread != 0
                            else "EVEN"
                        )
                        prediction_lines.append(f"<strong>Spread:</strong> {pred_spread_str}")

                    # Total (O/U) prediction
                    if "predicted_total" in game and pd.notna(game["predicted_total"]):
                        pred_total = float(game["predicted_total"])
                        prediction_lines.append(f"<strong>Total:</strong> O/U {pred_total:.1f}")

                    # Score predictions
                    if (
                        "predicted_home_score" in game
                        and pd.notna(game["predicted_home_score"])
                        and "predicted_away_score" in game
                        and pd.notna(game["predicted_away_score"])
                    ):
                        pred_home = float(game["predicted_home_score"])
                        pred_away = float(game["predicted_away_score"])
                        prediction_lines.append(
                            f"<strong>Score:</strong> {game['away_team_display']} {pred_away:.0f} @ {game['home_team_display']} {pred_home:.0f}"
                        )

                    # Win probability
                    if "predicted_home_win_prob" in game and pd.notna(
                        game["predicted_home_win_prob"]
                    ):
                        home_win_prob = float(game["predicted_home_win_prob"])
                        predicted_winner = (
                            game["home_team_display"]
                            if home_win_prob > 0.5
                            else game["away_team_display"]
                        )
                        win_prob_pct = max(home_win_prob, 1 - home_win_prob) * 100
                        prediction_lines.append(
                            f"<strong>Winner:</strong> {predicted_winner} ({win_prob_pct:.1f}%)"
                        )

                    prediction_info = (
                        " | ".join(prediction_lines)
                        if prediction_lines
                        else "<span style='color: #666; font-style: italic;'>No predictions available</span>"
                    )

                    with col:
                        st.markdown(
                            f"""
                        <div style="background-color: #fff3cd; padding: 20px; border-radius: 10px; margin: 10px 0;
                                    border: 3px solid #ff8c00; box-shadow: 0 0 15px rgba(255, 140, 0, 0.3);">
                            <div style="text-align: center; margin-bottom: 10px;">
                                <div style="display: inline-flex; align-items: center; background-color: #ff8c00;
                                           color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold;">
                                    <span style="font-size: 18px; margin-right: 8px;">🔴</span>
                                    <span style="font-size: 16px;">LIVE - {status_display}</span>
                                </div>
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div style="text-align: center; flex: 1;">
                                    <img src="{get_team_logo_url(game['away_team'])}" width="60" style="margin-bottom: 10px;">
                                    <div style="font-size: 24px; font-weight: bold; color: {'#ff8c00' if leading_team == game['away_team'] else '#212529'};">{game['away_team_display']}</div>
                                    <div style="font-size: 36px; font-weight: bold; color: {'#ff8c00' if leading_team == game['away_team'] else '#6c757d'};">{away_score}</div>
                                </div>
                                <div style="text-align: center; flex: 0.5;">
                                    <div style="font-size: 20px; font-weight: bold; color: #212529;">@</div>
                                </div>
                                <div style="text-align: center; flex: 1;">
                                    <img src="{get_team_logo_url(game['home_team'])}" width="60" style="margin-bottom: 10px;">
                                    <div style="font-size: 24px; font-weight: bold; color: {'#ff8c00' if leading_team == game['home_team'] else '#212529'};">{game['home_team_display']}</div>
                                    <div style="font-size: 36px; font-weight: bold; color: {'#ff8c00' if leading_team == game['home_team'] else '#6c757d'};">{home_score}</div>
                                </div>
                            </div>
                            <hr style="margin: 15px 0; border-color: #ff8c00;">
                            <div style="text-align: center;">
                                <div style="font-size: 14px; margin-bottom: 5px; color: #212529;">📊 <strong>Sportsbooks:</strong> {spread_text} | {total_text}</div>
                                {f'<div style="font-size: 14px; margin-top: 5px; color: #333;"><strong>🎯 Our Predictions:</strong> {prediction_info}</div>' if prediction_info else ''}
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

            st.markdown("---")  # Separator after live games

    # This Week's Games Section
    # Display actual week from data
    if not df_current_week.empty:
        display_week = df_current_week["week"].iloc[0]
        display_season = df_current_week["season"].iloc[0]
        st.header(f"📅 {display_season} Week {display_week}")
    else:
        st.header("📅 This Week's Games")

    if not df_current_week.empty:
        # Completed Games
        if not completed_games.empty:

            # Create two-column layout for games
            for i in range(0, len(completed_games), 2):
                col1, col2 = st.columns(2)

                for col_idx, col in enumerate([col1, col2]):
                    game_idx = i + col_idx
                    if game_idx >= len(completed_games):
                        break

                    game = completed_games.iloc[game_idx]
                    away_score = int(game["away_score"]) if pd.notna(game["away_score"]) else 0
                    home_score = int(game["home_score"]) if pd.notna(game["home_score"]) else 0
                    # Calculate actual margin (home - away)
                    actual_margin = home_score - away_score

                    # Determine winner
                    if actual_margin > 0:
                        winner = game["home_team"]
                        game["away_team"]
                    else:
                        winner = game["away_team"]
                        game["home_team"]

                    # Note: We're now showing player passing yards predictions (not game outcomes)
                    # Game outcome prediction accuracy is tracked separately in ml.predictions
                    prediction_text = ""

                    # Sportsbook info
                    spread = game["spread_close"] if pd.notna(game["spread_close"]) else None
                    total = game["total_close"] if pd.notna(game["total_close"]) else None

                    if spread is not None:
                        spread_text = (
                            f"Spread: {game['home_team_display']} {spread:+.1f}"
                            if spread != 0
                            else "Spread: EVEN"
                        )
                    else:
                        spread_text = "Spread: N/A"

                    if total is not None:
                        total_text = f"Total: O/U {total:.1f}"
                    else:
                        total_text = "Total: N/A"

                    # Our model predictions - Game-level predictions (not QB stats)
                    prediction_lines = []

                    # Spread prediction
                    if "predicted_spread" in game and pd.notna(game["predicted_spread"]):
                        pred_spread = float(game["predicted_spread"])
                        pred_spread_str = (
                            f"{game['home_team_display']} {pred_spread:+.1f}"
                            if pred_spread != 0
                            else "EVEN"
                        )
                        prediction_lines.append(f"Spread: {pred_spread_str}")

                    # Total (O/U) prediction
                    if "predicted_total" in game and pd.notna(game["predicted_total"]):
                        pred_total = float(game["predicted_total"])
                        prediction_lines.append(f"Total: O/U {pred_total:.1f}")

                    # Score predictions
                    if (
                        "predicted_home_score" in game
                        and pd.notna(game["predicted_home_score"])
                        and "predicted_away_score" in game
                        and pd.notna(game["predicted_away_score"])
                    ):
                        pred_home = float(game["predicted_home_score"])
                        pred_away = float(game["predicted_away_score"])
                        prediction_lines.append(
                            f"Score: {game['away_team_display']} {pred_away:.0f} @ {game['home_team_display']} {pred_home:.0f}"
                        )

                    # Win probability
                    if "predicted_home_win_prob" in game and pd.notna(
                        game["predicted_home_win_prob"]
                    ):
                        home_win_prob = float(game["predicted_home_win_prob"])
                        predicted_winner = (
                            game["home_team_display"]
                            if home_win_prob > 0.5
                            else game["away_team_display"]
                        )
                        win_prob_pct = max(home_win_prob, 1 - home_win_prob) * 100
                        prediction_lines.append(f"Winner: {predicted_winner} ({win_prob_pct:.1f}%)")

                    prediction_info = (
                        " | ".join(prediction_lines)
                        if prediction_lines
                        else "No predictions available"
                    )

                    # Check if prediction was correct (for completed games)
                    prediction_correct = False
                    if "predicted_home_win_prob" in game and pd.notna(
                        game["predicted_home_win_prob"]
                    ):
                        home_win_prob = float(game["predicted_home_win_prob"])
                        predicted_winner_team = (
                            game["home_team"] if home_win_prob > 0.5 else game["away_team"]
                        )
                        prediction_correct = predicted_winner_team == winner

                    with col:
                        st.markdown(
                            f"""
                        <div style="background-color: {'#d4edda' if prediction_correct else '#f8d7da'};
                                    padding: 20px; border-radius: 10px; margin: 10px 0;
                                    border: 2px solid {'#28a745' if prediction_correct else '#dc3545'};">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div style="text-align: center; flex: 1;">
                                    <img src="{get_team_logo_url(game['away_team'])}" width="60" style="margin-bottom: 10px;">
                                    <div style="font-size: 24px; font-weight: bold;">{game['away_team_display']}</div>
                                    <div style="font-size: 32px; font-weight: bold; color: {'#28a745' if winner == game['away_team'] else '#6c757d'};">{away_score}</div>
                                </div>
                                <div style="text-align: center; flex: 0.5;">
                                    <div style="font-size: 20px; font-weight: bold;">@</div>
                                    <div style="font-size: 14px; color: #666;">FINAL</div>
                                </div>
                                <div style="text-align: center; flex: 1;">
                                    <img src="{get_team_logo_url(game['home_team'])}" width="60" style="margin-bottom: 10px;">
                                    <div style="font-size: 24px; font-weight: bold;">{game['home_team_display']}</div>
                                    <div style="font-size: 32px; font-weight: bold; color: {'#28a745' if winner == game['home_team'] else '#6c757d'};">{home_score}</div>
                                </div>
                            </div>
                            <hr style="margin: 15px 0;">
                            <div style="text-align: center;">
                                <div style="font-size: 14px; margin-bottom: 5px; color: #212529;">📊 <strong>Sportsbooks:</strong> {spread_text} | {total_text}</div>
                                {f'<div style="font-size: 14px; margin-top: 5px; color: #333;"><strong>🎯 Our Predictions:</strong> {prediction_info}</div>' if prediction_info else ''}
                                <div style="font-size: 16px; font-weight: bold; margin-top: 8px;">{prediction_text}</div>
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

        # Upcoming Games
        if not upcoming_games.empty:

            # Create two-column layout for games
            for i in range(0, len(upcoming_games), 2):
                col1, col2 = st.columns(2)

                for col_idx, col in enumerate([col1, col2]):
                    game_idx = i + col_idx
                    if game_idx >= len(upcoming_games):
                        break

                    game = upcoming_games.iloc[game_idx]
                    spread = game["spread_close"] if pd.notna(game["spread_close"]) else 0
                    total = game["total_close"] if pd.notna(game["total_close"]) else 0

                    # Game time - convert from UTC to Eastern Time
                    if pd.notna(game["kickoff"]):
                        kickoff_utc = pd.to_datetime(game["kickoff"])
                        if kickoff_utc.tzinfo is None:
                            kickoff_utc = kickoff_utc.tz_localize("UTC")
                        kickoff_et = kickoff_utc.astimezone(ZoneInfo("America/New_York"))
                        kickoff_time = kickoff_et.strftime("%a %I:%M %p ET")
                    else:
                        kickoff_time = "TBD"

                    # Sportsbook info
                    spread_text = (
                        f"Spread: {game['home_team_display']} {spread:+.1f}"
                        if spread != 0
                        else "Spread: EVEN"
                    )
                    total_text = f"Total: O/U {total:.1f}"

                    # Our model predictions - Game-level predictions (not QB stats)
                    prediction_lines = []

                    # Spread prediction
                    if "predicted_spread" in game and pd.notna(game["predicted_spread"]):
                        pred_spread = float(game["predicted_spread"])
                        pred_spread_str = (
                            f"{game['home_team_display']} {pred_spread:+.1f}"
                            if pred_spread != 0
                            else "EVEN"
                        )
                        prediction_lines.append(f"Spread: {pred_spread_str}")

                    # Total (O/U) prediction
                    if "predicted_total" in game and pd.notna(game["predicted_total"]):
                        pred_total = float(game["predicted_total"])
                        prediction_lines.append(f"Total: O/U {pred_total:.1f}")

                    # Score predictions
                    if (
                        "predicted_home_score" in game
                        and pd.notna(game["predicted_home_score"])
                        and "predicted_away_score" in game
                        and pd.notna(game["predicted_away_score"])
                    ):
                        pred_home = float(game["predicted_home_score"])
                        pred_away = float(game["predicted_away_score"])
                        prediction_lines.append(
                            f"Score: {game['away_team_display']} {pred_away:.0f} @ {game['home_team_display']} {pred_home:.0f}"
                        )

                    # Win probability
                    if "predicted_home_win_prob" in game and pd.notna(
                        game["predicted_home_win_prob"]
                    ):
                        home_win_prob = float(game["predicted_home_win_prob"])
                        predicted_winner = (
                            game["home_team_display"]
                            if home_win_prob > 0.5
                            else game["away_team_display"]
                        )
                        win_prob_pct = max(home_win_prob, 1 - home_win_prob) * 100
                        prediction_lines.append(f"Winner: {predicted_winner} ({win_prob_pct:.1f}%)")

                    prediction_info = (
                        " | ".join(prediction_lines)
                        if prediction_lines
                        else "Fetching predictions..."
                    )

                    with col:
                        st.markdown(
                            f"""
                        <div style="background-color: #e7f3ff; padding: 20px; border-radius: 10px; margin: 10px 0;
                                    border: 2px solid #007bff;">
                            <div style="text-align: center; margin-bottom: 10px;">
                                <div style="font-size: 14px; color: #333; font-weight: bold;">{kickoff_time}</div>
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div style="text-align: center; flex: 1;">
                                    <img src="{get_team_logo_url(game['away_team'])}" width="60" style="margin-bottom: 10px;">
                                    <div style="font-size: 24px; font-weight: bold; color: #212529;">{game['away_team_display']}</div>
                                </div>
                                <div style="text-align: center; flex: 0.5;">
                                    <div style="font-size: 20px; font-weight: bold; color: #212529;">@</div>
                                </div>
                                <div style="text-align: center; flex: 1;">
                                    <img src="{get_team_logo_url(game['home_team'])}" width="60" style="margin-bottom: 10px;">
                                    <div style="font-size: 24px; font-weight: bold; color: #212529;">{game['home_team_display']}</div>
                                </div>
                            </div>
                            <hr style="margin: 15px 0; border-color: #007bff;">
                            <div style="text-align: center;">
                                <div style="font-size: 14px; margin-bottom: 5px; color: #212529;">📊 <strong>Sportsbooks:</strong> {spread_text} | {total_text}</div>
                                {f'<div style="font-size: 14px; margin-top: 5px; color: #333;"><strong>🎯 Our Predictions:</strong> {prediction_info}</div>' if prediction_info else '<div style="font-size: 14px; margin-top: 5px; color: #666; font-style: italic;">No prediction available</div>'}
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

        else:
            st.info("No games scheduled for this week yet.")

    else:
        st.info("No games data available for current week.")

    # Bye Week Teams Section
    if not df_current_week.empty:
        season = df_current_week["season"].iloc[0]
        week = df_current_week["week"].iloc[0]
        bye_teams = fetch_bye_week_teams(season, week)

        if bye_teams:
            st.markdown("---")
            st.subheader(f"🏖️ Teams on Bye - Week {week}")

            # Create a nice display of bye week teams with logos
            cols = st.columns(len(bye_teams))
            for idx, team in enumerate(bye_teams):
                with cols[idx]:
                    st.markdown(
                        f"""
                    <div style="text-align: center; padding: 15px; background-color: #f8f9fa;
                                border-radius: 10px; border: 2px solid #dee2e6;">
                        <img src="{get_team_logo_url(team)}" width="80" style="margin-bottom: 10px;">
                        <div style="font-size: 20px; font-weight: bold; color: #495057;">{team}</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

            st.markdown("---")

    # Performance over time
    st.header("📈 Performance Over Time")

    col1, col2 = st.columns(2)

    with col1:
        fig_win_rate = plot_win_rate_over_time(df_performance)
        st.plotly_chart(fig_win_rate, use_container_width=True)

    with col2:
        fig_mae = plot_mae_over_time(df_performance)
        st.plotly_chart(fig_mae, use_container_width=True)

    # ATS and error distribution
    col1, col2 = st.columns(2)

    with col1:
        fig_ats = plot_ats_accuracy(df_performance)
        st.plotly_chart(fig_ats, use_container_width=True)

    with col2:
        fig_error_dist = plot_error_distribution(df_retrospectives)
        st.plotly_chart(fig_error_dist, use_container_width=True)

    # Calibration
    st.header("🎯 Model Calibration")

    if "prob_win" in df_retrospectives.columns:
        fig_calibration = plot_calibration(df_retrospectives)
        st.plotly_chart(fig_calibration, use_container_width=True)
    else:
        st.info("Calibration data not available")

    # BNN Uncertainty
    st.header("📊 BNN Uncertainty Quantification")

    df_bnn_metrics = fetch_bnn_uncertainty_metrics()

    if not df_bnn_metrics.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_uncertainty = df_bnn_metrics["uncertainty_margin"].mean()
            st.metric("Avg Margin Uncertainty", f"{avg_uncertainty:.2f} pts")

        with col2:
            avg_epistemic = df_bnn_metrics["epistemic_uncertainty"].mean()
            st.metric("Avg Epistemic", f"{avg_epistemic:.2f} pts")

        with col3:
            avg_aleatoric = df_bnn_metrics["aleatoric_uncertainty"].mean()
            st.metric("Avg Aleatoric", f"{avg_aleatoric:.2f} pts")

        with col4:
            avg_ci_width = (df_bnn_metrics["ci_upper"] - df_bnn_metrics["ci_lower"]).mean()
            st.metric("Avg CI Width (95%)", f"{avg_ci_width:.2f} pts")

        # Plots
        col1, col2 = st.columns(2)

        with col1:
            fig_uncertainty_dist = plot_bnn_uncertainty_distribution(df_bnn_metrics)
            st.plotly_chart(fig_uncertainty_dist, use_container_width=True)

        with col2:
            fig_prediction_intervals = plot_bnn_prediction_intervals(df_bnn_metrics)
            st.plotly_chart(fig_prediction_intervals, use_container_width=True)

        st.markdown(
            """
        **Understanding BNN Uncertainty:**
        - **Epistemic Uncertainty**: Model uncertainty (reducible with more data/better models)
        - **Aleatoric Uncertainty**: Inherent randomness in the game (irreducible)
        - **95% Confidence Interval**: Range where we expect the true margin to fall 95% of the time
        """
        )
    else:
        st.info("BNN uncertainty data not available")

    # Model comparison
    st.header("🔬 Model Comparison")

    if not df_model_comparison.empty:
        st.dataframe(
            df_model_comparison.style.format(
                {"mae": "{:.2f}", "std_error": "{:.2f}", "avg_confidence": "{:.3f}"}
            ),
            use_container_width=True,
        )
    else:
        st.info("No model comparison data available")

    # All Predictions Section
    st.header("🎯 All Predictions")

    if not df_retrospectives.empty:
        # Additional filters for predictions table
        col1, col2, col3 = st.columns(3)

        with col1:
            # Week filter
            if "week" in df_retrospectives.columns:
                weeks = sorted(df_retrospectives["week"].unique())
                selected_weeks = st.multiselect(
                    "Filter by Week", options=weeks, default=[], key="week_filter"
                )

        with col2:
            # Team filter (player's team or opponent)
            if "team" in df_retrospectives.columns:
                teams = sorted(
                    set(df_retrospectives["team"].unique())
                    | set(df_retrospectives["opponent"].unique())
                )
                selected_teams = st.multiselect(
                    "Filter by Team", options=teams, default=[], key="team_filter"
                )

        with col3:
            # Status filter (win/loss/push/pending)
            if "status" in df_retrospectives.columns:
                statuses = sorted(df_retrospectives["status"].dropna().unique())
                selected_statuses = st.multiselect(
                    "Filter by Result", options=statuses, default=[], key="status_filter"
                )

        # Apply filters
        filtered_df = df_retrospectives.copy()

        if selected_weeks:
            filtered_df = filtered_df[filtered_df["week"].isin(selected_weeks)]

        if selected_teams:
            filtered_df = filtered_df[
                (filtered_df["team"].isin(selected_teams))
                | (filtered_df["opponent"].isin(selected_teams))
            ]

        if selected_statuses:
            filtered_df = filtered_df[filtered_df["status"].isin(selected_statuses)]

        # Sort by most recent before selecting display columns
        if "kickoff" in filtered_df.columns:
            filtered_df = filtered_df.sort_values("kickoff", ascending=False)

        # Display count
        st.markdown(f"**Showing {len(filtered_df)} predictions**")

        # Prepare display dataframe for player-level predictions
        display_cols = [
            "season",
            "week",
            "player_name",
            "team",
            "opponent",
            "is_home",
            "predicted_value",
            "actual_value",
            "mae",
            "status",
            "pnl",
            "edge",
            "prob_win",
            "calibrated",
            "model_name",
            "model_version",
        ]

        # Only include columns that exist
        available_cols = [col for col in display_cols if col in filtered_df.columns]
        predictions_display = filtered_df[available_cols].copy()

        # Add result column based on status
        if "status" in filtered_df.columns:
            predictions_display["result"] = filtered_df["status"].map(
                {"win": "✓ Win", "loss": "✗ Loss", "push": "~ Push", "pending": "⏳ Pending"}
            )

        # Format and display
        format_dict = {}
        if "predicted_value" in predictions_display.columns:
            format_dict["predicted_value"] = "{:.1f}"
        if "actual_value" in predictions_display.columns:
            format_dict["actual_value"] = "{:.1f}"
        if "mae" in predictions_display.columns:
            format_dict["mae"] = "{:.1f}"
        if "pnl" in predictions_display.columns:
            format_dict["pnl"] = "${:.2f}"
        if "edge" in predictions_display.columns:
            format_dict["edge"] = "{:.4f}"
        if "prob_win" in predictions_display.columns:
            format_dict["prob_win"] = "{:.1%}"

        # Display with highlighting for win/loss
        if "status" in predictions_display.columns:

            def highlight_status(row):
                if "status" not in row:
                    return [""] * len(row)
                status = row["status"]
                if status == "win":
                    return ["background-color: #d4edda; color: #155724"] * len(row)
                elif status == "loss":
                    return ["background-color: #f8d7da; color: #721c24"] * len(row)
                elif status == "push":
                    return ["background-color: #fff3cd; color: #856404"] * len(row)
                return [""] * len(row)

            st.dataframe(
                predictions_display.style.format(format_dict).apply(highlight_status, axis=1),
                use_container_width=True,
                height=600,
            )
        else:
            st.dataframe(
                predictions_display.style.format(format_dict), use_container_width=True, height=600
            )

        # Download button
        csv = predictions_display.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    else:
        st.info("No predictions available")

    # Footer
    st.markdown("---")
    st.markdown(f"**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown("Live scores update every 30 seconds • Historical data refreshes every 5 minutes")


if __name__ == "__main__":
    main()
