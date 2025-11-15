#!/usr/bin/env python3
"""
NFL Analytics Monitoring Dashboard

Real-time monitoring of prediction performance, CLV, and model metrics.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import psycopg
from datetime import datetime, timedelta
import numpy as np
import os
import logging
from zoneinfo import ZoneInfo
import requests
import time
import hashlib
import sys
from pathlib import Path

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

# Database configuration from environment variables
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5544")),
    "dbname": os.getenv("DB_NAME", "devdb01"),
    "user": os.getenv("DB_USER", "dro"),
    "password": os.getenv("DB_PASSWORD")  # Required: Set DB_PASSWORD environment variable
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
                        if isinstance(cached, dict) and 'type' in cached:
                            if cached['type'] == 'dataframe':
                                return pd.DataFrame(cached['data'])
                            elif cached['type'] == 'list':
                                return cached['data']
                            elif cached['type'] == 'dict':
                                return cached['data']
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
                        cache_data_value = {
                            'type': 'dataframe',
                            'data': result.to_dict('records')
                        }
                    elif isinstance(result, list):
                        cache_data_value = {
                            'type': 'list',
                            'data': result
                        }
                    elif isinstance(result, dict):
                        cache_data_value = {
                            'type': 'dict',
                            'data': result
                        }
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
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
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
""", unsafe_allow_html=True)

@cache_data(ttl=300)  # Redis cache for 5 minutes
def get_database_connection():
    """Get database connection."""
    return psycopg.connect(**DB_CONFIG)

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
        mr.model_version,
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
        query += " AND mr.model_version = %s"
        params.append(model_version)
    if season:
        query += " AND p.season = %s"
        params.append(season)

    query += " ORDER BY g.kickoff DESC LIMIT 1000"

    with psycopg.connect(**DB_CONFIG) as conn:
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

    with psycopg.connect(**DB_CONFIG) as conn:
        df = pd.read_sql_query(query, conn)

    return df

@cache_data(ttl=300)  # Redis cache for 5 minutes
def fetch_model_comparison():
    """Compare different model versions for player passing yards predictions."""
    query = """
    SELECT
        mr.model_name,
        mr.model_version,
        mr.status as model_status,
        COUNT(p.prediction_id) as total_predictions,
        ROUND(AVG(p.abs_prediction_error)::numeric, 2) as mae,
        ROUND(STDDEV(p.prediction_error)::numeric, 2) as std_error,
        ROUND(AVG(CASE WHEN p.status = 'win' THEN 1.0 ELSE 0.0 END)::numeric, 3) as win_rate,
        ROUND(AVG(CASE WHEN p.calibrated THEN 1.0 ELSE 0.0 END)::numeric, 3) as calibration_rate,
        ROUND(SUM(p.pnl)::numeric, 2) as total_pnl,
        ROUND(AVG(p.edge)::numeric, 4) as avg_edge
    FROM ml.model_registry mr
    LEFT JOIN ml.predictions p ON mr.model_id = p.model_id AND p.stat_type = 'passing_yards' AND p.actual_value IS NOT NULL
    WHERE mr.status IN ('production', 'staging', 'active')
    GROUP BY mr.model_id, mr.model_name, mr.model_version, mr.status
    ORDER BY mae ASC NULLS LAST
    """

    with psycopg.connect(**DB_CONFIG) as conn:
        df = pd.read_sql_query(query, conn)

    return df

@cache_data(ttl=30)  # Redis cache for 30 seconds (live data)
def fetch_live_scores(season, week):
    """Fetch live scores from ESPN API."""
    try:
        url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        params = {
            'limit': 100
        }

        # ESPN to database team abbreviation mapping
        # Note: Database now uses LAR for Rams (not LA) after migration 018
        espn_to_db_teams = {
            'LAR': 'LAR',  # LA Rams (database uses LAR)
            'LA': 'LAR',   # Some APIs still use LA for Rams
            'LAC': 'LAC',  # LA Chargers (stays same)
            'WSH': 'WAS',  # Washington (some APIs use WSH, we use WAS)
            'WAS': 'WAS'   # Database standard
        }

        # For current week, just get current scoreboard (no week/season params needed)
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        live_scores = {}

        if 'events' in data:
            for event in data['events']:
                try:
                    competitions = event.get('competitions', [])
                    if not competitions:
                        continue

                    competition = competitions[0]
                    competitors = competition.get('competitors', [])

                    # Get teams
                    home_team = None
                    away_team = None
                    home_score = 0
                    away_score = 0

                    for comp in competitors:
                        team_abbr = comp.get('team', {}).get('abbreviation', '')
                        # Map ESPN team abbr to our database team abbr
                        team_abbr = espn_to_db_teams.get(team_abbr, team_abbr)
                        score = int(comp.get('score', 0))

                        if comp.get('homeAway') == 'home':
                            home_team = team_abbr
                            home_score = score
                        elif comp.get('homeAway') == 'away':
                            away_team = team_abbr
                            away_score = score

                    # Get status
                    status = event.get('status', {})
                    status_type = status.get('type', {}).get('state', 'pre')

                    # Get clock and period
                    clock = status.get('displayClock', '')
                    period = status.get('period', 0)
                    detail = status.get('type', {}).get('detail', '')

                    # Create game key (away_home format like our game_id)
                    if home_team and away_team:
                        game_key = f"{away_team}_{home_team}"

                        live_scores[game_key] = {
                            'away_team': away_team,
                            'home_team': home_team,
                            'away_score': away_score,
                            'home_score': home_score,
                            'status': status_type,  # 'pre', 'in', 'post'
                            'clock': clock,
                            'period': period,
                            'detail': detail
                        }
                except Exception as e:
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

    with psycopg.connect(**DB_CONFIG) as conn:
        df = pd.read_sql_query(query, conn, params=(season, season, season, week, season, week))

    return df['team'].tolist() if not df.empty else []

@cache_data(ttl=300)  # Redis cache for 5 minutes
def fetch_bnn_uncertainty_metrics():
    """Fetch BNN uncertainty metrics for all predictions."""
    # Return empty DataFrame - predictions.bnn_predictions table doesn't exist yet
    return pd.DataFrame(columns=[
        'game_id', 'predicted_margin', 'uncertainty_margin', 'uncertainty_total',
        'ci_lower', 'ci_upper', 'epistemic_uncertainty', 'aleatoric_uncertainty'
    ])

@cache_data(ttl=30)  # Redis cache for 30 seconds (live updates)
def fetch_current_week_games():
    """Fetch current week's games with predictions and outcomes."""
    # Use most recent completed week for demonstration (2024 Week 12)
    # In production, this would query current week (2025 Week 10+)
    query = """
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
        CASE
            WHEN g.home_score IS NOT NULL THEN 'completed'
            WHEN g.kickoff < NOW() THEN 'in_progress'
            ELSE 'scheduled'
        END as game_status
    FROM games g
    WHERE g.season = 2024 AND g.week = 12 AND g.game_type = 'REG'
    ORDER BY g.kickoff DESC
    LIMIT 10
    """

    with psycopg.connect(**DB_CONFIG) as conn:
        df = pd.read_sql_query(query, conn)

    # Fetch live scores and predictions
    if not df.empty:
        # Add display columns (LA -> LAR for clarity)
        df['away_team_display'] = df['away_team'].apply(normalize_team_display)
        df['home_team_display'] = df['home_team'].apply(normalize_team_display)

        season = df['season'].iloc[0]
        week = df['week'].iloc[0]
        live_scores = fetch_live_scores(season, week)

        # Merge live scores into dataframe
        for idx, row in df.iterrows():
            # Use display team codes for matching with live scores
            game_key = f"{row['away_team_display']}_{row['home_team_display']}"
            if game_key in live_scores:
                live_data = live_scores[game_key]

                # Update status based on ESPN data
                if live_data['status'] == 'in':
                    df.at[idx, 'game_status'] = 'in_progress'
                    df.at[idx, 'away_score'] = live_data['away_score']
                    df.at[idx, 'home_score'] = live_data['home_score']
                    df.at[idx, 'clock'] = live_data['clock']
                    df.at[idx, 'period'] = live_data['period']
                    df.at[idx, 'detail'] = live_data['detail']
                elif live_data['status'] == 'post':
                    df.at[idx, 'game_status'] = 'completed'
                    df.at[idx, 'away_score'] = live_data['away_score']
                    df.at[idx, 'home_score'] = live_data['home_score']

            # Fetch predictions from FastAPI
            predictions = get_game_predictions(row['game_id'], season, week)
            if predictions:
                df.at[idx, 'predicted_home_qb_yards'] = predictions.get('home_qb_yards')
                df.at[idx, 'predicted_home_qb_std'] = predictions.get('home_qb_yards_std')
                df.at[idx, 'predicted_away_qb_yards'] = predictions.get('away_qb_yards')
                df.at[idx, 'predicted_away_qb_std'] = predictions.get('away_qb_yards_std')
                df.at[idx, 'predicted_total_yards'] = predictions.get('total_yards')
                df.at[idx, 'predicted_home_win_prob'] = predictions.get('home_win_prob')

    return df

def normalize_team_display(team_abbr):
    """Normalize team abbreviation for display (LA -> LAR for clarity)."""
    if team_abbr == 'LA':
        return 'LAR'  # Display as LAR to distinguish from LAC
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
    try:
        api_url = os.getenv("FASTAPI_URL", "http://localhost:8000")
        response = requests.post(
            f"{api_url}/predict/game",
            json={"game_id": game_id, "season": season, "week": week},
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()

            # Extract predictions
            home_yards = data['home_prediction']['predicted_passing_yards']
            home_std = data['home_prediction']['predicted_passing_yards_std']
            away_yards = data['away_prediction']['predicted_passing_yards']
            away_std = data['away_prediction']['predicted_passing_yards_std']
            total_yards = data.get('total_predicted_yards')

            # Estimate win probability based on QB passing yards advantage
            # Simple heuristic: team with higher QB yards has higher win prob
            # More sophisticated version would use game outcome models
            if home_yards and away_yards:
                yards_diff = home_yards - away_yards
                # Convert yards difference to win probability (sigmoid-like)
                # Each 50 yards ≈ 10% win prob advantage
                home_win_prob = 0.5 + (yards_diff / 500.0)
                home_win_prob = max(0.1, min(0.9, home_win_prob))  # Clamp to [0.1, 0.9]
            else:
                home_win_prob = 0.5

            return {
                'home_qb_yards': home_yards,
                'home_qb_yards_std': home_std,
                'away_qb_yards': away_yards,
                'away_qb_yards_std': away_std,
                'total_yards': total_yards,
                'home_win_prob': home_win_prob,
                'inference_time_ms': data.get('processing_ms', 0)
            }
        else:
            logger.warning(f"FastAPI returned {response.status_code} for {game_id}")
            return None

    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to fetch predictions from FastAPI: {e}")
        return None
    except Exception as e:
        logger.error(f"Error parsing prediction response: {e}")
        return None

def get_team_logo_url(team_abbr):
    """Get team logo URL from nflverse."""
    # Using ESPN's logo service which is reliable
    # Always use LAR for Rams logo (ESPN uses LAR)
    display_abbr = normalize_team_display(team_abbr)
    return f"https://a.espncdn.com/i/teamlogos/nfl/500/{display_abbr}.png"

def plot_win_rate_over_time(df):
    """Plot win rate over time."""
    df['week_label'] = df['season'].astype(str) + '-W' + df['week'].astype(str)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['week_label'],
        y=df['win_rate'] * 100,
        mode='lines+markers',
        name='Win Rate',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))

    # Add 50% reference line
    fig.add_hline(y=50, line_dash="dash", line_color="gray",
                  annotation_text="50% (Random)", annotation_position="right")

    # Add breakeven line (52.4% for -110 odds)
    fig.add_hline(y=52.4, line_dash="dash", line_color="red",
                  annotation_text="52.4% (Breakeven)", annotation_position="right")

    fig.update_layout(
        title="Win Rate Over Time",
        xaxis_title="Season-Week",
        yaxis_title="Win Rate (%)",
        hovermode='x unified',
        height=400
    )

    return fig

def plot_mae_over_time(df):
    """Plot Mean Absolute Error over time."""
    df['week_label'] = df['season'].astype(str) + '-W' + df['week'].astype(str)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['week_label'],
        y=df['mae'],
        mode='lines+markers',
        name='MAE',
        line=dict(color='#ff7f0e', width=3),
        marker=dict(size=8),
        fill='tozeroy',
        fillcolor='rgba(255, 127, 14, 0.2)'
    ))

    fig.update_layout(
        title="Mean Absolute Error Over Time (Passing Yards)",
        xaxis_title="Season-Week",
        yaxis_title="MAE (yards)",
        hovermode='x unified',
        height=400
    )

    return fig

def plot_ats_accuracy(df):
    """Plot calibration rate over time."""
    df['week_label'] = df['season'].astype(str) + '-W' + df['week'].astype(str)

    fig = go.Figure()

    # Check if calibration_rate column exists
    if 'calibration_rate' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['week_label'],
            y=df['calibration_rate'] * 100,
            mode='lines+markers',
            name='Calibration Rate',
            line=dict(color='#2ca02c', width=3),
            marker=dict(size=8)
        ))

        # Add 90% target line
        fig.add_hline(y=90, line_dash="dash", line_color="green",
                      annotation_text="90% (Target)", annotation_position="right")

        fig.update_layout(
            title="Calibration Rate Over Time",
            xaxis_title="Season-Week",
            yaxis_title="Calibration Rate (%)",
            hovermode='x unified',
            height=400
        )
    else:
        fig.update_layout(
            title="Calibration Rate Over Time (No Data)",
            xaxis_title="Season-Week",
            yaxis_title="Calibration Rate (%)",
            height=400
        )

    return fig

def plot_error_distribution(df):
    """Plot distribution of prediction errors for passing yards."""
    fig = go.Figure()

    # Use mae column which is abs_prediction_error from ml.predictions
    # Calculate signed error if actual_value and predicted_value exist
    if 'actual_value' in df.columns and 'predicted_value' in df.columns:
        error_col = df['actual_value'] - df['predicted_value']
        xlabel = "Prediction Error (yards)"
    elif 'mae' in df.columns:
        # If only MAE available, we can't show signed errors
        # Skip this plot or show absolute errors
        error_col = df['mae']
        xlabel = "Absolute Error (yards)"
    else:
        # No error data available
        fig.update_layout(
            title="Prediction Error Distribution (No Data)",
            xaxis_title="Error (yards)",
            yaxis_title="Count",
            height=400,
            showlegend=False
        )
        return fig

    fig.add_trace(go.Histogram(
        x=error_col,
        nbinsx=40,
        name='Error Distribution',
        marker_color='#9467bd',
        opacity=0.7
    ))

    fig.add_vline(x=0, line_dash="dash", line_color="black",
                  annotation_text="Perfect Prediction", annotation_position="top")

    fig.update_layout(
        title="Prediction Error Distribution (Passing Yards)",
        xaxis_title=xlabel,
        yaxis_title="Count",
        height=400,
        showlegend=False
    )

    return fig

def plot_calibration(df):
    """Plot calibration curve for player props (over/under)."""
    # For player props, we need to check if predictions fall within confidence intervals
    # This is different from game outcome probability calibration

    if 'prob_win' not in df.columns or 'status' not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            title="Calibration Curve (No Data)",
            xaxis_title="Predicted Probability",
            yaxis_title="Actual Win Rate",
            height=400,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        return fig

    # Filter valid probabilities
    valid_df = df[df['prob_win'].notna() & df['status'].notna()].copy()

    if len(valid_df) == 0:
        fig = go.Figure()
        fig.update_layout(
            title="Calibration Curve (No Data)",
            xaxis_title="Predicted Probability",
            yaxis_title="Actual Win Rate",
            height=400,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        return fig

    # Bin predictions by probability
    valid_df['prob_bin'] = pd.cut(valid_df['prob_win'], bins=10, labels=False)

    calibration = valid_df.groupby('prob_bin').agg({
        'prob_win': 'mean',
        'status': lambda x: (x == 'win').mean()
    }).reset_index()

    calibration.columns = ['bin', 'predicted_prob', 'actual_freq']

    fig = go.Figure()

    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='gray', dash='dash')
    ))

    # Actual calibration
    fig.add_trace(go.Scatter(
        x=calibration['predicted_prob'],
        y=calibration['actual_freq'],
        mode='markers+lines',
        name='Model Calibration',
        marker=dict(size=12, color='#d62728'),
        line=dict(width=2, color='#d62728')
    ))

    fig.update_layout(
        title="Calibration Curve (Player Props)",
        xaxis_title="Predicted Win Probability",
        yaxis_title="Actual Win Rate",
        height=400,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )

    return fig

def plot_bnn_uncertainty_distribution(df_bnn):
    """Plot BNN uncertainty distribution."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Epistemic Uncertainty", "Aleatoric Uncertainty")
    )

    # Epistemic uncertainty histogram
    fig.add_trace(go.Histogram(
        x=df_bnn['epistemic_uncertainty'],
        name='Epistemic',
        marker_color='#1f77b4',
        opacity=0.7,
        nbinsx=30
    ), row=1, col=1)

    # Aleatoric uncertainty histogram
    fig.add_trace(go.Histogram(
        x=df_bnn['aleatoric_uncertainty'],
        name='Aleatoric',
        marker_color='#ff7f0e',
        opacity=0.7,
        nbinsx=30
    ), row=1, col=2)

    fig.update_layout(
        title="BNN Uncertainty Decomposition",
        showlegend=False,
        height=400
    )

    fig.update_xaxes(title_text="Epistemic Uncertainty (points)", row=1, col=1)
    fig.update_xaxes(title_text="Aleatoric Uncertainty (points)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)

    return fig

def plot_bnn_prediction_intervals(df_bnn):
    """Plot BNN prediction intervals."""
    # Sort by predicted margin for better visualization
    df_sorted = df_bnn.sort_values('predicted_margin').reset_index(drop=True)
    df_sorted['game_index'] = range(len(df_sorted))

    fig = go.Figure()

    # Confidence interval
    fig.add_trace(go.Scatter(
        x=df_sorted['game_index'],
        y=df_sorted['ci_upper'],
        mode='lines',
        name='95% CI Upper',
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=df_sorted['game_index'],
        y=df_sorted['ci_lower'],
        mode='lines',
        name='95% CI',
        line=dict(width=0),
        fillcolor='rgba(31, 119, 180, 0.2)',
        fill='tonexty',
        showlegend=True
    ))

    # Predicted margin
    fig.add_trace(go.Scatter(
        x=df_sorted['game_index'],
        y=df_sorted['predicted_margin'],
        mode='markers',
        name='Predicted Margin',
        marker=dict(size=6, color='#d62728')
    ))

    fig.update_layout(
        title="BNN Prediction Intervals (95% Confidence)",
        xaxis_title="Game Index",
        yaxis_title="Predicted Margin (points)",
        hovermode='x unified',
        height=400
    )

    return fig

def check_service_health(host, port, service_name):
    """Check if a service is healthy by attempting to connect."""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        return False

def fetch_database_stats():
    """Fetch database statistics."""
    try:
        with psycopg.connect(**DB_CONFIG) as conn:
            stats = {}

            # Total games
            result = conn.execute("SELECT COUNT(*) FROM games").fetchone()
            stats['total_games'] = result[0] if result else 0

            # Total plays
            result = conn.execute("SELECT COUNT(*) FROM plays").fetchone()
            stats['total_plays'] = result[0] if result else 0

            # Season coverage
            result = conn.execute("SELECT MIN(season), MAX(season) FROM games").fetchone()
            stats['season_min'] = result[0] if result and result[0] else 0
            stats['season_max'] = result[1] if result and result[1] else 0

            # Database size
            result = conn.execute("""
                SELECT pg_size_pretty(pg_database_size('devdb01'))
            """).fetchone()
            stats['db_size'] = result[0] if result else 'N/A'

            # Recent data timestamp
            result = conn.execute("""
                SELECT MAX(kickoff) FROM games WHERE kickoff IS NOT NULL
            """).fetchone()
            stats['latest_game'] = result[0] if result and result[0] else None

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
        with psycopg.connect(**DB_CONFIG) as conn:
            df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
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
        with psycopg.connect(**DB_CONFIG) as conn:
            df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        return pd.DataFrame()

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
    for idx, (name, host, port) in enumerate(services):
        with cols[idx]:
            is_healthy = check_service_health(host, port, name)
            status_icon = "✅" if is_healthy else "❌"
            status_color = "#28a745" if is_healthy else "#dc3545"
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background-color: {'#d4edda' if is_healthy else '#f8d7da'};
                        border-radius: 10px; border: 2px solid {status_color};">
                <div style="font-size: 32px;">{status_icon}</div>
                <div style="font-size: 14px; font-weight: bold; color: #212529; margin-top: 5px;">{name}</div>
                <div style="font-size: 12px; color: #666;">:{port}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Database statistics
    st.subheader("💾 Database Statistics")
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
            st.metric("Database Size", db_stats['db_size'])

        with col5:
            if db_stats['latest_game']:
                latest = pd.to_datetime(db_stats['latest_game'])
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
        st.markdown("""
        **Current Configuration:**
        - **Core years**: 2018-2024 (full weight)
        - **Decay years**: 2015-2017 (exponential decay, H=4)
        - **Formula**: w(s; 2025, 4) = 0.5^((2025-s)/4)

        **Effective Weights:**
        - 2017: 0.297
        - 2016: 0.250
        - 2015: 0.210
        """)

    with col2:
        st.markdown("""
        **Validation Results:**
        - **Test Set**: 2025 weeks 1-6 (148 games, 37 QBs)
        - **MAE**: 47.77 yards
        - **Coverage**: 86.5%
        - **Statistical Significance**: p=0.0069 (vs baseline)
        - **Status**: ✅ Approved for production
        """)

    st.markdown("---")

    # External monitoring links
    st.subheader("🔗 Monitoring Links")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### Grafana
        [Open Grafana Dashboard →](http://localhost:3000)

        **Credentials:**
        - Username: `admin`
        - Password: See `.env` file or `GRAFANA_PASSWORD` environment variable
        """)

    with col2:
        st.markdown("""
        ### Prometheus
        [Open Prometheus →](http://localhost:9090)

        **Features:**
        - Metrics queries
        - Alert rules
        - Target health
        """)

    with col3:
        st.markdown("""
        ### Database
        [localhost:5544](postgresql://localhost:5544/devdb01)

        **Connection:**
        ```
        psql postgresql://dro:***@localhost:5544/devdb01
        ```
        """)

def render_model_status_page():
    """Render the model status page."""
    st.header("🤖 Model Status")

    # Model registry
    st.subheader("📋 Model Registry")
    df_models = fetch_model_registry()

    if not df_models.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Models", len(df_models))

        with col2:
            active_models = len(df_models[df_models['status'] == 'active'])
            st.metric("Active Models", active_models)

        with col3:
            in_production = len(df_models[df_models['status'] == 'production'])
            st.metric("In Production", in_production)

        with col4:
            archived = len(df_models[df_models['status'] == 'archived'])
            st.metric("Archived", archived)

        # Model table
        st.dataframe(
            df_models[[
                'model_id', 'model_name', 'model_version', 'model_type',
                'framework', 'status', 'updated_at', 'description'
            ]],
            use_container_width=True,
            height=400
        )
    else:
        st.info("No models found in registry. Models will appear here after training.")

    st.markdown("---")

    # Trained model files
    st.subheader("📦 Model Files")

    # Check models directory (handle both local and Docker paths)
    try:
        import os
        import glob

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
            st.warning("Models directory not found. Make sure models directory is mounted in Docker.")
            st.info("Expected paths: /app/models (Docker) or /Users/dro/rice/nfl-analytics/models (local)")
        else:
            model_files = []

            # Find all model files
            for ext in ['*.rds', '*.pkl', '*.pt', '*.pth']:
                for filepath in glob.glob(os.path.join(models_dir, '**', ext), recursive=True):
                    try:
                        stat = os.stat(filepath)
                        model_files.append({
                            'filename': os.path.relpath(filepath, models_dir),
                            'size_mb': stat.st_size / (1024 * 1024),
                            'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                        })
                    except Exception as e:
                        continue

            if model_files:
                df_files = pd.DataFrame(model_files)
                df_files = df_files.sort_values('modified', ascending=False)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Model Files", len(df_files))
                with col2:
                    total_size = df_files['size_mb'].sum()
                    st.metric("Total Size", f"{total_size:.1f} MB")
                with col3:
                    largest = df_files['size_mb'].max()
                    st.metric("Largest File", f"{largest:.1f} MB")

                st.dataframe(
                    df_files.style.format({'size_mb': '{:.2f}'}),
                    use_container_width=True,
                    height=400
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
        st.dataframe(
            df_runs,
            use_container_width=True,
            height=300
        )
    else:
        st.info("No training runs found")

    st.markdown("---")

    # Ensemble configuration
    st.subheader("🎯 4-Way Ensemble Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
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
        """)

    with col2:
        st.markdown("""
        **Combination Method:**
        - **Default**: Inverse variance weighting
        - **Advanced**: Stacked meta-learner with CV
        - **Portfolio**: Correlation-adjusted Kelly criterion

        **Expected Performance:**
        - **Target ROI**: +5-7%
        - **Win Rate**: 59-61%
        - **Sharpe Ratio**: 2.2-2.6
        - **Max Drawdown**: -8 to -10%
        """)

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
        st.metric(
            "Total Games",
            f"{db_stats['total_games']:,}",
            help="Total games in database"
        )

    with col2:
        st.metric(
            "Total Plays",
            f"{db_stats['total_plays']:,}",
            help="Total play-by-play records"
        )

    with col3:
        st.metric(
            "Season Coverage",
            f"{db_stats['season_min']}-{db_stats['season_max']}",
            help="Range of seasons in database"
        )

    st.markdown("---")

    # Schema statistics
    st.subheader("🗄️ Schema Statistics")

    try:
        with psycopg.connect(**DB_CONFIG) as conn:
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
                st.dataframe(
                    df_schemas,
                    use_container_width=True,
                    height=200
                )

            with col2:
                # Schema breakdown
                for _, row in df_schemas.iterrows():
                    st.metric(
                        f"{row['schemaname']} schema",
                        f"{row['table_count']} tables"
                    )
    except Exception as e:
        st.error(f"Error fetching schema stats: {str(e)}")

    st.markdown("---")

    # Recent data ingestion
    st.subheader("📥 Recent Data Ingestion")

    try:
        with psycopg.connect(**DB_CONFIG) as conn:
            # Get recent games by season and week
            recent_query = """
            SELECT
                season,
                week,
                game_type,
                COUNT(*) as games,
                MAX(kickoff) as latest_kickoff
            FROM games
            WHERE season >= 2024
            GROUP BY season, week, game_type
            ORDER BY season DESC, week DESC
            LIMIT 10
            """
            df_recent = pd.read_sql_query(recent_query, conn)

            st.dataframe(
                df_recent,
                use_container_width=True,
                height=300
            )
    except Exception as e:
        st.error(f"Error fetching recent data: {str(e)}")

    st.markdown("---")

    # Data quality notes
    st.subheader("✅ Data Quality")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Key Findings:**
        - ✅ All games have valid kickoff timestamps
        - ✅ Betting lines (spread, total) available for 99%+ of games
        - ✅ Play-by-play data includes EPA calculations
        - ✅ Weather data available for outdoor venues
        - ✅ Injury reports tracked by week
        """)

    with col2:
        st.markdown("""
        **Important Notes:**
        - Use `kickoff` column (not `game_date` or `gameday`)
        - Use `spread_close` and `total_close` (not `spread_line`, `total_line`)
        - Team abbreviations: LA Rams = `LAR` (not `LA`)
        - Weather stored as TEXT, requires parsing
        - See `docs/DATABASE_SCHEMA_AUDIT.md` for details
        """)

def main():
    st.title("🏈 NFL Analytics Monitoring Dashboard")
    st.markdown("Real-time monitoring of prediction performance and model metrics")

    # Sidebar navigation
    st.sidebar.header("Navigation")

    page = st.sidebar.radio(
        "Select Page",
        ["Predictions", "System Status", "Model Status", "Data Summary"],
        index=0
    )

    # Route to appropriate page
    if page == "System Status":
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
    selected_season = st.sidebar.selectbox(
        "Season",
        ["All"] + seasons,
        index=0
    )

    # Model version filter (dynamically fetch from database)
    try:
        with psycopg.connect(**DB_CONFIG) as conn:
            model_versions_df = pd.read_sql_query(
                "SELECT DISTINCT model_version FROM ml.model_registry WHERE status IN ('production', 'staging', 'active') ORDER BY model_version",
                conn
            )
            model_versions = model_versions_df['model_version'].tolist()
    except:
        model_versions = []

    selected_model = st.sidebar.selectbox(
        "Model Version",
        ["All"] + model_versions,
        index=0
    )

    # Fetch data
    with st.spinner("Loading data..."):
        season_filter = None if selected_season == "All" else selected_season
        model_filter = None if selected_model == "All" else selected_model

        df_retrospectives = fetch_retrospective_metrics(
            model_version=model_filter,
            season=season_filter
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
        if not df_retrospectives.empty and 'status' in df_retrospectives.columns:
            win_rate = (df_retrospectives['status'] == 'win').mean()
            st.metric("Win Rate", f"{win_rate*100:.1f}%",
                      delta=f"{(win_rate-0.5)*100:.1f}% vs 50%")
        else:
            st.metric("Win Rate", "N/A")

    with col3:
        if not df_retrospectives.empty and 'mae' in df_retrospectives.columns:
            mae = df_retrospectives['mae'].mean()
            st.metric("Mean Absolute Error", f"{mae:.2f} yards")
        else:
            st.metric("Mean Absolute Error", "N/A")

    with col4:
        if not df_retrospectives.empty and 'pnl' in df_retrospectives.columns:
            total_pnl = df_retrospectives['pnl'].sum()
            avg_pnl = df_retrospectives['pnl'].mean()
            st.metric("Total P&L", f"${total_pnl:,.2f}",
                     delta=f"${avg_pnl:.2f} avg")
        else:
            st.metric("Total P&L", "N/A")

    # Fetch current week games data
    df_current_week = fetch_current_week_games()

    if not df_current_week.empty:
        # Separate completed, in-progress, and upcoming games
        completed_games = df_current_week[df_current_week['game_status'] == 'completed']
        in_progress_games = df_current_week[df_current_week['game_status'] == 'in_progress']
        upcoming_games = df_current_week[df_current_week['game_status'] == 'scheduled']

        # In-Progress Games (LIVE) - Show at top
        if not in_progress_games.empty:
            # Add refresh button in top right
            col1, col2 = st.columns([3, 1])
            with col1:
                st.empty()  # Spacer
            with col2:
                if st.button("🔄 Refresh", type="primary"):
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
                    away_score = int(game['away_score']) if pd.notna(game['away_score']) else 0
                    home_score = int(game['home_score']) if pd.notna(game['home_score']) else 0

                    # Get clock and period info
                    clock = game.get('clock', '')
                    period = int(game.get('period', 0)) if pd.notna(game.get('period')) else 0
                    detail = game.get('detail', 'In Progress')

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
                        if pd.notna(game['kickoff']):
                            kickoff_utc = pd.to_datetime(game['kickoff'])
                            if kickoff_utc.tzinfo is None:
                                kickoff_utc = kickoff_utc.tz_localize('UTC')
                            kickoff_et = kickoff_utc.astimezone(ZoneInfo('America/New_York'))
                            status_display = kickoff_et.strftime('%I:%M %p ET')
                        else:
                            status_display = "TBD"

                    # Determine who's winning
                    leading_team = None
                    if away_score > home_score:
                        leading_team = game['away_team']
                    elif home_score > away_score:
                        leading_team = game['home_team']

                    # Sportsbook info
                    spread = game['spread_close'] if pd.notna(game['spread_close']) else None
                    total = game['total_close'] if pd.notna(game['total_close']) else None

                    if spread is not None:
                        spread_text = f"Spread: {game['home_team_display']} {spread:+.1f}" if spread != 0 else "Spread: EVEN"
                    else:
                        spread_text = "Spread: N/A"

                    if total is not None:
                        total_text = f"Total: O/U {total:.1f}"
                    else:
                        total_text = "Total: N/A"

                    # Our model predictions (REAL from FastAPI Ensemble v5.1)
                    prediction_lines = []

                    # QB passing yards predictions
                    if 'predicted_home_qb_yards' in game and pd.notna(game['predicted_home_qb_yards']):
                        home_qb_yards = float(game['predicted_home_qb_yards'])
                        home_qb_std = float(game.get('predicted_home_qb_std', 0))
                        prediction_lines.append(
                            f"<strong>{game['home_team_display']} QB:</strong> {home_qb_yards:.0f} ± {home_qb_std:.0f} yds"
                        )

                    if 'predicted_away_qb_yards' in game and pd.notna(game['predicted_away_qb_yards']):
                        away_qb_yards = float(game['predicted_away_qb_yards'])
                        away_qb_std = float(game.get('predicted_away_qb_std', 0))
                        prediction_lines.append(
                            f"<strong>{game['away_team_display']} QB:</strong> {away_qb_yards:.0f} ± {away_qb_std:.0f} yds"
                        )

                    # Total passing yards
                    if 'predicted_total_yards' in game and pd.notna(game['predicted_total_yards']):
                        total_pred = float(game['predicted_total_yards'])
                        prediction_lines.append(f"<strong>Total Passing:</strong> {total_pred:.0f} yds")

                    # Win probability
                    if 'predicted_home_win_prob' in game and pd.notna(game['predicted_home_win_prob']):
                        home_win_prob = float(game['predicted_home_win_prob'])
                        predicted_winner = game['home_team_display'] if home_win_prob > 0.5 else game['away_team_display']
                        win_prob_pct = max(home_win_prob, 1 - home_win_prob) * 100
                        prediction_lines.append(f"<strong>Winner:</strong> {predicted_winner} ({win_prob_pct:.1f}%)")

                    prediction_info = "<div style='font-size: 14px; color: #333;'>" + " | ".join(prediction_lines) + "</div>" if prediction_lines else "<div style='font-size: 14px; color: #666; font-style: italic;'>Fetching predictions...</div>"

                    with col:
                        st.markdown(f"""
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
                        """, unsafe_allow_html=True)

            st.markdown("---")  # Separator after live games

    # This Week's Games Section
    # Display actual week from data
    if not df_current_week.empty:
        display_week = df_current_week['week'].iloc[0]
        display_season = df_current_week['season'].iloc[0]
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
                    away_score = int(game['away_score']) if pd.notna(game['away_score']) else 0
                    home_score = int(game['home_score']) if pd.notna(game['home_score']) else 0
                    # Calculate actual margin (home - away)
                    actual_margin = home_score - away_score

                    # Determine winner
                    if actual_margin > 0:
                        winner = game['home_team']
                        winner_score = home_score
                        loser = game['away_team']
                        loser_score = away_score
                    else:
                        winner = game['away_team']
                        winner_score = away_score
                        loser = game['home_team']
                        loser_score = home_score

                    # Note: We're now showing player passing yards predictions (not game outcomes)
                    # Game outcome prediction accuracy is tracked separately in ml.predictions
                    prediction_text = ""

                    # Sportsbook info
                    spread = game['spread_close'] if pd.notna(game['spread_close']) else None
                    total = game['total_close'] if pd.notna(game['total_close']) else None

                    if spread is not None:
                        spread_text = f"Spread: {game['home_team_display']} {spread:+.1f}" if spread != 0 else "Spread: EVEN"
                    else:
                        spread_text = "Spread: N/A"

                    if total is not None:
                        total_text = f"Total: O/U {total:.1f}"
                    else:
                        total_text = "Total: N/A"

                    # Our model predictions (REAL from FastAPI Ensemble v5.1)
                    prediction_lines = []

                    # QB passing yards predictions
                    if 'predicted_home_qb_yards' in game and pd.notna(game['predicted_home_qb_yards']):
                        home_qb_yards = float(game['predicted_home_qb_yards'])
                        home_qb_std = float(game.get('predicted_home_qb_std', 0))
                        prediction_lines.append(
                            f"{game['home_team_display']} QB: {home_qb_yards:.0f} ± {home_qb_std:.0f} yds"
                        )

                    if 'predicted_away_qb_yards' in game and pd.notna(game['predicted_away_qb_yards']):
                        away_qb_yards = float(game['predicted_away_qb_yards'])
                        away_qb_std = float(game.get('predicted_away_qb_std', 0))
                        prediction_lines.append(
                            f"{game['away_team_display']} QB: {away_qb_yards:.0f} ± {away_qb_std:.0f} yds"
                        )

                    # Total passing yards
                    if 'predicted_total_yards' in game and pd.notna(game['predicted_total_yards']):
                        total_pred = float(game['predicted_total_yards'])
                        prediction_lines.append(f"Total Passing: {total_pred:.0f} yds")

                    # Win probability
                    if 'predicted_home_win_prob' in game and pd.notna(game['predicted_home_win_prob']):
                        home_win_prob = float(game['predicted_home_win_prob'])
                        predicted_winner = game['home_team_display'] if home_win_prob > 0.5 else game['away_team_display']
                        win_prob_pct = max(home_win_prob, 1 - home_win_prob) * 100
                        prediction_lines.append(f"Winner: {predicted_winner} ({win_prob_pct:.1f}%)")

                    prediction_info = " | ".join(prediction_lines) if prediction_lines else "No predictions available"

                    with col:
                        st.markdown(f"""
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
                        """, unsafe_allow_html=True)

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
                    spread = game['spread_close'] if pd.notna(game['spread_close']) else 0
                    total = game['total_close'] if pd.notna(game['total_close']) else 0

                    # Game time - convert from UTC to Eastern Time
                    if pd.notna(game['kickoff']):
                        kickoff_utc = pd.to_datetime(game['kickoff'])
                        if kickoff_utc.tzinfo is None:
                            kickoff_utc = kickoff_utc.tz_localize('UTC')
                        kickoff_et = kickoff_utc.astimezone(ZoneInfo('America/New_York'))
                        kickoff_time = kickoff_et.strftime('%a %I:%M %p ET')
                    else:
                        kickoff_time = "TBD"

                    # Sportsbook info
                    spread_text = f"Spread: {game['home_team_display']} {spread:+.1f}" if spread != 0 else "Spread: EVEN"
                    total_text = f"Total: O/U {total:.1f}"

                    # Our model predictions (REAL from FastAPI Ensemble v5.1)
                    prediction_lines = []

                    # QB passing yards predictions
                    if 'predicted_home_qb_yards' in game and pd.notna(game['predicted_home_qb_yards']):
                        home_qb_yards = float(game['predicted_home_qb_yards'])
                        home_qb_std = float(game.get('predicted_home_qb_std', 0))
                        prediction_lines.append(
                            f"{game['home_team_display']} QB: {home_qb_yards:.0f} ± {home_qb_std:.0f} yds"
                        )

                    if 'predicted_away_qb_yards' in game and pd.notna(game['predicted_away_qb_yards']):
                        away_qb_yards = float(game['predicted_away_qb_yards'])
                        away_qb_std = float(game.get('predicted_away_qb_std', 0))
                        prediction_lines.append(
                            f"{game['away_team_display']} QB: {away_qb_yards:.0f} ± {away_qb_std:.0f} yds"
                        )

                    # Total passing yards
                    if 'predicted_total_yards' in game and pd.notna(game['predicted_total_yards']):
                        total_pred = float(game['predicted_total_yards'])
                        prediction_lines.append(f"Total Passing: {total_pred:.0f} yds")

                    # Win probability
                    if 'predicted_home_win_prob' in game and pd.notna(game['predicted_home_win_prob']):
                        home_win_prob = float(game['predicted_home_win_prob'])
                        predicted_winner = game['home_team_display'] if home_win_prob > 0.5 else game['away_team_display']
                        win_prob_pct = max(home_win_prob, 1 - home_win_prob) * 100
                        prediction_lines.append(f"Winner: {predicted_winner} ({win_prob_pct:.1f}%)")

                    prediction_info = " | ".join(prediction_lines) if prediction_lines else "Fetching predictions..."

                    with col:
                        st.markdown(f"""
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
                        """, unsafe_allow_html=True)

        else:
            st.info("No games scheduled for this week yet.")

    else:
        st.info("No games data available for current week.")

    # Bye Week Teams Section
    if not df_current_week.empty:
        season = df_current_week['season'].iloc[0]
        week = df_current_week['week'].iloc[0]
        bye_teams = fetch_bye_week_teams(season, week)

        if bye_teams:
            st.markdown("---")
            st.subheader(f"🏖️ Teams on Bye - Week {week}")

            # Create a nice display of bye week teams with logos
            cols = st.columns(len(bye_teams))
            for idx, team in enumerate(bye_teams):
                with cols[idx]:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 15px; background-color: #f8f9fa;
                                border-radius: 10px; border: 2px solid #dee2e6;">
                        <img src="{get_team_logo_url(team)}" width="80" style="margin-bottom: 10px;">
                        <div style="font-size: 20px; font-weight: bold; color: #495057;">{team}</div>
                    </div>
                    """, unsafe_allow_html=True)

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

    if 'home_win_prob' in df_retrospectives.columns:
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
            avg_uncertainty = df_bnn_metrics['uncertainty_margin'].mean()
            st.metric("Avg Margin Uncertainty", f"{avg_uncertainty:.2f} pts")

        with col2:
            avg_epistemic = df_bnn_metrics['epistemic_uncertainty'].mean()
            st.metric("Avg Epistemic", f"{avg_epistemic:.2f} pts")

        with col3:
            avg_aleatoric = df_bnn_metrics['aleatoric_uncertainty'].mean()
            st.metric("Avg Aleatoric", f"{avg_aleatoric:.2f} pts")

        with col4:
            avg_ci_width = (df_bnn_metrics['ci_upper'] - df_bnn_metrics['ci_lower']).mean()
            st.metric("Avg CI Width (95%)", f"{avg_ci_width:.2f} pts")

        # Plots
        col1, col2 = st.columns(2)

        with col1:
            fig_uncertainty_dist = plot_bnn_uncertainty_distribution(df_bnn_metrics)
            st.plotly_chart(fig_uncertainty_dist, use_container_width=True)

        with col2:
            fig_prediction_intervals = plot_bnn_prediction_intervals(df_bnn_metrics)
            st.plotly_chart(fig_prediction_intervals, use_container_width=True)

        st.markdown("""
        **Understanding BNN Uncertainty:**
        - **Epistemic Uncertainty**: Model uncertainty (reducible with more data/better models)
        - **Aleatoric Uncertainty**: Inherent randomness in the game (irreducible)
        - **95% Confidence Interval**: Range where we expect the true margin to fall 95% of the time
        """)
    else:
        st.info("BNN uncertainty data not available")

    # Model comparison
    st.header("🔬 Model Comparison")

    if not df_model_comparison.empty:
        st.dataframe(
            df_model_comparison.style.format({
                'mae': '{:.2f}',
                'std_error': '{:.2f}',
                'accuracy': '{:.1%}',
                'avg_confidence': '{:.3f}'
            }).background_gradient(subset=['accuracy'], cmap='RdYlGn'),
            use_container_width=True
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
            if 'week' in df_retrospectives.columns:
                weeks = sorted(df_retrospectives['week'].unique())
                selected_weeks = st.multiselect(
                    "Filter by Week",
                    options=weeks,
                    default=[],
                    key="week_filter"
                )

        with col2:
            # Team filter (player's team or opponent)
            if 'team' in df_retrospectives.columns:
                teams = sorted(set(df_retrospectives['team'].unique()) |
                             set(df_retrospectives['opponent'].unique()))
                selected_teams = st.multiselect(
                    "Filter by Team",
                    options=teams,
                    default=[],
                    key="team_filter"
                )

        with col3:
            # Status filter (win/loss/push/pending)
            if 'status' in df_retrospectives.columns:
                statuses = sorted(df_retrospectives['status'].dropna().unique())
                selected_statuses = st.multiselect(
                    "Filter by Result",
                    options=statuses,
                    default=[],
                    key="status_filter"
                )

        # Apply filters
        filtered_df = df_retrospectives.copy()

        if selected_weeks:
            filtered_df = filtered_df[filtered_df['week'].isin(selected_weeks)]

        if selected_teams:
            filtered_df = filtered_df[
                (filtered_df['team'].isin(selected_teams)) |
                (filtered_df['opponent'].isin(selected_teams))
            ]

        if selected_statuses:
            filtered_df = filtered_df[filtered_df['status'].isin(selected_statuses)]

        # Sort by most recent before selecting display columns
        if 'kickoff' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('kickoff', ascending=False)

        # Display count
        st.markdown(f"**Showing {len(filtered_df)} predictions**")

        # Prepare display dataframe for player-level predictions
        display_cols = [
            'season', 'week', 'player_name', 'team', 'opponent', 'is_home',
            'predicted_value', 'actual_value', 'mae',
            'status', 'pnl', 'edge', 'prob_win', 'calibrated',
            'model_name', 'model_version'
        ]

        # Only include columns that exist
        available_cols = [col for col in display_cols if col in filtered_df.columns]
        predictions_display = filtered_df[available_cols].copy()

        # Add result column based on status
        if 'status' in filtered_df.columns:
            predictions_display['result'] = filtered_df['status'].map({
                'win': '✓ Win',
                'loss': '✗ Loss',
                'push': '~ Push',
                'pending': '⏳ Pending'
            })

        # Format and display
        format_dict = {}
        if 'predicted_value' in predictions_display.columns:
            format_dict['predicted_value'] = '{:.1f}'
        if 'actual_value' in predictions_display.columns:
            format_dict['actual_value'] = '{:.1f}'
        if 'mae' in predictions_display.columns:
            format_dict['mae'] = '{:.1f}'
        if 'pnl' in predictions_display.columns:
            format_dict['pnl'] = '${:.2f}'
        if 'edge' in predictions_display.columns:
            format_dict['edge'] = '{:.4f}'
        if 'prob_win' in predictions_display.columns:
            format_dict['prob_win'] = '{:.1%}'

        # Display with highlighting for win/loss
        if 'status' in predictions_display.columns:
            def highlight_status(row):
                if 'status' not in row:
                    return [''] * len(row)
                status = row['status']
                if status == 'win':
                    return ['background-color: #d4edda; color: #155724'] * len(row)
                elif status == 'loss':
                    return ['background-color: #f8d7da; color: #721c24'] * len(row)
                elif status == 'push':
                    return ['background-color: #fff3cd; color: #856404'] * len(row)
                return [''] * len(row)

            st.dataframe(
                predictions_display.style.format(format_dict).apply(highlight_status, axis=1),
                use_container_width=True,
                height=600
            )
        else:
            st.dataframe(
                predictions_display.style.format(format_dict),
                use_container_width=True,
                height=600
            )

        # Download button
        csv = predictions_display.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No predictions available")

    # Footer
    st.markdown("---")
    st.markdown(f"**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown("Live scores update every 30 seconds • Historical data refreshes every 5 minutes")

if __name__ == "__main__":
    main()
