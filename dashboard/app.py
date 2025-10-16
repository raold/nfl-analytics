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
from zoneinfo import ZoneInfo
import requests
import time

# Database configuration from environment variables
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5544")),
    "dbname": os.getenv("DB_NAME", "devdb01"),
    "user": os.getenv("DB_USER", "dro"),
    "password": os.getenv("DB_PASSWORD", "sicillionbillions")
}

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

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_database_connection():
    """Get database connection."""
    return psycopg.connect(**DB_CONFIG)

@st.cache_data(ttl=300)
def fetch_retrospective_metrics(model_version=None, season=None):
    """Fetch retrospective analysis metrics."""
    query = """
    SELECT
        r.game_id,
        g.season,
        g.week,
        g.home_team,
        g.away_team,
        g.home_score,
        g.away_score,
        r.actual_margin,
        r.predicted_margin,
        r.margin_error,
        r.abs_margin_error,
        r.outcome_type,
        r.surprise_factor,
        p.model_version,
        p.home_win_prob,
        p.predicted_spread,
        p.bet_confidence,
        g.spread_close,
        g.kickoff
    FROM predictions.retrospectives r
    JOIN games g ON r.game_id = g.game_id
    LEFT JOIN predictions.game_predictions p ON r.game_id = p.game_id
    WHERE 1=1
    """

    params = []
    if model_version:
        query += " AND p.model_version = %s"
        params.append(model_version)
    if season:
        query += " AND g.season = %s"
        params.append(season)

    query += " ORDER BY g.kickoff DESC LIMIT 1000"

    with psycopg.connect(**DB_CONFIG) as conn:
        df = pd.read_sql_query(query, conn, params=params if params else None)

    return df

@st.cache_data(ttl=300)
def fetch_performance_over_time():
    """Fetch rolling performance metrics."""
    query = """
    WITH weekly_metrics AS (
        SELECT
            g.season,
            g.week,
            COUNT(*) as total_games,
            AVG(r.abs_margin_error) as mae,
            AVG(CASE
                WHEN (r.actual_margin > 0 AND r.predicted_margin > 0) OR
                     (r.actual_margin < 0 AND r.predicted_margin < 0)
                THEN 1.0 ELSE 0.0
            END) as win_rate,
            AVG(CASE
                WHEN g.spread_close IS NOT NULL THEN
                    CASE WHEN (r.actual_margin + g.spread_close) > 0 THEN 1.0 ELSE 0.0 END
                ELSE NULL
            END) as ats_accuracy
        FROM predictions.retrospectives r
        JOIN games g ON r.game_id = g.game_id
        WHERE g.season >= 2022
        GROUP BY g.season, g.week
        ORDER BY g.season, g.week
    )
    SELECT * FROM weekly_metrics
    """

    with psycopg.connect(**DB_CONFIG) as conn:
        df = pd.read_sql_query(query, conn)

    return df

@st.cache_data(ttl=300)
def fetch_model_comparison():
    """Compare different model versions."""
    query = """
    SELECT
        p.model_version,
        COUNT(*) as total_predictions,
        AVG(r.abs_margin_error) as mae,
        STDDEV(r.margin_error) as std_error,
        AVG(CASE
            WHEN (r.actual_margin > 0 AND r.predicted_margin > 0) OR
                 (r.actual_margin < 0 AND r.predicted_margin < 0)
            THEN 1.0 ELSE 0.0
        END) as accuracy,
        AVG(p.bet_confidence) as avg_confidence
    FROM predictions.game_predictions p
    JOIN predictions.retrospectives r ON p.game_id = r.game_id
    GROUP BY p.model_version
    ORDER BY accuracy DESC
    """

    with psycopg.connect(**DB_CONFIG) as conn:
        df = pd.read_sql_query(query, conn)

    return df

@st.cache_data(ttl=30)  # Cache for 30 seconds for live data
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

@st.cache_data(ttl=300)  # Cache for 5 minutes
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

@st.cache_data(ttl=30)  # Cache for 30 seconds for live updates
def fetch_current_week_games():
    """Fetch current week's games with predictions and outcomes."""
    # Dynamically determine current week based on today's date
    # Oct 11, 2025 is Week 6
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
        END as game_status,
        p.home_win_prob,
        p.predicted_spread,
        p.recommended_bet,
        p.bet_confidence,
        r.predicted_margin,
        r.actual_margin,
        r.margin_error
    FROM games g
    LEFT JOIN predictions.game_predictions p ON g.game_id = p.game_id
    LEFT JOIN predictions.retrospectives r ON g.game_id = r.game_id
    WHERE g.season = 2025 AND g.week = 6 AND g.game_type = 'REG'
    ORDER BY g.kickoff
    """

    with psycopg.connect(**DB_CONFIG) as conn:
        df = pd.read_sql_query(query, conn)

    # Fetch live scores for in-progress games
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

    return df

def normalize_team_display(team_abbr):
    """Normalize team abbreviation for display (LA -> LAR for clarity)."""
    if team_abbr == 'LA':
        return 'LAR'  # Display as LAR to distinguish from LAC
    return team_abbr

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
        title="Mean Absolute Error Over Time",
        xaxis_title="Season-Week",
        yaxis_title="MAE (points)",
        hovermode='x unified',
        height=400
    )

    return fig

def plot_ats_accuracy(df):
    """Plot ATS (Against the Spread) accuracy."""
    df['week_label'] = df['season'].astype(str) + '-W' + df['week'].astype(str)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['week_label'],
        y=df['ats_accuracy'] * 100,
        mode='lines+markers',
        name='ATS Accuracy',
        line=dict(color='#2ca02c', width=3),
        marker=dict(size=8)
    ))

    # Add 52.4% breakeven line
    fig.add_hline(y=52.4, line_dash="dash", line_color="red",
                  annotation_text="52.4% (Breakeven)", annotation_position="right")

    fig.update_layout(
        title="ATS Accuracy Over Time",
        xaxis_title="Season-Week",
        yaxis_title="ATS Accuracy (%)",
        hovermode='x unified',
        height=400
    )

    return fig

def plot_error_distribution(df):
    """Plot distribution of prediction errors."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df['margin_error'],
        nbinsx=40,
        name='Error Distribution',
        marker_color='#9467bd',
        opacity=0.7
    ))

    fig.add_vline(x=0, line_dash="dash", line_color="black",
                  annotation_text="Perfect Prediction", annotation_position="top")

    fig.update_layout(
        title="Prediction Error Distribution",
        xaxis_title="Margin Error (points)",
        yaxis_title="Count",
        height=400,
        showlegend=False
    )

    return fig

def plot_calibration(df):
    """Plot calibration curve."""
    # Bin predictions by probability
    df['prob_bin'] = pd.cut(df['home_win_prob'], bins=10, labels=False)

    calibration = df.groupby('prob_bin').agg({
        'home_win_prob': 'mean',
        'actual_margin': lambda x: (x > 0).mean()
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
        title="Calibration Curve",
        xaxis_title="Predicted Probability",
        yaxis_title="Actual Frequency",
        height=400,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )

    return fig

def main():
    st.title("🏈 NFL Analytics Monitoring Dashboard")
    st.markdown("Real-time monitoring of prediction performance and model metrics")

    # Sidebar filters
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
                "SELECT DISTINCT model_version FROM predictions.game_predictions ORDER BY model_version",
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

        # Derive winner columns from margin data (only if we have data)
        if not df_retrospectives.empty:
            df_retrospectives['actual_winner'] = df_retrospectives['actual_margin'].apply(
                lambda x: 'home' if x > 0 else ('away' if x < 0 else 'push')
            )
            df_retrospectives['predicted_winner'] = df_retrospectives['predicted_margin'].apply(
                lambda x: 'home' if x > 0 else ('away' if x < 0 else 'push')
            )

        df_performance = fetch_performance_over_time()
        df_model_comparison = fetch_model_comparison()

    # Summary metrics
    st.header("📊 Summary Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_games = len(df_retrospectives)
        st.metric("Total Games", f"{total_games:,}")

    with col2:
        if not df_retrospectives.empty and 'actual_winner' in df_retrospectives.columns:
            win_rate = (df_retrospectives['actual_winner'] == df_retrospectives['predicted_winner']).mean()
            st.metric("Win Rate", f"{win_rate*100:.1f}%",
                      delta=f"{(win_rate-0.5)*100:.1f}% vs 50%")
        else:
            st.metric("Win Rate", "N/A")

    with col3:
        if not df_retrospectives.empty:
            mae = df_retrospectives['abs_margin_error'].mean()
            st.metric("Mean Absolute Error", f"{mae:.2f} pts")
        else:
            st.metric("Mean Absolute Error", "N/A")

    with col4:
        if not df_retrospectives.empty and 'spread_close' in df_retrospectives.columns:
            ats_mask = df_retrospectives['spread_close'].notna()
            if ats_mask.any():
                ats_correct = ((df_retrospectives.loc[ats_mask, 'actual_margin'] +
                               df_retrospectives.loc[ats_mask, 'spread_close']) > 0)
                ats_accuracy = ats_correct.mean()
                st.metric("ATS Accuracy", f"{ats_accuracy*100:.1f}%",
                         delta=f"{(ats_accuracy-0.524)*100:.1f}% vs BE")
            else:
                st.metric("ATS Accuracy", "N/A")
        else:
            st.metric("ATS Accuracy", "N/A")

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

                    # Our model predictions
                    prediction_lines = []
                    if pd.notna(game['home_win_prob']):
                        home_prob = float(game['home_win_prob']) * 100
                        away_prob = (1 - float(game['home_win_prob'])) * 100
                        predicted_winner = game['home_team_display'] if game['home_win_prob'] > 0.5 else game['away_team_display']
                        prediction_lines.append(f"<strong>Winner:</strong> {predicted_winner} ({max(home_prob, away_prob):.1f}%)")

                    if pd.notna(game['predicted_spread']):
                        pred_spread = float(game['predicted_spread'])
                        prediction_lines.append(f"<strong>Spread:</strong> {game['home_team_display']} {pred_spread:+.1f}")

                    # Calculate predicted total from predicted margin and average score
                    if pd.notna(game['predicted_margin']):
                        # Estimate total based on predicted margin (rough estimate assuming avg 45 pts total)
                        pred_total_estimate = 45.0  # You could make this more sophisticated
                        prediction_lines.append(f"<strong>Total:</strong> ~{pred_total_estimate:.0f}")

                    prediction_info = "<div style='font-size: 14px; color: #333;'>" + " | ".join(prediction_lines) + "</div>" if prediction_lines else ""

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
    st.header("📅 Week 6")

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
                    actual_margin = game['actual_margin'] if pd.notna(game['actual_margin']) else 0

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

                    # Check prediction accuracy
                    prediction_correct = False
                    prediction_text = ""
                    if pd.notna(game['predicted_margin']):
                        predicted_winner = game['home_team'] if game['predicted_margin'] > 0 else game['away_team']
                        prediction_correct = (predicted_winner == winner)
                        margin_error = abs(game['margin_error']) if pd.notna(game['margin_error']) else 0
                        prediction_text = f"{'✅ CORRECT' if prediction_correct else '❌ WRONG'} (Error: {margin_error:.1f} pts)"

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

                    # Our model predictions
                    prediction_lines = []
                    if pd.notna(game['home_win_prob']):
                        home_prob = float(game['home_win_prob']) * 100
                        away_prob = (1 - float(game['home_win_prob'])) * 100
                        predicted_winner = game['home_team_display'] if game['home_win_prob'] > 0.5 else game['away_team_display']
                        prediction_lines.append(f"Winner: {predicted_winner} ({max(home_prob, away_prob):.1f}%)")

                    if pd.notna(game['predicted_spread']):
                        pred_spread = float(game['predicted_spread'])
                        prediction_lines.append(f"Spread: {game['home_team_display']} {pred_spread:+.1f}")

                    # Calculate predicted total from predicted margin (rough estimate)
                    if pd.notna(game['predicted_margin']):
                        pred_total_estimate = 45.0
                        prediction_lines.append(f"Total: ~{pred_total_estimate:.0f}")

                    prediction_info = " | ".join(prediction_lines) if prediction_lines else ""

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

                    # Our model predictions
                    prediction_lines = []
                    if pd.notna(game['home_win_prob']):
                        home_prob = float(game['home_win_prob']) * 100
                        away_prob = (1 - float(game['home_win_prob'])) * 100
                        predicted_winner = game['home_team_display'] if game['home_win_prob'] > 0.5 else game['away_team_display']
                        prediction_lines.append(f"Winner: {predicted_winner} ({max(home_prob, away_prob):.1f}%)")

                    if pd.notna(game['predicted_spread']):
                        pred_spread = float(game['predicted_spread'])
                        prediction_lines.append(f"Spread: {game['home_team_display']} {pred_spread:+.1f}")

                    # Calculate predicted total from predicted margin (rough estimate)
                    if pd.notna(game['predicted_margin']):
                        pred_total_estimate = 45.0
                        prediction_lines.append(f"Total: ~{pred_total_estimate:.0f}")

                    prediction_info = " | ".join(prediction_lines) if prediction_lines else ""

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
            # Team filter
            if 'home_team' in df_retrospectives.columns:
                teams = sorted(set(df_retrospectives['home_team'].unique()) |
                             set(df_retrospectives['away_team'].unique()))
                selected_teams = st.multiselect(
                    "Filter by Team",
                    options=teams,
                    default=[],
                    key="team_filter"
                )

        with col3:
            # Outcome type filter
            if 'outcome_type' in df_retrospectives.columns:
                outcome_types = sorted(df_retrospectives['outcome_type'].dropna().unique())
                selected_outcomes = st.multiselect(
                    "Filter by Outcome",
                    options=outcome_types,
                    default=[],
                    key="outcome_filter"
                )

        # Apply filters
        filtered_df = df_retrospectives.copy()

        if selected_weeks:
            filtered_df = filtered_df[filtered_df['week'].isin(selected_weeks)]

        if selected_teams:
            filtered_df = filtered_df[
                (filtered_df['home_team'].isin(selected_teams)) |
                (filtered_df['away_team'].isin(selected_teams))
            ]

        if selected_outcomes:
            filtered_df = filtered_df[filtered_df['outcome_type'].isin(selected_outcomes)]

        # Sort by most recent before selecting display columns
        if 'kickoff' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('kickoff', ascending=False)

        # Display count
        st.markdown(f"**Showing {len(filtered_df)} predictions**")

        # Prepare display dataframe
        display_cols = [
            'season', 'week', 'home_team', 'away_team',
            'home_score', 'away_score', 'actual_margin',
            'predicted_margin', 'margin_error', 'abs_margin_error',
            'home_win_prob', 'predicted_spread', 'spread_close',
            'outcome_type', 'model_version'
        ]

        # Only include columns that exist
        available_cols = [col for col in display_cols if col in filtered_df.columns]
        predictions_display = filtered_df[available_cols].copy()

        # Add result column
        if 'actual_winner' in filtered_df.columns and 'predicted_winner' in filtered_df.columns:
            predictions_display['correct'] = (
                filtered_df['actual_winner'] == filtered_df['predicted_winner']
            ).map({True: '✓', False: '✗'})

        # Format and display
        format_dict = {}
        if 'actual_margin' in predictions_display.columns:
            format_dict['actual_margin'] = '{:.1f}'
        if 'predicted_margin' in predictions_display.columns:
            format_dict['predicted_margin'] = '{:.1f}'
        if 'margin_error' in predictions_display.columns:
            format_dict['margin_error'] = '{:.1f}'
        if 'abs_margin_error' in predictions_display.columns:
            format_dict['abs_margin_error'] = '{:.1f}'
        if 'home_win_prob' in predictions_display.columns:
            format_dict['home_win_prob'] = '{:.1%}'
        if 'predicted_spread' in predictions_display.columns:
            format_dict['predicted_spread'] = '{:.1f}'
        if 'spread_close' in predictions_display.columns:
            format_dict['spread_close'] = '{:.1f}'

        # Display with highlighting for correct/incorrect
        if 'correct' in predictions_display.columns:
            def highlight_correct(row):
                if row['correct'] == '✓':
                    return ['background-color: #d4edda; color: #155724'] * len(row)
                elif row['correct'] == '✗':
                    return ['background-color: #f8d7da; color: #721c24'] * len(row)
                return [''] * len(row)

            st.dataframe(
                predictions_display.style.format(format_dict).apply(highlight_correct, axis=1),
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
