#!/usr/bin/env python3
"""
GameOutcomeModel Predictions Dashboard
======================================

Displays game-level predictions from the GameOutcomeModel for the selected NFL season,
including spread, total, and win probability predictions with uncertainty estimates
and Vegas comparisons.

Author: Claude Code (updated by Codex)
Last Updated: December 2025
"""

import logging
import os
from typing import Optional, List

import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import requests

from py.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
DB_CONFIG = settings.database.psycopg_kwargs()
API_BASE_URL = settings.api.base_url
API_KEY = settings.api.api_key
LATEST_VIEW = "ml.game_outcome_predictions_latest"
FULL_TABLE = settings.game_outcome_table
DEFAULT_SEASON = int(os.getenv("GAME_OUTCOME_DEFAULT_SEASON", str(datetime.utcnow().year)))
STATE_LABELS = [
    ("Upcoming", "upcoming"),
    ("In Progress", "in_progress"),
    ("Final", "final"),
]
STATE_LOOKUP = {label: code for label, code in STATE_LABELS}
STATE_LOOKUP_REV = {code: label for label, code in STATE_LABELS}
EASTERN_TZ = pytz.timezone("US/Eastern")


def format_kickoff_time(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return "TBD"
    kickoff = pd.to_datetime(ts)
    if kickoff.tzinfo is None:
        kickoff = kickoff.tz_localize("UTC")
    kickoff_local = kickoff.tz_convert(EASTERN_TZ)
    return kickoff_local.strftime("%a %b %d • %I:%M %p ET")


def format_game_state(row: pd.Series) -> str:
    state_label = STATE_LOOKUP_REV.get(row.get('game_state'), row.get('game_state', ''))
    if row.get('game_state') == 'final' and pd.notna(row.get('actual_home_score')):
        return f"{state_label} • {row['home_team']} {row['actual_home_score']}-{row['actual_away_score']} {row['away_team']}"
    if row.get('game_state') == 'in_progress':
        home = row.get('actual_home_score')
        away = row.get('actual_away_score')
        if pd.notna(home) and pd.notna(away):
            return f"{state_label} • {row['home_team']} {int(home)}-{int(away)} {row['away_team']}"
    return state_label


def render_game_card(game: pd.Series):
    col_a, col_b, col_c = st.columns([2, 3, 2])
    with col_a:
        st.markdown(f"**{game['matchup']}**")
        st.caption(format_kickoff_time(game['kickoff']))
        st.caption(format_game_state(game))
        if game.get('odds_stale'):
            st.warning("Vegas line > 60 min old", icon="⚠️")

    with col_b:
        spread_std = game.get('spread_std')
        total_std = game.get('total_std')
        st.markdown(
            f"**Model Spread:** {game['pred_home_spread']:+.1f} "
            f"(σ {spread_std:.1f})" if pd.notna(spread_std) else f"**Model Spread:** {game['pred_home_spread']:+.1f}"
        )
        st.markdown(
            f"**Model Total:** {game['pred_total']:.1f} "
            f"(σ {total_std:.1f})" if pd.notna(total_std) else f"**Model Total:** {game['pred_total']:.1f}"
        )
        if pd.notna(game.get('vegas_spread')):
            vegas_line = f"**Vegas Spread:** {game['vegas_spread']:+.1f}"
            st.markdown(vegas_line)
        if pd.notna(game.get('vegas_total')):
            st.markdown(f"**Vegas Total:** {game['vegas_total']:.1f}")
        if pd.notna(game.get('home_moneyline')):
            st.caption(
                f"Moneyline: {game['home_team']} {game['home_moneyline']:+.0f} | "
                f"{game['away_team']} {game['away_moneyline']:+.0f}"
            )

    with col_c:
        win_prob = game.get('pred_home_win_prob')
        win_value = f"{win_prob*100:.1f}%" if pd.notna(win_prob) else "N/A"
        st.metric("Home Win %", win_value)
        st.metric("Spread Edge", f"{game['spread_edge']:+.1f}" if pd.notna(game.get('spread_edge')) else "N/A")
        st.metric("Total Edge", f"{game['total_edge']:+.1f}" if pd.notna(game.get('total_edge')) else "N/A")
        st.markdown("**Recommendations**")
        for rec in get_betting_recommendation(game):
            st.markdown(rec)

# Page config
st.set_page_config(
    page_title="Game Outcome Predictions",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def run_db_query(query: str, params: dict):
    """Execute a SQL query and return a DataFrame."""
    with psycopg2.connect(**DB_CONFIG) as conn:
        return pd.read_sql_query(query, conn, params=params)


def fetch_predictions_from_api(params: dict) -> Optional[pd.DataFrame]:
    """Fetch predictions from the FastAPI service if available."""
    if not API_BASE_URL:
        return None

    headers = {}
    if API_KEY:
        headers["X-API-Key"] = API_KEY

    try:
        response = requests.get(
            f"{API_BASE_URL.rstrip('/')}/game-outcomes/predictions",
            params=params,
            headers=headers,
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("predictions", [])
        return pd.DataFrame(data)
    except Exception as exc:
        logger.warning("API prediction fetch failed: %s", exc)
        return None


def fetch_prediction_runs_from_api() -> Optional[pd.DataFrame]:
    if not API_BASE_URL:
        return None

    headers = {}
    if API_KEY:
        headers["X-API-Key"] = API_KEY

    try:
        response = requests.get(
            f"{API_BASE_URL.rstrip('/')}/game-outcomes/runs",
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data)
    except Exception as exc:
        logger.warning("API run fetch failed: %s", exc)
        return None


@st.cache_data(ttl=300)
def get_default_week(season: int) -> Optional[int]:
    """Determine the default week (next upcoming game)."""
    upcoming_query = """
    SELECT MIN(week) AS week
    FROM public.games
    WHERE season = %(season)s
      AND kickoff >= NOW()
    """
    df = run_db_query(upcoming_query, {'season': season})
    week = df.at[0, 'week']
    if pd.notna(week):
        return int(week)

    fallback_query = """
    SELECT MAX(week) AS week
    FROM ml.game_outcome_predictions
    WHERE season = %(season)s
    """
    df = run_db_query(fallback_query, {'season': season})
    week = df.at[0, 'week']
    return int(week) if pd.notna(week) else None


def compute_confidence(edge: float, std: float) -> str:
    if pd.isna(edge) or pd.isna(std) or std == 0:
        return "Low"
    if abs(edge) > std * 1.5:
        return "High"
    if abs(edge) > std:
        return "Medium"
    return "Low"


def enrich_predictions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df['kickoff'] = pd.to_datetime(df['kickoff'])
    if 'odds_last_fetched' in df.columns:
        df['odds_last_fetched'] = pd.to_datetime(df['odds_last_fetched'])

    numeric_cols = [
        'pred_home_spread', 'pred_total', 'pred_home_win_prob',
        'spread_std', 'total_std', 'vegas_spread', 'vegas_total',
        'home_moneyline', 'away_moneyline', 'home_spread_odds',
        'away_spread_odds', 'over_odds', 'under_odds',
        'inference_time_ms', 'spread_edge', 'total_edge'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'spread_edge' not in df.columns:
        df['spread_edge'] = df['pred_home_spread'] - df['vegas_spread']
    if 'total_edge' not in df.columns:
        df['total_edge'] = df['pred_total'] - df['vegas_total']

    if 'game_state' not in df.columns:
        now = datetime.utcnow()
        df['game_state'] = np.where(
            (df.get('actual_home_score').notna()) & (df.get('actual_away_score').notna()),
            'final',
            np.where(df['kickoff'] <= now, 'in_progress', 'upcoming')
        )

    df['matchup'] = df['away_team'] + ' @ ' + df['home_team']
    df['spread_confidence'] = df.apply(
        lambda row: compute_confidence(row.get('spread_edge'), row.get('spread_std')),
        axis=1
    )
    df['total_confidence'] = df.apply(
        lambda row: compute_confidence(row.get('total_edge'), row.get('total_std')),
        axis=1
    )
    df['game_state_label'] = df['game_state'].map(STATE_LOOKUP_REV).fillna(df['game_state'].str.title())
    if 'odds_last_fetched' in df.columns:
        df['odds_age_min'] = (datetime.utcnow() - df['odds_last_fetched']).dt.total_seconds() / 60.0
        df['odds_stale'] = (df['odds_age_min'] > 60) & df['game_state'].isin(['upcoming', 'in_progress'])
    else:
        df['odds_stale'] = False
    return df


@st.cache_data(ttl=60)  # Cache for 1 minute
def load_predictions(
    season=DEFAULT_SEASON,
    week=None,
    team=None,
    prediction_run_id: Optional[str] = None,
    state_filters: Optional[List[str]] = None,
):
    """Load game predictions from API (preferred) or database."""
    params = {'season': season}
    if week:
        params['week'] = week
    if team:
        params['team'] = team
    if prediction_run_id:
        params['prediction_run_id'] = prediction_run_id

    normalized_states = [s.lower() for s in state_filters] if state_filters else []
    if len(normalized_states) == 1:
        params['game_state'] = normalized_states[0]

    use_api = API_BASE_URL is not None
    if use_api:
        api_df = fetch_predictions_from_api(params)
        if api_df is not None and not api_df.empty:
            df = enrich_predictions(api_df)
            if normalized_states:
                df = df[df['game_state'].isin(normalized_states)]
            return df

    table_name = FULL_TABLE if prediction_run_id else LATEST_VIEW
    query = f"""
    SELECT
        gp.prediction_id,
        gp.prediction_run_id,
        gp.game_id,
        gp.season,
        gp.week,
        gp.home_team,
        gp.away_team,
        g.kickoff,
        gp.pred_home_spread,
        gp.pred_total,
        gp.pred_home_win_prob,
        gp.spread_std,
        gp.total_std,
        g.spread_close AS vegas_spread,
        g.total_close AS vegas_total,
        g.home_moneyline,
        g.away_moneyline,
        g.home_spread_odds,
        g.away_spread_odds,
        g.over_odds,
        g.under_odds,
        g.odds_last_fetched,
        gp.actual_home_score,
        gp.actual_away_score,
        gp.actual_spread,
        gp.actual_total,
        gp.model_version,
        gp.prediction_timestamp,
        gp.inference_time_ms,
        (gp.pred_home_spread - g.spread_close) AS spread_edge,
        (gp.pred_total - g.total_close) AS total_edge,
        CASE
            WHEN COALESCE(g.game_status, '') ILIKE 'final%%'
                 OR (g.home_score IS NOT NULL AND g.away_score IS NOT NULL)
                THEN 'final'
            WHEN g.kickoff <= NOW()
                 AND (g.home_score IS NULL OR g.away_score IS NULL)
                THEN 'in_progress'
            ELSE 'upcoming'
        END AS game_state,
        g.game_status,
        g.home_score,
        g.away_score
    FROM {table_name} gp
    JOIN public.games g ON g.game_id = gp.game_id
    WHERE gp.season = %(season)s
    """

    if week:
        query += " AND gp.week = %(week)s"
    if team:
        query += " AND (gp.home_team = %(team)s OR gp.away_team = %(team)s)"
    if prediction_run_id:
        query += " AND gp.prediction_run_id = %(prediction_run_id)s"

    query += " ORDER BY gp.week, g.kickoff"

    try:
        df = run_db_query(query, params)
    except Exception as exc:
        logger.error("Failed to load predictions from DB: %s", exc)
        return pd.DataFrame()

    df = enrich_predictions(df)
    if normalized_states:
        df = df[df['game_state'].isin(normalized_states)]
    return df


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_prediction_runs():
    """Load prediction vintages."""
    if API_BASE_URL:
        api_runs = fetch_prediction_runs_from_api()
        if api_runs is not None:
            for col in ("season_start", "season_end"):
                if col not in api_runs.columns:
                    api_runs[col] = pd.NA
            return api_runs

    query = """
    SELECT
        prediction_run_id,
        total_games,
        started_at,
        completed_at,
        model_version,
        season_start,
        season_end
    FROM ml.game_outcome_prediction_runs
    ORDER BY completed_at DESC NULLS LAST
    """

    try:
        return run_db_query(query, {})
    except Exception as exc:
        logger.warning("Failed to load prediction runs: %s", exc)
        return pd.DataFrame()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_model_metrics():
    """Load model performance metrics from registry."""

    query = """
    SELECT
        name,
        version,
        author,
        metrics,
        created_at,
        status
    FROM ml.model_registry
    WHERE name = 'game_outcome_model'
    ORDER BY created_at DESC
    LIMIT 1
    """

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(query)
        result = cur.fetchone()
        cur.close()
        conn.close()
        return result
    except Exception as exc:
        logger.warning("Failed to load model metrics: %s", exc)
        return None

def format_spread(spread):
    """Format spread with + for favorites."""
    if pd.isna(spread):
        return "N/A"
    return f"{spread:+.1f}"

def format_probability(prob):
    """Format probability as percentage."""
    if pd.isna(prob):
        return "N/A"
    return f"{prob:.1%}"

def get_betting_recommendation(row):
    """Generate betting recommendation based on edges."""
    recommendations = []

    # Spread recommendations
    if abs(row['spread_edge']) > row['spread_std'] * 1.5:
        if row['spread_edge'] > 0:
            recommendations.append(f"🔥 **Strong:** {row['home_team']} {format_spread(row['vegas_spread'])}")
        else:
            recommendations.append(f"🔥 **Strong:** {row['away_team']} {format_spread(-row['vegas_spread'])}")
    elif abs(row['spread_edge']) > row['spread_std']:
        if row['spread_edge'] > 0:
            recommendations.append(f"⭐ **Medium:** {row['home_team']} {format_spread(row['vegas_spread'])}")
        else:
            recommendations.append(f"⭐ **Medium:** {row['away_team']} {format_spread(-row['vegas_spread'])}")

    # Total recommendations
    if abs(row['total_edge']) > row['total_std'] * 1.5:
        if row['total_edge'] > 0:
            recommendations.append(f"🔥 **Strong:** Over {row['vegas_total']:.1f}")
        else:
            recommendations.append(f"🔥 **Strong:** Under {row['vegas_total']:.1f}")
    elif abs(row['total_edge']) > row['total_std']:
        if row['total_edge'] > 0:
            recommendations.append(f"⭐ **Medium:** Over {row['vegas_total']:.1f}")
        else:
            recommendations.append(f"⭐ **Medium:** Under {row['vegas_total']:.1f}")

    return recommendations if recommendations else ["No strong edges"]


def format_run_label(row: pd.Series) -> str:
    """Format sidebar label for a prediction run."""
    completed = pd.to_datetime(row.get('completed_at'))
    timestamp = completed.strftime("%Y-%m-%d %H:%M") if pd.notna(completed) else "In Progress"
    model_version = row.get('model_version') or "unknown"
    games = row.get('total_games', 0)
    run_id = str(row.get('prediction_run_id'))[:8]
    return f"{timestamp} UTC • {model_version} • {games} games • {run_id}"


def get_season_options(run_df: pd.DataFrame) -> List[int]:
    """Derive available seasons from prediction runs."""
    seasons = set()
    if not run_df.empty:
        for col in ("season_start", "season_end"):
            if col in run_df.columns:
                valid = run_df[col].dropna().astype(int).tolist()
                seasons.update(valid)
    if not seasons:
        seasons.add(DEFAULT_SEASON)
    return sorted(seasons)

# Main dashboard
st.title("🏈 GameOutcomeModel Predictions")
st.markdown("*Game-level predictions using XGBoost multi-output regression with 293 features*")

# Load model metrics
model_info = load_model_metrics()

if model_info:
    metrics = model_info['metrics']

    # Display model performance in header
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Spread MAE", f"{metrics.get('spread_mae', 'N/A'):.1f} pts")
    with col2:
        st.metric("Total MAE", f"{metrics.get('total_mae', 'N/A'):.1f} pts")
    with col3:
        st.metric("Win Prob AUC", f"{metrics.get('win_prob_auc', 'N/A'):.3f}")
    with col4:
        st.metric("Win Accuracy", f"{metrics.get('win_prob_accuracy', 'N/A'):.1%}")
    with col5:
        st.metric("Training Games", f"{metrics.get('training_games', 'N/A'):,}")

    st.markdown(f"*Model Version: {model_info['version']} | Status: {model_info['status'].upper()}*")
    st.markdown("---")

if selected_run_id and selected_run_meta is not None:
    completed = selected_run_meta['completed_at']
    completed_str = completed.strftime("%Y-%m-%d %H:%M") if pd.notna(completed) else "in progress"
    st.info(
        f"Prediction Run `{selected_run_id[:8]}` • "
        f"{completed_str} UTC • Version {selected_run_meta.get('model_version', 'n/a')}"
    )
else:
    st.caption("Showing the most recent predictions (auto-refresh).")

# Sidebar filters
st.sidebar.header("Filters")

# Season selector (auto-derived)
season_options = get_season_options(prediction_runs)
default_season = DEFAULT_SEASON if DEFAULT_SEASON in season_options else season_options[-1]
season_index = season_options.index(default_season)
season = st.sidebar.selectbox("Season", options=season_options, index=season_index)

# Week selector
week_options = ["All Weeks"] + [f"Week {i}" for i in range(1, 19)]
default_week_value = get_default_week(season)
default_week_index = 0
if default_week_value:
    default_label = f"Week {default_week_value}"
    if default_label in week_options:
        default_week_index = week_options.index(default_label)
week_selection = st.sidebar.selectbox("Week", options=week_options, index=default_week_index)
selected_week = None if week_selection == "All Weeks" else int(week_selection.split()[1])

# Team selector
team_options = ["All Teams"] + [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"
]
selected_team = st.sidebar.selectbox("Team", options=team_options, index=0)
selected_team = None if selected_team == "All Teams" else selected_team

# Game state selector
state_labels = [label for label, _ in STATE_LABELS]
selected_state_labels = st.sidebar.multiselect(
    "Game State",
    options=state_labels,
    default=state_labels
)
selected_state_codes = [STATE_LOOKUP[label] for label in selected_state_labels] if selected_state_labels else [code for _, code in STATE_LABELS]

# Prediction vintage selector
prediction_runs = load_prediction_runs()
run_options = {"Latest (auto)": None}
run_metadata = {}

if not prediction_runs.empty:
    prediction_runs = prediction_runs.copy()
    prediction_runs['completed_at'] = pd.to_datetime(prediction_runs['completed_at'])
    prediction_runs['prediction_run_id'] = prediction_runs['prediction_run_id'].astype(str)
    for _, row in prediction_runs.iterrows():
        label = format_run_label(row)
        run_id = row['prediction_run_id']
        run_options[label] = run_id
        run_metadata[run_id] = row

run_choice = st.sidebar.selectbox("Vintage", options=list(run_options.keys()), index=0)
selected_run_id = run_options[run_choice]
selected_run_meta = run_metadata.get(selected_run_id) if selected_run_id else None

# Edge threshold
edge_threshold = st.sidebar.slider(
    "Min Edge (std devs)",
    min_value=0.0,
    max_value=2.0,
    value=1.0,
    step=0.25,
    help="Show only games where model disagrees with Vegas by this many standard deviations"
)

# Load predictions
predictions_df = load_predictions(
    season,
    selected_week,
    selected_team,
    selected_run_id,
    state_filters=selected_state_codes
)

if predictions_df.empty:
    st.warning("No predictions found for the selected filters. Predictions are being generated...")
    st.info(f"The GameOutcomeModel is currently processing all regular-season games in {season}. This takes approximately 5-6 minutes.")

    # Show progress indicator
    progress_placeholder = st.empty()
    with progress_placeholder.container():
        st.spinner("Generating predictions... (each game takes ~1.2 seconds)")

else:
    # Filter by edge threshold
    filtered_df = predictions_df[
        (abs(predictions_df['spread_edge']) > predictions_df['spread_std'] * edge_threshold) |
        (abs(predictions_df['total_edge']) > predictions_df['total_std'] * edge_threshold)
    ]

    # Display summary stats
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Games", len(predictions_df))
    with col2:
        st.metric("Games with Edge", len(filtered_df))
    with col3:
        avg_spread_edge = filtered_df['spread_edge'].abs().mean() if not filtered_df.empty else 0
        st.metric("Avg Spread Edge", f"{avg_spread_edge:.1f} pts")

    state_counts = predictions_df['game_state_label'].value_counts()
    state_cols = st.columns(len(STATE_LABELS))
    for idx, (label, code) in enumerate(STATE_LABELS):
        with state_cols[idx]:
            st.metric(label, int(state_counts.get(label, 0)))

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Predictions", "🎯 Top Edges", "📈 Analysis", "📋 Raw Data"])

    with tab1:
        st.header("Game Predictions")

        if not filtered_df.empty:
            for state_label, state_code in STATE_LABELS:
                state_subset = filtered_df[filtered_df['game_state'] == state_code]
                if state_subset.empty:
                    continue
                st.subheader(f"{state_label} ({len(state_subset)})")
                weeks = sorted(state_subset['week'].unique())
                for week in weeks:
                    week_games = state_subset[state_subset['week'] == week].sort_values('kickoff')
                    expanded = state_code == 'upcoming'
                    with st.expander(f"Week {week} ({len(week_games)})", expanded=expanded):
                        for _, game in week_games.iterrows():
                            render_game_card(game)
                            st.markdown("---")
        else:
            st.info("No games meet the edge threshold criteria. Try lowering the threshold.")

    with tab2:
        st.header("Top Betting Edges")

        # Top spread edges
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Top Spread Edges")
            top_spread = predictions_df.nlargest(10, 'spread_edge')[
                ['week', 'matchup', 'pred_home_spread', 'vegas_spread', 'spread_edge', 'spread_std', 'game_state']
            ]

            for _, game in top_spread.iterrows():
                confidence = "🔥" if abs(game['spread_edge']) > game['spread_std'] * 1.5 else "⭐"
                state_prefix = STATE_LOOKUP_REV.get(game['game_state'], '')
                st.markdown(
                    f"{confidence} **Week {game['week']}** - {state_prefix} - {game['matchup']}: "
                    f"Model {format_spread(game['pred_home_spread'])} vs Vegas {format_spread(game['vegas_spread'])} "
                    f"(Edge: {game['spread_edge']:+.1f})"
                )

        with col2:
            st.subheader("📊 Top Total Edges")
            top_total = predictions_df.nlargest(10, 'total_edge')[
                ['week', 'matchup', 'pred_total', 'vegas_total', 'total_edge', 'total_std', 'game_state']
            ]

            for _, game in top_total.iterrows():
                confidence = "🔥" if abs(game['total_edge']) > game['total_std'] * 1.5 else "⭐"
                direction = "Over" if game['total_edge'] > 0 else "Under"
                state_prefix = STATE_LOOKUP_REV.get(game['game_state'], '')
                st.markdown(
                    f"{confidence} **Week {game['week']}** - {state_prefix} - {game['matchup']}: "
                    f"{direction} (Model {game['pred_total']:.1f} vs Vegas {game['vegas_total']:.1f}, "
                    f"Edge: {game['total_edge']:+.1f})"
                )

    with tab3:
        st.header("Statistical Analysis")

        # Edge distribution
        col1, col2 = st.columns(2)

        with col1:
            fig_spread = px.histogram(
                predictions_df,
                x='spread_edge',
                nbins=30,
                title="Spread Edge Distribution",
                labels={'spread_edge': 'Spread Edge (points)', 'count': 'Number of Games'},
                color_discrete_sequence=['#1f77b4']
            )
            fig_spread.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="No Edge")
            st.plotly_chart(fig_spread, use_container_width=True)

        with col2:
            fig_total = px.histogram(
                predictions_df,
                x='total_edge',
                nbins=30,
                title="Total Edge Distribution",
                labels={'total_edge': 'Total Edge (points)', 'count': 'Number of Games'},
                color_discrete_sequence=['#2ca02c']
            )
            fig_total.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="No Edge")
            st.plotly_chart(fig_total, use_container_width=True)

        # Model vs Vegas correlation
        col3, col4 = st.columns(2)

        with col3:
            if predictions_df['vegas_spread'].notna().any():
                fig_spread_corr = px.scatter(
                    predictions_df[predictions_df['vegas_spread'].notna()],
                    x='vegas_spread',
                    y='pred_home_spread',
                    title="Model vs Vegas Spreads",
                    labels={'vegas_spread': 'Vegas Spread', 'pred_home_spread': 'Model Spread'},
                    color='spread_confidence',
                    color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'blue'}
                )
                # Add diagonal reference line
                x_range = [predictions_df['vegas_spread'].min(), predictions_df['vegas_spread'].max()]
                fig_spread_corr.add_trace(
                    go.Scatter(x=x_range, y=x_range, mode='lines',
                             name='Perfect Agreement', line=dict(dash='dash', color='gray'))
                )
                st.plotly_chart(fig_spread_corr, use_container_width=True)

        with col4:
            if predictions_df['vegas_total'].notna().any():
                fig_total_corr = px.scatter(
                    predictions_df[predictions_df['vegas_total'].notna()],
                    x='vegas_total',
                    y='pred_total',
                    title="Model vs Vegas Totals",
                    labels={'vegas_total': 'Vegas Total', 'pred_total': 'Model Total'},
                    color='total_confidence',
                    color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'blue'}
                )
                # Add diagonal reference line
                x_range = [predictions_df['vegas_total'].min(), predictions_df['vegas_total'].max()]
                fig_total_corr.add_trace(
                    go.Scatter(x=x_range, y=x_range, mode='lines',
                             name='Perfect Agreement', line=dict(dash='dash', color='gray'))
                )
                st.plotly_chart(fig_total_corr, use_container_width=True)

        # Summary statistics
        st.subheader("Summary Statistics")

        col5, col6, col7, col8 = st.columns(4)

        with col5:
            games_with_spread_edge = (abs(predictions_df['spread_edge']) > predictions_df['spread_std']).sum()
            st.metric("Games w/ Spread Edge", games_with_spread_edge)

        with col6:
            games_with_total_edge = (abs(predictions_df['total_edge']) > predictions_df['total_std']).sum()
            st.metric("Games w/ Total Edge", games_with_total_edge)

        with col7:
            if predictions_df['spread_edge'].notna().any():
                avg_spread_disagreement = predictions_df['spread_edge'].abs().mean()
                st.metric("Avg Spread Disagreement", f"{avg_spread_disagreement:.2f} pts")

        with col8:
            if predictions_df['total_edge'].notna().any():
                avg_total_disagreement = predictions_df['total_edge'].abs().mean()
                st.metric("Avg Total Disagreement", f"{avg_total_disagreement:.2f} pts")

    with tab4:
        st.header("Raw Prediction Data")

        # Select columns to display
        display_columns = [
            'game_id', 'week', 'matchup', 'kickoff',
            'pred_home_spread', 'vegas_spread', 'spread_edge', 'spread_std',
            'pred_total', 'vegas_total', 'total_edge', 'total_std',
            'pred_home_win_prob', 'inference_time_ms'
        ]

        # Filter columns that exist
        available_cols = [col for col in display_columns if col in predictions_df.columns]

        st.dataframe(
            predictions_df[available_cols].style.format({
                'pred_home_spread': '{:+.1f}',
                'vegas_spread': '{:+.1f}',
                'spread_edge': '{:+.2f}',
                'spread_std': '{:.2f}',
                'pred_total': '{:.1f}',
                'vegas_total': '{:.1f}',
                'total_edge': '{:+.2f}',
                'total_std': '{:.2f}',
                'pred_home_win_prob': '{:.1%}',
                'inference_time_ms': '{:.0f}'
            }),
            use_container_width=True
        )

        # Download button
        csv = predictions_df[available_cols].to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv,
            file_name=f"game_outcome_predictions_{season}_week{selected_week or 'all'}.csv",
            mime="text/csv"
        )

# Footer with refresh info
st.markdown("---")
st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data updates every minute")
st.caption("GameOutcomeModel v1.0.0 - Trained on 2022-2024 data (652 games) with 293 features per game")
