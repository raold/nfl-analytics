#!/usr/bin/env python3
"""
NFL Analytics Monitoring Dashboard - Refactored

Clean orchestration layer using modular components.
- Queries: dashboard.queries module (predictions, games, models, etc.)
- Charts: dashboard.components.charts
- Metrics: dashboard.components.metrics
- Tables: dashboard.components.tables
- Cards: dashboard.components.cards

Author: Claude Code
Date: November 22, 2025
Version: 2.0 (Refactored)
"""

import logging
import os
import sys
from pathlib import Path

import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import query modules
# Import component modules
from dashboard.components import (
    display_advanced_metrics,
    display_edge_statistics,
    # Metrics
    display_summary_metrics,
    format_model_comparison_table,
    format_predictions_table,
    game_prediction_card,
    plot_calibration,
    plot_edge_distribution,
    plot_error_distribution,
    plot_mae_over_time,
    plot_model_comparison_mae,
    plot_pnl_over_time,
    # Charts
    plot_win_rate_over_time,
    service_health_card,
)
from dashboard.queries import GameQueries, ModelQueries, PredictionQueries, SystemQueries

# Import utility functions
try:
    from dashboard.utils.db import get_db_connection

    POOLING_AVAILABLE = True
except ImportError:
    POOLING_AVAILABLE = False
    logger.warning("Connection pooling not available")

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5544")),
    "dbname": os.getenv("DB_NAME", "devdb01"),
    "user": os.getenv("DB_USER", "dro"),
    "password": os.getenv("DB_PASSWORD"),
}

# Page configuration
st.set_page_config(
    page_title="NFL Analytics Dashboard",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# PAGE RENDERING FUNCTIONS
# ============================================================================


def render_predictions_page():
    """Render the main predictions analytics page."""
    st.header("📊 Prediction Analytics")

    # Sidebar filters
    st.sidebar.header("Filters")

    # Season filter
    seasons = list(range(2025, 2021, -1))
    selected_season = st.sidebar.selectbox("Season", ["All"] + seasons, index=0)

    # Model version filter
    try:
        with get_db_connection() as conn:
            # Get production/staging models
            df_models = ModelQueries.models_by_status(conn, status="production")
            model_versions = (
                df_models["model_version"].unique().tolist() if not df_models.empty else []
            )
    except:
        model_versions = []

    selected_model = st.sidebar.selectbox("Model Version", ["All"] + model_versions, index=0)

    # Fetch data
    with st.spinner("Loading prediction data..."):
        season_filter = None if selected_season == "All" else selected_season
        model_filter = None if selected_model == "All" else selected_model

        try:
            with get_db_connection() as conn:
                # Get retrospective metrics
                df_predictions = PredictionQueries.retrospective_metrics(
                    conn, model_version=model_filter, season=season_filter
                )

                # Get performance over time
                df_performance = PredictionQueries.weekly_performance_metrics(conn)

                # Get model comparison
                df_comparison = PredictionQueries.model_comparison(conn)

        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return

    # Display summary metrics
    display_summary_metrics(df_predictions)

    st.markdown("---")

    # Performance charts (2 columns)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Win Rate Over Time")
        fig_win_rate = plot_win_rate_over_time(df_performance)
        st.plotly_chart(fig_win_rate, use_container_width=True)

    with col2:
        st.subheader("MAE Over Time")
        fig_mae = plot_mae_over_time(df_performance)
        st.plotly_chart(fig_mae, use_container_width=True)

    st.markdown("---")

    # Additional analytics (2 columns)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Calibration Plot")
        fig_calibration = plot_calibration(df_predictions)
        st.plotly_chart(fig_calibration, use_container_width=True)

    with col2:
        st.subheader("Error Distribution")
        fig_error = plot_error_distribution(df_predictions)
        st.plotly_chart(fig_error, use_container_width=True)

    st.markdown("---")

    # P&L and Edge
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cumulative P&L")
        fig_pnl = plot_pnl_over_time(df_performance)
        st.plotly_chart(fig_pnl, use_container_width=True)

    with col2:
        st.subheader("Edge Distribution")
        fig_edge = plot_edge_distribution(df_predictions)
        st.plotly_chart(fig_edge, use_container_width=True)

    st.markdown("---")

    # Advanced metrics
    st.subheader("📈 Advanced Performance Metrics")
    display_advanced_metrics(df_predictions)

    st.markdown("---")

    # Edge statistics
    display_edge_statistics(df_predictions)

    st.markdown("---")

    # Detailed predictions table
    st.subheader("📋 Recent Predictions")
    df_formatted = format_predictions_table(df_predictions)
    st.dataframe(df_formatted.head(100), use_container_width=True)


def render_game_predictions_page():
    """Render the game predictions page."""
    st.header("🎯 Game Predictions")

    try:
        with get_db_connection() as conn:
            # Get current week games
            df_games = GameQueries.current_week_games(conn, season=2025, week=None)
    except Exception as e:
        st.error(f"Error fetching game data: {str(e)}")
        return

    if df_games.empty:
        st.info("No games found for current week")
        return

    # Display games as cards
    for _, game in df_games.iterrows():
        game_prediction_card(game.to_dict())
        st.markdown("---")


def render_model_status_page():
    """Render the model status and comparison page."""
    st.header("🤖 Model Status")

    try:
        with get_db_connection() as conn:
            # Get model registry
            df_registry = ModelQueries.model_registry(conn)

            # Get model comparison
            df_comparison = PredictionQueries.model_comparison(conn)
    except Exception as e:
        st.error(f"Error fetching model data: {str(e)}")
        return

    # Model comparison chart
    st.subheader("📊 Model Comparison (MAE)")
    fig_models = plot_model_comparison_mae(df_comparison)
    st.plotly_chart(fig_models, use_container_width=True)

    st.markdown("---")

    # Model comparison table
    st.subheader("📋 Model Performance Table")
    df_formatted = format_model_comparison_table(df_comparison)
    st.dataframe(df_formatted, use_container_width=True)

    st.markdown("---")

    # Model registry
    st.subheader("📚 Model Registry")
    st.dataframe(df_registry, use_container_width=True)


def render_system_status_page():
    """Render the system health and status page."""
    st.header("⚙️ System Status")

    # Service health checks
    st.subheader("🏥 Service Health")

    services = [
        {"name": "PostgreSQL", "host": "localhost", "port": 5544},
        {"name": "Redis", "host": "localhost", "port": 6379},
        {"name": "FastAPI", "host": "localhost", "port": 8000},
        {"name": "Prometheus", "host": "localhost", "port": 9090},
        {"name": "Grafana", "host": "localhost", "port": 3000},
    ]

    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]

    for idx, service in enumerate(services):
        with cols[idx % 3]:
            service_health_card(service)

    st.markdown("---")

    # Database statistics
    st.subheader("📊 Database Statistics")

    try:
        with get_db_connection() as conn:
            stats = SystemQueries.database_statistics(conn)
    except Exception as e:
        st.error(f"Error fetching database stats: {str(e)}")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Games", f"{stats.get('total_games', 0):,}")

    with col2:
        st.metric("Total Plays", f"{stats.get('total_plays', 0):,}")

    with col3:
        st.metric("Season Coverage", f"{stats.get('season_min', 0)}-{stats.get('season_max', 0)}")

    with col4:
        st.metric("Total Tables", f"{stats.get('total_tables', 0)}")


def render_data_summary_page():
    """Render the data summary and coverage page."""
    st.header("📊 Data Summary")

    try:
        with get_db_connection() as conn:
            # Get database stats
            stats = SystemQueries.database_statistics(conn)

            # Get data coverage
            df_coverage = GameQueries.data_coverage_by_season_week(conn)
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return

    # Overview metrics
    st.subheader("📈 Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Games", f"{stats.get('total_games', 0):,}")

    with col2:
        st.metric("Total Plays", f"{stats.get('total_plays', 0):,}")

    with col3:
        st.metric("Season Coverage", f"{stats.get('season_min', 0)}-{stats.get('season_max', 0)}")

    st.markdown("---")

    # Data coverage table
    st.subheader("🗓️ Data Coverage by Season/Week")
    st.dataframe(df_coverage, use_container_width=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================


def main():
    """Main application entry point."""
    st.title("🏈 NFL Analytics Monitoring Dashboard")
    st.markdown("Real-time monitoring of prediction performance and model metrics")

    # Sidebar navigation
    st.sidebar.header("Navigation")

    page = st.sidebar.radio(
        "Select Page",
        [
            "📊 Predictions",
            "🎯 Game Predictions",
            "🤖 Model Status",
            "⚙️ System Status",
            "📊 Data Summary",
        ],
        index=0,
    )

    # Route to appropriate page
    if page == "📊 Predictions":
        render_predictions_page()
    elif page == "🎯 Game Predictions":
        render_game_predictions_page()
    elif page == "🤖 Model Status":
        render_model_status_page()
    elif page == "⚙️ System Status":
        render_system_status_page()
    elif page == "📊 Data Summary":
        render_data_summary_page()


if __name__ == "__main__":
    main()
