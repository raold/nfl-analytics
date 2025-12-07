#!/usr/bin/env python3
"""
Game Predictions Dashboard

Shows game-level predictions including:
- Spread predictions vs Vegas lines
- Total (O/U) predictions
- Win probability (moneyline) predictions
- Edge analysis and betting recommendations

Consumes the /predict/week and /predict/game/full API endpoints.
"""

import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg
import requests
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import connection pool for psycopg3
try:
    from py.utils.psycopg_pool import get_connection_pool

    POOL_AVAILABLE = True
except ImportError:
    POOL_AVAILABLE = False
    logger.warning("Connection pooling unavailable - falling back to individual connections")

# Database configuration from environment variables
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5544")),
    "dbname": os.getenv("DB_NAME", "devdb01"),
    "user": os.getenv("DB_USER", "dro"),
    "password": os.getenv("DB_PASSWORD", "sicillionbillions"),  # pragma: allowlist secret
}

# API configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(page_title="Game Predictions - NFL Analytics", page_icon="🏈", layout="wide")


# ============================================================================
# Database Connection
# ============================================================================


@st.cache_resource
def get_db_pool():
    """Get shared database connection pool."""
    if not POOL_AVAILABLE:
        logger.warning("Connection pool unavailable - falling back to individual connections")
        return None

    try:
        return get_connection_pool(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            dbname=DB_CONFIG["dbname"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            min_size=2,
            max_size=10,
        )
    except Exception as e:
        logger.error(f"Failed to initialize connection pool: {e}")
        return None


@contextmanager
def get_pooled_connection():
    """Context manager to get a pooled database connection."""
    pool = get_db_pool()

    if pool is not None:
        with pool.get_connection() as conn:
            yield conn
    else:
        # Fallback to individual connection
        logger.debug("Using individual connection (pool unavailable)")
        conn = psycopg.connect(**DB_CONFIG)
        try:
            yield conn
        finally:
            conn.close()


# ============================================================================
# Data Fetching Functions
# ============================================================================


@st.cache_data(ttl=300)  # 5-minute cache
def get_available_weeks():
    """Get list of available weeks for selection."""
    try:
        with get_pooled_connection() as conn:
            with conn.cursor() as cursor:
                # Get current season dynamically
                cursor.execute(
                    """
                    SELECT EXTRACT(YEAR FROM NOW())::int as current_year
                """
                )
                current_year = cursor.fetchone()[0]

                # Get all available season/week combinations
                cursor.execute(
                    """
                    SELECT DISTINCT season, week
                    FROM games
                    WHERE season >= %s - 5  -- Last 5 seasons
                    ORDER BY season DESC, week DESC
                """,
                    (current_year,),
                )
                rows = cursor.fetchall()

                return [(row[0], row[1]) for row in rows]
    except Exception as e:
        logger.error(f"Error fetching available weeks: {e}")
        return []


@st.cache_data(ttl=60)  # 1-minute cache for fresher data
def get_week_games(season: int, week: int):
    """Fetch all games for a given week from the database."""
    try:
        with get_pooled_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        game_id,
                        season,
                        week,
                        home_team,
                        away_team,
                        kickoff,
                        spread_close,
                        total_close,
                        home_moneyline,
                        away_moneyline,
                        home_score,
                        away_score,
                        CASE
                            WHEN home_score IS NOT NULL AND away_score IS NOT NULL
                            THEN 'final'
                            ELSE 'scheduled'
                        END as status
                    FROM games
                    WHERE season = %s AND week = %s
                    ORDER BY kickoff
                """,
                    (season, week),
                )

                columns = [
                    "game_id",
                    "season",
                    "week",
                    "home_team",
                    "away_team",
                    "kickoff",
                    "spread_close",
                    "total_close",
                    "home_moneyline",
                    "away_moneyline",
                    "home_score",
                    "away_score",
                    "status",
                ]
                rows = cursor.fetchall()

                if not rows:
                    return pd.DataFrame()

                return pd.DataFrame(rows, columns=columns)
    except Exception as e:
        logger.error(f"Error fetching week games: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)  # 5-minute cache
def get_stored_predictions(season: int, week: int):
    """Fetch stored predictions from database."""
    try:
        with get_pooled_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        gp.game_id,
                        gp.predicted_home_score,
                        gp.predicted_away_score,
                        gp.predicted_total,
                        gp.predicted_spread,
                        gp.predicted_home_win_prob,
                        gp.spread_std,
                        gp.total_std,
                        gp.spread_edge,
                        gp.total_edge,
                        gp.ml_edge,
                        gp.predicted_at,
                        mr.version as ensemble_version
                    FROM ml.game_predictions gp
                    LEFT JOIN ml.model_registry mr ON gp.model_id = mr.model_id
                    WHERE gp.season = %s AND gp.week = %s
                    ORDER BY gp.predicted_at DESC
                """,
                    (season, week),
                )

                columns = [
                    "game_id",
                    "predicted_home_score",
                    "predicted_away_score",
                    "predicted_total",
                    "predicted_spread",
                    "predicted_home_win_prob",
                    "spread_std",
                    "total_std",
                    "spread_edge",
                    "total_edge",
                    "ml_edge",
                    "predicted_at",
                    "ensemble_version",
                ]
                rows = cursor.fetchall()

                if not rows:
                    return pd.DataFrame()

                df = pd.DataFrame(rows, columns=columns)
                # Keep only the most recent prediction per game
                df = df.drop_duplicates(subset=["game_id"], keep="first")
                return df
    except Exception as e:
        logger.error(f"Error fetching stored predictions: {e}")
        return pd.DataFrame()


def generate_week_predictions(season: int, week: int):
    """Generate predictions for all games in a week via API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/week",
            json={"season": season, "week": week},
            timeout=120,  # 2 minute timeout for batch prediction
        )

        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"API error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        return None


# ============================================================================
# Helper Functions
# ============================================================================


def american_odds_to_implied_prob(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds is None:
        return None
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def get_edge_color(edge: float, threshold: float = 2.0) -> str:
    """Get color based on edge magnitude."""
    if edge is None:
        return "gray"
    if abs(edge) >= threshold:
        return "green" if edge > 0 else "red"
    return "gray"


def format_spread(spread: float, for_team: str = "home") -> str:
    """Format spread for display."""
    if spread is None:
        return "N/A"
    if spread > 0:
        return f"+{spread:.1f}"
    return f"{spread:.1f}"


def get_bet_recommendation(edge: float, std: float, edge_threshold: float = 2.0) -> tuple:
    """Get betting recommendation based on edge and uncertainty."""
    if edge is None or std is None:
        return "No Bet", "gray", 0

    # Edge must exceed threshold AND be > 0.5 std deviations
    confidence = abs(edge) / std if std > 0 else 0

    if abs(edge) >= edge_threshold and confidence > 0.5:
        if edge > 0:
            return "BET HOME", "green", confidence
        else:
            return "BET AWAY", "green", confidence
    return "No Bet", "gray", confidence


# ============================================================================
# Visualization Functions
# ============================================================================


def create_spread_comparison_chart(games_df: pd.DataFrame, predictions_df: pd.DataFrame):
    """Create spread comparison chart: Model vs Vegas."""
    # Merge games with predictions
    merged = games_df.merge(predictions_df, on="game_id", how="inner")

    if merged.empty:
        return None

    fig = go.Figure()

    # Vegas spread
    fig.add_trace(
        go.Bar(
            name="Vegas Spread",
            x=merged["away_team"] + " @ " + merged["home_team"],
            y=merged["spread_close"],
            marker_color="lightblue",
            text=merged["spread_close"].apply(lambda x: f"{x:+.1f}" if pd.notna(x) else "N/A"),
            textposition="outside",
        )
    )

    # Model spread
    fig.add_trace(
        go.Bar(
            name="Model Spread",
            x=merged["away_team"] + " @ " + merged["home_team"],
            y=merged["predicted_spread"],
            marker_color="darkblue",
            text=merged["predicted_spread"].apply(lambda x: f"{x:+.1f}" if pd.notna(x) else "N/A"),
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Spread Comparison: Model vs Vegas",
        barmode="group",
        xaxis_title="Game",
        yaxis_title="Spread (Home Favored = Negative)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig


def create_total_comparison_chart(games_df: pd.DataFrame, predictions_df: pd.DataFrame):
    """Create total (O/U) comparison chart."""
    merged = games_df.merge(predictions_df, on="game_id", how="inner")

    if merged.empty:
        return None

    fig = go.Figure()

    # Vegas total
    fig.add_trace(
        go.Bar(
            name="Vegas Total",
            x=merged["away_team"] + " @ " + merged["home_team"],
            y=merged["total_close"],
            marker_color="lightsalmon",
            text=merged["total_close"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"),
            textposition="outside",
        )
    )

    # Model total
    fig.add_trace(
        go.Bar(
            name="Model Total",
            x=merged["away_team"] + " @ " + merged["home_team"],
            y=merged["predicted_total"],
            marker_color="darkred",
            text=merged["predicted_total"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"),
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Total (O/U) Comparison: Model vs Vegas",
        barmode="group",
        xaxis_title="Game",
        yaxis_title="Total Points",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig


def create_edge_scatter_plot(predictions_df: pd.DataFrame):
    """Create scatter plot showing spread edge vs total edge."""
    if predictions_df.empty:
        return None

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=predictions_df["spread_edge"],
            y=predictions_df["total_edge"],
            mode="markers+text",
            text=predictions_df["game_id"].apply(lambda x: x.split("_")[2] + "@" + x.split("_")[3]),
            textposition="top center",
            marker=dict(
                size=12,
                color=predictions_df["ml_edge"].fillna(0),
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="ML Edge"),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Spread Edge: %{x:.1f}<br>"
                "Total Edge: %{y:.1f}<br>"
                "<extra></extra>"
            ),
        )
    )

    # Add threshold lines
    fig.add_hline(y=2, line_dash="dash", line_color="gray", annotation_text="Over Threshold")
    fig.add_hline(y=-2, line_dash="dash", line_color="gray", annotation_text="Under Threshold")
    fig.add_vline(x=2, line_dash="dash", line_color="gray", annotation_text="Home Threshold")
    fig.add_vline(x=-2, line_dash="dash", line_color="gray", annotation_text="Away Threshold")

    fig.update_layout(
        title="Edge Analysis: Spread vs Total",
        xaxis_title="Spread Edge (+ = Value on Home)",
        yaxis_title="Total Edge (+ = Value on Over)",
        height=500,
    )

    return fig


def create_win_probability_chart(games_df: pd.DataFrame, predictions_df: pd.DataFrame):
    """Create win probability comparison chart."""
    merged = games_df.merge(predictions_df, on="game_id", how="inner")

    if merged.empty:
        return None

    # Calculate Vegas implied probability
    merged["vegas_home_prob"] = merged["home_moneyline"].apply(american_odds_to_implied_prob)

    fig = go.Figure()

    game_labels = merged["away_team"] + " @ " + merged["home_team"]

    # Model probability
    fig.add_trace(
        go.Bar(
            name="Model Win Prob",
            x=game_labels,
            y=merged["predicted_home_win_prob"] * 100,
            marker_color="darkgreen",
            text=merged["predicted_home_win_prob"].apply(
                lambda x: f"{x*100:.0f}%" if pd.notna(x) else "N/A"
            ),
            textposition="outside",
        )
    )

    # Vegas probability
    fig.add_trace(
        go.Bar(
            name="Vegas Win Prob",
            x=game_labels,
            y=merged["vegas_home_prob"] * 100,
            marker_color="lightgreen",
            text=merged["vegas_home_prob"].apply(
                lambda x: f"{x*100:.0f}%" if pd.notna(x) else "N/A"
            ),
            textposition="outside",
        )
    )

    # Add 50% line
    fig.add_hline(y=50, line_dash="dash", line_color="gray")

    fig.update_layout(
        title="Home Team Win Probability: Model vs Vegas",
        barmode="group",
        xaxis_title="Game",
        yaxis_title="Win Probability (%)",
        yaxis_range=[0, 100],
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig


# ============================================================================
# Main Dashboard
# ============================================================================


def main():
    st.title("Game Predictions Dashboard")
    st.markdown("**Model-based spread, total, and moneyline predictions vs Vegas lines**")

    # Sidebar controls
    with st.sidebar:
        st.header("Settings")

        # Week selection
        available_weeks = get_available_weeks()
        if not available_weeks:
            st.error("Unable to load available weeks from database")
            return

        # Get current season/week as default
        current_year = datetime.now().year
        current_month = datetime.now().month
        # NFL season runs Sept-Feb, so adjust year accordingly
        if current_month < 9:
            current_year - 1
        else:
            pass

        # Find available seasons
        seasons = sorted(set([w[0] for w in available_weeks]), reverse=True)
        selected_season = st.selectbox("Season", seasons, index=0 if seasons else 0)

        # Filter weeks for selected season
        weeks_for_season = sorted(
            [w[1] for w in available_weeks if w[0] == selected_season], reverse=True
        )
        selected_week = st.selectbox("Week", weeks_for_season, index=0 if weeks_for_season else 0)

        st.divider()

        # Edge thresholds
        spread_threshold = st.slider("Spread Edge Threshold", 1.0, 5.0, 2.0, 0.5)
        total_threshold = st.slider("Total Edge Threshold", 1.0, 5.0, 2.0, 0.5)

        st.divider()

        # Generate predictions button
        if st.button("Generate Fresh Predictions", type="primary"):
            with st.spinner(
                f"Generating predictions for {selected_season} Week {selected_week}..."
            ):
                result = generate_week_predictions(selected_season, selected_week)
                if result:
                    st.success(
                        f"Generated {result['total_games']} predictions in {result['processing_ms']:.0f}ms"
                    )
                    st.cache_data.clear()  # Clear cache to show new predictions
                else:
                    st.error("Failed to generate predictions. Check API connection.")

        # API status
        st.divider()
        st.subheader("API Status")
        try:
            health = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if health.status_code == 200:
                st.success("API Connected")
            else:
                st.warning(f"API returned {health.status_code}")
        except:
            st.error("API Offline")

    # Main content area
    # Load data
    with st.spinner("Loading game data..."):
        games_df = get_week_games(selected_season, selected_week)
        predictions_df = get_stored_predictions(selected_season, selected_week)

    if games_df.empty:
        st.warning(f"No games found for {selected_season} Week {selected_week}")
        return

    # Summary metrics
    st.header(f"{selected_season} Week {selected_week} Predictions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Games", len(games_df))

    with col2:
        if not predictions_df.empty:
            spread_bets = len(
                predictions_df[predictions_df["spread_edge"].abs() >= spread_threshold]
            )
            st.metric("Spread Bets", spread_bets)
        else:
            st.metric("Spread Bets", "N/A")

    with col3:
        if not predictions_df.empty:
            total_bets = len(predictions_df[predictions_df["total_edge"].abs() >= total_threshold])
            st.metric("Total Bets", total_bets)
        else:
            st.metric("Total Bets", "N/A")

    with col4:
        if not predictions_df.empty:
            st.metric("Predictions Available", len(predictions_df))
        else:
            st.metric("Predictions Available", 0)

    # Check if predictions exist
    if predictions_df.empty:
        st.info(
            "No predictions stored for this week. Click 'Generate Fresh Predictions' in the sidebar."
        )

        # Still show game schedule
        st.subheader("Game Schedule")
        schedule_df = games_df[
            ["game_id", "away_team", "home_team", "kickoff", "spread_close", "total_close"]
        ].copy()
        schedule_df["matchup"] = schedule_df["away_team"] + " @ " + schedule_df["home_team"]
        schedule_df = schedule_df[["matchup", "kickoff", "spread_close", "total_close"]]
        schedule_df.columns = ["Matchup", "Kickoff", "Vegas Spread", "Vegas Total"]
        st.dataframe(schedule_df, use_container_width=True)
        return

    # Tab layout for different views
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Game Cards", "Charts", "Edge Analysis", "Betting Recommendations"]
    )

    # Tab 1: Game Cards
    with tab1:
        st.subheader("Game Predictions")

        # Merge for display
        merged_df = games_df.merge(predictions_df, on="game_id", how="left")

        # Display games in a grid (3 columns)
        cols = st.columns(3)
        for idx, row in merged_df.iterrows():
            col_idx = idx % 3
            with cols[col_idx]:
                # Game card
                with st.container(border=True):
                    st.markdown(f"**{row['away_team']} @ {row['home_team']}**")

                    if pd.isna(row.get("predicted_spread")):
                        st.caption("Prediction not available")
                        continue

                    # Scores if game is final
                    if row["status"] == "final":
                        st.markdown(
                            f"**Final: {row['away_team']} {int(row['away_score'])} - {row['home_team']} {int(row['home_score'])}**"
                        )

                    # Spread comparison
                    spread_edge = row.get("spread_edge")
                    spread_color = get_edge_color(spread_edge, spread_threshold)
                    st.markdown(
                        f"**Spread:** Model {row['predicted_spread']:+.1f} vs Vegas {row['spread_close']:+.1f if pd.notna(row['spread_close']) else 'N/A'}"
                    )
                    if spread_edge is not None:
                        st.markdown(f"Edge: :{spread_color}[{spread_edge:+.1f}]")

                    # Total comparison
                    total_edge = row.get("total_edge")
                    total_color = get_edge_color(total_edge, total_threshold)
                    st.markdown(
                        f"**Total:** Model {row['predicted_total']:.1f} vs Vegas {row['total_close']:.1f if pd.notna(row['total_close']) else 'N/A'}"
                    )
                    if total_edge is not None:
                        st.markdown(f"Edge: :{total_color}[{total_edge:+.1f}]")

                    # Win probability
                    win_prob = row.get("predicted_home_win_prob")
                    if win_prob is not None:
                        st.progress(win_prob, text=f"{row['home_team']} Win: {win_prob*100:.0f}%")

    # Tab 2: Charts
    with tab2:
        # Spread comparison chart
        spread_chart = create_spread_comparison_chart(games_df, predictions_df)
        if spread_chart:
            st.plotly_chart(spread_chart, use_container_width=True)

        # Total comparison chart
        total_chart = create_total_comparison_chart(games_df, predictions_df)
        if total_chart:
            st.plotly_chart(total_chart, use_container_width=True)

        # Win probability chart
        win_prob_chart = create_win_probability_chart(games_df, predictions_df)
        if win_prob_chart:
            st.plotly_chart(win_prob_chart, use_container_width=True)

    # Tab 3: Edge Analysis
    with tab3:
        st.subheader("Edge Distribution")

        # Edge scatter plot
        edge_scatter = create_edge_scatter_plot(predictions_df)
        if edge_scatter:
            st.plotly_chart(edge_scatter, use_container_width=True)

        # Edge histograms
        col1, col2 = st.columns(2)

        with col1:
            if "spread_edge" in predictions_df.columns:
                fig_spread_hist = px.histogram(
                    predictions_df,
                    x="spread_edge",
                    nbins=20,
                    title="Spread Edge Distribution",
                    labels={"spread_edge": "Spread Edge"},
                )
                fig_spread_hist.add_vline(x=spread_threshold, line_dash="dash", line_color="green")
                fig_spread_hist.add_vline(x=-spread_threshold, line_dash="dash", line_color="red")
                st.plotly_chart(fig_spread_hist, use_container_width=True)

        with col2:
            if "total_edge" in predictions_df.columns:
                fig_total_hist = px.histogram(
                    predictions_df,
                    x="total_edge",
                    nbins=20,
                    title="Total Edge Distribution",
                    labels={"total_edge": "Total Edge"},
                )
                fig_total_hist.add_vline(x=total_threshold, line_dash="dash", line_color="green")
                fig_total_hist.add_vline(x=-total_threshold, line_dash="dash", line_color="red")
                st.plotly_chart(fig_total_hist, use_container_width=True)

    # Tab 4: Betting Recommendations
    with tab4:
        st.subheader("Betting Recommendations")
        st.caption(f"Based on edge threshold: Spread {spread_threshold}+, Total {total_threshold}+")

        # Merge for recommendations
        merged_df = games_df.merge(predictions_df, on="game_id", how="inner")

        if merged_df.empty:
            st.info("No predictions available for recommendations")
        else:
            # Spread bets
            st.markdown("### Spread Bets")
            spread_bets = merged_df[merged_df["spread_edge"].abs() >= spread_threshold].copy()

            if len(spread_bets) == 0:
                st.info(f"No spread bets meet the {spread_threshold}+ edge threshold")
            else:
                for _, row in spread_bets.iterrows():
                    matchup = f"{row['away_team']} @ {row['home_team']}"
                    edge = row["spread_edge"]

                    if edge > 0:
                        bet_side = row["home_team"]
                        bet_type = "HOME"
                    else:
                        bet_side = row["away_team"]
                        bet_type = "AWAY"

                    confidence = abs(edge) / row["spread_std"] if row["spread_std"] > 0 else 0

                    with st.container(border=True):
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.markdown(f"**{matchup}**")
                            st.markdown(f"Bet: **{bet_side}** ({bet_type})")
                        with col2:
                            st.metric("Edge", f"{edge:+.1f}")
                        with col3:
                            st.metric("Confidence", f"{confidence:.2f}x STD")

            # Total bets
            st.markdown("### Total (O/U) Bets")
            total_bets = merged_df[merged_df["total_edge"].abs() >= total_threshold].copy()

            if len(total_bets) == 0:
                st.info(f"No total bets meet the {total_threshold}+ edge threshold")
            else:
                for _, row in total_bets.iterrows():
                    matchup = f"{row['away_team']} @ {row['home_team']}"
                    edge = row["total_edge"]

                    bet_side = "OVER" if edge > 0 else "UNDER"
                    confidence = abs(edge) / row["total_std"] if row["total_std"] > 0 else 0

                    with st.container(border=True):
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.markdown(f"**{matchup}**")
                            st.markdown(f"Bet: **{bet_side}** {row['total_close']:.1f}")
                        with col2:
                            st.metric("Edge", f"{edge:+.1f}")
                        with col3:
                            st.metric("Confidence", f"{confidence:.2f}x STD")

            # Summary table
            st.markdown("### All Predictions")
            display_df = merged_df[
                [
                    "away_team",
                    "home_team",
                    "spread_close",
                    "predicted_spread",
                    "spread_edge",
                    "total_close",
                    "predicted_total",
                    "total_edge",
                    "predicted_home_win_prob",
                ]
            ].copy()
            display_df.columns = [
                "Away",
                "Home",
                "Vegas Spread",
                "Model Spread",
                "Spread Edge",
                "Vegas Total",
                "Model Total",
                "Total Edge",
                "Home Win %",
            ]
            display_df["Home Win %"] = (display_df["Home Win %"] * 100).round(1)

            # Format numeric columns
            for col in [
                "Vegas Spread",
                "Model Spread",
                "Spread Edge",
                "Vegas Total",
                "Model Total",
                "Total Edge",
            ]:
                display_df[col] = display_df[col].round(1)

            st.dataframe(
                display_df.style.applymap(
                    lambda x: (
                        "background-color: lightgreen"
                        if isinstance(x, (int, float)) and x >= spread_threshold
                        else (
                            "background-color: lightcoral"
                            if isinstance(x, (int, float)) and x <= -spread_threshold
                            else ""
                        )
                    ),
                    subset=["Spread Edge"],
                ).applymap(
                    lambda x: (
                        "background-color: lightgreen"
                        if isinstance(x, (int, float)) and x >= total_threshold
                        else (
                            "background-color: lightcoral"
                            if isinstance(x, (int, float)) and x <= -total_threshold
                            else ""
                        )
                    ),
                    subset=["Total Edge"],
                ),
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
