#!/usr/bin/env python3
"""
Player Props Dashboard

Shows QB passing yards predictions vs sportsbook lines with edge analysis.
"""

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg
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

# Import schema validation
try:
    from dashboard.utils.schema_validator import (
        ValidationError,
        get_common_schemas,
        validate_schema,
    )
    SCHEMA_VALIDATION_AVAILABLE = True
except ImportError:
    SCHEMA_VALIDATION_AVAILABLE = False
    logger.warning("Schema validation not available - operating without validation")

# Database configuration from environment variables
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5544")),
    "dbname": os.getenv("DB_NAME", "devdb01"),
    "user": os.getenv("DB_USER", "dro"),
    "password": os.getenv("DB_PASSWORD")  # Required: Set DB_PASSWORD environment variable
}

# Initialize database connection pool (singleton)
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
            max_size=10
        )
    except Exception as e:
        logger.error(f"Failed to initialize connection pool: {e}")
        return None

@contextmanager
def get_pooled_connection():
    """
    Context manager to get a pooled database connection.

    Falls back to individual connections if pooling unavailable.
    """
    pool = get_db_pool()

    if pool is not None:
        # Use pooled connection
        with pool.get_connection() as conn:
            yield conn
    else:
        # Fallback to individual connection
        logger.debug("Using individual connection (pool unavailable)")
        with get_pooled_connection() as conn:
            yield conn

# Page configuration
st.set_page_config(
    page_title="Player Props - NFL Analytics",
    page_icon="🏈",
    layout="wide"
)

st.title("🏈 Player Props: QB Passing Yards")
st.markdown("**Real-time sportsbook lines vs model predictions with edge analysis**")

# Sidebar filters
st.sidebar.header("Filters")

# Season/Week selector
current_season = st.sidebar.selectbox("Season", [2024, 2023, 2022], index=0)
current_week = st.sidebar.slider("Week", 1, 18, 12)

# Edge filter
min_edge = st.sidebar.slider(
    "Min Edge (yards)",
    min_value=0.0,
    max_value=50.0,
    value=10.0,
    step=5.0,
    help="Only show props with at least this much edge"
)

# Bookmaker filter
bookmakers = st.sidebar.multiselect(
    "Bookmakers",
    ["draftkings", "fanduel", "betmgm", "caesars", "pointsbet"],
    default=["draftkings", "fanduel"]
)

# Refresh button
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()


@st.cache_data(ttl=300)
def fetch_predictions_with_odds(season: int, week: int) -> pd.DataFrame:
    """
    Fetch QB passing predictions with latest sportsbook odds and error handling.

    Args:
        season: Season to query
        week: Week to query

    Returns:
        DataFrame with predictions and odds, or empty DataFrame on error
    """
    query = """
    WITH latest_odds AS (
        SELECT DISTINCT ON (player_name, bookmaker_key)
            player_name,
            line_value,
            over_odds,
            under_odds,
            over_implied_prob,
            under_implied_prob,
            book_hold,
            bookmaker_key,
            bookmaker_title,
            snapshot_at
        FROM prop_lines_history
        WHERE prop_type = 'passing_yards'
          AND snapshot_at >= NOW() - INTERVAL '7 days'
        ORDER BY player_name, bookmaker_key, snapshot_at DESC
    )
    SELECT
        p.player_id,
        p.player_name,
        p.team,
        p.opponent,
        p.is_home,
        p.game_id,
        g.kickoff,
        p.prediction_mean,
        p.prediction_std,
        p.prediction_q10,
        p.prediction_q90,
        p.ensemble_version,
        p.actual_value,
        o.line_value as book_line,
        o.over_odds,
        o.under_odds,
        o.over_implied_prob,
        o.under_implied_prob,
        o.book_hold,
        o.bookmaker_key,
        o.bookmaker_title,
        o.snapshot_at as odds_updated_at,
        -- Calculate edge
        p.prediction_mean - o.line_value as edge,
        -- Probability our prediction is over the line
        CASE
            WHEN p.prediction_std > 0 THEN
                1.0 - (0.5 * (1 + erf((o.line_value - p.prediction_mean) / (p.prediction_std * sqrt(2)))))
            ELSE 0.5
        END as prob_over,
        -- Expected value (simplified)
        CASE
            WHEN o.over_odds > 0 THEN
                (1.0 - (0.5 * (1 + erf((o.line_value - p.prediction_mean) / (p.prediction_std * sqrt(2)))))) * (o.over_odds / 100.0) - (0.5 * (1 + erf((o.line_value - p.prediction_mean) / (p.prediction_std * sqrt(2))))) * 1.0
            ELSE
                (1.0 - (0.5 * (1 + erf((o.line_value - p.prediction_mean) / (p.prediction_std * sqrt(2)))))) * (100.0 / abs(o.over_odds)) - (0.5 * (1 + erf((o.line_value - p.prediction_mean) / (p.prediction_std * sqrt(2))))) * 1.0
        END as expected_value
    FROM ml.predictions p
    JOIN games g ON p.game_id = g.game_id
    LEFT JOIN latest_odds o ON p.player_name = o.player_name
    WHERE p.season = %s
      AND p.week = %s
      AND p.stat_type = 'passing_yards'
      AND p.ensemble_version IS NOT NULL
    ORDER BY ABS(p.prediction_mean - o.line_value) DESC NULLS LAST
    """

    try:
        with get_pooled_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(season, week))

        # Schema validation
        if SCHEMA_VALIDATION_AVAILABLE and not df.empty:
            schemas = get_common_schemas()
            df = validate_schema(df, schemas['predictions_with_odds'], strict=False)

        logger.info(f"Fetched {len(df)} predictions with odds for {season} Week {week}")
        return df

    except psycopg.Error as e:
        st.error(f"Database error fetching predictions: {e}")
        logger.error(f"Database error in fetch_predictions_with_odds: {e}", exc_info=True)
        return pd.DataFrame()

    except ValidationError as e:
        st.warning(f"Schema validation failed for predictions: {e}")
        logger.warning(f"Schema mismatch in fetch_predictions_with_odds: {e}")
        return pd.DataFrame()

    except Exception as e:
        st.error(f"Unexpected error fetching predictions: {e}")
        logger.error(f"Unexpected error in fetch_predictions_with_odds: {e}", exc_info=True)
        return pd.DataFrame()


@st.cache_data(ttl=300)
def fetch_line_movement(player_name: str) -> pd.DataFrame:
    """
    Fetch historical line movement for a player with error handling.

    Args:
        player_name: Player name to query

    Returns:
        DataFrame with line movement history or empty DataFrame on error
    """
    query = """
    SELECT
        snapshot_at,
        line_value,
        over_odds,
        under_odds,
        bookmaker_title
    FROM prop_lines_history
    WHERE player_name = %s
      AND prop_type = 'passing_yards'
      AND snapshot_at >= NOW() - INTERVAL '7 days'
    ORDER BY snapshot_at ASC
    """

    try:
        with get_pooled_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(player_name,))

        # Schema validation
        if SCHEMA_VALIDATION_AVAILABLE and not df.empty:
            schemas = get_common_schemas()
            df = validate_schema(df, schemas['line_movement'], strict=False)

        logger.info(f"Fetched {len(df)} line movement records for {player_name}")
        return df

    except psycopg.Error as e:
        st.error(f"Database error fetching line movement: {e}")
        logger.error(f"Database error in fetch_line_movement: {e}", exc_info=True)
        return pd.DataFrame()

    except ValidationError as e:
        st.warning(f"Schema validation failed for line movement: {e}")
        logger.warning(f"Schema mismatch in fetch_line_movement: {e}")
        return pd.DataFrame()

    except Exception as e:
        st.error(f"Unexpected error fetching line movement: {e}")
        logger.error(f"Unexpected error in fetch_line_movement: {e}", exc_info=True)
        return pd.DataFrame()


# Main content
st.markdown("---")

# Fetch data
with st.spinner("Loading predictions and sportsbook odds..."):
    df_props = fetch_predictions_with_odds(current_season, current_week)

if df_props.empty:
    st.warning(f"No predictions found for {current_season} Week {current_week}")
    st.stop()

# Filter by bookmakers
if bookmakers:
    df_props = df_props[df_props['bookmaker_key'].isin(bookmakers)]

# Filter by edge
if min_edge > 0:
    df_props = df_props[abs(df_props['edge']) >= min_edge]

if df_props.empty:
    st.warning("No props match the selected filters")
    st.stop()

# Summary metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Props",
        len(df_props),
        help="Number of QB passing props with odds available"
    )

with col2:
    avg_edge = df_props['edge'].mean()
    st.metric(
        "Avg Edge",
        f"{avg_edge:.1f} yds",
        help="Average difference between our prediction and sportsbook line"
    )

with col3:
    props_with_edge = len(df_props[abs(df_props['edge']) >= 10])
    st.metric(
        "Props w/ 10+ Edge",
        props_with_edge,
        help="Number of props with at least 10 yards edge"
    )

with col4:
    avg_book_hold = df_props['book_hold'].mean() * 100
    st.metric(
        "Avg Book Hold",
        f"{avg_book_hold:.1f}%",
        help="Average bookmaker profit margin (vig)"
    )

st.markdown("---")

# Edge distribution chart
st.subheader("📊 Edge Distribution")

fig_edge = px.histogram(
    df_props,
    x='edge',
    nbins=30,
    title="Distribution of Edge (Prediction - Sportsbook Line)",
    labels={'edge': 'Edge (yards)', 'count': 'Number of Props'},
    color_discrete_sequence=['#1f77b4']
)
fig_edge.update_layout(showlegend=False, height=300)
st.plotly_chart(fig_edge, use_container_width=True)

st.markdown("---")

# Props table with detailed view
st.subheader("🎯 Player Props with Odds")

# Sort by absolute edge
df_display = df_props.sort_values('edge', key=abs, ascending=False).copy()

# Format for display
df_display['matchup'] = df_display.apply(
    lambda x: f"{x['team']} @ {x['opponent']}" if not x['is_home'] else f"{x['opponent']} @ {x['team']}",
    axis=1
)
df_display['prediction'] = df_display.apply(
    lambda x: f"{x['prediction_mean']:.1f} ± {x['prediction_std']:.1f}",
    axis=1
)
df_display['80_ci'] = df_display.apply(
    lambda x: f"[{x['prediction_q10']:.1f}, {x['prediction_q90']:.1f}]",
    axis=1
)
df_display['odds_display'] = df_display.apply(
    lambda x: f"O {x['over_odds']:+d} / U {x['under_odds']:+d}" if pd.notna(x['over_odds']) else "N/A",
    axis=1
)

# Display each prop as expandable card
for idx, row in df_display.head(20).iterrows():
    with st.expander(
        f"**{row['player_name']}** ({row['team']}) - Edge: {row['edge']:+.1f} yds - {row['bookmaker_title']}",
        expanded=False
    ):
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            st.markdown("**Game Info**")
            st.markdown(f"- Matchup: {row['matchup']}")
            st.markdown(f"- Kickoff: {row['kickoff']}")
            if pd.notna(row['actual_value']):
                st.markdown(f"- **Actual: {row['actual_value']:.1f} yards**")

        with col2:
            st.markdown("**Our Prediction**")
            st.markdown(f"- Mean: **{row['prediction_mean']:.1f} yards**")
            st.markdown(f"- Std Dev: {row['prediction_std']:.1f} yards")
            st.markdown(f"- 80% CI: {row['80_ci']}")
            st.markdown(f"- Model: {row['ensemble_version']}")

        with col3:
            st.markdown("**Sportsbook**")
            if pd.notna(row['book_line']):
                st.markdown(f"- Line: **{row['book_line']:.1f} yards**")
                st.markdown(f"- Odds: {row['odds_display']}")
                st.markdown(f"- Hold: {row['book_hold']*100:.1f}%")
                st.markdown(f"- **Edge: {row['edge']:+.1f} yds**")
            else:
                st.markdown("- No odds available")

        # Show line movement chart if available
        if pd.notna(row['player_name']):
            df_movement = fetch_line_movement(row['player_name'])

            if not df_movement.empty and len(df_movement) > 1:
                st.markdown("**Line Movement (Last 7 Days)**")

                fig_movement = go.Figure()

                for book in df_movement['bookmaker_title'].unique():
                    book_data = df_movement[df_movement['bookmaker_title'] == book]
                    fig_movement.add_trace(go.Scatter(
                        x=book_data['snapshot_at'],
                        y=book_data['line_value'],
                        mode='lines+markers',
                        name=book,
                        line=dict(width=2),
                        marker=dict(size=6)
                    ))

                # Add our prediction as horizontal line
                fig_movement.add_hline(
                    y=row['prediction_mean'],
                    line_dash="dash",
                    line_color="green",
                    annotation_text=f"Our Prediction: {row['prediction_mean']:.1f}",
                    annotation_position="right"
                )

                fig_movement.update_layout(
                    title=f"{row['player_name']} - Line Movement",
                    xaxis_title="Time",
                    yaxis_title="Line (yards)",
                    height=300,
                    showlegend=True,
                    hovermode='x unified'
                )

                st.plotly_chart(fig_movement, use_container_width=True)

st.markdown("---")

# Best edges summary
st.subheader("🔥 Best Edges")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Biggest OVER Edges** (Our prediction >> Sportsbook line)")
    over_edges = df_display[df_display['edge'] > 0].nlargest(5, 'edge')

    if not over_edges.empty:
        for idx, row in over_edges.iterrows():
            st.markdown(
                f"- **{row['player_name']}** ({row['team']}): "
                f"{row['prediction_mean']:.1f} vs {row['book_line']:.1f} "
                f"(+{row['edge']:.1f} edge)"
            )
    else:
        st.info("No significant over edges found")

with col2:
    st.markdown("**Biggest UNDER Edges** (Our prediction << Sportsbook line)")
    under_edges = df_display[df_display['edge'] < 0].nsmallest(5, 'edge')

    if not under_edges.empty:
        for idx, row in under_edges.iterrows():
            st.markdown(
                f"- **{row['player_name']}** ({row['team']}): "
                f"{row['prediction_mean']:.1f} vs {row['book_line']:.1f} "
                f"({row['edge']:.1f} edge)"
            )
    else:
        st.info("No significant under edges found")

# Footer
st.markdown("---")
st.caption(
    "💡 **Note**: Edge = Our Prediction - Sportsbook Line. "
    "Positive edge suggests OVER, negative suggests UNDER. "
    "Always verify lines before placing bets."
)
