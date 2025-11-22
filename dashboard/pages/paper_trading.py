"""
Paper Trading Dashboard Page

Real-time monitoring of paper trading sessions with performance metrics,
trade history, and risk management indicators.

Author: Claude Code
Date: November 6, 2025
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import psycopg
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import connection pooling
try:
    from dashboard.utils.db import get_db_connection, DB_CONFIG
except ImportError:
    # Fallback configuration
    DB_CONFIG = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5544")),
        "dbname": os.getenv("DB_NAME", "devdb01"),
        "user": os.getenv("DB_USER", "dro"),
        "password": os.getenv("DB_PASSWORD", "sicillionbillions")
    }
    def get_db_connection():
        return psycopg.connect(**DB_CONFIG)

st.set_page_config(
    page_title="Paper Trading Monitor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Paper Trading Performance Monitor")

@st.cache_data(ttl=60)  # Refresh every minute
def fetch_active_sessions():
    """Fetch active paper trading sessions."""
    query = """
    SELECT
        session_id,
        started_at,
        ended_at,
        initial_bankroll,
        final_bankroll,
        total_bets,
        win_rate,
        roi,
        sharpe_ratio,
        max_drawdown,
        strategy
    FROM paper_trading_sessions
    WHERE ended_at IS NULL OR ended_at >= NOW() - INTERVAL '7 days'
    ORDER BY started_at DESC
    """

    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn)

    return df

@st.cache_data(ttl=60)
def fetch_session_trades(session_id):
    """Fetch trades for a specific session."""
    query = """
    SELECT
        paper_trade_id,
        trade_timestamp,
        player_name,
        prop_type,
        line_value,
        bet_side,
        odds,
        bet_amount,
        edge,
        model_prediction,
        model_std,
        actual_result,
        outcome,
        profit_loss,
        bookmaker_key
    FROM paper_trades
    WHERE session_id = %s
    ORDER BY trade_timestamp DESC
    """

    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn, params=[session_id])

    return df

@st.cache_data(ttl=60)
def fetch_performance_by_prop_type():
    """Fetch performance breakdown by prop type."""
    query = """
    SELECT * FROM paper_trades_by_prop_type
    """

    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn)

    return df

@st.cache_data(ttl=60)
def fetch_performance_by_bookmaker():
    """Fetch performance breakdown by bookmaker."""
    query = """
    SELECT * FROM paper_trades_by_bookmaker
    """

    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn)

    return df

# Main dashboard
sessions_df = fetch_active_sessions()

if sessions_df.empty:
    st.warning("No active paper trading sessions found.")
    st.info("Start a paper trading session using: `python py/production/paper_trader.py`")
else:
    # Session selector
    session_options = sessions_df['session_id'].tolist()
    selected_session = st.selectbox(
        "Select Paper Trading Session",
        session_options,
        format_func=lambda x: f"{x} ({sessions_df[sessions_df['session_id']==x]['strategy'].iloc[0]})"
    )

    session_data = sessions_df[sessions_df['session_id'] == selected_session].iloc[0]

    # Performance metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        roi = session_data['roi'] if pd.notna(session_data['roi']) else 0.0
        roi_color = "normal" if roi >= 0 else "inverse"
        st.metric(
            "ROI",
            f"{roi:.2%}",
            delta=f"{roi:.2%}",
            delta_color=roi_color
        )

    with col2:
        win_rate = session_data['win_rate'] if pd.notna(session_data['win_rate']) else 0.0
        st.metric(
            "Win Rate",
            f"{win_rate:.1%}",
            delta="Target: 52.4%"
        )

    with col3:
        sharpe = session_data['sharpe_ratio'] if pd.notna(session_data['sharpe_ratio']) else 0.0
        st.metric(
            "Sharpe Ratio",
            f"{sharpe:.2f}",
            delta="Target: >2.0"
        )

    with col4:
        max_dd = session_data['max_drawdown'] if pd.notna(session_data['max_drawdown']) else 0.0
        st.metric(
            "Max Drawdown",
            f"{max_dd:.1%}",
            delta=f"{-max_dd:.1%}",
            delta_color="inverse"
        )

    with col5:
        total_bets = session_data['total_bets'] if pd.notna(session_data['total_bets']) else 0
        st.metric(
            "Total Bets",
            f"{total_bets}",
            delta=f"Active"
        )

    # Bankroll progress
    st.subheader("💰 Bankroll Progression")

    trades_df = fetch_session_trades(selected_session)

    if not trades_df.empty:
        # Calculate cumulative P&L
        settled_trades = trades_df[trades_df['outcome'].isin(['win', 'loss', 'push'])].copy()
        settled_trades = settled_trades.sort_values('trade_timestamp')
        settled_trades['cumulative_pnl'] = settled_trades['profit_loss'].cumsum()
        settled_trades['bankroll'] = session_data['initial_bankroll'] + settled_trades['cumulative_pnl']

        # Plot bankroll progression
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=settled_trades['trade_timestamp'],
            y=settled_trades['bankroll'],
            mode='lines+markers',
            name='Bankroll',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ))

        # Add initial bankroll line
        fig.add_hline(
            y=session_data['initial_bankroll'],
            line_dash="dash",
            line_color="gray",
            annotation_text="Initial Bankroll"
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Bankroll ($)",
            hovermode='x unified',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Trade history table
        st.subheader("📋 Recent Trades")

        # Format trades dataframe for display
        display_df = trades_df.copy()
        display_df['trade_timestamp'] = pd.to_datetime(display_df['trade_timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        display_df['bet_side'] = display_df['bet_side'].str.upper()
        display_df['outcome'] = display_df['outcome'].apply(
            lambda x: {
                'win': '✅ Win',
                'loss': '❌ Loss',
                'push': '➖ Push',
                'pending': '⏳ Pending'
            }.get(x, x)
        )

        # Select columns for display
        display_cols = [
            'trade_timestamp', 'player_name', 'prop_type', 'bet_side',
            'line_value', 'odds', 'bet_amount', 'edge', 'actual_result',
            'outcome', 'profit_loss'
        ]

        st.dataframe(
            display_df[display_cols].head(50),
            use_container_width=True,
            hide_index=True
        )

        # Performance by prop type
        st.subheader("🎯 Performance by Prop Type")

        prop_perf = fetch_performance_by_prop_type()

        if not prop_perf.empty:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("ROI by Prop Type", "Win Rate by Prop Type"),
                specs=[[{"type": "bar"}, {"type": "bar"}]]
            )

            # ROI bars
            fig.add_trace(
                go.Bar(
                    x=prop_perf['prop_type'],
                    y=prop_perf['roi'] * 100,
                    name='ROI (%)',
                    marker_color='#1f77b4'
                ),
                row=1, col=1
            )

            # Win rate bars
            fig.add_trace(
                go.Bar(
                    x=prop_perf['prop_type'],
                    y=prop_perf['win_rate'] * 100,
                    name='Win Rate (%)',
                    marker_color='#2ca02c'
                ),
                row=1, col=2
            )

            # Add break-even line to win rate
            fig.add_hline(
                y=52.4,
                line_dash="dash",
                line_color="red",
                annotation_text="Break-even (52.4%)",
                row=1, col=2
            )

            fig.update_xaxes(title_text="Prop Type", row=1, col=1)
            fig.update_xaxes(title_text="Prop Type", row=1, col=2)
            fig.update_yaxes(title_text="ROI (%)", row=1, col=1)
            fig.update_yaxes(title_text="Win Rate (%)", row=1, col=2)

            fig.update_layout(height=400, showlegend=False)

            st.plotly_chart(fig, use_container_width=True)

        # Performance by bookmaker
        st.subheader("🏦 Performance by Bookmaker")

        bookmaker_perf = fetch_performance_by_bookmaker()

        if not bookmaker_perf.empty:
            col1, col2 = st.columns(2)

            with col1:
                # ROI comparison
                fig = px.bar(
                    bookmaker_perf,
                    x='bookmaker_key',
                    y='roi',
                    title='ROI by Bookmaker',
                    labels={'roi': 'ROI', 'bookmaker_key': 'Bookmaker'},
                    color='roi',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_traces(texttemplate='%{y:.1%}', textposition='outside')
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Volume comparison
                fig = px.bar(
                    bookmaker_perf,
                    x='bookmaker_key',
                    y='total_wagered',
                    title='Total Volume by Bookmaker',
                    labels={'total_wagered': 'Total Wagered ($)', 'bookmaker_key': 'Bookmaker'}
                )
                fig.update_traces(texttemplate='$%{y:,.0f}', textposition='outside')
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No trades found for this session yet.")

# Pending trades alert
with st.expander("⏳ Pending Trades (Awaiting Settlement)"):
    pending_query = """
    SELECT
        paper_trade_id,
        trade_timestamp,
        player_name,
        prop_type,
        line_value,
        bet_side,
        bet_amount,
        bookmaker_key
    FROM paper_trades
    WHERE outcome = 'pending'
    ORDER BY trade_timestamp DESC
    LIMIT 20
    """

    with get_db_connection() as conn:
        pending_df = pd.read_sql_query(pending_query, conn)

    if not pending_df.empty:
        st.warning(f"{len(pending_df)} trades awaiting settlement")
        st.dataframe(pending_df, use_container_width=True, hide_index=True)
        st.info("Run `python py/production/settle_paper_trades.py` to settle pending trades")
    else:
        st.success("No pending trades")

# Auto-refresh toggle
st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("Auto-refresh (60s)", value=False)
if auto_refresh:
    st.sidebar.info("Dashboard will refresh every 60 seconds")
    st.rerun()
