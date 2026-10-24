"""
production_dashboard.py

Streamlit Production Monitoring Dashboard

Real-time monitoring dashboard for NFL betting production system.

Features:
- Overview: Summary metrics, bankroll tracking, recent bets
- Performance: ROI, Sharpe ratio, win rate, CLV charts over time
- Model Comparison: XGBoost vs CQL vs IQL performance
- Thompson Sampling: Current model selection probabilities
- Stress Tests: Bootstrap test results, alerts, health checks
- Bet History: Searchable table of all bets with filters
- Alerts: Real-time alerts and warnings

Usage:
    streamlit run py/production/viz/production_dashboard.py

    # Custom port
    streamlit run py/production/viz/production_dashboard.py --server.port 8502

    # Auto-refresh every 30s
    streamlit run py/production/viz/production_dashboard.py --server.runOnSave true
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import beta

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from py.production.monitor_performance import PerformanceMonitor
from py.production.stress_test_monitor import StressTestMonitor
from py.production.thompson_switch_logic import ThompsonSampler

# ============================================================================
# Configuration
# ============================================================================

st.set_page_config(
    page_title="NFL Betting Dashboard",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_URL = "postgresql://dro:sicillionbillions@localhost:5544/devdb01"

# ============================================================================
# Sidebar
# ============================================================================

st.sidebar.title("🏈 NFL Betting Dashboard")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Performance",
        "Model Comparison",
        "Thompson Sampling",
        "Stress Tests",
        "Bet History",
        "Alerts",
    ],
)

st.sidebar.markdown("---")

# Refresh button
if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
if auto_refresh:
    st.empty().text("Auto-refreshing...")
    import time

    time.sleep(30)
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# Initialize Monitors
# ============================================================================


@st.cache_resource
def get_performance_monitor():
    return PerformanceMonitor(db_url=DB_URL)


@st.cache_resource
def get_stress_test_monitor():
    return StressTestMonitor(db_url=DB_URL)


@st.cache_resource
def get_thompson_sampler():
    return ThompsonSampler(db_url=DB_URL)


perf_monitor = get_performance_monitor()
stress_monitor = get_stress_test_monitor()
thompson = get_thompson_sampler()

# ============================================================================
# PAGE 1: Overview
# ============================================================================

if page == "Overview":
    st.title("📊 Overview")
    st.markdown("---")

    # Get data
    report = perf_monitor.generate_report(period="all")

    if not report:
        st.warning("No betting data available yet. Place your first bet to see statistics.")
        st.stop()

    metrics = report["metrics"]
    bankroll = report["bankroll"]

    # Top-level metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Bankroll",
            f"${bankroll['current']:,.0f}",
            delta=f"${bankroll['growth']:+,.0f}",
        )

    with col2:
        st.metric("ROI", f"{metrics['roi']:+.2%}", delta=None)

    with col3:
        st.metric("Win Rate", f"{metrics['win_rate']:.1%}", delta=None)

    with col4:
        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}", delta=None)

    with col5:
        st.metric("Total Bets", f"{metrics['n_bets']}", delta=None)

    st.markdown("---")

    # Bankroll growth chart
    st.subheader("💰 Bankroll Growth")

    bets = perf_monitor.get_bets()
    if len(bets) > 0:
        bets = bets.sort_values("timestamp")
        bets["cumulative_payout"] = bets["payout"].fillna(0).cumsum()
        bets["bankroll"] = bankroll["initial"] + bets["cumulative_payout"]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=bets["timestamp"],
                y=bets["bankroll"],
                mode="lines",
                name="Bankroll",
                line=dict(color="green", width=2),
            )
        )
        fig.add_hline(
            y=bankroll["initial"],
            line_dash="dash",
            line_color="red",
            annotation_text="Initial Bankroll",
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Bankroll ($)",
            hovermode="x unified",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Recent bets
    st.subheader("📋 Recent Bets (Last 10)")

    recent_bets = bets.head(10)[
        ["timestamp", "game_id", "bet_type", "side", "line", "odds", "stake", "result", "payout"]
    ]

    st.dataframe(recent_bets, use_container_width=True)

# ============================================================================
# PAGE 2: Performance
# ============================================================================

elif page == "Performance":
    st.title("📈 Performance Analytics")
    st.markdown("---")

    # Time period selector
    period = st.selectbox(
        "Select Period",
        ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Season", "All Time"],
    )

    period_map = {
        "Last 7 Days": 7,
        "Last 30 Days": 30,
        "Last 90 Days": 90,
        "Season": 120,  # ~4 months
        "All Time": None,
    }

    days = period_map[period]
    start_date = datetime.now() - timedelta(days=days) if days else None

    bets = perf_monitor.get_bets(start_date=start_date)
    settled = bets[bets["result"].notna()].sort_values("timestamp")

    if len(settled) == 0:
        st.warning(f"No settled bets in {period}")
        st.stop()

    # Calculate metrics
    metrics = perf_monitor.calculate_metrics(settled)

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("ROI", f"{metrics.roi:+.2%}")
    with col2:
        st.metric("Win Rate", f"{metrics.win_rate:.1%}")
    with col3:
        st.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.3f}")
    with col4:
        st.metric("Avg CLV", f"{metrics.avg_clv:+.2f} pts")

    st.markdown("---")

    # ROI over time
    st.subheader("ROI Over Time")

    settled["cumulative_staked"] = settled["stake"].cumsum()
    settled["cumulative_payout"] = settled["payout"].cumsum()
    settled["roi"] = settled["cumulative_payout"] / settled["cumulative_staked"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=settled["timestamp"],
            y=settled["roi"] * 100,
            mode="lines",
            name="ROI",
            line=dict(color="blue", width=2),
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="ROI (%)",
        hovermode="x unified",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Win rate over time (rolling 20-bet window)
    st.subheader("Win Rate (Rolling 20-Bet Average)")

    settled["win"] = (settled["result"] == "win").astype(int)
    settled["rolling_win_rate"] = settled["win"].rolling(window=20, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=settled["timestamp"],
            y=settled["rolling_win_rate"] * 100,
            mode="lines",
            name="Win Rate",
            line=dict(color="green", width=2),
        )
    )
    fig.add_hline(y=52.4, line_dash="dash", line_color="red", annotation_text="Break-even (52.4%)")
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Win Rate (%)",
        hovermode="x unified",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # CLV distribution
    st.subheader("Closing Line Value (CLV) Distribution")

    clv_data = settled[settled["clv"].notna()]["clv"]

    fig = px.histogram(
        clv_data,
        nbins=30,
        title="CLV Distribution",
        labels={"value": "CLV (points)", "count": "Frequency"},
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 3: Model Comparison
# ============================================================================

elif page == "Model Comparison":
    st.title("🤖 Model Comparison")
    st.markdown("---")

    # Get model performance
    bets = perf_monitor.get_bets()

    # TODO: Add model_name column to bets table
    # For now, show placeholder
    st.info("Model comparison requires adding 'model_name' column to bets table. " "Coming soon!")

    # Placeholder charts
    st.subheader("Model Performance (Coming Soon)")
    st.write("- ROI by model (XGBoost vs CQL vs IQL)")
    st.write("- Win rate by model")
    st.write("- Calibration plots by model")
    st.write("- Feature importance (XGBoost)")

# ============================================================================
# PAGE 4: Thompson Sampling
# ============================================================================

elif page == "Thompson Sampling":
    st.title("🎯 Thompson Sampling")
    st.markdown("---")

    # Get model stats
    stats = thompson.get_all_stats()

    # Display current probabilities
    st.subheader("Current Model Selection Probabilities")

    col1, col2, col3 = st.columns(3)

    for i, (_, row) in enumerate(stats.iterrows()):
        with [col1, col2, col3][i]:
            st.metric(
                row["model_name"].upper(),
                f"{row['expected_win_rate']:.1%}",
                delta=f"±{np.sqrt(row['variance']):.2%}",
            )
            st.caption(f"Bets: {row['n_bets']} (W: {row['n_wins']}, L: {row['n_losses']})")

    st.markdown("---")

    # Beta distributions
    st.subheader("Win Probability Distributions")

    x = np.linspace(0, 1, 1000)

    fig = go.Figure()

    for _, row in stats.iterrows():
        y = beta.pdf(x, row["alpha"], row["beta"])
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"{row['model_name']} (α={row['alpha']:.1f}, β={row['beta']:.1f})",
                line=dict(width=2),
            )
        )

    fig.update_layout(
        xaxis_title="Win Probability",
        yaxis_title="Density",
        hovermode="x unified",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Model usage over time
    st.subheader("Model Usage Over Time (Coming Soon)")
    st.write("- Stacked area chart of model selection frequency")
    st.write("- Cumulative regret analysis")

# ============================================================================
# PAGE 5: Stress Tests
# ============================================================================

elif page == "Stress Tests":
    st.title("⚠️ Stress Tests")
    st.markdown("---")

    # Get latest stress test
    tests = stress_monitor.get_stress_test_history(limit=1)

    if len(tests) == 0:
        st.warning("No stress tests found. Run a stress test first.")
        st.stop()

    latest = tests.iloc[0]

    # Status indicator
    if latest["passed"]:
        st.success("✅ Latest stress test PASSED")
    else:
        st.error("🚨 Latest stress test FAILED")

    st.markdown("---")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Actual ROI", f"{latest['actual_roi']:+.2%}")
    with col2:
        st.metric("Bootstrap Mean", f"{latest['bootstrap_mean_roi']:+.2%}")
    with col3:
        st.metric("Percentile Rank", f"{latest['percentile_rank']:.1f}%")
    with col4:
        st.metric("Worst Case (1%)", f"{latest['worst_case_roi']:+.2%}")

    st.markdown("---")

    # Bootstrap distribution
    st.subheader("Bootstrap Distribution")

    # Simulate distribution for visualization
    x = np.linspace(
        latest["worst_case_roi"] - 0.05,
        latest["best_case_roi"] + 0.05,
        1000,
    )

    # Approximate as normal distribution
    y = (
        1
        / (latest["bootstrap_std_roi"] * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * ((x - latest["bootstrap_mean_roi"]) / latest["bootstrap_std_roi"]) ** 2)
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x * 100,
            y=y,
            mode="lines",
            fill="tozeroy",
            name="Bootstrap Distribution",
            line=dict(color="blue", width=2),
        )
    )
    fig.add_vline(
        x=latest["actual_roi"] * 100,
        line_dash="dash",
        line_color="red",
        annotation_text="Actual ROI",
    )
    fig.add_vline(
        x=latest["bootstrap_5th_percentile"] * 100,
        line_dash="dot",
        line_color="orange",
        annotation_text="5th Percentile",
    )
    fig.update_layout(
        xaxis_title="ROI (%)",
        yaxis_title="Density",
        hovermode="x unified",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Health check alerts
    st.subheader("Health Check Alerts")

    alerts = stress_monitor.check_health()

    for alert in alerts:
        if "🚨" in alert:
            st.error(alert)
        elif "⚠️" in alert:
            st.warning(alert)
        else:
            st.success(alert)

# ============================================================================
# PAGE 6: Bet History
# ============================================================================

elif page == "Bet History":
    st.title("📜 Bet History")
    st.markdown("---")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        bet_type_filter = st.multiselect(
            "Bet Type",
            ["spread", "total", "moneyline"],
            default=["spread", "total", "moneyline"],
        )

    with col2:
        result_filter = st.multiselect(
            "Result",
            ["win", "loss", "push", "pending"],
            default=["win", "loss", "push", "pending"],
        )

    with col3:
        date_range = st.selectbox(
            "Date Range",
            ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"],
        )

    # Get bets
    days_map = {
        "Last 7 Days": 7,
        "Last 30 Days": 30,
        "Last 90 Days": 90,
        "All Time": None,
    }
    days = days_map[date_range]
    start_date = datetime.now() - timedelta(days=days) if days else None

    bets = perf_monitor.get_bets(start_date=start_date)

    # Apply filters
    bets = bets[bets["bet_type"].isin(bet_type_filter)]

    result_filter_db = [r if r != "pending" else None for r in result_filter]
    if None in result_filter_db:
        bets = bets[bets["result"].isin(result_filter_db[1:]) | bets["result"].isna()]
    else:
        bets = bets[bets["result"].isin(result_filter_db)]

    # Display table
    st.dataframe(
        bets[
            [
                "timestamp",
                "game_id",
                "bet_type",
                "side",
                "line",
                "odds",
                "stake",
                "prediction",
                "result",
                "payout",
                "clv",
            ]
        ],
        use_container_width=True,
        height=600,
    )

    # Download button
    csv = bets.to_csv(index=False)
    st.download_button(
        "📥 Download as CSV",
        csv,
        "bet_history.csv",
        "text/csv",
        key="download-csv",
    )

# ============================================================================
# PAGE 7: Alerts
# ============================================================================

elif page == "Alerts":
    st.title("🚨 Alerts & Warnings")
    st.markdown("---")

    # Performance alerts
    st.subheader("Performance Alerts")

    perf_alerts = perf_monitor.check_alerts()

    if len(perf_alerts) == 0:
        st.success("✅ No performance alerts")
    else:
        for alert in perf_alerts:
            if "🚨" in alert:
                st.error(alert)
            else:
                st.warning(alert)

    st.markdown("---")

    # Stress test alerts
    st.subheader("Stress Test Alerts")

    stress_alerts = stress_monitor.check_health()

    if len(stress_alerts) == 0:
        st.success("✅ No stress test alerts")
    else:
        for alert in stress_alerts:
            if "🚨" in alert:
                st.error(alert)
            elif "⚠️" in alert:
                st.warning(alert)
            else:
                st.success(alert)
