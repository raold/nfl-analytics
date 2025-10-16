#!/usr/bin/env python3
"""
Performance Monitoring Dashboard for NFL Props Models

Real-time tracking of model performance, ROI, and key metrics.
Supports multiple model versions and A/B test monitoring.
"""

import sys
sys.path.append('/Users/dro/rice/nfl-analytics')

import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceDashboard:
    """
    Real-time performance monitoring dashboard for NFL props models.

    Features:
    - Live ROI tracking
    - Model comparison (v1.0 vs v2.5 vs v3.0)
    - Calibration metrics
    - Weekly/monthly performance trends
    - A/B test monitoring
    """

    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'port': 5544,
            'database': 'devdb01',
            'user': 'dro',
            'password': 'sicillionbillions'
        }

    def load_performance_data(
        self,
        start_date: datetime,
        end_date: datetime,
        model_version: Optional[str] = None
    ) -> pd.DataFrame:
        """Load performance data from database"""
        conn = psycopg2.connect(**self.db_config)

        query = """
        SELECT
            p.prediction_date,
            p.player_id,
            p.model_version,
            p.stat_type,
            p.predicted_value,
            p.actual_value,
            p.predicted_line,
            p.odds,
            p.bet_amount,
            p.profit,
            ABS(p.predicted_value - p.actual_value) as abs_error,
            CASE
                WHEN p.predicted_value > p.predicted_line AND p.actual_value > p.predicted_line THEN 1
                WHEN p.predicted_value < p.predicted_line AND p.actual_value < p.predicted_line THEN 1
                ELSE 0
            END as win,
            p.created_at
        FROM predictions.performance_tracking p
        WHERE p.prediction_date BETWEEN %s AND %s
        """

        params = [start_date, end_date]

        if model_version:
            query += " AND p.model_version = %s"
            params.append(model_version)

        query += " ORDER BY p.prediction_date DESC"

        df = pd.read_sql(query, conn, params=params)
        conn.close()

        # Calculate additional metrics
        if not df.empty:
            df['roi'] = df['profit'] / df['bet_amount'] * 100
            df['week'] = pd.to_datetime(df['prediction_date']).dt.isocalendar().week

        return df

    def calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate key performance metrics"""
        if df.empty:
            return {
                'total_bets': 0,
                'total_wagered': 0,
                'total_profit': 0,
                'roi': 0,
                'win_rate': 0,
                'avg_odds': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'mae': 0,
                'calibration_score': 0
            }

        # Basic metrics
        total_bets = len(df)
        total_wagered = df['bet_amount'].sum()
        total_profit = df['profit'].sum()
        roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0
        win_rate = df['win'].mean() * 100

        # Average odds
        avg_odds = df['odds'].mean()

        # Sharpe ratio (daily)
        daily_profits = df.groupby('prediction_date')['profit'].sum()
        if len(daily_profits) > 1:
            sharpe_ratio = (daily_profits.mean() / daily_profits.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # Maximum drawdown
        cumulative = df.sort_values('prediction_date')['profit'].cumsum()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max.abs()
        max_drawdown = drawdown.min() * 100

        # Model accuracy
        mae = df['abs_error'].mean() if 'abs_error' in df.columns else 0

        # Calibration (simplified)
        calibration_score = self._calculate_calibration(df)

        return {
            'total_bets': total_bets,
            'total_wagered': total_wagered,
            'total_profit': total_profit,
            'roi': roi,
            'win_rate': win_rate,
            'avg_odds': avg_odds,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'mae': mae,
            'calibration_score': calibration_score
        }

    def _calculate_calibration(self, df: pd.DataFrame) -> float:
        """Calculate calibration score (0-100, higher is better)"""
        if df.empty or 'predicted_value' not in df.columns:
            return 0

        # Simple calibration: group by confidence bins
        # For props, we can use the predicted probability implied by the line
        # This is simplified - in production would use proper calibration metrics

        return 85.0  # Placeholder - implement proper calibration

    def create_dashboard(self):
        """Create Streamlit dashboard"""
        st.set_page_config(
            page_title="NFL Props Performance Dashboard",
            page_icon="🏈",
            layout="wide"
        )

        st.title("🏈 NFL Props Model Performance Dashboard")
        st.markdown("Real-time monitoring of model performance and betting outcomes")

        # Sidebar controls
        st.sidebar.header("Controls")

        # Date range
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=30)
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now()
            )

        # Model selection
        model_version = st.sidebar.selectbox(
            "Model Version",
            ["All", "hierarchical_v1.0", "informative_priors_v2.5", "ensemble_v3.0"]
        )

        # Refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            st.rerun()

        # Load data
        try:
            df = self.load_performance_data(
                start_date,
                end_date,
                None if model_version == "All" else model_version
            )

            if df.empty:
                st.warning("No data available for selected date range")
                return

            # Calculate metrics
            metrics = self.calculate_metrics(df)

            # Display key metrics
            st.header("📊 Key Metrics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "ROI",
                    f"{metrics['roi']:.2f}%",
                    f"{metrics['roi'] - 1.59:.2f}% vs baseline"
                )

            with col2:
                st.metric(
                    "Win Rate",
                    f"{metrics['win_rate']:.1f}%",
                    f"{metrics['win_rate'] - 55:.1f}% vs baseline"
                )

            with col3:
                st.metric(
                    "Total Profit",
                    f"${metrics['total_profit']:.2f}",
                    f"${metrics['total_profit'] / 30:.2f}/day"
                )

            with col4:
                st.metric(
                    "Sharpe Ratio",
                    f"{metrics['sharpe_ratio']:.2f}",
                    "Risk-adjusted return"
                )

            # Performance over time
            st.header("📈 Performance Trends")

            # Cumulative profit chart
            df_sorted = df.sort_values('prediction_date')
            df_sorted['cumulative_profit'] = df_sorted['profit'].cumsum()

            fig_profit = px.line(
                df_sorted,
                x='prediction_date',
                y='cumulative_profit',
                color='model_version' if model_version == "All" else None,
                title="Cumulative Profit Over Time"
            )
            st.plotly_chart(fig_profit, use_container_width=True)

            # Win rate by week
            weekly_stats = df.groupby(['week', 'model_version' if model_version == "All" else None]).agg({
                'win': 'mean',
                'profit': 'sum',
                'bet_amount': 'sum'
            }).reset_index()

            if not weekly_stats.empty:
                weekly_stats['win_rate'] = weekly_stats['win'] * 100
                weekly_stats['roi'] = weekly_stats['profit'] / weekly_stats['bet_amount'] * 100

                col1, col2 = st.columns(2)

                with col1:
                    fig_winrate = px.bar(
                        weekly_stats,
                        x='week',
                        y='win_rate',
                        color='model_version' if model_version == "All" else None,
                        title="Weekly Win Rate"
                    )
                    st.plotly_chart(fig_winrate, use_container_width=True)

                with col2:
                    fig_roi = px.bar(
                        weekly_stats,
                        x='week',
                        y='roi',
                        color='model_version' if model_version == "All" else None,
                        title="Weekly ROI"
                    )
                    st.plotly_chart(fig_roi, use_container_width=True)

            # Model comparison
            if model_version == "All" and len(df['model_version'].unique()) > 1:
                st.header("🔍 Model Comparison")

                model_metrics = []
                for mv in df['model_version'].unique():
                    df_model = df[df['model_version'] == mv]
                    m = self.calculate_metrics(df_model)
                    m['model_version'] = mv
                    model_metrics.append(m)

                comparison_df = pd.DataFrame(model_metrics)

                # Comparison table
                st.dataframe(
                    comparison_df[['model_version', 'roi', 'win_rate', 'sharpe_ratio', 'mae']],
                    use_container_width=True
                )

                # Radar chart for model comparison
                categories = ['ROI', 'Win Rate', 'Sharpe', 'Calibration']

                fig_radar = go.Figure()

                for _, row in comparison_df.iterrows():
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[
                            row['roi'] / 10,  # Scale to 0-10
                            row['win_rate'] / 10,
                            row['sharpe_ratio'] * 2,
                            row['calibration_score'] / 10
                        ],
                        theta=categories,
                        fill='toself',
                        name=row['model_version']
                    ))

                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 10]
                        )
                    ),
                    title="Model Performance Comparison",
                    showlegend=True
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            # Recent predictions
            st.header("📋 Recent Predictions")
            recent_df = df.head(50)[['prediction_date', 'player_id', 'model_version',
                                     'predicted_value', 'actual_value', 'profit', 'win']]
            st.dataframe(recent_df, use_container_width=True)

            # Risk metrics
            st.header("⚠️ Risk Metrics")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Max Drawdown",
                    f"{metrics['max_drawdown']:.1f}%",
                    "Largest peak-to-trough decline"
                )

            with col2:
                st.metric(
                    "Avg Odds",
                    f"{metrics['avg_odds']:.2f}",
                    "Average betting odds"
                )

            with col3:
                st.metric(
                    "MAE",
                    f"{metrics['mae']:.1f} yards",
                    "Mean absolute error"
                )

        except Exception as e:
            st.error(f"Error loading data: {e}")
            logger.error(f"Dashboard error: {e}")


def create_cli_dashboard():
    """Create a simple CLI version of the dashboard"""
    dashboard = PerformanceDashboard()

    # Load last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    print("\n" + "="*60)
    print("NFL PROPS PERFORMANCE DASHBOARD")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print("="*60)

    for model_version in ["hierarchical_v1.0", "informative_priors_v2.5"]:
        try:
            df = dashboard.load_performance_data(start_date, end_date, model_version)

            if df.empty:
                print(f"\n{model_version}: No data available")
                continue

            metrics = dashboard.calculate_metrics(df)

            print(f"\n{model_version}:")
            print(f"  Total Bets: {metrics['total_bets']}")
            print(f"  ROI: {metrics['roi']:.2f}%")
            print(f"  Win Rate: {metrics['win_rate']:.1f}%")
            print(f"  Total Profit: ${metrics['total_profit']:.2f}")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {metrics['max_drawdown']:.1f}%")
            print(f"  MAE: {metrics['mae']:.1f} yards")

        except Exception as e:
            print(f"\n{model_version}: Error loading data - {e}")

    print("\n" + "="*60)
    print("Use 'streamlit run performance_dashboard.py' for full dashboard")
    print("="*60 + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        # CLI mode
        create_cli_dashboard()
    else:
        # Streamlit mode
        dashboard = PerformanceDashboard()
        dashboard.create_dashboard()