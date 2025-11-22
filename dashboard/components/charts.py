"""
Dashboard Charts Module

All Plotly chart creation functions for the NFL Analytics Dashboard.
Returns Plotly figures only - no UI code or database queries.

Author: Claude Code
Date: November 22, 2025
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def plot_win_rate_over_time(df: pd.DataFrame) -> go.Figure:
    """
    Create win rate over time chart.

    Args:
        df: DataFrame with columns: season, week, win_rate

    Returns:
        Plotly Figure object
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Create season-week labels
    df = df.copy()
    df['period'] = df['season'].astype(str) + '-W' + df['week'].astype(str)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['period'],
        y=df['win_rate'],
        mode='lines+markers',
        name='Win Rate',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=8)
    ))

    # Add 50% reference line
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                  annotation_text="50% (Break-even)")

    fig.update_layout(
        title="Win Rate Over Time",
        xaxis_title="Season-Week",
        yaxis_title="Win Rate",
        yaxis=dict(tickformat='.1%'),
        hovermode='x unified',
        height=400
    )

    return fig


def plot_mae_over_time(df: pd.DataFrame) -> go.Figure:
    """
    Create MAE over time chart.

    Args:
        df: DataFrame with columns: season, week, mae

    Returns:
        Plotly Figure object
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    df = df.copy()
    df['period'] = df['season'].astype(str) + '-W' + df['week'].astype(str)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['period'],
        y=df['mae'],
        mode='lines+markers',
        name='MAE',
        line=dict(color='#ff7f0e', width=2),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title="Mean Absolute Error Over Time",
        xaxis_title="Season-Week",
        yaxis_title="MAE (yards)",
        hovermode='x unified',
        height=400
    )

    return fig


def plot_ats_accuracy(df: pd.DataFrame) -> go.Figure:
    """
    Create Against-The-Spread (ATS) accuracy chart.

    Args:
        df: DataFrame with columns: predicted_value, actual_value, spread_close

    Returns:
        Plotly Figure object
    """
    if df.empty or 'spread_close' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No ATS data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Calculate prediction errors relative to spread
    df = df.copy()
    df['prediction_error'] = df['predicted_value'] - df['actual_value']
    df['spread_error'] = df['spread_close'] - df['actual_value']
    df['model_beats_spread'] = abs(df['prediction_error']) < abs(df['spread_error'])

    # Group by bins of spread values
    df['spread_bin'] = pd.cut(df['spread_close'], bins=10)
    ats_accuracy = df.groupby('spread_bin')['model_beats_spread'].mean().reset_index()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[str(x) for x in ats_accuracy['spread_bin']],
        y=ats_accuracy['model_beats_spread'],
        marker_color='#2ca02c',
        name='Model ATS Accuracy'
    ))

    fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                  annotation_text="50% (Break-even)")

    fig.update_layout(
        title="Model Accuracy vs Spread (ATS)",
        xaxis_title="Spread Bins",
        yaxis_title="Model Beats Spread %",
        yaxis=dict(tickformat='.1%'),
        height=400
    )

    return fig


def plot_error_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Create prediction error distribution chart.

    Args:
        df: DataFrame with columns: predicted_value, actual_value

    Returns:
        Plotly Figure object
    """
    if df.empty or 'predicted_value' not in df.columns or 'actual_value' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No prediction data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    df = df.copy()
    df['error'] = df['predicted_value'] - df['actual_value']

    fig = go.Figure()

    # Histogram of errors
    fig.add_trace(go.Histogram(
        x=df['error'],
        nbinsx=50,
        name='Error Distribution',
        marker_color='#9467bd',
        opacity=0.7
    ))

    # Add normal distribution overlay
    mean_error = df['error'].mean()
    std_error = df['error'].std()

    fig.add_vline(x=mean_error, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean_error:.2f}")

    fig.update_layout(
        title=f"Prediction Error Distribution (μ={mean_error:.2f}, σ={std_error:.2f})",
        xaxis_title="Prediction Error (yards)",
        yaxis_title="Frequency",
        height=400,
        showlegend=True
    )

    return fig


def plot_calibration(df: pd.DataFrame) -> go.Figure:
    """
    Create calibration plot comparing predicted probabilities to observed frequencies.

    Args:
        df: DataFrame with columns: prob_win, status (or similar probability + outcome columns)

    Returns:
        Plotly Figure object
    """
    if df.empty or 'prob_win' not in df.columns or 'status' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No calibration data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    df = df.copy()
    df['outcome'] = (df['status'] == 'win').astype(int)

    # Bin predictions into deciles
    df['prob_bin'] = pd.cut(df['prob_win'], bins=10, labels=False)

    # Calculate observed frequency per bin
    calibration_data = df.groupby('prob_bin').agg({
        'prob_win': 'mean',
        'outcome': 'mean',
        'status': 'count'
    }).reset_index()
    calibration_data.columns = ['bin', 'predicted_prob', 'observed_freq', 'count']

    fig = go.Figure()

    # Scatter plot: predicted vs observed
    fig.add_trace(go.Scatter(
        x=calibration_data['predicted_prob'],
        y=calibration_data['observed_freq'],
        mode='markers',
        marker=dict(
            size=calibration_data['count'] / 10,  # Size by sample count
            color='#e377c2',
            line=dict(width=1, color='white')
        ),
        name='Calibration',
        text=calibration_data['count'],
        hovertemplate='<b>Predicted: %{x:.2%}</b><br>' +
                      'Observed: %{y:.2%}<br>' +
                      'Count: %{text}<extra></extra>'
    ))

    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Perfect Calibration',
        showlegend=True
    ))

    fig.update_layout(
        title="Prediction Calibration Plot",
        xaxis_title="Predicted Probability",
        yaxis_title="Observed Frequency",
        xaxis=dict(tickformat='.0%', range=[0, 1]),
        yaxis=dict(tickformat='.0%', range=[0, 1]),
        height=500,
        hovermode='closest'
    )

    return fig


def plot_bnn_uncertainty_distribution(df_bnn: pd.DataFrame) -> go.Figure:
    """
    Create BNN uncertainty distribution chart.

    Args:
        df_bnn: DataFrame with columns: epistemic_uncertainty, aleatoric_uncertainty

    Returns:
        Plotly Figure object
    """
    if df_bnn.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No BNN uncertainty data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    fig = go.Figure()

    # Epistemic uncertainty
    if 'epistemic_uncertainty' in df_bnn.columns:
        fig.add_trace(go.Histogram(
            x=df_bnn['epistemic_uncertainty'],
            name='Epistemic (Model)',
            opacity=0.7,
            marker_color='#8c564b'
        ))

    # Aleatoric uncertainty
    if 'aleatoric_uncertainty' in df_bnn.columns:
        fig.add_trace(go.Histogram(
            x=df_bnn['aleatoric_uncertainty'],
            name='Aleatoric (Data)',
            opacity=0.7,
            marker_color='#bcbd22'
        ))

    fig.update_layout(
        title="BNN Uncertainty Distribution",
        xaxis_title="Uncertainty",
        yaxis_title="Frequency",
        barmode='overlay',
        height=400
    )

    return fig


def plot_bnn_prediction_intervals(df_bnn: pd.DataFrame) -> go.Figure:
    """
    Create BNN prediction intervals chart.

    Args:
        df_bnn: DataFrame with columns: game_id, predicted_margin, ci_lower, ci_upper, actual_margin

    Returns:
        Plotly Figure object
    """
    if df_bnn.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No BNN interval data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Limit to most recent games for readability
    df_plot = df_bnn.head(20).copy()

    fig = go.Figure()

    # Prediction intervals (error bars)
    if all(col in df_plot.columns for col in ['ci_lower', 'ci_upper', 'predicted_margin']):
        fig.add_trace(go.Scatter(
            x=list(range(len(df_plot))),
            y=df_plot['predicted_margin'],
            error_y=dict(
                type='data',
                symmetric=False,
                array=df_plot['ci_upper'] - df_plot['predicted_margin'],
                arrayminus=df_plot['predicted_margin'] - df_plot['ci_lower']
            ),
            mode='markers',
            marker=dict(size=8, color='#17becf'),
            name='Predicted ± CI'
        ))

    # Actual outcomes
    if 'actual_margin' in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=list(range(len(df_plot))),
            y=df_plot['actual_margin'],
            mode='markers',
            marker=dict(size=10, symbol='x', color='red'),
            name='Actual'
        ))

    fig.update_layout(
        title="BNN Prediction Intervals (Most Recent 20 Games)",
        xaxis_title="Game Index",
        yaxis_title="Margin",
        height=400,
        hovermode='x unified'
    )

    return fig


def plot_pnl_over_time(df: pd.DataFrame) -> go.Figure:
    """
    Create cumulative P&L over time chart.

    Args:
        df: DataFrame with columns: season, week, weekly_pnl (or similar)

    Returns:
        Plotly Figure object
    """
    if df.empty or 'weekly_pnl' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No P&L data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    df = df.copy()
    df['period'] = df['season'].astype(str) + '-W' + df['week'].astype(str)
    df['cumulative_pnl'] = df['weekly_pnl'].cumsum()

    fig = go.Figure()

    # Cumulative P&L line
    fig.add_trace(go.Scatter(
        x=df['period'],
        y=df['cumulative_pnl'],
        mode='lines+markers',
        name='Cumulative P&L',
        line=dict(color='#2ca02c', width=2),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor='rgba(44, 160, 44, 0.1)'
    ))

    # Break-even line
    fig.add_hline(y=0, line_dash="dash", line_color="gray",
                  annotation_text="Break-even")

    fig.update_layout(
        title="Cumulative Profit & Loss Over Time",
        xaxis_title="Season-Week",
        yaxis_title="Cumulative P&L ($)",
        hovermode='x unified',
        height=400
    )

    return fig


def plot_model_comparison_mae(df: pd.DataFrame) -> go.Figure:
    """
    Create model comparison chart (MAE).

    Args:
        df: DataFrame with columns: model_name, version, mae

    Returns:
        Plotly Figure object
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No model comparison data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    df = df.copy()
    df['model_label'] = df['model_name'] + ' ' + df['version'].astype(str)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df['model_label'],
        y=df['mae'],
        marker_color='#d62728',
        text=df['mae'].round(2),
        textposition='auto'
    ))

    fig.update_layout(
        title="Model Comparison: Mean Absolute Error",
        xaxis_title="Model",
        yaxis_title="MAE (yards)",
        height=400
    )

    return fig


def plot_edge_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Create edge distribution chart.

    Args:
        df: DataFrame with column: edge

    Returns:
        Plotly Figure object
    """
    if df.empty or 'edge' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No edge data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df['edge'],
        nbinsx=50,
        marker_color='#7f7f7f',
        opacity=0.7,
        name='Edge Distribution'
    ))

    # Mean edge line
    mean_edge = df['edge'].mean()
    fig.add_vline(x=mean_edge, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean_edge:.4f}")

    fig.update_layout(
        title=f"Betting Edge Distribution (μ={mean_edge:.4f})",
        xaxis_title="Edge",
        yaxis_title="Frequency",
        height=400
    )

    return fig
