"""
Dashboard Metrics Module

All metric calculation and display functions for the NFL Analytics Dashboard.
Handles statistical calculations, performance metrics, and ROI computations.

Author: Claude Code
Date: November 22, 2025
"""

import logging

import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


def calculate_win_rate(df: pd.DataFrame, status_column: str = "status") -> float:
    """
    Calculate win rate from predictions.

    Args:
        df: DataFrame with status column
        status_column: Name of status column (default: 'status')

    Returns:
        Win rate as float (0.0 to 1.0)
    """
    if df.empty or status_column not in df.columns:
        return 0.0

    wins = (df[status_column] == "win").sum()
    total = len(df)

    return wins / total if total > 0 else 0.0


def calculate_mae(
    df: pd.DataFrame, predicted_column: str = "predicted_value", actual_column: str = "actual_value"
) -> float:
    """
    Calculate Mean Absolute Error.

    Args:
        df: DataFrame with predicted and actual columns
        predicted_column: Name of predicted values column
        actual_column: Name of actual values column

    Returns:
        MAE as float
    """
    if df.empty or predicted_column not in df.columns or actual_column not in df.columns:
        return 0.0

    # Filter out NaN values
    valid_rows = df[[predicted_column, actual_column]].dropna()
    if valid_rows.empty:
        return 0.0

    mae = (valid_rows[predicted_column] - valid_rows[actual_column]).abs().mean()
    return float(mae)


def calculate_rmse(
    df: pd.DataFrame, predicted_column: str = "predicted_value", actual_column: str = "actual_value"
) -> float:
    """
    Calculate Root Mean Squared Error.

    Args:
        df: DataFrame with predicted and actual columns
        predicted_column: Name of predicted values column
        actual_column: Name of actual values column

    Returns:
        RMSE as float
    """
    if df.empty or predicted_column not in df.columns or actual_column not in df.columns:
        return 0.0

    valid_rows = df[[predicted_column, actual_column]].dropna()
    if valid_rows.empty:
        return 0.0

    rmse = np.sqrt(((valid_rows[predicted_column] - valid_rows[actual_column]) ** 2).mean())
    return float(rmse)


def calculate_calibration_rate(df: pd.DataFrame, calibrated_column: str = "calibrated") -> float:
    """
    Calculate calibration rate (fraction of well-calibrated predictions).

    Args:
        df: DataFrame with calibrated column
        calibrated_column: Name of calibrated boolean column

    Returns:
        Calibration rate as float (0.0 to 1.0)
    """
    if df.empty or calibrated_column not in df.columns:
        return 0.0

    calibrated_count = df[calibrated_column].sum()
    total = len(df)

    return calibrated_count / total if total > 0 else 0.0


def calculate_roi(df: pd.DataFrame, pnl_column: str = "pnl", stake: float = 110.0) -> float:
    """
    Calculate Return on Investment (ROI).

    Args:
        df: DataFrame with P&L column
        pnl_column: Name of profit/loss column
        stake: Standard bet stake (default: $110 per bet)

    Returns:
        ROI as percentage (e.g., 5.0 for 5% ROI)
    """
    if df.empty or pnl_column not in df.columns:
        return 0.0

    total_pnl = df[pnl_column].sum()
    total_bets = len(df)
    total_invested = total_bets * stake

    roi = (total_pnl / total_invested * 100) if total_invested > 0 else 0.0
    return float(roi)


def calculate_sharpe_ratio(
    df: pd.DataFrame, pnl_column: str = "pnl", risk_free_rate: float = 0.0
) -> float:
    """
    Calculate Sharpe Ratio for betting performance.

    Args:
        df: DataFrame with P&L column
        pnl_column: Name of profit/loss column
        risk_free_rate: Risk-free rate (default: 0.0)

    Returns:
        Sharpe ratio as float
    """
    if df.empty or pnl_column not in df.columns:
        return 0.0

    returns = df[pnl_column].values
    if len(returns) == 0:
        return 0.0

    mean_return = returns.mean()
    std_return = returns.std()

    if std_return == 0:
        return 0.0

    sharpe = (mean_return - risk_free_rate) / std_return
    return float(sharpe)


def calculate_max_drawdown(df: pd.DataFrame, pnl_column: str = "weekly_pnl") -> float:
    """
    Calculate maximum drawdown from cumulative P&L.

    Args:
        df: DataFrame with P&L column (should be sorted by time)
        pnl_column: Name of profit/loss column

    Returns:
        Maximum drawdown as negative percentage (e.g., -15.0 for 15% drawdown)
    """
    if df.empty or pnl_column not in df.columns:
        return 0.0

    cumulative_pnl = df[pnl_column].cumsum()
    running_max = cumulative_pnl.expanding().max()
    drawdown = cumulative_pnl - running_max

    max_dd = drawdown.min()
    return float(max_dd)


def calculate_kelly_fraction(win_rate: float, odds: float = 1.91) -> float:
    """
    Calculate Kelly Criterion betting fraction.

    Args:
        win_rate: Win rate as float (0.0 to 1.0)
        odds: Decimal odds (default: 1.91 for -110 American)

    Returns:
        Kelly fraction as float (0.0 to 1.0)
    """
    if win_rate <= 0 or win_rate >= 1:
        return 0.0

    # Kelly formula: f = (bp - q) / b
    # where b = odds - 1, p = win_rate, q = 1 - win_rate
    b = odds - 1
    p = win_rate
    q = 1 - win_rate

    kelly = (b * p - q) / b
    return max(0.0, min(1.0, kelly))  # Clamp to [0, 1]


def display_summary_metrics(df: pd.DataFrame, stake: float = 110.0) -> None:
    """
    Display summary metrics in Streamlit columns.

    Args:
        df: DataFrame with prediction results
        stake: Bet stake for ROI calculation
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_predictions = len(df)
        st.metric("Total Predictions", f"{total_predictions:,}")

    with col2:
        win_rate = calculate_win_rate(df)
        win_rate_delta = (win_rate - 0.5) * 100
        st.metric("Win Rate", f"{win_rate*100:.1f}%", delta=f"{win_rate_delta:+.1f}% vs 50%")

    with col3:
        mae = calculate_mae(df)
        st.metric("Mean Absolute Error", f"{mae:.2f} yards")

    with col4:
        if "pnl" in df.columns:
            total_pnl = df["pnl"].sum()
            avg_pnl = df["pnl"].mean()
            st.metric("Total P&L", f"${total_pnl:,.2f}", delta=f"${avg_pnl:.2f} avg")
        else:
            st.metric("Total P&L", "N/A")


def display_advanced_metrics(df: pd.DataFrame, stake: float = 110.0) -> None:
    """
    Display advanced performance metrics.

    Args:
        df: DataFrame with prediction results
        stake: Bet stake for calculations
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        roi = calculate_roi(df, stake=stake)
        st.metric("ROI", f"{roi:.2f}%", help="Return on Investment")

    with col2:
        sharpe = calculate_sharpe_ratio(df)
        st.metric("Sharpe Ratio", f"{sharpe:.2f}", help="Risk-adjusted returns")

    with col3:
        if "weekly_pnl" in df.columns:
            max_dd = calculate_max_drawdown(df)
            st.metric("Max Drawdown", f"${max_dd:,.2f}", help="Largest cumulative loss")
        else:
            st.metric("Max Drawdown", "N/A")

    with col4:
        if "calibrated" in df.columns:
            calibration = calculate_calibration_rate(df)
            st.metric(
                "Calibration Rate", f"{calibration*100:.1f}%", help="Well-calibrated predictions"
            )
        else:
            st.metric("Calibration Rate", "N/A")


def display_kelly_recommendation(df: pd.DataFrame, odds: float = 1.91) -> None:
    """
    Display Kelly Criterion betting recommendation.

    Args:
        df: DataFrame with prediction results
        odds: Decimal odds (default: 1.91 for -110)
    """
    win_rate = calculate_win_rate(df)
    kelly_frac = calculate_kelly_fraction(win_rate, odds)

    st.info(
        f"""
    **Kelly Criterion Recommendation**

    Based on historical win rate of {win_rate*100:.1f}%:
    - **Full Kelly**: {kelly_frac*100:.1f}% of bankroll per bet
    - **Half Kelly** (conservative): {kelly_frac*50:.1f}% of bankroll per bet
    - **Quarter Kelly** (very conservative): {kelly_frac*25:.1f}% of bankroll per bet

    ⚠️ Note: Kelly can be aggressive. Most professionals use fractional Kelly (1/4 to 1/2).
    """
    )


def calculate_edge_statistics(df: pd.DataFrame, edge_column: str = "edge") -> dict[str, float]:
    """
    Calculate statistics about betting edges.

    Args:
        df: DataFrame with edge column
        edge_column: Name of edge column

    Returns:
        Dictionary with edge statistics
    """
    if df.empty or edge_column not in df.columns:
        return {
            "mean_edge": 0.0,
            "median_edge": 0.0,
            "std_edge": 0.0,
            "positive_edge_rate": 0.0,
            "max_edge": 0.0,
            "min_edge": 0.0,
        }

    edges = df[edge_column].dropna()

    return {
        "mean_edge": float(edges.mean()),
        "median_edge": float(edges.median()),
        "std_edge": float(edges.std()),
        "positive_edge_rate": float((edges > 0).mean()),
        "max_edge": float(edges.max()),
        "min_edge": float(edges.min()),
    }


def display_edge_statistics(df: pd.DataFrame, edge_column: str = "edge") -> None:
    """
    Display edge statistics in Streamlit.

    Args:
        df: DataFrame with edge column
        edge_column: Name of edge column
    """
    stats = calculate_edge_statistics(df, edge_column)

    st.subheader("📊 Betting Edge Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Mean Edge", f"{stats['mean_edge']:.4f}")
        st.metric("Median Edge", f"{stats['median_edge']:.4f}")

    with col2:
        st.metric("Std Dev", f"{stats['std_edge']:.4f}")
        st.metric("Positive Edge Rate", f"{stats['positive_edge_rate']*100:.1f}%")

    with col3:
        st.metric("Max Edge", f"{stats['max_edge']:.4f}")
        st.metric("Min Edge", f"{stats['min_edge']:.4f}")


def calculate_model_performance_summary(df: pd.DataFrame) -> dict[str, any]:
    """
    Calculate comprehensive model performance summary.

    Args:
        df: DataFrame with prediction results

    Returns:
        Dictionary with all key metrics
    """
    return {
        "total_predictions": len(df),
        "win_rate": calculate_win_rate(df),
        "mae": calculate_mae(df),
        "rmse": calculate_rmse(df),
        "calibration_rate": calculate_calibration_rate(df),
        "roi": calculate_roi(df),
        "sharpe_ratio": calculate_sharpe_ratio(df),
        "max_drawdown": calculate_max_drawdown(df),
        "edge_stats": calculate_edge_statistics(df),
    }
