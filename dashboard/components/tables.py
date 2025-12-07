"""
Dashboard Tables Module

DataFrame formatting and table display functions for the NFL Analytics Dashboard.
Returns formatted DataFrames ready for display - no queries or charts.

Author: Claude Code
Date: November 22, 2025
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def format_predictions_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format predictions DataFrame for display.

    Args:
        df: Raw predictions DataFrame

    Returns:
        Formatted DataFrame
    """
    if df.empty:
        return df

    # Create display copy
    df_display = df.copy()

    # Format numeric columns
    numeric_formats = {
        'predicted_value': '{:.1f}',
        'actual_value': '{:.1f}',
        'mae': '{:.1f}',
        'edge': '{:.4f}',
        'prob_win': '{:.1%}',
        'pnl': '${:.2f}',
        'spread_close': '{:+.1f}',
        'total_close': '{:.1f}'
    }

    for col, fmt in numeric_formats.items():
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda x: fmt.format(x) if pd.notna(x) else 'N/A'
            )

    # Format datetime columns
    if 'kickoff' in df_display.columns:
        df_display['kickoff'] = pd.to_datetime(df_display['kickoff']).dt.strftime('%Y-%m-%d %I:%M %p')

    # Rename columns for display
    column_renames = {
        'player_name': 'Player',
        'team': 'Team',
        'opponent': 'Opp',
        'predicted_value': 'Predicted',
        'actual_value': 'Actual',
        'mae': 'Error',
        'status': 'Result',
        'pnl': 'P&L',
        'edge': 'Edge',
        'prob_win': 'Win Prob',
        'kickoff': 'Game Time',
        'season': 'Season',
        'week': 'Week'
    }

    df_display.rename(columns=column_renames, inplace=True)

    # Select display columns (only those that exist)
    display_cols = [
        'Season', 'Week', 'Player', 'Team', 'Opp',
        'Predicted', 'Actual', 'Error', 'Result', 'P&L', 'Edge', 'Win Prob', 'Game Time'
    ]
    available_cols = [col for col in display_cols if col in df_display.columns]

    return df_display[available_cols]


def format_model_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format model comparison DataFrame for display.

    Args:
        df: Raw model comparison DataFrame

    Returns:
        Formatted DataFrame
    """
    if df.empty:
        return df

    df_display = df.copy()

    # Format numeric columns
    numeric_formats = {
        'mae': '{:.2f}',
        'std_error': '{:.2f}',
        'win_rate': '{:.1%}',
        'calibration_rate': '{:.1%}',
        'total_pnl': '${:,.2f}',
        'avg_edge': '{:.4f}',
        'total_predictions': '{:,}'
    }

    for col, fmt in numeric_formats.items():
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda x: fmt.format(x) if pd.notna(x) else 'N/A'
            )

    # Rename columns
    column_renames = {
        'model_name': 'Model',
        'version': 'Version',
        'model_status': 'Status',
        'total_predictions': 'Predictions',
        'mae': 'MAE',
        'std_error': 'Std Error',
        'win_rate': 'Win Rate',
        'calibration_rate': 'Calibration',
        'total_pnl': 'Total P&L',
        'avg_edge': 'Avg Edge'
    }

    df_display.rename(columns=column_renames, inplace=True)

    # Select display columns
    display_cols = [
        'Model', 'Version', 'Status', 'Predictions',
        'MAE', 'Std Error', 'Win Rate', 'Calibration', 'Total P&L', 'Avg Edge'
    ]
    available_cols = [col for col in display_cols if col in df_display.columns]

    return df_display[available_cols]


def format_performance_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format performance over time DataFrame for display.

    Args:
        df: Raw performance DataFrame

    Returns:
        Formatted DataFrame
    """
    if df.empty:
        return df

    df_display = df.copy()

    # Format numeric columns
    numeric_formats = {
        'total_player_games': '{:,}',
        'mae': '{:.2f}',
        'win_rate': '{:.1%}',
        'calibration_rate': '{:.1%}',
        'weekly_pnl': '${:,.2f}'
    }

    for col, fmt in numeric_formats.items():
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda x: fmt.format(x) if pd.notna(x) else 'N/A'
            )

    # Rename columns
    column_renames = {
        'season': 'Season',
        'week': 'Week',
        'total_player_games': 'Player Games',
        'mae': 'MAE',
        'win_rate': 'Win Rate',
        'calibration_rate': 'Calibration',
        'weekly_pnl': 'Weekly P&L'
    }

    df_display.rename(columns=column_renames, inplace=True)

    return df_display


def format_game_predictions_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format game predictions DataFrame for display.

    Args:
        df: Raw game predictions DataFrame

    Returns:
        Formatted DataFrame
    """
    if df.empty:
        return df

    df_display = df.copy()

    # Format scores and predictions
    if 'away_score' in df_display.columns and 'home_score' in df_display.columns:
        df_display['Score'] = df_display.apply(
            lambda row: f"{int(row['away_score'])} - {int(row['home_score'])}"
            if pd.notna(row['away_score']) and pd.notna(row['home_score'])
            else 'TBD',
            axis=1
        )

    # Format matchup
    if 'away_team' in df_display.columns and 'home_team' in df_display.columns:
        df_display['Matchup'] = df_display['away_team'] + ' @ ' + df_display['home_team']

    # Format spread
    if 'spread_close' in df_display.columns:
        df_display['Spread'] = df_display['spread_close'].apply(
            lambda x: f"{x:+.1f}" if pd.notna(x) else 'N/A'
        )

    # Format total
    if 'total_close' in df_display.columns:
        df_display['Total'] = df_display['total_close'].apply(
            lambda x: f"O/U {x:.1f}" if pd.notna(x) else 'N/A'
        )

    # Format kickoff time
    if 'kickoff' in df_display.columns:
        df_display['Time'] = pd.to_datetime(df_display['kickoff']).dt.strftime('%a %I:%M %p')

    # Format predictions
    if 'predicted_home_win_prob' in df_display.columns:
        df_display['Win Prob'] = df_display['predicted_home_win_prob'].apply(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) else 'N/A'
        )

    # Select display columns
    display_cols = ['Matchup', 'Time', 'Score', 'Spread', 'Total', 'Win Prob']
    available_cols = [col for col in display_cols if col in df_display.columns]

    return df_display[available_cols]


def format_database_stats_table(stats_dict: dict) -> pd.DataFrame:
    """
    Format database statistics as a table.

    Args:
        stats_dict: Dictionary with database statistics

    Returns:
        Formatted DataFrame
    """
    if not stats_dict:
        return pd.DataFrame()

    # Convert dict to DataFrame
    data = []
    for key, value in stats_dict.items():
        # Format key nicely
        display_key = key.replace('_', ' ').title()

        # Format value
        if isinstance(value, (int, np.integer)):
            display_value = f"{value:,}"
        elif isinstance(value, (float, np.floating)):
            display_value = f"{value:.2f}"
        else:
            display_value = str(value)

        data.append({'Metric': display_key, 'Value': display_value})

    return pd.DataFrame(data)


def format_model_registry_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format model registry DataFrame for display.

    Args:
        df: Raw model registry DataFrame

    Returns:
        Formatted DataFrame
    """
    if df.empty:
        return df

    df_display = df.copy()

    # Format timestamps
    timestamp_cols = ['created_at', 'updated_at', 'trained_at']
    for col in timestamp_cols:
        if col in df_display.columns:
            df_display[col] = pd.to_datetime(df_display[col]).dt.strftime('%Y-%m-%d %H:%M')

    # Format boolean columns
    bool_cols = ['is_ensemble']
    for col in bool_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: '✓' if x else '✗')

    # Rename columns
    column_renames = {
        'model_name': 'Model',
        'version': 'Version',
        'model_type': 'Type',
        'framework': 'Framework',
        'status': 'Status',
        'performance_mae': 'MAE',
        'performance_rmse': 'RMSE',
        'is_ensemble': 'Ensemble',
        'created_at': 'Created',
        'updated_at': 'Updated'
    }

    df_display.rename(columns=column_renames, inplace=True)

    # Select display columns
    display_cols = [
        'Model', 'Version', 'Type', 'Framework', 'Status',
        'MAE', 'RMSE', 'Ensemble', 'Created', 'Updated'
    ]
    available_cols = [col for col in display_cols if col in df_display.columns]

    return df_display[available_cols]


def format_training_runs_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format training runs DataFrame for display.

    Args:
        df: Raw training runs DataFrame

    Returns:
        Formatted DataFrame
    """
    if df.empty:
        return df

    df_display = df.copy()

    # Format timestamps
    if 'started_at' in df_display.columns:
        df_display['started_at'] = pd.to_datetime(df_display['started_at']).dt.strftime('%Y-%m-%d %H:%M')

    # Format duration (if exists)
    if 'duration_seconds' in df_display.columns:
        df_display['duration_display'] = df_display['duration_seconds'].apply(
            lambda x: f"{int(x//60)}m {int(x%60)}s" if pd.notna(x) else 'N/A'
        )

    # Format status with emoji
    if 'status' in df_display.columns:
        status_emoji = {
            'completed': '✓',
            'running': '▶',
            'failed': '✗',
            'pending': '○'
        }
        df_display['status_display'] = df_display['status'].apply(
            lambda x: f"{status_emoji.get(x, '?')} {x}" if pd.notna(x) else 'N/A'
        )

    # Rename columns
    column_renames = {
        'run_id': 'Run ID',
        'model_name': 'Model',
        'status_display': 'Status',
        'started_at': 'Started',
        'duration_display': 'Duration',
        'final_mae': 'MAE',
        'final_rmse': 'RMSE'
    }

    df_display.rename(columns=column_renames, inplace=True)

    # Select display columns
    display_cols = ['Run ID', 'Model', 'Status', 'Started', 'Duration', 'MAE', 'RMSE']
    available_cols = [col for col in display_cols if col in df_display.columns]

    return df_display[available_cols]


def highlight_best_model(df: pd.DataFrame, metric_column: str, lower_is_better: bool = True) -> pd.DataFrame:
    """
    Highlight best model in a comparison table.

    Args:
        df: DataFrame with model comparison
        metric_column: Column to use for highlighting
        lower_is_better: If True, lowest value is best; if False, highest is best

    Returns:
        Styled DataFrame
    """
    if df.empty or metric_column not in df.columns:
        return df

    def highlight_row(row):
        if lower_is_better:
            is_best = row[metric_column] == df[metric_column].min()
        else:
            is_best = row[metric_column] == df[metric_column].max()

        return ['background-color: lightgreen' if is_best else '' for _ in row]

    return df.style.apply(highlight_row, axis=1)


def add_rank_column(df: pd.DataFrame, metric_column: str, lower_is_better: bool = True) -> pd.DataFrame:
    """
    Add a rank column based on a metric.

    Args:
        df: DataFrame to rank
        metric_column: Column to rank by
        lower_is_better: If True, rank by ascending; if False, by descending

    Returns:
        DataFrame with Rank column added
    """
    if df.empty or metric_column not in df.columns:
        return df

    df_ranked = df.copy()
    df_ranked['Rank'] = df_ranked[metric_column].rank(
        method='min',
        ascending=lower_is_better
    ).astype(int)

    # Move Rank to first column
    cols = ['Rank'] + [col for col in df_ranked.columns if col != 'Rank']
    return df_ranked[cols]


def filter_table_by_status(df: pd.DataFrame, status_values: list[str], status_column: str = 'Status') -> pd.DataFrame:
    """
    Filter table by status values.

    Args:
        df: DataFrame to filter
        status_values: List of status values to include
        status_column: Name of status column

    Returns:
        Filtered DataFrame
    """
    if df.empty or status_column not in df.columns:
        return df

    return df[df[status_column].isin(status_values)]


def paginate_table(df: pd.DataFrame, page_size: int = 50, page: int = 1) -> pd.DataFrame:
    """
    Paginate a large DataFrame.

    Args:
        df: DataFrame to paginate
        page_size: Number of rows per page
        page: Page number (1-indexed)

    Returns:
        Paginated DataFrame
    """
    if df.empty:
        return df

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    return df.iloc[start_idx:end_idx]


def sort_table(df: pd.DataFrame, sort_column: str, ascending: bool = True) -> pd.DataFrame:
    """
    Sort table by a column.

    Args:
        df: DataFrame to sort
        sort_column: Column to sort by
        ascending: Sort order

    Returns:
        Sorted DataFrame
    """
    if df.empty or sort_column not in df.columns:
        return df

    return df.sort_values(by=sort_column, ascending=ascending)
