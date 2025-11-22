"""
Dashboard Components - Streamlit Native Components Library

All components use Streamlit native widgets (no unsafe_allow_html).

Modules:
    - cards: Pre-built UI cards for common dashboard elements
    - charts: Plotly chart creation functions
    - metrics: Metric calculation and display functions
    - tables: DataFrame formatting and table utilities

Author: Claude Code
Date: November 22, 2025
"""

# Card components (existing)
from .cards import (
    service_health_card,
    game_prediction_card,
    section_banner,
    live_games_banner,
    international_games_banner,
    scheduled_games_banner,
    metric_card,
    edge_summary_stats,
    database_stats_row,
)

# Chart components (new)
from .charts import (
    plot_win_rate_over_time,
    plot_mae_over_time,
    plot_ats_accuracy,
    plot_error_distribution,
    plot_calibration,
    plot_bnn_uncertainty_distribution,
    plot_bnn_prediction_intervals,
    plot_pnl_over_time,
    plot_model_comparison_mae,
    plot_edge_distribution,
)

# Metric components (new)
from .metrics import (
    calculate_win_rate,
    calculate_mae,
    calculate_rmse,
    calculate_calibration_rate,
    calculate_roi,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_kelly_fraction,
    display_summary_metrics,
    display_advanced_metrics,
    display_kelly_recommendation,
    display_edge_statistics,
    calculate_model_performance_summary,
)

# Table components (new)
from .tables import (
    format_predictions_table,
    format_model_comparison_table,
    format_performance_table,
    format_game_predictions_table,
    format_database_stats_table,
    format_model_registry_table,
    format_training_runs_table,
    highlight_best_model,
    add_rank_column,
    filter_table_by_status,
    paginate_table,
    sort_table,
)

__all__ = [
    # Cards
    "service_health_card",
    "game_prediction_card",
    "section_banner",
    "live_games_banner",
    "international_games_banner",
    "scheduled_games_banner",
    "metric_card",
    "edge_summary_stats",
    "database_stats_row",
    # Charts
    "plot_win_rate_over_time",
    "plot_mae_over_time",
    "plot_ats_accuracy",
    "plot_error_distribution",
    "plot_calibration",
    "plot_bnn_uncertainty_distribution",
    "plot_bnn_prediction_intervals",
    "plot_pnl_over_time",
    "plot_model_comparison_mae",
    "plot_edge_distribution",
    # Metrics
    "calculate_win_rate",
    "calculate_mae",
    "calculate_rmse",
    "calculate_calibration_rate",
    "calculate_roi",
    "calculate_sharpe_ratio",
    "calculate_max_drawdown",
    "calculate_kelly_fraction",
    "display_summary_metrics",
    "display_advanced_metrics",
    "display_kelly_recommendation",
    "display_edge_statistics",
    "calculate_model_performance_summary",
    # Tables
    "format_predictions_table",
    "format_model_comparison_table",
    "format_performance_table",
    "format_game_predictions_table",
    "format_database_stats_table",
    "format_model_registry_table",
    "format_training_runs_table",
    "highlight_best_model",
    "add_rank_column",
    "filter_table_by_status",
    "paginate_table",
    "sort_table",
]
