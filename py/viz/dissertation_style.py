#!/usr/bin/env python3
"""
Dissertation Plotting Style - Publication-Quality Figures
Implements all 31 visual improvements for consistent, professional figures

Usage:
    from py.viz.dissertation_style import (
        setup_plot_style, COLORS, SIZES, save_figure,
        plot_with_confidence, plot_calibration_enhanced, etc.
    )

    setup_plot_style()
    fig, ax = plt.subplots(figsize=SIZES['single'])
    # ... plotting code ...
    save_figure(fig, 'output.png')
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict

# =============================================================================
# COLOR PALETTES (#13: Colorblind-safe)
# =============================================================================

COLORS = {
    'primary': '#1f77b4',      # Blue
    'secondary': '#ff7f0e',    # Orange
    'success': '#2ca02c',      # Green
    'danger': '#d62728',       # Red
    'warning': '#bcbd22',      # Yellow-green
    'info': '#17becf',         # Cyan
    'neutral': '#7f7f7f',      # Gray
    'purple': '#9467bd',       # Purple
}

# Sequential palettes for heatmaps/continuous data
PALETTE_SEQUENTIAL_BLUE = sns.color_palette("Blues", n_colors=9)
PALETTE_SEQUENTIAL_GREEN = sns.color_palette("Greens", n_colors=9)

# Diverging palette for zero-centered data (#19)
PALETTE_DIVERGING = sns.diverging_palette(250, 10, n=11, as_cmap=False)
CMAP_DIVERGING = sns.diverging_palette(250, 10, n=256, as_cmap=True)
CMAP_SEQUENTIAL = sns.color_palette("Blues", n_colors=256, as_cmap=True)

# =============================================================================
# FIGURE SIZES (#12: Standard sizes)
# =============================================================================

SIZES = {
    'single': (6, 4),          # 0.8\linewidth - standard single figure
    'double': (3.2, 4),        # 0.48\linewidth each - side-by-side
    'full': (7.5, 5),          # \linewidth - full page width
    'wide': (7.5, 3),          # Wide aspect for time series
    'square': (5, 5),          # Square for heatmaps/scatter
    'multipanel_2x2': (8, 8),  # 2x2 grid
    'multipanel_2x3': (10, 6), # 2x3 grid
}

# =============================================================================
# FONT SIZES (#14: Consistent typography)
# =============================================================================

FONT_SIZES = {
    'title': 11,
    'label': 9,
    'tick': 8,
    'legend': 8,
    'annotation': 7,
}

# =============================================================================
# SETUP FUNCTIONS
# =============================================================================

def setup_plot_style():
    """
    Apply dissertation-wide plotting style
    Call this at the start of every plotting script
    """
    plt.style.use('seaborn-v0_8-paper')  # Clean base style

    plt.rcParams.update({
        # Fonts
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': FONT_SIZES['label'],
        'axes.titlesize': FONT_SIZES['title'],
        'axes.labelsize': FONT_SIZES['label'],
        'xtick.labelsize': FONT_SIZES['tick'],
        'ytick.labelsize': FONT_SIZES['tick'],
        'legend.fontsize': FONT_SIZES['legend'],
        'figure.titlesize': FONT_SIZES['title'],

        # Grid (#15: Add grid lines)
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'axes.axisbelow': True,

        # Lines and markers
        'lines.linewidth': 1.5,
        'lines.markersize': 4,

        # Axes
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',

        # Ticks
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.color': '#333333',
        'ytick.color': '#333333',

        # Legend (#16: Better positioning)
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#cccccc',
        'legend.fancybox': True,

        # Figure
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })

def save_figure(fig, path: Path, **kwargs):
    """
    Save figure with consistent DPI and settings

    Args:
        fig: Matplotlib figure
        path: Output path
        **kwargs: Additional savefig arguments
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Default settings
    save_kwargs = {
        'dpi': 300,
        'bbox_inches': 'tight',
        'facecolor': 'white',
        'edgecolor': 'none',
    }
    save_kwargs.update(kwargs)

    fig.savefig(path, **save_kwargs)
    print(f"✓ Saved figure: {path}")

# =============================================================================
# ENHANCED PLOT FUNCTIONS
# =============================================================================

def plot_with_confidence(ax, x, y, y_err=None, y_lower=None, y_upper=None,
                        label=None, color=None, alpha=0.2, **kwargs):
    """
    #17: Plot line with confidence interval shading

    Args:
        ax: Matplotlib axis
        x: X data
        y: Y data (central estimate)
        y_err: Symmetric error (± from y)
        y_lower, y_upper: Asymmetric bounds
        label: Line label
        color: Line color
        alpha: Transparency for CI band
        **kwargs: Additional plot arguments
    """
    if color is None:
        color = COLORS['primary']

    # Plot central line
    line = ax.plot(x, y, label=label, color=color, **kwargs)[0]

    # Add confidence band
    if y_err is not None:
        ax.fill_between(x, y - y_err, y + y_err,
                        alpha=alpha, color=color, linewidth=0)
    elif y_lower is not None and y_upper is not None:
        ax.fill_between(x, y_lower, y_upper,
                        alpha=alpha, color=color, linewidth=0)

    return line

def plot_calibration_enhanced(ax, y_true, y_pred, n_bins=10,
                             show_histogram=True, show_diagonal=True):
    """
    #18: Enhanced reliability diagram with reference diagonal and histogram

    Args:
        ax: Matplotlib axis
        y_true: True binary outcomes
        y_pred: Predicted probabilities
        n_bins: Number of calibration bins
        show_histogram: Show prediction distribution
        show_diagonal: Show perfect calibration line
    """
    # Bin predictions
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Calculate calibration
    bin_pred = []
    bin_true = []
    bin_counts = []

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_pred.append(y_pred[mask].mean())
            bin_true.append(y_true[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_pred.append(np.nan)
            bin_true.append(np.nan)
            bin_counts.append(0)

    bin_pred = np.array(bin_pred)
    bin_true = np.array(bin_true)
    bin_counts = np.array(bin_counts)

    # Plot calibration curve
    valid = ~np.isnan(bin_pred)

    # Color by sample size
    sizes = bin_counts[valid] / bin_counts.max() * 100 + 20
    colors = plt.cm.Blues(bin_counts[valid] / bin_counts.max())

    ax.scatter(bin_pred[valid], bin_true[valid], s=sizes, c=colors,
              edgecolors='black', linewidths=0.5, alpha=0.8, zorder=3)

    # Perfect calibration line
    if show_diagonal:
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1.5,
               label='Perfect calibration', zorder=1)

    # Formatting
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Observed Frequency')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=FONT_SIZES['legend'])

    # Add histogram of predictions at bottom
    if show_histogram:
        ax2 = ax.inset_axes([0, -0.2, 1, 0.15])
        ax2.hist(y_pred, bins=50, color=COLORS['info'], alpha=0.6, edgecolor='none')
        ax2.set_xlim(0, 1)
        ax2.set_yticks([])
        ax2.set_xlabel('Predicted Probability Distribution', fontsize=FONT_SIZES['tick'])
        ax2.tick_params(labelsize=FONT_SIZES['tick'] - 1)

    return ax

def boxplot_with_outliers(ax, data, positions=None, labels=None,
                          show_points=True, **kwargs):
    """
    #20: Box plot with individual outlier points shown

    Args:
        ax: Matplotlib axis
        data: List of arrays (one per box)
        positions: X positions for boxes
        labels: Labels for boxes
        show_points: Show individual outliers
        **kwargs: Additional boxplot arguments
    """
    # Create boxplot without fliers
    bp = ax.boxplot(data, positions=positions, labels=labels,
                    showfliers=False, patch_artist=True, **kwargs)

    # Style boxes
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS['primary'])
        patch.set_alpha(0.6)

    # Add outliers as scatter points
    if show_points:
        for i, d in enumerate(data):
            x_pos = positions[i] if positions is not None else i + 1

            # Calculate outliers (outside 1.5 IQR)
            q1, q3 = np.percentile(d, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = d[(d < lower_bound) | (d > upper_bound)]

            if len(outliers) > 0:
                # Add jitter to x position
                x_jitter = x_pos + np.random.normal(0, 0.04, len(outliers))
                ax.scatter(x_jitter, outliers, alpha=0.5, s=10,
                          color=COLORS['danger'], zorder=3)

    return bp

def plot_importance_with_ci(ax, features, importance, ci=None,
                            max_features=20, color_by_sign=True):
    """
    #22: Feature importance with confidence intervals and directional coloring

    Args:
        ax: Matplotlib axis
        features: Feature names
        importance: Importance values
        ci: Confidence intervals (±)
        max_features: Maximum features to show
        color_by_sign: Color bars by positive/negative
    """
    # Sort by absolute importance
    indices = np.argsort(np.abs(importance))[-max_features:]
    features = np.array(features)[indices]
    importance = importance[indices]
    if ci is not None:
        ci = ci[indices]

    # Colors
    if color_by_sign:
        colors = [COLORS['success'] if x > 0 else COLORS['danger']
                 for x in importance]
    else:
        colors = COLORS['primary']

    # Plot
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, importance, xerr=ci, color=colors,
                   alpha=0.7, edgecolor='black', linewidth=0.5)

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=FONT_SIZES['tick'])
    ax.set_xlabel('Importance', fontsize=FONT_SIZES['label'])
    ax.axvline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.3)

    return bars

def scatter_with_marginals(x, y, figsize=None, **scatter_kwargs):
    """
    #28: Scatter plot with marginal distribution histograms

    Args:
        x, y: Data arrays
        figsize: Figure size (default: SIZES['square'])
        **scatter_kwargs: Arguments for scatter plot

    Returns:
        fig, (ax_main, ax_top, ax_right)
    """
    if figsize is None:
        figsize = SIZES['square']

    # Create figure with GridSpec
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.05, wspace=0.05)

    ax_main = fig.add_subplot(gs[1:, :-1])
    ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)

    # Main scatter
    scatter_kwargs.setdefault('alpha', 0.5)
    scatter_kwargs.setdefault('s', 10)
    scatter_kwargs.setdefault('color', COLORS['primary'])
    ax_main.scatter(x, y, **scatter_kwargs)

    # Marginal histograms
    ax_top.hist(x, bins=30, color=COLORS['info'], alpha=0.6, edgecolor='none')
    ax_right.hist(y, bins=30, orientation='horizontal',
                 color=COLORS['info'], alpha=0.6, edgecolor='none')

    # Hide labels on marginals
    ax_top.tick_params(labelbottom=False)
    ax_right.tick_params(labelleft=False)
    ax_top.set_yticks([])
    ax_right.set_xticks([])

    return fig, (ax_main, ax_top, ax_right)

def add_direct_labels(ax, lines, labels, offset=0.02):
    """
    #30: Add labels directly on plot lines (instead of legend)

    Args:
        ax: Matplotlib axis
        lines: List of Line2D objects
        labels: List of labels
        offset: Offset from line end (fraction of x-range)
    """
    for line, label in zip(lines, labels):
        x, y = line.get_data()
        if len(x) > 0 and len(y) > 0:
            # Place label at end of line
            x_offset = offset * (ax.get_xlim()[1] - ax.get_xlim()[0])
            ax.text(x[-1] + x_offset, y[-1], label,
                   fontsize=FONT_SIZES['legend'],
                   verticalalignment='center',
                   color=line.get_color())

def add_risk_indicators(ax, data, var_level=0.95, cvar_level=0.95):
    """
    #31: Add risk visualization (VaR, CVaR thresholds) to betting/returns plots

    Args:
        ax: Matplotlib axis
        data: Return/PnL data
        var_level: Value-at-Risk confidence level
        cvar_level: Conditional VaR confidence level
    """
    # Calculate risk metrics
    var = np.percentile(data, (1 - var_level) * 100)
    cvar_mask = data <= var
    cvar = data[cvar_mask].mean() if cvar_mask.sum() > 0 else var

    # Add horizontal lines
    ax.axhline(var, color=COLORS['warning'], linestyle='--',
              linewidth=1.5, alpha=0.7,
              label=f'VaR {var_level:.0%}: {var:.2f}')
    ax.axhline(cvar, color=COLORS['danger'], linestyle='--',
              linewidth=1.5, alpha=0.7,
              label=f'CVaR {cvar_level:.0%}: {cvar:.2f}')

    return var, cvar

def shade_event_regions(ax, events: Dict[str, Tuple[float, float]], alpha=0.1):
    """
    #32: Shade significant event periods (COVID, rule changes, etc.)

    Args:
        ax: Matplotlib axis
        events: Dict of {label: (start_x, end_x)}
        alpha: Transparency
    """
    colors = [COLORS['danger'], COLORS['warning'], COLORS['info']]

    for i, (label, (start, end)) in enumerate(events.items()):
        color = colors[i % len(colors)]
        ax.axvspan(start, end, alpha=alpha, color=color, label=label)

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_multipanel_figure(nrows, ncols, figsize=None, **subplot_kwargs):
    """
    #11: Create multi-panel figure with consistent styling

    Args:
        nrows, ncols: Grid dimensions
        figsize: Figure size (auto-calculated if None)
        **subplot_kwargs: Arguments for subplots

    Returns:
        fig, axes
    """
    if figsize is None:
        # Auto-calculate based on grid
        width = min(ncols * 3, 12)
        height = min(nrows * 2.5, 10)
        figsize = (width, height)

    subplot_kwargs.setdefault('constrained_layout', True)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **subplot_kwargs)

    return fig, axes

def format_percentage_axis(ax, axis='y'):
    """Format axis to show percentages"""
    from matplotlib.ticker import PercentFormatter
    if axis == 'y':
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    else:
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))

def format_currency_axis(ax, axis='y', symbol='$'):
    """Format axis to show currency"""
    from matplotlib.ticker import FuncFormatter
    formatter = FuncFormatter(lambda x, p: f'{symbol}{x:,.0f}')
    if axis == 'y':
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)

# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == '__main__':
    # Example demonstrating all features
    setup_plot_style()

    # Example 1: Line plot with confidence band
    fig, ax = plt.subplots(figsize=SIZES['single'])
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    y_err = 0.1 * np.ones_like(x)
    plot_with_confidence(ax, x, y, y_err=y_err, label='Sin(x)', color=COLORS['primary'])
    ax.legend()
    ax.set_title('Example: Line with Confidence Band')
    save_figure(fig, '/tmp/example_confidence.png')

    # Example 2: Calibration plot
    fig, ax = plt.subplots(figsize=SIZES['single'])
    y_true = np.random.binomial(1, 0.5, 1000)
    y_pred = np.clip(y_true + np.random.normal(0, 0.2, 1000), 0, 1)
    plot_calibration_enhanced(ax, y_true, y_pred)
    ax.set_title('Example: Enhanced Calibration Plot')
    save_figure(fig, '/tmp/example_calibration.png')

    print("✓ dissertation_style.py examples generated")
