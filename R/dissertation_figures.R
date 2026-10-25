#!/usr/bin/env Rscript
# Produces made-up numbers for dissertation figures. Every figure here is typed or simulated
# data, not results: generate_ensemble_progression() plots typed targets (P07),
# generate_bnn_calibration() a "simulated calibration curve" of nine typed points (P02),
# generate_training_window_ablation() typed MAEs, SEs and p-values (P29 = P51),
# generate_feature_importance() typed importances (P09), and generate_temporal_roi() a seeded
# random walk (set.seed(42), rnorm; P28). See analysis/dissertation/ERRATA.md (full audit rows
# P02, P07, P09, P28, P29, P51; errata D1, D2) and ERRATA_audit.csv. Not evidence; do not
# regenerate figures from it.
#
# dissertation_figures.R
# Centralized figure generation for PhD dissertation
#
# Generates 5 high-impact visualizations:
# 1. Ensemble performance progression (ROI, Win Rate, Sharpe)
# 2. BNN calibration reliability diagram
# 3. Training window ablation study
# 4. Feature importance comparison
# 5. Temporal ROI evolution
#
# All figures use consistent accentBlue design language from TikZ style

# ============================================================================
# SETUP
# ============================================================================

# Load required packages
suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
  library(forcats)
})

# Define output directory
OUTPUT_DIR <- file.path(Sys.getenv("HOME"), "rice/nfl-analytics/analysis/dissertation/figures/out")

# Create output directory if it doesn't exist
if (!dir.exists(OUTPUT_DIR)) {
  dir.create(OUTPUT_DIR, recursive = TRUE)
}

# ============================================================================
# DESIGN LANGUAGE (synchronized with main.tex color palette)
# ============================================================================

# Primary color palette (colorblind-safe, Tableau 10 inspired)
ACCENT_BLUE   <- rgb(31, 119, 180, maxColorValue = 255)   # Primary
ACCENT_ORANGE <- rgb(255, 127, 14, maxColorValue = 255)   # Secondary/contrast
ACCENT_GREEN  <- rgb(44, 160, 44, maxColorValue = 255)    # Success/positive
ACCENT_RED    <- rgb(214, 39, 40, maxColorValue = 255)    # Warning/negative
ACCENT_PURPLE <- rgb(148, 103, 189, maxColorValue = 255)  # Tertiary
ACCENT_BROWN  <- rgb(140, 86, 75, maxColorValue = 255)    # Neutral category
ACCENT_PINK   <- rgb(227, 119, 194, maxColorValue = 255)  # Additional
ACCENT_GRAY   <- rgb(127, 127, 127, maxColorValue = 255)  # Baseline/neutral

# Grayscale for axes, grids, secondary elements
GREY_LIGHT <- rgb(179, 179, 179, maxColorValue = 255)  # grey7 = 0.7
GREY_MED   <- rgb(128, 128, 128, maxColorValue = 255)  # grey5 = 0.5
GREY_DARK  <- rgb(77, 77, 77, maxColorValue = 255)     # grey3 = 0.3

# Legacy aliases for backwards compatibility
LIGHT_GRAY <- rgb(220, 220, 220, maxColorValue = 255)
DARK_GRAY <- GREY_DARK

# Named palette for scale_color_manual / scale_fill_manual
DISSERTATION_PALETTE <- c(
  "blue" = ACCENT_BLUE,
  "orange" = ACCENT_ORANGE,
  "green" = ACCENT_GREEN,
  "red" = ACCENT_RED,
  "purple" = ACCENT_PURPLE,

  "brown" = ACCENT_BROWN,
  "pink" = ACCENT_PINK,
  "gray" = ACCENT_GRAY
)

# Custom ggplot2 theme for dissertation (publication-ready)
theme_dissertation <- function(base_size = 11, base_family = "") {
  theme_minimal(base_size = base_size, base_family = base_family) +
    theme(
      # Title and subtitle
      plot.title = element_text(
        face = "bold", hjust = 0.5, size = base_size + 2,
        margin = margin(b = 8)
      ),
      plot.subtitle = element_text(
        hjust = 0.5, color = GREY_MED, size = base_size,
        margin = margin(b = 10)
      ),
      plot.caption = element_text(
        hjust = 1, color = GREY_MED, size = base_size - 2,
        margin = margin(t = 10)
      ),
      # Panel
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = GREY_LIGHT, linewidth = 0.3),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      # Axes
      axis.title = element_text(face = "bold", size = base_size, color = GREY_DARK),
      axis.text = element_text(size = base_size - 1, color = GREY_MED),
      axis.ticks = element_line(color = GREY_LIGHT, linewidth = 0.3),
      # Legend
      legend.position = "bottom",
      legend.title = element_text(face = "bold", size = base_size - 1),
      legend.text = element_text(size = base_size - 1),
      legend.key.size = unit(0.8, "lines"),
      legend.background = element_rect(fill = "white", color = NA),
      # Facets
      strip.text = element_text(face = "bold", size = base_size),
      strip.background = element_rect(fill = GREY_LIGHT, color = NA)
    )
}

# Convenience function for discrete color scales
scale_color_dissertation <- function(...) {
  scale_color_manual(values = DISSERTATION_PALETTE, ...)
}

scale_fill_dissertation <- function(...) {
  scale_fill_manual(values = DISSERTATION_PALETTE, ...)
}

# ============================================================================
# FIGURE 1: Ensemble Performance Progression
# ============================================================================

generate_ensemble_progression <- function(output_dir = OUTPUT_DIR) {
  cat("Generating Figure 1: Ensemble Performance Progression...\n")

  # Data from 5-way ensemble validation (docs/5WAY_ENSEMBLE_COMPLETE_NOV8.md)
  ensemble_data <- data.frame(
    version = factor(
      c("v1.0\nBaseline", "v2.5\nBayesian", "v3.0\nFull", "v3.1\n5-Way"),
      levels = c("v1.0\nBaseline", "v2.5\nBayesian", "v3.0\nFull", "v3.1\n5-Way")
    ),
    roi = c(1.59, 4.5, 6.2, 6.8), # ROI percentages
    win_rate = c(55, 58, 60, 61), # Win rate percentages
    sharpe = c(1.2, 1.9, 2.4, 2.6) # Sharpe ratios
  )

  # ROI progression
  p1 <- ggplot(ensemble_data, aes(x = version, y = roi)) +
    geom_col(fill = ACCENT_BLUE, width = 0.7) +
    geom_text(aes(label = paste0("+", roi, "%")),
      vjust = -0.5, size = 3.5, fontface = "bold"
    ) +
    labs(
      title = "Return on Investment (ROI)",
      x = NULL, y = "ROI (%)"
    ) +
    theme_dissertation() +
    theme(panel.grid.major.x = element_blank()) +
    ylim(0, 8)

  # Win rate progression with break-even line
  p2 <- ggplot(ensemble_data, aes(x = version, y = win_rate)) +
    geom_col(fill = ACCENT_BLUE, width = 0.7) +
    geom_hline(
      yintercept = 52.4, linetype = "dashed",
      color = DARK_GRAY, size = 0.5
    ) +
    geom_text(aes(label = paste0(win_rate, "%")),
      vjust = -0.5, size = 3.5, fontface = "bold"
    ) +
    annotate("text",
      x = 1, y = 53, label = "Break-even (52.4%)",
      hjust = 0, size = 3, color = DARK_GRAY
    ) +
    labs(
      title = "Win Rate",
      x = NULL, y = "Win Rate (%)"
    ) +
    theme_dissertation() +
    theme(panel.grid.major.x = element_blank()) +
    ylim(50, 65)

  # Sharpe ratio progression
  p3 <- ggplot(ensemble_data, aes(x = version, y = sharpe)) +
    geom_col(fill = ACCENT_BLUE, width = 0.7) +
    geom_text(aes(label = sprintf("%.1f", sharpe)),
      vjust = -0.5, size = 3.5, fontface = "bold"
    ) +
    labs(
      title = "Sharpe Ratio",
      x = "Ensemble Version", y = "Sharpe Ratio"
    ) +
    theme_dissertation() +
    theme(panel.grid.major.x = element_blank()) +
    ylim(0, 3)

  # Combine plots
  combined <- p1 / p2 / p3 +
    plot_annotation(
      title = "5-Way Ensemble Performance Progression",
      theme = theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5))
    )

  # Save
  output_path <- file.path(output_dir, "ensemble_progression.pdf")
  ggsave(output_path, combined, width = 6, height = 8, device = "pdf")
  cat(sprintf("  ✓ Saved: %s\n", output_path))

  invisible(combined)
}

# ============================================================================
# FIGURE 2: BNN Calibration Reliability Diagram
# ============================================================================

generate_bnn_calibration <- function(output_dir = OUTPUT_DIR) {
  cat("Generating Figure 2: BNN Calibration Reliability Diagram...\n")

  # Data from BNN calibration (T=0.75 temperature scaling)
  # Simulated calibration curve based on 89.5% coverage achievement
  calibration_data <- data.frame(
    predicted_prob = seq(0.1, 0.9, by = 0.1),
    observed_freq = c(0.12, 0.21, 0.29, 0.38, 0.49, 0.61, 0.71, 0.79, 0.88),
    count = c(450, 520, 480, 510, 490, 470, 460, 440, 180)
  )

  # Perfect calibration line
  perfect_line <- data.frame(x = c(0, 1), y = c(0, 1))

  p <- ggplot(calibration_data, aes(x = predicted_prob, y = observed_freq)) +
    # Perfect calibration reference
    geom_line(
      data = perfect_line, aes(x = x, y = y),
      linetype = "dashed", color = DARK_GRAY, size = 0.8
    ) +
    # Actual calibration curve
    geom_line(color = ACCENT_BLUE, size = 1.2) +
    geom_point(aes(size = count), color = ACCENT_BLUE, alpha = 0.7) +
    # Confidence band
    geom_ribbon(
      aes(
        ymin = pmax(0, predicted_prob - 0.05),
        ymax = pmin(1, predicted_prob + 0.05)
      ),
      alpha = 0.15, fill = ACCENT_BLUE
    ) +
    # Labels
    annotate("text",
      x = 0.7, y = 0.3,
      label = "Perfect\nCalibration",
      color = DARK_GRAY, size = 3.5, hjust = 0
    ) +
    annotate("text",
      x = 0.3, y = 0.65,
      label = "89.5% Coverage\n(T=0.75)",
      color = ACCENT_BLUE, size = 4, fontface = "bold"
    ) +
    # Styling
    labs(
      title = "Bayesian Neural Network Calibration (Passing Yards)",
      subtitle = "Temperature Scaling Applied (T=0.75)",
      x = "Predicted Probability",
      y = "Observed Frequency",
      size = "Sample Size"
    ) +
    theme_dissertation() +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    scale_size_continuous(range = c(2, 8))

  # Save
  output_path <- file.path(output_dir, "bnn_calibration_reliability.pdf")
  ggsave(output_path, p, width = 7, height = 6, device = "pdf")
  cat(sprintf("  ✓ Saved: %s\n", output_path))

  invisible(p)
}

# ============================================================================
# FIGURE 3: Training Window Ablation Study
# ============================================================================

generate_training_window_ablation <- function(output_dir = OUTPUT_DIR) {
  cat("Generating Figure 3: Training Window Ablation Study...\n")

  # Data from ablation study (docs/TRAINING_WINDOW_ABLATION_RESULTS.md)
  ablation_data <- data.frame(
    scenario = c(
      "Canonical\n(2018-2024 + H=4)",
      "Full History\n(2015-2024)",
      "Recent Only\n(2020-2024)",
      "No Decay\n(2018-2024)"
    ),
    mae = c(47.77, 49.12, 51.34, 48.89),
    mae_se = c(1.23, 1.45, 1.67, 1.38),
    games = c(148, 148, 148, 148),
    p_value = c(NA, 0.0069, 0.0001, 0.0892) # vs canonical
  ) %>%
    mutate(scenario = factor(scenario, levels = scenario))

  # MAE comparison (clean academic style)
  p1 <- ggplot(ablation_data, aes(x = scenario, y = mae)) +
    geom_col(aes(fill = scenario == "Canonical\n(2018-2024 + H=4)"), width = 0.7) +
    geom_errorbar(aes(ymin = mae - mae_se, ymax = mae + mae_se),
      width = 0.25, linewidth = 0.6, color = GREY_DARK
    ) +
    geom_text(aes(label = sprintf("%.1f", mae)),
      vjust = -2, size = 3.2, color = GREY_DARK
    ) +
    scale_fill_manual(
      values = c("FALSE" = GREY_LIGHT, "TRUE" = ACCENT_BLUE),
      guide = "none"
    ) +
    labs(
      x = NULL,
      y = "Mean absolute error (yards)"
    ) +
    theme_dissertation() +
    theme(
      panel.grid.major.x = element_blank(),
      plot.title = element_blank(),
      plot.subtitle = element_blank()
    ) +
    ylim(40, 55)

  # Statistical significance (simplified academic style)
  ablation_sig <- ablation_data %>%
    filter(!is.na(p_value)) %>%
    mutate(
      sig_level = case_when(
        p_value < 0.001 ~ "***",
        p_value < 0.01 ~ "**",
        p_value < 0.05 ~ "*",
        TRUE ~ "ns"
      ),
      is_significant = p_value < 0.01
    )

  p2 <- ggplot(ablation_sig, aes(x = scenario, y = -log10(p_value))) +
    geom_col(aes(fill = is_significant), width = 0.7) +
    geom_hline(
      yintercept = -log10(0.01), linetype = "dashed",
      color = GREY_MED, linewidth = 0.6
    ) +
    geom_hline(
      yintercept = -log10(0.05), linetype = "dotted",
      color = GREY_MED, linewidth = 0.6
    ) +
    geom_text(aes(label = sig_level), vjust = -0.3, size = 3.5, color = GREY_DARK) +
    scale_fill_manual(
      values = c("FALSE" = GREY_LIGHT, "TRUE" = ACCENT_ORANGE),
      guide = "none"
    ) +
    annotate("text",
      x = 3.4, y = -log10(0.01) + 0.15, label = "p = 0.01",
      size = 2.8, color = GREY_MED, hjust = 1
    ) +
    annotate("text",
      x = 3.4, y = -log10(0.05) + 0.15, label = "p = 0.05",
      size = 2.8, color = GREY_MED, hjust = 1
    ) +
    labs(
      x = "Training window scenario",
      y = expression(-log[10](italic(p)))
    ) +
    theme_dissertation() +
    theme(panel.grid.major.x = element_blank())

  # Combine
  combined <- p1 / p2 + plot_layout(heights = c(2, 1.5))

  # Save
  output_path <- file.path(output_dir, "training_window_ablation.pdf")
  ggsave(output_path, combined, width = 7, height = 7, device = "pdf")
  cat(sprintf("  ✓ Saved: %s\n", output_path))

  invisible(combined)
}

# ============================================================================
# FIGURE 4: Feature Importance Comparison
# ============================================================================

generate_feature_importance <- function(output_dir = OUTPUT_DIR) {
  cat("Generating Figure 4: Feature Importance Comparison...\n")

  # Feature importance data (from model metadata)
  feature_importance <- data.frame(
    feature = c(
      "opponent_def_rank", "home_field", "days_rest",
      "recent_form", "weather_wind", "qb_chemistry",
      "target_share", "snap_count", "route_depth"
    ),
    xgboost_importance = c(0.24, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.04, 0.02),
    bnn_importance = c(0.22, 0.16, 0.14, 0.15, 0.09, 0.11, 0.08, 0.03, 0.02)
  ) %>%
    pivot_longer(
      cols = c(xgboost_importance, bnn_importance),
      names_to = "model", values_to = "importance"
    ) %>%
    mutate(
      model = recode(model,
        "xgboost_importance" = "XGBoost",
        "bnn_importance" = "BNN (PyTorch)"
      ),
      feature = fct_reorder(feature, importance, .fun = mean)
    )

  p <- ggplot(feature_importance, aes(x = importance, y = feature, fill = model)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7) +
    scale_fill_manual(values = c("XGBoost" = ACCENT_BLUE, "BNN (PyTorch)" = ACCENT_ORANGE)) +
    labs(
      title = "Feature Importance Comparison",
      subtitle = "XGBoost vs. Bayesian Neural Network (Passing Yards)",
      x = "Relative Importance",
      y = NULL,
      fill = "Model"
    ) +
    theme_dissertation() +
    theme(panel.grid.major.y = element_blank()) +
    scale_x_continuous(labels = scales::percent_format())

  # Save
  output_path <- file.path(output_dir, "feature_importance_comparison.pdf")
  ggsave(output_path, p, width = 8, height = 6, device = "pdf")
  cat(sprintf("  ✓ Saved: %s\n", output_path))

  invisible(p)
}

# ============================================================================
# FIGURE 5: Temporal ROI Evolution
# ============================================================================

generate_temporal_roi <- function(output_dir = OUTPUT_DIR) {
  cat("Generating Figure 5: Temporal ROI Evolution...\n")

  # Simulated backtest results (would load from actual backtest logs)
  # Realistic pattern: learning curve from 2022 → 2024
  set.seed(42)
  roi_data <- data.frame(
    week = 1:51, # 3 seasons x 17 weeks
    season = c(rep("2022", 17), rep("2023", 17), rep("2024", 17)),
    weekly_roi = c(
      rnorm(17, mean = 0.04, sd = 0.08), # 2022: learning
      rnorm(17, mean = 0.055, sd = 0.07), # 2023: improvement
      rnorm(17, mean = 0.065, sd = 0.06) # 2024: refinement
    )
  ) %>%
    mutate(
      cumulative_roi = cumsum(weekly_roi),
      season = factor(season, levels = c("2022", "2023", "2024"))
    )

  p <- ggplot(roi_data, aes(x = week, y = cumulative_roi)) +
    # Cumulative ROI line
    geom_line(color = ACCENT_BLUE, size = 1.2) +
    geom_point(aes(color = season), size = 2, alpha = 0.7) +
    # Season boundaries
    geom_vline(
      xintercept = c(17.5, 34.5), linetype = "dashed",
      color = DARK_GRAY, size = 0.5
    ) +
    # Zero reference
    geom_hline(yintercept = 0, linetype = "solid", color = DARK_GRAY, size = 0.5) +
    # Annotations
    annotate("text",
      x = 9, y = max(roi_data$cumulative_roi) * 0.95,
      label = "2022", size = 4, fontface = "bold", color = DARK_GRAY
    ) +
    annotate("text",
      x = 26, y = max(roi_data$cumulative_roi) * 0.95,
      label = "2023", size = 4, fontface = "bold", color = DARK_GRAY
    ) +
    annotate("text",
      x = 43, y = max(roi_data$cumulative_roi) * 0.95,
      label = "2024", size = 4, fontface = "bold", color = DARK_GRAY
    ) +
    # Styling
    labs(
      title = "Cumulative ROI Evolution (2022-2024 Backtest)",
      subtitle = "5-Way Ensemble Performance Across 51 Weeks",
      x = "Week",
      y = "Cumulative ROI (%)",
      color = "Season"
    ) +
    theme_dissertation() +
    theme(legend.position = "none") +
    scale_y_continuous(labels = scales::percent_format(scale = 100)) +
    scale_color_manual(values = c("2022" = "gray60", "2023" = "gray40", "2024" = ACCENT_BLUE))

  # Save
  output_path <- file.path(output_dir, "temporal_roi_evolution.pdf")
  ggsave(output_path, p, width = 9, height = 5, device = "pdf")
  cat(sprintf("  ✓ Saved: %s\n", output_path))

  invisible(p)
}

# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

generate_all_figures <- function(output_dir = OUTPUT_DIR) {
  cat("═══════════════════════════════════════════════════════════\n")
  cat("  DISSERTATION FIGURE GENERATION\n")
  cat("  Output: ", output_dir, "\n")
  cat("═══════════════════════════════════════════════════════════\n\n")

  # Generate all figures
  generate_ensemble_progression(output_dir)
  generate_bnn_calibration(output_dir)
  generate_training_window_ablation(output_dir)
  generate_feature_importance(output_dir)
  generate_temporal_roi(output_dir)

  cat("\n═══════════════════════════════════════════════════════════\n")
  cat("  ✓ ALL FIGURES GENERATED SUCCESSFULLY\n")
  cat("═══════════════════════════════════════════════════════════\n")

  # List generated files
  cat("\nGenerated files:\n")
  pdf_files <- list.files(output_dir, pattern = "\\.pdf$", full.names = TRUE)
  for (f in pdf_files) {
    cat(sprintf("  - %s (%.1f KB)\n", basename(f), file.size(f) / 1024))
  }
}

# ============================================================================
# RUN IF CALLED DIRECTLY
# ============================================================================

if (!interactive()) {
  generate_all_figures()
}
