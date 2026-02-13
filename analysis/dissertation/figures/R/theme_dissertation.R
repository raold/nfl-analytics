#!/usr/bin/env Rscript
#
# Dissertation Theme for ggplot2
# Palatino typography + Tableau 10 colorblind-safe palette
# Matches LaTeX document styling (newpxtext/newpxmath)
#

library(ggplot2)

# Tableau 10 colorblind-safe palette (matches LaTeX \definecolor)
COLORS_TABLEAU <- c(
  blue    = "#1f77b4",
  orange  = "#ff7f0e",
  green   = "#2ca02c",
  red     = "#d62728",
  purple  = "#9467bd",
  brown   = "#8c564b",
  pink    = "#e377c2",
  gray    = "#7f7f7f",
  yellow  = "#bcbd22",
  cyan    = "#17becf"
)

# Primary accent (matches accentBlue in LaTeX)
ACCENT_BLUE <- "#1f77b4"

#' Dissertation ggplot2 theme
#'
#' Clean, publication-ready theme with Palatino fonts matching the LaTeX document.
#' Based on theme_minimal with customizations for the NFL analytics dissertation.
#'
#' @param base_size Base font size (default 10)
#' @param base_family Font family (default "TeX Gyre Pagella")
#' @return A ggplot2 theme object
theme_dissertation <- function(base_size = 10, base_family = "TeX Gyre Pagella") {
  theme_minimal(base_size = base_size, base_family = base_family) %+replace%
    theme(
      # Title styling
      plot.title = element_text(face = "bold", size = rel(1.1), hjust = 0,
                                margin = margin(b = 4)),
      plot.subtitle = element_text(size = rel(0.9), color = "gray40", hjust = 0,
                                   margin = margin(b = 8)),
      plot.caption = element_text(size = rel(0.7), color = "gray50", hjust = 0,
                                  margin = margin(t = 6)),

      # Axis styling
      axis.title = element_text(size = rel(0.9), color = "#333333"),
      axis.text = element_text(size = rel(0.8), color = "#333333"),
      axis.ticks = element_line(linewidth = 0.4, color = "#333333"),

      # Grid — light dashed, y-axis only by default
      panel.grid.major.y = element_line(linewidth = 0.3, linetype = "dashed",
                                        color = "gray80"),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),

      # Legend
      legend.position = "bottom",
      legend.text = element_text(size = rel(0.8)),
      legend.title = element_text(size = rel(0.85), face = "bold"),

      # Plot margins
      plot.margin = margin(8, 8, 8, 8)
    )
}

#' Save ggplot with cairo_pdf device (required for TeX Gyre Pagella embedding)
#'
#' @param plot ggplot object
#' @param filename Output path
#' @param width Width in inches (default 6)
#' @param height Height in inches (default 4)
ggsave_dissertation <- function(plot, filename, width = 6, height = 4) {
  ggsave(filename, plot, width = width, height = height, device = cairo_pdf)
}

#' Tableau 10 discrete fill scale
scale_fill_tableau <- function(...) {
  scale_fill_manual(values = unname(COLORS_TABLEAU), ...)
}

#' Tableau 10 discrete color scale
scale_color_tableau <- function(...) {
  scale_color_manual(values = unname(COLORS_TABLEAU), ...)
}
