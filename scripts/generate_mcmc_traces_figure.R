#!/usr/bin/env Rscript
#
# Generate MCMC Convergence Traces Figure for Dissertation
#
# Creates Figure: mcmc_convergence_traces.png
# Shows trace plots demonstrating chain mixing and convergence
#
# Output: analysis/dissertation/figures/out/mcmc_convergence_traces.png

library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)

# Dissertation color palette (colorblind-safe)
COLORS <- c(
  "1" = "#1f77b4",  # Blue
  "2" = "#ff7f0e",  # Orange
  "3" = "#2ca02c",  # Green
  "4" = "#d62728"   # Red
)

# Set output directory
OUTPUT_DIR <- file.path(Sys.getenv("HOME"), "rice/nfl-analytics/analysis/dissertation/figures/out")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# Theme for dissertation figures
theme_dissertation <- function(base_size = 11) {
  theme_minimal(base_size = base_size) +
    theme(
      text = element_text(family = "serif"),
      plot.title = element_text(size = base_size + 1, hjust = 0.5),
      axis.title = element_text(size = base_size),
      axis.text = element_text(size = base_size - 1, color = "#4d4d4d"),
      legend.position = "top",
      legend.title = element_text(size = base_size - 1),
      legend.text = element_text(size = base_size - 2),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "#e0e0e0", linewidth = 0.3),
      strip.text = element_text(size = base_size, face = "bold")
    )
}

# Load the fitted model. There is no simulated fallback: a trace figure must show real draws.
generate_trace_data <- function() {
  model_path <- "models/bayesian/passing_yards_hierarchical_v2.rds"

  if (file.exists(model_path)) {
    message("Loading model from: ", model_path)

    tryCatch({
      suppressMessages({
        library(brms)
        model <- readRDS(model_path)

        # Extract posterior samples
        posterior <- as_draws_df(model)

        # Select key parameters to display
        # Look for intercept and key coefficients
        param_names <- names(posterior)
        key_params <- c()

        # Find intercept
        if ("b_Intercept" %in% param_names) key_params <- c(key_params, "b_Intercept")

        # Find sigma
        if ("sigma" %in% param_names) key_params <- c(key_params, "sigma")

        # Find some coefficients
        coef_params <- param_names[grepl("^b_", param_names) & !grepl("Intercept", param_names)]
        if (length(coef_params) > 0) {
          key_params <- c(key_params, head(coef_params, 2))
        }

        if (length(key_params) >= 3) {
          # Format for plotting
          trace_df <- posterior %>%
            select(all_of(key_params), .chain, .iteration) %>%
            pivot_longer(cols = all_of(key_params), names_to = "parameter", values_to = "value") %>%
            mutate(chain = factor(.chain))

          return(trace_df)
        }
      })
    }, error = function(e) {
      message("Could not load model: ", e$message)
    })
  }

  stop(
    "Could not extract MCMC draws from ", model_path, ". ",
    "The brms fits in models/bayesian/ were removed in the 2026-10 prune; restore the file from ",
    "tag archive/pre-prune-2026-10 (git LFS) to regenerate this figure. ",
    "The script no longer simulates traces."
  )
}

# Create trace plot
create_trace_plot <- function(trace_df) {
  # Create individual trace plots
  params <- unique(trace_df$parameter)

  plots <- lapply(params, function(p) {
    df <- trace_df %>% filter(parameter == p)

    ggplot(df, aes(x = iteration, y = value, color = chain)) +
      geom_line(alpha = 0.7, linewidth = 0.3) +
      scale_color_manual(values = COLORS) +
      labs(x = "Iteration", y = "Value", title = p) +
      theme_dissertation() +
      theme(
        legend.position = "none",
        plot.title = element_text(size = 10)
      )
  })

  # Combine plots
  combined <- wrap_plots(plots, ncol = 2) +
    plot_annotation(
      title = "MCMC Trace Plots: Chain Mixing and Convergence",
      subtitle = "Four independent chains (colors) exploring posterior distribution",
      theme = theme_dissertation() +
        theme(
          plot.title = element_text(size = 12, hjust = 0.5),
          plot.subtitle = element_text(size = 10, hjust = 0.5, color = "#666666")
        )
    )

  # Add legend separately
  legend_plot <- ggplot(data.frame(chain = factor(1:4)), aes(x = 1, y = 1, color = chain)) +
    geom_point(size = 3) +
    scale_color_manual(values = COLORS, labels = paste("Chain", 1:4), breaks = c("1", "2", "3", "4")) +
    guides(color = guide_legend(title = NULL, nrow = 1)) +
    theme_void() +
    theme(legend.position = "bottom")

  combined / legend_plot + plot_layout(heights = c(20, 1))
}

# Main execution
main <- function() {
  message("Generating MCMC Convergence Traces Figure...")

  # Generate or load trace data
  trace_df <- generate_trace_data()

  # Create the plot
  p <- create_trace_plot(trace_df)

  # Save
  output_path <- file.path(OUTPUT_DIR, "mcmc_convergence_traces.png")
  ggsave(output_path, p, width = 10, height = 8, dpi = 300, bg = "white")

  message("Saved to: ", output_path)
  message("Done!")
}

main()
