#!/usr/bin/env Rscript
# Bayesian Model Comparison and LaTeX Table Generation
#
# Generates publication-quality comparison tables for dissertation

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(DBI)
  library(RPostgres)
  library(knitr)
  library(kableExtra)
})

#' Generate LaTeX comparison table for dissertation
#' @param output_path Path to save .tex file
generate_bayesian_model_comparison_table <- function(output_path = "analysis/dissertation/figures/out/bayesian_model_comparison.tex") {

  # Model comparison data from training runs
  comparison_df <- tibble(
    Model = c("Basic Hierarchical", "Time-Varying", "Full Attack/Defense"),
    Formula = c(
      "Margin $\\sim$ Home + $(1|\\text{Home})$ + $(1|\\text{Away})$",
      "Margin $\\sim$ Home + $(1+t|\\text{Home})$ + $(1+t|\\text{Away})$",
      "Score $\\sim$ Home + $(1|\\text{Team})$ + $(1|\\text{Opp})$"
    ),
    `LOO-CV ELPD` = c(-11075, -11038.6, -20251.7),
    `Parameters` = c("66 (32 teams × 2 + 2 fixed)",
                     "130 (32 teams × 4 + 2 fixed)",
                     "68 (32 teams × 2 + 2 fixed)"),
    `Training Time` = c("5.2s", "14.8s", "11.9s"),
    `Key Feature` = c(
      "Simple pooling",
      "Temporal dynamics",
      "Explicit attack/defense"
    )
  )

  # Generate LaTeX table
  latex_table <- comparison_df %>%
    kbl(
      format = "latex",
      booktabs = TRUE,
      escape = FALSE,
      caption = "Bayesian hierarchical model comparison for NFL team ratings (2015-2024, n=2743 games). LOO-CV ELPD = Leave-One-Out Cross-Validation Expected Log Predictive Density (higher is better). Model 2 (Time-Varying) has the best predictive performance.",
      label = "bayesian-model-comparison"
    ) %>%
    kable_styling(
      latex_options = c("hold_position", "scale_down"),
      font_size = 10
    ) %>%
    column_spec(2, width = "6cm") %>%
    column_spec(6, width = "3cm") %>%
    footnote(
      general = "All models used brms/Stan with 4 chains, 2000 iterations, adapt\\\\_delta=0.95.",
      threeparttable = TRUE,
      escape = FALSE
    )

  # Write to file
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  writeLines(latex_table, output_path)

  message("✓ LaTeX table written to: ", output_path)

  # Also print markdown version for reference
  message("\nMarkdown preview:")
  print(kable(comparison_df, format = "markdown"))

  invisible(latex_table)
}

#' Generate top team ratings table
generate_top_teams_table <- function(output_path = "analysis/dissertation/figures/out/bayesian_top_teams.tex") {

  # Connect to database
  con <- dbConnect(
    RPostgres::Postgres(),
    dbname   = Sys.getenv("POSTGRES_DB", "devdb01"),
    host     = Sys.getenv("POSTGRES_HOST", "localhost"),
    port     = as.integer(Sys.getenv("POSTGRES_PORT", 5544)),
    user     = Sys.getenv("POSTGRES_USER", "dro"),
    password = Sys.getenv("POSTGRES_PASSWORD", "sicillionbillions")
  )
  on.exit(dbDisconnect(con), add = TRUE)

  # Fetch ratings
  ratings <- dbGetQuery(con, "
    SELECT
      team,
      ROUND(rating_mean::numeric, 2) AS rating_mean,
      ROUND(rating_sd::numeric, 2) AS rating_sd,
      ROUND(rating_q05::numeric, 2) AS ci_lower,
      ROUND(rating_q95::numeric, 2) AS ci_upper
    FROM mart.bayesian_team_ratings
    ORDER BY rating_mean DESC
    LIMIT 10
  ") %>% as_tibble()

  # Generate LaTeX table
  latex_table <- ratings %>%
    mutate(
      `Rating (Mean)` = rating_mean,
      `SD` = rating_sd,
      `90% CI` = sprintf("[%.2f, %.2f]", ci_lower, ci_upper)
    ) %>%
    select(Team = team, `Rating (Mean)`, SD, `90% CI`) %>%
    kbl(
      format = "latex",
      booktabs = TRUE,
      caption = "Top 10 NFL teams by Bayesian hierarchical rating (2015-2024). Ratings represent points above/below average on a neutral field.",
      label = "bayesian-top-teams"
    ) %>%
    kable_styling(latex_options = c("hold_position")) %>%
    footnote(
      general = "Based on Model 1 (Basic Hierarchical). Ratings are posterior means with standard deviations and 90\\% credible intervals.",
      threeparttable = TRUE
    )

  # Write to file
  writeLines(latex_table, output_path)
  message("✓ Top teams table written to: ", output_path)

  # Print markdown version
  message("\nMarkdown preview:")
  print(kable(ratings %>% mutate(`90% CI` = sprintf("[%.2f, %.2f]", ci_lower, ci_upper)) %>%
                select(Team = team, `Rating` = rating_mean, SD = rating_sd, `90% CI`),
              format = "markdown"))

  invisible(latex_table)
}

#' Main execution
if (!interactive()) {
  message("Generating Bayesian model comparison tables...\n")

  generate_bayesian_model_comparison_table()
  cat("\n")
  generate_top_teams_table()

  message("\n✓ All tables generated successfully!")
}
