#!/usr/bin/env Rscript
# Quick start: Install and test priority nflverse packages
# Run: Rscript R/setup_nflverse_extended.R

cat("Installing high-priority nflverse packages...\n\n")

# Function to safely install packages
install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat("Installing", pkg, "...\n")
    install.packages(pkg, repos = "https://cloud.r-project.org")
  } else {
    cat("✓", pkg, "already installed\n")
  }
}

# Core packages (should already be installed)
cat("Checking core packages:\n")
install_if_missing("dplyr")
install_if_missing("DBI")
install_if_missing("RPostgres")

# nflverse core (should already have these)
cat("\nChecking nflverse core:\n")
if (!requireNamespace("nflreadr", quietly = TRUE)) {
  cat("Installing nflreadr from GitHub...\n")
  if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
  remotes::install_github("nflverse/nflreadr")
} else {
  cat("✓ nflreadr already installed\n")
}

if (!requireNamespace("nflfastR", quietly = TRUE)) {
  cat("Installing nflfastR from GitHub...\n")
  if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
  remotes::install_github("nflverse/nflfastR")
} else {
  cat("✓ nflfastR already installed\n")
}

# Priority additions
cat("\nInstalling priority nflverse extensions:\n")
install_if_missing("nfl4th")      # 4th down decision modeling
install_if_missing("nflplotR")    # Visualization with logos
install_if_missing("nflseedR")    # Playoff simulation

cat("\n" , rep("=", 80), "\n")
cat("Installation complete! Testing packages...\n")
cat(rep("=", 80), "\n\n")

# Test nfl4th
cat("Testing nfl4th (4th down decisions)...\n")
suppressPackageStartupMessages(library(nfl4th))
suppressPackageStartupMessages(library(dplyr))

# Load a small sample of PBP data
pbp_sample <- tryCatch({
  nflreadr::load_pbp(2024, file_type = "rds") |>
    filter(down == 4, !is.na(ydstogo)) |>
    head(5)
}, error = function(e) {
  # Fallback to 2023 if 2024 not available yet
  nflreadr::load_pbp(2023, file_type = "rds") |>
    filter(down == 4, !is.na(ydstogo)) |>
    head(5)
})

if (nrow(pbp_sample) > 0) {
  pbp_4th <- pbp_sample |>
    nfl4th::add_4th_probs()
  
  cat("✓ nfl4th working! Sample 4th down decisions:\n")
  print(pbp_4th |> 
    select(game_id, desc, go_boost, fg_boost) |> 
    head(3))
} else {
  cat("⚠ Could not load PBP data for testing\n")
}

# Test nflplotR
cat("\n\nTesting nflplotR (team logos)...\n")
suppressPackageStartupMessages(library(nflplotR))
suppressPackageStartupMessages(library(ggplot2))

test_df <- data.frame(
  team = c("KC", "SF", "BAL", "DET"),
  rating = c(95, 92, 90, 88)
)

test_plot <- ggplot(test_df, aes(x = rating, y = reorder(team, rating))) +
  geom_col(fill = "#003366", alpha = 0.7) +
  geom_nfl_logos(aes(team_abbr = team), width = 0.08, x = 85) +
  theme_minimal() +
  labs(title = "Test: Team Ratings with Logos", x = "Rating", y = NULL)

cat("✓ nflplotR working! Saving test plot...\n")
dir.create("analysis/dissertation/figures/tests", recursive = TRUE, showWarnings = FALSE)
ggsave("analysis/dissertation/figures/tests/nflplotR_test.png", 
       test_plot, width = 6, height = 4, dpi = 150)
cat("  Saved to: analysis/dissertation/figures/tests/nflplotR_test.png\n")

# Test nflseedR
cat("\n\nTesting nflseedR (playoff simulation)...\n")
suppressPackageStartupMessages(library(nflseedR))

# Load current season schedules
schedules <- nflreadr::load_schedules(2024)

# Simulate playoffs for first 10 weeks
sim_result <- tryCatch({
  nflseedR::simulate_nfl(
    nfl_season = 2024,
    process_games = schedules |> filter(week <= 10),
    playoff_seeds = 7,
    sims = 100  # Small number for quick test
  )
}, error = function(e) {
  cat("⚠ Simulation error (may need more completed games):", e$message, "\n")
  NULL
})

if (!is.null(sim_result)) {
  playoff_probs <- sim_result |>
    group_by(team) |>
    summarise(
      playoff_prob = mean(made_playoffs),
      .groups = "drop"
    ) |>
    arrange(desc(playoff_prob)) |>
    head(5)
  
  cat("✓ nflseedR working! Top 5 playoff probabilities:\n")
  print(playoff_probs)
} else {
  cat("⚠ nflseedR test skipped (may need more completed games in season)\n")
}

# Test injury data
cat("\n\nTesting injury data from nflreadr...\n")
injuries_2024 <- tryCatch({
  nflreadr::load_injuries(2024)
}, error = function(e) {
  cat("⚠ Could not load 2024 injuries, trying 2023...\n")
  nflreadr::load_injuries(2023)
})

if (!is.null(injuries_2024) && nrow(injuries_2024) > 0) {
  injury_summary <- injuries_2024 |>
    count(report_status, sort = TRUE) |>
    head(5)
  
  cat("✓ Injury data available! Status distribution:\n")
  print(injury_summary)
} else {
  cat("⚠ Injury data not available\n")
}

# Summary
cat("\n", rep("=", 80), "\n")
cat("Setup Complete!\n")
cat(rep("=", 80), "\n\n")

cat("Next steps:\n")
cat("1. Run: Rscript R/features_4th_down.R (once you create it)\n")
cat("2. Run: Rscript R/features_playoff_context.R (once you create it)\n")
cat("3. Update database schema: psql < db/004_advanced_features.sql\n")
cat("4. Retrain models with new features\n\n")

cat("Useful resources:\n")
cat("- nfl4th docs:   https://nfl4th.nflverse.com/\n")
cat("- nflplotR docs: https://nflplotr.nflverse.com/\n")
cat("- nflseedR docs: https://nflseedr.com/\n")
cat("- nflverse home: https://nflverse.nflverse.com/\n")
