#!/usr/bin/env Rscript
#
# 2025 Hold-Out Validation for State-Space Model
#
# Purpose: Validate the simplified random walk model (trained on 2020-2024)
#          using 2025 as a completely unseen hold-out test set
#
# Model: models/bayesian/simplified_random_walk_v1.rds (trained on 2020-2024)
# Test:  2025 season data (weeks 1-6)
#
# Metrics:
# - Prediction accuracy (MAE, RMSE)
# - Calibration (90% credible interval coverage)
# - Comparison vs baseline models
#

library(tidyverse)
library(DBI)
library(RPostgres)
library(cmdstanr)
library(posterior)

cat("======================================================================\n")
cat("2025 Hold-Out Validation: State-Space Model\n")
cat("======================================================================\n")
cat("Training period: 2020-2024 (model already trained)\n")
cat("Test period: 2025 (weeks 1-6, completely unseen)\n")
cat("======================================================================\n\n")

# ============================================================================
# STEP 1: Load 2025 hold-out data
# ============================================================================

cat("STEP 1: Loading 2025 hold-out data from database...\n")

con <- dbConnect(
  RPostgres::Postgres(),
  host = "localhost",
  port = 5544,
  dbname = "devdb01",
  user = "dro",
  password = Sys.getenv("DB_PASSWORD", "sicillionbillions")
)

query_2025 <- "
  WITH player_games AS (
    SELECT
      pgs.player_id,
      pgs.player_display_name AS player_name,
      pgs.player_position AS position,
      pgs.season,
      pgs.week,
      pgs.current_team AS team,
      pgs.stat_yards AS passing_yards,
      pgs.stat_attempts AS passing_attempts,
      g.game_id,
      g.home_team,
      g.away_team,
      CASE
        WHEN pgs.current_team = g.home_team THEN 1
        WHEN pgs.current_team = g.away_team THEN 0
        ELSE NULL
      END AS is_home
    FROM mart.player_game_stats pgs
    LEFT JOIN public.games g ON (
      pgs.season = g.season AND
      pgs.week = g.week AND
      (pgs.current_team = g.home_team OR pgs.current_team = g.away_team)
    )
    WHERE pgs.season = 2025
      AND pgs.week > 0
      AND pgs.stat_category = 'passing'
      AND pgs.stat_attempts > 0
      AND pgs.stat_yards IS NOT NULL
      AND pgs.player_position = 'QB'
  )
  SELECT *
  FROM player_games
  WHERE is_home IS NOT NULL
  ORDER BY player_id, season, week
"

test_data_2025 <- dbGetQuery(con, query_2025)
dbDisconnect(con)

cat(sprintf(
  "  ✓ Loaded %d games from %d QBs (2025)\n",
  nrow(test_data_2025),
  n_distinct(test_data_2025$player_id)
))
cat(sprintf(
  "  ✓ Weeks: %d-%d\n",
  min(test_data_2025$week),
  max(test_data_2025$week)
))

# ============================================================================
# STEP 2: Load trained model (2020-2024)
# ============================================================================

cat("\nSTEP 2: Loading trained state-space model...\n")

model_path <- "models/bayesian/simplified_random_walk_v1.rds"
if (!file.exists(model_path)) {
  stop(sprintf("❌ Model not found: %s\n", model_path))
}

fit <- readRDS(model_path)
cat(sprintf("  ✓ Loaded model: %s\n", model_path))

# Extract global parameters
global_params <- fit$draws(
  variables = c(
    "mu_league", "sigma_player", "Q", "R",
    "beta_attempts", "beta_home"
  ),
  format = "draws_matrix"
)

mu_league_mean <- mean(global_params[, "mu_league"])
beta_attempts_mean <- mean(global_params[, "beta_attempts"])
beta_home_mean <- mean(global_params[, "beta_home"])
R_mean <- mean(global_params[, "R"])

cat("  ✓ Extracted global parameters:\n")
cat(sprintf("    - mu_league: %.3f\n", mu_league_mean))
cat(sprintf("    - beta_attempts: %.3f\n", beta_attempts_mean))
cat(sprintf("    - beta_home: %.3f\n", beta_home_mean))
cat(sprintf("    - R (measurement noise): %.3f\n", R_mean))

# ============================================================================
# STEP 3: Generate predictions for 2025
# ============================================================================

cat("\nSTEP 3: Generating predictions for 2025 hold-out data...\n")

# For new players in 2025 (not in training), use league mean
# For existing players, use their final 2024 skill estimate

# Load skill trajectories from training
skills_path <- "models/bayesian/simplified_random_walk_skills_v1.csv"
if (file.exists(skills_path)) {
  skills_2024 <- read_csv(skills_path, show_col_types = FALSE) %>%
    group_by(player_id, player_name) %>%
    slice_max(game_num, n = 1) %>% # Get final skill estimate (last game of 2024)
    ungroup() %>%
    select(player_id, player_name, skill_final = skill_mean)

  cat(sprintf(
    "  ✓ Loaded skill trajectories for %d players from training\n",
    nrow(skills_2024)
  ))
} else {
  # If skills file doesn't exist, use league mean for all
  skills_2024 <- data.frame(player_id = character(0), skill_final = numeric(0))
  cat("  ⚠  Skills file not found, will use league mean for all players\n")
}

# Generate predictions
predictions_2025 <- test_data_2025 %>%
  left_join(skills_2024, by = "player_id") %>%
  mutate(
    # Use final 2024 skill if available, otherwise league mean
    player_skill = ifelse(is.na(skill_final), mu_league_mean, skill_final),

    # Log-scale prediction
    log_pred = player_skill +
      beta_attempts_mean * log(passing_attempts) +
      beta_home_mean * is_home,

    # Back-transform to yards (undo log(y+1) transformation)
    pred_yards = exp(log_pred) - 1,

    # Prediction uncertainty (90% credible interval)
    pred_lower_90 = exp(log_pred - 1.645 * R_mean) - 1,
    pred_upper_90 = exp(log_pred + 1.645 * R_mean) - 1,

    # Actual yards
    actual_yards = passing_yards,

    # Prediction error
    error = pred_yards - actual_yards,
    abs_error = abs(error),
    sq_error = error^2,

    # Coverage: is actual within 90% CI?
    covered_90 = (actual_yards >= pred_lower_90) & (actual_yards <= pred_upper_90)
  )

cat(sprintf("  ✓ Generated predictions for %d games\n", nrow(predictions_2025)))

# ============================================================================
# STEP 4: Compute accuracy metrics
# ============================================================================

cat("\nSTEP 4: Computing hold-out accuracy metrics...\n")

mae <- mean(predictions_2025$abs_error)
rmse <- sqrt(mean(predictions_2025$sq_error))
coverage_90 <- mean(predictions_2025$covered_90) * 100
mean_interval_width <- mean(predictions_2025$pred_upper_90 - predictions_2025$pred_lower_90)

cat("\n")
cat("======================================================================\n")
cat("2025 HOLD-OUT VALIDATION RESULTS\n")
cat("======================================================================\n")
cat(sprintf(
  "Test set: %d games from %d QBs (2025 weeks 1-6)\n",
  nrow(predictions_2025),
  n_distinct(predictions_2025$player_id)
))
cat("\n")
cat("Prediction Accuracy:\n")
cat(sprintf("  MAE:  %.2f yards\n", mae))
cat(sprintf("  RMSE: %.2f yards\n", rmse))
cat("\n")
cat("Calibration:\n")
cat(sprintf("  90%% CI Coverage: %.1f%% (target: 85-95%%)\n", coverage_90))
cat(sprintf("  Mean Interval Width: %.1f yards\n", mean_interval_width))
cat("\n")

# Coverage assessment
if (coverage_90 >= 85 && coverage_90 <= 95) {
  cat("  ✅ WELL-CALIBRATED: Coverage within target range\n")
} else if (coverage_90 < 85) {
  cat(sprintf(
    "  ⚠️  UNDER-CALIBRATED: Intervals too narrow (missing %.1f%% of actual values)\n",
    90 - coverage_90
  ))
} else {
  cat(sprintf(
    "  ⚠️  OVER-CALIBRATED: Intervals too wide (covering %.1f%% extra)\n",
    coverage_90 - 90
  ))
}
cat("======================================================================\n\n")

# ============================================================================
# STEP 5: Save results
# ============================================================================

cat("STEP 5: Saving validation results...\n")

# Save predictions
output_file <- "analysis/results/state_space_2025_holdout_predictions.csv"
write_csv(predictions_2025, output_file)
cat(sprintf("  ✓ Predictions saved: %s\n", output_file))

# Save summary metrics
metrics <- data.frame(
  test_period = "2025 weeks 1-6",
  n_games = nrow(predictions_2025),
  n_players = n_distinct(predictions_2025$player_id),
  mae_yards = mae,
  rmse_yards = rmse,
  coverage_90_pct = coverage_90,
  mean_interval_width = mean_interval_width
)

metrics_file <- "analysis/results/state_space_2025_holdout_metrics.csv"
write_csv(metrics, metrics_file)
cat(sprintf("  ✓ Metrics saved: %s\n", metrics_file))

cat("\n")
cat("======================================================================\n")
cat("✅ 2025 HOLD-OUT VALIDATION COMPLETE\n")
cat("======================================================================\n")
cat(sprintf("MAE: %.2f yards | Coverage: %.1f%%\n", mae, coverage_90))
cat("======================================================================\n")
