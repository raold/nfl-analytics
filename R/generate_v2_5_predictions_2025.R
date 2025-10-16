#!/usr/bin/env Rscript
# Generate v2.5 Predictions for 2025 Season (Weeks 1-6)

library(brms)
library(dplyr)
library(DBI)
library(RPostgres)
library(logger)

log_info("Starting v2.5 prediction generation for 2025 season (weeks 1-6)")

# Database connection
con <- dbConnect(
  RPostgres::Postgres(),
  dbname = "devdb01",
  host = "localhost",
  port = 5544,
  user = "dro",
  password = "sicillionbillions"
)

# Load the v2.5 informative priors model
model_path <- "models/bayesian/passing_informative_priors_v1.rds"
log_info(paste("Loading v2.5 model from:", model_path))

model <- readRDS(model_path)
log_info("✓ Model loaded successfully")

# Fetch 2025 player data
query <- "
WITH player_2025_games AS (
  SELECT
    ngs.player_id,
    ngs.player_display_name as player_name,
    ngs.player_position as position,
    ngs.season,
    ngs.week,
    ngs.pass_yards as yards,
    ngs.attempts as attempts,
    ph.years_exp,
    ph.current_team as team,
    g.home_team,
    g.away_team,
    CASE WHEN ph.current_team = g.home_team THEN 1 ELSE 0 END as is_home,
    COALESCE(g.roof, 'unknown') as roof,
    CASE
      WHEN g.roof IN ('outdoors', 'open') AND CAST(g.temp AS NUMERIC) < 40
      THEN 1 ELSE 0
    END as is_bad_weather
  FROM nextgen_passing ngs
  JOIN mart.player_hierarchy ph ON ngs.player_id = ph.player_id
  LEFT JOIN games g ON
    g.season = ngs.season
    AND g.week = ngs.week
    AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
  WHERE ngs.season = 2025
    AND ngs.week BETWEEN 1 AND 6
    AND ngs.pass_yards IS NOT NULL
    AND ngs.attempts IS NOT NULL
    AND ngs.attempts >= 10
    AND ph.position = 'QB'
)
SELECT
  player_id,
  player_name,
  position,
  season,
  COUNT(*) as n_games_2025,
  AVG(yards) as avg_yards_actual,
  AVG(attempts) as avg_attempts,
  AVG(is_home) as pct_home,
  AVG(is_bad_weather) as pct_bad_weather,
  MAX(team) as team,
  MAX(years_exp) as years_exp
FROM player_2025_games
GROUP BY player_id, player_name, position, season
HAVING COUNT(*) >= 2
ORDER BY player_id
"

log_info("Fetching 2025 player data (weeks 1-6)...")
player_data <- dbGetQuery(con, query)
log_info(paste("✓ Loaded", nrow(player_data), "players for prediction"))

# Transform data for model prediction
pred_data <- player_data %>%
  mutate(
    log_attempts = log(avg_attempts),
    is_home = pct_home,
    is_bad_weather = pct_bad_weather,
    position = factor(position),
    team = factor(team),
    player_id = factor(player_id),
    season = factor(season)
  )

log_info("Generating predictions with posterior samples...")

# Generate predictions
predictions <- predict(
  model,
  newdata = pred_data,
  allow_new_levels = TRUE,
  probs = c(0.05, 0.95),
  summary = TRUE
)

# Create results dataframe
results <- data.frame(
  player_id = player_data$player_id,
  player_name = player_data$player_name,
  position = player_data$position,
  season = player_data$season,
  model_version = "informative_priors_v2.5",
  rating_mean_log = predictions[, "Estimate"],
  rating_sd_log = predictions[, "Est.Error"],
  rating_q05_log = predictions[, "Q5"],
  rating_q95_log = predictions[, "Q95"],
  pred_yards = exp(predictions[, "Estimate"]),
  pred_q05 = exp(predictions[, "Q5"]),
  pred_q95 = exp(predictions[, "Q95"]),
  n_games_2025 = player_data$n_games_2025,
  actual_yards = player_data$avg_yards_actual
)

log_info("✓ Predictions generated")
log_info(paste("  Mean predicted yards:", round(mean(results$pred_yards), 2)))
log_info(paste("  Mean actual yards:", round(mean(results$actual_yards), 2)))
log_info(paste("  Correlation:", round(cor(results$pred_yards, results$actual_yards), 3)))

# Save results
output_path <- "models/bayesian/v2_5_predictions_2025.csv"
write.csv(results, output_path, row.names = FALSE)
log_info(paste("✓ Saved predictions to:", output_path))

# Calculate metrics
mae <- mean(abs(results$pred_yards - results$actual_yards))
rmse <- sqrt(mean((results$pred_yards - results$actual_yards)^2))
correlation <- cor(results$pred_yards, results$actual_yards)

log_info("")
log_info("=== v2.5 Informative Priors Performance (2025 Weeks 1-6) ===")
log_info(paste("  n_predictions:", nrow(results)))
log_info(paste("  MAE:", round(mae, 2), "yards"))
log_info(paste("  RMSE:", round(rmse, 2), "yards"))
log_info(paste("  Correlation:", round(correlation, 3)))

# CI coverage
results$in_ci <- (results$actual_yards >= results$pred_q05) &
                 (results$actual_yards <= results$pred_q95)
ci_coverage <- mean(results$in_ci)
log_info(paste("  90% CI Coverage:", round(ci_coverage * 100, 1), "%"))

dbDisconnect(con)
log_info("✓ Complete")
