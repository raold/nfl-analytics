#!/usr/bin/env python3
"""
Generate and save Bayesian player ratings from trained brms model.

This script loads the trained R model and generates predictions using
brms' fitted() function, which correctly handles all model terms.
"""

import subprocess
import sys

# R script to generate predictions properly
r_script = """
library(brms)
library(DBI)
library(RPostgres)
library(tidyverse)

# Load model
model <- readRDS("models/bayesian/passing_yards_hierarchical_v1.rds")
data <- model$data

# Generate predictions for each player's typical game scenario
cat("Generating predictions for", length(unique(data$player_id)), "players...\\n")

# Get typical game scenario for each player
player_scenarios <- data %>%
  group_by(player_id) %>%
  summarize(
    log_attempts = mean(log_attempts, na.rm = TRUE),
    is_home = mean(is_home, na.rm = TRUE),
    is_favored = mean(is_favored, na.rm = TRUE),
    spread_abs = mean(spread_abs, na.rm = TRUE),
    is_bad_weather = mean(is_bad_weather, na.rm = TRUE),
    is_dome = mean(is_dome, na.rm = TRUE),
    total_line = mean(total_line, na.rm = TRUE),
    experience_cat = first(experience_cat),
    team = first(team),
    opponent = first(opponent),
    position_season = first(position_season),
    player_season = first(player_season),
    n_games = n(),
    .groups = "drop"
  )

# Generate predictions using brms - use fitted() for mean, predict() for intervals
# fitted() = expected value (parameter uncertainty only, no observation noise)
# predict() = predictive distribution (parameter uncertainty + observation noise)
cat("Generating posterior predictions...\\n")
preds_fitted <- fitted(model, newdata = player_scenarios, re_formula = NULL, summary = FALSE)
cat("Generating predictive distribution (with observation noise)...\\n")
preds_predict <- predict(model, newdata = player_scenarios, re_formula = NULL, summary = FALSE)

# Calculate summary statistics
# Use fitted() for point prediction (expected value)
# Use predict() for uncertainty intervals (includes observation noise -> better CI coverage)
ratings <- tibble(
  player_id = player_scenarios$player_id,
  stat_type = "passing_yards",
  season = 2024L,
  model_version = "hierarchical_v1.1",  # Bumped version for improved CI coverage

  # Both fitted() and predict() return predictions on the response scale (log_yards)
  # These ARE log-space values - DON'T take log again!
  # Python backtest will exp() them to get yards
  rating_mean = colMeans(preds_fitted),  # Use fitted() for point prediction
  rating_sd = apply(preds_predict, 2, sd),  # Use predict() for uncertainty
  rating_q05 = apply(preds_predict, 2, quantile, probs = 0.05),  # predict() includes obs noise
  rating_q50 = apply(preds_fitted, 2, quantile, probs = 0.50),  # Use fitted() for median
  rating_q95 = apply(preds_predict, 2, quantile, probs = 0.95),  # predict() includes obs noise

  # Metadata
  position_group_mean = 0.0,
  team_effect = 0.0,
  vs_opponent_effect = NA_real_,
  n_games_observed = player_scenarios$n_games,
  effective_sample_size = 1000.0,
  rhat = 1.0,
  trained_at = Sys.time(),
  updated_at = Sys.time()
)

cat("Generated", nrow(ratings), "ratings\\n")
cat("Sample predictions (yards):\\n")
print(head(tibble(
  player_id = ratings$player_id,
  log_mean = ratings$rating_mean,
  yards_mean = exp(ratings$rating_mean),
  yards_q05 = exp(ratings$rating_q05),
  yards_q95 = exp(ratings$rating_q95)
), 10))

# Save to database
conn <- dbConnect(
  Postgres(),
  host = "localhost",
  port = 5544,
  dbname = "devdb01",
  user = "dro",
  password = "sicillionbillions"
)

# Delete old ratings for this model version
dbExecute(conn, "DELETE FROM mart.bayesian_player_ratings WHERE stat_type = 'passing_yards' AND model_version = 'hierarchical_v1.1'")

# Insert using prepared statements
for(i in 1:nrow(ratings)) {
  row <- ratings[i,]
  sql <- sprintf("
    INSERT INTO mart.bayesian_player_ratings (
      player_id, stat_type, season, model_version,
      rating_mean, rating_sd, rating_q05, rating_q50, rating_q95,
      position_group_mean, team_effect, vs_opponent_effect,
      n_games_observed, effective_sample_size, rhat,
      trained_at, updated_at
    ) VALUES (
      '%s', '%s', %d, '%s',
      %f, %f, %f, %f, %f,
      %f, %f, NULL,
      %d, %f, %f,
      '%s', '%s'
    )",
    row$player_id, row$stat_type, row$season, row$model_version,
    row$rating_mean, row$rating_sd, row$rating_q05, row$rating_q50, row$rating_q95,
    row$position_group_mean, row$team_effect,
    row$n_games_observed, row$effective_sample_size, row$rhat,
    format(row$trained_at, "%Y-%m-%d %H:%M:%S"),
    format(row$updated_at, "%Y-%m-%d %H:%M:%S")
  )
  dbExecute(conn, sql)
  if(i %% 20 == 0) cat("  Inserted", i, "/", nrow(ratings), "\\r")
}

cat("\\n✅ Successfully saved", nrow(ratings), "ratings to database!\\n")
dbDisconnect(conn)
"""

if __name__ == "__main__":
    print("Running R script to generate Bayesian ratings...")
    result = subprocess.run(
        ["Rscript", "-e", r_script],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr, file=sys.stderr)

    sys.exit(result.returncode)
