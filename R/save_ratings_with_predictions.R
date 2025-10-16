#!/usr/bin/env Rscript
# Save Bayesian ratings with properly transformed predictions
# This script generates actual yard predictions (not log-space) for use in backtesting

library(DBI)
library(RPostgres)
library(tidyverse)
library(posterior)
library(brms)

# Connect to database
conn <- dbConnect(
  Postgres(),
  host = "localhost",
  port = 5544,
  dbname = "devdb01",
  user = "dro",
  password = "sicillionbillions"
)

cat("Loading trained model...\n")
model <- readRDS("models/bayesian/passing_yards_hierarchical_v1.rds")

cat("Extracting model data and coefficients...\n")
data <- model$data

# Get posterior samples
posterior_samples <- as_draws_df(model)

# Extract fixed effects (population-level parameters)
intercept_samples <- posterior_samples$b_Intercept
log_attempts_coef <- posterior_samples$b_log_attempts
is_home_coef <- posterior_samples$b_is_home
is_favored_coef <- posterior_samples$b_is_favored
spread_abs_coef <- posterior_samples$b_spread_abs
is_bad_weather_coef <- posterior_samples$b_is_bad_weather
is_dome_coef <- posterior_samples$b_is_dome
total_line_coef <- posterior_samples$`b_scaletotal_line`

# Extract experience category effects (lowercase in posterior samples)
exp_rookie_coef <- posterior_samples$`b_experience_catrookie`

# Extract player random effects
player_cols <- grep("^r_player_id\\[", names(posterior_samples), value = TRUE)
cat(glue::glue("Found {length(player_cols)} player effects\n"))

# Build ratings dataframe with actual predictions
ratings_list <- list()

for(player_col in player_cols) {
  player_id <- gsub("r_player_id\\[(.+),Intercept\\]", "\\1", player_col)

  if(player_id %in% unique(data$player_id)) {
    # Get player-specific effect
    player_effect <- posterior_samples[[player_col]]

    # Get player's typical game characteristics from training data
    player_data <- data %>%
      filter(player_id == !!player_id) %>%
      summarize(
        typical_log_attempts = mean(log_attempts, na.rm = TRUE),
        pct_home = mean(is_home, na.rm = TRUE),
        pct_favored = mean(is_favored, na.rm = TRUE),
        avg_spread_abs = mean(spread_abs, na.rm = TRUE),
        pct_bad_weather = mean(is_bad_weather, na.rm = TRUE),
        pct_dome = mean(is_dome, na.rm = TRUE),
        avg_total_line = mean(total_line, na.rm = TRUE),
        experience = first(experience_cat)
      )

    # Calculate actual attempts from log_attempts
    typical_attempts <- exp(player_data$typical_log_attempts)

    # Calculate experience effect (only rookie has coefficient, rest are reference)
    exp_effect <- if_else(
      tolower(player_data$experience) == "rookie",
      exp_rookie_coef,
      0  # Other levels are reference
    )

    # Calculate log-space prediction for typical game
    # Linear predictor: Intercept + player_effect + covariates * betas
    log_yards_pred <- (
      intercept_samples +
      player_effect +
      log_attempts_coef * player_data$typical_log_attempts +
      is_home_coef * player_data$pct_home +
      is_favored_coef * player_data$pct_favored +
      spread_abs_coef * player_data$avg_spread_abs +
      is_bad_weather_coef * player_data$pct_bad_weather +
      is_dome_coef * player_data$pct_dome +
      total_line_coef * scale(player_data$avg_total_line)[1] +
      exp_effect
    )

    # Transform back to yards scale
    yards_pred <- exp(log_yards_pred)

    # Calculate summary statistics in YARDS (not log-yards)
    ratings_list[[player_id]] <- tibble(
      player_id = player_id,
      stat_type = "passing_yards",
      season = 2024L,
      model_version = "hierarchical_v1.0",

      # Predictions in YARDS
      rating_mean = mean(yards_pred),
      rating_sd = sd(yards_pred),
      rating_q05 = quantile(yards_pred, 0.05),
      rating_q50 = quantile(yards_pred, 0.50),
      rating_q95 = quantile(yards_pred, 0.95),

      # Context for interpretation
      typical_attempts = typical_attempts,

      # Raw player effect (log-space) for reference
      player_effect_log = mean(player_effect),
      player_effect_log_sd = sd(player_effect),

      # Hierarchical components (keep as 0 for now - would need to extract from model)
      position_group_mean = 0.0,
      team_effect = 0.0,
      vs_opponent_effect = NA_real_,

      # Quality metrics
      n_games_observed = sum(data$player_id == player_id),
      effective_sample_size = posterior::ess_bulk(player_effect),
      rhat = posterior::rhat(player_effect),

      # Timestamps
      trained_at = Sys.time(),
      updated_at = Sys.time()
    )
  }
}

ratings_df <- bind_rows(ratings_list)
cat(glue::glue("Extracted {nrow(ratings_df)} ratings with yard predictions\n\n"))

# Delete old ratings
cat("Deleting old ratings...\n")
dbExecute(conn, "DELETE FROM mart.bayesian_player_ratings WHERE stat_type = 'passing_yards' AND model_version = 'hierarchical_v1.0'")

cat("Inserting ratings with yard predictions using SQL...\n")

for(i in 1:nrow(ratings_df)) {
  row <- ratings_df[i,]

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
      %f, %f, %s,
      %d, %f, %f,
      '%s', '%s'
    )",
    row$player_id, row$stat_type, row$season, row$model_version,
    row$rating_mean, row$rating_sd, row$rating_q05, row$rating_q50, row$rating_q95,
    row$position_group_mean, row$team_effect,
    ifelse(is.na(row$vs_opponent_effect), "NULL", row$vs_opponent_effect),
    row$n_games_observed, row$effective_sample_size, row$rhat,
    format(row$trained_at, "%Y-%m-%d %H:%M:%S"),
    format(row$updated_at, "%Y-%m-%d %H:%M:%S")
  )

  dbExecute(conn, sql)

  if(i %% 10 == 0) {
    cat(glue::glue("  Inserted {i}/{nrow(ratings_df)} ratings\r"))
  }
}

cat("\n")
dbDisconnect(conn)

cat("\n✅ Successfully saved", nrow(ratings_df), "ratings with yard predictions to database!\n\n")
cat("Predictions are now in YARDS (not log-space)\n")
cat("rating_mean = predicted yards for typical game scenario\n\n")

cat("Top 10 QBs by predicted yards:\n")
print(ratings_df %>%
  arrange(desc(rating_mean)) %>%
  select(player_id, rating_mean, rating_sd, typical_attempts, n_games_observed) %>%
  head(10))

cat("\nComparison with log-space player effects:\n")
print(ratings_df %>%
  arrange(desc(rating_mean)) %>%
  select(player_id, rating_mean, player_effect_log, typical_attempts) %>%
  head(10))
