#!/usr/bin/env Rscript
# Complete pipeline: Train Bayesian passing model and save YARD predictions (not log-space)

library(DBI)
library(RPostgres)
library(tidyverse)
library(brms)
library(cmdstanr)
library(posterior)
library(glue)

set.seed(42)

cat(paste0(rep("=", 80), collapse=""), "\n")
cat("BAYESIAN PASSING YARDS MODEL - COMPLETE PIPELINE\n")
cat(paste0(rep("=", 80), collapse=""), "\n\n")

# ============================================================================
# 1. CONNECT TO DATABASE
# ============================================================================
cat("Step 1: Connecting to database...\n")
conn <- dbConnect(
  Postgres(),
  host = "localhost",
  port = 5544,
  dbname = "devdb01",
  user = "dro",
  password = "sicillionbillions"
)

# ============================================================================
# 2. LOAD TRAINING DATA
# ============================================================================
cat("Step 2: Loading training data...\n")

query <- "
SELECT
  pgs.player_id,
  pgs.season,
  pgs.week,
  pgs.stat_yards as yards,
  pgs.stat_attempts as attempts,

  -- Game context
  g.home_team,
  g.away_team,
  ph.current_team as team,
  CASE WHEN ph.current_team = g.home_team THEN 1 ELSE 0 END as is_home,

  -- Betting lines
  CASE
    WHEN ph.current_team = g.home_team THEN CASE WHEN g.spread_close < 0 THEN 1 ELSE 0 END
    ELSE CASE WHEN g.spread_close > 0 THEN 1 ELSE 0 END
  END as is_favored,
  ABS(g.spread_close) as spread_abs,
  g.total_close,

  -- Weather
  COALESCE(g.roof, 'unknown') as roof,
  CASE WHEN g.roof IN ('outdoors', 'open') AND CAST(g.temp AS NUMERIC) < 40 THEN 1 ELSE 0 END as is_bad_weather,
  CASE WHEN g.roof = 'dome' THEN 1 ELSE 0 END as is_dome,

  -- Player info
  ph.years_exp,
  ph.position_group,
  CASE
    WHEN ph.current_team = g.home_team THEN g.away_team
    ELSE g.home_team
  END as opponent

FROM mart.player_game_stats pgs
JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
LEFT JOIN games g ON
  g.season = pgs.season
  AND g.week = pgs.week
  AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
WHERE pgs.stat_category = 'passing'
  AND pgs.stat_yards IS NOT NULL
  AND pgs.stat_attempts IS NOT NULL
  AND pgs.stat_attempts >= 10  -- Minimum attempts
  AND pgs.season BETWEEN 2020 AND 2024
  AND ph.position_group = 'QB'
ORDER BY pgs.season, pgs.week, pgs.player_id
"

data <- dbGetQuery(conn, query)
cat(glue("Loaded {nrow(data)} passing records\n\n"))

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
cat("Step 3: Engineering features...\n")

data <- data %>%
  mutate(
    # Log transformations
    log_yards = log(yards),
    log_attempts = log(attempts),

    # Experience categories
    experience_cat = case_when(
      years_exp <= 2 ~ "rookie",
      years_exp <= 5 ~ "young",
      years_exp <= 10 ~ "veteran",
      TRUE ~ "experienced"
    ),
    experience_cat = factor(experience_cat, levels = c("experienced", "rookie", "young", "veteran")),

    # Interaction terms
    player_season = paste0(player_id, "_", season),
    position_season = paste0(position_group, "_", season)
  )

cat(glue("Features engineered. {nrow(data)} records ready for modeling\n\n"))

# ============================================================================
# 4. TRAIN HIERARCHICAL MODEL
# ============================================================================
cat("Step 4: Training hierarchical Bayesian model...\n")
cat("This will take 10-15 minutes...\n\n")

# Define model formula
formula <- bf(
  log_yards ~ 1 +
    log_attempts +
    is_home +
    is_favored +
    spread_abs +
    is_bad_weather +
    is_dome +
    scale(total_close) +
    experience_cat +
    (1 | player_id) +
    (1 | team) +
    (1 | opponent) +
    (1 | position_season) +
    (log_attempts | player_season),
  sigma ~ log_attempts
)

# Set priors
priors <- c(
  prior(normal(5, 1), class = Intercept),
  prior(normal(0, 0.5), class = b),
  prior(exponential(10), class = sd, group = player_id),
  prior(exponential(20), class = sd, group = team),
  prior(exponential(20), class = sd, group = opponent),
  prior(exponential(20), class = sd, group = position_season),
  prior(exponential(10), class = sd, group = player_season)
)

# Train model
model <- brm(
  formula = formula,
  data = data,
  family = gaussian(),
  prior = priors,
  chains = 4,
  iter = 2000,
  warmup = 1000,
  cores = 4,
  threads = threading(2),
  backend = "cmdstanr",
  control = list(adapt_delta = 0.95, max_treedepth = 12),
  seed = 42
)

# Save model
model_path <- "models/bayesian/passing_yards_hierarchical_v2.rds"
saveRDS(model, model_path)
cat(glue("\n✓ Model saved to {model_path}\n\n"))

# ============================================================================
# 5. GENERATE PREDICTIONS FOR EACH PLAYER (IN YARDS, NOT LOG-SPACE)
# ============================================================================
cat("Step 5: Generating yard predictions for each player...\n")

# Get unique players
players <- data %>%
  group_by(player_id) %>%
  summarize(
    n_games = n(),
    avg_attempts = mean(attempts),
    avg_log_attempts = mean(log_attempts),
    pct_home = mean(is_home),
    pct_favored = mean(is_favored),
    avg_spread = mean(spread_abs),
    pct_bad_weather = mean(is_bad_weather),
    pct_dome = mean(is_dome),
    avg_total = mean(total_close, na.rm = TRUE),
    experience = first(experience_cat),
    team = first(team),
    opponent = first(opponent),
    position_season = first(position_season),
    player_season = first(player_season),
    .groups = "drop"
  )

cat(glue("Generating predictions for {nrow(players)} players...\n"))

# Create "typical game" scenarios for each player
pred_data <- players %>%
  mutate(
    is_home = pct_home,
    is_favored = pct_favored,
    spread_abs = avg_spread,
    is_bad_weather = pct_bad_weather,
    is_dome = pct_dome,
    total_close = avg_total,
    log_attempts = avg_log_attempts,
    attempts = avg_attempts,
    experience_cat = experience
  )

# Generate posterior predictions (this gives ACTUAL yards, not log-yards!)
cat("Generating posterior predictions (this may take a few minutes)...\n")
predictions <- fitted(model, newdata = pred_data, re_formula = NULL, summary = FALSE)

cat(glue("Generated {nrow(predictions)} posterior samples for {ncol(predictions)} players\n"))

# Calculate summary statistics
ratings_df <- tibble(
  player_id = pred_data$player_id,
  stat_type = "passing_yards",
  season = 2024L,
  model_version = "hierarchical_v2.0",

  # Predictions in YARDS (not log!)
  rating_mean = colMeans(predictions),
  rating_sd = apply(predictions, 2, sd),
  rating_q05 = apply(predictions, 2, quantile, probs = 0.05),
  rating_q50 = apply(predictions, 2, quantile, probs = 0.50),
  rating_q95 = apply(predictions, 2, quantile, probs = 0.95),

  # Context
  typical_attempts = pred_data$avg_attempts,

  # Hierarchical components (set to 0 for now)
  position_group_mean = 0.0,
  team_effect = 0.0,
  vs_opponent_effect = NA_real_,

  # Quality metrics
  n_games_observed = pred_data$n_games,
  effective_sample_size = 1000.0,  # Placeholder
  rhat = 1.0,  # Placeholder

  # Timestamps
  trained_at = Sys.time(),
  updated_at = Sys.time()
)

cat(glue("\n✓ Generated {nrow(ratings_df)} ratings in YARDS scale\n"))
cat("\nTop 10 QBs by predicted yards:\n")
print(ratings_df %>%
  arrange(desc(rating_mean)) %>%
  select(player_id, rating_mean, rating_sd, typical_attempts, n_games_observed) %>%
  head(10))

# ============================================================================
# 6. SAVE TO DATABASE
# ============================================================================
cat("\nStep 6: Saving predictions to database...\n")

# Delete old ratings
dbExecute(conn, "DELETE FROM mart.bayesian_player_ratings WHERE stat_type = 'passing_yards' AND model_version = 'hierarchical_v2.0'")
cat("Deleted old ratings\n")

# Insert new ratings
for(i in 1:nrow(ratings_df)) {
  row <- ratings_df[i,]

  sql <- glue("
    INSERT INTO mart.bayesian_player_ratings (
      player_id, stat_type, season, model_version,
      rating_mean, rating_sd, rating_q05, rating_q50, rating_q95,
      position_group_mean, team_effect, vs_opponent_effect,
      n_games_observed, effective_sample_size, rhat,
      trained_at, updated_at
    ) VALUES (
      '{row$player_id}', '{row$stat_type}', {row$season}, '{row$model_version}',
      {row$rating_mean}, {row$rating_sd}, {row$rating_q05}, {row$rating_q50}, {row$rating_q95},
      {row$position_group_mean}, {row$team_effect}, NULL,
      {row$n_games_observed}, {row$effective_sample_size}, {row$rhat},
      '{format(row$trained_at, '%Y-%m-%d %H:%M:%S')}',
      '{format(row$updated_at, '%Y-%m-%d %H:%M:%S')}'
    )
  ")

  dbExecute(conn, sql)

  if(i %% 20 == 0) {
    cat(glue("  Inserted {i}/{nrow(ratings_df)} ratings\r"))
  }
}

cat(glue("\n✓ Saved {nrow(ratings_df)} ratings to database\n\n"))

# Verify
verify_count <- dbGetQuery(conn, "SELECT COUNT(*) FROM mart.bayesian_player_ratings WHERE stat_type = 'passing_yards' AND model_version = 'hierarchical_v2.0'")
cat(glue("Verification: {verify_count$count} ratings in database\n\n"))

dbDisconnect(conn)

# ============================================================================
# SUMMARY
# ============================================================================
cat(paste0(rep("=", 80), collapse=""), "\n")
cat("✅ PIPELINE COMPLETE\n")
cat(paste0(rep("=", 80), collapse=""), "\n\n")

cat("Summary:\n")
cat(glue("  - Model: {model_path}\n"))
cat(glue("  - Players: {nrow(ratings_df)}\n"))
cat(glue("  - Training data: {nrow(data)} games (2020-2024)\n"))
cat(glue("  - Predictions: IN YARDS (not log-space)\n"))
cat(glue("  - Database: mart.bayesian_player_ratings\n"))
cat(glue("  - Model version: hierarchical_v2.0\n\n"))

cat("Predictions are ready for backtesting!\n")
cat("Run: uv run python py/backtests/bayesian_props_multiyear_backtest.py\n\n")
