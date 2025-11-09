#!/usr/bin/env Rscript
#
# Hierarchical Bayesian Model v2: Canonical Training Window
#
# Training window: 2015-2024 (uniform weights)
# Rationale: Ablation study showed Scenario B (2018-2024 + decay 2015-2017)
#            is equivalent to Scenario D (uniform 2015-2024).
#            Using uniform is simpler for brms (no weight implementation needed).
#
# This is the PRODUCTION model following empirical validation.
#

library(DBI)
library(RPostgres)
library(tidyverse)
library(brms)
library(cmdstanr)
library(posterior)
library(glue)

set.seed(42)

cat(paste0(rep("=", 80), collapse=""), "\n")
cat("HIERARCHICAL BAYESIAN MODEL v2 - CANONICAL TRAINING WINDOW\n")
cat(paste0(rep("=", 80), collapse=""), "\n")
cat("Training: 2015-2024 (uniform, equivalent to 2018-2024 + decay)\n")
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
# 2. LOAD TRAINING DATA (2015-2024)
# ============================================================================
cat("Step 2: Loading training data (2015-2024)...\n")

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
  AND pgs.season BETWEEN 2015 AND 2024
  AND ph.position_group = 'QB'
ORDER BY pgs.season, pgs.week, pgs.player_id
"

data <- dbGetQuery(conn, query)
cat(glue("Loaded {nrow(data)} passing records\n"))
cat(glue("Seasons: {min(data$season)}-{max(data$season)}\n"))
cat(glue("Unique QBs: {n_distinct(data$player_id)}\n\n"))

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
cat("Step 3: Engineering features...\n")

# Filter out NAs first
data <- data %>%
  filter(!is.na(total_close), !is.na(years_exp))

data <- data %>%
  mutate(
    # Log transformations
    log_yards = log(yards),
    log_attempts = log(attempts),

    # Interaction terms
    player_season = paste0(player_id, "_", season)
  )

cat(glue("Features engineered. {nrow(data)} records ready for modeling\n\n"))

# ============================================================================
# 4. TRAIN HIERARCHICAL MODEL
# ============================================================================
cat("Step 4: Training hierarchical Bayesian model...\n")
cat("This will take 15-20 minutes...\n\n")

# Define model formula (simplified - removed categorical experience_cat due to factor level issues)
formula <- bf(
  log_yards ~ 1 +
    log_attempts +
    is_home +
    is_favored +
    spread_abs +
    is_bad_weather +
    is_dome +
    scale(total_close) +
    scale(years_exp) +  # Use continuous instead of categorical
    (1 | player_id) +
    (1 | team) +
    (1 | opponent) +
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
  prior(exponential(10), class = sd, group = player_season)
)

# Train model
start_time <- Sys.time()

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

end_time <- Sys.time()
elapsed <- as.numeric(difftime(end_time, start_time, units = "mins"))

cat(sprintf("\n✓ Training complete in %.1f minutes\n\n", elapsed))

# ============================================================================
# 5. CONVERGENCE DIAGNOSTICS
# ============================================================================
cat("Step 5: Checking convergence...\n")

# Check Rhat and ESS
rhats <- rhat(model)
max_rhat <- max(rhats, na.rm = TRUE)
ess_values <- ess_bulk(model)
min_ess <- min(ess_values, na.rm = TRUE)

cat(sprintf("  Max Rhat: %.4f (threshold: < 1.01)\n", max_rhat))
cat(sprintf("  Min ESS: %.0f (threshold: > 400)\n", min_ess))

if (max_rhat > 1.01) {
  stop("❌ CONVERGENCE FAILURE: Rhat > 1.01")
}
if (min_ess < 400) {
  stop("❌ CONVERGENCE FAILURE: ESS < 400")
}

cat("\n✅ CONVERGENCE SUCCESS!\n\n")

# ============================================================================
# 6. SAVE MODEL
# ============================================================================
cat("Step 6: Saving model...\n")

model_path <- "models/bayesian/passing_yards_hierarchical_v2.rds"
saveRDS(model, model_path)
cat(glue("✓ Model saved to {model_path}\n"))

file_size_mb <- file.size(model_path) / 1024^2
cat(glue("✓ Model size: {round(file_size_mb, 1)} MB\n\n"))

# ============================================================================
# SUMMARY
# ============================================================================
cat(paste0(rep("=", 80), collapse=""), "\n")
cat("✅ HIERARCHICAL BAYESIAN MODEL v2 TRAINING COMPLETE\n")
cat(paste0(rep("=", 80), collapse=""), "\n\n")

cat("Summary:\n")
cat(glue("  - Training window: 2015-2024 (uniform, {nrow(data)} games)\n"))
cat(glue("  - Training time: {round(elapsed, 1)} minutes\n"))
cat(glue("  - Max Rhat: {round(max_rhat, 4)}\n"))
cat(glue("  - Min ESS: {round(min_ess, 0)}\n"))
cat(glue("  - Model file: {model_path}\n"))
cat(glue("  - Model size: {round(file_size_mb, 1)} MB\n\n"))

dbDisconnect(conn)

cat("Next: Generate 2025 predictions for validation\n\n")
