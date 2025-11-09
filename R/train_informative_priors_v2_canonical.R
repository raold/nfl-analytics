#!/usr/bin/env Rscript
#
# Informative Priors Model v2: Canonical Training Window
#
# Training window: 2015-2024 (uniform weights)
# Rationale: Same as hierarchical v2 - uniform 2015-2024 is equivalent
#            to 2018-2024 + decay per ablation study.
#
# This model uses informative priors based on empirical Bayes estimation
# from historical data (Chapter 3 of dissertation).
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
cat("INFORMATIVE PRIORS MODEL v2 - CANONICAL TRAINING WINDOW\n")
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
  password = Sys.getenv("DB_PASSWORD", "sicillionbillions")
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
  AND pgs.stat_attempts >= 10
  AND pgs.season BETWEEN 2015 AND 2024
  AND ph.position_group = 'QB'
ORDER BY pgs.season, pgs.week, pgs.player_id
"

data <- dbGetQuery(conn, query)
cat(glue("Loaded {nrow(data)} passing records\n"))
cat(glue("Seasons: {min(data$season)}-{max(data$season)}\n"))
cat(glue("Unique QBs: {n_distinct(data$player_id)}\n\n"))

# ============================================================================
# 3. EMPIRICAL BAYES PRIOR ESTIMATION
# ============================================================================
cat("Step 3: Estimating empirical Bayes priors...\n")

# Use first half of data (2015-2019) for prior estimation
prior_data <- data %>% filter(season <= 2019)

cat(glue("  Prior estimation dataset: {nrow(prior_data)} games\n"))

# Estimate key parameters
log_yards_mean <- mean(log(prior_data$yards))
log_yards_sd <- sd(log(prior_data$yards))
log_attempts_coef <- coef(lm(log(yards) ~ log(attempts), data = prior_data))[2]
home_effect <- mean(log(prior_data$yards[prior_data$is_home == 1])) -
               mean(log(prior_data$yards[prior_data$is_home == 0]))

cat(sprintf("  Intercept prior: N(%.2f, %.2f)\n", log_yards_mean, log_yards_sd / 2))
cat(sprintf("  log_attempts prior: N(%.3f, 0.1)\n", log_attempts_coef))
cat(sprintf("  home effect prior: N(%.3f, 0.01)\n", home_effect))
cat("\n")

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================
cat("Step 4: Engineering features...\n")

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
# 5. TRAIN MODEL WITH INFORMATIVE PRIORS
# ============================================================================
cat("Step 5: Training model with informative priors...\n")
cat("This will take 20-25 minutes...\n\n")

# Define model formula (simplified - use continuous years_exp instead of categorical)
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

# Set INFORMATIVE priors based on empirical Bayes
priors <- c(
  # Informative intercept (from historical data)
  prior(normal(5.3, 0.5), class = Intercept),

  # Informative attempts coefficient
  prior(normal(0.78, 0.1), class = b, coef = log_attempts),

  # Informative home advantage
  prior(normal(0.02, 0.01), class = b, coef = is_home),

  # Weakly informative for other fixed effects
  prior(normal(0, 0.3), class = b),

  # Hierarchical SDs
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
  iter = 4000,
  warmup = 2000,
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
# 6. CONVERGENCE DIAGNOSTICS
# ============================================================================
cat("Step 6: Checking convergence...\n")

# Check Rhat and ESS
rhats <- rhat(model)
max_rhat <- max(rhats, na.rm = TRUE)
ess_values <- ess_bulk(model)
min_ess <- min(ess_values, na.rm = TRUE)

cat(sprintf("  Max Rhat: %.4f (threshold: < 1.01)\n", max_rhat))
cat(sprintf("  Min ESS: %.0f (threshold: > 400)\n", min_ess))

# Check convergence (stop execution if failed)
convergence_failed <- FALSE
if (max_rhat > 1.01) {
  cat("\n❌ CONVERGENCE FAILURE: Rhat > 1.01\n")
  convergence_failed <- TRUE
}
if (min_ess < 400) {
  cat("\n❌ CONVERGENCE FAILURE: ESS < 400\n")
  convergence_failed <- TRUE
}

if (convergence_failed) {
  dbDisconnect(conn)
  stop("Model training failed convergence diagnostics. Increase iterations or adjust adapt_delta.")
}

cat("\n✅ CONVERGENCE SUCCESS!\n\n")

# ============================================================================
# 7. SAVE MODEL
# ============================================================================
cat("Step 7: Saving model...\n")

model_path <- "models/bayesian/passing_informative_priors_v2.rds"
saveRDS(model, model_path)
cat(glue("✓ Model saved to {model_path}\n"))

file_size_mb <- file.size(model_path) / 1024^2
cat(glue("✓ Model size: {round(file_size_mb, 1)} MB\n\n"))

# ============================================================================
# SUMMARY
# ============================================================================
cat(paste0(rep("=", 80), collapse=""), "\n")
cat("✅ INFORMATIVE PRIORS MODEL v2 TRAINING COMPLETE\n")
cat(paste0(rep("=", 80), collapse=""), "\n\n")

cat("Summary:\n")
cat(glue("  - Training window: 2015-2024 (uniform, {nrow(data)} games)\n"))
cat(glue("  - Prior estimation: 2015-2019 ({nrow(prior_data)} games)\n"))
cat(glue("  - Training time: {round(elapsed, 1)} minutes\n"))
cat(glue("  - Max Rhat: {round(max_rhat, 4)}\n"))
cat(glue("  - Min ESS: {round(min_ess, 0)}\n"))
cat(glue("  - Model file: {model_path}\n"))
cat(glue("  - Model size: {round(file_size_mb, 1)} MB\n\n"))

dbDisconnect(conn)

cat("Next: Generate 2025 predictions for validation\n\n")
