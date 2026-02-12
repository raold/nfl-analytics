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

cat(paste0(rep("=", 80), collapse = ""), "\n")
cat("HIERARCHICAL BAYESIAN MODEL v2 - CANONICAL TRAINING WINDOW\n")
cat(paste0(rep("=", 80), collapse = ""), "\n")
cat("Training: 2015-2024 (uniform, equivalent to 2018-2024 + decay)\n")
cat(paste0(rep("=", 80), collapse = ""), "\n\n")

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
  AND pgs.stat_attempts >= 10  -- Minimum attempts
  AND pgs.season BETWEEN 2015 AND 2024
  AND ph.position_group = 'QB'
ORDER BY pgs.season, pgs.week, pgs.player_id
"

data <- dbGetQuery(conn, query)

# Input validation
stopifnot(
  "No data returned from query" = nrow(data) > 0,
  "Insufficient data for modeling" = nrow(data) >= 100,
  "Missing required columns" = all(c("player_id", "yards", "attempts", "is_home") %in% colnames(data)),
  "Invalid yards values" = all(data$yards >= 0, na.rm = TRUE),
  "Invalid attempts values" = all(data$attempts > 0, na.rm = TRUE),
  "Missing season information" = !all(is.na(data$season))
)

cat(glue("Loaded {nrow(data)} passing records\n"))
cat(glue("Seasons: {min(data$season)}-{max(data$season)}\n"))
cat(glue("Unique QBs: {n_distinct(data$player_id)}\n\n"))

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
cat("Step 3: Engineering features...\n")

# Filter out NAs first
n_before_filter <- nrow(data)
data <- data %>%
  filter(!is.na(total_close), !is.na(years_exp))

n_filtered <- n_before_filter - nrow(data)
if (n_filtered > 0) {
  cat(sprintf(
    "  Filtered %d rows with missing total_close or years_exp (%.1f%%)\n",
    n_filtered, 100 * n_filtered / n_before_filter
  ))
}

data <- data %>%
  mutate(
    # Log transformations
    log_yards = log(yards),
    log_attempts = log(attempts),

    # Interaction terms
    player_season = paste0(player_id, "_", season)
  )

# Validate after feature engineering
stopifnot(
  "Insufficient data after filtering" = nrow(data) >= 100,
  "Invalid log_yards (contains Inf/NaN)" = all(is.finite(data$log_yards)),
  "Invalid log_attempts (contains Inf/NaN)" = all(is.finite(data$log_attempts)),
  "Too few unique players" = n_distinct(data$player_id) >= 10,
  "Too few unique seasons" = n_distinct(data$season) >= 2
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
    scale(years_exp) + # Use continuous instead of categorical
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

model <- tryCatch(
  brm(
    formula = formula,
    data = data,
    family = gaussian(),
    prior = priors,
    chains = 10,
    iter = 4000,
    warmup = 2000,
    cores = 10,
    threads = threading(2),
    backend = "cmdstanr",
    control = list(adapt_delta = 0.95, max_treedepth = 12),
    seed = 42
  ),
  error = function(e) {
    cat(sprintf("\n❌ Model training FAILED: %s\n", e$message))
    # Clean up partial artifacts
    partial_rds <- "models/bayesian/passing_hierarchical_v2.rds"
    if (file.exists(partial_rds)) file.remove(partial_rds)
    if (exists("conn") && inherits(conn, "PqConnection")) dbDisconnect(conn)
    stop(e)
  }
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
# 6. PRIOR PREDICTIVE CHECKS
# ============================================================================
cat("Step 6: Prior predictive checks...\n")

# Sample from prior
cat("  Sampling from prior distribution...\n")
prior_model <- update(model,
  sample_prior = "only", chains = 2, iter = 1000,
  warmup = 500, seed = 42
)

# Generate prior predictions
prior_preds <- posterior_predict(prior_model, ndraws = 1000)
prior_yards <- exp(prior_preds) # Convert from log scale

# Check if priors produce reasonable predictions
prior_mean <- mean(prior_yards)
prior_q05 <- quantile(prior_yards, 0.05)
prior_q95 <- quantile(prior_yards, 0.95)

cat(sprintf("  Prior predictive distribution:\n"))
cat(sprintf("    Mean: %.1f yards\n", prior_mean))
cat(sprintf("    5th percentile: %.1f yards\n", prior_q05))
cat(sprintf("    95th percentile: %.1f yards\n", prior_q95))

# Validate priors are reasonable (50-500 yard range for 95% interval)
if (prior_q05 < 0 || prior_q95 > 600) {
  cat("\n⚠️  WARNING: Prior predictions outside reasonable range\n")
  cat("    Expected 95% interval: 50-500 yards\n")
  cat(sprintf("    Actual 95%% interval: %.1f-%.1f yards\n", prior_q05, prior_q95))
} else {
  cat("  ✅ Prior predictions within reasonable range\n")
}

cat("\n")

# ============================================================================
# 7. POSTERIOR PREDICTIVE CHECKS
# ============================================================================
cat("Step 7: Posterior predictive checks...\n")

# Check distributional fit
library(bayesplot)
pp <- pp_check(model, type = "dens_overlay", ndraws = 100)
ggsave("models/bayesian/hierarchical_v2_pp_check_density.png", pp,
  width = 8, height = 6
)
cat("  ✅ Density overlay plot saved\n")

# Check calibration (90% intervals)
pp_int <- pp_check(model, type = "intervals", prob = 0.90)
ggsave("models/bayesian/hierarchical_v2_pp_check_intervals.png", pp_int,
  width = 10, height = 6
)
cat("  ✅ Calibration intervals plot saved\n")

# Check prediction accuracy
pp_stat <- pp_check(model, type = "stat", stat = "mean")
ggsave("models/bayesian/hierarchical_v2_pp_check_mean.png", pp_stat,
  width = 8, height = 6
)
cat("  ✅ Mean prediction check saved\n")

cat("\n")

# ============================================================================
# 8. RESIDUAL DIAGNOSTICS
# ============================================================================
cat("Step 8: Residual diagnostics...\n")

# Extract residuals
residuals <- residuals(model)[, "Estimate"]

# Q-Q plot for normality
png("models/bayesian/hierarchical_v2_qq_plot.png", width = 800, height = 600)
qqnorm(residuals, main = "Q-Q Plot: Hierarchical v2 Residuals")
qqline(residuals, col = "red", lwd = 2)
dev.off()
cat("  ✅ Q-Q plot saved\n")

# Shapiro-Wilk test (on sample if too large)
n_sample <- min(5000, length(residuals))
shapiro_test <- shapiro.test(sample(residuals, n_sample))
cat(sprintf(
  "  Shapiro-Wilk test: W = %.4f, p = %.4f\n",
  shapiro_test$statistic, shapiro_test$p.value
))

if (shapiro_test$p.value < 0.01) {
  cat("  ⚠️  WARNING: Residuals significantly non-normal (p < 0.01)\n")
} else {
  cat("  ✅ Residuals consistent with normality\n")
}

# Residual histogram
png("models/bayesian/hierarchical_v2_residual_hist.png", width = 800, height = 600)
hist(residuals,
  breaks = 50, main = "Residual Distribution",
  xlab = "Residual (log yards)", col = "lightblue"
)
dev.off()
cat("  ✅ Residual histogram saved\n")

# Autocorrelation check (time series structure)
png("models/bayesian/hierarchical_v2_acf.png", width = 800, height = 600)
acf(residuals, main = "Autocorrelation: Hierarchical v2 Residuals")
dev.off()
cat("  ✅ Autocorrelation plot saved\n")

cat("\n")

# ============================================================================
# 9. SAVE MODEL
# ============================================================================
cat("Step 9: Saving model...\n")

model_path <- "models/bayesian/passing_yards_hierarchical_v2.rds"
saveRDS(model, model_path)
cat(glue("✓ Model saved to {model_path}\n"))

file_size_mb <- file.size(model_path) / 1024^2
cat(glue("✓ Model size: {round(file_size_mb, 1)} MB\n\n"))

# ============================================================================
# SUMMARY
# ============================================================================
cat(paste0(rep("=", 80), collapse = ""), "\n")
cat("✅ HIERARCHICAL BAYESIAN MODEL v2 TRAINING COMPLETE\n")
cat(paste0(rep("=", 80), collapse = ""), "\n\n")

cat("Summary:\n")
cat(glue("  - Training window: 2015-2024 (uniform, {nrow(data)} games)\n"))
cat(glue("  - Training time: {round(elapsed, 1)} minutes\n"))
cat(glue("  - Max Rhat: {round(max_rhat, 4)}\n"))
cat(glue("  - Min ESS: {round(min_ess, 0)}\n"))
cat(glue("  - Shapiro-Wilk p-value: {round(shapiro_test$p.value, 4)}\n"))
cat(glue("  - Model file: {model_path}\n"))
cat(glue("  - Model size: {round(file_size_mb, 1)} MB\n"))
cat("  - Diagnostic plots:\n")
cat("    - models/bayesian/hierarchical_v2_pp_check_density.png\n")
cat("    - models/bayesian/hierarchical_v2_pp_check_intervals.png\n")
cat("    - models/bayesian/hierarchical_v2_qq_plot.png\n")
cat("    - models/bayesian/hierarchical_v2_residual_hist.png\n")
cat("    - models/bayesian/hierarchical_v2_acf.png\n\n")

dbDisconnect(conn)

cat("Next: Generate 2025 predictions for validation\n\n")
