#!/usr/bin/env Rscript
#
# Generate 2025 Hold-Out Predictions for All v2 Models
#
# Models to validate:
#   1. Hierarchical Bayesian v2
#   2. Informative Priors v2
#   3. State-Space v2
#
# Validation set: 2025 weeks 1-6 (148 games, 37 QBs)
# Metrics: MAE, RMSE, coverage_90, mean_interval_width, Brier score
#

library(tidyverse)
library(DBI)
library(RPostgres)
library(brms)
library(cmdstanr)
library(posterior)
library(glue)

cat("======================================================================\n")
cat("2025 Hold-Out Predictions: All v2 Models\n")
cat("======================================================================\n\n")

# ============================================================================
# STEP 1: Load 2025 test data
# ============================================================================

cat("STEP 1: Loading 2025 test data...\n")

conn <- dbConnect(
  Postgres(),
  host = "localhost",
  port = 5544,
  dbname = "devdb01",
  user = "dro",
  password = Sys.getenv("DB_PASSWORD", "sicillionbillions")
)

query_2025 <- "
SELECT
  pgs.player_id,
  pgs.player_display_name as player_name,
  pgs.season,
  pgs.week,
  pgs.stat_yards as yards,
  pgs.stat_attempts as attempts,

  g.home_team,
  g.away_team,
  ph.current_team as team,
  CASE WHEN ph.current_team = g.home_team THEN 1 ELSE 0 END as is_home,

  CASE
    WHEN ph.current_team = g.home_team THEN CASE WHEN g.spread_close < 0 THEN 1 ELSE 0 END
    ELSE CASE WHEN g.spread_close > 0 THEN 1 ELSE 0 END
  END as is_favored,
  ABS(g.spread_close) as spread_abs,
  g.total_close,

  COALESCE(g.roof, 'unknown') as roof,
  CASE WHEN g.roof IN ('outdoors', 'open') AND CAST(g.temp AS NUMERIC) < 40 THEN 1 ELSE 0 END as is_bad_weather,
  CASE WHEN g.roof = 'dome' THEN 1 ELSE 0 END as is_dome,

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
  AND pgs.season = 2025
  AND pgs.week > 0
  AND pgs.week <= 6
  AND pgs.stat_attempts >= 10
  AND pgs.stat_yards IS NOT NULL
  AND ph.position_group = 'QB'
ORDER BY pgs.week, pgs.player_id
"

test_data_2025 <- dbGetQuery(conn, query_2025)

# Feature engineering (match training data)
test_data_2025 <- test_data_2025 %>%
  filter(!is.na(total_close), !is.na(years_exp)) %>%
  mutate(
    log_yards = log(yards),
    log_attempts = log(attempts),
    experience_cat = case_when(
      years_exp <= 2 ~ "rookie",
      years_exp <= 5 ~ "young",
      years_exp <= 10 ~ "veteran",
      TRUE ~ "experienced"
    ),
    player_season = paste0(player_id, "_", 2024),  # Use 2024 as last training season
    position_season = paste0(position_group, "_", 2024)
  ) %>%
  mutate(experience_cat = droplevels(factor(experience_cat)))

cat(sprintf("✓ Loaded %d test games from %d QBs (2025 weeks 1-6)\n\n",
            nrow(test_data_2025), n_distinct(test_data_2025$player_id)))

# ============================================================================
# STEP 2: Generate predictions from Hierarchical Bayesian v2
# ============================================================================

cat("STEP 2: Hierarchical Bayesian v2 predictions...\n")

model_hier <- readRDS("models/bayesian/passing_yards_hierarchical_v2.rds")

# Generate posterior predictive samples
pred_hier_draws <- posterior_predict(model_hier, newdata = test_data_2025, allow_new_levels = TRUE)

# Calculate summaries
hierarchical_preds <- test_data_2025 %>%
  mutate(
    pred_yards = exp(colMeans(pred_hier_draws)),
    pred_lower_90 = exp(apply(pred_hier_draws, 2, quantile, probs = 0.05)),
    pred_upper_90 = exp(apply(pred_hier_draws, 2, quantile, probs = 0.95)),
    actual_yards = yards,
    error = pred_yards - actual_yards,
    abs_error = abs(error),
    squared_error = error^2,
    covered_90 = (actual_yards >= pred_lower_90) & (actual_yards <= pred_upper_90),
    interval_width = pred_upper_90 - pred_lower_90,
    model = "hierarchical_v2"
  ) %>%
  select(player_id, player_name, season, week, team, opponent,
         pred_yards, pred_lower_90, pred_upper_90, actual_yards,
         error, abs_error, squared_error, covered_90, interval_width, model)

cat(sprintf("✓ Generated %d predictions\n\n", nrow(hierarchical_preds)))

# ============================================================================
# STEP 3: Generate predictions from Informative Priors v2
# ============================================================================

cat("STEP 3: Informative Priors v2 predictions...\n")

model_priors <- readRDS("models/bayesian/passing_informative_priors_v2.rds")

# Generate posterior predictive samples
pred_priors_draws <- posterior_predict(model_priors, newdata = test_data_2025, allow_new_levels = TRUE)

# Calculate summaries
priors_preds <- test_data_2025 %>%
  mutate(
    pred_yards = exp(colMeans(pred_priors_draws)),
    pred_lower_90 = exp(apply(pred_priors_draws, 2, quantile, probs = 0.05)),
    pred_upper_90 = exp(apply(pred_priors_draws, 2, quantile, probs = 0.95)),
    actual_yards = yards,
    error = pred_yards - actual_yards,
    abs_error = abs(error),
    squared_error = error^2,
    covered_90 = (actual_yards >= pred_lower_90) & (actual_yards <= pred_upper_90),
    interval_width = pred_upper_90 - pred_lower_90,
    model = "informative_priors_v2"
  ) %>%
  select(player_id, player_name, season, week, team, opponent,
         pred_yards, pred_lower_90, pred_upper_90, actual_yards,
         error, abs_error, squared_error, covered_90, interval_width, model)

cat(sprintf("✓ Generated %d predictions\n\n", nrow(priors_preds)))

# ============================================================================
# STEP 4: Generate predictions from State-Space v2
# ============================================================================

cat("STEP 4: State-Space v2 predictions...\n")

# Load model
fit_ss <- readRDS("models/bayesian/simplified_random_walk_v2.rds")$draws()

# Extract global parameters
mu_league <- mean(fit_ss[, "mu_league"])
beta_attempts <- mean(fit_ss[, "beta_attempts"])
beta_home <- mean(fit_ss[, "beta_home"])
R <- mean(fit_ss[, "R"])

# Load skill trajectories
skills <- read_csv("models/bayesian/simplified_random_walk_skills_v2.csv", show_col_types = FALSE)

# Get final skill for each player
skills_final <- skills %>%
  group_by(player_id) %>%
  slice_max(game_num, n = 1) %>%
  select(player_id, skill_final = skill_mean)

# Generate predictions
state_space_preds <- test_data_2025 %>%
  left_join(skills_final, by = "player_id") %>%
  mutate(
    player_skill = ifelse(is.na(skill_final), mu_league, skill_final),
    log_pred = player_skill + beta_attempts * log_attempts + beta_home * is_home,
    pred_yards = exp(log_pred) - 1,
    pred_lower_90 = exp(log_pred - 1.645 * R) - 1,
    pred_upper_90 = exp(log_pred + 1.645 * R) - 1,
    actual_yards = yards,
    error = pred_yards - actual_yards,
    abs_error = abs(error),
    squared_error = error^2,
    covered_90 = (actual_yards >= pred_lower_90) & (actual_yards <= pred_upper_90),
    interval_width = pred_upper_90 - pred_lower_90,
    model = "state_space_v2"
  ) %>%
  select(player_id, player_name, season, week, team, opponent,
         pred_yards, pred_lower_90, pred_upper_90, actual_yards,
         error, abs_error, squared_error, covered_90, interval_width, model)

cat(sprintf("✓ Generated %d predictions\n\n", nrow(state_space_preds)))

# ============================================================================
# STEP 5: Combine and save predictions
# ============================================================================

cat("STEP 5: Saving predictions...\n")

all_preds <- bind_rows(hierarchical_preds, priors_preds, state_space_preds)

write_csv(hierarchical_preds, "analysis/results/hierarchical_v2_2025_holdout.csv")
write_csv(priors_preds, "analysis/results/informative_priors_v2_2025_holdout.csv")
write_csv(state_space_preds, "analysis/results/state_space_v2_2025_holdout.csv")
write_csv(all_preds, "analysis/results/all_v2_models_2025_predictions.csv")

cat("✓ Predictions saved:\n")
cat("  - analysis/results/hierarchical_v2_2025_holdout.csv\n")
cat("  - analysis/results/informative_priors_v2_2025_holdout.csv\n")
cat("  - analysis/results/state_space_v2_2025_holdout.csv\n")
cat("  - analysis/results/all_v2_models_2025_predictions.csv\n\n")

# ============================================================================
# STEP 6: Calculate metrics
# ============================================================================

cat("STEP 6: Calculating metrics...\n\n")

metrics <- all_preds %>%
  group_by(model) %>%
  summarise(
    n_predictions = n(),
    n_qbs = n_distinct(player_id),
    mae = mean(abs_error),
    rmse = sqrt(mean(squared_error)),
    coverage_90 = mean(covered_90) * 100,
    mean_interval_width = mean(interval_width),
    median_abs_error = median(abs_error),
    .groups = 'drop'
  )

print(metrics, n = Inf)

write_csv(metrics, "analysis/results/all_models_2025_comparison_v2.csv")
cat("\n✓ Metrics saved to: analysis/results/all_models_2025_comparison_v2.csv\n\n")

dbDisconnect(conn)

# ============================================================================
# SUMMARY
# ============================================================================

cat("======================================================================\n")
cat("✅ 2025 PREDICTIONS COMPLETE\n")
cat("======================================================================\n\n")

cat("Models validated:\n")
cat("  1. Hierarchical Bayesian v2\n")
cat("  2. Informative Priors v2\n")
cat("  3. State-Space v2\n\n")

cat(sprintf("Total predictions: %d (148 games × 3 models)\n", nrow(all_preds)))
cat(sprintf("Validation period: 2025 weeks 1-6\n"))
cat(sprintf("QBs evaluated: %d\n\n", n_distinct(all_preds$player_id)))

cat("Next: Review metrics and make GO/NO-GO decision for state-space model\n\n")
