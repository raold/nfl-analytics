#!/usr/bin/env Rscript
#
# State-Space Model v2: Canonical Training Window
#
# Training window: 2018-2024 + H=4 exponential decay on 2015-2017
# Based on ablation study results (Scenario B)
#
# This is the PRODUCTION model following empirical validation.
#

library(tidyverse)
library(DBI)
library(RPostgres)
library(cmdstanr)
library(posterior)
library(bayesplot)

cat("======================================================================\n")
cat("State-Space Model v2: Canonical Training Window\n")
cat("======================================================================\n")
cat("Training: 2018-2024 (full weight) + 2015-2017 (H=4 decay)\n")
cat("Formula: w(s; 2025, 4) = 0.5^((2025-s)/4)\n")
cat("======================================================================\n\n")

# Configuration
START_YEAR <- 2015
END_YEAR <- 2024
DECAY_YEARS <- 2015:2017
H <- 4
EVAL_YEAR <- 2025
MIN_GAMES <- 10

# Exponential decay weight function
exp_weight <- function(s, t, H) {
  0.5 ^ ((t - s) / H)
}

# ============================================================================
# STEP 1: Load data from database
# ============================================================================

cat("STEP 1: Loading player game data from database...\n")

con <- dbConnect(
  RPostgres::Postgres(),
  host = "localhost",
  port = 5544,
  dbname = "devdb01",
  user = "dro",
  password = Sys.getenv("DB_PASSWORD", "sicillionbillions")
)

query <- "
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
      END AS is_home,
      ROW_NUMBER() OVER (PARTITION BY pgs.player_id ORDER BY pgs.season, pgs.week) AS game_num
    FROM mart.player_game_stats pgs
    LEFT JOIN public.games g ON (
      pgs.season = g.season AND
      pgs.week = g.week AND
      (pgs.current_team = g.home_team OR pgs.current_team = g.away_team)
    )
    WHERE pgs.season BETWEEN $1 AND $2
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

player_data <- dbGetQuery(con, query, params = list(START_YEAR, END_YEAR))
dbDisconnect(con)

cat(sprintf("  ✓ Loaded %d player-game records\n", nrow(player_data)))
cat(sprintf("  ✓ Unique players: %d\n", n_distinct(player_data$player_id)))
cat(sprintf("  ✓ Seasons: %d-%d\n", min(player_data$season), max(player_data$season)))

# ============================================================================
# STEP 2: Apply exponential decay weights
# ============================================================================

cat("\nSTEP 2: Applying exponential decay weights...\n")

player_data <- player_data %>%
  mutate(
    weight = ifelse(season %in% DECAY_YEARS, exp_weight(season, EVAL_YEAR, H), 1.0)
  )

# Show weight distribution
weight_summary <- player_data %>%
  group_by(season) %>%
  summarise(
    n_games = n(),
    mean_weight = mean(weight),
    .groups = 'drop'
  )

cat("\nWeight distribution by season:\n")
print(weight_summary, n = Inf)
cat("\n")

# ============================================================================
# STEP 3: Filter to players with sufficient data
# ============================================================================

cat("STEP 3: Filtering to players with sufficient data...\n")

player_counts <- player_data %>%
  group_by(player_id, player_name) %>%
  summarise(n_games = n(), effective_games = sum(weight), .groups = 'drop') %>%
  filter(effective_games >= MIN_GAMES)

cat(sprintf("  ✓ %d players with ≥%.1f effective games\n", nrow(player_counts), MIN_GAMES))

# Filter data to selected players
final_data <- player_data %>%
  filter(player_id %in% player_counts$player_id)

cat(sprintf("  ✓ Final dataset: %d games from %d players\n",
            nrow(final_data), n_distinct(final_data$player_id)))

# ============================================================================
# STEP 4: Prepare weighted data for Stan
# ============================================================================

cat("\nSTEP 4: Preparing weighted data for Stan...\n")

# Create player index
player_ids <- unique(final_data$player_id)
N_players <- length(player_ids)

# Get max games per player
games_per_player <- final_data %>%
  group_by(player_id) %>%
  summarise(n_games = n(), .groups = 'drop') %>%
  arrange(match(player_id, player_ids))

T_max <- max(games_per_player$n_games)
T_player <- games_per_player$n_games

cat(sprintf("  ✓ Maximum games per player: %d\n", T_max))

# Create ragged arrays with weights
y <- matrix(0, nrow = N_players, ncol = T_max)
log_attempts <- matrix(0, nrow = N_players, ncol = T_max)
is_home <- matrix(0L, nrow = N_players, ncol = T_max)
observed <- matrix(0L, nrow = N_players, ncol = T_max)
weights <- matrix(1.0, nrow = N_players, ncol = T_max)  # New: weight matrix

for (i in 1:N_players) {
  pid <- player_ids[i]
  player_games <- final_data %>% filter(player_id == pid) %>% arrange(season, week)
  n_games <- nrow(player_games)

  # Fill in observed games
  y[i, 1:n_games] <- log(player_games$passing_yards + 1)
  log_attempts[i, 1:n_games] <- log(player_games$passing_attempts)
  is_home[i, 1:n_games] <- player_games$is_home
  observed[i, 1:n_games] <- 1L
  weights[i, 1:n_games] <- player_games$weight  # Apply decay weights
}

# Stan data
stan_data <- list(
  N_players = N_players,
  T_player = T_player,
  T_max = T_max,
  y = y,
  observed = observed,
  log_attempts = log_attempts,
  is_home = is_home,
  weights = weights  # Pass weights to Stan
)

cat(sprintf("  ✓ Stan data prepared:\n"))
cat(sprintf("    - N_players: %d\n", N_players))
cat(sprintf("    - T_max: %d\n", T_max))
cat(sprintf("    - Total observations: %d\n", sum(observed)))
cat(sprintf("    - Effective observations: %.1f\n", sum(observed * weights)))

# ============================================================================
# STEP 5: Compile Stan model
# ============================================================================

cat("\nSTEP 5: Compiling Stan model...\n")

stan_file <- "models/stan/simplified_random_walk_weighted.stan"
mod <- cmdstan_model(stan_file)

cat("  ✓ Stan model compiled successfully\n")

# ============================================================================
# STEP 6: Fit model
# ============================================================================

cat("\nSTEP 6: Fitting model using HMC sampling...\n")
cat("  Configuration:\n")
cat("    - Chains: 4 (parallel)\n")
cat("    - Warmup iterations: 1000\n")
cat("    - Sampling iterations: 2000\n")
cat("    - Adapt delta: 0.95\n")
cat("    - Max tree depth: 12\n\n")

start_time <- Sys.time()

fit <- mod$sample(
  data = stan_data,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 2000,
  adapt_delta = 0.95,
  max_treedepth = 12,
  refresh = 100,
  seed = 42
)

end_time <- Sys.time()
elapsed <- as.numeric(difftime(end_time, start_time, units = "mins"))

cat(sprintf("\n  ✓ Sampling complete in %.1f minutes\n", elapsed))

# ============================================================================
# STEP 7: Convergence diagnostics
# ============================================================================

cat("\nSTEP 7: Convergence diagnostics...\n\n")

# Extract global parameters
global_params <- c("mu_league", "sigma_player", "Q", "R", "beta_attempts", "beta_home")

draws <- fit$draws(variables = global_params)
summary_df <- summarise_draws(draws)

cat("  Global parameters:\n")
print(summary_df %>% select(variable, mean, rhat, ess_bulk, ess_tail), n = Inf)
cat("\n")

# Check convergence
max_rhat <- max(summary_df$rhat, na.rm = TRUE)
min_ess <- min(summary_df$ess_bulk, na.rm = TRUE)

cat(sprintf("  ✓ Max Rhat: %.4f (threshold: < 1.01)\n", max_rhat))
cat(sprintf("  ✓ Min ESS: %.0f (threshold: > 400)\n", min_ess))

if (max_rhat > 1.01 || min_ess < 400) {
  stop("❌ CONVERGENCE FAILURE: Model did not converge properly")
}

cat("\n✅ CONVERGENCE SUCCESS!\n\n")

# ============================================================================
# STEP 8: Save model and predictions
# ============================================================================

cat("STEP 8: Saving model and predictions...\n")

# Save fitted model
fit$save_object("models/bayesian/simplified_random_walk_v2.rds")
cat("  ✓ Model saved to: models/bayesian/simplified_random_walk_v2.rds\n")

# Extract time-varying skills
theta_draws <- fit$draws(variables = "theta", format = "draws_matrix")
theta_mean <- colMeans(theta_draws)

# Reshape and save
skill_trajectories <- data.frame()
for (i in 1:N_players) {
  pid <- player_ids[i]
  player_info <- final_data %>% filter(player_id == pid) %>% head(1)

  for (t in 1:T_player[i]) {
    theta_idx <- sprintf("theta[%d,%d]", i, t)
    skill_trajectories <- bind_rows(
      skill_trajectories,
      data.frame(
        player_id = pid,
        player_name = player_info$player_name,
        game_num = t,
        skill_mean = theta_mean[theta_idx]
      )
    )
  }
}

write_csv(skill_trajectories, "models/bayesian/simplified_random_walk_skills_v2.csv")
cat("  ✓ Skill trajectories saved to: models/bayesian/simplified_random_walk_skills_v2.csv\n")

cat("\n======================================================================\n")
cat("✅ STATE-SPACE MODEL v2 TRAINING COMPLETE!\n")
cat("======================================================================\n")
cat(sprintf("Training window: 2018-2024 + H=4 decay on 2015-2017\n"))
cat(sprintf("Training time: %.1f minutes\n", elapsed))
cat(sprintf("Max Rhat: %.4f\n", max_rhat))
cat(sprintf("Min ESS: %.0f\n", min_ess))
cat(sprintf("Total games: %d\n", sum(observed)))
cat(sprintf("Effective games: %.1f\n", sum(observed * weights)))
cat(sprintf("Model file: models/bayesian/simplified_random_walk_v2.rds\n"))
cat(sprintf("Skills file: models/bayesian/simplified_random_walk_skills_v2.csv\n"))
cat("======================================================================\n\n")
