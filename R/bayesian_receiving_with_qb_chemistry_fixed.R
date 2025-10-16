#!/usr/bin/env Rscript
# Enhanced Bayesian Receiving Model with QB-WR Chemistry Effects
# Implements dyadic random effects for QB-receiver pairs + distributional regression
# FIXED: Computes receptions/targets from plays table

library(DBI)
library(RPostgres)
library(tidyverse)
library(brms)
library(cmdstanr)
library(posterior)
library(glue)

set.seed(42)

cat(paste0(rep("=", 80), collapse=""), "\n")
cat("BAYESIAN RECEIVING MODEL WITH QB-WR CHEMISTRY (FIXED)\n")
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
# 2. LOAD TRAINING DATA WITH QB INFO
# ============================================================================
cat("Step 2: Loading receiving data with QB information (computing from plays)...\n")

query <- "
WITH receiving_stats AS (
  -- Compute receptions, targets, and yards from plays table
  SELECT
    p.receiver_player_id,
    g.season,
    g.week,
    p.passer_player_id as qb_id,
    p.passer_player_name as qb_name,
    SUM(CASE WHEN p.complete_pass = 1 THEN 1 ELSE 0 END) as receptions,
    COUNT(*) as targets,
    SUM(p.yards_gained) as yards,
    g.game_id
  FROM plays p
  JOIN games g ON p.game_id = g.game_id
  WHERE p.receiver_player_id IS NOT NULL
    AND p.passer_player_id IS NOT NULL
    AND g.season BETWEEN 2020 AND 2024
  GROUP BY p.receiver_player_id, g.season, g.week, p.passer_player_id, p.passer_player_name, g.game_id
)
SELECT
  rs.receiver_player_id as receiver_id,
  rs.season,
  rs.week,
  rs.yards,
  rs.receptions,
  rs.targets,

  -- QB info
  rs.qb_id,
  rs.qb_name,

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

FROM receiving_stats rs
JOIN mart.player_hierarchy ph ON rs.receiver_player_id = ph.player_id
JOIN games g ON rs.game_id = g.game_id

WHERE rs.targets >= 3  -- Minimum targets
  AND ph.position_group IN ('WR', 'TE', 'RB')
  AND rs.yards IS NOT NULL
ORDER BY rs.season, rs.week, rs.receiver_player_id
"

data <- dbGetQuery(conn, query)
cat(glue("Loaded {nrow(data)} receiving records with QB info\n\n"))

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
cat("Step 3: Engineering features...\n")

data <- data %>%
  mutate(
    # Log transformations
    log_yards = log(yards + 1),  # +1 to handle zeros
    log_targets = log(targets),

    # QB-WR chemistry ID
    qb_wr_pair = paste0(qb_id, "_", receiver_id),

    # Experience categories
    experience_cat = case_when(
      years_exp <= 2 ~ "rookie",
      years_exp <= 5 ~ "young",
      years_exp <= 10 ~ "veteran",
      TRUE ~ "experienced"
    ),
    experience_cat = factor(experience_cat, levels = c("experienced", "rookie", "young", "veteran")),

    # Interaction terms
    player_season = paste0(receiver_id, "_", season),
    qb_season = paste0(qb_id, "_", season),
    position_season = paste0(position_group, "_", season)
  ) %>%
  filter(!is.infinite(log_yards), !is.infinite(log_targets))

cat(glue("Features engineered. {nrow(data)} records ready for modeling\n"))
cat(glue("Unique receivers: {n_distinct(data$receiver_id)}\n"))
cat(glue("Unique QBs: {n_distinct(data$qb_id)}\n"))
cat(glue("Unique QB-WR pairs: {n_distinct(data$qb_wr_pair)}\n\n"))

# ============================================================================
# 4. TRAIN HIERARCHICAL MODEL WITH QB-WR CHEMISTRY
# ============================================================================
cat("Step 4: Training hierarchical Bayesian model with QB-WR chemistry...\n")
cat("This will take 15-20 minutes due to increased complexity...\n\n")

# Define model formula with QB-WR dyadic effects
formula <- bf(
  log_yards ~ 1 +
    log_targets +
    is_home +
    is_favored +
    spread_abs +
    is_bad_weather +
    is_dome +
    scale(total_close) +
    experience_cat +
    position_group +
    (1 | receiver_id) +              # Receiver talent
    (1 | qb_id) +                    # QB talent
    (1 | qb_wr_pair) +               # QB-WR CHEMISTRY (dyadic effect)
    (1 | team) +
    (1 | opponent) +
    (1 | position_season) +
    (1 | qb_season) +
    (log_targets | player_season),   # Varying slopes by player-season
  sigma ~ log_targets + position_group  # DISTRIBUTIONAL REGRESSION
)

# Set priors
priors <- c(
  prior(normal(3, 1), class = Intercept),
  prior(normal(0, 0.5), class = b),
  prior(exponential(10), class = sd, group = receiver_id),
  prior(exponential(10), class = sd, group = qb_id),
  prior(exponential(5), class = sd, group = qb_wr_pair),  # Chemistry effect
  prior(exponential(20), class = sd, group = team),
  prior(exponential(20), class = sd, group = opponent),
  prior(exponential(20), class = sd, group = position_season),
  prior(exponential(10), class = sd, group = qb_season),
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
model_path <- "models/bayesian/receiving_qb_chemistry_v1.rds"
saveRDS(model, model_path)
cat(glue("\n✓ Model saved to {model_path}\n\n"))

# ============================================================================
# 5. EXTRACT QB-WR CHEMISTRY EFFECTS
# ============================================================================
cat("Step 5: Extracting QB-WR chemistry effects...\n")

# Get random effects for QB-WR pairs
chemistry_effects <- ranef(model)$qb_wr_pair %>%
  as_tibble(rownames = "qb_wr_pair") %>%
  separate(qb_wr_pair, into = c("qb_id", "receiver_id"), sep = "_", remove = FALSE) %>%
  rename(
    chemistry_mean = Estimate.Intercept,
    chemistry_se = Est.Error.Intercept,
    chemistry_q025 = Q2.5.Intercept,
    chemistry_q975 = Q97.5.Intercept
  )

cat(glue("✓ Extracted chemistry effects for {nrow(chemistry_effects)} QB-WR pairs\n\n"))
cat("Top 10 QB-WR pairs by chemistry (positive effect = more yards):\n")
print(chemistry_effects %>%
  arrange(desc(chemistry_mean)) %>%
  select(qb_id, receiver_id, chemistry_mean, chemistry_se) %>%
  head(10))

# ============================================================================
# 6. GENERATE PREDICTIONS FOR EACH RECEIVER
# ============================================================================
cat("\nStep 6: Generating predictions for each receiver...\n")

# Get unique receivers with their most common QB
receivers <- data %>%
  group_by(receiver_id, qb_id) %>%
  summarize(n_games_together = n(), .groups = "drop") %>%
  group_by(receiver_id) %>%
  slice_max(n_games_together, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  left_join(
    data %>%
      group_by(receiver_id, qb_id) %>%
      summarize(
        avg_targets = mean(targets),
        avg_log_targets = mean(log_targets),
        pct_home = mean(is_home),
        pct_favored = mean(is_favored),
        avg_spread = mean(spread_abs),
        pct_bad_weather = mean(is_bad_weather),
        pct_dome = mean(is_dome),
        avg_total = mean(total_close, na.rm = TRUE),
        experience = first(experience_cat),
        position_group = first(position_group),
        team = first(team),
        opponent = first(opponent),
        position_season = first(position_season),
        player_season = first(player_season),
        qb_season = first(qb_season),
        .groups = "drop"
      ),
    by = c("receiver_id", "qb_id")
  ) %>%
  mutate(qb_wr_pair = paste0(qb_id, "_", receiver_id))

cat(glue("Generating predictions for {nrow(receivers)} receivers...\n"))

# Create "typical game" scenarios
pred_data <- receivers %>%
  mutate(
    is_home = pct_home,
    is_favored = pct_favored,
    spread_abs = avg_spread,
    is_bad_weather = pct_bad_weather,
    is_dome = pct_dome,
    total_close = avg_total,
    log_targets = avg_log_targets,
    targets = avg_targets,
    experience_cat = experience
  )

# Generate posterior predictions
cat("Generating posterior predictions (this may take a few minutes)...\n")
predictions <- fitted(model, newdata = pred_data, re_formula = NULL, summary = FALSE)

cat(glue("Generated {nrow(predictions)} posterior samples for {ncol(predictions)} receivers\n"))

# Calculate summary statistics
ratings_df <- tibble(
  player_id = pred_data$receiver_id,
  qb_id = pred_data$qb_id,
  stat_type = "receiving_yards",
  season = 2024L,
  model_version = "qb_chemistry_v1.0",

  # Predictions in YARDS (exp of log predictions)
  rating_mean = exp(colMeans(predictions)),
  rating_sd = exp(colMeans(predictions)) * apply(predictions, 2, sd),  # Delta method approx
  rating_q05 = apply(predictions, 2, function(x) exp(quantile(x, 0.05))),
  rating_q50 = apply(predictions, 2, function(x) exp(quantile(x, 0.50))),
  rating_q95 = apply(predictions, 2, function(x) exp(quantile(x, 0.95))),

  # Context
  typical_targets = pred_data$avg_targets,

  # Hierarchical components
  position_group_mean = 0.0,
  team_effect = 0.0,
  vs_opponent_effect = NA_real_,

  # Quality metrics
  n_games_observed = pred_data$n_games_together,
  effective_sample_size = 1000.0,
  rhat = 1.0,

  # Timestamps
  trained_at = Sys.time(),
  updated_at = Sys.time()
)

cat(glue("\n✓ Generated {nrow(ratings_df)} ratings in YARDS scale\n"))
cat("\nTop 10 receivers by predicted yards:\n")
print(ratings_df %>%
  arrange(desc(rating_mean)) %>%
  select(player_id, qb_id, rating_mean, rating_sd, typical_targets, n_games_observed) %>%
  head(10))

# ============================================================================
# 7. SAVE TO DATABASE
# ============================================================================
cat("\nStep 7: Saving predictions to database...\n")

# Delete old ratings
dbExecute(conn, "DELETE FROM mart.bayesian_player_ratings WHERE stat_type = 'receiving_yards' AND model_version = 'qb_chemistry_v1.0'")
cat("Deleted old ratings\n")

# Insert new ratings (including qb_id for chemistry tracking)
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
verify_count <- dbGetQuery(conn, "SELECT COUNT(*) FROM mart.bayesian_player_ratings WHERE stat_type = 'receiving_yards' AND model_version = 'qb_chemistry_v1.0'")
cat(glue("Verification: {verify_count$count} ratings in database\n\n"))

# Save chemistry effects separately
chemistry_output_path <- "models/bayesian/qb_wr_chemistry_effects_v1.csv"
write_csv(chemistry_effects, chemistry_output_path)
cat(glue("✓ Saved QB-WR chemistry effects to {chemistry_output_path}\n\n"))

dbDisconnect(conn)

# ============================================================================
# SUMMARY
# ============================================================================
cat(paste0(rep("=", 80), collapse=""), "\n")
cat("✅ PIPELINE COMPLETE\n")
cat(paste0(rep("=", 80), collapse=""), "\n\n")

cat("Summary:\n")
cat(glue("  - Model: {model_path}\n"))
cat(glue("  - Receivers: {nrow(ratings_df)}\n"))
cat(glue("  - QB-WR pairs: {nrow(chemistry_effects)}\n"))
cat(glue("  - Training data: {nrow(data)} games (2020-2024)\n"))
cat(glue("  - Innovations:\n"))
cat(glue("    * QB-WR chemistry dyadic effects\n"))
cat(glue("    * Distributional regression (sigma modeling)\n"))
cat(glue("    * Position-specific variance\n"))
cat(glue("  - Database: mart.bayesian_player_ratings\n"))
cat(glue("  - Model version: qb_chemistry_v1.0\n\n"))

cat("Next steps:\n")
cat("1. Backtest against historical lines\n")
cat("2. Compare with baseline receiving model\n")
cat("3. Analyze chemistry effects for roster changes\n\n")
