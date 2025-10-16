#!/usr/bin/env Rscript
# Simplified script to train passing yards model and save ratings

library(brms)
library(tidyverse)
library(DBI)
library(RPostgres)
library(cmdstanr)
library(posterior)

options(mc.cores = parallel::detectCores() - 1)

# Database connection
conn <- dbConnect(
  Postgres(),
  host = "localhost",
  port = 5544,
  dbname = "devdb01",
  user = "dro",
  password = "sicillionbillions"
)

cat("Loading passing yards data...\n")

# Load data
query <- "
SELECT
  pgs.player_id,
  pgs.season,
  pgs.week,
  pgs.player_display_name,
  pgs.stat_attempts,
  pgs.stat_yards,
  ph.position,
  ph.position_group,
  ph.current_team as team,
  ph.years_exp,
  g.home_team,
  g.away_team,
  g.spread_close as spread_line,
  g.total_close as total_line,
  g.temp,
  g.wind,
  g.roof,
  CASE WHEN g.home_team = ph.current_team THEN g.away_team ELSE g.home_team END as opponent,
  CASE WHEN g.home_team = ph.current_team THEN 1 ELSE 0 END as is_home
FROM mart.player_game_stats pgs
JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
LEFT JOIN games g ON
  g.season = pgs.season
  AND g.week = pgs.week
  AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
WHERE
  pgs.stat_category = 'passing'
  AND pgs.season >= 2020
  AND pgs.stat_yards IS NOT NULL
  AND ph.position_group IS NOT NULL
ORDER BY pgs.season, pgs.week, pgs.player_id
"

data <- dbGetQuery(conn, query)

cat(glue::glue("Loaded {nrow(data)} passing records\n"))

# Feature engineering
data <- data %>%
  mutate(
    log_yards = log1p(stat_yards),
    log_attempts = log1p(stat_attempts),
    experience_cat = case_when(
      years_exp <= 1 ~ "rookie",
      years_exp <= 3 ~ "early_career",
      years_exp <= 7 ~ "prime",
      TRUE ~ "veteran"
    ),
    is_bad_weather = ifelse(!is.na(wind) & wind > 15, 1, 0),
    is_dome = ifelse(!is.na(roof) & roof %in% c("dome", "closed"), 1, 0),
    is_favored = ifelse(!is.na(spread_line) & spread_line < 0, 1, 0),
    spread_abs = abs(ifelse(is.na(spread_line), 0, spread_line)),
    player_season = paste0(player_id, "_", season),
    position_season = paste0(position_group, "_", season)
  )

cat("Building and fitting model...\n")

# Build model with correct priors
priors <- c(
  prior(normal(200, 50), class = Intercept),
  prior(normal(0, 20), class = b),
  prior(exponential(0.05), class = sd, group = player_id),
  prior(exponential(0.1), class = sd, group = team),
  prior(exponential(0.2), class = sd, group = position_season),
  prior(exponential(0.1), class = sd, group = opponent)
)

formula <- bf(
  log_yards ~ 1 +
    log_attempts +
    is_home +
    is_favored +
    spread_abs +
    is_bad_weather +
    is_dome +
    scale(total_line) +
    experience_cat +
    (1 | player_id) +
    (1 | team) +
    (1 | opponent) +
    (1 | position_season) +
    (log_attempts | player_season),
  sigma ~ log_attempts
)

model <- brm(
  formula = formula,
  data = data,
  prior = priors,
  family = gaussian(),
  chains = 4,
  iter = 2000,
  warmup = 1000,
  control = list(adapt_delta = 0.95, max_treedepth = 12),
  backend = "cmdstanr",
  threads = threading(2),
  seed = 42
)

cat("\n✓ Model training complete\n")

# Save model
dir.create("models/bayesian", recursive = TRUE, showWarnings = FALSE)
saveRDS(model, "models/bayesian/passing_yards_hierarchical_v1.rds")
cat("✓ Model saved\n")

# Extract ratings
cat("\nExtracting player ratings...\n")

posterior_samples <- as_draws_df(model)
player_cols <- grep("^r_player_id\\[", names(posterior_samples), value = TRUE)

cat(glue::glue("Found {length(player_cols)} player effects\n"))

ratings_list <- list()

for(player_col in player_cols) {
  player_id <- gsub("r_player_id\\[(.+),Intercept\\]", "\\1", player_col)

  if(player_id %in% unique(data$player_id)) {
    player_effect <- posterior_samples[[player_col]]

    ratings_list[[player_id]] <- tibble(
      player_id = player_id,
      stat_type = "passing_yards",
      season = 2024,
      model_version = "hierarchical_v1.0",
      rating_mean = mean(player_effect),
      rating_sd = sd(player_effect),
      rating_q05 = quantile(player_effect, 0.05),
      rating_q50 = quantile(player_effect, 0.50),
      rating_q95 = quantile(player_effect, 0.95),
      position_group_mean = 0.0,
      team_effect = 0.0,
      vs_opponent_effect = NA_real_,
      n_games_observed = sum(data$player_id == player_id),
      effective_sample_size = posterior::ess_bulk(player_effect),
      rhat = posterior::rhat(player_effect),
      trained_at = Sys.time(),
      updated_at = Sys.time()
    )
  }
}

ratings_df <- bind_rows(ratings_list)

cat(glue::glue("\nExtracted ratings for {nrow(ratings_df)} players\n"))

# Save to database
cat("\nSaving to database...\n")
dbExecute(conn, "DELETE FROM mart.bayesian_player_ratings WHERE stat_type = 'passing_yards' AND model_version = 'hierarchical_v1.0'")

dbWriteTable(
  conn,
  c("mart", "bayesian_player_ratings"),
  ratings_df,
  append = TRUE,
  row.names = FALSE
)

dbDisconnect(conn)

cat("\n✅ SUCCESS! Saved", nrow(ratings_df), "player ratings\n")
cat("\nTop 10 QBs:\n")
print(ratings_df %>% arrange(desc(rating_mean)) %>% select(player_id, rating_mean, rating_sd, n_games_observed) %>% head(10))
