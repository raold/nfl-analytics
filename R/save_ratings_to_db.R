#!/usr/bin/env Rscript
# Quick script to save extracted ratings to database

library(DBI)
library(RPostgres)
library(tidyverse)

# Connect to database
conn <- dbConnect(
  Postgres(),
  host = "localhost",
  port = 5544,
  dbname = "devdb01",
  user = "dro",
  password = "sicillionbillions"
)

# Load the saved model to extract ratings
model <- readRDS("models/bayesian/passing_yards_hierarchical_v1.rds")
data <- model$data

cat("Extracting ratings from trained model...\n")

# Get posterior samples
library(posterior)
posterior_samples <- as_draws_df(model)

# Extract player effects
player_cols <- grep("^r_player_id\\[", names(posterior_samples), value = TRUE)
cat(glue::glue("Found {length(player_cols)} player effects\n"))

# Build ratings dataframe
ratings_list <- list()

for(player_col in player_cols) {
  player_id <- gsub("r_player_id\\[(.+),Intercept\\]", "\\1", player_col)

  if(player_id %in% unique(data$player_id)) {
    player_effect <- posterior_samples[[player_col]]

    ratings_list[[player_id]] <- tibble(
      player_id = player_id,
      stat_type = "passing_yards",
      season = 2024L,
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
cat(glue::glue("Extracted {nrow(ratings_df)} ratings\n\n"))

# Save using SQL INSERT instead of dbWriteTable with schema
cat("Deleting old ratings...\n")
dbExecute(conn, "DELETE FROM mart.bayesian_player_ratings WHERE stat_type = 'passing_yards' AND model_version = 'hierarchical_v1.0'")

cat("Inserting ratings using SQL...\n")

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
}

dbDisconnect(conn)

cat("\n✅ Successfully saved", nrow(ratings_df), "ratings to database!\n\n")
cat("Top 10 QBs by rating:\n")
print(ratings_df %>% arrange(desc(rating_mean)) %>% select(player_id, rating_mean, rating_sd, n_games_observed) %>% head(10))
