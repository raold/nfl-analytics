#!/usr/bin/env Rscript
# Quick script to extract ratings from trained Bayesian models and save to database

library(brms)
library(tidyverse)
library(DBI)
library(RPostgres)

# Database connection
get_db_connection <- function() {
  dbConnect(
    Postgres(),
    host = "localhost",
    port = 5544,
    dbname = "devdb01",
    user = "dro",
    password = "sicillionbillions"
  )
}

# Load model
model_file <- "models/bayesian/passing_yards_hierarchical_v1.rds"

if(file.exists(model_file)) {
  cat("Loading model from", model_file, "\n")
  model <- readRDS(model_file)

  # Get model data
  model_data <- model$data

  cat("Extracting player ratings...\n")

  # Extract posterior samples for player effects
  posterior_samples <- as_draws_df(model)

  # Get all player-specific random effects
  player_cols <- grep("^r_player_id\\[", names(posterior_samples), value = TRUE)

  cat(glue::glue("Found {length(player_cols)} player effects\n"))

  # Extract ratings for each player
  ratings_list <- list()

  for(player_col in player_cols) {
    # Extract player ID from column name
    player_id <- gsub("r_player_id\\[(.+),Intercept\\]", "\\1", player_col)

    if(player_id %in% unique(model_data$player_id)) {
      # Get posterior distribution
      player_effect <- posterior_samples[[player_col]]

      # Get player metadata
      player_info <- model_data %>%
        filter(player_id == !!player_id) %>%
        slice(1)

      # Calculate statistics
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
        position_group_mean = 0.0,  # Will calculate separately
        team_effect = 0.0,  # Will calculate separately
        vs_opponent_effect = NA_real_,
        n_games_observed = sum(model_data$player_id == player_id),
        effective_sample_size = posterior::ess_bulk(player_effect),
        rhat = posterior::rhat(player_effect),
        trained_at = Sys.time(),
        updated_at = Sys.time()
      )
    }
  }

  # Combine all ratings
  ratings_df <- bind_rows(ratings_list)

  cat(glue::glue("\nExtracted ratings for {nrow(ratings_df)} players\n"))

  # Save to database
  conn <- get_db_connection()

  cat("\nDeleting existing ratings...\n")
  dbExecute(conn, "
    DELETE FROM mart.bayesian_player_ratings
    WHERE stat_type = 'passing_yards'
      AND model_version = 'hierarchical_v1.0'
  ")

  cat("Inserting new ratings...\n")
  dbWriteTable(
    conn,
    c("mart", "bayesian_player_ratings"),
    ratings_df,
    append = TRUE,
    row.names = FALSE
  )

  dbDisconnect(conn)

  cat("\n✅ Successfully saved", nrow(ratings_df), "player ratings to database\n")
  cat("\nTop 10 QBs by rating:\n")
  print(ratings_df %>%
    arrange(desc(rating_mean)) %>%
    select(player_id, rating_mean, rating_sd, n_games_observed) %>%
    head(10))

} else {
  cat("❌ Model file not found:", model_file, "\n")
}
