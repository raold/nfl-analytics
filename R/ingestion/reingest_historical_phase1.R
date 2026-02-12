#!/usr/bin/env Rscript
# R/ingestion/reingest_historical_phase1.R
# Re-ingest historical play-by-play data (1999-2024) with Phase 1 columns
# Purpose: Populate new Phase 1 columns (passing_yards, qb_epa, etc.) in historical data
# Expected runtime: ~20-30 minutes for all seasons

suppressPackageStartupMessages({
  library(nflreadr)
  library(dplyr)
  library(DBI)
  library(RPostgres)
})

# Load utility functions
source("R/utils/db_helpers.R")

START_SEASON <- 1999
END_SEASON <- 2024

# CANONICAL_COLUMNS - EXACT copy from ingest_current_season.R lines 264-346
CANONICAL_COLUMNS <- c(
  # Core play identification
  "game_id", "play_id", "posteam", "defteam", "quarter",
  "time_seconds", "game_seconds_remaining", "half_seconds_remaining",
  "down", "ydstogo", "yardline_100",

  # Play type and outcome
  "play_type", "yards_gained", "first_down",
  "third_down_converted", "fourth_down_converted",

  # Play category flags
  "pass", "rush", "touchdown", "two_point_attempt",

  # Turnovers
  "fumble", "interception", "fumble_lost",

  # Other events
  "penalty", "penalty_yards", "penalty_team",
  "qb_spike", "qb_kneel",
  "extra_point_result", "field_goal_result",
  "kick_distance", "td_team",
  "first_down_pass", "first_down_rush",

  # Advanced metrics
  "epa", "wp", "wpa", "vegas_wp", "vegas_wpa", "success",

  # Passing metrics
  "air_yards", "yards_after_catch", "cpoe",
  "comp_air_epa", "comp_yac_epa",
  "complete_pass", "incomplete_pass",
  "pass_length", "pass_location",
  "qb_hit", "qb_scramble", "sack",

  # Running metrics
  "run_location", "run_gap",

  # Player IDs
  "passer_player_id", "passer_player_name",
  "rusher_player_id", "rusher_player_name",
  "receiver_player_id", "receiver_player_name",

  # Formation
  "shotgun", "no_huddle", "qb_dropback",

  # Score tracking
  "posteam_score", "defteam_score", "score_differential",
  "posteam_score_post", "defteam_score_post",

  # Drive tracking
  "drive", "fixed_drive", "fixed_drive_result",

  # Series tracking (added in migration 026)
  "series", "series_success", "series_result",

  # Goal-to-go flag
  "goal_to_go",

  # ============================================================
  # PHASE 1 CRITICAL COLUMNS (Migration 027 - Nov 7 2025)
  # Expected ROI impact: +1.5-3.5%
  # ============================================================

  # Props outcomes (direct betting targets, not yards_gained proxy)
  "passing_yards", # Net passing yards (excludes sacks)
  "receiving_yards", # Receiving yards (not yards_gained)
  "rushing_yards", # Rushing yards (not yards_gained)
  "return_yards", # Return yards (punt/kickoff)

  # Advanced EPA (QB vs receiver separation)
  "qb_epa", # QB EPA at throw (before YAC)
  "xyac_epa", # Expected yards after catch EPA
  "xyac_mean_yardage", # Mean expected YAC
  "xyac_median_yardage", # Median expected YAC
  "xyac_success", # Binary YAC success
  "xyac_fd", # YAC resulted in first down

  # Play validity & context
  "aborted_play", # Play aborted (exclude from stats)
  "out_of_bounds", # Clock management context
  "order_sequence", # Play number within drive
  "play_type_nfl", # NFL official play categorization
  "tackled_for_loss" # Defensive disruption
)

cat("=================================================================\n")
cat("  HISTORICAL RE-INGESTION: Phase 1 Column Population\n")
cat("  Seasons:", START_SEASON, "to", END_SEASON, "\n")
cat("  Columns:", length(CANONICAL_COLUMNS), "total columns\n")
cat("  Expected runtime: ~20-30 minutes\n")
cat("=================================================================\n\n")

# Connect to database
con <- get_db_connection()

# Process each season
for (season in START_SEASON:END_SEASON) {
  cat(sprintf("\n[%s] Processing season %d...\n", Sys.time(), season))

  tryCatch(
    {
      # Load play-by-play from nflreadr
      cat(sprintf("  Loading data from nflreadr... "))
      pbp_raw <- nflreadr::load_pbp(season)
      cat(sprintf("loaded %d plays\n", nrow(pbp_raw)))

      # Select only canonical columns
      pbp_clean <- pbp_raw %>%
        select(any_of(CANONICAL_COLUMNS))

      cat(sprintf("  Selected %d columns from nflreadr\n", ncol(pbp_clean)))

      # Boolean type conversion for Phase 1 columns
      boolean_cols <- c("xyac_success", "xyac_fd", "aborted_play", "out_of_bounds", "tackled_for_loss")
      for (col in boolean_cols) {
        if (col %in% names(pbp_clean) && !is.logical(pbp_clean[[col]])) {
          pbp_clean[[col]] <- as.logical(pbp_clean[[col]])
        }
      }

      # Integer type conversion (order_sequence comes as "1.0" from nflreadr)
      if ("order_sequence" %in% names(pbp_clean)) {
        pbp_clean$order_sequence <- as.integer(pbp_clean$order_sequence)
      }

      # Delete existing plays for this season
      deleted_count <- dbExecute(
        con,
        sprintf("DELETE FROM plays WHERE game_id LIKE '%s_%%'", as.character(season))
      )
      cat(sprintf("  Deleted %d existing plays\n", deleted_count))

      # Insert new plays in batches of 10,000
      batch_size <- 10000
      num_batches <- ceiling(nrow(pbp_clean) / batch_size)

      for (batch_idx in 1:num_batches) {
        start_row <- (batch_idx - 1) * batch_size + 1
        end_row <- min(batch_idx * batch_size, nrow(pbp_clean))
        batch_data <- pbp_clean[start_row:end_row, ]

        dbWriteTable(con, "plays", batch_data, append = TRUE, row.names = FALSE)
        cat(sprintf(
          "  Inserted batch %d/%d (%d-%d of %d plays)\n",
          batch_idx, num_batches, start_row, end_row, nrow(pbp_clean)
        ))
      }

      # Validate Phase 1 column population
      validation_query <- sprintf("
      SELECT
        COUNT(*) as total,
        COUNT(passing_yards) as has_passing,
        COUNT(qb_epa) as has_qb_epa,
        ROUND(100.0 * COUNT(passing_yards)::numeric / NULLIF(COUNT(*), 0), 1) as pct
      FROM plays
      WHERE game_id LIKE '%s_%%'
    ", as.character(season))

      validation <- dbGetQuery(con, validation_query)
      cat(sprintf(
        "  ✓ Season %d complete: %d plays, %.1f%% passing_yards, %.1f%% qb_epa\n",
        season, validation$total, validation$pct,
        100.0 * validation$has_qb_epa / validation$total
      ))
    },
    error = function(e) {
      cat(sprintf("  ✗ ERROR in season %d: %s\n", season, e$message))
    }
  )
}

# Final validation across all seasons
cat("\n=================================================================\n")
cat("  FINAL VALIDATION\n")
cat("=================================================================\n")

final_validation <- dbGetQuery(con, "
  SELECT
    SUBSTRING(game_id, 1, 4) as season,
    COUNT(DISTINCT game_id) as games,
    COUNT(*) as plays,
    COUNT(passing_yards) as has_passing,
    COUNT(rushing_yards) as has_rushing,
    COUNT(qb_epa) as has_qb_epa,
    ROUND(100.0 * COUNT(passing_yards)::numeric / NULLIF(COUNT(*), 0), 1) as pct_passing
  FROM plays
  WHERE game_id ~ '^[0-9]{4}_'
  GROUP BY SUBSTRING(game_id, 1, 4)
  ORDER BY season DESC
  LIMIT 10
")

print(final_validation)

cat("\n✓ Historical re-ingestion complete!\n")
cat(sprintf("  Processed seasons %d-%d\n", START_SEASON, END_SEASON))
cat("  Phase 1 columns now populated across all historical data\n\n")

dbDisconnect(con)
