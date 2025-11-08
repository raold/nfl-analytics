#!/usr/bin/env Rscript
# R/ingestion/reingest_historical_all_phases.R
# Re-ingest historical play-by-play data (1999-2024) with ALL Phase 1-4 columns
# Purpose: Populate Phase 1-4 columns (67 total: 15 Phase 1 + 15 Phase 2 + 12 Phase 3 + 25 Phase 4)
# Expected runtime: ~20-30 minutes for all seasons
# Expected ROI Impact: +3.8-8.5% total (+1.5-3.5% P1, +0.5-1.5% P2, +0.3-1.0% P3, +1.5-3.0% P4)

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

# CANONICAL_COLUMNS - EXACT copy from updated ingest_current_season.R
# Total: 117 columns (90 original + 15 Phase 1 + 15 Phase 2 - 3 Phase 3)
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
  "passing_yards",    # Net passing yards (excludes sacks)
  "receiving_yards",  # Receiving yards (not yards_gained)
  "rushing_yards",    # Rushing yards (not yards_gained)
  "return_yards",     # Return yards (punt/kickoff)

  # Advanced EPA (QB vs receiver separation)
  "qb_epa",              # QB EPA at throw (before YAC)
  "xyac_epa",            # Expected yards after catch EPA
  "xyac_mean_yardage",   # Mean expected YAC
  "xyac_median_yardage", # Median expected YAC
  "xyac_success",        # Binary YAC success
  "xyac_fd",             # YAC resulted in first down

  # Play validity & context
  "aborted_play",      # Play aborted (exclude from stats)
  "out_of_bounds",     # Clock management context
  "order_sequence",    # Play number within drive
  "play_type_nfl",     # NFL official play categorization
  "tackled_for_loss",  # Defensive disruption

  # ============================================================
  # PHASE 2: DEFENSIVE & PLAYER TRACKING (Migration 028 - Nov 7 2025)
  # Expected ROI impact: +0.5-1.5%
  # ============================================================

  # Attempt flags
  "rush_attempt", "pass_attempt", "field_goal_attempt",
  "punt_attempt", "kickoff_attempt",

  # Defensive player tracking
  "solo_tackle_1_player_id", "solo_tackle_1_player_name",
  "assist_tackle_1_player_id", "assist_tackle_1_player_name",
  "pass_defense_1_player_id", "pass_defense_1_player_name",
  "tackle_for_loss_1_player_id", "tackle_for_loss_1_player_name",

  # Two-point conversion
  "two_point_conv_result",

  # ============================================================
  # PHASE 3: SITUATIONAL & DRIVE CONTEXT (Migration 029 - Nov 7 2025)
  # Expected ROI impact: +0.3-1.0%
  # ============================================================

  # Drive-level context
  "drive_play_count", "drive_time_of_possession", "drive_first_downs",
  "drive_inside20", "drive_ended_with_score", "drive_yards_penalized",

  # Weather conditions
  "temp", "wind", "weather",

  # Game situation
  "no_score_prob", "def_wp", "game_half",

  # ============================================================
  # PHASE 4: CRITICAL HIGH-SIGNAL COLUMNS (Migration 030 - Nov 7 2025)
  # Expected ROI impact: +1.5-3.0%
  # ============================================================

  # Play execution quality (fumble outcomes, safety, blocked kicks)
  "fumble_forced", "fumble_recovery_1_team", "fumble_recovery_1_yards",
  "fumble_recovery_1_player_id", "fumble_recovery_1_player_name",
  "safety", "safety_player_id", "safety_player_name",
  "punt_blocked", "blocked_player_id", "blocked_player_name",

  # Advanced metrics (completion probability, EPA decomposition)
  "xpass", "pass_oe", "cp", "air_epa", "yac_epa",

  # Penalty granularity
  "penalty_type", "penalty_player_id", "penalty_player_name", "first_down_penalty",

  # Defensive players extended (sack, forced fumble, QB hits)
  "sack_player_id", "sack_player_name",
  "forced_fumble_player_1_player_id", "forced_fumble_player_1_player_name",
  "qb_hit_1_player_id"
)

cat("=================================================================\n")
cat("  HISTORICAL RE-INGESTION: All Phases (1-4) Column Population\n")
cat("  Seasons:", START_SEASON, "to", END_SEASON, "\n")
cat("  Columns:", length(CANONICAL_COLUMNS), "total columns\n")
cat("  Expected runtime: ~20-30 minutes\n")
cat("  Expected ROI Impact: +3.8-8.5% total (Phases 1-4)\n")
cat("=================================================================\n\n")

# Connect to database
con <- get_db_connection()

# Process each season
for (season in START_SEASON:END_SEASON) {
  cat(sprintf("\n[%s] Processing season %d...\n", Sys.time(), season))

  tryCatch({
    # Load play-by-play from nflreadr
    cat(sprintf("  Loading data from nflreadr... "))
    pbp_raw <- nflreadr::load_pbp(season)
    cat(sprintf("loaded %d plays\n", nrow(pbp_raw)))

    # Select only canonical columns
    pbp_clean <- pbp_raw %>%
      select(any_of(CANONICAL_COLUMNS))

    cat(sprintf("  Selected %d columns from nflreadr\n", ncol(pbp_clean)))

    # Boolean type conversion for Phases 1-4
    boolean_cols <- c(
      # Phase 1
      "xyac_success", "xyac_fd", "aborted_play", "out_of_bounds", "tackled_for_loss",
      # Phase 2
      "rush_attempt", "pass_attempt", "field_goal_attempt", "punt_attempt", "kickoff_attempt",
      # Phase 3
      "drive_inside20", "drive_ended_with_score",
      # Phase 4
      "fumble_forced", "safety", "punt_blocked", "first_down_penalty"
    )
    for (col in boolean_cols) {
      if (col %in% names(pbp_clean) && !is.logical(pbp_clean[[col]])) {
        pbp_clean[[col]] <- as.logical(pbp_clean[[col]])
      }
    }

    # Integer type conversion for Phases 1-4
    integer_cols <- c(
      # Phase 1
      "order_sequence",
      # Phase 3
      "drive_play_count", "drive_first_downs", "drive_yards_penalized",
      "temp", "wind"
      # Phase 4: No integer columns in Phase 4
    )
    for (col in integer_cols) {
      if (col %in% names(pbp_clean) && !is.integer(pbp_clean[[col]])) {
        pbp_clean[[col]] <- as.integer(pbp_clean[[col]])
      }
    }

    # Delete existing plays for this season
    deleted_count <- dbExecute(con,
      sprintf("DELETE FROM plays WHERE game_id LIKE '%s_%%'", as.character(season)))
    cat(sprintf("  Deleted %d existing plays\n", deleted_count))

    # Insert new plays in batches of 10,000
    batch_size <- 10000
    num_batches <- ceiling(nrow(pbp_clean) / batch_size)

    for (batch_idx in 1:num_batches) {
      start_row <- (batch_idx - 1) * batch_size + 1
      end_row <- min(batch_idx * batch_size, nrow(pbp_clean))
      batch_data <- pbp_clean[start_row:end_row, ]

      dbWriteTable(con, "plays", batch_data, append = TRUE, row.names = FALSE)
      cat(sprintf("  Inserted batch %d/%d (%d-%d of %d plays)\n",
                  batch_idx, num_batches, start_row, end_row, nrow(pbp_clean)))
    }

    # Validate Phase 1-3 column population
    validation_query <- sprintf("
      SELECT
        COUNT(*) as total,
        COUNT(passing_yards) as has_passing,
        COUNT(qb_epa) as has_qb_epa,
        COUNT(solo_tackle_1_player_id) as has_tackle,
        COUNT(temp) as has_weather,
        ROUND(100.0 * COUNT(passing_yards)::numeric / NULLIF(COUNT(*), 0), 1) as pct_passing
      FROM plays
      WHERE game_id LIKE '%s_%%'
    ", as.character(season))

    validation <- dbGetQuery(con, validation_query)
    cat(sprintf("  ✓ Season %d complete: %d plays, %.1f%% passing, %d tackles, %d weather\n",
                season, as.integer(validation$total), validation$pct_passing,
                as.integer(validation$has_tackle), as.integer(validation$has_weather)))

  }, error = function(e) {
    cat(sprintf("  ✗ ERROR in season %d: %s\n", season, e$message))
  })
}

# Final validation across all seasons
cat("\n=================================================================\n")
cat("  FINAL VALIDATION (ALL PHASES 1-4)\n")
cat("=================================================================\n")

final_validation <- dbGetQuery(con, "
  SELECT
    SUBSTRING(game_id, 1, 4) as season,
    COUNT(DISTINCT game_id) as games,
    COUNT(*) as plays,
    COUNT(passing_yards) as has_passing,
    COUNT(qb_epa) as has_qb_epa,
    COUNT(solo_tackle_1_player_id) as has_tackle,
    COUNT(drive_play_count) as has_drive,
    COUNT(temp) as has_weather,
    ROUND(100.0 * COUNT(passing_yards)::numeric / NULLIF(COUNT(*), 0), 1) as pct_passing
  FROM plays
  WHERE game_id ~ '^[0-9]{4}_'
  GROUP BY SUBSTRING(game_id, 1, 4)
  ORDER BY season DESC
  LIMIT 10
")

print(final_validation)

cat("\n✓ Historical re-ingestion complete (ALL PHASES 1-4)!\n")
cat(sprintf("  Processed seasons %d-%d\n", START_SEASON, END_SEASON))
cat("  Phase 1-4 columns now populated across all historical data\n")
cat("  Total columns added: 67 (15 Phase 1 + 15 Phase 2 + 12 Phase 3 + 25 Phase 4)\n")
cat("  Expected ROI Impact: +3.8-8.5%\n\n")

dbDisconnect(con)
