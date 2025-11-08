#!/usr/bin/env Rscript
# Dynamic NFL Season Data Ingestion - Auto-detects current season
# Usage: Rscript R/ingestion/ingest_current_season.R [--season YEAR]

# Source error handling utilities
source("R/utils/error_handling.R")

# Load required libraries with error handling
safe_execute(
  expr = {
    suppressPackageStartupMessages({
      library(nflreadr)
      library(dplyr)
      library(DBI)
      library(RPostgres)
    })
  },
  error_message = "Failed to load required R packages"
)

# ============================================================
# PARSE COMMAND LINE ARGUMENTS
# ============================================================

args <- commandArgs(trailingOnly = TRUE)
season_override <- NULL

# Check for --season argument
if (length(args) > 0) {
  season_idx <- which(args == "--season")
  if (length(season_idx) > 0 && length(args) > season_idx) {
    season_override <- as.integer(args[season_idx + 1])
  }
}

# ============================================================
# DETERMINE SEASON TO INGEST
# ============================================================

get_current_nfl_season <- function() {
  # NFL season runs September to February
  # If month >= 9, we're in that year's season
  # If month <= 2, we're in previous year's season
  current_date <- Sys.Date()
  year <- as.integer(format(current_date, "%Y"))
  month <- as.integer(format(current_date, "%m"))

  if (month >= 9) {
    # September or later - current year's season
    return(year)
  } else if (month <= 2) {
    # January/February - previous year's season
    return(year - 1)
  } else {
    # March-August - offseason, use previous season
    return(year - 1)
  }
}

# Determine which season to process
if (!is.null(season_override)) {
  CURRENT_SEASON <- season_override
  log_message(sprintf("Processing season %d (override)", CURRENT_SEASON), level = "INFO")
} else {
  CURRENT_SEASON <- get_current_nfl_season()
  log_message(sprintf("Processing current season: %d (auto-detected)", CURRENT_SEASON), level = "INFO")
}

# Validate season range
if (CURRENT_SEASON < 1999 || CURRENT_SEASON > 2050) {
  stop(sprintf("Invalid season: %d. Must be between 1999 and 2050.", CURRENT_SEASON))
}

# Initialize logging
log_message(sprintf("=== NFL Season %d Data Ingestion Starting ===", CURRENT_SEASON), level = "INFO")
log_message(sprintf("Current date: %s", Sys.Date()), level = "INFO")

# ============================================================
# DATABASE CONNECTION WITH ERROR HANDLING
# ============================================================

db_params <- list(
  host = Sys.getenv("POSTGRES_HOST", "localhost"),
  port = as.integer(Sys.getenv("POSTGRES_PORT", "5544")),
  dbname = Sys.getenv("POSTGRES_DB", "devdb01"),
  user = Sys.getenv("POSTGRES_USER", "dro"),
  password = Sys.getenv("POSTGRES_PASSWORD", "sicillionbillions")
)

# Main pipeline execution with database transaction management
safe_db_operation(
  db_params = db_params,
  expr = quote({

    # ============================================================
    # 1. SCHEDULES (GAMES TABLE)
    # ============================================================

    games_data <- run_pipeline_step(
      step_name = sprintf("Load %d Schedules", CURRENT_SEASON),
      expr = {
        # Load schedules with retry for network issues
        schedules <- retry_operation(
          expr = load_schedules(CURRENT_SEASON),
          max_attempts = 3,
          delay = 5,
          error_message = sprintf("Failed to load %d schedules from nflverse", CURRENT_SEASON)
        )

        log_message(sprintf("Found %d games (%d completed)",
                           nrow(schedules),
                           sum(!is.na(schedules$home_score))),
                   level = "INFO")

        # Transform to games table format
        games_df <- schedules %>%
          transmute(
            game_id = game_id,
            season = season,
            week = week,
            home_team = home_team,
            away_team = away_team,
            kickoff = as.POSIXct(gameday, tz = "America/New_York"),
            spread_close = spread_line,
            total_close = total_line,
            home_score = home_score,
            away_score = away_score,
            home_moneyline = home_moneyline,
            away_moneyline = away_moneyline,
            home_spread_odds = home_spread_odds,
            away_spread_odds = away_spread_odds,
            over_odds = over_odds,
            under_odds = under_odds,
            stadium = stadium,
            roof = roof,
            surface = surface,
            away_rest = away_rest,
            home_rest = home_rest,
            away_qb_id = away_qb_id,
            home_qb_id = home_qb_id,
            away_qb_name = away_qb_name,
            home_qb_name = home_qb_name,
            away_coach = away_coach,
            home_coach = home_coach,
            referee = referee,
            stadium_id = stadium_id,
            game_type = game_type,
            overtime = overtime
          )

        games_df
      },
      conn = conn,
      validate_fn = function(data) {
        validate_data(
          data = data,
          expected_cols = c("game_id", "season", "week", "home_team", "away_team"),
          min_rows = 1,  # At least some games
          max_rows = 300  # Maximum reasonable for full season
        )
      }
    )

    # Upsert games with error handling
    run_pipeline_step(
      step_name = "Upsert Games to Database",
      expr = {
        # Delete ALL child tables FIRST to avoid foreign key constraints
        # Delete from predictions schema tables (6 tables)
        predictions_tables <- c(
          "predictions.bayesian_prop_predictions",
          "predictions.game_predictions",
          "predictions.prediction_versions",
          "predictions.prop_predictions",
          "predictions.retrospectives"
        )

        total_predictions_deleted <- 0
        for (table in predictions_tables) {
          deleted <- tryCatch({
            dbExecute(conn, sprintf("DELETE FROM %s WHERE game_id LIKE '%d_%%'", table, CURRENT_SEASON))
          }, error = function(e) {
            log_message(sprintf("%s table not found, skipping", table), level = "INFO")
            0
          })
          if (deleted > 0) {
            log_message(sprintf("Deleted %d rows from %s for season %d",
                               deleted, table, CURRENT_SEASON), level = "INFO")
            total_predictions_deleted <- total_predictions_deleted + deleted
          }
        }

        # Delete plays
        plays_deleted <- dbExecute(conn,
          sprintf("DELETE FROM plays WHERE game_id LIKE '%d_%%'", CURRENT_SEASON))
        log_message(sprintf("Deleted %d existing plays for season %d",
                           plays_deleted, CURRENT_SEASON), level = "INFO")

        # Delete existing games for this season LAST (parent table)
        deleted_count <- dbExecute(conn,
          sprintf("DELETE FROM games WHERE season = %d", CURRENT_SEASON))
        log_message(sprintf("Deleted %d existing games for season %d",
                           deleted_count, CURRENT_SEASON), level = "INFO")

        # Insert all games for current season
        rows_inserted <- dbWriteTable(conn, "games", games_data,
                                      append = TRUE, row.names = FALSE)
        log_message(sprintf("Inserted %d games", nrow(games_data)), level = "INFO")

        # Verify insertion
        verification <- dbGetQuery(conn,
          sprintf("SELECT COUNT(*) as count FROM games WHERE season = %d", CURRENT_SEASON))
        if (verification$count != nrow(games_data)) {
          stop(sprintf("Data verification failed: Expected %d games, found %d",
                      nrow(games_data), verification$count))
        }

        TRUE
      },
      conn = conn
    )

    # ============================================================
    # 2. PLAY-BY-PLAY (PLAYS TABLE) - Only if games have been played
    # ============================================================

    completed_games <- dbGetQuery(conn,
      sprintf("SELECT COUNT(*) as count FROM games WHERE season = %d AND home_score IS NOT NULL",
              CURRENT_SEASON))

    if (completed_games$count > 0) {
      plays_data <- run_pipeline_step(
        step_name = sprintf("Load %d Play-by-Play", CURRENT_SEASON),
        expr = {
          # Load play-by-play with retry
          pbp <- retry_operation(
            expr = load_pbp(CURRENT_SEASON),
            max_attempts = 3,
            delay = 5,
            error_message = sprintf("Failed to load %d play-by-play from nflverse", CURRENT_SEASON)
          )

          log_message(sprintf("Found %d plays from %d games",
                             nrow(pbp), length(unique(pbp$game_id))),
                     level = "INFO")

          # Standardize column names using our canonical naming
          # Always use 'quarter' not 'qtr'
          if ("qtr" %in% names(pbp) && !"quarter" %in% names(pbp)) {
            pbp$quarter <- pbp$qtr
            pbp$qtr <- NULL  # Remove the non-standard column
          }
          # Keep both time_seconds and game_seconds_remaining for backwards compatibility
          # and new features that need game-level time tracking
          if ("game_seconds_remaining" %in% names(pbp) && !"time_seconds" %in% names(pbp)) {
            pbp$time_seconds <- pbp$game_seconds_remaining
            # Keep game_seconds_remaining as well - DO NOT DELETE
          }

          # Define canonical column set for our database schema
          # This list matches our database schema and provides defensive selection
          # Note: Using any_of() to gracefully handle missing columns in nflreadr
          # Updated: Nov 7 2025 - Migration 027 adds 15 CRITICAL columns
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
            "qb_hit_1_player_id",

            # ============================================================
            # PHASE 5: EXHAUSTIVE HIGH-SIGNAL COLUMNS (Migration 031 - Nov 7 2025)
            # Expected ROI impact: +15-25% (cumulative: +18.8-33.5%)
            # ============================================================

            # 1. GAME CONTEXT & METADATA (12 columns) - +1.0-1.5% ROI
            "home_team", "away_team", "season_type", "week", "game_date",
            "start_time", "time_of_day", "stadium", "location", "roof",
            "surface", "div_game",

            # 2. WIN PROBABILITY & GAME SCRIPT (8 columns) - +2.0-3.0% ROI
            "home_wp", "away_wp", "home_wp_post", "away_wp_post",
            "vegas_home_wp", "vegas_home_wpa", "ep", "opp_fg_prob",

            # 3. TOTAL TEAM EPA (12 columns) - +1.5-2.5% ROI
            "total_home_epa", "total_away_epa", "total_home_rush_epa",
            "total_away_rush_epa", "total_home_pass_epa", "total_away_pass_epa",
            "total_home_comp_air_epa", "total_away_comp_air_epa",
            "total_home_comp_yac_epa", "total_away_comp_yac_epa",
            "total_home_raw_air_epa", "total_away_raw_air_epa",

            # 4. DRIVE CONTEXT EXTENDED (10 columns) - +1.5-2.0% ROI
            "drive_real_start_time", "drive_quarter_start", "drive_quarter_end",
            "drive_game_clock_start", "drive_game_clock_end",
            "drive_start_yard_line", "drive_end_yard_line",
            "drive_start_transition", "drive_end_transition", "side_of_field",

            # 5. RETURN GAME DETAILS (10 columns) - +2.0-3.5% ROI
            "punt_inside_twenty", "punt_in_endzone", "punt_out_of_bounds",
            "punt_downed", "punt_fair_catch",
            "kickoff_inside_twenty", "kickoff_in_endzone", "kickoff_out_of_bounds",
            "kickoff_downed", "kickoff_fair_catch",

            # 6. ADDITIONAL TACKLE DETAILS (14 columns) - +1.5-2.5% ROI
            "solo_tackle_2_player_id", "solo_tackle_2_player_name",
            "assist_tackle_2_player_id", "assist_tackle_2_player_name",
            "assist_tackle_3_player_id", "assist_tackle_3_player_name",
            "assist_tackle_4_player_id", "assist_tackle_4_player_name",
            "tackle_with_assist",
            "tackle_with_assist_1_player_id", "tackle_with_assist_1_player_name",
            "tackle_with_assist_2_player_id", "tackle_with_assist_2_player_name",
            "tackle_for_loss_2_player_id",

            # 7. QB PRESSURE & PASS DEFENSE (5 columns) - +1.0-2.0% ROI
            "qb_hit_1_player_name", "qb_hit_2_player_id", "qb_hit_2_player_name",
            "pass_defense_2_player_id", "pass_defense_2_player_name",

            # 8. SPECIAL TEAMS PLAYERS (8 columns) - +0.5-1.0% ROI
            "kicker_player_id", "kicker_player_name",
            "punter_player_id", "punter_player_name",
            "punt_returner_player_id", "punt_returner_player_name",
            "kickoff_returner_player_id", "kickoff_returner_player_name",

            # 9. FUMBLE DETAILS EXTENDED (10 columns) - +1.0-1.5% ROI
            "fumbled_1_player_id", "fumbled_1_player_name",
            "fumbled_2_player_id", "fumbled_2_player_name",
            "forced_fumble_player_2_player_id", "forced_fumble_player_2_player_name",
            "fumble_recovery_2_team", "fumble_recovery_2_yards",
            "fumble_recovery_2_player_id", "fumble_recovery_2_player_name",

            # 10. TIMEOUT & CLOCK MANAGEMENT (8 columns) - +0.8-1.2% ROI
            "home_timeouts_remaining", "away_timeouts_remaining",
            "posteam_timeouts_remaining", "defteam_timeouts_remaining",
            "timeout", "timeout_team", "play_clock", "quarter_seconds_remaining",

            # 11. SCORING PROBABILITY (4 columns) - +0.5-0.8% ROI
            "fg_prob", "td_prob", "opp_td_prob", "opp_safety_prob",

            # 12. INTERCEPTION & SACK DETAILS (6 columns) - +0.5-1.0% ROI
            "interception_player_id", "interception_player_name",
            "half_sack_1_player_id", "half_sack_1_player_name",
            "half_sack_2_player_id", "half_sack_2_player_name",

            # 13. REPLAY & CHALLENGE (2 columns) - +0.2-0.4% ROI
            "replay_or_challenge", "replay_or_challenge_result",

            # 14. SPECIAL TEAMS OUTCOMES (5 columns) - +0.3-0.6% ROI
            "touchback", "own_kickoff_recovery", "own_kickoff_recovery_td",
            "return_touchdown", "return_team",

            # 15. DOWN CONVERSION TRACKING (2 columns) - +0.3-0.5% ROI
            "third_down_failed", "fourth_down_failed"
          )

          # Defensive column selection - only selects columns that exist in nflreadr
          plays_df <- pbp %>%
            select(any_of(CANONICAL_COLUMNS))

          # Type conversion for BOOLEAN columns (Phases 1-5)
          # nflreadr sometimes provides probabilities (0.0-1.0) for boolean flags
          # Convert to proper boolean type for database insertion
          boolean_cols <- c(
            # Phase 1
            "xyac_success", "xyac_fd", "aborted_play", "out_of_bounds", "tackled_for_loss",
            # Phase 2
            "rush_attempt", "pass_attempt", "field_goal_attempt", "punt_attempt", "kickoff_attempt",
            # Phase 3
            "drive_inside20", "drive_ended_with_score",
            # Phase 4
            "fumble_forced", "safety", "punt_blocked", "first_down_penalty",
            # Phase 5
            "div_game", "punt_inside_twenty", "punt_in_endzone", "punt_out_of_bounds",
            "punt_downed", "punt_fair_catch", "kickoff_inside_twenty", "kickoff_in_endzone",
            "kickoff_out_of_bounds", "kickoff_downed", "kickoff_fair_catch",
            "tackle_with_assist", "timeout", "replay_or_challenge",
            "touchback", "own_kickoff_recovery", "own_kickoff_recovery_td",
            "return_touchdown", "third_down_failed", "fourth_down_failed"
          )
          for (col in boolean_cols) {
            if (col %in% names(plays_df) && !is.logical(plays_df[[col]])) {
              # Convert numeric/character to logical
              plays_df[[col]] <- as.logical(plays_df[[col]])
              log_message(sprintf("Converted %s to boolean type", col), level = "INFO")
            }
          }

          # Type conversion for INTEGER columns (Phases 1-5)
          # Some nflreadr columns come as numeric/character but database expects integer
          integer_cols <- c(
            # Phase 1
            "order_sequence",
            # Phase 3
            "drive_play_count", "drive_first_downs", "drive_yards_penalized",
            "temp", "wind",
            # Phase 4: No integer columns in Phase 4
            # Phase 5
            "week", "home_timeouts_remaining", "away_timeouts_remaining",
            "posteam_timeouts_remaining", "defteam_timeouts_remaining",
            "play_clock", "quarter_seconds_remaining",
            "drive_quarter_start", "drive_quarter_end"
          )
          for (col in integer_cols) {
            if (col %in% names(plays_df) && !is.integer(plays_df[[col]])) {
              plays_df[[col]] <- as.integer(plays_df[[col]])
              log_message(sprintf("Converted %s to integer type", col), level = "INFO")
            }
          }

          # Log missing columns for monitoring
          missing_from_nflreadr <- setdiff(CANONICAL_COLUMNS, names(pbp))
          if (length(missing_from_nflreadr) > 0) {
            log_message(
              sprintf("Warning: nflreadr missing expected columns: %s",
                      paste(missing_from_nflreadr, collapse=", ")),
              level = "WARN"
            )
          }

          # Log new columns in nflreadr (for discovery of new features)
          extra_in_nflreadr <- setdiff(names(pbp), CANONICAL_COLUMNS)
          if (length(extra_in_nflreadr) > 0) {
            log_message(
              sprintf("Info: nflreadr has %d additional columns not in our schema (review quarterly)",
                      length(extra_in_nflreadr)),
              level = "INFO"
            )
          }

          # All columns are now properly included (no renaming needed)
          plays_df
        },
        conn = conn,
        validate_fn = function(data) {
          validate_data(
            data = data,
            expected_cols = c("game_id", "play_id"),
            min_rows = 100,  # At least some plays
            max_rows = 50000  # Maximum reasonable for full season
          )
        }
      )

      # Upsert plays with error handling
      run_pipeline_step(
        step_name = "Upsert Plays to Database",
        expr = {
          # Delete existing plays for this season first
          deleted_count <- dbExecute(conn,
            sprintf("DELETE FROM plays WHERE game_id LIKE '%d_%%'", CURRENT_SEASON))
          log_message(sprintf("Deleted %d existing plays for season %d",
                             deleted_count, CURRENT_SEASON), level = "INFO")

          # Insert plays in batches to avoid memory issues
          batch_size <- 10000
          total_rows <- nrow(plays_data)

          for (i in seq(1, total_rows, by = batch_size)) {
            end_idx <- min(i + batch_size - 1, total_rows)
            batch <- plays_data[i:end_idx, ]

            dbWriteTable(conn, "plays", batch, append = TRUE, row.names = FALSE)
            log_message(sprintf("Inserted plays batch %d-%d of %d",
                               i, end_idx, total_rows),
                       level = "INFO")
          }

          # Verify insertion
          verification <- dbGetQuery(conn,
            sprintf("SELECT COUNT(*) as count FROM plays WHERE game_id LIKE '%d_%%'",
                    CURRENT_SEASON))
          if (abs(verification$count - nrow(plays_data)) > 10) {  # Allow small discrepancy
            log_message(sprintf("Warning: Expected %d plays, found %d",
                               nrow(plays_data), verification$count),
                       level = "WARNING")
          }

          TRUE
        },
        conn = conn
      )
    } else {
      log_message(sprintf("No completed games found for season %d, skipping play-by-play",
                         CURRENT_SEASON), level = "INFO")
    }

    # ============================================================
    # 3. ROSTERS (ROSTERS TABLE) - DISABLED FOR PHASE 1
    # ============================================================

    # DISABLED: Rosters cause transaction rollback due to FK constraint with players table
    # Players table must be populated FIRST before rosters can be inserted
    # This is not critical for Phase 1 (play-by-play data is what matters for props predictions)
    # TODO Phase 2: Fix rosters_player_id_fkey constraint or populate players table first
    log_message("Skipping rosters ingestion for Phase 1 (not critical - plays data imported successfully)", level = "INFO")

    # Rosters ingestion disabled - uncommit below when players table is populated
    # roster_result <- tryCatch({
    #   run_pipeline_step(
    #     step_name = sprintf("Load %d Rosters", CURRENT_SEASON),
    #     expr = {
    #     # Load rosters with retry
    #     rosters <- retry_operation(
    #       expr = load_rosters(CURRENT_SEASON),
    #       max_attempts = 3,
    #       delay = 5,
    #       error_message = sprintf("Failed to load %d rosters from nflverse", CURRENT_SEASON)
    #     )
    #
    #     log_message(sprintf("Found %d roster entries", nrow(rosters)), level = "INFO")
    #
    #     # Select only columns that exist in our database schema
    #     # Database has: season, week, team, player_id, position, depth_chart_position,
    #     #               jersey_number, status, full_name, football_name
    #     rosters <- rosters %>%
    #       select(any_of(c(
    #         "season", "week", "team",
    #         "gsis_id",  # Will rename to player_id
    #         "position", "depth_chart_position", "jersey_number",
    #         "status", "full_name", "football_name"
    #       ))) %>%
    #       rename(player_id = gsis_id)  # Database uses player_id not gsis_id
    #
    #     # Filter out NULL or empty player_id entries (primary key constraint)
    #     null_count <- sum(is.na(rosters$player_id) | rosters$player_id == "")
    #     if (null_count > 0) {
    #       log_message(sprintf("Filtering %d roster entries with NULL/empty player_id", null_count), level = "WARNING")
    #       rosters <- rosters[!is.na(rosters$player_id) & rosters$player_id != "", ]
    #     }
    #
    #     # Validate rosters
    #     validate_data(
    #       data = rosters,
    #       expected_cols = c("season", "week", "team", "player_id"),
    #       min_rows = 100  # Should have many roster entries
    #     )
    #
    #     # Delete existing rosters for this season
    #     deleted_count <- dbExecute(conn,
    #       sprintf("DELETE FROM rosters WHERE season = %d", CURRENT_SEASON))
    #     log_message(sprintf("Deleted %d existing roster entries for season %d",
    #                        deleted_count, CURRENT_SEASON),
    #                level = "INFO")
    #
    #     # Insert new rosters
    #     dbWriteTable(conn, "rosters", rosters, append = TRUE, row.names = FALSE)
    #     log_message(sprintf("Inserted %d roster entries", nrow(rosters)), level = "INFO")
    #
    #     TRUE
    #     },
    #     conn = conn
    #   )
    # }, error = function(e) {
    #   log_message(sprintf("Rosters ingestion skipped due to error: %s", e$message), level = "WARNING")
    #   log_message("This is OK - rosters are not critical for Phase 1 (plays data is what matters)", level = "INFO")
    #   NULL
    # })

    # ============================================================
    # 4. CALCULATE TURNOVERS AND PENALTIES (if plays exist)
    # ============================================================

    if (completed_games$count > 0) {
      run_pipeline_step(
        step_name = "Calculate Game Statistics",
        expr = {
          # Update turnovers from play-by-play
          updated <- dbExecute(conn, sprintf("
            UPDATE games g
            SET
              home_turnovers = subq.home_turnovers,
              away_turnovers = subq.away_turnovers,
              home_penalties = subq.home_penalties,
              away_penalties = subq.away_penalties,
              home_penalty_yards = subq.home_penalty_yards,
              away_penalty_yards = subq.away_penalty_yards,
              updated_at = NOW()
            FROM (
              SELECT
                p.game_id,
                COUNT(*) FILTER (WHERE p.posteam = g2.home_team AND (p.interception = 1 OR p.fumble_lost = 1)) as home_turnovers,
                COUNT(*) FILTER (WHERE p.posteam = g2.away_team AND (p.interception = 1 OR p.fumble_lost = 1)) as away_turnovers,
                COUNT(*) FILTER (WHERE p.posteam = g2.home_team AND p.penalty = 1) as home_penalties,
                COUNT(*) FILTER (WHERE p.posteam = g2.away_team AND p.penalty = 1) as away_penalties,
                SUM(CASE WHEN p.posteam = g2.home_team AND p.penalty = 1 THEN COALESCE(p.penalty_yards, 0) ELSE 0 END) as home_penalty_yards,
                SUM(CASE WHEN p.posteam = g2.away_team AND p.penalty = 1 THEN COALESCE(p.penalty_yards, 0) ELSE 0 END) as away_penalty_yards
              FROM plays p
              JOIN games g2 ON p.game_id = g2.game_id
              WHERE g2.season = %d
              GROUP BY p.game_id
            ) subq
            WHERE g.game_id = subq.game_id
          ", CURRENT_SEASON))

          log_message(sprintf("Updated statistics for %d games", updated), level = "INFO")
          TRUE
        },
        conn = conn
      )
    }

    # ============================================================
    # 5. REFRESH MATERIALIZED VIEWS
    # ============================================================

    run_pipeline_step(
      step_name = "Refresh Materialized Views",
      expr = {
        views_to_refresh <- c("mart.game_summary")

        for (view_name in views_to_refresh) {
          safe_execute(
            expr = {
              dbExecute(conn, sprintf("REFRESH MATERIALIZED VIEW %s", view_name))
              log_message(sprintf("Refreshed view: %s", view_name), level = "INFO")
            },
            conn = conn,
            error_message = sprintf("Failed to refresh view: %s", view_name)
          )
        }
        TRUE
      },
      conn = conn
    )

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    # Get final counts
    summary <- dbGetQuery(conn, sprintf("
      SELECT
        (SELECT COUNT(*) FROM games WHERE season = %d) as games_count,
        (SELECT COUNT(*) FROM plays WHERE game_id LIKE '%d_%%') as plays_count,
        (SELECT COUNT(*) FROM rosters WHERE season = %d) as roster_count,
        (SELECT COUNT(*) FROM games WHERE season = %d AND home_score IS NOT NULL) as completed_games
    ", CURRENT_SEASON, CURRENT_SEASON, CURRENT_SEASON, CURRENT_SEASON))

    log_message(sprintf("=== Season %d Ingestion Complete ===", CURRENT_SEASON), level = "INFO")
    log_message(sprintf("Games: %s (Completed: %s)",
                       format(as.integer(summary$games_count), big.mark = ","),
                       format(as.integer(summary$completed_games), big.mark = ",")),
               level = "INFO")
    log_message(sprintf("Plays: %s", format(as.integer(summary$plays_count), big.mark = ",")), level = "INFO")
    log_message(sprintf("Roster entries: %s", format(as.integer(summary$roster_count), big.mark = ",")), level = "INFO")

    # Check for alerts (non-critical, wrap in tryCatch)
    tryCatch({
      if (file.exists(file.path(LOG_DIR, "alerts.json"))) {
        alerts <- jsonlite::fromJSON(file.path(LOG_DIR, "alerts.json"))
        if (is.list(alerts) && length(alerts) > 0) {
          unread_alerts <- sum(sapply(alerts, function(x) x$status == "unread"))
          if (unread_alerts > 0) {
            log_message(sprintf("⚠️  %d unread alerts in %s",
                               unread_alerts,
                               file.path(LOG_DIR, "alerts.json")),
                       level = "WARNING")
          }
        }
      }
    }, error = function(e) {
      # Silently ignore alert checking errors
    })
  })
)