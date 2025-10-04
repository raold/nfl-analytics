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
        # Delete existing games for this season first
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
          # Always use 'time_seconds' not 'game_seconds_remaining'
          if ("game_seconds_remaining" %in% names(pbp) && !"time_seconds" %in% names(pbp)) {
            pbp$time_seconds <- pbp$game_seconds_remaining
            pbp$game_seconds_remaining <- NULL  # Remove the non-standard column
          }

          # Select and transform columns
          plays_df <- pbp %>%
            select(any_of(c(
              "game_id", "play_id", "posteam", "defteam", "quarter",
              "time_seconds", "game_seconds_remaining", "down", "ydstogo",
              "yardline_100", "play_type", "yards_gained", "first_down",
              "pass", "rush", "special", "touchdown", "pass_touchdown",
              "rush_touchdown", "return_touchdown", "extra_point_attempt",
              "two_point_attempt", "field_goal_attempt", "kickoff_attempt",
              "punt_attempt", "fumble", "interception", "fumble_lost",
              "timeout", "penalty", "penalty_yards", "qb_spike", "qb_kneel",
              "epa", "wp", "wpa", "vegas_wp", "vegas_wpa", "success",
              "air_yards", "yards_after_catch", "cpoe", "comp_air_epa",
              "comp_yac_epa", "complete_pass", "incomplete_pass",
              "pass_length", "pass_location", "qb_hit", "qb_scramble",
              "run_location", "run_gap", "passer_player_id", "passer_player_name",
              "rusher_player_id", "rusher_player_name", "receiver_player_id",
              "receiver_player_name", "sack", "shotgun", "no_huddle", "qb_dropback",
              "posteam_score", "defteam_score", "score_differential",
              "posteam_score_post", "defteam_score_post", "score_differential_post"
            )))

          # Rename columns if needed
          if ("game_seconds_remaining" %in% names(plays_df) && !"time_seconds" %in% names(plays_df)) {
            plays_df <- plays_df %>% rename(time_seconds = game_seconds_remaining)
          }

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
    # 3. ROSTERS (ROSTERS TABLE)
    # ============================================================

    run_pipeline_step(
      step_name = sprintf("Load %d Rosters", CURRENT_SEASON),
      expr = {
        # Load rosters with retry
        rosters <- retry_operation(
          expr = load_rosters(CURRENT_SEASON),
          max_attempts = 3,
          delay = 5,
          error_message = sprintf("Failed to load %d rosters from nflverse", CURRENT_SEASON)
        )

        log_message(sprintf("Found %d roster entries", nrow(rosters)), level = "INFO")

        # Validate rosters
        validate_data(
          data = rosters,
          expected_cols = c("season", "week", "team", "player_id"),
          min_rows = 100  # Should have many roster entries
        )

        # Delete existing rosters for this season
        deleted_count <- dbExecute(conn,
          sprintf("DELETE FROM rosters WHERE season = %d", CURRENT_SEASON))
        log_message(sprintf("Deleted %d existing roster entries for season %d",
                           deleted_count, CURRENT_SEASON),
                   level = "INFO")

        # Insert new rosters
        dbWriteTable(conn, "rosters", rosters, append = TRUE, row.names = FALSE)
        log_message(sprintf("Inserted %d roster entries", nrow(rosters)), level = "INFO")

        TRUE
      },
      conn = conn
    )

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
    log_message(sprintf("Games: %d (Completed: %d)",
                       summary$games_count, summary$completed_games),
               level = "INFO")
    log_message(sprintf("Plays: %d", summary$plays_count), level = "INFO")
    log_message(sprintf("Roster entries: %d", summary$roster_count), level = "INFO")

    # Check for alerts
    if (file.exists(file.path(LOG_DIR, "alerts.json"))) {
      alerts <- jsonlite::fromJSON(file.path(LOG_DIR, "alerts.json"))
      unread_alerts <- sum(sapply(alerts, function(x) x$status == "unread"))
      if (unread_alerts > 0) {
        log_message(sprintf("⚠️  %d unread alerts in %s",
                           unread_alerts,
                           file.path(LOG_DIR, "alerts.json")),
                   level = "WARNING")
      }
    }
  })
)