#!/usr/bin/env Rscript
# Ingest 2025 NFL Season Data - Production Version with Error Handling
# Run this to update with latest 2025 games, plays, and rosters

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

# Initialize logging
log_message("=== 2025 NFL Season Data Ingestion Starting ===", level = "INFO")

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

    games_2025 <- run_pipeline_step(
      step_name = "Load 2025 Schedules",
      expr = {
        # Load schedules with retry for network issues
        schedules <- retry_operation(
          expr = load_schedules(2025),
          max_attempts = 3,
          delay = 5,
          error_message = "Failed to load schedules from nflverse"
        )

        log_message(sprintf("Found %d games (%d completed)",
                           nrow(schedules),
                           sum(!is.na(schedules$home_score))),
                   level = "INFO")

        # Transform to games table format
        games_data <- schedules %>%
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

        games_data
      },
      conn = conn,
      validate_fn = function(data) {
        validate_data(
          data = data,
          expected_cols = c("game_id", "season", "week", "home_team", "away_team"),
          min_rows = 250,  # NFL season should have 272 games
          max_rows = 300
        )
      }
    )

    # Upsert games with error handling
    run_pipeline_step(
      step_name = "Upsert Games to Database",
      expr = {
        # Delete existing 2025 games first
        deleted_count <- dbExecute(conn, "DELETE FROM games WHERE season = 2025")
        log_message(sprintf("Deleted %d existing 2025 games", deleted_count), level = "INFO")

        # Insert all 2025 games
        rows_inserted <- dbWriteTable(conn, "games", games_2025, append = TRUE, row.names = FALSE)
        log_message(sprintf("Inserted %d games", nrow(games_2025)), level = "INFO")

        # Verify insertion
        verification <- dbGetQuery(conn, "SELECT COUNT(*) as count FROM games WHERE season = 2025")
        if (verification$count != nrow(games_2025)) {
          stop(sprintf("Data verification failed: Expected %d games, found %d",
                      nrow(games_2025), verification$count))
        }

        TRUE
      },
      conn = conn
    )

    # ============================================================
    # 2. PLAY-BY-PLAY (PLAYS TABLE)
    # ============================================================

    plays_2025 <- run_pipeline_step(
      step_name = "Load 2025 Play-by-Play",
      expr = {
        # Load play-by-play with retry
        pbp <- retry_operation(
          expr = load_pbp(2025),
          max_attempts = 3,
          delay = 5,
          error_message = "Failed to load play-by-play from nflverse"
        )

        log_message(sprintf("Found %d plays from %d games",
                           nrow(pbp), length(unique(pbp$game_id))),
                   level = "INFO")

        # Handle column name variations (quarter vs qtr)
        if ("qtr" %in% names(pbp) && !"quarter" %in% names(pbp)) {
          pbp$quarter <- pbp$qtr
        }
        if ("game_seconds_remaining" %in% names(pbp) && !"time_seconds" %in% names(pbp)) {
          pbp$time_seconds <- pbp$game_seconds_remaining
        }

        # Select and transform columns
        plays_data <- pbp %>%
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
        if ("game_seconds_remaining" %in% names(plays_data) && !"time_seconds" %in% names(plays_data)) {
          plays_data <- plays_data %>% rename(time_seconds = game_seconds_remaining)
        }

        plays_data
      },
      conn = conn,
      validate_fn = function(data) {
        validate_data(
          data = data,
          expected_cols = c("game_id", "play_id"),
          min_rows = 5000,  # Minimum expected plays for partial season
          max_rows = 50000  # Maximum reasonable for full season
        )
      }
    )

    # Upsert plays with error handling
    run_pipeline_step(
      step_name = "Upsert Plays to Database",
      expr = {
        # Delete existing 2025 plays first
        deleted_count <- dbExecute(conn, "DELETE FROM plays WHERE game_id LIKE '2025_%'")
        log_message(sprintf("Deleted %d existing 2025 plays", deleted_count), level = "INFO")

        # Insert plays in batches to avoid memory issues
        batch_size <- 10000
        total_rows <- nrow(plays_2025)

        for (i in seq(1, total_rows, by = batch_size)) {
          end_idx <- min(i + batch_size - 1, total_rows)
          batch <- plays_2025[i:end_idx, ]

          dbWriteTable(conn, "plays", batch, append = TRUE, row.names = FALSE)
          log_message(sprintf("Inserted plays batch %d-%d of %d",
                             i, end_idx, total_rows),
                     level = "INFO")
        }

        # Verify insertion
        verification <- dbGetQuery(conn,
          "SELECT COUNT(*) as count FROM plays WHERE game_id LIKE '2025_%'")
        if (abs(verification$count - nrow(plays_2025)) > 10) {  # Allow small discrepancy
          log_message(sprintf("Warning: Expected %d plays, found %d",
                             nrow(plays_2025), verification$count),
                     level = "WARNING")
        }

        TRUE
      },
      conn = conn
    )

    # ============================================================
    # 3. ROSTERS (ROSTERS TABLE)
    # ============================================================

    run_pipeline_step(
      step_name = "Load 2025 Rosters",
      expr = {
        # Load rosters with retry
        rosters <- retry_operation(
          expr = load_rosters(2025),
          max_attempts = 3,
          delay = 5,
          error_message = "Failed to load rosters from nflverse"
        )

        log_message(sprintf("Found %d roster entries", nrow(rosters)), level = "INFO")

        # Validate rosters
        validate_data(
          data = rosters,
          expected_cols = c("season", "week", "team", "player_id"),
          min_rows = 1000  # Should have many roster entries
        )

        # Delete existing 2025 rosters
        deleted_count <- dbExecute(conn, "DELETE FROM rosters WHERE season = 2025")
        log_message(sprintf("Deleted %d existing 2025 roster entries", deleted_count),
                   level = "INFO")

        # Insert new rosters
        dbWriteTable(conn, "rosters", rosters, append = TRUE, row.names = FALSE)
        log_message(sprintf("Inserted %d roster entries", nrow(rosters)), level = "INFO")

        TRUE
      },
      conn = conn
    )

    # ============================================================
    # 4. CALCULATE TURNOVERS AND PENALTIES
    # ============================================================

    run_pipeline_step(
      step_name = "Calculate Game Statistics",
      expr = {
        # Update turnovers from play-by-play
        updated <- dbExecute(conn, "
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
            WHERE g2.season = 2025
            GROUP BY p.game_id
          ) subq
          WHERE g.game_id = subq.game_id
        ")

        log_message(sprintf("Updated statistics for %d games", updated), level = "INFO")
        TRUE
      },
      conn = conn
    )

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
    summary <- dbGetQuery(conn, "
      SELECT
        (SELECT COUNT(*) FROM games WHERE season = 2025) as games_count,
        (SELECT COUNT(*) FROM plays WHERE game_id LIKE '2025_%') as plays_count,
        (SELECT COUNT(*) FROM rosters WHERE season = 2025) as roster_count,
        (SELECT COUNT(*) FROM games WHERE season = 2025 AND home_score IS NOT NULL) as completed_games
    ")

    log_message("=== 2025 Season Ingestion Complete ===", level = "INFO")
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