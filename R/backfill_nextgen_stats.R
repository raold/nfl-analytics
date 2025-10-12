#!/usr/bin/env Rscript
# Backfill NFL Next Gen Stats from nflverse
# Loads passing, rushing, and receiving Next Gen Stats (2016-present)

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
log_message("=== NFL Next Gen Stats Backfill Starting ===", level = "INFO")

# ============================================================
# DATABASE CONNECTION
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
    # 1. LOAD NEXT GEN PASSING STATS
    # ============================================================

    passing_data <- run_pipeline_step(
      step_name = "Load Next Gen Passing Stats",
      expr = {
        log_message("Fetching Next Gen passing stats from nflverse (2016-2024)...", level = "INFO")

        # Load passing stats for all available seasons
        passing <- retry_operation(
          expr = load_nextgen_stats(seasons = 2016:2025, stat_type = "passing"),
          max_attempts = 3,
          delay = 5,
          error_message = "Failed to load NGS passing stats from nflverse"
        )

        log_message(sprintf("Loaded %d NGS passing records", nrow(passing)), level = "INFO")

        passing
      },
      conn = conn,
      validate_fn = function(data) {
        validate_data(
          data = data,
          expected_cols = c("player_gsis_id", "season", "week"),
          min_rows = 1000  # Should have thousands of QB-week records
        )
      }
    )

    # Upsert passing stats
    run_pipeline_step(
      step_name = "Upsert Next Gen Passing Stats",
      expr = {
        # Select and rename columns to match our schema
        passing_clean <- passing_data %>%
          select(
            player_id = player_gsis_id,
            player_display_name,
            player_position,
            season,
            week,
            attempts,
            pass_yards,
            pass_touchdowns,
            interceptions,
            passer_rating,
            completions,
            completion_percentage,
            avg_time_to_throw,
            avg_completed_air_yards,
            avg_intended_air_yards,
            avg_air_yards_differential,
            aggressiveness,
            max_completed_air_distance,
            avg_air_yards_to_sticks,
            completion_percentage_above_expectation,
            expected_completion_percentage
          ) %>%
          filter(!is.na(player_id), !is.na(season), !is.na(week))

        # Write to database with ON CONFLICT UPDATE
        rows_upserted <- dbExecute(conn, "
          CREATE TEMP TABLE passing_staging AS
          SELECT * FROM nextgen_passing LIMIT 0
        ")

        dbWriteTable(conn, "passing_staging", passing_clean, append = TRUE, row.names = FALSE)

        rows_updated <- dbExecute(conn, "
          INSERT INTO nextgen_passing
          SELECT * FROM passing_staging
          ON CONFLICT (player_id, season, week)
          DO UPDATE SET
            player_display_name = EXCLUDED.player_display_name,
            player_position = EXCLUDED.player_position,
            attempts = EXCLUDED.attempts,
            pass_yards = EXCLUDED.pass_yards,
            pass_touchdowns = EXCLUDED.pass_touchdowns,
            interceptions = EXCLUDED.interceptions,
            passer_rating = EXCLUDED.passer_rating,
            completions = EXCLUDED.completions,
            completion_percentage = EXCLUDED.completion_percentage,
            avg_time_to_throw = EXCLUDED.avg_time_to_throw,
            avg_completed_air_yards = EXCLUDED.avg_completed_air_yards,
            avg_intended_air_yards = EXCLUDED.avg_intended_air_yards,
            avg_air_yards_differential = EXCLUDED.avg_air_yards_differential,
            aggressiveness = EXCLUDED.aggressiveness,
            max_completed_air_distance = EXCLUDED.max_completed_air_distance,
            avg_air_yards_to_sticks = EXCLUDED.avg_air_yards_to_sticks,
            completion_percentage_above_expectation = EXCLUDED.completion_percentage_above_expectation,
            expected_completion_percentage = EXCLUDED.expected_completion_percentage,
            updated_at = NOW()
        ")

        dbExecute(conn, "DROP TABLE IF EXISTS passing_staging")

        log_message(sprintf("Upserted %d passing records", rows_updated), level = "INFO")
        TRUE
      },
      conn = conn
    )

    # ============================================================
    # 2. LOAD NEXT GEN RUSHING STATS
    # ============================================================

    rushing_data <- run_pipeline_step(
      step_name = "Load Next Gen Rushing Stats",
      expr = {
        log_message("Fetching Next Gen rushing stats from nflverse (2016-2024)...", level = "INFO")

        rushing <- retry_operation(
          expr = load_nextgen_stats(seasons = 2016:2025, stat_type = "rushing"),
          max_attempts = 3,
          delay = 5,
          error_message = "Failed to load NGS rushing stats from nflverse"
        )

        log_message(sprintf("Loaded %d NGS rushing records", nrow(rushing)), level = "INFO")

        rushing
      },
      conn = conn,
      validate_fn = function(data) {
        validate_data(
          data = data,
          expected_cols = c("player_gsis_id", "season", "week"),
          min_rows = 1000
        )
      }
    )

    # Upsert rushing stats
    run_pipeline_step(
      step_name = "Upsert Next Gen Rushing Stats",
      expr = {
        rushing_clean <- rushing_data %>%
          select(
            player_id = player_gsis_id,
            player_display_name,
            player_position,
            season,
            week,
            carries = rush_attempts,
            rush_yards,
            rush_touchdowns,
            efficiency,
            percent_attempts_gte_eight_defenders,
            avg_time_to_los,
            rush_yards_over_expected,
            avg_rush_yards,
            rush_yards_over_expected_per_att,
            rush_pct_over_expected
          ) %>%
          filter(!is.na(player_id), !is.na(season), !is.na(week))

        dbExecute(conn, "CREATE TEMP TABLE rushing_staging AS SELECT * FROM nextgen_rushing LIMIT 0")
        dbWriteTable(conn, "rushing_staging", rushing_clean, append = TRUE, row.names = FALSE)

        rows_updated <- dbExecute(conn, "
          INSERT INTO nextgen_rushing
          SELECT * FROM rushing_staging
          ON CONFLICT (player_id, season, week)
          DO UPDATE SET
            player_display_name = EXCLUDED.player_display_name,
            player_position = EXCLUDED.player_position,
            carries = EXCLUDED.carries,
            rush_yards = EXCLUDED.rush_yards,
            rush_touchdowns = EXCLUDED.rush_touchdowns,
            efficiency = EXCLUDED.efficiency,
            percent_attempts_gte_eight_defenders = EXCLUDED.percent_attempts_gte_eight_defenders,
            avg_time_to_los = EXCLUDED.avg_time_to_los,
            rush_yards_over_expected = EXCLUDED.rush_yards_over_expected,
            avg_rush_yards = EXCLUDED.avg_rush_yards,
            rush_yards_over_expected_per_att = EXCLUDED.rush_yards_over_expected_per_att,
            rush_pct_over_expected = EXCLUDED.rush_pct_over_expected,
            updated_at = NOW()
        ")

        dbExecute(conn, "DROP TABLE IF EXISTS rushing_staging")

        log_message(sprintf("Upserted %d rushing records", rows_updated), level = "INFO")
        TRUE
      },
      conn = conn
    )

    # ============================================================
    # 3. LOAD NEXT GEN RECEIVING STATS
    # ============================================================

    receiving_data <- run_pipeline_step(
      step_name = "Load Next Gen Receiving Stats",
      expr = {
        log_message("Fetching Next Gen receiving stats from nflverse (2016-2024)...", level = "INFO")

        receiving <- retry_operation(
          expr = load_nextgen_stats(seasons = 2016:2025, stat_type = "receiving"),
          max_attempts = 3,
          delay = 5,
          error_message = "Failed to load NGS receiving stats from nflverse"
        )

        log_message(sprintf("Loaded %d NGS receiving records", nrow(receiving)), level = "INFO")

        receiving
      },
      conn = conn,
      validate_fn = function(data) {
        validate_data(
          data = data,
          expected_cols = c("player_gsis_id", "season", "week"),
          min_rows = 1000
        )
      }
    )

    # Upsert receiving stats
    run_pipeline_step(
      step_name = "Upsert Next Gen Receiving Stats",
      expr = {
        receiving_clean <- receiving_data %>%
          select(
            player_id = player_gsis_id,
            player_display_name,
            player_position,
            season,
            week,
            targets,
            receptions,
            receiving_yards = yards,
            receiving_touchdowns = rec_touchdowns,
            avg_cushion,
            avg_separation,
            avg_intended_air_yards,
            percent_share_of_intended_air_yards,
            catch_percentage,
            avg_yac,
            avg_expected_yac,
            avg_yac_above_expectation
          ) %>%
          filter(!is.na(player_id), !is.na(season), !is.na(week))

        dbExecute(conn, "CREATE TEMP TABLE receiving_staging AS SELECT * FROM nextgen_receiving LIMIT 0")
        dbWriteTable(conn, "receiving_staging", receiving_clean, append = TRUE, row.names = FALSE)

        rows_updated <- dbExecute(conn, "
          INSERT INTO nextgen_receiving
          SELECT * FROM receiving_staging
          ON CONFLICT (player_id, season, week)
          DO UPDATE SET
            player_display_name = EXCLUDED.player_display_name,
            player_position = EXCLUDED.player_position,
            targets = EXCLUDED.targets,
            receptions = EXCLUDED.receptions,
            receiving_yards = EXCLUDED.receiving_yards,
            receiving_touchdowns = EXCLUDED.receiving_touchdowns,
            avg_cushion = EXCLUDED.avg_cushion,
            avg_separation = EXCLUDED.avg_separation,
            avg_intended_air_yards = EXCLUDED.avg_intended_air_yards,
            percent_share_of_intended_air_yards = EXCLUDED.percent_share_of_intended_air_yards,
            catch_percentage = EXCLUDED.catch_percentage,
            avg_yac = EXCLUDED.avg_yac,
            avg_expected_yac = EXCLUDED.avg_expected_yac,
            avg_yac_above_expectation = EXCLUDED.avg_yac_above_expectation,
            updated_at = NOW()
        ")

        dbExecute(conn, "DROP TABLE IF EXISTS receiving_staging")

        log_message(sprintf("Upserted %d receiving records", rows_updated), level = "INFO")
        TRUE
      },
      conn = conn
    )

    # ============================================================
    # 4. VERIFICATION
    # ============================================================

    run_pipeline_step(
      step_name = "Verify Next Gen Stats backfill",
      expr = {
        # Check coverage by season
        passing_summary <- dbGetQuery(conn, "
          SELECT
            season,
            COUNT(DISTINCT player_id) as unique_qbs,
            COUNT(*) as total_records,
            ROUND(CAST(AVG(completion_percentage_above_expectation) AS numeric), 2) as avg_cpoe
          FROM nextgen_passing
          WHERE attempts >= 10
          GROUP BY season
          ORDER BY season
        ")

        log_message("=== Next Gen Passing Stats Summary ===", level = "INFO")
        for (i in 1:nrow(passing_summary)) {
          log_message(sprintf("  Season %.0f: %.0f QBs, %.0f records, Avg CPOE: %.2f",
                             passing_summary$season[i],
                             passing_summary$unique_qbs[i],
                             passing_summary$total_records[i],
                             passing_summary$avg_cpoe[i]),
                     level = "INFO")
        }

        rushing_summary <- dbGetQuery(conn, "
          SELECT season, COUNT(DISTINCT player_id) as unique_rushers, COUNT(*) as total_records
          FROM nextgen_rushing
          WHERE carries >= 5
          GROUP BY season
          ORDER BY season
        ")

        log_message("=== Next Gen Rushing Stats Summary ===", level = "INFO")
        for (i in 1:nrow(rushing_summary)) {
          log_message(sprintf("  Season %.0f: %.0f rushers, %.0f records",
                             rushing_summary$season[i],
                             rushing_summary$unique_rushers[i],
                             rushing_summary$total_records[i]),
                     level = "INFO")
        }

        receiving_summary <- dbGetQuery(conn, "
          SELECT season, COUNT(DISTINCT player_id) as unique_receivers, COUNT(*) as total_records
          FROM nextgen_receiving
          WHERE targets >= 3
          GROUP BY season
          ORDER BY season
        ")

        log_message("=== Next Gen Receiving Stats Summary ===", level = "INFO")
        for (i in 1:nrow(receiving_summary)) {
          log_message(sprintf("  Season %.0f: %.0f receivers, %.0f records",
                             receiving_summary$season[i],
                             receiving_summary$unique_receivers[i],
                             receiving_summary$total_records[i]),
                     level = "INFO")
        }

        TRUE
      },
      conn = conn
    )

    log_message("=== Next Gen Stats Backfill Complete ===", level = "INFO")
    log_message("Next steps:", level = "INFO")
    log_message("  1. Integrate NGS features into py/features/asof_features_enhanced.py", level = "INFO")
    log_message("  2. Test team-level NGS aggregates (avg CPOE, avg separation, etc.)", level = "INFO")
    log_message("  3. Retrain models with new NGS features", level = "INFO")
  })
)
