#!/usr/bin/env Rscript
# COMPREHENSIVE DATA BACKFILL - ALL SOURCES
# Fixes all identified issues and loads EVERYTHING

source("R/utils/error_handling.R")

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

log_message("=== COMPREHENSIVE DATA BACKFILL STARTING ===", level = "INFO")

db_params <- list(
  host = Sys.getenv("POSTGRES_HOST", "localhost"),
  port = as.integer(Sys.getenv("POSTGRES_PORT", "5544")),
  dbname = Sys.getenv("POSTGRES_DB", "devdb01"),
  user = Sys.getenv("POSTGRES_USER", "dro"),
  password = Sys.getenv("POSTGRES_PASSWORD", "sicillionbillions")
)

# Main pipeline
safe_db_operation(
  db_params = db_params,
  expr = quote({

    # ============================================================
    # 1. ESPN QBR (WEEKLY DATA - FIX: USE summary_type="week")
    # ============================================================

    qbr_data <- run_pipeline_step(
      step_name = "Load ESPN QBR (Weekly)",
      expr = {
        log_message("Fetching ESPN QBR WEEKLY data from nflverse (2006-2025)...", level = "INFO")

        qbr <- retry_operation(
          expr = load_espn_qbr(seasons = 2006:2025, summary_type = "week"),  # KEY FIX!
          max_attempts = 3,
          delay = 5,
          error_message = "Failed to load ESPN QBR from nflverse"
        )

        log_message(sprintf("Loaded %d QBR weekly records", nrow(qbr)), level = "INFO")
        qbr
      },
      conn = conn,
      validate_fn = function(data) {
        validate_data(
          data = data,
          expected_cols = c("player_id", "season", "game_week"),
          min_rows = 1000
        )
      }
    )

    run_pipeline_step(
      step_name = "Upsert ESPN QBR",
      expr = {
        qbr_clean <- qbr_data %>%
          mutate(
            rank = as.integer(rank),  # Convert rank to integer
            qb_plays = as.integer(qb_plays)  # Convert qb_plays to integer
          ) %>%
          select(
            player_id,
            season,
            game_week,
            season_type,
            name_display,
            team_abb,
            qbr_total,
            qbr_raw,
            rank,
            qualified,
            pts_added,
            epa_total,
            qb_plays,
            pass,
            run,
            sack,
            penalty,
            exp_sack
          ) %>%
          filter(!is.na(player_id), !is.na(season), !is.na(game_week))

        dbExecute(conn, "CREATE TEMP TABLE qbr_staging AS SELECT * FROM espn_qbr LIMIT 0")
        dbWriteTable(conn, "qbr_staging", qbr_clean, append = TRUE, row.names = FALSE)

        rows_updated <- dbExecute(conn, "
          INSERT INTO espn_qbr
          SELECT * FROM qbr_staging
          ON CONFLICT (player_id, season, game_week, season_type)
          DO UPDATE SET
            name_display = EXCLUDED.name_display,
            team_abb = EXCLUDED.team_abb,
            qbr_total = EXCLUDED.qbr_total,
            qbr_raw = EXCLUDED.qbr_raw,
            rank = EXCLUDED.rank,
            qualified = EXCLUDED.qualified,
            pts_added = EXCLUDED.pts_added,
            epa_total = EXCLUDED.epa_total,
            qb_plays = EXCLUDED.qb_plays,
            pass = EXCLUDED.pass,
            run = EXCLUDED.run,
            sack = EXCLUDED.sack,
            penalty = EXCLUDED.penalty,
            exp_sack = EXCLUDED.exp_sack,
            updated_at = NOW()
        ")

        dbExecute(conn, "DROP TABLE IF EXISTS qbr_staging")
        log_message(sprintf("Upserted %d QBR records", rows_updated), level = "INFO")
        TRUE
      },
      conn = conn
    )

    # ============================================================
    # 2. PFR DEFENSE STATS (2018-2025)
    # ============================================================

    pfr_defense_data <- run_pipeline_step(
      step_name = "Load PFR Defense Stats",
      expr = {
        log_message("Fetching PFR defense stats from nflverse (2018-2025)...", level = "INFO")

        pfr_def <- retry_operation(
          expr = load_pfr_advstats(seasons = 2018:2025, stat_type = "def"),
          max_attempts = 3,
          delay = 5,
          error_message = "Failed to load PFR defense stats from nflverse"
        )

        log_message(sprintf("Loaded %d PFR defense records", nrow(pfr_def)), level = "INFO")
        pfr_def
      },
      conn = conn,
      validate_fn = function(data) {
        validate_data(
          data = data,
          expected_cols = c("pfr_player_id", "game_id", "season"),
          min_rows = 1000
        )
      }
    )

    run_pipeline_step(
      step_name = "Upsert PFR Defense Stats",
      expr = {
        pfr_clean <- pfr_defense_data %>%
          select(
            pfr_player_id,
            game_id,
            season,
            week,
            game_type,
            pfr_player_name,
            team,
            opponent,
            def_targets,
            def_completions_allowed,
            def_completion_pct,
            def_yards_allowed,
            def_yards_allowed_per_cmp,
            def_yards_allowed_per_tgt,
            def_receiving_td_allowed,
            def_passer_rating_allowed,
            def_adot,
            def_air_yards_completed,
            def_yards_after_catch,
            def_times_blitzed,
            def_times_hurried,
            def_times_hitqb,
            def_sacks,
            def_pressures,
            def_tackles_combined,
            def_missed_tackles,
            def_missed_tackle_pct,
            def_ints
          ) %>%
          filter(!is.na(pfr_player_id), !is.na(game_id))

        dbExecute(conn, "CREATE TEMP TABLE pfr_defense_staging AS SELECT * FROM pfr_defense LIMIT 0")
        dbWriteTable(conn, "pfr_defense_staging", pfr_clean, append = TRUE, row.names = FALSE)

        rows_updated <- dbExecute(conn, "
          INSERT INTO pfr_defense
          SELECT * FROM pfr_defense_staging
          ON CONFLICT (pfr_player_id, game_id)
          DO UPDATE SET
            pfr_player_name = EXCLUDED.pfr_player_name,
            team = EXCLUDED.team,
            opponent = EXCLUDED.opponent,
            def_targets = EXCLUDED.def_targets,
            def_completions_allowed = EXCLUDED.def_completions_allowed,
            def_completion_pct = EXCLUDED.def_completion_pct,
            def_yards_allowed = EXCLUDED.def_yards_allowed,
            def_yards_allowed_per_cmp = EXCLUDED.def_yards_allowed_per_cmp,
            def_yards_allowed_per_tgt = EXCLUDED.def_yards_allowed_per_tgt,
            def_receiving_td_allowed = EXCLUDED.def_receiving_td_allowed,
            def_passer_rating_allowed = EXCLUDED.def_passer_rating_allowed,
            def_adot = EXCLUDED.def_adot,
            def_air_yards_completed = EXCLUDED.def_air_yards_completed,
            def_yards_after_catch = EXCLUDED.def_yards_after_catch,
            def_times_blitzed = EXCLUDED.def_times_blitzed,
            def_times_hurried = EXCLUDED.def_times_hurried,
            def_times_hitqb = EXCLUDED.def_times_hitqb,
            def_sacks = EXCLUDED.def_sacks,
            def_pressures = EXCLUDED.def_pressures,
            def_tackles_combined = EXCLUDED.def_tackles_combined,
            def_missed_tackles = EXCLUDED.def_missed_tackles,
            def_missed_tackle_pct = EXCLUDED.def_missed_tackle_pct,
            def_ints = EXCLUDED.def_ints,
            updated_at = NOW()
        ")

        dbExecute(conn, "DROP TABLE IF EXISTS pfr_defense_staging")
        log_message(sprintf("Upserted %d PFR defense records", rows_updated), level = "INFO")
        TRUE
      },
      conn = conn
    )

    # ============================================================
    # 3. SNAP COUNTS (2012-2025)
    # ============================================================

    snap_counts_data <- run_pipeline_step(
      step_name = "Load Snap Counts",
      expr = {
        log_message("Fetching snap counts from nflverse (2012-2025)...", level = "INFO")

        snaps <- retry_operation(
          expr = load_snap_counts(seasons = 2012:2025),
          max_attempts = 3,
          delay = 5,
          error_message = "Failed to load snap counts from nflverse"
        )

        log_message(sprintf("Loaded %d snap count records", nrow(snaps)), level = "INFO")
        snaps
      },
      conn = conn,
      validate_fn = function(data) {
        validate_data(
          data = data,
          expected_cols = c("pfr_player_id", "game_id", "season"),
          min_rows = 10000
        )
      }
    )

    run_pipeline_step(
      step_name = "Upsert Snap Counts",
      expr = {
        snaps_clean <- snap_counts_data %>%
          select(
            pfr_player_id,
            game_id,
            season,
            week,
            game_type,
            player,
            position,
            team,
            opponent,
            offense_snaps,
            offense_pct,
            defense_snaps,
            defense_pct,
            st_snaps,
            st_pct
          ) %>%
          filter(!is.na(pfr_player_id), !is.na(game_id))

        dbExecute(conn, "CREATE TEMP TABLE snaps_staging AS SELECT * FROM snap_counts LIMIT 0")
        dbWriteTable(conn, "snaps_staging", snaps_clean, append = TRUE, row.names = FALSE)

        rows_updated <- dbExecute(conn, "
          INSERT INTO snap_counts
          SELECT * FROM snaps_staging
          ON CONFLICT (pfr_player_id, game_id)
          DO UPDATE SET
            player = EXCLUDED.player,
            position = EXCLUDED.position,
            team = EXCLUDED.team,
            opponent = EXCLUDED.opponent,
            offense_snaps = EXCLUDED.offense_snaps,
            offense_pct = EXCLUDED.offense_pct,
            defense_snaps = EXCLUDED.defense_snaps,
            defense_pct = EXCLUDED.defense_pct,
            st_snaps = EXCLUDED.st_snaps,
            st_pct = EXCLUDED.st_pct,
            updated_at = NOW()
        ")

        dbExecute(conn, "DROP TABLE IF EXISTS snaps_staging")
        log_message(sprintf("Upserted %d snap count records", rows_updated), level = "INFO")
        TRUE
      },
      conn = conn
    )

    # ============================================================
    # 4. DEPTH CHARTS (2001-2025) - FIX: Use game_type not season_type
    # ============================================================

    depth_charts_data <- run_pipeline_step(
      step_name = "Load Depth Charts",
      expr = {
        log_message("Fetching depth charts from nflverse (2001-2025)...", level = "INFO")

        depth <- retry_operation(
          expr = load_depth_charts(seasons = 2001:2025),
          max_attempts = 3,
          delay = 5,
          error_message = "Failed to load depth charts from nflverse"
        )

        log_message(sprintf("Loaded %d depth chart records", nrow(depth)), level = "INFO")
        depth
      },
      conn = conn,
      validate_fn = function(data) {
        validate_data(
          data = data,
          expected_cols = c("gsis_id", "club_code", "season"),
          min_rows = 10000
        )
      }
    )

    run_pipeline_step(
      step_name = "Upsert Depth Charts",
      expr = {
        # KEY FIX: Rename game_type to season_type to match database schema
        # Also deduplicate - source data has exact duplicates
        # Filter out NULL weeks - these are season-level depth charts, not game-specific
        depth_clean <- depth_charts_data %>%
          mutate(season_type = game_type) %>%  # Create season_type from game_type
          select(
            season,
            season_type,  # Now available
            week,
            club_code,
            gsis_id,
            position,
            depth_team,
            jersey_number,
            last_name,
            first_name,
            full_name,
            formation,
            depth_position
          ) %>%
          filter(!is.na(gsis_id), !is.na(club_code), !is.na(season), !is.na(week)) %>%
          distinct()  # Remove exact duplicates from source data

        dbExecute(conn, "CREATE TEMP TABLE depth_staging AS SELECT * FROM depth_charts LIMIT 0")
        dbWriteTable(conn, "depth_staging", depth_clean, append = TRUE, row.names = FALSE)

        rows_updated <- dbExecute(conn, "
          INSERT INTO depth_charts
          SELECT * FROM depth_staging
          ON CONFLICT (gsis_id, club_code, season, season_type, week, depth_team, formation, position, depth_position)
          DO UPDATE SET
            jersey_number = EXCLUDED.jersey_number,
            last_name = EXCLUDED.last_name,
            first_name = EXCLUDED.first_name,
            full_name = EXCLUDED.full_name,
            updated_at = NOW()
        ")

        dbExecute(conn, "DROP TABLE IF EXISTS depth_staging")
        log_message(sprintf("Upserted %d depth chart records", rows_updated), level = "INFO")
        TRUE
      },
      conn = conn
    )

    # ============================================================
    # 5. VERIFICATION
    # ============================================================

    run_pipeline_step(
      step_name = "Verify all data",
      expr = {
        # ESPN QBR
        qbr_summary <- dbGetQuery(conn, "
          SELECT
            season,
            COUNT(DISTINCT player_id) as unique_qbs,
            COUNT(*) as total_records,
            ROUND(CAST(AVG(qbr_total) AS numeric), 2) as avg_qbr
          FROM espn_qbr
          WHERE qbr_total IS NOT NULL
          GROUP BY season
          ORDER BY season DESC
          LIMIT 5
        ")

        log_message("=== ESPN QBR Summary (Last 5 Seasons) ===", level = "INFO")
        for (i in 1:nrow(qbr_summary)) {
          log_message(sprintf("  Season %.0f: %.0f QBs, %.0f records, Avg QBR: %.2f",
                             qbr_summary$season[i],
                             qbr_summary$unique_qbs[i],
                             qbr_summary$total_records[i],
                             qbr_summary$avg_qbr[i]),
                     level = "INFO")
        }

        # PFR Defense
        def_summary <- dbGetQuery(conn, "
          SELECT season, COUNT(DISTINCT pfr_player_id) as unique_defenders, COUNT(*) as total_records
          FROM pfr_defense
          GROUP BY season
          ORDER BY season DESC
          LIMIT 5
        ")

        log_message("=== PFR Defense Stats Summary (Last 5 Seasons) ===", level = "INFO")
        for (i in 1:nrow(def_summary)) {
          log_message(sprintf("  Season %.0f: %.0f defenders, %.0f records",
                             def_summary$season[i],
                             def_summary$unique_defenders[i],
                             def_summary$total_records[i]),
                     level = "INFO")
        }

        # Snap Counts
        snaps_summary <- dbGetQuery(conn, "
          SELECT season, COUNT(DISTINCT pfr_player_id) as unique_players, COUNT(*) as total_records
          FROM snap_counts
          GROUP BY season
          ORDER BY season DESC
          LIMIT 5
        ")

        log_message("=== Snap Counts Summary (Last 5 Seasons) ===", level = "INFO")
        for (i in 1:nrow(snaps_summary)) {
          log_message(sprintf("  Season %.0f: %.0f players, %.0f records",
                             snaps_summary$season[i],
                             snaps_summary$unique_players[i],
                             snaps_summary$total_records[i]),
                     level = "INFO")
        }

        # Depth Charts
        depth_summary <- dbGetQuery(conn, "
          SELECT season, COUNT(DISTINCT gsis_id) as unique_players, COUNT(*) as total_records
          FROM depth_charts
          WHERE depth_team = '1'
          GROUP BY season
          ORDER BY season DESC
          LIMIT 5
        ")

        log_message("=== Depth Charts Summary - Starters (Last 5 Seasons) ===", level = "INFO")
        for (i in 1:nrow(depth_summary)) {
          log_message(sprintf("  Season %.0f: %.0f starters, %.0f records",
                             depth_summary$season[i],
                             depth_summary$unique_players[i],
                             depth_summary$total_records[i]),
                     level = "INFO")
        }

        TRUE
      },
      conn = conn
    )

    log_message("=== COMPREHENSIVE DATA BACKFILL COMPLETE ===", level = "INFO")
    log_message("All data sources loaded successfully", level = "INFO")
  })
)
