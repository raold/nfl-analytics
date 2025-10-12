#!/usr/bin/env Rscript
# DBA AUDIT AGENT
# Automated database health monitoring and audit checks
# Based on the comprehensive DBA audit performed manually

source("R/utils/error_handling.R")

safe_execute(
  expr = {
    suppressPackageStartupMessages({
      library(DBI)
      library(RPostgres)
      library(dplyr)
      library(jsonlite)
    })
  },
  error_message = "Failed to load required R packages"
)

# Database connection parameters
db_params <- list(
  host = Sys.getenv("POSTGRES_HOST", "localhost"),
  port = as.integer(Sys.getenv("POSTGRES_PORT", "5544")),
  dbname = Sys.getenv("POSTGRES_DB", "devdb01"),
  user = Sys.getenv("POSTGRES_USER", "dro"),
  password = Sys.getenv("POSTGRES_PASSWORD", "sicillionbillions")
)

# Generate timestamped log file path
log_timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
log_file <- sprintf("logs/dba_audits/audit_%s.log", log_timestamp)

# Initialize audit results
audit_results <- list()
critical_failures <- 0
warnings <- 0

# Log function
log_audit <- function(message, level = "INFO") {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  log_line <- sprintf("[%s] [%s] %s", timestamp, level, message)
  cat(log_line, "\n")
  cat(log_line, "\n", file = log_file, append = TRUE)
}

# Store audit result in database
store_audit_result <- function(conn, audit_type, table_name = NULL, check_name, status, violation_count = NULL, message = "", metadata = NULL) {
  tryCatch({
    metadata_json <- if (!is.null(metadata)) toJSON(metadata, auto_unbox = TRUE) else NULL

    query <- "
      INSERT INTO dba_audit_log (audit_type, table_name, check_name, status, violation_count, message, metadata)
      VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
    "

    dbExecute(conn, query, list(
      audit_type,
      table_name,
      check_name,
      status,
      violation_count,
      message,
      metadata_json
    ))
  }, error = function(e) {
    log_audit(sprintf("Failed to store audit result: %s", e$message), level = "WARNING")
  })
}

# Check 1: Database Overview
check_database_overview <- function(conn) {
  log_audit("=== CHECK: Database Overview ===")

  result <- dbGetQuery(conn, "
    SELECT
      current_database() as database_name,
      pg_size_pretty(pg_database_size(current_database())) as total_size,
      (SELECT count(*) FROM pg_tables WHERE schemaname = 'public') as table_count,
      (SELECT count(*) FROM pg_views WHERE schemaname = 'public') as view_count,
      (SELECT count(*) FROM pg_indexes WHERE schemaname = 'public') as index_count
  ")

  log_audit(sprintf("Database: %s, Size: %s", result$database_name, result$total_size))
  log_audit(sprintf("Tables: %d, Views: %d, Indexes: %d",
                   as.integer(result$table_count), as.integer(result$view_count), as.integer(result$index_count)))

  store_audit_result(conn, "database_overview", NULL, "database_size",
                    "PASS", NULL, sprintf("Size: %s", result$total_size), as.list(result))

  return(list(status = "PASS", data = result))
}

# Check 2: Primary Key Validation
check_primary_keys <- function(conn) {
  log_audit("=== CHECK: Primary Key Validation ===")

  result <- dbGetQuery(conn, "
    SELECT t.table_name
    FROM information_schema.tables t
    WHERE t.table_schema = 'public'
      AND t.table_type = 'BASE TABLE'
      AND NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints tc
        WHERE tc.table_name = t.table_name
          AND tc.table_schema = 'public'
          AND tc.constraint_type = 'PRIMARY KEY'
      )
    ORDER BY t.table_name
  ")

  n_missing <- nrow(result)

  if (n_missing == 0) {
    log_audit("✅ PASS: All tables have primary keys")
    store_audit_result(conn, "schema_integrity", NULL, "primary_keys", "PASS", 0L, "All tables have primary keys")
    return(list(status = "PASS", violations = 0))
  } else {
    log_audit(sprintf("⚠️  WARNING: %d tables missing primary keys: %s",
                     n_missing, paste(result$table_name, collapse = ", ")), level = "WARNING")
    store_audit_result(conn, "schema_integrity", NULL, "primary_keys", "WARNING", as.integer(n_missing),
                      sprintf("Tables missing PKs: %s", paste(result$table_name, collapse = ", ")))
    warnings <<- warnings + 1
    return(list(status = "WARNING", violations = n_missing, tables = result$table_name))
  }
}

# Check 3: Referential Integrity
check_referential_integrity <- function(conn) {
  log_audit("=== CHECK: Referential Integrity ===")

  # plays -> games
  result <- dbGetQuery(conn, "
    SELECT COUNT(*) as violations
    FROM plays p
    WHERE NOT EXISTS (SELECT 1 FROM games g WHERE g.game_id = p.game_id)
  ")

  violations <- as.integer(result$violations)

  if (violations == 0) {
    log_audit("✅ PASS: plays->games referential integrity (0 orphaned records)")
    store_audit_result(conn, "referential_integrity", "plays", "plays_to_games", "PASS", 0L, "No orphaned plays")
    return(list(status = "PASS", violations = 0))
  } else {
    log_audit(sprintf("❌ FAIL: %d orphaned plays records", violations), level = "FAIL")
    store_audit_result(conn, "referential_integrity", "plays", "plays_to_games", "FAIL", violations,
                      "Orphaned play records found")
    critical_failures <<- critical_failures + 1
    return(list(status = "FAIL", violations = violations))
  }
}

# Check 4: Duplicate Primary Keys
check_duplicate_keys <- function(conn) {
  log_audit("=== CHECK: Duplicate Primary Keys ===")

  checks <- list(
    list(table = "games", check = "SELECT COUNT(*) - COUNT(DISTINCT game_id) as dups FROM games"),
    list(table = "players", check = "SELECT COUNT(*) - COUNT(DISTINCT player_id) as dups FROM players"),
    list(table = "injuries", check = "SELECT COUNT(*) - COUNT(DISTINCT (season, game_type, team, week, gsis_id)) as dups FROM injuries"),
    list(table = "rosters_weekly", check = "SELECT COUNT(*) - COUNT(DISTINCT (season, week, game_type, team, gsis_id)) as dups FROM rosters_weekly")
  )

  all_passed <- TRUE
  for (check in checks) {
    result <- dbGetQuery(conn, check$check)
    dups <- as.integer(result$dups)

    if (dups == 0) {
      log_audit(sprintf("✅ PASS: %s - no duplicate keys", check$table))
      store_audit_result(conn, "data_quality", check$table, "duplicate_primary_keys", "PASS", 0L, "No duplicates")
    } else {
      log_audit(sprintf("❌ FAIL: %s - %d duplicate keys", check$table, dups), level = "FAIL")
      store_audit_result(conn, "data_quality", check$table, "duplicate_primary_keys", "FAIL", dups, "Duplicate keys found")
      critical_failures <<- critical_failures + 1
      all_passed <- FALSE
    }
  }

  return(list(status = if (all_passed) "PASS" else "FAIL"))
}

# Check 5: NULL Value Validation
check_null_values <- function(conn) {
  log_audit("=== CHECK: NULL Value Validation ===")

  checks <- list(
    list(table = "games", col = "game_id", query = "SELECT COUNT(*) as nulls FROM games WHERE game_id IS NULL"),
    list(table = "games", col = "home_team", query = "SELECT COUNT(*) as nulls FROM games WHERE home_team IS NULL"),
    list(table = "plays", col = "game_id", query = "SELECT COUNT(*) as nulls FROM plays WHERE game_id IS NULL"),
    list(table = "plays", col = "play_id", query = "SELECT COUNT(*) as nulls FROM plays WHERE play_id IS NULL"),
    list(table = "players", col = "player_id", query = "SELECT COUNT(*) as nulls FROM players WHERE player_id IS NULL"),
    list(table = "players", col = "player_name", query = "SELECT COUNT(*) as nulls FROM players WHERE player_name IS NULL")
  )

  all_passed <- TRUE
  for (check in checks) {
    result <- dbGetQuery(conn, check$query)
    nulls <- as.integer(result$nulls)
    check_name <- sprintf("%s.%s_null_check", check$table, check$col)

    if (nulls == 0) {
      log_audit(sprintf("✅ PASS: %s.%s - no NULL values", check$table, check$col))
      store_audit_result(conn, "data_quality", check$table, check_name, "PASS", 0L, "No NULL values")
    } else {
      log_audit(sprintf("❌ FAIL: %s.%s - %d NULL values", check$table, check$col, nulls), level = "FAIL")
      store_audit_result(conn, "data_quality", check$table, check_name, "FAIL", nulls, "NULL values found in critical column")
      critical_failures <<- critical_failures + 1
      all_passed <- FALSE
    }
  }

  return(list(status = if (all_passed) "PASS" else "FAIL"))
}

# Check 6: Data Quality Red Flags
check_data_quality <- function(conn) {
  log_audit("=== CHECK: Data Quality Red Flags ===")

  checks <- list(
    list(name = "negative_scores", query = "SELECT COUNT(*) as violations FROM games WHERE home_score < 0 OR away_score < 0"),
    list(name = "null_scores_completed_games", query = "SELECT COUNT(*) as violations FROM games WHERE home_score IS NULL AND kickoff < NOW() - INTERVAL '3 hours'"),
    list(name = "extreme_negative_yards", query = "SELECT COUNT(*) as violations FROM plays WHERE yards_gained < -50"),
    list(name = "invalid_down", query = "SELECT COUNT(*) as violations FROM plays WHERE down > 4 AND down IS NOT NULL")
  )

  all_passed <- TRUE
  for (check in checks) {
    result <- dbGetQuery(conn, check$query)
    violations <- as.integer(result$violations)

    if (violations == 0) {
      log_audit(sprintf("✅ PASS: %s - no violations", check$name))
      store_audit_result(conn, "data_quality", NULL, check$name, "PASS", 0L, "No data quality issues")
    } else {
      # null_scores_completed_games can be a warning (future games)
      if (check$name == "null_scores_completed_games" && violations < 50) {
        log_audit(sprintf("⚠️  WARNING: %s - %d violations (acceptable for recent/future games)", check$name, violations), level = "WARNING")
        store_audit_result(conn, "data_quality", NULL, check$name, "WARNING", violations, "Acceptable for scheduled games")
        warnings <<- warnings + 1
      } else {
        log_audit(sprintf("❌ FAIL: %s - %d violations", check$name, violations), level = "FAIL")
        store_audit_result(conn, "data_quality", NULL, check$name, "FAIL", violations, "Data quality violations found")
        critical_failures <<- critical_failures + 1
        all_passed <- FALSE
      }
    }
  }

  return(list(status = if (all_passed) "PASS" else if (warnings > 0) "WARNING" else "FAIL"))
}

# Check 7: Table Maintenance (with plays table monitoring)
check_table_maintenance <- function(conn) {
  log_audit("=== CHECK: Table Maintenance & Dead Row Monitoring ===")

  result <- dbGetQuery(conn, "
    SELECT
      schemaname,
      relname as table_name,
      n_live_tup as live_rows,
      n_dead_tup as dead_rows,
      ROUND(100.0 * n_dead_tup / NULLIF(n_live_tup + n_dead_tup, 0), 2) as dead_row_pct,
      last_vacuum,
      last_autovacuum,
      last_analyze,
      last_autoanalyze
    FROM pg_stat_user_tables
    WHERE schemaname = 'public'
    ORDER BY n_live_tup DESC
    LIMIT 15
  ")

  log_audit(sprintf("Analyzed %d tables", nrow(result)))

  # Check plays table specifically
  plays_row <- result[result$table_name == "plays", ]

  if (nrow(plays_row) > 0 && !is.na(plays_row$dead_row_pct)) {
    dead_pct <- plays_row$dead_row_pct

    if (dead_pct > 15) {
      log_audit(sprintf("⚠️  WARNING: plays table dead row percentage is %.2f%% (threshold: 15%%)", dead_pct), level = "WARNING")
      log_audit("RECOMMENDATION: Consider running VACUUM ANALYZE on plays table", level = "WARNING")
      store_audit_result(conn, "table_maintenance", "plays", "dead_row_percentage", "WARNING",
                        as.integer(dead_pct), sprintf("Dead rows: %.2f%%", dead_pct),
                        list(live_rows = plays_row$live_rows, dead_rows = plays_row$dead_rows))
      warnings <<- warnings + 1
    } else {
      log_audit(sprintf("✅ PASS: plays table dead row percentage is %.2f%% (healthy)", dead_pct))
      store_audit_result(conn, "table_maintenance", "plays", "dead_row_percentage", "PASS",
                        as.integer(dead_pct), sprintf("Dead rows: %.2f%%", dead_pct),
                        list(live_rows = plays_row$live_rows, dead_rows = plays_row$dead_rows))
    }
  }

  # Check for tables with >20% dead rows
  high_dead_rows <- result[!is.na(result$dead_row_pct) & result$dead_row_pct > 20, ]

  if (nrow(high_dead_rows) > 0) {
    log_audit(sprintf("⚠️  WARNING: %d tables with >20%% dead rows: %s",
                     nrow(high_dead_rows), paste(high_dead_rows$table_name, collapse = ", ")), level = "WARNING")
    for (i in 1:nrow(high_dead_rows)) {
      row <- high_dead_rows[i, ]
      store_audit_result(conn, "table_maintenance", row$table_name, "dead_row_percentage", "WARNING",
                        as.integer(row$dead_row_pct), sprintf("Dead rows: %.2f%%", row$dead_row_pct))
    }
    warnings <<- warnings + 1
  }

  return(list(status = if (nrow(high_dead_rows) > 0) "WARNING" else "PASS", data = result))
}

# Check 8: Index Coverage
check_index_coverage <- function(conn) {
  log_audit("=== CHECK: Index Coverage ===")

  result <- dbGetQuery(conn, "
    SELECT COUNT(*) as index_count
    FROM pg_indexes
    WHERE schemaname = 'public'
  ")

  idx_count <- as.integer(result$index_count)
  log_audit(sprintf("Total indexes: %d", idx_count))

  if (idx_count >= 100) {
    log_audit("✅ PASS: Comprehensive index coverage")
    store_audit_result(conn, "schema_integrity", NULL, "index_coverage", "PASS",
                      idx_count, sprintf("%d indexes", idx_count))
    return(list(status = "PASS", count = idx_count))
  } else {
    log_audit(sprintf("⚠️  WARNING: Only %d indexes (expected >= 100)", idx_count), level = "WARNING")
    store_audit_result(conn, "schema_integrity", NULL, "index_coverage", "WARNING",
                      idx_count, "Low index count")
    warnings <<- warnings + 1
    return(list(status = "WARNING", count = idx_count))
  }
}

# Check 9: Season Coverage
check_season_coverage <- function(conn) {
  log_audit("=== CHECK: Season Coverage ===")

  result <- dbGetQuery(conn, "
    WITH season_coverage AS (
      SELECT season, 'games' as source FROM games
      UNION SELECT season, 'injuries' FROM injuries
      UNION SELECT season, 'rosters_weekly' FROM rosters_weekly
      UNION SELECT season, 'depth_charts' FROM depth_charts
    )
    SELECT season, COUNT(DISTINCT source) as data_sources
    FROM season_coverage
    WHERE season >= 2020
    GROUP BY season
    ORDER BY season DESC
  ")

  log_audit(sprintf("Recent seasons analyzed: %d", nrow(result)))

  for (i in 1:nrow(result)) {
    row <- result[i, ]
    log_audit(sprintf("  Season %d: %d data sources", as.integer(row$season), as.integer(row$data_sources)))
  }

  # Check if recent seasons have good coverage
  recent_coverage <- result[result$season >= 2022, ]

  if (nrow(recent_coverage) > 0 && all(recent_coverage$data_sources >= 4)) {
    log_audit("✅ PASS: Good data coverage for recent seasons")
    store_audit_result(conn, "data_completeness", NULL, "season_coverage", "PASS",
                      NULL, "Recent seasons have comprehensive data coverage")
    return(list(status = "PASS", data = result))
  } else {
    log_audit("⚠️  WARNING: Some recent seasons have incomplete data coverage", level = "WARNING")
    store_audit_result(conn, "data_completeness", NULL, "season_coverage", "WARNING",
                      NULL, "Incomplete coverage for some seasons")
    warnings <<- warnings + 1
    return(list(status = "WARNING", data = result))
  }
}

# Main execution
log_audit("════════════════════════════════════════════════════════════════")
log_audit("NFL ANALYTICS DATABASE - DBA AUDIT AGENT")
log_audit("════════════════════════════════════════════════════════════════")
log_audit("")

safe_db_operation(
  db_params = db_params,
  expr = quote({
    log_audit("Database connection established")
    log_audit("")

    # Run all audit checks
    audit_results$overview <- check_database_overview(conn)
    audit_results$primary_keys <- check_primary_keys(conn)
    audit_results$referential_integrity <- check_referential_integrity(conn)
    audit_results$duplicate_keys <- check_duplicate_keys(conn)
    audit_results$null_values <- check_null_values(conn)
    audit_results$data_quality <- check_data_quality(conn)
    audit_results$table_maintenance <- check_table_maintenance(conn)
    audit_results$index_coverage <- check_index_coverage(conn)
    audit_results$season_coverage <- check_season_coverage(conn)

    log_audit("")
    log_audit("════════════════════════════════════════════════════════════════")
    log_audit("AUDIT SUMMARY")
    log_audit("════════════════════════════════════════════════════════════════")
    log_audit(sprintf("Critical Failures: %d", critical_failures))
    log_audit(sprintf("Warnings: %d", warnings))

    if (critical_failures > 0) {
      log_audit("❌ AUDIT STATUS: FAILED", level = "FAIL")
    } else if (warnings > 0) {
      log_audit("⚠️  AUDIT STATUS: PASSED WITH WARNINGS", level = "WARNING")
    } else {
      log_audit("✅ AUDIT STATUS: PASSED")
    }

    log_audit("")
    log_audit(sprintf("Audit log saved to: %s", log_file))
    log_audit("════════════════════════════════════════════════════════════════")
  })
)

# Exit with appropriate code
if (critical_failures > 0) {
  quit(status = 1)
} else {
  quit(status = 0)
}
