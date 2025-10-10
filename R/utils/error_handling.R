# Error Handling Utilities for R ETL Scripts
# Provides robust error handling, logging, and notification functions
# Source this file in all R scripts: source("R/utils/error_handling.R")

# Load required libraries
suppressPackageStartupMessages({
  if (!require(DBI, quietly = TRUE)) install.packages("DBI")
  if (!require(jsonlite, quietly = TRUE)) install.packages("jsonlite")
})

# ============================================================
# CONFIGURATION
# ============================================================

# Set up logging directory
LOG_DIR <- Sys.getenv("R_LOG_DIR", "logs/r_etl")
if (!dir.exists(LOG_DIR)) {
  dir.create(LOG_DIR, recursive = TRUE)
}

# Error notification file (simple file-based alerts to start)
ERROR_LOG <- file.path(LOG_DIR, "errors.log")
NOTIFICATION_FILE <- file.path(LOG_DIR, "alerts.json")

# ============================================================
# LOGGING FUNCTIONS
# ============================================================

#' Log a message with timestamp and level
#' @param message The message to log
#' @param level One of: INFO, WARNING, ERROR, CRITICAL
#' @param script_name Name of the calling script
log_message <- function(message, level = "INFO", script_name = NULL) {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")

  if (is.null(script_name)) {
    # Try to get script name from command args
    args <- commandArgs(trailingOnly = FALSE)
    script_idx <- grep("--file=", args)
    if (length(script_idx) > 0) {
      script_name <- basename(sub("--file=", "", args[script_idx]))
    } else {
      script_name <- "interactive"
    }
  }

  log_entry <- sprintf("[%s] [%s] [%s] %s", timestamp, level, script_name, message)

  # Print to console
  if (level %in% c("ERROR", "CRITICAL")) {
    cat(log_entry, "\n", file = stderr())
  } else {
    cat(log_entry, "\n")
  }

  # Write to log file
  log_file <- file.path(LOG_DIR, paste0(script_name, "_", format(Sys.Date(), "%Y%m%d"), ".log"))
  cat(log_entry, "\n", file = log_file, append = TRUE)

  # If error, also write to error log
  if (level %in% c("ERROR", "CRITICAL")) {
    cat(log_entry, "\n", file = ERROR_LOG, append = TRUE)

    # Create notification
    create_notification(message, level, script_name)
  }
}

#' Create an error notification
create_notification <- function(message, level, script_name) {
  notification <- list(
    timestamp = Sys.time(),
    level = level,
    script = script_name,
    message = message,
    hostname = Sys.info()["nodename"],
    status = "unread"
  )

  # Read existing notifications
  if (file.exists(NOTIFICATION_FILE)) {
    existing <- jsonlite::fromJSON(NOTIFICATION_FILE, simplifyVector = FALSE)
  } else {
    existing <- list()
  }

  # Add new notification
  existing[[length(existing) + 1]] <- notification

  # Keep only last 100 notifications
  if (length(existing) > 100) {
    existing <- tail(existing, 100)
  }

  # Write back
  jsonlite::write_json(existing, NOTIFICATION_FILE, pretty = TRUE, auto_unbox = TRUE)
}

# ============================================================
# ERROR HANDLING WRAPPERS
# ============================================================

#' Safe execution wrapper with automatic rollback
#' @param expr Expression to execute
#' @param conn Database connection (optional, for rollback)
#' @param error_message Custom error message
#' @param finally_expr Expression to run in finally block
safe_execute <- function(expr, conn = NULL, error_message = NULL, finally_expr = NULL) {
  tryCatch({
    result <- eval(expr)
    result
  }, error = function(e) {
    # Log the error
    if (is.null(error_message)) {
      error_message <- "Operation failed"
    }
    log_message(
      sprintf("%s: %s", error_message, e$message),
      level = "ERROR"
    )

    # Rollback database transaction if connection provided
    if (!is.null(conn) && dbIsValid(conn)) {
      tryCatch({
        dbRollback(conn)
        log_message("Database transaction rolled back", level = "WARNING")
      }, error = function(rollback_err) {
        log_message(
          sprintf("Failed to rollback: %s", rollback_err$message),
          level = "CRITICAL"
        )
      })
    }

    # Re-throw the error
    stop(e)
  }, warning = function(w) {
    log_message(sprintf("Warning: %s", w$message), level = "WARNING")
    # Continue execution but log warning
    suppressWarnings(eval(expr))
  }, finally = {
    if (!is.null(finally_expr)) {
      eval(finally_expr)
    }
  })
}

#' Retry wrapper for flaky operations
#' @param expr Expression to execute
#' @param max_attempts Maximum number of attempts
#' @param delay Delay between attempts in seconds
#' @param error_message Custom error message
retry_operation <- function(expr, max_attempts = 3, delay = 2, error_message = NULL) {
  attempt <- 1

  while (attempt <= max_attempts) {
    result <- tryCatch({
      eval(expr)
    }, error = function(e) {
      if (attempt == max_attempts) {
        # Final attempt failed
        if (is.null(error_message)) {
          error_message <- "Operation failed after all retries"
        }
        log_message(
          sprintf("%s (attempt %d/%d): %s", error_message, attempt, max_attempts, e$message),
          level = "ERROR"
        )
        stop(e)
      } else {
        # Log retry
        log_message(
          sprintf("Attempt %d/%d failed, retrying in %d seconds: %s",
                  attempt, max_attempts, delay, e$message),
          level = "WARNING"
        )
        Sys.sleep(delay)
        NULL  # Continue to next attempt
      }
    })

    if (!is.null(result)) {
      if (attempt > 1) {
        log_message(sprintf("Operation succeeded on attempt %d", attempt), level = "INFO")
      }
      return(result)
    }

    attempt <- attempt + 1
  }
}

# ============================================================
# DATABASE HELPERS
# ============================================================

#' Safe database connection with automatic cleanup
#' @param db_params Database connection parameters
#' @param expr Expression to execute with the connection
safe_db_operation <- function(db_params = NULL, expr) {
  # Default parameters if not provided
  if (is.null(db_params)) {
    db_params <- list(
      host = Sys.getenv("POSTGRES_HOST", "localhost"),
      port = as.integer(Sys.getenv("POSTGRES_PORT", "5544")),
      dbname = Sys.getenv("POSTGRES_DB", "devdb01"),
      user = Sys.getenv("POSTGRES_USER", "dro"),
      password = Sys.getenv("POSTGRES_PASSWORD", "sicillionbillions")
    )
  }

  conn <- NULL

  tryCatch({
    # Establish connection
    conn <- do.call(DBI::dbConnect, c(list(RPostgres::Postgres()), db_params))
    log_message("Database connection established", level = "INFO")

    # Begin transaction
    dbBegin(conn)

    # Execute the expression
    result <- eval(expr)

    # Commit if successful
    dbCommit(conn)
    log_message("Database transaction committed", level = "INFO")

    result
  }, error = function(e) {
    log_message(sprintf("Database operation failed: %s", e$message), level = "ERROR")

    # Rollback if connection exists
    if (!is.null(conn) && dbIsValid(conn)) {
      tryCatch({
        dbRollback(conn)
        log_message("Transaction rolled back", level = "WARNING")
      }, error = function(rollback_err) {
        log_message(sprintf("Rollback failed: %s", rollback_err$message), level = "CRITICAL")
      })
    }
    stop(e)
  }, finally = {
    # Always disconnect
    if (!is.null(conn) && dbIsValid(conn)) {
      dbDisconnect(conn)
      log_message("Database connection closed", level = "INFO")
    }
  })
}

# ============================================================
# DATA VALIDATION
# ============================================================

#' Validate data before loading
#' @param data Data frame to validate
#' @param expected_cols Expected column names
#' @param min_rows Minimum expected rows
#' @param max_rows Maximum expected rows
validate_data <- function(data, expected_cols = NULL, min_rows = 1, max_rows = Inf) {
  # Check if data exists
  if (is.null(data) || !is.data.frame(data)) {
    stop("Data validation failed: Input is not a data frame")
  }

  # Check row count
  row_count <- nrow(data)
  if (row_count < min_rows) {
    stop(sprintf("Data validation failed: Too few rows (%d < %d)", row_count, min_rows))
  }
  if (row_count > max_rows) {
    stop(sprintf("Data validation failed: Too many rows (%d > %d)", row_count, max_rows))
  }

  # Check columns
  if (!is.null(expected_cols)) {
    missing_cols <- setdiff(expected_cols, names(data))
    if (length(missing_cols) > 0) {
      stop(sprintf("Data validation failed: Missing columns: %s",
                   paste(missing_cols, collapse = ", ")))
    }
  }

  # Check for all NA columns
  all_na_cols <- names(data)[sapply(data, function(x) all(is.na(x)))]
  if (length(all_na_cols) > 0) {
    log_message(sprintf("Warning: Columns with all NA values: %s",
                        paste(all_na_cols, collapse = ", ")),
                level = "WARNING")
  }

  log_message(sprintf("Data validation passed: %d rows, %d columns",
                      row_count, ncol(data)),
              level = "INFO")

  return(TRUE)
}

# ============================================================
# PIPELINE HELPERS
# ============================================================

#' Run a pipeline step with full error handling
#' @param step_name Name of the pipeline step
#' @param expr Expression to execute
#' @param conn Database connection (optional)
#' @param validate_fn Validation function to run on result (optional)
run_pipeline_step <- function(step_name, expr, conn = NULL, validate_fn = NULL) {
  log_message(sprintf("Starting pipeline step: %s", step_name), level = "INFO")
  step_start_time <- Sys.time()

  result <- safe_execute(
    expr = expr,
    conn = conn,
    error_message = sprintf("Pipeline step '%s' failed", step_name),
    finally_expr = substitute({
      elapsed <- difftime(Sys.time(), step_start_time, units = "secs")
      log_message(sprintf("Pipeline step '%s' completed in %.2f seconds",
                          step_name, elapsed),
                  level = "INFO")
    }, list(step_start_time = step_start_time, step_name = step_name))
  )

  # Run validation if provided
  if (!is.null(validate_fn)) {
    validate_fn(result)
  }

  result
}

# ============================================================
# EXPORT SUCCESS MESSAGE
# ============================================================

log_message("Error handling utilities loaded successfully", level = "INFO")

# Return TRUE to indicate successful loading
TRUE