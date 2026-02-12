#!/usr/bin/env Rscript
# Unit Tests for R/utils/error_handling.R
# Uses testthat framework

library(testthat)
library(jsonlite)

# Source the file under test
source("R/utils/error_handling.R")

# ==============================================================================
# TEST: log_message()
# ==============================================================================

test_that("log_message writes to log file with correct format", {
  # Create temp log directory
  temp_log_dir <- tempdir()
  old_log_dir <- Sys.getenv("R_LOG_DIR")
  Sys.setenv(R_LOG_DIR = temp_log_dir)

  # Log a message
  log_message("Test message", level = "INFO", script_name = "test_script")

  # Check log file was created
  log_file <- file.path(temp_log_dir, paste0("test_script_", format(Sys.Date(), "%Y%m%d"), ".log"))
  expect_true(file.exists(log_file))

  # Read log file
  log_content <- readLines(log_file)
  expect_true(length(log_content) >= 1)

  # Verify format
  expect_match(log_content[1], "\\[.*\\] \\[INFO\\] \\[test_script\\] Test message")

  # Clean up
  unlink(log_file)
  Sys.setenv(R_LOG_DIR = old_log_dir)
})

test_that("log_message handles different log levels", {
  temp_log_dir <- tempdir()
  old_log_dir <- Sys.getenv("R_LOG_DIR")
  Sys.setenv(R_LOG_DIR = temp_log_dir)

  levels <- c("INFO", "WARNING", "ERROR", "CRITICAL")

  for (level in levels) {
    log_message(paste("Test", level), level = level, script_name = "level_test")

    log_file <- file.path(temp_log_dir, paste0("level_test_", format(Sys.Date(), "%Y%m%d"), ".log"))
    log_content <- readLines(log_file)

    expect_match(tail(log_content, 1), sprintf("\\[%s\\]", level))
  }

  # Clean up
  unlink(file.path(temp_log_dir, paste0("level_test_", format(Sys.Date(), "%Y%m%d"), ".log")))
  Sys.setenv(R_LOG_DIR = old_log_dir)
})

test_that("log_message appends to existing log file", {
  temp_log_dir <- tempdir()
  old_log_dir <- Sys.getenv("R_LOG_DIR")
  Sys.setenv(R_LOG_DIR = temp_log_dir)

  # Log multiple messages
  log_message("Message 1", script_name = "append_test")
  log_message("Message 2", script_name = "append_test")
  log_message("Message 3", script_name = "append_test")

  log_file <- file.path(temp_log_dir, paste0("append_test_", format(Sys.Date(), "%Y%m%d"), ".log"))
  log_content <- readLines(log_file)

  expect_equal(length(log_content), 3)
  expect_match(log_content[1], "Message 1")
  expect_match(log_content[2], "Message 2")
  expect_match(log_content[3], "Message 3")

  # Clean up
  unlink(log_file)
  Sys.setenv(R_LOG_DIR = old_log_dir)
})

# ==============================================================================
# TEST: create_notification()
# ==============================================================================

test_that("create_notification creates notification JSON file", {
  temp_log_dir <- tempdir()
  old_log_dir <- Sys.getenv("R_LOG_DIR")
  Sys.setenv(R_LOG_DIR = temp_log_dir)

  # Re-source to pick up new LOG_DIR
  source("R/utils/error_handling.R")

  # Create notification
  create_notification("Test error", "ERROR", "test_script")

  notification_file <- file.path(temp_log_dir, "alerts.json")
  expect_true(file.exists(notification_file))

  # Read notification
  notifications <- fromJSON(notification_file, simplifyVector = FALSE)
  expect_equal(length(notifications), 1)

  # Verify structure
  notif <- notifications[[1]]
  expect_equal(notif$message, "Test error")
  expect_equal(notif$level, "ERROR")
  expect_equal(notif$script, "test_script")
  expect_equal(notif$status, "unread")

  # Clean up
  unlink(notification_file)
  Sys.setenv(R_LOG_DIR = old_log_dir)
})

test_that("create_notification limits to 100 notifications", {
  temp_log_dir <- tempdir()
  old_log_dir <- Sys.getenv("R_LOG_DIR")
  Sys.setenv(R_LOG_DIR = temp_log_dir)

  source("R/utils/error_handling.R")

  # Create 110 notifications
  for (i in 1:110) {
    create_notification(paste("Error", i), "ERROR", "test_script")
  }

  notification_file <- file.path(temp_log_dir, "alerts.json")
  notifications <- fromJSON(notification_file, simplifyVector = FALSE)

  # Should only keep last 100
  expect_equal(length(notifications), 100)

  # First notification should be #11 (1-10 were pruned)
  expect_match(notifications[[1]]$message, "Error 11")

  # Last notification should be #110
  expect_match(notifications[[100]]$message, "Error 110")

  # Clean up
  unlink(notification_file)
  Sys.setenv(R_LOG_DIR = old_log_dir)
})

# ==============================================================================
# INTEGRATION TEST: Error logging workflow
# ==============================================================================

test_that("ERROR level creates both log and notification", {
  temp_log_dir <- tempdir()
  old_log_dir <- Sys.getenv("R_LOG_DIR")
  Sys.setenv(R_LOG_DIR = temp_log_dir)

  source("R/utils/error_handling.R")

  # Log an error
  log_message("Critical database failure", level = "ERROR", script_name = "integration_test")

  # Check log file
  log_file <- file.path(temp_log_dir, paste0("integration_test_", format(Sys.Date(), "%Y%m%d"), ".log"))
  expect_true(file.exists(log_file))

  log_content <- readLines(log_file)
  expect_match(log_content[1], "\\[ERROR\\]")
  expect_match(log_content[1], "Critical database failure")

  # Check error log
  error_log <- file.path(temp_log_dir, "errors.log")
  expect_true(file.exists(error_log))

  error_content <- readLines(error_log)
  expect_match(error_content[length(error_content)], "Critical database failure")

  # Check notification
  notification_file <- file.path(temp_log_dir, "alerts.json")
  expect_true(file.exists(notification_file))

  notifications <- fromJSON(notification_file, simplifyVector = FALSE)
  expect_equal(notifications[[length(notifications)]]$message, "Critical database failure")

  # Clean up
  unlink(log_file)
  unlink(error_log)
  unlink(notification_file)
  Sys.setenv(R_LOG_DIR = old_log_dir)
})

test_that("INFO level does not create notification", {
  temp_log_dir <- tempdir()
  old_log_dir <- Sys.getenv("R_LOG_DIR")
  Sys.setenv(R_LOG_DIR = temp_log_dir)

  source("R/utils/error_handling.R")

  # Remove any existing notification file
  notification_file <- file.path(temp_log_dir, "alerts.json")
  if (file.exists(notification_file)) {
    unlink(notification_file)
  }

  # Log INFO message
  log_message("Informational message", level = "INFO", script_name = "info_test")

  # Check notification file was NOT created
  expect_false(file.exists(notification_file))

  # Clean up
  log_file <- file.path(temp_log_dir, paste0("info_test_", format(Sys.Date(), "%Y%m%d"), ".log"))
  unlink(log_file)
  Sys.setenv(R_LOG_DIR = old_log_dir)
})

# ==============================================================================
# RUN ALL TESTS
# ==============================================================================

cat("\n")
cat(strrep("=", 80) %+% "\n")
cat("R/utils/error_handling.R TEST SUITE\n")
cat(strrep("=", 80) %+% "\n")

test_results <- test_file("R/tests/test_error_handling.R", reporter = "summary")

cat("\n")
cat("✅ Test suite completed\n")
cat(sprintf("   Passed: %d\n", sum(test_results$passed)))
cat(sprintf("   Failed: %d\n", sum(test_results$failed)))
cat(sprintf("   Skipped: %d\n", sum(test_results$skipped)))
cat("\n")
