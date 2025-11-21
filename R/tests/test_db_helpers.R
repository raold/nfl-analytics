#!/usr/bin/env Rscript
# Unit Tests for R/utils/db_helpers.R
# Uses testthat framework

library(testthat)
library(DBI)

# Source the file under test
source("R/utils/db_helpers.R")

# ==============================================================================
# TEST: convert_logical_to_int()
# ==============================================================================

test_that("convert_logical_to_int converts TRUE to 1 and FALSE to 0", {
  df <- data.frame(
    logical_col = c(TRUE, FALSE, TRUE, FALSE),
    numeric_col = c(1, 2, 3, 4),
    char_col = c("a", "b", "c", "d")
  )

  result <- convert_logical_to_int(df)

  # Check conversion
  expect_type(result$logical_col, "integer")
  expect_equal(result$logical_col, c(1L, 0L, 1L, 0L))

  # Check other columns unchanged
  expect_type(result$numeric_col, "double")
  expect_type(result$char_col, "character")
})

test_that("convert_logical_to_int handles NA values", {
  df <- data.frame(
    logical_col = c(TRUE, FALSE, NA, TRUE)
  )

  result <- convert_logical_to_int(df)

  expect_equal(result$logical_col[1:2], c(1L, 0L))
  expect_true(is.na(result$logical_col[3]))
  expect_equal(result$logical_col[4], 1L)
})

test_that("convert_logical_to_int handles multiple logical columns", {
  df <- data.frame(
    col1 = c(TRUE, FALSE),
    col2 = c(FALSE, TRUE),
    col3 = c(1, 2)
  )

  result <- convert_logical_to_int(df)

  expect_type(result$col1, "integer")
  expect_type(result$col2, "integer")
  expect_type(result$col3, "double") # Should remain unchanged
})

test_that("convert_logical_to_int handles empty data frame", {
  df <- data.frame()

  result <- convert_logical_to_int(df)

  expect_equal(nrow(result), 0)
  expect_equal(ncol(result), 0)
})

test_that("convert_logical_to_int handles data frame with no logical columns", {
  df <- data.frame(
    numeric_col = c(1, 2, 3),
    char_col = c("a", "b", "c")
  )

  result <- convert_logical_to_int(df)

  # Should be unchanged
  expect_equal(result, df)
})

# ==============================================================================
# TEST: get_db_connection()
# ==============================================================================

test_that("get_db_connection returns a valid DBI connection", {
  skip_if_not(Sys.getenv("POSTGRES_PASSWORD") != "", "Database credentials not available")

  conn <- get_db_connection()

  # Check it's a valid connection
  expect_s4_class(conn, "PqConnection")
  expect_true(DBI::dbIsValid(conn))

  # Clean up
  DBI::dbDisconnect(conn)
})

test_that("get_db_connection uses environment variables", {
  # Save original env vars
  orig_host <- Sys.getenv("POSTGRES_HOST")
  orig_port <- Sys.getenv("POSTGRES_PORT")
  orig_db <- Sys.getenv("POSTGRES_DB")
  orig_user <- Sys.getenv("POSTGRES_USER")

  # Set custom env vars
  Sys.setenv(POSTGRES_HOST = "localhost")
  Sys.setenv(POSTGRES_PORT = "5544")
  Sys.setenv(POSTGRES_DB = "devdb01")
  Sys.setenv(POSTGRES_USER = "dro")

  # Test connection uses these values
  conn <- get_db_connection()
  expect_s4_class(conn, "PqConnection")

  # Clean up
  DBI::dbDisconnect(conn)

  # Restore env vars
  Sys.setenv(POSTGRES_HOST = orig_host)
  Sys.setenv(POSTGRES_PORT = orig_port)
  Sys.setenv(POSTGRES_DB = orig_db)
  Sys.setenv(POSTGRES_USER = orig_user)
})

test_that("get_db_connection falls back to defaults when env vars not set", {
  # Save original env vars
  orig_host <- Sys.getenv("POSTGRES_HOST")
  orig_port <- Sys.getenv("POSTGRES_PORT")
  orig_db <- Sys.getenv("POSTGRES_DB")
  orig_user <- Sys.getenv("POSTGRES_USER")

  # Unset env vars
  Sys.unsetenv("POSTGRES_HOST")
  Sys.unsetenv("POSTGRES_PORT")
  Sys.unsetenv("POSTGRES_DB")
  Sys.unsetenv("POSTGRES_USER")

  # Should still work with defaults
  conn <- get_db_connection()
  expect_s4_class(conn, "PqConnection")

  # Clean up
  DBI::dbDisconnect(conn)

  # Restore env vars
  Sys.setenv(POSTGRES_HOST = orig_host)
  Sys.setenv(POSTGRES_PORT = orig_port)
  Sys.setenv(POSTGRES_DB = orig_db)
  Sys.setenv(POSTGRES_USER = orig_user)
})

# ==============================================================================
# INTEGRATION TEST: Full workflow
# ==============================================================================

test_that("convert_logical_to_int + database roundtrip works", {
  skip_if_not(Sys.getenv("POSTGRES_PASSWORD") != "", "Database credentials not available")

  # Create test data with logical columns
  df <- data.frame(
    id = 1:3,
    is_active = c(TRUE, FALSE, TRUE),
    is_verified = c(FALSE, TRUE, FALSE)
  )

  # Convert logicals to integers
  df_converted <- convert_logical_to_int(df)

  # Write to temp table
  conn <- get_db_connection()
  table_name <- paste0("test_", format(Sys.time(), "%Y%m%d%H%M%S"))

  DBI::dbWriteTable(conn, table_name, df_converted, temporary = TRUE)

  # Read back
  df_from_db <- DBI::dbReadTable(conn, table_name)

  # Verify data integrity
  expect_equal(df_from_db$id, df$id)
  expect_equal(df_from_db$is_active, as.integer(df$is_active))
  expect_equal(df_from_db$is_verified, as.integer(df$is_verified))

  # Clean up
  DBI::dbDisconnect(conn)
})

# ==============================================================================
# RUN ALL TESTS
# ==============================================================================

cat("\n")
cat("=" %+% strrep("=", 79) %+% "\n")
cat("R/utils/db_helpers.R TEST SUITE\n")
cat(strrep("=", 80) %+% "\n")

test_results <- test_file("R/tests/test_db_helpers.R", reporter = "summary")

cat("\n")
cat("✅ Test suite completed\n")
cat(sprintf("   Passed: %d\n", sum(test_results$passed)))
cat(sprintf("   Failed: %d\n", sum(test_results$failed)))
cat(sprintf("   Skipped: %d\n", sum(test_results$skipped)))
cat("\n")
