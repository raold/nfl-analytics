#!/usr/bin/env Rscript
# Setup testthat framework for R testing
# Run this once to set up the testing infrastructure

# Install testthat if not already installed
if (!require(testthat, quietly = TRUE)) {
  cat("Installing testthat package...\n")
  install.packages("testthat", repos = "https://cran.r-project.org")
}

library(testthat)

# Create test directory structure
test_dirs <- c(
  "tests/testthat",
  "tests/fixtures"
)

for (dir in test_dirs) {
  if (!dir.exists(dir)) {
    dir.create(dir, recursive = TRUE)
    cat("Created directory:", dir, "\n")
  }
}

# Create main test runner
cat("Creating test runner...\n")

runner_content <- '# Test runner for NFL Analytics R code
# Run all tests with: Rscript tests/testthat.R

library(testthat)
library(nflreadr)
library(dplyr)
library(DBI)

# Source utility functions
source("R/utils/error_handling.R")

# Run all tests
test_results <- test_dir("tests/testthat")

# Print summary
print(test_results)

# Exit with appropriate code
if (length(test_results$failures) > 0 || length(test_results$errors) > 0) {
  quit(status = 1)
} else {
  quit(status = 0)
}
'

writeLines(runner_content, "tests/testthat.R")
cat("Created tests/testthat.R\n")

# Create helper file for test utilities
helper_content <- '# Test helper utilities
# Automatically loaded by testthat

# Mock database connection for testing
get_test_db_connection <- function() {
  # Use test database or mock connection
  if (Sys.getenv("TEST_DB") == "true") {
    # Real test database
    DBI::dbConnect(
      RPostgres::Postgres(),
      host = Sys.getenv("POSTGRES_HOST", "localhost"),
      port = as.integer(Sys.getenv("POSTGRES_PORT", "5544")),
      dbname = Sys.getenv("POSTGRES_TEST_DB", "testdb"),
      user = Sys.getenv("POSTGRES_USER", "dro"),
      password = Sys.getenv("POSTGRES_PASSWORD", "sicillionbillions")
    )
  } else {
    # Return NULL for mock testing
    NULL
  }
}

# Test data fixtures
get_test_game_data <- function() {
  data.frame(
    game_id = c("2024_01_SF_DAL", "2024_01_BUF_MIA"),
    season = c(2024, 2024),
    week = c(1, 1),
    home_team = c("DAL", "MIA"),
    away_team = c("SF", "BUF"),
    home_score = c(21, 24),
    away_score = c(17, 31),
    stringsAsFactors = FALSE
  )
}

get_test_play_data <- function() {
  data.frame(
    game_id = rep("2024_01_SF_DAL", 3),
    play_id = 1:3,
    posteam = c("SF", "SF", "DAL"),
    quarter = c(1, 1, 1),
    down = c(1, 2, 1),
    ydstogo = c(10, 7, 10),
    yards_gained = c(3, 12, -2),
    play_type = c("run", "pass", "run"),
    epa = c(-0.5, 1.2, -1.1),
    stringsAsFactors = FALSE
  )
}

# Temporary test directory
get_test_dir <- function() {
  test_dir <- file.path(tempdir(), "nfl_test", format(Sys.time(), "%Y%m%d_%H%M%S"))
  dir.create(test_dir, recursive = TRUE, showWarnings = FALSE)
  test_dir
}

# Clean up test files
cleanup_test_dir <- function(test_dir) {
  if (dir.exists(test_dir)) {
    unlink(test_dir, recursive = TRUE)
  }
}
'

writeLines(helper_content, "tests/testthat/helper_test_utils.R")
cat("Created tests/testthat/helper_test_utils.R\n")

# Create example test files
cat("\nCreating example test files...\n")

# Test for error handling utilities
test_error_handling <- '# Tests for error handling utilities

test_that("log_message creates log files", {
  # Set up test directory
  test_dir <- get_test_dir()
  old_log_dir <- Sys.getenv("R_LOG_DIR")
  Sys.setenv(R_LOG_DIR = test_dir)

  # Test logging
  log_message("Test message", level = "INFO")

  # Check that log file was created
  log_files <- list.files(test_dir, pattern = "\\\\.log$")
  expect_true(length(log_files) > 0)

  # Clean up
  Sys.setenv(R_LOG_DIR = old_log_dir)
  cleanup_test_dir(test_dir)
})

test_that("safe_execute handles errors correctly", {
  # Test successful execution
  result <- safe_execute(
    expr = { 1 + 1 },
    error_message = "Math failed"
  )
  expect_equal(result, 2)

  # Test error handling
  expect_error(
    safe_execute(
      expr = { stop("Test error") },
      error_message = "Expected failure"
    )
  )
})

test_that("retry_operation retries on failure", {
  # Counter to track attempts
  attempts <- 0

  # Function that fails twice then succeeds
  test_fn <- function() {
    attempts <<- attempts + 1
    if (attempts < 3) {
      stop("Temporary failure")
    }
    return("success")
  }

  # Test retry logic
  result <- retry_operation(
    expr = test_fn(),
    max_attempts = 3,
    delay = 0.1
  )

  expect_equal(result, "success")
  expect_equal(attempts, 3)
})

test_that("validate_data checks data integrity", {
  # Test with valid data
  valid_data <- data.frame(
    col1 = 1:10,
    col2 = letters[1:10]
  )

  expect_true(
    validate_data(
      data = valid_data,
      expected_cols = c("col1", "col2"),
      min_rows = 5,
      max_rows = 20
    )
  )

  # Test with missing columns
  expect_error(
    validate_data(
      data = valid_data,
      expected_cols = c("col1", "col2", "col3")
    )
  )

  # Test with too few rows
  expect_error(
    validate_data(
      data = valid_data,
      min_rows = 20
    )
  )
})
'

writeLines(test_error_handling, "tests/testthat/test_error_handling.R")
cat("Created tests/testthat/test_error_handling.R\n")

# Test for data ingestion
test_ingestion <- '# Tests for data ingestion functions

test_that("get_current_nfl_season returns correct season", {
  # Mock September (should return current year)
  mockery::stub(get_current_nfl_season, "Sys.Date", as.Date("2024-09-15"))
  expect_equal(get_current_nfl_season(), 2024)

  # Mock January (should return current year - 1)
  mockery::stub(get_current_nfl_season, "Sys.Date", as.Date("2024-01-15"))
  expect_equal(get_current_nfl_season(), 2023)

  # Mock June (offseason, should return current year - 1)
  mockery::stub(get_current_nfl_season, "Sys.Date", as.Date("2024-06-15"))
  expect_equal(get_current_nfl_season(), 2023)
})

test_that("games data transformation works correctly", {
  # Get test data
  test_games <- get_test_game_data()

  # Check required columns exist
  expect_true("game_id" %in% names(test_games))
  expect_true("season" %in% names(test_games))
  expect_true("week" %in% names(test_games))
  expect_true("home_team" %in% names(test_games))
  expect_true("away_team" %in% names(test_games))

  # Check data types
  expect_type(test_games$season, "double")
  expect_type(test_games$week, "double")
  expect_type(test_games$home_score, "double")

  # Check data values
  expect_true(all(test_games$season >= 1999))
  expect_true(all(test_games$week >= 1 & test_games$week <= 22))
})

test_that("plays data has valid structure", {
  # Get test data
  test_plays <- get_test_play_data()

  # Check required columns
  expect_true("game_id" %in% names(test_plays))
  expect_true("play_id" %in% names(test_plays))
  expect_true("quarter" %in% names(test_plays))

  # Check quarter values (1-6 for overtime)
  expect_true(all(test_plays$quarter >= 1 & test_plays$quarter <= 6))

  # Check down values (1-4)
  expect_true(all(test_plays$down >= 1 & test_plays$down <= 4))

  # Check EPA is numeric
  expect_type(test_plays$epa, "double")
})
'

writeLines(test_ingestion, "tests/testthat/test_ingestion.R")
cat("Created tests/testthat/test_ingestion.R\n")

# Test for batch updates
test_batch_updates <- '# Tests for batch update performance improvements

test_that("batch updates are faster than row-by-row", {
  skip_if(is.null(get_test_db_connection()), "Test database not available")

  conn <- get_test_db_connection()
  on.exit(DBI::dbDisconnect(conn))

  # Create test table
  test_table <- "test_performance_comparison"
  dbExecute(conn, sprintf("DROP TABLE IF EXISTS %s", test_table))
  dbExecute(conn, sprintf("
    CREATE TABLE %s (
      id INTEGER PRIMARY KEY,
      value TEXT
    )", test_table))

  # Prepare test data
  n_rows <- 100
  test_data <- data.frame(
    id = 1:n_rows,
    value = paste("value", 1:n_rows),
    stringsAsFactors = FALSE
  )

  # Measure row-by-row insertion time
  start_row_by_row <- Sys.time()
  for (i in 1:nrow(test_data)) {
    dbExecute(conn, sprintf("
      INSERT INTO %s (id, value) VALUES ($1, $2)
    ", test_table), params = list(test_data$id[i], test_data$value[i]))
  }
  time_row_by_row <- difftime(Sys.time(), start_row_by_row, units = "secs")

  # Clear table
  dbExecute(conn, sprintf("DELETE FROM %s", test_table))

  # Measure batch insertion time
  start_batch <- Sys.time()
  dbWriteTable(conn, test_table, test_data, append = TRUE, row.names = FALSE)
  time_batch <- difftime(Sys.time(), start_batch, units = "secs")

  # Batch should be significantly faster
  expect_lt(time_batch, time_row_by_row)

  # Clean up
  dbExecute(conn, sprintf("DROP TABLE IF EXISTS %s", test_table))
})

test_that("batch update with temp table works", {
  skip_if(is.null(get_test_db_connection()), "Test database not available")

  conn <- get_test_db_connection()
  on.exit(DBI::dbDisconnect(conn))

  # Create test tables
  dbExecute(conn, "DROP TABLE IF EXISTS test_main")
  dbExecute(conn, "CREATE TABLE test_main (id INT PRIMARY KEY, value TEXT)")

  # Insert initial data
  initial_data <- data.frame(
    id = 1:10,
    value = paste("old", 1:10),
    stringsAsFactors = FALSE
  )
  dbWriteTable(conn, "test_main", initial_data, append = TRUE, row.names = FALSE)

  # Prepare update data
  update_data <- data.frame(
    id = 1:10,
    value = paste("new", 1:10),
    stringsAsFactors = FALSE
  )

  # Perform batch update using temp table
  temp_table <- "temp_update_data"
  dbWriteTable(conn, temp_table, update_data, temporary = FALSE, overwrite = TRUE)

  rows_updated <- dbExecute(conn, sprintf("
    UPDATE test_main m
    SET value = t.value
    FROM %s t
    WHERE m.id = t.id
  ", temp_table))

  expect_equal(rows_updated, 10)

  # Verify updates
  result <- dbGetQuery(conn, "SELECT value FROM test_main WHERE id = 1")
  expect_equal(result$value, "new 1")

  # Clean up
  dbExecute(conn, sprintf("DROP TABLE IF EXISTS %s", temp_table))
  dbExecute(conn, "DROP TABLE IF EXISTS test_main")
})
'

writeLines(test_batch_updates, "tests/testthat/test_batch_updates.R")
cat("Created tests/testthat/test_batch_updates.R\n")

cat("\n✅ testthat framework setup complete!\n")
cat("\nUsage:\n")
cat("  Run all tests: Rscript tests/testthat.R\n")
cat("  Run specific test file: Rscript -e \"testthat::test_file('tests/testthat/test_error_handling.R')\"\n")
cat("  Run tests in RStudio: testthat::test_dir('tests/testthat')\n")
cat("\nTest files created:\n")
cat("  - tests/testthat/test_error_handling.R\n")
cat("  - tests/testthat/test_ingestion.R\n")
cat("  - tests/testthat/test_batch_updates.R\n")
cat("\nAdd more tests by creating files matching pattern: tests/testthat/test_*.R\n")