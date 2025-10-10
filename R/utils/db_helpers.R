#!/usr/bin/env Rscript
# Database Helper Functions
# Utilities for NFL analytics database operations

#' Convert logical columns to integers for PostgreSQL compatibility
#'
#' nflfastR returns many binary columns as R logical (TRUE/FALSE)
#' PostgreSQL INTEGER columns require 0/1
#' This function converts all logical columns to integers
#'
#' @param df Data frame with potential logical columns
#' @return Data frame with logical columns converted to integers
convert_logical_to_int <- function(df) {
  logical_cols <- names(df)[sapply(df, is.logical)]

  if (length(logical_cols) > 0) {
    message(sprintf("Converting %d logical columns to integer: %s",
                    length(logical_cols),
                    paste(head(logical_cols, 10), collapse = ", ")))

    for (col in logical_cols) {
      df[[col]] <- as.integer(df[[col]])
    }
  }

  df
}

#' Get database connection
#' @return DBI connection object
get_db_connection <- function() {
  DBI::dbConnect(
    RPostgres::Postgres(),
    host = Sys.getenv("POSTGRES_HOST", "localhost"),
    port = as.integer(Sys.getenv("POSTGRES_PORT", "5544")),
    dbname = Sys.getenv("POSTGRES_DB", "devdb01"),
    user = Sys.getenv("POSTGRES_USER", "dro"),
    password = Sys.getenv("POSTGRES_PASSWORD", "sicillionbillions")
  )
}

TRUE  # Return TRUE to indicate successful loading
