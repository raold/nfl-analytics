#!/usr/bin/env Rscript
#
# Ingest NFL injury reports from nflreadr into the `injuries` table.
#
# Data source: nflreadr::load_injuries()
# Seasons: 2009-present (official injury reports)
# Frequency: Weekly practice participation and game status
#
# Usage:
#   Rscript --vanilla data/ingest_injuries.R [--seasons 2020,2021,2022]
#
# Environment variables (override docker-compose defaults):
#   POSTGRES_HOST (default: localhost)
#   POSTGRES_PORT (default: 5544)
#   POSTGRES_DB   (default: devdb01)
#   POSTGRES_USER (default: dro)
#   POSTGRES_PASSWORD (default: sicillionbillions)

suppressPackageStartupMessages({
  library(nflreadr)
  library(DBI)
  library(RPostgres)
  library(dplyr)
  library(lubridate)
})

# Parse command-line arguments
args <- commandArgs(trailingOnly = TRUE)
seasons <- NULL

if (length(args) > 0) {
  for (arg in args) {
    if (grepl("^--seasons=", arg)) {
      seasons_str <- sub("^--seasons=", "", arg)
      seasons <- as.integer(strsplit(seasons_str, ",")[[1]])
    }
  }
}

# Default to all available seasons if not specified
if (is.null(seasons)) {
  seasons <- 2009:year(Sys.Date())
}

cat("Ingesting injury reports for seasons:", paste(seasons, collapse = ", "), "\n")

# Database connection
host <- Sys.getenv("POSTGRES_HOST", "localhost")
port <- as.integer(Sys.getenv("POSTGRES_PORT", "5544"))
dbname <- Sys.getenv("POSTGRES_DB", "devdb01")
user <- Sys.getenv("POSTGRES_USER", "dro")
password <- Sys.getenv("POSTGRES_PASSWORD", "sicillionbillions")

con <- dbConnect(
  RPostgres::Postgres(),
  host = host,
  port = port,
  dbname = dbname,
  user = user,
  password = password
)

# Create injuries table if not exists
create_table_sql <- "
CREATE TABLE IF NOT EXISTS injuries (
    season INTEGER NOT NULL,
    game_type VARCHAR(10) NOT NULL,
    team VARCHAR(3) NOT NULL,
    week INTEGER NOT NULL,
    gsis_id VARCHAR(20) NOT NULL,
    position VARCHAR(10),
    full_name VARCHAR(100),
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    report_primary_injury TEXT,
    report_secondary_injury TEXT,
    report_status VARCHAR(50),
    practice_primary_injury TEXT,
    practice_secondary_injury TEXT,
    practice_status VARCHAR(50),
    date_modified TIMESTAMPTZ,
    PRIMARY KEY (season, game_type, team, week, gsis_id)
)
"

# Execute table creation
dbExecute(con, create_table_sql)

# Create indexes separately
index_sqls <- c(
  "CREATE INDEX IF NOT EXISTS idx_injuries_season_week ON injuries(season, week)",
  "CREATE INDEX IF NOT EXISTS idx_injuries_gsis_id ON injuries(gsis_id)",
  "CREATE INDEX IF NOT EXISTS idx_injuries_team_week ON injuries(team, week, season)",
  "CREATE INDEX IF NOT EXISTS idx_injuries_report_status ON injuries(report_status) WHERE report_status IS NOT NULL"
)

for (sql in index_sqls) {
  dbExecute(con, sql)
}

cat("✓ injuries table schema verified\n")

# Load injury data from nflreadr
cat("Loading injury data from nflreadr...\n")
injuries_raw <- load_injuries(seasons = seasons)

cat("Loaded", nrow(injuries_raw), "injury records\n")

# Clean and prepare data
injuries <- injuries_raw %>%
  mutate(
    # Convert date_modified to proper timestamp
    date_modified = as.POSIXct(date_modified, tz = "UTC"),
    # Normalize team abbreviations (nflreadr uses 3-letter codes)
    team = toupper(team),
    # Convert empty strings to NULL for better SQL handling
    report_primary_injury = na_if(report_primary_injury, ""),
    report_secondary_injury = na_if(report_secondary_injury, ""),
    report_status = na_if(report_status, ""),
    practice_primary_injury = na_if(practice_primary_injury, ""),
    practice_secondary_injury = na_if(practice_secondary_injury, ""),
    practice_status = na_if(practice_status, "")
  ) %>%
  select(
    season, game_type, team, week, gsis_id, position,
    full_name, first_name, last_name,
    report_primary_injury, report_secondary_injury, report_status,
    practice_primary_injury, practice_secondary_injury, practice_status,
    date_modified
  ) %>%
  # Deduplicate by primary key (keep most recent date_modified)
  arrange(desc(date_modified)) %>%
  distinct(season, game_type, team, week, gsis_id, .keep_all = TRUE)

cat("After deduplication:", nrow(injuries), "unique injury records\n")

# Upsert strategy: DELETE existing records for these seasons, then INSERT
cat("Deleting existing records for seasons:", paste(seasons, collapse = ", "), "\n")
delete_sql <- sprintf(
  "DELETE FROM injuries WHERE season IN (%s)",
  paste(seasons, collapse = ", ")
)
dbExecute(con, delete_sql)

# Write to database
cat("Writing", nrow(injuries), "injury records to database...\n")
dbWriteTable(con, "injuries", injuries, append = TRUE, row.names = FALSE)

cat("✓ Injury data ingestion complete!\n")

# Summary stats
summary <- injuries %>%
  group_by(season, report_status) %>%
  summarize(count = n(), .groups = "drop") %>%
  arrange(season, desc(count))

cat("\nInjury Report Summary by Status:\n")
print(summary, n = 50)

# Disconnect
dbDisconnect(con)
