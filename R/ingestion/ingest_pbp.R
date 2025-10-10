# R/ingestion/ingest_pbp.R
# Ingest NFL play-by-play data from nflfastR (1999-2024)
# Compatible with comprehensive schema (000_complete_schema.sql)

library(nflfastR)
library(dplyr)
library(DBI)
library(RPostgres)

# Load helper functions
source("R/utils/db_helpers.R")

# Connect to database
con <- get_db_connection()

# Verify plays table exists (should be created by 000_complete_schema.sql)
table_exists <- dbExistsTable(con, "plays")
if (!table_exists) {
  stop("plays table does not exist! Run db/migrations/000_complete_schema.sql first")
}

# Idempotent load: truncate and refill
dbExecute(con, "TRUNCATE TABLE plays CASCADE;")

cat("Loading ALL play-by-play data from 1999-2024...\n")
cat("This will take approximately 10-15 minutes...\n\n")

for (yr in 1999:2024) {
  cat("Loading play-by-play data for", yr, "...\n")

  # Load play-by-play from nflfastR
  pbp_raw <- nflfastR::load_pbp(yr)

  # Select only columns that exist in our schema
  # Keep numeric types as-is (nflfastR already returns 0/1 for binary columns)
  pbp_clean <- pbp_raw |>
    transmute(
      game_id,
      play_id,
      posteam,
      defteam,
      quarter = qtr,
      time_seconds = half_seconds_remaining,
      down,
      ydstogo,
      epa,
      pass,    # Keep as numeric (0/1)
      rush     # Keep as numeric (0/1)
    )

  # Convert to data.frame for DBI compatibility
  pbp_clean <- as.data.frame(pbp_clean)

  # Write to database
  DBI::dbWriteTable(con, "plays", pbp_clean, append = TRUE, row.names = FALSE)

  # Progress update
  total_plays <- as.integer(dbGetQuery(con, "SELECT COUNT(*) FROM plays")[[1]])
  cat("  ->", nrow(pbp_clean), "plays inserted (cumulative:", format(total_plays, big.mark = ","), ")\n")
}

cat("\n")
cat("Play-by-play ingestion complete!\n")

# Final verification
final_count <- as.integer(dbGetQuery(con, "SELECT COUNT(*) as count FROM plays")$count)
cat(sprintf("Total plays in database: %s\n", format(final_count, big.mark = ",")))

dbDisconnect(con)
