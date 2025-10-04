# R/backfill_pbp_advanced.R
# Backfill advanced play-by-play features from nflfastR
# This adds ~40 critical columns: WP, success rate, air yards, player IDs, etc.

library(nflfastR)
library(dplyr)
library(DBI)
library(RPostgres)

cat("=== NFL Play-by-Play Advanced Features Backfill ===\n\n")

# Database connection
con <- dbConnect(
  RPostgres::Postgres(),
  host = Sys.getenv("POSTGRES_HOST", "localhost"),
  port = as.integer(Sys.getenv("POSTGRES_PORT", "5544")),
  dbname = Sys.getenv("POSTGRES_DB", "devdb01"),
  user = Sys.getenv("POSTGRES_USER", "dro"),
  password = Sys.getenv("POSTGRES_PASSWORD", "sicillionbillions")
)

SEASONS <- 1999:2024

# High-value columns to add
COLUMNS_TO_ADD <- c(
  # Win probability
  "wp", "wpa", "vegas_wp", "vegas_wpa",
  
  # Success & efficiency
  "success", "yards_gained", "first_down", "first_down_pass", "first_down_rush",
  
  # Passing details
  "air_yards", "yards_after_catch", "comp_air_epa", "comp_yac_epa",
  "cpoe", "complete_pass", "incomplete_pass", "interception",
  "pass_length", "pass_location", "qb_hit", "qb_scramble",
  
  # Rushing details
  "run_location", "run_gap",
  
  # Player IDs
  "passer_player_id", "passer_player_name",
  "rusher_player_id", "rusher_player_name",
  "receiver_player_id", "receiver_player_name",
  "tackle_for_loss", "tackler_1_player_id",
  
  # Situational
  "goal_to_go", "shotgun", "no_huddle", "qb_dropback", "qb_kneel", "qb_spike",
  
  # Scoring
  "touchdown", "td_team", "two_point_attempt", "extra_point_result",
  "field_goal_result", "kick_distance",
  
  # Penalties & turnovers
  "sack", "fumble", "fumble_lost", "penalty", "penalty_yards", "penalty_team",
  
  # Score state
  "posteam_score", "defteam_score", "score_differential",
  "posteam_score_post", "defteam_score_post",
  
  # Drive context
  "drive", "fixed_drive", "fixed_drive_result"
)

cat("Will attempt to add", length(COLUMNS_TO_ADD), "columns to plays table\n\n")

# Get current plays table schema
existing_cols <- dbGetQuery(con, "
  SELECT column_name 
  FROM information_schema.columns 
  WHERE table_name = 'plays' AND table_schema = 'public'
")$column_name

cat("Existing plays table has", length(existing_cols), "columns\n")
cat("Columns:", paste(existing_cols, collapse = ", "), "\n\n")

# Process seasons in batches
for (season in SEASONS) {
  cat("\n=== Processing", season, "season ===\n")
  
  # Load full nflfastR PBP data
  pbp <- tryCatch({
    nflfastR::load_pbp(season)
  }, error = function(e) {
    cat("ERROR loading", season, ":", e$message, "\n")
    return(NULL)
  })
  
  if (is.null(pbp) || nrow(pbp) == 0) {
    cat("No data for", season, ", skipping\n")
    next
  }
  
  cat("Loaded", nrow(pbp), "plays from nflfastR\n")
  
  # Select columns that exist in nflfastR data
  available_cols <- intersect(COLUMNS_TO_ADD, names(pbp))
  cat("Found", length(available_cols), "of", length(COLUMNS_TO_ADD), "target columns\n")
  
  # Build subset with core columns + available new columns
  pbp_subset <- pbp %>%
    select(
      game_id, play_id,
      any_of(available_cols)
    )
  
  # Check for plays already in database
  existing_plays <- dbGetQuery(con, sprintf(
    "SELECT game_id, play_id FROM plays WHERE game_id LIKE '%d_%%'",
    season
  ))
  
  cat("Database has", nrow(existing_plays), "existing plays for", season, "\n")
  
  # Add columns to plays table if they don't exist
  for (col in available_cols) {
    if (!col %in% existing_cols) {
      col_type <- if (is.numeric(pbp_subset[[col]])) {
        "DOUBLE PRECISION"
      } else if (is.logical(pbp_subset[[col]])) {
        "BOOLEAN"
      } else {
        "TEXT"
      }
      
      tryCatch({
        dbExecute(con, sprintf(
          "ALTER TABLE plays ADD COLUMN %s %s;",
          col, col_type
        ))
        cat("  Added column:", col, "(", col_type, ")\n")
        existing_cols <- c(existing_cols, col)  # Update tracking
      }, error = function(e) {
        cat("  Column", col, "already exists or error:", e$message, "\n")
      })
    }
  }
  
  # Write to temporary staging table
  dbExecute(con, "DROP TABLE IF EXISTS plays_staging;")
  dbWriteTable(con, "plays_staging", pbp_subset, row.names = FALSE, temporary = TRUE)
  cat("Created staging table with", nrow(pbp_subset), "plays\n")
  
  # Update existing plays with new column values
  update_cols <- setdiff(available_cols, c("game_id", "play_id"))
  
  if (length(update_cols) > 0) {
    update_set <- paste(
      sprintf("%s = COALESCE(plays_staging.%s, plays.%s)", update_cols, update_cols, update_cols),
      collapse = ", "
    )
    
    update_sql <- sprintf("
      UPDATE plays 
      SET %s
      FROM plays_staging
      WHERE plays.game_id = plays_staging.game_id 
        AND plays.play_id = plays_staging.play_id;
    ", update_set)
    
    rows_updated <- dbExecute(con, update_sql)
    cat("Updated", rows_updated, "plays with new feature values\n")
  }
  
  # Clean up
  dbExecute(con, "DROP TABLE IF EXISTS plays_staging;")
  
  cat("Completed", season, "\n")
}

# Verify results
cat("\n\n=== Verification ===\n")
verification <- dbGetQuery(con, "
  SELECT 
    SUBSTRING(game_id, 1, 4)::int as season,
    COUNT(*) as total_plays,
    COUNT(wp) as has_wp,
    COUNT(success) as has_success,
    COUNT(air_yards) as has_air_yards,
    COUNT(passer_player_id) as has_passer_id,
    COUNT(yards_gained) as has_yards_gained
  FROM plays
  GROUP BY SUBSTRING(game_id, 1, 4)
  ORDER BY season;
")

print(verification)

# Show sample of new data
cat("\n\nSample of enhanced play data (2023 season):\n")
sample_data <- dbGetQuery(con, "
  SELECT 
    game_id, play_id, posteam, quarter, down, ydstogo,
    ROUND(wp::numeric, 3) as wp,
    success,
    yards_gained,
    air_yards,
    passer_player_name,
    rusher_player_name
  FROM plays
  WHERE game_id LIKE '2023_%' 
    AND (passer_player_name IS NOT NULL OR rusher_player_name IS NOT NULL)
  LIMIT 5;
")

print(sample_data)

dbDisconnect(con)

cat("\n✅ Advanced PBP backfill complete!\n")
cat("Next steps:\n")
cat("  1. Run: Rscript R/backfill_rosters.R\n")
cat("  2. Run: Rscript R/backfill_game_metadata.R\n")
cat("  3. Refresh materialized views\n")
cat("  4. Update Python harness with new features\n")
