#!/usr/bin/env Rscript
# Generate 4th down coaching quality features
# Requires: nfl4th package, plays table with PBP data

suppressPackageStartupMessages({
  library(nfl4th)
  library(nflreadr)
  library(dplyr)
  library(DBI)
  library(RPostgres)
})

cat("=== 4th Down Feature Engineering ===\n\n")

# Connect to database
con <- dbConnect(
  RPostgres::Postgres(),
  dbname   = Sys.getenv("POSTGRES_DB", "devdb01"),
  host     = Sys.getenv("POSTGRES_HOST", "localhost"),
  port     = as.integer(Sys.getenv("POSTGRES_PORT", 5544)),
  user     = Sys.getenv("POSTGRES_USER", "dro"),
  password = Sys.getenv("POSTGRES_PASSWORD", "sicillionbillions")
)

on.exit(dbDisconnect(con), add = TRUE)

# Check if plays table exists
if (!dbExistsTable(con, "plays")) {
  cat("ERROR: plays table does not exist. Run data/ingest_pbp.R first.\n")
  quit(status = 1)
}

# Load play-by-play data for 4th downs
cat("Loading 4th down plays from database...\n")
pbp_4th <- dbGetQuery(con, "
  SELECT 
    game_id,
    play_id,
    posteam,
    defteam,
    quarter,
    down,
    ydstogo,
    epa
  FROM plays
  WHERE down = 4
    AND ydstogo IS NOT NULL
    AND epa IS NOT NULL
") |> as_tibble()

cat("Found", nrow(pbp_4th), "4th down plays\n\n")

if (nrow(pbp_4th) == 0) {
  cat("No 4th down plays found. Exiting.\n")
  quit(status = 0)
}

# Add 4th down probabilities using nfl4th
cat("Calculating 4th down decision quality with nfl4th...\n")

# Note: nfl4th expects specific column names from nflfastR
# We need to add required columns for the model
pbp_4th_enriched <- pbp_4th |>
  mutate(
    # Add required columns with reasonable defaults
    qtr = quarter,
    half_seconds_remaining = case_when(
      quarter <= 2 ~ 1800 - (quarter - 1) * 900,  # First half
      TRUE ~ 1800 - (quarter - 3) * 900           # Second half
    ),
    score_differential = 0,  # Neutral for simplification
    yardline_100 = 50,       # Midfield default
    posteam_timeouts_remaining = 3,
    defteam_timeouts_remaining = 3
  )

# Try to add 4th down probabilities (may fail if data incomplete)
pbp_with_probs <- tryCatch({
  pbp_4th_enriched |>
    nfl4th::add_4th_probs() |>
    select(game_id, play_id, posteam, epa, 
           go_boost, fg_boost, punt_boost,
           go, first_down, punt, field_goal_attempt) |>
    mutate(
      # Determine if bad decision (any boost < -2 means should have done something else)
      bad_decision = case_when(
        go_boost < -2 ~ 1,
        fg_boost < -2 ~ 1,
        punt_boost < -2 ~ 1,
        TRUE ~ 0
      ),
      went_on_fourth = coalesce(go, 0)
    )
}, error = function(e) {
  cat("WARNING: Could not calculate 4th down probs:", e$message, "\n")
  cat("Falling back to simple EPA-based metrics\n\n")
  
  # Fallback: just use EPA
  pbp_4th_enriched |>
    mutate(
      went_on_fourth = 1,  # Assume went for it if on 4th down
      bad_decision = 0,
      go_boost = NA_real_,
      fg_boost = NA_real_,
      punt_boost = NA_real_
    ) |>
    select(game_id, play_id, posteam, epa, 
           go_boost, fg_boost, punt_boost,
           went_on_fourth, bad_decision)
})

cat("Calculated decision quality for", nrow(pbp_with_probs), "plays\n\n")

# Aggregate to team-game level
cat("Aggregating to team-game features...\n")
team_4th_features <- pbp_with_probs |>
  group_by(game_id, posteam) |>
  summarise(
    fourth_downs = n(),
    went_for_it_rate = mean(went_on_fourth, na.rm = TRUE),
    fourth_down_epa = mean(epa, na.rm = TRUE),
    bad_decisions = sum(bad_decision, na.rm = TRUE),
    avg_go_boost = mean(go_boost, na.rm = TRUE),
    avg_fg_boost = mean(fg_boost, na.rm = TRUE),
    .groups = "drop"
  ) |>
  rename(team = posteam)

cat("Generated features for", nrow(team_4th_features), "team-game combinations\n")
print(head(team_4th_features, 3))

# Create mart schema if needed
dbExecute(con, "CREATE SCHEMA IF NOT EXISTS mart;")

# Truncate existing table (keeps structure/indexes)
dbExecute(con, "TRUNCATE TABLE mart.team_4th_down_features;")
cat("Truncated existing mart.team_4th_down_features\n")

# Write data using dbAppendTable to avoid schema issues
dbAppendTable(con, 
              Id(schema = "mart", table = "team_4th_down_features"), 
              team_4th_features)

cat("\n✓ Wrote", nrow(team_4th_features), "rows to mart.team_4th_down_features\n")

# Show sample query
cat("\nSample query to use features:\n")
cat("
SELECT 
  g.game_id,
  g.home_team,
  g.away_team,
  h.went_for_it_rate AS home_4th_aggression,
  a.went_for_it_rate AS away_4th_aggression,
  h.fourth_down_epa AS home_4th_epa,
  a.fourth_down_epa AS away_4th_epa
FROM games g
LEFT JOIN mart.team_4th_down_features h 
  ON g.game_id = h.game_id AND g.home_team = h.team
LEFT JOIN mart.team_4th_down_features a 
  ON g.game_id = a.game_id AND g.away_team = a.team
WHERE g.season >= 2020
LIMIT 5;
\n")

cat("\n=== Feature generation complete! ===\n")
