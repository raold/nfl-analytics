#!/usr/bin/env Rscript
# Generate injury load features from nflreadr injury data
# Aggregates to team-week level with position-specific indicators

suppressPackageStartupMessages({
  library(nflreadr)
  library(dplyr)
  library(DBI)
  library(RPostgres)
})

cat("=== Injury Load Feature Engineering ===\n\n")

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

# Load injury data
SEASONS <- 2020:2024

cat("Loading injury data for seasons:", paste(SEASONS, collapse = ", "), "\n")

all_injuries <- list()

for (season in SEASONS) {
  cat("Loading", season, "injuries...")
  
  injuries <- tryCatch({
    nflreadr::load_injuries(season)
  }, error = function(e) {
    cat(" ERROR:", e$message, "\n")
    return(NULL)
  })
  
  if (!is.null(injuries) && nrow(injuries) > 0) {
    all_injuries[[length(all_injuries) + 1]] <- injuries
    cat(" ", nrow(injuries), "records\n")
  } else {
    cat(" no data\n")
  }
}

if (length(all_injuries) == 0) {
  cat("No injury data available. Exiting.\n")
  quit(status = 0)
}

injuries_df <- bind_rows(all_injuries)

cat("\nTotal injury records:", nrow(injuries_df), "\n")
cat("Unique teams:", n_distinct(injuries_df$team), "\n")
cat("Seasons covered:", paste(unique(injuries_df$season), collapse = ", "), "\n\n")

# Show status distribution
cat("Injury status distribution:\n")
status_counts <- injuries_df |>
  count(report_status, sort = TRUE)
print(status_counts)

# Define key positions
KEY_POSITIONS <- c("QB", "LT", "RT", "C", "LG", "RG", "DE", "DT", "LB", "CB", "S")

# Aggregate to team-week level
cat("\nAggregating to team-week features...\n")

injury_features <- injuries_df |>
  mutate(
    # Binary indicators by status
    is_out = report_status %in% c("Out", "IR", "PUP", "Suspended"),
    is_questionable = report_status == "Questionable",
    is_doubtful = report_status == "Doubtful",
    is_key_position = position %in% KEY_POSITIONS,
    is_qb = position == "QB",
    is_oline = position %in% c("LT", "RT", "C", "LG", "RG")
  ) |>
  group_by(season, week, team) |>
  summarise(
    # Overall injury load
    total_injuries = n(),
    players_out = sum(is_out, na.rm = TRUE),
    players_questionable = sum(is_questionable, na.rm = TRUE),
    players_doubtful = sum(is_doubtful, na.rm = TRUE),
    
    # Position-specific
    key_position_out = any(is_key_position & is_out, na.rm = TRUE),
    qb_out = any(is_qb & is_out, na.rm = TRUE),
    oline_injuries = sum(is_oline, na.rm = TRUE),
    
    # Severity index (weighted by status)
    injury_severity_index = sum(
      case_when(
        report_status %in% c("Out", "IR") ~ 3,
        report_status == "Doubtful" ~ 2,
        report_status == "Questionable" ~ 1,
        TRUE ~ 0
      ),
      na.rm = TRUE
    ),
    
    .groups = "drop"
  )

cat("Generated features for", nrow(injury_features), "team-week combinations\n")
print(head(injury_features, 5))

# Create mart schema if needed
dbExecute(con, "CREATE SCHEMA IF NOT EXISTS mart;")

# Truncate existing table (keeps structure/indexes)
dbExecute(con, "TRUNCATE TABLE mart.team_injury_load;")
cat("Truncated existing mart.team_injury_load\n")

# Write data using dbAppendTable
dbAppendTable(con, 
              Id(schema = "mart", table = "team_injury_load"), 
              injury_features)

cat("\n✓ Wrote", nrow(injury_features), "rows to mart.team_injury_load\n")

# Summary statistics
cat("\nSummary statistics:\n")
summary_stats <- injury_features |>
  summarise(
    avg_injuries_per_team = mean(total_injuries, na.rm = TRUE),
    avg_players_out = mean(players_out, na.rm = TRUE),
    pct_qb_out = mean(qb_out, na.rm = TRUE) * 100,
    pct_key_position_out = mean(key_position_out, na.rm = TRUE) * 100,
    avg_severity_index = mean(injury_severity_index, na.rm = TRUE)
  )
print(summary_stats)

# Show sample query
cat("\nSample query to use features:\n")
cat("
SELECT 
  g.game_id,
  g.season,
  g.week,
  g.home_team,
  g.away_team,
  hi.players_out AS home_players_out,
  ai.players_out AS away_players_out,
  hi.qb_out AS home_qb_out,
  ai.qb_out AS away_qb_out,
  hi.injury_severity_index AS home_injury_severity,
  ai.injury_severity_index AS away_injury_severity
FROM games g
LEFT JOIN mart.team_injury_load hi 
  ON g.home_team = hi.team AND g.season = hi.season AND g.week = hi.week
LEFT JOIN mart.team_injury_load ai 
  ON g.away_team = ai.team AND g.season = ai.season AND g.week = ai.week
WHERE g.season >= 2020
LIMIT 10;
\n")

cat("\n=== Injury load features complete! ===\n")
