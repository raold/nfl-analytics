# R/backfill_game_metadata.R
# Add stadium, roof, surface, and game-level statistics

library(nflreadr)
library(dplyr)
library(DBI)
library(RPostgres)

cat("=== NFL Game Metadata Backfill ===\n\n")

# Database connection
con <- dbConnect(
  RPostgres::Postgres(),
  host = Sys.getenv("POSTGRES_HOST", "localhost"),
  port = as.integer(Sys.getenv("POSTGRES_PORT", "5544")),
  dbname = Sys.getenv("POSTGRES_DB", "devdb01"),
  user = Sys.getenv("POSTGRES_USER", "dro"),
  password = Sys.getenv("POSTGRES_PASSWORD", "sicillionbillions")
)

# Add columns to games table
cat("Adding metadata columns to games table...\n")
new_columns <- list(
  "stadium TEXT",
  "roof TEXT",
  "surface TEXT",
  "temp TEXT",
  "wind TEXT",
  "away_rest INT",
  "home_rest INT",
  "away_qb_id TEXT",
  "home_qb_id TEXT",
  "away_qb_name TEXT",
  "home_qb_name TEXT",
  "away_coach TEXT",
  "home_coach TEXT",
  "referee TEXT",
  "stadium_id TEXT",
  "game_type TEXT",
  "overtime INT",
  "home_timeouts_remaining INT",
  "away_timeouts_remaining INT",
  "home_turnovers INT",
  "away_turnovers INT",
  "home_penalties INT",
  "away_penalties INT",
  "home_penalty_yards INT",
  "away_penalty_yards INT",
  "home_time_of_possession TEXT",
  "away_time_of_possession TEXT"
)

for (col_def in new_columns) {
  tryCatch({
    dbExecute(con, sprintf("ALTER TABLE games ADD COLUMN IF NOT EXISTS %s;", col_def))
  }, error = function(e) {
    cat("  Column already exists or error:", e$message, "\n")
  })
}

cat("✓ Added/verified", length(new_columns), "columns\n\n")

# Load schedules with stadium metadata
cat("Loading schedules from nflreadr...\n")
schedules <- nflreadr::load_schedules(seasons = 1999:2024)

cat("Loaded", nrow(schedules), "games\n\n")

# Update games with stadium/venue info
cat("Updating games with stadium and venue metadata...\n\n")

for (season in 1999:2024) {
  season_sched <- schedules %>%
    filter(season == !!season) %>%
    select(
      game_id, stadium, roof, surface, temp, wind,
      away_rest, home_rest,
      away_qb_id, home_qb_id, away_qb_name, home_qb_name,
      away_coach, home_coach, referee, stadium_id,
      game_type, overtime
    )
  
  if (nrow(season_sched) == 0) {
    cat(season, ": No schedule data\n")
    next
  }
  
  # Update in batches
  updates <- 0
  for (i in 1:nrow(season_sched)) {
    result <- dbExecute(con, "
      UPDATE games 
      SET 
        stadium = $1, 
        roof = $2, 
        surface = $3,
        temp = $4,
        wind = $5,
        away_rest = $6,
        home_rest = $7,
        away_qb_id = $8,
        home_qb_id = $9,
        away_qb_name = $10,
        home_qb_name = $11,
        away_coach = $12,
        home_coach = $13,
        referee = $14,
        stadium_id = $15,
        game_type = $16,
        overtime = $17
      WHERE game_id = $18
    ", params = list(
      season_sched$stadium[i],
      season_sched$roof[i],
      season_sched$surface[i],
      season_sched$temp[i],
      season_sched$wind[i],
      season_sched$away_rest[i],
      season_sched$home_rest[i],
      season_sched$away_qb_id[i],
      season_sched$home_qb_id[i],
      season_sched$away_qb_name[i],
      season_sched$home_qb_name[i],
      season_sched$away_coach[i],
      season_sched$home_coach[i],
      season_sched$referee[i],
      season_sched$stadium_id[i],
      season_sched$game_type[i],
      season_sched$overtime[i],
      season_sched$game_id[i]
    ))
    
    updates <- updates + result
  }
  
  cat(season, ": Updated", updates, "games\n")
}

cat("\n\nCalculating game-level statistics from plays...\n")

# Calculate turnovers
cat("  Calculating turnovers...\n")
dbExecute(con, "
  WITH game_turnovers AS (
    SELECT 
      game_id,
      posteam,
      COUNT(CASE WHEN fumble_lost = 1 THEN 1 END) +
        COUNT(CASE WHEN interception = 1 THEN 1 END) as turnovers
    FROM plays
    WHERE posteam IS NOT NULL
    GROUP BY game_id, posteam
  ),
  home_tos AS (
    SELECT gt.game_id, turnovers as home_turnovers
    FROM game_turnovers gt
    JOIN games g ON gt.game_id = g.game_id AND gt.posteam = g.home_team
  ),
  away_tos AS (
    SELECT gt.game_id, turnovers as away_turnovers
    FROM game_turnovers gt
    JOIN games g ON gt.game_id = g.game_id AND gt.posteam = g.away_team
  )
  UPDATE games g
  SET 
    home_turnovers = h.home_turnovers,
    away_turnovers = a.away_turnovers
  FROM home_tos h
  JOIN away_tos a ON h.game_id = a.game_id
  WHERE g.game_id = h.game_id;
")

# Calculate penalties
cat("  Calculating penalties...\n")
dbExecute(con, "
  WITH game_penalties AS (
    SELECT 
      game_id,
      penalty_team as team,
      COUNT(CASE WHEN penalty = 1 THEN 1 END) as penalties,
      SUM(CASE WHEN penalty = 1 THEN penalty_yards ELSE 0 END) as penalty_yards
    FROM plays
    WHERE penalty_team IS NOT NULL
    GROUP BY game_id, penalty_team
  ),
  home_pens AS (
    SELECT gp.game_id, penalties as home_penalties, penalty_yards as home_penalty_yards
    FROM game_penalties gp
    JOIN games g ON gp.game_id = g.game_id AND gp.team = g.home_team
  ),
  away_pens AS (
    SELECT gp.game_id, penalties as away_penalties, penalty_yards as away_penalty_yards
    FROM game_penalties gp
    JOIN games g ON gp.game_id = g.game_id AND gp.team = g.away_team
  )
  UPDATE games g
  SET 
    home_penalties = h.home_penalties,
    home_penalty_yards = h.home_penalty_yards,
    away_penalties = a.away_penalties,
    away_penalty_yards = a.away_penalty_yards
  FROM home_pens h
  JOIN away_pens a ON h.game_id = a.game_id
  WHERE g.game_id = h.game_id;
")

cat("✓ Calculated turnovers and penalties\n\n")

# Verification
cat("=== Verification ===\n")

roof_summary <- dbGetQuery(con, "
  SELECT 
    roof,
    COUNT(*) as games,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM games), 1) as pct
  FROM games
  WHERE roof IS NOT NULL
  GROUP BY roof
  ORDER BY games DESC;
")

cat("\nGames by roof type:\n")
print(roof_summary)

surface_summary <- dbGetQuery(con, "
  SELECT 
    surface,
    COUNT(*) as games,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM games), 1) as pct
  FROM games
  WHERE surface IS NOT NULL
  GROUP BY surface
  ORDER BY games DESC;
")

cat("\nGames by surface:\n")
print(surface_summary)

qb_summary <- dbGetQuery(con, "
  SELECT 
    COUNT(*) as total_games,
    COUNT(home_qb_name) as has_home_qb,
    COUNT(away_qb_name) as has_away_qb,
    ROUND(100.0 * COUNT(home_qb_name) / COUNT(*), 1) as home_qb_pct,
    ROUND(100.0 * COUNT(away_qb_name) / COUNT(*), 1) as away_qb_pct
  FROM games;
")

cat("\nQB data coverage:\n")
print(qb_summary)

turnover_summary <- dbGetQuery(con, "
  SELECT 
    SUBSTRING(game_id, 1, 4)::int as season,
    COUNT(*) as games,
    COUNT(home_turnovers) as has_turnovers,
    ROUND(AVG(home_turnovers + away_turnovers), 2) as avg_turnovers_per_game
  FROM games
  WHERE home_turnovers IS NOT NULL
  GROUP BY SUBSTRING(game_id, 1, 4)
  ORDER BY season
  LIMIT 10;
")

cat("\nTurnover data (first 10 seasons):\n")
print(turnover_summary)

# Sample enhanced game
cat("\nSample enhanced game (2023 Super Bowl):\n")
sample_game <- dbGetQuery(con, "
  SELECT 
    game_id,
    home_team || ' vs ' || away_team as matchup,
    stadium,
    roof,
    surface,
    temp,
    wind,
    home_qb_name,
    away_qb_name,
    home_coach,
    away_coach,
    home_score,
    away_score,
    home_turnovers,
    away_turnovers
  FROM games
  WHERE game_type = 'SB' AND game_id LIKE '2023%'
  LIMIT 1;
")

if (nrow(sample_game) > 0) {
  print(t(sample_game))  # Transpose for readability
}

dbDisconnect(con)

cat("\n✅ Game metadata backfill complete!\n")
cat("Next steps:\n")
cat("  1. Refresh materialized views: SELECT mart.refresh_game_features();\n")
cat("  2. Update Python harness with new features\n")
cat("  3. Retrain models\n")
