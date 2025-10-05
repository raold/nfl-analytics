# R/backfill_rosters.R
# Create players and rosters tables from nflreadr

library(nflreadr)
library(dplyr)
library(DBI)
library(RPostgres)

cat("=== NFL Roster & Player Data Backfill ===\n\n")

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

# Create players table
cat("Creating players table...\n")
dbExecute(con, "DROP TABLE IF EXISTS rosters CASCADE;")
dbExecute(con, "DROP TABLE IF EXISTS players CASCADE;")
dbExecute(con, "
  CREATE TABLE players (
    player_id TEXT PRIMARY KEY,
    player_name TEXT,
    position TEXT,
    height TEXT,
    weight INT,
    college TEXT,
    birth_date DATE,
    rookie_year INT,
    draft_club TEXT,
    draft_number INT,
    headshot_url TEXT,
    status TEXT,
    entry_year INT,
    years_exp INT
  );
")

cat("Creating rosters table...\n")
dbExecute(con, "
  CREATE TABLE rosters (
    season INT,
    week INT,
    team TEXT,
    player_id TEXT,
    position TEXT,
    depth_chart_position TEXT,
    jersey_number INT,
    status TEXT,
    full_name TEXT,
    football_name TEXT,
    PRIMARY KEY (season, week, team, player_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id)
  );
")

cat("Loading rosters from nflreadr for seasons", min(SEASONS), "to", max(SEASONS), "...\n")

# Load rosters for all seasons
all_rosters <- nflreadr::load_rosters(seasons = SEASONS)

cat("Loaded", nrow(all_rosters), "roster-week entries\n\n")

# Extract unique players
cat("Extracting unique players...\n")
players <- all_rosters %>%
  dplyr::filter(!is.na(gsis_id)) %>%  # Filter before grouping - use dplyr:: to avoid stats::filter
  group_by(gsis_id) %>%
  slice(1) %>%  # Take first occurrence of each player
  ungroup() %>%
  transmute(
    player_id = gsis_id,
    player_name = full_name,
    position = position,
    height = height,
    weight = weight,
    college = college,
    birth_date = as.Date(birth_date),
    rookie_year = rookie_year,
    draft_club = draft_club,
    draft_number = draft_number,
    headshot_url = headshot_url,
    status = status,
    entry_year = entry_year,
    years_exp = years_exp
  )

cat("Found", nrow(players), "unique players\n")

# Write players to database
cat("Writing players to database...\n")
dbWriteTable(con, "players", players, append = TRUE, row.names = FALSE)

cat("✓ Wrote", nrow(players), "players\n\n")

# Process rosters by season
cat("Processing rosters by season...\n\n")
for (season in SEASONS) {
  season_rosters <- all_rosters %>%
    dplyr::filter(season == !!season, !is.na(gsis_id), !is.na(week)) %>%  # Filter before transmute - use dplyr::
    transmute(
      season = season,
      week = week,
      team = team,
      player_id = gsis_id,
      position = position,
      depth_chart_position = depth_chart_position,
      jersey_number = jersey_number,
      status = status,
      full_name = full_name,
      football_name = football_name
    ) %>%
    distinct(season, week, team, player_id, .keep_all = TRUE)  # Remove duplicates
  
  if (nrow(season_rosters) == 0) {
    cat(season, ": No data\n")
    next
  }
  
  dbWriteTable(con, "rosters", season_rosters, append = TRUE, row.names = FALSE)
  cat(season, ": Wrote", nrow(season_rosters), "roster entries\n")
}

cat("\n\nCreating indexes...\n")
dbExecute(con, "CREATE INDEX rosters_player_id_idx ON rosters(player_id);")
dbExecute(con, "CREATE INDEX rosters_team_season_idx ON rosters(team, season);")
dbExecute(con, "CREATE INDEX rosters_season_week_idx ON rosters(season, week);")
dbExecute(con, "CREATE INDEX players_position_idx ON players(position);")
dbExecute(con, "CREATE INDEX players_name_idx ON players(player_name);")

cat("✓ Indexes created\n\n")

# Verification
cat("=== Verification ===\n")

player_summary <- dbGetQuery(con, "
  SELECT 
    position,
    COUNT(*) as player_count,
    COUNT(DISTINCT player_name) as unique_names
  FROM players
  WHERE position IN ('QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'CB', 'S')
  GROUP BY position
  ORDER BY player_count DESC;
")

cat("\nPlayers by position:\n")
print(player_summary)

roster_summary <- dbGetQuery(con, "
  SELECT 
    season,
    COUNT(*) as roster_entries,
    COUNT(DISTINCT team) as teams,
    COUNT(DISTINCT player_id) as unique_players
  FROM rosters
  GROUP BY season
  ORDER BY season;
")

cat("\nRosters by season:\n")
print(roster_summary)

# Sample query
cat("\nSample: 2023 Week 1 Kansas City Chiefs roster:\n")
sample_roster <- dbGetQuery(con, "
  SELECT 
    r.player_id,
    p.player_name,
    r.position,
    r.jersey_number,
    r.depth_chart_position
  FROM rosters r
  JOIN players p ON r.player_id = p.player_id
  WHERE r.season = 2023 AND r.week = 1 AND r.team = 'KC'
  ORDER BY r.position, r.depth_chart_position
  LIMIT 10;
")

print(sample_roster)

dbDisconnect(con)

cat("\n✅ Roster backfill complete!\n")
cat("Created tables: players (", nrow(players), "rows), rosters (multiple seasons)\n")
cat("Next step: Rscript R/backfill_game_metadata.R\n")
