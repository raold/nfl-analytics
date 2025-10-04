# R/ingest_schedules.R
library(dplyr)
library(DBI)
library(RPostgres)
library(lubridate)

# connect to the DB – read from environment for portability
con <- dbConnect(
  RPostgres::Postgres(),
  dbname   = Sys.getenv("POSTGRES_DB", "devdb01"),
  host     = Sys.getenv("POSTGRES_HOST", "localhost"),
  user     = Sys.getenv("POSTGRES_USER", "dro"),
  password = Sys.getenv("POSTGRES_PASSWORD", "sicillionbillions"),
  port     = as.integer(Sys.getenv("POSTGRES_PORT", 5544))
)

# ensure games table exists
dbExecute(con, "
  CREATE TABLE IF NOT EXISTS games (
    game_id text PRIMARY KEY,
    season int,
    week int,
    home_team text,
    away_team text,
    kickoff timestamptz,
    spread_close real,
    total_close real,
    home_score int,
    away_score int,
    home_moneyline real,
    away_moneyline real,
    home_spread_odds real,
    away_spread_odds real,
    over_odds real,
    under_odds real
  );
")

snapshot_override <- Sys.getenv("NFLVERSE_SNAPSHOT_PATH", unset = "")

if (nzchar(snapshot_override) && file.exists(snapshot_override)) {
  schedules_raw <- readRDS(snapshot_override)
} else if (file.exists("data/raw/nflverse_schedules_1999_2024.rds")) {
  schedules_raw <- readRDS("data/raw/nflverse_schedules_1999_2024.rds")
} else {
  if (requireNamespace("nflreadr", quietly = TRUE)) {
    schedules_raw <- nflreadr::load_schedules(seasons = 1999:2024)
  } else {
    stop("nflreadr not installed and no local snapshot available")
  }
}

sched <- schedules_raw |>
  transmute(
    game_id,
    season,
    week,
    home_team,
    away_team,
    # convert gameday (YYYY-MM-DD) to a timestamp; use gametime if needed
    kickoff      = ymd(gameday),
    spread_close = spread_line,
    total_close  = total_line,
    home_score,
    away_score,
    home_moneyline,
    away_moneyline,
    home_spread_odds,
    away_spread_odds,
    over_odds,
    under_odds
  ) |>
  distinct()

# Ensure plain data.frame for DBI compatibility
sched <- as.data.frame(sched)

# replace existing contents so reruns stay idempotent
dbWithTransaction(con, {
  dbExecute(con, "TRUNCATE TABLE games;")
  dbWriteTable(con, "games", sched, append = TRUE, row.names = FALSE)
})
dbDisconnect(con)
