# R/ingest_pbp.R
library(nflfastR); library(dplyr); library(DBI); library(RPostgres)

# Align DSN with environment (compose defaults)
con <- dbConnect(
  RPostgres::Postgres(),
  dbname   = Sys.getenv("POSTGRES_DB", "devdb01"),
  host     = Sys.getenv("POSTGRES_HOST", "localhost"),
  user     = Sys.getenv("POSTGRES_USER", "dro"),
  password = Sys.getenv("POSTGRES_PASSWORD", "sicillionbillions"),
  port     = as.integer(Sys.getenv("POSTGRES_PORT", 5544))
)

# Ensure plays table exists
dbExecute(con, "CREATE TABLE IF NOT EXISTS plays (
  game_id text, play_id bigint, posteam text, defteam text,
  quarter int, time_seconds int, down int, ydstogo int,
  epa double precision, pass boolean, rush boolean,
  primary key(game_id, play_id)
);")

# Idempotent load: truncate and refill (simple baseline)
dbExecute(con, "TRUNCATE TABLE plays;")
cat("Loading ALL play-by-play data from 1999-2024...\n")
for (yr in 1999:2024) {
  cat("Loading play-by-play data for", yr, "...\n")
  pbp <- nflfastR::load_pbp(yr) |>
    transmute(game_id, play_id, posteam, defteam,
              quarter = qtr, time_seconds = half_seconds_remaining,
              down, ydstogo, epa,
              pass = as.logical(pass),
              rush = as.logical(rush))
  DBI::dbWriteTable(con, "plays", pbp, append = TRUE)
  cat("  ->", nrow(pbp), "plays inserted (cumulative:", 
      dbGetQuery(con, "SELECT COUNT(*) FROM plays")[[1]], ")\n")
}
cat("Play-by-play ingestion complete!\n")
dbDisconnect(con)
