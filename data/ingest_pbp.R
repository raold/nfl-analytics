# R/ingest_pbp.R
library(nflfastR); library(dplyr); library(DBI); library(RPostgres)

con <- dbConnect(RPostgres::Postgres(), dbname="nfl", host="localhost", user="nfl", password="nfl", port=5432)

for (yr in 1999:2024) {
  pbp <- nflfastR::load_pbp(yr) |>
    transmute(game_id, play_id, posteam, defteam,
              quarter = qtr, time_seconds = half_seconds_remaining,
              down, ydstogo, epa, pass = ifelse(pass==1, TRUE, FALSE),
              rush = ifelse(rush==1, TRUE, FALSE))
  DBI::dbWriteTable(con, "plays", pbp, append = TRUE)
}
dbDisconnect(con)