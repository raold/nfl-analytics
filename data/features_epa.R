# R/features_epa.R
library(DBI); library(RPostgres); library(dplyr); library(lubridate)

con <- dbConnect(
  RPostgres::Postgres(),
  dbname   = Sys.getenv("POSTGRES_DB", "devdb01"),
  host     = Sys.getenv("POSTGRES_HOST", "localhost"),
  user     = Sys.getenv("POSTGRES_USER", "dro"),
  password = Sys.getenv("POSTGRES_PASSWORD", "sicillionbillions"),
  port     = as.integer(Sys.getenv("POSTGRES_PORT", 5544))
)

pbp <- tbl(con, "plays")
agg <- pbp |>
  group_by(game_id, posteam) |>
  summarise(epa_sum = sum(epa, na.rm=TRUE),
            plays = n(),
            epa_mean = mean(epa, na.rm=TRUE),
            .groups = "drop")

g <- tbl(con, "games") |>
  select(game_id, season, week, home_team, away_team, home_score, away_score, spread_close, total_close) |>
  collect()

# Write to mart.team_epa so materialized view can pick it up
dbExecute(con, "CREATE SCHEMA IF NOT EXISTS mart;")
dbExecute(con, "CREATE TABLE IF NOT EXISTS mart.team_epa (
  game_id text,
  posteam text,
  plays int,
  epa_sum double precision,
  epa_mean double precision,
  explosive_pass double precision,
  explosive_rush double precision,
  primary key (game_id, posteam)
);")

team_epa <- agg |> collect()
DBI::dbWithTransaction(con, {
  dbExecute(con, "DELETE FROM mart.team_epa;")
  dbWriteTable(con, DBI::Id(schema="mart", table="team_epa"), team_epa, append = TRUE, row.names = FALSE)
})

dbDisconnect(con)
