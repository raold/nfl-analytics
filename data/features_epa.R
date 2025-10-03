# R/features_epa.R
library(DBI); library(RPostgres); library(dplyr); library(lubridate)

con <- dbConnect(
  RPostgres::Postgres(),
  dbname   = "devdb01",
  host     = "localhost",
  user     = "dro",
  password = "sicillionbillions",
  port     = 5544
)

pbp <- tbl(con, "plays")
agg <- pbp |>
  group_by(game_id, posteam) |>
  summarise(epa_off = mean(epa, na.rm=TRUE),
            pass_rate = mean(pass, na.rm=TRUE),
            rush_rate = mean(rush, na.rm=TRUE))

g <- tbl(con, "games") |>
  select(game_id, season, week, home_team, away_team, home_score, away_score, spread_close, total_close) |>
  collect()

feat <- g |>
  left_join(agg |> collect(), by=c("game_id"="game_id")) |>
  group_by(game_id) |>
  mutate(team = ifelse(row_number()==1, home_team, away_team)) |> ungroup()
