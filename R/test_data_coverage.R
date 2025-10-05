library(dplyr)
library(RPostgres)
library(DBI)

con <- dbConnect(Postgres(), dbname='devdb01', host='localhost', port=5544, user='dro', password='sicillionbillions')
games <- dbGetQuery(con, 'SELECT season, week, home_score, away_score, spread_close FROM games WHERE season BETWEEN 1999 AND 2024')
dbDisconnect(con)

cat("Raw games:", nrow(games), "\n")

games <- games |> mutate(
  margin = home_score - away_score,
  spread_target = if_else(!is.na(spread_close), as.integer(margin + spread_close > 0), NA_integer_)
) |> dplyr::filter(!is.na(spread_target), !is.na(spread_close))

cat('Games after filtering:', nrow(games), '\n')
print(games |> group_by(season) |> summarise(n=n()))
