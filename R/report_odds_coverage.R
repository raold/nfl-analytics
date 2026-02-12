#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(DBI)
  library(RPostgres)
  library(dplyr)
  library(readr)
})

con <- dbConnect(
  RPostgres::Postgres(),
  dbname   = Sys.getenv("POSTGRES_DB", "devdb01"),
  host     = Sys.getenv("POSTGRES_HOST", "localhost"),
  port     = as.integer(Sys.getenv("POSTGRES_PORT", 5544)),
  user     = Sys.getenv("POSTGRES_USER", "dro"),
  password = Sys.getenv("POSTGRES_PASSWORD", "sicillionbillions")
)

coverage <- tbl(con, "games") |>
  group_by(season) |>
  summarise(
    games = n(),
    spread = mean(!is.na(spread_close)),
    total = mean(!is.na(total_close)),
    home_ml = mean(!is.na(home_moneyline)),
    away_ml = mean(!is.na(away_moneyline)),
    home_spread_odds = mean(!is.na(home_spread_odds)),
    away_spread_odds = mean(!is.na(away_spread_odds)),
    over = mean(!is.na(over_odds)),
    under = mean(!is.na(under_odds)),
    .groups = "drop"
  ) |>
  collect() |>
  arrange(season)

dir.create("data/raw", showWarnings = FALSE, recursive = TRUE)
output_path <- file.path("data", "raw", "odds_coverage_by_season.csv")
readr::write_csv(coverage, output_path)

missing_moneyline <- coverage |>
  filter(home_ml < 1 | away_ml < 1)

if (nrow(missing_moneyline) > 0) {
  message("WARNING: Moneyline coverage incomplete for seasons: ", paste(missing_moneyline$season, collapse = ", "))
}

dbDisconnect(con)
