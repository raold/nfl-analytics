#!/usr/bin/env Rscript
# Baseline GLM models and calibration diagnostics (logit/probit)

suppressPackageStartupMessages({
  library(dplyr)
  library(DBI)
  library(RPostgres)
  library(ggplot2)
})

#' Fetch modeling frame (stub)
#' Join games with any mart features as desired.
fetch_model_frame <- function() {
  con <- dbConnect(
    RPostgres::Postgres(),
    dbname   = Sys.getenv("POSTGRES_DB", "devdb01"),
    host     = Sys.getenv("POSTGRES_HOST", "localhost"),
    port     = as.integer(Sys.getenv("POSTGRES_PORT", 5544)),
    user     = Sys.getenv("POSTGRES_USER", "dro"),
    password = Sys.getenv("POSTGRES_PASSWORD", "sicillionbillions")
  )
  on.exit(dbDisconnect(con), add = TRUE)
  tbl <- dbReadTable(con, "games") |>
    as_tibble() |>
    transmute(
      season, week, game_id,
      # outcome: home win
      y = as.integer(home_score > away_score),
      spread_close, total_close,
      # placeholder features; extend with mart joins
      fav = ifelse(spread_close < 0, 1, 0)
    ) |>
    tidyr::drop_na(y)
  tbl
}

#' Fit GLM baseline
#' @param df data frame
#' @param link "logit" or "probit"
fit_glm_baseline <- function(df, link = c("logit", "probit")) {
  link <- match.arg(link)
  fam <- if (link == "logit") binomial(link = "logit") else binomial(link = "probit")
  stats::glm(y ~ spread_close + fav, data = df, family = fam)
}

#' Calibration slope/intercept
calibration_slope_intercept <- function(y, p) {
  eps <- 1e-6
  logit <- function(x) log(pmin(1 - eps, pmax(eps, x)) / (1 - pmin(1 - eps, pmax(eps, x))))
  m <- stats::glm(y ~ logit(p), family = binomial())
  tibble::tibble(slope = coef(m)[2], intercept = coef(m)[1])
}

#' Reliability curve bins
reliability_curve <- function(y, p, bins = 10) {
  df <- tibble::tibble(y = y, p = p) |>
    mutate(bin = cut(p, breaks = seq(0, 1, length.out = bins + 1), include.lowest = TRUE)) |>
    group_by(bin) |>
    summarise(pred = mean(p), obs = mean(y), n = dplyr::n(), .groups = "drop")
  df
}

if (identical(environment(), globalenv())) {
  try({
    df <- fetch_model_frame()
    if (nrow(df) > 0) {
      m <- fit_glm_baseline(df, link = "logit")
      p <- stats::predict(m, type = "response")
      print(head(reliability_curve(df$y, p)))
      print(calibration_slope_intercept(df$y, p))
    } else {
      message("No games found in DB for GLM baseline demo")
    }
  })
}

