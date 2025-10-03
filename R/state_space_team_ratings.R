#!/usr/bin/env Rscript
# State-space team ratings (EWMA/RLS/DLM)
#
# Minimal, reproducible scaffold with three methods:
#  - ewma: exponential smoothing of team-specific margins (fast baseline)
#  - rls: recursive least squares with forgetting factor (Kalman-style)
#  - dlm: local-level Kalman filter per team (requires dlm)

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(DBI)
  library(RPostgres)
})
supp_dlmm <- suppressWarnings(requireNamespace("dlm", quietly = TRUE))

#' Fetch games table from Postgres
#' @return tibble with game_id, season, week, home_team, away_team, home_score, away_score
fetch_games <- function() {
  con <- dbConnect(
    RPostgres::Postgres(),
    dbname   = Sys.getenv("POSTGRES_DB", "devdb01"),
    host     = Sys.getenv("POSTGRES_HOST", "localhost"),
    port     = as.integer(Sys.getenv("POSTGRES_PORT", 5544)),
    user     = Sys.getenv("POSTGRES_USER", "dro"),
    password = Sys.getenv("POSTGRES_PASSWORD", "sicillionbillions")
  )
  on.exit(dbDisconnect(con), add = TRUE)
  dbReadTable(con, "games") |>
    as_tibble() |>
    transmute(game_id, season, week, home_team, away_team, home_score, away_score)
}

#' Estimate team strengths via exponential smoothing of margins (stub)
#' @param games tibble from fetch_games()
#' @param half_life half-life in games for EWMA
#' @return tibble(team, season, week, strength)
fit_team_strength_ewma <- function(games, half_life = 4) {
  lambda <- log(2) / half_life
  df <- games |>
    mutate(margin = home_score - away_score) |>
    select(game_id, season, week, home_team, away_team, margin)

  teams <- sort(unique(c(df$home_team, df$away_team)))
  out <- list()
  for (tm in teams) {
    hist <- df |>
      filter(home_team == tm | away_team == tm) |>
      arrange(season, week) |>
      mutate(sign = ifelse(home_team == tm, +1, -1),
             obs = sign * margin)
    s <- 0
    tprev <- NA_integer_
    rows <- vector("list", nrow(hist))
    for (i in seq_len(nrow(hist))) {
      # EWMA in game order; ignore gaps for now (stub)
      s <- (1 - lambda) * s + lambda * hist$obs[i]
      rows[[i]] <- tibble(team = tm,
                          game_id = hist$game_id[i],
                          season = hist$season[i],
                          week = hist$week[i],
                          strength = s)
    }
    out[[tm]] <- bind_rows(rows)
  }
  bind_rows(out) |>
    arrange(team, season, week)
}

#' Convenience runner: fetch, fit, and write to mart.team_epa-like table (stub)
run_state_space <- function(write_to_mart = FALSE, method = c("ewma", "rls", "dlm")) {
  games <- fetch_games()
  method <- match.arg(method)
  ratings <- switch(method,
                    ewma = fit_team_strength_ewma(games),
                    rls  = fit_team_strength_rls(games),
                    dlm  = if (supp_dlmm) fit_team_strength_dlm(games) else fit_team_strength_rls(games))
  if (isTRUE(write_to_mart)) {
    con <- dbConnect(
      RPostgres::Postgres(),
      dbname   = Sys.getenv("POSTGRES_DB", "devdb01"),
      host     = Sys.getenv("POSTGRES_HOST", "localhost"),
      port     = as.integer(Sys.getenv("POSTGRES_PORT", 5544)),
      user     = Sys.getenv("POSTGRES_USER", "dro"),
      password = Sys.getenv("POSTGRES_PASSWORD", "sicillionbillions")
    )
    on.exit(dbDisconnect(con), add = TRUE)
    dbExecute(con, "CREATE SCHEMA IF NOT EXISTS mart;")
    # Persist to mart.team_ratings (separate from mart.team_epa)
    dbExecute(con, "CREATE TABLE IF NOT EXISTS mart.team_ratings (game_id text, posteam text, method text, rating double precision, PRIMARY KEY(game_id, posteam, method))")
    dbExecute(con, "DELETE FROM mart.team_ratings WHERE method = $1", params = list(method))
    to_write <- ratings |>
      transmute(game_id, posteam = team, method = method, rating = strength)
    dbWriteTable(con, DBI::Id(schema = "mart", table = "team_ratings"), to_write, append = TRUE, row.names = FALSE)
  }
  ratings
}

if (identical(environment(), globalenv())) {
  # If executed as a script, just compute and print a head for sanity
  try({
    tbl <- run_state_space(write_to_mart = FALSE, method = "rls")
    print(head(tbl, 10))
  })
}

# --- Kalman-style recursive least squares (RLS) team ratings ---
# Treat per-team rating as a slowly varying parameter and update via RLS with
# forgetting factor, using design x = +1 for home team, -1 for away team.

#' Fit team ratings with recursive least squares (Kalman-style)
#' @param games tibble as from fetch_games()
#' @param lambda forgetting factor in (0,1); smaller -> faster adaptation
#' @return tibble(team, season, week, strength)
fit_team_strength_rls <- function(games, lambda = 0.98) {
  teams <- sort(unique(c(games$home_team, games$away_team)))
  p <- length(teams)
  idx <- stats::setNames(seq_len(p), teams)
  # State theta (p x 1) and covariance P (p x p)
  theta <- rep(0, p)
  P <- diag(1e2, p)  # large prior variance
  rows <- vector("list", nrow(games) * 2)
  rix <- 1L
  games_ord <- games |>
    arrange(season, week)
  for (i in seq_len(nrow(games_ord))) {
    g <- games_ord[i, ]
    # Observation: y = margin
    y <- as.numeric(g$home_score - g$away_score)
    # Design x has +1 at home team, -1 at away team
    x <- rep(0, p)
    x[idx[[g$home_team]]] <- 1
    x[idx[[g$away_team]]] <- -1
    # RLS update with forgetting factor lambda
    # K = P x / (lambda + x' P x)
    Px <- as.numeric(P %*% x)
    denom <- as.numeric(lambda + sum(x * Px))
    K <- Px / denom
    # Innovation
    yhat <- sum(x * theta)
    err <- y - yhat
    theta <- theta + K * err
    # Covariance update
    P <- (P - tcrossprod(K, Px)) / lambda
    # Record both teams
    rows[[rix]] <- tibble::tibble(team = g$home_team, game_id = g$game_id, season = g$season, week = g$week,
                                  strength = theta[idx[[g$home_team]]])
    rix <- rix + 1L
    rows[[rix]] <- tibble::tibble(team = g$away_team, game_id = g$game_id, season = g$season, week = g$week,
                                  strength = theta[idx[[g$away_team]]])
    rix <- rix + 1L
  }
  dplyr::bind_rows(rows) |>
    dplyr::arrange(team, season, week)
}

# Per-team local-level Kalman filter using dlm (if available)
fit_team_strength_dlm <- function(games, v = 9.0, w = 1.0) {
  if (!supp_dlmm) stop("Package 'dlm' not installed")
  df <- games |>
    mutate(margin = home_score - away_score) |>
    select(game_id, season, week, home_team, away_team, margin)
  teams <- sort(unique(c(df$home_team, df$away_team)))
  out <- list()
  for (tm in teams) {
    hist <- df |>
      filter(home_team == tm | away_team == tm) |>
      arrange(season, week) |>
      mutate(sign = ifelse(home_team == tm, +1, -1),
             obs = sign * margin)
    mod <- dlm::dlmModPoly(order = 1, dV = v, dW = w)
    fit <- suppressWarnings(dlm::dlmFilter(hist$obs, mod))
    s <- as.numeric(fit$m[-1])
    out[[tm]] <- tibble(team = tm,
                        game_id = hist$game_id,
                        season = hist$season,
                        week = hist$week,
                        strength = s)
  }
  bind_rows(out) |>
    arrange(team, season, week)
}
