library(dplyr)
library(RPostgres)
library(DBI)
library(purrr)
library(tibble)
library(glue)

con <- dbConnect(Postgres(), dbname='devdb01', host='localhost', port=5544, user='dro', password='sicillionbillions')
games <- dbGetQuery(con, 'SELECT season, week, game_id, home_team, away_team, home_score, away_score, spread_close FROM games WHERE season BETWEEN 1999 AND 2024')
dbDisconnect(con)

games <- games |> mutate(
  margin = home_score - away_score,
  spread_target = if_else(!is.na(spread_close), as.integer(margin + spread_close > 0), NA_integer_)
) |> dplyr::filter(!is.na(spread_target), !is.na(spread_close))

cat("Total games:", nrow(games), "\n\n")

exp_weight <- function(s, t, H) 0.5 ^ ((t - s) / H)

train_glm_weighted <- function(df, train_years, t_eval, H = NULL, recent_only_start = NULL) {
  dtr <- df |> dplyr::filter(season %in% train_years)
  if (!is.null(recent_only_start)) {
    if (max(train_years) >= recent_only_start) {
      dtr <- dtr |> dplyr::filter(season >= recent_only_start)
    }
  }
  if (nrow(dtr) == 0) {
    cat("    No training data\n")
    return(NULL)
  }
  if (is.null(H)) {
    dtr$sw <- 1
  } else {
    dtr$sw <- exp_weight(dtr$season, t_eval, H)
  }
  cat("    Training with", nrow(dtr), "games\n")
  suppressWarnings(glm(spread_target ~ spread_close, data = dtr, family = binomial(), weights = sw))
}

predict_metrics <- function(mod, dte) {
  if (is.null(mod) || is.null(dte) || nrow(dte) == 0) return(NULL)
  p <- as.numeric(predict(mod, dte, type = "response"))
  y <- dte$spread_target
  ll <- - (y * log(p + 1e-15) + (1 - y) * log(1 - p + 1e-15))
  tibble(p = p, y = y, ll = ll)
}

windows <- list(
  list(train = 1999:2010, test = 2011:2014),
  list(train = 2011:2014, test = 2015:2018),
  list(train = 2015:2018, test = 2019:2021),
  list(train = 2019:2021, test = 2022:2024)
)

rolling_metrics <- purrr::map_dfr(windows, function(w) {
  tr_years <- w$train
  te_years <- w$test
  t_eval <- max(te_years)
  dte <- games |> dplyr::filter(season %in% te_years)

  cat("Window:", min(tr_years), "-", max(tr_years), "→", min(te_years), "-", max(te_years), "| Test games:", nrow(dte), "\n")

  if (nrow(dte) == 0) {
    cat("  Skipping: no test data\n")
    return(NULL)
  }

  cat("  Training recent model (H=NULL, recent_start=2015):\n")
  mod_rec <- train_glm_weighted(games, train_years = tr_years, t_eval = t_eval, H = NULL, recent_only_start = 2015)

  cat("  Training decayed model (H=3):\n")
  mod_dec3 <- train_glm_weighted(games, train_years = tr_years, t_eval = t_eval, H = 3)

  cat("  Predicting...\n")
  m_rec <- predict_metrics(mod_rec, dte)
  m_dec3 <- predict_metrics(mod_dec3, dte)

  if (is.null(m_rec) || is.null(m_dec3)) {
    cat("  Skipping: model predictions failed\n")
    return(NULL)
  }

  cat("  Success! Metrics computed.\n\n")

  tibble(
    block = glue("{min(te_years)}–{max(te_years)}"),
    model = c("recent", "decH3"),
    logloss = c(mean(m_rec$ll), mean(m_dec3$ll))
  )
})

cat("\n=== Final rolling_metrics ===\n")
print(rolling_metrics)
cat("\nDimensions:", nrow(rolling_metrics), "rows x", ncol(rolling_metrics), "cols\n")
