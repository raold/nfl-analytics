#!/usr/bin/env Rscript
# Bayesian Model EV Analysis
#
# Evaluates whether Bayesian hierarchical models can generate betting EV
# and how they compare/complement XGBoost predictions

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(DBI)
  library(RPostgres)
  library(ggplot2)
})

#' Connect to database
get_db_connection <- function() {
  dbConnect(
    RPostgres::Postgres(),
    dbname   = Sys.getenv("POSTGRES_DB", "devdb01"),
    host     = Sys.getenv("POSTGRES_HOST", "localhost"),
    port     = as.integer(Sys.getenv("POSTGRES_PORT", 5544)),
    user     = Sys.getenv("POSTGRES_USER", "dro"),
    password = Sys.getenv("POSTGRES_PASSWORD", "sicillionbillions")
  )
}

#' Fetch Bayesian ratings and recent game results
fetch_ratings_and_results <- function(con) {
  # Get Bayesian ratings
  ratings <- dbGetQuery(con, "
    SELECT team, rating_mean, rating_sd
    FROM mart.bayesian_team_ratings
  ") %>% as_tibble()

  # Get 2024 games with spreads for testing
  games_2024 <- dbGetQuery(con, "
    SELECT
      game_id,
      season,
      week,
      home_team,
      away_team,
      home_score,
      away_score,
      home_score - away_score AS actual_margin,
      spread_close,
      total_close
    FROM games
    WHERE season = 2024
      AND home_score IS NOT NULL
      AND spread_close IS NOT NULL
    ORDER BY week
  ") %>% as_tibble()

  list(ratings = ratings, games = games_2024)
}

#' Generate Bayesian predictions for games
#' Simple model: margin ~ home_rating - away_rating + home_adv
predict_with_bayesian <- function(games, ratings, home_adv = 2.4) {
  games %>%
    left_join(ratings %>% rename(home_rating = rating_mean),
              by = c("home_team" = "team")) %>%
    left_join(ratings %>% rename(away_rating = rating_mean),
              by = c("away_team" = "team")) %>%
    mutate(
      # Bayesian prediction: difference in ratings + home advantage
      bayesian_margin = home_rating - away_rating + home_adv,
      bayesian_prob_home = pnorm(bayesian_margin / 13.5),  # ~13.5 SD from data

      # Prediction error
      pred_error = actual_margin - bayesian_margin,
      abs_error = abs(pred_error),

      # ATS prediction (against spread)
      bayesian_ats_edge = bayesian_margin - spread_close,
      bayesian_covers = case_when(
        actual_margin > spread_close ~ TRUE,   # Home covers
        actual_margin < spread_close ~ FALSE,  # Home doesn't cover
        TRUE ~ NA                              # Push
      ),
      bayesian_pred_covers = bayesian_ats_edge > 0
    )
}

#' Calculate predictive metrics
calculate_metrics <- function(predictions) {
  predictions %>%
    filter(!is.na(bayesian_covers)) %>%
    summarise(
      n_games = n(),
      mae = mean(abs_error, na.rm = TRUE),
      rmse = sqrt(mean(pred_error^2, na.rm = TRUE)),
      correlation = cor(bayesian_margin, actual_margin, use = "complete.obs"),

      # ATS performance
      ats_accuracy = mean(bayesian_covers == bayesian_pred_covers, na.rm = TRUE),
      n_correct = sum(bayesian_covers == bayesian_pred_covers, na.rm = TRUE),

      # Betting simulation (flat stakes)
      ats_win_rate = mean(bayesian_covers[bayesian_pred_covers], na.rm = TRUE),
      n_bets = sum(bayesian_pred_covers, na.rm = TRUE),
      expected_roi = (ats_win_rate - 0.524) * 100  # Need 52.4% to beat vig
    ) %>%
    mutate(
      status = case_when(
        expected_roi > 1 ~ "PROFITABLE",
        expected_roi > 0 ~ "MARGINAL",
        TRUE ~ "UNPROFITABLE"
      )
    )
}

#' Compare Bayesian to market (spread) efficiency
analyze_market_efficiency <- function(predictions) {
  predictions %>%
    mutate(
      market_error = abs(actual_margin - spread_close),
      model_better = abs_error < market_error
    ) %>%
    summarise(
      market_mae = mean(market_error, na.rm = TRUE),
      bayesian_mae = mean(abs_error, na.rm = TRUE),
      improvement = (market_mae - bayesian_mae) / market_mae * 100,
      pct_better_than_market = mean(model_better, na.rm = TRUE) * 100
    )
}

#' Analyze Bayesian model uncertainty for betting
analyze_uncertainty_calibration <- function(predictions, ratings) {
  # High uncertainty games (high combined SD)
  predictions_with_sd <- predictions %>%
    left_join(ratings %>% select(team, rating_sd) %>% rename(home_sd = rating_sd),
              by = c("home_team" = "team")) %>%
    left_join(ratings %>% select(team, rating_sd) %>% rename(away_sd = rating_sd),
              by = c("away_team" = "team")) %>%
    mutate(
      combined_sd = sqrt(home_sd^2 + away_sd^2),
      uncertainty_group = case_when(
        combined_sd < 1.3 ~ "Low Uncertainty",
        combined_sd < 1.5 ~ "Medium Uncertainty",
        TRUE ~ "High Uncertainty"
      )
    )

  # Performance by uncertainty
  uncertainty_analysis <- predictions_with_sd %>%
    filter(!is.na(bayesian_covers)) %>%
    group_by(uncertainty_group) %>%
    summarise(
      n = n(),
      mae = mean(abs_error, na.rm = TRUE),
      ats_accuracy = mean(bayesian_covers == bayesian_pred_covers, na.rm = TRUE),
      .groups = "drop"
    )

  list(
    by_uncertainty = uncertainty_analysis,
    predictions = predictions_with_sd
  )
}

#' Generate report comparing to XGBoost baseline
#' (Assumes XGBoost v2 Brier = 0.1641 as benchmark)
compare_to_xgboost <- function(bayesian_metrics) {
  xgboost_benchmark <- tibble(
    model = "XGBoost v2",
    brier = 0.1641,
    ats_accuracy_est = 0.52,  # Estimated from Brier
    roi_est = 0.0  # Assume break-even
  )

  bayesian_perf <- tibble(
    model = "Bayesian Hierarchical",
    brier = NA,  # Calculate separately if needed
    ats_accuracy_est = bayesian_metrics$ats_accuracy,
    roi_est = bayesian_metrics$expected_roi / 100
  )

  comparison <- bind_rows(xgboost_benchmark, bayesian_perf)

  list(
    comparison = comparison,
    advantage = bayesian_perf$ats_accuracy_est - xgboost_benchmark$ats_accuracy_est,
    verdict = if_else(
      bayesian_perf$roi_est > xgboost_benchmark$roi_est,
      "Bayesian shows promise - worth ensemble",
      "Bayesian underperforms - use for uncertainty only"
    )
  )
}

#' Simulate ensemble: Average Bayesian + XGBoost predictions
#' (Conceptual - would need actual XGBoost predictions)
simulate_ensemble_benefit <- function(predictions) {
  # Simulate XGBoost having 52% ATS accuracy (slightly better than random)
  set.seed(42)

  predictions_ensemble <- predictions %>%
    filter(!is.na(bayesian_covers)) %>%
    mutate(
      # Simulate XGBoost predictions (correlated with Bayesian but not perfectly)
      xgb_margin = bayesian_margin + rnorm(n(), 0, 5),  # Add noise
      xgb_pred_covers = xgb_margin > spread_close,

      # Ensemble: only bet when both agree
      ensemble_bet = bayesian_pred_covers & xgb_pred_covers,
      ensemble_correct = ensemble_bet & bayesian_covers
    )

  ensemble_metrics <- predictions_ensemble %>%
    filter(ensemble_bet) %>%
    summarise(
      n_bets = n(),
      win_rate = mean(bayesian_covers, na.rm = TRUE),
      roi = (win_rate - 0.524) * 100
    )

  list(
    metrics = ensemble_metrics,
    interpretation = sprintf(
      "Ensemble (both agree): %d bets, %.1f%% win rate, %.2f%% ROI",
      ensemble_metrics$n_bets,
      ensemble_metrics$win_rate * 100,
      ensemble_metrics$roi
    )
  )
}

#' Main analysis
main <- function() {
  con <- get_db_connection()
  on.exit(dbDisconnect(con), add = TRUE)

  cat("=== BAYESIAN MODEL EV ANALYSIS ===\n\n")

  # Fetch data
  cat("Fetching Bayesian ratings and 2024 games...\n")
  data <- fetch_ratings_and_results(con)
  cat(sprintf("Loaded %d team ratings and %d games from 2024\n\n",
              nrow(data$ratings), nrow(data$games)))

  # Generate predictions
  cat("Generating Bayesian predictions...\n")
  predictions <- predict_with_bayesian(data$games, data$ratings)

  # Calculate metrics
  cat("\n--- PREDICTIVE PERFORMANCE ---\n")
  metrics <- calculate_metrics(predictions)
  print(metrics, width = 100)

  cat("\n--- MARKET EFFICIENCY COMPARISON ---\n")
  efficiency <- analyze_market_efficiency(predictions)
  print(efficiency, width = 100)

  cat("\n--- UNCERTAINTY CALIBRATION ---\n")
  uncertainty <- analyze_uncertainty_calibration(predictions, data$ratings)
  print(uncertainty$by_uncertainty, width = 100)

  cat("\n--- COMPARISON TO XGBOOST BASELINE ---\n")
  comparison <- compare_to_xgboost(metrics)
  print(comparison$comparison, width = 100)
  cat(sprintf("\nAccuracy advantage: %+.1f percentage points\n",
              comparison$advantage * 100))
  cat(sprintf("Verdict: %s\n", comparison$verdict))

  cat("\n--- ENSEMBLE SIMULATION ---\n")
  ensemble <- simulate_ensemble_benefit(predictions)
  cat(ensemble$interpretation, "\n")

  cat("\n=== FINAL RECOMMENDATION ===\n")
  if (metrics$expected_roi > 0.5) {
    cat("✓ PROMISING: Bayesian models show positive expected ROI\n")
    cat("  Recommend: Include in production ensemble with 15-25% weight\n")
    cat("  Use uncertainty (SD) for position sizing and risk management\n")
  } else if (metrics$expected_roi > -0.5) {
    cat("⚠ MARGINAL: Bayesian models near break-even\n")
    cat("  Recommend: Use for uncertainty quantification, not primary predictions\n")
    cat("  Consider: Ensemble only when both Bayesian + XGBoost strongly agree\n")
  } else {
    cat("✗ UNPROFITABLE: Bayesian models underperform market\n")
    cat("  Recommend: Use for research/calibration only, not production betting\n")
  }

  # Return results
  invisible(list(
    predictions = predictions,
    metrics = metrics,
    efficiency = efficiency,
    uncertainty = uncertainty,
    comparison = comparison,
    ensemble = ensemble
  ))
}

# Run analysis
if (!interactive()) {
  results <- main()
}
