#!/usr/bin/env Rscript
#
# Test Script: Verify Hierarchical v2 handles new player_season factor levels
#
# This script tests that the fixed prediction code properly handles
# 2025 player_season combinations that weren't present during training (2015-2024).
#

library(tidyverse)
library(DBI)
library(RPostgres)
library(brms)

cat("======================================================================\n")
cat("TEST: Hierarchical v2 New Factor Level Handling\n")
cat("======================================================================\n\n")

# ============================================================================
# Load a small sample of 2025 data
# ============================================================================

cat("Step 1: Loading 2025 test sample...\n")

conn <- dbConnect(
  Postgres(),
  host = "localhost",
  port = 5544,
  dbname = "devdb01",
  user = "dro",
  password = Sys.getenv("DB_PASSWORD", "sicillionbillions")
)

query_2025_sample <- "
SELECT
  pgs.player_id,
  pgs.player_display_name as player_name,
  pgs.season,
  pgs.week,
  pgs.stat_yards as yards,
  pgs.stat_attempts as attempts,

  g.home_team,
  g.away_team,
  ph.current_team as team,
  CASE WHEN ph.current_team = g.home_team THEN 1 ELSE 0 END as is_home,

  CASE
    WHEN ph.current_team = g.home_team THEN CASE WHEN g.spread_close < 0 THEN 1 ELSE 0 END
    ELSE CASE WHEN g.spread_close > 0 THEN 1 ELSE 0 END
  END as is_favored,
  ABS(g.spread_close) as spread_abs,
  g.total_close,

  COALESCE(g.roof, 'unknown') as roof,
  CASE WHEN g.roof IN ('outdoors', 'open') AND CAST(g.temp AS NUMERIC) < 40 THEN 1 ELSE 0 END as is_bad_weather,
  CASE WHEN g.roof = 'dome' THEN 1 ELSE 0 END as is_dome,

  ph.years_exp,
  ph.position_group,
  CASE
    WHEN ph.current_team = g.home_team THEN g.away_team
    ELSE g.home_team
  END as opponent

FROM mart.player_game_stats pgs
JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
LEFT JOIN games g ON
  g.season = pgs.season
  AND g.week = pgs.week
  AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
WHERE pgs.stat_category = 'passing'
  AND pgs.season = 2025
  AND pgs.week = 1
  AND pgs.stat_attempts >= 10
  AND pgs.stat_yards IS NOT NULL
  AND ph.position_group = 'QB'
ORDER BY pgs.player_id
LIMIT 3
"

test_sample <- dbGetQuery(conn, query_2025_sample)

# Feature engineering
test_sample <- test_sample %>%
  filter(!is.na(total_close), !is.na(years_exp)) %>%
  mutate(
    log_yards = log(yards),
    log_attempts = log(attempts),
    player_season = paste0(player_id, "_", season) # Creates 2025 combinations
  )

cat(sprintf("✓ Loaded %d test samples\n", nrow(test_sample)))
cat("Sample player_season values:\n")
print(test_sample$player_season)
cat("\n")

# ============================================================================
# Load model and test prediction
# ============================================================================

cat("Step 2: Loading Hierarchical v2 model...\n")
model <- readRDS("models/bayesian/passing_yards_hierarchical_v2.rds")
cat("✓ Model loaded\n\n")

cat("Step 3: Testing prediction with allow_new_levels=TRUE...\n")

# This should work now (previously would fail with factor level error)
tryCatch(
  {
    pred_draws <- posterior_predict(
      model,
      newdata = test_sample,
      allow_new_levels = TRUE,
      sample_new_levels = "gaussian"
    )

    # Calculate predictions
    predictions <- test_sample %>%
      mutate(
        pred_yards = exp(colMeans(pred_draws)),
        pred_lower_90 = exp(apply(pred_draws, 2, quantile, probs = 0.05)),
        pred_upper_90 = exp(apply(pred_draws, 2, quantile, probs = 0.95)),
        actual_yards = yards
      )

    cat("✅ SUCCESS: Predictions generated without factor level errors!\n\n")

    cat("Prediction results:\n")
    print(predictions %>%
      select(player_name, season, week, actual_yards, pred_yards, pred_lower_90, pred_upper_90))

    cat("\n")
    cat("======================================================================\n")
    cat("✅ TEST PASSED\n")
    cat("======================================================================\n\n")
    cat("The model now correctly handles new 2025 player_season combinations\n")
    cat("that weren't present during training (2015-2024).\n\n")
    cat("Key fix: Using actual season in player_season feature engineering\n")
    cat("  OLD: player_season = paste0(player_id, '_', 2024)  # WRONG\n")
    cat("  NEW: player_season = paste0(player_id, '_', season)  # CORRECT\n\n")
    cat("Combined with allow_new_levels=TRUE and sample_new_levels='gaussian',\n")
    cat("the model uses population-level parameters for new factor levels.\n")
    cat("This is the correct behavior for true holdout validation.\n\n")
  },
  error = function(e) {
    cat("❌ FAILED: ", e$message, "\n\n")
    cat("The prediction still fails. This suggests the fix may not be complete.\n")
    cat("Check that allow_new_levels=TRUE is properly set in all predict calls.\n\n")
  }
)

dbDisconnect(conn)
