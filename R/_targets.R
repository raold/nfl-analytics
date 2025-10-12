#!/usr/bin/env Rscript
# Targets Pipeline for NFL Analytics
#
# The targets package provides a Make-like pipeline for R that:
#   - Tracks dependencies between analysis steps
#   - Only re-runs changed computations (caching)
#   - Parallelizes independent steps automatically
#   - Provides reproducibility guarantees
#
# This is R's most powerful orchestration tool and demonstrates why R is
# "unbelievably powerful and expressive" for data pipelines.
#
# Usage:
#   targets::tar_make()          # Run full pipeline
#   targets::tar_visnetwork()    # View dependency graph
#   targets::tar_outdated()      # Check what needs updating
#   targets::tar_make_future()   # Run with parallel workers

library(targets)
library(tarchetypes)  # Additional target factories

# Set options
tar_option_set(
  packages = c(
    "dplyr", "tidyr", "DBI", "RPostgres", "nflverse",
    "brms", "loo", "slider", "furrr", "future"
  ),
  format = "qs",  # Fast serialization format
  memory = "transient",  # Free memory after each target
  garbage_collection = TRUE,
  error = "continue"  # Continue after errors
)

# Enable parallel processing
future::plan(future::multisession, workers = 4)

# Source R scripts
source("R/utils/db_helpers.R")
source("R/bayesian_team_ratings_brms.R")

# Define pipeline
list(
  ## ─── Data Ingestion ───────────────────────────────────────────────────────
  tar_target(
    name = db_connection,
    command = {
      DBI::dbConnect(
        RPostgres::Postgres(),
        dbname   = Sys.getenv("POSTGRES_DB", "devdb01"),
        host     = Sys.getenv("POSTGRES_HOST", "localhost"),
        port     = as.integer(Sys.getenv("POSTGRES_PORT", 5544)),
        user     = Sys.getenv("POSTGRES_USER", "dro"),
        password = Sys.getenv("POSTGRES_PASSWORD", "sicillionbillions")
      )
    },
    deployment = "main"
  ),

  tar_target(
    name = games_raw,
    command = {
      query <- "
        SELECT game_id, season, week, home_team, away_team,
               home_score, away_score, spread_close, total_close, kickoff
        FROM games
        WHERE season >= 2015
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
        ORDER BY season, week
      "
      DBI::dbGetQuery(db_connection, query) |> dplyr::as_tibble()
    }
  ),

  tar_target(
    name = plays_raw,
    command = {
      query <- "
        SELECT game_id, posteam, epa, success, air_yards, yards_after_catch,
               cpoe, pass, rush, down, ydstogo
        FROM plays
        WHERE posteam IS NOT NULL
          AND season >= 2015
      "
      DBI::dbGetQuery(db_connection, query) |> dplyr::as_tibble()
    }
  ),

  ## ─── Feature Engineering ──────────────────────────────────────────────────
  tar_target(
    name = team_epa_aggregates,
    command = {
      plays_raw |>
        dplyr::group_by(game_id, posteam) |>
        dplyr::summarize(
          epa_total = sum(epa, na.rm = TRUE),
          epa_pass = sum(epa[pass == 1], na.rm = TRUE),
          epa_rush = sum(epa[rush == 1], na.rm = TRUE),
          success_rate = mean(success, na.rm = TRUE),
          plays = n(),
          .groups = "drop"
        )
    }
  ),

  tar_target(
    name = games_with_epa,
    command = {
      games_raw |>
        dplyr::left_join(
          team_epa_aggregates |> dplyr::rename(home_team = posteam),
          by = c("game_id", "home_team")
        ) |>
        dplyr::rename_with(~paste0("home_", .x), .cols = c(epa_total:plays)) |>
        dplyr::left_join(
          team_epa_aggregates |>
dplyr::rename(away_team = posteam),
          by = c("game_id", "away_team")
        ) |>
        dplyr::rename_with(~paste0("away_", .x), .cols = c(epa_total:plays)) |>
        dplyr::mutate(home_margin = home_score - away_score)
    }
  ),

  tar_target(
    name = rolling_team_stats,
    command = {
      # Use slider for efficient rolling window computations
      compute_rolling_stats_with_slider(games_with_epa)
    }
  ),

  ## ─── Baseline Models ──────────────────────────────────────────────────────
  tar_target(
    name = glm_baseline,
    command = {
      glm(
        home_margin ~ spread_close + home_epa_total + away_epa_total,
        data = games_with_epa,
        family = gaussian()
      )
    }
  ),

  tar_target(
    name = glm_diagnostics,
    command = {
      list(
        r_squared = summary(glm_baseline)$r.squared,
        rmse = sqrt(mean(residuals(glm_baseline)^2)),
        mae = mean(abs(residuals(glm_baseline)))
      )
    }
  ),

  ## ─── Bayesian Models ──────────────────────────────────────────────────────
  tar_target(
    name = brms_basic_model,
    command = {
      fit_brms_model_basic(
        games_with_epa,
        chains = 4,
        iter = 2000,
        cores = 4
      )
    },
    deployment = "main"  # Run on main process (Stan doesn't parallelize well)
  ),

  tar_target(
    name = brms_team_ratings,
    command = {
      extract_team_ratings(brms_basic_model, model_type = "basic")
    }
  ),

  tar_target(
    name = brms_loo_cv,
    command = {
      loo::loo(brms_basic_model)
    }
  ),

  tar_target(
    name = brms_full_model,
    command = {
      fit_brms_model_full(
        games_with_epa,
        chains = 4,
        iter = 2000,
        cores = 4
      )
    },
    deployment = "main"
  ),

  tar_target(
    name = brms_attack_defense_ratings,
    command = {
      extract_team_ratings(brms_full_model, model_type = "full")
    }
  ),

  ## ─── Model Comparison ─────────────────────────────────────────────────────
  tar_target(
    name = model_comparison,
    command = {
      list(
        glm = glm_diagnostics,
        brms_basic_loo = brms_loo_cv$estimates["elpd_loo", "Estimate"],
        brms_full_loo = loo::loo(brms_full_model)$estimates["elpd_loo", "Estimate"]
      )
    }
  ),

  ## ─── Reporting & Visualization ────────────────────────────────────────────
  tar_target(
    name = ratings_plot,
    command = {
      ggplot2::ggplot(brms_team_ratings,
                     ggplot2::aes(x = reorder(team, rating_mean),
                                 y = rating_mean)) +
        ggplot2::geom_point(size = 3) +
        ggplot2::geom_errorbar(ggplot2::aes(ymin = rating_q05, ymax = rating_q95),
                              width = 0.2) +
        ggplot2::coord_flip() +
        ggplot2::labs(
          title = "Bayesian Team Ratings (2015-2024)",
          x = "Team",
          y = "Net Rating (points)",
          caption = "Error bars: 90% credible interval"
        ) +
        ggplot2::theme_minimal()
    }
  ),

  tar_target(
    name = save_ratings_plot,
    command = {
      ggplot2::ggsave(
        "analysis/figures/bayesian/team_ratings_plot.png",
        plot = ratings_plot,
        width = 10,
        height = 12,
        dpi = 300
      )
      "analysis/figures/bayesian/team_ratings_plot.png"
    },
    format = "file"
  ),

  tar_target(
    name = write_ratings_to_db,
    command = {
      DBI::dbExecute(db_connection, "CREATE SCHEMA IF NOT EXISTS mart;")
      DBI::dbExecute(db_connection, "DROP TABLE IF EXISTS mart.bayesian_ratings_targets;")
      DBI::dbWriteTable(
        db_connection,
        DBI::Id(schema = "mart", table = "bayesian_ratings_targets"),
        brms_team_ratings |>
          dplyr::mutate(
            pipeline = "targets",
            updated_at = Sys.time()
          ),
        row.names = FALSE
      )
      nrow(brms_team_ratings)
    }
  ),

  ## ─── Validation & Testing ─────────────────────────────────────────────────
  tar_target(
    name = holdout_validation,
    command = {
      # Split 2024 data for validation
      train <- games_with_epa |> dplyr::filter(season < 2024)
      test <- games_with_epa |> dplyr::filter(season == 2024)

      # Fit on training data
      fit_train <- glm(
        home_margin ~ spread_close + home_epa_total + away_epa_total,
        data = train,
        family = gaussian()
      )

      # Predict on test data
      preds <- predict(fit_train, newdata = test)
      actuals <- test$home_margin

      list(
        rmse = sqrt(mean((actuals - preds)^2, na.rm = TRUE)),
        mae = mean(abs(actuals - preds), na.rm = TRUE),
        cor = cor(actuals, preds, use = "complete.obs")
      )
    }
  ),

  ## ─── Summary Report ───────────────────────────────────────────────────────
  tar_target(
    name = pipeline_summary,
    command = {
      list(
        games_processed = nrow(games_with_epa),
        teams = length(unique(c(games_raw$home_team, games_raw$away_team))),
        seasons = paste(min(games_raw$season), max(games_raw$season), sep = "-"),
        glm_rmse = glm_diagnostics$rmse,
        brms_elpd = brms_loo_cv$estimates["elpd_loo", "Estimate"],
        holdout_rmse = holdout_validation$rmse,
        top_team = brms_team_ratings$team[1],
        top_rating = brms_team_ratings$rating_mean[1],
        pipeline_completed_at = Sys.time()
      )
    }
  ),

  tar_target(
    name = print_summary,
    command = {
      cat("\n═══════════════════════════════════════════════════════════\n")
      cat("         NFL ANALYTICS TARGETS PIPELINE SUMMARY\n")
      cat("═══════════════════════════════════════════════════════════\n\n")
      cat(sprintf("Games Processed:    %d\n", pipeline_summary$games_processed))
      cat(sprintf("Teams:              %d\n", pipeline_summary$teams))
      cat(sprintf("Seasons:            %s\n", pipeline_summary$seasons))
      cat(sprintf("\nModel Performance:\n"))
      cat(sprintf("  GLM RMSE:         %.2f points\n", pipeline_summary$glm_rmse))
      cat(sprintf("  BRMS LOO ELPD:    %.1f\n", pipeline_summary$brms_elpd))
      cat(sprintf("  Holdout RMSE:     %.2f points\n", pipeline_summary$holdout_rmse))
      cat(sprintf("\nTop Rated Team:     %s (%.2f points)\n",
                  pipeline_summary$top_team,
                  pipeline_summary$top_rating))
      cat(sprintf("\nCompleted:          %s\n", pipeline_summary$pipeline_completed_at))
      cat("═══════════════════════════════════════════════════════════\n\n")
      invisible(pipeline_summary)
    }
  )
)

# Helper function for rolling stats with slider
compute_rolling_stats_with_slider <- function(games_df) {
  # This will be implemented in the slider features file
  # Placeholder for now
  games_df
}
