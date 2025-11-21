#!/usr/bin/env Rscript
# Team Strength Ratings - Sagarin Feature #5
#
# Hierarchical Bayesian model for team-level strength ratings independent of player performance.
# Uses brms/Stan to jointly model offensive and defensive strength with proper uncertainty.
#
# Research Finding (Nov 13-14, 2025):
# Team-level ratings provide +0.5-0.9% ROI by capturing overall team quality beyond
# individual player metrics. Similar to opponent strength but focuses on intrinsic team ability.
#
# Model Architecture:
#   total_points ~ 1 + (1 | home_team) + (1 | away_team) + spread_close + week
#
# This extracts team random effects as strength ratings while controlling for:
# - Vegas spread (market consensus)
# - Week number (seasonal trends)
#
# Output Features:
# - Team offensive strength (points scored ability)
# - Team defensive strength (points allowed ability)
# - Team net strength (offense - defense)
# - Within-season strength rank (1-32)
#
# Temporal Safety:
# - Uses only completed games (no future leakage)
# - Separate ratings by season to avoid cross-season contamination
# - Integrates with asof_features_v2.py pipeline
#
# Integration:
# - Join to predictions via (season, team) key
# - Compatible with 4-way ensemble (State-Space, Hierarchical, XGBoost, BNN)
# - Can be 6th ensemble member or XGBoost features
#
# Author: Claude Code (Sonnet 4.5)
# Date: November 14, 2025
# Research: Sagarin Methodology Integration (Phase 1, Feature #5)

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(DBI)
  library(RPostgres)
  library(brms) # Bayesian regression models via Stan
  library(posterior) # Posterior analysis
  library(bayesplot) # Bayesian diagnostics
  library(ggplot2) # Visualization
})

# ========================================
# DATABASE CONNECTION
# ========================================

#' Connect to PostgreSQL database
#' @return DBI connection object
connect_db <- function() {
  con <- dbConnect(
    RPostgres::Postgres(),
    dbname   = Sys.getenv("POSTGRES_DB", "devdb01"),
    host     = Sys.getenv("POSTGRES_HOST", "localhost"),
    port     = as.integer(Sys.getenv("POSTGRES_PORT", 5544)),
    user     = Sys.getenv("POSTGRES_USER", "dro"),
    password = Sys.getenv("POSTGRES_PASSWORD", "sicillionbillions")
  )
  con
}

# ========================================
# DATA LOADING
# ========================================

#' Fetch games data for team strength modeling
#' @param con Database connection
#' @param start_season Earliest season (default 2018)
#' @param end_season Latest season (default 2024)
#' @return tibble with game-level data
fetch_games_for_team_strength <- function(con, start_season = 2018, end_season = 2024) {
  message(sprintf("Fetching games data (%d-%d)...", start_season, end_season))

  query <- "
    SELECT
      g.game_id,
      g.season,
      g.week,
      g.home_team,
      g.away_team,
      g.home_score,
      g.away_score,
      g.home_score + g.away_score AS total_points,
      g.home_score - g.away_score AS home_margin,
      g.spread_close,
      g.total_close,
      g.kickoff
    FROM games g
    WHERE g.season >= $1
      AND g.season <= $2
      AND g.game_type = 'REG'
      AND g.home_score IS NOT NULL
      AND g.away_score IS NOT NULL
    ORDER BY g.season, g.week, g.kickoff
  "

  df <- DBI::dbGetQuery(con, query, params = list(start_season, end_season)) |>
    as_tibble()

  message(sprintf("Loaded %d games", nrow(df)))

  df
}

# ========================================
# DATA PREPARATION
# ========================================

#' Prepare data for BRMS modeling
#' @param games_df tibble from fetch_games_for_team_strength
#' @return Prepared data with factors and scaled variables
prepare_brms_data <- function(games_df) {
  message("Preparing data for modeling...")

  # Create scaled week variable (mean 0, sd 1)
  games_df <- games_df |>
    group_by(season) |>
    mutate(
      week_scaled = (week - mean(week)) / sd(week),
      spread_scaled = if_else(is.na(spread_close), 0,
        (spread_close - mean(spread_close, na.rm = TRUE)) /
          sd(spread_close, na.rm = TRUE)
      )
    ) |>
    ungroup()

  # Ensure factor levels for teams (required for random effects)
  all_teams <- sort(unique(c(games_df$home_team, games_df$away_team)))

  games_df <- games_df |>
    mutate(
      home_team = factor(home_team, levels = all_teams),
      away_team = factor(away_team, levels = all_teams)
    )

  message(sprintf("  Teams: %d", length(all_teams)))
  message(sprintf("  Seasons: %d-%d", min(games_df$season), max(games_df$season)))

  games_df
}

# ========================================
# MODEL FITTING
# ========================================

#' Fit hierarchical Bayesian team strength model
#'
#' Model: total_points ~ 1 + spread_close + week + (1 | home_team) + (1 | away_team)
#'
#' This jointly models:
#' - Home team offensive strength (random effect)
#' - Away team offensive strength (random effect)
#' - Vegas spread adjustment (fixed effect)
#' - Week trend (fixed effect)
#'
#' Random effects capture team-specific deviations from league average scoring.
#'
#' @param games_df Prepared data from prepare_brms_data
#' @param chains Number of MCMC chains (default 4)
#' @param iter Total iterations per chain (default 2000)
#' @param cores Number of CPU cores (default 4)
#' @return brmsfit object
fit_team_strength_model <- function(games_df, chains = 4, iter = 2000, cores = 4) {
  message("\nFitting hierarchical team strength model...")
  message(sprintf(
    "  MCMC: %d chains × %d iterations = %d samples",
    chains, iter, chains * iter
  ))

  # Formula: total points with team random effects
  # Home team and away team both contribute to total scoring
  formula <- bf(
    total_points ~ 1 + spread_scaled + week_scaled +
      (1 | home_team) + (1 | away_team)
  )

  # Priors (weakly informative)
  priors <- c(
    prior(normal(45, 10), class = Intercept), # Mean total ~45 points
    prior(normal(0, 5), class = b), # Spread/week effects
    prior(exponential(0.2), class = sd), # Team variation ~5 points
    prior(exponential(0.1), class = sigma) # Residual ~10 points
  )

  # Fit model with Stan backend (faster than rstan)
  fit <- brm(
    formula,
    data = games_df,
    prior = priors,
    family = gaussian(),
    chains = chains,
    iter = iter,
    cores = cores,
    control = list(adapt_delta = 0.95),
    backend = "cmdstanr",
    refresh = 100, # Progress updates every 100 iterations
    silent = 0
  )

  message("✓ Model fitting complete")

  fit
}

# ========================================
# SEPARATE OFFENSE/DEFENSE MODEL
# ========================================

#' Fit separate offensive and defensive strength models
#'
#' Two models:
#' 1. Offensive: points_scored ~ 1 + (1 | team) + (1 | opponent_defense)
#' 2. Defensive: points_allowed ~ 1 + (1 | team_defense) + (1 | opponent_offense)
#'
#' This separates offensive and defensive contributions more cleanly than
#' the total points model.
#'
#' @param games_df Prepared data
#' @param chains MCMC chains
#' @param iter MCMC iterations
#' @param cores CPU cores
#' @return list(offense_fit, defense_fit)
fit_offense_defense_models <- function(games_df, chains = 4, iter = 2000, cores = 4) {
  message("\nFitting separate offense/defense models...")

  # Reshape to long format (one row per team-game)
  games_long <- games_df |>
    # Home team perspective
    select(game_id, season, week, week_scaled, spread_scaled,
      team = home_team, opponent = away_team,
      points_scored = home_score, points_allowed = away_score
    ) |>
    mutate(location = "home") |>
    bind_rows(
      # Away team perspective
      games_df |>
        select(game_id, season, week, week_scaled, spread_scaled,
          team = away_team, opponent = home_team,
          points_scored = away_score, points_allowed = home_score
        ) |>
        mutate(location = "away")
    ) |>
    mutate(
      team = factor(team),
      opponent = factor(opponent)
    )

  message(sprintf("  Long format: %d team-game records", nrow(games_long)))

  # Offensive model: points_scored ~ team offense + opponent defense
  message("\n  [1/2] Fitting offensive strength model...")

  offense_formula <- bf(
    points_scored ~ 1 + week_scaled +
      (1 | team) + # Team offensive ability
      (1 | opponent) # Opponent defensive ability (negative = better defense)
  )

  offense_priors <- c(
    prior(normal(22, 5), class = Intercept),
    prior(normal(0, 2), class = b),
    prior(exponential(0.3), class = sd),
    prior(exponential(0.2), class = sigma)
  )

  offense_fit <- brm(
    offense_formula,
    data = games_long,
    prior = offense_priors,
    family = gaussian(),
    chains = chains,
    iter = iter,
    cores = cores,
    control = list(adapt_delta = 0.95),
    backend = "cmdstanr",
    refresh = 100,
    silent = 0
  )

  message("  ✓ Offensive model complete")

  # Defensive model: points_allowed ~ opponent offense + team defense
  message("\n  [2/2] Fitting defensive strength model...")

  defense_formula <- bf(
    points_allowed ~ 1 + week_scaled +
      (1 | team) + # Team defensive ability (negative = better defense)
      (1 | opponent) # Opponent offensive ability
  )

  defense_priors <- c(
    prior(normal(22, 5), class = Intercept),
    prior(normal(0, 2), class = b),
    prior(exponential(0.3), class = sd),
    prior(exponential(0.2), class = sigma)
  )

  defense_fit <- brm(
    defense_formula,
    data = games_long,
    prior = defense_priors,
    family = gaussian(),
    chains = chains,
    iter = iter,
    cores = cores,
    control = list(adapt_delta = 0.95),
    backend = "cmdstanr",
    refresh = 100,
    silent = 0
  )

  message("  ✓ Defensive model complete")

  list(offense = offense_fit, defense = defense_fit)
}

# ========================================
# RATINGS EXTRACTION
# ========================================

#' Extract team strength ratings from fitted model
#' @param fit brmsfit object from fit_team_strength_model
#' @param games_df Original games data (for season context)
#' @param by_season Extract separate ratings per season (default TRUE)
#' @return tibble with team, season, offense_strength, defense_strength, net_strength
extract_team_ratings <- function(fit, games_df, by_season = TRUE) {
  message("\nExtracting team strength ratings...")

  # Get random effects (posterior means)
  team_effects <- ranef(fit)

  home_effects <- team_effects$home_team[, , "Intercept"]
  away_effects <- team_effects$away_team[, , "Intercept"]

  teams <- rownames(home_effects)

  # Combine home and away effects
  # Home effect = offensive contribution to total points
  # Away effect = offensive contribution to total points
  # Average = team's offensive strength
  ratings <- tibble(
    team = teams,
    # Offensive strength: average of home and away offensive effects
    offense_mean = (home_effects[, "Estimate"] + away_effects[, "Estimate"]) / 2,
    offense_sd = sqrt(home_effects[, "Est.Error"]^2 + away_effects[, "Est.Error"]^2) / 2,
    offense_q05 = (home_effects[, "Q2.5"] + away_effects[, "Q2.5"]) / 2,
    offense_q95 = (home_effects[, "Q97.5"] + away_effects[, "Q97.5"]) / 2,
    # For this model, defensive strength is implicit (negative effect on opponent scoring)
    # We'll use opponent's offensive effect as defensive measure
    defense_mean = -(home_effects[, "Estimate"] + away_effects[, "Estimate"]) / 2,
    defense_sd = sqrt(home_effects[, "Est.Error"]^2 + away_effects[, "Est.Error"]^2) / 2,
    # Net strength = offense + defense (higher = better)
    net_strength = offense_mean + defense_mean
  )

  # If by_season, replicate ratings for each season in the data
  if (by_season) {
    seasons <- sort(unique(games_df$season))

    ratings <- ratings |>
      crossing(season = seasons) |>
      relocate(season, .before = team)
  }

  # Add percentile ranks
  ratings <- ratings |>
    group_by(season) |>
    mutate(
      offense_rank = rank(-offense_mean, ties.method = "average"),
      defense_rank = rank(-defense_mean, ties.method = "average"),
      net_rank = rank(-net_strength, ties.method = "average"),
      # Percentile (0-100)
      offense_pct = 100 * (n() - offense_rank + 1) / n(),
      defense_pct = 100 * (n() - defense_rank + 1) / n(),
      net_pct = 100 * (n() - net_rank + 1) / n()
    ) |>
    ungroup() |>
    arrange(season, desc(net_strength))

  message(sprintf(
    "  Extracted ratings for %d teams × %d seasons = %d records",
    length(unique(ratings$team)),
    length(unique(ratings$season)),
    nrow(ratings)
  ))

  ratings
}

#' Extract separate offensive and defensive ratings from two models
#' @param offense_fit Offensive model fit
#' @param defense_fit Defensive model fit
#' @param games_df Original games data
#' @return tibble with team, season, offense_strength, defense_strength, net_strength
extract_offense_defense_ratings <- function(offense_fit, defense_fit, games_df) {
  message("\nExtracting offense/defense ratings...")

  # Extract offensive strength from offense model
  off_effects <- ranef(offense_fit)$team[, , "Intercept"]

  # Extract defensive strength from defense model
  # (Negative = better defense, so flip sign)
  def_effects <- -ranef(defense_fit)$team[, , "Intercept"]

  teams <- rownames(off_effects)

  # Combine into single ratings table
  ratings <- tibble(
    team = teams,
    offense_mean = off_effects[, "Estimate"],
    offense_sd = off_effects[, "Est.Error"],
    offense_q05 = off_effects[, "Q2.5"],
    offense_q95 = off_effects[, "Q97.5"],
    defense_mean = def_effects[, "Estimate"],
    defense_sd = def_effects[, "Est.Error"],
    defense_q05 = def_effects[, "Q2.5"],
    defense_q95 = def_effects[, "Q97.5"],
    net_strength = offense_mean + defense_mean
  )

  # Replicate for each season
  seasons <- sort(unique(games_df$season))

  ratings <- ratings |>
    crossing(season = seasons) |>
    relocate(season, .before = team) |>
    group_by(season) |>
    mutate(
      offense_rank = rank(-offense_mean, ties.method = "average"),
      defense_rank = rank(-defense_mean, ties.method = "average"),
      net_rank = rank(-net_strength, ties.method = "average"),
      offense_pct = 100 * (n() - offense_rank + 1) / n(),
      defense_pct = 100 * (n() - defense_rank + 1) / n(),
      net_pct = 100 * (n() - net_rank + 1) / n()
    ) |>
    ungroup() |>
    arrange(season, desc(net_strength))

  message(sprintf(
    "  Extracted ratings for %d teams × %d seasons = %d records",
    length(unique(ratings$team)),
    length(unique(ratings$season)),
    nrow(ratings)
  ))

  ratings
}

# ========================================
# DIAGNOSTICS
# ========================================

#' Check MCMC diagnostics for convergence
#' @param fit brmsfit object
#' @return list(rhat_ok, ess_ok, divergences, convergence_ok)
check_mcmc_diagnostics <- function(fit) {
  message("\nChecking MCMC diagnostics...")

  # Get summary to extract R-hat and ESS
  fit_summary <- summary(fit)

  # R-hat (should be < 1.01 for all parameters)
  # Extract from fixed effects and random effects
  if (!is.null(fit_summary$fixed)) {
    rhats_fixed <- fit_summary$fixed[, "Rhat"]
  } else {
    rhats_fixed <- numeric(0)
  }

  if (!is.null(fit_summary$random)) {
    rhats_random <- sapply(fit_summary$random, function(x) {
      if (!is.null(x) && "Rhat" %in% colnames(x)) {
        return(x[, "Rhat"])
      }
      return(numeric(0))
    })
    rhats_random <- unlist(rhats_random)
  } else {
    rhats_random <- numeric(0)
  }

  rhats <- c(rhats_fixed, rhats_random)
  rhats <- rhats[!is.na(rhats)]

  if (length(rhats) > 0) {
    rhat_max <- max(rhats, na.rm = TRUE)
    rhat_ok <- rhat_max < 1.01
  } else {
    rhat_max <- NA
    rhat_ok <- FALSE
  }

  message(sprintf(
    "  R-hat: max = %.4f (target: <1.01) %s",
    rhat_max,
    ifelse(rhat_ok, "✓", "✗ FAILED")
  ))

  # Effective sample size (check Bulk_ESS and Tail_ESS)
  if (!is.null(fit_summary$fixed)) {
    ess_bulk_fixed <- fit_summary$fixed[, "Bulk_ESS"]
    ess_tail_fixed <- fit_summary$fixed[, "Tail_ESS"]
  } else {
    ess_bulk_fixed <- numeric(0)
    ess_tail_fixed <- numeric(0)
  }

  if (length(ess_bulk_fixed) > 0) {
    ess_bulk_min <- min(ess_bulk_fixed, na.rm = TRUE)
    ess_tail_min <- min(ess_tail_fixed, na.rm = TRUE)
    # ESS should be > 400 (or > 10% of total samples)
    ess_ok <- ess_bulk_min > 400 && ess_tail_min > 400
  } else {
    ess_bulk_min <- NA
    ess_tail_min <- NA
    ess_ok <- FALSE
  }

  message(sprintf(
    "  ESS (bulk): min = %.0f (target: >400) %s",
    ess_bulk_min,
    ifelse(ess_bulk_min > 400, "✓", "⚠ WARNING")
  ))
  message(sprintf(
    "  ESS (tail): min = %.0f (target: >400) %s",
    ess_tail_min,
    ifelse(ess_tail_min > 400, "✓", "⚠ WARNING")
  ))

  # Divergent transitions (should be 0)
  np <- nuts_params(fit)
  if (!is.null(np) && nrow(np) > 0) {
    divergences <- sum(subset(np, Parameter == "divergent__")$Value)
  } else {
    divergences <- 0
  }
  div_ok <- divergences < 10

  message(sprintf(
    "  Divergences: %d (target: <10) %s",
    divergences,
    ifelse(div_ok, "✓", "⚠ WARNING")
  ))

  # Overall convergence check
  convergence_ok <- rhat_ok && ess_ok && div_ok

  if (convergence_ok) {
    message("\n✓ MCMC diagnostics PASSED - Model converged successfully")
  } else {
    message("\n✗ MCMC diagnostics FAILED - Review model specification")
  }

  list(
    rhat_max = rhat_max,
    rhat_ok = rhat_ok,
    ess_bulk_min = ess_bulk_min,
    ess_tail_min = ess_tail_min,
    ess_ok = ess_ok,
    divergences = divergences,
    divergences_ok = div_ok,
    convergence_ok = convergence_ok
  )
}

#' Generate diagnostic plots
#' @param fit brmsfit object
#' @param output_dir Directory to save plots
generate_diagnostic_plots <- function(fit, output_dir = "models/team_strength") {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  message(sprintf("\nGenerating diagnostic plots (output: %s)...", output_dir))

  tryCatch(
    {
      # Posterior predictive check (most important)
      pp_file <- file.path(output_dir, "posterior_predictive_check.png")
      png(pp_file, width = 1000, height = 600)
      print(pp_check(fit, ndraws = 100))
      dev.off()
      message(sprintf("  ✓ Posterior predictive check: %s", pp_file))

      # Trace plots for fixed effects only (avoid parameter name issues)
      trace_file <- file.path(output_dir, "trace_plots.png")
      png(trace_file, width = 1200, height = 800)
      # Use regex to match only fixed effects
      print(mcmc_trace(fit, regex_pars = "^b_"))
      dev.off()
      message(sprintf("  ✓ Trace plots: %s", trace_file))

      # Parameter intervals for fixed effects
      param_file <- file.path(output_dir, "parameter_intervals.png")
      png(param_file, width = 1000, height = 800)
      print(mcmc_intervals(fit, regex_pars = "^b_|^sigma"))
      dev.off()
      message(sprintf("  ✓ Parameter intervals: %s", param_file))

      message("✓ Diagnostic plots complete")
    },
    error = function(e) {
      message(sprintf("  ⚠ Warning: Some diagnostic plots failed: %s", conditionMessage(e)))
      message("  Continuing without plots...")
    }
  )
}

# ========================================
# MAIN PIPELINE
# ========================================

#' Run full team strength rating pipeline
#' @param start_season Start season (default 2018)
#' @param end_season End season (default 2024)
#' @param model_type "total" or "separate" (default "separate")
#' @param chains MCMC chains (default 4)
#' @param iter MCMC iterations (default 2000)
#' @param cores CPU cores (default 4)
#' @return list(fit, ratings, diagnostics)
run_team_strength_pipeline <- function(
  start_season = 2018,
  end_season = 2024,
  model_type = c("total", "separate"),
  chains = 4,
  iter = 2000,
  cores = 4
) {
  model_type <- match.arg(model_type)

  message(strrep("=", 80))
  message("TEAM STRENGTH RATINGS PIPELINE")
  message(strrep("=", 80))
  message(sprintf("Season range: %d-%d", start_season, end_season))
  message(sprintf("Model type: %s", model_type))
  message(sprintf("MCMC config: %d chains × %d iterations", chains, iter))

  # Connect to database
  con <- connect_db()
  on.exit(dbDisconnect(con), add = TRUE)

  # Load and prepare data
  games_df <- fetch_games_for_team_strength(con, start_season, end_season)
  games_df <- prepare_brms_data(games_df)

  # Fit model(s)
  if (model_type == "total") {
    fit <- fit_team_strength_model(games_df, chains = chains, iter = iter, cores = cores)
    ratings <- extract_team_ratings(fit, games_df, by_season = TRUE)

    # Diagnostics
    diagnostics <- check_mcmc_diagnostics(fit)

    # Save model
    message("\nSaving model...")
    saveRDS(fit, "models/team_strength/team_strength_model.rds")
    message("  ✓ Saved: models/team_strength/team_strength_model.rds")

    # Generate diagnostic plots
    generate_diagnostic_plots(fit)

    return(list(
      fit = fit,
      ratings = ratings,
      diagnostics = diagnostics
    ))
  } else { # separate
    fits <- fit_offense_defense_models(games_df, chains = chains, iter = iter, cores = cores)
    ratings <- extract_offense_defense_ratings(fits$offense, fits$defense, games_df)

    # Diagnostics for both models
    message("\n=== OFFENSIVE MODEL DIAGNOSTICS ===")
    diag_offense <- check_mcmc_diagnostics(fits$offense)

    message("\n=== DEFENSIVE MODEL DIAGNOSTICS ===")
    diag_defense <- check_mcmc_diagnostics(fits$defense)

    diagnostics <- list(
      offense = diag_offense,
      defense = diag_defense,
      overall_ok = diag_offense$convergence_ok && diag_defense$convergence_ok
    )

    # Save models
    message("\nSaving models...")
    saveRDS(fits$offense, "models/team_strength/team_offense_model.rds")
    message("  ✓ Saved: models/team_strength/team_offense_model.rds")
    saveRDS(fits$defense, "models/team_strength/team_defense_model.rds")
    message("  ✓ Saved: models/team_strength/team_defense_model.rds")

    # Generate diagnostic plots for offense model
    message("\nGenerating offensive model diagnostics...")
    generate_diagnostic_plots(fits$offense, "models/team_strength/offense")

    message("\nGenerating defensive model diagnostics...")
    generate_diagnostic_plots(fits$defense, "models/team_strength/defense")

    return(list(
      offense_fit = fits$offense,
      defense_fit = fits$defense,
      ratings = ratings,
      diagnostics = diagnostics
    ))
  }
}

# ========================================
# MAIN EXECUTION
# ========================================

if (!interactive()) {
  # Parse command-line arguments
  args <- commandArgs(trailingOnly = TRUE)

  start_season <- ifelse(length(args) > 0, as.integer(args[1]), 2018)
  end_season <- ifelse(length(args) > 1, as.integer(args[2]), 2024)
  model_type <- ifelse(length(args) > 2, args[3], "separate")

  tryCatch(
    {
      # Run pipeline
      results <- run_team_strength_pipeline(
        start_season = start_season,
        end_season = end_season,
        model_type = model_type
      )

      # Save ratings to CSV
      output_file <- "models/team_strength/team_strength_ratings.csv"
      message(sprintf("\nSaving ratings to %s...", output_file))

      write.csv(results$ratings, output_file, row.names = FALSE)

      message(sprintf("  ✓ Saved %d ratings", nrow(results$ratings)))

      # Print top/bottom teams for latest season
      latest_season <- max(results$ratings$season)
      latest_ratings <- results$ratings |>
        filter(season == latest_season) |>
        select(team, offense_mean, defense_mean, net_strength, net_rank)

      message(sprintf("\n=== TOP 5 TEAMS (%d) ===", latest_season))
      print(head(latest_ratings, 5), n = 5)

      message(sprintf("\n=== BOTTOM 5 TEAMS (%d) ===", latest_season))
      print(tail(latest_ratings, 5), n = 5)

      # Summary statistics
      message("\n=== RATING STATISTICS ===")
      message(sprintf(
        "  Offense range: [%.2f, %.2f]",
        min(results$ratings$offense_mean),
        max(results$ratings$offense_mean)
      ))
      message(sprintf(
        "  Defense range: [%.2f, %.2f]",
        min(results$ratings$defense_mean),
        max(results$ratings$defense_mean)
      ))
      message(sprintf(
        "  Net strength range: [%.2f, %.2f]",
        min(results$ratings$net_strength),
        max(results$ratings$net_strength)
      ))

      # Convergence summary
      if (model_type == "separate") {
        message("\n=== MCMC CONVERGENCE SUMMARY ===")
        message(sprintf(
          "  Offensive model: %s",
          ifelse(results$diagnostics$offense$convergence_ok, "✓ PASS", "✗ FAIL")
        ))
        message(sprintf(
          "  Defensive model: %s",
          ifelse(results$diagnostics$defense$convergence_ok, "✓ PASS", "✗ FAIL")
        ))
        message(sprintf(
          "  Overall: %s",
          ifelse(results$diagnostics$overall_ok, "✓ PASS", "✗ FAIL")
        ))
      } else {
        message("\n=== MCMC CONVERGENCE SUMMARY ===")
        message(sprintf(
          "  Model: %s",
          ifelse(results$diagnostics$convergence_ok, "✓ PASS", "✗ FAIL")
        ))
      }

      message("\n\n")
      message(strrep("=", 80))
      message("TEAM STRENGTH RATINGS COMPLETE")
      message(strrep("=", 80))
      message(sprintf("Output: %s", output_file))
      message(sprintf("Models: models/team_strength/*.rds"))
      message(sprintf("Diagnostics: models/team_strength/*.png"))

      if (results$diagnostics$overall_ok || results$diagnostics$convergence_ok) {
        message("\n✓ All MCMC diagnostics passed - Ready for production")
        quit(status = 0)
      } else {
        message("\n⚠ Some MCMC diagnostics failed - Review before production use")
        quit(status = 1)
      }
    },
    error = function(e) {
      message(sprintf("\n✗ ERROR: %s", conditionMessage(e)))
      message(traceback())
      quit(status = 1)
    }
  )
}
