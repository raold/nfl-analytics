#!/usr/bin/env Rscript
# =============================================================================
# Bayesian Hierarchical Models for NFL Player Props
# =============================================================================
#
# This script implements multi-level hierarchical Bayesian models for player
# prop predictions using brms/Stan. The hierarchy is:
#   League → Position Group → Position → Team → Player → Game
#
# Models:
#   1. Passing Yards (QBs only)
#   2. Rushing Yards (RBs, QBs, WRs)
#   3. Receiving Yards (WRs, TEs, RBs)
#
# Author: Claude Opus
# Date: October 2025
# =============================================================================

library(brms)
library(tidyverse)
library(DBI)
library(RPostgres)  # Use RPostgres instead of RPostgreSQL
library(cmdstanr)
library(posterior)
library(loo)
library(bayesplot)
library(jsonlite)
library(lubridate)
library(glue)

# Configure Stan backend for performance
options(mc.cores = parallel::detectCores() - 1)
options(brms.backend = "cmdstanr")
options(brms.threads = 2)

# Database connection
get_db_connection <- function() {
  dbConnect(
    Postgres(),  # Updated for RPostgres
    host = Sys.getenv("DB_HOST", "localhost"),
    port = as.integer(Sys.getenv("DB_PORT", 5544)),
    dbname = Sys.getenv("DB_NAME", "devdb01"),
    user = Sys.getenv("DB_USER", "dro"),
    password = Sys.getenv("DB_PASSWORD", "sicillionbillions")
  )
}

# =============================================================================
# Data Loading Functions
# =============================================================================

load_player_data <- function(conn, stat_category, min_season = 2020) {
  "Load player-game statistics with hierarchical structure"

  query <- glue("
    SELECT
      pgs.player_id,
      pgs.season,
      pgs.week,
      pgs.stat_category,
      pgs.player_display_name,
      pgs.stat_attempts,
      pgs.stat_completions,
      pgs.stat_yards,
      pgs.stat_touchdowns,
      pgs.stat_negative,
      pgs.avg_time_to_throw,
      pgs.avg_air_yards_differential,
      pgs.cpoe,
      ph.position,
      ph.position_group,
      ph.current_team as team,
      ph.years_exp,
      ph.games_with_stats,
      ph.hierarchy_position,
      ph.hierarchy_team_position,
      -- Game context from games table
      g.home_team,
      g.away_team,
      g.spread_close as spread_line,
      g.total_close as total_line,
      g.temp,
      g.wind,
      g.roof,
      g.surface,
      -- Opponent info
      CASE
        WHEN g.home_team = ph.current_team THEN g.away_team
        ELSE g.home_team
      END as opponent,
      CASE
        WHEN g.home_team = ph.current_team THEN 1
        ELSE 0
      END as is_home
    FROM mart.player_game_stats pgs
    JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
    LEFT JOIN games g ON
      g.season = pgs.season
      AND g.week = pgs.week
      AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
    WHERE
      pgs.stat_category = '{stat_category}'
      AND pgs.season >= {min_season}
      AND pgs.stat_yards IS NOT NULL
      AND ph.position_group IS NOT NULL
    ORDER BY pgs.season, pgs.week, pgs.player_id
  ")

  data <- dbGetQuery(conn, query)

  # Add derived features
  data <- data %>%
    mutate(
      # Log transform for better model fit
      log_yards = log1p(stat_yards),
      log_attempts = log1p(stat_attempts),

      # Efficiency metrics
      yards_per_attempt = ifelse(stat_attempts > 0, stat_yards / stat_attempts, 0),

      # Experience categories
      experience_cat = case_when(
        years_exp <= 1 ~ "rookie",
        years_exp <= 3 ~ "early_career",
        years_exp <= 7 ~ "prime",
        TRUE ~ "veteran"
      ),

      # Weather impact
      is_bad_weather = ifelse(!is.na(wind) & wind > 15, 1, 0),
      is_dome = ifelse(!is.na(roof) & roof %in% c("dome", "closed"), 1, 0),

      # Game script
      is_favored = ifelse(!is.na(spread_line) & spread_line < 0, 1, 0),
      spread_abs = abs(spread_line),

      # Unique identifiers for hierarchical structure
      player_season = paste0(player_id, "_", season),
      team_season = paste0(team, "_", season),
      position_season = paste0(position_group, "_", season)
    )

  cat(glue("Loaded {nrow(data)} {stat_category} records\n"))
  cat(glue("Players: {length(unique(data$player_id))}\n"))
  cat(glue("Position groups: {length(unique(data$position_group))}\n"))
  cat(glue("Teams: {length(unique(data$team))}\n\n"))

  return(data)
}

# =============================================================================
# Model Specifications
# =============================================================================

build_passing_model <- function(data, chains = 4, iter = 2000, warmup = 1000) {
  "Hierarchical model for QB passing yards"

  cat("Building passing yards hierarchical model...\n")

  # Define priors based on domain knowledge
  priors <- c(
    # Population-level effects
    prior(normal(200, 50), class = Intercept),  # Average passing yards
    prior(normal(0, 20), class = b),            # Fixed effects

    # Group-level SDs (hierarchical structure)
    prior(exponential(0.05), class = sd, group = player_id),        # Player variation
    prior(exponential(0.1), class = sd, group = team),              # Team variation
    prior(exponential(0.2), class = sd, group = position_season),   # Position-season
    prior(exponential(0.1), class = sd, group = opponent)           # Opponent defense

    # Note: No sigma prior since we use distributional formula sigma ~ log_attempts
  )

  # Formula with hierarchical structure
  formula <- bf(
    log_yards ~ 1 +
      # Fixed effects
      log_attempts +
      is_home +
      is_favored +
      spread_abs +
      is_bad_weather +
      is_dome +
      scale(total_line) +
      experience_cat +

      # Hierarchical random effects (partial pooling)
      (1 | player_id) +                    # Player-specific intercept
      (1 | team) +                          # Team offensive system
      (1 | opponent) +                      # Opponent defensive strength
      (1 | position_season) +               # Position group by season
      (log_attempts | player_season),      # Player-season slope for attempts

    # Model variance as function of attempts (heteroscedasticity)
    sigma ~ log_attempts
  )

  # Fit model
  model <- brm(
    formula = formula,
    data = data,
    prior = priors,
    family = gaussian(),
    chains = chains,
    iter = iter,
    warmup = warmup,
    control = list(
      adapt_delta = 0.95,
      max_treedepth = 12
    ),
    save_pars = save_pars(all = TRUE),
    seed = 42,
    backend = "cmdstanr",
    threads = threading(2)
  )

  return(model)
}

build_rushing_model <- function(data, chains = 4, iter = 2000, warmup = 1000) {
  "Hierarchical model for rushing yards"

  cat("Building rushing yards hierarchical model...\n")

  # Different priors for rushing (lower average, higher variance)
  priors <- c(
    prior(normal(50, 30), class = Intercept),   # Average rushing yards
    prior(normal(0, 15), class = b),
    prior(exponential(0.03), class = sd, group = player_id),
    prior(exponential(0.1), class = sd, group = team),
    prior(exponential(0.15), class = sd, group = position_group),
    prior(exponential(0.1), class = sd, group = opponent)
    # Note: No sigma prior since we use distributional formula
  )

  formula <- bf(
    log_yards ~ 1 +
      # Fixed effects
      log_attempts +
      is_home +
      is_favored +
      spread_abs +
      scale(total_line) +
      experience_cat +
      position_group +  # Position matters more for rushing

      # Hierarchical structure
      (1 | player_id) +
      (1 | team) +
      (1 | opponent) +
      (1 | position_group:team) +  # Position-team interaction
      (log_attempts | player_season),

    sigma ~ log_attempts + position_group
  )

  model <- brm(
    formula = formula,
    data = data,
    prior = priors,
    family = gaussian(),
    chains = chains,
    iter = iter,
    warmup = warmup,
    control = list(
      adapt_delta = 0.95,
      max_treedepth = 12
    ),
    save_pars = save_pars(all = TRUE),
    seed = 42,
    backend = "cmdstanr",
    threads = threading(2)
  )

  return(model)
}

build_receiving_model <- function(data, chains = 4, iter = 2000, warmup = 1000) {
  "Hierarchical model for receiving yards"

  cat("Building receiving yards hierarchical model...\n")

  priors <- c(
    prior(normal(60, 30), class = Intercept),
    prior(normal(0, 15), class = b),
    prior(exponential(0.03), class = sd, group = player_id),
    prior(exponential(0.1), class = sd, group = team),
    prior(exponential(0.15), class = sd, group = position_group),
    prior(exponential(0.1), class = sd, group = opponent),
    prior(exponential(0.1), class = sd, group = qb_id)  # QB throwing to receiver
    # Note: No sigma prior since we use distributional formula
  )

  # Need to add QB information for receiving model
  data <- data %>%
    mutate(
      # Simplified: use team as proxy for QB (would need actual QB mapping)
      qb_id = paste0(team, "_QB")
    )

  formula <- bf(
    log_yards ~ 1 +
      # Fixed effects
      log_attempts +  # Targets
      is_home +
      is_favored +
      spread_abs +
      scale(total_line) +
      experience_cat +
      position_group +

      # Hierarchical structure
      (1 | player_id) +
      (1 | team) +
      (1 | opponent) +
      (1 | qb_id) +  # QB effect on receiver
      (1 | position_group:team) +
      (log_attempts | player_season),

    sigma ~ log_attempts + position_group
  )

  model <- brm(
    formula = formula,
    data = data,
    prior = priors,
    family = gaussian(),
    chains = chains,
    iter = iter,
    warmup = warmup,
    control = list(
      adapt_delta = 0.95,
      max_treedepth = 12
    ),
    save_pars = save_pars(all = TRUE),
    seed = 42,
    backend = "cmdstanr",
    threads = threading(2)
  )

  return(model)
}

# =============================================================================
# Model Diagnostics and Validation
# =============================================================================

diagnose_model <- function(model, model_name) {
  "Run comprehensive diagnostics on fitted model"

  cat(glue("\n{paste0(rep('=', 60), collapse='')}\n"))
  cat(glue("Diagnostics for {model_name}\n"))
  cat(glue("{paste0(rep('=', 60), collapse='')}\n\n"))

  # 1. Summary statistics
  cat("Model Summary:\n")
  print(summary(model))

  # 2. Check convergence (Rhat)
  rhats <- rhat(model)
  problematic_rhats <- rhats[rhats > 1.01]
  if(length(problematic_rhats) > 0) {
    cat("\nWARNING: Parameters with Rhat > 1.01:\n")
    print(problematic_rhats)
  } else {
    cat("\n✓ All parameters converged (Rhat < 1.01)\n")
  }

  # 3. Effective sample size
  ess <- ess_bulk(model)
  low_ess <- ess[ess < 400]
  if(length(low_ess) > 0) {
    cat("\nWARNING: Parameters with ESS < 400:\n")
    print(low_ess)
  } else {
    cat("✓ Adequate effective sample sizes (ESS > 400)\n")
  }

  # 4. Check for divergences
  divs <- nuts_params(model)$divergent__
  n_divs <- sum(divs)
  if(n_divs > 0) {
    cat(glue("\nWARNING: {n_divs} divergent transitions detected\n"))
  } else {
    cat("✓ No divergent transitions\n")
  }

  # 5. Posterior predictive checks
  cat("\nPosterior Predictive Check:\n")
  pp_check_plot <- pp_check(model, ndraws = 100)

  # Save diagnostic plots
  dir.create("figures/bayesian_diagnostics", recursive = TRUE, showWarnings = FALSE)

  # Trace plots
  trace_plot <- mcmc_trace(model, pars = c("b_Intercept", "b_log_attempts", "sd_player_id__Intercept"))
  ggsave(glue("figures/bayesian_diagnostics/{model_name}_trace.png"), trace_plot, width = 12, height = 8)

  # Posterior distributions
  post_plot <- mcmc_areas(model, pars = c("b_Intercept", "b_log_attempts", "b_is_home"))
  ggsave(glue("figures/bayesian_diagnostics/{model_name}_posteriors.png"), post_plot, width = 10, height = 6)

  # PP check
  ggsave(glue("figures/bayesian_diagnostics/{model_name}_pp_check.png"), pp_check_plot, width = 10, height = 6)

  cat(glue("\nDiagnostic plots saved to figures/bayesian_diagnostics/\n"))

  return(list(
    rhats = rhats,
    ess = ess,
    n_divergences = n_divs
  ))
}

# =============================================================================
# Extract and Save Model Results
# =============================================================================

extract_player_ratings <- function(model, data, stat_type, model_version = "v1.0") {
  "Extract hierarchical player effects and save to database"

  cat(glue("\nExtracting player ratings for {stat_type}...\n"))

  # Extract posterior samples
  posterior_samples <- as_draws_df(model)

  # Get player-specific effects
  player_cols <- grep("^r_player_id\\[", names(posterior_samples), value = TRUE)

  # Process each player
  results <- list()

  for(player_col in player_cols) {
    # Extract player ID from column name
    player_id <- gsub("r_player_id\\[(.+),Intercept\\]", "\\1", player_col)

    if(player_id %in% unique(data$player_id)) {
      # Get player's posterior distribution
      player_effect <- posterior_samples[[player_col]]

      # Get team and position group effects if available
      player_data <- data %>%
        filter(player_id == !!player_id) %>%
        slice(1)

      team_col <- glue("r_team[{player_data$team},Intercept]")
      position_col <- glue("r_position_season[{player_data$position_group}_{max(data$season)},Intercept]")

      team_effect <- if(team_col %in% names(posterior_samples)) {
        mean(posterior_samples[[team_col]])
      } else { 0 }

      position_effect <- if(position_col %in% names(posterior_samples)) {
        mean(posterior_samples[[position_col]])
      } else { 0 }

      # Calculate statistics
      results[[player_id]] <- data.frame(
        player_id = player_id,
        stat_type = stat_type,
        season = max(data$season),
        model_version = model_version,
        rating_mean = mean(player_effect),
        rating_sd = sd(player_effect),
        rating_q05 = quantile(player_effect, 0.05),
        rating_q50 = quantile(player_effect, 0.50),
        rating_q95 = quantile(player_effect, 0.95),
        position_group_mean = position_effect,
        team_effect = team_effect,
        vs_opponent_effect = NA,  # Would need opponent-specific extraction
        n_games_observed = sum(data$player_id == player_id),
        effective_sample_size = ess_bulk(player_effect),
        rhat = rhat(player_effect),
        trained_at = Sys.time(),
        updated_at = Sys.time()
      )
    }
  }

  # Combine results
  ratings_df <- bind_rows(results)

  cat(glue("Extracted ratings for {nrow(ratings_df)} players\n"))

  return(ratings_df)
}

save_to_database <- function(conn, ratings_df, stat_type) {
  "Save player ratings to mart.bayesian_player_ratings"

  cat(glue("\nSaving {nrow(ratings_df)} {stat_type} ratings to database...\n"))

  # Delete existing ratings for this stat_type and model version
  delete_query <- glue("
    DELETE FROM mart.bayesian_player_ratings
    WHERE stat_type = '{stat_type}'
      AND model_version = '{ratings_df$model_version[1]}'
  ")
  dbExecute(conn, delete_query)

  # Insert new ratings
  dbWriteTable(
    conn,
    c("mart", "bayesian_player_ratings"),
    ratings_df,
    append = TRUE,
    row.names = FALSE
  )

  cat("✓ Ratings saved to database\n")
}

# =============================================================================
# Model Comparison with LOO-CV
# =============================================================================

compare_models <- function(models, names) {
  "Compare multiple models using LOO-CV"

  cat(paste0("\n", paste0(rep("=", 60), collapse=""), "\n"))
  cat("MODEL COMPARISON (LOO-CV)\n")
  cat(paste0(paste0(rep("=", 60), collapse=""), "\n\n"))

  # Compute LOO for each model
  loos <- list()
  for(i in seq_along(models)) {
    cat(glue("Computing LOO for {names[i]}...\n"))
    loos[[names[i]]] <- loo(models[[i]], cores = parallel::detectCores() - 1)
  }

  # Compare models
  if(length(loos) > 1) {
    comparison <- loo_compare(loos)
    print(comparison)

    # Model weights for ensemble
    model_weights <- loo_model_weights(loos)
    cat("\nModel Weights for Ensemble:\n")
    print(model_weights)

    return(list(
      loo_results = loos,
      comparison = comparison,
      weights = model_weights
    ))
  } else {
    return(list(loo_results = loos))
  }
}

# =============================================================================
# Main Execution
# =============================================================================

main <- function() {
  cat(paste0(paste0(rep("=", 80), collapse=""), "\n"))
  cat("BAYESIAN HIERARCHICAL MODELS FOR NFL PLAYER PROPS\n")
  cat(paste0(paste0(rep("=", 80), collapse=""), "\n\n"))

  # Connect to database
  conn <- get_db_connection()
  on.exit(dbDisconnect(conn))

  # Track execution time
  start_time <- Sys.time()

  tryCatch({
    # 1. PASSING YARDS MODEL
    cat(paste0("\n", paste0(rep("=", 60), collapse=""), "\n"))
    cat("PHASE 1: PASSING YARDS MODEL\n")
    cat(paste0(paste0(rep("=", 60), collapse=""), "\n"))

    passing_data <- load_player_data(conn, "passing", min_season = 2020)

    if(nrow(passing_data) > 0) {
      passing_model <- build_passing_model(passing_data, chains = 4, iter = 2000)

      # Diagnostics
      passing_diag <- diagnose_model(passing_model, "passing_yards")

      # Extract and save ratings
      passing_ratings <- extract_player_ratings(
        passing_model,
        passing_data,
        "passing_yards",
        "hierarchical_v1.0"
      )
      save_to_database(conn, passing_ratings, "passing_yards")

      # Save model object
      dir.create("models/bayesian", recursive = TRUE, showWarnings = FALSE)
      saveRDS(passing_model, "models/bayesian/passing_yards_hierarchical_v1.rds")
      cat("✓ Passing model saved to models/bayesian/\n")
    }

    # 2. RUSHING YARDS MODEL
    cat(paste0("\n", paste0(rep("=", 60), collapse=""), "\n"))
    cat("PHASE 2: RUSHING YARDS MODEL\n")
    cat(paste0(paste0(rep("=", 60), collapse=""), "\n"))

    rushing_data <- load_player_data(conn, "rushing", min_season = 2020)

    if(nrow(rushing_data) > 0) {
      rushing_model <- build_rushing_model(rushing_data, chains = 4, iter = 2000)

      # Diagnostics
      rushing_diag <- diagnose_model(rushing_model, "rushing_yards")

      # Extract and save ratings
      rushing_ratings <- extract_player_ratings(
        rushing_model,
        rushing_data,
        "rushing_yards",
        "hierarchical_v1.0"
      )
      save_to_database(conn, rushing_ratings, "rushing_yards")

      # Save model
      saveRDS(rushing_model, "models/bayesian/rushing_yards_hierarchical_v1.rds")
      cat("✓ Rushing model saved to models/bayesian/\n")
    }

    # 3. RECEIVING YARDS MODEL
    cat(paste0("\n", paste0(rep("=", 60), collapse=""), "\n"))
    cat("PHASE 3: RECEIVING YARDS MODEL\n")
    cat(paste0(paste0(rep("=", 60), collapse=""), "\n"))

    receiving_data <- load_player_data(conn, "receiving", min_season = 2020)

    if(nrow(receiving_data) > 0) {
      receiving_model <- build_receiving_model(receiving_data, chains = 4, iter = 2000)

      # Diagnostics
      receiving_diag <- diagnose_model(receiving_model, "receiving_yards")

      # Extract and save ratings
      receiving_ratings <- extract_player_ratings(
        receiving_model,
        receiving_data,
        "receiving_yards",
        "hierarchical_v1.0"
      )
      save_to_database(conn, receiving_ratings, "receiving_yards")

      # Save model
      saveRDS(receiving_model, "models/bayesian/receiving_yards_hierarchical_v1.rds")
      cat("✓ Receiving model saved to models/bayesian/\n")
    }

    # 4. MODEL COMPARISON
    if(exists("passing_model") && exists("rushing_model") && exists("receiving_model")) {
      cat(paste0("\n", paste0(rep("=", 60), collapse=""), "\n"))
      cat("PHASE 4: MODEL COMPARISON\n")
      cat(paste0(paste0(rep("=", 60), collapse=""), "\n"))

      # Could compare different model specifications here
      # For now, just show individual LOO results

      cat("\nComputing LOO-CV for all models...\n")
      passing_loo <- loo(passing_model, cores = parallel::detectCores() - 1)
      rushing_loo <- loo(rushing_model, cores = parallel::detectCores() - 1)
      receiving_loo <- loo(receiving_model, cores = parallel::detectCores() - 1)

      cat("\nLOO Results:\n")
      cat("Passing Model ELPD:", passing_loo$estimates["elpd_loo", "Estimate"], "\n")
      cat("Rushing Model ELPD:", rushing_loo$estimates["elpd_loo", "Estimate"], "\n")
      cat("Receiving Model ELPD:", receiving_loo$estimates["elpd_loo", "Estimate"], "\n")
    }

    # Summary
    end_time <- Sys.time()
    duration <- difftime(end_time, start_time, units = "mins")

    cat(paste0("\n", paste0(rep("=", 80), collapse=""), "\n"))
    cat("EXECUTION COMPLETE\n")
    cat(paste0(paste0(rep("=", 80), collapse=""), "\n"))
    cat(glue("Total execution time: {round(duration, 2)} minutes\n"))
    cat("\nAll models trained and saved successfully!\n")
    cat("Player ratings stored in mart.bayesian_player_ratings\n")
    cat("Model objects saved to models/bayesian/\n")
    cat("Diagnostic plots saved to figures/bayesian_diagnostics/\n")

  }, error = function(e) {
    cat("\n❌ ERROR:", conditionMessage(e), "\n")
    traceback()
  })
}

# Run if called directly
if(!interactive()) {
  main()
}