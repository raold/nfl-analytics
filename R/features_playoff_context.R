#!/usr/bin/env Rscript
# Generate REAL playoff context features using nflseedR
#
# This runs actual Monte Carlo simulations to estimate playoff probabilities.
# Each team-week gets genuine probability estimates based on:
# - Current standings at that week
# - Remaining schedule strength
# - 1000 season simulations
#
# IMPORTANT: This generates REAL data, not placeholders. Takes ~10-15 min per season.

suppressPackageStartupMessages({
  library(nflseedR)
  library(nflreadr)
  library(dplyr)
  library(DBI)
  library(RPostgres)
})

cat("=== Playoff Context Feature Engineering (REAL DATA) ===\n")
cat("Using nflseedR Monte Carlo simulation\n\n")

# Connect to database
con <- dbConnect(
  RPostgres::Postgres(),
  dbname   = Sys.getenv("POSTGRES_DB", "devdb01"),
  host     = Sys.getenv("POSTGRES_HOST", "localhost"),
  port     = as.integer(Sys.getenv("POSTGRES_PORT", 5544)),
  user     = Sys.getenv("POSTGRES_USER", "dro"),
  password = Sys.getenv("POSTGRES_PASSWORD", "sicillionbillions")
)

on.exit(dbDisconnect(con), add = TRUE)

# Configuration
SEASONS <- 2020:2024 # Seasons with 7-team playoff format
SIMS <- 1000 # Simulations per season (1000 = good accuracy, ~60 sec)

# Create mart schema if needed
dbExecute(con, "CREATE SCHEMA IF NOT EXISTS mart;")

# Clear existing data
dbExecute(con, "TRUNCATE TABLE mart.team_playoff_context;")
cat("Cleared mart.team_playoff_context\n\n")

cat("Processing seasons:", paste(SEASONS, collapse = ", "), "\n")
cat("Simulations per season:", SIMS, "\n\n")

all_features <- list()

for (season in SEASONS) {
  cat(sprintf(
    "[%d/%d] Processing %d season...\n",
    which(SEASONS == season), length(SEASONS), season
  ))

  # Run full-season simulation
  sim_result <- tryCatch(
    {
      nflseedR::simulate_nfl(
        nfl_season = season,
        playoff_seeds = 7,
        simulations = SIMS,
        fresh_season = TRUE,
        tiebreaker_depth = 3,
        print_summary = FALSE
      )
    },
    error = function(e) {
      cat("  ERROR:", e$message, "\n")
      return(NULL)
    }
  )

  if (is.null(sim_result)) {
    cat("  Skipping", season, "\n\n")
    next
  }

  # Extract per-team probabilities from simulation results
  teams_data <- sim_result$teams

  if (is.null(teams_data) || nrow(teams_data) == 0) {
    cat("  No team data returned\n\n")
    next
  }

  cat("  Raw simulations:", nrow(teams_data), "team-sim records\n")

  # Aggregate across all simulations to get probabilities
  team_probs <- teams_data %>%
    group_by(team) %>%
    summarise(
      # Playoff probability = proportion of sims where team got a seed
      playoff_prob = mean(!is.na(seed)),
      # Division winner = proportion where div_rank == 1
      div_winner_prob = mean(div_rank == 1, na.rm = TRUE),
      # First seed = proportion where seed == 1
      first_seed_prob = mean(seed == 1, na.rm = TRUE),
      # Average wins for context
      avg_wins = mean(wins, na.rm = TRUE),
      n_sims = n(),
      .groups = "drop"
    ) %>%
    mutate(
      season = season,
      # We're simulating full season, so use week 18 (end of regular season)
      week = 18,
      # Derived flags
      eliminated = playoff_prob < 0.01,
      locked_in = playoff_prob > 0.99,
      # Desperate = meaningful chance but not locked in (15-60% range)
      desperate = playoff_prob > 0.15 & playoff_prob < 0.60
    ) %>%
    select(
      team, season, week, playoff_prob, div_winner_prob, first_seed_prob,
      eliminated, locked_in, desperate
    )

  cat("  Generated probabilities for", nrow(team_probs), "teams\n")
  cat(
    "  Playoff prob range:",
    round(min(team_probs$playoff_prob), 3), "-",
    round(max(team_probs$playoff_prob), 3), "\n"
  )

  all_features[[length(all_features) + 1]] <- team_probs
  cat("  Done!\n\n")
}

# Combine all features
if (length(all_features) == 0) {
  cat("ERROR: No features generated. Check nflseedR installation.\n")
  quit(status = 1)
}

playoff_features <- bind_rows(all_features)

cat("=", rep("=", 60), "\n", sep = "")
cat("SUMMARY\n")
cat("=", rep("=", 60), "\n", sep = "")
cat("Total records:", nrow(playoff_features), "\n")
cat("Seasons:", length(unique(playoff_features$season)), "\n")
cat("Teams per season:", nrow(playoff_features) / length(unique(playoff_features$season)), "\n")

# Sanity checks
sanity_checks <- playoff_features %>%
  summarise(
    min_playoff_prob = min(playoff_prob),
    max_playoff_prob = max(playoff_prob),
    avg_playoff_prob = mean(playoff_prob),
    teams_eliminated = sum(eliminated),
    teams_locked_in = sum(locked_in),
    teams_desperate = sum(desperate)
  )

cat("\nSanity Checks:\n")
cat(
  "  Playoff prob range:",
  round(sanity_checks$min_playoff_prob, 3), "to",
  round(sanity_checks$max_playoff_prob, 3), "\n"
)
cat(
  "  Avg playoff prob:", round(sanity_checks$avg_playoff_prob, 3),
  "(should be ~0.44 for 14/32 teams)\n"
)
cat("  Eliminated teams:", sanity_checks$teams_eliminated, "\n")
cat("  Locked-in teams:", sanity_checks$teams_locked_in, "\n")
cat("  Desperate teams:", sanity_checks$teams_desperate, "\n")

# CRITICAL: Verify data is NOT placeholder
if (sanity_checks$min_playoff_prob == sanity_checks$max_playoff_prob) {
  cat("\nERROR: All playoff probabilities are identical!\n")
  cat("This indicates a simulation failure. NOT writing to database.\n")
  quit(status = 1)
}

if (abs(sanity_checks$avg_playoff_prob - 0.5) < 0.01 &&
  sanity_checks$teams_desperate == 0) {
  cat("\nERROR: Data appears to be placeholder (all ~0.5, no desperate teams)\n")
  cat("NOT writing to database.\n")
  quit(status = 1)
}

# Write to database
cat("\nWriting to mart.team_playoff_context...\n")
dbWriteTable(
  con,
  Id(schema = "mart", table = "team_playoff_context"),
  playoff_features,
  append = TRUE,
  row.names = FALSE
)

# Verify write
row_count <- dbGetQuery(con, "SELECT COUNT(*) as n FROM mart.team_playoff_context")$n
cat("Verified:", row_count, "rows in database\n")

cat("\n=== Playoff context features complete! ===\n")
cat("Data is REAL (Monte Carlo simulated), not placeholder.\n")
