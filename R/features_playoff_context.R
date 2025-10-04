#!/usr/bin/env Rscript
# Generate playoff context features using nflseedR
# Includes: playoff probability, desperation indicators, elimination status

suppressPackageStartupMessages({
  library(nflseedR)
  library(nflreadr)
  library(dplyr)
  library(DBI)
  library(RPostgres)
})

cat("=== Playoff Context Feature Engineering ===\n\n")

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
SEASONS <- 2020:2024
SIMS <- 500  # Number of playoff simulations per season

# Create mart schema if needed
dbExecute(con, "CREATE SCHEMA IF NOT EXISTS mart;")

# Truncate existing table (keeps structure/indexes)
dbExecute(con, "TRUNCATE TABLE mart.team_playoff_context;")
cat("Truncated existing mart.team_playoff_context\n\n")

cat("Processing seasons:", paste(SEASONS, collapse = ", "), "\n\n")

all_features <- list()

for (season in SEASONS) {
  cat("Processing", season, "season...\n")
  
  # Load schedules for this season
  schedules <- tryCatch({
    nflreadr::load_schedules(season)
  }, error = function(e) {
    cat("  ERROR loading schedules:", e$message, "\n")
    return(NULL)
  })
  
  if (is.null(schedules) || nrow(schedules) == 0) {
    cat("  Skipping", season, "(no schedule data)\n\n")
    next
  }
  
  # Determine max completed week
  max_week <- schedules |>
    filter(!is.na(home_score)) |>
    pull(week) |>
    max(na.rm = TRUE)
  
  if (is.infinite(max_week) || max_week < 1) {
    cat("  Skipping", season, "(no completed games)\n\n")
    next
  }
  
  cat("  Max completed week:", max_week, "\n")
  
  # Simulate playoffs for each week
  for (wk in 1:min(max_week, 18)) {
    cat("  Simulating week", wk, "...")
    
    # Run simulation - nflseedR automatically uses completed games through current week
    sim_result <- tryCatch({
      nflseedR::simulate_nfl(
        nfl_season = season,
        playoff_seeds = 7,
        sims = SIMS,
        fresh_season = TRUE,
        fresh_week = wk,
        tiebreaker_depth = 3
      )
    }, error = function(e) {
      cat(" ERROR:", e$message, "\n")
      return(NULL)
    })
    
    if (is.null(sim_result)) {
      next
    }
    
    # Extract summary data from simulation result
    # nflseedR returns an S3 object with summary, teams, and games components
    week_probs <- tryCatch({
      sim_summary <- sim_result$teams  # This has the aggregated probabilities
      
      if (is.null(sim_summary) || nrow(sim_summary) == 0) {
        cat(" no results\n")
        return(NULL)
      }
      
      # Transform to our schema
      sim_summary |>
        mutate(
          season = season,
          week = wk,
          playoff_prob = ifelse("make_playoffs" %in% names(sim_summary), make_playoffs, NA),
          div_winner_prob = ifelse("win_division" %in% names(sim_summary), win_division, NA),
          first_seed_prob = ifelse("first_seed" %in% names(sim_summary), first_seed, NA)
        ) |>
        mutate(
          eliminated = playoff_prob < 0.01,
          locked_in = playoff_prob > 0.99,
          desperate = playoff_prob > 0.15 & playoff_prob < 0.60
        ) |>
        select(team, season, week, playoff_prob, div_winner_prob, first_seed_prob,
               eliminated, locked_in, desperate)
    }, error = function(e) {
      cat(" ERROR extracting results:", e$message, "\n")
      return(NULL)
    })
    
    if (is.null(week_probs) || nrow(week_probs) == 0) {
      next
    }
    
    all_features[[length(all_features) + 1]] <- week_probs
    cat(" done (", nrow(week_probs), "teams)\n")
  }
  
  cat("  Completed", season, "\n\n")
}

# Combine all features
if (length(all_features) == 0) {
  cat("No features generated. Exiting.\n")
  quit(status = 0)
}

playoff_features <- bind_rows(all_features)

cat("Generated", nrow(playoff_features), "team-week playoff context features\n")
print(head(playoff_features, 5))

  # Write batch to database
  cat("\nWriting to database...\n")
  dbAppendTable(con, 
                Id(schema = "mart", table = "team_playoff_context"), 
                season_features)cat("\n✓ Wrote", nrow(playoff_features), "rows to mart.team_playoff_context\n")

# Show summary statistics
cat("\nSummary statistics:\n")
summary_stats <- playoff_features |>
  summarise(
    avg_playoff_prob = mean(playoff_prob, na.rm = TRUE),
    pct_eliminated = mean(eliminated, na.rm = TRUE) * 100,
    pct_locked_in = mean(locked_in, na.rm = TRUE) * 100,
    pct_desperate = mean(desperate, na.rm = TRUE) * 100
  )
print(summary_stats)

# Show sample query
cat("\nSample query to use features:\n")
cat("
SELECT 
  g.game_id,
  g.season,
  g.week,
  g.home_team,
  g.away_team,
  hp.playoff_prob AS home_playoff_prob,
  ap.playoff_prob AS away_playoff_prob,
  hp.desperate AS home_desperate,
  ap.desperate AS away_desperate,
  hp.eliminated AS home_eliminated,
  ap.eliminated AS away_eliminated
FROM games g
LEFT JOIN mart.team_playoff_context hp 
  ON g.home_team = hp.team AND g.season = hp.season AND g.week = hp.week
LEFT JOIN mart.team_playoff_context ap 
  ON g.away_team = ap.team AND g.season = ap.season AND g.week = ap.week
WHERE g.season >= 2020
  AND g.week >= 12
LIMIT 10;
\n")

cat("\n=== Playoff context features complete! ===\n")
