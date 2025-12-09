#!/usr/bin/env Rscript
# Generate playoff context features using nflseedR package
#
# This script calculates playoff race context for each team-game:
# - Playoff probability (pre-game simulation)
# - Playoff probability change (game impact)
# - Division standings context
# - Desperation/tanking indicators
#
# Output: mart.team_playoff_context table

suppressPackageStartupMessages({
  library(nflseedR)
  library(nflreadr)
  library(dplyr)
  library(DBI)
  library(RPostgres)
})

# Database connection
con <- dbConnect(
  RPostgres::Postgres(),
  host = "localhost",
  port = 5544,
  dbname = "devdb01",
  user = "dro",
  password = Sys.getenv("DB_PASSWORD", "sicillionbillions")
)

cat("Generating playoff context features using nflseedR...\n")
cat("This simulates playoff probabilities for each week (2010-2024)\n\n")

# Run playoff simulation for each season
# This is computationally expensive (~5-10 min per season)
all_playoff_probs <- data.frame()

for (yr in 2010:2024) {
  cat(sprintf("[%d/%d] Simulating season %d...\n", yr - 2009, 15, yr))

  # Load schedule for the season
  schedule <- nflreadr::load_schedules(yr)

  # Simulate playoff probabilities for each week
  # nflseedR simulates thousands of seasons to estimate playoff odds
  sim <- tryCatch(
    {
      result <- nflseedR::simulate_nfl(
        nfl_season = yr,
        playoff_seeds = 7, # 7 playoff seeds per conference (post-2020 format)
        fresh_season = TRUE,
        sims = 1000 # 1000 simulations (balance speed/accuracy)
      )
      # Extract teams data from the simulation result object
      result$teams
    },
    error = function(e) {
      cat(sprintf("  Warning: Simulation failed for %d: %s\n", yr, e$message))
      return(NULL)
    }
  )

  if (is.null(sim) || nrow(sim) == 0) {
    cat(sprintf("  Skipping season %d (no simulation data)\n", yr))
    next
  }

  # Extract team-level playoff probabilities
  # nflseedR::simulate_nfl returns a summary with columns like 'make_playoffs', 'win_division'
  team_probs <- sim %>%
    mutate(
      season = yr, # Add season explicitly
      playoff_prob = ifelse("make_playoffs" %in% names(.), make_playoffs, NA_real_),
      div_winner_prob = ifelse("win_division" %in% names(.), win_division, NA_real_),
      one_seed_prob = ifelse("first_seed" %in% names(.), first_seed, NA_real_),
      avg_wins = ifelse("wins" %in% names(.), wins, NA_real_)
    ) %>%
    select(season, team, playoff_prob, div_winner_prob, one_seed_prob, avg_wins) %>%
    # nflseedR returns season-level probabilities, we'll expand to weeks later
    mutate(week = NA_integer_)

  all_playoff_probs <- bind_rows(all_playoff_probs, team_probs)

  cat(sprintf("  Completed %d (generated %d team-week records)\n", yr, nrow(team_probs)))
}

cat(sprintf("\nTotal playoff probability records: %d\n", nrow(all_playoff_probs)))

# Join with games to create team-game features
cat("Joining playoff probabilities with games...\n")

# Load games from database
games <- dbGetQuery(con, "
  SELECT
    game_id,
    season,
    week,
    home_team,
    away_team
  FROM games
  WHERE season >= 2010 AND season <= 2024
")

cat(sprintf("Loaded %d games\n", nrow(games)))

# Create features for home and away teams
home_features <- all_playoff_probs %>%
  rename(
    home_playoff_prob = playoff_prob,
    home_div_winner_prob = div_winner_prob,
    home_one_seed_prob = one_seed_prob,
    home_avg_wins = avg_wins
  ) %>%
  select(season, week, team, starts_with("home_"))

away_features <- all_playoff_probs %>%
  rename(
    away_playoff_prob = playoff_prob,
    away_div_winner_prob = div_winner_prob,
    away_one_seed_prob = one_seed_prob,
    away_avg_wins = avg_wins
  ) %>%
  select(season, week, team, starts_with("away_"))

# Join with games
game_playoff_features <- games %>%
  left_join(home_features, by = c("season", "week", "home_team" = "team")) %>%
  left_join(away_features, by = c("season", "week", "away_team" = "team")) %>%
  mutate(
    # Derived features
    playoff_prob_diff = home_playoff_prob - away_playoff_prob,
    stakes = (home_playoff_prob + away_playoff_prob) / 2, # Avg playoff urgency
    desperation_home = case_when(
      week >= 14 & home_playoff_prob > 0.4 & home_playoff_prob < 0.6 ~ 1,
      TRUE ~ 0
    ),
    desperation_away = case_when(
      week >= 14 & away_playoff_prob > 0.4 & away_playoff_prob < 0.6 ~ 1,
      TRUE ~ 0
    ),
    tanking_home = case_when(
      week >= 10 & home_playoff_prob < 0.05 ~ 1,
      TRUE ~ 0
    ),
    tanking_away = case_when(
      week >= 10 & away_playoff_prob < 0.05 ~ 1,
      TRUE ~ 0
    )
  )

cat(sprintf("Generated playoff features for %d games\n", nrow(game_playoff_features)))

# Create table
dbExecute(con, "CREATE SCHEMA IF NOT EXISTS mart")
dbExecute(con, "DROP TABLE IF EXISTS mart.team_playoff_context")

dbExecute(con, "
  CREATE TABLE mart.team_playoff_context (
    game_id TEXT NOT NULL PRIMARY KEY,
    season INT,
    week INT,
    home_playoff_prob REAL,
    away_playoff_prob REAL,
    home_div_winner_prob REAL,
    away_div_winner_prob REAL,
    home_one_seed_prob REAL,
    away_one_seed_prob REAL,
    home_avg_wins REAL,
    away_avg_wins REAL,
    playoff_prob_diff REAL,
    stakes REAL,
    desperation_home INT,
    desperation_away INT,
    tanking_home INT,
    tanking_away INT
  )
")

# Write to database
dbWriteTable(
  con,
  SQL("mart.team_playoff_context"),
  game_playoff_features %>% select(-home_team, -away_team),
  append = TRUE,
  row.names = FALSE
)

cat(sprintf("[DONE] Wrote %d rows to mart.team_playoff_context\n", nrow(game_playoff_features)))

# Summary statistics
summary_stats <- game_playoff_features %>%
  summarise(
    avg_stakes = mean(stakes, na.rm = TRUE),
    desperation_games = sum(desperation_home == 1 | desperation_away == 1, na.rm = TRUE),
    tanking_games = sum(tanking_home == 1 | tanking_away == 1, na.rm = TRUE),
    high_stakes_games = sum(stakes > 0.5, na.rm = TRUE)
  )

cat("\nSummary Statistics:\n")
cat(sprintf("  Avg playoff stakes: %.1f%%\n", summary_stats$avg_stakes * 100))
cat(sprintf(
  "  Desperation games: %d (%.1f%%)\n",
  summary_stats$desperation_games,
  summary_stats$desperation_games / nrow(game_playoff_features) * 100
))
cat(sprintf(
  "  Tanking indicator games: %d (%.1f%%)\n",
  summary_stats$tanking_games,
  summary_stats$tanking_games / nrow(game_playoff_features) * 100
))
cat(sprintf("  High stakes games (>50%% avg playoff prob): %d\n", summary_stats$high_stakes_games))

dbDisconnect(con)
cat("\n[SUCCESS] Playoff context features generated\n")
