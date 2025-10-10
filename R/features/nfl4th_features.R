#!/usr/bin/env Rscript
# Generate 4th down decision features using nfl4th package
#
# This script calculates coaching quality metrics based on 4th down decisions:
# - Fourth down aggression rate (go vs kick)
# - Expected points added for 4th down decisions
# - Bad decision frequency (suboptimal choices)
#
# Output: mart.team_4th_down_features table

suppressPackageStartupMessages({
  library(nflreadr)
  library(nfl4th)
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
  password = "sicillionbillions"
)

cat("Loading play-by-play data from nflfastR...\n")

# Load PBP data directly from nflfastR (2010-2024)
# This is more efficient than database and has all required columns
pbp_all <- nflreadr::load_pbp(2010:2024)

# Filter to 4th down plays with required fields
# Keep ALL columns to ensure nfl4th package has what it needs
pbp <- pbp_all %>%
  filter(
    down == 4,
    !is.na(yardline_100),
    !is.na(ydstogo),
    !is.na(posteam)
  )

cat(sprintf("Loaded %d fourth down plays (2010-2024)\n", nrow(pbp)))

# Add 4th down decision probabilities using nfl4th
cat("Adding 4th down decision probabilities...\n")

pbp_4th <- pbp %>%
  filter(!is.na(yardline_100), !is.na(ydstogo)) %>%
  # Add 4th down probabilities
  nfl4th::add_4th_probs() %>%
  mutate(
    # Identify decision type
    decision = case_when(
      grepl("field goal", desc, ignore.case = TRUE) ~ "kick_fg",
      grepl("punts", desc, ignore.case = TRUE) ~ "punt",
      play_type %in% c("pass", "run") ~ "go",
      TRUE ~ "other"
    ),
    # Calculate decision quality
    go_boost = go_wp - punt_wp,  # Advantage of going for it vs punt
    fg_boost = fg_wp - go_wp,    # Advantage of FG vs going
    # Bad decision flags
    bad_decision = case_when(
      decision == "go" & go_boost < -0.02 ~ 1,        # Should have kicked
      decision == "punt" & go_boost > 0.02 ~ 1,       # Should have gone
      decision == "kick_fg" & fg_boost < -0.02 ~ 1,   # Should have gone/punted
      TRUE ~ 0
    ),
    # Success
    success = case_when(
      decision == "go" & fourth_down_converted == 1 ~ 1,
      decision == "go" & fourth_down_failed == 1 ~ 0,
      TRUE ~ NA_real_
    )
  )

cat(sprintf("Analyzed %d fourth down decisions\n", nrow(pbp_4th)))

# Aggregate to team-game level
team_4th_features <- pbp_4th %>%
  group_by(game_id, posteam) %>%
  summarise(
    # Volume
    fourth_downs = n(),
    # Aggressiveness
    go_rate = mean(decision == "go", na.rm = TRUE),
    punt_rate = mean(decision == "punt", na.rm = TRUE),
    fg_rate = mean(decision == "kick_fg", na.rm = TRUE),
    # Decision quality
    avg_go_boost = mean(go_boost, na.rm = TRUE),
    avg_fg_boost = mean(fg_boost, na.rm = TRUE),
    bad_decisions = sum(bad_decision, na.rm = TRUE),
    bad_decision_rate = mean(bad_decision, na.rm = TRUE),
    # Success rate (when going for it)
    conversions = sum(decision == "go" & success == 1, na.rm = TRUE),
    conversion_rate = mean(success[decision == "go"], na.rm = TRUE),
    # EPA
    total_epa = sum(epa, na.rm = TRUE),
    avg_epa = mean(epa, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  # Replace NaN with 0 or NA as appropriate
  mutate(
    across(where(is.numeric), ~if_else(is.nan(.), 0, .))
  ) %>%
  # Rename posteam to team to match database schema
  rename(team = posteam)

cat(sprintf("Generated features for %d team-games\n", nrow(team_4th_features)))

# Create table if not exists
dbExecute(con, "CREATE SCHEMA IF NOT EXISTS mart")
dbExecute(con, "DROP TABLE IF EXISTS mart.team_4th_down_features")

dbExecute(con, "
  CREATE TABLE mart.team_4th_down_features (
    game_id TEXT NOT NULL,
    team TEXT NOT NULL,
    fourth_downs INT,
    go_rate REAL,
    punt_rate REAL,
    fg_rate REAL,
    avg_go_boost REAL,
    avg_fg_boost REAL,
    bad_decisions INT,
    bad_decision_rate REAL,
    conversions INT,
    conversion_rate REAL,
    total_epa REAL,
    avg_epa REAL,
    PRIMARY KEY (game_id, team)
  )
")

# Write to database
dbWriteTable(
  con,
  SQL("mart.team_4th_down_features"),
  team_4th_features,
  append = TRUE,
  row.names = FALSE
)

cat(sprintf("[DONE] Wrote %d rows to mart.team_4th_down_features\n", nrow(team_4th_features)))

# Summary statistics
summary_stats <- team_4th_features %>%
  summarise(
    avg_fourth_downs = mean(fourth_downs, na.rm = TRUE),
    avg_go_rate = mean(go_rate, na.rm = TRUE),
    avg_bad_decision_rate = mean(bad_decision_rate, na.rm = TRUE),
    avg_conversion_rate = mean(conversion_rate, na.rm = TRUE)
  )

cat("\nSummary Statistics:\n")
cat(sprintf("  Avg 4th downs per game: %.2f\n", summary_stats$avg_fourth_downs))
cat(sprintf("  Avg go rate: %.1f%%\n", summary_stats$avg_go_rate * 100))
cat(sprintf("  Avg bad decision rate: %.1f%%\n", summary_stats$avg_bad_decision_rate * 100))
cat(sprintf("  Avg conversion rate: %.1f%%\n", summary_stats$avg_conversion_rate * 100))

# Example: Teams with highest bad decision rates
bad_decision_teams <- pbp_4th %>%
  filter(bad_decision == 1) %>%
  count(posteam, season) %>%
  arrange(desc(n)) %>%
  head(10)

cat("\nTop 10 team-seasons with most bad 4th down decisions:\n")
print(bad_decision_teams)

dbDisconnect(con)
cat("\n[SUCCESS] 4th down features generated\n")
