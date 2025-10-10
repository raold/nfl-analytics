#!/usr/bin/env Rscript
# Generate injury load features from injury reports
#
# This script quantifies team health/injury burden:
# - Total injuries by position group
# - Injury severity (Out > Doubtful > Questionable)
# - Key position injuries (QB, OL, DL weighted higher)
# - Injury differential (home vs away)
#
# Output: mart.team_injury_load table

suppressPackageStartupMessages({
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

cat("Loading injury data from database...\n")

# Load injuries
injuries <- dbGetQuery(con, "
  SELECT
    season,
    week,
    team,
    full_name as player_name,
    position,
    report_status as injury_status
  FROM injuries
  WHERE game_type = 'REG'
")

cat(sprintf("Loaded %d injury records\n", nrow(injuries)))

# Define position importance weights
# QB and OL are most critical, followed by pass rushers, skill positions
position_weights <- data.frame(
  position_group = c("QB", "OL", "DL", "LB", "WR", "RB", "TE", "DB", "K", "LS", "P"),
  weight = c(3.0, 2.0, 1.5, 1.2, 1.3, 1.0, 1.1, 1.0, 0.3, 0.1, 0.2)
)

# Map positions to groups
injuries_grouped <- injuries %>%
  mutate(
    position_group = case_when(
      position == "QB" ~ "QB",
      position %in% c("T", "G", "C", "OL", "OT", "OG") ~ "OL",
      position %in% c("DE", "DT", "NT", "DL") ~ "DL",
      position %in% c("ILB", "OLB", "LB", "MLB") ~ "LB",
      position == "WR" ~ "WR",
      position %in% c("RB", "FB", "HB") ~ "RB",
      position == "TE" ~ "TE",
      position %in% c("CB", "S", "SS", "FS", "DB") ~ "DB",
      position == "K" ~ "K",
      position == "LS" ~ "LS",
      position == "P" ~ "P",
      TRUE ~ "OTHER"
    ),
    # Severity scoring (Out=3, Doubtful=2, Questionable=1)
    severity = case_when(
      grepl("Out", injury_status, ignore.case = TRUE) ~ 3,
      grepl("Doubtful", injury_status, ignore.case = TRUE) ~ 2,
      grepl("Questionable|Probable", injury_status, ignore.case = TRUE) ~ 1,
      TRUE ~ 0
    )
  ) %>%
  left_join(position_weights, by = "position_group") %>%
  mutate(
    weight = ifelse(is.na(weight), 0.5, weight),  # Default weight for OTHER
    weighted_severity = severity * weight
  )

cat("Aggregating injury load by team-week...\n")

# Aggregate to team-week level
team_injury_load <- injuries_grouped %>%
  group_by(season, week, team) %>%
  summarise(
    # Raw counts
    total_injuries = n(),
    injuries_out = sum(severity == 3, na.rm = TRUE),
    injuries_doubtful = sum(severity == 2, na.rm = TRUE),
    injuries_questionable = sum(severity == 1, na.rm = TRUE),

    # Position-specific
    qb_injuries = sum(position_group == "QB" & severity > 0, na.rm = TRUE),
    ol_injuries = sum(position_group == "OL" & severity > 0, na.rm = TRUE),
    dl_injuries = sum(position_group == "DL" & severity > 0, na.rm = TRUE),
    skill_injuries = sum(position_group %in% c("WR", "RB", "TE") & severity > 0, na.rm = TRUE),

    # Weighted severity score
    injury_load_weighted = sum(weighted_severity, na.rm = TRUE),

    .groups = "drop"
  )

cat(sprintf("Generated injury load for %d team-weeks\n", nrow(team_injury_load)))

# Join with games to create game-level features
cat("Joining injury load with games...\n")

games <- dbGetQuery(con, "
  SELECT
    game_id,
    season,
    week,
    home_team,
    away_team
  FROM games
")

cat(sprintf("Loaded %d games\n", nrow(games)))

# Create home and away features
home_injury_features <- team_injury_load %>%
  rename(
    home_total_injuries = total_injuries,
    home_injuries_out = injuries_out,
    home_injuries_doubtful = injuries_doubtful,
    home_injuries_questionable = injuries_questionable,
    home_qb_injuries = qb_injuries,
    home_ol_injuries = ol_injuries,
    home_dl_injuries = dl_injuries,
    home_skill_injuries = skill_injuries,
    home_injury_load = injury_load_weighted
  ) %>%
  select(season, week, team, starts_with("home_"))

away_injury_features <- team_injury_load %>%
  rename(
    away_total_injuries = total_injuries,
    away_injuries_out = injuries_out,
    away_injuries_doubtful = injuries_doubtful,
    away_injuries_questionable = injuries_questionable,
    away_qb_injuries = qb_injuries,
    away_ol_injuries = ol_injuries,
    away_dl_injuries = dl_injuries,
    away_skill_injuries = skill_injuries,
    away_injury_load = injury_load_weighted
  ) %>%
  select(season, week, team, starts_with("away_"))

# Join with games
game_injury_features <- games %>%
  left_join(home_injury_features, by = c("season", "week", "home_team" = "team")) %>%
  left_join(away_injury_features, by = c("season", "week", "away_team" = "team")) %>%
  mutate(
    # Replace NA with 0 (no injuries)
    across(starts_with("home_") | starts_with("away_"), ~ifelse(is.na(.), 0, .)),

    # Derived features
    injury_load_diff = home_injury_load - away_injury_load,
    total_injuries_diff = home_total_injuries - away_total_injuries,
    qb_injury_diff = home_qb_injuries - away_qb_injuries
  )

cat(sprintf("Generated injury features for %d games\n", nrow(game_injury_features)))

# Create table
dbExecute(con, "CREATE SCHEMA IF NOT EXISTS mart")
dbExecute(con, "DROP TABLE IF EXISTS mart.team_injury_load")

dbExecute(con, "
  CREATE TABLE mart.team_injury_load (
    game_id TEXT NOT NULL PRIMARY KEY,
    season INT,
    week INT,
    home_total_injuries INT,
    away_total_injuries INT,
    home_injuries_out INT,
    away_injuries_out INT,
    home_injuries_doubtful INT,
    away_injuries_doubtful INT,
    home_injuries_questionable INT,
    away_injuries_questionable INT,
    home_qb_injuries INT,
    away_qb_injuries INT,
    home_ol_injuries INT,
    away_ol_injuries INT,
    home_dl_injuries INT,
    away_dl_injuries INT,
    home_skill_injuries INT,
    away_skill_injuries INT,
    home_injury_load REAL,
    away_injury_load REAL,
    injury_load_diff REAL,
    total_injuries_diff INT,
    qb_injury_diff INT
  )
")

# Write to database
dbWriteTable(
  con,
  SQL("mart.team_injury_load"),
  game_injury_features %>% select(-home_team, -away_team),
  append = TRUE,
  row.names = FALSE
)

cat(sprintf("[DONE] Wrote %d rows to mart.team_injury_load\n", nrow(game_injury_features)))

# Summary statistics
summary_stats <- game_injury_features %>%
  summarise(
    avg_home_injuries = mean(home_total_injuries, na.rm = TRUE),
    avg_away_injuries = mean(away_total_injuries, na.rm = TRUE),
    games_with_qb_injury = sum(home_qb_injuries > 0 | away_qb_injuries > 0, na.rm = TRUE),
    avg_injury_load_diff = mean(abs(injury_load_diff), na.rm = TRUE)
  )

cat("\nSummary Statistics:\n")
cat(sprintf("  Avg injuries per team (home): %.2f\n", summary_stats$avg_home_injuries))
cat(sprintf("  Avg injuries per team (away): %.2f\n", summary_stats$avg_away_injuries))
cat(sprintf("  Games with QB injury: %d (%.1f%%)\n",
            summary_stats$games_with_qb_injury,
            summary_stats$games_with_qb_injury / nrow(game_injury_features) * 100))
cat(sprintf("  Avg injury load differential: %.2f\n", summary_stats$avg_injury_load_diff))

dbDisconnect(con)
cat("\n[SUCCESS] Injury load features generated\n")
