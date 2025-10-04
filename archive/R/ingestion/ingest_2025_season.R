#!/usr/bin/env Rscript
# Ingest 2025 NFL Season Data
# Run this to update with latest 2025 games, plays, and rosters

suppressPackageStartupMessages({
  library(nflreadr)
  library(dplyr)
  library(DBI)
  library(RPostgres)
})

cat("=== 2025 NFL Season Data Ingestion ===\n")
cat("Started:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")

# Database connection
conn <- dbConnect(
  RPostgres::Postgres(),
  host = Sys.getenv("POSTGRES_HOST", "localhost"),
  port = as.integer(Sys.getenv("POSTGRES_PORT", "5544")),
  dbname = Sys.getenv("POSTGRES_DB", "devdb01"),
  user = Sys.getenv("POSTGRES_USER", "dro"),
  password = Sys.getenv("POSTGRES_PASSWORD", "sicillionbillions")
)

on.exit(dbDisconnect(conn), add = TRUE)

# ============================================================
# 1. SCHEDULES (GAMES TABLE)
# ============================================================
cat("1. Loading 2025 schedules...\n")
schedules <- load_schedules(2025)

cat("   Found", nrow(schedules), "games\n")
cat("   Completed games:", sum(!is.na(schedules$home_score)), "\n")

# Transform to games table format
games_2025 <- schedules %>%
  transmute(
    game_id = game_id,
    season = season,
    week = week,
    home_team = home_team,
    away_team = away_team,
    kickoff = as.POSIXct(gameday, tz = "America/New_York"),
    spread_close = spread_line,
    total_close = total_line,
    home_score = home_score,
    away_score = away_score,
    home_moneyline = home_moneyline,
    away_moneyline = away_moneyline,
    home_spread_odds = home_spread_odds,
    away_spread_odds = away_spread_odds,
    over_odds = over_odds,
    under_odds = under_odds,
    stadium = stadium,
    roof = roof,
    surface = surface,
    away_rest = away_rest,
    home_rest = home_rest,
    away_qb_id = away_qb_id,
    home_qb_id = home_qb_id,
    away_qb_name = away_qb_name,
    home_qb_name = home_qb_name,
    away_coach = away_coach,
    home_coach = home_coach,
    referee = referee,
    stadium_id = stadium_id,
    game_type = game_type,
    overtime = overtime
  )

# Upsert (update existing, insert new)
cat("   Upserting", nrow(games_2025), "games...\n")

# Delete existing 2025 games first
dbExecute(conn, "DELETE FROM games WHERE season = 2025")

# Insert all 2025 games
dbWriteTable(conn, "games", games_2025, append = TRUE, row.names = FALSE)

cat("   ✓ Games ingested\n\n")

# ============================================================
# 2. PLAY-BY-PLAY (PLAYS TABLE)
# ============================================================
cat("2. Loading 2025 play-by-play...\n")
pbp <- load_pbp(2025)

cat("   Found", nrow(pbp), "plays from", length(unique(pbp$game_id)), "games\n")

# Keep only columns that exist in our plays table
# Note: nflreadr column names may vary, so select what's available
available_cols <- names(pbp)

plays_2025 <- pbp %>%
  transmute(
    game_id = game_id,
    play_id = play_id,
    posteam = posteam,
    defteam = defteam,
    quarter = if("qtr" %in% available_cols) qtr else if("quarter" %in% available_cols) quarter else NA_integer_,
    time_seconds = if("game_seconds_remaining" %in% available_cols) game_seconds_remaining else NA_integer_,
    down = down,
    ydstogo = ydstogo,
    epa = epa,
    pass = pass,
    rush = rush,
    # Advanced columns from backfill
    wp = wp,
    wpa = wpa,
    vegas_wp = vegas_wp,
    vegas_wpa = vegas_wpa,
    success = success,
    yards_gained = yards_gained,
    first_down = first_down,
    first_down_pass = if("first_down_pass" %in% available_cols) first_down_pass else NA_real_,
    first_down_rush = if("first_down_rush" %in% available_cols) first_down_rush else NA_real_,
    air_yards = air_yards,
    yards_after_catch = yards_after_catch,
    comp_air_epa = if("comp_air_epa" %in% available_cols) comp_air_epa else NA_real_,
    comp_yac_epa = if("comp_yac_epa" %in% available_cols) comp_yac_epa else NA_real_,
    cpoe = cpoe,
    complete_pass = complete_pass,
    incomplete_pass = incomplete_pass,
    interception = interception,
    pass_length = if("pass_length" %in% available_cols) pass_length else NA_character_,
    pass_location = pass_location,
    qb_hit = qb_hit,
    qb_scramble = qb_scramble,
    run_location = run_location,
    run_gap = run_gap,
    passer_player_id = passer_player_id,
    passer_player_name = passer_player_name,
    rusher_player_id = rusher_player_id,
    rusher_player_name = rusher_player_name,
    receiver_player_id = receiver_player_id,
    receiver_player_name = receiver_player_name,
    goal_to_go = goal_to_go,
    shotgun = shotgun,
    no_huddle = no_huddle,
    qb_dropback = qb_dropback,
    qb_kneel = qb_kneel,
    qb_spike = qb_spike,
    touchdown = touchdown,
    td_team = td_team,
    two_point_attempt = if("two_point_attempt" %in% available_cols) two_point_attempt else NA_real_,
    extra_point_result = extra_point_result,
    field_goal_result = field_goal_result,
    kick_distance = kick_distance,
    sack = sack,
    fumble = fumble,
    fumble_lost = fumble_lost,
    penalty = penalty,
    penalty_yards = penalty_yards,
    penalty_team = penalty_team,
    posteam_score = posteam_score,
    defteam_score = defteam_score,
    score_differential = score_differential,
    posteam_score_post = posteam_score_post,
    defteam_score_post = defteam_score_post,
    drive = drive,
    fixed_drive = fixed_drive,
    fixed_drive_result = fixed_drive_result
  )

cat("   Upserting", nrow(plays_2025), "plays...\n")

# Delete existing 2025 plays first
dbExecute(conn, "DELETE FROM plays WHERE game_id IN (SELECT game_id FROM games WHERE season = 2025)")

# Insert all 2025 plays
dbWriteTable(conn, "plays", plays_2025, append = TRUE, row.names = FALSE)

cat("   ✓ Plays ingested\n\n")

# ============================================================
# 3. ROSTERS
# ============================================================
cat("3. Loading 2025 rosters...\n")
rosters_raw <- load_rosters(2025)

cat("   Found", nrow(rosters_raw), "roster entries\n")
cat("   Roster columns available:", paste(head(names(rosters_raw), 15), collapse=", "), "...\n")

# Update players table (new players only)
players_2025 <- rosters_raw %>%
  dplyr::distinct(gsis_id, .keep_all = TRUE) %>%
  dplyr::filter(!is.na(gsis_id)) %>%
  dplyr::transmute(
    player_id = gsis_id,
    player_name = full_name,
    position = position,
    height = height,
    weight = weight,
    college = college,
    birth_date = as.Date(birth_date),
    rookie_year = rookie_year,
    draft_club = draft_club,
    draft_number = draft_number,
    headshot_url = headshot_url,
    status = status,
    entry_year = entry_year,
    years_exp = years_exp
  )

cat("   Upserting", nrow(players_2025), "players...\n")

# Upsert players (ON CONFLICT UPDATE)
for (i in 1:nrow(players_2025)) {
  player <- players_2025[i, ]
  dbExecute(conn, "
    INSERT INTO players (player_id, player_name, position, height, weight, college, birth_date, 
                         rookie_year, draft_club, draft_number, headshot_url, status, entry_year, years_exp)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
    ON CONFLICT (player_id) DO UPDATE SET
      player_name = EXCLUDED.player_name,
      position = EXCLUDED.position,
      status = EXCLUDED.status,
      years_exp = EXCLUDED.years_exp
  ", params = list(
    player$player_id, player$player_name, player$position, player$height, player$weight,
    player$college, player$birth_date, player$rookie_year, player$draft_club, player$draft_number,
    player$headshot_url, player$status, player$entry_year, player$years_exp
  ))
}

cat("   ✓ Players updated\n")

# Update rosters table
rosters_2025 <- rosters_raw %>%
  dplyr::filter(!is.na(gsis_id), !is.na(week)) %>%
  dplyr::distinct(season, week, team, gsis_id, .keep_all = TRUE) %>%
  dplyr::transmute(
    season = season,
    week = week,
    team = team,
    player_id = gsis_id,
    position = position,
    depth_chart_position = depth_chart_position,
    jersey_number = jersey_number,
    status = status,
    full_name = full_name,
    football_name = football_name
  )

cat("   Upserting", nrow(rosters_2025), "roster entries...\n")

# Delete existing 2025 rosters first
dbExecute(conn, "DELETE FROM rosters WHERE season = 2025")

# Insert all 2025 rosters
dbWriteTable(conn, "rosters", rosters_2025, append = TRUE, row.names = FALSE)

cat("   ✓ Rosters ingested\n\n")

# ============================================================
# 4. CALCULATE GAME STATS (TURNOVERS, PENALTIES)
# ============================================================
cat("4. Calculating game-level stats from plays...\n")

# Turnovers
dbExecute(conn, "
  UPDATE games g
  SET 
    home_turnovers = (
      SELECT COUNT(*)
      FROM plays p
      WHERE p.game_id = g.game_id
        AND p.posteam = g.home_team
        AND (p.fumble_lost = 1 OR p.interception = 1)
    ),
    away_turnovers = (
      SELECT COUNT(*)
      FROM plays p
      WHERE p.game_id = g.game_id
        AND p.posteam = g.away_team
        AND (p.fumble_lost = 1 OR p.interception = 1)
    )
  WHERE g.season = 2025
")

cat("   ✓ Turnovers calculated\n")

# Penalties
dbExecute(conn, "
  UPDATE games g
  SET 
    home_penalties = (
      SELECT COUNT(*)
      FROM plays p
      WHERE p.game_id = g.game_id
        AND p.penalty_team = g.home_team
        AND p.penalty = 1
    ),
    away_penalties = (
      SELECT COUNT(*)
      FROM plays p
      WHERE p.game_id = g.game_id
        AND p.penalty_team = g.away_team
        AND p.penalty = 1
    ),
    home_penalty_yards = (
      SELECT COALESCE(SUM(p.penalty_yards), 0)
      FROM plays p
      WHERE p.game_id = g.game_id
        AND p.penalty_team = g.home_team
        AND p.penalty = 1
    ),
    away_penalty_yards = (
      SELECT COALESCE(SUM(p.penalty_yards), 0)
      FROM plays p
      WHERE p.game_id = g.game_id
        AND p.penalty_team = g.away_team
        AND p.penalty = 1
    )
  WHERE g.season = 2025
")

cat("   ✓ Penalties calculated\n\n")

# ============================================================
# 5. UPDATE MART TABLES
# ============================================================
cat("5. Updating mart tables...\n")

# Recalculate team EPA for 2025 games
dbExecute(conn, "DELETE FROM mart.team_epa WHERE game_id IN (SELECT game_id FROM games WHERE season = 2025)")

dbExecute(conn, "
  INSERT INTO mart.team_epa (game_id, posteam, plays, epa_sum, epa_mean, explosive_pass, explosive_rush)
  SELECT 
    game_id,
    posteam,
    COUNT(*) as plays,
    COALESCE(SUM(epa), 0) as epa_sum,
    COALESCE(AVG(epa), 0) as epa_mean,
    SUM(CASE WHEN pass = true AND yards_gained >= 20 THEN 1 ELSE 0 END) as explosive_pass,
    SUM(CASE WHEN rush = true AND yards_gained >= 10 THEN 1 ELSE 0 END) as explosive_rush
  FROM plays
  WHERE posteam IS NOT NULL
    AND game_id IN (SELECT game_id FROM games WHERE season = 2025)
  GROUP BY game_id, posteam
")

cat("   ✓ mart.team_epa updated\n")

# Refresh materialized views
cat("   Refreshing materialized views...\n")
dbExecute(conn, "REFRESH MATERIALIZED VIEW mart.game_weather")
dbExecute(conn, "REFRESH MATERIALIZED VIEW mart.game_summary")
dbExecute(conn, "REFRESH MATERIALIZED VIEW mart.game_features_enhanced")

cat("   ✓ Materialized views refreshed\n\n")

# ============================================================
# SUMMARY
# ============================================================
cat("=== Ingestion Complete ===\n")

summary <- dbGetQuery(conn, "
  SELECT 
    'Games' as table_name,
    COUNT(*) as total_rows,
    SUM(CASE WHEN season = 2025 THEN 1 ELSE 0 END) as season_2025
  FROM games
  UNION ALL
  SELECT 
    'Plays',
    COUNT(*),
    SUM(CASE WHEN game_id IN (SELECT game_id FROM games WHERE season = 2025) THEN 1 ELSE 0 END)
  FROM plays
  UNION ALL
  SELECT 
    'Rosters',
    COUNT(*),
    SUM(CASE WHEN season = 2025 THEN 1 ELSE 0 END)
  FROM rosters
  UNION ALL
  SELECT 
    'Players',
    COUNT(*),
    COUNT(*)  -- All players (not season-specific)
  FROM players
  ORDER BY table_name
")

print(summary)

cat("\n✅ 2025 data ingestion complete!\n")
cat("Completed:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
