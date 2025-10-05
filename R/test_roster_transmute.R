library(nflreadr)
library(dplyr)

rosters <- load_rosters(seasons = 2024)
cat("Loaded", nrow(rosters), "roster entries\n")
cat("Columns:", paste(names(rosters), collapse = ", "), "\n\n")

cat("Testing transmute with all columns...\n")
players <- rosters %>%
  group_by(gsis_id) %>%
  slice(1) %>%
  ungroup() %>%
  transmute(
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
  ) %>%
  filter(!is.na(player_id))

cat("Success! Created", nrow(players), "players\n")
print(head(players, 3))
