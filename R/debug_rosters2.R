library(nflreadr)
library(dplyr)

cat("Loading rosters...\n")
all_rosters <- load_rosters(seasons = 2024)

cat("Converting to tibble...\n")
all_rosters <- as_tibble(all_rosters)

cat("Class:", class(all_rosters), "\n")

cat("\nTesting filter after conversion...\n")
test1 <- all_rosters %>% filter(!is.na(gsis_id))
cat("Filter worked! Rows:", nrow(test1), "\n")

cat("\nTesting full pipeline...\n")
players <- all_rosters %>%
  filter(!is.na(gsis_id)) %>%
  group_by(gsis_id) %>%
  slice(1) %>%
  ungroup() %>%
  transmute(
    player_id = gsis_id,
    player_name = full_name,
    position = position
  )

cat("Success! Players:", nrow(players), "\n")
print(head(players, 3))
