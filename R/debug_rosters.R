library(nflreadr)
library(dplyr)

cat("Loading rosters...\n")
all_rosters <- load_rosters(seasons = 2024)

cat("Class:", class(all_rosters), "\n")
cat("Dimensions:", nrow(all_rosters), "x", ncol(all_rosters), "\n")
cat("Column names:\n")
print(names(all_rosters))

cat("\nTesting filter...\n")
test1 <- all_rosters %>% filter(!is.na(gsis_id))
cat("Filter worked! Rows:", nrow(test1), "\n")

cat("\nTesting group_by...\n")
test2 <- test1 %>% group_by(gsis_id) %>% slice(1) %>% ungroup()
cat("Group_by worked! Unique players:", nrow(test2), "\n")

cat("\nTesting transmute...\n")
test3 <- test2 %>% transmute(player_id = gsis_id, player_name = full_name)
cat("Transmute worked! Players:", nrow(test3), "\n")
print(head(test3, 3))
