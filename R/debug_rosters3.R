library(nflreadr)
library(dplyr)

cat("Loading rosters...\n")
all_rosters <- load_rosters(seasons = 2024)

cat("Testing direct column access...\n")
cat("Has gsis_id column:", "gsis_id" %in% names(all_rosters), "\n")

cat("\nTesting bracket notation...\n")
ids <- all_rosters[["gsis_id"]]
cat("Extracted", length(ids), "IDs\n")
cat("NAs:", sum(is.na(ids)), "\n")

cat("\nTesting $ notation...\n")
ids2 <- all_rosters$gsis_id
cat("Extracted", length(ids2), "IDs\n")

cat("\nTesting dplyr with .data pronoun...\n")
test <- all_rosters %>% filter(!is.na(.data$gsis_id))
cat("Filter with .data worked! Rows:", nrow(test), "\n")
