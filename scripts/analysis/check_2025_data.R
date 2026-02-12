#!/usr/bin/env Rscript
# Check 2025 NFL data availability

library(nflreadr)

cat("=== Checking 2025 NFL Data Availability ===\n")
cat("Current date:", format(Sys.Date(), "%Y-%m-%d"), "\n\n")

# Check schedules
cat("Checking schedules...\n")
tryCatch({
  sched <- load_schedules(2025)
  cat("✅ 2025 Schedule available:", nrow(sched), "games\n")
  cat("   Weeks:", min(sched$week), "to", max(sched$week), "\n")
  cat("   First game:", min(sched$gameday, na.rm=TRUE), "\n")
  cat("   Games with scores:", sum(!is.na(sched$home_score)), "\n\n")
  
  if (sum(!is.na(sched$home_score)) > 0) {
    cat("Sample completed games:\n")
    completed <- sched[!is.na(sched$home_score), c("game_id", "week", "gameday", "away_team", "home_team", "home_score", "away_score")]
    print(head(completed, 10))
  }
}, error = function(e) {
  cat("⚠️ Error loading 2025 schedule:", e$message, "\n")
})

cat("\n")

# Check play-by-play
cat("Checking play-by-play...\n")
tryCatch({
  pbp <- load_pbp(2025)
  cat("✅ 2025 Play-by-play available:", nrow(pbp), "plays\n")
  cat("   Games:", length(unique(pbp$game_id)), "\n")
}, error = function(e) {
  cat("⚠️ Error loading 2025 PBP:", e$message, "\n")
})

cat("\n")

# Check rosters
cat("Checking rosters...\n")
tryCatch({
  rosters <- load_rosters(2025)
  cat("✅ 2025 Rosters available:", nrow(rosters), "roster entries\n")
  cat("   Weeks:", min(rosters$week, na.rm=TRUE), "to", max(rosters$week, na.rm=TRUE), "\n")
}, error = function(e) {
  cat("⚠️ Error loading 2025 rosters:", e$message, "\n")
})
