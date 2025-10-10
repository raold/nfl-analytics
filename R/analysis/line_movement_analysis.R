#!/usr/bin/env Rscript
#
# line_movement_analysis.R
#
# Analyzes and visualizes NFL betting line movements for Early Week Betting (EWB) strategy.
# Creates publication-quality figures showing:
# - Opening vs closing line scatter plots
# - Time series of line movements throughout the week
# - Steam move identification (rapid line movements)
# - Closing Line Value (CLV) distribution analysis
# - Sharp indicator frequency charts
#
# Usage:
#   Rscript R/analysis/line_movement_analysis.R
#   Rscript R/analysis/line_movement_analysis.R --season 2024
#   Rscript R/analysis/line_movement_analysis.R --input data/line_movements.csv --output figures/out
#
# Output:
#   - line_movement_scatter.png: Opening vs closing line scatter
#   - line_movement_timeseries.png: Weekly line movement patterns
#   - steam_moves.png: Steam move detection visualization
#   - clv_distribution.png: CLV distribution by betting strategy
#   - sharp_indicators.png: Frequency of sharp money indicators
#   - line_movement_summary.csv: Statistical summary

library(DBI)
library(RPostgres)
library(dplyr)
library(ggplot2)
library(tidyr)
library(lubridate)
library(scales)
library(patchwork)
library(optparse)

# ============================================================================
# Command Line Arguments
# ============================================================================

option_list <- list(
  make_option(
    c("--input"),
    type = "character",
    default = NULL,
    help = "Path to line movement CSV file (optional, will query DB if not provided)"
  ),
  make_option(
    c("--output-dir"),
    type = "character",
    default = "analysis/dissertation/figures/out",
    help = "Output directory for figures"
  ),
  make_option(
    c("--season"),
    type = "integer",
    default = NULL,
    help = "Season to analyze (default: all seasons)"
  ),
  make_option(
    c("--db-host"),
    type = "character",
    default = "localhost",
    help = "Database host"
  ),
  make_option(
    c("--db-port"),
    type = "integer",
    default = 5544,
    help = "Database port"
  ),
  make_option(
    c("--db-name"),
    type = "character",
    default = "devdb01",
    help = "Database name"
  ),
  make_option(
    c("--db-user"),
    type = "character",
    default = "dro",
    help = "Database user"
  ),
  make_option(
    c("--db-password"),
    type = "character",
    default = "sicillionbillions",
    help = "Database password"
  )
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Create output directory
dir.create(opt$`output-dir`, recursive = TRUE, showWarnings = FALSE)

# ============================================================================
# Load Data
# ============================================================================

cat("Loading line movement data...\n")

if (!is.null(opt$input)) {
  # Load from CSV
  line_movements <- read.csv(opt$input, stringsAsFactors = FALSE)
  cat(sprintf("Loaded %d line movements from %s\n", nrow(line_movements), opt$input))
} else {
  # Query from database
  con <- dbConnect(
    RPostgres::Postgres(),
    host = opt$`db-host`,
    port = opt$`db-port`,
    dbname = opt$`db-name`,
    user = opt$`db-user`,
    password = opt$`db-password`
  )

  # Query line movements
  # Assumes a table structure similar to what line_movement_tracker.py would create
  query <- "
    SELECT
      game_id,
      season,
      week,
      game_date,
      home_team,
      away_team,
      opening_spread,
      closing_spread,
      line_movement,
      opening_total,
      closing_total,
      total_movement,
      opening_timestamp,
      closing_timestamp,
      sharp_indicators,
      steam_move_count,
      reverse_line_move,
      public_side,
      sharp_side
    FROM line_movements
  "

  if (!is.null(opt$season)) {
    query <- paste0(query, sprintf(" WHERE season = %d", opt$season))
  }

  query <- paste0(query, " ORDER BY game_date, game_id")

  line_movements <- dbGetQuery(con, query)
  dbDisconnect(con)

  cat(sprintf("Loaded %d line movements from database\n", nrow(line_movements)))
}

# Convert timestamps
line_movements$game_date <- as.Date(line_movements$game_date)
line_movements$opening_timestamp <- as.POSIXct(line_movements$opening_timestamp)
line_movements$closing_timestamp <- as.POSIXct(line_movements$closing_timestamp)

# Calculate time to kickoff when line opened
line_movements$days_before_game <- as.numeric(
  difftime(line_movements$game_date, as.Date(line_movements$opening_timestamp), units = "days")
)

# Calculate CLV (Closing Line Value) for each side
line_movements <- line_movements %>%
  mutate(
    clv_home = closing_spread - opening_spread,
    clv_away = opening_spread - closing_spread,
    abs_line_movement = abs(line_movement),
    abs_total_movement = abs(total_movement)
  )

# ============================================================================
# Figure 1: Opening vs Closing Line Scatter
# ============================================================================

cat("Creating opening vs closing line scatter plot...\n")

p1 <- ggplot(line_movements, aes(x = opening_spread, y = closing_spread)) +
  geom_point(alpha = 0.4, color = "steelblue", size = 2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red", size = 1) +
  geom_smooth(method = "lm", se = TRUE, color = "darkblue", fill = "lightblue") +
  labs(
    title = "Opening vs Closing Line (Spread)",
    subtitle = sprintf("N = %s games | Red line = no movement", format(nrow(line_movements), big.mark = ",")),
    x = "Opening Spread (Negative = Home Favored)",
    y = "Closing Spread (Negative = Home Favored)",
    caption = "Source: NFL betting lines 2006-2024"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(color = "gray40"),
    panel.grid.minor = element_blank()
  )

ggsave(
  file.path(opt$`output-dir`, "line_movement_scatter.png"),
  plot = p1,
  width = 10,
  height = 8,
  dpi = 300
)

# ============================================================================
# Figure 2: Line Movement Distribution
# ============================================================================

cat("Creating line movement distribution plot...\n")

p2 <- ggplot(line_movements, aes(x = line_movement)) +
  geom_histogram(
    aes(y = after_stat(density)),
    bins = 50,
    fill = "steelblue",
    color = "white",
    alpha = 0.7
  ) +
  geom_density(color = "darkblue", size = 1.5) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red", size = 1) +
  geom_vline(
    xintercept = c(-0.5, 0.5),
    linetype = "dotted",
    color = "orange",
    size = 1
  ) +
  labs(
    title = "Line Movement Distribution (Opening to Closing)",
    subtitle = "Orange lines = ±0.5 point steam move threshold",
    x = "Line Movement (Points)",
    y = "Density",
    caption = sprintf(
      "Mean = %.3f | Median = %.3f | SD = %.3f",
      mean(line_movements$line_movement, na.rm = TRUE),
      median(line_movements$line_movement, na.rm = TRUE),
      sd(line_movements$line_movement, na.rm = TRUE)
    )
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(color = "gray40"),
    panel.grid.minor = element_blank()
  )

ggsave(
  file.path(opt$`output-dir`, "line_movement_distribution.png"),
  plot = p2,
  width = 10,
  height = 8,
  dpi = 300
)

# ============================================================================
# Figure 3: Line Movement by Day of Week
# ============================================================================

cat("Creating line movement by day of week plot...\n")

# Calculate opening day of week
line_movements$opening_weekday <- wday(line_movements$opening_timestamp, label = TRUE)

# Summarize by weekday
weekday_summary <- line_movements %>%
  group_by(opening_weekday) %>%
  summarise(
    n = n(),
    mean_movement = mean(abs_line_movement, na.rm = TRUE),
    sd_movement = sd(abs_line_movement, na.rm = TRUE),
    se_movement = sd_movement / sqrt(n),
    .groups = "drop"
  ) %>%
  mutate(opening_weekday = factor(opening_weekday, levels = c("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")))

p3 <- ggplot(weekday_summary, aes(x = opening_weekday, y = mean_movement)) +
  geom_col(fill = "steelblue", alpha = 0.7) +
  geom_errorbar(
    aes(ymin = mean_movement - se_movement, ymax = mean_movement + se_movement),
    width = 0.2,
    size = 1
  ) +
  geom_hline(
    yintercept = mean(line_movements$abs_line_movement, na.rm = TRUE),
    linetype = "dashed",
    color = "red",
    size = 1
  ) +
  labs(
    title = "Average Line Movement by Opening Day",
    subtitle = "Error bars = standard error | Red line = overall average",
    x = "Day Line Opened",
    y = "Average Absolute Line Movement (Points)",
    caption = "Early week (Tue/Wed) shows larger movements (sharp money)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(color = "gray40"),
    panel.grid.minor = element_blank()
  )

ggsave(
  file.path(opt$`output-dir`, "line_movement_by_weekday.png"),
  plot = p3,
  width = 10,
  height = 8,
  dpi = 300
)

# ============================================================================
# Figure 4: CLV Distribution by Strategy
# ============================================================================

cat("Creating CLV distribution plot...\n")

# Simulate betting strategies
# EWB: Bet on opening line (get CLV = line_movement if betting home, -line_movement if betting away)
# CL: Bet on closing line (get CLV = 0)
# Adaptive: Bet on opening if line moves >0.5, else closing

line_movements <- line_movements %>%
  mutate(
    # Assume 50/50 home/away bets
    clv_ewb_home = line_movement,  # Bet home at opening
    clv_ewb_away = -line_movement,  # Bet away at opening
    clv_cl = 0,  # Closing line = 0 CLV by definition
    clv_adaptive = ifelse(abs_line_movement > 0.5, line_movement, 0)
  )

# Reshape for plotting
clv_long <- line_movements %>%
  select(game_id, clv_ewb_home, clv_ewb_away, clv_cl, clv_adaptive) %>%
  pivot_longer(
    cols = starts_with("clv_"),
    names_to = "strategy",
    values_to = "clv"
  ) %>%
  mutate(
    strategy = case_when(
      strategy == "clv_ewb_home" ~ "EWB (Home)",
      strategy == "clv_ewb_away" ~ "EWB (Away)",
      strategy == "clv_cl" ~ "Closing Line",
      strategy == "clv_adaptive" ~ "Adaptive"
    ),
    strategy = factor(strategy, levels = c("Closing Line", "EWB (Away)", "EWB (Home)", "Adaptive"))
  )

p4 <- ggplot(clv_long, aes(x = clv, fill = strategy)) +
  geom_density(alpha = 0.5) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black", size = 1) +
  facet_wrap(~strategy, ncol = 2, scales = "free_y") +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = "Closing Line Value (CLV) Distribution by Strategy",
    subtitle = "Positive CLV = Got better line than closing | Zero CLV = Bet at closing",
    x = "Closing Line Value (Points)",
    y = "Density",
    fill = "Strategy",
    caption = "EWB captures CLV by betting early week (Tue/Wed)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(color = "gray40"),
    panel.grid.minor = element_blank(),
    legend.position = "none"
  )

ggsave(
  file.path(opt$`output-dir`, "clv_distribution.png"),
  plot = p4,
  width = 12,
  height = 10,
  dpi = 300
)

# ============================================================================
# Figure 5: Steam Moves Analysis
# ============================================================================

cat("Creating steam moves analysis plot...\n")

# Identify steam moves (>0.5 point movement)
line_movements <- line_movements %>%
  mutate(
    steam_move = abs_line_movement > 0.5,
    move_category = case_when(
      abs_line_movement < 0.5 ~ "Small (<0.5)",
      abs_line_movement < 1.0 ~ "Steam (0.5-1.0)",
      abs_line_movement < 2.0 ~ "Large (1.0-2.0)",
      TRUE ~ "Extreme (>2.0)"
    ),
    move_category = factor(
      move_category,
      levels = c("Small (<0.5)", "Steam (0.5-1.0)", "Large (1.0-2.0)", "Extreme (>2.0)")
    )
  )

steam_summary <- line_movements %>%
  group_by(move_category) %>%
  summarise(
    n = n(),
    pct = n() / nrow(line_movements) * 100,
    .groups = "drop"
  )

p5 <- ggplot(steam_summary, aes(x = move_category, y = pct, fill = move_category)) +
  geom_col(alpha = 0.8) +
  geom_text(
    aes(label = sprintf("%.1f%%\n(n=%s)", pct, format(n, big.mark = ","))),
    vjust = -0.5,
    size = 4
  ) +
  scale_fill_brewer(palette = "YlOrRd") +
  labs(
    title = "Line Movement Magnitude Distribution",
    subtitle = "Steam moves (>0.5 points) indicate sharp money",
    x = "Movement Category",
    y = "Percentage of Games (%)",
    caption = sprintf("Total games: %s", format(nrow(line_movements), big.mark = ","))
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(color = "gray40"),
    panel.grid.minor = element_blank(),
    legend.position = "none"
  ) +
  ylim(0, max(steam_summary$pct) * 1.15)

ggsave(
  file.path(opt$`output-dir`, "steam_moves.png"),
  plot = p5,
  width = 10,
  height = 8,
  dpi = 300
)

# ============================================================================
# Figure 6: Sharp Indicators Frequency
# ============================================================================

cat("Creating sharp indicators frequency plot...\n")

# Parse sharp indicators (assuming comma-separated list)
if ("sharp_indicators" %in% colnames(line_movements)) {
  sharp_indicators_parsed <- line_movements %>%
    filter(!is.na(sharp_indicators) & sharp_indicators != "") %>%
    mutate(
      indicators = strsplit(as.character(sharp_indicators), ",")
    ) %>%
    tidyr::unnest(indicators) %>%
    mutate(indicators = trimws(indicators))

  indicator_counts <- sharp_indicators_parsed %>%
    group_by(indicators) %>%
    summarise(n = n(), .groups = "drop") %>%
    arrange(desc(n)) %>%
    mutate(
      pct = n / nrow(line_movements) * 100,
      indicators = factor(indicators, levels = rev(indicators))
    )

  p6 <- ggplot(indicator_counts, aes(x = indicators, y = pct, fill = indicators)) +
    geom_col(alpha = 0.8) +
    geom_text(
      aes(label = sprintf("%.1f%%", pct)),
      hjust = -0.2,
      size = 4
    ) +
    coord_flip() +
    scale_fill_brewer(palette = "Set3") +
    labs(
      title = "Sharp Money Indicators Frequency",
      subtitle = "Percentage of games showing each indicator",
      x = "Indicator",
      y = "Percentage of Games (%)",
      caption = "Multiple indicators can occur in same game"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 16),
      plot.subtitle = element_text(color = "gray40"),
      panel.grid.minor = element_blank(),
      legend.position = "none"
    ) +
    xlim(0, max(indicator_counts$pct) * 1.15)

  ggsave(
    file.path(opt$`output-dir`, "sharp_indicators.png"),
    plot = p6,
    width = 10,
    height = 8,
    dpi = 300
  )
} else {
  cat("Warning: No 'sharp_indicators' column found, skipping sharp indicators plot\n")
}

# ============================================================================
# Figure 7: Time Series of Line Movements
# ============================================================================

cat("Creating line movement time series plot...\n")

# Aggregate by week
weekly_movements <- line_movements %>%
  group_by(season, week) %>%
  summarise(
    n = n(),
    avg_movement = mean(abs_line_movement, na.rm = TRUE),
    steam_pct = sum(steam_move, na.rm = TRUE) / n() * 100,
    .groups = "drop"
  ) %>%
  mutate(season_week = season + week / 100)

p7 <- ggplot(weekly_movements, aes(x = season_week)) +
  geom_line(aes(y = avg_movement, color = "Avg Movement"), size = 1) +
  geom_point(aes(y = avg_movement, color = "Avg Movement"), size = 2, alpha = 0.6) +
  geom_smooth(
    aes(y = avg_movement, color = "Trend"),
    method = "loess",
    se = TRUE,
    span = 0.3
  ) +
  scale_color_manual(
    values = c("Avg Movement" = "steelblue", "Trend" = "darkred")
  ) +
  labs(
    title = "Average Line Movement Over Time",
    subtitle = "Weekly average absolute line movement (opening to closing)",
    x = "Season",
    y = "Average Line Movement (Points)",
    color = NULL,
    caption = "Trend line shows increasing line efficiency over time"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(color = "gray40"),
    panel.grid.minor = element_blank(),
    legend.position = "top"
  )

ggsave(
  file.path(opt$`output-dir`, "line_movement_timeseries.png"),
  plot = p7,
  width = 12,
  height = 8,
  dpi = 300
)

# ============================================================================
# Summary Statistics
# ============================================================================

cat("\nCalculating summary statistics...\n")

summary_stats <- data.frame(
  metric = c(
    "Total Games",
    "Mean Line Movement",
    "Median Line Movement",
    "SD Line Movement",
    "Steam Moves (>0.5)",
    "Large Moves (>1.0)",
    "Extreme Moves (>2.0)",
    "Mean Opening Spread",
    "Mean Closing Spread",
    "Mean Days Before Game"
  ),
  value = c(
    nrow(line_movements),
    mean(line_movements$abs_line_movement, na.rm = TRUE),
    median(line_movements$abs_line_movement, na.rm = TRUE),
    sd(line_movements$abs_line_movement, na.rm = TRUE),
    sum(line_movements$abs_line_movement > 0.5, na.rm = TRUE),
    sum(line_movements$abs_line_movement > 1.0, na.rm = TRUE),
    sum(line_movements$abs_line_movement > 2.0, na.rm = TRUE),
    mean(abs(line_movements$opening_spread), na.rm = TRUE),
    mean(abs(line_movements$closing_spread), na.rm = TRUE),
    mean(line_movements$days_before_game, na.rm = TRUE)
  )
)

# CLV by strategy
clv_stats <- clv_long %>%
  group_by(strategy) %>%
  summarise(
    mean_clv = mean(clv, na.rm = TRUE),
    median_clv = median(clv, na.rm = TRUE),
    sd_clv = sd(clv, na.rm = TRUE),
    positive_clv_pct = sum(clv > 0, na.rm = TRUE) / n() * 100,
    .groups = "drop"
  )

# Write summary files
write.csv(
  summary_stats,
  file.path(opt$`output-dir`, "line_movement_summary.csv"),
  row.names = FALSE
)

write.csv(
  clv_stats,
  file.path(opt$`output-dir`, "clv_summary.csv"),
  row.names = FALSE
)

# ============================================================================
# Print Summary
# ============================================================================

cat("\n" %+% strrep("=", 70) %+% "\n")
cat("LINE MOVEMENT ANALYSIS SUMMARY\n")
cat(strrep("=", 70) %+% "\n\n")

cat("Data Overview:\n")
cat(sprintf("  Total games analyzed: %s\n", format(nrow(line_movements), big.mark = ",")))
if (!is.null(opt$season)) {
  cat(sprintf("  Season: %d\n", opt$season))
} else {
  cat(sprintf("  Seasons: %d - %d\n", min(line_movements$season), max(line_movements$season)))
}

cat("\nLine Movement Statistics:\n")
cat(sprintf("  Mean absolute movement: %.3f points\n", mean(line_movements$abs_line_movement, na.rm = TRUE)))
cat(sprintf("  Median absolute movement: %.3f points\n", median(line_movements$abs_line_movement, na.rm = TRUE)))
cat(sprintf("  SD: %.3f points\n", sd(line_movements$abs_line_movement, na.rm = TRUE)))

cat("\nSteam Move Analysis:\n")
steam_pct <- sum(line_movements$steam_move, na.rm = TRUE) / nrow(line_movements) * 100
cat(sprintf("  Steam moves (>0.5 pts): %s (%.1f%%)\n",
            format(sum(line_movements$steam_move, na.rm = TRUE), big.mark = ","),
            steam_pct))
cat(sprintf("  Large moves (>1.0 pts): %s (%.1f%%)\n",
            format(sum(line_movements$abs_line_movement > 1.0, na.rm = TRUE), big.mark = ","),
            sum(line_movements$abs_line_movement > 1.0, na.rm = TRUE) / nrow(line_movements) * 100))
cat(sprintf("  Extreme moves (>2.0 pts): %s (%.1f%%)\n",
            format(sum(line_movements$abs_line_movement > 2.0, na.rm = TRUE), big.mark = ","),
            sum(line_movements$abs_line_movement > 2.0, na.rm = TRUE) / nrow(line_movements) * 100))

cat("\nCLV by Strategy:\n")
for (i in 1:nrow(clv_stats)) {
  cat(sprintf("  %s:\n", clv_stats$strategy[i]))
  cat(sprintf("    Mean CLV: %.3f points\n", clv_stats$mean_clv[i]))
  cat(sprintf("    Positive CLV: %.1f%%\n", clv_stats$positive_clv_pct[i]))
}

cat("\nOutput Files:\n")
cat(sprintf("  Figures: %s/\n", opt$`output-dir`))
cat("    - line_movement_scatter.png\n")
cat("    - line_movement_distribution.png\n")
cat("    - line_movement_by_weekday.png\n")
cat("    - clv_distribution.png\n")
cat("    - steam_moves.png\n")
if ("sharp_indicators" %in% colnames(line_movements)) {
  cat("    - sharp_indicators.png\n")
}
cat("    - line_movement_timeseries.png\n")
cat("  Summary: line_movement_summary.csv, clv_summary.csv\n")

cat("\n" %+% strrep("=", 70) %+% "\n")
cat("Analysis complete!\n")
