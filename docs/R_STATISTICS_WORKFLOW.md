# R Statistics Workflow Guide

**Version**: 1.0
**Created**: November 7, 2025
**Purpose**: Guidelines for R-based statistical analysis, dissertation figures, and advanced probability work

---

## Overview

This document provides workflow patterns for R-based statistical work in the NFL analytics project, with emphasis on:
- Bayesian hierarchical modeling (brms, Stan, cmdstanr)
- Dissertation figure generation (ggplot2, tikzDevice)
- Advanced probability/statistics (tidyverse, dplyr, glue)
- Data frame manipulation and transformation

---

## Core Philosophy

**"R is weird but strangely powerful"**

R's quirks (data frames, NSE, `<-` vs `=`, piping) can be confusing, but its statistical ecosystem is unmatched. Follow these patterns for maintainable, reproducible R code.

---

## Critical R Patterns

### 1. Data Frame Manipulation

**Use tidyverse, not base R loops**:
```r
# GOOD - tidyverse pipeline
player_stats <- pbp_clean %>%
  filter(pass == 1, !is.na(passer_player_id)) %>%
  group_by(passer_player_id, passer_player_name) %>%
  summarize(
    attempts = n(),
    completions = sum(complete_pass, na.rm = TRUE),
    yards = sum(passing_yards, na.rm = TRUE),
    tds = sum(pass_touchdown, na.rm = TRUE),
    .groups = "drop"
  )

# AVOID - base R loops
player_stats <- data.frame()
for (player in unique(pbp_clean$passer_player_id)) {
  subset <- pbp_clean[pbp_clean$passer_player_id == player, ]
  # ... manual aggregation
}
```

### 2. String Interpolation with glue()

**Use glue() for readable SQL/strings**:
```r
library(glue)

# GOOD - glue for SQL queries
query <- glue("
  SELECT
    player_id,
    player_name,
    SUM(passing_yards) as total_yards
  FROM plays
  WHERE season = {season}
    AND week <= {max_week}
  GROUP BY player_id, player_name
")

# AVOID - sprintf hell
query <- sprintf(
  "SELECT player_id, player_name, SUM(passing_yards) as total_yards FROM plays WHERE season = %d AND week <= %d GROUP BY player_id, player_name",
  season, max_week
)
```

### 3. Defensive Column Selection

**Use any_of() to avoid column errors**:
```r
# GOOD - defensive selection (columns may not exist in all datasets)
pbp_clean <- pbp_raw %>%
  select(any_of(CANONICAL_COLUMNS))

# AVOID - assumes all columns exist
pbp_clean <- pbp_raw %>%
  select(all_of(CANONICAL_COLUMNS))  # Errors if any column missing
```

### 4. Type Coercion (Critical for nflreadr!)

**Explicitly convert types before database inserts**:
```r
# Boolean columns (nflreadr provides 0.0/1.0, database expects TRUE/FALSE)
boolean_cols <- c("xyac_success", "xyac_fd", "aborted_play", "out_of_bounds", "tackled_for_loss")
for (col in boolean_cols) {
  if (col %in% names(pbp_clean) && !is.logical(pbp_clean[[col]])) {
    pbp_clean[[col]] <- as.logical(pbp_clean[[col]])
  }
}

# Integer columns (nflreadr provides "1.0" strings)
if ("order_sequence" %in% names(pbp_clean)) {
  pbp_clean$order_sequence <- as.integer(pbp_clean$order_sequence)
}
```

---

## Bayesian Modeling with brms/Stan

### Standard Hierarchical Model Template

```r
library(brms)
library(cmdstanr)

# Model formula with hierarchical structure
model_formula <- bf(
  passing_yards ~ 1 +
    # Fixed effects
    home + temperature + wind_speed + precipitation +

    # Random effects (hierarchical structure)
    (1 | position_group) +          # Position group intercepts
    (1 | position) +                 # Position intercepts
    (1 | team_abbr) +                # Team intercepts
    (1 | passer_player_id) +         # Player intercepts
    (1 | game_id)                    # Game-specific variation
)

# Priors (informative based on domain knowledge)
priors <- c(
  prior(normal(250, 50), class = Intercept),     # Typical passing yards
  prior(normal(10, 5), class = b, coef = home),  # Home field advantage
  prior(cauchy(0, 25), class = sd),              # Hierarchical SDs
  prior(student_t(3, 0, 100), class = sigma)     # Residual SD
)

# Fit model (cmdstanr backend for performance)
fit <- brm(
  formula = model_formula,
  data = training_data,
  prior = priors,
  family = gaussian(),
  backend = "cmdstanr",
  chains = 4,
  iter = 2000,
  warmup = 1000,
  cores = 4,
  seed = 42
)

# Model diagnostics
summary(fit)
pp_check(fit, ndraws = 100)  # Posterior predictive check
```

### Extracting Posterior Samples for BNN Integration

```r
# Extract player-level random effects
player_effects <- ranef(fit)$passer_player_id %>%
  as.data.frame() %>%
  rownames_to_column("player_id") %>%
  select(player_id,
         posterior_mean = Estimate,
         posterior_sd = Est.Error)

# Export to database for Python BNN
dbWriteTable(con, "mart.bayesian_player_ratings", player_effects,
             overwrite = TRUE, row.names = FALSE)
```

---

## Dissertation Figures with ggplot2

### Standard Publication-Quality Plot

```r
library(ggplot2)
library(scales)

# Create plot with consistent styling
p <- ggplot(data, aes(x = week, y = yards, color = player_name)) +
  geom_line(size = 1.2) +
  geom_point(size = 2.5) +

  # Scales
  scale_x_continuous(breaks = 1:18) +
  scale_y_continuous(labels = comma) +
  scale_color_brewer(palette = "Set2") +

  # Labels
  labs(
    title = "QB Passing Yards by Week (2024 Season)",
    subtitle = "Top 5 QBs by total yards",
    x = "Week",
    y = "Passing Yards",
    color = "Quarterback",
    caption = "Source: nflreadr (via nflfastR)"
  ) +

  # Theme
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "gray40"),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

# Save for LaTeX (PDF for vector graphics)
ggsave("../out/qb_yards_by_week.pdf", p,
       width = 8, height = 5, device = cairo_pdf)
```

### TikZ Device for Perfect LaTeX Integration

```r
library(tikzDevice)

# Generate TikZ code (uses LaTeX fonts, perfect math rendering)
tikz("../out/calibration_plot.tex", width = 6, height = 4)

ggplot(calibration_data, aes(x = predicted, y = observed)) +
  geom_point(alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(
    x = "$\\hat{y}$ (Predicted Yards)",
    y = "$y$ (Observed Yards)",
    title = "BNN Calibration: Predicted vs Observed"
  ) +
  theme_minimal()

dev.off()
```

---

## Database Workflows

### Connection Pattern (PostgreSQL)

```r
library(DBI)
library(RPostgres)

# Standard connection
con <- dbConnect(
  RPostgres::Postgres(),
  host = "localhost",
  port = 5544,
  dbname = "devdb01",
  user = "dro",
  password = "sicillionbillions"
)

# Always disconnect when done
on.exit(dbDisconnect(con))
```

### Batch Inserts (Critical for Large Datasets)

```r
# Insert in batches to avoid memory issues
batch_size <- 10000
num_batches <- ceiling(nrow(pbp_clean) / batch_size)

for (batch_idx in 1:num_batches) {
  start_row <- (batch_idx - 1) * batch_size + 1
  end_row <- min(batch_idx * batch_size, nrow(pbp_clean))
  batch_data <- pbp_clean[start_row:end_row, ]

  dbWriteTable(con, "plays", batch_data, append = TRUE, row.names = FALSE)

  cat(sprintf("Inserted batch %d/%d (%d-%d of %d plays)\n",
              batch_idx, num_batches, start_row, end_row, nrow(pbp_clean)))
}
```

### Safe SQL Execution with Error Handling

```r
# Use tryCatch for robust database operations
tryCatch({
  # Delete old data
  deleted_count <- dbExecute(con,
    sprintf("DELETE FROM plays WHERE game_id LIKE '%s_%%'", season))

  # Insert new data
  dbWriteTable(con, "plays", pbp_clean, append = TRUE, row.names = FALSE)

  # Validate
  validation <- dbGetQuery(con, sprintf("
    SELECT COUNT(*) as total, COUNT(passing_yards) as has_passing
    FROM plays WHERE game_id LIKE '%s_%%'", season))

  cat(sprintf("✓ Season %s: %d plays, %.1f%% passing_yards\n",
              season, validation$total,
              100.0 * validation$has_passing / validation$total))

}, error = function(e) {
  cat(sprintf("✗ ERROR in season %s: %s\n", season, e$message))
})
```

---

## Common Pitfalls & Solutions

### Pitfall 1: sprintf Format String Errors

**Problem**: Using `%d` for non-integer types
```r
# WRONG - season is integer but used in string context
sprintf("DELETE FROM plays WHERE game_id LIKE '%d_%%'", season)
# Error: invalid format '%d'; use format %f, %e, %g or %a for numeric objects
```

**Solution**: Use `%s` with `as.character()`
```r
# CORRECT
sprintf("DELETE FROM plays WHERE game_id LIKE '%s_%%'", as.character(season))
```

### Pitfall 2: Column Name Mismatches

**Problem**: Assuming column names from nflreadr match database
```r
# WRONG - "special" column doesn't exist in database
CANONICAL_COLUMNS <- c("game_id", "play_id", "special", ...)
```

**Solution**: Use exact column list from working ingestion script
```r
# CORRECT - copy from ingest_current_season.R
CANONICAL_COLUMNS <- c(
  "game_id", "play_id", "posteam", "defteam", "quarter",
  # ... (exact 90-column list from working script)
)
```

### Pitfall 3: Data Frame vs Tibble Confusion

**Problem**: Expecting tibble behavior from data.frame
```r
# Returns vector, not data frame
df$column

# Returns data frame (tibble behavior)
df %>% select(column)
```

**Solution**: Use tidyverse consistently
```r
# Always use dplyr verbs for data frame operations
df %>%
  filter(condition) %>%
  select(columns) %>%
  mutate(new_col = ...)
```

---

## Advanced Probability & Statistics

### Empirical Bayes Prior Elicitation

```r
# Use historical data to inform priors
historical_summary <- training_data %>%
  group_by(position) %>%
  summarize(
    mean_yards = mean(passing_yards, na.rm = TRUE),
    sd_yards = sd(passing_yards, na.rm = TRUE)
  )

# Create position-specific priors
QB_prior <- prior(normal(250, 50), class = Intercept, coef = QB)
RB_prior <- prior(normal(180, 40), class = Intercept, coef = RB)
```

### Posterior Predictive Distributions

```r
# Generate predictions with full uncertainty
posterior_preds <- posterior_predict(fit, newdata = test_data, ndraws = 1000)

# Summarize posterior predictive distribution
pred_summary <- test_data %>%
  mutate(
    pred_mean = colMeans(posterior_preds),
    pred_sd = apply(posterior_preds, 2, sd),
    pred_q05 = apply(posterior_preds, 2, quantile, probs = 0.05),
    pred_q95 = apply(posterior_preds, 2, quantile, probs = 0.95)
  )
```

---

## Testing R Code

### Unit Testing with testthat

```r
library(testthat)

test_that("player stats aggregation works correctly", {
  # Setup test data
  test_pbp <- data.frame(
    passer_player_id = c("1", "1", "2"),
    passing_yards = c(250, 300, 180),
    pass_touchdown = c(2, 1, 0)
  )

  # Run aggregation
  result <- aggregate_player_stats(test_pbp)

  # Assertions
  expect_equal(nrow(result), 2)
  expect_equal(result$total_yards[1], 550)
  expect_equal(result$total_tds[1], 3)
})
```

### Integration Testing with Database

```r
test_that("database ingestion preserves Phase 1 columns", {
  # Setup test connection
  con <- dbConnect(RPostgres::Postgres(),
                   host = "localhost", port = 5544,
                   dbname = "testdb", user = "test", password = "test")

  # Ingest test data
  ingest_season(con, season = 2024)

  # Validate Phase 1 columns
  validation <- dbGetQuery(con, "
    SELECT COUNT(passing_yards) as has_passing
    FROM plays WHERE game_id LIKE '2024_%'
  ")

  expect_gt(validation$has_passing, 0)

  dbDisconnect(con)
})
```

---

## Performance Optimization

### Vectorization over Loops

```r
# SLOW - loop over rows
for (i in 1:nrow(df)) {
  df$result[i] <- complex_function(df$value[i])
}

# FAST - vectorized operation
df$result <- complex_function(df$value)
```

### data.table for Large Datasets

```r
library(data.table)

# Convert to data.table for speed
dt <- as.data.table(pbp_clean)

# Fast aggregation
player_stats <- dt[, .(
  attempts = .N,
  yards = sum(passing_yards, na.rm = TRUE)
), by = .(passer_player_id, passer_player_name)]
```

### Parallel Processing with future

```r
library(future)
library(furrr)

# Setup parallel backend
plan(multisession, workers = 4)

# Parallel map
results <- future_map(seasons, function(season) {
  ingest_season(con, season)
}, .progress = TRUE)
```

---

## Documentation Standards

### Script Header Template

```r
#!/usr/bin/env Rscript
# R/path/to/script.R
#
# Purpose: Brief description of what this script does
# Author: Claude Code (with human supervision)
# Created: YYYY-MM-DD
# Last Modified: YYYY-MM-DD
#
# Dependencies:
#   - nflreadr (>= 1.4.0)
#   - dplyr (>= 1.1.0)
#   - DBI, RPostgres
#
# Usage:
#   Rscript --vanilla R/path/to/script.R
#
# Expected Runtime: ~X minutes
# Expected Output: Description of outputs
#
# Notes:
#   - Important caveats or assumptions
#   - Known issues or limitations

suppressPackageStartupMessages({
  library(nflreadr)
  library(dplyr)
  library(DBI)
  library(RPostgres)
})

# ... script implementation
```

### Function Documentation Template

```r
#' Aggregate Player Statistics
#'
#' Aggregates play-by-play data to player-game level statistics
#'
#' @param pbp Play-by-play data frame (from nflreadr)
#' @param stat_type Character. One of "passing", "rushing", "receiving"
#' @param min_attempts Integer. Minimum attempts to include player
#'
#' @return Data frame with player-game statistics
#'
#' @examples
#' pbp <- nflreadr::load_pbp(2024)
#' passing_stats <- aggregate_player_stats(pbp, "passing", min_attempts = 10)
#'
#' @export
aggregate_player_stats <- function(pbp, stat_type = "passing", min_attempts = 1) {
  # ... implementation
}
```

---

## Resources

### Essential Packages

**Core tidyverse**:
- `dplyr` - Data manipulation
- `tidyr` - Data tidying
- `ggplot2` - Visualization
- `purrr` - Functional programming
- `readr` - Data import
- `stringr` - String manipulation
- `glue` - String interpolation

**Bayesian modeling**:
- `brms` - Bayesian regression models
- `cmdstanr` - Stan interface (faster than rstan)
- `bayesplot` - Posterior visualization
- `posterior` - Posterior analysis

**Database**:
- `DBI` - Database interface
- `RPostgres` - PostgreSQL driver

**NFL data**:
- `nflreadr` - NFL play-by-play data
- `nflplotR` - NFL-themed plots

### Learning Resources

- **R for Data Science**: https://r4ds.had.co.nz/
- **Advanced R**: https://adv-r.hadley.nz/
- **Bayesian Data Analysis**: http://www.stat.columbia.edu/~gelman/book/
- **brms documentation**: https://paul-buerkner.github.io/brms/
- **nflreadr vignettes**: https://nflreadr.nflverse.com/

---

## Quick Reference Card

### Common Data Wrangling Patterns

```r
# Filter rows
df %>% filter(column == value)

# Select columns
df %>% select(col1, col2, col3)
df %>% select(starts_with("player_"))
df %>% select(any_of(column_list))

# Create new columns
df %>% mutate(new_col = col1 + col2)

# Group and summarize
df %>%
  group_by(player_id) %>%
  summarize(
    avg_yards = mean(yards, na.rm = TRUE),
    total_tds = sum(touchdown, na.rm = TRUE)
  )

# Join data frames
left_join(df1, df2, by = "player_id")
inner_join(df1, df2, by = c("player_id", "season"))

# Reshape data
pivot_longer(df, cols = c(col1, col2), names_to = "stat", values_to = "value")
pivot_wider(df, names_from = stat, values_from = value)
```

### Common ggplot2 Patterns

```r
# Basic scatter plot
ggplot(df, aes(x = x_var, y = y_var)) + geom_point()

# Line plot with groups
ggplot(df, aes(x = week, y = yards, color = player)) + geom_line()

# Bar plot
ggplot(df, aes(x = player, y = yards)) + geom_col()

# Faceted plot
ggplot(df, aes(x = week, y = yards)) +
  geom_line() +
  facet_wrap(~player, ncol = 3)

# Save plot
ggsave("output.pdf", width = 8, height = 5)
```

---

## Version History

- **v1.0** (Nov 7, 2025): Initial version covering R workflows, Bayesian modeling, dissertation figures, and common pitfalls

---

**Maintained by**: Claude Code (Sonnet 4.5)
**Last Updated**: November 7, 2025
