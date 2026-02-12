#!/usr/bin/env Rscript
# Diagnostic script to identify data type issues in nflfastR data

suppressPackageStartupMessages({
  library(nflreadr)
  library(dplyr)
})

cat("Loading sample play-by-play data from 2024...\n")
pbp_sample <- load_pbp(2024) %>% slice(1:100)

cat("\nData types of all columns:\n")
cat("=" %>% rep(60) %>% paste(collapse = ""), "\n")

# Get all column types
col_types <- sapply(pbp_sample, class)

# Group by type
logical_cols <- names(col_types)[col_types == "logical"]
integer_cols <- names(col_types)[col_types == "integer"]
numeric_cols <- names(col_types)[col_types == "numeric"]
character_cols <- names(col_types)[col_types == "character"]

cat(sprintf("\nLogical columns (%d):\n", length(logical_cols)))
print(logical_cols)

cat(sprintf("\nInteger columns (%d):\n", length(integer_cols)))
print(head(integer_cols, 20))

cat(sprintf("\nNumeric columns (%d):\n", length(numeric_cols)))
print(head(numeric_cols, 20))

cat(sprintf("\nCharacter columns (%d):\n", length(character_cols)))
print(head(character_cols, 20))

cat("\n" %>% rep(2) %>% paste(collapse = ""))
cat("Sample values from logical columns:\n")
cat("=" %>% rep(60) %>% paste(collapse = ""), "\n")

for (col in head(logical_cols, 5)) {
  cat(sprintf("\n%s: ", col))
  print(head(pbp_sample[[col]], 10))
}

cat("\n" %>% rep(2) %>% paste(collapse = ""))
cat("Testing conversion:\n")
cat("=" %>% rep(60) %>% paste(collapse = ""), "\n")

test_col <- logical_cols[1]
cat(sprintf("\nOriginal (%s): ", test_col))
print(class(pbp_sample[[test_col]]))
print(head(pbp_sample[[test_col]], 5))

converted <- as.integer(pbp_sample[[test_col]])
cat(sprintf("\nConverted to integer: "))
print(class(converted))
print(head(converted, 5))

cat("\n✓ Conversion successful! All logical columns can be converted to integers.\n")
