#!/usr/bin/env Rscript
# R Code Quality Audit Script
# Equivalent to Python's ruff/mypy/black analysis

library(lintr)

cat("=", rep("=", 59), "\n", sep = "")
cat("R CODE QUALITY AUDIT (with .lintr config)\n")
cat("=", rep("=", 59), "\n\n", sep = "")

# Find all R files (excluding renv, venv)
r_files <- list.files(
  path = "R",
  pattern = "[.]R$",
  recursive = TRUE,
  full.names = TRUE
)

cat("Auditing", length(r_files), "R files...\n\n")

# Run lintr on all files
all_lints <- list()
files_checked <- 0

for (f in r_files) {
  files_checked <- files_checked + 1
  if (files_checked %% 10 == 0) {
    cat("  Checked", files_checked, "/", length(r_files), "files\n")
  }

  lints <- tryCatch(
    lint(f),
    error = function(e) {
      cat("  Error linting", f, ":", conditionMessage(e), "\n")
      list()
    }
  )
  if (length(lints) > 0) {
    all_lints[[f]] <- lints
  }
}

cat("\n")

# Flatten all lints
flat_lints <- unlist(all_lints, recursive = FALSE)

if (length(flat_lints) == 0) {
  cat("No lints found! R code is clean.\n")
  quit(status = 0)
}

# Aggregate by lint type
lint_types <- sapply(flat_lints, function(x) x$linter)
lint_counts <- sort(table(lint_types), decreasing = TRUE)

cat("=== LINT SUMMARY BY TYPE ===\n")
for (i in seq_along(lint_counts)) {
  cat(sprintf("%4d  %s\n", lint_counts[i], names(lint_counts)[i]))
}

cat("\n=== FILES WITH MOST ISSUES ===\n")
file_counts <- sapply(all_lints, length)
file_counts <- sort(file_counts, decreasing = TRUE)
for (i in seq_len(min(10, length(file_counts)))) {
  cat(sprintf("%4d  %s\n", file_counts[i], names(file_counts)[i]))
}

cat("\n=== SUMMARY ===\n")
cat("Total lints:", sum(lint_counts), "\n")
cat("Files with issues:", length(all_lints), "/", length(r_files), "\n")
cat("Most common issue:", names(lint_counts)[1], "(", lint_counts[1], ")\n")

# Show sample issues for top 3 lint types
cat("\n=== SAMPLE ISSUES (top 3 types) ===\n")
top_types <- names(lint_counts)[1:min(3, length(lint_counts))]

for (lt in top_types) {
  cat("\n", lt, ":\n", sep = "")
  examples <- flat_lints[lint_types == lt][1:min(3, sum(lint_types == lt))]
  for (ex in examples) {
    cat("  ", basename(ex$filename), ":", ex$line_number, " - ", ex$message, "\n", sep = "")
  }
}
