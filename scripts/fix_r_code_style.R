#!/usr/bin/env Rscript
#
# Fix R Code Style Issues
#
# This script uses the styler package to automatically fix code style violations
# across the R codebase, specifically:
# - Convert `=` to `<-` for assignment
# - Fix indentation and spacing
# - Apply tidyverse style guide
#
# Usage: Rscript scripts/fix_r_code_style.R
#

library(styler)

cat("========================================\n")
cat("R CODE STYLE FIXER\n")
cat("========================================\n\n")

cat("Installing/loading styler package...\n")
if (!require("styler", quietly = TRUE)) {
  install.packages("styler")
  library(styler)
}

cat("\nApplying tidyverse style to R/ directory...\n\n")

# Style all R files in the R/ directory
results <- style_dir(
  "R/",
  transformers = tidyverse_style(),
  include_roxygen_examples = FALSE,
  dry = "off"  # Actually make changes
)

cat("\n========================================\n")
cat("STYLING COMPLETE!\n")
cat("========================================\n\n")

# Print summary
if (is.null(results)) {
  cat("All files already conform to tidyverse style ✅\n")
} else {
  cat("Files styled:\n")
  print(results)
}

cat("\nKey changes made:\n")
cat("  1. Converted `=` to `<-` for assignment\n")
cat("  2. Fixed indentation (2 spaces)\n")
cat("  3. Normalized spacing around operators\n")
cat("  4. Formatted function calls\n\n")

cat("Next steps:\n")
cat("  1. Review changes: git diff R/\n")
cat("  2. Test modified files\n")
cat("  3. Commit: git add R/ && git commit -m 'style: apply tidyverse style guide'\n\n")
