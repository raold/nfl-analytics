# Initialize renv for R package reproducibility

if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv", repos = "https://cloud.r-project.org")
}

library(renv)

# Initialize renv if not already initialized
if (!file.exists("renv.lock")) {
  renv::init(bare = TRUE)
}

# Snapshot current package versions
renv::snapshot(prompt = FALSE)

cat("\n✅ renv.lock created/updated with R package versions\n")
cat("Snapshot includes", length(renv::dependencies()$Package), "packages\n")
