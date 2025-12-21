# setup_packages.R - Install and load required packages

# Function to install packages if not already installed
install_if_missing <- function(packages) {
  for (pkg in packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      install.packages(pkg)
    }
  }
}

# Function to install GitHub packages if not already installed
install_github_if_missing <- function(repo, package_name) {
  if (!requireNamespace(package_name, quietly = TRUE)) {
    if (!requireNamespace("remotes", quietly = TRUE)) {
      install.packages("remotes")
    }
    remotes::install_github(repo)
  }
}

# Install required CRAN packages
required_packages <- c(
  "dplyr", "arrow", "data.table", "janitor", "lubridate",
  "xgboost", "parsnip", "yardstick", "rsample", "recipes",
  "glmnet", "lightgbm", "DBI", "RPostgres", "dbplyr",
  "ggplot2", "gt", "tidyr", "pacman", "tidymodels",
  "jsonlite", "purrr", "glue", "readr", "dlm"
)

cat("Installing missing CRAN packages...\n")
install_if_missing(required_packages)

# Install GitHub packages
cat("Installing missing GitHub packages...\n")
install_github_if_missing("nflverse/nflreadr", "nflreadr")
install_github_if_missing("nflverse/nflfastR", "nflfastR")

cat("Package installation complete!\n")

# Load commonly used packages (suppress conflict messages)
suppressPackageStartupMessages({
  library(dplyr)
  library(tidymodels)
  library(DBI)
  library(RPostgres)
})

cat("Core packages loaded!\n")
