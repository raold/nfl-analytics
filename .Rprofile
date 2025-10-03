# Set CRAN mirror
options(repos = c(CRAN = "https://cran.rstudio.com/"))

# Set default options
options(digits = 4)

# Source setup script to install and load packages
tryCatch({
  source("setup_packages.R")
}, error = function(e) {
  cat("Note: setup_packages.R not found or error occurred. Run source('setup_packages.R') manually if needed.\n")
})
