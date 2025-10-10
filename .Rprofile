# ============================================================
# .Rprofile - Cross-Platform R Environment Configuration
# For Windows 11 (RTX 4090) and Mac M4 development
# ============================================================

# 1. CRAN MIRROR
options(
  repos = c(CRAN = "https://cloud.r-project.org/"),
  download.file.method = if (.Platform$OS.type == "windows") "wininet" else "libcurl"
)

# 2. DEFAULT OPTIONS
options(
  scipen = 999,                # Avoid scientific notation
  stringsAsFactors = FALSE,    # Modern R behavior
  digits = 4,
  max.print = 1000,
  width = 120
)

# 3. DATABASE CONNECTION DEFAULTS (from environment)
Sys.setenv(
  POSTGRES_HOST = Sys.getenv("POSTGRES_HOST", "localhost"),
  POSTGRES_PORT = Sys.getenv("POSTGRES_PORT", "5544"),
  POSTGRES_DB = Sys.getenv("POSTGRES_DB", "devdb01"),
  POSTGRES_USER = Sys.getenv("POSTGRES_USER", "dro")
)

# 4. PLATFORM-SPECIFIC CONFIGURATION
if (.Platform$OS.type == "windows") {
  # Windows 11 specific
  options(encoding = "UTF-8")
  Sys.setenv(R_USER_CACHE_DIR = file.path(Sys.getenv("LOCALAPPDATA"), "R"))
} else if (Sys.info()["sysname"] == "Darwin") {
  # Mac M4 specific
  Sys.setenv(OBJC_DISABLE_INITIALIZE_FORK_SAFETY = "YES")  # For forking on Mac
}

# 5. LOGGING AND WARNINGS
options(
  warn = 1,                    # Print warnings immediately
  error = function() {
    cat("\nError traceback:\n")
    traceback(2)
  }
)

# 6. INTERACTIVE SESSION HELPERS
if (interactive()) {
  cat("\n")
  cat("==============================================================\n")
  cat("NFL Analytics R Environment\n")
  cat("==============================================================\n")
  cat("Platform:  ", .Platform$OS.type, "\n")
  cat("R version: ", R.version$version.string, "\n")
  cat("Database:  ", Sys.getenv("POSTGRES_HOST"), ":",
      Sys.getenv("POSTGRES_PORT"), "/", Sys.getenv("POSTGRES_DB"), "\n")
  cat("==============================================================\n")
  cat("\n")

  # Utility functions
  .count_table <- function(table_name) {
    con <- get_db_connection()
    result <- DBI::dbGetQuery(con, sprintf("SELECT COUNT(*) FROM %s", table_name))
    DBI::dbDisconnect(con)
    result[[1]]
  }

  .list_tables <- function() {
    con <- get_db_connection()
    tables <- DBI::dbListTables(con)
    DBI::dbDisconnect(con)
    tables
  }
}

# 7. AUTO-LOAD PROJECT UTILITIES
if (file.exists("R/utils/db_helpers.R")) {
  tryCatch({
    source("R/utils/db_helpers.R")
    if (interactive()) cat("Loaded db_helpers.R\n")
  }, error = function(e) {
    if (interactive()) cat("Warning: Could not load db_helpers.R\n")
  })
}

# 8. CLEANUP ON EXIT
if (interactive()) {
  .Last <- function() {
    # Close any open database connections
    cons <- DBI::dbListConnections(RPostgres::Postgres())
    if (length(cons) > 0) {
      lapply(cons, DBI::dbDisconnect)
      cat("\nClosed", length(cons), "database connection(s)\n")
    }
  }
}
