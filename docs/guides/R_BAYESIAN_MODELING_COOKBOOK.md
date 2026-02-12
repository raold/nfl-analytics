# R Bayesian Modeling Cookbook

**Last Updated:** November 16, 2025
**Version:** 1.0
**Purpose:** Complete guide for training, validating, and deploying production R Bayesian models
**Audience:** Data scientists, ML engineers, solo developers (bus factor mitigation)

---

## Table of Contents

1. [Quick Start (5 minutes)](#quick-start-5-minutes)
2. [Environment Setup (Detailed)](#environment-setup-detailed)
3. [Training Workflow (Step-by-Step)](#training-workflow-step-by-step)
4. [Model-Specific Guides](#model-specific-guides)
   - [State-Space Random Walk v2](#state-space-random-walk-v2)
   - [Hierarchical Bayesian v2](#hierarchical-bayesian-v2)
   - [Informative Priors v2](#informative-priors-v2)
5. [MCMC Diagnostics (Critical Validation)](#mcmc-diagnostics-critical-validation)
6. [Common Errors & Solutions](#common-errors--solutions)
7. [Model Registry Integration](#model-registry-integration)
8. [Production Checklist](#production-checklist)
9. [Troubleshooting Guide](#troubleshooting-guide)
10. [References & Resources](#references--resources)

---

## Quick Start (5 minutes)

**Goal:** Get to a working "Hello World" Bayesian model in 15 minutes.

### Prerequisites

- R 4.0+ installed
- PostgreSQL running on port 5544 (`devdb01` database)
- Terminal access

### Installation Script

```bash
# 1. Install R packages (run from project root)
Rscript setup_packages.R

# 2. Install cmdstanr and cmdstan backend
R --vanilla <<'EOF'
install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
cmdstanr::install_cmdstan(cores = 4)
EOF

# 3. Verify installation
R --vanilla <<'EOF'
library(cmdstanr)
library(brms)
cat("✓ cmdstanr version:", as.character(packageVersion("cmdstanr")), "\n")
cat("✓ brms version:", as.character(packageVersion("brms")), "\n")
cat("✓ cmdstan path:", cmdstan_path(), "\n")
EOF
```

**Expected output:**
```
✓ cmdstanr version: 0.7.1
✓ brms version: 2.21.0
✓ cmdstan path: /Users/dro/.cmdstan/cmdstan-2.34.1
```

### Hello World Model

```r
#!/usr/bin/env Rscript
# test_bayesian_install.R - Verify Bayesian stack works

library(brms)
library(cmdstanr)

# Simple synthetic data
set.seed(42)
data <- data.frame(
  x = rnorm(100),
  y = 2 * rnorm(100) + rnorm(100, sd = 0.5)
)

# Fit simple linear model
model <- brm(
  y ~ x,
  data = data,
  family = gaussian(),
  prior = c(
    prior(normal(0, 1), class = Intercept),
    prior(normal(0, 1), class = b)
  ),
  chains = 2,
  iter = 1000,
  warmup = 500,
  backend = "cmdstanr",
  cores = 2,
  seed = 42,
  refresh = 0  # Suppress iteration output
)

# Check convergence
summary(model)
cat("\n✅ SUCCESS! Bayesian modeling stack is working.\n")
```

**Run:**
```bash
Rscript test_bayesian_install.R
```

**Success criteria:**
- Model compiles without errors
- All Rhat values < 1.01
- ESS (bulk) > 200
- No divergent transitions

---

## Environment Setup (Detailed)

### R Version Requirements

**Minimum:** R 4.0.0
**Recommended:** R 4.3.0+
**Check version:**
```bash
R --version
```

### Core Package Installation

#### Method 1: Automated (Recommended)

```bash
# From project root
Rscript setup_packages.R
```

**Installs:**
- **Data manipulation:** `dplyr`, `tidyr`, `purrr`, `glue`, `lubridate`
- **Database:** `DBI`, `RPostgres`, `dbplyr`
- **Visualization:** `ggplot2`, `bayesplot`
- **Bayesian modeling:** `brms`, `cmdstanr`, `posterior`

#### Method 2: Manual Installation

```r
# Core CRAN packages
install.packages(c(
  "dplyr", "tidyr", "purrr", "glue", "lubridate",
  "DBI", "RPostgres",
  "ggplot2", "bayesplot"
))

# cmdstanr (special repository)
install.packages(
  "cmdstanr",
  repos = c("https://mc-stan.org/r-packages/", getOption("repos"))
)

# brms (requires cmdstanr)
install.packages("brms")

# posterior (for diagnostics)
install.packages("posterior")
```

### cmdstan Backend Installation

**What is cmdstan?** Stan's command-line interface for MCMC sampling. Required by `cmdstanr`.

```r
library(cmdstanr)

# Install cmdstan (downloads ~250 MB, compiles ~5-10 min)
install_cmdstan(
  cores = parallel::detectCores() - 1,
  quiet = FALSE,
  overwrite = FALSE
)

# Verify installation
cmdstan_path()
cmdstan_version()
```

**Expected output:**
```
CmdStan path set to: /Users/dro/.cmdstan/cmdstan-2.34.1
CmdStan version: 2.34.1
```

**Troubleshooting cmdstan install:**
- **Error: "C++ compiler not found"**
  - macOS: Install Xcode command line tools (`xcode-select --install`)
  - Linux: Install `build-essential` (`sudo apt-get install build-essential`)
  - Windows: Install Rtools (https://cran.r-project.org/bin/windows/Rtools/)

### Database Connection Setup

#### Environment Variables (.Renviron)

Create `~/.Renviron` file:
```bash
cat > ~/.Renviron <<'EOF'
# PostgreSQL credentials
DB_HOST=localhost
DB_PORT=5544
DB_NAME=devdb01
DB_USER=dro
DB_PASSWORD=sicillionbillions
EOF
```

**Security note:** This is for local development only. Use secrets manager for production.

#### Test Database Connection

```r
library(DBI)
library(RPostgres)

conn <- dbConnect(
  Postgres(),
  host = Sys.getenv("DB_HOST", "localhost"),
  port = as.integer(Sys.getenv("DB_PORT", "5544")),
  dbname = Sys.getenv("DB_NAME", "devdb01"),
  user = Sys.getenv("DB_USER", "dro"),
  password = Sys.getenv("DB_PASSWORD", "sicillionbillions")
)

# Test query
result <- dbGetQuery(conn, "SELECT COUNT(*) as n_games FROM games;")
cat(sprintf("✓ Connected to database: %d games found\n", result$n_games))

dbDisconnect(conn)
```

**Expected output:**
```
✓ Connected to database: 7263 games found
```

### RStudio Configuration (Optional)

**Recommended settings for MCMC sampling:**

1. **Increase memory limit:**
   ```r
   # Add to ~/.Rprofile
   options(java.parameters = "-Xmx8g")  # 8 GB heap
   Sys.setenv("R_MAX_VSIZE" = 32 * 1024^3)  # 32 GB virtual memory
   ```

2. **Set parallel cores:**
   ```r
   # Add to ~/.Rprofile
   options(mc.cores = parallel::detectCores() - 1)
   ```

3. **Enable multithreading (M1/M2/M3 Macs):**
   ```r
   # Add to ~/.Rprofile
   Sys.setenv("STAN_NUM_THREADS" = 2)
   ```

4. **Disable graphics device (faster sampling):**
   ```r
   # Add to script header
   options(device = "null")
   ```

---

## Training Workflow (Step-by-Step)

### Canonical Training Window

**Decision:** 2018-2024 + H=4 exponential decay on 2015-2017
**Rationale:** Ablation study (Nov 8, 2025) showed optimal performance
**Documentation:** `docs/archive/2025_nov/STATE_SPACE_VALIDATION_REPORT.md`

**Decay Formula:**
```
w(s; t, H) = 0.5^((t - s) / H)

where:
  s = historical season (2015-2017)
  t = evaluation season (2025)
  H = half-life parameter (4 years)
```

**Weight Examples:**
- 2018-2024: w = 1.0 (full weight)
- 2017: w = 0.5^((2025-2017)/4) = 0.5^2 = 0.25
- 2016: w = 0.5^((2025-2016)/4) = 0.5^2.25 = 0.21
- 2015: w = 0.5^((2025-2015)/4) = 0.5^2.5 = 0.18

**Implementation Note:** For brms models (Hierarchical, Informative Priors), use uniform 2015-2025 instead. Ablation study showed equivalent performance with simpler implementation.

### Step-by-Step Workflow

#### Step 1: Connect to Database

```r
library(DBI)
library(RPostgres)

conn <- dbConnect(
  Postgres(),
  host = "localhost",
  port = 5544,
  dbname = "devdb01",
  user = "dro",
  password = Sys.getenv("DB_PASSWORD", "sicillionbillions")
)
```

**Best practice:** Always use `Sys.getenv()` for credentials (never hardcode).

#### Step 2: Extract Training Data

**Critical SQL pattern:** See `docs/database/DATABASE_QUICK_REFERENCE.md` for schema details.

```r
query <- "
SELECT
  pgs.player_id,
  pgs.season,
  pgs.week,
  pgs.stat_yards as yards,
  pgs.stat_attempts as attempts,

  -- Game context
  g.home_team,
  g.away_team,
  ph.current_team as team,
  CASE WHEN ph.current_team = g.home_team THEN 1 ELSE 0 END as is_home,

  -- Betting lines
  CASE
    WHEN ph.current_team = g.home_team THEN CASE WHEN g.spread_close < 0 THEN 1 ELSE 0 END
    ELSE CASE WHEN g.spread_close > 0 THEN 1 ELSE 0 END
  END as is_favored,
  ABS(g.spread_close) as spread_abs,
  g.total_close,

  -- Weather
  COALESCE(g.roof, 'unknown') as roof,
  CASE WHEN g.roof IN ('outdoors', 'open') AND CAST(g.temp AS NUMERIC) < 40 THEN 1 ELSE 0 END as is_bad_weather,
  CASE WHEN g.roof = 'dome' THEN 1 ELSE 0 END as is_dome,

  -- Player info
  ph.years_exp,
  CASE
    WHEN ph.current_team = g.home_team THEN g.away_team
    ELSE g.home_team
  END as opponent

FROM mart.player_game_stats pgs
JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
LEFT JOIN games g ON
  g.season = pgs.season
  AND g.week = pgs.week
  AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
WHERE pgs.stat_category = 'passing'
  AND pgs.stat_yards IS NOT NULL
  AND pgs.stat_attempts IS NOT NULL
  AND pgs.stat_attempts >= 10  -- Minimum attempts filter
  AND pgs.season BETWEEN 2015 AND 2025
  AND ph.position_group = 'QB'
ORDER BY pgs.season, pgs.week, pgs.player_id
"

data <- dbGetQuery(conn, query)
cat(sprintf("Loaded %d passing records\n", nrow(data)))
```

**Common mistakes:**
- ❌ `JOIN games g ON g.game_id = pgs.game_id` (game_id doesn't exist in player_game_stats)
- ✅ `JOIN games g ON g.season = pgs.season AND g.week = pgs.week AND (...)`

#### Step 3: Feature Engineering

```r
library(dplyr)

data <- data %>%
  # Filter NAs (critical for brms)
  filter(!is.na(total_close), !is.na(years_exp)) %>%

  # Log transformations (for log-normal modeling)
  mutate(
    log_yards = log(yards),
    log_attempts = log(attempts),

    # Interaction terms
    player_season = paste0(player_id, "_", season)
  )

cat(sprintf("Features engineered: %d records ready\n", nrow(data)))
```

**Important:**
- `log()` not `log1p()` for yards (yards > 0 always)
- Filter NAs before modeling (brms will fail on missing values)
- Create `player_season` for varying slopes

#### Step 4: Specify Model Formula

**brms formula syntax:**
```r
formula <- bf(
  log_yards ~ 1 +                    # Intercept
    log_attempts +                   # Fixed effect
    is_home +                        # Fixed effect
    is_favored +                     # Fixed effect
    spread_abs +                     # Fixed effect
    is_bad_weather +                 # Fixed effect
    is_dome +                        # Fixed effect
    scale(total_close) +             # Scaled continuous
    scale(years_exp) +               # Scaled continuous
    (1 | player_id) +                # Random intercept (player baseline)
    (1 | team) +                     # Random intercept (team effect)
    (1 | opponent) +                 # Random intercept (opponent defense)
    (log_attempts | player_season),  # Varying slope (player form)
  sigma ~ log_attempts               # Distributional regression (heteroskedasticity)
)
```

**Formula components:**
- `log_yards ~` - Response variable (log-transformed passing yards)
- `1` - Intercept (league-wide average)
- `log_attempts` - Fixed effect (more attempts = more yards)
- `(1 | player_id)` - Random intercept per player (partial pooling)
- `(log_attempts | player_season)` - Varying slope (each player-season has unique attempts effect)
- `sigma ~ log_attempts` - Model variance as function of attempts (reduces heteroskedasticity)

#### Step 5: Set Priors

**Weakly informative priors (default for Hierarchical v2):**
```r
priors <- c(
  # Intercept: log(200 yards) ≈ 5.3
  prior(normal(5, 1), class = Intercept),

  # Fixed effects: Weakly informative
  prior(normal(0, 0.5), class = b),

  # Random effect SDs: Exponential (encourages shrinkage)
  prior(exponential(10), class = sd, group = player_id),
  prior(exponential(20), class = sd, group = team),
  prior(exponential(20), class = sd, group = opponent),
  prior(exponential(10), class = sd, group = player_season)
)
```

**Informative priors (for Informative Priors v2):**
```r
# Estimate from historical data (2015-2019)
prior_data <- data %>% filter(season <= 2019)

log_yards_mean <- mean(log(prior_data$yards))  # ~5.3
log_attempts_coef <- coef(lm(log(yards) ~ log(attempts), data = prior_data))[2]  # ~0.78
home_effect <- mean(log(prior_data$yards[prior_data$is_home == 1])) -
               mean(log(prior_data$yards[prior_data$is_home == 0]))  # ~0.02

# Informative priors based on empirical Bayes
priors <- c(
  prior(normal(5.3, 0.5), class = Intercept),                    # From historical mean
  prior(normal(0.78, 0.1), class = b, coef = log_attempts),      # From regression
  prior(normal(0.02, 0.01), class = b, coef = is_home),          # From historical effect
  prior(normal(0, 0.3), class = b),                              # Other fixed effects
  prior(exponential(10), class = sd, group = player_id),
  prior(exponential(20), class = sd, group = team),
  prior(exponential(20), class = sd, group = opponent),
  prior(exponential(10), class = sd, group = player_season)
)
```

**Prior choice guidelines:**
- Use weakly informative priors when domain knowledge is limited
- Use informative priors when you have strong empirical evidence
- Exponential priors for SDs enforce non-negativity and shrinkage
- Normal priors for regression coefficients are robust defaults

#### Step 6: Train Model

```r
library(brms)
library(cmdstanr)

# Start timing
start_time <- Sys.time()

# Fit model
model <- brm(
  formula = formula,
  data = data,
  family = gaussian(),
  prior = priors,

  # MCMC configuration
  chains = 4,                              # Run 4 independent chains
  iter = 6000,                             # 6000 total iterations per chain
  warmup = 3000,                           # First 3000 are warmup (discarded)
  cores = 4,                               # Parallel chains
  threads = threading(2),                  # 2 threads per chain (M1/M2 Macs)

  # Backend and control
  backend = "cmdstanr",                    # Use cmdstan (faster than rstan)
  control = list(
    adapt_delta = 0.99,                    # Higher = more accurate (slower)
    max_treedepth = 12                     # Deeper trees = better exploration
  ),

  seed = 42                                # Reproducibility
)

end_time <- Sys.time()
elapsed <- as.numeric(difftime(end_time, start_time, units = "mins"))

cat(sprintf("\n✓ Training complete in %.1f minutes\n", elapsed))
```

**MCMC parameters explained:**
- `chains = 4` - Standard practice (detect non-convergence)
- `iter = 6000` - Increased from 2000 for complex hierarchical models
- `warmup = 3000` - 50% warmup ratio (standard)
- `adapt_delta = 0.99` - High value reduces divergences (important for hierarchical models)
- `max_treedepth = 12` - Increased from 10 for deep posteriors
- `threads = threading(2)` - Use within-chain parallelization (Mac M-series only)

**Expected training times:**
- Hierarchical v2: 30-40 minutes (6,000 iterations, complex hierarchy)
- Informative Priors v2: 35-45 minutes (6,000 iterations, complex hierarchy)
- State-Space v2: 2-3 hours (custom Stan, Kalman filtering)

#### Step 7: Save Model

```r
# Save model to disk
model_path <- "models/bayesian/passing_yards_hierarchical_v2.rds"
saveRDS(model, model_path)

file_size_mb <- file.size(model_path) / 1024^2
cat(sprintf("✓ Model saved to %s (%.1f MB)\n", model_path, file_size_mb))
```

**File sizes:**
- Hierarchical v2: ~150-200 MB
- Informative Priors v2: ~150-200 MB
- State-Space v2: ~50-100 MB

**Loading saved model:**
```r
model <- readRDS("models/bayesian/passing_yards_hierarchical_v2.rds")
```

---

## Model-Specific Guides

### State-Space Random Walk v2

**Purpose:** Time-varying player skills with random walk dynamics
**File:** `R/state_space/train_state_space_v2_canonical.R`
**Training time:** 2-3 hours
**Expected MAE:** 47.77 yards (validated Nov 8, 2025)
**Status:** ✅ APPROVED for production (30% ensemble weight)

#### What is a State-Space Model?

**Intuition:** Player skill evolves over time (not static).

- **Hot streaks:** QB performing above baseline
- **Cold streaks:** QB performing below baseline
- **Injury recovery:** Gradual return to form
- **Aging curves:** Decline over career

**Mathematical formulation:**

**State equation** (skill evolution):
```
θ_{i,t} = θ_{i,t-1} + ω_{i,t}

where:
  θ_{i,t} = player i's skill at time t
  ω_{i,t} ~ N(0, Q) = process noise
```

**Observation equation** (measured performance):
```
y_{i,t} = θ_{i,t} + β_attempts * log(attempts_{i,t}) + β_home * is_home_{i,t} + ε_{i,t}

where:
  y_{i,t} = log(passing_yards_{i,t})
  ε_{i,t} ~ N(0, R) = measurement noise
```

**Key parameters:**
- `Q` - Process noise (how much skill changes per game)
- `R` - Measurement noise (random game-to-game variance)
- `θ_{i,t}` - Time-varying skill trajectory

#### Canonical Training Window

**Different from Hierarchical/IP models:**
- 2018-2024: Full weight (w = 1.0)
- 2015-2017: Exponential decay (H = 4)

**Why decay?** State-space models are sensitive to data recency. Old games less informative for current skill.

#### Implementation Steps

**Step 1: Load data**
```r
library(tidyverse)
library(DBI)
library(RPostgres)
library(cmdstanr)
library(posterior)

conn <- dbConnect(
  Postgres(),
  host = "localhost",
  port = 5544,
  dbname = "devdb01",
  user = "dro",
  password = Sys.getenv("DB_PASSWORD", "sicillionbillions")
)

# Query: Same as Step 2 in Training Workflow, but 2015-2024 range
data <- dbGetQuery(conn, query)
```

**Step 2: Apply exponential weights**
```r
# Decay function
exp_weight <- function(s, t, H) {
  0.5 ^ ((t - s) / H)
}

EVAL_YEAR <- 2025
H <- 4
DECAY_YEARS <- 2015:2017

data <- data %>%
  mutate(
    weight = ifelse(season %in% DECAY_YEARS, exp_weight(season, EVAL_YEAR, H), 1.0)
  )

# Verify weights
data %>%
  group_by(season) %>%
  summarise(mean_weight = mean(weight), .groups = 'drop') %>%
  print()
```

**Step 3: Prepare Stan data (ragged arrays)**

State-space models require ragged arrays (different number of games per player).

```r
# Create player index
player_ids <- unique(data$player_id)
N_players <- length(player_ids)

# Get games per player
games_per_player <- data %>%
  group_by(player_id) %>%
  summarise(n_games = n(), .groups = 'drop') %>%
  arrange(match(player_id, player_ids))

T_max <- max(games_per_player$n_games)
T_player <- games_per_player$n_games

# Initialize matrices
y <- matrix(0, nrow = N_players, ncol = T_max)
log_attempts <- matrix(0, nrow = N_players, ncol = T_max)
is_home <- matrix(0L, nrow = N_players, ncol = T_max)
observed <- matrix(0L, nrow = N_players, ncol = T_max)
weights <- matrix(1.0, nrow = N_players, ncol = T_max)

# Fill ragged arrays
for (i in 1:N_players) {
  pid <- player_ids[i]
  player_games <- data %>% filter(player_id == pid) %>% arrange(season, week)
  n_games <- nrow(player_games)

  y[i, 1:n_games] <- log(player_games$passing_yards + 1)
  log_attempts[i, 1:n_games] <- log(player_games$passing_attempts)
  is_home[i, 1:n_games] <- player_games$is_home
  observed[i, 1:n_games] <- 1L
  weights[i, 1:n_games] <- player_games$weight
}

# Stan data list
stan_data <- list(
  N_players = N_players,
  T_player = T_player,
  T_max = T_max,
  y = y,
  observed = observed,
  log_attempts = log_attempts,
  is_home = is_home,
  weights = weights
)
```

**Step 4: Compile and fit Stan model**

```r
# Compile Stan model
stan_file <- "models/stan/simplified_random_walk_weighted.stan"
mod <- cmdstan_model(stan_file)

# Fit model
fit <- mod$sample(
  data = stan_data,
  chains = 6,                    # 6 chains for state-space models
  parallel_chains = 6,
  iter_warmup = 1000,
  iter_sampling = 2000,
  adapt_delta = 0.95,
  max_treedepth = 12,
  refresh = 100,
  seed = 42
)
```

**Training time:** 2-3 hours (complex state dynamics)

**Step 5: Extract skill trajectories**

```r
# Extract time-varying skills
theta_draws <- fit$draws(variables = "theta", format = "draws_matrix")
theta_mean <- colMeans(theta_draws)

# Reshape to dataframe
skill_trajectories <- data.frame()
for (i in 1:N_players) {
  pid <- player_ids[i]
  player_info <- data %>% filter(player_id == pid) %>% head(1)

  for (t in 1:T_player[i]) {
    theta_idx <- sprintf("theta[%d,%d]", i, t)
    skill_trajectories <- bind_rows(
      skill_trajectories,
      data.frame(
        player_id = pid,
        player_name = player_info$player_name,
        game_num = t,
        skill_mean = theta_mean[theta_idx]
      )
    )
  }
}

write_csv(skill_trajectories, "models/bayesian/simplified_random_walk_skills_v2.csv")
```

**Step 6: Save model**

```r
fit$save_object("models/bayesian/simplified_random_walk_v2.rds")
```

#### Validation

**Check convergence:**
```r
library(posterior)

# Extract global parameters
global_params <- c("mu_league", "sigma_player", "Q", "R", "beta_attempts", "beta_home")
draws <- fit$draws(variables = global_params)
summary_df <- summarise_draws(draws)

print(summary_df %>% select(variable, mean, rhat, ess_bulk, ess_tail))

# Convergence checks
max_rhat <- max(summary_df$rhat, na.rm = TRUE)
min_ess <- min(summary_df$ess_bulk, na.rm = TRUE)

if (max_rhat > 1.01 || min_ess < 400) {
  stop("❌ CONVERGENCE FAILURE")
}

cat("✅ CONVERGENCE SUCCESS!\n")
```

**Expected parameters:**
- `mu_league`: 5.3-5.5 (league average log yards)
- `Q`: 0.01-0.05 (process noise - small for stable skills)
- `R`: 0.3-0.5 (measurement noise - larger than Q)
- `beta_attempts`: 0.7-0.9 (positive coefficient)
- `beta_home`: 0.01-0.05 (small home advantage)

#### Using State-Space Model for Predictions

```r
# Load model
fit <- readRDS("models/bayesian/simplified_random_walk_v2.rds")

# Get latest skill for each player
latest_skills <- skill_trajectories %>%
  group_by(player_id) %>%
  filter(game_num == max(game_num)) %>%
  select(player_id, skill_mean)

# Predict for new game
predict_state_space <- function(player_id, log_attempts, is_home, latest_skills) {
  skill <- latest_skills %>% filter(player_id == !!player_id) %>% pull(skill_mean)

  # Use posterior mean parameters (extract from fit)
  beta_attempts <- 0.78  # From fit
  beta_home <- 0.02      # From fit

  log_yards_pred <- skill + beta_attempts * log_attempts + beta_home * is_home
  yards_pred <- exp(log_yards_pred) - 1

  return(yards_pred)
}

# Example
predict_state_space(
  player_id = "00-0033873",  # Patrick Mahomes
  log_attempts = log(35),
  is_home = 1,
  latest_skills = latest_skills
)
```

---

### Hierarchical Bayesian v2

**Purpose:** Partial pooling across players, teams, opponents
**File:** `R/train_hierarchical_v2_production.R`
**Training time:** 30-40 minutes
**Expected MAE:** 30-45 yards
**Status:** ✅ Production ready (30% ensemble weight)

#### What is Hierarchical Modeling?

**Problem:** Low-sample players (rookies, backups) have noisy estimates.

**Solution:** Borrow strength from similar players/teams.

**Example:**
- **Rookie QB:** Only 3 games of data
- **Naive estimate:** 180 yards/game (high variance)
- **Hierarchical estimate:** Shrink toward league average (220 yards) and team effect
- **Result:** More stable, less overconfident

**Mathematical formulation:**
```
log_yards_{i,j,k} = α + β * X + α_player[i] + α_team[j] + α_opponent[k] + ε

where:
  α_player[i] ~ N(0, σ_player)    # Player random effect
  α_team[j] ~ N(0, σ_team)        # Team random effect
  α_opponent[k] ~ N(0, σ_opponent) # Opponent defense effect
```

**Key insight:** Players, teams, opponents share a common distribution (partial pooling).

#### Canonical Training Window

**Uniform 2015-2025** (no decay)

**Rationale:** Ablation study showed Scenario D (uniform 2015-2025) equivalent to Scenario B (2018-2025 + decay). Uniform is simpler for brms.

#### Implementation

**Full script:** See `R/train_hierarchical_v2_production.R` (241 lines)

**Key formula:**
```r
formula <- bf(
  log_yards ~ 1 +
    log_attempts +
    is_home +
    is_favored +
    spread_abs +
    is_bad_weather +
    is_dome +
    scale(total_close) +
    scale(years_exp) +
    (1 | player_id) +           # Partial pooling: players
    (1 | team) +                # Partial pooling: teams
    (1 | opponent) +            # Partial pooling: opponent defenses
    (log_attempts | player_season),  # Varying slope: player-season form
  sigma ~ log_attempts          # Heteroskedasticity modeling
)
```

**Training:**
```bash
# Run from project root
Rscript R/train_hierarchical_v2_production.R
```

**Output:**
- Model: `models/bayesian/passing_yards_hierarchical_v2.rds`
- Size: ~150-200 MB
- Convergence: Rhat < 1.01, ESS > 400

#### Interpreting Random Effects

**Extract random effects:**
```r
library(brms)

model <- readRDS("models/bayesian/passing_yards_hierarchical_v2.rds")

# Player effects
player_effects <- ranef(model)$player_id

# Top 10 players (above league average)
head(player_effects[order(-player_effects[, "Estimate", "Intercept"]), ], 10)

# Bottom 10 players (below league average)
tail(player_effects[order(-player_effects[, "Estimate", "Intercept"]), ], 10)
```

**Expected values:**
- Elite QBs (Mahomes, Allen): +0.15 to +0.25 log yards (~30-60 yards/game above average)
- Average QBs: -0.05 to +0.05 log yards (within 10 yards of average)
- Backup QBs: -0.15 to -0.30 log yards (20-60 yards below average)

---

### Informative Priors v2

**Purpose:** Incorporate domain knowledge and empirical Bayes
**File:** `R/train_informative_priors_v2_production.R`
**Training time:** 35-45 minutes
**Expected MAE:** 30-45 yards
**Status:** ✅ Production ready (15% ensemble weight)

#### What are Informative Priors?

**Weakly informative priors (default):**
```r
prior(normal(0, 0.5), class = b)  # "I have no idea what β is"
```

**Informative priors (domain knowledge):**
```r
prior(normal(0.78, 0.1), class = b, coef = log_attempts)  # "β ≈ 0.78 based on 5 years of data"
```

**Benefits:**
1. **Faster convergence:** Less exploration needed
2. **Better low-sample estimates:** Rookies shrink toward informed prior
3. **Regularization:** Prevents overfitting
4. **Domain expertise:** Incorporates football knowledge

#### Empirical Bayes Prior Estimation

**Step 1: Split data**
```r
# Use first half for prior estimation
prior_data <- data %>% filter(season <= 2019)
training_data <- data  # All data (2015-2025)
```

**Step 2: Estimate key parameters**
```r
# Intercept: Historical mean
log_yards_mean <- mean(log(prior_data$yards))  # 5.30
log_yards_sd <- sd(log(prior_data$yards))      # 0.60

# Attempts coefficient: Regression
lm_fit <- lm(log(yards) ~ log(attempts), data = prior_data)
log_attempts_coef <- coef(lm_fit)[2]  # 0.78

# Home advantage: Difference in means
home_effect <- mean(log(prior_data$yards[prior_data$is_home == 1])) -
               mean(log(prior_data$yards[prior_data$is_home == 0]))  # 0.02

cat(sprintf("Estimated priors:\n"))
cat(sprintf("  Intercept: N(%.2f, %.2f)\n", log_yards_mean, log_yards_sd / 2))
cat(sprintf("  log_attempts: N(%.3f, 0.1)\n", log_attempts_coef))
cat(sprintf("  home effect: N(%.3f, 0.01)\n", home_effect))
```

**Step 3: Specify informative priors**
```r
priors <- c(
  # Informative intercept (from historical data)
  prior(normal(5.3, 0.5), class = Intercept),

  # Informative attempts coefficient
  prior(normal(0.78, 0.1), class = b, coef = log_attempts),

  # Informative home advantage
  prior(normal(0.02, 0.01), class = b, coef = is_home),

  # Weakly informative for other fixed effects
  prior(normal(0, 0.3), class = b),

  # Hierarchical SDs (exponential for shrinkage)
  prior(exponential(10), class = sd, group = player_id),
  prior(exponential(20), class = sd, group = team),
  prior(exponential(20), class = sd, group = opponent),
  prior(exponential(10), class = sd, group = player_season)
)
```

**Step 4: Train model**
```bash
Rscript R/train_informative_priors_v2_production.R
```

**Output:**
- Model: `models/bayesian/passing_informative_priors_v2.rds`
- Size: ~150-200 MB

#### Prior Predictive Checks

**Validate priors before fitting:**
```r
# Sample from prior (no data)
prior_samples <- brm(
  formula = formula,
  data = data,
  prior = priors,
  family = gaussian(),
  sample_prior = "only",  # Sample from prior only
  chains = 1,
  iter = 1000,
  backend = "cmdstanr"
)

# Check if prior predictions are reasonable
library(bayesplot)
pp_check(prior_samples, ndraws = 50)
```

**Expected:** Prior predictions should cover plausible range (100-400 yards), not extreme values (0-1000 yards).

---

## MCMC Diagnostics (Critical Validation)

**⚠️ CRITICAL:** Always check diagnostics before using model. Deployed models MUST pass all checks.

### Convergence Criteria

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| **Rhat** | < 1.01 | Chains have converged to same distribution |
| **ESS (bulk)** | > 400 | Sufficient effective samples in posterior center |
| **ESS (tail)** | > 400 | Sufficient effective samples in posterior tails |
| **Divergences** | < 10 | No numerical instabilities in sampling |
| **Max treedepth** | < 5% of samples | Efficient exploration of posterior |

### Checking Diagnostics

```r
library(brms)
library(posterior)

model <- readRDS("models/bayesian/passing_yards_hierarchical_v2.rds")

# 1. Summary statistics
summary(model)

# 2. Extract Rhat values
rhats <- rhat(model)
max_rhat <- max(rhats, na.rm = TRUE)
cat(sprintf("Max Rhat: %.4f (threshold: < 1.01)\n", max_rhat))

# 3. Extract ESS values
ess_bulk_values <- ess_bulk(model)
ess_tail_values <- ess_tail(model)
min_ess_bulk <- min(ess_bulk_values, na.rm = TRUE)
min_ess_tail <- min(ess_tail_values, na.rm = TRUE)

cat(sprintf("Min ESS (bulk): %.0f (threshold: > 400)\n", min_ess_bulk))
cat(sprintf("Min ESS (tail): %.0f (threshold: > 400)\n", min_ess_tail))

# 4. Check divergences
num_divergent <- sum(nuts_params(model)$Value[nuts_params(model)$Parameter == "divergent__"])
cat(sprintf("Divergent transitions: %d (threshold: < 10)\n", num_divergent))

# 5. Check max treedepth
max_treedepth_hit <- sum(nuts_params(model)$Value[nuts_params(model)$Parameter == "treedepth__"] >= 12)
total_samples <- nrow(as_draws_df(model))
treedepth_pct <- 100 * max_treedepth_hit / total_samples

cat(sprintf("Max treedepth hit: %.1f%% (threshold: < 5%%)\n", treedepth_pct))
```

### Interpreting Diagnostics

#### Rhat (Gelman-Rubin statistic)

**What it measures:** Do all chains converge to same distribution?

**Interpretation:**
- Rhat = 1.00: Perfect convergence
- Rhat < 1.01: Acceptable convergence ✅
- Rhat > 1.01: Non-convergence ❌

**If Rhat > 1.01:**
1. Increase `iter` (e.g., 2000 → 6000)
2. Increase `adapt_delta` (e.g., 0.95 → 0.99)
3. Check for identification issues (collinear predictors)

**Visual check:**
```r
library(bayesplot)
mcmc_trace(model, pars = c("b_Intercept", "b_log_attempts", "b_is_home"))
```

**Good trace plot:**
- Chains overlap (not separated)
- Rapid mixing (fuzzy caterpillar, not smooth lines)
- Stationary (no trends up/down)

#### ESS (Effective Sample Size)

**What it measures:** How many independent samples from posterior?

**Why lower than actual samples?** Autocorrelation in MCMC chains.

**Example:**
- 4 chains × 3000 samples = 12,000 total samples
- ESS (bulk) = 5,000 → Equivalent to 5,000 independent samples
- ESS (tail) = 3,000 → Equivalent to 3,000 independent samples in tails

**Interpretation:**
- ESS > 1000: Excellent ✅
- ESS > 400: Acceptable ✅
- ESS < 400: Increase iterations ❌

**If ESS < 400:**
1. Increase `iter` (e.g., 2000 → 6000)
2. Check for high autocorrelation (increase `adapt_delta`)
3. Reparameterize model (use non-centered parameterization)

#### Divergent Transitions

**What they indicate:** Numerical instabilities during sampling (gradient issues).

**Common causes:**
1. **Too low `adapt_delta`:** Stepsize too large
2. **Poorly identified model:** Collinear predictors, flat priors
3. **Neal's funnel:** Hierarchical model with extreme shrinkage

**If divergences > 10:**
1. Increase `adapt_delta` (0.95 → 0.99)
2. Use stronger priors on SD parameters
3. Check for collinearity (`cor(data)`)
4. Reparameterize (non-centered → centered or vice versa)

**Visual check:**
```r
library(bayesplot)
mcmc_nuts_divergence(nuts_params(model), lp = log_posterior(model))
```

#### Posterior Predictive Checks

**Goal:** Does model reproduce data distribution?

```r
library(bayesplot)

# Plot observed vs predicted
pp_check(model, ndraws = 100)

# Check specific statistics
pp_check(model, type = "stat", stat = "mean")
pp_check(model, type = "stat", stat = "sd")

# Check by covariate
pp_check(model, type = "intervals_grouped", group = "is_home")
```

**Good pp_check:**
- Blue density (predicted) overlaps dark line (observed)
- No systematic bias (over/under prediction)
- Similar spread (not too narrow/wide)

---

## Common Errors & Solutions

### Error: "Stan model compilation failed"

**Symptoms:**
```
Error in compileCode(f, code, language = language, verbose = verbose) :
  Compilation ERROR, function(s)/method(s) not created!
```

**Causes:**
1. cmdstan not installed
2. C++ compiler not found
3. Incompatible cmdstan version

**Solutions:**

**Check cmdstan installation:**
```r
library(cmdstanr)
cmdstan_path()
cmdstan_version()
```

**Reinstall cmdstan:**
```r
cmdstanr::install_cmdstan(overwrite = TRUE, cores = 4)
```

**Check C++ compiler:**
```bash
# macOS
xcode-select -p  # Should return /Applications/Xcode.app/...
# If not: xcode-select --install

# Linux
gcc --version  # Should return gcc version
# If not: sudo apt-get install build-essential
```

**Update cmdstanr:**
```r
install.packages(
  "cmdstanr",
  repos = c("https://mc-stan.org/r-packages/", getOption("repos"))
)
```

---

### Error: "Rhat > 1.01" (Non-convergence)

**Symptoms:**
```
Warning: 15 of 24000 iterations ended with a divergence.
Warning: The largest R-hat is 1.08, indicating chains have not mixed.
```

**Causes:**
1. Too few iterations
2. Complex posterior geometry
3. Poor parameterization
4. Identification issues

**Solutions:**

**Solution 1: Increase iterations**
```r
# Before (default)
model <- brm(..., iter = 2000, warmup = 1000)

# After (more iterations)
model <- brm(..., iter = 6000, warmup = 3000)
```

**Solution 2: Increase adapt_delta**
```r
# Before
model <- brm(..., control = list(adapt_delta = 0.95))

# After (higher adapt_delta)
model <- brm(..., control = list(adapt_delta = 0.99))
```

**Solution 3: Check for collinearity**
```r
# Check correlation matrix
cor_matrix <- cor(data %>% select(log_attempts, is_home, spread_abs, total_close))
print(cor_matrix)

# If |cor| > 0.9, remove one variable or use PCA
```

**Solution 4: Reparameterize model**
```r
# Non-centered parameterization (better for hierarchical models)
# Before:
formula <- bf(log_yards ~ 1 + (1 | player_id))

# After (explicitly non-centered):
# (brms uses non-centered by default, so this is rare)
```

**Solution 5: Stronger priors**
```r
# Before (weakly informative)
prior(normal(0, 10), class = b)

# After (more informative)
prior(normal(0, 1), class = b)
```

---

### Error: "Low ESS" (< 400)

**Symptoms:**
```
Warning: Bulk Effective Samples Size (ESS) is too low, indicating posterior estimates are unreliable.
```

**Causes:**
1. High autocorrelation
2. Too few iterations
3. Poor mixing

**Solutions:**

**Solution 1: Increase iterations**
```r
model <- brm(..., iter = 8000, warmup = 4000)  # Double iterations
```

**Solution 2: Thin chains**
```r
# Reduce autocorrelation by thinning
model <- brm(..., iter = 10000, warmup = 5000, thin = 5)
# Result: 1000 saved samples per chain (every 5th sample)
```

**Solution 3: Check trace plots**
```r
library(bayesplot)
mcmc_trace(model, pars = c("b_Intercept", "b_log_attempts"))

# Look for:
# - Slow mixing (smooth lines)
# - Stuck chains (flat regions)
# - High autocorrelation (trends)
```

**Solution 4: Increase adapt_delta**
```r
model <- brm(..., control = list(adapt_delta = 0.99, max_treedepth = 15))
```

---

### Error: "Divergent transitions"

**Symptoms:**
```
Warning: There were 47 divergent transitions after warmup.
```

**Causes:**
1. Stepsize too large (numerical instability)
2. Neal's funnel (hierarchical models with extreme shrinkage)
3. Poorly identified model

**Solutions:**

**Solution 1: Increase adapt_delta** (MOST COMMON FIX)
```r
# Before
model <- brm(..., control = list(adapt_delta = 0.95))

# After
model <- brm(..., control = list(adapt_delta = 0.99))
```

**Solution 2: Stronger priors on SDs**
```r
# Before (too vague)
prior(exponential(1), class = sd, group = player_id)

# After (more informative)
prior(exponential(10), class = sd, group = player_id)
```

**Solution 3: Check for Neal's funnel**
```r
# Extract SD parameter and random effects
sd_player <- as_draws_df(model)$sd_player_id__Intercept
ranef_player <- ranef(model)$player_id[, "Estimate", "Intercept"]

# Plot
plot(sd_player, ranef_player[1], xlab = "SD", ylab = "Random Effect")
# If funnel shape (V or Y), reparameterize
```

**Solution 4: Non-centered parameterization** (advanced)
```r
# brms uses non-centered by default, but you can force centered:
# (Rarely needed - consult Stan documentation)
```

---

### Error: "Memory allocation failed"

**Symptoms:**
```
Error: cannot allocate vector of size 8.0 Gb
```

**Causes:**
1. Too much data
2. Too many parameters (complex hierarchy)
3. Too many iterations saved

**Solutions:**

**Solution 1: Reduce data size**
```r
# Subsample large datasets
data_subset <- data %>%
  filter(season >= 2018) %>%  # Recent data only
  sample_frac(0.8)  # Use 80% of data
```

**Solution 2: Reduce saved iterations**
```r
# Before
model <- brm(..., iter = 6000, warmup = 3000)  # Saves 12,000 samples (4 chains × 3000)

# After (thin to save memory)
model <- brm(..., iter = 6000, warmup = 3000, thin = 3)  # Saves 4,000 samples
```

**Solution 3: Reduce warmup**
```r
# Before
model <- brm(..., iter = 6000, warmup = 3000)

# After (less warmup, more sampling)
model <- brm(..., iter = 4000, warmup = 500)  # Faster, but check convergence!
```

**Solution 4: Run on high-memory machine**
```bash
# Transfer to Windows RTX 4090 machine (64 GB RAM)
# Use Git LFS for model transfer (see docs/guides/GIT_LFS_WORKFLOW.md)
```

**Solution 5: Save posterior samples separately**
```r
# Fit model without saving object
model <- brm(..., save_pars = save_pars(all = FALSE))  # Don't save all parameters

# Save only key parameters
posterior_samples <- as_draws_df(model) %>%
  select(b_Intercept, b_log_attempts, b_is_home)

write_csv(posterior_samples, "posterior_samples.csv")
```

---

### Error: "Database connection timeout"

**Symptoms:**
```
Error: could not connect to server: Connection refused
```

**Causes:**
1. PostgreSQL not running
2. Wrong port/host
3. Firewall blocking connection

**Solutions:**

**Solution 1: Check PostgreSQL status**
```bash
# Check if PostgreSQL is running (Docker)
docker ps | grep postgres

# If not running, start it
docker compose up -d pg
```

**Solution 2: Verify connection details**
```r
# Print environment variables
cat("Host:", Sys.getenv("DB_HOST", "localhost"), "\n")
cat("Port:", Sys.getenv("DB_PORT", "5544"), "\n")
cat("Database:", Sys.getenv("DB_NAME", "devdb01"), "\n")

# Test connection
conn <- dbConnect(
  Postgres(),
  host = "localhost",
  port = 5544,
  dbname = "devdb01",
  user = "dro",
  password = Sys.getenv("DB_PASSWORD")
)
```

**Solution 3: Check Docker network**
```bash
# List Docker networks
docker network ls

# Inspect network
docker network inspect nfl-analytics_default
```

**Solution 4: Use localhost alias**
```r
# macOS: Sometimes localhost doesn't resolve
conn <- dbConnect(
  Postgres(),
  host = "127.0.0.1",  # Use IP instead of localhost
  port = 5544,
  dbname = "devdb01",
  user = "dro",
  password = Sys.getenv("DB_PASSWORD")
)
```

---

### Error: "Column 'game_id' does not exist"

**Symptoms:**
```sql
SELECT pgs.*, g.spread_close
FROM mart.player_game_stats pgs
JOIN games g ON g.game_id = pgs.game_id

ERROR:  column pgs.game_id does not exist
```

**Cause:** `mart.player_game_stats` does NOT have `game_id` column.

**Solution:** Join via `(season, week)` composite key.

**See:** `docs/database/DATABASE_QUICK_REFERENCE.md` (lines 9-53)

**Correct SQL:**
```sql
SELECT pgs.*, g.spread_close
FROM mart.player_game_stats pgs
JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
LEFT JOIN games g ON
    g.season = pgs.season
    AND g.week = pgs.week
    AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
```

**Validate query before running:**
```python
from py.utils.sql_validator import validate_query

errors = validate_query("YOUR SQL QUERY HERE")
if errors:
    print("Errors found:", errors)
```

---

## Model Registry Integration

**Purpose:** Track trained models in database for versioning and deployment.

**Database table:** `ml.model_metadata`

### Manual Registration

```bash
# Register model using CLI
./scripts/model_registry.sh register \
  --model models/bayesian/passing_yards_hierarchical_v2.rds \
  --name "Hierarchical Bayesian v2" \
  --type "brms" \
  --stat_type "passing_yards" \
  --training_window "2015-2025" \
  --convergence_rhat 1.005 \
  --convergence_ess 4200
```

### Automatic Registration (Python)

```python
import joblib
from py.utils.model_registry import ModelRegistry

# Load model metadata
registry = ModelRegistry()

# Register model
registry.register_model(
    name="Hierarchical Bayesian v2",
    model_type="brms",
    stat_type="passing_yards",
    model_path="models/bayesian/passing_yards_hierarchical_v2.rds",
    training_window="2015-2025",
    metadata={
        "convergence": {
            "max_rhat": 1.005,
            "min_ess_bulk": 4200,
            "min_ess_tail": 3800,
            "divergences": 0
        },
        "performance": {
            "mae": 32.5,
            "calibration_coverage": 0.91
        }
    }
)
```

### Validation Before Registration

**⚠️ CRITICAL:** Only register models that pass convergence checks.

```r
# Convergence validation function
validate_convergence <- function(model, model_name) {
  rhats <- rhat(model)
  max_rhat <- max(rhats, na.rm = TRUE)

  ess_bulk_values <- ess_bulk(model)
  min_ess_bulk <- min(ess_bulk_values, na.rm = TRUE)

  ess_tail_values <- ess_tail(model)
  min_ess_tail <- min(ess_tail_values, na.rm = TRUE)

  # Check thresholds
  passed <- TRUE
  if (max_rhat > 1.01) {
    cat(sprintf("❌ %s: Rhat = %.4f (> 1.01)\n", model_name, max_rhat))
    passed <- FALSE
  }

  if (min_ess_bulk < 400) {
    cat(sprintf("❌ %s: ESS (bulk) = %.0f (< 400)\n", model_name, min_ess_bulk))
    passed <- FALSE
  }

  if (min_ess_tail < 400) {
    cat(sprintf("❌ %s: ESS (tail) = %.0f (< 400)\n", model_name, min_ess_tail))
    passed <- FALSE
  }

  if (passed) {
    cat(sprintf("✅ %s: All convergence checks passed\n", model_name))
  }

  return(passed)
}

# Usage
model <- readRDS("models/bayesian/passing_yards_hierarchical_v2.rds")
if (validate_convergence(model, "Hierarchical v2")) {
  # Safe to register
}
```

---

## Production Checklist

**Use this checklist before deploying any R Bayesian model to production.**

### Training

- [ ] Model trained on canonical window (2018-2024 + H=4 decay OR uniform 2015-2025)
- [ ] Training data extracted from database (not CSV files)
- [ ] All features engineered correctly (log transformations, scaling)
- [ ] Model formula matches production specification
- [ ] Priors specified (weakly informative or informative)
- [ ] MCMC configuration documented (chains, iter, warmup, adapt_delta)

### Convergence

- [ ] Max Rhat < 1.01 (all parameters)
- [ ] Min ESS (bulk) > 400 (all parameters)
- [ ] Min ESS (tail) > 400 (all parameters)
- [ ] Divergent transitions < 10
- [ ] Max treedepth hit < 5% of samples
- [ ] Trace plots show good mixing (visual check)
- [ ] Posterior predictive checks look reasonable

### Performance

- [ ] MAE validated on holdout set (2025 season)
- [ ] MAE within expected range for model type:
  - State-Space v2: 45-50 yards
  - Hierarchical v2: 30-45 yards
  - Informative Priors v2: 30-45 yards
- [ ] Calibration validated (coverage 85-95%)
- [ ] No systematic bias (residuals centered at 0)

### Model Artifact

- [ ] Model saved to `models/bayesian/` directory
- [ ] File size reasonable (< 500 MB)
- [ ] Model loads successfully (`readRDS()` works)
- [ ] Predictions run without errors
- [ ] Model metadata documented (training time, data size, convergence)

### Registration

- [ ] Model registered in database (`ml.model_metadata`)
- [ ] Model name follows convention: `{stat_type}_{method}_v{version}`
- [ ] Model type specified correctly (brms, cmdstan, stan)
- [ ] Training window recorded
- [ ] Convergence metrics stored in metadata
- [ ] File path stored in database

### Integration

- [ ] Model loaded successfully in ensemble (`py/ensemble/enhanced_ensemble_v5.py`)
- [ ] rpy2 integration working (Python ↔ R communication)
- [ ] Predictions match expected format (mean, std)
- [ ] Ensemble weights configured correctly
- [ ] Temperature scaler applied (if needed)

### Testing

- [ ] Dry-run backtest on 2024 season (historical validation)
- [ ] No errors or warnings during backtest
- [ ] Performance metrics meet thresholds:
  - MAE improvement > 5 yards vs baseline
  - Calibration coverage > 85%
  - Sharpe ratio > 1.5
- [ ] Model predictions stable (no extreme values)

### Deployment

- [ ] Model documented in `CLAUDE.md` (ensemble configuration)
- [ ] Model documented in `models/MODEL_REGISTRY.yaml`
- [ ] Training script committed to Git
- [ ] Model file committed to Git LFS (if < 100 MB) or documented in registry
- [ ] Deployment checklist reviewed
- [ ] Monitoring configured (track predictions, errors, performance)

---

## Troubleshooting Guide

### Debugging Stan Compilation Errors

**Symptom:** Stan model won't compile.

**Steps:**
1. **Check cmdstan installation:**
   ```r
   cmdstan_path()
   cmdstan_version()
   ```

2. **Check Stan file syntax:**
   ```bash
   # Validate Stan syntax
   cmdstan_model("models/stan/your_model.stan", compile = FALSE)
   ```

3. **Enable verbose output:**
   ```r
   mod <- cmdstan_model("models/stan/your_model.stan", compile = TRUE, quiet = FALSE)
   ```

4. **Check C++ compiler:**
   ```bash
   # macOS
   clang --version

   # Linux
   gcc --version
   ```

5. **Reinstall cmdstan:**
   ```r
   cmdstanr::install_cmdstan(overwrite = TRUE)
   ```

---

### Debugging Convergence Issues

**Symptom:** Rhat > 1.01, low ESS, divergences.

**Step 1: Increase adapt_delta**
```r
# Start with 0.99
model <- brm(..., control = list(adapt_delta = 0.99))

# If still diverging, try 0.999
model <- brm(..., control = list(adapt_delta = 0.999, max_treedepth = 15))
```

**Step 2: Check trace plots**
```r
library(bayesplot)

# Extract posterior samples
posterior <- as.array(model)

# Plot traces for key parameters
mcmc_trace(posterior, pars = c("b_Intercept", "b_log_attempts", "sd_player_id__Intercept"))
```

**What to look for:**
- **Good:** Chains overlap, fuzzy caterpillar
- **Bad:** Chains separated, smooth lines, trends

**Step 3: Check pairs plots**
```r
# Check for Neal's funnel
mcmc_pairs(
  posterior,
  pars = c("b_Intercept", "sd_player_id__Intercept"),
  diag_fun = "dens",
  off_diag_fun = "hex"
)
```

**What to look for:**
- **Good:** Elliptical or circular scatter
- **Bad:** Funnel shape (V or Y)

**Step 4: Simplify model**
```r
# Remove complex terms one at a time
# Before (complex):
formula <- bf(
  log_yards ~ 1 + log_attempts + is_home +
    (1 | player_id) + (1 | team) + (1 | opponent) + (log_attempts | player_season)
)

# After (simplified):
formula <- bf(
  log_yards ~ 1 + log_attempts + is_home +
    (1 | player_id)
)

# Refit and check convergence
```

**Step 5: Check for collinearity**
```r
# Correlation matrix
cor_matrix <- cor(data %>% select(log_attempts, spread_abs, total_close, years_exp))
print(cor_matrix)

# If |cor| > 0.9, remove one variable
```

---

### Debugging Memory Issues

**Symptom:** "Cannot allocate vector of size X GB"

**Solution 1: Increase R memory limit**
```r
# Check current limit
memory.limit()  # Windows only

# Increase limit (Windows)
memory.limit(size = 32000)  # 32 GB

# macOS/Linux: Set environment variable
Sys.setenv("R_MAX_VSIZE" = 64 * 1024^3)  # 64 GB
```

**Solution 2: Reduce data size**
```r
# Subsample large datasets
data_subset <- data %>%
  filter(season >= 2020) %>%  # Recent seasons only
  sample_frac(0.7)  # Use 70% of data
```

**Solution 3: Thin posterior samples**
```r
model <- brm(..., iter = 6000, warmup = 3000, thin = 5)
# Saves only every 5th sample (reduces memory by 80%)
```

**Solution 4: Don't save all parameters**
```r
model <- brm(..., save_pars = save_pars(all = FALSE))
# Only save fixed effects and key parameters
```

---

### Debugging rpy2 Integration Issues

**Symptom:** Model trained in R but won't load in Python ensemble.

**Common error:**
```python
rpy2.rinterface_lib.embedded.RRuntimeError: Error in readRDS("model.rds") : unknown input format
```

**Solution 1: Check R version compatibility**
```bash
# R version (should be 4.0+)
R --version

# Python rpy2 version
python -c "import rpy2; print(rpy2.__version__)"
```

**Solution 2: Save model with version info**
```r
# Save with R version metadata
saveRDS(model, "model.rds", version = 3)  # RDS format version 3 (most compatible)
```

**Solution 3: Test loading in R first**
```r
# Verify model loads in R
model <- readRDS("models/bayesian/passing_yards_hierarchical_v2.rds")
summary(model)
```

**Solution 4: Check file permissions**
```bash
# Verify file is readable
ls -lh models/bayesian/passing_yards_hierarchical_v2.rds

# Fix permissions if needed
chmod 644 models/bayesian/passing_yards_hierarchical_v2.rds
```

**Solution 5: Use R interface guide**

See `docs/guides/R_MODEL_INTERFACE_GUIDE.md` for complete Python ↔ R integration instructions.

---

## References & Resources

### Official Documentation

**brms:**
- Documentation: https://paul-buerkner.github.io/brms/
- Vignettes: https://paul-buerkner.github.io/brms/articles/
- Formula syntax: https://paul-buerkner.github.io/brms/reference/brmsformula.html
- Prior specification: https://paul-buerkner.github.io/brms/reference/set_prior.html

**cmdstanr:**
- Documentation: https://mc-stan.org/cmdstanr/
- Installation: https://mc-stan.org/cmdstanr/articles/cmdstanr.html
- Diagnostics: https://mc-stan.org/cmdstanr/articles/posterior.html

**Stan:**
- User's Guide: https://mc-stan.org/docs/stan-users-guide/
- Language Reference: https://mc-stan.org/docs/reference-manual/
- Prior Choice: https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations

**posterior:**
- Documentation: https://mc-stan.org/posterior/
- Diagnostics: https://mc-stan.org/posterior/reference/diagnostics.html
- Convergence checks: https://mc-stan.org/posterior/reference/convergence.html

### Papers and Books

**Bayesian Workflow:**
- Gelman et al. (2020): "Bayesian Workflow" - https://arxiv.org/abs/2011.01808
- Betancourt (2017): "A Conceptual Introduction to Hamiltonian Monte Carlo" - https://arxiv.org/abs/1701.02434

**Hierarchical Models:**
- Gelman & Hill (2007): "Data Analysis Using Regression and Multilevel/Hierarchical Models"
- McElreath (2020): "Statistical Rethinking" (2nd edition)

**State-Space Models:**
- Durbin & Koopman (2012): "Time Series Analysis by State Space Methods" (2nd edition)
- Petris et al. (2009): "Dynamic Linear Models with R"

### Project-Specific Documentation

**Architecture:**
- [CLAUDE.md](CLAUDE.md) - Complete project documentation
- [docs/ARCHITECTURE_REFERENCE.md](docs/ARCHITECTURE_REFERENCE.md) - Architecture deep dive
- [docs/COMPONENT_DEEP_DIVES.md](docs/COMPONENT_DEEP_DIVES.md) - Component details

**Database:**
- [docs/database/DATABASE_QUICK_REFERENCE.md](docs/database/DATABASE_QUICK_REFERENCE.md) - Schema reference
- [docs/database/SQL_PATTERNS.md](docs/database/SQL_PATTERNS.md) - Common query patterns

**Guides:**
- [docs/guides/R_MODEL_INTERFACE_GUIDE.md](docs/guides/R_MODEL_INTERFACE_GUIDE.md) - Python ↔ R integration
- [docs/guides/GIT_LFS_WORKFLOW.md](docs/guides/GIT_LFS_WORKFLOW.md) - Large model file management
- [docs/guides/PRODUCTION_DEPLOYMENT.md](docs/guides/PRODUCTION_DEPLOYMENT.md) - Deployment checklist

**Milestones:**
- [docs/archive/2025_nov/STATE_SPACE_VALIDATION_REPORT.md](docs/archive/2025_nov/STATE_SPACE_VALIDATION_REPORT.md) - State-Space v2 validation
- [docs/milestones/HIERARCHICAL_V2_VERIFICATION_COMPLETE.md](docs/milestones/HIERARCHICAL_V2_VERIFICATION_COMPLETE.md) - Hierarchical v2 verification
- [docs/ADVANCED_BAYESIAN_V3.md](docs/ADVANCED_BAYESIAN_V3.md) - Advanced Bayesian enhancements

### Community Resources

**Stan Forums:**
- Stan Discourse: https://discourse.mc-stan.org/
- Search before posting (many common issues already answered)

**Stack Overflow:**
- [brms] tag: https://stackoverflow.com/questions/tagged/brms
- [stan] tag: https://stackoverflow.com/questions/tagged/stan
- [r] tag: https://stackoverflow.com/questions/tagged/r

**GitHub Issues:**
- brms issues: https://github.com/paul-buerkner/brms/issues
- cmdstanr issues: https://github.com/stan-dev/cmdstanr/issues

---

## Appendix: Example Training Scripts

### Minimal Example (Hello World)

```r
#!/usr/bin/env Rscript
# minimal_bayesian_example.R

library(brms)
library(cmdstanr)

# Synthetic data
set.seed(42)
data <- data.frame(
  x = rnorm(100),
  y = 2 * rnorm(100) + rnorm(100, sd = 0.5)
)

# Fit model
model <- brm(
  y ~ x,
  data = data,
  family = gaussian(),
  prior = c(
    prior(normal(0, 1), class = Intercept),
    prior(normal(0, 1), class = b)
  ),
  chains = 2,
  iter = 1000,
  warmup = 500,
  backend = "cmdstanr",
  cores = 2,
  seed = 42
)

# Check convergence
summary(model)

# Save
saveRDS(model, "minimal_model.rds")
```

### Production Example (Hierarchical v2)

See `R/train_hierarchical_v2_production.R` for complete production script (241 lines).

**Key features:**
- Database connection
- Feature engineering
- Hierarchical formula
- Weakly informative priors
- MCMC configuration (6000 iter, adapt_delta 0.99)
- Convergence validation
- Model persistence

---

**Version:** 1.0
**Last Updated:** November 16, 2025
**Maintainer:** Claude Code (Sonnet 4.5)
**Questions?** See [docs/guides/TROUBLESHOOTING_MASTER.md](docs/guides/TROUBLESHOOTING_MASTER.md)
