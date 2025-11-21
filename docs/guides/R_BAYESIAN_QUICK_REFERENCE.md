# R Bayesian Modeling Quick Reference

**Last Updated:** November 16, 2025
**Full Cookbook:** [R_BAYESIAN_MODELING_COOKBOOK.md](R_BAYESIAN_MODELING_COOKBOOK.md)

---

## 🚀 Quick Start (Copy-Paste)

### Install Dependencies (One-Time)
```bash
# From project root
Rscript setup_packages.R

# Install cmdstan backend
R --vanilla <<'EOF'
install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
cmdstanr::install_cmdstan(cores = 4)
EOF
```

### Train Production Models
```bash
# Hierarchical Bayesian v2 (30-40 min)
Rscript R/train_hierarchical_v2_production.R

# Informative Priors v2 (35-45 min)
Rscript R/train_informative_priors_v2_production.R

# State-Space v2 (2-3 hours)
Rscript R/state_space/train_state_space_v2_canonical.R
```

---

## 📋 Production Model Summary

| Model | Weight | File | Training Time | Expected MAE |
|-------|--------|------|---------------|--------------|
| **State-Space v2** | 30% | `R/state_space/train_state_space_v2_canonical.R` | 2-3 hours | 47.77 yards |
| **Hierarchical v2** | 30% | `R/train_hierarchical_v2_production.R` | 30-40 min | 30-45 yards |
| **Informative Priors v2** | 15% | `R/train_informative_priors_v2_production.R` | 35-45 min | 30-45 yards |

**Total ensemble R contribution:** 75% (State-Space 30% + Hierarchical 30% + IP 15%)

---

## 🔍 Common Commands

### Database Connection
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

# Test connection
dbGetQuery(conn, "SELECT COUNT(*) FROM games;")

dbDisconnect(conn)
```

### Load Saved Model
```r
library(brms)

model <- readRDS("models/bayesian/passing_yards_hierarchical_v2.rds")
summary(model)
```

### Check Convergence
```r
library(posterior)

# Rhat (< 1.01)
rhats <- rhat(model)
max(rhats, na.rm = TRUE)

# ESS (> 400)
ess_bulk_values <- ess_bulk(model)
min(ess_bulk_values, na.rm = TRUE)

# Divergences (< 10)
sum(nuts_params(model)$Value[nuts_params(model)$Parameter == "divergent__"])
```

### Make Predictions
```r
# New data
new_data <- data.frame(
  log_attempts = log(35),
  is_home = 1,
  is_favored = 1,
  spread_abs = 3.5,
  is_bad_weather = 0,
  is_dome = 1,
  total_close = 48.5,
  years_exp = 8,
  player_id = "00-0033873",  # Mahomes
  team = "KC",
  opponent = "BUF",
  player_season = "00-0033873_2025"
)

# Predict
predictions <- predict(model, newdata = new_data)
yards_pred <- exp(predictions[, "Estimate"])
```

---

## 🛠️ Common Fixes

### Fix: Rhat > 1.01
```r
# Increase iterations + adapt_delta
model <- brm(
  ...,
  iter = 6000,
  warmup = 3000,
  control = list(adapt_delta = 0.99)
)
```

### Fix: ESS < 400
```r
# Increase iterations
model <- brm(
  ...,
  iter = 8000,
  warmup = 4000
)
```

### Fix: Divergent transitions
```r
# Increase adapt_delta
model <- brm(
  ...,
  control = list(adapt_delta = 0.99, max_treedepth = 12)
)
```

### Fix: Memory error
```r
# Reduce data size or thin samples
data_subset <- data %>% filter(season >= 2020)

model <- brm(..., thin = 5)  # Save every 5th sample
```

### Fix: Database connection error
```bash
# Check PostgreSQL is running
docker ps | grep postgres

# Restart if needed
docker compose up -d pg
```

---

## ⚠️ Critical Reminders

### Database Schema
- ❌ **WRONG:** `JOIN games g ON g.game_id = pgs.game_id` (column doesn't exist)
- ✅ **CORRECT:** `JOIN games g ON g.season = pgs.season AND g.week = pgs.week AND (...)`

See: `docs/database/DATABASE_QUICK_REFERENCE.md`

### Training Window
- **State-Space v2:** 2018-2024 + H=4 decay on 2015-2017
- **Hierarchical v2:** Uniform 2015-2025
- **Informative Priors v2:** Uniform 2015-2025

### Convergence Thresholds
- Rhat < 1.01 ✅
- ESS (bulk) > 400 ✅
- ESS (tail) > 400 ✅
- Divergences < 10 ✅
- Max treedepth hit < 5% ✅

### Production Checklist
Before deploying model:
- [ ] Convergence checks passed
- [ ] MAE within expected range
- [ ] Model registered in database
- [ ] Dry-run backtest successful
- [ ] rpy2 integration working

---

## 📚 Key Resources

**Full Documentation:**
- [R_BAYESIAN_MODELING_COOKBOOK.md](R_BAYESIAN_MODELING_COOKBOOK.md) - Complete guide (2,078 lines)
- [R_MODEL_INTERFACE_GUIDE.md](R_MODEL_INTERFACE_GUIDE.md) - Python ↔ R integration
- [DATABASE_QUICK_REFERENCE.md](../database/DATABASE_QUICK_REFERENCE.md) - Schema reference

**External:**
- brms documentation: https://paul-buerkner.github.io/brms/
- cmdstanr documentation: https://mc-stan.org/cmdstanr/
- Stan User's Guide: https://mc-stan.org/docs/stan-users-guide/

**Project-Specific:**
- [CLAUDE.md](../../CLAUDE.md) - Complete project documentation
- [docs/ADVANCED_BAYESIAN_V3.md](../ADVANCED_BAYESIAN_V3.md) - Advanced Bayesian enhancements

---

## 🆘 Troubleshooting

| Issue | Quick Fix | Full Guide |
|-------|-----------|------------|
| Stan won't compile | `cmdstanr::install_cmdstan(overwrite = TRUE)` | Cookbook §6.1 |
| Rhat > 1.01 | Increase `iter` and `adapt_delta` | Cookbook §6.2 |
| ESS < 400 | Increase `iter` to 8000 | Cookbook §6.3 |
| Divergences | Set `adapt_delta = 0.99` | Cookbook §6.4 |
| Memory error | Reduce data size or `thin = 5` | Cookbook §6.5 |
| DB connection | `docker compose up -d pg` | Cookbook §6.6 |
| game_id error | Use `(season, week)` composite key | Cookbook §6.7 |
| rpy2 error | Check R version compatibility | Cookbook §9 |

---

**Version:** 1.0
**Last Updated:** November 16, 2025
**Questions?** See [TROUBLESHOOTING_MASTER.md](TROUBLESHOOTING_MASTER.md)
