# Weekly R Predictions Cron Job Setup Guide

**Created**: November 20, 2025
**Purpose**: Install automated weekly regeneration of R model predictions

---

## Overview

The weekly cron job runs every Tuesday at 3:00 AM (after Monday Night Football) to regenerate R model predictions for offline ensemble inference.

**Script**: `scripts/regenerate_r_predictions_weekly.sh`
**Schedule**: Tuesday 3:00 AM
**Duration**: ~15-20 minutes per full season

---

## Installation

### Step 1: Verify Script Works

Test the script manually before scheduling:

```bash
cd /Users/dro/rice/nfl-analytics

# Test run (dry run with --test flag)
Rscript R/batch_generate_r_predictions.R --test

# Full run for current season
./scripts/regenerate_r_predictions_weekly.sh
```

Check log output in `logs/r_predictions/regeneration_YYYYMMDD_HHMMSS.log`

---

### Step 2: Install Cron Job

Open crontab editor:

```bash
crontab -e
```

Add the following line:

```cron
# Weekly R predictions regeneration (Tuesday 3:00 AM after MNF)
0 3 * * 2 /Users/dro/rice/nfl-analytics/scripts/regenerate_r_predictions_weekly.sh >> /Users/dro/rice/nfl-analytics/logs/r_predictions/cron.log 2>&1
```

**Explanation**:
- `0 3 * * 2`: Run at 3:00 AM on Tuesday (day 2)
- `>>`: Append output to `logs/r_predictions/cron.log`
- `2>&1`: Redirect stderr to stdout

Save and exit the editor.

---

### Step 3: Verify Cron Job Installed

List current cron jobs:

```bash
crontab -l
```

You should see the R predictions regeneration line.

---

## Configuration

Edit `scripts/regenerate_r_predictions_weekly.sh` to customize:

### Update Current Season (Annual Task)

```bash
# Line 19: Update at start of each NFL season
CURRENT_SEASON=$(date +%Y)  # Or hardcode: CURRENT_SEASON=2025
```

### Email Notifications (Optional)

```bash
# Line 36: Set email for notifications
NOTIFY_EMAIL="your-email@example.com"
```

Requires `mail` command installed on macOS.

### Docker Container Names

```bash
# Lines 38-39: If using different container names
FASTAPI_CONTAINER="nfl-analytics-api"
STREAMLIT_CONTAINER="nfl-analytics-dashboard"
```

---

## Monitoring

### Check Logs

**Latest log file**:
```bash
ls -lt logs/r_predictions/ | head -5
tail -100 logs/r_predictions/regeneration_*.log
```

**Cron execution log**:
```bash
tail -50 logs/r_predictions/cron.log
```

### Check Prediction Coverage

Query database to verify predictions were generated:

```sql
-- Latest predictions per model
SELECT
    model_name,
    season,
    n_players,
    n_predictions,
    oldest_prediction,
    newest_prediction
FROM ml.r_prediction_coverage
WHERE season = 2025  -- Current season
ORDER BY model_name;
```

### Check Staleness

```sql
SELECT * FROM ml.check_r_prediction_staleness();
```

If any models/seasons show stale predictions (>7 days old), regeneration may have failed.

---

## Troubleshooting

### Cron Job Not Running

**Check cron is running**:
```bash
ps aux | grep cron
```

**Check cron logs (macOS)**:
```bash
log show --predicate 'process == "cron"' --last 24h
```

**Common issues**:
1. **Script not executable**: Run `chmod +x scripts/regenerate_r_predictions_weekly.sh`
2. **Wrong path**: Use absolute paths in crontab
3. **Environment variables**: Cron runs in limited environment - all paths/vars set in script

### R Script Failures

**Check R is accessible**:
```bash
which Rscript
/usr/local/bin/Rscript  # Should return R installation path
```

**Check R packages installed**:
```bash
Rscript -e "library(brms); library(cmdstanr)"
```

**Check R models exist**:
```bash
ls -lh models/bayesian/*.rds
```

Should show:
- `simplified_random_walk_v2.rds`
- `passing_yards_hierarchical_v2.rds`
- `passing_informative_priors_v2.rds`

### Database Connection Failures

**Test PostgreSQL connection**:
```bash
psql -h localhost -p 5544 -U dro -d devdb01 -c "SELECT 1;"
```

**Check database credentials in script** (lines 23-27):
```bash
export DB_HOST="localhost"
export DB_PORT="5544"
export DB_NAME="devdb01"
export DB_USER="dro"
export DB_PASSWORD="sicillionbillions"
```

### Docker Restart Failures

**Check containers exist**:
```bash
docker ps -a
```

**Manually restart containers**:
```bash
docker restart nfl-analytics-api nfl-analytics-dashboard
```

---

## Manual Execution

To manually regenerate predictions outside of scheduled cron:

**Test mode (100 games)**:
```bash
./scripts/regenerate_r_predictions_weekly.sh
# Or directly:
Rscript R/batch_generate_r_predictions.R --test
```

**Full current season**:
```bash
./scripts/regenerate_r_predictions_weekly.sh
```

**Specific seasons**:
```bash
Rscript R/batch_generate_r_predictions.R --seasons 2022,2023,2024
```

**Single model only**:
```bash
Rscript R/batch_generate_r_predictions.R --seasons 2025 --model state_space_v2
```

---

## Performance Metrics

**Expected performance** (per full season):
- **Runtime**: ~15-20 minutes
- **Predictions generated**: ~1,500-2,000 player-games × 3 models = 4,500-6,000 total
- **Database storage**: ~20-30 MB per season
- **Generation speed**: ~50ms per prediction per model

**Performance indicators**:
- ✅ **Good**: 15-20 min runtime, 4,500+ predictions, avg_generation_time_ms < 100
- ⚠️ **Slow**: >30 min runtime, avg_generation_time_ms > 150
- ❌ **Failed**: <1,000 predictions, errors in log

---

## Cleanup

### Remove Cron Job

```bash
crontab -e
# Delete the R predictions line
```

### Remove Old Logs

Script automatically cleans up logs >30 days old.

Manual cleanup:
```bash
find logs/r_predictions -name "regeneration_*.log" -mtime +30 -delete
```

---

## Next Steps

After cron job is installed and verified:

1. **Monitor first run**: Check logs on first Tuesday 3am execution
2. **Verify predictions**: Query `ml.r_prediction_coverage` after first run
3. **Integrate with ensemble**: Update `enhanced_ensemble_v6_production.py` to use lookup client
4. **Test end-to-end**: Run backtest using offline predictions

---

**Status**: ✅ Phase 4 Complete (Cron Job Setup)
**Next**: Phase 5 - 2022-2024 Backfill & A/B Testing
