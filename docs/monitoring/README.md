# Feature Staleness Monitoring System

**Purpose:** Automated monitoring of data freshness for NFL analytics predictions
**Status:** Production Ready
**Created:** December 7, 2025

---

## Overview

The Feature Staleness Monitoring System tracks the freshness of critical data tables and materialized views to ensure predictions are based on current data. It automatically detects:

- Materialized views that need refreshing
- Seasonal data (rosters, playoff context) that's behind current week
- Missing weather data for recent games
- Stale odds captures
- Empty or missing tables

---

## Components

| Component | Purpose | Location |
|-----------|---------|----------|
| Monitor Script | Main staleness checking logic | `scripts/feature_staleness_monitor.py` |
| Metadata Table | Stores staleness history | `metadata.feature_freshness` |
| SQL Migration | Creates metadata table | `sql/migrations/create_feature_freshness_metadata.sql` |
| Cron Job | Daily automated checks | `config/nfl_analytics_crontab.txt` |
| Sample Output | Example reports | `docs/monitoring/FEATURE_STALENESS_SAMPLE_OUTPUT.md` |

---

## Quick Start

### 1. Initialize Metadata Table

```bash
# Option 1: Using Python script
python scripts/feature_staleness_monitor.py --init

# Option 2: Using SQL migration
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
    -f sql/migrations/create_feature_freshness_metadata.sql
```

---

### 2. Run First Check

```bash
python scripts/feature_staleness_monitor.py
```

**Output:**
- Prints staleness report to stdout
- Updates `metadata.feature_freshness` table
- Exits with code: 0 (OK), 1 (STALE), 2 (CRITICAL)

---

### 3. Schedule Daily Monitoring

The cron job is already configured in `config/nfl_analytics_crontab.txt`:

```cron
# Daily 6am ET (11:00 UTC) - Check feature freshness
0 11 * * * cd $NFL_ANALYTICS_DIR && uv run python scripts/feature_staleness_monitor.py >> /tmp/feature_staleness_cron.log 2>&1
```

To install:
```bash
crontab config/nfl_analytics_crontab.txt
```

---

## Monitored Features

| Feature | Check Type | Criticality | Freshness Criteria |
|---------|-----------|-------------|-------------------|
| `mart.team_playoff_context` | Season/Week | Important | Current week during season |
| `public.rosters_weekly` | Season/Week | Critical | Current week during season |
| `mart.player_game_stats` | Materialized View | Critical | Refreshed within 24-48h |
| `mart.game_summary` | Materialized View | Critical | Refreshed within 24-48h |
| `public.weather` | Coverage | Important | 50%+ coverage of recent games |
| `public.odds_history` | Timestamp | Important | Captured within 24h |

---

## Staleness Levels

### OK
- All checks passed
- Data is current
- No action needed

### STALE
- Data is behind but not critical
- Examples:
  - MV refreshed 25 hours ago (>24h threshold)
  - Week behind during season (with 1-week grace period)
  - Odds captured 26 hours ago
- **Action:** Refresh data soon, but predictions still viable

### CRITICAL
- Data is significantly stale
- Examples:
  - MV refreshed 50 hours ago (>48h threshold)
  - Empty table that should have data
  - Season behind current season
- **Action:** IMMEDIATE refresh required before predictions

### UNKNOWN
- Error occurred during check
- Table may not exist or query failed
- **Action:** Investigate error

---

## Usage Examples

### Check All Features (Default)
```bash
python scripts/feature_staleness_monitor.py
```

### View Current Status (No New Checks)
```bash
python scripts/feature_staleness_monitor.py --report
```

### Alert Mode (Only Show Problems)
```bash
python scripts/feature_staleness_monitor.py --alert-only
```

### Verbose Output
```bash
python scripts/feature_staleness_monitor.py --verbose
```

---

## Exit Codes

Use exit codes for automated alerting:

```bash
python scripts/feature_staleness_monitor.py
EXIT_CODE=$?

if [ $EXIT_CODE -eq 2 ]; then
    echo "CRITICAL: Immediate attention required!"
    # Send alert, page on-call, etc.
elif [ $EXIT_CODE -eq 1 ]; then
    echo "WARNING: Features are stale"
    # Log warning, send notification
else
    echo "OK: All features fresh"
fi
```

| Exit Code | Meaning | Action |
|-----------|---------|--------|
| 0 | All features OK | None |
| 1 | Some features STALE | Refresh soon |
| 2 | Critical issues | Immediate action |

---

## Database Schema

### metadata.feature_freshness

```sql
CREATE TABLE metadata.feature_freshness (
    feature_name TEXT PRIMARY KEY,
    last_updated TIMESTAMP WITH TIME ZONE,
    row_count BIGINT,
    max_season INT,
    max_week INT,
    is_stale BOOLEAN,
    staleness_reason TEXT,
    checked_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Indexes:**
- `idx_feature_freshness_checked_at` - For time-series queries
- `idx_feature_freshness_stale` - Quick lookup of stale features

---

## Query Examples

### Find All Stale Features
```sql
SELECT feature_name, staleness_reason, checked_at
FROM metadata.feature_freshness
WHERE is_stale = TRUE
ORDER BY feature_name;
```

### Summary Dashboard
```sql
SELECT
    CASE WHEN is_stale THEN 'STALE' ELSE 'OK' END as status,
    COUNT(*) as count
FROM metadata.feature_freshness
GROUP BY is_stale;
```

### Features Not Checked Recently
```sql
SELECT feature_name, checked_at,
    NOW() - checked_at as time_since_check
FROM metadata.feature_freshness
WHERE checked_at < NOW() - INTERVAL '2 days'
ORDER BY checked_at;
```

---

## Adding New Features to Monitor

Edit `scripts/feature_staleness_monitor.py` in the `run_all_checks()` function:

```python
def run_all_checks(conn, context: dict, save_to_db: bool = True) -> list[FeatureStatus]:
    statuses = []

    # Add your new feature
    statuses.append(
        check_mart_table_freshness(conn, "mart.your_new_table", context, "critical")
    )

    # For custom checks, create a new check function:
    statuses.append(check_your_custom_feature(conn, context))

    # ... rest of checks
    return statuses
```

---

## Adjusting Staleness Thresholds

Current thresholds are defined in the check functions:

### Materialized Views
- **STALE:** 24 hours since refresh
- **CRITICAL:** 48 hours since refresh

Location: `check_materialized_view_freshness()`

### Odds Data
- **STALE:** 24 hours since capture

Location: `check_odds_freshness()`

### Season/Week Data
- **STALE:** 1 week behind (during season)
- **CRITICAL:** 1+ season behind

Location: `check_mart_table_freshness()`

To adjust, modify the relevant check function in `scripts/feature_staleness_monitor.py`.

---

## Integration with Other Systems

### 1. Automated Alerting

```bash
#!/bin/bash
# In monitoring script or cron job

python scripts/feature_staleness_monitor.py --alert-only > /tmp/staleness_check.txt

if [ $? -eq 2 ]; then
    # Send critical alert
    cat /tmp/staleness_check.txt | mail -s "NFL Analytics: CRITICAL Data Staleness" ops@example.com
elif [ $? -eq 1 ]; then
    # Send warning
    cat /tmp/staleness_check.txt | mail -s "NFL Analytics: Data Staleness Warning" ops@example.com
fi
```

---

### 2. Grafana Dashboard (Future)

Query `metadata.feature_freshness` to create dashboard panels:

- Pie chart: OK vs STALE vs CRITICAL
- Table: Current staleness by feature
- Time series: Staleness over time (requires history tracking)

---

### 3. Slack/Discord Notifications (Future)

```python
# Add to main() in feature_staleness_monitor.py

if critical_count > 0:
    send_slack_notification(
        f"🚨 CRITICAL: {critical_count} features stale!",
        statuses
    )
```

---

## Troubleshooting

### Issue: "Table does not exist" for metadata.feature_freshness

**Solution:**
```bash
python scripts/feature_staleness_monitor.py --init
```

---

### Issue: All features show UNKNOWN status

**Possible Causes:**
1. Database connection issue
2. Missing permissions
3. Table schema changes

**Diagnosis:**
```bash
# Test connection
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT 1"

# Check table existence
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "\dt mart.*"

# Run with verbose logging
python scripts/feature_staleness_monitor.py --verbose
```

---

### Issue: False positives (features marked stale when they're current)

**Possible Causes:**
1. NFL season context detection issue (offseason vs in-season)
2. Threshold too aggressive
3. Clock skew between systems

**Solutions:**
1. Check current context: `python -c "from scripts.feature_staleness_monitor import get_current_nfl_context; print(get_current_nfl_context())"`
2. Adjust thresholds in check functions
3. Verify system time: `date`

---

### Issue: Cron job not running

**Diagnosis:**
```bash
# Check if crontab is installed
crontab -l | grep feature_staleness

# Check cron logs
tail -100 /tmp/feature_staleness_cron.log

# Check syslog for cron errors
grep cron /var/log/syslog | tail -20
```

---

## Maintenance

### Weekly Review
- Check for false positives/negatives
- Adjust thresholds if needed
- Add new features as data sources grow

### Monthly Review
- Analyze staleness patterns
- Identify chronic stale features
- Update documentation

### Quarterly Review
- Evaluate effectiveness
- Consider adding history tracking
- Plan integrations (Grafana, alerts)

---

## Future Enhancements

### Planned
1. **History Tracking:** Store check results over time (not just latest)
2. **Trend Analysis:** Detect patterns in staleness
3. **Auto-Remediation:** Trigger refresh scripts automatically
4. **Slack/Discord Alerts:** Real-time notifications
5. **Grafana Dashboard:** Visual monitoring

### Under Consideration
1. **Custom Check Functions:** User-defined staleness criteria
2. **Feature Groups:** Monitor related features together
3. **Dependency Tracking:** Alert if upstream feature is stale
4. **Performance Metrics:** Track check execution time
5. **Anomaly Detection:** ML-based staleness prediction

---

## Related Documentation

- **Sample Output:** [FEATURE_STALENESS_SAMPLE_OUTPUT.md](FEATURE_STALENESS_SAMPLE_OUTPUT.md)
- **Database Health Check:** `scripts/db_health_check.py`
- **Mart Refresh:** `scripts/refresh_marts.py`
- **Cron Jobs:** `config/nfl_analytics_crontab.txt`
- **Database Schema:** `docs/database/DATABASE_QUICK_REFERENCE.md`

---

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review sample output documentation
3. Run with `--verbose` flag for detailed logging
4. Check `/tmp/feature_staleness_cron.log` for cron job issues

---

**Version:** 1.0
**Last Updated:** December 7, 2025
**Maintainer:** Claude Code (Sonnet 4.5)
**License:** Same as NFL Analytics project
