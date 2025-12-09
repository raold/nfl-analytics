# Feature Staleness Monitor - Sample Output

**Purpose:** Example output from the feature staleness monitoring system
**Script:** `scripts/feature_staleness_monitor.py`
**Database:** `metadata.feature_freshness`

---

## Sample Report Output

### Scenario 1: All Features Fresh (Exit Code 0)

```
==========================================================================================
FEATURE STALENESS REPORT
==========================================================================================
Current NFL Context: Season 2025, Week 14
In Season: True
Checked At: 2025-12-07 06:00:15
==========================================================================================

Summary: 6 OK, 0 STALE, 0 CRITICAL, 0 UNKNOWN

OK FEATURES:
------------------------------------------------------------------------------------------
  mart.team_playoff_context          | OK       | Data is current
    Rows: 384 | Season: 2025 | Week: 14

  public.rosters_weekly              | OK       | Data is current
    Rows: 53,128 | Season: 2025 | Week: 14

  mart.player_game_stats             | OK       | Last refresh 4.2 hours ago
    Rows: 142,891 | Season: None | Week: None

  mart.game_summary                  | OK       | Last refresh 4.2 hours ago
    Rows: 8,247 | Season: None | Week: None

  public.weather                     | OK       | Weather data coverage: 12/14 recent games
    Rows: 5,891 | Season: None | Week: None

  public.odds_history                | OK       | Last capture 3.8 hours ago
    Rows: 47,293 | Season: None | Week: None

==========================================================================================
```

---

### Scenario 2: Stale Features Detected (Exit Code 1)

```
==========================================================================================
FEATURE STALENESS REPORT
==========================================================================================
Current NFL Context: Season 2025, Week 14
In Season: True
Checked At: 2025-12-07 06:00:15
==========================================================================================

Summary: 3 OK, 3 STALE, 0 CRITICAL, 0 UNKNOWN

STALE FEATURES:
------------------------------------------------------------------------------------------
  mart.player_game_stats             | STALE    | Last refresh 28.3 hours ago (>24h)
    Rows: 142,891 | Season: None | Week: None
    Action: Run 'python scripts/refresh_marts.py'

  public.weather                     | STALE    | Low weather coverage: 7/14 games
    Rows: 5,891 | Season: None | Week: None
    Action: Check weather data ingestion pipeline

  public.odds_history                | STALE    | Last capture 26.1 hours ago
    Rows: 47,293 | Season: None | Week: None
    Action: Run 'python scripts/fetch_odds.py'

OK FEATURES:
------------------------------------------------------------------------------------------
  mart.team_playoff_context          | OK       | Data is current
    Rows: 384 | Season: 2025 | Week: 14

  public.rosters_weekly              | OK       | Data is current
    Rows: 53,128 | Season: 2025 | Week: 14

  mart.game_summary                  | OK       | Last refresh 4.2 hours ago
    Rows: 8,247 | Season: None | Week: None

==========================================================================================
```

---

### Scenario 3: Critical Issues (Exit Code 2)

```
==========================================================================================
FEATURE STALENESS REPORT
==========================================================================================
Current NFL Context: Season 2025, Week 14
In Season: True
Checked At: 2025-12-07 06:00:15
==========================================================================================

Summary: 2 OK, 2 STALE, 2 CRITICAL, 0 UNKNOWN

CRITICAL ISSUES:
------------------------------------------------------------------------------------------
  mart.player_game_stats             | CRITICAL | Last refresh 52.7 hours ago (>48h)
    Rows: 142,891 | Season: None | Week: None
    Action: IMMEDIATE REFRESH REQUIRED

  public.rosters_weekly              | CRITICAL | Week behind: has 12, current is 14
    Rows: 53,128 | Season: 2025 | Week: 12
    Action: Check data ingestion for public.rosters_weekly

STALE FEATURES:
------------------------------------------------------------------------------------------
  mart.game_summary                  | STALE    | Last refresh 30.1 hours ago (>24h)
    Rows: 8,247 | Season: None | Week: None
    Action: Run 'python scripts/refresh_marts.py'

  mart.team_playoff_context          | STALE    | Week behind: has 13, current is 14
    Rows: 384 | Season: 2025 | Week: 13
    Action: Check data ingestion for mart.team_playoff_context

OK FEATURES:
------------------------------------------------------------------------------------------
  public.weather                     | OK       | Weather data coverage: 12/14 recent games
    Rows: 5,891 | Season: None | Week: None

  public.odds_history                | OK       | Last capture 3.8 hours ago
    Rows: 47,293 | Season: None | Week: None

==========================================================================================
```

---

## Usage Examples

### 1. Initialize Metadata Table (First Time Setup)

```bash
python scripts/feature_staleness_monitor.py --init
```

**Output:**
```
2025-12-07 06:00:00 | INFO | Initializing metadata.feature_freshness table...
2025-12-07 06:00:00 | INFO | Metadata table initialized successfully
Metadata table initialized successfully!
```

---

### 2. Run Full Check (Default - Updates Database)

```bash
python scripts/feature_staleness_monitor.py
```

This will:
- Check all features
- Update `metadata.feature_freshness` table
- Print full report
- Exit with appropriate code (0/1/2)

---

### 3. View Current Status (Read-Only)

```bash
python scripts/feature_staleness_monitor.py --report
```

Reads from `metadata.feature_freshness` without running new checks.

---

### 4. Alert Mode (Only Show Problems)

```bash
python scripts/feature_staleness_monitor.py --alert-only
```

Hides OK features, only shows STALE and CRITICAL issues.

---

### 5. Verbose Mode

```bash
python scripts/feature_staleness_monitor.py --verbose
```

Shows detailed logging of each check being performed.

---

## Database Table Schema

```sql
SELECT * FROM metadata.feature_freshness ORDER BY is_stale DESC, feature_name;
```

**Sample Data:**

| feature_name | last_updated | row_count | max_season | max_week | is_stale | staleness_reason | checked_at |
|-------------|--------------|-----------|------------|----------|----------|-----------------|------------|
| mart.player_game_stats | 2025-12-07 02:00:15 | 142891 | NULL | NULL | true | Last refresh 28.3 hours ago (>24h) | 2025-12-07 06:00:15 |
| public.odds_history | 2025-12-06 04:00:23 | 47293 | NULL | NULL | true | Last capture 26.1 hours ago | 2025-12-07 06:00:15 |
| mart.team_playoff_context | 2025-12-06 18:30:12 | 384 | 2025 | 14 | false | Data is current | 2025-12-07 06:00:15 |
| public.rosters_weekly | NULL | 53128 | 2025 | 14 | false | Data is current | 2025-12-07 06:00:15 |
| mart.game_summary | 2025-12-07 02:00:20 | 8247 | NULL | NULL | false | Last refresh 4.2 hours ago | 2025-12-07 06:00:15 |
| public.weather | NULL | 5891 | NULL | NULL | false | Weather data coverage: 12/14 recent games | 2025-12-07 06:00:15 |

---

## Query Examples

### Find All Stale Features

```sql
SELECT feature_name, staleness_reason, checked_at
FROM metadata.feature_freshness
WHERE is_stale = TRUE
ORDER BY feature_name;
```

---

### Feature Freshness History (if tracking over time)

```sql
-- Note: Current implementation overwrites on each check
-- To track history, modify script to INSERT instead of UPSERT
SELECT
    feature_name,
    checked_at,
    is_stale,
    staleness_reason
FROM metadata.feature_freshness
WHERE feature_name = 'mart.player_game_stats'
ORDER BY checked_at DESC
LIMIT 30;
```

---

### Summary Dashboard Query

```sql
SELECT
    CASE
        WHEN is_stale THEN 'STALE'
        ELSE 'OK'
    END as status,
    COUNT(*) as count
FROM metadata.feature_freshness
GROUP BY is_stale
ORDER BY is_stale DESC;
```

---

## Integration with Monitoring Systems

### 1. Alerting with Exit Codes

```bash
# In cron or monitoring script
python scripts/feature_staleness_monitor.py
EXIT_CODE=$?

if [ $EXIT_CODE -eq 2 ]; then
    echo "CRITICAL: Feature staleness detected!" | mail -s "NFL Analytics Alert" admin@example.com
elif [ $EXIT_CODE -eq 1 ]; then
    echo "WARNING: Some features are stale" | mail -s "NFL Analytics Warning" admin@example.com
fi
```

---

### 2. Prometheus Metrics (Future Enhancement)

```python
# Add to script:
from prometheus_client import Gauge, push_to_gateway

staleness_gauge = Gauge('nfl_feature_staleness', 'Feature staleness status', ['feature'])

for status in statuses:
    staleness_gauge.labels(feature=status.feature_name).set(
        1 if status.is_stale else 0
    )
```

---

### 3. Slack/Discord Notifications (Future Enhancement)

```python
import requests

def send_slack_alert(statuses):
    critical = [s for s in statuses if s.staleness_level == StalenessLevel.CRITICAL]
    if critical:
        webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        message = {
            "text": f"🚨 CRITICAL: {len(critical)} features are stale!",
            "attachments": [
                {"text": f"• {s.feature_name}: {s.staleness_reason}"}
                for s in critical
            ]
        }
        requests.post(webhook_url, json=message)
```

---

## Maintenance

### Adding New Features to Monitor

Edit `scripts/feature_staleness_monitor.py` in the `run_all_checks()` function:

```python
def run_all_checks(conn, context: dict, save_to_db: bool = True) -> list[FeatureStatus]:
    statuses = []

    # Add your new feature check
    statuses.append(
        check_mart_table_freshness(conn, "mart.your_new_table", context, "critical")
    )

    # ... rest of checks
    return statuses
```

---

### Adjusting Staleness Thresholds

Modify the check functions to adjust thresholds:

- **Materialized Views:** Currently 24h warning, 48h critical
- **Odds Data:** Currently 24h warning
- **Season/Week Data:** Currently allows 1 week lag during season

---

## Troubleshooting

### Issue: "Table does not exist" errors

**Solution:** Ensure all tables are created. Run database health check:
```bash
python scripts/db_health_check.py
```

---

### Issue: All features show "Unknown" status

**Solution:** Check database connection and permissions:
```bash
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT 1"
```

---

### Issue: Metadata table not found

**Solution:** Initialize the metadata table:
```bash
python scripts/feature_staleness_monitor.py --init
```

Or run SQL migration:
```bash
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
    -f sql/migrations/create_feature_freshness_metadata.sql
```

---

**Last Updated:** December 7, 2025
**Maintainer:** Claude Code
**Related Docs:** `scripts/db_health_check.py`, `scripts/refresh_marts.py`
