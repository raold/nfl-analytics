# Feature Staleness Monitor - Quick Reference Card

**Script:** `scripts/feature_staleness_monitor.py`
**Database:** `metadata.feature_freshness`
**Cron:** Daily 6am ET

---

## Commands

```bash
# Initialize (first time)
python scripts/feature_staleness_monitor.py --init

# Run check
python scripts/feature_staleness_monitor.py

# View status
python scripts/feature_staleness_monitor.py --report

# Alerts only
python scripts/feature_staleness_monitor.py --alert-only

# Verbose
python scripts/feature_staleness_monitor.py --verbose
```

---

## Exit Codes

| Code | Status | Action |
|------|--------|--------|
| 0 | OK | None |
| 1 | STALE | Refresh soon |
| 2 | CRITICAL | Immediate |

---

## Monitored Features

| Feature | Check |
|---------|-------|
| `mart.team_playoff_context` | Season/Week |
| `public.rosters_weekly` | Season/Week |
| `mart.player_game_stats` | MV Refresh |
| `mart.game_summary` | MV Refresh |
| `public.weather` | Coverage |
| `public.odds_history` | Timestamp |

---

## Staleness Thresholds

- **MV:** 24h STALE, 48h CRITICAL
- **Odds:** 24h STALE
- **Season/Week:** 1 week grace period

---

## Quick Queries

```sql
-- Find stale features
SELECT feature_name, staleness_reason
FROM metadata.feature_freshness
WHERE is_stale = TRUE;

-- Summary
SELECT
    CASE WHEN is_stale THEN 'STALE' ELSE 'OK' END,
    COUNT(*)
FROM metadata.feature_freshness
GROUP BY is_stale;
```

---

## Troubleshooting

```bash
# Check logs
tail -f /tmp/feature_staleness_cron.log

# Check DB
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
    -c "SELECT * FROM metadata.feature_freshness;"

# Reinitialize
python scripts/feature_staleness_monitor.py --init
```

---

## Files

- **Script:** `scripts/feature_staleness_monitor.py`
- **SQL:** `sql/migrations/create_feature_freshness_metadata.sql`
- **Cron:** `config/nfl_analytics_crontab.txt` (line 69)
- **Docs:** `docs/monitoring/`

---

**Version:** 1.0 | **Created:** 2025-12-07
