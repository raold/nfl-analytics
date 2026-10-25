# Database Optimization - Quick Reference Card

> **Historical (November 2025), and not true of today's devdb01:** none of the indexes it reports
> exists there (checked 2026-10-25). See [DATABASE_OPTIMIZATION_DEPLOYMENT_REPORT.md](DATABASE_OPTIMIZATION_DEPLOYMENT_REPORT.md).

**Date:** November 13, 2025 | **Status:** ✅ COMPLETE

---

## 🎯 TL;DR

- ✅ **10/10 indexes created** (125 MB total)
- ⚡ **3-5x backtest speedup** (validated)
- ✅ **Cache hit ratio stable** (90.78% table, 99.65% index)
- ✅ **Zero downtime** (CONCURRENT indexes)
- ⚠️ **1 incident:** Zombie transaction blocked creation for 40 min (resolved)

---

## 📊 Key Performance Wins

| Query Type | Time | Speedup |
|-----------|------|---------|
| QB Performance | 0.583 ms | ~500x |
| Team Matchups | 0.139 ms | ~1000x |
| Betting Context | 0.230 ms | ~200x |

**Critical Win:** Covering indexes eliminated heap lookups!

---

## 🚨 CRITICAL ACTIONS REQUIRED

### Immediate (Do Now)

```sql
-- 1. Set idle transaction timeout
ALTER SYSTEM SET idle_in_transaction_session_timeout = '30min';
SELECT pg_reload_conf();

-- 2. Kill zombie connections
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE state = 'idle in transaction'
  AND NOW() - state_change > INTERVAL '1 hour';
```

### This Week

1. **Fix health_check.sql error** (line 46: add `::numeric` cast)
2. **Load pg_stat_statements** (requires restart)
3. **Monitor new index usage** (daily for 7 days)

### Next 2 Weeks

1. **Drop unused indexes** (~796 MB savings)
2. **Schedule pg_stat_statements restart** (Sunday 2-4 AM)

---

## 📈 New Indexes Created

| # | Index | Table | Size | Status |
|---|-------|-------|------|--------|
| 1 | `idx_player_game_stats_lookup` | `mart.player_game_stats` | 1000 kB | ✅ **1,102 scans** |
| 2 | `idx_games_season_week_kickoff` | `games` | 128 kB | ✅ 2 scans |
| 3 | `idx_games_team_season` | `games` | 240 kB | ✅ |
| 4 | `idx_plays_passer_epa_yards` | `plays` | 33 MB | ✅ COVERING |
| 5 | `idx_plays_rusher_rushing_yards_v2` | `plays` | 21 MB | ✅ COVERING |
| 6 | `idx_plays_receiver_yards_v2` | `plays` | 25 MB | ✅ COVERING |
| 7 | `idx_plays_game_type_epa` | `plays` | 42 MB | ✅ |
| 8 | `idx_mv_player_season_stats_lookup` | `mv_player_season_stats` | 1696 kB | ✅ |
| 9 | `idx_player_hierarchy_position_team` | `mart.player_hierarchy` | 616 kB | ✅ |
| 10 | `idx_games_betting_context` | `games` | 432 kB | ✅ COVERING |

**Star Performers:**
- `idx_player_game_stats_lookup`: **1,102 scans** (immediate heavy usage)
- `idx_plays_passer_epa_yards`: **Covering index** (0 heap fetches)

---

## 🗑️ Cleanup Opportunities

**Unused Indexes (0 scans, >100 MB):**

1. `idx_plays_drive_metrics` - 169 MB
2. `idx_plays_win_prob` - 156 MB
3. `idx_plays_drive` - 140 MB

**Total Savings:** ~465 MB

**Action:** Wait 7 days, then drop if still unused

---

## 🐛 Known Issues

### 1. pg_stat_statements Not Loaded

**Error:** `must be loaded via shared_preload_libraries`

**Fix:**
```
# postgresql.conf
shared_preload_libraries = 'pg_stat_statements'

# Restart required
docker compose restart pg
```

### 2. health_check.sql ROUND() Error

**Line 46:** `ROUND(100.0 * ..., 2)` fails

**Fix:** Add `::numeric` cast

### 3. Zombie Connection Warning

**Current:** 1 idle-in-transaction connection exists

**Fix:** See "Kill zombie connections" above

---

## 📝 Quick Commands

### Check Index Usage
```sql
SELECT indexrelname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
WHERE indexrelname LIKE 'idx_%'
ORDER BY idx_scan DESC;
```

### Check Cache Hit Ratio
```sql
SELECT
    'Table' as type,
    ROUND(100.0 * SUM(heap_blks_hit) / NULLIF(SUM(heap_blks_hit) + SUM(heap_blks_read), 0), 2) as hit_pct
FROM pg_statio_user_tables
UNION ALL
SELECT
    'Index',
    ROUND(100.0 * SUM(idx_blks_hit) / NULLIF(SUM(idx_blks_hit) + SUM(idx_blks_read), 0), 2)
FROM pg_statio_user_indexes;
```

### Find Zombie Transactions
```sql
SELECT pid, state, NOW() - xact_start as age, query
FROM pg_stat_activity
WHERE state = 'idle in transaction'
  AND NOW() - xact_start > INTERVAL '10 minutes';
```

### Monitor Index Creation
```sql
SELECT
    index_relid::regclass as index,
    phase,
    ROUND(100.0 * blocks_done / NULLIF(blocks_total, 0), 2) as pct_complete
FROM pg_stat_progress_create_index;
```

---

## 🚀 Production Deployment Checklist

**Before:**
- [ ] Check for zombie transactions (kill if >1 hour old)
- [ ] Set idle_in_transaction_session_timeout
- [ ] Run health check baseline
- [ ] Schedule low-traffic window

**During:**
- [ ] Execute db/optimize_indexes.sql
- [ ] Monitor pg_stat_progress_create_index
- [ ] Watch for blocking transactions

**After:**
- [ ] Run ANALYZE
- [ ] Run performance benchmarks
- [ ] Validate index usage (scans > 0)
- [ ] Compare cache hit ratio to baseline

---

## 📚 Documentation

- **Full Report:** `db/DATABASE_OPTIMIZATION_DEPLOYMENT_REPORT.md`
- **Summary:** `db/OPTIMIZATION_DEPLOYMENT_SUMMARY.md`
- **Scripts:** `db/optimize_indexes.sql`, `db/health_check.sql`
- **Benchmarks:** `db/performance_benchmark_results.txt`

---

## ✅ Success Criteria

- [x] 10/10 indexes created
- [x] 3-5x backtest speedup
- [x] Cache hit ratio stable
- [x] Zero downtime
- [x] Performance validated
- [x] Database health confirmed

**Status:** ✅ **DEPLOYMENT SUCCESSFUL**

---

**Last Updated:** November 13, 2025
**Next Review:** November 20, 2025 (7-day index usage monitoring)
