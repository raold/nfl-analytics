# Database Optimization Quick Reference

> **Historical (November 2025).** The "critical indexes" below do not exist in today's devdb01
> (checked 2026-10-25), and `docs/DATABASE_OPTIMIZATION_COMPLETE.md` is in tag
> `docs-archive-2026-10`. The read-only SQL files it runs (`db/health_check.sql`,
> `db/monitoring_queries_performance.sql`) are still here. For maintenance now see
> [db/README.md](README.md).

**Date:** November 13, 2025

---

## Daily Operations

### Health Check
```bash
psql -f db/health_check.sql
```

### Performance Monitoring
```bash
psql -f db/monitoring_queries_performance.sql
```

### Refresh Materialized Views (Before Training)
```bash
psql -c "REFRESH MATERIALIZED VIEW CONCURRENTLY mart.player_game_stats;"
psql -c "REFRESH MATERIALIZED VIEW CONCURRENTLY mart.game_summary;"
psql -c "REFRESH MATERIALIZED VIEW CONCURRENTLY mart.player_hierarchy;"
```

---

## Weekly Maintenance

### Vacuum & Analyze
```bash
psql -c "VACUUM ANALYZE games, plays, mart.player_game_stats;"
```

### Check for Bloat
```bash
psql -c "SELECT relname, n_dead_tup, n_live_tup,
    ROUND(100.0 * n_dead_tup / NULLIF(n_live_tup, 0), 2) as bloat_pct
FROM pg_stat_user_tables
WHERE n_dead_tup > 1000 ORDER BY n_dead_tup DESC LIMIT 10;"
```

---

## Troubleshooting

### Check Cache Hit Ratio
```bash
psql -c "SELECT
    ROUND(100.0 * SUM(heap_blks_hit) / NULLIF(SUM(heap_blks_hit) + SUM(heap_blks_read), 0), 2) as cache_hit_pct
FROM pg_statio_user_tables
WHERE schemaname IN ('public', 'mart');"
```
**Target:** >99%

### Find Slow Queries
```bash
psql -c "SELECT query, calls, total_exec_time, mean_exec_time
FROM pg_stat_statements
ORDER BY total_exec_time DESC LIMIT 10;"
```

### Check Index Usage
```bash
psql -c "SELECT schemaname, relname, idx_scan, seq_scan
FROM pg_stat_user_tables
WHERE schemaname IN ('public', 'mart')
ORDER BY seq_scan DESC LIMIT 10;"
```

---

## Critical Indexes

| Index | Table | Purpose |
|-------|-------|---------|
| `idx_player_game_stats_lookup` | mart.player_game_stats | Player stats by season/week |
| `idx_games_season_week_kickoff` | games | Backtest date ranges |
| `idx_plays_passer_epa_yards` | plays | QB performance queries |
| `idx_plays_game_type_epa` | plays | Game-level aggregations |

---

## Emergency Procedures

### Database Not Responding
```bash
# Check connections
psql -c "SELECT state, COUNT(*) FROM pg_stat_activity GROUP BY state;"

# Kill long-running query
psql -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity
WHERE state = 'active' AND query_start < NOW() - INTERVAL '5 minutes';"
```

### Index Rebuild (If Corrupted)
```bash
psql -c "REINDEX INDEX CONCURRENTLY idx_player_game_stats_lookup;"
```

---

## Production Deployment

### 1. Pre-Deployment
```bash
pg_dump devdb01 > backup_$(date +%Y%m%d).sql
psql -f db/health_check.sql > baseline_health.txt
```

### 2. Deploy Indexes (Zero Downtime)
```bash
psql -f db/optimize_indexes.sql
psql -c "ANALYZE;"
```

### 3. Verify
```bash
psql -f db/health_check.sql > post_deploy_health.txt
diff baseline_health.txt post_deploy_health.txt
```

---

## Monitoring Thresholds

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Cache Hit Ratio | <95% | <90% | Increase shared_buffers |
| Dead Tuples | >10% | >20% | Run VACUUM ANALYZE |
| Connections | >80 | >90 | Enable connection pooling |
| Bloat | >20% | >40% | Run VACUUM FULL (off-hours) |
| Query Time | >1s | >5s | Add indexes or optimize |

---

## Contact

For issues, see: `docs/DATABASE_OPTIMIZATION_COMPLETE.md`
