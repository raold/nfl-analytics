# NFL Analytics System Monitoring Dashboard

## Overview

Comprehensive Grafana dashboard for monitoring the NFL Analytics infrastructure stack.

## Files

- `nfl-analytics-system.json` (27KB) - Dashboard configuration
- `dashboards.yaml` (421B) - Provisioning configuration
- `README.md` - This file

## Dashboard Details

**Dashboard Name:** NFL Analytics - System Monitoring
**Auto-refresh:** Every 30 seconds
**Default Time Range:** Last 1 hour
**Total Panels:** 27 panels across 4 rows

## Panel Categories

### Row 1: System Overview (4 panels)
- **CPU Usage** - Overall CPU utilization with thresholds
- **Memory Usage** - System memory consumption
- **Disk I/O** - Read/write operations per second
- **Network Traffic** - RX/TX bytes per second

### Row 2: Database Metrics - PostgreSQL (8 panels)
- **Active Connections** - Current connections vs max
- **Query Duration** - p50, p95, p99 percentiles
- **Transaction Rate** - Commits and rollbacks per second
- **Database Size** - Total size of devdb01
- **Top 5 Table Sizes** - Largest tables by row count
- **Cache Hit Ratio** - PostgreSQL buffer cache efficiency
- **Deadlocks** - Deadlock detection rate
- **Rows Fetched vs Returned** - Query efficiency metrics

### Row 3: Redis Metrics (6 panels)
- **Redis Memory Usage** - Used vs max memory
- **Cache Hit Rate** - Redis cache efficiency
- **Connected Clients** - Number of active connections
- **Total Keys** - Total keys stored
- **Commands Processed** - Operations per second
- **Evicted Keys** - Key eviction rate

### Row 4: Prometheus & Application (9 panels)
- **Scrape Targets Status** - Health of all monitored targets
- **Scrape Duration** - Time to scrape metrics
- **Time Series Count** - TSDB size
- **Prometheus Storage** - Storage blocks size
- **Query Rate** - Prometheus query throughput
- **Streamlit App Status** - Dashboard application health
- **Active Alerts** - Firing alerts count
- **Container Restarts** - Service restart frequency
- **System Uptime** - Time since boot

## Metrics Sources

### Node Exporter (port 9100)
- System metrics: CPU, memory, disk, network
- **Metrics Count:** ~1,672 metrics

### PostgreSQL Exporter (port 9187)
- Database metrics: connections, queries, transactions
- **Metrics Count:** ~8,313 metrics
- **Database:** devdb01

### Redis Exporter (port 9121)
- Cache metrics: memory, hits, keys, commands
- **Note:** Requires redis_exporter service in docker-compose.yml

### Prometheus Self-Monitoring (port 9090)
- Scrape targets, TSDB, query rate, alerts
- **Metrics Count:** ~838 metrics

## Deployment

### Automatic Provisioning

Dashboard is automatically loaded when Grafana starts:

```bash
docker compose up -d grafana
```

Grafana will:
1. Scan `/etc/grafana/provisioning/dashboards/`
2. Load `nfl-analytics-system.json`
3. Create dashboard in "System" folder

### Manual Access

1. Open Grafana: http://localhost:3000
2. Navigate to: **Dashboards** → **System** → **NFL Analytics - System Monitoring**

### Verification

Check all services are up:
```bash
# Prometheus targets
curl http://localhost:9090/targets

# Node Exporter
curl http://localhost:9100/metrics

# PostgreSQL Exporter
curl http://localhost:9187/metrics | grep pg_stat

# Redis Exporter (if configured)
curl http://localhost:9121/metrics
```

## Customization

### Thresholds

Adjust warning/critical thresholds in JSON:

```json
"thresholds": {
  "mode": "absolute",
  "steps": [
    {"value": 0, "color": "green"},
    {"value": 70, "color": "yellow"},
    {"value": 85, "color": "red"}
  ]
}
```

### Time Range

Change default time range:

```json
"time": {
  "from": "now-1h",  // Change this
  "to": "now"
}
```

### Refresh Rate

Modify auto-refresh interval:

```json
"refresh": "30s"  // Options: 10s, 30s, 1m, 5m, 15m, 30m, 1h
```

## Troubleshooting

### No Data in Panels

**Issue:** Panels show "No Data"

**Fix:**
1. Check Prometheus is scraping targets: http://localhost:9090/targets
2. Verify exporters are running: `docker compose ps`
3. Check Prometheus logs: `docker compose logs prometheus`

### PostgreSQL Metrics Missing

**Issue:** Database panels empty

**Fix:**
```bash
# Verify postgres_exporter
docker compose ps postgres_exporter
curl http://localhost:9187/metrics | grep pg_stat_database

# Restart if needed
docker compose restart postgres_exporter
```

### Redis Metrics Missing

**Issue:** Redis panels empty

**Fix:**
```bash
# Check if redis_exporter is configured
docker compose ps redis_exporter

# If not configured, add to docker-compose.yml:
redis_exporter:
  image: oliver006/redis_exporter:latest
  ports:
    - "9121:9121"
  environment:
    REDIS_ADDR: "redis:6379"
```

### Dashboard Not Appearing

**Issue:** Dashboard not in Grafana UI

**Fix:**
```bash
# Check Grafana logs
docker compose logs grafana | grep provisioning

# Verify files exist
ls -l grafana/provisioning/dashboards/

# Restart Grafana
docker compose restart grafana
```

## Key Thresholds Reference

### System
- **CPU:** Green (0-70%), Yellow (70-85%), Red (85-100%)
- **Memory:** Green (0-80%), Yellow (80-90%), Red (90-100%)

### Database
- **Cache Hit Ratio:** Red (0-80%), Yellow (80-90%), Green (90-100%)
- **Deadlocks:** Green (0), Yellow (0.1+/sec), Red (1+/sec)
- **Database Size:** Green (0-10GB), Yellow (10-50GB), Red (50GB+)

### Redis
- **Cache Hit Rate:** Red (0-70%), Yellow (70-85%), Green (85-100%)
- **Memory:** Green (0-800MB), Yellow (800MB-1GB), Red (1GB+)
- **Evictions:** Green (0), Yellow (10+/sec), Red (100+/sec)

### Prometheus
- **Scrape Duration:** Green (0-5s), Yellow (5-10s), Red (10s+)
- **Active Alerts:** Green (0), Yellow (1-4), Red (5+)
- **Container Restarts:** Green (0), Yellow (1-2), Red (3+/hour)

## Performance Considerations

### Dashboard Load Time
- 27 panels × 30s refresh = 0.9 panels/second
- Typical load time: 2-5 seconds
- Prometheus query load: Low (simple aggregations)

### Resource Usage
- Grafana memory: ~100-200MB
- Prometheus storage: ~50-100MB/day
- TSDB retention: Default 15 days

### Optimization Tips
1. Increase refresh interval for less critical panels
2. Use recording rules for expensive queries
3. Adjust time range to reduce data points
4. Enable query caching in Prometheus

## Related Documentation

- Prometheus Configuration (no longer in the repo; see [HISTORY](../../../docs/HISTORY.md))
- [Docker Compose Stack](../../../docker-compose.yml)
- Production Deployment Plan (no longer in the repo; see [HISTORY](../../../docs/HISTORY.md))
- [CLAUDE.md](../../../CLAUDE.md)

## Maintenance

### Regular Tasks
- Review and adjust thresholds monthly
- Archive old Prometheus data (>15 days)
- Update panel queries as schema evolves
- Test dashboard after infrastructure changes

### Version History
- **v1.0** (Nov 8, 2025) - Initial creation with 27 panels

---

**Status:** ✅ Production Ready
**Last Updated:** November 8, 2025
**Maintainer:** Claude Code (Sonnet 4.5)
