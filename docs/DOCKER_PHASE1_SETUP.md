# Docker Phase 1 Setup Guide

**Date:** November 8, 2025
**Version:** 1.0
**Status:** Production Ready
**Purpose:** Comprehensive Docker deployment documentation for Phase 1 infrastructure

---

## Table of Contents

1. [Overview](#overview)
2. [Phase 1 Architecture](#phase-1-architecture)
3. [Services Breakdown](#services-breakdown)
4. [Quick Start Guide](#quick-start-guide)
5. [Service URLs & Ports](#service-urls--ports)
6. [Health Check Commands](#health-check-commands)
7. [Viewing Logs](#viewing-logs)
8. [Troubleshooting](#troubleshooting)
9. [Redis Cache Verification](#redis-cache-verification)
10. [API Testing Examples](#api-testing-examples)
11. [Monitoring Dashboard Access](#monitoring-dashboard-access)
12. [Stop, Restart & Cleanup](#stop-restart--cleanup)
13. [Volume Backup Instructions](#volume-backup-instructions)
14. [Phase 2 Preview](#phase-2-preview)

---

## Overview

Phase 1 establishes the foundational Docker infrastructure for the NFL Analytics platform, including:

- **PostgreSQL 16 TimescaleDB** database layer (time-series optimized)
- **Streamlit Dashboard** for interactive analysis and monitoring
- **Prometheus & Grafana** for comprehensive system monitoring
- **MCP Server** (optional) for AI-powered analytics integration
- **PostgreSQL Exporter** for database metrics
- **Node Exporter** for system-level metrics
- **Alertmanager** for alert notifications (optional)

**Key Benefits:**
- Reproducible, containerized environment across all machines
- Isolated service dependencies
- Health checks and automatic recovery
- Comprehensive monitoring and alerting
- Seamless database persistence
- Production-ready security and networking

---

## Phase 1 Architecture

### Infrastructure Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    NFL Analytics Phase 1                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Monitoring & Analysis Layer                │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ Dashboard (Streamlit) │ Prometheus │ Grafana │ Alertmgr │   │
│  │ :8501                │ :9090      │ :3000   │ :9093    │   │
│  └────────────────────────────┬────────────────────────────┘   │
│                                │                                 │
│  ┌────────────────────────────┴────────────────────────────┐   │
│  │         Metrics Exporters (Prometheus Targets)         │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ PostgreSQL Exporter │ Node Exporter | MCP Server       │   │
│  │ :9187              │ :9100         | :3000 (opt)       │   │
│  └────────────────────────────┬────────────────────────────┘   │
│                                │                                 │
│  ┌────────────────────────────┴────────────────────────────┐   │
│  │          Data & Processing Layer                        │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │   PostgreSQL 16 + TimescaleDB                           │   │
│  │   Host: localhost | Port: 5544 | DB: devdb01           │   │
│  │   Persistence: ./pgdata volume                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Docker Network: nfl-analytics              │   │
│  │   (External network - must be created manually)         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Service Dependencies

```
docker-compose up -d

  pg-1 (PostgreSQL + TimescaleDB)
    │
    ├─→ postgres_exporter (metrics collection)
    │
    ├─→ dashboard (Streamlit + Prometheus + Grafana)
    │   └─→ Depends on: pg-1 healthy
    │
    ├─→ mcp-server (optional, requires --profile mcp)
    │   └─→ Depends on: pg-1 healthy
    │
    ├─→ alertmanager (optional, requires --profile alerts)
    │
    └─→ node_exporter (system metrics)
```

---

## Services Breakdown

### 1. PostgreSQL 16 + TimescaleDB (`pg-1`)

**Purpose:** Primary data warehouse for all NFL analytics data
**Image:** `timescale/timescaledb:latest-pg16`
**Container Name:** `nfl-analytics-pg-1`

**Key Configuration:**
- **Port:** 5544 → 5432 (mapped to host)
- **Database:** `devdb01`
- **User:** `dro`
- **Password:** `sicillionbillions` (dev only)
- **Volume:** `./pgdata` (persistent storage)
- **Health Check:** pg_isready (10s interval, 5 retries)

**Connection String (external):**
```
postgresql://dro:sicillionbillions@localhost:5544/devdb01
```

**Connection String (internal, from other containers):**
```
postgresql://dro:sicillionbillions@nfl-analytics-pg-1:5432/devdb01
```

**Key Capabilities:**
- TimescaleDB extension for time-series data (hypertables)
- 6,991 games (1999-2024)
- 1,230,857 play-by-play records
- 820,080 odds history records
- 56 tables across 5 schemas (public, mart, ml, staging, logs)

---

### 2. Streamlit Dashboard (`dashboard`)

**Purpose:** Interactive web dashboard for monitoring, analysis, and predictions
**Build Context:** `./dashboard`
**Container Name:** `dashboard`
**Restart Policy:** unless-stopped

**Ports:**
- **8501:** Streamlit web UI
- **9090:** Prometheus (metrics server)
- **3000:** Grafana (visualization dashboards)

**Environment Variables:**
```
DB_HOST=nfl-analytics-pg-1
DB_PORT=5432
DB_NAME=devdb01
DB_USER=dro
DB_PASSWORD=sicillionbillions
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=sicillionbillions (override with GRAFANA_ADMIN_PASSWORD env var)
GF_USERS_ALLOW_SIGN_UP=false
GF_SERVER_ROOT_URL=http://localhost:3000
```

**Volumes:**
- `./prometheus:/etc/prometheus:ro` - Prometheus configuration
- `./grafana:/etc/grafana:ro` - Grafana configuration
- `prometheus_data:/var/lib/prometheus` - Prometheus time-series data
- `grafana_data:/var/lib/grafana` - Grafana dashboards and settings

**Health Check:**
- Streamlit endpoint: `http://localhost:8501/_stcore/health`
- Prometheus endpoint: `http://localhost:9090/-/healthy`
- Grafana endpoint: `http://localhost:3000/api/health`
- Interval: 30s | Timeout: 10s | Retries: 3 | Start delay: 20s

**Integrated Services:**
- **Prometheus:** Metrics collection and time-series storage
- **Grafana:** Visualization and dashboard creation
- Both run via supervisor inside the container

---

### 3. MCP Server (`mcp-server`) - OPTIONAL

**Purpose:** Model Context Protocol server for AI-powered analytics
**Build Context:** Root directory (`.`)
**Dockerfile:** `mcp_server/Dockerfile`
**Container Name:** `nfl-analytics-mcp`
**Restart Policy:** "no" (manual restart)
**Profile:** `mcp` (only started with `--profile mcp`)

**Port:** 3000 (internal, not exposed to host by default)

**Environment Variables:**
```
DATABASE_URL=postgresql://dro:sicillionbillions@nfl-analytics-pg-1:5432/devdb01
MCP_LOG_LEVEL=INFO
MCP_SERVER_PORT=3000
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}  (from .env)
APIFY_API_KEY=${APIFY_API_KEY:-}          (from .env)
```

**Volumes:**
- `./logs/mcp:/app/logs/mcp` - MCP server logs
- `./cache/mcp:/app/cache/mcp` - MCP cache directory

**Dependencies:** pg-1 (healthy)

---

### 4. PostgreSQL Exporter (`postgres_exporter`)

**Purpose:** Metrics collection from PostgreSQL for Prometheus
**Image:** `prometheuscommunity/postgres-exporter:latest`
**Container Name:** `nfl-analytics-postgres-exporter`

**Port:** 9187 (Prometheus metrics endpoint)

**Metrics Collected:**
- Database size and statistics
- Table/index sizes
- Query performance
- Connection counts
- Lock information
- Replication status

**Health:** Depends on pg-1 (healthy), auto-restart on failure

---

### 5. Node Exporter (`node_exporter`)

**Purpose:** System-level metrics (CPU, memory, disk, network)
**Image:** `prom/node-exporter:latest`
**Container Name:** `nfl-analytics-node-exporter`

**Port:** 9100 (Prometheus metrics endpoint)

**Metrics Collected:**
- CPU usage and load average
- Memory utilization
- Disk I/O and capacity
- Network interface statistics
- System uptime
- Process counts

**Mount Points (read-only):**
- `/proc:/host/proc:ro` - Process information
- `/sys:/host/sys:ro` - System information
- `/:/rootfs:ro` - Root filesystem

---

### 6. Alertmanager (`alertmanager`) - OPTIONAL

**Purpose:** Alert notification routing and grouping
**Image:** `prom/alertmanager:latest`
**Container Name:** `nfl-analytics-alertmanager`
**Port:** 9093
**Profile:** `alerts` (only started with `--profile alerts`)

**Configuration:**
- Config file: `./prometheus/alertmanager.yml`
- Storage: `alertmanager_data` volume
- Restart policy: unless-stopped

**Features:**
- Groups related alerts
- Routes alerts to different destinations
- Manages alert silencing
- Deduplicates duplicate alerts

---

## Quick Start Guide

### Prerequisites

- Docker and Docker Compose installed
- At least 4 GB RAM available
- 2 GB disk space for pgdata volume
- Network port availability: 5544, 8501, 9090, 3000, 9187, 9100, 9093
- `.env` file (optional, for Grafana password override)

### Step 1: Create Docker Network

The docker-compose.yml uses an **external network**. Create it first:

```bash
docker network create nfl-analytics
```

**Verify:**
```bash
docker network ls | grep nfl-analytics
```

### Step 2: Start Phase 1 Infrastructure

**Option A: Start all services (except optional profiles)**
```bash
cd \
docker compose up -d
```

**Option B: Start including MCP server**
```bash
docker compose --profile mcp up -d
```

**Option C: Start including Alertmanager**
```bash
docker compose --profile alerts up -d
```

**Option D: Start everything**
```bash
docker compose --profile mcp --profile alerts up -d
```

### Step 3: Verify All Services are Running

```bash
docker compose ps
```

Expected output:
```
CONTAINER ID   IMAGE                                        STATUS              PORTS
...
nfl-analytics-pg-1                  timescale/timescaledb:latest-pg16           Up (healthy)  0.0.0.0:5544->5432/tcp
dashboard                           nfl-analytics-dashboard:latest              Up (healthy)  0.0.0.0:8501->8501/tcp, 0.0.0.0:9090->9090/tcp, 0.0.0.0:3000->3000/tcp
nfl-analytics-postgres-exporter     prometheuscommunity/postgres-exporter:latest Up           0.0.0.0:9187->9187/tcp
nfl-analytics-node-exporter         prom/node-exporter:latest                   Up           0.0.0.0:9100->9100/tcp
nfl-analytics-mcp                   nfl-analytics-mcp:latest                     Up           (optional)
nfl-analytics-alertmanager          prom/alertmanager:latest                     Up           0.0.0.0:9093->9093/tcp (optional)
```

### Step 4: Initialize Database (First Time Only)

Load the schema and seed data:

```bash
# Apply schema migrations
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -f db/001_init.sql

# Apply TimescaleDB extensions
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -f db/002_timescale.sql

# Apply player hierarchy schema
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -f db/023_player_hierarchy_schema.sql

# Run all migrations in order
for migration in db/migrations/*.sql; do
  echo "Applying: $migration"
  psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -f "$migration"
done
```

### Step 5: Verify Database Connection

```bash
# From host
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -c "SELECT version();"

# From Docker
docker compose exec pg-1 psql -U dro -d devdb01 -c "SELECT version();"
```

### Step 6: Access Services

See [Service URLs & Ports](#service-urls--ports) section below

---

## Service URLs & Ports

### Direct Access (From Host Machine)

| Service | URL | Port | Credentials |
|---------|-----|------|-------------|
| **Streamlit Dashboard** | http://localhost:8501 | 8501 | Public |
| **Grafana Dashboards** | http://localhost:3000 | 3000 | admin / sicillionbillions |
| **Prometheus Metrics** | http://localhost:9090 | 9090 | Public |
| **PostgreSQL** | localhost:5544 | 5544 | dro / sicillionbillions |
| **PostgreSQL Exporter** | http://localhost:9187/metrics | 9187 | Public |
| **Node Exporter** | http://localhost:9100/metrics | 9100 | Public |
| **Alertmanager** | http://localhost:9093 | 9093 | Public (if profile enabled) |

### Internal Container Network (nfl-analytics)

| Service | Host | Port |
|---------|------|------|
| **PostgreSQL** | nfl-analytics-pg-1 | 5432 |
| **Streamlit** | dashboard | 8501 |
| **Prometheus** | dashboard | 9090 |
| **Grafana** | dashboard | 3000 |

### Quick Access Links

```bash
# Open Streamlit Dashboard
open http://localhost:8501

# Open Grafana
open http://localhost:3000

# Open Prometheus
open http://localhost:9090

# Test PostgreSQL connection
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT COUNT(*) FROM games;"
```

---

## Health Check Commands

### Check Individual Service Health

**PostgreSQL:**
```bash
# Using psql
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT 1;"

# Using Docker
docker compose exec pg-1 pg_isready -U dro -d devdb01

# Expected: "accepting connections"
```

**Streamlit:**
```bash
curl -s http://localhost:8501/_stcore/health | jq .
```

**Prometheus:**
```bash
curl -s http://localhost:9090/-/healthy
```

**Grafana:**
```bash
curl -s http://localhost:3000/api/health | jq .
```

**PostgreSQL Exporter:**
```bash
curl -s http://localhost:9187/metrics | head -20
```

**Node Exporter:**
```bash
curl -s http://localhost:9100/metrics | head -20
```

### Health Summary Script

Create `scripts/dev/health_check.sh`:

```bash
#!/bin/bash

echo "=== NFL Analytics Phase 1 Health Check ==="
echo ""

echo "1. PostgreSQL..."
docker compose exec pg-1 pg_isready -U dro -d devdb01 && echo "   ✓ Healthy" || echo "   ✗ Unhealthy"

echo ""
echo "2. Streamlit Dashboard..."
curl -s -f http://localhost:8501/_stcore/health > /dev/null && echo "   ✓ Healthy" || echo "   ✗ Unhealthy"

echo ""
echo "3. Prometheus..."
curl -s -f http://localhost:9090/-/healthy > /dev/null && echo "   ✓ Healthy" || echo "   ✗ Unhealthy"

echo ""
echo "4. Grafana..."
curl -s -f http://localhost:3000/api/health > /dev/null && echo "   ✓ Healthy" || echo "   ✗ Unhealthy"

echo ""
echo "5. PostgreSQL Exporter..."
curl -s -f http://localhost:9187/metrics > /dev/null && echo "   ✓ Healthy" || echo "   ✗ Unhealthy"

echo ""
echo "6. Node Exporter..."
curl -s -f http://localhost:9100/metrics > /dev/null && echo "   ✓ Healthy" || echo "   ✗ Unhealthy"

echo ""
echo "=== Service Status ==="
docker compose ps
```

Run with:
```bash
bash scripts/dev/health_check.sh
```

---

## Viewing Logs

### Docker Compose Logs

**All services:**
```bash
docker compose logs -f
```

**Specific service:**
```bash
docker compose logs -f pg-1          # PostgreSQL
docker compose logs -f dashboard     # Dashboard (Streamlit, Prometheus, Grafana)
docker compose logs -f postgres_exporter
docker compose logs -f node_exporter
```

**Last N lines:**
```bash
docker compose logs --tail 50 pg-1
```

**Timestamps included:**
```bash
docker compose logs -f --timestamps pg-1
```

### View Raw Logs Inside Containers

**PostgreSQL logs:**
```bash
docker compose exec pg-1 tail -f /var/lib/postgresql/data/log/postgresql.log
```

**MCP Server logs:**
```bash
docker compose exec mcp-server tail -f /app/logs/mcp/server.log
```

### Export Logs for Analysis

```bash
# Export all logs
docker compose logs > logs/all_services_$(date +%Y%m%d_%H%M%S).log

# Export specific service
docker compose logs pg-1 > logs/postgres_$(date +%Y%m%d_%H%M%S).log
```

---

## Troubleshooting

### Issue 1: Docker Network Not Found

**Error:**
```
Error response from daemon: network nfl-analytics not found
```

**Solution:**
```bash
docker network create nfl-analytics
docker compose up -d
```

### Issue 2: PostgreSQL Health Check Failing

**Error:**
```
pg-1 is not healthy
```

**Solution:**
```bash
# Check logs
docker compose logs pg-1

# Restart container
docker compose restart pg-1

# Check if port is in use
lsof -i :5544

# If port in use by another process, kill it or use different port
```

### Issue 3: Connection Refused (5544)

**Error:**
```
psql: error: could not connect to server: Connection refused
```

**Possible Causes:**
1. PostgreSQL container not running
2. Port 5544 already in use
3. Network not created

**Solution:**
```bash
# Verify container is running
docker compose ps pg-1

# Verify port is mapped
docker compose port pg-1 5432

# Check for port conflicts
netstat -tulpn | grep 5544

# Restart
docker compose down
docker compose up -d
```

### Issue 4: Dashboard Not Loading

**Error:**
```
Connection refused on 8501
```

**Solution:**
```bash
# Check dashboard health
curl http://localhost:8501/_stcore/health

# Check logs
docker compose logs dashboard

# Restart
docker compose restart dashboard

# Verify it depends on pg-1
docker compose logs dashboard | grep "depends_on"
```

### Issue 5: Prometheus Data Not Collecting

**Check data collection:**
```bash
# Query Prometheus
curl 'http://localhost:9090/api/v1/query?query=up'

# Should return targets for:
# - PostgreSQL exporter (postgres_exporter:9187)
# - Node exporter (node_exporter:9100)
```

**If targets show "down":**
```bash
# Check network connectivity
docker compose exec dashboard ping nfl-analytics-postgres-exporter
docker compose exec dashboard ping nfl-analytics-node-exporter

# Check exporter ports
docker compose logs postgres_exporter | grep "listening"
docker compose logs node_exporter | grep "listening"
```

### Issue 6: Disk Space Issues

**Check pgdata volume size:**
```bash
du -sh pgdata/
```

**If pgdata too large:**
```bash
# Vacuum database
docker compose exec pg-1 vacuumdb -U dro -d devdb01 -a

# Clean temporary data
docker compose exec pg-1 \
  psql -U dro -d devdb01 -c "SELECT pg_database_size('devdb01') / 1024 / 1024 / 1024 as size_gb;"
```

**Reset if necessary (WARNING: LOSES DATA):**
```bash
# Backup first!
docker compose exec pg-1 pg_dump -U dro -d devdb01 > backup.sql

# Remove volume
docker compose down -v

# Remove pgdata directory
rm -rf pgdata/

# Restart
docker compose up -d
```

---

## Redis Cache Verification

**Status:** Redis is not included in Phase 1 (reserved for Phase 2)

**Planned for Phase 2:**
- In-memory caching for frequently accessed features
- Session management
- Real-time cache invalidation
- Performance metrics caching

**Future Usage:**
```yaml
redis:
  image: redis:latest
  ports:
    - "6379:6379"
  volumes:
    - redis_data:/data
  restart: unless-stopped
```

---

## API Testing Examples

### Test Database Connection

```bash
# Basic connectivity
curl -X POST postgresql://dro:sicillionbillions@localhost:5544/devdb01 \
  -c "SELECT COUNT(*) as game_count FROM games;"

# Expected output: game_count = 6991

# Using psql
psql postgresql://dro:sicillionbillions@localhost:5544/devdb01 -c "SELECT COUNT(*) FROM games;"
```

### Test Prometheus Metrics Collection

**Query active targets:**
```bash
curl -s 'http://localhost:9090/api/v1/targets' | jq '.data.activeTargets[].labels.job'
```

Expected:
```
prometheus
node
postgres
```

**Query specific metric:**
```bash
curl -s 'http://localhost:9090/api/v1/query?query=pg_database_size_bytes' | jq '.data.result[0]'
```

**Query PostgreSQL connection count:**
```bash
curl -s 'http://localhost:9090/api/v1/query?query=pg_stat_activity_count' | jq '.data.result[]'
```

### Test Streamlit Dashboard

**Check health:**
```bash
curl -v http://localhost:8501/_stcore/health
```

Expected status: 200 OK

### Test Grafana API

**Get dashboard list:**
```bash
curl -s -u admin:sicillionbillions http://localhost:3000/api/search | jq '.[] | {title, type, id}'
```

**Get health status:**
```bash
curl -s http://localhost:3000/api/health | jq .
```

### Test MCP Server (Optional)

```bash
# Check if running (with --profile mcp)
docker compose --profile mcp logs mcp-server | grep "listening"

# Test endpoint
curl http://localhost:3000/health
```

---

## Monitoring Dashboard Access

### Grafana Dashboards

**URL:** http://localhost:3000
**Default Credentials:** admin / sicillionbillions

**Key Dashboards:**
1. **System Overview** - CPU, Memory, Disk, Network
2. **PostgreSQL Metrics** - Database size, query count, connections
3. **Application Metrics** - Streamlit activity, request latency
4. **Data Quality** - Games, plays, odds history counts
5. **Performance** - Query duration, transaction counts

**Create New Dashboard:**
1. Navigate to http://localhost:3000
2. Click "Create" → "Dashboard"
3. Click "Add Panel"
4. Select "Prometheus" as data source
5. Enter PromQL query (e.g., `pg_database_size_bytes`)

**Example PromQL Queries:**

```promql
# PostgreSQL database size
pg_database_size_bytes{datname="devdb01"}

# PostgreSQL connections
pg_stat_activity_count{state="active"}

# System CPU usage
100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# System memory usage
(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100

# Disk usage
(1 - (node_filesystem_avail_bytes / node_filesystem_size_bytes)) * 100
```

### Prometheus Queries

**URL:** http://localhost:9090

**Graph View:**
1. Click "Graph" tab
2. Enter PromQL query
3. Click "Execute"

**Common Queries:**

```promql
# Service availability
up

# PostgreSQL version
pg_version_info

# Database size in bytes
pg_database_size_bytes

# Number of relations (tables)
pg_relations_info

# Index size
pg_indexes_size_bytes

# Table bloat
pg_table_bloat_ratio

# Slow queries (if enabled)
rate(pg_statement_time_seconds[5m])
```

### Alert Monitoring (If Profile Enabled)

**Alertmanager URL:** http://localhost:9093

**Features:**
- Active alerts
- Alert groups and silences
- Receiver configuration
- Integration with notification channels (Slack, Email, PagerDuty)

---

## Stop, Restart & Cleanup

### Stop All Services

```bash
docker compose stop
```

Containers remain in memory, can be quickly restarted.

### Stop Specific Service

```bash
docker compose stop pg-1
docker compose stop dashboard
docker compose stop postgres_exporter
```

### Start Services

```bash
# Start all (picks up where they left off)
docker compose start

# Start specific service
docker compose start pg-1
```

### Restart Services

```bash
# Restart all
docker compose restart

# Restart specific
docker compose restart pg-1
docker compose restart dashboard

# Restart with full rebuild
docker compose up -d --force-recreate
```

### Graceful Shutdown

```bash
# Stop and remove containers (preserves volumes and data)
docker compose down

# Verify
docker compose ps  # Should be empty

# Restart
docker compose up -d
```

### Full Cleanup (WARNING: DELETES DATA)

```bash
# Stop and remove everything (containers, networks, unnamed volumes)
docker compose down -v

# Remove pgdata directory (DELETES DATABASE!)
rm -rf pgdata/

# Verify cleanup
docker compose ps
docker volume ls | grep nfl-analytics
docker network ls | grep nfl-analytics
```

### Prune Unused Docker Resources

```bash
# Remove dangling images, containers, networks (safe)
docker system prune

# Include unused volumes (WARNING: may delete needed data)
docker system prune -a --volumes
```

### Container Recycling (Fresh Restart)

```bash
# Stop all
docker compose down

# Rebuild images
docker compose build --no-cache

# Start fresh
docker compose up -d
```

---

## Volume Backup Instructions

### Backup Database (pg_dump)

**Method 1: From Docker**
```bash
docker compose exec pg-1 pg_dump -U dro -d devdb01 > backup_$(date +%Y%m%d_%H%M%S).sql

# Verify
ls -lh backup_*.sql
```

**Method 2: From Host (via Docker)**
```bash
docker compose exec -T pg-1 pg_dump -U dro -d devdb01 \
  | gzip > backups/devdb01_$(date +%Y%m%d_%H%M%S).sql.gz

# Verify
gunzip -t backups/devdb01_*.sql.gz
```

**Method 3: Full pgdata Volume**
```bash
# Create tar archive (container must be running)
tar -czf pgdata_backup_$(date +%Y%m%d_%H%M%S).tar.gz pgdata/

# Verify
tar -tzf pgdata_backup_*.tar.gz | head -20
```

### Restore Database

**From SQL dump:**
```bash
# Stop connections
docker compose exec pg-1 psql -U dro -d devdb01 \
  -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'devdb01' AND pid != pg_backend_pid();"

# Drop and recreate
docker compose exec pg-1 psql -U dro -c "DROP DATABASE devdb01;"
docker compose exec pg-1 psql -U dro -c "CREATE DATABASE devdb01;"

# Restore
docker compose exec -T pg-1 psql -U dro -d devdb01 < backup_20251108.sql
```

**From tar archive:**
```bash
# Stop container
docker compose stop pg-1

# Remove old pgdata
rm -rf pgdata/

# Extract backup
tar -xzf pgdata_backup_20251108.tar.gz

# Start container
docker compose start pg-1
```

### Backup Volumes

**Backup Prometheus data:**
```bash
docker run --rm \
  -v nfl-analytics_prometheus_data:/data \
  -v $(pwd)/backups:/backup \
  ubuntu tar czf /backup/prometheus_$(date +%Y%m%d).tar.gz -C /data .
```

**Backup Grafana data:**
```bash
docker run --rm \
  -v nfl-analytics_grafana_data:/data \
  -v $(pwd)/backups:/backup \
  ubuntu tar czf /backup/grafana_$(date +%Y%m%d).tar.gz -C /data .
```

### Automated Backup Script

Create `scripts/dev/backup_phase1.sh`:

```bash
#!/bin/bash

BACKUP_DIR="backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

echo "Backing up PostgreSQL..."
docker compose exec -T pg-1 pg_dump -U dro -d devdb01 \
  | gzip > $BACKUP_DIR/devdb01_$TIMESTAMP.sql.gz

echo "Backing up Prometheus data..."
docker run --rm \
  -v nfl-analytics_prometheus_data:/data \
  -v $(pwd)/$BACKUP_DIR:/backup \
  ubuntu tar czf /backup/prometheus_$TIMESTAMP.tar.gz -C /data . 2>/dev/null

echo "Backing up Grafana data..."
docker run --rm \
  -v nfl-analytics_grafana_data:/data \
  -v $(pwd)/$BACKUP_DIR:/backup \
  ubuntu tar czf /backup/grafana_$TIMESTAMP.tar.gz -C /data . 2>/dev/null

echo "Backup complete!"
ls -lh $BACKUP_DIR/ | tail -5
```

Run with:
```bash
bash scripts/dev/backup_phase1.sh
```

---

## Phase 2 Preview

**Planned for Phase 2 (December 2025):**

### 1. Redis Caching Layer
```yaml
redis:
  image: redis:7-alpine
  ports:
    - "6379:6379"
  volumes:
    - redis_data:/data
  restart: unless-stopped
```
- Feature caching (50% latency reduction)
- Session management
- Real-time cache invalidation
- Performance metrics caching

### 2. Model Training Orchestration
- GPU training service (NVIDIA CUDA)
- Distributed training coordinator
- Model versioning and registry
- Hyperparameter optimization service

### 3. API Gateway
- FastAPI/Flask wrapper around models
- Request validation and sanitization
- Rate limiting
- Authentication (JWT)
- Versioning (v1, v2, v3 endpoints)

### 4. Message Queue
- Celery with RabbitMQ or Redis backend
- Asynchronous job processing
- Model inference queue
- Backtesting pipeline

### 5. Real-time Prediction Service
- WebSocket support for live odds
- Streaming predictions
- Confidence intervals
- Position sizing recommendations

### 6. Enhanced Monitoring
- Custom metrics for model performance
- Data quality dashboards
- Prediction confidence tracking
- Profit/loss monitoring

### Architecture Diagram (Phase 2)

```
┌─────────────────────────────────────────────────────────────┐
│                 NFL Analytics Phase 1 + 2                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │        API Gateway (FastAPI)                       │    │
│  │        - Request validation                        │    │
│  │        - Authentication (JWT)                      │    │
│  │        - Rate limiting                             │    │
│  │        - Versioning (v1, v2, v3)                   │    │
│  └────────────────────────────────────────────────────┘    │
│           │                    │                  │          │
│           ▼                    ▼                  ▼          │
│  ┌──────────────┐    ┌──────────────┐   ┌────────────────┐ │
│  │ Prediction   │    │ Backtesting  │   │ Feature        │ │
│  │ Service      │    │ Pipeline     │   │ Engineering    │ │
│  │ (Real-time)  │    │              │   │                │ │
│  └──────────────┘    └──────────────┘   └────────────────┘ │
│           │                    │                  │          │
│           └────────┬───────────┴──────────────────┘          │
│                    ▼                                         │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Redis Cache                                       │    │
│  │  - Feature caching                                 │    │
│  │  - Session management                             │    │
│  │  - Performance metrics                             │    │
│  └────────────────────────────────────────────────────┘    │
│           │                                                  │
│           ▼                                                  │
│  ┌────────────────────────────────────────────────────┐    │
│  │  PostgreSQL 16 + TimescaleDB (Phase 1)             │    │
│  │  + Redis Cache (Phase 2)                           │    │
│  └────────────────────────────────────────────────────┘    │
│           │                                                  │
│           ▼                                                  │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Monitoring Stack                                  │    │
│  │  - Prometheus                                      │    │
│  │  - Grafana                                         │    │
│  │  - Custom dashboards for model performance         │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Phase 2 Services to Add

```yaml
# Message Queue
rabbitmq:
  image: rabbitmq:3-management-alpine
  ports:
    - "5672:5672"
    - "15672:15672"

# Redis Cache
redis:
  image: redis:7-alpine
  ports:
    - "6379:6379"

# GPU Training Service
training:
  image: nfl-analytics:training-gpu
  runtime: nvidia
  environment:
    - CUDA_VISIBLE_DEVICES=0

# API Gateway
api:
  image: nfl-analytics:api
  ports:
    - "8000:8000"
  depends_on:
    - pg-1
    - redis
    - rabbitmq

# Worker Pool (Celery)
workers:
  image: nfl-analytics:worker
  depends_on:
    - rabbitmq
    - redis
  deploy:
    replicas: 4
```

**Estimated Phase 2 Timeline:** 4-6 weeks
**Expected Performance Improvements:**
- 50% latency reduction (Redis caching)
- 5-10% ROI improvement (optimized prediction pipeline)
- 99.9% uptime (HA architecture)

---

## Summary

**Phase 1 Status:** Production Ready (November 8, 2025)

**Services Deployed:**
- ✅ PostgreSQL 16 + TimescaleDB (data warehouse)
- ✅ Streamlit Dashboard (interactive analysis)
- ✅ Prometheus (metrics collection)
- ✅ Grafana (visualization)
- ✅ PostgreSQL Exporter (database metrics)
- ✅ Node Exporter (system metrics)
- ✅ MCP Server (optional AI integration)
- ✅ Alertmanager (optional notifications)

**Key Capabilities:**
- 6,991 games (1999-2024)
- 1,230,857 play-by-play records
- 820,080 odds history records
- Automated health checks
- Comprehensive monitoring
- Persistent data storage
- Multi-machine support (coming Phase 2)

**Next Steps:**
1. Run Phase 1 setup: `docker compose up -d`
2. Verify all services: `docker compose ps`
3. Initialize database: See [Quick Start Guide](#quick-start-guide)
4. Access dashboards: http://localhost:8501, http://localhost:3000
5. Monitor health: `bash scripts/dev/health_check.sh`
6. Prepare for Phase 2: Configure API gateway and Redis

---

**Document:** DOCKER_PHASE1_SETUP.md
**Version:** 1.0
**Last Updated:** November 8, 2025
**Maintainer:** David Rodriguez-Obregon
**Status:** Ready for Production Deployment
