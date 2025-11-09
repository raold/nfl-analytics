# Docker Compose Phase 1 Refactoring Summary

**Date**: November 8, 2025
**Status**: Complete
**File**: `/docker-compose.yml`

## Overview

Refactored the monolithic `dashboard` service into a modular, scalable architecture with separated concerns. This enables independent scaling, deployment, and development of each component while maintaining unified networking and orchestration.

## Changes Made

### 1. Service Architecture Reorganization

**OLD STRUCTURE**: Monolithic dashboard container (148 lines)
```
dashboard (Streamlit + Prometheus + Grafana integrated via supervisor)
  └─ 3 ports: 8501, 9090, 3000
  └─ Single healthcheck (complex, fragile)
```

**NEW STRUCTURE**: Modular services (341 lines, +232% growth for clarity)
```
Primary Data Layer:
  └─ pg-1 (PostgreSQL + TimescaleDB) [UNCHANGED]

Caching Layer:
  └─ redis (NEW: Redis 7-Alpine)

API Layer:
  └─ api (NEW: FastAPI service)

Presentation Layer:
  ├─ streamlit (Streamlit dashboard)
  ├─ prometheus (Prometheus TSDB)
  └─ grafana (Grafana visualization)

Integration:
  └─ mcp-server (Claude MCP, optional profile)

Monitoring:
  ├─ postgres_exporter
  ├─ node_exporter
  └─ alertmanager (optional profile)
```

### 2. New Services Added

#### Redis (Caching Layer)
**Service**: `redis`
**Image**: `redis:7-alpine`
**Port**: `6379:6379`
**Configuration**:
- Max memory: 2GB
- Eviction policy: `allkeys-lru` (automatic cleanup on memory pressure)
- Data persistence: `redis_data` volume
- Health check: `redis-cli ping` (10s interval)

**Purpose**:
- Session caching for Streamlit
- Cache layer for API responses
- Feature caching for model predictions
- Real-time betting odds cache

#### FastAPI Service (API Layer)
**Service**: `api`
**Build**: `./api/Dockerfile`
**Port**: `8000:8000`
**Environment**:
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `API_HOST=0.0.0.0`, `API_PORT=8000`
- `LOG_LEVEL=INFO`

**Dependencies**:
- pg-1 (requires healthy)
- redis (requires healthy)

**Health Check**: `curl -f http://localhost:8000/health`

**Purpose**:
- RESTful API for model predictions
- Data validation and enrichment
- Cache management
- Asynchronous task processing

### 3. Split Dashboard Services

#### Streamlit Service
**Service**: `streamlit`
**Build**: `./dashboard/streamlit/Dockerfile` (new path)
**Port**: `8501:8501`
**Volumes**:
- `./dashboard/streamlit:/app` (source code)
- `./logs/streamlit:/app/logs` (logging)

**Dependencies**:
- pg-1 (healthy)
- redis (healthy)
- api (healthy)

**Health Check**: `curl -f http://localhost:8501/_stcore/health`

**Purpose**:
- Interactive data exploration
- Real-time model prediction interface
- Portfolio visualization
- Risk monitoring dashboard

#### Prometheus Service
**Service**: `prometheus`
**Image**: `prom/prometheus:latest`
**Port**: `9090:9090`
**User**: `65534` (prometheus user)
**Configuration Files**:
- `prometheus.yml` (scrape config)
- `recording_rules.yml` (computed metrics)
- `alert_rules.yml` (alerting rules)

**Volume**: `prometheus_data:/prometheus` (30-day retention)

**Health Check**: `wget --quiet --tries=1 --spider http://localhost:9090/-/healthy`

**Purpose**:
- Time-series metrics collection
- Recording rules evaluation
- Alert rule evaluation
- Metrics API for Grafana

#### Grafana Service
**Service**: `grafana`
**Image**: `grafana/grafana:latest`
**Port**: `3000:3000`
**User**: `472:472` (grafana user)
**Configuration**:
- Admin user: `admin` (default)
- Admin password: `${GRAFANA_ADMIN_PASSWORD:-sicillionbillions}`
- Sign-up disabled
- Root URL: `http://localhost:3000`

**Configuration Files**:
- `grafana/grafana.ini` (Grafana config)
- `grafana/provisioning/` (dashboards & datasources)

**Volume**: `grafana_data:/var/lib/grafana` (persistent dashboards)

**Dependencies**:
- prometheus (health check required)
- pg-1 (health check required)

**Health Check**: `wget --quiet --tries=1 --spider http://localhost:3000/api/health`

**Purpose**:
- Visual dashboards (model performance, system metrics)
- Data exploration interface
- Alert visualization
- Custom metric analysis

### 4. Service Dependencies & Health Checks

**Dependency Chain**:
```
pg-1 (foundation)
├─ ↓
redis (caching layer)
├─ ↓
api (API layer)
├─ ↓
streamlit (presentation)

prometheus
├─ ↓
grafana

mcp-server
├─ pg-1
└─ redis
```

**All new services include health checks**:
- Redis: `redis-cli ping`
- API: `curl -f http://localhost:8000/health`
- Streamlit: `curl -f http://localhost:8501/_stcore/health`
- Prometheus: `wget http://localhost:9090/-/healthy`
- Grafana: `wget http://localhost:3000/api/health`
- Exporters: Metrics endpoint checks

**Health Check Improvements**:
- Consistent 10s intervals (faster detection of failures)
- 5-second timeouts (prevent hanging health checks)
- 3-5 retries (avoid false negatives)
- 20s start_period (allow time for service initialization)

### 5. Networking

**Network**: `nfl-analytics` (external, pre-created)

**Service Connectivity**:
```
All services connected to nfl-analytics bridge network
├─ DNS resolution: container_name:port within network
├─ Example: api connects to pg-1 as nfl-analytics-pg-1:5432
└─ Services can communicate without port mapping
```

**Container Naming Convention**:
```
nfl-analytics-{service-name}
├─ nfl-analytics-pg-1
├─ nfl-analytics-redis
├─ nfl-analytics-api
├─ nfl-analytics-streamlit
├─ nfl-analytics-prometheus
├─ nfl-analytics-grafana
├─ nfl-analytics-mcp
├─ nfl-analytics-postgres-exporter
├─ nfl-analytics-node-exporter
└─ nfl-analytics-alertmanager
```

### 6. Volume Management

**New Volume**: `redis_data`
- Driver: local
- Purpose: Redis persistent data store
- Mount point: `/data` (inside container)

**Updated Volumes Section**:
```yaml
volumes:
  redis_data:        # NEW: Redis persistence
  prometheus_data:   # KEPT: Prometheus TSDB
  grafana_data:      # KEPT: Grafana dashboards
  alertmanager_data: # KEPT: Alert history
```

### 7. Environment Variables

**New Environment Variables**:

Redis (auto-configured):
```bash
redis://nfl-analytics-redis:6379
```

API Service:
```bash
DATABASE_URL=postgresql://dro:sicillionbillions@nfl-analytics-pg-1:5432/devdb01
REDIS_URL=redis://nfl-analytics-redis:6379
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

Streamlit Service:
```bash
REDIS_HOST=nfl-analytics-redis
REDIS_PORT=6379
```

MCP Server (updated):
```bash
REDIS_URL=redis://nfl-analytics-redis:6379
```

### 8. MCP Server Updates

**Changes**:
- Added `REDIS_URL` environment variable
- Updated health check: `curl -f http://localhost:3000/health`
- Port conflict resolution: MCP server runs on 3000 *internally*, Grafana on 3000 *externally*
  - Both are fine: MCP is optional (profile: mcp) and doesn't expose port
  - Grafana uses external port 3000

### 9. Updated Exporters

**postgres_exporter**:
- Added health check: `wget http://localhost:9187/metrics`
- Consistent with monitoring pattern

**node_exporter**:
- Added health check: `wget http://localhost:9100/metrics`
- Monitors host system metrics

**alertmanager** (optional profile: alerts):
- Added health check: `wget http://localhost:9093/-/healthy`
- Must be explicitly enabled: `docker compose --profile alerts up`

## Usage Guide

### Start Core Services (PostgreSQL + Redis)
```bash
docker compose up -d pg-1 redis
```

### Start Full Stack (data + API + dashboards)
```bash
docker compose up -d
```

### Start with Monitoring
```bash
docker compose --profile alerts up -d
```

### Start with MCP Server
```bash
docker compose --profile mcp up -d
```

### View Service Status
```bash
docker compose ps
docker compose logs -f [service-name]
```

### Health Check All Services
```bash
docker compose ps --format "table {{.Service}}\t{{.Status}}\t{{.Health}}"
```

### Update Individual Services
```bash
# Rebuild and restart API
docker compose up --build -d api

# Restart Streamlit
docker compose restart streamlit

# View logs
docker compose logs -f api streamlit
```

## Benefits of This Architecture

### 1. Independent Scaling
- Each service can be scaled independently
- Streamlit and Prometheus don't interfere with each other
- Can run multiple instances of API behind load balancer

### 2. Easier Debugging
- Failed health checks identify specific service
- Individual service logs without noise
- Can restart one component without restarting all

### 3. Separate Development
- Streamlit team works independently
- API team works independently
- Monitoring team works independently
- Changes don't require monolithic rebuild

### 4. Resource Efficiency
- Only run services needed for your task
- No overhead from unused components
- Can profile individual service resource usage

### 5. Production-Ready
- Each service has proper health checks
- Clear dependency ordering
- Separate restart policies
- Better suited for Kubernetes migration

### 6. Improved Monitoring
- Prometheus collects metrics independently
- Grafana visualizes all metrics in one place
- Can add more exporters (Redis exporter, API metrics, etc.)
- Exporters run alongside services, not mixed in

## Migration Steps

### For Development
1. Backup current setup: `docker compose down && cp -r pgdata pgdata.backup`
2. Update docker-compose.yml (already done)
3. Create API and Streamlit Dockerfiles if not present
4. Start services: `docker compose up -d`
5. Verify: `docker compose ps`

### For Production
1. Test new compose file on staging environment
2. Update PostgreSQL data volume mount if needed
3. Configure Redis persistence for production
4. Add Prometheus scrape targets for any new exporters
5. Update Grafana dashboards to point to new services
6. Set up monitoring alerts in alertmanager

## File Structure Required

```
nfl-analytics/
├── docker-compose.yml                    # Refactored
├── api/                                  # NEW: FastAPI service
│   ├── Dockerfile                        # NEW
│   ├── main.py                           # NEW
│   └── requirements.txt                  # NEW
├── dashboard/
│   ├── streamlit/                        # RENAMED from dashboard/ structure
│   │   ├── Dockerfile                    # MOVED/NEW
│   │   └── app.py                        # MOVED
│   ├── prometheus/                       # Moved out to top-level
│   │   ├── prometheus.yml
│   │   ├── recording_rules.yml
│   │   ├── alert_rules.yml
│   │   └── alertmanager.yml
│   └── grafana/                          # Moved out to top-level
│       ├── grafana.ini
│       └── provisioning/
├── prometheus/                           # New top-level config directory
│   ├── prometheus.yml
│   ├── recording_rules.yml
│   ├── alert_rules.yml
│   └── alertmanager.yml
├── grafana/                              # New top-level config directory
│   ├── grafana.ini
│   └── provisioning/
├── pgdata/                               # PostgreSQL data (unchanged)
├── logs/                                 # NEW: Centralized logging
│   ├── api/
│   ├── streamlit/
│   └── mcp/
└── mcp_server/                           # MCP service (unchanged)
    └── Dockerfile
```

## Performance Characteristics

### Resource Utilization
- **pg-1**: 2-4 GB RAM (database workload dependent)
- **redis**: Fixed 2GB maxmemory (allkeys-lru eviction)
- **api**: 200-500 MB RAM (depends on concurrent requests)
- **streamlit**: 300-800 MB RAM (interactive state)
- **prometheus**: 500 MB - 2 GB (retention: 30 days)
- **grafana**: 100-300 MB RAM
- **exporters**: 50-100 MB each
- **Total baseline**: ~4-6 GB

### Network Traffic
- **pg-1 ↔ api**: Low latency (localhost)
- **pg-1 ↔ postgres_exporter**: Polling (10s interval)
- **api ↔ streamlit**: HTTP (user-initiated)
- **prometheus ↔ exporters**: Polling (15-30s intervals)
- **grafana ↔ prometheus**: Queries (user-initiated)

## Rollback Plan

If issues arise:
```bash
# Restore old docker-compose.yml
git checkout docker-compose.yml

# Stop new services
docker compose down

# Restart old monolithic dashboard
docker compose up -d

# Verify
docker compose ps
```

## Next Steps

### Phase 1B: API Implementation
- [ ] Create `api/Dockerfile` (FastAPI service)
- [ ] Implement `/health` endpoint
- [ ] Add prediction endpoints
- [ ] Implement caching with Redis
- [ ] Add request validation and logging

### Phase 1C: Streamlit Reorganization
- [ ] Move Streamlit code to `dashboard/streamlit/`
- [ ] Create dedicated Dockerfile
- [ ] Add Redis client integration
- [ ] Test independent startup

### Phase 2: Prometheus Configuration
- [ ] Update `prometheus/prometheus.yml` with all scrape targets
- [ ] Add API metrics scraping
- [ ] Create `recording_rules.yml` for computed metrics
- [ ] Add `alert_rules.yml` with thresholds

### Phase 3: Grafana Setup
- [ ] Create `grafana/provisioning/dashboards/` directory
- [ ] Provision Prometheus datasource
- [ ] Create system monitoring dashboard
- [ ] Create model performance dashboard
- [ ] Create prediction accuracy dashboard

## Validation

To verify the refactoring is complete:

1. **Check service startup**:
   ```bash
   docker compose up -d
   sleep 30
   docker compose ps
   # All services should show "healthy" or "running"
   ```

2. **Test connectivity**:
   ```bash
   docker compose exec api curl http://nfl-analytics-pg-1:5432
   docker compose exec api redis-cli -h nfl-analytics-redis ping
   ```

3. **Verify logging**:
   ```bash
   docker compose logs api | head -20
   docker compose logs streamlit | head -20
   ```

4. **Check Prometheus targets**:
   ```bash
   curl http://localhost:9090/api/v1/targets
   ```

5. **Verify Grafana datasource**:
   ```bash
   curl -u admin:sicillionbillions http://localhost:3000/api/datasources
   ```

---

**Status**: Phase 1 Docker Compose refactoring complete and ready for Phase 1B (API implementation).
