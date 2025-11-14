# C4 Container Diagram - NFL Analytics Docker Services

**Purpose:** Docker services architecture showing 7 production containers and data flows

**Date:** November 14, 2025

**Version:** 1.0

---

## Mermaid C4 Container Diagram

```mermaid
C4Container
    title Container Diagram - NFL Analytics Docker Services

    Person(bettor, "Bettor", "Places bets based on predictions")
    Person(ml_engineer, "ML Engineer", "Deploys and monitors models")

    System_Ext(nflverse, "nflverse/nflfastR", "NFL data provider")
    System_Ext(odds_api, "The Odds API", "Betting lines provider")

    Container_Boundary(docker_services, "Docker Compose Network (nfl-analytics)") {

        Container_Boundary(data_layer, "Data Layer") {
            ContainerDb(postgres, "PostgreSQL + TimescaleDB", "Port 5544", "Primary data store: 7,263 games, 1.25M plays, 56 tables")
            ContainerDb(redis, "Redis 7 Alpine", "Port 6379", "LRU cache: 2GB max memory, allkeys-lru eviction policy")
        }

        Container_Boundary(api_layer, "API Layer") {
            Container(api, "FastAPI Service", "Port 8000", "Ensemble v5.1 predictions: 4-way weighted (State-Space 30%, Hierarchical 30%, XGBoost 25%, Priors 15%)")
        }

        Container_Boundary(presentation_layer, "Presentation Layer") {
            Container(streamlit, "Streamlit Dashboard", "Port 8501", "4 pages: System Status, Model Status, Data Summary, Predictions")
        }

        Container_Boundary(monitoring_layer, "Monitoring Layer") {
            Container(prometheus, "Prometheus", "Port 9090", "Metrics collection: 30-day retention, recording rules, alert rules")
            Container(grafana, "Grafana", "Port 3000", "27-panel dashboard: System (4), DB (8), Redis (6), App (9)")
        }

        Container_Boundary(exporters, "Metrics Exporters") {
            Container(node_exporter, "Node Exporter", "Port 9100", "System metrics: CPU, memory, disk I/O, network")
            Container(postgres_exporter, "Postgres Exporter", "Port 9187", "Database metrics: connections, queries, cache hit ratio")
        }

        Container_Boundary(optional_services, "Optional Services (Profiles)") {
            Container(mcp_server, "MCP Server", "Port 3000", "Claude integration: Filesystem access, DB queries, model analysis")
            Container(odds_fetcher, "Odds Fetcher", "15min poll", "Live odds ingestion: Line movement alerts, threshold 5.0 points")
            Container(alertmanager, "Alertmanager", "Port 9093", "Alert routing: Webhook notifications, silencing, grouping")
        }
    }

    Rel(bettor, api, "POST /predict", "HTTPS/JSON")
    Rel(bettor, streamlit, "Views dashboards", "HTTPS")
    Rel(ml_engineer, grafana, "Monitors health", "HTTPS")

    Rel(api, postgres, "Fetches features", "psycopg2 connection pool")
    Rel(api, redis, "Caches predictions", "redis-py (TTL: 3600s)")
    Rel(streamlit, postgres, "Queries stats", "psycopg (read-only)")
    Rel(streamlit, redis, "Caches queries", "redis-py (TTL: 300s)")
    Rel(streamlit, api, "Fetches predictions", "HTTP GET /predict")

    Rel(prometheus, api, "Scrapes /metrics", "HTTP every 15s")
    Rel(prometheus, postgres_exporter, "Scrapes DB metrics", "HTTP every 10s")
    Rel(prometheus, node_exporter, "Scrapes system metrics", "HTTP every 10s")
    Rel(grafana, prometheus, "Queries metrics", "PromQL")
    Rel(grafana, postgres, "Direct queries", "PostgreSQL datasource")

    Rel(postgres_exporter, postgres, "Queries pg_stat_*", "SQL")
    Rel(node_exporter, docker_services, "Reads /proc, /sys", "Filesystem")

    Rel(mcp_server, postgres, "Queries data", "psycopg2")
    Rel(mcp_server, redis, "Caches results", "redis-py")
    Rel(odds_fetcher, odds_api, "Polls odds", "REST API")
    Rel(odds_fetcher, postgres, "Inserts odds", "psycopg2")
    Rel(alertmanager, prometheus, "Receives alerts", "Webhook")

    Rel(postgres, nflverse, "Ingests data", "R/nflverse (external)")

    UpdateLayoutConfig($c4ShapeInRow="2", $c4BoundaryInRow="1")
```

---

## Container Details

### 1. PostgreSQL + TimescaleDB (Port 5544)

**Image:** `timescale/timescaledb:latest-pg16`

**Purpose:** Primary data store for NFL play-by-play, game metadata, player stats, odds history

**Key Tables (56 total):**
- `public.plays` - 1,254,173 play-by-play records
- `public.games` - 7,263 games (2006-2024)
- `mart.player_game_stats` - Player-level stats by game
- `public.odds_history` - Historical betting lines
- `ml.training_runs` - Model training metadata

**Health Check:**
```bash
pg_isready -U dro -d devdb01
# Interval: 10s, Timeout: 5s, Retries: 5
```

**Volumes:**
- `./pgdata:/var/lib/postgresql/data` - Persistent storage

**Configuration:**
- User: `dro`
- Password: `sicillionbillions` (env var) <!-- pragma: allowlist secret -->
- Database: `devdb01`
- Extensions: TimescaleDB (time-series optimization)

**Performance:**
- Connection pooling: psycopg2 default (5 connections)
- Materialized views: `mart.game_summary` (refresh nightly)
- Indexes: Game date, player ID, week, season

---

### 2. Redis 7 Alpine (Port 6379)

**Image:** `redis:7-alpine`

**Purpose:** LRU cache for predictions, features, dashboard queries

**Configuration:**
```
--requirepass ${REDIS_PASSWORD}
--maxmemory 2gb
--maxmemory-policy allkeys-lru
```

**Cache TTLs:**
- Predictions: 3600s (1 hour)
- Dashboard queries: 300s (5 minutes)
- Features: 1800s (30 minutes)

**Performance:**
- Cache hit latency: 5-20ms
- Cache miss penalty: +125ms (DB query)
- Hit rate target: >50%

**Health Check:**
```bash
redis-cli -a ${REDIS_PASSWORD} ping
# Interval: 10s, Timeout: 5s, Retries: 5
```

**Volumes:**
- `redis_data:/data` - Persistent storage (RDB snapshots)

---

### 3. FastAPI Service (Port 8000)

**Build Context:** `.` (Dockerfile: `containers/Dockerfile.api`)

**Purpose:** Real-time NFL player prop predictions with uncertainty quantification

**Key Endpoints:**
- `GET /health` - Service health status
- `POST /predict` - Player prop prediction (145ms avg, 5ms cached)
- `POST /predict/game` - Game-level predictions (both QBs)
- `GET /metrics` - Prometheus metrics
- `GET /status` - Service status summary

**Ensemble v5.1 Weights:**
- State-Space v2: 30%
- Hierarchical v2: 30%
- Hybrid XGBoost v1: 25%
- Informative Priors v2: 15%

**Dependencies:**
```yaml
depends_on:
  pg-1:
    condition: service_healthy
  redis:
    condition: service_healthy
```

**Environment Variables:**
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `MODELS_DIR`: `/app/models` (mounted read-only)
- `LOG_LEVEL`: `INFO`

**Volumes:**
- `./api:/app/api` - API code (hot reload in dev)
- `./models:/app/models:ro` - Trained models (read-only)
- `./logs/api:/app/logs` - Application logs

**Health Check:**
```bash
curl -f http://localhost:8000/health
# Interval: 30s, Timeout: 10s, Retries: 5, Start period: 20s
```

**Performance:**
- Cold start: 5-8s (with cache warming)
- Prediction latency: 145ms (uncached), 5-20ms (cached)
- Throughput: ~100 req/s (single worker)

---

### 4. Streamlit Dashboard (Port 8501)

**Build Context:** `./dashboard` (Dockerfile: `dashboard/Dockerfile`)

**Purpose:** Interactive monitoring of predictions, CLV, model performance

**Pages (4 total):**
1. **System Status** - Service health checks (7 Docker services)
2. **Model Status** - Model registry, training runs, calibration
3. **Data Summary** - Database statistics (7,263 games, 1.25M plays)
4. **Predictions** - Visualization, confidence intervals, CLV tracking

**Dependencies:**
```yaml
depends_on:
  pg-1:
    condition: service_healthy
  redis:
    condition: service_healthy
  api:
    condition: service_healthy
```

**Environment Variables:**
- `DB_HOST`: `nfl-analytics-pg-1`
- `REDIS_HOST`: `nfl-analytics-redis`
- `LOG_LEVEL`: `INFO`

**Volumes:**
- `./dashboard:/app` - Dashboard code
- `./logs/streamlit:/app/logs` - Application logs
- `./models:/app/models:ro` - Model metadata (read-only)

**Health Check:**
```bash
curl -f http://localhost:8501/_stcore/health
# Interval: 30s, Timeout: 10s, Retries: 3, Start period: 20s
```

**Performance:**
- Page load: 1-3s (with Redis caching)
- Query cache TTL: 300s (5 minutes)
- Concurrent users: 10-20 (single container)

---

### 5. Prometheus (Port 9090)

**Image:** `prom/prometheus:latest`

**Purpose:** Metrics collection and time-series database

**Configuration:**
- Retention: 30 days (`--storage.tsdb.retention.time=30d`)
- Scrape interval: 15s (default)
- Query timeout: 2m

**Scrape Targets:**
- FastAPI `/metrics` (port 8000) - Every 15s
- Postgres Exporter (port 9187) - Every 10s
- Node Exporter (port 9100) - Every 10s
- Redis Exporter (via API metrics) - Every 15s

**Volumes:**
- `./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro` - Config
- `./prometheus/recording_rules.yml` - Aggregation rules
- `./prometheus/alert_rules.yml` - Alert definitions
- `prometheus_data:/prometheus` - Time-series storage

**Health Check:**
```bash
wget --quiet --tries=1 --spider http://localhost:9090/-/healthy
# Interval: 10s, Timeout: 5s, Retries: 5
```

**Key Metrics:**
- `nfl_predictions_total` - Prediction counter
- `nfl_prediction_duration_seconds` - Latency histogram
- `nfl_cache_hits_total` - Cache hit counter
- `pg_stat_database_tup_fetched` - DB queries
- `redis_memory_used_bytes` - Redis memory usage

---

### 6. Grafana (Port 3000)

**Image:** `grafana/grafana:latest`

**Purpose:** 27-panel dashboard for system, database, Redis, application metrics

**Dashboard Panels:**
- **System Overview (4 panels):** CPU, Memory, Disk I/O, Network
- **Database Metrics (8 panels):** Connections, queries, transactions, cache hit ratio, deadlocks, replication lag, table sizes, index usage
- **Redis Metrics (6 panels):** Memory usage, hit rate, clients, keys, evictions, commands/sec
- **Application Metrics (9 panels):** Request rate, latency (p50/p95/p99), error rate, prediction rate, cache hit rate, model load status, uptime, API health, Prometheus targets

**Configuration:**
- Admin user: `admin`
- Password: `sicillionbillions` (env var `GRAFANA_ADMIN_PASSWORD`) <!-- pragma: allowlist secret -->
- Root URL: `http://localhost:3000`

**Volumes:**
- `./grafana/grafana.ini:/etc/grafana/grafana.ini:ro` - Config
- `./grafana/provisioning:/etc/grafana/provisioning:ro` - Dashboards
- `grafana_data:/var/lib/grafana` - Persistent storage

**Health Check:**
```bash
wget --quiet --tries=1 --spider http://localhost:3000/api/health
# Interval: 10s, Timeout: 5s, Retries: 5, Start period: 20s
```

**Datasources:**
- Prometheus (default)
- PostgreSQL (direct queries for game data)

---

### 7. Node Exporter (Port 9100)

**Image:** `prom/node-exporter:latest`

**Purpose:** System-level metrics (CPU, memory, disk, network)

**Configuration:**
```
--path.procfs=/host/proc
--path.sysfs=/host/sys
--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)
```

**Volumes:**
- `/proc:/host/proc:ro` - Process metrics
- `/sys:/host/sys:ro` - System metrics
- `/:/rootfs:ro` - Filesystem metrics

**Key Metrics:**
- `node_cpu_seconds_total` - CPU usage
- `node_memory_MemAvailable_bytes` - Available memory
- `node_disk_io_time_seconds_total` - Disk I/O
- `node_network_receive_bytes_total` - Network RX
- `node_network_transmit_bytes_total` - Network TX

---

### 8. Postgres Exporter (Port 9187)

**Image:** `prometheuscommunity/postgres-exporter:latest`

**Purpose:** PostgreSQL-specific metrics for Prometheus

**Configuration:**
```
DATA_SOURCE_NAME=postgresql://dro:sicillionbillions@nfl-analytics-pg-1:5432/devdb01?sslmode=disable <!-- pragma: allowlist secret -->
```

**Key Metrics:**
- `pg_stat_database_tup_fetched` - Rows fetched
- `pg_stat_database_tup_returned` - Rows returned
- `pg_stat_database_blks_hit` - Cache hits
- `pg_stat_database_blks_read` - Disk reads
- `pg_stat_database_xact_commit` - Transactions committed
- `pg_stat_database_conflicts` - Replication conflicts
- `pg_stat_activity_count` - Active connections

**Health Check:**
```bash
wget --quiet --tries=1 --spider http://localhost:9187/metrics
# Interval: 10s, Timeout: 5s, Retries: 3
```

---

## Optional Services (Profiles)

### 9. MCP Server (Port 3000, Profile: `mcp`)

**Build Context:** `.` (Dockerfile: `mcp_server/Dockerfile`)

**Purpose:** Claude integration for AI-assisted analysis

**Capabilities:**
- Filesystem access to models, logs, data (read-only)
- Database queries (psycopg2)
- Redis caching
- Model artifact analysis

**Volumes:**
- `./models:/app/models:ro` - Model artifacts
- `./logs:/app/logs` - Log analysis
- `./data:/app/data:ro` - Data files
- `./results:/app/results` - Results output

**Usage:**
```bash
docker compose --profile mcp up -d
```

---

### 10. Odds Fetcher (15min poll, Profile: `odds`)

**Build Context:** `.` (Dockerfile: `containers/Dockerfile.odds-fetcher`)

**Purpose:** Real-time odds ingestion with line movement detection

**Configuration:**
- Poll interval: 900s (15 minutes)
- Line move threshold: 5.0 points
- API key: `${ODDS_API_KEY}` (env var)

**Features:**
- Fetches game spreads, totals, moneylines
- Tracks player props (passing yards, TDs)
- Alerts on significant line movements
- Webhook notifications (optional)

**Volumes:**
- `./logs/odds:/app/logs/odds` - Odds service logs

**Usage:**
```bash
docker compose --profile odds up -d
```

---

### 11. Alertmanager (Port 9093, Profile: `alerts`)

**Image:** `prom/alertmanager:latest`

**Purpose:** Alert routing, grouping, silencing

**Configuration:**
- `./prometheus/alertmanager.yml` - Routing rules
- Webhook receivers for Slack, PagerDuty, email

**Usage:**
```bash
docker compose --profile alerts up -d
```

---

## Data Flows

### Prediction Flow (Happy Path)

1. **Request:** Bettor → FastAPI (`POST /predict`)
2. **Cache Check:** FastAPI → Redis (`GET prediction:{player_id}:{week}:{season}`)
3. **Cache Miss:** FastAPI → PostgreSQL (fetch features)
4. **Ensemble:** FastAPI loads 4 models (State-Space, Hierarchical, XGBoost, Priors)
5. **Weighted Average:** 30% + 30% + 25% + 15% = ensemble prediction
6. **Temperature Scaling:** Apply T=0.5534 for calibration
7. **Cache Store:** FastAPI → Redis (`SET prediction:... TTL 3600`)
8. **Response:** FastAPI → Bettor (JSON with mean, std, CI)
9. **Metrics:** FastAPI → Prometheus (increment `nfl_predictions_total`)

**Total Latency:**
- Cache hit: 5-20ms
- Cache miss: 145ms

---

### Monitoring Flow

1. **Scrape:** Prometheus → API/Exporters (every 10-15s)
2. **Store:** Prometheus → Time-series DB (30-day retention)
3. **Query:** Grafana → Prometheus (PromQL)
4. **Visualize:** Grafana → Dashboard (27 panels)
5. **Alert:** Prometheus → Alertmanager (if thresholds exceeded)
6. **Notify:** Alertmanager → Webhook (Slack, email)

---

### Data Ingestion Flow

1. **Fetch:** R script → nflverse API (weekly)
2. **Transform:** R → PostgreSQL (via `DBI::dbWriteTable`)
3. **Materialize:** PostgreSQL → `REFRESH MATERIALIZED VIEW mart.game_summary`
4. **Index:** PostgreSQL → Create indexes on `kickoff`, `player_id`, `week`
5. **Cache Invalidate:** Redis → `FLUSHDB` (on schema changes)

---

## Network Configuration

**Docker Network:** `nfl-analytics` (external)

**Create Network:**
```bash
docker network create nfl-analytics
```

**Internal DNS:**
- `nfl-analytics-pg-1` (PostgreSQL)
- `nfl-analytics-redis` (Redis)
- `nfl-analytics-api` (FastAPI)
- `nfl-analytics-streamlit` (Streamlit)
- `nfl-analytics-prometheus` (Prometheus)
- `nfl-analytics-grafana` (Grafana)

**Port Mapping (Host → Container):**
- 5544 → 5432 (PostgreSQL)
- 6379 → 6379 (Redis)
- 8000 → 8000 (FastAPI)
- 8501 → 8501 (Streamlit)
- 9090 → 9090 (Prometheus)
- 3000 → 3000 (Grafana)
- 9100 → 9100 (Node Exporter)
- 9187 → 9187 (Postgres Exporter)

---

## Deployment

### Start All Core Services

```bash
docker compose up -d
```

**Services Started:**
1. PostgreSQL + TimescaleDB
2. Redis
3. FastAPI API
4. Streamlit Dashboard
5. Prometheus
6. Grafana
7. Node Exporter
8. Postgres Exporter

---

### Start with Optional Profiles

```bash
# With odds fetching
docker compose --profile odds up -d

# With MCP server
docker compose --profile mcp up -d

# With alerts
docker compose --profile alerts up -d

# All profiles
docker compose --profile odds --profile mcp --profile alerts up -d
```

---

### Health Checks

```bash
# Check all services
docker compose ps

# Individual health checks
curl http://localhost:8000/health       # FastAPI
curl http://localhost:8501/_stcore/health  # Streamlit
curl http://localhost:9090/-/healthy    # Prometheus
curl http://localhost:3000/api/health   # Grafana
```

---

### Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f api
docker compose logs -f streamlit
docker compose logs -f prometheus
```

---

### Stop Services

```bash
# Stop all
docker compose down

# Stop and remove volumes
docker compose down -v
```

---

## Volume Management

**Named Volumes:**
- `redis_data` - Redis RDB snapshots
- `prometheus_data` - Time-series metrics
- `grafana_data` - Dashboards and datasources
- `alertmanager_data` - Alert state

**Bind Mounts:**
- `./pgdata` - PostgreSQL data (7.2 GB)
- `./models` - Trained models (67 files, ~1 GB)
- `./logs` - Application logs
- `./grafana/provisioning` - Dashboard configs

**Backup:**
```bash
# Backup volumes
docker run --rm -v redis_data:/data -v $(pwd):/backup alpine tar czf /backup/redis_data.tar.gz /data

# Restore volumes
docker run --rm -v redis_data:/data -v $(pwd):/backup alpine tar xzf /backup/redis_data.tar.gz -C /
```

---

## Environment Variables

**Required:**
- `DB_PASSWORD` - PostgreSQL password
- `REDIS_PASSWORD` - Redis password

**Optional:**
- `ODDS_API_KEY` - The Odds API key (for odds fetcher)
- `GRAFANA_ADMIN_PASSWORD` - Grafana admin password (default: `sicillionbillions`)
- `CACHE_TTL_SECONDS` - Cache TTL (default: 3600)
- `ODDS_POLL_INTERVAL` - Odds poll frequency (default: 900)
- `CORS_ORIGINS` - Allowed CORS origins (default: `*`)

**Example `.env` file:**
```bash
DB_PASSWORD=sicillionbillions
REDIS_PASSWORD=changeme_redis_password
ODDS_API_KEY=your_api_key_here
GRAFANA_ADMIN_PASSWORD=sicillionbillions
CACHE_TTL_SECONDS=3600
```

---

**Author:** Claude Code (Sonnet 4.5)
**Date:** November 14, 2025
**Version:** 1.0
**Related Docs:** `c4_context.md`, `c4_component_ensemble.md`
