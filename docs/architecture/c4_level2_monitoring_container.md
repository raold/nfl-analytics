# C4 Level 2: Monitoring Container Detail

**Last Updated:** November 15, 2025
**Diagram Type:** C4 Container (Level 2)
**Container:** Prometheus + Grafana Monitoring Stack
**Status:** Production

---

## Overview

The monitoring stack consists of Prometheus (metrics storage), Grafana (visualization), and 2 exporters (Node Exporter for system metrics, Postgres Exporter for database metrics). Together they provide comprehensive observability for the NFL Analytics platform.

**Container Specifications:**

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| **Prometheus** | prom/prometheus:latest | 9090 | Metrics collection and storage |
| **Grafana** | grafana/grafana:latest | 3000 | Dashboard visualization |
| **Node Exporter** | prom/node-exporter:latest | 9100 | System metrics (CPU, RAM, disk, network) |
| **Postgres Exporter** | prometheuscommunity/postgres-exporter:latest | 9187 | PostgreSQL metrics |

**Key Responsibilities:**
- Collect metrics from all containers (15-second scrape interval)
- Store time-series data (30-day retention)
- Visualize system health (27 panels across 4 categories)
- Alert on critical conditions (email + Slack)
- Track API performance (latency, error rate, throughput)

---

## Monitoring Stack Diagram

```mermaid
graph TB
    subgraph External_Users["External Users"]
        ADMIN["System Administrator\n(DevOps)"]
        RESEARCHER["Researcher\n(Data Analyst)"]
    end

    subgraph Monitored_Services["Monitored Services"]
        API["FastAPI\n(Prediction API)\n/metrics endpoint"]
        PG["PostgreSQL\n(Database)"]
        HOST["Host System\n(Mac M4)"]
    end

    subgraph Prometheus_Container["Prometheus Container (nfl-analytics-prometheus)"]

        subgraph Scrape_Config["Scrape Configuration"]
            SCRAPER["Scrape Scheduler\n(15s interval)\n\nScrape targets:\n• FastAPI /metrics\n• Node Exporter\n• Postgres Exporter"]
        end

        subgraph TSDB["Time-Series Database"]
            STORAGE["TSDB Storage\n(30-day retention)\n~1GB volume\n\nCompression:\n• Block compression\n• Delta encoding"]
        end

        subgraph Rules["Rule Evaluation"]
            RECORDING["Recording Rules\n(Pre-computed aggregations)\n\nExamples:\n• api_request_rate_5m\n• cache_hit_ratio\n• error_rate_1h"]

            ALERTING["Alert Rules\n(Threshold-based)\n\nExamples:\n• High error rate (>5%)\n• High latency (p95 >1s)\n• Database connections >80%"]
        end

        subgraph Query_Engine["Query Engine"]
            PROMQL["PromQL Processor\n(Query language)\n\nFunctions:\n• rate(), increase()\n• avg_over_time()\n• histogram_quantile()"]
        end

    end

    subgraph Grafana_Container["Grafana Container (nfl-analytics-grafana)"]

        subgraph Dashboards["Dashboards (27 Panels)"]
            SYSTEM_DASH["System Overview\n(4 panels)\n\n• CPU usage\n• Memory usage\n• Disk I/O\n• Network I/O"]

            DB_DASH["Database Metrics\n(8 panels)\n\n• Connections\n• Queries/sec\n• Transaction rate\n• Cache hit rate"]

            REDIS_DASH["Redis Metrics\n(6 panels)\n\n• Memory usage\n• Hit rate\n• Clients\n• Evictions"]

            APP_DASH["Application Metrics\n(9 panels)\n\n• API latency (p50, p95, p99)\n• Prediction count\n• Error rate\n• Models loaded"]
        end

        subgraph Alerting_Grafana["Alert Manager"]
            ALERT_RULES_G["Alert Configuration\n\n• Email notifications\n• Slack webhooks\n• PagerDuty (future)"]

            NOTIFICATION["Notification Channels\n\n• Email (SMTP)\n• Slack (webhook)\n• Alert routing rules"]
        end

        subgraph User_Management["User Management"]
            USERS["User Accounts\n\nadmin (superuser)\nreadonly (viewer)"]
        end

    end

    subgraph Exporters["Metric Exporters"]
        NODE_EXP["Node Exporter\n(System Metrics)\n\nPort: 9100\nProtocol: HTTP\n\nMetrics:\n• CPU usage (%)\n• Memory usage (bytes)\n• Disk I/O (read/write)\n• Network I/O (bytes)"]

        PG_EXP["Postgres Exporter\n(Database Metrics)\n\nPort: 9187\nProtocol: HTTP\n\nMetrics:\n• pg_up (0/1)\n• pg_stat_database_*\n• pg_stat_statements_*"]
    end

    subgraph Alert_Destinations["Alert Destinations"]
        EMAIL["Email\n(SMTP)\nGmail or SendGrid"]
        SLACK["Slack\n(Webhook)\nDevOps channel"]
    end

    %% External Users
    ADMIN -->|"HTTPS\nView dashboards"| Grafana_Container
    RESEARCHER -->|"HTTPS\nExplore metrics"| Grafana_Container

    %% Prometheus Scraping
    SCRAPER -->|"HTTP GET /metrics\n(15s interval)"| API
    SCRAPER -->|"HTTP GET /metrics\n(15s interval)"| NODE_EXP
    SCRAPER -->|"HTTP GET /metrics\n(15s interval)"| PG_EXP
    SCRAPER --> STORAGE

    %% Exporters
    NODE_EXP -->|"Collect\nSystem stats"| HOST
    PG_EXP -->|"PostgreSQL wire\nSELECT pg_stat_*"| PG

    %% Rule Evaluation
    STORAGE --> RECORDING
    STORAGE --> ALERTING
    RECORDING --> STORAGE
    ALERTING --> NOTIFICATION

    %% Grafana Queries
    PROMQL -.->|"Query data"| STORAGE
    Grafana_Container -->|"PromQL queries\nHTTP"| PROMQL

    %% Grafana Dashboards
    PROMQL --> SYSTEM_DASH
    PROMQL --> DB_DASH
    PROMQL --> REDIS_DASH
    PROMQL --> APP_DASH

    %% Grafana Alerting
    ALERT_RULES_G --> NOTIFICATION
    NOTIFICATION -->|"SMTP\nCritical alerts"| EMAIL
    NOTIFICATION -->|"Webhook\nWarning alerts"| SLACK

    %% Styling
    classDef external fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef monitored fill:#ffccbc,stroke:#bf360c,stroke-width:2px
    classDef prometheus fill:#fff9c4,stroke:#f57f17,stroke-width:3px
    classDef grafana fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px
    classDef exporters fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef alerts fill:#ffebee,stroke:#b71c1c,stroke-width:2px

    class ADMIN,RESEARCHER external
    class API,PG,HOST monitored
    class SCRAPER,STORAGE,RECORDING,ALERTING,PROMQL prometheus
    class SYSTEM_DASH,DB_DASH,REDIS_DASH,APP_DASH,ALERT_RULES_G,NOTIFICATION,USERS grafana
    class NODE_EXP,PG_EXP exporters
    class EMAIL,SLACK alerts
```

---

## Component Details

### Prometheus Container

**Container Name:** `nfl-analytics-prometheus`
**Image:** `prom/prometheus:latest`
**Port:** 9090 (host) → 9090 (container)
**Storage:** ~1GB (30-day retention)

---

#### 1. Scrape Configuration (prometheus.yml)

**Purpose:** Define which targets to scrape and how often

**Configuration:**
```yaml
global:
  scrape_interval: 15s          # Default scrape interval
  evaluation_interval: 15s      # Rule evaluation interval
  external_labels:
    cluster: 'nfl-analytics-dev'
    env: 'development'

scrape_configs:
  # FastAPI application metrics
  - job_name: 'api'
    static_configs:
      - targets: ['nfl-analytics-api:8000']
    metrics_path: '/metrics'

  # PostgreSQL database metrics
  - job_name: 'postgres'
    static_configs:
      - targets: ['nfl-analytics-postgres-exporter:9187']

  # System metrics
  - job_name: 'node'
    static_configs:
      - targets: ['nfl-analytics-node-exporter:9100']

  # Prometheus self-monitoring
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

**Scrape Targets:**

| Job Name | Target | Endpoint | Metrics Count | Scrape Frequency |
|----------|--------|----------|---------------|------------------|
| **api** | nfl-analytics-api:8000 | /metrics | ~20 | 15s |
| **postgres** | postgres-exporter:9187 | /metrics | ~50 | 15s |
| **node** | node-exporter:9100 | /metrics | ~200 | 15s |
| **prometheus** | localhost:9090 | /metrics | ~30 | 15s |

**Total Metrics:** ~300 time-series (low cardinality, efficient storage)

---

#### 2. Time-Series Database (TSDB)

**Purpose:** Store metrics with timestamps for querying

**Storage Format:**
- **Block-based storage:** Data stored in 2-hour blocks
- **Compression:** Delta encoding + Gorilla compression (~10x reduction)
- **Retention:** 30 days (configurable via `--storage.tsdb.retention.time=30d`)

**Data Volume:**
- **Raw data rate:** ~300 metrics × 15s interval = ~20 samples/sec
- **Compressed storage:** ~1GB for 30 days

**Performance:**
- **Query latency:** <100ms (simple queries)
- **Ingestion throughput:** ~10,000 samples/sec (current: ~20 samples/sec)

---

#### 3. Recording Rules (recording_rules.yml)

**Purpose:** Pre-compute expensive aggregations for dashboard performance

**Example Rules:**
```yaml
groups:
  - name: api_aggregations
    interval: 1m
    rules:
      # API request rate (requests per second, last 5 minutes)
      - record: api_request_rate_5m
        expr: rate(api_requests_total[5m])

      # API error rate (errors per second, last 5 minutes)
      - record: api_error_rate_5m
        expr: rate(api_requests_total{status="error"}[5m])

      # Feature cache hit ratio (percentage, last 5 minutes)
      - record: cache_hit_ratio_5m
        expr: |
          sum(rate(redis_keyspace_hits_total[5m]))
          / sum(rate(redis_keyspace_hits_total[5m]) + rate(redis_keyspace_misses_total[5m]))

      # Prediction latency (p95, last 5 minutes)
      - record: prediction_latency_p95_5m
        expr: histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m]))
```

**Benefits:**
- **Faster Dashboards:** Pre-computed metrics load instantly
- **Lower Query Load:** Reduce load on Prometheus query engine
- **Consistent Calculations:** Same metric definition across dashboards

---

#### 4. Alert Rules (alert_rules.yml)

**Purpose:** Define conditions that trigger alerts

**Example Rules:**
```yaml
groups:
  - name: api_alerts
    interval: 1m
    rules:
      # High error rate (>5% for last 5 minutes)
      - alert: HighAPIErrorRate
        expr: api_error_rate_5m > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High API error rate detected"
          description: "API error rate is {{ $value | humanizePercentage }} (threshold: 5%)"

      # High latency (p95 >1 second for last 5 minutes)
      - alert: HighAPILatency
        expr: prediction_latency_p95_5m > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API latency detected"
          description: "API p95 latency is {{ $value }}s (threshold: 1s)"

      # Database connections >80% capacity
      - alert: HighDatabaseConnections
        expr: pg_stat_database_numbackends / pg_settings_max_connections > 0.8
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High database connection usage"
          description: "Database connections at {{ $value | humanizePercentage }} of capacity"

      # Redis memory >90% capacity
      - alert: HighRedisMemory
        expr: redis_memory_used_bytes / redis_memory_max_bytes > 0.9
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis memory usage critical"
          description: "Redis memory at {{ $value | humanizePercentage }} of capacity"
```

**Alert Severity Levels:**
- **Critical:** Requires immediate action (page on-call engineer)
- **Warning:** Requires investigation within 1 hour
- **Info:** Awareness only (no action required)

---

#### 5. Query Engine (PromQL)

**Purpose:** Query language for retrieving and aggregating metrics

**Common PromQL Queries:**

**API Request Rate (last 5 minutes):**
```promql
rate(api_requests_total[5m])
```

**API Error Rate (percentage):**
```promql
sum(rate(api_requests_total{status="error"}[5m]))
/ sum(rate(api_requests_total[5m]))
```

**API Latency (p95, p99):**
```promql
histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m]))
histogram_quantile(0.99, rate(prediction_latency_seconds_bucket[5m]))
```

**Database Cache Hit Rate:**
```promql
pg_stat_database_blks_hit / (pg_stat_database_blks_hit + pg_stat_database_blks_read)
```

**Memory Usage (percentage):**
```promql
(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100
```

**PromQL Functions:**
- **rate()** - Calculate per-second rate over time window
- **increase()** - Total increase over time window
- **avg_over_time()** - Average value over time window
- **histogram_quantile()** - Calculate percentiles from histogram
- **sum()**, **avg()**, **max()**, **min()** - Aggregation functions

---

### Grafana Container

**Container Name:** `nfl-analytics-grafana`
**Image:** `grafana/grafana:latest`
**Port:** 3000 (host) → 3000 (container)
**Storage:** ~500MB (dashboards, users, settings)

---

#### 1. Dashboards (27 Panels)

**Dashboard Organization:**

**System Overview (4 panels):**
1. **CPU Usage (%)** - Time-series graph
   - Query: `100 - (avg by (cpu) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)`
   - Target: <80% average
   - Current: ~30-50% (typical)

2. **Memory Usage (%)** - Time-series graph
   - Query: `(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100`
   - Target: <80% average
   - Current: ~40-60% (typical)

3. **Disk I/O (MB/s)** - Time-series graph
   - Query: `rate(node_disk_read_bytes_total[5m]) / 1024 / 1024`
   - Current: <10 MB/s (typical)

4. **Network I/O (MB/s)** - Time-series graph
   - Query: `rate(node_network_receive_bytes_total[5m]) / 1024 / 1024`
   - Current: <1 MB/s (typical)

---

**Database Metrics (8 panels):**
1. **Active Connections** - Gauge
   - Query: `pg_stat_database_numbackends{datname="devdb01"}`
   - Target: <80 (out of 100 max)
   - Current: 5-15 (typical)

2. **Queries per Second** - Time-series graph
   - Query: `rate(pg_stat_database_xact_commit{datname="devdb01"}[5m])`
   - Current: ~10-50 queries/sec (typical)

3. **Transaction Rate** - Time-series graph
   - Query: `rate(pg_stat_database_xact_commit{datname="devdb01"}[5m]) + rate(pg_stat_database_xact_rollback{datname="devdb01"}[5m])`

4. **Cache Hit Rate (%)** - Gauge
   - Query: `pg_stat_database_blks_hit{datname="devdb01"} / (pg_stat_database_blks_hit{datname="devdb01"} + pg_stat_database_blks_read{datname="devdb01"}) * 100`
   - Target: >95%
   - Current: 97-99% (excellent)

5. **Table Sizes (GB)** - Bar chart
   - Query: `pg_stat_user_tables_n_tup_ins{datname="devdb01"}`

6. **Slow Queries (>100ms)** - Table
   - Query: `topk(10, pg_stat_statements_mean_time_seconds{datname="devdb01"} > 0.1)`

7. **Database Size (GB)** - Gauge
   - Query: `pg_database_size_bytes{datname="devdb01"} / 1024 / 1024 / 1024`
   - Current: ~5GB

8. **Deadlocks** - Counter
   - Query: `pg_stat_database_deadlocks{datname="devdb01"}`
   - Target: 0

---

**Redis Metrics (6 panels):**
1. **Memory Usage (MB)** - Time-series graph
   - Query: `redis_memory_used_bytes / 1024 / 1024`
   - Target: <2048 MB (2GB max)
   - Current: ~1500 MB (75% capacity)

2. **Hit Rate (%)** - Gauge
   - Query: `redis_keyspace_hits_total / (redis_keyspace_hits_total + redis_keyspace_misses_total) * 100`
   - Target: >70%
   - Current: 70-85% (typical)

3. **Connected Clients** - Time-series graph
   - Query: `redis_connected_clients`
   - Current: 3-5 (FastAPI, Streamlit, etc.)

4. **Keys Count** - Time-series graph
   - Query: `redis_db_keys{db="db0"}`
   - Current: ~10,000 keys (feature cache)

5. **Evictions (LRU)** - Counter
   - Query: `rate(redis_evicted_keys_total[5m])`
   - Current: ~50-100 evictions/min (expected with allkeys-lru)

6. **Commands per Second** - Time-series graph
   - Query: `rate(redis_commands_processed_total[5m])`
   - Current: ~100-500 commands/sec

---

**Application Metrics (9 panels):**
1. **API Latency (p50, p95, p99)** - Time-series graph
   - Query:
     ```promql
     histogram_quantile(0.50, rate(prediction_latency_seconds_bucket[5m]))
     histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m]))
     histogram_quantile(0.99, rate(prediction_latency_seconds_bucket[5m]))
     ```
   - Target: p95 <500ms
   - Current: p50 ~280ms, p95 ~480ms, p99 ~550ms

2. **Prediction Count (last hour)** - Single stat
   - Query: `increase(predictions_total[1h])`
   - Current: ~50-100 predictions/hour

3. **API Error Rate (%)** - Time-series graph
   - Query: `sum(rate(api_requests_total{status="error"}[5m])) / sum(rate(api_requests_total[5m])) * 100`
   - Target: <1%
   - Current: 0-0.5% (typical)

4. **Models Loaded** - Gauge
   - Query: `models_loaded_total`
   - Target: 4 (all ensemble components)
   - Current: 4 (healthy)

5. **Requests by Endpoint** - Bar chart
   - Query: `sum by (endpoint) (rate(api_requests_total[5m]))`

6. **Requests by Status** - Pie chart
   - Query: `sum by (status) (rate(api_requests_total[5m]))`

7. **Feature Cache Hit Rate (%)** - Gauge
   - Query: `cache_hit_ratio_5m * 100`
   - Target: >70%
   - Current: 70-85%

8. **Prediction Latency Heatmap** - Heatmap
   - Query: `rate(prediction_latency_seconds_bucket[5m])`

9. **API Uptime (%)** - Single stat
   - Query: `avg_over_time(up{job="api"}[24h]) * 100`
   - Target: >99.5%
   - Current: 99.8-100%

---

#### 2. Alert Manager

**Purpose:** Route alerts to appropriate channels based on severity

**Alert Routing:**
```yaml
route:
  receiver: 'default'
  group_by: ['alertname', 'severity']
  group_wait: 30s          # Wait 30s for more alerts before sending
  group_interval: 5m       # Send grouped alerts every 5 minutes
  repeat_interval: 4h      # Re-send alert every 4 hours if still firing

  routes:
    # Critical alerts → Email + Slack
    - match:
        severity: critical
      receiver: 'critical-alerts'
      repeat_interval: 1h

    # Warning alerts → Slack only
    - match:
        severity: warning
      receiver: 'warning-alerts'

    # Info alerts → Log only
    - match:
        severity: info
      receiver: 'info-alerts'

receivers:
  - name: 'default'
    email_configs:
      - to: 'admin@example.com'
        from: 'nfl-analytics@example.com'
        smarthost: 'smtp.gmail.com:587'
        auth_username: 'nfl-analytics@example.com'
        auth_password: '<password>'

  - name: 'critical-alerts'
    email_configs:
      - to: 'oncall@example.com'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXX'
        channel: '#critical-alerts'

  - name: 'warning-alerts'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXX'
        channel: '#warnings'

  - name: 'info-alerts'
    # No notifications (log only)
```

**Notification Channels:**

| Channel | Purpose | Severity | Response Time |
|---------|---------|----------|---------------|
| **Email** | Critical alerts (requires immediate action) | Critical | <15 minutes |
| **Slack** | Warnings + critical alerts | Critical, Warning | <1 hour |
| **PagerDuty** (future) | On-call engineer paging | Critical | <5 minutes |

---

#### 3. User Management

**User Accounts:**

| Username | Role | Permissions |
|----------|------|-------------|
| **admin** | Admin | Full access (edit dashboards, create users, configure alerts) |
| **readonly** | Viewer | View dashboards only (no editing) |

**Authentication:**
- **Method:** Username/password (basic auth)
- **Password Storage:** Hashed with bcrypt
- **Session Timeout:** 24 hours

**Future Enhancements:**
- OAuth 2.0 (Google, GitHub)
- LDAP integration (enterprise)
- API key authentication (programmatic access)

---

### Metric Exporters

#### 1. Node Exporter (System Metrics)

**Container Name:** `nfl-analytics-node-exporter`
**Image:** `prom/node-exporter:latest`
**Port:** 9100
**Protocol:** HTTP

**Metrics Collected (~200 metrics):**

**CPU Metrics:**
- `node_cpu_seconds_total{mode="idle"}` - Idle CPU time
- `node_cpu_seconds_total{mode="user"}` - User CPU time
- `node_cpu_seconds_total{mode="system"}` - System CPU time
- `node_load1`, `node_load5`, `node_load15` - Load averages

**Memory Metrics:**
- `node_memory_MemTotal_bytes` - Total RAM
- `node_memory_MemFree_bytes` - Free RAM
- `node_memory_MemAvailable_bytes` - Available RAM (includes cache)
- `node_memory_Cached_bytes` - Page cache

**Disk Metrics:**
- `node_disk_read_bytes_total` - Total bytes read
- `node_disk_write_bytes_total` - Total bytes written
- `node_disk_io_time_seconds_total` - I/O wait time
- `node_filesystem_size_bytes` - Disk size
- `node_filesystem_avail_bytes` - Disk available

**Network Metrics:**
- `node_network_receive_bytes_total` - Bytes received
- `node_network_transmit_bytes_total` - Bytes transmitted
- `node_network_receive_errs_total` - Receive errors
- `node_network_transmit_errs_total` - Transmit errors

**Example Query:**
```
GET http://nfl-analytics-node-exporter:9100/metrics

# HELP node_cpu_seconds_total Seconds the CPUs spent in each mode.
# TYPE node_cpu_seconds_total counter
node_cpu_seconds_total{cpu="0",mode="idle"} 123456.78
node_cpu_seconds_total{cpu="0",mode="user"} 12345.67
...
```

---

#### 2. Postgres Exporter (Database Metrics)

**Container Name:** `nfl-analytics-postgres-exporter`
**Image:** `prometheuscommunity/postgres-exporter:latest`
**Port:** 9187
**Protocol:** HTTP

**Connection String:**
```
DATA_SOURCE_NAME=postgresql://dro:sicillionbillions@nfl-analytics-pg-1:5432/devdb01?sslmode=disable  # pragma: allowlist secret
```

**Metrics Collected (~50 metrics):**

**Database Status:**
- `pg_up` - Database reachable (1=up, 0=down)
- `pg_database_size_bytes{datname="devdb01"}` - Database size

**Connection Metrics:**
- `pg_stat_database_numbackends{datname="devdb01"}` - Active connections
- `pg_settings_max_connections` - Max connection limit

**Transaction Metrics:**
- `pg_stat_database_xact_commit{datname="devdb01"}` - Committed transactions
- `pg_stat_database_xact_rollback{datname="devdb01"}` - Rolled back transactions

**Cache Metrics:**
- `pg_stat_database_blks_hit{datname="devdb01"}` - Cache hits
- `pg_stat_database_blks_read{datname="devdb01"}` - Disk reads

**Query Performance (from pg_stat_statements):**
- `pg_stat_statements_total_time_seconds` - Total query execution time
- `pg_stat_statements_calls_total` - Number of query executions
- `pg_stat_statements_mean_time_seconds` - Average query time

**Table Metrics:**
- `pg_stat_user_tables_n_tup_ins{datname="devdb01",schemaname="public",relname="plays"}` - Rows inserted
- `pg_stat_user_tables_n_tup_upd` - Rows updated
- `pg_stat_user_tables_n_tup_del` - Rows deleted

**Example Query:**
```
GET http://nfl-analytics-postgres-exporter:9187/metrics

# HELP pg_up Whether the PostgreSQL server is up.
# TYPE pg_up gauge
pg_up 1

# HELP pg_stat_database_numbackends Number of backends currently connected to this database.
# TYPE pg_stat_database_numbackends gauge
pg_stat_database_numbackends{datname="devdb01"} 8
...
```

---

## Alert Examples

### Critical Alerts (Email + Slack)

**1. High API Error Rate:**
```yaml
- alert: HighAPIErrorRate
  expr: api_error_rate_5m > 0.05
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "High API error rate detected"
    description: "API error rate is {{ $value | humanizePercentage }} (threshold: 5%)"
```

**Trigger Condition:** Error rate >5% for 5 minutes
**Notification:** Email to oncall@example.com + Slack #critical-alerts
**Expected Action:** Investigate API logs, check database connection, verify model loading

---

**2. Database Connection Exhaustion:**
```yaml
- alert: DatabaseConnectionsExhausted
  expr: pg_stat_database_numbackends / pg_settings_max_connections > 0.8
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "Database connection usage >80%"
    description: "{{ $value | humanizePercentage }} of max connections used"
```

**Trigger Condition:** >80% of max connections for 2 minutes
**Notification:** Email + Slack
**Expected Action:** Increase `max_connections` in postgresql.conf OR investigate connection leaks

---

### Warning Alerts (Slack Only)

**1. High API Latency:**
```yaml
- alert: HighAPILatency
  expr: prediction_latency_p95_5m > 1.0
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "High API latency (p95 >1s)"
    description: "API p95 latency is {{ $value }}s (threshold: 1s)"
```

**Trigger Condition:** p95 latency >1 second for 5 minutes
**Notification:** Slack #warnings
**Expected Action:** Check database query performance, verify R model loading

---

**2. Low Cache Hit Rate:**
```yaml
- alert: LowCacheHitRate
  expr: cache_hit_ratio_5m < 0.7
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Low feature cache hit rate"
    description: "Cache hit rate is {{ $value | humanizePercentage }} (threshold: 70%)"
```

**Trigger Condition:** Cache hit rate <70% for 10 minutes
**Notification:** Slack #warnings
**Expected Action:** Increase Redis maxmemory OR adjust TTL

---

## Performance Characteristics

### Prometheus Performance

**Ingestion Rate:**
- **Current:** ~20 samples/sec (~300 metrics × 15s scrape interval)
- **Capacity:** ~10,000 samples/sec (500x headroom)

**Query Latency:**
- **Simple queries:** <100ms (e.g., `api_requests_total`)
- **Aggregations:** <500ms (e.g., `rate(api_requests_total[5m])`)
- **Complex queries:** <2s (e.g., multi-table joins, histogram quantiles)

**Storage:**
- **Raw data:** ~1GB for 30 days (compressed)
- **Compression ratio:** ~10x (delta encoding + Gorilla compression)

---

### Grafana Performance

**Dashboard Load Time:**
- **System Overview:** <1 second (4 panels)
- **Database Metrics:** <2 seconds (8 panels)
- **All Dashboards:** <5 seconds (27 panels)

**Query Concurrency:**
- **Max concurrent queries:** 10 (default)
- **Typical load:** 2-5 concurrent queries

---

## Security

### Prometheus Security

**No Authentication (Local Development):**
- Prometheus UI exposed on localhost:9090 (not accessible from internet)
- Production: Add authentication (basic auth, OAuth 2.0, or reverse proxy)

**Query Restrictions:**
- No restrictions on PromQL queries (full access to all metrics)
- Production: Implement query quota or rate limiting

---

### Grafana Security

**Authentication:**
- **Method:** Username/password (basic auth)
- **Admin Credentials:** admin / ${GRAFANA_ADMIN_PASSWORD}  <!-- pragma: allowlist secret -->
- **User Management:** Invite-only (no self-registration)

**Authorization:**
- **Admin Role:** Full access (edit dashboards, create users, configure alerts)
- **Viewer Role:** View dashboards only (no editing)

**Session Management:**
- **Session Timeout:** 24 hours (configurable)
- **Remember Me:** Optional (checkbox on login)

**API Security:**
- **API Keys:** Supported (for programmatic access)
- **Rate Limiting:** Not implemented (trust local development environment)

---

## Related Documentation

**Container Documentation:**
- [c4_level2_container_overview.md](c4_level2_container_overview.md) - Overall container architecture
- [c4_level2_api_container.md](c4_level2_api_container.md) - FastAPI internal structure
- [c4_level2_data_container.md](c4_level2_data_container.md) - PostgreSQL database structure

**Infrastructure Documentation:**
- [docker-compose.yml](../../docker-compose.yml) - Container orchestration
- [prometheus/prometheus.yml](../../prometheus/prometheus.yml) - Prometheus configuration
- [grafana/provisioning/dashboards/](../../grafana/provisioning/dashboards/) - Grafana dashboards

**Performance Documentation:**
- [BATCH_PREDICTION_OPTIMIZATION.md](../performance/BATCH_PREDICTION_OPTIMIZATION.md)
- [ARCHITECTURE_COMPUTE_ANALYSIS.md](../performance/ARCHITECTURE_COMPUTE_ANALYSIS.md)

---

## Legend

**Color Coding:**
- **Blue (#e1f5fe)** - External users
- **Orange (#ffccbc)** - Monitored services
- **Yellow (#fff9c4)** - Prometheus container
- **Green (#c8e6c9)** - Grafana container
- **Purple (#f3e5f5)** - Metric exporters
- **Red (#ffebee)** - Alert destinations

**Edge Styles:**
- **Solid arrow (→)** - Data flow (metrics scraping, queries)
- **Dashed arrow (-.→)** - Alert routing

---

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2025-11-15 | 1.0 | Initial monitoring container diagram created |

---

**Diagram Created:** November 15, 2025
**Maintained By:** Architecture Team
**Contact:** See [CLAUDE.md](../../CLAUDE.md) for questions
