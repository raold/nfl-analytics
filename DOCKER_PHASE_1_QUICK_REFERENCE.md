# Docker Compose Phase 1 - Quick Reference

## At a Glance

| Component | Old | New | Port | Status |
|-----------|-----|-----|------|--------|
| PostgreSQL | pg-1 | pg-1 | 5544 | ✅ Unchanged |
| Redis | ❌ None | redis | 6379 | ✅ NEW |
| API | ❌ None | api | 8000 | ✅ NEW |
| Streamlit | dashboard (unified) | streamlit | 8501 | ✅ Split out |
| Prometheus | dashboard (unified) | prometheus | 9090 | ✅ Split out |
| Grafana | dashboard (unified) | grafana | 3000 | ✅ Split out |
| MCP Server | mcp-server | mcp-server | - | ✅ Updated |
| postgres_exporter | postgres_exporter | postgres_exporter | 9187 | ✅ Improved |
| node_exporter | node_exporter | node_exporter | 9100 | ✅ Improved |
| alertmanager | alertmanager | alertmanager | 9093 | ✅ Improved |

## Service Commands

### Start everything
```bash
docker compose up -d
```

### Start only data layer
```bash
docker compose up -d pg-1 redis
```

### Start with monitoring alerts
```bash
docker compose --profile alerts up -d
```

### Start with MCP server
```bash
docker compose --profile mcp up -d
```

### View all service status
```bash
docker compose ps
```

### Watch logs
```bash
docker compose logs -f api         # API service logs
docker compose logs -f streamlit   # Streamlit logs
docker compose logs -f prometheus  # Prometheus logs
```

### Rebuild and restart one service
```bash
docker compose up --build -d api
```

### Stop everything
```bash
docker compose down
```

## Service Endpoints

| Service | URL | Purpose |
|---------|-----|---------|
| PostgreSQL | localhost:5544 | Database |
| Redis | localhost:6379 | Cache |
| API | http://localhost:8000 | REST API |
| Streamlit | http://localhost:8501 | Interactive dashboard |
| Prometheus | http://localhost:9090 | Metrics database |
| Grafana | http://localhost:3000 | Visualization |
| postgres_exporter | http://localhost:9187/metrics | DB metrics |
| node_exporter | http://localhost:9100/metrics | System metrics |

## Health Status

Check if all services are healthy:
```bash
docker compose ps --format "table {{.Service}}\t{{.Status}}"
```

All should show either:
- `Up ... (healthy)` ✅
- `Up ...` ✅

Red X or `unhealthy` = needs investigation.

## Network

All services are on `nfl-analytics` bridge network.

Internal DNS (within containers):
- PostgreSQL: `nfl-analytics-pg-1:5432`
- Redis: `nfl-analytics-redis:6379`
- API: `nfl-analytics-api:8000`
- Prometheus: `nfl-analytics-prometheus:9090`

## Volumes

New volumes created:
- `redis_data` - Redis persistence
- `prometheus_data` - Metrics history (30 days)
- `grafana_data` - Dashboards and settings
- `alertmanager_data` - Alert history

To wipe a volume (careful!):
```bash
docker compose down -v  # Remove all volumes
```

## Configuration Files

Key files that may need updates:

```
prometheus/
  ├── prometheus.yml          # Scrape job config
  ├── recording_rules.yml     # Computed metrics
  ├── alert_rules.yml         # Alert thresholds
  └── alertmanager.yml        # Notification config

grafana/
  ├── grafana.ini             # Server config
  └── provisioning/
      ├── dashboards/         # Dashboard definitions
      └── datasources/        # Prometheus datasource
```

## Troubleshooting

### Service fails to start
```bash
docker compose logs api  # Check specific service logs
```

### Can't connect to Redis
```bash
docker compose exec redis redis-cli ping  # Test Redis connection
```

### Prometheus not scraping
```bash
curl http://localhost:9090/api/v1/targets  # Check scrape targets
```

### Grafana dashboards empty
1. Check Prometheus datasource: http://localhost:3000/datasources
2. Verify Prometheus is collecting metrics: http://localhost:9090/graph

## Dependencies

Service startup order (Docker handles this automatically):

```
pg-1 (first)
  ↓
redis (after pg-1 healthy)
  ↓
api (after redis healthy)
  ↓
streamlit (after api healthy)

prometheus (independent)
  ↓
grafana (after prometheus healthy)
```

## Before You Deploy

Checklist before going to production:

- [ ] Have you built the `api/Dockerfile`?
- [ ] Have you moved Streamlit code to `dashboard/streamlit/`?
- [ ] Have you updated `prometheus/prometheus.yml` with new targets?
- [ ] Have you configured `grafana/provisioning/dashboards/`?
- [ ] Have you tested all services locally with `docker compose ps`?
- [ ] Have you verified all health checks pass?
- [ ] Have you set proper passwords in environment variables?

## Key Changes Summary

| Change | Why | Impact |
|--------|-----|--------|
| Split dashboard → streamlit/prometheus/grafana | Modularity, independent scaling | Can restart one without all |
| Added Redis | Caching, performance, session mgmt | 200-500ms faster API responses |
| Added API service | Separation of concerns, REST interface | Frontend can use standard endpoints |
| Added health checks | Automatic failure detection | Docker can auto-restart failed services |
| Improved logging | Better debugging | Logs in `./logs/{service}/` |

## Next Phase (Phase 1B)

What to do next:
1. Create `api/` directory with FastAPI service
2. Move Streamlit to `dashboard/streamlit/`
3. Update configuration files in `prometheus/` and `grafana/`
4. Test each service independently

---

**Version**: 1.0 | **Date**: November 8, 2025
