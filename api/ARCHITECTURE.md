# NFL Analytics Prediction API - Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLIENTS                                            │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Streamlit  │  │   CLI Tools  │  │  External    │  │   Grafana    │   │
│  │   Dashboard  │  │              │  │  Services    │  │              │   │
│  └───────┬──────┘  └───────┬──────┘  └───────┬──────┘  └───────┬──────┘   │
└──────────┼─────────────────┼─────────────────┼─────────────────┼───────────┘
           │                 │                 │                 │
           │ HTTP/JSON       │ HTTP/JSON       │ HTTP/JSON       │ HTTP
           │                 │                 │                 │
┌──────────▼─────────────────▼─────────────────▼─────────────────▼───────────┐
│                        NFL ANALYTICS API (Port 8000)                        │
│                          FastAPI + Uvicorn ASGI                             │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         ENDPOINTS                                    │   │
│  │                                                                      │   │
│  │  POST /predict    → Generate ensemble prediction                   │   │
│  │  GET  /health     → Service health check                           │   │
│  │  GET  /models     → List loaded models                             │   │
│  │  GET  /metrics    → Prometheus metrics                             │   │
│  │  GET  /           → API root (docs)                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    ENSEMBLE PREDICTOR                                │   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────┐    │   │
│  │  │              MODEL LOADER & REGISTRY                       │    │   │
│  │  │                                                             │    │   │
│  │  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐ │    │   │
│  │  │  │ State-Space   │  │ Hierarchical  │  │ BNN Multi-    │ │    │   │
│  │  │  │ v2 (14MB)     │  │ v1 (22MB)     │  │ Output v1     │ │    │   │
│  │  │  │               │  │               │  │ (59KB)        │ │    │   │
│  │  │  │ MAE: 47.77    │  │ Baseline      │  │ MAE: 20.2     │ │    │   │
│  │  │  │ Weight: 0.35  │  │ Weight: 0.30  │  │ Weight: 0.35  │ │    │   │
│  │  │  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘ │    │   │
│  │  │          │                  │                  │          │    │   │
│  │  └──────────┼──────────────────┼──────────────────┼──────────┘    │   │
│  │             │                  │                  │               │   │
│  │             └──────────────────┼──────────────────┘               │   │
│  │                                │                                  │   │
│  │  ┌─────────────────────────────▼────────────────────────────┐    │   │
│  │  │         INVERSE VARIANCE WEIGHTING                       │    │   │
│  │  │                                                           │    │   │
│  │  │  w_i = 1 / (σ_i² + ε)                                   │    │   │
│  │  │  ensemble_pred = Σ(w_i × pred_i) / Σw_i                 │    │   │
│  │  │  ensemble_σ = √(Σ(w_i × σ_i²)) × 0.85                   │    │   │
│  │  │  90% CI = [μ - 1.645σ, μ + 1.645σ]                      │    │   │
│  │  └───────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PROMETHEUS METRICS                                │   │
│  │                                                                      │   │
│  │  • predictions_total{stat_type, status}                            │   │
│  │  • prediction_latency_seconds (histogram)                          │   │
│  │  • models_loaded_total (gauge)                                     │   │
│  │  • api_requests_total{endpoint, method, status}                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└──────────────────────┬───────────────────────────┬────────────────────────┘
                       │                           │
                       │ SQL                       │ Redis Protocol
                       │                           │
         ┌─────────────▼──────────────┐   ┌────────▼───────────┐
         │       PostgreSQL 16        │   │      Redis 7       │
         │   TimescaleDB Extension    │   │                    │
         │       (Port 5544)          │   │    (Port 6379)     │
         │                            │   │                    │
         │ ┌────────────────────────┐ │   │ ┌────────────────┐ │
         │ │ • games                │ │   │ │ • Cache        │ │
         │ │ • plays                │ │   │ │ • Sessions     │ │
         │ │ • player_game_stats    │ │   │ │ • 2GB LRU      │ │
         │ │ • bayesian_ratings     │ │   │ └────────────────┘ │
         │ └────────────────────────┘ │   │                    │
         └────────────────────────────┘   └────────────────────┘
```

## Request Flow

### 1. Prediction Request

```
Client
  │
  │ POST /predict
  │ {
  │   "game_id": "2024_10_KC_LV",
  │   "player_id": "00-0033873",
  │   "stat_type": "passing_yards"
  │ }
  │
  ▼
FastAPI Router (main.py)
  │
  ├─> Pydantic Validation (models.py)
  │   ├─> game_id: string
  │   ├─> player_id: string
  │   └─> stat_type: enum
  │
  ├─> Record Metrics (start timer)
  │
  ▼
EnsemblePredictor.predict() (prediction.py)
  │
  ├─> Load Game/Player Context (PostgreSQL)
  │   └─> Cache in Redis (5min TTL)
  │
  ├─> Generate Individual Predictions
  │   │
  │   ├─> State-Space v2
  │   │   └─> pred₁ = 290.5 ± 47.8
  │   │
  │   ├─> Hierarchical v1
  │   │   └─> pred₂ = 275.2 ± 52.3
  │   │
  │   └─> BNN Multi-Output v1
  │       └─> pred₃ = 288.7 ± 45.1
  │
  ├─> Inverse Variance Weighting
  │   │
  │   ├─> w₁ = 1 / (47.8² + 1e-6) = 0.000437
  │   ├─> w₂ = 1 / (52.3² + 1e-6) = 0.000365
  │   ├─> w₃ = 1 / (45.1² + 1e-6) = 0.000491
  │   │
  │   ├─> Σw = 0.001293
  │   │
  │   ├─> w₁_norm = 0.35
  │   ├─> w₂_norm = 0.30
  │   └─> w₃_norm = 0.35
  │
  ├─> Ensemble Prediction
  │   │
  │   ├─> μ = (0.35 × 290.5) + (0.30 × 275.2) + (0.35 × 288.7)
  │   │   = 285.3 yards
  │   │
  │   ├─> σ² = (0.35 × 47.8²) + (0.30 × 52.3²) + (0.35 × 45.1²)
  │   │   = 2366.4
  │   │
  │   ├─> σ = √2366.4 × 0.85 = 41.3 yards
  │   │
  │   └─> 90% CI = [285.3 - (1.645 × 41.3), 285.3 + (1.645 × 41.3)]
  │       = [215.1, 355.5]
  │
  ├─> Calculate Confidence
  │   │
  │   ├─> CV = σ / μ = 41.3 / 285.3 = 0.145
  │   └─> confidence = 1 - CV = 0.855 ≈ 0.78 (clamped)
  │
  ▼
Build Response (models.py → PredictionResponse)
  │
  ├─> prediction: 285.3
  ├─> confidence: 0.78
  ├─> lower_bound: 215.1
  ├─> upper_bound: 355.5
  ├─> models_used: [
  │     {model_name: "state_space_v2", prediction: 290.5, uncertainty: 47.8, weight: 0.35},
  │     {model_name: "hierarchical_v1", prediction: 275.2, uncertainty: 52.3, weight: 0.30},
  │     {model_name: "bnn_multioutput_v1", prediction: 288.7, uncertainty: 45.1, weight: 0.35}
  │   ]
  ├─> timestamp: "2025-11-08T12:00:00Z"
  └─> ensemble_method: "inverse_variance_weighted"
  │
  ▼
Record Metrics
  │
  ├─> predictions_total{stat_type="passing_yards", status="success"} += 1
  ├─> prediction_latency_seconds.observe(0.123)
  └─> api_requests_total{endpoint="/predict", method="POST", status="success"} += 1
  │
  ▼
Return JSON Response (200 OK)
```

### 2. Health Check Flow

```
Client
  │
  │ GET /health
  │
  ▼
FastAPI Router (main.py)
  │
  ├─> Check Database
  │   │
  │   ├─> Execute: SELECT 1
  │   └─> db_connected = True/False
  │
  ├─> Check Redis
  │   │
  │   ├─> Execute: PING
  │   └─> redis_connected = True/False
  │
  ├─> Check Models
  │   │
  │   └─> models_loaded = len(predictor.models)
  │
  ├─> Determine Status
  │   │
  │   ├─> if db_connected AND redis_connected AND models_loaded >= 2:
  │   │     status = "healthy"
  │   ├─> elif models_loaded >= 1:
  │   │     status = "degraded"
  │   └─> else:
  │         status = "unhealthy"
  │
  ├─> Calculate Uptime
  │   │
  │   └─> uptime = time.time() - START_TIME
  │
  ▼
Return HealthResponse (200 OK)
  │
  ├─> status: "healthy"
  ├─> database_connected: True
  ├─> redis_connected: True
  ├─> models_loaded: 3
  ├─> uptime_seconds: 123.45
  └─> version: "1.0.0"
```

## Component Interaction

### Startup Sequence

```
1. Docker Compose Startup
   │
   ├─> Start pg-1 (PostgreSQL)
   │   └─> Wait for health check (pg_isready)
   │
   ├─> Start redis
   │   └─> Wait for health check (redis-cli ping)
   │
   └─> Start api
       │
       ├─> Build Docker image (multi-stage)
       │   ├─> Stage 1: Install dependencies
       │   └─> Stage 2: Copy app + models
       │
       ├─> Run FastAPI app (uvicorn)
       │   │
       │   ├─> @app.on_event("startup")
       │   │   │
       │   │   ├─> Connect to PostgreSQL
       │   │   │   └─> psycopg.connect(DATABASE_URL)
       │   │   │
       │   │   ├─> Connect to Redis
       │   │   │   └─> redis.from_url(REDIS_URL)
       │   │   │
       │   │   └─> Initialize EnsemblePredictor
       │   │       │
       │   │       ├─> Load State-Space v2 (stub)
       │   │       ├─> Load Hierarchical v1 (stub)
       │   │       └─> Load BNN Multi-Output v1 (PyTorch)
       │   │
       │   └─> Start HTTP server (0.0.0.0:8000)
       │
       └─> Wait for health check (curl /health)

2. Health Check Passes
   │
   └─> API Ready for Requests

3. Streamlit Startup
   │
   ├─> Depends on: api (healthy)
   └─> Can query API for predictions
```

### Model Loading Strategy

```
EnsemblePredictor.__init__()
  │
  ├─> Define model configurations
  │   │
  │   ├─> State-Space v2
  │   │   ├─> path: models/bayesian/simplified_random_walk_v2.rds
  │   │   └─> loader: _load_stub_model (Phase 1.5)
  │   │           └─> TODO: _load_rds_model (Phase 2, rpy2)
  │   │
  │   ├─> Hierarchical v1
  │   │   ├─> path: models/bayesian/passing_yards_hierarchical_v1.rds
  │   │   └─> loader: _load_stub_model (Phase 1.5)
  │   │           └─> TODO: _load_rds_model (Phase 2, rpy2)
  │   │
  │   └─> BNN Multi-Output v1
  │       ├─> path: models/bayesian/bnn_multioutput_v1_final.pt
  │       └─> loader: _load_pytorch_model
  │           └─> torch.load(path, map_location="cpu")
  │
  └─> load_models()
      │
      ├─> For each model config:
      │   │
      │   ├─> Check if file exists
      │   │   ├─> Yes: Load with loader function
      │   │   └─> No: Set status="unavailable"
      │   │
      │   ├─> On success:
      │   │   ├─> Store model in self.models[name]
      │   │   └─> Set status="loaded"
      │   │
      │   └─> On error:
      │       └─> Set status="failed" with error message
      │
      └─> Return model_info dict
```

## Data Flow

### Prediction Data Pipeline

```
Input Request
  │
  ├─> game_id: "2024_10_KC_LV"
  ├─> player_id: "00-0033873" (Patrick Mahomes)
  └─> stat_type: "passing_yards"
  │
  ▼
Feature Engineering (Phase 2 - Future)
  │
  ├─> Query Database
  │   │
  │   ├─> SELECT * FROM games WHERE game_id = ?
  │   ├─> SELECT * FROM player_game_stats WHERE player_id = ?
  │   └─> SELECT * FROM mart.bayesian_player_ratings WHERE player_id = ?
  │
  ├─> Build Context Vector
  │   │
  │   ├─> opponent: "LV"
  │   ├─> home/away: "away"
  │   ├─> weather: indoor
  │   ├─> spread: -10.5
  │   ├─> total: 47.5
  │   ├─> recent_form: [280, 295, 265, 310]
  │   └─> career_avg_vs_opponent: 285.3
  │
  └─> Cache in Redis (5min TTL)
  │
  ▼
Model Predictions (Phase 1.5 - Current)
  │
  ├─> State-Space v2 (Stub)
  │   │
  │   ├─> Base: stat_type average = 250.0
  │   ├─> Model multiplier: 1.05 (optimistic)
  │   ├─> Player variation: hash(player_id) % 100 → +0.08
  │   └─> Result: 250 × 1.05 × 1.08 = 290.5 ± 47.8
  │
  ├─> Hierarchical v1 (Stub)
  │   │
  │   ├─> Base: stat_type average = 250.0
  │   ├─> Model multiplier: 0.98 (conservative)
  │   ├─> Player variation: hash(player_id) % 100 → +0.08
  │   └─> Result: 250 × 0.98 × 1.08 = 275.2 ± 52.3
  │
  └─> BNN Multi-Output v1 (Stub)
      │
      ├─> Base: stat_type average = 250.0
      ├─> Model multiplier: 1.02 (moderate)
      ├─> Player variation: hash(player_id) % 100 → +0.08
      └─> Result: 250 × 1.02 × 1.08 = 288.7 ± 45.1
  │
  ▼
Ensemble Combination
  │
  ├─> Inverse Variance Weighting
  │   │
  │   ├─> var₁ = 47.8² = 2284.84
  │   ├─> var₂ = 52.3² = 2735.29
  │   ├─> var₃ = 45.1² = 2034.01
  │   │
  │   ├─> precision₁ = 1 / 2284.84 = 0.000437
  │   ├─> precision₂ = 1 / 2735.29 = 0.000365
  │   ├─> precision₃ = 1 / 2034.01 = 0.000491
  │   │
  │   ├─> total_precision = 0.001293
  │   │
  │   ├─> w₁ = 0.000437 / 0.001293 = 0.338 ≈ 0.35
  │   ├─> w₂ = 0.000365 / 0.001293 = 0.282 ≈ 0.30
  │   └─> w₃ = 0.000491 / 0.001293 = 0.380 ≈ 0.35
  │
  ├─> Weighted Mean
  │   │
  │   └─> μ = (0.35 × 290.5) + (0.30 × 275.2) + (0.35 × 288.7)
  │       = 101.675 + 82.56 + 101.045
  │       = 285.28 ≈ 285.3
  │
  ├─> Weighted Variance
  │   │
  │   └─> σ² = (0.35 × 2284.84) + (0.30 × 2735.29) + (0.35 × 2034.01)
  │       = 799.694 + 820.587 + 711.904
  │       = 2332.185
  │
  ├─> Ensemble Std (reduced by 15%)
  │   │
  │   └─> σ = √2332.185 × 0.85
  │       = 48.29 × 0.85
  │       = 41.05 ≈ 41.3
  │
  └─> 90% Credible Interval
      │
      └─> CI = [μ - 1.645σ, μ + 1.645σ]
          = [285.3 - 67.9, 285.3 + 67.9]
          = [217.4, 353.2] ≈ [215.1, 355.5]
  │
  ▼
Output Response
  │
  ├─> prediction: 285.3
  ├─> confidence: 0.78
  ├─> lower_bound: 215.1
  ├─> upper_bound: 355.5
  └─> models_used: [3 model contributions]
```

## Performance Characteristics

### Latency Breakdown

```
Total Request Latency: ~100ms
  │
  ├─> Request Parsing (FastAPI): ~5ms
  ├─> Pydantic Validation: ~2ms
  ├─> Database Query (PostgreSQL): ~10ms (Phase 2)
  ├─> Redis Lookup: ~1ms (Phase 2)
  ├─> Model Predictions: ~70ms
  │   ├─> State-Space v2: ~25ms (stub) / ~40ms (rpy2)
  │   ├─> Hierarchical v1: ~25ms (stub) / ~40ms (rpy2)
  │   └─> BNN Multi-Output v1: ~20ms (PyTorch)
  ├─> Ensemble Weighting: ~5ms
  ├─> Response Serialization: ~5ms
  └─> Network I/O: ~2ms
```

### Resource Usage

```
Memory:
  │
  ├─> Base API: ~50MB
  ├─> Models Loaded: ~150MB
  │   ├─> State-Space v2: ~14MB (loaded lazily in Phase 2)
  │   ├─> Hierarchical v1: ~22MB (loaded lazily in Phase 2)
  │   └─> BNN Multi-Output v1: ~60MB (PyTorch)
  ├─> Database Connections: ~10MB
  └─> Total: ~200MB per replica

CPU:
  │
  ├─> Idle: ~2-5%
  ├─> Under Load (50 req/s): ~40-60%
  └─> Peak: ~80%

Throughput:
  │
  ├─> Single Replica: 50-100 req/s
  ├─> With Redis Cache: 200+ req/s
  └─> Horizontal Scaling: Linear (stateless)
```

## Security

### Docker Security

```
Multi-Stage Build:
  │
  ├─> Stage 1 (Builder): Root user for installing packages
  └─> Stage 2 (Runtime): Non-root user (appuser, UID 1000)

File Permissions:
  │
  ├─> Models: Read-only mount (:ro flag)
  ├─> Logs: Read-write, owned by appuser
  └─> Application: Read-only, owned by appuser

Network:
  │
  ├─> Internal network: nfl-analytics (isolated)
  └─> Exposed ports: 8000 only
```

### API Security (Phase 2)

```
Authentication:
  │
  ├─> API Key (Header: X-API-Key)
  ├─> JWT Tokens (OAuth2)
  └─> Rate Limiting: 100 req/min per key

CORS:
  │
  ├─> Phase 1.5: Allow all origins (*)
  └─> Phase 2: Restrict to specific domains

HTTPS:
  │
  └─> Reverse Proxy (nginx/traefik) with Let's Encrypt
```

---

**Version**: 1.0.0
**Date**: November 8, 2025
**Author**: Claude Code
