# C4 Level 2: API Container Detail

**Last Updated:** November 15, 2025
**Diagram Type:** C4 Container (Level 2)
**Container:** FastAPI Prediction Service
**Status:** Production

---

## Overview

The FastAPI container is the core prediction service, responsible for loading ensemble models, generating predictions with uncertainty quantification, and exposing a production REST API with authentication and rate limiting.

**Container Specifications:**
- **Name:** nfl-analytics-api
- **Image:** Custom (built from `containers/Dockerfile.api`)
- **Port:** 8000 (host) → 8000 (container)
- **Language:** Python 3.9+
- **Framework:** FastAPI 0.100+

**Key Responsibilities:**
- Load 4-component ensemble (State-Space v2, Hierarchical v2, XGBoost v4, Informative Priors v2)
- Extract features from PostgreSQL or Redis cache
- Generate predictions with confidence intervals
- Authenticate API requests (API key validation)
- Enforce rate limiting (Redis-backed sliding window)
- Expose Prometheus metrics

---

## API Container Diagram

```mermaid
graph TB
    subgraph External["External Clients"]
        USER["End User\n(Sports Bettor)"]
        PROM_SCRAPER["Prometheus\n(Metrics Scraper)"]
    end

    subgraph FastAPI_Container["FastAPI Container (nfl-analytics-api)"]

        subgraph Entry_Points["Entry Points"]
            ROUTER["Main Router\n(api/main.py)\n\n• /health\n• /predict\n• /models\n• /metrics"]
        end

        subgraph Middleware["Middleware Stack"]
            CORS["CORS Middleware\n(Allow origins)"]
            AUTH["API Key Auth\n(api/auth.py)\nX-API-Key validation"]
            RATE_LIMIT["Rate Limiter\n(api/middleware.py)\nRedis sliding window"]
            ERROR_HANDLER["Error Handler\n(HTTP exceptions)"]
        end

        subgraph Core_Logic["Core Prediction Logic"]
            PREDICTOR["EnsemblePredictor\n(api/prediction.py)\n\n• Load models\n• Feature extraction\n• Parallel prediction\n• Aggregation"]

            FEATURE_EXT["Feature Extractor\n(py/features/)\n\n• Cache lookup\n• Database query\n• 40+ features"]

            ENSEMBLE["Ensemble Manager\n(py/ensemble/enhanced_ensemble_v5.py)\n\n• State-Space v2 (30%)\n• Hierarchical v2 (30%)\n• XGBoost v4 (25%)\n• Informative Priors v2 (15%)"]

            TEMP_SCALER["Temperature Scaler\n(py/calibration/)\n\nT=0.5534\nIsotonic regression"]
        end

        subgraph Model_Loading["Model Loading"]
            R_BRIDGE["R Bridge\n(rpy2)\n\nLoad R models\nCall R functions"]

            STATE_SPACE["State-Space v2\n(R/Stan)\nstate_space_v2.rds"]
            HIERARCHICAL["Hierarchical v2\n(R/brms)\nhierarchical_v2.rds"]
            IP["Informative Priors v2\n(R/brms)\ninformative_priors_v2.rds"]

            XGB["XGBoost v4\n(Python)\nxgboost_v4.pkl"]
            SCALER["Temperature Scaler v6\n(Python)\ntemperature_scaler_v6.pkl"]
        end

        subgraph Data_Clients["Data Access Clients"]
            PG_CLIENT["PostgreSQL Client\n(psycopg3)\n\nConnection pool\n2-10 connections"]

            REDIS_CLIENT["Redis Client\n(redis-py)\n\nFeature cache\nRate limit store"]
        end

        subgraph Metrics["Prometheus Metrics"]
            PRED_COUNTER["predictions_total\n(Counter)\nBy stat_type, status"]
            PRED_LATENCY["prediction_latency_seconds\n(Histogram)\nBuckets: 0.01-5.0s"]
            MODELS_GAUGE["models_loaded_total\n(Gauge)\nCurrent count"]
            API_COUNTER["api_requests_total\n(Counter)\nBy endpoint, method, status"]
        end

    end

    subgraph External_Services["External Services"]
        PG["PostgreSQL\n(nfl-analytics-pg-1)"]
        REDIS["Redis\n(nfl-analytics-redis)"]
        MODELS_VOL["models/ Volume\n(Read-only mount)"]
    end

    %% External to Entry
    USER -->|"POST /predict\nX-API-Key header"| ROUTER
    PROM_SCRAPER -->|"GET /metrics\n(15s interval)"| ROUTER

    %% Entry to Middleware
    ROUTER --> CORS
    CORS --> AUTH
    AUTH --> RATE_LIMIT
    RATE_LIMIT --> ERROR_HANDLER

    %% Middleware to Core Logic
    ERROR_HANDLER -->|"/predict request"| PREDICTOR

    %% Core Prediction Flow
    PREDICTOR --> FEATURE_EXT
    FEATURE_EXT -->|"Cache miss"| PG_CLIENT
    FEATURE_EXT -->|"Cache hit"| REDIS_CLIENT
    FEATURE_EXT --> ENSEMBLE
    ENSEMBLE -->|"Parallel calls"| STATE_SPACE
    ENSEMBLE -->|"Parallel calls"| HIERARCHICAL
    ENSEMBLE -->|"Parallel calls"| XGB
    ENSEMBLE -->|"Parallel calls"| IP
    STATE_SPACE --> TEMP_SCALER
    HIERARCHICAL --> TEMP_SCALER
    XGB --> TEMP_SCALER
    IP --> TEMP_SCALER
    TEMP_SCALER --> PREDICTOR

    %% Model Loading
    STATE_SPACE -.->|"Load from file"| R_BRIDGE
    HIERARCHICAL -.->|"Load from file"| R_BRIDGE
    IP -.->|"Load from file"| R_BRIDGE
    R_BRIDGE -.->|"Read .rds files"| MODELS_VOL
    XGB -.->|"Load from file"| MODELS_VOL
    SCALER -.->|"Load from file"| MODELS_VOL

    %% Data Access
    PG_CLIENT -->|"PostgreSQL wire\nprotocol"| PG
    REDIS_CLIENT -->|"Redis protocol\nGET/SET"| REDIS

    %% Metrics Collection
    PREDICTOR -.->|"Record metrics"| PRED_COUNTER
    PREDICTOR -.->|"Record metrics"| PRED_LATENCY
    PREDICTOR -.->|"Record metrics"| MODELS_GAUGE
    ROUTER -.->|"Record metrics"| API_COUNTER
    ROUTER -->|"/metrics response"| PROM_SCRAPER

    %% Styling
    classDef external fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef entry fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    classDef middleware fill:#ffccbc,stroke:#bf360c,stroke-width:2px
    classDef core fill:#f3e5f5,stroke:#4a148c,stroke-width:3px
    classDef models fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
    classDef data fill:#b3e5fc,stroke:#01579b,stroke-width:2px
    classDef metrics fill:#d1c4e9,stroke:#311b92,stroke-width:2px
    classDef services fill:#ffebee,stroke:#b71c1c,stroke-width:2px

    class USER,PROM_SCRAPER external
    class ROUTER entry
    class CORS,AUTH,RATE_LIMIT,ERROR_HANDLER middleware
    class PREDICTOR,FEATURE_EXT,ENSEMBLE,TEMP_SCALER core
    class R_BRIDGE,STATE_SPACE,HIERARCHICAL,IP,XGB,SCALER models
    class PG_CLIENT,REDIS_CLIENT data
    class PRED_COUNTER,PRED_LATENCY,MODELS_GAUGE,API_COUNTER metrics
    class PG,REDIS,MODELS_VOL services
```

---

## Internal Components

### Entry Points (api/main.py)

**Main Router:**
- **File:** `api/main.py` (420 lines)
- **Framework:** FastAPI with dependency injection
- **Responsibilities:**
  - Route incoming HTTP requests
  - Coordinate middleware execution
  - Manage application lifecycle (startup/shutdown)

**Endpoints:**

| Endpoint | Method | Auth | Rate Limited | Purpose |
|----------|--------|------|--------------|---------|
| `/health` | GET | No | No | Health check (database, Redis, models loaded) |
| `/predict` | POST | Yes | Yes | Generate ensemble prediction |
| `/models` | GET | Yes | Yes | List loaded models with metadata |
| `/metrics` | GET | No | No | Prometheus metrics (Prometheus format) |
| `/docs` | GET | No | No | Swagger UI (interactive API documentation) |
| `/` | GET | No | No | API root (redirect to /docs) |

**Request/Response Models (Pydantic):**
```python
# Request
class PredictionRequest(BaseModel):
    game_id: str  # "2024_10_KC_LV"
    player_id: str  # "00-0033873"
    stat_type: StatType  # Enum: passing_yards, rushing_yards, receiving_yards

# Response
class PredictionResponse(BaseModel):
    prediction: float  # Ensemble mean (e.g., 265.3 yards)
    uncertainty: float  # Ensemble std (e.g., 52.1 yards)
    confidence_intervals: Dict[str, Tuple[float, float]]  # 10%, 50%, 90%
    components: Dict[str, float]  # Individual model predictions
    timestamp: datetime  # UTC timestamp
```

---

### Middleware Stack

**1. CORS Middleware:**
- **Purpose:** Enable cross-origin requests (for Streamlit dashboard)
- **Configuration:**
  - `allow_origins`: ["*"] (development), specific domains (production)
  - `allow_credentials`: True
  - `allow_methods`: ["*"]
  - `allow_headers`: ["*"]

**2. API Key Authentication (api/auth.py):**
- **File:** `api/auth.py` (200 lines)
- **Method:** X-API-Key header validation
- **Storage:** PostgreSQL (api_keys table, hashed with bcrypt)
- **Validation Logic:**
  1. Extract X-API-Key from request header
  2. Hash API key (bcrypt)
  3. Query database: `SELECT * FROM api_keys WHERE key_hash = $1`
  4. Check if key is active and not expired
  5. Return APIKey object or raise 401 Unauthorized

**Example:**
```python
from fastapi import Depends, HTTPException, Header

async def require_api_key(x_api_key: str = Header(...)):
    """Validate API key from X-API-Key header."""
    authenticator = get_authenticator()
    api_key = authenticator.validate_key(x_api_key)

    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return api_key
```

**3. Rate Limiting (api/middleware.py):**
- **File:** `api/middleware.py` (150 lines)
- **Implementation:** Redis-backed sliding window
- **Default Limit:** 100 requests/hour per API key
- **Algorithm:**
  1. Generate key: `ratelimit:{api_key}:{current_hour}`
  2. Increment counter in Redis: `INCR ratelimit:{api_key}:{current_hour}`
  3. Set TTL if new key: `EXPIRE ratelimit:{api_key}:{current_hour} 3600`
  4. Check count: If > limit, return 429 Too Many Requests
  5. Add headers: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset

**Response Headers:**
```
HTTP/1.1 200 OK
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 73
X-RateLimit-Reset: 1731672000
```

**4. Error Handler:**
- **Purpose:** Standardize error responses
- **Handles:**
  - 401 Unauthorized (invalid API key)
  - 403 Forbidden (inactive API key)
  - 422 Unprocessable Entity (invalid request body)
  - 429 Too Many Requests (rate limit exceeded)
  - 500 Internal Server Error (prediction failure)
  - 503 Service Unavailable (models not loaded)

---

### Core Prediction Logic

**1. EnsemblePredictor (api/prediction.py):**
- **File:** `api/prediction.py` (400 lines)
- **Responsibilities:**
  - Load ensemble models on startup
  - Coordinate feature extraction
  - Execute parallel predictions
  - Aggregate results
  - Apply temperature scaling

**Initialization:**
```python
class EnsemblePredictor:
    def __init__(self, models_dir: str, device: str = "cpu"):
        self.models_dir = models_dir
        self.device = device
        self.ensemble = None  # Will load on first call

    def load_models(self) -> Dict[str, ModelInfo]:
        """Load all ensemble models from disk."""
        self.ensemble = EnhancedEnsemble(models_dir=self.models_dir)
        model_info = self.ensemble.get_model_info()
        return model_info

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Generate ensemble prediction."""
        # 1. Extract features
        features = self._extract_features(request)

        # 2. Parallel prediction
        predictions = self.ensemble.predict(features)

        # 3. Aggregate
        ensemble_mean, ensemble_std = self._aggregate(predictions)

        # 4. Temperature scaling
        calibrated_std = ensemble_std * TEMPERATURE

        # 5. Compute confidence intervals
        intervals = self._compute_intervals(ensemble_mean, calibrated_std)

        return PredictionResponse(...)
```

**2. Feature Extractor (py/features/):**
- **File:** `py/features/player_feature_extractor.py` (600 lines)
- **Features Generated:** 40+ features
  - Rolling averages (last 3, 5 games, season)
  - Opponent defense (EPA allowed, yards allowed)
  - Game script (spread, total, implied team total)
  - Environmental (home/away, weather if available)
  - Usage (attempts, targets, carries - prop-specific)

**Caching Strategy:**
```python
def extract_features(player_id: str, season: int, week: int) -> np.ndarray:
    """Extract features with caching."""
    # 1. Check Redis cache
    cache_key = f"features:{player_id}:{season}:{week}"
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # 2. Query PostgreSQL
    features = _query_database(player_id, season, week)

    # 3. Write to cache (TTL=7200s)
    redis_client.setex(cache_key, 7200, json.dumps(features))

    return features
```

**3. Ensemble Manager (py/ensemble/enhanced_ensemble_v5.py):**
- **File:** `py/ensemble/enhanced_ensemble_v5.py` (1,200 lines)
- **Architecture:** 4-way ensemble with fixed weights
  - State-Space v2: 30% (R/Stan, dynamic Kalman filter)
  - Hierarchical v2: 30% (R/brms, position-level hierarchy)
  - XGBoost v4: 25% (Python, Ridge + residual correction)
  - Informative Priors v2: 15% (R/brms, position-specific priors)

**Prediction Flow:**
```python
def predict(self, features: np.ndarray) -> Dict[str, float]:
    """Execute 4-way ensemble prediction."""
    # Parallel execution (all 4 models simultaneously)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            'state_space_v2': executor.submit(self.state_space.predict, features),
            'hierarchical_v2': executor.submit(self.hierarchical.predict, features),
            'xgboost_v4': executor.submit(self.xgboost.predict, features),
            'informative_priors_v2': executor.submit(self.ip.predict, features),
        }

        results = {k: f.result() for k, f in futures.items()}

    # Weighted aggregation
    weights = [0.30, 0.30, 0.25, 0.15]
    ensemble_mean = sum(w * r['mean'] for w, r in zip(weights, results.values()))
    ensemble_std = np.sqrt(sum(w**2 * r['std']**2 for w, r in zip(weights, results.values())))

    return {
        'mean': ensemble_mean,
        'std': ensemble_std,
        'components': {k: v['mean'] for k, v in results.items()}
    }
```

**4. Temperature Scaler (py/calibration/temperature_scaling.py):**
- **File:** `py/calibration/temperature_scaling.py` (347 lines)
- **Method:** Isotonic regression (sklearn)
- **Optimal Temperature:** T = 0.5534 (calibrated on 2022-2024 data)
- **Purpose:** Adjust prediction intervals to achieve 90% calibration coverage

**Calibration:**
```python
def calibrate(ensemble_std: float) -> float:
    """Apply temperature scaling."""
    calibrated_std = ensemble_std * TEMPERATURE  # 0.5534
    return calibrated_std
```

**Result:** 97.3% coverage → 89.7% coverage (closer to 90% target)

---

### Model Loading

**R Bridge (rpy2):**
- **Purpose:** Load and call R models from Python
- **Challenges:**
  - Python ↔ R data type conversion (numpy array ↔ R data.frame)
  - R environment initialization (library loading, memory management)
  - Error handling (R errors → Python exceptions)

**Initialization:**
```python
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()

# Load R libraries
ro.r('library(brms)')
ro.r('library(rstan)')

# Load models
state_space_model = ro.r['readRDS']('models/state_space_v2.rds')
hierarchical_model = ro.r['readRDS']('models/hierarchical_v2.rds')
ip_model = ro.r['readRDS']('models/informative_priors_v2.rds')
```

**Prediction (R → Python):**
```python
def predict_brms(model, features: pd.DataFrame) -> Dict[str, float]:
    """Call R brms predict function from Python."""
    # Convert pandas DataFrame to R data.frame
    r_features = pandas2ri.py2rpy(features)

    # Call R posterior_predict function
    r_predictions = ro.r['posterior_predict'](model, newdata=r_features, ndraws=1000)

    # Convert R matrix to numpy array
    predictions = np.array(r_predictions)

    # Compute mean and std
    mean = predictions.mean()
    std = predictions.std()

    return {'mean': mean, 'std': std}
```

**Model Files:**

| Model | File | Size | Format | Loading Time |
|-------|------|------|--------|--------------|
| **State-Space v2** | state_space_v2.rds | ~200MB | R RDS (Stan compiled) | 2-3s |
| **Hierarchical v2** | hierarchical_v2.rds | ~500MB | R RDS (brms compiled) | 5-7s |
| **Informative Priors v2** | informative_priors_v2.rds | ~400MB | R RDS (brms compiled) | 4-6s |
| **XGBoost v4** | xgboost_v4.pkl | ~50MB | Python pickle | 0.5s |
| **Temperature Scaler v6** | temperature_scaler_v6.pkl | ~10MB | Python pickle | 0.1s |

**Total Loading Time:** ~12-17 seconds (one-time cost at startup)

---

### Data Access Clients

**1. PostgreSQL Client (psycopg3):**
- **File:** `py/utils/database_pool.py` (373 lines)
- **Configuration:**
  - Min connections: 2
  - Max connections: 10
  - Connection lifetime: 3600s (1 hour)
  - Idle timeout: 300s (5 minutes)

**Connection Pooling:**
```python
from psycopg_pool import ConnectionPool

pool = ConnectionPool(
    conninfo="postgresql://dro:sicillionbillions@nfl-analytics-pg-1:5432/devdb01",  # pragma: allowlist secret
    min_size=2,
    max_size=10,
    timeout=30,
)

# Query execution
with pool.connection() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM mart.player_game_stats WHERE player_id = %s", (player_id,))
        result = cur.fetchall()
```

**Performance:** 50-100ms (unoptimized) → <1ms (pooled)

**2. Redis Client (redis-py):**
- **File:** `py/utils/feature_cache.py` (373 lines)
- **Configuration:**
  - Host: nfl-analytics-redis
  - Port: 6379
  - Password: ${REDIS_PASSWORD}
  - Decode responses: True (return strings not bytes)

**Caching Operations:**
```python
import redis

redis_client = redis.Redis(
    host='nfl-analytics-redis',
    port=6379,
    password=os.getenv('REDIS_PASSWORD'),
    decode_responses=True
)

# Feature cache
def get_features(player_id: str, season: int, week: int) -> Optional[Dict]:
    key = f"features:{player_id}:{season}:{week}"
    cached = redis_client.get(key)
    if cached:
        return json.loads(cached)
    return None

def set_features(player_id: str, season: int, week: int, features: Dict):
    key = f"features:{player_id}:{season}:{week}"
    redis_client.setex(key, 7200, json.dumps(features))  # TTL=7200s (2 hours)
```

---

### Prometheus Metrics

**Metrics Exported:**

**1. predictions_total (Counter):**
```
# HELP predictions_total Total number of predictions made
# TYPE predictions_total counter
predictions_total{stat_type="passing_yards",status="success"} 1234
predictions_total{stat_type="rushing_yards",status="success"} 567
predictions_total{stat_type="receiving_yards",status="error"} 12
```

**2. prediction_latency_seconds (Histogram):**
```
# HELP prediction_latency_seconds Time to generate prediction
# TYPE prediction_latency_seconds histogram
prediction_latency_seconds_bucket{le="0.01"} 0
prediction_latency_seconds_bucket{le="0.05"} 50
prediction_latency_seconds_bucket{le="0.1"} 150
prediction_latency_seconds_bucket{le="0.5"} 900
prediction_latency_seconds_bucket{le="1.0"} 1200
prediction_latency_seconds_bucket{le="2.0"} 1230
prediction_latency_seconds_bucket{le="5.0"} 1234
prediction_latency_seconds_sum 450.5
prediction_latency_seconds_count 1234
```

**3. models_loaded_total (Gauge):**
```
# HELP models_loaded_total Number of models successfully loaded
# TYPE models_loaded_total gauge
models_loaded_total 4
```

**4. api_requests_total (Counter):**
```
# HELP api_requests_total Total API requests
# TYPE api_requests_total counter
api_requests_total{endpoint="/predict",method="POST",status="success"} 1234
api_requests_total{endpoint="/models",method="GET",status="success"} 56
api_requests_total{endpoint="/health",method="GET",status="success"} 7890
api_requests_total{endpoint="/predict",method="POST",status="error"} 12
```

---

## Performance Characteristics

### Latency Breakdown (POST /predict)

| Phase | Latency | Percentage |
|-------|---------|------------|
| **Request Validation** | 10ms | 3% |
| **API Key Validation** | 5ms | 1% |
| **Rate Limit Check** | 3ms | 1% |
| **Feature Extraction** | 50-150ms | 15-40% |
| **Parallel Prediction** | 200-300ms | 50-75% |
| **Aggregation** | 10ms | 3% |
| **Temperature Scaling** | 5ms | 1% |
| **JSON Serialization** | 5ms | 1% |
| **Total** | **280-480ms** | **100%** |

**Bottleneck:** Hierarchical v2 prediction (200ms, R/brms posterior_predict)

**Target:** <500ms (currently meeting SLA ✅)

---

### Throughput

**Current (Development):**
- **Predictions per second:** ~2 (limited by 280-480ms latency)
- **Predictions per hour:** ~7,200 (theoretical max with single instance)
- **Actual usage:** ~100 predictions/hour (manual usage)

**Future (Production with Auto-Scaling):**
- **Instances:** 2-10 (behind load balancer)
- **Predictions per second:** ~4-20 (2-10 instances × 2 pred/sec)
- **Predictions per hour:** ~14,400-72,000

---

### Resource Usage

**CPU:**
- **Idle:** ~0.5 cores (health checks, monitoring)
- **Per Prediction:** ~1-2 cores (parallel model execution)
- **Peak:** ~3 cores (if 2-3 concurrent requests)

**Memory:**
- **Baseline:** ~500MB (Python runtime + dependencies)
- **Model Loading:** +1-1.5GB (4 models loaded into memory)
- **Per Prediction:** +50-100MB (temporary arrays, R environment)
- **Total:** ~1.5-2GB typical, 2.5GB peak

**Disk I/O:**
- **Model Loading (startup):** ~1.5GB read (one-time)
- **Logging:** <10MB/day (INFO level)

---

## Error Handling

### Error Scenarios

**1. Model Loading Failure:**
- **Symptom:** R model fails to load (corrupted .rds file, missing dependency)
- **Detection:** Startup checks (`EnsemblePredictor.load_models()`)
- **Response:** Log error, set `is_ready = False`, return 503 Service Unavailable
- **Recovery:** Fix model file, restart container

**2. Feature Extraction Failure:**
- **Symptom:** PostgreSQL query fails (table doesn't exist, network error)
- **Detection:** Exception in `_extract_features()`
- **Response:** Retry 3x, fall back to league averages (if available)
- **Fallback Values:**
  - QB passing: 250 yards (league avg)
  - RB rushing: 60 yards (league avg)
  - WR receiving: 50 yards (league avg)

**3. Prediction Timeout:**
- **Symptom:** R model prediction takes >10 seconds
- **Detection:** ThreadPoolExecutor timeout
- **Response:** Cancel prediction, log error, return 500 Internal Server Error
- **Mitigation:** Set timeout (default 10s) for each model prediction

**4. Redis Unavailable:**
- **Symptom:** Redis connection fails (service down, network error)
- **Detection:** redis.exceptions.ConnectionError
- **Response:** Graceful degradation (skip cache, query PostgreSQL directly)
- **Impact:** Higher latency (50-150ms → 150-300ms per prediction)

**5. Rate Limit Exceeded:**
- **Symptom:** API key exceeds 100 requests/hour
- **Detection:** Redis counter > limit
- **Response:** Return 429 Too Many Requests with Retry-After header
- **Client Action:** Wait until next hour (X-RateLimit-Reset timestamp)

---

## Security Considerations

### Authentication

**API Key Validation Flow:**
```
1. Client sends request with X-API-Key header
2. Middleware extracts header value
3. Hash API key with bcrypt
4. Query database: SELECT * FROM api_keys WHERE key_hash = $1
5. Check:
   - Key exists?
   - Key is active (not revoked)?
   - Key not expired?
6. If valid: Attach APIKey object to request context
7. If invalid: Return 401 Unauthorized
```

**API Key Storage:**
```sql
CREATE TABLE api_keys (
    id SERIAL PRIMARY KEY,
    key_name VARCHAR(255) NOT NULL,  -- Human-readable name
    key_hash VARCHAR(255) NOT NULL,  -- bcrypt hash
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,
    UNIQUE(key_hash)
);
```

**Key Generation:**
```python
import secrets
import bcrypt

def generate_api_key() -> Tuple[str, str]:
    """Generate API key and hash."""
    key = secrets.token_urlsafe(32)  # 32-byte random token
    key_hash = bcrypt.hashpw(key.encode(), bcrypt.gensalt()).decode()
    return key, key_hash
```

---

### Rate Limiting

**Redis Sliding Window:**
```python
def check_rate_limit(api_key: str, limit: int = 100, window: int = 3600) -> bool:
    """Check if API key has exceeded rate limit."""
    current_time = int(time.time())
    window_start = current_time - window

    # Use sorted set to track request timestamps
    redis_key = f"ratelimit:{api_key}"

    # Remove old requests (outside window)
    redis_client.zremrangebyscore(redis_key, 0, window_start)

    # Count requests in current window
    request_count = redis_client.zcard(redis_key)

    if request_count >= limit:
        return False  # Rate limit exceeded

    # Add current request
    redis_client.zadd(redis_key, {current_time: current_time})
    redis_client.expire(redis_key, window)

    return True  # Within rate limit
```

---

### Input Validation

**Pydantic Validation:**
```python
from pydantic import BaseModel, validator
from enum import Enum

class StatType(str, Enum):
    PASSING_YARDS = "passing_yards"
    RUSHING_YARDS = "rushing_yards"
    RECEIVING_YARDS = "receiving_yards"

class PredictionRequest(BaseModel):
    game_id: str
    player_id: str
    stat_type: StatType

    @validator('game_id')
    def validate_game_id(cls, v):
        # Format: "2024_10_KC_LV"
        if not re.match(r'^\d{4}_\d{1,2}_[A-Z]{2,3}_[A-Z]{2,3}$', v):
            raise ValueError('Invalid game_id format')
        return v

    @validator('player_id')
    def validate_player_id(cls, v):
        # Format: "00-0033873" (nflfastR player ID)
        if not re.match(r'^\d{2}-\d{7}$', v):
            raise ValueError('Invalid player_id format')
        return v
```

**Validation Errors:**
```
HTTP/1.1 422 Unprocessable Entity
{
  "detail": [
    {
      "loc": ["body", "game_id"],
      "msg": "Invalid game_id format",
      "type": "value_error"
    }
  ]
}
```

---

## Testing Strategy

### Unit Tests

**Test Coverage:**
- API endpoints (health, predict, models)
- Authentication (valid/invalid API keys)
- Rate limiting (within/exceeding limits)
- Feature extraction (cache hit/miss, database query)
- Ensemble prediction (mock models, aggregation)
- Temperature scaling (calibration)

**Example Test:**
```python
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_predict_valid_request():
    """Test /predict endpoint with valid API key."""
    headers = {"X-API-Key": "test_api_key_123"}
    payload = {
        "game_id": "2024_10_KC_LV",
        "player_id": "00-0033873",
        "stat_type": "passing_yards"
    }

    response = client.post("/predict", json=payload, headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "uncertainty" in data
    assert data["prediction"] > 0

def test_predict_invalid_api_key():
    """Test /predict endpoint with invalid API key."""
    headers = {"X-API-Key": "invalid_key"}
    payload = {"game_id": "2024_10_KC_LV", "player_id": "00-0033873", "stat_type": "passing_yards"}

    response = client.post("/predict", json=payload, headers=headers)

    assert response.status_code == 401
```

---

### Integration Tests

**Test Scenarios:**
- End-to-end prediction (database → prediction → response)
- Redis cache hit/miss behavior
- Database connection pooling
- Model loading (verify all 4 models load successfully)
- Prometheus metrics collection

**Fixtures:**
```python
@pytest.fixture
def db_connection():
    """Provide test database connection."""
    conn = psycopg.connect("postgresql://dro:sicillionbillions@localhost:5544/testdb")  # pragma: allowlist secret
    yield conn
    conn.close()

@pytest.fixture
def redis_client():
    """Provide test Redis client."""
    client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    yield client
    client.flushdb()  # Clean up after test
```

---

## Related Documentation

**Container Documentation:**
- [c4_level2_container_overview.md](c4_level2_container_overview.md) - Overall container architecture
- [c4_level2_data_container.md](c4_level2_data_container.md) - PostgreSQL database structure
- [c4_level2_monitoring_container.md](c4_level2_monitoring_container.md) - Prometheus/Grafana stack

**Component Documentation:**
- [c4_component_training_pipeline.md](c4_component_training_pipeline.md) - Model training workflow
- [sequence_prediction_flow.md](sequence_prediction_flow.md) - Prediction sequence diagram

**Implementation Files:**
- [api/main.py](../../api/main.py) - Main router (420 lines)
- [api/prediction.py](../../api/prediction.py) - Ensemble predictor (400 lines)
- [py/ensemble/enhanced_ensemble_v5.py](../../py/ensemble/enhanced_ensemble_v5.py) - Ensemble manager (1,200 lines)

---

## Legend

**Color Coding:**
- **Blue (#e1f5fe)** - External clients
- **Yellow (#fff9c4)** - Entry points
- **Orange (#ffccbc)** - Middleware
- **Purple (#f3e5f5)** - Core prediction logic
- **Green (#c8e6c9)** - Model loading
- **Cyan (#b3e5fc)** - Data access clients
- **Light Purple (#d1c4e9)** - Prometheus metrics
- **Red (#ffebee)** - External services

---

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2025-11-15 | 1.0 | Initial API container diagram created |

---

**Diagram Created:** November 15, 2025
**Maintained By:** Architecture Team
**Contact:** See [CLAUDE.md](../../CLAUDE.md) for questions
