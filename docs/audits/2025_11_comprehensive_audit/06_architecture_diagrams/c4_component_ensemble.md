# C4 Component Diagram - Ensemble Architecture v5.1

**Purpose:** Component-level view of 4-way ensemble prediction pipeline

**Date:** November 14, 2025

**Version:** 5.1 (Hybrid XGBoost Integration)

---

## Mermaid C4 Component Diagram

```mermaid
C4Component
    title Component Diagram - Enhanced Ensemble v5.1 (4-Way Prediction Pipeline)

    Container_Boundary(api_container, "FastAPI Service (Port 8000)") {

        Component_Boundary(api_layer, "API Layer") {
            Component(predict_endpoint, "Predict Endpoint", "POST /predict", "Handles prediction requests, validates input, orchestrates ensemble")
            Component(health_endpoint, "Health Endpoint", "GET /health", "Service health checks, model load status")
            Component(metrics_endpoint, "Metrics Endpoint", "GET /metrics", "Prometheus metrics export")
        }

        Component_Boundary(model_manager, "Model Manager") {
            Component(ensemble_orchestrator, "Ensemble Orchestrator", "Python Class", "Loads models, coordinates predictions, applies weights")
            Component(cache_manager, "Cache Manager", "Redis Client", "Prediction memoization (TTL: 3600s)")
        }

        Component_Boundary(ensemble_models, "4-Way Ensemble Models") {
            Component(state_space_v2, "State-Space v2", "R/Stan (30%)", "Random walk with skill evolution, MAE: 47.77 yards, 2018-2024 + H=4 decay")
            Component(hierarchical_v2, "Hierarchical v2", "R/brms (30%)", "Multilevel model: QB, team, opponent effects, partial pooling")
            Component(hybrid_xgboost_v1, "Hybrid XGBoost v1", "XGBoost (25%)", "70% static + 30% incremental, MAE: 38.03 yards, +4.2% improvement")
            Component(informative_priors_v2, "Informative Priors v2", "R/brms (15%)", "Domain-expert priors, conservative uncertainty")
        }

        Component_Boundary(post_processing, "Post-Processing") {
            Component(weight_combiner, "Weight Combiner", "Inverse Variance", "Weighted average: w_i = 1/σ²_i, normalize to sum=1")
            Component(temperature_scaler, "Temperature Scaler", "T=0.5534", "Calibrates 90% coverage from 78.5% to 90.2%")
            Component(uncertainty_aggregator, "Uncertainty Aggregator", "Variance Pooling", "√(Σ w²_i σ²_i) for ensemble uncertainty")
        }

        Component_Boundary(feature_engineering, "Feature Engineering") {
            Component(feature_fetcher, "Feature Fetcher", "SQL Queries", "Fetches player stats, game context, weather, spreads")
            Component(feature_transformer, "Feature Transformer", "Pandas", "Log transforms, rolling windows (L3), interactions")
            Component(feature_cache, "Feature Cache", "Redis", "Caches transformed features (TTL: 1800s)")
        }

        Component_Boundary(r_bridge, "R Bridge (rpy2)") {
            Component(r_model_loader, "R Model Loader", "rpy2", "Loads .rds models into R session")
            Component(r_predictor, "R Predictor", "brms::posterior_predict", "Generates posterior samples (4000 draws)")
            Component(r_postprocessor, "R Post-Processor", "exp() transform", "Converts log_yards to yards scale")
        }

        Component_Boundary(python_models, "Python Model Components") {
            Component(xgb_loader, "XGBoost Loader", "xgboost.Booster", "Loads hybrid static-incremental model")
            Component(xgb_predictor, "XGBoost Predictor", "DMatrix", "Point predictions (no uncertainty)")
        }
    }

    ComponentDb(postgres, "PostgreSQL", "psycopg2", "Feature store: player_game_stats, games, weather")
    ComponentDb(redis, "Redis", "redis-py", "LRU cache: predictions, features")

    Rel(predict_endpoint, ensemble_orchestrator, "Calls predict()", "Python method")
    Rel(ensemble_orchestrator, cache_manager, "Check cache", "Redis GET")
    Rel(cache_manager, redis, "Get cached prediction", "TTL: 3600s")

    Rel(ensemble_orchestrator, feature_fetcher, "Fetch features", "SQL query")
    Rel(feature_fetcher, postgres, "SELECT player stats", "psycopg2")
    Rel(feature_fetcher, feature_cache, "Check cache", "Redis GET")
    Rel(feature_cache, redis, "Get cached features", "TTL: 1800s")
    Rel(feature_fetcher, feature_transformer, "Transform raw data", "Pandas pipeline")

    Rel(ensemble_orchestrator, state_space_v2, "Predict", "rpy2 bridge")
    Rel(ensemble_orchestrator, hierarchical_v2, "Predict", "rpy2 bridge")
    Rel(ensemble_orchestrator, hybrid_xgboost_v1, "Predict", "Python API")
    Rel(ensemble_orchestrator, informative_priors_v2, "Predict", "rpy2 bridge")

    Rel(state_space_v2, r_model_loader, "Load .rds", "readRDS()")
    Rel(state_space_v2, r_predictor, "Generate samples", "posterior_predict()")
    Rel(r_predictor, r_postprocessor, "Transform", "exp(log_yards)")

    Rel(hierarchical_v2, r_model_loader, "Load .rds", "readRDS()")
    Rel(hierarchical_v2, r_predictor, "Generate samples", "posterior_predict()")
    Rel(r_predictor, r_postprocessor, "Transform", "exp(log_yards)")

    Rel(informative_priors_v2, r_model_loader, "Load .rds", "readRDS()")
    Rel(informative_priors_v2, r_predictor, "Generate samples", "posterior_predict()")
    Rel(r_predictor, r_postprocessor, "Transform", "exp(log_yards)")

    Rel(hybrid_xgboost_v1, xgb_loader, "Load model", "Booster.load_model()")
    Rel(hybrid_xgboost_v1, xgb_predictor, "Predict", "predict(DMatrix)")

    Rel(ensemble_orchestrator, weight_combiner, "Combine predictions", "Weighted average")
    Rel(weight_combiner, uncertainty_aggregator, "Pool variance", "Sqrt sum w²σ²")
    Rel(uncertainty_aggregator, temperature_scaler, "Scale variance", "σ_cal = σ_raw / T")
    Rel(temperature_scaler, cache_manager, "Cache result", "Redis SET TTL 3600")
    Rel(cache_manager, predict_endpoint, "Return JSON", "HTTP 200")

    Rel(predict_endpoint, metrics_endpoint, "Record metrics", "Prometheus counter")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

---

## Component Details

### API Layer

#### 1. Predict Endpoint (`POST /predict`)

**Purpose:** Accept prediction requests, validate input, orchestrate ensemble

**Request Schema:**
```json
{
  "player_id": "00-0036355",
  "stat_type": "passing_yards",
  "week": 8,
  "season": 2024,
  "confidence_level": 0.90
}
```

**Validation:**
- Player ID format: `XX-XXXXXXX`
- Week: 1-18
- Season: 1999-2099
- Confidence level: 0.50-0.99
- Stat type: `passing_yards` (only supported type currently)

**Response Schema:**
```json
{
  "player_id": "00-0036355",
  "player_name": "Patrick Mahomes",
  "stat_type": "passing_yards",
  "week": 8,
  "season": 2024,
  "prediction": 268.5,
  "prediction_unit": "yards",
  "uncertainty": 32.4,
  "confidence_level": 0.90,
  "confidence_interval": {
    "lower": 210.2,
    "upper": 326.8,
    "level": 0.90
  },
  "model_versions": {
    "bayesian": "simplified_random_walk_v2",
    "bnn": "passing_yards_hierarchical_v2",
    "xgboost": "hybrid_static_incremental_v1",
    "state_space": "passing_informative_priors_v2",
    "ensemble": "v5.1"
  },
  "ensemble_weights": {
    "bayesian": 0.30,
    "bnn": 0.30,
    "xgboost": 0.25,
    "state_space": 0.15
  },
  "timestamp": "2024-11-08T14:30:45.123Z",
  "cache_hit": false,
  "processing_ms": 145.3
}
```

**Error Handling:**
- 400 Bad Request - Invalid input (player not found, unsupported stat)
- 503 Service Unavailable - Models not loaded
- 500 Internal Server Error - Unexpected failure

---

#### 2. Health Endpoint (`GET /health`)

**Purpose:** Service health checks for container orchestration

**Response Schema:**
```json
{
  "status": "healthy",
  "timestamp": "2024-11-08T14:30:45.123Z",
  "version": "1.0.0",
  "models_loaded": 4,
  "cache_available": true,
  "cache_stats": {
    "hit_rate": 0.64,
    "total_keys": 1523,
    "memory_used_mb": 456.2
  },
  "uptime_seconds": 3600.5
}
```

**Status Values:**
- `healthy` - All models loaded, cache available
- `degraded` - Some models missing or cache unavailable
- `unhealthy` - Critical failure (no models loaded)

---

#### 3. Metrics Endpoint (`GET /metrics`)

**Purpose:** Prometheus-formatted metrics export

**Key Metrics:**
- `nfl_predictions_total{stat_type="passing_yards", cache_hit="true"}` - Counter
- `nfl_prediction_duration_seconds{stat_type="passing_yards"}` - Histogram (buckets: 0.01, 0.05, 0.1, 0.5, 1.0)
- `nfl_prediction_uncertainty_yards{stat_type="passing_yards"}` - Histogram (buckets: 5, 10, 15, 20, 30, 50, 100)
- `nfl_cache_hits_total{cache_type="redis"}` - Counter
- `nfl_cache_misses_total{cache_type="redis"}` - Counter
- `nfl_model_load_status{model_name="ensemble_v5_1"}` - Gauge (1=success, 0=failure)
- `nfl_service_uptime_seconds` - Gauge

---

### Model Manager

#### 4. Ensemble Orchestrator

**Purpose:** Coordinates prediction workflow across 4 models

**Key Methods:**
```python
class EnhancedEnsemble4Way:
    def __init__(self, weights=None):
        """Initialize with model paths and weights"""

    def load_models(self):
        """Lazy load models into memory"""

    def predict(self, player_id, season, week, return_individual=False):
        """Generate ensemble prediction"""
        # 1. Check cache
        # 2. Fetch features
        # 3. Predict with each model
        # 4. Combine with weights
        # 5. Apply temperature scaling
        # 6. Cache result
        # 7. Return prediction

    def _get_r_predictions(self, model_path, player_data):
        """Bridge to R models via rpy2"""

    def _get_xgboost_prediction(self, player_data):
        """Python XGBoost prediction"""
```

**Weights (Default):**
```python
DEFAULT_WEIGHTS = {
    "state_space_v2": 0.30,
    "hierarchical_v2": 0.30,
    "hybrid_xgboost_v1": 0.25,
    "informative_priors_v2": 0.15,
}
```

**Weight Validation:**
- Weights must sum to 1.0 (±0.01 tolerance)
- If weights don't sum to 1.0, normalize: `w_i = w_i / Σw_i`

---

#### 5. Cache Manager

**Purpose:** Redis-backed prediction memoization

**Cache Strategy:**
- **Key:** `prediction:{player_id}:{stat_type}:{week}:{season}:{confidence_level}`
- **Value:** JSON-serialized `PredictionResponse`
- **TTL:** 3600s (1 hour)
- **Eviction:** LRU (least recently used)

**Cache Hit Benefits:**
- Latency: 145ms → 5-20ms (96% reduction)
- Database load: 0 queries
- Model inference: 0 computations

**Cache Miss Penalty:**
- +125ms for database queries
- +20ms for model inference

**Cache Warming (Startup):**
- Top 50 players pre-loaded
- BNN + XGBoost models pre-loaded into memory
- Reduces cold start from 8-12s to 5-8s

---

### 4-Way Ensemble Models

#### 6. State-Space v2 (30% Weight)

**Type:** R/Stan - Random walk with skill evolution

**Training Window:** 2018-2024 + H=4 decay on 2015-2017

**Architecture:**
```r
log(yards_i) ~ Normal(μ_i, σ_obs)
μ_i = α + β_player[i] + β_attempts * log(attempts_i) + β_home * is_home_i
β_player[i,t] ~ Normal(β_player[i,t-1], σ_skill)  # Random walk
```

**Performance:**
- MAE: 47.77 yards (2024 holdout)
- 90% Calibration: 92.1%
- Inference time: ~50ms per player

**Model File:**
- Path: `models/bayesian/simplified_random_walk_v2.rds`
- Size: 125 MB
- Format: Stan compiled model (cmdstanr)

**Prediction API:**
```python
# Via rpy2
ro.r('model <- readRDS("simplified_random_walk_v2.rds")')
ro.r('posterior <- posterior_predict(model, newdata=player_df)')
mean_yards = ro.r('mean(exp(posterior))')[0]
std_yards = ro.r('sd(exp(posterior))')[0]
```

**Key Features:**
- Player skill trajectories (random walk)
- Attempts normalization (log scale)
- Home/away adjustment
- Temporal correlation (week-to-week)

---

#### 7. Hierarchical v2 (30% Weight)

**Type:** R/brms - Multilevel Bayesian regression

**Training Window:** 2018-2024 + H=4 decay on 2015-2017

**Architecture:**
```r
log(yards_i) ~ Normal(μ_i, σ)
μ_i = α + β_QB[i] + β_team[i] + β_opponent[i] + β_attempts * log(attempts_i) + β_home * is_home_i
β_QB ~ Normal(0, σ_QB)        # QB-level random effects
β_team ~ Normal(0, σ_team)     # Team-level random effects
β_opponent ~ Normal(0, σ_opp)  # Opponent-level random effects
```

**Performance:**
- MAE: 48.2 yards (2024 holdout)
- 90% Calibration: 91.5%
- Inference time: ~60ms per player

**Model File:**
- Path: `models/bayesian/passing_yards_hierarchical_v2.rds`
- Size: 98 MB
- Format: brms fitted model

**Key Features:**
- Partial pooling across QBs, teams, opponents
- Shrinkage for rookies/limited data
- Multilevel uncertainty quantification
- Group-level variance estimates

---

#### 8. Hybrid XGBoost v1 (25% Weight)

**Type:** XGBoost - 70% static + 30% incremental blend

**Training Window:** 2018-2024 (static), last 400 games (incremental)

**Architecture:**
```python
# Static model (70% weight)
static_pred = xgb_static.predict(X)

# Incremental model (30% weight)
incremental_pred = xgb_incremental.predict(X)

# Hybrid blend
hybrid_pred = 0.70 * static_pred + 0.30 * incremental_pred
```

**Performance:**
- MAE: 38.03 yards (vs 39.66 static-only, +4.2% improvement)
- 90% Calibration: N/A (point estimates only)
- Inference time: ~15ms per player

**Model File:**
- Path: `models/xgboost/hybrid_static_incremental_v1/`
- Size: 45 MB (static) + 12 MB (incremental)
- Format: XGBoost JSON

**Key Features:**
- Gradient boosting (100 trees static, 50 trees incremental)
- L2 regularization (λ=1.0)
- Learning rate: 0.1 (static), 0.05 (incremental)
- Max depth: 6 (static), 4 (incremental)
- Incremental model adapts to recent trends (last 400 games)

**Uncertainty Estimation:**
- Fixed σ = 34.13 yards (based on test RMSE)
- Conservative estimate (no distributional modeling)

---

#### 9. Informative Priors v2 (15% Weight)

**Type:** R/brms - Domain-expert priors

**Training Window:** 2018-2024 + H=4 decay on 2015-2017

**Architecture:**
```r
log(yards_i) ~ Normal(μ_i, σ)
μ_i = α + β_attempts * log(attempts_i) + β_home * is_home_i + β_spread * spread_i

# Informative priors (domain expertise)
α ~ Normal(log(250), 0.2)           # Typical QB: 250 yards
β_attempts ~ Normal(0.8, 0.1)       # Strong positive effect
β_home ~ Normal(0.05, 0.05)         # Small home advantage
β_spread ~ Normal(-0.01, 0.01)      # Small spread effect
σ ~ Exponential(0.05)               # Conservative variance
```

**Performance:**
- MAE: 49.5 yards (2024 holdout)
- 90% Calibration: 94.2% (over-conservative)
- Inference time: ~55ms per player

**Model File:**
- Path: `models/bayesian/passing_informative_priors_v2.rds`
- Size: 82 MB
- Format: brms fitted model

**Key Features:**
- Expert-informed priors (conservative)
- Strong regularization (prevents overfitting)
- High calibration (90%+ coverage)
- Useful for sparse data (rookies, backups)

---

### Post-Processing

#### 10. Weight Combiner (Inverse Variance)

**Purpose:** Combine 4 model predictions using inverse variance weighting

**Algorithm:**
```python
def combine_predictions(predictions):
    """
    predictions: [(mean_1, std_1), (mean_2, std_2), ...]
    """
    means = np.array([p[0] for p in predictions])
    stds = np.array([p[1] for p in predictions])

    # Inverse variance weights
    weights = 1.0 / (stds ** 2)
    weights /= weights.sum()  # Normalize to sum=1

    # Weighted mean
    ensemble_mean = np.sum(weights * means)

    return ensemble_mean, weights
```

**Example:**
```python
predictions = [
    (265.0, 42.0),  # State-space v2
    (270.0, 38.0),  # Hierarchical v2
    (258.0, 34.0),  # Hybrid XGBoost
    (268.0, 46.0),  # Informative priors v2
]

# Inverse variance weights
weights = 1/42², 1/38², 1/34², 1/46²
weights = [0.000567, 0.000693, 0.000865, 0.000473]
weights /= sum(weights)  # [0.224, 0.274, 0.342, 0.187]

# Note: XGBoost gets highest weight due to lowest variance
ensemble_mean = 0.224*265 + 0.274*270 + 0.342*258 + 0.187*268 = 263.5 yards
```

**Rationale:**
- Models with lower uncertainty get higher weight
- Accounts for model confidence (not just accuracy)
- Self-adjusting based on input complexity

---

#### 11. Temperature Scaler (T=0.5534)

**Purpose:** Calibrate 90% confidence intervals from 78.5% to 90.2%

**Algorithm:**
```python
def temperature_scale(ensemble_std, T=0.5534):
    """
    Adjusts variance to improve calibration.

    Before scaling:
    - Ensemble 90% CI coverage: 78.5% (undercalibrated)

    After scaling:
    - Ensemble 90% CI coverage: 90.2% (well-calibrated)
    """
    return ensemble_std / T
```

**Derivation:**
- Trained on 2024 holdout set (148 games)
- Optimize T to minimize calibration error
- Cross-validated to prevent overfitting

**Impact:**
```python
# Before temperature scaling
ensemble_std = 32.4 yards
CI_90 = [268.5 - 1.645*32.4, 268.5 + 1.645*32.4]
      = [215.2, 321.8]  # Too narrow (78.5% coverage)

# After temperature scaling
ensemble_std_cal = 32.4 / 0.5534 = 58.5 yards
CI_90 = [268.5 - 1.645*58.5, 268.5 + 1.645*58.5]
      = [172.3, 364.7]  # Well-calibrated (90.2% coverage)
```

**File:**
- Path: `models/calibration/temperature_scaler_4way.pkl`
- Format: Pickled dict `{'temperature': 0.5534, 'metadata': {...}}`

---

#### 12. Uncertainty Aggregator (Variance Pooling)

**Purpose:** Combine individual model uncertainties into ensemble uncertainty

**Algorithm:**
```python
def pool_variance(stds, weights):
    """
    Pools variance across models accounting for weights.

    Formula:
    σ²_ensemble = Σ w²_i σ²_i

    Note: This assumes model predictions are independent.
    """
    variances = stds ** 2
    ensemble_var = np.sum((weights ** 2) * variances)
    ensemble_std = np.sqrt(ensemble_var)

    return ensemble_std
```

**Example:**
```python
stds = [42.0, 38.0, 34.0, 46.0]  # Individual model uncertainties
weights = [0.30, 0.30, 0.25, 0.15]  # Ensemble weights

# Variance pooling
variances = [42², 38², 34², 46²] = [1764, 1444, 1156, 2116]
ensemble_var = 0.30²*1764 + 0.30²*1444 + 0.25²*1156 + 0.15²*2116
             = 158.8 + 130.0 + 72.3 + 47.6 = 408.7
ensemble_std = sqrt(408.7) = 20.2 yards

# Note: Ensemble uncertainty is LOWER than individual models
# This is expected - ensemble reduces variance through averaging
```

**Assumptions:**
- Model predictions are independent (violated if models share features)
- Model errors are uncorrelated (violated if systematic biases exist)

**Empirical Calibration:**
- Raw ensemble uncertainty often underestimates true variance
- Temperature scaling corrects for this (T=0.5534)

---

### Feature Engineering

#### 13. Feature Fetcher

**Purpose:** Query PostgreSQL for player stats, game context, weather

**SQL Query:**
```sql
SELECT
    pgs.player_id,
    pgs.season,
    pgs.week,
    pgs.stat_yards as yards,
    pgs.stat_touchdowns as touchdowns,
    pgs.stat_attempts as attempts,
    ph.current_team as team,
    g.spread_close as spread_line,
    g.total_close,
    g.temp,
    g.wind,
    CASE WHEN ph.current_team = g.home_team THEN 1 ELSE 0 END as is_home,
    g.home_team,
    g.away_team,
    CASE
      WHEN ph.current_team = g.home_team THEN CASE WHEN g.spread_close < 0 THEN 1 ELSE 0 END
      ELSE CASE WHEN g.spread_close > 0 THEN 1 ELSE 0 END
    END as is_favored,
    ABS(g.spread_close) as spread_abs,
    COALESCE(g.roof, 'unknown') as roof,
    CASE WHEN g.roof IN ('outdoors', 'open') AND CAST(g.temp AS NUMERIC) < 40 THEN 1 ELSE 0 END as is_bad_weather,
    CASE WHEN g.roof = 'dome' THEN 1 ELSE 0 END as is_dome,
    ph.years_exp,
    CASE
      WHEN ph.current_team = g.home_team THEN g.away_team
      ELSE g.home_team
    END as opponent,
    CONCAT(pgs.player_id, '_', pgs.season) as player_season
FROM mart.player_game_stats pgs
JOIN mart.player_hierarchy ph ON pgs.player_id = ph.player_id
LEFT JOIN games g ON
  g.season = pgs.season
  AND g.week = pgs.week
  AND (g.home_team = ph.current_team OR g.away_team = ph.current_team)
WHERE pgs.stat_category = 'passing'
  AND pgs.player_id = %s
  AND pgs.season = %s
  AND pgs.week = %s
  AND ph.position_group = 'QB'
LIMIT 1
```

**Query Time:**
- With indexes: 10-20ms
- Without indexes: 100-200ms

**Cache Strategy:**
- Check Redis before querying PostgreSQL
- Cache key: `features:{player_id}:{season}:{week}`
- TTL: 1800s (30 minutes)

---

#### 14. Feature Transformer

**Purpose:** Transform raw features into model-ready format

**Transformations:**
```python
def transform_features(df):
    """
    Transforms raw database features for model input.
    """
    # Log transformations
    df["log_attempts"] = np.log1p(df["attempts"])

    # Rate statistics
    df["yards_per_attempt"] = df["yards"] / df["attempts"].replace(0, np.nan)
    df["td_rate"] = df["touchdowns"] / df["attempts"].replace(0, np.nan)

    # Spread features
    df["spread_abs"] = df["spread_line"].abs()

    # Weather features
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    df["wind"] = pd.to_numeric(df["wind"], errors="coerce")
    df["is_bad_weather"] = (
        ((df["temp"] < 32) | (df["wind"] > 15)).fillna(False).astype(int)
    )

    # Week normalization
    df["week_in_season"] = df["week"].astype(float)

    # Rolling features (last 3 games)
    # For real-time, query historical games
    df["yards_L3"] = df["yards"].rolling(3).mean()
    df["yards_per_attempt_L3"] = df["yards_per_attempt"].rolling(3).mean()
    df["td_rate_L3"] = df["td_rate"].rolling(3).mean()

    # Fill missing values
    df = df.fillna({
        "temp": 70.0,  # Default moderate temp
        "wind": 5.0,   # Default calm wind
    })
    df = df.fillna(df.median(numeric_only=True))

    # Alias for backward compatibility
    df["total_line"] = df["total_close"]

    return df
```

**Feature List (13 total):**
1. `log_attempts` - Log-transformed attempts (reduces skewness)
2. `yards_per_attempt` - Efficiency metric
3. `td_rate` - Touchdown efficiency
4. `is_home` - Home field advantage
5. `spread_abs` - Game competitiveness
6. `total_line` - Over/under (game pace proxy)
7. `is_bad_weather` - Cold/windy conditions
8. `yards_L3` - Rolling 3-game average yards
9. `yards_per_attempt_L3` - Rolling efficiency
10. `td_rate_L3` - Rolling TD rate
11. `week_in_season` - Temporal trend
12. `temp` - Temperature (degrees F)
13. `wind` - Wind speed (mph)

---

#### 15. Feature Cache

**Purpose:** Redis cache for transformed features

**Cache Strategy:**
- Store transformed features (not raw database rows)
- Key: `features_transformed:{player_id}:{season}:{week}`
- TTL: 1800s (30 minutes)
- Format: JSON-serialized pandas DataFrame

**Benefits:**
- Reduces database load (1 query instead of N models)
- Reduces transformation time (1 transform instead of N models)
- Shared across models (all 4 models use same features)

---

### R Bridge (rpy2)

#### 16. R Model Loader

**Purpose:** Load .rds models into R session via rpy2

**Implementation:**
```python
import rpy2.robjects as ro

# Load brms library
ro.r('library(brms)')

# Load model
ro.r(f'model <- readRDS("{model_path}")')
```

**Model Caching:**
- Models loaded once per API startup
- Held in R session memory (avoids repeated loading)
- R session persists across requests (same FastAPI worker)

**Memory Overhead:**
- State-Space v2: 125 MB
- Hierarchical v2: 98 MB
- Informative Priors v2: 82 MB
- **Total R models:** 305 MB

---

#### 17. R Predictor

**Purpose:** Generate posterior samples using brms::posterior_predict

**Implementation:**
```python
# Convert pandas DataFrame to R dataframe
with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
    r_df = ro.conversion.py2rpy(df)

# Generate posterior predictions
ro.r.assign('new_data', r_df)
ro.r('new_data$log_attempts <- log(new_data$attempts)')
ro.r('posterior_preds <- posterior_predict(model, newdata = new_data)')
```

**Posterior Samples:**
- Default: 4000 draws (from MCMC posterior)
- Each draw represents one possible outcome
- Mean: `mean(posterior_preds)` → point prediction
- Std: `sd(posterior_preds)` → uncertainty

---

#### 18. R Post-Processor

**Purpose:** Convert log_yards to yards scale

**Implementation:**
```python
# CRITICAL: R models predict log_yards, must exponentiate
ro.r('posterior_preds_exp <- exp(posterior_preds)')
ro.r('pred_mean <- mean(posterior_preds_exp)')
ro.r('pred_sd <- sd(posterior_preds_exp)')

mean_pred = ro.r('pred_mean')[0]
std_pred = ro.r('pred_sd')[0]
```

**Why Log Scale?**
- Yards are right-skewed (0 to 500+)
- Log-normal distribution better fits data
- Log scale stabilizes variance
- Prevents negative predictions

---

### Python Model Components

#### 19. XGBoost Loader

**Purpose:** Load hybrid static-incremental XGBoost model

**Implementation:**
```python
from py.online.hybrid_static_incremental import HybridStaticIncrementalXGBoost

hybrid_xgb = HybridStaticIncrementalXGBoost()
hybrid_xgb.load("models/xgboost/hybrid_static_incremental_v1/")
```

**Model Files:**
- `static_model.json` - Static XGBoost (trained on 2018-2024)
- `incremental_model.json` - Incremental XGBoost (trained on last 400 games)
- `metadata.json` - Hyperparameters, feature names, training stats

---

#### 20. XGBoost Predictor

**Purpose:** Generate hybrid predictions (70% static + 30% incremental)

**Implementation:**
```python
def predict(self, X):
    """
    X: numpy array (n_samples, n_features)
    """
    dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)

    # Static prediction
    static_pred = self.static_model.predict(dmatrix)

    # Incremental prediction
    incremental_pred = self.incremental_model.predict(dmatrix)

    # Hybrid blend
    hybrid_pred = 0.70 * static_pred + 0.30 * incremental_pred

    return hybrid_pred
```

**Uncertainty:**
- XGBoost provides point estimates (no distributional output)
- Fixed σ = 34.13 yards (based on test RMSE)
- Conservative estimate (doesn't adapt to input)

---

## Prediction Flow (End-to-End)

### 1. Request Arrives

```
POST /predict HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "player_id": "00-0036355",
  "stat_type": "passing_yards",
  "week": 9,
  "season": 2024,
  "confidence_level": 0.90
}
```

---

### 2. Cache Check (5-20ms if hit)

```python
cache_key = "prediction:00-0036355:passing_yards:9:2024:0.90"
cached_response = redis.get(cache_key)

if cached_response:
    return cached_response  # Fast path: 5-20ms
```

---

### 3. Cache Miss → Fetch Features (10-20ms)

```python
# Check feature cache
feature_key = "features:00-0036355:2024:9"
cached_features = redis.get(feature_key)

if not cached_features:
    # Query PostgreSQL
    features_df = fetch_from_postgres(player_id, season, week)
    redis.set(feature_key, features_df, ttl=1800)
else:
    features_df = cached_features
```

---

### 4. Transform Features (5ms)

```python
features_df["log_attempts"] = np.log1p(features_df["attempts"])
features_df["yards_per_attempt"] = features_df["yards"] / features_df["attempts"]
features_df["is_bad_weather"] = (features_df["temp"] < 32) | (features_df["wind"] > 15)
# ... (13 total features)
```

---

### 5. Model Predictions (100ms total)

**State-Space v2 (50ms):**
```python
ro.r('model_ss <- readRDS("simplified_random_walk_v2.rds")')
ro.r('posterior_ss <- posterior_predict(model_ss, newdata=features_df)')
mean_ss = ro.r('mean(exp(posterior_ss))')[0]  # 265.0 yards
std_ss = ro.r('sd(exp(posterior_ss))')[0]     # 42.0 yards
```

**Hierarchical v2 (60ms):**
```python
ro.r('model_hier <- readRDS("passing_yards_hierarchical_v2.rds")')
ro.r('posterior_hier <- posterior_predict(model_hier, newdata=features_df)')
mean_hier = ro.r('mean(exp(posterior_hier))')[0]  # 270.0 yards
std_hier = ro.r('sd(exp(posterior_hier))')[0]     # 38.0 yards
```

**Hybrid XGBoost v1 (15ms):**
```python
hybrid_pred = hybrid_xgb.predict(features_df)  # 258.0 yards
std_xgb = 34.13  # Fixed RMSE-based uncertainty
```

**Informative Priors v2 (55ms):**
```python
ro.r('model_priors <- readRDS("passing_informative_priors_v2.rds")')
ro.r('posterior_priors <- posterior_predict(model_priors, newdata=features_df)')
mean_priors = ro.r('mean(exp(posterior_priors))')[0]  # 268.0 yards
std_priors = ro.r('sd(exp(posterior_priors))')[0]     # 46.0 yards
```

---

### 6. Weight Combination (1ms)

```python
predictions = [
    (265.0, 42.0),  # State-space v2
    (270.0, 38.0),  # Hierarchical v2
    (258.0, 34.0),  # Hybrid XGBoost
    (268.0, 46.0),  # Informative priors v2
]

weights = {
    "state_space_v2": 0.30,
    "hierarchical_v2": 0.30,
    "hybrid_xgboost_v1": 0.25,
    "informative_priors_v2": 0.15,
}

ensemble_mean = (
    0.30 * 265.0 +
    0.30 * 270.0 +
    0.25 * 258.0 +
    0.15 * 268.0
) = 265.3 yards
```

---

### 7. Variance Pooling (1ms)

```python
ensemble_var = (
    0.30² * 42.0² +
    0.30² * 38.0² +
    0.25² * 34.0² +
    0.15² * 46.0²
) = 408.7

ensemble_std = sqrt(408.7) = 20.2 yards
```

---

### 8. Temperature Scaling (1ms)

```python
T = 0.5534
ensemble_std_cal = ensemble_std / T = 20.2 / 0.5534 = 36.5 yards
```

---

### 9. Confidence Interval (1ms)

```python
confidence_level = 0.90
z_score = 1.645  # For 90% CI

CI_lower = ensemble_mean - z_score * ensemble_std_cal
         = 265.3 - 1.645 * 36.5 = 205.2 yards

CI_upper = ensemble_mean + z_score * ensemble_std_cal
         = 265.3 + 1.645 * 36.5 = 325.4 yards
```

---

### 10. Cache Store (5ms)

```python
cache_key = "prediction:00-0036355:passing_yards:9:2024:0.90"
response = {
    "prediction": 265.3,
    "uncertainty": 36.5,
    "confidence_interval_lower": 205.2,
    "confidence_interval_upper": 325.4,
    "ensemble_weights": weights,
    "model_versions": {...},
    "processing_ms": 145.3,
}

redis.set(cache_key, response, ttl=3600)
```

---

### 11. Response Returned (145ms total)

```json
{
  "player_id": "00-0036355",
  "player_name": "Patrick Mahomes",
  "stat_type": "passing_yards",
  "week": 9,
  "season": 2024,
  "prediction": 265.3,
  "prediction_unit": "yards",
  "uncertainty": 36.5,
  "confidence_level": 0.90,
  "confidence_interval": {
    "lower": 205.2,
    "upper": 325.4,
    "level": 0.90
  },
  "model_versions": {
    "bayesian": "simplified_random_walk_v2",
    "bnn": "passing_yards_hierarchical_v2",
    "xgboost": "hybrid_static_incremental_v1",
    "state_space": "passing_informative_priors_v2",
    "ensemble": "v5.1"
  },
  "ensemble_weights": {
    "bayesian": 0.30,
    "bnn": 0.30,
    "xgboost": 0.25,
    "state_space": 0.15
  },
  "timestamp": "2024-11-08T14:30:45.123Z",
  "cache_hit": false,
  "processing_ms": 145.3
}
```

---

## Performance Characteristics

**Latency Breakdown:**
- Cache check: 5ms
- Feature fetch (DB): 10-20ms
- Feature transform: 5ms
- State-space v2: 50ms
- Hierarchical v2: 60ms
- Hybrid XGBoost: 15ms
- Informative priors v2: 55ms
- Weight combination: 1ms
- Variance pooling: 1ms
- Temperature scaling: 1ms
- Cache store: 5ms
- **Total:** ~145ms (uncached)

**Cache Hit Path:**
- Cache check: 5ms
- Cache retrieval: 5-20ms
- **Total:** 5-20ms (96% faster)

**Throughput:**
- Single worker: ~100 req/s (cached), ~7 req/s (uncached)
- Horizontal scaling: 4 workers → ~400 req/s (cached)

**Memory Usage:**
- R models: 305 MB (State-space 125 MB, Hierarchical 98 MB, Priors 82 MB)
- XGBoost models: 57 MB (Static 45 MB, Incremental 12 MB)
- Redis cache: 2 GB max (LRU eviction)
- **Total:** ~2.4 GB per API container

---

## Technology Stack

**Python Libraries:**
- FastAPI - API framework
- rpy2 - R bridge
- XGBoost - Gradient boosting
- pandas - Data manipulation
- numpy - Numerical computation
- psycopg2 - PostgreSQL client
- redis-py - Redis client
- prometheus-client - Metrics export

**R Libraries:**
- brms - Bayesian regression
- cmdstanr - Stan interface
- posterior - Posterior sampling

**Infrastructure:**
- Docker - Containerization
- Redis - Cache
- PostgreSQL - Database
- Prometheus - Metrics
- Grafana - Dashboards

---

**Author:** Claude Code (Sonnet 4.5)
**Date:** November 14, 2025
**Version:** 5.1 (Hybrid XGBoost Integration)
**Related Docs:** `c4_context.md`, `c4_container.md`, `architecture_documentation_gaps.md`
