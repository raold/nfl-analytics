# Architecture Documentation Gaps

**Purpose:** Identify missing diagrams and documentation for NFL analytics system

**Date:** November 14, 2025

**Version:** 1.0

---

## Completed C4 Diagrams

### 1. C4 Context Diagram ✅

**File:** `c4_context.md`

**Coverage:**
- External systems (nflverse/nflfastR, The Odds API, Git LFS)
- User personas (Bettor, ML Engineer, Data Scientist)
- System boundaries (NFL Analytics System)
- High-level data flows

**Stakeholder Value:**
- Onboarding new team members
- Communicating with non-technical stakeholders
- Understanding system dependencies

---

### 2. C4 Container Diagram ✅

**File:** `c4_container.md`

**Coverage:**
- Docker services (7 core + 3 optional)
- Port mappings and network configuration
- Inter-container communication
- Health checks and dependencies

**Stakeholder Value:**
- Infrastructure planning
- DevOps deployment
- Troubleshooting service issues

---

### 3. C4 Component Diagram - Ensemble ✅

**File:** `c4_component_ensemble.md`

**Coverage:**
- 4-way ensemble architecture (State-Space, Hierarchical, XGBoost, Priors)
- Prediction pipeline (request → cache → features → models → response)
- Post-processing (weight combination, temperature scaling, uncertainty)
- R bridge (rpy2) and Python model components

**Stakeholder Value:**
- ML Engineer model integration
- Data Scientist model development
- Performance optimization

---

## Missing C4 Diagrams

### 4. C4 Component Diagram - Data Ingestion Pipeline

**Purpose:** Document nflverse → PostgreSQL ETL workflow

**Recommended Components:**
- **R Ingestion Scripts**
  - `ingest_pbp.R` - Play-by-play ingestion
  - `ingest_schedules.R` - Game schedule ingestion
  - `ingest_current_season.R` - Live season updates
- **nflverse API Client**
  - `load_pbp()` - Fetch play-by-play data
  - `load_schedules()` - Fetch game schedules
  - `load_player_stats()` - Fetch player stats
- **Data Transformations**
  - Team name normalization (LAR/LAC)
  - Date/time conversions (kickoff timestamps)
  - Player ID mapping (GSIS format)
- **PostgreSQL Writer**
  - `DBI::dbWriteTable()` - Bulk inserts
  - Schema validation
  - Materialized view refresh
- **Error Handling**
  - API rate limiting (429 errors)
  - Retry logic (exponential backoff)
  - Data quality checks (missing values, duplicates)

**Key Flows:**
1. R script triggers nflverse API
2. API returns JSON data (paginated)
3. R transforms JSON → data.frame
4. DBI writes to PostgreSQL
5. Refresh materialized views (`mart.game_summary`)
6. Log ingestion metrics (rows inserted, errors)

**Stakeholder Value:**
- Data Engineer troubleshooting
- Understanding data lineage
- Monitoring ingestion failures

---

### 5. C4 Component Diagram - Odds Polling Service

**Purpose:** Document The Odds API → PostgreSQL polling workflow

**Recommended Components:**
- **Odds Fetcher Service**
  - `odds_update_service.py` - Main polling loop
  - Poll interval: 15 minutes (configurable)
  - API key management (env var)
- **The Odds API Client**
  - REST endpoints (`/v4/sports/americanfootball_nfl/odds`)
  - Authentication (API key header)
  - Rate limiting (500 requests/month free tier)
- **Line Movement Detector**
  - Compare current odds to last poll
  - Threshold: 5.0 points (configurable)
  - Alert webhook (Slack, email)
- **PostgreSQL Writer**
  - `odds_history` table (game spreads, totals)
  - `player_prop_odds` table (passing yards, TDs)
  - Upsert logic (ON CONFLICT DO UPDATE)
- **Monitoring**
  - Prometheus metrics (polls, errors, latency)
  - Grafana dashboard (line movement alerts)

**Key Flows:**
1. Cron trigger (every 15 minutes)
2. Fetch odds from API
3. Compare to last stored odds
4. Detect line movements >5.0 points
5. Insert/update PostgreSQL
6. Send webhook alerts if significant movement
7. Log metrics to Prometheus

**Stakeholder Value:**
- Understanding live odds integration
- Debugging line movement alerts
- Monitoring API quota usage

---

### 6. C4 Component Diagram - Training Pipeline

**Purpose:** Document model training workflow (R + Python)

**Recommended Components:**
- **Training Data Fetcher**
  - Query PostgreSQL for training window (2018-2024)
  - Feature engineering (log transforms, rolling windows)
  - Train/test split (temporal validation)
- **R Model Training**
  - State-space v2: `train_state_space_v2.R`
  - Hierarchical v2: `train_hierarchical_v2.R`
  - Informative priors v2: `train_informative_priors_v2.R`
  - MCMC diagnostics (R-hat, ESS, divergences)
- **Python Model Training**
  - Hybrid XGBoost: `train_hybrid_xgboost.py`
  - Static model (100 trees, 2018-2024)
  - Incremental model (50 trees, last 400 games)
  - Hyperparameter tuning (GridSearchCV)
- **Model Validation**
  - MAE, RMSE on 2024 holdout set
  - 90% calibration check (coverage test)
  - Sharpe ratio, ROI simulation
- **Model Serialization**
  - Save to `models/` directory
  - Git LFS tracking (`.pkl`, `.rds`, `.json`)
  - Model registry update (metadata JSON)
- **Temperature Scaling**
  - Calibrate on 2024 holdout set
  - Optimize T to minimize calibration error
  - Save scaler to `models/calibration/`

**Key Flows:**
1. Fetch training data from PostgreSQL
2. Split into train/test (temporal)
3. Train 4 models (R + Python)
4. Validate on holdout set
5. Calibrate temperature scaler
6. Serialize models to disk
7. Push to Git LFS
8. Update model registry

**Stakeholder Value:**
- Reproducible model training
- Understanding training workflow
- Debugging model performance issues

---

### 7. C4 Code Diagram - State-Space Model Implementation

**Purpose:** Detailed class diagram for state-space v2 predictor

**Recommended Components:**
- **StateSpacePredictor**
  - `__init__(model_path, skills_path)`
  - `load_model()` - Load .rds via rpy2
  - `load_skills()` - Load player skill trajectories
  - `predict(player_id, season, week, attempts, is_home)`
- **Skills DataFrame**
  - Columns: `player_id`, `season`, `week`, `skill_mean`, `skill_std`
  - Precomputed from MCMC posterior
  - Enables fast vectorized predictions
- **R Bridge (rpy2)**
  - `ro.r('model <- readRDS("...")')`
  - `ro.r('posterior <- posterior_predict(model, newdata=...)')`
  - `ro.r('mean(exp(posterior))')`
- **Prediction Algorithm**
  1. Look up player skill from skills DataFrame
  2. Construct prediction data (attempts, is_home)
  3. Call R posterior_predict()
  4. Exponentiate log_yards → yards
  5. Return (mean, std)

**Key Methods:**
```python
class StateSpacePredictor:
    def __init__(self, model_path, skills_path):
        self.model = self._load_rds(model_path)
        self.skills_df = pd.read_csv(skills_path)

    def predict(self, player_id, season, week, attempts, is_home):
        # Look up skill
        skill = self.skills_df.query(
            "player_id == @player_id & season == @season & week == @week"
        )

        # Predict via R
        ro.r.assign('attempts', attempts)
        ro.r.assign('is_home', is_home)
        ro.r.assign('skill_mean', skill['skill_mean'].values[0])
        ro.r('posterior <- posterior_predict(model, ...)')

        mean = ro.r('mean(exp(posterior))')[0]
        std = ro.r('sd(exp(posterior))')[0]

        return mean, std
```

**Stakeholder Value:**
- Understanding state-space implementation
- Debugging prediction issues
- Extending to other stat types

---

### 8. C4 Deployment Diagram - Production Environment

**Purpose:** Document production deployment architecture

**Recommended Components:**
- **Load Balancer**
  - NGINX or AWS ALB
  - Round-robin to 4 API replicas
  - Health check: `GET /health`
- **API Replicas (4 containers)**
  - Horizontal scaling
  - Stateless (Redis for shared cache)
  - Rolling updates (zero downtime)
- **Database (PostgreSQL)**
  - Primary-replica setup (if scaled)
  - Read replicas for analytics queries
  - Connection pooling (PgBouncer)
- **Cache (Redis Cluster)**
  - 3-node cluster for high availability
  - Redis Sentinel for failover
  - 6 GB total memory (2 GB per node)
- **Monitoring**
  - Prometheus (HA pair)
  - Grafana (read-only replicas)
  - Alertmanager (PagerDuty integration)

**Key Flows:**
1. User request → Load balancer
2. Load balancer → API replica (round-robin)
3. API replica → Redis cluster (cache check)
4. Cache miss → PostgreSQL read replica
5. API response → User
6. Metrics → Prometheus → Grafana
7. Alerts → Alertmanager → PagerDuty

**Stakeholder Value:**
- Production deployment planning
- Scaling strategy
- High availability design

---

### 9. C4 Deployment Diagram - Development Environment

**Purpose:** Document local Docker Compose setup

**Recommended Components:**
- **Docker Compose Network** (`nfl-analytics`)
- **Core Services**
  - PostgreSQL (port 5544)
  - Redis (port 6379)
  - FastAPI (port 8000)
  - Streamlit (port 8501)
  - Prometheus (port 9090)
  - Grafana (port 3000)
- **Exporters**
  - Node Exporter (port 9100)
  - Postgres Exporter (port 9187)
- **Volume Mounts**
  - `./pgdata` - PostgreSQL data
  - `./models` - Model artifacts
  - `./logs` - Application logs
- **Hot Reload**
  - API code: `./api:/app/api`
  - Dashboard code: `./dashboard:/app`

**Key Differences from Production:**
- Single-node deployment (no load balancer)
- Bind mounts for code (hot reload)
- Development credentials (not secrets)
- Lower resource limits (2 GB Redis vs 6 GB)

**Stakeholder Value:**
- Developer onboarding
- Local development setup
- Troubleshooting Docker issues

---

### 10. Sequence Diagram - Prediction Request Flow

**Purpose:** Step-by-step sequence of API prediction request

**Actors:**
- Bettor
- FastAPI
- Redis Cache
- PostgreSQL
- Ensemble Orchestrator
- R Models (3 instances)
- XGBoost Model
- Temperature Scaler

**Sequence:**
1. Bettor → FastAPI: `POST /predict`
2. FastAPI → Predict Endpoint: Validate request
3. Predict Endpoint → Cache Manager: Check cache
4. Cache Manager → Redis: `GET prediction:...`
5. Redis → Cache Manager: Cache miss
6. Cache Manager → Feature Fetcher: Fetch features
7. Feature Fetcher → Redis: `GET features:...`
8. Redis → Feature Fetcher: Cache miss
9. Feature Fetcher → PostgreSQL: SELECT player stats
10. PostgreSQL → Feature Fetcher: Return features
11. Feature Fetcher → Feature Transformer: Transform features
12. Feature Transformer → Feature Fetcher: Transformed features
13. Feature Fetcher → Redis: `SET features:... TTL 1800`
14. Feature Fetcher → Ensemble Orchestrator: Features ready
15. Ensemble Orchestrator → State-Space v2: Predict
16. State-Space v2 → R Predictor: posterior_predict()
17. R Predictor → State-Space v2: (265.0, 42.0)
18. Ensemble Orchestrator → Hierarchical v2: Predict
19. Hierarchical v2 → R Predictor: posterior_predict()
20. R Predictor → Hierarchical v2: (270.0, 38.0)
21. Ensemble Orchestrator → Hybrid XGBoost: Predict
22. Hybrid XGBoost → XGBoost Predictor: predict()
23. XGBoost Predictor → Hybrid XGBoost: (258.0, 34.0)
24. Ensemble Orchestrator → Informative Priors v2: Predict
25. Informative Priors v2 → R Predictor: posterior_predict()
26. R Predictor → Informative Priors v2: (268.0, 46.0)
27. Ensemble Orchestrator → Weight Combiner: Combine predictions
28. Weight Combiner → Uncertainty Aggregator: Pool variance
29. Uncertainty Aggregator → Temperature Scaler: Scale variance
30. Temperature Scaler → Ensemble Orchestrator: (265.3, 36.5)
31. Ensemble Orchestrator → Cache Manager: Cache result
32. Cache Manager → Redis: `SET prediction:... TTL 3600`
33. Cache Manager → Predict Endpoint: Return prediction
34. Predict Endpoint → Metrics Endpoint: Record metrics
35. Metrics Endpoint → Prometheus: Increment counters
36. Predict Endpoint → FastAPI: Return JSON
37. FastAPI → Bettor: HTTP 200 (prediction JSON)

**Timing:**
- Steps 1-14: 25ms (cache + features)
- Steps 15-26: 180ms (model predictions)
- Steps 27-30: 3ms (post-processing)
- Steps 31-37: 5ms (cache store + response)
- **Total:** 213ms (uncached), 10ms (cached)

**Stakeholder Value:**
- Understanding request lifecycle
- Performance optimization
- Debugging latency issues

---

### 11. Entity Relationship Diagram (ERD) - Database Schema

**Purpose:** Visual representation of 56 database tables and relationships

**Key Entities:**
- **Games** - 7,263 games (2006-2024)
- **Plays** - 1,254,173 play-by-play records
- **Players** - Player hierarchy (current team, position, years exp)
- **Player Game Stats** - Player-level stats by game
- **Odds History** - Historical betting lines
- **Player Prop Odds** - Player-specific props
- **Training Runs** - Model training metadata
- **Predictions** - Stored ensemble predictions

**Relationships:**
- Games (1) → (N) Plays
- Games (1) → (N) Player Game Stats
- Players (1) → (N) Player Game Stats
- Games (1) → (N) Odds History
- Players (1) → (N) Player Prop Odds
- Training Runs (1) → (N) Predictions

**Stakeholder Value:**
- Database design understanding
- Query optimization
- Schema evolution planning

**Note:** Partial ERD exists in `analysis/dissertation/figures/out/database_erd.tex`, but needs update for latest schema (56 tables vs 48 documented).

---

### 12. Data Flow Diagram - End-to-End Pipeline

**Purpose:** Show data flow from ingestion to prediction to monitoring

**Stages:**
1. **Ingestion** - nflverse → PostgreSQL
2. **Feature Engineering** - PostgreSQL → Transformed features
3. **Model Training** - Features → Trained models → Git LFS
4. **Prediction** - Features → Ensemble → Prediction
5. **Caching** - Prediction → Redis → Fast retrieval
6. **Monitoring** - Metrics → Prometheus → Grafana

**Data Stores:**
- nflverse API (external)
- PostgreSQL (56 tables, 1.25M plays)
- Redis (2 GB cache)
- Git LFS (67 models, ~1 GB)
- Prometheus (30-day metrics)
- Grafana (dashboards)

**Stakeholder Value:**
- Understanding data lineage
- Identifying bottlenecks
- Planning data governance

---

### 13. Technology Stack Diagram

**Purpose:** Visual representation of tech stack (languages, frameworks, tools)

**Layers:**
1. **Languages**
   - Python 3.11+ (60% of codebase)
   - R 4.3+ (30% of codebase)
   - SQL (PostgreSQL 16)
   - LaTeX (Dissertation)
2. **Frameworks**
   - FastAPI (API)
   - Streamlit (Dashboard)
   - brms/Stan (Bayesian models)
   - XGBoost (Gradient boosting)
3. **Infrastructure**
   - Docker Compose (7 services)
   - PostgreSQL + TimescaleDB
   - Redis 7
   - Prometheus + Grafana
4. **Development Tools**
   - Git LFS (model versioning)
   - pytest (testing)
   - pre-commit (linting)
   - uv (Python package manager)

**Stakeholder Value:**
- Technology evaluation
- Hiring/skill requirements
- Dependency management

---

### 14. Security Architecture Diagram

**Purpose:** Document security controls and threat model

**Components:**
- **Authentication**
  - Database: Password-based (env var)
  - Redis: Password-based (env var)
  - Grafana: Admin user/password
  - API: No authentication (internal network only)
- **Network Isolation**
  - Docker network: `nfl-analytics` (internal)
  - External ports: 8000, 8501, 3000, 9090
  - Firewall rules (if production)
- **Secrets Management**
  - Environment variables (`.env` file)
  - No secrets in code/Git
  - Rotate passwords quarterly
- **Data Protection**
  - Database encryption at rest (optional)
  - TLS for external connections (optional)
  - Volume backups (PostgreSQL, Redis)
- **Threat Model**
  - SQL injection (mitigated by parameterized queries)
  - Credential exposure (mitigated by env vars)
  - DDoS (mitigated by rate limiting)
  - Data leakage (mitigated by read-only mounts)

**Stakeholder Value:**
- Security audit preparation
- Compliance (if production betting)
- Incident response planning

---

## Priority Recommendations

### High Priority (Implement within 1 week)

1. **Data Ingestion Component Diagram** - Critical for understanding ETL failures
2. **Training Pipeline Component Diagram** - Required for reproducible model updates
3. **Sequence Diagram (Prediction Flow)** - Essential for performance optimization

### Medium Priority (Implement within 1 month)

4. **Odds Polling Component Diagram** - Important for live betting integration
5. **Entity Relationship Diagram (ERD)** - Useful for database optimization
6. **Deployment Diagrams (Dev + Prod)** - Needed for scaling planning

### Low Priority (Implement as needed)

7. **State-Space Code Diagram** - Detailed implementation reference
8. **Data Flow Diagram** - Holistic data lineage view
9. **Technology Stack Diagram** - Useful for onboarding
10. **Security Architecture Diagram** - Required for production deployment

---

## Documentation Standards

### Diagram Format

**Preferred:** Mermaid syntax (text-based, version-controlled)

**Why Mermaid?**
- Text-based (no binary files)
- Git-friendly (diffs visible)
- Rendered in Markdown viewers (GitHub, GitLab)
- Easy to update (no graphical editor)

**Alternative:** PlantUML (more features, but less GitHub integration)

---

### Diagram Conventions

**Naming:**
- Use consistent naming across diagrams (e.g., "State-Space v2", not "State Space Model v2.0")
- Match code/file names exactly (e.g., `simplified_random_walk_v2.rds`)

**Colors:**
- External systems: Light gray
- Data stores: Blue
- Compute components: Green
- Monitoring: Yellow
- User personas: Orange

**Labels:**
- Include ports (e.g., "Port 8000")
- Include weights (e.g., "30% weight")
- Include latencies (e.g., "145ms avg")

---

### Documentation Maintenance

**Update Triggers:**
- New model added to ensemble
- Docker service added/removed
- Database schema changes
- Major performance optimizations

**Review Cadence:**
- Quarterly review of all diagrams
- Update version numbers and dates
- Archive outdated diagrams (don't delete)

**Ownership:**
- Context/Container: ML Engineer + DevOps
- Component: ML Engineer + Data Scientist
- Code: Data Scientist

---

## Related Documentation

### Existing Diagrams

1. **Database ERD** - `analysis/dissertation/figures/out/database_erd.tex` (outdated, 48 tables vs 56 current)
2. **TikZ Diagrams** - `analysis/dissertation/style/tikz_diagram_style.tex` (standardized styling)
3. **Architecture Reference** - `docs/ARCHITECTURE_REFERENCE.md` (text-based overview)

### Existing Documentation

1. **CLAUDE.md** - Comprehensive project documentation (4,909 lines)
2. **README.md** - Quick start guide
3. **SETUP.md** - Detailed setup instructions
4. **docs/COMPONENT_DEEP_DIVES.md** - Component usage guides
5. **docs/database/DATABASE_QUICK_REFERENCE.md** - Schema reference

---

## Implementation Plan

### Phase 1: High-Priority Diagrams (1 week)

**Week 1:**
- Create Data Ingestion Component Diagram
- Create Training Pipeline Component Diagram
- Create Sequence Diagram (Prediction Flow)

**Deliverables:**
- 3 Mermaid diagrams in `docs/audits/2025_11_comprehensive_audit/06_architecture_diagrams/`
- Update ARCHITECTURE_REFERENCE.md with diagram links

---

### Phase 2: Medium-Priority Diagrams (1 month)

**Weeks 2-4:**
- Create Odds Polling Component Diagram
- Update Entity Relationship Diagram (ERD)
- Create Deployment Diagrams (Dev + Prod)

**Deliverables:**
- 4 Mermaid diagrams
- Update DATABASE_QUICK_REFERENCE.md with ERD link

---

### Phase 3: Low-Priority Diagrams (As needed)

**Ongoing:**
- Create State-Space Code Diagram (if needed for debugging)
- Create Data Flow Diagram (if needed for data governance)
- Create Technology Stack Diagram (if needed for hiring)
- Create Security Architecture Diagram (if production deployment)

---

## Tools and Resources

### Mermaid Documentation

- **Official Docs:** https://mermaid.js.org/
- **Live Editor:** https://mermaid.live/
- **C4 Extension:** https://github.com/plantuml-stdlib/C4-PlantUML

### PlantUML Documentation

- **Official Docs:** https://plantuml.com/
- **C4 Extension:** https://github.com/plantuml-stdlib/C4-PlantUML

### Diagram Reviews

**Checklist:**
- [ ] Consistent naming across diagrams
- [ ] Accurate ports, weights, latencies
- [ ] Version numbers and dates updated
- [ ] Legend included (if needed)
- [ ] Renders correctly in GitHub Markdown
- [ ] Cross-references to code/docs

---

**Author:** Claude Code (Sonnet 4.5)
**Date:** November 14, 2025
**Version:** 1.0
**Related Docs:** `c4_context.md`, `c4_container.md`, `c4_component_ensemble.md`
