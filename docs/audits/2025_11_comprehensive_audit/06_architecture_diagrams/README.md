# C4 Architecture Diagrams - NFL Analytics System

**Purpose:** Visual documentation of NFL analytics system architecture using C4 model

**Date:** November 14, 2025

**Version:** 1.0

---

## Quick Navigation

| Diagram | Level | Purpose | Audience |
|---------|-------|---------|----------|
| [C4 Context](#c4-context-diagram) | 1 - System | External dependencies, user interactions | Executives, Business stakeholders |
| [C4 Container](#c4-container-diagram) | 2 - Container | Docker services, data flows | DevOps, ML Engineers |
| [C4 Component (Ensemble)](#c4-component-diagram---ensemble) | 3 - Component | 4-way ensemble prediction pipeline | ML Engineers, Data Scientists |
| [Documentation Gaps](#missing-diagrams) | N/A | Identified missing diagrams | All stakeholders |

---

## C4 Model Overview

**What is C4?**

The C4 model is a hierarchical set of software architecture diagrams for context, containers, components, and code. Each level zooms into the previous level for increasing detail.

**Levels:**
1. **Context** - System-level view (external dependencies, users)
2. **Container** - High-level architecture (applications, databases, services)
3. **Component** - Internal structure (classes, modules, pipelines)
4. **Code** - Implementation details (classes, methods, UML)

**Why C4?**
- **Hierarchical** - Each level builds on the previous
- **Standardized** - Common notation across teams
- **Scalable** - Works for small and large systems
- **Complementary** - Use with UML, ER diagrams, sequence diagrams

**Reference:** [C4 Model Documentation](https://c4model.com/)

---

## C4 Context Diagram

**File:** [c4_context.md](c4_context.md)

**Purpose:** System-level view showing NFL analytics system in context of external dependencies and users

**Key Elements:**
- **Users:** Bettor, ML Engineer, Data Scientist
- **NFL Analytics System:** Prediction API, Dashboard, Monitoring, Database, Cache
- **External Systems:** nflverse/nflfastR, The Odds API, Git LFS

**Key Insights:**
- System ingests data from nflverse (play-by-play, player stats)
- Betting lines fetched from The Odds API
- Models versioned in Git LFS (67+ files, up to 298 MB)
- 7,263 games, 1.25M plays in PostgreSQL database

**Audience:**
- Executives (understand system value and dependencies)
- Business stakeholders (understand data sources and costs)
- New team members (high-level system overview)

**View Diagram:** [c4_context.md](c4_context.md)

---

## C4 Container Diagram

**File:** [c4_container.md](c4_container.md)

**Purpose:** Docker services architecture showing 7 core containers and 3 optional profiles

**Key Elements:**
- **Data Layer:** PostgreSQL + TimescaleDB (port 5544), Redis (port 6379)
- **API Layer:** FastAPI (port 8000)
- **Presentation Layer:** Streamlit (port 8501)
- **Monitoring Layer:** Prometheus (port 9090), Grafana (port 3000)
- **Exporters:** Node Exporter (port 9100), Postgres Exporter (port 9187)
- **Optional Services:** MCP Server, Odds Fetcher, Alertmanager

**Key Insights:**
- All services run in single Docker network (`nfl-analytics`)
- FastAPI depends on PostgreSQL + Redis (health checks)
- Streamlit depends on FastAPI (prediction visualization)
- Grafana has 27-panel dashboard (System, DB, Redis, App metrics)

**Data Flows:**
- Prediction: Bettor → FastAPI → Redis (cache) → PostgreSQL (features) → Ensemble models → Response
- Monitoring: Exporters → Prometheus → Grafana → Dashboards
- Ingestion: nflverse → PostgreSQL → Materialized views

**Audience:**
- DevOps engineers (deployment, scaling, troubleshooting)
- ML Engineers (service integration, health monitoring)
- Infrastructure planners (resource allocation, costs)

**View Diagram:** [c4_container.md](c4_container.md)

---

## C4 Component Diagram - Ensemble

**File:** [c4_component_ensemble.md](c4_component_ensemble.md)

**Purpose:** Component-level view of 4-way ensemble prediction pipeline (v5.1)

**Key Elements:**
- **API Layer:** Predict Endpoint, Health Endpoint, Metrics Endpoint
- **Model Manager:** Ensemble Orchestrator, Cache Manager
- **4-Way Ensemble Models:**
  1. State-Space v2 (30% weight) - Random walk, 47.77 MAE
  2. Hierarchical v2 (30% weight) - Multilevel model, 48.2 MAE
  3. Hybrid XGBoost v1 (25% weight) - 70% static + 30% incremental, 38.03 MAE
  4. Informative Priors v2 (15% weight) - Domain-expert priors, 49.5 MAE
- **Post-Processing:** Weight Combiner, Temperature Scaler (T=0.5534), Uncertainty Aggregator
- **Feature Engineering:** Feature Fetcher, Feature Transformer, Feature Cache
- **R Bridge:** R Model Loader, R Predictor, R Post-Processor
- **Python Models:** XGBoost Loader, XGBoost Predictor

**Key Insights:**
- Ensemble combines 4 models using weighted average (30% + 30% + 25% + 15%)
- Temperature scaling calibrates 90% coverage from 78.5% to 90.2%
- R models (3/4) use rpy2 bridge for posterior sampling
- XGBoost provides point estimates (no distributional output)
- Total latency: 145ms (uncached), 5-20ms (cached)

**Prediction Flow:**
1. Request arrives → Cache check (5ms)
2. Cache miss → Fetch features from PostgreSQL (10-20ms)
3. Transform features (5ms)
4. Predict with 4 models (State-Space 50ms, Hierarchical 60ms, XGBoost 15ms, Priors 55ms)
5. Combine with weights (1ms)
6. Pool variance (1ms)
7. Apply temperature scaling (1ms)
8. Cache result (5ms)
9. Return JSON response (145ms total)

**Audience:**
- ML Engineers (model integration, performance tuning)
- Data Scientists (model development, algorithm understanding)
- Performance engineers (latency optimization, caching strategy)

**View Diagram:** [c4_component_ensemble.md](c4_component_ensemble.md)

---

## Missing Diagrams

**File:** [architecture_documentation_gaps.md](architecture_documentation_gaps.md)

**Purpose:** Identify missing diagrams and prioritize implementation

**High-Priority (1 week):**
1. **Data Ingestion Component Diagram** - nflverse → PostgreSQL ETL workflow
2. **Training Pipeline Component Diagram** - R + Python model training workflow
3. **Sequence Diagram (Prediction Flow)** - Step-by-step prediction request

**Medium-Priority (1 month):**
4. **Odds Polling Component Diagram** - The Odds API → PostgreSQL polling
5. **Entity Relationship Diagram (ERD)** - 56 database tables and relationships
6. **Deployment Diagrams (Dev + Prod)** - Docker Compose vs production architecture

**Low-Priority (As needed):**
7. **State-Space Code Diagram** - Detailed class diagram for state-space predictor
8. **Data Flow Diagram** - End-to-end data lineage
9. **Technology Stack Diagram** - Languages, frameworks, tools
10. **Security Architecture Diagram** - Threat model, controls, secrets management

**View Details:** [architecture_documentation_gaps.md](architecture_documentation_gaps.md)

---

## Diagram Conventions

### Naming Standards

**Consistent naming across all diagrams:**
- Models: "State-Space v2", "Hierarchical v2", "Hybrid XGBoost v1", "Informative Priors v2"
- Services: "FastAPI Service", "Streamlit Dashboard", "PostgreSQL + TimescaleDB"
- Ports: "Port 8000", "Port 5544"
- Files: Match exact file names (e.g., `simplified_random_walk_v2.rds`)

### Color Coding (Recommended)

**For visual clarity:**
- External systems: Light gray
- Data stores: Blue
- Compute components: Green
- Monitoring: Yellow
- User personas: Orange

**Note:** Mermaid C4 diagrams have limited color customization. PlantUML offers more flexibility.

### Labels

**Include critical details:**
- Ports (e.g., "Port 8000")
- Weights (e.g., "30% weight")
- Latencies (e.g., "145ms avg")
- Data volumes (e.g., "1.25M plays")
- Percentages (e.g., "70% static + 30% incremental")

---

## How to Use These Diagrams

### For Onboarding

**New ML Engineer:**
1. Start with Context Diagram (understand system boundaries)
2. Read Container Diagram (understand Docker services)
3. Study Component Diagram (understand ensemble architecture)
4. Dive into code (cross-reference diagram components with files)

**New Data Scientist:**
1. Start with Context Diagram (understand data sources)
2. Study Component Diagram (understand model pipeline)
3. Review Training Pipeline Diagram (once created)
4. Experiment with models (predict, retrain, evaluate)

**New DevOps Engineer:**
1. Start with Container Diagram (understand services)
2. Study Deployment Diagrams (once created)
3. Review docker-compose.yml (match diagram to code)
4. Deploy services (test health checks, monitor metrics)

---

### For Troubleshooting

**Slow predictions (>500ms):**
1. Check Container Diagram → Identify bottleneck service
2. Check Component Diagram → Identify slow model
3. Check Sequence Diagram (once created) → Find slow step
4. Optimize (cache, index, scale)

**Cache miss rate >50%:**
1. Check Container Diagram → Redis configuration
2. Check Component Diagram → Cache Manager logic
3. Increase TTL or Redis memory limit

**Database query timeout:**
1. Check Container Diagram → PostgreSQL configuration
2. Check ERD (once created) → Missing indexes
3. Add indexes on frequently queried columns

---

### For Architecture Reviews

**Quarterly review checklist:**
- [ ] Diagrams match current code (no drift)
- [ ] Version numbers updated (ensemble v5.1, etc.)
- [ ] New services documented (if added)
- [ ] Performance metrics updated (latencies, MAE)
- [ ] Missing diagrams identified and prioritized

**Who should attend:**
- ML Engineers (model architecture)
- DevOps Engineers (infrastructure)
- Data Scientists (algorithms)
- Product Manager (roadmap alignment)

---

## Rendering Diagrams

### GitHub Markdown

**Diagrams render natively in GitHub:**
- View `.md` files directly on GitHub
- Mermaid syntax auto-renders (no plugins needed)
- Zoom with browser zoom (Ctrl/Cmd + +/-)

**Example:** [https://github.com/raold/nfl-analytics/blob/main/docs/audits/2025_11_comprehensive_audit/06_architecture_diagrams/c4_context.md](https://github.com/raold/nfl-analytics/blob/main/docs/audits/2025_11_comprehensive_audit/06_architecture_diagrams/c4_context.md)

---

### Local Markdown Viewers

**Recommended tools:**
- **VS Code** - Install "Markdown Preview Mermaid Support" extension
- **Typora** - WYSIWYG Markdown editor with native Mermaid support
- **Obsidian** - Knowledge base with Mermaid plugin

**Setup (VS Code):**
1. Install extension: `bierner.markdown-mermaid`
2. Open `.md` file
3. Click "Open Preview" icon (top right)

---

### Mermaid Live Editor

**For editing/experimenting:**
- URL: https://mermaid.live/
- Copy Mermaid code from `.md` file
- Paste into editor
- Live preview updates automatically
- Export as PNG/SVG

**Use case:**
- Generating images for presentations
- Testing new diagram syntax
- Sharing with non-GitHub users

---

### PlantUML (Alternative)

**If Mermaid is too limited:**
- More flexible C4 support (colors, icons, shapes)
- Requires PlantUML server or local installation
- Not natively supported in GitHub Markdown

**Setup:**
1. Install PlantUML: `brew install plantuml` (Mac)
2. Create `.puml` file with C4 syntax
3. Generate PNG: `plantuml diagram.puml`
4. Embed PNG in Markdown: `![Diagram](diagram.png)`

**Reference:** [PlantUML C4 Extension](https://github.com/plantuml-stdlib/C4-PlantUML)

---

## Maintenance

### Update Triggers

**Update diagrams when:**
- New model added to ensemble
- Docker service added/removed
- Database schema changes (new tables, columns)
- Major performance optimizations
- Infrastructure changes (Redis cluster, load balancer)

### Review Cadence

**Quarterly review (every 3 months):**
- Review all diagrams for accuracy
- Update version numbers and dates
- Archive outdated diagrams (don't delete)
- Identify new diagram needs

**Ad-hoc updates:**
- After major code changes
- Before architecture presentations
- During onboarding sessions

### Ownership

**Who maintains diagrams:**
- **Context Diagram:** ML Engineer + Product Manager
- **Container Diagram:** DevOps Engineer + ML Engineer
- **Component Diagram:** ML Engineer + Data Scientist
- **Code Diagram:** Data Scientist

**Review process:**
1. Author creates/updates diagram
2. Peer review (2 reviewers)
3. Commit to Git (with version bump)
4. Announce in team chat (Slack, Discord)

---

## Related Documentation

### Architecture Documentation

- [ARCHITECTURE_REFERENCE.md](/docs/ARCHITECTURE_REFERENCE.md) - Text-based architecture overview
- [COMPONENT_DEEP_DIVES.md](/docs/COMPONENT_DEEP_DIVES.md) - Component usage guides
- [DOCKER_DASHBOARD_COMPLETE.md](/docs/deployment/DOCKER_DASHBOARD_COMPLETE.md) - Docker deployment guide

### Database Documentation

- [DATABASE_QUICK_REFERENCE.md](/docs/database/DATABASE_QUICK_REFERENCE.md) - Schema reference
- [DATABASE_SCHEMA_AUDIT.md](/docs/DATABASE_SCHEMA_AUDIT.md) - Full schema audit
- [database_erd.tex](/analysis/dissertation/figures/out/database_erd.tex) - ERD (LaTeX, outdated)

### Model Documentation

- [CLAUDE.md](/CLAUDE.md) - Comprehensive project documentation (4,909 lines)
- [MODEL_REGISTRY.yaml](/MODEL_REGISTRY.yaml) - Production model manifest
- [BNN_INVESTIGATION_COMPLETE.md](/docs/milestones/BNN_INVESTIGATION_COMPLETE.md) - BNN research journey

### Deployment Documentation

- [docker-compose.yml](/docker-compose.yml) - Docker Compose configuration
- [DOCKER_PHASE1_SETUP.md](/docs/DOCKER_PHASE1_SETUP.md) - Docker setup guide
- [MONITORING_STACK_COMPLETE.md](/MONITORING_STACK_COMPLETE.md) - Prometheus/Grafana setup

---

## Tools and Resources

### Mermaid Resources

- **Official Docs:** [https://mermaid.js.org/](https://mermaid.js.org/)
- **Live Editor:** [https://mermaid.live/](https://mermaid.live/)
- **Syntax Guide:** [https://mermaid.js.org/syntax/](https://mermaid.js.org/syntax/)

### C4 Model Resources

- **Official Site:** [https://c4model.com/](https://c4model.com/)
- **C4 + Mermaid:** [https://github.com/mermaid-js/mermaid/blob/develop/packages/mermaid/src/docs/syntax/c4c.md](https://github.com/mermaid-js/mermaid/blob/develop/packages/mermaid/src/docs/syntax/c4c.md)
- **C4 + PlantUML:** [https://github.com/plantuml-stdlib/C4-PlantUML](https://github.com/plantuml-stdlib/C4-PlantUML)

### Architecture Resources

- **Software Architecture Guide (Martin Fowler):** [https://martinfowler.com/architecture/](https://martinfowler.com/architecture/)
- **Arc42 Template:** [https://arc42.org/](https://arc42.org/) (comprehensive architecture documentation)
- **ADR (Architectural Decision Records):** [https://adr.github.io/](https://adr.github.io/)

---

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-14 | Claude Code (Sonnet 4.5) | Initial creation: Context, Container, Component (Ensemble), Gaps |

---

## Contact and Support

**For Questions:**
- Architecture reviews: Schedule quarterly meeting
- Diagram updates: Submit pull request
- New diagram requests: Create GitHub issue

**Maintainer:** Claude Code (Sonnet 4.5)

**Last Updated:** November 14, 2025

---

**Related Files:**
- [c4_context.md](c4_context.md) - System context diagram
- [c4_container.md](c4_container.md) - Docker services diagram
- [c4_component_ensemble.md](c4_component_ensemble.md) - Ensemble architecture diagram
- [architecture_documentation_gaps.md](architecture_documentation_gaps.md) - Missing diagrams and priorities
