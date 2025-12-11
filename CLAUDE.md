# CLAUDE.md - NFL Analytics Project Documentation

**Last Updated:** December 9, 2025 | **Version:** 5.9 | **Status:** Production Ready (Grade A+)

## Quick Status

| Component | State | Key Metric |
|-----------|-------|------------|
| Ensemble | v6.2 + equal-weight props | 54.5% spread (games), 44.25 MAE (props) |
| Features | 342 total (Phase 2 + Synthetic) | Base 305 + 37 synthetic/research |
| Database | 56 tables, 1.25M plays | PostgreSQL + Redis |
| Dissertation | 664 pages | Compiles cleanly |
| Infrastructure | 8/9 Docker services | ~105ms inference |

**Critical Files:** `py/ensemble/enhanced_ensemble_v6_production.py`, `py/ensemble/strategy/base.py`, `py/features/game_feature_extractor.py`

**Quick Links:** [DATABASE_QUICK_REFERENCE](docs/database/DATABASE_QUICK_REFERENCE.md) | [SQL_PATTERNS](docs/database/SQL_PATTERNS.md) | [ARCHITECTURE](docs/ARCHITECTURE_REFERENCE.md)

---

## Production Configuration

### Ensemble Weights

**v6.0 (Games):** State-Space 27%, Hierarchical 27%, XGBoost 23%, Informative Priors 13%, Hybrid 10%
- Use: Spreads, totals, win probabilities | Performance: 54.5% cover, +4.0% ROI

**v8.0 (Props):** Hierarchical 39.6%, Informative Priors 39.5%, Hybrid 10.1%, State-Space 8.2%, XGBoost 2.6%
- Use: Player props (yards, TDs, receptions) | Performance: 31.5% MAE improvement
- Learned via: Dirichlet grid search (108 configs, α₀=1.0, σ=20.0)

**Weight Selection:** `ensemble.use_default_weights()` (games) or `ensemble.use_dirichlet_weights()` (props)

**Why Static Weights?** Dynamic meta-learning tested and rejected (ADR-009). See [docs/adr/009-abandon-ensemble-v7-meta-learning.md](docs/adr/009-abandon-ensemble-v7-meta-learning.md)

**2025 Holdout Update (Dec 9):** Equal-weight averaging outperformed Dirichlet v8.0 on true out-of-sample 2025 data. Consider using simple average for props until more data confirms v8.0 generalization.

### Services

| Service | Port | Purpose |
|---------|------|---------|
| PostgreSQL | 5544 | `postgresql://dro:sicillionbillions@localhost:5544/devdb01` |
| Redis | 6379 | Prediction caching |
| FastAPI | 8000 | API endpoint |
| Streamlit | 8501 | Dashboard |
| Prometheus | 9090 | Metrics |
| Grafana | 3000 | Monitoring |

---

## Architecture

**Languages:** R (brms, Stan), Python (PyMC, XGBoost), SQL (PostgreSQL), LaTeX

**Key Directories:**
- `py/` - Python ML modules (causal, ensemble, RL, features)
- `R/` - Statistical models (brms, Stan, state-space)
- `analysis/dissertation/` - 667-page LaTeX dissertation
- `models/` - Trained artifacts (~60 files)
- `docs/` - 400+ documentation files

**Features (342 total):**

| Category | Count | Module |
|----------|-------|--------|
| TeamOffensive | 42 | `game_feature_extractor.py` |
| TeamDefensive | 38 | `game_feature_extractor.py` |
| NextGen | 28 | `nextgen_features.py` |
| QBR | 18 | `qbr_features.py` |
| OpponentAdjusted | 45 | `opponent_adjusted_features.py` |
| TurnoverLuck | 12 | `turnover_luck_features.py` |
| Primetime | 8 | `primetime_features.py` |
| Strength of Victory | 24 | `strength_of_victory_features.py` |
| PossessionValue | 16 | `possession_value_features.py` |
| Playcalling | 22 | `playcalling_features.py` |
| **Synthetic Interaction** | 11 | `synthetic_interaction_features.py` (NEW) |
| **Synthetic Momentum** | 11 | `synthetic_momentum_features.py` (NEW) |
| **Research-Driven** | 15 | `research_driven_features.py` (NEW) |
| Props Stability | N/A | `py/ensemble/props_stability_weights.py` (NEW) |

**Entry Points:**
- `GameFeatureExtractor` - Base 305 features (unified API, <500ms)
- `SyntheticInteractionEngine` - QB-OL synergy, defensive balance, explosive mismatch
- `SyntheticMomentumEngine` - Form intensity, pressure consistency, RZ efficiency
- `ResearchDrivenEngine` - nfelo EPA weighting (1.6:1.0), coaching rationality

---

## Database Schema & Data Layer

**Connection:** `postgresql://dro:sicillionbillions@localhost:5544/devdb01`

### Schema Overview

| Schema | Purpose | Key Tables |
|--------|---------|------------|
| `public` | Core NFL data | `games`, `plays`, `players`, `rosters`, `nextgen_*` |
| `mart` | Aggregated analytics | `team_offensive_stats`, `team_defensive_stats`, `player_game_stats` |
| `ml` | ML pipeline tables | `game_predictions`, `model_registry`, `feature_store` |
| `metadata` | System metadata | `table_documentation`, `column_statistics` |

### Critical Tables

| Table | Rows | Key Columns | Notes |
|-------|------|-------------|-------|
| `games` | 6,991 | `game_id`, `season`, `week`, `kickoff` | Use `kickoff` not `game_date` |
| `plays` | 1.25M | `game_id`, `play_id`, `epa`, `wpa` | Core analytics source |
| `nextgen_passing` | 25,677 | `player_id`, `season`, `week` | Tracking data (2016+) |
| `participation` | 433,805 | `game_id`, `play_id`, `player_id` | Player-play mapping |

### Column Naming Conventions

| Correct | WRONG | Notes |
|---------|-------|-------|
| `kickoff` | `game_date` | Timestamp field for game time |
| `spread_close` | `spread_line` | Closing spread value |
| `quarter` | `qtr` | Quarter number (1-5) |
| `pass` | `pass_attempt` | Boolean pass play indicator |
| `posteam` | `team` | Possession team |
| `defteam` | `opponent` | Defensive team |

### Team Abbreviations
**IMPORTANT:** Database uses nflverse format, models accept both:
- Rams: Database uses `LA`, canonical is `LAR` - **BOTH are valid**
- Chargers: `LAC` in both
- Raiders: Database uses `LV` (current), `OAK` (historical)
- Washington: `WAS` (current)
- Use `py.utils.team_mapping.normalize_team()` for conversions

### SCHEMA-FIRST DISCIPLINE
- `mart.player_game_stats` has NO `game_id` column - use `(season, week, team)` composite key
- Always check schema before writing SQL: `docs/database/DATABASE_QUICK_REFERENCE.md`
- Validate queries: `python py/utils/sql_validator.py "YOUR QUERY"`

### Data Ingestion Pipelines

| Pipeline | Source | Schedule | Script |
|----------|--------|----------|--------|
| NFL play-by-play | nflreadr | Weekly (Tue) | `R/ingestion/ingest_current_season.R` |
| Odds data | The Odds API | 3x daily | `py/ingest_odds_smart.py` |
| Injuries | nflreadr | Weekly | `R/ingestion/ingest_injuries.R` |
| Materialized views | Internal | Daily 5am | `scripts/refresh_marts.py` |

### Quick Schema Queries
```sql
-- List all columns in a table
SELECT column_name, data_type FROM information_schema.columns
WHERE table_name = 'plays' ORDER BY ordinal_position;

-- Check if column exists
SELECT EXISTS (SELECT 1 FROM information_schema.columns
WHERE table_name = 'plays' AND column_name = 'epa');
```

---

## Development Workflow

```bash
# Setup
uv venv .venv && uv pip install -r requirements.txt
Rscript -e 'renv::restore()'
bash scripts/dev/init_dev.sh

# Testing
pytest tests/unit -m unit
pytest tests/integration -m integration

# Pre-commit (enforced)
pre-commit install
```

**Conventions:** Never commit secrets, test before commit, follow conventional commits, main branch: `main`

---

## Agent Infrastructure

**19 specialized agents across 5 phases.** See [docs/AGENT_INFRASTRUCTURE.md](docs/AGENT_INFRASTRUCTURE.md)

| Phase | Purpose | Key Agents |
|-------|---------|------------|
| 1: Security | Prevent vulnerabilities | Code Reviewer, Security Auditor, Data Leakage Detector |
| 2: Performance | MCMC tuning, testing | Performance Optimizer, Test Generator, Model Validator |
| 3: Production | Multi-language coordination | Ensemble Orchestrator, Betting Strategy Validator |
| 4: Infrastructure | Pre-backtest validation | Ensemble Health Monitor, Model Registry Auditor |
| 5: Production Infra | Schema migrations, caching | Database Migration, Prediction Cache, A/B Test Monitor |

**ROI:** $90k-165k estimated annual value, 100+ hours saved

---

## Enforcement Framework

**Purpose:** Catch errors at commit time through automated validation routing.

| Rule | Level | Prevents |
|------|-------|----------|
| SQL Validation | BLOCKING | Schema errors |
| Backtest Health | BLOCKING | R integration failures |
| Model Training | BLOCKING | Data leakage |
| Security Audit | BLOCKING | Hardcoded secrets |

**Usage:** Automatic via pre-commit hooks. Manual: `python py/utils/enforcement_router.py <file>`

See [docs/ENFORCEMENT_FRAMEWORK_COMPLETE.md](docs/ENFORCEMENT_FRAMEWORK_COMPLETE.md)

---

## Dissertation Standards

**Non-Negotiable:** Statistical rigor (R-hat < 1.01, ESS > 400), narrative arc, publication-ready visuals, reproducibility, zero TODOs/placeholders.

See [docs/guides/DISSERTATION_STANDARDS.md](docs/guides/DISSERTATION_STANDARDS.md)

---

## Code Style

**Python:** Black formatter, type hints, Google-style docstrings, 100 char max
**R:** tidyverse style, `<-` for assignment, explicit `library()` calls
**SQL:** UPPERCASE keywords, lowercase table/column names, indent subqueries

---

## Current Status (Dec 9, 2025)

**Completed:**
- 2025 Holdout Validation: 74.9% spread accuracy, 44.25 MAE props (Dec 9)
- Phase 2 Advanced Stats: 120 new features
- Dirichlet Optimization: Equal weights outperform v8.0 on holdout
- **Synthetic Features (Dec 9):** 37 new features across 3 modules
  - Interaction features: QB-OL synergy, defensive balance, explosive mismatch
  - Momentum features: Form intensity, pressure consistency, RZ efficiency
  - Research-driven: nfelo EPA weighting (1.6:1.0), coaching rationality
- **Props Stability Weighting (Dec 9):** SumerSports YoY correlation research integrated

**Key Findings (Dec 9):**
- Equal-weight ensemble beats Dirichlet v8.0 on 2025 holdout (44.25 vs 44.91 MAE)
- Offensive EPA is 60% more predictive than defensive EPA (nfelo research)
- RB stats have ~0 YoY correlation (downweight in props models)

**In Progress:**
- XGBoost v5 retraining with 342-feature set
- Ensemble v6.2 integration with synthetic features

**Strategic Decisions:**
- BNN research abandoned (Nov 9) - 84 hours, 6 approaches, only BNN v3.1 Rushing remains
- Dynamic meta-learning rejected (Nov 18) - static weights outperform, see ADR-009
- Dual-weight strategy adopted (Nov 27) - different domains need different weights

**Recent Work:** [docs/milestones/RECENT_WORK_HISTORY.md](docs/milestones/RECENT_WORK_HISTORY.md)

---

## Documentation Index

| Category | Key Docs |
|----------|----------|
| Quick Start | README.md, SETUP.md, CONTRIBUTING.md |
| Database | [DATABASE_QUICK_REFERENCE](docs/database/DATABASE_QUICK_REFERENCE.md), [SQL_PATTERNS](docs/database/SQL_PATTERNS.md) |
| Architecture | [ARCHITECTURE_REFERENCE](docs/ARCHITECTURE_REFERENCE.md), [COMPONENT_DEEP_DIVES](docs/COMPONENT_DEEP_DIVES.md) |
| Milestones | [RECENT_WORK_HISTORY](docs/milestones/RECENT_WORK_HISTORY.md), [docs/milestones/](docs/milestones/) |
| Synthesis | [DIRICHLET_OPTIMIZATION](docs/synthesis/DIRICHLET_WEIGHT_OPTIMIZATION_SYNTHESIS_NOV27.md) |
| ADRs | [docs/adr/](docs/adr/) (009: Dynamic weights rejection) |

---

## AI Assistant Guidelines

**Do:**
1. Check PROJECT_STATUS.md for recent updates
2. Follow naming/style conventions
3. Test before committing
4. Update relevant docs
5. Ask when uncertain

**Don't:**
1. Skip errors - ALWAYS stop and fix or ask
2. Commit secrets
3. Skip tests or use --no-verify
4. Break LaTeX compilation
5. Modify pgdata/ (Docker-managed)

**Preferred:** Modular code, explicit naming, fail fast, document why not what, reproducible analysis

---

## Lessons Learned

1. Always check factor levels before model training
2. Dead code can break models - remove unused variables
3. Model file corruption occurs on failure - delete before retraining
4. Know when to abandon research lines (BNN: 84 hours, 6 approaches)
5. Always validate baselines independently
6. **Schema-first discipline** - `mart.player_game_stats.game_id` error occurred DOZENS of times

---

**Version:** 5.9 | **Maintainer:** Claude Code | **Repository:** github.com/raold/nfl-analytics

**Project Rules:**
- Never take shortcuts, never use synthetic data
- Survey, read, audit, synthesize before writing code
- When doing git commit NEVER add co-authorship
