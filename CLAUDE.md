# CLAUDE.md - NFL Analytics Project Documentation

**Last Updated:** February 13, 2026 | **Version:** 8.0 | **Status:** Production Ready (2025 Season Complete + Kalshi Trading)

## Quick Status

| Component | State | Key Metric |
|-----------|-------|------------|
| Ensemble | v6.2 + equal-weight props | 54.5% spread (games), 44.25 MAE (props) |
| Features | 386 active (Kalman/GP disabled) | Base 305 + 37 synthetic + 38 semantic + 6 rest |
| Semantic Stack | Complete (Dec 21) | Ollama/MLX inference, 38 LLM features (17/team + 4 matchup) |
| **Game Outcome Model** | **v2 PRODUCTION (Dec 23)** | **382 features, 11.05 spread MAE (full 2025), 56.5% win prob** |
| **Feedback Loop** | **NEW (Dec 22)** | **Weekly learning system, 10 modules** |
| Database | 50 tables, 1.25M plays | PostgreSQL + Redis (custom AVG/SUM boolean aggregates) |
| **Dissertation** | **Updated (Feb 12)** | **753 pages, postseason validation added, 0 undefined refs** |
| **2025 Season** | **COMPLETE (Feb 2026)** | **285 games evaluated, -3.2% ATS ROI, market efficiency confirmed** |
| **Kalshi Trading System** | **Complete (Feb 13)** | **6 phases, 437 tests, 37 files** |
| Infrastructure | 8/9 Docker services | ~105ms inference |

**Critical Files:** `nfl_analytics/ensemble/enhanced_ensemble_v6_production.py`, `nfl_analytics/ensemble/strategy/base.py`, `nfl_analytics/features/game_feature_extractor.py`, `nfl_analytics/feedback/`

**Quick Links:** [DATABASE_QUICK_REFERENCE](docs/database/DATABASE_QUICK_REFERENCE.md) | [SQL_PATTERNS](docs/database/SQL_PATTERNS.md) | [ARCHITECTURE](docs/architecture/ARCHITECTURE_REFERENCE.md)

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

### Docker Services (8 containers)

| Service | Container | Port | Purpose |
|---------|-----------|------|---------|
| PostgreSQL | nfl-analytics-pg-1 | 5544 | `postgresql://dro:sicillionbillions@localhost:5544/devdb01` |
| Redis | nfl-analytics-redis | 6379 | Prediction caching |
| FastAPI | nfl-analytics-api | 8000 | API endpoint |
| Streamlit | nfl-analytics-streamlit | 8501 | Dashboard |
| Prometheus | nfl-analytics-prometheus | 9090 | Metrics |
| Grafana | nfl-analytics-grafana | 3000 | Monitoring |
| Node Exporter | nfl-analytics-node-exporter | 9100 | Host metrics |
| Postgres Exporter | nfl-analytics-postgres-exporter | 9187 | DB metrics |

**Docker Networking:** Use `DB_HOST=host.docker.internal` from containers, `localhost` from host.

### Python Environment (uv)

```bash
uv sync                    # Install all dependencies from pyproject.toml
uv pip install <pkg>       # Add package
uv pip list               # List installed packages (79+ packages)
```

**Key:** pyproject.toml manages dependencies, uv.lock pins versions.

---

## Architecture

**Languages:** R (brms, Stan), Python (PyMC, XGBoost), SQL (PostgreSQL), LaTeX

**Key Directories:**
- `nfl_analytics/` - Python ML modules (causal, ensemble, RL, features)
- `R/` - Statistical models (brms, Stan, state-space)
- `analysis/dissertation/` - 693-page LaTeX dissertation
- `models/` - Trained artifacts (~60 files)
- `docs/` - 400+ documentation files

**Features (386 active, Kalman/GP disabled):**

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
| **Synthetic Interaction** | 11 | `synthetic_interaction_features.py` |
| **Synthetic Momentum** | 11 | `synthetic_momentum_features.py` |
| **Research-Driven** | 15 | `research_driven_features.py` |
| **Semantic (LLM)** | 38 | `nfl_analytics/semantic/features/semantic_features.py` (17/team × 2 + 4 matchup) |
| ~~Kalman EPA~~ | ~~18~~ | `nfl_analytics/features/kalman_epa_filter.py` (**DISABLED** ADR-015) |
| ~~GP Injury Recovery~~ | ~~17~~ | `nfl_analytics/features/gp_injury_recovery.py` (**DISABLED** ADR-015) |
| Props Stability | N/A | `nfl_analytics/ensemble/props_stability_weights.py` |

**Entry Points:**
- `GameFeatureExtractor` - Base 305 + 37 synthetic + 6 rest + 38 semantic = 386 features (unified API, <500ms)
- `SyntheticInteractionEngine` - QB-OL synergy, defensive balance, explosive mismatch
- `SyntheticMomentumEngine` - Form intensity, pressure consistency, RZ efficiency
- `ResearchDrivenEngine` - nfelo EPA weighting (1.6:1.0), coaching rationality
- `SemanticFeatureExtractor` - LLM-derived sentiment, injury severity, media signals (NEW)
- ~~`KalmanEPAFilter`~~ - Team EPA regime detection with CUSUM (**DISABLED** - negative signal)
- ~~`InjuryRecoveryGP`~~ - Uncertainty-aware injury recovery curves (**DISABLED** - GP never trains)

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
- Use `nfl_analytics.utils.team_mapping.normalize_team()` for conversions

### SCHEMA-FIRST DISCIPLINE
- `mart.player_game_stats` has NO `game_id` column - use `(season, week, team)` composite key
- Always check schema before writing SQL: `docs/database/DATABASE_QUICK_REFERENCE.md`
- Validate queries: `python nfl_analytics/utils/sql_validator.py "YOUR QUERY"`

### Data Ingestion Pipelines

| Pipeline | Source | Schedule | Script |
|----------|--------|----------|--------|
| NFL play-by-play | nflreadr | Weekly (Tue) | `R/ingestion/ingest_current_season.R` |
| Odds data | The Odds API | 3x daily | `nfl_analytics/ingest_odds_smart.py` |
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

### Git Worktrees

The project uses git worktrees for parallel development workflows:

| Worktree | Directory | Branch | Purpose |
|----------|-----------|--------|---------|
| Main | `nfl-analytics/` | `main` | Primary development |
| Experiments | `../nfl-experiments/` | `experiments` | Model experiments, XGBoost tuning |
| Dissertation | `../nfl-dissertation/` | `dissertation` | Isolated LaTeX compilation |
| Hotfix | `../nfl-hotfix/` | `hotfix` | Quick production fixes |
| Backtest | `../nfl-backtest/` | `backtest` | Long-running backtests |

```bash
# List all worktrees
git worktree list

# Remove a worktree when done
git worktree remove ../nfl-hotfix

# Prune stale references
git worktree prune
```

**Note:** Each worktree needs its own `.venv` for Python dependencies.

---

## Agent Infrastructure

**19 specialized agents across 5 phases.** See [docs/agents/AGENT_INFRASTRUCTURE.md](docs/agents/AGENT_INFRASTRUCTURE.md)

| Phase | Purpose | Key Agents |
|-------|---------|------------|
| 1: Security | Prevent vulnerabilities | Code Reviewer, Security Auditor, Data Leakage Detector |
| 2: Performance | MCMC tuning, testing | Performance Optimizer, Test Generator, Model Validator |
| 3: Production | Multi-language coordination | Ensemble Orchestrator, Betting Strategy Validator |
| 4: Infrastructure | Pre-backtest validation | Ensemble Health Monitor, Model Registry Auditor |
| 5: Production Infra | Schema migrations, caching | Database Migration, Prediction Cache, A/B Test Monitor |

**ROI:** $90k-165k estimated annual value, 100+ hours saved

---

## Kalshi Trading System (Feb 2026)

**Status:** All 6 phases complete, 437 unit tests passing, ~9,200 LOC across 37 files

### Architecture

```
Pipeline: MarketData → Strategy → RiskManager → TradeExecutor → Exchange
                                                      ↓
                                    PnLTracker ← DrawdownMonitor
                                         ↓
                                  TradeLogger + AlertManager
```

### Module Map

| Module | Files | LOC | Purpose |
|--------|-------|-----|---------|
| `nfl_analytics/trading/` | 6 | ~3,400 | Strategies (directional, market maker, hybrid) |
| `nfl_analytics/markets/` | 8 | ~3,500 | MarketClient ABC, Kalshi API, paper trading, market router |
| `nfl_analytics/risk/` | 9 | ~4,000 | Kill switch, drawdown, position limits, feature health, CVaR |
| `nfl_analytics/execution/` | 8 | ~2,800 | Trade executor, live trader, reconciliation, execution models |
| `nfl_analytics/monitoring/` | 6 | ~3,200 | PnL tracker, trade logger, alert manager, calibration |

### Key Classes & Entry Points

- `TradingStrategy` (ABC) → `DirectionalStrategy`, `MarketMaker`, `HybridStrategy`
- `MarketClient` (ABC) → `KalshiClient` (live), `PaperClient` (sim)
- `RiskManager.check_signal()` → gates ALL signals through risk hierarchy
- `LiveTrader.run()` → main async orchestration loop
- `MarketRouter.get_strategy()` → multi-sport contract routing

### Risk Layers (priority order)

1. Feature health (`FeatureHealthChecker`) — blocks on stale/missing data
2. Kill switch (`KillSwitch`) — daily loss 10%, drawdown 20%, API errors 10 consecutive
3. Position limits (`PositionLimitChecker`) — per-contract, per-side, correlation groups

### Kalshi Auth

Environment variables: `KALSHI_USER_ID`, `KALSHI_API_KEY`, `KALSHI_PRIVATE_KEY_PATH`
Fallback: `~/.kalshi/private_key.pem` (RSA-PSS)

### Running Tests

```bash
uv run pytest tests/unit/risk/ tests/unit/trading/ tests/unit/markets/ tests/unit/execution/ tests/unit/monitoring/ -o "addopts=" -q
```

### Phase History

| Phase | Scope | Tests |
|-------|-------|-------|
| 1 | Foundation (base.py, signal, strategy ABC, paper client) | 67 |
| 2 | NFL Strategy (directional, risk hierarchy, drawdown) | 23 |
| 3 | Kalshi Live (auth, client, schemas, websocket) | 99 |
| 4 | Market Making (market_maker, hybrid) | 52 |
| 5 | Monitoring & Safety (PnL, logger, alerts, live_trader) | 69 |
| 6 | Multi-Sport (market_router) | 15 |

---

## Semantic Stack (LLM Integration)

**Status:** Phase 1 Complete (Dec 21, 2025)
**Goal:** Add AI/LLM capabilities to push win rate past 52.4% breakeven toward 56-57%

### Architecture

```
nfl_analytics/semantic/
├── models/           # LLM inference clients
│   ├── ollama_client.py   # qwen3:8b (chat endpoint, 240s timeout)
│   ├── mlx_client.py      # Apple Silicon (Qwen2.5-7B-Instruct-4bit)
│   └── inference.py       # Unified client with fallback chain
├── ingestion/        # Data pipelines
│   ├── news_ingester.py   # ESPN news classification
│   └── social_ingester.py # Twitter/X verified accounts
├── features/         # Feature engineering
│   └── semantic_features.py  # 26 LLM-derived features
└── training/         # Model fine-tuning
    ├── annotation_pipeline.py  # DB → training data
    └── fine_tune.py           # Ollama/MLX fine-tuning
```

### Semantic Features (38 total, prefix: `sem_`)

**Per-Team (17 each = 34 total):**
| Feature | Type | Description |
|---------|------|-------------|
| `sentiment_24h` | float | News/social sentiment (-1 to 1) |
| `sentiment_7d_trend` | float | Week-over-week sentiment change |
| `sentiment_confidence` | float | LLM confidence in sentiment assessment |
| `injury_mentions` | float | Count of injury mentions in sources |
| `injury_severity_llm` | int | LLM-assessed injury impact (1-5) |
| `key_player_risk` | float | Star player injury probability |
| `media_buzz_volume` | float | Media attention (0-1 normalized) |
| `media_buzz_trend` | float | Media attention trend |
| `coaching_change` | bool | Recent coaching staff changes |
| `scheme_mentions` | float | Scheme/play-calling discussion count |
| `public_overreaction` | float | Market sentiment extremity |
| `injury_impact_decay` | float | ODE-inspired exponential injury decay |
| `injury_velocity` | float | Rate of change in injury severity |
| `recovery_momentum` | float | Injury recovery trajectory |
| `sentiment_velocity` | float | Rate of change in sentiment |
| `sentiment_acceleration` | float | Second derivative of sentiment |
| `media_half_life` | float | Decay rate of media attention |

**Matchup (4):**
| Feature | Description |
|---------|-------------|
| `sentiment_differential` | Home - away sentiment gap |
| `injury_advantage` | Relative injury situation |
| `media_attention_ratio` | Coverage imbalance |
| `narrative_momentum` | Sentiment trend differential |

### Inference Configuration

| Setting | Value | Notes |
|---------|-------|-------|
| Primary Model | qwen3:8b (Ollama) | Chat endpoint, not generate |
| Fallback | MLX → Gemini → Claude | Automatic failover |
| Timeout | 240 seconds | 4 min for complex queries |
| Caching | Content-hash based | Redis integration |

### Training Data (Dec 21, 2025)

| Dataset | Samples | Source |
|---------|---------|--------|
| NER (entities) | 5,000 | 84K+ injury records (2009-2025) |
| Sentiment | 2,000 | Game outcomes (2022-2025) |

**Files:** `data/semantic_training/` (train.jsonl, Modelfile, ner/sentiment splits)

### Usage

```python
# Semantic features auto-integrated into GameFeatureExtractor
from nfl_analytics.features.game_feature_extractor import GameFeatureExtractor
extractor = GameFeatureExtractor()
features = extractor.extract_game_features(game_id)
# Access: features['sem_home_sentiment_24h'], features['sem_injury_advantage'], etc.

# Value bet detection with LLM explanations
from nfl_analytics.production.value_bet_detector import ValueBetDetector
detector = ValueBetDetector(min_edge=0.03, use_llm_explanations=True)
bets = detector.detect_value_bets(predictions)
```

---

## Feedback Loop System (Weekly Learning)

**Status:** Complete (Dec 28, 2025) - includes WHY System
**Goal:** Close the loop between predictions and outcomes, enabling continuous learning and LLM fine-tuning

### Architecture

```
nfl_analytics/feedback/
├── outcome_updater.py           # Phase 6.1: Update actuals after games
├── semantic_attribution.py      # Phase 6.2: Track feature contribution (True SHAP)
├── ensemble_weight_adjuster.py  # Phase 6.3: Adjust weights on performance
├── retrain_trigger.py           # Phase 6.4: Detect retraining needs
├── annotation_generator.py      # Phase 6.5: Generate LLM training data
├── meta_learner.py              # Phase 7.1: Bidirectional Model↔Semantic learning
├── feature_memory.py            # Phase 7.2: Rolling feature performance tracking
├── auto_feature_selector.py     # Phase 7.3: Automatic feature enable/disable
├── prompt_evolver.py            # Phase 7.4: Evolve LLM prompts from outcomes
├── semantic_ab_test.py          # Phase 8.1: A/B test semantic vs non-semantic
├── model_error_decomposition.py # Phase 9.1: WHY - Error attribution per model
├── temporal_causality.py        # Phase 9.2: WHY - Trace prediction chain
├── feature_interactions.py      # Phase 9.3: WHY - Semantic × base synergies
└── __init__.py                  # Module exports (13 components)
```

### Components Summary

| Phase | Component | Purpose |
|-------|-----------|---------|
| 6.1 | WeeklyOutcomeUpdater | Populate `ml.game_predictions.actual_*` after games complete |
| 6.2 | SemanticAttributionTracker | SHAP-like attribution for semantic features |
| 6.3 | EnsembleWeightAdjuster | Inverse-MAE weighted adjustment with shrinkage |
| 6.4 | RetrainTrigger | Detect MAE degradation, accuracy drop, calibration drift |
| 6.5 | AnnotationGenerator | Generate JSONL for LLM fine-tuning (sentiment + NER) |
| 7.1 | SemanticMetaLearner | Bidirectional: Model→Semantic and Semantic→Model |
| 7.2 | FeaturePerformanceMemory | Rolling 8-week memory of feature contributions |
| 7.3 | AutoFeatureSelector | States: probation → active → disabled |
| 7.4 | PromptEvolver | Evolve LLM prompts based on prediction outcomes |
| 8.1 | SemanticABTest | McNemar test for paired accuracy comparison |
| 9.1 | ModelErrorDecomposer | C_i = w_i × (Y - Ŷ_i) per-model error contribution |
| 9.2 | TemporalCausalityTracker | Trace News→Sentiment→Feature→Prediction→Outcome |
| 9.3 | FeatureInteractionAnalyzer | SHAP interaction values for semantic × base synergies |

### Database Tables (11 new)

| Table | Purpose |
|-------|---------|
| `ml.feedback_runs` | Weekly orchestration runs + summaries |
| `ml.semantic_attribution` | Per-game feature attribution scores |
| `ml.feature_performance` | Rolling feature contribution tracking |
| `ml.feature_selection_state` | Feature status (active/disabled/probation) |
| `ml.prompt_versions` | LLM prompt versioning for evolution |
| `ml.prompt_ab_tests` | Prompt A/B test configurations |
| `ml.ab_test_status` | A/B test metadata and state |
| `ml.ab_test_results` | Per-game control vs treatment predictions |
| `ml.model_error_decomposition` | Per-model error contributions (WHY) |
| `ml.causal_chains` | Temporal causality tracking (WHY) |
| `ml.feature_interactions` | Semantic × base feature synergies (WHY) |

### Weekly Workflow

```bash
# TUESDAY (after Week N games complete)
python scripts/run_weekly_feedback.py --season 2025 --week N

# What it does:
# 1. Update ml.game_predictions with actual scores
# 2. Compute semantic feature attribution (SHAP-like)
# 3. Check retrain triggers (alerts if fired)
# 4. Generate annotations for LLM fine-tuning
# 5. Log summary to ml.feedback_runs

# Options:
python scripts/run_weekly_feedback.py --season 2025 --week 16 --dry-run
python scripts/run_weekly_feedback.py --season 2025 --week 16 --auto-adjust
```

### A/B Testing (2025 Holdout)

```bash
# Compare semantic vs non-semantic on 2025 holdout
python scripts/ab_test_semantic_2025.py --weeks 1-16

# Output:
# ============================================================
# 2025 HOLDOUT A/B TEST: Semantic Features
# ============================================================
# Games analyzed: 195 (Weeks 1-16)
#
# SPREAD PREDICTION:
#   Control (no semantic):  XX.X% accuracy
#   Treatment (semantic):   XX.X% accuracy
#   Lift: +X.X%
#   McNemar p-value: 0.XXX (significant at α=0.05)
```

### Usage

```python
# Weekly feedback orchestration
from nfl_analytics.feedback import WeeklyFeedbackOrchestrator
orchestrator = WeeklyFeedbackOrchestrator()
result = orchestrator.run(season=2025, week=16)
if result.triggers_fired:
    print(f"Retraining recommended: {result.triggers_fired}")

# Meta-learning insights
from nfl_analytics.feedback import SemanticMetaLearner
learner = SemanticMetaLearner()
result = learner.update_from_outcomes(season=2025, week=16)
for insight in result.feature_insights:
    print(f"{insight.feature_name}: {insight.recommendation}")

# Automatic feature selection
from nfl_analytics.feedback import AutoFeatureSelector
selector = AutoFeatureSelector()
selection = selector.evaluate_features(season=2025, week=16)
selector.apply_selection(selection)  # Updates ml.feature_selection_state

# A/B testing
from nfl_analytics.feedback import SemanticABTest
test = SemanticABTest()
test_id = test.setup_test(season=2025, start_week=1)
results = test.analyze_results(test_id)
test.print_results(results)
```

### Feature Lifecycle

```
NEW FEATURE
    │
    ▼
┌─────────────┐  positive contribution  ┌─────────────┐
│  PROBATION  │ ───────────────────────▶│   ACTIVE    │
│  (weight=0.5)│  (2+ weeks strong)     │ (weight=1.0)│
└─────────────┘                         └─────────────┘
    │                                        │
    │ negative contribution                  │ moderate negative
    │ (early detection)                      │ (4+ weeks)
    ▼                                        ▼
┌─────────────┐◀────────────────────────┌─────────────┐
│  DISABLED   │    demotion path        │  PROBATION  │
│ (weight=0.0)│                         │  (review)   │
└─────────────┘                         └─────────────┘
```

---

## Enforcement Framework

**Purpose:** Catch errors at commit time through automated validation routing.

| Rule | Level | Prevents |
|------|-------|----------|
| SQL Validation | BLOCKING | Schema errors |
| Backtest Health | BLOCKING | R integration failures |
| Model Training | BLOCKING | Data leakage |
| Security Audit | BLOCKING | Hardcoded secrets |

**Usage:** Automatic via pre-commit hooks. Manual: `python nfl_analytics/utils/enforcement_router.py <file>`

See [docs/agents/ENFORCEMENT_FRAMEWORK.md](docs/agents/ENFORCEMENT_FRAMEWORK.md)

---

## Dissertation Standards

**Non-Negotiable:** Statistical rigor (R-hat < 1.01, ESS > 400), narrative arc, publication-ready visuals, reproducibility, zero TODOs/placeholders.

See [docs/guides/DISSERTATION_STANDARDS.md](docs/guides/DISSERTATION_STANDARDS.md)

---

## LaTeX Tooling & Workflow

**Location:** `analysis/dissertation/main/main.tex` (753 pages)

### Workflow Scripts (`~/.local/bin/diss-*`)

| Command | Alias | Purpose |
|---------|-------|---------|
| `diss-watch` | `dw` | Continuous compilation with `latexmk -pdf -pvc` (recompiles on save) |
| `diss-diff` | `dd` | Generate tracked-changes PDF via `latexdiff-vc` for advisor review |
| `diss-stats` | `ds` | Word count (`texcount`), page count, font embedding check |
| `diss-compress` | `dc` | Compress PDF via Ghostscript (ebook/prepress/screen quality) |
| `diss-clean` | `dclean` | Remove auxiliary files (`.aux`, `.log`, `.bbl`, etc.) |
| `diss-lint` | `dl` | LaTeX linting via `chktex` and `lacheck` |
| `diss-figures` | `df` | List/crop/optimize figures (`pdfcrop`, `optipng`) |
| `diss-chapter` | `dchap` | Per-chapter compilation (fast feedback, chapter-specific PDFs) |

**Environment:** `DISS_DIR` points to dissertation main directory (set in `.zshrc`)

### Available LaTeX Tools

**Engines:** `latex`, `pdflatex`, `xelatex`, `lualatex`, `context`, `tex`
**Build Systems:** `latexmk`, `arara`, `rubber`, `texliveonfly`
**Bibliography:** `bibtex`, `biber`, `bibtex8`, `bib2bib`, `bibexport`, `bibclean`
**Linting/QA:** `chktex`, `lacheck`, `latexindent`, `latexdiff`, `latexdiff-vc`, `detex`, `texcount`, `texlogsieve`
**Indexing:** `makeindex`, `xindy`, `makeglossaries`, `splitindex`
**DVI/PS:** `dvips`, `dvipdf`, `dvipdfm`, `dvipdfmx`, `ps2pdf`, `pdf2ps`
**PDF Tools:** `pdfcrop`, `pdfjam`, `pdfinfo`, `pdftotext`, `pdftoppm`, `pdfimages`, `pdffonts`, `pdfseparate`, `pdfunite`, `pdftk`, `qpdf`, `pdfannotextractor`
**Ghostscript:** `gs`, `ghostscript`, `mutool`
**Graphics:** `epstopdf`, `eps2pdf`, `svg2pdf`, `img2pdf`, `convert`, `magick`, `inkscape`, `mpost`, `metapost`, `ltximg`
**Image Optimization:** `optipng`, `pngquant`, `jpegoptim`
**TeX Live:** `tlmgr`, `texhash`, `mktexlsr`, `fmtutil`, `updmap`, `kpsewhich`, `texdoc`
**Conversion:** `pandoc`
**Viewers:** `zathura`, `evince`, `okular`, `mupdf`, `xdg-open`

### Common Workflows

```bash
# Per-chapter compilation (FAST - seconds vs minutes)
dchap list                      # Show all chapters with build status
dchap build 3                   # Build chapter 3 → chapter_03_data_foundation.pdf
dchap watch 5                   # Watch chapter 5, recompile on save
dchap clean 3                   # Clean chapter 3 build files

# Start continuous compilation (Terminal 1)
dw                              # or: diss-watch (full dissertation)

# Check stats anytime (Terminal 2)
ds                              # Word count, pages, font check

# Before sending to advisor
dd HEAD~5                       # Diff against 5 commits ago
# Output: ~/.diffs/diff_HEAD~5_<timestamp>.pdf

# Before final submission
ds                              # Verify all fonts embedded
dc prepress                     # Compress with high quality
# Output: main-prepress.pdf

# Clean up after compilation
dclean                          # Remove aux files
dclean --all                    # Also remove PDF

# Manage figures
df list                         # Show all figures with sizes
df crop figure.pdf              # Crop whitespace from single PDF
df crop-all                     # Crop all PDFs in figures/
df optimize                     # Optimize PNG file sizes

# Filter noisy LaTeX logs
latexmk -pdf main.tex 2>&1 | texlogsieve
```

### Quick LaTeX Aliases (in `.zshrc`)

```bash
lmk                             # latexmk -pdf
lmkc                            # latexmk -c (clean aux)
lmkca                           # latexmk -C (clean all)
texwc                           # texcount -inc -total
pdfpg                           # pdfinfo (shows page count)
```

---

## Dissertation Structure & File Map

**Root:** `analysis/dissertation/` (753 pages, 62 figures, 313 tables)
**Main Entry:** `analysis/dissertation/main/main.tex`

### Directory Structure

```
analysis/dissertation/
├── main/                           # Main compilation directory
│   ├── main.tex                    # Master document (includes all chapters)
│   ├── main.pdf                    # Compiled full dissertation
│   ├── main.log                    # Compilation log (check for overfull boxes)
│   ├── main.bbl                    # Compiled bibliography
│   └── references.bib              # Bibliography database
├── chapter_01_introduction/        # Ch1: Introduction (8 pages, 0 figures)
├── chapter_02_literature_review/   # Ch2: Literature Review (38 pages, 3 figures)
├── chapter_03_data_foundation/     # Ch3: Data Foundation (42 pages, 3 figures, 17 tables)
├── chapter_04_baseline_modeling/   # Ch4: Baseline Models (43 pages, 11 figures, 25 tables)
├── chapter_05_advanced_bayesian/   # Ch5: Advanced Bayesian (63 pages, 3 figures, 22 tables)
├── chapter_06_rl_design/           # Ch6: RL Framework (31 pages, 4 figures, 3 tables)
├── chapter_07_uncertainty_risk_betting/  # Ch7: Uncertainty & Risk (26 pages, 3 figures, 7 tables)
├── chapter_08_simulation/          # Ch8: Simulation (11 pages, 5 figures, 4 tables)
├── chapter_09_results_discussion/  # Ch9: Results (163 pages, 14 figures, 57 tables) ← HEAVIEST
├── chapter_10_production_deployment/  # Ch10: Production (40 pages, 2 figures, 5 tables)
├── chapter_11_lessons_learned/     # Ch11: Lessons (12 pages, 0 figures, 3 tables)
├── chapter_12_conclusion/          # Ch12: Conclusion and Future Work (updated Feb 2026 with full-season results)
├── chapter_12_profit_optimization/ # Ch12b: Profit Optimization (included via \input in Ch12)
├── appendix_*/                     # Appendices A-R (9 figures, 136 tables)
├── figures/                        # Shared figures directory
│   ├── out/                        # Generated figures (PDF, PNG, TEX)
│   └── *.tex                       # TikZ source files
├── style/                          # LaTeX style files
│   ├── tikz_diagram_style.tex      # TikZ diagram standards
│   └── *.sty                       # Custom packages
└── references.bib                  # Backup bibliography
```

### Chapter Standalone Compilation

Each chapter has a `standalone.tex` for isolated compilation:

```bash
# Build single chapter (FAST - ~10s vs 2min for full)
cd analysis/dissertation/chapter_04_baseline_modeling/
latexmk -pdf standalone.tex

# Or use the alias
dchap build 4                       # Build chapter 4
dchap list                          # Show all chapters with build status
```

### Key Line Numbers for Common Edits

| Location | File | Line | Purpose |
|----------|------|------|---------|
| Title page | `main/main.tex` | ~50 | Title, author, date |
| Preamble colors | `main/main.tex` | ~80-100 | `\definecolor{accentBlue}` |
| headheight fix | `main/main.tex` | ~95 | `\setlength{\headheight}{27.12pt}` |
| Bibliography style | `main/main.tex` | ~555 | `\bibliographystyle{plainnat}` |
| Chapter includes | `main/main.tex` | ~450-550 | `\include{chapter_XX}` |
| TikZ standards | `style/tikz_diagram_style.tex` | all | Node sizes, colors |

### Figure Sources & Regeneration

| Figure Type | Source | Regenerate Command |
|-------------|--------|-------------------|
| R/ggplot2 plots | `R/dissertation_figures.R` | `Rscript R/dissertation_figures.R` |
| Python plots | `nfl_analytics/visualizations/*.py` | `python nfl_analytics/visualizations/create_*.py` |
| TikZ diagrams | `figures/*.tex` | Edit directly, auto-compiled |
| Performance tables | `figures/out/*.tex` | Generated by R/Python scripts |

### Debugging Overfull Boxes

```bash
# Check compilation log for issues
texlogsieve main.log --only-summary

# Filter by chapter
texlogsieve main.log | grep "chapter_03"

# Common fixes:
# 1. Long \texttt{} paths → use \path{} or \allowbreak
# 2. Wide tables → add \small, reduce \tabcolsep, or use landscape
# 3. TikZ overflow → reduce scale factor (0.88 → 0.82)
# 4. Bibliography URLs → shorten or add \sloppy

# Target: <25 overfull boxes (all <10pt acceptable)
```

### Visual Inspection Workflow

```bash
# Convert chapter pages to PNG for visual review
pdftoppm -png -r 150 -f 1 -l 10 chapter_XX.pdf /tmp/ch_XX_page

# Quick preview specific pages
pdftk main.pdf cat 50-60 output /tmp/section.pdf
evince /tmp/section.pdf

# Check text legibility (all text ≥8pt)
# Check table margins (no overflow)
# Check figure labels (no overlap)
```

---

## Code Style

**Python:** Black formatter, type hints, Google-style docstrings, 100 char max
**R:** tidyverse style, `<-` for assignment, explicit `library()` calls
**SQL:** UPPERCASE keywords, lowercase table/column names, indent subqueries

---

## Current Status (February 12, 2026)

**Completed:**
- **2025 FULL SEASON ANALYSIS COMPLETE (Feb 12, 2026):** All 285 games (272 REG + 13 playoff) evaluated
  - Spread MAE: 11.05 (vs Vegas 9.67) — model underperforms market by 1.38 points
  - ATS ROI: -3.2% on 208 qualified bets — below 52.4% breakeven (market efficiency confirmed)
  - Win Prob Accuracy: 56.5% — better than coin flip but not exploitable
  - Playoffs: 8.43 spread MAE (better than regular season), 70% ATS on 10 bets
  - Super Bowl LIX: SEA 29, NE 13 — model predicted NE +1.4 (17.4pt error)
  - Critical finding: Feature pipeline incompleteness (missing QBR, partial NextGen, no semantic data)
  - Dissertation updated: new postseason validation section, Ch 9 + Ch 12 updated (753 pages)
  - ADR-017 created: 2026 Season Improvement Roadmap (3 tiers)
  - Database: Created `ml.game_outcome_predictions` table, custom `AVG(boolean)` / `SUM(boolean)` aggregates
  - Code: Fixed DEFAULT_MODEL_DIR to v2, extended inference to Week 22 (playoffs)
- **Code Review Remediation (Feb 12, 2026):** 8-chunk systematic review of entire Python codebase
  - Fixed hardcoded credentials → env var fallbacks
  - Tightened exception handling across API/ETL/production
  - SQL identifier quoting, mutable default fixes
  - Eliminated shell=True, os.system/eval usage
  - Added MCMC convergence checks and TRUNCATE safeguards
- **Path Normalization (Feb 12, 2026):** Replaced /Users/dro → /home/dro across 300+ files (macOS → Linux migration)
- **Kalman/GP Features DISABLED (Dec 31):** ADR-015 documents decision to disable negative-signal features
  - Created `nfl_analytics/config/feature_weights.yaml` for feature group enable/disable via config
  - Kalman EPA (18 features): -10.2pp accuracy drop in A/B test
  - GP Injury (17 features): GP never trained, used fake estimated data
  - Code preserved with weight=0.0 for potential future iteration
- **Possession Value Schema Fix (Dec 31):** `fixed_drive` column doesn't exist in nflverse schema
  - Returns league averages instead of failing with SQL errors
  - TODO: Implement drive inference from play sequences if feature needed
- **Docker Networking Fix (Dec 31):** Fixed hardcoded localhost:5544 in feature extractors
  - SemanticRepository, KalmanEPAFilter now use `get_db_config()` with env vars
  - v2.1 training went from 22% → 71.6% non-zero features after fix
- **True SHAP Implementation (Dec 28, H2):** Replaced heuristic attribution with TreeExplainer
  - `semantic_attribution.py` now uses `shap.TreeExplainer` on production XGBoost models
  - Computes real Shapley values for 38 semantic features (`sem_*` prefix)
  - Graceful fallback to heuristic when SHAP unavailable
  - Enables accurate feature importance measurement for semantic pipeline
- **NER STATUS Extraction Fix (Dec 28, H4):** Fixed train-test mismatch in NER system prompt
  - Root cause: Training data had STATUS labels but inference prompt didn't list STATUS category
  - Added STATUS to system prompt with valid values: out, doubtful, questionable, probable
  - **Validation: 0%→100% STATUS extraction** (exceeds 70% target)
  - File: `nfl_analytics/semantic/models/inference.py:56-60`
- **Rest Differential & BTB Detection (Dec 28):** Fixed `rest_differential=0` placeholder in game_feature_extractor.py
  - 7 new features: rest_differential, home/away_days_rest, is_home/away_short_week, is_home/away_long_week
  - **Strong 2025 signal**: Home +3 days rest = 69.2% cover, Away +3 days = 44.4% (25 point swing!)
  - r=0.146 correlation with ATS outcome (meaningful for sports betting)
  - Total features now 348 (305 base + 37 synthetic + 6 rest/BTB)
- **S1.1: Kalman EPA Filter (Dec 28):** `nfl_analytics/features/kalman_epa_filter.py` - Team EPA regime detection
  - Local Level (Random Walk + Noise) state-space model
  - CUSUM on standardized innovations for regime change detection
  - Noise parameters: σ_v²=0.025 (observation), σ_w²=0.001 (process) calibrated for NFL EPA
  - Features: kalman_epa, kalman_epa_uncertainty, kalman_cusum_up/down, kalman_regime_change
- **S1.2: GP Injury Recovery (Dec 28):** `nfl_analytics/features/gp_injury_recovery.py` - Uncertainty-aware recovery curves
  - Kernel: Matern 5/2 + Constant + White Noise (Gemini consultation)
  - Logit transform for bounded [0, 100%] performance output
  - Hierarchical per injury category: muscle, structural, concussion, other
  - Features: gp_recovery_pct, gp_recovery_lower_95, gp_recovery_upper_95, injury_days_since_return
- **S2: WHY System Complete (Dec 28):** 3 new modules in `nfl_analytics/feedback/` for prediction explainability
  - `model_error_decomposition.py`: C_i = w_i × (Y - Ŷ_i) signed contribution per model
  - `temporal_causality.py`: Trace News→Sentiment→Feature→Prediction→Outcome chain
  - `feature_interactions.py`: SHAP interaction_values for semantic × base synergies
  - New tables: ml.model_error_decomposition, ml.causal_chains, ml.feature_interactions
  - Gemini MCP consultation for mathematical formulations
- 2025 Holdout Validation: 74.9% spread accuracy, 44.25 MAE props (Dec 9)
- Phase 2 Advanced Stats: 120 new features
- Dirichlet Optimization: Equal weights outperform v8.0 on holdout
- **MLX vs Ollama Benchmark (Dec 23):** MLX 12x faster (2.4s vs 28.6s median), 23x faster for sentiment
- **NER Model Evaluation (Dec 23):** 76% weighted accuracy, status extraction is weak point (40%)
- **Semantic A/B Test (Dec 23):** 195 games tested, INCONCLUSIVE (p=0.21), v2 did not outperform v1
- **XGBoost v4 Ridge Final Validation (Dec 23):** 59.14 MAE (47% WORSE than baseline), permanently DISABLED
- **Synthetic Features (Dec 9):** 37 new features across 3 modules
- **Props Stability Weighting (Dec 9):** SumerSports YoY correlation research integrated
- **Code Quality Remediation (Dec 13):** 6-phase systematic fix (90+ issues)
  - Phase 1: Foundation tests (85 unit tests added)
  - Phase 2: Strategy pattern for feature extractors (FeatureExtractorStrategy interface)
  - Phase 3: Unified asof features (consolidated 3 modules into `asof_features_unified.py`)
  - Phase 4: Connection pool verification (infrastructure already existed)
  - Phase 5: Critical bug fixes (temporal split, connection leak, uncertainty estimation)
  - Phase 6: Documentation (XGBoost hyperparams, team_mapping, CLAUDE.md)
- **Git Worktrees (Dec 20):** 4 parallel development worktrees (experiments, dissertation, hotfix, backtest)
- **Semantic Stack Phase 1 (Dec 21):** LLM-powered feature engineering
  - 26 semantic features integrated into GameFeatureExtractor (prefix: `sem_`)
  - Ollama (qwen3:8b) + MLX + Gemini/Claude fallback chain
  - News/social ingestion, NER/sentiment training data (5K + 2K samples)
  - ValueBetDetector with LLM-generated explanations
  - Fine-tuning utilities for Ollama Modelfile and MLX LoRA
- **Weekly Feedback Loop (Dec 22):** Complete bidirectional learning system
  - 10 modules across 3 phases (Phase 6: Feedback, Phase 7: Meta-Learning, Phase 8: A/B Testing)
  - 8 new database tables for tracking (ml.feedback_runs, ml.feature_performance, etc.)
  - Automatic feature selection: probation → active → disabled lifecycle
  - LLM prompt evolution based on prediction outcomes
  - McNemar test for A/B significance on 2025 holdout
  - Weekly orchestrator: `scripts/run_weekly_feedback.py`
- **Game Outcome Model v2 PROMOTED TO PRODUCTION (Dec 23):** Semantic-enhanced XGBoost models
  - 382 features (344 base + 38 semantic LLM-derived features)
  - Spread MAE: 10.45, Total MAE: 10.08, Win Prob AUC: 0.691
  - A/B test: +4.6% win accuracy improvement vs v1 (non-semantic baseline)
  - SHAP analysis: sem_away_media_buzz_volume in top 50 features
  - Production artifacts: `models/production/game_outcome_v2/`

**Key Findings (Dec 13-28):**
- Season-based temporal split prevents data leakage (train on seasons < max, validate on max)
- RMSE is correct uncertainty estimate (not MAE * 0.5)
- Context managers prevent connection leaks in high-load scenarios
- **Feature weighting has NO effect on XGBoost** - tree models are scale-invariant (Dec 14)
- **Feature pruning trade-off discovered** (Dec 14): More features = better spread MAE, fewer = better win AUC
- **Best XGBoost config**: reg_alpha=1.0, reg_lambda=2.0 (only 0.09% improvement over baseline)
- **qwen3:8b requires chat endpoint** (Dec 21): generate endpoint returns empty, use chat with 240s timeout
- **Bidirectional meta-learning** (Dec 22): Model performance informs semantic features, semantic patterns suggest new features
- **MLX is production-ready** (Dec 23): 12x faster than Ollama on Apple Silicon, use as primary with Ollama fallback
- **Semantic features need validation** (Dec 23): A/B test inconclusive (p=0.21), don't enable until statistical significance achieved
- **Target leakage inflates metrics by 14x** (Dec 23): XGBoost v4 went from 4.27 MAE (leaked) to 59.14 MAE (real)
- **Kalman filter noise ratio** (Dec 28): σ_w²/σ_v² = 4% optimal for NFL EPA (slow underlying, noisy observations)
- **GP injury recovery** (Dec 28): Matern 5/2 preferred for once-differentiable recovery curves, logit transform bounds output
- **Gemini MCP for research** (Dec 28): Effective for mathematical consultation (kernel design, state-space parameters)
- **Error decomposition formula** (Dec 28): C_i = w_i × (Y - Ŷ_i) enables per-model attribution
- **SHAP interactions** (Dec 28): O(M² × N × T) complexity - subsample to ~2000 for tractability

**In Progress (2026 Season Preparation — ADR-017):**
- Tier 1: Fix feature pipeline health checks (QBR, NextGen, semantic data staleness alerts)
- Tier 1: Register Game Outcome v2 in model registry + populate ml.model_metrics
- Tier 1: Unified weekly pipeline (`make weekly-pipeline SEASON=2026 WEEK=N`)
- Tier 1: Restore 4 missing production model files (state-space v2, GNN, temp scaler, v1 staging)
- Tier 2: Add playoff-specific features (game_type, seed_differential, bye_week_advantage)
- Tier 2: Retrain on 2022-2025 window (25% more training data)
- Tier 2: Boolean/integer schema cleanup (proper migration, remove custom aggregates)

**Strategic Decisions:**
- BNN research abandoned (Nov 9) - 84 hours, 6 approaches, only BNN v3.1 Rushing remains
- Dynamic meta-learning rejected (Nov 18) - static weights outperform, see ADR-009
- Dual-weight strategy adopted (Nov 27) - different domains need different weights
- ValidatorRegistry pattern adopted (Dec 13) - Strategy pattern for validation
- **Feature optimization tested, baseline retained (Dec 14)** - see ADR-011
- **Feedback loop with automatic feature selection (Dec 22)** - see ADR-012

**Recent Work:** [docs/milestones/RECENT_WORK_HISTORY.md](docs/milestones/RECENT_WORK_HISTORY.md)

---

## Documentation Index

| Category | Key Docs |
|----------|----------|
| Quick Start | README.md, SETUP.md, CONTRIBUTING.md |
| Database | [DATABASE_QUICK_REFERENCE](docs/database/DATABASE_QUICK_REFERENCE.md), [SQL_PATTERNS](docs/database/SQL_PATTERNS.md) |
| Architecture | [ARCHITECTURE_REFERENCE](docs/architecture/ARCHITECTURE_REFERENCE.md), [COMPONENT_DEEP_DIVES](docs/architecture/COMPONENT_DEEP_DIVES.md) |
| Milestones | [RECENT_WORK_HISTORY](docs/milestones/RECENT_WORK_HISTORY.md), [docs/milestones/](docs/milestones/) |
| Synthesis | [DIRICHLET_OPTIMIZATION](docs/synthesis/DIRICHLET_WEIGHT_OPTIMIZATION_SYNTHESIS_NOV27.md) |
| ADRs | [docs/adr/](docs/adr/) (009: Dynamic weights, 011: Feature optimization, 012: Feedback loop, 015: Kalman/GP disabled, 016: CQL target leakage, 017: 2026 season roadmap) |

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
7. **Season-based temporal splits** - percentage splits can leak data across train/val
8. **RMSE for uncertainty** - MAE * arbitrary factor has no statistical basis
9. **Context managers for DB** - always use `with conn:` to prevent resource leaks
10. **Strategy pattern for extensibility** - ValidatorRegistry enables plugin validators
11. **Feature weighting doesn't help tree models** - XGBoost splits on values, not magnitudes (scale-invariant)
12. **Pruning creates trade-offs** - fewer features improves classification but hurts regression accuracy
13. **Test hypotheses empirically** - Bayesian/Dirichlet weighting sounded promising but had zero effect
14. **True holdout is essential** - 2022-2024 validation showed 18% improvement; 2025 holdout showed 47% degradation
15. **A/B test before enabling features** - Semantic features underperformed baseline despite theoretical appeal
16. **Local inference beats cloud on Apple Silicon** - MLX 12x faster than Ollama, 40% less power
17. **More features can mean worse models** - 423 features with Kalman/GP underperformed 382 feature v2 baseline
18. **Config-based feature gating** - YAML config allows disabling features without code changes (easy iteration)
19. **Check schema before writing SQL** - `fixed_drive` column didn't exist, causing silent failures
20. **Fake data is worse than no data** - GP injury recovery using estimated days provided negative signal

### Full 2025 Season Lessons (Feb 2026)

21. **Feature pipeline health > model sophistication** - Missing QBR/NextGen/semantic data compressed predictions to [-2.6, +8.7] vs actual [-35, +25]. A single missing data source degrades ALL 382 features.
22. **Full season reveals truth** - Week 16 showed 10.45 spread MAE; full season showed 11.05 (5.7% worse). Always evaluate on complete holdout.
23. **Playoffs behave differently** - 8.43 playoff MAE vs 11.14 regular season. Late-season team quality differentials are more informative. Dedicated playoff features merit investigation.
24. **Custom PostgreSQL aggregates are fragile** - `AVG(boolean)` doesn't exist natively; creating custom aggregates works but adds schema complexity across machines. Prefer explicit `::int` casts.
25. **Model registry hygiene matters** - Production v2 not registered in ml.model_registry caused deployment confusion and audit gaps (ADR-017 Tier 1.2).
26. **Market efficiency is real** - -3.2% ATS ROI on 208 bets confirms semi-strong EMH for NFL spreads with public data. The 4.5% vig is insurmountable.

### LaTeX/Dissertation Lessons (Jan 2026)

21. **texlogsieve for debugging** - Parse main.log to find exact line numbers of overfull boxes, much faster than visual scanning
22. **Two references.bib files** - `main/references.bib` is the active one, `analysis/dissertation/references.bib` is backup; edit the correct one
23. **BibTeX regeneration requires cleanup** - If `.bbl` doesn't update after `.bib` changes, delete `.bbl` and run `bibtex main` manually
24. **TikZ scale for overflow** - Reduce `scale=0.88` to `scale=0.82` to fix 10-15pt overfull boxes in diagrams
25. **`\allowbreak` for long texttt** - Insert `\allowbreak` in long file paths: `\texttt{off\_epa\_\allowbreak opponent\_adjusted}`
26. **`\sloppy` for bibliography** - Wrap bibliography in `\begingroup\sloppy\emergencystretch=3em ... \endgroup` to fix URL overflows
27. **Standalone chapter PDFs are faster** - Each chapter's `standalone.tex` compiles in ~10s vs 2min for full dissertation
28. **pdftoppm for visual inspection** - Convert pages to PNG at 150 DPI: `pdftoppm -png -r 150 -f 1 -l 10 chapter.pdf /tmp/ch_page`
29. **<25 overfull boxes acceptable** - Issues <10pt are typically invisible to readers; focus on >15pt issues first
30. **fancyhdr headheight warning** - Fix globally with `\setlength{\headheight}{27.12pt}` in preamble (affects 100+ pages)

---

**Version:** 7.0 | **Maintainer:** Claude Code | **Repository:** github.com/raold/nfl-analytics

**Project Rules:**
- Never take shortcuts, never use synthetic data
- Survey, read, audit, synthesize before writing code
- When doing git commit NEVER add co-authorship
