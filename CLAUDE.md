# CLAUDE.md - NFL Analytics Project Documentation

**Last Updated:** December 24, 2025 | **Version:** 6.6 | **Status:** Production Ready (Grade A+)

## Quick Status

| Component | State | Key Metric |
|-----------|-------|------------|
| Ensemble | v6.2 + equal-weight props | 54.5% spread (games), 44.25 MAE (props) |
| Features | 382 total (Phase 2 + Synthetic + Semantic) | Base 305 + 37 synthetic + 38 semantic (v2) |
| Semantic Stack | Complete (Dec 21) | Ollama/MLX inference, 26 LLM features |
| **Game Outcome Model** | **v2 PRODUCTION (Dec 23)** | **382 features, +4.6% win accuracy, 10.45 spread MAE** |
| **Feedback Loop** | **NEW (Dec 22)** | **Weekly learning system, 10 modules** |
| Database | 64 tables, 1.25M plays | PostgreSQL + Redis |
| Dissertation | 693 pages | Compiles cleanly |
| Infrastructure | 8/9 Docker services | ~105ms inference |

**Critical Files:** `py/ensemble/enhanced_ensemble_v6_production.py`, `py/ensemble/strategy/base.py`, `py/features/game_feature_extractor.py`, `py/feedback/`

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
- `analysis/dissertation/` - 693-page LaTeX dissertation
- `models/` - Trained artifacts (~60 files)
- `docs/` - 400+ documentation files

**Features (368 total):**

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
| **Semantic (LLM)** | 26 | `py/semantic/features/semantic_features.py` (NEW) |
| Props Stability | N/A | `py/ensemble/props_stability_weights.py` |

**Entry Points:**
- `GameFeatureExtractor` - Base 305 + 26 semantic features (unified API, <500ms)
- `SyntheticInteractionEngine` - QB-OL synergy, defensive balance, explosive mismatch
- `SyntheticMomentumEngine` - Form intensity, pressure consistency, RZ efficiency
- `ResearchDrivenEngine` - nfelo EPA weighting (1.6:1.0), coaching rationality
- `SemanticFeatureExtractor` - LLM-derived sentiment, injury severity, media signals (NEW)

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

## Semantic Stack (LLM Integration)

**Status:** Phase 1 Complete (Dec 21, 2025)
**Goal:** Add AI/LLM capabilities to push win rate past 52.4% breakeven toward 56-57%

### Architecture

```
py/semantic/
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

### Semantic Features (26 total, prefix: `sem_`)

**Per-Team (11 each):**
| Feature | Type | Description |
|---------|------|-------------|
| `sentiment_24h` | float | News/social sentiment (-1 to 1) |
| `sentiment_7d_trend` | float | Week-over-week sentiment change |
| `injury_severity_llm` | int | LLM-assessed injury impact (1-5) |
| `key_player_risk` | float | Star player injury probability |
| `media_buzz_volume` | float | Media attention (0-1 normalized) |
| `coaching_change` | bool | Recent coaching staff changes |
| `roster_churn` | float | Player turnover indicator |
| `public_overreaction` | float | Market sentiment extremity |
| `contrarian_signal` | float | Inverse public sentiment |

**Matchup (4):**
| Feature | Description |
|---------|-------------|
| `sentiment_differential` | Home - away sentiment gap |
| `injury_advantage` | Relative injury situation |
| `media_attention_ratio` | Coverage imbalance |
| `combined_semantic_edge` | Composite semantic signal |

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
from py.features.game_feature_extractor import GameFeatureExtractor
extractor = GameFeatureExtractor()
features = extractor.extract_game_features(game_id)
# Access: features['sem_home_sentiment_24h'], features['sem_injury_advantage'], etc.

# Value bet detection with LLM explanations
from py.production.value_bet_detector import ValueBetDetector
detector = ValueBetDetector(min_edge=0.03, use_llm_explanations=True)
bets = detector.detect_value_bets(predictions)
```

---

## Feedback Loop System (Weekly Learning)

**Status:** Complete (Dec 22, 2025)
**Goal:** Close the loop between predictions and outcomes, enabling continuous learning and LLM fine-tuning

### Architecture

```
py/feedback/
├── outcome_updater.py        # Phase 6.1: Update actuals after games
├── semantic_attribution.py   # Phase 6.2: Track feature contribution
├── ensemble_weight_adjuster.py # Phase 6.3: Adjust weights on performance
├── retrain_trigger.py        # Phase 6.4: Detect retraining needs
├── annotation_generator.py   # Phase 6.5: Generate LLM training data
├── meta_learner.py           # Phase 7.1: Bidirectional Model↔Semantic learning
├── feature_memory.py         # Phase 7.2: Rolling feature performance tracking
├── auto_feature_selector.py  # Phase 7.3: Automatic feature enable/disable
├── prompt_evolver.py         # Phase 7.4: Evolve LLM prompts from outcomes
├── semantic_ab_test.py       # Phase 8.1: A/B test semantic vs non-semantic
└── __init__.py               # Module exports (10 components)
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

### Database Tables (8 new)

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
from py.feedback import WeeklyFeedbackOrchestrator
orchestrator = WeeklyFeedbackOrchestrator()
result = orchestrator.run(season=2025, week=16)
if result.triggers_fired:
    print(f"Retraining recommended: {result.triggers_fired}")

# Meta-learning insights
from py.feedback import SemanticMetaLearner
learner = SemanticMetaLearner()
result = learner.update_from_outcomes(season=2025, week=16)
for insight in result.feature_insights:
    print(f"{insight.feature_name}: {insight.recommendation}")

# Automatic feature selection
from py.feedback import AutoFeatureSelector
selector = AutoFeatureSelector()
selection = selector.evaluate_features(season=2025, week=16)
selector.apply_selection(selection)  # Updates ml.feature_selection_state

# A/B testing
from py.feedback import SemanticABTest
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

## Current Status (Dec 28, 2025)

**Completed:**
- **True SHAP Implementation (Dec 28, H2):** Replaced heuristic attribution with TreeExplainer
  - `semantic_attribution.py` now uses `shap.TreeExplainer` on production XGBoost models
  - Computes real Shapley values for 38 semantic features (`sem_*` prefix)
  - Graceful fallback to heuristic when SHAP unavailable
  - Enables accurate feature importance measurement for semantic pipeline
- **NER STATUS Extraction Fix (Dec 28, H4):** Fixed train-test mismatch in NER system prompt
  - Root cause: Training data had STATUS labels but inference prompt didn't list STATUS category
  - Added STATUS to system prompt with valid values: out, doubtful, questionable, probable
  - **Validation: 0%→100% STATUS extraction** (exceeds 70% target)
  - File: `py/semantic/models/inference.py:56-60`
- **Rest Differential & BTB Detection (Dec 28):** Fixed `rest_differential=0` placeholder in game_feature_extractor.py
  - 7 new features: rest_differential, home/away_days_rest, is_home/away_short_week, is_home/away_long_week
  - **Strong 2025 signal**: Home +3 days rest = 69.2% cover, Away +3 days = 44.4% (25 point swing!)
  - r=0.146 correlation with ATS outcome (meaningful for sports betting)
  - Total features now 348 (305 base + 37 synthetic + 6 rest/BTB)
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

**Key Findings (Dec 13-22):**
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

**In Progress:**
- Extended semantic A/B testing through Week 18 for larger sample size
- Feature ablation to identify which semantic features provide positive signal
- Spoke 1 Research: Kalman Filter on EPA trends, GP injury recovery curves (Week 2-3)

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
| Architecture | [ARCHITECTURE_REFERENCE](docs/ARCHITECTURE_REFERENCE.md), [COMPONENT_DEEP_DIVES](docs/COMPONENT_DEEP_DIVES.md) |
| Milestones | [RECENT_WORK_HISTORY](docs/milestones/RECENT_WORK_HISTORY.md), [docs/milestones/](docs/milestones/) |
| Synthesis | [DIRICHLET_OPTIMIZATION](docs/synthesis/DIRICHLET_WEIGHT_OPTIMIZATION_SYNTHESIS_NOV27.md) |
| ADRs | [docs/adr/](docs/adr/) (009: Dynamic weights, 011: Feature optimization, 012: Feedback loop) |

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

---

**Version:** 6.6 | **Maintainer:** Claude Code | **Repository:** github.com/raold/nfl-analytics

**Project Rules:**
- Never take shortcuts, never use synthetic data
- Survey, read, audit, synthesize before writing code
- When doing git commit NEVER add co-authorship
