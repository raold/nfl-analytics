# Architecture Reference

**Last Updated:** November 8, 2025
**Version:** 4.4
**Purpose:** Complete directory structure reference for NFL Analytics project

---

## Complete Directory Tree

```
nfl-analytics/
├── py/                                   # Python modules
│   ├── causal/                           # Causal inference framework (8 modules)
│   │   ├── panel_constructor.py          # Build longitudinal datasets
│   │   ├── treatment_definitions.py      # Define shock events
│   │   ├── confounder_identification.py  # Identify confounding
│   │   ├── synthetic_control.py          # Counterfactual estimation
│   │   ├── diff_in_diff.py               # DiD estimators
│   │   ├── structural_causal_models.py   # Causal DAGs
│   │   ├── model_integration.py          # Integrate with BNN
│   │   └── validation.py                 # Validation/backtesting
│   ├── compute/                          # Distributed compute system
│   │   ├── statistics/                   # Statistical testing
│   │   ├── sync/                         # Google Drive sync
│   │   ├── hardware/                     # M4 vs 4090 routing
│   │   ├── task_queue.py                 # Task management (WAL mode)
│   │   ├── adaptive_scheduler.py         # Multi-armed bandit
│   │   └── compute_worker.py             # Worker processes
│   ├── features/                         # Feature engineering
│   │   ├── asof_features.py              # Leakage-safe as-of features
│   │   ├── asof_features_enhanced.py     # 157-feature enhanced set
│   │   ├── bayesian_player_features.py   # Extract from posteriors
│   │   ├── defensive_situational_strength.py        # Phase 2.4
│   │   ├── defensive_game_script_indicators.py      # Phase 2.5
│   │   ├── vegas_line_bias_features.py              # Phase 3
│   │   ├── environmental_contextual_features.py     # Phase 3
│   │   └── feature_interactions.py                  # Phase 3.5
│   ├── models/                           # ML models
│   │   ├── bayesian_neural_network.py    # BNN with PyMC
│   │   ├── train_bnn_passing.py          # Passing yards BNN
│   │   ├── train_bnn_rushing.py          # Rushing yards BNN
│   │   ├── bnn_multioutput_pytorch.py    # Multi-output BNN (v3.1)
│   │   ├── bnn_calibrated_v2.py          # Calibrated BNN (T=0.75)
│   │   ├── xgboost_gpu_v3.py             # GPU-accelerated XGBoost
│   │   └── props_predictor.py            # Unified prediction interface
│   ├── ensemble/                         # Ensemble methods
│   │   ├── enhanced_ensemble_v3.py       # 4-way ensemble
│   │   └── stacked_meta_learner.py       # Meta-learning
│   ├── optimization/                     # Risk/optimization
│   │   └── portfolio_optimizer.py        # Correlation-adjusted Kelly
│   ├── rl/                               # Reinforcement learning
│   │   ├── dataset.py                    # RL logged dataset
│   │   ├── cql_agent.py                  # Conservative Q-Learning
│   │   └── ope_gate.py                   # Off-policy evaluation
│   ├── production/                       # Production services
│   │   └── cql_bet_sizer.py              # CQL bet sizing (4 risk gates)
│   ├── backtest/                         # Backtesting
│   │   ├── baseline_glm.py               # GLM baseline
│   │   ├── harness.py                    # Systematic backtesting
│   │   ├── bayesian_props_backtest.py    # Bayesian validation
│   │   └── comprehensive_safe_backtest.py # Leakage-free backtest
│   └── validation/                       # Data quality
│       └── data_quality_checks.py        # Validation framework
│
├── R/                                    # R statistical models
│   ├── bayesian_player_props.R           # v1.0 baseline hierarchical
│   ├── bayesian_receiving_with_qb_chemistry.R  # QB-WR dyadic effects
│   ├── train_and_save_passing_model.R    # v2.5 fixed pipeline
│   ├── advanced_priors_elicitation.R     # v3.0 empirical Bayes
│   ├── bayesian_team_ratings_brms.R      # Team hierarchical model
│   ├── extract_bayesian_ratings.R        # Database export utility
│   └── state_space/                      # State-space models
│       ├── train_simplified_random_walk.R  # v2 canonical training
│       ├── training_window_ablation.R      # Ablation study framework
│       └── validate_2025_holdout.R         # 2025 validation
│
├── data/                                 # Data ingestion scripts
│   ├── ingest_schedules.R                # Schedule ingestion
│   ├── ingest_pbp.R                      # Play-by-play ingestion
│   └── ingest_odds_history.py            # Historical odds (API)
│
├── db/                                   # SQL schema and migrations
│   ├── 001_init.sql                      # Core schema
│   ├── 002_timescale.sql                 # Timescale setup
│   ├── 023_player_hierarchy_schema.sql   # Player hierarchy
│   ├── 025_model_registry.sql            # ML model tracking
│   ├── 026_canonical_training_window.sql # Canonical window registry
│   └── migrations/                       # Schema migrations
│
├── analysis/                             # Analysis outputs
│   ├── dissertation/                     # Dissertation (LaTeX)
│   │   ├── main/                         # Main dissertation
│   │   │   ├── main.tex                  # 324-page dissertation
│   │   │   └── main.pdf                  # Compiled PDF
│   │   ├── chapter_1_introduction/       # Chapter 1
│   │   ├── chapter_2_literature_review/  # Chapter 2
│   │   ├── chapter_3_data_foundation/    # Chapter 3
│   │   ├── chapter_4_baseline_modeling/  # Chapter 4
│   │   ├── chapter_5_reinforcement_learning/ # Chapter 5
│   │   ├── chapter_6_advanced_bayesian/  # Chapter 6
│   │   ├── chapter_7_causal_inference/   # Chapter 7
│   │   ├── chapter_8_results_discussion/ # Chapter 8
│   │   ├── chapter_9_production_deployment/ # Chapter 9
│   │   ├── chapter_10_conclusion/        # Chapter 10
│   │   ├── chapter_11_offline_rl_bet_sizing/ # Chapter 11
│   │   ├── chapter_12_profit_optimization/   # Chapter 12
│   │   ├── appendix/                     # Consolidated appendices
│   │   ├── figures/                      # Figure generation
│   │   │   ├── out/                      # Generated LaTeX tables/figures
│   │   │   └── R/                        # R figure scripts
│   │   ├── style/                        # LaTeX styles
│   │   │   ├── tikz_diagram_style.tex    # Standardized TikZ style
│   │   │   └── dissertation_preamble.tex # LaTeX preamble
│   │   └── docs/                         # Dissertation documentation
│   ├── features/                         # Feature engineering outputs
│   ├── results/                          # Experiment results
│   ├── calibration/                      # Calibration studies
│   └── reports/                          # Analysis reports
│
├── models/                               # Trained models
│   ├── bayesian/                         # Bayesian models
│   │   ├── simplified_random_walk_v2.rds      # 14MB - v2 canonical (APPROVED)
│   │   ├── passing_yards_hierarchical_v1.rds  # 22MB - v1.0 baseline
│   │   ├── passing_informative_priors_v1.rds  # 5.2MB - v3.0
│   │   ├── receiving_qb_chemistry_v1.rds      # 210MB - v3.0 QB-WR
│   │   ├── bnn_passing_v1.pkl                 # 80MB - v3.0 BNN
│   │   ├── bnn_multioutput_v1_final.pt        # 59KB - v3.1 multi-output
│   │   └── player_skill_trajectories_v1.csv   # State-space output
│   ├── cql/                              # CQL models
│   │   ├── best_model.pth                # 207KB - trained CQL
│   │   └── cql_training_log.json         # Training metrics
│   └── xgboost/                          # XGBoost models
│       └── v2_1.pkl                      # GPU-trained XGBoost
│
├── tests/                                # Test suite
│   ├── unit/                             # Unit tests
│   │   └── test_*.py                     # Unit test modules
│   ├── integration/                      # Integration tests
│   │   ├── test_temporal_leakage.py      # Leakage detection (10 tests)
│   │   └── test_*.py                     # Other integration tests
│   ├── e2e/                              # End-to-end tests
│   └── README.md                         # Testing documentation
│
├── scripts/                              # Automation scripts
│   ├── dev/                              # Development scripts
│   │   └── init_dev.sh                   # Initialize dev environment
│   ├── setup_testing.sh                  # Setup testing environment
│   ├── gpu_training_orchestrator.py      # GPU training coordinator
│   └── train_all_phases_scaled.py        # Scaled training pipeline
│
├── containers/                           # Docker files (Phase 1+)
│   ├── Dockerfile.api                    # FastAPI prediction service
│   ├── Dockerfile.streamlit              # Streamlit dashboard
│   ├── Dockerfile.worker                 # Celery workers (Phase 2)
│   ├── Dockerfile.jupyter                # Jupyter notebooks (Phase 2)
│   └── Dockerfile.gpu-trainer            # GPU training (Phase 3)
│
├── docs/                                 # Project documentation
│   ├── CAUSAL_INFERENCE_FRAMEWORK.md     # Causal inference guide
│   ├── ADVANCED_BAYESIAN_V3.md           # v3.0 technical docs
│   ├── BAYESIAN_ENHANCEMENTS_v2.md       # v2.5 enhancements
│   ├── STATE_SPACE_FINAL_DECISION.md     # State-space GO/NO-GO
│   ├── TRAINING_WINDOW_ABLATION_RESULTS.md # Ablation study (500+ lines)
│   ├── BNN_CALIBRATION_COMPLETE.md       # BNN calibration validation
│   ├── AUTOMATED_LEAKAGE_TESTS_COMPLETE.md # Leakage test suite
│   ├── DATABASE_SCHEMA_AUDIT.md          # Comprehensive schema audit
│   ├── ARCHITECTURE_REFERENCE.md         # This file
│   ├── database/                         # Database docs
│   │   ├── schema.md                     # Schema documentation
│   │   ├── erd.md                        # Entity-relationship diagram
│   │   └── DATABASE_QUICK_REFERENCE.md   # Quick lookup guide
│   ├── guides/                           # User guides
│   │   └── COMMON_TASKS_REFERENCE.md     # Task cookbook
│   ├── milestones/                       # Milestone summaries
│   │   ├── CQL_COMPLETE_SUMMARY.md       # CQL training complete
│   │   ├── BAYESIAN_PROPS_COMPLETE.md    # Bayesian methodology
│   │   └── RECENT_WORK_HISTORY.md        # Historical work log
│   └── COMPONENT_DEEP_DIVES.md           # Component usage details
│
├── .github/workflows/                    # CI/CD workflows
│   ├── test.yml                          # Test suite workflow
│   ├── pre-commit.yml                    # Code quality checks
│   └── data-quality.yml                  # Nightly data validation
│
├── dashboard/                            # Monitoring dashboard
│   ├── Dockerfile.streamlit              # Streamlit container
│   └── streamlit_app.py                  # Dashboard app
│
├── mcp_server/                           # Model Context Protocol server
│   ├── Dockerfile                        # MCP container
│   └── server.py                         # FastMCP server
│
├── prometheus/                           # Prometheus config
│   └── prometheus.yml                    # Scrape configs
│
├── grafana/                              # Grafana dashboards
│   └── provisioning/                     # Dashboard provisioning
│
├── pgdata/                               # PostgreSQL data volume
├── run_compute.py                        # Compute system entry point
├── docker-compose.yml                    # Docker services (Phase 1)
├── docker-compose.dev.yml                # Dev services (Phase 2)
├── docker-compose.gpu.yml                # GPU services (Phase 3)
├── requirements.txt                      # Python dependencies
├── renv.lock                             # R package versions
├── pytest.ini                            # Test configuration
├── .pre-commit-config.yaml               # Pre-commit hooks
├── .env                                  # Environment variables (DO NOT COMMIT)
├── .env.example                          # Environment template
├── README.md                             # Project overview
├── CLAUDE.md                             # AI assistant context (refactored)
├── PROJECT_STATUS.md                     # Current project status
├── DOCUMENTATION_INDEX.md                # Complete documentation catalog
└── PHASE_PLAN.md                         # 24-week implementation plan
```

---

## Key Directory Purposes

### `/py` - Python ML/Data Pipeline
- **causal/**: Causal inference framework (DiD, synthetic control, SCM)
- **compute/**: Distributed compute coordination (Mac M4 ↔ Windows RTX 4090)
- **features/**: Feature engineering (leakage-safe, temporal, contextual)
- **models/**: ML models (BNN, XGBoost, ensemble)
- **rl/**: Reinforcement learning (CQL bet sizing)
- **production/**: Production services (API, bet sizer)
- **backtest/**: Systematic backtesting framework
- **validation/**: Data quality checks

### `/R` - Statistical Models
- Bayesian hierarchical models (brms, Stan)
- State-space models (random walk, AR(1))
- QB-WR chemistry (dyadic random effects)
- Empirical Bayes priors
- Model training scripts

### `/db` - Database Schema
- SQL migrations (001-026)
- Schema definitions
- Model registry
- Canonical training window tracking

### `/analysis` - Research Outputs
- **dissertation/**: 324-page LaTeX dissertation
- **features/**: Feature engineering outputs
- **results/**: Experiment results (ablation, backtest, calibration)
- **reports/**: Analysis reports

### `/models` - Trained Artifacts
- **bayesian/**: R/PyMC models (rds, pkl, pt files)
- **cql/**: Reinforcement learning models
- **xgboost/**: GPU-trained XGBoost models

### `/tests` - Test Suite
- **unit/**: Fast, isolated tests (no DB)
- **integration/**: DB-dependent tests (Docker)
- **e2e/**: End-to-end workflow tests

### `/scripts` - Automation
- Development initialization
- GPU training orchestration
- Testing setup
- Deployment automation

### `/containers` - Docker Infrastructure
- API service (FastAPI)
- Dashboard (Streamlit)
- Workers (Celery)
- Jupyter/RStudio (dev tools)
- GPU training (CUDA/MPS)

### `/docs` - Documentation
- Technical guides (causal, Bayesian, state-space)
- Database reference
- Milestone summaries
- Component deep dives
- Common tasks cookbook

---

## Maintenance Notes

### Regenerating Directory Tree

Use the following command to regenerate the directory tree:

```bash
# From project root
tree -L 3 --dirsfirst -I 'pgdata|__pycache__|*.pyc|.git|node_modules|renv' > tree_output.txt

# Or for full tree:
tree -L 4 --dirsfirst -I 'pgdata|__pycache__|*.pyc|.git|node_modules|renv|*.log|*.aux|*.bbl'
```

### Key Exclusions
- `pgdata/`: PostgreSQL data (large, gitignored)
- `__pycache__/`: Python bytecode
- `.git/`: Git internals
- `node_modules/`: Node dependencies
- `renv/`: R package cache
- Log files, LaTeX aux files

---

## Cross-References

- **Database Schema**: See [docs/database/DATABASE_QUICK_REFERENCE.md](database/DATABASE_QUICK_REFERENCE.md)
- **Component Details**: See [docs/COMPONENT_DEEP_DIVES.md](COMPONENT_DEEP_DIVES.md)
- **Complete Documentation**: See [DOCUMENTATION_INDEX.md](../DOCUMENTATION_INDEX.md)
- **Recent Work**: See [docs/milestones/RECENT_WORK_HISTORY.md](milestones/RECENT_WORK_HISTORY.md)

---

**Last Updated:** November 8, 2025
**Maintainer:** Claude Code (Sonnet 4.5)
**Purpose:** Complete directory reference for AI assistants and developers
