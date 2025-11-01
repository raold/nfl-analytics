# NFL Analytics Codebase Audit – January 2025

**Status**: Comprehensive review completed  
**Auditor**: GitHub Copilot (Claude-based agent)  
**Date**: January 2025  
**Scope**: Full repository scan to assess implementation status vs TODO.tex

---

## Executive Summary

The codebase is **significantly more complete** than TODO.tex indicated. Most P0 baseline modeling, risk management, and simulation infrastructure items are **production-ready**. The primary gaps are:

1. **State-space ratings models** (Glickman-Stern) – P1 priority, not yet implemented
2. **RL agent training** (DQN/PPO) – P0 priority, dataset/OPE done but no training loop
3. **Weather & injury data integration** – P1 priority, schemas exist but tables empty
4. **Test coverage expansion** – P1 priority, currently 6% (17 tests for odds parsing only)

**Recommendation**: Shift focus to DQN/PPO implementation (highest ROI for dissertation novelty) and state-space models (fills classical benchmark gap). Weather/injury integrations can be P2 if marginal lift is modest.

---

## Implementation Status by Milestone

### ✅ **Data Foundations** (80% complete)

| Component | Status | Implementation | Lines | Notes |
|-----------|--------|----------------|-------|-------|
| nflverse ingestion | ✅ **DONE** | `data/ingest_pbp.R` | 39 | 1.23M plays, 1999-2024 |
| Schedule/odds ingestion | ✅ **DONE** | `data/ingest_schedules.R` | - | 6,991 games with closing lines |
| Odds history API | ✅ **DONE** | `py/ingest_odds_history.py` | - | 820K rows via TheOddsAPI; nightly GH Actions |
| Database schema | ✅ **DONE** | `db/001_init.sql`, `db/002_timescale.sql` | - | TimescaleDB hypertables, indexes, marts |
| Weather integration | ⚠️ **PARTIAL** | `py/weather_meteostat.py` | - | Code exists, table empty, needs data pull |
| Injury integration | ⚠️ **PARTIAL** | Schema exists | - | Table created but empty, needs data source |
| EPA features | ✅ **DONE** | `data/features_epa.R` | 49 | Writes to `mart.team_epa` |
| As-of features (leakage-safe) | ✅ **DONE** | `py/features/asof_features.py` | 412 | SQL-based rolling aggregation |

**Key Wins**:
- Full nflverse integration with 25+ years of play-by-play
- Real-time odds monitoring operational
- Analytic marts auto-created (`mart.team_epa`, `mart.game_summary`)

**Gaps**:
- Weather table empty (have code, need to run ingestion)
- Injury data source not identified yet
- Advanced trench/role stability features (P1) not implemented

---

### ✅ **Baseline Models** (70% complete)

| Model | Status | Implementation | Lines | Notes |
|-------|--------|----------------|-------|-------|
| GLM (Logistic Regression) | ✅ **DONE** | `py/backtest/baseline_glm.py` | 410 | Walk-forward validation, Platt/isotonic calibration, TeX output |
| XGBoost | ✅ **DONE** | `notebooks/10_model_spread_xgb.qmd` | 147 | Grid search, EPA features, configurable hyperparams |
| Skellam/Poisson scores | ✅ **DONE** | `py/models/score_distributions.py` | 253 | Key-number reweighting (3,6,7,10), moment-preserving IPF |
| Gaussian copula | ✅ **DONE** | `py/models/copulas.py` | 111 | Spread/total dependence, Acklam inverse-CDF |
| Monte Carlo engine | ✅ **DONE** | `py/monte_carlo.py` | - | Basic GradientBoosting; could swap Skellam in |
| State-space ratings | ❌ **TODO** | - | - | Glickman-Stern Kalman/Stan; **next P1 priority** |
| In-play RF (Lock-Nettleton) | ❌ **TODO** | - | - | Optional, P1 |

**Key Wins**:
- GLM is **production-ready** with calibration and dissertation TeX output
- Score distributions handle key-number mass properly (critical for NFL)
- Multiple notebooks validate models (04_score_validation, 05_copula_gof)

**Gaps**:
- State-space models are the biggest classical modeling gap
- Stern (1991) normal mapping noted but not implemented as standalone

**Outputs**:
- 14 CSV files in `analysis/results/` (GLM metrics, predictions, harness outputs)
- 54+ TeX tables in `analysis/dissertation/figures/out/` (GLM baseline, harness comparisons, calibration)

---

### ✅ **Risk & Sizing Infrastructure** (95% complete)

| Component | Status | Implementation | Lines | Notes |
|-----------|--------|----------------|-------|-------|
| CVaR LP solver | ✅ **DONE** | `py/risk/cvar_lp.py` | - | Linear programming for α=0.90, 0.95 |
| Scenario generator | ✅ **DONE** | `py/risk/generate_scenarios.py` | - | Monte Carlo with seeded RNG |
| CVaR TeX reports | ✅ **DONE** | `py/risk/cvar_report.py` | - | Multi-alpha comparison tables |
| Risk sizing notebook | ✅ **DONE** | `notebooks/12_risk_sizing.qmd` | - | End-to-end CVaR workflow |

**Key Wins**:
- Complete fractional Kelly/CVaR stack ready for dissertation
- TeX output auto-generated for dissertation figures
- Notebook 12 provides reproducible workflow

**Gaps**:
- Uncertainty-aware policy (downweight under wide posteriors) is P1, not implemented

---

### ✅ **Simulation & Pricing** (90% complete)

| Component | Status | Implementation | Lines | Notes |
|-----------|--------|----------------|-------|-------|
| Teaser pricing | ✅ **DONE** | `py/pricing/teaser.py` | - | EV with correlation, middle breakeven |
| Acceptance testing | ✅ **DONE** | `py/sim/acceptance.py` | - | EMD, key-mass deltas, frictions; JSON+TeX |
| Execution engine | ✅ **DONE** | `py/sim/execution.py` | - | Bucket params, simulate_order |
| Simulator acceptance notebook | ✅ **DONE** | `notebooks/90_simulator_acceptance.qmd` | - | Outcomes validation |

**Key Wins**:
- Teaser/alt-line pricing operational (dissertation differentiator)
- Acceptance tests validate simulation realism (EMD < 0.03 target, key-mass chi-squared)

**Gaps**:
- Multi-book arbitrage scan is basic (P1 enhancement)

---

### ⚠️ **RL Capstone** (50% complete)

| Component | Status | Implementation | Lines | Notes |
|-----------|--------|----------------|-------|-------|
| Offline dataset builder | ✅ **DONE** | `py/rl/dataset.py` | - | `fetch_games()`, `build_logged_dataset()` |
| OPE estimators (SNIS/DR) | ✅ **DONE** | `py/rl/ope.py` | - | Clipping, shrinkage, outcome model |
| OPE gate | ✅ **DONE** | `py/rl/ope_gate.py` | - | Grid eval, JSON+TeX output |
| RL ablation notebook | ✅ **DONE** | `notebooks/80_rl_ablation.qmd` | - | RL vs stateless baseline |
| DQN agent | ❌ **TODO** | - | - | **Critical P0 gap for dissertation** |
| PPO agent | ❌ **TODO** | - | - | P1 stretch goal |
| MPS/CUDA training config | ❌ **TODO** | - | - | Blocked on DQN/PPO implementation |

**Key Wins**:
- OPE infrastructure is **production-grade** (grid search, ESS checks, TeX output)
- Dataset builder enables offline RL experiments
- Notebook 80 sets up ablation framework

**Critical Gap**:
- **No DQN/PPO training loop implemented yet**. This is the highest-priority missing piece for dissertation novelty. OPE validates that offline RL is feasible, but need actual agent training to demonstrate lift over baselines.

---

### ✅ **Testing & CI/CD** (NEW – 60% complete)

| Component | Status | Implementation | Lines | Notes |
|-----------|--------|----------------|-------|-------|
| Unit test infrastructure | ✅ **DONE** | `tests/unit/test_odds_parsing.py` | - | 17 passing tests, 0.07s runtime |
| pytest configuration | ✅ **DONE** | `pytest.ini` | - | Coverage, markers, paths |
| Dev requirements | ✅ **DONE** | `requirements-dev.txt` | - | pytest, black, ruff, mypy, responses |
| Pre-commit hooks | ✅ **DONE** | `.pre-commit-config.yaml` | - | black, ruff, mypy, detect-secrets |
| GitHub Actions CI | ✅ **DONE** | `.github/workflows/test.yml` | - | Python 3.11-3.13 matrix, integration tests |
| Pre-commit CI | ✅ **DONE** | `.github/workflows/pre-commit.yml` | - | Auto-format on PR |
| Nightly data quality | ✅ **DONE** | `.github/workflows/nightly-data-quality.yml` | - | ETL validation |
| Integration tests | ⚠️ **PARTIAL** | `tests/integration/` scaffold | - | Directory exists, no tests yet |

**Key Wins**:
- Testing infrastructure created from scratch in this audit session
- CI/CD operational with 3 GitHub Actions workflows
- Pre-commit hooks enforce code quality automatically

**Gaps**:
- Coverage at 6% (only odds parsing tested; need model/feature/DB tests)
- Integration tests scaffolded but not implemented
- Target 40%+ coverage before dissertation submission

---

### ✅ **Documentation** (80% complete)

| Document | Status | Lines | Notes |
|----------|--------|-------|-------|
| README.md | ✅ **DONE** | - | Architecture, setup, testing, deployment |
| AGENTS.md | ✅ **DONE** | - | Repository guidelines for AI agents |
| TESTING.md | ✅ **DONE** | - | Testing strategy, CI/CD, pre-commit |
| TODO.tex | ✅ **UPDATED** | 300+ | **Just refreshed with audit findings** |
| Quarto notebooks | ✅ **DONE** | 15+ | Living documentation for workflows |
| Python docstrings | ⚠️ **PARTIAL** | - | Inconsistent; needs Sphinx autodoc |

**Key Wins**:
- README is comprehensive with quickstart, architecture diagrams (ASCII), testing instructions
- AGENTS.md provides clear onboarding for future AI collaborators
- 15+ Quarto notebooks serve as executable research documentation

**Gaps**:
- API documentation (Sphinx) not set up yet
- Some modules lack docstrings (e.g., monte_carlo.py)

---

## Detailed Module Inventory

### `/py/backtest/`
- **`baseline_glm.py`** (410 lines): Logistic regression with walk-forward validation, `CalibratedClassifierCV` (Platt/isotonic), Brier/LogLoss/ROI metrics, TeX table output to `analysis/results/` and `figures/out/`. **Production-ready**.
- **`harness.py`**: Multi-model comparison harness with configurable thresholds. Outputs to `analysis/results/glm_harness_*.csv`. **Operational**.

### `/py/models/`
- **`score_distributions.py`** (253 lines): Skellam PMF with Bessel approximation, `reweight_key_masses()` for 3/6/7/10 point margins, `reweight_with_moments()` using iterative proportional fitting. **Production-ready**.
- **`copulas.py`** (111 lines): Gaussian copula with `_Phi_inv()` Acklam approximation for spread/total dependence. **Production-ready**.

### `/py/features/`
- **`asof_features.py`** (412 lines): SQL-based rolling aggregation with leakage-safe cutoffs, team-game grain, writes to `mart.asof_team_features`. **Production-ready**.

### `/py/rl/`
- **`dataset.py`**: Offline RL dataset builder (`fetch_games()`, `build_logged_dataset()`). **Complete**.
- **`ope.py`**: SNIS/DR estimators with clipping/shrinkage, outcome model fitting. **Complete**.
- **`ope_gate.py`**: Grid evaluation over `c∈{5,10,20}`, `λ∈{0,0.1,0.2}`, ESS checks, JSON+TeX output. **Complete**.

### `/py/risk/`
- **`cvar_lp.py`**: CVaR optimization via LP, `cvar_of_stakes()` and `heuristic_stakes()`. **Complete**.
- **`generate_scenarios.py`**: Monte Carlo scenario generator with seeded RNG. **Complete**.
- **`cvar_report.py`**: TeX report emitter for multi-alpha CVaR benchmarks. **Complete**.

### `/py/pricing/`
- **`teaser.py`**: `teaser_ev()` with correlation, `middle_breakeven()` threshold checks. **Complete**.

### `/py/sim/`
- **`acceptance.py`**: EMD, key-mass chi-squared, friction checks; JSON+TeX output. **Complete**.
- **`execution.py`**: `ExecutionEngine` class, `simulate_order()` method. **Complete**.

### `/py/ops/`
- **`report_failure_analysis.py`**: Zero-week failure analysis. **Exists**.

### `/py/registry/`
- **`oos_to_tex.py`**: Out-of-sample results to TeX converter. **Exists**.

### `/py/` (root)
- **`monte_carlo.py`**: Basic Monte Carlo with `GradientBoostingRegressor`, Gaussian approximation. Could enhance with Skellam swap. **Functional but basic**.
- **`ingest_odds_history.py`**: TheOddsAPI ingestion with rate limiting, writes to `odds_history`. **Operational** (820K rows, nightly ETL).
- **`weather_meteostat.py`**: Weather ingestion scaffold. **Code exists, table empty**.

### `/data/` (R scripts)
- **`ingest_pbp.R`** (39 lines): nflfastR loader, truncate-and-refill for 1999-2024. **Operational** (1.23M plays).
- **`ingest_schedules.R`**: nflverse schedule/odds ingestion, 6,991 games. **Operational**.
- **`features_epa.R`** (49 lines): EPA aggregation to `mart.team_epa`. **Operational**.

### `/notebooks/` (Quarto)
15+ notebooks covering:
- `01_ingest_schedules.qmd` – Data ingestion workflow
- `02_pbp_features.qmd` – Play-by-play feature engineering
- `03_odds_weather_join.qmd` – Odds/weather integration
- `04_score_validation.qmd` – Key-number validation
- `05_copula_gof.qmd` – Copula goodness-of-fit
- `10_model_spread_xgb.qmd` – XGBoost spread model
- `11_monte_carlo_skellam.qmd` – Skellam Monte Carlo
- `12_risk_sizing.qmd` – CVaR risk sizing
- `80_rl_ablation.qmd` – RL vs stateless baseline
- `90_simulator_acceptance.qmd` – Simulator validation
- `00_timeframe_ablation.qmd` – Era weighting experiments

---

## Priority Recommendations

### **Immediate (P0)**
1. **Implement DQN training loop** (`py/rl/dqn_agent.py`):
   - State: (model probs, market odds, features)
   - Actions: {no-bet, bet-small, bet-medium, bet-large} or continuous stakes
   - Reward: PnL or CLV-shaped
   - Use offline dataset from `dataset.py`
   - Target networks, experience replay
   - MPS (Mac) and CUDA configs

2. **State-space ratings models** (`py/models/state_space.py`):
   - Glickman-Stern Kalman filter or Stan implementation
   - Weekly team strength posteriors
   - Integrate into backtest harness for fair comparison

3. **Expand test coverage to 40%+**:
   - Unit tests for `baseline_glm.py`, `score_distributions.py`, `copulas.py`
   - Integration tests for DB ingestion idempotency
   - Feature engineering tests (as-of correctness)

### **Short-term (P1)**
4. **Weather ingestion**: Run `weather_meteostat.py` to populate `weather` table, validate stadium roof/surface logic.

5. **Injury data integration**: Identify data source (nflverse `load_injuries()`?), build ingestion pipeline, create QB-out binary and AGL index features.

6. **PPO agent** (if time permits): Actor-critic for richer action spaces (alt-lines, teasers).

### **Documentation (P1)**
7. **API documentation**: Set up Sphinx autodoc, generate HTML docs for Python modules.

8. **Pipeline DAG diagram**: Visualize data lineage (TikZ flowchart in dissertation or `mermaid` in README).

### **Deferred (P2)**
- Referee crew features (low priority unless marginal lift is high)
- Market microstructure features (hold, line velocity) – research question, not critical path
- Multi-book arbitrage scan enhancements

---

## Database State

| Table | Row Count | Notes |
|-------|-----------|-------|
| `games` | 6,991 | 1999-2024 schedules with closing lines |
| `plays` | 1,230,857 | nflfastR play-by-play with EPA |
| `odds_history` | 820,080 | TheOddsAPI snapshots (spreads, totals, moneylines) |
| `weather` | 0 | Schema exists, needs ingestion |
| `injuries` | 0 | Schema exists, needs data source |
| `mart.team_epa` | - | Materialized view, auto-refreshed |
| `mart.game_summary` | - | Materialized view, auto-refreshed |

**Database**: PostgreSQL 16 + TimescaleDB 2.22.0, port 5544, 485 MB, hypertables on `odds_history`.

---

## Outputs & Artifacts

### CSV Results (`analysis/results/`)
- `glm_baseline_metrics.csv` – Walk-forward validation metrics
- `glm_baseline_preds.csv` – Predictions with probabilities
- `glm_harness_metrics*.csv` – Multi-model harness outputs
- `glm_calibration_platt.csv` – Platt calibration curves

### TeX Tables (`analysis/dissertation/figures/out/`)
54+ auto-generated tables including:
- `glm_baseline_table.tex` – Baseline GLM metrics by season
- `glm_harness_table_small.tex` – Harness comparison table
- `ope_grid_table.tex` – OPE grid results (c × λ)
- `cvar_benchmark_table.tex` – CVaR α=0.90/0.95 comparison
- `sim_acceptance_table.tex` – Simulator validation (EMD, key masses)
- `reweighting_ablation_table.tex` – Key-number reweighting ablation
- `rl_vs_baseline_table.tex` – RL ablation results
- `teaser_ev_oos_table.tex` – Teaser out-of-sample EV

---

## Conclusion

You've built a **substantial research infrastructure** that is far more complete than TODO.tex suggested. The baseline modeling stack is **production-ready**, risk management is **complete**, and simulation/pricing tools are **operational**. The testing infrastructure created in this session provides a foundation for quality assurance.

**The dissertation is now blocked on**:
1. DQN/PPO implementation (highest ROI for novelty)
2. State-space models (fills classical benchmark gap)
3. Weather/injury data integration (marginal but expected in modern NFL analytics)

**Estimated effort**:
- DQN: 3-5 days (agent class, training loop, MPS/CUDA configs, validation)
- State-space: 2-3 days (Kalman filter, Stan alternative, harness integration)
- Weather: 1 day (run existing code, validate joins)
- Injury: 2-3 days (identify source, build pipeline, feature engineering)
- Test expansion: Ongoing, target 2-3 tests per sprint

**Next session**: Start with DQN skeleton (`py/rl/dqn_agent.py`) using the offline dataset. This unblocks the RL dissertation chapter.

---

**Audit complete.** TODO.tex updated with accurate completion status.
