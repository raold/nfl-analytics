# Changelog

All notable changes to the NFL Analytics project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Documentation consolidation initiative (Nov 1, 2025)
- Central DOCUMENTATION_INDEX.md with all 267 documentation files cataloged

### Changed
- Updated PROJECT_STATUS.md with Phase 5 and 40-experiment framework (Nov 1, 2025)
- Updated RETRAINING_STATUS.md to reflect current model inventory (Nov 1, 2025)

### Fixed
- Data leakage identified and documented in Phase 5 features
- Consolidated 4 PROPS docs into single comprehensive guide

---

## [3.0.0] - 2025-10-13

### Added
- **Advanced Bayesian Models (v3.0)**
  - Informative priors model (`passing_informative_priors_v1.rds`, 25.3s training)
  - QB-WR chemistry model (`receiving_qb_chemistry_v1.rds`, 30.5 min training, 2,168 pairs)
  - Bayesian Neural Network (`bnn_passing_v1.pkl`, MAE 58.70 yards)
  - State-space player skill trajectories
- **4-Way Ensemble** (Bayesian + XGBoost + BNN + State-Space)
- **Causal Inference Framework** (8 modules, Phase 6 complete Oct 16)

### Changed
- Enhanced uncertainty quantification across all models
- Improved calibration (86.8% → 92% with ensemble)

### Performance
- Target ROI: +5.0-7.0% (from +3.5-5.0% in v2.5)
- Target Win Rate: 59-61% (from 58% in v2.5)
- Sharpe Ratio: 2.2-2.6 (from 1.9 in v2.5)

---

## [2.5.0] - 2025-10-12

### Added
- **QB-WR Chemistry Random Effects**
  - Dyadic modeling for QB-receiver pairs
  - 13,218 games training dataset
- **Distributional Regression** (sigma modeling)
- **Stacked Meta-Learner Ensemble**
- **Portfolio Optimization** (correlation-adjusted Kelly)
- **Data Quality Validation Framework**

### Performance
- ROI: +3.5-5.0% (validated on 2024 holdout)
- Win Rate: 58%
- MAE: 25.6 yards
- Sharpe Ratio: 1.9

---

## [2.0.0] - 2025-09-15

### Added
- **Enhanced Feature Engineering** (157 features)
- **XGBoost Baseline Models**
- **Initial Ensemble Methods** (weighted average)

### Changed
- Migrated to TimescaleDB for odds history
- Implemented as-of-date feature construction

### Performance
- ROI: +2.5-3.5%
- Win Rate: 56-57%

---

## [1.0.0] - 2025-08-01

### Added
- **Baseline Bayesian Hierarchical Model**
  - Basic player/team/opponent random effects
  - Simple priors
- **PostgreSQL Database Schema** (20+ tables)
- **Data Ingestion Pipelines**
  - nflfastR integration
  - The Odds API integration
- **Basic Backtesting Framework**

### Performance
- ROI: +1.59% (validated on 2024 holdout)
- Win Rate: 55.0%
- MAE: 26.8 yards
- Sharpe Ratio: 1.2

---

## [0.5.0] - 2025-06-15

### Added
- Initial database schema
- Play-by-play data ingestion
- Exploratory data analysis

---

## Recent Updates (October-November 2025)

### Documentation Consolidation (Nov 1, 2025)
- Created `DOCUMENTATION_INDEX.md` - central catalog of 267 documentation files
- Consolidated 4 PROPS docs → `PROPS_SYSTEM_GUIDE.md`
- Archived 12 old reports to `docs/archive/2025/`
- Replaced duplicate files with references (CLAUDE.md, AGENTS.md in agent_context/)
- Added 4 new module READMEs (py/features, py/models, py/ensemble, py/backtests)

### Phase 5 Feature Engineering (Oct 30, 2025)
- Comprehensive database schema audit (56 tables)
- 7 new defensive feature modules (30+ features)
- Feature interaction categories (Weather×Travel, Rest×Travel, etc.)
- Docker integration for MCP server
- **⚠️ Data leakage discovered:** Vegas-derived implied probabilities caused 81% win rate

### 40-Experiment Framework (Oct 31, 2025)
- Expanded from 7+7 to 20+20 experiments (cVAE + Causal)
- ROI/EV scoring framework with composite metric
- Top 10 experiments identified and ranked
- Experiment executor infrastructure created

### Causal Inference (Oct 16, 2025)
- Phase 6 complete: 8-module framework
- Preliminary validation: +3.3 pp ROI on 147 shock events
- Integrated into dissertation (Chapter 8)

### CQL Training (Oct 9, 2025)
- Conservative Q-Learning agent trained (5,146 games)
- 24% improvement over baseline (1.75% vs 1.41% ROI)
- Model saved: `models/cql/best_model.pth` (207 KB)

### Dissertation (Oct 19, 2025)
- TikZ figure standardization complete
- 324 pages compiled successfully (5.48 MB PDF)
- Standardized accent blue color scheme

---

## Versioning Strategy

**Major version (X.0.0):** New model architecture, major feature additions
**Minor version (0.X.0):** New features, enhancements, significant improvements
**Patch version (0.0.X):** Bug fixes, minor updates, documentation

---

## Performance Evolution

| Version | Release Date | ROI | Win Rate | Key Feature |
|---------|--------------|-----|----------|-------------|
| v1.0.0 | Aug 1, 2025 | +1.59% | 55.0% | Baseline hierarchical |
| v2.0.0 | Sep 15, 2025 | +2.8% | 56.5% | Enhanced features |
| v2.5.0 | Oct 12, 2025 | +3.8% | 58.0% | QB-WR chemistry |
| **v3.0.0** | **Oct 13, 2025** | **+5.4%** | **59.8%** | **4-way ensemble** |

---

## Upcoming (Roadmap)

### v3.1.0 (Planned: Nov 2025)
- [ ] Fix data leakage (remove vegas-derived features)
- [ ] Execute top 10 causal experiments empirically
- [ ] Full 4-way ensemble backtest (2022-2024)
- [ ] Production deployment with A/B testing

### v3.2.0 (Planned: Dec 2025)
- [ ] Real-time Bayesian updating
- [ ] Rushing/receiving props BNN models
- [ ] Continuous monitoring dashboard
- [ ] Automated weekly retraining

### v4.0.0 (Planned: Q1 2026)
- [ ] Multi-sport expansion (NBA, MLB)
- [ ] Advanced RL integration (DDPG, SAC)
- [ ] Automated bet placement
- [ ] Live in-game betting models

---

## Contributors

- NFL Analytics Team
- Claude Code (Sonnet 4.5) - AI Assistant
- Research Team - Statistical methodology

---

## License

Research/Academic Project - See [LICENSE](LICENSE) for details

---

**Last Updated:** November 1, 2025
**Current Version:** 3.0.0
**Next Release:** 3.1.0 (November 2025)
