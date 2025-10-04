# NFL Analytics Documentation

Welcome to the NFL Analytics project documentation. This directory contains all project documentation organized by category.

---

## 📖 Documentation Index

### Setup Guides
- **[Database Setup](setup/database.md)** - PostgreSQL + TimescaleDB configuration
- **[Development Environment](setup/development.md)** - Local development setup
- **[Production Deployment](setup/production.md)** - Production environment configuration

### Architecture
- **[Data Pipeline Architecture](architecture/data_pipeline.md)** - ETL framework and data flow
- **[Database Schema](architecture/database_schema.md)** - Table definitions and relationships
- **[Feature Engineering](architecture/feature_engineering.md)** - Feature generation methodology

### Operations
- **[Deployment Guide](operations/deployment.md)** - Deploying updates
- **[Monitoring & Alerts](operations/monitoring.md)** - System monitoring and alerting
- **[Troubleshooting](operations/troubleshooting.md)** - Common issues and solutions

### Reports
- **[2025 Season Data Ingestion](reports/2025_season_data_ingestion.md)** - 2025 NFL season data loading
- **[Database Audit Report](reports/database_audit_report.md)** - Comprehensive database audit
- **[Database Production Certification](reports/database_production_cert.md)** - Production readiness certification
- **[Backfill Execution Summary](reports/database_backfill_execution_summary.md)** - Historical data backfill results
- **[Feature Engineering Complete](reports/feature_engineering_complete.md)** - Enhanced feature engineering results
- **[Database Gap Analysis](reports/database_gap_analysis_and_backfill_plan.md)** - Data gaps and remediation plan
- **[Backfill Results](reports/backfill_complete_results.md)** - Detailed backfill execution results
- **[Codebase Audit 2025](reports/codebase_audit_2025.md)** - Code quality and structure audit

### AI Agent Context
- **[Agent Guidelines](agent_context/AGENTS.md)** - Guidelines for AI agents working on this project
- **[Claude Context](agent_context/CLAUDE.md)** - Claude-specific context and patterns
- **[Gemini Context](agent_context/GEMINI.md)** - Gemini-specific context and patterns

---

## 🚀 Quick Start

1. **New to the project?** Start with [Development Environment Setup](setup/development.md)
2. **Setting up production?** Read [Production Deployment](setup/production.md)
3. **Understanding the data?** See [Database Schema](architecture/database_schema.md)
4. **Running ETL pipelines?** Check [Data Pipeline Architecture](architecture/data_pipeline.md)
5. **Having issues?** Consult [Troubleshooting](operations/troubleshooting.md)

---

## 📊 Project Overview

### Current State (October 2025)
- **Database**: 7,263 games (1999-2025), 1.24M plays, 60K roster entries
- **Data Sources**: nflverse, The Odds API, Meteostat weather
- **Coverage**: Complete historical data + live 2025 season (Week 5 complete)
- **Features**: 157-column enhanced feature set
- **Models**: Baseline GLM, ensemble methods, reinforcement learning
- **Status**: ✅ Production-ready database with automated ETL

### Key Metrics
- **Database Size**: ~700 MB
- **ETL Frequency**: Daily (6am EST) + Weekly full refresh (Monday 10am EST)
- **Data Quality Score**: 95%+ (validated)
- **Feature Coverage**: 2003-2025 seasons

---

## 🏗️ Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                      DATA SOURCES                            │
│  nflverse • The Odds API • Meteostat • Manual Entry         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    ETL FRAMEWORK                             │
│  Extract → Validate → Transform → Deduplicate → Load        │
│  ✓ Schema validation    ✓ Rate limiting                     │
│  ✓ Data quality checks  ✓ Retry logic                       │
│  ✓ Referential integrity ✓ Monitoring                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 PostgreSQL + TimescaleDB                     │
│  games • plays • players • rosters • odds • weather         │
│  mart.team_epa • mart.game_summary • mart.team_stats        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 FEATURE ENGINEERING                          │
│  157 columns: EPA, success rate, explosiveness, trends      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   ML MODELS                                  │
│  Baseline GLM • Ensemble • Reinforcement Learning           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              ANALYSIS & BETTING                              │
│  Predictions • Risk Sizing • Portfolio Optimization          │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Directory Structure

```
nfl-analytics/
├── docs/               # This directory - all documentation
├── etl/                # ETL framework (extract, transform, load, validate)
├── py/                 # Python package (features, models, analysis)
├── R/                  # R scripts (ingestion, features, reports)
├── db/                 # Database migrations and schema
├── data/               # Data storage (raw, processed, staging)
├── scripts/            # Operational scripts (dev, deploy, maintenance)
├── infrastructure/     # Docker, deployment configs
├── tests/              # Test suite
├── analysis/           # Analysis notebooks and dissertation
├── logs/               # Application logs
└── models/             # Trained model artifacts
```

---

## 🔧 Common Tasks

### Running Daily ETL
```bash
python etl/pipelines/daily.py --date $(date +%Y-%m-%d)
```

### Generating Features
```bash
python py/features/asof_features_enhanced.py \
  --output data/processed/features/enhanced_features.csv \
  --season-start 2003 \
  --validate
```

### Refreshing Materialized Views
```sql
REFRESH MATERIALIZED VIEW mart.game_summary;
REFRESH MATERIALIZED VIEW mart.team_season_stats;
```

### Checking Data Quality
```bash
python etl/validate/quality.py --date $(date +%Y-%m-%d)
```

### Database Backup
```bash
bash scripts/maintenance/backup_data.sh
```

---

## 📝 Contributing

When updating documentation:

1. **Keep it current** - Update docs when code changes
2. **Be specific** - Include examples and commands
3. **Link related docs** - Help readers find related information
4. **Test instructions** - Verify commands work before documenting

---

## 🆘 Getting Help

**Issues with setup?** → [Troubleshooting Guide](operations/troubleshooting.md)  
**Data quality concerns?** → [Monitoring & Alerts](operations/monitoring.md)  
**Architecture questions?** → [Data Pipeline Architecture](architecture/data_pipeline.md)  
**Can't find what you need?** → Check the [Reports](reports/) directory

---

## 📮 Contact & Maintenance

**Database**: PostgreSQL 15 + TimescaleDB on localhost:5544  
**Primary Data Source**: nflverse/nflreadr  
**Betting Data**: The Odds API  
**Weather Data**: Meteostat  
**Update Frequency**: Daily ETL at 6am EST, Weekly full refresh Mondays at 10am EST  

**Last Updated**: October 4, 2025
